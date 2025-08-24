from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time
from skyfield.searchlib import find_discrete

MAX_VISIBLE_DISTANCE_KM = 12000.0


@dataclass
class AccessWindow:
    start: datetime
    end: datetime


def build_time_grid(tscale, start: datetime, end: datetime, step_seconds: int) -> Time:
    """Return a Skyfield Time vector with uniform spacing."""
    total = int(np.floor((end - start).total_seconds() / step_seconds)) + 1
    dts = [start + timedelta(seconds=i * step_seconds) for i in range(total)]
    return tscale.utc([dt.year for dt in dts],
                      [dt.month for dt in dts],
                      [dt.day for dt in dts],
                      [dt.hour for dt in dts],
                      [dt.minute for dt in dts],
                      [dt.second + dt.microsecond/1e6 for dt in dts])


class SatelliteCache:
    """Cache Geocentric positions for satellites on a shared Time grid."""

    def __init__(self, t_grid: Time):
        self.t_grid = t_grid
        self._geo: Dict[int, object] = {}  # satnum -> Geocentric (vector)

    def propagate_once(self, sat: EarthSatellite):
        sat_id = sat.model.satnum
        if sat_id not in self._geo:
            self._geo[sat_id] = sat.at(self.t_grid)
        return self._geo[sat_id]

    def geocentric(self, sat: EarthSatellite):
        return self._geo[sat.model.satnum]


class AccessComputer:
    def __init__(self, t_grid: Time, cache: SatelliteCache):
        self.t_grid = t_grid
        self.cache = cache
        self._tt = np.asarray(t_grid.tt)
        if len(self._tt) < 2:
            raise ValueError("Time grid must have at least two samples.")
        self._step_days = float(self._tt[1] - self._tt[0])

    def _nearest_grid_index(self, t: Time) -> np.ndarray:
        """Map arbitrary Time to nearest preseeded index."""
        tt = np.atleast_1d(np.asarray(t.tt))
        r = np.searchsorted(self._tt, tt, side='left')
        r = np.clip(r, 1, len(self._tt) - 1)
        l = r - 1
        pick_left = (tt - self._tt[l]) <= (self._tt[r] - tt)
        return np.where(pick_left, l, r)

    def _visible_idx(self, obs: EarthSatellite, tgt: EarthSatellite, idx):
        obs_geo = self.cache.geocentric(obs)[idx]
        tgt_geo = self.cache.geocentric(tgt)[idx]

        # Distance
        d = obs_geo.position.km - tgt_geo.position.km
        dist = np.sqrt((d * d).sum(axis=0))
        if np.isscalar(dist):
            if dist > MAX_VISIBLE_DISTANCE_KM:
                return False
        else:
            dist_ok = dist <= MAX_VISIBLE_DISTANCE_KM

        # Earth occultation
        occult = tgt_geo.is_behind_earth(obs_geo)

        return np.logical_and(dist_ok if not np.isscalar(dist) else dist <= MAX_VISIBLE_DISTANCE_KM,
                              np.logical_not(occult))

    def find_access_windows(self, obs: EarthSatellite, tgt: EarthSatellite) -> List[AccessWindow]:
        """Compute access windows between two satellites."""
        self.cache.propagate_once(obs)
        self.cache.propagate_once(tgt)

        def visibility_func(t: Time):
            idx = self._nearest_grid_index(t)
            return self._visible_idx(obs, tgt, idx)

        visibility_func.step_days = self._step_days

        t0, t1 = self.t_grid[0], self.t_grid[-1]
        t_events, values = find_discrete(t0, t1, visibility_func)

        windows: List[AccessWindow] = []
        if len(t_events) == 0:
            if bool(self._visible_idx(obs, tgt, 0)):
                windows.append(AccessWindow(t0.utc_datetime(), t1.utc_datetime()))
            return windows

        # Need initial state to know if a window started before first event
        state = bool(self._visible_idx(obs, tgt, 0))
        current_start: Optional[datetime] = t0.utc_datetime() if state else None

        for te, val in zip(t_events, values):
            if not state and val:
                current_start = te.utc_datetime()
            elif state and not val:
                windows.append(AccessWindow(current_start, te.utc_datetime()))
                current_start = None
            state = bool(val)

        if state and current_start is not None:
            windows.append(AccessWindow(current_start, t1.utc_datetime()))

        return windows

