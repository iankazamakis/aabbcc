from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

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
    # Build arrays for utc() (robust across Skyfield versions)
    years   = [dt.year for dt in dts]
    months  = [dt.month for dt in dts]
    days    = [dt.day for dt in dts]
    hours   = [dt.hour for dt in dts]
    minutes = [dt.minute for dt in dts]
    seconds = [dt.second + dt.microsecond / 1e6 for dt in dts]
    return tscale.utc(years, months, days, hours, minutes, seconds)


class SatelliteCache:
    """Cache Geocentric positions for satellites on a shared Time grid."""

    def __init__(self, t_grid: Time):
        self.t_grid = t_grid            # Skyfield Time (vector)
        self._geo: Dict[int, object] = {}  # sat_id -> Geocentric (vector)

    def propagate_once(self, sat: EarthSatellite, sat_id: int):
        if sat_id not in self._geo:
            self._geo[sat_id] = sat.at(self.t_grid)
        return self._geo[sat_id]

    def geo_slice(self, sat_id: int, idx: Union[int, np.ndarray]):
        # Skyfield Geocentric supports NumPy-style indexing/slicing
        return self._geo[sat_id][idx]


class AccessComputer:
    def __init__(self, t_grid: Time, cache: SatelliteCache):
        self.t_grid = t_grid
        self.cache = cache
        tt = self.t_grid.tt
        self._tt = np.asarray(tt)  # 1D array of TT days
        if len(self._tt) < 2:
            raise ValueError("Time grid must have at least two samples.")
        self._step_days = float(self._tt[1] - self._tt[0])

    def _visible_idx(self, observer_id: int, target_id: int, idx: Union[int, np.ndarray]):
        """Visibility at preseeded index/indices."""
        obs_geo = self.cache.geo_slice(observer_id, idx)
        tgt_geo = self.cache.geo_slice(target_id, idx)

        # Distance check (works for scalar or vector)
        d = obs_geo.position.km - tgt_geo.position.km           # shape: (3,) or (3, N)
        dist = np.sqrt((d * d).sum(axis=0))                     # scalar or (N,)

        within = dist <= MAX_VISIBLE_DISTANCE_KM
        occult = tgt_geo.is_behind_earth(obs_geo)               # bool or array
        return np.logical_and(within, np.logical_not(occult))   # bool or array

    def _nearest_grid_index(self, t: Time) -> np.ndarray:
        """Map arbitrary Time to nearest preseeded index (no repropagation)."""
        tt = np.atleast_1d(np.asarray(t.tt))
        # right index
        r = np.searchsorted(self._tt, tt, side='left')
        r = np.clip(r, 1, len(self._tt) - 1)
        l = r - 1
        # choose nearer of l or r
        pick_left = (tt - self._tt[l]) <= (self._tt[r] - tt)
        return np.where(pick_left, l, r)

    def find_access_windows(
        self,
        observer: EarthSatellite, target: EarthSatellite,
        observer_id: int, target_id: int
    ) -> List[AccessWindow]:
        # Ensure both satellites are propagated once on the shared grid
        self.cache.propagate_once(observer, observer_id)
        self.cache.propagate_once(target, target_id)

        def visibility_func(t: Time):
            idx = self._nearest_grid_index(t)
            return self._visible_idx(observer_id, target_id, idx)

        # Tell Skyfield how coarsely to step during its search
        visibility_func.step_days = self._step_days

        t0 = self.t_grid[0]
        t1 = self.t_grid[-1]

        t_events, values = find_discrete(t0, t1, visibility_func)

        # Build windows from (t_events, values)
        windows: List[AccessWindow] = []
        if len(t_events) == 0:
            # constant state over [t0, t1]
            if bool(self._visible_idx(observer_id, target_id, 0)):
                windows.append(AccessWindow(t0.utc_datetime(), t1.utc_datetime()))
            return windows

        # We need the state just BEFORE the first event to know if a window starts at t0
        initial_state = bool(self._visible_idx(observer_id, target_id, 0))
        state = initial_state
        current_start: Optional[datetime] = t0.utc_datetime() if state else None

        for te, val in zip(t_events, values):
            # After each event, the state equals 'val'
            if not state and val:            # False -> True: window opens
                current_start = te.utc_datetime()
            elif state and not val:          # True -> False: window closes
                windows.append(AccessWindow(current_start, te.utc_datetime()))
                current_start = None
            state = bool(val)

        # If visibility remains True through the end, close the final window at t1
        if state and current_start is not None:
            windows.append(AccessWindow(current_start, t1.utc_datetime()))

        return windows

