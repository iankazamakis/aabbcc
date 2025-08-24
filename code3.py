from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from skyfield.api import EarthSatellite, Topos
import numpy as np

MAX_VISIBLE_DISTANCE_KM = 12000.0  # max line-of-sight distance


@dataclass
class AccessWindow:
    """Represents a visibility window between two satellites."""
    start: datetime
    end: datetime


@dataclass
class Access:
    """Represents an access, which can be sat-to-sat or ground-to-sat."""
    sat: EarthSatellite
    observer: Topos | EarthSatellite
    start: datetime
    end: datetime
    uuid: Optional[UUID] = None  # optional, defaults to None


class SatelliteAccessComputer:
    """
    Efficient satellite access computation with vectorized caching.
    - Incrementally caches satellites as they arrive.
    - Shared coarse time grid.
    - Visibility transitions refined to sub-second accuracy.
    """

    def __init__(self, ts, step_seconds: int = 60):
        self.ts = ts
        self.step_seconds = step_seconds
        self._time_grid = None
        # Cache: (satnum, start_tt, end_tt) -> {"times": tt_array, "positions": (3, N) array}
        self._position_cache = {}

    def _ensure_time_grid(self, start: datetime, end: datetime):
        if self._time_grid is None:
            total_seconds = int((end - start).total_seconds())
            self._time_grid = self.ts.utc(
                start.year, start.month, start.day,
                start.hour, start.minute,
                np.arange(0, total_seconds, self.step_seconds)
            )
        return self._time_grid

    def _ensure_cached(self, sat: EarthSatellite, start: datetime, end: datetime):
        times = self._ensure_time_grid(start, end)
        key = (sat.model.satnum, times[0].tt, times[-1].tt)

        if key in self._position_cache:
            return

        geocentric = sat.at(times)
        pos = geocentric.position.km  # shape (3, N)
        self._position_cache[key] = {"times": times.tt, "positions": pos}

    def _get_position(self, sat: EarthSatellite, t):
        for (satnum, t0, t1), data in self._position_cache.items():
            if satnum == sat.model.satnum and t0 <= t.tt <= t1:
                step_days = self.step_seconds / 86400.0
                idx = int(round((t.tt - t0) / step_days))
                return data["positions"][:, idx]

        raise ValueError("Time not in cache. Call _ensure_cached first.")

    def _distance_ok(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) < MAX_VISIBLE_DISTANCE_KM

    def _is_visible(self, sat1, sat2, t):
        p1 = self._get_position(sat1, t)
        p2 = self._get_position(sat2, t)

        if not self._distance_ok(p1, p2):
            return False
        return not sat1.is_behind_earth(p2 - p1)

    def _refine_boundary(self, sat1, sat2, t1, t2, val1, val2, tol_seconds=0.1):
        while (t2 - t1).seconds > tol_seconds:
            mid = self.ts.tt((t1.tt + t2.tt) / 2)
            mid_val = self._is_visible(sat1, sat2, mid)
            if mid_val == val1:
                t1, val1 = mid, mid_val
            else:
                t2, val2 = mid, mid_val
        return t2 if val2 else t1

    def compute_access_windows(self, sat1: EarthSatellite, sat2: EarthSatellite,
                               start: datetime, end: datetime) -> List[AccessWindow]:
        self._ensure_cached(sat1, start, end)
        self._ensure_cached(sat2, start, end)

        times = self._time_grid
        windows = []
        prev_t = times[0]
        prev_val = self._is_visible(sat1, sat2, prev_t)
        current_window_start = None

        for t in times[1:]:
            val = self._is_visible(sat1, sat2, t)
            if val != prev_val:
                boundary = self._refine_boundary(sat1, sat2, prev_t, t, prev_val, val)
                if val:
                    current_window_start = boundary.utc_datetime()
                else:
                    if current_window_start:
                        windows.append(AccessWindow(
                            start=current_window_start,
                            end=boundary.utc_datetime()
                        ))
                        current_window_start = None
            prev_t, prev_val = t, val

        if prev_val and current_window_start:
            windows.append(AccessWindow(
                start=current_window_start,
                end=end
            ))

        return windows


def generate_sat_to_sat_accesses(computer: SatelliteAccessComputer,
                                 sat1: EarthSatellite,
                                 sat2: EarthSatellite,
                                 start: datetime,
                                 end: datetime) -> List[Access]:
    """
    Generate Access objects for sat-to-sat windows.
    """
    windows = computer.compute_access_windows(sat1, sat2, start, end)
    accesses: List[Access] = []

    for w in windows:
        accesses.append(
            Access(
                sat=sat1,
                observer=sat2,
                start=w.start,
                end=w.end,
                uuid=uuid4()
            )
        )

    return accesses

