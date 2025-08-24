from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np

MAX_VISIBLE_DISTANCE_KM = 12000.0


@dataclass
class AccessWindow:
    start: datetime
    end: datetime


class SatelliteAccessComputer:
    def __init__(self, satellites, ts):
        self.satellites = satellites
        self.ts = ts
        self._position_cache = {}

    def _cache_positions(self, times):
        """Pre-propagate all satellite positions at once and store them."""
        for sat in self.satellites:
            geocentric = sat.at(times)
            pos = geocentric.position.km.T  # shape = (len(times), 3)
            for i, t in enumerate(times.tt):
                self._position_cache[(sat.model.satnum, t)] = pos[i]

    def _get_position(self, sat: EarthSatellite, t) -> np.ndarray:
        return self._position_cache[(sat.model.satnum, t.tt)]

    def _distance_ok(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) < MAX_VISIBLE_DISTANCE_KM

    def _is_visible(self, sat1: EarthSatellite, sat2: EarthSatellite, t) -> bool:
        p1 = self._get_position(sat1, t)
        p2 = self._get_position(sat2, t)
        if not self._distance_ok(p1, p2):
            return False
        return not sat1.is_behind_earth(p2 - p1)

    def _refine_boundary(self, sat1, sat2, t1, t2, val1, val2, tol_seconds=0.1):
        """Binary search to refine the transition point to sub-second accuracy."""
        while (t2 - t1).seconds > tol_seconds:
            mid = self.ts.tt((t1.tt + t2.tt) / 2)
            mid_val = self._is_visible(sat1, sat2, mid)
            if mid_val == val1:
                t1 = mid
                val1 = mid_val
            else:
                t2 = mid
                val2 = mid_val
        return t2 if val2 else t1

    def compute_access_windows(
        self,
        sat1: EarthSatellite,
        sat2: EarthSatellite,
        start: datetime,
        end: datetime,
        step_seconds: int = 60,
    ) -> List[AccessWindow]:

        # Build coarse time grid
        total_seconds = int((end - start).total_seconds())
        times = self.ts.utc(
            start.year, start.month, start.day,
            start.hour, start.minute, np.arange(0, total_seconds, step_seconds)
        )

        self._cache_positions(times)

        windows = []
        prev_t = times[0]
        prev_val = self._is_visible(sat1, sat2, prev_t)
        current_window_start = None

        for t in times[1:]:
            val = self._is_visible(sat1, sat2, t)

            if val != prev_val:  # visibility changed
                boundary = self._refine_boundary(sat1, sat2, prev_t, t, prev_val, val)

                if val:  # became visible
                    current_window_start = boundary.utc_datetime()
                else:  # became invisible
                    if current_window_start:
                        windows.append(AccessWindow(
                            start=current_window_start,
                            end=boundary.utc_datetime()
                        ))
                        current_window_start = None

            prev_t, prev_val = t, val

        # If still visible at the end, close the last window
        if prev_val and current_window_start:
            windows.append(AccessWindow(
                start=current_window_start,
                end=end
            ))

        return windows

