from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time
from skyfield.searchlib import find_discrete
import numpy as np

MAX_VISIBLE_DISTANCE_KM = 12000.0


@dataclass
class AccessWindow:
    start: datetime
    end: datetime


class SatelliteCache:
    """Cache of propagated positions for satellites on a common set of timestamps."""

    def __init__(self, ts: List[Time]):
        self.ts = ts
        self._cache: Dict[int, any] = {}

    def propagate(self, sat: EarthSatellite, sat_id: int):
        if sat_id not in self._cache:
            self._cache[sat_id] = sat.at(self.ts)
        return self._cache[sat_id]

    def get_position(self, sat_id: int, idx: int):
        return self._cache[sat_id][idx]


class AccessComputer:
    def __init__(self, ts: List[Time], cache: SatelliteCache):
        self.ts = ts
        self.cache = cache

    def _visible(self, observer_id: int, target_id: int, i: int) -> bool:
        obs_geo = self.cache.get_position(observer_id, i)
        tgt_geo = self.cache.get_position(target_id, i)

        # Distance check
        dist = np.linalg.norm(obs_geo.position.km - tgt_geo.position.km)
        if dist > MAX_VISIBLE_DISTANCE_KM:
            return False

        # Earth occultation check
        if tgt_geo.is_behind_earth(obs_geo):
            return False

        return True

    def find_access_windows(
        self, observer: EarthSatellite, target: EarthSatellite,
        observer_id: int, target_id: int
    ) -> List[AccessWindow]:
        """Compute access windows between two satellites."""
        self.cache.propagate(observer, observer_id)
        self.cache.propagate(target, target_id)

        def visibility_func(t: Time) -> bool:
            idx = np.searchsorted(self.ts.tt, t.tt)
            if idx >= len(self.ts):
                idx = len(self.ts) - 1
            return self._visible(observer_id, target_id, idx)

        t_events, values = find_discrete(self.ts[0], self.ts[-1], visibility_func)

        windows: List[AccessWindow] = []
        for i in range(0, len(t_events) - 1, 2):
            if values[i]:  # True → start of window
                windows.append(
                    AccessWindow(start=t_events[i].utc_datetime(),
                                 end=t_events[i+1].utc_datetime())
                )
        return windows

