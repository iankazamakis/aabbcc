from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time
from skyfield.searchlib import find_discrete

MAX_VISIBLE_DISTANCE_KM = 12000.0


@dataclass
class AccessWindow:
    start: datetime
    end: datetime


class EphemerisCache:
    """Cache satellite positions at given times to avoid recomputation."""
    def __init__(self, sats: List[EarthSatellite], times: Time):
        self.times = times
        self._cache: Dict[int, object] = {}
        for sat in sats:
            self._cache[sat.model.satnum] = sat.at(times)

    def at(self, sat: EarthSatellite, t: Time):
        # Skyfield's .at() can interpolate between precomputed times
        return sat.at(t)


def find_access_windows(ts, start: datetime, end: datetime,
                        obs: EarthSatellite, tgt: EarthSatellite,
                        step_seconds: int = 60,
                        cache: EphemerisCache = None) -> List[AccessWindow]:
    """Find refined access windows between two satellites."""

    t0 = ts.utc(start)
    t1 = ts.utc(end)

    def visible(t: Time):
        obs_geo = (cache.at(obs, t) if cache else obs.at(t))
        tgt_geo = (cache.at(tgt, t) if cache else tgt.at(t))

        obs_pos = np.asarray(obs_geo.position.km)
        tgt_pos = np.asarray(tgt_geo.position.km)

        dist = np.linalg.norm(obs_pos - tgt_pos, axis=0)
        close = dist <= MAX_VISIBLE_DISTANCE_KM
        not_occulted = ~tgt_geo.is_behind_earth(obs_geo)
        return close & not_occulted

    # silence type checker: functions can have arbitrary attrs
    setattr(visible, "step_days", step_seconds / 86400.0)

    t_events, values = find_discrete(t0, t1, visible)

    windows: List[AccessWindow] = []
    state = bool(visible(t0))
    current_start = start if state else None

    for te, val in zip(t_events, values):
        if not state and val:
            current_start = te.utc_datetime()
        elif state and not val:
            windows.append(AccessWindow(current_start, te.utc_datetime()))
            current_start = None
        state = bool(val)

    if state and current_start is not None:
        windows.append(AccessWindow(current_start, end))

    return windows

