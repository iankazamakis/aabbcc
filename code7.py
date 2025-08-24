from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
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


def find_access_windows(ts, start: datetime, end: datetime,
                        obs: EarthSatellite, tgt: EarthSatellite,
                        step_seconds: int = 60) -> List[AccessWindow]:
    """Find refined access windows between two satellites using Skyfield find_discrete()."""

    t0 = ts.utc(start)
    t1 = ts.utc(end)

    def visible(t: Time):
        obs_geo = obs.at(t)
        tgt_geo = tgt.at(t)

        dist = np.linalg.norm(obs_geo.position.km - tgt_geo.position.km, axis=0)
        close = dist <= MAX_VISIBLE_DISTANCE_KM
        not_occulted = ~tgt_geo.is_behind_earth(obs_geo)
        return close & not_occulted

    # Tell Skyfield how coarse to step initially
    visible.step_days = step_seconds / 86400.0

    t_events, values = find_discrete(t0, t1, visible)

    windows: List[AccessWindow] = []
    if len(t_events) == 0:
        # Constant state
        if visible(t0):
            windows.append(AccessWindow(start, end))
        return windows

    state = visible(t0)
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

