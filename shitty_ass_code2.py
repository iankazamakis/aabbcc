import numpy as np
from skyfield.api import load, EarthSatellite, Time, utc
from datetime import datetime, timedelta
from numpy import linspace, multiply, flatnonzero, diff
import time
from skyfield.searchlib import _find_discrete

class OrbitalAccessGen:
    def __init__(self, ts, time_step: timedelta = timedelta(seconds=30)):
        self.ts = ts # timescale shared across all of access-gen
        self.time_step = time_step.total_seconds() / 86400.0 # coarse time step for propagation
        self.position_cache = {} # cache for satellite positions, so we don't have to repropogate for each pair

    def _get_or_propagate_positions(self, satellite: EarthSatellite, start: datetime, end: datetime, times: Time) -> np.ndarray:
        """Cache or compute satellite positions for given times"""
        sat_id = id(satellite)
        cache_key = (sat_id, (start, end))
        if cache_key not in self.position_cache:
            self.position_cache[cache_key] = satellite.at(times)
        return self.position_cache[cache_key]

    def find_access_windows(self, target: EarthSatellite, observer: EarthSatellite, start_time: datetime, end_time: datetime, max_distance_km: float = 12000.0) -> list:
        """Find access windows where two satellites are within max_distance_km and not occluded by Earth"""
        # default to UTC in case the timezone is missing
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=utc)

        start_jd = self.ts.from_datetime(start_time).tt
        end_jd = self.ts.from_datetime(end_time).tt
        # create array of julian dates from start to end with time_step as a coarse interval
        jds = np.arange(start_jd, end_jd + self.time_step, self.time_step)
        times = self.ts.tt_jd(jds)

        # retrieve or compute positions
        target_geo = self._get_or_propagate_positions(target, start_time, end_time, times)
        observer_geo = self._get_or_propagate_positions(observer, start_time, end_time, times)
        pos_target = target_geo.position.km
        pos_observer = observer_geo.position.km
        distances = np.sqrt(np.sum((pos_observer - pos_target) ** 2, axis=0)) # euclidean distance between satellites at each time

        def visibility_function(t: Time) -> np.ndarray:
            """Check if satellites are in range and not occluded at given times."""
            tt = np.atleast_1d(t.tt)  # numpy 1d Time array check 
            times_t = self.ts.tt_jd(tt)
            # use cached positions for initial grid evaluation; for refinements, compute directly
            if len(tt) == len(jds) and np.allclose(tt, jds):  # check if times match the cached grid
                target_geo_t = target_geo
                observer_geo_t = observer_geo
            else:
                # otherwise compute the finer times, so that we don't lose windows that are < self.time_step long
                target_geo_t = target.at(times_t)
                observer_geo_t = observer.at(times_t)
            pos_vec = observer_geo_t - target_geo_t
            # is_behind_earth() requires a Geocentric vector with an observer attached, so we need to that manually here for the occlusion check
            pos_vec._observer_gcrs_au = target_geo_t.position.au
            occluded = pos_vec.is_behind_earth()
            distances_t = np.sqrt(np.sum((observer_geo_t.position.km - target_geo_t.position.km) ** 2, axis=0))
            in_range = distances_t <= max_distance_km
            return in_range & ~occluded

        epsilon = 0.5 / 86400.0 # TODO(eheyssler): make epsilon configurable, right now it's just 0.5 seconds
        ends, values = _find_discrete(self.ts, jds, visibility_function, epsilon=epsilon, num=12) # TODO(eheyssler): also make num configurable

        # handle edge case for access window spanning entire period by sampling initial point
        if len(ends) == 0 and len(values) == 0:
            check_time = self.ts.from_datetime(start_time)
            if visibility_function(check_time):
                return [(start_time.replace(tzinfo=utc), end_time.replace(tzinfo=utc))]

        # build access windows
        access_windows = [] 
        for i in range(len(values) - 1):
            if values[i]:
                start_t = ends[i].utc_datetime()
                end_t = ends[i + 1].utc_datetime()
                access_windows.append((start_t, end_t))

        return access_windows

def compute_access_windows(target: EarthSatellite, observer: EarthSatellite, start_time: datetime, end_time: datetime) -> list:
    """Compute access windows between target and observer satellites."""
    ts = load.timescale()  # Create a Skyfield timescale object
    calculator = OrbitalAccessGen(ts, time_step=timedelta(seconds=30))  # Initialize calculator with 30-second step
    return calculator.find_access_windows(target, observer, start_time, end_time)  # Call find_access_windows and return results

if __name__ == "__main__":
    from skyfield.api import load, EarthSatellite

    # Load timescale
    ts = load.timescale()

    # Define 15 satellites with varied TLEs (inspired by LEO satellites)
    satellites = []
    base_tle = [
        "1 99999U 25001A   25236.12345678  .00016717  00000-0  10270-3 0  9999",
        "2 99999  {inclination} {raan} 0003456  {arg_perigee} {ma} 15.72123456789012"
    ]
    for i in range(15):
        inclination = 51.6416 + (i % 5) * 5.0
        raan = (i * 24.0) % 360.0
        ma = (i * 24.0) % 360.0
        arg_perigee = 87.6543 + (i % 5) * 10.0
        tle = [base_tle[0].replace("99999", f"999{i:02d}"),
               base_tle[1].format(inclination=f"{inclination:.4f}", raan=f"{raan:.4f}", ma=f"{ma:.4f}", arg_perigee=f"{arg_perigee:.4f}")]
        satellites.append(EarthSatellite(tle[0], tle[1], f"Sat{i+1}", ts))

    # Define time range (~3 hours)
    start = datetime(2025, 8, 24, 0, 0, 0, tzinfo=utc)
    end = datetime(2025, 8, 24, 3, 0, 0, tzinfo=utc)

    # Measure runtime for 100 pairs
    start_time = time.perf_counter()
    
    # Compute access windows for first 100 unique pairs
    pair_count = 0
    for i in range(len(satellites)):
        for j in range(i + 1, len(satellites)):
            if pair_count >= 100:
                break
            target = satellites[i]
            observer = satellites[j]
            
            if pair_count < 10:
                print(f"\nComputing access windows for {target.name} (target) and {observer.name} (observer):")
                sample_times = [start, start + (end - start) / 2, end]
                for t in sample_times:
                    t_jd = ts.from_datetime(t)
                    pos_target = target.at(t_jd).position.km
                    pos_observer = observer.at(t_jd).position.km
                    distance = np.sqrt(np.sum((pos_observer - pos_target) ** 2))
                    print(f"  Distance at {t}: {distance:.2f} km")

            windows = compute_access_windows(target, observer, start, end)
            if pair_count < 10:
                if windows:
                    for start_t, end_t in windows:
                        print(f"  Access window: {start_t} ({start_t.microsecond} µs) to {end_t} ({end_t.microsecond} µs)")
                else:
                    print("  No access windows found.")

            pair_count += 1
        if pair_count >= 100:
            break

    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"\nTotal runtime for {pair_count} pairs: {runtime:.3f} seconds")
