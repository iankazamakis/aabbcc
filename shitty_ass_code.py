import numpy as np
import time
from skyfield.api import load, EarthSatellite, Time, utc
from datetime import datetime, timedelta
from numpy import linspace, multiply, flatnonzero, diff
from skyfield.searchlib import _find_discrete

class SatelliteAccessCalculator:
    def __init__(self, ts, time_step: timedelta = timedelta(seconds=30)):
        """Initialize calculator with time scale."""
        self.ts = ts
        self.time_step = time_step.total_seconds() / 86400.0  # Convert to Julian days
        self.position_cache = {}  # Cache for satellite positions
        self.earth_radius_km = 6378.137  # Earth's equatorial radius in km

    def _cache_positions(self, satellite: EarthSatellite, jds: np.ndarray) -> np.ndarray:
        """Cache satellite positions for given times if not already cached."""
        sat_id = id(satellite)
        cache_key = (sat_id, tuple(jds))  # Include jds in cache key for uniqueness
        if cache_key not in self.position_cache:
            times = self.ts.tt_jd(jds)
            self.position_cache[cache_key] = satellite.at(times).position.km
        return self.position_cache[cache_key]

    def _is_line_of_sight_blocked(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """Check if Earth blocks the line of sight between two satellite positions."""
        # Ensure pos1 and pos2 are 2D (3, N) for vectorized computation
        if pos1.ndim == 1:
            pos1 = pos1[:, np.newaxis]
            pos2 = pos2[:, np.newaxis]
        
        # Vector from satellite 1 to satellite 2
        d = pos2 - pos1
        # Vector from satellite 1 to Earth's center (origin in GCRS)
        p1 = pos1
        # Parameter t for closest point on line to Earth's center
        dot_dd = np.sum(d * d, axis=0)
        dot_dd = np.where(dot_dd == 0, 1e-10, dot_dd)  # Avoid division by zero
        t = -np.sum(p1 * d, axis=0) / dot_dd
        # If t < 0 or t > 1, closest point is outside the segment
        outside_segment = (t < 0) | (t > 1)
        # Closest point on the line segment to Earth's center
        closest_point = p1 + t * d
        # Distance from Earth's center to closest point
        distance = np.sqrt(np.sum(closest_point ** 2, axis=0))
        # Earth blocks LOS if closest point is within Earth's radius and within segment
        blocked = (distance <= self.earth_radius_km) & ~outside_segment
        return blocked

    def find_access_windows(self, sat1: EarthSatellite, sat2: EarthSatellite, start_time: datetime, end_time: datetime, max_distance_km: float = 12000.0) -> list:
        """Find access windows where satellites are within max_distance_km and not occluded by Earth."""
        # Ensure datetimes have UTC timezone
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=utc)

        # Convert time range to Julian dates
        start_jd = self.ts.from_datetime(start_time).tt
        end_jd = self.ts.from_datetime(end_time).tt
        jds = np.arange(start_jd, end_jd + self.time_step, self.time_step)
        times = self.ts.tt_jd(jds)

        # Cache positions for both satellites at initial grid
        pos1 = self._cache_positions(sat1, jds)
        pos2 = self._cache_positions(sat2, jds)

        # Compute distances between satellites at initial grid
        distances = np.sqrt(np.sum((pos1 - pos2) ** 2, axis=0))

        # Check for visibility (within max_distance and not behind Earth)
        def visibility_function(t: Time) -> np.ndarray:
            # Handle array of times
            tt = np.atleast_1d(t.tt)
            idx = np.searchsorted(jds, tt, side='left')
            idx = np.clip(idx, 0, len(jds) - 1)

            # For refined times not on the initial grid, compute positions directly
            refined_jds = tt
            if not np.all(np.isin(refined_jds, jds)):
                refined_times = self.ts.tt_jd(refined_jds)
                pos1_refined = sat1.at(refined_times).position.km
                pos2_refined = sat2.at(refined_times).position.km
                distances_refined = np.sqrt(np.sum((pos1_refined - pos2_refined) ** 2, axis=0))
                in_range = distances_refined <= max_distance_km
                not_occluded = ~self._is_line_of_sight_blocked(pos1_refined, pos2_refined)
            else:
                in_range = distances[idx] <= max_distance_km
                not_occluded = ~self._is_line_of_sight_blocked(pos1[:, idx], pos2[:, idx])
            
            return in_range & not_occluded

        # Find discrete visibility windows with higher precision
        epsilon = 0.5 / 86400.0  # 0.1 seconds in Julian days
        ends, values = _find_discrete(self.ts, jds, visibility_function, epsilon=epsilon, num=100)

        # Convert to access window tuples (start_time, end_time)
        access_windows = []
        for i in range(len(values) - 1):
            if values[i]:  # True indicates visible
                start_t = ends[i].utc_datetime()  # Directly use Time object
                end_t = ends[i + 1].utc_datetime()  # Directly use Time object
                access_windows.append((start_t, end_t))

        return access_windows

def compute_access_windows(sat1: EarthSatellite, sat2: EarthSatellite, start_time: datetime, end_time: datetime) -> list:
    """Compute access windows between two satellites."""
    ts = load.timescale()
    calculator = SatelliteAccessCalculator(ts, time_step=timedelta(seconds=30))
    return calculator.find_access_windows(sat1, sat2, start_time, end_time)

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
