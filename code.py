import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID
from math import pi, radians
from skyfield.api import EarthSatellite, Topos, load
from skyfield.searchlib import find_discrete
from skyfield.constants import ERAD

# Assuming Config, Entity, PlanningConstraints, FanGeometry, FanBeamMode1, etc., are defined elsewhere
# Also assuming helper functions like calculate_night_periods, calculate_sunlit_periods, adjust_access_for_illumination,
# apply_cutout_azimuth, apply_cutout_elevation, apply_fan_limits_vertical, apply_fan_limits_horizontal,
# apply_fan_limits_maximum_range, create_access_points, events_to_accesses, _process_events are defined.

# Earth radius in km
EARTH_RADIUS_KM = ERAD / 1000.0

@dataclass
class AccessPoint:
    epoch: datetime
    azimuth_radians: float
    elevation_radians: float
    range_meters: float

@dataclass
class Access:
    sat: EarthSatellite
    observer: object  # Can be Topos or EarthSatellite
    start: datetime
    end: datetime
    uuid: UUID
    culmination: Optional[datetime] = None
    access_points: Optional[list[AccessPoint]] = None
    generated_time: Optional[datetime] = None

def _event_finder(
    cfg: Config,
    sat: EarthSatellite,
    start: datetime,
    end: datetime,
    gs: Entity  # Now can be ground station or satellite entity
) -> Tuple[List[Access], List[Access]]:
    # Load timescale if not global
    ts = load.timescale()

    # Defaults for everything
    min_az_rad = 0.0
    max_az_rad = 2 * pi
    min_el_rad = radians(cfg.min_access_elevation_degrees)
    max_el_rad = pi / 2
    fan_params = FanGeometry()

    # Determine if gs is ground or satellite
    is_ground = True  # Assume based on gs properties; adjust logic as needed
    # Example: if gs.type == 'satellite': is_ground = False
    # For now, assume user modifies to detect

    if is_ground:
        # Existing ground logic
        is_telescope = True

        if "mission_planning_constraints" in gs.prototype_extensions.extensions:
            constraints = PlanningConstraints()
            constraints.ParseFromString(
                gs.prototype_extensions.extensions["mission_planning_constraints"].value
            )

            if constraints.HasField("sensor_pointing_limits"):
                pointing_limits = constraints.sensor_pointing_limits
                min_az_rad = pointing_limits.minimum_azimuth_radians
                max_az_rad = pointing_limits.maximum_azimuth_radians
                min_el_rad = pointing_limits.minimum_elevation_radians
                max_el_rad = pointing_limits.maximum_elevation_radians

            if constraints.HasField("fan_geometry"):
                is_telescope = False
                fan_params = constraints.fan_geometry

        loc = gs.location.position
        lat = loc.latitude_degrees
        lon = loc.longitude_degrees
        alt = loc.altitude_hae_meters.value

        observer = Topos(
            latitude_degrees=lat,
            longitude_degrees=lon,
            elevation_m=alt
        )
    else:
        # Satellite observer
        # Assume gs has TLE data to create EarthSatellite
        # Example: observer = EarthSatellite(gs.tle_line1, gs.tle_line2, gs.name, ts)
        # User to fill in the creation
        observer = ...  # Create EarthSatellite for observer
        is_telescope = False  # No pointing limits for sat-to-sat

    if cfg.accesses_calculate_day_night:
        if is_ground:
            night_periods = calculate_night_periods(observer, start, end)
        else:
            # For sat-to-sat, perhaps calculate eclipse periods or skip
            # Here, skipping day/night for simplicity; adjust if needed
            night_periods = [(start, end)]
    else:
        night_periods = [(start, end)]

    all_accesses: List[Access] = []
    illumination_filtered_accesses: List[Access] = []
    filtered_accesses: List[Access] = []

    for start_night, end_night in night_periods:
        if is_ground:
            # Existing ground logic
            events = sat.find_events(
                observer,
                ts.from_datetime(start_night),
                ts.from_datetime(end_night),
                altitude_degrees=sensor_elevation_effective(degrees(min_el_rad))  # Assuming degrees() is from math
            )

            if events:
                accesses = events_to_accesses(
                    _process_events(events),
                    start_night,
                    end_night,
                    sat,
                    observer
                )
                all_accesses.extend(accesses)
            else:
                continue  # No accesses in this period
        else:
            # Satellite-to-satellite logic
            def visibility_func(t):
                # t can be scalar or array Time
                # Compute positions in km
                r1 = observer.at(t).position.km  # Observer position
                r2 = sat.at(t).position.km  # Target position
                d = r2 - r1
                dd = np.sum(d**2, axis=0)  # |d|^2, handles array
                distance = np.sqrt(dd)  # |d|

                # Max distance check (assume cfg.max_visible_distance_meters exists)
                max_dist_km = cfg.max_visible_distance_meters / 1000.0
                if np.any(dd == 0):  # Coincident, visible if <= max
                    vis = (distance <= max_dist_km)
                    return vis.astype(float)

                rd = np.sum(r1 * d, axis=0)  # r1 · d
                lambda_f = -rd / dd

                # dist = |r1 x r2| / |d|
                cross = np.cross(r1, r2)  # Shape (3,) or (3,N) if array
                if cross.ndim == 1:
                    cross_mag = np.linalg.norm(cross)
                else:
                    cross_mag = np.linalg.norm(cross, axis=0)
                dist = cross_mag / distance

                # Visibility
                occulted = (dist < EARTH_RADIUS_KM) & (lambda_f >= 0) & (lambda_f <= 1)
                too_far = distance > max_dist_km
                vis = np.logical_not(occulted | too_far)

                return vis.astype(float)  # 1.0 if visible, 0.0 if not

            t0 = ts.from_datetime(start_night)
            t1 = ts.from_datetime(end_night)

            # Step size in days (e.g., 30 seconds)
            step_days = 30.0 / 86400.0

            times, y = find_discrete(t0, t1, visibility_func, stepsize=step_days)

            # Convert to datetime
            transition_times = [t.utc_datetime() for t in times.tt]  # Assuming .tt for jd, but .utc_datetime()

            # Initial state
            initial_vis = visibility_func(t0)

            accesses = []
            current_start = start_night
            current_state = 1 if initial_vis > 0.5 else 0  # Since float 1.0 or 0.0

            idx = 0
            while idx < len(times):
                ti = transition_times[idx]
                yi = 1 if y[idx] > 0.5 else 0

                if current_state == 1:
                    # End of access
                    access = Access(
                        sat=sat,
                        observer=observer,
                        start=current_start,
                        end=ti,
                        uuid=UUID(...)  # Generate UUID as needed
                    )
                    accesses.append(access)
                current_state = yi
                current_start = ti
                idx += 1

            # If ends visible
            if current_state == 1:
                access = Access(
                    sat=sat,
                    observer=observer,
                    start=current_start,
                    end=end_night,
                    uuid=UUID(...)
                )
                accesses.append(access)

            # Optionally compute culmination for each access, e.g., min distance time
            for access in accesses:
                def neg_distance(t):
                    r1 = observer.at(t).position.km
                    r2 = sat.at(t).position.km
                    dist = np.linalg.norm(r2 - r1, axis=0)
                    return -dist

                t_start = ts.from_datetime(access.start)
                t_end = ts.from_datetime(access.end)
                max_times, max_y = find_maxima(t_start, t_end, neg_distance, stepsize=step_days)
                if len(max_times) > 0:
                    access.culmination = max_times[0].utc_datetime()  # Take first/local max of -dist, i.e., min dist

            all_accesses.extend(accesses)

    # Filter on illumination geometry (same for both)
    if cfg.accesses_calculate_sunlit_accesses:
        for access in all_accesses:
            sunlit_periods = calculate_sunlit_periods(
                sat, access.start, access.end, 0.3
            )
            adjusted_accesses = adjust_access_for_illumination(
                access, sunlit_periods
            )
            illumination_filtered_accesses.extend(adjusted_accesses)
    else:
        illumination_filtered_accesses = all_accesses[:]  # copy

    # Filter on cutout geometry
    if len(illumination_filtered_accesses) > 0 and cfg.accesses_calculate_sensor_pointing_limits:
        if is_ground:
            if is_telescope:
                for access in illumination_filtered_accesses:
                    # Only perform azimuth filtering if not default
                    if are_default_azimuths(min_az_rad, max_az_rad):
                        az_filtered_accesses = [access]
                    else:
                        az_filtered_accesses = apply_cutout_azimuth(
                            access, min_az_rad, max_az_rad
                        )

                    # Elevation filtering
                    el_filtered_accesses = []
                    if round(max_el_rad, 2) < round(pi / 2, 2):
                        for a in az_filtered_accesses:
                            el_filtered_accesses.extend(
                                apply_cutout_elevation(a, min_el_rad, max_el_rad)
                            )
                        filtered_accesses.extend(el_filtered_accesses)
                    else:
                        filtered_accesses.extend(az_filtered_accesses)

            else:  # fan
                fan_beam = FanBeamMode1(
                    outer_radius=fan_params.maximum_range_meters / 1000.0,  # km
                    horizontal_beamwidth=np.radians(fan_params.horizontal_beamwidth_deg),
                    vertical_beamwidth=np.radians(fan_params.vertical_beamwidth_deg),
                    tilt_angle=np.radians(fan_params.boresight_elevation_deg),
                    rotation_angle_z=np.radians(90) - np.radians(fan_params.boresight_azimuth_deg),
                )

                for access in illumination_filtered_accesses:
                    vertical_filtered_accesses = apply_fan_limits_vertical(access, fan_beam)
                    horizontal_filtered_accesses = []
                    for va in vertical_filtered_accesses:
                        horizontal_filtered_accesses.extend(
                            apply_fan_limits_horizontal(va, fan_beam)
                        )

                    range_filtered_accesses = []
                    for ha in horizontal_filtered_accesses:
                        range_filtered_accesses.extend(
                            apply_fan_limits_maximum_range(ha, fan_beam)
                        )

                    filtered_accesses.extend(range_filtered_accesses)
        else:
            # For sat-to-sat, no pointing limits, skip filter
            filtered_accesses = illumination_filtered_accesses[:]
    else:
        filtered_accesses = illumination_filtered_accesses[:]  # copy

    # Populate access points
    for a in filtered_accesses:
        a.access_points = create_access_points(a)  # May need adaptation for sat-to-sat

    return all_accesses, filtered_accesses
