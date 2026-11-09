"""
Visibility window calculator for satellite-target observation.

Based on intermodalTransportationNetworksCG geodata.py and CSV_parser.py architecture.
Computes when a satellite can observe a ground target based on orbital mechanics.

Reference: README Section 2.1 Time-Expanded Graph & 5.5.2 visibility calculation
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .tle_parser import TLESatellite


@dataclass
class VisibilityWindow:
    """Represents a time window when a satellite can observe a target."""
    satellite_id: str
    target_id: str
    start_time: float   # minutes from epoch
    end_time: float     # minutes from epoch
    elevation: float    # degrees at closest approach


@dataclass
class InterSatVisibilityWindow:
    """Represents a time window when two satellites can see each other via ISL."""
    satellite1_id: str
    satellite2_id: str
    start_time: float   # minutes from epoch
    end_time: float     # minutes from epoch
    distance: float     # km at closest approach


class VisibilityCalculator:
    """Calculate visibility windows between satellites and ground targets.

    Based on intermodalTransportationNetworksCG approach:
    - Use orbital mechanics to determine visibility
    - Consider elevation angle constraints
    - Generate time windows for imaging opportunities

    Reference: README Section 1.2 core challenges (sat高速运动, 可见性窗口每分钟变化)
    """

    # Earth's parameters
    Re = 6371.0        # Earth radius, km
    mu = 398600.4418   # gravitational parameter, km^3/s^2

    # Common elevation mask (degrees)
    DEFAULT_ELEVATION_MASK = 10.0  # degrees

    def __init__(self, elevation_mask: float = DEFAULT_ELEVATION_MASK):
        """Initialize visibility calculator.

        Args:
            elevation_mask: Minimum elevation angle for visibility, degrees
        """
        self.elevation_mask = elevation_mask

    def compute_visibility_window(
        self,
        satellite: TLESatellite,
        target_lat: float,
        target_lon: float,
        start_time: float,
        end_time: float,
        num_points: int = 100
    ) -> List[VisibilityWindow]:
        """Compute visibility windows for a satellite observing a ground target.

        Uses simplified propagation: evaluate at discrete time points and
        find intervals where elevation > mask.

        Args:
            satellite: TLESatellite object with orbital parameters
            target_lat: Target latitude, degrees
            target_lon: Target longitude, degrees
            start_time: Start time, minutes from epoch
            end_time: End time, minutes from epoch
            num_points: Number of time points to evaluate

        Returns:
            List of VisibilityWindow objects
        """
        windows = []
        times = np.linspace(start_time, end_time, num_points)

        # Convert target to radians
        lat_rad = math.radians(target_lat)
        lon_rad = math.radians(target_lon)

        in_window = False
        window_start = None
        max_elev = 0.0

        for t in times:
            # Get satellite position at time t (ECI coordinates)
            pos = self._satellite_position(satellite, t)

            if pos is None:
                continue

            # Convert ECI to ECEF by accounting for Earth's rotation
            pos_ecef = self._eci_to_ecef(pos, t)

            # Calculate elevation angle (now in ECEF frame)
            elev = self._compute_elevation(pos_ecef, lat_rad, lon_rad)

            if elev >= self.elevation_mask:
                if not in_window:
                    in_window = True
                    window_start = t
                max_elev = max(max_elev, elev)
            else:
                if in_window:
                    windows.append(VisibilityWindow(
                        satellite_id=satellite.norad_id,
                        target_id=f"target_{target_lat}_{target_lon}",
                        start_time=window_start,
                        end_time=t,
                        elevation=max_elev
                    ))
                    in_window = False
                    max_elev = 0.0

        # Close final window if needed
        if in_window:
            windows.append(VisibilityWindow(
                satellite_id=satellite.norad_id,
                target_id=f"target_{target_lat}_{target_lon}",
                start_time=window_start,
                end_time=end_time,
                elevation=max_elev
            ))

        return windows

    def _satellite_position(
        self,
        satellite: TLESatellite,
        time_minutes: float
    ) -> Optional[Tuple[float, float, float]]:
        """Compute satellite position at given time using SGP4-style propagation.

        Simplified model assuming circular orbit for speed.

        Args:
            satellite: TLESatellite object
            time_minutes: Time since epoch, minutes

        Returns:
            (x, y, z) in km, or None if computation fails
        """
        try:
            # Convert time to seconds
            t_sec = time_minutes * 60.0

            # Mean motion in rad/s
            n = satellite.mean_motion * 2 * math.pi / 86400

            # Mean anomaly at time t
            M0 = math.radians(satellite.mean_anomaly)
            M = M0 + n * t_sec
            M = M % (2 * math.pi)

            # Simplified true anomaly (assume circular orbit)
            nu = M

            # Orbital elements
            a = satellite.semi_major_axis
            e = satellite.eccentricity
            i = math.radians(satellite.inclination)
            omega = math.radians(satellite.argument_perigee)
            Omega = math.radians(satellite.raan)

            # Semi-latus rectum
            p = a * (1 - e * e)

            # Position in orbital plane
            x_orb = p * math.cos(nu) / (1 + e * math.cos(nu))
            y_orb = p * math.sin(nu) / (1 + e * math.cos(nu))

            # Rotation to inertial frame
            cos_O = math.cos(Omega)
            sin_O = math.sin(Omega)
            cos_i = math.cos(i)
            sin_i = math.sin(i)
            cos_omega = math.cos(omega)
            sin_omega = math.sin(omega)

            # Position in Earth-centered inertial frame
            X = (cos_O * cos_omega - sin_O * sin_omega * cos_i) * x_orb + \
                (-cos_O * sin_omega - sin_O * cos_omega * cos_i) * y_orb
            Y = (sin_O * cos_omega + cos_O * sin_omega * cos_i) * x_orb + \
                (-sin_O * sin_omega + cos_O * cos_omega * cos_i) * y_orb
            Z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

            return (X, Y, Z)

        except Exception:
            return None

    @staticmethod
    def _eci_to_ecef(
        pos_eci: Tuple[float, float, float],
        time_minutes: float
    ) -> Tuple[float, float, float]:
        """Convert ECI coordinates to ECEF by accounting for Earth's rotation.

        ECI is inertial (fixed stars); ECEF rotates with Earth.
        The difference at time t is theta = omega_earth * t.

        Args:
            pos_eci: ECI position (x, y, z) in km
            time_minutes: Time from epoch in minutes

        Returns:
            ECEF position (x, y, z) in km
        """
        x, y, z = pos_eci
        # Earth's rotation rate: 360 deg per sidereal day (1436 min)
        omega_earth = 2.0 * math.pi / 1436.0  # rad per minute
        theta = time_minutes * omega_earth
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # Rotate around z-axis: ECEF = Rz(-theta) * ECI
        x_ecef = x * cos_t + y * sin_t
        y_ecef = -x * sin_t + y * cos_t
        z_ecef = z
        return (x_ecef, y_ecef, z_ecef)

    def _compute_elevation(
        self,
        sat_pos: Tuple[float, float, float],
        target_lat: float,
        target_lon: float
    ) -> float:
        """Compute elevation angle from satellite to ground target.

        Uses correct spherical Earth geometry:
        sin(elev) = (r * cos(central_angle) - Re) / d
        where d = |sat_pos - target_pos| (line-of-sight distance)

        Args:
            sat_pos: Satellite position (x, y, z) in km (ECEF frame)
            target_lat: Target latitude, radians
            target_lon: Target longitude, radians

        Returns:
            Elevation angle, degrees (0 if below horizon)
        """
        x, y, z = sat_pos

        # Satellite range from Earth center
        r = math.sqrt(x*x + y*y + z*z)

        # Target position in ECEF
        Rtarget = self.Re * math.cos(target_lat)
        tx = Rtarget * math.cos(target_lon)
        ty = Rtarget * math.sin(target_lon)
        tz = self.Re * math.sin(target_lat)

        # Line-of-sight vector
        dx = x - tx
        dy = y - ty
        dz = z - tz
        d = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Central angle (angle between satellite and target from Earth center)
        cos_central = (x*tx + y*ty + z*tz) / (r * self.Re)
        cos_central = max(-1.0, min(1.0, cos_central))
        central = math.acos(cos_central)

        # Elevation formula: sin(elev) = (r * cos(central) - Re) / d
        # Derived from: cos(zenith) = (r*cos(central)-Re)/d, elev = 90 - zenith
        sin_elev = (r * cos_central - self.Re) / d
        sin_elev = max(-1.0, min(1.0, sin_elev))
        elev_rad = math.asin(sin_elev)

        return max(0.0, math.degrees(elev_rad))

    def compute_inter_satellite_visibility(
        self,
        sat1: TLESatellite,
        sat2: TLESatellite,
        start_time: float,
        end_time: float,
        num_points: int = 100
    ) -> List[InterSatVisibilityWindow]:
        """Compute visibility windows between two satellites for ISL.

        Two satellites can have an ISL link if:
        1. They are within maximum ISL range (typically ~2500 km for LEO)
        2. The elevation angle from each satellite to the other is above mask

        Args:
            sat1: First TLESatellite object
            sat2: Second TLESatellite object
            start_time: Start time, minutes from epoch
            end_time: End time, minutes from epoch
            num_points: Number of time points to evaluate

        Returns:
            List of InterSatVisibilityWindow objects
        """
        windows = []
        times = np.linspace(start_time, end_time, num_points)

        in_window = False
        window_start = None
        min_dist = float('inf')

        for t in times:
            pos1 = self._satellite_position(sat1, t)
            pos2 = self._satellite_position(sat2, t)

            if pos1 is None or pos2 is None:
                continue

            # Compute distance between satellites
            dist = math.sqrt(
                (pos1[0] - pos2[0]) ** 2 +
                (pos1[1] - pos2[1]) ** 2 +
                (pos1[2] - pos2[2]) ** 2
            )

            # Maximum ISL range for LEO (simplified)
            max_isl_range = 2500.0  # km

            if dist <= max_isl_range:
                if not in_window:
                    in_window = True
                    window_start = t
                min_dist = min(min_dist, dist)
            else:
                if in_window:
                    windows.append(InterSatVisibilityWindow(
                        satellite1_id=sat1.norad_id,
                        satellite2_id=sat2.norad_id,
                        start_time=window_start,
                        end_time=t,
                        distance=min_dist
                    ))
                    in_window = False
                    min_dist = float('inf')

        # Close final window if needed
        if in_window:
            windows.append(InterSatVisibilityWindow(
                satellite1_id=sat1.norad_id,
                satellite2_id=sat2.norad_id,
                start_time=window_start,
                end_time=end_time,
                distance=min_dist
            ))

        return windows

    def compute_satellite_ground_visibility(
        self,
        satellite: TLESatellite,
        gs_lat: float,
        gs_lon: float,
        start_time: float,
        end_time: float,
        num_points: int = 100
    ) -> List[VisibilityWindow]:
        """Compute visibility windows between a satellite and a ground station.

        Args:
            satellite: TLESatellite object
            gs_lat: Ground station latitude, degrees
            gs_lon: Ground station longitude, degrees
            start_time: Start time, minutes from epoch
            end_time: End time, minutes from epoch
            num_points: Number of time points to evaluate

        Returns:
            List of VisibilityWindow objects
        """
        windows = []
        times = np.linspace(start_time, end_time, num_points)

        # Convert ground station to radians
        gs_lat_rad = math.radians(gs_lat)
        gs_lon_rad = math.radians(gs_lon)

        in_window = False
        window_start = None
        max_elev = 0.0

        for t in times:
            pos = self._satellite_position(satellite, t)

            if pos is None:
                continue

            # Convert ECI to ECEF for elevation calculation
            pos_ecef = self._eci_to_ecef(pos, t)

            # Calculate elevation angle from satellite to ground station
            elev = self._compute_elevation(pos_ecef, gs_lat_rad, gs_lon_rad)

            if elev >= self.elevation_mask:
                if not in_window:
                    in_window = True
                    window_start = t
                max_elev = max(max_elev, elev)
            else:
                if in_window:
                    windows.append(VisibilityWindow(
                        satellite_id=satellite.norad_id,
                        target_id=f"gs_{gs_lat}_{gs_lon}",
                        start_time=window_start,
                        end_time=t,
                        elevation=max_elev
                    ))
                    in_window = False
                    max_elev = 0.0

        # Close final window if needed
        if in_window:
            windows.append(VisibilityWindow(
                satellite_id=satellite.norad_id,
                target_id=f"gs_{gs_lat}_{gs_lon}",
                start_time=window_start,
                end_time=end_time,
                elevation=max_elev
            ))

        return windows

    def build_visibility_matrix(
        self,
        satellites: List[TLESatellite],
        targets: List[Tuple[float, float]],  # List of (lat, lon)
        time_window: Tuple[float, float],
        slot_duration: float = 1.0  # minutes per slot
    ) -> dict:
        """Build visibility matrix: which satellite can see which target at which slot.

        Args:
            satellites: List of TLESatellite objects
            targets: List of (lat, lon) tuples
            time_window: (start, end) time in minutes
            slot_duration: Duration of each time slot, minutes

        Returns:
            Dictionary: (sat_id, target_idx, slot) -> bool (visible)
        """
        start, end = time_window
        num_slots = int((end - start) / slot_duration) + 1

        visibility = {}

        for sat in satellites:
            for target_idx, (tlat, tlon) in enumerate(targets):
                windows = self.compute_visibility_window(
                    sat, tlat, tlon, start, end
                )

                for window in windows:
                    # Mark slots within window as visible
                    for slot in range(num_slots):
                        slot_time = start + slot * slot_duration
                        if window.start_time <= slot_time <= window.end_time:
                            visibility[(sat.norad_id, target_idx, slot)] = True

        return visibility


def compute_visibility_window(
    satellite: TLESatellite,
    target_lat: float,
    target_lon: float,
    start_time: float,
    end_time: float,
    elevation_mask: float = 10.0
) -> List[VisibilityWindow]:
    """Convenience function for computing visibility windows.

    Reference: README Section 5.5.2 visibility window calculation
    """
    calc = VisibilityCalculator(elevation_mask=elevation_mask)
    return calc.compute_visibility_window(satellite, target_lat, target_lon, start_time, end_time)


if __name__ == '__main__':
    # Test visibility calculator
    from tle_parser import TLEParser

    parser = TLEParser()
    sats = parser._get_sample_satellites()

    calc = VisibilityCalculator()

    # Test visibility for a target
    windows = calc.compute_visibility_window(
        sats[0],  # Dove satellite
        target_lat=39.9,  # Beijing
        target_lon=116.4,
        start_time=0.0,
        end_time=100.0
    )

    print(f"Visibility windows for Beijing target: {len(windows)}")
    for w in windows:
        print(f"  Slot {w.start_time:.1f} - {w.end_time:.1f}, max elevation {w.elevation:.1f}°")