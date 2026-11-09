"""
TLE (Two-Line Element) parser for satellite orbit data.
Based on intermodalTransportationNetworksCG XML_parser.py architecture.

Data source: Celestrak (https://celestrak.org)
Satellites: Starlink (communication), Planet Dove (remote sensing)

Reference: README Section 4.1.1 TLE数据获取
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class TLESatellite:
    """Satellite object parsed from TLE data."""
    norad_id: str
    name: str
    tle_line1: str
    tle_line2: str
    # Orbital elements
    inclination: float        # degrees
    raan: float               # right ascension of ascending node, degrees
    eccentricity: float       # dimensionless
    argument_perigee: float    # degrees
    mean_anomaly: float       # degrees
    mean_motion: float         # rev/day
    epoch: str                # epoch timestamp
    # Derived parameters (set by TLEParser)
    semi_major_axis: float = 0.0   # km
    period: float = 0.0           # minutes
    altitude: float = 0.0          # km

    @property
    def satellite_type(self) -> str:
        """Determine satellite type from name."""
        name_lower = self.name.lower()
        if 'starlink' in name_lower:
            return 'CS'  # Communication satellite
        elif any(x in name_lower for x in ['planet', 'dove', 'sentinel', 'landsat']):
            return 'RS'  # Remote sensing satellite
        return 'RS'  # Default to RS


class TLEParser:
    """Parses TLE data from Celestrak format.

    Based on intermodalTransportationNetworksCG XML_parser.py pattern:
    - Read and parse format-specific data files
    - Extract relevant parameters
    - Build satellite objects
    """

    # TLE format constants
    TLE_LINE_LENGTH = 70

    # Celestrak URLs for real satellite data (from README Section 4.1.1)
    CELESTRAK_URLS = {
        'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
        'planet': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=tle',
    }

    def __init__(self, tle_file: Optional[str] = None):
        """Initialize TLE parser.

        Args:
            tle_file: Path to local TLE file. If None, use default sample.
        """
        self.tle_file = tle_file
        self.satellites: List[TLESatellite] = []

    def parse_file(self, filepath: Optional[str] = None) -> List[TLESatellite]:
        """Parse TLE file and return list of satellite objects.

        Args:
            filepath: Path to TLE file. Uses self.tle_file if None.

        Returns:
            List of TLESatellite objects.
        """
        filepath = filepath or self.tle_file
        if not filepath:
            # Use sample data
            return self._get_sample_satellites()

        satellites = []
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse TLE format: each satellite has 3 lines (name + 2 TLE lines)
        lines = content.strip().split('\n')

        i = 0
        while i < len(lines) - 2:
            name_line = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            # Validate TLE format (line 1 starts with '1 ', line 2 starts with '2 ')
            if line1.startswith('1 ') and line2.startswith('2 '):
                sat = self._parse_tle_lines(name_line, line1, line2)
                if sat:
                    satellites.append(sat)

            i += 3

        self.satellites = satellites
        return satellites

    def _parse_tle_lines(self, name: str, line1: str, line2: str) -> Optional[TLESatellite]:
        """Parse TLE lines into TLESatellite object.

        Reference: README Section 4.1.2 Real Satellite Parameters
        """
        try:
            # Extract NORAD catalog number
            norad_id = line1[2:7].strip()

            # Extract epoch year and day
            epoch_year = line1[18:20].strip()
            epoch_day = line1[20:32].strip()

            # Handle 2-digit year
            year = int(epoch_year)
            year = 2000 + year if year < 57 else 1900 + year
            epoch = f"{year}-{float(epoch_day):.8f}"

            # Extract orbital elements from line 2
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33].strip().zfill(7))
            argument_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])

            sat = TLESatellite(
                norad_id=norad_id,
                name=name,
                tle_line1=line1,
                tle_line2=line2,
                inclination=inclination,
                raan=raan,
                eccentricity=eccentricity,
                argument_perigee=argument_perigee,
                mean_anomaly=mean_anomaly,
                mean_motion=mean_motion,
                epoch=epoch,
            )

            # Calculate derived parameters
            self._compute_derived_params(sat)

            return sat

        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse TLE for {name}: {e}")
            return None

    def _compute_derived_params(self, sat: TLESatellite) -> None:
        """Compute semi-major axis, period, and altitude from TLE data.

        Based on: README Section 4.1.2 Real Satellite Parameters table
        """
        # Mean motion in rev/day -> rad/s
        n = sat.mean_motion * 2 * math.pi / 86400  # rad/s

        # Earth's gravitational parameter
        mu = 398600.4418  # km^3/s^2

        # Semi-major axis from mean motion: a = (mu/n^2)^(1/3)
        sat.semi_major_axis = (mu / (n ** 2)) ** (1/3)

        # Orbital period in minutes
        sat.period = 1440 / sat.mean_motion

        # Altitude (assuming circular orbit)
        Re = 6371.0  # Earth radius, km
        sat.altitude = sat.semi_major_axis - Re

    def filter_by_type(self, sat_type: str) -> List[TLESatellite]:
        """Filter satellites by type (RS or CS).

        Args:
            sat_type: 'RS' for remote sensing, 'CS' for communication

        Returns:
            List of matching satellites.
        """
        return [s for s in self.satellites if s.satellite_type == sat_type]

    def get_rs_satellites(self) -> List[TLESatellite]:
        """Get remote sensing satellites (Dove/Planet)."""
        return self.filter_by_type('RS')

    def get_cs_satellites(self) -> List[TLESatellite]:
        """Get communication satellites (Starlink)."""
        return self.filter_by_type('CS')

    @staticmethod
    def _get_sample_satellites() -> List[TLESatellite]:
        """Return sample TLE data for testing.

        Based on README Section 4.1.2 satellite parameters:
        - Dove (RS): 475km altitude, 97.4° SSO
        - Starlink (CS): 550km altitude, 53° inclination
        """
        # Sample Starlink-like TLE
        starlink_sample = TLESatellite(
            norad_id='44001',
            name='STARLINK-1001',
            tle_line1='1 44001U 19029A   19348.87962839 -.00032895  00000-0 -91110-3 0  9991',
            tle_line2='2 44001  53.0054 292.7005 0001577  73.3240 286.8011 15.11234592    15',
            inclination=53.0,
            raan=292.7,
            eccentricity=0.0001577,
            argument_perigee=73.3,
            mean_anomaly=286.8,
            mean_motion=15.11,
            epoch='2019-12-14',
        )
        starlink_sample.semi_major_axis = 6928.0  # km (550km altitude)
        starlink_sample.period = 96.0  # minutes
        starlink_sample.altitude = 550.0  # km

        # Sample Dove-like TLE
        dove_sample = TLESatellite(
            norad_id='40000',
            name='DOVE-1',
            tle_line1='1 40000U 14033A   19348.50000000  .00000667  00000-0  16333-4 0  9992',
            tle_line2='2 40000  97.4400 292.7005 0001342  83.3240 276.8011 15.23600000    19',
            inclination=97.44,
            raan=292.7,
            eccentricity=0.0001342,
            argument_perigee=83.3,
            mean_anomaly=276.8,
            mean_motion=15.24,
            epoch='2019-12-14',
        )
        dove_sample.semi_major_axis = 6809.0  # km (475km altitude)
        dove_sample.period = 94.6  # minutes
        dove_sample.altitude = 475.0  # km

        return [starlink_sample, dove_sample]


def load_tle_from_file(filepath: str) -> List[TLESatellite]:
    """Convenience function to load TLE data from file."""
    parser = TLEParser(tle_file=filepath)
    return parser.parse_file()


if __name__ == '__main__':
    # Test TLE parser
    parser = TLEParser()
    sats = parser._get_sample_satellites()

    print("Sample satellites loaded:")
    for sat in sats:
        print(f"  {sat.name}: type={sat.satellite_type}, "
              f"altitude={sat.altitude:.0f}km, period={sat.period:.1f}min")