"""
TLE Data Fetcher - Download real satellite TLE data from CelesTrak.

Usage:
    python tle_fetcher.py --type starlink --output data/tle/starlink.tle
    python tle_fetcher.py --type planet --output data/tle/planet.tle
    python tle_fetcher.py --all --output data/tle/
"""

import argparse
import os
import urllib.request
import ssl
from typing import List, Optional

# CelesTrak TLE URLs
CELESTRAK_URLS = {
    'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
    'planet': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=tle',
    'stations': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle',
    'weather': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle',
    'resource': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle',
}


def fetch_tle_data(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch TLE data from URL.

    Args:
        url: CelesTrak URL
        timeout: Request timeout in seconds

    Returns:
        TLE data as string, or None if failed
    """
    print(f"Fetching TLE data from {url}...")

    # Create SSL context that doesn't verify certificates (for corporate proxies)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            data = response.read().decode('utf-8')
            print(f"  Received {len(data)} bytes")
            return data
    except Exception as e:
        print(f"  Failed to fetch: {e}")
        return None


def parse_tle_from_content(content: str) -> List[str]:
    """Parse TLE entries from raw content.

    Args:
        content: Raw TLE content

    Returns:
        List of TLE entries (name\nline1\nline2\n format)
    """
    lines = content.strip().split('\n')
    entries = []
    i = 0

    while i < len(lines) - 2:
        name_line = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()

        # Validate TLE format
        if line1.startswith('1 ') and line2.startswith('2 '):
            entries.append(f"{name_line}\n{line1}\n{line2}")

        i += 1

    return entries


def save_tle_file(filepath: str, entries: List[str]):
    """Save TLE entries to file.

    Args:
        filepath: Output file path
        entries: List of TLE entries
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries))

    print(f"Saved {len(entries)} TLE entries to {filepath}")


def fetch_and_save(tle_type: str, output_path: str, max_entries: Optional[int] = None):
    """Fetch TLE data and save to file.

    Args:
        tle_type: Type of TLE ('starlink', 'planet', etc.)
        output_path: Output file path
        max_entries: Maximum number of entries to save (None for all)
    """
    if tle_type not in CELESTRAK_URLS:
        print(f"Unknown TLE type: {tle_type}")
        print(f"Available types: {list(CELESTRAK_URLS.keys())}")
        return

    url = CELESTRAK_URLS[tle_type]
    content = fetch_tle_data(url)

    if content:
        entries = parse_tle_from_content(content)
        if max_entries:
            entries = entries[:max_entries]
        save_tle_file(output_path, entries)
    else:
        print(f"Failed to fetch {tle_type} TLE data")


def main():
    parser = argparse.ArgumentParser(description='Download TLE data from CelesTrak')
    parser.add_argument('--type', '-t', choices=['starlink', 'planet', 'stations', 'weather', 'resource'],
                       help='Type of satellites to download')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--all', '-a', action='store_true', help='Download all types')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max entries per file')

    args = parser.parse_args()

    base_dir = args.output if args.output else 'data/tle'

    if args.all:
        for tle_type in CELESTRAK_URLS.keys():
            output_path = os.path.join(base_dir, f"{tle_type}.tle")
            fetch_and_save(tle_type, output_path, args.max)
    elif args.type and args.output:
        fetch_and_save(args.type, args.output, args.max)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()