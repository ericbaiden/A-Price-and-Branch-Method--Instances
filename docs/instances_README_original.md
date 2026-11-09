# LEO Constellation Scheduling Instances

This folder contains instance descriptions for the LEO hybrid constellation scheduling problem.

## Instance Structure

Each instance is defined by:
- **RS (Remote Sensing) satellites**: Real SKYSAT/FLOCK Earth observation satellites from TLE data
- **CS (Communication) satellites**: Real Starlink communication satellites from TLE data
- **Ground Stations**: Beijing, Shanghai, Guangzhou

## Constellation Data (from Real TLE)

### RS Layer (Remote Sensing)
- **Source**: `data/tle/planet.tle` (136 real SKYSAT/FLOCK satellites)
- **Inclination**: ~97бу (sun-synchronous orbit, range 96.9бу-98.4бу)
- **Altitude**: 319-612 km (varies by satellite)
- **Description**: Real Earth observation satellites (SKYSAT, FLOCK) with optical imaging capability

### CS Layer (Communication)
- **Source**: `data/tle/starlink.tle` (1321 Starlink satellites at 550km, 53бу)
- **Inclination**: 53.1бу-53.2бу (Starlink Shell 1)
- **Altitude**: 543-547 km (clustered around 550km)
- **Description**: Real Starlink communication satellites for data relay

## Instance Categories

### Validation Instances (`validation/`)
- **RS=10, CS=50**: Reduced-scale for exact validation vs full MIP
- Tasks: 50, 100
- Seeds: 0-9

### Large Scale Instances (`large_scale/`)
- **RS=136, CS=1321**: Full constellation from real TLE data
- Tasks: 100, 200, 500, 1000
- Seeds: 0-9

### Paper Instances (`paper/`)
- Subset of instances used for figure generation
- Small: 10 RS + 50 CS, 50 tasks, seeds 0-2
- Large: 136 RS + 1321 CS, 500 tasks, seeds 0-2

## Instance Parameters

- **Planning horizon**: 180 minutes
- **Time slot**: 200 seconds
- **Number of slots**: 54
- **ISL range**: 2500 km
- **Downlink bandwidth**: 1000 Mbps

## Generating Instances

To regenerate instances from TLE data:
```bash
python experiments.py run --rs 10 --cs 50 --seed 0 --subset 0.25
```

## Result Files

Experiment results are stored in `results/` folder with matching filenames.
