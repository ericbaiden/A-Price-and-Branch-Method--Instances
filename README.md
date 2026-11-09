# A Price-and-Branch Method for Large-Scale Collaborative Task Scheduling in Low-Earth-Orbit Hybrid Constellations

Code and reproducibility materials for the EJOR study of integrated task scheduling across remote-sensing (RS), communication (CS), and ground-station (GS) layers.

## Contents

- `Solver/`: column generation, A* and Dijkstra pricing, price-and-branch integer recovery, branch-and-price reference, GRASP, ALNS, and the optional native shortest-path backend.
- `Generator/`: TLE parsing, satellite/contact generation, and graph construction.
- `experiment/`: experiment configuration and suite entry points (including `experiment/suites/`).
- `data/tle/`: the TLE inputs used by the instance generator.
- `instances/`: instance descriptions, archived graph exports, and input checksums.
- `dataset/instances/`: 100 frozen three-hour graph/task instances; `h24/` contains 30 frozen 24-hour instances.
- `docs/`: campaign protocols and reproducibility notes.
- `scripts/`: result checks and table helpers.
- Top-level campaign and export scripts for the three-hour, 24-hour, heterogeneous-volume, certification, and ablation studies.

The repository includes the experiment code, lightweight inputs, instance descriptions, and 130 complete frozen graph/task instances. The instance input files total about 730 MB; the largest individual file is about 19.3 MB. Raw solver-result collections are supplied in the accompanying experimental archive.

## Main protocols

All reported main comparisons use rejection penalty `M=600`.

- Three-hour unit-volume comparison: 100--1000 tasks, 10 seeds per scale, 60-second ALNS search budget.
- 24-hour unit-volume comparison: 1000--10000 tasks, 3 seeds per scale, 600-second ALNS search budget.
- Heterogeneous volumes: task volume uniformly drawn from `{1,2,3}` GB, 10 seeds at 100/500/1000 tasks, 600-second ALNS budget.
- Pricing ablation: A* versus Dijkstra and filter versus no filter on the 24-hour horizon.

Read the matching protocol JSON in `docs/` before running a campaign. Existing records and figure summaries are intentionally kept separate by protocol and source fingerprint.

## Environment

Python dependencies are listed in `requirements.txt`. The experiments use Gurobi 13.0 and require a valid user-provided Gurobi license. No license file or credentials are included. The optional native backend is platform-specific; the Python shortest-path implementation remains available as a fallback.

```bash
python -m pip install -r requirements.txt
```

The campaign scripts write results to their selected output directories. Use a new output directory when testing changes so that frozen records are not overwritten.

## Data and reproducibility

TLE inputs, protocol files, instance descriptions, and the frozen graph/task inputs are included here. See [instances/README.md](instances/README.md) for the data layout and loading examples. The frozen unit-volume suites contain 100 three-hour cases and 30 full-day cases. Archived descriptions and graph exports retain their own historical settings.

```bash
python scripts/verify_instances.py
```

This verifies file checksums, compressed-file integrity, and the frozen-instance manifests without running an optimization experiment. Raw solve records, audit records, and figure-data exports remain in the accompanying experimental archive. Each campaign record carries its source fingerprint and instance hash; use the corresponding input and protocol.

## Citation

Please cite the associated manuscript:

> LIN Xiang, Yuning Chen, Baichuan Wang, Qinghua Wu, Yang Wang, Yingwu Chen. *A Price-and-Branch Method for Large-Scale Collaborative Task Scheduling in Low-Earth-Orbit Hybrid Constellations*. European Journal of Operational Research, manuscript EJOR-D-26-02206.
