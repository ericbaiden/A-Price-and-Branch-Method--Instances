# Reproducibility notes

The public repository includes the implementation, protocol material, instance descriptions, and 130 frozen graph/task inputs in `dataset/instances/`. The accompanying experimental archive contains the larger raw campaign-result collections. See `instances/README.md` for the input formats, counts, checksums, and loading examples.

## Campaign separation

- `alns_m600_final`: three-hour unit-volume P&B/GRASP/ALNS comparison.
- `alns_h24_600s`: 24-hour unit-volume comparison with a 600-second ALNS search budget.
- `heterogeneous_m600_600s`: volume draws from `{1,2,3}` GB with the recorded 600-second budget.
- `revision_audit_m600`: independent full-task pricing certification and reduced validation.
- `pb_ablation_h24`: independent pricing-ablation campaign.

The timing campaigns are kept separate because setup, certification, and search-phase accounting differ. Total pricing-call fields in the audit records already include certification calls.

## Numerical scope

The unit-volume suites report certified root bounds after a full pricing pass. The heterogeneous suite is reported as near-optimal and is not generalized to a universal exactness claim. A branch-and-price reference is included for closed-tree verification; time-limited trees remain explicitly unclosed.

## Entry points

The top-level runners are intentionally kept separate by campaign. The main three-hour wrapper is `run_campaign_from_dataset.py`; the 24-hour driver is `rerun_alns_h24.py` with `LEO_HORIZON_HOURS=24`; the heterogeneous driver is `run_heterogeneous_m600.py`; and the pricing ablation driver is `run_pb_ablation_h24.py`. Use a fresh output directory for local checks.
