# Reproducibility archive manifest

This archive accompanies the second-audit revision dated 13 August 2026.

## Canonical code

- `theory_experiments/`: NumPy implementation, Torch training backend,
  theorem-aligned paths, validation, and analysis modules.
- `tests/`: twelve invariant and backend tests, run with
  `python -m unittest discover -s tests -v`.
- `configs/`: frozen Huber and binary-cross-entropy configurations.
- `dss.py`: corrected legacy DSS implementation retained for provenance.

## Frozen results

- `results/reviewer_main_gpu_n1`, `n2`, `n4`: primary Huber run.
- `results/reviewer_cross_entropy_gpu_n1`, `n2`, `n4`: matched loss-robustness
  run.
- `results/reviewer_main_analysis`: validated main-run summaries.
- `results/reviewer_loss_robustness`: validated cross-loss summary.
- `results/dense_endpoint_stress`: 720-record active merge stress test.

Every dimension-split run includes its metadata, tabular records, and
`states_and_paths.npz` array archive. The manuscript figures and tables are in
the separate submission archive. Historical pre-fix CSV files are intentionally
excluded.
