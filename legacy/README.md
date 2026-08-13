# Legacy experiment

This directory preserves the pre-audit notebook pipeline for provenance only.
It is not used by the manuscript figures, tables, or numerical conclusions.

- `0_energy_gap_experiment.ipynb`: historical Two Moons/MSE study.
- `1_energy_gap_cancer_experiment.ipynb`: historical breast-cancer study.
- `dss.py`: corrected standalone DSS helper used by those notebooks.
- `configs/`: superseded pilot, unsplit, and GPU-smoke configurations retained
  to document development history; they did not generate the frozen results.

The notebooks do not instantiate the theorem's model: they include biases,
normalize first-layer rows rather than projecting to the closed unit ball, and
the Two Moons notebook uses a loss that is not globally Lipschitz in the logit.
Their original percentile CSV outputs also predate the correction from the
minimum to the maximum terminal-segment exceedance. Those invalid tables are
excluded from the public tree and remain available only in Git history.

Use `theory_experiments/` for the canonical journal experiment and
`results/` for the frozen evidentiary artifacts.
