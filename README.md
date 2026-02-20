# Energy Gap Experiments (Moons + Breast Cancer)

This repository contains two experiments measuring pairwise energy gaps for a one-hidden-layer ReLU network, plus the DSS implementation.
The CSV outputs are intended for citation in the paper.

## Associated publication
This repository accompanies the preprint:

Saveliy Baturin (2026). *Asymptotic Smoothing of the Lipschitz Loss Landscape in Overparameterized One-Hidden-Layer ReLU Networks*. arXiv:2602.17596. https://arxiv.org/abs/2602.17596

Archived release record:

Zenodo DOI: https://doi.org/10.5281/zenodo.18607965

## Method at a glance
- Train multiple independent models of the same width.
- For each pair, set the energy level:
  `E = max(L(theta_A), L(theta_B))`.
- Run DSS to connect the pair and measure the energy gap:
  `max_t L(gamma(t)) - E`.

## What's inside
- `dss.py`: DSS implementation and supporting utilities.
- Moons experiment (regression, MSE):
  - `0_energy_gap_experiment.ipynb`
  - `energy_gap_moons_percentiles.csv`
- Breast Cancer experiment (classification, BCE):
  - `1_energy_gap_cancer_experiment.ipynb`
  - `energy_gap_cancer_percentiles.csv`

## Experiment parameters
Key parameters are set inside each experiment:
- `widths`, `num_models`, `num_pairs`, `max_epochs_*`
- `threshold_mode`:
  - `"pairwise"`: the main mode used in the paper
  - `"percentile"`: alternative mode with percentile sublevel sets
- `normalize_first_layer_weights`: optional first-layer normalization

## Outputs
The CSV files contain:
- `gap_mean`, `gap_median`, `gap_max`, `hit_rate`, `n_pairs`
- width and mode metadata

## How to cite
If you use this code or generated outputs, please cite:

```bibtex
@misc{baturin2026asymptoticsmoothinglipschitzloss,
  title = {Asymptotic Smoothing of the Lipschitz Loss Landscape in Overparameterized One-Hidden-Layer ReLU Networks},
  author = {Baturin, Saveliy},
  year = {2026},
  eprint = {2602.17596},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2602.17596}
}
```

## Reproducibility note
For exact reproducibility, use the git commit/tag corresponding to the arXiv preprint version and the Zenodo archive DOI above.

## Notes
- RNG seeds are fixed in code.
- In `threshold_mode="pairwise"` percentiles are not used.
- `hit_rate` is the fraction of runs where DSS hits `max_depth`.
