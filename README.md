# Theory-Aligned Loss-Landscape Experiments

This repository accompanies the revised theoretical manuscript on barrier decay
in constrained shallow ReLU networks. The canonical experiment mirrors the
bias-free model and regularized objective in the theorem. It measures
finite-distribution objective values, constructive neuron compression, and
upper bounds for returned path barriers over a sequence of widths.

Read the [`experiment audit`](docs/audits/EXPERIMENT_AUDIT.md) for the code
review and the [`frozen-run protocol`](docs/EXPERIMENT_PROTOCOL.md) for the
reproduction workflow.

## Associated publication

Saveliy Baturin (2026), *From Approximation Rates to Loss-Landscape Barrier
Decay in Shallow ReLU Networks*. Revised manuscript in `jamc_article/`.

- [Compiled manuscript](jamc_article/article.pdf)
- [LaTeX source](jamc_article/article.tex)
- [Supplementary reproducibility notes](jamc_article/supplementary_reproducibility_notes.pdf)

The earlier arXiv record is 2602.17596 and the archived development release is
https://doi.org/10.5281/zenodo.18607965. Those records predate the present audit.

## Canonical method

- Bias-free model `sum_i theta_i ReLU(w_i^T x)` with `||w_i|| <= 1`.
- Globally Lipschitz Huber or logistic loss plus `kappa ||theta||_1`.
- Regression or deterministic binary ridge-teacher targets; the binary mode
  uses cross-entropy with logits and has global logit Lipschitz constant one.
- Five or more widths and independent pool replicates.
- Disjoint endpoint pairs within each replicate.
- Fixed and moving empirical sublevels.
- Direct, corrected DSS, and proof-inspired path constructions.
- Corrected DSS is a documented modification of Freeman--Bruna Greedy DSS;
  the proof-inspired construction is a separate compression/reference path.
- Analytic continuous-path control for the proof-inspired construction and a
  Lipschitz discretization certificate for DSS.
- Dense-representation stress analysis that activates every endpoint
  coordinate and forces a certified nontrivial cluster merge.

## Repository map

- `theory_experiments/`: canonical NumPy implementation.
- `configs/reviewer_smoke.json`: fast Huber correctness run.
- `configs/reviewer_cross_entropy_smoke.json`: fast binary-cross-entropy run.
- `configs/reviewer_main_gpu_n*.json`: completed dimension-split Huber run.
- `configs/reviewer_cross_entropy_gpu_n*.json`: completed dimension-split
  binary-cross-entropy run.
- `tests/`: domain, compression, path, certificate, and optimizer invariants.
- `results/`: frozen primary, cross-loss, and dense-endpoint artifacts used by
  the manuscript analyses.
- `docs/`: frozen-run protocol, reproducibility manifest, and resolved audit
  notes.
- `jamc_article/`: revised LaTeX source, compiled paper, supplement, generated
  tables, vector figures, and resolved proof-audit notes.
- `legacy/`: historical notebooks and their corrected standalone DSS helper,
  retained for provenance but not used by the article.

Historical pre-fix `energy_gap_*_percentiles.csv` tables are not valid evidence,
are excluded from the public tree, and must not be cited.

## Quick smoke run

```powershell
python -m pip install -r requirements-experiment.txt
python -m unittest discover -s tests -v
python -m theory_experiments.run_experiment --config configs/reviewer_smoke.json --output results/reviewer_smoke
python -m theory_experiments.plot_results --results results/reviewer_smoke
```

Every run writes model, pair, compression, summary, and rate CSV files; exact
states and selected paths in compressed NPZ form; configuration, versions,
commit, and interpretation warnings in `metadata.json`; and reviewer-facing
figures.

## Frozen-result analysis

The smoke directory verifies the software path only. The manuscript figures
and numerical table are generated from the completed dimension-split main run:

```powershell
python -m theory_experiments.analyze_main_results `
  --results-root results `
  --analysis-dir results/reviewer_main_analysis `
  --article-dir jamc_article
```

The command expects `reviewer_main_gpu_n1`, `reviewer_main_gpu_n2`, and
`reviewer_main_gpu_n4` under `results/`. It validates the balanced 900-pair
design, disjoint pairing within each replicate--width--level group, the
first-layer ball constraint, and all 1200 compression inequalities before
writing submission figures or tables.

The loss-robustness analysis repeats that complete design with deterministic
binary ridge-teacher labels and binary cross-entropy in logits:

```powershell
python -m theory_experiments.analyze_loss_robustness `
  --results-root results `
  --analysis-dir results/reviewer_loss_robustness `
  --article-dir jamc_article
```

It expects `reviewer_cross_entropy_gpu_n1`, `n2`, and `n4`, applies the same
design and invariant checks to both losses, and only then writes the comparison
figure, LaTeX table, and machine-readable report.

The dense-endpoint stress test is a deterministic post-processing analysis of
the stored fixed-level Huber endpoints; it does not retrain the models:

```powershell
python -m theory_experiments.analyze_dense_endpoint_stress `
  --results-root results `
  --analysis-dir results/dense_endpoint_stress `
  --article-dir jamc_article
```

It exact-splits trained atoms until all coordinates are active, applies a
sublevel-calibrated tangent perturbation in dimensions 2 and 4, frees four
coordinates by nearest-cluster merging, and verifies the individual
diameter-based objective certificate on every record.

## Citation

The public frozen directories include their tabular records, metadata, and
`states_and_paths.npz` archives; the
[`reproducibility manifest`](docs/REPRODUCIBILITY_ARCHIVE_MANIFEST.md) lists the
exact evidentiary snapshot. The older Zenodo DOI predates this revision. Cite
the revised manuscript and the repository commit used in an analysis. Do not
cite the historical pre-fix CSV files.
