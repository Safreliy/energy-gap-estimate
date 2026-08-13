# Reviewer-oriented experimental protocol

## Driving question

For the same constrained shallow ReLU objective as in the theorem, do wider
networks show (i) cheaper proof-inspired compression and (ii) smaller returned
path upper gaps at fixed and moving empirical sublevels?

The fixed level is common across widths and mirrors the fixed-threshold
theorem. The moving level follows a width-dependent empirical optimization
proxy and mirrors, without proving, the approximation-transfer regime.

## Run order

Use Python 3.11 or newer.

```powershell
python -m pip install -r requirements-experiment.txt
python -m unittest discover -s tests -v
python -m theory_experiments.run_experiment `
  --config configs/reviewer_smoke.json `
  --output results/reviewer_smoke
python -m theory_experiments.plot_results --results results/reviewer_smoke
```

The smoke configuration is a software check only. Its slopes and plots must not
be reported in the paper. The frozen Huber computation uses the three
dimension-split `configs/reviewer_main_gpu_n*.json` configurations. Run each
one with an output directory having the same stem; for example,

```powershell
python -m theory_experiments.run_experiment `
  --config configs/reviewer_main_gpu_n2.json `
  --output results/reviewer_main_gpu_n2
```

After all three dimensions complete, validate the balanced design and generate
the manuscript figures and table with:

```powershell
python -m theory_experiments.analyze_main_results `
  --results-root results `
  --analysis-dir results/reviewer_main_analysis `
  --article-dir jamc_article
```

For the pre-specified binary-cross-entropy robustness rerun, use the three
dimension-split `configs/reviewer_cross_entropy_gpu_n*.json` files. After all
three complete, validate and compare both losses with:

```powershell
python -m theory_experiments.analyze_loss_robustness `
  --results-root results `
  --analysis-dir results/reviewer_loss_robustness `
  --article-dir jamc_article
```

## Primary analysis

1. **Fixed level.** Use one common empirical level across widths and replicates.
2. **Moving level.** Use width-specific balanced empirical sublevels with a
   pre-specified vanishing slack. Report the selected bridge width.
3. **Compression mechanism.** Report the positive objective increment from
   deterministic cube-cell merging and its explicit ambient-cube increment.
4. **Approximation proxy.** Plot the monotone envelope of best optimized values.
   Call it an upper proxy, not `e(m)` or a certified approximation exponent.

The statistical unit is an independent replicate, not a pair. Pair-level rows
are diagnostic; block-bootstrap slopes resample complete replicates.

## Required ablations

- Direct interpolation versus corrected DSS versus proof-inspired construction.
- Repeat the frozen design with binary cross-entropy in scalar logits. Because
  Huber and cross-entropy have different objective scales, compare both raw
  gaps and gap divided by the selected level, aggregated by replicate.
- Dimensions `n = 1, 2, 4`.
- Double final DSS grid resolution; certificate width should shrink.
- Double the number of optimizer starts for narrow reference networks.

## Optional real-data stress test

Breast Cancer may be retained in a supplement with bias-free logistic loss and
ball projection, but it is not the headline result. Report it as qualitative
robustness, with predictive test metrics separate from the training objective
used for path calculations.

## Stop rules

- Do not run the main benchmark if unit tests fail.
- If a DSS certificate width is large relative to its gap, increase resolution.
- If a sublevel contains too few disjoint pairs, report the achieved count and
  revise only through a new versioned configuration.
- Plotting floors for exact zeros are visual only; CSV files retain exact zeros.
- If at least half of replicate-width summaries are exact zero, do not fit or
  report a log--log slope. Report zero frequency and the width at which the
  metric first becomes unresolved from zero instead.

Dimension `n=1` is a pre-specified saturation control: the bias-free atom
dictionary has only the two rays `ReLU(x)` and `ReLU(-x)`. Quantitative rate
discussion should focus on `n=2,4`; the one-dimensional row checks whether the
pipeline correctly reports early exact connectivity rather than inventing a
slope.
