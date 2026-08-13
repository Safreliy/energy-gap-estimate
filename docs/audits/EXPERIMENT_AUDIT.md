# Code and experiment audit for the revised manuscript

Date: 2026-08-12

## Audit question

Does the computational study measure finite-sample analogues of the quantities
in the manuscript without presenting a path-finding heuristic as a proof of a
uniform topological statement?

## High-priority findings in the legacy pipeline

1. **[Critical, corrected in `legacy/dss.py`] Wrong aggregation of a returned path.**
   The old DSS output used the smallest unresolved segment exceedance. A
   piecewise path is controlled by the maximum over all of its segments.
   Historical CSV files were generated before the fix and remain invalid as
   evidence.

2. **[Critical] Model mismatch.** `SimpleNN` and
   `SimpleNNClassification` contain first- and second-layer biases. The theorem
   uses the bias-free model `sum_i theta_i ReLU(w_i^T x)` and penalizes exactly
   `theta`. The legacy code also renormalizes rows to the sphere, while the
   revised theorem constrains them to the closed ball.

3. **[Critical] The principal Moons/MSE experiment violates the standing loss
   hypothesis.** Mean squared error is not globally Lipschitz in the prediction.
   It cannot be advertised as a direct empirical instance of the theorem.

4. **[High] Only two widths were compared.** A two-sample Mann--Whitney test does
   not test an asymptotic rate. At least five log-spaced widths are required for
   a width-scaling diagnostic.

5. **[High] Pair pseudoreplication.** Random pairs repeatedly reuse the same
   trained endpoints, but pair rows were analyzed as independent observations.
   This can make uncertainty intervals and p-values far too optimistic.

6. **[High] The path maximum was only sampled.** Even after correcting the
   segment aggregation, a grid maximum can miss an interior peak. The old
   output contains no deterministic bound on this discretization error.

7. **[High] DSS success and failure were interpreted asymmetrically.** A returned
   low-loss path is an upper bound on the infimum over paths. Hitting the depth
   limit is not evidence of a positive infimal barrier. Accordingly,
   `hit_rate` is an algorithmic diagnostic, not a landscape statistic.

8. **[High] The empirical maximum over sampled pairs is not the uniform sublevel
   thickening.** It should be called a sampled-endpoint diagnostic, never an
   estimator of exact `SubBar_m(lambda)` without an endpoint-net argument.

9. **[Medium] Training and evaluation loaders could be shuffled, and batch
   averages were averaged without batch-size weights.** This is harmless for the
   old full-batch settings but makes the helper wrong for unequal final batches.

10. **[Medium] Reproducibility state was incomplete.** Global module seeds,
    lambda worker initializers, checkpoint deserialization failures under newer
    PyTorch, and missing environment/commit metadata make exact reruns fragile.

11. **[Medium] Extreme-value inference was not stable.** Bootstrapping a maximum
    of dependent pair observations and permuting pair rows does not test the
    theorem's uniform maximum. Independent repeats must be resampled as blocks.

12. **[Medium] Breast Cancer is poorly matched to the geometric rate.** Its
    ambient dimension is 30, so the worst-case exponent is about `1/30`; no
    feasible width range can visibly resolve it. It is a secondary stress test,
    not the principal validation.

## Replacement design

The canonical pipeline is now `theory_experiments/`.

- Exact theorem model: bias-free one-hidden-layer scalar ReLU network.
- Exact constraint: every first-layer row lies in the closed unit ball.
- Exact regularizer: `kappa * ||theta||_1`; no output bias is present.
- Globally Lipschitz Huber loss for the synthetic regression benchmark;
  logistic loss is also implemented.
- Fixed finite empirical distribution, so the computational objective is an
  exact expectation over its atoms.
- Primary dimensions `n = 1, 2, 4` and five log-spaced widths.
- Independent unit: a complete pool replicate. Pairs are disjoint within each
  replicate; uncertainty is obtained by block-resampling replicates.

Three path controls are saved:

1. direct linear interpolation;
2. corrected DSS with a global-Lipschitz upper certificate between grid points;
3. a proof-inspired construction that explicitly merges a neuron cluster,
   frees coordinates, inserts a common narrow reference, and connects both
   endpoints through it.

The proof-inspired path has an analytic continuous-path bound: each segment is
convex output interpolation or a constant-objective parameter motion, so its
largest node objective controls the full continuous path.

## Reviewer-facing metrics

- `objective_min`: best found value, labelled an optimization upper proxy and
  never claimed equal to `e(m)`.
- Fixed-level and moving-level sampled-pair path upper gaps.
- Maximum DSS certificate width.
- Direct/DSS/proof-path algorithm ablation.
- Constructive compression objective increase and analytic Lemma 5.1 increment.
- Active support, output L1 norm, and maximum first-layer row norm.
- Descriptive log--log slopes with replicate-level block bootstrap intervals.

## What remains unproved by the experiment

- optimizer convergence to `e(m)`;
- coverage of every endpoint in a continuous sublevel;
- a lower bound on the infimum over all paths;
- an asymptotic exponent from a finite range of widths;
- transfer from the finite empirical distribution to population risk.

These limitations are part of the reporting contract in `metadata.json`.

## Negative controls and counterexamples to overinterpretation

- **One-dimensional saturation (explicit counterexample to generic rate
  fitting).** Without biases, every one-dimensional atom is a nonnegative
  multiple of either `ReLU(x)` or `ReLU(-x)`. Width beyond two adds no new
  predictor directions. Exact-zero barriers and an unidentifiable slope are
  therefore expected; `n=1` is a saturation control, not evidence for a fitted
  asymptotic exponent.
- **Sublevel slack.** If a chosen level lies above every constructed path,
  all recorded gaps are exactly zero. Replacing those zeros by an arbitrary
  plotting floor can manufacture any apparent log--log slope. The analysis now
  refuses to fit a slope when at least half of summaries are zero.
- **Finite endpoint sampling.** A low maximum over sampled trained pairs does not
  rule out an unsampled difficult endpoint. The repaired statement is only that
  the recorded paths give upper bounds for those recorded pairs.
