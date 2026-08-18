# Dimension-free barrier decay: v2

This directory contains the current JOTA revision of the manuscript and its
theorem-aligned computational evidence. The previous `jamc_article` and the
historical experiment code remain available for provenance.

## Contents

- `article/`: JOTA `svjour3` source, generated tables/figures, the built main
  PDF, and Online Resource 1.
- `experiments/`: reusable constrained-network code, the fixed-dictionary
  sampling diagnostic, plotting code, and frozen JSON configurations.
- `results/self_sparsification_gpu/`: the frozen aggregate GPU records used by
  the new figure and table.
- `tests/`: finite proof-obligation and implementation-invariant checks.

## Main revision

The v2 proof replaces the dimension-dependent spherical covering rate by a
zero-radius, fixed-dictionary compression.  For convex Lipschitz losses it
gives an `O(m^-1/2)` moving-level barrier, while smallest-mass pruning gives
`O(m^-1)` at every fixed level above the limiting objective.  If the loss
derivative is alpha-Hölder, the moving-level rate becomes
`O(m^{-(1+alpha)/2})`; smooth Huber, half-squared error, and binary cross-entropy therefore
give `O(m^-1)`.  A balancing deformation transfers the result to standard
bias-free two-layer quadratic weight decay.

## Reproduce

From the repository root, with the dependencies in `requirements.txt`:

```powershell
python -m unittest discover -s v2/tests -v
python -m v2.experiments.run_self_sparsification `
  --config v2/experiments/configs/self_sparsification_gpu.json `
  --output-dir v2/results/self_sparsification_gpu
python -m v2.experiments.analyze_self_sparsification `
  --result-dir v2/results/self_sparsification_gpu `
  --article-dir v2/article
Set-Location v2/article
latexmk -pdf -interaction=nonstopmode -halt-on-error article.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_reproducibility_notes.tex
```

The frozen main diagnostic was run with PyTorch 2.8.0/CUDA 12.9 on an NVIDIA
GeForce RTX 5070 Ti.  It contains 192 endpoint/support records and 2048 atom
samplings per record.  Credentials and host-specific launch helpers are not
stored in the repository.

## Evidence discipline

The analytic bounds are theorem statements.  The GPU run is an observation
on seeded finite distributions that checks sampling invariants, the exact half-squared-loss
conditional expectation, and the balancing lift.  It neither estimates the
supremum over a full sublevel nor supplies a barrier lower bound.
