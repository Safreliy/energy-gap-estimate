# Dimension-Free Loss-Landscape Barrier Decay

This repository accompanies the JOTA manuscript *Dimension-Free
Loss-Landscape Barrier Decay via Atomic Self-Sparsification*. It contains the
current paper, theorem-aligned diagnostics, frozen aggregate records, and the
earlier complete-path stress pipeline retained for reproducibility.

## Current manuscript

- [JOTA manuscript PDF](v2/article/article_JOTA.pdf)
- [LaTeX source](v2/article/article.tex)
- [Online Resource 1](v2/article/ESM_1.pdf)
- [Supplement source](v2/article/supplementary_reproducibility_notes.tex)

The margin-corrected immutable submission snapshot is tagged
[`jota-submission-2026-08-18-r2`](https://github.com/Safreliy/energy-gap-estimate/tree/jota-submission-2026-08-18-r2)
and resolves to commit `4a6114cd625f528051276f04058555c5f5a4c5a7`.

The main result replaces the dimension-dependent covering step in the shallow
ReLU path construction by fixed-dictionary atomic compression. It gives
dimension-free inverse-square-root barrier decay at bounded moving levels and
inverse-linear decay at every fixed level above the limiting objective. Smooth
losses attain the inverse-linear moving-level rate as well. A balancing
deformation transfers the bounds to standard bias-free two-layer quadratic
weight decay.

The numerical results are mechanism and implementation diagnostics. They do
not estimate the worst-case population sublevel barrier and are not used as
proof of the asymptotic statements.

## Repository map

- `v2/article/`: current JOTA source, compiled paper, Online Resource, two
  vector figures, and generated tables used by the source.
- `v2/experiments/`: fixed-dictionary sampling diagnostic and shared
  theory-aligned experiment code.
- `v2/results/self_sparsification_gpu/`: frozen aggregate records for the
  primary diagnostic.
- `v2/tests/`: sampling, pruning, smooth-loss cancellation, and balancing
  proof-obligation tests.
- `theory_experiments/`, `configs/`, and `results/`: complementary complete-path
  stress pipeline and frozen Huber/cross-entropy runs.
- `docs/`: experiment protocol, reproducibility manifest, and resolved audits.
- `jamc_article/`: previous manuscript revision, retained for provenance.
- `legacy/`: historical notebooks, not used as evidence in the current paper.

The local `research_frontier/` library and submission-administration files are
intentionally excluded from the public repository.

## Quick verification

```powershell
python -m pip install -r v2/requirements.txt
python -m unittest discover -s v2/tests -v
python -m v2.experiments.run_self_sparsification `
  --config v2/experiments/configs/self_sparsification_smoke.json `
  --output-dir v2/results/self_sparsification_smoke
python -m v2.experiments.analyze_self_sparsification `
  --result-dir v2/results/self_sparsification_smoke `
  --article-dir v2/article
```

Build the paper with:

```powershell
Set-Location v2/article
latexmk -pdf -interaction=nonstopmode -halt-on-error article.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_reproducibility_notes.tex
```

The primary frozen diagnostic was run with PyTorch 2.8.0 and CUDA 12.9 on an
NVIDIA GeForce RTX 5070 Ti. It contains 192 endpoint-support records and 2048
atom samplings per record. The complete-path Huber and cross-entropy records
remain available under `results/` and are described in Online Resource 1.

## Archived identifiers

The earlier arXiv record is `2602.17596`, and the archived development release
is [Zenodo 10.5281/zenodo.18607965](https://doi.org/10.5281/zenodo.18607965).
Both predate the present fixed-dictionary revision. For reproducibility, cite
the manuscript together with the repository commit used in the analysis.
