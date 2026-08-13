# Resolution of the final external review

This note records the final bounded revision after the article had already
acquired its complete theorem chain. No new empirical claims or model classes
were added.

## Mathematical and editorial repairs

- Replaced broad uses of "unconditional" in the abstract, contributions, and
  conclusion by the precise phrase "under the standing assumptions"; the
  corollary title now states the same scope.
- Isolated the entropy estimate
  `N(S^(n-1), epsilon) <= C_n epsilon^(-(n-1))` as a separate lemma before the
  cluster-merging lemma.
- Expanded the zero-output branch of the Maurey proof and retained the common
  epsilon-dependent upper bound before passing to `epsilon -> 0`.
- Strengthened the one-dimensional conclusion from eventual exactness to the
  explicit result `b_m(lambda)=0` for every `m>=4`, both at a fixed level above
  `e_infinity` and for bounded moving levels `lambda_m>=e(m)`.

## Repository repair

- The theorem-aligned implementation, tests, frozen Huber and BCE records,
  dense-endpoint records, article sources, and compiled outputs remain in the
  canonical tree.
- Historical notebooks, their corrected standalone DSS helper, and superseded
  development configurations are isolated under `legacy/`.
- Invalid pre-fix percentile CSV files remain excluded from the public tree.
- The run protocol names the exact dimension-split configurations and analysis
  commands that reproduce the journal figures and tables.

## Verification

- All 12 invariant/backend tests pass in the project Python 3.11 environment.
- The Huber, loss-robustness, and dense-endpoint analyses regenerate without
  invariant violations from the frozen artifacts.
- The article and supplementary notes compile without undefined references,
  undefined citations, errors, or overfull boxes.

## Remaining external action

The public arXiv record `2602.17596` is still v1 from 19 February 2026. It has
the superseded title, abstract, and empirical claims. Uploading v2 is an author
account action and remains the only submission-facing synchronization step not
performed in this repository.
