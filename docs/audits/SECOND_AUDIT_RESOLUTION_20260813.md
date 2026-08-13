# Resolution of the second external audit

## Implemented

- Corrected "900 disjoint pairs" to 900 pairs with disjoint pairing inside
  each replicate--width--level group.
- Proved monotone sphericalization and replaced the ambient exponent `n` by
  the active dictionary dimension `n-1` for `n>=2`.
- Added the separate `n=1` two-ray theorem and made the exact threshold
  `m>=4` explicit for fixed and moving levels.
- Added a qualitative moving-level corollary without an assumed rate.
- Proved the model-internal approximation-value rate
  `e(l)-e_infinity=O(l^(-1/2))`, yielding the near-optimal barrier rate
  `O(m^(-1/(n+1)))` under the standing assumptions for `n>=2`.
- Isolated the spherical metric-entropy estimate as a separate lemma and made
  the zero-output case in the Maurey argument explicit.
- Added an explicit manuscript-level DSS segment certificate.
- Added and checked Nurisso--Leroy--Vaccarino (NeurIPS 2024), their ICLR 2026
  continuation with Petri, and Wu--Simsek--Ged (ICLR 2025), plus recent
  Constructive Approximation positioning.
- Removed the unreported breast-cancer experiment from Data Availability.
- Added a 720-endpoint dense-representation stress test that forces nontrivial
  merging and verifies every individual diameter certificate.

## Verified external-state findings

- The public arXiv record is still only v1 from 19 February 2026 and contains
  the superseded title, Moons/breast-cancer experiments, and `p_perm=0` claim.
- The four-commit legacy default branch was identified as stale during the
  audit. The curated revision commit replaces that public snapshot with the
  canonical code, tests, configurations, article sources, and frozen artifacts.

The repository finding is resolved by the curated public revision, with the
historical notebooks and DSS helper isolated under `legacy/`. An arXiv v2
source package must still be uploaded by the author. The Code Availability
statement now identifies the exact repository contents rather than the legacy
development snapshot.

## Not added as a claim

- No fitted experimental exponent is claimed. The dense stress plot validates
  the merge mechanism and finite inequalities only.
- No target-class approximation theorem from the literature is imported
  without matching its dictionary, loss, distribution, and regularizer.
