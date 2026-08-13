# Proof re-audit after the second revision

Status labels are deliberately strict: `THM` means established in
`article.tex`; `OBS` means a finite computation; `OPEN` means not proved.

## Normalized statements

1. `THM` Finite width: for every pair in `Omega_m(lambda)`, every
   `1 <= l <= m`, and every `r >= 0`, a continuous admissible path has the
   explicit loss-consistent compression/bridge upper bound.
2. `THM` Fixed level: for `n >= 2` and fixed `lambda > e_infinity`, the
   uniform thickening is `O(m^(-1/(n-1)))`; for `n=1` it is zero for every
   `m>=4`.
3. `THM` Objective approximation: `e(l)-e_infinity = O(l^(-1/2))` under the
   standing hypotheses, with no target-class assumption.
4. `THM` Moving level: for bounded `lambda_m >= e(m)`, thickening tends to
   zero; if `e(l)-e_infinity=O(l^(-s))`, the rate for `n>=2` is
   `O(m^(-s/((n-1)s+1)))`.
5. `THM` Standing-assumption specialization: the preceding sampling rate gives
   `O(m^(-1/(n+1)))` for `n>=2`; for `n=1` thickening is zero for every `m>=4`.

## Dependency DAG

```text
standing hypotheses
  |-- lower risk bound + kappa > 0
  |      `-- sublevel l1 control
  |              |-- compactness/attainment
  |              `-- uniform near-minimizer variation bound
  |                         `-- Maurey sampling O(l^(-1/2))
  |
  |-- finite second moment
  |      |-- ReLU perturbation bound
  |      `-- sampling variance bound
  |
  `-- convex globally L-Lipschitz loss
         |-- fixed-W output interpolation
         |-- finite-width Lipschitz bridge
         `-- sampled-predictor risk transfer

closed first-layer ball
  |-- compactness and convex interpolation
  `-- monotone sphericalization path
           |-- active rows on S^(n-1)
           |      `-- (n-1)-dimensional cover + signed merge
           |              `-- fixed-level O(m^(-1/(n-1)))
           |              `-- approximation/compression trade-off
           |                       `-- moving-level rates
           `-- n=1 two-ray exact compression
                  `-- exact connectivity for m>=4
```

## First-failure audit and repairs

### 1. Spherical normalization could have repeated the old invalid WLOG step

Resolved. The paper does not identify arbitrary ball parameters with sphere
parameters reversibly. For an active row of radius `rho in (0,1)`, it gives the
explicit path `w(t)=c(t)w`, `theta(t)=theta/c(t)`, with
`1 <= c(t) <= 1/rho`. The predictor is constant, the row remains in the ball,
and the output penalty decreases. A zero row with a nonzero coefficient is
handled separately by deleting its coefficient at constant predictor.

### 2. The sphere covering exponent could have been cited without enough detail

Resolved. The proof covers the sphere by `2n` signed coordinate graph patches,
uses the lower bound `|u_j| >= 1/sqrt(n)` to obtain a patchwise Lipschitz graph,
and partitions the `(n-1)` free coordinates into cubes. This yields at most
`N` sets of diameter `C_n N^(-1/(n-1))`, including a separate constant
adjustment for `N<2n`.

### 3. The covering must cluster active rows, not zero-output rows

Resolved. `s` is the active support. With `k=s-(m-l)` and
`N=floor(s/(k+1))`, assigning each active row to one cover cell gives the
contradiction `s <= kN <= ks/(k+1) < s` unless a cell contains at least `k+1`
active rows.

### 4. Signed merging might increase the penalty

Resolved. The representative receives the signed sum of the cluster; the
triangle inequality gives nonincrease of the output `l1` norm. The prediction
error is expressed as a signed sum of atom differences from the representative.

### 5. The endpoint sphericalization path must be included in the final path

Resolved. Each endpoint is first connected to its sphericalized endpoint by a
nonincreasing path. The finite-width construction joins those sphericalized
points. The reverse of the second endpoint path completes the connection; its
objective also stays below the original endpoint level because it traverses the
same nonincreasing path backwards.

### 6. The Maurey estimate must control both risk and regularization

Resolved. A near-minimizer has uniformly bounded output `l1` mass `a`. Sampling
atoms with probabilities `|theta_i|/a` and coefficients
`(a/l) sign(theta_i)` keeps total output `l1` mass at most `a` for every
realization. Independence gives mean squared prediction error at most
`a^2 E||X||^2/l`; Cauchy--Schwarz and loss Lipschitzness transfer it to
expected risk. One realization is no worse than the sampling expectation.

### 7. Near-minimizers of arbitrary width need a width-independent mass bound

Resolved. The zero predictor gives the finite uniform upper bound
`B0=F(0)`. Choosing an epsilon-near minimizer of `e_infinity` and using the
risk lower bound gives `||theta||_1 <= (B0-lower_R+1)/kappa`, independent of
its width.

### 8. The moving-level trade-off must use the right comparison with lambda_m

Resolved. Since `lambda_m >= e(m) >= e_infinity`, the reference contribution
above the selected level satisfies
`(e(l)-lambda_m)_+ <= e(l)-e_infinity`. The compression contribution is uniform
because `(lambda_m)` is bounded. Choosing
`l=floor(m^(1/((n-1)s+1)))` balances the two terms for `n>=2`.

### 9. The degenerate dictionary dimension n=1 cannot use exponent 1/(n-1)

Resolved separately. Sphericalization places all active rows in
`S^0={-1,+1}`. Coefficients at each direction merge exactly and continuously,
leaving at most two active neurons. Thus `e(l)=e(2)=e_infinity` for `l>=2`, and
The finite-width theorem with reference width two gives zero thickening for `m>=4` at every
bounded level `lambda_m >= e(m)`.

### 10. “Certified DSS” needed a manuscript-level certificate

Resolved. The article now gives an explicit segment Lipschitz constant and the
grid correction `K_gamma/[2(G-1)]`, then takes the maximum over the returned
piecewise-linear segments. This certifies the reported upper value of that
returned path, not optimality among paths.

## Experimental status

- `OBS`: 900 Huber pairs and 900 BCE pairs are balanced, with disjoint pairing
  inside each replicate--width--level group, not globally disjoint endpoints.
- `OBS`: the dense-representation stress test has 720 records; every coordinate
  is active before merging, at least four coefficients are removed, all exact
  diameter certificates hold, and the maximum increment/certificate ratio is
  `0.0395906`.
- These observations do not prove either covering exponent or a uniform
  sublevel statement.

## Remaining OPEN extensions

- Faster target-specific rates for `e(l)-e_infinity` under assumptions matched
  to this exact bias-free regularized dictionary.
- Entropy improvements below dimension `n-1` under a uniform structural
  hypothesis on all endpoints of a sublevel.
- Non-globally-Lipschitz losses under separately established moment and
  pathwise-logit control.

No `OPEN` obligation remains in the statements currently labeled `THM`.
