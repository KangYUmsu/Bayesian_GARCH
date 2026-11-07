# Research plan — independent assessment of the Bayesian GARCH project

Written 2026-11-05 **before** running any new experiment. Changes made later are
recorded in `experiment_log.md` with the reason.

## Question

Does Bayesian *parameter* uncertainty in a Gaussian GARCH(1,1) add value for
(i) volatility forecasting, (ii) tail-risk forecasting and (iii) a one-asset
volatility-targeting allocation, beyond
- classical MLE plug-in,
- Bayesian point estimates (MAP, posterior-mean plug-in),
- a classical (asymptotic-normal / Laplace) parameter-uncertainty analogue, and
- simple conservative exposure reduction (a constant scaling of a point-estimate rule)?

Negative results are acceptable and will be reported as such.

## What already exists (audit inputs)

- `src/` implements Gaussian zero-mean GARCH(1,1), multi-start MLE, MAP in the
  transformed coordinates, adaptive random-walk Metropolis with strict ArviZ
  diagnostics, walk-forward forecasting (63-day refits, 500/1000-day windows),
  one-asset/cash portfolio simulation with DFF cash and 5 bp costs.
- Milestones 10–14 evaluate SPY, QQQ, TLT, GLD on **2023-01-03 → 2026-11-04**.
  That window has therefore been inspected repeatedly by the project; it is
  *not* a pristine holdout for any question the project already asked.

## Data splits

| Segment | Dates | Status |
|---|---|---|
| Training history | 2013-01-02 → 2016-12-30 | used only for the first fits |
| **Extended OOS (new)** | 2017-01-03 → 2022-12-30 | never used for evaluation by the project; used here as a *pre-registered replication* of the project's claims with frozen specifications |
| Project test | 2023-01-03 → 2026-11-04 | already inspected by Milestones 10–14; reproduced here |

For the optional strategy-discovery task (decided now, before any new strategy is
evaluated):
- development: 2017-01-03 → 2020-12-31
- validation: 2021-01-04 → 2022-12-30
- final holdout: 2023-01-03 → 2026-11-04 — never used for tuning or selection
  of *new* strategies; inspected exactly once, after candidates are frozen.
  It is not pristine with respect to the baseline strategies (seen by the
  project), and the report will say so.

## Frozen specifications (no tuning)

Exactly the project's Milestone 10 settings: moderate prior, 4 chains,
2,500 warm-up + 4,000 draws, 300 retained posterior draws per refit, 63-day
refits, windows 500 and 1000, 10% annual volatility target, weights in
[0, 1], 5 bp one-way costs, DFF cash with one-day lag, one-close execution delay.

## Pre-registered hypotheses and tests

- **H1 (variance forecasts).** Posterior-integrated variance has lower QLIKE than
  MLE plug-in. Diebold–Mariano test with Newey–West (HAC) standard errors per
  configuration; pooled across assets.
- **H2 (predictive density).** Posterior predictive (Gaussian scale mixture) log
  score exceeds MLE-Gaussian plug-in log score. DM/HAC.
- **H3 (tail risk).** Posterior-mixture 1% and 5% VaR has better unconditional
  (Kupiec) and independence (Christoffersen) coverage and lower quantile loss
  than plug-in VaR. A Student-t GARCH MLE is included to size the effect of
  *innovation misspecification* relative to *parameter uncertainty*.
- **H4 (information content).** The relative posterior dispersion
  `premium = q90/mean − 1` predicts standardized forecast errors / large shocks
  beyond the variance level (HAC regressions).
- **H5 (allocation).** `bayes_upper90` improves Sharpe / drawdown / downside
  risk vs (a) MLE, (b) Bayes mean plug-in, (c) a classical Laplace-upper90 rule,
  (d) **implementable** exposure-matched point-estimate rules whose scaling is
  calibrated only on earlier data, and (e) ex-post exposure- and vol-matched
  controls (diagnostic only). Paired circular block bootstrap for Sharpe and
  drawdown differences; HAC for mean daily return differences; cost
  sensitivity at 0 / 5 / 20 bp.

Multiplicity: 8 configurations (4 assets × 2 windows; windows of the same asset
are strongly dependent). Conclusions rely on pooled/sign-consistency evidence and
Holm-adjusted p-values, not on the best single configuration.

## Audit checklist

1. Likelihood: independent re-implementation check.
2. MLE: boundary solutions, optimizer reliability, effect of the bound.
3. MCMC: posterior check against an independent importance-sampling estimate for
   selected refits; Monte Carlo (seed) variability of Bayesian forecasts and
   portfolio metrics (re-run test period with 2 extra seeds).
4. Lookahead: forecast/execution alignment, cash-rate lag, static benchmark
   calibration windows, EWMA initialization.
5. Accounting: replay the saved equity curves.
6. Reproduction: rerun the full Milestone 10 grid and compare with saved outputs.

## Compute budget

Primary investigation: walk-forward refits cost ≈1.2 s each; the full plan is
≈ 8 configs × (≈24 extended + 16 test) refits × 3 seeds ≲ 30 min on 8 cores.
Strategy discovery: ≤ 30 min additional, reusing saved posterior draws only.
