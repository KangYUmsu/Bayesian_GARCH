# Experiment log

Machine: Apple M3, 8 cores; Python 3 (anaconda), numpy 1.26.4, scipy 1.12.0, pandas 2.2.2,
numba 0.59.1, arviz 0.21.0. All times are wall-clock.

## 2026-11-05 — Primary investigation

| # | Step | Command | Result | Time |
|---|---|---|---|---|
| P0 | Existing test suite | `python -m pytest -q` | 47 passed | 3 s |
| P1 | Read every module in `src/`, the README, milestone notes, and all saved M10–M14 outputs | — | See audit section of report | — |
| P2 | Bit-reproduction of first SPY-500 refit block with the saved M10 output | inline | MLE/MAP agree to 1e-12; Bayesian quantities differ 0.5–1.7% (sampler start points differ across platforms/library builds) | 1 s |
| P3 | Wrote `code/walkforward.py`, a mirror of `run_forecasts` that also saves draws, a Laplace (classical) uncertainty analogue, Student-t GARCH MLE, and 1%/5% VaR. Verified against `src` on this machine: all variance columns agree to ≤1e-12 | inline check | equivalent | 1 s |
| P4 | Full grid `code/run_grid.py`: 8 configs × {2023 seg seeds 2026, 11, 12; 2017–22 seg seeds 102026, 100011; broad prior both segs} = 56 jobs, 1,088 refits | `python agent_research/code/run_grid.py` | 28 failed strict diagnostics (moderate 9/768, broad 19/320); see `results/run_grid.log` | 135 s |
| P5 | Audit checks `code/audit_checks.py` | `python agent_research/code/audit_checks.py` | likelihood OK; MCMC vs importance sampling ≤0.3%; no lookahead; cash lag OK; M10 replay exact; 25 MLE refits on the logit-p=7 box bound | 73 s |
| P6 | Primary analysis `code/analysis_primary.py` | `python agent_research/code/analysis_primary.py` | `results/primary/*.csv` | 23 s |
| P7 | Figures `code/make_figures_primary.py` | | `figures/fig1–4` | 5 s |

Decisions and deviations from the plan:
- Seeds for the new 2017–22 segment were offset (102026) so its MCMC random streams are not identical to the 2023 segment's.
- Convergence failures: the project's strict rule would halt. I keep all days in the primary analysis and
  rerun the forecast tables on converged-fit days only (`forecast_*_converged_only.csv`) as a sensitivity check; conclusions unchanged.
- Bug in my own first draft: the bootstrap used Python's salted `hash()` for seeds (not reproducible). Replaced with CRC32 before
  any result was recorded in the report.
- Added (not pre-registered): `boundary_decomposition()` splitting the Bayesian forecast gains by whether the MLE
  was on its box bound. Exploratory.

## 2026-11-05 — Secondary task: strategy discovery (protocol fixed BEFORE any candidate was evaluated)

Budget: ≤3 strategies, ≤5 configurations each, reuse saved draws only, ≤30 min runtime.

Data: development = 2017-01-03…2020-12-31, validation = 2021-01-04…2022-12-30 (both from the
`ext` walk-forward), holdout = 2023-01-03…2026-11-04 (`test` walk-forward, seed 2026).
Each period is backtested as a separate portfolio that starts in cash.

Candidates (all built on the posterior-integrated variance `m_t`, posterior q90 `q_t`, draws `v_t^(i)`):
- **S1 uncertainty-z scaling** — `w = min(1, 0.10/sqrt(252 m_t) · exp(−λ z_t))`, `z_t` = premium
  `q_t/m_t − 1` standardized by its trailing-252-day mean/sd (past rows only), clipped to ±3.
  λ ∈ {−0.25, 0.25, 0.5}.
- **S2 predictive downside-risk budget** — `w = min(1, B_α / ES_α^mix,t)`, ES of the posterior
  Gaussian scale mixture, budget `B_α` = Gaussian ES_α at 10% annual vol. α ∈ {0.01, 0.025}.
- **S3 transition-conditional conservatism** — use `q_t` (conservative) when volatility is rising,
  `m_t` otherwise; rising := `m_t > (1+δ) m_{t−5}`. δ ∈ {0, 0.10}, plus a "falling" variant (δ = 0).
Total: 8 configurations.

Controls: `bayes_integrated`, `bayes_upper90`, `mle`, `mle_const_rt`, `ewma`, `static_target`, and for
every candidate an **implementable exposure control** `ctrl = min(1, k_t · w_integrated)` with `k_t` the
trailing-252 mean (past rows only) of `w_cand / w_integrated`. Costs 5 bp (20 bp sensitivity).

Selection rule (development): per strategy, keep the one configuration with the highest pooled
(4-asset, both windows averaged) ΔSharpe vs its implementable control.
Validation pass criterion (all required): pooled ΔSharpe vs control ≥ +0.05 in **both** windows,
positive in ≥ 6/8 asset-window configs, turnover ≤ 1.5× `bayes_integrated`.
Stop rule: if no strategy passes validation, the holdout is **not** evaluated for strategy selection
purposes and the search stops.

### Discovery runs

| Stage | Command | Outcome | Time |
|---|---|---|---|
| dev | `python agent_research/code/strategy_discovery.py dev` | selected S1 λ=0.25, S2 α=0.025, S3 δ=0.10 (`results/discovery/dev_selection.json`) | 8.2 s |
| val | `python agent_research/code/strategy_discovery.py val` | `passed: []` → stop rule triggered; holdout not evaluated | 2.1 s |
| S1 cost check (not a selection step) | inline `run_period(..., cost_bps=20)` | S1 vs control ΔSharpe −0.19/−0.16 (dev), −0.12/−0.12 (val) | ~3 s |

Total discovery compute: ≈15 s of the 30-minute budget. No new MCMC.

## Report verification
Claims in `final_report.md` were rechecked against the CSVs; two corrections made before finalizing
(QLIKE effect expressed in units rather than a wrong percentage; boundary-refit counts by asset).
