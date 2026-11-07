# Report-to-notebook coverage checklist

> **Repository layout note.** This document was written in the development workspace and carried over unchanged
> in substance. File paths have been updated to this repository's layout (`reports/`, `notebooks/`, `scripts/`,
> `artifacts/`, `course/presentations/`); dated log entries describe work done in that workspace. Notebook section
> numbers refer to `notebooks/walkthrough.ipynb`.

This file maps every table, figure and substantive quantitative claim of the report (`reports/source/research_report.tex` →
`reports/research_report.pdf`) to the section of `walkthrough.ipynb` that reproduces it. Both documents (and the slides in
`course/presentations/` and `quick_walkthrough.ipynb`) follow the same question-driven narrative since 2026-11-06:
**question → methods → rolling design → H1 → H2 → H3 → limitations → summary.**

| Report section | Notebook section |
|---|---|
| §1 Research question and motivation | §1 (and §2 stylised facts) |
| §2.1 Gaussian GARCH and MLE | §3–§4 |
| §2.2 Bayesian Gaussian GARCH | §5–§6 |
| §2.3 Student-$t$ GARCH (MLE); §2.4 other comparators | §7 |
| §3 Data and rolling design | §2, §8 |
| §4 H1 — conditional variance (QLIKE) | §9 |
| §5 H2 — predictive distributions (log score, Student-$t$, $3\,\mathrm{CV}^2$) | §10, §11 |
| §6 H3 — portfolio construction and simulation (flowchart, accounting) | §12 |
| §7 H3 — evaluation and the exposure-matched control | §13 |
| §8 Limitations and future research | §14.3–14.4 |
| §9 Summary and scientific conclusions | §16 |
| App. A model adequacy; App. B robustness; App. D reproduction | §11; §4.5, §5.7, §10.4, §11.7, §13.8, §15.5; §8.4, §15 |

**Reproducibility levels**

| Level | Meaning |
|---|---|
| 🟢 **raw data** | computed in the notebook from adjusted prices / DFF, or from a model fitted in the notebook |
| 🔵 **saved forecasts** | recomputed in the notebook from the saved daily forecasts, posterior draws or fit logs of the full rolling experiment (`agent_research/results/forecasts`, SHA-256 verified) |
| ⚪ **artifact** | displayed from a checksum-verified saved research artifact, because recomputing it needs extra MCMC runs or came from an external audit |
| ✅ **verified** | covered by an automated check: a `check(...)` in the notebook (130 in total, summarised in §15.6) and/or a pytest test |

**Global verifications.**
- **Report tables.** All 13 LaTeX tables generated for the report (the 7 original tables and 6 added for the narrative:
  `forecasts_qlike`, `forecasts_logscore`, `ratios`, `exposure_compare`, `exposure_stage1`, `exposure_stage2`) are regenerated from the notebook's own
  computations, using `scripts/make_report_tables.py` unchanged. They are **byte-identical** to `reports/source/tables/*.tex` (§14.2).
  The 7 original tables are themselves unchanged by the restructuring.
- **Report figures.** The 21 data figures from the original report (`fig1–4` and the 17 `adequacy_*` figures) are
  regenerated from the saved daily data and are **pixel-identical** to `figures/*.png` (§11, §13.6).
- **Narrative figures and diagrams** (`artifacts/figures/narrative/`): `rolling_design`, `portfolio_flowchart` (TikZ sources),
  `qlike_contrasts`, `predictive_densities`, `exposure_control` are drawn by
  `scripts/make_narrative_figures.py` from the same saved tables and draws. Every number they show is checked
  by `scripts/verify_presentation_numbers.py` (33/33) and `scripts/verify_summary_numbers.py` (26/26), and the notebooks recompute the underlying quantities.

## Tables

| Report table (file) | Content | Notebook | Level | Verified |
|---|---|---|---|---|
| `fits.tex` (§3) | convergence: refits, failures, max R-hat, min tail ESS, MLE on bound, median ν | §5.8 | 🔵 (fit logs) | ✅ byte-identical; "4 of 320 failures" check |
| `forecasts_qlike.tex` (§4, H1) | pooled DM contrasts of QLIKE | §9.4 | 🔵 | ✅ equal to research tables to 1e-12; byte-identical |
| `forecasts_logscore.tex` (§5, H2) | pooled DM contrasts of the log score | §10.2 (mixture score recomputed from posterior draws in §10.1) | 🔵 | ✅ byte-identical |
| `ratios.tex` (§5, H2) | Student-$t$ gain / Bayesian gain; Laplace share | §10.2 (`RAT`), §10.3 | 🔵 | ✅ byte-identical; ratio checks (5.44, 6.11; 3.64–7.52; 0.432–0.886) |
| `var.tex` (§5.1) | VaR violation rates and Kupiec rejections | §11.4 | 🔵 | ✅ byte-identical; 2% vs 1.5% check |
| `exposure_stage1.tex` (§7.1, H3 Stage 1) | standard MLE vs Bayes upper-90: exposure, return, volatility, Sharpe, max DD; counts of configuration CIs excluding 0 | §13.2 | 🔵 | ✅ byte-identical; Stage 1 checks (values; CIs 8/8 drawdown, 0/8 Sharpe) |
| `exposure_stage2.tex` (§7.3, H3 Stage 2) | Bayes upper-90 vs exposure-matched MLE, same metrics | §13.5 | 🔵 | ✅ byte-identical; Stage 2 checks (0/8 and 0/8) |
| `exposure_compare.tex` (no longer included in the report; kept for reference) | all three rules side by side | §13.6 | 🔵 | ✅ byte-identical |
| `exposure.tex` (§7.3, both periods) | upper-90 minus MLE / control: exposure, drawdown, Sharpe | §13.6 | 🔵 | ✅ byte-identical; 8/8, 0.8–1.0 pp, 3.6–4.1 pp, −0.07/+0.09, +0.003/−0.001 |
| `allocation.tex` (§7) | four-ETF sleeve Sharpe and max drawdown | §13.6 | 🔵 + 🟢 | ✅ byte-identical |
| `paired.tex` (§7) | paired ΔSharpe with block-bootstrap CIs | §13.6; bootstrap rebuilt by hand in §13.7 | 🔵 | ✅ byte-identical; hand-built sleeve equal to table entry |
| `forecasts.tex` (combined QLIKE + log score; kept for the full presentation and earlier documents) | — | §9.4, §10.2 | 🔵 | ✅ byte-identical |
| `adequacy.tex` (App. A) | out-of-sample residual / PIT diagnostics | §11.3 | 🔵 (forecasts + draws) | ✅ byte-identical; equal to `artifacts/results/adequacy` |

## Figures

| Report figure | Notebook | Level | Verified |
|---|---|---|---|
| Rolling out-of-sample design (`rolling_design`, §3) | §8 (diagram), §8.2–8.3 (timeline, look-ahead checks) | diagram | structure checks on all 16 jobs; 63-day refits checked |
| H1 QLIKE contrasts (`qlike_contrasts`, §4) | §9.4 | 🔵 | values equal to report Table 2 QLIKE block |
| H2 log-score gains (`fig1_logscore_gains.png`, §5) | drawn in §10.2; regenerated with the report code in §13.6 | 🔵 | ✅ pixel-identical |
| H2 predictive densities, 5 Feb 2018 (`predictive_densities`, §5.1) | same construction for 2023-01-03 in §6.2 / §7 and in `quick_walkthrough.ipynb` §5 | 🔵 | numbers checked by `verify_numbers_5min.py` |
| Observed vs parameter-implied kurtosis (`fig4_kurtosis.png`, §5.1) | §11.7 (table), §14.2 (figure); $3\,\mathrm{CV}^2$ identity checked in §10.3 | 🔵 | ✅ pixel-identical |
| Portfolio simulation flowchart (`portfolio_flowchart`, §6) | §12.3 (flowchart, step table), followed by the explicit ledger and worked calculation | diagram | ledger equal to simulator; drift and NAV identities; execution-lag, cash and control canaries |
| Exposure vs drawdown (`exposure_control`, §7) | §13.4 | 🔵 | exposure-match check (+0.1 / −0.3 pp) |
| Upper-90 ΔSharpe vs comparators (`fig2_sharpe_differences.png`, §7) | §13.6 | 🔵 | ✅ pixel-identical |
| SPY 2020 weight paths, MLE / upper-90 / exposure-matched MLE (`course/presentations/source/figures/weights_spy2020.pdf`, report Figure 7, §7.2) | §12.2 (extended weight plot with the same strategies); `fig3_weight_paths_spy2020.png` (posterior-mean-plug-in control) is still regenerated and pixel-checked in §13.6 but no longer shown in the report | 🔵 | `scripts/make_presentation_figures.py` from saved forecasts; `fig3` ✅ pixel-identical |
| 1% exceedances (App. A) | §11.6 | 🔵 | ✅ pixel-identical |
| Adequacy figures (`adequacy_returns_*`, `_models_*`, `_diagnostics_*`, `_stress_*`) | §11.1, §11.2, §11.3, §11.5 | 🔵 | ✅ pixel-identical (16 files) |

## Quantitative claims in the text

| Report § | Claim | Notebook | Level | Verified |
|---|---|---|---|---|
| Abstract, §4 | QLIKE Bayes − MLE about −0.005 ($p=0.10$, $0.08$ in 2017–22); not significant at 5% in any cell | §9.4 | 🔵 | ✅ (min $p$ = 0.076) |
| §4 | expected QLIKE minimised at the true variance; excess loss $x-1-\log x$; no Gaussianity needed | §9.1 (derivation), §9.1 simulation with $t_5$ returns | 🟢 | ✅ |
| §4, §5 | for one Gaussian, $-\log$ score $=\tfrac12(\log2\pi+\text{QLIKE})$; mixture scored as the log of the average density (Jensen) | §10.1 | 🔵 | ✅ |
| Abstract, §5 | mixture log-score gain 0.002–0.005 (0.0015–0.005) nats/day, $p\le0.024$ in every pooled test | §10.2 | 🔵 | ✅ |
| Abstract, §5 | Laplace captures 43–89% of the gain; Student-$t$ gains 0.017–0.029, about 5–6× the Bayesian gain in 2017–22 (3.6–7.5× across all cells), beats mixture in all cells ($p\le0.011$) | §10.2, §10.3 | 🔵 | ✅ ratios checked directly — **corrected, item 3** |
| §2.3, §5.1 | Student-$t$ fitted by MLE at all 320 refits; median $\hat\nu$ 7.2; 7 refits at the upper bound | §7 (fresh fit), §10.3 | 🟢 / 🔵 | ✅ |
| §5.1 | posterior CV 7–12%; implied excess kurtosis $3\,\mathrm{CV}^2$ = 0.015–0.046 vs observed 0.4–2.9 | §10.3 (identity), §11.7 | 🔵 | ✅ |
| §5.1 | 1% VaR: Gaussian about 2% with or without integration, Student-$t$ 1.5% | §11.4 | 🔵 | ✅ |
| §5.1 | 5 Feb 2018 illustration (−4.3%, densities 8.6× apart, tail probabilities) | — (report figure only) | 🔵 | `verify_numbers_5min.py` and `narrative_numbers.json` |
| §2.1 | GARCH model, transformation $u=(\log V,\operatorname{logit}p,\operatorname{logit}q)$, $h_1=V$ | §3, §4.2 | 🟢 | ✅ round-trip and validity checks; `tests/test_library.py` |
| §2.1 | MLE: multi-start L-BFGS-B, bound $p\le0.99909$, Powell fallback | §4.3 (fresh fit) | 🟢 | ✅ equal to saved and Milestone 10 fit; independent Nelder–Mead |
| §2.1, App. B | MLE on the bound in 30 of 320 refits; relaxing it: log-lik ≤ 1.15, variance median 5% | §4.5 | 🔵 / ⚪ `audit_checks.json` | ✅ / displayed |
| §2.2 | prior, RWM, 4 chains, 2,500 warm-up, 4,000 draws, 23.4%; strict rule $\hat R<1.01$, ESS ≥ 400 | §5.2–5.5, §5.8 | 🟢 / 🔵 | ✅ posterior mean equal to saved fit |
| §2.2 | MCMC agrees with importance sampling within 0.3% | §5.6 | ⚪ `audit_checks.json` | ✅ (max 0.303%) |
| App. B | same-seed rerun on another platform: quantiles 0.5–1.7%; upper-90 Sharpe varies by up to 0.010 across 3 seeds | §5.7 | ⚪ research record, `seed_prior_sensitivity.csv` | ✅ (max 0.0099) |
| §2.2 | forecast definitions (plug-in, mean plug-in, integrated, mixture, $M=300$) | §6.1–6.2 | 🟢 | ✅ fresh values equal to saved columns |
| §2.4 | EWMA $\lambda=0.94$, Laplace comparators | §7 | 🟢 | ✅ fresh values equal to saved columns |
| §3 | data spans; 1,510 and 964 evaluation days; 63-day refits; fits end before forecasts | §2.1, §8 | 🟢 / 🔵 | ✅ structure checks on all 16 jobs |
| §6 | timing (information to close $d-1$, trade at close $d$), daily rebalance, 5 bp turnover cost on wealth, DFF dated before close $d$ ACT/360, drift | §12.1–12.3, worked calculation | 🟢 + 🔵 | ✅ hand ledger equal to simulator; drift and NAV identities; execution-lag and cash canaries |
| §7 | control $k_d$ = trailing 252-row mean of $\sqrt{\hat h/q_{0.9}}$, past rows only; exposure gap +0.1 pp (2017–22), −0.3 pp (2023–26) | §13.4 | 🔵 | ✅ control-causality canary; exposure-match check |
| §7 | upper-90 smaller drawdown than MLE in 8/8 (0.82, 1.02 pp), 3.6–4.1 pp less exposure; vs control 2/8 and 4/8 | §13.6 | 🔵 | ✅ |
| §7 | Gaussian-GARCH sleeve Sharpe ratios within about 0.02; with Student-$t$ the spread reaches 0.047 (2017–22) | §13.6 | 🔵 | ✅ (0.019, 0.016) — **corrected, item 2** |
| Abstract, §7 | every Sharpe-difference interval (vs MLE, control, Laplace upper-90) contains zero | §13.6–13.7 | 🔵 | ✅ (12 intervals) |
| §7, App. B | circular block bootstrap, 20-day blocks, 2,000 replications; costs 0/5/20 bp reverse no conclusion | §13.7, §13.8 | 🔵 | ✅ |
| App. B | premium correlation > 0.99; 4 of 32 HAC coefficients significant | §11.7 | 🔵 | ✅ — **corrected, item 1** |
| App. B | broad prior gives the same conclusions (bimodal TLT); conclusions unchanged without failed refits | §15.5, §10.4 | ⚪ / 🔵 | ✅ |
| App. B | exploratory strategy search: no candidate passed the validation gate | §15.5 | ⚪ `discovery/validation_decision.json` | displayed |
| App. A | ACF₁($r^2$) 0.43–0.51; ≈0 after standardising; TLT ≈0.1 to lag 8; excess kurtosis of $z$ 1.9–2.9; $\lvert z\rvert>3$ on 23–33 of 2,474 days | §11.1, §11.3 | 🔵 | ✅ |
| App. A | lower 1% exceedance 2.5%; mixture normal-score kurtosis 2.9 → 2.7; skew −0.67 to −0.85; PIT rejected for equities | §11.3 | 🔵 | ✅ |
| App. A | breaches highest when predicted volatility is low; stress half-lives 9–16 days, EWMA 2–3× longer | §11.5–11.6 | 🔵 | ✅ / displayed |
| App. D | 54 tests; 48 artifacts verified; quick mode about 25 s; full mode 320 refits about 70 s | §8.4, §15.3 | ⚪ `full_run.json` + live checksum verification | ✅ |

## Discrepancies found and corrected (2026-11-06)

All are wording errors in summary text. No table, figure, model, test or result changed: every report table and
figure still reproduces exactly (byte- and pixel-identical). The corrections were made in `reports/source/research_report.tex`
(recompiled to `reports/research_report.pdf`, author now **Kang Yu**), in the notebook's summary text, in `README.md`,
`docs/study_guide.md`, and as a dated erratum at the top of `agent_research/final_report.md`.

1. **Number of significant premium regressions (report §6 at the time; now Appendix B).** The text said "3 of 32 coefficients are significant at
   5%". The recomputed information tables, identical to `agent_research/results/primary/info_{ext,test}.csv`, give
   **4 of 32**:
   - 2017–22: TLT $W=500$, both outcomes ($p=0.009$, $0.004$); QQQ $W=500$, large-shock ($p=0.048$);
   - 2023–26: SPY $W=500$, absolute error ($p=0.035$).

   The conclusion is unaffected: close to the chance level (1.6 expected; $P(\ge4)\approx0.07$ for independent
   tests), mixed signs, no timing information. Notebook §11.7.
2. **Sharpe spread of "all GARCH-based rules" (report §5 at the time; now §7).** "Within about 0.03 of one another in each period" holds for
   the Gaussian-GARCH rules (MLE, MAP, Bayesian, Laplace, controls; spread at most 0.019). Including **Student-$t$
   GARCH** the spread is 0.047 in 2017–22 ($W=500$: 0.903 vs 0.856). The report now says "within about 0.02 …
   including Student-$t$ GARCH, the spread reaches 0.047 in 2017–22", and the notebook check uses 0.02. Notebook §13.4.
3. **Size of the Student-$t$ and Laplace comparisons (abstract, §5 at the time, conclusions).** "Student-$t$ is worth four to six
   times more" and "the Laplace approximation captures most of the gain" were too broad. From the pooled contrasts:

   | Cell | Student-$t$ gain / Bayesian gain | Laplace share of Bayesian gain |
   |---|---|---|
   | 2017–22, $W=500$ | 5.44 | 0.886 |
   | 2017–22, $W=1000$ | 6.11 | 0.432 |
   | 2023–26, $W=500$ | 3.64 | 0.819 |
   | 2023–26, $W=1000$ | 7.52 | 0.690 |

   The text now reads "about 5–6× in 2017–22 (3.6–7.5× across all cells)" and "43–89% (more than half in three of
   the four cells)". Three notebook checks (§10.2) test these ratios directly.

## Items intentionally not recomputed in the standard notebook run

| Item | Why | How to recompute |
|---|---|---|
| the 320 rolling refits (MLE, MAP, MCMC, Laplace, Student-$t$) | about 45 s of CPU on 7 cores and Monte Carlo platform dependence; the notebook verifies checksums and structure, refits one window live, and recomputes everything downstream | `python -m bgarch_experiments.reproduce full --out NEW_DIR` (last two runs: identical outputs, `artifacts/results/full_run/`, `artifacts/results/package_refactor_check/`) |
| broad-prior and extra-seed runs | additional full MCMC grids | research code in `agent_research/code` (see `agent_research/research_plan.md`) |
| relaxed-bound MLE audit, importance-sampling audit | external audit artifacts | `agent_research/code` (audit); results in `agent_research/results/audit_checks.json` |

## Original expansion requirements: completion checklist (re-verified 2026-11-06 after the restructuring)

The section numbers below are those of the restructured notebook; the content of each original requirement is unchanged.

| Requirement | Status | Evidence |
|---|---|---|
| Introduction: abstract, H1–H3, chain estimation → forecasting → allocation → evaluation, roadmap | complete | §1 |
| Data: ETFs, adjusted closes, log returns, annualisation, stylised facts | complete | §2.1–2.4 |
| GARCH: conditional variance, parameters, persistence, positivity, stationarity, long-run variance, initialisation, forecast, likelihood | complete | §3.1–3.4 |
| MLE: formulation, transformation, optimiser and safeguards, live fit, boundary cases | complete | §4.1–4.5 |
| Bayes: theorem, priors, RWM with pseudocode, adaptation, trace/ACF/ESS/$\hat R$/MCSE, live fit, cost, limitations | complete | §5.1–5.8 |
| Posterior forecasts: plug-in vs integrated vs upper-quantile, predictive mixture, intervals, VaR, tails | complete | §6.1–6.4 |
| Benchmarks: Gaussian MLE, Student-$t$ (MLE), EWMA, Laplace; parameter vs innovation uncertainty | complete | §7 |
| Design: windows, 63-day refits, daily updates, periods, diagram, timeline, look-ahead safeguards, computational strategy | complete | §8 |
| Evaluation: QLIKE (H1) and log score (H2) recomputed, DM with Newey–West, CIs, significance vs importance | complete | §9, §10 |
| Adequacy and stress: bands, residuals, ACF of $z^2$, Ljung–Box, PIT, VaR, March 2020, model comparison | complete | §11.1–11.7 |
| Portfolio construction: formula, flowchart, timing, costs, drift, cash, NAV, explicit ledger, worked calculation | complete | §12.1–12.3 |
| Portfolio evaluation: measures, exposure effect, exposure-matched control, Tables, bootstrap, costs | complete | §13.1–13.6 |
| Discussion, limitations, future research | complete | §14 |
| Reproducibility: environment, seeds, artifacts, workflows, derivations, check summary | complete | §15.1–15.6 |
| Final H1/H2/H3 summary with the two central findings | complete | §16 |
| Consistent format per experimental section (motivation … limitations) | complete | experiment cards in §4, §5, §9, §10, §11, §12 |
| Report-to-notebook checklist with reproducibility levels | complete | this file |
| Executed notebook, clean-kernel run, tests | complete | clean-kernel run 2026-11-06: 130/130 checks, about 40 s; 54 tests |

## Question-driven restructuring (2026-11-06)

**What changed.** Sections were reordered and renumbered around H1 → H2 → H3: §9 is now H1 only (QLIKE derivation,
a simulation check that QLIKE is minimised by the true variance with $t_5$ returns, and an explicit H1 conclusion); §10
is H2 (log score, its relation to QLIKE, the Jensen/density-averaging check, and a new §10.3 on Student-$t$ versus
Bayesian integration with Finding 1); the old adequacy section became §11 (H2 continued); §12 adds the portfolio
simulation flowchart and a worked calculation; §13 adds the explanation-A-versus-B framing, the exposure-match table
and the exposure-versus-drawdown figure, and Finding 2; §16 is the final research summary. Ten checks were added
(116 → 126). No existing code cell, model, artifact or validated number changed.

**Implementation details confirmed while drawing the flowchart.**
- The control's scale $k_d$ matches upper-90's average *haircut* (the ratio of uncapped weights) over the previous
  252 forecast rows of the evaluation segment, not its realised exposure; with the cap at 1 the exposure match is
  approximate (+0.1 pp in 2017–22, range −0.06 to +0.41; −0.3 pp in 2023–26, range −1.32 to +0.10).
- The executed weight equals the target every day (full rebalance); costs are charged on wealth, not on the weight.
- Cash uses the DFF value dated strictly before the decision close $d$, accrued over calendar days.
- The kurtosis comparison standardises returns by the posterior-integrated forecast (not the MLE forecast); the report
  text was corrected accordingly.

## Concise notebook (`quick_walkthrough.ipynb`)

The recommended starting point: 52 cells (24 code) in 12 sections, 9 figures plus the two shared diagrams, 45 checks,
about 30 s from a clean kernel. It follows the same narrative: question → data → GARCH/MLE (live fit) → Bayesian GARCH
(live 4-chain MCMC) → Student-$t$ GARCH (live MLE fit) → rolling design → H1 (QLIKE derivation and simulation check) →
H2 (log score, density-averaging check, Student-$t$ comparison, $3\,\mathrm{CV}^2$ argument, crisis diagnostics) → H3
(flowchart and worked ledger) → exposure-matched control (rebuilt $k_d$ with a look-ahead canary, exposure match,
re-simulated allocation tables, block bootstrap by hand) → limitations → summary. Recomputed from daily data (🔵):
pooled contrasts equal to report Table 2 within $10^{-7}$, all allocation tables equal to the research tables. The
Laplace mixture log score is taken from the saved daily column, because the Laplace draws were not stored.

## Cross-document consistency audit (2026-11-06, second pass)

Checked the report, the 12-slide presentation and notes, both notebooks, the new executive summary, the source code and
the saved results. **Verified errors corrected** (no result changed):
1. *Posterior draws.* The report said the $M=300$ predictive draws were "thinned"; the code
   (`walkforward.py`, `rng.choice(..., replace=False)`) samples them at random without replacement from the 16,000
   retained draws. Report §2.2 corrected.
2. *Kurtosis comparison.* The 0.015–0.046 range is the day-average of $3\,\mathrm{CV}^2$ (`tables.info_tables`), not
   $3\times(\text{average CV})^2$; "CV of 7–12% implies 0.015–0.046" was imprecise (e.g. SPY 500d: $3\cdot0.094^2=0.027$
   vs 0.029 reported). Report, slide 6, speaker notes and notebook §10.3 corrected.
3. *Stale cross-references.* After the report was restructured its tables and figures were renumbered (old Table 2 →
   Tables 2–3, Table 3 → 8, Table 4 → 9, Table 5 → 7, Table 6 → 5; Figure 1 → 3, 2 → 8, 3 → 9, 4 → 5), and claims moved
   (old §2/§5/§6 → §2.1, §2.2, §5, §5.1, §7, App. B). All references in `walkthrough.ipynb` (markdown, display captions
   and check labels) and `quick_walkthrough.ipynb` were updated.
4. *Density averaging.* The report and notebook called the average of component log densities "a different (and
   improper) object" and limited the Jensen inequality to tail returns. Now stated precisely: the mixture score is
   $\ge$ the average component score for every outcome, and the exponentiated average log density (a geometric mean of
   densities) integrates to less than one, so it is not the score of any predictive density.

**Precision and presentation improvements.** QLIKE ranking statements now refer to *expected* loss, note that finite
samples still need a test, and state the finite-second-moment condition (report §4, slides 4–5, notes, both notebooks);
"QLIKE equals the Gaussian NLL up to constants" made exact ($-\log N=\tfrac12(\log2\pi+\text{QLIKE})$); the Sharpe
definition stated exactly (excess-return mean over excess-return standard deviation, $\times\sqrt{252}$); the
aggregation (8 single-ETF configurations vs equal-weight four-ETF sleeve) labelled wherever Sharpe or drawdown
results appear (report abstract and §7, notebook §13.4); the control's start-of-sample convention ($k_0$) added to
slide 9; the flowchart's legend and loop caption enlarged after inspection at 1280×720.

## Two-stage H3 presentation (2026-11-06, third pass)

H3 is now presented in two stages in every deliverable (slides 7–11, report §7, executive summary §7, comprehensive
notebook §13, concise notebook §10). **Stage 1** compares standard MLE and Bayesian upper-90 only (report Table 6,
`exposure_stage1.tex`): upper-90 has lower volatility (9.81% vs 10.37%) and a shallower drawdown (+0.82 pp; CIs exclude
0 in 8 of 8 configurations) but 3.6 pp less exposure and an almost unchanged Sharpe ratio (CIs exclude 0 in 0 of 8),
which raises the question *timing or less risk?* **Stage 2** then introduces the exposure-matched MLE control and
compares it with upper-90 on the same metrics (report Table 7, `exposure_stage2.tex`): no Sharpe or drawdown interval
excludes 0. The control is not shown in any table before it is introduced; the flowchart marks its branch as the
Stage 2 control. No experiment or number changed. Tables, the notebook cross-check (now 13 tables) and the helper test
were extended accordingly; the deck reads both stage tables directly from `reports/source/tables/`.

**Errors found while doing this.**
- *Report Figure 7 caption.* The draft caption described `fig3_weight_paths_spy2020.png` as showing the exposure-matched
  MLE control; its dashed line is actually the posterior-mean-plug-in control (`meanplug_const_rt`,
  `bgarch_experiments/figures.py`). The report now uses `course/presentations/source/figures/weights_spy2020.pdf`, which shows the MLE
  control (`mle_const_rt`), with a matching caption.
- *Notebook cross-check inputs.* The comprehensive notebook's report-table regeneration did not pass the per-configuration
  pair tables (`alloc_pairs_*`) to `scripts/make_report_tables.py`; it now does (13/13 byte-identical). A failed execution had
  briefly left a stale "126/126" output in place; the final run was checked through its exit status.

## Bibliography standardisation (2026-11-06)

The report and the executive summary now share one BibTeX file, `references.bib`, with numeric square-bracket citations
(`natbib` `[numbers,square,sort&compress]`, style `unsrtnat`), numbered automatically in order of first citation. The
hand-written `thebibliography` of the report was replaced; the summary's plain-text "Key references" list became
`\cite` commands at the points where each method is introduced. Every DOI-bearing entry (18 of 20) was checked
against Crossref metadata (authors, year, title, journal, volume, issue, start page); the two entries without a DOI
(Politis and Romano 1992; RiskMetrics 1996) keep their existing details and carry no DOI. Two previously listed but
uncited entries (Christoffersen 1998; Holm 1979) were removed; Engle (1982) is now cited where ARCH is introduced.
Both PDFs were checked for unresolved citations, `[?]`, `??` and raw keys (none), and each bibliography contains
exactly the cited keys, in citation order (report 20, summary 14).
