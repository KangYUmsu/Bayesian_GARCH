# AI-agent research records

This directory preserves the records of the research phase in which an AI agent (Claude, by Anthropic, used through
Claude Code in the author's project directory and under the author's direction) carried out an independent
assessment of the original Bayesian GARCH project: an audit of the implementation, a pre-registered out-of-sample
replication on 2017–2022, and the analysis that the final reports build on. The files are the original records,
unchanged except that the restricted forecast artifacts were moved out (see below).

## Contents

| File or folder | What it is |
|---|---|
| [`research_plan.md`](research_plan.md) | The plan, written on 2026-11-05 **before** any new experiment was run: research question, data splits, frozen specifications, pre-registered hypotheses and tests, audit checklist, compute budget, and the protocol for an exploratory strategy search |
| [`experiment_log.md`](experiment_log.md) | What was run, with commands, outcomes and run times; deviations from the plan and their reasons; errors the agent found in its own work and how they were fixed before results were reported |
| [`final_report.md`](final_report.md) | The agent's report of the assessment (2026-11-05), with a dated erratum (2026-11-06) correcting three summary statements; tables and tests unchanged |
| [`code/`](code/) | The scripts that produced the results: rolling estimation (`walkforward.py`, `run_grid.py`), audit checks, primary analysis, figures, strategy search |
| [`figures/`](figures/) | Figures of the assessment report |
| [`results/`](results/) | Aggregate outputs: `primary/` (the research tables that the package's quick mode reproduces column for column), `discovery/` (strategy-search selection and validation decision), `audit_checks.json`, `audit_mle_boundary.csv`, `run_grid.log` |

Not included: the daily forecast artifacts of this phase (`results/forecasts/`, 95 MB), because they contain returns
derived from restricted market data. The repository expects them under `artifacts/forecasts/`; see the main
[README](../README.md#data-and-saved-artifacts). The scripts in `code/` are kept as records and assume the original
workspace layout; the maintained implementation is the package in [`../src/`](../src/), where the functions taken
from these scripts are marked in each module docstring and tested for identical output.

## How the agent contributed

The records document the following, in the agent's own plan and log:

- **Audit of the existing implementation.** Likelihood, MLE boundary behaviour, MCMC (checked against an independent
  importance sampler), forecast and execution timing, cash-rate lag and portfolio accounting were checked
  (log steps P1–P5). The audit found that the original project's "all fits converged" claim did not reproduce and that
  Bayesian quantities differ across platforms at Monte Carlo precision.
- **Pre-registered design.** The plan fixed the hypotheses, data splits (including the new 2017–2022 evaluation
  period), specifications and tests before any result was examined.
- **Student-*t* GARCH comparison.** The original project contained no Student-*t* model. The plan added a Student-*t*
  GARCH MLE "to size the effect of *innovation misspecification* relative to *parameter uncertainty*" (hypothesis H3
  of the plan). This comparison became the project's first central finding.
- **Implementable exposure-matched controls.** The original project had an *ex-post* exposure attribution (its
  Milestone 11, not included here), whose matching constants used the evaluation period, with a note that an
  implementable, pre-test-calibrated version would be needed. The plan pre-registered implementable controls "whose
  scaling is calibrated only on earlier data" (plan hypothesis H5(d)); the trailing 252-row control used in the final
  results (`mle_const_rt`) comes from this phase. It underlies the project's second central finding.
- **Self-correction and negative results.** The log records a non-reproducible bootstrap seed in the agent's first
  draft, replaced before any result was reported, and two corrections made while checking the report. An
  exploratory search for uncertainty-aware strategies was pre-registered with a validation gate; no candidate passed,
  so the final holdout was never used.

## Relation to the final results

The research tables in `results/primary/` are the reference that the package reproduces: quick mode rebuilds every
table from the saved forecasts and matches 282 of 282 columns, and a full rerun of all 320 refits reproduced them
exactly on the development machine. The later phases (packaging the code as `bgarch`, the notebooks, the reports and
the presentation, and their consistency audits) were also carried out with Claude Code under the author's direction;
their record is the dated change log in [`../docs/notebook_coverage.md`](../docs/notebook_coverage.md) and the
validation scripts and tests themselves. Conversation transcripts are not included.
