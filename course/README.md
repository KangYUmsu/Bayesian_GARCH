# Erdős Institute reviewer package

**Bayesian GARCH: From Volatility Forecasting to Portfolio Allocation** · Kang Yu · Quantitative Methods in Finance,
Fall 2026

*Does accounting for Bayesian parameter uncertainty improve conditional variance forecasts, predictive return
distributions, and volatility-targeted investment performance?* In brief: no reliable gain for variance forecasts
(H1); a small, reliable gain for predictive densities that Student-*t* GARCH exceeds about five to six times (H2); and
no portfolio benefit once risky exposure is matched (H3).

Choose the depth you want; the materials can be read in any order.

| Material | Purpose |
|---|---|
| [Executive summary](executive_summary.pdf) (8 pages) | Quick overview of the research questions, methodology and findings |
| [Full research report](research_report.pdf) (23 pages) | Complete technical explanation and numerical evidence, with appendices |
| [Quick walkthrough](quick_walkthrough.ipynb) | Concise executable introduction: live MLE, MCMC and Student-*t* fits, H1–H3 recomputed |
| [Comprehensive walkthrough](walkthrough.ipynb) | Detailed derivations, algorithms, experiments, diagnostics and 130 numerical checks |
| [Presentation](slides.pdf) (13 slides) | Visual overview of the project; [speaker notes](presentations/source/speaker_notes.md) include a five-minute cut for the course video |

The notebooks are stored executed, so all outputs and figures are visible without running them. Re-executing them
needs the market data and saved artifacts, which are not redistributed; see the repository
[README](../README.md#data-and-saved-artifacts).

**Sources.** Editable LaTeX sources are in [`reports/source/`](reports/source/) and
[`presentations/source/`](presentations/source/).

**Relation to the rest of the repository.** The reports and notebooks here are copies of the authoritative files in
[`../reports/`](../reports/) and [`../notebooks/`](../notebooks/), kept identical by `python scripts/sync_course.py`
(run from the repository root; `--check` verifies without copying). The code, tests and reproduction pipeline are
described in the repository [README](../README.md).
