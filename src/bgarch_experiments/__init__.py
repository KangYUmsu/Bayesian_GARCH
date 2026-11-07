"""bgarch_experiments: configuration and orchestration of the course study (outside the reusable library).

    config       segments, windows, seeds, artifact locations
    strategies   the strategy set compared in the report
    walkforward  rolling estimation / forecasting engine (one job = asset x window x segment)
    tables       report tables from walk-forward artifacts (Store locates the files)
    figures      report figures fig1-fig4
    adequacy     model-adequacy tables and figures (report Appendix A)
    artifacts    SHA-256 manifest of the saved artifacts
    reproduce    quick / full reproduction CLI
    notebook     helpers used only by walkthrough.ipynb
"""
