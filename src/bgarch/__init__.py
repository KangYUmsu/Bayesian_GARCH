"""bgarch: a small library for Bayesian and classical GARCH(1,1) volatility forecasting and volatility targeting.

Modules (import them explicitly, e.g. ``from bgarch import garch, bayes``):

    garch        model, parameter transformations, Gaussian likelihood, variance filter, MLE
    bayes        prior, MAP, adaptive random-walk Metropolis, MCMC diagnostics, Laplace covariance,
                 posterior summaries
    forecast     posterior-predictive quantities: variance draws, integrated / upper-quantile variance,
                 mixture log density, CDF, VaR
    student_t    Student-t GARCH MLE, Gaussian / t VaR and log density
    ewma         RiskMetrics EWMA variance
    evaluation   QLIKE, Newey-West / Diebold-Mariano, circular block bootstrap, Holm, Kupiec, Christoffersen
    diagnostics  standardised residuals, PIT, normal scores, ACF, Ljung-Box, realised-volatility proxy
    portfolio    volatility-target weights, exposure-matched control, one-asset + cash backtest
    data         local price / rate files and log returns

The core numerics (likelihood, MLE, MAP, MCMC, variance filter, portfolio simulator, cash accrual,
EWMA) are IMPORTED from the original validated project package ``src/`` in the parent directory,
never re-implemented, so the scientific definitions cannot drift (see ``bgarch.paths``).
Experiment configuration, the rolling pipeline, report tables/figures and notebook helpers live in
the separate package ``bgarch_experiments``.
"""
from . import paths  # noqa: F401  (puts the original project on sys.path)

__all__ = ['paths', 'data', 'garch', 'bayes', 'forecast', 'student_t', 'ewma', 'evaluation', 'diagnostics', 'portfolio']
__version__ = '1.1.0'
