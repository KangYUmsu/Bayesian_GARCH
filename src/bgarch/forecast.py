"""Posterior-predictive and plug-in forecasts for Gaussian GARCH(1,1).

Notation. ``local`` is a 1-D array of returns whose first ``offset`` entries are the
estimation window and whose remaining entries are the forecast block. A variance *path*
(``garch.variance_path``) has one entry per forecast row t = offset..len(local)-1 and
h_t uses only local[:t]. ``us`` is an (M, 3) array of parameter draws in the coordinates
u = (log V, logit p, logit q).

Definitions (M posterior draws u^(m), forecast row t):

    h_t^(m)                = h_t(u^(m))                           variance_draws -> (M, n_days)
    posterior-integrated    hbar_t = M^-1 sum_m h_t^(m)            (= predictive variance)
    upper-q variance        q-quantile over m of h_t^(m)           (np.quantile, linear interpolation)
    predictive density      p(r_t | F_{t-1}) ~= M^-1 sum_m N(r_t; 0, h_t^(m))     (Gaussian scale mixture)

The plug-in density uses a single variance: N(r_t; 0, h_t(u_hat)). ``predictive_summary``
reproduces exactly the columns written by the walk-forward pipeline.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .garch import variance_path
from src.posterior_tail_risk import mixture_var_cvar  # noqa: F401  (validated bisection VaR/ES of a scale mixture)


def variance_draws(local, offset, us):
    """Matrix (M draws x n_days) of conditional variances, one GARCH filter per parameter draw."""
    return np.stack([variance_path(local, offset, u) for u in us])


def predictive_summary(v_draws):
    """Summaries of an (M, n_days) matrix of variance draws, as stored by the pipeline.

    Returns a dict of 1-D arrays: integrated (mean), q05, q90, q95 (``np.quantile``) and sd (ddof=1).
    """
    v = np.asarray(v_draws, float)
    return {'integrated': v.mean(0), 'q90': np.quantile(v, .9, axis=0), 'q05': np.quantile(v, .05, axis=0),
            'q95': np.quantile(v, .95, axis=0), 'sd': v.std(0, ddof=1)}


def gaussian_logpdf(y, variance):
    """log N(y_t; 0, h_t), elementwise."""
    return norm.logpdf(y, scale=np.sqrt(variance))


def mixture_logpdf(y, v_draws):
    """log of (1/M) sum_m N(y_t; 0, v_t^(m)) for each column t of ``v_draws`` (shape (M, n)).

    Evaluated with log-sum-exp for numerical stability. COPIED from
    agent_research/code/walkforward.mixture_logpdf.
    """
    return logsumexp(-.5 * (np.log(2 * np.pi * v_draws) + y[None, :] ** 2 / v_draws), axis=0) - np.log(len(v_draws))


def mixture_cdf(y, v_draws):
    """Predictive CDF of the scale mixture, F_t(y_t) = M^-1 sum_m Phi(y_t / sqrt(v_t^(m)))."""
    return norm.cdf(np.asarray(y)[None, :] / np.sqrt(v_draws)).mean(axis=0)


def mixture_var(v_draws, alpha):
    """One-day Value-at-Risk (a positive loss) of the scale mixture: P(r_t < -VaR_t) = alpha."""
    return mixture_var_cvar(np.asarray(v_draws, float), alpha=alpha)[0]


def mixture_excess_kurtosis(v):
    """Excess kurtosis of a zero-mean Gaussian scale mixture with mixing variances v: 3 E[v^2]/E[v]^2 - 3."""
    v = np.asarray(v, float)
    return 3 * np.mean(v ** 2) / np.mean(v) ** 2 - 3
