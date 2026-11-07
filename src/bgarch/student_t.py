"""Student-t GARCH(1,1) MLE (innovation-misspecification robustness check) and VaR / density helpers.

COPIED VERBATIM from agent_research/code/walkforward.py (lines 46-98 of that file; its duplicate
mixture_logpdf now lives only in bgarch.forecast);
tests/test_reproduction.py checks identical output against the original module.

Standardised-t innovations with nu > 2 degrees of freedom so that Var(z)=1:
    log f(r_t | h_t) = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5 log(pi (nu-2) h_t)
                       - (nu+1)/2 * log(1 + r_t^2 / ((nu-2) h_t)).
nu is optimised as eta = log(nu-2) within [log 0.05, log 200]; the GARCH coordinates
use the SAME box bounds as the Gaussian MLE.
"""
import math

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

from .garch import MLE_BOUNDS

# ---------------------------------------------------------------- Student-t GARCH
@njit(cache=True)
def _lg(x):
    return math.lgamma(x)


@njit(cache=True)
def _ll_t(r, u, eta):
    V = np.exp(u[0]); p = 1.0 / (1.0 + np.exp(-u[1])); q = 1.0 / (1.0 + np.exp(-u[2]))
    nu = 2.0 + np.exp(eta)
    omega = V * (1 - p); alpha = p * q; beta = p * (1 - q)
    c = np.log(np.pi * (nu - 2.0))
    v = V; total = 0.0
    for t in range(len(r)):
        if t > 0:
            v = omega + alpha * r[t - 1] ** 2 + beta * v
        if v <= 0 or not np.isfinite(v):
            return -1e300
        total += -0.5 * c - 0.5 * np.log(v) - 0.5 * (nu + 1.0) * np.log(1.0 + r[t] ** 2 / (v * (nu - 2.0)))
    return total + len(r) * (_lg((nu + 1.0) / 2.0) - _lg(nu / 2.0))


def t_mle_fit(r, u_start):
    """Multi-start bounded MLE for standardized-t GARCH(1,1); nu in (2.05, 202).

    r: 1-D returns; u_start: (3,) GARCH coordinates used as the starting point (clipped to MLE_BOUNDS).
    L-BFGS-B from eta0 = log(nu0 - 2) for nu0 in {6, 10, 27} (i.e. eta0 = log 4, log 8, log 25).
    Returns (u_t of shape (3,), nu, best scipy OptimizeResult).
    """
    bounds = MLE_BOUNDS + [(np.log(0.05), np.log(200.0))]
    obj = lambda x: float(-_ll_t(r, np.asarray(x[:3]), float(x[3])))
    best = None
    for eta0 in (np.log(4.0), np.log(8.0), np.log(25.0)):
        x0 = np.r_[np.clip(u_start, [b[0] for b in MLE_BOUNDS], [b[1] for b in MLE_BOUNDS]), eta0]
        fit = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12})
        if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
            best = fit
    return best.x[:3], 2.0 + float(np.exp(best.x[3])), best


# ---------------------------------------------------------------- helpers
def gaussian_var(variance, alpha):
    """Gaussian one-day VaR (positive loss) with P(r < -VaR) = alpha: z_{1-alpha} sqrt(h)."""
    return norm.isf(alpha) * np.sqrt(variance)


def t_var(variance, nu, alpha):
    """Standardised-t one-day VaR: -t_nu^{-1}(alpha) sqrt(h (nu-2)/nu)."""
    return -student_t.ppf(alpha, nu) * np.sqrt(variance * (nu - 2.0) / nu)


def t_logpdf(y, variance, nu):
    """log density of a standardised-t return with variance h: scale s = sqrt(h (nu-2)/nu)."""
    s = np.sqrt(variance * (nu - 2.0) / nu)
    return student_t.logpdf(y / s, nu) - np.log(s)


