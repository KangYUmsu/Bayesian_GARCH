"""Gaussian zero-mean GARCH(1,1): parameter transformations, likelihood, variance filter, MLE.

Model (daily decimal log returns r_t, information F_{t-1} = sigma(r_1..r_{t-1})):

    r_t = sqrt(h_t) z_t,   z_t ~ iid N(0, 1)
    h_t = Var(r_t | F_{t-1}) = omega + alpha r_{t-1}^2 + beta h_{t-1},   omega > 0, alpha, beta >= 0, p = alpha + beta < 1

Coordinates. Estimation works in u = (log V, logit p, logit q) with V = omega/(1-p) the
unconditional (long-run) variance, p = alpha + beta the persistence and q = alpha/p:

    omega = V (1-p),  alpha = p q,  beta = p (1-q).

Every u in R^3 is a positive, covariance-stationary model. Arrays ``u`` have shape (3,).
Initialisation: the filter starts at h_1 = V (the first return of the array is scored with h = V).

Functions (validated implementations imported unchanged from the original project):

    loglik_u(r, u) -> float          sum_t log N(r_t; 0, h_t(u))        <- src.garch._ll (numba);
                                     returns -1e300 for an invalid u or a non-finite variance
    mle_fit(r) -> (u_hat, result)    <- src.garch.mle_fit: three-start L-BFGS-B (ftol 1e-12, maxiter 2000)
                                     on -loglik with box bounds MLE_BOUNDS, Powell fallback when a failed
                                     start reaches a better objective, refusal (RuntimeError) if the
                                     conflict cannot be resolved. ``result`` is a scipy OptimizeResult with
                                     extra fields method, fallback_used, num_attempts, attempts.
                                     The bound logit p <= 7 caps p at P_MAX_MLE = 0.99909 (NOT changed).
    variance_path(local, start, u)   <- src.market_uncertainty_backtest._garch_path: array of length
                                     len(local) - start with h_t for t = start..len(local)-1, computed from
                                     local[:t] only (parameters frozen, filter restarted at local[0] with V)
    encode(V, p, q) -> u, unpack(u) -> dict, simulate(n, seed, V, alpha, beta)   <- src.garch

The *_reference functions are plain-Python re-derivations used for teaching and as independent checks.
"""
import math

import numpy as np
from scipy.special import expit

from src.garch import _ll as loglik_u, mle_fit, encode, unpack, simulate  # noqa: F401
from src.market_uncertainty_backtest import _garch_path as variance_path  # noqa: F401

MLE_BOUNDS = [(-15, -5), (-7, 7), (-7, 7)]       # box bounds of src.garch.mle_fit (copied for display/tests)
P_MAX_MLE = float(expit(7.0))                     # 0.999088..., the persistence cap implied by the bound


def to_natural(u):
    """u -> (omega, alpha, beta)."""
    V = math.exp(u[0]); p = float(expit(u[1])); q = float(expit(u[2]))
    return V * (1 - p), p * q, p * (1 - q)


def from_natural(omega, alpha, beta):
    """(omega, alpha, beta) -> u; requires omega > 0, alpha > 0, beta > 0 and alpha + beta < 1."""
    p = alpha + beta
    if not (omega > 0 and alpha > 0 and beta > 0 and p < 1):
        raise ValueError('need omega>0, alpha>0, beta>0, alpha+beta<1')
    return encode(omega / (1 - p), p, alpha / p)


def half_life(p):
    """Trading days for a variance shock to halve: log(1/2) / log(p), since E[h_{t+k}] - V = p^(k-1) (h_{t+1} - V)."""
    return np.log(.5) / np.log(p)


def next_variance(train, u):
    """One-step-ahead forecast h_{n+1} after filtering the n returns in ``train`` with parameters u."""
    return float(variance_path(np.r_[np.asarray(train, float), 0.0], len(train), u)[0])


def variance_path_reference(r, start, omega, alpha, beta):
    """h_t for t = start..len(r)-1, using only r[:t]; plain Python loop (teaching/check)."""
    h = omega / (1 - alpha - beta)
    out = []
    for t in range(len(r)):
        if t > 0:
            h = omega + alpha * r[t - 1] ** 2 + beta * h
        if t >= start:
            out.append(h)
    return np.array(out)


def loglik_reference(r, omega, alpha, beta):
    """sum_t log N(r_t; 0, h_t) with h_1 = omega/(1-alpha-beta); plain Python loop (teaching/check)."""
    h = omega / (1 - alpha - beta); s = 0.0
    for t, x in enumerate(r):
        if t > 0:
            h = omega + alpha * r[t - 1] ** 2 + beta * h
        s += -0.5 * (math.log(2 * math.pi) + math.log(h) + x * x / h)
    return s
