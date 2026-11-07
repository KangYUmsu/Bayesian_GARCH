"""Forecast-evaluation and paired-comparison statistics (losses, HAC / Diebold-Mariano, block bootstrap, VaR tests).

Array conventions: 1-D float arrays of daily values aligned on the same dates.
    hac_mean(d) is the Diebold-Mariano test when d = L_A - L_B is a loss differential:
        returns (mean, Newey-West se, z = mean/se, two-sided normal p).
    paired_bootstrap(ra, rb, rf) resamples the SAME circular blocks of days for both strategies
        and returns point differences (a - b) of Sharpe and max drawdown with 95% percentile intervals.
    kupiec(viol, a) / christoffersen_ind(viol): unconditional-coverage and independence LR tests of a
        boolean violation series.

COPIED VERBATIM from agent_research/code/common.py (hac_mean ... christoffersen_ind) and
agent_research/code/analysis_primary.py (qlike).

    QLIKE(y^2, h)   = log h + y^2 / h          (y = realised log return, h = forecast variance)
    Diebold-Mariano: d_t = L_A,t - L_B,t ;  z = mean(d) / NW-se(mean(d))
    Newey-West lags = floor(4 (n/100)^(2/9)), Bartlett weights
    Circular block bootstrap: 20-day blocks, paired (same indices for both strategies)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

def qlike(y2, v):
    return np.log(v) + y2 / v


# ------------------------------------------------------------------ inference
def nw_lags(n):
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def hac_mean(x, lags=None):
    """Newey–West mean test. Returns (mean, se, z, two-sided p)."""
    x = np.asarray(x, float); n = len(x)
    lags = nw_lags(n) if lags is None else lags
    z = x - x.mean(); s = z @ z / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (z[k:] @ z[:-k]) / n
    se = np.sqrt(max(s, 0) / n)
    t = x.mean() / se if se > 0 else np.nan
    return float(x.mean()), float(se), float(t), float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan


def hac_ols(y, X, lags=None):
    y = np.asarray(y, float); X = np.asarray(X, float); n, k = X.shape
    lags = nw_lags(n) if lags is None else lags
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; zz = X * e[:, None]
    S = zz.T @ zz
    for L in range(1, lags + 1):
        c = zz[L:].T @ zz[:-L]; S += (1 - L / (lags + 1)) * (c + c.T)
    B = np.linalg.pinv(X.T @ X); V = B @ S @ B
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, se, 2 * norm.sf(np.abs(b / se))


def block_indices(n, B, block=20, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def sharpe(ex, axis=-1):
    return np.mean(ex, axis=axis) / np.std(ex, axis=axis, ddof=1) * np.sqrt(252)


def maxdd(ret, axis=-1):
    eq = np.cumprod(1 + ret, axis=axis)
    peak = np.maximum.accumulate(np.concatenate([np.ones(eq.shape[:-1] + (1,)), eq], axis=-1), axis=-1)[..., 1:]
    return np.min(eq / peak - 1, axis=axis)


def paired_bootstrap(ra, rb, rf, B=2000, block=20, seed=0):
    """Circular block bootstrap of Sharpe and max-drawdown differences (a minus b)."""
    ra, rb, rf = map(lambda x: np.asarray(x, float), (ra, rb, rf))
    idx = block_indices(len(ra), B, block, seed)
    A, Bm, F = ra[idx], rb[idx], rf[idx]
    ds = sharpe(A - F) - sharpe(Bm - F)
    dd = maxdd(A) - maxdd(Bm)
    point_s = float(sharpe(ra - rf) - sharpe(rb - rf)); point_d = float(maxdd(ra) - maxdd(rb))
    # basic percentile CIs and a two-sided p from the centred bootstrap distribution
    p_s = float(np.mean(np.abs(ds - ds.mean()) >= abs(point_s)))
    p_d = float(np.mean(np.abs(dd - dd.mean()) >= abs(point_d)))
    return dict(d_sharpe=point_s, d_sharpe_lo=float(np.quantile(ds, .025)), d_sharpe_hi=float(np.quantile(ds, .975)),
                p_sharpe=p_s, d_maxdd=point_d, d_maxdd_lo=float(np.quantile(dd, .025)),
                d_maxdd_hi=float(np.quantile(dd, .975)), p_maxdd=p_d)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


# ------------------------------------------------------------------ VaR backtests
def kupiec(viol, a):
    n = len(viol); x = int(viol.sum()); ph = x / n
    if x in (0, n):
        ll1 = 0.0
    else:
        ll1 = x * np.log(ph) + (n - x) * np.log(1 - ph)
    ll0 = x * np.log(a) + (n - x) * np.log(1 - a)
    return float(chi2.sf(-2 * (ll0 - ll1), 1))


def christoffersen_ind(viol):
    v = np.asarray(viol, int); a, b = v[:-1], v[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(len(a), 1)

    def L(k, pr):
        return k * np.log(pr) if pr > 0 and k > 0 else 0.0
    ll0 = L(n00 + n10, 1 - p) + L(n01 + n11, p)
    ll1 = L(n00, 1 - p01) + L(n01, p01) + L(n10, 1 - p11) + L(n11, p11)
    return float(chi2.sf(max(-2 * (ll0 - ll1), 0), 1))
