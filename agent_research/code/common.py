"""Shared loaders, portfolio metrics and dependence-aware inference for agent_research."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.portfolio import simulate_portfolio  # noqa: E402
from src.real_backtest import read_rate_csv, cash_returns_for_periods  # noqa: E402

FC = ROOT / 'agent_research' / 'results' / 'forecasts'
RES = ROOT / 'agent_research' / 'results'
FIG = ROOT / 'agent_research' / 'figures'
ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD']
WINDOWS = [500, 1000]
SEGMENTS = {'ext': ('2017', 102026), 'test': ('2023', 2026)}  # moderate prior, primary seeds
TARGET = 0.10
COST = 5.0
_RATES = None


def rates():
    global _RATES
    if _RATES is None:
        _RATES = read_rate_csv(ROOT / 'data' / 'DFF.csv')
    return _RATES


def load_daily(asset, window, segment='ext', prior='moderate', seed=None):
    year, s0 = SEGMENTS[segment]
    seed = s0 if seed is None else seed
    f = pd.read_csv(FC / f'{asset}_{window}_{prior}_s{seed}_{year}_daily.csv.gz', parse_dates=['date', 'refit_date'])
    return f


def load_fits(asset, window, segment='ext', prior='moderate', seed=None):
    year, s0 = SEGMENTS[segment]
    seed = s0 if seed is None else seed
    return pd.read_csv(FC / f'{asset}_{window}_{prior}_s{seed}_{year}_fits.csv')


def load_draws(asset, window, segment='ext', prior='moderate', seed=None):
    year, s0 = SEGMENTS[segment]
    seed = s0 if seed is None else seed
    z = np.load(FC / f'{asset}_{window}_{prior}_s{seed}_{year}_draws.npz', allow_pickle=True)
    return z['v'].astype(float), pd.to_datetime(z['dates'])


def vol_weight(variance, target=TARGET):
    return np.minimum(1.0, target / np.sqrt(252 * np.asarray(variance, float)))


def portfolio(f, weights, cost_bps=COST):
    """Weight in forecast row d (information < d) is held over close(d)->close(d+1).

    Mirrors src.market_uncertainty_backtest.evaluate_strategies exactly.
    Returns (summary dict incl. extra risk metrics, daily DataFrame).
    """
    w = np.asarray(weights, float)
    asset = np.expm1(f['return_log'].to_numpy(float)[1:])
    cash = cash_returns_for_periods(f['date'], rates())
    s, c = simulate_portfolio(asset, cash, w[:-1], cost_bps=cost_bps, dates=f['date'].iloc[1:])
    s.update(extra_metrics(c.portfolio_return.to_numpy(), c.risk_free_return.to_numpy()))
    return s, c


def extra_metrics(ret, rf):
    ex = ret - rf
    down = np.minimum(ex, 0)
    dd = np.sqrt(np.mean(down ** 2)) * np.sqrt(252)
    q = np.quantile(ret, .05)
    roll = pd.Series(ret).rolling(63).std() * np.sqrt(252)
    return {'sortino_excess_cash': float(np.mean(ex) * 252 / dd) if dd > 0 else np.nan,
            'downside_dev_annual': float(dd),
            'cvar95_daily': float(-ret[ret <= q].mean()),
            'worst_day': float(ret.min()),
            'mean_abs_vol_target_error_63d': float(np.nanmean(np.abs(roll - TARGET)))}


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
