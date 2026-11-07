"""Volatility targeting and portfolio accounting for one risky asset plus cash.

Target-volatility rule (annual target ``target``, long-only, no leverage):

    w_t = min(1, target / sqrt(252 h_t))                                  vol_weight

Exposure-matched ("constant haircut") control, using only past rows:

    k_t = mean_{s in [t-L, t)} sqrt(h^point_s / h^cons_s),   k_0 = sqrt(h^point_0 / h^cons_0)
    w_t = min(1, k_t * target / sqrt(252 h^point_t))                      exposure_matched_weights

Timing. The weight computed from forecast row d (information strictly before d) is held over
the close(d) -> close(d+1) return: one close of execution delay.

Accounting (``simulate_portfolio``, IMPORTED UNCHANGED from src.portfolio). Wealth starts at 1 in
cash. Each period: pre-trade weight = previous weight drifted with returns,
w~ = w_{t-1}(1 + R^a)/(1 + R^p); cost fraction = |w_t - w~| * cost_bps/1e4, charged on wealth;
net return = (1 - cost)(1 + R^p) - 1 with R^p = w_t R^a + (1 - w_t) R^cash (simple returns).
Cash (``cash_returns_for_periods``, src.real_backtest): the last DFF value strictly before the
decision date, accrued ACT/360 over calendar days.

vol_weight, portfolio, extra_metrics: COPIED VERBATIM from agent_research/code/common.py.
trailing_mean_known: COPIED VERBATIM from agent_research/code/analysis_primary.py.
The project's set of strategies (static rule, all variance columns) is experiment configuration
and lives in ``bgarch_experiments.strategies``.
"""
import numpy as np
import pandas as pd

from src.portfolio import simulate_portfolio  # noqa: F401
from src.real_backtest import cash_returns_for_periods  # noqa: F401
from .data import load_rates

TARGET = 0.10          # annualised volatility target of the project
COST = 5.0             # one-way transaction cost, basis points
_RATES = None


def rates():
    """DFF (annual percent) loaded once from data/DFF.csv."""
    global _RATES
    if _RATES is None:
        _RATES = load_rates()
    return _RATES


def vol_weight(variance, target=TARGET):
    """w = min(1, target / sqrt(252 h)) elementwise; ``variance`` is the DAILY variance forecast."""
    return np.minimum(1.0, target / np.sqrt(252 * np.asarray(variance, float)))


def portfolio(f, weights, cost_bps=COST):
    """Backtest one asset + cash. Weight in forecast row d (information < d) is held over close(d)->close(d+1).

    f: DataFrame with columns ``date`` (trading dates) and ``return_log`` (log return of row d,
    close(d-1)->close(d)); weights: array of len(f). The last weight is unused. Mirrors
    src.market_uncertainty_backtest.evaluate_strategies exactly.
    Returns (summary dict incl. extra risk metrics, daily DataFrame with len(f)-1 periods).
    """
    w = np.asarray(weights, float)
    asset = np.expm1(f['return_log'].to_numpy(float)[1:])
    cash = cash_returns_for_periods(f['date'], rates())
    s, c = simulate_portfolio(asset, cash, w[:-1], cost_bps=cost_bps, dates=f['date'].iloc[1:])
    s.update(extra_metrics(c.portfolio_return.to_numpy(), c.risk_free_return.to_numpy()))
    return s, c


def extra_metrics(ret, rf):
    """Sortino ratio, downside deviation, daily CVaR95, worst day and mean |63-day vol - TARGET|."""
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


def trailing_mean_known(x, L=252):
    """k_t = mean(x[t-L:t]) using only rows strictly before t; k_0 = x_0 (known at decision)."""
    x = np.asarray(x, float); c = np.r_[0, np.cumsum(x)]
    out = np.empty(len(x)); out[0] = x[0]
    for t in range(1, len(x)):
        lo = max(0, t - L); out[t] = (c[t] - c[lo]) / (t - lo)
    return out


def exposure_matched_weights(h_point, h_conservative, target=TARGET, L=252):
    """Real-time constant-haircut control: min(1, k_t target / sqrt(252 h_point)), k_t = trailing mean of sqrt(h_point/h_cons)."""
    h_point = np.asarray(h_point, float)
    k = trailing_mean_known(np.sqrt(h_point / np.asarray(h_conservative, float)), L)
    return np.minimum(1., k * target / np.sqrt(252 * h_point))
