"""Hand-checked portfolio accounting: costs, weight drift, cash accrual, metrics."""
import numpy as np
import pandas as pd

from bgarch.portfolio import simulate_portfolio, cash_returns_for_periods
from bgarch.data import load_rates


def test_three_period_hand_calculation():
    a = np.array([.01, -.02, .03]); f = np.array([.001, .001, .001]); w = np.array([.5, .5, 1.0])
    s, c = simulate_portfolio(a, f, w, cost_bps=10)
    # period 1: from cash to 50%: cost = 0.5*10bp
    g1 = .5 * .01 + .5 * .001; n1 = (1 - .5e-3) * (1 + g1) - 1
    d1 = .5 * 1.01 / (1 + g1)                     # drifted risky weight
    g2 = .5 * -.02 + .5 * .001; n2 = (1 - abs(.5 - d1) * 1e-3) * (1 + g2) - 1
    d2 = .5 * .98 / (1 + g2)
    g3 = .03; n3 = (1 - abs(1 - d2) * 1e-3) * (1 + g3) - 1
    assert np.allclose(c.portfolio_return, [n1, n2, n3], rtol=0, atol=1e-15)
    assert np.isclose(s['total_return'], (1 + n1) * (1 + n2) * (1 + n3) - 1)
    assert np.allclose(c.turnover, [.5, abs(.5 - d1), abs(1 - d2)])


def test_explicit_loop_equals_simulator_on_real_segment():
    rng = np.random.default_rng(0)
    a = rng.normal(0, .01, 300); f = np.full(300, 1e-4); w = rng.uniform(0, 1, 300)
    _, c = simulate_portfolio(a, f, w, cost_bps=5)
    wealth, prev, out = 1.0, 0.0, []
    for t in range(300):
        cost = abs(w[t] - prev) * 5e-4; gross = w[t] * a[t] + (1 - w[t]) * f[t]
        net = (1 - cost) * (1 + gross) - 1; out.append(net); prev = w[t] * (1 + a[t]) / (1 + gross)
    assert np.allclose(c.portfolio_return, out, rtol=0, atol=1e-15)


def test_cash_rate_is_lagged_and_act360():
    rates = load_rates()
    d = pd.DatetimeIndex(['2023-03-17', '2023-03-20'])
    used = rates[rates.index < pd.Timestamp('2023-03-17')].iloc[-1]
    assert np.isclose(cash_returns_for_periods(d, rates)[0], used / 100 * 3 / 360)
