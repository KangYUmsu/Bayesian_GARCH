"""No lookahead: forecast row t depends on returns strictly before t; weights are executed one close later."""
import numpy as np
import pandas as pd

from bgarch.data import log_returns
from bgarch.garch import encode, variance_path
from bgarch.ewma import ewma_variance
from bgarch.portfolio import portfolio, vol_weight

r = log_returns('SPY')
RR = r.to_numpy()


def test_variance_filter_uses_only_past_returns():
    u = encode(1e-4, .97, .1)
    base = variance_path(RR[:700], 500, u)
    shocked = RR[:700].copy(); shocked[600] *= 40
    pert = variance_path(shocked, 500, u)
    changed = np.flatnonzero(pert != base) + 500
    assert changed.min() == 601        # return 600 first affects the forecast for day 601


def test_ewma_uses_only_past_returns():
    x = RR[:400].copy(); a = ewma_variance(x); x[300] *= 40; b = ewma_variance(x)
    assert np.all(a[:301] == b[:301]) and a[301] != b[301]


def test_weight_from_row_d_earns_return_after_close_d():
    f = pd.DataFrame({'date': r.index[1000:1100], 'return_log': RR[1000:1100]})
    w = np.full(len(f), .5); w[40] = 1.0               # signal on forecast row 40 only
    _, c = portfolio(f, w)
    # period k covers close(row k) -> close(row k+1): the 1.0 weight appears in period 40, dated row 41
    assert c.weight.iloc[40] == 1.0 and c.date.iloc[40] == f.date.iloc[41]
    assert np.allclose(c.asset_return.to_numpy(), np.expm1(RR[1001:1100]))


def test_vol_weight_cap_and_target():
    assert vol_weight(np.array([(0.05 / np.sqrt(252)) ** 2]))[0] == 1.0      # 5% vol -> capped at 100%
    assert np.isclose(vol_weight(np.array([(0.20 / np.sqrt(252)) ** 2]))[0], 0.5)
