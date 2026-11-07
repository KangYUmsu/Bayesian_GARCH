"""Model-adequacy module: alignment, PIT consistency with saved VaR forecasts, helper correctness."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acf as sm_acf

from bgarch_experiments import adequacy as A
from bgarch.data import log_returns
from bgarch.garch import mle_fit, variance_path


def test_draw_artifacts_checksums():
    assert A.verify_manifest() == 16


@pytest.mark.parametrize('asset', ['SPY', 'QQQ', 'TLT', 'GLD'])
@pytest.mark.parametrize('window', [500, 1000])
def test_panel_alignment_and_pit_matches_saved_var(asset, window):
    f, D = A.load_panel(asset, window)          # raises on any misalignment
    loss = -f.return_log.to_numpy()
    for model, col in [('mle', 'mle'), ('bayes_meanplug', 'bayes_meanplug'), ('t', 't'), ('ewma', 'ewma')]:
        u = A.pit(f, model, D)
        for a, k in ((.01, '01'), (.05, '05')):
            assert np.sum(u < a) == np.sum(loss > f[f'var{k}_{col}'].to_numpy()), (model, a)
    u = A.pit(f, 'bayes_mixture', D)            # mixture VaR was found by bisection: allow boundary ties
    for a, k in ((.01, '01'), (.05, '05')):
        assert abs(np.sum(u < a) - np.sum(loss > f[f'var{k}_bayes_mixture'].to_numpy())) <= 1


def test_first_2017_block_is_strictly_out_of_sample():
    """Fresh MLE on the 500 returns before 2017-01-03 reproduces the saved forecasts of that block."""
    r = log_returns('SPY'); st = int(r.index.searchsorted(pd.Timestamp('2017-01-01')))
    u, _ = mle_fit(r.to_numpy()[st - 500:st])
    f, _ = A.load_panel('SPY', 500, ('ext',))
    h = variance_path(r.to_numpy()[st - 500:st + 63], 500, u)
    assert f.date.iloc[0] == r.index[st]
    assert np.allclose(h, f.variance_mle.to_numpy()[:63], rtol=1e-10, atol=0)


def test_realized_vol_is_backward_looking_and_correct():
    r = np.random.default_rng(1).normal(0, .01, 100)
    rv = A.realized_vol(r, 21)
    assert np.isnan(rv[19]) and np.isclose(rv[20], np.sqrt(252 * np.mean(r[:21] ** 2)))
    r2 = r.copy(); r2[60] = .2
    assert np.allclose(A.realized_vol(r2)[:60], rv[:60], equal_nan=True)   # no dependence on future returns


def test_acf_matches_statsmodels():
    x = np.random.default_rng(2).normal(size=500)
    assert np.allclose(A.acf(x, 10), sm_acf(x, nlags=10, fft=False)[1:], atol=1e-12)


def test_saved_adequacy_tables_reproduce():
    p = A.RESULTS / 'adequacy' / 'diagnostics_w500_ext.csv'
    if not p.exists():
        pytest.skip('run `python -m bgarch.adequacy` first')
    new = A.diagnostics_table(500, 'ext'); old = pd.read_csv(p)
    num = old.select_dtypes('number').columns
    assert np.allclose(new[num].to_numpy(float), old[num].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True)
