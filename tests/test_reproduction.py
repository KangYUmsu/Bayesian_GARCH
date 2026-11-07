"""Representative saved results are reproduced (classical exactly; Bayesian on this platform or within MC tolerance)."""
import sys

import numpy as np
import pandas as pd
import pytest

from bgarch_experiments.config import PROJECT, SAVED_FORECASTS, SAVED_PRIMARY, SAVED_M10, RESULTS
from bgarch.data import log_returns
from bgarch.garch import mle_fit, unpack
from bgarch_experiments.tables import Store
from bgarch import student_t
from bgarch_experiments import walkforward
from bgarch_experiments.artifacts import verify_manifest

STORE = Store(SAVED_FORECASTS)


def _agent_walkforward():
    sys.path.insert(0, str(PROJECT / 'agent_research' / 'code'))
    import walkforward as agent  # the original module (numba cache expects this name)
    return agent


def test_saved_artifacts_checksums():
    assert verify_manifest() == 48


def test_fresh_mle_matches_saved_fit():
    r = log_returns('SPY'); st = int(r.index.searchsorted(pd.Timestamp('2023-01-01')))
    u, fit = mle_fit(r.to_numpy()[st - 500:st])
    saved = STORE.fits('SPY', 500, 'test').iloc[0]
    assert np.isclose(-fit.fun, saved.mle_loglik, rtol=0, atol=1e-8)
    assert np.isclose(unpack(u)['persistence'], saved.mle_persistence, rtol=0, atol=1e-10)
    # and the original Milestone 10 diagnostics
    m10 = pd.read_csv(SAVED_M10 / 'SPY_500_moderate/fit_diagnostics.csv').iloc[0]
    assert np.isclose(-fit.fun, -m10.mle_objective, atol=1e-8)


def test_student_t_copy_identical_to_original():
    agent = _agent_walkforward()
    r = log_returns('GLD').to_numpy()[1000:1500]
    u, _ = mle_fit(r)
    a = student_t.t_mle_fit(r, u); b = agent.t_mle_fit(r, u)
    assert np.allclose(a[0], b[0], rtol=0, atol=0) and a[1] == b[1]


def test_walkforward_block_reproduces_saved(tmp_path):
    """First two refit blocks of SPY-500, 2023 segment, rerun from market data."""
    meta = walkforward.run('SPY', 500, start='2023-01-01', end='2023-04-14', seed=2026, tag='blk', out_dir=tmp_path)
    new = pd.read_csv(tmp_path / 'blk_daily.csv.gz'); old = STORE.daily('SPY', 500, 'test').iloc[:len(new)]
    for c in ['variance_mle', 'variance_ewma', 'variance_t']:            # classical: exact
        assert np.allclose(new[c], old[c], rtol=1e-11, atol=0), c
    for c in ['variance_map', 'variance_bayes_meanplug', 'variance_bayes_integrated', 'variance_bayes_q90']:
        rel = np.max(np.abs(new[c] / old[c] - 1))
        assert rel < 0.05, (c, rel)                                       # Monte Carlo tolerance across platforms
    assert meta['n_refits'] == 2


@pytest.mark.parametrize('name', ['forecast_pooled_ext.csv', 'alloc_pooled_pairs_test.csv', 'info_ext.csv'])
def test_quick_mode_tables_equal_research_tables(name):
    q = RESULTS / 'quick' / 'tables' / name
    if not q.exists():
        pytest.skip('run `python -m bgarch.reproduce quick` first')
    a, b = pd.read_csv(q), pd.read_csv(SAVED_PRIMARY / name)
    num = b.select_dtypes('number').columns
    assert np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True)


def test_reported_headline_numbers():
    """Numbers quoted in the course report, recomputed from the research tables."""
    p = pd.read_csv(SAVED_PRIMARY / 'forecast_pooled_ext.csv').set_index(['window', 'contrast'])
    assert round(p.loc[(500, 'ls_mixture_minus_mle'), 'pooled_mean'], 4) == 0.0052
    assert round(p.loc[(500, 'ls_t_minus_mle'), 'pooled_mean'], 3) == 0.028
    assert p.loc[(500, 'qlike_integrated_minus_mle'), 'p'] > 0.05
    pp = pd.read_csv(SAVED_PRIMARY / 'alloc_pooled_pairs_ext.csv')
    sel = pp[(pp.a == 'bayes_upper90') & (pp.b == 'mle_const_rt')]
    assert (sel.d_sharpe_lo < 0).all() and (sel.d_sharpe_hi > 0).all()
