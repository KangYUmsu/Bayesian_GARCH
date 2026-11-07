"""Notebook-support helpers (bgarch_experiments.notebook): artifact structure, report-table regeneration, utilities."""
import numpy as np
import pandas as pd

from bgarch_experiments import notebook as N
from bgarch_experiments.config import SAVED_FORECASTS, RESULTS, FIGURES
from bgarch_experiments.tables import Store, SEGS, WINDOWS
from bgarch.data import ASSETS


def test_all_saved_jobs_pass_structure_and_timing_checks():
    st = Store(SAVED_FORECASTS)
    rows = pd.DataFrame([N.artifact_structure(st, a, w, s, draws_dir=SAVED_FORECASTS)
                         for s in SEGS for a in ASSETS for w in WINDOWS])
    flags = rows.select_dtypes(bool)
    assert flags.all().all(), rows[~flags.all(axis=1)]
    assert rows.refits.sum() == 320
    assert set(rows[rows.segment == 'ext'].rows) == {1510} and set(rows[rows.segment == 'test'].rows) == {964}


def test_report_tables_regenerate_identically_from_quick_tables(tmp_path):
    tabs = {p.stem: pd.read_csv(p) for p in (RESULTS / 'quick' / 'tables').glob('*.csv')}
    out = N.report_tables(tabs, pd.read_csv(RESULTS / 'adequacy' / 'diagnostics_w500_ext.csv'), tmp_path / 'rt')
    assert len(out) == 13 and out.identical.all(), out          # 7 original tables + 6 narrative tables


def test_png_identical_and_fd_hessian():
    f = FIGURES / 'fig1_logscore_gains.png'
    assert N.png_identical(f, f)
    assert not N.png_identical(f, FIGURES / 'fig2_sharpe_differences.png')
    A = np.array([[2.0, .5, 0], [.5, 1.0, .2], [0, .2, 3.0]])
    H = N.fd_hessian(lambda x: .5 * x @ A @ x, np.array([.3, -.1, .7]), h=1e-3)
    assert np.allclose(H, A, atol=1e-6)


def test_checks_registry():
    c = N.Checks()
    c(name='a', ok=True); c(name='b', ok=False, level=N.RAW)
    df = c.frame()
    assert list(df.passed) == [True, False] and list(df.level) == [N.FORECASTS, N.RAW]
