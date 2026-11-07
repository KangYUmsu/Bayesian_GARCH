"""Helpers used only by ``walkthrough.ipynb`` (no research logic).

Nothing here estimates a model or changes a result. The module provides:

* ``Checks``          -- a registry of numerical verifications, each tagged with its reproducibility level;
* ``artifact_structure`` -- structural/look-ahead checks of one saved walk-forward job;
* ``report_tables``   -- regenerates the LaTeX tables of ``reports/source/research_report.tex`` from tables computed
                         in the notebook (using ``scripts/make_report_tables.py`` unchanged) and compares them
                         byte for byte with ``reports/source/tables/*.tex``;
* ``png_identical``   -- pixel comparison of a regenerated figure with the report figure;
* ``script_style``    -- run report figure code under matplotlib's default rcParams (as in a script);
* ``fd_hessian``      -- a central finite-difference Hessian for the teaching cells of Section 4.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bgarch.data import log_returns
from .config import ROOT, REPORT_SOURCE

# Reproducibility levels used throughout the notebook and NOTEBOOK_COVERAGE.md
RAW = 'raw data'               # computed directly from prices / rates, or a fresh model fit
FORECASTS = 'saved forecasts'  # recomputed from saved daily forecasts, draws or fit logs
ARTIFACT = 'artifact'          # displayed from a checksum-verified saved artifact (expensive or external)
LEVELS = (RAW, FORECASTS, ARTIFACT)


class Checks:
    """Records every numerical verification so the notebook can end with a summary table.

    ``check(name, ok, detail='', level=FORECASTS)`` prints PASS/FAIL and returns None (so a check on the
    last line of a notebook cell does not echo a value); ``frame()`` returns all checks as a DataFrame.
    """

    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail='', level=FORECASTS):
        self.rows.append(dict(check=name, passed=bool(ok), level=level, detail=detail))
        print(('PASS  ' if ok else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))

    def frame(self):
        return pd.DataFrame(self.rows)


def artifact_structure(store, asset, window, seg, refit_days=63, draws_dir=None):
    """Structural and timing checks of one saved walk-forward job. Returns a dict of booleans/counts.

    Checks that (i) the rows are exactly the trading days of the evaluation segment, (ii) every
    fit window has ``window`` returns and ends strictly before its first forecast row, (iii) refits
    occur every ``refit_days`` rows, (iv) realised returns equal the price data, and (v) all
    forecast columns are finite and variances positive.
    """
    from .config import SEGMENTS
    f = store.daily(asset, window, seg); fits = store.fits(asset, window, seg)
    r = log_returns(asset)
    start, end, _ = SEGMENTS[seg]
    cal = r.index[(r.index >= pd.Timestamp(start)) & ((r.index < pd.Timestamp(end)) if end else True)]
    var_cols = [c for c in f.columns if c.startswith('variance_')]
    num = f.select_dtypes('number')
    refit_rows = np.flatnonzero(f.refit_date.ne(f.refit_date.shift()).to_numpy())
    pos = {d: i for i, d in enumerate(r.index)}
    fit_len = [pos[pd.Timestamp(e)] - pos[pd.Timestamp(s)] + 1 for s, e in zip(fits.fit_start, fits.fit_end)]
    fe = pd.to_datetime(fits.set_index('refit_date').fit_end)
    fit_end_row = pd.to_datetime(f.refit_date.dt.strftime('%Y-%m-%d').map(fe))
    out = dict(asset=asset, window=window, segment=seg, rows=len(f), refits=len(fits),
               dates_equal_calendar=bool(pd.DatetimeIndex(f.date).equals(cal)),
               refit_every_63=bool(np.all(np.diff(refit_rows) == refit_days)) and refit_rows[0] == 0,
               refit_dates_match_log=bool((f.date.iloc[refit_rows].dt.strftime('%Y-%m-%d').to_numpy() == fits.refit_date.to_numpy()).all()),
               fit_windows_have_W_returns=bool(np.all(np.asarray(fit_len) == window)),
               fit_ends_before_forecast=bool((fit_end_row < f.date).all()),
               returns_match_prices=bool(np.allclose(f.return_log.to_numpy(), r.loc[f.date].to_numpy(), rtol=0, atol=1e-15)),
               all_finite=bool(np.isfinite(num.to_numpy(float)).all()),
               variances_positive=bool((f[var_cols] > 0).all().all()))
    if draws_dir is not None:
        z = np.load(Path(draws_dir) / f'{store.name(asset, window, seg)}_draws.npz', allow_pickle=True)
        out['draws_shape_ok'] = bool(z['v'].shape == (300, len(f)) and (pd.to_datetime(z['dates']) == f.date).all())
    return out


def _load_make_tables():
    spec = importlib.util.spec_from_file_location('make_tables', ROOT / 'scripts' / 'make_report_tables.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def report_tables(tables: dict, adequacy_ext: pd.DataFrame, workdir):
    """Regenerate reports/source/tables/*.tex from notebook-computed tables and compare with the report.

    ``tables`` maps CSV names used by quick mode (e.g. 'forecast_pooled_ext') to DataFrames.
    ``scripts/make_report_tables.py`` is imported and run unchanged; only its input/output folders
    are redirected to ``workdir``. Returns a DataFrame with one row per LaTeX table.
    """
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    tdir = work / 'results' / 'quick' / 'tables'; tdir.mkdir(parents=True)
    (work / 'results' / 'adequacy').mkdir(parents=True)
    for name, df in tables.items():
        df.to_csv(tdir / f'{name}.csv', index=False)
    adequacy_ext.to_csv(work / 'results' / 'adequacy' / 'diagnostics_w500_ext.csv', index=False)
    mt = _load_make_tables()
    mt.HERE = work / 'report'; mt.T = tdir; mt.OUT = work / 'report' / 'tables'; mt.OUT.mkdir(parents=True)
    rows = []
    for fn, tex in [(mt.forecast_table, 'forecasts'), (mt.allocation_table, 'allocation'), (mt.paired_table, 'paired'),
                    (mt.exposure_table, 'exposure'), (mt.var_table, 'var'), (mt.fit_table, 'fits'),
                    (mt.adequacy_table, 'adequacy'), (mt.forecast_split_tables, 'forecasts_qlike'),
                    (mt.forecast_split_tables, 'forecasts_logscore'), (mt.forecast_split_tables, 'ratios'),
                    (mt.exposure_compare_table, 'exposure_compare'),
                    (mt.exposure_stage_tables, 'exposure_stage1'), (mt.exposure_stage_tables, 'exposure_stage2')]:
        fn()
        new = (mt.OUT / f'{tex}.tex').read_text(); ref = (REPORT_SOURCE / 'tables' / f'{tex}.tex').read_text()
        rows.append(dict(report_table=f'reports/source/tables/{tex}.tex', identical=new == ref,
                         lines=len(ref.splitlines())))
    return pd.DataFrame(rows)


def png_identical(a, b):
    """True if two PNG files decode to identical pixel arrays."""
    import matplotlib.image as mpimg
    x, y = mpimg.imread(str(a)), mpimg.imread(str(b))
    return x.shape == y.shape and bool(np.array_equal(x, y))


def script_style():
    """Context manager: matplotlib defaults, i.e. the style a command-line run of the figure code sees.

    The notebook sets its own rcParams; report figures must be regenerated without them to be pixel-identical.
    """
    import matplotlib
    return matplotlib.rc_context({k: v for k, v in matplotlib.rcParamsDefault.items() if k not in ('backend', 'interactive')})


def fd_hessian(fun, x, h=1e-4):
    """Central finite-difference Hessian of a scalar function (teaching use only)."""
    x = np.asarray(x, float); d = len(x); H = np.empty((d, d)); E = np.eye(d) * h
    for i in range(d):
        for j in range(i, d):
            H[i, j] = H[j, i] = (fun(x + E[i] + E[j]) - fun(x + E[i] - E[j]) - fun(x - E[i] + E[j]) + fun(x - E[i] - E[j])) / (4 * h * h)
    return H
