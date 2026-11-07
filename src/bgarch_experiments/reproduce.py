"""Two reproduction modes.

  python -m bgarch_experiments.reproduce quick            # verified saved artifacts -> tables + figures (~25 s)
  python -m bgarch_experiments.reproduce full --out DIR   # rerun 16 walk-forward jobs from market data (~70 s, 7 cores)

(`python -m bgarch.reproduce ...` is kept as an alias because the report cites it.)

FULL mode never reads cached forecasts: it requires a new/empty output directory, refits
every rolling window (MLE, MAP, 4-chain MCMC, Laplace, Student-t), then builds tables
and figures from its own output and compares them with the saved results. MCMC draws
are not expected to be bit-identical across platforms (see README).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SAVED_FORECASTS, SAVED_PRIMARY, RESULTS, FIGURES, DATA, ASSETS, SEGMENTS, rel
from . import tables as T
from .artifacts import verify_manifest
from .figures import make_all


def build_tables(store_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    T.use_store(store_dir)
    for seg in T.SEGS:
        t, p = T.forecast_tables(seg)
        t.to_csv(out / f'forecast_{seg}.csv', index=False); p.to_csv(out / f'forecast_pooled_{seg}.csv', index=False)
        T.tail_tables(seg).to_csv(out / f'tail_{seg}.csv', index=False)
        T.info_tables(seg).to_csv(out / f'info_{seg}.csv', index=False)
        s, P, ps, pp = T.allocation_tables(seg)
        s.to_csv(out / f'alloc_summary_{seg}.csv', index=False); P.to_csv(out / f'alloc_pairs_{seg}.csv', index=False)
        ps.to_csv(out / f'alloc_pooled_summary_{seg}.csv', index=False); pp.to_csv(out / f'alloc_pooled_pairs_{seg}.csv', index=False)
    T.fit_summary().to_csv(out / 'fit_summary.csv', index=False)
    return out


def compare_tables(new_dir, ref_dir, atol=1e-10):
    """Max absolute difference per numeric column, for every table present in both folders."""
    rows = []
    for ref in sorted(Path(ref_dir).glob('*.csv')):
        new = Path(new_dir) / ref.name
        if not new.exists():
            continue
        a, b = pd.read_csv(new), pd.read_csv(ref)
        if a.shape != b.shape:
            rows.append(dict(table=ref.name, column='<shape>', max_abs_diff=np.inf)); continue
        for c in b.columns:
            if pd.api.types.is_numeric_dtype(b[c]):
                d = np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) if len(b) else 0.0
                rows.append(dict(table=ref.name, column=c, max_abs_diff=float(d)))
            else:
                rows.append(dict(table=ref.name, column=c, max_abs_diff=0.0 if a[c].astype(str).equals(b[c].astype(str)) else np.inf))
    df = pd.DataFrame(rows)
    df['identical'] = df.max_abs_diff <= atol
    return df


def quick(out=RESULTS / 'quick', figures=FIGURES):
    t0 = time.time()
    n = verify_manifest()
    tabs = build_tables(SAVED_FORECASTS, Path(out) / 'tables')
    figs = make_all(tabs, T.Store(SAVED_FORECASTS), figures)
    cmp = compare_tables(tabs, SAVED_PRIMARY)
    cmp.to_csv(Path(out) / 'verification_vs_research_tables.csv', index=False)
    summary = dict(mode='quick', artifacts_verified=n, tables_dir=rel(tabs), figures=[rel(f) for f in figs],
                   columns_compared=len(cmp), columns_identical=int(cmp.identical.sum()),
                   max_abs_diff=float(cmp.max_abs_diff.max()), seconds=round(time.time() - t0, 1))
    (Path(out) / 'quick_run.json').write_text(json.dumps(summary, indent=2))
    return summary


def _job(args):
    from .walkforward import run
    asset, window, seg, out_dir = args
    start, end, seed = SEGMENTS[seg]
    tag = T.Store(out_dir).name(asset, window, seg)
    return run(asset, window, start=start, end=end, seed=seed, tag=tag, out_dir=out_dir)


def full(out, processes=7):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty. Full mode never reuses earlier outputs; choose a new directory.')
    for a in ASSETS:
        if not (DATA / f'{a}_adjusted.csv').exists():
            raise SystemExit(f'Missing market data {DATA}/{a}_adjusted.csv; see README "Data" to download it.')
    if not (DATA / 'DFF.csv').exists():
        raise SystemExit('Missing data/DFF.csv; see README "Data".')
    fc = out / 'forecasts'; fc.mkdir(parents=True)
    t0 = time.time()
    jobs = [(a, w, s, str(fc)) for s in T.SEGS for a in ASSETS for w in T.WINDOWS]
    with Pool(processes) as pool:
        metas = list(pool.imap_unordered(_job, jobs))
    t_fit = time.time() - t0
    tabs = build_tables(fc, out / 'tables')
    figs = make_all(tabs, T.Store(fc), out / 'figures')
    cmp = compare_tables(tabs, SAVED_PRIMARY)
    cmp.to_csv(out / 'comparison_vs_saved_research_tables.csv', index=False)
    # forecast-level comparison against the saved artifacts
    T.use_store(fc); new = T.STORE
    old = T.Store(SAVED_FORECASTS); rows = []
    classical = ['variance_mle', 'variance_ewma', 'variance_t', 'logscore_mle']
    bayes = ['variance_map', 'variance_bayes_meanplug', 'variance_bayes_integrated', 'variance_bayes_q90']
    for s in T.SEGS:
        for a in ASSETS:
            for w in T.WINDOWS:
                x, y = new.daily(a, w, s), old.daily(a, w, s)
                rec = dict(segment=s, asset=a, window=w, dates_identical=bool(x.date.equals(y.date)))
                for c in classical + bayes:
                    rec[f'maxrel_{c}'] = float(np.max(np.abs(x[c] / y[c] - 1))) if c.startswith('variance') else float(np.max(np.abs(x[c] - y[c])))
                fx, fy = new.fits(a, w, s), old.fits(a, w, s)
                rec['failed_fits_new'] = int((~fx.convergence_pass).sum()); rec['failed_fits_saved'] = int((~fy.convergence_pass).sum())
                rows.append(rec)
    fcmp = pd.DataFrame(rows); fcmp.to_csv(out / 'forecast_comparison_vs_saved.csv', index=False)
    summary = dict(mode='full', output=rel(out), jobs=len(metas), refits=int(sum(m['n_refits'] for m in metas)),
                   failed_convergence=int(sum(m['failed_fits'] for m in metas)), fit_seconds=round(t_fit, 1),
                   total_seconds=round(time.time() - t0, 1), figures=figs,
                   table_columns_identical=int(cmp.identical.sum()), table_columns_compared=len(cmp),
                   max_rel_diff_classical_variance=float(fcmp[[f'maxrel_{c}' for c in classical if c.startswith('variance')]].max().max()),
                   max_rel_diff_bayesian_variance=float(fcmp[[f'maxrel_{c}' for c in bayes]].max().max()))
    (out / 'full_run.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['quick', 'full'])
    ap.add_argument('--out', default=None, help='output directory (required, new or empty, for full mode)')
    ap.add_argument('--processes', type=int, default=7)
    a = ap.parse_args()
    if a.mode == 'quick':
        print(json.dumps(quick(), indent=2))
    else:
        if a.out is None:
            ap.error('full mode needs --out NEW_DIR')
        print(json.dumps(full(a.out, a.processes), indent=2))


if __name__ == '__main__':
    main()
