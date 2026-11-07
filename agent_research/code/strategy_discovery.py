"""Exploratory strategy discovery (secondary task). Reuses saved forecasts and draws only.

Usage: python strategy_discovery.py dev|val|holdout
The holdout stage refuses to run unless results/discovery/validation_decision.json says a
candidate passed the pre-registered validation criterion.
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS, WINDOWS, RES, TARGET, load_daily, load_draws, vol_weight, portfolio, paired_bootstrap  # noqa: E402
from analysis_primary import trailing_mean_known, strategy_weights  # noqa: E402

OUT = RES / 'discovery'; OUT.mkdir(parents=True, exist_ok=True)
PERIODS = {'dev': ('ext', '2017-01-01', '2021-01-01'), 'val': ('ext', '2021-01-01', '2023-01-01'),
           'holdout': ('test', '2023-01-01', '2100-01-01')}
CONFIGS = [('S1', -0.25), ('S1', 0.25), ('S1', 0.5), ('S2', 0.01), ('S2', 0.025),
           ('S3', 0.0), ('S3', 0.10), ('S3_falling', 0.0)]


def trailing_sd_known(x, L=252):
    s = pd.Series(x).shift(1).rolling(L, min_periods=20).std().to_numpy()
    s[np.isnan(s)] = np.nanmedian(s) if np.isfinite(s).any() else 1.0
    return s


def mixture_es(v, alpha):
    """Exact ES of a zero-mean Gaussian scale mixture; v has shape (draws, days)."""
    sig = np.sqrt(v); lo = np.zeros(v.shape[1]); hi = norm.isf(alpha) * sig.max(0)
    for _ in range(60):
        mid = .5 * (lo + hi); sf = norm.sf(mid[None] / sig).mean(0)
        lo = np.where(sf > alpha, mid, lo); hi = np.where(sf <= alpha, mid, hi)
    var = .5 * (lo + hi)
    return (sig * norm.pdf(var[None] / sig)).mean(0) / alpha


def candidate_weights(f, v_draws, kind, par):
    m = f.variance_bayes_integrated.to_numpy(); q = f.variance_bayes_q90.to_numpy()
    w_int = vol_weight(m); w_up = vol_weight(q)
    if kind == 'S1':
        prem = q / m - 1
        mu = trailing_mean_known(prem); sd = trailing_sd_known(prem)
        z = np.clip((prem - mu) / sd, -3, 3)
        return np.minimum(1, TARGET / np.sqrt(252 * m) * np.exp(-par * z))
    if kind == 'S2':
        budget = norm.pdf(norm.isf(par)) / par * TARGET / np.sqrt(252)
        return np.minimum(1, budget / mixture_es(v_draws, par))
    m_lag = pd.Series(m).shift(5).bfill().to_numpy()  # m_{t-5} is known at t
    rising = m > (1 + par) * m_lag
    if kind == 'S3':
        return np.where(rising, w_up, w_int)
    return np.where(m < m_lag, w_up, w_int)  # S3_falling


def run_period(period, configs=CONFIGS, cost_bps=5.0):
    seg, lo, hi = PERIODS[period]
    rows, daily = [], {}
    for a in ASSETS:
        for w in WINDOWS:
            f_all = load_daily(a, w, seg); v_all, d_all = load_draws(a, w, seg)
            assert (d_all == f_all.date).all()
            keep = ((f_all.date >= lo) & (f_all.date < hi)).to_numpy()
            f = f_all[keep].reset_index(drop=True); v = v_all[:, keep]
            base = strategy_weights(f, a, seg)
            W = {k: base[k] for k in ('bayes_integrated', 'bayes_upper90', 'mle', 'mle_const_rt', 'ewma', 'static_target')}
            for kind, par in configs:
                name = f'{kind}_{par:g}'
                wc = candidate_weights(f, v, kind, par)
                W[name] = wc
                k = trailing_mean_known(wc / W['bayes_integrated'])
                W[name + '__ctrl'] = np.minimum(1, k * W['bayes_integrated'])
            res = {k: portfolio(f, x, cost_bps) for k, x in W.items()}
            rf = res['mle'][1].risk_free_return.to_numpy()
            daily.setdefault(w, {})
            for k, (s, c) in res.items():
                rows.append(dict(period=period, asset=a, window=w, strategy=k, **s))
                daily[w].setdefault(k, []).append(c.portfolio_return.to_numpy())
            daily[w].setdefault('_rf', []).append(rf)
    S = pd.DataFrame(rows)
    pooled = []
    for w in WINDOWS:
        rf = np.mean(daily[w]['_rf'], axis=0)
        R = {k: np.mean(v, axis=0) for k, v in daily[w].items() if k != '_rf'}
        for kind, par in configs:
            name = f'{kind}_{par:g}'
            for comp in (name + '__ctrl', 'bayes_integrated', 'bayes_upper90', 'mle_const_rt'):
                bs = paired_bootstrap(R[name], R[comp], rf, B=1000, seed=zlib.crc32(f'{period}{w}{name}{comp}'.encode()))
                pooled.append(dict(period=period, window=w, candidate=name, comparator=comp, **bs))
    return S, pd.DataFrame(pooled)


def per_config_wins(S, name):
    a = S[S.strategy == name].set_index(['asset', 'window']).sharpe_excess_cash
    b = S[S.strategy == name + '__ctrl'].set_index(['asset', 'window']).sharpe_excess_cash
    return int(((a - b) > 0).sum())


if __name__ == '__main__':
    stage = sys.argv[1]; t0 = time.time()
    if stage == 'holdout':
        dec = json.loads((OUT / 'validation_decision.json').read_text())
        if not dec['passed']:
            raise SystemExit('No candidate passed validation: holdout stays untouched (pre-registered stop rule).')
        configs = [tuple(c) for c in dec['passed']]
    elif stage == 'val':
        sel = json.loads((OUT / 'dev_selection.json').read_text())
        configs = [tuple(c) for c in sel['selected']]
    else:
        configs = CONFIGS
    S, P = run_period(stage, configs)
    S.to_csv(OUT / f'{stage}_summary.csv', index=False); P.to_csv(OUT / f'{stage}_pooled_pairs.csv', index=False)
    ctrl = P[P.comparator.str.endswith('__ctrl')].groupby('candidate').d_sharpe.mean()
    if stage == 'dev':
        selected = []
        for kind in ('S1', 'S2', 'S3'):
            names = [f'{k}_{p:g}' for k, p in CONFIGS if k.split('_')[0] == kind]
            best = ctrl.loc[names].idxmax()
            selected.append([c for c in CONFIGS if f'{c[0]}_{c[1]:g}' == best][0])
        (OUT / 'dev_selection.json').write_text(json.dumps({'dev_pooled_dsharpe_vs_ctrl': ctrl.to_dict(),
                                                            'selected': selected}, indent=2))
    if stage == 'val':
        passed = []
        integ_turn = S[S.strategy == 'bayes_integrated'].mean_daily_turnover.mean()
        for kind, par in configs:
            name = f'{kind}_{par:g}'
            pw = P[(P.candidate == name) & (P.comparator == name + '__ctrl')]
            turn = S[S.strategy == name].mean_daily_turnover.mean()
            ok = bool((pw.d_sharpe >= .05).all() and per_config_wins(S, name) >= 6 and turn <= 1.5 * integ_turn)
            if ok:
                passed.append([kind, par])
        (OUT / 'validation_decision.json').write_text(json.dumps({'passed': passed}, indent=2))
    print(P[P.comparator.str.endswith('__ctrl') | (P.comparator == 'bayes_integrated')].round(4).to_string())
    print(f'stage {stage} seconds {time.time() - t0:.1f}')
