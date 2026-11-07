"""Primary investigation: forecasts, tail risk, information content, allocation.

Reads walk-forward outputs from results/forecasts (no refits) and writes CSVs to
agent_research/results/primary/.
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ASSETS, WINDOWS, RES, ROOT, load_daily, load_fits, vol_weight, portfolio, hac_mean,
                    hac_ols, paired_bootstrap, holm, kupiec, christoffersen_ind, TARGET)  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.exposure_attribution import _calibrate_multiplier  # noqa: E402

OUT = RES / 'primary'; OUT.mkdir(parents=True, exist_ok=True)
SEGS = ['ext', 'test']
VAR_METHODS = ['mle', 'map', 'bayes_meanplug', 'bayes_integrated', 'bayes_q90', 'laplace_integrated', 't', 'ewma']
LS_METHODS = ['mle', 'map', 'bayes_meanplug', 'bayes_mixture', 'laplace_mixture', 't', 'ewma']
VAR_RISK = ['mle', 'bayes_meanplug', 'bayes_mixture', 'laplace_mixture', 't', 'ewma']


def qlike(y2, v):
    return np.log(v) + y2 / v


# ------------------------------------------------------------------ A. forecasts
def forecast_tables(seg, exclude_failed=False):
    rows, pooled = [], {}
    for a in ASSETS:
        for w in WINDOWS:
            f = load_daily(a, w, seg)
            if exclude_failed:
                f = f[f.convergence_pass.astype(bool)].reset_index(drop=True)
            y2 = f.return_log.to_numpy() ** 2
            L = {m: qlike(y2, f[f'variance_{m}'].to_numpy()) for m in VAR_METHODS}
            S = {m: f[f'logscore_{m}'].to_numpy() for m in LS_METHODS}
            rec = dict(asset=a, window=w, n=len(f))
            for m in VAR_METHODS:
                rec[f'qlike_{m}'] = L[m].mean()
            for m in LS_METHODS:
                rec[f'logscore_{m}'] = S[m].mean()
            contrasts = {'qlike_integrated_minus_mle': L['bayes_integrated'] - L['mle'],
                         'qlike_meanplug_minus_mle': L['bayes_meanplug'] - L['mle'],
                         'qlike_integrated_minus_meanplug': L['bayes_integrated'] - L['bayes_meanplug'],
                         'qlike_q90_minus_integrated': L['bayes_q90'] - L['bayes_integrated'],
                         'qlike_laplace_minus_mle': L['laplace_integrated'] - L['mle'],
                         'qlike_t_minus_mle': L['t'] - L['mle'],
                         'qlike_ewma_minus_mle': L['ewma'] - L['mle'],
                         'ls_mixture_minus_mle': S['bayes_mixture'] - S['mle'],
                         'ls_mixture_minus_meanplug': S['bayes_mixture'] - S['bayes_meanplug'],
                         'ls_laplace_minus_mle': S['laplace_mixture'] - S['mle'],
                         'ls_t_minus_mle': S['t'] - S['mle'],
                         'ls_t_minus_mixture': S['t'] - S['bayes_mixture']}
            for k, d in contrasts.items():
                m, se, z, p = hac_mean(d)
                rec[k] = m; rec[k + '_z'] = z; rec[k + '_p'] = p
                if not exclude_failed:
                    pooled.setdefault((w, k), []).append(pd.Series(d, index=f.date))
            rows.append(rec)
    t = pd.DataFrame(rows)
    prow = []
    for (w, k), lst in pooled.items():
        d = pd.concat(lst, axis=1).dropna().mean(axis=1).to_numpy()
        m, se, z, p = hac_mean(d)
        prow.append(dict(window=w, contrast=k, pooled_mean=m, z=z, p=p,
                         n_assets_negative=int(sum(x.mean() < 0 for x in lst))))
    return t, pd.DataFrame(prow)


# ------------------------------------------------------------------ B. tail risk
def tail_tables(seg):
    rows = []
    for a in ASSETS:
        for w in WINDOWS:
            f = load_daily(a, w, seg); loss = -f.return_log.to_numpy()
            base = {}
            for alpha in (0.01, 0.05):
                k = f'{int(alpha * 100):02d}'
                for m in VAR_RISK:
                    var = f[f'var{k}_{m}'].to_numpy(); viol = loss > var
                    pin = np.where(viol, (1 - alpha) * (loss - var), alpha * (var - loss))
                    if m == 'mle':
                        base[k] = pin
                    dm = hac_mean(pin - base[k]) if m != 'mle' else (0, 0, np.nan, np.nan)
                    rows.append(dict(asset=a, window=w, alpha=alpha, method=m, n=len(f), violations=int(viol.sum()),
                                     rate=float(viol.mean()), kupiec_p=kupiec(viol, alpha),
                                     christoffersen_ind_p=christoffersen_ind(viol), pinball=float(pin.mean()),
                                     pinball_minus_mle=float(dm[0]), pinball_dm_p=dm[3]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C. information content
def info_tables(seg):
    rows = []
    for a in ASSETS:
        for w in WINDOWS:
            f = load_daily(a, w, seg)
            m = f.variance_bayes_integrated.to_numpy(); q = f.variance_bayes_q90.to_numpy()
            y = f.return_log.to_numpy(); prem = q / m - 1; z = y * y / m
            cv = f.variance_bayes_sd.to_numpy() / m
            X = np.column_stack([np.ones(len(f)), prem, np.log(m)])
            b1, se1, p1 = hac_ols(np.abs(z - 1), X)
            b2, se2, p2 = hac_ols((z > 4).astype(float), X)
            # standardized premium coefficient: effect of +1 sd premium on outcome
            sd = prem.std()
            zstd = y / np.sqrt(m)
            rows.append(dict(asset=a, window=w, n=len(f), premium_mean=prem.mean(), premium_sd=sd,
                             premium_within_refit_share=float(1 - f.assign(p=prem).groupby('refit_date').p.transform('mean').var() / prem.var()),
                             corr_q90_integrated=float(np.corrcoef(q, m)[0, 1]),
                             abs_err_coef_per_sd=b1[1] * sd, abs_err_p=p1[1],
                             large_shock_coef_per_sd=b2[1] * sd, large_shock_p=p2[1],
                             spearman_prem_abs_err=spearmanr(prem, np.abs(z - 1))[0],
                             mean_param_cv_variance=float(cv.mean()),
                             implied_mixture_excess_kurtosis=float(3 * np.mean(cv ** 2)),
                             empirical_excess_kurtosis_std_returns=float(pd.Series(zstd).kurt()),
                             large_shock_rate=float((z > 4).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ D. allocation
def trailing_mean_known(x, L=252):
    """k_t = mean(x[t-L:t]) using only rows strictly before t; k_0 = x_0 (known at decision)."""
    x = np.asarray(x, float); c = np.r_[0, np.cumsum(x)]
    out = np.empty(len(x)); out[0] = x[0]
    for t in range(1, len(x)):
        lo = max(0, t - L); out[t] = (c[t] - c[lo]) / (t - lo)
    return out


def static_weight(asset, seg):
    from walkforward import load_returns
    r = load_returns(asset); start = '2017-01-01' if seg == 'ext' else '2023-01-01'
    dev = r[r.index < pd.Timestamp(start)].to_numpy()
    return min(1., TARGET / (np.std(dev, ddof=1) * np.sqrt(252)))


def strategy_weights(f, asset, seg):
    v = lambda k: f[f'variance_{k}'].to_numpy()
    W = {'buy_hold': np.ones(len(f)), 'static_target': np.full(len(f), static_weight(asset, seg)),
         'ewma': vol_weight(v('ewma')), 't_garch': vol_weight(v('t')), 'mle': vol_weight(v('mle')),
         'map': vol_weight(v('map')), 'bayes_meanplug': vol_weight(v('bayes_meanplug')),
         'bayes_integrated': vol_weight(v('bayes_integrated')), 'bayes_upper90': vol_weight(v('bayes_q90')),
         'laplace_upper90': vol_weight(v('laplace_q90'))}
    k_mle = trailing_mean_known(np.sqrt(v('mle') / v('bayes_q90')))
    k_mp = trailing_mean_known(np.sqrt(v('bayes_meanplug') / v('bayes_q90')))
    W['mle_const_rt'] = np.minimum(1., k_mle * TARGET / np.sqrt(252 * v('mle')))
    W['meanplug_const_rt'] = np.minimum(1., k_mp * TARGET / np.sqrt(252 * v('bayes_meanplug')))
    return W


PAIRS = [('bayes_upper90', 'mle'), ('bayes_upper90', 'mle_const_rt'), ('bayes_upper90', 'meanplug_const_rt'),
         ('bayes_upper90', 'laplace_upper90'), ('bayes_integrated', 'mle'), ('bayes_meanplug', 'mle'),
         ('bayes_upper90', 'mle_exposure_matched_expost'), ('bayes_upper90', 'mle_vol_matched_expost'),
         ('mle', 'static_target'), ('t_garch', 'mle'), ('ewma', 'mle')]


def allocation_tables(seg, cost_bps=5.0, B=2000):
    summ, pairs, daily = [], [], {}
    for a in ASSETS:
        for w in WINDOWS:
            f = load_daily(a, w, seg)
            W = strategy_weights(f, a, seg)
            res = {k: portfolio(f, x, cost_bps) for k, x in W.items()}
            up = res['bayes_upper90'][0]
            c0 = res['mle'][1]
            asset_r = c0.asset_return.to_numpy(); rf = c0.risk_free_return.to_numpy()
            # ex-post diagnostics (hindsight), applied to the executed weights w[:-1]
            wm = W['mle'][:-1]
            m1, _, r1 = _calibrate_multiplier(wm, up['mean_exposure'], 'exposure')
            m2, _, r2 = _calibrate_multiplier(wm, up['realized_annual_volatility'], 'volatility', returns=asset_r,
                                              cash=rf, dates=c0.date, cost_bps=cost_bps)
            for name, ww in (('mle_exposure_matched_expost', m1), ('mle_vol_matched_expost', m2)):
                res[name] = portfolio(f, np.r_[ww, ww[-1]], cost_bps)
            for k, (s, c) in res.items():
                summ.append(dict(asset=a, window=w, strategy=k, **s))
                daily.setdefault((w, k), []).append(c.set_index('date').portfolio_return.rename(a))
            daily.setdefault((w, '_rf'), []).append(c0.set_index('date').risk_free_return.rename(a))
            for x, y in PAIRS:
                bs = paired_bootstrap(res[x][1].portfolio_return.to_numpy(), res[y][1].portfolio_return.to_numpy(), rf,
                                      B=B, seed=zlib.crc32(f'{a}{w}{x}{y}'.encode()))
                d = res[x][1].portfolio_return.to_numpy() - res[y][1].portfolio_return.to_numpy()
                pairs.append(dict(asset=a, window=w, a=x, b=y, **bs, d_ann_return_hac_p=hac_mean(d)[3],
                                  d_exposure=res[x][0]['mean_exposure'] - res[y][0]['mean_exposure'],
                                  d_turnover=res[x][0]['mean_daily_turnover'] - res[y][0]['mean_daily_turnover'],
                                  d_vol=res[x][0]['realized_annual_volatility'] - res[y][0]['realized_annual_volatility']))
    # pooled equal-weight four-asset sleeve
    pooled_s, pooled_p = [], []
    for w in WINDOWS:
        rf = pd.concat(daily[(w, '_rf')], axis=1).mean(axis=1).to_numpy()
        R = {k: pd.concat(v, axis=1).mean(axis=1).to_numpy() for (ww, k), v in daily.items() if ww == w and k != '_rf'}
        for k, r in R.items():
            ex = r - rf; eq = np.cumprod(1 + r)
            pooled_s.append(dict(window=w, strategy=k, ann_return=float(eq[-1] ** (252 / len(r)) - 1),
                                 ann_vol=float(r.std(ddof=1) * np.sqrt(252)),
                                 sharpe=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)),
                                 max_dd=float(np.min(eq / np.maximum.accumulate(np.r_[1, eq])[1:] - 1))))
        for x, y in PAIRS:
            bs = paired_bootstrap(R[x], R[y], rf, B=B, seed=zlib.crc32(f'{w}{x}{y}'.encode()))
            pooled_p.append(dict(window=w, a=x, b=y, **bs, d_ann_return_hac_p=hac_mean(R[x] - R[y])[3]))
    P = pd.DataFrame(pairs)
    P['p_sharpe_holm_within_pair'] = P.groupby(['a', 'b']).p_sharpe.transform(lambda s: holm(s.to_numpy()))
    return pd.DataFrame(summ), P, pd.DataFrame(pooled_s), pd.DataFrame(pooled_p)


def cost_sensitivity(seg):
    rows = []
    for cost in (0.0, 5.0, 20.0):
        for a in ASSETS:
            for w in WINDOWS:
                f = load_daily(a, w, seg); W = strategy_weights(f, a, seg)
                s = {k: portfolio(f, W[k], cost)[0] for k in ('mle', 'bayes_upper90', 'mle_const_rt', 'bayes_integrated', 'static_target')}
                rows.append(dict(cost_bps=cost, asset=a, window=w,
                                 **{f'sharpe_{k}': v['sharpe_excess_cash'] for k, v in s.items()},
                                 **{f'turnover_{k}': v['mean_daily_turnover'] for k, v in s.items()}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E. MC seeds, prior, reproduction
def seed_and_prior():
    rows = []
    variants = [('test', 'moderate', s) for s in (2026, 11, 12)] + [('ext', 'moderate', s) for s in (102026, 100011)] + \
               [('test', 'broad', 2026), ('ext', 'broad', 102026)]
    for seg, prior, seed in variants:
        for a in ASSETS:
            for w in WINDOWS:
                f = load_daily(a, w, seg, prior=prior, seed=seed)
                fits = load_fits(a, w, seg, prior=prior, seed=seed)
                y2 = f.return_log.to_numpy() ** 2
                W = strategy_weights(f, a, seg)
                s = {k: portfolio(f, W[k])[0] for k in ('mle', 'bayes_integrated', 'bayes_upper90', 'mle_const_rt')}
                rows.append(dict(segment=seg, prior=prior, seed=seed, asset=a, window=w,
                                 failed_fits=int((~fits.convergence_pass).sum()), n_refits=len(fits),
                                 qlike_int_minus_mle=float(np.mean(qlike(y2, f.variance_bayes_integrated) - qlike(y2, f.variance_mle))),
                                 ls_mix_minus_mle=float(np.mean(f.logscore_bayes_mixture - f.logscore_mle)),
                                 sharpe_upper90=s['bayes_upper90']['sharpe_excess_cash'],
                                 d_sharpe_upper90_mle=s['bayes_upper90']['sharpe_excess_cash'] - s['mle']['sharpe_excess_cash'],
                                 d_sharpe_upper90_constrt=s['bayes_upper90']['sharpe_excess_cash'] - s['mle_const_rt']['sharpe_excess_cash'],
                                 d_maxdd_upper90_mle=s['bayes_upper90']['max_drawdown'] - s['mle']['max_drawdown'],
                                 d_exposure_upper90_mle=s['bayes_upper90']['mean_exposure'] - s['mle']['mean_exposure']))
    return pd.DataFrame(rows)


def reproduction():
    rows = []
    for a in ASSETS:
        for w in WINDOWS:
            saved = pd.read_csv(ROOT / f'results/milestone10_market/{a}_{w}_moderate/strategy_summary.csv').set_index('strategy')
            f = load_daily(a, w, 'test'); W = strategy_weights(f, a, 'test')
            for k in ('buy_hold', 'mle', 'map', 'bayes_meanplug', 'bayes_integrated', 'bayes_upper90', 'ewma', 'static_target'):
                s = portfolio(f, W[k])[0]
                rows.append(dict(asset=a, window=w, strategy=k, sharpe_saved=saved.loc[k, 'sharpe_excess_cash'],
                                 sharpe_repro=s['sharpe_excess_cash'], maxdd_saved=saved.loc[k, 'max_drawdown'],
                                 maxdd_repro=s['max_drawdown'], exposure_saved=saved.loc[k, 'mean_exposure'],
                                 exposure_repro=s['mean_exposure']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import time
    t0 = time.time()
    reproduction().to_csv(OUT / 'reproduction_vs_saved_m10.csv', index=False)
    for seg in SEGS:
        t, p = forecast_tables(seg); t.to_csv(OUT / f'forecast_{seg}.csv', index=False); p.to_csv(OUT / f'forecast_pooled_{seg}.csv', index=False)
        t2, _ = forecast_tables(seg, exclude_failed=True); t2.to_csv(OUT / f'forecast_{seg}_converged_only.csv', index=False)
        tail_tables(seg).to_csv(OUT / f'tail_{seg}.csv', index=False)
        info_tables(seg).to_csv(OUT / f'info_{seg}.csv', index=False)
        s, P, ps, pp = allocation_tables(seg)
        s.to_csv(OUT / f'alloc_summary_{seg}.csv', index=False); P.to_csv(OUT / f'alloc_pairs_{seg}.csv', index=False)
        ps.to_csv(OUT / f'alloc_pooled_summary_{seg}.csv', index=False); pp.to_csv(OUT / f'alloc_pooled_pairs_{seg}.csv', index=False)
        cost_sensitivity(seg).to_csv(OUT / f'cost_sensitivity_{seg}.csv', index=False)
    seed_and_prior().to_csv(OUT / 'seed_prior_sensitivity.csv', index=False)
    print(f'done in {time.time() - t0:.1f}s')


# ------------------------------------------------------------------ F. where does the Bayesian forecast gain come from?
def boundary_decomposition():
    """Split QLIKE / log-score differences by whether the refit's MLE sat on the logit(p)=7 box bound."""
    rows = []
    for seg in SEGS:
        for a in ASSETS:
            for w in WINDOWS:
                f = load_daily(a, w, seg); fits = load_fits(a, w, seg)
                atb = f.refit_date.dt.strftime('%Y-%m-%d').map(fits.set_index('refit_date').mle_at_bound).astype(bool)
                y2 = f.return_log.to_numpy() ** 2
                d_shrink = qlike(y2, f.variance_bayes_meanplug) - qlike(y2, f.variance_mle)
                d_integ = qlike(y2, f.variance_bayes_integrated) - qlike(y2, f.variance_bayes_meanplug)
                ls_shrink = f.logscore_bayes_meanplug - f.logscore_mle
                ls_integ = f.logscore_bayes_mixture - f.logscore_bayes_meanplug
                for flag in (True, False):
                    m = (atb == flag).to_numpy()
                    if m.sum() == 0:
                        continue
                    rows.append(dict(segment=seg, asset=a, window=w, mle_at_bound=flag, days=int(m.sum()),
                                     qlike_point_shrinkage=float(d_shrink[m].mean()), qlike_integration=float(d_integ[m].mean()),
                                     ls_point_shrinkage=float(ls_shrink[m].mean()), ls_integration=float(ls_integ[m].mean())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    boundary_decomposition().to_csv(OUT / 'boundary_decomposition.csv', index=False)
