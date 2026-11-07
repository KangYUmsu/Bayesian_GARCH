"""Report tables, computed from walk-forward artifacts.

forecast_tables, tail_tables, info_tables, allocation_tables are COPIED VERBATIM from
agent_research/code/analysis_primary.py.  The only difference is where ``load_daily``
reads from: a ``Store`` pointing either at the verified saved artifacts (quick mode)
or at a fresh full-mode rebuild directory.  tests/test_reproduction.py checks that
quick-mode tables equal agent_research/results/primary exactly.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.exposure_attribution import _calibrate_multiplier
from bgarch.data import ASSETS
from bgarch.evaluation import qlike, hac_mean, hac_ols, paired_bootstrap, holm, kupiec, christoffersen_ind  # noqa: F401
from bgarch.portfolio import vol_weight, portfolio  # noqa: F401
from .strategies import strategy_weights  # noqa: F401
from .config import WINDOWS, SEGS, SEG_FILES  # noqa: F401



class Store:
    """Locates ``<ASSET>_<WINDOW>_moderate_s<SEED>_<YEAR>_{daily.csv.gz,fits.csv}``."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def name(self, asset, window, seg):
        year, seed = SEG_FILES[seg]
        return f'{asset}_{window}_moderate_s{seed}_{year}'

    def daily(self, asset, window, seg):
        return pd.read_csv(self.folder / f'{self.name(asset, window, seg)}_daily.csv.gz', parse_dates=['date', 'refit_date'])

    def fits(self, asset, window, seg):
        return pd.read_csv(self.folder / f'{self.name(asset, window, seg)}_fits.csv')


STORE = None


def use_store(store):
    global STORE
    STORE = store if isinstance(store, Store) else Store(store)


def load_daily(a, w, seg):
    return STORE.daily(a, w, seg)


def load_fits(a, w, seg):
    return STORE.fits(a, w, seg)


VAR_METHODS = ['mle', 'map', 'bayes_meanplug', 'bayes_integrated', 'bayes_q90', 'laplace_integrated', 't', 'ewma']
LS_METHODS = ['mle', 'map', 'bayes_meanplug', 'bayes_mixture', 'laplace_mixture', 't', 'ewma']
VAR_RISK = ['mle', 'bayes_meanplug', 'bayes_mixture', 'laplace_mixture', 't', 'ewma']


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



def fit_summary():
    """Convergence and MLE-boundary summary per configuration (course addition)."""
    rows = []
    for seg in SEGS:
        for a in ASSETS:
            for w in WINDOWS:
                d = load_fits(a, w, seg)
                rows.append(dict(segment=seg, asset=a, window=w, refits=len(d),
                                 failed_convergence=int((~d.convergence_pass).sum()),
                                 max_rhat=float(d.max_rhat.max()), min_bulk_ess=float(d.min_bulk_ess.min()),
                                 min_tail_ess=float(d.min_tail_ess.min()), mle_at_bound=int(d.mle_at_bound.sum()),
                                 median_mle_persistence=float(d.mle_persistence.median()),
                                 median_post_mean_persistence=float(d.post_mean_persistence.median()),
                                 median_t_nu=float(d.t_nu.median())))
    return pd.DataFrame(rows)
