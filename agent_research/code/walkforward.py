"""Walk-forward GARCH forecasts that mirror src.market_uncertainty_backtest.run_forecasts
exactly (same fits, same RNG stream for posterior subsampling) and additionally save:

- the 300 retained posterior parameter draws per refit and the per-draw
  conditional-variance paths (needed for later analysis without refitting);
- a classical parameter-uncertainty analogue: draws from N(u_MLE, H^-1) where H is
  the observed-information (finite-difference Hessian of -loglik) at the MLE;
- a Student-t GARCH(1,1) MLE plug-in (innovation-misspecification benchmark);
- VaR forecasts at 1% and 5% for each method.

Nothing in src/ is modified. Forecast row t uses returns strictly before t.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, gammaln
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.garch import mle_fit, unpack, _ll  # noqa: E402
from src.prior_source_study import PRIORS, map_fit  # noqa: E402
from src.reliable_mcmc import reliable_bayes_fit, diagnostics, local_covariance  # noqa: E402
from src.market_uncertainty_backtest import _garch_path, _mean_u_natural  # noqa: E402
from src.posterior_tail_risk import mixture_var_cvar  # noqa: E402
from src.real_backtest import read_price_csv, ewma_variance  # noqa: E402

OUT = ROOT / 'agent_research' / 'results' / 'forecasts'
MLE_BOUNDS = [(-15, -5), (-7, 7), (-7, 7)]  # identical to src.garch.mle_fit


def load_returns(asset):
    price = read_price_csv(ROOT / 'data' / f'{asset}_adjusted.csv')
    return np.log(price / price.shift(1)).dropna()


# ---------------------------------------------------------------- Student-t GARCH
@njit(cache=True)
def _lg(x):
    return math.lgamma(x)


@njit(cache=True)
def _ll_t(r, u, eta):
    V = np.exp(u[0]); p = 1.0 / (1.0 + np.exp(-u[1])); q = 1.0 / (1.0 + np.exp(-u[2]))
    nu = 2.0 + np.exp(eta)
    omega = V * (1 - p); alpha = p * q; beta = p * (1 - q)
    c = np.log(np.pi * (nu - 2.0))
    v = V; total = 0.0
    for t in range(len(r)):
        if t > 0:
            v = omega + alpha * r[t - 1] ** 2 + beta * v
        if v <= 0 or not np.isfinite(v):
            return -1e300
        total += -0.5 * c - 0.5 * np.log(v) - 0.5 * (nu + 1.0) * np.log(1.0 + r[t] ** 2 / (v * (nu - 2.0)))
    return total + len(r) * (_lg((nu + 1.0) / 2.0) - _lg(nu / 2.0))


def t_mle_fit(r, u_start):
    """Multi-start bounded MLE for standardized-t GARCH(1,1); nu in (2.05, 202)."""
    bounds = MLE_BOUNDS + [(np.log(0.05), np.log(200.0))]
    obj = lambda x: float(-_ll_t(r, np.asarray(x[:3]), float(x[3])))
    best = None
    for eta0 in (np.log(4.0), np.log(8.0), np.log(25.0)):
        x0 = np.r_[np.clip(u_start, [b[0] for b in MLE_BOUNDS], [b[1] for b in MLE_BOUNDS]), eta0]
        fit = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12})
        if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
            best = fit
    return best.x[:3], 2.0 + float(np.exp(best.x[3])), best


# ---------------------------------------------------------------- helpers
def gaussian_var(variance, alpha):
    return norm.isf(alpha) * np.sqrt(variance)


def t_var(variance, nu, alpha):
    return -student_t.ppf(alpha, nu) * np.sqrt(variance * (nu - 2.0) / nu)


def t_logpdf(y, variance, nu):
    s = np.sqrt(variance * (nu - 2.0) / nu)
    return student_t.logpdf(y / s, nu) - np.log(s)


def mixture_logpdf(y, v_draws):
    return logsumexp(-.5 * (np.log(2 * np.pi * v_draws) + y[None, :] ** 2 / v_draws), axis=0) - np.log(len(v_draws))


def run(asset, window, *, start, end=None, prior_name='moderate', seed=2026, refit_days=63,
        chains=4, warmup=2500, draws=4000, sampled_draws=300, tag=None):
    """Walk forward from `start` (first forecast date) up to but excluding `end`.

    The return series is truncated at `end` before any computation, so nothing
    after `end` can enter a fit or a variance filter.
    """
    ret = load_returns(asset)
    if end is not None:
        ret = ret[ret.index < pd.Timestamp(end)]
    r = ret.to_numpy(); dates = ret.index
    first_test = int(dates.searchsorted(pd.Timestamp(start)))
    if first_test < window:
        raise ValueError(f'{asset}: not enough history for window {window} before {start}')
    prior = PRIORS[prior_name]
    rng = np.random.default_rng(seed)               # same stream as run_forecasts
    lap_rng = np.random.default_rng(seed + 7_777_777)  # separate: does not perturb rng
    ewma = ewma_variance(r)
    frames, diag_rows, u_blocks, vdraw_blocks, vlap_blocks = [], [], [], [], []
    t0 = time.time()
    for idx, st in enumerate(range(first_test, len(r), refit_days)):
        stop = min(st + refit_days, len(r))
        from_idx = st - window
        train = r[from_idx:st]
        mle_u, mle_res = mle_fit(train)
        if not mle_res.success:
            raise RuntimeError('MLE failure')
        map_u, map_res = map_fit(train, mle_u, prior)
        if not map_res.success:
            raise RuntimeError('MAP failure')
        chains_u, meta = reliable_bayes_fit(train, mle_u, n_chains=chains, warmup=warmup, draws=draws,
                                            seed=seed + idx * 173, log_density=prior.log_posterior)
        diag, passed = diagnostics(chains_u)
        flat = chains_u.reshape(-1, 3)
        ids = rng.choice(len(flat), size=min(sampled_draws, len(flat)), replace=False)
        us = flat[ids]
        local = r[from_idx:stop]; off = st - from_idx; y = local[off:]
        v_mle = _garch_path(local, off, mle_u)
        v_map = _garch_path(local, off, map_u)
        v_mp = _garch_path(local, off, _mean_u_natural(flat))
        v_draws = np.stack([_garch_path(local, off, u) for u in us])
        # classical asymptotic-normal analogue (observed information at the MLE)
        cov_lap = local_covariance(train, mle_u, log_density=lambda u, rr: float(_ll(rr, np.asarray(u, float))))
        us_lap = lap_rng.multivariate_normal(mle_u, cov_lap, size=sampled_draws)
        v_lap = np.stack([_garch_path(local, off, u) for u in us_lap])
        ok_lap = np.isfinite(v_lap).all(axis=1) & (v_lap > 0).all(axis=1)
        v_lap = v_lap[ok_lap]
        # Student-t GARCH MLE
        tu, nu, tres = t_mle_fit(train, mle_u)
        v_t = _garch_path(local, off, tu)
        rec = {'refit_date': str(dates[st].date()), 'fit_start': str(dates[from_idx].date()),
               'fit_end': str(dates[st - 1].date()), 'window': window, 'prior': prior_name,
               'convergence_pass': bool(passed), 'max_rhat': float(diag.r_hat.max()),
               'min_bulk_ess': float(diag.ess_bulk.min()), 'min_tail_ess': float(diag.ess_tail.min()),
               'mean_acceptance': float(np.mean([m['acceptance'] for m in meta])),
               'mle_persistence': unpack(mle_u)['persistence'], 'mle_logit_p': float(mle_u[1]),
               'mle_at_bound': bool(np.any(np.isclose(mle_u, [b[0] for b in MLE_BOUNDS], atol=1e-6)) or
                                    np.any(np.isclose(mle_u, [b[1] for b in MLE_BOUNDS], atol=1e-6))),
               'mle_fallback_used': bool(getattr(mle_res, 'fallback_used', False)),
               'map_persistence': unpack(map_u)['persistence'],
               'post_mean_persistence': float(expit(flat[:, 1]).mean()),
               'post_sd_persistence': float(expit(flat[:, 1]).std()),
               'post_q05_persistence': float(np.quantile(expit(flat[:, 1]), .05)),
               'post_q95_persistence': float(np.quantile(expit(flat[:, 1]), .95)),
               'lap_sd_logit_p': float(np.sqrt(cov_lap[1, 1])), 'lap_valid_draws': int(ok_lap.sum()),
               't_persistence': unpack(tu)['persistence'], 't_nu': nu, 't_success': bool(tres.success),
               'mle_loglik': float(-mle_res.fun), 't_loglik': float(-tres.fun)}
        diag_rows.append(rec)
        row = {'date': dates[st:stop], 'return_log': y, 'refit_date': dates[st], 'convergence_pass': passed,
               'variance_mle': v_mle, 'variance_map': v_map, 'variance_bayes_meanplug': v_mp,
               'variance_bayes_integrated': v_draws.mean(0),
               'variance_bayes_q90': np.quantile(v_draws, .9, axis=0),
               'variance_bayes_q05': np.quantile(v_draws, .05, axis=0),
               'variance_bayes_q95': np.quantile(v_draws, .95, axis=0),
               'variance_bayes_sd': v_draws.std(0, ddof=1),
               'variance_laplace_integrated': v_lap.mean(0),
               'variance_laplace_q90': np.quantile(v_lap, .9, axis=0),
               'variance_t': v_t, 't_nu': nu, 'variance_ewma': ewma[st:stop],
               'logscore_mle': norm.logpdf(y, scale=np.sqrt(v_mle)),
               'logscore_map': norm.logpdf(y, scale=np.sqrt(v_map)),
               'logscore_bayes_meanplug': norm.logpdf(y, scale=np.sqrt(v_mp)),
               'logscore_bayes_mixture': mixture_logpdf(y, v_draws),
               'logscore_laplace_mixture': mixture_logpdf(y, v_lap),
               'logscore_t': t_logpdf(y, v_t, nu),
               'logscore_ewma': norm.logpdf(y, scale=np.sqrt(ewma[st:stop]))}
        for a in (0.01, 0.05):
            k = f'{int(a * 100):02d}'
            row[f'var{k}_mle'] = gaussian_var(v_mle, a)
            row[f'var{k}_bayes_meanplug'] = gaussian_var(v_mp, a)
            row[f'var{k}_bayes_mixture'] = mixture_var_cvar(v_draws, alpha=a)[0]
            row[f'var{k}_laplace_mixture'] = mixture_var_cvar(v_lap, alpha=a)[0]
            row[f'var{k}_t'] = t_var(v_t, nu, a)
            row[f'var{k}_ewma'] = gaussian_var(ewma[st:stop], a)
        frames.append(pd.DataFrame(row))
        u_blocks.append(us); vdraw_blocks.append(v_draws.astype(np.float32))
    f = pd.concat(frames, ignore_index=True)
    if f.date.duplicated().any() or len(f) != len(r) - first_test:
        raise AssertionError('forecast dates missing/duplicated')
    d = pd.DataFrame(diag_rows)
    tag = tag or f'{asset}_{window}_{prior_name}_s{seed}_{pd.Timestamp(start).year}'
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / f'{tag}_daily.csv.gz', index=False)
    d.to_csv(OUT / f'{tag}_fits.csv', index=False)
    np.savez_compressed(OUT / f'{tag}_draws.npz', u=np.stack(u_blocks),
                        v=np.concatenate(vdraw_blocks, axis=1), dates=f.date.astype(str).to_numpy())
    meta = dict(asset=asset, window=window, prior=prior_name, seed=seed, start=str(start), end=str(end),
                n_days=len(f), n_refits=len(d), failed_fits=int((~d.convergence_pass).sum()),
                seconds=time.time() - t0)
    (OUT / f'{tag}_meta.json').write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', required=True)
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', default=None)
    ap.add_argument('--prior', default='moderate')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.asset, a.window, start=a.start, end=a.end, prior_name=a.prior, seed=a.seed, tag=a.tag)))
