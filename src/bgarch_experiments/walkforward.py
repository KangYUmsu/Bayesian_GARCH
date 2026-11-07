"""Walk-forward forecasting engine (FULL reproduction mode).

COPIED from agent_research/code/walkforward.run. Changes: (i) the output directory is an explicit
argument (``out_dir``) instead of the fixed
agent_research/results/forecasts, so a full rebuild can never overwrite or silently
reuse the verified saved artifacts; (ii) the numerical functions are called through their bgarch
names (bgarch.garch / bayes / forecast), which are the same validated objects, and the Laplace
covariance and predictive summaries use bgarch.bayes.laplace_cov / bgarch.forecast.predictive_summary
(identical arithmetic; tests/test_reproduction.py and full mode check bit-identical output).  agent_research/code/walkforward.run itself
mirrors src.market_uncertainty_backtest.run_forecasts (identical fits and RNG stream)
and adds saved draws, a Laplace (classical) analogue, Student-t GARCH and VaR columns.

Timing: row t of the output is the forecast for return r_t, computed from r_{<t}.
Parameters are refitted every 63 sessions on the previous ``window`` returns; between
refits the parameters are frozen and only the variance filter is updated daily.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

# All numerics come from the reusable bgarch interfaces (which re-export the validated src/ code).
from bgarch.data import log_returns as load_returns
from bgarch.garch import mle_fit, unpack, variance_path as _garch_path, MLE_BOUNDS
from bgarch.bayes import PRIORS, map_fit, reliable_bayes_fit, diagnostics, laplace_cov, posterior_mean_u as _mean_u_natural
from bgarch.forecast import mixture_logpdf, mixture_var_cvar, predictive_summary
from bgarch.ewma import ewma_variance
from bgarch.student_t import t_mle_fit, gaussian_var, t_var, t_logpdf
from .config import SEGMENTS  # noqa: F401  (segment -> (start, end, seed) of the primary jobs)


def run(asset, window, *, start, end=None, prior_name='moderate', seed=2026, refit_days=63,
        chains=4, warmup=2500, draws=4000, sampled_draws=300, tag=None, out_dir):
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
        cov_lap = laplace_cov(train, mle_u)
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
        ps = predictive_summary(v_draws)
        row = {'date': dates[st:stop], 'return_log': y, 'refit_date': dates[st], 'convergence_pass': passed,
               'variance_mle': v_mle, 'variance_map': v_map, 'variance_bayes_meanplug': v_mp,
               'variance_bayes_integrated': ps['integrated'],
               'variance_bayes_q90': ps['q90'],
               'variance_bayes_q05': ps['q05'],
               'variance_bayes_q95': ps['q95'],
               'variance_bayes_sd': ps['sd'],
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
    OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / f'{tag}_daily.csv.gz', index=False)
    d.to_csv(OUT / f'{tag}_fits.csv', index=False)
    np.savez_compressed(OUT / f'{tag}_draws.npz', u=np.stack(u_blocks),
                        v=np.concatenate(vdraw_blocks, axis=1), dates=f.date.astype(str).to_numpy())
    meta = dict(asset=asset, window=window, prior=prior_name, seed=seed, start=str(start), end=str(end),
                n_days=len(f), n_refits=len(d), failed_fits=int((~d.convergence_pass).sum()),
                seconds=time.time() - t0)
    (OUT / f'{tag}_meta.json').write_text(json.dumps(meta, indent=2))
    return meta
