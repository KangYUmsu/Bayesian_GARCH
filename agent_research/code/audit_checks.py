"""Independent audit checks of the existing implementation (no src/ modification).

Writes agent_research/results/audit_checks.json and prints a summary.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import multivariate_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.garch import _ll, mle_fit, encode  # noqa: E402
from src.prior_source_study import PRIORS  # noqa: E402
from src.reliable_mcmc import reliable_bayes_fit, local_covariance  # noqa: E402
from src.market_uncertainty_backtest import _garch_path, evaluate_strategies  # noqa: E402
from src.real_backtest import read_rate_csv, cash_returns_for_periods  # noqa: E402
from src.portfolio import simulate_portfolio  # noqa: E402

sys.path.insert(0, str(ROOT / 'agent_research' / 'code'))
from walkforward import load_returns  # noqa: E402

out = {}


# 1. Likelihood: plain-Python re-implementation in (omega, alpha, beta)
def ll_reference(r, omega, alpha, beta):
    v = omega / (1 - alpha - beta); s = 0.0
    for t, x in enumerate(r):
        if t:
            v = omega + alpha * r[t - 1] ** 2 + beta * v
        s += -0.5 * (math.log(2 * math.pi) + math.log(v) + x * x / v)
    return s


r = load_returns('SPY').to_numpy()[:1000]
errs = []
for V, p, q in [(1e-4, .98, .1), (2e-4, .9, .2), (5e-5, .999, .05)]:
    u = encode(V, p, q)
    errs.append(abs(_ll(r, u) - ll_reference(r, V * (1 - p), p * q, p * (1 - q))))
out['likelihood_max_abs_error_vs_reference'] = float(max(errs))

# 2. MLE boundary: how often is logit(p) at the +7 box bound and what does relaxing it change?
rows = []
for asset in ['SPY', 'QQQ', 'TLT', 'GLD']:
    ret = load_returns(asset); rr = ret.to_numpy(); dates = ret.index
    for window in (500, 1000):
        ft = int(dates.searchsorted(pd.Timestamp('2017-01-01')))
        for st in range(ft, len(rr), 63):
            train = rr[st - window:st]
            u, fit = mle_fit(train)
            if u[1] < 7 - 1e-6:
                continue
            obj = lambda x: -_ll(train, np.asarray(x))
            wide = minimize(obj, u, method='L-BFGS-B', bounds=[(-15, -3), (-7, 20), (-7, 7)])
            nxt_b = _garch_path(rr[st - window:st + 1], window, u)[0]
            nxt_w = _garch_path(rr[st - window:st + 1], window, wide.x)[0]
            rows.append(dict(asset=asset, window=window, refit=str(dates[st].date()),
                             loglik_gain_if_unbounded=float(fit.fun - wide.fun),
                             unbounded_persistence=float(expit(wide.x[1])),
                             next_variance_rel_change=float(nxt_w / nxt_b - 1)))
b = pd.DataFrame(rows)
b.to_csv(ROOT / 'agent_research/results/audit_mle_boundary.csv', index=False)
out['mle_boundary'] = dict(n_refits_at_bound_2017_2026=len(b),
                           max_loglik_gain=float(b.loglik_gain_if_unbounded.max()) if len(b) else 0.,
                           median_abs_next_var_change=float(b.next_variance_rel_change.abs().median()) if len(b) else 0.,
                           max_abs_next_var_change=float(b.next_variance_rel_change.abs().max()) if len(b) else 0.)

# 3. MCMC vs. independent importance sampling (multivariate-t proposal around the mode)
prior = PRIORS['moderate']


def is_check(asset, window, date, n=60000, seed=1):
    ret = load_returns(asset); rr = ret.to_numpy(); st = int(ret.index.searchsorted(pd.Timestamp(date)))
    train = rr[st - window:st]
    mle_u, _ = mle_fit(train)
    chains, _ = reliable_bayes_fit(train, mle_u, seed=99, log_density=prior.log_posterior)
    flat = chains.reshape(-1, 3)
    mode = minimize(lambda x: -prior.log_posterior(x, train), flat.mean(0), method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 4000}).x
    cov = np.cov(flat.T) * 1.5  # deliberately inflated proposal
    prop = multivariate_t(loc=mode, shape=cov, df=5, seed=seed)
    x = prop.rvs(n)
    lw = np.array([prior.log_posterior(xi, train) for xi in x]) - prop.logpdf(x)
    w = np.exp(lw - lw.max()); w /= w.sum()
    ess = 1 / np.sum(w ** 2)
    nv = lambda U: np.array([_garch_path(rr[st - window:st + 1], window, ui)[0] for ui in U])
    v_is = nv(x); v_mc = nv(flat)
    order = np.argsort(v_is); cw = np.cumsum(w[order])
    q90_is = float(v_is[order][np.searchsorted(cw, .9)])
    p_is = float(np.sum(w * expit(x[:, 1])))
    return dict(asset=asset, window=window, date=date, is_ess=float(ess),
                post_mean_p_mcmc=float(expit(flat[:, 1]).mean()), post_mean_p_is=p_is,
                next_var_mean_rel_diff=float(v_mc.mean() / np.sum(w * v_is) - 1),
                next_var_q90_rel_diff=float(np.quantile(v_mc, .9) / q90_is - 1))


out['mcmc_vs_importance_sampling'] = [is_check('SPY', 500, '2023-01-03'), is_check('TLT', 500, '2023-01-03'),
                                      is_check('GLD', 500, '2023-01-03'), is_check('QQQ', 1000, '2020-03-02')]

# 4. Lookahead canaries
ret = load_returns('SPY'); rr = ret.to_numpy().copy()
u = encode(1e-4, .97, .1)
base = _garch_path(rr[:600], 500, u)
rr2 = rr.copy(); rr2[550] *= 50  # shock a single return
pert = _garch_path(rr2[:600], 500, u)
first_changed = int(np.argmax(np.abs(pert - base) > 0)) + 500
out['canary_variance_filter'] = dict(shocked_index=550, first_forecast_index_changed=first_changed,
                                     ok=first_changed == 551)
# portfolio: weight decided from forecast row d earns return of row d+1
f = pd.DataFrame({'date': ret.index[500:600], 'return_log': rr[500:600]})
for k in ['mle', 'map', 'bayes_meanplug', 'bayes_integrated', 'bayes_q90']:
    f[f'variance_{k}'] = base
cash = np.zeros(len(f) - 1)
s1, c1 = evaluate_strategies(f, cash_returns=cash, include_ewma=False)
f2 = f.copy(); f2.loc[60, 'variance_mle'] *= 100  # change ONE forecast row
s2, c2 = evaluate_strategies(f2, cash_returns=cash, include_ewma=False)
w1 = c1[c1.strategy == 'mle'].weight.to_numpy(); w2 = c2[c2.strategy == 'mle'].weight.to_numpy()
changed = np.flatnonzero(w1 != w2)
out['canary_execution_lag'] = dict(changed_forecast_row=60, changed_portfolio_period_rows=changed.tolist(),
                                   portfolio_period_date=str(c1[c1.strategy == 'mle'].date.iloc[changed[0]].date()),
                                   forecast_row_date=str(f.date.iloc[60].date()),
                                   note='forecast row d (info < d) sets the weight for the close(d)->close(d+1) return')

# 5. Cash-rate lag
rates = read_rate_csv(ROOT / 'data/DFF.csv')
d = pd.DatetimeIndex(['2023-03-17', '2023-03-20', '2023-03-21'])
cr = cash_returns_for_periods(d, rates)
out['cash_lag_example'] = dict(period='2023-03-17 -> 2023-03-20 (3 calendar days)', accrual=float(cr[0]),
                               rate_used=float(rates[rates.index < pd.Timestamp('2023-03-17')].iloc[-1]),
                               implied=float(rates[rates.index < pd.Timestamp('2023-03-17')].iloc[-1] / 100 * 3 / 360))

# 6. Replay of saved Milestone 10 equity curves
mx = 0.
for folder in sorted((ROOT / 'results/milestone10_market').glob('*_moderate')):
    cur = pd.read_csv(folder / 'equity_curves.csv')
    for name, g in cur.groupby('strategy'):
        s, c = simulate_portfolio(g.asset_return.to_numpy(), g.risk_free_return.to_numpy(), g.weight.to_numpy(), 5.)
        mx = max(mx, float(np.max(np.abs(c.portfolio_return.to_numpy() - g.portfolio_return.to_numpy()))))
out['m10_equity_replay_max_abs_error'] = mx

# 7. Data sanity
sanity = {}
cal = None
for a in ['SPY', 'QQQ', 'TLT', 'GLD']:
    x = load_returns(a)
    sanity[a] = dict(n=len(x), start=str(x.index[0].date()), end=str(x.index[-1].date()),
                     max_abs_log_return=float(x.abs().max()), date_of_max=str(x.abs().idxmax().date()),
                     max_gap_days=int(np.diff(x.index.values).max() / np.timedelta64(1, 'D')),
                     excess_kurtosis=float(x.kurt()))
    cal = x.index if cal is None else cal
    sanity[a]['same_calendar_as_SPY'] = bool(x.index.equals(cal))
out['data_sanity'] = sanity

(ROOT / 'agent_research/results/audit_checks.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
