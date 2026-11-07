"""Milestone 14: one-day lower-tail risk under a Gaussian GARCH posterior mixture.

Conditional on a parameter draw, r ~ N(0,v_draw). Integrating over parameter
samples creates a scale mixture. We compute loss VaR and CVaR analytically
from its mixture survival functions; no additional return Monte Carlo noise.

These are forecasts of one-day LOG-RETURN risk on the risky asset. The
portfolio simulator subsequently uses simple returns and cash yields, so the
CVaR limit is a risk-sizing proxy rather than a guarantee on net portfolio loss.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from .portfolio import simulate_portfolio
from .exposure_attribution import _calibrate_multiplier
from .posterior_uncertainty_ablation import _nw_mean_inference

ALPHA=.05

def gaussian_var_cvar(variance, alpha=ALPHA):
    """Positive one-sided loss VaR/ES for zero-mean normal log returns."""
    v=np.asarray(variance,dtype=float)
    if not (0<alpha<.5) or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('invalid Gaussian variance or tail probability')
    z=norm.isf(alpha)
    return z*np.sqrt(v), norm.pdf(z)*np.sqrt(v)/alpha

def mixture_var_cvar(variance_draws, alpha=ALPHA, iterations=56):
    """Vectorized exact Gaussian scale-mixture VaR/ES for columns of days.

    Input has shape (posterior_draws, forecast_days). Uses deterministic
    bisection for mixture survival=P(return < -VaR)=alpha, then
    ES=mean_m sigma_m*phi(VaR/sigma_m)/alpha.
    """
    v=np.asarray(variance_draws,float)
    if v.ndim!=2 or v.shape[0]<2 or v.shape[1]<1 or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Need positive finite (draw, day) mixture variances')
    if not (0<alpha<.5): raise ValueError('invalid tail probability')
    sig=np.sqrt(v)
    hi=norm.isf(alpha)*sig.max(axis=0)
    lo=np.zeros(v.shape[1])
    for _ in range(iterations):
        mid=(lo+hi)*.5
        survival=norm.sf(mid[None,:]/sig).mean(axis=0)
        lo=np.where(survival>alpha,mid,lo)
        hi=np.where(survival<=alpha,mid,hi)
    var=(lo+hi)*.5
    es=(sig*norm.pdf(var[None,:]/sig)).mean(axis=0)/alpha
    return var,es

def _required_frame(f):
    columns=['date','return_log','convergence_pass',
             'cvar95_bayes_mixture','var95_bayes_mixture',
             'cvar95_bayes_meanplug','var95_bayes_meanplug',
             'cvar95_bayes_integrated_gaussian','var95_bayes_integrated_gaussian']
    missing=set(columns)-set(f)
    if missing:raise ValueError(f'Missing exact Milestone 14 forecasts: {sorted(missing)}. Rerun Notebook 14 with tail_risk=True; saved M10 CSV alone is insufficient.')
    f=f.copy();f['date']=pd.to_datetime(f['date'])
    if f['date'].duplicated().any() or not f['date'].is_monotonic_increasing:raise ValueError('Invalid forecast dates')
    if not f['convergence_pass'].astype(bool).all():raise ValueError('Posterior convergence not validated')
    for c in columns[3:]:
        v=f[c].to_numpy(float)
        if not np.isfinite(v).all() or (v<=0).any():raise ValueError(f'Invalid {c}')
    return f

def evaluate_tail_forecasts(f):
    """Proper VaR quantile loss plus descriptive ES exceedance diagnostics."""
    f=_required_frame(f)
    # Loss = -log return. Forecast row predicts return at that same date.
    loss=-f.return_log.to_numpy(float)
    rows=[]
    for name,suffix in [('bayes_mixture','bayes_mixture'),
                         ('bayes_meanplug','bayes_meanplug'),
                         ('bayes_integrated_gaussian','bayes_integrated_gaussian')]:
        var=f['var95_'+suffix].to_numpy(float)
        es=f['cvar95_'+suffix].to_numpy(float)
        exceed=loss>var
        # Quantile loss for alpha-upper-tail loss VaR = 95th percentile.
        quantile_loss=np.where(exceed,(1-ALPHA)*(loss-var),ALPHA*(var-loss))
        rows.append(dict(method=name,mean_pinball=float(np.mean(quantile_loss)),
            violation_rate=float(exceed.mean()),target_rate=ALPHA,
            n_violations=int(exceed.sum()),n_days=len(loss),
            mean_es_when_exceeded=float(np.mean(es[exceed])) if exceed.any() else np.nan,
            mean_loss_when_exceeded=float(np.mean(loss[exceed])) if exceed.any() else np.nan,
            mean_predicted_es=float(np.mean(es)),mean_var=float(np.mean(var))))
    return pd.DataFrame(rows)

def evaluate_tail_portfolios(f, cash_returns, *, target_volatility=.10,cost_bps=5.,
                             exposure_match=True):
    """Three ex-ante common-CVaR-budget rules, plus EX-POST exposure controls.

    row t forecast drives weights earned during period t->t+1, matching M10.
    Returns risk/strategy summary, all dated curves, and paired daily inference.
    """
    f=_required_frame(f)
    a=np.expm1(f['return_log'].to_numpy(float)[1:])
    rf=np.asarray(cash_returns,float)
    if len(rf)!=len(a):raise ValueError('cash-return length mismatch')
    if target_volatility<=0:raise ValueError('invalid target volatility')
    # Pick common benchmark limit: normal CVaR at daily 10% annual vol.
    _,risk_budget=gaussian_var_cvar((target_volatility/np.sqrt(252))**2)
    risk_budget=float(risk_budget)
    forecasts={
       'bayes_mixture':f.cvar95_bayes_mixture.to_numpy(float)[:-1],
       'bayes_meanplug':f.cvar95_bayes_meanplug.to_numpy(float)[:-1],
       'bayes_integrated_gaussian':f.cvar95_bayes_integrated_gaussian.to_numpy(float)[:-1]}
    w={name:np.minimum(1.,risk_budget/cvar) for name,cvar in forecasts.items()}
    records=[]; curves=[]
    if exposure_match:
        target_mean=float(w['bayes_mixture'].mean())
        for baseline in ['bayes_meanplug','bayes_integrated_gaussian']:
            match,scale,residual=_calibrate_multiplier(w[baseline],target_mean,'exposure')
            if abs(residual)>1e-7:raise RuntimeError('Exposure control infeasible')
            w[baseline+'_exposure_matched_expost']=match
    for name,weight in w.items():
        s,c=simulate_portfolio(a,rf,weight,cost_bps=cost_bps,dates=f.date.iloc[1:])
        records.append(dict(strategy=name,expost_control=name.endswith('_expost'),**s))
        c.insert(1,'strategy',name);curves.append(c)
    table=pd.DataFrame(records); curves=pd.concat(curves,ignore_index=True)
    wide=curves.pivot(index='date',columns='strategy',values='portfolio_return')
    tests=[]
    for baseline in ['bayes_meanplug','bayes_integrated_gaussian']:
        for suffix in (['','_exposure_matched_expost'] if exposure_match else ['']):
            key=baseline+suffix
            diff=(wide['bayes_mixture']-wide[key]).to_numpy(float)
            h=_nw_mean_inference(diff)
            tests.append(dict(comparison=f'bayes_mixture_minus_{key}',**h))
    return table,curves,pd.DataFrame(tests),risk_budget

def analyze_saved(folder, *, output=None):
    """Reuse M14 per-day forecasts and reconstructed historical cash returns."""
    folder=Path(folder)
    f=_required_frame(pd.read_csv(folder/'daily_forecasts.csv'))
    curves=pd.read_csv(folder/'equity_curves.csv')
    meta=json.loads((folder/'report.json').read_text())
    # Use M10 cash-return recording, not an approximation or hypothetical rate.
    old=curves[curves.strategy=='buy_hold'].sort_values('date')
    if len(old)!=len(f)-1 or not np.array_equal(pd.to_datetime(old.date).to_numpy(),f.date.iloc[1:].to_numpy()):
        raise ValueError('M10 cash path dates inconsistent with tail forecasts')
    if not np.allclose(old.asset_return.to_numpy(float),np.expm1(f.return_log.to_numpy(float)[1:]),atol=1e-12,rtol=0):
        raise ValueError('M10 returns mismatch')
    risk=evaluate_tail_forecasts(f)
    summary,paths,paired,budget=evaluate_tail_portfolios(f,old.risk_free_return.to_numpy(float),
        target_volatility=meta['target_volatility'],cost_bps=meta['trading_cost_bps'])
    out=Path(output) if output is not None else folder/'tail_risk_attribution'
    out.mkdir(parents=True,exist_ok=True)
    risk.to_csv(out/'tail_forecast_scores.csv',index=False)
    summary.to_csv(out/'tail_strategy_summary.csv',index=False)
    paths.to_csv(out/'tail_equity_curves.csv',index=False)
    paired.to_csv(out/'tail_paired_hac.csv',index=False)
    info=dict(asset=meta['asset'],window=meta['window'],prior=meta['prior'],
              convergence_validated=meta.get('results_validated'),
              risk_budget_daily_log_return=budget,
              note='Exposure matching ex-post; the three original CVaR rules use a common predeclared risk budget. Gaussian scale mixture conditional on fitted Gaussian GARCH. VaR calibration and ES exceedance are descriptive; HAC inference is only for paired mean net returns.')
    (out/'tail_report.json').write_text(json.dumps(info,indent=2))
    for table in (risk,summary,paired):table.insert(0,'experiment',folder.name)
    return risk,summary,paired
