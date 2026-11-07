"""Milestone 11: ex-post, exposure/risk-matched COUNTERFACTUAL attribution.

Uses saved Milestone 10 equity_curves.csv only, so the actual market GARCH
fits NEVER need to be repeated. The controls use knowledge of entire test
period solely to attribute performance; they are NOT tradable strategies,
and their performance must NOT be reported as out-of-sample implementable.

All counterfactual portfolios are resimulated with the original asset/cash
returns and transaction costs, including cash accrual and weight drift.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .portfolio import simulate_portfolio

REFERENCE = 'bayes_upper90'


def _series(frame, name):
    sub = frame.loc[frame.strategy == name].sort_values('date').reset_index(drop=True)
    if sub.empty:
        raise ValueError(f'Missing strategy {name!r}')
    if sub.date.duplicated().any():
        raise ValueError(f'Duplicate dates for {name}')
    return sub


def _calibrate_multiplier(weights, target, metric, *, returns=None, cash=None,
                          dates=None, cost_bps=5.):
    """Fit c in clip(c*w,0,1) for test-period exposure or realized volatility.

    A true solution may not exist due to the 100% equity cap or a nonmonotonic
    volatility objective. Return a best possible boundary with diagnostics.
    """
    w = np.asarray(weights, float)
    if not (np.isfinite(w).all() and (0 <= w).all() and (w <= 1).all()):
        raise ValueError('weights must be finite in [0,1]')
    if metric not in ('exposure', 'volatility'):
        raise ValueError(metric)
    def value(c):
        candidate = np.clip(c*w, 0., 1.)
        if metric == 'exposure':
            return float(candidate.mean())
        s, _ = simulate_portfolio(returns, cash, candidate, cost_bps=cost_bps, dates=dates)
        return float(s['realized_annual_volatility'])
    if target < 0 or not np.isfinite(target):
        raise ValueError('Invalid target')
    low, high = 0., 1.
    while value(high) < target and high < 1e6:
        high *= 2.
    # If non-monotonic in scaling, choose best numerical match on a coarse grid
    # followed by local bounded scalar optimization rather than assuming a root.
    if metric == 'volatility':
        from scipy.optimize import minimize_scalar
        upper = min(high, 1e6)
        grid = np.linspace(0, upper, 101)
        vals = np.array([abs(value(g) - target) for g in grid])
        ix = int(vals.argmin()); a = grid[max(0,ix-1)]; b = grid[min(100,ix+1)]
        if b > a:
            fit = minimize_scalar(lambda c:(value(c)-target)**2, bounds=(a,b), method='bounded',
                                  options={'xatol':1e-9})
            c = float(fit.x)
        else:
            c = float(grid[ix])
    elif value(high) < target:
        c = high
    else:
        c = float(brentq(lambda x: value(x)-target, low, high, xtol=1e-12))
    return np.clip(c*w,0,1), c, value(c)-target


def attribute_exposure(curves: pd.DataFrame, *, cost_bps=5., reference=REFERENCE,
                       comparison='mle'):
    """Produce summary, dated curves, and paired daily diagnostics.

    All input methods must share identical test dates/asset/cash returns.
    This deliberately does not optimize or learn any trading policy from data.
    """
    required={'date','strategy','asset_return','risk_free_return','weight'}
    missing=required-set(curves)
    if missing: raise ValueError(f'Missing curve columns: {sorted(missing)}')
    if not (0<=cost_bps<10000):raise ValueError('invalid transaction costs')
    base=_series(curves,comparison);risk=_series(curves,reference)
    dates=pd.DatetimeIndex(pd.to_datetime(base.date))
    if len(base)!=len(risk) or not dates.equals(pd.DatetimeIndex(pd.to_datetime(risk.date))):
        raise ValueError('Cannot align reference and comparison dates')
    if len(base)<3:raise ValueError('Need >=3 daily observations')
    a=base.asset_return.to_numpy(float)
    rf=base.risk_free_return.to_numpy(float)
    if not np.allclose(a,risk.asset_return.to_numpy(float),atol=1e-12,rtol=0) or not np.allclose(rf,risk.risk_free_return.to_numpy(float),atol=1e-12,rtol=0):
        raise ValueError('Reference and comparison asset/cash returns differ')
    w_base=base.weight.to_numpy(float);w_risk=risk.weight.to_numpy(float)
    if not (np.isfinite(w_base).all() and np.isfinite(w_risk).all()):raise ValueError('Non-finite weights')
    target_exposure=float(w_risk.mean())
    risk_summary, risk_curve=simulate_portfolio(a,rf,w_risk,cost_bps=cost_bps,dates=dates)
    control_weights={
      'mle_original':w_base,
      'bayes_upper90_original':w_risk,
      'static_mean_exposure_expost':np.full(len(a),target_exposure),
    }
    matched,c_exposure,resid_exposure=_calibrate_multiplier(w_base,target_exposure,'exposure')
    control_weights['mle_exposure_matched_expost']=matched
    # Risk matching is a distinct ex-post sensitivity test, not a substitute
    # for exposure matching or a tradeable solution.
    riskmatched,c_risk,resid_risk=_calibrate_multiplier(
       w_base,risk_summary['realized_annual_volatility'],'volatility',
       returns=a,cash=rf,dates=dates,cost_bps=cost_bps)
    control_weights['mle_volatility_matched_expost']=riskmatched
    records=[];daily=[]
    for name,w in control_weights.items():
        s,c=simulate_portfolio(a,rf,w,cost_bps=cost_bps,dates=dates)
        s={'strategy':name,**s,'hindsight_control':name.endswith('_expost'),
           'target_exposure':target_exposure,
           'exposure_gap':s['mean_exposure']-target_exposure,
           'volatility_gap':s['realized_annual_volatility']-risk_summary['realized_annual_volatility']}
        records.append(s)
        c.insert(1,'strategy',name)
        daily.append(c)
    out=pd.concat(daily,ignore_index=True)
    byname={r['strategy']:r for r in records}
    # Paired daily net excess versus hindsight matched MLE is the direct
    # risk-aware-timing contrast after exposure equalization.
    wide=out.pivot(index='date',columns='strategy',values='portfolio_return')
    asset_series=pd.Series(a,index=dates)
    baseline=wide['mle_exposure_matched_expost']
    delta=wide['bayes_upper90_original']-baseline
    paired=pd.DataFrame({'date':dates,'asset_return':a,'risk_free_return':rf,
                         'bayes_upper90_return':wide['bayes_upper90_original'].to_numpy(),
                         'mle_exposure_matched_return':baseline.to_numpy(),
                         'daily_return_advantage':delta.to_numpy(),
                         'down_market':a<0,
                         'bayes_weight':w_risk,
                         'mle_exposure_matched_weight':matched})
    summary={'reference':reference,'comparison':comparison,
       'method':'EX-POST DIAGNOSTIC ONLY: control calibration sees full test period',
       'cost_bps':cost_bps,'n_days':len(a),
       'reference_exposure':target_exposure,
       'reference_volatility':risk_summary['realized_annual_volatility'],
       'mle_exposure_multiplier':c_exposure,
       'mle_volatility_multiplier':c_risk,
       'exposure_matching_residual':resid_exposure,
       'volatility_matching_residual':resid_risk,
       'mean_daily_paired_net_return_advantage':float(delta.mean()),
       'mean_daily_advantage_when_asset_negative':float(delta[asset_series<0].mean()) if (a<0).any() else None,
       'mean_daily_advantage_when_asset_nonnegative':float(delta[asset_series>=0].mean()) if (a>=0).any() else None,
       'fraction_of_days_outperforming_matched_mle':float((delta>0).mean()),
       'warnings':['Hindsight matching is not deployable: never claim these controls are real-time strategies',
                   'Matched mean exposure does not imply equal turnover or identical exposure timing',
                   'Volatility-matched comparison uses realized test-period volatility',
                   'Daily paired return differences are serially dependent; inferential confidence requires block/bootstrap or HAC analysis']}
    return pd.DataFrame(records), out, paired, summary


def analyze_saved(folder, *, output=None, cost_bps=None):
    """Generate attribution from existing Milestone 10 outputs, without refitting."""
    folder=Path(folder)
    if not (folder/'equity_curves.csv').is_file():raise FileNotFoundError(folder/'equity_curves.csv')
    report_path=folder/'report.json'
    meta=json.loads(report_path.read_text()) if report_path.is_file() else {}
    if cost_bps is None:cost_bps=meta.get('trading_cost_bps',5.)
    raw=pd.read_csv(folder/'equity_curves.csv')
    summary, curves, paired, diagnostics=attribute_exposure(raw,cost_bps=cost_bps)
    diagnostics['asset']=meta.get('asset','unknown')
    diagnostics['window']=meta.get('window','unknown')
    diagnostics['prior']=meta.get('prior','unknown')
    diagnostics['original_convergence_validated']=meta.get('results_validated',None)
    out=Path(output) if output is not None else folder/'exposure_attribution'
    out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/'strategy_summary_matched.csv',index=False)
    curves.to_csv(out/'equity_curves_matched.csv',index=False)
    paired.to_csv(out/'paired_daily_returns.csv',index=False)
    (out/'attribution_report.json').write_text(json.dumps(diagnostics,indent=2))
    return summary,paired,diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',nargs='+',help='Existing Milestone 10 output folders containing equity_curves.csv')
    p.add_argument('--cost-bps',type=float,default=None,help='Override original report cost')
    args=p.parse_args()
    for folder in args.folders:
        summary,_,d=analyze_saved(folder,cost_bps=args.cost_bps)
        print(f'\n{folder}: matched MLE multiplier={d["mle_exposure_multiplier"]:.5f}, n={d["n_days"]}')
        print(summary[['strategy','mean_exposure','realized_annual_volatility',
                       'sharpe_excess_cash','max_drawdown','annualized_return']].to_string(index=False))

if __name__=='__main__':main()
