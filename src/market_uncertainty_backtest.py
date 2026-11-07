"""Milestone 10: leakage-aware, multi-asset GARCH risk targeting.

For each asset/window/prior, use identical price observations and fit dates.
MLE/MAP/posterior-parameter-plug-in/posterior-integrated forecasts, plus
posterior upper-90% volatility risk targeting. All parameter inference uses
observations STRICTLY before the predicted return date. Execution uses a
further one-close delay: a forecast for return at t determines the desired
weight for return at t+1, after trading at close t. The lag is conservative.

Priors on u=(log V, logit p, logit q) are exactly those in prior_source_study.
All backtest paths are conditional on GARCH model correctness; squared log
returns are noisy proxies for unobservable conditional variance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm
from .garch import mle_fit, encode, unpack
from .prior_source_study import PRIORS, map_fit
from .reliable_mcmc import reliable_bayes_fit, diagnostics
from .real_backtest import (read_price_csv, read_rate_csv, cash_returns_for_periods,
                            ewma_variance)
from .portfolio import constant_cash_returns, simulate_portfolio
from .posterior_tail_risk import gaussian_var_cvar, mixture_var_cvar


def download_assets(tickers, data_dir, start='2013-01-01', end=None):
    """Explicit network operation. Save immutable local inputs for repeatability."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError('Install yfinance first: pip install yfinance') from e
    folder=Path(data_dir);folder.mkdir(parents=True,exist_ok=True)
    stats={}
    for ticker in tickers:
        series=yf.download(ticker,start=start,end=end,auto_adjust=True,
                           progress=False,actions=False,threads=False)
        if series is None or series.empty:
            raise RuntimeError(f'No adjusted prices returned for {ticker}')
        close=series['Close']
        if isinstance(close,pd.DataFrame):
            if close.shape[1]!=1:raise RuntimeError(f'Ambiguous Close columns for {ticker}')
            close=close.iloc[:,0]
        frame=pd.DataFrame({'Date':pd.DatetimeIndex(close.index).strftime('%Y-%m-%d'),
                            'Close':np.asarray(close,dtype=float)})
        if len(frame)<100:raise RuntimeError(f'Insufficient data for {ticker}')
        path=folder/f'{ticker.replace("-","_")}_adjusted.csv'
        frame.to_csv(path,index=False)
        stats[ticker]={'observations':len(frame),'path':str(path)}
    return stats


def _garch_path(local, start, u):
    """Filtered conditional variance, predict r[t] using strictly r[:t].

    local begins at the fit window's first observation. start is the offset
    of the forecast's first return day. This recomputes the variance filter
    using realized past returns while leaving the fitted parameters frozen.
    """
    V=np.exp(u[0]);p=expit(u[1]);q=expit(u[2])
    omega=V*(1-p);alpha=p*q;beta=p*(1-q)
    v=float(V)
    result=np.empty(len(local)-start)
    for t in range(len(local)):
        if t:
            v=omega+alpha*local[t-1]**2+beta*v
        if t>=start:
            result[t-start]=v
    return result


def _mean_u_natural(flat):
    """Average natural (V,p,q) parameters, not transformed u coordinates."""
    V=float(np.exp(flat[:,0]).mean());p=float(expit(flat[:,1]).mean())
    q=float(expit(flat[:,2]).mean())
    return encode(V,p,q)


def _qlike(y,v):
    return np.log(v)+y/v


def run_forecasts(r, dates, *, first_test, window=500, prior_name='moderate',
                  refit_days=63, chains=4, warmup=2500, draws=4000,
                  sampled_draws=300, seed=2026, require_convergence=True, tail_risk=False):
    """Return per-day forecasts and comprehensive fit metadata.

    If any chain fails diagnostics, strict mode refuses to use its forecasts.
    For research diagnostics set require_convergence=False and inspect flags;
    *never* present results with failed fits as validated findings.
    """
    r=np.asarray(r,float);dates=pd.DatetimeIndex(dates)
    if prior_name not in PRIORS:raise ValueError('unknown prior')
    if len(r)!=len(dates) or not np.isfinite(r).all() or not dates.is_monotonic_increasing:
        raise ValueError('invalid returns/dates')
    if not (100<=window<=first_test<len(r)-2):
        raise ValueError('window must be at least 100 and fully precede test start')
    if not (refit_days>=1 and sampled_draws>=2 and chains>=2):
        raise ValueError('invalid sampling/refit settings')
    prior=PRIORS[prior_name];rng=np.random.default_rng(seed)
    frames=[];diagnostics_rows=[]
    for idx,start in enumerate(range(first_test,len(r),refit_days)):
        stop=min(start+refit_days,len(r))
        from_idx=start-window;train=r[from_idx:start]
        mle_u,mle_res=mle_fit(train)
        if not mle_res.success:
            raise RuntimeError(f'MLE optimizer failure {dates[start].date()}')
        map_u,map_res=map_fit(train,mle_u,prior)
        if not map_res.success:
            raise RuntimeError(f'MAP optimizer failure {dates[start].date()}')
        chains_u,meta=reliable_bayes_fit(train,mle_u,n_chains=chains,
            warmup=warmup,draws=draws,seed=seed+idx*173,
            log_density=prior.log_posterior)
        diag,passed=diagnostics(chains_u)
        record={'refit_date':str(dates[start].date()),
                'fit_start':str(dates[from_idx].date()),
                'fit_end':str(dates[start-1].date()),
                'window':window,'prior':prior_name,'convergence_pass':bool(passed),
                'max_rhat':float(diag.r_hat.max()),
                'min_bulk_ess':float(diag.ess_bulk.min()),
                'min_tail_ess':float(diag.ess_tail.min()),
                'mean_acceptance':float(np.mean([m['acceptance'] for m in meta])),
                'mle_persistence':unpack(mle_u)['persistence'],
                'mle_optimizer':str(getattr(mle_res,'method','unknown')),
                'mle_fallback_used':bool(getattr(mle_res,'fallback_used',False)),
                'mle_objective':float(mle_res.fun),
                'mle_attempts':int(getattr(mle_res,'num_attempts',1)),
                'map_persistence':unpack(map_u)['persistence'],
                'map_optimizer':str(getattr(map_res,'method','unknown')),
                'map_fallback_used':bool(getattr(map_res,'fallback_used',False)),
                'map_objective':float(map_res.fun),
                'map_attempts':int(getattr(map_res,'num_attempts',1))}
        diagnostics_rows.append(record)
        if not passed and require_convergence:
            raise RuntimeError(f'Posterior failed convergence at {dates[start].date()}: '
                               f'Rhat {record["max_rhat"]:.4f}, bulk ESS '
                               f'{record["min_bulk_ess"]:.0f}, tail ESS '
                               f'{record["min_tail_ess"]:.0f}. '
                               'Increase sampling or investigate this fit.')
        flat=chains_u.reshape(-1,3)
        ids=rng.choice(len(flat),size=min(sampled_draws,len(flat)),replace=False)
        us=flat[ids]
        local=r[from_idx:stop];offset=start-from_idx
        v_mle=_garch_path(local,offset,mle_u)
        v_map=_garch_path(local,offset,map_u)
        mean_u=_mean_u_natural(flat)
        v_meanplug=_garch_path(local,offset,mean_u)
        v_draws=np.stack([_garch_path(local,offset,u) for u in us])
        v_integrated=v_draws.mean(axis=0)
        # Predictive Gaussian mixture: parameter uncertainty + return shocks.
        # Since normal components share mean zero, mixture tail probabilities
        # come from posterior averaging normal CDFs.
        y=local[offset:]
        pred_log_score=(logsumexp(-.5*(np.log(2*np.pi*v_draws)+y[None,:]**2/v_draws),axis=0)
                        -np.log(len(us)))
        if tail_risk:
            tail_var,tail_es=mixture_var_cvar(v_draws)
            mean_var,mean_es=gaussian_var_cvar(v_meanplug)
            integrated_var,integrated_es=gaussian_var_cvar(v_integrated)
            tail_columns={
              'var95_bayes_mixture':tail_var,
              'cvar95_bayes_mixture':tail_es,
              'var95_bayes_meanplug':mean_var,
              'cvar95_bayes_meanplug':mean_es,
              'var95_bayes_integrated_gaussian':integrated_var,
              'cvar95_bayes_integrated_gaussian':integrated_es,
            }
        else:
            tail_columns={}
        frames.append(pd.DataFrame({
            'date':dates[start:stop], 'return_log':y,'refit_date':dates[start],
            'convergence_pass':passed,
            'variance_mle':v_mle,'variance_map':v_map,
            'variance_bayes_meanplug':v_meanplug,
            'variance_bayes_integrated':v_integrated,
            'variance_bayes_q90':np.quantile(v_draws,.9,axis=0),
            'variance_bayes_q05':np.quantile(v_draws,.05,axis=0),
            'variance_bayes_q95':np.quantile(v_draws,.95,axis=0),
            'bayes_predictive_log_density':pred_log_score,
            'mle_predictive_log_density':norm.logpdf(y,scale=np.sqrt(v_mle)),
            'map_predictive_log_density':norm.logpdf(y,scale=np.sqrt(v_map)),
            **tail_columns,
        }))
    result=pd.concat(frames,ignore_index=True)
    if (result['date'].duplicated().any() or len(result)!=len(r)-first_test):
        raise AssertionError('forecast dates missing or duplicated')
    return result,pd.DataFrame(diagnostics_rows)


def score_forecasts(forecast):
    y=forecast['return_log'].to_numpy()**2
    records=[]
    for name in ('mle','map','bayes_meanplug','bayes_integrated','bayes_q90'):
        v=forecast[f'variance_{name}'].to_numpy()
        if np.any(v<=0):raise AssertionError('nonpositive forecast variance')
        records.append({'method':name,'mse_squared_log_return':float(np.mean((y-v)**2)),
                        'qlike':float(np.mean(_qlike(y,v))),
                        'realized_to_predicted_variance_ratio':float(y.sum()/v.sum())})
    for row in records:
        if row['method']=='mle':
            row['predictive_logscore']=float(forecast.mle_predictive_log_density.mean())
        elif row['method']=='map':
            row['predictive_logscore']=float(forecast.map_predictive_log_density.mean())
        elif row['method']=='bayes_integrated':
            row['predictive_logscore']=float(forecast.bayes_predictive_log_density.mean())
        else:row['predictive_logscore']=np.nan
    return pd.DataFrame(records)


def _allocation_variance(forecast, method):
    return forecast[f'variance_{method}'].to_numpy()


def evaluate_strategies(forecast, *, cash_returns, target=.10,
                        cost_bps=5., static_weights=None, include_ewma=True):
    """Conservative close-t execution: weight based on row[t] forecast,
    applied to close[t]->close[t+1] return. row[t] forecast actually uses
    information through close[t-1], hence no lookahead at close[t].
    """
    if not (0<target and 0<=cost_bps<10000):raise ValueError('invalid target/cost')
    cash_returns=np.asarray(cash_returns,float)
    if len(cash_returns)!=len(forecast)-1:raise ValueError('cash length mismatch')
    weights={
      'buy_hold':np.ones(len(forecast)),
      'mle':np.minimum(1.,target/np.sqrt(252*_allocation_variance(forecast,'mle'))),
      'map':np.minimum(1.,target/np.sqrt(252*_allocation_variance(forecast,'map'))),
      'bayes_meanplug':np.minimum(1.,target/np.sqrt(252*_allocation_variance(forecast,'bayes_meanplug'))),
      'bayes_integrated':np.minimum(1.,target/np.sqrt(252*_allocation_variance(forecast,'bayes_integrated'))),
      'bayes_upper90':np.minimum(1.,target/np.sqrt(252*_allocation_variance(forecast,'bayes_q90'))),
    }
    if static_weights:
        for key,value in static_weights.items():weights[key]=np.full(len(forecast),value)
    if include_ewma and 'variance_ewma' in forecast:
        weights['ewma']=np.minimum(1.,target/np.sqrt(252*forecast['variance_ewma'].to_numpy()))
    asset=np.expm1(forecast['return_log'].to_numpy()[1:])
    rows=[];curves=[]
    for method,full_w in weights.items():
        summ,curve=simulate_portfolio(asset,cash_returns,full_w[:-1],
                                      cost_bps=cost_bps,dates=forecast['date'].iloc[1:])
        rows.append({'strategy':method,**summ})
        curve.insert(1,'strategy',method);curves.append(curve)
    return pd.DataFrame(rows),pd.concat(curves,ignore_index=True)


def run_market(csv_path, *, asset='SPY', window=500,prior_name='moderate',
               test_start='2023-01-01',refit_days=63,chains=4,warmup=2500,
               draws=4000,sampled_draws=300,seed=2026,rf_csv=None,
               flat_rf_annual=None,target=.10,cost_bps=5.,
               output='results/milestone10',require_convergence=True,tail_risk=False):
    """Single-asset real-price experiment. Requires a local saved price CSV."""
    if (rf_csv is None)==(flat_rf_annual is None):
        raise ValueError('Specify exactly one of rf_csv and flat_rf_annual')
    price=read_price_csv(csv_path)
    returns=np.log(price/price.shift(1)).dropna()
    first_test=int(returns.index.searchsorted(pd.Timestamp(test_start)))
    f,fitdiag=run_forecasts(returns.to_numpy(),returns.index,first_test=first_test,
       window=window,prior_name=prior_name,refit_days=refit_days,chains=chains,
       warmup=warmup,draws=draws,sampled_draws=sampled_draws,
       seed=seed,require_convergence=require_convergence,tail_risk=tail_risk)
    f['variance_ewma']=ewma_variance(returns.to_numpy())[first_test:]
    if rf_csv is not None:
        cash=cash_returns_for_periods(f.date,read_rate_csv(rf_csv))
        cash_source=f'Historical DFF proxy (lagged): {rf_csv}'
    else:
        cash=constant_cash_returns(len(f)-1,flat_rf_annual)
        cash_source=f'HYPOTHETICAL constant {flat_rf_annual:.2%} effective annual yield'
    scores=score_forecasts(f)
    # Static benchmark uses ONLY observations prior to the test window.
    development=returns.to_numpy()[:first_test]
    w_static=min(1.,target/(np.std(development,ddof=1)*np.sqrt(252)))
    static_weights={'static_target':w_static}
    # Exposure-matched static benchmark calibrated ONLY during the prior
    # validation interval. Never use mean exposure measured in test.
    if first_test-window>=63:
        validation_start=max(window,first_test-252)
        validation_weights=[]
        raw_r=returns.to_numpy()
        for val_start in range(validation_start,first_test,refit_days):
            val_stop=min(first_test,val_start+refit_days)
            u,fit=mle_fit(raw_r[val_start-window:val_start])
            if not fit.success:
                raise RuntimeError('pre-test exposure calibration MLE failed')
            vf=_garch_path(raw_r[val_start-window:val_stop],window,u)
            validation_weights.extend(np.minimum(1.,target/np.sqrt(252*vf)))
        static_weights['static_exposure_match']=float(np.mean(validation_weights))
    strategies,curves=evaluate_strategies(f,cash_returns=cash,target=target,
                         cost_bps=cost_bps,static_weights=static_weights)
    folder=Path(output)/f'{asset}_{window}_{prior_name}';folder.mkdir(parents=True,exist_ok=True)
    f.to_csv(folder/'daily_forecasts.csv',index=False)
    fitdiag.to_csv(folder/'fit_diagnostics.csv',index=False)
    scores.to_csv(folder/'forecast_scores.csv',index=False)
    strategies.to_csv(folder/'strategy_summary.csv',index=False)
    curves.to_csv(folder/'equity_curves.csv',index=False)
    report={'asset':asset,'input_csv':str(csv_path),'prior':prior_name,
            'prior_mean_u':PRIORS[prior_name].mean_u.tolist(),
            'prior_sd_u':PRIORS[prior_name].sd_u.tolist(),
            'window':window,'test_start':str(f.date.iloc[0].date()),
            'test_end':str(f.date.iloc[-1].date()),'number_of_refits':len(fitdiag),
            'failed_fits':int((~fitdiag.convergence_pass).sum()),
            'cash_source':cash_source,'seed':seed,'target_volatility':target,
            'trading_cost_bps':cost_bps,'static_weights_frozen_pretest':static_weights,'execution':'Signal for return t based on returns <t; execute at close t and hold through t+1',
            'variance_proxy':'Squared close-to-close log return; NOT observed latent variance',
            'warnings':['GARCH Gaussian and stationary assumptions may be misspecified',
                        'DFF proxy is not necessarily an available investment yield',
                        'No realized intraday volatility in this dataset',
                        'Asset-level rather than portfolio covariance targeting',
                        'Posterior draws fixed between quarterly refits; filter updates daily'],
            'results_validated':bool(fitdiag.convergence_pass.all()),
            'tail_risk_computed':bool(tail_risk),
            'methods':strategies.to_dict('records')}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download',action='store_true',help='Download adjusted CSVs via yfinance before processing')
    ap.add_argument('--tickers',nargs='+',default=['SPY','QQQ','TLT','GLD'])
    ap.add_argument('--data-dir',default='data')
    ap.add_argument('--download-start',default='2013-01-01')
    ap.add_argument('--download-end',default=None)
    ap.add_argument('--test-start',default='2023-01-01')
    ap.add_argument('--windows',nargs='+',type=int,default=[500,1000])
    ap.add_argument('--priors',nargs='+',choices=list(PRIORS),default=['moderate'])
    ap.add_argument('--refit-days',type=int,default=63)
    ap.add_argument('--warmup',type=int,default=2500)
    ap.add_argument('--draws',type=int,default=4000)
    ap.add_argument('--chains',type=int,default=4)
    ap.add_argument('--posterior-draws',type=int,default=300)
    ap.add_argument('--cash-csv',default=None,help='Date,DFF historical annual percentages')
    ap.add_argument('--flat-rf-annual',type=float,default=None,help='HYPOTHETICAL annual rate')
    ap.add_argument('--output',default='results/milestone10')
    ap.add_argument('--allow-failed-diagnostics',action='store_true')
    args=ap.parse_args()
    if args.download:
        print('Saved adjusted prices:',download_assets(args.tickers,args.data_dir,
                start=args.download_start,end=args.download_end))
    if (args.cash_csv is None)==(args.flat_rf_annual is None):
        ap.error('Provide exactly one of --cash-csv or --flat-rf-annual')
    all_rows=[]
    for ticker in args.tickers:
        csv=Path(args.data_dir)/f'{ticker.replace("-","_")}_adjusted.csv'
        for w in args.windows:
            for prior in args.priors:
                print(f'Fitting {ticker}, window={w}, prior={prior}',flush=True)
                report=run_market(csv,asset=ticker,window=w,prior_name=prior,
                                  test_start=args.test_start,refit_days=args.refit_days,
                                  warmup=args.warmup,draws=args.draws,chains=args.chains,
                                  sampled_draws=args.posterior_draws,
                                  rf_csv=args.cash_csv,flat_rf_annual=args.flat_rf_annual,
                                  output=args.output,
                                  require_convergence=not args.allow_failed_diagnostics)
                all_rows.append({k:report[k] for k in ('asset','window','prior','number_of_refits','failed_fits','results_validated')})
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(destination/'grid_status.csv',index=False)
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__=='__main__':main()
