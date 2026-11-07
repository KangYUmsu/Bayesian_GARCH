"""Real-data walk-forward SPY GARCH study (or explicitly labeled demo CSV).

Uses daily adjusted close-to-close LOG returns. A forecast issued after close t
is executed at close t+1 and earns close(t+1)->close(t+2) simple returns.
This one-close delay avoids using the price that generates a signal to fill
at the same close. Closing auction fills remain an approximation.

Refit once every refit_days trading observations, using only prior returns;
Bayesian draws are frozen between refits, while conditional variance filters
incorporate newly observed returns daily. Priors are Gaussian in u-space.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from .garch import mle_fit, bayes_fit, rhat, unpack
from .synthetic_backtest import _variance_forecast_path
from .portfolio import simulate_portfolio, constant_cash_returns


def read_price_csv(path):
    """CSV Date,Close OR Date,Adj Close. Reject duplicates/nonfinite prices."""
    df = pd.read_csv(path)
    cols = {str(c).strip().lower():c for c in df.columns}
    dcol = cols.get('date')
    pcol = cols.get('adj close', cols.get('close'))
    if dcol is None or pcol is None:
        raise ValueError('CSV needs Date and Close (or Adj Close) columns')
    dates = pd.to_datetime(df[dcol], errors='raise')
    prices = pd.to_numeric(df[pcol], errors='raise')
    if dates.isna().any() or prices.isna().any() or (prices<=0).any():
        raise ValueError('invalid dates or nonpositive/nonfinite prices')
    out = pd.Series(prices.to_numpy(dtype=float), index=pd.DatetimeIndex(dates),name='price')
    if not np.isfinite(out.to_numpy()).all() or out.index.has_duplicates:
        raise ValueError('duplicate dates or nonfinite prices')
    if not out.index.is_monotonic_increasing:
        raise ValueError('CSV dates must be sorted ascending')
    if len(out)<100:
        raise ValueError('at least 100 prices required')
    return out


def download_spy(path, start='2015-01-01', end=None):
    """Requires internet and optional yfinance; writes adjusted daily Close CSV.

    Auto-adjusted Close includes dividend/split adjustment conventions of Yahoo.
    Download is not executed implicitly by the backtest.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install yfinance: pip install yfinance') from exc
    raw = yf.download('SPY', start=start, end=end, auto_adjust=True,
                      progress=False, actions=False)
    if raw is None or raw.empty:
        raise RuntimeError('No SPY observations returned; check internet/data provider')
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]
    out = pd.DataFrame({'Date':pd.DatetimeIndex(close.index).strftime('%Y-%m-%d'),
                        'Close':np.asarray(close, dtype=float)})
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest,index=False)
    return len(out)


@njit(cache=True)
def _forecast_future(r, train_end, us, stop):
    """For each draw in us, produce var forecast at t for train_end<=t<stop;
    t=next return index, using only returns r[:t]. No future observations.
    """
    result=np.empty((len(us),stop-train_end))
    for i in range(len(us)):
        u=us[i]
        V=np.exp(u[0]);p=1.0/(1.0+np.exp(-u[1]));q=1.0/(1.0+np.exp(-u[2]))
        omega=V*(1-p);alpha=p*q;beta=p*(1-q)
        v=V
        for t in range(stop):
            if t>0:
                v=omega+alpha*r[t-1]*r[t-1]+beta*v
            if t>=train_end:
                result[i,t-train_end]=v
    return result


def fit_predict_real(r,dates,first_test,refit_days=63,train_window=1000,
                     n_chains=4,n_steps=4000,burn=1500,thin=5,
                     posterior_draws=250,seed=2026):
    """Walk forward, reestimating only at t; train = r[max(0,t-window):t].
    Initial volatility filter starts at training window's beginning.
    Forecast for return day t generated at close t-1; trade at close t;
    strategy earns return on day t+1. No final-day strategy position.
    """
    r=np.asarray(r,dtype=float)
    dates=pd.DatetimeIndex(dates)
    if len(r)!=len(dates) or first_test<100 or first_test>=len(r)-1:
        raise ValueError('invalid dates or first_test; need >=100 training and >=2 test returns')
    if not np.isfinite(r).all() or refit_days<1 or train_window<100 or posterior_draws<1:
        raise ValueError('invalid refit/window settings')
    forecasts=[]; diagnostics=[]
    rng=np.random.default_rng(seed)
    for round_idx,start in enumerate(range(first_test,len(r),refit_days)):
        stop=min(start+refit_days,len(r))
        from_idx=max(0,start-train_window)
        train=r[from_idx:start]
        mle_u,opt=mle_fit(train)
        chains,acc=bayes_fit(train,mle_u,n_chains=n_chains,n_steps=n_steps,
                             burn=burn,thin=thin,seed=seed+round_idx*101)
        all_u=chains.reshape(-1,3)
        sampled=all_u[rng.choice(len(all_u),size=min(posterior_draws,len(all_u)),replace=False)]
        local=r[from_idx:stop]
        local_start=start-from_idx
        pred_mle=_forecast_future(local,local_start,np.asarray([mle_u]),len(local))[0]
        pred_bayes=_forecast_future(local,local_start,sampled,len(local))
        values={
            'date':dates[start:stop],
            'return_log':r[start:stop],
            'variance_mle':pred_mle,
            'variance_bayes_mean':pred_bayes.mean(axis=0),
            'variance_bayes_q05':np.quantile(pred_bayes,.05,axis=0),
            'variance_bayes_q95':np.quantile(pred_bayes,.95,axis=0),
            'volatility_bayes_q90_ann':np.quantile(np.sqrt(252*pred_bayes),.9,axis=0),
            'refit_start':dates[start],
        }
        forecasts.append(pd.DataFrame(values))
        diagnostics.append({'date':str(dates[start].date()),'train_start':str(dates[from_idx].date()),
                            'train_end':str(dates[start-1].date()),
                            'n_train':len(train), 'mle_success':bool(opt.success),
                            'mle_params':unpack(mle_u),
                            'acceptance':acc.tolist(),
                            'split_rhat':rhat(chains).tolist()})
    return pd.concat(forecasts,ignore_index=True),diagnostics


def read_rate_csv(path):
    """Read Date,DFF (or Date,Rate) of annualized percentages.

    DFF is the effective federal funds rate, a proxy for remunerated cash, not
    necessarily an available retail cash yield. Missing values '.' are dropped.
    """
    df=pd.read_csv(path)
    cols={str(c).strip().lower():c for c in df.columns}
    dcol=cols.get('date');rcol=cols.get('dff',cols.get('rate'))
    if dcol is None or rcol is None:
        raise ValueError('rate CSV requires Date,DFF or Date,Rate')
    dates=pd.to_datetime(df[dcol],errors='raise')
    vals=pd.to_numeric(df[rcol].replace('.',np.nan),errors='coerce')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('rate dates must be valid unique ascending dates')
    s=pd.Series(vals.to_numpy(dtype=float),index=pd.DatetimeIndex(dates)).dropna()
    if s.empty or not np.isfinite(s).all() or (s < -100).any():
        raise ValueError('invalid rate series')
    return s


def download_fred_dff(path, start='2014-01-01'):
    """Download daily effective federal funds rate as annualized percent, FRED DFF.

    Network access required. The historical series is stored locally for repeatability.
    Backtest applies a strict previous-date lag when the rate becomes usable.
    """
    url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF'
    df=pd.read_csv(url)
    datecol='DATE' if 'DATE' in df.columns else 'observation_date'
    df=df.rename(columns={datecol:'Date'})
    df=df[pd.to_datetime(df['Date'])>=pd.Timestamp(start)]
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    df[['Date','DFF']].to_csv(dest,index=False)
    return len(read_rate_csv(dest))


def cash_returns_for_periods(period_end_dates, rate_series):
    """Map historical DFF to cash returns with conservative publication lag.

    Each row covers close(previous session)->close(period_end). For a row ending
    on date d, only the most recently dated DFF observation STRICTLY BEFORE the
    previous session's close date is used. This avoids same-day look-ahead and
    assumes prior-day DFF is known by the previous close. Accrue ACT/360 simple
    interest across the actual calendar days (including weekends). The initial
    previous session must be passed as the first date in period_end_dates, so
    output has length len(dates)-1.
    """
    dates=pd.DatetimeIndex(period_end_dates)
    if len(dates)<2 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('period dates must be unique and ascending')
    rates=rate_series.sort_index()
    if len(rates)==0:
        raise ValueError('empty rates')
    decisions=dates[:-1]
    source_dates=rates.index.to_numpy(dtype='datetime64[ns]')
    loc=np.searchsorted(source_dates,decisions.to_numpy(dtype='datetime64[ns]'),side='left')-1
    if np.any(loc<0):
        raise ValueError('risk-free history must start before the first trade decision')
    y=rates.to_numpy()[loc]/100.0
    days=(dates[1:]-dates[:-1]).days.to_numpy()
    out=y*days/360.0
    if not np.isfinite(out).all() or (out <= -1).any():
        raise ValueError('invalid daily accrued rates')
    return out


def calibrate_static_benchmarks(returns, dates, first_test, target=.10,
                                validation_start=None, refit_days=63,
                                train_window=1000):
    """Freeze two pre-test weights, using no test-period information.

    - target_vol_static: sample volatility of returns before validation.
    - exposure_matched_static: average *validation* MLE-GARCH target
      allocation, computed from forecasts that see only prior returns.

    If validation_start is None, use up to the last 252 pre-test observations,
    retaining at least 100 observations for development. Explicit validation
    dates are preferable for an actual pre-registered historical study.
    """
    r=np.asarray(returns,dtype=float)
    dates=pd.DatetimeIndex(dates)
    if len(r)!=len(dates) or not 0<target or first_test>=len(r) or first_test<102:
        raise ValueError('invalid calibration data, volatility target, or test index')
    if validation_start is None:
        validation_start=max(100,first_test-min(252,first_test-100))
    elif isinstance(validation_start,(str,pd.Timestamp,np.datetime64)):
        validation_start=int(np.searchsorted(dates.values,np.datetime64(validation_start)))
    else:
        validation_start=int(validation_start)
    if validation_start<100 or validation_start>=first_test:
        raise ValueError('validation_start must leave >=100 development returns and precede test_start')
    dev=r[:validation_start]
    dev_vol=float(np.std(dev,ddof=1)*np.sqrt(252))
    if not np.isfinite(dev_vol) or dev_vol<=0:
        raise ValueError('development volatility must be positive and finite')
    w_target=float(min(1.0,target/dev_vol))
    validation_weights=[]
    for start in range(validation_start,first_test,refit_days):
        stop=min(start+refit_days,first_test)
        from_idx=max(0,start-train_window)
        mle_u,opt=mle_fit(r[from_idx:start])
        if not opt.success:
            raise RuntimeError(f'validation MLE failed at {dates[start]}')
        v=_forecast_future(r[from_idx:stop],start-from_idx,np.asarray([mle_u]),stop-from_idx)[0]
        validation_weights.extend(np.minimum(1.0,target/np.sqrt(252*v)))
    w_matched=float(np.mean(validation_weights))
    return {
        'target_vol_static':w_target,
        'exposure_matched_static':w_matched,
        'development_start':str(dates[0].date()),
        'development_end':str(dates[validation_start-1].date()),
        'validation_start':str(dates[validation_start].date()),
        'validation_end':str(dates[first_test-1].date()),
        'development_spy_volatility':dev_vol,
        'validation_mle_mean_exposure':w_matched,
        'validation_observations':first_test-validation_start,
        'rule':'Weights are frozen before test. Target-vol weight is min(1,target/development SPY historical annualized SD); exposure-match weight is mean of validation-only dynamic MLE weights, refit with past data only.'
    }


def _backtest_strategy(forecast,kind,target=.10,cost_bps=5.,cash_returns=None,static_weights=None):
    """One-close-delayed allocation with cash interest and drift-aware costs.

    Forecast row t contains a signal available after t-1 close; weights from
    row t are applied to return at row t+1 (close t to close t+1).
    The cash return for that same holding interval is passed explicitly.
    """
    if static_weights is not None and kind in static_weights:
        weight=np.full(len(forecast),float(static_weights[kind]))
    elif kind=='buy_hold':
        weight=np.ones(len(forecast))
    elif kind=='mle':
        weight=np.minimum(1.,target/np.sqrt(252*forecast['variance_mle'].to_numpy()))
    elif kind=='bayes_mean':
        weight=np.minimum(1.,target/np.sqrt(252*forecast['variance_bayes_mean'].to_numpy()))
    elif kind=='bayes_upper90':
        weight=np.minimum(1.,target/forecast['volatility_bayes_q90_ann'].to_numpy())
    elif kind=='ewma':
        weight=np.minimum(1.,target/np.sqrt(252*forecast['variance_ewma'].to_numpy()))
    else:
        raise ValueError('unrecognized strategy')
    realized=np.expm1(forecast['return_log'].to_numpy()[1:])
    if cash_returns is None:
        raise ValueError('cash_returns required; provide historical DFF or explicit flat rate')
    summary,curve=simulate_portfolio(realized,cash_returns,weight[:-1],cost_bps=cost_bps,
                         dates=forecast['date'].iloc[1:])
    summary={'strategy':kind,**summary}
    curve.insert(1,'strategy',kind)
    return summary,curve


def ewma_variance(r, lam=.94):
    r=np.asarray(r,dtype=float)
    v=np.empty(len(r));v[0]=np.var(r[:min(100,len(r))])
    for t in range(1,len(r)):
        v[t]=lam*v[t-1]+(1-lam)*r[t-1]**2
    return v


def run_real(csv_path,output='results/milestone3',test_start='2025-01-01',
             refit_days=63,train_window=1000,n_chains=4,n_steps=4000,
             burn=1500,thin=5,posterior_draws=250,seed=2026,
             target=.10,cost_bps=5.,label='SPY',rf_csv=None,flat_rf_annual=None,
             validation_start=None):
    price=read_price_csv(csv_path)
    log_returns=np.log(price/price.shift(1)).dropna()
    # Do not allow the test to begin before the first observed date.
    first_test=int(np.searchsorted(log_returns.index.values,np.datetime64(test_start)))
    if first_test<100 or first_test>=len(log_returns)-1:
        raise ValueError('test_start must leave >=100 training returns and >=2 test returns')
    if (rf_csv is None)==(flat_rf_annual is None):
        raise ValueError('provide exactly one of rf_csv or flat_rf_annual')
    static_calibration=calibrate_static_benchmarks(
        log_returns.to_numpy(), log_returns.index, first_test, target=target,
        validation_start=validation_start, refit_days=refit_days, train_window=train_window)
    static_weights={k:static_calibration[k] for k in ('target_vol_static','exposure_matched_static')}
    forecast,diagnostics=fit_predict_real(log_returns.to_numpy(),log_returns.index,
        first_test,refit_days,train_window,n_chains,n_steps,burn,thin,
        posterior_draws,seed)
    forecast['variance_ewma']=ewma_variance(log_returns.to_numpy())[first_test:]
    # The strategy period on date[i+1] earns the cash rate fixed at date[i].
    # Require explicit real rate data or an explicitly labeled synthetic scenario.
    if (rf_csv is None)==(flat_rf_annual is None):
        raise ValueError('provide exactly one of rf_csv or flat_rf_annual')
    if rf_csv is not None:
        risk_free_source=f'historical FRED DFF proxy: {rf_csv}; ACT/360, strict prior-date lag'
        period_cash=cash_returns_for_periods(forecast['date'],read_rate_csv(rf_csv))
    else:
        risk_free_source=f'constant hypothetical {flat_rf_annual:.4%} effective annual cash yield, 252 trading periods'
        period_cash=constant_cash_returns(len(forecast)-1,flat_rf_annual)
    forecast['holding_period_cash_return']=np.r_[period_cash,np.nan]

    forecast['realized_squared_return']=forecast['return_log']**2
    scores={}
    for col in ['variance_mle','variance_bayes_mean','variance_ewma']:
        v=forecast[col].to_numpy(); y=forecast['realized_squared_return'].to_numpy()
        scores[col]={'mse_squared_return':float(np.mean((v-y)**2)),
                     'qlike_squared_return':float(np.mean(np.log(v)+y/v))}
    summaries=[];curves=[]
    for name in ['buy_hold','target_vol_static','exposure_matched_static','ewma','mle','bayes_mean','bayes_upper90']:
        summ,curve=_backtest_strategy(forecast,name,target=target,cost_bps=cost_bps,
                                      cash_returns=period_cash,static_weights=static_weights)
        summaries.append(summ);curves.append(curve)
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    forecast.to_csv(dest/'daily_forecasts.csv',index=False)
    pd.DataFrame(summaries).to_csv(dest/'strategy_summary.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(dest/'equity_curves.csv',index=False)
    (dest/'refit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    report={'label':label,'data_file':str(csv_path),'first_price_date':str(price.index[0].date()),
            'last_price_date':str(price.index[-1].date()),
            'test_start':str(forecast['date'].iloc[0].date()),
            'test_end':str(forecast['date'].iloc[-1].date()),
            'refit_days':refit_days,'train_window':train_window,'n_refits':len(diagnostics),
            'n_test_returns':len(forecast), 'n_traded_days':len(forecast)-1,
            'cash_rate_source':risk_free_source,
            'baseline_prior':'Normal on (log V, logit p, logit q) from src/garch.py',
            'static_benchmark_calibration':static_calibration,
            'forecast_scores':scores,'strategy_metrics':summaries,
            'max_split_rhat':float(max(max(d['split_rhat']) for d in diagnostics)),
            'execution_assumption':'forecast available after close t-1; execute at close t; earn close t to t+1 return; costs applied on traded notional, cash accrues stated cash yield',
            'limitations':'One-day close execution approximation, no spread/slippage beyond fee, DFF is a funding/cash proxy not guaranteed investable, adjusted-price provider convention, posterior frozen between scheduled refits; no true conditional variance observable.'}
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download-spy',action='store_true')
    parser.add_argument('--csv',default='data/SPY_adjusted.csv')
    parser.add_argument('--start',default='2015-01-01')
    parser.add_argument('--test-start',default='2025-01-01')
    parser.add_argument('--validation-start',default=None,help='start of pre-test validation, e.g. 2023-01-01; default last 252 pre-test returns')
    parser.add_argument('--output',default='results/milestone3')
    parser.add_argument('--refit-days',type=int,default=63)
    parser.add_argument('--train-window',type=int,default=1000)
    parser.add_argument('--chains',type=int,default=4)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--burn',type=int,default=1500)
    parser.add_argument('--thin',type=int,default=5)
    parser.add_argument('--draws',type=int,default=250)
    parser.add_argument('--rf-csv',default='data/DFF.csv',help='historical DFF (Date,DFF) annual percentages')
    parser.add_argument('--download-dff',action='store_true')
    parser.add_argument('--flat-rf-annual',type=float,default=None,help='explicit constant effective annual cash yield; do not use for historical conclusions')
    a=parser.parse_args()
    if a.download_spy:
        print('Downloaded adjusted SPY daily observations:',download_spy(a.csv,start=a.start))
    if a.download_dff:
        print('Downloaded DFF observations:',download_fred_dff(a.rf_csv,start='2014-01-01'))
    print(json.dumps(run_real(a.csv,a.output,a.test_start,a.refit_days,
                             a.train_window,a.chains,a.steps,a.burn,a.thin,
                             a.draws,rf_csv=None if a.flat_rf_annual is not None else a.rf_csv,
                             flat_rf_annual=a.flat_rf_annual,
                             validation_start=a.validation_start),indent=2))

if __name__=='__main__':
    main()
