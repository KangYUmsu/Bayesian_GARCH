"""Held-out synthetic GARCH variance forecasts and volatility-targeting backtest.

The posterior is estimated on the training window ONCE and held fixed during the
held-out period. The conditional variance state is filtered through newly observed
returns each day. No test return enters its own forecast. This is a fixed-parameter
out-of-sample experiment, not rolling posterior re-estimation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from numba import njit
from .garch import simulate, mle_fit, bayes_fit, unpack, rhat
from .portfolio import simulate_portfolio, constant_cash_returns

@njit(cache=True)
def _variance_forecast_path(returns, u, start):
    """Variance v_t conditional on r_0,...,r_(t-1), for t=start,...,T-1."""
    V = np.exp(u[0])
    p = 1.0 / (1.0 + np.exp(-u[1]))
    q = 1.0 / (1.0 + np.exp(-u[2]))
    omega = V*(1-p)
    alpha = p*q
    beta = p*(1-q)
    v = V
    out = np.empty(len(returns)-start)
    for t in range(len(returns)):
        if t > 0:
            v = omega + alpha*returns[t-1]**2 + beta*v
        if t >= start:
            out[t-start] = v
    return out


def predict_test_variance(returns, start, u, chains):
    """Point MLE, Bayesian mean, and Bayesian quantile paths (daily variance)."""
    returns = np.asarray(returns, dtype=float)
    if not (0 < start < len(returns)):
        raise ValueError('invalid train/test split')
    mle = _variance_forecast_path(returns, np.asarray(u), start)
    draws = np.asarray(chains).reshape(-1,3)
    # Store only test predictions, no simulation based on future test observations.
    bayes_draws = np.array([_variance_forecast_path(returns, d, start) for d in draws])
    return {
        'mle': mle,
        'bayes_mean': bayes_draws.mean(axis=0),
        'bayes_q05': np.quantile(bayes_draws, .05, axis=0),
        'bayes_q95': np.quantile(bayes_draws, .95, axis=0),
        'bayes_vol_q90': np.quantile(np.sqrt(bayes_draws*252), .90, axis=0),
    }


def variance_scores(pred, truth_variance, realized_returns):
    pred = np.asarray(pred)
    truth = np.asarray(truth_variance)
    sq = np.asarray(realized_returns)**2
    if np.any(pred <= 0):
        raise ValueError('nonpositive variance predictions')
    return {
        'mse_true_variance': float(np.mean((pred-truth)**2)),
        'qlike_true_variance': float(np.mean(np.log(pred)+truth/pred)),
        'qlike_squared_returns': float(np.mean(np.log(pred)+sq/pred)),
    }


def strategy_metrics(returns_log, daily_variance_forecast=None, annualized_vol_forecast=None,
                     target=.10, cost_bps=5., cash_returns=None, flat_rf_annual=.04):
    """Synthetic SPY/cash volatility-targeting backtest with interest on cash.

    Forecast for simulated return day t uses observed returns strictly before t.
    Default 4% effective annual cash yield is hypothetical, not market evidence.
    """
    r=np.asarray(returns_log,dtype=float)
    if (daily_variance_forecast is None)==(annualized_vol_forecast is None):
        raise ValueError('provide exactly one variance or annual volatility forecast')
    ann_vol=(np.sqrt(252*np.asarray(daily_variance_forecast)) if annualized_vol_forecast is None
             else np.asarray(annualized_vol_forecast))
    if len(ann_vol)!=len(r) or not np.isfinite(ann_vol).all() or np.any(ann_vol<=0):
        raise ValueError('invalid volatility forecasts')
    weights=np.clip(target/ann_vol,0.,1.)
    cash=constant_cash_returns(len(r),flat_rf_annual) if cash_returns is None else np.asarray(cash_returns)
    summary,_=simulate_portfolio(np.expm1(r),cash,weights,cost_bps=cost_bps)
    return summary


def run_one(seed=2026, train_n=750, test_n=250, n_chains=3,
            n_steps=1800, burn=600, thin=3, target=.10, cost_bps=5.):
    returns, latent = simulate(n=train_n+test_n,seed=seed)
    u, opt = mle_fit(returns[:train_n])
    chains, acceptance = bayes_fit(returns[:train_n],u,n_chains=n_chains,
                                   n_steps=n_steps,burn=burn,thin=thin,seed=seed+10000)
    preds=predict_test_variance(returns,train_n,u,chains)
    held_returns = returns[train_n:]
    truth=latent[train_n:]
    scores={name:variance_scores(preds[name],truth,held_returns)
            for name in ['mle','bayes_mean']}
    scores['bayes_90_interval_coverage_true_variance']=float(np.mean(
        (preds['bayes_q05']<=truth)&(truth<=preds['bayes_q95'])))
    scores['bayes_mean_interval_width']=float(np.mean(preds['bayes_q95']-preds['bayes_q05']))
    strategies={
        'oracle':strategy_metrics(held_returns,daily_variance_forecast=truth,target=target,cost_bps=cost_bps),
        'mle':strategy_metrics(held_returns,daily_variance_forecast=preds['mle'],target=target,cost_bps=cost_bps),
        'bayes_mean':strategy_metrics(held_returns,daily_variance_forecast=preds['bayes_mean'],target=target,cost_bps=cost_bps),
        'bayes_upper_vol90':strategy_metrics(held_returns,annualized_vol_forecast=preds['bayes_vol_q90'],target=target,cost_bps=cost_bps),
    }
    # Buy and hold uses the identical accounting convention and costs.
    strategies['buy_hold']=strategy_metrics(held_returns,annualized_vol_forecast=np.full(test_n,target/1.),
                                            target=target,cost_bps=cost_bps)
    return {
        'seed':seed, 'train_n':train_n,'test_n':test_n,
        'mle_success':bool(opt.success),
        'mle_parameters':unpack(u),
        'bayes_posterior_mean_parameters':{k:float(np.mean([unpack(draw)[k] for draw in chains.reshape(-1,3)]))
            for k in ['omega','alpha','beta','long_run_variance','persistence']},
        'mcmc_acceptance':acceptance.tolist(),
        'mcmc_split_rhat':rhat(chains).tolist(),
        'scores':scores,'strategies':strategies,
    },preds,truth,held_returns


def run_replicates(n_rep=12,train_n=750,test_n=250,n_chains=3,n_steps=1800,burn=600,
                   thin=3,seed=2026,output='results/milestone2'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for i in range(n_rep):
        result,preds,truth,r=run_one(seed=seed+i,train_n=train_n,test_n=test_n,n_chains=n_chains,
                                     n_steps=n_steps,burn=burn,thin=thin)
        results.append(result)
        if i==0:
            df=pd.DataFrame({'true_variance':truth,'mle_variance':preds['mle'],
                             'bayes_mean_variance':preds['bayes_mean'],
                             'bayes_q05':preds['bayes_q05'],'bayes_q95':preds['bayes_q95'],
                             'return_log':r})
            df.to_csv(output/'first_rep_forecasts.csv',index_label='test_day')
        print(f"rep {i+1}/{n_rep}: MLE QLIKE {result['scores']['mle']['qlike_true_variance']:.4f}, "
              f"Bayes QLIKE {result['scores']['bayes_mean']['qlike_true_variance']:.4f}; "
              f"max Rhat {max(result['mcmc_split_rhat']):.3f}", flush=True)
    (output/'replications.json').write_text(json.dumps(results,indent=2))
    rows=[]
    for r in results:
        for strat,metrics in r['strategies'].items():
            rows.append({'seed':r['seed'],'strategy':strat,**metrics})
    pd.DataFrame(rows).to_csv(output/'strategy_metrics.csv',index=False)
    summary={
        'experiment':{'n_rep':n_rep,'train_n':train_n,'test_n':test_n,
                      'n_chains':n_chains,'n_steps':n_steps,'burn':burn,'thin':thin,
                      'prior':'independent normal on unconstrained u; see src/garch.py',
                      'parameter_refitting':'once on train; filtering updated daily on test',
                      'cash_assumption':'hypothetical 4% effective annual cash yield compounded over 252 trading periods'},
        'mle_success_fraction':float(np.mean([r['mle_success'] for r in results])),
        'max_split_rhat_per_rep':[float(max(r['mcmc_split_rhat'])) for r in results],
        'mean_test_mse_mle':float(np.mean([r['scores']['mle']['mse_true_variance'] for r in results])),
        'mean_test_mse_bayes':float(np.mean([r['scores']['bayes_mean']['mse_true_variance'] for r in results])),
        'mean_test_qlike_mle':float(np.mean([r['scores']['mle']['qlike_true_variance'] for r in results])),
        'mean_test_qlike_bayes':float(np.mean([r['scores']['bayes_mean']['qlike_true_variance'] for r in results])),
        'mean_bayesian_90_interval_coverage':float(np.mean([r['scores']['bayes_90_interval_coverage_true_variance'] for r in results])),
        'strategy_mean_by_name':pd.DataFrame(rows).groupby('strategy').mean(numeric_only=True).drop(columns=['seed']).round(8).to_dict('index'),
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=12)
    p.add_argument('--train',type=int,default=750)
    p.add_argument('--test',type=int,default=250)
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--burn',type=int,default=600)
    p.add_argument('--chains',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',default='results/milestone2')
    a=p.parse_args()
    print(json.dumps(run_replicates(n_rep=a.reps,train_n=a.train,test_n=a.test,
         n_steps=a.steps,burn=a.burn,n_chains=a.chains,seed=a.seed,output=a.output),indent=2))
