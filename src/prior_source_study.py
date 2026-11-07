"""Milestone 9: diagnose the sources of Bayesian GARCH differences.

Four estimators on IDENTICAL synthetic samples:
  MLE: likelihood-optimized plug-in forecast
  MAP(u): posterior-density mode in unconstrained u coordinates, plug-in forecast
  posterior-parameter-mean plug-in: forecast using mean of (V,p,q)
  posterior-averaged: average forecasts evaluated at each joint posterior draw

MAP(u) depends on coordinates; posterior mean and posterior forecast are distinct.
All priors are specified DIRECTLY in u=(log V, logit p, logit q); therefore
no Jacobian is inserted in the u-density. Prior sensitivity is explicit.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from .garch import simulate, mle_fit, encode, unpack, _ll, _next_variance
from .reliable_mcmc import reliable_bayes_fit, diagnostics

@dataclass(frozen=True)
class Prior:
    name: str
    persistence_center: float
    persistence_sd_logit: float
    logV_center: float = float(np.log(1e-4))
    logV_sd: float = 1.5
    q_center: float = 0.1
    q_sd_logit: float = 1.5

    @property
    def mean_u(self):
        return np.array([self.logV_center,logit(self.persistence_center),logit(self.q_center)])

    @property
    def sd_u(self):
        return np.array([self.logV_sd,self.persistence_sd_logit,self.q_sd_logit])

    def log_prior(self,u):
        z=(np.asarray(u)-self.mean_u)/self.sd_u
        return float(-0.5*np.dot(z,z))

    def log_posterior(self,u,r):
        return float(_ll(r,np.asarray(u,float)))+self.log_prior(u)

PRIORS={
    'current_high':Prior('current_high',0.95,1.5),
    'broad':Prior('broad',0.50,3.0),
    'moderate':Prior('moderate',0.80,1.5),
    'low':Prior('low',0.50,1.5),
}

def map_fit(r, mle_u, prior):
    """Robust multi-start MAP optimizer in the prior's u coordinates.

    Retains the existing bounds and prior; tries L-BFGS-B from three starts,
    then Powell from promising candidates if needed. Never silently accepts
    a failed optimizer's point or a success with materially worse objective.

    Returns (u_map, scipy OptimizeResult). The result includes the ``method``
    and ``fallback_used`` fields for logging.
    """
    bounds = [(-18.0, -3.0), (-13.0, 13.0), (-13.0, 13.0)]
    mle_u = np.asarray(mle_u, dtype=float)
    if mle_u.shape != (3,) or not np.all(np.isfinite(mle_u)):
        raise ValueError("mle_u must contain three finite transformed parameters")

    def clip_bounds(u):
        return np.clip(np.asarray(u, dtype=float),
                       [b[0] for b in bounds], [b[1] for b in bounds])

    def objective(u):
        value = -prior.log_posterior(u, r)
        return float(value) if np.isfinite(value) else 1e100

    starts = [clip_bounds(mle_u), clip_bounds(prior.mean_u),
              clip_bounds(0.5 * (mle_u + prior.mean_u))]
    fits = []
    for z in starts:
        res = minimize(objective, z, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "maxls": 50, "ftol": 1e-12})
        res.method = "L-BFGS-B"
        fits.append(res)

    def acceptable(res):
        return (bool(res.success) and np.isfinite(res.fun)
                and np.asarray(res.x).shape == (3,)
                and np.all(np.isfinite(res.x))
                and all(lo - 1e-9 <= x <= hi + 1e-9
                        for x, (lo, hi) in zip(res.x, bounds)))

    # If a failed run has the lowest objective, it deserves investigation,
    # even when another start technically succeeded.
    finite = [f for f in fits if np.isfinite(f.fun)]
    successful = [f for f in fits if acceptable(f)]
    needs_fallback = (not successful or
                      (finite and min(f.fun for f in finite) <
                       min(f.fun for f in successful) - 1e-5))

    if needs_fallback:
        candidates = sorted(finite, key=lambda f: f.fun)[:2]
        fallback_starts = [clip_bounds(f.x) for f in candidates]
        fallback_starts.append(clip_bounds(prior.mean_u))
        for z in fallback_starts:
            res = minimize(objective, z, method="Powell", bounds=bounds,
                           options={"maxiter": 3000, "maxfev": 30000,
                                    "xtol": 1e-7, "ftol": 1e-10})
            res.method = "Powell"
            fits.append(res)

    successful = [f for f in fits if acceptable(f)]
    if not successful:
        detail = [(f.method, int(f.status), str(f.message), float(f.fun))
                  for f in fits]
        raise RuntimeError(f"MAP optimization failed from every start: {detail}")

    best = min(successful, key=lambda f: f.fun)
    finite = [f for f in fits if np.isfinite(f.fun)]
    lowest = min(finite, key=lambda f: f.fun)
    if lowest.fun < best.fun - 1e-5:
        raise RuntimeError(
            "MAP optimization unresolved: a failed run has materially better "
            f"objective ({lowest.fun:.9g}) than the best successful run "
            f"({best.fun:.9g}); method={lowest.method}, "
            f"message={lowest.message}"
        )
    best.fallback_used = bool(needs_fallback)
    best.num_attempts = len(fits)
    return np.asarray(best.x, dtype=float), best

def posterior_moments_and_forecasts(train,chain):
    flat=np.asarray(chain).reshape(-1,3)
    V=np.exp(flat[:,0]);p=expit(flat[:,1]);q=expit(flat[:,2])
    avg_u=encode(float(np.mean(V)),float(np.mean(p)),float(np.mean(q)))
    plug=float(_next_variance(train,avg_u))
    forecast_draws=np.array([_next_variance(train,u) for u in flat])
    return {
        'bayes_persistence':float(np.mean(p)),
        'bayes_meanplug_variance':plug,
        'bayes_integrated_variance':float(np.mean(forecast_draws)),
        'bayes_forecast_low90':float(np.quantile(forecast_draws,.05)),
        'bayes_forecast_high90':float(np.quantile(forecast_draws,.95)),
        'bayes_forecast_sd':float(np.std(forecast_draws,ddof=1)),
    }

def one_case(train_size=250,replicate=0,prior_name='current_high',true_p=.98,
             seed=2026,chains=4,warmup=1500,draws=2500):
    if not (60<=train_size and 0<true_p<1): raise ValueError('invalid train size or persistence')
    prior=PRIORS[prior_name]
    # Constant long-run variance and shock-fraction across generating regimes.
    true_V=1e-4;true_q=.08/.98
    alpha=true_p*true_q;beta=true_p*(1-true_q)
    # Maintain compatibility with the 30-replication study for true_p=.98.
    sim_seed=seed+train_size*1000+replicate*101
    values,latent=simulate(n=train_size+1,seed=sim_seed,V=true_V,alpha=alpha,beta=beta)
    train=values[:-1];true_next=float(latent[-1]);mle_u,mle_result=mle_fit(train)
    map_u,map_result=map_fit(train,mle_u,prior)
    posterior,meta=reliable_bayes_fit(train,mle_u,n_chains=chains,warmup=warmup,
                                     draws=draws,seed=sim_seed+29,log_density=prior.log_posterior)
    table,passed=diagnostics(posterior)
    moments=posterior_moments_and_forecasts(train,posterior)
    mle_forecast=float(_next_variance(train,mle_u))
    map_forecast=float(_next_variance(train,map_u))
    rec=dict(train_size=train_size,replicate=replicate+1,seed=sim_seed,prior=prior_name,
             true_p=true_p,true_next_variance=true_next,
             mle_persistence=unpack(mle_u)['persistence'],
             map_u_persistence=unpack(map_u)['persistence'],
             mle_variance=mle_forecast,map_u_variance=map_forecast,
             mle_success=bool(mle_result.success),map_success=bool(map_result.success),
             convergence_pass=passed,max_rhat=float(table.r_hat.max()),
             min_bulk_ess=float(table.ess_bulk.min()),min_tail_ess=float(table.ess_tail.min()),
             mean_acceptance=float(np.mean([m['acceptance'] for m in meta])),
             **moments)
    for name,p_est in [('mle',rec['mle_persistence']),('map_u',rec['map_u_persistence']),('bayes',rec['bayes_persistence'])]:
        rec[name+'_p_abs_error']=abs(float(p_est)-true_p)
    for name,forecast in [('mle',mle_forecast),('map_u',map_forecast),('bayes_meanplug',moments['bayes_meanplug_variance']),('bayes_integrated',moments['bayes_integrated_variance'])]:
        rec[name+'_forecast_sqerr']=float((forecast-true_next)**2)
    rec['forecast_integration_effect']=moments['bayes_integrated_variance']-moments['bayes_meanplug_variance']
    rec['forecast_cover90']=float(moments['bayes_forecast_low90']<=true_next<=moments['bayes_forecast_high90'])
    return rec

def run_study(sizes=(250,500),reps=3,priors=('current_high','broad','low'),true_ps=(.98,),
              seed=2026,chains=4,warmup=1500,draws=2500,verbose=True):
    records=[]
    for p in true_ps:
        for n in sizes:
            for rep in range(reps):
                for prior in priors:
                    row=one_case(n,rep,prior,p,seed,chains,warmup,draws)
                    records.append(row)
                    if verbose:
                        print(f'true p={p:.2f} n={n:4d} rep={rep+1}/{reps} prior={prior:12} '
                              f'p MLE/MAP/Bayes={row["mle_persistence"]:.3f}/'
                              f'{row["map_u_persistence"]:.3f}/{row["bayes_persistence"]:.3f} '
                              f'Rhat={row["max_rhat"]:.3f} pass={row["convergence_pass"]}',flush=True)
    return pd.DataFrame(records)

def summarize(df):
    cols=['mle_p_abs_error','map_u_p_abs_error','bayes_p_abs_error',
          'mle_forecast_sqerr','map_u_forecast_sqerr','bayes_meanplug_forecast_sqerr',
          'bayes_integrated_forecast_sqerr','forecast_integration_effect',
          'forecast_cover90','convergence_pass','max_rhat']
    out=df.groupby(['true_p','train_size','prior'],as_index=False)[cols].mean()
    for prefix in ('mle','map_u','bayes_meanplug','bayes_integrated'):
        out[prefix+'_forecast_rmse']=np.sqrt(out[prefix+'_forecast_sqerr'])
    return out

def save(df,folder):
    import matplotlib.pyplot as plt
    output=Path(folder);output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'replications.csv',index=False)
    s=summarize(df);s.to_csv(output/'summary.csv',index=False)
    for p,frame in s.groupby('true_p'):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for prior,group in frame.groupby('prior'):
            ax.plot(group.train_size,group.bayes_p_abs_error,'-o',label=f'Bayesian: {prior}')
        mle=frame.drop_duplicates('train_size').sort_values('train_size')
        ax.plot(mle.train_size,mle.mle_p_abs_error,'k--o',label='MLE')
        ax.set(xlabel='Training observations',ylabel='Mean absolute error in persistence',
               title=f'Prior sensitivity / true persistence = {p:.2f}')
        ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(output/f'persistence_comparison_p{p:.2f}.png',dpi=160)
        plt.close(fig)
    return s

def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--sizes',type=int,nargs='+',default=[250,500])
    arg.add_argument('--reps',type=int,default=3)
    arg.add_argument('--priors',nargs='+',choices=list(PRIORS),default=['current_high','broad','low'])
    arg.add_argument('--true-ps',type=float,nargs='+',default=[.98])
    arg.add_argument('--chains',type=int,default=4)
    arg.add_argument('--warmup',type=int,default=1500)
    arg.add_argument('--draws',type=int,default=2500)
    arg.add_argument('--seed',type=int,default=2026)
    arg.add_argument('--output',default='results/milestone9_prior_sources')
    args=arg.parse_args()
    df=run_study(args.sizes,args.reps,args.priors,args.true_ps,args.seed,args.chains,args.warmup,args.draws)
    s=save(df,args.output)
    print('\nSummary (all runs, inspect convergence before inference):')
    print(s[['true_p','train_size','prior','mle_p_abs_error','map_u_p_abs_error',
             'bayes_p_abs_error','mle_forecast_rmse','map_u_forecast_rmse',
             'bayes_integrated_forecast_rmse','convergence_pass']].to_string(index=False))
if __name__=='__main__': main()
