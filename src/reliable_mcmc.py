"""Validated GARCH random-walk Metropolis with warm-up-only adaptation.

The target is the EXISTING log_posterior(u, returns): Gaussian priors directly on
u=(log V, logit p, logit q). No redefinition of prior or missing Jacobian.
Proposal covariance uses a regularized local posterior Hessian and is adapted
only during warm-up. Post-warm-up samples use a fixed symmetric random walk.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .garch import log_posterior, mle_fit


def local_covariance(r, center, h=0.01, log_density=None):
    """Finite-difference negative log posterior Hessian -> SPD covariance.

    A failure to obtain a positive curvature yields a regularized approximation;
    covariance tuning, not this Laplace estimate, determines validity of MCMC.
    """
    center=np.asarray(center,dtype=float)
    d=len(center)
    target = log_posterior if log_density is None else log_density
    fn=lambda x: -float(target(x,r))
    f0=fn(center)
    hessian=np.zeros((d,d))
    for i in range(d):
        ei=np.eye(d)[i]*h
        hessian[i,i]=(fn(center+ei)-2*f0+fn(center-ei))/(h*h)
        for j in range(i):
            ej=np.eye(d)[j]*h
            value=(fn(center+ei+ej)-fn(center+ei-ej)-fn(center-ei+ej)+fn(center-ei-ej))/(4*h*h)
            hessian[i,j]=hessian[j,i]=value
    if not np.isfinite(hessian).all():
        return np.diag([0.15**2,0.25**2,0.30**2])
    values,vectors=np.linalg.eigh(hessian)
    # Regularization only for proposal shape, NEVER modifies posterior density.
    values=np.clip(values,0.02,1e8)
    return (vectors/values)@vectors.T


def adaptive_chain(r, start, *, warmup=2500, draws=4000, seed=0,
                   initial_cov=None, adapt_every=100, target_accept=0.234, log_density=None):
    """Warmup covariance+scale adaptation; then frozen-proposal MH sampling.

    Returns (draws, metadata). 1 draw per iteration, no thinning.
    """
    if warmup < 200 or draws < 10 or adapt_every < 20:
        raise ValueError('warmup>=200, draws>=10, adapt_every>=20 required')
    r=np.asarray(r,dtype=float)
    target = log_posterior if log_density is None else log_density
    rng=np.random.default_rng(seed)
    u=np.asarray(start,dtype=float).copy()
    lp=float(target(u,r))
    if not np.isfinite(lp) or lp <= -1e200:
        raise ValueError('invalid sampler start')
    d=len(u)
    base=np.asarray(initial_cov if initial_cov is not None else np.diag([.1,.2,.2])**2,dtype=float)
    scale=2.38/np.sqrt(d)
    history=[]
    accepted_warm=0
    batch_accept=0
    n_batches=0
    chol=np.linalg.cholesky(base+1e-10*np.eye(d))
    for i in range(warmup):
        prop=u+scale*(chol@rng.standard_normal(d))
        lpnew=float(target(prop,r))
        if np.log(rng.random()) < lpnew-lp:
            u=prop;lp=lpnew;accepted_warm+=1;batch_accept+=1
        history.append(u.copy())
        if (i+1)%adapt_every==0:
            n_batches+=1
            rate=batch_accept/adapt_every
            scale *= np.exp((rate-target_accept)*min(.65,1/np.sqrt(n_batches)))
            # Do not estimate full covariance until enough warm-up samples.
            if i+1 >= max(300,3*d*adapt_every):
                empirical=np.cov(np.asarray(history).T)
                if np.isfinite(empirical).all():
                    base=empirical+1e-5*np.eye(d)
                    chol=np.linalg.cholesky(base)
            batch_accept=0
    out=np.empty((draws,d),float)
    accepted=0
    for i in range(draws):
        prop=u+scale*(chol@rng.standard_normal(d))
        lpnew=float(target(prop,r))
        if np.log(rng.random()) < lpnew-lp:
            u=prop;lp=lpnew;accepted+=1
        out[i]=u
    return out,{'acceptance':accepted/draws,'warmup_acceptance':accepted_warm/warmup,
                'final_scale':scale,'proposal_cov':scale**2*(chol@chol.T)}


def reliable_bayes_fit(r, mle_u=None, *, n_chains=4, warmup=2500, draws=4000, seed=2026, log_density=None):
    """Run independent chains for unchanged GARCH posterior."""
    r=np.asarray(r,dtype=float)
    if mle_u is None:
        mle_u,_=mle_fit(r)
    mle_u=np.asarray(mle_u,float)
    target = log_posterior if log_density is None else log_density
    result=minimize(lambda u:-target(u,r),mle_u,method='BFGS',
                    options={'gtol':1e-5,'maxiter':250})
    map_u=result.x if np.isfinite(result.fun) and result.fun<=-target(mle_u,r) else mle_u
    cov=local_covariance(r,map_u,log_density=target)
    # Start chains deliberately dispersed (not identical at MLE); local covariance
    # sets scales without influencing the target posterior density.
    rng=np.random.default_rng(seed)
    starts=map_u+rng.multivariate_normal(np.zeros(3),cov*3,size=n_chains)
    sampled=[];metadata=[]
    for k in range(n_chains):
        chain,info=adaptive_chain(r,starts[k],warmup=warmup,draws=draws,
                                  seed=seed+1000+97*k,initial_cov=cov,log_density=target)
        sampled.append(chain);metadata.append(info)
    return np.stack(sampled),metadata


def diagnostics(chains):
    """Rank-normalized R-hat, bulk/tail ESS for transformed and natural parameters.

    Compatible with both arviz-base 1.x and classic ArviZ APIs (array inputs).
    """
    import arviz as az
    import pandas as pd
    from scipy.special import expit
    u=np.asarray(chains)
    V=np.exp(u[:,:,0]);p=expit(u[:,:,1]);q=expit(u[:,:,2])
    arrays={'u0':u[:,:,0],'u1':u[:,:,1],'u2':u[:,:,2],
            'V':V,'p':p,'q':q,'omega':V*(1-p),'alpha':p*q,'beta':p*(1-q)}
    rows=[]
    for name,values in arrays.items():
        rows.append(dict(parameter=name, r_hat=float(np.asarray(az.rhat(values,method='rank'))),
                         ess_bulk=float(np.asarray(az.ess(values,method='bulk'))),
                         ess_tail=float(np.asarray(az.ess(values,method='tail',prob=(0.05,0.95))))))
    summary=pd.DataFrame(rows).set_index('parameter')
    valid=bool(np.isfinite(summary.to_numpy()).all()
               and (summary['r_hat']<1.01).all()
               and (summary['ess_bulk']>=400).all()
               and (summary['ess_tail']>=400).all())
    return summary, valid
