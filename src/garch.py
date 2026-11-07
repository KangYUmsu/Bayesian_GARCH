"""Gaussian zero-mean GARCH(1,1): simulator, MLE, and reproducible Bayesian inference.

Returns are in decimal units. Inference parameters u=(log(V), logit(p), logit(q))
with omega=V*(1-p), alpha=p*q, beta=p*(1-q).
The initial conditional variance is V (stationary initialization).
Bayesian prior: independent N(log(1e-4), 1.5**2), N(logit(.95),1.5**2),
N(logit(.1),1.5**2) on unconstrained u coordinates. Thus posterior is
with respect to du; no Jacobian is needed. This is NOT a flat prior on
(omega,alpha,beta). Both MLE and Bayesian likelihood use the same model.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from numba import njit

PRIOR_MEAN=np.array([np.log(1e-4), logit(.95), logit(.1)])
PRIOR_SD=np.array([1.5,1.5,1.5])

@njit(cache=True)
def _sigmoid(x):
    if x>=0:
        return 1.0/(1.0+np.exp(-x))
    z=np.exp(x)
    return z/(1.0+z)

@njit(cache=True)
def _ll(r, u):
    V=np.exp(u[0]); p=_sigmoid(u[1]); q=_sigmoid(u[2])
    if not np.isfinite(V) or V<=0 or p<=0 or p>=1 or q<=0 or q>=1:
        return -1e300
    omega=V*(1-p); alpha=p*q; beta=p*(1-q)
    v=V
    total=0.0
    for t in range(len(r)):
        if t>0:
            v=omega+alpha*r[t-1]**2+beta*v
        if v<=0 or not np.isfinite(v):
            return -1e300
        total-=0.5*(np.log(2*np.pi)+np.log(v)+r[t]**2/v)
    return total

@njit(cache=True)
def _next_variance(r,u):
    V=np.exp(u[0]); p=_sigmoid(u[1]); q=_sigmoid(u[2])
    omega=V*(1-p);alpha=p*q;beta=p*(1-q)
    v=V
    for t in range(len(r)):
        if t>0:
            v=omega+alpha*r[t-1]**2+beta*v
    return omega+alpha*r[-1]**2+beta*v

@njit(cache=True)
def _latent_variances(r,u):
    V=np.exp(u[0]); p=_sigmoid(u[1]); q=_sigmoid(u[2])
    omega=V*(1-p);alpha=p*q;beta=p*(1-q)
    v=V
    out=np.empty(len(r))
    for t in range(len(r)):
        if t>0:
            v=omega+alpha*r[t-1]**2+beta*v
        out[t]=v
    return out

def unpack(u):
    V=float(np.exp(u[0]));p=float(expit(u[1]));q=float(expit(u[2]))
    return dict(omega=V*(1-p),alpha=p*q,beta=p*(1-q),long_run_variance=V,persistence=p,shock_fraction=q)

def encode(V,p,q):
    return np.array([np.log(V),logit(p),logit(q)],dtype=float)

def simulate(n=1000,seed=2026,V=1e-4,alpha=.08,beta=.90,burn=500):
    """Generate stationary-initialized Gaussian GARCH data, keeping hidden truth."""
    if not (n>0 and V>0 and alpha>=0 and beta>=0 and alpha+beta<1):
        raise ValueError('invalid GARCH parameters')
    rng=np.random.default_rng(seed); omega=V*(1-alpha-beta)
    r=np.empty(n+burn);v=np.empty(n+burn)
    for t in range(n+burn):
        v[t]=V if t==0 else omega+alpha*r[t-1]**2+beta*v[t-1]
        r[t]=np.sqrt(v[t])*rng.normal()
    return r[burn:],v[burn:]

def mle_fit(r):
    """Robust multi-start Gaussian GARCH MLE; same likelihood and bounds.

    A failed optimizer status is not silently selected over a valid optimum.
    If L-BFGS-B cannot provide a trustworthy result, use Powell as a fallback.
    The returned scipy OptimizeResult contains method, fallback_used,
    num_attempts and an attempts diagnostic list.
    """
    r = np.asarray(r, dtype=float)
    if len(r) < 30 or not np.isfinite(r).all():
        raise ValueError('need >=30 finite returns')

    V0 = max(float(np.var(r)), 1e-8)
    initial = encode(V0, .94, .12)
    starts = [initial, encode(V0, .8, .2), encode(V0, .98, .08)]
    bounds = [(-15, -5), (-7, 7), (-7, 7)]  # preserve original model bounds

    def objective(u):
        value = -_ll(r, np.asarray(u, dtype=float))
        return float(value) if np.isfinite(value) else 1e100

    fits = []
    for x in starts:
        fit = minimize(objective, x, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12})
        fit.method = 'L-BFGS-B'
        fits.append(fit)

    def valid(f):
        return (bool(f.success) and np.isfinite(f.fun)
                and np.all(np.isfinite(f.x)))

    successful = [f for f in fits if valid(f)]
    finite = [f for f in fits if np.isfinite(f.fun)
              and np.all(np.isfinite(f.x))]
    if not finite:
        raise RuntimeError('MLE: no finite L-BFGS-B optimization result')

    best_finite = min(finite, key=lambda f: f.fun)
    best_success = min(successful, key=lambda f: f.fun) if successful else None
    # A non-successful fit may have better objective than a successful one.
    # Verify with an alternative optimizer instead of trusting the success flag.
    needs_fallback = (best_success is None or
                      best_finite.fun < best_success.fun - 1e-5)
    if needs_fallback:
        fallback_starts = [best_finite.x]
        if best_success is not None:
            fallback_starts.append(best_success.x)
        else:
            fallback_starts.append(initial)
        for x in fallback_starts:
            fit = minimize(objective, x, method='Powell', bounds=bounds,
                           options={'maxiter': 3000, 'xtol': 1e-7,
                                    'ftol': 1e-10})
            fit.method = 'Powell'
            fits.append(fit)

    successful = [f for f in fits if valid(f)]
    details = [dict(method=str(getattr(f, 'method', 'unknown')),
                    success=bool(f.success), status=int(f.status),
                    message=str(f.message),
                    objective=float(f.fun),
                    iterations=int(getattr(f, 'nit', 0))) for f in fits]
    if not successful:
        raise RuntimeError(f'MLE optimizer failed at all starting points: {details}')
    best = min(successful, key=lambda f: f.fun)
    finite = [f for f in fits if np.isfinite(f.fun)
              and np.all(np.isfinite(f.x))]
    lowest = min(finite, key=lambda f: f.fun)
    if lowest.fun < best.fun - 1e-5:
        raise RuntimeError('MLE: unsuccessful optimizer found materially '
                           'better likelihood than the best successful fit: '
                           f'best_success={best.fun:.12g}, '
                           f'best_attempt={lowest.fun:.12g}; attempts={details}')
    best.fallback_used = bool(needs_fallback)
    best.num_attempts = len(fits)
    best.attempts = details
    return best.x, best

def log_posterior(u,r):
    z=(u-PRIOR_MEAN)/PRIOR_SD
    return _ll(r,u)-0.5*np.dot(z,z)

def mh_chain(r, start, n_steps=5500, burn=1500, thin=4, seed=0, step=(.12,.20,.27)):
    """Random-walk Metropolis in unconstrained coordinates (no tuning on test data)."""
    if not (0<=burn<n_steps and thin>=1):raise ValueError('invalid MCMC settings')
    rng=np.random.default_rng(seed)
    u=np.array(start,dtype=float).copy(); lp=log_posterior(u,r)
    draws=[];accepted=0
    for i in range(n_steps):
        proposal=u+rng.normal(size=3)*np.asarray(step)
        lp_new=log_posterior(proposal,r)
        if np.log(rng.random())<lp_new-lp:
            u=proposal;lp=lp_new;accepted+=1
        if i>=burn and (i-burn)%thin==0:
            draws.append(u.copy())
    return np.asarray(draws), accepted/n_steps

def bayes_fit(r, mle_u=None, n_chains=4,n_steps=5500,burn=1500,thin=4,seed=2026):
    r=np.asarray(r,dtype=float)
    if mle_u is None:mle_u,_=mle_fit(r)
    rng=np.random.default_rng(seed);chains=[];acceptance=[]
    for i in range(n_chains):
        start=np.asarray(mle_u)+rng.normal(scale=[.08,.15,.2],size=3)
        d,a=mh_chain(r,start,n_steps,burn,thin,seed+i+1)
        chains.append(d);acceptance.append(a)
    return np.stack(chains),np.array(acceptance)

def rhat(chains):
    """Basic split-Rhat; check ESS/trace as well before scientific conclusions."""
    c=np.asarray(chains); m,n,p=c.shape
    if n<4:raise ValueError('not enough draws')
    halves=np.concatenate([c[:,:n//2,:],c[:,n//2:2*(n//2),:]],axis=0)
    k,l,_=halves.shape
    means=halves.mean(axis=1); W=halves.var(axis=1,ddof=1).mean(axis=0)
    B=l*means.var(axis=0,ddof=1)
    return np.sqrt(((l-1)/l*W+B/l)/W)

def posterior_next_variance(r,chains):
    flat=np.asarray(chains).reshape(-1,3)
    return np.array([_next_variance(np.asarray(r),u) for u in flat])
