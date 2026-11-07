"""Bayesian GARCH(1,1): prior, posterior mode, MCMC, convergence diagnostics, posterior summaries.

Coordinates and prior. Parameters are u = (log V, logit p, logit q) (see ``bgarch.garch``).
The project's 'moderate' prior (``PRIORS['moderate']``) is a product of independent Gaussians
directly on u:

    log V ~ N(log 1e-4, 1.5^2),  logit p ~ N(logit 0.80, 1.5^2),  logit q ~ N(logit 0.10, 1.5^2).

Because the density is defined on u, the posterior pi(u | r) ∝ L(u) pi(u) needs no Jacobian.
``Prior.log_posterior(u, r)`` returns log L(u) + log pi(u) (up to a constant).

Sampler (``reliable_bayes_fit(r, mle_u, n_chains=4, warmup=2500, draws=4000, seed, log_density)``).
Random-walk Metropolis with ``n_chains`` independent chains started at MAP + N(0, 3 Sigma), where
Sigma is the regularised inverse Hessian at the MAP. During ``warmup`` iterations the proposal
scale is adapted every 100 iterations towards 23.4% acceptance and, from iteration 900 on, the
proposal covariance is set to the empirical covariance of the warm-up history; both are then
frozen for the ``draws`` retained iterations (no thinning). Returns ``chains`` of shape
(n_chains, draws, 3) and per-chain metadata (acceptance, final_scale, proposal_cov). RNG seeds:
``seed`` for the starting points and ``seed + 1000 + 97 k`` for chain k.

``diagnostics(chains)`` returns rank-normalised split R-hat and bulk/tail ESS (ArviZ) for u and for
(V, p, q, omega, alpha, beta), and ``passed`` = all R-hat < 1.01 and all ESS >= 400.

Numerical note: chain starting points use ``numpy.random.Generator.multivariate_normal``, whose
output depends on the linear-algebra backend, so draws are bit-reproducible only on one platform.

Imported unchanged:  PRIORS, Prior, map_fit (src.prior_source_study);
                     reliable_bayes_fit, adaptive_chain, diagnostics, local_covariance (src.reliable_mcmc);
                     posterior_mean_u (src.market_uncertainty_backtest._mean_u_natural).
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from src.prior_source_study import PRIORS, Prior, map_fit  # noqa: F401
from src.reliable_mcmc import reliable_bayes_fit, adaptive_chain, diagnostics, local_covariance  # noqa: F401
from src.market_uncertainty_backtest import _mean_u_natural as posterior_mean_u  # noqa: F401
from .garch import loglik_u
from .forecast import variance_draws, mixture_logpdf  # noqa: F401  (kept importable from bayes)

# posterior_mean_u(flat): u = encode(E[V], E[p], E[q]) from flat draws of shape (N, 3) -- the means of the
# NATURAL parameters, not of u. It defines the "posterior-mean plug-in" forecast.


def laplace_cov(train, mle_u):
    """Inverse observed information at the MLE, Cov(u_hat) ~= [-d^2 l(u)/du du']^{-1}, shape (3, 3).

    Central finite differences with step 0.01 in u; eigenvalues of the Hessian are clipped to
    [0.02, 1e8] so the result is symmetric positive definite (exactly as the pipeline's Laplace
    comparator). Uses the log-LIKELIHOOD only (no prior).
    """
    return local_covariance(train, mle_u, log_density=lambda u, rr: float(loglik_u(rr, np.asarray(u, float))))


def natural_draws(flat):
    """Map an (N, 3) array of u-draws to a DataFrame of natural parameters.

    Columns: omega, alpha, beta, persistence p, long-run variance V, long-run annualised volatility
    sqrt(252 V), q = alpha/p, and the shock half-life log(0.5)/log(p) in trading days.
    """
    u = np.asarray(flat, float).reshape(-1, 3)
    V = np.exp(u[:, 0]); p = expit(u[:, 1]); q = expit(u[:, 2])
    return pd.DataFrame({'omega': V * (1 - p), 'alpha': p * q, 'beta': p * (1 - q), 'persistence': p,
                         'long_run_variance': V, 'long_run_ann_vol': np.sqrt(252 * V), 'q': q,
                         'half_life_days': np.log(.5) / np.log(p)})


def posterior_summary(chains, probs=(0.05, 0.5, 0.95)):
    """Posterior mean, sd and quantiles of the natural parameters; ``chains`` has shape (..., 3).

    Returns a DataFrame indexed by parameter with columns mean, sd and q05/q50/q95 (for the default probs).
    """
    d = natural_draws(np.asarray(chains).reshape(-1, 3))
    out = pd.DataFrame({'mean': d.mean(), 'sd': d.std(ddof=1)})
    for pr in probs:
        out[f'q{int(round(pr * 100)):02d}'] = d.quantile(pr)
    return out
