"""Out-of-sample forecast diagnostics on plain arrays (residuals, PIT, autocorrelation).

For one-step forecasts of r_t made with information before t:

    standardised residual   z_t = r_t / sqrt(h_t)
    PIT                     u_t = F_t(r_t)  (iid U(0,1) if the predictive CDF F_t is correct)
    normal scores           Phi^{-1}(u_t)   (iid N(0,1) under correct calibration; Berkowitz 2001)

MCMC convergence diagnostics are in ``bgarch.bayes.diagnostics``.
"""
import numpy as np
from scipy.stats import norm, t as student_t

from .forecast import mixture_cdf


def std_resid(r, h):
    """z_t = r_t / sqrt(h_t)."""
    return np.asarray(r, float) / np.sqrt(np.asarray(h, float))


def pit_gaussian(r, h):
    """u_t = Phi(r_t / sqrt(h_t))."""
    return norm.cdf(std_resid(r, h))


def pit_t(r, h, nu):
    """u_t for standardised Student-t innovations with variance h_t: F_nu(r_t / s_t), s_t = sqrt(h_t (nu-2)/nu)."""
    nu = np.asarray(nu, float)
    return student_t.cdf(np.asarray(r, float) / np.sqrt(np.asarray(h, float) * (nu - 2) / nu), nu)


def pit_mixture(r, v_draws):
    """u_t under the Gaussian scale-mixture predictive distribution (v_draws has shape (M, n))."""
    return mixture_cdf(r, v_draws)


def normal_scores(u):
    """Berkowitz (2001) transform Phi^{-1}(u), clipped to avoid infinities at u in {0, 1}."""
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def acf(x, L=20):
    """Sample autocorrelations at lags 1..L (demeaned, biased denominator, as statsmodels with fft=False)."""
    x = np.asarray(x, float) - np.mean(x); d = x @ x
    return np.array([x[k:] @ x[:-k] / d for k in range(1, L + 1)])


def ljung_box_p(x, lag):
    """p-value of the Ljung-Box statistic Q(m) = n(n+2) sum_{k<=m} rho_k^2/(n-k) against chi^2_m (statsmodels)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return float(acorr_ljungbox(x, lags=[lag], return_df=True).lb_pvalue.iloc[0])


def realized_vol(r, n=21):
    """Backward-looking descriptive proxy sqrt(252 * mean of the last n squared returns, INCLUDING day t).

    Not a forecast target: it contains r_t itself and lags turning points.
    """
    import pandas as pd
    return np.sqrt(252 * pd.Series(np.asarray(r) ** 2).rolling(n).mean().to_numpy())
