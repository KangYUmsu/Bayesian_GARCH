"""Reusable bgarch interfaces: transformations, forecasts, posterior summaries, diagnostics, exposure control."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from bgarch import garch, bayes, forecast, diagnostics as dg, portfolio
from bgarch.data import log_returns
from bgarch_experiments.config import SAVED_FORECASTS
from bgarch_experiments.tables import Store

R = log_returns('SPY').to_numpy()


# --------------------------------------------------------------------------- transformations
@pytest.mark.parametrize('om,al,be', [(2e-6, .08, .9), (1e-5, .2, .5), (3e-7, .03, .965)])
def test_from_natural_inverts_to_natural(om, al, be):
    u = garch.from_natural(om, al, be)
    assert np.allclose(garch.to_natural(u), (om, al, be), rtol=1e-12, atol=0)


def test_every_u_is_a_valid_stationary_model():
    for u in np.random.default_rng(0).normal(0, 4, size=(500, 3)):
        om, al, be = garch.to_natural(u)
        assert om > 0 and al >= 0 and be >= 0 and al + be < 1


def test_from_natural_rejects_nonstationary():
    with pytest.raises(ValueError):
        garch.from_natural(1e-6, .1, .95)


def test_half_life_and_next_variance():
    assert np.isclose(garch.half_life(.5), 1.0) and np.isclose(garch.half_life(.98), np.log(.5) / np.log(.98))
    u = garch.encode(1e-4, .97, .1); tr = R[:500]
    om, al, be = garch.to_natural(u)
    h = garch.variance_path_reference(np.r_[tr, 0.0], 500, om, al, be)[0]
    assert np.isclose(garch.next_variance(tr, u), h, rtol=1e-13)


def test_k_step_forecast_reverts_geometrically():
    """E[h_{t+k}] - V = p^(k-1)(h_{t+1} - V): iterate the recursion with r^2 replaced by its expectation h."""
    V, p, q = 1e-4, .95, .1; om, al, be = garch.to_natural(garch.encode(V, p, q))
    h1 = 4e-4; h = h1
    for k in range(1, 30):
        assert np.isclose(h - V, p ** (k - 1) * (h1 - V), rtol=1e-12)
        h = om + (al + be) * h


# --------------------------------------------------------------------------- forecasts
def test_predictive_summary_reproduces_saved_columns():
    st = Store(SAVED_FORECASTS); f = st.daily('SPY', 500, 'test')
    z = np.load(SAVED_FORECASTS / f'{st.name("SPY", 500, "test")}_draws.npz')
    v = z['v'][:, :63].astype(float)                  # saved as float32
    s = forecast.predictive_summary(v)
    assert np.allclose(s['integrated'], f.variance_bayes_integrated[:63], rtol=1e-6)
    assert np.allclose(s['q90'], f.variance_bayes_q90[:63], rtol=1e-6)
    assert np.allclose(forecast.mixture_logpdf(f.return_log.to_numpy()[:63], v), f.logscore_bayes_mixture[:63], atol=1e-6)


def test_mixture_density_cdf_and_var_are_consistent():
    rng = np.random.default_rng(3); v = rng.uniform(.5e-4, 2e-4, size=(300, 4)); y = np.array([-.03, -.01, .0, .02])
    dens = np.exp(forecast.mixture_logpdf(y, v))
    assert np.allclose(dens, norm.pdf(y[None, :], scale=np.sqrt(v)).mean(0), rtol=1e-12)
    var = forecast.mixture_var(v, .01)
    assert np.allclose(forecast.mixture_cdf(-var, v), .01, atol=1e-10)
    # one-draw mixture is the Gaussian plug-in
    assert np.allclose(forecast.mixture_logpdf(y, v[:1]), forecast.gaussian_logpdf(y, v[0]))


def test_mixture_excess_kurtosis_formula():
    rng = np.random.default_rng(4); v = rng.lognormal(np.log(1e-4), .3, 400)
    x = rng.normal(size=(4000, 400)) * np.sqrt(v)       # draws from the mixture
    k_mc = pd.Series(x.ravel()).kurt()
    assert abs(forecast.mixture_excess_kurtosis(v) - k_mc) < .05
    assert np.isclose(forecast.mixture_excess_kurtosis(np.full(5, 2e-4)), 0, atol=1e-12)


# --------------------------------------------------------------------------- posterior summaries
def test_natural_draws_and_posterior_summary():
    u = np.random.default_rng(5).normal([np.log(1e-4), 3, -2], .2, size=(2, 1000, 3))
    d = bayes.natural_draws(u.reshape(-1, 3))
    assert np.allclose(d.alpha + d.beta, d.persistence) and np.allclose(d.omega, d.long_run_variance * (1 - d.persistence))
    assert np.allclose(d.half_life_days, garch.half_life(d.persistence))
    s = bayes.posterior_summary(u)
    assert np.isclose(s.loc['persistence', 'mean'], expit(u[..., 1]).mean())
    assert (s.q05 <= s.q50).all() and (s.q50 <= s.q95).all()


def test_posterior_mean_plug_in_uses_natural_means():
    u = np.random.default_rng(6).normal([np.log(1e-4), 2, -2], .5, size=(800, 3))
    pm = bayes.posterior_mean_u(u)
    assert np.isclose(np.exp(pm[0]), np.exp(u[:, 0]).mean()) and np.isclose(expit(pm[1]), expit(u[:, 1]).mean())


# --------------------------------------------------------------------------- diagnostics
def test_pit_functions():
    rng = np.random.default_rng(7); h = rng.uniform(1e-5, 4e-4, 2000); r = rng.normal(size=2000) * np.sqrt(h)
    u = dg.pit_gaussian(r, h)
    assert abs(u.mean() - .5) < .02 and np.allclose(dg.normal_scores(u), r / np.sqrt(h), atol=1e-8)
    assert np.allclose(dg.pit_mixture(r, np.vstack([h, h])), u)
    assert np.allclose(dg.pit_t(r, h, np.full(2000, 1e6)), u, atol=1e-5)    # t -> Gaussian as nu -> inf


# --------------------------------------------------------------------------- exposure control
def test_exposure_matched_weights_is_causal_and_matches_definition():
    rng = np.random.default_rng(8); hp = rng.uniform(5e-5, 3e-4, 600); hc = hp * rng.uniform(1.05, 1.4, 600)
    w = portfolio.exposure_matched_weights(hp, hc)
    k = portfolio.trailing_mean_known(np.sqrt(hp / hc))
    assert np.allclose(w, np.minimum(1, k * .10 / np.sqrt(252 * hp)))
    hc2 = hc.copy(); hc2[400:] *= 3                     # change the future only
    assert np.array_equal(portfolio.exposure_matched_weights(hp, hc2)[:401], w[:401])
