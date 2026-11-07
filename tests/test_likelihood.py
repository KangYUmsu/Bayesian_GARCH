"""GARCH likelihood, recursion and parameter transformation."""
import numpy as np
import pytest
from scipy.special import expit

from bgarch.data import log_returns
from bgarch.garch import (loglik_u, encode, to_natural, loglik_reference, variance_path,
                          variance_path_reference, simulate, mle_fit, P_MAX_MLE)

R = log_returns('SPY').to_numpy()[:800]


@pytest.mark.parametrize('V,p,q', [(1e-4, .98, .1), (2e-4, .9, .25), (5e-5, .999, .05), (1e-4, .5, .5)])
def test_loglik_matches_plain_python(V, p, q):
    u = encode(V, p, q)
    assert abs(loglik_u(R, u) - loglik_reference(R, *to_natural(u))) < 1e-8


def test_transformation_roundtrip():
    u = encode(1.3e-4, .97, .08)
    om, al, be = to_natural(u)
    assert np.isclose(al + be, .97) and np.isclose(al / (al + be), .08) and np.isclose(om / (1 - .97), 1.3e-4)


def test_variance_path_matches_reference():
    u = encode(1e-4, .97, .1)
    assert np.allclose(variance_path(R, 500, u), variance_path_reference(R, 500, *to_natural(u)), rtol=1e-13, atol=0)


def test_mle_recovers_simulated_parameters():
    r, _ = simulate(n=4000, seed=7, V=1e-4, alpha=.08, beta=.9)
    u, fit = mle_fit(r)
    om, al, be = to_natural(u)
    assert fit.success and abs(al - .08) < .03 and abs(be - .9) < .04


def test_mle_bound_is_documented_value():
    assert np.isclose(P_MAX_MLE, expit(7.0)) and 0.99908 < P_MAX_MLE < 0.99909
