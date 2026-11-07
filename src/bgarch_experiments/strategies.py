"""The project's strategy set (experiment configuration).

static_weight, strategy_weights: COPIED VERBATIM from agent_research/code/analysis_primary.py;
the control weights now call ``bgarch.portfolio.exposure_matched_weights`` (identical arithmetic).
"""
import numpy as np
import pandas as pd

from bgarch.data import log_returns as load_returns
from bgarch.portfolio import vol_weight, exposure_matched_weights, TARGET
from .config import SEGMENTS


def static_weight(asset, seg):
    """Constant weight targeting 10% volatility with the sample sd of ALL returns before the segment starts."""
    r = load_returns(asset); start = SEGMENTS[seg][0]
    dev = r[r.index < pd.Timestamp(start)].to_numpy()
    return min(1., TARGET / (np.std(dev, ddof=1) * np.sqrt(252)))


def strategy_weights(f, asset, seg):
    """Weights (len(f),) of every strategy compared in the report, from one saved forecast frame."""
    v = lambda k: f[f'variance_{k}'].to_numpy()
    W = {'buy_hold': np.ones(len(f)), 'static_target': np.full(len(f), static_weight(asset, seg)),
         'ewma': vol_weight(v('ewma')), 't_garch': vol_weight(v('t')), 'mle': vol_weight(v('mle')),
         'map': vol_weight(v('map')), 'bayes_meanplug': vol_weight(v('bayes_meanplug')),
         'bayes_integrated': vol_weight(v('bayes_integrated')), 'bayes_upper90': vol_weight(v('bayes_q90')),
         'laplace_upper90': vol_weight(v('laplace_q90'))}
    W['mle_const_rt'] = exposure_matched_weights(v('mle'), v('bayes_q90'))
    W['meanplug_const_rt'] = exposure_matched_weights(v('bayes_meanplug'), v('bayes_q90'))
    return W
