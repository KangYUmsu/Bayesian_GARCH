"""Milestone 12: isolate posterior-uncertainty sizing from Bayesian center estimates.

Uses existing Milestone 10 equity_curves.csv; NO GARCH/MCMC refits.
Ex-post exposure and realized-volatility matching are diagnostics, NOT tradable rules.
The three Bayesian strategies must come from the SAME original posterior fits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from .portfolio import simulate_portfolio
from .exposure_attribution import _series, _calibrate_multiplier

METHODS = ('bayes_meanplug', 'bayes_integrated')
REFERENCE = 'bayes_upper90'


def _nw_mean_inference(x, max_lag=None):
    """Two-sided, approximate normal HAC inference for mean daily paired net return.

    Lag choice is a descriptive sensitivity choice; this does not provide
    inference on path-dependent drawdown or Sharpe differences.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 5 or not np.isfinite(x).all():
        raise ValueError('Need >=5 finite paired returns')
    n = len(x)
    lag = min(n - 1, max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))) if max_lag is None else int(max_lag)
    if not (0 <= lag < n):
        raise ValueError('Invalid HAC lag')
    z = x - x.mean()
    lrvar = float(z @ z / n)
    for k in range(1, lag + 1):
        lrvar += 2 * (1 - k / (lag + 1)) * float(z[k:] @ z[:-k] / n)
    se = float(np.sqrt(max(lrvar, 0) / n))
    mean = float(x.mean())
    score = mean / se if se > 0 else float('nan')
    return {
        'mean_daily_difference': mean, 'hac_standard_error': se,
        'hac_lags': lag, 'hac_z': score,
        'hac_p_two_sided': float(2 * norm.sf(abs(score))) if np.isfinite(score) else float('nan'),
        'hac_ci95_lower': mean - 1.96 * se,
        'hac_ci95_upper': mean + 1.96 * se,
    }


def run_ablation(curves, *, cost_bps=5., hac_lags=None):
    """Return (strategy summary, dated replay curves, paired differences, diagnostics).

    Portfolio return timing and transaction costs use the same simulator as M10.
    Every comparator is a Bayesian estimator based on the same fitted posterior.
    Controls scale the comparator's *daily weight path* and preserve its timing.
    """
    required = {'date', 'strategy', 'asset_return', 'risk_free_return', 'weight'}
    if missing := required - set(curves.columns):
        raise ValueError(f'Missing curve columns: {sorted(missing)}')
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError('Invalid transaction costs')
    available = set(curves.strategy.unique())
    for name in (REFERENCE, *METHODS):
        if name not in available:
            raise ValueError(f'Missing required Bayesian strategy {name!r}; rerun Notebook 10 if it was not saved')
    ref = _series(curves, REFERENCE)
    dates = pd.DatetimeIndex(pd.to_datetime(ref.date))
    if len(ref) < 5:
        raise ValueError('Need >=5 observations')
    a = ref.asset_return.to_numpy(float)
    rf = ref.risk_free_return.to_numpy(float)
    w_ref = ref.weight.to_numpy(float)
    if not all(np.isfinite(v).all() for v in (a, rf, w_ref)):
        raise ValueError('Nonfinite portfolio inputs')
    originals = {REFERENCE: w_ref}
    for name in METHODS:
        f = _series(curves, name)
        if len(f) != len(ref) or not dates.equals(pd.DatetimeIndex(pd.to_datetime(f.date))):
            raise ValueError(f'Misaligned dates for {name}')
        if not np.allclose(f.asset_return.to_numpy(float), a, rtol=0, atol=1e-12):
            raise ValueError(f'Mismatched asset returns for {name}')
        if not np.allclose(f.risk_free_return.to_numpy(float), rf, rtol=0, atol=1e-12):
            raise ValueError(f'Mismatched cash returns for {name}')
        originals[name] = f.weight.to_numpy(float)
    ref_summary, _ = simulate_portfolio(a, rf, w_ref, cost_bps=cost_bps, dates=dates)
    target_exposure = ref_summary['mean_exposure']
    target_volatility = ref_summary['realized_annual_volatility']
    weight_paths = {'bayes_upper90_original': w_ref,
                    'bayes_meanplug_original': originals['bayes_meanplug'],
                    'bayes_integrated_original': originals['bayes_integrated'],
                    'static_mean_exposure_expost': np.full(len(a), target_exposure)}
    fit_info = {}
    for name in METHODS:
        path = originals[name]
        matched, c_exposure, residual = _calibrate_multiplier(path, target_exposure, 'exposure')
        key = f'{name}_exposure_matched_expost'
        weight_paths[key] = matched
        riskmatched, c_vol, residual_vol = _calibrate_multiplier(
            path, target_volatility, 'volatility',
            returns=a, cash=rf, dates=dates, cost_bps=cost_bps)
        key_risk = f'{name}_volatility_matched_expost'
        weight_paths[key_risk] = riskmatched
        fit_info[name] = {'exposure_multiplier': c_exposure,
                          'exposure_matching_residual': residual,
                          'volatility_multiplier': c_vol,
                          'volatility_matching_residual': residual_vol}
    summaries, daily = [], []
    for name, weights in weight_paths.items():
        summary, curve = simulate_portfolio(a, rf, weights, cost_bps=cost_bps, dates=dates)
        summaries.append({'strategy': name, **summary,
                          'hindsight_control': name.endswith('_expost'),
                          'exposure_gap_to_upper90': summary['mean_exposure'] - target_exposure,
                          'volatility_gap_to_upper90': summary['realized_annual_volatility'] - target_volatility})
        curve.insert(1, 'strategy', name)
        daily.append(curve)
    summary_frame = pd.DataFrame(summaries)
    daily_frame = pd.concat(daily, ignore_index=True)
    pivot_ret = daily_frame.pivot(index='date', columns='strategy', values='portfolio_return')
    pivot_w = daily_frame.pivot(index='date', columns='strategy', values='weight')
    paired_frames = []
    for comparator in (f'{s}_{m}_expost' for s in METHODS for m in ('exposure_matched', 'volatility_matched')):
        diff = pivot_ret['bayes_upper90_original'].to_numpy() - pivot_ret[comparator].to_numpy()
        daily_pair = pd.DataFrame({
            'date': dates, 'comparison': comparator,
            'asset_return': a, 'risk_free_return': rf,
            'upper90_weight': pivot_w['bayes_upper90_original'].to_numpy(),
            'comparator_weight': pivot_w[comparator].to_numpy(),
            'upper90_portfolio_return': pivot_ret['bayes_upper90_original'].to_numpy(),
            'comparator_portfolio_return': pivot_ret[comparator].to_numpy(),
            'daily_net_return_difference': diff,
            'down_market': a < 0,
        })
        paired_frames.append(daily_pair)
    paired = pd.concat(paired_frames, ignore_index=True)
    inference = []
    for name, sub in paired.groupby('comparison', sort=True):
        d = sub.daily_net_return_difference.to_numpy(float)
        estimates = _nw_mean_inference(d, max_lag=hac_lags)
        estimates['comparison'] = name
        estimates['mean_daily_difference_down_market'] = float(d[sub.down_market.to_numpy()].mean()) if sub.down_market.any() else float('nan')
        estimates['mean_daily_difference_up_market'] = float(d[~sub.down_market.to_numpy()].mean()) if (~sub.down_market).any() else float('nan')
        inference.append(estimates)
    # Verify saved portfolio replay with same cost. Catch changed accounting conventions.
    max_replay_error = 0.
    for name, orig in ((REFERENCE, 'bayes_upper90_original'),
                       ('bayes_meanplug', 'bayes_meanplug_original'),
                       ('bayes_integrated', 'bayes_integrated_original')):
        sub = _series(curves, name)
        replay = daily_frame.loc[daily_frame.strategy == orig].sort_values('date')
        if 'portfolio_return' in sub:
            err = float(np.max(np.abs(sub.portfolio_return.to_numpy(float) - replay.portfolio_return.to_numpy(float))))
            max_replay_error = max(max_replay_error, err)
            if err > 1e-9:
                raise ValueError(f'Replay mismatch for {name}: max net-return difference {err:.3g}; check cost_bps and original backtest version')
    diagnostics = {
        'n_days': len(dates), 'cost_bps': cost_bps,
        'reference_exposure': target_exposure,
        'reference_volatility': target_volatility,
        'matching': fit_info,
        'max_original_replay_return_error': max_replay_error,
        'paired_hac': inference,
        'interpretation': 'EX-POST diagnostic only; multipliers use complete test period',
        'limitations': [
            'Same posterior-based point estimates, but comparing posterior upper quantile against a point estimate also changes conservatism and allocation timing',
            'Mean-exposure matching is not daily risk matching; volatility matching uses future realized volatility',
            'Upper-90% GARCH variance is parameter-uncertainty conditional-variance quantile, not automatically a calibrated return VaR',
            'HAC inference applies to mean paired daily net returns only, not Sharpe or drawdown differences',
            'Eight overlapping windows for four ETFs are not eight independent experiments',
        ],
    }
    return summary_frame, daily_frame, paired, diagnostics


def analyze_saved(folder, *, output=None, cost_bps=None, hac_lags=None):
    folder = Path(folder)
    if not (folder / 'equity_curves.csv').is_file():
        raise FileNotFoundError(folder / 'equity_curves.csv')
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if cost_bps is None:
        cost_bps = report.get('trading_cost_bps', 5.)
    raw = pd.read_csv(folder / 'equity_curves.csv')
    summaries, curves, paired, diag = run_ablation(raw, cost_bps=cost_bps, hac_lags=hac_lags)
    diag.update({'asset': report.get('asset', 'unknown'), 'window': report.get('window', 'unknown'),
                 'prior': report.get('prior', 'unknown'),
                 'original_convergence_validated': report.get('results_validated', None)})
    out = Path(output) if output is not None else folder / 'posterior_uncertainty_ablation'
    out.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(out / 'strategy_summary.csv', index=False)
    curves.to_csv(out / 'equity_curves.csv', index=False)
    paired.to_csv(out / 'paired_daily_returns.csv', index=False)
    pd.DataFrame(diag['paired_hac']).to_csv(out / 'paired_hac_summary.csv', index=False)
    (out / 'diagnostics.json').write_text(json.dumps(diag, indent=2))
    return summaries, paired, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--cost-bps', type=float)
    parser.add_argument('--hac-lags', type=int)
    args = parser.parse_args()
    for folder in args.folders:
        summary, _, diag = analyze_saved(folder, cost_bps=args.cost_bps, hac_lags=args.hac_lags)
        print(f'\n{folder}, validated={diag["original_convergence_validated"]}')
        print(summary[['strategy','mean_exposure','realized_annual_volatility', 'sharpe_excess_cash', 'max_drawdown']].to_string(index=False))

if __name__ == '__main__':
    main()
