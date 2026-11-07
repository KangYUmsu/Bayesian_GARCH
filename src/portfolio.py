"""Shared SPY/cash portfolio accounting with positive cash accrual and trading costs.

All input returns are SIMPLE holding-period returns. Target weights apply to the
period immediately following a completed rebalance; initial holdings are 100% cash.
Costs reduce wealth before the holding period. Weight drift uses BOTH asset and
cash returns so turnover is calculated from the correct pre-trade allocation.
"""
import numpy as np
import pandas as pd


def constant_cash_returns(n, annual_rate=0.04, periods_per_year=252):
    """Constant effective annual cash yield converted to trading-period returns.

    This is an explicitly artificial cash scenario; it is not historical FRED data.
    """
    if n < 1 or periods_per_year <= 0 or not np.isfinite(annual_rate) or annual_rate <= -1:
        raise ValueError('invalid annual cash rate or number of periods')
    return np.full(n, np.expm1(np.log1p(annual_rate) / periods_per_year))


def simulate_portfolio(asset_returns, cash_returns, target_weights, cost_bps=5., dates=None):
    """Return period-level portfolio series and aggregate performance metrics.

    Accounting convention: initial wealth is 1 entirely in cash; at beginning of
    each period rebalance from *pre-trade* weight to target weight, pay proportional
    one-way costs, then earn weighted asset and cash simple returns. Wealth cost
    applied first. No leverage/shorting. Outputs include excess-return Sharpe.
    """
    a=np.asarray(asset_returns,dtype=float)
    f=np.asarray(cash_returns,dtype=float)
    w=np.asarray(target_weights,dtype=float)
    if a.ndim!=1 or f.shape!=a.shape or w.shape!=a.shape or len(a)<2:
        raise ValueError('asset, cash and weights must be matching 1D arrays of length >=2')
    if not np.isfinite(a).all() or not np.isfinite(f).all() or not np.isfinite(w).all():
        raise ValueError('non-finite portfolio input')
    if np.any(a <= -1) or np.any(f <= -1) or np.any(w < 0) or np.any(w > 1):
        raise ValueError('invalid asset/cash simple return or portfolio weight')
    if not np.isfinite(cost_bps) or not (0 <= cost_bps < 10000):
        raise ValueError('invalid cost_bps')
    equity=np.empty(len(a)); rr=np.empty(len(a)); tr=np.empty(len(a)); interest=np.empty(len(a))
    previous_end_weight=0.0
    wealth=1.0
    for t in range(len(a)):
        turnover=abs(w[t]-previous_end_weight)
        cost_fraction=turnover*cost_bps/10000.0
        gross_return=w[t]*a[t]+(1-w[t])*f[t]
        net_return=(1-cost_fraction)*(1+gross_return)-1
        wealth*=1+net_return
        rr[t]=net_return;tr[t]=turnover;equity[t]=wealth
        interest[t]=(1-w[t])*f[t]
        previous_end_weight=w[t]*(1+a[t])/(1+gross_return)
    peaks=np.maximum.accumulate(np.r_[1.0,equity])[1:]
    excess=rr-f
    annual_vol=np.std(rr,ddof=1)*np.sqrt(252)
    excess_vol=np.std(excess,ddof=1)*np.sqrt(252)
    summary={
        'total_return':float(wealth-1),
        'annualized_return':float(wealth**(252/len(a))-1),
        'realized_annual_volatility':float(annual_vol),
        'sharpe_excess_cash':float(np.mean(excess)*np.sqrt(252)/np.std(excess,ddof=1)) if excess_vol>0 else None,
        'max_drawdown':float(np.min(equity/peaks-1)),
        'mean_exposure':float(np.mean(w)),
        'mean_cash_weight':float(np.mean(1-w)),
        'mean_daily_turnover':float(np.mean(tr)),
        'cash_only_total_return':float(np.prod(1+f)-1),
        'mean_daily_cash_contribution':float(np.mean(interest)),
    }
    curve=pd.DataFrame({
        'portfolio_return':rr, 'risk_free_return':f,'asset_return':a,
        'excess_return':excess,'cash_contribution':interest,
        'equity':equity,'weight':w,'cash_weight':1-w,'turnover':tr,
    })
    if dates is not None:
        if len(dates)!=len(a):
            raise ValueError('dates length mismatch')
        curve.insert(0,'date',pd.DatetimeIndex(dates))
    return summary,curve
