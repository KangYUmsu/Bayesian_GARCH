"""Market data and return construction.

Prices: ``data/<TICKER>_adjusted.csv`` = Yahoo Finance daily closes downloaded with
``yfinance`` and ``auto_adjust=True`` (split- and dividend-adjusted), by
``src.market_uncertainty_backtest.download_assets``. Cash: FRED series DFF
(effective federal funds rate, annual percent). These files are read in place and
are NOT included in this repository (Yahoo data may not be redistributed).

Return definition (identical to ``agent_research/code/walkforward.load_returns``
and ``src.market_uncertainty_backtest.run_market``):
    r_t = log(P_t / P_{t-1})   (daily close-to-close log return, decimal units)
"""
import numpy as np
import pandas as pd

from .paths import DATA
from src.real_backtest import read_price_csv, read_rate_csv, cash_returns_for_periods  # noqa: F401  (re-exported)

ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD']
DESCRIPTIONS = {
    'SPY': 'SPDR S&P 500 ETF -- large-cap US equities',
    'QQQ': 'Invesco QQQ -- Nasdaq-100, technology-heavy US equities',
    'TLT': 'iShares 20+ Year Treasury Bond ETF -- long-duration US government bonds',
    'GLD': 'SPDR Gold Shares -- physical gold',
}


def load_prices(asset):
    """Adjusted close series indexed by date (validated by src.real_backtest.read_price_csv)."""
    return read_price_csv(DATA / f'{asset}_adjusted.csv')


def log_returns(asset):
    p = load_prices(asset)
    return np.log(p / p.shift(1)).dropna()


def load_rates():
    """DFF in annual percent, indexed by date."""
    return read_rate_csv(DATA / 'DFF.csv')


def all_returns():
    return pd.DataFrame({a: log_returns(a) for a in ASSETS})
