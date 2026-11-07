"""Download the market data used by the study (network access required; thin wrapper, no new logic).

    pip install yfinance
    python scripts/download_data.py                       # SPY, QQQ, TLT, GLD (Yahoo) into data/
    python scripts/download_data.py --dff                 # also refresh data/DFF.csv from FRED
    python scripts/download_data.py --end 2026-11-05      # stop at a fixed date (exclusive, yfinance semantics)

Uses the original project's downloaders unchanged: ``src.market_uncertainty_backtest.download_assets``
(yfinance, ``auto_adjust=True``, written as ``data/<TICKER>_adjusted.csv``) and ``src.real_backtest.download_fred_dff``.

Important: Yahoo revises adjusted closes over time, so a fresh download will not reproduce the checksums of the
saved research artifacts, and every result can change slightly. Quick reproduction therefore needs the original
data snapshot; with fresh data, use full reproduction (``python -m bgarch_experiments.reproduce full``) and expect
small numerical differences. Yahoo data may not be redistributed: do not commit the downloaded files.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bgarch.paths import DATA  # noqa: E402  (also makes `import src` resolve to the validated core)
from bgarch.data import ASSETS  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Download ETF prices (Yahoo) and optionally DFF (FRED) into data/.')
    ap.add_argument('--start', default='2013-01-01', help='first date requested (the study uses 2013-01-01)')
    ap.add_argument('--end', default=None, help='end date passed to yfinance (exclusive); the study sample ends 2026-11-04')
    ap.add_argument('--dff', action='store_true', help='also download FRED DFF into data/DFF.csv')
    a = ap.parse_args()
    from src.market_uncertainty_backtest import download_assets
    from src.real_backtest import download_fred_dff
    print('prices:', download_assets(ASSETS, DATA, start=a.start, end=a.end))
    if a.dff:
        print('DFF rows:', download_fred_dff(DATA / 'DFF.csv', start='1954-07-01'))   # full history, as in the saved file
