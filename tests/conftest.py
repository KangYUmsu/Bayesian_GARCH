"""Test configuration.

Most tests compare against the saved research run or use the market data. Those inputs are not distributed
with the repository (see the README, "Data and saved artifacts"). When they are absent, the affected test
modules are skipped with a reason instead of failing; everything that needs only public inputs still runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import bgarch.paths  # noqa: E402,F401  (puts the repository root first on sys.path for `import src`)
from bgarch.data import ASSETS  # noqa: E402
from bgarch.paths import DATA  # noqa: E402
from bgarch_experiments.config import SAVED_FORECASTS, SAVED_M10  # noqa: E402

HAVE_PRICES = all((DATA / f'{a}_adjusted.csv').exists() for a in ASSETS)
HAVE_FORECASTS = SAVED_FORECASTS.is_dir() and any(SAVED_FORECASTS.glob('*_daily.csv.gz'))
HAVE_M10 = SAVED_M10.is_dir() and any(SAVED_M10.iterdir())

NEEDS = {   # module -> restricted inputs it reads (module-level code reads the prices)
    'test_adequacy.py': HAVE_PRICES and HAVE_FORECASTS,
    'test_forecast_timing.py': HAVE_PRICES,
    'test_library.py': HAVE_PRICES and HAVE_FORECASTS,
    'test_likelihood.py': HAVE_PRICES,
    'test_reproduction.py': HAVE_PRICES and HAVE_FORECASTS and HAVE_M10,
}
collect_ignore = [m for m, ok in NEEDS.items() if not ok]


def pytest_report_header(config):
    return (f'restricted inputs: prices={HAVE_PRICES}, saved forecasts={HAVE_FORECASTS}, '
            f'milestone-10 outputs={HAVE_M10}; modules not collected: {collect_ignore or "none"}')


def pytest_collection_modifyitems(config, items):
    import pytest
    if not HAVE_FORECASTS:
        skip = pytest.mark.skip(reason='saved rolling forecasts not available (artifacts/forecasts or $BGARCH_FORECASTS_DIR)')
        for it in items:
            if it.name == 'test_all_saved_jobs_pass_structure_and_timing_checks':
                it.add_marker(skip)
