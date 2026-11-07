"""Experiment configuration of the course study (not part of the reusable ``bgarch`` library).

Segments (first forecast date, exclusive end, master seed) of the primary moderate-prior jobs:
    ext  = 2017-01-03 .. 2022-12-30  independent evaluation period (seed 102026)
    test = 2023-01-03 .. 2026-11-04  previously inspected period    (seed 2026)
Rolling estimation windows of 500 and 1000 returns, refits every 63 trading days.
"""
import os
from pathlib import Path

from bgarch.paths import ROOT, PROJECT, COURSE, DATA  # noqa: F401
from bgarch.data import ASSETS  # noqa: F401

SEGMENTS = {'ext': ('2017-01-01', '2023-01-01', 102026), 'test': ('2023-01-01', None, 2026)}
WINDOWS = [500, 1000]
SEGS = ['ext', 'test']
SEG_FILES = {'ext': ('2017', 102026), 'test': ('2023', 2026)}   # artifact file names: year of start, seed
REFIT_DAYS = 63
CHAINS, WARMUP, DRAWS, SAMPLED_DRAWS = 4, 2500, 4000, 300
PRIOR = 'moderate'

# Saved rolling-experiment artifacts. The daily files contain returns derived from market data and are NOT
# distributed with the repository: place them in artifacts/forecasts and artifacts/milestone10_market, or point
# the environment variables below at a private copy (see the README, "Data and saved artifacts").
SAVED_FORECASTS = Path(os.environ.get('BGARCH_FORECASTS_DIR', ROOT / 'artifacts' / 'forecasts')).resolve()
SAVED_M10 = Path(os.environ.get('BGARCH_M10_DIR', ROOT / 'artifacts' / 'milestone10_market')).resolve()
SAVED_PRIMARY = ROOT / 'agent_research' / 'results' / 'primary'          # aggregate tables of the research run
RESULTS = ROOT / 'artifacts' / 'results'
FIGURES = ROOT / 'artifacts' / 'figures'
REPORT_SOURCE = ROOT / 'reports' / 'source'                                # LaTeX sources and generated tables

# Manifest keys are repository-relative; the two restricted artifact folders may live elsewhere.
_ALIASES = {'artifacts/forecasts': SAVED_FORECASTS, 'artifacts/milestone10_market': SAVED_M10}


def resolve_artifact(key):
    """Repository-relative manifest key -> actual path (honours the environment overrides)."""
    for prefix, base in _ALIASES.items():
        if key == prefix or key.startswith(prefix + '/'):
            return base / key[len(prefix):].lstrip('/')
    return ROOT / key


def artifact_key(path):
    """Actual path -> repository-relative manifest key (inverse of resolve_artifact)."""
    path = Path(path).resolve()
    for prefix, base in _ALIASES.items():
        try:
            return f'{prefix}/{path.relative_to(base).as_posix()}'
        except ValueError:
            pass
    return path.relative_to(ROOT).as_posix()


def rel(path):
    """Path for display: repository-relative when possible, otherwise the manifest-style key or the name."""
    try:
        return artifact_key(path)
    except ValueError:
        return Path(path).name
