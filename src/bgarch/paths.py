"""Repository root, validated numerical core, and local market-data location.

``bgarch`` re-exports the validated numerical core of the original project (the package ``src`` in this
repository: ``src/garch.py``, ``src/portfolio.py``, ...) instead of re-implementing it. Importing this module
puts the repository root at ``sys.path[0]`` so that ``import src`` resolves to it. This works from a source
checkout and after an editable install (``pip install -e .``), which keeps the files in place.

Market data are not redistributed. Prices are read from ``data/`` (or from ``$BGARCH_DATA_DIR``); see the README,
"Data and saved artifacts". Experiment-specific locations (saved artifacts, results, figures) live in
``bgarch_experiments.config``.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # repository root (Bayesian_GARCH/)
PROJECT = ROOT                                         # names kept so existing imports keep working
COURSE = ROOT
DATA = Path(os.environ.get('BGARCH_DATA_DIR', ROOT / 'data')).resolve()

if not (ROOT / 'src' / 'garch.py').exists():
    raise ImportError(f'bgarch needs the validated core package at {ROOT / "src"}; '
                      'use a source checkout with an editable install (pip install -e .).')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
