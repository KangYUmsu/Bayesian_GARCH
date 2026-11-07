"""Run every walk-forward job of the primary investigation in parallel.

Jobs (asset x window = 8 configurations each):
  repro    : 2023-01-03 -> end, moderate prior, seed 2026      (reproduces Milestone 10)
  ext      : 2017-01-03 -> 2022-12-30, moderate prior, seed 102026 (new OOS replication)
  seedA/B  : 2023 segment with seeds 11, 12; 2017 segment with seed 100011 (Monte Carlo noise)
  broad    : broad prior, both segments (prior sensitivity)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import walkforward as wf  # noqa: E402

ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD']
WINDOWS = [500, 1000]


def jobs():
    out = []
    for a in ASSETS:
        for w in WINDOWS:
            out += [dict(asset=a, window=w, start='2023-01-01', end=None, prior_name='moderate', seed=2026),
                    dict(asset=a, window=w, start='2017-01-01', end='2023-01-01', prior_name='moderate', seed=102026),
                    dict(asset=a, window=w, start='2023-01-01', end=None, prior_name='moderate', seed=11),
                    dict(asset=a, window=w, start='2023-01-01', end=None, prior_name='moderate', seed=12),
                    dict(asset=a, window=w, start='2017-01-01', end='2023-01-01', prior_name='moderate', seed=100011),
                    dict(asset=a, window=w, start='2023-01-01', end=None, prior_name='broad', seed=2026),
                    dict(asset=a, window=w, start='2017-01-01', end='2023-01-01', prior_name='broad', seed=102026)]
    return out


def _one(kw):
    tag = f"{kw['asset']}_{kw['window']}_{kw['prior_name']}_s{kw['seed']}_{kw['start'][:4]}"
    if (wf.OUT / f'{tag}_meta.json').exists():
        return json.loads((wf.OUT / f'{tag}_meta.json').read_text())
    a = dict(kw); asset = a.pop('asset'); window = a.pop('window')
    return wf.run(asset, window, tag=tag, **a)


if __name__ == '__main__':
    t0 = time.time()
    with Pool(7) as pool:
        for m in pool.imap_unordered(_one, jobs()):
            print(json.dumps(m), flush=True)
    print(f'TOTAL_SECONDS {time.time() - t0:.1f}', flush=True)
