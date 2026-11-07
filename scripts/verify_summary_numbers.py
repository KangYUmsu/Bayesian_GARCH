"""Check every number in executive_summary.tex against the saved, validated course-project results.

    python scripts/verify_summary_numbers.py

Sources: results/quick/tables (identical to the research tables), figures/narrative/narrative_numbers.json,
course/presentations/source/figures/numbers.json, the saved forecast files, both executed notebooks and the pytest suite.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bgarch_experiments.config import SAVED_FORECASTS  # noqa: E402
COURSE = ROOT
QT = ROOT / 'artifacts' / 'results' / 'quick' / 'tables'
NN = json.loads((ROOT / 'artifacts' / 'figures' / 'narrative' / 'narrative_numbers.json').read_text())
N5 = json.loads((ROOT / 'course' / 'presentations' / 'source' / 'figures' / 'numbers.json').read_text())
checks = []
claim = lambda sec, text, ok: checks.append((sec, text, bool(ok)))
r = round

fp = {s: pd.read_csv(QT / f'forecast_pooled_{s}.csv').set_index(['contrast', 'window']) for s in ('ext', 'test')}
row = lambda k, w, s='ext': fp[s].loc[(k, w)]
ci = lambda k, w, s='ext': (row(k, w, s).pooled_mean - 1.96 * row(k, w, s).pooled_mean / row(k, w, s).z,
                            row(k, w, s).pooled_mean + 1.96 * row(k, w, s).pooled_mean / row(k, w, s).z)
CELLS = [(s, w) for s in ('ext', 'test') for w in (500, 1000)]

# §1-3
claim(1, 'SPY lag-1 autocorrelation of squared returns 0.42', r(N5['acf1_r2_spy_full'], 2) == 0.42)
claim(2, 'Student-t median nu 7.2 over 320 refits', r(NN['nu']['median'], 1) == 7.2 and NN['nu']['n_refits'] == 320)
fs = pd.read_csv(QT / 'fit_summary.csv')
claim(3, '320 refits; 4 marginal convergence failures; MLE bound in 30', (int(fs.refits.sum()), int(fs.failed_convergence.sum()), int(fs.mle_at_bound.sum())) == (320, 4, 30))
fc = SAVED_FORECASTS
d17 = pd.read_csv(fc / 'SPY_500_moderate_s102026_2017_daily.csv.gz', parse_dates=['refit_date'])
claim(3, '1,510 and 964 evaluation days; 63-day refits',
      len(d17) == 1510 and len(pd.read_csv(fc / 'SPY_500_moderate_s2026_2023_daily.csv.gz')) == 964
      and (d17.groupby('refit_date').size().iloc[:-1] == 63).all())

# §4 H1 (Table 1)
exp_h1 = {('ext', 500): (-0.00534, (-0.0118, 0.0011), 0.105), ('ext', 1000): (-0.00558, (-0.0117, 0.0006), 0.076),
          ('test', 500): (-0.00331, (-0.0084, 0.0018), 0.206), ('test', 1000): (-0.00085, (-0.0032, 0.0015), 0.484)}
ok = all(r(row('qlike_integrated_minus_mle', w, s).pooled_mean, 5) == v[0]
         and tuple(r(x, 4) for x in ci('qlike_integrated_minus_mle', w, s)) == v[1]
         and r(row('qlike_integrated_minus_mle', w, s).p, 3) == v[2] for (s, w), v in exp_h1.items())
claim(4, 'Table 1: QLIKE differences, 95% NW intervals and p-values in all four cells', ok)
claim(4, 'no QLIKE contrast significant at 5%', all(row('qlike_integrated_minus_mle', w, s).p > .05 for s, w in CELLS))
claim(4, 'QLIKE levels about -8.3', all(r(NN[f'qlike_level_ext_{w}'][m], 1) == -8.3 for w in (500, 1000) for m in ('mle', 'bayes_integrated')))

# §5 H2 (Table 2)
exp = {'ls_mixture_minus_mle': [0.00518, 0.00469, 0.00519, 0.00226], 'ls_t_minus_mle': [0.02822, 0.02865, 0.01887, 0.01699],
       'ls_t_minus_mixture': [0.02303, 0.02396, 0.01369, 0.01473]}
claim(5, 'Table 2 gains (5 d.p.)', all([r(row(k, w, s).pooled_mean, 5) for s, w in CELLS] == v for k, v in exp.items()))
claim(5, 'Table 2 primary p-values: mixture 0.012 / 0.024; Student-t and t-mixture p < 0.001',
      [r(row('ls_mixture_minus_mle', w).p, 3) for w in (500, 1000)] == [0.012, 0.024]
      and all(row(k, w).p < .001 for k in ('ls_t_minus_mle', 'ls_t_minus_mixture') for w in (500, 1000)))
claim(5, 'caption: 2023-26 Bayesian gains p <= 0.024; Student-t comparisons p <= 0.011',
      all(row('ls_mixture_minus_mle', w, 'test').p <= .0245 for w in (500, 1000))
      and all(row(k, w, 'test').p <= .011 for k in ('ls_t_minus_mle', 'ls_t_minus_mixture') for w in (500, 1000)))
claim(5, 'ratios 5.44 6.11 3.64 7.52; Laplace 0.89 0.43 0.82 0.69 (43-89%)',
      [r(NN[f'ratio_{s}_{w}']['t_over_bayes'], 2) for s, w in CELLS] == [5.44, 6.11, 3.64, 7.52]
      and [r(NN[f'ratio_{s}_{w}']['laplace_share'], 2) for s, w in CELLS] == [0.89, 0.43, 0.82, 0.69])
claim(5, 'mixture gain range 0.002-0.005 nats/day', [r(min(exp['ls_mixture_minus_mle']), 3), r(max(exp['ls_mixture_minus_mle']), 3)] == [0.002, 0.005])
dd = NN['density_day']
claim(5, 'Figure 2: 2018-02-05, -4.3%, 3.8 MLE sd, nu 3.1, variances within 5%, density ratio 8.6',
      dd['date'] == '2018-02-05' and r(100 * dd['return'], 1) == -4.3 and r(-dd['z_mle'], 1) == 3.8 and r(dd['nu'], 1) == 3.1
      and max(dd['vol_mle'], dd['vol_bayes_int'], dd['vol_t']) / min(dd['vol_mle'], dd['vol_bayes_int'], dd['vol_t']) < 1.05
      and r(np.exp(dd['logscore_t'] - dd['logscore_mle']), 1) == 8.6)
claim(5, 'posterior CV 7-12%; day-average 3 CV^2 0.015-0.046; observed 0.4-2.9',
      [r(100 * x) for x in NN['cv_range']] == [7, 12] and [r(x, 3) for x in NN['implied_kurt_range']] == [0.015, 0.046]
      and [r(x, 1) for x in NN['observed_kurt_range']] == [0.4, 2.9])
tail = pd.read_csv(QT / 'tail_ext.csv'); t01 = tail[tail.alpha == .01].groupby('method').rate.mean()
claim(5, '1% VaR breach rates 2.2% (MLE), 2.1% (mixture), 1.5% (Student-t), 2017-22',
      [r(100 * t01[m], 1) for m in ('mle', 'bayes_mixture', 't')] == [2.2, 2.1, 1.5])

# §7 H3 (Tables 3-4)
A = pd.read_csv(QT / 'alloc_summary_ext.csv')
g = A[A.strategy.isin(['mle', 'bayes_upper90', 'mle_const_rt'])].groupby('strategy').mean(numeric_only=True)
exp3 = {'mle': (73.1, 10.37, 6.25, 0.518, -19.55, 3.43), 'bayes_upper90': (69.6, 9.81, 6.06, 0.520, -18.74, 3.50),
        'mle_const_rt': (69.4, 9.85, 6.01, 0.516, -18.66, 3.43)}
claim(7, 'Table 3 (means of 8 single-ETF configurations)', all(
    (r(100 * g.loc[k, 'mean_exposure'], 1), r(100 * g.loc[k, 'realized_annual_volatility'], 2), r(100 * g.loc[k, 'annualized_return'], 2),
     r(g.loc[k, 'sharpe_excess_cash'], 3), r(100 * g.loc[k, 'max_drawdown'], 2), r(100 * g.loc[k, 'mean_daily_turnover'], 2)) == v
    for k, v in exp3.items()))
S = A.pivot_table(index=['asset', 'window'], columns='strategy', values=['mean_exposure', 'max_drawdown', 'sharpe_excess_cash'])
e, m, sh = S['mean_exposure'], S['max_drawdown'], S['sharpe_excess_cash']
claim(7, 'upper-90: -3.6 pp exposure, smaller max DD in 8/8 by 0.82 pp', r(100 * (e.bayes_upper90 - e.mle).mean(), 1) == -3.6
      and int((m.bayes_upper90 > m.mle).sum()) == 8 and r(100 * (m.bayes_upper90 - m.mle).mean(), 2) == 0.82)
gap = 100 * (e.bayes_upper90 - e.mle_const_rt)
claim(7, 'control exposure gap +0.1 pp, range -0.06 to +0.41', r(gap.mean(), 1) == 0.1 and (r(gap.min(), 2), r(gap.max(), 2)) == (-0.06, 0.41))
claim(7, 'vs control: -0.07 pp drawdown (2 of 8), mean Sharpe difference +0.003', r(100 * (m.bayes_upper90 - m.mle_const_rt).mean(), 2) == -0.07
      and int((m.bayes_upper90 > m.mle_const_rt).sum()) == 2 and r((sh.bayes_upper90 - sh.mle_const_rt).mean(), 3) == 0.003)
PA = pd.read_csv(QT / 'alloc_pairs_ext.csv')
excl = lambda b, lo, hi: int(((PA[(PA.a == 'bayes_upper90') & (PA.b == b)][lo] > 0) | (PA[(PA.a == 'bayes_upper90') & (PA.b == b)][hi] < 0)).sum())
claim(7, 'Stage 1 CIs: drawdown excludes 0 in 8 of 8, Sharpe in 0 of 8; Stage 2: 0 of 8 for both',
      (excl('mle', 'd_maxdd_lo', 'd_maxdd_hi'), excl('mle', 'd_sharpe_lo', 'd_sharpe_hi'),
       excl('mle_const_rt', 'd_maxdd_lo', 'd_maxdd_hi'), excl('mle_const_rt', 'd_sharpe_lo', 'd_sharpe_hi')) == (8, 0, 0, 0))
st1 = (ROOT / 'reports/source/tables/exposure_stage1.tex').read_text(); st2 = (ROOT / 'reports/source/tables/exposure_stage2.tex').read_text()
claim(7, 'Stage 1 table has no matched-MLE row; Stage 2 table compares upper-90 with the control',
      'Exposure-matched' not in st1 and 'Standard MLE' in st1 and 'Exposure-matched MLE' in st2 and 'Standard MLE' not in st2)
PP = pd.read_csv(QT / 'alloc_pooled_pairs_ext.csv').set_index(['a', 'b', 'window'])
exp4 = {('mle', 500): (0.011, -0.021, 0.042, 0.76, 0.22, 1.21), ('mle', 1000): (0.000, -0.025, 0.026, 0.55, 0.11, 1.01),
        ('mle_const_rt', 500): (0.009, -0.016, 0.036, -0.01, -0.22, 0.43), ('mle_const_rt', 1000): (0.005, -0.016, 0.027, -0.03, -0.20, 0.45)}
claim(7, 'Table 4: sleeve Delta Sharpe and Delta max DD with bootstrap CIs', all(
    tuple(r(x, 3) for x in (q.d_sharpe, q.d_sharpe_lo, q.d_sharpe_hi)) + tuple(r(100 * x, 2) for x in (q.d_maxdd, q.d_maxdd_lo, q.d_maxdd_hi)) == v
    for (b, w), v in exp4.items() for q in [PP.loc[('bayes_upper90', b, w)]]))
pp = pd.concat([pd.read_csv(QT / f'alloc_pooled_pairs_{s}.csv') for s in ('ext', 'test')])
sel = pp[(pp.a == 'bayes_upper90') & pp.b.isin(['mle', 'mle_const_rt', 'laplace_upper90'])]
claim(7, 'all 12 sleeve Sharpe intervals contain zero', len(sel) == 12 and ((sel.d_sharpe_lo < 0) & (sel.d_sharpe_hi > 0)).all())

# reproducibility statements
def nb_checks(name):
    nb = json.loads((ROOT / 'notebooks' / name).read_text())
    t = [l for c in nb['cells'] for o in c.get('outputs', []) for l in ''.join(o.get('text', '')).split('\n') if 'checks passed' in l]
    return t[-1] if t else ''
claim('repro', '45 and 130 notebook checks passed (executed notebooks)', '45 / 45 checks passed' in nb_checks('quick_walkthrough.ipynb')
      and '130 / 130 checks passed' in nb_checks('walkthrough.ipynb'))
out = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'], cwd=COURSE, capture_output=True, text=True).stdout
claim('repro', '54 unit tests', '54 tests collected' in out)
fr = json.loads((ROOT / 'artifacts/results/full_run/full_run.json').read_text())
claim('repro', 'full rebuild reproduced every table column (development machine)', fr['table_columns_identical'] == fr['table_columns_compared'] == 282)

bad = [c for c in checks if not c[2]]
for s_, t_, ok_ in checks:
    print(('PASS ' if ok_ else 'FAIL ') + f'[{s_}] {t_}')
print(f'\n{len(checks) - len(bad)}/{len(checks)} claims verified')
sys.exit(1 if bad else 0)
