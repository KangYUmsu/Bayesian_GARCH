"""Check every number on the 13 slides and in the narration against the validated course-project results.

    python scripts/verify_presentation_numbers.py

Sources: results/quick/tables (identical to the research tables), course/presentations/source/figures/numbers.json
(presentation/make_figures.py), figures/narrative/narrative_numbers.json (figures/narrative/make_narrative_figures.py),
the saved forecast files and the speaker notes themselves (word counts and timing tables).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bgarch_experiments.config import SAVED_FORECASTS  # noqa: E402
PRES = ROOT / 'course' / 'presentations' / 'source'
COURSE = ROOT
QT = ROOT / 'artifacts' / 'results' / 'quick' / 'tables'
N = json.loads((PRES / 'figures' / 'numbers.json').read_text())
NN = json.loads((ROOT / 'artifacts' / 'figures' / 'narrative' / 'narrative_numbers.json').read_text())
checks = []
claim = lambda slide, text, ok: checks.append((slide, text, bool(ok)))
r = round

fp = {s: pd.read_csv(QT / f'forecast_pooled_{s}.csv').set_index(['contrast', 'window']) for s in ('ext', 'test')}
q = lambda k, w, s='ext': fp[s].loc[(k, w)]
CELLS = [(s, w) for s in ('ext', 'test') for w in (500, 1000)]

# ---------------------------------------------------------------- slide 1
claim(1, 'SPY lag-1 autocorrelation of squared returns 0.42', r(N['acf1_r2_spy_full'], 2) == 0.42)
# ---------------------------------------------------------------- slide 2
claim(2, 'figure: MLE 20.1%, mean plug-in 19.2%, integrated 19.4%, 90th pct 20.4%; 300 draws',
      [r(100 * N[k], 1) for k in ('post_vol_mle', 'post_vol_mp', 'post_vol_int', 'post_vol_q90')] == [20.1, 19.2, 19.4, 20.4] and N['post_n_draws'] == 300)
fs = pd.read_csv(QT / 'fit_summary.csv')
claim(2, 'strict rule (R-hat < 1.01, ESS >= 400) missed by only 4 of 320 fits, all marginal (max R-hat < 1.02)',
      int(fs.failed_convergence.sum()) == 4 and fs.max_rhat.max() < 1.02)
# ---------------------------------------------------------------- slide 3
claim(3, '320 refits', int(fs.refits.sum()) == 320)
fc = SAVED_FORECASTS
d17 = pd.read_csv(fc / 'SPY_500_moderate_s102026_2017_daily.csv.gz', parse_dates=['date', 'refit_date'])
d23 = pd.read_csv(fc / 'SPY_500_moderate_s2026_2023_daily.csv.gz', parse_dates=['date'])
claim(3, '1,510 and 964 evaluation days per ETF', (len(d17), len(d23)) == (1510, 964))
blocks = d17.groupby('refit_date').size()
claim(3, 'parameters refitted every 63 trading days (last block shorter)', (blocks.iloc[:-1] == 63).all() and blocks.iloc[-1] <= 63)
claim(3, 'ext = 2017-01-03..2022-12-30; test starts 2023-01-03',
      (str(d17.date.min().date()), str(d17.date.max().date()), str(d23.date.min().date())) == ('2017-01-03', '2022-12-30', '2023-01-03'))
# ---------------------------------------------------------------- slide 4 (H1)
claim(4, 'QLIKE Bayes-MLE -0.005, -0.006 (p = 0.10, 0.08) in 2017-22',
      [r(q('qlike_integrated_minus_mle', w).pooled_mean, 3) for w in (500, 1000)] == [-0.005, -0.006]
      and [r(q('qlike_integrated_minus_mle', w).p, 2) for w in (500, 1000)] == [0.10, 0.08])
claim(4, 'QLIKE Bayes-MLE not significant at 5% in any period or window', all(q('qlike_integrated_minus_mle', w, s).p > .05 for s, w in CELLS))
claim(4, 'QLIKE levels about -8.3 (2017-22)', all(r(NN[f'qlike_level_ext_{w}'][m], 1) == -8.3 for w in (500, 1000) for m in ('mle', 'bayes_integrated')))
# ---------------------------------------------------------------- slide 5 (H2)
claim(5, 'log score mixture-MLE +0.005 nats/day, p <= 0.024 (2017-22)',
      all(r(q('ls_mixture_minus_mle', w).pooled_mean, 3) == 0.005 and q('ls_mixture_minus_mle', w).p <= 0.0245 for w in (500, 1000)))
claim(5, 'mixture-MLE significant (p <= 0.024) in all four cells', all(q('ls_mixture_minus_mle', w, s).p <= 0.0245 for s, w in CELLS))
sh = [NN[f'ratio_{s}_{w}']['laplace_share'] for s, w in CELLS]
claim(5, 'Laplace recovers 43-89% of the Bayesian gain', (r(100 * min(sh)), r(100 * max(sh))) == (43, 89))
dd_ = NN['density_day']
claim(5, 'illustrative day 2018-02-05: SPY -4.3%, Student-t nu 3.1, 300 draws',
      dd_['date'] == '2018-02-05' and r(100 * dd_['return'], 1) == -4.3 and r(dd_['nu'], 1) == 3.1)
claim(5, 'on that day the log scores rank Student-t > mixture > Gaussian MLE',
      dd_['logscore_t'] > dd_['logscore_bayes_mixture'] > dd_['logscore_mle'])
# ---------------------------------------------------------------- slide 6 (Student-t)
claim(6, 'Student-t median nu about 7 over the 320 refits', r(NN['nu']['median']) == 7 and NN['nu']['n_refits'] == 320)
claim(6, 'Student-t - MLE +0.028 nats/day (2017-22, 500d) and +0.029 (1000d)',
      [r(q('ls_t_minus_mle', w).pooled_mean, 3) for w in (500, 1000)] == [0.028, 0.029])
rt = {(s, w): NN[f'ratio_{s}_{w}']['t_over_bayes'] for s, w in CELLS}
claim(6, '5-6x the Bayesian gain in 2017-22', all(5 <= rt[('ext', w)] < 6.5 for w in (500, 1000)))
claim(6, '3.6-7.5x across all four cells', (r(min(rt.values()), 1), r(max(rt.values()), 1)) == (3.6, 7.5))
claim(6, 'Student-t beats the Bayes mixture in every cell (p <= 0.011)',
      all(q('ls_t_minus_mixture', w, s).pooled_mean > 0 and q('ls_t_minus_mixture', w, s).p <= 0.011 for s, w in CELLS))
claim(6, 'posterior CV 7-12% -> implied excess kurtosis 0.015-0.046; observed 0.4-2.9',
      [r(100 * x) for x in NN['cv_range']] == [7, 12] and [r(x, 3) for x in NN['implied_kurt_range']] == [0.015, 0.046]
      and [r(x, 1) for x in NN['observed_kurt_range']] == [0.4, 2.9])
# ---------------------------------------------------------------- slides 9-10 (H3)
A = pd.read_csv(QT / 'alloc_summary_ext.csv')
g = A.groupby('strategy')[['mean_exposure', 'annualized_return', 'realized_annual_volatility', 'sharpe_excess_cash', 'max_drawdown']].mean()
S = A.pivot_table(index=['asset', 'window'], columns='strategy', values=['mean_exposure', 'max_drawdown', 'sharpe_excess_cash'])
e, dd, shp = S['mean_exposure'], S['max_drawdown'], S['sharpe_excess_cash']
PA = pd.read_csv(QT / 'alloc_pairs_ext.csv')
excl = lambda b, lo, hi: int(((PA[(PA.a == 'bayes_upper90') & (PA.b == b)][lo] > 0) | (PA[(PA.a == 'bayes_upper90') & (PA.b == b)][hi] < 0)).sum())
claim(9, 'Stage 1 table equals reports/source/tables/exposure_stage1.tex (input directly on the slide)',
      (ROOT / 'reports/source/tables/exposure_stage1.tex').exists() and '73.1\\%' in (ROOT / 'reports/source/tables/exposure_stage1.tex').read_text())
claim(9, 'Stage 1: volatility 10.4% -> 9.8%; exposure -3.6 pp; max DD +0.82 pp; smaller DD 8 of 8',
      (r(100 * g.loc['mle', 'realized_annual_volatility'], 1), r(100 * g.loc['bayes_upper90', 'realized_annual_volatility'], 1)) == (10.4, 9.8)
      and r(100 * (e.bayes_upper90 - e.mle).mean(), 1) == -3.6 and r(100 * (dd.bayes_upper90 - dd.mle).mean(), 2) == 0.82
      and int((dd.bayes_upper90 > dd.mle).sum()) == 8)
claim(9, 'Stage 1 CIs: drawdown CI excludes 0 in 8 of 8; Sharpe CI excludes 0 in 0 of 8',
      excl('mle', 'd_maxdd_lo', 'd_maxdd_hi') == 8 and excl('mle', 'd_sharpe_lo', 'd_sharpe_hi') == 0)
PS = pd.read_csv(QT / 'alloc_pooled_pairs_ext.csv').set_index(['a', 'b', 'window'])
sci = lambda b, w: tuple(r(x, 3) for x in PS.loc[('bayes_upper90', b, w), ['d_sharpe', 'd_sharpe_lo', 'd_sharpe_hi']])
claim(9, 'Stage 1 sleeve dSharpe +0.011 [-0.021, 0.042], +0.000 [-0.025, 0.026]', (sci('mle', 500), sci('mle', 1000)) == ((0.011, -0.021, 0.042), (0.0, -0.025, 0.026)))
claim(10, 'control matches exposure within +0.1 pp (2017-22)', r(100 * (e.bayes_upper90 - e.mle_const_rt).mean(), 1) == 0.1)
claim(11, 'Stage 2 CIs: no Sharpe or drawdown CI excludes 0 (8 configurations)',
      excl('mle_const_rt', 'd_maxdd_lo', 'd_maxdd_hi') == 0 and excl('mle_const_rt', 'd_sharpe_lo', 'd_sharpe_hi') == 0)
claim(11, 'Stage 2 sleeve dSharpe +0.009 [-0.016, 0.036], +0.005 [-0.016, 0.027]', (sci('mle_const_rt', 500), sci('mle_const_rt', 1000)) == ((0.009, -0.016, 0.036), (0.005, -0.016, 0.027)))
claim(11, 'Stage 2: delta max DD -0.07 pp (2 of 8 smaller), delta Sharpe +0.003 (mean of 8)',
      r(100 * (dd.bayes_upper90 - dd.mle_const_rt).mean(), 2) == -0.07 and int((dd.bayes_upper90 > dd.mle_const_rt).sum()) == 2
      and r((shp.bayes_upper90 - shp.mle_const_rt).mean(), 3) == 0.003)
pp = pd.concat([pd.read_csv(QT / f'alloc_pooled_pairs_{s}.csv') for s in ('ext', 'test')])
sel = pp[(pp.a == 'bayes_upper90') & pp.b.isin(['mle', 'mle_const_rt', 'laplace_upper90'])]
claim(11, 'every sleeve Sharpe-difference interval contains zero (12 intervals)', len(sel) == 12 and ((sel.d_sharpe_lo < 0) & (sel.d_sharpe_hi > 0)).all())
# ---------------------------------------------------------------- slide 12
claim(12, 'MLE persistence bound hit in 30 of 320 refits', int(fs.mle_at_bound.sum()) == 30)

# ---------------------------------------------------------------- narration: timing tables are consistent with the text
notes = (PRES / 'speaker_notes.md').read_text()
wc = lambda b: len(re.findall(r"[A-Za-z0-9][\w'’\-\.]*", b))
full = notes.split('## Timing of the full narration')[0]
secs = re.split(r'^### Slide (\d+):.*$', full, flags=re.M)
counts = [wc(secs[i + 1].replace('---', '').replace('*Transition:*', '')) for i in range(1, len(secs), 2)]
table = [int(m) for m in re.findall(r'^\| \d+\. [^|]+\| (\d+) \|', notes, flags=re.M)]
claim('notes', f'full-narration word counts match the table ({sum(counts)} words)', counts == table and len(counts) == 13)
cut = notes.split('## Five-minute cut (course video)')[1]
cc = [wc(b) for _, b in re.findall(r'\*\*Slide (\d+)\.\*\*(.*?)(?=\n\n)', cut, flags=re.S)]
ctab = [int(m) for m in re.findall(r'^\| \d+ \| (\d+) \|', cut, flags=re.M)]
claim('notes', f'five-minute cut: {sum(cc)} words = {sum(cc) / 145:.2f} min at 145 wpm, under 5 min even at 125 wpm',
      cc == ctab and sum(cc) / 125 < 5)

bad = [c for c in checks if not c[2]]
for s_, t_, ok in checks:
    print(('PASS ' if ok else 'FAIL ') + f'[{s_}] {t_}')
print(f'\n{len(checks) - len(bad)}/{len(checks)} claims verified')
sys.exit(1 if bad else 0)
