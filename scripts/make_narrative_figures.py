"""Figures and numbers for the question-driven narrative (H1 -> H2 -> H3), shared by the slides, report and notebooks.

Reads ONLY verified, saved course-project artifacts (no fitting, no new experiments):
  - saved walk-forward forecasts and posterior draws (via bgarch_experiments, checksum-verified)
  - artifacts/results/quick/tables/*.csv (identical to the research tables, verified 282/282)
Writes, next to this file:
  qlike_contrasts.{pdf,png}       H1: pooled QLIKE contrasts with 95% Newey-West intervals
  predictive_densities.{pdf,png}  H2: Gaussian MLE vs Bayesian mixture vs Student-t for one illustrative day
  exposure_control.{pdf,png}      H3: mean exposure vs maximum drawdown, MLE / upper-90 / exposure-matched MLE
  narrative_numbers.json          every number quoted from these figures
The two diagrams (portfolio_flowchart, rolling_design) are TikZ sources here; ``--diagrams`` recompiles them
(needs pdflatex and Ghostscript).

    python scripts/make_narrative_figures.py [--diagrams]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, t as student_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'artifacts' / 'figures' / 'narrative'      # output folder (also holds the TikZ sources)
sys.path.insert(0, str(ROOT / 'src'))
from bgarch import forecast as fc, student_t as tg  # noqa: E402
from bgarch_experiments import adequacy as A  # noqa: E402
from bgarch_experiments.artifacts import verify_manifest  # noqa: E402

QT = ROOT / 'artifacts' / 'results' / 'quick' / 'tables'
C = ['#2a78d6', '#eb6834', '#1baf7a']
GREY, LIGHT, INK, MUTED = '#6b6a63', '#c9c7bd', '#1f1f1e', '#6b6a63'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3dc', 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'pdf.fonttype': 42})
CELLS = [('ext', 500, 'o', '2017–22, 500d'), ('ext', 1000, 's', '2017–22, 1000d'),
         ('test', 500, 'o', '2023–26*, 500d'), ('test', 1000, 's', '2023–26*, 1000d')]
N = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / f'{name}.pdf', bbox_inches='tight', pad_inches=.03)
    fig.savefig(HERE / f'{name}.png', bbox_inches='tight', pad_inches=.03, dpi=200)
    plt.close(fig)


def forest(ax, P, rows, scale=1000):
    for j, (sg, w, mk, lab) in enumerate(CELLS):
        y = np.arange(len(rows)) + (j - 1.5) * .17
        mu = np.array([P[sg].loc[(k, w), 'pooled_mean'] for k, _ in rows])
        se = mu / np.array([P[sg].loc[(k, w), 'z'] for k, _ in rows])
        col = C[0] if sg == 'ext' else GREY
        ax.errorbar(mu * scale, y, xerr=1.96 * se * scale, fmt=mk, ms=5.5, color=col,
                    mfc=col if sg == 'ext' else 'white', lw=1.4, label=lab)
    ax.axvline(0, color=MUTED, lw=.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows])
    ax.invert_yaxis()


def diagrams():
    for name in ('portfolio_flowchart', 'rolling_design'):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'{name}.tex'], cwd=HERE,
                       check=True, capture_output=True)
        subprocess.run(['gs', '-q', '-sDEVICE=png16m', '-r200', '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4',
                        '-o', f'{name}.png', f'{name}.pdf'], cwd=HERE, check=True)
        for ext in ('aux', 'log'):
            (HERE / f'{name}.{ext}').unlink(missing_ok=True)


verify_manifest(); A.verify_manifest()
P = {sg: pd.read_csv(QT / f'forecast_pooled_{sg}.csv').set_index(['contrast', 'window']) for sg in ('ext', 'test')}

# ---------------------------------------------------------------- H1: QLIKE contrasts
rows = [('qlike_integrated_minus_mle', 'Bayes integrated − MLE'),
        ('qlike_meanplug_minus_mle', 'Posterior-mean plug-in − MLE\n(shrinkage only)'),
        ('qlike_integrated_minus_meanplug', 'Integrated − mean plug-in\n(integration only)'),
        ('qlike_t_minus_mle', 'Student-t − MLE (reference)')]
fig, (ax, lax) = plt.subplots(2, 1, figsize=(6.4, 2.75), gridspec_kw={'height_ratios': [6.5, 1]})
forest(ax, P, rows); lax.axis('off'); ax.tick_params(axis='y', labelsize=8.5)
ax.set_xlabel(r'mean daily QLIKE difference ($\times10^{-3}$), 95% HAC CI; negative = better')
h, l = ax.get_legend_handles_labels(); lax.legend(h, l, loc='center', ncol=4, fontsize=8.5, handlelength=1.6, columnspacing=1.2)
save(fig, 'qlike_contrasts')
fq = {sg: pd.read_csv(QT / f'forecast_{sg}.csv') for sg in ('ext', 'test')}
for sg in ('ext', 'test'):
    for w in (500, 1000):
        g = fq[sg][fq[sg].window == w]
        N[f'qlike_level_{sg}_{w}'] = {m: float(g[f'qlike_{m}'].mean()) for m in ('mle', 'bayes_meanplug', 'bayes_integrated', 't', 'ewma')}
        N[f'logscore_level_{sg}_{w}'] = {m: float(g[f'logscore_{m}'].mean()) for m in ('mle', 'bayes_mixture', 'laplace_mixture', 't', 'ewma')}
        for k, _ in rows:
            N[f'{k}_{sg}_{w}'] = [float(P[sg].loc[(k, w), 'pooled_mean']), float(P[sg].loc[(k, w), 'p'])]
        bg = P[sg].loc[('ls_mixture_minus_mle', w), 'pooled_mean']
        N[f'ratio_{sg}_{w}'] = {'t_over_bayes': float(P[sg].loc[('ls_t_minus_mle', w), 'pooled_mean'] / bg),
                                'laplace_share': float(P[sg].loc[('ls_laplace_minus_mle', w), 'pooled_mean'] / bg),
                                'ls_mixture_minus_mle': [float(bg), float(P[sg].loc[('ls_mixture_minus_mle', w), 'p'])],
                                'ls_t_minus_mle': [float(P[sg].loc[('ls_t_minus_mle', w), 'pooled_mean']), float(P[sg].loc[('ls_t_minus_mle', w), 'p'])],
                                'ls_t_minus_mixture': [float(P[sg].loc[('ls_t_minus_mixture', w), 'pooled_mean']), float(P[sg].loc[('ls_t_minus_mixture', w), 'p'])]}

# ---------------------------------------------------------------- H2: predictive densities for one illustrative day
DAY = '2018-02-05'
f, D = A.load_panel('SPY', 500, ('ext',)); i = int(np.flatnonzero(f.date == DAY)[0]); row = f.iloc[i]; v = D[:, i]
sd = np.sqrt(row.variance_mle); x = np.linspace(-7, 7, 1401); xs = x * sd
dens = {'Gaussian GARCH, MLE': (norm.pdf(xs, scale=sd), GREY, '--', 1.5),
        f'Bayesian Gaussian GARCH: mixture of {v.size} normals': (np.exp(fc.mixture_logpdf(xs, v[:, None] * np.ones((1, xs.size)))), C[0], '-', 2.2),
        f'Student-t GARCH, MLE (ν = {row.t_nu:.1f})': (np.exp(tg.t_logpdf(xs, np.full(xs.size, row.variance_t), row.t_nu)), C[1], '-', 1.8)}
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for lab, (d, col, ls, lw) in dens.items():
    ax.semilogy(x, d * sd, color=col, ls=ls, lw=lw, label=lab)
zr = row.return_log / sd
ax.axvline(zr, color=INK, lw=1); ax.text(zr + .12, 2e-6, f'realised\n{DAY}\n{row.return_log:+.1%}', fontsize=8, color=INK, va='bottom')
ax.set_ylim(1e-8, 1.5); ax.set_xlim(-7, 7)
ax.set_xlabel('return in units of the MLE predictive sd'); ax.set_ylabel('density (log scale)')
ax.legend(loc='lower center', fontsize=7.6, handlelength=2.2, framealpha=1, frameon=True, edgecolor='none')
ax.set_title(f'SPY, 500-day window: predictive densities for {DAY}', loc='left', fontsize=9.5)
save(fig, 'predictive_densities')
cv = float(v.std() / v.mean())
N['density_day'] = {'date': DAY, 'return': float(row.return_log), 'z_mle': float(zr), 'nu': float(row.t_nu),
                    'vol_mle': float(np.sqrt(252 * row.variance_mle)), 'vol_bayes_int': float(np.sqrt(252 * row.variance_bayes_integrated)),
                    'vol_t': float(np.sqrt(252 * row.variance_t)), 'posterior_cv': cv, 'mixture_excess_kurtosis': float(fc.mixture_excess_kurtosis(v)),
                    'three_cv2': 3 * cv ** 2, 'logscore_mle': float(row.logscore_mle), 'logscore_bayes_mixture': float(row.logscore_bayes_mixture),
                    'logscore_t': float(row.logscore_t),
                    'p_below_4sd': {'gaussian': float(norm.cdf(-4)),
                                    'mixture': float(fc.mixture_cdf(np.array([-4 * sd]), v[:, None])[0]),
                                    't': float(student_t.cdf(-4 * sd / np.sqrt(row.variance_t * (row.t_nu - 2) / row.t_nu), row.t_nu))}}
nu = []
for a in A.ASSETS:
    for sg in ('ext', 'test'):
        for w in (500, 1000):
            ff = A.STORE.daily(a, w, sg)
            nu.append(pd.DataFrame({'asset': a, 'segment': sg, 'window': w, 'nu': ff.drop_duplicates('refit_date').t_nu.to_numpy()}))
nu = pd.concat(nu)
N['nu'] = {'min': float(nu.nu.min()), 'median': float(nu.nu.median()), 'max': float(nu.nu.max()), 'n_refits': int(len(nu)),
           'median_by_asset': {a: float(g.nu.median()) for a, g in nu.groupby('asset')}}
info = pd.concat([pd.read_csv(QT / f'info_{sg}.csv') for sg in ('ext', 'test')])
N['cv_range'] = [float(info.mean_param_cv_variance.min()), float(info.mean_param_cv_variance.max())]
N['implied_kurt_range'] = [float(info.implied_mixture_excess_kurtosis.min()), float(info.implied_mixture_excess_kurtosis.max())]
N['observed_kurt_range'] = [float(info.empirical_excess_kurtosis_std_returns.min()), float(info.empirical_excess_kurtosis_std_returns.max())]

# ---------------------------------------------------------------- H3: exposure vs drawdown
K = [('mle', 'MLE', GREY, 'o', 'white'), ('bayes_upper90', 'Bayes upper-90', C[0], 'o', C[0]),
     ('mle_const_rt', 'Exposure-matched MLE', C[1], 'D', 'none')]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.75), sharey=False)
for ax, sg, title in zip(axs, ('ext', 'test'), ('2017–22 (primary)', '2023–26 (previously inspected)')):
    S = pd.read_csv(QT / f'alloc_summary_{sg}.csv').pivot_table(index=['asset', 'window'], columns='strategy',
                                                                 values=['mean_exposure', 'max_drawdown'])
    e, dd = S['mean_exposure'] * 100, -S['max_drawdown'] * 100
    for idx in e.index:
        ax.plot([e.loc[idx, 'mle'], e.loc[idx, 'bayes_upper90']], [dd.loc[idx, 'mle'], dd.loc[idx, 'bayes_upper90']], color=LIGHT, lw=1, zorder=1)
    for k, lab, col, mk, fcol in K:
        ax.scatter(e[k], dd[k], s=40 if k != 'mle_const_rt' else 46, color=col, marker=mk, facecolor=fcol, edgecolor=col,
                   lw=1.5, zorder=3 if k != 'mle_const_rt' else 4, label=lab)
    ax.set_title(title, loc='left', fontsize=9.5); ax.set_xlabel('mean ETF weight (%)')
    de = e['bayes_upper90'] - e['mle']; dc = e['bayes_upper90'] - e['mle_const_rt']
    d1 = dd['mle'] - dd['bayes_upper90']; d2 = dd['mle_const_rt'] - dd['bayes_upper90']
    N[f'exposure_{sg}'] = {'d_exposure_vs_mle_pp': float(de.mean()), 'd_exposure_vs_control_pp': float(dc.mean()),
                           'd_exposure_vs_control_pp_range': [float(dc.min()), float(dc.max())],
                           'dd_smaller_vs_mle_pp': float(d1.mean()), 'n_smaller_vs_mle': int((d1 > 0).sum()),
                           'dd_smaller_vs_control_pp': float(d2.mean()), 'n_smaller_vs_control': int((d2 > 0).sum())}
    A_ = pd.read_csv(QT / f'alloc_summary_{sg}.csv')
    g = A_[A_.strategy.isin(['mle', 'bayes_integrated', 'bayes_upper90', 'mle_const_rt'])].groupby('strategy')
    N[f'config_means_{sg}'] = {k: {c: float(v_) for c, v_ in r.items()} for k, r in
                               g[['mean_exposure', 'realized_annual_volatility', 'annualized_return', 'sharpe_excess_cash',
                                  'max_drawdown', 'mean_daily_turnover']].mean().iterrows()}
axs[0].set_ylabel('maximum drawdown (%)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.06))
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(HERE / 'exposure_control.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig(HERE / 'exposure_control.png', bbox_inches='tight', pad_inches=.03, dpi=200); plt.close(fig)
for sg in ('ext', 'test'):
    ps = pd.read_csv(QT / f'alloc_pooled_pairs_{sg}.csv')
    sel = ps[(ps.a == 'bayes_upper90') & ps.b.isin(['mle', 'mle_const_rt', 'laplace_upper90'])]
    N[f'sleeve_pairs_{sg}'] = {f'{r.b}_{r.window}': [float(r.d_sharpe), float(r.d_sharpe_lo), float(r.d_sharpe_hi),
                                                    float(r.d_maxdd), float(r.d_maxdd_lo), float(r.d_maxdd_hi)] for r in sel.itertuples()}

# ---------------------------------------------------------------- slide-sized variants (*_slide): same data, short labels
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})


def slide_forest(rows_, name, xlabel, size=(3.5, 2.3)):
    fig, (ax, lax) = plt.subplots(2, 1, figsize=size, gridspec_kw={'height_ratios': [5, 1]})
    forest(ax, P, rows_); lax.axis('off'); ax.set_xlabel(xlabel)
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, loc='center', ncol=2, fontsize=7, handlelength=1.4, columnspacing=1.0, borderaxespad=0)
    save(fig, name)


slide_forest([('qlike_integrated_minus_mle', 'Integrated − MLE'), ('qlike_meanplug_minus_mle', 'Mean plug-in − MLE'),
              ('qlike_integrated_minus_meanplug', 'Integrated − mean pl.'), ('qlike_t_minus_mle', 'Student-t − MLE')],
             'qlike_contrasts_slide', r'QLIKE difference ($\times10^{-3}$); < 0 better')
slide_forest([('ls_mixture_minus_mle', 'Bayes mixture − MLE'), ('ls_laplace_minus_mle', 'Laplace − MLE'),
              ('ls_t_minus_mle', 'Student-t − MLE')],
             'logscore_gains_slide', r'log-score gain ($\times10^{-3}$ nats/day)', size=(3.5, 2.2))
fig, ax = plt.subplots(figsize=(3.5, 2.25))
for (lab, (d, col, ls, lw)), short in zip(dens.items(), ['Gaussian MLE', f'Bayes mixture ({v.size} draws)', f'Student-t MLE (ν = {row.t_nu:.1f})']):
    ax.semilogy(x, d * sd, color=col, ls=ls, lw=lw * .85, label=short)
ax.axvline(zr, color=INK, lw=.9); ax.text(zr + .15, 2e-5, f'realised\n{row.return_log:+.1%}', fontsize=7, color=INK, va='bottom')
ax.set_ylim(1e-8, 1.5); ax.set_xlim(-7, 7); ax.set_xlabel('return / MLE predictive sd'); ax.set_ylabel('density (log)')
ax.legend(loc='lower center', fontsize=6.6, handlelength=1.8, framealpha=1, frameon=True, edgecolor='none', borderpad=.2)
ax.set_title(f'SPY, one illustrative day ({DAY})', loc='left')
save(fig, 'predictive_densities_slide')
fig, ax = plt.subplots(figsize=(3.4, 2.55))
S = pd.read_csv(QT / 'alloc_summary_ext.csv').pivot_table(index=['asset', 'window'], columns='strategy', values=['mean_exposure', 'max_drawdown'])
e, dd = S['mean_exposure'] * 100, -S['max_drawdown'] * 100
for idx in e.index:
    ax.plot([e.loc[idx, 'mle'], e.loc[idx, 'bayes_upper90']], [dd.loc[idx, 'mle'], dd.loc[idx, 'bayes_upper90']], color=LIGHT, lw=1, zorder=1)
for k, lab, col, mk, fcol in K:
    ax.scatter(e[k], dd[k], s=30 if k != 'mle_const_rt' else 36, color=col, marker=mk, facecolor=fcol, edgecolor=col,
               lw=1.3, zorder=3 if k != 'mle_const_rt' else 4, label=lab)
ax.set_xlabel('mean ETF weight (%)'); ax.set_ylabel('max drawdown (%)'); ax.set_title('2017–22, each ETF × window', loc='left')
ax.legend(loc='upper left', fontsize=7, handletextpad=.3, borderaxespad=.2)
save(fig, 'exposure_control_slide')

(HERE / 'narrative_numbers.json').write_text(json.dumps(N, indent=1))
if '--diagrams' in sys.argv:
    if not (shutil.which('pdflatex') and shutil.which('gs')):
        sys.exit('pdflatex and gs are needed to rebuild the diagrams')
    diagrams()
print('written:', sorted(p.name for p in HERE.glob('*.p[dn][fg]')))
