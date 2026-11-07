"""Figures for the primary investigation (reads agent_research/results/primary)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, FIG, ASSETS, WINDOWS, load_daily, vol_weight  # noqa: E402
from analysis_primary import strategy_weights  # noqa: E402

P = RES / 'primary'; FIG.mkdir(parents=True, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # fixed categorical order
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True})

# 1. density-forecast gains vs MLE: parameter uncertainty vs fat tails
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
labels = {'ls_mixture_minus_meanplug': 'Bayes mixture − Bayes mean plug-in\n(pure parameter integration)',
          'ls_mixture_minus_mle': 'Bayes mixture − MLE', 'ls_laplace_minus_mle': 'Classical Laplace mixture − MLE',
          'ls_t_minus_mle': 'Student-t GARCH MLE − MLE'}
for ax, seg, title in zip(axes, ['ext', 'test'], ['2017–2022 (new out-of-sample)', '2023–2026 (project test period)']):
    d = pd.read_csv(P / f'forecast_pooled_{seg}.csv')
    for j, w in enumerate(WINDOWS):
        sub = d[(d.window == w) & d.contrast.isin(labels)].set_index('contrast').loc[list(labels)]
        se = sub.pooled_mean / sub.z
        y = np.arange(len(labels)) + (j - .5) * .3
        ax.errorbar(sub.pooled_mean, y, xerr=1.96 * se, fmt='o', ms=5, color=C[j], lw=1.5, capsize=0,
                    label=f'{w}-day window')
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(list(labels.values()) if seg == 'ext' else [])
    ax.set_title(title, fontsize=9, color=INK, loc='left'); ax.set_xlabel('Mean daily log-score gain vs comparator (nats)')
    ax.invert_yaxis()
axes[1].legend(frameon=False, loc='upper right')
fig.suptitle('4-asset pooled, 95% Newey–West CI', x=.99, ha='right', fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(FIG / 'fig1_logscore_gains.png', dpi=160); plt.close(fig)

# 2. Sharpe differences of bayes_upper90 vs comparators (pooled sleeve, block bootstrap)
names = {'mle': 'MLE (unmatched)', 'mle_const_rt': 'MLE × real-time constant haircut',
         'meanplug_const_rt': 'Bayes mean plug-in × real-time haircut', 'laplace_upper90': 'Classical Laplace upper-90',
         'mle_exposure_matched_expost': 'MLE exposure-matched (ex post)'}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
for ax, seg, title in zip(axes, ['ext', 'test'], ['2017–2022', '2023–2026']):
    d = pd.read_csv(P / f'alloc_pooled_pairs_{seg}.csv')
    d = d[(d.a == 'bayes_upper90') & d.b.isin(names)]
    for j, w in enumerate(WINDOWS):
        sub = d[d.window == w].set_index('b').loc[list(names)]
        y = np.arange(len(names)) + (j - .5) * .3
        ax.errorbar(sub.d_sharpe, y, xerr=[sub.d_sharpe - sub.d_sharpe_lo, sub.d_sharpe_hi - sub.d_sharpe], fmt='o',
                    ms=5, color=C[j], lw=1.5, capsize=0, label=f'{w}-day window')
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(list(names.values()) if seg == 'ext' else [])
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.set_xlabel('ΔSharpe, 4-asset sleeve, 95% block-bootstrap CI'); ax.invert_yaxis()
axes[1].legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 1.08), ncol=2)
fig.suptitle('Bayes upper-90 Sharpe minus comparator', x=.01, ha='left', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(FIG / 'fig2_sharpe_differences.png', dpi=160); plt.close(fig)

# 3. weight paths: daily posterior quantile vs a slowly varying constant haircut (SPY, 500, 2020)
f = load_daily('SPY', 500, 'ext'); W = strategy_weights(f, 'SPY', 'ext')
m = (f.date >= '2019-10-01') & (f.date <= '2020-12-31')
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(f.date[m], W['mle'][m], color=MUTED, lw=1.2, label='MLE plug-in')
ax.plot(f.date[m], W['bayes_upper90'][m], color=C[0], lw=2, label='Bayes upper-90 (daily posterior quantile)')
ax.plot(f.date[m], W['meanplug_const_rt'][m], color=C[1], lw=1.2, ls='--', label='Bayes mean plug-in × trailing constant haircut')
ax.set_ylabel('Equity weight'); ax.set_ylim(0, 1.05)
ax.set_title('SPY, 500-day window: the posterior-quantile rule is almost a constant haircut of the point-estimate rule',
             fontsize=9, color=INK, loc='left')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
fig.tight_layout(); fig.savefig(FIG / 'fig3_weight_paths_spy2020.png', dpi=160); plt.close(fig)

# 4. fat tails: what parameter uncertainty can explain vs what is observed
fig, ax = plt.subplots(figsize=(9, 3.5))
d = pd.concat([pd.read_csv(P / f'info_{s}.csv').assign(seg=s) for s in ['ext', 'test']])
d = d[d.window == 500]
lab = [f'{a}\n{"17–22" if s == "ext" else "23–26"}' for a, s in zip(d.asset, d.seg)]
x = np.arange(len(d))
ax.bar(x - .2, d.empirical_excess_kurtosis_std_returns, width=.38, color=C[0], label='Observed excess kurtosis of standardized returns')
ax.bar(x + .2, d.implied_mixture_excess_kurtosis, width=.38, color=C[1], label='Excess kurtosis implied by posterior parameter uncertainty')
for xi, v in zip(x, d.implied_mixture_excess_kurtosis):
    ax.text(xi + .2, v + .05, f'{v:.2f}', ha='center', fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('Excess kurtosis')
ax.set_title('500-day window: parameter uncertainty explains almost none of the fat tails', fontsize=9, color=INK, loc='left')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
fig.tight_layout(); fig.savefig(FIG / 'fig4_kurtosis.png', dpi=160); plt.close(fig)
print('figures written')
