"""Slide-sized figures and a numbers file for slides.tex.

Reads ONLY verified, saved course-project artifacts (no fitting, no new experiments):
  - saved walk-forward forecasts and posterior draws (via bgarch, checksum-verified)
  - artifacts/results/quick/tables/*.csv  (== research tables, verified 282/282)
  - artifacts/results/adequacy/*.csv
Writes presentation/figures/*.pdf and presentation/figures/numbers.json.

    python scripts/make_presentation_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bgarch import data, portfolio  # noqa: E402,F401
from bgarch_experiments import adequacy as A, tables, strategies  # noqa: E402
from bgarch_experiments.artifacts import verify_manifest  # noqa: E402
from bgarch_experiments.config import SAVED_FORECASTS  # noqa: E402

FIG = ROOT / 'course' / 'presentations' / 'source' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
QT = ROOT / 'artifacts' / 'results' / 'quick' / 'tables'
AD = ROOT / 'artifacts' / 'results' / 'adequacy'
C = ['#2a78d6', '#eb6834', '#1baf7a']           # validated palette slots 1-3
GREY, LIGHT, INK, MUTED = '#6b6a63', '#c9c7bd', '#1f1f1e', '#6b6a63'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3dc', 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'pdf.fonttype': 42})
N = {}                                           # numbers quoted on slides


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.03); plt.close(fig)


def with_legend_row(figsize, ratio):
    """Axes plus a separate legend row below it, so the legend never collides with axis labels."""
    fig, (ax, lax) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [ratio, 1]})
    lax.axis('off')
    return fig, ax, lax


def legend_row(ax, lax, ncol=2):
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, loc='center', ncol=ncol, fontsize=8.5, handlelength=1.6, columnspacing=1.2)


def date_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


verify_manifest(); A.verify_manifest()

# ---------------------------------------------------------------- 1. volatility clustering
r = data.log_returns('SPY')
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.25), gridspec_kw={'width_ratios': [1.6, 1]})
ax[0].plot(r.index, r, lw=.4, color=INK); ax[0].set_ylabel('SPY daily log return'); date_axis(ax[0])
ax[0].set_title('Calm and turbulent periods cluster', loc='left')
lags = np.arange(1, 21)
a1, a2 = A.acf(r.to_numpy(), 20), A.acf(r.to_numpy() ** 2, 20)
ax[1].axhspan(-1.96 / np.sqrt(len(r)), 1.96 / np.sqrt(len(r)), color=LIGHT, alpha=.5, lw=0)
ax[1].bar(lags - .2, a1, width=.4, color=GREY, label=r'returns $r_t$')
ax[1].bar(lags + .2, a2, width=.4, color=C[0], label=r'squared $r_t^2$')
ax[1].set_xlabel('lag (trading days)'); ax[1].set_title('Autocorrelation', loc='left'); ax[1].legend(loc='upper right')
save(fig, 'clustering')
N['acf1_r2_spy_full'] = float(a2[0]); N['acf1_r_spy_full'] = float(a1[0])

# ---------------------------------------------------------------- 2. posterior -> next-day variance
f, D = A.load_panel('SPY', 500, ('test',))
v = D[:, 0]; row = f.iloc[0]
fig, ax, lax = with_legend_row((3.2, 3.0), 3.3)
ax.hist(np.sqrt(252 * v), bins=30, color=C[0], alpha=.45)
for val, lab, col, ls in [(row.variance_mle, 'MLE plug-in', GREY, ':'),
                          (row.variance_bayes_meanplug, 'mean plug-in', C[2], '--'),
                          (row.variance_bayes_integrated, 'integrated', INK, '-'),
                          (row.variance_bayes_q90, '90th pct', C[1], '-')]:
    ax.axvline(np.sqrt(252 * val), color=col, lw=2, ls=ls, label=f'{lab}: {np.sqrt(252*val):.1%}')
ax.set_xlabel(f'predicted SPY vol. for {row.date.date()} (ann.)'); ax.set_ylabel('posterior draws')
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.set_title(f'{D.shape[0]} posterior draws: uncertainty about\nthe forecast, not about the future return', loc='left')
legend_row(ax, lax)
save(fig, 'posterior_nextday')
N['post_vol_mle'] = float(np.sqrt(252 * row.variance_mle)); N['post_vol_int'] = float(np.sqrt(252 * row.variance_bayes_integrated))
N['post_vol_q90'] = float(np.sqrt(252 * row.variance_bayes_q90))
N['post_vol_mp'] = float(np.sqrt(252 * row.variance_bayes_meanplug)); N['post_n_draws'] = int(D.shape[0])

# ---------------------------------------------------------------- 3. SPY March 2020: returns vs bands
f, _ = A.load_panel('SPY', 500, ('ext',))
m = ((f.date >= '2020-02-03') & (f.date <= '2020-05-29')).to_numpy()
d = f.date[m]; rr = f.return_log.to_numpy()[m]; sd = np.sqrt(f.variance_mle.to_numpy()[m]); z = rr / sd
fig, ax = plt.subplots(figsize=(6.2, 2.45))
ax.fill_between(d, -2 * sd, 2 * sd, step='mid', color=C[0], alpha=.14, lw=0, label=r'$\pm2\sigma$ predicted')
ax.fill_between(d, -sd, sd, step='mid', color=C[0], alpha=.25, lw=0, label=r'$\pm1\sigma$ predicted')
ax.plot(d, rr, 'o-', ms=2.5, lw=.8, color=INK, label='realised return')
big = np.abs(z) > 3
ax.scatter(d[big], rr[big], s=55, facecolor='none', edgecolor=C[1], lw=1.8, zorder=4, label=r'$|z|>3$')
ax.set_ylabel('daily log return'); date_axis(ax); ax.legend(loc='lower left', ncol=4, fontsize=8)
ax.set_title('SPY, Feb–May 2020: one-step-ahead Gaussian GARCH (MLE) bands', loc='left')
save(fig, 'spy2020_bands')
fa, _ = A.load_panel('SPY', 500)
za = A.std_resid(fa, 'mle')
N['spy2020_absz3'] = int(big.sum()); N['spy_full_absz3'] = int((np.abs(za) > 3).sum()); N['spy_full_days'] = len(za)
N['gauss_expected_absz3_full'] = float(2 * norm.sf(3) * len(za))
N['spy_full_inside2'] = float(np.mean(np.abs(za) < 2))

# ---------------------------------------------------------------- 4. forecast comparison SPY 2020
fig, ax = plt.subplots(figsize=(6.2, 2.45))
ax.scatter(d, np.abs(rr) * np.sqrt(252), s=12, color=LIGHT, label=r'$|r_t|\sqrt{252}$: noisy one-day proxy', zorder=1)
ax.plot(d, A.ann_vol(f, 'mle')[m], color=GREY, lw=1.3, ls='--', label='Gaussian MLE')
for k, mod in enumerate(['bayes_integrated', 't', 'ewma']):
    ax.plot(d, A.ann_vol(f, mod)[m], color=C[k], lw=1.8, label=A.LABEL[mod])
ax.set_ylabel('annualised vol'); date_axis(ax); ax.set_ylim(0, None); ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1, fontsize=8)
ax.set_title('SPY, Feb–May 2020: next-day volatility forecasts (lines) vs. annualised |return| (dots)', loc='left')
save(fig, 'spy2020_models')
st = pd.read_csv(AD / 'stress_w500.csv'); cv = st[st.period == 'covid_2020']
g = cv[cv.model.isin(['mle', 'map', 'bayes_meanplug', 'bayes_integrated', 'bayes_q90', 'laplace_integrated', 't'])]
N['covid_halflife_garch_min'] = float(g.halflife_after_peak.min()); N['covid_halflife_garch_max'] = float(g.halflife_after_peak.max())
e = cv[cv.model == 'ewma']; N['covid_halflife_ewma_min'] = float(e.halflife_after_peak.min()); N['covid_halflife_ewma_max'] = float(e.halflife_after_peak.max())
s = cv[(cv.asset == 'SPY')].set_index('model')
N['spy_covid_jump_mle'] = float(s.loc['mle', 'jump_ratio']); N['spy_covid_peak_mle'] = float(s.loc['mle', 'peak_vol'])
N['spy_covid_peak_ewma'] = float(s.loc['ewma', 'peak_vol']); N['spy_covid_days_to_peak_ewma'] = int(s.loc['ewma', 'days_shock_to_peak'])
N['spy_covid_halflife_mle'] = float(s.loc['mle', 'halflife_after_peak']); N['spy_covid_halflife_ewma'] = float(s.loc['ewma', 'halflife_after_peak'])

# ---------------------------------------------------------------- 5. log-score gains
labels = [('ls_mixture_minus_meanplug', 'Bayes mixture − mean plug-in\n(integration only)'),
          ('ls_mixture_minus_mle', 'Bayes mixture − MLE'),
          ('ls_laplace_minus_mle', 'Classical Laplace − MLE'),
          ('ls_t_minus_mle', 'Student-t GARCH − MLE')]
fig, ax, lax = with_legend_row((5.2, 2.1), 6.5)
P = {sg: pd.read_csv(QT / f'forecast_pooled_{sg}.csv').set_index(['contrast', 'window']) for sg in ('ext', 'test')}
for j, (sg, w, mk, lab) in enumerate([('ext', 500, 'o', '2017–22, 500d'), ('ext', 1000, 's', '2017–22, 1000d'),
                                       ('test', 500, 'o', '2023–26*, 500d'), ('test', 1000, 's', '2023–26*, 1000d')]):
    y = np.arange(len(labels)) + (j - 1.5) * .17
    mu = np.array([P[sg].loc[(k, w), 'pooled_mean'] for k, _ in labels]); se = mu / np.array([P[sg].loc[(k, w), 'z'] for k, _ in labels])
    col = C[0] if sg == 'ext' else GREY
    ax.errorbar(mu * 1000, y, xerr=1.96 * se * 1000, fmt=mk, ms=5.5, color=col, mfc=col if sg == 'ext' else 'white', lw=1.4, label=lab)
ax.axvline(0, color=MUTED, lw=.8); ax.set_yticks(range(len(labels))); ax.set_yticklabels([l for _, l in labels]); ax.invert_yaxis()
ax.set_xlabel(r'mean daily log-score gain ($\times10^{-3}$ nats), 95% HAC CI')
legend_row(ax, lax, ncol=4)
save(fig, 'logscore_gains')
for sg in ('ext', 'test'):
    for w in (500, 1000):
        for k, _ in labels + [('qlike_integrated_minus_mle', ''), ('qlike_meanplug_minus_mle', ''), ('qlike_integrated_minus_meanplug', ''), ('ls_t_minus_mixture', '')]:
            N[f'{k}_{sg}_{w}'] = float(P[sg].loc[(k, w), 'pooled_mean']); N[f'{k}_{sg}_{w}_p'] = float(P[sg].loc[(k, w), 'p'])

# ---------------------------------------------------------------- 6. kurtosis
info = pd.read_csv(QT / 'info_ext.csv'); i5 = info[info.window == 500]
fig, ax = plt.subplots(figsize=(3.9, 2.9))
x = np.arange(4)
ax.bar(x - .2, i5.empirical_excess_kurtosis_std_returns, .38, color=C[0], label='observed (standardised returns)')
ax.bar(x + .2, i5.implied_mixture_excess_kurtosis, .38, color=C[1], label='implied by parameter uncertainty')
for xi, val in zip(x, i5.implied_mixture_excess_kurtosis):
    ax.text(xi + .2, val + .06, f'{val:.2f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels(i5.asset); ax.set_ylabel('excess kurtosis'); ax.legend(loc='upper right', fontsize=8)
ax.set_title('2017–22, 500-day window', loc='left')
save(fig, 'kurtosis')
allinfo = pd.concat([pd.read_csv(QT / f'info_{sg}.csv') for sg in ('ext', 'test')])
N['cv_min'], N['cv_max'] = float(allinfo.mean_param_cv_variance.min()), float(allinfo.mean_param_cv_variance.max())
N['impl_kurt_min'], N['impl_kurt_max'] = float(allinfo.implied_mixture_excess_kurtosis.min()), float(allinfo.implied_mixture_excess_kurtosis.max())
N['obs_kurt_min'], N['obs_kurt_max'] = float(allinfo.empirical_excess_kurtosis_std_returns.min()), float(allinfo.empirical_excess_kurtosis_std_returns.max())
N['prem_mean_min'], N['prem_mean_max'] = float(allinfo.premium_mean.min()), float(allinfo.premium_mean.max())
N['corr_q90_int_min'] = float(allinfo.corr_q90_integrated.min())

# ---------------------------------------------------------------- 7. weight paths
st_ = tables.Store(SAVED_FORECASTS); f = st_.daily('SPY', 500, 'ext'); W = strategies.strategy_weights(f, 'SPY', 'ext')
m = ((f.date >= '2020-01-02') & (f.date <= '2020-07-31')).to_numpy()
fig, ax, lax = with_legend_row((3.6, 2.9), 4.5)
ax.plot(f.date[m], W['mle'][m], color=GREY, lw=1.4, ls='--', label='MLE')
ax.plot(f.date[m], W['bayes_upper90'][m], color=C[0], lw=2.2, label='Bayes upper-90')
ax.plot(f.date[m], W['mle_const_rt'][m], color=C[1], lw=1.3, label='control (MLE × haircut)')
ax.set_ylabel('equity weight $w_t$'); ax.set_ylim(0, 1.05); date_axis(ax)
legend_row(ax, lax)
ax.set_title('SPY, 500-day window, 2020', loc='left')
save(fig, 'weights_spy2020')

# ---------------------------------------------------------------- 8. Sharpe differences
comps = [('mle', 'MLE'), ('mle_const_rt', 'Matched\ncontrol'), ('laplace_upper90', 'Laplace\nupper-90')]
PP = {sg: pd.read_csv(QT / f'alloc_pooled_pairs_{sg}.csv').set_index(['a', 'b', 'window']) for sg in ('ext', 'test')}
fig, ax, lax = with_legend_row((3.5, 2.6), 3.4)
for j, (sg, w, mk, lab) in enumerate([('ext', 500, 'o', '2017–22, 500d'), ('ext', 1000, 's', '2017–22, 1000d'),
                                       ('test', 500, 'o', '2023–26*, 500d'), ('test', 1000, 's', '2023–26*, 1000d')]):
    y = np.arange(len(comps)) + (j - 1.5) * .17
    rows = [PP[sg].loc[('bayes_upper90', b, w)] for b, _ in comps]
    mu = np.array([q.d_sharpe for q in rows]); lo = np.array([q.d_sharpe_lo for q in rows]); hi = np.array([q.d_sharpe_hi for q in rows])
    col = C[0] if sg == 'ext' else GREY
    ax.errorbar(mu, y, xerr=[mu - lo, hi - mu], fmt=mk, ms=5.5, color=col, mfc=col if sg == 'ext' else 'white', lw=1.4, label=lab)
    for (b, _), q in zip(comps, rows):
        N[f'dsharpe_{b}_{sg}_{w}'] = [float(q.d_sharpe), float(q.d_sharpe_lo), float(q.d_sharpe_hi)]
ax.axvline(0, color=MUTED, lw=.8); ax.set_yticks(range(len(comps))); ax.set_yticklabels([l for _, l in comps]); ax.invert_yaxis()
ax.set_xlabel('ΔSharpe, upper-90 minus comparator\n(4-ETF sleeve, 95% block-bootstrap CI)')
legend_row(ax, lax)
save(fig, 'sharpe_diffs')

# ---------------------------------------------------------------- numbers for tables on slides
for sg in ('ext', 'test'):
    ps = pd.read_csv(QT / f'alloc_pooled_summary_{sg}.csv').set_index(['strategy', 'window'])
    for k in ('mle', 'bayes_integrated', 'bayes_upper90', 'mle_const_rt', 't_garch', 'ewma', 'static_target', 'buy_hold'):
        for w in (500, 1000):
            N[f'sleeve_{k}_{sg}_{w}'] = {'sharpe': float(ps.loc[(k, w), 'sharpe']), 'max_dd': float(ps.loc[(k, w), 'max_dd'])}
    S = pd.read_csv(QT / f'alloc_summary_{sg}.csv').pivot_table(index=['asset', 'window'], columns='strategy',
                                                                values=['mean_exposure', 'max_drawdown', 'sharpe_excess_cash'])
    de = S['mean_exposure']['bayes_upper90'] - S['mean_exposure']['mle']
    d1 = S['max_drawdown']['bayes_upper90'] - S['max_drawdown']['mle']
    d2 = S['max_drawdown']['bayes_upper90'] - S['max_drawdown']['mle_const_rt']
    ds = S['sharpe_excess_cash']['bayes_upper90'] - S['sharpe_excess_cash']['mle_const_rt']
    N[f'exposure_{sg}'] = dict(d_exposure=float(de.mean()), d_dd_mle=float(d1.mean()), d_dd_mle_pos=int((d1 > 0).sum()),
                               d_dd_ctrl=float(d2.mean()), d_dd_ctrl_pos=int((d2 > 0).sum()), d_sh_ctrl=float(ds.mean()),
                               d_sh_ctrl_pos=int((ds > 0).sum()))
    tail = pd.read_csv(QT / f'tail_{sg}.csv')
    for mdl in ('mle', 'bayes_mixture', 't'):
        for al in (.01, .05):
            x = tail[(tail.method == mdl) & (tail.alpha == al)]
            N[f'var_{mdl}_{sg}_{al}'] = [float(x.rate.mean()), int((x.kupiec_p < .05).sum())]
fs = pd.read_csv(QT / 'fit_summary.csv')
N['refits'] = int(fs.refits.sum()); N['failed'] = int(fs.failed_convergence.sum()); N['at_bound'] = int(fs.mle_at_bound.sum())
N['max_rhat'] = float(fs.max_rhat.max()); N['min_tail_ess'] = float(fs.min_tail_ess.min())
dg = pd.read_csv(AD / 'diagnostics_w500_ext.csv').set_index(['asset', 'model'])
for a in data.ASSETS:
    for mdl in ('raw returns (no model)', 'mle', 'bayes_mixture', 't'):
        x = dg.loc[(a, mdl)]
        N[f'diag_{a}_{mdl}'] = {k: (None if pd.isna(x[k]) else float(x[k])) for k in
                                ('acf1_z2', 'lb10_z2_p', 'skew_z', 'exkurt_z', 'exkurt_normal_scores', 'lower01', 'upper01', 'ks_pit_p')}
sp = pd.read_csv(ROOT / 'agent_research/results/primary/seed_prior_sensitivity.csv')
rng = sp[(sp.prior == 'moderate') & (sp.segment == 'test')].groupby(['asset', 'window']).sharpe_upper90.agg(lambda q: q.max() - q.min())
N['seed_range_sharpe_max'] = float(rng.max())
(FIG / 'numbers.json').write_text(json.dumps(N, indent=1))
print('figures:', sorted(p.name for p in FIG.glob('*.pdf')))
