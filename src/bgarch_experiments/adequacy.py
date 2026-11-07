"""Out-of-sample model-adequacy diagnostics and figures (course extension).

Everything here is computed from the SAVED walk-forward forecasts and posterior draws
(no refitting). Alignment convention of every saved row:

    row date d:  return_log = r_d  (close d-1 -> close d)
                 variance_* = forecast of Var(r_d | r_{<d}), made with parameters fitted on a
                              window that ends strictly before d (fit_end < refit_date <= d)

so z_d = r_d / sqrt(h_d) and every PIT u_d = F_d(r_d) are genuine one-step-ahead,
out-of-sample quantities. They are NOT in-sample goodness-of-fit residuals: the parameters
never saw r_d.

Models (all forecast the same r_d):
    mle, map, bayes_meanplug, bayes_integrated, laplace_integrated, ewma  -> Gaussian N(0, h)
    bayes_q90   -> the posterior 90th percentile of the conditional VARIANCE, used as a
                   Gaussian scale. It is a conservative position-sizing input, not a
                   predictive return quantile (its implied quantiles are not calibrated by design).
    t           -> standardised Student-t with nu from the t-GARCH MLE
    bayes_mixture -> posterior predictive (1/M) sum_m N(0, h^(m)); full CDF from saved draws
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, t as student_t, jarque_bera, kstest  # noqa: E402

from .config import SAVED_FORECASTS, RESULTS, FIGURES, PROJECT, ASSETS, SEG_FILES, resolve_artifact, artifact_key  # noqa: E402,F401
from bgarch.data import log_returns  # noqa: E402
from bgarch.evaluation import kupiec  # noqa: E402
from bgarch import diagnostics as _dg  # noqa: E402
from .tables import Store  # noqa: E402
from .artifacts import sha256  # noqa: E402

MANIFEST = RESULTS / 'adequacy_manifest.json'
STORE = Store(SAVED_FORECASTS)
SEGS = ['ext', 'test']

GAUSS = {'mle': 'Gaussian GARCH MLE', 'map': 'Gaussian GARCH MAP', 'bayes_meanplug': 'Bayes posterior-mean plug-in',
         'bayes_integrated': 'Bayes posterior-integrated variance', 'bayes_q90': 'Bayes upper-90% variance',
         'laplace_integrated': 'Laplace (classical) integrated', 'ewma': 'EWMA (lambda=0.94)'}
LABEL = dict(GAUSS, t='Student-t GARCH MLE', bayes_mixture='Bayes predictive mixture')
VAR_COL = {k: f'variance_{k}' for k in GAUSS} | {'t': 'variance_t', 'bayes_mixture': 'variance_bayes_integrated'}
ALL_MODELS = list(GAUSS) + ['t', 'bayes_mixture']
GROUPS = [('Point estimates', ['mle', 'map', 'bayes_meanplug']),
          ('Parameter uncertainty', ['bayes_integrated', 'bayes_q90', 'laplace_integrated']),
          ('Other models', ['t', 'ewma'])]
PERIODS = {'covid_2020': ('2020-02-03', '2020-05-29', 'COVID crash, Feb-May 2020'),
           'stress_2022': ('2022-04-01', '2022-07-29', 'Rate-hike sell-off, Apr-Jul 2022'),
           'shock_2025': ('2025-03-03', '2025-06-30', 'Equity shock, Mar-Jun 2025 (previously inspected period)')}

C = ['#2a78d6', '#eb6834', '#1baf7a']           # palette slots 1-3 (validated all-pairs)
GREY, LIGHT, INK, MUTED, RED = '#6b6a63', '#c9c7bd', '#1f1f1e', '#6b6a63', '#e34948'
STYLE = {'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
         'grid.color': '#e4e3dc', 'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False}


# ----------------------------------------------------------------------------- data
def _draw_path(asset, window, seg):
    year, seed = SEG_FILES[seg]
    return SAVED_FORECASTS / f'{asset}_{window}_moderate_s{seed}_{year}_draws.npz'


def build_manifest():
    files = [_draw_path(a, w, s) for s in SEGS for a in ASSETS for w in (500, 1000)]
    data = {artifact_key(p): sha256(p) for p in files}
    MANIFEST.write_text(json.dumps(data, indent=1))
    return data


def verify_manifest():
    exp = json.loads(MANIFEST.read_text())
    bad = [k for k, v in exp.items() if not resolve_artifact(k).exists() or sha256(resolve_artifact(k)) != v]
    if bad:
        raise RuntimeError(f'posterior-draw artifacts missing or altered: {bad}')
    return len(exp)


def load_panel(asset, window=500, segments=('ext', 'test')):
    """Concatenated saved forecasts with alignment checks and posterior draws.

    Returns (frame, draws) where draws has shape (300, n_days), column-aligned with frame.
    """
    frames, draws = [], []
    rr = log_returns(asset)
    for seg in segments:
        f = STORE.daily(asset, window, seg)
        fits = STORE.fits(asset, window, seg)
        z = np.load(_draw_path(asset, window, seg), allow_pickle=True)
        if not (pd.to_datetime(z['dates']) == f.date).all():
            raise AssertionError('draws not aligned with forecast rows')
        # (1) the realised return in row d is the close(d-1)->close(d) log return of the price file
        if not np.allclose(f.return_log.to_numpy(), rr.loc[f.date].to_numpy(), rtol=0, atol=1e-15):
            raise AssertionError('return_log does not match the price data at the row date')
        # (2) every forecast row was produced by a fit whose window ended strictly before the row date
        fe = pd.to_datetime(fits.set_index('refit_date').fit_end)
        fit_end = f.refit_date.dt.strftime('%Y-%m-%d').map(fe)
        if not ((fit_end < f.date) & (f.refit_date <= f.date)).all():
            raise AssertionError('a forecast row uses a fit that ends on/after its own date')
        frames.append(f.assign(segment=seg)); draws.append(z['v'].astype(float))
    f = pd.concat(frames, ignore_index=True)
    if not f.date.is_monotonic_increasing or f.date.duplicated().any():
        raise AssertionError('segments overlap or are unordered')
    return f, np.concatenate(draws, axis=1)


realized_vol = _dg.realized_vol
normal_scores = _dg.normal_scores
acf = _dg.acf
_lb = _dg.ljung_box_p


def ann_vol(f, model):
    return np.sqrt(252 * f[VAR_COL[model]].to_numpy())


# ----------------------------------------------------------------------------- PIT / residuals
def std_resid(f, model):
    """z_d = r_d / sqrt(h_d); for the mixture, h_d is its predictive variance (= integrated variance)."""
    return _dg.std_resid(f.return_log.to_numpy(), f[VAR_COL[model]].to_numpy())


def pit(f, model, draws=None):
    """u_d = F_d(r_d) under the model's full one-step predictive CDF."""
    r = f.return_log.to_numpy()
    if model in GAUSS:
        return _dg.pit_gaussian(r, f[VAR_COL[model]].to_numpy())
    if model == 't':
        return _dg.pit_t(r, f.variance_t.to_numpy(), f.t_nu.to_numpy())
    if model == 'bayes_mixture':
        return _dg.pit_mixture(r, draws)
    raise KeyError(model)




# ----------------------------------------------------------------------------- tables
def diagnostics_table(window=500, segment='ext'):
    """Residual and calibration diagnostics per ETF and model (out-of-sample)."""
    rows = []
    for a in ASSETS:
        f, D = load_panel(a, window, (segment,))
        r = f.return_log.to_numpy()
        rows.append(dict(asset=a, model='raw returns (no model)', n=len(r), mean_z2=np.nan,
                         acf1_z2=acf(r ** 2, 1)[0], lb10_z_p=_lb(r, 10), lb10_z2_p=_lb(r ** 2, 10), lb20_z2_p=_lb(r ** 2, 20)))
        for m in ALL_MODELS:
            z = std_resid(f, m); u = pit(f, m, D); ns = normal_scores(u)
            rows.append(dict(asset=a, model=m, n=len(z), mean_z2=float(np.mean(z ** 2)),
                             skew_z=float(pd.Series(z).skew()), exkurt_z=float(pd.Series(z).kurt()),
                             exkurt_normal_scores=float(pd.Series(ns).kurt()), jb_normal_scores_p=float(jarque_bera(ns).pvalue),
                             acf1_z2=float(acf(z ** 2, 1)[0]), lb10_z_p=_lb(z, 10), lb10_z2_p=_lb(z ** 2, 10), lb20_z2_p=_lb(z ** 2, 20),
                             lower01=float(np.mean(u < .01)), lower05=float(np.mean(u < .05)),
                             upper01=float(np.mean(u > .99)), upper05=float(np.mean(u > .95)),
                             kupiec_lower01_p=kupiec(u < .01, .01), kupiec_lower05_p=kupiec(u < .05, .05),
                             kupiec_upper01_p=kupiec(u > .99, .01), ks_pit_p=float(kstest(u, 'uniform').pvalue)))
    return pd.DataFrame(rows)


def performance_table(window=500, segment='ext'):
    """Per-ETF QLIKE, log score and VaR violation rates for all models."""
    rows = []
    for a in ASSETS:
        f, D = load_panel(a, window, (segment,)); r = f.return_log.to_numpy()
        for m in ALL_MODELS:
            h = f[VAR_COL[m]].to_numpy(); u = pit(f, m, D)
            if m == 'bayes_q90':
                ls = np.nan                                  # not a predictive distribution
            elif m == 'bayes_integrated':
                ls = float(norm.logpdf(r, scale=np.sqrt(h)).mean())
            elif m == 'laplace_integrated':
                ls = float(f.logscore_laplace_mixture.mean())  # Laplace predictive mixture
            else:
                ls = float(f[f'logscore_{m}'].mean())
            rows.append(dict(asset=a, window=window, segment=segment, model=m,
                             qlike=float(np.mean(np.log(h) + r ** 2 / h)), log_score=ls,
                             var01_violation=float(np.mean(u < .01)), var05_violation=float(np.mean(u < .05)),
                             mean_z2=float(np.mean(r ** 2 / h))))
    return pd.DataFrame(rows)


def regime_table(window=500):
    """Where is modelling hard? By calendar year and by ex-ante predicted-volatility tercile."""
    rows = []
    for a in ASSETS:
        f, D = load_panel(a, window)
        u = {m: pit(f, m, D) for m in ('mle', 't', 'bayes_mixture')}
        z2 = std_resid(f, 'mle') ** 2
        # ex-ante regime: tercile of the MLE predicted vol within the asset (uses only h_d, known before r_d)
        terc = pd.qcut(f.variance_mle.rank(method='first'), 3, labels=['calm', 'normal', 'turbulent'])
        for key, grp in (('year', f.date.dt.year.astype(str)), ('predicted_vol_tercile', terc.astype(str))):
            for g, idx in f.groupby(grp).groups.items():
                i = np.asarray(idx)
                rows.append(dict(asset=a, split=key, regime=g, n=len(i), mean_z2_mle=float(z2[i].mean()),
                                 max_abs_z_mle=float(np.sqrt(z2[i]).max()),
                                 lower01_mle=float(np.mean(u['mle'][i] < .01)), lower01_t=float(np.mean(u['t'][i] < .01)),
                                 lower01_mixture=float(np.mean(u['bayes_mixture'][i] < .01)),
                                 ls_t_minus_mle=float((f.logscore_t.to_numpy()[i] - f.logscore_mle.to_numpy()[i]).mean()),
                                 ls_mix_minus_mle=float((f.logscore_bayes_mixture.to_numpy()[i] - f.logscore_mle.to_numpy()[i]).mean())))
    return pd.DataFrame(rows)


def stress_table(window=500):
    """Reaction speed, persistence and under-prediction inside each stress window.

    shock day s = day of the largest |r| in the window (ex-post, descriptive).
    jump    = predicted vol for s+1 (first forecast that has seen r_s) / predicted vol for s.
    peak    = max predicted vol within the window; days_to_peak counted from s.
    halflife= trading days after the peak until predicted vol is back halfway to its level at the
              window start (searched up to 250 days beyond; NaN if not reached).
    """
    rows = []
    for a in ASSETS:
        f, _ = load_panel(a, window); r = f.return_log.to_numpy()
        for pk, (lo, hi, _) in PERIODS.items():
            win = np.flatnonzero(((f.date >= lo) & (f.date <= hi)).to_numpy())
            s = win[np.argmax(np.abs(r[win]))]
            for m in ALL_MODELS[:-1]:
                v = ann_vol(f, m); z = std_resid(f, m)
                p = win[np.argmax(v[win])]; base = v[win[0]]
                after = np.flatnonzero(v[p:p + 250] <= base + .5 * (v[p] - base))
                rows.append(dict(period=pk, asset=a, model=m, shock_date=str(f.date[s].date()), shock_return=float(r[s]),
                                 vol_before=float(v[s]), jump_ratio=float(v[s + 1] / v[s]), peak_vol=float(v[p]),
                                 days_shock_to_peak=int(p - s), halflife_after_peak=float(after[0]) if len(after) else np.nan,
                                 max_abs_z=float(np.abs(z[win]).max()), n_abs_z_gt3=int((np.abs(z[win]) > 3).sum()),
                                 mean_z2=float(np.mean(z[win] ** 2))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _date_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9); ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def fig_returns_bands(asset, window=500, model='mle', out=FIGURES):
    plt.rcParams.update(STYLE)
    f, _ = load_panel(asset, window); r = f.return_log.to_numpy(); sd = np.sqrt(f[VAR_COL[model]].to_numpy())
    z = r / sd; big = np.abs(z) > 3
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    ax[0].fill_between(f.date, -2 * sd, 2 * sd, step='mid', color=C[0], alpha=.13, lw=0, label='±2σ predicted (Gaussian MLE)')
    ax[0].fill_between(f.date, -sd, sd, step='mid', color=C[0], alpha=.22, lw=0, label='±1σ predicted')
    ax[0].plot(f.date, r, lw=.45, color=INK, label='realised daily log return')
    ax[0].scatter(f.date[big], r[big], s=14, color=C[1], zorder=3, label=f'|z| > 3  ({big.sum()} days; Gaussian expects {2*norm.sf(3)*len(r):.1f})')
    ax[0].set_ylabel('daily log return'); ax[0].legend(loc='lower left', ncol=2, fontsize=7.5)
    ax[1].plot(f.date, realized_vol(r), color=LIGHT, lw=1.6, label='21-day realised vol (backward-looking, includes day d)')
    ax[1].plot(f.date, np.sqrt(252) * sd, color=C[0], lw=1, label='one-step-ahead predicted vol (MLE)')
    ax[1].set_ylabel('annualised vol'); ax[1].legend(loc='upper left', fontsize=7.5)
    for a_ in ax:
        a_.axvline(pd.Timestamp('2023-01-03'), color=MUTED, lw=.8, ls=':')
    ax[0].text(pd.Timestamp('2023-01-20'), ax[0].get_ylim()[1] * .9, '2023–26: previously inspected', fontsize=7, color=MUTED)
    ax[0].set_title(f'{asset}, {window}-day window: realised returns vs one-step-ahead predicted ±σ bands', loc='left', fontsize=9)
    _date_axis(ax[1]); fig.tight_layout()
    p = Path(out) / f'adequacy_returns_{asset}.png'; fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_model_comparison(asset, start='2017-01-01', end='2022-12-31', window=500, out=FIGURES, tag='2017_2022'):
    """Small multiples: each panel <=3 coloured series; MLE grey reference; right column = ratio to MLE."""
    plt.rcParams.update(STYLE)
    f, _ = load_panel(asset, window); m = ((f.date >= start) & (f.date <= end)).to_numpy()
    d = f.date[m]; rv = realized_vol(f.return_log.to_numpy())[m]; ref = ann_vol(f, 'mle')[m]
    fig, ax = plt.subplots(3, 2, figsize=(11, 7.2), sharex=True, gridspec_kw={'width_ratios': [1.6, 1]})
    for i, (title, models) in enumerate(GROUPS):
        ax[i, 0].plot(d, rv, color=LIGHT, lw=1.6, label='21-day realised vol (descriptive)')
        ax[i, 0].plot(d, ref, color=GREY, lw=.8, ls='--', label='MLE (reference)')
        ax[i, 1].axhline(1, color=GREY, lw=.8, ls='--')
        for k, mod in enumerate(x for x in models if x != 'mle'):
            v = ann_vol(f, mod)[m]
            ax[i, 0].plot(d, v, color=C[k], lw=1, label=LABEL[mod])
            ax[i, 1].plot(d, v / ref, color=C[k], lw=1, label=LABEL[mod])
        ax[i, 0].set_ylabel('annualised vol'); ax[i, 0].set_title(title, loc='left', fontsize=9)
        ax[i, 0].legend(loc='upper left', fontsize=7, ncol=2); ax[i, 1].set_title('ratio to MLE predicted vol', loc='left', fontsize=8.5)
    for a_ in ax[-1]:
        _date_axis(a_)
    fig.suptitle(f'{asset}, {window}-day window: one-step-ahead predicted volatility, all models ({start[:4]}–{end[:4]})',
                 x=.01, ha='left', fontsize=9.5)
    fig.tight_layout()
    p = Path(out) / f'adequacy_models_{asset}_{tag}.png'; fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_stress(period, window=500, out=FIGURES):
    plt.rcParams.update(STYLE)
    lo, hi, title = PERIODS[period]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    for ax, a in zip(axes.flat, ASSETS):
        f, _ = load_panel(a, window); m = ((f.date >= lo) & (f.date <= hi)).to_numpy(); d = f.date[m]
        r = f.return_log.to_numpy()
        ax.scatter(d, np.abs(r[m]) * np.sqrt(252), s=7, color=LIGHT, zorder=1, label='|r_d|·√252 (realised, same units)')
        ax.plot(d, ann_vol(f, 'mle')[m], color=GREY, lw=1, ls='--', label='MLE')
        for k, mod in enumerate(['bayes_q90', 't', 'ewma']):
            ax.plot(d, ann_vol(f, mod)[m], color=C[k], lw=1.2, label=LABEL[mod])
        ax.set_title(a, loc='left', fontsize=9); ax.set_ylabel('annualised vol'); _date_axis(ax)
        ax.set_ylim(0, None)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=7.5)
    fig.suptitle(f'{title}: predicted vol (lines) vs realised absolute returns (dots); {window}-day window',
                 x=.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = Path(out) / f'adequacy_stress_{period}.png'; fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_diagnostics(asset, window=500, segment='ext', out=FIGURES):
    plt.rcParams.update(STYLE)
    f, D = load_panel(asset, window, (segment,)); r = f.return_log.to_numpy(); n = len(r)
    show = [('mle', C[0]), ('bayes_mixture', C[1]), ('t', C[2])]
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.4))
    lags = np.arange(1, 21); band = 1.96 / np.sqrt(n)
    # (a) ACF of squared returns vs squared standardised residuals
    a = ax[0, 0]; a.axhspan(-band, band, color=LIGHT, alpha=.5, lw=0)
    a.plot(lags, acf(r ** 2), 'o-', ms=3, color=GREY, label='r² (raw)')
    a.plot(lags, acf(std_resid(f, 'mle') ** 2), 'o-', ms=3, color=C[0], label='z² (MLE)')
    a.plot(lags, acf(std_resid(f, 't') ** 2), 'o-', ms=3, color=C[2], label='z² (Student-t)')
    a.set_title('ACF of squared returns / residuals', loc='left', fontsize=9); a.set_xlabel('lag (days)'); a.legend(fontsize=7)
    # (b) ACF of z
    a = ax[0, 1]; a.axhspan(-band, band, color=LIGHT, alpha=.5, lw=0)
    a.bar(lags, acf(std_resid(f, 'mle')), color=C[0], width=.6)
    a.set_title('ACF of z (MLE); shaded ±1.96/√n', loc='left', fontsize=9); a.set_xlabel('lag (days)')
    # (c) histogram of z with N(0,1) and standardised t, log scale
    a = ax[0, 2]; z = std_resid(f, 'mle'); x = np.linspace(-7, 7, 400); nu = float(np.median(f.t_nu))
    a.hist(z, bins=np.linspace(-7, 7, 71), density=True, color=C[0], alpha=.55, label='z (MLE), out-of-sample')
    a.plot(x, norm.pdf(x), color=INK, lw=1, label='N(0,1)')
    a.plot(x, student_t.pdf(x * np.sqrt(nu / (nu - 2)), nu) * np.sqrt(nu / (nu - 2)), color=C[2], lw=1.2, label=f'unit-variance t, ν={nu:.1f}')
    a.set_yscale('log'); a.set_ylim(1e-4, 1); a.set_title('histogram of z (log density)', loc='left', fontsize=9); a.legend(fontsize=7)
    # (d) normal Q-Q of Berkowitz normal scores Phi^-1(PIT)
    a = ax[1, 0]; q = norm.ppf((np.arange(1, n + 1) - .5) / n)
    for mod, col in show:
        a.plot(q, np.sort(normal_scores(pit(f, mod, D))), '.', ms=2.5, color=col, label=LABEL[mod])
    a.plot([-4, 4], [-4, 4], color=INK, lw=.8); a.set_xlim(-4, 4); a.set_ylim(-7, 7)
    a.set_xlabel('N(0,1) quantile'); a.set_ylabel('sorted Φ⁻¹(PIT)')
    a.set_title('Q–Q of Φ⁻¹(PIT): full-distribution calibration', loc='left', fontsize=9); a.legend(fontsize=7, loc='upper left')
    # (e) PIT histograms
    a = ax[1, 1]
    for mod, col in show:
        a.hist(pit(f, mod, D), bins=20, range=(0, 1), density=True, histtype='step', lw=1.4, color=col, label=LABEL[mod])
    a.axhline(1, color=INK, lw=.8); a.set_title('PIT histogram (flat = calibrated)', loc='left', fontsize=9); a.set_xlabel('u = F(r)')
    # (f) tail exceedance vs nominal
    a = ax[1, 2]; levels = np.array([.005, .01, .025, .05])
    for k, (mod, col) in enumerate(show):
        u = pit(f, mod, D)
        a.plot(levels * 100, [np.mean(u < l) * 100 for l in levels], 'o-', ms=4, color=col, label=f'{LABEL[mod]} lower')
        a.plot(levels * 100, [np.mean(u > 1 - l) * 100 for l in levels], 's--', ms=3.5, color=col, lw=.8, label=f'{LABEL[mod]} upper')
    a.plot([0, 5.5], [0, 5.5], color=INK, lw=.8); a.set_xlabel('nominal tail probability (%)'); a.set_ylabel('observed exceedance (%)')
    a.set_title('quantile exceedances (lower ●, upper ■)', loc='left', fontsize=9)
    fig.suptitle(f'{asset}: out-of-sample adequacy diagnostics, {window}-day window, 2017–2022 (n={n})', x=.01, ha='left', fontsize=9.5)
    fig.tight_layout()
    p = Path(out) / f'adequacy_diagnostics_{asset}.png'; fig.savefig(p, dpi=150); plt.close(fig); return p


def fig_exceedance_summary(window=500, segment='ext', out=FIGURES):
    """All 9 forecast distributions, 4 ETFs: lower/upper 1% exceedance (identity by row label, not colour)."""
    plt.rcParams.update(STYLE)
    t = diagnostics_table(window, segment); t = t[t.model.isin(ALL_MODELS)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.8), sharey=True)
    y = np.arange(len(ALL_MODELS))
    for ax, a in zip(axes, ASSETS):
        s = t[t.asset == a].set_index('model').loc[ALL_MODELS]
        ax.axvline(1, color=INK, lw=.8)
        ax.plot(s.lower01 * 100, y, 'o', color=C[0], ms=5, label='lower 1% tail (losses)')
        ax.plot(s.upper01 * 100, y, 's', color=C[1], ms=4.5, label='upper 1% tail (gains)')
        ax.set_title(a, loc='left', fontsize=9); ax.set_xlabel('exceedance rate (%)'); ax.set_xlim(0, 3.4)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[m] for m in ALL_MODELS]); axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=7.5)
    fig.suptitle(f'1% predictive-quantile exceedances, {window}-day window, 2017–2022 (nominal = 1%; ±2 binomial s.e. ≈ ±0.5 pp)',
                 x=.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, .07, 1, 1))
    p = Path(out) / 'adequacy_exceedance_summary.png'; fig.savefig(p, dpi=150); plt.close(fig); return p


def make_all(out_tables=RESULTS / 'adequacy', out_figs=FIGURES):
    verify_manifest()
    out_tables = Path(out_tables); out_tables.mkdir(parents=True, exist_ok=True)
    for w in (500, 1000):
        for s in SEGS:
            diagnostics_table(w, s).to_csv(out_tables / f'diagnostics_w{w}_{s}.csv', index=False)
            performance_table(w, s).to_csv(out_tables / f'performance_w{w}_{s}.csv', index=False)
    regime_table().to_csv(out_tables / 'regimes_w500.csv', index=False)
    stress_table().to_csv(out_tables / 'stress_w500.csv', index=False)
    figs = [fig_exceedance_summary()]
    for a in ASSETS:
        figs += [fig_returns_bands(a), fig_model_comparison(a), fig_diagnostics(a)]
    figs += [fig_model_comparison('SPY', '2023-01-01', '2026-12-31', tag='2023_2026')]
    figs += [fig_stress(p) for p in PERIODS]
    return [str(p) for p in figs]


def main():
    import time
    t0 = time.time()
    print('\n'.join(make_all())); print(f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
