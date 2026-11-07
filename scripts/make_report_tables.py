"""Write the LaTeX tables of report.tex from results/quick/tables (run quick mode first).

    python scripts/make_report_tables.py   # from the repository root
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T = ROOT / 'artifacts' / 'results' / 'quick' / 'tables'
OUT = ROOT / 'reports' / 'source' / 'tables'; OUT.mkdir(parents=True, exist_ok=True)
SEG = {'ext': '2017--22', 'test': '2023--26'}


def fmt_p(p):
    return '<.001' if p < .001 else f'{p:.3f}'.lstrip('0')


def forecast_table():
    labels = [('qlike_meanplug_minus_mle', r'QLIKE: posterior-mean plug-in $-$ MLE (shrinkage)'),
              ('qlike_integrated_minus_meanplug', r'QLIKE: integrated $-$ mean plug-in (integration)'),
              ('qlike_integrated_minus_mle', r'QLIKE: integrated $-$ MLE'),
              ('qlike_ewma_minus_mle', r'QLIKE: EWMA $-$ MLE'),
              ('qlike_t_minus_mle', r'QLIKE: Student-$t$ $-$ MLE'),
              ('ls_mixture_minus_meanplug', r'Log score: predictive mixture $-$ mean plug-in (integration)'),
              ('ls_mixture_minus_mle', r'Log score: predictive mixture $-$ MLE'),
              ('ls_laplace_minus_mle', r'Log score: Laplace mixture $-$ MLE'),
              ('ls_t_minus_mle', r'Log score: Student-$t$ $-$ MLE'),
              ('ls_t_minus_mixture', r'Log score: Student-$t$ $-$ predictive mixture')]
    P = {s: pd.read_csv(T / f'forecast_pooled_{s}.csv').set_index(['contrast', 'window']) for s in SEG}
    rows = []
    for k, lab in labels:
        cells = []
        for s in SEG:
            for w in (500, 1000):
                r = P[s].loc[(k, w)]
                cells.append(f'{r.pooled_mean * 1000:+.2f} ({fmt_p(r.p)})')
        rows.append(lab + ' & ' + ' & '.join(cells) + r' \\')
        if k == 'qlike_t_minus_mle':
            rows.append(r'\midrule')
    head = (r'\begin{tabular}{lrrrr}\toprule' '\n'
            r' & \multicolumn{2}{c}{2017--22 (independent)} & \multicolumn{2}{c}{2023--26 (previously inspected)}\\'
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}' '\n'
            r'Contrast ($\times10^{-3}$, DM $p$-value) & $W{=}500$ & $W{=}1000$ & $W{=}500$ & $W{=}1000$\\\midrule' '\n')
    (OUT / 'forecasts.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def forecast_split_tables():
    """forecasts.tex split into the H1 (QLIKE) and H2 (log score) blocks, plus the Student-t / Laplace ratio table."""
    P = {s: pd.read_csv(T / f'forecast_pooled_{s}.csv').set_index(['contrast', 'window']) for s in SEG}
    head = (r'\begin{tabular}{lrrrr}\toprule' '\n'
            r' & \multicolumn{2}{c}{2017--22 (primary)} & \multicolumn{2}{c}{2023--26 (previously inspected)}\\'
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}' '\n'
            r'Contrast ($\times10^{-3}$, DM $p$-value) & $W{=}500$ & $W{=}1000$ & $W{=}500$ & $W{=}1000$\\\midrule' '\n')
    blocks = {'forecasts_qlike.tex': [('qlike_integrated_minus_mle', r'Bayes integrated $-$ MLE'),
                                      ('qlike_meanplug_minus_mle', r'posterior-mean plug-in $-$ MLE (shrinkage only)'),
                                      ('qlike_integrated_minus_meanplug', r'integrated $-$ mean plug-in (integration only)'),
                                      ('qlike_t_minus_mle', r'Student-$t$ $-$ MLE'),
                                      ('qlike_ewma_minus_mle', r'EWMA $-$ MLE')],
              'forecasts_logscore.tex': [('ls_mixture_minus_mle', r'Bayes predictive mixture $-$ MLE'),
                                         ('ls_mixture_minus_meanplug', r'predictive mixture $-$ mean plug-in (integration only)'),
                                         ('ls_laplace_minus_mle', r'Laplace mixture $-$ MLE'),
                                         ('ls_t_minus_mle', r'Student-$t$ $-$ MLE'),
                                         ('ls_t_minus_mixture', r'Student-$t$ $-$ Bayes predictive mixture')]}
    for name, labels in blocks.items():
        rows = []
        for k, lab in labels:
            cells = [f'{P[s].loc[(k, w)].pooled_mean * 1000:+.2f} ({fmt_p(P[s].loc[(k, w)].p)})' for s in SEG for w in (500, 1000)]
            rows.append(lab + ' & ' + ' & '.join(cells) + r' \\')
        (OUT / name).write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
    g = lambda k, s, w: P[s].loc[(k, w)].pooled_mean
    rows = []
    for lab, fn in [(r'Bayes mixture $-$ MLE ($\times10^{-3}$)', lambda s, w: f'{1000 * g("ls_mixture_minus_mle", s, w):+.2f}'),
                    (r'Student-$t$ $-$ MLE ($\times10^{-3}$)', lambda s, w: f'{1000 * g("ls_t_minus_mle", s, w):+.2f}'),
                    (r'Student-$t$ gain $\div$ Bayesian gain', lambda s, w: f'{g("ls_t_minus_mle", s, w) / g("ls_mixture_minus_mle", s, w):.2f}'),
                    (r'Laplace gain $\div$ Bayesian gain', lambda s, w: f'{g("ls_laplace_minus_mle", s, w) / g("ls_mixture_minus_mle", s, w):.2f}')]:
        rows.append(lab + ' & ' + ' & '.join(fn(s, w) for s in SEG for w in (500, 1000)) + r' \\')
    (OUT / 'ratios.tex').write_text(head.replace(r'Contrast ($\times10^{-3}$, DM $p$-value)', 'Pooled log-score gain')
                                    + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def exposure_compare_table():
    """Mean over the 8 ETF-window configurations: MLE vs Bayes upper-90 vs exposure-matched MLE (single-ETF strategies)."""
    K = [('mle', 'MLE'), ('bayes_upper90', 'Bayes upper-90'), ('mle_const_rt', 'Exposure-matched MLE')]
    rows = []
    for s in SEG:
        A = pd.read_csv(T / f'alloc_summary_{s}.csv')
        for k, lab in K:
            x = A[A.strategy == k][['mean_exposure', 'realized_annual_volatility', 'annualized_return', 'sharpe_excess_cash',
                                    'max_drawdown', 'mean_daily_turnover']].mean()
            rows.append(f'{SEG[s] if k == "mle" else ""} & {lab} & {100 * x.mean_exposure:.1f} & {100 * x.realized_annual_volatility:.2f} & '
                        f'{100 * x.annualized_return:.2f} & {x.sharpe_excess_cash:.3f} & {100 * x.max_drawdown:.2f} & '
                        f'{100 * x.mean_daily_turnover:.2f}' + r' \\')
        rows.append(r'\midrule')
    head = (r'\begin{tabular}{llrrrrrr}\toprule' '\n'
            r'Period & Strategy & Exposure (\%) & Vol.\ (\%) & Return (\%) & Sharpe & Max DD (\%) & Turnover (\%/day)\\\midrule' '\n')
    (OUT / 'exposure_compare.tex').write_text(head + '\n'.join(rows[:-1]) + '\n\\bottomrule\\end{tabular}\n')


def exposure_stage_tables():
    """Two-stage H3 tables (2017-22, means over the 8 single-ETF configurations).
    Stage 1: MLE vs Bayes upper-90 only. Stage 2: Bayes upper-90 vs exposure-matched MLE.
    The last row counts configurations whose paired 95% block-bootstrap interval excludes zero."""
    A = pd.read_csv(T / 'alloc_summary_ext.csv'); P = pd.read_csv(T / 'alloc_pairs_ext.csv')
    cols = ['mean_exposure', 'annualized_return', 'realized_annual_volatility', 'sharpe_excess_cash', 'max_drawdown']
    g = A.groupby('strategy')[cols].mean()
    fmt = lambda x: f'{100 * x.mean_exposure:.1f}\\% & {100 * x.annualized_return:.2f}\\% & {100 * x.realized_annual_volatility:.2f}\\% & {x.sharpe_excess_cash:.3f} & ${100 * x.max_drawdown:.2f}\\%$'
    dfmt = lambda d: f'${100 * d.mean_exposure:+.1f}$ pp & ${100 * d.annualized_return:+.2f}$ pp & ${100 * d.realized_annual_volatility:+.2f}$ pp & ${d.sharpe_excess_cash:+.3f}$ & ${100 * d.max_drawdown:+.2f}$ pp'
    head = (r'\begin{tabular}{lrrrrr}\toprule' '\n'
            r'2017--22, mean of 8 single-ETF configurations & Exposure & Return & Volatility & Sharpe & Max DD\\\midrule' '\n')
    for name, (a, b), (la, lb) in [('exposure_stage1.tex', ('bayes_upper90', 'mle'), ('Bayesian upper-90', 'Standard MLE')),
                                   ('exposure_stage2.tex', ('bayes_upper90', 'mle_const_rt'), ('Bayesian upper-90', 'Exposure-matched MLE'))]:
        x = P[(P.a == a) & (P.b == b)]
        sh = int(((x.d_sharpe_lo > 0) | (x.d_sharpe_hi < 0)).sum()); dd = int(((x.d_maxdd_lo > 0) | (x.d_maxdd_hi < 0)).sum())
        rows = [f'{lb} & ' + fmt(g.loc[b]) + r' \\', f'{la} & ' + fmt(g.loc[a]) + r' \\', r'\midrule',
                f'Difference (upper-90 $-$ {"MLE" if b == "mle" else "matched MLE"}) & ' + dfmt(g.loc[a] - g.loc[b]) + r' \\',
                f'Configurations whose 95\\% CI excludes 0 & & & & {sh} of 8 & {dd} of 8' + r' \\']
        (OUT / name).write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def allocation_table():
    names = [('buy_hold', 'Buy and hold'), ('static_target', 'Static weight (pre-period calibrated)'), ('ewma', 'EWMA'),
             ('t_garch', 'Student-$t$ GARCH (MLE)'), ('mle', 'Gaussian GARCH, MLE'), ('bayes_integrated', 'Bayes, integrated variance'),
             ('bayes_upper90', 'Bayes, upper-90\\% variance'), ('mle_const_rt', 'MLE $\\times$ real-time haircut (control)'),
             ('laplace_upper90', 'Laplace (classical) upper-90\\%')]
    P = {s: pd.read_csv(T / f'alloc_pooled_summary_{s}.csv').set_index(['strategy', 'window']) for s in SEG}
    rows = []
    for k, lab in names:
        c = []
        for s in SEG:
            c += [f'{P[s].loc[(k, w), "sharpe"]:.3f}' for w in (500, 1000)]
            c += [f'{100 * P[s].loc[(k, w), "max_dd"]:.1f}' for w in (500, 1000)]
        rows.append(lab + ' & ' + ' & '.join(c) + r' \\')
    head = (r'\begin{tabular}{lrrrrrrrr}\toprule' '\n'
            r' & \multicolumn{4}{c}{2017--22} & \multicolumn{4}{c}{2023--26}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}' '\n'
            r' & \multicolumn{2}{c}{Sharpe} & \multicolumn{2}{c}{Max DD (\%)} & \multicolumn{2}{c}{Sharpe} & \multicolumn{2}{c}{Max DD (\%)}\\' '\n'
            r'Strategy & 500 & 1000 & 500 & 1000 & 500 & 1000 & 500 & 1000\\\midrule' '\n')
    (OUT / 'allocation.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def paired_table():
    pairs = [('bayes_upper90', 'mle', r'Upper-90 $-$ MLE'), ('bayes_upper90', 'mle_const_rt', r'Upper-90 $-$ MLE$\times$haircut (control)'),
             ('bayes_upper90', 'laplace_upper90', r'Upper-90 $-$ Laplace upper-90'), ('bayes_integrated', 'mle', r'Integrated $-$ MLE')]
    P = {s: pd.read_csv(T / f'alloc_pooled_pairs_{s}.csv').set_index(['a', 'b', 'window']) for s in SEG}
    rows = []
    for a, b, lab in pairs:
        c = []
        for s in SEG:
            for w in (500, 1000):
                r = P[s].loc[(a, b, w)]
                c.append(f'{r.d_sharpe:+.3f} [{r.d_sharpe_lo:+.3f}, {r.d_sharpe_hi:+.3f}]')
        rows.append(lab + ' & ' + ' & '.join(c) + r' \\')
    head = (r'\begin{tabular}{lcccc}\toprule' '\n'
            r' & \multicolumn{2}{c}{2017--22} & \multicolumn{2}{c}{2023--26}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}' '\n'
            r'$\Delta$Sharpe [95\% block-bootstrap CI] & $W{=}500$ & $W{=}1000$ & $W{=}500$ & $W{=}1000$\\\midrule' '\n')
    (OUT / 'paired.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def exposure_table():
    rows = []
    for s in SEG:
        S = pd.read_csv(T / f'alloc_summary_{s}.csv').pivot_table(index=['asset', 'window'], columns='strategy',
                                                                   values=['mean_exposure', 'max_drawdown', 'sharpe_excess_cash'])
        de = S['mean_exposure']['bayes_upper90'] - S['mean_exposure']['mle']
        d1 = S['max_drawdown']['bayes_upper90'] - S['max_drawdown']['mle']
        d2 = S['max_drawdown']['bayes_upper90'] - S['max_drawdown']['mle_const_rt']
        ds = S['sharpe_excess_cash']['bayes_upper90'] - S['sharpe_excess_cash']['mle_const_rt']
        rows.append(f'{SEG[s]} & {100*de.mean():+.1f} & {100*d1.mean():+.2f} ({int((d1>0).sum())}/8) & '
                    f'{100*d2.mean():+.2f} ({int((d2>0).sum())}/8) & {ds.mean():+.4f} ({int((ds>0).sum())}/8)' + r' \\')
    head = (r'\begin{tabular}{lrrrr}\toprule' '\n'
            r'Period & $\Delta$exposure (pp) & $\Delta$MaxDD vs MLE (pp) & $\Delta$MaxDD vs control (pp) & $\Delta$Sharpe vs control\\\midrule' '\n')
    (OUT / 'exposure.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def var_table():
    rows = []
    names = [('mle', 'Gaussian, MLE'), ('bayes_meanplug', 'Gaussian, posterior-mean plug-in'),
             ('bayes_mixture', 'Bayesian predictive mixture'), ('laplace_mixture', 'Laplace mixture'),
             ('t', 'Student-$t$ GARCH'), ('ewma', 'EWMA (Gaussian)')]
    D = {s: pd.read_csv(T / f'tail_{s}.csv') for s in SEG}
    for k, lab in names:
        c = []
        for s in SEG:
            for a in (0.01, 0.05):
                x = D[s][(D[s].method == k) & (D[s].alpha == a)]
                c.append(f'{100*x.rate.mean():.2f} ({int((x.kupiec_p < .05).sum())})')
        rows.append(lab + ' & ' + ' & '.join(c) + r' \\')
    head = (r'\begin{tabular}{lrrrr}\toprule' '\n'
            r' & \multicolumn{2}{c}{2017--22} & \multicolumn{2}{c}{2023--26}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}' '\n'
            r'VaR violation rate \% (Kupiec rejections of 8) & 1\% & 5\% & 1\% & 5\%\\\midrule' '\n')
    (OUT / 'var.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def fit_table():
    f = pd.read_csv(T / 'fit_summary.csv')
    g = f.groupby('segment').agg(refits=('refits', 'sum'), failed=('failed_convergence', 'sum'), bound=('mle_at_bound', 'sum'),
                                 rhat=('max_rhat', 'max'), ess=('min_tail_ess', 'min'), nu=('median_t_nu', 'median'))
    rows = [f'{SEG[s]} & {int(r.refits)} & {int(r.failed)} & {r.rhat:.3f} & {r.ess:.0f} & {int(r.bound)} & {r.nu:.1f}' + r' \\'
            for s, r in g.iterrows()]
    head = (r'\begin{tabular}{lrrrrrr}\toprule' '\n'
            r'Period & refits & failed strict diag. & max $\hat R$ & min tail ESS & MLE on $p$-bound & median $\hat\nu$ ($t$-GARCH)\\\midrule' '\n')
    (OUT / 'fits.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


def adequacy_table():
    """Out-of-sample residual diagnostics, 500-day window, 2017-22 (from results/adequacy)."""
    d = pd.read_csv(T.parents[1] / 'adequacy' / 'diagnostics_w500_ext.csv')
    names = [('mle', 'Gaussian MLE'), ('bayes_mixture', 'Bayes mixture'), ('t', 'Student-$t$')]
    rows = []
    for a in ['SPY', 'QQQ', 'TLT', 'GLD']:
        raw = d[(d.asset == a) & (d.model == 'raw returns (no model)')].iloc[0]
        for i, (m, lab) in enumerate(names):
            x = d[(d.asset == a) & (d.model == m)].iloc[0]
            first = f'{a} & {raw.acf1_z2:.2f}' if i == 0 else ' & '
            rows.append(f'{first} & {lab} & {x.acf1_z2:+.2f} & {fmt_p(x.lb10_z2_p)} & {x.skew_z:+.2f} & '
                        f'{x.exkurt_normal_scores:.2f} & {100*x.lower01:.1f} & {100*x.upper01:.1f} & {fmt_p(x.ks_pit_p)}' + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    head = (r'\begin{tabular}{llllrrrrrr}\toprule' '\n'
            r' & ACF$_1(r^2)$ & Model & ACF$_1(z^2)$ & LB$_{10}(z^2)$ $p$ & skew $z$ & ex.\ kurt $\Phi^{-1}$(PIT) & lower 1\% & upper 1\% & KS(PIT) $p$\\\midrule' '\n')
    (OUT / 'adequacy.tex').write_text(head + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')


if __name__ == '__main__':
    if not T.exists():
        sys.exit('run `python -m bgarch.reproduce quick` first')
    for fn in (forecast_table, forecast_split_tables, allocation_table, paired_table, exposure_table,
               exposure_compare_table, exposure_stage_tables, var_table, fit_table, adequacy_table):
        fn()
    print('tables written to', OUT)
