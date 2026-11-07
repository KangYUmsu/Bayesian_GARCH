"""Keep the course reviewer copies identical to the authoritative top-level reports and notebooks.

    python scripts/sync_course.py                    # copy reports/ and notebooks/ into course/, then verify
    python scripts/sync_course.py --check            # verify only; exit 1 if any copy differs or is missing
    python scripts/sync_course.py --refresh-figures  # also re-copy the figures the LaTeX sources depend on

Authoritative: reports/ (PDFs + source/) and notebooks/*.ipynb. Mirrors: course/reports/ and course/notebooks/.
Real file copies are made (no symbolic links); stale files inside the mirrors are removed; LaTeX auxiliary files
and notebook checkpoints are never copied.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(ROOT / 'reports', ROOT / 'course' / 'reports'), (ROOT / 'notebooks', ROOT / 'course' / 'notebooks')]
SKIP_SUFFIX = {'.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk', '.bbl', '.blg', '.nav', '.snm', '.synctex.gz'}
SKIP_PARTS = {'.ipynb_checkpoints', '__pycache__', '.DS_Store'}

# Figures used by the LaTeX sources, and where their authoritative versions are produced.
NARR = ROOT / 'artifacts' / 'figures' / 'narrative'
PRES_FIG = ROOT / 'course' / 'presentations' / 'source' / 'figures'      # scripts/make_presentation_figures.py output
REPORT_FIGS = {**{n: ROOT / 'artifacts' / 'figures' / n for n in
                  ('fig1_logscore_gains.png', 'fig2_sharpe_differences.png', 'fig4_kurtosis.png', 'adequacy_exceedance_summary.png')},
               **{n: NARR / n for n in ('rolling_design.pdf', 'qlike_contrasts.pdf', 'predictive_densities.pdf',
                                       'portfolio_flowchart.pdf', 'exposure_control.pdf')},
               'weights_spy2020.pdf': PRES_FIG / 'weights_spy2020.pdf'}
SLIDE_FIGS = [NARR / n for n in ('rolling_design.pdf', 'qlike_contrasts_slide.pdf', 'predictive_densities_slide.pdf',
                                 'logscore_gains_slide.pdf', 'portfolio_flowchart.pdf', 'exposure_control_slide.pdf')]


def files(base):
    return sorted(p.relative_to(base) for p in base.rglob('*') if p.is_file()
                  and p.suffix not in SKIP_SUFFIX and not (set(p.parts) & SKIP_PARTS))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def refresh_figures():
    dst = ROOT / 'reports' / 'source' / 'figures'
    for name, src in REPORT_FIGS.items():
        shutil.copy2(src, dst / name)
    for src in SLIDE_FIGS:
        shutil.copy2(src, PRES_FIG / src.name)
    print(f'refreshed {len(REPORT_FIGS)} report figures and {len(SLIDE_FIGS)} slide figures')


def mirror(src, dst):
    want = set(files(src))
    for rel in want:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dst / rel).exists() or sha(src / rel) != sha(dst / rel):
            shutil.copy2(src / rel, dst / rel)
    for rel in set(files(dst)) - want:                     # stale copies
        (dst / rel).unlink()


def check():
    bad = []
    for src, dst in PAIRS:
        a, b = set(files(src)), set(files(dst)) if dst.exists() else set()
        bad += [f'missing in {dst.relative_to(ROOT)}: {r}' for r in sorted(a - b)]
        bad += [f'extra in {dst.relative_to(ROOT)}: {r}' for r in sorted(b - a)]
        bad += [f'differs: {dst.relative_to(ROOT) / r}' for r in sorted(a & b) if sha(src / r) != sha(dst / r)]
        bad += [f'symbolic link: {dst.relative_to(ROOT) / r}' for r in sorted(b) if (dst / r).is_symlink()]
        print(f'{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}: {len(a & b)} of {len(a)} files identical')
    for msg in bad:
        print('  ' + msg)
    return not bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='verify only')
    ap.add_argument('--refresh-figures', action='store_true', help='re-copy figure dependencies of the LaTeX sources first')
    a = ap.parse_args()
    if not a.check:
        if a.refresh_figures:
            refresh_figures()
        for s, d in PAIRS:
            mirror(s, d)
    ok = check()
    print('synchronized' if ok else 'NOT synchronized')
    sys.exit(0 if ok else 1)
