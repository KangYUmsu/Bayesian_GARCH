"""Integrity checks for quick mode: SHA-256 manifest of the saved, verified artifacts.

Quick mode refuses to run if any artifact is missing or differs from the manifest
recorded when the course package was built (results/artifact_manifest.json)."""
import hashlib
import json

from .config import SAVED_FORECASTS, SAVED_PRIMARY, RESULTS, PROJECT, ASSETS, WINDOWS, SEGS, resolve_artifact, artifact_key  # noqa: F401
from .tables import Store

MANIFEST = RESULTS / 'artifact_manifest.json'
PRIMARY_TABLES = [f'{k}_{s}.csv' for s in SEGS for k in
                  ('forecast_pooled', 'forecast', 'tail', 'info', 'alloc_summary', 'alloc_pairs',
                   'alloc_pooled_summary', 'alloc_pooled_pairs')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact_files():
    st = Store(SAVED_FORECASTS); files = []
    for seg in SEGS:
        for a in ASSETS:
            for w in WINDOWS:
                n = st.name(a, w, seg)
                files += [SAVED_FORECASTS / f'{n}_daily.csv.gz', SAVED_FORECASTS / f'{n}_fits.csv']
    return files + [SAVED_PRIMARY / t for t in PRIMARY_TABLES]


def build_manifest():
    data = {artifact_key(p): sha256(p) for p in artifact_files()}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=1))
    return data


def verify_manifest():
    expected = json.loads(MANIFEST.read_text())
    problems = []
    for rel, digest in expected.items():
        p = resolve_artifact(rel)
        if not p.exists():
            problems.append(f'missing: {rel}')
        elif sha256(p) != digest:
            problems.append(f'checksum mismatch: {rel}')
    if problems:
        raise RuntimeError('Saved artifacts failed verification; quick mode refuses to use them:\n' + '\n'.join(problems))
    return len(expected)
