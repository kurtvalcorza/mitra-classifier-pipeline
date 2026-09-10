#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / 'tutorials' / 'mitra_classifier_colab.ipynb'
AINF = ROOT / 'tutorials' / 'mitra_classifier_predictor_inference_colab.ipynb'


def source_text(cell):
    src = cell.get('source', '')
    return ''.join(src) if isinstance(src, list) else str(src)


def set_source(cell, text):
    old = cell.get('source', '')
    cell['source'] = text.splitlines(keepends=True) if isinstance(old, list) else text


def normalize(path: Path) -> None:
    nb = json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> None:
    nb = json.loads(AINF.read_text(encoding='utf-8'))
    old = '- Predictor provenance: read from `tutorial_run_metadata.json` when available'
    new = '- Predictor provenance: required from and validated against `tutorial_run_metadata.json` before deserialization'
    found = 0
    for cell in nb['cells']:
        text = source_text(cell)
        if old in text:
            if text.count(old) != 1:
                raise RuntimeError('ambiguous stale provenance footer')
            set_source(cell, text.replace(old, new, 1))
            found += 1
    if found != 1:
        raise RuntimeError(f'expected one stale provenance footer, found {found}')
    AINF.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    normalize(E2E)

    for path in (E2E, AINF):
        text = path.read_text(encoding='utf-8')
        if 'DIMER ZIP' in text:
            raise RuntimeError(f'{path.name} still contains stale DIMER ZIP wording')
    print('Normalized notebook JSON and removed final stale provenance wording.')


if __name__ == '__main__':
    main()
