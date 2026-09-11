#!/usr/bin/env python3
import json
from pathlib import Path

notebook_path = Path('tutorials/mitra_classifier_colab.ipynb')
nb = json.loads(notebook_path.read_text(encoding='utf-8'))
old_loop = "manifest_files = []\nfor file_path in sorted(p for p in active_path.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json'):\n"
new_loop = "artifact_manifest_path = (active_path / 'artifact-manifest.json').resolve()\nmanifest_files = []\nfor file_path in sorted(p for p in active_path.rglob('*') if p.is_file() and p.resolve() != artifact_manifest_path):\n"
old_write = "(active_path / 'artifact-manifest.json').write_text(json.dumps(artifact_manifest, indent=2))"
new_write = "artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2))"
changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    text = ''.join(source) if isinstance(source, list) else str(source)
    if old_loop not in text:
        continue
    if old_write not in text:
        raise SystemExit('manifest write anchor missing')
    text = text.replace(old_loop, new_loop, 1).replace(old_write, new_write, 1)
    cell['source'] = text.splitlines(keepends=True)
    changed = True
    break
if not changed:
    raise SystemExit('manifest producer loop anchor missing')
notebook_path.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')

validator_path = Path('scripts/validate_colab_tutorial.py')
validator = validator_path.read_text(encoding='utf-8')
anchor = '''    require(
        re.search(r"\\bDIMER_[A-Z0-9_]+\\b", code_text) is None,
        "standalone tutorial must not depend on DIMER_* runtime variables",
    )
'''
insert = '''    require(
        "p.name != 'artifact-manifest.json'" not in code_text,
        "artifact manifest producer must not exclude nested files by basename",
    )
    require(
        "artifact_manifest_path = (active_path / 'artifact-manifest.json').resolve()" in code_text
        and "p.resolve() != artifact_manifest_path" in code_text
        and "artifact_manifest_path.write_text" in code_text,
        "artifact manifest producer must exclude only the canonical root manifest path",
    )
'''
if insert not in validator:
    if validator.count(anchor) != 1:
        raise SystemExit(f'validator training anchor count={validator.count(anchor)}')
    validator = validator.replace(anchor, anchor + insert, 1)
validator_path.write_text(validator, encoding='utf-8')
