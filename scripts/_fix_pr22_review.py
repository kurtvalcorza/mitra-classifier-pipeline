#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
MAIN = TUTORIALS / "mitra_classifier_colab.ipynb"
INFER = TUTORIALS / "mitra_classifier_predictor_inference_colab.ipynb"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"
README = TUTORIALS / "README.md"


def get_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict, text: str) -> None:
    if isinstance(cell.get("source"), list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def find_cell(nb: dict, needle: str) -> dict:
    matches = [cell for cell in nb["cells"] if needle in get_source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one cell containing {needle!r}; found {len(matches)}")
    return matches[0]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {text.count(old)}")
    return text.replace(old, new, 1)


def replace_block(text: str, start: str, end: str, replacement: str, label: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"{label}: start marker not found")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:i] + replacement + text[j:]


def write_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Restore the source inputs referenced by the committed pip-compile lock headers.
(TUTORIALS / "requirements-colab.in").write_text(
    "autogluon.tabular[mitra]==1.5.0\nlightgbm>=4.0,<4.8\n",
    encoding="utf-8",
)
(TUTORIALS / "requirements-inference.in").write_text(
    "autogluon.tabular[mitra]==1.5.0\n",
    encoding="utf-8",
)

main = json.loads(MAIN.read_text(encoding="utf-8"))
infer = json.loads(INFER.read_text(encoding="utf-8"))

# Keep notebook-embedded lock graphs byte-for-byte synchronized with repository locks.
for nb, lock_name in (
    (main, "requirements-colab.lock.txt"),
    (infer, "requirements-inference.lock.txt"),
):
    cell = find_cell(nb, "LOCKED_REQUIREMENTS =")
    src = get_source(cell)
    lock_text = (TUTORIALS / lock_name).read_text(encoding="utf-8")
    new_line = f"LOCKED_REQUIREMENTS = {lock_text!r}"
    src, count = re.subn(
        r"^LOCKED_REQUIREMENTS\s*=.*$",
        lambda _: new_line,
        src,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Could not synchronize embedded {lock_name}")
    set_source(cell, src)

# E2E: scope the declared target environment to Colab, matching its UI/API usage.
intro = find_cell(main, "**Profile:** `E2E`")
src = get_source(intro)
src = src.replace(
    "Google Colab or Jupyter with **Python 3.12**",
    "Google Colab with **Python 3.12**",
)
set_source(intro, src)

# E2E: DIMER ZIP is now a real self-describing offline package, not a weights-only ZIP.
step2_md = find_cell(main, "## 2. Acquire, verify, and lock the checkpoint")
src = get_source(step2_md)
src = replace_once(
    src,
    "- **DIMER ZIP** uploads the ZIP distributed through the DIMER Model Repository, which contains `model.safetensors`; the matching pinned `config.json` is fetched from upstream.",
    "- **DIMER ZIP** uploads a self-describing offline ZIP distributed through the DIMER Model Repository. Notebook Spec v1.0 packages must contain `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest is validated before either model file is staged; legacy weights-only ZIPs are refused rather than silently completed from the network.",
    "DIMER package documentation",
)
src = replace_once(
    src,
    "For a DIMER upload, the notebook validates the package against the repository-defined DIMER release manifest: exact model identifier, immutable revision, expected file name, and SHA-256.",
    "For a DIMER upload, the notebook reads and validates the manifest **from the uploaded ZIP itself**: manifest version, exact model identifier, immutable revision, exact package file set, and SHA-256 for both weights and configuration. A failed or incomplete package never falls back to upstream acquisition.",
    "DIMER manifest documentation",
)
set_source(step2_md, src)

step2 = find_cell(main, "def weights_from_dimer")
src = get_source(step2)
old_constants = """RELEASE_PACKAGE_MANIFEST = {
    'model_id': MODEL_ID,
    'revision': PINNED_REVISION,
    'files': {
        'model.safetensors': EXPECTED_WEIGHTS_SHA256,
        'config.json': EXPECTED_CONFIG_SHA256,
    },
}
"""
new_constants = """DIMER_PACKAGE_MANIFEST_FILENAME = 'dimer-model-manifest.json'
DIMER_PACKAGE_MANIFEST_VERSION = '1.0'
"""
src = replace_once(src, old_constants, new_constants, "DIMER manifest constants")

new_loader = '''def load_dimer_package(weights_dest, config_dest):
    from google.colab import files

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError('Upload exactly one DIMER offline model ZIP.')
    name, payload = next(iter(uploaded.items()))
    package_path = MODEL_DIR / Path(name).name
    if package_path.suffix.lower() != '.zip':
        raise ValueError('DIMER ZIP mode requires a .zip package with an embedded manifest.')
    package_path.write_bytes(payload)

    expected_release_files = {
        'model.safetensors': EXPECTED_WEIGHTS_SHA256,
        'config.json': EXPECTED_CONFIG_SHA256,
    }
    expected_archive_files = {DIMER_PACKAGE_MANIFEST_FILENAME, *expected_release_files}

    with zipfile.ZipFile(package_path) as z:
        by_name = {}
        for info in z.infolist():
            if info.is_dir():
                continue
            if '\\\\' in info.filename:
                raise RuntimeError(f'Backslash archive member paths are not allowed: {info.filename!r}')
            member = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if member.is_absolute() or '..' in member.parts:
                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')
            if mode == stat.S_IFLNK:
                raise RuntimeError(f'Symlink entries are not allowed: {info.filename!r}')
            if len(member.parts) != 1:
                raise RuntimeError(
                    f'DIMER package entries must be at the archive root; found {info.filename!r}.'
                )
            if member.name in by_name:
                raise RuntimeError(f'Duplicate DIMER package member: {member.name!r}')
            by_name[member.name] = info

        actual_archive_files = set(by_name)
        if actual_archive_files != expected_archive_files:
            raise RuntimeError(
                'DIMER package file set mismatch. '
                f'Missing={sorted(expected_archive_files - actual_archive_files)}; '
                f'unexpected={sorted(actual_archive_files - expected_archive_files)}'
            )

        try:
            package_manifest = json.loads(
                z.read(by_name[DIMER_PACKAGE_MANIFEST_FILENAME]).decode('utf-8')
            )
        except Exception as exc:
            raise RuntimeError('DIMER package manifest is not valid UTF-8 JSON.') from exc

        if not isinstance(package_manifest, dict):
            raise RuntimeError('DIMER package manifest must be a JSON object.')
        if package_manifest.get('manifest_version') != DIMER_PACKAGE_MANIFEST_VERSION:
            raise RuntimeError(
                f"Unsupported DIMER package manifest version: {package_manifest.get('manifest_version')!r}."
            )
        if package_manifest.get('model_id') != MODEL_ID:
            raise RuntimeError(
                f"DIMER package model identity mismatch: {package_manifest.get('model_id')!r}."
            )
        if package_manifest.get('revision') != PINNED_REVISION:
            raise RuntimeError(
                f"DIMER package revision mismatch: {package_manifest.get('revision')!r}."
            )
        manifest_files = package_manifest.get('files')
        if not isinstance(manifest_files, dict) or set(manifest_files) != set(expected_release_files):
            raise RuntimeError(
                'DIMER package manifest must enumerate exactly model.safetensors and config.json.'
            )

        verified_payloads = {}
        for filename, expected_release_digest in expected_release_files.items():
            declared_digest = str(manifest_files.get(filename, '')).lower()
            if len(declared_digest) != 64 or any(ch not in '0123456789abcdef' for ch in declared_digest):
                raise RuntimeError(f'DIMER package manifest has invalid SHA-256 for {filename!r}.')
            if declared_digest != expected_release_digest:
                raise RuntimeError(
                    f'DIMER package manifest digest for {filename!r} does not match this notebook release.'
                )
            file_payload = z.read(by_name[filename])
            actual_digest = hashlib.sha256(file_payload).hexdigest()
            if actual_digest != declared_digest:
                raise RuntimeError(
                    f'DIMER package payload digest mismatch for {filename!r}. '
                    f'Expected {declared_digest}; got {actual_digest}.'
                )
            verified_payloads[filename] = file_payload

    weights_dest.write_bytes(verified_payloads['model.safetensors'])
    config_dest.write_bytes(verified_payloads['config.json'])
    print(
        '✓ DIMER package manifest, identity, revision, exact file set, and payload digests verified:',
        MODEL_ID,
        PINNED_REVISION,
    )

'''
src = replace_block(
    src,
    "def weights_from_dimer(dest):\n",
    "def install_offline_snapshot",
    new_loader,
    "DIMER package loader",
)
src = replace_once(
    src,
    "if MODEL_SOURCE == 'DIMER ZIP':\n    weights_from_dimer(weights_path)\n    fetch_pinned('config.json', config_path)\nelse:\n",
    "if MODEL_SOURCE == 'DIMER ZIP':\n    load_dimer_package(weights_path, config_path)\nelse:\n",
    "DIMER source branch",
)
old_post_verify = """if MODEL_SOURCE == 'DIMER ZIP':
    manifest = RELEASE_PACKAGE_MANIFEST
    if manifest['model_id'] != MODEL_ID or manifest['revision'] != PINNED_REVISION:
        raise RuntimeError('DIMER release manifest model identity/revision does not match this notebook release.')
    if manifest['files'].get('model.safetensors') != weights_digest or manifest['files'].get('config.json') != config_digest:
        raise RuntimeError('DIMER release manifest file digests do not match the verified package/config bytes.')
    print('✓ DIMER package validated against release manifest:', manifest['model_id'], manifest['revision'])
"""
src = replace_once(src, old_post_verify, "", "remove hard-coded manifest self-check")
set_source(step2, src)

# E2E: fresh reload now exercises the same validation boundary as the companion.
step7_md = find_cell(main, "## 7. Reload smoke test")
src = get_source(step7_md)
src = replace_once(
    src,
    "Before treating the ZIP as reusable, this cell extracts the **exported archive** into a fresh directory, reloads it with `TabularPredictor.load(...)`, and confirms that predictions and probabilities match the in-memory predictor on a small holdout sample.",
    "Before treating the ZIP as reusable, this cell extracts the **exported archive** into a fresh directory, applies the same archive-safety, exact-manifest, provenance, runtime, and offline-loading gates used by the companion notebook, then reloads it with `TabularPredictor.load(...)` and confirms that predictions and probabilities match the in-memory predictor on a small holdout sample.",
    "reload boundary documentation",
)
src = replace_once(
    src,
    "This checks the actual packaging boundary users will rely on later, which is the check most tutorials skip.",
    "The reload fails before deserialization if required manifest files, digests, model identity, revision, format version, feature schema, or runtime provenance are absent or inconsistent. This is the actual packaging boundary downstream users rely on.",
    "reload failure documentation",
)
set_source(step7_md, src)

step7 = find_cell(main, "RELOAD_DIR = Path('/content/mitra-predictor-reload')")
reload_code = '''import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from autogluon.tabular import TabularPredictor

RELOAD_DIR = Path('/content/mitra-predictor-reload')
MAX_RELOAD_EXPANDED_BYTES = 4 * 1024 ** 3
if RELOAD_DIR.exists():
    shutil.rmtree(RELOAD_DIR)
RELOAD_DIR.mkdir(parents=True)


def safe_extract_predictor_archive(zip_path, destination):
    destination = destination.resolve()
    seen = set()
    expanded_bytes = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if '\\\\' in info.filename:
                raise RuntimeError(f'Backslash archive member paths are not allowed: {info.filename!r}')
            member = PurePosixPath(info.filename)
            if member.is_absolute() or '..' in member.parts:
                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeError(f'Symlink entries are not allowed: {info.filename!r}')
            normalized = member.as_posix()
            if normalized in seen:
                raise RuntimeError(f'Duplicate archive member path: {normalized!r}')
            seen.add(normalized)
            expanded_bytes += int(info.file_size)
            if expanded_bytes > MAX_RELOAD_EXPANDED_BYTES:
                raise RuntimeError('Predictor archive exceeds the 4 GiB expanded-size safety limit.')
            target = (destination / Path(*member.parts)).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f'Archive member escapes extraction root: {info.filename!r}')
        z.extractall(destination)


safe_extract_predictor_archive(archive, RELOAD_DIR)
reload_candidates = sorted({p.parent.resolve() for p in RELOAD_DIR.rglob('predictor.pkl')})
if len(reload_candidates) != 1:
    raise RuntimeError(
        f'Expected exactly one AutoGluon predictor root after reload extraction; found {len(reload_candidates)}.'
    )
reload_predictor_root = reload_candidates[0]

all_extracted_files = [p.resolve() for p in RELOAD_DIR.rglob('*') if p.is_file()]
outside_predictor_root = [
    p for p in all_extracted_files
    if p != reload_predictor_root and reload_predictor_root not in p.parents
]
if outside_predictor_root:
    raise RuntimeError(
        'Reload archive contains file(s) outside the predictor root: '
        f'{[str(p.relative_to(RELOAD_DIR.resolve())) for p in outside_predictor_root]}'
    )

reload_manifest_path = reload_predictor_root / 'artifact-manifest.json'
reload_metadata_path = reload_predictor_root / 'tutorial_run_metadata.json'
if not reload_manifest_path.exists():
    raise RuntimeError('Reload artifact is missing required artifact-manifest.json.')
if not reload_metadata_path.exists():
    raise RuntimeError('Reload artifact is missing required tutorial_run_metadata.json.')

reload_manifest = json.loads(reload_manifest_path.read_text())
if reload_manifest.get('artifact_format') != 'dimer-mitra-autogluon-predictor' or reload_manifest.get('artifact_format_version') != '1.0':
    raise RuntimeError('Reload artifact manifest format/version is unsupported.')
if reload_manifest.get('base_model') != MODEL_ID or reload_manifest.get('base_model_revision') != PINNED_REVISION:
    raise RuntimeError('Reload artifact manifest model identity/revision is inconsistent.')
reload_entries = reload_manifest.get('files')
if not isinstance(reload_entries, list) or not reload_entries:
    raise RuntimeError('Reload artifact manifest must contain a non-empty files list.')

reload_listed = {}
for entry in reload_entries:
    if not isinstance(entry, dict) or not {'path', 'size_bytes', 'sha256'} <= set(entry):
        raise RuntimeError(f'Malformed reload artifact manifest entry: {entry!r}')
    rel = str(entry['path'])
    rel_path = PurePosixPath(rel)
    if '\\\\' in rel or rel_path.is_absolute() or '..' in rel_path.parts or rel in reload_listed:
        raise RuntimeError(f'Unsafe or duplicate reload artifact manifest path: {rel!r}')
    if rel == 'artifact-manifest.json':
        raise RuntimeError('artifact-manifest.json must not list itself.')
    reload_listed[rel] = entry

reload_actual = {
    p.relative_to(reload_predictor_root).as_posix()
    for p in reload_predictor_root.rglob('*')
    if p.is_file() and p.resolve() != reload_manifest_path.resolve()
}
reload_expected = set(reload_listed)
if reload_actual != reload_expected:
    raise RuntimeError(
        f'Reload artifact manifest file set mismatch. Missing={sorted(reload_expected - reload_actual)}; '
        f'unexpected={sorted(reload_actual - reload_expected)}'
    )
for rel, entry in reload_listed.items():
    p = reload_predictor_root / Path(*PurePosixPath(rel).parts)
    if p.stat().st_size != int(entry['size_bytes']):
        raise RuntimeError(f'Reload artifact manifest size mismatch for {rel!r}.')
    expected_digest = str(entry['sha256']).lower()
    if len(expected_digest) != 64 or any(ch not in '0123456789abcdef' for ch in expected_digest):
        raise RuntimeError(f'Reload artifact manifest has invalid SHA-256 for {rel!r}.')
    if sha256_file(p) != expected_digest:
        raise RuntimeError(f'Reload artifact manifest SHA-256 mismatch for {rel!r}.')
print(f'✓ Reload artifact manifest verified: {len(reload_listed)} files, exact file set, sizes, and SHA-256 digests.')

reload_metadata = json.loads(reload_metadata_path.read_text())
required_reload_provenance = [
    'artifact_format', 'artifact_format_version', 'base_model', 'base_model_revision',
    'autogluon_version', 'weights_sha256', 'config_sha256', 'features',
]
missing_reload_provenance = [k for k in required_reload_provenance if not reload_metadata.get(k)]
if missing_reload_provenance:
    raise RuntimeError(f'Reload artifact provenance is incomplete: {missing_reload_provenance}')
if reload_metadata['artifact_format'] != 'dimer-mitra-autogluon-predictor' or reload_metadata['artifact_format_version'] != '1.0':
    raise RuntimeError('Reload artifact provenance format/version is unsupported.')
if reload_metadata['base_model'] != MODEL_ID or reload_metadata['base_model_revision'] != PINNED_REVISION:
    raise RuntimeError('Reload artifact provenance model identity/revision is inconsistent.')
if reload_metadata['weights_sha256'] != EXPECTED_WEIGHTS_SHA256 or reload_metadata['config_sha256'] != EXPECTED_CONFIG_SHA256:
    raise RuntimeError('Reload artifact provenance weight/config digests are inconsistent.')
if reload_metadata['autogluon_version'] != AUTOGLUON_VERSION:
    raise RuntimeError(
        f"Reload artifact expects AutoGluon {reload_metadata['autogluon_version']}, but runtime has {AUTOGLUON_VERSION}."
    )
if reload_manifest.get('base_model') != reload_metadata['base_model'] or reload_manifest.get('base_model_revision') != reload_metadata['base_model_revision']:
    raise RuntimeError('Reload artifact manifest and provenance disagree on model identity/revision.')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
print('✓ Reload provenance validated; network/model fallback disabled before deserialization.')

reloaded_predictor = TabularPredictor.load(str(reload_predictor_root))
smoke_X = holdout_data[FEATURE_COLUMNS].head(5).copy()

expected_pred = active_predictor.predict(smoke_X).reset_index(drop=True)
reloaded_pred = reloaded_predictor.predict(smoke_X).reset_index(drop=True)
if not expected_pred.equals(reloaded_pred):
    raise RuntimeError('Reload smoke test failed: class predictions changed after ZIP export/reload.')

expected_proba = active_predictor.predict_proba(smoke_X, as_multiclass=True).reset_index(drop=True)
reloaded_proba = reloaded_predictor.predict_proba(smoke_X, as_multiclass=True).reset_index(drop=True)
if list(expected_proba.columns) != list(reloaded_proba.columns):
    raise RuntimeError('Reload smoke test failed: probability class columns changed after reload.')
if not np.allclose(expected_proba.to_numpy(), reloaded_proba.to_numpy(), rtol=1e-6, atol=1e-8):
    raise RuntimeError('Reload smoke test failed: class probabilities changed after ZIP export/reload.')

print('✓ Exported predictor passed the downstream artifact boundary and reproduced smoke-test predictions.')
'''
set_source(step7, reload_code)

# Companion: declare only the environment the notebook actually supports.
infer_intro = find_cell(infer, "**Profile:** `ARTIFACT-INFERENCE`")
src = get_source(infer_intro)
src = src.replace(
    "Google Colab or Jupyter with **Python 3.12**",
    "Google Colab with **Python 3.12**",
)
set_source(infer_intro, src)

# Companion: enforce the entire extracted archive file set, and exclude only the canonical manifest.
infer_upload = find_cell(infer, "MANIFEST_PATH = PREDICTOR_ROOT / 'artifact-manifest.json'")
src = get_source(infer_upload)
src = replace_once(
    src,
    "PREDICTOR_ROOT = candidates[0]\nprint('✓ Predictor root:', PREDICTOR_ROOT)\n\nMANIFEST_PATH = PREDICTOR_ROOT / 'artifact-manifest.json'\n",
    "PREDICTOR_ROOT = candidates[0].resolve()\nprint('✓ Predictor root:', PREDICTOR_ROOT)\n\nall_extracted_files = [p.resolve() for p in EXTRACT_ROOT.rglob('*') if p.is_file()]\noutside_predictor_root = [\n    p for p in all_extracted_files\n    if p != PREDICTOR_ROOT and PREDICTOR_ROOT not in p.parents\n]\nif outside_predictor_root:\n    raise RuntimeError(\n        'Predictor ZIP contains file(s) outside the single predictor root: '\n        f'{[str(p.relative_to(EXTRACT_ROOT.resolve())) for p in outside_predictor_root]}'\n    )\n\nMANIFEST_PATH = PREDICTOR_ROOT / 'artifact-manifest.json'\n",
    "companion extraction root enforcement",
)
src = replace_once(
    src,
    "    if p.is_file() and p.name != 'artifact-manifest.json'\n",
    "    if p.is_file() and p.resolve() != MANIFEST_PATH.resolve()\n",
    "companion exact manifest exclusion",
)
src = replace_once(
    src,
    "    if rel_path.is_absolute() or '..' in rel_path.parts or rel in listed:\n        raise RuntimeError(f'Unsafe or duplicate artifact manifest path: {rel!r}')\n    listed[rel] = entry\n",
    "    if rel_path.is_absolute() or '..' in rel_path.parts or rel in listed:\n        raise RuntimeError(f'Unsafe or duplicate artifact manifest path: {rel!r}')\n    if rel == 'artifact-manifest.json':\n        raise RuntimeError('artifact-manifest.json must not list itself.')\n    listed[rel] = entry\n",
    "companion manifest self-entry guard",
)
set_source(infer_upload, src)

# README: provenance is mandatory now, and document the real DIMER package contract.
readme = README.read_text(encoding="utf-8")
readme = replace_once(
    readme,
    "- reads `tutorial_run_metadata.json` when present;",
    "- requires and validates `tutorial_run_metadata.json` before deserialization;",
    "README provenance wording",
)
anchor = "The exact dependency graphs used by the notebooks are committed as `requirements-colab.lock.txt` and `requirements-inference.lock.txt`; both release locks target Python 3.12. Release-grade status requires clean target-runtime execution evidence for the exact release revision; static CI alone is not execution evidence.\n"
addition = anchor + "\nThe lock inputs are committed as `requirements-colab.in` and `requirements-inference.in`. CI verifies that each notebook's embedded install graph is byte-for-byte identical to its committed lock, preventing notebook/lock drift.\n\nFor `MODEL_SOURCE = 'DIMER ZIP'`, Notebook Spec v1.0 expects a self-describing offline package containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. Legacy weights-only DIMER ZIPs are refused rather than completed from the network.\n"
readme = replace_once(readme, anchor, addition, "README lock/package documentation")
README.write_text(readme, encoding="utf-8")

# Validator: source inputs + embedded lock drift + stronger manifest/reload markers.
validator = VALIDATOR.read_text(encoding="utf-8")
insert_after = '''def validate_lockfile(path: Path) -> None:
    require(path.exists(), f"missing notebook lockfile: {path}")
    lock_text = path.read_text(encoding='utf-8')
    require('/home/runner/' not in lock_text, f"runner-local path leaked into {path.name}")
    for line in lock_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('--') or stripped.startswith('    #'):
            continue
        if stripped.startswith('    --hash='):
            continue
        token = stripped.split('\\\\', 1)[0].strip()
        require('==' in token, f"unlocked requirement in {path.name}: {stripped}")

'''
new_helpers = insert_after + '''def literal_string_assignment(code_cells: list[tuple[int, str]], name: str) -> str:
    values: list[str] = []
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                continue
            require(
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str),
                f"{name} must be a literal string",
            )
            values.append(node.value.value)
    require(len(values) == 1, f"expected exactly one literal assignment to {name}; found {len(values)}")
    return values[0]


def validate_lock_input(path: Path, expected: str) -> None:
    require(path.exists(), f"missing lock input: {path}")
    require(path.read_text(encoding='utf-8') == expected, f"unexpected lock input contents: {path.name}")


def validate_embedded_lock(notebook: Path, lockfile: Path) -> None:
    _, _, code = load_notebook(notebook)
    embedded = literal_string_assignment(code, 'LOCKED_REQUIREMENTS')
    committed = lockfile.read_text(encoding='utf-8')
    require(
        embedded == committed,
        f"{notebook.name} embedded dependency graph drifted from {lockfile.name}",
    )

'''
validator = replace_once(validator, insert_after, new_helpers, "validator lock helpers")
validator = validator.replace('"TabularPredictor.load(str(RELOAD_DIR))",', '"TabularPredictor.load(str(reload_predictor_root))",')
validator = validator.replace('"RELEASE_PACKAGE_MANIFEST",', '"DIMER_PACKAGE_MANIFEST_FILENAME",\n        "dimer-model-manifest.json",\n        "load_dimer_package",\n        "manifest_version",\n        "legacy weights-only ZIPs are refused",\n        "Reload artifact manifest verified",\n        "Reload provenance validated",')
old_direct_guard = '''    require(
        has_safe_direct_weights_copy_guard(parsed_code),
        "direct model.safetensors upload must avoid copying a path onto itself",
    )
'''
validator = replace_once(
    validator,
    old_direct_guard,
    '''    require(
        "weights_from_dimer" not in code_text and "load_dimer_package" in code_text,
        "DIMER ZIP path must use the manifest-validating offline package loader",
    )
''',
    "validator DIMER loader guard",
)
validator = validator.replace(
    '        "Artifact manifest verified",\n',
    '        "Artifact manifest verified",\n        "outside the single predictor root",\n        "artifact-manifest.json must not list itself",\n',
)
old_main = '''def main() -> int:
    validate_lockfile(ROOT / "tutorials" / "requirements-colab.lock.txt")
    validate_lockfile(ROOT / "tutorials" / "requirements-inference.lock.txt")
    validate_training_tutorial()
'''
new_main = '''def main() -> int:
    colab_lock = ROOT / "tutorials" / "requirements-colab.lock.txt"
    inference_lock = ROOT / "tutorials" / "requirements-inference.lock.txt"
    validate_lock_input(
        ROOT / "tutorials" / "requirements-colab.in",
        "autogluon.tabular[mitra]==1.5.0\\nlightgbm>=4.0,<4.8\\n",
    )
    validate_lock_input(
        ROOT / "tutorials" / "requirements-inference.in",
        "autogluon.tabular[mitra]==1.5.0\\n",
    )
    validate_lockfile(colab_lock)
    validate_lockfile(inference_lock)
    validate_embedded_lock(NOTEBOOK, colab_lock)
    validate_embedded_lock(INFERENCE_NOTEBOOK, inference_lock)
    validate_training_tutorial()
'''
validator = replace_once(validator, old_main, new_main, "validator main")
VALIDATOR.write_text(validator, encoding="utf-8")

write_nb(MAIN, main)
write_nb(INFER, infer)
print("Applied PR #22 reviewer remediation.")
