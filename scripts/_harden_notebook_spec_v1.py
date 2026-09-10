#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
MAIN_NB = TUTORIALS / "mitra_classifier_colab.ipynb"
INF_NB = TUTORIALS / "mitra_classifier_predictor_inference_colab.ipynb"
README = TUTORIALS / "README.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"


def source(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def put(cell, text):
    if isinstance(cell.get("source", ""), list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def one(nb, needle, kind=None):
    cells = [c for c in nb["cells"] if needle in source(c) and (kind is None or c.get("cell_type") == kind)]
    if len(cells) != 1:
        raise RuntimeError(f"expected one {kind or ''} cell containing {needle!r}; got {len(cells)}")
    return cells[0]


def repl(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence; got {count}: {old!r}")
    return text.replace(old, new, 1)


def clean_lock(path: Path, input_name: str):
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"#    pip-compile .*",
        f"#    pip-compile --output-file=tutorials/{path.name} --strip-extras tutorials/{input_name}",
        text,
        count=1,
    )
    text = re.sub(
        r"/home/runner/work/mitra-classifier-pipeline/mitra-classifier-pipeline/tutorials/" + re.escape(input_name),
        f"tutorials/{input_name}",
        text,
    )
    text = re.sub(
        r"/home/runner/work/mitra-classifier-pipeline/mitra-classifier-pipeline/tutorials/" + re.escape(path.name),
        f"tutorials/{path.name}",
        text,
    )
    if "/home/runner/" in text:
        raise RuntimeError(f"runner-local path survived in {path.name}")
    path.write_text(text, encoding="utf-8")
    return text


def replace_embedded_lock(nb, runtime_file: str, cleaned_lock: str):
    cell = one(nb, f"Path('{runtime_file}').write_text(LOCKED_REQUIREMENTS", "code")
    text = source(cell)
    lines = text.splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.startswith("LOCKED_REQUIREMENTS = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"LOCKED_REQUIREMENTS = {cleaned_lock!r}{newline}"
            found = True
            break
    if not found:
        raise RuntimeError(f"LOCKED_REQUIREMENTS assignment missing for {runtime_file}")
    put(cell, "".join(lines))


def python_guard(text: str, label: str):
    guard = (
        "import sys\n"
        "\n"
        "if sys.version_info[:2] != (3, 12):\n"
        "    raise RuntimeError(\n"
        "        f'This notebook release lock targets Python 3.12, but this runtime is {sys.version.split()[0]}. '\n"
        "        'Use a supported Python 3.12 Colab/Jupyter runtime, restart the session, and run top-to-bottom.'\n"
        "    )\n"
    )
    return repl(text, "import sys\n", guard, label)


def patch_main(lock_text: str):
    nb = json.loads(MAIN_NB.read_text(encoding="utf-8"))
    opening = one(nb, "# Mitra Classifier — End-to-End", "markdown")
    text = source(opening)
    text = text.replace(
        "- **Runtime:** any Colab runtime works for the default path (pretrained, in-context evaluation runs on CPU). A GPU is needed only for `RUN_FINE_TUNING`.",
        "- **Runtime:** Google Colab or Jupyter with **Python 3.12**. The default pretrained/in-context path runs on CPU; a GPU is needed only for `RUN_FINE_TUNING`.",
    )
    anchor = "**No DIMER Workbench access is required.** Data is processed in Google Colab, not by DIMER. Do not upload confidential, sensitive, or restricted data unless that environment is permitted.\n"
    addition = anchor + (
        "\n**This notebook does not demonstrate:** DIMER portal serving, regression, non-tabular tasks, or production fitness. Successful tutorial execution is workflow evidence, not deployment validation.\n\n"
        "Reference: [MODEL_CARD.md](https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/main/MODEL_CARD.md).\n"
    )
    text = repl(text, anchor, addition, "main exclusions")
    put(opening, text)

    setup = one(nb, "LOCKED_REQUIREMENTS =", "code")
    text = python_guard(source(setup), "main Python guard")
    put(setup, text)
    replace_embedded_lock(nb, "/content/mitra-requirements.lock.txt", lock_text)

    acquire_md = one(nb, "## 2. Acquire, verify, and lock the checkpoint", "markdown")
    text = source(acquire_md)
    text += (
        "\n**Remote-code boundary:** this tutorial does not enable Hugging Face `trust_remote_code`; execution uses the installed AutoGluon implementation plus the verified `safetensors` weights and pinned configuration.\n"
        "\nFor a DIMER upload, the notebook validates the package against the repository-defined DIMER release manifest: exact model identifier, immutable revision, expected file name, and SHA-256.\n"
    )
    put(acquire_md, text)

    acquire = one(nb, "MODEL_ID = 'autogluon/mitra-classifier'", "code")
    text = source(acquire)
    old = "EXPECTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'\nNETWORK_TIMEOUT_SECONDS = 30\n"
    new = "EXPECTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'\nDIMER_RELEASE_MANIFEST = {\n    'model_id': MODEL_ID,\n    'revision': PINNED_REVISION,\n    'files': {\n        'model.safetensors': EXPECTED_WEIGHTS_SHA256,\n        'config.json': EXPECTED_CONFIG_SHA256,\n    },\n}\nNETWORK_TIMEOUT_SECONDS = 30\n"
    text = repl(text, old, new, "DIMER release manifest")
    old = "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']\n\nweights_path = MODEL_DIR / 'model.safetensors'\n"
    new = "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']\n\nprint('Model:', MODEL_ID)\nprint('Revision:', PINNED_REVISION)\nweights_path = MODEL_DIR / 'model.safetensors'\n"
    text = repl(text, old, new, "model identity output")
    old = "weights_digest = verify(weights_path, EXPECTED_WEIGHTS_SHA256, 'model.safetensors')\nverify(config_path, EXPECTED_CONFIG_SHA256, 'config.json')\nSNAPSHOT_PATH = install_offline_snapshot(weights_path, config_path, weights_digest)\n"
    new = "weights_digest = verify(weights_path, EXPECTED_WEIGHTS_SHA256, 'model.safetensors')\nconfig_digest = verify(config_path, EXPECTED_CONFIG_SHA256, 'config.json')\nif MODEL_SOURCE == 'DIMER ZIP':\n    manifest = DIMER_RELEASE_MANIFEST\n    if manifest['model_id'] != MODEL_ID or manifest['revision'] != PINNED_REVISION:\n        raise RuntimeError('DIMER release manifest model identity/revision does not match this notebook release.')\n    if manifest['files'].get('model.safetensors') != weights_digest or manifest['files'].get('config.json') != config_digest:\n        raise RuntimeError('DIMER release manifest file digests do not match the verified package/config bytes.')\n    print('✓ DIMER package validated against release manifest:', manifest['model_id'], manifest['revision'])\nSNAPSHOT_PATH = install_offline_snapshot(weights_path, config_path, weights_digest)\n"
    text = repl(text, old, new, "DIMER release manifest verification")
    put(acquire, text)

    baseline_md = one(nb, "## 4b. Companion classical tree baselines", "markdown")
    text = source(baseline_md)
    text += (
        "\n**Baseline variability:** LightGBM and Random Forest both use `random_state=SEED` and `n_estimators=100`; the blend searches a fixed 101-point weight grid. The seeded tree baselines are intended to be repeatable under the locked software stack, while Mitra GPU execution may retain framework/kernel nondeterminism described in Step 1.\n"
    )
    put(baseline_md, text)

    inference = one(nb, "RUN_NEW_DATA_INFERENCE = False", "code")
    text = source(inference)
    old = "    pred = active.predict(X)\n    proba = active.predict_proba(X, as_multiclass=True)\n    out = new_data.copy()\n"
    new = "    pred = active.predict(X)\n    proba = active.predict_proba(X, as_multiclass=True)\n    reserved_output_columns = ['prediction'] + [f'probability_{label}' for label in proba.columns]\n    output_collisions = [name for name in reserved_output_columns if name in new_data.columns]\n    if output_collisions:\n        raise ValueError(\n            f'Inference CSV contains output column(s) reserved by this notebook: {output_collisions}. '\n            'Rename or remove them before inference.'\n        )\n    out = new_data.copy()\n"
    text = repl(text, old, new, "main output collision guard")
    put(inference, text)

    MAIN_NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_inference(lock_text: str):
    nb = json.loads(INF_NB.read_text(encoding="utf-8"))
    opening = one(nb, "# Mitra Classifier — Use an Exported Predictor", "markdown")
    text = source(opening)
    text = text.replace(
        "- Any compatible Colab runtime; completion time varies with dependency-cache state, archive size, and hardware, so no fixed runtime is promised.",
        "- Google Colab or Jupyter with **Python 3.12**; inference runs on CPU. Completion time varies with dependency-cache state, archive size, and hardware, so no fixed runtime is promised.",
    )
    anchor = "**You do not need the original DIMER ZIP, `model.safetensors`, `config.json`, or DIMER Workbench.** The ZIP already contains everything the predictor needs.\n"
    addition = anchor + (
        "\n**This notebook is inference-only:** it does not train, fine-tune, refit preprocessing, reacquire the base checkpoint, or validate production fitness.\n\n"
        "Reference: [MODEL_CARD.md](https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/main/MODEL_CARD.md).\n"
    )
    text = repl(text, anchor, addition, "inference exclusions")
    put(opening, text)

    setup = one(nb, "LOCKED_REQUIREMENTS =", "code")
    put(setup, python_guard(source(setup), "inference Python guard"))
    replace_embedded_lock(nb, "/content/mitra-inference-requirements.lock.txt", lock_text)

    upload_md = one(nb, "## 2. Upload and validate `mitra-predictor.zip`", "markdown")
    text = source(upload_md)
    text = text.replace(
        "- rejects archive members with absolute paths, `..` segments, or symlinks, and any member that would land outside the extraction folder;",
        "- rejects absolute paths, `..` traversal, backslash-style ambiguous member paths, symlinks, members escaping the extraction root, and archives whose declared expanded size exceeds 4 GiB;",
    )
    text += (
        "\n`artifact-manifest.json` is verified before deserialization: every listed file must exist with the recorded size and SHA-256, and unexpected unlisted files are rejected. This proves **internal archive consistency**, not sender authenticity. An attacker who can replace both the ZIP and its manifest can make them agree; use `EXPECTED_ZIP_SHA256` from a trusted channel when sender/archive authenticity matters.\n"
    )
    put(upload_md, text)

    upload = one(nb, "def safe_extract_zip", "code")
    text = source(upload)
    text = text.replace("import hashlib\n", "import hashlib\nimport json\n", 1)
    text = text.replace(
        "EXPECTED_ZIP_SHA256 = ''  # @param {type:'string'}\n",
        "EXPECTED_ZIP_SHA256 = ''  # @param {type:'string'}\nMAX_EXPANDED_BYTES = 4 * 1024 ** 3\n",
        1,
    )
    old = """def safe_extract_zip(zip_path, destination):\n    destination = destination.resolve()\n    with zipfile.ZipFile(zip_path) as z:\n        for info in z.infolist():\n            normalized = info.filename.replace('\\\\', '/')\n            member = Path(normalized)\n            if member.is_absolute() or '..' in member.parts:\n                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')\n            mode = (info.external_attr >> 16) & 0o170000\n            if mode == stat.S_IFLNK:\n                raise RuntimeError(f'Symlink entries are not allowed: {info.filename!r}')\n            target = (destination / member).resolve()\n            if target != destination and destination not in target.parents:\n                raise RuntimeError(f'Archive member escapes extraction root: {info.filename!r}')\n        z.extractall(destination)\n"""
    new = """def safe_extract_zip(zip_path, destination):\n    destination = destination.resolve()\n    with zipfile.ZipFile(zip_path) as z:\n        expanded_bytes = 0\n        for info in z.infolist():\n            if '\\\\' in info.filename:\n                raise RuntimeError(f'Backslash archive member paths are not allowed: {info.filename!r}')\n            member = Path(info.filename)\n            if member.is_absolute() or '..' in member.parts:\n                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')\n            mode = (info.external_attr >> 16) & 0o170000\n            if mode == stat.S_IFLNK:\n                raise RuntimeError(f'Symlink entries are not allowed: {info.filename!r}')\n            expanded_bytes += int(info.file_size)\n            if expanded_bytes > MAX_EXPANDED_BYTES:\n                raise RuntimeError(\n                    f'Archive expanded size exceeds the {MAX_EXPANDED_BYTES / (1024 ** 3):.0f} GiB safety limit.'\n                )\n            target = (destination / member).resolve()\n            if target != destination and destination not in target.parents:\n                raise RuntimeError(f'Archive member escapes extraction root: {info.filename!r}')\n        z.extractall(destination)\n"""
    text = repl(text, old, new, "companion archive hardening")
    anchor = "PREDICTOR_ROOT = candidates[0]\nprint('✓ Predictor root:', PREDICTOR_ROOT)\n"
    manifest_verify = anchor + """\nMANIFEST_PATH = PREDICTOR_ROOT / 'artifact-manifest.json'\nif not MANIFEST_PATH.exists():\n    raise RuntimeError('Required artifact-manifest.json is missing; refusing to deserialize an unverifiable predictor bundle.')\nartifact_manifest = json.loads(MANIFEST_PATH.read_text())\nif artifact_manifest.get('artifact_format') != 'dimer-mitra-autogluon-predictor' or artifact_manifest.get('artifact_format_version') != '1.0':\n    raise RuntimeError(\n        f\"Unsupported artifact manifest format/version: {artifact_manifest.get('artifact_format')!r} / {artifact_manifest.get('artifact_format_version')!r}.\"\n    )\nentries = artifact_manifest.get('files')\nif not isinstance(entries, list) or not entries:\n    raise RuntimeError('artifact-manifest.json must contain a non-empty files list.')\nlisted = {}\nfor entry in entries:\n    if not isinstance(entry, dict) or not {'path', 'size_bytes', 'sha256'} <= set(entry):\n        raise RuntimeError(f'Malformed artifact manifest entry: {entry!r}')\n    rel = str(entry['path'])\n    if '\\\\' in rel:\n        raise RuntimeError(f'Backslash manifest paths are not allowed: {rel!r}')\n    rel_path = Path(rel)\n    if rel_path.is_absolute() or '..' in rel_path.parts or rel in listed:\n        raise RuntimeError(f'Unsafe or duplicate artifact manifest path: {rel!r}')\n    listed[rel] = entry\nactual = {\n    p.relative_to(PREDICTOR_ROOT).as_posix()\n    for p in PREDICTOR_ROOT.rglob('*')\n    if p.is_file() and p.name != 'artifact-manifest.json'\n}\nexpected = set(listed)\nif actual != expected:\n    raise RuntimeError(\n        f'Artifact manifest file set mismatch. Missing={sorted(expected - actual)}; unexpected={sorted(actual - expected)}'\n    )\nfor rel, entry in listed.items():\n    p = PREDICTOR_ROOT / rel\n    if p.stat().st_size != int(entry['size_bytes']):\n        raise RuntimeError(f'Artifact manifest size mismatch for {rel!r}.')\n    digest = sha256_file(p)\n    expected_digest = str(entry['sha256']).lower()\n    if len(expected_digest) != 64 or any(ch not in '0123456789abcdef' for ch in expected_digest):\n        raise RuntimeError(f'Artifact manifest has invalid SHA-256 for {rel!r}.')\n    if digest != expected_digest:\n        raise RuntimeError(f'Artifact manifest SHA-256 mismatch for {rel!r}.')\nprint(f'✓ Artifact manifest verified: {len(listed)} files, exact file set, sizes, and SHA-256 digests.')\n"""
    text = repl(text, anchor, manifest_verify, "manifest verification")
    put(upload, text)

    load = one(nb, "METADATA_PATH = PREDICTOR_ROOT", "code")
    text = source(load)
    insert = "METADATA_PATH = PREDICTOR_ROOT / 'tutorial_run_metadata.json'\n"
    constants = """SUPPORTED_MODEL_ID = 'autogluon/mitra-classifier'\nSUPPORTED_MODEL_REVISION = 'c425e9fa0910a6be1c494321792e7ba2a1367b1a'\nSUPPORTED_WEIGHTS_SHA256 = 'e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19'\nSUPPORTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'\n\n""" + insert
    text = repl(text, insert, constants, "supported artifact identity constants")
    anchor = "if run_metadata['artifact_format'] != 'dimer-mitra-autogluon-predictor' or run_metadata['artifact_format_version'] != '1.0':\n    raise RuntimeError(\n        f\"Unsupported predictor artifact {run_metadata['artifact_format']!r} version {run_metadata['artifact_format_version']!r}.\"\n    )\n"
    stronger = anchor + """if run_metadata['base_model'] != SUPPORTED_MODEL_ID or run_metadata['base_model_revision'] != SUPPORTED_MODEL_REVISION:\n    raise RuntimeError(\n        f\"Unsupported model identity/revision: {run_metadata['base_model']!r} @ {run_metadata['base_model_revision']!r}.\"\n    )\nif run_metadata.get('weights_sha256') != SUPPORTED_WEIGHTS_SHA256 or run_metadata.get('config_sha256') != SUPPORTED_CONFIG_SHA256:\n    raise RuntimeError('Predictor provenance weight/config digests do not match this notebook release.')\nif artifact_manifest.get('base_model') != SUPPORTED_MODEL_ID or artifact_manifest.get('base_model_revision') != SUPPORTED_MODEL_REVISION:\n    raise RuntimeError('Artifact manifest model identity/revision does not match this notebook release.')\n"""
    text = repl(text, anchor, stronger, "supported artifact identity validation")
    text = text.replace(
        "    'AutoGluon runtime': AUTOGLUON_VERSION,\n",
        "    'Artifact format': run_metadata['artifact_format'],\n    'Artifact format version': run_metadata['artifact_format_version'],\n    'AutoGluon runtime': AUTOGLUON_VERSION,\n",
        1,
    )
    put(load, text)

    INF_NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_readme():
    text = README.read_text(encoding="utf-8")
    text = text.replace(
        "The exact dependency graphs used by the notebooks are committed as `requirements-colab.lock.txt` and `requirements-inference.lock.txt`.",
        "The exact dependency graphs used by the notebooks are committed as `requirements-colab.lock.txt` and `requirements-inference.lock.txt`; both release locks target Python 3.12.",
    )
    needle = "- rejects path traversal and symlink entries before extraction;"
    replacement = "- rejects absolute/traversal/backslash/symlink archive paths, enforces extraction containment and a 4 GiB expanded-size ceiling;\n- requires and verifies `artifact-manifest.json` against the exact extracted file set, per-file sizes, and SHA-256 digests;"
    if needle in text:
        text = text.replace(needle, replacement, 1)
    README.write_text(text, encoding="utf-8")


def patch_validator():
    text = VALIDATOR.read_text(encoding="utf-8")
    old = """def validate_lockfile(path: Path) -> None:\n    require(path.exists(), f\"missing notebook lockfile: {path}\")\n    for line in path.read_text(encoding='utf-8').splitlines():\n"""
    new = """def validate_lockfile(path: Path) -> None:\n    require(path.exists(), f\"missing notebook lockfile: {path}\")\n    lock_text = path.read_text(encoding='utf-8')\n    require('/home/runner/' not in lock_text, f\"runner-local path leaked into {path.name}\")\n    for line in lock_text.splitlines():\n"""
    text = repl(text, old, new, "lock path hygiene validator")

    main_marker_anchor = '        "Symlink entries are not allowed",\n'
    main_extra = main_marker_anchor + (
        '        "Python 3.12",\n'
        '        "DIMER_RELEASE_MANIFEST",\n'
        '        "Remote-code boundary",\n'
        '        "Baseline variability",\n'
        '        "output column(s) reserved by this notebook",\n'
        '        "This notebook does not demonstrate",\n'
    )
    text = repl(text, main_marker_anchor, main_extra, "main final markers")

    inf_marker_anchor = '        "What a successful artifact-inference run proves",\n'
    inf_extra = inf_marker_anchor + (
        '        "MAX_EXPANDED_BYTES",\n'
        '        "Backslash archive member paths are not allowed",\n'
        '        "Artifact manifest file set mismatch",\n'
        '        "Artifact manifest verified",\n'
        '        "SUPPORTED_MODEL_REVISION",\n'
        '        "Artifact format version",\n'
        '        "internal archive consistency",\n'
        '        "Python 3.12",\n'
        '        "This notebook is inference-only",\n'
    )
    text = repl(text, inf_marker_anchor, inf_extra, "inference final markers")
    VALIDATOR.write_text(text, encoding="utf-8")


def main():
    main_lock = clean_lock(TUTORIALS / "requirements-colab.lock.txt", "requirements-colab.in")
    inf_lock = clean_lock(TUTORIALS / "requirements-inference.lock.txt", "requirements-inference.in")
    patch_main(main_lock)
    patch_inference(inf_lock)
    patch_readme()
    patch_validator()
    print("Final Notebook Spec v1 hardening applied.")


if __name__ == "__main__":
    main()
