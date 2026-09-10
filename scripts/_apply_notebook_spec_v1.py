#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
MAIN_NB = TUTORIALS / "mitra_classifier_colab.ipynb"
INF_NB = TUTORIALS / "mitra_classifier_predictor_inference_colab.ipynb"
README = TUTORIALS / "README.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"
RELEASE = ROOT / "RELEASE_CHECKLIST.md"


def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def set_src(cell, text):
    old = cell.get("source", "")
    if isinstance(old, list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def cell_with(nb, needle, cell_type=None):
    matches = [c for c in nb["cells"] if needle in src(c) and (cell_type is None or c.get("cell_type") == cell_type)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {needle!r}; found {len(matches)}")
    return matches[0]


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one occurrence of {old!r}; found {n}")
    return text.replace(old, new, 1)


def compile_locks():
    inputs = {
        TUTORIALS / "requirements-colab.in": "autogluon.tabular[mitra]==1.5.0\nlightgbm==4.6.0\n",
        TUTORIALS / "requirements-inference.in": "autogluon.tabular[mitra]==1.5.0\n",
    }
    for path, content in inputs.items():
        path.write_text(content, encoding="utf-8")
    for infile, outfile in (
        (TUTORIALS / "requirements-colab.in", TUTORIALS / "requirements-colab.lock.txt"),
        (TUTORIALS / "requirements-inference.in", TUTORIALS / "requirements-inference.lock.txt"),
    ):
        subprocess.check_call([
            "pip-compile",
            "--quiet",
            "--resolver=backtracking",
            "--strip-extras",
            "--no-emit-index-url",
            "--no-emit-trusted-host",
            "--output-file",
            str(outfile),
            str(infile),
        ])
        infile.unlink()


def locked_install_code(lock_path: Path, runtime_path: str) -> str:
    lock = lock_path.read_text(encoding="utf-8")
    return (
        "from pathlib import Path\n"
        f"LOCKED_REQUIREMENTS = {lock!r}\n"
        f"Path({runtime_path!r}).write_text(LOCKED_REQUIREMENTS, encoding='utf-8')\n"
        f"%pip install -q -r {runtime_path}\n"
    )


def patch_main():
    nb = json.loads(MAIN_NB.read_text(encoding="utf-8"))

    opening = cell_with(nb, "# Mitra Classifier — End-to-End", "markdown")
    text = src(opening)
    marker = "# Mitra Classifier — End-to-End Classification with Your Own Data\n"
    addition = (
        marker
        + "\n**Notebook Specification:** DIMER Notebook Specification v1.0  \n"
        + "**Profile:** `E2E`  \n"
        + "**Release status:** Candidate — static conformance checks are automated; a clean Google Colab execution of this exact revision remains the release gate.\n"
    )
    text = replace_once(text, marker, addition, "main opening profile")
    text = text.replace(
        "With the bundled sample the default run typically takes a few minutes, most of it the dependency install and the 300 MB checkpoint download; that figure comes from development runs on a local GPU workstation, not from a measured Colab session.",
        "Runtime varies with Colab hardware, dependency-cache state, and network throughput; this notebook does not promise a fixed completion time.",
    )
    set_src(opening, text)

    setup = cell_with(nb, "%pip install -q \"autogluon.tabular[mitra]==1.5.0\"", "code")
    code = src(setup)
    old = "%pip install -q \"autogluon.tabular[mitra]==1.5.0\" \"lightgbm>=4.0,<4.8\"\n"
    new = locked_install_code(TUTORIALS / "requirements-colab.lock.txt", "/content/mitra-requirements.lock.txt")
    code = replace_once(code, old, new, "main locked install")
    code = replace_once(code, "import sys\n", "import sys\n", "main sys import")
    code = code.replace(
        "print('AutoGluon:', AUTOGLUON_VERSION)\n",
        "print('Python:', sys.version.split()[0])\nprint('AutoGluon:', AUTOGLUON_VERSION)\n",
        1,
    )
    set_src(setup, code)

    setup_md = cell_with(nb, "## 1. Install and inspect the runtime", "markdown")
    text = src(setup_md)
    text = text.replace(
        "Mitra ships as an extra of AutoGluon (`autogluon.tabular[mitra]`), which pins a compatible PyTorch range",
        "This notebook installs from an embedded, exact-version lock generated from `tutorials/requirements-colab.lock.txt`. Mitra ships as an extra of AutoGluon (`autogluon.tabular[mitra]`), which pins a compatible PyTorch range",
    )
    text += (
        "\n**Determinism:** the notebook seeds Python, NumPy, and PyTorch where stochastic operations are used. "
        "GPU kernels and AutoGluon/PyTorch internals may still be nondeterministic, so repeated fine-tuning runs can differ slightly even with the same seed.\n"
    )
    set_src(setup_md, text)

    acquire = cell_with(nb, "def weights_from_dimer", "code")
    code = src(acquire)
    code = code.replace(
        "import hashlib, json, os, random, shutil, urllib.request, zipfile\nfrom pathlib import Path\n",
        "import hashlib, json, os, random, shutil, stat, urllib.request, zipfile\nfrom pathlib import Path, PurePosixPath\n",
        1,
    )
    old = """    with zipfile.ZipFile(p) as z:\n        matches = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).name == 'model.safetensors']\n        if len(matches) != 1:\n            raise RuntimeError(f'Expected one model.safetensors in the DIMER ZIP; found {len(matches)}.')\n        with z.open(matches[0]) as src, open(dest, 'wb') as dst:\n            shutil.copyfileobj(src, dst)\n"""
    new = """    with zipfile.ZipFile(p) as z:\n        for info in z.infolist():\n            normalized = info.filename.replace('\\\\', '/')\n            member = PurePosixPath(normalized)\n            mode = (info.external_attr >> 16) & 0o170000\n            if member.is_absolute() or '..' in member.parts:\n                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')\n            if mode == stat.S_IFLNK:\n                raise RuntimeError(f'Symlink entries are not allowed: {info.filename!r}')\n        matches = [i for i in z.infolist() if not i.is_dir() and PurePosixPath(i.filename.replace('\\\\', '/')).name == 'model.safetensors']\n        if len(matches) != 1:\n            raise RuntimeError(f'Expected one model.safetensors in the DIMER ZIP; found {len(matches)}.')\n        with z.open(matches[0]) as src, open(dest, 'wb') as dst:\n            shutil.copyfileobj(src, dst)\n"""
    code = replace_once(code, old, new, "main DIMER ZIP validation")
    set_src(acquire, code)

    data = cell_with(nb, "def prepare(df, name):", "code")
    code = src(data)
    old = """    out = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore').dropna(subset=[TARGET_COLUMN]).copy()\n    duplicate_rows = int(out.duplicated().sum())\n"""
    new = """    dropped_feature_columns = [c for c in drop_columns if c in df.columns]\n    null_target_rows = int(df[TARGET_COLUMN].isna().sum())\n    out = df.drop(columns=dropped_feature_columns, errors='ignore').dropna(subset=[TARGET_COLUMN]).copy()\n    if dropped_feature_columns or null_target_rows:\n        print(f'ℹ {name}: preprocessing modified the input: dropped feature columns={dropped_feature_columns or []}; rows dropped for null target={null_target_rows}.')\n    duplicate_rows = int(out.duplicated().sum())\n"""
    code = replace_once(code, old, new, "main data mutation reporting")
    set_src(data, code)

    fit_md = cell_with(nb, "## 4. Evaluate pretrained Mitra", "markdown")
    text = src(fit_md)
    text += (
        "\n**Reproducibility note:** `SEED` fixes the notebook-controlled random choices, but exact fine-tuning reproducibility is not guaranteed across GPU kernels, driver/runtime revisions, or AutoGluon/PyTorch internals. Treat small run-to-run deltas accordingly.\n"
    )
    set_src(fit_md, text)

    inference_md = cell_with(nb, "## 5. Classify new rows", "markdown")
    text = src(inference_md)
    text += (
        "\n**Decision rule and uncertainty:** `predict()` returns the class selected by the predictor; for standard multiclass use this is equivalent to choosing the class with the highest returned class probability. The probability vector is useful for ranking and thresholding, but calibration for your deployment population has not been established by this notebook.\n"
    )
    set_src(inference_md, text)

    meaning = cell_with(nb, "**What the numbers mean.**", "markdown")
    text = src(meaning)
    text = text.replace(
        "Use it, not accuracy, when you care about calibrated probabilities.",
        "Use it alongside accuracy when probability quality matters; calibration itself is not established by this notebook and must be evaluated separately for the deployment domain.",
    )
    set_src(meaning, text)

    export_md = cell_with(nb, "## 6. Export the reusable predictor", "markdown")
    text = src(export_md)
    text += (
        "\n**Data disclosure warning:** this predictor bundle contains the fitted preprocessing state and Mitra's training/support context. Depending on the data and AutoGluon serialization details, the archive may retain values derived from or copied from the training dataset. Apply the same confidentiality, licensing, retention, and sharing rules to the predictor ZIP that apply to the source data.\n"
        "\nThe export also writes `artifact-manifest.json` with an explicit artifact format/version plus per-file sizes and SHA-256 digests, then prints the SHA-256 of the final ZIP for transfer verification.\n"
    )
    set_src(export_md, text)

    export = cell_with(nb, "metadata = {", "code")
    code = src(export)
    code = code.replace(
        "    'base_model': MODEL_ID,\n",
        "    'artifact_format': 'dimer-mitra-autogluon-predictor',\n    'artifact_format_version': '1.0',\n    'base_model': MODEL_ID,\n",
        1,
    )
    code = code.replace(
        "    'cuda_available': torch.cuda.is_available(),\n",
        "    'cuda_available': torch.cuda.is_available(),\n    'determinism_note': 'Python/NumPy/PyTorch are seeded; GPU kernels and framework internals may remain nondeterministic.',\n",
        1,
    )
    old = """(active_path / 'tutorial_run_metadata.json').write_text(json.dumps(metadata, indent=2))\nPath('/content/mitra-predictor.zip').unlink(missing_ok=True)\narchive = shutil.make_archive('/content/mitra-predictor', 'zip', root_dir=active_path)\nprint('✓ Predictor archive:', archive)\n"""
    new = """(active_path / 'tutorial_run_metadata.json').write_text(json.dumps(metadata, indent=2))\n\nmanifest_files = []\nfor file_path in sorted(p for p in active_path.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json'):\n    manifest_files.append({\n        'path': file_path.relative_to(active_path).as_posix(),\n        'size_bytes': file_path.stat().st_size,\n        'sha256': sha256_file(file_path),\n    })\nartifact_manifest = {\n    'artifact_format': 'dimer-mitra-autogluon-predictor',\n    'artifact_format_version': '1.0',\n    'base_model': MODEL_ID,\n    'base_model_revision': PINNED_REVISION,\n    'files': manifest_files,\n}\n(active_path / 'artifact-manifest.json').write_text(json.dumps(artifact_manifest, indent=2))\n\nPath('/content/mitra-predictor.zip').unlink(missing_ok=True)\narchive = shutil.make_archive('/content/mitra-predictor', 'zip', root_dir=active_path)\narchive_sha256 = sha256_file(archive)\nprint('✓ Predictor archive:', archive)\nprint('SHA-256:', archive_sha256)\n"""
    code = replace_once(code, old, new, "main manifest/digest export")
    set_src(export, code)

    recap = cell_with(nb, "## What a successful run proves", "markdown")
    text = src(recap)
    text = text.replace(
        "If every cell ran, this session has shown:",
        "If every cell ran in a clean runtime, this session has shown:",
        1,
    )
    set_src(recap, text)

    MAIN_NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_inference():
    nb = json.loads(INF_NB.read_text(encoding="utf-8"))

    opening = cell_with(nb, "# Mitra Classifier — Use an Exported Predictor", "markdown")
    text = src(opening)
    marker = "# Mitra Classifier — Use an Exported Predictor\n"
    addition = (
        marker
        + "\n**Notebook Specification:** DIMER Notebook Specification v1.0  \n"
        + "**Profile:** `ARTIFACT-INFERENCE`  \n"
        + "**Release status:** Candidate — static conformance checks are automated; clean Google Colab execution of this exact revision remains the release gate.\n"
    )
    text = replace_once(text, marker, addition, "inference opening profile")
    text = text.replace(
        "- Any Colab runtime; inference runs on CPU in seconds. About two minutes end to end, mostly the install.",
        "- Any compatible Colab runtime; completion time varies with dependency-cache state, archive size, and hardware, so no fixed runtime is promised.",
    )
    text += (
        "\n**Data handling:** the uploaded predictor ZIP and inference CSV are processed inside the Google Colab runtime. This notebook does not send inference rows to DIMER, Hugging Face, or another model API. Google Colab remains the hosting environment, so its data-handling policies still apply.\n"
    )
    set_src(opening, text)

    setup = cell_with(nb, "%pip install -q \"autogluon.tabular[mitra]==1.5.0\"", "code")
    code = src(setup)
    old = "%pip install -q \"autogluon.tabular[mitra]==1.5.0\"\n"
    new = locked_install_code(TUTORIALS / "requirements-inference.lock.txt", "/content/mitra-inference-requirements.lock.txt")
    code = replace_once(code, old, new, "inference locked install")
    code = code.replace(
        "print('AutoGluon:', AUTOGLUON_VERSION)\n",
        "print('Python:', sys.version.split()[0])\nprint('AutoGluon:', AUTOGLUON_VERSION)\n",
        1,
    )
    set_src(setup, code)

    setup_md = cell_with(nb, "## 1. Install the matching runtime", "markdown")
    text = src(setup_md)
    text += "\nThe dependency graph is installed from an exact-version lock generated into `tutorials/requirements-inference.lock.txt`.\n"
    set_src(setup_md, text)

    upload_md = cell_with(nb, "## 2. Upload and validate `mitra-predictor.zip`", "markdown")
    text = src(upload_md)
    text += (
        "\nThe archive must contain `tutorial_run_metadata.json` declaring artifact format `dimer-mitra-autogluon-predictor`, format version `1.0`, immutable base-model revision, runtime version, and required feature schema. Missing or malformed provenance is fatal before any predictor object is deserialized.\n"
    )
    set_src(upload_md, text)

    upload = cell_with(nb, "def safe_extract_zip", "code")
    code = src(upload)
    code = code.replace(
        "            member = Path(info.filename)\n",
        "            normalized = info.filename.replace('\\\\', '/')\n            member = Path(normalized)\n",
        1,
    )
    set_src(upload, code)

    load_md = cell_with(nb, "## 3. Load the predictor and inspect provenance", "markdown")
    text = src(load_md)
    text = text.replace(
        "If its recorded AutoGluon version differs from this runtime, the notebook stops rather than load a predictor across versions.",
        "Required provenance is validated before loading. If the artifact format/version, immutable model identity, feature schema, or recorded AutoGluon version is missing or incompatible, the notebook stops before deserialization.",
    )
    text += "\nNetwork fallback is explicitly disabled before `TabularPredictor.load(...)`; this notebook must reconstruct solely from the supplied artifact.\n"
    set_src(load_md, text)

    load = cell_with(nb, "METADATA_PATH = PREDICTOR_ROOT", "code")
    code = src(load)
    old = """METADATA_PATH = PREDICTOR_ROOT / 'tutorial_run_metadata.json'\nrun_metadata = {}\nif METADATA_PATH.exists():\n    run_metadata = json.loads(METADATA_PATH.read_text())\n    recorded_ag = run_metadata.get('autogluon_version')\n    if recorded_ag and recorded_ag != AUTOGLUON_VERSION:\n        raise RuntimeError(\n            f'Predictor was exported with AutoGluon {recorded_ag}, but this runtime has {AUTOGLUON_VERSION}. '\n            'Use the recorded version for best compatibility.'\n        )\nelse:\n    print('⚠ tutorial_run_metadata.json not found; continuing with predictor-internal metadata.')\n\npredictor = TabularPredictor.load(str(PREDICTOR_ROOT))\n"""
    new = """METADATA_PATH = PREDICTOR_ROOT / 'tutorial_run_metadata.json'\nif not METADATA_PATH.exists():\n    raise RuntimeError('Required tutorial_run_metadata.json is missing; refusing to deserialize an artifact with unknown provenance.')\nrun_metadata = json.loads(METADATA_PATH.read_text())\nrequired_provenance = [\n    'artifact_format',\n    'artifact_format_version',\n    'base_model',\n    'base_model_revision',\n    'autogluon_version',\n    'features',\n]\nmissing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\nif run_metadata['artifact_format'] != 'dimer-mitra-autogluon-predictor' or run_metadata['artifact_format_version'] != '1.0':\n    raise RuntimeError(\n        f\"Unsupported predictor artifact {run_metadata['artifact_format']!r} version {run_metadata['artifact_format_version']!r}.\"\n    )\nrecorded_ag = run_metadata['autogluon_version']\nif recorded_ag != AUTOGLUON_VERSION:\n    raise RuntimeError(\n        f'Predictor was exported with AutoGluon {recorded_ag}, but this runtime has {AUTOGLUON_VERSION}. '\n        'Use the recorded version for compatibility.'\n    )\n\nos.environ['HF_HUB_OFFLINE'] = '1'\nos.environ['TRANSFORMERS_OFFLINE'] = '1'\nos.environ['HF_DATASETS_OFFLINE'] = '1'\nprint('✓ Required provenance validated; network/model fallback disabled before deserialization.')\n\npredictor = TabularPredictor.load(str(PREDICTOR_ROOT))\n"""
    code = replace_once(code, old, new, "inference required provenance/offline")
    set_src(load, code)

    infer_md = cell_with(nb, "## 4. Upload new rows for inference", "markdown")
    text = src(infer_md)
    text += "\nThe notebook uses only the feature schema recorded in the predictor provenance; it does not infer a new schema or fit preprocessing on the inference rows.\n"
    set_src(infer_md, text)

    predict_md = cell_with(nb, "## 5. Predict and download `predictions.csv`", "markdown")
    text = src(predict_md)
    text = text.replace(
        "**Reading the probabilities:** they are the model's calibrated-ish confidence per class; the `prediction` is the argmax. For a cost-sensitive decision, threshold the probability of the costly class instead of using the argmax.",
        "**Reading the probabilities:** these are the predictor's per-class probabilities; calibration for your deployment population has not been established by this notebook. The `prediction` is the class with the highest returned probability (argmax). For a cost-sensitive decision, evaluate an application-specific threshold on held-out data rather than assuming argmax is optimal.",
    )
    set_src(predict_md, text)

    ai_cell = cell_with(nb, "## AI use and provenance", "markdown")
    idx = nb["cells"].index(ai_cell)
    interpretation = {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "## What a successful artifact-inference run proves — and does not prove\n\n"
            "A successful run proves that the supplied artifact passed the notebook's archive/provenance gates, loaded under the declared compatible runtime without network fallback, accepted the required feature schema, and produced class predictions plus per-class probabilities for the supplied rows.\n\n"
            "It does **not** establish predictive quality, probability calibration, fairness, robustness to distribution shift, or suitability for a consequential deployment. Those claims require labelled evaluation data from the intended population and the relevant governance review.\n"
        ),
    }
    nb["cells"].insert(idx, interpretation)

    INF_NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_readme():
    text = README.read_text(encoding="utf-8")
    anchor = "There are now two standalone Colab workflows:\n"
    registry = """There are now two standalone Colab workflows. Both are aligned to **DIMER Notebook Specification v1.0** and declare their normative profile explicitly.\n\n| Notebook | Profile | Spec | Release status |\n|---|---|---|---|\n| `mitra_classifier_colab.ipynb` | `E2E` | v1.0 | Candidate — static checks enforced; clean Colab execution of the release revision pending |\n| `mitra_classifier_predictor_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | v1.0 | Candidate — static checks enforced; clean Colab execution of the release revision pending |\n\nThe exact dependency graphs used by the notebooks are committed as `requirements-colab.lock.txt` and `requirements-inference.lock.txt`. Release-grade status requires clean target-runtime execution evidence for the exact release revision; static CI alone is not execution evidence.\n"""
    text = replace_once(text, anchor, registry, "tutorial README registry")
    text = text.replace(
        "- computes the uploaded archive's SHA-256 for provenance;",
        "- computes the uploaded archive's SHA-256 and verifies it when an expected digest is supplied;",
    )
    README.write_text(text, encoding="utf-8")


def patch_validator():
    text = VALIDATOR.read_text(encoding="utf-8")
    text = text.replace(
        '        "GPT-5.6 Sol High",\n',
        '        "GPT-5.6 Sol High",\n        "**Profile:** `E2E`",\n        "DIMER Notebook Specification v1.0",\n        "requirements-colab.lock.txt",\n        "artifact_format_version",\n        "artifact-manifest.json",\n        "SHA-256:",\n        "Data disclosure warning",\n        "calibration itself is not established",\n        "Unsafe archive member path",\n        "Symlink entries are not allowed",\n',
        1,
    )
    marker = '        "provenance, not sign-off",\n    ):\n        require(required in text, f"inference tutorial missing required marker: {required}")\n'
    replacement = '        "provenance, not sign-off",\n        "**Profile:** `ARTIFACT-INFERENCE`",\n        "DIMER Notebook Specification v1.0",\n        "requirements-inference.lock.txt",\n        "artifact_format_version",\n        "Required provenance validated",\n        "HF_HUB_OFFLINE",\n        "calibration for your deployment population has not been established",\n        "What a successful artifact-inference run proves",\n    ):\n        require(required in text, f"inference tutorial missing required marker: {required}")\n'
    if marker not in text:
        raise RuntimeError("validator inference required-marker block not found")
    text = text.replace(marker, replacement, 1)

    extra = """\n\ndef validate_lockfile(path: Path) -> None:\n    require(path.exists(), f\"missing notebook lockfile: {path}\")\n    for line in path.read_text(encoding='utf-8').splitlines():\n        stripped = line.strip()\n        if not stripped or stripped.startswith('#') or stripped.startswith('--') or stripped.startswith('    #'):\n            continue\n        if stripped.startswith('    --hash='):\n            continue\n        token = stripped.split('\\\\', 1)[0].strip()\n        require('==' in token, f\"unlocked requirement in {path.name}: {stripped}\")\n\n"""
    insert_at = text.find("\ndef validate_training_tutorial()")
    if insert_at < 0:
        raise RuntimeError("validator training function anchor not found")
    text = text[:insert_at] + extra + text[insert_at:]

    tail = 'if __name__ == "__main__":\n    validate_training_tutorial()\n    validate_inference_tutorial()\n    print("Standalone Colab tutorial checks passed.")\n'
    if tail not in text:
        raise RuntimeError("validator __main__ block not found")
    new_tail = 'if __name__ == "__main__":\n    validate_lockfile(ROOT / "tutorials" / "requirements-colab.lock.txt")\n    validate_lockfile(ROOT / "tutorials" / "requirements-inference.lock.txt")\n    validate_training_tutorial()\n    validate_inference_tutorial()\n    print("Standalone Colab tutorial checks passed.")\n'
    text = text.replace(tail, new_tail, 1)
    VALIDATOR.write_text(text, encoding="utf-8")


def patch_release_checklist():
    text = RELEASE.read_text(encoding="utf-8")
    stale = "- [x] Build → offline Mitra load → fine-tune → save → reload → predict, exercised by `.github/workflows/integration.yml` (manual/nightly GPU) and run live on the 5070 Ti 2026-08-19 (`problemType=multiclass`)."
    replacement = "- [x] Build → offline Mitra load → fine-tune → save → reload → predict was exercised live on the 5070 Ti on 2026-08-19 (`problemType=multiclass`). The repository does not currently contain `.github/workflows/integration.yml`; current notebook CI is static and must not be cited as clean-runtime execution evidence."
    if stale in text:
        text = text.replace(stale, replacement, 1)
    RELEASE.write_text(text, encoding="utf-8")


def main():
    compile_locks()
    patch_main()
    patch_inference()
    patch_readme()
    patch_validator()
    patch_release_checklist()
    print("Notebook Spec v1 migration applied.")


if __name__ == "__main__":
    main()
