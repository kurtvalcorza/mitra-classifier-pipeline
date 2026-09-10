from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / 'tutorials' / 'mitra_classifier_colab.ipynb'
INF = ROOT / 'tutorials' / 'mitra_classifier_predictor_inference_colab.ipynb'
README = ROOT / 'tutorials' / 'README.md'
DEPLOY = ROOT / 'DEPLOYMENT.md'
VALIDATOR = ROOT / 'scripts' / 'validate_colab_tutorial.py'


def load_nb(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_nb(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def replace_in_nb(nb, old, new, *, count=1):
    hits = 0
    for cell in nb['cells']:
        src = cell.get('source', '')
        if isinstance(src, list):
            joined = ''.join(src)
            if old in joined:
                joined2 = joined.replace(old, new, count if hits == 0 else 0)
                cell['source'] = joined2.splitlines(keepends=True)
                hits += 1
        elif old in src:
            cell['source'] = src.replace(old, new, count if hits == 0 else 0)
            hits += 1
    if hits == 0:
        raise RuntimeError(f'Notebook replacement anchor not found: {old[:120]!r}')


def replace_text(path, old, new, *, count=1):
    text = path.read_text(encoding='utf-8')
    if old not in text:
        raise RuntimeError(f'Text replacement anchor not found in {path}: {old[:120]!r}')
    path.write_text(text.replace(old, new, count), encoding='utf-8')


e2e = load_nb(E2E)
inf = load_nb(INF)

# 1) Stop asserting that the current DIMER platform already emits the new manifest package.
replace_in_nb(
    e2e,
    "This notebook takes you from the pinned Mitra checkpoint (from the DIMER Model Repository, or the identical upstream release) to a verified, reusable predictor ZIP",
    "This notebook takes you from the pinned Mitra checkpoint (from the immutable upstream release, or from a repository-defined self-describing offline package carrying the identical bytes) to a verified, reusable predictor ZIP",
)
replace_in_nb(
    e2e,
    "- **DIMER ZIP** uploads a self-describing offline ZIP distributed through the DIMER Model Repository. Notebook Spec v1.0 packages must contain `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest is validated before either model file is staged; legacy weights-only ZIPs are refused rather than silently completed from the network.",
    "- **Verified offline ZIP** uploads a self-describing package that follows this repository's tutorial package contract. It must contain `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest is validated before either model file is staged; incomplete/legacy weights-only ZIPs are refused rather than silently completed from the network. **This is not a claim that the current DIMER platform export already emits this manifest format.** The repository's deployment path currently documents uploaded checkpoints as `model.safetensors` + `config.json`; until the platform publishes or adopts the manifest package contract, MOD7 conformance is not claimed for a DIMER-produced ZIP.",
)
replace_in_nb(
    e2e,
    "For a DIMER upload, the notebook reads and validates the manifest **from the uploaded ZIP itself**: manifest version, exact model identifier, immutable revision, exact package file set, and SHA-256 for both weights and configuration. A failed or incomplete package never falls back to upstream acquisition.",
    "For a verified offline upload, the notebook reads and validates the manifest **from the uploaded ZIP itself**: manifest version, exact model identifier, immutable revision, exact package file set, and SHA-256 for both weights and configuration. A failed or incomplete package never falls back to upstream acquisition. **MOD7 status:** repository-local offline package validation is implemented, but current DIMER-platform package production/adoption remains unverified and is not represented as release evidence.",
)
replace_in_nb(e2e, "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']", "MODEL_SOURCE = 'Pinned upstream'  # @param ['Verified offline ZIP', 'Pinned upstream']")
replace_in_nb(e2e, "if MODEL_SOURCE == 'DIMER ZIP':", "if MODEL_SOURCE == 'Verified offline ZIP':")
replace_in_nb(e2e, "Upload exactly one DIMER offline model ZIP.", "Upload exactly one verified offline model ZIP.")
replace_in_nb(e2e, "DIMER ZIP mode requires a .zip package with an embedded manifest.", "Verified offline ZIP mode requires a .zip package with an embedded manifest.")
replace_in_nb(e2e, "DIMER package entries must be at the archive root", "Offline package entries must be at the archive root")
replace_in_nb(e2e, "Duplicate DIMER package member", "Duplicate offline package member")
replace_in_nb(e2e, "DIMER package file set mismatch", "Offline package file set mismatch")
replace_in_nb(e2e, "DIMER package manifest is not valid UTF-8 JSON.", "Offline package manifest is not valid UTF-8 JSON.")
replace_in_nb(e2e, "DIMER package manifest must be a JSON object.", "Offline package manifest must be a JSON object.")
replace_in_nb(e2e, "Unsupported DIMER package manifest version", "Unsupported offline package manifest version")
replace_in_nb(e2e, "DIMER package model identity mismatch", "Offline package model identity mismatch")
replace_in_nb(e2e, "DIMER package revision mismatch", "Offline package revision mismatch")
replace_in_nb(e2e, "DIMER package manifest must enumerate exactly model.safetensors and config.json.", "Offline package manifest must enumerate exactly model.safetensors and config.json.")
replace_in_nb(e2e, "DIMER package manifest has invalid SHA-256", "Offline package manifest has invalid SHA-256")
replace_in_nb(e2e, "DIMER package manifest digest", "Offline package manifest digest")
replace_in_nb(e2e, "DIMER package payload digest mismatch", "Offline package payload digest mismatch")
replace_in_nb(e2e, "✓ DIMER package manifest, identity, revision, exact file set, and payload digests verified:", "✓ Offline package manifest, identity, revision, exact file set, and payload digests verified:")

# 2) Bound offline package decompression/memory before z.read().
replace_in_nb(
    e2e,
    "PACKAGE_MANIFEST_VERSION = '1.0'\nNETWORK_TIMEOUT_SECONDS = 30",
    "PACKAGE_MANIFEST_VERSION = '1.0'\nMAX_OFFLINE_PACKAGE_EXPANDED_BYTES = 1024 ** 3  # 1 GiB total declared expansion ceiling\nMAX_OFFLINE_METADATA_BYTES = 1024 ** 2  # 1 MiB each for manifest/config\nNETWORK_TIMEOUT_SECONDS = 30",
)
replace_in_nb(
    e2e,
    "with zipfile.ZipFile(package_path) as z:\n        by_name = {}\n        for info in z.infolist():",
    "with zipfile.ZipFile(package_path) as z:\n        by_name = {}\n        expanded_bytes = 0\n        for info in z.infolist():",
)
replace_in_nb(
    e2e,
    "            if info.is_dir():\n                continue\n            if '\\\\' in info.filename:",
    "            if info.is_dir():\n                continue\n            expanded_bytes += int(info.file_size)\n            if expanded_bytes > MAX_OFFLINE_PACKAGE_EXPANDED_BYTES:\n                raise RuntimeError('Offline model package exceeds the 1 GiB expanded-size safety limit.')\n            if info.filename in {PACKAGE_MANIFEST_FILENAME, 'config.json'} and info.file_size > MAX_OFFLINE_METADATA_BYTES:\n                raise RuntimeError(f'Offline package metadata member is unexpectedly large: {info.filename!r}.')\n            if '\\\\' in info.filename:",
)

# 3) Strengthen inference provenance feature schema pre-load and bind it to loaded predictor post-load.
replace_in_nb(
    inf,
    "missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')",
    "missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\nrecorded_features = run_metadata['features']\nif (\n    not isinstance(recorded_features, list)\n    or not recorded_features\n    or any(not isinstance(name, str) or not name.strip() for name in recorded_features)\n    or len(set(recorded_features)) != len(recorded_features)\n):\n    raise RuntimeError('Predictor provenance feature schema must be a non-empty list of unique, non-blank strings.')\nrecorded_target = run_metadata.get('target_column')\nif recorded_target is not None and (not isinstance(recorded_target, str) or not recorded_target.strip()):\n    raise RuntimeError('Predictor provenance target_column must be a non-blank string when present.')\nif recorded_target in recorded_features:\n    raise RuntimeError('Predictor provenance feature schema must not include the target column.')",
)
replace_in_nb(
    inf,
    "if run_metadata.get('features'):\n    FEATURE_COLUMNS = list(run_metadata['features'])\nelse:\n    feature_metadata = getattr(predictor, 'feature_metadata_in', None)\n    if feature_metadata is None:\n        raise RuntimeError('Could not determine required input feature columns from predictor metadata.')\n    FEATURE_COLUMNS = list(feature_metadata.get_features())\n\nTARGET_COLUMN = run_metadata.get('target_column') or getattr(predictor, 'label', None)",
    "FEATURE_COLUMNS = list(recorded_features)\nfeature_metadata = getattr(predictor, 'feature_metadata_in', None)\nif feature_metadata is None:\n    raise RuntimeError('Loaded predictor does not expose input feature metadata for schema verification.')\nloaded_features = list(feature_metadata.get_features())\nif loaded_features != FEATURE_COLUMNS:\n    raise RuntimeError(\n        'Predictor provenance feature schema does not match the loaded predictor schema. '\n        f'Recorded={FEATURE_COLUMNS}; loaded={loaded_features}'\n    )\n\nTARGET_COLUMN = recorded_target or getattr(predictor, 'label', None)\nif recorded_target is not None and getattr(predictor, 'label', None) != recorded_target:\n    raise RuntimeError(\n        f'Predictor provenance target column {recorded_target!r} does not match loaded predictor label {getattr(predictor, "label", None)!r}.'\n    )",
)
replace_in_nb(inf, "- Predictor provenance: read from `tutorial_run_metadata.json` when available", "- Predictor provenance: required from `tutorial_run_metadata.json` and validated before deserialization")

save_nb(E2E, e2e)
save_nb(INF, inf)

# 4) README: narrow DIMER claim and document repository-local package contract / MOD7 status.
replace_text(
    README,
    "For `MODEL_SOURCE = 'DIMER ZIP'`, Notebook Spec v1.0 expects a self-describing offline package containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. Legacy weights-only DIMER ZIPs are refused rather than completed from the network.",
    "For `MODEL_SOURCE = 'Verified offline ZIP'`, this repository defines a tutorial-only self-describing package contract containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. The current DIMER deployment documentation in this repository describes uploaded checkpoints as `model.safetensors` + `config.json`; it does not establish that the platform emits this manifest package. Accordingly, MOD7 is implemented for the repository-local offline package contract but is **not claimed as verified for a DIMER-produced ZIP** until the platform adopts/publishes that producer contract.",
)
replace_text(README, "Users can download the self-describing offline model package from DIMER and run the notebook independently in Google Colab.", "Users can run the repository-defined self-describing offline package independently in Google Colab when such a package is provided by a trusted source.")
replace_text(README, "- DIMER ZIP upload or pinned-upstream checkpoint source selection;", "- repository-defined verified offline ZIP upload or pinned-upstream checkpoint source selection;")
replace_text(README, "DIMER self-describing offline model package OR pinned upstream checkpoint", "verified self-describing offline model package OR pinned upstream checkpoint")
replace_text(README, "- Distributed DIMER artifact: self-describing offline package containing the pinned model weights, configuration, and package manifest", "- Offline tutorial package contract: repository-defined manifest + pinned weights/config; current DIMER-platform production of this format is not yet verified")

# 5) Reconcile DEPLOYMENT.md with actual workflows.
deploy = DEPLOY.read_text(encoding='utf-8')
deploy = deploy.replace(
    "Two GitHub Actions workflows guard the repo:\n\n- **`ci`** (every push/PR) — compiles the deployable sources, runs the unit suite, and enforces\n  the shared dataset-resolution block is byte-identical across the validator/finetuner copies and\n  matches the cross-repo pinned SHA (so this repo and the standalone deployment repos cannot\n  drift). It deliberately does **not** install AutoGluon.\n- **`integration`** (manual `workflow_dispatch`; nightly only when the repo variable\n  `ENABLE_NIGHTLY_GPU` is `true`) — needs a **self-hosted runner labelled `gpu`** with the NVIDIA\n  Container Toolkit. It builds the image, loads Mitra offline from the pinned weights, fine-tunes\n  a tiny model, saves it, reloads the `TabularPredictor`, and predicts — catching torch/base-image\n  drift, Mitra API changes, offline-weight failures, and seed/metric propagation that the unit\n  suite cannot. Attach a GPU runner and set `ENABLE_NIGHTLY_GPU` to turn on the nightly run.\n",
    "The repository currently has one code/test workflow relevant to the pipeline, **`ci`** (every push/PR). It compiles the deployable sources, runs the unit suite, and enforces the shared dataset-resolution block is byte-identical across the validator/finetuner copies and matches the cross-repo pinned SHA. It deliberately does **not** install AutoGluon or execute a real GPU integration path.\n\nHistorical documentation referred to an **`integration`** workflow, but `.github/workflows/integration.yml` is not present in the current repository. The 2026-08-19 5070 Ti save → reload → predict run remains historical execution evidence only; it must not be represented as an active workflow or as clean Colab notebook execution evidence.\n",
)
deploy = deploy.replace(
    "This pipeline's artifact is an AutoGluon\n`TabularPredictor` directory; the `integration` workflow exercises the save → reload → predict\nround-trip on a GPU runner, so the artifact is verified to reload and serve predictions outside\nthe training process.",
    "This pipeline's artifact is an AutoGluon\n`TabularPredictor` directory. A 2026-08-19 5070 Ti run historically exercised the save → reload →\npredict round-trip outside the training process; there is no current `integration.yml` workflow\nin this repository providing ongoing automated evidence.",
)
DEPLOY.write_text(deploy, encoding='utf-8')

# 6) Strengthen static validator so the new reviewer fixes cannot regress silently.
validator = VALIDATOR.read_text(encoding='utf-8')
validator = validator.replace(
    '        "PACKAGE_MANIFEST_FILENAME",\n        "dimer-model-manifest.json",\n        "load_dimer_package",',
    '        "PACKAGE_MANIFEST_FILENAME",\n        "dimer-model-manifest.json",\n        "MAX_OFFLINE_PACKAGE_EXPANDED_BYTES",\n        "Verified offline ZIP",\n        "MOD7 status",\n        "load_dimer_package",',
)
validator = validator.replace(
    '        "This notebook is inference-only",\n',
    '        "This notebook is inference-only",\n        "feature schema must be a non-empty list of unique, non-blank strings",\n        "does not match the loaded predictor schema",\n',
)
validator = validator.replace(
    '    validate_docs()\n',
    "    validate_docs()\n    deployment_text = (ROOT / 'DEPLOYMENT.md').read_text(encoding='utf-8')\n    require('.github/workflows/integration.yml' not in deployment_text, 'DEPLOYMENT.md must not claim a current integration.yml workflow')\n    require('Two GitHub Actions workflows guard the repo' not in deployment_text, 'DEPLOYMENT.md workflow count is stale')\n    _, inference_text, _ = load_notebook(INFERENCE_NOTEBOOK)\n    require('read from `tutorial_run_metadata.json` when available' not in inference_text, 'inference notebook still describes required provenance as optional')\n",
)
VALIDATOR.write_text(validator, encoding='utf-8')

print('round-2 reviewer remediation applied')
