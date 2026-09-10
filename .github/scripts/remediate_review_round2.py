from pathlib import Path
import json

ROOT = Path('.')
E2E = ROOT / 'tutorials/mitra_classifier_colab.ipynb'
INF = ROOT / 'tutorials/mitra_classifier_predictor_inference_colab.ipynb'
README = ROOT / 'tutorials/README.md'
DEPLOY = ROOT / 'DEPLOYMENT.md'
VALIDATOR = ROOT / 'scripts/validate_colab_tutorial.py'


def nb_load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def nb_save(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def sources(nb):
    for cell in nb['cells']:
        src = cell.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        yield cell, src


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one match, found {count}')
    return text.replace(old, new, 1)


# --- E2E: align DIMER acquisition to the repository's real two-file contract. ---
nb = nb_load(E2E)
for cell, src in list(sources(nb)):
    if '## 2. Acquire, verify, and lock the checkpoint' in src:
        src = replace_once(
            src,
            "- **DIMER ZIP** uploads a self-describing offline ZIP distributed through the DIMER Model Repository. Notebook Spec v1.0 packages must contain `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest is validated before either model file is staged; legacy weights-only ZIPs are refused rather than silently completed from the network.",
            "- **DIMER files** uploads the two checkpoint files supported by this repository's current DIMER deployment contract: `model.safetensors` and `config.json`. Their bytes are checked against this notebook release's pinned SHA-256 values. The current DIMER contract does **not** define a self-describing package manifest, so this path does not claim Notebook Spec MOD7 package-manifest provenance; that remains an explicit Candidate-status gap until the platform defines such a producer contract.",
            'E2E DIMER description',
        )
        src = src.replace("For a DIMER upload, the notebook reads and validates the manifest **from the uploaded ZIP itself**: manifest version, exact model identifier, immutable revision, exact package file set, and SHA-256 for both weights and configuration. A failed or incomplete package never falls back to upstream acquisition.\n", "For a DIMER-file upload, provenance is anchored by this notebook release's immutable model revision and pinned file digests. Because the platform contract currently supplies files rather than a self-describing package, **MOD7 remains pending**; the notebook does not manufacture or infer an embedded DIMER package manifest. A failed or incomplete DIMER-file upload never falls back to upstream acquisition.\n")
        cell['source'] = src

for cell, src in list(sources(nb)):
    if "PACKAGE_MANIFEST_FILENAME = 'dimer-model-manifest.json'" in src:
        start = src.index("PACKAGE_MANIFEST_FILENAME = 'dimer-model-manifest.json'")
        src = src[:start] + src[src.index("NETWORK_TIMEOUT_SECONDS = 30", start):]
        fn_start = src.index('def load_dimer_package(')
        fn_end = src.index('\ndef install_offline_snapshot(', fn_start)
        new_fn = '''def load_dimer_files(weights_dest, config_dest):\n    from google.colab import files\n\n    MAX_DIMER_FILE_BYTES = 1024 ** 3\n    uploaded = files.upload()\n    expected_names = {'model.safetensors', 'config.json'}\n    if set(uploaded) != expected_names:\n        raise RuntimeError(\n            'DIMER files mode requires exactly model.safetensors and config.json. '\n            f'Missing={sorted(expected_names - set(uploaded))}; unexpected={sorted(set(uploaded) - expected_names)}'\n        )\n    for name, payload in uploaded.items():\n        if len(payload) > MAX_DIMER_FILE_BYTES:\n            raise RuntimeError(f'{name} exceeds the 1 GiB per-file safety ceiling.')\n    weights_dest.write_bytes(uploaded['model.safetensors'])\n    config_dest.write_bytes(uploaded['config.json'])\n    verify(weights_dest, EXPECTED_WEIGHTS_SHA256, 'DIMER model.safetensors')\n    verify(config_dest, EXPECTED_CONFIG_SHA256, 'DIMER config.json')\n    print(\n        '✓ DIMER checkpoint files match the notebook release digests:',\n        MODEL_ID,\n        PINNED_REVISION,\n    )\n    print('ℹ Current DIMER producer contract has no embedded package manifest; MOD7 remains pending for this source path.')\n\n'''
        src = src[:fn_start] + new_fn + src[fn_end+1:]
        src = src.replace("MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']", "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER files', 'Pinned upstream']")
        src = src.replace("if MODEL_SOURCE == 'DIMER ZIP':\n    load_dimer_package(weights_path, config_path)", "if MODEL_SOURCE == 'DIMER files':\n    load_dimer_files(weights_path, config_path)")
        cell['source'] = src.splitlines(keepends=True)

nb_save(E2E, nb)

# --- Companion: stronger feature-schema validation + post-load equality check + stale wording. ---
nb = nb_load(INF)
for cell, src in list(sources(nb)):
    if "required_provenance = [" in src and "TabularPredictor.load" in src:
        needle = "missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\n"
        replacement = needle + "features = run_metadata.get('features')\nif (\n    not isinstance(features, list)\n    or not features\n    or any(not isinstance(name, str) or not name.strip() for name in features)\n    or len(set(features)) != len(features)\n):\n    raise RuntimeError('Predictor provenance features must be a non-empty list of unique, non-blank strings.')\nif run_metadata.get('target_column') in set(features):\n    raise RuntimeError('Predictor provenance target_column must not also appear in features.')\n"
        src = replace_once(src, needle, replacement, 'feature schema pre-load validation')
        needle2 = "if predictor.problem_type not in {'binary', 'multiclass'}:\n    raise RuntimeError(f'Expected a classification predictor, but loaded problem_type={predictor.problem_type!r}.')\n\nif run_metadata.get('features'):\n    FEATURE_COLUMNS = list(run_metadata['features'])\nelse:\n    feature_metadata = getattr(predictor, 'feature_metadata_in', None)\n    if feature_metadata is None:\n        raise RuntimeError('Could not determine required input feature columns from predictor metadata.')\n    FEATURE_COLUMNS = list(feature_metadata.get_features())\n"
        replacement2 = "if predictor.problem_type not in {'binary', 'multiclass'}:\n    raise RuntimeError(f'Expected a classification predictor, but loaded problem_type={predictor.problem_type!r}.')\n\nFEATURE_COLUMNS = list(features)\nfeature_metadata = getattr(predictor, 'feature_metadata_in', None)\nif feature_metadata is None:\n    raise RuntimeError('Loaded predictor does not expose feature_metadata_in for schema verification.')\nloaded_features = list(feature_metadata.get_features())\nif loaded_features != FEATURE_COLUMNS:\n    raise RuntimeError(\n        'Predictor provenance feature schema does not match the loaded predictor. '\n        f'Recorded={FEATURE_COLUMNS}; loaded={loaded_features}'\n    )\nprint('✓ Predictor feature schema matches required provenance.')\n"
        src = replace_once(src, needle2, replacement2, 'post-load schema equality')
        cell['source'] = src.splitlines(keepends=True)
    if 'Predictor provenance: read from `tutorial_run_metadata.json` when available' in src:
        src = src.replace('Predictor provenance: read from `tutorial_run_metadata.json` when available', 'Predictor provenance: required and validated from `tutorial_run_metadata.json` before deserialization')
        cell['source'] = src
nb_save(INF, nb)

# --- README: stop claiming a platform package producer that is not established. ---
text = README.read_text(encoding='utf-8')
text = text.replace(
    "For `MODEL_SOURCE = 'DIMER ZIP'`, Notebook Spec v1.0 expects a self-describing offline package containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. Legacy weights-only DIMER ZIPs are refused rather than completed from the network.",
    "The repository's current DIMER deployment contract exposes an uploaded checkpoint as `model.safetensors` + `config.json`; it does not define a self-describing package-manifest producer. Accordingly, `MODEL_SOURCE = 'DIMER files'` accepts exactly those two files, verifies both against the pinned release digests, and explicitly records Notebook Spec MOD7 package-manifest provenance as pending. The separate `Pinned upstream` source remains fully pinned by immutable revision and digests.",
)
text = text.replace('Users can download the self-describing offline model package from DIMER and run the notebook independently in Google Colab.', 'Users can download the two checkpoint files supported by the current DIMER contract and run the notebook independently in Google Colab.')
text = text.replace('- DIMER ZIP upload or pinned-upstream checkpoint source selection;', '- DIMER two-file checkpoint upload or pinned-upstream checkpoint source selection;')
text = text.replace('DIMER self-describing offline model package OR pinned upstream checkpoint', 'DIMER model.safetensors + config.json OR pinned upstream checkpoint')
text = text.replace('- Distributed DIMER artifact: self-describing offline package containing the pinned model weights, configuration, and package manifest', '- Distributed DIMER artifact: checkpoint files (`model.safetensors` + `config.json`) under the repository\'s current deployment contract; no platform package-manifest producer is claimed')
README.write_text(text, encoding='utf-8')

# --- DEPLOYMENT: reconcile stale integration-workflow claims with actual workflow directory. ---
text = DEPLOY.read_text(encoding='utf-8')
text = text.replace('Two GitHub Actions workflows guard the repo:', 'One GitHub Actions workflow currently guards the repo:')
old = '''- **`integration`** (manual `workflow_dispatch`; nightly only when the repo variable\n  `ENABLE_NIGHTLY_GPU` is `true`) — needs a **self-hosted runner labelled `gpu`** with the NVIDIA\n  Container Toolkit. It builds the image, loads Mitra offline from the pinned weights, fine-tunes\n  a tiny model, saves it, reloads the `TabularPredictor`, and predicts — catching torch/base-image\n  drift, Mitra API changes, offline-weight failures, and seed/metric propagation that the unit\n  suite cannot. Attach a GPU runner and set `ENABLE_NIGHTLY_GPU` to turn on the nightly run.\n'''
new = '''- A historical/local GPU integration exercise covered image build → pinned offline Mitra load →\n  fine-tune → save → reload → predict, but **`.github/workflows/integration.yml` is not present in\n  the repository today**. Do not treat that historical exercise as an active CI workflow. If GPU\n  integration automation is restored, document the actual workflow and runner contract here.\n'''
text = replace_once(text, old, new, 'DEPLOYMENT integration block')
text = text.replace('the `integration` workflow exercises the save → reload → predict\nround-trip on a GPU runner, so the artifact is verified to reload and serve predictions outside\nthe training process.', 'a historical/local GPU exercise covered the save → reload → predict\nround-trip, but there is currently no `integration.yml` workflow providing recurring CI evidence.')
DEPLOY.write_text(text, encoding='utf-8')

# --- Static validator: make the remediation load-bearing. ---
text = VALIDATOR.read_text(encoding='utf-8')
text = text.replace('"PACKAGE_MANIFEST_FILENAME",\n        "dimer-model-manifest.json",\n        "load_dimer_package",\n        "manifest_version",\n        "legacy weights-only ZIPs are refused",', '"load_dimer_files",\n        "MAX_DIMER_FILE_BYTES",\n        "MOD7 remains pending",')
text = text.replace('"Required provenance validated",', '"Required provenance validated",\n        "features must be a non-empty list of unique, non-blank strings",\n        "Predictor feature schema matches required provenance",')
# Documentation invariants.
anchor = "def validate_docs() -> None:\n"
idx = text.index(anchor) + len(anchor)
insert = "    deploy_text = (ROOT / 'DEPLOYMENT.md').read_text(encoding='utf-8')\n    require('.github/workflows/integration.yml` is not present' in deploy_text, 'DEPLOYMENT must not claim an active integration workflow')\n    require('Two GitHub Actions workflows guard the repo' not in deploy_text, 'DEPLOYMENT has stale two-workflow claim')\n\n"
text = text[:idx] + insert + text[idx:]
VALIDATOR.write_text(text, encoding='utf-8')

print('round-2 remediation applied')
