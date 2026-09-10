from pathlib import Path
import json
import re

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
step2_markdown = '''## 2. Acquire, verify, and lock the checkpoint

A model file is code you are about to run and weights you are about to trust. This step makes both claims checkable:

- **DIMER files** uploads the two checkpoint files supported by this repository's current DIMER deployment contract: `model.safetensors` and `config.json`. Their bytes are checked against this notebook release's pinned SHA-256 values. The current DIMER contract does **not** define a self-describing package manifest, so this path does not claim Notebook Spec MOD7 package-manifest provenance; that remains an explicit Candidate-status gap until the platform defines such a producer contract.
- **Pinned upstream** fetches both files from the exact AutoGluon revision `c425e9fa…` on the Hugging Face Hub. A revision is an immutable commit; a model *name* is a branch that can change.

Both paths verify the same pinned bytes, then stage them into an isolated Hugging Face cache and mark it offline. Finally the notebook asks Hugging Face to resolve the model *exactly as AutoGluon will* and refuses to continue unless both resolved files come from that verified snapshot and still match the expected digests.

If the lock check fails, use *Runtime ▸ Restart session* and run from Step 1 downward. Network requests in the pinned-upstream mode use a finite timeout so outages fail clearly instead of hanging.

**What to look for:** two verified digest lines with prefixes `e06a055e91a3…` and `2c96c24dd25f…`, then `✓ Hugging Face resolver locked …`.

**Remote-code boundary:** this tutorial does not enable Hugging Face `trust_remote_code`; execution uses the installed AutoGluon implementation plus the verified `safetensors` weights and pinned configuration.

For a DIMER-file upload, provenance is anchored by this notebook release's immutable model revision and pinned file digests. Because the platform contract currently supplies files rather than a self-describing package, **MOD7 remains pending**; the notebook does not manufacture or infer an embedded DIMER package manifest. A failed or incomplete DIMER-file upload never falls back to upstream acquisition.
'''
step2_done = False
for cell, src in list(sources(nb)):
    if '## 2. Acquire, verify, and lock the checkpoint' in src:
        cell['source'] = step2_markdown
        step2_done = True
        break
if not step2_done:
    raise RuntimeError('E2E Step 2 markdown cell not found')

code_done = False
for cell, src in list(sources(nb)):
    if "MODEL_ID = 'autogluon/mitra-classifier'" in src and 'MODEL_SOURCE =' in src:
        # Remove any package-only constants between pinned config digest and network timeout.
        m = re.search(
            r"(EXPECTED_CONFIG_SHA256 = '[0-9a-f]{64}'\\n)(.*?)(NETWORK_TIMEOUT_SECONDS = 30\\n)",
            src,
            flags=re.S,
        )
        if not m:
            raise RuntimeError('could not locate model constants block')
        src = src[:m.start()] + m.group(1) + m.group(3) + src[m.end():]

        # Replace whatever DIMER helper currently exists with the actual two-file contract.
        fn = re.search(r'def load_dimer_[A-Za-z0-9_]*\(.*?\n(?=def install_offline_snapshot\()', src, flags=re.S)
        if not fn:
            raise RuntimeError('could not locate current DIMER loader function')
        new_fn = '''def load_dimer_files(weights_dest, config_dest):
    from google.colab import files

    MAX_DIMER_FILE_BYTES = 1024 ** 3
    uploaded = files.upload()
    expected_names = {'model.safetensors', 'config.json'}
    if set(uploaded) != expected_names:
        raise RuntimeError(
            'DIMER files mode requires exactly model.safetensors and config.json. '
            f'Missing={sorted(expected_names - set(uploaded))}; unexpected={sorted(set(uploaded) - expected_names)}'
        )
    for name, payload in uploaded.items():
        if len(payload) > MAX_DIMER_FILE_BYTES:
            raise RuntimeError(f'{name} exceeds the 1 GiB per-file safety ceiling.')
    weights_dest.write_bytes(uploaded['model.safetensors'])
    config_dest.write_bytes(uploaded['config.json'])
    verify(weights_dest, EXPECTED_WEIGHTS_SHA256, 'DIMER model.safetensors')
    verify(config_dest, EXPECTED_CONFIG_SHA256, 'DIMER config.json')
    print(
        '✓ DIMER checkpoint files match the notebook release digests:',
        MODEL_ID,
        PINNED_REVISION,
    )
    print('ℹ Current DIMER producer contract has no embedded package manifest; MOD7 remains pending for this source path.')

'''
        src = src[:fn.start()] + new_fn + src[fn.end():]
        src = re.sub(
            r"MODEL_SOURCE = 'Pinned upstream'\s*# @param \[[^\n]+\]",
            "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER files', 'Pinned upstream']",
            src,
            count=1,
        )
        src = re.sub(
            r"if MODEL_SOURCE == 'DIMER[^']*':\n\s+load_dimer_[A-Za-z0-9_]+\([^\n]+\)",
            "if MODEL_SOURCE == 'DIMER files':\n    load_dimer_files(weights_path, config_path)",
            src,
            count=1,
        )
        cell['source'] = src.splitlines(keepends=True)
        code_done = True
        break
if not code_done:
    raise RuntimeError('E2E model acquisition code cell not found')
nb_save(E2E, nb)

# --- Companion: stronger feature-schema validation + post-load equality check + stale wording. ---
nb = nb_load(INF)
inf_done = False
for cell, src in list(sources(nb)):
    if "required_provenance = [" in src and "TabularPredictor.load" in src:
        needle = "missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\n"
        if 'features must be a non-empty list of unique, non-blank strings' not in src:
            replacement = needle + "features = run_metadata.get('features')\nif (\n    not isinstance(features, list)\n    or not features\n    or any(not isinstance(name, str) or not name.strip() for name in features)\n    or len(set(features)) != len(features)\n):\n    raise RuntimeError('Predictor provenance features must be a non-empty list of unique, non-blank strings.')\nif run_metadata.get('target_column') in set(features):\n    raise RuntimeError('Predictor provenance target_column must not also appear in features.')\n"
            src = replace_once(src, needle, replacement, 'feature schema pre-load validation')
        old_block = "if run_metadata.get('features'):\n    FEATURE_COLUMNS = list(run_metadata['features'])\nelse:\n    feature_metadata = getattr(predictor, 'feature_metadata_in', None)\n    if feature_metadata is None:\n        raise RuntimeError('Could not determine required input feature columns from predictor metadata.')\n    FEATURE_COLUMNS = list(feature_metadata.get_features())\n"
        if old_block in src:
            new_block = "FEATURE_COLUMNS = list(features)\nfeature_metadata = getattr(predictor, 'feature_metadata_in', None)\nif feature_metadata is None:\n    raise RuntimeError('Loaded predictor does not expose feature_metadata_in for schema verification.')\nloaded_features = list(feature_metadata.get_features())\nif loaded_features != FEATURE_COLUMNS:\n    raise RuntimeError(\n        'Predictor provenance feature schema does not match the loaded predictor. '\n        f'Recorded={FEATURE_COLUMNS}; loaded={loaded_features}'\n    )\nprint('✓ Predictor feature schema matches required provenance.')\n"
            src = src.replace(old_block, new_block, 1)
        elif 'Predictor feature schema matches required provenance' not in src:
            raise RuntimeError('post-load schema block not found')
        cell['source'] = src.splitlines(keepends=True)
        inf_done = True
    if 'Predictor provenance: read from `tutorial_run_metadata.json` when available' in src:
        cell['source'] = src.replace(
            'Predictor provenance: read from `tutorial_run_metadata.json` when available',
            'Predictor provenance: required and validated from `tutorial_run_metadata.json` before deserialization',
        )
if not inf_done:
    raise RuntimeError('companion provenance/load cell not found')
nb_save(INF, nb)

# --- README: align descriptions to the actual current platform contract. ---
text = README.read_text(encoding='utf-8')
# Replace any paragraph beginning with a DIMER MODEL_SOURCE package claim.
text = re.sub(
    r"For `MODEL_SOURCE = 'DIMER[^\n]*\n(?:\n|$)",
    "The repository's current DIMER deployment contract exposes an uploaded checkpoint as `model.safetensors` + `config.json`; it does not define a self-describing package-manifest producer. Accordingly, `MODEL_SOURCE = 'DIMER files'` accepts exactly those two files, verifies both against the pinned release digests, and explicitly records Notebook Spec MOD7 package-manifest provenance as pending. The separate `Pinned upstream` source remains fully pinned by immutable revision and digests.\n\n",
    text,
    count=1,
)
text = text.replace('self-describing offline model package', 'two-file checkpoint')
text = text.replace('repository-defined notebook package built from the current DIMER checkpoint pair', 'DIMER checkpoint pair')
text = text.replace('- DIMER ZIP upload or pinned-upstream checkpoint source selection;', '- DIMER two-file checkpoint upload or pinned-upstream checkpoint source selection;')
text = text.replace('DIMER self-describing offline model package OR pinned upstream checkpoint', 'DIMER model.safetensors + config.json OR pinned upstream checkpoint')
text = re.sub(
    r'- Distributed DIMER artifact: .*',
    "- Distributed DIMER artifact: checkpoint files (`model.safetensors` + `config.json`) under the repository's current deployment contract; no platform package-manifest producer is claimed",
    text,
    count=1,
)
README.write_text(text, encoding='utf-8')

# --- DEPLOYMENT: reconcile stale integration-workflow claims with actual workflow directory. ---
text = DEPLOY.read_text(encoding='utf-8')
text = text.replace('Two GitHub Actions workflows guard the repo:', 'One GitHub Actions workflow currently guards the repo:')
text = re.sub(
    r"- \*\*`integration`\*\* \(manual `workflow_dispatch`.*?nightly run\.\n",
    "- A historical/local GPU integration exercise covered image build → pinned offline Mitra load →\n  fine-tune → save → reload → predict, but **`.github/workflows/integration.yml` is not present in\n  the repository today**. Do not treat that historical exercise as an active CI workflow. If GPU\n  integration automation is restored, document the actual workflow and runner contract here.\n",
    text,
    flags=re.S,
    count=1,
)
text = text.replace(
    'the `integration` workflow exercises the save → reload → predict\nround-trip on a GPU runner, so the artifact is verified to reload and serve predictions outside\nthe training process.',
    'a historical/local GPU exercise covered the save → reload → predict\nround-trip, but there is currently no `integration.yml` workflow providing recurring CI evidence.',
)
DEPLOY.write_text(text, encoding='utf-8')

# --- Static validator: make the remediation load-bearing. ---
text = VALIDATOR.read_text(encoding='utf-8')
for marker in [
    '"PACKAGE_MANIFEST_FILENAME",\n',
    '        "dimer-model-manifest.json",\n',
    '        "load_dimer_package",\n',
    '        "manifest_version",\n',
    '        "legacy weights-only ZIPs are refused",\n',
]:
    text = text.replace(marker, '')
if '"load_dimer_files"' not in text:
    text = text.replace('        "Remote-code boundary",\n', '        "Remote-code boundary",\n        "load_dimer_files",\n        "MAX_DIMER_FILE_BYTES",\n        "MOD7 remains pending",\n')
if 'features must be a non-empty list of unique, non-blank strings' not in text:
    text = text.replace('        "Required provenance validated",\n', '        "Required provenance validated",\n        "features must be a non-empty list of unique, non-blank strings",\n        "Predictor feature schema matches required provenance",\n')
if "DEPLOYMENT must not claim an active integration workflow" not in text:
    anchor = "def validate_docs() -> None:\n"
    idx = text.index(anchor) + len(anchor)
    insert = "    deploy_text = (ROOT / 'DEPLOYMENT.md').read_text(encoding='utf-8')\n    require('.github/workflows/integration.yml` is not present' in deploy_text, 'DEPLOYMENT must not claim an active integration workflow')\n    require('Two GitHub Actions workflows guard the repo' not in deploy_text, 'DEPLOYMENT has stale two-workflow claim')\n\n"
    text = text[:idx] + insert + text[idx:]
VALIDATOR.write_text(text, encoding='utf-8')

print('round-2 remediation applied')
