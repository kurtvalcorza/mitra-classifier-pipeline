from pathlib import Path
import json
import re

ROOT = Path('.')
E2E = ROOT / 'tutorials/mitra_classifier_colab.ipynb'
INF = ROOT / 'tutorials/mitra_classifier_predictor_inference_colab.ipynb'
README = ROOT / 'tutorials/README.md'
DEPLOY = ROOT / 'DEPLOYMENT.md'
VALIDATOR = ROOT / 'scripts/validate_colab_tutorial.py'
CI = ROOT / '.github/workflows/ci.yml'


def load_nb(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_nb(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def src(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


# E2E source contract: use the two files the repository actually documents for DIMER.
nb = load_nb(E2E)
step2 = '''## 2. Acquire, verify, and lock the checkpoint

A model file is code you are about to run and weights you are about to trust. This step makes both claims checkable:

- **DIMER files** uploads the two checkpoint files supported by this repository's current DIMER deployment contract: `model.safetensors` and `config.json`. Their bytes are checked against this notebook release's pinned SHA-256 values. The current DIMER contract does **not** define a self-describing package manifest, so this path does not claim Notebook Spec MOD7 package-manifest provenance; that remains an explicit Candidate-status gap until the platform defines such a producer contract.
- **Pinned upstream** fetches both files from the exact AutoGluon revision `c425e9fa…` on the Hugging Face Hub. A revision is an immutable commit; a model *name* is a branch that can change.

Both paths verify the same pinned bytes, then stage them into an isolated Hugging Face cache and mark it offline. Finally the notebook asks Hugging Face to resolve the model exactly as AutoGluon will and refuses to continue unless both resolved files come from that verified snapshot and still match the expected digests.

If the lock check fails, use *Runtime ▸ Restart session* and run from Step 1 downward. Network requests in pinned-upstream mode use a finite timeout so outages fail clearly instead of hanging.

**What to look for:** two verified digest lines with prefixes `e06a055e91a3…` and `2c96c24dd25f…`, then `✓ Hugging Face resolver locked …`.

**Remote-code boundary:** this tutorial does not enable Hugging Face `trust_remote_code`; execution uses the installed AutoGluon implementation plus the verified `safetensors` weights and pinned configuration.

For a DIMER-file upload, provenance is anchored by this notebook release's immutable model revision and pinned file digests. Because the platform contract currently supplies files rather than a self-describing package, **MOD7 remains pending**; the notebook does not manufacture or infer an embedded DIMER package manifest. A failed or incomplete DIMER-file upload never falls back to upstream acquisition.
'''
found = False
for cell in nb['cells']:
    s = src(cell)
    if '## 2. Acquire, verify, and lock the checkpoint' in s:
        cell['source'] = step2
        found = True
        break
assert found, 'Step 2 markdown not found'

found = False
for cell in nb['cells']:
    s = src(cell)
    if "MODEL_ID = 'autogluon/mitra-classifier'" not in s or 'MODEL_SOURCE =' not in s:
        continue
    # Delete package-only constants by keeping only the pinned constants and NETWORK timeout.
    cfg = s.index("EXPECTED_CONFIG_SHA256 = '")
    cfg_eol = s.index('\n', cfg) + 1
    timeout = s.index('NETWORK_TIMEOUT_SECONDS = 30', cfg_eol)
    s = s[:cfg_eol] + s[timeout:]

    fn_start = s.find('def load_dimer_')
    fn_end = s.find('def install_offline_snapshot(', fn_start)
    assert fn_start >= 0 and fn_end > fn_start, 'DIMER loader boundaries not found'
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
    print('✓ DIMER checkpoint files match the notebook release digests:', MODEL_ID, PINNED_REVISION)
    print('ℹ Current DIMER producer contract has no embedded package manifest; MOD7 remains pending for this source path.')

'''
    s = s[:fn_start] + new_fn + s[fn_end:]
    s, n = re.subn(
        r"MODEL_SOURCE = 'Pinned upstream'\s*# @param \[[^\n]+\]",
        "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER files', 'Pinned upstream']",
        s,
        count=1,
    )
    assert n == 1, 'MODEL_SOURCE declaration not replaced'
    s, n = re.subn(
        r"if MODEL_SOURCE == 'DIMER[^']*':\n\s+load_dimer_[A-Za-z0-9_]+\([^\n]+\)",
        "if MODEL_SOURCE == 'DIMER files':\n    load_dimer_files(weights_path, config_path)",
        s,
        count=1,
    )
    assert n == 1, 'DIMER source dispatch not replaced'
    cell['source'] = s.splitlines(keepends=True)
    found = True
    break
assert found, 'model acquisition code cell not found'

# Opening prose may still mention the repository-defined transport envelope from an interim attempt.
for cell in nb['cells']:
    s = src(cell)
    if 'repository-defined notebook package built from the current DIMER checkpoint pair' in s:
        cell['source'] = s.replace('repository-defined notebook package built from the current DIMER checkpoint pair', 'DIMER checkpoint pair')
save_nb(E2E, nb)

# Companion: make the feature schema structurally valid before load and equal to predictor schema after load.
nb = load_nb(INF)
load_cell_found = False
for cell in nb['cells']:
    s = src(cell)
    if 'required_provenance = [' in s and 'TabularPredictor.load' in s:
        load_cell_found = True
        if 'features must be a non-empty list of unique, non-blank strings' not in s:
            marker = "missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\n"
            assert marker in s, 'missing provenance marker not found'
            s = s.replace(marker, marker + "features = run_metadata.get('features')\nif (\n    not isinstance(features, list)\n    or not features\n    or any(not isinstance(name, str) or not name.strip() for name in features)\n    or len(set(features)) != len(features)\n):\n    raise RuntimeError('Predictor provenance features must be a non-empty list of unique, non-blank strings.')\nif run_metadata.get('target_column') in set(features):\n    raise RuntimeError('Predictor provenance target_column must not also appear in features.')\n", 1)
        # Normalize existing schema comparison, whether this head has the earlier hash remediation or not.
        if 'Predictor feature schema matches required provenance.' not in s:
            old = "if run_metadata.get('features'):\n    FEATURE_COLUMNS = list(run_metadata['features'])\nelse:\n    feature_metadata = getattr(predictor, 'feature_metadata_in', None)\n    if feature_metadata is None:\n        raise RuntimeError('Could not determine required input feature columns from predictor metadata.')\n    FEATURE_COLUMNS = list(feature_metadata.get_features())\n"
            if old in s:
                replacement = "FEATURE_COLUMNS = list(features)\nfeature_metadata = getattr(predictor, 'feature_metadata_in', None)\nif feature_metadata is None:\n    raise RuntimeError('Loaded predictor does not expose feature_metadata_in for schema verification.')\nloaded_features = list(feature_metadata.get_features())\nif loaded_features != FEATURE_COLUMNS:\n    raise RuntimeError(\n        'Loaded predictor feature schema does not match required provenance. '\n        f'Recorded={FEATURE_COLUMNS}; loaded={loaded_features}'\n    )\nprint('✓ Predictor feature schema matches required provenance.')\n"
                s = s.replace(old, replacement, 1)
            elif 'Loaded predictor feature schema does not match' in s:
                # Existing stronger interim implementation: retain it, but add a stable success marker.
                load_pos = s.find('predictor = TabularPredictor.load')
                insert_pos = s.find('\n', load_pos) + 1
                # Only add marker after the existing schema check later in the cell if possible.
                schema_pos = s.find('Loaded predictor feature schema does not match', insert_pos)
                if schema_pos >= 0:
                    next_blank = s.find('\n\n', schema_pos)
                    if next_blank >= 0:
                        s = s[:next_blank] + "\nprint('✓ Predictor feature schema matches required provenance.')" + s[next_blank:]
        cell['source'] = s.splitlines(keepends=True)
    s = src(cell)
    if 'Predictor provenance: read from `tutorial_run_metadata.json` when available' in s:
        cell['source'] = s.replace(
            'Predictor provenance: read from `tutorial_run_metadata.json` when available',
            'Predictor provenance: required and validated from `tutorial_run_metadata.json` before deserialization',
        )
assert load_cell_found, 'companion load cell not found'
save_nb(INF, nb)

# README: remove interim producer-envelope claims and state the actual platform boundary.
text = README.read_text(encoding='utf-8')
# Replace paragraph containing MODEL_SOURCE and self-describing/package language.
paragraphs = text.split('\n\n')
new_paragraphs = []
inserted_contract = False
for p in paragraphs:
    if "MODEL_SOURCE" in p and ('manifest' in p.lower() or 'package' in p.lower()):
        if not inserted_contract:
            new_paragraphs.append("The repository's current DIMER deployment contract exposes an uploaded checkpoint as `model.safetensors` + `config.json`; it does not define a self-describing package-manifest producer. Accordingly, `MODEL_SOURCE = 'DIMER files'` accepts exactly those two files, verifies both against the pinned release digests, and explicitly records Notebook Spec MOD7 package-manifest provenance as pending. The separate `Pinned upstream` source remains fully pinned by immutable revision and digests.")
            inserted_contract = True
        continue
    new_paragraphs.append(p)
text = '\n\n'.join(new_paragraphs)
text = text.replace('repository-defined notebook package', 'DIMER checkpoint files')
text = text.replace('self-describing offline model package', 'two-file checkpoint')
text = text.replace('- DIMER ZIP upload or pinned-upstream checkpoint source selection;', '- DIMER two-file checkpoint upload or pinned-upstream checkpoint source selection;')
text = text.replace('DIMER self-describing offline model package OR pinned upstream checkpoint', 'DIMER model.safetensors + config.json OR pinned upstream checkpoint')
text = re.sub(r'- Distributed DIMER artifact: .*', "- Distributed DIMER artifact: checkpoint files (`model.safetensors` + `config.json`) under the repository's current deployment contract; no platform package-manifest producer is claimed", text, count=1)
# Drop links/claims about the interim repo-defined package producer.
text = re.sub(r'^.*DIMER_NOTEBOOK_PACKAGE\.md.*\n?', '', text, flags=re.M)
text = re.sub(r'^.*build_dimer_notebook_package\.py.*\n?', '', text, flags=re.M)
README.write_text(text, encoding='utf-8')

# DEPLOYMENT: reconcile with the actual workflow directory.
text = DEPLOY.read_text(encoding='utf-8')
text = text.replace('Two GitHub Actions workflows guard the repo:', 'One GitHub Actions workflow currently guards the repo:')
text = re.sub(
    r"- \*\*`integration`\*\* \(manual `workflow_dispatch`.*?nightly run\.\n",
    "- A historical/local GPU integration exercise covered image build → pinned offline Mitra load →\n  fine-tune → save → reload → predict, but **`.github/workflows/integration.yml` is not present in\n  the repository today**. Do not treat that historical exercise as an active CI workflow. If GPU\n  integration automation is restored, document the actual workflow and runner contract here.\n",
    text,
    flags=re.S,
    count=1,
)
text = text.replace('No `integration.yml` workflow exists on the current branch.', '`.github/workflows/integration.yml` is not present in the repository today.')
text = text.replace(
    'the `integration` workflow exercises the save → reload → predict\nround-trip on a GPU runner, so the artifact is verified to reload and serve predictions outside\nthe training process.',
    'a historical/local GPU exercise covered the save → reload → predict\nround-trip, but there is currently no `integration.yml` workflow providing recurring CI evidence.',
)
DEPLOY.write_text(text, encoding='utf-8')

# Restore native CI to repository checks only; remove the unestablished interim package producer.
text = CI.read_text(encoding='utf-8')
text = re.sub(r"\s+- name: DIMER notebook-package producer\n\s+run: python scripts/build_dimer_notebook_package\.py --self-test\n", '\n', text, count=1)
text = text.replace('          python -m py_compile scripts/build_dimer_notebook_package.py\n', '')
CI.write_text(text, encoding='utf-8')

# Validator: require the honest contract and remove interim producer assertions.
text = VALIDATOR.read_text(encoding='utf-8')
for marker in [
    '        "PACKAGE_MANIFEST_FILENAME",\n',
    '        "MAX_DIMER_UPLOAD_BYTES",\n',
    '        "MAX_DIMER_EXPANDED_BYTES",\n',
    '        "DIMER notebook package",\n',
    '        "dimer-model-manifest.json",\n',
    '        "load_dimer_package",\n',
    '        "manifest_version",\n',
    '        "DIMER itself is not claimed to emit this ZIP",\n',
]:
    text = text.replace(marker, '')
if '        "load_dimer_files",\n' not in text:
    text = text.replace('        "Remote-code boundary",\n', '        "Remote-code boundary",\n        "load_dimer_files",\n        "MAX_DIMER_FILE_BYTES",\n        "MOD7 remains pending",\n')
text = text.replace(
    '        "weights_from_dimer" not in code_text and "load_dimer_package" in code_text,\n        "DIMER ZIP path must use the manifest-validating offline package loader",',
    '        "load_dimer_files" in code_text and "load_dimer_package" not in code_text,\n        "DIMER source path must use the bounded two-file loader matching the current platform contract",',
)
if 'features must be a non-empty list of unique, non-blank strings' not in text:
    text = text.replace('        "Required provenance validated",\n', '        "Required provenance validated",\n        "features must be a non-empty list of unique, non-blank strings",\n')
if 'Predictor feature schema matches required provenance' not in text:
    text = text.replace('        "Loaded predictor feature schema does not match",\n', '        "Loaded predictor feature schema does not match",\n        "Predictor feature schema matches required provenance",\n')
# Remove package-producer doc assertions.
text = text.replace('        "DIMER_NOTEBOOK_PACKAGE.md",\n', '')
text = text.replace('        "repository-defined",\n', '')
text = re.sub(
    r"\n    package_doc = ROOT / \"docs\" / \"DIMER_NOTEBOOK_PACKAGE\.md\".*?\n\n    deployment_text =",
    '\n\n    deployment_text =',
    text,
    flags=re.S,
    count=1,
)
text = text.replace('require("No `integration.yml` workflow exists on the current branch." in deployment_text, "deployment docs must state current integration-workflow absence")', 'require(".github/workflows/integration.yml` is not present" in deployment_text, "deployment docs must state current integration-workflow absence")')
text = text.replace('require("the `integration` workflow exercises" not in deployment_text, "stale integration-workflow evidence claim remains in DEPLOYMENT.md")', 'require("Two GitHub Actions workflows guard the repo" not in deployment_text, "stale two-workflow claim remains in DEPLOYMENT.md")')
VALIDATOR.write_text(text, encoding='utf-8')

print('robust round-2 remediation applied')
