#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
README = ROOT / "tutorials" / "README.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"


def get_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict, text: str) -> None:
    if isinstance(cell.get("source"), list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
step2_md = """## 2. Acquire, verify, and lock the checkpoint

A model file is code you are about to run and weights you are about to trust. This step makes the supported acquisition paths explicit and checkable:

- **DIMER files** uploads exactly the two files exposed by the current DIMER model-download boundary: `model.safetensors` and `config.json`. Both are SHA-256 verified against this notebook release before either file is staged. DIMER does not currently provide this notebook with a package manifest (or equivalent producer metadata) that independently declares model identity and immutable revision, so the Notebook Spec MOD7 provenance gap remains explicit and this notebook stays **Candidate**. A failed or incomplete DIMER upload never falls back to the network.
- **Pinned upstream** fetches those same two files from the exact AutoGluon revision `c425e9fa…` on the Hugging Face Hub. A revision is an immutable commit; a model *name* is a branch that can change.

For both paths, the notebook records the intended model ID and immutable revision, verifies the exact release digests, stages the files into an isolated Hugging Face cache, and marks that cache offline. Finally it asks Hugging Face to resolve the model *exactly as AutoGluon will* and refuses to continue unless both resolved files come from that verified snapshot **and still match the expected digests**. The digest pair establishes byte identity with the pinned release; it does not manufacture DIMER-side provenance that the current producer does not emit.

If the lock check fails, use *Runtime ▸ Restart session* and run from Step 1 downward. Network requests in the pinned-upstream path use a finite timeout so outages fail clearly instead of hanging.

**What to look for:** two `✓ … verified` lines with digest prefixes `e06a055e91a3…` and `2c96c24dd25f…`, then `✓ Hugging Face resolver locked …`.

**Remote-code boundary:** this tutorial does not enable Hugging Face `trust_remote_code`; execution uses the installed AutoGluon implementation plus the verified `safetensors` weights and pinned configuration.
"""

md_hits = 0
code_hits = 0
for cell in nb.get("cells", []):
    src = get_source(cell)
    if cell.get("cell_type") == "markdown" and src.startswith("## 2. Acquire, verify, and lock the checkpoint"):
        set_source(cell, step2_md)
        md_hits += 1
        continue
    if cell.get("cell_type") != "code" or "def load_dimer_package" not in src or "MODEL_SOURCE" not in src:
        continue

    src = src.replace("PACKAGE_MANIFEST_FILENAME = 'dimer-model-manifest.json'\n", "")
    src = src.replace("PACKAGE_MANIFEST_VERSION = '1.0'\n", "")
    start = src.index("def load_dimer_package(weights_dest, config_dest):")
    end = src.index("def install_offline_snapshot(weights, config, weights_digest):")
    new_loader = '''def load_dimer_files(weights_dest, config_dest):
    from google.colab import files

    print('Upload exactly two files downloaded from DIMER: model.safetensors and config.json')
    uploaded = files.upload()
    expected = {
        'model.safetensors': EXPECTED_WEIGHTS_SHA256,
        'config.json': EXPECTED_CONFIG_SHA256,
    }
    if set(uploaded) != set(expected):
        raise RuntimeError(
            'DIMER files mode requires exactly model.safetensors and config.json. '
            f'Missing={sorted(set(expected) - set(uploaded))}; '
            f'unexpected={sorted(set(uploaded) - set(expected))}'
        )

    verified_payloads = {}
    for filename, expected_digest in expected.items():
        payload = uploaded[filename]
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(
                f'DIMER {filename} checksum mismatch. '
                f'Expected {expected_digest}; got {actual_digest}.'
            )
        verified_payloads[filename] = payload
        print(f'✓ DIMER {filename} matched pinned release digest: {actual_digest[:12]}…')

    weights_dest.write_bytes(verified_payloads['model.safetensors'])
    config_dest.write_bytes(verified_payloads['config.json'])
    print(
        '✓ DIMER checkpoint pair accepted for the pinned notebook release. '
        'Producer manifest/model-revision provenance is not supplied by the current DIMER download boundary.'
    )

'''
    src = src[:start] + new_loader + src[end:]
    src = src.replace(
        "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']",
        "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER files', 'Pinned upstream']",
    )
    src = src.replace(
        "if MODEL_SOURCE == 'DIMER ZIP':\n    load_dimer_package(weights_path, config_path)",
        "if MODEL_SOURCE == 'DIMER files':\n    load_dimer_files(weights_path, config_path)",
    )
    set_source(cell, src)
    code_hits += 1

if md_hits != 1 or code_hits != 1:
    raise RuntimeError(f"unexpected notebook patch cardinality: markdown={md_hits}, code={code_hits}")

notebook_text = "\n".join(get_source(c) for c in nb.get("cells", []))
for stale in ("dimer-model-manifest.json", "load_dimer_package", "DIMER ZIP", "PACKAGE_MANIFEST_FILENAME", "PACKAGE_MANIFEST_VERSION"):
    if stale in notebook_text:
        raise RuntimeError(f"stale invented DIMER package contract remains in notebook: {stale}")
for required in ("DIMER files", "load_dimer_files", "MOD7 provenance gap", "model.safetensors and config.json"):
    if required not in notebook_text:
        raise RuntimeError(f"required DIMER two-file marker missing from notebook: {required}")
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

readme = README.read_text(encoding="utf-8")n