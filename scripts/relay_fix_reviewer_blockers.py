#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
README = ROOT / "tutorials" / "README.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"


def get_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True) if isinstance(cell.get("source"), list) else text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


# --- E2E notebook: use the actual DIMER two-file boundary; do not invent a package producer. ---
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

md_hits = code_hits = 0
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
    loader = '''def load_dimer_files(weights_dest, config_dest):
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
    src = src[:start] + loader + src[end:]
    src = replace_once(
        src,
        "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER ZIP', 'Pinned upstream']",
        "MODEL_SOURCE = 'Pinned upstream'  # @param ['DIMER files', 'Pinned upstream']",
        "MODEL_SOURCE form options",
    )
    src = replace_once(
        src,
        "if MODEL_SOURCE == 'DIMER ZIP':\n    load_dimer_package(weights_path, config_path)",
        "if MODEL_SOURCE == 'DIMER files':\n    load_dimer_files(weights_path, config_path)",
        "DIMER acquisition dispatch",
    )
    set_source(cell, src)
    code_hits += 1

if (md_hits, code_hits) != (1, 1):
    raise RuntimeError(f"unexpected notebook patch cardinality: markdown={md_hits}, code={code_hits}")

notebook_text = "\n".join(get_source(c) for c in nb.get("cells", []))
for stale in ("dimer-model-manifest.json", "load_dimer_package", "DIMER ZIP", "PACKAGE_MANIFEST_FILENAME", "PACKAGE_MANIFEST_VERSION"):
    if stale in notebook_text:
        raise RuntimeError(f"stale invented DIMER package contract remains in notebook: {stale}")
for required in ("DIMER files", "load_dimer_files", "MOD7 provenance gap", "model.safetensors and config.json"):
    if required not in notebook_text:
        raise RuntimeError(f"required DIMER two-file marker missing from notebook: {required}")
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# --- Tutorial registry/docs: describe the real boundary and the explicit provenance limitation. ---
readme = README.read_text(encoding="utf-8")
readme = replace_once(
    readme,
    "For `MODEL_SOURCE = 'DIMER ZIP'`, Notebook Spec v1.0 expects a self-describing offline package containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. Legacy weights-only DIMER ZIPs are refused rather than completed from the network.",
    "The current DIMER model-download boundary exposes `model.safetensors` + `config.json`, not a self-describing package manifest. `MODEL_SOURCE = 'DIMER files'` therefore requires exactly those two files and verifies both against the pinned release SHA-256 values before staging. The notebook records the intended model ID/revision but does not claim DIMER-side manifest provenance that the current producer does not emit; that Notebook Spec MOD7 provenance gap is why the notebook remains Candidate. A failed DIMER upload never falls back to the network.",
    "DIMER contract paragraph",
)
readme = replace_once(
    readme,
    "Users can download the self-describing offline model package from DIMER and run the notebook independently in Google Colab. As a separate explicitly selected source mode, the notebook can instead retrieve the exact pinned upstream checkpoint associated with the DIMER release; a failed DIMER package validation never falls back to the network.",
    "Users can download `model.safetensors` and `config.json` from DIMER and run the notebook independently in Google Colab. As a separate explicitly selected source mode, the notebook can instead retrieve the exact pinned upstream checkpoint associated with the DIMER release; a failed or incomplete DIMER upload never falls back to the network.",
    "build tutorial DIMER paragraph",
)
readme = replace_once(readme, "- DIMER ZIP upload or pinned-upstream checkpoint source selection;", "- DIMER two-file checkpoint upload or pinned-upstream checkpoint source selection;", "DIMER tutorial bullet")
readme = replace_once(readme, "DIMER self-describing offline model package OR pinned upstream checkpoint", "DIMER model.safetensors + config.json OR pinned upstream checkpoint", "DIMER flow")
readme = replace_once(
    readme,
    "- Distributed DIMER artifact: self-describing offline package containing the pinned model weights, configuration, and package manifest",
    "- Distributed DIMER artifact: `model.safetensors` + `config.json` (both pinned by SHA-256; no DIMER package manifest is currently supplied to this notebook)",
    "AI provenance DIMER artifact",
)
for stale in ("dimer-model-manifest.json", "DIMER ZIP", "self-describing offline model package"):
    if stale in readme:
        raise RuntimeError(f"stale invented DIMER package claim remains in tutorials/README.md: {stale}")
for required in ("MODEL_SOURCE = 'DIMER files'", "MOD7 provenance gap", "model.safetensors` + `config.json"):
    if required not in readme:
        raise RuntimeError(f"required README DIMER marker missing: {required}")
README.write_text(readme, encoding="utf-8")

# --- Static validator: enforce the two-file contract and forbid the invented wrapper. ---
validator = VALIDATOR.read_text(encoding="utf-8")
validator = replace_once(validator, '        "DIMER ZIP",\n', '        "DIMER files",\n', "validator DIMER source marker")
validator = replace_once(
    validator,
    '        "PACKAGE_MANIFEST_FILENAME",\n        "dimer-model-manifest.json",\n        "load_dimer_package",\n        "manifest_version",\n        "legacy weights-only ZIPs are refused",\n',
    '        "load_dimer_files",\n        "MOD7 provenance gap",\n        "Upload exactly two files downloaded from DIMER",\n',
    "validator package-marker block",
)
validator = replace_once(
    validator,
    'FORBIDDEN = (\n',
    'FORBIDDEN = (\n    "DIMER ZIP",\n    "dimer-model-manifest.json",\n    "load_dimer_package",\n    "PACKAGE_MANIFEST_FILENAME",\n    "PACKAGE_MANIFEST_VERSION",\n',
    "validator forbidden tuple",
)
VALIDATOR.write_text(validator, encoding="utf-8")

print("Reviewer blockers patched: actual DIMER two-file boundary, docs, and validator.")
