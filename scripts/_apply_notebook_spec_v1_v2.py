#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "_apply_notebook_spec_v1.py"
spec = importlib.util.spec_from_file_location("notebook_migration_v1", OLD)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load migration module")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def patched_validator():
    text = m.VALIDATOR.read_text(encoding="utf-8")
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

    old_main = '''def main() -> int:\n    validate_training_tutorial()\n    validate_inference_tutorial()\n    validate_docs()\n    print("Standalone Mitra Colab tutorials: OK")\n    return 0\n'''
    new_main = '''def main() -> int:\n    validate_lockfile(ROOT / "tutorials" / "requirements-colab.lock.txt")\n    validate_lockfile(ROOT / "tutorials" / "requirements-inference.lock.txt")\n    validate_training_tutorial()\n    validate_inference_tutorial()\n    validate_docs()\n    print("Standalone Mitra Colab tutorials: OK")\n    return 0\n'''
    if old_main not in text:
        raise RuntimeError("validator main() block not found")
    text = text.replace(old_main, new_main, 1)
    m.VALIDATOR.write_text(text, encoding="utf-8")


m.patch_validator = patched_validator
m.main()
