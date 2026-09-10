#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
AINF = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"


def source_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def set_source(cell: dict, text: str) -> None:
    old = cell.get("source", "")
    if isinstance(old, list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_e2e() -> None:
    nb = json.loads(E2E.read_text(encoding="utf-8"))
    stale = "| `checksum mismatch` in Step 2 | the uploaded ZIP or download is not the pinned release |"
    fresh = "| `checksum mismatch` in Step 2 | an uploaded DIMER file or upstream download is not the pinned release |"
    stale_count = 0
    for cell in nb["cells"]:
        text = source_text(cell)
        if stale in text:
            text = replace_once(text, stale, fresh, "E2E stale DIMER ZIP troubleshooting")
            set_source(cell, text)
            stale_count += 1
    if stale_count != 1:
        raise RuntimeError(f"E2E stale troubleshooting reference: expected one cell, found {stale_count}")

    patched = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source_text(cell)
        if "required_reload_provenance = [" not in text or "reloaded_predictor = TabularPredictor.load" not in text:
            continue

        anchor = """missing_reload_provenance = [k for k in required_reload_provenance if not reload_metadata.get(k)]\nif missing_reload_provenance:\n    raise RuntimeError(f'Reload artifact provenance is incomplete: {missing_reload_provenance}')\n"""
        schema_guard = anchor + """\nreload_feature_columns = reload_metadata['features']\nif (\n    not isinstance(reload_feature_columns, list)\n    or not reload_feature_columns\n    or any(not isinstance(name, str) or not name.strip() for name in reload_feature_columns)\n    or len(set(reload_feature_columns)) != len(reload_feature_columns)\n):\n    raise RuntimeError(\n        'Reload artifact provenance features must be a non-empty list of unique, non-blank string column names.'\n    )\nif reload_feature_columns != list(FEATURE_COLUMNS):\n    raise RuntimeError(\n        'Reload artifact provenance feature schema does not match the producing notebook feature schema.'\n    )\nprint('✓ Reload feature schema validated before deserialization.')\n"""
        text = replace_once(text, anchor, schema_guard, "E2E pre-deserialization feature schema guard")

        load_anchor = "reloaded_predictor = TabularPredictor.load(str(reload_predictor_root))\n"
        load_guard = load_anchor + """reloaded_feature_metadata = getattr(reloaded_predictor, 'feature_metadata_in', None)\nif reloaded_feature_metadata is None:\n    raise RuntimeError('Reloaded predictor does not expose feature_metadata_in; cannot reconcile artifact feature schema.')\nreloaded_predictor_features = list(reloaded_feature_metadata.get_features())\nif reloaded_predictor_features != reload_feature_columns:\n    raise RuntimeError(\n        'Reload artifact feature schema disagrees with the loaded predictor feature schema.'\n    )\nprint('✓ Reload artifact feature schema reconciled with loaded predictor.')\n"""
        text = replace_once(text, load_anchor, load_guard, "E2E post-load feature schema reconciliation")
        set_source(cell, text)
        patched += 1

    if patched != 1:
        raise RuntimeError(f"E2E reload cell: expected one patch target, found {patched}")

    rendered = "\n".join(source_text(c) for c in nb["cells"])
    if "DIMER ZIP" in rendered:
        raise RuntimeError("E2E still contains stale 'DIMER ZIP' wording")
    E2E.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_inference() -> None:
    nb = json.loads(AINF.read_text(encoding="utf-8"))

    stale = "**You do not need the original DIMER ZIP, `model.safetensors`, `config.json`, or DIMER Workbench.**"
    fresh = "**You do not need the original DIMER checkpoint files (`model.safetensors` + `config.json`) or DIMER Workbench.**"
    stale_count = 0
    for cell in nb["cells"]:
        text = source_text(cell)
        if stale in text:
            text = replace_once(text, stale, fresh, "artifact-inference stale DIMER ZIP wording")
            set_source(cell, text)
            stale_count += 1
    if stale_count != 1:
        raise RuntimeError(f"artifact-inference stale DIMER ZIP reference: expected one cell, found {stale_count}")

    archive_patched = 0
    schema_patched = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source_text(cell)

        if "def safe_extract_zip(zip_path, destination):" in text:
            if "from pathlib import Path, PurePosixPath" not in text:
                text = replace_once(
                    text,
                    "from pathlib import Path\n",
                    "from pathlib import Path, PurePosixPath\n",
                    "artifact-inference PurePosixPath import",
                )
            old = """def safe_extract_zip(zip_path, destination):\n    destination = destination.resolve()\n    with zipfile.ZipFile(zip_path) as z:\n        expanded_bytes = 0\n        for info in z.infolist():\n            if '\\\\' in info.filename:\n                raise RuntimeError(f'Backslash archive member paths are not allowed: {info.filename!r}')\n            member = Path(info.filename)\n            if member.is_absolute() or '..' in member.parts:\n                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')\n"""
            new = """def safe_extract_zip(zip_path, destination):\n    destination = destination.resolve()\n    with zipfile.ZipFile(zip_path) as z:\n        expanded_bytes = 0\n        seen_members = set()\n        for info in z.infolist():\n            if '\\\\' in info.filename:\n                raise RuntimeError(f'Backslash archive member paths are not allowed: {info.filename!r}')\n            portable_member = PurePosixPath(info.filename)\n            normalized_member = portable_member.as_posix()\n            if normalized_member in seen_members:\n                raise RuntimeError(f'Duplicate archive member path: {normalized_member!r}')\n            seen_members.add(normalized_member)\n            member = Path(*portable_member.parts)\n            if member.is_absolute() or '..' in member.parts:\n                raise RuntimeError(f'Unsafe archive member path: {info.filename!r}')\n"""
            text = replace_once(text, old, new, "artifact-inference duplicate ZIP member guard")
            set_source(cell, text)
            archive_patched += 1

        if "required_provenance = [" in text and "predictor = TabularPredictor.load" in text:
            anchor = """missing_provenance = [k for k in required_provenance if not run_metadata.get(k)]\nif missing_provenance:\n    raise RuntimeError(f'Predictor provenance is incomplete; missing required field(s): {missing_provenance}')\n"""
            schema_guard = anchor + """\nartifact_feature_columns = run_metadata['features']\nif (\n    not isinstance(artifact_feature_columns, list)\n    or not artifact_feature_columns\n    or any(not isinstance(name, str) or not name.strip() for name in artifact_feature_columns)\n    or len(set(artifact_feature_columns)) != len(artifact_feature_columns)\n):\n    raise RuntimeError(\n        'Predictor provenance features must be a non-empty list of unique, non-blank string column names.'\n    )\nprint('✓ Artifact feature schema validated before deserialization.')\n"""
            text = replace_once(text, anchor, schema_guard, "artifact-inference pre-deserialization feature schema guard")

            old_features = """if run_metadata.get('features'):\n    FEATURE_COLUMNS = list(run_metadata['features'])\nelse:\n    feature_metadata = getattr(predictor, 'feature_metadata_in', None)\n    if feature_metadata is None:\n        raise RuntimeError('Could not determine required input feature columns from predictor metadata.')\n    FEATURE_COLUMNS = list(feature_metadata.get_features())\n"""
            new_features = """feature_metadata = getattr(predictor, 'feature_metadata_in', None)\nif feature_metadata is None:\n    raise RuntimeError('Loaded predictor does not expose feature_metadata_in; cannot reconcile artifact feature schema.')\npredictor_feature_columns = list(feature_metadata.get_features())\nif predictor_feature_columns != artifact_feature_columns:\n    raise RuntimeError('Artifact feature schema disagrees with the loaded predictor feature schema.')\nFEATURE_COLUMNS = list(artifact_feature_columns)\nprint('✓ Artifact feature schema reconciled with loaded predictor.')\n"""
            text = replace_once(text, old_features, new_features, "artifact-inference post-load feature schema reconciliation")
            set_source(cell, text)
            schema_patched += 1

    if archive_patched != 1:
        raise RuntimeError(f"artifact-inference archive cell: expected one patch target, found {archive_patched}")
    if schema_patched != 1:
        raise RuntimeError(f"artifact-inference provenance cell: expected one patch target, found {schema_patched}")

    rendered = "\n".join(source_text(c) for c in nb["cells"])
    if "DIMER ZIP" in rendered:
        raise RuntimeError("artifact-inference still contains stale 'DIMER ZIP' wording")
    AINF.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_validator() -> None:
    text = VALIDATOR.read_text(encoding="utf-8")

    training_anchor = '        "Reload provenance validated",\n'
    training_new = training_anchor + '        "Reload feature schema validated before deserialization",\n        "Reload artifact feature schema reconciled with loaded predictor",\n'
    text = replace_once(text, training_anchor, training_new, "validator E2E feature-schema markers")

    inference_anchor = '        "Backslash archive member paths are not allowed",\n'
    inference_new = inference_anchor + '        "Duplicate archive member path",\n        "Artifact feature schema validated before deserialization",\n        "Artifact feature schema reconciled with loaded predictor",\n'
    text = replace_once(text, inference_anchor, inference_new, "validator artifact-inference markers")

    code_anchor = '    code_text = "\\n".join(source for _, source in parsed_code)\n'
    # The first occurrence belongs to the E2E validator; patch the second occurrence only.
    first = text.find(code_anchor)
    second = text.find(code_anchor, first + 1)
    if first < 0 or second < 0:
        raise RuntimeError("validator: could not locate artifact-inference code_text anchor")
    inference_forbidden = """    for forbidden_text in (\n        "DIMER ZIP",\n        "dimer-model-manifest.json",\n        "load_dimer_package",\n    ):\n        require(\n            forbidden_text not in text,\n            f"inference tutorial contains stale/unsupported DIMER package wording: {forbidden_text}",\n        )\n\n"""
    text = text[:second] + inference_forbidden + text[second:]

    VALIDATOR.write_text(text, encoding="utf-8")


def main() -> None:
    patch_e2e()
    patch_inference()
    patch_validator()
    print("Reviewer round 3 remediation applied.")


if __name__ == "__main__":
    main()
