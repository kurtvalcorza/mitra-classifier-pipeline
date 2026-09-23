#!/usr/bin/env python3
"""Repository-specific static checks for the standalone Mitra Classifier tutorials (NOTEBOOK_SPEC 2.0).

The carrier, parity, hygiene and profile checks live in ``tools/validate_release_assets.py`` (run first). This
script keeps the invariants specific to this repository's contract: the notebooks exercise the public API in
``mitra_pipeline`` (never reimplementing fit/inference/archive checks), the fine-tuning and inference gates default
off, the artifact-inference notebook fails closed without a trusted digest, and the tutorial requirements mirror the
``pyproject.toml`` pins the notebooks carry. It claims no runtime execution evidence.
"""
# ruff: noqa: E501  -- rule messages name the requirement in full; they are kept on one line
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"
TUTORIAL_README = ROOT / "tutorials" / "README.md"
PUBLIC_API = ROOT / "mitra_pipeline" / "tutorial_api.py"
REQUIREMENTS = ROOT / "tutorials" / "requirements-colab.in"
PYPROJECT = ROOT / "pyproject.toml"

MODEL_ID = "autogluon/mitra-classifier"
PINNED_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
PLACEHOLDERS = re.compile(r"\b(TODO|TBD|FIXME)\b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def load_notebook(path: Path) -> tuple[dict, str, list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    cells = payload.get("cells", [])
    require(bool(cells), f"{path.name}: no cells")
    cell_ids = [cell.get("id") for cell in cells]
    require(all(isinstance(cell_id, str) and cell_id.strip() for cell_id in cell_ids), f"{path.name}: every cell needs a stable id")
    require(len(set(cell_ids)) == len(cell_ids), f"{path.name}: cell ids must be unique")
    code_cells, own_cells = [], []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = source_text(cell)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise AssertionError(f"{path.name}: cell {index} does not compile: {exc}") from exc
        code_cells.append(code)
        if not cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            own_cells.append(code)
    full_text = "\n".join(source_text(cell) for cell in cells)
    require(not PLACEHOLDERS.search(full_text), f"{path.name}: placeholder marker survives")
    require(all(cell.get("execution_count") is None for cell in cells if cell.get("cell_type") == "code"), f"{path.name}: execution counts must be cleared")
    require(all(not cell.get("outputs") for cell in cells if cell.get("cell_type") == "code"), f"{path.name}: outputs must be cleared")
    return payload, full_text, code_cells, own_cells


def require_profile(payload: dict, filename: str, expected: str) -> None:
    dimer = payload.get("metadata", {}).get("dimer", {})
    require(dimer.get("notebook_profile") == expected, f"{filename}: metadata profile must be {expected}")
    require(str(dimer.get("notebook_spec")) == "2.0" and dimer.get("standalone") is True, f"{filename}: must be standalone spec 2.0")


def require_markers(text: str, markers: tuple[str, ...], label: str) -> None:
    for marker in markers:
        require(marker in text, f"{label}: missing marker {marker!r}")


def top_level_literal(code_cells: list[str], name: str, expected) -> bool:
    for code in code_cells:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == name and isinstance(node.value, ast.Constant):
                    return node.value.value == expected
    return False


def main_tutorial() -> None:
    payload, text, _code_cells, own = load_notebook(MAIN)
    require_profile(payload, MAIN.name, "E2E")
    require_markers(
        text,
        (
            "**Profile:** `E2E`",
            "**This notebook is standalone.**",
            "validate_inputs(",
            "validate_labeled_frame(",
            "split_overlap_report(",
            "cap_training_rows(",
            "majority_class_baseline(",
            "stratified_holdout(",
            "classification_metrics(",
            "align_probabilities(",
            "require_class_coverage(",
            "LGBMClassifier",
            "RandomForestClassifier",
            "evaluation_report(",
            "no weight is gradient-updated",
            "uncalibrated",
            "no threshold",
            "probability_",
            "artifact_manifest.json",
            "tutorial_run_metadata.json",
            "write_artifact_manifest(",
            "validate_artifact_directory(",
            "safe_extract_archive(",
            "np.testing.assert_allclose(",
            "## Interpretation and limits",
            "It does **not** establish",
        ),
        MAIN.name,
    )
    own_text = "\n".join(own)
    require(top_level_literal(own, "RUN_FINE_TUNING", False), "fine-tuning must be gated off by default")
    require(top_level_literal(own, "RUN_NEW_DATA_INFERENCE", False), "new-data inference must be gated off by default")
    require(top_level_literal(own, "USE_BYOD", False), "BYOD must be gated off by default")
    require(top_level_literal(own, "MIN_SELECTION_HOLDOUT_ROWS", 50), "selection evidence guard missing")
    require("pipe.fit(" in own_text and "ACTIVE_MODEL.predict(" in own_text, "main notebook must exercise the carried pipeline API")
    require("TabularPredictor(" not in own_text and "hyperparameters=" not in own_text, "main notebook must not construct AutoGluon predictors outside the carried API")
    require("predict_proba(" in own_text and "argmax" in text, "main classification notebook must expose class probabilities under the argmax rule")
    require("github.com/kurtvalcorza" not in "\n".join(_code_cells), "main notebook must not reach this repository (ST1)")
    require("from mitra_pipeline" not in own_text and "import mitra_pipeline" not in own_text, "main notebook must not import the repository package (ST1)")


def inference_tutorial() -> None:
    payload, text, code_cells, own = load_notebook(INFERENCE)
    require_profile(payload, INFERENCE.name, "ARTIFACT-INFERENCE")
    require_markers(
        text,
        (
            "**Profile:** `ARTIFACT-INFERENCE`",
            "**This notebook is standalone.**",
            "ARTIFACT_ZIP_PATH",
            "NEW_DATA_PATH",
            "EXPECTED_ZIP_SHA256",
            "ALLOW_UNVERIFIED_ARTIFACT",
            "artifact_manifest",
            "run_metadata",
            "validate_artifact_directory(",
            "safe_extract_archive(",
            "TabularPredictor.load(",
            "validate_inputs(",
            "uncalibrated",
            "no threshold is shipped",
            "probability_",
            "_predictions.csv",
            "A successful run proves",
            "It does **not** prove",
        ),
        INFERENCE.name,
    )
    own_text = "\n".join(own)
    require(top_level_literal(own, "ALLOW_UNVERIFIED_ARTIFACT", False), "artifact-inference notebook must fail closed unless unverified loading is explicitly enabled")
    require(".fit(" not in own_text, "artifact-inference notebook must not fit/train")
    require("shutil.make_archive(" not in own_text and "write_artifact_manifest(" not in own_text, "artifact-inference notebook must not create an artifact")
    require(own_text.index("validate_artifact_directory(") < own_text.index("TabularPredictor.load("), "manifest/provenance must be verified before deserialization")
    require("AUTOGLUON_VERSION:" in own_text and "runtime_python_mm" in own_text, "runtime compatibility must be checked before deserialization")
    require("to_csv(" in own_text, "artifact-inference notebook must export CSV")
    require("github.com/kurtvalcorza" not in "\n".join(code_cells), "artifact notebook must not reach this repository (ST1)")


def docs_and_api() -> None:
    readme = TUTORIAL_README.read_text(encoding="utf-8")
    api = PUBLIC_API.read_text(encoding="utf-8")
    req = [line.strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    deps_block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", pyproject, re.M | re.S)
    require(deps_block is not None, "pyproject.toml must declare [project].dependencies (the notebooks' PINS)")
    pins = re.findall(r'"([^"]+)"', deps_block.group(1))
    require(all("==" in p for p in pins), "pyproject runtime deps must be == pinned")
    # requirements-colab.in is the pip-compile input (a range for lightgbm); the exact pins live in pyproject.toml.
    require(req[0] == pins[0] and req[0].startswith("autogluon.tabular[mitra]=="), f"tutorials/requirements-colab.in must lead with the pyproject AutoGluon pin: {req} vs {pins}")
    require("autogluon.tabular[mitra]==1.5.0" in pins and "lightgbm==4.6.0" in pins, "pins must keep AutoGluon 1.5.0 and LightGBM 4.6.0")

    require_markers(
        readme,
        (
            "DIMER Notebook Specification 2.0",
            "`E2E`",
            "`ARTIFACT-INFERENCE`",
            "standalone (generated)",
            "release-verification",
            MODEL_ID,
            PINNED_REVISION,
        ),
        "tutorials/README.md",
    )
    require_markers(
        api,
        (
            f'MODEL_ID = "{MODEL_ID}"',
            f'MODEL_REVISION = "{PINNED_REVISION}"',
            "PINNED_REVISION = MODEL_REVISION",
            f'WEIGHTS_SHA256 = "{WEIGHTS_SHA256}"',
            f'CONFIG_SHA256 = "{CONFIG_SHA256}"',
            'ARTIFACT_FORMAT = "dimer-autogluon-predictor"',
            "ARTIFACT_FORMAT_VERSION = 1",
            "MAX_ARCHIVE_MEMBER_BYTES",
            "MAX_ARCHIVE_EXPANDED_BYTES",
            "MAX_COMPRESSION_RATIO",
            "def validate_dimer_model_package",
            "def write_artifact_manifest",
            "def validate_artifact_directory",
            "def safe_extract_archive",
            "def fit_mitra_predictor",
            "def verify_snapshot",
            "def stage_missing_files",
            "class MitraClassificationPipeline",
            "def validate_inputs",
            "def evaluation_report",
        ),
        "mitra_pipeline/tutorial_api.py",
    )


def main() -> int:
    main_tutorial()
    inference_tutorial()
    docs_and_api()
    print("Mitra Classifier repository-specific Notebook Specification 2.0 static conformance: OK")
    print("NOTE: static validation is not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
