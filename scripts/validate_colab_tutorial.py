#!/usr/bin/env python3
"""Static checks for the standalone Mitra Colab tutorials."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
INFERENCE_NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"
ROOT_README = ROOT / "README.md"
TUTORIAL_README = ROOT / "tutorials" / "README.md"

PINNED_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
SAMPLE_REVISION = "8fc19e80ae3166ec6bf964d194a28c80e6ba3b1f"

FORBIDDEN = (
    "mitra-classifier-finetuner",
    "mitra-classifier-dataset-validator",
    "load_breast_cancer",
    "Built-in demo",
    "mitra-classifier-pipeline/main/examples/sample-data/freshretailnet-band-h7.zip",
)
REPO_INTERNAL_IMPORT_PARTS = {"finetuner", "validator"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def code_sources(cells: list[dict]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = "\n".join(
            line
            for line in str(source).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        result.append((i, source))
    return result


def load_notebook(path: Path) -> tuple[list[dict], str, list[tuple[int, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("nbformat") == 4, f"{path.name} must use nbformat 4")
    cells = payload.get("cells", [])
    require(bool(cells), f"{path.name} has no cells")
    text = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in cells
    )
    parsed = code_sources(cells)
    for i, source in parsed:
        if source.strip():
            ast.parse(source, filename=f"{path.name}:cell-{i}")
    return cells, text, parsed


def top_level_literal_assignments_match(
    code_cells: list[tuple[int, str]], name: str, expected: object
) -> bool:
    found = False
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in tree.body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
                continue
            found = True
            if not isinstance(value, ast.Constant):
                return False
            if type(value.value) is not type(expected) or value.value != expected:
                return False
    return found


def repo_internal_imports(code_cells: list[tuple[int, str]]) -> set[str]:
    found: set[str] = set()
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if REPO_INTERNAL_IMPORT_PARTS.intersection(module.split(".")):
                    found.add(module)
    return found


def memory_guard_keys(code_cells: list[tuple[int, str]]) -> set[str]:
    keys: set[str] = set()
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "ag_args_fit" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
    return keys


def has_memory_guard(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "ag_args_fit" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "max_memory_usage_ratio"
                        and isinstance(value, ast.Name)
                        and value.id == "MAX_MEMORY_USAGE_RATIO"
                    ):
                        return True
    return False


def call_attributes(code_cells: list[tuple[int, str]]) -> set[str]:
    attrs: set[str] = set()
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attrs.add(node.func.attr)
    return attrs


def direct_tabular_predictor_construction(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TabularPredictor"
            ):
                return True
    return False


def has_current_fit_completion_gate(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        false_positions: list[int] = []
        fit_positions: list[int] = []
        true_positions: list[int] = []
        for position, node in enumerate(tree.body):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "FIT_RUN_COMPLETED" in names and isinstance(node.value, ast.Constant):
                    if node.value.value is False:
                        false_positions.append(position)
                    elif node.value.value is True:
                        true_positions.append(position)
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "fit_mitra"
                ):
                    fit_positions.append(position)
        if (
            false_positions
            and fit_positions
            and true_positions
            and min(false_positions) < min(fit_positions) < max(true_positions)
        ):
            return True
    return False


def predict_proba_calls_are_multiclass(
    code_cells: list[tuple[int, str]], minimum_calls: int
) -> bool:
    calls = 0
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "predict_proba"
            ):
                continue
            calls += 1
            keyword = next((k for k in node.keywords if k.arg == "as_multiclass"), None)
            if not (
                keyword
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return False
    return calls >= minimum_calls


def inference_step_is_self_contained(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if "RUN_NEW_DATA_INFERENCE" not in source:
            continue
        tree = ast.parse(source)
        imported_io = False
        imported_pandas_as_pd = False
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "io":
                        imported_io = True
                    if alias.name == "pandas" and alias.asname == "pd":
                        imported_pandas_as_pd = True
        return imported_io and imported_pandas_as_pd
    return False


def has_safe_direct_weights_copy_guard(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if "weights_from_dimer" not in source:
            continue
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or function.name != "weights_from_dimer":
                continue
            for candidate in ast.walk(function):
                if not isinstance(candidate, ast.If):
                    continue
                test = candidate.test
                if not (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.NotEq)
                    and len(test.comparators) == 1
                    and ast.unparse(test.left) == "p.resolve()"
                    and ast.unparse(test.comparators[0]) == "dest.resolve()"
                ):
                    continue
                for statement in candidate.body:
                    for node in ast.walk(statement):
                        if not (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "shutil"
                            and node.func.attr == "copy2"
                            and len(node.args) >= 2
                            and ast.unparse(node.args[0]) == "p"
                            and ast.unparse(node.args[1]) == "dest"
                        ):
                            continue
                        return True
    return False


def validate_training_tutorial() -> None:
    _, text, parsed_code = load_notebook(NOTEBOOK)

    for required in (
        PINNED_REVISION,
        WEIGHTS_SHA256,
        CONFIG_SHA256,
        SAMPLE_REVISION,
        "autogluon.tabular[mitra]==1.5.0",
        "DIMER ZIP",
        "Pinned upstream",
        "Sample dataset (FreshRetailNet)",
        "Upload pre-split train/val/test",
        "freshretailnet-band-h7.zip",
        "DATASET_CARD.md",
        "train.csv",
        "val.csv",
        "test.csv",
        "test_data",
        "assert_resolver_locked",
        "resolved_digest",
        "contains unseen target classes",
        "train_feature_set",
        "reindex(columns=ordered_columns)",
        "require_min_training_class_count",
        "Capped training split",
        "Inference CSV contains duplicate column names",
        "as_multiclass=True",
        "torch_version",
        "fine_tune_steps_requested",
        "one-row accuracy resolution",
        "FIT_RUN_COMPLETED",
        "Cleared stale predictor state and output paths before fitting.",
        "No predictor was successfully trained in this Step 4 execution.",
        "shutil.rmtree",
        "gc.collect",
        "torch.cuda.empty_cache",
        "## 7. Reload smoke test",
        "TabularPredictor.load(str(RELOAD_DIR))",
        "np.allclose",
        "CC BY 4.0",
        "## AI use and provenance",
        "GPT-5.6 Sol High",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(required in text, f"missing required tutorial marker: {required}")

    for forbidden in FORBIDDEN:
        require(
            forbidden not in text,
            f"standalone tutorial leaked forbidden dependency/configuration: {forbidden}",
        )

    code_text = "\n".join(source for _, source in parsed_code)
    require(
        re.search(r"\bDIMER_[A-Z0-9_]+\b", code_text) is None,
        "standalone tutorial must not depend on DIMER_* runtime variables",
    )
    imports = repo_internal_imports(parsed_code)
    require(
        not imports,
        f"standalone tutorial must not import repo-internal worker modules: {sorted(imports)}",
    )

    for name, expected in (
        ("RUN_FINE_TUNING", False),
        ("RUN_NEW_DATA_INFERENCE", False),
        ("FINE_TUNE_STEPS", 50),
        ("MAX_MEMORY_USAGE_RATIO", 1.1),
        ("NETWORK_TIMEOUT_SECONDS", 30),
    ):
        require(
            top_level_literal_assignments_match(parsed_code, name, expected),
            f"every top-level assignment to {name} must be the literal default {expected!r}",
        )

    require(has_memory_guard(parsed_code), "tutorial must pass max_memory_usage_ratio through .fit(...)")
    require(
        "ag.max_memory_usage_ratio" not in memory_guard_keys(parsed_code),
        "tutorial must not use the prefixed ag.max_memory_usage_ratio key",
    )
    require(
        has_current_fit_completion_gate(parsed_code),
        "tutorial must invalidate FIT_RUN_COMPLETED before fitting and set it true only after a current fit completes",
    )
    require(
        text.count("globals().get('FIT_RUN_COMPLETED', False)") >= 2,
        "inference and export must both gate on the current Step 4 completion flag",
    )
    require(
        text.count("require_min_training_class_count(train_data") >= 2,
        "training class counts must be checked before and after the optional 10k cap",
    )
    require(
        predict_proba_calls_are_multiclass(parsed_code, minimum_calls=3),
        "every training-tutorial predict_proba call must request as_multiclass=True",
    )
    require(
        inference_step_is_self_contained(parsed_code),
        "Step 5 inference must import io and pandas locally",
    )
    require(
        has_safe_direct_weights_copy_guard(parsed_code),
        "direct model.safetensors upload must avoid copying a path onto itself",
    )


def validate_inference_tutorial() -> None:
    _, text, parsed_code = load_notebook(INFERENCE_NOTEBOOK)

    for required in (
        "autogluon.tabular[mitra]==1.5.0",
        "mitra-predictor.zip",
        "predictor.pkl",
        "tutorial_run_metadata.json",
        "safe_extract_zip",
        "stat.S_IFLNK",
        "TabularPredictor.load",
        "FEATURE_COLUMNS",
        "Inference CSV contains duplicate column names",
        "predict_proba",
        "as_multiclass=True",
        "predictions.csv",
        "## AI use and provenance",
        "GPT-5.6 Sol High",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(required in text, f"inference tutorial missing required marker: {required}")

    code_text = "\n".join(source for _, source in parsed_code)
    require(
        re.search(r"\bDIMER_[A-Z0-9_]+\b", code_text) is None,
        "inference tutorial must not depend on DIMER_* runtime variables",
    )
    for forbidden_code in (
        "model.safetensors",
        "config.json",
        "hf_hub_download",
        "huggingface_hub",
        "fine_tune_steps",
    ):
        require(
            forbidden_code not in code_text,
            f"inference tutorial must not reacquire/train base model artifacts: {forbidden_code}",
        )

    imports = repo_internal_imports(parsed_code)
    require(
        not imports,
        f"inference tutorial must not import repo-internal worker modules: {sorted(imports)}",
    )
    attrs = call_attributes(parsed_code)
    require("fit" not in attrs, "inference tutorial must not call .fit(...)")
    require("load" in attrs, "inference tutorial must load an exported predictor")
    require("predict" in attrs, "inference tutorial must call predict(...)")
    require("predict_proba" in attrs, "inference tutorial must call predict_proba(...)")
    require(
        predict_proba_calls_are_multiclass(parsed_code, minimum_calls=1),
        "inference tutorial predict_proba must request as_multiclass=True",
    )
    require(
        not direct_tabular_predictor_construction(parsed_code),
        "inference tutorial must reload an exported predictor, not construct a new TabularPredictor",
    )


def validate_docs() -> None:
    root_readme = ROOT_README.read_text(encoding="utf-8")
    for required in (
        "## Try Mitra yourself in Google Colab",
        "tutorials/mitra_classifier_colab.ipynb",
        "colab.research.google.com",
        "GPT-5.6 Sol High",
        "Agent Relay",
    ):
        require(required in root_readme, f"root README missing tutorial/provenance marker: {required}")

    tutorial_readme = TUTORIAL_README.read_text(encoding="utf-8")
    for required in (
        PINNED_REVISION,
        WEIGHTS_SHA256,
        CONFIG_SHA256,
        "freshretailnet-band-h7.zip",
        "DATASET_CARD.md",
        "purged chronological split",
        "CC BY 4.0",
        "mitra_classifier_predictor_inference_colab.ipynb",
        "mitra-predictor.zip",
        "TabularPredictor.load",
        "predictions.csv",
        "## AI use and provenance",
        "GPT-5.6 Sol High",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(required in tutorial_readme, f"tutorial README missing marker: {required}")


def main() -> int:
    validate_training_tutorial()
    validate_inference_tutorial()
    validate_docs()
    print("Standalone Mitra Colab tutorials: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
