#!/usr/bin/env python3
"""Static checks for the standalone Mitra Colab tutorial."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
ROOT_README = ROOT / "README.md"
TUTORIAL_README = ROOT / "tutorials" / "README.md"

PINNED_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
SAMPLE_REVISION = "8fc19e80ae3166ec6bf964d194a28c80e6ba3b1f"

FORBIDDEN = (
    "mitra-classifier-finetuner",
    "mitra-classifier-dataset-validator",
    "DIMER_MODEL_DIR",
    "DIMER_DATASET_DIR",
    "DIMER_OUTPUT_DIR",
    "DIMER_HYPERPARAMETERS_JSON",
    "DIMER_PREPROCESSING_ARGS_JSON",
    "ag.max_memory_usage_ratio",
    "load_breast_cancer",
    "Built-in demo",
    "mitra-classifier-pipeline/main/examples/sample-data/freshretailnet-band-h7.zip",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def code_sources(cells: list[dict]) -> list[tuple[int, str]]:
    result = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        result.append((i, source))
    return result


def has_memory_guard(code_cells: list[tuple[int, str]]) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "fit":
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


def has_literal_assignment(
    code_cells: list[tuple[int, str]], variable: str, expected: object
) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if not isinstance(value, ast.Constant) or value.value != expected:
                continue
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                return True
    return False


def main() -> int:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    require(payload.get("nbformat") == 4, "notebook must use nbformat 4")
    cells = payload.get("cells", [])
    require(bool(cells), "notebook has no cells")

    text = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in cells
    )

    for required in (
        PINNED_REVISION,
        WEIGHTS_SHA256,
        CONFIG_SHA256,
        SAMPLE_REVISION,
        "autogluon.tabular[mitra]==1.5.0",
        "DIMER ZIP",
        "Pinned upstream",
        "MAX_MEMORY_USAGE_RATIO = 1.10",
        "NETWORK_TIMEOUT_SECONDS = 30",
        "Sample dataset (FreshRetailNet)",
        "freshretailnet-band-h7.zip",
        "DATASET_CARD.md",
        "train.csv",
        "val.csv",
        "test.csv",
        "sample_test_data",
        "CC BY 4.0",
        "## AI use and provenance",
        "OpenAI ChatGPT",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(required in text, f"missing required tutorial marker: {required}")

    for forbidden in FORBIDDEN:
        require(
            forbidden not in text,
            f"standalone tutorial leaked forbidden dependency/configuration: {forbidden}",
        )

    parsed_code = code_sources(cells)
    for i, source in parsed_code:
        if source.strip():
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{i}")

    require(
        has_memory_guard(parsed_code),
        "tutorial must pass max_memory_usage_ratio through a .fit(...) ag_args_fit keyword",
    )
    require(
        has_literal_assignment(parsed_code, "RUN_FINE_TUNING", False),
        "RUN_FINE_TUNING must default to False",
    )
    require(
        has_literal_assignment(parsed_code, "RUN_NEW_DATA_INFERENCE", False),
        "RUN_NEW_DATA_INFERENCE must default to False",
    )

    root_readme = ROOT_README.read_text(encoding="utf-8")
    for required in (
        "## Try Mitra yourself in Google Colab",
        "tutorials/mitra_classifier_colab.ipynb",
        "colab.research.google.com",
        "OpenAI ChatGPT",
        "Agent Relay",
    ):
        require(
            required in root_readme,
            f"root README missing tutorial/provenance marker: {required}",
        )

    tutorial_readme = TUTORIAL_README.read_text(encoding="utf-8")
    for required in (
        "freshretailnet-band-h7.zip",
        "DATASET_CARD.md",
        "purged chronological split",
        "CC BY 4.0",
        "## AI use and provenance",
        "OpenAI ChatGPT",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(
            required in tutorial_readme,
            f"tutorial README missing sample/provenance marker: {required}",
        )

    print("Standalone Mitra Colab tutorial: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
