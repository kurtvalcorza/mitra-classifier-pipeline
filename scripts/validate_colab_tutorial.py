#!/usr/bin/env python3
"""Static checks for the standalone Mitra Colab tutorial."""

from __future__ import annotations

import ast
import json
import re
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


def has_literal_assignment(
    code_cells: list[tuple[int, str]], name: str, expected: object
) -> bool:
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            targets = []
            value = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if value is None or not isinstance(value, ast.Constant):
                continue
            if type(value.value) is not type(expected) or value.value != expected:
                continue
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return True
    return False


def memory_guard_keys(code_cells: list[tuple[int, str]]) -> set[str]:
    keys: set[str] = set()
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
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
    return keys


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
        "Sample dataset (FreshRetailNet)",
        "Upload pre-split train/val/test",
        "freshretailnet-band-h7.zip",
        "DATASET_CARD.md",
        "train.csv",
        "val.csv",
        "test.csv",
        "test_data",
        "assert_resolver_locked",
        "torch_version",
        "fine_tune_steps_requested",
        "one-row accuracy resolution",
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

    code_text = "\n".join(source for _, source in parsed_code)
    require(
        re.search(r"\bDIMER_[A-Z0-9_]+\b", code_text) is None,
        "standalone tutorial must not depend on DIMER_* runtime variables",
    )

    for name, expected in (
        ("RUN_FINE_TUNING", False),
        ("RUN_NEW_DATA_INFERENCE", False),
        ("FINE_TUNE_STEPS", 50),
        ("MAX_MEMORY_USAGE_RATIO", 1.1),
        ("NETWORK_TIMEOUT_SECONDS", 30),
    ):
        require(
            has_literal_assignment(parsed_code, name, expected),
            f"tutorial must assign {name} the literal default {expected!r}",
        )

    require(
        has_memory_guard(parsed_code),
        "tutorial must pass max_memory_usage_ratio through a .fit(...) ag_args_fit keyword",
    )
    require(
        "ag.max_memory_usage_ratio" not in memory_guard_keys(parsed_code),
        "tutorial must not pass the prefixed ag.max_memory_usage_ratio key to .fit(...)",
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
        PINNED_REVISION,
        WEIGHTS_SHA256,
        CONFIG_SHA256,
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
            f"tutorial README missing sample/model/provenance marker: {required}",
        )

    print("Standalone Mitra Colab tutorial: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
