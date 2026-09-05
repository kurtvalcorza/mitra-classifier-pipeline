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

FORBIDDEN = (
    "mitra-classifier-finetuner",
    "mitra-classifier-dataset-validator",
    "DIMER_MODEL_DIR",
    "DIMER_DATASET_DIR",
    "DIMER_OUTPUT_DIR",
    "DIMER_HYPERPARAMETERS_JSON",
    "DIMER_PREPROCESSING_ARGS_JSON",
    '"ag.max_memory_usage_ratio"',
    "load_breast_cancer",
    "Built-in demo",
)


def main() -> int:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    cells = payload.get("cells", [])
    assert cells, "notebook has no cells"

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
        "autogluon.tabular[mitra]==1.5.0",
        "DIMER ZIP",
        "Pinned upstream",
        "RUN_FINE_TUNING = False",
        "RUN_NEW_DATA_INFERENCE = False",
        "MAX_MEMORY_USAGE_RATIO = 1.10",
        'ag_args_fit={"max_memory_usage_ratio": MAX_MEMORY_USAGE_RATIO}',
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
        assert required in text, f"missing required tutorial marker: {required}"

    for forbidden in FORBIDDEN:
        assert forbidden not in text, (
            f"standalone tutorial leaked forbidden dependency/configuration: {forbidden}"
        )

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
        if source.strip():
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{i}")

    root_readme = ROOT_README.read_text(encoding="utf-8")
    for required in (
        "## Try Mitra yourself in Google Colab",
        "tutorials/mitra_classifier_colab.ipynb",
        "colab.research.google.com",
        "OpenAI ChatGPT",
        "Agent Relay",
    ):
        assert required in root_readme, (
            f"root README missing tutorial/provenance marker: {required}"
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
        assert required in tutorial_readme, (
            f"tutorial README missing sample/provenance marker: {required}"
        )

    print("Standalone Mitra Colab tutorial: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
