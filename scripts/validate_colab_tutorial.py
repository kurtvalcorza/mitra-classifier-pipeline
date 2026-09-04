#!/usr/bin/env python3
"""Static checks for the standalone Mitra Colab tutorial."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"

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
)


def main() -> int:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    cells = payload.get("cells", [])
    assert cells, "notebook has no cells"

    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    for required in (
        PINNED_REVISION,
        WEIGHTS_SHA256,
        CONFIG_SHA256,
        "autogluon.tabular[mitra]==1.5.0",
        "DIMER ZIP",
        "Pinned upstream",
        "RUN_FINE_TUNING",
        "RUN_NEW_DATA_INFERENCE",
    ):
        assert required in text, f"missing required tutorial marker: {required}"

    for forbidden in FORBIDDEN:
        assert forbidden not in text, f"standalone tutorial leaked Workbench dependency: {forbidden}"

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        if source.strip():
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{i}")

    print("Standalone Mitra Colab tutorial: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
