#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "_apply_notebook_spec_v1_v2.py"), run_name="__main__")

path = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    source = cell.get("source", "")
    text = "".join(source) if isinstance(source, list) else str(source)
    if "## 1. Install the matching runtime" not in text:
        continue
    if "autogluon.tabular[mitra]==1.5.0" not in text:
        text += "\nThe lock is rooted in the exact Mitra-capable requirement `autogluon.tabular[mitra]==1.5.0`; the committed lock then pins its resolved dependency graph.\n"
    cell["source"] = text.splitlines(keepends=True) if isinstance(source, list) else text
    break
else:
    raise RuntimeError("inference setup markdown cell not found")

path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Preserved explicit inference root requirement for CI and readers.")
