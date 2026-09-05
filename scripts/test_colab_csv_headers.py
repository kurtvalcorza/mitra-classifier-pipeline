"""Exercise the real notebook CSV input cells without Colab or model fitting."""

import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ("mitra_classifier_colab.ipynb", 10),
    ("mitra_classifier_predictor_inference_colab.ipynb", 8),
)


def read_input_cell(name, index, payload, features):
    notebook = json.loads((ROOT / "tutorials" / name).read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][index]["source"]))
    # Enable the optional training-notebook inference path, stopping immediately
    # before model execution. Keep imports, upload, parsing and validation intact.
    if index == 10:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RUN_NEW_DATA_INFERENCE"
                for t in node.targets
            ):
                node.value = ast.Constant(True)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                end = next(
                    i for i, stmt in enumerate(node.body)
                    if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "X" for t in stmt.targets
                    )
                )
                node.body = node.body[:end + 1]
    files = types.SimpleNamespace(upload=lambda: {"input.csv": payload})
    colab = types.ModuleType("google.colab")
    colab.files = files
    namespace = {
        "pd": pd, "files": files, "FEATURE_COLUMNS": features,
        "FIT_RUN_COMPLETED": True, "baseline_predictor": object(),
        "display": lambda *args: None, "print": lambda *args: None,
    }
    with patch.dict("sys.modules", {"google.colab": colab}):
        exec(compile(ast.fix_missing_locations(tree), name, "exec"), namespace)
    return namespace["X"]


class CsvHeaderTests(unittest.TestCase):
    def test_duplicate_headers_rejected(self):
        for name, index in NOTEBOOKS:
            for payload, features in (
                (b"amount,amount\n1,999\n", ["amount"]),
                (b'"sale,amount","sale,amount"\n1,999\n', ["sale,amount"]),
                (b'\xef\xbb\xbfamount,"amount"\r\n1,999\r\n', ["amount"]),
                (b'\namount,amount\n1,999\n', ["amount"]),
            ):
                with self.subTest(notebook=name, payload=payload):
                    with self.assertRaisesRegex(ValueError, "duplicate column names"):
                        read_input_cell(name, index, payload, features)

    def test_valid_headers_and_feature_order_preserved(self):
        for name, index in NOTEBOOKS:
            for payload in (
                b'amount.1,"sale,amount",amount\n7,9,1\n',
                b'\xef\xbb\xbfamount.1,"sale,amount",amount\r\n7,9,1\r\n',
            ):
                with self.subTest(notebook=name, payload=payload):
                    frame = read_input_cell(name, index, payload, ["amount", "sale,amount"])
                    self.assertEqual(frame.to_dict("list"), {"amount": [1], "sale,amount": [9]})


if __name__ == "__main__":
    unittest.main()
