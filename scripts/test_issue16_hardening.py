"""Regression checks for tutorial hardening tracked in issue #16."""

import ast
import csv
import io
import json
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"


def notebook_sources(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        yield "".join(source) if isinstance(source, list) else str(source)


def notebook_text(path):
    return "\n".join(notebook_sources(path))


def uploaded_csv_reader():
    for source in notebook_sources(MAIN):
        if "def uploaded_csv" not in source:
            continue
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "uploaded_csv"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"csv": csv, "io": io, "pd": pd}
        exec(compile(ast.fix_missing_locations(module), MAIN.name, "exec"), namespace)
        return namespace["uploaded_csv"]
    raise AssertionError("Could not locate uploaded_csv")


class Issue16HardeningTests(unittest.TestCase):
    def test_training_reader_rejects_duplicate_target_header(self):
        reader = uploaded_csv_reader()
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            reader(b"feature,target,target\n1,a,a\n", "uploaded CSV")

    def test_main_notebook_selects_recommended_predictor(self):
        text = notebook_text(MAIN)
        self.assertIn("MIN_SELECTION_HOLDOUT_ROWS = 50", text)
        self.assertIn("recommended_predictor = baseline_predictor", text)
        self.assertIn("selection_basis = f'holdout:{EVAL_METRIC}'", text)
        self.assertIn("active_predictor = recommended_predictor", text)
        self.assertIn("'selection_basis': selection_basis", text)
        self.assertNotIn("active_predictor = finetuned_predictor or baseline_predictor", text)

    def test_companion_verifies_archive_before_load(self):
        text = notebook_text(INFERENCE)
        self.assertIn("EXPECTED_ZIP_SHA256", text)
        self.assertIn("Predictor ZIP checksum mismatch", text)
        self.assertLess(text.index("Predictor ZIP checksum mismatch"), text.index("TabularPredictor.load"))
        self.assertIn("Trust boundary", text)

    def test_companion_rejects_wrong_problem_type_and_output_collisions(self):
        text = notebook_text(INFERENCE)
        self.assertIn("predictor.problem_type not in {'binary', 'multiclass'}", text)
        self.assertIn("reserved_output_columns", text)
        self.assertIn("output column(s) reserved by this notebook", text)

    def test_autogluon_version_spelling_is_consistent(self):
        for path in (MAIN, INFERENCE):
            text = notebook_text(path)
            self.assertIn("AUTOGLUON_VERSION", text)
            self.assertNotIn("AUTOGLOUON_VERSION", text)


if __name__ == "__main__":
    unittest.main()
