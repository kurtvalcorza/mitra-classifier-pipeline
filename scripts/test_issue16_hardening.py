"""Regression checks for the tutorial hardening tracked in issue #16, restated for the standalone carrier
(NOTEBOOK_SPEC 1.1 §3.6): the CSV reader and the selection/verification invariants now live in the carried
`mitra_pipeline/tutorial_api.py` module and the generated notebooks."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"


def notebook_sources(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        yield "".join(source) if isinstance(source, list) else str(source)


def notebook_text(path):
    return "\n".join(notebook_sources(path))


class Issue16HardeningTests(unittest.TestCase):
    def test_training_reader_rejects_duplicate_target_header(self):
        from mitra_pipeline import read_csv_bytes

        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            read_csv_bytes(b"feature,target,target\n1,a,a\n", "uploaded CSV")

    def test_main_notebook_selects_recommended_predictor(self):
        text = notebook_text(MAIN)
        self.assertIn("MIN_SELECTION_HOLDOUT_ROWS = 50", text)
        self.assertIn("ACTIVE_MODEL, ACTIVE_MODE, SELECTION_BASIS = pipe, 'pretrained', 'default:pretrained'", text)
        self.assertIn("SELECTION_BASIS = f'holdout:{EVAL_METRIC}'", text)
        self.assertIn("'selection_basis': SELECTION_BASIS", text)
        self.assertIn("(evidence only; never used for selection)", text)

    def test_companion_verifies_archive_before_load(self):
        text = notebook_text(INFERENCE)
        self.assertIn("EXPECTED_ZIP_SHA256", text)
        self.assertIn("checksum mismatch", text)
        self.assertLess(text.index("checksum mismatch"), text.index("predictor = TabularPredictor.load(str(EXTRACT_ROOT))"))
        self.assertLess(text.index("validate_artifact_directory(EXTRACT_ROOT)"), text.index("predictor = TabularPredictor.load(str(EXTRACT_ROOT))"))
        self.assertIn("Trust boundary", text)

    def test_companion_rejects_wrong_problem_type_and_output_collisions(self):
        from mitra_pipeline import validate_inference_frame
        import pandas as pd

        text = notebook_text(INFERENCE)
        self.assertIn("if predictor.problem_type not in {'binary', 'multiclass'}:", text)
        with self.assertRaisesRegex(ValueError, "output column\\(s\\) reserved by this notebook"):
            validate_inference_frame(pd.DataFrame({"x": [1], "probability_yes": [0.5]}), ["x"])
        with self.assertRaisesRegex(ValueError, "output column\\(s\\) reserved by this notebook"):
            validate_inference_frame(pd.DataFrame({"x": [1], "prediction": ["yes"]}), ["x"])

    def test_autogluon_version_spelling_is_consistent(self):
        for path in (MAIN, INFERENCE):
            text = notebook_text(path)
            self.assertIn("importlib.metadata.version('autogluon.tabular')", text)
            self.assertNotIn("AUTOGLOUON_VERSION", text)


if __name__ == "__main__":
    unittest.main()
