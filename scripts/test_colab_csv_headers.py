"""CSV/header classification tests for Mitra Classifier public tutorial API and notebook wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitra_pipeline import read_csv_bytes, validate_inference_frame

NOTEBOOKS = (
    "mitra_classifier_colab.ipynb",
    "mitra_classifier_predictor_inference_colab.ipynb",
)

DUPLICATE_CASES = (
    b"amount,amount\n1,999\n",
    b'"sale,amount","sale,amount"\n1,999\n',
    b'\xef\xbb\xbfamount,"amount"\r\n1,999\r\n',
    b"\namount,amount\n1,999\n",
)


def notebook_text(name: str) -> str:
    payload = json.loads((ROOT / "tutorials" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in payload["cells"]
    )


class CsvHeaderTests(unittest.TestCase):
    def test_duplicate_raw_headers_rejected_by_public_api(self):
        for payload in DUPLICATE_CASES:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "duplicate column names"):
                    read_csv_bytes(payload, "input.csv")

    def test_valid_bom_and_quoted_headers_parse(self):
        payload = b'\xef\xbb\xbfamount.1,"sale,amount",amount\r\n7,9,1\r\n'
        frame = read_csv_bytes(payload, "input.csv")
        X, extra = validate_inference_frame(frame, ["amount", "sale,amount"])
        self.assertEqual(list(X.columns), ["amount", "sale,amount"])
        self.assertEqual(X.to_dict("list"), {"amount": [1], "sale,amount": [9]})
        self.assertEqual(extra, ["amount.1"])

    def test_both_notebooks_use_repository_csv_reader(self):
        # The standalone notebooks carry the package module, so the calls are unqualified kernel globals.
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                text = notebook_text(name)
                self.assertIn("read_csv_bytes(", text)
                self.assertIn("validate_inference_frame(", text)

    def test_main_presplit_upload_uses_repository_reader(self):
        text = notebook_text("mitra_classifier_colab.ipynb")
        for filename in ("train.csv", "val.csv", "test.csv"):
            self.assertIn(f"read_csv_bytes(payloads['{filename}'], '{filename}')", text)


if __name__ == "__main__":
    unittest.main()
