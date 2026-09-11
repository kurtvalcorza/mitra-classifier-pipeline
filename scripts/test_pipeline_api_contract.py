#!/usr/bin/env python3
"""Durable contract tests for the shared production-facing Mitra pipeline API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetuner"))
import pipeline_api as api  # noqa: E402


def expect_raises(fn, exc_type=Exception):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def main() -> None:
    import hashlib
    # canonical config source-of-truth
    assert api.BASE_MODEL == 'autogluon/mitra-classifier'
    assert api.PINNED_MITRA_REVISION == 'c425e9fa0910a6be1c494321792e7ba2a1367b1a'
    assert len(api.CANONICAL_CONFIG_BYTES) == 86
    assert hashlib.sha256(api.CANONICAL_CONFIG_BYTES).hexdigest() == api.EXPECTED_CONFIG_SHA256
    frame = pd.DataFrame({"x": range(20), "drop": range(20), "target": [0, 1] * 10})
    clean, features, mutations = api.prepare_tabular_frame(
        frame, "target", drop_columns=["drop", "target"], name="train.csv"
    )
    assert features == ["x"]
    assert mutations["dropped_feature_columns"] == ["drop"]
    assert api.validate_classification_target(clean, "target", min_rows=10, min_class_count=2) == ["x"]

    dup = pd.DataFrame([[1, 2, 0]], columns=["x", "x", "target"])
    expect_raises(lambda: api.prepare_tabular_frame(dup, "target"), ValueError)
    expect_raises(lambda: api.prepare_tabular_frame(frame, "missing"), ValueError)

    train, holdout = api.stratified_holdout(clean, "target", 0.2, 7)
    assert set(train.target.unique()) == {0, 1}
    assert set(holdout.target.unique()) == {0, 1}
    capped = api.stratified_cap(pd.concat([clean] * 3, ignore_index=True), "target", 20, 7)
    assert len(capped) == 20 and set(capped.target.unique()) == {0, 1}
    api.require_class_coverage(train, holdout, "target", "holdout")

    class FakePredictor:
        def evaluate(self, frame, auxiliary_metrics=True, silent=True):
            assert auxiliary_metrics is True and silent is True
            return {"accuracy": 0.5, "log_loss": -0.7}
        def predict(self, frame):
            return pd.Series([0] * len(frame))
        def predict_proba(self, frame, as_multiclass=False):
            assert as_multiclass is True
            return pd.DataFrame(np.tile([[0.6, 0.4]], (len(frame), 1)), columns=[0, 1])

    fake = FakePredictor()
    assert api.evaluate_mitra(fake, holdout)["accuracy"] == 0.5
    assert len(api.predict_mitra(fake, holdout)) == len(holdout)
    assert api.predict_mitra_proba(fake, holdout).shape == (len(holdout), 2)

    worker = (ROOT / "finetuner" / "train.py").read_text(encoding="utf-8")
    for marker in (
        "pipeline_materialize_dimer_checkpoint",
        "pipeline_prepare_tabular_frame",
        "pipeline_validate_classification_target",
        "pipeline_stratified_holdout",
        "pipeline_stratified_cap",
        "pipeline_fit_mitra_predictor",
        "pipeline_evaluate_mitra",
    ):
        assert marker in worker, f"worker is not delegated through {marker}"

    assert 'config_source = "platform-provided"' not in worker

    nb = json.loads((ROOT / "tutorials" / "mitra_classifier_colab.ipynb").read_text(encoding="utf-8"))
    nb_text = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source", []), list) else str(c.get("source", ""))
        for c in nb["cells"]
    )
    for marker in (
        "PIPELINE_API.prepare_tabular_frame",
        "PIPELINE_API.validate_classification_target",
        "PIPELINE_API.stratified_holdout",
        "PIPELINE_API.stratified_cap",
        "PIPELINE_API.fit_mitra_predictor",
        "PIPELINE_API.evaluate_mitra",
        "PIPELINE_API.predict_mitra",
        "PIPELINE_API.predict_mitra_proba",
    ):
        assert marker in nb_text, f"E2E notebook does not exercise {marker}"

    companion = json.loads((ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb").read_text(encoding="utf-8"))
    companion_text = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source", []), list) else str(c.get("source", ""))
        for c in companion["cells"]
    )
    assert "PIPELINE_API.predict_mitra" in companion_text
    assert "PIPELINE_API.predict_mitra_proba" in companion_text
    print("Shared production-facing Mitra pipeline API contract: PASS")


if __name__ == "__main__":
    main()
