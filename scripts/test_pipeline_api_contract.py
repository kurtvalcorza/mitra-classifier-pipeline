#!/usr/bin/env python3
"""Durable contract tests for the shared production-facing Mitra pipeline API."""
from __future__ import annotations

import json
import re
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

    check_pipeline_api_pin_parity(nb_text, companion_text)
    check_worker_partition_coverage_rule()
    print("Shared production-facing Mitra pipeline API contract: PASS")


def check_pipeline_api_pin_parity(nb_text: str, companion_text: str) -> None:
    """Worker<->tutorial parity is only durable if the module the notebooks fetch is the module
    the worker ships. Both notebooks download ``finetuner/pipeline_api.py`` from a pinned
    revision and refuse to import it unless its SHA-256 equals ``PIPELINE_API_SHA256``; the
    README states the same pin. Any edit to the shared module therefore has to re-pin both
    notebooks and the README in the same change, or the tutorials keep executing the old copy
    while the worker runs the new one. Hash LF-normalized bytes so a CRLF checkout matches the
    blob GitHub serves."""
    import hashlib

    api_bytes = (ROOT / "finetuner" / "pipeline_api.py").read_bytes().replace(b"\r\n", b"\n")
    api_sha = hashlib.sha256(api_bytes).hexdigest()
    sha_re = re.compile(r"PIPELINE_API_SHA256 = '([0-9a-f]{64})'")
    rev_re = re.compile(r"PIPELINE_API_REVISION = '([0-9a-f]{40})'")
    revisions: set[str] = set()
    for name, text in (("E2E notebook", nb_text), ("companion notebook", companion_text)):
        pins = set(sha_re.findall(text))
        assert pins == {api_sha}, (
            f"{name} PIPELINE_API_SHA256 {sorted(pins)} != sha256(LF finetuner/pipeline_api.py) "
            f"{api_sha}; re-pin PIPELINE_API_REVISION/PIPELINE_API_SHA256 in both notebooks and "
            "tutorials/README.md in the same change as the shared module"
        )
        found = rev_re.findall(text)
        assert len(set(found)) == 1, f"{name} must pin exactly one PIPELINE_API_REVISION, found {found}"
        revisions.update(found)
    assert len(revisions) == 1, f"notebooks pin different PIPELINE_API_REVISION values: {sorted(revisions)}"
    revision = revisions.pop()

    readme = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    heading = "### Production API parity"
    assert heading in readme, "tutorials/README.md lost the 'Production API parity' section"
    section = readme.split(heading, 1)[1].split("\n### ", 1)[0]
    assert f"immutable revision `{revision}`" in section, (
        f"tutorials/README.md 'Production API parity' does not cite notebook revision {revision}"
    )
    assert f"verify SHA-256 `{api_sha}`" in section, (
        f"tutorials/README.md 'Production API parity' does not cite sha256(LF finetuner/pipeline_api.py) {api_sha}"
    )


def check_worker_partition_coverage_rule() -> None:
    """The worker must accept every val/test partition the validator accepts. The validator's
    contract rule (TABULAR_CLASSIFICATION_DATASET_SPEC.md ``val_labels_subset_train`` /
    ``test_labels_subset_train``) is subset-only: a user-supplied partition may not contain a
    class train never saw, but it need not contain every training class -- an embargoed or
    chronological test window can legitimately lack a rare one. Round-4 review R4-P2-COVERAGE:
    at c1a7333 the worker rejected train {0,1,2} / test {0,1} at data preparation."""
    import tempfile
    import types

    if "requests" not in sys.modules:  # train.py imports it at module scope; CI does not install it
        try:
            import requests  # noqa: F401
        except ImportError:
            sys.modules["requests"] = types.ModuleType("requests")
    import train as worker  # noqa: E402

    def frame(classes: list[int], rows_per_class: int) -> pd.DataFrame:
        labels = [c for c in classes for _ in range(rows_per_class)]
        return pd.DataFrame({"x": range(len(labels)), "y": [v % 3 for v in range(len(labels))], "target": labels})

    def prepare(train_classes, val_classes, test_classes):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame(train_classes, 10).to_csv(root / "train.csv", index=False)
            frame(val_classes, 3).to_csv(root / "val.csv", index=False)
            if test_classes is not None:
                frame(test_classes, 3).to_csv(root / "test.csv", index=False)
            cfg = types.SimpleNamespace(
                target_column="target", drop_columns=[], validation_split=0.2, max_train_rows=10_000, seed=0
            )
            return worker._prepare_frames(cfg, worker.DatasetSource(root))

    # validator-accepted: holdout labels are a subset of train; a train class is absent from test
    train, val, test, num_classes = prepare([0, 1, 2], [0, 1, 2], [0, 1])
    assert num_classes == 3 and set(train.target.unique()) == {0, 1, 2}
    assert set(test.target.unique()) == {0, 1}, "worker must accept a test partition lacking a trained class"
    # ...and absent from val as well
    _, val, _, _ = prepare([0, 1, 2], [1, 2], [0, 1, 2])
    assert set(val.target.unique()) == {1, 2}, "worker must accept a val partition lacking a trained class"
    # validator-rejected: an unseen class must still fail, in both partitions
    expect_raises(lambda: prepare([0, 1, 2], [0, 1, 2], [0, 3]), ValueError)
    expect_raises(lambda: prepare([0, 1, 2], [0, 3], [0, 1, 2]), ValueError)


if __name__ == "__main__":
    main()
