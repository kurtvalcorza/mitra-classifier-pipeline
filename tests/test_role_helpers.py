"""Offline tests for the fleet snapshot helpers, the serving wrapper and the role-stage helpers.

No weights, no model: snapshot directories are temporary, the downloader is injected, the digest constants are
redirected to stand-in files where the happy path needs package constants and manifest to agree, and AutoGluon
is never imported.
"""

# ruff: noqa: E501  -- single-line fixtures and assertion messages
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mitra_pipeline import (
    CONFIG_FILE,
    DECISION_RULE,
    INPUT_SCHEMA,
    MANIFEST_NAME,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    METRIC_IDS,
    MIN_CLASSES,
    MIN_TRAIN_ROWS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PINNED_REVISION,
    WEIGHTS_FILE,
    MitraClassificationPipeline,
    align_probabilities,
    cap_training_rows,
    classification_metrics,
    evaluation_report,
    majority_class_baseline,
    require_class_coverage,
    stage_missing_files,
    stratified_holdout,
    validate_inference_frame,
    validate_inputs,
    validate_labeled_frame,
    verify_snapshot,
)
from mitra_pipeline import tutorial_api as api

WEIGHTS_PAYLOAD = b"not-the-real-safetensors"
CONFIG_PAYLOAD = b'{"dim": 1}'
WDIGEST = hashlib.sha256(WEIGHTS_PAYLOAD).hexdigest()
CDIGEST = hashlib.sha256(CONFIG_PAYLOAD).hexdigest()


def _manifest(wsha: str = WDIGEST, csha: str = CDIGEST) -> dict:
    return {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": WEIGHTS_FILE, "bytes": len(WEIGHTS_PAYLOAD), "sha256": wsha},
            {"path": CONFIG_FILE, "bytes": len(CONFIG_PAYLOAD), "sha256": csha},
        ],
        "totalBytes": len(WEIGHTS_PAYLOAD) + len(CONFIG_PAYLOAD),
    }


def _snapshot(tmp_path: Path, *, manifest: dict | None = None, write_files: bool = True) -> Path:
    root = tmp_path / "weights" / MODEL_KEY
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest or _manifest()), encoding="utf-8")
    if write_files:
        (root / WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD)
        (root / CONFIG_FILE).write_bytes(CONFIG_PAYLOAD)
    return root


@pytest.fixture
def constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "WEIGHTS_SHA256", WDIGEST)
    monkeypatch.setattr(api, "CONFIG_SHA256", CDIGEST)


def test_identity_constants_are_the_fleet_shape() -> None:
    assert MODEL_ID == "autogluon/mitra-classifier"
    assert MODEL_REVISION == PINNED_REVISION
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_LICENSE == "apache-2.0"
    assert api.DEFAULT_WEIGHTS_DIR.name == MODEL_KEY and api.DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert api.DEFAULT_WEIGHTS_DIR.parent.parent == Path(api.__file__).resolve().parents[1]  # root-level package


def test_committed_manifest_matches_the_digest_constants() -> None:
    manifest_path = api.DEFAULT_WEIGHTS_DIR / MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("snapshot manifest is not staged in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    digests = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert digests[WEIGHTS_FILE] == api.WEIGHTS_SHA256 and digests[CONFIG_FILE] == api.CONFIG_SHA256


def test_verify_snapshot_happy_path_and_rejections(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    result = verify_snapshot(root)
    assert result["path"] == str(root) and [f["path"] for f in result["files"]] == [WEIGHTS_FILE, CONFIG_FILE]
    (root / WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD[:-1] + b"!")  # same size, different bytes
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    with pytest.raises(ValueError, match="package constant"):
        verify_snapshot(_snapshot(tmp_path / "bad", manifest=_manifest(wsha="0" * 64)))
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(_snapshot(tmp_path / "foreign", manifest={**_manifest(), "modelId": "x/y"}))
    with pytest.raises(FileNotFoundError, match="snapshot file missing"):
        verify_snapshot(_snapshot(tmp_path / "missing", write_files=False))


def test_stage_missing_files_contract(tmp_path: Path, constants: None) -> None:
    assert stage_missing_files(_snapshot(tmp_path)) == []
    root = _snapshot(tmp_path / "empty", write_files=False)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, destination: Path) -> None:
        calls.append((relative_path, destination))
        payload = WEIGHTS_PAYLOAD if relative_path == WEIGHTS_FILE else CONFIG_PAYLOAD
        (destination / relative_path).write_bytes(payload)

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE, CONFIG_FILE]
    assert [c[0] for c in calls] == [WEIGHTS_FILE, CONFIG_FILE]
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(_snapshot(tmp_path / "rev", manifest={**_manifest(), "revision": "a" * 40}, write_files=False), allow_download=True, downloader=lambda *a: None)


def test_from_pretrained_verifies_then_stages_the_offline_hf_snapshot(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    staged: list[tuple[Path, Path, Path]] = []

    def fake_stage(weights_path, config_path, *, hf_home):
        staged.append((Path(weights_path), Path(config_path), Path(hf_home)))
        return Path(hf_home) / "snapshot"

    api_stage = api.stage_verified_hf_snapshot
    api.stage_verified_hf_snapshot = fake_stage
    try:
        pipe = MitraClassificationPipeline.from_pretrained(weights_dir=root, device="cpu")
    finally:
        api.stage_verified_hf_snapshot = api_stage
    assert staged == [(root / WEIGHTS_FILE, root / CONFIG_FILE, root / ".cache" / "hf")]
    assert pipe.model_weight_path == root / WEIGHTS_FILE and pipe.source == "local-snapshot" and pipe.device == "cpu"
    assert pipe.predictor is None
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.predict(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(FileNotFoundError):
        MitraClassificationPipeline.from_pretrained(weights_dir=_snapshot(tmp_path / "u", write_files=False), device="cpu")


def _table(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x1": rng.normal(size=rows), "city": rng.choice(["a", "b"], size=rows), "target": rng.choice(["no", "yes"], size=rows, p=[0.4, 0.6])})


def test_validate_inputs_fit_mode_matches_validate_labeled_frame() -> None:
    frame = _table()
    frame.loc[0, "target"] = None
    manifest = validate_inputs(frame, target_column="target", names=["sample"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["train_rows"] == [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS]
    assert manifest["schema"]["features"] == [1, MAX_FEATURES]
    assert manifest["schema"]["classes"] == [MIN_CLASSES, MAX_CLASSES]
    assert manifest["schema"]["decision_rule"] == DECISION_RULE == "argmax"
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample" and entry["mode"] == "fit" and entry["rows"] == 59
    assert entry["feature_columns"] == ["x1", "city"] and entry["categorical_columns"] == ["city"]
    assert entry["report"]["dropped_missing_target_rows"] == 1
    assert entry["classes"] == ["no", "yes"] and sum(entry["class_counts"].values()) == 59
    assert 0.5 <= entry["majority_class_share"] < 1.0
    _clean, _features, report = validate_labeled_frame(frame, "target", name="sample", min_rows=MIN_TRAIN_ROWS, min_classes=MIN_CLASSES, min_class_count=2)
    assert entry["report"] == report
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    for bad, message in (
        (_table().rename(columns={"target": "y"}), "target 'target' not found"),
        (_table(10), "at least 50 labelled rows"),
        (_table().assign(target="x"), "target has 1 classes"),
        (_table().assign(target=[str(i) for i in range(60)]), "target has 60 classes"),
        (pd.concat([_table(), _table()[["x1"]]], axis=1), "duplicate column names"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, target_column="target")
        with pytest.raises(ValueError, match=message):
            validate_labeled_frame(bad, "target", name="table-0", min_rows=MIN_TRAIN_ROWS, min_classes=MIN_CLASSES, min_class_count=2)
    rare = _table()
    rare.loc[rare.index[:59], "target"] = "no"
    with pytest.raises(ValueError, match="every class needs at least 2 row"):
        validate_inputs(rare, target_column="target")
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_table(), names=["a", "b"])


def test_class_coverage_split_and_cap_helpers() -> None:
    frame = _table(100)
    train, holdout = stratified_holdout(frame, "target", 0.2, 42)
    assert set(train["target"]) == set(holdout["target"]) == {"no", "yes"}
    assert len(train) + len(holdout) == 100
    require_class_coverage(train, holdout, "target", "holdout")
    with pytest.raises(ValueError, match="unseen target classes"):
        require_class_coverage(train, holdout.assign(target="maybe"), "target", "holdout")
    with pytest.raises(ValueError, match="missing trained target classes"):
        require_class_coverage(train, holdout[holdout["target"] == "yes"], "target", "holdout")
    with pytest.raises(ValueError, match="fewer than 2 rows"):
        stratified_holdout(pd.DataFrame({"x1": [1, 2, 3], "target": ["a", "a", "b"]}), "target", 0.3, 1)
    capped, report = cap_training_rows(frame, "target", seed=1, max_rows=20)
    assert len(capped) == 20 and report == {"applied": True, "before": 100, "after": 20}
    assert set(capped["target"]) == {"no", "yes"}
    same, report = cap_training_rows(frame, "target", seed=1)
    assert len(same) == 100 and report["applied"] is False
    with pytest.raises(ValueError, match="cannot exceed"):
        cap_training_rows(frame, "target", seed=1, max_rows=MAX_TRAIN_ROWS + 1)


def test_validate_inputs_inference_mode_matches_validate_inference_frame() -> None:
    rows = _table(3).drop(columns=["target"]).assign(extra=1)
    manifest = validate_inputs(rows, None, feature_columns=["x1", "city"], names=["new"])
    (entry,) = manifest["inputs"]
    assert entry == {"id": "new", "mode": "inference", "rows": 3, "feature_columns": ["x1", "city"], "extra_columns": ["extra"], "missing_value_columns": {}}
    for bad, message in (
        (rows.drop(columns=["city"]), "missing required feature columns"),
        (rows.assign(prediction="yes"), "output column\\(s\\) reserved by this notebook"),
        (rows.assign(probability_yes=0.5), "output column\\(s\\) reserved by this notebook"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, None, feature_columns=["x1", "city"])
        with pytest.raises(ValueError, match=message):
            validate_inference_frame(bad, ["x1", "city"])
    with pytest.raises(ValueError, match="feature_columns is required"):
        validate_inputs(rows, None)


def test_majority_class_baseline_metrics_alignment_and_evaluation_report() -> None:
    baseline = majority_class_baseline(["yes", "yes", "no"], ["yes", "no"])
    assert baseline["accuracy"] == pytest.approx(0.5) and np.isnan(baseline["roc_auc"])
    assert set(baseline) <= set(METRIC_IDS)
    with pytest.raises(ValueError, match="unseen in training"):
        majority_class_baseline(["yes", "no"], ["maybe"])
    y = ["no", "yes", "yes", "no"]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    metrics = classification_metrics(y, ["no", "yes", "yes", "no"], proba, ["no", "yes"])
    assert metrics["accuracy"] == 1.0 and 0.0 < metrics["log_loss"] < 1.0 and metrics["roc_auc"] == 1.0
    assert {"balanced_accuracy", "f1_macro", "mcc"} <= set(metrics)
    multi = classification_metrics(["a", "b", "c"], ["a", "b", "b"], np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.5, 0.3]]), ["a", "b", "c"])
    assert multi["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="one column per class"):
        classification_metrics(y, y, proba[:, :1], ["no", "yes"])
    assert align_probabilities(np.array([[0.2, 0.8]]), ["yes", "no"], ["no", "yes"]).tolist() == [[0.8, 0.2]]
    with pytest.raises(RuntimeError, match="never saw classes"):
        align_probabilities(np.array([[1.0]]), ["yes"], ["no", "yes"])
    report = evaluation_report(metrics, baseline=baseline, independent_test=metrics, n_holdout=4, n_test=4, class_labels=["no", "yes"], target_column="target", selection="default:pretrained", sample_kind="sample", estimation="e")
    assert report["verdict"] == "sample-sanity" and report["decision_rule"] == "argmax"
    assert [m["id"] for m in report["metrics"]] == [m for m in METRIC_IDS if m in metrics]
    assert all(m["estimation"] == "e" for m in report["metrics"])
    assert report["baselines"][0]["id"] == "majority_class" and report["selection"] == "default:pretrained"
    flags = {m["id"]: m["higher_is_better"] for m in report["metrics"]}
    assert flags["log_loss"] is False and flags["accuracy"] is True
    nan = evaluation_report({"accuracy": 1.0, "roc_auc": float("nan")}, n_holdout=1)
    assert nan["metrics"][1]["value"] is None
    autogluon_like = evaluation_report({"accuracy": 0.9, "balanced_accuracy": 0.8, "mcc": 0.5, "f1": 0.7, "precision": 0.6, "recall": 0.9, "roc_auc": 0.95, "log_loss": 0.3}, n_holdout=10)
    assert len(autogluon_like["metrics"]) == 8
    missing = evaluation_report(None, n_holdout=0, target_column="target", sample_kind="BYOD", class_labels=["no", "yes"])
    assert missing["verdict"] == "not-measurable" and "majority_class_baseline" in missing["needs"]
    assert missing["class_labels"] == ["no", "yes"]
    assert (missing["model_id"], missing["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"mae": 0.5})
