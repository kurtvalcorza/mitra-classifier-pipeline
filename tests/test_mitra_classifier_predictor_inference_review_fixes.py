"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on
mitra_classifier_predictor_inference_colab (MCP-B1, MCP-M1, MCP-M2, MCP-m1..m4;
docs/reviews/2026-10-02-notebook-review/mitra_classifier_predictor_inference_colab_Review.md).

CI's dependencies only: the notebook's own cells run with the carried module's pure-pandas helpers and stand-ins for
the download, extraction and AutoGluon; no network, no torch.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"
E2E = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
PINNED_URL = "https://github.com/kurtvalcorza/mitra-classifier-pipeline/releases/download/sample-bundle-v1/mitra_classifier_predictor.zip"
PINNED_SHA = "d2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01"
# DATA_DIGEST printed by the producing run (E2E notebook blob 2748c0a2, 2026-10-08 Colab T4) for the default sample.
PRODUCER_DATA_SHA256 = "09d9b60e54c33916e4bea6f888a24958e0014fe8e8ac0a9ab301e46d0c0d2ffb"


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path: Path, marker: str) -> str:
    found = [c["source"] for c in _cells(path) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(path: Path) -> str:
    return "\n".join(c["source"] for c in _cells(path) if c["cell_type"] == "markdown")


def _set(source: str, name: str, value) -> str:
    pattern = re.compile(rf"^{name} = .*?(  # @param.*)?$", re.M)
    assert pattern.search(source), name
    return pattern.sub(lambda m: f"{name} = {value!r}" + (m.group(1) or ""), source, count=1)


def _api():
    spec = importlib.util.spec_from_file_location("_mcp_tutorial_api", ROOT / "mitra_pipeline" / "tutorial_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- MCP-B1: the default path downloads the pinned sample bundle and verifies it before extraction --------------------


def test_mcp_b1_sample_artifact_is_pinned_by_release_url_sha256_and_producer():
    source = _code(ART, "SAMPLE_ARTIFACT = {")
    pinned = ast.literal_eval(re.search(r"^SAMPLE_ARTIFACT = (\{.*\})$", source, re.M).group(1))
    assert pinned["url"] == PINNED_URL and pinned["sha256"] == PINNED_SHA
    producer = pinned["producer"]
    assert producer["notebook"] == "tutorials/mitra_classifier_colab.ipynb" and producer["release"] == "sample-bundle-v1"
    assert re.fullmatch(r"[0-9a-f]{40}", producer["notebook_blob"]) and re.fullmatch(r"[0-9a-f]{40}", producer["commit"])
    assert "Colab" in producer["run"] and producer["evidence"].startswith("docs/execution-evidence/")
    assert "ARTIFACT_ZIP_PATH = ''  # @param" in source  # the location field defaults to the pinned asset


def _run_section_4(tmp_path, monkeypatch, payload: bytes, *, pinned_sha: str | None = None, **fields):
    monkeypatch.chdir(tmp_path)
    source = _code(ART, "SAMPLE_ARTIFACT = {")
    if pinned_sha is not None:
        source = source.replace(PINNED_SHA, pinned_sha)
    for name, value in fields.items():
        source = _set(source, name, value)
    calls = {"download": [], "extract": []}

    def _download(url, destination, *, timeout=30):
        calls["download"].append(url)
        Path(destination).write_bytes(payload)

    def safe_extract_archive(zip_path, root):
        calls["extract"].append(str(zip_path))

    metadata = {"base_model": "m", "base_model_revision": "r", "weights_sha256": "w", "config_sha256": "c", "features": ["a"], "target_column": "target",
                "problem_type": "binary", "mode": "pretrained", "selection_basis": "default:pretrained", "autogluon_version": "1.5.0", "python_version": "3.12.12"}
    ns = {"os": os, "Path": Path, "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(), "_download": _download,
          "safe_extract_archive": safe_extract_archive, "validate_artifact_directory": lambda root: ({"artifact_format": "f", "artifact_format_version": 1}, metadata),
          "MODEL_ID": "m", "MODEL_REVISION": "r", "WEIGHTS_SHA256": "w", "CONFIG_SHA256": "c"}
    exec(compile(source, "section4", "exec"), ns)
    return ns, calls


def test_mcp_b1_default_path_downloads_verifies_then_extracts(tmp_path, monkeypatch):
    payload = b"PK sample bundle"
    ns, calls = _run_section_4(tmp_path, monkeypatch, payload, pinned_sha=hashlib.sha256(payload).hexdigest())
    assert calls["download"] == [PINNED_URL] and len(calls["extract"]) == 1
    assert ns["artifact_source"] == f"sample artifact: {PINNED_URL}" and ns["expected_digest"] == ns["zip_sha256"]
    # A complete earlier download is reused, not fetched again.
    ns, calls = _run_section_4(tmp_path, monkeypatch, payload, pinned_sha=hashlib.sha256(payload).hexdigest())
    assert calls["download"] == [] and len(calls["extract"]) == 1


def test_mcp_b1_a_substituted_sample_fails_before_extraction(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _run_section_4(tmp_path, monkeypatch, b"tampered", pinned_sha=hashlib.sha256(b"original").hexdigest())
    assert not (tmp_path / "external-artifact" / "bundle").exists()


def test_mcp_m4_upload_without_colab_names_the_path_field(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="needs the Colab upload dialog.*set ARTIFACT_ZIP_PATH to its path"):
        _run_section_4(tmp_path, monkeypatch, b"x", ARTIFACT_ZIP_PATH="upload")
    with pytest.raises(FileNotFoundError, match="ARTIFACT_ZIP_PATH 'missing.zip' is not a file"):
        _run_section_4(tmp_path, monkeypatch, b"x", ARTIFACT_ZIP_PATH="missing.zip")
    prerequisites = _markdown(ART)
    assert "Kaggle dataset, Google Drive or `gsutil` copy" in prerequisites


# --- MCP-B1 sample input / MCP-M1 / MCP-m3: Section 6 ------------------------------------------------------------------


def _run_section_6(monkeypatch, **fields):
    api = _api()
    source = _code(ART, "NEW_DATA_PATH = ''  # @param")
    for name, value in fields.items():
        source = _set(source, name, value)
    features = [c for c in __import__("sklearn.datasets", fromlist=["x"]).load_breast_cancer(as_frame=True).data.columns]
    ns = {"os": os, "Path": Path, "json": json, "TARGET_COLUMN": "target", "FEATURE_COLUMNS": features,
          "run_metadata": {"data_source": "Sample: Breast Cancer", "seed": 42, "data_sha256": PRODUCER_DATA_SHA256, "class_labels": ["benign", "malignant"]},
          "read_csv_bytes": api.read_csv_bytes, "validate_inputs": api.validate_inputs, "validate_inference_frame": api.validate_inference_frame}
    os.makedirs("outputs", exist_ok=True)
    exec(compile(source, "section6", "exec"), ns)
    return ns


def test_mcp_b1_sample_input_is_the_producers_independent_test_partition(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _run_section_6(monkeypatch)
    out = capsys.readouterr().out
    assert ns["sample_kind"] == "sample" and len(ns["new_data"]) == 114 and "target" not in ns["new_data"].columns
    assert out.index("'inference_limits'") < out.index("'sample_input'")
    # Same rows as the E2E notebook's stratified_60_20_20 test partition (the bundle's support rows are disjoint).
    e2e = _code(E2E, "def stratified_60_20_20(")
    helper = e2e[e2e.index("def stratified_60_20_20(") : e2e.index("def byod_payloads(")]
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    e2e_ns = {"train_test_split": train_test_split, "SEED": 42, "TARGET_COLUMN": "target"}
    exec(helper, e2e_ns)
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={dataset.target.name: "target"})
    frame["target"] = frame["target"].map({0: "malignant", 1: "benign"})
    train, _holdout, test = e2e_ns["stratified_60_20_20"](frame)
    pd = __import__("pandas")
    pd.testing.assert_frame_equal(ns["new_data"].reset_index(drop=True), test.drop(columns=["target"]).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12)
    assert not set(test.index) & set(train.index)


def test_mcp_b1_sample_input_refuses_a_different_table_or_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = _api()
    source = _code(ART, "NEW_DATA_PATH = ''  # @param")
    base = {"os": os, "Path": Path, "json": json, "TARGET_COLUMN": "target", "FEATURE_COLUMNS": ["a"], "read_csv_bytes": api.read_csv_bytes,
            "validate_inputs": api.validate_inputs, "validate_inference_frame": api.validate_inference_frame}
    with pytest.raises(RuntimeError, match="fits only a bundle trained on 'Sample: Breast Cancer'"):
        exec(compile(source, "s6", "exec"), {**base, "run_metadata": {"data_source": "Upload CSV", "seed": 42, "class_labels": ["x"]}})
    with pytest.raises(RuntimeError, match="is not the table the bundle was produced from"):
        exec(compile(source, "s6", "exec"), {**base, "run_metadata": {"data_source": "Sample: Breast Cancer", "seed": 42, "data_sha256": "0" * 64, "class_labels": ["x"]}})


def test_mcp_m1_rows_by_upload_work_when_the_bundle_came_by_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sklearn.datasets import load_breast_cancer

    rows = load_breast_cancer(as_frame=True).data.head(5).to_csv(index=False).encode("utf-8")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"mine.csv": rows})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = _run_section_6(monkeypatch, NEW_DATA_PATH="upload")  # raised NameError: files on the reviewed blob
    assert ns["csv_name"] == "mine.csv" and len(ns["new_data"]) == 5 and ns["sample_kind"] == "BYOD"


# --- MCP-m1 / m2 / M2 ---------------------------------------------------------------------------------------------------


def test_mcp_m1_no_regression_wording():
    assert "continuous class labels" not in ART.read_text(encoding="utf-8")


def test_mcp_m2_prose_states_section_3_is_not_needed_for_prediction():
    markdown = _markdown(ART)
    assert "Prediction needs only the bundle" in markdown and "proves that the pin the bundle names still resolves" in markdown


def test_mcp_m2_in_notebook_activity_runs_on_the_default_sample_with_one_field():
    source = _code(ART, "ACTIVITY_SCALE = ")
    assert "ACTIVITY_FEATURE = 'worst area'  # @param" in source and "ACTIVITY_SCALE = 2.0  # @param" in source
    assert ".fit(" not in source and "to_csv(" not in source  # changes nothing above it
    markdown = _markdown(ART)
    assert "## 8. Activity: change one feature" in markdown and "*Change* `ACTIVITY_SCALE` to `0.5` (one field)" in markdown
    assert markdown.count("<details><summary>Check your reasoning</summary>") == 4


# --- ST1 exemption (fleet pattern, language-model-pipeline #78) ---------------------------------------------------------


def test_st1_allows_only_the_pinned_sample_bundle_asset_url():
    spec = importlib.util.spec_from_file_location("_validator", ROOT / "tools" / "validate_release_assets.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    st1 = dict(validator.FORBIDDEN_PATTERNS)["repository clone (ST1)"]
    assert not st1.search(f"'{PINNED_URL}'")
    for bad in ("'https://github.com/kurtvalcorza/mitra-classifier-pipeline.git'", "'https://github.com/kurtvalcorza/mitra-classifier-pipeline/archive/main.zip'",
                "'https://github.com/kurtvalcorza/mitra-classifier-pipeline/releases/download/sample-bundle-v1/x.py'", "'https://github.com/kurtvalcorza/other-repo/releases/download/sample-bundle-v1/x.zip'"):
        assert st1.search(bad), bad
