"""CPU integration with real snapshots/logistic exports and an explicit synthetic Mitra stub."""

import importlib
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
runtime = importlib.import_module("customer_runtime")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    raw = []
    for cutoff in runtime.DEFAULT["train"] + runtime.DEFAULT["dev"] + runtime.DEFAULT["test"]:
        t = pd.Timestamp(cutoff)
        for i in range(40):
            for offset in [-10 - i % 7, 5] if i < 20 else [-10 - i % 7]:
                raw.append(
                    dict(
                        invoice_id=f"{cutoff}-{i}-{offset}",
                        customer_id=str(i),
                        stock_code="10001",
                        quantity=1,
                        invoice_time=t + pd.Timedelta(days=offset),
                        unit_price=1 + i,
                        country="fixture",
                    )
                )
    frame = pd.DataFrame(raw)
    monkeypatch.setattr(
        runtime.data, "prepare", lambda root: (frame.copy(), {"fixture": True, "rows": len(frame)})
    )
    monkeypatch.setattr(runtime.models, "load_mitra", lambda *args: object())
    # Numeric deterministic test double; not a claim about pretrained Mitra behavior.
    monkeypatch.setattr(
        runtime.models,
        "mitra_predict",
        lambda model, support, labels, query: 1 / (1 + np.exp(np.asarray(query)[:, 0] / 30 - 1)),
    )
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    actual_version = runtime.importlib.metadata.version
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: "synthetic-fixture" if name in ("torch", "autogluon.tabular") else actual_version(name),
    )
    root = tmp_path / "run"
    root.mkdir()
    runtime.write(root / "model_manifest.json", {"revision": runtime.models.REVISION})
    for name in (
        "data_manifest.json",
        "DATA_LICENSE.md",
        "requirements.txt",
        "customer_data.py",
        "customer_models.py",
        "customer_runtime.py",
        "customer_metrics.py",
        "customer_byod.py",
        "RECONSTRUCT.md",
    ):
        (root / name).write_text("fixture\n", encoding="utf-8")
    runtime.write(root / "source.json", {"files": {"DATA_LICENSE.md": runtime.sha(root / "DATA_LICENSE.md")}})
    return root


def through(root, stage, monkeypatch):
    for i, name in enumerate(runtime.STAGES[: runtime.STAGES.index(stage) + 1]):
        monkeypatch.setattr(runtime.os, "getpid", lambda i=i: 20000 + i)
        runtime.run_stage(root, name)


def test_full_cpu_pipeline_snapshots_real_logistic_and_stub_mitra(experiment, monkeypatch):
    wall = dict(
        elapsed_seconds_including_setup_before_report=123.0,
        initial_free_disk_bytes=21 * 1024**3,
        gpu_name="synthetic fixture; no GPU used",
    )
    runtime.write(experiment / "wall_resources.json", wall)
    through(experiment, "report", monkeypatch)
    out = runtime.output(experiment)
    summary = runtime.read(out / "run_summary.json")
    assert summary["verification"]["passed"] and summary["verification"]["different_process"]
    assert summary["verification"]["replayed_rows"] == 40
    assert summary["wall_resources_before_report"] == wall
    assert summary["within_target_runtime_including_setup_before_report"] is True
    assert runtime.read(out / "wall_resources.json") == wall
    assert len(runtime.csv(experiment, "test_predictions.csv")) == 40 * 2 * 6
    final = runtime.csv(experiment, "inference_predictions.csv")
    assert "y_true" not in final and set(final.evaluation_status) == {"not_measurable"}
    assert runtime.read(out / "paired_intervals.json")["replicates"] == 2000
    assert (out / "test_comparison.png").stat().st_size > 1000
    with zipfile.ZipFile(out / "results.zip") as archive:
        assert archive.testzip() is None
        assert not any("transactions" in name for name in archive.namelist())
    for role in ("dev", "test"):
        expected = runtime.metrics.evaluate(runtime.prediction_rows(experiment, role))
        assert expected == runtime.read(out / f"{role}_metrics.json")


def test_receipt_tamper_refused_before_next_stage(experiment, monkeypatch):
    through(experiment, "prepare", monkeypatch)
    target = runtime.output(experiment) / "train_snapshots.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Stage output changed"):
        runtime.run_stage(experiment, "fit")


def test_frozen_inputs_tamper_refused(experiment, monkeypatch):
    through(experiment, "freeze", monkeypatch)
    target = runtime.output(experiment) / "test_snapshots.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        runtime.check_lock(experiment)


def test_earlier_rerun_invalidates_downstream_and_retains_output_ownership(experiment, monkeypatch):
    through(experiment, "freeze", monkeypatch)
    initial = runtime.read(runtime.output(experiment) / "receipt_prepare.json")["outputs"]
    runtime.run_stage(experiment, "prepare")
    repeat = runtime.read(runtime.output(experiment) / "receipt_prepare.json")["outputs"]
    assert set(initial) == set(repeat)
    assert not runtime.read(runtime.output(experiment) / "receipt_fit.json")["completed"]
    assert not runtime.read(runtime.output(experiment) / "receipt_freeze.json")["completed"]
    with pytest.raises(ValueError, match="Stale stage receipt"):
        runtime.run_stage(experiment, "evaluate")


def test_csv_recomputation_detects_changed_probabilities(experiment, monkeypatch):
    through(experiment, "verify", monkeypatch)
    path = runtime.output(experiment) / "test_predictions.csv"
    frame = pd.read_csv(path)
    index = frame.index[frame.system == "prevalence"][0]
    frame.loc[index, "probability"] = 0.01
    runtime.save_csv(path, frame)
    with pytest.raises(ValueError, match="CSV/metrics recomputation failed"):
        runtime.report(experiment)


def test_source_change_refused(experiment):
    (experiment / "DATA_LICENSE.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Embedded source integrity"):
        runtime.identity(experiment)


def test_report_recomputes_paired_intervals_from_full_csv(experiment, monkeypatch):
    through(experiment, "verify", monkeypatch)
    path = runtime.output(experiment) / "paired_intervals.json"
    summary = runtime.read(path)
    name = next(iter(summary["contrasts"]))
    summary["contrasts"][name]["difference"] += 0.01
    runtime.write(path, summary)
    with pytest.raises(ValueError, match="CSV/paired-interval recomputation failed"):
        runtime.report(experiment)


def test_report_refuses_invalid_wall_resource_evidence(experiment, monkeypatch):
    monkeypatch.setattr(runtime, "check_lock", lambda root: None)
    runtime.write(
        experiment / "wall_resources.json",
        dict(elapsed_seconds_including_setup_before_report=-1, initial_free_disk_bytes=1, gpu_name="fixture"),
    )
    with pytest.raises(ValueError, match="Invalid wall-resource evidence values"):
        runtime.report(experiment)


def test_failed_stage_retry_binds_preexisting_partial_outputs(experiment, monkeypatch):
    original = runtime.prepare

    def partial(root):
        original(root)
        raise RuntimeError("synthetic late-stage failure")

    monkeypatch.setattr(runtime, "prepare", partial)
    with pytest.raises(RuntimeError, match="late-stage"):
        runtime.run_stage(experiment, "prepare")
    monkeypatch.setattr(runtime, "prepare", original)
    runtime.run_stage(experiment, "prepare")
    receipt = runtime.read(runtime.output(experiment) / "receipt_prepare.json")
    assert "train_snapshots.csv" in receipt["outputs"]


def test_real_fresh_process_logistic_numeric_artifact(experiment, monkeypatch, tmp_path):
    through(experiment, "fit", monkeypatch)
    frame = runtime.csv(experiment, "dev_snapshots.csv")
    artifact = runtime.output(experiment) / "scorer/logistic_RFM"
    matrix = tmp_path / "query.npy"
    np.save(matrix, runtime.features(frame).to_numpy(), allow_pickle=False)
    expected = runtime.models.predict(runtime.models.reload(artifact), runtime.features(frame))
    target = tmp_path / "fresh.npy"
    script = """
import sys,numpy as np
sys.path.insert(0,sys.argv[1])
import customer_models as m
state=m.reload(sys.argv[2])
np.save(sys.argv[4],m.predict(state,np.load(sys.argv[3],allow_pickle=False)),allow_pickle=False)
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    process = subprocess.run(
        [sys.executable, "-c", script, str(TOOLS), str(artifact), str(matrix), str(target)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    np.testing.assert_array_equal(expected, np.load(target, allow_pickle=False))
