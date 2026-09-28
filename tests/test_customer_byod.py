"""BYOD privacy, refusal and safe numeric reconstruction checks; no model weights."""

import importlib
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
byod = importlib.import_module("customer_byod")
models = importlib.import_module("customer_models")
runtime = importlib.import_module("customer_runtime")


def setup(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    for name in (
        "customer_data.py",
        "customer_models.py",
        "customer_runtime.py",
        "customer_metrics.py",
        "customer_byod.py",
    ):
        shutil.copyfile(TOOLS / name, root / name)
    manifest = {"modelId": "autogluon/mitra-classifier", "revision": models.REVISION, "files": []}
    runtime.write(root / "model_manifest.json", manifest)
    runtime.write(root / "data_manifest.json", {})
    (root / "DATA_LICENSE.md").write_text("UCI sample notice", encoding="utf-8")
    files = {p.name: runtime.sha(p) for p in root.iterdir()}
    runtime.write(root / "source.json", {"files": files})
    path = tmp_path / "private.csv"
    pd.DataFrame(
        [
            {
                "invoice_id": "SECRET-invoice",
                "stock_code": "SECRET-product",
                "customer_id": "SECRET-customer",
                "invoice_time": "2011-03-01",
                "quantity": 2,
                "unit_price": 3,
                "country": "Private free text",
            }
        ]
    ).to_csv(path, index=False)
    cfg = dict(
        rights_confirmed=True,
        complete_observation_coverage=True,
        source_clock="timezone-naive",
        currency="GBP",
        coverage_start="2010-01-01",
        coverage_end="2011-04-15",
        inference_cutoff="2011-04-01",
    )
    return root, path, cfg, manifest


def artifact(root, manifest):
    target = root / "artifact"
    target.mkdir()
    for name in ("customer_data.py", "customer_models.py", "model_manifest.json", "source.json"):
        shutil.copyfile(root / name, target / name)
    runtime.write(
        target / "feature_schema.json",
        {
            "ordered_features": list(models.FEATURE_NAMES),
            "currency": "GBP",
            "lookback_days": 90,
            "horizon_days": 30,
        },
    )
    state = models.fit("mitra", np.array([[1, 1, 6, 6, 1], [10, 2, 12, 6, 1]], float), [0, 1])
    models.export(state, target / "scorer/mitra_RFM", manifest)
    runtime.write(
        target / "checksums.json",
        {
            p.relative_to(target).as_posix(): {"bytes": p.stat().st_size, "sha256": runtime.sha(p)}
            for p in target.rglob("*")
            if p.is_file()
        },
    )
    return target


def test_original_ids_and_mapping_stay_private(tmp_path):
    root, path, cfg, _ = setup(tmp_path)
    target, _ = byod.initialize(root, path, cfg, True)
    data = pd.read_csv(target / "private/input.csv")
    assert all("SECRET" not in str(v) for v in data.iloc[0])
    assert "SECRET-customer" in (target / "private/original_id_mappings.json").read_text()
    for p in target.iterdir():
        if p.is_file():
            assert "SECRET" not in p.read_text(encoding="utf-8")
    assert "No public redistribution license" in (target / "DATA_LICENSE.md").read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"rights_confirmed": False},
        {"currency": "PHP"},
        {"complete_observation_coverage": False},
        {"source_clock": "UTC"},
        {"customer_email": "secret@example.invalid"},
    ],
)
def test_configuration_refusals(tmp_path, change):
    root, path, cfg, _ = setup(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError):
        byod.initialize(root, path, cfg, True)


def test_pii_column_refused(tmp_path):
    root, path, cfg, _ = setup(tmp_path)
    frame = pd.read_csv(path)
    frame["email"] = "secret@example.invalid"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        byod.initialize(root, path, cfg, True)


def test_artifact_requires_full_inventory_and_identity(tmp_path):
    root, _, _, manifest = setup(tmp_path)
    target = artifact(root, manifest)
    byod.check_artifact(root, target)
    inventory = runtime.read(target / "checksums.json")
    inventory.pop("scorer/mitra_RFM/support_y.npy")
    runtime.write(target / "checksums.json", inventory)
    with pytest.raises(ValueError, match="omits"):
        byod.check_artifact(root, target)


def test_inference_reconstruction_crosses_real_process(tmp_path, monkeypatch):
    root, path, cfg, manifest = setup(tmp_path)
    saved = artifact(root, manifest)
    target, _ = byod.initialize(root, path, cfg, True)
    monkeypatch.setattr(models, "load_mitra", lambda *a: object())
    monkeypatch.setattr(models, "unload", lambda model: None)
    monkeypatch.setattr(
        models, "mitra_predict", lambda model, support, labels, query: 1 / (1 + np.exp(-query[:, 0] / 100))
    )
    byod.inference(target, saved)
    with pytest.raises(ValueError, match="Fresh process"):
        byod.inference(target, saved, True)
    script = """import sys,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import customer_byod as b
b.models.load_mitra=lambda *args:object()
b.models.unload=lambda model:None
b.models.mitra_predict=lambda model,support,labels,query:1/(1+np.exp(-query[:,0]/100))
b.inference(Path(sys.argv[2]),Path(sys.argv[3]),True)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(TOOLS), str(target), str(saved)],
        check=True,
        capture_output=True,
        text=True,
    )
    proof = runtime.read(runtime.output(target) / "verification.json")
    assert proof["passed"] and proof["fresh_process"]
    predictions = pd.read_csv(runtime.output(target) / "predictions.csv")
    assert set(predictions.evaluation_status) == {"not_measurable"}
    assert "SECRET" not in predictions.to_csv(index=False)


def test_evaluation_configuration_requires_mature_labels(tmp_path):
    root, path, cfg, _ = setup(tmp_path)
    cfg.update(train_cutoffs=["2011-01-01"], development_cutoffs=["2011-01-10"], test_cutoffs=["2011-03-01"])
    with pytest.raises(ValueError, match="mature"):
        byod.initialize(root, path, cfg, False)


def test_surrogates_preserve_cancellation_admin_and_invoice_ambiguity(tmp_path):
    root, path, cfg, _ = setup(tmp_path)
    base = pd.read_csv(path).iloc[0].to_dict()
    rows = [
        base,
        dict(base, invoice_id="c-PRIVATE-cancel"),
        dict(base, invoice_id="PRIVATE-fee", stock_code="POST"),
        dict(base, invoice_id="PRIVATE-ambig", customer_id="PRIVATE-A"),
        dict(base, invoice_id="PRIVATE-ambig", customer_id="PRIVATE-B"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    before, audit_before = byod.data.clean_transactions(pd.DataFrame(rows))
    target, _ = byod.initialize(root, path, cfg, True)
    after, audit_after = byod.data.clean_transactions(pd.read_csv(target / "private/input.csv"))
    assert audit_before["exclusion_counts_overlapping"] == audit_after["exclusion_counts_overlapping"]
    assert len(before) == len(after) == 1
    assert before.line_value.sum() == after.line_value.sum()


def test_future_new_identifiers_cannot_renumber_past_customers(tmp_path):
    root, path, cfg, _ = setup(tmp_path)
    first, _ = byod.initialize(root, path, cfg, True)
    before = pd.read_csv(first / "private/input.csv")
    raw = pd.read_csv(path)
    extra = dict(
        raw.iloc[0],
        customer_id="AAAA-future-customer",
        invoice_id="AAAA-future-invoice",
        stock_code="AAAA-future-product",
        invoice_time="2012-01-01",
    )
    pd.concat([raw, pd.DataFrame([extra])], ignore_index=True).to_csv(path, index=False)
    second, _ = byod.initialize(root, path, cfg, True)
    after = pd.read_csv(second / "private/input.csv")
    assert before.iloc[0][["customer_id", "invoice_id", "stock_code"]].equals(
        after.iloc[0][["customer_id", "invoice_id", "stock_code"]]
    )


def test_byod_evaluation_full_cpu_lifecycle_and_private_bundle(tmp_path, monkeypatch):
    root, path, cfg, _ = setup(tmp_path)
    rows = []
    for cutoff in runtime.DEFAULT["train"] + runtime.DEFAULT["dev"] + runtime.DEFAULT["test"]:
        for i in range(40):
            for offset in [-10 - i % 7, 5] if i < 20 else [-10 - i % 7]:
                rows.append(
                    dict(
                        invoice_id=f"SECRET-{cutoff}-{i}-{offset}",
                        customer_id=f"SECRET-person-{i}",
                        stock_code="10001",
                        quantity=1,
                        unit_price=i + 1,
                        invoice_time=pd.Timestamp(cutoff) + pd.Timedelta(days=offset),
                    )
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    cfg.update(
        coverage_start="2009-12-01",
        coverage_end="2011-12-09",
        inference_cutoff="2011-12-01",
        train_cutoffs=runtime.DEFAULT["train"],
        development_cutoffs=runtime.DEFAULT["dev"],
        test_cutoffs=runtime.DEFAULT["test"],
    )
    for name in ("requirements.txt", "RECONSTRUCT.md"):
        (root / name).write_text("Synthetic fixture\n", encoding="utf-8")
    source = runtime.read(root / "source.json")
    for name in ("requirements.txt", "RECONSTRUCT.md"):
        source["files"][name] = runtime.sha(root / name)
    runtime.write(root / "source.json", source)
    target, _ = byod.initialize(root, path, cfg, False)
    monkeypatch.setattr(models, "load_mitra", lambda *args: object())
    monkeypatch.setattr(models, "unload", lambda model: None)
    monkeypatch.setattr(
        models, "mitra_predict", lambda model, support, labels, query: 1 / (1 + np.exp(query[:, 0] / 30 - 1))
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
        lambda name: "synthetic" if name in ("torch", "autogluon.tabular") else actual_version(name),
    )
    for i, stage in enumerate(runtime.STAGES):
        monkeypatch.setattr(runtime.os, "getpid", lambda i=i: 50000 + i)
        runtime.run_stage(target, stage)
    results = runtime.output(target)
    assert runtime.read(results / "run_summary.json")["mode"] == "BYOD"
    with zipfile.ZipFile(results / "results.zip") as archive:
        assert not any("private" in name or "mapping" in name for name in archive.namelist())
        for name in archive.namelist():
            if name.endswith((".csv", ".json", ".md")):
                assert b"SECRET-" not in archive.read(name)
