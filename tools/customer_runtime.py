"""Standalone temporal customer experiment; each notebook stage runs in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import customer_data as data
import customer_metrics as metrics
import customer_models as models
import numpy as np
import pandas as pd

STAGES = ("prepare", "fit", "develop", "freeze", "evaluate", "infer", "verify", "report")
SYSTEMS = {
    "prevalence": ("prevalence", "R"),
    "recency": ("recency", "R"),
    "logistic_R": ("logistic", "R"),
    "logistic_RFM": ("logistic", "RFM"),
    "mitra_R": ("mitra", "R"),
    "mitra_RFM": ("mitra", "RFM"),
}
DEFAULT = {
    "train": ["2010-04-01", "2010-07-01", "2010-10-01", "2011-01-01"],
    "dev": ["2011-04-01", "2011-07-01"],
    "test": ["2011-09-01", "2011-11-01"],
    "inference_cutoff": "2011-12-01",
    "coverage_start": "2009-12-01",
    "coverage_end": "2011-12-09",
    "currency": "GBP",
    "source_clock": "timezone-naive",
}
CONTRASTS = [
    ("mitra_RFM", "logistic_RFM"),
    ("mitra_RFM", "recency"),
    ("mitra_RFM", "mitra_R"),
    ("logistic_RFM", "logistic_R"),
]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def output(root: Path) -> Path:
    return root / "results"


def config(root: Path) -> dict:
    return read(root / "configuration.json") if (root / "configuration.json").exists() else DEFAULT.copy()


def identity(root: Path) -> str:
    source = read(root / "source.json")
    for name, checksum in source["files"].items():
        if Path(name).name != name or sha(root / name) != checksum:
            raise ValueError("Embedded source integrity failed: " + name)
    return digest({"source": source, "configuration": config(root)})


def csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(output(root) / name, dtype={"customer_id": str, "cutoff": str})


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(models.FEATURE_NAMES)]


def load_raw(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        root / "private/transactions.csv.gz",
        dtype={
            "invoice_id": str,
            "stock_code": str,
            "customer_id": str,
        },
        keep_default_na=False,
    )
    raw["invoice_time"] = pd.to_datetime(raw["invoice_time"], errors="coerce")
    return raw


def prepare(root: Path) -> None:
    cfg = config(root)
    if cfg.get("byod"):
        raw = data.validate_byod(root / "private/input.csv", cfg)
        audit = {
            "mode": "BYOD",
            "rows": len(raw),
            "declared_coverage": cfg,
            "assumption": "User asserts complete observation coverage",
        }
    else:
        raw, audit = data.prepare(root)
    (root / "private").mkdir(exist_ok=True)
    raw.to_csv(root / "private/transactions.csv.gz", index=False, compression="gzip")
    audit["snapshot_sensitivity"] = data.audit_snapshots(
        raw,
        {
            "coverage_start": cfg["coverage_start"],
            "coverage_end": cfg["coverage_end"],
            "train_cutoffs": cfg["train"],
            "development_cutoffs": cfg["dev"],
            "test_cutoffs": cfg["test"],
        },
    )
    write(output(root) / "dataset_audit.json", audit)
    cohorts = []
    for role, cap in (("train", 512), ("dev", 1000), ("test", 1000)):
        pieces = []
        for cutoff in cfg[role]:
            frame = data.snapshots(raw, cutoff, cfg["coverage_start"], cfg["coverage_end"])
            selected = data.select_cohort(frame, cap)
            counts = selected.y_true.value_counts()
            if role != "train" and min(counts.get(0, 0), counts.get(1, 0)) < 20:
                raise ValueError(f"Feasibility gate: {role} {cutoff} needs >=20 of each class")
            cohorts.append(
                {
                    "role": role,
                    "cutoff": cutoff,
                    "eligible": len(frame),
                    "selected": len(selected),
                    "positives": int(selected.y_true.sum()),
                    "id_hash": digest(selected.customer_id.tolist()),
                }
            )
            pieces.append(selected)
        combined = pd.concat(pieces, ignore_index=True)
        save_csv(output(root) / f"{role}_snapshots.csv", combined)
    support = csv(root, "train_snapshots.csv")
    if set(support.y_true) != {0, 1}:
        raise ValueError("Support must contain both classes")
    if pd.Timestamp(max(cfg["train"])) + pd.Timedelta(days=30) > pd.Timestamp(min(cfg["dev"])):
        raise ValueError("Support labels do not mature before development")
    final = data.select_cohort(
        data.snapshots(
            raw, cfg["inference_cutoff"], cfg["coverage_start"], cfg["coverage_end"], require_labels=False
        ),
        1000,
    )
    save_csv(output(root) / "inference_snapshots.csv", final)
    write(
        output(root) / "cohort_manifest.json",
        {
            "cohorts": cohorts,
            "inference_rows": len(final),
            "sampling": "label-blind SHA-256, seed42",
            "configuration": cfg,
            "recurring_customers_allowed": True,
            "log_completeness": "assumed, not independently verified",
        },
    )
    write(
        output(root) / "feature_schema.json",
        {
            "ordered_features": list(models.FEATURE_NAMES),
            "currency": cfg["currency"],
            "lookback_days": 90,
            "horizon_days": 30,
            "value_meaning": "gross positive purchase value, not net revenue or profit",
        },
    )
    # A concrete timeline is constructed only from development data.
    first = csv(root, "dev_snapshots.csv").iloc[0]
    t = pd.Timestamp(first.cutoff)
    timeline = raw[
        (raw.customer_id == first.customer_id)
        & (raw.invoice_time >= t - pd.Timedelta(days=90))
        & (raw.invoice_time < t + pd.Timedelta(days=30))
    ].copy()
    timeline["window"] = np.where(timeline.invoice_time < t, "history", "outcome")
    save_csv(root / "private/teaching_timeline.csv", timeline)


def fit(root: Path) -> None:
    support = csv(root, "train_snapshots.csv")
    for system, (kind, feature_set) in SYSTEMS.items():
        state = models.fit(kind, features(support), support.y_true.to_numpy(), feature_set=feature_set)
        models.export(
            state, output(root) / "scorer" / system, model_manifest=read(root / "model_manifest.json")
        )
    write(
        output(root) / "support_metadata.json",
        {
            "rows": len(support),
            "positive_count": int(support.y_true.sum()),
            "snapshots_sha256": sha(output(root) / "train_snapshots.csv"),
            "adaptation": "Mitra labelled in-context conditioning; no gradient weight fine-tuning",
            "systems": SYSTEMS,
        },
    )


def score(root: Path, frame: pd.DataFrame) -> list[dict]:
    support_ids = set(csv(root, "train_snapshots.csv").customer_id)
    rows = []
    for system in SYSTEMS:
        state = models.reload(output(root) / "scorer" / system)
        model = None
        if system.startswith("mitra"):
            model = models.load_mitra(root / "model_cache", read(root / "model_manifest.json"))
        try:
            values = models.predict(state, features(frame), model=model)
        finally:
            if model is not None:
                del model
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
        for record, value in zip(frame.to_dict("records"), values, strict=True):
            row = {
                "customer_id": str(record["customer_id"]),
                "cutoff": str(record["cutoff"]),
                "system": system,
                "score": float(value),
                "seen_in_support": str(record["customer_id"]) in support_ids,
            }
            if "y_true" in record and pd.notna(record["y_true"]):
                row["y_true"] = int(record["y_true"])
            if system != "recency":
                row["probability"] = float(value)
            rows.append(row)
    return rows


def prediction_rows(root: Path, role: str) -> list[dict]:
    frame = csv(root, role + "_predictions.csv")
    return [{k: v for k, v in row.items() if pd.notna(v)} for row in frame.to_dict("records")]


def plot(root: Path, role: str, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    frame = pd.DataFrame(rows)
    for name, group in frame.groupby("system"):
        if name != "recency":
            observed, predicted = calibration_curve(group.y_true, group.probability, n_bins=5)
            axes[1].plot(predicted, observed, marker="o", label=name)
    scores = metrics.evaluate(rows)
    # Compute the displayed quantities from row-level predictions, not hard-coded outcomes.
    values = []
    for name in SYSTEMS:
        subset = frame[frame.system == name]
        rates = []
        for _, group in subset.groupby("cutoff"):
            ranked = ranked_ids(group.to_dict("records"))
            labels = dict(zip(group.customer_id, group.y_true, strict=True))
            rates.append(np.mean([labels[i] for i in ranked]))
        values.append(float(np.mean(rates)))
    axes[0].barh(list(SYSTEMS), values)
    axes[0].set(xlabel="Equal-cutoff mean Precision@20%", xlim=(0, 1), title=role)
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Reliability diagnostic, not proof of calibration",
    )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output(root) / f"{role}_comparison.png", dpi=130)
    plt.close(fig)
    write(output(root) / f"{role}_metrics.json", scores)


def ranked_ids(rows: list[dict], budget: float = 0.2) -> list[str]:
    rank = sorted(
        rows,
        key=lambda r: (
            -r["score"],
            hashlib.sha256(("42:customer-budget-tie-v1:" + str(r["customer_id"])).encode()).hexdigest(),
        ),
    )
    return [str(r["customer_id"]) for r in rank[: int(np.ceil(budget * len(rows)))]]


def develop(root: Path) -> None:
    rows = score(root, csv(root, "dev_snapshots.csv"))
    save_csv(output(root) / "dev_predictions.csv", pd.DataFrame(rows))
    plot(root, "dev", prediction_rows(root, "dev"))
    write(
        output(root) / "development_activity.json",
        {
            "change": "R to RFM within the same learned model; support/horizon/budget unchanged",
            "canonical_budget_20": metrics.evaluate(rows),
            "optional_display_budget_10": metrics.evaluate(rows, budget=0.1),
            "test_used_for_selection": False,
            "completion_record_required": False,
        },
    )


def freeze(root: Path) -> None:
    write(
        output(root) / "experiment_lock.json",
        {
            "identity": identity(root),
            "cohort_manifest": read(output(root) / "cohort_manifest.json"),
            "files": {
                p.relative_to(output(root)).as_posix(): sha(p)
                for p in output(root).rglob("*")
                if p.is_file() and ("scorer" in p.parts or p.name.endswith("snapshots.csv"))
            },
            "budget": 0.2,
            "seed": 42,
            "bootstrap_replicates": 2000,
            "systems": SYSTEMS,
            "predefined_inference_scorer": "mitra_RFM",
            "model_identity": read(root / "model_manifest.json"),
        },
    )


def check_lock(root: Path) -> None:
    lock = read(output(root) / "experiment_lock.json")
    if lock["identity"] != identity(root):
        raise ValueError("Experiment configuration changed after freeze")
    for name, expected in lock["files"].items():
        if (
            not (output(root) / name).resolve().is_relative_to(output(root).resolve())
            or sha(output(root) / name) != expected
        ):
            raise ValueError("Frozen input changed: " + name)


def evaluate(root: Path) -> None:
    check_lock(root)
    rows = score(root, csv(root, "test_snapshots.csv"))
    save_csv(output(root) / "test_predictions.csv", pd.DataFrame(rows))
    rows = prediction_rows(root, "test")
    plot(root, "test", rows)
    write(output(root) / "paired_intervals.json", metrics.paired_bootstrap(rows, CONTRASTS, n_boot=2000))
    diagnostic = []
    predictions = pd.DataFrame(rows)
    for _cutoff, group in predictions[predictions.system == "mitra_RFM"].groupby("cutoff"):
        selected = set(ranked_ids(group.to_dict("records")))
        for row in group.to_dict("records"):
            row["selected_at_budget"] = row["customer_id"] in selected
            row["outcome_group"] = (
                ("selected_positive" if row["y_true"] else "selected_negative")
                if row["selected_at_budget"]
                else ("missed_positive" if row["y_true"] else "unselected_negative")
            )
            diagnostic.append(row)
    details = pd.DataFrame(diagnostic).merge(
        csv(root, "test_snapshots.csv"), on=["customer_id", "cutoff", "y_true"], validate="one_to_one"
    )
    save_csv(output(root) / "customer_diagnostics.csv", details)


def infer(root: Path) -> None:
    check_lock(root)
    rows = score(root, csv(root, "inference_snapshots.csv"))
    for row in rows:
        row.pop("y_true", None)
        row["evaluation_status"] = "not_measurable"
    save_csv(output(root) / "inference_predictions.csv", pd.DataFrame(rows))
    write(
        output(root) / "inference_provenance.json",
        {
            "pid": os.getpid(),
            "selection": "Predefined Mitra-RFM example; not selected using held-out outcomes",
            "evaluation_status": "not_measurable",
            "reason": "No complete final 30-day outcome window",
        },
    )


def verify(root: Path) -> None:
    check_lock(root)
    cfg = config(root)
    prior = csv(root, "inference_snapshots.csv")
    rebuilt = data.select_cohort(
        data.snapshots(
            load_raw(root),
            cfg["inference_cutoff"],
            cfg["coverage_start"],
            cfg["coverage_end"],
            require_labels=False,
        ),
        1000,
    )
    if prior.customer_id.tolist() != rebuilt.customer_id.tolist():
        raise ValueError("Reconstructed cohort/order changed")
    np.testing.assert_allclose(
        features(prior).to_numpy(), features(rebuilt).to_numpy(), atol=1e-10, rtol=1e-12
    )
    rows = score(root, rebuilt)
    saved = csv(root, "inference_predictions.csv")
    for name in SYSTEMS:
        actual = [r for r in rows if r["system"] == name]
        expected = saved[saved.system == name].to_dict("records")
        a = np.array([r["score"] for r in actual])
        b = np.array([r["score"] for r in expected])
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
        if name != "recency":
            np.testing.assert_array_equal(a >= 0.5, b >= 0.5)
        if ranked_ids(actual) != ranked_ids(expected):
            raise ValueError("Reload top-k mismatch: " + name)
    write(
        output(root) / "verification.json",
        {
            "passed": True,
            "pid": os.getpid(),
            "different_process": os.getpid() != read(output(root) / "inference_provenance.json")["pid"],
            "reconstructed_customers": prior.customer_id.head(3).tolist(),
            "replayed_rows": len(prior),
            "atol": 1e-5,
            "rtol": 1e-4,
            "exact_labels_and_top_k": True,
        },
    )


def report(root: Path) -> None:
    check_lock(root)
    wall = None
    wall_path = root / "wall_resources.json"
    if wall_path.exists():
        wall = read(wall_path)
        expected_fields = {
            "elapsed_seconds_including_setup_before_report",
            "initial_free_disk_bytes",
            "gpu_name",
        }
        if set(wall) != expected_fields:
            raise ValueError("Invalid wall-resource evidence schema")
        elapsed = wall["elapsed_seconds_including_setup_before_report"]
        disk = wall["initial_free_disk_bytes"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not np.isfinite(elapsed)
            or elapsed < 0
            or type(disk) is not int
            or disk < 0
            or not isinstance(wall["gpu_name"], str)
            or not wall["gpu_name"].strip()
        ):
            raise ValueError("Invalid wall-resource evidence values")
        # Retain the notebook's actual pre-report reading; this does not include report/export duration.
        write(output(root) / "wall_resources.json", wall)
    verification = read(output(root) / "verification.json")
    if not verification["passed"] or not verification["different_process"]:
        raise ValueError("Fresh-process reload verification is required")
    for role in ("dev", "test"):
        recomputed = metrics.evaluate(prediction_rows(root, role))
        if digest(recomputed) != digest(read(output(root) / f"{role}_metrics.json")):
            raise ValueError("CSV/metrics recomputation failed")
    paired = metrics.paired_bootstrap(prediction_rows(root, "test"), CONTRASTS, n_boot=2000)
    if digest(paired) != digest(read(output(root) / "paired_intervals.json")):
        raise ValueError("CSV/paired-interval recomputation failed")
    for name in (
        "data_manifest.json",
        "model_manifest.json",
        "DATA_LICENSE.md",
        "source.json",
        "requirements.txt",
        "customer_data.py",
        "customer_models.py",
        "customer_runtime.py",
        "customer_metrics.py",
        "customer_byod.py",
        "RECONSTRUCT.md",
    ):
        shutil.copyfile(root / name, output(root) / name)
    write(
        output(root) / "environment.json",
        {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "torch", "autogluon.tabular", "openpyxl")
        },
    )
    receipts = {s: read(output(root) / f"receipt_{s}.json") for s in STAGES[:-1]}
    seconds = sum(r["seconds"] for r in receipts.values())
    peak = max(r.get("peak_gpu_bytes", 0) for r in receipts.values())
    write(
        output(root) / "run_summary.json",
        {
            "status": "Candidate",
            "mode": "BYOD" if config(root).get("byod") else "sample",
            "verification": verification,
            "stage_seconds_excluding_install": seconds,
            "peak_gpu_bytes": peak,
            "wall_resources_before_report": wall,
            "within_target_runtime_including_setup_before_report": (
                wall["elapsed_seconds_including_setup_before_report"] <= 45 * 60 if wall is not None else None
            ),
            "within_target_runtime_excluding_install": seconds <= 45 * 60,
            "within_gpu_target": peak <= 12 * 1024**3,
            "stages": receipts,
            "limitations": [
                "Single historical UK retailer, many wholesale customers",
                "Not Philippine MSME evidence",
                "No campaign response or causal uplift measurement",
                "Two test dates; customer-cluster intervals conditional on dates and frozen models",
                "Log completeness assumed; pretraining overlap unknown; hosted review still required",
            ],
        },
    )
    files = {
        p.relative_to(output(root)).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in output(root).rglob("*")
        if p.is_file() and p.name not in ("results.zip", "checksums.json", "receipt_report.json")
    }
    write(output(root) / "checksums.json", files)
    with zipfile.ZipFile(output(root) / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*files, "checksums.json"]:
            archive.write(output(root) / name, name)
    with zipfile.ZipFile(output(root) / "results.zip") as archive:
        if archive.testzip():
            raise ValueError("Export ZIP integrity failed")


def run_stage(root: Path, stage: str) -> None:
    output(root).mkdir(parents=True, exist_ok=True)
    run_id = identity(root)
    previous = None
    for earlier in STAGES[: STAGES.index(stage)]:
        path = output(root) / f"receipt_{earlier}.json"
        receipt = read(path)
        if not receipt.get("completed") or receipt["identity"] != run_id or receipt["previous"] != previous:
            raise ValueError("Stale stage receipt: " + earlier)
        for name, checksum in receipt["outputs"].items():
            if sha(output(root) / name) != checksum:
                raise ValueError("Stage output changed: " + name)
        previous = sha(path)
    target = output(root) / f"receipt_{stage}.json"
    owned = set(read(target)["outputs"]) if target.exists() else set()
    write(target, {"completed": False, "outputs": dict.fromkeys(owned)})
    for later in STAGES[STAGES.index(stage) + 1 :]:
        path = output(root) / f"receipt_{later}.json"
        if path.exists():
            old = read(path)
            old["completed"] = False
            write(path, old)
    before = {p.relative_to(output(root)).as_posix(): sha(p) for p in output(root).rglob("*") if p.is_file()}
    start = time.monotonic()
    gpu = stage in ("develop", "evaluate", "infer", "verify")
    if gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Model stages require the documented hosted Colab T4")
        torch.cuda.reset_peak_memory_stats()
    try:
        globals()[stage](root)
    except Exception:
        partial = {
            p.relative_to(output(root)).as_posix(): sha(p)
            for p in output(root).rglob("*")
            if p.is_file() and not p.name.startswith("receipt_")
        }
        write(
            target,
            {
                "completed": False,
                "outputs": {n: h for n, h in partial.items() if before.get(n) != h or n in owned},
            },
        )
        raise
    produced = {
        p.relative_to(output(root)).as_posix(): sha(p)
        for p in output(root).rglob("*")
        if p.is_file() and not p.name.startswith("receipt_")
    }
    write(
        target,
        {
            "completed": True,
            "identity": run_id,
            "previous": previous,
            "pid": os.getpid(),
            "seconds": time.monotonic() - start,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if gpu else 0,
            "outputs": {n: h for n, h in produced.items() if before.get(n) != h or n in owned},
        },
    )
    print(stage + ": PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    run_stage(args.root.resolve(), args.stage)
