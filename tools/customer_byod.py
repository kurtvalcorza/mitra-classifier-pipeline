"""Opt-in authorized transaction evaluation or explicit compatible artifact inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import customer_data as data
import customer_models as models
import customer_runtime as runtime
import numpy as np
import pandas as pd


def initialize(root: Path, csv_path: Path, cfg: dict, inference_only: bool) -> tuple[Path, dict]:
    """Keep raw identifiers and their mapping outside the shareable results directory."""
    if cfg.get("rights_confirmed") is not True or cfg.get("complete_observation_coverage") is not True:
        raise ValueError("Confirm processing rights and complete observation coverage")
    if cfg.get("source_clock") != "timezone-naive" or cfg.get("currency") != "GBP":
        raise ValueError("V1 requires GBP and timezone-naive source-clock compatibility")
    if csv_path.stat().st_size > 200_000_000:
        raise ValueError("BYOD CSV exceeds200MB")
    raw = data.read_transactions_csv(csv_path)
    if not set(raw.columns).issubset(data.COLUMNS) or not (set(data.COLUMNS) - {"country"}).issubset(raw):
        raise ValueError("Refuse unexpected identifying columns or missing transaction fields")
    raw = data.canonicalize(raw)
    intake = {
        "rows": len(raw),
        "empty_identifier_fields": {
            c: int(raw[c].eq("").sum()) for c in ("customer_id", "invoice_id", "stock_code")
        },
        "missing_value_policy": "Only empty fields are missing; literal text such as NA is an identifier",
        "timestamp_policy": "Timezone-naive ISO date or date-time; other or missing values are refused",
    }
    mappings = {}
    for column, prefix in (("customer_id", "customer"), ("invoice_id", "invoice"), ("stock_code", "product")):
        ids = sorted(set(raw[column]) - {""})
        mapping = {}
        for key in ids:
            # Stable under future-row additions; positional IDs would change earlier sampling/ties.
            token = hashlib.sha256(f"byod-v1:{prefix}:{key}".encode()).hexdigest()[:24]
            value = f"local_{prefix}_{token}"
            if column == "invoice_id" and key.upper().startswith("C"):
                value = "C" + value
            if column == "stock_code" and key.upper() in data.ADMIN_CODES:
                value = key.upper()
            mapping[key] = value
        raw[column] = raw[column].map(mapping).fillna("")
        mappings[column] = mapping
    raw["country"] = ""  # Audit-only free text is not needed for BYOD inference.
    target = root / (("byod_inference_" if inference_only else "byod_evaluation_") + uuid.uuid4().hex)
    target.mkdir()
    (target / "private").mkdir(exist_ok=True)
    for name in [*runtime.read(root / "source.json")["files"], "source.json"]:
        if Path(name).name != name:
            raise ValueError("Unexpected carried source path")
        shutil.copyfile(root / name, target / name)
    runtime.write(target / "private/original_id_mappings.json", mappings)
    runtime.save_csv(target / "private/input.csv", raw)
    allowed = {
        "rights_confirmed",
        "complete_observation_coverage",
        "source_clock",
        "currency",
        "coverage_start",
        "coverage_end",
        "train_cutoffs",
        "development_cutoffs",
        "test_cutoffs",
        "inference_cutoff",
    }
    if set(cfg) - allowed:
        raise ValueError("Unexpected BYOD configuration fields")
    adjusted = dict(cfg, byod=True)
    if not inference_only:
        adjusted.update(train=cfg["train_cutoffs"], dev=cfg["development_cutoffs"], test=cfg["test_cutoffs"])
        data.validate_byod(target / "private/input.csv", adjusted)
    runtime.write(target / "configuration.json", adjusted)
    runtime.write(
        target / "data_manifest.json",
        {
            "mode": "BYOD",
            "currency": "GBP",
            "source_clock": "timezone-naive",
            "raw_csv_sha256": runtime.sha(csv_path),
            "intake": intake,
            "administrative_codes": sorted(data.ADMIN_CODES),
            "license": "User-supplied rights acknowledgement",
        },
    )
    (target / "DATA_LICENSE.md").write_text(
        "# BYOD rights\n\nThe user acknowledged processing rights. "
        "No public redistribution license is inferred. "
        "Original identifiers and transaction logs remain outside the shareable results bundle.\n",
        encoding="utf-8",
    )
    carried = runtime.read(target / "source.json")
    for name in ("data_manifest.json", "DATA_LICENSE.md"):
        if name in carried["files"]:
            carried["files"][name] = runtime.sha(target / name)
    runtime.write(target / "source.json", carried)
    return target, adjusted


def check_artifact(root: Path, artifact: Path) -> None:
    """Validate the supplied extracted evidence bundle; never unpickle user state."""
    inventory = runtime.read(artifact / "checksums.json")
    required = {
        "feature_schema.json",
        "model_manifest.json",
        "source.json",
        "customer_data.py",
        "customer_models.py",
        "scorer/mitra_RFM/manifest.json",
        "scorer/mitra_RFM/scorer.json",
        "scorer/mitra_RFM/support_X.npy",
        "scorer/mitra_RFM/support_y.npy",
    }
    if not required.issubset(inventory):
        raise ValueError("Artifact inventory omits required scorer files")
    for name, item in inventory.items():
        path = (artifact / name).resolve()
        if (
            not path.is_relative_to(artifact.resolve())
            or path.stat().st_size != item["bytes"]
            or runtime.sha(path) != item["sha256"]
        ):
            raise ValueError("Artifact checksum/path/size mismatch: " + name)
    schema = runtime.read(artifact / "feature_schema.json")
    if (
        schema["ordered_features"] != list(models.FEATURE_NAMES)
        or schema["currency"] != "GBP"
        or schema["lookback_days"] != 90
        or schema["horizon_days"] != 30
    ):
        raise ValueError("Incompatible artifact feature policy/currency")
    if runtime.read(artifact / "model_manifest.json") != runtime.read(root / "model_manifest.json"):
        raise ValueError("Incompatible pinned model identity")
    source = runtime.read(artifact / "source.json")
    for module in ("customer_data.py", "customer_models.py"):
        if source["files"][module] != runtime.sha(root / module):
            raise ValueError("Incompatible cleaning/model implementation")
        if runtime.sha(artifact / module) != source["files"][module]:
            raise ValueError("Artifact source implementation differs from declared identity")
    state = models.reload(artifact / "scorer/mitra_RFM")
    if state["kind"] != "mitra" or state["feature_set"] != "RFM":
        raise ValueError("Explicit Mitra-RFM scoring artifact required")
    if state["model_manifest"] != runtime.read(root / "model_manifest.json"):
        raise ValueError("Scorer model identity differs from supplied pinned snapshot")


def inference(root: Path, artifact: Path, verify_only: bool = False) -> None:
    check_artifact(root, artifact)
    cfg = runtime.config(root)
    raw = data.canonicalize(data.read_transactions_csv(root / "private/input.csv"))
    frame = data.select_cohort(
        data.snapshots(
            raw, cfg["inference_cutoff"], cfg["coverage_start"], cfg["coverage_end"], require_labels=False
        ),
        1000,
    )
    state = models.reload(artifact / "scorer/mitra_RFM")
    model = models.load_mitra(root / "model_cache", runtime.read(root / "model_manifest.json"))
    try:
        values = models.predict(state, runtime.features(frame), model=model)
    finally:
        models.unload(model)
    result = pd.DataFrame(
        {
            "customer_id": frame.customer_id,
            "cutoff": frame.cutoff,
            "score": values,
            "evaluation_status": "not_measurable",
        }
    )
    results = runtime.output(root)
    results.mkdir(exist_ok=True)
    if verify_only:
        provenance = runtime.read(results / "inference_provenance.json")
        if provenance["pid"] == os.getpid() or provenance["input_sha256"] != runtime.sha(
            root / "private/input.csv"
        ):
            raise ValueError("Fresh process and unchanged input required")
        if provenance["configuration_sha256"] != runtime.sha(root / "configuration.json"):
            raise ValueError("Configuration changed before verification")
        saved = pd.read_csv(results / "predictions.csv", dtype={"customer_id": str})
        if saved.customer_id.tolist() != result.customer_id.tolist():
            raise ValueError("Reload customer cohort mismatch")
        np.testing.assert_allclose(saved.score, result.score, atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(saved.score >= 0.5, result.score >= 0.5)
        if runtime.ranked_ids(saved.to_dict("records")) != runtime.ranked_ids(result.to_dict("records")):
            raise ValueError("Reload top-k mismatch")
        runtime.write(
            results / "verification.json",
            {"passed": True, "fresh_process": True, "atol": 1e-5, "rtol": 1e-4, "rows": len(result)},
        )
    else:
        runtime.write(
            results / "inference_provenance.json",
            {
                "pid": os.getpid(),
                "input_sha256": runtime.sha(root / "private/input.csv"),
                "configuration_sha256": runtime.sha(root / "configuration.json"),
            },
        )
        runtime.save_csv(results / "predictions.csv", result)
        runtime.write(
            results / "run_summary.json",
            {
                "status": "Candidate",
                "mode": "artifact-inference",
                "evaluation_status": "not_measurable",
                "rows": len(result),
                "artifact_checksums_sha256": runtime.sha(artifact / "checksums.json"),
                "warning": "Explicit supplied compatible scorer; transfer quality is not measured",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--mode", choices=("evaluate", "artifact-inference"), default="evaluate")
    parser.add_argument("--verify-inference", action="store_true")
    args = parser.parse_args()
    if args.verify_inference:
        inference(args.root, args.artifact, verify_only=True)
        return
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    target, _ = initialize(args.root, args.csv, cfg, args.mode == "artifact-inference")
    if args.mode == "evaluate":
        for stage in runtime.STAGES:
            subprocess.run(
                [
                    sys.executable,
                    str(target / "customer_runtime.py"),
                    "--root",
                    str(target),
                    "--stage",
                    stage,
                ],
                check=True,
            )
    else:
        if args.artifact is None:
            raise ValueError("Explicit compatible artifact required; no tutorial fallback")
        inference(target, args.artifact.resolve())
        subprocess.run(
            [
                sys.executable,
                str(target / "customer_byod.py"),
                "--root",
                str(target),
                "--artifact",
                str(args.artifact.resolve()),
                "--verify-inference",
            ],
            check=True,
        )
        results = runtime.output(target)
        files = {
            p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in results.iterdir()
            if p.is_file() and p.name not in ("results.zip", "checksums.json")
        }
        runtime.write(results / "checksums.json", files)
        with zipfile.ZipFile(results / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in [*files, "checksums.json"]:
                archive.write(results / name, name)
    print("BYOD evidence:", runtime.output(target) / "results.zip")


if __name__ == "__main__":
    main()
