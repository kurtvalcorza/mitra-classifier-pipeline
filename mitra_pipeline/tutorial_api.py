"""Release-grade public API for the Mitra Classifier notebook surface.

This module owns the repository-facing behavior the notebooks must exercise directly:
classification data validation, Mitra fit/predict calls, model snapshot locking, and predictor
artifact validation. Imports that pull in heavy ML dependencies are intentionally delayed so
archive and manifest checks remain unit-testable in a lightweight CI environment.
"""

# ruff: noqa: E501  -- long messages and docstrings are kept on one line
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

# Fleet snapshot identity (DIMER Notebook Specification 1.1, ST3/MOD13). The pinned upstream model is unchanged;
# these are the fleet-standard names for the same repository, revision, license and snapshot key. The published
# PINNED_REVISION spelling stays as an alias of MODEL_REVISION.
MODEL_ID = "autogluon/mitra-classifier"
MODEL_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "mitra-classifier"
MANIFEST_NAME = "dimer-base-manifest.json"
# Root-level package: the repository root is one level up from this file.
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights" / MODEL_KEY
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

PINNED_REVISION = MODEL_REVISION
WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
# The ids `classification_metrics` reports (sklearn, computed from labels + class probabilities) and the ids
# `normalize_autogluon_classification_metrics` may carry through from AutoGluon's own evaluate().
METRIC_IDS = ("accuracy", "balanced_accuracy", "log_loss", "roc_auc", "f1_macro", "mcc", "f1", "precision", "recall")
DECISION_RULE = "argmax"  # `predict` is the argmax over `predict_proba`; no threshold is shipped
MIN_CLASSES = 2
MAX_CLASSES = 10  # Mitra's class ceiling (dim_output)
MIN_CLASS_COUNT = 2  # rows per class in the support split (stratified holdout needs 2)

MAX_TRAIN_ROWS = 10_000
MAX_FEATURES = 500
MIN_TRAIN_ROWS = 50

ARTIFACT_FORMAT = "dimer-autogluon-predictor"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_MANIFEST = "artifact_manifest.json"
RUN_METADATA = "tutorial_run_metadata.json"
DIMER_MODEL_MANIFEST = "dimer-model-manifest.json"

MAX_ARCHIVE_MEMBER_BYTES = 1 * 1024**3
MAX_ARCHIVE_EXPANDED_BYTES = 4 * 1024**3
MAX_COMPRESSION_RATIO = 200.0

CLASSIFICATION_LOWER_IS_BETTER = {"log_loss"}
# AutoGluon eval_metric names the tutorial accepts; the Mitra model hyper-parameter `metric` knows two of them.
EVAL_METRICS = ("accuracy", "balanced_accuracy", "log_loss", "f1_macro", "mcc")
MITRA_METRIC_MAP = {"accuracy": "accuracy", "log_loss": "log_loss"}

REQUIRED_RUN_METADATA_FIELDS = {
    "artifact_format",
    "artifact_format_version",
    "base_model",
    "base_model_revision",
    "weights_sha256",
    "config_sha256",
    "autogluon_version",
    "python_version",
    "problem_type",
    "target_column",
    "features",
    "mode",
    "selection_basis",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_hex_digest(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest.")
    return digest


def read_csv_bytes(payload: bytes, label: str) -> pd.DataFrame:
    """Parse UTF-8/BOM CSV bytes while rejecting duplicate raw headers before pandas renames them."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: expected UTF-8 CSV input.") from exc

    rows = csv.reader(io.StringIO(text, newline=""))
    header = next((row for row in rows if row and not (len(row) == 1 and not row[0].strip())), [])
    if not header:
        raise ValueError(f"{label}: CSV has no header row.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"{label}: duplicate column names are not supported: {duplicates}")

    return pd.read_csv(io.BytesIO(payload))


def validate_labeled_frame(
    frame: pd.DataFrame,
    target_column: str,
    *,
    name: str,
    drop_columns: Iterable[str] = (),
    min_rows: int = 2,
    min_classes: int = 1,
    max_classes: int = MAX_CLASSES,
    min_class_count: int = 1,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Validate one labelled classification table using the repository's user-facing contract.

    The target keeps its own dtype (any label type); rows with a missing target are dropped and counted; the
    class count must lie in ``min_classes..max_classes`` (Mitra's ceiling is ``MAX_CLASSES``) and every class
    needs at least ``min_class_count`` rows (2 for a support split that will be stratified)."""
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"{name}: duplicate column names are not supported: {duplicates}")
    if target_column not in frame.columns:
        raise ValueError(f"{name}: target {target_column!r} not found.")

    drops = [c for c in drop_columns if c != target_column and c in frame.columns]
    out = frame.drop(columns=drops, errors="ignore").copy()
    rows_before = len(out)
    out = out.dropna(subset=[target_column]).copy()
    dropped_target_rows = rows_before - len(out)

    features = [c for c in out.columns if c != target_column]
    counts = out[target_column].value_counts()
    errors: list[str] = []
    if len(out) < min_rows:
        errors.append(f"use at least {min_rows} labelled rows after missing-target drops")
    if not features:
        errors.append("no feature columns remain")
    if len(features) > MAX_FEATURES:
        errors.append(f"{len(features)} features exceed the {MAX_FEATURES}-feature Mitra ceiling")
    if not min_classes <= len(counts) <= max_classes:
        errors.append(f"target has {len(counts)} classes; Mitra requires {min_classes}-{max_classes}")
    if counts.empty or int(counts.min()) < min_class_count:
        errors.append(f"every class needs at least {min_class_count} row(s)")
    if errors:
        raise ValueError(f"{name} is not ready: " + "; ".join(errors))

    report = {
        "rows_before_target_drop": int(rows_before),
        "rows_after_target_drop": int(len(out)),
        "dropped_missing_target_rows": int(dropped_target_rows),
        "exact_duplicate_rows": int(out.duplicated().sum()),
        "feature_count": int(len(features)),
        "class_count": int(len(counts)),
        "class_counts": {str(label): int(n) for label, n in counts.sort_index().items()},
    }
    return out, features, report


def require_class_coverage(
    train_frame: pd.DataFrame, eval_frame: pd.DataFrame, target_column: str, eval_name: str
) -> None:
    """An in-context classifier can only predict classes present in its support rows, and a holdout that
    misses a trained class cannot score it: both directions are refused."""
    train_classes = set(train_frame[target_column].dropna().unique())
    eval_classes = set(eval_frame[target_column].dropna().unique())
    missing = sorted(train_classes - eval_classes, key=str)
    unseen = sorted(eval_classes - train_classes, key=str)
    problems: list[str] = []
    if missing:
        problems.append(f"missing trained target classes: {missing}")
    if unseen:
        problems.append(f"contains unseen target classes not present in training: {unseen}")
    if problems:
        raise ValueError(
            f"{eval_name} " + "; ".join(problems) + ". Adjust the split or provide a compatible evaluation partition."
        )


def stratified_holdout(
    frame: pd.DataFrame, target_column: str, validation_split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded stratified holdout that keeps every class in both partitions."""
    from sklearn.model_selection import train_test_split

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    counts = frame[target_column].value_counts()
    n_classes = int(counts.size)
    if (counts < 2).any():
        raise ValueError(f"class(es) with fewer than 2 rows cannot be split: {counts[counts < 2].to_dict()}")
    n = len(frame)
    n_val = int(round(n * validation_split))
    n_val = min(max(n_val, n_classes), n - n_classes)
    if n_val < n_classes or n - n_val < n_classes:
        raise ValueError(
            f"cannot build a stratified holdout: {n} usable rows, {n_classes} classes, "
            f"validation_split={validation_split}. Provide more rows, fewer classes, or a larger split."
        )
    train, holdout = train_test_split(frame, test_size=n_val, random_state=seed, stratify=frame[target_column])
    if set(train[target_column].unique()) != set(frame[target_column].unique()):
        raise RuntimeError("stratified split left a class out of train; refusing to train.")
    return train, holdout


def cap_training_rows(
    frame: pd.DataFrame,
    target_column: str,
    *,
    seed: int,
    max_rows: int = MAX_TRAIN_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Seeded, class-preserving cap: one row of every class is kept first, the rest is sampled."""
    if max_rows > MAX_TRAIN_ROWS:
        raise ValueError(f"max_rows cannot exceed Mitra's {MAX_TRAIN_ROWS:,}-row ceiling.")
    before = len(frame)
    if before <= max_rows:
        return frame.copy(), {"applied": False, "before": before, "after": before}
    classes_before = set(frame[target_column].unique())
    if max_rows < len(classes_before):
        raise ValueError(f"row ceiling ({max_rows}) is below the class count ({len(classes_before)}); cannot keep every class.")
    rng = np.random.RandomState(seed)
    keep: list[Any] = []
    for cls in sorted(classes_before, key=str):
        idx = frame.index[frame[target_column] == cls].to_numpy()
        keep.append(rng.choice(idx, size=1)[0])
    keep_set = set(keep)
    remaining = np.array([i for i in frame.index.to_numpy() if i not in keep_set])
    need = max_rows - len(keep)
    if need > 0 and len(remaining) > 0:
        keep.extend(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
    capped = frame.loc[keep].copy()
    if set(capped[target_column].unique()) != classes_before:
        raise RuntimeError("class-preserving cap dropped a class; refusing to train.")
    if capped[target_column].value_counts().min() < MIN_CLASS_COUNT:
        raise ValueError("Capped training split leaves fewer than 2 rows for a class; provide a representative pre-split set.")
    return capped, {"applied": True, "before": before, "after": len(capped)}


def split_overlap_report(named_frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Detect exact record overlap across labelled partitions without mutating them."""
    names = list(named_frames)
    hashes = {
        name: set(pd.util.hash_pandas_object(frame, index=False).astype("uint64").tolist())
        for name, frame in named_frames.items()
    }
    report: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report[f"{left}_vs_{right}"] = len(hashes[left].intersection(hashes[right]))
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def fit_mitra_predictor(
    train_data: pd.DataFrame,
    *,
    target_column: str,
    eval_metric: str,
    path: str | Path,
    fine_tune: bool,
    time_limit: int,
    seed: int,
    problem_type: str | None = None,
    fine_tune_steps: int | None = None,
    max_memory_usage_ratio: float = 1.10,
):
    """Fit/register Mitra through the repository's supported notebook API.

    With ``fine_tune=False`` AutoGluon's ``fit`` registers support/context rows and model
    configuration; Mitra weights are not gradient-updated. With ``fine_tune=True`` the Mitra
    weights are adapted for the requested number of steps, subject to the time limit.
    ``problem_type`` defaults to ``binary`` for two classes and ``multiclass`` otherwise.
    """
    if eval_metric not in EVAL_METRICS:
        raise ValueError(f"Unsupported eval_metric {eval_metric!r}; choose {list(EVAL_METRICS)}")
    if fine_tune and (fine_tune_steps is None or fine_tune_steps <= 0):
        raise ValueError("fine_tune_steps must be a positive integer when fine_tune=True.")
    n_classes = int(train_data[target_column].nunique())
    if problem_type is None:
        problem_type = "binary" if n_classes == 2 else "multiclass"
    if problem_type not in ("binary", "multiclass"):
        raise ValueError("problem_type must be 'binary' or 'multiclass'")

    seed_everything(seed)
    hp: dict[str, Any] = {"fine_tune": bool(fine_tune), "seed": int(seed)}
    if eval_metric in MITRA_METRIC_MAP:
        hp["metric"] = MITRA_METRIC_MAP[eval_metric]
    if fine_tune:
        hp["fine_tune_steps"] = int(fine_tune_steps)

    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=target_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(path),
        verbosity=2,
    )
    predictor.fit(
        train_data,
        hyperparameters={"MITRA": hp},
        fit_weighted_ensemble=False,
        time_limit=int(time_limit),
        ag_args_fit={"max_memory_usage_ratio": float(max_memory_usage_ratio)},
    )
    trained = list(predictor.model_names())
    if not trained or not any("mitra" in model.lower() for model in trained):
        raise RuntimeError(f"Expected Mitra to fit/register; AutoGluon returned models={trained}.")
    return predictor


def normalize_autogluon_classification_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """AutoGluon reports lower-is-better metrics negated (log_loss); restore positive values."""
    return {
        str(key): float(-value if key in CLASSIFICATION_LOWER_IS_BETTER else value)
        for key, value in raw.items()
    }


def classification_metrics(
    y_true: Iterable[Any], y_pred: Iterable[Any], y_proba: Any, classes: Sequence[Any]
) -> dict[str, float]:
    """The repository's metric set: accuracy, balanced_accuracy, f1_macro, mcc (discrete correctness under
    the argmax rule), log_loss (probability quality; lower is better) and roc_auc (ranking; binary or one-vs-rest;
    NaN when undefined). ``y_proba`` has one column per entry of ``classes`` in that order."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    truth = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    proba = np.asarray(y_proba, dtype=float)
    labels = list(classes)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError(f"y_true shape {truth.shape} != y_pred shape {pred.shape} (or empty)")
    if proba.ndim != 2 or proba.shape != (truth.size, len(labels)):
        raise ValueError("y_proba must have one row per example and one column per class")
    out = {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "log_loss": float(log_loss(truth, proba, labels=labels)),
        "f1_macro": float(f1_score(truth, pred, average="macro")),
        "mcc": float(matthews_corrcoef(truth, pred)),
    }
    try:
        if len(labels) == 2:
            out["roc_auc"] = float(roc_auc_score(truth, proba[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(truth, proba, multi_class="ovr", labels=labels))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def validate_inference_frame(
    frame: pd.DataFrame,
    required_features: Iterable[str],
    *,
    output_column: str = "prediction",
) -> tuple[pd.DataFrame, list[str]]:
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"Inference CSV contains duplicate column names: {duplicates}")
    required = list(required_features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Inference CSV is missing required feature columns: {missing}")
    reserved = [c for c in frame.columns if c == output_column or str(c).startswith("probability_")]
    if reserved:
        raise ValueError(f"Inference CSV already contains output column(s) reserved by this notebook: {reserved}; rename or remove them first.")
    extra = [column for column in frame.columns if column not in required]
    return frame.reindex(columns=required).copy(), extra


def predict_classification(predictor, frame: pd.DataFrame, required_features: Iterable[str]) -> np.ndarray:
    """Class labels (the argmax of the class probabilities), one per row."""
    features = list(required_features)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing required features: {missing}")
    return np.asarray(predictor.predict(frame.reindex(columns=features)))


def predict_proba_classification(
    predictor, frame: pd.DataFrame, required_features: Iterable[str], classes: Sequence[Any] | None = None
) -> pd.DataFrame:
    """Class probabilities as a DataFrame with one column per class, in ``classes`` order (default: the
    predictor's ``class_labels``); values are finite and rows sum to ~1."""
    features = list(required_features)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing required features: {missing}")
    proba = predictor.predict_proba(frame.reindex(columns=features), as_multiclass=True)
    order = list(classes) if classes is not None else list(predictor.class_labels)
    absent = [label for label in order if label not in proba.columns]
    if absent:
        raise RuntimeError(f"predictor never saw classes: {absent}")
    proba = proba[order]
    values = proba.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Mitra produced non-finite class probabilities")
    return proba


def align_probabilities(proba: Any, model_classes: Sequence[Any], classes: Sequence[Any]) -> np.ndarray:
    """Reorder probability columns from ``model_classes`` order into ``classes`` order (strict: every class
    must be a column, so blending two models can never permute labels silently)."""
    proba = np.asarray(proba, dtype=float)
    lookup = {label: index for index, label in enumerate(model_classes)}
    missing = [label for label in classes if label not in lookup]
    if missing:
        raise RuntimeError(f"model never saw classes: {missing}")
    return proba[:, [lookup[label] for label in classes]]


def _download(url: str, destination: Path, *, timeout: int = 30) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def stage_verified_hf_snapshot(
    weights_path: str | Path,
    config_path: str | Path,
    *,
    hf_home: str | Path,
) -> Path:
    """Verify exact Mitra bytes and stage them as the immutable pinned Hugging Face snapshot."""
    weights = Path(weights_path)
    config = Path(config_path)
    if sha256_file(weights) != WEIGHTS_SHA256:
        raise RuntimeError("model.safetensors checksum mismatch for the pinned Mitra Classifier release.")
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("config.json checksum mismatch for the pinned Mitra Classifier release.")

    home = Path(hf_home)
    repo = home / "hub" / ("models--" + MODEL_ID.replace("/", "--"))
    snapshot = repo / "snapshots" / PINNED_REVISION
    refs = repo / "refs"
    snapshot.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, snapshot / "model.safetensors")
    shutil.copy2(config, snapshot / "config.json")
    (refs / "main").write_text(PINNED_REVISION, encoding="utf-8")

    import huggingface_hub.constants

    real_cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    try:
        if real_cache.resolve() != (home / "hub").resolve():
            real_repo = real_cache / ("models--" + MODEL_ID.replace("/", "--"))
            real_snap = real_repo / "snapshots" / PINNED_REVISION
            real_refs = real_repo / "refs"
            real_snap.mkdir(parents=True, exist_ok=True)
            real_refs.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights, real_snap / "model.safetensors")
            shutil.copy2(config, real_snap / "config.json")
            (real_refs / "main").write_text(PINNED_REVISION, encoding="utf-8")
    except Exception:
        pass

    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from huggingface_hub import hf_hub_download

    for filename, expected_digest in (
        ("model.safetensors", WEIGHTS_SHA256),
        ("config.json", CONFIG_SHA256),
    ):
        try:
            resolved = Path(
                hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=filename,
                    revision=PINNED_REVISION,
                    cache_dir=str(home / "hub"),
                    local_files_only=True,
                )
            ).resolve()
        except Exception:
            resolved = Path(
                hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=filename,
                    revision=PINNED_REVISION,
                    local_files_only=True,
                )
            ).resolve()
        expected = (snapshot / filename).resolve()
        if resolved != expected and not resolved.is_file():
            raise RuntimeError(f"Offline resolver mismatch for {filename}: {resolved} != {expected}")
        if sha256_file(resolved) != expected_digest:
            raise RuntimeError(f"Resolved {filename} digest changed after staging.")
    return snapshot


def _validate_zip_members(zf: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    destination = destination.resolve()
    total = 0
    files: list[zipfile.ZipInfo] = []
    seen_paths: set[str] = set()
    seen_parent_paths: set[str] = set()
    for info in zf.infolist():
        name = info.filename
        if not name or info.is_dir():
            continue
        if "\\" in name:
            raise RuntimeError(f"Backslash archive paths are not allowed: {name!r}")
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe archive member path: {name!r}")
        normalized_name = member.as_posix()
        if normalized_name in seen_paths:
            raise RuntimeError(f"Duplicate archive member path is not allowed: {name!r}")
        parent_paths = {PurePosixPath(*member.parts[:i]).as_posix() for i in range(1, len(member.parts))}
        if normalized_name in seen_parent_paths or parent_paths.intersection(seen_paths):
            raise RuntimeError(f"Archive member path conflicts with a file/directory boundary: {name!r}")
        seen_paths.add(normalized_name)
        seen_parent_paths.update(parent_paths)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise RuntimeError(f"Symlink entries are not allowed: {name!r}")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeError(f"Archive member is too large: {name!r} ({info.file_size:,} bytes).")
        if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise RuntimeError(f"Archive member has suspicious compression ratio: {name!r}.")
        total += info.file_size
        if total > MAX_ARCHIVE_EXPANDED_BYTES:
            raise RuntimeError("Archive expanded size exceeds the configured safety ceiling.")
        target = (destination / Path(*member.parts)).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Archive member escapes extraction root: {name!r}")
        files.append(info)
    return files


def safe_extract_archive(zip_path: str | Path, destination: str | Path) -> Path:
    destination_path = Path(destination)
    with zipfile.ZipFile(zip_path) as zf:
        # Validate the complete archive before touching any prior extraction destination.
        infos = _validate_zip_members(zf, destination_path)
        if destination_path.is_symlink():
            raise RuntimeError("Archive extraction destination must not be a symlink.")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.extract-", dir=destination_path.parent))
        try:
            for info in infos:
                member = PurePosixPath(info.filename)
                target = staging.joinpath(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
            if destination_path.exists():
                shutil.rmtree(destination_path)
            staging.replace(destination_path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination_path


def validate_dimer_model_package(
    zip_path: str | Path,
    destination: str | Path,
    *,
    expected_archive_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate the normative DIMER offline Mitra package before any model load."""
    zip_path = Path(zip_path)
    if expected_archive_sha256:
        expected = _assert_hex_digest(expected_archive_sha256, "expected_archive_sha256")
        actual = sha256_file(zip_path)
        if actual != expected:
            raise RuntimeError(f"DIMER model ZIP checksum mismatch: expected {expected}; got {actual}.")

    root = safe_extract_archive(zip_path, destination)
    manifest_path = root / DIMER_MODEL_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"DIMER model ZIP must contain root-level {DIMER_MODEL_MANIFEST}; legacy weight-only ZIPs do not satisfy Notebook Spec 1.0."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported DIMER model manifest schema_version.")
    if manifest.get("model_id") != MODEL_ID:
        raise RuntimeError(f"DIMER package model_id must be {MODEL_ID!r}.")
    if manifest.get("revision") != PINNED_REVISION:
        raise RuntimeError(f"DIMER package revision must be {PINNED_REVISION!r}.")

    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("DIMER model manifest must contain a non-empty files list.")
    entries: dict[str, dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise RuntimeError("DIMER model manifest file entries must be objects.")
        path = str(entry.get("path", ""))
        if not path or "\\" in path:
            raise RuntimeError(f"Invalid DIMER manifest path: {path!r}")
        member = PurePosixPath(path)
        if member.is_absolute() or ".." in member.parts or path in entries:
            raise RuntimeError(f"Unsafe or duplicate DIMER manifest path: {path!r}")
        entries[path] = entry

    for required in ("model.safetensors", "config.json"):
        if required not in entries:
            raise RuntimeError(f"DIMER model manifest is missing required file {required!r}.")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / DIMER_MODEL_MANIFEST
    }
    if actual_files != set(entries):
        missing = sorted(set(entries) - actual_files)
        unexpected = sorted(actual_files - set(entries))
        raise RuntimeError(f"DIMER model package file inventory mismatch; missing={missing}, unexpected={unexpected}")

    for rel, entry in entries.items():
        path = root / rel
        expected_size = int(entry.get("size", -1))
        expected_digest = _assert_hex_digest(str(entry.get("sha256", "")), f"sha256 for {rel}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"DIMER model package size mismatch for {rel}.")
        if sha256_file(path) != expected_digest:
            raise RuntimeError(f"DIMER model package digest mismatch for {rel}.")

    weights = root / "model.safetensors"
    config = root / "config.json"
    if sha256_file(weights) != WEIGHTS_SHA256 or sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("DIMER package bytes do not match the pinned Mitra Classifier release.")
    return weights, config, manifest


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != root / ARTIFACT_MANIFEST):
        rel = path.relative_to(root).as_posix()
        if "\\" in rel or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
            raise RuntimeError(f"Unsafe artifact path: {rel!r}")
        inventory.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return inventory


def write_artifact_manifest(root: str | Path) -> Path:
    root = Path(root)
    metadata_path = root / RUN_METADATA
    if not metadata_path.is_file():
        raise RuntimeError(f"Cannot manifest artifact without {RUN_METADATA}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_RUN_METADATA_FIELDS - set(metadata))
    if missing:
        raise RuntimeError(f"Run metadata missing required fields: {missing}")
    if metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise RuntimeError("Run metadata artifact_format mismatch.")
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Run metadata artifact_format_version mismatch.")

    manifest = {
        "schema_version": 1,
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "base_model": metadata["base_model"],
        "base_model_revision": metadata["base_model_revision"],
        "problem_type": metadata["problem_type"],
        "metadata_file": RUN_METADATA,
        "files": _artifact_inventory(root),
    }
    path = root / ARTIFACT_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_artifact_directory(
    root: str | Path,
    *,
    expected_model_id: str = MODEL_ID,
    expected_revision: str = PINNED_REVISION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify format, provenance, inventory, sizes and digests before deserializing predictor state."""
    root = Path(root)
    manifest_path = root / ARTIFACT_MANIFEST
    metadata_path = root / RUN_METADATA
    if not manifest_path.is_file():
        raise RuntimeError(f"Predictor artifact is missing required {ARTIFACT_MANIFEST}.")
    if not metadata_path.is_file():
        raise RuntimeError(f"Predictor artifact is missing required {RUN_METADATA}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported artifact manifest schema_version.")
    if manifest.get("metadata_file") != RUN_METADATA:
        raise RuntimeError(f"Artifact manifest metadata_file must be {RUN_METADATA!r}.")
    if manifest.get("artifact_format") != ARTIFACT_FORMAT or metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise RuntimeError("Predictor artifact_format is not the DIMER AutoGluon predictor format.")
    if (
        manifest.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION
        or metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION
    ):
        raise RuntimeError("Predictor artifact format version is incompatible with this notebook.")

    missing = sorted(REQUIRED_RUN_METADATA_FIELDS - set(metadata))
    if missing:
        raise RuntimeError(f"Predictor provenance is missing required fields: {missing}")
    if metadata.get("base_model") != expected_model_id or manifest.get("base_model") != expected_model_id:
        raise RuntimeError(f"Predictor base model must be {expected_model_id!r}.")
    if metadata.get("base_model_revision") != expected_revision or manifest.get("base_model_revision") != expected_revision:
        raise RuntimeError(f"Predictor base model revision must be {expected_revision!r}.")
    if metadata.get("problem_type") not in ("binary", "multiclass") or manifest.get("problem_type") not in ("binary", "multiclass"):
        raise RuntimeError("Predictor artifact is not a classification (binary/multiclass) artifact.")
    if metadata.get("weights_sha256") != WEIGHTS_SHA256 or metadata.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("Predictor provenance does not identify the pinned Mitra Classifier bytes.")

    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("Artifact manifest contains no file inventory.")
    entries: dict[str, dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise RuntimeError("Artifact manifest file entries must be objects.")
        rel = str(entry.get("path", ""))
        member = PurePosixPath(rel)
        if not rel or "\\" in rel or member.is_absolute() or ".." in member.parts or rel in entries:
            raise RuntimeError(f"Unsafe or duplicate artifact manifest path: {rel!r}")
        entries[rel] = entry

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / ARTIFACT_MANIFEST
    }
    if actual != set(entries):
        missing_files = sorted(set(entries) - actual)
        unexpected_files = sorted(actual - set(entries))
        raise RuntimeError(
            f"Artifact inventory mismatch; missing={missing_files}, unexpected={unexpected_files}"
        )

    for rel, entry in entries.items():
        path = root / rel
        expected_size = int(entry.get("size", -1))
        expected_digest = _assert_hex_digest(str(entry.get("sha256", "")), f"sha256 for {rel}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Artifact file size mismatch: {rel}")
        if sha256_file(path) != expected_digest:
            raise RuntimeError(f"Artifact file digest mismatch: {rel}")

    if not (root / "predictor.pkl").is_file():
        raise RuntimeError("Artifact does not contain the required root-level AutoGluon predictor.pkl.")
    return manifest, metadata


# ---------------------------------------------------------------------------
# Fleet snapshot scheme (NOTEBOOK_SPEC 1.1 ST3/ST4, MOD13): manifest-driven verification and staging. The existing
# `stage_verified_hf_snapshot` (digest check + offline HF cache staging) stays the loader path and is called by
# `MitraClassificationPipeline.from_pretrained` after the manifest has been verified.
# ---------------------------------------------------------------------------


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local pinned snapshot against its manifest; raise naming the first mismatch.

    The manifest entries for ``model.safetensors`` and ``config.json`` must equal the package's own
    ``WEIGHTS_SHA256`` / ``CONFIG_SHA256`` constants, so the two can never diverge silently.
    """
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = manifest.get("files", [])
    declared = {entry["path"]: entry["sha256"] for entry in entries}
    for filename, expected in ((WEIGHTS_FILE, WEIGHTS_SHA256), (CONFIG_FILE, CONFIG_SHA256)):
        if declared.get(filename) != expected:
            raise ValueError(
                f"manifest {filename} sha256 {declared.get(filename)!r} != package constant {expected!r}"
            )
    for entry in entries:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=MODEL_ID, filename=relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a clone commits the manifest but git-ignores the
    weights). Returns the relative paths fetched; ``verify_snapshot`` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


class MitraClassificationPipeline:
    """Serving wrapper: the digest-verified pinned snapshot behind AutoGluon's Mitra classifier.

    ``from_pretrained`` stages and verifies the snapshot, then stages the verified bytes as the immutable offline
    Hugging Face snapshot AutoGluon resolves (``stage_verified_hf_snapshot``: HF_HUB_OFFLINE, no network path).
    ``fit`` registers the support rows through ``fit_mitra_predictor`` (in-context; gradient fine-tuning only with
    ``fine_tune=True``); ``evaluate``, ``predict`` (``argmax`` label) and ``predict_proba`` (one column per class
    in ``class_labels`` order) go through the repository's AutoGluon helpers.
    """

    def __init__(
        self,
        *,
        weights_path: Path,
        config_path: Path,
        snapshot_path: Path,
        device: str,
        source: str = "local-snapshot",
        predictor: Any = None,
        features: Sequence[str] | None = None,
    ) -> None:
        self.model_weight_path = Path(weights_path)
        self.config_path = Path(config_path)
        self.snapshot_path = Path(snapshot_path)
        self.device = device
        self.source = source
        self.predictor = predictor
        self.features: list[str] = list(features or [])
        self.target_column: str | None = None
        self.problem_type: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
        hf_home: str | Path | None = None,
        device: str | None = None,
    ) -> MitraClassificationPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        home = Path(hf_home) if hf_home is not None else root / ".cache" / "hf"
        snapshot = stage_verified_hf_snapshot(root / WEIGHTS_FILE, root / CONFIG_FILE, hf_home=home)
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return cls(
            weights_path=root / WEIGHTS_FILE, config_path=root / CONFIG_FILE, snapshot_path=snapshot, device=device
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        *,
        target_column: str,
        eval_metric: str,
        path: str | Path,
        fine_tune: bool = False,
        time_limit: int = 300,
        seed: int = 42,
        problem_type: str | None = None,
        fine_tune_steps: int | None = None,
        max_memory_usage_ratio: float = 1.10,
    ) -> MitraClassificationPipeline:
        self.predictor = fit_mitra_predictor(
            train_data,
            target_column=target_column,
            eval_metric=eval_metric,
            path=path,
            fine_tune=fine_tune,
            time_limit=time_limit,
            seed=seed,
            problem_type=problem_type,
            fine_tune_steps=fine_tune_steps,
            max_memory_usage_ratio=max_memory_usage_ratio,
        )
        self.target_column = target_column
        self.problem_type = str(self.predictor.problem_type)
        self.features = [column for column in train_data.columns if column != target_column]
        self.source = "fine-tuned" if fine_tune else self.source
        return self

    @property
    def class_labels(self) -> list[Any]:
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        return list(self.predictor.class_labels)

    def evaluate(self, frame: pd.DataFrame) -> dict[str, float]:
        """AutoGluon's own evaluation of a labelled frame, normalised so log_loss is positive."""
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        raw = self.predictor.evaluate(frame, auxiliary_metrics=True, silent=True)
        return normalize_autogluon_classification_metrics(raw)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        return predict_classification(self.predictor, frame, self.features)

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        return predict_proba_classification(self.predictor, frame, self.features, self.class_labels)


# ---------------------------------------------------------------------------
# Role stages (DAT24 / EVAL21) on top of the existing validation and metric helpers.
# ---------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": "pandas.DataFrame, one row per example; feature columns of any dtype plus a categorical target",
    "columns": "unique names; `drop_columns` are removed before validation",
    "target": (
        "any label type; rows with a missing target are dropped and counted; MIN_CLASSES..MAX_CLASSES classes; "
        "every class needs MIN_CLASS_COUNT rows in the support split; holdout/test classes must match the support classes"
    ),
    "train_rows": [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS],
    "eval_rows": [2, None],
    "features": [1, MAX_FEATURES],
    "classes": [MIN_CLASSES, MAX_CLASSES],
    "inference_input": "every fitted feature column present; no `prediction`/`probability_*` column; extras pass through",
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "none by the package (AutoGluon's Mitra handles raw columns); training rows above MAX_TRAIN_ROWS are "
        "capped by seeded class-preserving sampling and the cap is reported"
    ),
}


def majority_class_baseline(train_targets: Iterable[Any], holdout_targets: Iterable[Any]) -> dict[str, float]:
    """The trivial baseline: always predict the most frequent support class with the support class shares as
    probabilities (log_loss defined; roc_auc NaN because constant scores rank nothing)."""
    train = pd.Series(list(train_targets))
    holdout = np.asarray(list(holdout_targets))
    if train.size == 0 or holdout.size == 0:
        raise ValueError("targets must be non-empty")
    shares = train.value_counts(normalize=True).sort_index()
    classes = list(shares.index)
    unseen = sorted(set(holdout) - set(classes), key=str)
    if unseen:
        raise ValueError(f"holdout has target classes unseen in training: {unseen}")
    majority = shares.idxmax()
    proba = np.tile(shares.to_numpy(dtype=float), (holdout.size, 1))
    metrics = classification_metrics(holdout, np.full(holdout.shape, majority, dtype=object), proba, classes)
    metrics["roc_auc"] = float("nan")
    return metrics


def validate_inputs(
    frame: pd.DataFrame,
    target_column: str | None = "target",
    *,
    drop_columns: Iterable[str] = (),
    min_rows: int = MIN_TRAIN_ROWS,
    min_classes: int = MIN_CLASSES,
    min_class_count: int = MIN_CLASS_COUNT,
    feature_columns: Iterable[str] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed table properties, verdict).

    With a ``target_column`` the table is checked exactly as ``validate_labeled_frame`` checks it (its report:
    dropped missing-target rows, exact duplicates, feature count, class counts — is carried, not hidden); with
    ``target_column=None`` it is an inference table checked exactly as ``validate_inference_frame`` checks it.
    Rejection is reported by raising the same error the core helper raises.
    """
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (the table's id)")
    table_id = names[0] if names else "table-0"
    if target_column is None:
        if feature_columns is None:
            raise ValueError("feature_columns is required to validate an inference table")
        checked, extra = validate_inference_frame(frame, feature_columns)
        entry: dict[str, Any] = {
            "id": table_id,
            "mode": "inference",
            "rows": len(checked),
            "feature_columns": list(checked.columns),
            "extra_columns": extra,
            "missing_value_columns": {str(c): int(n) for c, n in checked.isna().sum().items() if n > 0},
        }
    else:
        clean, features, report = validate_labeled_frame(
            frame,
            target_column,
            name=table_id,
            drop_columns=drop_columns,
            min_rows=min_rows,
            min_classes=min_classes,
            min_class_count=min_class_count,
        )
        counts = clean[target_column].value_counts()
        entry = {
            "id": table_id,
            "mode": "fit",
            "rows": len(clean),
            "feature_columns": features,
            "categorical_columns": [c for c in features if not pd.api.types.is_numeric_dtype(clean[c])],
            "missing_value_columns": {
                str(c): int(n) for c, n in clean[features].isna().sum().items() if n > 0
            },
            "report": report,
            "classes": [str(label) for label in counts.sort_index().index],
            "class_counts": report["class_counts"],
            "majority_class_share": float(counts.max() / counts.sum()),
        }
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [entry],
        "target_column": target_column,
        "drop_columns": list(drop_columns),
        "min_rows": min_rows if target_column is not None else None,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    metrics: Mapping[str, float] | None,
    *,
    baseline: Mapping[str, float] | None = None,
    independent_test: Mapping[str, float] | None = None,
    n_holdout: int | None = None,
    n_test: int | None = None,
    class_labels: Sequence[Any] | None = None,
    target_column: str | None = None,
    selection: str | None = None,
    sample_kind: str = "sample",
    estimation: str = "single seeded stratified split; no dispersion estimate",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` / ``independent_test`` are dicts from ``classification_metrics`` (or AutoGluon's evaluate through
    ``normalize_autogluon_classification_metrics``) and ``baseline`` from ``majority_class_baseline``; the verdict
    is ``sample-sanity``. Without metrics (no labelled rows) the verdict is ``not-measurable`` and the report says
    what labelled data would make the task measurable.
    """

    def _entries(source: Mapping[str, float]) -> list[dict[str, Any]]:
        unknown = sorted(set(source) - set(METRIC_IDS))
        if unknown:
            raise ValueError(f"unknown metric ids {unknown}; the repository reports {list(METRIC_IDS)}")
        return [
            {
                "id": metric_id,
                "value": None if not np.isfinite(float(source[metric_id])) else float(source[metric_id]),
                "units": "nats" if metric_id == "log_loss" else "unitless",
                "higher_is_better": metric_id not in CLASSIFICATION_LOWER_IS_BETTER,
            }
            for metric_id in METRIC_IDS
            if metric_id in source
        ]

    base: dict[str, Any] = {
        "task": "tabular classification by in-context conditioning on labelled support rows (AutoGluon Mitra)",
        "decision_rule": DECISION_RULE,
        "score_semantics": "class probabilities in class_labels order, uncalibrated; no threshold shipped",
        "sample_kind": sample_kind,
        "n_holdout": n_holdout,
        "n_test": n_test,
        "class_labels": None if class_labels is None else [str(label) for label in class_labels],
        "target_column": target_column,
        "selection": selection,
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if metrics is None:
        return {
            **base,
            "metrics": [],
            "independent_test": [],
            "verdict": "not-measurable",
            "reason": "no labelled holdout rows were supplied for the scored table",
            "needs": (
                "a labelled holdout table whose classes all appear in the support rows, scored with "
                "`classification_metrics` (accuracy, balanced_accuracy, log_loss, roc_auc, f1_macro, mcc) against "
                "`majority_class_baseline`; an independent test partition from the deployment domain for any "
                "generalisable claim"
            ),
        }
    reported = [{**entry, "estimation": estimation} for entry in _entries(metrics)]
    test_entries: list[dict[str, Any]] = []
    if independent_test is not None:
        test_estimation = "independent test partition, single run"
        test_entries = [{**e, "estimation": test_estimation} for e in _entries(independent_test)]
    baselines = [] if baseline is None else [{"id": "majority_class", "metrics": _entries(baseline)}]
    rows = "an unstated number of" if n_holdout is None else str(n_holdout)
    return {
        **base,
        "metrics": reported,
        "independent_test": test_entries,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": f"{rows} labelled holdout row(s) from one seeded stratified split; tutorial evidence, not a benchmark",
        "needs": (
            "an independent, domain-representative labelled test set for any generalisable quality claim; the "
            "class probabilities are uncalibrated and any decision threshold must be chosen on the caller's data"
        ),
    }
