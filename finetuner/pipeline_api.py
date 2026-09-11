"""Public production-facing Mitra classifier pipeline API.

This module owns the reusable data-validation, split/cap, fit/evaluate, and prediction
contract shared by the DIMER fine-tuner and the user-facing notebooks. Keep environment
parsing, callbacks, and artifact-export plumbing in ``train.py``; keep model/data behavior here.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_MODEL = "autogluon/mitra-classifier"
PINNED_MITRA_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
EXPECTED_WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
EXPECTED_CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
CANONICAL_CONFIG_BYTES = b'{"dim": 512, "dim_output": 10, "n_layers": 12, "n_heads": 4, "task": "CLASSIFICATION"}'
MITRA_MODEL_KEY = "MITRA"
MITRA_ROW_LIMIT = 10_000
MITRA_CLASS_LIMIT = 10
MAX_DIMER_WEIGHTS_BYTES = 1 * 1024**3


def materialize_dimer_checkpoint(
    weights_payload: bytes,
    weights_dest: str | Path,
    config_dest: str | Path,
    *,
    max_weights_bytes: int = MAX_DIMER_WEIGHTS_BYTES,
) -> dict[str, str]:
    # MODEL_CARD.md: DIMER hosts model.safetensors only. Reconstruct canonical config locally.
    if not isinstance(weights_payload, (bytes, bytearray)):
        raise TypeError("DIMER model.safetensors payload must be bytes.")
    payload = bytes(weights_payload)
    if len(payload) > max_weights_bytes:
        raise ValueError(
            f"DIMER model.safetensors is {len(payload):,} bytes; refusing payload above {max_weights_bytes:,} bytes."
        )
    weights_digest = hashlib.sha256(payload).hexdigest()
    if weights_digest != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"DIMER model.safetensors checksum mismatch. Expected {EXPECTED_WEIGHTS_SHA256}; got {weights_digest}."
        )
    config_digest = hashlib.sha256(CANONICAL_CONFIG_BYTES).hexdigest()
    if len(CANONICAL_CONFIG_BYTES) != 86 or config_digest != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Repository canonical Mitra config bytes failed their pinned invariant.")
    weights_path = Path(weights_dest)
    config_path = Path(config_dest)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(payload)
    config_path.write_bytes(CANONICAL_CONFIG_BYTES)
    return {"weights_sha256": weights_digest, "config_sha256": config_digest}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_tabular_frame(
    frame: pd.DataFrame,
    target_column: str,
    *,
    drop_columns: Iterable[str] = (),
    name: str = "data",
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Apply the canonical production-facing structural preprocessing contract."""
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"{name}: duplicate column names are not supported: {duplicates}")
    if target_column not in frame.columns:
        raise ValueError(f"{name}: target {target_column!r} not found.")

    dropped = [c for c in drop_columns if c in frame.columns and c != target_column]
    null_target_rows = int(frame[target_column].isna().sum())
    clean = frame.drop(columns=dropped, errors="ignore").dropna(subset=[target_column]).copy()
    features = [c for c in clean.columns if c != target_column]
    mutations: dict[str, object] = {
        "dropped_feature_columns": dropped,
        "null_target_rows_dropped": null_target_rows,
    }
    return clean, features, mutations


def validate_classification_target(
    frame: pd.DataFrame,
    target_column: str,
    *,
    name: str = "data",
    min_rows: int = 1,
    max_features: int | None = None,
    min_classes: int = 2,
    max_classes: int = MITRA_CLASS_LIMIT,
    min_class_count: int = 1,
) -> list[str]:
    if target_column not in frame.columns:
        raise ValueError(f"{name}: target {target_column!r} not found.")
    features = [c for c in frame.columns if c != target_column]
    counts = frame[target_column].value_counts()
    errors: list[str] = []
    if len(frame) < min_rows:
        errors.append(f"use at least {min_rows} labelled rows")
    if not features:
        errors.append("no feature columns remain")
    if max_features is not None and len(features) > max_features:
        errors.append(f"{len(features)} features exceed the {max_features}-feature limit")
    if not min_classes <= len(counts) <= max_classes:
        errors.append(f"target has {len(counts)} classes; Mitra requires {min_classes}–{max_classes}")
    if counts.empty or counts.min() < min_class_count:
        errors.append(f"every class needs at least {min_class_count} row(s)")
    if errors:
        raise ValueError(f"{name} is not ready: " + "; ".join(errors))
    return features


def require_class_coverage(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target_column: str,
    eval_name: str,
) -> None:
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
        raise ValueError(f"{eval_name} " + "; ".join(problems) + ". Adjust the split or provide a compatible evaluation partition.")


def stratified_holdout(
    frame: pd.DataFrame,
    target_column: str,
    validation_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    train, holdout = train_test_split(
        frame, test_size=n_val, random_state=seed, stratify=frame[target_column]
    )
    if set(train[target_column].unique()) != set(frame[target_column].unique()):
        raise RuntimeError("stratified split left a class out of train; refusing to train.")
    return train, holdout


def stratified_cap(
    frame: pd.DataFrame,
    target_column: str,
    ceiling: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) <= ceiling:
        return frame
    classes_before = set(frame[target_column].unique())
    if ceiling < len(classes_before):
        raise ValueError(f"row ceiling ({ceiling}) is below the class count ({len(classes_before)}); cannot keep every class.")
    rng = np.random.RandomState(seed)
    keep: list[object] = []
    for cls in sorted(classes_before, key=str):
        idx = frame.index[frame[target_column] == cls].to_numpy()
        keep.append(rng.choice(idx, size=1)[0])
    keep_set = set(keep)
    remaining = np.array([i for i in frame.index.to_numpy() if i not in keep_set])
    need = ceiling - len(keep)
    if need > 0 and len(remaining) > 0:
        keep.extend(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
    capped = frame.loc[keep]
    if set(capped[target_column].unique()) != classes_before:
        raise RuntimeError("class-preserving cap dropped a class; refusing to train.")
    return capped


def fit_mitra_predictor(
    train: pd.DataFrame,
    *,
    target_column: str,
    problem_type: str,
    eval_metric: str,
    output_path: str | Path,
    fine_tune: bool,
    seed: int,
    time_limit: int,
    fine_tune_steps: int | None = None,
    mitra_metric: str | None = None,
    max_memory_usage_ratio: float | None = None,
    verbosity: int = 2,
):
    """Fit/register the canonical Mitra predictor used by DIMER and tutorials."""
    from autogluon.tabular import TabularPredictor

    seed_everything(seed)
    hp: dict[str, object] = {"fine_tune": bool(fine_tune), "seed": int(seed)}
    if fine_tune and fine_tune_steps:
        hp["fine_tune_steps"] = int(fine_tune_steps)
    if mitra_metric:
        hp["metric"] = mitra_metric

    predictor = TabularPredictor(
        label=target_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(output_path),
        verbosity=verbosity,
    )
    fit_kwargs: dict[str, object] = {
        "hyperparameters": {MITRA_MODEL_KEY: hp},
        "fit_weighted_ensemble": False,
        "time_limit": int(time_limit),
    }
    if max_memory_usage_ratio is not None:
        fit_kwargs["ag_args_fit"] = {"max_memory_usage_ratio": float(max_memory_usage_ratio)}
    predictor.fit(train, **fit_kwargs)
    trained = list(predictor.model_names())
    if not trained or not any("mitra" in name.lower() for name in trained):
        raise RuntimeError(f"Expected Mitra; AutoGluon trained {trained}.")
    return predictor


def evaluate_mitra(predictor, frame: pd.DataFrame) -> dict[str, float]:
    raw = predictor.evaluate(frame, auxiliary_metrics=True, silent=True)
    return {str(k): float(-v if "log_loss" in str(k) else v) for k, v in raw.items()}


def predict_mitra(predictor, frame: pd.DataFrame):
    return predictor.predict(frame)


def predict_mitra_proba(predictor, frame: pd.DataFrame):
    return predictor.predict_proba(frame, as_multiclass=True)
