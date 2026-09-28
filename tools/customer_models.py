"""Standalone numeric scorers and pinned Mitra in-context inference.

No pickled predictors, gradient training, target-driven support selection, or
repository imports. The caller runs pretrained inference only on hosted GPUs.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import urllib.request
import warnings
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

FEATURE_NAMES = ("recency_days", "order_count", "gross_value", "avg_order_value", "product_count")
FEATURE_UNITS = ("days", "orders", "source currency", "source currency per order", "products")
REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
PINNED_FILES = {
    "config.json": (86, "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"),
    "model.safetensors": (302717904, "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"),
}
SETTINGS = {
    "seed": 42,
    "max_support": 2048,
    "query_batch_size": 64,
    "n_estimators": 1,
    "fine_tune": False,
    "fine_tune_steps": 0,
    "precision": "float32",
    "shuffle_classes": False,
    "shuffle_features": False,
    "random_mirror_x": False,
    "random_mirror_regression": False,
    "use_random_transforms": False,
    "use_feature_count_scaling": False,
    "use_quantile_transformer": False,
    "mitra_preprocessing": "upstream support-fitted singular-feature removal and support-only GPU quantiles",
    "support_policy": "all rows; seeded permutation per query chunk",
    "feature_transform": "raw recency; log1p all other nonnegative features",
    "logistic": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "scikit_learn": "1.7.2",
        "scaling": "training-only mean and population standard deviation",
    },
}


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _features(X: Any, feature_names=FEATURE_NAMES) -> np.ndarray:
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Feature order/schema mismatch")
    if hasattr(X, "columns") and tuple(X.columns) != FEATURE_NAMES:
        raise ValueError("DataFrame must contain exactly the ordered five feature columns")
    values = np.array(X, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite N by 5 feature matrix")
    if np.any(values < 0):
        raise ValueError("Historical features must be nonnegative")
    return values


def _transform(raw: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "R":
        return raw[:, :1].copy()
    if feature_set != "RFM":
        raise ValueError("Unknown feature set")
    values = raw.copy()
    values[:, 1:] = np.log1p(values[:, 1:])
    return values


def positive_probability(probabilities: Any, classes: Any) -> np.ndarray:
    """Select P(actual label 1), including reversed class-column order."""
    classes = np.asarray(classes)
    values = np.asarray(probabilities, dtype=float)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Classes must identify labels 0 and 1 exactly")
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid probability shape or finiteness")
    if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities must be in [0,1] and sum to one")
    return values[:, int(np.flatnonzero(classes == 1)[0])]


def fit(kind: str, X: Any, y: Any, feature_set: str = "RFM", feature_names=FEATURE_NAMES) -> dict:
    """Fit only on supplied support; Mitra records context without weight training."""
    if kind not in ("prevalence", "recency", "logistic", "mitra"):
        raise ValueError("Unknown scorer")
    raw = _features(X, feature_names)
    labels = np.asarray(y)
    if labels.shape != (len(raw),) or set(labels.tolist()) != {0, 1} or len(raw) > SETTINGS["max_support"]:
        raise ValueError("Support must have both binary classes and at most 2048 rows")
    transformed = _transform(raw, feature_set)
    state = {
        "format_version": 1,
        "kind": kind,
        "feature_set": feature_set,
        "feature_names": list(FEATURE_NAMES),
        "feature_units": list(FEATURE_UNITS),
        "classes": [0, 1],
        "settings": json.loads(json.dumps(SETTINGS)),
        "support_X": raw,
        "support_y": labels.astype(np.int64),
        "support_rows": len(raw),
        "prevalence": float(labels.mean()),
        "output": "ranking_score" if kind == "recency" else "probability",
    }
    if kind == "logistic":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        if version("scikit-learn") != SETTINGS["logistic"]["scikit_learn"]:
            raise RuntimeError("Logistic fitting requires scikit-learn==1.7.2")
        mean = transformed.mean(axis=0)
        scale = transformed.std(axis=0)
        scale[scale == 0] = 1
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42, class_weight=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit((transformed - mean) / scale, labels)
        if np.max(model.n_iter_) >= 2000:
            raise ValueError("Logistic convergence not established")
        state.update(
            mean=mean.tolist(),
            scale=scale.tolist(),
            coef=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            classes=model.classes_.tolist(),
            iterations=model.n_iter_.tolist(),
        )
    if kind == "mitra" and not np.any(np.ptp(transformed, axis=0) > 0):
        raise ValueError("Mitra requires at least one varying support feature")
    return state


def predict(state: dict, X: Any, model: Any = None, feature_names=FEATURE_NAMES) -> np.ndarray:
    """Score the exact supplied cohort, preserving order and using no query labels."""
    raw = _features(X, feature_names)
    _validate_state(state)
    kind = state["kind"]
    if kind == "prevalence":
        return np.full(len(raw), state["prevalence"], dtype=float)
    if kind == "recency":
        return -raw[:, 0]
    transformed = _transform(raw, state["feature_set"])
    if kind == "logistic":
        from scipy.special import expit

        values = (transformed - np.array(state["mean"])) / np.array(state["scale"])
        p = expit((values @ np.array(state["coef"]).T + np.array(state["intercept"])).ravel())
        return positive_probability(np.column_stack((1 - p, p)), state["classes"])
    if model is None:
        raise ValueError("Mitra requires a verified frozen model handle")
    return mitra_predict(
        model, _transform(state["support_X"], state["feature_set"]), state["support_y"], transformed
    )


def _validate_state(state: dict) -> None:
    if state.get("format_version") != 1 or state.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Scorer feature schema/version changed")
    if state.get("feature_units") != list(FEATURE_UNITS) or state.get("settings") != SETTINGS:
        raise ValueError("Scorer units or inference settings changed")
    if state.get("kind") not in ("prevalence", "recency", "logistic", "mitra") or state.get(
        "feature_set"
    ) not in ("R", "RFM"):
        raise ValueError("Unknown scorer kind or feature set")
    raw = _features(state["support_X"])
    labels = np.asarray(state["support_y"])
    if labels.shape != (len(raw),) or set(labels.tolist()) != {0, 1} or len(raw) != state["support_rows"]:
        raise ValueError("Support labels/count changed")
    if len(raw) > SETTINGS["max_support"] or state["prevalence"] != float(labels.mean()):
        raise ValueError("Support/prevalence mismatch")
    if state.get("classes") != [0, 1]:
        raise ValueError("Stored class mapping changed")
    if state.get("output") != ("ranking_score" if state["kind"] == "recency" else "probability"):
        raise ValueError("Scorer output semantics changed")
    if state["kind"] == "logistic":
        d = 1 if state["feature_set"] == "R" else 5
        for key, shape in [("mean", (d,)), ("scale", (d,)), ("coef", (1, d)), ("intercept", (1,))]:
            value = np.asarray(state[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid logistic parameter: " + key)
        if np.any(np.asarray(state["scale"]) <= 0):
            raise ValueError("Invalid scaling denominator")


def export(state: dict, directory: Path | str, model_manifest: dict | None = None) -> dict:
    """Export safe arrays/JSON only; caller separately binds currency and event policy."""
    _validate_state(state)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in state.items() if k not in ("support_X", "support_y")}
    if state["kind"] == "mitra":
        if model_manifest is None or model_manifest.get("revision") != REVISION:
            raise ValueError("Mitra artifact requires its exact pinned model manifest")
        metadata["model_manifest"] = model_manifest
    _write(root / "scorer.json", metadata)
    np.save(root / "support_X.npy", state["support_X"], allow_pickle=False)
    np.save(root / "support_y.npy", state["support_y"], allow_pickle=False)
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha(root / name)}
        for name in ("scorer.json", "support_X.npy", "support_y.npy")
    }
    manifest = {"format": "dimer_customer_scorer", "version": 1, "files": files}
    _write(root / "manifest.json", manifest)
    return manifest


def reload(directory: Path | str) -> dict:
    """Verify bounded inventory and reconstruct numeric inference state without pickle."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "dimer_customer_scorer" or manifest.get("version") != 1:
        raise ValueError("Unknown scorer artifact")
    if set(manifest["files"]) != {"scorer.json", "support_X.npy", "support_y.npy"}:
        raise ValueError("Invalid scorer inventory")
    for name, item in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe artifact")
        if (
            not 0 < item["bytes"] < 2_000_000
            or path.stat().st_size != item["bytes"]
            or sha(path) != item["sha256"]
        ):
            raise ValueError("Scorer artifact integrity failure: " + name)
    state = json.loads((root / "scorer.json").read_text())
    state["support_X"] = np.load(root / "support_X.npy", allow_pickle=False)
    state["support_y"] = np.load(root / "support_y.npy", allow_pickle=False)
    _validate_state(state)
    if state["kind"] == "mitra" and state.get("model_manifest", {}).get("revision") != REVISION:
        raise ValueError("Scorer model revision changed")
    return state


def stage_snapshot(manifest: dict, cache: Path | str) -> Path:
    """Verify each immutable snapshot file, including cache hits; never unpickle weights."""
    if manifest.get("modelId") != "autogluon/mitra-classifier" or manifest.get("revision") != REVISION:
        raise ValueError("Wrong model identity")
    root = Path(cache).resolve() / REVISION
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("Unsafe snapshot directory")
    if {item["path"] for item in manifest["files"]} != {"config.json", "model.safetensors"} or len(
        manifest["files"]
    ) != 2:
        raise ValueError("Unexpected model inventory")
    for item in manifest["files"]:
        if (item["bytes"], item["sha256"]) != PINNED_FILES[item["path"]]:
            raise ValueError("Snapshot differs from pinned model file identities")
        relative = PurePosixPath(item["path"])
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe snapshot path")
        if not re.fullmatch("[0-9a-f]{64}", item["sha256"]) or not 0 < item["bytes"] < 1_000_000_000:
            raise ValueError("Invalid snapshot integrity metadata")
        if not path.exists():
            temporary = path.with_suffix(".partial")
            count = 0
            try:
                url = f"https://huggingface.co/{manifest['modelId']}/resolve/{REVISION}/{item['path']}"
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                    while chunk := response.read(1 << 20):
                        count += len(chunk)
                        if count > item["bytes"]:
                            raise ValueError("Model download exceeds byte ceiling")
                        stream.write(chunk)
                if count != item["bytes"] or sha(temporary) != item["sha256"]:
                    raise ValueError("Model download mismatch")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Cached model mismatch")
    return root


def _seed() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_mitra(cache: Path | str, manifest: dict | Path | str) -> Any:
    if version("autogluon.tabular") != "1.5.0":
        raise RuntimeError("Requires autogluon.tabular==1.5.0")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run pretrained models on a hosted Colab GPU only")
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
    from safetensors.torch import load_file

    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    path = stage_snapshot(manifest, cache)
    _seed()
    config = json.loads((path / "config.json").read_text())
    model = Tab2D(**config, use_pretrained_weights=False, path_to_weights="", device="cuda")
    model.load_state_dict(load_file(path / "model.safetensors", device="cpu"), strict=True)
    model.to("cuda").eval()
    return model


def mitra_predict(model: Any, support: np.ndarray, labels: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Full-support deterministic in-context inference; never calls fit/train wrappers."""
    if version("autogluon.tabular") != "1.5.0":
        raise RuntimeError("Requires autogluon.tabular==1.5.0")
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier, TrainerFinetune

    _seed()
    classes = np.unique(labels)
    if classes.tolist() != [0, 1]:
        raise ValueError("Binary class mapping required")
    if not 2 <= len(support) <= SETTINGS["max_support"]:
        raise ValueError("Support capacity exceeded")
    device = str(next(model.parameters()).device)
    adapter = MitraClassifier(
        device=device,
        fine_tune=False,
        fine_tune_steps=0,
        n_estimators=1,
        shuffle_classes=False,
        shuffle_features=False,
        random_mirror_x=False,
        random_mirror_regression=False,
        use_random_transforms=False,
        seed=42,
        verbose=False,
    )
    config, _ = adapter._create_config("classification", 10)
    config.hyperparams.update(
        precision="float32",
        max_samples_support=len(support),
        max_samples_query=64,
        max_epochs=0,
        n_ensembles=1,
        use_feature_count_scaling=False,
        use_quantile_transformer=False,
        shuffle_classes=False,
        shuffle_features=False,
        random_mirror_x=False,
        random_mirror_regression=False,
        use_random_transforms=False,
    )
    trainer = TrainerFinetune(
        config, model, n_classes=2, device=device, rng=np.random.RandomState(42), verbose=False
    )
    trainer.preprocessor.fit(support.copy(), labels.copy())
    trainer.post_fit_optimize()
    logits = np.asarray(trainer.predict(support.copy(), labels.copy(), query.copy()), dtype=float)
    if logits.shape != (len(query), 2) or not np.isfinite(logits).all():
        raise ValueError("Invalid Mitra logits")
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return positive_probability(probabilities, classes)


def unload(model: Any) -> None:
    """Move the network off GPU; caller drops its final Python reference."""
    model.to("cpu")
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
