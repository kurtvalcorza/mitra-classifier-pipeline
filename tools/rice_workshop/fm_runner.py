"""Isolated-environment runner for the rice pixel-tables notebook's foundation-model conditions.

The notebook writes this file to its run folder (hash-checked) and starts it with the Python of one hash-locked model
environment (TabPFN-3, TabICLv2, TabDPT v1.2 or Mitra), one process per stage, so GPU memory is released between models.

Modes (all read `--config`, a JSON file the notebook writes):

- `develop`: load the pinned checkpoint, register the fit rows as context (in-context learning: no gradient step),
  predict the development rows, save their rice probabilities and the fitted artifact;
- `score`:   in a fresh process, reload the artifact from its files (inventory checked first), re-predict the first
  development chunk and compare it with the saved probabilities (reload parity), then predict the evaluation rows;
- `probe`:   after scoring, reload once more and re-predict the first evaluation chunk (validation reload parity);
- `sweep`:   the activity: refit on nested subsets of the fit rows and predict the development rows for each size.

Adapted from the workshop smoke runner (verified on a Colab T4, 2026-10-11) and the v2 workshop runner's per-family
save/reload (tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb, cell 28). The adapters are
the notebook-carried v2 adapters; nothing is downloaded except the pinned checkpoints.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

TARGET = "target"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root):
    root = Path(root)
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact inventory refuses symbolic links: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    return inventory


def positive_column(raw, classes):
    """Rice (class 1) probability column, whatever order the model reports its classes in."""
    labels = [str(c) for c in classes]
    p = np.asarray(raw, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(labels) or "1" not in labels:
        raise ValueError(f"unexpected probability shape {p.shape} for classes {labels}")
    if not np.isfinite(p).all():
        raise ValueError("non-finite probabilities")
    return np.clip(p[:, labels.index("1")], 0.0, 1.0)


def predict_chunks(predict, x, names, chunk):
    """Predict every row in fixed chunks (no row is skipped); returns probabilities and seconds."""
    import pandas as pd

    out, started = [], time.perf_counter()
    for start in range(0, len(x), chunk):
        out.append(np.asarray(predict(pd.DataFrame(x[start:start + chunk], columns=names)), dtype=np.float64))
        done = start + len(out[-1])
        print(f"    predicted {done}/{len(x)} rows in {time.perf_counter() - started:.1f} s", flush=True)
    p = np.concatenate(out) if out else np.zeros(0)
    if len(p) != len(x):
        raise RuntimeError(f"predicted {len(p)} of {len(x)} rows")
    return p, time.perf_counter() - started


def load_rows(cfg, fs, key, index_key=None):
    data = Path(cfg["data_dir"])
    x = np.load(data / f"{fs}_{key}_X.npy")
    y = np.load(data / f"{fs}_{key}_y.npy")
    if key in ("dev", "eval"):  # raw tables: apply the fit-only fill of the foundation-model fit sample
        fill = np.load(data / f"{fs}_fill_fm.npy")
        x = np.where(np.isfinite(x), x, fill).astype(np.float32)
    index = cfg.get("indices", {}).get(fs, {}).get(index_key or key)
    if index is not None:  # a declared, seeded sample (Mitra); the notebook states its size beside the results
        keep = np.load(index)
        x, y = x[keep], y[keep]
    return x, y


def load_family(family, cfg, device):
    wd = Path(cfg["weights_dir"])
    seed = int(cfg["seed"])
    s = cfg["settings"]
    if family == "tabpfn":
        from tabpfn_classifier_pipeline.pipeline import TabPFNClassifierPipeline
        return TabPFNClassifierPipeline.from_pretrained(device=device, weights_dir=wd, allow_download=True,
                                                        n_estimators=int(s["n_estimators"]), random_state=seed)
    if family == "tabicl":
        from tabicl_classifier_pipeline.api import TabICLClassificationPipeline
        return TabICLClassificationPipeline.from_pretrained(weights_dir=wd, allow_download=True,
                                                            n_estimators=int(s["n_estimators"]), random_state=seed,
                                                            device=device)
    if family == "mitra":
        from mitra_pipeline.tutorial_api import MitraClassificationPipeline
        return MitraClassificationPipeline.from_pretrained(weights_dir=wd, allow_download=True, device=device)
    if family == "tabdpt":
        from tabdpt_classifier_pipeline.pipeline import TabDPTClassificationPipeline
        return TabDPTClassificationPipeline.from_pretrained(weights_dir=wd, allow_download=True, device=device,
                                                            use_flash=False, compile_model=False, seed=seed)
    raise ValueError(f"unknown model family {family!r}")


def fit_family(family, pipe, fit_x, fit_y, tune_x, tune_y, names, cfg, artifact):
    """Register the fit rows as context; return a rice-probability function and a function that saves the artifact."""
    import pandas as pd

    s = cfg["settings"]
    seed = int(cfg["seed"])
    train = pd.DataFrame(fit_x, columns=names)
    train[TARGET] = fit_y
    if family == "tabpfn":
        pipe.fit(train[names], train[TARGET], target_column=TARGET)
        return (lambda df: positive_column(pipe.predict_proba(df), pipe.classes)), (lambda: pipe.save_artifact(artifact / "tabpfn"))
    if family == "tabicl":
        from tabicl_classifier_pipeline.api import (
            apply_categorical_encoder,
            fit_categorical_encoder,
            prepare_classification_table,
        )
        clean, _ = prepare_classification_table(train, TARGET)
        encoders = fit_categorical_encoder(clean, names)
        x, _ = apply_categorical_encoder(clean[names], encoders)
        pipe.fit(x, clean[TARGET])

        def save():
            (artifact / "tabicl").mkdir(parents=True, exist_ok=True)
            with open(artifact / "tabicl" / "tabicl.pkl", "wb") as handle:
                pickle.dump({"pipe": pipe, "encoders": encoders}, handle)
        return (lambda df: positive_column(pipe.predict_proba(apply_categorical_encoder(df, encoders)[0]), pipe.classes_)), save
    if family == "mitra":
        from autogluon.tabular import TabularPredictor
        from mitra_pipeline.tutorial_api import seed_everything
        seed_everything(seed)
        tune = pd.DataFrame(tune_x, columns=names)
        tune[TARGET] = tune_y
        # AutoGluon needs tuning rows; in-context Mitra runs zero training epochs, so they are only scored. AutoGluon
        # also tunes its own predict() threshold on them: the notebook never uses it (predict_proba + its own threshold).
        predictor = TabularPredictor(label=TARGET, problem_type="binary", eval_metric="balanced_accuracy",
                                     path=str(artifact / "mitra_predictor"), verbosity=2)
        predictor.fit(train_data=train, tuning_data=tune, hyperparameters={"MITRA": {"fine_tune": False, "seed": seed}},
                      num_bag_folds=0, num_stack_levels=0, fit_weighted_ensemble=False, dynamic_stacking=False,
                      time_limit=int(s["time_limit_seconds"]),
                      ag_args_fit={"max_memory_usage_ratio": float(s["max_memory_usage_ratio"])})
        if not any("mitra" in m.lower() for m in predictor.model_names()):
            raise RuntimeError(f"Mitra did not fit (AutoGluon returned {predictor.model_names()}); see the log above "
                               "for AutoGluon's memory pre-check message")
        return (lambda df: positive_column(predictor.predict_proba(df).to_numpy(dtype=float), predictor.class_labels)), (lambda: None)
    if family == "tabdpt":
        from tabdpt_classifier_pipeline.artifact import export_artifact_bundle
        pipe.fit(train, target_column=TARGET, seed=seed)
        kw = {"n_ensembles": int(s["n_ensembles"]), "context_size": int(s["context_size"]),
              "batch_size": int(s["batch_size"]), "seed": seed}
        return ((lambda df: positive_column(pipe.predict_proba(df, **kw).to_numpy(dtype=float), pipe.class_labels_)),
                (lambda: export_artifact_bundle(pipe, train, artifact / "tabdpt")))
    raise ValueError(f"unknown model family {family!r}")


def reload_family(family, cfg, device, artifact):
    """Rebuild the predictor from the saved files only (VER2), as a downstream consumer would."""
    s = cfg["settings"]
    seed = int(cfg["seed"])
    if family == "tabpfn":
        from tabpfn_classifier_pipeline.pipeline import TabPFNClassifierPipeline
        pipe = TabPFNClassifierPipeline.from_artifact(artifact / "tabpfn", device=device)
        return lambda df: positive_column(pipe.predict_proba(df), pipe.classes)
    if family == "tabicl":
        from tabicl_classifier_pipeline.api import apply_categorical_encoder
        # Pickle runs code on load: this file was written by this notebook's own develop stage in this runtime, and its
        # SHA-256 matched the inventory recorded then (checked above). Never load a pickle you did not create.
        with open(artifact / "tabicl" / "tabicl.pkl", "rb") as handle:
            restored = pickle.load(handle)
        pipe, encoders = restored["pipe"], restored["encoders"]
        return lambda df: positive_column(pipe.predict_proba(apply_categorical_encoder(df, encoders)[0]), pipe.classes_)
    if family == "mitra":
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(str(artifact / "mitra_predictor"))
        return lambda df: positive_column(predictor.predict_proba(df).to_numpy(dtype=float), predictor.class_labels)
    if family == "tabdpt":
        from tabdpt_classifier_pipeline.artifact import load_verified_artifact
        pipe = load_verified_artifact(artifact / "tabdpt" / "artifact.json",
                                      model_weight_path=Path(cfg["weights_dir"]) / "tabdpt1_2.safetensors",
                                      device=device, use_flash=False, compile_model=False, seed=seed)
        kw = {"n_ensembles": int(s["n_ensembles"]), "context_size": int(s["context_size"]),
              "batch_size": int(s["batch_size"]), "seed": seed}
        return lambda df: positive_column(pipe.predict_proba(df, **kw).to_numpy(dtype=float), pipe.class_labels_)
    raise ValueError(f"unknown model family {family!r}")


def peak_gpu_gib(torch):
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return None


def develop(cfg, fs, out, device, torch):
    names = cfg["feature_names"][fs]
    fit_x, fit_y = load_rows(cfg, fs, "fit")
    dev_x, dev_y = load_rows(cfg, fs, "dev")
    tune_x, tune_y = load_rows(cfg, fs, "dev", "tune") if cfg["family"] == "mitra" else (None, None)
    artifact = out / "artifact"
    if artifact.exists():
        raise RuntimeError(f"{artifact} already exists; the notebook clears it before a development run")
    record = {"feature_set": fs, "fit_rows": int(len(fit_y)), "fit_rice": int(fit_y.sum()), "dev_rows": int(len(dev_y))}
    started = time.perf_counter()
    pipe = load_family(cfg["family"], cfg, device)
    record["load_s"] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    predict, save = fit_family(cfg["family"], pipe, fit_x, fit_y, tune_x, tune_y, names, cfg, artifact)
    record["fit_s"] = round(time.perf_counter() - started, 2)
    print(f"[{cfg['family']}/{fs}] context registered: {len(fit_y)} rows in {record['fit_s']} s", flush=True)
    prob, seconds = predict_chunks(predict, dev_x, names, int(cfg["chunk_rows"]))
    record["predict_dev_s"] = round(seconds, 2)
    np.save(out / "dev_proba.npy", prob.astype(np.float32))
    save()
    record["artifact_inventory"] = file_inventory(artifact)
    record["peak_gpu_gib"] = peak_gpu_gib(torch)
    return record


def score(cfg, fs, out, device, torch):
    names = cfg["feature_names"][fs]
    artifact = out / "artifact"
    expected = cfg["artifact_inventory"][fs]
    observed = file_inventory(artifact)
    if observed != expected:  # checked before anything is deserialised
        raise RuntimeError(f"the saved artifact for {fs} differs from the inventory recorded at development; "
                           f"missing={sorted(set(expected) - set(observed))} extra={sorted(set(observed) - set(expected))}")
    record = {"feature_set": fs}
    started = time.perf_counter()
    predict = reload_family(cfg["family"], cfg, device, artifact)
    record["reload_s"] = round(time.perf_counter() - started, 2)
    chunk = int(cfg["chunk_rows"])
    dev_x, _dev_y = load_rows(cfg, fs, "dev")
    saved = np.load(out / "dev_proba.npy").astype(np.float64)[:chunk]
    again, _ = predict_chunks(predict, dev_x[:chunk], names, chunk)  # the same first chunk, so batch content matches
    difference = float(np.abs(again - saved).max()) if len(saved) else 0.0
    record["reload_parity"] = {
        "rows": int(len(saved)), "max_abs_difference": difference, "tolerance": float(cfg["reload_tolerance"]),
        "within_tolerance": difference <= float(cfg["reload_tolerance"]),
        "decisions_identical_at_frozen_threshold": bool(np.array_equal(again >= cfg["thresholds"][fs], saved >= cfg["thresholds"][fs])),
    }
    eval_x, eval_y = load_rows(cfg, fs, "eval")
    prob, seconds = predict_chunks(predict, eval_x, names, chunk)
    record["eval_rows"] = int(len(eval_y))
    record["predict_eval_s"] = round(seconds, 2)
    np.save(out / "eval_proba.npy", prob.astype(np.float32))
    record["peak_gpu_gib"] = peak_gpu_gib(torch)
    return record


def probe(cfg, fs, out, device, torch):
    """After scoring: reload the artifact once more in a fresh process and re-predict the first evaluation chunk."""
    names = cfg["feature_names"][fs]
    artifact = out / "artifact"
    if file_inventory(artifact) != cfg["artifact_inventory"][fs]:
        raise RuntimeError(f"the saved artifact for {fs} changed after scoring")
    predict = reload_family(cfg["family"], cfg, device, artifact)
    chunk = int(cfg["chunk_rows"])
    eval_x, _eval_y = load_rows(cfg, fs, "eval")
    saved = np.load(out / "eval_proba.npy").astype(np.float64)[:chunk]
    again, seconds = predict_chunks(predict, eval_x[:chunk], names, chunk)
    difference = float(np.abs(again - saved).max()) if len(saved) else 0.0
    return {"feature_set": fs, "rows": int(len(saved)), "max_abs_difference": difference,
            "tolerance": float(cfg["reload_tolerance"]), "within_tolerance": difference <= float(cfg["reload_tolerance"]),
            "decisions_identical_at_frozen_threshold": bool(np.array_equal(again >= cfg["thresholds"][fs], saved >= cfg["thresholds"][fs])),
            "seconds": round(seconds, 2)}


def sweep(cfg, fs, out, device, torch):
    names = cfg["feature_names"][fs]
    fit_x, fit_y = load_rows(cfg, fs, "fit")
    dev_x, _dev_y = load_rows(cfg, fs, "dev")
    rows = []
    for size, index in sorted(cfg["sweep_indices"].items(), key=lambda kv: int(kv[0])):
        keep = np.load(index)
        started = time.perf_counter()
        pipe = load_family(cfg["family"], cfg, device)
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        predict, _save = fit_family(cfg["family"], pipe, fit_x[keep], fit_y[keep], None, None, names, cfg, out / "unused")
        fit_s = time.perf_counter() - started
        prob, seconds = predict_chunks(predict, dev_x, names, int(cfg["chunk_rows"]))
        np.save(out / f"dev_proba_{int(size)}.npy", prob.astype(np.float32))
        rows.append({"context_rows": int(size), "context_rice": int(fit_y[keep].sum()), "load_s": round(load_s, 2),
                     "fit_s": round(fit_s, 2), "predict_dev_s": round(seconds, 2), "peak_gpu_gib": peak_gpu_gib(torch)})
        del pipe, predict
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return {"feature_set": fs, "sizes": rows}


MODES = {"develop": develop, "score": score, "probe": probe, "sweep": sweep}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"family": cfg["family"], "mode": cfg["mode"], "device": device, "torch": torch.__version__,
              "python": sys.version.split()[0], "settings": cfg["settings"], "feature_sets": {}}
    if device == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
    for fs in cfg["feature_sets"]:
        out = Path(cfg["output_dir"]) / fs
        out.mkdir(parents=True, exist_ok=True)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        result["feature_sets"][fs] = MODES[cfg["mode"]](cfg, fs, out, device, torch)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    Path(cfg["result_path"]).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("RESULT " + json.dumps(result, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
