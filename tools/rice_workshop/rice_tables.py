"""Kernel-side core of the DIMER Philippines rice pixel-tables notebook (spec: docs/RICE_TABULAR_WORKSHOP_SPEC.md).

The notebook writes this file next to its run directory (hash-checked) and imports it. Everything here is numpy-only
at import time; h5py, lightgbm and kagglehub are imported inside the functions that need them, so the repository's CPU
tests exercise the data, scoring and protocol contract on synthetic arrays.

Carried from the rice capstone (kurtvalcorza/prithvi-eo-feature-extraction-pipeline origin/main f4680a5,
tools/rice_capstone/capstone.py), unchanged in behaviour: split_fit_dev, check_roles, confusion, scores, pooled,
select_threshold, m1_features, fit_fill, fill_features, pixel_sample, validate_chip (S2 part), collect_byod,
build_zip, verify_zip, write_csv and the receipt / STAGE_OUTPUTS invalidation design. ae_features is the study L3/L3B
bundle's preprocessing.ae_features. New here: the AlphaEarth checks for BYOD chips, the seeded development subset,
nested context subsets for the activity, the score-once ledger, and the table staging shared by sample and BYOD.
"""

# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------------
# identities and frozen protocol constants
# --------------------------------------------------------------------------------------------------

DATASET_SLUG = "kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching"
DATASET_VERSION = 1  # pinned: later versions change only the card and rights wording
INDEX_SHA256 = "e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d"
DATASET_LICENSE = "CC BY 4.0"
DEVELOPMENT_EXAMPLE_GROUP = "ph_96_766"  # a parent training sample: never read, never a scoring chip

N_BINS = 24
N_BANDS = 6
CHIP = 128
QUARTERS = 4
BINS_PER_QUARTER = N_BINS // QUARTERS
BAND_NAMES = ("B2", "B3", "B4", "B8A", "B11", "B12")
AE_DIM = 64
IGNORE = 255
REFLECTANCE_MIN = -0.5
REFLECTANCE_MAX = 2.0  # larger values are reflectance x 10 000 and are refused, never rescaled

SPLIT_SALT = "rice-capstone-split"  # the capstone's rule, so both notebooks score identical chips
DEV_FRACTION = 0.25
MIN_DEV_RICE_CHIPS = 2
BYOD_SPLIT_RULE = (
    "whole sites (HDF5 attribute group / group_id, else the file name without its _YYYY year), ordered by "
    "SHA-256('rice-tables-byod:<k>:<site>'): about 20% development, 20% test, the rest fit; the first k giving rice in "
    "development and test and both classes in fit is used"
)
SPLIT_RULE = (
    "whole regions, ordered by SHA-256('rice-capstone-split:<k>:<region>'), go to development while it holds at most "
    "round(0.25 x training chips); the first k with >= target - 2 development chips, >= 2 rice development chips and "
    "both strata in fit is used"
)
THRESHOLDS = tuple(i / 100 for i in range(10, 91, 5))

PIXEL_SEED = 20261003  # capstone pixel_sample seed
LGBM_FIT_ROWS = 65_536  # capstone PIXEL_CAP: LightGBM's canonical fit table
FM_FIT_ROWS = 10_000  # every foundation model gets the same 10,000 class-stratified fit rows
DEV_SUBSET_ROWS = 30_000  # thresholds are chosen on a seeded 30,000-pixel development subset
DEV_SUBSET_SEED = 20261012
ACTIVITY_SEED = 20261013
RELOAD_TOLERANCE = 1e-4

LGBM_PARAMS = {
    "objective": "binary", "metric": "None", "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 1.0, "lambda_l2": 1.0, "deterministic": True,
    "force_col_wise": True, "device_type": "cpu", "num_threads": 2, "seed": 0, "feature_fraction_seed": 0,
    "data_random_seed": 0, "verbosity": -1,
}
LGBM_ROUND_CAP = 200
LGBM_PATIENCE = 50

M1_STATS = ("mean", "std", "min", "max", "p10", "p50", "p90")
M1_NAMES = [f"{b}_{s}" for b in BAND_NAMES for s in M1_STATS]
M1_NAMES += [f"q{q + 1}_{b}_mean" for q in range(QUARTERS) for b in BAND_NAMES]
M1_NAMES += [f"q{q + 1}_observed_fraction" for q in range(QUARTERS)]
AE_NAMES = [f"A{i:02d}" for i in range(AE_DIM)] + ["AE_valid"]
FEATURE_SETS = {"alphaearth": AE_NAMES, "s2m1": M1_NAMES}
FEATURE_SET_LABELS = {"alphaearth": "AlphaEarth (64 + AE_valid)", "s2m1": "Sentinel-2 M1 (70)"}

BYOD_MAX_FILES = 200
BYOD_MAX_BYTES = 4 * 2**30
BYOD_MIN_ADAPT_CHIPS = 6
BYOD_MIN_SITES = 3

ATTRIBUTION = (
    "Rice labels: Chen et al., 20 m paddy rice maps for Monsoon Asia 2018-2021 (Zenodo, https://doi.org/10.5281/zenodo.15124936), CC BY 4.0.\n"
    "Contains modified Copernicus Sentinel data 2019-2021.\n"
    "AlphaEarth Foundations Satellite Embedding (Google / Google DeepMind; Brown et al., 2025), CC BY 4.0.\n"
    "Administrative regions: geoBoundaries PHL ADM1, CC BY 3.0 IGO.\n"
    "Dataset compilation: Kurt Valcorza, rice-philippines-s2-alphaearth-v1-teaching (Kaggle, version 1), CC BY 4.0.\n"
    "TabPFN-3 weights (Prior-Labs/tabpfn_3): TabPFN-3 licence v1.0, non-commercial use only.\n"
    "TabICLv2 (jingang/TabICL): BSD-3-Clause. TabDPT v1.2 (Layer6/TabDPT): Apache-2.0. Mitra (autogluon/mitra-classifier): Apache-2.0.\n"
)


# --------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing: run the earlier cells in order (Runtime > Run all).")
    return json.loads(path.read_text(encoding="utf-8"))


def chip_id(row: Mapping[str, Any]) -> str:
    return f"{row['group_id']}_{row['year']}"


def file_inventory(root: Path) -> dict[str, str]:
    """SHA-256 of every file under `root` (symbolic links refused), keyed by POSIX relative path."""
    if not root.exists():
        raise FileNotFoundError(root)
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact inventory refuses symbolic links: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    return inventory


# --------------------------------------------------------------------------------------------------
# chip roles (capstone rule, verbatim)
# --------------------------------------------------------------------------------------------------


def split_fit_dev(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assign the training chips to fit and development by whole region. Regions are ordered by
    SHA-256("rice-capstone-split:<k>:<region>") and added to development while it holds at most round(0.25 n) chips;
    the first attempt k = 0, 1, ... whose development set has at least target - 2 chips, at least two rice chips,
    and leaves both classes in fit is used. Deterministic across library versions."""
    train = sorted((r for r in rows if r["split"] == "train"), key=chip_id)
    if not train:
        raise ValueError("no training chips to split")
    target = round(DEV_FRACTION * len(train))
    regions = sorted({r["region_iso"] for r in train})
    for attempt in range(100):
        order = sorted(regions, key=lambda g: sha256_bytes(f"{SPLIT_SALT}:{attempt}:{g}".encode()))
        dev_regions: list[str] = []
        count = 0
        for region in order:
            size = sum(r["region_iso"] == region for r in train)
            if count + size <= target:
                dev_regions.append(region)
                count += size
            if count == target:
                break
        dev = [r for r in train if r["region_iso"] in dev_regions]
        fit = [r for r in train if r["region_iso"] not in dev_regions]
        rice_dev = sum(r["stratum"] == "rice" for r in dev)
        fit_strata = {r["stratum"] for r in fit}
        if len(dev) >= target - 2 and rice_dev >= MIN_DEV_RICE_CHIPS and fit_strata == {"rice", "nonrice"}:
            return {
                "rule": SPLIT_RULE,
                "attempt": attempt, "target_dev_chips": target,
                "development_regions": sorted(dev_regions), "fit_regions": sorted({r["region_iso"] for r in fit}),
                "fit": [chip_id(r) for r in fit], "development": [chip_id(r) for r in dev],
                "development_rice_chips": rice_dev,
            }
    raise ValueError("no region assignment satisfies the declared development constraints")


def check_roles(roles: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for role, ids in roles.items():
        for cid in ids:
            if cid in seen:
                raise ValueError(f"{cid} is in both {seen[cid]} and {role}")
            if cid.startswith(DEVELOPMENT_EXAMPLE_GROUP + "_"):
                raise ValueError("the development example is a training sample and must never be a scoring chip")
            seen[cid] = role


def sample_roles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """fit / development from split_fit_dev, validation = the dataset's validation chips; roles are disjoint."""
    split = split_fit_dev(rows)
    roles = {"fit": split["fit"], "development": split["development"],
             "validation": sorted(chip_id(r) for r in rows if r["split"] == "validation")}
    check_roles(roles)
    return {"roles": roles, "split": split}


# --------------------------------------------------------------------------------------------------
# metrics (pooled over label != 255; undefined ratios are null with a reason) — capstone, verbatim
# --------------------------------------------------------------------------------------------------


def confusion(probability: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, int]:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label)
    if probability.shape != label.shape:
        raise ValueError(f"prediction shape {probability.shape} != label shape {label.shape}")
    if not np.isin(label, (0, 1, IGNORE)).all():
        raise ValueError("label values outside 0/1/255")
    scored = label != IGNORE
    if not np.isfinite(probability[scored]).all():
        raise ValueError("non-finite score on a scored pixel; pixels are never hidden")
    if ((probability[scored] < 0) | (probability[scored] > 1)).any():
        raise ValueError("scores must lie in [0, 1]")
    positive = probability >= threshold
    return {
        "tp": int((scored & positive & (label == 1)).sum()), "fp": int((scored & positive & (label == 0)).sum()),
        "fn": int((scored & ~positive & (label == 1)).sum()), "tn": int((scored & ~positive & (label == 0)).sum()),
    }


def scores(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    values = {
        "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }
    reasons = {
        "f1": "no reference rice and no predicted rice", "iou": "no reference rice and no predicted rice",
        "precision": "no pixel predicted rice", "recall": "no reference rice pixel", "accuracy": "no scored pixel",
    }
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "scored_pixels": tp + fp + fn + tn, "reference_rice_pixels": tp + fn,
        **values, "undefined": "; ".join(f"{k}: {reasons[k]}" for k, v in values.items() if v is None),
    }


def pooled(probabilities: Sequence[np.ndarray], labels: Sequence[np.ndarray], threshold: float) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for probability, label in zip(probabilities, labels, strict=True):
        for key, value in confusion(probability, label, threshold).items():
            total[key] += value
    return scores(total)


def select_threshold(probabilities: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> dict[str, Any]:
    """Highest pooled development rice F1 over THRESHOLDS; ties closest to 0.5, then the lower threshold."""
    sweep = [{"threshold": t, **pooled(probabilities, labels, t)} for t in THRESHOLDS]
    defined = [row for row in sweep if row["f1"] is not None]
    if not defined:
        raise ValueError("development F1 is undefined at every threshold (no reference rice on development chips)")
    best = min(defined, key=lambda row: (-row["f1"], abs(row["threshold"] - 0.5), row["threshold"]))
    return {"threshold": best["threshold"], "f1": best["f1"], "sweep": sweep}


# --------------------------------------------------------------------------------------------------
# feature recipes
# --------------------------------------------------------------------------------------------------


def m1_features(chip: Mapping[str, Any]) -> np.ndarray:
    """(H*W, 70) float32: per band mean/std/min/max/p10/p50/p90 of observed values, quarterly band means, quarterly
    observed fractions. Missing statistics stay NaN here and are filled by the fit-only fill (study recipe)."""
    import warnings

    valid = np.asarray(chip["S2_valid"]).astype(bool)
    values = np.where(valid[:, None], chip["S2"], np.nan)
    blocks: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for band in range(N_BANDS):
            x = values[:, band]
            blocks += [np.nanmean(x, axis=0), np.nanstd(x, axis=0), np.nanmin(x, axis=0), np.nanmax(x, axis=0)]
            blocks += list(np.nanpercentile(x, [10, 50, 90], axis=0))
        for quarter in range(QUARTERS):
            blocks.extend(np.nanmean(values[quarter * BINS_PER_QUARTER:(quarter + 1) * BINS_PER_QUARTER], axis=0))
    blocks += [valid[q * BINS_PER_QUARTER:(q + 1) * BINS_PER_QUARTER].mean(axis=0) for q in range(QUARTERS)]
    return np.stack(blocks, axis=-1).reshape(-1, len(M1_NAMES)).astype(np.float32)


def ae_features(chip: Mapping[str, Any]) -> np.ndarray:
    """(H*W, 65) float32: the 64 AlphaEarth values (NaN where AE_valid = 0) and AE_valid itself (study L3B recipe)."""
    valid = np.asarray(chip["AE_valid"]).astype(bool)
    vectors = np.where(valid[None], np.asarray(chip["AlphaEarth"], dtype=np.float32), np.nan)
    return np.concatenate([vectors.transpose(1, 2, 0), valid[..., None]], axis=-1).reshape(-1, len(AE_NAMES)).astype(np.float32)


def fit_fill(features: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features)
    count = finite.sum(axis=0)
    if not count.all():
        raise ValueError("a feature has no finite fit observation")
    return np.where(finite, features, 0).sum(axis=0, dtype=np.float64) / count


def fill_features(features: np.ndarray, fill: np.ndarray) -> np.ndarray:
    if features.shape[-1] != len(fill) or not np.isfinite(fill).all():
        raise ValueError("feature/fill mismatch")
    return np.where(np.isfinite(features), features, fill).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# pixel samples
# --------------------------------------------------------------------------------------------------


def pixel_sample(labels: Sequence[np.ndarray], cap: int, seed: int) -> list[np.ndarray]:
    """Round-robin quotas over non-empty (chip, class) strata, then seeded selection without replacement (study)."""
    if cap < 1:
        raise ValueError("cap must be positive")
    strata = [(i, c, np.flatnonzero(np.asarray(y).reshape(-1) == c)) for i, y in enumerate(labels) for c in (0, 1)]
    strata = [s for s in strata if len(s[2])]
    quotas = np.zeros(len(strata), dtype=int)
    budget = min(cap, sum(len(s[2]) for s in strata))
    while budget:
        active = [j for j, s in enumerate(strata) if quotas[j] < len(s[2])]
        step = max(1, budget // len(active))
        for j in active:
            take = min(step, len(strata[j][2]) - quotas[j], budget)
            quotas[j] += take
            budget -= take
            if not budget:
                break
    selected: list[list[int]] = [[] for _ in labels]
    rng = np.random.default_rng(seed)
    for (i, _c, pixels), n in zip(strata, quotas, strict=True):
        selected[i].extend(rng.choice(pixels, n, replace=False).tolist())
    return [np.asarray(sorted(p), dtype=np.int64) for p in selected]


def proportional_sample(labels: Sequence[np.ndarray], cap: int, seed: int) -> list[np.ndarray]:
    """Seeded sample of labelled pixels stratified by (chip, class) in proportion to each stratum's size (largest
    remainder), so the subset keeps the source's rice share and each chip's share. Used for the development subset
    and for Mitra's declared validation sample: a threshold chosen for F1 must see the true class mix."""
    if cap < 1:
        raise ValueError("cap must be positive")
    strata = [(i, c, np.flatnonzero(np.asarray(y).reshape(-1) == c)) for i, y in enumerate(labels) for c in (0, 1)]
    strata = [s for s in strata if len(s[2])]
    total = sum(len(s[2]) for s in strata)
    if total <= cap:
        return [np.flatnonzero(np.isin(np.asarray(y).reshape(-1), (0, 1))).astype(np.int64) for y in labels]
    exact = np.array([cap * len(s[2]) / total for s in strata])
    quotas = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - quotas), kind="stable")[: cap - int(quotas.sum())]:
        quotas[j] += 1
    for c in (0, 1):  # a class present in the source keeps at least one row, so F1 stays defined on sparse rice
        own = [j for j, s in enumerate(strata) if s[1] == c]
        if own and not quotas[own].sum():
            donor = max((j for j, s in enumerate(strata) if s[1] != c), key=lambda j: quotas[j])
            quotas[donor] -= 1
            quotas[max(own, key=lambda j: len(strata[j][2]))] += 1
    selected: list[list[int]] = [[] for _ in labels]
    rng = np.random.default_rng(seed)
    for (i, _c, pixels), n in zip(strata, quotas, strict=True):
        selected[i].extend(rng.choice(pixels, int(n), replace=False).tolist())
    return [np.asarray(sorted(p), dtype=np.int64) for p in selected]


def nested_context(y: np.ndarray, sizes: Sequence[int], seed: int) -> dict[int, np.ndarray]:
    """Row indices of `y` for each requested context size, class-proportional and nested (the 1,000 rows are inside
    the 3,000, which are inside the 10,000), so the activity changes only how many rows the model sees."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    order = {c: rng.permutation(np.flatnonzero(y == c)) for c in (0, 1)}
    share = float((y == 1).mean())
    out = {}
    for size in sorted(set(int(s) for s in sizes)):
        if size < 2 or size > len(y):
            raise ValueError(f"context size {size} must be between 2 and the {len(y)} fit rows")
        n_rice = min(len(order[1]), max(1, round(size * share)))
        n_non = size - n_rice
        if n_non > len(order[0]) or n_non < 1:
            raise ValueError(f"context size {size} cannot keep both classes")
        out[size] = np.sort(np.concatenate([order[1][:n_rice], order[0][:n_non]]))
    return out


# --------------------------------------------------------------------------------------------------
# chip schema (the BYOD contract is the dataset's own schema)
# --------------------------------------------------------------------------------------------------


class ChipError(ValueError):
    """A chip does not satisfy the BYOD schema; the message names the failed rule."""


def validate_chip(arrays: Mapping[str, Any], *, name: str = "chip", require_label: bool = True) -> dict[str, Any]:
    """Check one chip and return a normalised copy: S2 float32 (NaN where S2_valid is 0), S2_valid uint8, label uint8
    (or None), and AlphaEarth float32 / AE_valid uint8 when both are present (else None). Raises ChipError."""
    for key in ("S2", "S2_valid"):
        if key not in arrays:
            raise ChipError(f"{name}: missing key {key!r}; expected keys S2, S2_valid, label and optionally AlphaEarth, AE_valid")
    s2 = np.asarray(arrays["S2"])
    valid = np.asarray(arrays["S2_valid"])
    if s2.shape != (N_BINS, N_BANDS, CHIP, CHIP):
        raise ChipError(f"{name}: S2 has shape {s2.shape}; expected {(N_BINS, N_BANDS, CHIP, CHIP)} (bins, bands, rows, cols)")
    if not np.issubdtype(s2.dtype, np.floating):
        raise ChipError(f"{name}: S2 has dtype {s2.dtype}; expected float32 reflectance")
    if valid.shape != (N_BINS, CHIP, CHIP):
        raise ChipError(f"{name}: S2_valid has shape {valid.shape}; expected {(N_BINS, CHIP, CHIP)}")
    if not np.isin(valid, (0, 1)).all():
        raise ChipError(f"{name}: S2_valid must contain only 0 and 1")
    valid = valid.astype(np.uint8)
    observed = np.broadcast_to(valid[:, None].astype(bool), s2.shape)
    values = s2[observed]
    if values.size and not np.isfinite(values).all():
        raise ChipError(f"{name}: S2 has non-finite values where S2_valid = 1; mark those observations invalid")
    if values.size and float(values.max()) > REFLECTANCE_MAX:
        raise ChipError(
            f"{name}: S2 maximum {float(values.max()):.1f} exceeds {REFLECTANCE_MAX}; the file looks like reflectance "
            "x 10 000 - multiply by 0.0001 before uploading"
        )
    if values.size and float(values.min()) < REFLECTANCE_MIN:
        raise ChipError(f"{name}: S2 minimum {float(values.min()):.3f} is below {REFLECTANCE_MIN}; not surface reflectance")
    clean = np.where(observed, s2, np.nan).astype(np.float32)
    label = None
    if "label" in arrays:
        label = np.asarray(arrays["label"])
        if label.shape != (CHIP, CHIP):
            raise ChipError(f"{name}: label has shape {label.shape}; expected {(CHIP, CHIP)}")
        if not np.isin(label, (0, 1, IGNORE)).all():
            bad = sorted(set(np.unique(label).tolist()) - {0, 1, IGNORE})[:5]
            raise ChipError(f"{name}: label values {bad} are outside 0 (non-rice), 1 (rice), 255 (ignore)")
        label = label.astype(np.uint8)
    elif require_label:
        raise ChipError(f"{name}: missing key 'label'; scoring and adaptation need the 0/1/255 rice label")
    ae = ae_valid = None
    if ("AlphaEarth" in arrays) != ("AE_valid" in arrays):
        raise ChipError(f"{name}: AlphaEarth and AE_valid must be supplied together (or both left out for an S2-only chip)")
    if "AlphaEarth" in arrays:
        ae = np.asarray(arrays["AlphaEarth"])
        ae_valid = np.asarray(arrays["AE_valid"])
        if ae.shape != (AE_DIM, CHIP, CHIP):
            raise ChipError(f"{name}: AlphaEarth has shape {ae.shape}; expected {(AE_DIM, CHIP, CHIP)} (embedding, rows, cols)")
        if not np.issubdtype(ae.dtype, np.floating):
            raise ChipError(f"{name}: AlphaEarth has dtype {ae.dtype}; expected float32 embedding values")
        if ae_valid.shape != (CHIP, CHIP):
            raise ChipError(f"{name}: AE_valid has shape {ae_valid.shape}; expected {(CHIP, CHIP)}")
        if not np.isin(ae_valid, (0, 1)).all():
            raise ChipError(f"{name}: AE_valid must contain only 0 and 1")
        ae_valid = ae_valid.astype(np.uint8)
        if not np.isfinite(ae[:, ae_valid.astype(bool)]).all():
            raise ChipError(f"{name}: AlphaEarth has non-finite values where AE_valid = 1; mark those pixels invalid")
        ae = ae.astype(np.float32)
    return {"S2": clean, "S2_valid": valid, "label": label, "AlphaEarth": ae, "AE_valid": ae_valid}


CHIP_DATASETS = {
    "S2": (N_BINS, N_BANDS, CHIP, CHIP), "S2_valid": (N_BINS, CHIP, CHIP), "label": (CHIP, CHIP),
    "AlphaEarth": (AE_DIM, CHIP, CHIP), "AE_valid": (CHIP, CHIP),
}


def read_chip_h5(path: Path, *, require_label: bool = True) -> dict[str, Any]:
    """Read only the five schema datasets (shape and dtype checked before reading) and the site attribute, then
    validate. `label_raw` and any other dataset are never read."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            arrays = {}
            for key, shape in CHIP_DATASETS.items():
                if key not in handle:
                    continue
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != shape or dataset.dtype.kind not in "fiub":
                    declared = getattr(dataset, "shape", "a group")
                    raise ChipError(f"{path.name}: {key} is {declared} ({getattr(dataset, 'dtype', 'no dtype')}); "
                                    f"expected a numeric dataset of shape {shape} (checked before reading)")
                arrays[key] = dataset[...]
            group = None
            for key in ("group", "group_id"):
                if key in handle.attrs:
                    value = handle.attrs[key]
                    try:
                        group = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    except UnicodeDecodeError:
                        group = None
                    if group:
                        break
    except OSError as exc:
        raise ChipError(f"{path.name}: not a readable HDF5 file ({exc})") from exc
    chip = validate_chip(arrays, name=path.name, require_label=require_label)
    chip["group"] = group
    return chip


def chip_tables(chip: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per-pixel feature rows for one validated chip: both feature sets when AlphaEarth is present, else S2-M1 only."""
    out = {"s2m1": m1_features(chip)}
    if chip.get("AlphaEarth") is not None:
        out["alphaearth"] = ae_features(chip)
    return out


# --------------------------------------------------------------------------------------------------
# dataset acquisition (public, anonymous; INDEX and member SHA-256 verified; expanded or zipped layouts)
# --------------------------------------------------------------------------------------------------


def locate_member(root: Path, row: Mapping[str, Any], cache: Path) -> Path:
    """Find one INDEX member in an expanded (`data-train/<path>`) or zipped (`data-train.zip`) layout, verify its
    SHA-256 against INDEX and return a local path. Zip members are read in memory and written only after verifying."""
    stem = row["archive"].removesuffix(".zip")
    for candidate in (root / stem / row["path"], root / row["path"]):
        if candidate.is_file():
            if sha256_file(candidate) != row["sha256"]:
                raise ValueError(f"{row['path']}: SHA-256 differs from INDEX.jsonl")
            return candidate
    archive_path = root / row["archive"]
    if archive_path.is_file():
        name = row["path"]
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe member name {name!r} in INDEX.jsonl")
        with zipfile.ZipFile(archive_path) as archive:
            data = archive.read(name)
        if sha256_bytes(data) != row["sha256"]:
            raise ValueError(f"{name} in {row['archive']}: SHA-256 differs from INDEX.jsonl")
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
    raise FileNotFoundError(f"{row['path']} not found in {root} (neither expanded nor zipped)")


def read_index(root: Path) -> list[dict[str, Any]]:
    data = (root / "INDEX.jsonl").read_bytes()
    if sha256_bytes(data) != INDEX_SHA256:
        raise ValueError(f"INDEX.jsonl SHA-256 {sha256_bytes(data)} != pinned {INDEX_SHA256}; the dataset changed")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(r["group_id"] == DEVELOPMENT_EXAMPLE_GROUP for r in rows):
        raise ValueError("the development example must not appear in the scoring index")
    return rows


def download_dataset(cache: Path) -> Path:
    """Anonymous kagglehub download of the pinned version; on Kaggle the attached copy under /kaggle/input is used."""
    import os

    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache / "kagglehub"))
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        os.environ.pop(name, None)  # anonymous: no credential is needed or used
    import kagglehub

    started = time.perf_counter()
    path = Path(kagglehub.dataset_download(f"{DATASET_SLUG}/versions/{DATASET_VERSION}"))
    print(f"dataset resolved anonymously by kagglehub in {time.perf_counter() - started:.1f} s: {path}", flush=True)
    return path


# --------------------------------------------------------------------------------------------------
# table staging (shared by the sample path and BYOD)
# --------------------------------------------------------------------------------------------------


def stage_tables(chips: Mapping[str, Mapping[str, Any]], roles: Mapping[str, Sequence[str]], out: Path, *,
                 evaluation_role: str, feature_sets: Sequence[str]) -> dict[str, Any]:
    """Write the pixel tables one run needs, per feature set:

    - `<fs>_fit_X/y.npy`: the FM_FIT_ROWS class-stratified fit sample (capstone pixel_sample), filled with its own
      column means (`<fs>_fill_fm.npy`); every foundation model gets these rows;
    - `<fs>_fitlgbm_X/y.npy`: LightGBM's LGBM_FIT_ROWS sample, filled with its own means (`<fs>_fill_lgbm.npy`);
    - `<fs>_dev_X/y/ids.npy`: the seeded DEV_SUBSET_ROWS development subset (raw, NaN where missing);
    - `<fs>_eval_X/y/ids.npy`: every labelled pixel of the evaluation chips (raw).

    `chips[cid]` holds `y` (flat labels) and one feature array per feature set. Fills are fitted on fit rows only.
    ids are (chip ordinal in `chip_order`, pixel index) pairs, so every row maps back to its chip and pixel."""
    out.mkdir(parents=True, exist_ok=True)
    order = [cid for role in ("fit", "development", evaluation_role) for cid in roles[role]]
    ordinal = {cid: i for i, cid in enumerate(order)}
    fit_labels = [chips[c]["y"] for c in roles["fit"]]
    dev_labels = [chips[c]["y"] for c in roles["development"]]
    picks = {
        "fit": pixel_sample(fit_labels, FM_FIT_ROWS, PIXEL_SEED),
        "fitlgbm": pixel_sample(fit_labels, LGBM_FIT_ROWS, PIXEL_SEED),
        "dev": proportional_sample(dev_labels, DEV_SUBSET_ROWS, DEV_SUBSET_SEED),
        "eval": [np.flatnonzero(chips[c]["y"] != IGNORE) for c in roles[evaluation_role]],
    }
    role_of = {"fit": "fit", "fitlgbm": "fit", "dev": "development", "eval": evaluation_role}
    manifest: dict[str, Any] = {"chip_order": order, "evaluation_role": evaluation_role, "feature_sets": {}}
    for fs in feature_sets:
        names = FEATURE_SETS[fs]
        record: dict[str, Any] = {"features": len(names)}
        for key, pick in picks.items():
            ids = roles[role_of[key]]
            x = np.concatenate([chips[c][fs][k] for c, k in zip(ids, pick, strict=True)]).astype(np.float32)
            y = np.concatenate([chips[c]["y"][k] for c, k in zip(ids, pick, strict=True)]).astype(np.int64)
            if x.shape[1] != len(names):
                raise ValueError(f"{fs}: {x.shape[1]} columns, expected {len(names)}")
            if key in ("fit", "fitlgbm"):
                fill = fit_fill(x)
                np.save(out / f"{fs}_fill_{'fm' if key == 'fit' else 'lgbm'}.npy", fill)
                x = fill_features(x, fill)
            else:
                ident = np.concatenate([np.stack([np.full(len(k), ordinal[c]), k], axis=1) for c, k in zip(ids, pick, strict=True)])
                np.save(out / f"{fs}_{key}_ids.npy", ident.astype(np.int32))
            np.save(out / f"{fs}_{key}_X.npy", x)
            np.save(out / f"{fs}_{key}_y.npy", y)
            record[key] = {"rows": int(len(y)), "rice": int(y.sum()), "chips": len(ids)}
        manifest["feature_sets"][fs] = record
    write_json(out / "tables.json", manifest)
    return manifest


def load_split(data: Path, fs: str, key: str, fill: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One staged table; `fill` ('fm' or 'lgbm') applies that fit-only fill to a raw development/evaluation table."""
    x = np.load(data / f"{fs}_{key}_X.npy")
    y = np.load(data / f"{fs}_{key}_y.npy")
    if fill is not None:
        x = fill_features(x, np.load(data / f"{fs}_fill_{fill}.npy"))
    return x, y


# --------------------------------------------------------------------------------------------------
# baseline B (LightGBM, capstone baseline-B settings) — lightgbm is imported only here
# --------------------------------------------------------------------------------------------------


def fit_lightgbm(x: np.ndarray, y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, names: Sequence[str]):
    """Capstone baseline B: at most LGBM_ROUND_CAP rounds, early stopping (patience LGBM_PATIENCE) on pooled
    development rice F1 at 0.5. Returns the booster; its best_iteration is the development choice."""
    import lightgbm as lgb

    def dev_f1(prob, data):
        value = scores(confusion(prob, data.get_label().astype(np.int64), 0.5))["f1"]
        return "dev_f1", (value if value is not None else 0.0), True

    return lgb.train(LGBM_PARAMS, lgb.Dataset(x, label=y.astype(np.float32), feature_name=list(names)),
                     num_boost_round=LGBM_ROUND_CAP,
                     valid_sets=[lgb.Dataset(dev_x, label=dev_y.astype(np.float32), feature_name=list(names))],
                     feval=dev_f1, callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False)])


# --------------------------------------------------------------------------------------------------
# rows for the result tables
# --------------------------------------------------------------------------------------------------


METRIC_COLUMNS = ["condition", "system", "feature_set", "split", "pixels", "threshold_rule", "threshold", "f1",
                  "iou", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn", "undefined"]


def grid_name(name: str, pixels: str) -> str:
    """Row label for a results grid: a system scored on a declared sample says so, so that its F1 is not read as a
    score on the same pixels as the systems scored on all of them."""
    match = re.match(r"declared seeded sample of (\d+) ", pixels or "")
    return f"{name} ({int(match.group(1)):,}-pixel sample)" if match else name


def metric_rows(condition: str, system: str, fs: str, split: str, probability: np.ndarray, label: np.ndarray,
                selected: float, pixels: str) -> list[dict[str, Any]]:
    """Pooled scores at 0.5 and at the development-selected threshold (both always reported)."""
    rows = []
    for rule, threshold in (("0.5", 0.5), ("development-selected", selected)):
        rows.append({"condition": condition, "system": system, "feature_set": fs, "split": split, "pixels": pixels,
                     "threshold_rule": rule, "threshold": threshold,
                     **scores(confusion(probability, label, threshold))})
    return rows


def per_chip_rows(condition: str, probability: np.ndarray, label: np.ndarray, ids: np.ndarray,
                  chip_order: Sequence[str], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for ordinal in sorted(set(ids[:, 0].tolist())):
        keep = ids[:, 0] == ordinal
        rows.append({"condition": condition, "chip": chip_order[ordinal], "threshold": threshold,
                     **scores(confusion(probability[keep], label[keep], threshold))})
    return rows


# --------------------------------------------------------------------------------------------------
# the run ledger: receipts, stale-output removal, freeze and score-once
# --------------------------------------------------------------------------------------------------

# Stages whose receipt must exist before a stage may start. Starting a stage deletes its own receipt and outputs and
# those of every stage that depends on it, so a failed or repeated upstream stage never leaves downstream work looking
# current (capstone design). After the evaluation chips are scored, nothing upstream may run again in that run folder.
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "baseline": ("prepare",), "foundation": ("prepare",), "activity": ("prepare",),
    "freeze": ("baseline", "foundation", "activity"), "score": ("freeze",), "export": ("score",),
}
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "prepare": ("tables", "data_manifest.json", "features.json", "chips.csv"), "baseline": ("baseline",), "foundation": ("foundation",), "activity": ("activity",),
    "freeze": ("selection.json", "frozen.json"), "score": ("scores", "scored.json"),
    "export": ("export", "results.zip"),
}
SCORE_ONCE = {"prepare", "baseline", "foundation", "activity", "freeze", "score"}


def _closure(stage: str, edges: Mapping[str, Sequence[str]]) -> set[str]:
    found: set[str] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for name, needs in edges.items():
            if current in needs and name not in found:
                found.add(name)
                frontier.append(name)
    return found


def dependents(stage: str) -> set[str]:
    return _closure(stage, PREREQUISITES)


class Ledger:
    """Receipts under `<root>/receipts`. `start` refuses a stage whose prerequisites have not completed, refuses any
    development stage once the evaluation chips were scored, and clears the stage's and its dependents' outputs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(parents=True, exist_ok=True)

    def done(self, stage: str) -> bool:
        return (self.receipts / f"{stage}.json").is_file()

    def start(self, stage: str) -> None:
        if stage in SCORE_ONCE and (self.root / "scored.json").exists():
            raise RuntimeError(f"{stage}: the evaluation chips were already scored in this run folder, and scoring "
                               "happens exactly once. Start a fresh runtime (Runtime > Disconnect and delete runtime) to "
                               "change a development choice.")
        missing = [name for name in PREREQUISITES.get(stage, ()) if not self.done(name)]
        if missing:
            raise RuntimeError(f"{stage} needs the {', '.join(missing)} stage(s) to finish first; run the cells in order")
        for name in {stage} | dependents(stage):
            (self.receipts / f"{name}.json").unlink(missing_ok=True)
            for relative in STAGE_OUTPUTS.get(name, ()):
                path = self.root / relative
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def finish(self, stage: str, **record: Any) -> dict[str, Any]:
        receipt = {"stage": stage, "finished_unix": round(time.time(), 1), **record}
        write_json(self.receipts / f"{stage}.json", receipt)
        return receipt

    def receipt(self, stage: str) -> dict[str, Any]:
        return read_json(self.receipts / f"{stage}.json")


def freeze(ledger: Ledger, selection: Mapping[str, Any]) -> dict[str, Any]:
    """Write selection.json and frozen.json (its SHA-256 and one identity hash per condition)."""
    write_json(ledger.root / "selection.json", selection)
    record = {
        "selection_sha256": sha256_file(ledger.root / "selection.json"),
        "conditions": {key: fingerprint(value) for key, value in selection["conditions"].items()},
        "frozen_unix": round(time.time(), 1),
    }
    write_json(ledger.root / "frozen.json", record)
    return record


def table_inventory(data: Path) -> dict[str, str]:
    """SHA-256 of every staged table file (features, labels, ids, fills and tables.json), frozen with the selection."""
    return {p.name: sha256_file(p) for p in sorted(data.iterdir()) if p.is_file()}


def verify_frozen(ledger: Ledger) -> dict[str, Any]:
    """Re-read the frozen selection and confirm it is byte-identical, every condition identity is unchanged, and the
    staged tables and manifests it names still have their frozen hashes (the data scored is the data frozen)."""
    frozen = read_json(ledger.root / "frozen.json")
    if sha256_file(ledger.root / "selection.json") != frozen["selection_sha256"]:
        raise RuntimeError("selection.json changed after the freeze; start a fresh runtime")
    selection = read_json(ledger.root / "selection.json")
    for key, value in selection["conditions"].items():
        if fingerprint(value) != frozen["conditions"].get(key):
            raise RuntimeError(f"condition {key} changed after the freeze; start a fresh runtime")
    for name, field in (("data_manifest.json", "data_manifest_sha256"), ("features.json", "features_sha256"),
                        ("tables/tables.json", "tables_sha256")):
        if field in selection and sha256_file(ledger.root / name) != selection[field]:
            raise RuntimeError(f"{name} changed after the freeze; start a fresh runtime")
    if "tables_inventory" in selection and table_inventory(ledger.root / "tables") != selection["tables_inventory"]:
        raise RuntimeError("a staged table file changed, appeared or disappeared after the freeze; start a fresh runtime")
    return selection


def mark_scored(ledger: Ledger, **record: Any) -> None:
    write_json(ledger.root / "scored.json", {"scored_unix": round(time.time(), 1), **record})


# --------------------------------------------------------------------------------------------------
# CSV and ZIP helpers (capstone)
# --------------------------------------------------------------------------------------------------


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 6))
    return value


def table_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    columns = list(columns or table_columns(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})


def build_zip(zip_path: Path, base: Path, members: Iterable[Path]) -> dict[str, str]:
    """Deterministic ZIP (sorted names, fixed timestamps) with a checksums.json entry; returns the member digests."""
    entries = sorted({p.resolve() for p in members if p.is_file()})
    checksums = {p.relative_to(base.resolve()).as_posix(): sha256_file(p) for p in entries}
    payload = json.dumps(checksums, indent=2, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(checksums):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (base / name).read_bytes())
        info = zipfile.ZipInfo("checksums.json", date_time=(2026, 1, 1, 0, 0, 0))
        archive.writestr(info, payload)
    return checksums


def verify_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe member name {name!r}")
        checksums = json.loads(archive.read("checksums.json"))
        missing = sorted(set(checksums) - set(names))
        mismatched = [n for n, d in checksums.items() if n in names and sha256_bytes(archive.read(n)) != d]
        extra = sorted(set(names) - set(checksums) - {"checksums.json"})
    if mismatched or extra or missing:
        raise ValueError(f"zip integrity failure: mismatched {mismatched}, missing {missing}, unlisted {extra}")
    return {"members": len(checksums), "verified": True}


# --------------------------------------------------------------------------------------------------
# BYOD: collection (capstone, verbatim), site grouping and split
# --------------------------------------------------------------------------------------------------


def byod_group(stem: str, group: str | None) -> str:
    """A chip's site: the HDF5 attribute `group` / `group_id` when present, else the file name without a trailing
    _YYYY year, so the years of one place never straddle fit and test."""
    return str(group) if group else re.sub(r"_(19|20)\d\d$", "", stem)


def byod_split(ids: Sequence[str], groups: Mapping[str, str], rice: Mapping[str, bool],
               nonrice: Mapping[str, bool]) -> dict[str, list[str]]:
    """Fit / development / test by whole site (SHA-256 order of 'rice-tables-byod:<k>:<site>', about 60/20/20); the
    first k giving rice in development and test and both classes in fit is used."""
    names = sorted({groups[i] for i in ids})
    if len(names) < BYOD_MIN_SITES:
        raise ChipError(f"adaptation needs at least {BYOD_MIN_SITES} sites so that fit, development and test differ; "
                        f"found {len(names)}: {names}")
    for attempt in range(200):
        order = sorted(names, key=lambda g: sha256_bytes(f"rice-tables-byod:{attempt}:{g}".encode()))
        n_dev = max(1, round(0.2 * len(order)))
        role_sites = {"development": order[:n_dev], "test": order[n_dev:2 * n_dev], "fit": order[2 * n_dev:]}
        split = {role: sorted(i for i in ids if groups[i] in members) for role, members in role_sites.items()}
        if (split["fit"] and any(rice[i] for i in split["development"]) and any(rice[i] for i in split["test"])
                and any(rice[i] for i in split["fit"]) and any(nonrice[i] for i in split["fit"])):
            return split
    raise ChipError("no site assignment gives rice in fit, development and test and non-rice in fit; "
                    "add more labelled sites")


def collect_byod(sources: Sequence[Path], target: Path) -> list[Path]:
    """Copy the .h5 chips named by `sources` (folders, .zip archives or .h5 files; a folder's .zip archives are opened
    too, as the upload dialog hands over a folder) into an emptied `target`. The folder is emptied first, so even a
    rejected retry keeps nothing from an earlier upload; unsafe or oversized archives and two chips with the same file
    name are refused before anything is copied."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    plan: dict[str, tuple[Path, str | None]] = {}
    total = 0

    def add(name: str, origin: Path, member: str | None, size: int) -> None:
        nonlocal total
        if name in plan:
            first = plan[name][0] if plan[name][1] is None else f"{plan[name][0]}:{plan[name][1]}"
            second = origin if member is None else f"{origin}:{member}"
            raise ChipError(f"two chips share the file name {name}: {first} and {second}; "
                            "rename one so each chip-year has a unique name")
        plan[name] = (origin, member)
        total += size

    def add_zip(source: Path) -> None:
        unreadable = ChipError(f"{source.name}: not a readable .zip archive (truncated, corrupted or misnamed); "
                               "re-create it or send the .h5 files directly")
        if not zipfile.is_zipfile(source):
            raise unreadable
        try:
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
        except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
            raise unreadable from exc
        for info in members:
            if info.filename.startswith("/") or "\\" in info.filename or ".." in Path(info.filename).parts:
                raise ChipError(f"{source.name}: unsafe member name {info.filename!r} (absolute, backslash or '..'); refused")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ChipError(f"{source.name}: member {info.filename!r} is a symbolic link; refused")
            if info.filename.endswith(".h5"):
                add(Path(info.filename).name, source, info.filename, info.file_size)

    for source in sources:
        if source.is_dir():
            for path in sorted(p for p in source.rglob("*") if p.suffix in (".h5", ".zip")):
                if path.is_symlink():
                    raise ChipError(f"{path}: symbolic links are refused; copy the file instead")
                if path.suffix == ".zip":
                    add_zip(path)
                else:
                    add(path.name, path, None, path.stat().st_size)
        elif source.suffix == ".zip":
            add_zip(source)
        elif source.suffix == ".h5":
            add(source.name, source, None, source.stat().st_size)
        else:
            raise ChipError(f"{source}: expected a folder, a .zip or .h5 files")
        if len(plan) > BYOD_MAX_FILES or total > BYOD_MAX_BYTES:
            raise ChipError(f"the input exceeds {BYOD_MAX_FILES} chips or {BYOD_MAX_BYTES // 2**30} GiB; send fewer files")
    for name, (origin, member) in plan.items():
        if member is None:
            shutil.copy2(origin, target / name)
        else:
            try:
                with zipfile.ZipFile(origin) as archive, archive.open(member) as source, \
                        (target / name).open("wb") as sink:
                    copied = 0
                    while chunk := source.read(1 << 20):  # bounded memory; the declared size is not trusted
                        copied += len(chunk)
                        if copied > BYOD_MAX_BYTES:
                            raise ChipError(f"{origin.name}:{member} expands beyond {BYOD_MAX_BYTES // 2**30} GiB; "
                                            "refused")
                        sink.write(chunk)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                if isinstance(exc, ChipError):
                    raise
                raise ChipError(f"{origin.name}:{member} could not be extracted ({type(exc).__name__}); "
                                "re-create the archive or send the .h5 files directly") from exc
    return sorted(target.glob("*.h5"))


def byod_plan(chips: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Which chips can adapt, which feature sets every adapting chip supports, and the site split. Raises ChipError
    with the failed rule before any model runs."""
    labelled = sorted(c for c, chip in chips.items()
                      if chip["label"] is not None and bool((np.asarray(chip["label"]) != IGNORE).any()))
    if len(labelled) < BYOD_MIN_ADAPT_CHIPS:
        raise ChipError(f"re-adaptation needs at least {BYOD_MIN_ADAPT_CHIPS} labelled chips (label values 0/1, not all "
                        f"255) from at least {BYOD_MIN_SITES} sites; found {len(labelled)} labelled chips")
    groups = {c: byod_group(c, chips[c].get("group")) for c in labelled}
    rice = {c: bool((chips[c]["label"] == 1).any()) for c in labelled}
    nonrice = {c: bool((chips[c]["label"] == 0).any()) for c in labelled}
    split = byod_split(labelled, groups, rice, nonrice)
    with_ae = all(chips[c].get("AlphaEarth") is not None for c in labelled)
    return {"labelled": labelled, "groups": groups, "split": split,
            "feature_sets": ["alphaearth", "s2m1"] if with_ae else ["s2m1"],
            "prediction_only": sorted(set(chips) - set(labelled))}


# --------------------------------------------------------------------------------------------------
# systems and the stages the notebook runs (sample path and BYOD share them)
# --------------------------------------------------------------------------------------------------

BASELINE_SYSTEMS = {
    "A": "All non-rice",
    "B": "LightGBM (65,536 fit rows)",
    "B10k": "LightGBM (10,000 fit rows, same-data control)",
}
FOUNDATION_SYSTEMS: dict[str, dict[str, Any]] = {
    "C1": {"name": "TabPFN-3", "family": "tabpfn", "settings": {"n_estimators": 4}, "chunk_rows": 8192,
           "process_per_feature_set": False, "declared_sample": False},
    "C2": {"name": "TabICLv2", "family": "tabicl", "settings": {"n_estimators": 8}, "chunk_rows": 8192,
           "process_per_feature_set": False, "declared_sample": False},
    "C3": {"name": "Mitra", "family": "mitra",
           "settings": {"time_limit_seconds": 900, "max_memory_usage_ratio": 8.0}, "chunk_rows": 1024,
           "process_per_feature_set": True, "declared_sample": True},
    "C4": {"name": "TabDPT v1.2", "family": "tabdpt",
           "settings": {"n_ensembles": 4, "context_size": 4096, "batch_size": 4096}, "chunk_rows": 8192,
           "process_per_feature_set": False, "declared_sample": False},
}
MITRA_SAMPLE_SEED = 20261014
MITRA_TUNING_ROWS = 2_000
PROTOCOL = {
    "roles": SPLIT_RULE,
    "fit_rows_foundation_models": FM_FIT_ROWS, "fit_rows_lightgbm": LGBM_FIT_ROWS, "fit_sample_seed": PIXEL_SEED,
    "fit_sample_rule": "capstone pixel_sample: round-robin quotas over (chip, class) strata, seeded",
    "development_subset_rows": DEV_SUBSET_ROWS, "development_subset_seed": DEV_SUBSET_SEED,
    "development_subset_rule": "seeded, stratified by (chip, class) in proportion to stratum size",
    "thresholds": list(THRESHOLDS), "threshold_rule": "highest pooled development rice F1; ties nearest 0.5, then lower",
    "lightgbm": {"params": LGBM_PARAMS, "round_cap": LGBM_ROUND_CAP, "patience": LGBM_PATIENCE,
                 "early_stopping": "pooled development rice F1 at 0.5"},
    "fills": "missing feature values filled with the column mean of the model's own fit sample",
    "evaluation": "all labelled evaluation pixels, pooled over label != 255, scored once after the freeze",
    "reload_tolerance": RELOAD_TOLERANCE,
}


def _chip_record(cid: str, role: str, chip: Mapping[str, Any], *, show_rice: bool, **extra: Any) -> dict[str, Any]:
    label = np.asarray(chip["label"]).reshape(-1)
    labelled = int((label != IGNORE).sum())
    ae_valid = chip.get("AE_valid")
    return {"chip": cid, "role": role, **extra, "labelled_pixels": labelled,
            "rice_share": round(float((label == 1).sum()) / labelled, 4) if show_rice and labelled else None,
            "s2_observed_share": round(float(np.asarray(chip["S2_valid"]).mean()), 4),
            "ae_valid_share": None if ae_valid is None else round(float(np.asarray(ae_valid).mean()), 4)}


def prepare_sample(ledger: Ledger, source: Path, cache: Path) -> dict[str, Any]:
    """Verify INDEX and every chip, assign roles, featurise both feature sets and stage the tables. Validation rice
    shares are not shown before the freeze."""
    ledger.start("prepare")
    started = time.perf_counter()
    rows = read_index(source)
    role_info = sample_roles(rows)
    roles = role_info["roles"]
    role_of = {c: r for r, ids in roles.items() for c in ids}
    chips: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    digests = {}
    for row in sorted(rows, key=chip_id):
        cid = chip_id(row)
        path = locate_member(source, row, cache / "chips")
        digests[cid] = row["sha256"]
        chip = read_chip_h5(path)
        if chip["AlphaEarth"] is None:
            raise ChipError(f"{cid}: the teaching dataset's chips carry AlphaEarth; this file does not")
        chips[cid] = {"y": chip["label"].reshape(-1), **chip_tables(chip)}
        table.append(_chip_record(cid, role_of[cid], chip, show_rice=role_of[cid] != "validation",
                                  region=row["region_iso"], year=row["year"]))
    manifest = stage_tables(chips, roles, ledger.root / "tables", evaluation_role="validation",
                            feature_sets=list(FEATURE_SETS))
    expanded = any((source / r["archive"].removesuffix(".zip")).is_dir() for r in rows)
    data_manifest = {
        "dataset": f"{DATASET_SLUG}/versions/{DATASET_VERSION}", "license": DATASET_LICENSE,
        "index_sha256": INDEX_SHA256, "chips_sha256": digests, "layout": "expanded" if expanded else "zipped",
        "development_example": f"{DEVELOPMENT_EXAMPLE_GROUP} (a training sample; never read)",
        "roles": roles, "split": {k: v for k, v in role_info["split"].items() if k not in ("fit", "development")},
        "label_dataset": "label (0 non-rice, 1 rice, 255 ignored); label_raw is never read",
    }
    return _finish_prepare(ledger, data_manifest, manifest, table, started)


def prepare_byod(ledger: Ledger, chips: Mapping[str, Mapping[str, Any]], plan: Mapping[str, Any],
                 digests: Mapping[str, str]) -> dict[str, Any]:
    """The same staging for validated BYOD chips: fit / development / test by site (plan from byod_plan)."""
    ledger.start("prepare")
    started = time.perf_counter()
    roles = {"fit": plan["split"]["fit"], "development": plan["split"]["development"], "test": plan["split"]["test"]}
    check_roles(roles)
    staged = {c: {"y": np.asarray(chips[c]["label"]).reshape(-1), **chip_tables(chips[c])}
              for ids in roles.values() for c in ids}
    table = [_chip_record(c, r, chips[c], show_rice=r != "test", site=plan["groups"][c])
             for r, ids in roles.items() for c in ids]
    manifest = stage_tables(staged, roles, ledger.root / "tables", evaluation_role="test",
                            feature_sets=plan["feature_sets"])
    data_manifest = {"dataset": "user-supplied chips (BYOD); processed inside this runtime only",
                     "chips_sha256": dict(digests), "roles": roles, "sites": dict(plan["groups"]),
                     "prediction_only_chips": plan["prediction_only"], "feature_sets": plan["feature_sets"],
                     "split_rule": BYOD_SPLIT_RULE}
    return _finish_prepare(ledger, data_manifest, manifest, table, started)


def _finish_prepare(ledger: Ledger, data_manifest: Mapping[str, Any], manifest: Mapping[str, Any],
                    table: Sequence[Mapping[str, Any]], started: float) -> dict[str, Any]:
    write_json(ledger.root / "data_manifest.json", data_manifest)
    data = ledger.root / "tables"
    features = {
        "recipes": {
            "s2m1": "per band mean/std/min/max/p10/p50/p90 of observed values, quarterly band means, quarterly "
                    "observed fractions (capstone m1_features)",
            "alphaearth": "64 AlphaEarth values (NaN where AE_valid = 0) + AE_valid (study ae_features)",
        },
        "names": {fs: FEATURE_SETS[fs] for fs in manifest["feature_sets"]},
        "fills": {fs: {kind: np.load(data / f"{fs}_fill_{kind}.npy").tolist() for kind in ("fm", "lgbm")}
                  for fs in manifest["feature_sets"]},
    }
    write_json(ledger.root / "features.json", features)
    write_csv(ledger.root / "chips.csv", table)
    ids = [np.load(data / f"{fs}_eval_ids.npy") for fs in manifest["feature_sets"]]
    if any(not np.array_equal(ids[0], other) for other in ids[1:]):
        raise RuntimeError("the feature sets disagree on which evaluation pixels are scored")
    ledger.finish("prepare", seconds=round(time.perf_counter() - started, 1), chips=len(table))
    return {"tables": dict(manifest), "chips": list(table), "data_manifest": dict(data_manifest)}


def baseline_stage(ledger: Ledger, systems: Sequence[str] = ("A", "B", "B10k")) -> dict[str, Any]:
    """A (all non-rice) and LightGBM on every staged feature set; development choices only."""
    ledger.start("baseline")
    data = ledger.root / "tables"
    out = ledger.root / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    feature_sets = list(read_json(data / "tables.json")["feature_sets"])
    records: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for fs in feature_sets:
        names = FEATURE_SETS[fs]
        for system in systems:
            key = f"{system}/{fs}"
            started = time.perf_counter()
            if system == "A":
                _x, dev_y = load_split(data, fs, "dev")
                prob = np.zeros(len(dev_y))
                record: dict[str, Any] = {"kind": "trivial"}
            else:
                fit_key, fill = ("fitlgbm", "lgbm") if system == "B" else ("fit", "fm")
                x, y = load_split(data, fs, fit_key)
                dev_x, dev_y = load_split(data, fs, "dev", fill)
                booster = fit_lightgbm(x, y, dev_x, dev_y, names)
                model = out / f"{system}_{fs}.txt"
                booster.save_model(str(model), num_iteration=booster.best_iteration)
                prob = booster.predict(dev_x, num_iteration=booster.best_iteration)
                record = {"kind": "lightgbm", "fit_table": fit_key, "fill": fill, "fit_rows": int(len(y)),
                          "best_iteration": int(booster.best_iteration), "model_file": model.name,
                          "model_sha256": sha256_file(model)}
            np.save(out / f"{system}_{fs}_dev_proba.npy", np.asarray(prob, dtype=np.float32))
            chosen = select_threshold([prob], [dev_y])
            record.update(system=system, name=BASELINE_SYSTEMS[system], feature_set=fs,
                          threshold=chosen["threshold"], dev_f1_selected=chosen["f1"],
                          dev_f1_at_0p5=scores(confusion(prob, dev_y, 0.5))["f1"], dev_pixels=int(len(dev_y)),
                          seconds=round(time.perf_counter() - started, 2))
            records[key] = record
            rows += metric_rows(key, record["name"], fs, "development subset", prob, dev_y, chosen["threshold"],
                                f"{len(dev_y)} seeded development pixels")
    write_json(out / "records.json", records)
    write_csv(out / "dev_metrics.csv", rows, METRIC_COLUMNS)
    ledger.finish("baseline", conditions=sorted(records))
    return records


def _declared_indices(path: Path, labels: np.ndarray, size: int, seed: int,
                      exclude: np.ndarray | None = None) -> Path:
    y = np.asarray(labels).copy()
    if exclude is not None:
        y[exclude] = IGNORE
    index = proportional_sample([y], size, seed)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, index)
    return path


def foundation_stage(ledger: Ledger, systems: Sequence[str], runner, identities: Mapping[str, Any], *,
                     mitra_dev_pixels: int = 10_000, require_all: bool = True) -> dict[str, Any]:
    """Develop each selected foundation-model condition through `runner(system, mode, feature_sets, extra, out_dir)`
    (a child process of the model's isolated environment in the notebook), then choose its threshold on development."""
    ledger.start("foundation")
    data = ledger.root / "tables"
    root = ledger.root / "foundation"
    root.mkdir(parents=True, exist_ok=True)
    feature_sets = list(read_json(data / "tables.json")["feature_sets"])
    records: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for system in systems:
        spec = FOUNDATION_SYSTEMS[system]
        sdir = root / system
        extra: dict[str, Any] = {"data_dir": str(data)}
        if spec["declared_sample"]:
            extra["indices"] = {}
            for fs in feature_sets:
                _x, dev_y = load_split(data, fs, "dev")
                labelled = int((dev_y != IGNORE).sum())
                tune_rows = min(MITRA_TUNING_ROWS, labelled // 5)  # small BYOD sets keep a fifth for tuning
                dev_path = _declared_indices(sdir / "indices" / f"{fs}_dev.npy", dev_y,
                                             min(mitra_dev_pixels, labelled - tune_rows), MITRA_SAMPLE_SEED)
                tune_path = _declared_indices(sdir / "indices" / f"{fs}_tune.npy", dev_y, max(tune_rows, 1),
                                              MITRA_SAMPLE_SEED + 1, exclude=np.load(dev_path))
                tune_classes = set(np.unique(dev_y[np.load(tune_path)]).tolist())
                if tune_rows < 1 or tune_classes != {0, 1}:
                    errors[system] = (f"{fs}: {labelled} labelled development pixels leave no Mitra tuning set with "
                                      "both classes; add development chips or run without Mitra")
                    break
                extra["indices"][fs] = {"dev": str(dev_path), "tune": str(tune_path)}
            if system in errors:
                continue
        groups = [[fs] for fs in feature_sets] if spec["process_per_feature_set"] else [feature_sets]
        results: dict[str, Any] = {}
        try:
            for group in groups:
                result = runner(system, "develop", group, extra, sdir)
                for fs, r in result["feature_sets"].items():
                    results[fs] = {**r, "device": result.get("device"), "torch": result.get("torch"),
                                   "gpu": result.get("gpu")}
        except Exception as exc:  # recorded; raised below when every selected model is required
            errors[system] = f"{type(exc).__name__}: {exc}"[:2000]
            continue
        for fs in feature_sets:
            r = results[fs]
            prob = np.load(sdir / fs / "dev_proba.npy").astype(np.float64)
            _x, dev_y = load_split(data, fs, "dev")
            pixels = f"{len(dev_y)} seeded development pixels"
            if spec["declared_sample"]:
                dev_y = dev_y[np.load(extra["indices"][fs]["dev"])]
                pixels = f"declared seeded sample of {len(dev_y)} development pixels (seed {MITRA_SAMPLE_SEED})"
            chosen = select_threshold([prob], [dev_y])
            key = f"{system}/{fs}"
            records[key] = {
                "kind": "foundation", "system": system, "name": spec["name"], "family": spec["family"],
                "feature_set": fs, "settings": spec["settings"], "chunk_rows": spec["chunk_rows"],
                "identity": identities.get(spec["family"], {}), "fit_rows": r["fit_rows"], "fit_rice": r["fit_rice"],
                "artifact_inventory": r["artifact_inventory"], "threshold": chosen["threshold"],
                "dev_f1_selected": chosen["f1"], "dev_f1_at_0p5": scores(confusion(prob, dev_y, 0.5))["f1"],
                "dev_pixels": int(len(dev_y)), "dev_pixels_rule": pixels,
                "indices": extra.get("indices", {}).get(fs, {}),
                "timings_s": {k: r.get(k) for k in ("load_s", "fit_s", "predict_dev_s")},
                "peak_gpu_gib": r.get("peak_gpu_gib"), "device": r.get("device"), "torch": r.get("torch"),
                "gpu": r.get("gpu"),
            }
            rows += metric_rows(key, spec["name"], fs, "development subset", prob, dev_y, chosen["threshold"], pixels)
    write_json(root / "records.json", records)
    write_json(root / "errors.json", errors)
    if rows:
        write_csv(root / "dev_metrics.csv", rows, METRIC_COLUMNS)
    if errors and require_all:
        named = ", ".join(f"{FOUNDATION_SYSTEMS[s]['name']} ({m[:300]})" for s, m in errors.items())
        raise RuntimeError(f"foundation-model development failed for {named}. The log above names the cause; see "
                           "Troubleshooting. Set REQUIRE_ALL_SELECTED_MODELS = False only for an explicitly partial run.")
    ledger.finish("foundation", conditions=sorted(records), errors=errors)
    return {"records": records, "errors": errors}


def activity_stage(ledger: Ledger, runner, *, enabled: bool, system: str, feature_set: str,
                   sizes: Sequence[int], available: Sequence[str]) -> dict[str, Any]:
    """The context-size activity on development only. It changes nothing canonical: every model keeps the
    10,000-row context for the frozen comparison."""
    ledger.start("activity")
    out = ledger.root / "activity"
    out.mkdir(parents=True, exist_ok=True)
    data = ledger.root / "tables"
    if not enabled or system not in available:
        reason = "RUN_ACTIVITY is False" if not enabled else f"{FOUNDATION_SYSTEMS[system]['name']} was not run"
        summary: dict[str, Any] = {"ran": False, "reason": reason}
        write_json(out / "activity.json", summary)
        ledger.finish("activity", ran=False, reason=reason)
        return summary
    if feature_set not in read_json(data / "tables.json")["feature_sets"]:
        raise ValueError(f"ACTIVITY_FEATURE_SET {feature_set!r} is not staged in this run")
    _x, fit_y = load_split(data, feature_set, "fit")
    sizes = sorted({int(s) for s in sizes})
    if not sizes or max(sizes) > len(fit_y):
        raise ValueError(f"context sizes must be between 2 and the {len(fit_y)} fit rows staged for the foundation models")
    paths = {}
    for size, index in nested_context(fit_y, sizes, ACTIVITY_SEED).items():
        path = out / "indices" / f"{size}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, index)
        paths[str(size)] = str(path)
    result = runner(system, "sweep", [feature_set], {"sweep_indices": paths, "data_dir": str(data)}, out / system)
    _x, dev_y = load_split(data, feature_set, "dev")
    rows = []
    for item in result["feature_sets"][feature_set]["sizes"]:
        prob = np.load(out / system / feature_set / f"dev_proba_{item['context_rows']}.npy").astype(np.float64)
        chosen = select_threshold([prob], [dev_y])
        rows.append({"system": FOUNDATION_SYSTEMS[system]["name"], "feature_set": feature_set, **item,
                     "dev_f1_at_0p5": scores(confusion(prob, dev_y, 0.5))["f1"],
                     "dev_selected_threshold": chosen["threshold"], "dev_f1_selected": chosen["f1"]})
    write_csv(out / "activity.csv", rows)
    summary = {"ran": True, "system": system, "feature_set": feature_set, "sizes": sizes, "seed": ACTIVITY_SEED,
               "rows": rows, "canonical_context_rows": FM_FIT_ROWS}
    write_json(out / "activity.json", summary)
    ledger.finish("activity", ran=True, sizes=sizes)
    return summary


def freeze_stage(ledger: Ledger, **declared: Any) -> dict[str, Any]:
    """Collect every development choice into selection.json and freeze it (hash + per-condition identity)."""
    ledger.start("freeze")
    baseline = read_json(ledger.root / "baseline" / "records.json")
    foundation = read_json(ledger.root / "foundation" / "records.json")
    split_rule = read_json(ledger.root / "data_manifest.json").get("split_rule")  # BYOD records its own
    selection = {
        "protocol": {**PROTOCOL, "roles": split_rule} if split_rule else PROTOCOL,
        "data_manifest_sha256": sha256_file(ledger.root / "data_manifest.json"),
        "features_sha256": sha256_file(ledger.root / "features.json"),
        "tables_sha256": sha256_file(ledger.root / "tables" / "tables.json"),
        "tables_inventory": table_inventory(ledger.root / "tables"),
        "conditions": {**baseline, **foundation},
        "activity": read_json(ledger.root / "activity" / "activity.json"),
        **declared,
    }
    frozen = freeze(ledger, selection)
    ledger.finish("freeze", selection_sha256=frozen["selection_sha256"], conditions=sorted(selection["conditions"]))
    return selection


def runner_config(system: str, mode: str, feature_sets: Sequence[str], extra: Mapping[str, Any], out_dir: Path, *,
                  weights_dir: Path, seed: int) -> dict[str, Any]:
    """The JSON configuration fm_runner.py reads; `extra` comes from the stage (data_dir, indices, inventories...)."""
    spec = FOUNDATION_SYSTEMS[system]
    return {
        "family": spec["family"], "mode": mode, "settings": spec["settings"], "seed": int(seed),
        "weights_dir": str(weights_dir), "feature_sets": list(feature_sets),
        "feature_names": {fs: FEATURE_SETS[fs] for fs in feature_sets}, "output_dir": str(out_dir),
        "chunk_rows": spec["chunk_rows"], "reload_tolerance": RELOAD_TOLERANCE,
        "result_path": str(Path(out_dir) / f"{mode}-{'-'.join(feature_sets)}.json"), **extra,
    }


def _lightgbm_scores(ledger: Ledger, condition: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    model = ledger.root / "baseline" / condition["model_file"]
    if sha256_file(model) != condition["model_sha256"]:
        raise RuntimeError(f"{model.name} changed after the freeze")
    return lgb.Booster(model_file=str(model)).predict(x)  # reloaded from its file: the saved model is what is scored


def score_stage(ledger: Ledger, runner, *, split_label: str = "validation") -> dict[str, Any]:
    """Score every frozen condition on the evaluation chips, once."""
    ledger.start("score")
    selection = verify_frozen(ledger)
    mark_scored(ledger, status="started")  # from here on no development stage may run again in this run folder
    data = ledger.root / "tables"
    order = read_json(data / "tables.json")["chip_order"]
    out = ledger.root / "scores"
    out.mkdir(parents=True, exist_ok=True)
    conditions = selection["conditions"]
    probabilities: dict[str, np.ndarray] = {}
    samples: dict[str, np.ndarray] = {}
    parity: dict[str, Any] = {}
    by_system: dict[str, list[str]] = {}
    for key, condition in conditions.items():
        if condition["kind"] == "foundation":
            by_system.setdefault(condition["system"], []).append(key)
    for system, keys in by_system.items():
        spec = FOUNDATION_SYSTEMS[system]
        fss = [conditions[k]["feature_set"] for k in keys]
        extra: dict[str, Any] = {"artifact_inventory": {}, "thresholds": {}, "indices": {}, "data_dir": str(data)}
        for key, fs in zip(keys, fss, strict=True):
            extra["artifact_inventory"][fs] = conditions[key]["artifact_inventory"]
            extra["thresholds"][fs] = conditions[key]["threshold"]
            extra["indices"][fs] = dict(conditions[key]["indices"])
            if spec["declared_sample"]:
                _x, eval_y = load_split(data, fs, "eval")
                path = _declared_indices(out / "indices" / f"{system}_{fs}_eval.npy", eval_y,
                                         int(selection["mitra_validation_pixels"]), MITRA_SAMPLE_SEED + 2)
                extra["indices"][fs]["eval"] = str(path)
                samples[key] = np.load(path)
        groups = [[fs] for fs in fss] if spec["process_per_feature_set"] else [fss]
        for group in groups:
            result = runner(system, "score", group, extra, ledger.root / "foundation" / system)
            for fs, r in result["feature_sets"].items():
                parity[f"{system}/{fs}"] = {**r["reload_parity"], "reload_s": r.get("reload_s"),
                                            "predict_eval_s": r.get("predict_eval_s")}
        for key, fs in zip(keys, fss, strict=True):
            probabilities[key] = np.load(ledger.root / "foundation" / system / fs / "eval_proba.npy").astype(np.float64)
    for key, condition in conditions.items():
        if condition["kind"] == "trivial":
            _x, eval_y = load_split(data, condition["feature_set"], "eval")
            probabilities[key] = np.zeros(len(eval_y))
        elif condition["kind"] == "lightgbm":
            x, _y = load_split(data, condition["feature_set"], "eval", condition["fill"])
            probabilities[key] = np.asarray(_lightgbm_scores(ledger, condition, x), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    chip_rows: list[dict[str, Any]] = []
    for key, condition in conditions.items():
        fs = condition["feature_set"]
        _x, eval_y = load_split(data, fs, "eval")
        ids = np.load(data / f"{fs}_eval_ids.npy")
        prob = probabilities[key]
        pixels = f"all {len(eval_y)} labelled {split_label} pixels"
        if key in samples:
            eval_y, ids = eval_y[samples[key]], ids[samples[key]]
            pixels = f"declared seeded sample of {len(eval_y)} {split_label} pixels (seed {MITRA_SAMPLE_SEED + 2})"
        if len(prob) != len(eval_y):
            raise RuntimeError(f"{key}: {len(prob)} probabilities for {len(eval_y)} pixels")
        rows += metric_rows(key, condition["name"], fs, split_label, prob, eval_y, condition["threshold"], pixels)
        chip_rows += per_chip_rows(key, prob, eval_y, ids, order, condition["threshold"])
    for key, index in samples.items():  # every other condition on Mitra's sample, so that comparison is like-for-like
        fs = conditions[key]["feature_set"]
        _x, eval_y = load_split(data, fs, "eval")
        for other, condition in conditions.items():
            if other != key and condition["feature_set"] == fs and other not in samples:
                rows += metric_rows(other, condition["name"], fs, f"{split_label} (Mitra sample)",
                                    probabilities[other][index], eval_y[index], condition["threshold"],
                                    f"the same {len(index)} pixels as Mitra")
    write_csv(out / "metrics.csv", rows, METRIC_COLUMNS)
    write_csv(out / "per_chip_metrics.csv", chip_rows)
    write_json(out / "verification.json", {"reload_parity": parity, "tolerance": RELOAD_TOLERANCE})
    _write_predictions(out / f"{split_label}_predictions.csv.gz", data, order, conditions, probabilities, samples)
    failures = [k for k, v in parity.items()
                if not v["within_tolerance"] or v.get("decisions_identical_at_frozen_threshold") is False]
    mark_scored(ledger, status="complete", conditions=sorted(conditions), parity_failures=failures)
    ledger.finish("score", conditions=sorted(conditions), parity_failures=failures)
    if failures:
        raise RuntimeError(f"reload parity failed for {failures}: the reloaded artifact did not reproduce its "
                           f"development probabilities within {RELOAD_TOLERANCE} or its decisions at the frozen "
                           "threshold. The scores are recorded; preserve "
                           "verification.json and the logs and report it.")
    return {"rows": rows, "per_chip": chip_rows, "parity": parity}


def probe_reload(ledger: Ledger, runner, preferred: Sequence[str] = ("C1/alphaearth",)) -> dict[str, Any]:
    """Fresh-process reload of one frozen foundation-model condition after scoring: re-predict the first evaluation
    chunk from the saved artifact and compare with the scored probabilities (scores/probe.json)."""
    if not ledger.done("score"):
        raise RuntimeError("probe_reload needs the score stage to finish first; run the cells in order")
    selection = verify_frozen(ledger)
    conditions = {k: v for k, v in selection["conditions"].items() if v["kind"] == "foundation"}
    if not conditions:
        record: dict[str, Any] = {"ran": False, "reason": "no foundation model was selected; LightGBM was scored from "
                                                          "its saved model file"}
    else:
        key = next((k for k in preferred if k in conditions), sorted(conditions)[0])
        condition = conditions[key]
        fs = condition["feature_set"]
        indices = dict(condition["indices"])
        sample = ledger.root / "scores" / "indices" / f"{condition['system']}_{fs}_eval.npy"
        if sample.is_file():
            indices["eval"] = str(sample)
        extra = {"artifact_inventory": {fs: condition["artifact_inventory"]}, "thresholds": {fs: condition["threshold"]},
                 "indices": {fs: indices}, "data_dir": str(ledger.root / "tables")}
        result = runner(condition["system"], "probe", [fs], extra, ledger.root / "foundation" / condition["system"])
        record = {"ran": True, "condition": key, **result["feature_sets"][fs]}
    write_json(ledger.root / "scores" / "probe.json", record)
    if record.get("ran") and (not record["within_tolerance"]
                              or record.get("decisions_identical_at_frozen_threshold") is False):
        raise RuntimeError(f"validation reload parity failed for {record['condition']}: max difference "
                           f"{record['max_abs_difference']:.2e} (tolerance {record['tolerance']}), decisions identical "
                           f"at the frozen threshold: {record.get('decisions_identical_at_frozen_threshold')}; "
                           "preserve scores/probe.json")
    return record


def _write_predictions(path: Path, data: Path, order: Sequence[str], conditions: Mapping[str, Any],
                       probabilities: Mapping[str, np.ndarray], samples: Mapping[str, np.ndarray]) -> None:
    """One row per scored pixel: chip, pixel index (row-major in the 128 x 128 chip), label, and each condition's rice
    probability (blank where a declared-sample condition did not score that pixel)."""
    import gzip

    fs0 = next(iter(read_json(data / "tables.json")["feature_sets"]))
    ids = np.load(data / f"{fs0}_eval_ids.npy")
    _x, label = load_split(data, fs0, "eval")
    columns = {}
    for key in sorted(conditions):
        full = np.full(len(label), np.nan)
        if key in samples:
            full[samples[key]] = probabilities[key]
        else:
            full[:] = probabilities[key]
        columns[key] = full
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["chip", "pixel", "label", *columns])
        for i in range(len(label)):
            writer.writerow([order[ids[i, 0]], int(ids[i, 1]), int(label[i]),
                             *("" if np.isnan(v[i]) else repr(float(v[i])) for v in columns.values())])


EXPORT_MEMBERS = ("data_manifest.json", "features.json", "chips.csv", "selection.json", "frozen.json", "scored.json",
                  "baseline/records.json", "baseline/dev_metrics.csv", "foundation/records.json",
                  "foundation/errors.json", "foundation/dev_metrics.csv", "activity/activity.json",
                  "activity/activity.csv", "scores/metrics.csv", "scores/per_chip_metrics.csv",
                  "scores/verification.json", "scores/probe.json")


def export_stage(ledger: Ledger, summary: Mapping[str, Any], conclusion_template: str) -> dict[str, Any]:
    """results.zip: records, metrics, predictions, LightGBM models, receipts and attribution. Foundation-model
    artifacts stay in the runtime (their file inventories are in selection.json): the bundle carries no model weights
    and no source imagery."""
    ledger.start("export")
    export = ledger.root / "export"
    export.mkdir(parents=True)
    members = [*EXPORT_MEMBERS]
    members += [p.relative_to(ledger.root).as_posix() for p in (ledger.root / "baseline").glob("*.txt")]
    members += [p.relative_to(ledger.root).as_posix() for p in (ledger.root / "scores").glob("*_predictions.csv.gz")]
    members += [p.relative_to(ledger.root).as_posix() for p in ledger.receipts.glob("*.json")]
    for relative in members:
        source = ledger.root / relative
        if source.is_file():
            (export / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, export / relative)
    write_json(export / "run_summary.json", summary)
    (export / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    (export / "CONCLUSION_TEMPLATE.md").write_text(conclusion_template, encoding="utf-8")
    bundle = ledger.root / "results.zip"
    checksums = build_zip(bundle, export, export.rglob("*"))
    check = verify_zip(bundle)
    digest = sha256_file(bundle)
    ledger.finish("export", members=len(checksums), results_zip_sha256=digest)
    return {"path": bundle, "members": len(checksums), "verified": check["verified"], "sha256": digest}
