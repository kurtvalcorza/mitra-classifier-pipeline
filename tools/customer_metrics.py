"""Auditable repeat-purchase ranking metrics and customer-cluster uncertainty."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

TIE_VERSION = "42:customer-budget-tie-v1:"


def tie_key(customer_id: str) -> str:
    """Stable label-independent tie key shared across all systems and cutoffs."""
    return hashlib.sha256((TIE_VERSION + customer_id).encode("utf-8")).hexdigest()


def _records(rows) -> list[dict]:
    return rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)


def validate(rows) -> list[dict]:
    """Require one row per system/snapshot and identical labelled cohorts across systems."""
    rows = _records(rows)
    if not rows:
        raise ValueError("Empty scored cohort")
    seen, cohorts, truth, probability_policy = set(), defaultdict(set), {}, {}
    support = {}
    for row in rows:
        for name in ("customer_id", "cutoff", "system"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Nonempty string {name} required")
        y = row["y_true"]
        if isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) or y not in (0, 1):
            raise ValueError("Labels must be binary integers")
        score = row["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float, np.number))
            or not math.isfinite(score)
        ):
            raise ValueError("Ranking scores must be finite")
        key = row["customer_id"], row["cutoff"]
        full = row["system"], *key
        if full in seen:
            raise ValueError("Duplicate system/customer/cutoff row")
        seen.add(full)
        cohorts[row["system"]].add(key)
        if key in truth and truth[key] != y:
            raise ValueError("Systems disagree on labels")
        truth[key] = y
        has_probability = row.get("probability") is not None
        if probability_policy.setdefault(row["system"], has_probability) != has_probability:
            raise ValueError("Probability availability must be consistent per system")
        if has_probability:
            p = row["probability"]
            if (
                isinstance(p, bool)
                or not isinstance(p, (int, float, np.number))
                or not math.isfinite(p)
                or not 0 <= p <= 1
            ):
                raise ValueError("Probabilities must be finite within [0,1]")
        value = row.get("seen_in_support")
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise ValueError("seen_in_support must be boolean")
        if key in support and support[key] != value:
            raise ValueError("Systems disagree on support membership")
        support[key] = value
    if any(keys != next(iter(cohorts.values())) for keys in cohorts.values()):
        raise ValueError("All systems must score identical cohorts")
    return rows


def ranking_metrics(customer_ids, y_true, scores, probability=None, budget: float = 0.2) -> dict:
    """Compute one cutoff; AP/AUROC use tied-score groups, not hash-broken ranking.

    Duplicate customer IDs are allowed here specifically for bootstrap weights.
    Caller-facing evaluate/paired_bootstrap first refuse duplicate source rows.
    """
    ids, y, score = list(customer_ids), np.asarray(y_true), np.asarray(scores, dtype=float)
    if not len(ids) or y.shape != (len(ids),) or score.shape != y.shape:
        raise ValueError("Matching nonempty arrays required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or not 0 < budget <= 1:
        raise ValueError("Invalid labels/scores/budget")
    if any(not isinstance(customer, str) or not customer for customer in ids):
        raise ValueError("Nonempty customer IDs required")
    y = y.astype(int)
    n, positives = len(y), int(y.sum())
    k = math.ceil(budget * n)
    order = np.lexsort((np.asarray([tie_key(customer) for customer in ids]), -score))
    selected = int(y[order[:k]].sum())
    # Group actual score ties for standard noninterpolated average precision.
    score_order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[score_order], score[score_order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_score)), n - 1]
    cumulative_positive = np.cumsum(sorted_y)[endpoints]
    group_positive = np.diff(np.r_[0, cumulative_positive])
    group_size = np.diff(np.r_[0, endpoints + 1])
    ap = (
        float(np.sum(group_positive * cumulative_positive / (endpoints + 1)) / positives)
        if positives
        else None
    )
    negatives = n - positives
    auc = (
        float(
            np.sum((cumulative_positive - group_positive / 2) * (group_size - group_positive))
            / (positives * negatives)
        )
        if positives and negatives
        else None
    )
    result = dict(
        n=n,
        positives=positives,
        negatives=negatives,
        prevalence=positives / n,
        k=k,
        selected_fraction=k / n,
        selected_positives=selected,
        selected_negatives=k - selected,
        missed_positives=positives - selected,
        unselected_negatives=negatives - (k - selected),
        precision_at_budget=selected / k,
        recall_at_budget=selected / positives if positives else None,
        lift_at_budget=(selected / k) / (positives / n) if positives else None,
        average_precision=ap,
        auroc=auc,
        selected_customer_ids=[ids[i] for i in order[:k]],
        budget=budget,
        brier=None,
        reliability=None,
        class_status="both_classes" if positives and negatives else "single_class",
    )
    if probability is not None:
        p = np.asarray(probability, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid probability vector")
        result["brier"] = float(np.mean((p - y) ** 2))
        bins = np.minimum((p * 10).astype(int), 9)
        result["reliability"] = [
            dict(
                bin=i,
                lower=i / 10,
                upper=(i + 1) / 10,
                count=int((bins == i).sum()),
                mean_probability=float(p[bins == i].mean()) if np.any(bins == i) else None,
                observed_rate=float(y[bins == i].mean()) if np.any(bins == i) else None,
            )
            for i in range(10)
        ]
    return result


MACRO_FIELDS = (
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
    "average_precision",
    "auroc",
    "brier",
)


def evaluate(rows, budget: float = 0.2) -> dict:
    """Return each cutoff, equal-cutoff macro means and seen/unseen support diagnostics."""
    rows = validate(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["system"], row["cutoff"]].append(row)
    per_cutoff, subgroups = [], []

    def metric(subset):
        return ranking_metrics(
            [r["customer_id"] for r in subset],
            [r["y_true"] for r in subset],
            [r["score"] for r in subset],
            [r["probability"] for r in subset] if subset[0].get("probability") is not None else None,
            budget,
        )

    for (system, cutoff), group in sorted(grouped.items()):
        per_cutoff.append(dict(system=system, cutoff=cutoff, **metric(group)))
        for seen in (True, False):
            subset = [
                r for r in group if r.get("seen_in_support") is not None and r["seen_in_support"] == seen
            ]
            subgroups.append(
                dict(
                    system=system,
                    cutoff=cutoff,
                    seen_in_support=seen,
                    status="measured" if subset else "not_measurable",
                    metrics=metric(subset) if subset else None,
                )
            )
    macro = {}
    for system in sorted({r["system"] for r in rows}):
        values = [r for r in per_cutoff if r["system"] == system]
        macro[system] = {
            name: float(np.mean([r[name] for r in values]))
            if all(r[name] is not None for r in values)
            else None
            for name in MACRO_FIELDS
        }
        macro[system]["cutoff_count"] = len(values)
    return dict(
        per_cutoff=per_cutoff,
        macro=macro,
        subgroups=subgroups,
        macro_weighting="equal_cutoff",
        undefined_policy="null; never silently omitted from macro",
        tie_version=TIE_VERSION,
        budget=budget,
    )


def paired_bootstrap(
    rows, contrasts: list[tuple[str, str]], n_boot: int = 2000, seed: int = 42, budget: float = 0.2
) -> dict:
    """A-minus-B precision; resample customers, preserve multiplicity and every snapshot.

    A replicate is invalid if any fixed cutoff has no rows or lacks either class.
    All contrasts use the same valid replicates. Intervals condition on frozen dates.
    """
    rows = validate(rows)
    if type(n_boot) is not int or n_boot < 1 or not 0 < budget <= 1:
        raise ValueError("Positive bootstrap count and valid budget required")
    systems, cutoffs = sorted({r["system"] for r in rows}), sorted({r["cutoff"] for r in rows})
    if not contrasts or any(a not in systems or b not in systems or a == b for a, b in contrasts):
        raise ValueError("Contrasts require distinct known systems")
    customers = sorted({r["customer_id"] for r in rows})
    customer_index = {customer: i for i, customer in enumerate(customers)}
    # All system cohorts are already checked identical, and each source snapshot is unique.
    lookup = {(r["system"], r["cutoff"], r["customer_id"]): r for r in rows}
    blocks = []
    for cutoff in cutoffs:
        ids = sorted(r["customer_id"] for r in rows if r["system"] == systems[0] and r["cutoff"] == cutoff)
        y = np.array([lookup[systems[0], cutoff, customer]["y_true"] for customer in ids])
        indices = np.array([customer_index[customer] for customer in ids])
        hashes = np.array([tie_key(customer) for customer in ids])
        orders = {
            system: np.lexsort(
                (hashes, -np.array([lookup[system, cutoff, customer]["score"] for customer in ids]))
            )
            for system in systems
        }
        blocks.append((y, indices, orders))
    observed = evaluate(rows, budget)["macro"]
    rng = np.random.default_rng(seed)
    samples = {f"{a}_minus_{b}": [] for a, b in contrasts}
    invalid_empty, invalid_classes = 0, 0
    for _ in range(n_boot):
        multiplicity = np.bincount(
            rng.integers(len(customers), size=len(customers)), minlength=len(customers)
        )
        scores = defaultdict(list)
        reason = None
        for y, indices, orders in blocks:
            weights = multiplicity[indices]
            n, positives = int(weights.sum()), int(weights @ y)
            if n == 0:
                reason = "empty"
                break
            if positives in (0, n):
                reason = "classes"
                break
            k = math.ceil(budget * n)
            for system in systems:
                order = orders[system]
                w = weights[order]
                # Prefix weights exactly reproduce expanding duplicate draws, including a partial last unit.
                taken = np.minimum(w, np.maximum(k - (np.cumsum(w) - w), 0))
                scores[system].append(float(taken @ y[order]) / k)
        if reason:
            invalid_empty += reason == "empty"
            invalid_classes += reason == "classes"
            continue
        for a, b in contrasts:
            samples[f"{a}_minus_{b}"].append(float(np.mean(scores[a]) - np.mean(scores[b])))
    return dict(
        replicates=n_boot,
        seed=seed,
        customer_count=len(customers),
        cutoff_count=len(cutoffs),
        resampling_unit="customer_all_snapshots_with_multiplicity",
        budget=budget,
        invalid_replicates=invalid_empty + invalid_classes,
        invalid_empty_cutoff=invalid_empty,
        invalid_single_class=invalid_classes,
        valid_replicates=n_boot - invalid_empty - invalid_classes,
        invalid_policy="exclude replicate from every contrast if any fixed cutoff is empty or single-class",
        contrasts={
            f"{a}_minus_{b}": dict(
                difference=observed[a]["precision_at_budget"] - observed[b]["precision_at_budget"],
                ci95=np.quantile(samples[f"{a}_minus_{b}"], [0.025, 0.975]).tolist()
                if samples[f"{a}_minus_{b}"]
                else None,
                measure="equal_cutoff_macro_precision_at_budget",
            )
            for a, b in contrasts
        },
    )
