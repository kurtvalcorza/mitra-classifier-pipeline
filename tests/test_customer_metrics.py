"""Known-answer metric contracts and explicit repeated-customer bootstrap parity."""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "customer_metrics", Path(__file__).parents[1] / "tools/customer_metrics.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture_rows():
    rows = []
    for cutoff in ("2011-09-01", "2011-11-01"):
        for i in range(6):
            for system in ("good", "bad"):
                rows.append(
                    dict(
                        customer_id=str(i),
                        cutoff=cutoff,
                        y_true=int(i < 2),
                        system=system,
                        score=float(6 - i if system == "good" else i),
                        seen_in_support=i < 3,
                    )
                )
    return rows


def test_budget_ceil_and_known_metrics():
    result = m.ranking_metrics(list("abcdef"), [1, 0, 1, 0, 0, 0], [6, 5, 4, 3, 2, 1])
    assert result["k"] == 2
    assert result["precision_at_budget"] == result["recall_at_budget"] == 0.5
    assert result["lift_at_budget"] == 1.5
    assert result["average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["auroc"] == 7 / 8
    assert result["selected_negatives"] == result["missed_positives"] == 1


def test_tied_ap_auc_independent_of_hash_order():
    result = m.ranking_metrics(list("abcd"), [1, 0, 1, 0], [0.5] * 4)
    assert result["average_precision"] == result["auroc"] == 0.5
    assert result["selected_customer_ids"] == [min("abcd", key=m.tie_key)]


def test_probability_only_brier_and_reliability_endpoints():
    assert m.ranking_metrics(["a", "b"], [0, 1], [-1, -2])["brier"] is None
    result = m.ranking_metrics(["a", "b"], [0, 1], [0, 1], [0, 1])
    assert result["brier"] == 0
    assert result["reliability"][0]["count"] == result["reliability"][9]["count"] == 1
    assert sum(r["count"] for r in result["reliability"]) == 2


@pytest.mark.parametrize("labels,ap,recall", [([0, 0], None, None), ([1, 1], 1, 0.5)])
def test_single_class_undefined_is_null(labels, ap, recall):
    result = m.ranking_metrics(["a", "b"], labels, [1, 0])
    assert result["average_precision"] == ap
    assert result["recall_at_budget"] == recall
    assert result["auroc"] is None


def test_macro_equal_dates_not_pooled_rows_and_support_subgroups():
    rows = [
        dict(
            customer_id=str(i), cutoff=date, y_true=int(i == 0), system="s", score=-i, seen_in_support=i == 0
        )
        for date, n in [("a", 2), ("b", 10)]
        for i in range(n)
    ]
    result = m.evaluate(rows)
    assert result["macro"]["s"]["precision_at_budget"] == 0.75
    assert len(result["subgroups"]) == 4
    assert result["subgroups"][0]["metrics"]["n"] == 1


def test_missing_subgroup_is_not_measurable():
    rows = fixture_rows()
    for r in rows:
        r.pop("seen_in_support")
    assert all(row["status"] == "not_measurable" for row in m.evaluate(rows)["subgroups"])


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "different_labels",
        "cohort",
        "nan",
        "probability",
        "inconsistent_probability",
        "bad_label",
    ],
)
def test_source_validation_refusals(kind):
    rows = fixture_rows()
    if kind == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "different_labels":
        rows[0]["y_true"] = 0
    elif kind == "cohort":
        rows.pop()
    elif kind == "nan":
        rows[0]["score"] = np.nan
    elif kind == "probability":
        rows[0]["probability"] = 1.1
    elif kind == "inconsistent_probability":
        rows[0]["probability"] = 0.5
    elif kind == "bad_label":
        rows[0]["y_true"] = True
    with pytest.raises(ValueError):
        m.evaluate(rows)


def test_customer_bootstrap_matches_expanded_repeated_draws():
    rows = fixture_rows()
    expected, invalid = [], 0
    customers = sorted({r["customer_id"] for r in rows})
    rng = np.random.default_rng(42)
    for _ in range(200):
        draw = rng.integers(len(customers), size=len(customers))
        expanded = [r for index in draw for r in rows if r["customer_id"] == customers[index]]
        differences = []
        good = True
        for date in sorted({r["cutoff"] for r in rows}):
            values = {}
            for system in ("good", "bad"):
                block = [r for r in expanded if r["cutoff"] == date and r["system"] == system]
                if len({r["y_true"] for r in block}) < 2:
                    good = False
                    break
                metric = m.ranking_metrics(
                    [r["customer_id"] for r in block],
                    [r["y_true"] for r in block],
                    [r["score"] for r in block],
                )
                values[system] = metric["precision_at_budget"]
            if not good:
                break
            differences.append(values["good"] - values["bad"])
        if good:
            expected.append(np.mean(differences))
        else:
            invalid += 1
    result = m.paired_bootstrap(rows, [("good", "bad")], n_boot=200)
    assert result["invalid_replicates"] == invalid
    assert result["contrasts"]["good_minus_bad"]["ci95"] == np.quantile(expected, [0.025, 0.975]).tolist()
    assert result["contrasts"]["good_minus_bad"]["difference"] == 1
    assert result == m.paired_bootstrap(rows, [("good", "bad")], n_boot=200)


def test_all_invalid_bootstrap_returns_null_interval():
    rows = fixture_rows()
    for row in rows:
        row["y_true"] = 1
    result = m.paired_bootstrap(rows, [("good", "bad")], n_boot=10)
    assert result["valid_replicates"] == 0
    assert result["invalid_single_class"] == 10
    assert result["contrasts"]["good_minus_bad"]["ci95"] is None


def test_repeated_customer_is_not_silently_deduplicated():
    result = m.ranking_metrics(["positive"] * 4 + ["negative"], [1] * 4 + [0], [1] * 4 + [0], budget=0.5)
    assert result["n"] == 5 and result["k"] == math.ceil(0.5 * 5)
    assert result["selected_customer_ids"] == ["positive"] * 3
    assert result["recall_at_budget"] == 0.75


def test_budget_ten_vs_twenty_recomputes_k():
    args = [list(map(str, range(11))), [1] + [0] * 10, list(range(11))]
    assert m.ranking_metrics(*args, budget=0.1)["k"] == 2
    assert m.ranking_metrics(*args, budget=0.2)["k"] == 3
