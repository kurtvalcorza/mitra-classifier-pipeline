"""Known-answer temporal, purchase-event and bounded cohort contracts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import customer_data as data


def row(invoice="1", time="2011-03-01", customer="a", stock="12345", quantity=2, price=3):
    return dict(
        invoice_id=invoice,
        stock_code=stock,
        quantity=quantity,
        invoice_time=time,
        unit_price=price,
        customer_id=customer,
        country="UK",
    )


def snapshot(rows, labels=True, dedup=True):
    return data.snapshots(pd.DataFrame(rows), "2011-04-01", "2010-01-01", "2011-06-01", labels, dedup)


def test_half_open_boundaries_and_order_counts():
    rows = [
        row("1", "2011-01-01"),
        row("2", "2011-01-02"),
        row("2", "2011-01-02", stock="99999"),
        row("3", "2011-04-01"),
        row("4", "2011-05-01"),
    ]
    result = snapshot(rows).iloc[0]
    assert result.order_count == 2  # Jan 1 is exactly 90 days before April 1.
    assert result.gross_value == 18 and result.product_count == 2
    assert result.y_true == 1
    assert snapshot([row(), row("4", "2011-05-01")]).iloc[0].y_true == 0


def test_future_poison_and_return_do_not_rewrite_features():
    initial = pd.DataFrame([row()])
    poisoned = pd.DataFrame(
        [
            row(),
            row("1", "2011-04-02", customer="other"),
            row("C1", "2011-04-03", quantity=-2),
            row("9", "2012-01-01", price=1e20),
        ]
    )
    before = snapshot(initial.to_dict("records"), labels=False)
    after = snapshot(poisoned.to_dict("records"), labels=False)
    pd.testing.assert_frame_equal(before, after)


def test_duplicates_change_value_not_purchase_labels():
    rows = [row(), row(), row("2", "2011-04-02")]
    a, b = snapshot(rows), snapshot(rows, dedup=False)
    assert a.iloc[0].gross_value == 6 and b.iloc[0].gross_value == 12
    assert a.iloc[0].order_count == b.iloc[0].order_count == 1
    assert a.iloc[0].y_true == b.iloc[0].y_true == 1


@pytest.mark.parametrize(
    "mutation",
    [
        dict(customer=""),
        dict(invoice=""),
        dict(stock=""),
        dict(time=pd.NaT),
        dict(price=0),
        dict(price=-2),
        dict(price=np.inf),
        dict(quantity=0),
        dict(quantity=-1),
        dict(invoice="c100"),
        dict(stock="POST"),
        dict(stock="TEST001"),
    ],
)
def test_invalid_purchase_rows_are_reported(mutation):
    clean, audit = data.clean_transactions(pd.DataFrame([row(**mutation)]))
    assert clean.empty and audit["excluded_union_rows"] == 1


@pytest.mark.parametrize("value", ["broken", "", "03/20/2011", "2011-02-30", "2011-04-02T12:00:00+08:00"])
def test_unsupported_or_missing_text_timestamps_are_refused_with_row_and_value(value):
    rows = [row("1", "2011-03-20"), row("2", value)]
    with pytest.raises(ValueError, match=r"source clock \(row 2: "):
        data.canonicalize(pd.DataFrame(rows))


def test_mixed_iso_representations_give_identical_histories_and_labels():
    consistent = [row("1", "2011-03-20 00:00:00"), row("2", "2011-04-02 12:00:00")]
    mixed = [row("1", "2011-03-20"), row("2", "2011-04-02 12:00:00")]
    iso_t = [row("1", "2011-03-20"), row("2", "2011-04-02T12:00")]
    expected = snapshot(consistent)
    assert expected.iloc[0].y_true == 1
    pd.testing.assert_frame_equal(snapshot(mixed), expected)
    pd.testing.assert_frame_equal(snapshot(iso_t), expected)


def test_unknown_time_stays_in_every_prefix_audit():
    rows = pd.DataFrame([row("1", "2011-03-20"), row("2", "2011-03-21")])
    rows["invoice_time"] = pd.to_datetime(rows.invoice_time)
    rows.loc[1, "invoice_time"] = pd.NaT
    _, audit = data.clean_transactions(rows, asof="2011-04-01")
    assert audit["exclusion_counts_overlapping"]["invalid_timestamp"] == 1


def test_ambiguous_invoice_quarantined():
    clean, audit = data.clean_transactions(pd.DataFrame([row(), row(customer="b"), row("2")]))
    assert clean.invoice_id.tolist() == ["2"]
    assert audit["ambiguous_invoice_ids"] == ["1"]


def test_incomplete_coverage_not_false_negative():
    with pytest.raises(ValueError, match="outcome"):
        data.snapshots(pd.DataFrame([row()]), "2011-04-01", "2010-01-01", "2011-04-15")
    with pytest.raises(ValueError, match="history"):
        data.snapshots(pd.DataFrame([row()]), "2011-04-01", "2011-03-01", "2011-06-01")
    result = data.snapshots(pd.DataFrame([row()]), "2011-04-01", "2010-01-01", "2011-04-15", False)
    assert result.y_true.isna().all()


def test_sampling_does_not_consult_labels_or_input_order():
    frame = snapshot([row(str(i), customer=str(i)) for i in range(30)])
    a = data.select_cohort(frame, 10)
    other = frame.iloc[::-1].copy()
    other["y_true"] = 1 - other.y_true
    b = data.select_cohort(other, 10)
    assert a.customer_id.tolist() == b.customer_id.tolist()


def test_numeric_ids_and_aliases():
    frame = pd.DataFrame(
        [
            {
                "Invoice": 123.0,
                "StockCode": 456,
                "Quantity": 1,
                "InvoiceDate": "2011-01-01",
                "Price": 2,
                "Customer ID": 789.0,
            }
        ]
    )
    result = data.canonicalize(frame).iloc[0]
    assert result.invoice_id == "123" and result.customer_id == "789"


def test_byod_rejects_identifying_columns_and_unmatured_labels(tmp_path):
    path = tmp_path / "rows.csv"
    frame = pd.DataFrame([row()])
    frame["email"] = "private@example.invalid"
    frame.to_csv(path, index=False)
    config = dict(
        rights_confirmed=True,
        source_clock="timezone-naive",
        currency="GBP",
        complete_observation_coverage=True,
        coverage_start="2010-01-01",
        coverage_end="2012-01-01",
        train_cutoffs=["2011-01-01"],
        development_cutoffs=["2011-04-01"],
        test_cutoffs=["2011-07-01"],
    )
    with pytest.raises(ValueError, match="columns"):
        data.validate_byod(path, config)
    frame.drop(columns="email").to_csv(path, index=False)
    data.validate_byod(path, config)
    config["development_cutoffs"] = ["2011-01-10"]
    with pytest.raises(ValueError, match="mature"):
        data.validate_byod(path, config)
