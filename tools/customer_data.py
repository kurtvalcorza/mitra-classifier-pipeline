"""Audited positive-purchase snapshots; all feature cleaning uses only past rows."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "Invoice": "invoice_id",
    "InvoiceNo": "invoice_id",
    "StockCode": "stock_code",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_time",
    "Price": "unit_price",
    "UnitPrice": "unit_price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}
COLUMNS = ["invoice_id", "stock_code", "quantity", "invoice_time", "unit_price", "customer_id", "country"]
FEATURES = ["recency_days", "order_count", "gross_value", "avg_order_value", "product_count"]
# Meanings are verified against source workbook descriptions in the frozen data manifest.
ADMIN_CODES = {
    "POST",
    "DOT",
    "D",
    "M",
    "C2",
    "BANK CHARGES",
    "AMAZONFEE",
    "ADJUST",
    "ADJUST2",
    "CRUK",
    "S",
    "B",
    "TEST001",
    "TEST002",
    "GIFT_0001_10",
    "GIFT_0001_20",
    "GIFT_0001_30",
    "GIFT_0001_40",
    "GIFT_0001_50",
    "GIFT_0001_70",
    "GIFT_0001_80",
}
TRAIN_CUTOFFS = ["2010-04-01", "2010-07-01", "2010-10-01", "2011-01-01"]
DEV_CUTOFFS = ["2011-04-01", "2011-07-01"]
TEST_CUTOFFS = ["2011-09-01", "2011-11-01"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.rename(columns=ALIASES).copy()
    required = set(COLUMNS) - {"country"}
    if not required.issubset(data.columns) or data.columns.duplicated().any():
        raise ValueError("Missing or ambiguous transaction schema")
    if "country" not in data:
        data["country"] = ""
    data = data[COLUMNS].copy()
    for name in ("invoice_id", "stock_code", "customer_id", "country"):
        data[name] = data[name].map(normalize_id)
    data["invoice_time"] = pd.to_datetime(data.invoice_time, errors="coerce")
    if getattr(data.invoice_time.dt, "tz", None) is not None:
        raise ValueError("Expected explicit timezone-naive source clock")
    for name in ("quantity", "unit_price"):
        data[name] = pd.to_numeric(data[name], errors="coerce").astype(float)
    return data


def clean_transactions(raw: pd.DataFrame, asof=None, deduplicate: bool = True) -> tuple[pd.DataFrame, dict]:
    data = canonicalize(raw)
    if asof is not None:
        data = data.loc[data.invoice_time < pd.Timestamp(asof)].copy()
    invalid_number = ~np.isfinite(data.quantity) | ~np.isfinite(data.unit_price)
    flags = {
        "missing_customer": data.customer_id.eq(""),
        "missing_invoice_or_product": data.invoice_id.eq("") | data.stock_code.eq(""),
        "invalid_timestamp": data.invoice_time.isna(),
        "nonfinite_number": invalid_number,
        "cancellation": data.invoice_id.str.upper().str.startswith("C"),
        "nonpositive_quantity": data.quantity.le(0),
        "zero_quantity": data.quantity.eq(0),
        "negative_quantity": data.quantity.lt(0),
        "nonpositive_price": data.unit_price.le(0),
        "zero_price": data.unit_price.eq(0),
        "negative_price": data.unit_price.lt(0),
        "administrative_code": data.stock_code.str.upper().isin(ADMIN_CODES),
    }
    # Invoice ambiguity is assessed only in the observed prefix, never using future rows.
    known = data.loc[data.invoice_id.ne("") & data.customer_id.ne("") & data.invoice_time.notna()]
    identity = known.groupby("invoice_id").agg(
        customers=("customer_id", "nunique"), times=("invoice_time", "nunique")
    )
    ambiguous = identity.index[(identity.customers > 1) | (identity.times > 1)]
    flags["ambiguous_invoice"] = data.invoice_id.isin(ambiguous)
    excluded = pd.concat(flags, axis=1).any(axis=1)
    clean = data.loc[~excluded].copy()
    duplicates = clean.duplicated(COLUMNS)
    audit = {
        "input_rows": len(data),
        "exclusion_counts_overlapping": {k: int(v.sum()) for k, v in flags.items()},
        "excluded_union_rows": int(excluded.sum()),
        "ambiguous_invoice_ids": sorted(ambiguous.tolist()),
        "exact_duplicate_qualifying_lines": int(duplicates.sum()),
        "duplicate_policy": "exact-line-deduplicated benchmark; provenance unresolved",
        "missing_customer_gross_positive_value_share": None,
    }
    gross = (data.quantity * data.unit_price).where(
        (data.quantity > 0) & (data.unit_price > 0) & ~invalid_number, 0
    )
    if gross.sum() > 0:
        audit["missing_customer_gross_positive_value_share"] = float(
            gross[flags["missing_customer"]].sum() / gross.sum()
        )
    if deduplicate:
        clean = clean.loc[~duplicates].copy()
    clean["line_value"] = clean.quantity * clean.unit_price
    if not np.isfinite(clean.line_value).all():
        raise ValueError("Nonfinite line value")
    audit["retained_lines"] = len(clean)
    audit["retained_orders"] = len(clean[["invoice_id", "customer_id"]].drop_duplicates())
    return clean.reset_index(drop=True), audit


def snapshots(
    raw: pd.DataFrame,
    cutoff,
    coverage_start,
    coverage_end,
    require_labels: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    t, start, end = map(pd.Timestamp, (cutoff, coverage_start, coverage_end))
    if any(pd.isna(x) for x in (t, start, end)) or start >= end:
        raise ValueError("Invalid observation coverage or cutoff")
    if t != t.normalize() or any(x.tzinfo is not None for x in (t, start, end)):
        raise ValueError("Cutoffs must be midnight on the timezone-naive source clock")
    if start > t - pd.Timedelta(days=90) or end < t:
        raise ValueError("Incomplete 90-day history coverage")
    if require_labels and end < t + pd.Timedelta(days=30):
        raise ValueError("Incomplete outcome window: evaluation not measurable")
    history, _ = clean_transactions(raw, asof=t, deduplicate=deduplicate)
    history = history.loc[history.invoice_time >= t - pd.Timedelta(days=90)]
    frame = (
        history.groupby("customer_id")
        .agg(
            last_order=("invoice_time", "max"),
            order_count=("invoice_id", "nunique"),
            gross_value=("line_value", "sum"),
            product_count=("stock_code", "nunique"),
            quantity=("quantity", "sum"),
        )
        .reset_index()
    )
    frame["recency_days"] = (t - frame.pop("last_order")).dt.total_seconds() / 86400
    frame["avg_order_value"] = frame.gross_value / frame.order_count
    frame["cutoff"] = t.strftime("%Y-%m-%d")
    frame["y_true"] = np.nan
    if require_labels:
        future, _ = clean_transactions(raw, asof=t + pd.Timedelta(days=30), deduplicate=deduplicate)
        buyers = future.loc[future.invoice_time >= t, "customer_id"].unique()
        frame["y_true"] = frame.customer_id.isin(buyers).astype(int)
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite features")
    return frame.sort_values("customer_id").reset_index(drop=True)


def select_cohort(frame: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    if cap < 1 or frame.duplicated(["customer_id", "cutoff"]).any():
        raise ValueError("Invalid cap or duplicate customer snapshot")
    chosen = frame.copy()
    chosen["sampling_hash"] = [
        hashlib.sha256(f"v1:{seed}:{c}:{t}".encode()).hexdigest()
        for c, t in zip(chosen.customer_id, chosen.cutoff, strict=True)
    ]
    return chosen.sort_values("sampling_hash").head(cap).reset_index(drop=True)


def acquire(root: Path) -> Path:
    manifest_path = root / "data_manifest.json"
    if not manifest_path.exists():
        manifest_path = (
            Path(__file__).resolve().parents[1] / "tutorials/customer_analytics/data_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "source.zip"
    item = manifest["archive"]
    if not archive.exists():
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(item["url"], timeout=120) as response, partial.open("wb") as f:
            size = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > item["bytes"]:
                    raise ValueError("Archive exceeds pinned size")
                f.write(chunk)
        partial.replace(archive)
    if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
        raise ValueError("Archive integrity failure")
    workbook = cache / "online_retail_II.xlsx"
    if not workbook.exists():
        with zipfile.ZipFile(archive) as z:
            info = z.getinfo("online_retail_II.xlsx")
            if info.file_size != manifest["workbook"]["bytes"]:
                raise ValueError("Workbook size mismatch")
            workbook.write_bytes(z.read(info))
    if (
        workbook.stat().st_size != manifest["workbook"]["bytes"]
        or sha256(workbook) != manifest["workbook"]["sha256"]
    ):
        raise ValueError("Workbook integrity failure")
    return workbook


def prepare(root: Path | str) -> tuple[pd.DataFrame, dict]:
    manifest_path = Path(root) / "data_manifest.json"
    if not manifest_path.exists():
        manifest_path = (
            Path(__file__).resolve().parents[1] / "tutorials/customer_analytics/data_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["administrative_codes"] != sorted(ADMIN_CODES) or manifest["aliases"] != ALIASES:
        raise ValueError("Frozen cleaning policy differs from runtime")
    workbook = acquire(Path(root))
    book = pd.ExcelFile(workbook)
    sheets, inventories = [], []
    for name in book.sheet_names:
        original = pd.read_excel(book, sheet_name=name)
        frame = canonicalize(original)
        inventories.append(
            {
                "sheet": name,
                "rows": len(frame),
                "columns": original.columns.tolist(),
                "missingness": {str(k): int(v) for k, v in original.isna().sum().items()},
                "start": str(frame.invoice_time.min()),
                "end": str(frame.invoice_time.max()),
            }
        )
        sheets.append(frame)
    raw = pd.concat(sheets, ignore_index=True)
    _, cleaning = clean_transactions(raw)
    audit = {
        "sheets": inventories,
        "rows": len(raw),
        "timezone": "naive source clock; timezone not supplied",
        "observed_start": str(raw.invoice_time.min()),
        "observed_end": str(raw.invoice_time.max()),
        "coverage_start": "2009-12-01",
        "coverage_end": "2011-12-09",
        "coverage_assumption": "Transaction log completeness assumed, not demonstrated by extrema.",
        "exact_duplicate_lines": int(raw.duplicated(COLUMNS).sum()),
        "cross_sheet_exact_overlap": int(
            pd.util.hash_pandas_object(sheets[1], index=False)
            .isin(pd.util.hash_pandas_object(sheets[0], index=False))
            .sum()
        ),
        "numeric_ranges": {
            k: {"min": float(raw[k].min()), "max": float(raw[k].max())} for k in ("quantity", "unit_price")
        },
        "cleaning": cleaning,
    }
    return raw, audit


def audit_snapshots(raw: pd.DataFrame, config: dict) -> dict:
    """Recompute sample feasibility and duplicate sensitivity without predictions."""
    result = {"cutoffs": [], "duplicate_labels_stable": True, "dataset_feasible": True}
    support_classes = set()
    for role, key, cap in (
        ("train", "train_cutoffs", 512),
        ("development", "development_cutoffs", 1000),
        ("test", "test_cutoffs", 1000),
    ):
        for cutoff in config[key]:
            args = (raw, cutoff, config["coverage_start"], config["coverage_end"])
            canonical = snapshots(*args)
            retained = snapshots(*args, deduplicate=False)
            joined = canonical.merge(retained, on="customer_id", suffixes=("_dedup", "_keep"))
            if len(joined) != len(canonical) or len(joined) != len(retained):
                raise ValueError("Duplicate policy changed customer eligibility")
            if not joined.y_true_dedup.equals(joined.y_true_keep):
                raise ValueError("Duplicate policy changed order-existence labels")
            selected = select_cohort(canonical, cap)
            positive = int(selected.y_true.sum())
            negative = len(selected) - positive
            if role == "train":
                support_classes.update(selected.y_true.unique().tolist())
            elif min(positive, negative) < 20:
                raise ValueError("Scored cutoff has fewer than 20 examples of each class")
            _, cleaning = clean_transactions(raw, asof=cutoff)
            result["cutoffs"].append(
                {
                    "role": role,
                    "cutoff": cutoff,
                    "eligible": len(canonical),
                    "selected": len(selected),
                    "positive": positive,
                    "negative": negative,
                    "duplicate_labels_stable": True,
                    "duplicate_sensitivity_gross_value_sum_delta": float(
                        (joined.gross_value_keep - joined.gross_value_dedup).sum()
                    ),
                    "duplicate_sensitivity_quantity_sum_delta": float(
                        (joined.quantity_keep - joined.quantity_dedup).sum()
                    ),
                    "observable_prefix_cleaning": cleaning,
                }
            )
    if support_classes != {0, 1}:
        raise ValueError("Training support needs both classes")
    return result


def validate_byod(csv_path: Path, config: dict) -> pd.DataFrame:
    if config.get("rights_confirmed") is not True:
        raise ValueError("Explicit processing-rights acknowledgement required")
    if config.get("source_clock") != "timezone-naive" or config.get("currency") != "GBP":
        raise ValueError("Compatible GBP currency and timezone-naive source clock required")
    if config.get("complete_observation_coverage") is not True:
        raise ValueError("Complete observation coverage acknowledgement required")
    if csv_path.stat().st_size > 200_000_000:
        raise ValueError("CSV exceeds 200 MB")
    raw = pd.read_csv(csv_path, dtype={"customer_id": str, "invoice_id": str, "stock_code": str})
    if not set(raw.columns).issubset(COLUMNS) or not (set(COLUMNS) - {"country"}).issubset(raw.columns):
        raise ValueError("Unexpected/identifying columns or missing schema fields")
    data = canonicalize(raw)
    start, end = pd.Timestamp(config["coverage_start"]), pd.Timestamp(config["coverage_end"])
    train, dev, test = [
        list(map(pd.Timestamp, config[k])) for k in ("train_cutoffs", "development_cutoffs", "test_cutoffs")
    ]
    if not all((train, dev, test)) or max(train) + pd.Timedelta(days=30) > min(dev) or max(dev) >= min(test):
        raise ValueError("Support labels must mature before ordered development/test cutoffs")
    ordered = train + dev + test
    if any(pd.isna(x) or x.tzinfo is not None for x in [start, end, *ordered]) or start >= end:
        raise ValueError("Invalid source-clock coverage or cutoffs")
    if any(x != x.normalize() for x in ordered):
        raise ValueError("Cutoffs must occur at midnight")
    if ordered != sorted(set(ordered)):
        raise ValueError("Cutoffs must be unique and ordered")
    if start > min(ordered) - pd.Timedelta(days=90) or end < max(ordered) + pd.Timedelta(days=30):
        raise ValueError("Incomplete history/outcome coverage; use explicit artifact-inference path")
    return data
