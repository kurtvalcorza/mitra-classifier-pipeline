"""
Build sample datasets for the Mitra Classifier pipeline
======================================================
Builds the two cross-sectional companion sample datasets from OpenML:
1. Telco Customer Churn (OpenML 42178) -> telco-customer-churn.zip
2. Adult Census Income (OpenML 1590)   -> adult-census-income.zip

The archives are byte-reproducible: OpenML datasets are addressed by immutable
numeric data IDs, splits use a fixed seed, CSV line endings and ZIP member
timestamps are fixed, and the final archive SHA-256 must match the digest used by
the tutorial notebook.

Usage:
    python examples/build_sample_datasets.py
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "sample-data"
ARCHIVE_TIMESTAMP = (2026, 9, 7, 23, 31, 4)
TELCO_OPENML_DATA_ID = 42178
ADULT_OPENML_DATA_ID = 1590
EXPECTED_TELCO_SHA256 = "6f4f7a2583b2479c90bde1b34dd0dd51ea7fe9712c25ce6101f0033d6cf3c3e1"
EXPECTED_ADULT_SHA256 = "9469b59f3f50b5f05c2969ad59a08fe0973d36c4a1b360220b23e6395a4b44a1"


def write_archive(
    out_zip: Path,
    parts: list[tuple[str, pd.DataFrame]],
    expected_sha256: str,
) -> Path:
    """Write a deterministic train/val/test ZIP and fail closed on byte drift."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, part in parts:
            info = zipfile.ZipInfo(name, ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, part.to_csv(index=False, lineterminator="\n"))

    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Built {out_zip.name} does not match the recorded digest: "
            f"expected {expected_sha256}, got {digest}"
        )
    out_zip.write_bytes(payload)
    print(f"  Wrote {out_zip} ({len(payload):,} bytes, SHA-256 {digest})")
    return out_zip


def build_telco_churn() -> Path:
    print(f"Fetching Telco Customer Churn (OpenML data_id={TELCO_OPENML_DATA_ID})...")
    telco_raw = sklearn.datasets.fetch_openml(
        data_id=TELCO_OPENML_DATA_ID, as_frame=True, parser="auto"
    )
    df = telco_raw.frame.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges whitespace to NaN.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    # Stratified sample of 2,400 rows: 1,440 train (60%), 480 val (20%), 480 test (20%).
    df_sub, _ = train_test_split(
        df, train_size=2400, random_state=SEED, stratify=df["Churn"]
    )
    train_df, temp_df = train_test_split(
        df_sub, train_size=1440, random_state=SEED, stratify=df_sub["Churn"]
    )
    val_df, test_df = train_test_split(
        temp_df, train_size=480, random_state=SEED, stratify=temp_df["Churn"]
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # Train-only imputation: compute median TotalCharges strictly on train partition.
    if "TotalCharges" in train_df.columns:
        train_median = float(train_df["TotalCharges"].median())
        train_df["TotalCharges"] = train_df["TotalCharges"].fillna(train_median)
        val_df["TotalCharges"] = val_df["TotalCharges"].fillna(train_median)
        test_df["TotalCharges"] = test_df["TotalCharges"].fillna(train_median)
        print(f"  Imputed TotalCharges with train-only median: {train_median:.2f}")

    return write_archive(
        OUT_DIR / "telco-customer-churn.zip",
        [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)],
        EXPECTED_TELCO_SHA256,
    )


def build_adult_income() -> Path:
    print(f"Fetching Adult Census Income (OpenML data_id={ADULT_OPENML_DATA_ID})...")
    adult_raw = sklearn.datasets.fetch_openml(
        data_id=ADULT_OPENML_DATA_ID, as_frame=True, parser="auto"
    )
    df = adult_raw.frame.copy()

    # Stratified sample of 2,500 rows: 1,500 train (60%), 500 val (20%), 500 test (20%).
    df_sub, _ = train_test_split(
        df, train_size=2500, random_state=SEED, stratify=df["class"]
    )
    train_df, temp_df = train_test_split(
        df_sub, train_size=1500, random_state=SEED, stratify=df_sub["class"]
    )
    val_df, test_df = train_test_split(
        temp_df, train_size=500, random_state=SEED, stratify=temp_df["class"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return write_archive(
        OUT_DIR / "adult-census-income.zip",
        [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)],
        EXPECTED_ADULT_SHA256,
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_telco_churn()
    build_adult_income()
