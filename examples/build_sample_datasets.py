"""
Build sample datasets for the Mitra Classifier pipeline
======================================================
Builds the two cross-sectional companion sample datasets from OpenML:
1. Telco Customer Churn (OpenML 42178) -> telco-customer-churn.zip
2. Adult Census Income (OpenML 1590)   -> adult-census-income.zip

Usage:
    python examples/build_sample_datasets.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "sample-data"


def build_telco_churn() -> Path:
    print("Fetching Telco Customer Churn (OpenML 42178)...")
    telco_raw = sklearn.datasets.fetch_openml(data_id=42178, as_frame=True, parser="auto")
    df = telco_raw.frame.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Stratified sample of 2,400 rows: 1,440 train (60%), 480 val (20%), 480 test (20%)
    df_sub, _ = train_test_split(df, train_size=2400, random_state=SEED, stratify=df["Churn"])
    train_df, temp_df = train_test_split(df_sub, train_size=1440, random_state=SEED, stratify=df_sub["Churn"])
    val_df, test_df = train_test_split(temp_df, train_size=480, random_state=SEED, stratify=temp_df["Churn"])

    out_zip = OUT_DIR / "telco-customer-churn.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("train.csv", train_df.to_csv(index=False))
        z.writestr("val.csv", val_df.to_csv(index=False))
        z.writestr("test.csv", test_df.to_csv(index=False))
    print(f"  Wrote {out_zip} ({out_zip.stat().st_size:,} bytes)")
    return out_zip


def build_adult_income() -> Path:
    print("Fetching Adult Census Income (OpenML version 2)...")
    adult_raw = sklearn.datasets.fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult_raw.frame.copy()

    # Stratified sample of 2,500 rows: 1,500 train (60%), 500 val (20%), 500 test (20%)
    df_sub, _ = train_test_split(df, train_size=2500, random_state=SEED, stratify=df["class"])
    train_df, temp_df = train_test_split(df_sub, train_size=1500, random_state=SEED, stratify=df_sub["class"])
    val_df, test_df = train_test_split(temp_df, train_size=500, random_state=SEED, stratify=temp_df["class"])

    out_zip = OUT_DIR / "adult-census-income.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("train.csv", train_df.to_csv(index=False))
        z.writestr("val.csv", val_df.to_csv(index=False))
        z.writestr("test.csv", test_df.to_csv(index=False))
    print(f"  Wrote {out_zip} ({out_zip.stat().st_size:,} bytes)")
    return out_zip


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_telco_churn()
    build_adult_income()
