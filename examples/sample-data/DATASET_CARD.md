---
license: cc-by-4.0
task_categories:
  - tabular-classification
source_datasets:
  - Dingdong-Inc/FreshRetailNet-50K
  - OpenML/42178-telco-customer-churn
  - OpenML/1590-adult-census-income
tags:
  - sample
  - smoke-test
  - derived
  - benchmark
---

# Mitra Classifier — Sample Datasets

This directory holds ready-to-upload datasets for smoke-testing and learning with the Mitra classifier pipeline. Each dataset exercises different tabular characteristics: leakage-aware temporal demand classification, cross-sectional customer churn with nominal strings, and demographic tabular benchmarking.

These are tutorial/sanity fixtures, not benchmark evidence or deployment recommendations. Preserve the documented splits when comparing tutorial runs.

---

## 1. freshretailnet-band-h7 (Temporal Retail Demand)

A small, leakage-aware dataset derived from FreshRetailNet-50K for temporal panel classification.

- **Archive:** `freshretailnet-band-h7.zip` (238 KB)
- **Rows:** 4,180 train · 1,600 val · 1,600 test
- **Target:** `target` — 3-class binned demand band (`low` / `mid` / `high`).
- **Features:** 17 numeric features: sales history (`lag_1/7/14`, `roll_7_mean`, `roll_28_mean`, `roll_7_std`), stockout signal (`stockout_hours`, `roll_7_stockout`), and weather/calendar covariates.
- **Split method:** Purged chronological split with a 7-day embargo at split boundaries to prevent temporal leakage.
- **Licence:** CC BY 4.0 (Dingdong Inc).

---

## 2. telco-customer-churn (Cross-Sectional Customer Churn)

A fast, cross-sectional customer retention dataset demonstrating Mitra on nominal string features and binary classification.

- **Archive:** `telco-customer-churn.zip` (39 KB)
- **Rows:** 1,440 train · 480 val · 480 test (stratified sample from IBM Telco Churn, OpenML 42178)
- **Target:** `Churn` — binary target (`Yes` / `No`). Class balance: ~26.5% `Yes`.
- **Features:** 19 features (15 categorical strings, 4 numeric):
  - Categoricals: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
  - Numerics: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Split method:** Seed-stratified random split (seed 42). Missing values in `TotalCharges` are imputed strictly using the training partition's median ($1,278.80) to prevent validation/test leakage.
- **Duplicate note:** the committed training partition contains two exact duplicate labelled rows inherited from the sampled source records. The tutorial's duplicate detector intentionally reports them. They are retained so the committed archive remains reproducible; treat the warning as a prompt to inspect duplicates rather than as evidence of cross-split leakage.
- **Licence:** Apache 2.0 (IBM Corporation / OpenML 42178).

---

## 3. adult-census-income (Demographic Tabular Benchmark)

The classic tabular machine-learning benchmark for predicting individual income levels from census attributes.

- **Archive:** `adult-census-income.zip` (36 KB)
- **Rows:** 1,500 train · 500 val · 500 test (stratified sample from Adult Census, OpenML 1590)
- **Target:** `class` — binary target (`<=50K` / `>50K`). Class balance: ~24% `>50K`.
- **Features:** 14 features (8 categorical strings, 6 numeric):
  - Categoricals: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
  - Numerics: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **Split method:** Seed-stratified random split (seed 42).
- **Fairness / use caveat:** Adult includes sensitive and proxy attributes such as `race`, `sex`, and `native-country` and is widely used to study algorithmic fairness. High predictive performance on this sample does **not** establish fairness, suitability for consequential income decisions, or acceptable subgroup error rates. Do not use this tutorial fixture for eligibility, employment, credit, benefits, or other high-impact decisions without a separate fairness, legal, and domain-specific assessment.
- **Licence:** CC BY 4.0 (UCI Machine Learning Repository / OpenML 1590).

---

## Reproducibility

`build_sample_datasets.py` addresses both OpenML sources by numeric data ID, uses seed 42 for all sampling/splits, writes ZIP members with the fixed timestamp `2026-09-07 23:31:04`, and asserts the final SHA-256 values used by the tutorial. A rebuild therefore either reproduces the committed archive bytes or fails closed with a digest mismatch.

The Telco and Adult tutorial archives are served from immutable repository revision `3cdd68e427bd3fd652369924c5d0f05cca9d9b89`, which is retained by durable branch `anchors/sample-data-pr21-reviewed-20260911`. The notebook verifies each archive's SHA-256 before extraction, so deleting the feature branch after integration does not break the published samples.

## How Datasets Were Built

- **FreshRetailNet:** Generated by [`../build_freshretailnet_dataset.py`](../build_freshretailnet_dataset.py).
- **Telco Churn & Adult Census:** Generated by [`../build_sample_datasets.py`](../build_sample_datasets.py).
