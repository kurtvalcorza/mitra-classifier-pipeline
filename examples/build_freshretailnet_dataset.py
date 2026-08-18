"""
Build a DIMER-contract classification dataset from FreshRetailNet-50K
====================================================================
Turns the raw FreshRetailNet-50K daily panel into a train.csv / val.csv zip that the
Mitra tabular-classification pipeline accepts.

One row per (store, product, day):
  - target   = the demand BAND HORIZON days ahead — the future daily sale_amount cut into
    n_bins quantile classes (default 3: low / mid / high). Bin edges are computed on the
    training rows only and applied to validation, so no validation signal leaks into the labels.
  - features = sales history (lags, rolling mean/std), the stockout signal
    (stock_hour6_22_cnt and its rolling mean), and same-day known covariates
    (discount, holiday, activity, weather, day-of-week, month)

This is the classification counterpart of the regressor pipeline's builder: identical panel
and features, but the numeric future value is binned into ordered demand classes instead of
being predicted directly.

FreshRetailNet-50K is CC BY 4.0, so datasets built from it may be used and served without
a non-commercial restriction. Source: Dingdong-Inc/FreshRetailNet-50K on Hugging Face.

Usage:
    python build_freshretailnet_dataset.py \
        --src /path/to/freshretailnet-50k/train.parquet \
        --out ./out --horizon 7 --n-series 220 --max-rows 8000 --n-bins 3

Output: <out>/freshretailnet-band-h<HORIZON>.zip  (train.csv + val.csv)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt",
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]


def _band(values: np.ndarray, edges: np.ndarray, names: list[str]) -> list[str]:
    """Map continuous values to ordered class names using interior quantile edges."""
    idx = np.digitize(values, edges)  # 0 .. len(edges) == n_bins - 1 .. n_bins-1
    return [names[i] for i in idx]


def build(src: Path, out: Path, horizon: int, n_series: int,
          max_rows: int, val_frac: float, seed: int, n_bins: int) -> Path:
    if n_bins < 2 or n_bins > 10:
        raise ValueError("n_bins must be between 2 and 10 (Mitra's class ceiling)")

    df = pd.read_parquet(src, columns=COLUMNS)
    df["dt"] = pd.to_datetime(df["dt"])

    # Sample a subset of store-product series for a smoke-test-sized table.
    keys = df[["store_id", "product_id"]].drop_duplicates()
    pick = keys.sample(n=min(n_series, len(keys)), random_state=seed)
    df = (df.merge(pick, on=["store_id", "product_id"])
            .sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True))

    sales = df.groupby(["store_id", "product_id"])["sale_amount"]
    stock = df.groupby(["store_id", "product_id"])["stock_hour6_22_cnt"]
    feat = pd.DataFrame({
        "lag_1": sales.shift(1),
        "lag_7": sales.shift(7),
        "lag_14": sales.shift(14),
        "roll_7_mean": sales.transform(lambda s: s.shift(1).rolling(7).mean()),
        "roll_28_mean": sales.transform(lambda s: s.shift(1).rolling(28).mean()),
        "roll_7_std": sales.transform(lambda s: s.shift(1).rolling(7).std()),
        "stockout_hours": df["stock_hour6_22_cnt"],
        "roll_7_stockout": stock.transform(lambda s: s.shift(1).rolling(7).mean()),
        "discount": df["discount"],
        "holiday_flag": df["holiday_flag"],
        "activity_flag": df["activity_flag"],
        "precpt": df["precpt"],
        "avg_temperature": df["avg_temperature"],
        "avg_humidity": df["avg_humidity"],
        "avg_wind_level": df["avg_wind_level"],
        "dow": df["dt"].dt.dayofweek,
        "month": df["dt"].dt.month,
        "future_sale": sales.shift(-horizon),
    }).dropna().reset_index(drop=True)

    if len(feat) > max_rows:
        feat = feat.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    val = feat.sample(frac=val_frac, random_state=seed)
    train = feat.drop(index=val.index)

    # Demand band: cut the future value into n_bins quantile classes. Edges come from the
    # training rows only and are applied to validation — the labels carry no holdout leakage.
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = train["future_sale"].quantile(qs).to_numpy()
    names = ["low", "mid", "high"] if n_bins == 3 else [f"q{i}" for i in range(n_bins)]

    train = train.assign(
        target=_band(train["future_sale"].to_numpy(), edges, names)
    ).drop(columns="future_sale").reset_index(drop=True)
    val = val.assign(
        target=_band(val["future_sale"].to_numpy(), edges, names)
    ).drop(columns="future_sale").reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    train_p, val_p = out / "train.csv", out / "val.csv"
    train.to_csv(train_p, index=False)
    val.to_csv(val_p, index=False)
    zip_p = out / f"freshretailnet-band-h{horizon}.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(train_p, "train.csv")
        zf.write(val_p, "val.csv")
    train_p.unlink()
    val_p.unlink()

    print(f"series={n_series} horizon={horizon} n_bins={n_bins}")
    print(f"train={len(train)} val={len(val)} features={train.shape[1] - 1}")
    print(f"band edges (from train): {[round(float(e), 3) for e in edges]}")
    print(f"train class counts: {train['target'].value_counts().to_dict()}")
    print(f"wrote {zip_p} ({zip_p.stat().st_size // 1024} KiB)")
    return zip_p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to FreshRetailNet-50K train.parquet")
    ap.add_argument("--out", type=Path, default=Path("./out"))
    ap.add_argument("--horizon", type=int, default=7, help="Days ahead to forecast")
    ap.add_argument("--n-series", type=int, default=220,
                    help="Number of store-product series to sample")
    ap.add_argument("--max-rows", type=int, default=8000,
                    help="Row cap (Mitra accepts at most 10000)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=3,
                    help="Number of demand-band classes (2–10; default 3 = low/mid/high)")
    args = ap.parse_args()
    build(args.src, args.out, args.horizon, args.n_series,
          args.max_rows, args.val_frac, args.seed, args.n_bins)


if __name__ == "__main__":
    main()
