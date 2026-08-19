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


def _temporal_splits(feat: pd.DataFrame, val_frac: float, test_frac: float,
                     horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purged per-series chronological split into train / val / test.

    Within each store-product series (daily, contiguous) the layout by row index is
    ``[train | embargo | val | embargo | test]`` with an embargo of ``horizon`` rows at each
    boundary. Because a feature row at day ``t`` carries the target for day ``t + horizon``, the
    embargo guarantees a training row's target cannot fall inside the validation window, and a
    validation row's target cannot fall inside the test window — a strictly forward-looking,
    leak-free split, unlike a random holdout or a plain tail split. A series too short to purge
    cleanly is kept entirely in train (it then contributes no val/test rows, so it cannot leak)."""
    train_parts, val_parts, test_parts = [], [], []
    for _, group in feat.groupby("_key", sort=False):
        group = group.sort_values("_dt")
        n = len(group)
        n_test = min(max(int(round(n * test_frac)), 1 if n > 2 and test_frac > 0 else 0), n)
        rest = n - n_test
        n_val = min(max(int(round(n * val_frac)), 1 if rest > 1 else 0), max(rest - 1, 0))
        test_start = n - n_test
        val_end = test_start - horizon
        val_start = val_end - n_val
        train_end = val_start - horizon
        if train_end < 1:
            train_parts.append(group)  # too short to purge; train-only, cannot leak
            continue
        train_parts.append(group.iloc[:train_end])
        if n_val > 0 and val_end > val_start:
            val_parts.append(group.iloc[val_start:val_end])
        if n_test > 0:
            test_parts.append(group.iloc[test_start:])
    empty = feat.iloc[:0]
    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else empty
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else empty
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else empty
    return train, val, test


def build(src: Path, out: Path, horizon: int, n_series: int, max_rows: int,
          val_frac: float, seed: int, n_bins: int, test_frac: float = 0.2) -> Path:
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
        "_dt": df["dt"],
        "_key": df["store_id"].astype(str) + "_" + df["product_id"].astype(str),
    }).dropna().reset_index(drop=True)

    # Split on the contiguous per-series rows FIRST (the embargo relies on daily contiguity),
    # then cap each split for smoke-test size (holdouts need no contiguity after the split).
    train, val, test = _temporal_splits(feat, val_frac, test_frac, horizon)

    def _cap(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        return frame.sample(n=n, random_state=seed).reset_index(drop=True) if len(frame) > n else frame

    train = _cap(train, max_rows)
    val = _cap(val, max(1, int(max_rows * val_frac)))
    test = _cap(test, max(1, int(max_rows * test_frac)))

    # Demand band: cut the future value into n_bins quantile classes. Edges come from the
    # training rows only and are applied to val/test — the labels carry no holdout leakage.
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = train["future_sale"].quantile(qs).to_numpy()
    names = ["low", "mid", "high"] if n_bins == 3 else [f"q{i}" for i in range(n_bins)]

    feature_cols = [c for c in feat.columns if not c.startswith("_") and c != "future_sale"]

    def _banded(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            target=_band(frame["future_sale"].to_numpy(), edges, names)
        )[feature_cols + ["target"]].reset_index(drop=True)

    train, val, test = _banded(train), _banded(val), _banded(test)

    out.mkdir(parents=True, exist_ok=True)
    zip_p = out / f"freshretailnet-band-h{horizon}.zip"
    frames = {"train.csv": train, "val.csv": val}
    if len(test) > 0:
        frames["test.csv"] = test
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, frame in frames.items():
            path = out / arc
            frame.to_csv(path, index=False)
            zf.write(path, arc)
            path.unlink()

    print(f"series={n_series} horizon={horizon} n_bins={n_bins} "
          f"split=temporal(val_frac={val_frac}, test_frac={test_frac})")
    print(f"train={len(train)} val={len(val)} test={len(test)} features={train.shape[1] - 1}")
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
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Per-series chronological test tail (0 to omit test.csv)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=3,
                    help="Number of demand-band classes (2–10; default 3 = low/mid/high)")
    args = ap.parse_args()
    build(args.src, args.out, args.horizon, args.n_series,
          args.max_rows, args.val_frac, args.seed, args.n_bins, args.test_frac)


if __name__ == "__main__":
    main()
