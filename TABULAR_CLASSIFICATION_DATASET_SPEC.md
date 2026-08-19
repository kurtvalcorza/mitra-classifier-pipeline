# Mitra Classifier — Dataset and Validator Specification

## Dataset Format: CSV tables

```
dataset.zip
├── train.csv              (required)
├── val.csv                (optional — a holdout is split from train if missing)
└── test.csv               (optional — scored if present)
```

Each CSV is one row per training example. Column names are the header row. One column is
the classification **target**; its distinct values are the class labels. Every other column,
except those named in `drop_columns`, is a **feature**. Feature columns may be numeric or
categorical — Mitra handles both.

All CSVs in one dataset must share the same columns.

## Target and Features

| Concept | Rule |
|---|---|
| Target column | Named `target` by default; set the `target_column` preprocessing field to use another name |
| Target type | Categorical: 2–10 distinct class labels (Mitra's ceiling is 10 classes) |
| Class labels | The distinct values of the target column; may be strings or integers |
| Feature columns | Every column except the target and any listed in `drop_columns` |
| Dropped columns | Comma-separated list in `drop_columns` — use for row ids and raw date strings |

Binary (two classes) and multiclass (three to ten) are both supported. The fine-tuner infers
which from the number of distinct labels.

## Validation Checks

All row-based checks count **usable rows** — rows with a non-null target, the exact population
the fine-tuner trains on.

| Check | Required? | Rule |
|---|---|---|
| `no_nested_zip` | YES | The archive must not contain another `.zip` |
| `no_duplicate_tables` | YES | No two members may resolve to the same `train`/`val`/`test` table (e.g. `train.csv` and `dataset/train.csv`) |
| `train_csv_present` | YES | A `train.csv` must exist in the archive |
| `train_csv_parses` | YES | `train.csv` must parse as CSV |
| `target_column_present` | YES | The configured target column must exist |
| `target_has_values` | YES | The target must have at least one non-null value |
| `target_class_count` | YES | The target must have 2–10 distinct classes |
| `min_rows_per_class` | YES | The smallest class must have at least 2 rows (for a stratified split) |
| `feature_columns_present` | YES | At least one feature column must remain after removing target and `drop_columns` |
| `feature_limit` | YES | At most 500 feature columns (Mitra's limit) |
| `minimum_rows` | YES | At least 50 usable (non-null-target) training rows |
| `val_schema_matches_train` | YES if `val.csv` present | `val.csv` columns must equal `train.csv` columns |
| `test_schema_matches_train` | YES if `test.csv` present | `test.csv` columns must equal `train.csv` columns |
| `val_labels_subset_train` | YES if `val.csv` present | `val.csv` target labels must all appear in `train.csv` |
| `test_labels_subset_train` | YES if `test.csv` present | `test.csv` target labels must all appear in `train.csv` |
| `row_limit_advisory` | WARNING | Flags tables above the 10,000-row ceiling; the fine-tuner class-preservingly samples down |

The archive is also rejected before any read if it is a zip bomb (a member's compression ratio
or the total uncompressed size exceeds a safety bound).

## Row Ceiling

Mitra accepts at most **10,000 training rows**. This is a property of the model, not of the
hardware. Tables above the ceiling validate successfully and are seed-sampled to 10,000 rows
before fitting. `max_train_rows` may be lowered but not raised above 10,000.

## Class Ceiling

Mitra classifies at most **10 classes**. A target with more distinct values fails validation.
Merge rare labels, or bin a fine-grained target into a smaller set of ordered classes, to bring
the count within range.

## Missing Validation Split

If `val.csv` is absent, the fine-tuner carves a deterministic **stratified** holdout from
`train.csv` using `validation_split` (default 0.2) and the configured seed, keeping every class
represented in train, and reports the split in `result.json`. Setting `validation_split` to `0`
explicitly disables the holdout: the run reports `valRows: 0`, records a note, and trains on all
rows. Otherwise, a dataset with at least 20 rows is always trained with a reported holdout.

## Result JSON (validator output)

```json
{
  "successful": true,
  "message": "Tabular dataset validation succeeded.",
  "datasetSummary": {
    "fileCount": 2,
    "extensions": { ".csv": 2 },
    "sampleFiles": ["train.csv", "val.csv"],
    "source": "zip",
    "archive": "my-dataset.zip"
  },
  "checks": [
    { "name": "train_csv_present", "successful": true, "message": "Found training table at train.csv." },
    { "name": "target_class_count", "successful": true, "message": "Target 'target' has 3 classes (2–10 allowed)." }
  ],
  "metadata": {
    "targetColumn": "target",
    "classCount": 3,
    "classes": ["low", "mid", "high"],
    "featureColumnCount": 17,
    "rowCount": 6400
  }
}
```

## Dataloader (used by the fine-tuner)

```python
import io
import zipfile
from pathlib import Path

import pandas as pd


def read_csv_from_dataset(dataset_dir: Path, stem: str) -> pd.DataFrame | None:
    """Read <stem>.csv from a raw zip or an unzipped directory."""
    zips = sorted(dataset_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as zf:
            for member in zf.namelist():
                p = Path(member.lstrip("./"))
                if p.suffix.lower() == ".csv" and p.stem.lower() == stem:
                    with zf.open(member) as handle:
                        return pd.read_csv(io.BytesIO(handle.read()))
        return None
    for path in sorted(dataset_dir.rglob("*.csv")):
        if path.stem.lower() == stem:
            return pd.read_csv(path)
    return None
```

## Fine-tuner Model Call

```python
from autogluon.tabular import TabularPredictor

# The fine-tuner infers the problem type from the target's cardinality.
num_classes = train[target_column].nunique()
problem_type = "binary" if num_classes == 2 else "multiclass"

predictor = TabularPredictor(
    label=target_column,
    problem_type=problem_type,
    eval_metric="accuracy",
    path=output_dir / "mitra_predictor",
)
# fine_tune=True (GPU) fine-tunes the weights; fine_tune=False runs zero-shot in-context
# inference. The fine-tuner selects the mode from GPU availability at runtime.
predictor.fit(
    train,
    hyperparameters={"MITRA": {"fine_tune": use_gpu}},
    fit_weighted_ensemble=False,
    time_limit=time_limit_seconds,
)

# Confirm Mitra actually trained before reporting a result.
trained = list(predictor.model_names())
assert any("mitra" in m.lower() for m in trained), f"expected Mitra, got {trained}"

predictions = predictor.predict(val.drop(columns=[target_column]))
```

**Fine-tune versus zero-shot.** Fine-tuning Mitra requires a GPU; on CPU its backward pass hits
an unsupported low-precision path. When no GPU is available, the fine-tuner sets
`fine_tune=False` and Mitra runs **zero-shot** — in-context inference with no weight update.
Zero-shot is faster and CPU-safe, at some cost in accuracy. Each run records the effective
`mode` (`fine-tune` or `zero-shot`) and `device` in its `result.json`.

The saved `TabularPredictor` directory is the model artifact in either mode. It reloads with
`TabularPredictor.load(path)` and predicts on new rows with matching columns.

## Worked Example

[`examples/build_freshretailnet_dataset.py`](examples/build_freshretailnet_dataset.py) builds
a complete, valid dataset zip from the
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) daily
panel — one row per store-product-day, with sales-history features, the stockout signal, and
promo/holiday/weather covariates. The target is the demand a chosen number of days ahead,
binned into ordered classes (low/mid/high by default). It is a template for turning any
labelled `(entity, date, value)` panel into the CSV contract above. FreshRetailNet-50K is
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```
python examples/build_freshretailnet_dataset.py --src <train.parquet> --out ./out --horizon 7 --n-bins 3
```

## Common Dataset Mistakes

1. **Only one class.** A target with a single distinct value fails validation — classification
   needs at least two classes.
2. **Too many classes.** More than 10 distinct labels fails validation; merge rare labels or
   bin the target.
3. **Ids or raw dates left in.** Row ids and unparsed date strings add noise; list them in
   `drop_columns`.
4. **Mismatched splits.** `val.csv` or `test.csv` with different columns than `train.csv`
   fails validation.
5. **A zip inside the zip.** Extract and upload the CSVs at the archive root.
6. **Over 10,000 rows expecting all to be used.** The excess is sampled away; curate the most
   informative rows if the cap matters.
