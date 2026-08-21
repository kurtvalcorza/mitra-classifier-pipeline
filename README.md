# Mitra Classifier — DIMER Pipeline

A DIMER pipeline that fine-tunes [Mitra](https://huggingface.co/autogluon/mitra-classifier), a
pretrained tabular foundation model, on your own tabular-classification dataset. You supply a
table of rows with one categorical target column. The pipeline validates the table, fine-tunes
Mitra, and produces a saved model artifact and a holdout score.

Mitra is [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) licensed, so the trained
pipeline can be enabled and served without a usage restriction on the model itself. The
training data carries its own licence — see [Data licence](#data-licence-governs-the-served-model).

For platform-administrator setup and operations — resource profiles, weights delivery, network
egress, enable, and monitoring — see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## The model: Mitra

Mitra is a tabular foundation model built by the [AutoGluon](https://auto.gluon.ai) team at
AWS and released with open weights under Apache-2.0. It is pretrained only on synthetic data
and applies **in-context learning**: it reads a table of labelled examples as context and
predicts on new rows, the same paradigm as TabPFN and TabICL. See
[MODEL_CARD.md](MODEL_CARD.md) for provenance, checksums, licence, and how to supply the
weights to DIMER.

Mitra's distinguishing feature is its training mixture. According to
[Zhang et al. (2025)](https://arxiv.org/abs/2510.21204), Mitra is pretrained on a curated
mixture of synthetic priors chosen for three properties: standalone performance on real
tabular data, diversity, and distinctiveness within the mixture. The mixture combines
structural causal models (SCM) with tree-based priors — gradient boosting, random forest,
decision tree, and extra trees. Pretraining used 45 million synthetic datasets on eight A100
GPUs over roughly 60 hours, with no real data seen. On the TabRepo, TabZilla, and AMLB
benchmarks the authors report Mitra outperforming TabPFNv2 and TabICL on both classification
and regression, with better sample efficiency. Mitra was state of the art on these benchmarks
(and TabArena) at its 2025 release; AutoGluon notes that newer tabular foundation models have
since overtaken it.

### Checkpoints

Mitra ships as two checkpoints. This pipeline uses the classifier.

| Checkpoint | Target type | Hugging Face id |
|---|---|---|
| Classifier (this pipeline) | categorical | [`autogluon/mitra-classifier`](https://huggingface.co/autogluon/mitra-classifier) |
| Regressor | numeric | [`autogluon/mitra-regressor`](https://huggingface.co/autogluon/mitra-regressor) |

AutoGluon selects the checkpoint from the predictor's `problem_type`. With
`problem_type="binary"` or `"multiclass"` it loads the classifier. The classifier is a 12-layer
Transformer (512 embedding size, 4 attention heads, ~75.7M parameters) that applies both
row-wise and column-wise attention in each layer.

### Applicability

Mitra is designed for small tabular data and is strongest below about 5,000 samples and 100
features. Its hard limits are 10,000 training rows, 500 features, and **10 classes**. Because
it is an in-context learner, its accuracy depends on whether the features carry signal about
the class. Treat its benchmark results as evidence of strong performance where signal exists,
not as a guarantee on any table.

### Fine-tuning and zero-shot

Mitra supports two modes, both exposed as fine-tuning fields:

- **Fine-tune** (`fine_tune=true`, the default) adapts the pretrained weights to the uploaded
  table. It requires a GPU. A measured fine-tune on the 4,180-row sample ran within the 600 s
  time budget on a single GPU.
- **Zero-shot** (`fine_tune=false`) runs Mitra as an in-context learner with no weight update.
  It is CPU-safe and faster, at some cost in accuracy.

`fine_tune_steps` sets the number of fine-tuning steps; `0` uses AutoGluon's default.

Fine-tuning Mitra requires a GPU — on CPU the backward pass uses a low-precision path that
many CPUs do not support. The fine-tuner detects the GPU at runtime: with a GPU it fine-tunes;
without one it runs zero-shot automatically, regardless of the `fine_tune` setting. Each run
records the effective device and mode in `result.json` (`metrics.device`, `metrics.mode`).
See [Container images](#container-images).

### Binary and multiclass

The fine-tuner infers the problem type from the target's distinct-value count: two classes
give a binary problem, three to ten give multiclass. Both use the same classifier checkpoint.
A target with more than ten classes fails the run with a clear message. The effective type is
recorded in `result.json` (`metrics.problemType`, `metrics.numClasses`).

---

## When to use this pipeline

Use this pipeline for tabular classification: predicting a categorical label from a row of
features. Risk tiers, demand bands, churn/no-churn, quality grades, and any row-per-record
categorical prediction fit here. Do not use it for images. For vision tasks, use the Image
Classification, Object Detection, or Segmentation pipelines. For a numeric target, use the
[Mitra regressor pipeline](https://github.com/kurtvalcorza/mitra-regressor-pipeline).

---

## Repositories

The pipeline is two containers, one repository each. Each `Dockerfile` sits at its repository
root.

| Container | Repository | Runs on |
|---|---|---|
| Validator | `mitra-classifier-dataset-validator` | CPU |
| Fine-tuner | `mitra-classifier-finetuner` | GPU |

```
mitra-classifier-dataset-validator/     (CPU)
├── Dockerfile
├── validate.py          DIMER-facing entrypoint (delegates to validator.py)
├── validator.py         validation implementation
├── requirements.txt
└── README.md

mitra-classifier-finetuner/             (GPU)
├── Dockerfile
├── train.py             DIMER-facing entrypoint
├── requirements.txt
├── README.md
└── dimer-pipeline.json  preprocessing + fine-tuning fields
```

DIMER builds each repository from its root and launches the container by the portal naming
convention: `validate.py` for the validator and `train.py` for the fine-tuner. The validator's
tested logic lives in `validator.py`; `validate.py` is a thin entrypoint that delegates to it.

Keep `dimer-pipeline.json` at the fine-tuner repository root. It defines the preprocessing and
fine-tuning fields that end users see. Without it, the workbench preprocessing step renders
empty and the fine-tuning step stays locked.

The dataset-building helpers (`examples/`) and the dataset specs live in this umbrella
repository, not in the container repositories.

---

## Creating the pipeline

Prerequisites: portal access as AI Engineer, and both repositories reachable by the portal's
GitHub App.

1. Open **AI Engineer → New Pipeline** and set these fields:

   | Field | Value |
   |---|---|
   | Pipeline Name | `Mitra Tabular Classification` |
   | Description | `Fine-tune the Mitra tabular foundation model for classification using your own tabular dataset. Supports binary and multiclass classification, dataset validation, configurable preprocessing, evaluation, and export of the trained model.` |
   | Task Type | `Custom / Other` |
   | Base Model | `autogluon/mitra-classifier` |
   | Validator repository | `https://github.com/kurtvalcorza/mitra-classifier-dataset-validator` |
   | Fine-tuner repository | `https://github.com/kurtvalcorza/mitra-classifier-finetuner` |

2. Build both images.
3. Run the smoke test with a small dataset.
4. Enable the pipeline.

### Portal implementation notes

- **`Custom / Other` is the correct portal card** for tabular pipelines; DIMER has no native
  tabular task type. The pipeline therefore declares its own task identity: the fine-tuner image
  sets `DIMER_TASK_TYPE=tabular_classification` and relies on that baked fallback rather than
  trusting DIMER's current generic resolved task type, treating any value the platform sends as
  an override.
- **`dimer-pipeline.json` stays at the fine-tuner repository root.** The portal reads it there to
  render the preprocessing and fine-tuning fields.
- **Field-to-runtime mapping.** `datasetPreprocessing` keys are passed to the fine-tuner as
  `DIMER_PREPROCESSING_ARGS_JSON`; `modelFinetuning` keys as `DIMER_HYPERPARAMETERS_JSON`.
- **`model_id` is not declared in `dimer-pipeline.json`.** The DIMER **Base Model** field is
  authoritative for the checkpoint that loads.

---

## The dataset

### Format

A zip of CSV files. The full contract is in
[`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md).

```
dataset.zip
├── train.csv          (required)   one row per example; one categorical target column
├── val.csv            (optional)   same columns as train; a holdout is split off if absent
└── test.csv           (optional)   scored if present
```

The target column is named `target` by default; change it with the `target_column`
preprocessing field. Its distinct values are the class labels (2–10 classes). Every other
column, except those listed in `drop_columns`, is a feature. Feature columns may be numeric or
categorical.

### How to build a dataset

Mitra consumes a feature table, not raw records. Convert a time series or transaction log
(`entity, date, value`) into a training table by engineering one row per `(entity, date)`:

- **features** — history and context at that point: lags, rolling means and standard
  deviations, calendar fields, and any known covariates such as promotions, holidays, weather,
  or stock status.
- **target** — the categorical label to predict, for example a demand band a chosen number of
  days ahead, or a binary event flag.

[`examples/build_freshretailnet_dataset.py`](examples/build_freshretailnet_dataset.py) is a
runnable template that performs exactly this transformation, turning the
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) daily
panel into a valid dataset zip whose target is the demand band (low/mid/high) a chosen number
of days ahead:

```bash
python examples/build_freshretailnet_dataset.py --src <train.parquet> --out ./out --horizon 7 --n-bins 3
```

A ready-made 279 KB sample — [`examples/sample-data/freshretailnet-band-h7.zip`](examples/sample-data/freshretailnet-band-h7.zip)
(see its [dataset card](examples/sample-data/DATASET_CARD.md)) — is included for smoke-testing.

### Data licence governs the served model

The model is Apache-2.0, but a served pipeline is also bound by the licence of the data it was
trained on. A model fine-tuned on non-commercial data — for example CC BY-NC — may not be
appropriate to expose as a hosted service. FreshRetailNet-50K, used by the example, is
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): usable and servable without a
non-commercial restriction. Confirm the licence of any corpus before you enable a pipeline
built from it.

### Row and class ceilings

Mitra accepts at most 10,000 training rows and 10 classes. These are limits of the model, not
the hardware. The validator flags larger tables (and rejects more than 10 classes), and the
fine-tuner seed-samples rows down to the ceiling. Raise `max_train_rows` only up to 10,000.

---

## Configurable fields

Preprocessing (`datasetPreprocessing`):

| Field | Default | Purpose |
|---|---|---|
| `target_column` | `target` | Name of the categorical column to predict |
| `drop_columns` | — | Comma-separated columns to exclude from features (ids, raw dates) |
| `max_train_rows` | `10000` | Cap on training rows; larger tables are sampled to it |
| `validation_split` | `0.2` | Holdout fraction when the zip has no `val.csv` |

Fine-tuning (`modelFinetuning`):

| Field | Default | Purpose |
|---|---|---|
| `time_limit_seconds` | `600` | Fit time budget |
| `seed` | `0` | RNG seed; pin it for reproducible runs |
| `eval_metric` | `accuracy` | Metric optimized and reported (`accuracy`, `balanced_accuracy`, `log_loss`, `f1_macro`, `mcc`) |
| `fine_tune` | `true` | Fine-tune weights (GPU) or run zero-shot; a CPU instance forces zero-shot |
| `fine_tune_steps` | `0` | Fine-tuning steps; `0` uses AutoGluon's default. Ignored for zero-shot |

---

## Outputs

The fine-tuner writes the trained model to the run's output directory and a `result.json`
describing the run:

```json
{
  "successful": true,
  "message": "Mitra fine-tune succeeded on 4180 rows; holdout accuracy 0.5906.",
  "metrics": {
    "trainedModels": ["Mitra"],
    "mode": "fine-tune",
    "device": "cuda",
    "problemType": "multiclass",
    "numClasses": 3,
    "trainRows": 4180,
    "valRows": 1600,
    "requestedValidationSplit": 0.2,
    "effectiveValidationRows": 1600,
    "effectiveValidationSplit": 0.2768,
    "evalMetric": "accuracy",
    "headlineMetric": "accuracy",
    "headlineScore": 0.5906,
    "valEvaluation": { "accuracy": 0.5906, "balanced_accuracy": 0.5945, "mcc": 0.3880 },
    "test": { "rows": 1600, "evaluation": { "accuracy": 0.5588 } },
    "artifactPath": "…/mitra_predictor"
  },
  "provenance": {
    "baseModel": "autogluon/mitra-classifier",
    "baseModelRevision": "c425e9fa0910a6be1c494321792e7ba2a1367b1a",
    "baseModelRevisionExpected": "c425e9fa0910a6be1c494321792e7ba2a1367b1a",
    "weightsSha256": "e06a055e…", "expectedSha256": "e06a055e…",
    "source": "huggingface", "enforced": true,
    "dataset": { "file": "dataset.zip", "sha256": "…" },
    "autogluonVersion": "1.5.0"
  },
  "metadata": { "baseModel": "autogluon/mitra-classifier", "targetColumn": "target", "seed": 0 }
}
```

The headline score is the metric named in `eval_metric` (here `accuracy`); `valEvaluation` and
`test.evaluation` carry the full metric set AutoGluon reports. When the dataset zip includes a
`test.csv`, it is scored after fitting and its metrics appear under `test`. Numbers above are
illustrative; see the model card for measured smoke-test values.

The saved artifact is an AutoGluon `TabularPredictor` directory. It reloads with
`TabularPredictor.load(path)` and predicts on new rows with matching columns. Reload-and-serve
was verified in a process separate from training.

---

## Reproducibility

Mitra's fine-tuning is stochastic: two runs on identical data can differ unless the seed is
fixed. `seed` is a first-class hyperparameter, and every RNG the fit touches is seeded from it.
GPU kernel autotuning can still leave small residual variation, so runs are reproducible in
ranking but not guaranteed byte-identical. For byte-stable artifacts, also pin the model
weights into the image — see the fine-tuner `Dockerfile`.

---

## Resource profile

Each fine-tuning run executes as a Kubernetes job under a GPU profile. The platform's default
profile is 1 GPU and 8Gi memory. Request a larger profile from a platform administrator when
you create the pipeline. The default is a starting point, not a ceiling; the HPC deployment
has capacity well beyond it.

Mitra holds the training table in memory as in-context context, so its footprint grows with
the number of rows and features. A run on ~4,200 rows and 17 features used about 10 GB, already
above the 8Gi default. AutoGluon also declines to train a model whose projected footprint
exceeds roughly 90% of available memory, so the requested memory must clear the footprint with
headroom rather than match it.

Minimum profile to request:

| Resource | Minimum | Notes |
|---|---|---|
| GPU | 1 | Mitra runs on a single GPU |
| Memory | 12 Gi | Clears the measured ~8.7 GB with headroom for AutoGluon's memory guard |

Raise memory toward 16 Gi for datasets near the 10,000-row ceiling or with many feature
columns. If you request less, the memory guard can skip the fit. The pipeline reports that as
a failed run, never as a silent success.

### Container images

The fine-tuner provides two images. Both run the same `train.py`, which detects the GPU at
runtime and selects fine-tune (GPU) or zero-shot (CPU).

| Image | Base | Runs on | Notes |
|---|---|---|---|
| `Dockerfile` (default) | CUDA | GPU or CPU | Fine-tunes on a GPU; **auto-falls back to zero-shot on CPU** when no GPU is present. Large (~10 GB). |
| `Dockerfile.cpu` | slim | CPU only | Zero-shot; small image, no CUDA runtime. |

DIMER builds the repository's root `Dockerfile`. Choose per instance:

- **GPU instance** — use the default `Dockerfile`.
- **No-GPU instance, size not a concern** — use the default `Dockerfile`; it runs on the CPU
  node and falls back to zero-shot.
- **No-GPU instance, lean image wanted** — make `Dockerfile.cpu` the root `Dockerfile` (rename
  the CUDA one aside, then rename `Dockerfile.cpu` to `Dockerfile`).

Verified from the one default image: `--gpus all` → `device: cuda, mode: fine-tune`; without a
GPU → `device: cpu, mode: zero-shot`. The validator is CPU-only and needs no change.

---

## Provenance and traceability

This section records how the pipeline was built and how the models it produces stay auditable.

### How this pipeline was authored

The validator, fine-tuner, configuration, and documentation in this repository were drafted
with AI assistance (Anthropic Claude Opus 4.8, via Claude Code) and are pending human review
before production deployment. The following were verified by execution, not only generated:

- both container scripts byte-compile, `dimer-pipeline.json` validates against the field
  schema, and a unit-test suite covers the validator checks, the class-preserving split/cap,
  ambiguous-archive rejection, unseen-label detection, and the uploaded-weights path;
- the validator passes its full check set on the derived sample dataset;
- the fine-tuner trains Mitra on GPU, writes a valid artifact, and that artifact reloads and
  serves predictions in a separate process;
- the same image run without a GPU falls back to zero-shot on CPU;
- the base weights' SHA-256 is verified before fitting, and `test.csv` is scored when present.

Not yet verified, and requiring human sign-off: the DIMER portal image build, the on-platform
smoke test, the memory-profile request, and the platform's inference-serving integration.
Treat the generated code as a reviewed draft, not audited production code.

### Model lineage

| Field | Value |
|---|---|
| Base model | [`autogluon/mitra-classifier`](https://huggingface.co/autogluon/mitra-classifier) |
| Pinned weights revision | `c425e9fa0910a6be1c494321792e7ba2a1367b1a` |
| Licence | Apache-2.0 |
| Origin | [Zhang et al. (2025)](https://arxiv.org/abs/2510.21204); weights by the AutoGluon team |
| Framework | AutoGluon 1.5.0 |

Pinning the revision (fine-tuner `Dockerfile`, Option A) makes every run start from identical
weights. Without it, AutoGluon fetches the current revision at runtime, and the model can
change between builds.

### Data lineage

A trained model inherits the provenance and licence of the table it was fine-tuned on. Each
dataset should carry its source, its licence, and — for a derived table — the transformation
that produced it. The worked example documents its own: FreshRetailNet-50K (CC BY 4.0), a named
upstream revision, and a deterministic, seeded feature and label construction.

### Per-run record

Every fine-tuning run writes a `result.json` that serves as the run's provenance record. It
includes the base model, target column, dropped columns, seed, time budget, eval metric,
training device, row counts, the problem type and class count, the models actually trained, and
the resulting scores. Its `provenance` block also records the base-model revision resolved at
runtime, the expected pinned revision, a SHA-256 of the uploaded dataset, and the AutoGluon
version. Paired with the container image tag, this record forms a chain from data to served
model.

---

## References

- Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu,
  L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025).
  [*Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models*](https://arxiv.org/abs/2510.21204).
  NeurIPS 2025, pp. 17831–17876. https://doi.org/10.48550/arXiv.2510.21204 ·
  [OpenReview](https://openreview.net/forum?id=t8YRsWY6HM)
- AutoGluon. [*Tabular Foundational Models*](https://auto.gluon.ai/dev/tutorials/tabular/tabular-foundational-models.html) tutorial ([run in Colab](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-foundational-models.ipynb)).
- [Mitra classifier model card](https://huggingface.co/autogluon/mitra-classifier) · [Mitra regressor model card](https://huggingface.co/autogluon/mitra-regressor), Hugging Face.
- Amazon Science. [*Mitra: Mixed synthetic priors for enhancing tabular foundation models*](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models).
- [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) dataset (CC BY 4.0), Dingdong Inc.
