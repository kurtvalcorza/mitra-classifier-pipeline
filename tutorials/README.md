# Mitra Classifier standalone Colab tutorials
 
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/mitra-classifier-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--classifier-ffcc4d?style=flat)](https://huggingface.co/autogluon/mitra-classifier)
[![Upstream](https://img.shields.io/badge/Upstream-autogluon%2Fautogluon-181717?style=flat&logo=github&logoColor=white)](https://github.com/autogluon/autogluon)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21204-b31b1b.svg)](https://arxiv.org/abs/2510.21204)

There are now two standalone Colab workflows. Both are aligned to **DIMER Notebook Specification v1.0** and declare their normative profile explicitly.

| Notebook | Profile | Spec | Release status |
|---|---|---|---|
| `mitra_classifier_colab.ipynb` | `E2E` | v1.0 | Candidate — static checks enforced; clean Colab execution of the release revision pending |
| `mitra_classifier_predictor_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | v1.0 | Candidate — static checks enforced; clean Colab execution of the release revision pending |

The exact dependency graphs used by the notebooks are committed as `requirements-colab.lock.txt` and `requirements-inference.lock.txt`; both release locks target Python 3.12. Release-grade status requires clean target-runtime execution evidence for the exact release revision; static CI alone is not execution evidence.

The lock inputs are committed as `requirements-colab.in` and `requirements-inference.in`. CI verifies that each notebook's embedded install graph is byte-for-byte identical to its committed lock, preventing notebook/lock drift.

The declared primary runtime for both notebooks is **Google Colab with Python 3.12**. Generic Jupyter compatibility is not claimed because the workflows intentionally use Colab upload/download primitives. Runtime-error guidance in both notebooks uses the same Colab-only support boundary.

The repository `MODEL_CARD.md` states that DIMER hosts **only `model.safetensors`**; canonical `config.json` is reconstructed locally and digest-verified. `MODEL_SOURCE = 'DIMER weights'` therefore requires exactly the weights file, verifies its pinned SHA-256, reconstructs the exact 86-byte canonical config, and verifies the config digest before staging. The current single-file DIMER distribution is not an offline ZIP/snapshot, so Notebook Spec MOD7's package-manifest condition is not triggered today. If DIMER later distributes an offline ZIP/snapshot, the notebook must validate that producer provenance before model loading. The notebooks remain Candidate because clean supported-Colab release evidence is still pending. A failed DIMER upload never falls back to the network.

Both inference workflows reject duplicate CSV headers before pandas can rename them.
Quoted column names and UTF-8 files with a byte-order mark are supported.

| Notebook | Purpose |
|---|---|
| [`mitra_classifier_colab.ipynb`](mitra_classifier_colab.ipynb) | Acquire/verify Mitra, bring data, evaluate, optionally fine-tune, infer, and export `mitra-predictor.zip` |
| [`mitra_classifier_predictor_inference_colab.ipynb`](mitra_classifier_predictor_inference_colab.ipynb) | Reload an exported `mitra-predictor.zip`, validate a new CSV, run inference, and download `predictions.csv` |

### Build/evaluate/export

[![Open build/evaluate/export tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_colab.ipynb)

`mitra_classifier_colab.ipynb` is a standalone tutorial for the Mitra Classifier checkpoint distributed through the DIMER Model Repository.

It does **not** depend on DIMER Workbench, DIMER APIs, or the DIMER validator/fine-tuner workers. Users can download `model.safetensors` from DIMER and run the notebook independently in Google Colab. As a separate explicitly selected source mode, the notebook can instead retrieve the exact pinned upstream checkpoint associated with the DIMER release; a failed or incomplete DIMER upload never falls back to the network.

The tutorial covers:

- DIMER `model.safetensors` upload or pinned-upstream checkpoint source selection;
- SHA-256 verification of `model.safetensors` and `config.json`;
- an explicit post-staging resolver check that refuses to continue unless Hugging Face resolves the verified offline snapshot;
- reporting the actual AutoGluon, PyTorch, CUDA-build, Python, and GPU runtime state used for the run;
- a bundled FreshRetailNet sample dataset for users who do not yet have their own CSV;
- preservation of the sample's provided `train.csv` / `val.csv` / `test.csv` splits;
- BYOD single-CSV inspection with a stratified random holdout for approximately IID data;
- a pre-split `train.csv` / `val.csv` / `test.csv` upload path for temporal, grouped, embargoed, or otherwise leakage-sensitive workflows;
- class-coverage and duplicate-row checks before evaluation;
- pretrained/in-context Mitra evaluation;
- optional GPU fine-tuning with an explicit requested step count;
- before/after metric comparison with metric direction and holdout-resolution guidance;
- companion classical tree baselines (LightGBM and Random Forest) with holdout leaderboard and device latency;
- in-memory post-hoc probability blending with strict label alignment and generalization assessment;
- inference on new CSV rows; and
- export of predictions, run metadata, and a reusable AutoGluon predictor ZIP.

### Use an exported predictor

[![Open exported-predictor inference tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_predictor_inference_colab.ipynb)

`mitra_classifier_predictor_inference_colab.ipynb` is intentionally inference-only. It starts from the `mitra-predictor.zip` created by the main tutorial.

The inference tutorial:

- installs `autogluon.tabular[mitra]==1.5.0`;
- uploads exactly one `mitra-predictor.zip`;
- computes the uploaded archive's SHA-256 and verifies it when an expected digest is supplied;
- rejects absolute/traversal/backslash/symlink archive paths, enforces extraction containment and a 4 GiB expanded-size ceiling;
- requires and verifies `artifact-manifest.json` against the exact extracted file set, per-file sizes, and SHA-256 digests;
- locates the saved AutoGluon predictor root via `predictor.pkl`;
- requires and validates `tutorial_run_metadata.json` before deserialization;
- reloads the saved predictor with `TabularPredictor.load(...)`;
- shows model/task/feature/provenance information;
- uploads one new CSV;
- validates required feature columns while allowing harmless column reordering and extra columns;
- runs `predict()` and `predict_proba()`; and
- writes and downloads `predictions.csv`.

It does **not** reacquire `model.safetensors` or `config.json`, does not call DIMER, and does not train or fine-tune. The whole point of `mitra-predictor.zip` is that it is already the reusable downstream predictor artifact.

The end-user flow is therefore:

```text
DIMER model.safetensors OR pinned upstream checkpoint
        ↓
mitra_classifier_colab.ipynb
        ↓
mitra-predictor.zip
        ↓
mitra_classifier_predictor_inference_colab.ipynb
        ↓
predictions.csv
```

## Runtime note

`autogluon.tabular[mitra]==1.5.0` requires a compatible PyTorch range and can replace the PyTorch version preinstalled by Google Colab. In the executed Tesla T4 run used during development, pip replaced PyTorch 2.11.0 with PyTorch 2.9.1. CUDA remained available and the Mitra workflow completed, but unrelated preinstalled packages reported resolver conflicts.

The notebooks therefore do **not** claim that PyTorch is left untouched. They print the installed PyTorch version, CUDA build, and CUDA availability immediately after installation. If PyTorch had already been imported and pip changes the installed version, the notebooks require a session restart before continuing.

For memory safety, the build/evaluate/export notebook keeps `MAX_MEMORY_USAGE_RATIO=1.10`, the setting that cleared AutoGluon's default training skip in the executed Colab run. AutoGluon may still suggest a larger value in a warning. Values above 1.0 deliberately accept more OOM risk, so the tutorial recommends reducing rows/context or using a higher-memory runtime before increasing the ratio simply to silence the warning.

## Bundled sample dataset

The default data source is [`freshretailnet-band-h7.zip`](../examples/sample-data/freshretailnet-band-h7.zip), a convenience sample derived from FreshRetailNet-50K and redistributed under **CC BY 4.0**.

Pinned sample revision: `8fc19e80ae3166ec6bf964d194a28c80e6ba3b1f`.

It contains:

| split | rows |
|---|---:|
| `train.csv` | 4,180 |
| `val.csv` | 1,600 |
| `test.csv` | 1,600 |

Each split has 17 features plus a three-class `target` (`low`, `mid`, `high`) representing the demand band seven days ahead. The notebook preserves the supplied split rather than randomly re-splitting it. See the [sample dataset card](../examples/sample-data/DATASET_CARD.md) for provenance, feature construction, the purged chronological split with embargo, and licence details.

The sample is intended for tutorial and smoke-test use, **not benchmarking**.

## BYOD split guidance

The single-CSV upload path uses a stratified random holdout and assumes rows are approximately IID. It should not be used blindly for time-dependent, panel, grouped, lagged, rolling-window, or other leakage-sensitive data.

For those cases, prepare leakage-aware partitions externally and use **Upload pre-split train/val/test**. The notebook preserves those partitions exactly, reorders validation/test features to the training order when names match, and verifies that evaluation labels are compatible with the classes learned from training.

## Model context carried into the tutorial

The notebook also mirrors the model-card guidance that matters for end users:

- Mitra is a Transformer-based tabular foundation model pretrained on approximately 45 million synthetic tabular datasets;
- this classifier checkpoint is approximately 75.7M parameters and supports numerical/categorical features, 2–10 classes, up to 500 features, and up to 10,000 training rows;
- its particularly strong reported regime is roughly 5,000 or fewer samples and 100 or fewer features;
- the published 85.8% mean accuracy is for the MITRA (+ef) benchmark configuration (ensembling + fine-tuning) across 137 datasets, not a guaranteed zero-shot accuracy for the downloaded checkpoint; and
- downstream evaluation, class-appropriate metrics, distribution-shift checks, subgroup analysis where relevant, and domain governance remain necessary before deployment.

## Supported model release

- Model: `autogluon/mitra-classifier`
- AutoGluon: `1.5.0`
- Revision: `c425e9fa0910a6be1c494321792e7ba2a1367b1a`
- Weights SHA-256: `e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19`
- Config SHA-256: `2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df`

The build/evaluate/export notebook refuses to run a checkpoint whose checksum does not match this release and refuses to continue if Hugging Face resolves outside the staged verified snapshot.

## Export provenance

`tutorial_run_metadata.json` records the actual AutoGluon, PyTorch, CUDA-build, and Python versions; checkpoint identity; row counts and row-cap status; requested fine-tuning steps/time limit; selected evaluation metric; memory-guard ratio; and holdout/independent-test metrics. A time limit can truncate the requested fine-tune schedule, so the metadata records that caveat rather than claiming an exact completed step count that AutoGluon does not expose here.

The artifact-inference notebook requires this metadata and validates its artifact format/version, immutable model identity/revision, pinned weight/config digests, feature schema, and AutoGluon runtime before deserializing the predictor.

## AI use and provenance

These tutorials were developed with substantial AI assistance using **GPT-5.6 Sol High** under human direction and review. The maintainer defined the goal, scope, model release, DIMER constraints, and acceptance criteria and remains responsible for repository changes and release decisions.

- AI model/configuration: **GPT-5.6 Sol High**
- Provider/client: **OpenAI / ChatGPT**
- Agent Relay role: **Builder**
- Base-model developer: **AutoGluon team, Amazon Web Services (AWS)**
- Distributed DIMER artifact: `model.safetensors`; canonical `config.json` is reconstructed locally (both pinned by SHA-256; no DIMER package manifest is currently supplied to this notebook)
- Model identity: the pinned revision and SHA-256 values above

AI attribution is **provenance, not sign-off**. It does not authenticate authorship, imply endorsement by OpenAI, AWS, AutoGluon, or DIMER, or independently verify correctness. Executed checks and reproducible outputs remain the evidence for a particular run, and users should review the notebooks and their results before consequential use.

### Production API parity

Both notebooks load `finetuner/pipeline_api.py` from immutable revision `9c53a1cc563e75f9a1e60a2a6beefb5e62757bfb` and verify SHA-256 `47b40a412ef9ff427e27632e0e32141cb1c004c946c1575a79c622113bd08490` before import. The DIMER fine-tuner delegates preprocessing, validation, split/cap, fit/evaluate, and prediction behavior to the same module.
