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

The declared primary runtime for both notebooks is **Google Colab with Python 3.12**. Generic Jupyter compatibility is not claimed because the workflows intentionally use Colab upload/download primitives.

For `MODEL_SOURCE = 'DIMER ZIP'`, Notebook Spec v1.0 expects a self-describing offline package containing exactly `dimer-model-manifest.json`, `model.safetensors`, and `config.json` at the archive root. The manifest must declare `manifest_version: 1.0`, model ID `autogluon/mitra-classifier`, immutable revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, and SHA-256 values for both model files. Legacy weights-only DIMER ZIPs are refused rather than completed from the network.

Both inference workflows reject duplicate CSV headers before pandas can rename them.
Quoted column names and UTF-8 files with a byte-order mark are supported.

| Notebook | Purpose |
|---|---|
| [`mitra_classifier_colab.ipynb`](mitra_classifier_colab.ipynb) | Acquire/verify Mitra, bring data, evaluate, optionally fine-tune, infer, and export `mitra-predictor.zip` |
| [`mitra_classifier_predictor_inference_colab.ipynb`](mitra_classifier_predictor_inference_colab.ipynb) | Reload an exported `mitra-predictor.zip`, validate a new CSV, run inference, and download `predictions.csv` |

### Build/evaluate/export

[![Open build/evaluate/export tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_colab.ipynb)

`mitra_classifier_colab.ipynb` is a standalone tutorial for the Mitra Classifier checkpoint distributed through the DIMER Model Repository.

It does **not** depend on DIMER Workbench, DIMER APIs, or the DIMER validator/fine-tuner workers. Users can download the model weights from DIMER and run the notebook independently in Google Colab. If the DIMER download is unavailable, the notebook can retrieve the exact pinned upstream checkpoint associated with the DIMER release.

The tutorial covers:

- DIMER ZIP upload or pinned-upstream checkpoint fallback;
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
