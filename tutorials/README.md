# Mitra Classifier standalone Colab tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_colab.ipynb)

`mitra_classifier_colab.ipynb` is a standalone tutorial for the Mitra Classifier checkpoint distributed through the DIMER Model Repository.

It does **not** depend on DIMER Workbench, DIMER APIs, or the DIMER validator/fine-tuner workers. Users can download the model weights from DIMER and run the notebook independently in Google Colab. If the DIMER download is unavailable, the notebook can retrieve the exact pinned upstream checkpoint associated with the DIMER release.

The tutorial covers:

- DIMER ZIP upload or pinned-upstream checkpoint fallback;
- SHA-256 verification of `model.safetensors` and `config.json`;
- a bundled FreshRetailNet sample dataset for users who do not yet have their own CSV;
- preservation of the sample's provided `train.csv` / `val.csv` / `test.csv` splits;
- BYOD CSV inspection and stratified holdout splitting;
- pretrained/in-context Mitra evaluation;
- optional GPU fine-tuning;
- before/after metric comparison;
- inference on new CSV rows; and
- export of predictions, run metadata, and a reusable AutoGluon predictor.

## Bundled sample dataset

The default data source is [`freshretailnet-band-h7.zip`](../examples/sample-data/freshretailnet-band-h7.zip), a convenience sample derived from FreshRetailNet-50K and redistributed under **CC BY 4.0**.

It contains:

| split | rows |
|---|---:|
| `train.csv` | 4,180 |
| `val.csv` | 1,600 |
| `test.csv` | 1,600 |

Each split has 17 features plus a three-class `target` (`low`, `mid`, `high`) representing the demand band seven days ahead. The notebook preserves the supplied split rather than randomly re-splitting it. See the [sample dataset card](../examples/sample-data/DATASET_CARD.md) for provenance, feature construction, the purged chronological split with embargo, and licence details.

The sample is intended for tutorial and smoke-test use, **not benchmarking**.

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

The notebook refuses to run a checkpoint whose checksum does not match this release.

## AI use and provenance

This tutorial was developed with substantial AI assistance from **OpenAI ChatGPT** under human direction and review. The maintainer defined the goal, scope, model release, DIMER constraints, and acceptance criteria and remains responsible for repository changes and release decisions.

- Generated with AI assistance by: **OpenAI ChatGPT**
- Agent Relay role: **Builder**
- Base-model developer: **AutoGluon team, Amazon Web Services (AWS)**
- Distributed DIMER artifact: `model.safetensors`
- Model identity: the pinned revision and SHA-256 values above

AI attribution is **provenance, not sign-off**. It does not authenticate authorship, imply endorsement by OpenAI, AWS, AutoGluon, or DIMER, or independently verify correctness. Executed checks and reproducible outputs remain the evidence for a particular run, and users should review the notebook and its results before consequential use.
