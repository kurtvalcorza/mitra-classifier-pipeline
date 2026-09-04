# Mitra Classifier standalone Colab tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-classifier-pipeline/blob/main/tutorials/mitra_classifier_colab.ipynb)

`mitra_classifier_colab.ipynb` is a standalone, bring-your-own-data tutorial for the Mitra
Classifier checkpoint distributed through the DIMER Model Repository.

It does **not** depend on DIMER Workbench, DIMER APIs, or the DIMER validator/fine-tuner workers.
Users can download the model weights from DIMER and run the notebook independently in Google
Colab. If the DIMER download is unavailable, the notebook retrieves the exact pinned upstream
checkpoint associated with the DIMER release.

The tutorial covers:

- DIMER ZIP upload or pinned-upstream checkpoint fallback;
- SHA-256 verification of `model.safetensors` and `config.json`;
- CSV dataset inspection and compatibility checks;
- stratified train/holdout splitting;
- pretrained/in-context Mitra evaluation;
- optional GPU fine-tuning;
- holdout metric comparison;
- inference on new CSV rows; and
- export of predictions, run metadata, and a reusable AutoGluon predictor.

## Supported model release

- Model: `autogluon/mitra-classifier`
- AutoGluon: `1.5.0`
- Revision: `c425e9fa0910a6be1c494321792e7ba2a1367b1a`
- Weights SHA-256: `e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19`
- Config SHA-256: `2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df`

The notebook refuses to run a checkpoint whose checksum does not match this release.
