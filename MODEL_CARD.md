---
license: apache-2.0
model_card_spec: "1.0"
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-classifier
---

# Mitra Classifier

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--classifier-ffcc4d?style=flat)](https://huggingface.co/autogluon/mitra-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-autogluon%2Fautogluon-181717?style=flat&logo=github&logoColor=white)](https://github.com/autogluon/autogluon)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21204-b31b1b.svg)](https://arxiv.org/abs/2510.21204)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


###### Description

Mitra Classifier packages the `autogluon/mitra-classifier` checkpoint at Hugging Face revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, a pretrained tabular foundation model developed by the AutoGluon team at Amazon Web Services for supervised classification on structured datasets. The model is a Transformer specialised for tables: it applies row-wise and column-wise attention so that relationships across observations and across features are both represented, with 12 layers, a model dimension of 512, four attention heads, and approximately 75.7 million parameters according to the safetensors metadata. It was pretrained across roughly 45 million synthetically generated datasets drawn from structural causal models and tree-based priors (gradient boosting, random forests, decision trees, extra trees); the developers report that no real-world dataset was used directly in pretraining.

At inference the model conditions on the labelled training table as in-context support and emits a class-probability vector per query row; adaptation happens through in-context conditioning by default and, in this pipeline, through gradient fine-tuning when `fine_tune=true` (the DIMER default, `finetuner/train.py`). What this repository adds is the DIMER composition around those weights: a dataset validator (`validator/`), a fine-tuner that verifies the checkpoint digests before loading when the weights come from the Hub, enforces Mitra's row and class ceilings, seeds every RNG, evaluates on a held-out split, and writes the result, provenance, and context artifacts DIMER consumes (`finetuner/`), plus the contract documents and Colab tutorial. The upstream weights are not modified by this repository.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the code enforces.

###### Primary Intended Uses

Supervised classification of tabular data where each observation is one row of numerical and categorical predictor columns and one categorical target. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared target column and produces a fine-tuned or in-context Mitra predictor, per-row class labels and class probabilities, and holdout metrics.

Concrete application domains envisioned during development: binary and multiclass risk or category classification, demand-class and churn prediction, quality-grade classification, event classification, and scientific or research classification represented as feature tables — in particular the small-data regime, where the upstream authors report the model strongest (below roughly 5,000 samples and 100 features). The pipeline is meant to play the role of a strong zero-configuration baseline or a fine-tuned production model inside DIMER for tables that fit the ceilings: at most 10,000 training rows (`MITRA_ROW_LIMIT`; larger tables are class-preservingly sampled down), at most 500 features (`MITRA_FEATURE_LIMIT`, validator check `feature_limit`), and at most 10 target classes (`MITRA_CLASS_LIMIT`). Time-series, transactional, sensor, or panel data must first be represented as a supervised feature table; Mitra is not a forecasting model.

###### Primary Intended Users

Machine-learning researchers, data scientists, machine-learning engineers, and software developers working with structured datasets, and practitioners who want a pretrained foundation model for small-data tabular classification. The envisioned deployment setting is internal enterprise or research use through the DIMER platform — the fine-tuner and validator run as DIMER workers — not a public-facing service.

The pipeline assumes its users understand the provenance and semantics of their input data, the meaning of the target variable, the consequences of classification errors, and the limits of their own evaluation methodology: a user is expected to know that `predict()` is an argmax over uncalibrated class probabilities, that a holdout metric on a 50-row table has wide variance, and that a public benchmark table may overlap the tutorial data. A user who cannot tell a stratified holdout from an in-sample score is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** regression or continuous-value prediction (the sibling `mitra-regressor-pipeline` does that); image, video, audio, natural-language, or other unstructured inputs; unsupervised clustering; causal-effect estimation; generative modelling; raw time-series forecasting.
- **Input boundaries:** more than 10 target classes (validator refuses); more than 10,000 training rows (sampled down, never trained on in full); more than 500 features (validator refuses); fewer than 50 usable non-null-target rows (`MIN_TRAIN_ROWS`) or fewer than 2 rows in any class (`MIN_ROWS_PER_CLASS`); validation or test labels absent from training (refused); archives whose members exceed 1 GiB uncompressed, 5,000,000 rows, or a 200× compression ratio (refused as zip-bomb guards).
- **Decision boundaries:** autonomous high-impact decisions — health, safety, criminal justice, credit, employment, housing — without application-specific validation and a human decision-maker; treating published benchmark accuracy as a guarantee on a new dataset.

#### Factors

Mitra's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: Mitra was pretrained on synthetic datasets rather than any fixed human population, so no demographic group — age, sex, gender, ethnicity, nationality, socioeconomic status, disability — is an intrinsic development group of the foundation model, and the pretraining corpus is not group-audited because it contains no people. General demographic fairness or subgroup parity has therefore **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group accuracy, log loss, and ROC-AUC on the holdout split, and check for disparate error rates before deployment. The pipeline's provenance artifact records the class distribution, not any demographic one.

###### Instrumentation

Mitra consumes an abstract tabular representation rather than a raw sensor stream; the upstream pretraining did not depend on cameras, microphones, assays, or any real acquisition hardware. The instrument does not disappear because a table sits between it and the model: the operator's training and evaluation rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, survey instruments, sensors — and their characteristics (sampling rate, resolution, calibration, encoding of missing values) determine feature quality.

Instrument error reaches the model as feature error. Drift, miscalibration, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks schema, row counts, class counts, and label consistency, not whether a column's meaning has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** The fine-tuner runs in the DIMER container on the `pytorch:2.8.0-cuda12.8` base image with AutoGluon 1.5.0; training expects a CUDA device, and the torch build is pinned by the image so that sm_120 (RTX 50-series) hosts keep the cu128 wheel. Precision follows AutoGluon's Mitra defaults. The validator is CPU-only. Fine-tuning under AutoGluon 1.5.0 is seeded (`_seed_everything`) but not guaranteed bit-deterministic.

**Data environment.** The reported behaviour assumes the inference rows are drawn from the same distribution as the training table: same feature semantics, same encoding, same class prevalence. Performance degrades, without warning from the pipeline, under geographic, institutional, temporal, or population shift, and with the technical factors that dominate tabular performance — number of observations, number and quality of features, predictive signal, missing or erroneous values, label quality, class count and imbalance, categorical cardinality, preprocessing, leakage, and the fine-tuning configuration. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a probabilistic multiclass classifier whose intended use spans balanced and imbalanced tables.

###### Performance Measures

The fine-tuner evaluates the trained predictor on the held-out split with AutoGluon's `predictor.evaluate(..., auxiliary_metrics=True)` and writes every returned metric under `metrics.valEvaluation` in `result.json`, with `log_loss` sign-flipped to its conventional lower-is-better form. The headline metric is the DIMER hyperparameter `eval_metric` (default `accuracy`; `log_loss` and `roc_auc` map to Mitra-native early-stopping metrics, other AutoGluon metric names are reported but do not steer early stopping), recorded as `headlineMetric`/`headlineScore`.

Why these: accuracy captures discrete correctness and is the right summary when classes are reasonably balanced and error costs are similar; log loss captures probability quality and penalises confident mistakes, which matters whenever the class probabilities are used operationally; ROC-AUC captures ranking quality independent of any threshold and is the informative one for imbalanced binary problems. Reading only accuracy hides both calibration and imbalance failures, which is why all three are written even when only one is the headline. Upstream, the Mitra paper reports mean accuracy 0.858 ± 0.143 and AUC 0.905 ± 0.124 for its `+ef` configuration across 137 datasets; that is a published aggregate for a different configuration, not a number this pipeline measures.

###### Decision thresholds

The default decision rule is an implicit `argmax`: AutoGluon's `predict()` returns the class with the highest predicted probability, and this pipeline ships that rule unchanged. No acceptance threshold on accuracy, log loss, or ROC-AUC was set during development, because the pipeline is domain-agnostic and the tolerable error rate is a property of the deployment; published benchmark results are explicitly not production acceptance thresholds.

No probability cutoff is applied, and none is shipped, because the emitted probabilities are not calibrated for the operator's domain (see next section). Calibrating and thresholding are the deployment owner's responsibility: set the operating point from the asymmetric cost of false positives against false negatives and the class prevalence on held-out data, and revisit it when either changes. For a screening use where a missed positive is the expensive error, the threshold on the positive-class probability belongs below 0.5; for a use where a false alarm is expensive, above it.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single stratified holdout split of the operator's table (requested size `validation_split`, effective size recorded as `effectiveValidationSplit`), optionally capped at `DIMER_MAX_EVAL_ROWS` (default 50,000) rows. No dispersion is reported alongside the point value: one split, one run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the class-preserving down-sampling when the table exceeds 10,000 rows, the holdout split, and gradient fine-tuning; all three are driven by the DIMER `seed` hyperparameter, which the fine-tuner propagates to Python, NumPy, and torch (`_seed_everything`). A fixed seed nonetheless does not guarantee bit-identical fine-tuning under AutoGluon 1.5.0 because of non-deterministic CUDA kernels. The class probabilities the predictor emits are raw softmax outputs and have not been calibrated; a caller who needs calibrated probabilities must fit a calibrator (Platt or isotonic) on their own holdout data. Upstream's ± 0.143 accuracy spread across 137 datasets is a between-dataset dispersion, not an estimate of this pipeline's variance on any one table.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

Mitra was pretrained exclusively on synthetic datasets, so the pretraining data do not consist of personally identifiable information, health, biometric, financial, or classified records — this is known from the upstream disclosure, which ends at the description of the synthetic priors; the generated tables themselves are not published. What this repository distributes: the DIMER worker code, contract documents, a tutorial, and small sample datasets built by `examples/build_freshretailnet_dataset.py`; it does **not** distribute the checkpoint (the fine-tuner fetches it from the Hub and verifies the SHA-256 of `model.safetensors` and `config.json` against the pinned digests before loading, and `weights/` is gitignored).

Operators may fine-tune or evaluate Mitra on sensitive real-world tables. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure, not content — so the legality, privacy, consent, access control, and governance of downstream data remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract testing in `scripts/` and the DIMER holdout evaluation on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body, and general benchmark performance is not evidence of suitability.

Where such a use is foreseeable — a triage classifier built on a clinical feature table, for example — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in this repository, each inspectable in the named code:

- **Supply-chain integrity:** the base model is `autogluon/mitra-classifier`, expected at revision `c425e9fa…`. Because AutoGluon 1.5.0's Mitra loader calls `hf_hub_download` without a revision argument, the enforceable guarantee is a digest check, not a revision pin: `resolve_and_verify_weights` in `finetuner/train.py` computes SHA-256 over the resolved `model.safetensors` and `config.json` and raises when either differs from `EXPECTED_WEIGHTS_SHA256` / `EXPECTED_CONFIG_SHA256`, and records the resolved commit and both digests in provenance with `enforced: true`. Weights uploaded through DIMER (`model_dir` set) are used verbatim and recorded with `enforced: false` — that path is deliberately not checked against the public digest, and the provenance says so. The torch/CUDA build is asserted by the Dockerfile.
- **Input integrity:** the validator resolves `train`/`val`/`test` deterministically and rejects ambiguous archives, oversized members (> 1 GiB), tables over 5,000,000 rows, zip-bomb ratios (> 200×), fewer than 50 usable rows, any class with fewer than 2 rows, more than 10 classes, and validation/test labels absent from training.
- **Statistical mitigations:** tables over 10,000 rows are sampled down with `_stratified_cap`, which guarantees every class survives; a requested split that would empty a class is raised as an error rather than crashing in training.
- **Reproducibility:** `seed` propagates to Python, NumPy, and torch; the result artifact records the base revision, weight and config digests, AutoGluon version, effective split, and effective row counts.
- **Refusals:** `classNames` is written on every result payload, success or failure, because DIMER requires it; the fine-tuner trains a single Mitra model with `fit_weighted_ensemble=False` and asserts that the requested model actually trained, so no silent fallback to another AutoGluon learner can occur.

###### Risks and harms

- **Overconfidence outside the training distribution** (model-intrinsic): the probabilities are uncalibrated and carry no out-of-distribution signal; the harm falls on whoever the operator's decision affects, realised whenever inference rows drift from the training table, likely under normal use over time, with magnitude set by what the classification gates.
- **Amplification of input bias** (model-intrinsic): a table whose labels encode a historical disparity yields a classifier that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup evaluation.
- **Small-sample variance** (model-intrinsic): a holdout metric on a few hundred rows can move by several points between seeds; borne by the operator who ships on one lucky split.
- **Automation bias** (use-context): a numerically precise probability displaces human judgement; borne by the data subject; likely in any workflow that surfaces the score without the uncertainty.
- **Undetected leakage** (use-context): a feature derived from the target inflates the holdout score and collapses in production; the validator does not detect it; borne by the operator and downstream users.
- **Benchmark over-generalisation** (use-context): reading the upstream 85.8 % aggregate as an expected accuracy; borne by whoever sets expectations from it.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible label:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including classification on a target that proxies a protected attribute;
- deceptive, manipulative, or predatory applications, including presenting an uncalibrated class probability as a certified risk estimate;
- criminal-justice, medical-diagnosis, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use that violates the Apache-2.0 terms of the upstream `autogluon/mitra-classifier` weights and AutoGluon code, or the terms of the DIMER deployment.

---

## Model Details

**Model name:** Mitra Classifier

**Model identifier:** `autogluon/mitra-classifier`

**Code repository:** [autogluon/autogluon](https://github.com/autogluon/autogluon)

**Hugging Face repository:** [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier)

**Developer:** AutoGluon team, Amazon Web Services (AWS)

**Model family:** Tabular Foundation Model

**Task:** Tabular Classification

**Supported problem types:** Binary and multiclass classification

**Architecture:** Transformer with row-wise and column-wise attention

**Transformer layers:** 12

**Model / embedding dimension:** 512

**Attention heads:** 4

**Maximum classifier output dimension:** 10

**Approximate parameter count:** 75.7 million according to Hugging Face safetensors metadata

**Pretraining:** Approximately 45 million synthetic datasets

**Pretraining compute:** Eight NVIDIA A100 GPUs for approximately 60 hours

**Real-world pretraining data:** None reported

**License:** Apache License 2.0

## Checkpoint and Artifact Provenance

This card documents the following upstream Mitra Classifier checkpoint:

**Hugging Face repository:** `autogluon/mitra-classifier`

**Pinned revision:**

```text
c425e9fa0910a6be1c494321792e7ba2a1367b1a
```

The checkpoint consists of both model weights and architecture configuration.

### model.safetensors

**Size:** 302,717,904 bytes

**SHA-256:**

```text
e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19
```

### config.json

**Size:** 86 bytes

**SHA-256:**

```text
2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df
```

The exact raw content of `config.json` (86 bytes, single line without trailing newline) is:

```json
{"dim": 512, "dim_output": 10, "n_layers": 12, "n_heads": 4, "task": "CLASSIFICATION"}
```

> **DIMER Model Hosting Note:** When retrieving Mitra model weights from DIMER, only the `model.safetensors` binary is hosted. Because AutoGluon requires `config.json` in the same directory to initialize the Transformer architecture, you can recreate `weights/config.json` alongside `model.safetensors` using the exact text above:
> 
> ```bash
> printf '{"dim": 512, "dim_output": 10, "n_layers": 12, "n_heads": 4, "task": "CLASSIFICATION"}' > weights/config.json
> ```
> Or in PowerShell:
> ```powershell
> [System.IO.File]::WriteAllText("weights\config.json", '{"dim": 512, "dim_output": 10, "n_layers": 12, "n_heads": 4, "task": "CLASSIFICATION"}', [System.Text.Encoding]::ASCII)
> ```

These configuration parameters define the architecture into which the serialized model weights are loaded.

The `model.safetensors` file should therefore not be considered fully self-describing in isolation. The architecture configuration is part of the model-version definition and should be preserved together with the weights when reproducing this checkpoint.

A change to `config.json` could alter how otherwise identical weight bytes are interpreted. For reproducible use of this model version, both the model weights and configuration should be verified.

## Input

Mitra expects structured tabular data representing a supervised classification problem.

Each dataset conceptually contains:

- rows representing observations;
- feature columns containing numerical and/or categorical predictor variables; and
- a categorical target variable.

The classifier supports binary problems and multiclass problems with up to 10 classes.

Input dimensionality and dataset size must remain within Mitra's supported limits:

- maximum 10,000 training samples;
- maximum 500 features;
- maximum 10 classes.

## Output

Mitra Classifier produces categorical predictions for the target variable.

For each input observation, the model predicts one of the classes defined by the downstream classification problem.

Depending on the prediction interface, class-probability estimates may also be available.

The semantic meaning of the predicted classes is determined by the downstream dataset and is not fixed by the pretrained model.

## Model Architecture

Mitra Classifier uses a Transformer architecture designed for tabular data.

Its defining architecture configuration is:

```json
{
  "dim": 512,
  "dim_output": 10,
  "n_layers": 12,
  "n_heads": 4,
  "task": "CLASSIFICATION"
}
```

The architecture contains:

- 12 Transformer layers;
- a 512-dimensional internal representation;
- four attention heads;
- an output dimension of 10 for classification; and
- both row-wise and column-wise attention.

The output dimension represents the classifier architecture's maximum class capacity. A downstream classification task may use fewer than 10 classes.

Mitra's use of both row and column attention allows the model to model interactions among observations as well as relationships among features.

## Training Data

### Pretraining Dataset

Mitra was pretrained on approximately **45 million synthetically generated tabular datasets**.

The synthetic training distribution combines several families of priors, including:

- structural causal models;
- gradient boosting;
- random forests;
- decision trees; and
- extra trees.

The developers report that **no real-world datasets were directly used during pretraining**.

### Motivation

A central design principle of Mitra is that the mixture of synthetic priors used during pretraining strongly influences how effectively a tabular foundation model transfers to real-world datasets.

The prior mixture was designed around three principal considerations:

1. **Standalone performance** — whether a prior generates useful transferable behaviour.
2. **Diversity** — whether a prior contributes substantially different statistical structures.
3. **Distinctiveness** — whether a prior adds useful behaviour not already represented by other components of the mixture.

Synthetic generation enables Mitra to encounter a very large and diverse collection of tabular learning problems without requiring a correspondingly large corpus of real-world datasets.

### Pretraining Compute

Pretraining used approximately:

- **45 million synthetic datasets**
- **8 NVIDIA A100 GPUs**
- **approximately 60 hours of training**

## In-Context Learning and Fine-Tuning

Mitra is fundamentally an **in-context learning tabular foundation model**.

The pretrained model can use labelled examples from a previously unseen tabular task as context when predicting labels for new observations, without requiring conventional training from randomly initialized parameters.

Mitra additionally supports **fine-tuning**, in which the pretrained model weights are adapted to a particular downstream dataset.

Fine-tuning may provide additional performance gains depending on the dataset, task complexity, and available compute.

Fine-tuning should not be conflated with the base pretrained checkpoint. Any fine-tuned derivative represents an application-specific model version derived from the upstream Mitra Classifier.

## Evaluation Datasets

The Mitra paper evaluates the model across established collections of real-world tabular-learning benchmarks.

The principal classification evaluation includes datasets drawn from:

- **TabRepo**
- **TabZilla**
- **AutoML Benchmark (AMLB)**

After removing overlap among collections, the merged classification evaluation contains **137 unique datasets**.

The authors additionally report evaluation using **TabArena**.

These real-world datasets were used for evaluation rather than pretraining.

Using heterogeneous benchmark collections allows the model to be evaluated across differences in dataset size, feature dimensionality, numerical and categorical feature composition, number of classes, class balance, statistical structure, and application domain.

## Quantitative Evaluation

The Mitra paper reports results across a heterogeneous collection of tabular classification datasets rather than assigning one intrinsic accuracy value to the foundation model.

For the merged classification evaluation containing **137 unique datasets from TabRepo, TabZilla, and AMLB**, the strongest reported configuration, **MITRA (+ef)**, achieved:

- **Mean accuracy:** `0.858 ± 0.143`
- **Mean accuracy expressed as percentage:** **85.8%**
- **AUC:** `0.905 ± 0.124`
- **Average rank:** `7.2`
- **Elo rating:** `1136`
- **Win rate:** `0.69`

The `+ef` configuration combines:

- `+e` — ensembling; and
- `+f` — fine-tuning.

The 85.8% figure is therefore a **published aggregate benchmark result for this specific evaluation configuration**. It is not a guaranteed accuracy for the standalone pretrained checkpoint or for arbitrary downstream datasets.

The reported `± 0.143` variability in accuracy demonstrates substantial variation across datasets.

Published results should therefore be interpreted as evidence of strong general performance within the evaluated regime, not as a fixed operational accuracy.

## Reproducibility

### Checkpoint Pinning

The documented upstream checkpoint is pinned to revision:

```text
c425e9fa0910a6be1c494321792e7ba2a1367b1a
```

Reproduction should use both:

```text
model.safetensors
SHA-256:
e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19
```

and:

```text
config.json
SHA-256:
2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df
```

Because `config.json` defines the model architecture before the weights are loaded, matching only the weight file is insufficient to establish complete model-version identity.

### AutoGluon Loader Limitation

AutoGluon 1.5.0's Mitra loader resolves a checkpoint using its Hugging Face repository identifier but does not expose a revision argument for directly pinning the underlying Hugging Face revision during normal model loading.

For strict reproduction of this documented version, the exact resolved `model.safetensors` and `config.json` should therefore be verified against the revision and SHA-256 values recorded in this card.

### Random Seed Limitation

AutoGluon 1.5.0 does not fully enable Mitra's global `set_seed` behaviour.

A fixed seed can make some stochastic components, including internal validation splitting, reproducible, but it should **not be assumed to guarantee complete bit-for-bit deterministic fine-tuning**.

Where reproducibility is important, users should record software versions, random seeds, train/validation/test partitions, preprocessing, model and configuration hashes, and fine-tuning parameters, and should repeat experiments when estimating performance variability.

## Limitations

Important limitations include:

1. Mitra is primarily designed for relatively small tabular datasets.
2. It supports a maximum of approximately 10,000 training samples.
3. It supports a maximum of approximately 500 features.
4. The classifier supports a maximum of 10 classes.
5. Performance depends strongly on the information contained in the input features.
6. Strong benchmark results do not guarantee strong performance on a particular downstream dataset.
7. General demographic fairness has not been established.
8. Robustness to arbitrary distribution shift has not been established.
9. Domain-specific safety has not been established.
10. Suitability for high-impact applications cannot be inferred from general benchmark performance.
11. Downstream fine-tuning may introduce application-specific biases and failure modes.
12. A fixed random seed does not guarantee completely deterministic fine-tuning under AutoGluon 1.5.0.
13. Exact checkpoint reproduction requires preserving both `model.safetensors` and `config.json`.

## License

Mitra Classifier is distributed under the **Apache License 2.0**.

Apache-2.0 permits use, modification, redistribution, and hosted serving, including commercial use, subject to the license terms.

Redistributions should retain the applicable license and notices, and modifications should be documented as required by Apache-2.0.

The Apache-2.0 license text is distributed with the upstream model.

Licensing of downstream datasets and applications must be considered separately. The model's Apache-2.0 license does not override restrictions associated with downstream data.

## Model Ownership and Attribution

Mitra Classifier was developed by the AutoGluon team at Amazon Web Services (AWS). Upstream source code is part of the AutoGluon project hosted at [autogluon/autogluon](https://github.com/autogluon/autogluon), and base model artifacts are distributed on Hugging Face at [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier).

A downstream integration or fine-tuned derivative should distinguish the upstream foundation model from any subsequent modifications and preserve applicable license and attribution information.

## Citation

Cite the original Mitra work, the AutoGluon framework, and the upstream repository:

#### Papers

- **Mitra (2025):**  
  Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025. arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

- **AutoGluon-Tabular (2020):**  
  Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.* arXiv:2003.06505. https://doi.org/10.48550/arXiv.2003.06505

#### Upstream Repository

- **AutoGluon Codebase:**  
  AutoGluon team, Amazon Web Services (AWS). *AutoGluon: AutoML for Image, Text, and Tabular Data* [Software]. GitHub. https://github.com/autogluon/autogluon

#### BibTeX

```bibtex
@article{zhang2025mitra,
  title={Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models},
  author={Zhang, Xingjian and Maddix, Danielle C and Yin, Junwei and Erickson, Nick and Ansari, Abdul Fatir and Han, Boran and Zhang, Shenghao and Akoglu, Leman and Faloutsos, Christos and Mahoney, Michael and Hu, Tianpuxin and Rangwala, Huzefa and Karypis, George and Wang, Yuyang},
  journal={arXiv preprint arXiv:2510.21204},
  year={2025}
}

@article{erickson2020autogluon,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}

@misc{autogluon_repo,
  author = {Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander and others},
  title = {AutoGluon: AutoML for Image, Text, and Tabular Data},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/autogluon/autogluon}}
}
```

## Evaluation Status

### Established by the Upstream Work

The upstream work establishes tabular classification capability, binary and multiclass classification, in-context learning, fine-tuning capability, synthetic-prior pretraining, evaluation across established real-world tabular benchmark suites, strong performance within the evaluated small-data regime, and comparative performance against contemporary tabular foundation models and conventional approaches.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal accuracy on a particular downstream dataset, demographic fairness, subgroup parity, calibration, adversarial robustness, robustness to arbitrary distribution shift, domain-specific safety, operational reliability, service-level guarantees, or suitability for high-impact decision-making.

These properties must be evaluated for the particular downstream model and application.
