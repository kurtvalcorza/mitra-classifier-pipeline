---
license: apache-2.0
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-classifier
---

# Mitra Classifier

## Description

Mitra Classifier is a pretrained tabular foundation model developed by the AutoGluon team at Amazon Web Services (AWS) for supervised classification on structured or tabular datasets.

The model predicts categorical targets from numerical and categorical input features and supports both binary and multiclass classification.

Mitra uses a Transformer architecture specialized for tabular data. It applies both row-wise and column-wise attention, allowing the model to represent relationships across observations and features. The classifier has 12 Transformer layers, a model dimension of 512, four attention heads, and approximately 75.7 million parameters according to Hugging Face safetensors metadata.

Unlike conventional tabular models trained directly on one application dataset, Mitra is pretrained across approximately 45 million synthetically generated tabular datasets. Its synthetic pretraining distribution combines structural causal models with several tree-based prior families, including gradient boosting, random forests, decision trees, and extra trees.

The developers report that no real-world datasets were used directly during pretraining.

Mitra operates as an in-context learning tabular foundation model and additionally supports fine-tuning on downstream datasets.

# Model Details

**Model name:** Mitra Classifier

**Model identifier:** `autogluon/mitra-classifier`

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

# Checkpoint and Artifact Provenance

This card documents the following upstream Mitra Classifier checkpoint:

**Hugging Face repository:** `autogluon/mitra-classifier`

**Pinned revision:**

```text
c425e9fa0910a6be1c494321792e7ba2a1367b1a
```

The checkpoint consists of both model weights and architecture configuration.

## model.safetensors

**Size:** 302,717,904 bytes

**SHA-256:**

```text
e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19
```

## config.json

**Size:** 86 bytes

**SHA-256:**

```text
2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df
```

The associated `config.json` contains:

```json
{
  "dim": 512,
  "dim_output": 10,
  "n_layers": 12,
  "n_heads": 4,
  "task": "CLASSIFICATION"
}
```

These configuration parameters define the architecture into which the serialized model weights are loaded.

The `model.safetensors` file should therefore not be considered fully self-describing in isolation. The architecture configuration is part of the model-version definition and should be preserved together with the weights when reproducing this checkpoint.

A change to `config.json` could alter how otherwise identical weight bytes are interpreted. For reproducible use of this model version, both the model weights and configuration should be verified.

# Intended Use and Limitations

## Primary Intended Uses

Mitra Classifier is intended for supervised classification of structured or tabular data where each observation can be represented as one row containing predictor variables and a categorical target.

Appropriate applications include:

- binary classification;
- multiclass classification with up to 10 classes;
- risk or category classification;
- demand classification;
- churn prediction;
- quality-grade classification;
- event classification;
- scientific or research classification tasks represented as structured feature tables; and
- other supervised classification problems involving relatively small tabular datasets.

Mitra is particularly targeted at the small-data regime. The model is reported to be strongest on datasets below approximately 5,000 samples and 100 features.

Its supported upper limits are:

- **10,000 training samples**
- **500 features**
- **10 target classes**

For time-series, transactional, sensor, or panel datasets, the source data must first be represented as an appropriate supervised feature table. Mitra is not itself a general-purpose time-series forecasting model.

## Primary Intended Users

Mitra Classifier is intended primarily for:

- machine-learning researchers;
- data scientists;
- machine-learning engineers;
- software developers;
- researchers working with structured datasets; and
- practitioners seeking a pretrained foundation model for small-data tabular classification.

Users should understand the provenance and semantics of their input data, the meaning of the target variable, the consequences of classification errors, and the limitations of their evaluation methodology.

## Out-of-Scope Use Cases

Mitra Classifier is not intended for:

- regression or continuous-value prediction; a separate Mitra regressor checkpoint is available;
- image, video, audio, natural-language, or other unstructured-data tasks;
- datasets containing more than 10 target classes;
- datasets exceeding the model's supported sample or feature limits;
- unsupervised clustering;
- causal-effect estimation;
- generative modelling;
- direct raw time-series forecasting; or
- autonomous high-impact decision-making without application-specific validation and appropriate oversight.

Published benchmark performance should not be interpreted as a guarantee of performance on a new dataset.

# Input

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

# Output

Mitra Classifier produces categorical predictions for the target variable.

For each input observation, the model predicts one of the classes defined by the downstream classification problem.

Depending on the prediction interface, class-probability estimates may also be available.

The semantic meaning of the predicted classes is determined by the downstream dataset and is not fixed by the pretrained model.

# Model Architecture

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

# Training Data

## Pretraining Dataset

Mitra was pretrained on approximately **45 million synthetically generated tabular datasets**.

The synthetic training distribution combines several families of priors, including:

- structural causal models;
- gradient boosting;
- random forests;
- decision trees; and
- extra trees.

The developers report that **no real-world datasets were directly used during pretraining**.

## Motivation

A central design principle of Mitra is that the mixture of synthetic priors used during pretraining strongly influences how effectively a tabular foundation model transfers to real-world datasets.

The prior mixture was designed around three principal considerations:

1. **Standalone performance** — whether a prior generates useful transferable behaviour.
2. **Diversity** — whether a prior contributes substantially different statistical structures.
3. **Distinctiveness** — whether a prior adds useful behaviour not already represented by other components of the mixture.

Synthetic generation enables Mitra to encounter a very large and diverse collection of tabular learning problems without requiring a correspondingly large corpus of real-world datasets.

## Pretraining Compute

Pretraining used approximately:

- **45 million synthetic datasets**
- **8 NVIDIA A100 GPUs**
- **approximately 60 hours of training**

# In-Context Learning and Fine-Tuning

Mitra is fundamentally an **in-context learning tabular foundation model**.

The pretrained model can use labelled examples from a previously unseen tabular task as context when predicting labels for new observations, without requiring conventional training from randomly initialized parameters.

Mitra additionally supports **fine-tuning**, in which the pretrained model weights are adapted to a particular downstream dataset.

Fine-tuning may provide additional performance gains depending on the dataset, task complexity, and available compute.

Fine-tuning should not be conflated with the base pretrained checkpoint. Any fine-tuned derivative represents an application-specific model version derived from the upstream Mitra Classifier.

# Evaluation Datasets

The Mitra paper evaluates the model across established collections of real-world tabular-learning benchmarks.

The principal classification evaluation includes datasets drawn from:

- **TabRepo**
- **TabZilla**
- **AutoML Benchmark (AMLB)**

After removing overlap among collections, the merged classification evaluation contains **137 unique datasets**.

The authors additionally report evaluation using **TabArena**.

These real-world datasets were used for evaluation rather than pretraining.

Using heterogeneous benchmark collections allows the model to be evaluated across differences in dataset size, feature dimensionality, numerical and categorical feature composition, number of classes, class balance, statistical structure, and application domain.

# Quantitative Evaluation

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

# Performance Measures

Appropriate downstream classification metrics may include accuracy, balanced accuracy, macro F1, precision, recall, Matthews correlation coefficient, ROC-AUC, PR-AUC, and log loss.

Metric selection should depend on the downstream application. Accuracy may be appropriate where classes are reasonably balanced and error consequences are similar. Balanced accuracy, macro F1, or MCC may be more informative for imbalanced classification problems. When predicted probabilities are used operationally, probability-sensitive metrics and calibration should also be evaluated.

# Decision Thresholds

Mitra does not define a universal minimum accuracy requirement, confidence cutoff, or probability threshold applicable to every classification problem.

Application-specific thresholds should be selected according to the consequences of false positives, consequences of false negatives, class prevalence, operational objectives, uncertainty requirements, and applicable governance or regulatory requirements.

Published benchmark results should not themselves be treated as production acceptance thresholds.

# Factors

## Groups

Mitra was pretrained using synthetic datasets rather than datasets representing a fixed human population.

No demographic groups such as age, sex, gender, ethnicity, nationality, socioeconomic status, or disability are therefore intrinsic development groups of the foundation model.

Where Mitra is used on human-related datasets, relevant groups and subgroup performance must be identified and evaluated for the particular downstream application.

General demographic fairness or subgroup parity has not been established for the foundation model.

## Instrumentation

Mitra consumes structured tabular features rather than raw signals from a specific physical instrument.

The upstream pretraining process therefore did not depend on cameras, microphones, medical devices, laboratory instrumentation, or other real-world acquisition hardware.

For downstream datasets derived from physical measurements, the instrumentation used to produce those features should be documented separately.

## Environment

Mitra was not developed for one physical environment.

Environmental conditions become relevant when they influence input variables or the statistical distribution of downstream data.

Performance across geography, climate, institutions, populations, operating conditions, or time periods must therefore be evaluated in the context of the specific downstream application.

## Technical Factors

Performance may be affected by number of training observations, number of features, feature quality, predictive signal, missing values, erroneous observations, target-label quality, number of classes, class imbalance, categorical-cardinality patterns, feature engineering, preprocessing, data leakage, random variation, fine-tuning configuration, and distribution shift.

# Reproducibility

## Checkpoint Pinning

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

## AutoGluon Loader Limitation

AutoGluon 1.5.0's Mitra loader resolves a checkpoint using its Hugging Face repository identifier but does not expose a revision argument for directly pinning the underlying Hugging Face revision during normal model loading.

For strict reproduction of this documented version, the exact resolved `model.safetensors` and `config.json` should therefore be verified against the revision and SHA-256 values recorded in this card.

## Random Seed Limitation

AutoGluon 1.5.0 does not fully enable Mitra's global `set_seed` behaviour.

A fixed seed can make some stochastic components, including internal validation splitting, reproducible, but it should **not be assumed to guarantee complete bit-for-bit deterministic fine-tuning**.

Where reproducibility is important, users should record software versions, random seeds, train/validation/test partitions, preprocessing, model and configuration hashes, and fine-tuning parameters, and should repeat experiments when estimating performance variability.

# Approaches to Uncertainty and Variability

Published aggregate benchmark results reflect performance variation across many heterogeneous datasets.

For downstream use, appropriate uncertainty assessment may include repeated experiments using multiple random seeds, confidence intervals, cross-validation where methodologically appropriate, independent holdout evaluation, temporal validation, external validation, subgroup analysis, and probability-calibration assessment.

Evaluation design should reflect the consequences of prediction errors and the characteristics of the intended deployment environment.

# Ethical Considerations and Biases

## Data

Mitra was pretrained exclusively on synthetic datasets rather than a corpus of real-world human records.

The pretraining data therefore do not directly consist of personally identifiable information, health records, biometric records, financial records, classified information, or other real-world sensitive data.

This does not remove privacy, fairness, or governance risks from downstream applications. Users may fine-tune or evaluate Mitra using sensitive real-world datasets.

Responsibility for the legality, privacy, security, provenance, consent, access controls, and governance of downstream data remains with the application developer and data owner.

## Human Life

Mitra is a general-purpose tabular foundation model.

It was not specifically developed or validated for autonomous decisions concerning health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, public benefits, or other high-impact matters affecting human welfare.

General benchmark performance is insufficient evidence of suitability for these applications.

## Mitigations

Appropriate downstream risk mitigations include dataset provenance checks, data-quality validation, separation of training and evaluation data, leakage prevention, comparison against meaningful baselines, subgroup evaluation, appropriate metrics for class imbalance, distribution-shift assessment, independent test-set evaluation, preservation of model provenance, human review where errors have material consequences, and post-deployment monitoring.

These are recommended downstream controls rather than claims that every mitigation was part of upstream Mitra development.

## Risks and Harms

Potential risks include incorrect classification, dataset bias, unequal subgroup performance, distribution shift, data leakage, class imbalance, automation bias, and benchmark overgeneralization.

The severity and likelihood of these risks depend on the downstream application and the consequences attached to prediction errors.

## Use Cases Requiring Particular Caution

Applications requiring substantial additional validation and governance include discriminatory profiling, unlawful surveillance or social scoring, criminal-justice decisions, medical diagnosis or treatment, employment decisions, lending or credit decisions, insurance eligibility, public-benefit allocation, legal-rights determinations, and other safety-critical or high-impact decision-making.

# Limitations

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

# License

Mitra Classifier is distributed under the **Apache License 2.0**.

Apache-2.0 permits use, modification, redistribution, and hosted serving, including commercial use, subject to the license terms.

Redistributions should retain the applicable license and notices, and modifications should be documented as required by Apache-2.0.

The Apache-2.0 license text is distributed with the upstream model.

Licensing of downstream datasets and applications must be considered separately. The model's Apache-2.0 license does not override restrictions associated with downstream data.

# Model Ownership and Attribution

Mitra Classifier was developed by the AutoGluon team at Amazon Web Services (AWS).

A downstream integration or fine-tuned derivative should distinguish the upstream foundation model from any subsequent modifications and preserve applicable license and attribution information.

# Citation

Cite the original Mitra work:

Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025. arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

# Evaluation Status

## Established by the Upstream Work

The upstream work establishes tabular classification capability, binary and multiclass classification, in-context learning, fine-tuning capability, synthetic-prior pretraining, evaluation across established real-world tabular benchmark suites, strong performance within the evaluated small-data regime, and comparative performance against contemporary tabular foundation models and conventional approaches.

## Application-Dependent or Not Generally Established

The upstream evidence does not establish universal accuracy on a particular downstream dataset, demographic fairness, subgroup parity, calibration, adversarial robustness, robustness to arbitrary distribution shift, domain-specific safety, operational reliability, service-level guarantees, or suitability for high-impact decision-making.

These properties must be evaluated for the particular downstream model and application.
