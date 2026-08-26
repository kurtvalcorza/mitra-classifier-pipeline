# Data Card

## Version Details

### Input

Mitra Classifier expects **structured tabular data** representing a supervised classification problem.

Each dataset consists conceptually of:

- **Rows:** individual observations or examples.
- **Feature columns:** predictor variables describing each observation.
- **Target column:** a categorical variable to be predicted.

Mitra supports numerical and categorical tabular features. The classifier supports **binary and multiclass classification with up to 10 target classes**.

The model is designed primarily for relatively small tabular datasets. The upstream implementation specifies the following limits:

- Maximum training samples: **10,000**
- Maximum features: **500**
- Maximum target classes: **10**

The model is reported to perform particularly well in the small-data regime, especially on datasets with approximately **5,000 or fewer samples and 100 or fewer features**.

Mitra is not intended to consume raw images, text, audio, video, or other unstructured data directly. Such information must first be transformed into an appropriate tabular feature representation.

### Output

Mitra Classifier produces **categorical classification predictions** for the target variable.

For each input observation, the model predicts one of the classes defined by the downstream classification task. Depending on the prediction interface, class-probability estimates may also be produced in addition to the predicted class.

The semantic meaning of the output classes is determined by the downstream dataset and is not fixed by the pretrained model.

### Type

Mitra Classifier is a **Transformer-based tabular foundation model** developed by the AutoGluon team at Amazon Web Services (AWS).

It uses an architecture specialized for structured data, including **row-wise and column-wise attention**, allowing the model to capture relationships across both observations and features.

The accompanying upstream `config.json` defines the classifier architecture as:

```json
{
  "dim": 512,
  "dim_output": 10,
  "n_layers": 12,
  "n_heads": 4,
  "task": "CLASSIFICATION"
}
```

The configuration specifies:

- `dim: 512` — internal model or embedding dimension.
- `dim_output: 10` — classifier output dimension, corresponding to a maximum supported capacity of 10 classes.
- `n_layers: 12` — number of Transformer layers.
- `n_heads: 4` — number of attention heads.
- `task: "CLASSIFICATION"` — identifies the checkpoint as the classification variant of Mitra.

The model contains approximately **76 million parameters**. Hugging Face safetensors metadata reports approximately **75.7 million parameters**, while some upstream descriptive materials round the architecture to approximately 72 million parameters.

Mitra is an **in-context learning tabular foundation model**. Instead of being pretrained for one fixed classification problem, it is pretrained across a very large collection of synthetically generated tabular learning problems so that it can generalize to previously unseen tabular datasets.

Mitra also supports **fine-tuning**, in which the pretrained weights are adapted to a particular downstream dataset.

### Paper or Other Resource for Information

Primary publication:

**Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models**

Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, and Bernie Wang.

NeurIPS 2025.

arXiv:2510.21204  
DOI: 10.48550/arXiv.2510.21204

Additional resources:

- Hugging Face model repository: `autogluon/mitra-classifier`
- AutoGluon documentation and source code
- Amazon Science materials describing Mitra
- Mitra research paper and associated experimental results

### Citation Details

Recommended citation:

Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* Advances in Neural Information Processing Systems (NeurIPS 2025). arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

### Other Relevant Information

**Developer:** AutoGluon team, Amazon Web Services (AWS)

**Model identifier:** `autogluon/mitra-classifier`

**Model family:** Tabular Foundation Model

**Task:** Tabular Classification

**Supported problem types:** Binary and multiclass classification

**License:** Apache License 2.0

**Pretraining data:** Approximately 45 million synthetic tabular datasets

**Pretraining compute:** Eight NVIDIA A100 GPUs for approximately 60 hours

**Real-world pretraining data:** None reported. Mitra was pretrained using synthetically generated tabular datasets.

**Published aggregate classification accuracy:** **85.8%**, reported as `0.858 ± 0.143`, for the **MITRA (+ef)** configuration across the paper's merged classification evaluation involving **137 unique datasets** from TabRepo, TabZilla, and AMLB.

The `+ef` configuration combines **ensembling (`+e`) and fine-tuning (`+f`)**. The reported 85.8% value is therefore an aggregate experimental result obtained under this evaluation configuration. It should not be interpreted as a guaranteed accuracy for every dataset or as the standalone zero-shot performance of the pretrained checkpoint.

Performance on a downstream dataset depends on factors such as the available predictive signal, sample size, number and quality of features, class balance, label quality, preprocessing, and distributional similarity between evaluation and deployment data.

#### Associated Configuration File

The upstream model distribution includes both:

- `model.safetensors` — the pretrained model weights.
- `config.json` — the architecture configuration required to interpret and instantiate those weights correctly.

The model version represented in this card is associated with the following configuration:

```json
{
  "dim": 512,
  "dim_output": 10,
  "n_layers": 12,
  "n_heads": 4,
  "task": "CLASSIFICATION"
}
```

The configuration should be treated as part of the model-version definition even when the model registry only accepts the `model.safetensors` artifact for upload.

The weights file is therefore **not fully self-describing in isolation**. Correct reconstruction of the Mitra Classifier architecture requires the associated configuration parameters recorded above.

# 1. Evaluation Datasets

## i. Dataset

The Mitra paper evaluates the model using several established collections of real-world tabular machine-learning benchmarks.

The principal classification evaluation combines datasets from:

- **TabRepo**
- **TabZilla**
- **AutoML Benchmark (AMLB)**

After accounting for datasets occurring in more than one benchmark collection, the merged classification evaluation contains **137 unique datasets**.

The paper additionally reports results using **TabArena**, another benchmark framework for evaluating tabular machine-learning systems.

These benchmark collections contain heterogeneous tabular classification tasks spanning different application domains, dataset sizes, feature characteristics, and class structures.

Importantly, these real-world datasets were used for **evaluation**, not for Mitra's synthetic pretraining.

## ii. Motivation

Multiple benchmark collections were used because performance on one or a few datasets would provide limited evidence about the generalization ability of a tabular foundation model.

Real-world tabular datasets can differ substantially in number of observations, number of features, numerical and categorical feature composition, class structure, class imbalance, missing-data patterns, statistical relationships, feature distributions, and application domain.

Using TabRepo, TabZilla, AMLB, and TabArena allows Mitra to be evaluated over a broad collection of established tabular-learning tasks and compared with other tabular foundation models and conventional machine-learning approaches.

The evaluation is particularly important because Mitra was pretrained only on synthetic datasets. Real-world benchmark performance tests whether the inductive biases learned from synthetic priors transfer to previously unseen real tabular data.

## iii. Preprocessing

The evaluation follows the tabular-processing and benchmark procedures described in the Mitra publication and associated AutoGluon implementation.

Individual datasets are converted into the representation required by Mitra while preserving the training and evaluation structure of the respective benchmark.

Mitra supports mixed tabular features, including numerical and categorical variables.

Exact preprocessing, splitting, and benchmark protocols differ across TabRepo, TabZilla, AMLB, and TabArena. The original Mitra paper and corresponding benchmark documentation should therefore be consulted when exact reproduction of a specific published result is required.

The benchmark data should not be confused with Mitra's pretraining corpus. Real-world benchmark datasets are used to measure transfer performance after synthetic pretraining.

# 2. Training Datasets

## i. Dataset

Mitra was pretrained on approximately **45 million synthetically generated tabular datasets** rather than on a fixed corpus of real-world datasets.

The synthetic pretraining mixture incorporates several families of priors, including:

- structural causal models (SCMs);
- gradient-boosting-based priors;
- random-forest-based priors;
- decision-tree-based priors; and
- extra-trees-based priors.

These mechanisms generate many different artificial supervised-learning problems containing relationships between features and targets.

The developers report that **no real-world datasets were directly used during Mitra pretraining**.

## ii. Motivation

A central design principle of Mitra is that the choice and mixture of synthetic priors substantially influence how effectively a tabular foundation model generalizes to real-world problems.

Synthetic generation allows the model to experience an extremely large number of diverse learning problems without requiring a correspondingly massive corpus of real tabular datasets.

The authors select and combine prior families according to three principal considerations:

1. **Standalone performance** — whether a prior produces transferable behaviour that performs well when evaluated on real tabular datasets.
2. **Diversity** — whether the prior adds substantially different types of statistical relationships to the pretraining distribution.
3. **Distinctiveness** — whether the prior contributes useful behaviour not already represented by the other priors in the mixture.

Combining structural causal models with multiple tree-based priors exposes the model to different classes of functional relationships and inductive biases.

The objective is to train a general-purpose tabular model capable of rapidly adapting to previously unseen classification tasks.

## iii. Preprocessing

Because Mitra's pretraining data are synthetically generated, preprocessing is integrated into the data-generation and pretraining procedure rather than consisting of conventional cleaning of a fixed real-world dataset.

Synthetic datasets are generated according to the selected prior families and transformed into the structured representation consumed by the Transformer.

Each generated learning problem contains observations, features, and target relationships from which Mitra learns general-purpose tabular prediction behaviour.

Detailed prior distributions, synthetic-generation procedures, sampling strategies, and training methodology are documented in the Mitra research paper and accompanying implementation.

Pretraining used approximately **45 million synthetic datasets** and was conducted using **eight NVIDIA A100 GPUs for approximately 60 hours**.

# Quantitative Analyses

## Unitary Results

The Mitra paper evaluates performance across a large and heterogeneous collection of tabular datasets rather than defining one universal model-accuracy figure.

For the merged classification benchmark containing **137 unique datasets from TabRepo, TabZilla, and AMLB**, the strongest reported configuration, **MITRA (+ef)**, achieved the following aggregate results:

- **Accuracy:** `0.858 ± 0.143`, equivalent to a mean accuracy of **85.8%**
- **AUC:** `0.905 ± 0.124`
- **Average rank:** `7.2`
- **Elo rating:** `1136`
- **Win rate:** `0.69`

The 85.8% accuracy reported for this version should therefore be interpreted as the **mean result across a large collection of classification datasets**, not as performance on one individual dataset.

The reported variability of `± 0.143` also demonstrates that model accuracy differs substantially across datasets. Consequently, the average should not be interpreted as the accuracy expected for an arbitrary downstream problem.

The `+ef` designation identifies an experimental configuration incorporating both **ensembling** and **fine-tuning**. Other Mitra configurations, including the base or purely in-context model, have their own results and should not be conflated with the `+ef` value.

The paper compares Mitra with contemporary tabular foundation models including TabPFNv2 and TabICL, as well as established conventional tabular-learning approaches. The results provide evidence of strong general tabular performance and sample efficiency within the evaluated regime.

## Intersectional Results

Traditional intersectional analysis based on demographic attributes such as age, sex, gender, ethnicity, disability, or socioeconomic status is **not reported as a general property of the Mitra foundation model**.

Mitra is a domain-general tabular model, and its pretraining corpus consists of synthetic learning problems rather than a fixed population of human subjects.

The published benchmarks evaluate performance across heterogeneous datasets and dataset characteristics rather than across predefined human demographic intersections.

Accordingly, no general claim can be made that Mitra provides equal performance across demographic or protected groups.

When Mitra is used in a human-centred application, the downstream model should be separately evaluated across relevant groups and intersections of groups. These may include demographic, geographic, socioeconomic, institutional, temporal, or operational factors depending on the application.

Such application-specific fairness and subgroup results cannot be inferred from Mitra's aggregate foundation-model benchmark performance.

# Caveats and Recommendations

## Details

### Published Accuracy Is Not a Universal Accuracy

The **85.8% accuracy** associated with this model version is the published mean classification accuracy for the **MITRA (+ef)** configuration over the merged benchmark containing 137 unique classification datasets.

It does not mean that Mitra will achieve 85.8% accuracy on every dataset.

Performance on a particular downstream problem may be substantially higher or lower.

### The Reported Accuracy Corresponds to a Specific Evaluation Configuration

The `+ef` configuration combines:

- `+e` — ensembling; and
- `+f` — fine-tuning.

The mandatory single accuracy value associated with a model-version registry cannot fully represent this experimental context.

The 85.8% value should therefore always be interpreted together with this Data Card and should not be described as guaranteed standalone zero-shot checkpoint accuracy.

### Model Weights Require Architecture Metadata

The uploaded `model.safetensors` artifact contains the pretrained model parameters, while the accompanying upstream `config.json` describes the architecture required to interpret those parameters.

The associated configuration is:

```json
{
  "dim": 512,
  "dim_output": 10,
  "n_layers": 12,
  "n_heads": 4,
  "task": "CLASSIFICATION"
}
```

If a model registry cannot store `config.json` as a separate artifact, this configuration should remain preserved in model-version documentation.

A system attempting to reconstruct the model should use architecture settings consistent with this configuration.

### Downstream Evaluation Remains Necessary

Before operational deployment, Mitra should be evaluated on a test dataset representative of the intended application.

An appropriate evaluation dataset should be independent of training and fine-tuning data, represent the intended deployment population, contain sufficient observations from all relevant target classes, contain difficult and uncommon cases where practical, reflect realistic missingness, noise, and measurement error, represent relevant temporal, geographic, institutional, or operational conditions, permit subgroup analysis where predictions affect people, and avoid target, feature, or temporal leakage.

### Accuracy Alone Is Insufficient

Applications should select metrics based on the structure and consequences of the classification problem.

For imbalanced datasets, overall accuracy may obscure poor performance on minority classes.

Other useful metrics may include balanced accuracy, macro F1, Matthews correlation coefficient, class-specific precision, class-specific recall, ROC-AUC, PR-AUC, and log loss.

Where predicted probabilities are used operationally, probability calibration should also be evaluated.

### Intended Operating Regime

Mitra is primarily designed for **small tabular datasets**.

Its particularly strong reported regime is approximately:

- **5,000 or fewer samples**; and
- **100 or fewer features**.

Its supported upper limits are approximately:

- **10,000 training samples**;
- **500 features**; and
- **10 target classes**.

Problems substantially outside this regime may be better addressed using other tabular modelling approaches.

### Distribution Shift

Performance may deteriorate when deployment data differ materially from the data used for model adaptation or evaluation.

Potential shifts include temporal drift, geographic differences, demographic differences, institutional differences, changes in measurement systems, changes in business or operational processes, changes in class prevalence, and changes in feature relationships.

Relevant distribution shifts should be evaluated and monitored for downstream deployments.

### Fairness and Subgroup Performance

The published aggregate benchmark results do not establish demographic fairness.

Applications involving people should separately evaluate model performance across relevant groups and intersections of groups.

Aggregate accuracy should not be assumed to imply similar error rates across all populations.

### High-Impact Applications

Mitra is a general-purpose tabular foundation model.

It has not been established as specifically validated for autonomous decision-making in areas such as medicine or health care, criminal justice, employment, lending or credit, insurance, education admissions, public-benefit eligibility, legal rights, critical infrastructure, or other safety-critical or high-impact contexts.

Such applications require domain-specific validation, appropriate governance, risk assessment, and human oversight.

### Recommended Interpretation

The published evidence supports describing Mitra Classifier as a **high-performing general-purpose tabular foundation model within its evaluated small-data regime**.

The evidence does not support describing Mitra as having a guaranteed accuracy of 85.8%, uniformly outperforming alternatives on every dataset, universally calibrated, demographically fair by default, robust to arbitrary distribution shifts, or validated for arbitrary high-impact decision-making.

Selection of Mitra for a particular application should ultimately be based on evaluation against relevant alternatives using representative downstream data.
