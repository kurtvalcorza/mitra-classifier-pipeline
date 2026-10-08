"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded package module
(``mitra_pipeline/tutorial_api.py``, a root-level package: ``package_dir``), and the model pin/stage/verify cell
are produced by the generator from repository sources so they cannot drift from the package. The
ARTIFACT-INFERENCE companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "mitra-classifier-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/mitra_classifier_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--classifier-ffcc4d?style=flat",
        "https://huggingface.co/autogluon/mitra-classifier",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-autogluon%2Fautogluon-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/autogluon/autogluon",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2508.02927-b31b1b.svg", "https://arxiv.org/abs/2508.02927"),
]

TEMPLATE = {
    "package": "mitra_pipeline",
    "package_dir": "mitra_pipeline",  # root-level package (no src/)
    "repo_name": REPO,
    "stem": "mitra_classifier",
    "notebook_name": "mitra_classifier_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab-isolated.lock.txt`
    # (the pip-compile `requirements-colab*.lock.txt` files beside it are the pre-existing reference locks and are unchanged).
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab-isolated.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the pinned Mitra snapshot, loads scikit-learn's bundled Breast Cancer Wisconsin table (no download), validates the tables into an input manifest and checks class coverage and split overlap, fits the pretrained Mitra predictor by **in-context conditioning on the training split** (the adaptation stage that runs by default — no gradient update) alongside executable baselines, evaluates on the held-out split and writes the evaluation report, exports the deployable predictor bundle and reloads it from disk to prove the fresh boundary. Gradient fine-tuning of the Mitra weights is an optional experiment (`RUN_FINE_TUNING`, off by default, Section 6) because it needs a GPU-sized time budget. In-context conditioning on the support rows is this notebook's adaptation stage. No repository clone, DIMER worker or service, credential, upload dialog or configuration edit is required (§5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 (with an `Upload …` `DATA_SOURCE`, and `BYOD_PATH` pointing at one labelled CSV or at a directory holding pre-split `train.csv`/`val.csv`/`test.csv`; on Colab an empty path opens the upload dialog) and re-run from that cell; it enters the same validation, split, in-context fitting, baseline, evaluation, export and fresh-reload cells as the sample (DAT14), and `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows with the fitted predictor. Expected schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 4; uploads stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "MitraClassificationPipeline",
    "weights_key": "mitra-classifier",
    "modules": ["tutorial_api.py"],
    "entry_module": "tutorial_api.py",
    # `from_pretrained` stages + verifies the snapshot, then stages the verified bytes as the immutable offline
    # Hugging Face snapshot AutoGluon resolves (HF_HUB_OFFLINE). No predictor is built until `fit`.
    "model_load": "MitraClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "numpy", "pandas", "sklearn"],
    "title": "Mitra Classifier — DIMER E2E tabular classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "end-to-end tabular classification with the pinned `autogluon/mitra-classifier` checkpoint through AutoGluon: verified model acquisition, validated support data with class coverage, in-context evaluation against executable baselines, optional GPU fine-tuning, new-data inference with class probabilities, a deployable predictor bundle and its fresh-boundary reload",
    "intro": (
        "Mitra is an in-context tabular foundation model served through AutoGluon's `TabularPredictor`: with "
        "`fine_tune=False`, `fit` registers the support rows and the model configuration and no weight is "
        "gradient-updated; fine-tuning is an opt-in gate that needs a GPU. The upstream project supplies the model and "
        "the checkpoint; the carried package adds the pinned snapshot scheme, the table validation, class-coverage, "
        "stratified split, capping and overlap checks, the metric set, the `validate_inputs` / "
        "`majority_class_baseline` / `evaluation_report` helpers, and the archive-safety and artifact-manifest "
        "functions. `predict` applies an implicit `argmax` over class probabilities that are raw model outputs, **not "
        "calibrated probabilities**; the package ships no acceptance threshold. The default sample is scikit-learn's "
        "bundled Breast Cancer Wisconsin table; its metrics are tutorial sanity evidence, not a benchmark or "
        "production claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint and stage it as an offline Hugging Face snapshot, load a public sample or your own CSV(s) "
        "and validate them into an input manifest with class coverage preserved, evaluate the pretrained model on a "
        "stratified holdout and an independent test partition against the majority-class, LightGBM and Random Forest "
        "baselines with strict label alignment, optionally fine-tune on a GPU with holdout-based selection, write an "
        "evaluation report, optionally score new rows with class probabilities under an explicit `argmax` rule, export "
        "the deployable AutoGluon predictor bundle with its manifest and provenance, and prove it reloads from a fresh "
        "directory."
    ),
    "exclusions": (
        "regression, forecasting, calibrated probabilities, or any deployment threshold. `prediction` is the `argmax` "
        "over uncalibrated class probabilities; the fine-tuning path runs only on a GPU and only when "
        "`RUN_FINE_TUNING` is switched on; Mitra supports at most `MAX_CLASSES` classes."
    ),
    "about_details": (
        "**What you will do.** Condition a pretrained tabular foundation model on a small labelled table, compare it row by row with three baselines on the same partitions, and export a bundle that reloads from fresh files. **How:** Runtime → Run all (one pass, no restart, no upload); the declarations are in the collapsible block below."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic pandas, has used Colab or Jupyter and has met a train/holdout split and accuracy, and wants to see what an in-context tabular foundation model does with a small table: how it is *conditioned* on support rows instead of trained, how its numbers are read against trivial and classical baselines on the same rows, and what the exported predictor bundle contains. The audience is students and practitioners deciding whether Mitra fits their own tables; no prior experience with AutoGluon or Mitra is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough for the default path; the optional fine-tuning gate needs a GPU.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a labelled table (`DATA_SOURCE`): the default is scikit-learn's bundled Breast Cancer Wisconsin table (569 rows, 30 numeric features, two classes), split 60/20/20 into support, holdout and an independent test partition; or your own CSV (one file, or pre-split `train.csv` / `val.csv` / `test.csv`) via `BYOD_PATH` or the Colab upload dialog |\n| Model | the pinned `autogluon/mitra-classifier` checkpoint (302,717,904-byte `model.safetensors`) served through AutoGluon's `TabularPredictor`; `fit` with `fine_tune=False` registers the support rows — no weight is gradient-updated — and `RUN_FINE_TUNING` (off) is the only path that trains |\n| Output | accuracy, balanced accuracy, macro F1, MCC, log loss and ROC-AUC on the holdout and the test partition beside a majority-class predictor, LightGBM and Random Forest; an input manifest with one recorded refusal; an evaluation report with the verdict `sample-sanity`; eight scored rows; a predictor bundle (`outputs/mitra_classifier_predictor.zip`) that is reloaded from fresh files and checked against the in-memory model; `result.json` |\n\n**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried module and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. The answers give directions and magnitudes, not numbers to match; the few counts they quote (partition sizes, file sizes) are deterministic and are labelled with the run they come from — the Kaggle T4 run of 14 September 2026 recorded for the previous notebook revision (blob `9599d5c`). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about ten minutes; the AutoGluon install and the 300 MB checkpoint dominate.\n\n**Roadmap:** 1–3 infrastructure → 4 the table and its split *(core concept: support, holdout and an independent test partition)* → 5 validation, the input manifest and one deliberate refusal *(core concept: the data contract, class coverage and leakage checks)* → 6 the pretrained model beside a majority-class predictor, LightGBM and Random Forest on the same rows *(evaluation practice: baselines first; in-context conditioning versus training)* → 7 the evaluation report and its verdict *(evaluation practice)* → 8 optional new-data inference → 9 export the bundle and prove a fresh reload *(engineering)* → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic pandas and Colab or Jupyter familiarity; no prior experience with AutoGluon or Mitra. In-context conditioning, the partitions, the metrics, the baselines and the bundle are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (AutoGluon 1.5.0 and its torch), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU and uses CUDA automatically when available; the fine-tuning gate requires a GPU. The pinned `autogluon.tabular[mitra]==1.5.0` install (with its torch) is the largest download of the run.",
        "- **Knowledge:** basic pandas; what a stratified holdout, an independent test partition, accuracy, balanced accuracy, log loss and ROC-AUC are.",
        "- **Data:** the default sample is scikit-learn's bundled Breast Cancer Wisconsin table (569 rows, 30 numeric features, two classes), loaded from the installed package, so nothing is downloaded and no private data is needed; `Sample: Wine` is the bundled three-class table. BYOD (one labelled CSV, or pre-split `train.csv`/`val.csv`/`test.csv` — the repository's `examples/sample-data/` archives such as FreshRetailNet, Telco Churn and Adult Census can be supplied this way) is selected through `DATA_SOURCE` and is off by default so the sample path runs top-to-bottom without interaction. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the sample or your own data\n\n"
                "`Sample: Breast Cancer` (default) is the bundled binary sanity check, split 60/20/20 with stratification "
                "into support, holdout and an independent test partition; `Sample: Wine` is the bundled three-class "
                "table split the same way. `Upload CSV` takes one labelled CSV and carves a stratified holdout with "
                "`stratified_holdout` (`VALIDATION_SPLIT`, every class kept in both partitions); `Upload pre-split "
                "train/val/test` takes your own partitions and preserves them — use it for temporal, grouped or lagged "
                "data, where a random holdout would leak. `DROP_COLUMNS` names identifier or leakage columns to remove "
                "before validation. **BYOD checklist:** the inputs are CSV files, not archives — extract a ZIP first and point `BYOD_PATH` at the extracted CSV (or at the directory holding `train.csv`, `val.csv` and `test.csv`); set `TARGET_COLUMN` to your label column (the repository's `examples/sample-data` archives use `target` for FreshRetailNet, `Churn` for Telco Churn and `class` for Adult Census) — this cell checks it before anything else runs. `USE_BYOD` is a deliberate second switch: changing `DATA_SOURCE` alone never opens an upload dialog during **Run all**. Rows whose target is missing are **dropped and counted** (reported in the input "
                "manifest of Section 5, never hidden). The cell prints the sample kind, the partition sizes and the "
                "SHA-256 of the data payloads so the exported provenance can be tied to the exact data.\n\n"
                "**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent "
                "by this notebook to an external inference or training service; the only network request on the default "
                "path is the pinned checkpoint download of Section 3.\n\n"
                "**Predict:** 569 rows split 60/20/20 with stratification. How many rows land in each partition, and will the class shares of the holdout match the support rows' exactly, approximately, or not at all?"
            ),
            "code": (
                "import hashlib\n\n"
                "from sklearn.datasets import load_breast_cancer, load_wine\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "DATA_SOURCE = 'Sample: Breast Cancer'  # @param [\"Sample: Breast Cancer\", \"Sample: Wine\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "DROP_COLUMNS = ''  # @param {{type:\"string\"}}\n"
                "VALIDATION_SPLIT = 0.20  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "if USE_BYOD and DATA_SOURCE.startswith('Sample'):\n"
                "    raise ValueError('USE_BYOD=True requires an Upload DATA_SOURCE.')\n"
                "if DATA_SOURCE.startswith('Upload') and not USE_BYOD:\n"
                "    raise ValueError('Set USE_BYOD=True to use an upload DATA_SOURCE.')\n"
                "drop_columns = [c.strip() for c in DROP_COLUMNS.split(',') if c.strip() and c.strip() != TARGET_COLUMN]\n\n"
                "def stratified_60_20_20(frame):\n"
                "    train, remainder = train_test_split(frame, test_size=0.4, random_state=SEED, stratify=frame[TARGET_COLUMN])\n"
                "    holdout, test = train_test_split(remainder, test_size=0.5, random_state=SEED, stratify=remainder[TARGET_COLUMN])\n"
                "    return train, holdout, test\n\n"
                'def byod_payloads(path, expected=None):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter): one labelled CSV, or a directory holding the expected files; on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if expected:\n'
                '            if not source.is_dir():\n'
                "                raise FileNotFoundError(f'BYOD_PATH {{str(source)!r}} must be a directory holding {{list(expected)}} for the pre-split option (relative paths start at {{os.getcwd()}}).')\n"
                '            missing = sorted(name for name in expected if not (source / name).is_file())\n'
                '            if missing:\n'
                "                raise FileNotFoundError(f'BYOD directory {{str(source)!r}} is missing {{missing}}.')\n"
                '            return {{name: (source / name).read_bytes() for name in expected}}\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD_PATH {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one labelled CSV.')\n"
                "        if source.name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{source.name}} is an archive: extract it and set BYOD_PATH to the extracted CSV, or to the directory holding train.csv, val.csv and test.csv.')\n"
                "        if not source.name.lower().endswith('.csv'):\n"
                "            raise ValueError(f'{{source.name}}: expected a labelled CSV file.')\n"
                '        return {{source.name: source.read_bytes()}}\n'
                '    try:\n'
                '        from google.colab import files\n'
                '    except ImportError:\n'
                "        raise RuntimeError('USE_BYOD is on but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: copy the CSV (or the pre-split directory) into this runtime, or attach it as a Kaggle dataset, and set BYOD_PATH.') from None\n"
                '    uploaded = files.upload()\n'
                "    archives = sorted(name for name in uploaded if name.lower().endswith('.zip'))\n"
                '    if archives:\n'
                "        raise RuntimeError(f'{{archives}}: archives are not read here. Extract the archive and upload its CSV file(s) instead (for the pre-split option: train.csv, val.csv and test.csv together).')\n"
                '    if expected:\n'
                '        by_base = {{Path(name).name.lower(): payload for name, payload in uploaded.items()}}\n'
                '        missing = sorted(set(expected) - set(by_base))\n'
                '        if missing:\n'
                '            raise RuntimeError(f\'Upload {{", ".join(expected)}} together (received {{sorted(by_base) or "nothing; a cancelled dialog sends none"}}). Missing: {{missing}}. Run this cell again.\')\n'
                '        return {{name: by_base[name] for name in expected}}\n'
                "    csvs = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.csv')]\n"
                '    if len(csvs) != 1:\n'
                "        raise RuntimeError(f'Upload exactly one labelled CSV (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '    return dict(csvs)\n'
                '\n'
                "def require_target(frame, name):\n"
                "    if TARGET_COLUMN not in frame.columns:\n"
                "        raise ValueError(f'TARGET_COLUMN {{TARGET_COLUMN!r}} is not a column of {{name}}. Set TARGET_COLUMN in this cell (Section 4) to the label column; available columns: {{list(frame.columns)}}')\n"
                "    return frame\n\n"
                "test_data = None\n"
                "if DATA_SOURCE == 'Sample: Breast Cancer':\n"
                "    dataset = load_breast_cancer(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({{0: 'malignant', 1: 'benign'}})\n"
                "    train_data, holdout_data, test_data = stratified_60_20_20(frame)\n"
                "    payloads = {{'sample.csv': frame.to_csv(index=False).encode('utf-8')}}\n"
                "    data_name, sample_kind = 'sklearn-breast-cancer', 'sample'\n"
                "elif DATA_SOURCE == 'Sample: Wine':\n"
                "    dataset = load_wine(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map(dict(enumerate(dataset.target_names)))\n"
                "    train_data, holdout_data, test_data = stratified_60_20_20(frame)\n"
                "    payloads = {{'sample.csv': frame.to_csv(index=False).encode('utf-8')}}\n"
                "    data_name, sample_kind = 'sklearn-wine', 'sample'\n"
                "elif DATA_SOURCE == 'Upload pre-split train/val/test':\n"
                "    payloads = byod_payloads(BYOD_PATH, ('train.csv', 'val.csv', 'test.csv'))\n"
                "    train_data = require_target(read_csv_bytes(payloads['train.csv'], 'train.csv'), 'train.csv')\n"
                "    holdout_data = require_target(read_csv_bytes(payloads['val.csv'], 'val.csv'), 'val.csv')\n"
                "    test_data = require_target(read_csv_bytes(payloads['test.csv'], 'test.csv'), 'test.csv')\n"
                "    data_name, sample_kind = 'pre-split upload', 'BYOD'\n"
                "else:\n"
                "    payloads = byod_payloads(BYOD_PATH)\n"
                "    data_name, payload = next(iter(payloads.items()))\n"
                "    data, _features, _report = validate_labeled_frame(require_target(read_csv_bytes(payload, data_name), data_name), TARGET_COLUMN, name=data_name, drop_columns=drop_columns, min_rows=MIN_TRAIN_ROWS, min_classes=MIN_CLASSES, min_class_count=MIN_CLASS_COUNT)\n"
                "    if not 0.05 <= VALIDATION_SPLIT <= 0.40:\n"
                "        raise ValueError('VALIDATION_SPLIT must be between 0.05 and 0.40.')\n"
                "    train_data, holdout_data = stratified_holdout(data, TARGET_COLUMN, VALIDATION_SPLIT, SEED)\n"
                "    sample_kind = 'BYOD'\n"
                "    print('Upload CSV uses a seeded stratified random holdout and assumes approximately IID rows; use the pre-split option for temporal/grouped data.')\n"
                "DATA_DIGEST = hashlib.sha256(json.dumps({{name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(payloads.items())}}, sort_keys=True).encode()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': data_name, 'target': TARGET_COLUMN, 'drop_columns': drop_columns, 'train_rows': len(train_data), 'holdout_rows': len(holdout_data), 'test_rows': 0 if test_data is None else len(test_data), 'data_sha256': DATA_DIGEST}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>341 support, 114 holdout and 114 test rows (60/20/20 of 569; the Kaggle T4 run recorded for the previous revision, blob `9599d5c`, printed the same sizes). Stratification keeps the class shares approximately equal across partitions (the table is about 63 % benign / 37 % malignant), not exactly: 114 rows cannot split 62.7 / 37.3 to the row. The data SHA-256 printed here is what ties the exported provenance to this exact table.</details>'
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then check class coverage, overlaps and the cap\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`validate_labeled_frame` applies — unique column names, the target present, rows with a missing target "
                "dropped and counted, at least `MIN_TRAIN_ROWS` support rows (2 for a holdout), at most `MAX_FEATURES` "
                "features, between `MIN_CLASSES` and `MAX_CLASSES` classes (Mitra's ceiling), at least "
                "`MIN_CLASS_COUNT` support rows per class — and returns an **input manifest** naming the schema, the "
                "observed structure (categorical columns, missing values, the validation report with dropped and "
                "exact-duplicate counts, the class list, class counts and the majority-class share) and the verdict. It "
                "is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a probe with too few rows and records the package's own error message as a finding. The "
                "ceilings and the decision rule are printed before any model runs.\n\n"
                "The holdout and test partitions are then re-ordered to the support schema, `require_class_coverage` "
                "refuses a partition that misses a trained class or carries an unseen one (an in-context classifier can "
                "only predict classes present in its support rows), exact cross-partition overlaps are reported "
                "(`split_overlap_report`), support rows above `MAX_TRAIN_ROWS` are capped with the class-preserving "
                "`cap_training_rows`, and the trivial baseline is computed with `majority_class_baseline` (always predict "
                "the most frequent support class). Everything in Section 6 should be read against that baseline.\n\n"
                "**Predict:** the cell validates a probe with `MIN_TRAIN_ROWS - 1` rows. Which rule refuses it, and does the refusal stop the notebook? And what accuracy will the majority-class baseline score on a stratified holdout of a 63 / 37 table?"
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_TRAIN_ROWS': MIN_TRAIN_ROWS, 'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES, 'MIN_CLASSES': MIN_CLASSES, 'MAX_CLASSES': MAX_CLASSES, 'MIN_CLASS_COUNT': MIN_CLASS_COUNT}}, 'decision_rule': DECISION_RULE}})\n"
                "input_manifest = validate_inputs(train_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, names=[data_name + ':train'])\n"
                "input_manifest['inputs'].extend(validate_inputs(holdout_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, min_rows=2, min_classes=1, min_class_count=1, names=[data_name + ':holdout'])['inputs'])\n"
                "if test_data is not None:\n"
                "    input_manifest['inputs'].extend(validate_inputs(test_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, min_rows=2, min_classes=1, min_class_count=1, names=[data_name + ':test'])['inputs'])\n"
                "# Demonstrate rejection on a probe that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(train_data.head(MIN_TRAIN_ROWS - 1), target_column=TARGET_COLUMN, drop_columns=drop_columns)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'too-few-rows-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2))\n"
                "print('findings:', input_manifest['findings'])\n\n"
                "train_data, FEATURE_COLUMNS, train_report = validate_labeled_frame(train_data, TARGET_COLUMN, name='train', drop_columns=drop_columns, min_rows=MIN_TRAIN_ROWS, min_classes=MIN_CLASSES, min_class_count=MIN_CLASS_COUNT)\n"
                "holdout_data, val_features, _ = validate_labeled_frame(holdout_data, TARGET_COLUMN, name='holdout', drop_columns=drop_columns)\n"
                "ordered = FEATURE_COLUMNS + [TARGET_COLUMN]\n"
                "if set(val_features) != set(FEATURE_COLUMNS):\n"
                "    raise ValueError('train/holdout feature column names do not match.')\n"
                "holdout_data = holdout_data.reindex(columns=ordered)\n"
                "require_class_coverage(train_data, holdout_data, TARGET_COLUMN, 'holdout')\n"
                "if test_data is not None:\n"
                "    test_data, test_features, _ = validate_labeled_frame(test_data, TARGET_COLUMN, name='test', drop_columns=drop_columns)\n"
                "    if set(test_features) != set(FEATURE_COLUMNS):\n"
                "        raise ValueError('train/test feature column names do not match.')\n"
                "    test_data = test_data.reindex(columns=ordered)\n"
                "    require_class_coverage(train_data, test_data, TARGET_COLUMN, 'test')\n"
                "overlaps = split_overlap_report({{'train': train_data[ordered], 'holdout': holdout_data, **({{'test': test_data}} if test_data is not None else {{}})}})\n"
                "if any(overlaps.values()):\n"
                "    print('WARNING exact cross-split overlap detected; investigate leakage before interpreting metrics:', overlaps)\n"
                "train_data, cap_report = cap_training_rows(train_data, TARGET_COLUMN, seed=SEED)\n"
                "CLASSES = sorted(train_data[TARGET_COLUMN].unique(), key=str)\n"
                "PROBLEM_TYPE = 'binary' if len(CLASSES) == 2 else 'multiclass'\n"
                "baseline = majority_class_baseline(train_data[TARGET_COLUMN], holdout_data[TARGET_COLUMN])\n"
                "class_shares = train_data[TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict()\n"
                "print({{'train': len(train_data), 'holdout': len(holdout_data), 'test': 0 if test_data is None else len(test_data), 'features': len(FEATURE_COLUMNS), 'classes': CLASSES, 'problem_type': PROBLEM_TYPE, 'training_class_shares': class_shares, 'cap': cap_report, 'overlaps': overlaps}})\n"
                "if len(FEATURE_COLUMNS) > 100 or len(train_data) > 5_000:\n"
                "    print('Above the <=100-feature / <=5,000-row regime where Mitra is reported to be particularly strong.')\n"
                "print('majority-class baseline on the holdout', baseline)"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>The `MIN_TRAIN_ROWS` rule (50 support rows): `validate_inputs` raises a `ValueError` naming it, and the cell records the message as a finding in the input manifest — the manifest holds three accepted tables and this one rejection — so the notebook continues. The majority-class baseline predicts `benign` for every holdout row and scores about 0.63 accuracy with chance-level balanced accuracy and macro F1: that is the floor, and any model number is read against it.</details>"
            ),
        },
        {
            "md": (
                "## 6. Evaluate pretrained Mitra and executable baselines, then optionally fine-tune\n\n"
                "`pipe.fit(...)` with `fine_tune=False` registers the support rows and the model configuration through "
                "AutoGluon (`fit_mitra_predictor`); no weight is gradient-updated. `classification_metrics` scores the "
                "holdout and, when present, the independent test partition: **accuracy**, **balanced accuracy**, "
                "**macro F1** and **MCC** are discrete correctness under the implicit `argmax` rule; **log loss** scores "
                "the class probabilities (lower is better; a perfect model scores 0); **ROC-AUC** (binary, or one-vs-rest "
                "for more classes) scores ranking quality independent of any threshold and is `NaN` when undefined. "
                "AutoGluon's own evaluation (`pipe.evaluate`) is printed alongside.\n\n"
                "**What Mitra is conditioned on.** `fit` is given all support rows, but AutoGluon first sets 20 % of them "
                "aside as its own internal validation split (`holdout_frac`, automatic when no `tuning_data` is passed): "
                "the `Validation score` line of its log is measured on those internal rows, **not** on your holdout, and "
                "Mitra is conditioned on the remaining rows only (272 of 341 on the default sample). The cell prints "
                "`support_rows`, `mitra_context_rows` and `autogluon_internal_validation_rows`, and the exported bundle "
                "records them. A constant majority-class predictor, LightGBM and Random Forest are fitted on **all** "
                "support rows (with train-fitted median imputation and ordinal encoding that is never refitted on "
                "holdout/test) and scored on the exact same partitions, so the comparison gives the trees about 25 % "
                "more rows than Mitra sees — it favours the baselines, not Mitra. Their probability columns are mapped onto the Mitra class order with "
                "`align_probabilities` **before** they are scored, so a multiclass label permutation can never pass "
                "silently. AutoGluon's log is shortened to its warnings and the split and validation-score lines "
                "(`AUTOGLUON_LOG = 'full'` restores it); Mitra's float32 probabilities are renormalised to sum to one "
                "before log loss is computed, and for a two-class table the cell prints AutoGluon's **positive class** "
                "(the class its `f1`, `precision` and `recall` refer to).\n\n"
                "**Fine-tuning gate (off by default; GPU only).** With `RUN_FINE_TUNING=True` a second predictor is fitted "
                "with `fine_tune=True` for `FINE_TUNE_STEPS`, and it replaces the pretrained predictor **only** if it "
                "beats it on the holdout under `EVAL_METRIC` and the holdout has at least `MIN_SELECTION_HOLDOUT_ROWS` "
                "rows (a smaller holdout — `Sample: Wine` has 36 — keeps the pretrained predictor and says so in "
                "`selection_basis`). On a runtime without a GPU, `RUN_FINE_TUNING = True` stops before any fit. The independent test is evidence only; a worse test result is surfaced as a warning and never "
                "changes the selection. **Reproducibility boundary.** `SEED` drives the split, the cap and Mitra's "
                "`random_state`; bitwise-identical results across devices and library builds are not promised.\n\n"
                "**Predict:** pretrained Mitra (no training, conditioned on 272 of the 341 support rows), LightGBM and Random Forest (all 341) are scored on the same 114 holdout rows. Will Mitra beat the trees, and by how many **rows** — more than one or two?"
            ),
            "code": (
                "import contextlib\n"
                "import gc\n"
                "import logging\n"
                "import math\n"
                "import shutil\n"
                "import warnings\n\n"
                "from lightgbm import LGBMClassifier\n"
                "from sklearn.ensemble import RandomForestClassifier\n"
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.preprocessing import OrdinalEncoder\n\n"
                "EVAL_METRIC = 'accuracy'  # @param [\"accuracy\", \"balanced_accuracy\", \"log_loss\", \"f1_macro\", \"mcc\"]\n"
                "BASELINE_TIME_LIMIT = 300  # @param {{type:\"integer\"}}\n"
                "RUN_FINE_TUNING = False  # @param {{type:\"boolean\"}}\n"
                "FINE_TUNE_STEPS = 50  # @param {{type:\"integer\"}}\n"
                "FINE_TUNE_TIME_LIMIT = 600  # @param {{type:\"integer\"}}\n"
                "MAX_MEMORY_USAGE_RATIO = 1.10  # @param {{type:\"number\"}}\n"
                "AUTOGLUON_LOG = 'summary'  # @param [\"summary\", \"full\"]\n"
                "MIN_SELECTION_HOLDOUT_ROWS = 50\n"
                "if EVAL_METRIC not in EVAL_METRICS:\n"
                "    raise ValueError(f'Unsupported EVAL_METRIC: {{EVAL_METRIC}}')\n"
                "if RUN_FINE_TUNING and not torch.cuda.is_available():\n"
                "    raise RuntimeError('Fine-tuning requires a GPU. Choose Runtime -> Change runtime type -> GPU.')\n"
                "PRETRAINED_PATH, FINETUNED_PATH = Path('outputs') / 'mitra-pretrained', Path('outputs') / 'mitra-finetuned'\n"
                "for path in (PRETRAINED_PATH, FINETUNED_PATH):\n"
                "    shutil.rmtree(path, ignore_errors=True)\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.empty_cache()\n\n"
                "def score(model, frame):\n"
                "    proba = model.predict_proba(frame)\n"
                "    values = proba.to_numpy(dtype=float)\n"
                "    values = values / values.sum(axis=1, keepdims=True)  # Mitra returns float32 rows that miss 1.0 by ~1e-7\n"
                "    return classification_metrics(frame[TARGET_COLUMN].to_numpy(), model.predict(frame), values, list(proba.columns))\n\n"
                "class _AutoGluonSummary(logging.Filter):\n"
                "    KEEP = ('Train Rows', 'Val Rows', 'Validation score', 'positive class', 'Fitting model')\n\n"
                "    def filter(self, record):\n"
                "        return record.levelno >= logging.WARNING or any(key in record.getMessage() for key in self.KEEP)\n\n"
                "@contextlib.contextmanager\n"
                "def autogluon_log():\n"
                "    import autogluon.tabular  # noqa: F401 - importing it installs the AutoGluon log handler the filter attaches to\n"
                "    handlers = [] if AUTOGLUON_LOG == 'full' else [h for name in ('autogluon', '') for h in logging.getLogger(name).handlers]\n"
                "    summary = _AutoGluonSummary()\n"
                "    for handler in handlers:\n"
                "        handler.addFilter(summary)\n"
                "    try:\n"
                "        yield\n"
                "    finally:\n"
                "        for handler in handlers:\n"
                "            handler.removeFilter(summary)\n\n"
                "def context_rows(model):\n"
                "    # AutoGluon conditions Mitra on its internal train split; the rest scored its 'Validation score'.\n"
                "    try:\n"
                "        return len(model.predictor.load_data_internal('train', return_y=False)[0])\n"
                "    except Exception as exc:  # noqa: BLE001 - reported, never hidden\n"
                "        print('could not read the internal split:', type(exc).__name__, exc)\n"
                "        return None\n\n"
                "with autogluon_log():\n"
                "    pipe.fit(train_data, target_column=TARGET_COLUMN, eval_metric=EVAL_METRIC, path=PRETRAINED_PATH, fine_tune=False, time_limit=BASELINE_TIME_LIMIT, seed=SEED, problem_type=PROBLEM_TYPE, max_memory_usage_ratio=MAX_MEMORY_USAGE_RATIO)\n"
                "MITRA_CONTEXT_ROWS = context_rows(pipe)\n"
                "CONTEXT = {{'support_rows': len(train_data), 'mitra_context_rows': MITRA_CONTEXT_ROWS, 'autogluon_internal_validation_rows': None if MITRA_CONTEXT_ROWS is None else len(train_data) - MITRA_CONTEXT_ROWS, 'tree_baseline_rows': len(train_data)}}\n"
                "print(CONTEXT)\n"
                "if PROBLEM_TYPE == 'binary':\n"
                "    print({{'autogluon_positive_class': pipe.predictor.positive_class, 'note': 'AutoGluon f1 / precision / recall refer to this class'}})\n"
                "if sorted(pipe.class_labels, key=str) != CLASSES:\n"
                "    raise RuntimeError(f'AutoGluon registered classes {{pipe.class_labels}} != support classes {{CLASSES}}')\n"
                "CLASSES = list(pipe.class_labels)  # the predictor's own order drives every probability column below\n"
                "pretrained_metrics = score(pipe, holdout_data)\n"
                "pretrained_test_metrics = score(pipe, test_data) if test_data is not None else None\n"
                "with warnings.catch_warnings():\n"
                "    warnings.filterwarnings('ignore', message='.*do not sum to one.*')  # float32 probabilities; see the markdown above\n"
                "    print('pretrained holdout', pretrained_metrics, '| AutoGluon evaluate:', pipe.evaluate(holdout_data))\n"
                "if pretrained_test_metrics:\n"
                "    print('pretrained independent test', pretrained_test_metrics)\n\n"
                "candidate = candidate_metrics = candidate_test_metrics = None\n"
                "if RUN_FINE_TUNING:\n"
                "    candidate = MitraClassificationPipeline(weights_path=pipe.model_weight_path, config_path=pipe.config_path, snapshot_path=pipe.snapshot_path, device=pipe.device)\n"
                "    with autogluon_log():\n"
                "        candidate.fit(train_data, target_column=TARGET_COLUMN, eval_metric=EVAL_METRIC, path=FINETUNED_PATH, fine_tune=True, fine_tune_steps=FINE_TUNE_STEPS, time_limit=FINE_TUNE_TIME_LIMIT, seed=SEED, problem_type=PROBLEM_TYPE, max_memory_usage_ratio=MAX_MEMORY_USAGE_RATIO)\n"
                "    if list(candidate.class_labels) != CLASSES:\n"
                "        raise RuntimeError('fine-tuned predictor class order differs from the pretrained predictor')\n"
                "    candidate_metrics = score(candidate, holdout_data)\n"
                "    candidate_test_metrics = score(candidate, test_data) if test_data is not None else None\n"
                "    print('fine-tuned holdout', candidate_metrics)\n\n"
                "def better(current, reference, name):\n"
                "    if not (math.isfinite(current[name]) and math.isfinite(reference[name])):\n"
                "        raise ValueError(f'{{name}} unavailable for selection')\n"
                "    return current[name] < reference[name] if name == 'log_loss' else current[name] > reference[name]\n\n"
                "ACTIVE_MODEL, ACTIVE_MODE, SELECTION_BASIS = pipe, 'pretrained', 'default:pretrained'\n"
                "if candidate is not None:\n"
                "    if len(holdout_data) < MIN_SELECTION_HOLDOUT_ROWS:\n"
                "        SELECTION_BASIS = f'default:pretrained; holdout-too-small:{{len(holdout_data)}}<{{MIN_SELECTION_HOLDOUT_ROWS}}'\n"
                "    else:\n"
                "        SELECTION_BASIS = f'holdout:{{EVAL_METRIC}}'\n"
                "        if better(candidate_metrics, pretrained_metrics, EVAL_METRIC):\n"
                "            ACTIVE_MODEL, ACTIVE_MODE = candidate, 'fine-tuned'\n"
                "    if candidate_test_metrics and pretrained_test_metrics:\n"
                "        degraded = [name for name in candidate_test_metrics if name in pretrained_test_metrics and math.isfinite(candidate_test_metrics[name]) and math.isfinite(pretrained_test_metrics[name]) and candidate_test_metrics[name] != pretrained_test_metrics[name] and not better(candidate_test_metrics, pretrained_test_metrics, name)]\n"
                "        if degraded:\n"
                "            print('WARNING independent-test metrics worsened after fine-tuning:', degraded, '(evidence only; never used for selection)')\n"
                "active_metrics = candidate_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_metrics\n"
                "active_test_metrics = candidate_test_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_test_metrics\n"
                "print({{'recommended_for_export': ACTIVE_MODE, 'selection_basis': SELECTION_BASIS, 'device': ACTIVE_MODEL.device, 'classes': CLASSES}})\n\n"
                "# Executable baselines on the exact same partitions, probabilities aligned to the Mitra class order.\n"
                "y_train = train_data[TARGET_COLUMN].to_numpy()\n"
                "y_holdout = holdout_data[TARGET_COLUMN].to_numpy()\n"
                "y_test = test_data[TARGET_COLUMN].to_numpy() if test_data is not None else None\n"
                "class_array = np.asarray(CLASSES, dtype=object)\n"
                "baseline_rows = [{{'model': 'Majority-class', 'split': 'holdout', **baseline}}]\n"
                "if y_test is not None:\n"
                "    baseline_rows.append({{'model': 'Majority-class', 'split': 'test', **majority_class_baseline(y_train, y_test)}})\n"
                "cat_cols = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(train_data[c])]\n"
                "num_cols = [c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(train_data[c])]\n"
                "num_imputer = SimpleImputer(strategy='median', keep_empty_features=True) if num_cols else None\n"
                "cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) if cat_cols else None\n\n"
                "def tree_features(frame, fit=False):\n"
                "    parts = []\n"
                "    if num_cols:\n"
                "        values = num_imputer.fit_transform(frame[num_cols]) if fit else num_imputer.transform(frame[num_cols])\n"
                "        parts.append(pd.DataFrame(values, columns=num_cols, index=frame.index))\n"
                "    if cat_cols:\n"
                "        values = cat_encoder.fit_transform(frame[cat_cols].astype(str)) if fit else cat_encoder.transform(frame[cat_cols].astype(str))\n"
                "        parts.append(pd.DataFrame(values, columns=cat_cols, index=frame.index))\n"
                "    return pd.concat(parts, axis=1)[FEATURE_COLUMNS]\n\n"
                "def tree_scores(model, frame, y_true):\n"
                "    proba = align_probabilities(model.predict_proba(tree_features(frame)), list(model.classes_), CLASSES)\n"
                "    return classification_metrics(y_true, class_array[np.argmax(proba, axis=1)], proba, CLASSES)\n\n"
                "X_train_tree = tree_features(train_data, fit=True)\n"
                "for model_name, model in {{'LightGBM': LGBMClassifier(random_state=SEED, n_estimators=100, verbose=-1), 'RandomForest': RandomForestClassifier(random_state=SEED, n_estimators=100)}}.items():\n"
                "    model.fit(X_train_tree, y_train)\n"
                "    baseline_rows.append({{'model': model_name, 'split': 'holdout', **tree_scores(model, holdout_data, y_holdout)}})\n"
                "    if test_data is not None:\n"
                "        baseline_rows.append({{'model': model_name, 'split': 'test', **tree_scores(model, test_data, y_test)}})\n"
                "baseline_rows.append({{'model': f'Mitra-{{ACTIVE_MODE}}', 'split': 'holdout', **active_metrics}})\n"
                "if active_test_metrics:\n"
                "    baseline_rows.append({{'model': f'Mitra-{{ACTIVE_MODE}}', 'split': 'test', **active_test_metrics}})\n"
                "metrics_table = pd.DataFrame(baseline_rows)\n"
                "split_rows = {{'holdout': len(holdout_data), 'test': 0 if test_data is None else len(test_data)}}\n"
                "metrics_table.insert(2, 'rows_correct', [round(row['accuracy'] * split_rows[row['split']]) for row in baseline_rows])\n"
                "print(metrics_table.to_string(index=False))\n"
                "print({{'holdout_rows': len(holdout_data), 'one_row_resolution_pct': round(100.0 / len(holdout_data), 2), 'mitra_context_rows': MITRA_CONTEXT_ROWS, 'tree_baseline_rows': len(train_data)}})\n"
                "print('All values above are current-run tutorial metrics. Lower log_loss is better; higher accuracy / balanced_accuracy / f1_macro / mcc / roc_auc is better.')"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>On this well-separated table all three learners land high. Mitra is conditioned on 272 of the 341 support rows (AutoGluon kept 69 for its own validation score) with no gradient update, while the trees use all 341; even so the three are typically within one to three rows of each other, in either direction. One holdout row is 0.88 % of accuracy (100 / 114), printed as `one_row_resolution_pct`, and the `rows_correct` column turns every accuracy into rows: a gap of one to three rows on 114 is not a ranking.</details>\n\n**What to notice.** (1) Convert each holdout gap into rows with `rows_correct` before you call one model better; a gap under three rows on 114 is within what a different `SEED` moves. (2) Compare the holdout ranking with the test ranking: if the order changes, neither partition alone ranks the models. (3) Read `log_loss` and `roc_auc`, which score the probabilities and separate the models more smoothly than accuracy. (4) Remember that the trees saw about 25 % more support rows than Mitra.'
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. Here it "
                "carries the active model's holdout metrics (`accuracy`, `balanced_accuracy`, `log_loss`, `roc_auc`, "
                "`f1_macro`, `mcc` — the repository's own metric ids), the independent-test metrics when a test partition "
                "exists, and the majority-class baseline, with the verdict `sample-sanity`: one seeded stratified split "
                "with no dispersion estimate, tutorial evidence rather than a benchmark; the executable-baseline table "
                "is attached. Without a labelled holdout the verdict would be `not-measurable`. The report is written "
                "to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict:** which verdict will the report give for the sample path, and what would change it to `not-measurable`? Which partition's metrics does the `selection` field refer to?"
            ),
            "code": (
                "report = evaluation_report(active_metrics, baseline=baseline, independent_test=active_test_metrics, n_holdout=len(holdout_data), n_test=None if test_data is None else len(test_data), class_labels=CLASSES, target_column=TARGET_COLUMN, selection=SELECTION_BASIS, sample_kind=sample_kind, estimation='single seeded stratified split (support/holdout/independent test); no dispersion estimate')\n"
                "report['executable_baselines'] = baseline_rows\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: report[key] for key in ('verdict', 'reason', 'decision_rule', 'selection', 'n_holdout', 'n_test')}}, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>`sample-sanity`: one seeded stratified split of a public sample with 114 holdout and 114 test rows and no dispersion estimate — tutorial evidence that the contract works, not a benchmark. The verdict becomes `not-measurable` only without a labelled holdout. `selection` reads `default:pretrained` because the fine-tuning gate was off; with it on, the holdout alone selects and the independent test is evidence only.</details>'
            ),
        },
        {
            "md": (
                "## 8. Optional new-data inference with class probabilities\n\n"
                "Off by default so a top-to-bottom run needs no upload dialog. Switch `RUN_NEW_DATA_INFERENCE` on and upload "
                "one CSV with the support feature columns (extra columns are preserved in the output and not passed to "
                "the model), or set `NEW_DATA_PATH` for a non-interactive executor. `validate_inputs(..., "
                "target_column=None, feature_columns=...)` applies exactly the checks `validate_inference_frame` applies "
                "— unique header, every feature present, no pre-existing `prediction`/`probability_*` columns. The output "
                "adds `prediction` (the `argmax` label) and one `probability_<class>` column per class **in the "
                "predictor's class order**; the probabilities are uncalibrated and the package ships no threshold, so a "
                "cost-sensitive decision must threshold the relevant probability on the caller's own labelled data. On "
                "the sample path eight held-out rows are scored instead so a prediction CSV always exists."
            ),
            "code": (
                "RUN_NEW_DATA_INFERENCE = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "def with_predictions(raw_rows, model_rows):\n"
                "    out = raw_rows.copy()\n"
                "    proba = ACTIVE_MODEL.predict_proba(model_rows)\n"
                "    out['prediction'] = ACTIVE_MODEL.predict(model_rows)\n"
                "    for class_label in CLASSES:\n"
                "        out[f'probability_{{class_label}}'] = proba[class_label].to_numpy()\n"
                "    return out\n\n"
                "new_data_result = None\n"
                "if RUN_NEW_DATA_INFERENCE:\n"
                "    if NEW_DATA_PATH:\n"
                "        csv_name, payload = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        new_upload = files.upload()\n"
                "        csvs = [(name, payload) for name, payload in new_upload.items() if name.lower().endswith('.csv')]\n"
                "        if len(csvs) != 1:\n"
                "            raise RuntimeError('Upload exactly one inference CSV.')\n"
                "        csv_name, payload = csvs[0]\n"
                "    new_data = read_csv_bytes(payload, csv_name)\n"
                "    inference_manifest = validate_inputs(new_data, None, feature_columns=FEATURE_COLUMNS, names=[csv_name])\n"
                "    X_new, extra_columns = validate_inference_frame(new_data, FEATURE_COLUMNS)\n"
                "    out = with_predictions(new_data, X_new)\n"
                "    new_data_result = {{'input': csv_name, 'rows': len(out), 'extra_columns': extra_columns, 'input_manifest': inference_manifest}}\n"
                "else:\n"
                "    smoke = holdout_data[FEATURE_COLUMNS].head(8).copy()\n"
                "    out = with_predictions(smoke, smoke)\n"
                "    print('Inference upload skipped; eight held-out rows scored instead.')\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "print(out.head())"
            ),
        },
        {
            "md": (
                "## 9. Export the deployable predictor bundle, then prove a fresh reload\n\n"
                "The deployable artifact is the selected AutoGluon predictor directory (ART1), not a replacement "
                "`model.safetensors`: for Mitra it contains the registered support rows, the model configuration and "
                "AutoGluon's serialised state, so it inherits the source data's confidentiality, licensing, retention "
                "and disclosure obligations (ART3/ART7). `tutorial_run_metadata.json` records the base-model identity "
                "and digests, the AutoGluon/Python versions, the problem type, the target, features and class labels, "
                "the mode and selection basis, the data digest and the metrics; `write_artifact_manifest` inventories "
                "every file with its size and SHA-256 (ART5/ART6). The directory is zipped and its SHA-256 printed.\n\n"
                "An in-memory predictor is not evidence that serialisation worked. The cell extracts the exact ZIP into a "
                "fresh directory with `safe_extract_archive` (path, symlink, size and compression-ratio checks; never "
                "`extractall`), verifies the manifest and provenance with `validate_artifact_directory` **before** "
                "deserialising, reloads the predictor and checks that its labels equal and its class probabilities "
                "agree with the in-memory model's on eight held-out rows within `rtol=1e-6, atol=1e-8` (VER1–VER5). "
                "The result JSON then records everything: predictions, metrics, the evaluation report, the input "
                "manifest, the data digest, the bundle identity, the notebook's source, the model identity, revision "
                "and licence, and the runtime.\n\n"
                "**Predict:** the reloaded predictor is built from the ZIP alone in a fresh directory. Will its predictions on eight held-out rows equal the in-memory model's exactly, within `rtol=1e-6`, or differ? And what inside the bundle makes it confidential?"
            ),
            "code": (
                "from datetime import datetime, timezone\n\n"
                "from autogluon.tabular import TabularPredictor\n\n"
                "active_path = Path(ACTIVE_MODEL.predictor.path)\n"
                "run_metadata = {{\n"
                "    'artifact_format': ARTIFACT_FORMAT, 'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'base_model': MODEL_ID, 'base_model_revision': MODEL_REVISION, 'weights_sha256': WEIGHTS_SHA256, 'config_sha256': CONFIG_SHA256,\n"
                "    'notebook_source': NOTEBOOK_SOURCE, 'model_source': 'verified local snapshot (Section 3)',\n"
                "    'autogluon_version': importlib.metadata.version('autogluon.tabular'), 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'device': ACTIVE_MODEL.device,\n"
                "    'problem_type': ACTIVE_MODEL.problem_type, 'target_column': TARGET_COLUMN, 'features': FEATURE_COLUMNS, 'class_labels': [str(label) for label in CLASSES], 'decision_rule': DECISION_RULE,\n"
                "    'mode': ACTIVE_MODE, 'selection_basis': SELECTION_BASIS, 'seed': SEED, 'data_source': DATA_SOURCE, 'data_sha256': DATA_DIGEST,\n"
                "    'train_rows_before_cap': cap_report['before'], 'train_rows_used': len(train_data), 'train_row_cap_applied': cap_report['applied'],\n"
                "    'support_rows': len(train_data), 'mitra_context_rows': context_rows(ACTIVE_MODEL), 'autogluon_internal_validation': 'automatic holdout_frac split of the support rows; Mitra is conditioned on mitra_context_rows of them', 'tree_baseline_rows': len(train_data),\n"
                "    'holdout_rows': len(holdout_data), 'independent_test_rows': None if test_data is None else len(test_data), 'eval_metric': EVAL_METRIC,\n"
                "    'fine_tuning_requested': RUN_FINE_TUNING, 'fine_tune_steps_requested': FINE_TUNE_STEPS if RUN_FINE_TUNING else None, 'fine_tune_time_limit_seconds': FINE_TUNE_TIME_LIMIT if RUN_FINE_TUNING else None, 'max_memory_usage_ratio': MAX_MEMORY_USAGE_RATIO,\n"
                "    'pretrained_holdout_metrics': pretrained_metrics, 'pretrained_test_metrics': pretrained_test_metrics, 'finetuned_holdout_metrics': candidate_metrics, 'finetuned_test_metrics': candidate_test_metrics,\n"
                "    'probability_calibration': 'uncalibrated class probabilities; argmax decision rule; no threshold shipped', 'exported_at_utc': datetime.now(timezone.utc).isoformat(),\n"
                "}}\n"
                "(active_path / 'tutorial_run_metadata.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')\n"
                "manifest_path = write_artifact_manifest(active_path)\n"
                "archive_base = Path('outputs') / '{stem}_predictor'\n"
                "Path(str(archive_base) + '.zip').unlink(missing_ok=True)\n"
                "archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=active_path))\n"
                "archive_digest = sha256_file(archive)\n"
                "print({{'predictor_zip': str(archive), 'zip_sha256': archive_digest, 'artifact_manifest': str(manifest_path)}})\n\n"
                "RELOAD_DIR = Path('outputs') / 'artifact-reload'\n"
                "shutil.rmtree(RELOAD_DIR, ignore_errors=True)\n"
                "safe_extract_archive(archive, RELOAD_DIR)\n"
                "verified_manifest, verified_metadata = validate_artifact_directory(RELOAD_DIR)\n"
                "if verified_metadata['autogluon_version'] != importlib.metadata.version('autogluon.tabular'):\n"
                "    raise RuntimeError('Artifact/runtime AutoGluon version mismatch.')\n"
                "reloaded = TabularPredictor.load(str(RELOAD_DIR))\n"
                "if list(reloaded.class_labels) != CLASSES:\n"
                "    raise RuntimeError('Reloaded class labels differ from the exported class order')\n"
                "smoke_X = holdout_data[FEATURE_COLUMNS].head(8).copy()\n"
                "if not np.array_equal(ACTIVE_MODEL.predict(smoke_X), predict_classification(reloaded, smoke_X, FEATURE_COLUMNS)):\n"
                "    raise RuntimeError('Prediction mismatch after reload')\n"
                "np.testing.assert_allclose(ACTIVE_MODEL.predict_proba(smoke_X).to_numpy(dtype=float), predict_proba_classification(reloaded, smoke_X, FEATURE_COLUMNS, CLASSES).to_numpy(dtype=float), rtol=1e-6, atol=1e-8)\n"
                "print('PASS: artifact manifest/provenance verified before deserialisation; predictor reloaded from fresh files; labels identical and probabilities equivalent (rtol=1e-6, atol=1e-8).')\n\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'new_data': new_data_result,\n"
                "    'metrics': {{'active_mode': ACTIVE_MODE, 'holdout': active_metrics, 'independent_test': active_test_metrics, 'pretrained_holdout': pretrained_metrics, 'finetuned_holdout': candidate_metrics, 'executable_baselines': baseline_rows}},\n"
                "    'majority_class_baseline': baseline,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'inference': {{'decisionRule': DECISION_RULE + ' over uncalibrated class probabilities', 'threshold': None, 'classLabels': [str(label) for label in CLASSES], 'problemType': ACTIVE_MODEL.problem_type}},\n"
                "    'sample': {{'kind': sample_kind, 'name': data_name, 'source': DATA_SOURCE, 'data_sha256': DATA_DIGEST, 'train_rows': len(train_data), 'holdout_rows': len(holdout_data), 'test_rows': 0 if test_data is None else len(test_data), 'training_class_shares': class_shares}},\n"
                "    'artifact': {{'zip': archive.name, 'zip_sha256': archive_digest, 'run_metadata': run_metadata}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'autogluon': importlib.metadata.version('autogluon.tabular'), 'lightgbm': importlib.metadata.version('lightgbm'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': ACTIVE_MODEL.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>Equal within `rtol=1e-6, atol=1e-8`, and the labels identical: the cell prints `PASS` only after `validate_artifact_directory` verified every file's size and SHA-256 and the provenance before `TabularPredictor.load` ran. The bundle contains the registered support rows themselves (Mitra predicts by attending over them), so it inherits the source data's confidentiality, licensing and retention obligations — a point the `tutorial_run_metadata.json` records as the data digest.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "`prediction` is the `argmax` of uncalibrated class probabilities over the classes registered from the "
        "support rows; the package ships no threshold, and any deployment cut-off must be chosen on the caller's own "
        "labelled, domain-representative data. The evaluation report's `sample-sanity` verdict names what it is: one "
        "seeded stratified split of a public sample with no dispersion estimate — tutorial evidence that must not be "
        "generalised. The executable baselines show whether the foundation model adds value on this table only when the gap is larger than a few rows and holds on both the holdout and the test partition; the fine-tuned "
        "variant, when requested, is selected on the holdout only. Rows that are not independent, classes absent from "
        "the support rows (refused, not predicted), cross-partition overlaps, tables above the ≤100-feature / "
        "≤5,000-row regime, more than `MAX_CLASSES` classes and capped support sets all change results in ways these "
        "metrics do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint and stage it offline, validate the demonstrated tables with "
        "class coverage, register classification support rows through AutoGluon, compute sample metrics against "
        "trivial and classical baselines with strict label alignment, write the input manifest and the evaluation "
        "report, score rows with class probabilities under an explicit `argmax` rule, export the deployable predictor "
        "bundle with its manifest and provenance, and reload an equivalent predictor from that bundle alone — without "
        "the repository being reachable. It does **not** establish benchmark superiority, domain generalisation, "
        "fairness, robustness, probability calibration, production safety, or deployment fitness.\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file (`model.safetensors`, 302,717,904 bytes, or `config.json`). Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        "- **Section 6 stops with \"No models were trained successfully\" after a long log** — AutoGluon estimated that Mitra needs more memory than the runtime has free and *skipped* it (the log line names the estimate). `MAX_MEMORY_USAGE_RATIO` (Section 6, default 1.10) is how much of the free memory AutoGluon may plan to use: raise it (for example to 3.0) only on a runtime you know has headroom, or use a larger runtime, a smaller table or a GPU runtime. AutoGluon's estimate is conservative; the default sample fits on a standard Colab or Kaggle runtime.\n"
        "- **`RUN_FINE_TUNING = True` stops with \"Fine-tuning requires a GPU\"** — the check runs before any fit; switch to a GPU runtime and run Section 6 again.\n"
        '- **The metrics differ from another run** — expected in the last decimals: the one-row resolution of the holdout is printed in Section 6, and bitwise-identical results across devices and library builds are not promised. A Mitra row below LightGBM or Random Forest is a finding to read, not an error.\n'
        '- **`USE_BYOD=True requires an Upload DATA_SOURCE` / `Set USE_BYOD=True`** — the gate and the source must agree: pick an `Upload …` source and set `USE_BYOD = True`.\n'
        "- **BYOD: `TARGET_COLUMN 'target' is not a column of …` or `… is an archive`** — set `TARGET_COLUMN` in Section 4 to the label column the message lists (`Churn` for Telco Churn, `class` for Adult Census), and extract ZIP archives before pointing `BYOD_PATH` at the CSV or the pre-split directory.\n"
        '- **BYOD: "BYOD path … does not exist" / "is missing [...]" / "the upload dialog exists only in Google Colab" / "Upload exactly one labelled CSV"** — set `BYOD_PATH` to a CSV file (or, for the pre-split option, a directory holding `train.csv`, `val.csv` and `test.csv`) in the runtime; it works on Kaggle and Jupyter. On Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` from `validate_inputs` or `validate_labeled_frame`** — it names the table and the rule: a missing or duplicate column, fewer than `MIN_TRAIN_ROWS` rows, more than `MAX_FEATURES` features, fewer than `MIN_CLASSES` or more than `MAX_CLASSES` classes, a class with fewer than `MIN_CLASS_COUNT` support rows, or a holdout class absent from the support rows. Fix the table rather than the check.\n'
        "- **Section 9's reload check fails** — the export or the reload is broken; run Sections 6–9 again. Do not use the bundle.\n"
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        "Each of these changes one default and keeps the rest of the path. Set the value, then re-run the cells named; the earlier cells keep their results.\n"
        '\n'
        "**Activity (about 3 minutes on CPU): does Mitra's advantage survive a three-class table?** *Predict* — on `Sample: Wine` (178 rows, three classes) will Mitra beat LightGBM on the holdout, tie, or trail, and by how many rows? *Change* — set `DATA_SOURCE = 'Sample: Wine'` in Section 4. *Run* — Sections 4–7 (Runtime → Run after, from Section 4). *Observe* — 106 / 36 / 36 rows, `problem_type` `multiclass`, one-vs-rest `roc_auc`, and the `rows_correct` column on both partitions. *Explain* — with 36 holdout rows one row is 2.8 % of accuracy; write whether the holdout and the test rankings agree and what that says about a ranking from one small split.\n"
        '\n'
        "- **`EVAL_METRIC = 'log_loss'`** — re-run Sections 6–9. The metrics table does not change (every metric is always computed); AutoGluon's `Validation score` line switches to log loss, and with fine-tuning on, the gate compares log loss instead of accuracy.\n"
        "- **A different `SEED`** — re-run Sections 4–9. Expect the partition contents, and so every metric, to move by a few rows; compare the size of that movement with the gaps between models.\n"
        "- **`RUN_FINE_TUNING = True` on a GPU runtime** — re-run Sections 6–9 (several minutes). Expect a `fine-tuned holdout` line and `recommended_for_export` naming whichever predictor wins on the holdout under `EVAL_METRIC`; a worse independent-test result prints a warning but never changes the choice. On `Sample: Wine` the 36-row holdout is below `MIN_SELECTION_HOLDOUT_ROWS` (50), so the pretrained predictor is always kept and `selection_basis` reads `holdout-too-small:36<50`.\n"
        "- **Your own partitions** — extract one of the repository's `examples/sample-data` archives, set `DATA_SOURCE = 'Upload pre-split train/val/test'`, `USE_BYOD = True`, `BYOD_PATH` to the extracted directory and `TARGET_COLUMN` to its label (`Churn` for Telco Churn, `class` for Adult Census), then re-run Sections 4–9. Expect the Section 5 manifest to describe your table and a larger, slower Section 6.\n"
        "- **The companion notebook** — feed the exported `outputs/mitra_classifier_predictor.zip` to the predictor-inference notebook in a separate session (its `ARTIFACT_ZIP_PATH` field).\n"
        '\n'
        '## Glossary\n'
        '\n'
        '- **In-context conditioning** — Mitra predicts a query row by attending over the support rows registered at `fit`; nothing is gradient-updated unless `RUN_FINE_TUNING` is on.\n'
        '- **Support / holdout / independent test** — the rows the model is conditioned on; the partition that scores it and, with fine-tuning on, selects between the two predictors; the partition that is evidence only and never drives a selection.\n'
        '- **Stratified split / class coverage** — every class keeps its share in each partition; a holdout or test class absent from the support rows is refused, because an in-context classifier can only predict classes it has seen.\n'
        '- **Accuracy, balanced accuracy, macro F1, MCC** — discrete correctness under the `argmax` rule; **log loss** scores the probabilities (lower is better); **ROC-AUC** scores ranking quality independent of any threshold.\n'
        '- **Majority-class baseline** — always predict the most frequent support class; the floor every model number is read against.\n'
        "- **`align_probabilities`** — maps a baseline's probability columns onto Mitra's class order before scoring, so a label permutation can never pass silently.\n"
        '- **Executable baselines** — LightGBM and Random Forest fitted on all support rows with train-fitted imputation and encoding, scored on exactly the same partitions.\n'
        '- **Mitra context rows / AutoGluon internal validation** — AutoGluon sets 20 % of the support rows aside to compute its own `Validation score`; Mitra is conditioned on the rest (272 of 341 on the default sample).\n'
        '- **Positive class** — for a two-class table, the class AutoGluon\'s `f1`, `precision` and `recall` refer to; Section 6 prints it.\n'
        '- **One-row resolution** — 100 / holdout rows, the smallest step a holdout metric can move; differences below it are noise.\n'
        "- **`sample-sanity` / `not-measurable`** — the evaluation report's verdict on one seeded split of a public sample (no dispersion estimate), and the verdict when no labelled holdout exists.\n"
        '- **Predictor bundle** — the selected AutoGluon predictor directory with `tutorial_run_metadata.json` and `artifact_manifest.json`, zipped; it contains the support rows, so it inherits their confidentiality.\n'
        '- **Fresh reload** — the ZIP extracted with `safe_extract_archive` into a new directory, verified by `validate_artifact_directory` before `TabularPredictor.load`, and checked against the in-memory model on eight rows.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: a CSV (or a pre-split directory) via `BYOD_PATH`, or the Colab upload dialog when the path is empty.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        'Before you leave, write three lines in this cell: (1) the pretrained Mitra row of Section 6 beside the majority-class predictor, LightGBM and Random Forest on the holdout, and whether the differences exceed the one-row resolution; (2) what the `sample-sanity` verdict does and does not license you to claim; (3) one property of your own table (row count, feature count, class balance, independence of rows) that would change how you read these numbers.\n'
        '\n'
        "**Next experiments (summary):** the Wine activity (re-run Sections 4–7), `EVAL_METRIC` (Sections 6–9), `SEED` "
        "(Sections 4–9), fine-tuning on a GPU (Sections 6–9; never selects on Wine's 36-row holdout), your own extracted "
        "partitions with `TARGET_COLUMN` set (Sections 4–9), and the exported `outputs/{stem}_predictor.zip` in the "
        "companion predictor-inference notebook.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- Sample dataset card: https://github.com/kurtvalcorza/{REPO}/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/autogluon/autogluon\n"
        "- Mitra paper: https://arxiv.org/abs/2508.02927"
    ),
}
