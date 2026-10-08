"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The
notebook carries the same package module and the same pinned snapshot as the E2E notebook; it consumes an
AutoGluon predictor bundle (`mitra_classifier_predictor.zip`) produced by a *separate* execution and never creates one.
The default path downloads the trusted sample bundle pinned in ``SAMPLE_ARTIFACT`` (release asset
``sample-bundle-v1`` of this repository, written by the E2E notebook in a recorded Colab T4 run) and verifies its
whole-archive SHA-256 before extraction (NOTEBOOK_SPEC SART6-SART8); its sample input is the producer's independent
test partition re-derived from scikit-learn's bundled table. ``ARTIFACT_ZIP_PATH`` / ``NEW_DATA_PATH`` switch to
your own bundle and rows (a path, or ``upload`` for the Colab dialog).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, _E2E = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.TEMPLATE

TEMPLATE = {
    **{k: _E2E[k] for k in ("package", "package_dir", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module", "model_load", "runtime_imports", "isolated_runtime", "infrastructure_labels", "managed_python", "uv", "lock")},
    "stem": "mitra_classifier_predictor_inference",
    "notebook_name": "mitra_classifier_predictor_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** with no field edited downloads the **trusted sample bundle** pinned in Section 4 (release asset `sample-bundle-v1` of this repository, produced by the E2E tutorial in a recorded Colab T4 run) and checks its whole-archive SHA-256 before extraction, and scores a **sample input** — the producer's independent test partition re-derived from scikit-learn's bundled Breast Cancer table, with the label removed — so there is no upload dialog and no configuration edit. It builds an isolated, hash-locked environment with the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), validates the bundle (path-safe extraction, manifest digests, provenance, pinned model identity) before any deserialisation, checks runtime compatibility, reconstructs the predictor from the bundle alone, validates the new rows into an input manifest, predicts with class probabilities, reports what cannot be measured, and exports outputs — all inside this kernel, with no DIMER worker or service and no credential."
    ),
    "byod": (
        "New-input BYOD is the `NEW_DATA_PATH`/upload branch in Section 6: your own unlabelled CSV with the bundle's required feature columns passes through the same validation, prediction and export cells. A user-supplied predictor bundle is the separate `ARTIFACT_ZIP_PATH`/upload branch in Section 4, validated before deserialisation (`ALLOW_UNVERIFIED_ARTIFACT` stays `False`). Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "title": "Mitra Classifier — DIMER exported-predictor inference tutorial (standalone)",
    "badges": [
        badge
        if badge[0] != "Open In Colab"
        else (
            badge[0],
            badge[1],
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/mitra_classifier_predictor_inference_colab.ipynb",
        )
        for badge in BADGES
    ],
    "capability": "consume an externally supplied AutoGluon predictor bundle (`mitra_classifier_predictor.zip`), reconstruct the Mitra serving state, validate new tabular input, and produce classification class-probability inferences",
    "intro": (
        "This notebook consumes a predictor bundle produced **outside this execution** (for example by the E2E tutorial "
        "in a separate session): it verifies a trusted whole-archive SHA-256 before any Python deserialisation, extracts "
        "only after every member passed the path, symlink, size and compression-ratio checks, verifies the bundle's "
        "digest manifest and provenance, checks runtime compatibility, reconstructs the AutoGluon predictor from the "
        "bundle alone, accepts unlabelled rows the bundle never saw, predicts class labels with probabilities, and exports results. "
        "**No artifact is created here** and nothing is trained. Prediction needs only the bundle: it carries the model "
        "state and the support rows. Section 3 still downloads and digest-verifies the pinned 303 MB base checkpoint, "
        "which proves that the pin the bundle names still resolves to the recorded bytes; the bundle's base-model check in "
        "Section 4 compares against the identity carried in this notebook.\n\n"
        "**Trust boundary.** Archive path checks and file digests establish integrity and consistency, not sender "
        "authenticity or the safety of Python object deserialisation: `TabularPredictor.load()` executes trusted "
        "serialised model state (AINF10). Load only bundles from a trusted producer; a trusted whole-archive digest is "
        "required by default and can be waived only by an explicit override for an already-trusted local bundle."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint, supply an externally produced bundle and verify its whole-archive digest, extract it "
        "safely and validate its manifest and provenance before deserialisation, check runtime compatibility, "
        "reconstruct the predictor from the bundle alone, validate new unlabelled rows into an input manifest, predict "
        "class labels with probabilities, change one input feature and explain how the predictions move, produce an evaluation report that is `not-measurable` because no labels exist, "
        "and export machine-readable predictions plus provenance."
    ),
    "exclusions": (
        "artifact creation, in-notebook fitting or fine-tuning, regression, calibrated probabilities, or any quality "
        "claim: without labelled rows nothing is measured, and the exported probabilities are uncalibrated model outputs with no "
        "prediction interval."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner or practitioner who has an exported Mitra predictor bundle from the E2E tutorial (or from a trusted producer) and wants to score new rows with it in a separate session — and to see what must be checked before a Python-serialised artifact is deserialised. Basic pandas and Colab or Jupyter familiarity are enough; no prior experience with AutoGluon is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | by default, the trusted sample bundle `mitra_classifier_predictor.zip` (release `sample-bundle-v1`, SHA-256 pinned) and 114 unlabelled sample rows; optionally your own bundle (`ARTIFACT_ZIP_PATH` + `EXPECTED_ZIP_SHA256`) and your own unlabelled CSV (`NEW_DATA_PATH`) |\n| Model | the AutoGluon `TabularPredictor` reconstructed from the bundle alone (the pinned `autogluon/mitra-classifier` base checkpoint is verified in Section 3 only so the bundle's recorded digests can be checked) |\n| Output | a label (the `argmax` of uncalibrated class probabilities) and one `probability_<class>` column per class for every input row; an input manifest with one recorded refusal; an evaluation report whose verdict is `not-measurable` (no labels); `result.json` with the bundle identity and provenance |\n\n**How to use this notebook.** Choose a runtime and **Runtime → Run all**. With no field edited, Run all completes in one pass, scoring the sample rows with the pinned sample bundle: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried module and the verified base snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning**. The answers describe what the checks do, not numbers to match; recorded runs of each notebook revision are listed in the repository's `docs/release-verification.md`. Section 8 is a short activity: change one feature and explain how the predictions move. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end.\n\n**Roadmap:** 1–3 infrastructure → 4 the bundle: whole-archive digest, safe extraction, manifest and provenance *(core concept: what is verified before any deserialisation, and what is not)* → 5 runtime compatibility and reconstruction from the bundle alone *(engineering)* → 6 new rows, the input manifest and a deliberate refusal *(core concept: the fitted schema is the contract)* → 7 predict, report `not-measurable`, export *(evaluation practice: what cannot be measured without labels)* → 8 activity: change one feature *(interpretation)* → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic pandas and Colab or Jupyter familiarity; no prior experience with AutoGluon. The digest, extraction, manifest and provenance checks, runtime compatibility and the input manifest are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (AutoGluon 1.5.0 and its torch), so nothing is installed into the kernel; a Windows or macOS kernel is not supported. The AutoGluon and Python major/minor versions of that environment must match the ones recorded in the bundle (the E2E tutorial of this revision exports under the same environment). The default path runs on CPU and uses CUDA automatically when available.",
        "- **Artifact:** by default the trusted sample bundle, downloaded from this repository's release `sample-bundle-v1` (about 280 MB, no sign-in) and checked against its pinned SHA-256 before extraction; it was produced by the E2E tutorial in a recorded run, outside this execution. Your own bundle (the E2E tutorial writes `outputs/mitra_classifier_predictor.zip` and prints its SHA-256) is optional: copy it into the runtime — a Kaggle dataset, Google Drive or `gsutil` copy — and set `ARTIFACT_ZIP_PATH` to it with its trusted SHA-256 in `EXPECTED_ZIP_SHA256`. `ARTIFACT_ZIP_PATH = 'upload'` opens the Colab dialog, which is impractical for a bundle this size. Nothing in this notebook manufactures a bundle.",
        "- **Data:** by default 114 sample rows: the producer's independent test partition of scikit-learn's bundled Breast Cancer Wisconsin table (re-derived from the installed package with the bundle's recorded seed, label removed), rows the bundle's support set never contained. Optionally one separate, unlabelled UTF-8 CSV with the bundle's feature columns via `NEW_DATA_PATH` (`upload` opens the Colab dialog). The sample rows fit only the sample bundle; with your own bundle, give your own rows. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Obtain the external bundle and validate it before any deserialisation\n\n"
                "**Default: the trusted sample bundle.** With `ARTIFACT_ZIP_PATH` empty the cell downloads the bundle pinned "
                "in `SAMPLE_ARTIFACT` — release asset `sample-bundle-v1` of this repository, written by the E2E tutorial in a "
                "recorded Colab T4 run (the printed `producer` names the notebook blob, commit and run) — and compares its "
                "whole-archive SHA-256 with the pinned digest **before extraction**; a mismatch stops the cell and nothing is "
                "extracted. A complete earlier download in this runtime is reused. Nothing is uploaded and `google.colab` is not "
                "imported on this path.\n\n"
                "**Your own bundle (optional).** Set `ARTIFACT_ZIP_PATH` to a bundle ZIP already in the runtime (Kaggle dataset, "
                "Drive or `gsutil` copy), or to `upload` for the Colab dialog. A trusted whole-archive SHA-256 in `EXPECTED_ZIP_SHA256` is "
                "required by default; `ALLOW_UNVERIFIED_ARTIFACT=True` waives it only for an already-trusted local "
                "bundle and prints a warning. `safe_extract_archive` extracts only after every member passed the path, "
                "symlink, per-member size, expanded-size and compression-ratio checks (AINF3), and "
                "`validate_artifact_directory` verifies the bundle's digest manifest (every listed file present, no "
                "unlisted file, sizes and SHA-256 equal) and its `tutorial_run_metadata.json` provenance (artifact "
                "format and version, base model, revision and digests, AutoGluon/Python versions, target, features, mode, "
                "selection basis) before anything is deserialised (AINF4). The bundle's base-model identity and digests "
                "are also compared with the carried package's pinned values, so a bundle built on another base model is "
                "refused.\n\n"
                "**Predict:** list the checks that run before `TabularPredictor.load`, in order. Which of them would catch an archive that was tampered with after export, and which property of the archive does none of them establish?"
            ),
            "code": (
                "import shutil\n\n"
                "# Location of the bundle: empty = the pinned sample bundle below; a path = your own bundle ZIP; 'upload' = the Colab dialog.\n"
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "ALLOW_UNVERIFIED_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "# The trusted sample artifact (NOTEBOOK_SPEC SART6-SART8): one bundle written by the E2E tutorial in a recorded hosted run,\n"
                "# published as an immutable release asset, pinned by URL and whole-archive SHA-256 (checked before extraction).\n"
                "SAMPLE_ARTIFACT = {{'url': 'https://github.com/kurtvalcorza/mitra-classifier-pipeline/releases/download/sample-bundle-v1/mitra_classifier_predictor.zip', 'sha256': 'd2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01', 'producer': {{'notebook': 'tutorials/mitra_classifier_colab.ipynb', 'notebook_blob': '2748c0a2e7f84c4c2799a7ce0673f984686e33f0', 'commit': '83b4d0c738ef1d207e4900921e954e688fb64c31', 'run': '2026-10-08 Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, default path', 'evidence': 'docs/execution-evidence/2026-10-08/mitra_classifier_colab/', 'release': 'sample-bundle-v1', 'asset_id': 620969480, 'bytes': 280280389}}}}\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "zip_dir = Path('external-artifact')\n"
                "zip_dir.mkdir(parents=True, exist_ok=True)\n"
                "location = ARTIFACT_ZIP_PATH.strip()\n"
                "if not location:\n"
                "    zip_path = zip_dir / ('sample-' + Path(SAMPLE_ARTIFACT['url']).name)\n"
                "    if not (zip_path.is_file() and sha256_file(zip_path) == SAMPLE_ARTIFACT['sha256']):\n"
                "        partial = zip_path.with_suffix('.part')\n"
                "        _download(SAMPLE_ARTIFACT['url'], partial, timeout=300)  # the carried module's download helper\n"
                "        partial.replace(zip_path)\n"
                "    print({{'sample_artifact': SAMPLE_ARTIFACT['url'], 'pinned_sha256': SAMPLE_ARTIFACT['sha256'], 'producer': SAMPLE_ARTIFACT['producer']}})\n"
                "    zip_name, artifact_source, expected_digest = zip_path.name, f\"sample artifact: {{SAMPLE_ARTIFACT['url']}}\", SAMPLE_ARTIFACT['sha256']\n"
                "elif location.lower() == 'upload':\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(\"ARTIFACT_ZIP_PATH = 'upload' needs the Colab upload dialog, which this runtime does not have: copy the bundle ZIP into the runtime and set ARTIFACT_ZIP_PATH to its path.\") from None\n"
                "    uploaded = files.upload() or {{}}\n"
                "    zips = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.zip')]\n"
                "    if len(zips) != 1:\n"
                "        raise RuntimeError(f'Upload exactly one predictor bundle ZIP (received {{sorted(uploaded) or \"nothing; a cancelled dialog sends none\"}}). Run this cell again, or set ARTIFACT_ZIP_PATH to a path.')\n"
                "    zip_name, zip_payload = zips[0]\n"
                "    zip_path = zip_dir / Path(zip_name).name\n"
                "    zip_path.write_bytes(zip_payload)\n"
                "    artifact_source, expected_digest = 'upload dialog', EXPECTED_ZIP_SHA256.strip().lower()\n"
                "else:\n"
                "    source_path = Path(location).expanduser()\n"
                "    if not source_path.is_file():\n"
                "        raise FileNotFoundError(f'ARTIFACT_ZIP_PATH {{location!r}} is not a file (relative paths start at {{os.getcwd()}}): give the predictor bundle ZIP, leave it empty for the sample bundle, or use upload on Colab.')\n"
                "    zip_name = source_path.name\n"
                "    zip_path = zip_dir / zip_name\n"
                "    if source_path.resolve() != zip_path.resolve():\n"
                "        shutil.copyfile(source_path, zip_path)\n"
                "    artifact_source, expected_digest = f'path: {{location}}', EXPECTED_ZIP_SHA256.strip().lower()\n"
                "zip_sha256 = sha256_file(zip_path)\n"
                "if expected_digest:\n"
                "    if len(expected_digest) != 64 or any(ch not in '0123456789abcdef' for ch in expected_digest):\n"
                "        raise ValueError('Expected ZIP SHA-256 must be a 64-character hexadecimal digest.')\n"
                "    if zip_sha256 != expected_digest:\n"
                "        raise RuntimeError(f'Predictor ZIP checksum mismatch. Expected {{expected_digest}}; got {{zip_sha256}}.')\n"
                "    print('Whole-archive SHA-256 matches the trusted expected digest.')\n"
                "elif not ALLOW_UNVERIFIED_ARTIFACT:\n"
                "    raise RuntimeError('A trusted whole-archive SHA-256 is required before loading this Python-serialized artifact. Set EXPECTED_ZIP_SHA256, or set ALLOW_UNVERIFIED_ARTIFACT=True only for an already-trusted local artifact.')\n"
                "else:\n"
                "    print('WARNING explicit unverified-artifact override enabled; sender authenticity/substitution is not checked.')\n"
                "EXTRACT_ROOT = Path('external-artifact') / 'bundle'\n"
                "shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)\n"
                "safe_extract_archive(zip_path, EXTRACT_ROOT)\n"
                "artifact_manifest, run_metadata = validate_artifact_directory(EXTRACT_ROOT)\n"
                "if (run_metadata['base_model'], run_metadata['base_model_revision'], run_metadata['weights_sha256'], run_metadata['config_sha256']) != (MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, CONFIG_SHA256):\n"
                "    raise RuntimeError('Bundle was not produced on the pinned base checkpoint carried by this notebook.')\n"
                "FEATURE_COLUMNS = list(run_metadata['features'])\n"
                "TARGET_COLUMN = run_metadata['target_column']\n"
                "print({{'artifact_source': artifact_source, 'digest_verified': bool(expected_digest), 'zip': zip_name, 'zip_sha256': zip_sha256, 'format': artifact_manifest['artifact_format'], 'format_version': artifact_manifest['artifact_format_version'], 'base_model': run_metadata['base_model'], 'base_revision': run_metadata['base_model_revision'][:12], 'problem_type': run_metadata['problem_type'], 'mode': run_metadata['mode'], 'selection_basis': run_metadata['selection_basis']}})\n"
                "print({{'target': TARGET_COLUMN, 'required_features': FEATURE_COLUMNS, 'producer_autogluon': run_metadata['autogluon_version'], 'producer_python': run_metadata['python_version'], 'data_source': run_metadata.get('data_source')}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In order: the whole-archive SHA-256 against the pinned sample digest (or `EXPECTED_ZIP_SHA256` for your own bundle); `safe_extract_archive`'s per-member path, symlink, size, expanded-size and compression-ratio checks; `validate_artifact_directory`'s manifest (every listed file present, no unlisted file, sizes and digests equal) and provenance checks; then the bundle's base model, revision and digests against the carried pinned values. A tampered archive fails the whole-archive digest first and the manifest digests second. None of them establishes who produced the archive or that Python deserialisation is safe: `TabularPredictor.load` executes trusted serialised state, which is why the digest is required by default.</details>"
            ),
        },
        {
            "md": (
                "## 5. Check runtime compatibility, then reconstruct the predictor from the bundle alone\n\n"
                "AutoGluon predictor state is Python-serialised, so the consumer's AutoGluon version must equal the "
                "producer's and the Python major/minor must match; both are asserted before `TabularPredictor.load` "
                "runs. The predictor is reconstructed from the extracted bundle only (AINF5): the base checkpoint bytes "
                "and the support context live inside it, and no network path is used (`HF_HUB_OFFLINE` is set by the "
                "Section 3 staging). The pinned snapshot `pipe` of Section 3 is not used for inference: only its file "
                "locations are passed to the serving wrapper, and every prediction comes from the bundle's own model state."
            ),
            "code": (
                "from autogluon.tabular import TabularPredictor\n\n"
                "AUTOGLUON_VERSION = importlib.metadata.version('autogluon.tabular')\n"
                "if run_metadata['autogluon_version'] != AUTOGLUON_VERSION:\n"
                "    raise RuntimeError(f\"Artifact requires AutoGluon {{run_metadata['autogluon_version']}}; runtime has {{AUTOGLUON_VERSION}}.\")\n"
                "artifact_python_mm = '.'.join(str(run_metadata['python_version']).split('.')[:2])\n"
                "runtime_python_mm = '.'.join(platform.python_version().split('.')[:2])\n"
                "if artifact_python_mm != runtime_python_mm:\n"
                "    raise RuntimeError(f\"Artifact was exported under Python {{run_metadata['python_version']}}; runtime is {{platform.python_version()}}. Use matching Python major/minor.\")\n"
                "predictor = TabularPredictor.load(str(EXTRACT_ROOT))\n"
                "if predictor.problem_type not in {{'binary', 'multiclass'}}:\n"
                "    raise RuntimeError(f'Expected a binary/multiclass classification predictor; loaded {{predictor.problem_type!r}}.')\n"
                "serving = MitraClassificationPipeline(weights_path=pipe.model_weight_path, config_path=pipe.config_path, snapshot_path=pipe.snapshot_path, device=pipe.device, source='artifact', predictor=predictor, features=FEATURE_COLUMNS)\n"
                "serving.target_column = TARGET_COLUMN\n"
                "serving.problem_type = str(predictor.problem_type)\n"
                "CLASSES = list(serving.class_labels)\n"
                "if run_metadata.get('class_labels') is not None and [str(label) for label in CLASSES] != list(run_metadata['class_labels']):\n"
                "    raise RuntimeError('Reconstructed class labels disagree with the validated provenance.')\n"
                "print({{'reconstructed_from': str(EXTRACT_ROOT), 'models': list(predictor.model_names()), 'features': len(FEATURE_COLUMNS), 'classes': CLASSES, 'problem_type': serving.problem_type, 'device': serving.device, 'source': serving.source}})"
            ),
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest\n\n"
                "With `NEW_DATA_PATH` empty the cell uses the **sample input**: the producer's independent test partition "
                "(114 rows), re-derived from scikit-learn's bundled Breast Cancer Wisconsin table with the seed recorded in the "
                "bundle and with the label column removed — rows the bundle's support set never contained. Set `NEW_DATA_PATH` to "
                "your own CSV in the runtime, or to `upload` for the Colab dialog (it needs no bundle-related setting). "
                "`validate_inputs(..., target_column=None, feature_columns=...)` is the package's public validation stage "
                "for inference tables: it applies exactly the checks `validate_inference_frame` applies — unique header "
                "(rejected by `read_csv_bytes` before pandas can rename duplicates), every required feature present, no "
                "pre-existing `prediction` column — and returns an **input manifest** naming the schema, the row count, "
                "the extra columns (preserved in the output, not passed to the model) and the missing-value columns; it "
                "is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a probe with one required column removed and records the package's own error message as a "
                "finding. Before reading the rows the cell prints the **inference-stage limits** — the required feature columns, "
                "the reserved output columns, and the fact that inference has no row ceiling — not the producer's training ceilings.\n\n"
                "**Predict:** the sample rows come from the producer's independent test partition. Could any of them be among the support rows stored inside the bundle? Then: the probe drops the first required feature column. Which rule refuses it, and does the refusal stop the notebook? What happens to a column in your CSV that the bundle never saw?"
            ),
            "code": (
                "import hashlib\n\n"
                "from sklearn.datasets import load_breast_cancer\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "# Location of the rows: empty = the sample input (the producer's independent test partition); a path = your CSV; 'upload' = the Colab dialog.\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "print({{'inference_limits': {{'required_features': len(FEATURE_COLUMNS), 'reserved_output_columns': ['prediction'] + [f'probability_{{c}}' for c in run_metadata.get('class_labels', [])], 'extra_columns': 'kept in the output, not passed to the model', 'row_ceiling': None}}}})\n"
                "sample_kind = 'BYOD'\n"
                "if not NEW_DATA_PATH.strip():\n"
                "    if run_metadata.get('data_source') != 'Sample: Breast Cancer':\n"
                "        raise RuntimeError(f\"The sample input fits only a bundle trained on 'Sample: Breast Cancer'; this bundle was trained on {{run_metadata.get('data_source')!r}}. Set NEW_DATA_PATH to unlabelled rows with its features.\")\n"
                "    dataset = load_breast_cancer(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({{0: 'malignant', 1: 'benign'}})\n"
                "    # The producer recorded the digest of exactly this table (E2E Section 4); a different table version is refused.\n"
                "    table_digest = hashlib.sha256(json.dumps({{'sample.csv': hashlib.sha256(frame.to_csv(index=False, lineterminator='\\n').encode('utf-8')).hexdigest()}}, sort_keys=True).encode()).hexdigest()\n"
                "    if table_digest != run_metadata.get('data_sha256'):\n"
                "        raise RuntimeError(f\"scikit-learn's bundled table (digest {{table_digest[:16]}}...) is not the table the bundle was produced from ({{str(run_metadata.get('data_sha256'))[:16]}}...): set NEW_DATA_PATH to your own unlabelled rows.\")\n"
                "    _, remainder = train_test_split(frame, test_size=0.4, random_state=run_metadata['seed'], stratify=frame[TARGET_COLUMN])\n"
                "    _, sample_rows = train_test_split(remainder, test_size=0.5, random_state=run_metadata['seed'], stratify=remainder[TARGET_COLUMN])\n"
                "    SAMPLE_LABELS = sample_rows[TARGET_COLUMN].reset_index(drop=True)  # kept aside; never passed to the model\n"
                "    csv_name = 'sample-independent-test-rows.csv'\n"
                "    csv_payload = sample_rows.drop(columns=[TARGET_COLUMN]).to_csv(index=False).encode('utf-8')\n"
                "    sample_kind = 'sample'\n"
                "    print({{'sample_input': csv_name, 'rows': len(sample_rows), 'derived_from': 'sklearn load_breast_cancer, the producer split (60/20/20, seed ' + str(run_metadata['seed']) + '), independent test partition', 'label_removed': TARGET_COLUMN}})\n"
                "elif NEW_DATA_PATH.strip().lower() != 'upload':\n"
                "    csv_name, csv_payload = os.path.basename(NEW_DATA_PATH.strip()), Path(NEW_DATA_PATH.strip()).expanduser().read_bytes()\n"
                "else:\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(\"NEW_DATA_PATH = 'upload' needs the Colab upload dialog, which this runtime does not have: copy the CSV into the runtime (or attach it as a Kaggle dataset) and set NEW_DATA_PATH to its path.\") from None\n"
                "    new_upload = files.upload()\n"
                "    csvs = [(name, payload) for name, payload in new_upload.items() if name.lower().endswith('.csv')]\n"
                "    if len(csvs) != 1:\n"
                "        raise RuntimeError('Upload exactly one inference CSV.')\n"
                "    csv_name, csv_payload = csvs[0]\n"
                "new_data = read_csv_bytes(csv_payload, csv_name)\n"
                "input_manifest = validate_inputs(new_data, None, feature_columns=FEATURE_COLUMNS, names=[csv_name])\n"
                "# Demonstrate rejection on a probe that breaks the fitted schema; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(new_data.drop(columns=[FEATURE_COLUMNS[0]]), None, feature_columns=FEATURE_COLUMNS)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-column-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "X_new, extra_columns = validate_inference_frame(new_data, FEATURE_COLUMNS)\n"
                "if extra_columns:\n"
                "    print('Extra columns preserved in the output but not passed to the model:', extra_columns)"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>No: the producer split the table 60/20/20 with the recorded seed and conditioned Mitra on the 60 % support partition only; re-deriving the split with the same seed gives back the independent test partition. `validate_inputs` refuses the probe with a `ValueError` naming the missing required feature, and the cell records the message as a finding in the input manifest, so the notebook continues. A column the bundle never saw is an *extra* column: it is kept in the output CSV and not passed to the model, and the manifest lists it.</details>'
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export\n\n"
                "`predict_proba` returns one column per class **in the predictor's stored class order** and `prediction` is the "
                "`argmax` label — **the probabilities are uncalibrated and no threshold is shipped by this pipeline**; for "
                "cost-sensitive decisions threshold the relevant `probability_<class>` column on your own labelled data. `evaluation_report` is the package's public evaluation stage and is produced even here: "
                "with no labelled rows its verdict is `not-measurable` and it states what labelled data would make the "
                "task measurable; it is written to `outputs/{stem}_evaluation_report.json`. The prediction CSV keeps "
                "every input column plus `prediction` and one `probability_<class>` column per class, and the result JSON records the externally supplied bundle "
                "identity (ZIP digest, manifest, provenance), the input manifest, the notebook's source, the pinned "
                "model identity, revision and licence, and the runtime identity; it contains no credentials.\n\n"
                "**Predict:** which verdict will the evaluation report give, and what would make the task measurable? Will the prediction CSV have more columns than the input CSV, and which?"
            ),
            "code": (
                "out = new_data.copy()\n"
                "proba = serving.predict_proba(X_new)\n"
                "out['prediction'] = serving.predict(X_new)\n"
                "for class_label in CLASSES:\n"
                "    out[f'probability_{{class_label}}'] = proba[class_label].to_numpy()\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "report = evaluation_report(None, n_holdout=0, class_labels=CLASSES, target_column=TARGET_COLUMN, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'artifact': {{'source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256, 'digest_verified': bool(expected_digest), 'manifest': artifact_manifest, 'run_metadata': run_metadata}},\n"
                "    'inference': {{'decisionRule': DECISION_RULE + ' over uncalibrated class probabilities', 'threshold': None, 'classLabels': [str(label) for label in CLASSES], 'problemType': serving.problem_type, 'extra_columns': extra_columns}},\n"
                "    'input': {{'filename': csv_name, 'rows': len(out), 'features': FEATURE_COLUMNS, 'kind': sample_kind}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'autogluon': AUTOGLUON_VERSION, 'lightgbm': importlib.metadata.version('lightgbm'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': serving.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(out.head())\n"
                "print(json.dumps(report, indent=2))\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>`not-measurable`: no labelled rows are passed here (the sample's labels are kept aside and never used), and the report states that a labelled copy of the rows with the bundle's target column would make the task measurable (the E2E tutorial's metrics helper does that). The prediction CSV keeps every input column and adds `prediction` plus one `probability_<class>` column per class in the predictor's stored class order — uncalibrated, with no threshold shipped.</details>"
            ),
        },
        {
            "md": (
                "## 8. Activity: change one feature and explain how the predictions move\n\n"
                "The cell multiplies one feature column by `ACTIVITY_SCALE` for every scored row, predicts again with the same "
                "predictor and reports how many predicted labels flip and how far the probability of each class moves on "
                "average. It writes nothing and changes nothing above it.\n\n"
                "**Predict:** the default doubles `worst area` (the area of the largest nuclei, larger for malignant tumours on this "
                "table). Will labels flip, and in which direction will `probability_malignant` move? *Change* `ACTIVITY_SCALE` to "
                "`0.5` (one field) and run this cell again. *Explain* the difference in two sentences — and why a label flip is not "
                "evidence that the model is right or wrong here."
            ),
            "code": (
                "ACTIVITY_FEATURE = 'worst area'  # @param {{type:\"string\"}}\n"
                "ACTIVITY_SCALE = 2.0  # @param {{type:\"number\"}}\n"
                "if ACTIVITY_FEATURE not in FEATURE_COLUMNS:\n"
                "    raise ValueError(f'ACTIVITY_FEATURE {{ACTIVITY_FEATURE!r}} is not a bundle feature; choose one of {{FEATURE_COLUMNS}}')\n"
                "if not pd.api.types.is_numeric_dtype(X_new[ACTIVITY_FEATURE]):\n"
                "    raise ValueError(f'ACTIVITY_FEATURE {{ACTIVITY_FEATURE!r}} is not numeric; choose a numeric feature')\n"
                "changed = X_new.copy()\n"
                "changed[ACTIVITY_FEATURE] = changed[ACTIVITY_FEATURE] * ACTIVITY_SCALE\n"
                "changed_proba = serving.predict_proba(changed)\n"
                "changed_prediction = np.asarray(serving.predict(changed))\n"
                "flips = changed_prediction != out['prediction'].to_numpy()\n"
                "shift = {{f'probability_{{c}}': round(float((changed_proba[c].to_numpy() - proba[c].to_numpy()).mean()), 4) for c in CLASSES}}\n"
                "print({{'feature': ACTIVITY_FEATURE, 'scale': ACTIVITY_SCALE, 'rows': len(changed), 'labels_flipped': int(flips.sum()), 'flips_to': pd.Series(changed_prediction[flips]).value_counts().to_dict(), 'mean_probability_shift': shift}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>Larger nuclei are characteristic of malignant samples in this table, so doubling `worst area` pushes `probability_malignant` up on average and the only flips go towards `malignant`; halving it moves the other way. The flips say how sensitive the model is to that one feature, not whether it is right: without labels nothing here is measured, and a doubled area is not a real patient.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that the supplied archive passed the package's path, symlink, size and manifest/digest "
        "checks, that its recorded model and runtime contract matched this notebook and the carried pinned base-model "
        "identity, that AutoGluon reconstructed the predictor from the bundle alone, that the new CSV satisfied the "
        "required feature schema, and that machine-readable classification predictions were produced — without the "
        "repository being reachable. It does **not** prove that the archive came from a trustworthy sender, that Python "
        "deserialisation is safe for untrusted artifacts, or that the predictor is accurate, calibrated, fair, robust, or "
        "suitable for a production decision; the evaluation report says `not-measurable` because no labels exist here. "
        "Never bypass a failed digest, path, manifest, base-model or schema check; obtain a correct trusted bundle.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate and reconstruct an external predictor bundle, validate "
        "the supplied inference table, execute the public prediction path and emit the shown machine-readable outputs in "
        "the tested runtime. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "| Failure | Meaning | Corrective action |\n|---|---|---|\n"
        "| whole-archive checksum mismatch | archive differs from the trusted digest | reject it and obtain the artifact again |\n"
        "| unsafe path/symlink/size failure | archive violates extraction safety limits | reject it |\n"
        "| missing/unlisted/digest-mismatched artifact file | archive is incomplete or altered | reject and reproduce/retransfer |\n"
        "| artifact format/version or base-model mismatch | consumer and producer contracts differ | use the matching notebook/runtime and a bundle built on the pinned base model |\n"
        "| AutoGluon/Python mismatch | Python serialisation compatibility is not established | use the recorded runtime |\n"
        "| missing required features / existing `prediction` column | input schema differs from the fitted predictor | add/rename the listed columns |\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file (`model.safetensors`, 302,717,904 bytes, or `config.json`). Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **"A trusted whole-archive SHA-256 is required"** — paste the digest the E2E tutorial printed into `EXPECTED_ZIP_SHA256`; `ALLOW_UNVERIFIED_ARTIFACT = True` is only for an already-trusted local bundle and prints a warning.\n'
        "- **The sample download fails or times out** — run Section 4 again (a complete download is reused); `github.com` release downloads must be reachable. The pinned SHA-256 is checked before extraction, so a partial or altered file is never used.\n"
        "- **\"The sample input fits only a bundle trained on 'Sample: Breast Cancer'\"** — you supplied your own bundle: set `NEW_DATA_PATH` to unlabelled rows with its features.\n"
        '- **"Predictor ZIP checksum mismatch" / an unsafe-path, size or digest failure / "Bundle was not produced on the pinned base checkpoint"** — the archive is not the one you trust, is altered, or was built on another base model: reject it and obtain the bundle again. Never bypass a failed check.\n'
        '- **"Artifact requires AutoGluon …" / "exported under Python …"** — the bundle is Python-serialised: use the runtime the producer used (the isolated environment pins AutoGluon 1.5.0 on CPython 3.12.12, the same as the E2E tutorial\'s).\n'
        "- **BYOD: \"… = 'upload' needs the Colab upload dialog\" / \"Upload exactly one …\"** — set `ARTIFACT_ZIP_PATH` and `NEW_DATA_PATH` to files in the runtime (Kaggle dataset, Drive, Jupyter); `upload` works only on Colab, and a cancelled dialog stops with that message. A bundle of about 280 MB is better copied than uploaded.\n"
        '- **A `ValueError` from `validate_inputs` or `read_csv_bytes`** — it names the rule: a duplicate header, a missing required feature, or a pre-existing `prediction` column. Add or rename the listed columns.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        "Run the Section 8 activity with a different feature or scale (one field; re-run Section 8 only); hand a labelled copy of the same rows to the E2E tutorial's metrics helper to obtain a `sample-sanity` report; compare bundles exported with `mode` pretrained versus fine-tuned on the same rows; drop one feature column from the CSV and read the refusal; set `ALLOW_UNVERIFIED_ARTIFACT = True` with an empty digest and read the warning it prints.\n"
        '\n'
        '## Glossary\n'
        '\n'
        "- **Predictor bundle** — the AutoGluon predictor directory the E2E tutorial exported, zipped: serialised model state, the registered support rows, `tutorial_run_metadata.json` (provenance) and `artifact_manifest.json` (every file's size and SHA-256).\n"
        '- **Whole-archive digest** — the SHA-256 of the ZIP as a whole, compared with the value you trust before anything is extracted.\n'
        '- **Safe extraction** — every member is checked for path traversal, symlinks, size and compression ratio before it is written; `extractall` is never used.\n'
        '- **Manifest and provenance checks** — every listed file present with the recorded size and digest, no unlisted file, and the recorded base model, revision, digests, AutoGluon and Python versions, target and features.\n'
        '- **Trust boundary** — digests prove integrity and consistency, not who produced the archive; `TabularPredictor.load` executes Python-serialised state, so load only bundles from a trusted producer.\n'
        "- **Runtime compatibility** — the consumer's AutoGluon version must equal the producer's and the Python major/minor must match before the predictor is loaded.\n"
        '- **Input manifest** — the validation record of the new rows: schema, row count, extra columns (kept in the output, not passed to the model), missing values and the verdict.\n'
        '- **`argmax` rule / `probability_<class>`** — the predicted label is the class with the highest uncalibrated probability; no threshold is shipped.\n'
        "- **`not-measurable`** — the evaluation report's verdict when no labelled rows exist; it names what labelled data would make the task measurable.\n"
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **Trusted sample bundle** — the bundle pinned in `SAMPLE_ARTIFACT` by release-asset URL and whole-archive SHA-256, produced by the E2E tutorial in a recorded run; the default path verifies it before extraction.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        'Before you leave, write three lines in this cell: (1) which checks ran before `TabularPredictor.load` and which property (sender authenticity, deserialisation safety) none of them establishes; (2) what the `not-measurable` verdict means for the predictions you exported; (3) what you would need (labelled rows, a matching runtime) before trusting them.\n'
        '\n'
        "**Next experiments (summary):** hand a labelled copy of the same rows to `classification_metrics` in the E2E tutorial to obtain a "
        "`sample-sanity` report; compare bundles exported with `mode` pretrained versus fine-tuned on the same rows.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- E2E companion (produces the bundle): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/mitra_classifier_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/autogluon/autogluon\n"
        "- Mitra paper: https://arxiv.org/abs/2508.02927"
    ),
}
