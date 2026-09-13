"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The
notebook carries the same package module and the same pinned snapshot as the E2E notebook; it consumes an
AutoGluon predictor bundle (`mitra_classifier_predictor.zip`) produced by a *separate* execution (upload, or an
explicit path for non-interactive executors) and never creates one.
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
    **{k: _E2E[k] for k in ("package", "package_dir", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module", "model_load", "runtime_imports")},
    "stem": "mitra_classifier_predictor_inference",
    "notebook_name": "mitra_classifier_predictor_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
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
        "bundle alone, accepts genuinely new unlabelled rows, predicts continuous class labels with probabilities, and exports results. "
        "**No artifact is created here**, nothing is trained, and the pinned base checkpoint verified in Section 3 is "
        "not reacquired for inference — it exists so the bundle's recorded base-model digests can be checked against "
        "known-good values.\n\n"
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
        "continuous class labels with probabilities, produce an evaluation report that is `not-measurable` because no labels exist, "
        "and export machine-readable predictions plus provenance."
    ),
    "exclusions": (
        "artifact creation, in-notebook fitting or fine-tuning, regression, calibrated probabilities, or any quality "
        "claim: without labelled rows nothing is measured, and the exported probabilities are uncalibrated model outputs with no "
        "prediction interval."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.10–3.13 as required by AutoGluon 1.5.0); the AutoGluon and Python major/minor versions must match the ones recorded in the bundle. The default path runs on CPU and uses CUDA automatically when available.",
        "- **Artifact:** an externally produced AutoGluon predictor bundle (the E2E tutorial writes `outputs/mitra_classifier_predictor.zip` and prints its SHA-256). Supply it through the upload dialog, or set `ARTIFACT_ZIP_PATH` to a file already present in the runtime for non-interactive execution; paste its trusted SHA-256 into `EXPECTED_ZIP_SHA256`. Nothing in this notebook manufactures it.",
        "- **Data:** one separate, unlabelled UTF-8 CSV with the bundle's required feature columns. It is supplied by upload or by `NEW_DATA_PATH`; no sample is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Supply the external bundle and validate it before any deserialisation\n\n"
                "Leave `ARTIFACT_ZIP_PATH` empty to upload the ZIP; set it to a file already in the runtime to skip the "
                "dialog (an executor places the file there). A trusted whole-archive SHA-256 in `EXPECTED_ZIP_SHA256` is "
                "required by default; `ALLOW_UNVERIFIED_ARTIFACT=True` waives it only for an already-trusted local "
                "bundle and prints a warning. `safe_extract_archive` extracts only after every member passed the path, "
                "symlink, per-member size, expanded-size and compression-ratio checks (AINF3), and "
                "`validate_artifact_directory` verifies the bundle's digest manifest (every listed file present, no "
                "unlisted file, sizes and SHA-256 equal) and its `tutorial_run_metadata.json` provenance (artifact "
                "format and version, base model, revision and digests, AutoGluon/Python versions, target, features, mode, "
                "selection basis) before anything is deserialised (AINF4). The bundle's base-model identity and digests "
                "are also compared with the carried package's pinned values, so a bundle built on another base model is "
                "refused."
            ),
            "code": (
                "import shutil\n\n"
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "ALLOW_UNVERIFIED_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if ARTIFACT_ZIP_PATH:\n"
                "    zip_name, zip_payload = os.path.basename(ARTIFACT_ZIP_PATH), Path(ARTIFACT_ZIP_PATH).read_bytes()\n"
                "    artifact_source = f'path: {{ARTIFACT_ZIP_PATH}}'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    zips = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.zip')]\n"
                "    if len(zips) != 1:\n"
                "        raise RuntimeError('Upload exactly one predictor bundle ZIP.')\n"
                "    zip_name, zip_payload = zips[0]\n"
                "    artifact_source = 'upload dialog'\n"
                "zip_path = Path('external-artifact') / Path(zip_name).name\n"
                "zip_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "zip_path.write_bytes(zip_payload)\n"
                "zip_sha256 = sha256_file(zip_path)\n"
                "expected_digest = EXPECTED_ZIP_SHA256.strip().lower()\n"
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
                "print({{'artifact_source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256, 'format': artifact_manifest['artifact_format'], 'format_version': artifact_manifest['artifact_format_version'], 'base_model': run_metadata['base_model'], 'base_revision': run_metadata['base_model_revision'][:12], 'problem_type': run_metadata['problem_type'], 'mode': run_metadata['mode'], 'selection_basis': run_metadata['selection_basis']}})\n"
                "print({{'target': TARGET_COLUMN, 'required_features': FEATURE_COLUMNS, 'producer_autogluon': run_metadata['autogluon_version'], 'producer_python': run_metadata['python_version'], 'data_source': run_metadata.get('data_source')}})"
            ),
        },
        {
            "md": (
                "## 5. Check runtime compatibility, then reconstruct the predictor from the bundle alone\n\n"
                "AutoGluon predictor state is Python-serialised, so the consumer's AutoGluon version must equal the "
                "producer's and the Python major/minor must match; both are asserted before `TabularPredictor.load` "
                "runs. The predictor is reconstructed from the extracted bundle only (AINF5): the base checkpoint bytes "
                "and the support context live inside it, and no network path is used (`HF_HUB_OFFLINE` is set by the "
                "Section 3 staging). The pinned snapshot `pipe` of Section 3 is not used for inference."
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
                "Leave `NEW_DATA_PATH` empty to upload one CSV, or set it to a file already in the runtime. "
                "`validate_inputs(..., target_column=None, feature_columns=...)` is the package's public validation stage "
                "for inference tables: it applies exactly the checks `validate_inference_frame` applies — unique header "
                "(rejected by `read_csv_bytes` before pandas can rename duplicates), every required feature present, no "
                "pre-existing `prediction` column — and returns an **input manifest** naming the schema, the row count, "
                "the extra columns (preserved in the output, not passed to the model) and the missing-value columns; it "
                "is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a probe with one required column removed and records the package's own error message as a "
                "finding. The ceilings the producer enforced are printed for reference."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "if NEW_DATA_PATH:\n"
                "    csv_name, csv_payload = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "else:\n"
                "    new_upload = files.upload()\n"
                "    csvs = [(name, payload) for name, payload in new_upload.items() if name.lower().endswith('.csv')]\n"
                "    if len(csvs) != 1:\n"
                "        raise RuntimeError('Upload exactly one inference CSV.')\n"
                "    csv_name, csv_payload = csvs[0]\n"
                "print({{'ceilings': {{'MIN_TRAIN_ROWS': MIN_TRAIN_ROWS, 'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES}}, 'required_features': FEATURE_COLUMNS}})\n"
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
                "## 7. Predict, report what cannot be measured, and export\n\n"
                "`predict_proba` returns one column per class **in the predictor's stored class order** and `prediction` is the "
                "`argmax` label — **the probabilities are uncalibrated and no threshold is shipped by this pipeline**; for "
                "cost-sensitive decisions threshold the relevant `probability_<class>` column on your own labelled data. `evaluation_report` is the package's public evaluation stage and is produced even here: "
                "with no labelled rows its verdict is `not-measurable` and it states what labelled data would make the "
                "task measurable; it is written to `outputs/{stem}_evaluation_report.json`. The prediction CSV keeps "
                "every input column plus `prediction` and one `probability_<class>` column per class, and the result JSON records the externally supplied bundle "
                "identity (ZIP digest, manifest, provenance), the input manifest, the notebook's source, the pinned "
                "model identity, revision and licence, and the runtime identity; it contains no credentials."
            ),
            "code": (
                "out = new_data.copy()\n"
                "proba = serving.predict_proba(X_new)\n"
                "out['prediction'] = serving.predict(X_new)\n"
                "for class_label in CLASSES:\n"
                "    out[f'probability_{{class_label}}'] = proba[class_label].to_numpy()\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "report = evaluation_report(None, n_holdout=0, class_labels=CLASSES, target_column=TARGET_COLUMN, sample_kind='BYOD')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'artifact': {{'source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256, 'digest_verified': bool(expected_digest), 'manifest': artifact_manifest, 'run_metadata': run_metadata}},\n"
                "    'inference': {{'decisionRule': DECISION_RULE + ' over uncalibrated class probabilities', 'threshold': None, 'classLabels': [str(label) for label in CLASSES], 'problemType': serving.problem_type, 'extra_columns': extra_columns}},\n"
                "    'input': {{'filename': csv_name, 'rows': len(out), 'features': FEATURE_COLUMNS}},\n"
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
        "**Next experiments:** hand a labelled copy of the same rows to `classification_metrics` in the E2E tutorial to obtain a "
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
