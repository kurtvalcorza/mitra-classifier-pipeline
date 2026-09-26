# Release verification

`tutorials/mitra_classifier_colab.ipynb` (`E2E`) and `tutorials/mitra_classifier_predictor_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the durable
release-gate record for both generated standalone notebooks. The separate FreshRetailNet multi-model workshop is an auxiliary workshop carrier: it is statically validated by `tools/validate_release_assets.py`, but it requires its own clean-runtime execution evidence before teaching and does not inherit this pair's release evidence.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which preserves full generator/parity checks for the two generated standalone notebooks and separately validates the explicitly allowlisted comparative workshop:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- the tutorial directory contains exactly the two generated standalone notebooks plus explicitly allowlisted auxiliary notebooks; the generated pair remains named in `tutorials/README.md` with its profile, notebook-spec version and standalone carrier, while the comparative workshop is registered separately as `E2E` / `WORKSHOP`, Notebook Spec `2.1`, with a standalone embedded-adapter carrier; it does not inherit generated-notebook parity evidence;
- the original auxiliary FreshRetailNet workshop requires no committed outputs or execution counts; the v2 executed-record exception retains outputs and counts and rejects saved errors. For both, every code cell and its embedded isolated runner parse as Python, `metadata.dimer` records `E2E` / `WORKSHOP`, spec `2.1`, `standalone: true`, and the pinned dataset SHA / balanced-accuracy / `uv`-managed Python-3.12 / identity-bound cache / frozen-artifact verification markers are present;
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path
  (the previous pair's `git clone` of this repository is gone); one cell tagged `embedded_module` equal to
  `mitra_pipeline/tutorial_api.py` after the generator's documented rewrite (the `DEFAULT_WEIGHTS_DIR` line); the inline
  `MANIFEST` equal to the committed `weights/mitra-classifier/dimer-base-manifest.json` and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output for its template; the
  pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity
  string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls — E2E: `stage_missing_files`, `verify_snapshot`,
  `MitraClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` (which stages the verified bytes into the offline HF
  cache through `stage_verified_hf_snapshot`), `validate_inputs` (with the too-few-rows rejection probe),
  `validate_labeled_frame`, `split_overlap_report`, `cap_training_rows`, `majority_class_baseline`, `fit` (in-context,
  `fine_tune=False`), the GPU-guarded `fine_tune=True` candidate behind `RUN_FINE_TUNING`, majority-class, LightGBM /
  Random Forest on the same partitions, `evaluation_report` with the independent test, inference-mode `validate_inputs` +
  `validate_inference_frame`, `write_artifact_manifest`, `safe_extract_archive` + `validate_artifact_directory` before
  `TabularPredictor.load` on the fresh reload and the `rtol=1e-6, atol=1e-8` equivalence check; companion: the required
  whole-archive digest (`ALLOW_UNVERIFIED_ARTIFACT` gated off), `safe_extract_archive`, `validate_artifact_directory`, the
  base-model identity/digest comparison, the AutoGluon/Python compatibility checks before `TabularPredictor.load`,
  inference-mode `validate_inputs` with a rejection probe, `predict`, a `not-measurable` `evaluation_report` — the ceiling
  prints, the four exports per notebook, the learner-facing statements (uncalibrated probabilities / argmax rule, class coverage, dropped rows counted, no
  gradient update, fine-tuning gate, reproducibility boundary, verify-before-deserialise, trust boundary, no artifact created
  in the companion) and the gated-off form parameters (`USE_BYOD`, `RUN_FINE_TUNING`, `RUN_NEW_DATA_INFERENCE`,
  `ALLOW_UNVERIFIED_ARTIFACT`); forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://`
  dependency without a 40-hex SHA, a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside
  the install cell, direct `urllib.request` / `TabularPredictor(` / `predictor.fit(` / `hyperparameters=` / `zipfile.ZipFile(`
  / `hf_hub_download(` / `sklearn.metrics` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`; in the companion also `load_breast_cancer(` / `load_wine(` / `majority_class_baseline(`, `shutil.make_archive(`, `.fit(`, `write_artifact_manifest(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (license, base model), single H1, placeholder/claim hygiene and the
  `## Checkpoint and Artifact Provenance` section (the card predates the MODEL_CARD_SPEC heading structure; the structural
  checks are switched off in the validator until the fleet's card pass reaches this branch).

CI also runs `ruff`, `tools/build_notebook.py --check` for both templates, `scripts/check_shared.py`,
`scripts/check_contract.py`, `scripts/validate_colab_tutorial.py` (the repository's own spec-1.1 checks), the two
`scripts/test_*.py` classification suites (`test_tutorial_api.py` against the package, `test_colab_csv_headers.py` against the
package and the generated notebooks) and the offline unit suite (`tests/`: `test_role_helpers.py`, `test_notebook_parity.py`,
`test_companion_parity.py`; injected downloader, no weights, AutoGluon never imported). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime, Python 3.10–3.13 | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle kernel, Python 3.10–3.13 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone). For the companion the shim also places the E2E run's `outputs/mitra_classifier_predictor.zip`, its printed SHA-256 and a separately generated unlabelled CSV, and sets `ARTIFACT_ZIP_PATH` / `EXPECTED_ZIP_SHA256` / `NEW_DATA_PATH` |
| GitHub Actions `notebook-release.yml` (previous pair) | hosted runner, real kernels | Executed the previous repository-installing pair (`/content` workspace, `DIMER_*` / `MITRA_*` env vars); it must be re-pointed at the standalone pair (`outputs/` workspace, the form parameters above) before it counts again |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new CPU (or CUDA) runtime with **no repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults: `DATA_SOURCE = 'Sample: Breast Cancer'`,
   `USE_BYOD = False`, `RUN_FINE_TUNING = False`, `EVAL_METRIC = 'accuracy'`, `RUN_NEW_DATA_INFERENCE = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`:
   `autogluon.tabular[mitra]==1.5.0`, `lightgbm==4.6.0`, `huggingface-hub==0.36.2`; torch and friends are AutoGluon's transitive pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `MitraClassificationPipeline`, the validation/metric helpers and the archive-safety
     functions) with no import of the repository package;
   - pinned `autogluon/mitra-classifier` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/mitra-classifier/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)`
     reports the two manifest entries (`model.safetensors` 302,717,904 bytes, `config.json` 81 bytes) on a clean runtime,
     `verify_snapshot` returns the manifest dict, `from_pretrained` reports `source == 'local-snapshot'` and the offline HF
     snapshot path;
   - the Breast Cancer sample split 60/20/20 with stratification (`train=341, holdout=114, test=114`, two classes) with its data SHA-256 printed; the ceilings (`MIN_TRAIN_ROWS` 50, `MAX_TRAIN_ROWS` 10,000,
     `MAX_FEATURES` 500) surfaced; `validate_inputs` writes `outputs/mitra_classifier_input_manifest.json` (three accepted tables,
     one recorded rejection finding from the too-few-rows probe); overlap and cap reports printed;
   - `fit` (in-context conditioning through AutoGluon), `classification_metrics` on holdout and independent test, `pipe.evaluate`,
     `majority_class_baseline`; the fine-tuning gate skipped; the majority-class predictor, LightGBM and Random Forest (probabilities aligned to the Mitra class order) on the same rows;
   - `evaluation_report` writes `outputs/mitra_classifier_evaluation_report.json` with verdict `sample-sanity`, the three metric ids,
     the independent-test block and the majority-class baseline;
   - eight held-out rows scored into `outputs/mitra_classifier_predictions.csv` (inference upload skipped);
   - the predictor bundle `outputs/mitra_classifier_predictor.zip` written with `tutorial_run_metadata.json` and `artifact_manifest.json`,
     extracted with `safe_extract_archive` into `outputs/artifact-reload/`, `validate_artifact_directory` passes before
     `TabularPredictor.load`, and the reloaded predictor's predictions equal the in-memory model's within `rtol=1e-6, atol=1e-8`;
     `outputs/mitra_classifier_result.json` written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. in a **second** clean runtime, run the exact companion notebook revision with `ARTIFACT_ZIP_PATH` pointing at a copy of the E2E
   run's bundle, `EXPECTED_ZIP_SHA256` set to its printed digest and `NEW_DATA_PATH` at a separately generated unlabelled CSV (or
   supply the files through the upload dialog); verify the digest, archive, manifest, base-model and runtime-compatibility checks
   pass before `TabularPredictor.load`, that `validate_inputs(..., target_column=None, ...)` writes
   `outputs/mitra_classifier_predictor_inference_input_manifest.json` with one recorded rejection finding, and that the
   `not-measurable` evaluation report, `..._predictions.csv` and `..._result.json` are written;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, AutoGluon, device), model identifier and immutable
   revision, whether the model cache was clean, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `0e13847` / `9599d5c4832c` | Kaggle T4 (`kurtvalcorza/dimer-nb2-mitra-classifier` v2) | Standalone E2E default sample path | 215.5 s | **PASSED** — 9/9 ok code cells executed cleanly, 9 files, 605 MB staged |
| | | | Standalone ARTIFACT-INFERENCE with an external bundle | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the standalone notebooks has been recorded yet; clean GPU execution evidence for the E2E path is now recorded below. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blobs
under review and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch
into a fresh `weights/mitra-classifier/` has not been executed); `from_pretrained` builds no predictor — AutoGluon loads the
model inside `fit`; the previous workflow executions covered the old repository-installing notebooks, not this carrier; the
default sample changed from the repository-hosted FreshRetailNet archive to scikit-learn's Breast Cancer Wisconsin table (the repository
archives are reachable only through the upload path now); and the standalone carrier itself — executing the carried module
cell in a runtime that has no repository checkout — has been validated statically only (parity PASS, carrier probe with the
package import blocked), never run. The clean runs will be the first execution of the standalone path, of the staging path,
of the new helpers (`validate_inputs`, `majority_class_baseline`, `evaluation_report`) and of `MitraClassificationPipeline`
against the real weights.

## FreshRetailNet Classification v2: saved Colab execution and guided update

Evidence recorded on 2026-09-26; **Candidate**, not a release promotion. This section applies only to `tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb`, not the original workshop or generated tutorial pair.

| Evidence | Recorded fact / limit |
|---|---|
| Original notebook commit | `33602f8952dbcab61f3b19925d36ffc4f696bc28` |
| Original notebook Git blob | `d8a4c6290b29261a44364dc0b42a483dc7af4b3d` |
| Execution report | Maintainer confirmed a fresh Colab runtime and default Run all, with no manual restarts or rerunning cells. Saved outputs were independently inspected; execution was not independently rerun. The precise run timestamp is not recorded. |
| Saved outputs inspected | 22/22 code cells have execution counts and outputs; zero saved error outputs. Terminal summary records `pinned-public-sample`, 9 validation models, 11 frozen test models, no foundation failures, and completion of validation, freeze, artifact reload, independent test evaluation, inference preview, and export. |
| Declared default configuration | `USE_BYOD=False`; four in-context foundation models enabled; optional fine-tuned conditions disabled; classical ablations enabled; optional foundation ablation disabled; freeze, test evaluation, and ZIP export enabled. These saved source settings match the maintainer-confirmed fresh default Run all. |
| Runtime boundary | Maintainer-confirmed fresh Google Colab execution; the notebook uses isolated Python 3.12 model environments. Fresh/default execution provenance is established by that confirmation for the original code revision. A complete runtime/device/package inventory should accompany the next exact-revision run; no independent rerun is claimed. |
| Revised guided layer | Markdown adds Input → Model → Output, section roles, predictions before results, and a bounded validation-only ablation exercise. Every code-cell object, including source, output, count and metadata, is retained from the original blob. Saved outputs therefore document the original execution, not a new run of the revised notebook. |
| Local BYOD evidence | Actual v2 acquisition and validation code accepted a path-based compatible ZIP (60 rows per split, 17 numeric features, all three classes) and rejected a ZIP whose validation CSV lacked `lag_1` with `val: schema mismatch`. Only existing form assignments were overridden in memory; helpers were extracted from the notebook. No models were executed. This is validation-only evidence, not full REL12. |

### Remaining exact-revision release checks

1. Record the revised notebook commit/blob, a fresh supported Colab runtime, Python/package/device inventory, clean model-cache status, and unchanged default controls. Run all and retain the full executed notebook plus exported report. Inspect the final summary and every selected model result, not just the terminal message.
2. For positive BYOD coverage, use a separate fresh runtime and a lawful representative retail dataset with pre-split `train.csv`, `val.csv`, and `test.csv`. Match the exact ordered 17-feature schema from Section 2.1 plus `target`, use finite numeric features, and include `low`, `mid`, and `high` in every split. Preserve leakage-aware temporal splits and document how the training-only band edges were obtained. Do not use the 60-row parser fixture as proof of real model compatibility.
3. Stage that ZIP outside the notebook workspace, for example `/content/classification-byod.zip`. In Section 1.1 set `USE_BYOD=True`, `BYOD_METHOD="path"`, and `BYOD_ZIP_PATH="/content/classification-byod.zip"`. Leave the other defaults unchanged. Execute the full downstream path: acquisition, schema/class validation, baseline fits, all four selected foundation models, validation comparison, default classical ablations, freeze, saved-artifact reload, independent test scoring, inference preview, and export. Require no foundation failures; record ZIP/split hashes, actual row counts, package/device inventory, model outcomes and export inventory.
4. In another fresh session, copy the valid ZIP and remove `lag_1` from `val.csv`. Select that file through the same path controls and run through Section 2.1. Require `val: schema mismatch` before any model fit; save the exact error and invalid-fixture digest. This negative check must fail, so do not count its intentional error as a successful Run all.
5. Retain both BYOD records against the exact revised blob and reconcile the registry only after reviewing their outcomes. Do not claim REL12 from validation-only checks. The original default-path record combines maintainer-confirmed fresh execution with saved-output inspection; it does not establish a rerun of the revised prose blob.

The optional learning activity stops before freeze/test and leaves the canonical default unchanged. A learner who has seen test outcomes must not reuse them to reselect models or claim a fresh unbiased evaluation.
