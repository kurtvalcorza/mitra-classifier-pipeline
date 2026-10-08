# Release verification

## Customer analytics capstone — Candidate

The new `DIMER_Small_Business_Customer_Analytics_Capstone.ipynb` has its own
[implementation and evidence record](customer-capstone.md). It inherits no execution evidence
from FreshRetailNet. Source/data checks and CPU tests do not qualify its pretrained Mitra path.
Required next evidence: exact-revision fresh Colab T4 default Run all, exported verification bundle,
measured resources, and representative BYOD evaluation plus artifact-inference runs.

Review fixes (Notebook Review Framework v1, 2026-09-28; see the
[finding table](customer-capstone.md#notebook-review-framework-v1-findings-and-fixes-2026-09-28)):
source-checked and CPU-tested only. The real `prepare` stage was rerun on the pinned UCI archive
(SHA-256 `572e3627…7bfb`) for the reviewed and the fixed revision on CPU. All eight results files
and the decompressed prepared transactions (`ec29448e…78e3db`) are byte-identical, and the stage
receipts differ only in source identity, pid and seconds. The new teaching timeline reconciles to
its snapshot on the workbook. Later stages were not rerun on the workbook.
No pretrained Mitra, GPU or hosted execution was run; the fixed revision still needs its own fresh
Colab T4 Run all and representative BYOD evaluation and artifact-inference runs.

### Maintainer-supplied Colab execution of revision `c81db75` — 2026-09-28

| Evidence | Recorded fact / limit |
|---|---|
| File | [`execution-evidence/2026-09-28/DIMER_Small_Business_Customer_Analytics_Capstone_c81db75.ipynb`](execution-evidence/2026-09-28/DIMER_Small_Business_Customer_Analytics_Capstone_c81db75.ipynb), byte-for-byte copy, SHA-256 `f462d4495e707c1f5b2d484528eb4ec13d0baac8bcd8fb846f31fa87a1f17d07` |
| Source match | Executed against commit `c81db75`, notebook blob `5db77406b629`. All 21 cells match in id, order and source, except the `# @title` line Colab inserts into the two collapsed infrastructure cells, which is non-substantive. `RUN_BYOD=False` (default). |
| Runtime | Colab Tesla T4. Isolated Python 3.12.12 with the hash-locked torch 2.9.1, AutoGluon 1.5.0, pandas 2.3.3, NumPy 2.3.5 and scikit-learn 1.7.2. Initial free disk was 202 GB. |
| Executed cells | Execution counts 1–10, sequential, with no error outputs. All eight stages report PASS. |
| Resources | 10.4 min wall time including setup before the report; stage seconds excluding install were 524 (`prepare` 340, `develop` 66, `evaluate` 50, `infer` 29, `verify` 35). Peak allocated GPU memory was 0.72 GB. All three targets were met: under 45 min, at least 20 GiB free disk, and at most 12 GiB GPU memory. |
| Cohorts | Identical to the frozen audit: 4 × 512 support rows (155/168/188/122 positive); development 270 and 321 positive of 1,000; test 379 and 452 positive of 1,000. |
| Held-out results (20% budget, equal-cutoff macro) | Mitra-RFM and Logistic-RFM both reach P@20% 0.735 (Sep 0.710, Nov 0.760), lift 1.78, AP 0.660 vs 0.659 and AUROC 0.704 vs 0.702. The primary paired contrast, Mitra-RFM minus Logistic-RFM P@20%, is **0.000, 95% CI [−0.015, 0.010]**: no measurable advantage. RFM adds about 0.23 over recency for both models (Mitra-RFM − Mitra-R 0.225 [0.181, 0.287]). All 2,000 bootstrap replicates are valid. |
| Reconstruction | `verify` ran in a different process and replayed 1,000 final customers × 6 systems, with exact labels and top-k at atol 1e-5 and rtol 1e-4. |
| Review fixes observed | Compact tables replaced inline JSON, with the full JSON collapsed. The development RFM-minus-R table and the ranked Mitra-RFM review list (200 of 1,000 selected) are shown. The timeline reconciles to the stored snapshot. Default-mode limitations are unchanged. No table was truncated. |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated. This single default run does not exercise BYOD evaluation, artifact inference or the invalid-timestamp path. |

| Journey | Verdict |
|---|---|
| Default Run all (fresh T4) | **Pass** |
| Fresh-process reconstruction | **Pass** |
| BYOD evaluation | Not assessed in this run |
| BYOD artifact inference | Not assessed in this run |
| Invalid-then-corrected BYOD timestamp | Not assessed in this run |

Remaining before promotion: representative BYOD evaluation and artifact-inference runs under the pinned
lock, a learner observation and maintainer review. Status remains **Candidate**.

### Notebook source layout change (2026-10-02)

The infrastructure cell `customer-03` used to hold all carried files on one 235,880-character source
line. The generator now writes each carried string as short implicitly concatenated pieces (at most
1,000 characters each); the longest notebook line is now 637 characters. The ten carried files are
unchanged byte for byte, and so are their SHA-256 values in `source.json`. The only carried difference
is `source.json`'s `builder_sha256`, which records the generator's own hash and changes whenever the
generator changes. The notebook blob changes from `5db77406b629` to `0eade8804f02`. The hosted run
above executed blob `5db77406b629`; the new blob was re-run on 2026-10-03 (below).

### Colab CLI execution of revision `9698515` (blob `0eade880`) — 2026-10-03

| Item | Record |
| --- | --- |
| File | [`execution-evidence/2026-10-03/DIMER_Small_Business_Customer_Analytics_Capstone_9698515_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Small_Business_Customer_Analytics_Capstone_9698515_colab-cli-t4.ipynb), SHA-256 `c53a3f6d6559fe1cf670beca70dc65a5ce6924b12b17c8bf791ab1717d91f34b`, byte-for-byte copy of the CLI's output notebook |
| Executor | Google Colab CLI 0.7.4 (`colab new --gpu T4`, `colab exec -f`, `colab stop`) from WSL, on a fresh Colab Tesla T4 session (`Shape: Standard`). This is sequential execution of every code cell in one kernel, not a browser Run all; the CLI does not record `execution_count`, so order is evidenced by its `Executing cell k/10` log |
| Source match | The notebook was downloaded from GitHub at `9698515` and its git blob `0eade8804f02` checked before the session was created. All 21 cells match the PR head in id, order and source |
| Result | **PASSED**: 10/10 code cells, no error output, stages prepare / fit / develop / freeze / evaluate / infer / verify / report all PASS; wall 638 s (10.44 min including setup) |
| Equivalence | Every printed stage result equals the 2026-09-28 run of blob `5db77406b629`; only the dependency-resolve time and wall time differ. The carrier split changed no runtime behaviour |
| Boundary | Saved outputs were inspected; BYOD was not exercised. Status remains **Candidate** |

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
  isolated-environment bootstrap cell (generator /2.2: hash-locked `uv` environment, nothing installed into the kernel, no restart); `NOTEBOOK_SOURCE` recorded in exports;
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
   - pinned runtime installed from the inline `PINS` (hash-locked, into the isolated environment of Section 1) with no GitHub access;
   - the carried module cell executes (defines `MitraClassificationPipeline`, the validation/metric helpers and the archive-safety
     functions) with no import of the repository package;
   - pinned `autogluon/mitra-classifier` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/mitra-classifier/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)`
     reports the two manifest entries (`model.safetensors` 302,717,904 bytes, `config.json` 86 bytes) on a clean runtime,
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
| | | | Standalone ARTIFACT-INFERENCE, default path (pinned `sample-bundle-v1` + sample rows) | | pending for the current blob |
| 2026-10-08 | `83b4d0c` / `2748c0a2e7f8` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Standalone E2E default sample path (no field edited; optional journeys not exercised) | 170.4 s | **PASSED** — one pass, no restart, 0 errors: 10/10 code cells in order (`exec.log`), isolated environment built in 83 s; evidence in `docs/execution-evidence/2026-10-08/mitra_classifier_colab/` |
| 2026-09-28 | `c81db75` / `5db77406b629` | Colab Tesla T4 (maintainer-supplied) | Customer analytics capstone, default Run all | 10.4 min | **PASSED**: 10/10 code cells, 8/8 stages, fresh-process verification passed; BYOD not exercised |
| 2026-10-03 | `9698515` / `0eade8804f02` | Colab CLI 0.7.4, fresh Colab Tesla T4 | Customer analytics capstone, all code cells in order | 638 s | **PASSED**: 10/10 code cells, 8/8 stages; results equal the 2026-09-28 run |

### Colab CLI execution of the E2E notebook at `83b4d0c` (blob `2748c0a2e7f8`) — 2026-10-08

- **Executor:** Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM (`colab exec -f`): every code cell in order in one
  kernel, order taken from `exec.log` ("Executing cell k/10"). Not a browser **Run all**; no execution counts; forms not rendered.
- **Notebook:** `tutorials/mitra_classifier_colab.ipynb`, commit `83b4d0c738ef1d207e4900921e954e688fb64c31`, blob
  `2748c0a2e7f84c4c2799a7ce0673f984686e33f0` (fetched byte-exact at the commit; the executed copy's code cells equal the source).
- **Path:** default sample path only, no field edited. Section 1 built the isolated uv environment (74 locked packages, Python
  3.12.12; kernel Python 3.13.15) in 83 s; the model ran on `cuda`.
- **Outcome:** **one pass, no restart, 0 errors**; 10/10 code cells; the carried-module cell has no output by design. Wall 170.4 s.
- **Printed results:** partitions 341 / 114 / 114; Mitra conditioned on 272 of the 341 support rows (69 kept by AutoGluon for its
  internal `Validation score` 0.9565); positive class `malignant`. Holdout rows correct of 114: Mitra 112, LightGBM 110,
  Random Forest 109, majority class 71; test: 112 / 111 / 112 / 72. Mitra holdout accuracy 0.9825, log loss 0.0497, ROC-AUC 0.9993.
  Evaluation verdict `sample-sanity`; `selection_basis` `default:pretrained`; fresh reload **PASS** (labels identical,
  probabilities within `rtol=1e-6, atol=1e-8`).
- **Exported bundle:** `outputs/mitra_classifier_predictor.zip`, 280,280,389 bytes, SHA-256
  `d2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01` (printed by the run; the downloaded copy has the same digest).
  It is the source of the `sample-bundle-v1` release asset used by the predictor-inference notebook.
- **Evidence files** (`docs/execution-evidence/2026-10-08/mitra_classifier_colab/`, byte-exact, covered by the `-text` rule):
  executed notebook `518e27b3d78199e5302cb937e3cf6157ac04b477647e6479d680c9188a7d55e0`, `run_summary.json` `a96513e2eee6d28b031187d25e28f213f576b7fd41ee9fb2bffb2f6f566631c8`, `exec.log` `75bfe90bdfc0da8dde6d463fa8fed6ab61e18bfe242264f6799fc7a242b1f1c3`, `mitra_classifier_result.json` `9a5280b7cb8554c43d8397fccefe9fbcc2e6daf58342692a881e67a387de0cdb`.
- **Not exercised:** browser Run all, BYOD (single CSV and pre-split), `RUN_FINE_TUNING` on GPU, new-data inference upload,
  the Wine activity. The worked answers were checked against this run (NOTEBOOK_SPEC REL13): they hold; the Section 5 answer's
  "about 0.63" majority-class accuracy is 0.6228 on the holdout here (rounding).

## Current status

**Generated tutorial pair: Candidate.** The E2E notebook's current blob `2748c0a2e7f8` has one recorded execution, the 2026-10-08 Colab CLI T4 run above (default path only); browser Run all, BYOD, the fine-tuning gate and the activity remain unexercised, and promotion is an integrator's decision. The 2026-09-14 Kaggle T4 row is evidence for the previous E2E blob (`9599d5c4832c`) only: an nbclient execution of the default sample path with a `google.colab` shim, not a browser Colab `Run all`. On 2026-10-08 both notebooks were regenerated (generator /2.2 isolated uv environment, DIMER Notebook Specification 2.2, Notebook Review Framework v1 fixes MCC-M1..M3 / MCC-m1..m6), so under NOTEBOOK_SPEC REL14 they return to Candidate until a run of the exact new blobs is recorded; the earlier row stays as history. The ARTIFACT-INFERENCE companion's default path now downloads the trusted sample bundle (release `sample-bundle-v1`, asset `mitra_classifier_predictor.zip`, SHA-256 `d2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01`, produced by the 2026-10-08 E2E run recorded above; the repository's immutable-releases setting was enabled before it was published) and checks the digest before extraction (NOTEBOOK_SPEC SART6–SART8); it has no recorded run of its current blob yet. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
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


## Maintainer-supplied Colab execution — 2026-09-26

The maintainer reported that this notebook passed an end-to-end Colab run and authorized merging its open PR. The supplied [executed notebook](execution-evidence/2026-09-26/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb) is preserved byte-for-byte as evidence.

- Reviewed source commit: `2b1ff008f178159aa42a321f8491469747c8a7b7`.
- Executed-file SHA-256: `d0566df5b8db6bda7ccf9e784b8e713e039add117b5e2f5f30c7a78b3fb8205e`.
- Independently inspected: 22 executed code cells; zero saved error outputs; terminal completion and exports present.
- Configuration/source comparison: Default controls; cell sources match the reviewed PR exactly.
- Evidence boundary: saved outputs were inspected; execution was not independently repeated. This submission establishes the recorded path, not optional FULL/BYOD paths. Fresh-runtime/restart details beyond the maintainer's explicit prior confirmations are not inferred.

This record supersedes the pending rerun item for the source/configuration above. It does not promote the whole pipeline or close untested optional-path qualification.

The older v1 tutorial saved outputs were moved unchanged to [legacy evidence](execution-evidence/2026-09-26/legacy-classification-saved-outputs.ipynb) so its source carrier satisfies the existing clean-output validator. No v1 code changed.


## FreshRetailNet Classification v2: Notebook Review Framework v1 findings — revision 2.1.1 (2026-09-27)

A review under the Notebook Review Framework v1 (reviewed commit `6c9f911`, notebook blob `c4e850d3`) concluded **Needs revision**. It reported four major and three minor findings, and fresh Colab verification remains pending. Revision 2.1.1 of `tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb` addresses every finding. `tests/test_workshop_v2_review_fixes.py` executes the notebook's own cell code against synthetic inputs. All of its 11 checks fail on the reviewed revision and pass on 2.1.1. These are logic checks, not model runs.

| Finding | Correction in 2.1.1 | Acceptance check |
|---|---|---|
| **NR-01** (major): the fine-tuned TabICLv2 ablation silently ran in-context, because its condition string became `fine_tuned_no_stockout` | The adaptation mode stays the base condition, and the ablation is a separate `ablation_group` identity field. The runner accepts only `in_context` or `fine_tuned`. §7.1 refuses to report a comparison whose two runs used different effective modes | The §5.4 run identity for a fine-tuned ablation keeps `condition="fine_tuned"`; the runner rejects `fine_tuned_no_stockout` before writing any output |
| **NR-02** (major): the worked answer cited an interval against Random Forest as evidence about ablated versus full-feature LightGBM | Every bootstrap row names its `comparator`. A new paired contrast table compares each LightGBM ablation with full-feature LightGBM on the same draws (`ABLATION_CONTRASTS`, exported). The worked answer cites only the point difference and directs learners to the contrast table | Identical full and ablated predictions give a contrast of exactly 0 with a [0, 0] interval, while their interval against a stronger reference stays below 0 |
| **NR-03** (major): a failed or repeated acquisition hit `FileExistsError` | The archive is validated and extracted into a fresh temporary directory, which becomes the active data only on success; a failure leaves nothing behind. A different dataset clears every downstream result and the freeze record. §4.1 refits into a clean folder | Missing split, then corrected ZIP, then repeat all behave as documented; switching datasets removes stale results and `freeze.json` |
| **NR-04** (major): the frozen test used live §5.1 settings while checking only the development file's hash | The test request is built from the verified development `run_config.json`. Only `phase`, `split_paths`, `output_dir`, `source_run_dir` and `frozen_run_fingerprint` may differ. A changed runner or dataset is rejected, and the persisted test configuration must equal the request | With a frozen `n_ensembles=4` and a live value of 8, the test executes with 4, and the exported effective configuration matches |
| **NR-05** (minor): host package versions were unconstrained, yet the text promised identical results | §0.1 checks NumPy, pandas, scikit-learn, LightGBM and matplotlib against a documented tested range, taken from the recorded Colab run and the CI pins. It warns without forcing a restart and records the result in the export. The seed and worked-answer wording no longer promise invariant results | Static markers; the recorded 2.1.0 versions and the CI versions fall inside the range |
| **NR-06** (minor): test-feature summaries were shown before the freeze | §2.2 summarises the training and validation files only. The test-file ranges move to §8.1b, after the freeze. §8 states what is known before the freeze (the split dates) and what is withheld (test labels, scores and feature distributions) | Static checks of both cells and the §7 wording |
| **NR-07** (minor): the export could not recompute its own metrics | The bundle adds per-row validation and test predictions (optional via `EXPORT_ROW_LEVEL_PREDICTIONS`, with a notice for user data), each foundation run's effective development and test configuration, the notebook identity (file, revision 2.1.1, runner SHA-256) and comparison definitions | Static markers |
| Probe finding: `align_probabilities` accepted negative probabilities that sum to 1, and scikit-learn's `log_loss` then crashed | The host and runner both reject negative probabilities with a clear error | Both copies reject `[-0.2, 0.6, 0.6]` |

Code cells changed, so the saved outputs of the 2.1.0 run no longer describe the notebook and were cleared. That run remains byte-for-byte in `execution-evidence/2026-09-26/`. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 default `Run all` of revision 2.1.1;
- the documented Mitra/TabICLv2 fast path;
- a corrected fine-tuned TabICLv2 ablation on a GPU;
- an invalid-then-corrected BYOD archive in one session;
- a post-freeze change to a §5.1 setting, showing the test uses the frozen value;
- the valid BYOD run listed above.

The review's learner-observation recommendation (a representative basic-Python learner completing the notebook unaided) is not addressed by code and remains open.


### Maintainer-supplied Colab execution of revision 2.1.1 — 2026-09-27

The maintainer supplied an executed copy of revision 2.1.1 and authorized merging. It is preserved byte-for-byte as [evidence](execution-evidence/2026-09-27/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb).

- Source: branch `fix/notebook-review-findings` at `7c444f0`, notebook blob `1b166029e174`. All 56 cell ids and sources match exactly.
- Executed-file SHA-256: `248429a7e73608c608634799893b3d856a10961879bc72b7be2b8a62f5b450a5`.
- Runtime: Colab `gpuType` T4 (AutoGluon reports one 14.56 GB CUDA GPU). Host Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, matplotlib 3.10.0; `host_within_tested_range` True.
- Execution: 22 of 22 code cells executed in order (counts 1–22); no error outputs. The completion summary reports validation 9, frozen test models 11, no foundation failures, and "Run-all complete".
- Validation balanced accuracy: TabPFN-3 0.6021, TabICLv2 0.5962, TabDPT 0.5956, Mitra 0.5839, Random Forest 0.5733 (all four foundation models on `cuda`).
- Test balanced accuracy after the freeze: TabPFN-3 0.5587, TabICLv2 0.5452, TabDPT 0.5421, Mitra 0.5399, Logistic Regression 0.5247, Random Forest 0.5244, LightGBM 0.5205.
- Bootstrap (1,000 draws, comparator Random Forest, the validation-best classical baseline): TabPFN-3 +0.0341 [0.0139, 0.0552]; TabICLv2, TabDPT and Mitra intervals include 0.
- The review fixes are visible in the outputs. §7.1 reports the ablations as `trained from scratch` against full-feature LightGBM (−0.0056 without stockout, −0.0032 without period proxies). §8.2 names the comparator on every row. Test-file feature ranges appear only in §8.1b, after the freeze. The export ZIP SHA-256 is `18031397ba00ff9c…`.
- Evidence boundary: saved outputs were inspected; execution was not independently repeated. This covers the default in-context path only.

This record satisfies the fresh default `Run all` item above. These items remain open: the Mitra/TabICLv2 fast path, the fine-tuned TabICLv2 ablation on a GPU, the invalid-then-corrected and valid BYOD runs, and the post-freeze settings check. **Status: Candidate.**


## FreshRetailNet Classification v1 (compact edition): Notebook Review Framework v1 findings — revision 2.1.1 (2026-10-03)

A separate review pass under the Notebook Review Framework v1 (reviewed commit `0434a02`, notebook blob `6b8986df`) concluded **Needs revision**: no blocker, five major, seven minor and four suggestions. The review is archived in `reviews/2026-10-03-notebook-review/`, with a verification addendum that re-checks every major finding against the source before it was fixed. Revision 2.1.1 of `tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb` addresses all twelve major and minor findings. Three of the majors were v2 fixes not yet ported to v1; their code cells are now byte-identical to the v2 2.1.1 cells. `tests/test_frc1_workshop_v1_review_fixes.py` executes the notebook's own cell code against synthetic inputs. These are logic checks, not model runs.

| Finding | Correction in 2.1.1 | Acceptance check |
|---|---|---|
| **FRC1-M1** (major): the fine-tuned TabICLv2 ablation silently ran in-context | v2 NR-01 ported: the ablation is a separate `ablation_group` field, the runner accepts only `in_context` or `fine_tuned`, and §7.1 refuses a comparison whose two runs used different effective modes | A fine-tuned ablation keeps `condition="fine_tuned"`; the runner rejects `fine_tuned_no_stockout` before writing output; a mode mismatch raises and adds no ablation row |
| **FRC1-M2** (major): a failed or repeated acquisition, or a rerun of §4.1, raised `FileExistsError`, and the §4.1 rerun emptied the baseline registry | v2 NR-03 ported: transactional staging, stale results and the freeze cleared when the dataset changes, and §4.1 refits into a clean folder. v1 keeps its own BYOD path/upload selection. A troubleshooting table documents recovery | Invalid then corrected ZIP, a repeat, and a dataset switch behave as documented; §4.1 run twice gives 5 registry entries and the same table |
| **FRC1-M3** (major): the frozen test used live §5.1 settings, and the freeze could be rewritten after the test | v2 NR-04 ported (test request built from the verified development `run_config.json`). New in v1: §8.0 refuses to replace a freeze whose models were already scored on the test partition unless `REFREEZE_REASON` is entered; the replaced freeze, its digest and the reason are kept in `previous_freezes` in `freeze.json`, which the export copies | Frozen `n_ensembles=4` with a live 8 tests with 4; a refreeze after the test raises, and with a reason it records the earlier freeze |
| **FRC1-M4** (major): the README did not say which notebook to use, and contradicted the v2 notebook's state | `tutorials/README.md` and the root README name the guided edition (v2) for participants and new learners, and v1 as the compact edition; the v2 saved-outputs and evidence sentences match the notebook; the v1 row cites its legacy evidence. The notebook's opening states its edition | Static markers and README text |
| **FRC1-M5** (major): three of eight learning objectives had no activity | New activities with observable output: §3.2 band edges from training rows versus validation rows; §4.2 misaligned probability columns; §5.6 adaptation mode of every run, with an optional GPU fine-tuning comparison; §6.2 band distance of errors (classification versus regression). Cell 2 maps each objective to its activity; the §5 checkpoint has collapsible guidance | Static markers; the new cells run on synthetic data in the tests |
| **FRC1-m1** (minor): no test class balance or per-class test recall after the freeze | New §8.1b, after §8.1: class balance in all three files and per-class test recall with the weakest band for every frozen model; the conclusion names the partition | The cell refuses to run before the test evaluation |
| **FRC1-m2** (minor): the lag-7 heuristic's log loss was an unexplained artefact of its 1e-6 floor | A note beside the baseline table explains it and excludes the rule from log-loss comparisons. The table values are unchanged | Static marker |
| **FRC1-m3** (minor): host packages unpinned | v2 NR-05 ported: tested-range check, warning without restart, recorded in the export | Same §0.1 cell as v2 |
| **FRC1-m4** (minor): the BYOD contract omitted labels, column order and model ceilings | "Before you begin" states the column order, finite-feature, label-set and training-only band rules, and each model's limits. New §5.1b rejects an over-ceiling training split before any environment install or model run | An 11,000-row training split is rejected for Mitra; the default sample passes |
| **FRC1-m5** (minor): the export could not reconstruct its own metrics | v2 NR-07 ported; the notebook identity names the v1 file | Static markers |
| **FRC1-m6** (minor): the ablation had no paired contrast against its own full model; the opening misstated the freeze | v2 NR-02 code ported (comparator column, paired ablation contrasts); §7 and §8.2 text, the checklist and the opening corrected | Identical full and ablated predictions give a zero contrast |
| **FRC1-m7** (minor): execution evidence not bound to a revision | Bound below; an exact-revision run of 2.1.1 is still required | This record |

**Legacy evidence binding (FRC1-m7).** `execution-evidence/2026-09-26/legacy-classification-saved-outputs.ipynb` (SHA-256 `fb30617f3a4ee705067648dd4ab4666194aac5811b3c6dce3b81029fec353363`) is git blob `2f840ffe`, the notebook the maintainer committed as `33602f8` ("End to end Colab run", 2026-09-26). All 43 of its cell sources are identical to revision 2.1.0 (blob `6b8986df`). It records Colab `gpuType` T4, execution counts 1–19 in order, no error outputs, 4 of 4 foundation-model runs, 11 frozen test models and the "Run-all complete" line. It is a maintainer-supplied saved-output record of revision 2.1.0's default in-context path, not an independent rerun, and it does not cover revision 2.1.1.

Code cells changed in 2.1.1. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 default `Run all` of revision 2.1.1, including the §3.2, §4.2, §5.1b, §5.6, §6.2 and §8.1b outputs;
- a rerun of §8.0 after §8.1, showing the refusal, and once with `REFREEZE_REASON` showing the recorded earlier freeze;
- a corrected fine-tuned TabICLv2 ablation on a GPU;
- an invalid-then-corrected BYOD archive in one session, and a valid representative BYOD run (REL12).

## FreshRetailNet classification workshops (v1 and v2): 2026-10-03 uv isolated environment

Both editions moved together to the uv isolated environment, so their shared cells stay identical: compact v1 revision 2.1.1 → 2.2.0 (notebook blob `d378e906` → `d17c298b`) and guided v2 revision 2.1.1 → 2.2.0 (blob `1b166029` → `b3c33de9`). Nothing is installed into the notebook kernel and Run all needs no restart. Section 0.1 checks for a Linux x86_64 kernel with Colab's scikit-learn and LightGBM instead of pip-installing LightGBM. Section 5.2 no longer pip-installs `uv`: it downloads uv 0.12.15 by SHA-256, creates each model environment with `uv venv --managed-python --python 3.12.12`, and installs the carried `tutorials/requirements-workshop-<stack>.lock.txt` with `--require-hashes --only-binary :all:`, then `uv pip check`. TabDPT's `antlr4-python3-runtime` 4.9.3 has no wheel, so its hash-pinned source archive is built with the locked setuptools. Top-level pins, data, seeds, models and metrics are unchanged. The notebooks are now **Linux x86_64 only** (Colab, Kaggle, Linux Jupyter). `tests/test_workshop_uv_environment.py` covers the change: 16 of its 22 checks fail on the previous revisions, and the other 6 check the lock files and the shared cells only.

A hosted re-run of both revisions passed on 2026-10-03; see the record below. **Status: Candidate.**

### Colab CLI execution of compact v1 revision 2.2.0 and guided v2 revision 2.2.0 — 2026-10-03

Both editions were run separately at commit `05d54cdd968f34b9168404968b9a294041ae441c`, each on its own fresh Colab session. The executed notebooks are preserved byte for byte:

| Edition | Notebook blob | Executed file | SHA-256 | Cells | Session wall |
|---|---|---|---|---|---|
| Compact v1, revision 2.2.0 | `d17c298b2bde4fa0c525177adfd02f917342d80b` | [`…Workshop_05d54cd_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_05d54cd_colab-cli-t4.ipynb) | `1e7292d0e80ba2344993589c038ebc48a8d5b4902771e98bd143fa43dc8fc9ab` | 25/25 | 591.9 s |
| Guided v2, revision 2.2.0 | `b3c33de91ec669fe3d95e3c04d19c83ad16722b6` | [`…Workshop_v2_05d54cd_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2_05d54cd_colab-cli-t4.ipynb) | `f118dc4242599dd781376ab6b7f4c7b447c454170e6bdbe0dcc9c515d4893d2f` | 22/22 | 668.6 s |

- **Source:** each notebook was downloaded from GitHub at the PR head and its Git blob verified before the session.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session per notebook, through the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log.
- **Path exercised:** default controls only — pinned FreshRetailNet sample, balanced accuracy as the primary metric, all four foundation models in context, freeze and test on. Not run: the §8.0 refreeze refusal and `REFREEZE_REASON` path, the fine-tuned TabICLv2 ablation, the fast path, and the invalid-then-corrected and valid BYOD runs.
- **Outcome:** PASSED, 25/25 (v1) and 22/22 (v2) code cells with no error outputs. Both completion summaries report 9 validation models, 11 frozen test models, no foundation failures, and "Run-all complete". The v1 log has one Hugging Face Hub notice about unauthenticated requests; it is a warning only.
- **Kernel:** Python 3.13.15 on Linux x86_64; section 0.1 installed nothing. NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0 and matplotlib 3.10.0, with `host_within_tested_range` True.
- **Isolated environments (section 5.2):** in both runs all four uv environments (`mitra`, `tabdpt`, `tabicl`, `tabpfn`) were created with CPython 3.12.12, installed from the hash-locked files, and passed `uv pip check` and the adapter import check. TabDPT's `antlr4-python3-runtime` 4.9.3 was built from its source archive in under 1 s. All four foundation models ran on `cuda`.
- **Setup time:** uv reported about 261 s (v1) and 258 s (v2) preparing and installing packages across the four environments: Mitra 79 s / 81 s, TabDPT 81 s / 70 s, TabICL 60 s / 66 s, TabPFN 37 s / 39 s. The CLI records no per-cell times, so the wall time of section 5.2 as a whole was not measured.
- **Time against the notebooks' estimates:** both editions say the first installation takes several minutes per model; uv's measured preparation was 37–81 s per model. The guided edition's roadmap gives times per section, not a total. Measured session wall, including session start and stop, was 591.9 s (9.9 minutes) for v1 and 668.6 s (11.1 minutes) for v2. The estimates were not edited.
- **Validation balanced accuracy (both editions, identical):** TabPFN-3 0.6021, TabICLv2 0.5962, TabDPT 0.5956, Mitra 0.5839, Random Forest 0.5733.
- **Test balanced accuracy after the freeze (both editions, identical):** TabPFN-3 0.5587, TabICLv2 0.5452, TabDPT 0.5421, Mitra 0.5399, Logistic Regression 0.5247, Random Forest 0.5244, LightGBM 0.5205.
- **Bootstrap (1,000 draws, comparator Random Forest):** TabPFN-3 +0.0341 [0.0139, 0.0552]; the TabICLv2, TabDPT and Mitra intervals include 0. The LightGBM ablations are −0.0056 without stockout features and −0.0032 without period proxies.
- **Report ZIP SHA-256:** v1 `ca7f75258e48419f…`, v2 `6a82288f7ac04273…`. These are new sessions, so the digests differ from earlier runs by construction.
- **Comparison with the most recent recorded hosted run:**
  - Guided v2: the last recorded run is revision 2.1.1 above (maintainer-supplied, 2026-09-27). Every printed metric of revision 2.2.0 equals it: the data checks and EDA, the baselines, the validation table, the confusion matrix, the one-row walkthrough, the ablations, the frozen-test results, §8.1b, the bootstrap intervals and the §8.3 preview. Only host memory figures in the AutoGluon log and the export digest differ.
  - Compact v1: the last recorded run is the legacy saved-output record of revision 2.1.0 (2026-09-26, bound above). Every metric shared by both revisions is equal: the EDA, the baselines, the validation table, the confusion matrix, the ablation deltas, the frozen-test results, the bootstrap intervals and gains, and the §8.3 preview. Revision 2.2.0's §8.2 adds the comparator column and the paired ablation contrasts. The cells added in 2.1.1 (§3.2, §4.2, §5.1b, §5.6, §6.2, §8.1b) have no earlier hosted output. Where the two editions print the same values, they match the v2 run.
- **Evidence boundary:** the saved outputs were inspected. The journeys not exercised, listed above, remain open.

This record satisfies the fresh default-path run of both revisions, including the v1 §3.2, §4.2, §5.1b, §5.6, §6.2 and §8.1b outputs. The other items listed for revision 2.1.1 of each edition above remain open. **Status: Candidate.**
