# Mitra Classifier Predictor-Inference Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/mitra-classifier-pipeline`  
**Notebook:** `tutorials/mitra_classifier_predictor_inference_colab.ipynb`  
**Reviewed commit:** `374a15055fb7f48bd0d36ad5c1c23893847936e7` (`main`, confirmed with `gh api repos/kurtvalcorza/mitra-classifier-pipeline/commits/main`)  
**Notebook Git blob:** `d863d0ce3ac55562f51d0e9008365311073fea64` (last changed in `0e13847`, 2026-09-14)  
**Finding prefix:** `MCP` (the companion E2E notebook is `MCC`, reviewed in its own row; the FreshRetailNet workshops use `NR`/`FRC1`)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The external-artifact machinery is sound. Given a bundle path, its trusted digest and a CSV path, all seven code cells ran on CPU in a fresh process. The cells verified the whole-archive SHA-256, extracted the bundle safely and checked its manifest, provenance and base-model identity before `TabularPredictor.load`. They then reconstructed a binary predictor, validated 114 new rows into an input manifest with a recorded rejection probe, and exported four files. The probability columns sum to 1 (max deviation 3e-8) and `prediction` equals their argmax on every row. Five invalid inputs were refused before deserialisation or model execution, each with a message that names the failed condition. The trust-boundary prose is accurate: it says digests prove integrity, not sender authenticity or safe unpickling. The prose on uncalibrated probabilities and the shipped decision rule is also accurate.

The notebook still cannot do what its profile promises by default:

1. **The default `Run all` has no sample artifact and no sample input (MCP-B1).** With the forms untouched, Section 4 opens an upload dialog on Colab. On Jupyter it raises `ModuleNotFoundError: No module named 'google'`. Even after a learner uploads a valid bundle, cell 9 raises because `EXPECTED_ZIP_SHA256` is empty by default. The notebook's own opening says so, and the 2026-09-14 Kaggle pass skipped it for the same reason. No execution evidence exists for any revision.
2. **Supplying the bundle by path and the rows by upload crashes (MCP-M1).** The upload branch in cell 13 calls `files.upload()`, but `files` is imported only inside cell 9's upload branch. This is the exact combination the B1 fix will create, because the sample bundle will come by path and BYOD rows by upload.
3. **The declared `GUIDED` layer is absent (MCP-M2).** There are no expected-result notes after Section 1, no prediction prompts, checkpoints, glossary or conclusion template, and no in-notebook activity. The 1,190-line carrier cell is neither labelled nor collapsed.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `ARTIFACT-INFERENCE` / `GUIDED` (`metadata.dimer`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0**, standalone (§4) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2**; IDs below are 2.2 IDs |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) with template `tools/notebook_template_artifact_inference.py`. Carried module `mitra_pipeline/tutorial_api.py` at `4143cbf0319f`, sha256 `c346eedf…`. `build_notebook.py --template … --check` → `OK … is up to date` |
| Intended audience | Not stated. The prerequisites list the runtime, the artifact, the data and external access, but not the learner's assumed level |
| Supported runtime | "Google Colab or Jupyter, Python 3.10–3.13". The default path is CPU, with CUDA used if present. The bundle's AutoGluon version and Python major/minor must match the runtime |
| Promised outcomes | Externally produced bundle → whole-archive digest → safe extraction → manifest/provenance/base-model checks before deserialisation → runtime-compatibility check → reconstruction from the bundle alone → validated new rows with an input manifest → class predictions with probabilities → `not-measurable` evaluation report → machine-readable export. BYOD for new input. Optional user artifact |
| Cells | 17 cells: 7 code, 10 markdown. Code cells 3, 5 (1,190-line carrier), 7, 9, 11, 13 and 15. No persisted outputs; all code cells compile |

**Existing execution evidence.** None for this notebook at any revision:
- `tutorials/README.md` lists it as **Candidate**, with execution evidence "unverified — no clean-runtime `Run all` execution recorded".
- `docs/release-verification.md` §"Manual verification" step 6 describes the second-runtime procedure but records no run.
- `.agent/backups/kaggle-pass-2026-09-14/SKIPPED.md:4` skipped it: "needs a published sample artifact (§19 SART1/RUN5) — none exists".

**Journeys and evidence basis**

| Journey | Basis | What was done |
|---|---|---|
| First-time learner | Source inspection | All 17 cells read in order against GDL/UX/§19 |
| Clean default | Direct execution (CPU, fresh process, no form edited) | P1, plain Jupyter with no `google.colab`: cell 9 → `ModuleNotFoundError`; cells 11/13/15 → `NameError`. P2, Colab upload stubbed with a valid bundle and default digest field: cell 9 → `RuntimeError: A trusted whole-archive SHA-256 is required…`. Colab browser `Run all` **not verified** (no sample asset exists to make it pass) |
| External-artifact path (the executor path in `release-verification.md` step 6) | Direct execution (CPU) | P3: bundle produced by a **separate process** that ran the companion E2E notebook at the same commit (breast-cancer sample, ZIP 280,280,272 bytes, sha256 `aabe41f7…`), plus 114 unlabelled rows from that run's independent test partition. With `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256` and `NEW_DATA_PATH` set, cells 3–15 all pass (cell 15: 14.5 s); four outputs written. Reviewer-side accuracy against the withheld labels was 0.982; the notebook reports nothing because it has no labels |
| Active learning | Not applicable within the notebook / partly probed | The notebook has no exercise; its "Next experiments" send the learner to the E2E notebook (MCP-M2). Closest probe, R10: a labelled CSV is accepted, `target` passes through as an extra column, and the verdict stays `not-measurable` (MCP-S2) |
| Reuse and recovery | Direct execution (CPU) | P4: artifact by path + rows by upload (stub) → `NameError: name 'files' is not defined` (MCP-M1). P5 refusals: wrong digest, non-hex digest, unlisted member (digest re-trusted), `../` traversal, CSV missing a feature, CSV with a `prediction` column. All are refused with the condition named. `ALLOW_UNVERIFIED_ARTIFACT=True` passes with its warning. A one-row CSV predicts. P6: the bundle reconstructs and predicts **offline with an empty HF cache and without Section 3** (MCP-m2). Python/AutoGluon mismatch was read in source only |

**Deviations and limitations.**
- **Host and environment.** Windows CPU host, using `mitra-regressor-pipeline/.venv` read-only. It holds the exact pins: autogluon.tabular 1.5.0, lightgbm 4.6.0, huggingface-hub 0.36.2, Python 3.12.12, torch 2.9.1+cpu. The install cell was skipped with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so the restart guard in cell 3 was not exercised.
- **Memory override, labelled.** The bundle producer, the companion E2E run, set its own form control `MAX_MEMORY_USAGE_RATIO=3.0`. The host had about 3.5–7.8 GB free, and AutoGluon's Mitra estimate is about 7.1 GB. The notebook under review needed no override.
- **Weights.** The Mitra snapshot was copied in, with digests equal to the manifest. The Hub download was not exercised.
- **Stubs.** `google.colab.files.upload` was stubbed in P2 and P4 only. The real Colab dialog, including uploading a 280 MB file through the browser, was not exercised.
- **No GPU run.**

## 2. Separate judgments

- **Technical correctness.** The core path is correct when files are supplied by path. The checks run in the stated order, before any deserialisation (`validate_artifact_directory` reads only JSON). The class order is explicit, probabilities are finite and sum to 1, and outputs carry every input column. There are two defects. The default path cannot complete (MCP-B1), and `files` is used across cells as hidden state (MCP-M1, SRC2). Section 3 downloads and loads a checkpoint that inference does not need (MCP-m2).
- **Promise fulfilment.** Every artifact-handling promise holds on the path journey. "Reconstruct the predictor from the bundle alone" was confirmed directly (P6). The profile's central promise fails: "the default `Run all` automatically obtains a trusted sample artifact and sample input" (§7.4/§19). The notebook says so, but saying so does not satisfy a MUST. The opening's "predicts continuous class labels" is wrong (MCP-m1).
- **Learner experience.** The prose is accurate but written for an engineer or reviewer. The guided layer is missing, and the only activity sends the learner to another notebook (MCP-M2). Upload-based inputs assume Colab and a practical 280 MB browser upload (MCP-m4).
- **Spec conformance.** These applicable MUSTs are **unmet**: SART1, RUN1, RUN2, RUN5, DAT1, DAT3, the §19 bullets for automatic sample artifact, no default upload and automatic sample input, and REL1/REL4 (no execution evidence). These MUSTs are **met** on the path journey: §20 archive security (traversal, symlink, size, ratio, unlisted-file rejection; whole-archive digest), the §19 trust boundary, validation before deserialisation, display of model identity, format version and provenance, no refit from inference data, MOD1–MOD3, MOD7, UNC2–UNC4, INF3 and INF5, §21.1 explicit class order and decision rule, DAT17/DAT18, VAL4 and OUT6. These SHOULDs have unrecorded deviations: GDL1–GDL14, UX5/UX8, VAL6 (MCP-m3) and EXE2 (MCP-M1/m4).

## 3. Findings

### MCP-B1 (Blocker): the default `Run all` has no trusted sample artifact or sample input

- **Cell/section:** opening cell 0 ("Known NOTEBOOK_SPEC 2.0 gap"); Section 4, cell 9; Section 6, cell 13.
- **Observed issue:** `ARTIFACT_ZIP_PATH = ''` and `NEW_DATA_PATH = ''` fall through to `from google.colab import files; files.upload()`. `EXPECTED_ZIP_SHA256 = ''`, together with `ALLOW_UNVERIFIED_ARTIFACT = False`, makes cell 9 raise even after a valid upload. Nothing in the notebook obtains a bundle or rows. The prerequisites say "no sample is bundled".
- **Consequence:** A learner who opens the notebook and chooses **Run all** cannot complete it. On Colab, Run all stops at a dialog and needs a ~280 MB bundle produced in another session, plus its digest pasted into a field. On Jupyter it fails outright. The profile's default demonstration never runs, and no clean-runtime evidence can be recorded (the Kaggle pass skipped it for this reason).
- **Evidence:** Direct execution: P1 (Jupyter: `ModuleNotFoundError: No module named 'google'` at cell 9, then `NameError`s) and P2 (valid bundle uploaded, default digest field: `RuntimeError: A trusted whole-archive SHA-256 is required…`). Source inspection: cell 0 and cell 1 text. Documented: `SKIPPED.md:4`, and `tutorials/README.md` status "Candidate".
- **Spec:** SART1, SART2, RUN1, RUN2, RUN5, DAT1, DAT3, §7.4, §19 bullets 1, 2 and 7, REL1, REL4.
- **Recommended correction (generator: `tools/notebook_template_artifact_inference.py`, the cell-9 source at lines ~95–115, the cell-13 source at ~180–190, and the opening text at line 26):**
  1. Publish one trusted sample bundle that the companion E2E notebook produced at a pinned revision on the breast-cancer sample. Use a stable public, credential-free location with an immutable reference, for example a Hugging Face Hub repo at a commit SHA or a GitHub release asset. ST6/SART2 allow a downloaded trusted sample artifact; ST4 forbids only DIMER source.
  2. Make the default branch download it, with its SHA-256 as the default trusted digest, before `safe_extract_archive`. State its provenance, format version and digest in markdown before deserialisation (SART3).
  3. Generate the default sample input in-notebook with `sklearn.datasets.load_breast_cancer`, restricted to rows disjoint from the bundle's support rows. For example, reproduce the E2E notebook's seeded test partition and assert the overlap is zero.
  4. Gate both uploads behind default-off switches (`USE_OWN_ARTIFACT`, `USE_BYOD`).
  5. Because AutoGluon bundles are bound to the Python major/minor, record the sample bundle's producer Python. Document that the bundle must be rebuilt when Colab's Python changes, or the default path fails in cell 11.
  6. Set `sample_kind` to `sample` on the default path (it is currently hard-coded `BYOD` in cell 15).
  7. Remove the "Known gap" paragraph and set `tutorials/README.md` accordingly.
- **Acceptance check:** In a fresh Colab runtime with no form field edited, **Run all** completes all code cells with no dialog and no error. `outputs/` holds the four `mitra_classifier_predictor_inference_*` files. `result.json` has `artifact.source` naming the published sample location and `digest_verified: true`, and `evaluation_report.sample_kind` is `sample`. The run is recorded in `docs/release-verification.md` with commit, blob and runtime (REL10).

### MCP-M1 (Major): the CSV upload branch raises `NameError` whenever the bundle came by path

- **Cell/section:** Section 6, cell 13 (`new_upload = files.upload()`).
- **Observed issue:** `files` is bound only by `from google.colab import files` inside cell 9's `else:` branch. When `ARTIFACT_ZIP_PATH` is set and `NEW_DATA_PATH` is empty, cell 13 uses a name that was never defined. This is a hidden cross-cell dependency.
- **Consequence:** An executor or learner who supplies the bundle by path and the rows by upload hits `NameError` with no recovery message. After MCP-B1 is fixed, the default sample bundle will always come by path, so this becomes the BYOD-upload path for every learner.
- **Evidence:** Direct execution, P4: cells 3–11 pass with the artifact by path and digest; cell 13, with the Colab stub holding one CSV, raises `NameError: name 'files' is not defined`.
- **Spec:** SRC2, EXE2, DAT10/DAT15, UX10.
- **Recommended correction:** Import `from google.colab import files` inside cell 13's upload branch (template line ~183). Better, follow the B1 gating so that each upload branch imports its own dependency and fails with an actionable message outside Colab.
- **Acceptance check:** P4 (`run_probes.py mixed`) completes cell 13 and cell 15 with no error, and an equivalent Colab run with the artifact by path and the CSV by dialog writes the four outputs.

### MCP-M2 (Major): the notebook is declared GUIDED, but the guided layer and any learner activity are absent

- **Cell/section:** whole notebook; opening cells 0–1; Sections 3–7; cell 16 "Interpretation and limits"; carrier cell 5.
- **Observed issue:**
  - No intended learner or assumed level (GDL1), "How to use this notebook" (GDL2), roadmap (GDL3) or Input → Model → Output contract (GDL4).
  - The 11 "learning objectives" are pipeline steps ("install the pinned runtime", "read what the carried package guarantees"), not observable learner outcomes (GDL5/UX1).
  - No glossary for predictor bundle, whole-archive digest, manifest, deserialisation or in-context support (GDL6).
  - Only Section 1 says what to look for; Sections 3–7 have no expected-result notes (GDL8/UX4), and no prediction prompts or checkpoints with sample answers (GDL7/GDL9). Static probe: none of "expected result", "what to notice", "glossary", "troubleshoot", "how to use", "infrastructure" or "conclusion" appears.
  - No Predict → Change → Run → Observe → Explain activity (GDL10/UX5). Both "Next experiments" require the E2E notebook (and one needs a GPU fine-tune there).
  - The predictions table (`out.head()`) and the probability columns get no reading guidance.
  - No conclusion template (GDL14). The failure table covers integrity failures but not upload size, Python/AutoGluon mismatch with the current Colab, or memory (GDL13).
  - The 1,190-line, 54 kB carrier cell is not labelled Infrastructure or collapsed (GDL11/GDL12).
- **Consequence:** The learner can run the cells but is not taught to read a predictor bundle's provenance, to judge what the probabilities mean, or to decide what to do with a refusal. The intended audience is undefined, so the dense engineering prose cannot be checked against it.
- **Evidence:** Source inspection of all 10 markdown cells; static marker probe (`S_static.markers`).
- **Spec:** GDL1–GDL14, UX1, UX4, UX5, UX8, UX9.
- **Recommended correction (template text in `tools/notebook_template_artifact_inference.py`):**
  - Add an audience and how-to-use block and an I/O contract.
  - Rewrite the objectives as observable actions, for example "explain which checks run before deserialisation and why".
  - Add an expected-result note after Sections 3–7.
  - Add one bounded in-notebook activity on the sample. For example, predict and then observe how a tampered bundle (an extra file) or a CSV missing one feature is refused, or which rows have probabilities near 0.5. A labelled copy of the sample input would also allow a sample-sanity report (MCP-S2).
  - Add a glossary, a troubleshooting section and a conclusion template.
  - Label and collapse the carrier as Infrastructure.
- **Acceptance check:** A reviewer finds each of GDL1–GDL14 addressed in the regenerated notebook, or each deviation recorded in `tutorials/README.md`. At least one activity runs inside this notebook on the default sample without editing more than one form field, and does not break **Run all**.

### MCP-m1 (Minor): "continuous class labels" is regression wording

- **Cell/section:** cell 0, twice: the "This notebook consumes…" paragraph and the learning objectives.
- **Issue → consequence:** A classifier predicts discrete class labels. The phrase, apparently carried over from the regressor companion, tells a learner new to the task the wrong output type.
- **Evidence:** Source inspection; the static probe counts 2 occurrences. Template lines 48 and 62.
- **Spec:** UX1, §21.1.
- **Correction:** Use "class labels (argmax) with per-class probabilities".
- **Acceptance check:** `grep -c "continuous class labels"` on the regenerated notebook returns 0.

### MCP-m2 (Minor): Section 3 downloads and loads a 303 MB checkpoint that inference does not use, and the stated reason is inaccurate

- **Cell/section:** Section 3 (cell 7); cell 0 ("it exists so the bundle's recorded base-model digests can be checked against known-good values"); cell 11 (`serving = MitraClassificationPipeline(weights_path=pipe.model_weight_path, …)`).
- **Observed issue:** The bundle carries `models/Mitra/model.pt` (302,822,483 bytes). It reconstructs and predicts with an empty HF cache, offline, without Section 3 (P6: 114 rows, 14.7 s, no cache files written). Cell 9 compares the bundle's metadata digests with the carried constants `WEIGHTS_SHA256`/`CONFIG_SHA256`, not with the downloaded file. `from_pretrained` also stages extra copies of the weights into `weights/.cache/hf` and the user's default HF cache.
- **Consequence:** A fresh runtime downloads about 303 MB and writes up to two further copies for no inference purpose (RUN12). The learner is told the download is needed for a check it does not perform. `serving` depends on `pipe`, a second hidden dependency on Section 3.
- **Evidence:** Direct execution (P6, P3 `pipe_snapshot_used_for_serving`); source inspection of `validate_artifact_directory` and `stage_verified_hf_snapshot` in the carried module.
- **Spec:** RUN12, UX2, ENV9.
- **Correction:** Either drop the snapshot staging from this profile and build `serving` from the bundle and provenance alone, or keep it as an explicit optional step and say plainly that the base-model check uses the pinned constants.
- **Acceptance check:** The regenerated notebook either completes Run all without fetching `model.safetensors`, or its prose states that Section 3 is not needed for inference and what it adds.

### MCP-m3 (Minor): inference ceilings are not surfaced; training ceilings are printed instead

- **Cell/section:** Section 6, cell 13 (`print({'ceilings': {'MIN_TRAIN_ROWS', 'MAX_TRAIN_ROWS', 'MAX_FEATURES'}…})`).
- **Issue → consequence:** The printed ceilings govern the producer's fit, not this stage. `validate_inference_frame` enforces no row or size limit, so a large CSV is read fully into memory and scored without warning. A learner reading "ceilings" may assume a 10,000-row inference limit that does not exist.
- **Evidence:** Source inspection (`INPUT_SCHEMA['eval_rows'] = [2, None]`; no inference cap in `validate_inputs`). Direct execution showed a one-row CSV accepted (R9).
- **Spec:** VAL6, DAT12.
- **Correction:** Print the inference contract (required features, reserved columns, no row cap, or an explicit `MAX_INFERENCE_ROWS`) and label the training ceilings as producer-side context.
- **Acceptance check:** Cell 13 prints the inference-stage limits under an inference label before `read_csv_bytes`.

### MCP-m4 (Minor): the upload branches assume Colab, and a browser upload of a ~280 MB bundle is impractical

- **Cell/section:** cell 1 ("Google Colab or Jupyter"); cells 9 and 13 (`from google.colab import files`).
- **Issue → consequence:** On Jupyter, which the notebook names as a supported runtime, both upload branches fail with `ModuleNotFoundError` and no guidance. On Colab, the user-artifact branch asks for a ~280 MB ZIP through `files.upload()`, which is slow and is held in memory twice (payload plus written copy). No mounted-storage option is mentioned.
- **Evidence:** Direct execution, P1 (Jupyter). Measured bundle size 280,280,272 bytes (B_bundle). The Colab dialog itself was not verified.
- **Spec:** EXE2, UX10, GDL13, DAT16.
- **Correction:** Guard the upload branches with an actionable message ("set `ARTIFACT_ZIP_PATH`/`NEW_DATA_PATH` outside Colab"). Recommend a path or Google Drive mount for user artifacts, and mention the expected size.
- **Acceptance check:** With no `google.colab`, setting `USE_OWN_ARTIFACT=True` and leaving the path empty raises a message naming the path field. The prose names a non-upload route for bundles.

### Suggestions

- **MCP-S1:** Regenerate against NOTEBOOK_SPEC 2.2 (metadata and opening declare 2.0), and use the §29 artifact-inference opening template.
- **MCP-S2:** When the scored CSV contains the bundle's `target_column` (R10: it passes through as an extra column and the verdict stays `not-measurable`), offer an optional labelled check with `classification_metrics` and `majority_class_baseline` in this notebook, labelled sample-sanity. That would let the "Next experiment" run here instead of in the E2E notebook.
- **MCP-S3:** Show the learner a compact provenance summary from the bundle (data source, support rows, mode, selection basis) next to the predictions, so that "what was this predictor trained on?" is answered where the predictions are read.

## 4. Readiness

**Needs revision.** One Blocker (MCP-B1) and two Majors (MCP-M1, MCP-M2) are open. Applicable MUSTs fail: SART1, RUN1, RUN2, RUN5, DAT1, DAT3, the §19 default-path bullets, and REL1/REL4. No clean-runtime execution evidence exists for any revision.

Remaining gates, in order:
1. Fix MCP-B1 and MCP-M1 in the generator and regenerate; `--check` and the parity tests must stay green.
2. Address MCP-M2, or record each GDL deviation durably.
3. Run a fresh Colab `Run all` with no field edited and record it in `docs/release-verification.md`.
4. Separately exercise the user-artifact and BYOD branches (REL12).

The Minors and Suggestions do not gate release.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, this host, exact pins):**
  - The default path fails on Jupyter (P1), and with an uploaded bundle and the default digest field (P2).
  - The path journey passes end to end, with four outputs, probabilities summing to 1 and argmax consistent (P3).
  - The mixed path raises `NameError` (P4).
  - Six refusals and the override warning (P5).
  - The bundle reconstructs offline without Section 3 (P6).
  - The generator `--check` is clean.
- **Verified by source inspection:** check order before deserialisation; the trust-boundary prose; the absence of the guided layer; the "continuous" wording; the ceilings printed.
- **Inferred / not verified:**
  - Real Colab behaviour, including the upload dialog and a 280 MB browser upload.
  - The Hub download in Section 3.
  - The install cell's restart guard.
  - GPU execution.
  - A Python/AutoGluon-mismatch bundle (read in source only).
  - Learner understanding (no learner observation).
- **Only Kurt can confirm:** where a public sample bundle may be hosted, and whether the 280 MB breast-cancer bundle is acceptable as the published sample.
- **Finding most likely to be wrong:** MCP-M1's severity. Today it bites only when an executor mixes path and upload, so it could be argued Minor. It is graded Major because the MCP-B1 fix will make "bundle by path, rows by upload" the standard BYOD route for every learner.

*Probe bundle:* `mitra_classifier_predictor_inference_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
