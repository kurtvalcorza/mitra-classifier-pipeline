# Mitra Classifier E2E Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/mitra-classifier-pipeline`  
**Notebook:** `tutorials/mitra_classifier_colab.ipynb`  
**Reviewed commit:** `374a15055fb7f48bd0d36ad5c1c23893847936e7` (`main`, confirmed with `gh api repos/kurtvalcorza/mitra-classifier-pipeline/commits/main`)  
**Notebook Git blob:** `9599d5c4832c1e3851a9c6ddf394af40e4d9ae53`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-14 (commit `0e13847`, kernel `dimer-nb2-mitra-classifier` v2).  
**Finding prefix:** `MCC` (the FreshRetailNet workshops use `NR`/`FRC1`; the companion inference notebook is a separate row)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The default path is well engineered and runs. It installs three exact pins, carries `mitra_pipeline/tutorial_api.py` verbatim, digest-verifies the pinned `autogluon/mitra-classifier` snapshot and validates the Breast Cancer table into an input manifest. It checks class coverage and cross-split overlap, registers Mitra in context, and scores it next to majority-class, LightGBM and Random Forest baselines on a holdout and an independent test. It then writes the evaluation report, exports the AutoGluon predictor bundle and reloads it through `safe_extract_archive` and `validate_artifact_directory` with an explicit tolerance. Its prose on uncalibrated probabilities, the `argmax` rule, the class ceiling and the artifact's data obligations is accurate. The recorded Kaggle T4 run of this exact blob passed 9/9 code cells in one pass, with no restart. A CPU run in this review reproduced its metrics.

Three problems stand in the way of `Ready for intended use`:

1. **The optional fine-tuning experiment crashes at model selection (MCC-M1).** Cell 13 calls `math.isfinite`, but no cell imports `math`. On a GPU with the default sample, the fine-tune runs and then the cell raises `NameError` before selection, export or the comparison table. This review reproduced that control flow.
2. **Mitra is not conditioned on the rows the notebook says it is (MCC-M2).** AutoGluon silently holds out 20% of the support rows, so Mitra's context is 272 of 341 rows. LightGBM and Random Forest are fitted on all 341 rows, yet the notebook describes the comparison as "the exact same support rows". The exported provenance records `train_rows_used: 341`.
3. **The guided layer is largely absent (MCC-M3).** The notebook declares `GUIDED`, but there is no how-to-use guidance, Input → Model → Output contract, expected-result notes, prediction prompts, checkpoints, glossary, troubleshooting or conclusion template. The 1,190-line carrier cell is not labelled or collapsed as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (`metadata.dimer`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0**, standalone (§4) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2**. ID citations below are 2.2 IDs |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) with template `tools/notebook_template.py`. Carried module `mitra_pipeline/tutorial_api.py` at `4143cbf0319f`, sha256 `c346eedf…` |
| Intended audience | Not stated explicitly. Prerequisites say "basic pandas" plus stratified holdout, accuracy, balanced accuracy, log loss and ROC-AUC |
| Supported runtime | Colab or Jupyter, Python 3.10–3.13. Default path on CPU, CUDA used if present. Fine-tuning needs a GPU |
| Promised outcomes | Verified acquisition; validated support data with class coverage; in-context evaluation against executable baselines on holdout and independent test; optional GPU fine-tuning with holdout-only selection; evaluation report; optional new-data inference with probabilities; predictor bundle export and fresh reload; BYOD through the same cells |
| Cells | 21 cells: 9 code, 12 markdown. Code cells 3, 5 (1,190-line carrier), 7, 9, 11, 13, 15, 17 and 19 |

**Existing execution evidence.**

- `docs/release-verification.md` records a Kaggle T4 run of commit `0e13847`, blob `9599d5c4832c`, as **PASSED** (9/9 cells, 215.5 s). That is the reviewed blob.
- The executed notebook and `run_summary.json` are in `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-mitra-classifier/v2/`. They show:
  - one pass, with no restart;
  - a clean Hugging Face cache;
  - the Kaggle image's torch 2.10.0 replaced by AutoGluon's torch 2.9.1, with nothing imported before the install cell;
  - two Tesla T4 GPUs, device `cuda`.
- That run used an nbclient executor with a shim cell, not a browser `Run all` on Colab.
- No Colab execution of any revision is recorded. No BYOD, fine-tuning, new-data upload or Wine run is recorded.

**Journeys and evidence basis**

| Journey | Basis | What was done |
|---|---|---|
| First-time learner | Source inspection | All 21 cells read in order against GDL/UX/EVAL |
| Clean default | Documented execution (Kaggle T4, exact blob) **and** direct execution (CPU, see deviations) | All code cells 3–19 ran in order in one namespace, with the install skipped; every cell passed. Holdout and test metrics equal the Kaggle record to 4 d.p., except Mitra's log loss (0.0500 vs 0.0497) |
| Active learning | Direct execution (CPU) | (a) The documented next experiment, `Sample: Wine`, run in a fresh process through cells 9–19: all pass. (b) Cell 13 with `RUN_FINE_TUNING=True`, with the GPU check stubbed and the candidate's fine-tune replaced by a **stand-in** that reuses the pretrained predictor: `NameError: name 'math' is not defined`. (c) Real CPU gate: GPU refusal message |
| Reuse and recovery | Direct execution (CPU, `google.colab.files.upload` stubbed) | Telco pre-split CSVs through cells 9–19, with new-data inference by `NEW_DATA_PATH`: all pass. Four invalid inputs. Adult single-CSV upload validated and split, but the fit was not verified (host RAM). Colab upload dialog not verified |

**Deviations and limitations.**

- **Host and environment.** This review ran on a Windows CPU host using `mitra-regressor-pipeline/.venv` read-only. That environment has the notebook's exact pins: autogluon.tabular 1.5.0, lightgbm 4.6.0 and huggingface-hub 0.36.2, with torch 2.9.1+cpu.
- **Memory override.** The host had about 3.5 GB of free RAM, and AutoGluon's pre-fit estimate for Mitra is 7.08 GB. With the notebook's default `MAX_MEMORY_USAGE_RATIO = 1.10`, AutoGluon therefore *skipped* Mitra and raised "No models were trained successfully". Every cell-13 run here sets that form control to 3.0. This is a host limit, not a Colab finding.
- **Weights.** The Mitra snapshot was copied in with its digests checked, so the Hub download was not exercised here. The Kaggle record exercises it.
- **Not verified.** A real GPU fine-tune was not run. Learner observation was not attempted.
- **Probe wall time.** About 16 minutes.

## 2. Separate judgments

- **Technical correctness:** The default path is sound and reproducible: hosted run, plus this review's CPU rerun with matching metrics. Two defects:
  - The optional fine-tune selection crashes (MCC-M1).
  - The support-row count misreports what Mitra was conditioned on (MCC-M2).

  Validation, coverage, overlap, alignment, archive safety, verify-before-deserialise and the reload tolerance all behaved as documented.
- **Promise fulfilment:** Most promises are delivered and observable. Two are not:
  - "optionally fine-tune on a GPU with holdout-based selection" (MCC-M1);
  - "in-context conditioning on the training split" with baselines "on the exact same support rows" (MCC-M2).

  BYOD reaches every downstream stage for a representative pre-split table.
- **Learner experience:** The notebook is accurate but dense, and is written as an engineering record. The opening cell alone is 5,826 characters, and it addresses a reviewer about an "open decision" in the spec. The guided layer is absent (MCC-M3). The baseline comparison is printed without help reading 1–3-row differences (MCC-m3). Cell 13 floods the learner with roughly 100 lines of AutoGluon logging and unexplained warnings (MCC-m2).
- **Spec conformance (2.2):**
  - SHOULD shortfalls: GDL1–GDL14, UX4, UX8 and UX10. One `MUST`-level weakness is EVAL14/ART8: the selection path for the fine-tuned variant crashes, so its selection is never recorded.
  - OUT8 (MUST): the adaptation provenance misstates the support rows used.
  - The notebook declares spec 2.0, and the repository documents disagree about status (MCC-m5).
  - RUN1/RUN10/ENV6 are met on the hosted record: one pass, no restart.
  - FT2/RUN7: in-context support registration is an adaptation stage under §FT ("in-context support registration"), so the default path satisfies FT2. The opening's "open decision" sentence is unnecessary (MCC-m1).

## 3. Findings

### MCC-M1 (Major): the optional GPU fine-tuning experiment raises `NameError` at selection

- **Cell/section:** cell 13 (Section 6), functions `better()` and the `degraded` list comprehension. Generator: `tools/notebook_template.py` lines 304 and 316.
- **Observed issue:** Both code paths call `math.isfinite(...)`. `math` is imported by no cell: neither the carrier module nor cells 3–19 contain `import math`. With `RUN_FINE_TUNING = True` on a GPU, the candidate is fine-tuned and scored. Then, whenever the holdout has at least `MIN_SELECTION_HOLDOUT_ROWS = 50` rows (the default sample has 114), `better(candidate_metrics, pretrained_metrics, EVAL_METRIC)` raises `NameError: name 'math' is not defined`. This happens before `ACTIVE_MODEL` is chosen and before the baseline table is printed.
- **Consequence:** The notebook offers fine-tuning in five places as the way to see "holdout-based selection and the independent-test warning": a learning objective, the capability line, Section 6, the "Next experiments" and the README. Every learner who tries it on a GPU loses the fine-tune time and gets a crash. Their notebook is left in a partial state: `ACTIVE_MODEL` is stale from any earlier run, and cells 15–19 then report the previous predictor.
- **Evidence:**
  - Source inspection: `math.` appears only in cell 13, and `import math` appears nowhere (`results.json` → `S_static`).
  - Direct execution, CPU, with a stand-in candidate: cell 13 was run with `RUN_FINE_TUNING=True` and `torch.cuda.is_available` stubbed to `True`. The candidate's `fit` was replaced by a stand-in that reuses the pretrained predictor, with no gradient step. Cell 13's own code then printed `fine-tuned holdout {...}` and raised `NameError: name 'math' is not defined` (`P2_finetune_selection_standin`).
  - The real fine-tune itself was not run.
- **Recommended correction:** Add `import math` to the cell-13 template (or the carrier module), or replace `math.isfinite` with `np.isfinite`. Add an offline test that executes cell 13's selection block with stub metrics.
- **Acceptance check:** Run cell 13 with `RUN_FINE_TUNING=True`, a holdout of at least 50 rows and any candidate. It completes and prints `recommended_for_export` with `selection_basis` `holdout:<metric>`. A test in `tests/` fails on the reviewed blob and passes on the fix.
- **Spec:** EVAL14, ART8, UX7 (the experiment leaves the notebook inconsistent), UX5.

### MCC-M2 (Major): Mitra's in-context support is 80% of the stated support rows; the "same rows" comparison and the provenance are wrong

- **Cell/section:** cell 13 (Section 6) and the carried `fit_mitra_predictor` (`mitra_pipeline/tutorial_api.py`, `predictor.fit(...)` at line 338). Related places:
  - the Section 6 prose, "fitted on the exact same support rows" (template line 246);
  - the opening, "in-context conditioning on the training split";
  - the cell-19 `run_metadata['train_rows_used']` (template line 457).
- **Observed issue:** `TabularPredictor.fit` is called without `tuning_data`. AutoGluon therefore "Automatically generat[es] train/validation split with holdout_frac=0.2, Train Rows: 272, Val Rows: 69". AutoGluon's Mitra wrapper keeps only the train part as its context: `sklearn_interface.py` sets `self.X, self.y = X, y` and predicts with `trainer.predict(self.X, self.y, X)`.
  - The registered Mitra predictor is conditioned on **272** of the 341 support rows.
  - LightGBM and Random Forest are fitted on all **341**.
  - The exported `tutorial_run_metadata.json` records `train_rows_used: 341`.
  - The log line `0.9565 = Validation score (accuracy)` is computed on the 69 hidden rows and is never explained. It sits next to the holdout accuracy of 0.982.
  - On Wine, the context is 84 of 106 rows. On Telco it is 1,152 of 1,440.
  - The prose also cites "EVAL15" for equal conditions; in spec 2.2, EVAL15 is the single-metric rule.
- **Consequence:** The central comparison is presented as equal-condition when it is not. Here it favours the baselines, so it does not inflate Mitra. The deployed bundle's provenance misstates what the model was conditioned on. A learner who reads "0.9565" in the log cannot tell which split produced it.
- **Evidence:**
  - Documented execution (Kaggle T4 log, cell 13): "Train Rows: 272, Val Rows: 69".
  - Direct execution (CPU): the fitted model's `model.X` has 272 rows, `load_data_internal('val')` has 69 rows, `X_train_tree` has 341 rows and `run_metadata.train_rows_used` is 341 (`P1_default`). Wine 84/106 (`P3_active_wine`).
  - Source inspection of AutoGluon 1.5.0 `models/mitra/sklearn_interface.py` lines 295–330 and 374.
- **Recommended correction:** Condition Mitra on the full support set and state it. Options:
  - Pass AutoGluon a fixed, documented split. Supply the holdout as `tuning_data` only if it is not also used for selection. Or carve a separate validation slice from support, and report it.
  - Or call `predictor.refit_full()` and export the refit model, so the context equals the support rows.
  - Or fit the baselines on the same 272 rows that AutoGluon kept.

  In every case, print the context size and record it truthfully in `run_metadata`. Explain AutoGluon's internal validation score, or suppress it.
- **Acceptance check:** After cell 13, the printed Mitra context size equals `len(X_train_tree)` and equals `run_metadata['train_rows_used']`. Alternatively, the notebook prints both numbers and the prose states the difference. The Section 6 prose no longer claims "exact same support rows" unless the first condition holds.
- **Spec:** OUT8, FT3, GDL7, UX2. Comparison validity is framework §2 dimension 3.

### MCC-M3 (Major): declared GUIDED, but the guided layer is absent and infrastructure dominates

- **Cell/section:** the whole notebook, especially cells 0–2, cell 5, and the markdown before cells 9–19. Generator: `tools/notebook_template.py` markdown blocks; `tools/build_notebook.py` carrier cell.
- **Observed issue:** Missing elements, with the spec ID each would satisfy:

  | Missing element | Spec ID |
  |---|---|
  | Stated audience | GDL1 |
  | How to use (Run all, forms, which cells are infrastructure) | GDL2 |
  | Roadmap | GDL3 |
  | Input → Model → Output contract | GDL4 |
  | Observable objectives (the objectives are one long sentence of procedures) | GDL5 |
  | Glossary (in-context conditioning, support rows, MCC, one-vs-rest AUC) | GDL6 |
  | Question or prediction before the baseline comparison | GDL7 |
  | "Expected result / What to notice" notes | GDL8, UX4 |
  | Checkpoints with sample answers | GDL9 |
  | Predict → change → run → observe → explain activity (the "Next experiments" are a list of switches) | GDL10 |
  | Section syntheses | UX8 |
  | Troubleshooting (memory skip, Hub download, GPU, BYOD target name) | GDL13 |
  | Conclusion template | GDL14 |

  The 1,190-line carrier (cell 5) has no Infrastructure label or "you may run without reading" note, and no `cellView: form` / hidden-source metadata (GDL11, GDL12). The static probe's case-insensitive "Predict"/"Checkpoint" hits are API words (`predict`, model checkpoint), not guided elements.
- **Consequence:** A learner who meets the stated prerequisites can run the notebook but gets no help deciding what normal output looks like, or what the comparison means. They meet roughly 100 lines of AutoGluon log and a 54 kB carrier before the lesson.

  A troubleshooting note matters concretely here. When AutoGluon's memory estimate exceeds the available RAM, Mitra is *skipped*, and the only message is the generic "No models were trained successfully", raised after a long log. This review hit exactly that on a 3.5 GB-free host. The fix is the notebook's own `MAX_MEMORY_USAGE_RATIO` control, which nothing tells the learner about.
- **Evidence:** Source inspection, cells 0–20; static marker scan (`S_static`). Direct execution: the memory-skip message (`notes.host_deviation`).
- **Recommended correction:** Add the GDL elements in the template's markdown:
  - who the notebook is for and how to use it;
  - Input → Model → Output;
  - a prediction prompt before Section 6, for example "Will a pretrained in-context model beat LightGBM on 341 rows? By how many holdout rows?";
  - "What to notice" after Sections 3, 5, 6, 7 and 9;
  - two checkpoints with collapsible answers;
  - one bounded predict–change–run–explain activity (Wine, or a capped support set);
  - a troubleshooting table that includes the memory skip and `MAX_MEMORY_USAGE_RATIO`;
  - a conclusion template.

  Label and collapse cell 5 as Infrastructure.
- **Acceptance check:** The rendered notebook contains each element named above. Cell 5 has `cellView: form` (or equivalent hidden-source metadata) and an Infrastructure label. A learner-facing troubleshooting entry names `MAX_MEMORY_USAGE_RATIO`.
- **Spec:** GDL1–GDL14 (SHOULD), UX4, UX8.

### MCC-m1 (Minor): the opening is a reviewer-facing wall of text

- **Cell/section:** cell 0 (5,826 characters in one markdown cell); template line 44.
- **Observed issue:** The "Run all" paragraph tells "a reviewer reading NOTEBOOK_SPEC 2.0 RUN7/FT2" to "treat that as an open decision". Capability, standalone carrier, Run all, BYOD, model notes, objectives and non-goals are packed into one cell.
- **Consequence:** A learner cannot tell what is essential, and the first instruction they read is addressed to someone else. The "open decision" is moot under spec 2.2: §FT names in-context support registration as an adaptation stage.
- **Evidence:** Source inspection.
- **Recommended correction:** Remove the reviewer sentence. Split cell 0 into a short learner opening and a collapsible "About this notebook" block.
- **Acceptance check:** No learner-facing cell mentions reviewers or spec IDs as open decisions. Cell 0 is at most about 1,500 characters.
- **Spec:** GDL1, GDL2, GDL12.

### MCC-m2 (Minor): unexplained warnings and log volume in Section 6

- **Cell/section:** cell 13.
- **Observed issue:**
  - AutoGluon prints its system-info block, a presets advertisement ("presets='extreme' … Massively better than 'best'") and about 90 lines of preprocessing log.
  - scikit-learn warns that the probabilities "do not sum to one" (Kaggle record: `y_pred` wording, twice; this review's CPU run: `y_prob` wording). This comes from Mitra's float32 probabilities in `classification_metrics`' `log_loss`.
  - AutoGluon notes that it "arbitrarily selected" `malignant` as the positive class.

  None of these is explained.
- **Consequence:** The learner can mistake the warnings for errors, or the advert for advice. Which class is positive matters for the F1, precision and recall in the `AutoGluon evaluate` dict, and it is not stated.
- **Evidence:**
  - Documented execution: Kaggle T4 cell 13 stderr shows the two sum-to-one warnings and the positive-class note.
  - Direct execution: `P1_default.cells[5].warnings`.
- **Recommended correction:** Fit with `verbosity=1` (or capture the log into a collapsed output), and renormalise probabilities before `log_loss`. Add a one-line note that names the positive class.
- **Acceptance check:** Default cell-13 output is under about 30 lines, with no `y_prob` warning, and the positive class is printed.
- **Spec:** UX4, GDL8, UX11.

### MCC-m3 (Minor): the model comparison is printed without help reading row-level differences

- **Cell/section:** cell 13 table; "Interpretation and limits" (template line 513).
- **Observed issue:** On the default sample, Mitra scores 112/114 on the holdout, LightGBM 110 and Random Forest 109. On the test partition, Mitra and Random Forest both score 112 and LightGBM 111. The notebook prints `one_row_resolution_pct: 0.88`. The interpretation then says "The executable baselines show when the foundation model adds value on this table", with no prompt to count rows or note that the test ranking differs from the holdout ranking. On Wine, Mitra ties the trees on the holdout and trails them on test (0.972 vs 1.000). On Telco BYOD its log loss is clearly better (0.41 vs 0.54/0.60).
- **Consequence:** A learner is likely to read a 1–3-row accuracy gap as Mitra being better.
- **Evidence:** Documented execution (Kaggle table) and direct execution (`P1_default`, `P3_active_wine`, `P4_byod.R5_presplit_telco_full`).
- **Recommended correction:** After the table, add a "What to notice" that converts differences into rows. Point to the probability metrics (log loss, AUC), where the gap is larger, and to the holdout/test ranking change. Optionally add a paired bootstrap interval, as the customer capstone does.
- **Acceptance check:** The markdown after cell 13 asks the learner to express the holdout gap in rows and to compare holdout and test rankings. No sentence implies superiority from a gap below 3 rows.
- **Spec:** GDL7, GDL14, EVAL6.

### MCC-m4 (Minor): BYOD instructions don't match the upload contract

- **Cell/section:** cell 9 (Section 4); "Next experiments" (template line 526); Prerequisites.
- **Observed issue:**
  - "Next experiments" says to use the repository's `examples/sample-data` **archives** with `Upload pre-split train/val/test`. That branch accepts only `train.csv`, `val.csv` and `test.csv`. Uploading the ZIP fails with `Upload train.csv, val.csv, and test.csv together. Missing: ['test.csv', 'train.csv', 'val.csv']`, with no "extract the archive" hint.
  - The archives' targets are `Churn` and `class`, but `TARGET_COLUMN` stays `'target'`, and no BYOD text says to change it. The failure appears only in Section 5, after the upload: `pre-split upload:train: target 'target' not found.`
  - Selecting an upload source without `USE_BYOD` raises "Set USE_BYOD=True …", which is actionable, but the two controls are redundant.
- **Consequence:** A learner following the notebook's own next step fails twice before reaching a working BYOD run.
- **Evidence:** Direct execution, CPU, upload stubbed (`P4_byod` R1, R2, R3). Positive path: Telco with `TARGET_COLUMN='Churn'` reached every stage (validation, overlap 0/0/0, Mitra, baselines, report, export, verified reload, 20-row `NEW_DATA_PATH` inference). Invalid inputs were refused clearly: 11 classes ("target has 11 classes; Mitra requires 2-10"), and an inference CSV missing `gender`. Adult single-CSV validated and split 1,200/300, but its fit was not verified (host RAM).
- **Recommended correction:** Say "extract the archive and upload its three CSVs". Name each archive's target column. Make the missing-target error say "set `TARGET_COLUMN` in Section 4". Validate the target in cell 9, before the upload is consumed. Either derive `USE_BYOD` from `DATA_SOURCE`, or document why both exist.
- **Acceptance check:** Uploading the Telco ZIP produces a message that names extraction. Pre-split Telco with default `TARGET_COLUMN` fails in cell 9 with a message naming `TARGET_COLUMN` and the available columns.
- **Spec:** DAT12, DAT19, UX10.

### MCC-m5 (Minor): repository status records contradict each other and the notebook

- **Cell/section:** `docs/release-verification.md` "Current status"; `STATUS.md`; `README.md`; `tutorials/README.md`.
- **Observed issue:**
  - `release-verification.md` records the Kaggle T4 PASS of this blob, but its "Current status" says "No clean-runtime execution of the standalone notebooks has been recorded yet", and in the same sentence that GPU evidence "is now recorded below".
  - `STATUS.md` says "awaiting clean-runtime execution" and cites spec 1.1. `README.md` says spec 1.1. The notebook and `tutorials/README.md` say 2.0.
  - `tutorials/README.md` marks the evidence "verified" while the status column says it "must be reviewed".
  - Procedure step 5 expects `config.json` at 81 bytes; the manifest says 86.
- **Consequence:** A maintainer cannot tell whether the default path is evidenced, or against which spec. A reader may rerun work, or promote on the wrong basis.
- **Evidence:** Source inspection of the four files at the reviewed commit.
- **Recommended correction:** Rewrite "Current status" to cite the Kaggle row and its boundary: nbclient with a shim, not a browser Colab `Run all`; default path only. Align `STATUS.md` and `README.md` with the notebook's declared spec. Fix the byte count.
- **Acceptance check:** The four files name the same spec version and the same evidence state. "Current status" no longer contradicts the table.
- **Spec:** REL1, REL10.

### MCC-m6 (Minor): the "Next experiments" have no rerun scope or expected outcome, and two of them interact badly

- **Cell/section:** cell 20 "Next experiments"; cell 13 `MIN_SELECTION_HOLDOUT_ROWS = 50`.
- **Observed issue:**
  - "Switch `DATA_SOURCE` to `Sample: Wine`" does not say to rerun from Section 4. This review did that, and it worked.
  - Wine's holdout is 36 rows, below `MIN_SELECTION_HOLDOUT_ROWS = 50`. Fine-tuning on Wine therefore always reports `default:pretrained; holdout-too-small:36<50`, and the selection the learner was told to "watch" never happens. Nothing warns of this.
  - On CPU, `RUN_FINE_TUNING=True` refits the pretrained predictor before refusing with the GPU message. The refusal itself is clear.
- **Consequence:** The learner cannot predict what each experiment should show, and one combination silently does nothing.
- **Evidence:**
  - Direct execution: Wine through cells 9–19, 106/36/36 rows, multiclass, with all cells passing (`P3_active_wine`).
  - Real CPU gate message observed after the pretrained refit (`notes.P2b_first_run`).
  - The 36 < 50 interaction follows from the source (not executed on a GPU).
- **Recommended correction:** For each experiment, state the cells to rerun and the expected observation. Note the 50-row selection floor. Move the GPU check before the pretrained refit.
- **Acceptance check:** Each "Next experiment" names its rerun range and what to look for. The Wine plus fine-tune combination is either documented or made selectable.
- **Spec:** GDL10, UX5, UX9.

### Suggestions

- **MCC-S1:** Regenerate against NOTEBOOK_SPEC 2.2. The notebook declares 2.0.
- **MCC-S2:** Use the Wine test result (Mitra 0.972 vs trees 1.000) and the Telco log-loss gap as a short interpretation activity on when a foundation model helps.
- **MCC-S3:** Expose `positive_class` (or name it in the prose), so that AutoGluon's `f1`/`precision`/`recall` refer to a stated class.

## 4. Readiness

**Needs revision.** No blocker. Three majors are open:

- MCC-M1: the promised fine-tuning path crashes;
- MCC-M2: the comparison conditions and provenance are misstated;
- MCC-M3: the guided layer is missing for a `GUIDED` notebook.

OUT8 is a MUST-level conformance gap (M2).

Remaining gates after the fixes:

1. An exact-revision clean `Run all` on Colab. Only a Kaggle nbclient run exists.
2. A real GPU fine-tuning run that reaches selection, export and reload.
3. A representative BYOD run under the hosted runtime (REL12).
4. Learner observation, if a claim about learning effectiveness is wanted.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, labelled deviations):**
  - The default path passes, and its metrics equal the hosted record.
  - Mitra's context is 272/341 rows (84/106 on Wine).
  - `run_metadata.train_rows_used` is 341.
  - With a stand-in candidate, cell 13's selection code raises `NameError: math`.
  - The Wine experiment passes through cells 9–19.
  - Telco pre-split BYOD reaches every stage, including verified reload and `NEW_DATA_PATH` inference.
  - Four invalid inputs give the messages quoted.
- **Verified from documented execution:** the Kaggle T4 run of this blob, with one pass, no restart, 9/9 cells, and the AutoGluon log lines quoted.
- **Inferred:**
  - That a real GPU fine-tune reaches `better()` and crashes. The stand-in exercises the same lines after `candidate.fit`.
  - That the Colab browser `Run all` behaves like the Kaggle run.
  - That the memory skip affects only low-RAM hosts. Colab CPU has about 12 GB against the 7.08 GB estimate.
- **Most likely to be wrong:** MCC-M2's severity. The unequal conditions favour the baselines, so the notebook does not overstate Mitra, and AutoGluon's internal validation split is standard behaviour. It could be argued Minor. It is graded Major because two things are factually wrong: the stated equal-condition comparison (the central demonstration), and the exported provenance, which records 341 support rows for a model conditioned on 272.

Probe files: `mitra_classifier_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
