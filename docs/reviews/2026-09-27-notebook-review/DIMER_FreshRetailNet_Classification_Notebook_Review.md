# DIMER Notebook Review — FreshRetailNet Multi-Model Classification, guided v2

**Review date:** 27 September 2026  
**Decision:** **Needs revision. Fresh Colab/T4 execution verification remains pending.**  
**Method:** Notebook Review Framework v1; source and recorded-output inspection, with targeted offline reproductions of selected logic.

## Executive assessment

The notebook substantially implements its central comparative lesson. It is not merely a sequence of model-loading examples: it constructs classical baselines, runs four tabular foundation-model conditions, aligns probability columns, compares validation results, creates model artifacts, reloads them for held-out prediction, and supports an evidence-based conclusion. Its learner orientation and explanation of the retail prediction task are considerable strengths. [S1: opening; §§4–9]

The reasons to withhold readiness are more specific: an optional fine-tuned TabICLv2 ablation changes the adaptation mode as well as the features; a worked answer attributes a paired interval to the wrong comparison; data-acquisition retries can fail after an ordinary correction; and the frozen-test stage takes effective prediction configuration from mutable notebook state rather than the checked development configuration. These are important precisely because a learner could follow the activity, obtain a plausible result, and misunderstand what was actually tested.

There are **four major findings and three minor findings** below. No fresh-default-path blocker was established. That is not a clean-runtime pass: this review did not execute the complete notebook in Colab or reproduce its model scores.

## 1. Review contract and evidence

| Item | Reviewed value |
|---|---|
| Notebook | `tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb` |
| Repository | `kurtvalcorza/mitra-classifier-pipeline` |
| Reviewed commit | `6c9f911294712eccf993817ef7c48c9f342bc194` |
| Notebook Git blob | `c4e850d37d62f1b12f15a6df798b4fe388ded63b` |
| Declared profile / mode | `E2E` / `WORKSHOP` |
| Intended learner | Can operate a hosted notebook and read basic Python; new to machine learning |
| Documented runtime | T4 GPU recommended; isolated Python 3.12 model environments |
| Notebook's declared specification | 2.1 |
| Current specification inspected | 2.2, dated 26 September 2026; Git blob `046d866eac7cc67b1539a4ba370ebc965341c87b` |
| Sample-data source revision | `469d91252f3583b38d08b5c4d90fef4848b93f24` |
| Sample archive SHA-256 declared by notebook | `ad2d2a8729749bb055754e4867acfb048fc27816f4eb344961d962b59c0be6dd` |

Source: [S1; S2; S3]. The sample digest is recorded from source, not independently verified against a downloaded sample during this review.

**Evidence inspected.** Learner-facing instructions and worked answers; principal host-side control flow; the embedded local model runner; acquisition and validation; baseline construction; optional ablations; freeze and test logic; bootstrap calculations; inference preview; report export and completion summary; saved text/table outputs; and the pinned dataset builder.

**Direct local work.** Executed small, explicitly transcribed or reduced reproductions using synthetic inputs. These exercised archive staging/recovery, probability alignment, TabICLv2 adaptation dispatch, and the distinction between a checked configuration file and a live configuration mapping. A synthetic paired-comparison counterexample illustrates why the worked answer's inference does not follow. The scripts and outputs are supplied separately.

**Not established.** A complete fresh Colab/T4 run; current download/package compatibility; numerical reproduction of the saved foundation-model results; full valid-BYOD processing; the fast path end-to-end; optional GPU fine-tuning; artifact relocation or prediction-equivalence testing; rendered Colab layout/accessibility; or learning outcomes observed in actual participants. The large carried adapter payloads were not independently line-audited in their entirety. External model-license and dataset-card assertions were not independently audited.

The notebook retains saved outputs while its code-cell execution counts are null. Those outputs are useful recorded evidence, not proof of a clean execution of the reviewed revision. Specification references below refer to the inspected **2.2** text; they do not imply that all requirements were audited or that this notebook claims 2.2 conformance.

## 2. Judgments across the eight dimensions

| Dimension | Assessment |
|---|---|
| Promise fulfillment | The central comparison is materially implemented and supported by saved outputs. The optional fine-tuned ablation breaks its one-factor promise. |
| Technical correctness | Useful identity and artifact checks coexist with a reproducible acquisition-retry defect and a frozen-configuration integrity gap. |
| Scientific and experimental validity | The builder preserves per-series temporal separation and training-derived bands. The ablation dispatch and worked interval interpretation require correction. |
| Learner orientation and progression | Strong source-level design: audience, roadmap, task contract, terms, infrastructure labels, fast path, and checkpoints are present. No learner observation or rendered-interface test was performed. |
| Explanations and interpretation | Generally substantial, but one worked answer teaches the wrong comparator for a paired interval. Some reproducibility statements are too absolute. |
| Meaningful learner activity | Prediction and bounded ablation activities are real, rather than placeholder exercises. Not every supported condition preserves the advertised experimental factor. |
| Interaction, pacing, and recovery | Progress logs and troubleshooting exist. Normal data corrections can leave learners stuck; practical cold-start pacing remains unverified. |
| Completion and transfer | Clear conclusion prompts, checklist, and export exist. The report bundle omits some useful audit evidence; a direct unlabeled-input demonstration would make reuse easier. |

These are evidence-qualified judgments, not a numerical score or a substitute for execution validation.

## 3. Major findings

### NR-01 — Fine-tuned TabICLv2 ablation silently changes adaptation mode

**Severity:** Major.  
**Affected journey:** Optional active-learning path after enabling TabICLv2 fine-tuning. The default in-context-only run is not affected by this particular defect.  
**Locations:** §7.1, cell `UKzaZmJxFXEm`; §5.4 `run_foundation_condition`; §5.3 embedded `run_tabicl_development`, cell `CUjAQzh9FXEl`.

The exercise tells learners to hold the model condition fixed while removing a feature group. The ablation scheduler instead appends the feature-group name to the condition, producing values such as `fine_tuned_no_stockout`. The TabICLv2 runner enters its fine-tuning branch only when `config["condition"] == "fine_tuned"`. The appended condition fails that test, so the runner uses the pretrained in-context path. Its returned effective mode also follows that exact-equality decision. [S1: §§5.3–5.4, 7.1]

The resulting comparison can therefore be:

`fine-tuned, full features` versus `in-context, reduced features`

rather than the promised feature-only ablation. A learner could attribute a difference to removing stockout columns when the adaptation method also changed. The exercise's comparison table does not expose this effective-mode mismatch prominently enough to prevent that reading.

**Evidence:** Source tracing and an isolated dispatch reproduction. Both `fine_tuned_no_stockout` and `fine_tuned_no_period_proxies` select the non-fine-tuned branch. The corresponding in-context cases retain their original mode. No actual GPU fine-tuning run was performed.

**Correction:** Represent adaptation mode separately from the ablation label, for example `adaptation_mode="fine_tuned"` and `ablation_group="no_stockout"`. Dispatch from the immutable adaptation field, not a display string. Assert and display that the full and ablated conditions have the same effective adaptation mode before calculating their difference.

**Acceptance check:** Test both TabICLv2 adaptation modes with both feature groups. Only the intended feature set should change; the adaptation protocol, checkpoint-selection rule, seed, support rows, and splits should remain fixed. A mode mismatch must stop or invalidate the comparison, not yield an ordinary ablation row. Confirm the corrected fine-tuned path with a recorded GPU run.

**Framework mapping:** Promise fulfillment; experimental validity; meaningful learner activity. Related specification expectations: GDL10, FT3, UX5–UX6.

### NR-02 — The worked test answer uses an interval against the wrong model

**Severity:** Major.  
**Affected journey:** Default learner interpretation and final conclusion.  
**Locations:** §8.2 bootstrap cell `166858dd2aa4`; Test checkpoint cell `79f4f7fdc9f4`, answer 4.

The bootstrap selects the best **validation** classical baseline as its reference. In the saved run, that is Random Forest. Each `gain_ci_*` interval then compares a candidate with that reference. This reference selection itself is appropriate; it is not selected by test performance. [S1: §8.2]

The worked answer discusses removing month and weather from **LightGBM**, cites its saved test scores of about **0.5205** with the features and **0.4965** without them, and then invokes an interval below zero to support the ablation interpretation. The displayed interval for the ablated model is against **Random Forest**, not against full-feature LightGBM. [S1: Test checkpoint]

The observed score decrease is real in the saved table. What is not established by that displayed interval is the uncertainty of the **ablated LightGBM minus full LightGBM** difference. An interval against another model cannot answer that question.

**Evidence:** Source and saved-output inspection. A synthetic counterexample supplied with this review has identical full and ablated predictions: their paired difference is exactly zero, while their differences from a stronger separate reference are negative. This illustrates the logical issue; it is not a reanalysis of the notebook's dataset.

**Correction:** Compute a separate paired ablation contrast using full-feature LightGBM as the comparator. Keep the existing Random Forest reference comparison for the broader model comparison, but explicitly name the comparator in each table or row. Generate the worked answer from the appropriate contrast, or confine it to the observed point-score difference.

Retain the caveat about correlated store–product rows. Row-wise bootstrap intervals should remain an illustration with a dependence limitation, not a certification that an effect generalizes across series or periods.

**Acceptance check:** Every sentence about a paired interval must identify the same pair of models used to compute it. If the notebook only computed ablated-versus-Random-Forest intervals, it must not present them as ablated-versus-full-LightGBM evidence. Include a regression test with identical full and ablated predictions.

**Framework mapping:** Explanations and interpretation; experimental validity; evidence-based conclusions. Related specification expectations: EVAL3, GDL8–GDL9, GDL14.

### NR-03 — Correcting or repeating data acquisition can fail because staging is not retry-safe

**Severity:** Major.  
**Affected journey:** BYOD, correction of an invalid archive, or rerunning acquisition in an existing session. A first acquisition in a new session is not shown to fail.  
**Location:** §1.1, cell `ZDCzBnsvFXEe`.

The acquisition cell creates `DATA_ROOT / "staged"` with `exist_ok=False` before checking that the archive contains every required split. If an archive is missing `test.csv`, the notebook raises a useful error but leaves the staging directory behind. After correcting the ZIP, rerunning the cell fails at directory creation with `FileExistsError`, before the corrected input can be checked. Repeating a successful acquisition also reaches this failure. [S1: §1.1]

This turns an ordinary learner correction into a workspace-management problem. Re-running setup creates a new session and can work around it, but the data-acquisition instructions do not provide a complete recovery procedure that also handles downstream state.

**Evidence:** Direct execution of the transcribed staging block with synthetic archives:

| Scenario | Observed result |
|---|---|
| First valid archive | Splits staged successfully |
| Missing `test.csv` | Expected missing-split `ValueError` |
| Correct the ZIP and retry in the same session | `FileExistsError` |
| Repeat a successful acquisition | `FileExistsError` |
| Traversal or duplicate split member | Rejected as expected |

**Correction:** Validate/stage transactionally in a new temporary directory, clean it on failure, and switch active data only after validation succeeds. When the dataset changes, invalidate or explicitly reset dependent frames, model results, freeze records, and test outputs. Simply changing to `exist_ok=True` without addressing stale state is not a sufficient fix.

**Acceptance check:** A first run, a repeat run, an invalid-then-corrected ZIP, and a switch from sample to BYOD must all have predictable documented behavior. Failed attempts must not block correction; old results must not be represented as results for the new data.

**Framework mapping:** Interaction and recovery; technical correctness; completion and transfer. Related specification expectations: DAT19, UX10, GDL13.

### NR-04 — The frozen-test stage is not bound to the configuration it verifies

**Severity:** Major.  
**Affected journey:** Changing notebook configuration after freezing, then rerunning the test stage. This is not evidence that the recorded default results used changed settings.  
**Locations:** §8.0, cell `aNhMl_zxFXEn`; §8.1, cell `S8SYDvNeFXEn`; §5.3 `run_test`.

The freeze records a digest of the development `run_config.json`. The test stage verifies that file's digest, but builds a new test configuration using the current `CONDITION_SPECS[key]["configuration"]` and other live notebook variables. It does not reconstruct the effective prediction settings from the verified development configuration. [S1: §§8.0–8.1]

For TabDPT, this matters directly: the test runner uses the supplied ensemble count, context size, and batch size when predicting. Changing an inference setting in §5.1 and rerunning that configuration cell can therefore alter test execution while the original development file remains unchanged and its hash check still passes. [S1: §5.3 `run_test`]

This undercuts the assurance that the frozen test measures exactly the configuration selected during development. The check protects the stored file, but not the full configuration actually consumed by the test runner.

**Evidence:** Source tracing and a reduced control-flow reproduction: a checked development file retains an ensemble count of 4 while the test configuration is built from a live mapping with 8. No resulting numerical model-score change was measured in this review.

**Correction:** Read the verified development configuration and derive the test request from it. Permit only an explicit list of non-semantic changes such as phase, output location, and evaluation-input location. Treat inference hyperparameters, feature order, class order, model identity, and relevant execution settings as frozen. Where portability requires a change, record it as a new or explicitly different evaluation protocol.

**Acceptance check:** Freeze a TabDPT run, change the live ensemble count or context size, and invoke the test stage. It must either use the original frozen value or reject the mismatch. Its exported effective configuration must match what was actually executed.

**Framework mapping:** Technical correctness; promise fulfillment; experimental validity. Related specification expectations: EVAL8, EVAL14, OUT7–OUT8.

## 4. Minor findings

### NR-05 — Host-side dependency control does not support the strongest reproducibility statements

**Location:** §0.1, cell `xUUykd5OFXEc`; setup seed explanation; Test checkpoint.

The model stacks have isolated dependency handling, but the host stack used for baselines and metrics accepts the installed NumPy, pandas, scikit-learn, and matplotlib versions without a tested-range check. LightGBM is pinned only when absent; an already-installed version is accepted. Printing versions records the environment but does not constrain it. The text nevertheless suggests that fixing the seed guarantees identical results, and the Test checkpoint describes classical results as unchanged across revisions. [S1: §0.1; Test checkpoint]

**Impact:** A learner may mistake a legitimate environment difference for an error, or assume the reference answers are invariant. This is a conformance gap against the inspected 2.2 ENV1–ENV2, separate from the severity of any particular numerical difference; no such difference was measured here.

**Correction and acceptance:** Verify a documented tested host range before baseline execution, without requiring a mid-run restart; record the effective versions. Label worked numbers as a particular reference run, explain material residual variability, and ensure the instructions remain valid when a compatible run produces different scores. Do not equate a fixed seed with a universal reproducibility guarantee.

### NR-06 — Pre-freeze test-feature inspection makes the development boundary inconsistent

**Location:** §2.2 Look at the table; §7 month/weather rationale; §8 freeze explanation.

The table summary explicitly computes test-feature minima, medians, and maxima before freezing. The month/weather explanation also discusses the test period in advance. Elsewhere, the narrative describes the test partition as held back until the freeze and asks learners to predict from training/validation evidence only. [S1: §§2.2, 7–8]

This is **not an established test-label leak**, and it does not prove the saved scores are invalid. It is a methodological and teaching-boundary inconsistency: test-feature knowledge can inform which hypotheses a learner chooses to investigate.

**Correction and acceptance:** Either move test-specific descriptive summaries and interpretations after the freeze, or explicitly distinguish permitted advance knowledge of the test covariates from withheld test labels and acknowledge the resulting protocol. The instructions, tables, and interpretation prompts should describe the same boundary.

### NR-07 — The exported report is weaker as an audit record than the runtime's available evidence

**Location:** §9.1, cell `Da6jGcVwFXEo`.

The report ZIP includes aggregate tables, intervals, figures, freeze metadata, a manifest, and a short inference preview. It does not copy the complete per-row prediction records or full run configurations into the bundle. It records a generic notebook revision string, rather than the exact reviewed notebook commit or source identity. Some freeze entries point to runtime-local files that are not included. [S1: §9.1]

The notebook explicitly says it does not bundle foundation-model weights; that exclusion is not itself a defect. The narrower issue is that a participant retaining only the report has less evidence for independently checking its metrics and paired comparisons after the session ends.

**Correction and acceptance:** Include the small full prediction tables, effective configurations, exact notebook identity, comparison definitions, and relevant environment records, subject to a clear BYOD privacy/export notice. It should be possible to recompute reported metrics and paired contrasts from the permitted exported evidence without rerunning the models. Do not redistribute model weights merely to solve this reporting issue.

**Framework mapping:** Completion and transfer; provenance and reproducibility. Related specification expectations: OUT2–OUT3, OUT7–OUT8.

## 5. Learner experience: preserve what is working

The notebook gives the learner a concrete unit of analysis before asking them to interpret model scores: one store–product–day row, historical features, and an observed-sales category seven days later. Its distinction between observed sales and unconstrained customer demand is especially useful. The pinned builder supports the described training-derived bands and temporal embargo structure. This is source corroboration, not an independent regeneration of the sample. [S1: Know your data; S3]

The baseline ladder makes the comparison meaningful rather than treating a foundation model as automatically useful. Class-order alignment is explicit, and the host metric implementation maps class labels consistently before probability-aware scoring. The selected-model requirement is strict by default: failures are not silently presented as a successful four-model comparison. [S1: §§0.1, 4, 5.4]

Orientation, terminology, worked checkpoints, collapsed infrastructure, and an evidence-based conclusion template are already present. The optional activity asks learners to predict, change one experimental factor, observe, and explain, and warns against reusing exposed test scores as unbiased evidence. The fix should repair its supported branches, not replace the activity with another passive display. [S1: How to use; §7; conclusion]

The highest-priority instructional repair is the answer key. A successful code run does not protect a beginner from learning an incorrect inferential rule. The wrong-comparator finding demonstrates why the prose and results must be reviewed together.

### Useful improvements, not mandatory defects

**Add a direct unlabeled-input example.** Section 8.3 is accurately described as a preview of already-computed predictions. It does not itself call a model or accept a fresh feature-only CSV. A small additional example that loads a frozen model, validates an unlabeled input, predicts, and exports would make the transition from evaluation to practical use clearer. This is not an automatic INF2 violation: the actual test stage already predicts rows distinct from adaptation data. [S1: §§5.3, 8.1, 8.3; S2: INF2]

**Make the report easy to retain.** A visible download action or file link, plus a short explanation of which exported files answer which learner questions, would improve the after-session experience. Keep any interactive download optional so it does not block the canonical run.

**Tighten explanations without adding more material.** Separate true metric ranges from chance/reference values; qualify calibration diagnoses based only on aggregate log loss; and specify when the prediction is issued relative to same-day stockout and weather measurements. These are focused edits, not a request to turn the notebook into a larger textbook.

## 6. Promise-to-evidence summary

| Promise | Evidence and conclusion |
|---|---|
| Compare classical methods with four foundation models | Implemented in source; all four have saved validation/test outputs. Not freshly rerun here. |
| Keep class-probability semantics consistent | Explicit alignment and numeric-label mapping; targeted alignment probes behaved as expected for valid ordering and missing/non-finite inputs. |
| Practice a one-factor ablation | Implemented, but the fine-tuned TabICLv2 branch violates the one-factor condition. |
| Interpret test uncertainty correctly | Bootstrap and paired reference comparison are implemented; one worked ablation interpretation names the wrong underlying comparison. |
| Evaluate frozen choices | Artifacts/configuration files are checked, but live prediction settings can still diverge from the checked development configuration. |
| Bring compatible user data | Real upload/path branch exists. Retry failure reproduced; complete valid-BYOD workflow remains unverified. |
| Demonstrate held-out inference | Actual held-out prediction occurs in the reloaded test stage; §8.3 is only a preview of those predictions. |
| Retain an experiment record | Report export is implemented; the bundle can retain more of the small evidence already available in the runtime. |

## 7. Readiness gates

Before calling this notebook ready for its intended guided/workshop use, resolve the four major findings, close or explicitly address the applicable host-dependency requirements, and record the remaining runtime evidence.

The execution record should identify the exact commit, environment, hardware, defaults, selected models, and completion status. It should cover a fresh T4 default run; the documented Mitra/TabICLv2 fast path; a valid BYOD example; invalid-input correction without stale state; the corrected fine-tuned TabICLv2 ablation; and attempted post-freeze configuration changes. Include at least one before/after-reload prediction-equivalence check with a declared tolerance where appropriate.

For learner validation, have a representative basic-Python learner complete the notebook without unrecorded instructor intervention. Ask them to identify the prediction target, distinguish in-context conditioning from fine-tuning, interpret the named paired comparison, perform and explain an ablation, recover from a data error, and retain the outputs. Record actual confusion points rather than assuming that the presence of a glossary or checkpoint proves learning.

**Final judgment:** Keep the core notebook and its guided structure. Repair the experiment semantics, answer-key evidence, recovery, and freeze boundary. Then verify the supported journeys in a fresh runtime. This review does not authorize or make repository changes.

## Sources and reproducibility attachments

- **S1 — Reviewed notebook, pinned revision:** https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/6c9f911294712eccf993817ef7c48c9f342bc194/tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb
- **S2 — DIMER Notebook Specification:** https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md. Inspected text: version 2.2, 26 September 2026; blob `046d866eac7cc67b1539a4ba370ebc965341c87b`. The URL follows `main`; the recorded version/blob identify the review baseline.
- **S3 — Pinned sample builder:** https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/469d91252f3583b38d08b5c4d90fef4848b93f24/examples/build_freshretailnet_dataset.py
- **Framework:** `DIMER_Notebook_Review_Framework_v1.md`, agreed in this conversation.
- **Offline staging/probability probes:** `review_probes.py` and `probe_results.json`.
- **Offline dispatch/freeze/comparator probes:** `control_flow_probes.py` and `control_flow_results.json`.

The companion probe archive contains scripts and their recorded outputs, not a copy of the notebook, the dataset, model weights, or a Colab execution log. Local probes used Python 3.13.5, NumPy 2.3.5, and scikit-learn 1.8.0. Synthetic examples must not be mistaken for measured performance of the notebook's models.
