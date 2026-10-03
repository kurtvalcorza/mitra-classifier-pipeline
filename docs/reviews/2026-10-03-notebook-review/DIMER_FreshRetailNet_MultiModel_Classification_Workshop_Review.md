# Notebook review (Framework v1): FreshRetailNet multi-model classification workshop, original v1

**Readiness: Needs revision.** 5 Major, 7 Minor, 4 Suggestions. No Blocker. The default path has
documented execution evidence for byte-identical code: a Colab T4 run with saved outputs, all 19 code cells,
4/4 foundation models, and a run-all completion line. That evidence is not bound to a commit, date or
executor, though. Three of the four defects that the 2026-09-27 review found in v2 (and that v2 2.1.1 fixed)
are still present in v1. The README does not tell a learner which of the two notebooks to open.

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/mitra-classifier-pipeline` |
| Revision | `0434a026cf755071e89bc33c876e67040c2a614d` (`main`, from `gh api .../commits/main`) |
| Notebook | `tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb`, git blob `6b8986dffb1af653b9129ae3ac2e005d76afc693`, file sha256 `c7986ea3…63599` |
| Declared spec / profile / mode | NOTEBOOK_SPEC **2.1**, `E2E`, `WORKSHOP`, standalone (embedded adapters); `workshop_revision` 2.1.0; `clean_runtime_evidence: pending` |
| Requirements baseline | NOTEBOOK_SPEC **2.2** at ml-worker `origin/main` `b1cfe133`. Its §32 note says 2.1 notebooks are not nonconformant solely for lacking the guided layer (GDL), so GDL gaps are treated as SHOULDs here |
| Audience / prerequisites | "College students with introductory Python and machine-learning experience" (cell 0) |
| Supported runtime | Colab or Jupyter. GPU recommended for the four-model exercise and required only for optional fine-tuning. Foundation models run in `uv` Python 3.12 venvs |
| Promised outcomes | The 9 "What you will do" items and 8 learning objectives (cells 0 and 2). Run all covers acquire → validate → baselines → 4 ICL foundation models → freeze → reload saved artifacts → test → bootstrap → inference preview → export |
| Scope | All 43 cells: markdown, code, form fields, embedded runner and adapters, export, failure messages; plus `tutorials/README.md` guidance |

### Evidence used

| Evidence | Basis | What it shows |
|---|---|---|
| `docs/execution-evidence/2026-09-26/legacy-classification-saved-outputs.ipynb` | Documented execution evidence | **0 of 43 cells differ** from the reviewed blob's source (P01). Execution counts run 1..19 with no gaps, 0 error outputs, Colab `gpuType: T4`, `cuda` device. Output includes "Successful foundation-model runs: 4 / 4", 11 frozen test models, and "Run-all complete". **No commit, date, runtime record or executor** is recorded for it. `release-verification.md:245` only says the outputs were "moved unchanged". |
| Direct execution: local Windows CPU, Python 3.12.10, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, lightgbm 4.6.0 | Direct execution | The notebook's own cells 0.1–4.1, run against the pinned sample, reproduce the legacy validation balanced accuracy of all 5 baselines to 4 dp (P04). Retry and rerun failures (P03, P04). BYOD rejections (P09). No foundation model was run. |
| Carried adapter check against the local clones at the pinned commits | Source inspection | 5/5 carried modules are byte-identical to the pinned upstream files (P02) |
| v2 review `docs/reviews/2026-09-27-notebook-review/` | Context only | Each NR finding was re-checked against v1 source and not assumed (see §3) |

**Not verified:** any foundation-model run at this revision, the optional fine-tuning paths (no GPU here),
the full BYOD downstream path, and CPU-only Run all time. Learner understanding is also not verified: there
was no learner observation.

### Journeys

| Journey | Result | Basis |
|---|---|---|
| First-time learner | **Partial.** The orientation is strong: research question, metric rationale, glossary, test-set rule, licence boundary. But 3 of the 8 objectives have no activity (FRC1-M5). There are no expected-output notes, no sample answers and no troubleshooting (FRC1-S1), and no guidance on which notebook to use (FRC1-M4). | Source inspection |
| Clean default | **Evidence exists for identical code; exact-revision record pending.** The saved run completed end to end on T4. The host stages reproduce locally to 4 dp. | Documented execution (unbound) + direct execution (CPU, host stages only) |
| Active learning | **Fails on the documented controls.** The fine-tuned TabICLv2 ablation silently runs in-context (FRC1-M1). Rerunning §4.1 crashes and empties the baseline registry (FRC1-M2). Changing §5.1 settings after freezing changes the test run without detection (FRC1-M3). | Source inspection + direct execution + reduced reproduction |
| Reuse and recovery | **Mixed.** Invalid BYOD input gets a clear rejection (P09). Correcting the ZIP or switching sample→BYOD in the same session is blocked by `FileExistsError` (FRC1-M2). The full BYOD downstream path is not verified. | Direct execution |

## 2. Separate judgments

- **Technical correctness:** the core path is sound. Probabilities are aligned to `low/mid/high` before
  scoring, the test stage reloads the saved development artifacts, the frozen artifacts are hashed, the ZIP is
  staged safely, and the carried adapters are verbatim. The defects are in state handling: retry and rerun
  safety (M2), the freeze-to-test binding (M3), and the ablation dispatch (M1).
- **Promise fulfilment:** the 9 "What you will do" items are all delivered on the default path (legacy
  evidence). The cell 7 claim about the time periods is correct: train covers months 4–5, validation 5, test 6
  (P07). The optional fine-tuned ablation does not do what it says (M1). Some learning objectives are not
  exercised (M5).
- **Learner experience:** good scientific framing and limitations, and the "avoid 'Model X is best'" prompt
  is appropriate. Weaknesses: no expected-result notes, no worked guidance, no troubleshooting, no view of the
  test shift after the freeze (m1), and an unexplained log-loss artefact (m2).
- **Spec conformance (2.2 baseline):** unresolved MUSTs are REL1/REL10 (no exact-revision execution
  record, m7), REL12 (BYOD not verified), ENV2 (host libraries unpinned, m3), DAT12/VAL6 (BYOD class-label
  rule and model ceilings not stated, m4), UX1 (objectives not matched to executed code, M5) and EVAL8 (test
  configuration is not guaranteed to equal the evaluated development configuration, M3). The GDL1–15 gaps are
  SHOULDs.

## 3. Findings

How v2 findings map onto v1, each checked against v1 source: NR-01 applies (FRC1-M1). NR-02 does not apply
as written, because v1 has no worked answer; the related ablation-contrast gap is FRC1-m6. NR-03 applies and
is wider in v1 (FRC1-M2). NR-04 applies (FRC1-M3). NR-05 applies (FRC1-m3). NR-06 applies only weakly,
because v1's EDA is development-only (FRC1-S3). NR-07 applies (FRC1-m5).

### FRC1-M1 (Major): the fine-tuned TabICLv2 ablation silently switches to in-context
- **Cell:** §7.1 (cell 29) `condition_override=... + "_" + FOUNDATION_ABLATION_GROUP`. Runner
  `run_tabicl_development` (cell 22) uses `if config["condition"] == "fine_tuned":` and computes
  `effective_mode` with the same exact match.
- **Issue:** with `RUN_TABICLV2_FINETUNED=True` and `RUN_FOUNDATION_ABLATION=True`, the ablated condition
  becomes `fine_tuned_no_stockout` or `fine_tuned_no_period_proxies`. Neither equals `fine_tuned`, so the
  runner takes the in-context branch. The ablation table then reports the difference between fine-tuned full
  features and in-context reduced features as a feature effect. Mitra is not affected, because it dispatches
  on `model_configuration.fine_tune`.
- **Consequence:** learners attribute a change of adaptation mode to removing stockout or period features.
- **Evidence:** source inspection plus a reduced dispatch reproduction: 2 of 2 fine-tuned ablations fall to
  in-context (P05). No GPU run was done.
- **Correction:** dispatch on a separate immutable `adaptation_mode` field and keep the ablation label
  separate. Assert that the full and ablated runs have the same `effective_mode` before computing `delta`.
- **Acceptance check:** for TabICLv2 in both modes × both ablation groups, the ablated run's `effective_mode`
  equals the base run's. A forced mismatch raises an error and produces no ablation row. A recorded GPU run of
  the fine-tuned ablation shows `effective_mode == "fine_tuned"`. Spec: FT3, GDL10, UX5.

### FRC1-M2 (Major): acquisition and baseline cells are not retry-safe; a failed rerun leaves state inconsistent
- **Cells:** §1.1 (cell 8) `extract_root.mkdir(exist_ok=False)`, run before the archive is validated. §4.1
  (cell 16) `register_baseline` uses `run_dir.mkdir(exist_ok=False)` after the registries have been reset to
  `{}`.
- **Issue and consequence:** a BYOD ZIP missing `test.csv` correctly raises a `ValueError`. Rerunning with a
  corrected ZIP then fails with `FileExistsError` before the corrected ZIP is checked. Repeating a successful
  acquisition, or switching sample→BYOD, fails the same way. Rerunning §4.1 also fails, and because the cell
  first resets `baseline_prediction_registry`, it **empties it (5 → 0 entries)**. Downstream cells (6.1, 8.0)
  then see missing baselines. The only workaround (rerun §0.1 to get a new session, then everything after it)
  is not documented.
- **Evidence:** direct execution of the notebook's own cells on local CPU. P03: 2 of 4 steps blocked by
  `FileExistsError` (invalid→corrected, repeat). P04: §4.1 rerun raises `FileExistsError`, and the registry
  goes from 5 to 0.
- **Correction:** stage into a fresh temporary directory, validate, then switch atomically; clean up on
  failure. When the dataset changes, reset all dependent state: frames, results, freeze, test outputs. Make
  §4.1 idempotent: replace its own run directories, as §7.1 already does. Document the recovery steps.
- **Acceptance check:** in one session, (a) an invalid ZIP followed by the corrected ZIP proceeds,
  (b) repeating §1.1 succeeds, (c) sample followed by BYOD succeeds and invalidates earlier results, and
  (d) running §4.1 twice gives the same table, with 5 registry entries. None of these may raise
  `FileExistsError`. Spec: DAT19, UX10, SRC2.

### FRC1-M3 (Major): the frozen test stage is not bound to the frozen configuration, and the freeze can be rewritten after test labels are seen
- **Cells:** §8.1 (cell 32) builds the test config from the live `CONDITION_SPECS[key]["configuration"]`,
  `DEVICE_PREFERENCE` and so on. The runner's `run_test` (cell 22) uses `model_configuration` for TabDPT
  `n_ensembles`, `context_size` and `batch_size`. §8.0 (cell 31) overwrites `freeze.json` on every run, with
  no guard against a test result that already exists.
- **Issue:** the hash check covers the stored development `run_config.json` but not the configuration the
  test actually consumes. Separately, a learner can view the test results, change settings, re-freeze and
  re-test with no trace.
- **Consequence:** the promise to "freeze the selected development runs before evaluating the test
  partition" is not enforced. The test score can come from a configuration that was never validated.
- **Evidence:** source inspection plus a reduced reproduction (P06): the frozen file says `n_ensembles=4`
  while the live spec feeds 8 to the test. P06 also found no refreeze guard. No numeric impact was measured.
- **Correction:** derive the test request from the verified frozen `run_config.json` and allow-list only
  phase and paths. Refuse to re-freeze once a test result exists, or require an explicit, recorded
  `REFREEZE_REASON`.
- **Acceptance check:** freeze a TabDPT run, edit `TABDPT_N_ENSEMBLES` and rerun §5.1 and then §8.1. The test
  either uses 4 or refuses. Rerunning §8.0 after §8.1 refuses or records a second freeze event in the export.
  Spec: EVAL8, EVAL14, OUT7–OUT8.

### FRC1-M4 (Major): the README offers v1 with a Colab badge but does not say which notebook to use, or that v1 lacks v2's fixes
- **Location:** `tutorials/README.md`, the registry table and the "Guided edition" paragraph.
- **Issue:** both notebooks have Colab badges. v2 is described as running "the same experiment with the same
  defaults" for "readers who are new to machine learning". There is no statement of which notebook a workshop
  participant should open, and none saying that v2 2.1.1 carries fixes for NR-01, NR-03 and NR-04, which are
  still open in v1 (FRC1-M1–M3). The README prose also says of v2 that "All code, defaults, execution counts,
  and saved outputs are retained from the recorded run". The registry row says those outputs were cleared, and
  v2 at this SHA has 0 code cells with outputs (P12). The v1 legacy evidence is not mentioned at all.
- **Consequence:** a learner or facilitator can pick v1 by its badge and hit defects that were fixed in the
  sibling notebook, with no way to know.
- **Evidence:** source inspection (P12): `explicit_recommendation_which_to_use=false`,
  `prose_claims_v2_saved_outputs_retained=true`, `v2_code_cells_with_outputs_at_sha=0`.
- **Correction:** add a one-line recommendation that says which notebook to use, for whom and why. Either
  backport the fixes to v1, or mark v1 as superseded or reference-only and remove or demote its badge. Fix the
  v2 saved-outputs sentence.
- **Acceptance check:** the README names one default notebook for participants and gives the reason. The v1
  row states its known open defects or its superseded status. No README sentence contradicts the notebook
  state at the same SHA. Spec: SRC10 (badges point to canonical, current resources).

### FRC1-M5 (Major): three of the eight learning objectives have no activity or executed evidence
- **Location:** cell 2 objectives against cells 6–41.
- **Issue:** "distinguish classification from regression" has no regression run, comparison or prompt.
  "Explain why the class bands must be constructed without validation/test leakage" covers band construction,
  which happens in the external builder; it gets only prose (cell 3) and no learner task. "Identify why
  probability columns must be aligned" gets only a glossary row: `align_probabilities` runs silently, and no
  prompt shows what misalignment would do. "Distinguish in-context conditioning from gradient fine-tuning" can
  be exercised only through GPU-only optional toggles, with no comparison prompt. The remaining objectives are
  exercised by the cell 25 checkpoint questions, the conclusion template (cell 37) and the ablation, but none
  of these has worked guidance for self-paced use.
- **Consequence:** the notebook promises outcomes it never asks the learner to practise. UX1 requires
  objectives that correspond to code actually executed.
- **Evidence:** source inspection: the objective-to-activity trace is in §4.
- **Correction:** either remove or reword these objectives, or add a bounded activity for each. For example:
  a cell that scores one model with deliberately misordered columns and shows the metric change; a
  predict→change→run prompt using the ICL vs fine-tuned toggle on GPU; a short prompt on how the band edges
  would leak if computed on all rows. Add collapsible sample guidance to the cell 25 checkpoint.
- **Acceptance check:** every objective in cell 2 maps to at least one cell where the learner predicts,
  changes, interprets or diagnoses, and that cell produces observable output. Spec: UX1 (MUST), UX6, GDL5,
  GDL9, GDL10.

### FRC1-m1 (Minor): test class shift and test per-class behaviour are never shown, even after the freeze
- **Cells:** 12 (test distribution hidden, which is correct before the freeze), 27 (confusion matrix on
  validation only), 37 §4 ("state which class had the weakest recall" after the test).
- **Issue:** the test labels are 40.3% high / 33.8% mid / 25.9% low, against 31.5 / 33.4 / 35.1% in
  validation (P07, reviewer-only). In the saved run every model loses about 0.02–0.05 balanced accuracy (foundation models 0.043–0.054) from
  validation to test. The learner has no test class balance and no test confusion matrix, so cannot explain the
  drop or answer conclusion step 4 from test evidence.
- **Correction:** after §8.1, show the test class balance and a per-class recall / confusion view for the
  frozen models.
- **Acceptance check:** the notebook displays test class proportions and per-class test recall only in cells
  after `freeze.json` exists, and the conclusion step names which partition it refers to.

### FRC1-m2 (Minor): the lag-7 heuristic's log loss is an artefact of its 1e-6 epsilon and is not explained
- **Cell:** 16 (`lag_proba` filled with 1e-6) and checkpoint Q4 (cell 25) on overconfidence.
- **Evidence:** direct execution (P08). Validation log loss is 6.88 at ε=1e-6, 3.44 at 1e-3 and 1.55 at 0.05.
  The table value reflects an arbitrary constant, not the heuristic's ranking quality.
- **Correction:** state that a hard rule has no real probabilities, so its log loss reflects the chosen ε.
  Exclude it from log-loss comparisons, or use a calibrated ε.
- **Acceptance check:** the markdown next to the baseline table explains the heuristic's log loss, or the
  table marks it "n/a (hard rule)".

### FRC1-m3 (Minor; spec ENV2 MUST): host-side libraries are unpinned
- **Cell:** 6. `lightgbm==4.6.0` is installed only if lightgbm is missing. numpy, pandas, scikit-learn and
  matplotlib are whatever the runtime has. These libraries compute every baseline and every metric.
- **Evidence:** source inspection (P11). The impact appears small: the Colab run (numpy 2.1.3, sklearn 1.6.1)
  and local CPU (numpy 2.5.2, sklearn 1.9.0) agree to 4 dp on the validation balanced accuracy of all 5
  baselines (P04).
- **Correction and acceptance check:** check the host versions against a documented tested range, warning
  without forcing a restart, and record the result in the export manifest. This is the same fix v2 2.1.1
  applied for NR-05.

### FRC1-m4 (Minor; spec DAT12/VAL6 MUST): the BYOD contract omits the class-label rule, column order and model ceilings
- **Cells:** 0, 4 and 8–10.
- **Issue:** the prose asks for "the same 17 feature columns and `target`". The code also requires the
  target labels to be exactly `low`, `mid`, `high`, the columns to be in exact order, and all features to be
  finite. The model operational ceilings (for example Mitra's 10,000 training rows) are not stated before
  model execution. The rejections themselves are clear (P09): `Expected semantic classes ['low','mid','high'],
  observed ['A','B','C']`, and an explicit expected/observed schema listing.
- **Acceptance check:** cell 4 states the label set, the column-order rule, the missing-value rule and the
  per-model row/feature ceilings. An over-ceiling BYOD training split is rejected before any foundation-model
  run.

### FRC1-m5 (Minor): the export bundle cannot reconstruct its own metrics
- **Cell:** 40. The bundle has aggregate CSVs, `freeze.json`, figures, a manifest and an 8-row inference
  preview. It has no per-row validation or test predictions, no effective run configurations, and no notebook
  commit or blob identity (P10). `freeze.json` points to runtime-local paths that are not included.
- **Acceptance check:** the paired bootstrap and the metric tables can be recomputed from the exported ZIP
  alone, and the manifest records the notebook identity. Spec: OUT2, OUT3, OUT7 (SHOULD).

### FRC1-m6 (Minor): the ablation result after the freeze has no paired contrast against its own full-feature model, and the opening misstates what is frozen
- **Cells:** 29 (validation-only `delta`), 31 (default `INCLUDE_ABLATIONS_IN_FREEZE=True`), 34 (gains only
  versus the Random Forest reference), 0 ("freezes the successful full-feature runs").
- **Issue:** after the freeze, the two LightGBM ablations appear in the bootstrap table only with
  `gain_vs_reference` against Random Forest. A learner answering the checklist item "stockout ablation
  result" can read that interval as ablation-versus-full-LightGBM evidence, which it is not. Cell 0 also says
  only full-feature runs are frozen, but by default the ablations are frozen too.
- **Acceptance check:** any interval quoted for an ablation is computed between the ablated and the full
  LightGBM, and the comparator is named in the table. Cell 0 matches the default freeze set.

### FRC1-m7 (Minor; spec REL1/REL10/REL12 MUST): the execution evidence is not bound to a revision
- **Location:** `docs/release-verification.md:245`, `tutorials/README.md` (v1 row "unverified for Revision
  2.1.0"), and notebook metadata `clean_runtime_evidence: pending`.
- **Issue:** a complete T4 run of byte-identical source exists (P01), but no commit, date, runtime or
  executor is recorded for it, and the README does not cite it. The BYOD branch has never been verified
  (REL12).
- **Acceptance check:** `release-verification.md` holds a record for this notebook blob or its successor
  naming the commit, runtime type, date and outcome (a new Colab or Kaggle run, or a maintainer attestation for
  the legacy outputs). A representative BYOD run plus one rejection is recorded.

### Suggestions
- **FRC1-S1:** add the guided layer that v2 already has: expected-result notes (GDL8), troubleshooting
  (GDL13), **Infrastructure** labels on cells 20–23 (GDL11), an Input→Model→Output line (GDL4) and a transfer
  prompt (UX9). Alternatively, retire v1 in favour of v2 (see M4).
- **FRC1-S2:** GDL15: the title "DIMER Workshop: …" names the notebook as a workshop. Prefer "notebook".
- **FRC1-S3:** cell 28 describes the test period (June, unseen month) before the freeze. The claim is
  correct (P07), but the notebook should say that knowing the test covariates' timing in advance is allowed
  while test labels stay withheld.
- **FRC1-S4:** document that the ensemble sizes differ (TabDPT 4, TabPFN 4, TabICL 8), and that optional
  TabICL fine-tuning selects checkpoints by `accuracy` while the primary metric is balanced accuracy.

## 4. Promise and objective trace

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Acquire and verify pinned artifact | cell 8, sha256 gate | provenance table | — | Delivered (legacy) |
| Development-only EDA | cells 12, 14 | train/val balance, lag_7 histogram, stockout table | cell 13 questions | Delivered |
| Baseline ladder | cell 16 | 5-row table | — | Delivered; reproduced on CPU |
| Four foundation models, one protocol | cells 18–24 | 4/4 succeeded, aligned metrics | cell 25 checkpoint | Delivered (legacy) |
| Confusion and per-class recall | cell 27 | validation only | conclusion step 4 | Partial (m1) |
| Stockout ablation | cell 29 | validation delta | checklist | Delivered for LightGBM; fine-tuned TabICL ablation broken (M1); no paired test contrast (m6) |
| Freeze before test | cells 31–32 | `freeze.json`, hash checks | — | Partial (M3) |
| Bootstrap precision | cell 34 | CIs, paired gains vs RF | cell 33 reading guide | Delivered |
| Export reproducibility bundle | cell 40 | ZIP and digest | — | Partial (m5) |
| Obj: classification vs regression | none | none | none | **Not exercised** (M5) |
| Obj: band construction without leakage | external builder, prose | none | none | **Not exercised** (M5) |
| Obj: interpret metrics | cells 16, 24, 27, 34 | tables | cells 25, 33 | Exercised; no worked guidance |
| Obj: ICL vs fine-tuning | optional GPU toggles | — | conclusion step 5 | Weak (M5) |
| Obj: foundation models vs baselines | cells 24, 34 | tables, CIs | cells 25, 37 | Exercised |
| Obj: class-order alignment | `align_probabilities` (silent) | none | none | **Not exercised** (M5) |
| Obj: validation for selection, test once | cells 31–32 | freeze | cell 30 | Exercised; enforcement weak (M3) |
| Obj: scoped conclusions | cell 37 template, cell 25 caution | — | conclusion | Exercised |

## 5. Readiness

**Needs revision.**

**Gates before teaching v1:**
1. Resolve M1–M5, or retire v1 per M4 (backport, or mark it superseded and point learners to v2).
2. Record exact-revision clean-runtime evidence (REL1/REL10) and a BYOD run plus one rejection (REL12).
3. Resolve the remaining MUST-level spec gaps: ENV2 (m3) and DAT12/VAL6 (m4).

**Verified versus inferred.**
- **Verified by direct execution (local CPU):** the retry and rerun failures and the registry wipe (M2); the
  baseline reproduction; the BYOD rejection messages; the log-loss sensitivity to ε; the month and
  class-balance facts.
- **Verified by source inspection:** M1 and M3 (reduced reproductions); the carried-adapter fidelity; the
  README contradictions.
- **Documented, but not bound to a revision:** the default T4 run.
- **Not verified:** any GPU path, foundation-model behaviour at this revision, CPU Run all time, and learner
  understanding.

**The finding most likely to be wrong:** FRC1-M5's Major severity. Facilitators might cover the unexercised
objectives verbally, and the 2.2 spec relaxes guided-layer expectations for 2.1 notebooks. Even so, UX1 is a
MUST and the notebook text makes the promises. Second candidate: M4 as Major rather than Minor, since the
README does label v1 "unverified" and Candidate.

Probe ZIP: `DIMER_FreshRetailNet_MultiModel_Classification_Workshop_Review_Probes.zip`. It contains
`run_probes.py` (run as `python run_probes.py <export_root> <projects_root>`), `results.json` and
`source_manifest.json`.
