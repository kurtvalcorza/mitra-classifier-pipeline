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

### Review-fix revision (2026-10-12): blob change

The SBC review fixes ([finding table](customer-capstone.md#notebook-review-framework-v1-findings-and-fixes-2026-10-12-sbc))
change the notebook blob from `0eade880` to `0c497aceb100`. The 2026-09-28 and 2026-10-03 runs above are history
for earlier blobs (REL14). The new blob passed a maintainer browser session on a Colab T4 on 2026-10-13 (below): Run all, BYOD with a valid
file, and a malformed file refused cleanly. Artifact-inference mode has not been exercised.

### Maintainer browser execution of revision `afa94d4` (blob `0c497ace`) — 2026-10-13

| Item | Record |
| --- | --- |
| Files | One Colab browser session saved three times, archived byte for byte: [`DIMER_Small_Business_Customer_Analytics_Capstone_afa94d4_colab-browser-t4_1-run-all.ipynb`](execution-evidence/2026-10-13/DIMER_Small_Business_Customer_Analytics_Capstone_afa94d4_colab-browser-t4_1-run-all.ipynb) (SHA-256 `b48052431b5440d8525da4beefc30fe0e4c08727d5a22dd638dffbc17b57f643`), [`…_2-byod-valid.ipynb`](execution-evidence/2026-10-13/DIMER_Small_Business_Customer_Analytics_Capstone_afa94d4_colab-browser-t4_2-byod-valid.ipynb) (`f1f0ca403ccbebfc80dcf0a6e5bea493017d22989a937ebb0d14c6b1a10578c8`), [`…_3-byod-refused.ipynb`](execution-evidence/2026-10-13/DIMER_Small_Business_Customer_Analytics_Capstone_afa94d4_colab-browser-t4_3-byod-refused.ipynb) (`23685bccbd11a12e6b9b0047fa46285a02d92269aeb74d5bbe3343c1f83a6da3`) |
| Executor | Google Colab in the browser (maintainer-run), Tesla T4, `Run all`, then the BYOD cell run twice. The Colab CLI runner could not get a T4 on 2026-10-12 (HTTP 503 twice, no cell ran) |
| Source match | Same 21 cell ids and order as the PR head; the only source differences are the `# @title` lines Colab adds to collapsed cells (cells 3 and 4) and, in saves 2 and 3, the `@param` values `RUN_BYOD`, `BYOD_CSV` and `BYOD_CONFIG` |
| Run all | **PASSED**: execution counts 1–11 in order, 0 error outputs; the three computed "This run's answer" blocks printed. Every printed metric equals the 2026-10-03 run of blob `0eade880` (127 of 127 metric values compared): the review fixes changed text and BYOD handling, not the default computation |
| BYOD (valid) | **PASSED**: `customer_byod_valid.csv`, a fixed 20% customer subsample of the notebook's own pinned UCI Online Retail II data (1,206 customers, 159,409 rows; SHA-256 `603417d0…`), with the default configuration; stages prepare / fit / develop / freeze / evaluate / infer / verify / report all PASS; `results.zip` written |
| BYOD (malformed) | **REFUSED as designed** (DAT20, VAL4): the same file without `unit_price` printed one `Refused: customer_byod_MALFORMED_missing_unit_price.csv: missing required column(s): unit_price. …` line and the recovery instruction; no traceback, no error output, no results |
| Boundary | The BYOD file is a subsample of the tutorial's own data, so it exercises the BYOD path, not new-data skill. Artifact-inference mode was not exercised. Status remains **Candidate** |


`tutorials/mitra_classifier_colab.ipynb` (`E2E`, NOTEBOOK_SPEC 3.0) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the durable
release-gate record for the generated standalone notebook. NOTEBOOK_SPEC 3.0 folded the predictor-inference companion
(`tutorials/mitra_classifier_predictor_inference_colab.ipynb`, deleted) into the E2E notebook's Section 10 external-artifact
section; the companion's recorded runs below stay as history (REL14) and are evidence for that deleted notebook only. The separate FreshRetailNet multi-model workshop is an auxiliary workshop carrier: it is statically validated by `tools/validate_release_assets.py`, but it requires its own clean-runtime execution evidence before teaching and does not inherit this pair's release evidence.

## Rice pixel tables workshop — Candidate

`tutorials/DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb`, revision 1.0.0, generated by
`tools/rice_workshop/build_workshop.py` (2026-10-12). **Status: Candidate.**

The current blob is `7d960e2d` (review fixes RPT2 plus a peer-review pass, 2026-10-14). Its default path passed a
Colab CLI run on a fresh Colab T4 (commit `03071f0`, 2026-10-14, first row of the table below); its Mitra opt-in,
BYOD upload and refused-upload paths have no hosted run. The peer-review pass, over `bc91b008`:
- adds a teaching-data check that fit, development and validation share no region, no `group_id` location and no
  chip footprint (it passes on the pinned INDEX; smallest cross-role footprint gaps 28.6, 190.3 and 384.0 km);
- adds display-only tables: the matched classifier comparison (each foundation model minus B10k on the same
  pixels), Mitra's matched-sample grid, the per-chip unscored share, false positives and predicted-rice share;
- records the BYOD status (disabled, refused, failed or completed) in the completion summary;
- documents the feature lineage, Mitra's tuning rows and the per-run scope of score-once;
- rewords the activity, the shared-sensor wording, the spread label ("spread among these system recipes") and the
  transfer and uncertainty text.

The scoring, sampling and metric code is unchanged, so no default-path number is expected to change. The `03071f0`
CLI run confirms this for the default path: compared with the `373aaf5` CLI run (normalised printed lines, `difflib`),
every F1, threshold, tree count, row count, peak GPU figure, validation score and reload check is equal. The
differences are timings, session ids and uv download order; the new display-only lines and tables listed above (the
geography check line, the matched F1-minus-B10k tables, the per-chip unscored share, false positives and
predicted-rice share, the BYOD status line); the reworded spread label and completion line; and the SHA-256 of
`selection.json` and `results.zip`, which embed the serialised TabPFN-3 artifact hashes that differ from run to run
(the open observation recorded for the two `373aaf5` runs below; TabICLv2 and TabDPT artifact hashes were equal this
time).

`bc91b008` differs from `6f109ec7` in learner text (a label-lineage table; disclosure that the labels come mainly from Sentinel-1 SAR,
which AlphaEarth includes and M1 does not; the feature effect worded as a comparison against this reference, with a
Section 7 checkpoint on label-source agreement; a fixed-recipe statement; lesson numbers linked to their lessons;
expectation text matching the recorded runs; cell titles 3.2 and 8.1) and in failure handling (a scoring parity
failure writes no score receipt, so the reload probe, the export and the completion line refuse; a refused BYOD input
prints one `Refused:` message with a re-run instruction instead of raising). The default-path metrics are expected to
be unchanged: the scoring and metric code did not change (code reading and CPU tests; not a hosted run).

`6f109ec7` differs from the last hosted-run blob `1374c3b4` (commit `98e7725`) in display text and in review fixes:

- display: grid rows and per-chip columns scored on a declared sample (Mitra) are labelled as such
  (`rt.grid_name`), the B versus B10k text no longer calls their gap a pure sample-size effect, and internal decision
  and specification codes were removed from learner text and code comments;
- guards: Mitra's declared evaluation sample is drawn and hashed at the freeze, not at scoring; the freeze records a SHA-256 for every staged table file and re-checks it (and the data, feature and table
  manifests) before scoring; a reload that flips any decision at the frozen threshold now fails parity, in scoring and
  in the probe; the context-size activity runs only if TabPFN-3's development succeeded;
- provenance: `run_summary.json` lists the systems that ran (and any that failed) rather than those selected; a BYOD
  run freezes its own whole-site split rule; model probabilities outside [0, 1] beyond 1e-6 are an error, not clipped;
  environment-setup failures are exported too; per-pixel predictions are written with full float precision;
- sampling: a class present in a development source keeps at least one sampled row (sparse-rice BYOD sets); with
  abundant rice, as on the sample path, the allocation is unchanged;
- Mitra on small development sets (BYOD) keeps a non-empty tuning set with both classes. On the sample path (30,000
  development pixels) the declared samples are unchanged.

None of these changes a score, threshold or stage output that the earlier runs recorded, which is inferred from the
code and the CPU tests, not from a hosted run. `1374c3b4` differs from
the first hosted-run blob `95d0ad11` (commit `373aaf5`) only in the BYOD input path: `collect_byod` now opens `.zip`
archives found inside a folder (the Colab upload dialog hands over a folder), and the upload folder is emptied before
each attempt. The maintainer's browser runs of `1374c3b4` (first three rows below) cover Run all with Mitra on, a BYOD
upload through the dialog, and a refused upload. The feasibility smoke of 2026-10-11 (two Colab T4 runs of a scratch notebook) informed the design; it
is not evidence for this notebook.

| Date | Commit / blob | Executor | Scope | Outcome | Evidence |
|---|---|---|---|---|---|
| 2026-10-14 | `03071f0` / `7d960e2d` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-mitra-03071f0-8547`); not a browser Run all | Default path, no field edited; blob verified against GitHub before execution and source identity checked after (41/41 cells, no source difference); forms and uploads not exercised | **PASSED**: one pass, 0 errors, 15/15 code cells in order, 1,861.6 s (notebook wall 30.7 min). The teaching-data check printed "Roles share no region, no location and no chip footprint" (smallest cross-role gaps 28.6, 190.3 and 384.0 km); INDEX and 40 chips hash-verified; reload probe within tolerance; `results.zip` 31 files, checksums verified; BYOD reported disabled. Every metric equals the `373aaf5` CLI run (differences explained above) | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_03071f0_colab-cli-t4.ipynb`](execution-evidence/2026-10-14/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_03071f0_colab-cli-t4.ipynb) (SHA-256 `a6727d63fa7f22e155000ae017408f5e6655d37568f1208dfb8c01767d54871e`), [`exec.log`](execution-evidence/2026-10-14/rice_pixel_tables_workshop/exec.log), [`run_summary.json`](execution-evidence/2026-10-14/rice_pixel_tables_workshop/run_summary.json) |
| 2026-10-12 | `98e7725` / `1374c3b4` + `RUN_MITRA = True` | Google Colab in the browser, Tesla T4 (maintainer-run) | Run all with the Mitra opt-in; only that field edited | **PASSED**: 15/15 code cells, 0 errors, wall 49.1 min, 14 conditions scored. Every non-Mitra metric equals the `373aaf5` runs. Mitra (10,000 fit rows; development and validation on declared 10,000-pixel samples): development F1 0.8890 / 0.8374, validation 0.9301 / 0.7373 (AlphaEarth / Sentinel-2 M1), fit 130 s / 114 s, peak GPU 11.9 / 4.4 GiB, reload difference 0.0 | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-mitra.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-mitra.ipynb) (SHA-256 `3dfbde8047f402e3abbef0d95519b718c1deb23b36fee1a05f0857bd6937ab64`) |
| 2026-10-12 | `98e7725` / `1374c3b4` + `RUN_MITRA`, `USE_BYOD` | Same browser session, BYOD cell run after Run all | Upload dialog with `rice_byod_valid_8chips.zip` (the 8 validation chips; not independent of the sample validation) | **PASSED**: 8 chips valid, 8 labelled; split by site 4 fit / 2 development / 2 test; all six systems re-adapted, frozen and scored once on 30,938 test pixels (Mitra: declared sample); `byod_results.zip` 29 members verified. Test F1 (AlphaEarth / M1): TabDPT 0.8825 / 0.5283, TabPFN-3 0.8792 / 0.5224, TabICLv2 0.8571 / 0.4886, LightGBM 65,536 0.8560 / 0.5272, Mitra 0.8557 / 0.5098, LightGBM 10,000 0.8220 / 0.5172 | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-byod.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-byod.ipynb) (SHA-256 `1eb384add52b18dfadc8637daf719ea9c21af4e85781f4b9d4b477747ca43dd7`) |
| 2026-10-12 | `98e7725` / `1374c3b4` + `RUN_MITRA`, `USE_BYOD` | Same browser session, BYOD cell re-run | Upload dialog with `rice_byod_MALFORMED_reflectance_x10000_2019.h5` | **REFUSED AS DESIGNED**: `ChipError` naming the file and the rule ("S2 maximum 8608.5 exceeds 2.0; the file looks like reflectance x 10 000 - multiply by 0.0001 before uploading") before any model ran | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-byod-refused.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_98e7725_colab-browser-t4-byod-refused.ipynb) (SHA-256 `e8ccb73a1d191de8e40c23626862d65663613ab6f61d7deebc00b6027ddc6662`) |
| 2026-10-12 | `373aaf5` / `95d0ad11` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-mitra-373aaf5-8547`); not a browser Run all | Default path, no field edited; blob verified against GitHub before execution; dataset version 1 served from Colab's Kaggle cache and verified against the pinned INDEX digest and every chip hash | **PASSED**: one pass, 0 errors, 15/15 code cells in order, 2,002.7 s | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-cli-t4.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-cli-t4.ipynb) (SHA-256 `23799e093dad6d6350b8fcb46ce023530bf5dd1b64fe2cfa4806253e3280463c`), [`exec.log`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/exec.log), [`run_summary.json`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/run_summary.json) |
| 2026-10-12 | `373aaf5` / `95d0ad11` | Google Colab in the browser, Tesla T4 (maintainer-run) | Run all of the default path; source identical to the blob (no field edited) | **PASSED**: 15/15 code cells, 0 errors, wall 32.4 min. Compared with the CLI run above: every metric, threshold, row count, peak GPU figure and reload check is equal; timings, session paths and a `gc.collect()` count differ; the SHA-256 of the serialised TabPFN-3 and TabICLv2 fitted artifacts differ while TabDPT's are identical (open observation, probably non-deterministic serialisation, as predictions are identical) | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-browser-t4.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-browser-t4.ipynb) (SHA-256 `c2fb3f93408c7979152c020bab271b6bef31fa287bd1c7af73a96a40ef50ba23`) |
| 2026-10-12 | `373aaf5` / `95d0ad11` + `USE_BYOD = True` | Same browser session, BYOD cell re-run after Run all | Upload dialog with `rice_byod_valid_8chips.zip` (the 8 validation chips as flat `.h5` members, schema-valid) | **FAILED (notebook defect)**: `ChipError: no .h5 chips found`; the dialog's upload folder was searched for loose `.h5` files only, so the zip was never opened. Fixed in `1374c3b4` (regression test `test_collect_byod_opens_zip_archives_inside_a_folder`) | [`DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-browser-t4-byod-upload-refused.ipynb`](execution-evidence/2026-10-12/rice_pixel_tables_workshop/DIMER_Philippines_Rice_Pixel_Tables_Workshop_373aaf5_colab-browser-t4-byod-upload-refused.ipynb) (SHA-256 `1e89f6cf59aabe29aa0052f00bb59f545d487f8bc6229710054022379ddd42d8`) |
| 2026-10-12 | `373aaf5` / `95d0ad11` + `RUN_MITRA`, `USE_BYOD`, `BYOD_PATH` | Colab CLI 0.7.4, fresh Colab Tesla T4 (variant, not suite-run) | Mitra opt-in and BYOD by path | **NOT ASSESSED (harness stall)**: the run stalled in cell 6 while uv downloaded the TabDPT wheels after the Mitra environment had built and passed `uv pip check`; killed by the 6,330 s guard, no output notebook | Not archived (no executed notebook) |

Validation F1 at the development-selected threshold (all three hosted default runs, 113,821 labelled pixels, scored once):
TabDPT v1.2 0.9329 / 0.7617, TabICLv2 0.9280 / 0.7680, LightGBM 65,536 rows 0.9277 / 0.7549, LightGBM 10,000 rows
0.9049 / 0.7300, TabPFN-3 0.8958 / 0.7584 (AlphaEarth / Sentinel-2 M1); all-non-rice 0. Reload parity: maximum absolute
difference 0.0 for every foundation model and feature set. `results.zip`: 31 members, checksums verified.

Required before promotion, for the exact notebook blob. On the current blob `7d960e2d` only the default path has run
(Colab CLI, 2026-10-14); the other rice runs noted below are of earlier blobs whose code differs from it as described
above:

- a fresh Colab T4 Run all of the default path (A, B, B10k, TabPFN-3, TabICLv2, TabDPT on both feature sets,
  activity, freeze, one validation scoring, reload probe, `results.zip`), with wall time and per-model peak GPU memory
  (earlier blobs: `95d0ad11` with Mitra off; `1374c3b4` with Mitra on, a superset of the default path);
- a run with `RUN_MITRA = True` (own process per feature set, `max_memory_usage_ratio` 8, declared samples) (earlier blob
  `1374c3b4`);
- a BYOD run on real schema-matched chips (and one refused upload) through the Colab upload dialog (earlier blob `1374c3b4`;
  the chips were the sample's validation chips, so independent BYOD data remains untested);
- maintainer review of the learner text and of the reload-parity results for every foundation model.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which preserves full generator/parity checks for the two generated standalone notebooks and separately validates the explicitly allowlisted comparative workshop:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- the tutorial directory contains exactly the two generated standalone notebooks plus explicitly allowlisted auxiliary notebooks; the generated pair remains named in `tutorials/README.md` with its profile, notebook-spec version and standalone carrier, while the comparative workshop is registered separately as `E2E` / `WORKSHOP`, Notebook Spec `2.1`, with a standalone embedded-adapter carrier; it does not inherit generated-notebook parity evidence;
- the original auxiliary FreshRetailNet workshop requires no committed outputs or execution counts; the v2 executed-record exception permits committed outputs and execution counts from a recorded run (the committed v2 notebook currently has none) and rejects saved errors. For both, every code cell and its embedded isolated runner parse as Python, `metadata.dimer` records `E2E` / `WORKSHOP`, spec `2.1`, `standalone: true`, and the pinned dataset SHA / balanced-accuracy / `uv`-managed Python-3.12 / identity-bound cache / frozen-artifact verification markers are present;
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
  `TabularPredictor.load` on the fresh reload and the `rtol=1e-6, atol=1e-8` equivalence check; Section 10: the pinned
  `SAMPLE_ARTIFACT_URL` / `SAMPLE_ARTIFACT_SHA256` download (`fetch_pinned_artifact`), the own-bundle digest (`ALLOW_UNVERIFIED_ARTIFACT` gated off), `safe_extract_archive`, `validate_artifact_directory`, the
  base-model identity/digest comparison, the AutoGluon/Python compatibility checks before `TabularPredictor.load`,
  `check_data_contract` (SART10), inference-mode `validate_inputs` with a rejection probe, `predict`, a `not-measurable` `evaluation_report` — the ceiling
  prints, the four exports per notebook, the learner-facing statements (uncalibrated probabilities / argmax rule, class coverage, dropped rows counted, no
  gradient update, fine-tuning gate, reproducibility boundary, verify-before-deserialise, trust boundary, Section 10 never reading this run's export) and the gated-off form parameters (`USE_BYOD`, `RUN_FINE_TUNING`, `RUN_NEW_DATA_INFERENCE`,
  `ALLOW_UNVERIFIED_ARTIFACT`); forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://`
  dependency without a 40-hex SHA, a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside
  the install cell, direct `urllib.request` / `TabularPredictor(` / `predictor.fit(` / `hyperparameters=` / `zipfile.ZipFile(`
  / `hf_hub_download(` / `sklearn.metrics` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`, and `shutil.make_archive(` (the export is `deterministic_zip`); a `github.com/kurtvalcorza` reference anywhere
  except the exact pinned sample-artifact URL, as a whole token, in the carried module (ST1);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (license, base model), single H1, placeholder/claim hygiene and the
  `## Checkpoint and Artifact Provenance` section (the card predates the MODEL_CARD_SPEC heading structure; the structural
  checks are switched off in the validator until the fleet's card pass reaches this branch).

CI also runs `ruff`, `tools/build_notebook.py --check`, `scripts/check_shared.py`,
`scripts/check_contract.py`, `scripts/validate_colab_tutorial.py` (the repository's own spec-1.1 checks), the two
`scripts/test_*.py` classification suites (`test_tutorial_api.py` against the package, `test_colab_csv_headers.py` against the
package and the generated notebooks) and the offline unit suite (`tests/`: `test_role_helpers.py`, `test_notebook_parity.py`,
`test_external_artifact_section.py`; injected downloader, no weights, AutoGluon never imported). These are source/provenance and unit
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
6. on the same default path, Section 10 prints the pinned URL, SHA-256 and producer before downloading, accepts the asset only at
   `SAMPLE_ARTIFACT_SHA256`, passes the archive, manifest, base-model, data-contract and runtime-compatibility checks before
   `TabularPredictor.load`, refuses the missing-column probe, and writes
   `outputs/external_artifact/mitra_classifier_external_predictions.csv` (114 rows) and
   `outputs/external_artifact/mitra_classifier_external_result.json` (verdict `not-measurable`, artifact `zip_sha256` equal to the pin);
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
| 2026-10-08 | `fb4e0eb` / `54e9966c16bd` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Standalone ARTIFACT-INFERENCE default path: pinned `sample-bundle-v1` downloaded and digest-verified before extraction, 114 sample rows (no field edited) | 146.4 s | **PASSED** — one pass, no restart, 0 errors: 9/9 code cells in order; evidence in `docs/execution-evidence/2026-10-08/mitra_classifier_predictor_inference_colab/` |
| 2026-10-08 | `83b4d0c` / `2748c0a2e7f8` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Standalone E2E default sample path (no field edited; optional journeys not exercised) | 170.4 s | **PASSED** — one pass, no restart, 0 errors: 10/10 code cells in order (`exec.log`), isolated environment built in 83 s; evidence in `docs/execution-evidence/2026-10-08/mitra_classifier_colab/` |
| 2026-09-28 | `c81db75` / `5db77406b629` | Colab Tesla T4 (maintainer-supplied) | Customer analytics capstone, default Run all | 10.4 min | **PASSED**: 10/10 code cells, 8/8 stages, fresh-process verification passed; BYOD not exercised |
| 2026-10-03 | `9698515` / `0eade8804f02` | Colab CLI 0.7.4, fresh Colab Tesla T4 | Customer analytics capstone, all code cells in order | 638 s | **PASSED**: 10/10 code cells, 8/8 stages; results equal the 2026-09-28 run |
| 2026-10-13 | `afa94d4` / `0c497aceb100` | Colab Tesla T4 in the browser (maintainer-run), one session | Customer analytics capstone: Run all, then BYOD with a valid file, then a malformed file | — | **PASSED**: 11/11 code cells, 0 errors, metrics equal the 2026-10-03 run; valid BYOD 8/8 stages PASS; malformed BYOD refused with one `Refused:` line, no traceback |
| 2026-10-11 | `373aaf5` / `95d0ad114f2d` | Colab CLI 0.7.4, fresh Colab Tesla T4 | Rice pixel-tables workshop, default path | 2,002.7 s | **PASSED**: 15/15 code cells, 0 errors (see the rice workshop section) |
| 2026-10-12 | `373aaf5` / `95d0ad114f2d` | Colab Tesla T4 in the browser (maintainer-run) | Rice workshop default Run all; then BYOD zip via the upload dialog | 32.4 min | **PASSED** Run all; BYOD upload **FAILED** on a notebook defect fixed in `1374c3b4` |
| 2026-10-12 | `98e7725` / `1374c3b456c1` | Colab Tesla T4 in the browser (maintainer-run) | Rice workshop Run all with Mitra; BYOD zip via the upload dialog; refused upload | 49.1 min | **PASSED**: 15/15, BYOD scored once, malformed chip refused (see the rice workshop section) |
| 2026-10-14 | `ac0998c` / `a25bfbfb310b` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | FreshRetailNet v2 guided workshop, default path (no field edited; forms, uploads and optional journeys not exercised) | 546.4 s | **PASSED**: 23/23 code cells, 0 errors; metrics equal the 2026-10-03 run; `evaluation_status` confirmatory, `previous_freezes` `[]` (see the NR2 section) |
| 2026-10-14 | `03071f0` / `7d960e2d` | Colab CLI 0.7.4, fresh Colab Tesla T4 | Rice pixel-tables workshop, default path (forms and uploads not exercised) | 1,861.6 s | **PASSED**: 15/15 code cells, 0 errors; geography check passed; metrics equal the `373aaf5` runs (see the rice workshop section) |

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
  executed notebook `518e27b3d78199e5302cb937e3cf6157ac04b477647e6479d680c9188a7d55e0`, `run_summary.json` `a96513e2eee6d28b031187d25e28f213f576b7fd41ee9fb2bffb2f6f566631c8`, `exec.log` `75bfe90bdfc0da8dde6d463fa8fed6ab61e18bfe242264f6799fc7a242b1f1c3` (as first committed; redacted 2026-10-22, now `c545b111d225dde006dc33da829168761a24f9a61df15d11a3fbe6b1e5f8cf8c`), `mitra_classifier_result.json` `9a5280b7cb8554c43d8397fccefe9fbcc2e6daf58342692a881e67a387de0cdb`.
- **Not exercised:** browser Run all, BYOD (single CSV and pre-split), `RUN_FINE_TUNING` on GPU, new-data inference upload,
  the Wine activity. The worked answers were checked against this run (NOTEBOOK_SPEC REL13): they hold; the Section 5 answer's
  "about 0.63" majority-class accuracy is 0.6228 on the holdout here (rounding).

### Colab CLI execution of the predictor-inference notebook at `fb4e0eb` (blob `54e9966c16bd`) — 2026-10-08

- **Executor:** Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM; every code cell in order in one kernel (order
  from `exec.log`). Not a browser **Run all**; no execution counts; no upload dialog was needed or answered.
- **Notebook:** `tutorials/mitra_classifier_predictor_inference_colab.ipynb`, commit `fb4e0eb9f4d69961ffc06bc9d6a529d2cb4b6c4e`,
  blob `54e9966c16bd20fa2de6c9113009e4be36d3d3b2` (fetched byte-exact; executed code cells equal the source).
- **Default path, no field edited:** Section 4 downloaded release asset `sample-bundle-v1/mitra_classifier_predictor.zip`, matched
  its whole-archive SHA-256 `d2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01` against the pin **before**
  safe extraction, verified the manifest, provenance and base-model identity, and printed the producer (E2E blob `2748c0a2`,
  commit `83b4d0c`, the 2026-10-08 run above). The predictor (`Mitra`, binary, `benign`/`malignant`) was reconstructed from the
  bundle alone on `cuda`. Section 6 scored the 114 sample rows (the producer's independent test partition, label removed);
  the evaluation report is `not-measurable` and four `mitra_classifier_predictor_inference_*` files were written.
- **Activity (Section 8, default `worst area` × 2.0):** 3 labels flipped, all to `malignant`; mean `probability_malignant` +0.0899.
- **Outcome:** **one pass, no restart, 0 errors**; 9/9 code cells (the carried-module cell has no output by design); isolated
  environment built in 81 s; wall 146.4 s.
- **Evidence files** (`docs/execution-evidence/2026-10-08/mitra_classifier_predictor_inference_colab/`, byte-exact, `-text`):
  executed notebook `af19781f12ecc1a94e79cf79e7ac6799942b1db06aa83337f22380d6f407cf12`, `run_summary.json` `11ad039da588b4202c8a9bcbe3d74197265ef92e290ee5d61400058e29d750e2`, `exec.log` `399f4a5c4b2a06751996a15c09ff20aef56cae3be2e1c624577b5454928c3ece` (as first committed; redacted 2026-10-22, now `5b5d0ebea4f2a661415c9f65b332219f1e53c200ab7b67877c30d4040ac017e6`).
- **Not exercised:** browser Run all, your-own-bundle path (`ARTIFACT_ZIP_PATH`), BYOD rows (`NEW_DATA_PATH`), the Colab upload
  dialog, the activity at another scale. The GitHub-hosted integration smoke (`notebook-release.yml`) exercises the own-bundle path
  with a CI-produced bundle on every pull request; that is regression coverage, not release evidence.

## Current status

**Current blob `1b310f76` (2026-10-23): Candidate; the default path, the optional activity and the `RUN_FINE_TUNING` branch each executed once on a hosted T4.** The refusal-display fix (branch `fix/refusal-display`, PR #56, NOTEBOOK_SPEC 3.0
DAT19/DAT20: a refused user input is shown as one `Refused: <file name>: <rule>` line with no traceback, and the run stops at that
cell) changed the E2E notebook blob from `f426780d` to `1b310f76` (`tutorials/mitra_classifier_colab.ipynb` at `90c60e4`). Under
REL14 the `20a13b1` default-path run and the `c5bb8a6` activity and fine-tune runs are history for blob `f426780d`, not evidence for
`1b310f76`. The E2E notebook closes with the Section 10 external-artifact section; the pinned asset `sample-artifact-breast-cancer-v1`
was produced by a recorded hosted Colab T4 run of the default path at `be831ef` (SHA-256
`17cbea6fbd8ed5a97decd4b359ef2c47d1d6025d26447e7500bc65a4dde364dd`, 280,280,365 bytes). The paragraph after the run records describes the 2.2 pair.

**What is recorded for the current blob, and what promotion still needs (2026-10-23).** Three hosted runs of blob `1b310f76` at
`90c60e4` are recorded below, each Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4, exit 0,
13/13 code cells in the executor log, 0 error outputs, input unchanged, verdict PASS:

- **Default path** (the committed blob as is): 178.7 s; the fresh reload passed and Section 10 verified the pinned asset before extraction;
- **Optional activity** (Section 11, `DATA_SOURCE = 'Sample: Wine'` run copy): 451.8 s;
- **`RUN_FINE_TUNING` branch** (`RUN_FINE_TUNING = True` run copy): 393.6 s; the owner's 2026-10-22 ruling (the optional activity and
  the `RUN_FINE_TUNING` branch run once on the blob before promotion) is satisfied for `1b310f76` by this run and the activity run.

The notebook stays **Candidate** until these two are recorded for this blob:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused (now shown as one `Refused:` line);
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-stage checkboxes, own-artifact ZIP); the scripted CLI runs count for the default path, the activity and the fine-tune branch only.

Promotion is an integrator decision; the builder does not change the status.

**Default-path run of the refusal-display fix (2026-10-23, `90c60e4` / blob `1b310f76`).** Colab CLI `colab exec -f` (a scripted run,
not a browser Run all), fresh Colab Tesla T4, default path of `tutorials/mitra_classifier_colab.ipynb` on PR #56 at
`90c60e46c8dcc5417c79b9c5a02f7d2756ab91a1`, started 2026-10-23T08:56:01Z, 178.7 s wall, exit code 0, input notebook unchanged
(the run's input equals blob `1b310f76`), 13/13 code cells in the executor log, 0 errors, verdict PASS. The notebook printed
`'name': 'sklearn-breast-cancer'`, `'train_rows': 341, 'holdout_rows': 114, 'test_rows': 114`; pretrained holdout accuracy 0.9825
(balanced 0.9767, log loss 0.0497, ROC-AUC 0.9993), independent test accuracy 0.9825 (log loss 0.0649),
`'recommended_for_export': 'pretrained'`; the deterministic export printed `zip_sha256`
`c6a6f50bd95eecaa7b16c9828d315d5d6ecd7cb4d0b05055ebf6e045005fd247` and the fresh reload printed PASS. Section 10 verified the pinned
asset before extraction (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`), rebuilt the predictor from the bundle (30
features, binary, on CUDA) and scored 114 rows (verdict `not-measurable`, as designed). Evidence in
`docs/execution-evidence/2026-10-22/mitra_classifier_colab_90c60e4/` (the repository's 2026-10-22 evidence folder; the run date is
2026-10-23). The local run directory was redacted to `<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_classifier_colab_90c60e4_colab-cli-t4.ipynb` | `187a376a33d56f86ca6b6da55c75c1a1ea02e67f662c79659d37e29e1c12dc9e` | same (no local path) |
| `exec.log` | `203b06a7ad3fd5c101452fb98b94a40fcdb3c2b0c24b4397cae6c5e3e81ca832` | `6c7c17779431d7b46fd32289286e538f05161fa8603486747d05ba3ef4617f56` (2 replacements) |
| `run_summary.json` | `d05ccb9351c7996e47c4abb8be2bdf6618835442f105564c1a3f18b896d1ec59` | same (no local path) |

**Optional-activity run of the refusal-display fix (2026-10-23, `90c60e4` / committed blob `1b310f76`, `DATA_SOURCE = 'Sample: Wine'`).**
A run copy of committed blob `1b310f76` with one override, `DATA_SOURCE = 'Sample: Breast Cancer'` → `'Sample: Wine'` (run copy blob
`601391175c58`, SHA-256 `c48ace5fd60395639c9aedb28374a429858c5c00e1c28de4a97eadc98e32762f`; it differs from the committed blob only
in that assignment literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI `colab exec -f` (a scripted run, not
a browser Run all), fresh Colab Tesla T4, at `90c60e4`, started 2026-10-23T08:59:12Z, 451.8 s wall, exit code 0, input (the run copy)
unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. The notebook printed `'name': 'sklearn-wine'`,
`'train_rows': 106, 'holdout_rows': 36, 'test_rows': 36`, 13 features, three classes (`class_0`, `class_1`, `class_2`),
`'problem_type': 'multiclass'`; pretrained holdout accuracy 0.9722 (balanced 0.9762, log loss 0.1122) beside the majority-class
baseline's 0.3889, independent test accuracy 0.9722 (log loss 0.1097); the fresh reload printed PASS; Section 10 still verified the
pinned Breast Cancer asset (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`). Evidence in
`docs/execution-evidence/2026-10-22/mitra_classifier_colab_90c60e4-activity/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_classifier_colab_90c60e4_colab-cli-t4_activity.ipynb` | `386d4d513f5f658aa3fff1afc860be7709b0cf49b1df483b470bf2b6589248ce` | same (no local path) |
| `exec.log` | `daf40691dc6d2c773924b55a4c7209cc699e458f95273abdd8b5e95a280e1171` | `ef2a9846bda43338c796e336396e7050f7ce78f6f34607b655c11e271ef1bdb3` (2 replacements) |
| `run_summary.json` | `03bc0fa6c6285d2bd49b1120bbb5b8e98a79ced081f81fa7677a4304e29f19ea` | same (no local path) |
| `run_copy.ipynb` | `c48ace5fd60395639c9aedb28374a429858c5c00e1c28de4a97eadc98e32762f` | same (no local path) |

**`RUN_FINE_TUNING` run of the refusal-display fix (2026-10-23, `90c60e4` / committed blob `1b310f76`, `RUN_FINE_TUNING = True`).**
A run copy of committed blob `1b310f76` with one override, `RUN_FINE_TUNING = False` → `True` (run copy blob `f574eb547042`, SHA-256
`8c64cc5df28e71e6962853cace696207853bc49c629222069314831f84e60c4e`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `run_copy.ipynb`). A first attempt in the same run directory hung when Colab
reclaimed the VM and was killed; the directory holds only the second attempt's files, recorded here. Colab CLI `colab exec -f` (a
scripted run, not a browser Run all), fresh Colab Tesla T4, at `90c60e4`, started 2026-10-23T09:48:45Z, 393.6 s wall, exit code 0,
input (the run copy) unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. On the default Breast Cancer sample
(341 / 114 / 114 rows) the GPU fine-tune ran on cuda. Fine-tuned holdout: accuracy 0.9825, balanced accuracy 0.9767, log loss 0.0484,
macro-F1 0.9812, MCC 0.9630, ROC-AUC 0.9993, against the pretrained holdout's accuracy 0.9825 and log loss 0.0497 (equal accuracy, a
0.0013 lower log loss). The notebook printed "WARNING independent-test metrics worsened after fine-tuning: ['log_loss', 'roc_auc']
(evidence only; never used for selection)". Selection stayed on the holdout and kept the pretrained model
(`'recommended_for_export': 'pretrained', 'selection_basis': 'holdout:accuracy'`); the exported bundle and the fresh reload are of
`mitra-pretrained` (reload PASS). The fine-tuned predictor was saved, not exported: `mitra-finetuned` appears in the outputs
listing beside `mitra-pretrained`. Section 10 passed as on the default path. Evidence in
`docs/execution-evidence/2026-10-22/mitra_classifier_colab_90c60e4-finetune/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_classifier_colab_90c60e4_colab-cli-t4_finetune.ipynb` | `b066a158dee772fa368195ff8f8f00c8096c5d3cc4b08852394b94c6e954cc13` | same (no local path) |
| `exec.log` | `aca6eb73e1cf86fe53bbe7fd0b56e3f3cc392808367795abdfa64af8700e5ccd` | `707be685e7c6afa3125fb31e5ea9f2b11380797f44472d93a7e63f0affe18711` (2 replacements) |
| `run_summary.json` | `275e04a0a4b933abdca23db2376c5e8032f7611682792c6d7b0eefab3751e717` | same (no local path) |
| `run_copy.ipynb` | `8c64cc5df28e71e6962853cace696207853bc49c629222069314831f84e60c4e` | same (no local path) |

**First hosted run of the migrated notebook (2026-10-22, `20a13b1` / blob `f426780d`, Notebook Spec 3.0).** Colab CLI 0.7.4
`colab exec -f`, fresh Colab Tesla T4, default path of `tutorials/mitra_classifier_colab.ipynb` on PR #52 at
`20a13b167c072ec643f0f50d04b0a090dac96c2a`, started 2026-10-22T02:14:03Z, 189.7 s wall, exit code 0, input notebook unchanged;
executed cell sources equal blob `f426780d` (40 cells). The fresh reload of the run's own export passed (labels identical,
probabilities at `rtol=1e-6`; its deterministic export printed `zip_sha256` `a3362cc603b07f345502c2c90926f64924e81722f7c4f93caadd85d2850fcd1e`).
**Section 10 ran end to end on the default path:** it downloaded the pinned release asset, verified SHA-256
`17cbea6fbd8ed5a97decd4b359ef2c47d1d6025d26447e7500bc65a4dde364dd` before extraction, rebuilt the Mitra predictor from the bundle
(30 features, binary, on CUDA), scored 114 rows and wrote `outputs/external_artifact/mitra_classifier_external_predictions.csv`
and `..._external_result.json` (verdict `not-measurable`, as designed: the section scores rows as new input without labels).
Evidence in `docs/execution-evidence/2026-10-22/mitra_classifier_colab_20a13b1/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `run_summary.json` | `ee859aba5ec2542d847af4f905973bd7e3cbe7bf7624af4622624383e40668d5` | same (no local path) |
| `exec.log` | `67cdf9a1045f677bbd14a13fc459c5b69ec024724eed325805d27912a223e077` | `c6f23209fcf0f81765a2fd9a13910d7ccb31185971e70aba5b7ba14439f3b304` |
| `mitra_classifier_colab_20a13b1_colab-cli-t4_output.ipynb` | `9b38176d612e6be3589378ed2a8ad3b96c8930599daa72c635de593277609c6d` | same (no local path) |
| `suite.log` | `132151706ce63b46f5e6ee6d93656ce40f758f573b00ad91a91009caa769865f` | `5d415ed1c3b2e04921d5a1350e4da1538944f74cae1b7966b538f45c53ac0493` |

**Optional-activity run (2026-10-22, `c5bb8a6` / committed blob `f426780d`, `DATA_SOURCE = 'Sample: Wine'`).** The Section 11
activity is a re-run of Sections 4–7 on the Wine sample, so it was run as a run copy of committed blob `f426780d` with one override,
`DATA_SOURCE = 'Sample: Breast Cancer'` → `'Sample: Wine'` (run copy blob `63e09ec9b3ca`, SHA-256
`9080bb67f75625e0581b3803ab3748970e6fcf8ea81886a2bff3eaf04cf80482`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser
Run all), fresh Colab Tesla T4, at `main` `c5bb8a6f9f446c74ff7efd9cde46d84e2fe77f77`, started 2026-10-22T13:33:02Z, 187.0 s wall,
exit code 0, sequential, input (the run copy) unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. The notebook
printed `'name': 'sklearn-wine'`, `'train_rows': 106, 'holdout_rows': 36, 'test_rows': 36`, 13 features, three classes,
`'problem_type': 'multiclass'`; pretrained holdout accuracy 0.9722 (balanced 0.9762, log loss 0.1122) beside the majority-class
baseline's 0.3889, independent test accuracy 0.9722; the fresh reload printed PASS; Section 10 still verified the pinned
Breast Cancer asset (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`). Evidence in
`docs/execution-evidence/2026-10-22/mitra_classifier_colab_c5bb8a6-activity/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_classifier_colab_c5bb8a6_colab-cli-t4_activity.ipynb` | `ab142b86d4d6642706d0c5fe40ba2fe59d724f543a0e70aa8176996db20b0891` | same (no local path) |
| `exec.log` | `72345b4f25d46c98bdbfbc75d799fade89f40aeb5885581ae2d1f6c7b3ac274d` | `0961c380d8ef5307f96c874da18f28f3c2ce3b3d348fe99fec86e2632654db40` (2 replacements) |
| `run_summary.json` | `2e243303652a3932dc098249d8255c83664afaef5d686a1648631c9a8a6ac981` | same (no local path) |
| `run_copy.ipynb` | `9080bb67f75625e0581b3803ab3748970e6fcf8ea81886a2bff3eaf04cf80482` | same (no local path) |

**`RUN_FINE_TUNING` run (2026-10-22, `c5bb8a6` / committed blob `f426780d`, `RUN_FINE_TUNING = True`).** A run copy of committed
blob `f426780d` with one override, `RUN_FINE_TUNING = False` → `True` (run copy blob `695c5f8f9789`, SHA-256
`ae9e6af2edec0c3b3f8a6ca14187a3245dd12e2b34d24751a6674058593ba0e8`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser
Run all), fresh Colab Tesla T4, at `c5bb8a6`, started 2026-10-22T13:49:27Z, 487.4 s wall, exit code 0, sequential, input (the run
copy) unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. On the default Breast Cancer sample (341 / 114 / 114
rows) the GPU fine-tune ran on cuda: holdout log loss improved from 0.0497 (pretrained) to 0.0484 (fine-tuned) at equal holdout
accuracy 0.9825, and the notebook printed "WARNING independent-test metrics worsened after fine-tuning: ['log_loss', 'roc_auc']
(evidence only; never used for selection)". Selection stayed on the holdout (`'recommended_for_export': 'pretrained',
'selection_basis': 'holdout:accuracy'`); the fresh reload and Section 10 passed as on the default path. Evidence in
`docs/execution-evidence/2026-10-22/mitra_classifier_colab_c5bb8a6-finetune/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_classifier_colab_c5bb8a6_colab-cli-t4_finetune.ipynb` | `699de772ce5b641394e8274c08eef966712886e7906eafff3d947e344fc11747` | same (no local path) |
| `exec.log` | `fe34f6fd951a3fb964d57a0bc489cadf6899ee4a49f1d678a33c695a3a0a989f` | `603c87b8a947792ea8828a6fc9779cf42746020c8efe91d251e0d69a180a5a5d` (2 replacements) |
| `run_summary.json` | `092393fa5b3f961d2d6973ba2fcf57fdc83d770e8aeace4705f188010e8e8ce1` | same (no local path) |
| `run_copy.ipynb` | `ae9e6af2edec0c3b3f8a6ca14187a3245dd12e2b34d24751a6674058593ba0e8` | same (no local path) |

**Producing run of the pinned sample artifact (2026-10-22).** Colab CLI 0.7.4 `colab exec -f`, fresh Colab Tesla T4, default
path of `tutorials/mitra_classifier_colab.ipynb` at `be831efb226378f342d1aabac72ee56597c9403d` (blob `2748c0a2e7f8`), exit code 0,
161.2 s wall, input notebook unchanged; the run's own fresh reload verified the manifest and provenance before deserialisation and
matched labels exactly and probabilities at `rtol=1e-6`. Its exported `outputs/mitra_classifier_predictor.zip` was published as
the release asset `sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip` (SHA-256 above). It was written by that
revision's `shutil.make_archive`, so its digest is not reproducible by re-running; later exports use `deterministic_zip`.
Evidence in `docs/execution-evidence/2026-10-22/mitra_classifier_colab_be831ef/` (byte-exact, `-text`). The local run directory
was redacted to `<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `run_summary.json` | `dcb03aa2537229aae24d5ab6ba39b2014d095dcb0922202c4d96a30231075015` | same (no local path) |
| `exec.log` | `84483bcf9d8cf3a282d19abf6a60d673bee96cfc3912328c3d5d20f341c707ed` | `b5504232fcaae7402c29030b8af3cd8bdd0ed457effc0b838ce25438c27185df` |
| `mitra_classifier_colab_be831ef_colab-cli-t4_output.ipynb` | `687bb3d83ec8f476a6c74dc738ed819ba6d23d2feaa4c489c991cb46543b9c0d` | same (no local path) |
| `suite.log` | `776edc54bdb49a7e11b22b96941d1bf4c2205eedb9200a2a72123280887bdc09` | `89cf8db196c05c4bb3ea178e339a8e0d1c0ab52e2c9c28cb1b15722ee96aab3f` |

**Generated tutorial pair (2.2, history): Candidate.** The E2E notebook's then-current blob `2748c0a2e7f8` (superseded by `f426780d`; not evidence for it, REL14) had one recorded execution, the 2026-10-08 Colab CLI T4 run above (default path only); browser Run all, BYOD, the fine-tuning gate and the activity remain unexercised, and promotion is an integrator's decision. The 2026-09-14 Kaggle T4 row is evidence for the previous E2E blob (`9599d5c4832c`) only: an nbclient execution of the default sample path with a `google.colab` shim, not a browser Colab `Run all`. On 2026-10-08 both notebooks were regenerated (generator /2.2 isolated uv environment, DIMER Notebook Specification 2.2, Notebook Review Framework v1 fixes MCC-M1..M3 / MCC-m1..m6), so under NOTEBOOK_SPEC REL14 they return to Candidate until a run of the exact new blobs is recorded; the earlier row stays as history. The ARTIFACT-INFERENCE companion's default path now downloads the trusted sample bundle (release `sample-bundle-v1`, asset `mitra_classifier_predictor.zip`, SHA-256 `d2a330457fe8dbda2e7ad016e2ac0a0e8a461a738b1864ff5861654acb3cca01`, produced by the 2026-10-08 E2E run recorded above; the repository's immutable-releases setting was enabled before it was published) and checks the digest before extraction (NOTEBOOK_SPEC SART6–SART8); its then-current blob `54e9966c16bd` (deleted with the companion) passed the 2026-10-08 Colab CLI T4 run above (default path only). Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
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

## FreshRetailNet Classification v2: NR2 review fixes — 2026-10-12

A Notebook Review Framework v1 review (prefix NR2) of the guided edition at `6b10569` (blob `f00c9351`) found 1 Major,
7 Minor and 1 Suggestion. The revision label stays **2.2.0**: `WORKSHOP_REVISION` is set in Section 0.1, which must stay
byte-identical to the compact edition's. The change is logged in `metadata.dimer.review_fixes`.

- **NR2-M1 (Major):** Section 8.0 now carries the compact edition's refreeze guard. Once the frozen models have been
  scored on the test partition (`TEST_RESULTS`, or a saved `runs/test/*/result.json`), it refuses to replace the freeze
  unless `REFREEZE_REASON` is entered; a replaced freeze is kept in `previous_freezes` in `freeze.json` and the export,
  and stale test-stage results are cleared. Section 8's text says what is enforced instead of "prevents this by design".
- **NR2-m4:** reproduced on a CPU harness of the notebook's own cells: 4.1 → 7.1 → 4.1 → 8.0 raised a bare
  `FileNotFoundError` on a deleted ablation `result.json`. Section 8.0 now skips ablation runs whose folders are gone
  and says to rerun 7.1. Cells 4.1 and 8.1 must stay identical to the compact edition's, so they were not changed; a
  4.1 rerun after the freeze still stops 8.1 (with "frozen development result changed after freeze"), and
  Troubleshooting now says how to recover from both.
- **NR2-m1:** the four sample answers that quote numbers name the run they come from (revision 2.2.0, commit `05d54cd`,
  blob `b3c33de9`, Colab CLI T4, 2026-10-03). The stale "revision 2.0.1" caveat is gone, and "Lower for every model"
  now reads "every fitted model", since the training-prior majority stays at 0.333.
- **NR2-m2 (owner ruling C):** the limitations and the multi-model and test answers state that the classical baselines
  ran fixed default hyperparameters without validation tuning and the foundation models ran in context at their
  defaults. A validation-tuned, early-stopped LightGBM baseline is a follow-up; no number changes here.
- **NR2-m3:** no cell advises hiding saved outputs the committed notebook does not have; `metadata.dimer`, the tutorials
  README and this record now name the same evidence for the same blob.
- **NR2-m5:** the opening states the measured total (668.6 s on a fresh Colab T4, about 258 s of it `uv`) and the
  download (about 13 GB of packages plus 0.9 GB of checkpoints, computed from PyPI and Hugging Face listings of the
  pinned files, not measured).
- **NR2-m6:** Section 5 states that fine-tuned TabICLv2 selects its checkpoint by validation accuracy (the shared runner
  passes `eval_metric="accuracy"`), while fine-tuned Mitra uses balanced accuracy. The runner is a shared cell, so it
  was not changed.
- **NR2-m7:** References cite each model's paper and model card (arXiv identifiers checked on 2026-10-13).
- **NR2-S1:** not addressed (Suggestion).

**Round 2 (2026-10-13), after an independent peer review of the PR.** Only v2-only cells changed (8.0, 8.1b, 9.1, and a new 5.5b pair); every code cell that equals the compact edition's stayed byte-identical.

- **Refreeze bypass (reproduced, fixed):** freezing one baseline, scoring it, rerunning 8.1 with `EVALUATE_FROZEN_TEST` off (which resets `TEST_RESULTS`) and rerunning 8.0 was allowed without a reason. 8.0 now also treats as exposure the freeze that 8.1 read before scoring (kept when 8.1 is rerun disabled or fails part-way), saved foundation-model test results, and an append-only `test_exposure.json` that 8.0 and 8.1b write, bound to the freeze's SHA-256. It cannot detect test scores seen in an earlier runtime session; writing the record inside 8.1 itself would need a change to a cell shared with the compact edition.
- **Exploratory status:** a refreeze after test exposure marks the new freeze `evaluation_status: exploratory (test-informed)`, and the status never returns to confirmatory in that session. It is exported and printed; the text says a reason records the contamination and does not undo it.
- **Replaced freezes kept in full** under `freeze_history/` (exported); `previous_freezes` points at each copy.
- **The freeze names the validation-selected model, the comparator, the tie-break rule, the planned contrasts and any omitted ablation (with its reason)**; stale 7.1 summary rows of deleted ablations are dropped. 8.2 is a shared cell, so it still ranks the live baseline table; 8.1b now refuses to continue if that ranking no longer gives the frozen comparator. 8.1b shows the validation-selected model's test result as the confirmatory one, with a test confusion matrix and per-band test recall.
- **5.5b** lists each foundation condition's adaptation mode and checkpoint-selection criterion. Correction to round 1: fine-tuned Mitra does not select by balanced accuracy. The runner's balanced-accuracy choice is not passed to Mitra, whose metric setting accepts only accuracy or log loss.
- **Export:** bootstrap settings (draws, seed, method), recomputability with and without row-level files, freeze history, exposure record, adaptation table, test confusion and the optional foundation ablation.
- **Wording:** the 8.2 intervals are an illustrative IID row bootstrap, probably too narrow and not a lower bound, with no superiority claims and a multiplicity note. Metric ranges are 0–1 for balanced accuracy and −1 to 1 for MCC. Log loss suggests possible overconfidence rather than diagnosing it. The forecast is issued after day *t* closes, for sales on day *t* + 7. Pretraining is described by model, and the lag rule is called a relative-band persistence rule.

Eight new behavioural and source checks in `tests/test_workshop_v2_review_fixes.py` cover the bypass, a part-way failure, exposure persistence, exploratory status, full retention, the selection record, the comparator check, omitted ablations and the adaptation table. Default-path numbers are unchanged; the freeze and export gain fields only.

`tests/test_workshop_v2_review_fixes.py` adds four checks: two run the notebook's own 4.1, 7.1 and 8.0 cells on a
synthetic table with a logistic model standing in for LightGBM (logic checks, not model runs), and two check the text.
All four fail on `6b10569`. **Evidence boundary:** when written, no hosted run of this blob existed; the Colab CLI fresh
Run all of blob `a25bfbfb` (REL14) is now recorded below. Still open: a browser BYOD session with one valid and one
refused ZIP (REL12), the fast path, the optional fine-tuning and foundation-ablation toggles, and the §8.0 refusal and
`REFREEZE_REASON` path in a hosted session. **Status: Candidate.**

### Colab CLI execution of the NR2 fixes at `ac0998c` (blob `a25bfbfb`) — 2026-10-14

The guided edition was run at commit `ac0998ceb5cef6efcff8f2624d5c38e5c90bbca7` (notebook blob
`a25bfbfb310be605a565f02dfc91a156ca02316d`, revision label 2.2.0). The executed notebook is preserved byte for byte as
[`…Workshop_v2_ac0998c_colab-cli-t4.ipynb`](execution-evidence/2026-10-14/DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2_ac0998c_colab-cli-t4.ipynb)
(SHA-256 `62af67b9e557c9502e73f9dbcc9a600a70d9eb141cbd67387e75dc140c0714d1`).

- **Source:** downloaded from GitHub at the PR head; Git blob verified before the session. The executed copy has the
  same 58 cells in the same order and the same source (no `# @param` edited).
- **Executor:** Colab CLI 0.7.4, fresh Colab Tesla T4, through the workspace `colab-cli-serial-test-suite`
  (`colab new --gpu T4`, `colab exec -f`, `colab stop`), 17:30–17:39 UTC, session wall 546.4 s. Code cells ran in order
  in one kernel; this is CLI sequential execution, not a browser Run all, and order is evidenced by the `Executing cell
  k/N` log. Forms and uploads were not exercised.
- **Path exercised:** default controls only, as in the 2026-10-03 run. Not run: the §8.0 refusal and `REFREEZE_REASON`
  path, the fine-tuned TabICLv2 and foundation ablations, the fast path, and BYOD.
- **Outcome:** PASSED, 23/23 code cells with no error outputs (one more code cell than 2.2.0 at `05d54cd`: the new
  5.5b). The completion summary reports 9 validation models, 11 frozen test models, no foundation failures, and
  "Run-all complete". The AutoGluon memory warning seen in the earlier run recurs; it is a warning only.
- **Isolated environments:** all four uv environments built from the hash-locked files and passed `uv pip check`; uv
  reported 81 s (Mitra), 57 s (TabDPT), 77 s (TabICL) and 37 s (TabPFN) preparing packages.
- **Freeze and exposure (default single Run all):** the session was fresh, so §8.0 found no earlier `freeze.json` and no
  exposure signal: `previous_freezes` is written as `[]` and `evaluation_status` as `confirmatory`. This is read from
  the cell's code path and its output: §8.0 printed no "Replaced an earlier freeze" line and §8.0, §8.1b and §9.1
  printed no exploratory warning. The `freeze.json` file itself was not retrieved before the VM was released. §8.0
  names `tabpfn3_icl` as the validation-selected model and `random_forest` as the comparator.
- **Comparison with the 2026-10-03 run of blob `b3c33de9`:** every printed metric is equal — the data checks and EDA,
  the baselines, the validation table (TabPFN-3 0.6021 … Random Forest 0.5733), the confusion matrix, the ablations,
  the frozen-test results (TabPFN-3 0.5587 … LightGBM 0.5205), the bootstrap intervals and the §8.3 preview. The
  differences are: (1) host memory and disk figures in the AutoGluon log and its throughput estimate (77.8 → 77.9
  rows/s), runtime noise; (2) new display-only outputs from this PR — the §5.5b adaptation table (all four conditions
  `in_context`, no checkpoint selection), the §8.0 selection line, and §8.1b's confirmatory TabPFN-3 test row (balanced
  accuracy 0.5587, accuracy 0.5725, macro F1 0.5608, MCC 0.3512, log loss 0.9158; equal to the §8.1 row) with a test
  confusion matrix and per-band test recall (low 0.4638, mid 0.5564, high 0.6558); (3) the export folder and report
  ZIP digest (`1492c68e5152c9b6…`), which differ by construction in a new session and because the export now carries
  the freeze history, exposure record and adaptation table; (4) uv download order and timings.
- **Not updated:** `metadata.dimer.review_fixes[].execution_evidence` inside the notebook still says "pending"; changing
  it would change the notebook blob this record binds to.

This record satisfies REL14 for blob `a25bfbfb`. The other items listed above remain open. **Status: Candidate.**

### Serialization scan of the pinned sample artifact (2026-10-23, retroactive)

ARTIFACT_SPEC 2.0 (ml-worker#90): SER9 (a): AutoGluon has no faithful non-code-executing format for a `TabularPredictor`, so the predictor stays pickle-based (including `models/Mitra/model.pt`, which pickles the whole model object). It needs a recorded trust note (SER6, under "Artifact contract" in [`tutorials/README.md`](../tutorials/README.md)) and a scan with an acceptable outcome (SER13). SER13 asks for the scan **before** publication; this asset was published on
2026-10-22, before 2.0 existed, so this is a **retroactive** scan of the published bytes, not the pre-publication control. The
next sample artifact is to be scanned before upload and published only at the digest that passed.

The release asset `sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip` was downloaded on 2026-10-23; its SHA-256 (`17cbea6fbd8ed5a97decd4b359ef2c47d1d6025d26447e7500bc65a4dde364dd`) matches the pinned value. `scan.py`
(picklescan 1.0.5, joblib 1.6.0, numpy 2.5.4, Python 3.13) extracted it with member-path checks, reached every nested archive,
and judged each pickle file on its own: **dangerous** if any global is dangerous, **incomplete** if no recorded method read the
whole file, otherwise **clean**. Content decides what is a pickle, not the file name, and every archive (PyTorch zips
included) is unpacked so each member gets its own verdict.

| File | Kind | Verdict |
|---|---|---|
| `predictor.pkl` | pickle | clean |
| `learner.pkl` | pickle | clean |
| `models/trainer.pkl` | pickle | clean |
| `models/Mitra/model.pt` | pytorch-zip | extracted |
| `models/Mitra/model.pkl` | pickle | clean |
| `utils/data/y_val.pkl` | pickle | clean |
| `utils/data/y.pkl` | pickle | clean |
| `utils/data/X_val.pkl` | pickle | clean |
| `utils/data/X.pkl` | pickle | clean |
| `utils/attr/Mitra/y_pred_proba_val.pkl` | pickle | clean |
| `models/Mitra/model.pt!/model/data.pkl` | pickle | clean |

- **Result:** 10 pickle files; clean=10 dangerous=0 incomplete=0; `scan_status=0`. Dangerous globals: none.
- **Globals (distinct, all files):** 56, by top-level package: autogluon (27), numpy (7), pandas (7), torch (7), builtins (2), collections (2), networkx (2), __builtin__ (1), sklearn (1). The per-file lists with picklescan's
  classification are in the report.
- **Evidence:** `docs/execution-evidence/2026-10-23/ser13-sample-artifact-scan/` holds `scan-report.txt` exactly as `scan.py`
  wrote it (it records only paths inside the asset), `scan.py`, and `test_scan.py`, its regression tests (dangerous globals
  behind joblib buffers or beside joblib files, protocol-0 pickles under other names, pickle/zip polyglots, every PyTorch zip
  member, object-dtype .npy files, torch payloads embedded in complete pickles, compressed (including zlib of any
  window size) and truncated pickles, nested zip and tar archives that try to overwrite queued members, duplicate,
  escaping and malformed members).
  `scan.py` exits 0 clean, 1 dangerous, 2 incomplete, 3 could not scan.

### Local-path redaction of earlier evidence (2026-10-22)

The local run directory in these earlier committed evidence files was replaced by `<local-run-dir>/`, the
convention the later evidence in this repository already uses. Only those bytes changed: line endings and every
other byte are as committed. Git history is not rewritten, so earlier commits still hold the original files.
SHA-256 before (committed on `main`) → after:

| File | Replacements | SHA-256 before | SHA-256 after |
| :--- | ---: | :--- | :--- |
| `docs/execution-evidence/2026-10-08/mitra_classifier_colab/exec.log` | 2 | `75bfe90bdfc0da8dde6d463fa8fed6ab61e18bfe242264f6799fc7a242b1f1c3` | `c545b111d225dde006dc33da829168761a24f9a61df15d11a3fbe6b1e5f8cf8c` |
| `docs/execution-evidence/2026-10-08/mitra_classifier_predictor_inference_colab/exec.log` | 2 | `399f4a5c4b2a06751996a15c09ff20aef56cae3be2e1c624577b5454928c3ece` | `5b5d0ebea4f2a661415c9f65b332219f1e53c200ab7b67877c30d4040ac017e6` |
| `docs/execution-evidence/2026-10-12/rice_pixel_tables_workshop/exec.log` | 2 | `6d6c2c6133f1572103bb197ad5c2f019904237d54b7bfd518f8f71f2d60ecea6` | `ca6ff8cbcf5344242e61af2ae5c5c000b88db855e2a4b8b667eaf542ccda6017` |
| `docs/execution-evidence/2026-10-14/rice_pixel_tables_workshop/exec.log` | 2 | `2b4b7e33cc778ca432e1f01ebe5e9e59918df29605ce9510e3731ea65971e7c2` | `f8efb345741c8b4fa66d5c474d2f0ee4e1ecc70e9e12d9409afe127aeeaf8f30` |

Where this document quotes one of these files' earlier hash as its committed hash, the new hash is now
given beside it.

