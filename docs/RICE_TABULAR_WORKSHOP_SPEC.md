# DIMER Rice Pixel Tables Workshop — Specification (draft)

Prepared: 2026-10-11 by the workshop-smoke Builder, from a hosted feasibility smoke (Colab CLI 0.7.4, two fresh Tesla T4
VMs, `workshop-smoke/run1`, `run2`; 1.99 compute units). Status: **draft for owner review**. Nothing is built, committed or pushed. Smoke
numbers below are feasibility measurements, not release evidence and not results to quote.

## 1. Learning artifact and scope

**Title (working):** Rice Pixels as a Table: Do Tabular Foundation Models Need Better Features or Better Models?

**Driving question:** On the same Philippine rice pixels, how much does the *feature set* (Sentinel-2 summary
statistics vs AlphaEarth embeddings) change rice-mapping quality compared with the *model* (LightGBM vs four in-context
tabular foundation models), and what can 8 validation chips actually establish?

- Artifact: `tutorials/DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb` (kind `WORKSHOP`, KND2 filename pattern).
- Host: `kurtvalcorza/mitra-classifier-pipeline` (owner decision, final). Reuses the v2 workshop
  (`DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb`, origin/main `7fc6d240`) carriage verbatim:
  carried adapters (cell 26), `CARRIED_MODEL_MANIFESTS` with pinned revisions and digests, the four hash locks
  `tutorials/requirements-workshop-{mitra,tabdpt,tabicl,tabpfn}.lock.txt`, uv 0.12.15 by SHA-256, managed CPython
  3.12.12, `--require-hashes --only-binary :all: --no-deps`, `uv pip check`. Preserve the existing workshops.
- Normative basis: DIMER Notebook Specification **2.4** (ml-worker origin/main); profile **`E2E`**, mode
  **`WORKSHOP`**, kind `WORKSHOP`, standalone. `E2E` holds because the workshop adapts (in-context fit) five systems,
  selects thresholds on development data, freezes, scores once, and exports a reloadable artifact (§10).
- Placement: after course lessons L3 (Sentinel-2 LightGBM) and L3B (AlphaEarth LightGBM), and beside the Prithvi rice
  capstone. Links to them instead of re-teaching feature engineering.
- Defaults: public pinned data, credential-free, no uploads, no restart, fresh Colab T4. Nothing installed into the
  kernel (Colab is Python 3.13; the smoke confirmed the kernel's numpy 2.1.3 / lightgbm 4.6.0 / h5py 3.14.0 /
  kagglehub 1.0.2 suffice for data and baselines).

## 2. Learning outcomes

Learners will turn image chips into a pixel table without leaking chips across roles; explain what a 70-feature
spectral-temporal summary and a 64-value AlphaEarth embedding each encode; explain in-context learning (fit = register
support rows, no gradient step) and its context-size limits; compare five systems under one frozen protocol; read
pooled precision, recall, F1 and IoU on a minority class; and separate a feature effect from a model effect.

Shared sequence: orient -> predict -> build tables -> baseline -> foundation models -> notice -> compare (2 x 5 grid)
-> change one thing -> freeze -> score validation once -> conclude.

## 3. Data and split (verified by the smoke)

Dataset: Kaggle `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, **versions/1**, anonymous `kagglehub`,
CC BY 4.0. `INDEX.jsonl` SHA-256 `e92bbb9f…8344d` checked; all 40 chips SHA-256-verified against INDEX in both runs.
Colab served the expanded layout (`data-train/`, `data-validation/`); download 32 s.

- 40 chip-years: 32 train, 8 validation. The `development-example` (`ph_96_766`) is a training sample, never read.
- Split: the rice capstone's `split_fit_dev` (whole regions, salted SHA-256 order, ~25% development, >= 2 rice dev
  chips): 24 fit / 8 development chips. Validation = the 8 validation chips. Same rule as the capstone, so the two
  notebooks score identical chips (state this).
- Labels `label` (0/1, 255 ignore); never `label_raw`.
- Measured sizes: development 119,885 labelled pixels; validation **113,821** labelled pixels (12,889 rice).
- Feature sets (the workshop's experimental axis):
  - **S2-M1** (70): the capstone/L3 `m1_features` recipe — per band mean/std/min/max/p10/p50/p90 of observed values,
    quarterly band means, quarterly observed fractions; missing values filled with the fit-sample column mean.
  - **AlphaEarth** (65): 64 embedding values (NaN where `AE_valid` = 0) + `AE_valid`, fit-only mean fill (the L3/L3B
    bundle's `ae_features`). Validation `AE_valid` fraction recorded in the smoke summary.
- Fit table: class-stratified pixel sample of the 24 fit chips with the capstone's `pixel_sample`
  (round-robin over (chip, class) strata, seed 20261003). Sizes in §5.

## 4. Systems

| ID | System | Weights | Revision (carried manifest) | Licence |
|---|---|---|---|---|
| A | All-non-rice | — | — | — |
| B | LightGBM, capstone baseline-B settings (31 leaves, lr 0.05, 200-round cap, patience 50 on dev F1) | — | lightgbm in kernel | MIT |
| C1 | TabPFN-3 classifier, in-context | `Prior-Labs/tabpfn_3` | `24a16a89d245878b846555110985634aa2e656d7` | **TabPFN-3 licence v1.0 — non-commercial weights** |
| C2 | TabICLv2 classifier, in-context | `jingang/TabICL` | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | BSD-3-Clause |
| C3 | Mitra classifier (AutoGluon 1.5.0), in-context — **see D2: not feasible for full scoring on T4** | `autogluon/mitra-classifier` | `c425e9fa0910a6be1c494321792e7ba2a1367b1a` | Apache-2.0 |
| C4 | TabDPT v1.2 classifier, in-context | `Layer6/TabDPT` | `4462ffbd1d8dea25d4862d30beed4b70cd596ae5` | Apache-2.0 |
| M | MOMENT (optional, off by default) | — | — | not part of the smoke (D-item) |

Each FM × each feature set = 8 FM conditions + 2 LightGBM + 2 trivial = the 2 × 5 comparison grid.

**TabPFN-3 licence statement (must appear in the opening and in §References):** the TabPFN-3 weights are released
under the TabPFN-3 licence v1.0, which does not permit commercial use. The workshop uses them for teaching and research
only; learners must not deploy the exported TabPFN-3 conditions commercially. Other weights carry their own licences
(table above). Data: CC BY 4.0 with the four-component attribution block used by the capstone.

## 5. Sampling and context sizes (justified by the smoke)

- **FM fit rows: 10,000 per feature set** (3,513 rice). Mitra's adapter ceiling (`MAX_TRAIN_ROWS = 10_000`) is the
  binding limit; the same 10,000 rows go to all four FMs so the comparison is like-for-like. TabPFN-3 (adapter cap
  1,000,000) and TabICLv2 (50,000) could take more; not needed for feasibility.
- TabPFN-3 n_estimators 4; TabICLv2 n_estimators 8; TabDPT n_ensembles 4, context 4,096 (adapter's
  `subsample-balanced` reduction), batch 4,096 — v2 defaults except TabDPT context (v2 used its full 4,180-row train).
- **LightGBM: 65,536 rows** (capstone `PIXEL_CAP`) as the canonical baseline, plus the 10,000-row table as a
  same-data control (smoke: S2-M1 drops 0.758 -> 0.731, AE 0.924 -> 0.920 at 0.5), so learners see that sample size
  matters for B.
- **Mitra:** see §11 R1 — AutoGluon's memory pre-check refuses Mitra on a standard Colab T4 at any useful context
  size unless `max_memory_usage_ratio` is raised. With the ratio raised to 6 (run 2), Mitra fits 10,000 x 65 AlphaEarth rows (real peak RSS 2.66 GB, peak GPU 11.92 GB) but S2-M1 at 10,000 x 70 was refused again (estimate 49.3 GB vs 6 x 8.8 GB free in the same process) and ran at the 4,096-row fallback (peak RSS 3.13 GB, GPU 12.98 GB). Mitra prediction runs at ~48–52 rows/s at both context sizes, so context size is not the lever for Mitra on T4 (§11 R1).
- Prediction: 8,192-row chunks; all labelled development and validation pixels are scored (no pixel hidden).

## 6. Protocol (frozen before validation)

1. Roles fixed by `split_fit_dev`; fit sample drawn from fit chips only; fills fitted on the fit sample only.
2. Every choice — each system's threshold (capstone sweep 0.10–0.90 step 0.05, highest pooled dev F1, ties nearest
   0.5), LightGBM's early-stopping round, and the activity's choice — is made on development pixels and written to
   `selection.json`.
3. Freeze (hash `selection.json` and each condition's identity), then score all conditions on validation **once**.
   Keep the result whichever system wins.
4. Runtime option (recommended, D-item D4): score development on a fixed, seeded, class-stratified 30,000-pixel
   subset for threshold selection, instead of all 119,885, to roughly halve FM prediction time (§9). Validation is
   always scored in full.

## 7. Metrics and evidence boundaries

- Pooled over `label != 255`: rice F1 (primary), IoU, precision, recall; undefined ratios are null with a reason.
  Report each at 0.5 and at the dev-selected threshold.
- Per-chip table for diagnosis only. Pixels are spatially autocorrelated and chips cluster in two regions (PH-07/PH-08
  for validation): no pixel-level significance tests, no narrow confidence intervals, no national claim. Two
  validation chips have no rice (per-chip recall undefined).
- The comparison is tutorial evidence (KND5, DAT8), not a model ranking. Smoke sanity values (val F1 at 0.5, not to
  quote): AE — LightGBM 0.924, TabPFN-3 0.932, TabDPT 0.931, TabICLv2 0.911; S2-M1 — LightGBM 0.758, TabICLv2 0.765,
  TabPFN-3 0.757, TabDPT 0.755. The feature-set gap (~0.16 F1) dwarfs the model spread (~0.02) on these chips; that is
  the teaching point the notebook must let learners discover, not assert in advance.
- AlphaEarth pretraining may include these locations and years; state it. Labels are published map products.

## 8. Learner activity (one controlled change, on development only)

**Context size for one foundation model:** TabPFN-3 (fastest and lightest in the smoke) on S2-M1 with 1,000 /
3,000 / 10,000 fit rows, everything else fixed; learners predict first, then read dev F1 and seconds. Asks whether an
in-context model's weakness on S2-M1 is a data-volume issue or a feature issue (compare against the AE row). Optional
extension: same on AlphaEarth. Cost estimate from the smoke: fit < 7 s, dev scoring ~120 s per setting (or ~30 s with
the D4 dev subset).

## 9. Runtime budget (measured, fresh Colab T4, Tesla T4 15 GB, 12.7 GB RAM, 207 GB free disk)

| Stage | Measured |
|---|---|
| Dataset download (kagglehub, 344 MB) | 32 s |
| Verify 40 chips + build both feature tables | 338 s (M1 `nanpercentile` dominates; ~8.5 s/chip) |
| LightGBM, 4 fits + predictions | ~37 s total |
| TabPFN-3 env build / size | 57.6 s / 5.05 GB |
| TabPFN-3 per feature set: load + fit + dev (119,885) + val (113,821) | AE 4 + 7 + 118 + 116 s; S2 1 + 1 + 131 + 123 s; peak GPU 0.96 GB |
| TabICLv2 env / size | 48.1 s / 6.16 GB |
| TabICLv2 per feature set | AE 7 + 2 + 168 + 164 s; S2 0 + 2 + 184 + 172 s; peak GPU 9.75 GB |
| TabDPT env / size | 69.9 s / 5.94 GB |
| TabDPT per feature set | AE 5 + 3 + 125 + 125 s; S2 1 + 2 + 135 + 127 s; peak GPU 0.49 GB |
| Mitra env / size | 86.5 s / 7.56 GB |
| Mitra per feature set | AE (10,000 rows): load 13 s, fit 134 s, then ~52 rows/s — dev projected 2,296 s, val projected 2,181 s (both stopped by budget after 16,384 rows). S2-M1 (4,096 rows): load 4 s, fit 135 s, ~48 rows/s — dev projected 2,506 s. **Not feasible to score all pixels on T4.** |

Run 1 notebook wall time 2,454 s (41 min) for data + LightGBM + three FMs + one failed Mitra attempt.
**Budget derived from the measurements** (full development + validation scoring): data 6–7 min (download 32 s + 338–434 s featurisation, two fresh VMs) + LightGBM < 1 min + TabPFN-3 9.3 min + TabICLv2 12.5 min + TabDPT 9.9 min = **~39 min for A, B, C1, C2, C4 on both feature sets** (run 1 measured 41 min including a ~2 min failed Mitra attempt). With D4 (30,000-pixel dev subset) the three FMs save ~11 min -> **~28 min**. Mitra with full scoring adds ~2.6 h on T4 (≈ 2 x (2,300 + 2,200) s) — infeasible. Mitra on a 10,000-pixel dev subset and a seeded 10,000-pixel validation subsample would add ~20 min (env 100 s + per feature set ~135 + 200 + 200 s), giving ~48 min with D4 — feasible but no longer "all validation pixels" for Mitra, and over budget once explanations and learner pauses are added.

## 10. Artifacts and reproducibility

`results.zip`: `run_summary.json`, `data_manifest.json` (INDEX digest, chip digests, roles), `features.json` (recipes,
fills), `selection.json` (frozen thresholds and condition identities), `metrics.csv` (2 x 5 grid), `per_chip_metrics.csv`,
`activity.csv`, saved condition artifacts as v2 saves them (TabPFN `save_artifact`, TabDPT artifact bundle, AutoGluon
predictor), except TabICL: its support rows as `.npy` arrays (`allow_pickle=False`) plus a JSON sidecar with their
SHA-256, never a pickle (ARTIFACT_SPEC 2.0 SER7), `checksums.json`, attribution text. Fresh-process reload of one condition (v2 pattern)
must reproduce validation probabilities within a declared tolerance. No base weights or source imagery in the bundle.

## 11. Risks and findings from the smoke

- **R1 Mitra on standard Colab T4.** AutoGluon 1.5.0 skipped Mitra in run 1 before loading it: "Estimated to require
  45.571 GB out of 10.431 GB available". Cause: `autogluon/tabular/models/mitra/mitra_model.py:313-398` estimates
  0.9 x max(cpu_icl, cpu_ft_icl, gpu_cpu, gpu_gpu), quadratic in rows; ~13.9 GB already at 2,000 x 65. v2's
  `max_memory_usage_ratio=1.10` therefore cannot run Mitra on this data. Run 2 measured the estimate's error: real peak RSS 2.66–3.13 GB vs 45–49 GB estimated, so raising `max_memory_usage_ratio` (AutoGluon's own suggested override) is safe for in-context Mitra on this data; it still trips when free RAM shrinks inside a long-lived process, so each Mitra condition must run in its own child process (as v2 does) with a ratio >= ~8. The binding constraint is **throughput**: AutoGluon 1.5.0's Mitra interface hard-codes `max_samples_query=1024`, `max_samples_support=8192`, `precision="bfloat16"` (`models/mitra/sklearn_interface.py:140-150`) and predicts in 1,024-row query passes; measured ~50 rows/s on T4 at both 10,000 and 4,096 context. Inferred, not measured: T4 (Turing) has no native bf16, so an L4 may be several times faster. AutoGluon also set Mitra's `predict()` threshold to 0.063 / 0.334 on the tuning rows — the workshop must use `predict_proba` with its own dev-selected threshold and say why.
- **R2 Total runtime is near the 60-minute line** with four FMs × two feature sets and full development scoring; D4
  (dev subset) and caching the feature tables are the levers. Disk is ample (~25 GB for four envs).
- **R3 Featurisation cost.** 338 s for 40 chips on Colab (vs 3.2 s/chip locally on synthetic data). Acceptable; a
  vectorised percentile would cut it but must reproduce the capstone recipe bit-for-bit (test against the capstone).
- **R4 TabICLv2 GPU memory** peaks at 9.75 GB with 8,192-row prediction chunks; keep chunks <= 8,192 on T4.
- **R5 Torch stacks differ per env** (smoke: TabPFN 2.11.0+cu130, TabICL 2.14.0+cu130, TabDPT 2.7.1+cu126, Mitra
  2.9.1+cu128) — fine in isolated envs, but the opening must state four ~5–7.5 GB environments.

## 12. BYOD disposition

Proposed (D5): BYOD = HDF5 chips in the teaching dataset's schema (`S2`, `S2_valid`, `AlphaEarth`, `AE_valid`,
`label`), validated with the capstone's chip checks plus AlphaEarth shape/mask checks; the same fit/dev/test grouping
by site and the same frozen protocol re-run on the user's chips (DAT14: an `E2E` notebook's BYOD re-adapts, it does not
only infer). Chips without AlphaEarth run the S2-M1 column only. Minimum: 6 labelled chips from >= 3 sites (capstone
rule). Alternative: BYOD out of scope for v1, declared in the opening (needs owner sign-off against NOTEBOOK_SPEC 2.4
§10.3).

## 13. Owner decisions (Kurt, 2026-10-11/12)

- **D1 Profile/mode: `E2E` / `WORKSHOP`.**
- **D2 Mitra: opt-in, off by default on T4**, with its measured cost stated (about 50 rows/s on a T4; ~36 min per full validation pass). When enabled it runs in its own child process with AutoGluon `max_memory_usage_ratio` about 8, is scored with `predict_proba` and the development-chosen threshold (never AutoGluon's own tuned `predict()` threshold), and scores a declared, seeded pixel sample, stated beside its results.
- **D3 Context size: 10,000 class-stratified fit rows for every foundation model** (like-for-like).
- **D4 Development scoring: a seeded 30,000-pixel development subset** for threshold selection; validation is always scored on all 113,821 labelled pixels.
- **D5 BYOD: schema-matched HDF5 with re-adaptation** under the same frozen protocol (DAT14).
- **D6 MOMENT: optional, off-by-default section** (decided earlier); it needs its own hash lock and a smoke before it is written.
- **D7 TabPFN-3: included**, with the non-commercial licence statement up front and in the attribution section, as in the FreshRetailNet workshops.
- **D8 Activity: TabPFN-3 context-size sweep** on development data (e.g. 1k / 3k / 10k rows), canonical validation fixed at 10k.

## Sources

- Smoke evidence: `.agent/backups/rice-notebooks-2026-10-11/workshop-smoke/` (`REPORT.md`, `RESULT.json`, run folders).
- Notebook standard: ml-worker `integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md` 2.4.
- Reused carriage: mitra-classifier-pipeline origin/main `7fc6d240` v2 workshop and locks.
- Data and recipes: prithvi-eo-feature-extraction-pipeline origin/main `f4680a5` `tools/rice_capstone/capstone.py`,
  `docs/philippines-rice-mapping-capstone-spec.md`; rice study L3/L3B bundle `preprocessing.py`.
- Dataset: https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching
- Rice labels: https://doi.org/10.5281/zenodo.15124936
