# Release checklist — Mitra tabular-classification pipeline (issue #1)

Production-enablement gate for the validator + finetuner pair. Repo-side contract items are
satisfied on `main` (evidence below); the final gate is an on-platform DIMER verification the
platform team owns. See [DEPLOYMENT.md](DEPLOYMENT.md) and [MODEL_CARD.md](MODEL_CARD.md).

## Contract and spec (repo-side — complete)

- [x] `tabular_classification` semantics documented, independent of DIMER's `Custom / Other → object_detection` fallback; safe runtime override `DIMER_TASK_TYPE=tabular_classification` baked into the finetuner image — README / DEPLOYMENT / `TABULAR_CLASSIFICATION_DATASET_SPEC.md`.
- [x] `dimer-pipeline.json` is the authoritative parameter schema; every `datasetPreprocessing`/`modelFinetuning` field is consumed; `model_id` is **not** declared as a hyperparameter (verified: absent from the manifest).
- [x] Base-model handoff specified — DEPLOYMENT "Weights delivery"; requested id, resolved revision, and loaded artifact recorded in `provenance`.
- [x] Validator/finetuner result and provenance schemas documented — MODEL_CARD; `weightsSha256`/`configSha256`, `baseModelRevision`, `problemType`, `numClasses`.
- [x] Classification smoke matrix documented — binary/multiclass, imbalance, nullable targets, mixed features, explicit/absent `val.csv` (stratified auto-split), optional `test.csv`, duplicate-split rejection.
- [x] Cross-repo synchronization automated — `scripts/check_shared.py` enforces the shared dataset-resolution block byte-identical + against a cross-repo pinned SHA in every repo's CI.

## Real-stack verification (complete this session)

- [x] Build → offline Mitra load → fine-tune → save → reload → predict, exercised by `.github/workflows/integration.yml` (manual/nightly GPU) and run live on the 5070 Ti 2026-08-19 (`problemType=multiclass`).

## Release gate (open — platform-owned)

- [ ] Validator smoke run passes **inside DIMER**.
- [ ] Finetuner smoke run passes **inside DIMER**; base model recorded matches the model loaded.
- [ ] Saved `TabularPredictor` served by DIMER's inference-serving layer (tabular, not vision).
- [ ] All manifest controls confirmed to affect runtime on-platform.

**This is the only residual and it cannot be closed from these repos** — it needs the DIMER
portal. Tracked as a separate platform follow-up so closing #1 (repo contract) does not bury it.

`Closes #1` — repo-side contract complete; platform serving gate carried forward separately.
