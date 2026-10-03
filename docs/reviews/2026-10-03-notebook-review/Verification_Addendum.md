# Verification addendum: FRC1 review of the v1 FreshRetailNet classification workshop

The review in this folder was produced on 2026-10-03 by a separate review pass. Before any fix, the fixing pass
treated it as a hypothesis and re-checked it against the source on 2026-10-03.

## Revision check

| Item | Review | Re-checked |
|---|---|---|
| Reviewed commit | `0434a026cf755071e89bc33c876e67040c2a614d` | equals `origin/main` and `gh api repos/kurtvalcorza/mitra-classifier-pipeline/commits/main` at fix time |
| Notebook blob | `6b8986dffb1af653b9129ae3ac2e005d76afc693` | equals `git rev-parse origin/main:tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb` |

## Majors (all five re-verified)

| Finding | Verdict | How it was re-checked |
|---|---|---|
| FRC1-M1 fine-tuned TabICLv2 ablation runs in-context | **Confirmed** | Source: §7.1 passes `condition_override=... + "_" + FOUNDATION_ABLATION_GROUP`; the embedded runner tests `config["condition"] == "fine_tuned"` exactly. Executing the 2.1.0 §7.1 cell with a fine-tuned full run and an in-context ablated run reports the mixed-mode row (delta 0.0) instead of refusing it |
| FRC1-M2 acquisition and §4.1 not retry-safe | **Confirmed** | Executing the 2.1.0 cells: §1.1 after a rejected ZIP raises `FileExistsError` on `data/staged`; a second §4.1 raises `FileExistsError` on `baselines/training_prior_majority` after the registries were reset |
| FRC1-M3 frozen test not bound to the frozen configuration; refreeze unguarded | **Confirmed** | Source: 2.1.0 §8.1 builds the test config from `CONDITION_SPECS`, `MODEL_REPOSITORIES` and `DEVICE_PREFERENCE` (live state; the reduced test reaches those names). §8.0 rewrites `freeze.json` on every run with no test-result check |
| FRC1-M4 README does not say which notebook to use; v2 outputs sentence wrong | **Confirmed** | `tutorials/README.md` has no recommendation; it says v2 retains "saved outputs" while v2 at `0434a02` has 0 code cells with outputs; the v1 row does not cite the legacy evidence. The v2 row also still said "not yet run" although a 2.1.1 run is recorded for 2026-09-27 in `docs/release-verification.md` |
| FRC1-M5 three of eight objectives have no activity | **Confirmed** | Source read of cells 2–41: no cell asks the learner to predict, change, interpret or diagnose for classification-versus-regression, leakage-free band construction or class-order alignment; in-context versus fine-tuning is reachable only through GPU toggles with no prompt. Severity kept at Major (UX1 is a MUST) |

No major was refuted.

## Minors (spot-checked: m1, m2, m3, m6, m7)

| Finding | Verdict | Evidence |
|---|---|---|
| FRC1-m1 | Confirmed | Pinned sample (sha256 `ad2d2a87…be6dd`): test 25.9% low / 33.8% mid / 40.3% high against validation 35.1 / 33.4 / 31.5; months train {4, 5}, val {5}, test {6}. No post-freeze class-balance or test-recall cell exists |
| FRC1-m2 | Confirmed | Validation log loss of the lag-7 rule: 6.882 at ε=1e-6, 3.442 at 1e-3, 1.545 at 0.05 (local CPU, scikit-learn 1.7.2) |
| FRC1-m3 | Confirmed | §0.1 installs only `lightgbm==4.6.0` when missing; NumPy, pandas, scikit-learn and matplotlib are unconstrained |
| FRC1-m6 | Confirmed | Cell 0 says the run "freezes the successful full-feature runs" while `INCLUDE_ABLATIONS_IN_FREEZE = True` by default; §8.2 compares ablations only with the reference model |
| FRC1-m7 | Confirmed, with more binding found | The legacy evidence file is git blob `2f840ffe`, the notebook committed by the maintainer as `33602f8` ("End to end Colab run", 2026-09-26). All 43 cell sources equal the reviewed blob; execution counts 1–19, 0 error outputs, `gpuType` T4, "4 / 4" and "Run-all complete" present. The record is now in `docs/release-verification.md` |

FRC1-m4 and FRC1-m5 were not separately re-run; their source claims (no label/order/ceiling rule in the BYOD text, no row-level predictions or notebook identity in the export) match the 2.1.0 cells read during fixing.

## Note on the review's evidence

The review's statement that v2 2.1.1 fixed NR-01, NR-03 and NR-04 holds: the v2 cells carry those fixes, and the
fixes in v1 2.1.1 are ports of those cells. The review did not mention the 2026-09-27 executed run of v2 2.1.1;
that only strengthens FRC1-M4 (the README was stale about v2 as well).
