# mitra_classifier_predictor_inference_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`374a150`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R | Fixed — hosted confirmation pending | Confirmed: Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab-isolated.lock.txt` (74 entries, compiled from the pyproject pins; the pre-existing pip-compile reference locks beside it are unchanged) (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab-isolated.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check); `tools/notebook_template_artifact_inference.py` inherits the isolated-runtime keys, so both notebooks share one lock | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G | Fixed | Confirmed: 1 of 9 guided markers on `main`. Guided opening added (audience, Input → Model → Output, How to use this notebook — with both path fields set Run all completes in one pass — and roadmap); a **Predict** prompt before Sections 4, 6 and 7 and three collapsible **Check your reasoning** answers. No hosted run of this companion is recorded (the registry says it is queued), so the answers describe what each check does and quote no number. Troubleshooting (isolated runtime, digest, archive and runtime-compatibility failures, BYOD), Change one thing, Glossary and Conclusion (your notes) sections; Sections 1–3 labelled Infrastructure and collapsed. | `tools/notebook_template_artifact_inference.py` | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders`, `test_swp_g_checkpoint_answers_quote_only_recorded_facts` |
| SWP-A / SWP-F | Not applicable | No metric is computed (verdict `not-measurable`); nothing is trained. | — | — |
| SWP-B | Fixed | `ARTIFACT_ZIP_PATH` and `NEW_DATA_PATH` already existed. Confirmed in the Section 6 cell: with `ARTIFACT_ZIP_PATH` set and `NEW_DATA_PATH` empty, `files.upload()` raised a bare `NameError` (the `google.colab.files` import lived in Section 4's upload branch). Section 6 now imports inside a guard and names the missing dialog and the path field. | Section 6 code | `test_swp_b_inference_notebook_names_the_missing_dialog_instead_of_a_name_error` |

## User-visible changes

- Section 1 of both notebooks is one collapsed infrastructure cell that builds (or reuses) `dimer_isolated_env_<lock digest>/` from the new hash-locked `tutorials/requirements-colab-isolated.lock.txt` (74 entries: AutoGluon 1.5.0 with its torch, LightGBM, huggingface-hub) and routes later cells to it; nothing is installed into the kernel and Run all needs no restart. Linux x86_64 runtimes only.
- Guided-layer markdown throughout both notebooks; Sections 1–3 collapsed; Troubleshooting, Change one thing, Glossary and Conclusion sections at the end.
- `mitra_classifier_colab`: Section 4 gains `BYOD_PATH` (a CSV, or a pre-split directory); BYOD refusals name the file and the rule. `mitra_classifier_predictor_inference_colab`: Section 6 names the missing dialog instead of a `NameError`.
- The workshop and capstone notebooks and their generators are untouched.
- `docs/release-verification.md`: the two lines describing the in-kernel install and restart now describe the isolated bootstrap. No status text changed.

## Verification (offline; not clean-runtime evidence)

- No model stage ran here (AutoGluon and torch are never installed in CI or here; the Hub is unreachable); no pretrained-inference evidence is claimed. Checkpoint answers quote only the structural facts of the 2026-09-14 record and give no metric number.
- Stand-ins: the Section 1 kernel cell is executed with a stand-in environment and IPython shell; the Section 4 `byod_payloads` helper with temporary files and a fake `google.colab.files.upload`.
- `python tools/build_notebook.py --check` (both templates): OK. `python tools/build_customer_capstone.py --check`: PASS. `python tools/validate_release_assets.py`: PASS. Every CI runner-Python script (`check_shared`, `check_contract`, `validate_colab_tutorial`, `test_pipeline_api_contract`, `test_issue16_hardening`, `test_colab_csv_headers`): OK. `ruff check mitra_pipeline tests tools`: clean. `pytest` (CI deps: pandas, numpy, scikit-learn, openpyxl, matplotlib, nbformat): 166 passed, 1 skipped before → 182 passed, 1 skipped after.
- `uv pip install --dry-run --require-hashes --only-binary :all: -r tutorials/requirements-colab-isolated.lock.txt` into a `uv venv --managed-python --python 3.12.12`: resolves, would install 74 packages (the two repositories' locks are identical apart from the header).

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- Hosted confirmation that AutoGluon's Mitra model resolves the offline Hugging Face snapshot from inside the isolated environment as it did from the in-kernel install (same pins, `HF_HUB_OFFLINE` set by Section 3).
- The companion notebook still has no hosted run (registry: queued); its Run all needs `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256` and `NEW_DATA_PATH` set.
- A full Notebook Review Framework v1 review has not been done.
