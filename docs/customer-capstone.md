# Customer analytics capstone — Candidate

The standalone notebook asks whether purchase history predicts a recorded positive purchase in
the next 30 days. It is an optional small-business workflow capstone, using a historical UK online
retailer with many wholesale customers. Business size is unknown. It is not a Philippine MSME
benchmark, marketing-response experiment, permanent-churn label or causal revenue/profit estimate.

## Implementation

- `tools/build_customer_capstone.py` generates the notebook and verifies parity with `--check`.
- `tools/customer_data.py` verifies official dataset bytes, audits cleaning, builds causal snapshots
  and samples customers without consulting labels.
- `tools/customer_models.py` implements six systems, immutable model acquisition, full-support
  Mitra in-context conditioning and safe JSON/NPY scorer reconstruction. No gradient fine-tuning.
- `tools/customer_metrics.py` computes per-cutoff and equal-cutoff macro scores, probability
  diagnostics and paired customer-cluster bootstrap intervals with repeated draws preserved.
- `tools/customer_runtime.py` enforces ordered stages, source identity, output receipts,
  held-out freeze, fresh-process reconstruction and CSV-derived metric recomputation.
- `tools/customer_byod.py` implements separate authorized evaluation and explicit supplied-artifact
  inference. Original identifier mappings and raw logs stay outside results ZIPs.

The generated carrier contains all source, immutable data/model identities, attribution,
reconstruction instructions and a hashed Linux/Python 3.12 dependency lock. It uses a pinned,
hash-verified uv bootstrap to isolate dependencies without restarting the notebook kernel.
No production worker or production package behavior is changed.

## Verified source feasibility

The [frozen audit](../tutorials/customer_analytics/dataset_audit.json) records both workbook sheets,
1,067,371 rows, timezone-naive coverage from December 2009 through December 2011, 243,007 lines
without customer IDs, and 22,523 cross-sheet canonical overlaps. The official archive and workbook
are bound by byte size and SHA-256 in the [data manifest](../tutorials/customer_analytics/data_manifest.json).
Attribution and transformation notices are in [DATA_LICENSE.md](../tutorials/customer_analytics/DATA_LICENSE.md).

Canonical duplicate keys omit descriptions, producing 34,337 duplicates, versus 34,335 full-original-row
duplicates. Deduplication is a benchmark convention with unresolved provenance. Its monetary effect
is substantial; the audit exposes it rather than calling it unquestionably correct. Labels remain
unchanged when duplicate lines are retained. Future invoice ambiguities cannot alter past features.

The planned label-blind cohorts are feasible: 2,048 support snapshots, 2,000 development and 2,000
test snapshots, plus 1,000 unlabelled December inference snapshots. September test has 379 positive
and 621 negative outcomes; November has 452 positive and 548 negative. The workbook's final partial
December outcome window is never scored as a complete 30-day label window.

## Validation boundaries

The baseline before changes was 33 passing tests and two pre-existing Windows-only path-escaping
failures in `tests/test_workshop_v2_review_fixes.py`. Those failures interpolate a Windows temporary
path into a Python string literal. They are unrelated to this capstone and were not changed.

New CPU tests exercise temporal boundaries, future poisoning, duplicate sensitivity, class mapping,
support-only transforms, ranking ties, bootstrap multiplicity, artifact corruption, stage retries,
full metric recomputation and BYOD refusal/reconstruction. Runtime integration uses a clearly
labelled synthetic Mitra test double; real logistic export is reconstructed in a fresh subprocess.
Actual AutoGluon 1.5 internal preprocessing/support contracts are inspected and exercised separately.
No pretrained weights or GPU inference are run locally.

Final local suite: **108 passed, 1 skipped, 2 baseline failures**. This includes **75 passing new
capstone tests**; the optional real-AutoGluon test skips in the lightweight repository environment.
Separately, the actual upstream trainer and a randomly initialized 6,970-parameter Tab2D network
were exercised on CPU: 32 support rows, 65 queries in 64+1 chunks, zero repeat difference and
5.38e-8 maximum batch-split difference. This used existing CPU Torch 2.11, not the hosted pinned
Torch 2.9.1/pretrained checkpoint, and is only an interface check.

Ruff, release-asset validation, all three generator parity checks, shared-block parity, notebook
spec checks, production API checks, and the existing issue-16/CSV-header checks pass. The Windows
working-tree contract check fails because pre-existing CRLF checkout bytes differ from its LF pins;
the HEAD blob hashes match all expected pins. No contract files were changed.

The integrator independently reran official archive/workbook hash verification, workbook loading,
all snapshot/sensitivity gates and safe six-system fit/export on the actual source. Mitra fitting
here means storing labelled support, not performing pretrained inference. Full new-data scorer
replay is exercised with real logistic models and explicit synthetic Mitra fixtures locally.

CPU/source success does **not** establish that pretrained Mitra completes within the target 45 minutes,
20 GiB initial free disk or 12 GiB peak allocated GPU memory. Hosted qualification is pending.
The run summary records resources and remains Candidate for maintainer review even after execution.

## Notebook Review Framework v1 findings and fixes (2026-09-28)

The review of head `c113dde` (notebook blob `df062f6`) requested changes. The fixes are in the
generator and carried modules; the notebook is regenerated, never hand-edited.

| Finding | Change | Regression evidence |
|---|---|---|
| M1a: mixed date representations silently became missing events and flipped labels | `customer_data.parse_source_times` parses text `invoice_time` strictly as timezone-naive ISO date or date-time (mixed forms allowed) and refuses missing, locale, impossible or offset values with row number and original value before any snapshot. Prefix cleaning keeps unknown-time rows so the invalid-timestamp count cannot be filtered away. Workbook datetimes are unchanged. | The review's 50-customer fixture stays 25/25 through `prepare` at all eight cutoffs; mixed and consistent ISO inputs give identical snapshots; unsupported values stop with diagnostics. |
| M1b: literal `NA` identifiers became missing before surrogate mapping | `customer_data.read_transactions_csv` reads every BYOD and intermediate CSV as text with `keep_default_na=False`; only empty fields are missing. The data manifest records empty identifier counts. | `NA`, `001`, `null` and `N/A` survive surrogate mapping; `NA`, `001` and `null` customers are scored through test, inference and export in the full lifecycle, and an empty customer ID is counted and excluded. |
| E1: new lifecycle test failed CI (`torch.Tensor` missing from the stub) | The stand-in `torch` module now defines `Tensor`, so SciPy's array-API probe works whichever test first imports sklearn. | The test passes alone in a fresh interpreter and in the full suite (SciPy 1.18.1, scikit-learn 1.7.2). |
| m1: raw JSON displayed as the lesson | Compact per-cutoff tables (n, prevalence, k, selected positives/negatives, P@budget, recall, lift, AP, AUROC, Brier), an explicit RFM-minus-R development table at both budgets and a paired-interval table. Full JSON stays in collapsed details and in `results.zip`. | On the reviewer's 1,000-customer fixtures, visible output falls from 8,292 to 14 lines (metrics) and from 14,189 to 9 lines (activity). |
| m2: timeline was a raw preview | The teaching timeline marks each line as qualifying or excluded (cancellation, return, fee, duplicate/ambiguous/invalid), shows half-open window bounds, and `prepare` fails unless recomputed orders, value, products, recency and label equal the stored snapshot. | Checked in the synthetic lifecycle; remains private. Matched timelines for selected/missed test customers were not added. |
| m3: final preview was sampling order | `infer` writes `inference_review_list.csv`: Mitra-RFM rank, score and 20% selection flag using the frozen hash-tie rule. BYOD shows intake counts, a compact preview and the bundle path. | Selected IDs equal `ranked_ids` in the lifecycle test. |
| m4: fixed UCI limitations in BYOD mode | Source identity and the number of test dates come from the current mode and configuration. Default wording is unchanged. | BYOD run summary omits UCI/Philippine source claims and states the actual test-date count. |

User-visible changes: BYOD CSVs with missing or non-ISO timestamps are now refused (previously
coerced silently); literal `NA`-style identifiers are now kept; `inference_review_list.csv` and a
private `teaching_timeline_summary.json` are new outputs; the default run summary text is unchanged.
The default workbook path does not change prepared data: see the parity record in
[release verification](release-verification.md).

After the fixes, the full suite under CI's pins (Python 3.12, pandas 2.3.3, SciPy 1.18.1,
scikit-learn 1.7.2) gives **123 passed, 1 skipped**; the reviewed head gave 1 failed, 109 passed, 1 skipped.

## Required hosted evidence

Use a fresh Colab T4 with default Run all. Preserve exact notebook revision, executed output,
`run_summary.json`, `verification.json`, receipts, checksums and `results.zip`. Run representative
authorized BYOD evaluation and explicit artifact-inference as separate cases. Investigate failures
without shrinking the cohort, changing the hypothesis or widening parity tolerance after scoring.
Do not claim a foundation-model win unless the observed comparison supports it.

Completion reflections are optional, not required submissions. AI-assisted implementation and writing
remain subject to maintainer review; assistance is not independent verification or release approval.
