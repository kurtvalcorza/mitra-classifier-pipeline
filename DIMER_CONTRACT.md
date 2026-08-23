# Mitra ↔ DIMER Workbench contract

Repo-side only. Exact tasks: `tabular_classification` / `tabular_regression`. Runtime inputs are inventoried in `dimer-runtime-contract.json`. Validator and finetuner share deterministic dataset identity. Results are schema v1 with stable codes. `best.pt` is a ZIP of an AutoGluon `TabularPredictor`, checksummed and verified by unpack→reload→predict before success. Resolved configuration and run identity are provenance. Shared strategy: `KEEP_PARITY_COPIES`.

## DIMER-side requirements — documentation only
No DIMER/backend code is changed: platform work remains task typing, task-neutral defaults, typed artifacts, registry/manifest validation, result schema validation, validated-dataset identity enforcement, non-vision serving, and Base Model transport confirmation (#9). Nested ZIP remains separate validator #8 work.
