# Reconstruct a customer scorer

This candidate uses JSON and safe numeric NPY arrays only. Do not load pickle,
joblib, or serialized AutoGluon predictors. Install the accompanying hash-locked
Linux/Python 3.12 dependencies. The original pretrained weights are acquired
from the pinned upstream snapshot, checked against their size and SHA-256, and
are deliberately not redistributed in the results ZIP.

Each `scorer/<system>/` contains a manifest, support arrays and preprocessing or
logistic parameters. `customer_models.reload(directory)` verifies that manifest.
Pass a DataFrame with the exact ordered raw feature columns in
`customer_models.FEATURE_NAMES` to `predict`. The adapter applies the stored
transformations. Mitra prediction additionally receives the verified model from
`load_mitra(cache, model_manifest)`. Preserve the full query cohort order and
fixed batch size for exact replay; do not reconfigure query batching.

The default notebook performs stronger verification: it reconstructs the final
cohort from past transactions in a fresh subprocess and compares all six
systems' scores, classes and top-20% IDs. Probability tolerances are absolute
1e-5 and relative 1e-4; mismatches stop execution. Private transaction logs are
needed to rebuild features and are not in the shareable bundle.

## Authorized BYOD

CSV schema: `invoice_id,stock_code,quantity,invoice_time,unit_price,customer_id`
and optional `country`. Names, emails, phone numbers and extra columns are
refused. This schema check does not prove that an identifier is anonymous: use
local pseudonymous IDs and keep the original mapping private. The helper creates
its own surrogate mapping outside the results directory before processing.

Evaluation configuration example (replace these dates with fully covered dates
in your own history):

```json
{
  "rights_confirmed": true,
  "complete_observation_coverage": true,
  "currency": "GBP",
  "source_clock": "timezone-naive",
  "coverage_start": "2009-12-01",
  "coverage_end": "2011-12-09",
  "train_cutoffs": ["2010-04-01", "2010-07-01", "2010-10-01", "2011-01-01"],
  "development_cutoffs": ["2011-04-01", "2011-07-01"],
  "test_cutoffs": ["2011-09-01", "2011-11-01"],
  "inference_cutoff": "2011-12-01"
}
```

V1 intentionally requires GBP and the same feature/cleaning policy. Adapting it
to another currency requires an explicitly reviewed configuration change, not
silent conversion. Full evaluation needs 90-day histories, 30-day outcomes,
mature support labels, both support classes and at least 20 customers in each
class at each scored cutoff. The log-completeness declaration is an assumption.

The evaluation helper invokes prepare → fit → develop → freeze → evaluate →
infer → verify → report in separate processes under an independent output root.
All models receive the business support; the tutorial scorer is not substituted.

Artifact-inference mode requires a previously extracted evidence bundle passed
explicitly with `--artifact`. It checks the complete checksums inventory, pinned
model, schema and cleaning implementation before use. Only rights, currency,
clock, complete history coverage and inference cutoff are needed in this mode;
labels are not constructed and evaluation is reported as not measurable.
It scores a bounded cohort, reconstructs it in a fresh process, compares scores,
classes and top-k IDs, and exports predictions and a verification record.

Checksums detect corruption or accidental modification; they are not a signature
from a trusted publisher. Review artifact provenance before use.
