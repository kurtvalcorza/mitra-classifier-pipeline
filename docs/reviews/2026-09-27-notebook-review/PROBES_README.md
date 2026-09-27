# Notebook review probes

Reviewed notebook commit: `6c9f911294712eccf993817ef7c48c9f342bc194`.

These are local, synthetic reproductions of selected source logic, not a full notebook run or GPU/model verification. Some probes intentionally reproduce defects and report them instead of failing the script.

## Contents

- `review_probes.py`: transcribed archive-staging and probability-alignment logic, with synthetic cases.
- `probe_results.json`: recorded observations from those probes.
- `control_flow_probes.py`: reduced adaptation-dispatch and freeze-configuration reproductions, plus a synthetic counterexample for interpreting paired contrasts.
- `control_flow_results.json`: recorded results.

## Running

The scripts require Python, NumPy, and scikit-learn. The recorded environment was Python 3.13.5, NumPy 2.3.5, and scikit-learn 1.8.0.

```bash
python review_probes.py
python control_flow_probes.py
```

Each script writes its JSON output beside itself. The scripts do not download models or datasets, contact external services, or modify the GitHub repository. Temporary ZIP test data is removed automatically.

The source review, not these reduced probes alone, establishes where the corresponding logic is used in the real notebook. A fresh Colab/T4 run and end-to-end branch tests remain separate requirements.
