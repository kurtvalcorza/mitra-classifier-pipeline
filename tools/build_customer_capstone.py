"""Generate the standalone guided customer analytics notebook; no runtime repository dependency."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials/DIMER_Small_Business_Customer_Analytics_Capstone.ipynb"


def carried_files() -> dict[str, str]:
    paths = {
        f"customer_{n}.py": f"tools/customer_{n}.py" for n in ("data", "models", "metrics", "runtime", "byod")
    }
    paths.update(
        {
            n: f"tutorials/customer_analytics/{n}"
            for n in ("data_manifest.json", "model_manifest.json", "DATA_LICENSE.md", "RECONSTRUCT.md")
        }
    )
    paths["requirements.txt"] = "tools/customer-requirements.lock"
    files = {name: (REPO / path).read_text(encoding="utf-8") for name, path in paths.items()}
    files["source.json"] = (
        json.dumps(
            {
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return files


BOOTSTRAP = r"""
from pathlib import Path
import hashlib, json, os, platform, shutil, subprocess, time, urllib.request, zipfile
from IPython.display import display, Markdown, Image
ROOT = Path('/content/dimer_customer_analytics')
assert platform.system() == 'Linux', 'Use a fresh hosted Colab T4 runtime.'
INITIAL_FREE_DISK = shutil.disk_usage('/').free
assert INITIAL_FREE_DISK >= 20 * 1024**3, 'At least 20 GiB free disk required.'
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True)
assert 'T4' in gpu, 'Select a Colab T4 GPU; candidate targets this runtime.'
ROOT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()
for name, text in FILES.items():
    (ROOT/name).write_bytes(text.encode('utf-8'))
for name, checksum in json.loads(FILES['source.json'])['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == checksum, name
uv_url = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
uv_sha = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
wheel = ROOT/'uv.whl'
if not wheel.exists():
    with urllib.request.urlopen(uv_url, timeout=120) as response:
        raw = response.read(20_081_405)
    assert len(raw)==20_081_404 and hashlib.sha256(raw).hexdigest()==uv_sha
    wheel.write_bytes(raw)
assert wheel.stat().st_size==20_081_404 and hashlib.sha256(wheel.read_bytes()).hexdigest()==uv_sha
with zipfile.ZipFile(wheel) as archive:
    matches = [n for n in archive.namelist() if n.endswith('/uv')]
    assert len(matches)==1
    UV=ROOT/'uv'; UV.write_bytes(archive.read(matches[0]))
UV.chmod(0o755)
ENV=dict(os.environ, UV_PYTHON_INSTALL_DIR=str(ROOT/'python'), MPLBACKEND='Agg',
         HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
def command(args, logname):
    with (ROOT/logname).open('w', encoding='utf-8') as log:
        process=subprocess.Popen([str(a) for a in args], cwd=ROOT, env=ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line,end=''); log.write(line); log.flush()
        code=process.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); see {logname}. Do not skip this stage.')
command([UV,'python','install','3.12.12'],'python.log')
PY=ROOT/'venv/bin/python'
if not PY.exists():
    command([UV,'venv','--python','3.12.12',ROOT/'venv'],'venv.log')
command([UV,'pip','sync','--python',PY,'--require-hashes','--only-binary',':all:',ROOT/'requirements.txt'],'install.log')
RESULTS=ROOT/'results'
def stage(name):
    command([PY,ROOT/'customer_runtime.py','--root',ROOT,'--stage',name],name+'.log')
def show(name):
    display(Markdown('```json\n'+(RESULTS/name).read_text(encoding='utf-8')+'\n```'))
print('Isolated environment ready; no notebook restart required.')
"""


def build() -> dict:
    cells = []

    def add(kind, text, hidden=False):
        source = text.strip() + "\n"
        if kind == "code":
            ast.parse(source)
        cell = {
            "cell_type": kind,
            "id": f"customer-{len(cells):02d}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        if hidden:
            cell["metadata"] = {"cellView": "form", "jupyter": {"source_hidden": True}}
        cells.append(cell)

    def md(text):
        add("markdown", text)

    def code(text, hidden=False):
        add("code", text, hidden)

    md("""# Who Is Likely to Buy Again?
## DIMER Small-Business Customer Analytics Capstone

**E2E · GUIDED · NOTEBOOK_SPEC 2.2 · Candidate**

Can historical purchases help rank existing customers by the likelihood of another recorded purchase within 30 days? Compare a transparent rule, logistic regression and Mitra with identical information and customers.

**Input → System → Output:** dated transaction lines → audited 90-day customer histories → six fixed scoring systems → 30-day repeat-purchase scores and an evidence bundle.

**For learners:** basic Python and Colab familiarity; no prior ML experience needed. Select **Runtime → Change runtime type → T4 GPU**, then **Run all**. No token, paid API, repository clone, upload or manual restart is required. Infrastructure cells may remain collapsed. Allow a target 45 minutes and 20 GiB free disk; runtime and 12 GiB GPU-memory targets await hosted validation.

**Roadmap:** audit → build timelines → condition models → compare development results → freeze → evaluate later dates → reconstruct scores → conclude. You will identify leakage, distinguish lines from orders, interpret a fixed review budget and explain limits.

This is a historical **UK online retailer**, with many wholesale customers. Its business size is not established. It illustrates a workflow relevant to small businesses; it is **not Philippine MSME evidence**. Repeat-purchase likelihood does not measure marketing responsiveness, campaign uplift or profit. No outreach is performed.""")
    md("""### 1. Know the data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), Daqing Chen, [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The official workbook has 1,067,371 rows in two sheets. Acquisition verifies archive and workbook byte sizes and SHA-256 hashes; changed files fail visibly.

One row is a transaction **line**, not an order. Positive merchandise lines are grouped by invoice/customer. Returns, cancellations, fees, malformed lines and ambiguous invoices are excluded. Exact-line deduplication is an explicit assumption because provenance cannot distinguish repeated legitimate lines from duplicate exports. A sensitivity audit compares retained duplicates and requires identical order-existence labels.

Money stays in **GBP gross positive purchase value**, never net revenue or profit. Missing customer IDs cannot be combined into a fictitious customer. Timestamps retain the source's timezone-naive clock. Log completeness is an assumption, not proven by the last timestamp.

**Predict:** Can ten lines on one invoice tell us that a customer placed ten orders?
<details><summary>Worked answer</summary>No. They can describe ten products on one order. Counting lines as orders changes the meaning of purchase frequency.</details>""")
    md("""### Infrastructure — carried implementation and isolated environment
Run these cells without studying the packaging code. All experiment source is embedded below; runtime downloads are limited to pinned upstream dependencies, model files and the official dataset.""")
    code("FILES = " + repr(carried_files()), True)
    code(BOOTSTRAP, True)
    md("""### 2. Build histories without seeing the future

For a midnight cutoff **t**, history is **[t − 90 days,t)** and the label window is **[t,t + 30 days)**. An event exactly at t is an outcome; it is never a feature. No complete future window means **not measurable**, not a negative label.

Training cutoffs: April/July/October 2010 and January 2011. Development: April/July 2011. Held-out test: September/November 2011. Up to 512 customers per training cutoff and 1,000 per scored cutoff are selected by label-blind hashes. Customers can recur: this measures future behaviour in an existing-customer population, not customer-disjoint generalization.

**Predict:** Would total spending through the end of the workbook be a valid feature for an April 2011 prediction?
<details><summary>Worked answer</summary>No. That total contains purchases after the prediction date. Every feature uses only records visible before its cutoff, including cleaning decisions.</details>""")
    code("stage('prepare')\nshow('cohort_manifest.json')\nshow('dataset_audit.json')")
    code(
        "import pandas as pd\ntimeline=pd.read_csv(ROOT/'private/teaching_timeline.csv')\ndisplay(timeline.head(20))\nprint('History builds features; outcome establishes the label. Raw logs stay outside results.zip.')"
    )
    md("""**What to notice:** inspect exclusions, eligible versus selected counts and both label classes at each cutoff. The data gate stops if either scored class has fewer than 20 customers. A negative label means no qualifying event at this retailer in that window, not permanent churn.

### 3. Six fair comparisons

**R** is recency: days since the last qualifying order. **RFM** adds order count, gross value, average order value and product count in the same 90 days. Count/value features use log1p. IDs are grouping keys, never model features.

| System | Information | Meaning |
|---|---|---|
| Training prevalence | Support labels only | Constant probability baseline |
| Recency rule | R | Recent purchasers ranked first; no probability claim |
| Logistic R / RFM | One / five features | Fixed L2 logistic models, support-only scaling |
| Mitra R / RFM | Same one / five features | Frozen pretrained weights conditioned on the identical labelled support |

Mitra uses **in-context conditioning**, not gradient fine-tuning. Support labels mature before development. Development/test rows never become support. Foundation-model superiority is not a success requirement.

**Predict:** Will the richer feature set consistently beat recency alone?""")
    code(
        "stage('fit')\nshow('support_metadata.json')\nstage('develop')\ndisplay(Image(filename=str(RESULTS/'dev_comparison.png')))\nshow('dev_metrics.json')"
    )
    md("""### 4. Change one thing: R → RFM

Compare the two logistic conditions, then the two Mitra conditions **on development only**. Support, customers, horizon and 20% budget remain fixed. Record whether the extra historical information helps. The 10% budget output is optional display-only exploration; it does not change the frozen 20% test budget.

**Precision@20%:** among the top ceil(0.2 × n) customers, what fraction actually buy again? **Recall:** what fraction of all repeat buyers are captured? **Lift:** precision divided by the cutoff's observed repeat-purchase prevalence. **AP** uses standard tied-score average precision, not trapezoidal PR-AUC. AUROC measures ranking across thresholds. Brier score applies only to probability systems; a reliability plot alone does not establish calibration.

<details><summary>Interpretation guidance</summary>RFM can help by distinguishing equally recent customers with different purchasing histories. It can also add noise. State the actual difference and period rather than assuming a larger model or more features must win. A likely buyer may buy anyway without an offer.</details>""")
    code(
        "show('development_activity.json')\nstage('freeze')\nprint('Dataset, support, settings, cohorts and 20% budget frozen before held-out predictions.')"
    )
    md("""### 5. Evaluate later periods

**Predict:** Will the development pattern hold in September and November? Each cutoff is reported separately and receives equal weight in the macro average. The primary contrast is Mitra-RFM minus Logistic-RFM Precision@20%.

The 2,000-replicate paired bootstrap samples **customers**, retaining their snapshots and repeated draws. Intervals are conditional on these dates and frozen systems. Two test dates do not establish temporal robustness or transfer to another business. Invalid replicates are counted explicitly.""")
    code(
        "stage('evaluate')\ndisplay(Image(filename=str(RESULTS/'test_comparison.png')))\nshow('test_metrics.json')\nshow('paired_intervals.json')\ndiagnostics=pd.read_csv(RESULTS/'customer_diagnostics.csv')\ndisplay(diagnostics.groupby(['cutoff','outcome_group']).head(2))"
    )
    md("""**What to notice:** selected negatives, missed positives, differences across dates, recurring versus unseen-in-support customers and the paired interval. Do not tune settings from these outcomes. Unknown pretraining overlap and single-retailer history limit every system's claim.

### 6. Score a new historical snapshot and reconstruct the scorer

The December 1, 2011 demonstration uses earlier purchases only. The source does not contain its full next 30 days, so evaluation is **not measurable**. Mitra-RFM is the predefined example scorer whether or not it wins; the logistic reference is exported too.

Safe JSON and numeric arrays preserve schema, preprocessing, labels and support. A fresh process reconstructs the histories and scores, checks hashes, and requires probability parity (atol 1e−5, rtol 1e−4), exact predicted classes and exact top 20% IDs. Raw transactions and pretrained weights are excluded from the results ZIP. Pseudonymous dataset IDs are not a promise of anonymity for business records.""")
    code(
        "stage('infer')\nstage('verify')\nshow('verification.json')\ndisplay(pd.read_csv(RESULTS/'inference_predictions.csv').query(\"system == 'mitra_RFM'\").head())"
    )
    md("""### 7. Conclude with evidence

Complete this optional reflection: “For ___ customers at ___ cutoffs, Mitra-RFM Precision@20% was ___ versus ___ for Logistic-RFM. The paired difference and interval were ___. The important failure/uncertainty was ___. This does not establish ___.” **Completion records are optional and are not required submissions.**

The notebook stays **Candidate** until its exact revision completes a fresh hosted default run and representative BYOD validation, and the maintainer reviews the evidence. Local CPU tests are not hosted model execution evidence.""")
    code(
        "(ROOT/'wall_resources.json').write_text(json.dumps({'elapsed_seconds_including_setup_before_report':time.monotonic()-STARTED,'initial_free_disk_bytes':INITIAL_FREE_DISK,'gpu_name':gpu.strip()}))\nstage('report')\nshow('run_summary.json')\nprint('Wall minutes including setup:',round((time.monotonic()-STARTED)/60,2))\nprint('Evidence:',RESULTS/'results.zip')\nDOWNLOAD_RESULTS=False\nif DOWNLOAD_RESULTS:\n    from google.colab import files\n    files.download(str(RESULTS/'results.zip'))"
    )
    md("""### 8. Optional bring your own transactions

Disabled during default Run all. Upload your prepared files using Colab's file browser, then set paths below. Use only records you are authorized to process. CSV columns: `invoice_id,stock_code,quantity,invoice_time,unit_price,customer_id` and optional `country`. Do not include names, emails, phone numbers or payment details. Unexpected columns are refused.

The config declares `rights_confirmed:true`, `complete_observation_coverage:true`, `source_clock:"timezone-naive"`, `currency:"GBP"`, `coverage_start`, `coverage_end`, `train_cutoffs`, `development_cutoffs`, `test_cutoffs`, and `inference_cutoff`. Provide full 90-day histories and 30-day labelled windows, chronologically mature support and ordered cutoffs. V1 refuses currencies other than GBP and incompatible policies rather than silently transferring a model. Local surrogate IDs replace original customer IDs in the shareable bundle; the private mapping stays outside it.

**Evaluation mode** fits/conditions a separate business model and runs the same workflow. **Artifact-inference mode** explicitly consumes your compatible prior scorer, requires only history coverage, and exports unlabelled scores; it never silently substitutes the tutorial model. See the embedded reconstruction guide for the precise contract.""")
    code(
        "RUN_BYOD=False\nBYOD_MODE='evaluate' # or 'artifact-inference'\nBYOD_CSV=ROOT/'my_transactions.csv'\nBYOD_CONFIG=ROOT/'my_configuration.json'\nBYOD_ARTIFACT=ROOT/'my_prior_results'\nif RUN_BYOD:\n    command([PY,ROOT/'customer_byod.py','--root',ROOT,'--csv',BYOD_CSV,'--config',BYOD_CONFIG,'--mode',BYOD_MODE,'--artifact',BYOD_ARTIFACT],'byod.log')"
    )
    md("""### Troubleshooting and glossary

| Problem | Action |
|---|---|
| Wrong accelerator / insufficient memory | Use the documented fresh T4. Do not silently shrink cohorts after seeing scores. |
| Download or dependency failure | Check the failing log and network, then rerun from the failed stage. Hash changes require review. |
| Stale receipt or modified output | Rerun the changed stage and every downstream stage. Do not bypass the check. |
| Incomplete coverage / class gate failure | Revise the proposed design before modelling; do not relabel censored customers as negatives. |
| Reload mismatch | Preserve evidence and stop. Do not widen tolerance. |

**Cutoff:** the time at which a prediction is made. **Support:** labelled historical examples used to fit/condition a system. **Censoring:** the full outcome is not observable. **Held-out:** outcomes reserved for final evaluation. **Cluster bootstrap:** resampling customer groups rather than pretending their repeated rows are independent. **Calibration:** agreement between probabilities and observed frequencies.

### AI Assistance Disclosure
Code, instructional text and design documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for implementation review, result validation and release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python3", "language": "python"},
            "language_info": {"name": "python", "version": "3.12.12"},
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "provenance": []},
            "dimer": {
                "notebook_profile": "E2E",
                "notebook_mode": "GUIDED",
                "notebook_spec": "2.2",
                "standalone": True,
                "status": "Candidate",
            },
        },
    }


def materialize(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in carried_files().items():
        (root / name).write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = (json.dumps(build(), indent=1, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not NOTEBOOK.exists() or NOTEBOOK.read_bytes() != payload:
            raise SystemExit("Customer capstone generator parity failed")
        print("PASS: customer capstone generator parity")
    else:
        NOTEBOOK.write_bytes(payload)
        print(f"Wrote {NOTEBOOK}: {len(payload):,} bytes")
