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


# Carried strings are written as short implicitly concatenated pieces so no notebook line is
# hundreds of thousands of characters long (one huge line can freeze a browser tab). Python joins
# the pieces back into identical text, so the written files and their SHA-256 values are unchanged.
CARRIER_PIECE = 1000


def carried_literal(value: object) -> str:
    """Return a Python literal for ``value`` (str, or dict of them) with no long source lines."""
    if isinstance(value, dict):
        return "{\n" + "".join(f"{key!r}: {carried_literal(item)},\n" for key, item in value.items()) + "}"
    if isinstance(value, str):
        pieces = [
            line[start : start + CARRIER_PIECE]
            for line in value.splitlines(keepends=True)
            for start in range(0, len(line), CARRIER_PIECE)
        ]
        if not pieces:
            return repr(value)
        return "(\n" + "".join(f"    {piece!r}\n" for piece in pieces) + ")"
    return repr(value)


BOOTSTRAP = r"""
from pathlib import Path
import hashlib, html, json, os, platform, shutil, subprocess, time, urllib.request, zipfile
import pandas as pd
from IPython.display import display, HTML, Image
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
REFUSED_EXIT=3  # exit code of customer_byod.py for an input it refused with one printed message; not a crash
def command(args, logname, refusal_ok=False):
    # A refusal (exit 3) is returned only when refusal_ok; any other nonzero exit is a failure and raises.
    with (ROOT/logname).open('w', encoding='utf-8') as log:
        process=subprocess.Popen([str(a) for a in args], cwd=ROOT, env=ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line,end=''); log.write(line); log.flush()
        code=process.wait()
    if code==REFUSED_EXIT and refusal_ok:
        return code
    if code:
        raise RuntimeError(f'Command failed ({code}); see {logname}. Fix the cause printed above, then rerun '
                           'this cell; the cells after it depend on it.')
    return code
command([UV,'python','install','3.12.12'],'python.log')
PY=ROOT/'venv/bin/python'
if not PY.exists():
    command([UV,'venv','--python','3.12.12',ROOT/'venv'],'venv.log')
command([UV,'pip','sync','--python',PY,'--require-hashes','--only-binary',':all:',ROOT/'requirements.txt'],'install.log')
RESULTS=ROOT/'results'
def stage(name):
    command([PY,ROOT/'customer_runtime.py','--root',ROOT,'--stage',name],name+'.log')
def load(name, folder=None):
    return json.loads(((folder or RESULTS)/name).read_text(encoding='utf-8'))
def show(name, expanded=False, folder=None):
    # Complete machine-readable record, collapsed; the compact tables below are for reading.
    text=((folder or RESULTS)/name).read_text(encoding='utf-8')
    display(HTML(f"<details{' open' if expanded else ''}><summary>Full {html.escape(name)} "
        f"({len(text.splitlines()):,} lines; also in results.zip)</summary><pre>{html.escape(text)}</pre></details>"))
def table(frame):
    print(frame.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
COMPACT={'system':'system','cutoff':'cutoff','n':'n','prevalence':'prevalence','k':'k',
    'selected_positives':'sel_pos','selected_negatives':'sel_neg','precision_at_budget':'P@budget',
    'recall_at_budget':'recall','lift_at_budget':'lift','average_precision':'AP','auroc':'AUROC','brier':'Brier'}
def metric_table(report):
    print(f"Budget {report['budget']:.0%}: top ceil({report['budget']} x n) customers per cutoff; Brier only for probability systems.")
    table(pd.DataFrame(report['per_cutoff'])[list(COMPACT)].rename(columns=COMPACT))
def feature_contrast(activity):
    rows=[]
    for key in ('canonical_budget_20','optional_display_budget_10'):
        cells={(r['system'],r['cutoff']):r for r in activity[key]['per_cutoff']}
        for model in ('logistic','mitra'):
            for cutoff in sorted({c for _,c in cells}):
                r,f=cells[(model+'_R',cutoff)],cells[(model+'_RFM',cutoff)]
                rows.append({'budget':f"{activity[key]['budget']:.0%}",'model':model,'cutoff':cutoff,'k':r['k'],
                    'P@budget R':r['precision_at_budget'],'P@budget RFM':f['precision_at_budget'],
                    'RFM-R P@budget':f['precision_at_budget']-r['precision_at_budget'],
                    'RFM-R AP':None if None in (f['average_precision'],r['average_precision']) else f['average_precision']-r['average_precision'],
                    'RFM-R AUROC':None if None in (f['auroc'],r['auroc']) else f['auroc']-r['auroc']})
    table(pd.DataFrame(rows))
def interval_table(paired):
    print(f"{paired['valid_replicates']:,} valid of {paired['replicates']:,} customer-cluster replicates; "
          f"{paired['invalid_replicates']:,} invalid (empty or single-class cutoff).")
    table(pd.DataFrame([{'contrast':name,'difference':c['difference'],'ci95_low':c['ci95'][0],'ci95_high':c['ci95'][1],
        'measure':c['measure']} for name,c in paired['contrasts'].items()]))
"""

# This run's answers (SBC-M1): computed from the run's own saved results, so the stated conclusion always
# matches the printed tables. Plain Python on loaded JSON; tests exec this text directly.
ANSWERS = r"""
def _per_cutoff(report):
    return {(r['system'], r['cutoff']): r for r in report['per_cutoff']}
def _same(a, b):
    return a == b or (a is not None and b is not None and abs(a - b) <= 1e-12)
def _interval(contrast):
    if contrast.get('ci95') is None:
        return f"{contrast['difference']:+.3f} (no valid bootstrap replicate, so no interval)", None
    low, high = contrast['ci95']
    verdict = 'above 0' if low > 0 else 'below 0' if high < 0 else 'includes 0'
    return f"{contrast['difference']:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}]: interval {verdict}", verdict
def development_answer(activity):
    report = activity['canonical_budget_20']
    cells = _per_cutoff(report)
    cutoffs = sorted({c for _, c in cells})
    diffs = [cells[(m + '_RFM', c)]['precision_at_budget'] - cells[(m + '_R', c)]['precision_at_budget']
             for m in ('logistic', 'mitra') for c in cutoffs]
    wins = sum(d > 0 for d in diffs)
    verdict = ('yes, RFM beat R for both models at every development date' if wins == len(diffs) else
               'no, RFM never beat R' if wins == 0 else 'not consistently, RFM beat R in some comparisons only')
    same = all(_same(cells[('logistic_R', c)][k], cells[('recency', c)][k]) for c in cutoffs
               for k in ('precision_at_budget', 'average_precision', 'auroc'))
    return [f"This run's answer ({report['budget']:.0%} budget, development {', '.join(cutoffs)}): {verdict}.",
            f"RFM minus R Precision@budget ranged from {min(diffs):+.3f} to {max(diffs):+.3f} "
            f"({wins} of {len(diffs)} model-and-date comparisons positive).",
            'Logistic R and the recency rule have identical P@budget, AP and AUROC at every development date: '
            'they rank customers in the same order.' if same else
            'Logistic R and the recency rule rank customers differently on at least one development date; '
            'compare their rows.']
def test_answer(test, paired):
    cells = _per_cutoff(test)
    cutoffs = sorted({c for _, c in cells})
    held = all(cells[(m + '_RFM', c)]['precision_at_budget'] > cells[(m + '_R', c)]['precision_at_budget']
               for m in ('logistic', 'mitra') for c in cutoffs)
    lines = [f"This run's answer (test {', '.join(cutoffs)}): " + (
        'the development pattern held; RFM beat R for both models at every test date.' if held else
        'the development pattern did not hold everywhere; RFM did not beat R for both models at every test date.')]
    for m, name in (('mitra', 'Mitra'), ('logistic', 'Logistic')):
        lines.append(f"{name} RFM minus {name} R, macro Precision@20%: "
                     f"{_interval(paired['contrasts'][f'{m}_RFM_minus_{m}_R'])[0]}.")
    text, verdict = _interval(paired['contrasts']['mitra_RFM_minus_logistic_RFM'])
    lines.append(f"Primary contrast, Mitra-RFM minus Logistic-RFM, macro Precision@20%: {text}.")
    lines.append({
        'includes 0': 'A tie: these dates cannot tell the two scorers apart, so neither model won. Both ran fixed, '
                      'untuned recipes, so a tie here does not show the models are equivalent in general.',
        'above 0': 'Mitra-RFM ranked better on these dates, under these fixed, untuned recipes and this data; '
                   'this is not a general ranking of the models.',
        'below 0': 'Logistic-RFM ranked better on these dates, under these fixed, untuned recipes and this data; '
                   'this is not a general ranking of the models.',
        None: 'No interval could be computed, so no comparative claim is made.'}[verdict])
    return lines
def conclusion(test, paired):
    cells = _per_cutoff(test)
    cutoffs = sorted({c for _, c in cells})
    lifts = {s: test['macro'][s]['lift_at_budget'] for s in ('logistic_RFM', 'mitra_RFM')}
    lift_text = f"macro lift {lifts['logistic_RFM']:.2f} for Logistic-RFM and {lifts['mitra_RFM']:.2f} for Mitra-RFM"
    rfm = [_interval(paired['contrasts'][f'{m}_RFM_minus_{m}_R'])[1] for m in ('mitra', 'logistic')]
    primary = _interval(paired['contrasts']['mitra_RFM_minus_logistic_RFM'])[1]
    if min(lifts.values()) > 1 and all(v == 'above 0' for v in rfm):
        signal = ('1. Yes, for this retailer. Ranking by 90-day purchase history put more repeat buyers in the top 20% '
                  f'than the base rate ({lift_text}), and the five RFM features beat recency alone for both models '
                  '(intervals above 0). Claim type: the task signal is present.')
    elif min(lifts.values()) > 1:
        signal = ('1. Partly. Both RFM scorers put more repeat buyers in the top 20% than the base rate '
                  f'({lift_text}), but RFM did not clearly beat recency alone for both models. Claim type: some task '
                  'signal; the extra features are not shown to help.')
    else:
        signal = (f'1. Not shown. At least one RFM scorer did not beat the base rate at the 20% budget ({lift_text}), '
                  'so this run does not support ranking customers by purchase history.')
    choice = {
        'includes 0': '2. Model choice: a tie between Mitra-RFM and Logistic-RFM (interval includes 0). For this use '
                      'the simpler, cheaper and more transparent logistic model is the reasonable pick; Mitra added '
                      'nothing measurable here. Claim type: comparator, under fixed recipes.',
        'above 0': '2. Model choice: Mitra-RFM ranked better than Logistic-RFM on these dates (interval above 0), '
                   'under fixed recipes. Weigh that gain against its larger download and GPU need.',
        'below 0': '2. Model choice: Logistic-RFM ranked better than Mitra-RFM on these dates (interval below 0), '
                   'under fixed recipes; prefer it.',
        None: '2. Model choice: no interval was available, so no comparative claim is made.'}[primary]
    counts = ', '.join(f"{cells[('recency', c)]['n']:,} on {c}" for c in cutoffs)
    return ["This run's answer to the opening question:", signal, choice,
            f'3. Evidence level: tutorial evidence from one historical UK online retailer; {len(cutoffs)} test dates '
            f'({counts} customers); only customers with at least one qualifying order in the 90 days before each '
            'cutoff; fixed, untuned recipes for both learned models; intervals conditional on these dates.',
            '4. Not established: response to a campaign or uplift (a likely buyer may buy anyway), profit, other '
            'businesses or countries (not Philippine MSME evidence), customers inactive for more than 90 days, or a '
            'general ranking of Mitra versus logistic regression.']
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

**E2E · GUIDED · CAPSTONE · NOTEBOOK_SPEC 2.5 · Candidate**

> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,
> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party
> model weights and datasets under their own licenses (see the [model card](https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/main/MODEL_CARD.md)) and uses compute
> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are
> responsible for having the right to use it. Results here are tutorial evidence, not
> benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum
> extent permitted by law, the maintainers and the DIMER platform accept no liability for
> its use.

### Problem statement

A small business keeps a record of past purchases and has limited time to follow up with customers. It wants to know which recent customers are most likely to buy again soon, so that a fixed review effort goes first to the customers most likely to buy again. **Question:** can historical purchases help rank existing customers by the likelihood of another recorded purchase within 30 days?

- **Who is scored:** at each prediction date, customers with at least one qualifying order in the previous 90 days. Customers inactive for longer are out of scope.
- **What a useful outcome looks like:** with a review budget of the top 20% of customers, the ranked list holds clearly more repeat buyers than the base rate (Precision@20% above the prevalence, lift above 1). The comparison also shows whether Mitra, a pretrained tabular foundation model [1], [2], adds anything over logistic regression [4] when both get identical information and customers.
- **Scope boundary:** the data come from a historical **UK online retailer** with many wholesale customers [3]. Its business size is not established. The notebook illustrates a workflow relevant to small businesses; it is **not Philippine MSME evidence**. Repeat-purchase likelihood does not measure marketing responsiveness, campaign uplift or profit. No outreach is performed.

**Input → System → Output:** dated transaction lines → audited 90-day customer histories → six fixed scoring systems → 30-day repeat-purchase scores, this run's answer to the question, and an evidence bundle.

**For learners:** basic Python and Colab familiarity; no prior ML experience needed. Select **Runtime → Change runtime type → T4 GPU**, then **Run all**. No token, paid API, repository clone, upload or manual restart is required. Infrastructure cells may remain collapsed.

**Runtime and resources:** measured at 10.4 minutes including setup, with 0.72 GB peak allocated GPU memory, on a fresh Colab T4 (browser Run all of revision `c81db75`, 2026-09-28); a Colab CLI run of revision `9698515` on a fresh T4 took 10.44 minutes (2026-10-03). Both were measured on earlier revisions of this notebook and are not re-run here. Allow up to 45 minutes and 20 GiB free disk. The run downloads the 302.7 MB Mitra checkpoint, the 45.6 MB UCI archive and the pinned Python packages.

**Roadmap:** audit → build timelines → condition models → compare development results → freeze → evaluate later dates → reconstruct scores → conclude. You will identify leakage, distinguish lines from orders, interpret a fixed review budget and explain limits.""")
    md("""### 1. Know the data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) [3], Daqing Chen, [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The official workbook has 1,067,371 rows in two sheets. Acquisition verifies archive and workbook byte sizes and SHA-256 hashes; changed files fail visibly.

One row is a transaction **line**, not an order. Positive merchandise lines are grouped by invoice/customer. Returns, cancellations, fees, malformed lines and ambiguous invoices are excluded. Exact-line deduplication is an explicit assumption because provenance cannot distinguish repeated legitimate lines from duplicate exports. A sensitivity audit compares retained duplicates and requires identical order-existence labels.

Money stays in **GBP gross positive purchase value**, never net revenue or profit. Missing customer IDs cannot be combined into a fictitious customer. Timestamps retain the source's timezone-naive clock. Log completeness is an assumption, not proven by the last timestamp.

**Predict:** Can ten lines on one invoice tell us that a customer placed ten orders?
<details><summary>Worked answer</summary>No. They can describe ten products on one order. Counting lines as orders changes the meaning of purchase frequency.</details>""")
    md("""### Infrastructure — carried implementation and isolated environment
Run these cells without studying the packaging code. All experiment source is embedded below; runtime downloads are limited to pinned upstream dependencies, model files and the official dataset.""")
    code("FILES = " + carried_literal(carried_files()), True)
    code(BOOTSTRAP + ANSWERS + "print('Isolated environment ready; no notebook restart required.')", True)
    md("""### 2. Build histories without seeing the future

For a midnight cutoff **t**, history is **[t − 90 days,t)** and the label window is **[t,t + 30 days)**. An event exactly at t is an outcome; it is never a feature. No complete future window means **not measurable**, not a negative label.

**Who is scored:** a customer is eligible at cutoff t only with at least one qualifying order in [t − 90 days,t). Customers inactive for longer are out of scope, so recency never exceeds 90 days and the results say nothing about lapsed customers. In the cohort table below, `eligible` counts these customers and `selected` is the label-blind sample drawn from them.

Training cutoffs: April/July/October 2010 and January 2011. Development: April/July 2011. Held-out test: September/November 2011. Up to 512 customers per training cutoff and 1,000 per scored cutoff are selected by label-blind hashes. Customers can recur: this measures future behaviour in an existing-customer population, not customer-disjoint generalization.

**Predict:** Would total spending through the end of the workbook be a valid feature for an April 2011 prediction?
<details><summary>Worked answer</summary>No. That total contains purchases after the prediction date. Every feature uses only records visible before its cutoff, including cleaning decisions.</details>""")
    code(
        "stage('prepare')\ntable(pd.DataFrame(load('cohort_manifest.json')['cohorts']).drop(columns='id_hash'))\nshow('cohort_manifest.json')\nshow('dataset_audit.json')"
    )
    code(
        "summary=load('teaching_timeline_summary.json', ROOT/'private')\ntimeline=pd.read_csv(ROOT/'private/teaching_timeline.csv', dtype=str, keep_default_na=False)\nfor column in ('unit_price','line_value'):  # money at display precision; the private CSV keeps full precision\n    timeline[column]=pd.to_numeric(timeline[column], errors='coerce').round(2).map('{:.2f}'.format)\nprint(f\"Cutoff t = {summary['cutoff']}. History {summary['history_window']} and outcome {summary['outcome_window']} are half-open: start included, end excluded.\")\nprint(timeline.to_string(index=False))\nfor name,values in (('Recomputed from qualifying lines',summary['recomputed_from_qualifying_lines']),('Stored snapshot features and label',summary['snapshot_features_and_label'])):\n    print(name+':', {k:(round(v,2) if isinstance(v,float) else v) for k,v in values.items()})\nprint('Money is shown in GBP to 2 decimal places. History builds features; outcome establishes the label. Raw logs stay outside results.zip.')"
    )
    md("""**Try it:** in the timeline, count the distinct invoices marked *qualifying* in the history window and add their line values (shown to 2 decimal places, so allow a penny of rounding). They must equal the stored order count and gross value; a qualifying outcome invoice makes the label 1. Cancelled, returned, fee and duplicate lines are listed but never count as orders.

**What to notice:** inspect exclusions, eligible versus selected counts and both label classes at each cutoff. The data gate stops if either scored class has fewer than 20 customers. A negative label means no qualifying event at this retailer in that window, not permanent churn.

### 3. Six systems, identical information

**R** is recency: days since the last qualifying order. **RFM** adds order count, gross value, average order value and product count in the same 90 days. Count/value features use log1p. IDs are grouping keys, never model features.

| System | Information | Meaning |
|---|---|---|
| Training prevalence | Support labels only | Constant probability baseline |
| Recency rule | R | Recent purchasers ranked first; no probability claim |
| Logistic R / RFM | One / five features | L2 logistic regression [4] with fixed settings (C = 1.0), support-only scaling |
| Mitra R / RFM | Same one / five features | Frozen pretrained Mitra weights conditioned on the identical labelled support |

**Mitra** is AutoGluon's tabular foundation model: a 12-layer transformer pretrained on about 45 million synthetic tables. It predicts for new rows by reading a labelled support set as part of its input (in-context learning) instead of updating its weights [1]. This notebook uses the checkpoint [`autogluon/mitra-classifier`](https://huggingface.co/autogluon/mitra-classifier/tree/c425e9fa0910a6be1c494321792e7ba2a1367b1a) at revision `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, licensed Apache-2.0 [2]. Its files are checked against pinned sizes and SHA-256 hashes when downloaded.

**What this comparison can and cannot show.** Neither learned system is tuned. Logistic regression uses fixed settings, and Mitra runs a fixed recipe: one estimator (`n_estimators=1`), no fine-tuning and no ensembling. Every system sees the same features, support, customers, horizon and 20% budget. So Mitra-RFM versus Logistic-RFM isolates the scorer under these fixed recipes, and R versus RFM isolates the feature set within each model. A difference here is evidence about these recipes on this data (a comparator claim). It is not a general ranking of the models, and it does not show that a ranked list is useful in operation.

A one-feature logistic model is a monotone function of recency. Whenever its fitted coefficient gives recent buyers higher probabilities, logistic R ranks customers exactly like the recency rule, with identical P@20%, AP and AUROC; only its probabilities, and so its Brier score, differ. The answer printed after section 4 checks this.

Mitra uses **in-context conditioning**, not gradient fine-tuning. Support labels mature before development. Development/test rows never become support. Foundation-model superiority is not a success requirement.

**Predict:** Will the richer feature set consistently beat recency alone?""")
    code(
        "stage('fit')\nmitra=load('model_manifest.json', ROOT)\nprint(f\"Mitra checkpoint: {mitra['modelId']} at revision {mitra['revision']} ({mitra['totalBytes']/1e6:.1f} MB, pinned by SHA-256).\")\nshow('support_metadata.json')\nstage('develop')\ndisplay(Image(filename=str(RESULTS/'dev_comparison.png')))\nmetric_table(load('dev_metrics.json'))\nshow('dev_metrics.json')"
    )
    md("""### 4. Change one thing: R → RFM

The notebook already ran both conditions for you: each learned model was scored once with R and once with RFM, with support, customers, horizon and the 20% budget fixed. Compare the two logistic conditions, then the two Mitra conditions, **on development only**, and record whether the extra historical information helps. Your own change comes after the table: a review-budget control that you set and rerun. The 10% budget rows are optional display-only exploration; they do not change the frozen 20% test budget.

**Precision@20%:** among the top ceil(0.2 × n) customers, what fraction actually buy again? **Recall:** what fraction of all repeat buyers are captured? **Lift:** precision divided by the cutoff's observed repeat-purchase prevalence. **AP** uses standard tied-score average precision, not trapezoidal PR-AUC. AUROC measures ranking across thresholds. Brier score applies only to probability systems; a reliability plot alone does not establish calibration.

<details><summary>Interpretation guidance</summary>RFM can help by distinguishing equally recent customers with different purchasing histories. It can also add noise. State the actual difference and period rather than assuming a larger model or more features must win. A likely buyer may buy anyway without an offer.</details>""")
    code(
        "print('RFM minus R on development, same model, support, customers and horizon:')\nactivity=load('development_activity.json')\nfeature_contrast(activity)\nprint()\nprint(*development_answer(activity), sep='\\n')\nshow('development_activity.json')\nstage('freeze')\nprint('Dataset, support, settings, cohorts and 20% budget frozen before held-out predictions.')"
    )
    md("""<details><summary>Worked answer: will the richer feature set consistently beat recency alone?</summary>

Read the **This run's answer** lines above. They are computed from this run's development table, so they always agree with it. "Consistently" means for both models at every development date: one date where RFM − R is zero or negative is enough to answer "not consistently". A positive difference means that, among customers who bought equally recently, order count, spend and product variety helped separate likely repeat buyers from the rest. This is development evidence only; section 5 checks whether it holds on later dates.

The last line checks the claim from section 3. When it reports them identical, the `recency` and `logistic_R` rows of the development metric table match because a one-feature logistic model only rescales recency; that is expected, not a bug. The comparison that adds information is R versus RFM.
</details>

**Your change (optional, display only).** **Predict** first: if the business could review twice as many customers (a 40% budget), would Precision@budget rise or fall, and would RFM's advantage over R grow or shrink? Then **change** `EXPLORE_BUDGET` in the next cell to 0.4, **run** that one cell, **observe** the new table and **explain** the difference. The control re-ranks the saved development predictions only. It writes nothing, changes no frozen file and never touches the test budget, so you can try several values and rerun the cell each time without Run all. Its default, 0.2, reproduces the 20% rows above.

<details><summary>Worked answer: a larger review budget</summary>

A larger budget reaches further down each ranked list, into customers the scorer rated less likely to buy, so Precision@budget usually falls while recall rises. RFM's advantage over R often shrinks at large budgets, because both lists then include most of the repeat buyers. If your table shows something else, report what it shows: the saved predictions are the evidence, not this expectation.
</details>""")
    code(
        "EXPLORE_BUDGET = 0.2  # @param {type:\"slider\", min:0.05, max:0.5, step:0.05}\ncommand([PY,ROOT/'customer_runtime.py','--root',ROOT,'--explore-budget',EXPLORE_BUDGET],'explore_budget.log')"
    )
    md("""### 5. Evaluate later periods

**Predict:** Will the development pattern hold in September and November? Each cutoff is reported separately and receives equal weight in the macro average. The primary contrast is Mitra-RFM minus Logistic-RFM Precision@20%: a comparator claim under the fixed recipes of section 3.

The 2,000-replicate paired bootstrap samples **customers**, retaining their snapshots and repeated draws. Intervals are conditional on these dates and frozen systems. Two test dates do not establish temporal robustness or transfer to another business. Invalid replicates are counted explicitly.""")
    code(
        "stage('evaluate')\ndisplay(Image(filename=str(RESULTS/'test_comparison.png')))\ntest,paired=load('test_metrics.json'),load('paired_intervals.json')\nmetric_table(test)\nshow('test_metrics.json')\ninterval_table(paired)\nshow('paired_intervals.json')\ndiagnostics=pd.read_csv(RESULTS/'customer_diagnostics.csv', dtype={'customer_id':str})\ntable(diagnostics.groupby(['cutoff','outcome_group']).head(2))\nprint()\nprint(*test_answer(test, paired), sep='\\n')"
    )
    md("""<details><summary>Worked answer: did the development pattern hold, and which scorer won?</summary>

Read the **This run's answer** lines above, computed from this run's test metrics and paired intervals. They answer two different questions.

- **RFM versus R within each model.** An interval above 0 means the extra history features kept helping on the later dates, not only on development.
- **Mitra-RFM versus Logistic-RFM, the primary contrast.** Check whether the 95% interval includes 0. If it does, the result is a tie: these dates cannot tell the two scorers apart at a 20% budget, so neither model won. A difference near 0 with an interval on both sides of 0 is the clearest case. A tie does not show that the models are equivalent in general, because both ran fixed, untuned recipes on one retailer and two dates.

The interval resamples customers, so it reflects which customers happened to be scored. It says nothing about other businesses or other dates.
</details>

**What to notice:** selected negatives, missed positives, differences across dates, recurring versus unseen-in-support customers and the paired interval. Do not tune settings from these outcomes. Unknown pretraining overlap and single-retailer history limit every system's claim.

### 6. Score a new historical snapshot and reconstruct the scorer

The December 1, 2011 demonstration uses earlier purchases only. The source does not contain its full next 30 days, so evaluation is **not measurable**. Mitra-RFM is the predefined example scorer whether or not it wins; the logistic reference is exported too.

Safe JSON and numeric arrays preserve schema, preprocessing, labels and support. A fresh process reconstructs the histories and scores, checks hashes, and requires probability parity (atol 1e−5, rtol 1e−4), exact predicted classes and exact top 20% IDs. Raw transactions and pretrained weights are excluded from the results ZIP. Pseudonymous dataset IDs are not a promise of anonymity for business records.""")
    code(
        "stage('infer')\nstage('verify')\nshow('verification.json', expanded=True)\nreview=pd.read_csv(RESULTS/'inference_review_list.csv', dtype={'customer_id':str})\nprint(f\"Mitra-RFM review list: top {int(review.selected_at_budget.sum())} of {len(review)} customers selected at the 20% budget (score, then the frozen hash tie rule). Outcomes are not measurable.\")\ntable(review.head(10)[['rank','customer_id','score','selected_at_budget','evaluation_status']])"
    )
    md("""### 7. Conclude with evidence

The next cell writes the evidence bundle and then prints **this run's answer** to the opening question: whether purchase history helps, which scorer to use, the evidence level and what the run does not establish. The answer is computed from this run's test results, so it stays consistent with the tables above if the numbers ever change.""")
    code(
        "(ROOT/'wall_resources.json').write_text(json.dumps({'elapsed_seconds_including_setup_before_report':time.monotonic()-STARTED,'initial_free_disk_bytes':INITIAL_FREE_DISK,'gpu_name':gpu.strip()}))\nstage('report')\nsummary=load('run_summary.json')\nprint('Mode:', summary['mode'])\nprint('Limitations:', *summary['limitations'], sep='\\n- ')\nshow('run_summary.json')\nprint('Wall minutes including setup:',round((time.monotonic()-STARTED)/60,2))\nprint('Evidence:',RESULTS/'results.zip')\nprint()\nprint(*conclusion(load('test_metrics.json'), load('paired_intervals.json')), sep='\\n')\nDOWNLOAD_RESULTS = False  # @param {type:\"boolean\"}\nif DOWNLOAD_RESULTS:\n    from google.colab import files\n    files.download(str(RESULTS/'results.zip'))"
    )
    md("""**Optional reflection.** Before opening the worked answer, write your own version: “For ___ customers at ___ test dates, Mitra-RFM Precision@20% was ___ versus ___ for Logistic-RFM. The paired difference and interval were ___. The important failure or uncertainty was ___. This does not establish ___.” **Completion records are optional and are not required submissions.**

<details><summary>Worked answer: the completed reflection</summary>

Fill each blank from this run's printed output, not from memory: the customer counts and dates from the test table in section 5, both Precision@20% values from its `P@budget` column (macro means are in `test_metrics.json`), and the difference and interval from the primary-contrast line. A strong answer names the claim type (a comparator claim under fixed recipes), says "tie" if the interval includes 0, names one uncertainty (for example, only two test dates, or customers who recur between support and test) and lists at least one thing the run cannot show, such as campaign uplift. Then compare it with **This run's answer** above.
</details>

**Status: Candidate.** The notebook stays Candidate until the maintainer reviews hosted evidence for this exact revision, including the bring-your-own-data journeys. Local CPU tests are not hosted model execution evidence.""")
    md("""### 8. Optional bring your own transactions

Off during the default Run all. Use only records you are authorized to process; they stay in this runtime.

1. **Prepare a CSV.** `invoice_time` must be a timezone-naive ISO date or date-time (`2011-04-02` or `2011-04-02 12:00:00`); missing or other formats are refused with the row and value so you can correct them. Only empty fields count as missing: a literal `NA` or `001` stays an identifier. CSV columns: `invoice_id,stock_code,quantity,invoice_time,unit_price,customer_id` and optional `country`. Do not include names, emails, phone numbers or payment details. Unexpected columns are refused.
2. **Prepare a configuration JSON**, for example:
```json
{"rights_confirmed": true, "complete_observation_coverage": true,
 "source_clock": "timezone-naive", "currency": "GBP",
 "coverage_start": "2010-01-01", "coverage_end": "2011-12-31",
 "train_cutoffs": ["2010-04-01", "2010-07-01", "2010-10-01", "2011-01-01"],
 "development_cutoffs": ["2011-04-01", "2011-07-01"],
 "test_cutoffs": ["2011-09-01", "2011-11-01"],
 "inference_cutoff": "2011-12-01"}
```
   Cutoffs are midnight dates in increasing order. Coverage must start at least 90 days before the first cutoff and end at least 30 days after the last scored cutoff, and each training cutoff's 30-day labels must mature before the first development cutoff. V1 refuses currencies other than GBP and incompatible policies rather than silently transferring a model.
3. **Upload.** With Colab's file browser (the folder icon on the left), upload them to `/content`, the folder it opens in. The default `BYOD_CSV` and `BYOD_CONFIG` point there; change them in the form if your files have other names. Uploaded files disappear when the runtime ends.
4. **Run.** In the next cell, tick `RUN_BYOD`, choose `BYOD_MODE` and run that cell only. Run all is not needed again.

**Evaluation mode** fits/conditions a separate business model and runs the same eight stages on your data. Each scored cutoff needs at least 20 repeat buyers and 20 non-buyers among the selected customers. **Artifact-inference mode** scores your customers at `inference_cutoff` with the Mitra-RFM scorer from an earlier evaluation run of this same notebook version, requires only history coverage, and exports unlabelled scores; it never silently substitutes the tutorial model. Set `BYOD_ARTIFACT` to the folder made by extracting that run's `results.zip`, which holds `checksums.json` and `scorer/mitra_RFM/`. For this session's default run, for example: `!unzip -q -o /content/dimer_customer_analytics/results/results.zip -d /content/my_prior_results`.

**If an input is refused**, the cell prints one `Refused:` line naming the file, the column or field, and the rule it broke, and shows no results; nothing from the refused attempt is kept. Fix the file, upload it again and rerun the cell. Each successful run writes a new `byod_*` folder; the cell prints the one that belongs to this run.

Local surrogate IDs replace original customer IDs in the shareable bundle; the private mapping stays outside it. The embedded reconstruction guide (`RECONSTRUCT.md`) gives the full contract.""")
    code(
        "RUN_BYOD = False  # @param {type:\"boolean\"}\nBYOD_MODE = 'evaluate'  # @param [\"evaluate\", \"artifact-inference\"]\nBYOD_CSV = '/content/my_transactions.csv'  # @param {type:\"string\"}\nBYOD_CONFIG = '/content/my_configuration.json'  # @param {type:\"string\"}\nBYOD_ARTIFACT = '/content/my_prior_results'  # @param {type:\"string\"}\nif not RUN_BYOD:\n    print('BYOD is off: tick RUN_BYOD in the form to run it.')\nelse:\n    existing=set(ROOT.glob('byod_*'))\n    returncode=command([PY,ROOT/'customer_byod.py','--root',ROOT,'--csv',BYOD_CSV,'--config',BYOD_CONFIG,'--mode',BYOD_MODE,'--artifact',BYOD_ARTIFACT],'byod.log',refusal_ok=True)\n    if returncode==REFUSED_EXIT:\n        print('No BYOD results were produced by this attempt. Fix the file named in the Refused: line above, upload it again and rerun this cell.')\n    else:\n        (run,)=set(ROOT.glob('byod_*'))-existing\n        out=run/'results'\n        print('This run:', run)\n        print('Intake:', load('data_manifest.json', run)['intake'])\n        if BYOD_MODE=='evaluate':\n            metric_table(load('test_metrics.json', out))\n            review=pd.read_csv(out/'inference_review_list.csv', dtype={'customer_id':str})\n        else:\n            review=pd.read_csv(out/'predictions.csv', dtype={'customer_id':str}).sort_values('score', ascending=False)\n        table(review.head(10))\n        print('Shareable BYOD bundle:', out/'results.zip')"
    )
    md("""### Troubleshooting and glossary

| Problem | Action |
|---|---|
| Wrong accelerator / insufficient memory | Use the documented fresh T4. Do not silently shrink cohorts after seeing scores. |
| Download or dependency failure | Check the failing log and network, then rerun from the failed stage. Hash changes require review. |
| Stale receipt or modified output | Rerun the changed stage and every downstream stage. Do not bypass the check. |
| Incomplete coverage / class gate failure | Revise the proposed design before modelling; do not relabel censored customers as negatives. |
| Reload mismatch | Preserve evidence and stop. Do not widen tolerance. |
| BYOD input refused (`Refused:` in the BYOD cell) | Read the line: it names the file, the column or field, and the rule. Nothing was scored. Fix the file, upload it again and rerun the BYOD cell; Run all is not needed. |

**Cutoff:** the time at which a prediction is made. **Support:** labelled historical examples used to fit/condition a system. **Censoring:** the full outcome is not observable. **Held-out:** outcomes reserved for final evaluation. **Cluster bootstrap:** resampling customer groups rather than pretending their repeated rows are independent. **Calibration:** agreement between probabilities and observed frequencies.

### AI Assistance Disclosure
Code, instructional text and design documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for implementation review, result validation and release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.

### References

[1] X. Zhang, D. C. Maddix, J. Yin, N. Erickson, A. F. Ansari, B. Han, S. Zhang, L. Akoglu, C. Faloutsos, M. Mahoney, T. Hu, H. Rangwala, G. Karypis, and Y. Wang, "Mitra: Mixed synthetic priors for enhancing tabular foundation models," arXiv:2510.21204, 2025, doi: 10.48550/arXiv.2510.21204. [Online]. Available: https://doi.org/10.48550/arXiv.2510.21204

[2] AutoGluon, "autogluon/mitra-classifier," Hugging Face model card, revision c425e9fa0910a6be1c494321792e7ba2a1367b1a, Apache-2.0. [Online]. Available: https://huggingface.co/autogluon/mitra-classifier/tree/c425e9fa0910a6be1c494321792e7ba2a1367b1a

[3] D. Chen, "Online Retail II," UCI Machine Learning Repository, 2019, doi: 10.24432/C5CG6D. [Online]. Available: https://doi.org/10.24432/C5CG6D

[4] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html""")
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
                "notebook_kind": "CAPSTONE",
                "notebook_spec": "2.5",
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
