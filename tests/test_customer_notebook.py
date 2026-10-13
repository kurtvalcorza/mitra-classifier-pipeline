"""The distributable notebook must match its embedded, standalone source."""
import ast
import importlib
import json
import re
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
builder = importlib.import_module("build_customer_capstone")


def test_notebook_schema_embedded_sources_and_generated_parity():
    notebook = builder.build()
    nbformat.validate(nbformat.from_dict(notebook))
    assert json.loads(builder.NOTEBOOK.read_text(encoding="utf-8")) == notebook
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for name, content in builder.carried_files().items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    assert notebook["metadata"]["dimer"]["standalone"] is True


def test_hash_lock_has_pins_and_distribution_hashes():
    lock = builder.carried_files()["requirements.txt"]
    assert "--hash=sha256:" in lock
    for entry in lock.replace("\\\n", " ").splitlines():
        if entry and not entry.startswith(("#", " ", "--")):
            assert "==" in entry and "--hash=sha256:" in entry


def test_notebook_has_no_long_lines_and_carrier_round_trips():
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        for line in cell["source"]:
            assert len(line) <= 2000, (cell["id"], len(line))
    samples = {"empty": "", "long": "x" * 2500 + "\n", "multi": "a\r\nb\n\nc'\"\\é d", "tail": "no newline"}
    for value in (samples, *samples.values()):
        literal = builder.carried_literal(value)
        assert ast.literal_eval(literal) == value
        assert max(map(len, literal.splitlines())) <= 2000


def _markdown(notebook):
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _markdown_cells(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"]


def _code_cell(notebook, marker):
    (cell,) = [
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and marker in "".join(c["source"])
        and not "".join(c["source"]).startswith("FILES")
    ]
    return cell


def test_write_up_layer_kind_identity_and_references():
    notebook = builder.build()
    dimer = notebook["metadata"]["dimer"]
    assert (dimer["notebook_kind"], dimer["notebook_spec"]) == ("CAPSTONE", "2.5")
    opening = "".join(notebook["cells"][0]["source"])
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "> **⚠️ Notice.**" in opening and "### Problem statement" in opening
    assert "Data you bring stays in your runtime" in opening and "await hosted validation" not in opening
    assert "measured at 10.4 minutes" in opening and "not re-run here" in opening
    body, references = _markdown(notebook).split("### References")
    assert "autogluon/mitra-classifier" in body and "c425e9fa0910a6be1c494321792e7ba2a1367b1a" in body
    assert "at least one qualifying order" in body and "monotone" in body and "fair" not in body
    entries = re.findall(r"^\[(\d+)\] .*https://", references, re.M)
    assert entries == ["1", "2", "3", "4"]
    assert set(re.findall(r"\[(\d)\]", body)) == set(entries)  # DOC16: cited one to one


def test_byod_cell_uses_form_fields_and_routes_refusals_without_a_traceback():
    notebook = builder.build()
    cell = _code_cell(notebook, "'--mode',BYOD_MODE")
    for name in ("RUN_BYOD", "BYOD_MODE", "BYOD_CSV", "BYOD_CONFIG", "BYOD_ARTIFACT"):
        assert re.search(rf"^{name} = .*# @param", cell, re.M), name
    assert re.search(r"^RUN_BYOD = False  # @param", cell, re.M)
    assert "refusal_ok=True" in cell and "returncode==REFUSED_EXIT" in cell and "rerun this cell" in cell
    assert "Do not skip this stage" not in json.dumps(notebook)
    bootstrap = _code_cell(notebook, "def command(")
    assert "if code==REFUSED_EXIT and refusal_ok:" in bootstrap and "raise RuntimeError" in bootstrap
    section8 = next(m for m in _markdown_cells(notebook) if m.startswith("### 8."))
    assert "upload them to `/content`" in section8 and "results.zip" in section8 and "Refused:" in section8


def test_learner_change_is_a_form_field_and_timeline_rounds_money():
    notebook = builder.build()
    explore = _code_cell(notebook, "--explore-budget")
    assert re.search(r'^EXPLORE_BUDGET = 0\.2  # @param \{type:"slider"', explore, re.M)
    assert "already ran both conditions" in _markdown(notebook)
    timeline = _code_cell(notebook, "teaching_timeline.csv")
    assert ".round(2).map('{:.2f}'.format)" in timeline and "round(v,2)" in timeline


def _report(values):
    """Synthetic per-cutoff metrics: values[system] = (P@budget, AP, AUROC) at each of two cutoffs."""
    rows = []
    for system, per in values.items():
        for cutoff, (p, ap, auc) in zip(("2011-09-01", "2011-11-01"), per, strict=True):
            rows.append(
                dict(
                    system=system, cutoff=cutoff, n=1000, precision_at_budget=p, average_precision=ap,
                    auroc=auc,
                )
            )
    return {"per_cutoff": rows, "macro": {s: {"lift_at_budget": 1.8} for s in values}, "budget": 0.2}


BASE = {
    "prevalence": [(0.4, 0.4, 0.5)] * 2,
    "recency": [(0.5, 0.5, 0.6), (0.52, 0.5, 0.6)],
    "logistic_R": [(0.5, 0.5, 0.6), (0.52, 0.5, 0.6)],
    "mitra_R": [(0.51, 0.5, 0.6), (0.5, 0.5, 0.6)],
    "logistic_RFM": [(0.7, 0.6, 0.7), (0.75, 0.6, 0.7)],
    "mitra_RFM": [(0.7, 0.6, 0.7), (0.76, 0.6, 0.7)],
}


def _contrast(difference, ci):
    return {"difference": difference, "ci95": ci, "measure": "equal_cutoff_macro_precision_at_budget"}


def _paired(primary, rfm=(0.18, 0.28)):
    return {
        "contrasts": {
            "mitra_RFM_minus_logistic_RFM": _contrast(0.0 if primary is None else sum(primary) / 2, primary),
            "mitra_RFM_minus_recency": _contrast(0.22, list(rfm)),
            "mitra_RFM_minus_mitra_R": _contrast(0.22, list(rfm)),
            "logistic_RFM_minus_logistic_R": _contrast(0.21, list(rfm)),
        }
    }


def test_this_runs_answer_follows_the_results_not_fixed_text():
    answers = {}
    exec(builder.ANSWERS, answers)
    dev = answers["development_answer"]({"canonical_budget_20": _report(BASE)})
    assert "yes, RFM beat R for both models" in dev[0] and "4 of 4" in dev[1] and "identical" in dev[2]
    mixed = dict(BASE, mitra_RFM=[(0.4, 0.6, 0.7), (0.76, 0.6, 0.7)], logistic_R=[(0.45, 0.5, 0.6)] * 2)
    dev = answers["development_answer"]({"canonical_budget_20": _report(mixed)})
    assert "not consistently" in dev[0] and "rank customers differently" in dev[2]
    test = _report(BASE)
    tie = answers["test_answer"](test, _paired([-0.015, 0.010]))
    assert "pattern held" in tie[0] and "interval includes 0" in tie[3] and tie[4].startswith("A tie")
    assert answers["test_answer"](test, _paired([0.01, 0.05]))[4].startswith("Mitra-RFM ranked better")
    assert answers["test_answer"](test, _paired([-0.05, -0.01]))[4].startswith("Logistic-RFM ranked better")
    assert answers["test_answer"](test, _paired(None))[4].startswith("No interval")
    final = answers["conclusion"](test, _paired([-0.015, 0.010]))
    assert final[1].startswith("1. Yes") and final[2].startswith("2. Model choice: a tie")
    assert "1,000 on 2011-09-01" in final[3] and final[4].startswith("4. Not established")
    weak = answers["conclusion"](test, _paired([-0.015, 0.010], rfm=(-0.02, 0.05)))
    assert weak[1].startswith("1. Partly")
