"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on mitra_classifier_colab (MCC-M1..M3,
MCC-m1..m6; docs/reviews/2026-10-02-notebook-review/mitra_classifier_colab_Review.md) and for the isolated
environment's worker shims (NOTEBOOK_SPEC 2.3 ENV15/ENV16).

CI's dependencies only: the notebook's own cell sources are executed with stand-ins; no AutoGluon, torch or network.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import logging
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
ART = ROOT / "tutorials" / "mitra_classifier_predictor_inference_colab.ipynb"


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path: Path, marker: str) -> str:
    found = [c["source"] for c in _cells(path) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(path: Path) -> str:
    return "\n".join(c["source"] for c in _cells(path) if c["cell_type"] == "markdown")


def _section6() -> str:
    return _code(E2E, "EVAL_METRIC = ")


def _stdlib_imports(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if re.match(r"^import (contextlib|gc|logging|math|shutil|warnings)$", line))


# --- MCC-M1: the fine-tuning selection no longer raises NameError: math -------------------------------------------------


def test_mcc_m1_selection_block_runs_with_a_candidate_and_a_large_holdout():
    source = _section6()
    assert re.search(r"^import math$", source, re.M), "Section 6 must import math before math.isfinite is used"
    block = source[source.index("def better(") : source.index("active_metrics = ")]
    pretrained = {"accuracy": 0.95, "log_loss": 0.20}
    candidate = {"accuracy": 0.97, "log_loss": 0.25}
    ns = {"candidate": object(), "pipe": object(), "candidate_metrics": candidate, "pretrained_metrics": pretrained,
          "candidate_test_metrics": dict(candidate), "pretrained_test_metrics": dict(pretrained),
          "holdout_data": list(range(114)), "MIN_SELECTION_HOLDOUT_ROWS": 50, "EVAL_METRIC": "accuracy"}
    exec(_stdlib_imports(source) + "\n" + block, ns)  # raised NameError: name 'math' is not defined on the reviewed blob
    assert ns["SELECTION_BASIS"] == "holdout:accuracy" and ns["ACTIVE_MODE"] == "fine-tuned"
    assert ns["degraded"] == ["log_loss"]


def test_mcc_m6_gpu_gate_runs_before_any_fit_and_wine_floor_is_documented():
    source = _section6()
    gate = source.index("if RUN_FINE_TUNING and not torch.cuda.is_available():")
    assert gate < source.index("pipe.fit(") and gate < source.index("candidate.fit(")
    closing = _markdown(E2E)
    assert "holdout-too-small:36<50" in closing and "*Run* — Sections 4–7" in closing
    for experiment in ("re-run Sections 6–9", "re-run Sections 4–9"):
        assert experiment in closing, experiment
    assert "**Activity (about 3 minutes on CPU)" in closing and "*Predict*" in closing and "*Explain*" in closing


# --- MCC-M2: Mitra's in-context support is printed, recorded and explained ---------------------------------------------


def test_mcc_m2_context_rows_are_printed_recorded_and_the_prose_is_truthful():
    source = _section6()
    markdown = _markdown(E2E)
    assert "exact same support rows" not in markdown and "(EVAL15)" not in markdown
    assert "`Validation score` line of its log is measured on those internal rows, **not** on your holdout" in markdown
    assert "'mitra_context_rows': MITRA_CONTEXT_ROWS" in source and "'autogluon_internal_validation_rows'" in source
    export = _code(E2E, "archive_base = ")
    assert "'mitra_context_rows': context_rows(ACTIVE_MODEL)" in export and "'support_rows': len(train_data)" in export
    helper = source[source.index("def context_rows(") : source.index("with autogluon_log():")]
    ns: dict = {}
    exec(helper, ns)

    class _Predictor:
        def load_data_internal(self, data="train", return_X=True, return_y=True):
            assert data == "train" and return_y is False
            return list(range(272)), None

    assert ns["context_rows"](types.SimpleNamespace(predictor=_Predictor())) == 272
    assert ns["context_rows"](types.SimpleNamespace(predictor=None)) is None  # reported, never hidden


# --- MCC-m2: shortened AutoGluon log, renormalised probabilities, stated positive class --------------------------------


def test_mcc_m2_autogluon_log_summary_keeps_only_split_score_and_warning_lines(monkeypatch):
    source = _section6()
    assert "AUTOGLUON_LOG = 'summary'  # @param" in source
    start, end = source.index("class _AutoGluonSummary("), source.index("def context_rows(")
    stub = types.ModuleType("autogluon.tabular")
    monkeypatch.setitem(sys.modules, "autogluon", types.ModuleType("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.tabular", stub)
    ns = {"logging": logging, "contextlib": contextlib, "AUTOGLUON_LOG": "summary"}
    exec(source[start:end], ns)
    logger = logging.getLogger("autogluon")
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    try:
        with ns["autogluon_log"]():
            child = logging.getLogger("autogluon.tabular.learner")
            child.setLevel(logging.INFO)
            child.info("Presets specified: ['extreme']  Massively better than 'best'")
            child.info("Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 272, Val Rows: 69")
            child.info("\t0.9565\t = Validation score   (accuracy)")
            child.warning("Not enough memory to train Mitra")
        assert records == ["Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 272, Val Rows: 69", "\t0.9565\t = Validation score   (accuracy)", "Not enough memory to train Mitra"]
        assert not handler.filters  # removed after the fit
    finally:
        logger.removeHandler(handler)
    assert "values = values / values.sum(axis=1, keepdims=True)" in source
    assert "warnings.filterwarnings('ignore', message='.*do not sum to one.*')" in source
    assert "'autogluon_positive_class': pipe.predictor.positive_class" in source


def test_mcc_m3_rows_correct_and_what_to_notice_follow_the_comparison():
    source = _section6()
    assert "metrics_table.insert(2, 'rows_correct'" in source
    markdown = _markdown(E2E)
    notice = markdown[markdown.index("**What to notice.** (1) Convert") :]
    assert "`rows_correct`" in notice and "holdout ranking with the test ranking" in notice
    assert "The executable baselines show when the foundation model adds value" not in markdown


# --- MCC-m4: archives and a wrong TARGET_COLUMN are refused in Section 4 with the fix named ----------------------------


def _section4_helpers():
    source = _code(E2E, "def byod_payloads(")
    helper = source[source.index("def byod_payloads(") : source.index("test_data = None")]
    ns = {"os": os, "Path": Path, "TARGET_COLUMN": "target"}
    exec(helper, ns)
    return ns


def test_mcc_m4_archives_are_refused_with_an_extract_instruction(tmp_path, monkeypatch):
    ns = _section4_helpers()
    archive = tmp_path / "telco-customer-churn.zip"
    archive.write_bytes(b"PK")
    with pytest.raises(ValueError, match="is an archive: extract it"):
        ns["byod_payloads"](str(archive))
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"telco-customer-churn.zip": b"PK"})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(RuntimeError, match="Extract the archive and upload its CSV"):
        ns["byod_payloads"]("", ("train.csv", "val.csv", "test.csv"))


def test_mcc_m4_wrong_target_column_is_named_in_section_4():
    import pandas as pd

    ns = _section4_helpers()
    frame = pd.DataFrame({"tenure": [1], "Churn": ["No"]})
    with pytest.raises(ValueError, match=r"TARGET_COLUMN 'target' is not a column of train.csv. Set TARGET_COLUMN in this cell \(Section 4\).*'Churn'"):
        ns["require_target"](frame, "train.csv")
    ns["TARGET_COLUMN"] = "Churn"
    assert ns["require_target"](frame, "train.csv") is frame
    markdown = _markdown(E2E)
    assert "`Churn` for Telco Churn and `class` for Adult Census" in markdown and "extract a ZIP first" in markdown


# --- MCC-M3 / MCC-m1: guided layer, short opening, no reviewer-facing text ----------------------------------------------


def test_mcc_m1_short_learner_opening_and_no_open_decision_text():
    cells = _cells(E2E)
    assert cells[0]["cell_type"] == "markdown" and len(cells[0]["source"]) <= 1800
    assert cells[1]["source"].startswith("<details><summary><b>About this notebook</b>") and "**Run all:**" in cells[1]["source"]
    markdown = _markdown(E2E)
    assert "open decision" not in markdown and "a reviewer reading" not in markdown


def test_mcc_m3_guided_elements_and_memory_troubleshooting_are_present():
    markdown = _markdown(E2E)
    for element in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Input → Model → Output.**", "**Roadmap:**",
                    "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "**What to notice.**", "**Activity"):
        assert element in markdown, element
    assert markdown.count("**Predict:**") >= 5 and markdown.count("<details><summary>Check your reasoning</summary>") >= 5
    trouble = markdown[markdown.index("## Troubleshooting") :]
    assert "No models were trained successfully" in trouble and "`MAX_MEMORY_USAGE_RATIO`" in trouble
    carrier = [c for c in _cells(E2E) if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
    assert carrier and all(c["metadata"].get("cellView") == "form" for c in carrier)


def test_rel13_quoted_counts_are_labelled_with_their_source_revision():
    markdown = _markdown(E2E)
    assert "previous notebook revision (blob `9599d5c`)" in markdown
    assert "the recorded hosted run of the previous version needed one" not in markdown


# --- MCC-S1 / spec declaration -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [E2E, ART])
def test_generated_notebooks_declare_notebook_spec_2_2(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone**" in _markdown(path)


# --- ENV15 / ENV16: the isolated worker's google.colab shims are well-formed modules ------------------------------------


@pytest.mark.parametrize("path", [E2E, ART])
@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(path: Path, real_google: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    router = _code(path, '_WORKER_SOURCE = r"""')
    worker = router[router.index('_WORKER_SOURCE = r"""') + len('_WORKER_SOURCE = r"""') :]
    worker = worker[: worker.index('"""')]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
        for n in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(n)  # raised ValueError: google.colab.__spec__ is None before the fix
            assert spec is not None and spec.name == n
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def test_section_6_cell_parses():
    ast.parse(_section6())
