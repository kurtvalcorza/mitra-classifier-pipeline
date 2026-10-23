"""Refusal display (NOTEBOOK_SPEC 3.0 DAT19/DAT20): a refused user input is one `Refused: <file>: <rule>` line.

Offline, with CI's dependency budget (pandas, NumPy, scikit-learn, pytest). The notebook's own cells are executed
with their form fields set; a refusal must render as exactly one `Refused:` line with no traceback, name the refused
file, and stop the run (it is raised, so Run all halts at that cell). IPython renders an exception through its
`_render_traceback_` method when it has one; `_render` below applies that rule. The isolated-environment worker test
starts the real worker/kernel pair from the generator, so it needs `os.pipe` file-descriptor passing (POSIX only).
"""
# ruff: noqa: E501
from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import re
import sys
import traceback
import types
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest
from mitra_pipeline import tutorial_api as api

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
TARGET = "species"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code(marker: str) -> str:
    found = [c["source"] for c in _cells() if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, marker
    return found[0]


def _form(source: str, **values: object) -> str:
    for name, value in values.items():
        line = re.search(rf"^{name} = .*# @param.*$", source, re.M)
        assert line, name
        source = source.replace(line.group(0), f"{name} = {value!r}", 1)
    return source


def _namespace(**extra: object) -> dict:
    ns = {name: getattr(api, name) for name in dir(api) if not name.startswith("__")}
    ns.update({"__builtins__": builtins, "__name__": "notebook_cell", "os": os, "json": json, "Path": Path, "pd": pd, "zipfile": zipfile})
    ns.update(extra)
    return ns


def _render(source: str, ns: dict) -> tuple[str, bool]:
    """Run one cell as IPython would show it: printed text, then the error (its _render_traceback_ when defined)."""
    out = io.StringIO()
    halted = False
    with redirect_stdout(out):
        try:
            exec(compile(source, "<cell>", "exec"), ns)  # noqa: S102 - the notebook's own cell
        except Exception as exc:  # noqa: BLE001 - rendered exactly as the kernel would
            halted = True
            render = getattr(exc, "_render_traceback_", None)
            print("\n".join(render()) if render else traceback.format_exc())
    return out.getvalue(), halted


def _assert_one_clean_refusal(out: str, halted: bool, name: str, rule: str) -> None:
    lines = [line for line in out.splitlines() if line.strip()]
    refused = [line for line in lines if "Refused:" in line]
    assert len(refused) == 1 and refused[0].startswith(f"Refused: {name}: "), out
    assert re.search(rule, refused[0]), out
    assert "Traceback" not in out and halted, out  # no traceback, and Run all stops at this cell


def _iris(target: str = TARGET) -> pd.DataFrame:
    """The browser kit's table: scikit-learn iris with readable column names and class names as the label."""
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    frame = data.frame.rename(columns={"sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width", "petal length (cm)": "petal_length", "petal width (cm)": "petal_width"})
    frame["target"] = frame["target"].map(dict(enumerate(data.target_names)))
    return frame.rename(columns={"target": target})


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)


def _colab(monkeypatch, uploaded: dict) -> None:
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


# ---- Section 4: your own labelled data -------------------------------------------------------------------------------


def _section4(**form: object) -> str:
    values = {"DATA_SOURCE": "Upload CSV", "USE_BYOD": True, "BYOD_PATH": "", "TARGET_COLUMN": TARGET, "DROP_COLUMNS": "", "VALIDATION_SPLIT": 0.2, "SEED": 42}
    values.update(form)
    return _form(_code("def byod_payloads("), **values)


def test_the_browser_kit_refusal_is_one_line_in_section_4(tmp_path, monkeypatch) -> None:
    # The kit's refuse.csv: the iris rows with the label renamed `variety`.
    _colab(monkeypatch, {"refuse.csv": _iris("variety").to_csv(index=False).encode()})
    out, halted = _render(_section4(), _namespace())
    _assert_one_clean_refusal(out, halted, "refuse.csv", r"TARGET_COLUMN 'species' is not a column of refuse\.csv\. .*run it again")


def test_a_single_class_table_is_refused_in_section_4(tmp_path, monkeypatch) -> None:
    frame = _iris()
    frame[TARGET] = "setosa"
    path = tmp_path / "one_class.csv"
    frame.to_csv(path, index=False)
    _no_colab(monkeypatch)
    out, halted = _render(_section4(BYOD_PATH=str(path)), _namespace())
    _assert_one_clean_refusal(out, halted, "one_class.csv", r"class.*run this cell again")


@pytest.mark.parametrize(
    "case, name, rule",
    [
        ("two-uploads", "a.csv, b.csv", r"Upload exactly one labelled CSV \(received 2 files"),
        ("archive", "data.zip", "archives are not read here"),
        ("missing-path", "absent.csv", "does not exist or is not a file"),
        ("too-few-rows", "tiny.csv", "MIN_TRAIN_ROWS|at least|rows"),
        ("split-out-of-range", "VALIDATION_SPLIT", "between 0.05 and 0.40"),
    ],
)
def test_section_4_refusals_are_one_line(tmp_path, monkeypatch, case: str, name: str, rule: str) -> None:
    form: dict = {}
    if case == "two-uploads":
        _colab(monkeypatch, {"a.csv": b"x", "b.csv": b"y"})
    elif case == "archive":
        _colab(monkeypatch, {"data.zip": b"PK"})
    elif case == "missing-path":
        _no_colab(monkeypatch)
        form["BYOD_PATH"] = str(tmp_path / "absent.csv")
    elif case == "too-few-rows":
        _no_colab(monkeypatch)
        path = tmp_path / "tiny.csv"
        _iris().head(20).to_csv(path, index=False)
        form["BYOD_PATH"] = str(path)
    else:
        _colab(monkeypatch, {"accept.csv": _iris().to_csv(index=False).encode()})  # the split is checked after the upload here
        form["VALIDATION_SPLIT"] = 0.9
    out, halted = _render(_section4(**form), _namespace())
    _assert_one_clean_refusal(out, halted, name, rule)


def test_a_usable_upload_passes_section_4(monkeypatch) -> None:
    _colab(monkeypatch, {"accept.csv": _iris().to_csv(index=False).encode()})
    ns = _namespace()
    out, halted = _render(_section4(), ns)
    assert not halted and "Refused" not in out and "'train_rows': 120" in out and "'holdout_rows': 30" in out, out


# ---- Section 8: new-data rows ----------------------------------------------------------------------------------------


def test_section_8_refuses_rows_without_the_features(tmp_path, monkeypatch) -> None:
    path = tmp_path / "own_rows.csv"
    pd.DataFrame({"sepal_length": [5.0], "sepal_width": [3.0]}).to_csv(path, index=False)
    ns = _namespace(FEATURE_COLUMNS=["sepal_length", "sepal_width", "petal_length"], ACTIVE_MODEL=None)
    out, halted = _render(_form(_code("RUN_NEW_DATA_INFERENCE = False"), RUN_NEW_DATA_INFERENCE=True, NEW_DATA_PATH=str(path)), ns)
    _assert_one_clean_refusal(out, halted, "own_rows.csv", r"petal_length.*nothing was scored")


def test_section_8_refuses_two_uploads(monkeypatch) -> None:
    _colab(monkeypatch, {"one.csv": b"a\n1\n", "two.csv": b"a\n2\n"})
    out, halted = _render(_form(_code("RUN_NEW_DATA_INFERENCE = False"), RUN_NEW_DATA_INFERENCE=True, NEW_DATA_PATH=""), _namespace(FEATURE_COLUMNS=["a"], ACTIVE_MODEL=None))
    _assert_one_clean_refusal(out, halted, "one.csv, two.csv", "Upload exactly one inference CSV")


# ---- Section 10: your own artifact ZIP and your own rows -------------------------------------------------------------


def _section10(**form: object) -> str:
    return _form(_code("ARTIFACT_SOURCE = 'Pinned sample artifact'  # @param"), **form)


@pytest.mark.parametrize(
    "case, rule",
    [
        ("no-digest", "A trusted whole-archive SHA-256 is required"),
        ("wrong-digest", "Trusted digest mismatch"),
        ("not-a-zip", "not a predictor bundle this notebook can use"),
    ],
)
def test_section_10_refuses_an_unusable_own_artifact(tmp_path, monkeypatch, case: str, rule: str) -> None:
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "my_bundle.zip"
    archive.write_bytes(b"this is not a zip archive")
    form = {"ARTIFACT_SOURCE": "My own artifact ZIP", "ARTIFACT_ZIP_PATH": str(archive), "EXPECTED_ZIP_SHA256": "", "ALLOW_UNVERIFIED_ARTIFACT": False}
    if case == "wrong-digest":
        form["EXPECTED_ZIP_SHA256"] = "0" * 64
    elif case == "not-a-zip":
        form["ALLOW_UNVERIFIED_ARTIFACT"] = True
    out, halted = _render(_section10(**form), _namespace())
    _assert_one_clean_refusal(out, halted, "own-my_bundle.zip", rule)


def test_section_10_refuses_own_rows_without_a_feature(tmp_path, monkeypatch) -> None:
    path = tmp_path / "own_rows.csv"
    pd.DataFrame({"sepal_length": [5.0, 6.0]}).to_csv(path, index=False)
    ns = _namespace(EXT_FEATURES=["sepal_length", "petal_width"], EXT_TARGET=TARGET, EXT_CLASSES=["setosa", "versicolor"], ext_metadata={"data_source": "Upload CSV"}, ext_zip=Path("own-my_bundle.zip"))
    source = _form(_code("USE_OWN_ROWS = False  # @param"), USE_OWN_ROWS=True, EXTERNAL_ROWS_PATH=str(path))
    out, halted = _render(source, ns)
    _assert_one_clean_refusal(out, halted, "own_rows.csv", r"petal_width.*nothing was scored")


def test_genuine_failures_keep_their_traceback(monkeypatch) -> None:
    # The sample rows (not a user input) failing a check is a genuine error: it is not presented as a refusal.
    ns = _namespace(EXT_FEATURES=["mean radius"], EXT_TARGET=TARGET, EXT_CLASSES=["setosa", "versicolor"], ext_metadata={"data_source": "Upload CSV"}, ext_zip=Path("pinned.zip"))
    source = _form(_code("USE_OWN_ROWS = False  # @param"), USE_OWN_ROWS=False, EXTERNAL_ROWS_PATH="")
    ns["ext_metadata"] = {"data_source": "Sample: Breast Cancer", "data_sha256": "0" * 64, "seed": 42}
    out, halted = _render(source, ns)
    assert halted and "Traceback" in out and "Refused:" not in out, out


# ---- The isolated-environment worker passes a refusal back as its one line -------------------------------------------


def _router() -> dict:
    spec = importlib.util.spec_from_file_location("refusal_build_notebook", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module._ISOLATED_ROUTER.split("\n\nif SKIP_INSTALL:")[0]
    import signal
    import subprocess
    from multiprocessing.connection import Connection

    ns = {"__builtins__": builtins, "__name__": "router", "os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection}
    exec(compile(source, "<router>", "exec"), ns)  # noqa: S102 - the generator's own router source
    return ns


@pytest.mark.skipif(os.name != "posix", reason="the worker needs os.pipe file-descriptor passing (POSIX), as on Colab")
def test_the_worker_sends_a_refusal_as_one_line_and_a_genuine_error_with_its_traceback(capfd) -> None:
    router = _router()
    runtime = router["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        refusal = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom mitra_pipeline.tutorial_api import Refused\nraise Refused('refuse.csv', 'TARGET_COLUMN is not a column. Fix it and run this cell again.')\n"
        with pytest.raises(router["InputRefused"]) as caught:
            runtime.run(refusal)
        assert caught.value._render_traceback_() == ["Refused: refuse.csv: TARGET_COLUMN is not a column. Fix it and run this cell again."]
        streamed = capfd.readouterr()
        assert "Traceback" not in streamed.out + streamed.err
        with pytest.raises(router["IsolatedCellError"]):
            runtime.run("raise ValueError('a genuine bug')\n")
        assert "Traceback" in capfd.readouterr().err  # genuine failures still show their traceback
    finally:
        runtime.close()
