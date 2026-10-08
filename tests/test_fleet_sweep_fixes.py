"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['mitra_classifier_colab', 'mitra_classifier_predictor_inference_colab']
LOCK = ROOT / 'tutorials/requirements-colab-isolated.lock.txt'
MIN_PREDICT = {'mitra_classifier_colab': 5, 'mitra_classifier_predictor_inference_colab': 3}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


# --- repository-specific: SWP-G recorded facts, SWP-B BYOD path helper, inference-notebook upload guard ---------------

import os  # noqa: E402

MAIN, INFERENCE = NOTEBOOKS
RECORD = ROOT / "docs" / "release-verification.md"
KIND = "classifier"


def test_swp_g_checkpoint_answers_quote_only_recorded_facts():
    """The recorded Kaggle T4 run kept stages and sizes, not metric values: the answers quote those facts and no metric number."""
    record = RECORD.read_text(encoding="utf-8")
    markdown = "\n".join(c["source"] for c in _nb(MAIN)["cells"] if c["cell_type"] == "markdown")
    checks = re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", markdown, re.S)
    assert len(checks) == 5
    facts = ("302,717,904", "train=341, holdout=114, test=114", "215.5 s") if KIND == "classifier" else ("302,683,140", "159.4 s")
    for fact in facts:
        assert fact in record, fact
    if KIND == "classifier":
        assert "341" in checks[0] and "114" in checks[0] and "0.88 %" in checks[2]
    else:
        assert "442" in checks[0] and "recorded" in checks[0]
    # No invented metric: no accuracy/MAE value is asserted as the run's result.
    assert not re.search(r"(scored|reached|measured) 0\.\d{2,}", " ".join(checks))
    inference = "\n".join(c["source"] for c in _nb(INFERENCE)["cells"] if c["cell_type"] == "markdown")
    assert len(re.findall(r"<details><summary>Check your reasoning</summary>", inference)) == 3
    assert "No hosted run of this companion is recorded yet" in inference


def _byod_payloads(nb: dict):
    source = _cell(nb, "def byod_payloads(")
    helper = source[source.index("def byod_payloads(") : source.index("test_data = None")]
    ns = {"os": os, "Path": Path}
    exec(helper, ns)
    return ns["byod_payloads"]


def test_swp_b_byod_path_reads_a_csv_or_a_presplit_directory_and_refusals_name_the_file(tmp_path, monkeypatch):
    helper = _byod_payloads(_nb(MAIN))
    csv = tmp_path / "mine.csv"
    csv.write_bytes(b"a,target\n1,2\n")
    assert helper(str(csv)) == {"mine.csv": b"a,target\n1,2\n"}
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        helper(str(tmp_path / "missing.csv"))
    bad = tmp_path / "wrong.xlsx"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="wrong.xlsx: expected a labelled CSV"):
        helper(str(bad))
    folder = tmp_path / "split"
    folder.mkdir()
    (folder / "train.csv").write_bytes(b"t")
    with pytest.raises(FileNotFoundError, match=r"missing \['test.csv', 'val.csv'\]"):
        helper(str(folder), ("train.csv", "val.csv", "test.csv"))
    (folder / "val.csv").write_bytes(b"v")
    (folder / "test.csv").write_bytes(b"s")
    assert helper(str(folder), ("train.csv", "val.csv", "test.csv")) == {"train.csv": b"t", "val.csv": b"v", "test.csv": b"s"}
    with pytest.raises(FileNotFoundError, match="must be a directory"):
        helper(str(csv), ("train.csv", "val.csv", "test.csv"))
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match="only in Google Colab"):
        helper("")


def test_swp_b_cancelled_or_partial_uploads_are_named_and_a_good_upload_works(monkeypatch):
    helper = _byod_payloads(_nb(MAIN))
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="Upload exactly one labelled CSV"):
        helper("")
    with pytest.raises(RuntimeError, match=r"received nothing; a cancelled dialog sends none.*Missing: \['test.csv', 'train.csv', 'val.csv'\]"):
        helper("", ("train.csv", "val.csv", "test.csv"))
    _colab(monkeypatch, lambda: {"train.csv": b"t"})
    with pytest.raises(RuntimeError, match=r"Missing: \['test.csv', 'val.csv'\]"):
        helper("", ("train.csv", "val.csv", "test.csv"))
    _colab(monkeypatch, lambda: {"Train.CSV": b"t", "val.csv": b"v", "test.csv": b"s"})
    assert helper("", ("train.csv", "val.csv", "test.csv")) == {"train.csv": b"t", "val.csv": b"v", "test.csv": b"s"}
    _colab(monkeypatch, lambda: {"rows.csv": b"r"})
    assert helper("") == {"rows.csv": b"r"}


def test_swp_b_byod_gate_is_off_by_default_and_has_a_path_field():
    code = "\n".join(c["source"] for c in _code_cells(_nb(MAIN)))
    assert "USE_BYOD = False  # @param" in code and "BYOD_PATH = ''  # @param" in code
    assert "data_name, payload = csvs[0]" not in code


def test_swp_b_inference_notebook_names_the_missing_dialog_instead_of_a_name_error():
    """The companion's Section 6 relied on Section 4's conditional import of google.colab.files (a NameError when
    ARTIFACT_ZIP_PATH was set but NEW_DATA_PATH was empty); it now imports inside a guard with a named message."""
    s6 = _cell(_nb(INFERENCE), "NEW_DATA_PATH = ''  # @param")
    branch = s6[s6.index("else:") : s6.index("new_upload = files.upload()")]
    assert "from google.colab import files" in branch and "except ImportError" in branch
    assert "the upload dialog exists only in Google Colab" in branch
