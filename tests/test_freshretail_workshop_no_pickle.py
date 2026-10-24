"""The FreshRetailNet workshop notebooks (v1 and v2) save their TabICLv2 condition without pickle (ARTIFACT_SPEC 2.0
SER7), using the rice runner's `save_tabicl_context` / `load_tabicl_context` carried verbatim, and the release
validator refuses any pickle export or read in their cells. CPU only: no torch, no model, no network."""

# ruff: noqa: E501  (long assertion messages and fixtures are kept on one line)
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "v1": REPO / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb",
    "v2": REPO / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb",
}
RICE_RUNNER = REPO / "tools" / "rice_workshop" / "fm_runner.py"
CARRIED = ("TABICL_CONTEXT_FORMAT", "TABICL_CONTEXT_FILES", "sha256_file", "save_tabicl_context", "load_tabicl_context")
RUNNER_CELL, BASELINE_CELL, HELPERS_CELL, TEST_CELL = "CUjAQzh9FXEl", "32NWYtJvFXEg", "xUUykd5OFXEc", "S8SYDvNeFXEn"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load("validate_release_assets_freshretail", REPO / "tools" / "validate_release_assets.py")


def _notebook(edition: str) -> dict:
    return json.loads(NOTEBOOKS[edition].read_text(encoding="utf-8"))


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell["metadata"].get("id") == cell_id)


def _runner_source(notebook: dict) -> str:
    tree = ast.parse("".join(_cell(notebook, RUNNER_CELL)["source"]))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RUNNER_SOURCE")
    return ast.literal_eval(node.value)


def _segments(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else getattr(getattr(node, "targets", [None])[0], "id", None)
        if name in CARRIED:
            found[name] = ast.get_source_segment(source, node)
    return found


def _runner(edition: str):
    module = types.ModuleType(f"freshretail_runner_{edition}")
    exec(compile(_runner_source(_notebook(edition)), module.__name__, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("edition", sorted(NOTEBOOKS))
def test_the_runner_carries_the_rice_tabicl_helpers_verbatim(edition: str) -> None:
    carried = _segments(_runner_source(_notebook(edition)))
    rice = _segments(RICE_RUNNER.read_text(encoding="utf-8"))
    assert sorted(rice) == sorted(CARRIED)
    assert carried == rice, "the notebook copy of the TabICL artifact helpers drifted from tools/rice_workshop/fm_runner.py"


class _FakeEstimator:
    """In-context fake: probabilities depend on the conditioned rows, so a lost or altered context changes them."""

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y).astype(str)
        self.classes_ = np.asarray(sorted(set(labels.tolist())), dtype=object)
        self.centres = np.stack([values[labels == label].mean(axis=0) for label in self.classes_])
        return self

    def predict_proba(self, X):
        distance = ((np.asarray(X, dtype=float)[:, None, :] - self.centres[None]) ** 2).sum(axis=2)
        weights = np.exp(-distance)
        return weights / weights.sum(axis=1, keepdims=True)


def _fake_api(monkeypatch, built: list) -> None:
    class Pipe:
        def __init__(self, estimator, *, model_path, n_estimators, random_state, device, source="local-snapshot"):
            self.estimator, self.model_path, self.source = estimator, Path(model_path), source
            built.append({"model_path": Path(model_path), "source": source, "n_estimators": n_estimators, "random_state": random_state})

        @classmethod
        def from_pretrained(cls, weights_dir=None, *, allow_download=False, n_estimators=8, random_state=42, device=None):
            assert allow_download is False, "a reload must never download"
            return cls(_FakeEstimator(), model_path=Path(weights_dir) / "base.ckpt", n_estimators=n_estimators, random_state=random_state, device=device)

        @property
        def classes_(self):
            return list(self.estimator.classes_)

        def fit(self, X, y):
            self.estimator.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

    api = types.ModuleType("tabicl_classifier_pipeline.api")
    api.TabICLClassificationPipeline = Pipe
    api.create_classifier = lambda **kwargs: _FakeEstimator()
    package = types.ModuleType("tabicl_classifier_pipeline")
    package.api = api
    monkeypatch.setitem(sys.modules, "tabicl_classifier_pipeline", package)
    monkeypatch.setitem(sys.modules, "tabicl_classifier_pipeline.api", api)


def _support(rows: int = 60):
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"lag_7": rng.normal(size=rows), "roll_7_mean": rng.normal(size=rows), "month": rng.integers(4, 7, size=rows)})
    y = pd.Series(np.asarray(["low", "mid", "high"], dtype=object)[rng.integers(0, 3, size=rows)], name="target")
    return x, y


def _config(tmp_path: Path, condition: str = "in_context") -> dict:
    return {"condition": condition, "seed": 42, "model_configuration": {"n_estimators": 4}, "weights_dir": str(tmp_path / "weights")}


@pytest.mark.parametrize("edition", sorted(NOTEBOOKS))
def test_the_tabicl_condition_saves_as_arrays_and_json_and_reloads_exactly(edition: str, tmp_path: Path, monkeypatch) -> None:
    runner = _runner(edition)
    built: list = []
    _fake_api(monkeypatch, built)
    x, y = _support()
    config = _config(tmp_path)
    pipe = sys.modules["tabicl_classifier_pipeline.api"].TabICLClassificationPipeline.from_pretrained(weights_dir=tmp_path / "weights").fit(x, y)
    out = tmp_path / "run"
    runner.save_tabicl_artifact(out, pipe, x, y, {}, config)
    saved = out / "artifact" / "tabicl"
    assert sorted(p.name for p in saved.iterdir()) == ["context_X.npy", "context_y.npy", "model_source.json", "tabicl.json"]
    for name in ("context_X.npy", "context_y.npy"):
        np.load(saved / name, allow_pickle=False)  # plain arrays: they load with pickle disabled
    assert json.loads((saved / "model_source.json").read_text()) == {"checkpoint": None, "condition": "in_context", "sha256": None}
    query = pd.DataFrame({"lag_7": np.linspace(-2, 2, 9), "roll_7_mean": np.linspace(1, -1, 9), "month": [4, 5, 6] * 3})
    restored, encoders = runner.restore_tabicl(out, config, "cpu")
    assert encoders == {} and restored.classes_ == ["high", "low", "mid"]
    assert np.array_equal(restored.predict_proba(query), pipe.predict_proba(query))
    assert runner.tabicl_reload_parity(out, pipe, query, config, "cpu") == 0.0


@pytest.mark.parametrize("edition", sorted(NOTEBOOKS))
def test_a_tampered_tabicl_artifact_or_checkpoint_is_refused(edition: str, tmp_path: Path, monkeypatch) -> None:
    runner = _runner(edition)
    built: list = []
    _fake_api(monkeypatch, built)
    x, y = _support()
    out = tmp_path / "run"
    checkpoint = out / "model_work" / "finetune" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fine-tuned weights")
    api = sys.modules["tabicl_classifier_pipeline.api"]
    pipe = api.TabICLClassificationPipeline(_FakeEstimator(), model_path=checkpoint, n_estimators=4, random_state=42, device="cpu", source="fine-tuned").fit(x, y)
    config = _config(tmp_path, "fine_tuned")
    runner.save_tabicl_artifact(out, pipe, x, y, {}, config)
    source = json.loads((out / "artifact" / "tabicl" / "model_source.json").read_text())
    assert source["checkpoint"] == "model_work/finetune/best.ckpt" and len(source["sha256"]) == 64
    runner.restore_tabicl(out, config, "cpu")
    assert built[-1]["model_path"] == checkpoint.resolve() and built[-1]["source"] == "fine-tuned"

    with pytest.raises(ValueError, match="saved for 'fine_tuned'"):
        runner.restore_tabicl(out, _config(tmp_path), "cpu")
    checkpoint.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="does not match the digest in model_source.json"):
        runner.restore_tabicl(out, config, "cpu")
    checkpoint.write_bytes(b"fine-tuned weights")
    np.save(out / "artifact" / "tabicl" / "context_X.npy", np.zeros((60, 3)), allow_pickle=False)
    with pytest.raises(ValueError, match="does not match the digest in its sidecar"):
        runner.restore_tabicl(out, config, "cpu")


# Main's forms of the cells this change rewrote (03884d7): the validator must refuse each of them.
LEGACY_FORMS = {
    "helpers import": (HELPERS_CELL, "import os\nimport platform\n", "import os\nimport pickle\nimport platform\n"),
    "baseline save": (BASELINE_CELL, '    write_json(run_dir / "artifact.json", baseline_artifact_record(artifact))\n', '    with (run_dir / "artifact.pkl").open("wb") as handle:\n        pickle.dump(artifact, handle)\n'),
    "baseline reload": (TEST_CELL, '    artifact = json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))\n', '    with (run_dir / "artifact.pkl").open("rb") as handle:\n        artifact = pickle.load(handle)\n'),
    "runner reload": (RUNNER_CELL, "    '        pipe, encoders = restore_tabicl(source, config, device)\\n'\n", "    '        restored = load_pickle(source / \"artifact\" / \"tabicl.pkl\")\\n'\n    '        pipe, encoders = restored[\"pipe\"], restored[\"encoders\"]\\n'\n    'def load_pickle(path):\\n'\n    '    with path.open(\"rb\") as handle:\\n'\n    '        return pickle.load(handle)\\n'\n"),
}


def _validate(edition: str, notebook: dict) -> None:
    path = NOTEBOOKS[edition]
    validator._validate_auxiliary_workshop(path, notebook, validator.AUXILIARY_NOTEBOOKS[path.name], (REPO / "tutorials" / "README.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("edition", sorted(NOTEBOOKS))
def test_the_committed_notebooks_pass_the_pickle_ban(edition: str) -> None:
    _validate(edition, _notebook(edition))


@pytest.mark.parametrize("form", sorted(LEGACY_FORMS))
@pytest.mark.parametrize("edition", sorted(NOTEBOOKS))
def test_the_validator_refuses_main_s_pickle_cells(edition: str, form: str) -> None:
    notebook = _notebook(edition)
    cell_id, current, legacy = LEGACY_FORMS[form]
    cell = _cell(notebook, cell_id)
    source = "".join(cell["source"])
    assert source.count(current) == 1, form
    cell["source"] = source.replace(current, legacy).splitlines(keepends=True)
    with pytest.raises(validator.ValidationError, match=r"must not export or read a pickle artifact \(SER7\)"):
        _validate(edition, notebook)
