"""The rice pixel-tables runner saves the TabICL condition as `.npy` arrays plus a JSON sidecar, never as a pickle
(ARTIFACT_SPEC 2.0 SER7). CPU only: no torch, no model, no network; TabICL is a deterministic fake."""

# ruff: noqa: E501  (long assertion messages and fixtures are kept on one line)
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "rice_workshop" / "fm_runner.py"
SETTINGS = {"n_estimators": 4, "random_state": 42}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("rice_workshop_fm_runner_tabicl", RUNNER)
validator = _load("validate_release_assets_tabicl", REPO / "tools" / "validate_release_assets.py")


def _support(rows: int = 40):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(rows, 3)).astype(np.float32), columns=["b02", "b03", "b04"])
    y = pd.Series(rng.integers(0, 2, size=rows), name="target")
    return x, y


def test_round_trip_restores_the_support_rows_exactly(tmp_path: Path) -> None:
    x, y = _support()
    runner.save_tabicl_context(tmp_path, x, y, {}, [0, 1], SETTINGS)
    x2, y2, columns, encoders, classes = runner.load_tabicl_context(tmp_path, SETTINGS)
    assert x2.dtype == np.float32 and np.array_equal(x2, x.to_numpy())
    assert np.array_equal(y2, y.to_numpy())
    assert columns == ["b02", "b03", "b04"] and encoders == {} and classes == ["0", "1"]


def test_only_arrays_and_a_pure_json_sidecar_are_written(tmp_path: Path) -> None:
    x, y = _support()
    runner.save_tabicl_context(tmp_path, x, y, {"crop": ["maize", "rice"]}, [0, 1], SETTINGS)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["context_X.npy", "context_y.npy", "tabicl.json"]
    sidecar = json.loads((tmp_path / "tabicl.json").read_text(encoding="utf-8"))
    assert sidecar["format"] == runner.TABICL_CONTEXT_FORMAT
    assert sidecar["encoders"] == {"crop": ["maize", "rice"]} and sidecar["settings"] == SETTINGS
    assert sidecar["sha256"]["context_X.npy"] == runner.sha256_file(tmp_path / "context_X.npy")
    assert not list(tmp_path.glob("*.pkl")) and not list(tmp_path.glob("*.pt"))
    for name in ("context_X.npy", "context_y.npy"):
        np.load(tmp_path / name, allow_pickle=False)  # plain arrays: loads with pickle disabled


def test_a_digest_mismatch_is_refused_before_any_array_is_read(tmp_path: Path, monkeypatch) -> None:
    x, y = _support()
    runner.save_tabicl_context(tmp_path, x, y, {}, [0, 1], SETTINGS)
    data = bytearray((tmp_path / "context_X.npy").read_bytes())
    data[-1] ^= 0xFF
    (tmp_path / "context_X.npy").write_bytes(bytes(data))

    def never(*_args, **_kwargs):
        raise AssertionError("np.load reached before the digest check")

    monkeypatch.setattr(runner.np, "load", never)
    with pytest.raises(ValueError, match="does not match the digest"):
        runner.load_tabicl_context(tmp_path, SETTINGS)


def test_missing_sidecar_wrong_format_and_wrong_settings_are_refused(tmp_path: Path) -> None:
    x, y = _support()
    runner.save_tabicl_context(tmp_path, x, y, {}, [0, 1], SETTINGS)
    with pytest.raises(ValueError, match="settings"):
        runner.load_tabicl_context(tmp_path, {"n_estimators": 8, "random_state": 42})
    sidecar = json.loads((tmp_path / "tabicl.json").read_text(encoding="utf-8"))
    (tmp_path / "tabicl.json").write_text(json.dumps({**sidecar, "format": "pickle"}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        runner.load_tabicl_context(tmp_path, SETTINGS)
    (tmp_path / "tabicl.json").unlink()
    with pytest.raises(ValueError, match="no sidecar"):
        runner.load_tabicl_context(tmp_path, SETTINGS)


def test_non_numeric_support_rows_are_refused_rather_than_pickled(tmp_path: Path) -> None:
    x = pd.DataFrame({"crop": ["rice", "maize"]})
    with pytest.raises(TypeError, match="numeric"):
        runner.save_tabicl_context(tmp_path, x, pd.Series([0, 1]), {}, [0, 1], SETTINGS)
    assert not (tmp_path / "context_X.npy").exists()


class _FakeTabICL:
    """In-context fake: probabilities depend on the conditioned rows, so a lost context changes the output."""

    def fit(self, X, y):
        self.centre = float(np.asarray(X)[:, 0].mean())
        self.classes_ = sorted(np.unique(np.asarray(y)).tolist())
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=float)[:, 0] - self.centre)))
        return np.column_stack([1.0 - p, p])


def test_the_tabicl_condition_saves_and_reloads_without_pickle(tmp_path: Path, monkeypatch) -> None:
    api = types.ModuleType("tabicl_classifier_pipeline.api")
    api.prepare_classification_table = lambda frame, target: (frame.reset_index(drop=True), 0)
    api.fit_categorical_encoder = lambda frame, columns: {}
    api.apply_categorical_encoder = lambda frame, encoders: (frame.copy(), {})
    package = types.ModuleType("tabicl_classifier_pipeline")
    package.api = api
    monkeypatch.setitem(sys.modules, "tabicl_classifier_pipeline", package)
    monkeypatch.setitem(sys.modules, "tabicl_classifier_pipeline.api", api)
    monkeypatch.setattr(runner, "load_family", lambda family, cfg, device: _FakeTabICL())
    cfg = {"settings": {"n_estimators": 4}, "seed": 42}
    x, y = _support()
    names = list(x.columns)
    predict, save = runner.fit_family("tabicl", _FakeTabICL(), x.to_numpy(), y.to_numpy(), None, None, names, cfg,
                                      tmp_path)
    save()
    assert sorted(p.name for p in (tmp_path / "tabicl").iterdir()) == ["context_X.npy", "context_y.npy", "tabicl.json"]
    reloaded = runner.reload_family("tabicl", cfg, "cpu", tmp_path)
    query = pd.DataFrame(np.linspace(-2, 2, 15, dtype=np.float32).reshape(5, 3), columns=names)
    assert np.array_equal(reloaded(query), predict(query))


def test_the_runner_has_no_pickle_and_the_validator_rejects_one() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert validator.pickle_export_hits(source) == []
    legacy = 'import pickle\nwith open(artifact / "tabicl" / "tabicl.pkl", "wb") as handle:\n    pickle.dump(x, handle)\n'
    assert validator.pickle_export_hits(legacy) == ["import pickle", '"tabicl.pkl"', "pickle.dump("]
    assert validator.pickle_export_hits("torch.save(state, path)", validator.PICKLE_ARTIFACT_CALLS) == ["torch.save("]
    assert validator.pickle_export_hits('root / "predictor.pkl"', validator.PICKLE_ARTIFACT_CALLS) == []
