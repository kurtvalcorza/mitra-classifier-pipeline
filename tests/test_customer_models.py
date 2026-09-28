"""Offline scorer contracts; real logistic CPU fit, fake direct Mitra trainer."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import customer_models as cm  # noqa:E402


@pytest.fixture
def support():
    x = np.random.default_rng(42).uniform(1, 50, (80, 5))
    y = (x[:, 0] < 25).astype(int)
    return pd.DataFrame(x, columns=cm.FEATURE_NAMES), y


def test_train_only_scaling_and_query_independence(support):
    X, y = support
    state = cm.fit("logistic", X, y, "RFM")
    cooked = X.to_numpy().copy()
    cooked[:, 1:] = np.log1p(cooked[:, 1:])
    np.testing.assert_allclose(state["mean"], cooked.mean(axis=0))
    original = cm.predict(state, X.iloc[:3])
    altered = pd.concat([X.iloc[:3], X.iloc[:1] * 1e6], ignore_index=True)
    np.testing.assert_array_equal(cm.predict(state, altered)[:3], original)
    np.testing.assert_array_equal(state["support_X"], X.to_numpy())


def test_class_probability_reversal_and_bad_values():
    p = np.array([[0.2, 0.8], [0.9, 0.1]])
    np.testing.assert_array_equal(cm.positive_probability(p, [1, 0]), [0.2, 0.9])
    np.testing.assert_array_equal(cm.positive_probability(p, [0, 1]), [0.8, 0.1])
    for bad in ([[1.1, -0.1]], [[float("nan"), 1]], [[0.2, 0.3]]):
        with pytest.raises(ValueError):
            cm.positive_probability(bad, [0, 1])


@pytest.mark.parametrize(
    "kind,feature_set",
    [
        ("prevalence", "R"),
        ("recency", "R"),
        ("logistic", "R"),
        ("logistic", "RFM"),
        ("mitra", "R"),
        ("mitra", "RFM"),
    ],
)
def test_safe_export_reload_all_systems(support, tmp_path, kind, feature_set, monkeypatch):
    X, y = support
    state = cm.fit(kind, X, y, feature_set)
    manifest = json.loads(
        (Path(cm.__file__).parents[1] / "tutorials/customer_analytics/model_manifest.json").read_text()
    )
    if kind == "mitra":
        monkeypatch.setattr(
            cm, "mitra_predict", lambda model, support, labels, query: np.full(len(query), 0.4)
        )
    before = cm.predict(state, X, model=object())
    cm.export(state, tmp_path, manifest)
    restored = cm.reload(tmp_path)
    np.testing.assert_array_equal(cm.predict(restored, X, model=object()), before)
    np.testing.assert_array_equal(restored["support_y"], y)
    np.testing.assert_array_equal(restored["support_X"], X.to_numpy())
    assert not list(tmp_path.glob("*.pkl"))


def test_fresh_process_logistic_reconstruction(support, tmp_path):
    X, y = support
    state = cm.fit("logistic", X, y)
    cm.export(state, tmp_path)
    np.save(tmp_path / "query.npy", X.to_numpy(), allow_pickle=False)
    script = (
        "import sys,numpy as np;sys.path.insert(0,sys.argv[1]);import customer_models as m;"
        "from pathlib import Path;r=Path(sys.argv[2]);"
        "np.save(r/'pred.npy',m.predict(m.reload(r),np.load(r/'query.npy',allow_pickle=False)),allow_pickle=False)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(cm.__file__).parent), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "pred.npy", allow_pickle=False), cm.predict(state, X))


def test_schema_support_and_artifact_refusals(support, tmp_path):
    X, y = support
    with pytest.raises(ValueError, match="ordered"):
        cm.fit("logistic", X.iloc[:, ::-1], y)
    with pytest.raises(ValueError):
        cm.fit("mitra", np.ones((2049, 5)), np.arange(2049) % 2)
    state = cm.fit("logistic", X, y)
    cm.export(state, tmp_path)
    (tmp_path / "support_y.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        cm.reload(tmp_path)
    altered = copy.deepcopy(state)
    altered["classes"] = [1, 0]
    with pytest.raises(ValueError, match="mapping"):
        cm.predict(altered, X)
    altered = copy.deepcopy(state)
    altered["settings"]["seed"] = 7
    with pytest.raises(ValueError, match="settings"):
        cm.predict(altered, X)
    assert cm.SETTINGS["seed"] == 42


def test_mitra_direct_context_keeps_all_support(support, monkeypatch):
    X, y = support
    calls = {}

    class Adapter:
        def __init__(self, **kwargs):
            calls["adapter"] = kwargs

        def _create_config(self, task, outputs):
            assert task == "classification" and outputs == 10
            return SimpleNamespace(hyperparams={}), None

    class Trainer:
        def __init__(self, config, model, n_classes, device, rng, verbose):
            calls["config"] = config.hyperparams.copy()
            assert n_classes == 2
            self.preprocessor = SimpleNamespace(fit=lambda x, y: calls.update(fit_x=x.copy(), fit_y=y.copy()))

        def post_fit_optimize(self):
            pass

        def predict(self, x, y, q):
            calls["predict_x"] = x.copy()
            calls["predict_y"] = y.copy()
            return np.tile([1000.0, 1001.0], (len(q), 1))

    module = SimpleNamespace(MitraClassifier=Adapter, TrainerFinetune=Trainer)
    monkeypatch.setitem(sys.modules, "autogluon.tabular.models.mitra.sklearn_interface", module)
    monkeypatch.setattr(cm, "version", lambda package: "1.5.0")
    monkeypatch.setattr(cm, "_seed", lambda: None)
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    state = cm.fit("mitra", X, y)
    predictions = cm.predict(state, X.iloc[:5], model=model)
    assert calls["config"]["max_samples_support"] == len(X)
    assert calls["config"]["max_samples_query"] == 64
    assert calls["adapter"]["fine_tune"] is False and calls["adapter"]["n_estimators"] == 1
    np.testing.assert_array_equal(calls["fit_y"], y)
    np.testing.assert_array_equal(calls["predict_y"], y)
    np.testing.assert_array_equal(calls["fit_x"], calls["predict_x"])
    np.testing.assert_allclose(predictions, np.full(5, 1 / (1 + np.exp(-1))))


def test_changed_model_manifest_refused_before_download(tmp_path):
    manifest = json.loads(
        (Path(cm.__file__).parents[1] / "tutorials/customer_analytics/model_manifest.json").read_text()
    )
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned"):
        cm.stage_snapshot(manifest, tmp_path)


def test_actual_tiny_upstream_cpu_contract_when_available():
    """No pretrained weights: verify real upstream trainer, logits and batching."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("autogluon.tabular.models.mitra.sklearn_interface")
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    torch.manual_seed(42)
    model = Tab2D(
        dim=16,
        dim_output=10,
        n_layers=1,
        n_heads=2,
        task="CLASSIFICATION",
        use_pretrained_weights=False,
        path_to_weights="",
        device="cpu",
    ).eval()
    rng = np.random.default_rng(42)
    support = rng.uniform(0, 40, (32, 5))
    labels = np.arange(32) % 2
    query = rng.uniform(0, 40, (65, 5))
    whole = cm.mitra_predict(model, support, labels, query)
    repeat = cm.mitra_predict(model, support, labels, query)
    split = np.concatenate(
        [
            cm.mitra_predict(model, support, labels, query[:64]),
            cm.mitra_predict(model, support, labels, query[64:]),
        ]
    )
    assert whole.shape == (65,) and np.isfinite(whole).all()
    assert np.all((whole >= 0) & (whole <= 1))
    np.testing.assert_allclose(whole, repeat, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(whole, split, atol=1e-5, rtol=1e-4)
