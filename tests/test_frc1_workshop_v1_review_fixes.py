"""Acceptance checks for the FRC1 Notebook Review Framework v1 findings on the v1 FreshRetailNet workshop.

The review (docs/reviews/2026-10-03-notebook-review/) found five majors and seven minors in the compact v1
edition. Each test executes the notebook's own cell code (not a transcription) against small synthetic inputs,
with only the heavyweight notebook state stubbed. They are logic checks, not model runs or Colab execution
evidence. LightGBM is not in the CI install budget, so a scikit-learn stand-in plays its role where a cell
fits one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import shutil
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
NOTEBOOK = TUTORIALS / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb"
GUIDED = TUTORIALS / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb"


def _cells(path: Path) -> dict[str, str]:
    return {
        cell["metadata"]["id"]: "".join(cell["source"])
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
    }


CELLS = _cells(NOTEBOOK)
CLASSES = ["low", "mid", "high"]


def _helpers() -> dict:
    """Every function defined in Section 0.1, executed from the notebook source."""
    tree = ast.parse(CELLS["xUUykd5OFXEc"])
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "pd": pd,
        "Path": Path,
        "TARGET_COLUMN": "target",
        "SEMANTIC_CLASS_ORDER": CLASSES,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "log_loss": log_loss,
        "matthews_corrcoef": matthews_corrcoef,
        "roc_auc_score": roc_auc_score,
        "label_binarize": label_binarize,
    }
    exec(compile(module, "section-0.1-helpers", "exec"), namespace)
    return namespace


def _runner() -> dict:
    node = next(
        n
        for n in ast.parse(CELLS["CUjAQzh9FXEl"]).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "RUNNER_SOURCE"
    )
    namespace: dict = {"__name__": "runner_under_test"}
    exec(compile(ast.literal_eval(node.value), "runner", "exec"), namespace)
    return namespace


def _run_cell(cell_id: str, namespace: dict, replacements: dict[str, str] | None = None) -> dict:
    source = CELLS[cell_id]
    for old, new in (replacements or {}).items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, cell_id, "exec"), namespace)
    return namespace


class _Anything:
    """Absorbs matplotlib calls in cells whose figures are not under test."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return (self, self) if kwargs.get("figsize") else self


class _BoostedTreesStandIn(RandomForestClassifier):
    """Accepts LGBMClassifier's arguments so the notebook's LightGBM cells run without LightGBM."""

    def __init__(self, n_estimators=300, random_state=None, verbosity=None):
        super().__init__(n_estimators=10, random_state=random_state)
        self.verbosity = verbosity


def _quiet(**extra) -> dict:
    return {"display": lambda *a, **k: None, **extra}


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "lag_7": rng.gamma(2.0, 0.4, n),
            "roll_7_mean": rng.gamma(2.0, 0.4, n),
            "stockout_hours": rng.integers(0, 8, n).astype(float),
        }
    )
    score = frame["roll_7_mean"] + 0.3 * rng.normal(size=n)
    frame["target"] = np.asarray(CLASSES, dtype=object)[
        np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    ]
    return frame


# --- FRC1-M2: acquisition and the baseline ladder are retry-safe --------------------------------------------


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def _acquire(namespace: dict, archive: Path) -> dict:
    return _run_cell(
        "ZDCzBnsvFXEe",
        namespace,
        {
            "USE_BYOD = False # @param": "USE_BYOD = True # @param",
            'BYOD_ZIP_PATH = "" # @param': f'BYOD_ZIP_PATH = "{archive.as_posix()}" # @param',
        },
    )


@pytest.fixture
def acquisition(tmp_path: Path) -> dict:
    session = tmp_path / "session"
    (session / "data").mkdir(parents=True)
    namespace = _helpers()
    namespace.update(
        _quiet(),
        DATA_ROOT=session / "data",
        SESSION_ROOT=session,
        shutil=shutil,
        zipfile=zipfile,
        urllib=urllib,
        uuid=uuid,
        PurePosixPath=PurePosixPath,
    )
    return namespace


SPLITS = {"train.csv": "a,target\n1,low\n", "val.csv": "a,target\n2,mid\n", "test.csv": "a,target\n3,high\n"}


def test_invalid_archive_then_correction_and_repeat_in_one_session(acquisition: dict, tmp_path: Path) -> None:
    broken = _zip(tmp_path / "broken.zip", {k: v for k, v in SPLITS.items() if k != "test.csv"})
    with pytest.raises(ValueError, match="missing required split files"):
        _acquire(acquisition, broken)
    assert not list(acquisition["DATA_ROOT"].glob(".staging-*")), "a failed attempt must leave nothing behind"
    good = _zip(tmp_path / "good.zip", SPLITS)
    _acquire(acquisition, good)  # was FileExistsError at 2.1.0
    _acquire(acquisition, good)  # a repeat acquisition also works
    assert sorted(acquisition["staged_paths"]) == ["test", "train", "val"]
    assert acquisition["staged_paths"]["test"].read_text() == SPLITS["test.csv"]


def test_switching_datasets_clears_results_and_the_freeze(acquisition: dict, tmp_path: Path) -> None:
    first = _zip(tmp_path / "a.zip", SPLITS)
    second = _zip(tmp_path / "b.zip", {k: v.replace("1,low", "9,low") for k, v in SPLITS.items()})
    _acquire(acquisition, first)
    acquisition.update(
        TEST_RESULTS="stale", FOUNDATION_RUN_DIRS={"x": 1}, baseline_validation_results="stale"
    )
    freeze = acquisition["SESSION_ROOT"] / "freeze.json"
    freeze.write_text("{}")
    _acquire(acquisition, first)  # same data: results are kept
    assert acquisition["TEST_RESULTS"] == "stale" and freeze.exists()
    _acquire(acquisition, second)  # different data: every downstream result is invalidated
    for name in ("TEST_RESULTS", "FOUNDATION_RUN_DIRS", "baseline_validation_results"):
        assert name not in acquisition
    assert not freeze.exists()


def _baseline_namespace(tmp_path: Path) -> dict:
    namespace = _helpers()
    train, val = _frame(240, 1), _frame(120, 2)
    features = ["lag_7", "roll_7_mean", "stockout_hours"]
    namespace.update(
        _quiet(),
        RUN_ROOT=tmp_path,
        shutil=shutil,
        pickle=pickle,
        time=time,
        frames={"train": train, "val": val},
        X_train=train[features],
        y_train=train["target"].astype(str),
        X_val=val[features],
        y_val=val["target"].astype(str),
        CLASS_LABELS=CLASSES,
        FEATURE_COLUMNS=features,
        DATASET_FINGERPRINT="d",
        RANDOM_SEED=42,
        PRIMARY_METRIC="balanced_accuracy",
        Pipeline=Pipeline,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        RandomForestClassifier=RandomForestClassifier,
        LGBMClassifier=_BoostedTreesStandIn,
    )
    return namespace


def test_baseline_ladder_can_be_rerun(tmp_path: Path) -> None:
    namespace = _run_cell("32NWYtJvFXEg", _baseline_namespace(tmp_path))
    first = namespace["baseline_validation_results"].copy()
    _run_cell("32NWYtJvFXEg", namespace)  # was FileExistsError, and the registry was emptied (5 -> 0)
    assert len(namespace["baseline_prediction_registry"]) == 5
    volatile = ["runtime_seconds", "artifact_sha256"]
    pd.testing.assert_frame_equal(
        first.drop(columns=volatile), namespace["baseline_validation_results"].drop(columns=volatile)
    )


# --- FRC1-M5: each objective has an activity that runs ------------------------------------------------------


def test_alignment_activity_shows_the_cost_of_misaligned_columns(tmp_path: Path) -> None:
    namespace = _run_cell("32NWYtJvFXEg", _baseline_namespace(tmp_path))
    _run_cell("frc1a7c3e0b2", namespace)
    demo = namespace["alignment_demo"]
    aligned, misaligned = demo.iloc[0], demo.iloc[1]
    assert namespace["raw_order"] == ["high", "low", "mid"]
    assert misaligned["balanced_accuracy"] < aligned["balanced_accuracy"]
    assert misaligned["log_loss"] > aligned["log_loss"]


def test_band_edge_activity_uses_only_development_rows() -> None:
    namespace = _quiet(
        np=np, pd=pd, CLASS_LABELS=CLASSES, frames={"train": _frame(300, 3), "val": _frame(200, 4)}
    )
    namespace["frames"]["val"]["roll_7_mean"] *= 1.4  # a later period with higher sales
    _run_cell("frc1b4d2a1c2", namespace)
    leaky = namespace["edge_demo"]["edges recomputed on validation rows (leaky)"]
    built = namespace["edge_demo"]["edges from training rows (as built)"]
    assert np.allclose(leaky, 1 / 3, atol=0.02) and built["high"] > 0.4
    assert "test" not in CELLS["frc1b4d2a1c2"]


def test_adaptation_mode_and_band_distance_activities(tmp_path: Path) -> None:
    run_dir = tmp_path / "tabicl"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "model": "TabICLv2",
                "condition": "in_context",
                "effective_mode": "in_context",
                "metrics": {"balanced_accuracy": 0.6},
            }
        )
    )
    namespace = _quiet(json=json, pd=pd, FOUNDATION_RUN_DIRS={"tabiclv2_icl": run_dir})
    _run_cell("frc1d6f4c3e2", namespace)
    row = namespace["ADAPTATION_MODES_TABLE"].iloc[0]
    assert row["effective_mode"] == "in_context" and not row["weights_updated"]

    diagnostic = pd.DataFrame(
        {"target": ["low", "low", "mid", "high"], "prediction": ["low", "high", "low", "mid"]}
    )
    namespace = _quiet(pd=pd, CLASS_LABELS=CLASSES, diagnostic=diagnostic, MODEL_TO_INSPECT="m")
    _run_cell("frc1e7a5d4f2", namespace)
    assert namespace["error_profile"]["rows"].tolist() == [1, 2, 1]


def test_every_objective_names_its_activity() -> None:
    objectives = CELLS["baY-MBeeFXEa"]
    for section in ("Section 6.2", "Section 3.2", "Section 5.6", "Section 4.2"):
        assert section in objectives
    for cell_id in ("frc1e7a5d4f1", "frc1b4d2a1c1", "frc1d6f4c3e1", "frc1a7c3e0b1"):
        assert "**Predict:**" in CELLS[cell_id]


# --- FRC1-M1: an ablation keeps the adaptation mode; the runner accepts only exact modes --------------------


def test_fine_tuned_ablation_keeps_the_fine_tuned_mode(tmp_path: Path) -> None:
    namespace = _helpers()
    persisted = {}

    def stream_command(command, log_path):
        config = json.loads(Path(command[-1]).read_text())
        out = Path(config["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps(config))
        persisted.update(config)

    namespace.update(
        _quiet(),
        RUN_ROOT=tmp_path,
        SUPPORT_ROOT=tmp_path,
        FORCE_MODEL_RERUN=False,
        stream_command=stream_command,
        CONDITION_SPECS={
            "tabiclv2_finetuned": {
                "environment": "tabicl",
                "display_name": "TabICLv2",
                "condition": "fine_tuned",
                "configuration": {"fine_tune_epochs": 10},
            }
        },
        MODEL_ENVIRONMENT_PYTHONS={"tabicl": Path("python")},
        MODEL_REPOSITORIES={"tabicl": {"repository": "r", "commit": "c"}},
        MODEL_WEIGHTS_DIRS={"tabicl": tmp_path},
        FEATURE_COLUMNS=["a", "b"],
        CLASS_LABELS=CLASSES,
        RANDOM_SEED=42,
        DEVICE_PREFERENCE="auto",
        DATASET_FINGERPRINT="d",
        RUNNER_SHA256="r",
        RUNNER_PATH=Path("runner.py"),
        staged_paths={},
        selected_foundation_conditions=[],
        REQUIRE_ALL_SELECTED_MODELS=True,
        pd=pd,
        shutil=shutil,
    )
    _run_cell("auqhsq19FXEl", namespace)
    namespace["run_foundation_condition"](
        "tabiclv2_finetuned", features=["a"], output_key="ablation", ablation_group="no_stockout"
    )
    assert persisted["condition"] == "fine_tuned" and persisted["ablation_group"] == "no_stockout"

    with pytest.raises(ValueError, match="Unknown adaptation mode 'fine_tuned_no_stockout'"):
        _runner()["execute"]({"condition": "fine_tuned_no_stockout", "output_dir": str(tmp_path / "never")})
    assert not (tmp_path / "never").exists()


def test_a_mode_mismatch_produces_no_ablation_row(tmp_path: Path) -> None:
    full_dir, ablated_dir = tmp_path / "full", tmp_path / "ablated"
    for directory, mode in ((full_dir, "fine_tuned"), (ablated_dir, "in_context")):
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "model": "TabICLv2",
                    "condition": "fine_tuned",
                    "effective_mode": mode,
                    "metrics": {"balanced_accuracy": 0.5},
                }
            )
        )
    namespace = _quiet(
        json=json,
        pd=pd,
        FEATURE_COLUMNS=["a", "stockout_hours"],
        PRIMARY_METRIC="balanced_accuracy",
        FOUNDATION_RUN_DIRS={"tabiclv2_finetuned": full_dir},
        CONDITION_SPECS={"tabiclv2_finetuned": {"condition": "fine_tuned"}},
        foundation_validation_results=pd.DataFrame(
            {"model_key": ["tabiclv2_finetuned"], "balanced_accuracy": [0.5]}
        ),
        run_foundation_condition=lambda *a, **k: ablated_dir,
    )
    _run_cell(
        "UKzaZmJxFXEm",
        namespace,
        {
            "RUN_CLASSICAL_ABLATION = True # @param": "RUN_CLASSICAL_ABLATION = False # @param",
            "RUN_FOUNDATION_ABLATION = False # @param": "RUN_FOUNDATION_ABLATION = True # @param",
        },
    )
    assert namespace["ablation_validation_results"].empty


# --- probe finding carried with the v2 port: negative probabilities are rejected ----------------------------


@pytest.mark.parametrize(
    "align", [lambda: _helpers()["align_probabilities"], lambda: _runner()["align_probabilities"]]
)
def test_negative_probabilities_are_rejected(align) -> None:
    with pytest.raises(ValueError, match="[Nn]egative"):
        align()([[-0.2, 0.6, 0.6]], CLASSES, CLASSES)


# --- FRC1-M3: the frozen test uses the frozen configuration, and a seen test blocks a silent refreeze -------


def _foundation_freeze(tmp_path: Path, namespace: dict) -> tuple[Path, dict]:
    run_dir = tmp_path / "dev" / "tabdpt_icl"
    (run_dir / "artifact").mkdir(parents=True)
    (run_dir / "artifact" / "weights.bin").write_bytes(b"w")
    development = {
        "phase": "development",
        "model_key": "tabdpt_icl",
        "base_condition_key": "tabdpt_icl",
        "model_family": "tabdpt",
        "display_name": "TabDPT",
        "condition": "in_context",
        "ablation_group": None,
        "model_configuration": {"n_ensembles": 4, "context_size": 4180, "batch_size": 4096},
        "repository": "r",
        "repository_commit": "c",
        "features": ["a"],
        "class_labels": CLASSES,
        "seed": 42,
        "device_preference": "auto",
        "dataset_fingerprint": "d",
        "runner_sha256": "r",
        "weights_dir": str(tmp_path),
        "split_paths": {"train": "old"},
        "output_dir": str(run_dir),
        "run_fingerprint": "f",
    }
    (run_dir / "run_config.json").write_text(json.dumps(development))
    (run_dir / "result.json").write_text("{}")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "d",
                "selected": {
                    "tabdpt_icl": {
                        "backend": "foundation",
                        "run_dir": str(run_dir),
                        "environment": "tabdpt",
                        "result_sha256": namespace["sha256_file"](run_dir / "result.json"),
                        "run_config_sha256": namespace["sha256_file"](run_dir / "run_config.json"),
                        "artifact_inventory": namespace["file_inventory"](run_dir / "artifact"),
                    }
                },
            }
        )
    )
    return freeze, development


def _fake_test_run(command, log_path):
    config = json.loads(Path(command[-1]).read_text())
    out = Path(config["output_dir"])
    (out / "run_config.json").write_text(json.dumps(config))
    metrics = {
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "f1_macro": 0.5,
        "mcc": 0.1,
        "log_loss": 1.0,
        "roc_auc_ovr_macro": 0.6,
    }
    (out / "result.json").write_text(
        json.dumps({"model": "TabDPT", "condition": "in_context", "metrics": metrics, "device": "cpu"})
    )
    (out / "test_predictions.csv").write_text("row_id,target,prediction\n0,low,low\n")


def test_frozen_test_ignores_a_live_configuration_change(tmp_path: Path) -> None:
    namespace = _helpers()
    freeze, development = _foundation_freeze(tmp_path, namespace)
    namespace.update(
        _quiet(),
        FREEZE_PATH=freeze,
        DATASET_FINGERPRINT="d",
        RUNNER_SHA256="r",
        RUN_ROOT=tmp_path,
        SUPPORT_ROOT=tmp_path,
        CONDITION_SPECS={
            "tabdpt_icl": {"configuration": {"n_ensembles": 8, "context_size": 4180, "batch_size": 4096}}
        },
        MODEL_ENVIRONMENT_PYTHONS={"tabdpt": Path("python")},
        RUNNER_PATH=Path("runner.py"),
        stream_command=_fake_test_run,
        staged_paths={"test": tmp_path / "test.csv"},
        PRIMARY_METRIC="balanced_accuracy",
        pickle=pickle,
        BASELINE_ROOT=tmp_path,
        Pipeline=object,
    )
    _run_cell("S8SYDvNeFXEn", namespace)
    executed = namespace["TEST_EFFECTIVE_CONFIGS"]["tabdpt_icl"]
    assert executed["model_configuration"]["n_ensembles"] == 4  # the frozen value, not the live 8
    assert executed["frozen_run_fingerprint"] == development["run_fingerprint"]


def _freeze_namespace(tmp_path: Path) -> dict:
    baseline_root = tmp_path / "baselines"
    run_dir = baseline_root / "random_forest"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"balanced_accuracy": 0.5}))
    (run_dir / "artifact.pkl").write_bytes(b"artifact")
    namespace = _helpers()
    namespace.update(
        _quiet(),
        SESSION_ROOT=tmp_path,
        RUN_ROOT=tmp_path,
        BASELINE_ROOT=baseline_root,
        baseline_validation_results=pd.DataFrame({"model_key": ["random_forest"]}),
        FOUNDATION_RUN_DIRS={},
        WORKSHOP_REVISION="2.1.1",
        DATASET_IDENTITY={},
        DATASET_FINGERPRINT="d",
        PRIMARY_METRIC="balanced_accuracy",
        CLASS_LABELS=CLASSES,
        FEATURE_COLUMNS=["a"],
    )
    return namespace


def test_refreeze_after_the_test_is_refused_unless_recorded(tmp_path: Path) -> None:
    namespace = _run_cell("aNhMl_zxFXEn", _freeze_namespace(tmp_path))
    _run_cell("aNhMl_zxFXEn", namespace)  # refreezing before any test score is seen is allowed
    first = json.loads((tmp_path / "freeze.json").read_text())
    assert first["previous_freezes"][0]["test_scores_seen"] is False

    namespace["TEST_RESULTS"] = pd.DataFrame({"model_key": ["random_forest"]})  # Section 8.1 has run
    before = (tmp_path / "freeze.json").read_bytes()
    with pytest.raises(RuntimeError, match="already scored on the test partition"):
        _run_cell("aNhMl_zxFXEn", namespace)
    assert (tmp_path / "freeze.json").read_bytes() == before

    _run_cell(
        "aNhMl_zxFXEn",
        namespace,
        {'REFREEZE_REASON = "" # @param': 'REFREEZE_REASON = "added a model after a bug fix" # @param'},
    )
    record = json.loads((tmp_path / "freeze.json").read_text())["previous_freezes"][-1]
    assert (
        record["test_scores_seen"] is True and record["replaced_because"] == "added a model after a bug fix"
    )
    assert record["freeze_sha256"] == hashlib.sha256(before).hexdigest()
    assert namespace["TEST_RESULTS"] is None  # the replaced freeze's test results cannot be reported


# --- FRC1-m1: test balance and per-class test recall only after the freeze ---------------------------------


def test_test_class_balance_and_recall_after_the_freeze() -> None:
    truth = ["low", "mid", "high", "high"]
    namespace = _quiet(
        pd=pd,
        recall_score=recall_score,
        TARGET_COLUMN="target",
        CLASS_LABELS=CLASSES,
        frames={split: pd.DataFrame({"target": truth}) for split in ("train", "val", "test")},
        TEST_RESULTS=None,
        TEST_PREDICTION_REGISTRY={},
    )
    with pytest.raises(RuntimeError, match="test partition first"):
        _run_cell("frc1f8b6e5a2", namespace)
    namespace.update(
        TEST_RESULTS=pd.DataFrame({"model_key": ["m"], "model": ["Model"]}),
        TEST_PREDICTION_REGISTRY={
            "m": pd.DataFrame({"target": truth, "prediction": ["low", "mid", "mid", "high"]})
        },
    )
    _run_cell("frc1f8b6e5a2", namespace)
    assert namespace["TEST_PER_CLASS_RECALL"].loc["m", "weakest_band"] == "high"
    order = list(CELLS)
    assert order.index("frc1f8b6e5a2") > order.index("S8SYDvNeFXEn") > order.index("aNhMl_zxFXEn")


# --- FRC1-m4: model ceilings are checked before any foundation-model run ----------------------------------


def test_over_ceiling_training_split_is_rejected_before_any_model_runs() -> None:
    specs = {"mitra_icl": {"environment": "mitra"}, "tabdpt_icl": {"environment": "tabdpt"}}

    def namespace(rows: int) -> dict:
        return _quiet(
            pd=pd,
            X_train=pd.DataFrame({"a": np.zeros(rows)}),
            FEATURE_COLUMNS=["a"] * 17,
            CONDITION_SPECS=specs,
            selected_foundation_conditions=list(specs),
            TABDPT_CONTEXT_SIZE=4180,
        )

    _run_cell("frc1c5e3b2d1", namespace(4180))
    with pytest.raises(ValueError, match=r"too large for \['mitra_icl'\]"):
        _run_cell("frc1c5e3b2d1", namespace(11_000))
    order = list(CELLS)
    assert order.index("LiNZD1UYFXEh") < order.index("frc1c5e3b2d1") < order.index("nfx5Ry7iFXEh")
    contract = CELLS["_Of8pCh-FXEb"]
    for rule in ("**in that order**", "`low`, `mid` and `high`", "10,000 training rows", "finite"):
        assert rule in contract


# --- FRC1-m6: the ablation interval compares the ablated model with its own full-feature version ----------


def test_identical_full_and_ablated_predictions_give_a_zero_contrast() -> None:
    truth = np.tile(np.array(CLASSES), 50)
    weak = np.full(truth.shape, "low", dtype=object)

    def frame(pred):
        return pd.DataFrame({"row_id": range(len(truth)), "target": truth, "prediction": pred})

    names = {
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "lightgbm_no_period_proxies": "LightGBM without month and weather",
    }
    namespace = _quiet(
        np=np,
        pd=pd,
        plt=_Anything(),
        RANDOM_SEED=42,
        PRIMARY_METRIC="balanced_accuracy",
        CLASS_LABELS=CLASSES,
        TARGET_COLUMN="target",
        frames={"test": pd.DataFrame({"target": truth})},
        TEST_PREDICTION_REGISTRY={
            "random_forest": frame(truth.copy()),
            "lightgbm": frame(weak),
            "lightgbm_no_period_proxies": frame(weak),
        },
        FIGURE_ROOT=Path("."),
        ABLATION_BASELINE_KEYS=["lightgbm_no_period_proxies"],
        baseline_validation_results=pd.DataFrame(
            {"model_key": ["random_forest", "lightgbm"], "balanced_accuracy": [0.6, 0.5]}
        ),
        TEST_RESULTS=pd.DataFrame(
            {
                "model_key": list(names),
                "model": list(names.values()),
                "condition": ["trained from scratch"] * 3,
                "balanced_accuracy": [1.0, 1 / 3, 1 / 3],
            }
        ),
    )
    _run_cell("166858dd2aa4", namespace, {"BOOTSTRAP_RESAMPLES = 1000 #": "BOOTSTRAP_RESAMPLES = 200 #"})
    contrast = namespace["ABLATION_CONTRASTS"].iloc[0]
    assert contrast["comparator"] == "LightGBM (all features)"
    assert (
        contrast["difference"] == 0
        and contrast["difference_ci_low"] == 0
        and contrast["difference_ci_high"] == 0
    )
    assert "and the two LightGBM feature-group ablations" in CELLS["pMgrx9X5FXEX"]


# --- FRC1-M4, m2, m3, m5: static markers -----------------------------------------------------------------


def test_readme_names_which_notebook_to_use_and_matches_the_notebooks() -> None:
    readme = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    assert "**Which notebook to use.**" in readme and "**compact edition**" in readme
    assert "saved outputs are retained from the recorded run" not in readme
    guided = json.loads(GUIDED.read_text(encoding="utf-8"))
    assert not any(cell.get("outputs") for cell in guided["cells"] if cell["cell_type"] == "code")
    assert "**Edition:** compact edition" in CELLS["pMgrx9X5FXEX"]


def test_lag7_log_loss_is_explained_next_to_the_baseline_table() -> None:
    order = list(CELLS)
    assert order.index("frc1a7c3e0b1") == order.index("32NWYtJvFXEg") + 1
    assert "hard rule" in CELLS["frc1a7c3e0b1"] and "log-loss comparisons" in CELLS["frc1a7c3e0b1"]


def test_export_is_recomputable_and_names_this_notebook() -> None:
    export = CELLS["Da6jGcVwFXEo"]
    for needle in (
        '"notebook_identity"',
        '"comparison_definitions"',
        "TEST_EFFECTIVE_CONFIGS",
        "frozen_test_ablation_contrasts.csv",
        "EXPORT_ROW_LEVEL_PREDICTIONS",
    ):
        assert needle in export
    assert '"file": "tutorials/DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb"' in export


def test_ported_cells_stay_identical_to_the_guided_edition() -> None:
    guided = _cells(GUIDED)
    for cell_id in (
        "xUUykd5OFXEc",
        "32NWYtJvFXEg",
        "CUjAQzh9FXEl",
        "auqhsq19FXEl",
        "UKzaZmJxFXEm",
        "S8SYDvNeFXEn",
        "166858dd2aa4",
    ):
        assert CELLS[cell_id] == guided[cell_id], cell_id
    assert "HOST_OUTSIDE_TESTED_RANGE" in CELLS["xUUykd5OFXEc"]
