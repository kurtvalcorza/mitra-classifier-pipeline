"""Acceptance checks for the Notebook Review Framework v1 findings on the FreshRetailNet v2 workshop.

Each test executes the notebook's own cell code (not a transcription) against small synthetic inputs, with
only the heavyweight notebook state stubbed. These are logic checks, not model runs or Colab execution
evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import shutil
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb"
CELLS = {
    cell["metadata"]["id"]: "".join(cell["source"])
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    if "id" in cell.get("metadata", {})
}


def _helpers() -> dict:
    """The pure helper functions from Section 0.1, executed from the notebook source."""
    tree = ast.parse(CELLS["xUUykd5OFXEc"])
    wanted = {
        "sha256_file",
        "canonical_json",
        "fingerprint",
        "write_json",
        "file_inventory",
        "verify_file_inventory",
        "align_probabilities",
        "prediction_frame",
    }
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[]
    )
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "pd": pd,
        "Path": Path,
        "TARGET_COLUMN": "target",
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


# --- NR-03: acquisition is retry-safe and a dataset change invalidates stale state ------------------------


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
            'BYOD_METHOD = "upload" # @param': 'BYOD_METHOD = "path" # @param',
            'BYOD_ZIP_PATH = "" # @param': f'BYOD_ZIP_PATH = "{archive}" # @param',
        },
    )


@pytest.fixture
def acquisition(tmp_path: Path) -> dict:
    session = tmp_path / "session"
    (session / "data").mkdir(parents=True)
    namespace = _helpers()
    namespace.update(
        DATA_ROOT=session / "data",
        SESSION_ROOT=session,
        display=lambda *a, **k: None,
        shutil=shutil,
        zipfile=zipfile,
        urllib=urllib,
        uuid=uuid,
        PurePosixPath=PurePosixPath,
    )
    return namespace


def test_invalid_archive_then_correction_in_the_same_session(acquisition: dict, tmp_path: Path) -> None:
    splits = {
        "train.csv": "a,target\n1,low\n",
        "val.csv": "a,target\n2,mid\n",
        "test.csv": "a,target\n3,high\n",
    }
    broken = _zip(tmp_path / "broken.zip", {k: v for k, v in splits.items() if k != "test.csv"})
    with pytest.raises(ValueError, match="missing required split files"):
        _acquire(acquisition, broken)
    assert not list((acquisition["DATA_ROOT"]).glob(".staging-*")), (
        "a failed attempt must leave nothing behind"
    )
    good = _zip(tmp_path / "good.zip", splits)
    _acquire(acquisition, good)  # previously FileExistsError
    _acquire(acquisition, good)  # a repeat acquisition also works
    assert sorted(acquisition["staged_paths"]) == ["test", "train", "val"]
    assert acquisition["staged_paths"]["test"].read_text() == splits["test.csv"]


def test_switching_datasets_clears_results_and_the_freeze(acquisition: dict, tmp_path: Path) -> None:
    first = _zip(tmp_path / "a.zip", {n: f"a,target\n1,{n}\n" for n in ("train.csv", "val.csv", "test.csv")})
    second = _zip(tmp_path / "b.zip", {n: f"a,target\n2,{n}\n" for n in ("train.csv", "val.csv", "test.csv")})
    _acquire(acquisition, first)
    acquisition.update(TEST_RESULTS="stale", FOUNDATION_RUN_DIRS={"x": 1}, frames={"train": "stale"})
    freeze = acquisition["SESSION_ROOT"] / "freeze.json"
    freeze.write_text("{}")
    _acquire(acquisition, first)  # same data: results are kept
    assert acquisition["TEST_RESULTS"] == "stale" and freeze.exists()
    _acquire(acquisition, second)  # different data: every downstream result is invalidated
    assert (
        "TEST_RESULTS" not in acquisition
        and "FOUNDATION_RUN_DIRS" not in acquisition
        and "frames" not in acquisition
    )
    assert not freeze.exists()
    assert acquisition["DATASET_SHA256"] == acquisition["sha256_file"](second)


# --- NR-01: an ablation keeps the adaptation mode; the runner accepts only exact modes --------------------


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
        CLASS_LABELS=["low", "mid", "high"],
        RANDOM_SEED=42,
        DEVICE_PREFERENCE="auto",
        DATASET_FINGERPRINT="d",
        RUNNER_SHA256="r",
        RUNNER_PATH=Path("runner.py"),
        staged_paths={},
        selected_foundation_conditions=[],
        REQUIRE_ALL_SELECTED_MODELS=True,
        pd=pd,
        display=lambda *a, **k: None,
        shutil=shutil,
    )
    _run_cell("auqhsq19FXEl", namespace)
    namespace["run_foundation_condition"](
        "tabiclv2_finetuned", features=["a"], output_key="ablation", ablation_group="no_stockout"
    )
    assert persisted["condition"] == "fine_tuned" and persisted["ablation_group"] == "no_stockout"

    runner = _runner()
    with pytest.raises(ValueError, match="Unknown adaptation mode 'fine_tuned_no_stockout'"):
        runner["execute"]({"condition": "fine_tuned_no_stockout", "output_dir": str(tmp_path / "never")})
    assert not (tmp_path / "never").exists()


def test_ablation_cell_refuses_a_mode_mismatch() -> None:
    source = CELLS["UKzaZmJxFXEm"]
    assert "condition_override" not in source and "ablation_group=FOUNDATION_ABLATION_GROUP" in source
    assert 'raise ValueError(\n                    f"Adaptation mode differs' in source


# --- probe finding: negative probabilities are rejected on the host and in the runner ---------------------


@pytest.mark.parametrize(
    "align", [lambda: _helpers()["align_probabilities"], lambda: _runner()["align_probabilities"]]
)
def test_negative_probabilities_are_rejected(align) -> None:
    function = align()
    assert function([[0.7, 0.2, 0.1]], ["high", "low", "mid"], ["low", "mid", "high"]).tolist() == [
        [0.2, 0.1, 0.7]
    ]
    with pytest.raises(ValueError, match="[Nn]egative"):
        function([[-0.2, 0.6, 0.6]], ["low", "mid", "high"], ["low", "mid", "high"])


# --- NR-04: the frozen test uses the frozen configuration -------------------------------------------------


def test_frozen_test_ignores_a_live_configuration_change(tmp_path: Path) -> None:
    namespace = _helpers()
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
        "class_labels": ["low", "mid", "high"],
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

    def stream_command(command, log_path):
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

    namespace.update(
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
        stream_command=stream_command,
        staged_paths={"test": tmp_path / "test.csv"},
        PRIMARY_METRIC="balanced_accuracy",
        display=lambda *a, **k: None,
        pickle=pickle,
        BASELINE_ROOT=tmp_path,
        Pipeline=object,
    )
    _run_cell("S8SYDvNeFXEn", namespace)
    executed = namespace["TEST_EFFECTIVE_CONFIGS"]["tabdpt_icl"]
    assert executed["model_configuration"]["n_ensembles"] == 4  # the frozen value, not the live 8
    assert executed["phase"] == "test" and executed["frozen_run_fingerprint"] == "f"
    assert {
        k: v
        for k, v in executed.items()
        if k not in ("phase", "split_paths", "output_dir", "source_run_dir", "frozen_run_fingerprint")
    } == {
        k: v
        for k, v in development.items()
        if k not in ("phase", "split_paths", "output_dir", "run_fingerprint")
    }


# --- NR-02: the ablation interval compares the ablated model with its own full-feature version ------------


def test_identical_full_and_ablated_predictions_give_a_zero_contrast() -> None:
    truth = np.tile(np.array(["low", "mid", "high"]), 50)
    strong = truth.copy()
    weak = np.full(truth.shape, "low", dtype=object)

    def frame(pred):
        return pd.DataFrame({"row_id": range(len(truth)), "target": truth, "prediction": pred})

    registry = {
        "random_forest": frame(strong),
        "lightgbm": frame(weak),
        "lightgbm_no_period_proxies": frame(weak),
    }
    names = {
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "lightgbm_no_period_proxies": "LightGBM without month and weather",
    }
    namespace = {
        "np": np,
        "pd": pd,
        "plt": _Anything(),
        "display": lambda *a, **k: None,
        "RANDOM_SEED": 42,
        "PRIMARY_METRIC": "balanced_accuracy",
        "CLASS_LABELS": ["low", "mid", "high"],
        "TARGET_COLUMN": "target",
        "frames": {"test": pd.DataFrame({"target": truth})},
        "TEST_PREDICTION_REGISTRY": registry,
        "FIGURE_ROOT": Path("."),
        "ABLATION_BASELINE_KEYS": ["lightgbm_no_period_proxies"],
        "baseline_validation_results": pd.DataFrame(
            {"model_key": ["random_forest", "lightgbm"], "balanced_accuracy": [0.6, 0.5]}
        ),
        "TEST_RESULTS": pd.DataFrame(
            {
                "model_key": list(names),
                "model": list(names.values()),
                "condition": ["trained from scratch"] * 3,
                "balanced_accuracy": [1.0, 1 / 3, 1 / 3],
            }
        ),
    }
    _run_cell("166858dd2aa4", namespace, {"BOOTSTRAP_RESAMPLES = 1000 #": "BOOTSTRAP_RESAMPLES = 200 #"})
    contrast = namespace["ABLATION_CONTRASTS"].iloc[0]
    assert contrast["comparator"] == "LightGBM (all features)"
    assert (
        contrast["difference"] == 0
        and contrast["difference_ci_low"] == 0
        and contrast["difference_ci_high"] == 0
    )
    against_reference = (
        namespace["BOOTSTRAP_RESULTS"].set_index("model_key").loc["lightgbm_no_period_proxies"]
    )
    assert against_reference["comparator"] == "Random Forest" and against_reference["gain_ci_high"] < 0


def test_worked_answer_names_the_ablation_contrast() -> None:
    answer = CELLS["79f4f7fdc9f4"]
    assert "paired interval lies below 0" not in answer
    assert "paired ablation contrast" in answer and "do not change between revisions" not in answer


# --- NR-06 / NR-07 ----------------------------------------------------------------------------------------


def test_test_file_summaries_wait_for_the_freeze() -> None:
    assert '"test min / median / max"' not in CELLS["4d1310c2128d"]
    assert '"test min / median / max"' in CELLS["de7423f26aa7"]
    assert "warmer and wetter" not in CELLS["KSNqzrMVFXEm"]


def test_export_keeps_recomputable_evidence() -> None:
    export = CELLS["Da6jGcVwFXEo"]
    for needle in (
        '"notebook_identity"',
        '"comparison_definitions"',
        "validation_predictions.csv",
        "TEST_EFFECTIVE_CONFIGS",
        "frozen_test_ablation_contrasts.csv",
        "EXPORT_ROW_LEVEL_PREDICTIONS",
    ):
        assert needle in export


# --- NR2 review (2026-10-12): freeze integrity and rerun safety -------------------------------------------


def _section_state(tmp_path: Path) -> dict:
    """Notebook state for Sections 4.1, 7.1, 8.0 and 8.1 on a tiny synthetic table. LightGBM is outside CI's
    dependency budget, so a logistic model stands in for LGBMClassifier: these are logic checks, not model
    runs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, label_binarize

    tree = ast.parse(CELLS["xUUykd5OFXEc"])
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classification_metrics"],
        type_ignores=[],
    )
    namespace = _helpers()
    namespace.update(
        accuracy_score=accuracy_score,
        balanced_accuracy_score=balanced_accuracy_score,
        f1_score=f1_score,
        log_loss=log_loss,
        matthews_corrcoef=matthews_corrcoef,
        roc_auc_score=roc_auc_score,
        label_binarize=label_binarize,
    )
    exec(compile(module, "section-0.1-metrics", "exec"), namespace)

    rng = np.random.default_rng(0)
    features = ["lag_7", "stockout_hours", "roll_7_stockout", "month", "precpt"]
    labels = ["low", "mid", "high"]

    def frame(n: int, month: int) -> pd.DataFrame:
        target = np.asarray(labels)[np.arange(n) % 3]
        data = {name: rng.normal(size=n) for name in features}
        data["lag_7"] = data["lag_7"] + np.arange(n) % 3
        data["month"] = np.full(n, month, dtype=float)
        return pd.DataFrame({**data, "target": target})

    frames = {"train": frame(60, 4), "val": frame(30, 5), "test": frame(30, 6)}
    session = tmp_path / "session"
    (session / "runs").mkdir(parents=True)
    namespace.update(
        frames=frames,
        X_train=frames["train"][features],
        y_train=frames["train"]["target"].astype(str),
        X_val=frames["val"][features],
        y_val=frames["val"]["target"].astype(str),
        FEATURE_COLUMNS=features,
        CLASS_LABELS=labels,
        RANDOM_SEED=42,
        PRIMARY_METRIC="balanced_accuracy",
        DATASET_FINGERPRINT="synthetic",
        DATASET_IDENTITY={"synthetic": True},
        WORKSHOP_REVISION="test",
        SESSION_ROOT=session,
        RUN_ROOT=session / "runs",
        FOUNDATION_RUN_DIRS={},
        CONDITION_SPECS={},
        display=lambda *a, **k: None,
        shutil=shutil,
        pickle=pickle,
        time=__import__("time"),
        Pipeline=Pipeline,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        RandomForestClassifier=RandomForestClassifier,
        LGBMClassifier=lambda n_estimators, random_state, verbosity: LogisticRegression(
            max_iter=500, random_state=random_state
        ),
    )
    return namespace


def test_rerunning_the_baselines_after_the_ablations_does_not_break_the_freeze(
    tmp_path: Path, capsys
) -> None:
    """NR2-m4: 4.1 -> 7.1 -> 4.1 -> 8.0 used to raise FileNotFoundError on a deleted ablation result.json."""
    state = _section_state(tmp_path)
    _run_cell("32NWYtJvFXEg", state)  # 4.1
    _run_cell("UKzaZmJxFXEm", state)  # 7.1
    assert state["ABLATION_BASELINE_KEYS"] == ["lightgbm_no_stockout", "lightgbm_no_period_proxies"]
    _run_cell("32NWYtJvFXEg", state)  # 4.1 again: the ablation folders are deleted with the baselines
    capsys.readouterr()
    _run_cell("aNhMl_zxFXEn", state)  # 8.0
    assert "Rerun 7.1" in capsys.readouterr().out
    frozen = json.loads(state["FREEZE_PATH"].read_text())
    assert not any(key.startswith("lightgbm_no_") for key in frozen["selected"])
    assert "lightgbm" in frozen["selected"]


def test_a_freeze_cannot_be_replaced_silently_after_test_scores(tmp_path: Path) -> None:
    """NR2-M1: once the frozen models were scored on test, 8.0 refuses to refreeze without a reason."""
    state = _section_state(tmp_path)
    _run_cell("32NWYtJvFXEg", state)  # 4.1
    _run_cell("aNhMl_zxFXEn", state)  # 8.0: the first freeze is always allowed
    first = state["FREEZE_PATH"].read_bytes()
    assert json.loads(first)["previous_freezes"] == []

    state["TEST_RESULTS"] = pd.DataFrame({"model_key": ["lightgbm"]})  # as left by 8.1
    with pytest.raises(RuntimeError, match="REFREEZE_REASON"):
        _run_cell(
            "aNhMl_zxFXEn",
            state,
            {'SELECTED_MODEL_KEYS = "" # @param': 'SELECTED_MODEL_KEYS = "lightgbm" # @param'},
        )
    assert state["FREEZE_PATH"].read_bytes() == first, "a refused refreeze must leave freeze.json unchanged"

    # A saved test result counts too, even when the in-memory table is gone (e.g. after a kernel restart).
    state["TEST_RESULTS"] = None
    saved = state["RUN_ROOT"] / "test" / "lightgbm"
    saved.mkdir(parents=True)
    (saved / "result.json").write_text("{}")
    with pytest.raises(RuntimeError, match="REFREEZE_REASON"):
        _run_cell("aNhMl_zxFXEn", state)
    assert state["FREEZE_PATH"].read_bytes() == first

    _run_cell(
        "aNhMl_zxFXEn",
        state,
        {'REFREEZE_REASON = "" # @param': 'REFREEZE_REASON = "instructor demo of a refreeze" # @param'},
    )
    replaced = json.loads(state["FREEZE_PATH"].read_text())
    assert len(replaced["previous_freezes"]) == 1
    record = replaced["previous_freezes"][0]
    assert record["freeze_sha256"] == hashlib.sha256(first).hexdigest()
    assert record["test_scores_seen"] is True
    assert record["replaced_because"] == "instructor demo of a refreeze"
    assert state["TEST_RESULTS"] is None


def test_the_text_states_what_the_freeze_enforces_and_how_the_comparison_was_configured() -> None:
    freeze_text = CELLS["kwt1IimAFXEm"]
    assert "prevents this by design" not in freeze_text and "REFREEZE_REASON" in freeze_text  # NR2-M1
    troubleshooting = CELLS["83fda0b2603d"]
    assert "REFREEZE_REASON" in troubleshooting  # the refusal has a recovery row
    assert "frozen development result changed after freeze" in troubleshooting  # NR2-m4: 4.1 rerun after 8.0
    limitations = CELLS["kLiPqnp9FXEo"]
    assert "default hyperparameters" in limitations and "not tuned" in limitations  # NR2-m2
    for cell_id in ("ccf87e15ac16", "faafc7ab3106", "thuW_LzJFXEm", "79f4f7fdc9f4"):
        answer = CELLS[cell_id]
        assert "commit `05d54cd`" in answer  # NR2-m1: every quoted answer names its recorded run
        assert "revision 2.0.1" not in answer and "Lower for every model" not in answer
    # NR2-m6
    assert "TabICLv2 fine-tuning selects its checkpoint by validation accuracy" in CELLS["jkJ6XuNHFXEh"]
    references = CELLS["PSJigkamFXEp"]
    for link in (
        "arxiv.org/abs/2510.21204",
        "arxiv.org/abs/2410.18164",
        "arxiv.org/abs/2605.13986",
        "arxiv.org/abs/2602.11139",
    ):
        assert link in references  # NR2-m7


def test_saved_output_advice_matches_the_committed_notebook() -> None:
    """NR2-m3: the committed notebook has no outputs, so no cell tells learners to hide saved outputs."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert not any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell_id in ("fbcbb4ee401d", "KSNqzrMVFXEm"):
        assert "hide them first" not in CELLS[cell_id] and "saved outputs hidden" not in CELLS[cell_id]
