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
