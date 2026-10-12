"""CPU tests for the rice pixel-tables notebook (tools/rice_workshop/): data recipes, protocol, runner contract and
generator parity. No torch, no model, no network: models are replaced by a deterministic fake behind the real runner."""

# ruff: noqa: E501  (long assertion messages and fixtures are kept on one line)
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools" / "rice_workshop"
NOTEBOOK = REPO / "tutorials" / "DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb"
V2 = REPO / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"rice_workshop_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rt = _load("rice_tables")
runner_module = _load("fm_runner")
build = _load("build_workshop")


# ------------------------------------------------------------------------------------------------ recipes


def _chip(seed: int, *, rice: float = 0.3, ae: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    s2 = rng.uniform(0.01, 0.6, size=(rt.N_BINS, rt.N_BANDS, rt.CHIP, rt.CHIP)).astype(np.float32)
    valid = (rng.random((rt.N_BINS, rt.CHIP, rt.CHIP)) > 0.3).astype(np.uint8)
    label = (rng.random((rt.CHIP, rt.CHIP)) < rice).astype(np.uint8)
    label[:2] = rt.IGNORE
    chip = {"S2": s2, "S2_valid": valid, "label": label}
    if ae:
        chip["AlphaEarth"] = rng.normal(0, 0.2, size=(rt.AE_DIM, rt.CHIP, rt.CHIP)).astype(np.float32)
        chip["AE_valid"] = (rng.random((rt.CHIP, rt.CHIP)) > 0.1).astype(np.uint8)
    return chip


def test_feature_names_match_the_declared_recipes() -> None:
    assert len(rt.M1_NAMES) == 70 and len(set(rt.M1_NAMES)) == 70
    assert len(rt.AE_NAMES) == 65 and rt.AE_NAMES[-1] == "AE_valid"


def test_m1_and_alphaearth_features_have_the_study_shapes() -> None:
    chip = rt.validate_chip(_chip(1))
    m1 = rt.m1_features(chip)
    assert m1.shape == (rt.CHIP * rt.CHIP, 70) and m1.dtype == np.float32
    fractions = m1[:, -4:]
    assert np.all((fractions >= 0) & (fractions <= 1))
    ae = rt.ae_features(chip)
    invalid = chip["AE_valid"].reshape(-1) == 0
    assert ae.shape == (rt.CHIP * rt.CHIP, 65)
    assert np.isnan(ae[invalid, :64]).all() and np.isfinite(ae[~invalid, :64]).all()
    assert np.array_equal(ae[:, 64], chip["AE_valid"].reshape(-1))


def test_split_is_the_capstone_rule_and_never_uses_the_development_example() -> None:
    rows = []
    for region in range(6):
        for k in range(4):
            rows.append({"group_id": f"ph_{region}_{k}", "year": 2020, "split": "train",
                         "region_iso": f"PH-0{region}", "stratum": "rice" if k % 2 == 0 else "nonrice"})
    rows += [{"group_id": "ph_v_1", "year": 2021, "split": "validation", "region_iso": "PH-07", "stratum": "rice"}]
    first = rt.sample_roles(rows)
    assert first == rt.sample_roles(rows)
    roles = first["roles"]
    assert set(roles["fit"]).isdisjoint(roles["development"]) and roles["validation"] == ["ph_v_1_2021"]
    with pytest.raises(ValueError, match="development example"):
        rt.check_roles({"fit": [f"{rt.DEVELOPMENT_EXAMPLE_GROUP}_2020"]})


def test_fit_sample_is_the_capstone_pixel_sample() -> None:
    labels = [np.array([0] * 90 + [1] * 10), np.array([0] * 50 + [1] * 50)]
    picks = rt.pixel_sample(labels, 40, rt.PIXEL_SEED)
    assert sum(len(p) for p in picks) == 40
    again = rt.pixel_sample(labels, 40, rt.PIXEL_SEED)
    assert all(np.array_equal(a, b) for a, b in zip(picks, again, strict=True))
    assert all(set(np.unique(y[p])) == {0, 1} for y, p in zip(labels, picks, strict=True))
    assert all(len(set(p.tolist())) == len(p) for p in picks)


def test_development_subset_keeps_the_class_mix() -> None:
    rng = np.random.default_rng(0)
    labels = [(rng.random(5000) < 0.1).astype(np.uint8) for _ in range(4)]
    labels[0][:100] = rt.IGNORE
    picks = rt.proportional_sample(labels, 3000, rt.DEV_SUBSET_SEED)
    chosen = np.concatenate([y[p] for y, p in zip(labels, picks, strict=True)])
    assert len(chosen) == 3000 and rt.IGNORE not in chosen
    pool = np.concatenate([y[y != rt.IGNORE] for y in labels])
    assert abs(chosen.mean() - pool.mean()) < 1e-3
    every = rt.proportional_sample(labels, 10**6, 1)
    assert sum(len(p) for p in every) == len(pool)


def test_activity_contexts_are_nested_and_class_proportional() -> None:
    y = np.array([1] * 351 + [0] * 649)
    subsets = rt.nested_context(y, [100, 300, 1000], rt.ACTIVITY_SEED)
    assert set(subsets[100]) <= set(subsets[300]) <= set(subsets[1000])
    assert abs(y[subsets[300]].mean() - 0.351) < 0.01
    with pytest.raises(ValueError):
        rt.nested_context(y, [5000], 1)


# ------------------------------------------------------------------------------------------------ BYOD contract


def test_chip_validation_names_the_failed_rule() -> None:
    chip = _chip(2)
    s2_only = {k: v for k, v in chip.items() if k not in ("AlphaEarth", "AE_valid")}
    assert rt.validate_chip(s2_only)["AlphaEarth"] is None
    with pytest.raises(rt.ChipError, match="together"):
        rt.validate_chip({**s2_only, "AlphaEarth": chip["AlphaEarth"]})
    with pytest.raises(rt.ChipError, match="AlphaEarth has shape"):
        rt.validate_chip({**chip, "AlphaEarth": chip["AlphaEarth"][:10]})
    bad = chip["AlphaEarth"].copy()
    bad[:, chip["AE_valid"].astype(bool)] = np.nan
    with pytest.raises(rt.ChipError, match="non-finite"):
        rt.validate_chip({**chip, "AlphaEarth": bad})
    with pytest.raises(rt.ChipError, match="10 000"):
        rt.validate_chip({**chip, "S2": chip["S2"] * 10000})
    with pytest.raises(rt.ChipError, match="label values"):
        rt.validate_chip({**chip, "label": np.full((rt.CHIP, rt.CHIP), 3, np.uint8)})


def test_collect_byod_refuses_unsafe_archives_and_duplicates(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../escape.h5", b"x")
    target = tmp_path / "input"
    target.mkdir()
    (target / "stale.h5").write_bytes(b"old")
    with pytest.raises(rt.ChipError, match="unsafe member"):
        rt.collect_byod([bad], target)
    assert not (target / "stale.h5").exists()
    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("a.h5")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"target")
    with pytest.raises(rt.ChipError, match="symbolic link"):
        rt.collect_byod([link], target)
    one, two = tmp_path / "one", tmp_path / "two"
    for folder in (one, two):
        folder.mkdir()
        (folder / "site_2020.h5").write_bytes(b"x")
    with pytest.raises(rt.ChipError, match="share the file name"):
        rt.collect_byod([one, two], target)


def test_collect_byod_opens_zip_archives_inside_a_folder(tmp_path: Path) -> None:
    # Kurt's browser run of 373aaf5: the upload dialog hands collect_byod a folder holding the uploaded .zip, which
    # was never opened ("no .h5 chips found").
    upload = tmp_path / "byod_upload"
    upload.mkdir()
    with zipfile.ZipFile(upload / "chips.zip", "w") as archive:
        archive.writestr("site_a_2019.h5", b"a")
        archive.writestr("nested/site_b_2020.h5", b"b")
    (upload / "site_c_2021.h5").write_bytes(b"c")
    paths = rt.collect_byod([upload], tmp_path / "input")
    assert [p.name for p in paths] == ["site_a_2019.h5", "site_b_2020.h5", "site_c_2021.h5"]
    assert (tmp_path / "input" / "site_b_2020.h5").read_bytes() == b"b"
    with zipfile.ZipFile(upload / "bad.zip", "w") as archive:
        archive.writestr("../escape.h5", b"x")
    with pytest.raises(rt.ChipError, match="unsafe member"):
        rt.collect_byod([upload], tmp_path / "input")
    assert not list((tmp_path / "input").iterdir())


def test_codex_round1_freeze_covers_tables_and_decisions(tmp_path: Path) -> None:
    # Codex on 455c357: staged tables were not re-verified at scoring, and a reload that flipped decisions passed.
    ledger = rt.Ledger(tmp_path / "run")
    (ledger.root / "tables").mkdir(parents=True)
    (ledger.root / "tables" / "s2m1_eval_X.npy").write_bytes(b"frozen")
    selection = {"conditions": {}, "tables_inventory": rt.table_inventory(ledger.root / "tables")}
    rt.freeze(ledger, selection)
    assert rt.verify_frozen(ledger) == selection
    (ledger.root / "tables" / "s2m1_eval_X.npy").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="staged table"):
        rt.verify_frozen(ledger)
    source = Path(rt.__file__).read_text(encoding="utf-8")
    assert source.count('decisions_identical_at_frozen_threshold") is False') == 2


def test_codex_round1_mitra_keeps_a_two_class_tuning_set_on_small_development(tmp_path: Path) -> None:
    # Codex on 455c357: with <= MITRA_DEV_PIXELS development rows, the threshold sample took every row and the tuning
    # set came out empty.
    ledger = rt.Ledger(tmp_path / "run")
    data = ledger.root / "tables"
    data.mkdir(parents=True)
    rt.write_json(data / "tables.json", {"feature_sets": {"s2m1": {}}})
    ledger.finish("prepare")
    dev_y = np.array([0] * 300 + [1] * 100, dtype=np.int64)
    np.save(data / "s2m1_dev_y.npy", dev_y)
    np.save(data / "s2m1_dev_X.npy", np.zeros((len(dev_y), 70), np.float32))
    seen = {}

    def runner(system, mode, feature_sets, extra, out_dir):
        seen.update(extra["indices"]["s2m1"])
        raise RuntimeError("stop after staging")

    result = rt.foundation_stage(ledger, ["C3"], runner, {}, mitra_dev_pixels=10_000, require_all=False)
    dev, tune = np.load(seen["dev"]), np.load(seen["tune"])
    assert len(tune) == 80 and len(dev) == 320 and not set(dev) & set(tune)
    assert set(dev_y[tune].tolist()) == {0, 1}
    assert "stop after staging" in result["errors"]["C3"]


def test_codex_round2_findings(tmp_path: Path) -> None:
    # Codex on 4c66d02. (b) out-of-range probabilities are rejected, float noise is clipped.
    with pytest.raises(ValueError, match="outside"):
        runner_module.positive_column(np.array([[-0.5, 1.5]]), [0, 1])
    assert runner_module.positive_column(np.array([[0.0, 1.0 + 1e-9]]), [0, 1])[0] == 1.0
    # (a) the run summary lists the systems that ran, not the ones selected.
    assert "FOUNDATION['records'].values()})]," in build.CELL_EXPORT
    # (c) a BYOD freeze records the BYOD split rule.
    ledger = rt.Ledger(tmp_path / "run")
    for name in ("baseline", "foundation", "activity"):
        (ledger.root / name).mkdir(parents=True, exist_ok=True)
        ledger.finish(name)
    (ledger.root / "tables").mkdir()
    rt.write_json(ledger.root / "tables" / "tables.json", {"feature_sets": {}})
    rt.write_json(ledger.root / "features.json", {})
    rt.write_json(ledger.root / "baseline" / "records.json", {})
    rt.write_json(ledger.root / "foundation" / "records.json", {})
    rt.write_json(ledger.root / "activity" / "activity.json", {"ran": False})
    rt.write_json(ledger.root / "data_manifest.json", {"split_rule": rt.BYOD_SPLIT_RULE})
    assert rt.freeze_stage(ledger)["protocol"]["roles"] == rt.BYOD_SPLIT_RULE
    rt.write_json(ledger.root / "data_manifest.json", {})
    (ledger.root / "receipts" / "freeze.json").unlink()
    assert rt.freeze_stage(ledger)["protocol"]["roles"] == rt.SPLIT_RULE


def test_codex_round3_findings() -> None:
    # Codex on 820a780. (1) four development chips with one rice pixel each: the rice class keeps a row.
    labels = [np.array([1] + [0] * 16_383) for _ in range(4)]
    picked = rt.proportional_sample(labels, 30_000, 7)
    assert sum(int(y[p].sum()) for y, p in zip(labels, picked, strict=True)) >= 1
    assert sum(len(p) for p in picked) == 30_000
    # an abundant-rice source is unchanged by the guard (same rows as plain largest-remainder allocation)
    rich = [np.array([1] * 4_000 + [0] * 12_384), np.array([1] * 2_000 + [0] * 14_384)]
    assert sum(len(p) for p in rt.proportional_sample(rich, 10_000, 7)) == 10_000
    # (2) environment failures reach the exported failures; (3) exact probability serialisation.
    assert "ENVIRONMENT_FAILED_SYSTEMS" in build.CELL_ENV_TAIL and "ENVIRONMENT_FAILED_SYSTEMS}" in build.CELL_EXPORT
    assert "repr(float(v[i]))" in Path(rt.__file__).read_text(encoding="utf-8")
    assert float(repr(float(np.float32(0.4999996)))) == float(np.float32(0.4999996))


def test_codex_round4_mitra_evaluation_sample_is_frozen(tmp_path: Path) -> None:
    # Codex on 44c75ed: the Mitra evaluation sample was drawn after verify_frozen, from mutable constants.
    ledger = rt.Ledger(tmp_path / "run")
    for name in ("baseline", "foundation", "activity"):
        (ledger.root / name).mkdir(parents=True, exist_ok=True)
        ledger.finish(name)
    data = ledger.root / "tables"
    data.mkdir()
    rt.write_json(data / "tables.json", {"feature_sets": {"s2m1": {}}})
    eval_y = np.array([0] * 900 + [1] * 300, dtype=np.int64)
    np.save(data / "s2m1_eval_y.npy", eval_y)
    np.save(data / "s2m1_eval_X.npy", np.zeros((len(eval_y), 70), np.float32))
    for name in ("features.json", "data_manifest.json"):
        rt.write_json(ledger.root / name, {})
    rt.write_json(ledger.root / "baseline" / "records.json", {})
    rt.write_json(ledger.root / "foundation" / "records.json",
                  {"C3/s2m1": {"kind": "foundation", "system": "C3", "feature_set": "s2m1"}})
    rt.write_json(ledger.root / "activity" / "activity.json", {"ran": False})
    selection = rt.freeze_stage(ledger, mitra_validation_pixels=1000)
    item = selection["frozen_eval_indices"]["C3/s2m1"]
    assert len(np.load(ledger.root / item["path"])) == 1000
    assert rt.verify_frozen(ledger)["frozen_eval_indices"] == selection["frozen_eval_indices"]
    np.save(ledger.root / item["path"], np.arange(1000))
    with pytest.raises(RuntimeError, match="frozen evaluation sample"):
        rt.verify_frozen(ledger)


def test_codex_round1_activity_and_chip_labels_in_cells() -> None:
    assert "available=sorted({r['system'] for r in FOUNDATION['records'].values()})" in build.CELL_ACTIVITY
    assert "(sample)" in build.CELL_RESULTS and "declared sample" in build.CELL_RESULTS


def test_grid_name_marks_declared_samples() -> None:
    # Kurt's Mitra run of 98e7725: Mitra's sample F1 sat unlabelled beside all-pixel F1 in the grids.
    assert rt.grid_name("Mitra", "declared seeded sample of 10000 validation pixels (seed 20261016)") == \
        "Mitra (10,000-pixel sample)"
    assert rt.grid_name("TabDPT v1.2", "all 113821 labelled validation pixels") == "TabDPT v1.2"
    assert rt.grid_name("All non-rice", "") == "All non-rice"
    for cell in (build.CELL_DEV_GRID, build.CELL_RESULTS, build.CELL_BYOD):
        assert "rt.grid_name(" in cell


def test_byod_upload_folder_is_cleared_per_attempt() -> None:
    source = build.CELL_BYOD
    assert "shutil.rmtree(upload_dir)" in source
    assert source.index("shutil.rmtree(upload_dir)") < source.index("files.upload()")


def test_byod_plan_needs_six_labelled_chips_from_three_sites() -> None:
    small = {f"s{i}_2020": rt.validate_chip(_chip(i)) for i in range(5)}
    with pytest.raises(rt.ChipError, match="at least 6 labelled chips"):
        rt.byod_plan(small)
    chips = {f"site{i % 4}_{2019 + i // 4}": {**rt.validate_chip(_chip(10 + i)), "group": None} for i in range(8)}
    plan = rt.byod_plan(chips)
    assert plan["feature_sets"] == ["alphaearth", "s2m1"]
    sites = {role: {plan["groups"][c] for c in ids} for role, ids in plan["split"].items()}
    assert sites["fit"].isdisjoint(sites["test"]) and sites["fit"].isdisjoint(sites["development"])
    s2_only = dict(chips)
    s2_only["site0_2019"] = {**chips["site0_2019"], "AlphaEarth": None, "AE_valid": None}
    assert rt.byod_plan(s2_only)["feature_sets"] == ["s2m1"]


def test_h5_reader_reads_only_schema_datasets(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    chip = _chip(3)
    path = tmp_path / "a_2020.h5"
    with h5py.File(path, "w") as handle:
        for key, value in chip.items():
            handle[key] = value
        handle["label_raw"] = np.full((rt.CHIP, rt.CHIP), 9, np.uint8)
        handle.attrs["group_id"] = "site-a"
    read = rt.read_chip_h5(path)
    assert read["group"] == "site-a" and np.array_equal(read["label"], chip["label"])
    with h5py.File(tmp_path / "b.h5", "w") as handle:
        handle["S2"] = np.zeros((3, 3))
    with pytest.raises(rt.ChipError, match="checked before reading"):
        rt.read_chip_h5(tmp_path / "b.h5")


# ------------------------------------------------------------------------------------------------ protocol, end to end


def _synthetic_run(root: Path) -> rt.Ledger:
    """Stage tables from synthetic per-pixel features (skipping the slow M1 recipe) for 6 fit / 2 dev / 2 val chips."""
    rng = np.random.default_rng(7)
    roles = {"fit": [f"f{i}" for i in range(6)], "development": ["d0", "d1"], "validation": ["v0", "v1"]}
    chips = {}
    for cid in [c for ids in roles.values() for c in ids]:
        n = 4000
        y = (rng.random(n) < 0.25).astype(np.uint8)
        y[:50] = rt.IGNORE
        signal = (y == 1).astype(np.float32)
        chips[cid] = {"y": y}
        for fs, names in rt.FEATURE_SETS.items():
            x = rng.normal(0, 1, size=(n, len(names))).astype(np.float32)
            x[:, 0] += 2.0 * signal * (1.5 if fs == "alphaearth" else 0.6)
            x[rng.random(x.shape) < 0.01] = np.nan
            chips[cid][fs] = x
    ledger = rt.Ledger(root)
    ledger.start("prepare")
    manifest = rt.stage_tables(chips, roles, root / "tables", evaluation_role="validation",
                               feature_sets=list(rt.FEATURE_SETS))
    table = [{"chip": c, "role": r} for r, ids in roles.items() for c in ids]
    rt._finish_prepare(ledger, {"dataset": "synthetic", "roles": roles}, manifest, table, 0.0)
    return ledger


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.__version__ = "fake"
        self.cuda = types.SimpleNamespace(is_available=lambda: False)


def _fake_model(monkeypatch) -> None:
    """Replace the model families inside the real fm_runner with a deterministic logistic score of column 0."""

    def predictor(df):
        return 1.0 / (1.0 + np.exp(-2.0 * (df.to_numpy()[:, 0] - 1.0)))

    def fit_family(family, pipe, fit_x, fit_y, tune_x, tune_y, names, cfg, artifact):
        def save():
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "context.npy").write_bytes(np.asarray(fit_x[:5]).tobytes())
        return predictor, save

    monkeypatch.setattr(runner_module, "load_family", lambda family, cfg, device: object())
    monkeypatch.setattr(runner_module, "fit_family", fit_family)
    monkeypatch.setattr(runner_module, "reload_family", lambda family, cfg, device, artifact: predictor)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())


def _runner(calls: list):
    def run(system, mode, feature_sets, extra, out_dir):
        cfg = rt.runner_config(system, mode, feature_sets, extra, out_dir, weights_dir=Path("w"), seed=42)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / f"{mode}.config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        assert runner_module.main(["--config", str(path)]) == 0
        calls.append((system, mode, tuple(feature_sets)))
        return json.loads(Path(cfg["result_path"]).read_text(encoding="utf-8"))
    return run


def test_the_protocol_freezes_scores_once_and_exports(tmp_path: Path, monkeypatch) -> None:
    _fake_model(monkeypatch)
    calls: list = []
    runner = _runner(calls)
    ledger = _synthetic_run(tmp_path / "run")
    with pytest.raises(RuntimeError, match="freeze needs"):
        ledger.start("freeze")
    rt.baseline_stage(ledger, systems=("A",))
    found = rt.foundation_stage(ledger, ["C1", "C3"], runner, {"tabpfn": {"revision": "x"}}, mitra_dev_pixels=500)
    assert set(found["records"]) == {"C1/alphaearth", "C1/s2m1", "C3/alphaearth", "C3/s2m1"}
    assert found["records"]["C3/s2m1"]["dev_pixels"] == 500  # Mitra's declared sample
    assert ("C3", "develop", ("alphaearth",)) in calls and ("C3", "develop", ("s2m1",)) in calls  # own process each
    activity = rt.activity_stage(ledger, runner, enabled=True, system="C1", feature_set="s2m1", sizes=[100, 300],
                                 available=["C1", "C3"])
    assert [r["context_rows"] for r in activity["rows"]] == [100, 300]
    selection = rt.freeze_stage(ledger, mitra_validation_pixels=700)
    assert set(selection["conditions"]) >= {"A/alphaearth", "C1/s2m1"}
    result = rt.score_stage(ledger, runner)
    assert all(v["within_tolerance"] for v in result["parity"].values())
    validation = [r for r in result["rows"] if r["split"] == "validation"]
    assert len(validation) == 2 * len(selection["conditions"])
    mitra = [r for r in validation if r["condition"] == "C3/s2m1"]
    assert "declared seeded sample of 700" in mitra[0]["pixels"]
    frozen_sample = ledger.root / selection["frozen_eval_indices"]["C3/s2m1"]["path"]
    assert np.array_equal(np.load(frozen_sample), np.load(ledger.root / "scores" / "indices" / "C3_s2m1_eval.npy"))
    assert any(r["split"] == "validation (Mitra sample)" for r in result["rows"])
    a_rows = [r for r in validation if r["condition"] == "A/alphaearth"]
    assert a_rows[0]["recall"] == 0.0 and a_rows[0]["precision"] is None
    probe = rt.probe_reload(ledger, runner)
    assert probe["ran"] and probe["within_tolerance"]
    for stage in ("prepare", "baseline", "foundation", "activity", "freeze", "score"):
        with pytest.raises(RuntimeError, match="exactly once"):
            ledger.start(stage)
    exported = rt.export_stage(ledger, {"run": "test"}, "# conclusion\n")
    with zipfile.ZipFile(exported["path"]) as archive:
        names = set(archive.namelist())
    assert {"scores/metrics.csv", "selection.json", "run_summary.json", "ATTRIBUTION.txt",
            "scores/validation_predictions.csv.gz"} <= names
    assert not any(name.startswith("tables/") or "artifact" in name for name in names)
    assert rt.verify_zip(exported["path"])["verified"]


def test_a_changed_selection_is_refused(tmp_path: Path, monkeypatch) -> None:
    _fake_model(monkeypatch)
    runner = _runner([])
    ledger = _synthetic_run(tmp_path / "run")
    rt.baseline_stage(ledger, systems=("A",))
    rt.foundation_stage(ledger, ["C1"], runner, {})
    rt.activity_stage(ledger, runner, enabled=False, system="C1", feature_set="s2m1", sizes=[100], available=["C1"])
    rt.freeze_stage(ledger)
    path = ledger.root / "selection.json"
    path.write_text(path.read_text().replace('"threshold": 0.', '"threshold": 1.'), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after the freeze"):
        rt.score_stage(ledger, runner)


def test_restarting_an_upstream_stage_clears_downstream_results(tmp_path: Path, monkeypatch) -> None:
    _fake_model(monkeypatch)
    ledger = _synthetic_run(tmp_path / "run")
    rt.baseline_stage(ledger, systems=("A",))
    assert (ledger.root / "baseline" / "records.json").exists()
    ledger.start("prepare")
    assert not ledger.done("baseline") and not (ledger.root / "baseline").exists()
    assert not (ledger.root / "tables").exists()


def test_artifact_tampering_is_caught_before_reload(tmp_path: Path, monkeypatch) -> None:
    _fake_model(monkeypatch)
    runner = _runner([])
    ledger = _synthetic_run(tmp_path / "run")
    rt.baseline_stage(ledger, systems=("A",))
    rt.foundation_stage(ledger, ["C1"], runner, {})
    rt.activity_stage(ledger, runner, enabled=False, system="C1", feature_set="s2m1", sizes=[100], available=["C1"])
    rt.freeze_stage(ledger)
    (ledger.root / "foundation" / "C1" / "s2m1" / "artifact" / "context.npy").write_bytes(b"swapped")
    with pytest.raises(RuntimeError, match="differs from the inventory"):
        rt.score_stage(ledger, runner)


def test_lightgbm_baseline_is_reloaded_from_its_file(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("lightgbm")
    _fake_model(monkeypatch)
    runner = _runner([])
    ledger = _synthetic_run(tmp_path / "run")
    records = rt.baseline_stage(ledger, systems=("A", "B", "B10k"))
    assert records["B/s2m1"]["fit_rows"] > records["B10k/s2m1"]["fit_rows"]
    rt.foundation_stage(ledger, [], runner, {})
    rt.activity_stage(ledger, runner, enabled=True, system="C1", feature_set="s2m1", sizes=[100], available=[])
    rt.freeze_stage(ledger)
    result = rt.score_stage(ledger, runner)
    assert {r["condition"] for r in result["rows"]} == {f"{s}/{fs}" for s in ("A", "B", "B10k") for fs in rt.FEATURE_SETS}
    assert not rt.probe_reload(ledger, runner)["ran"]


def test_runner_maps_the_rice_column_and_predicts_every_row() -> None:
    p = runner_module.positive_column(np.array([[0.2, 0.8], [0.9, 0.1]]), [0, 1])
    assert np.allclose(p, [0.8, 0.1])
    assert np.allclose(runner_module.positive_column(np.array([[0.8, 0.2]]), ["1", "0"]), [0.8])
    with pytest.raises(ValueError):
        runner_module.positive_column(np.array([[1.0]]), [0])
    pytest.importorskip("pandas")
    x = np.arange(25, dtype=np.float32).reshape(25, 1)
    prob, _seconds = runner_module.predict_chunks(lambda df: df.to_numpy()[:, 0] / 100, x, ["a"], 8)
    assert len(prob) == 25 and np.isclose(prob[-1], 0.24)


# ------------------------------------------------------------------------------------------------ the notebook


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(kind: str) -> list[str]:
    return ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == kind]


def test_generator_parity() -> None:
    assert NOTEBOOK.read_text(encoding="utf-8") == build.build(), "run python tools/rice_workshop/build_workshop.py"


def test_notebook_is_output_free_parseable_and_magic_free() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        assert max((len(line) for line in source.splitlines()), default=0) <= 2000, index
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse(source)
            assert "get_ipython" not in source and '"-m", "pip"' not in source
            assert not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())


def test_metadata_declares_profile_mode_kind_and_spec() -> None:
    dimer = _notebook()["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.4" and dimer["notebook_profile"] == "E2E"
    assert dimer["notebook_mode"] == "WORKSHOP" and dimer["notebook_kind"] == "WORKSHOP"
    assert dimer["standalone"] is True and dimer["release_status"] == "Candidate"


def test_carried_sources_match_the_repository() -> None:
    code = "\n".join(_cells("code"))
    for name in ("rice_tables.py", "fm_runner.py"):
        text = (TOOLS / name).read_bytes().decode("utf-8").replace("\r\n", "\n")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() in code
    head = build.v2_environment_head()
    assert head in code, "the v2 environment cell must be carried verbatim"
    for stack in ("tabpfn", "tabicl", "tabdpt", "mitra"):
        lock = (REPO / "tutorials" / f"requirements-workshop-{stack}.lock.txt").read_text(encoding="utf-8")
        assert "".join(ast.literal_eval(_lock_literal(head, stack))) == lock.replace("\r\n", "\n")


def _lock_literal(head: str, stack: str) -> str:
    tree = ast.parse(head)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CARRIED_ENVIRONMENT_LOCKS":
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if ast.literal_eval(key) == stack:
                    return repr([ast.literal_eval(value)])
    raise AssertionError(stack)


def test_defaults_pins_and_scope() -> None:
    code = "\n".join(_cells("code"))
    assert "RUN_MITRA = False" in code and "USE_BYOD = False" in code and "RUN_ACTIVITY = True" in code
    for system in ("RUN_TABPFN3 = True", "RUN_TABICLV2 = True", "RUN_TABDPT = True"):
        assert system in code
    module = (TOOLS / "rice_tables.py").read_text(encoding="utf-8")
    assert 'INDEX_SHA256 = "e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d"' in module
    assert "DATASET_VERSION = 1" in module and '["label_raw"]' not in module
    assert re.search(r"MOMENT|momentfm", code) is None  # MOMENT is out of scope for this revision


def test_opening_carries_licence_notice_and_numbered_references() -> None:
    markdown = _cells("markdown")
    opening = markdown[0]
    assert "non-commercial" in opening and "TabPFN-3" in opening
    assert "> **⚠️ Notice.**" in opening and "Problem statement" in opening
    text = "\n".join(markdown)
    references = text[text.index("## References"):]
    entries = {int(n) for n in re.findall(r"(?m)^\[(\d+)\] ", references)}
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text[: text.index("## References")])}
    assert entries and entries == set(range(1, max(entries) + 1))
    assert cited == entries, f"uncited {entries - cited}, unresolved {cited - entries}"
    assert "https://doi.org/" in references
    assert "AI" in text and "What did not work" in text and "Troubleshooting" in text
