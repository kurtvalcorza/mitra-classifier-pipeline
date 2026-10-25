"""The E2E notebook's closing external-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1-XAS7, SART1-SART11).

Offline and model-free (CI's light pins only; AutoGluon is never imported): the pin, the digest refusal before
extraction, the SART10 contract check, that the section's cells use only names the setup cells define (XAS5), the
section order (§7.1), the ST1 exact-URL rule, and an end-to-end run of the section's own cells against a stand-in
bundle (test-only URL/digest override, stub predictor) that writes only under `outputs/external_artifact/` (XAS6).
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mitra_pipeline import tutorial_api as api

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "mitra_classifier_colab.ipynb"
PINNED_URL = "https://github.com/kurtvalcorza/mitra-classifier-pipeline/releases/download/sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip"
PINNED_SHA256 = "17cbea6fbd8ed5a97decd4b359ef2c47d1d6025d26447e7500bc65a4dde364dd"
SECTION_START = "## 10. Using an artifact made elsewhere"
SECTION_END = "## 11. Optional activity"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load("xas_validate_release_assets", ROOT / "tools" / "validate_release_assets.py")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cells() -> list[dict]:
    return json.loads(E2E.read_text(encoding="utf-8"))["cells"]


def _index(cells: list[dict], heading: str) -> int:
    return next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(heading))


def _section_cells() -> list[dict]:
    cells = _cells()
    return cells[_index(cells, SECTION_START) : _index(cells, SECTION_END)]


def _section_code() -> list[str]:
    return [_src(c) for c in _section_cells() if c["cell_type"] == "code"]


# ---------------------------------------------------------------- the pin (SART6, XAS2)


def test_section_pins_the_release_asset_by_url_and_digest() -> None:
    assert api.SAMPLE_ARTIFACT_URL == PINNED_URL == VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL
    assert api.SAMPLE_ARTIFACT_SHA256 == PINNED_SHA256
    assert api.SAMPLE_ARTIFACT_TAG == "sample-artifact-breast-cancer-v1" and api.SAMPLE_ARTIFACT_TAG in PINNED_URL
    assert api.SAMPLE_ARTIFACT_PRODUCER["commit"] == "be831ef" and api.SAMPLE_ARTIFACT_PRODUCER["data_source"] == "Sample: Breast Cancer"


def test_section_never_uses_this_runs_export() -> None:
    """XAS2 / §25.11: no section cell names the Section 9 export, its reload directory or the in-memory model."""
    code = "\n".join(_section_code())
    for forbidden in ("mitra_classifier_predictor.zip", "artifact-reload", "RELOAD_DIR", "ACTIVE_MODEL", "archive_digest", "active_path"):
        assert forbidden not in code, forbidden
    assert "fetch_pinned_artifact(EXT_INPUTS / 'pinned_sample_artifact.zip', SAMPLE_ARTIFACT_URL, SAMPLE_ARTIFACT_SHA256)" in code


# ---------------------------------------------------------------- integrity (XAS4, VER6)


def _zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("artifact_manifest.json", "{}")
    return path


def test_pinned_digest_mismatch_refuses_before_extraction(tmp_path) -> None:
    archive = _zip(tmp_path / "tampered.zip")
    cache = tmp_path / "cache" / "pinned_sample_artifact.zip"
    with pytest.raises(ValueError, match=r"Trusted digest mismatch .* the pinned SAMPLE_ARTIFACT_SHA256 is .*nothing was extracted"):
        api.fetch_pinned_artifact(cache, archive.resolve().as_uri(), PINNED_SHA256)
    assert not cache.exists() and not cache.with_name(cache.name + ".part").exists()


def test_pinned_digest_match_is_kept_and_reused(tmp_path) -> None:
    archive = _zip(tmp_path / "standin.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache" / "pinned_sample_artifact.zip"
    assert api.fetch_pinned_artifact(cache, archive.resolve().as_uri(), digest) == cache
    archive.unlink()  # a second call reuses the verified cache without the network
    assert api.fetch_pinned_artifact(cache, "file:///nonexistent.zip", digest) == cache


def test_check_trusted_digest_rejects_a_malformed_digest(tmp_path) -> None:
    archive = _zip(tmp_path / "a.zip")
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        api.check_trusted_digest(archive, "abc", "EXPECTED_ZIP_SHA256")


def test_artifact_from_another_dataset_breaks_the_data_contract() -> None:
    """SART10: a hash-valid bundle built on another table or label set is refused."""
    from sklearn.datasets import load_breast_cancer, load_wine

    contract = {"problemType": "binary", "featureColumns": list(load_breast_cancer().feature_names), "targetColumn": "target", "classLabels": ["benign", "malignant"]}
    good = {"problem_type": "binary", "features": contract["featureColumns"], "target_column": "target", "class_labels": ["malignant", "benign"]}
    api.check_data_contract(good, contract)
    wine = {"problem_type": "multiclass", "features": list(load_wine().feature_names), "target_column": "target", "class_labels": ["class_0", "class_1", "class_2"]}
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        api.check_data_contract(wine, contract)


# ---------------------------------------------------------------- deterministic export


def test_deterministic_zip_digest_depends_only_on_names_and_bytes(tmp_path) -> None:
    import os
    import time

    root = tmp_path / "bundle"
    (root / "sub").mkdir(parents=True)
    (root / "b.txt").write_bytes(b"b")
    (root / "sub" / "a.bin").write_bytes(b"a" * 1000)
    first = api.sha256_file(api.deterministic_zip(root, tmp_path / "one.zip"))
    later = time.time() + 3600
    for path in root.rglob("*"):
        os.utime(path, (later, later))
    second = api.sha256_file(api.deterministic_zip(root, tmp_path / "two.zip"))
    assert first == second
    with zipfile.ZipFile(tmp_path / "one.zip") as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ["b.txt", "sub/a.bin"]
    assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)} and {i.external_attr >> 16 for i in infos} == {0o644}
    assert not list(tmp_path.glob("*.tmp"))


def test_export_cell_uses_the_deterministic_zip() -> None:
    code = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "code" and not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    assert "shutil.make_archive(" not in code
    assert "archive = deterministic_zip(active_path, Path(str(archive_base) + '.zip'))" in code


# ---------------------------------------------------------------- XAS5: setup cells and its own cells only


def _defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_section_cells_reference_only_setup_cell_names() -> None:
    cells = _cells()
    setup = [_src(c) for c in cells[: _index(cells, "## 4. ")] if c["cell_type"] == "code"]
    setup_names = set().union(*(_defined(ast.parse(s)) for s in setup))
    section_code = _section_code()
    assert len(section_code) == 3
    own = set().union(*(_defined(ast.parse(s)) for s in section_code))
    used = {n.id for s in section_code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    foreign = used - own - setup_names - set(dir(builtins))
    assert not foreign, f"the section uses names defined outside the setup cells and itself: {sorted(foreign)}"
    assert {"pipe", "sha256_file", "safe_extract_archive", "validate_artifact_directory", "fetch_pinned_artifact"} <= used & setup_names
    # It overwrites none of the main path's names (they keep an ext_/EXT_ prefix).
    main = set().union(*(_defined(ast.parse(_src(c))) for c in cells[_index(cells, "## 4. ") : _index(cells, SECTION_START)] if c["cell_type"] == "code"))
    clobbered = (own & main) - {"shutil", "hashlib", "load_breast_cancer", "train_test_split", "TabularPredictor", "files", "exc", "name", "payload", "class_label", "handle", "p", "c", "_", "label"}  # throwaway and comprehension names
    assert not clobbered, sorted(clobbered)


def test_section_states_its_dependency_trust_boundary_and_difference_from_the_reload() -> None:
    md = "\n".join(_src(c) for c in _section_cells() if c["cell_type"] == "markdown")
    assert "**This section needs only the setup cells (Sections 1–3).**" in md
    assert "the reload shows that this run's export survives serialization, and this section shows that an artifact made elsewhere can be validated and served here" in md
    assert "**Trust boundary.**" in md and "outputs/external_artifact/" in md and "**The download is about 280 MB**" in md
    assert "github.com/kurtvalcorza" not in "\n".join(_section_code())


# ---------------------------------------------------------------- §7.1 order and metadata


def test_section_runs_last_on_the_default_path_after_the_reload() -> None:
    cells = _cells()
    markdown = "\n" + "\n".join(_src(c) for c in cells if c["cell_type"] == "markdown")
    order = ["## 9. Export the deployable predictor bundle, then prove a fresh reload", "## Interpretation and limits", "## Conclusion (your notes)", SECTION_START, "## 11. Optional activity: how do Mitra and the trees compare on a three-class table?", "## Troubleshooting", "## Glossary", "## References"]
    positions = [markdown.index("\n" + heading) for heading in order]
    assert positions == sorted(positions) and all(markdown.count("\n" + heading) == 1 for heading in order)
    assert not any(c["cell_type"] == "code" for c in cells[_index(cells, SECTION_END) :]), "only optional, non-executing material follows Section 10"
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "3.0" and meta["notebook_kind"] == "TUTORIAL"
    assert not list((ROOT / "tutorials").glob("*_predictor_inference_colab.ipynb")) and not list((ROOT / "tutorials").glob("*_artifact_inference_colab.ipynb")), "KND11"


# ---------------------------------------------------------------- XAS6: an end-to-end run of the section's cells


class _Predictor:
    problem_type = "binary"
    class_labels = ["benign", "malignant"]

    def model_names(self):
        return ["Mitra"]


class _Serving:
    def __init__(self, *, predictor, features, device, source, **_):
        self.predictor, self.features, self.device, self.source = predictor, features, device, source
        self.class_labels = list(predictor.class_labels)

    def predict_proba(self, X):
        return pd.DataFrame(np.tile([0.75, 0.25], (len(X), 1)), columns=self.class_labels)

    def predict(self, X):
        return np.array(["benign"] * len(X), dtype=object)


def _standin_bundle(tmp_path: Path) -> tuple[Path, str]:
    """A stand-in bundle with the default path's provenance (no model inside: the predictor is stubbed)."""
    from sklearn.datasets import load_breast_cancer

    table = load_breast_cancer(as_frame=True)
    frame = table.frame.rename(columns={table.target.name: "target"})
    frame["target"] = frame["target"].map({0: "malignant", 1: "benign"})
    data_sha = hashlib.sha256(json.dumps({"sample.csv": hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()}, sort_keys=True).encode()).hexdigest()
    root = tmp_path / "standin"
    root.mkdir()
    (root / "predictor.pkl").write_bytes(b"stand-in: never unpickled here")
    metadata = {
        "artifact_format": api.ARTIFACT_FORMAT, "artifact_format_version": api.ARTIFACT_FORMAT_VERSION,
        "base_model": api.MODEL_ID, "base_model_revision": api.MODEL_REVISION, "weights_sha256": api.WEIGHTS_SHA256, "config_sha256": api.CONFIG_SHA256,
        "autogluon_version": "1.5.0", "python_version": __import__("platform").python_version(), "problem_type": "binary",
        "target_column": "target", "features": list(table.feature_names), "class_labels": ["benign", "malignant"],
        "mode": "pretrained", "selection_basis": "default:pretrained", "seed": 42, "data_source": "Sample: Breast Cancer", "data_sha256": data_sha,
    }
    (root / api.RUN_METADATA).write_text(json.dumps(metadata), encoding="utf-8")
    api.write_artifact_manifest(root)
    archive = api.deterministic_zip(root, tmp_path / "standin.zip")
    return archive, api.sha256_file(archive)


def _namespace(tmp_path: Path, archive: Path, digest: str) -> dict:
    ns = {name: getattr(api, name) for name in dir(api) if not name.startswith("__")}
    ns.update({
        "__builtins__": builtins, "os": __import__("os"), "json": json, "platform": __import__("platform"),
        "importlib": types.SimpleNamespace(metadata=types.SimpleNamespace(version=lambda _name: "1.5.0")),
        "torch": types.SimpleNamespace(__version__="stub"), "NOTEBOOK_SOURCE": {"repository_revision": "test"},
        "pipe": types.SimpleNamespace(model_weight_path="w", config_path="c", snapshot_path="s", device="cpu"),
        "MitraClassificationPipeline": _Serving,
        # Test-only override of the pin: the stand-in's file URL and digest replace the release asset's.
        "SAMPLE_ARTIFACT_URL": archive.resolve().as_uri(), "SAMPLE_ARTIFACT_SHA256": digest,
    })
    return ns


@pytest.fixture()
def _stub_autogluon(monkeypatch):
    fake = types.ModuleType("autogluon.tabular")
    fake.TabularPredictor = types.SimpleNamespace(load=lambda path: _Predictor())
    monkeypatch.setitem(__import__("sys").modules, "autogluon", types.ModuleType("autogluon"))
    monkeypatch.setitem(__import__("sys").modules, "autogluon.tabular", fake)


def test_section_writes_only_under_outputs_external_artifact(tmp_path, monkeypatch, _stub_autogluon) -> None:
    archive, digest = _standin_bundle(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    (work / "outputs").mkdir()
    (work / "outputs" / "mitra_classifier_predictor.zip").write_bytes(b"this run's export: must never be opened here")
    ns = _namespace(tmp_path, archive, digest)
    for source in _section_code():
        exec(compile(source, "<section>", "exec"), ns)
    written = sorted(p.relative_to(work / "outputs").as_posix() for p in (work / "outputs").rglob("*") if p.is_file())
    assert written == ["external_artifact/mitra_classifier_external_predictions.csv", "external_artifact/mitra_classifier_external_result.json", "mitra_classifier_predictor.zip"]
    result = json.loads((work / "outputs" / "external_artifact" / "mitra_classifier_external_result.json").read_text(encoding="utf-8"))
    assert result["artifact"]["zip_sha256"] == digest and result["artifact"]["trusted_digest"] == "verified (pinned SAMPLE_ARTIFACT_SHA256)"
    assert result["input"]["kind"] == "sample" and result["input"]["rows"] == 114 and result["evaluation_report"]["verdict"] == "not-measurable"
    assert [f["verdict"] for f in result["input_manifest"]["findings"] if f["input"] == "missing-column-probe"] == ["rejected"]
    predictions = pd.read_csv(work / "outputs" / "external_artifact" / "mitra_classifier_external_predictions.csv")
    assert list(predictions.columns[-3:]) == ["prediction", "probability_benign", "probability_malignant"] and "target" not in predictions.columns


def test_wrong_pinned_digest_stops_the_section_before_extraction(tmp_path, monkeypatch, _stub_autogluon) -> None:
    archive, _digest = _standin_bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path, archive, "0" * 64)
    ns["safe_extract_archive"] = lambda *a, **k: pytest.fail("extraction must not run after a digest refusal")
    with pytest.raises(ValueError, match="Trusted digest mismatch"):
        exec(compile(_section_code()[0], "<section>", "exec"), ns)
    assert not (tmp_path / "outputs" / "external_artifact").exists()


def test_own_artifact_with_a_wrong_digest_is_refused_before_extraction(tmp_path, monkeypatch, _stub_autogluon) -> None:
    archive, _digest = _standin_bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path, archive, "0" * 64)
    ns["safe_extract_archive"] = lambda *a, **k: pytest.fail("extraction must not run after a digest refusal")
    source = _section_code()[0]
    source = source.replace("ARTIFACT_SOURCE = 'Pinned sample artifact'", "ARTIFACT_SOURCE = 'My own artifact ZIP'", 1)
    source = source.replace("ARTIFACT_ZIP_PATH = ''", f"ARTIFACT_ZIP_PATH = {str(archive)!r}", 1)
    source = source.replace("EXPECTED_ZIP_SHA256 = ''", f"EXPECTED_ZIP_SHA256 = {'0' * 64!r}", 1)
    with pytest.raises(api.Refused, match="EXPECTED_ZIP_SHA256 is 0{64}.*nothing was extracted"):
        exec(compile(source, "<section>", "exec"), ns)
    unverified = _section_code()[0].replace("ARTIFACT_SOURCE = 'Pinned sample artifact'", "ARTIFACT_SOURCE = 'My own artifact ZIP'", 1).replace("ARTIFACT_ZIP_PATH = ''", f"ARTIFACT_ZIP_PATH = {str(archive)!r}", 1)
    with pytest.raises(api.Refused, match="A trusted whole-archive SHA-256 is required"):
        exec(compile(unverified, "<section>", "exec"), ns)


# ---------------------------------------------------------------- ST1: only the exact pinned URL (tabicl PR #29 round 2)

_REPO = "https://github.com/kurtvalcorza/mitra-classifier-pipeline"


@pytest.mark.parametrize(
    "reference",
    [
        f"{_REPO}/releases/download/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-breast-cancer-v1/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/%2e%2e/%2E%2E/archive/refs/heads/main.zip",
        "https://GITHUB.COM/kurtvalcorza/mitra-classifier-pipeline/releases/download/%2E%2E/archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip?raw=1",
        f"{_REPO}/releases/download/sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip#x",
        "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip",
        f"{_REPO}/releases/download/sample-bundle-v1/mitra_classifier_predictor.zip",
        f"{_REPO}/releases/download/sample-artifact-breast-cancer-v2/mitra_classifier_predictor.zip",
        f"{_REPO}/releases/download/sample-artifact-breast-cancer-v1/mitra_classifier_predictor.zip.sig",
        f"{_REPO}.git",
    ],
    ids=["traversal", "tag-then-traversal", "pct2e", "pct2E-uppercase-host", "query", "fragment", "other-repo", "other-tag", "next-tag", "suffix", "clone"],
)
def test_st1_only_the_exact_pinned_release_asset_is_allowed(reference: str) -> None:
    source = f'SAMPLE_ARTIFACT_URL = "{reference}"\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == [reference]


def test_st1_exact_pinned_url_passes_and_the_real_notebook_validates() -> None:
    source = f'SAMPLE_ARTIFACT_URL = "{PINNED_URL}"  # literal\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    assert VALIDATOR.disallowed_own_references(source) == [PINNED_URL], "the notebook's own cells allow no repository reference"
    carried = "\n".join(_src(c) for c in _cells() if c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    assert PINNED_URL in carried
    assert VALIDATOR.disallowed_own_references(carried, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    VALIDATOR.validate_notebooks()
