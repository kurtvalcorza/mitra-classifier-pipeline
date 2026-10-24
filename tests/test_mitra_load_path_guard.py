"""Guard: Mitra fit stays on the safetensors path, so tab2d.py's `torch.load(weights_only=False)` stays dead.

AutoGluon 1.5.0 has two `torch.load(weights_only=False)` sites in Mitra:

- `_internal/models/tab2d.py` (`Tab2D.__init__`, CPU branch): reached only when `sklearn_interface.USE_HF`
  is false and a `state_dict_classification` / `state_dict_regression` path is passed. With `USE_HF = True`
  (hard-coded at `sklearn_interface.py:56`), fit builds the model with `Tab2D.from_pretrained`, which reads
  `model.safetensors` with `safetensors.load_file`, so this site is dead code.
- `mitra_model.py:239` (`_load_model_artifact`): reads a saved bundle's `models/Mitra/model.pt`. It is
  documented as code-capable next to `predictor.pkl` in the notebook's trust-boundary text (SER9); it is not
  guarded here.

No test here touches the network. `test_use_hf_is_true_on_import` needs a full AutoGluon install and is
skipped without one. `test_pinned_wheel_source_keeps_fit_on_safetensors` reads the source of the installed
`autogluon.tabular` distribution without importing it, so CI installs that wheel with `--no-deps` and sets
`MITRA_GUARD_REQUIRE_AUTOGLUON_SOURCE=1`, which turns a missing wheel into a failure instead of a skip.
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "USE_HF",
    "state_dict_classification",
    "state_dict_regression",
    "hf_cls_model",
    "hf_reg_model",
)
SCANNED_SUFFIXES = {".py", ".ipynb", ".json", ".toml", ".yml", ".yaml", ".sh", ".cfg"}
SKIPPED_DIRS = {
    ".git",
    "docs",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
MITRA_PKG = "autogluon/tabular/models/mitra"


def _pinned_autogluon_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"autogluon\.tabular\[mitra\]==([0-9][^"]*)"', text)
    assert match, "pyproject.toml must pin autogluon.tabular[mitra] exactly"
    return match.group(1)


def _mitra_sources() -> dict[str, str]:
    """Return {relative path: source} for the installed Mitra package, without importing it."""
    try:
        dist = importlib.metadata.distribution("autogluon.tabular")
    except importlib.metadata.PackageNotFoundError:
        if os.environ.get("MITRA_GUARD_REQUIRE_AUTOGLUON_SOURCE") == "1":
            pytest.fail("autogluon.tabular is not installed; CI installs the pinned wheel with --no-deps")
        pytest.skip("autogluon.tabular is not installed")
    assert dist.version == _pinned_autogluon_version(), (
        f"installed autogluon.tabular {dist.version} differs from the pyproject pin; "
        "re-check the Mitra load paths"
    )
    sources = {}
    for entry in dist.files or ():
        rel = str(entry).replace("\\", "/")
        if rel.startswith(MITRA_PKG + "/") and rel.endswith(".py"):
            sources[rel] = Path(dist.locate_file(entry)).read_text(encoding="utf-8")
    assert f"{MITRA_PKG}/sklearn_interface.py" in sources, "Mitra sklearn interface not found in the wheel"
    return sources


def _is_tab2d_with_pretrained_weights(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Tab2D"):
        return False
    for kw in node.keywords:
        if kw.arg == "use_pretrained_weights":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
    return False


def _calls_outside_use_hf_else(tree: ast.AST) -> list[int]:
    """Lines of `Tab2D(..., use_pretrained_weights=<not False>)` calls outside the else of `if USE_HF:`."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "USE_HF":
            for stmt in node.orelse:
                guarded.update(id(sub) for sub in ast.walk(stmt))
    return [
        node.lineno
        for node in ast.walk(tree)
        if _is_tab2d_with_pretrained_weights(node) and id(node) not in guarded
    ]


def test_use_hf_is_true_on_import():
    sklearn_interface = pytest.importorskip("autogluon.tabular.models.mitra.sklearn_interface")
    assert sklearn_interface.USE_HF is True


def test_pinned_wheel_source_keeps_fit_on_safetensors():
    sources = _mitra_sources()
    tree = ast.parse(sources[f"{MITRA_PKG}/sklearn_interface.py"])
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "USE_HF" for target in node.targets)
    ]
    assert len(assignments) == 1, "USE_HF must be assigned exactly once in sklearn_interface.py"
    value = assignments[0].value
    assert isinstance(value, ast.Constant) and value.value is True, "USE_HF must stay hard-coded to True"
    assert assignments[0] in tree.body, "USE_HF must be a module-level constant"

    tab2d = ast.parse(sources[f"{MITRA_PKG}/_internal/models/tab2d.py"])
    from_pretrained = [
        node
        for node in ast.walk(tab2d)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    ]
    assert from_pretrained, "Tab2D.from_pretrained is gone; re-check how fit loads the checkpoint"
    body = ast.unparse(from_pretrained[0]).replace("'", '"')
    assert 'filename="model.safetensors"' in body and "load_file(" in body, (
        "Tab2D.from_pretrained must read model.safetensors with safetensors.load_file"
    )

    offenders = {}
    for path, src in sources.items():
        lines = _calls_outside_use_hf_else(ast.parse(src))
        if lines:
            offenders[path] = lines
    assert not offenders, f"Tab2D built from a torch.load path outside `if USE_HF: ... else:`: {offenders}"


def test_repository_never_selects_the_state_dict_path():
    this_file = Path(__file__).resolve()
    hits = []
    for path in sorted(ROOT.rglob("*")):
        rel_parts = path.relative_to(ROOT).parts
        if path.is_dir() or path.suffix not in SCANNED_SUFFIXES or SKIPPED_DIRS.intersection(rel_parts):
            continue
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        hits.extend(f"{rel}: {token}" for token in FORBIDDEN if token in text)
    assert not hits, (
        f"repository code must not set USE_HF or pass a state_dict_*/hf_*_model hyperparameter: {hits}"
    )
