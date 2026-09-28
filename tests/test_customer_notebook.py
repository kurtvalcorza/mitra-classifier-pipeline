"""The distributable notebook must match its embedded, standalone source."""
import ast
import importlib
import json
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
builder = importlib.import_module("build_customer_capstone")


def test_notebook_schema_embedded_sources_and_generated_parity():
    notebook = builder.build()
    nbformat.validate(nbformat.from_dict(notebook))
    assert json.loads(builder.NOTEBOOK.read_text(encoding="utf-8")) == notebook
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for name, content in builder.carried_files().items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    assert notebook["metadata"]["dimer"]["standalone"] is True


def test_hash_lock_has_pins_and_distribution_hashes():
    lock = builder.carried_files()["requirements.txt"]
    assert "--hash=sha256:" in lock
    for entry in lock.replace("\\\n", " ").splitlines():
        if entry and not entry.startswith(("#", " ", "--")):
            assert "==" in entry and "--hash=sha256:" in entry
