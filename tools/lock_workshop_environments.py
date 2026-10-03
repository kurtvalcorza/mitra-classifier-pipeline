"""Write the hash-locked foundation-model environments used by the FreshRetailNet classification workshops.

Both workshop editions build one isolated uv environment per foundation-model stack (Section 5.2). This tool
turns each stack's top-level pins, read from the v1 notebook's ``MODEL_ENVIRONMENT_DEPENDENCIES``, into
``tutorials/requirements-workshop-<stack>.lock.txt``: every transitive package pinned, with SHA-256 hashes for
the wheels a Linux x86_64 CPython 3.12 interpreter can install. Hashes of other platforms' files are dropped so
the notebooks stay small; the install refuses any file whose hash is not listed.

A package with no wheel at all must be named in ``SOURCE_BUILT``. It is moved to a separate
``requirements-workshop-<stack>-sdist.lock.txt`` that pins its source archive by hash; the notebook installs it
after the wheels, with the locked setuptools and no build isolation, so no unhashed build tool is fetched.

Needs network access to PyPI and ``uv`` on PATH. It is a maintainer tool, not part of CI. Run from the
repository root:  python tools/lock_workshop_environments.py
"""

# ruff: noqa: E501
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop.ipynb"
UV_VERSION = "0.12.15"
PYTHON_VERSION = "3.12.12"
PLATFORM = "x86_64-manylinux_2_28"
INDEX_URL = "https://pypi.org/simple"
# Packages published only as source archives. Each entry is a deliberate, reviewed exception.
SOURCE_BUILT = {"tabdpt": {"antlr4-python3-runtime"}}
# Linux x86_64 glibc platforms a hosted runtime (Ubuntu 22.04 or later) can install.
LINUX_PLATFORMS = {f"manylinux_2_{minor}_x86_64" for minor in range(5, 40)} | {
    "manylinux1_x86_64",
    "manylinux2010_x86_64",
    "manylinux2014_x86_64",
}


def notebook_dependencies() -> dict[str, list[str]]:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if "MODEL_ENVIRONMENT_DEPENDENCIES = " not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "MODEL_ENVIRONMENT_DEPENDENCIES" for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise SystemExit("MODEL_ENVIRONMENT_DEPENDENCIES not found in the v1 workshop notebook")


def wheel_installable(filename: str) -> bool:
    """True when a CPython 3.12 interpreter on Linux x86_64 (glibc) can install this wheel."""
    if not filename.endswith(".whl"):
        return False
    python_tags, abi_tags, platform_tags = filename[:-4].split("-")[-3:]
    platforms = set(platform_tags.split("."))
    if not (platforms & (LINUX_PLATFORMS | {"any"})):
        return False
    for python_tag in python_tags.split("."):
        for abi in abi_tags.split("."):
            if abi == "none" and python_tag in ("py3", "py312", "cp312"):
                return True
            if abi == "cp312" and python_tag == "cp312":
                return True
            stable = re.fullmatch(r"cp3(\d+)", python_tag)
            if abi == "abi3" and stable and int(stable.group(1)) <= 12:
                return True
    return False


def release_files(name: str, version: str) -> list[dict]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
        return json.load(response)["urls"]


def parse_lock(text: str) -> list[tuple[str, str, list[str]]]:
    """Return (name, version, hashes) for each requirement in a ``uv pip compile --generate-hashes`` file."""
    entries = []
    for block in re.split(r"\n(?=\S)", text):
        match = re.match(r"([A-Za-z0-9_.\-\[\]]+)==([^\s\\;]+)", block)
        if match:
            entries.append((match.group(1), match.group(2), re.findall(r"--hash=sha256:([0-9a-f]{64})", block)))
    return entries


def render(entries: list[tuple[str, str, list[str]]]) -> str:
    return "".join(
        f"{name}=={version} \\\n" + " \\\n".join(f"    --hash=sha256:{digest}" for digest in hashes) + "\n"
        for name, version, hashes in entries
    )


def lock(stack: str, pins: list[str]) -> None:
    allowed_sources = SOURCE_BUILT.get(stack, set())
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / f"{stack}.in"
        requirements.write_text("\n".join(pins) + "\n", encoding="utf-8")
        output = Path(scratch) / f"{stack}.txt"
        command = ["uv", "pip", "compile", str(requirements), "--generate-hashes", "--no-header", "--no-annotate",
                   "--python-platform", PLATFORM, "--python-version", PYTHON_VERSION, "--index-url", INDEX_URL,
                   "--only-binary", ":all:", "-o", str(output), "-q"]
        for package in sorted(allowed_sources):
            command += ["--no-binary", package]
        subprocess.run(command, check=True)
        entries = parse_lock(output.read_text(encoding="utf-8"))
    wheels, sources = [], []
    for name, version, hashes in entries:
        files = release_files(name, version)
        if name in allowed_sources:
            sdist = [f["digests"]["sha256"] for f in files if f["packagetype"] == "sdist"]
            if not sdist or not set(sdist) <= set(hashes):
                raise SystemExit(f"{stack}: no hashed sdist for {name}=={version}")
            sources.append((name, version, sdist))
            continue
        keep = [f["digests"]["sha256"] for f in files if wheel_installable(f["filename"])]
        keep = [digest for digest in hashes if digest in keep]
        if not keep:
            raise SystemExit(f"{stack}: {name}=={version} has no Linux x86_64 CPython 3.12 wheel")
        wheels.append((name, version, keep))
    if {name for name, _, _ in sources} != allowed_sources:
        raise SystemExit(f"{stack}: source-built packages {sources} differ from the reviewed list {allowed_sources}")
    header = [
        f"# Hash-locked '{stack}' foundation-model environment for the FreshRetailNet classification workshops (v1 and v2).",
        "# Generated by tools/lock_workshop_environments.py; do not edit by hand.",
        f"# Target: CPython {PYTHON_VERSION} on Linux x86_64 (glibc), created by uv {UV_VERSION}",
        "# (uv venv --managed-python) and installed with",
        "#   uv pip install --require-hashes --only-binary :all: --no-deps -r <this file>",
        "# Top-level pins are the notebook's MODEL_ENVIRONMENT_DEPENDENCIES entry, unchanged from the",
        "# earlier unlocked install:",
        *(f"#   {pin}" for pin in pins),
        f"# Transitive pins resolved with uv pip compile --python-platform {PLATFORM} --python-version {PYTHON_VERSION}",
        "# --only-binary :all:. Hashes are kept only for wheels a Linux x86_64 CPython 3.12 interpreter can install.",
    ]
    if sources:
        header += [
            "# Not in this file (no wheel is published): "
            + ", ".join(f"{name}=={version}" for name, version, _ in sources) + ".",
            f"# It is pinned by source-archive hash in requirements-workshop-{stack}-sdist.lock.txt and built after",
            "# these wheels with the locked setuptools and --no-build-isolation.",
        ]
    target = ROOT / "tutorials" / f"requirements-workshop-{stack}.lock.txt"
    target.write_text("\n".join(header) + "\n" + render(wheels), encoding="utf-8", newline="\n")
    print(f"{target.name}: {len(wheels)} wheels")
    if sources:
        sdist_target = ROOT / "tutorials" / f"requirements-workshop-{stack}-sdist.lock.txt"
        sdist_header = [
            f"# Source-built package for the '{stack}' workshop environment: no wheel is published for it.",
            "# Generated by tools/lock_workshop_environments.py; do not edit by hand.",
            "# Installed after requirements-workshop-" + stack + ".lock.txt with",
            "#   uv pip install --require-hashes --no-deps --no-build-isolation -r <this file>",
            "# so the hash-verified source archive is built by the locked setuptools, not a downloaded build tool.",
        ]
        sdist_target.write_text("\n".join(sdist_header) + "\n" + render(sources), encoding="utf-8", newline="\n")
        print(f"{sdist_target.name}: {len(sources)} source archive(s)")


def main() -> int:
    for stack, pins in notebook_dependencies().items():
        lock(stack, pins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
