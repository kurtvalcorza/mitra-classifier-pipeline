#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

here = Path(__file__).resolve().parent
source = (here / "_harden_notebook_spec_v1.py").read_text(encoding="utf-8")
source = source.replace("DIMER_RELEASE_MANIFEST", "RELEASE_PACKAGE_MANIFEST")
compiled = compile(source, str(here / "_harden_notebook_spec_v1.py"), "exec")
namespace = {"__name__": "__main__", "__file__": str(here / "_harden_notebook_spec_v1.py")}
exec(compiled, namespace)
