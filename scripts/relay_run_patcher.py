# Temporary bootstrap for the compressed Agent Relay patcher. Self-deletes with the relay workflow.
from __future__ import annotations

import ast
import base64
import sys
import zlib
from pathlib import Path

PATCHER = Path(__file__).with_name("relay_builder_fix_all.py")
source = PATCHER.read_text(encoding="utf-8")
tree = ast.parse(source)
payload = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PAYLOAD":
                payload = ast.literal_eval(node.value)
                break
    if payload is not None:
        break
if not isinstance(payload, str):
    raise RuntimeError("relay patch payload not found")
raw = base64.b64decode(payload + "=" * (-len(payload) % 4))
program = zlib.decompress(raw).decode("utf-8")
sys.argv = [str(PATCHER), *sys.argv[1:]]
exec(compile(program, str(PATCHER), "exec"))
