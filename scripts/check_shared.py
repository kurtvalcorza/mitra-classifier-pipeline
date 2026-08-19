#!/usr/bin/env python3
"""Enforce that the canonical dataset-resolution + archive-safety block is byte-identical
across the validator and finetuner containers.

Findings #2/#3 (validator and finetuner must resolve the same file the same way) are only
safe if both containers carry the *same* implementation. The block is duplicated by necessity
(DIMER builds one image per repo), so this check keeps the duplicates in lock-step. CI runs it;
it exits non-zero on any drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

START = "# CANONICAL DATASET RESOLUTION + ARCHIVE SAFETY"
END = "# END shared block"

FILES = [
    Path("validator/validator.py"),
    Path("finetuner/train.py"),
]


def extract_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{path}: shared-block markers not found")
    body = text.split(START, 1)[1].split(END, 1)[0]
    return body.strip("\n")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    blocks = {f: extract_block(root / f) for f in FILES}
    reference = next(iter(blocks.values()))
    drift = [str(f) for f, b in blocks.items() if b != reference]
    if drift:
        print(f"Shared block DRIFT in: {', '.join(drift)}")
        print("The canonical dataset-resolution block must be identical across containers.")
        return 1
    print(f"Shared block identical across {len(blocks)} files ({len(reference.splitlines())} lines).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
