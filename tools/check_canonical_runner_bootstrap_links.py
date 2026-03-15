#!/usr/bin/env python3
"""Check canonical runner bootstrap resources for the packaged WS-OVVIS repo.

This version is intentionally conservative for G0:
- vendored local components are checked for existence, not forced to be sibling symlinks,
- externally supplied resources are also checked for existence,
- the tool reports mismatches but does not silently redefine canonical validation semantics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_WRONG_TYPE = "WRONG_TYPE"


@dataclass(frozen=True)
class PathSpec:
    rel_path: str
    kind: str  # dir or any
    required: bool = True


SPECS: tuple[PathSpec, ...] = (
    PathSpec("third_party/CutLER", "dir"),
    PathSpec("third_party/VNext", "dir"),
    PathSpec("third_party/dinov2", "dir"),
    PathSpec("data", "dir"),
    PathSpec("weights", "dir"),
    PathSpec("runs", "dir"),
)


def inspect(root: Path, spec: PathSpec) -> tuple[str, str]:
    p = root / spec.rel_path
    if not p.exists():
        return STATUS_MISSING, "path missing"
    if spec.kind == "dir" and not p.is_dir():
        return STATUS_WRONG_TYPE, "expected directory"
    return STATUS_OK, "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-root", type=Path, default=Path.cwd())
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root = args.runner_root.resolve()
    print("MODE=CHECK")
    print(f"RUNNER_ROOT={root}")
    print("BOOTSTRAP_LINK_CHECK_BEGIN")
    print("path	status	note")
    rc = 0
    for spec in SPECS:
        status, note = inspect(root, spec)
        print(f"{spec.rel_path}	{status}	{note}")
        if status != STATUS_OK and spec.required:
            rc = 1
    print("BOOTSTRAP_LINK_CHECK_END")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
