#!/usr/bin/env python3
"""Check/fix canonical fresh-runner bootstrap symlink dependencies.

Managed linkage convention:
- third_party/CutLER -> ../../wsovvis_live/third_party/CutLER
- third_party/dinov2 -> ../../wsovvis_live/third_party/dinov2
- runs -> ../wsovvis_live/runs
- weights -> ../wsovvis_live/weights
- data -> ../wsovvis_live/data
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_BROKEN = "BROKEN_SYMLINK"
STATUS_WRONG = "WRONG_TARGET"
STATUS_FIXED = "FIXED"
STATUS_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class LinkSpec:
    rel_path: str
    expected_target: str


SPECS: tuple[LinkSpec, ...] = (
    LinkSpec("third_party/CutLER", "../../wsovvis_live/third_party/CutLER"),
    LinkSpec("third_party/dinov2", "../../wsovvis_live/third_party/dinov2"),
    LinkSpec("runs", "../wsovvis_live/runs"),
    LinkSpec("weights", "../wsovvis_live/weights"),
    LinkSpec("data", "../wsovvis_live/data"),
)


@dataclass
class LinkResult:
    rel_path: str
    status: str
    expected_target: str
    actual_target: str | None
    note: str = ""


def _norm_target(target: str) -> str:
    return target.rstrip("/")


def inspect_link(root: Path, spec: LinkSpec) -> LinkResult:
    link_path = root / spec.rel_path
    expected = _norm_target(spec.expected_target)

    if not os.path.lexists(link_path):
        return LinkResult(spec.rel_path, STATUS_MISSING, expected, None)

    if not link_path.is_symlink():
        return LinkResult(
            spec.rel_path,
            STATUS_WRONG,
            expected,
            "<not-a-symlink>",
            "path exists but is not a symlink",
        )

    actual_raw = os.readlink(link_path)
    actual = _norm_target(actual_raw)
    if actual != expected:
        return LinkResult(spec.rel_path, STATUS_WRONG, expected, actual_raw)

    resolved_target = (link_path.parent / actual_raw).resolve(strict=False)
    if not resolved_target.exists():
        return LinkResult(spec.rel_path, STATUS_BROKEN, expected, actual_raw)

    return LinkResult(spec.rel_path, STATUS_OK, expected, actual_raw)


def fix_link(root: Path, spec: LinkSpec, current: LinkResult) -> LinkResult:
    link_path = root / spec.rel_path
    expected = spec.expected_target

    if link_path.exists() and not link_path.is_symlink():
        return LinkResult(
            spec.rel_path,
            STATUS_SKIPPED,
            _norm_target(expected),
            "<not-a-symlink>",
            "refusing to replace non-symlink path",
        )

    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link_path):
        link_path.unlink()
    link_path.symlink_to(expected)

    post = inspect_link(root, spec)
    if post.status == STATUS_OK:
        return LinkResult(spec.rel_path, STATUS_FIXED, post.expected_target, expected)
    return LinkResult(
        spec.rel_path,
        STATUS_SKIPPED,
        post.expected_target,
        post.actual_target,
        f"fix attempted but verification ended in {post.status}",
    )


def render_results(results: Iterable[LinkResult]) -> None:
    print("BOOTSTRAP_LINK_CHECK_BEGIN")
    print("path\tstatus\texpected\tactual\tnote")
    for res in results:
        actual = res.actual_target if res.actual_target is not None else "-"
        note = res.note if res.note else "-"
        print(f"{res.rel_path}\t{res.status}\t{res.expected_target}\t{actual}\t{note}")
    print("BOOTSTRAP_LINK_CHECK_END")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check/fix canonical runner bootstrap symlink dependencies."
    )
    parser.add_argument(
        "--runner-root",
        type=Path,
        default=Path.cwd(),
        help="Runner path to inspect/fix (default: current working directory).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode (default when neither --check nor --fix is supplied).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix missing/broken/wrong-target symlinks using wsovvis_live convention.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner_root = args.runner_root.resolve()
    mode_fix = bool(args.fix)
    mode_check = bool(args.check) or not mode_fix

    if mode_check:
        print(f"MODE=CHECK")
    if mode_fix:
        print(f"MODE=FIX")
    print(f"RUNNER_ROOT={runner_root}")

    if not runner_root.exists() or not runner_root.is_dir():
        print(f"ERROR: runner root does not exist or is not a directory: {runner_root}")
        return 2

    results: list[LinkResult] = []
    for spec in SPECS:
        current = inspect_link(runner_root, spec)
        if mode_fix and current.status in {STATUS_MISSING, STATUS_BROKEN, STATUS_WRONG}:
            current = fix_link(runner_root, spec, current)
        results.append(current)

    render_results(results)

    unresolved = [r for r in results if r.status in {STATUS_MISSING, STATUS_BROKEN, STATUS_WRONG}]
    if unresolved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
