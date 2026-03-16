#!/usr/bin/env python3
"""Manifest-driven watcher for long-running training or evaluation jobs.

This helper is intentionally generic:
- it polls for completion conditions,
- writes durable watcher-state output,
- optionally runs post-completion commands,
- never decides scientific PASS by itself.

Typical usage:
  nohup python tools/watch_training_job.py --manifest codex/job_watch.json > watcher.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs/mainline/reports/training_watch_latest.txt"
DEFAULT_ARCHIVE = ROOT / "docs/mainline/reports/archive"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_archive(base_report: Path, text: str) -> None:
    archive_dir = base_report.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = base_report.stem
    suffix = base_report.suffix or ".txt"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_text(archive_dir / f"{stem}_{ts}{suffix}", text)


def path_from_manifest(raw: str | None, manifest_dir: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (manifest_dir / p)


def file_contains(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    try:
        return pattern in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def latest_mtime(paths: list[Path]) -> float | None:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def run_shell_commands(commands: list[str], cwd: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cmd in commands:
        cp = subprocess.run(cmd, cwd=str(cwd), shell=True, text=True, capture_output=True)
        results.append(
            {
                "command": cmd,
                "returncode": cp.returncode,
                "stdout": cp.stdout[-4000:],
                "stderr": cp.stderr[-4000:],
            }
        )
        if cp.returncode != 0:
            break
    return results


@dataclass
class Evaluation:
    state: str
    reason: str
    tracked_paths: list[Path]


def evaluate_completion(manifest: dict[str, Any], manifest_dir: Path) -> Evaluation:
    watch = manifest.get("watch", {})
    tracked: list[Path] = []

    exist_paths = [path_from_manifest(p, manifest_dir) for p in watch.get("all_exist", [])]
    exist_paths = [p for p in exist_paths if p is not None]
    tracked.extend(exist_paths)
    if exist_paths and not all(p.exists() for p in exist_paths):
        return Evaluation("running", "waiting for required artifact paths", tracked)

    contains_specs = watch.get("all_contains", [])
    for spec in contains_specs:
        p = path_from_manifest(spec.get("path"), manifest_dir)
        if p:
            tracked.append(p)
            if not file_contains(p, str(spec.get("pattern", ""))):
                return Evaluation("running", f"waiting for pattern in {p}", tracked)

    pid_file = path_from_manifest(watch.get("pid_file"), manifest_dir)
    if pid_file:
        tracked.append(pid_file)
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                return Evaluation("running", f"process {pid} still alive", tracked)
            except ProcessLookupError:
                pass
            except Exception:
                return Evaluation("running", "pid file present but process state unreadable", tracked)

    stale_minutes = watch.get("stale_after_minutes")
    if stale_minutes:
        recent = latest_mtime(tracked)
        if recent is not None and (time.time() - recent) > int(stale_minutes) * 60:
            return Evaluation("stale", f"tracked paths unchanged for > {stale_minutes} minutes", tracked)

    return Evaluation("completed", "all completion conditions satisfied", tracked)


def render_report(manifest: dict[str, Any], state: str, reason: str, action_results: list[dict[str, Any]], manifest_path: Path) -> str:
    gate = manifest.get("scientific_gate", "UNKNOWN")
    eng = manifest.get("engineering_gates", [])
    eng_text = ", ".join(eng) if eng else "[]"
    lines = [
        f"timestamp: {now()}",
        f"manifest: {manifest_path}",
        f"name: {manifest.get('name', manifest_path.stem)}",
        f"scientific_gate: {gate}",
        f"engineering_gates: {eng_text}",
        f"state: {state}",
        f"reason: {reason}",
        "",
        "post_actions:",
    ]
    if not action_results:
        lines.append("- none")
    else:
        for item in action_results:
            lines.append(f"- command: {item['command']}")
            lines.append(f"  returncode: {item['returncode']}")
            if item.get("stdout"):
                lines.append("  stdout_tail:")
                for ln in item["stdout"].splitlines():
                    lines.append(f"    {ln}")
            if item.get("stderr"):
                lines.append("  stderr_tail:")
                for ln in item["stderr"].splitlines():
                    lines.append(f"    {ln}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON manifest path")
    ap.add_argument("--poll-seconds", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = read_json(manifest_path)
    poll_seconds = args.poll_seconds or int(manifest.get("poll_seconds", 120))
    report_path = path_from_manifest(manifest.get("report_path"), manifest_dir) or DEFAULT_REPORT
    cwd = path_from_manifest(manifest.get("working_dir"), manifest_dir) or ROOT

    completed_once = False
    while True:
        manifest = read_json(manifest_path)
        ev = evaluate_completion(manifest, manifest_dir)
        post_actions: list[dict[str, Any]] = []

        if ev.state == "completed" and not completed_once:
            post_actions = run_shell_commands(list(manifest.get("on_complete_commands", [])), cwd)
            if post_actions and any(item["returncode"] != 0 for item in post_actions):
                ev = Evaluation("post-complete-failed", "a post-completion command failed", ev.tracked_paths)
            completed_once = True
        elif ev.state in {"stale", "failed"}:
            post_actions = run_shell_commands(list(manifest.get("on_fail_commands", [])), cwd)

        report = render_report(manifest, ev.state, ev.reason, post_actions, manifest_path)
        write_text(report_path, report)
        append_archive(report_path, report)

        state_json_path = path_from_manifest(manifest.get("state_json_path"), manifest_dir)
        if state_json_path:
            state_json = {
                "timestamp": now(),
                "manifest": str(manifest_path),
                "state": ev.state,
                "reason": ev.reason,
                "completed_once": completed_once,
            }
            write_text(state_json_path, json.dumps(state_json, ensure_ascii=False, indent=2))

        if args.once or ev.state in {"completed", "post-complete-failed", "stale", "failed"}:
            return 0 if ev.state == "completed" else 1
        time.sleep(max(5, poll_seconds))


if __name__ == "__main__":
    sys.exit(main())
