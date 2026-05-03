#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REMOTE_HOST = "gpu4090d"
REMOTE_REPO_ROOTS = [
    Path("/mnt/sda/zyy/code/wsovvis"),
    Path("/home/zyy/code/wsovvis"),
]
LOCAL_REPO_ROOT = Path("/mnt/e/code/wsovvis")
TARGET_ROOTS = [
    Path("codex/outputs/G8_inference_and_eval/vc_full_y_nohub_validation_15ep_20260502"),
    Path("codex/outputs/G8_inference_and_eval/gt_clean_base_overfit_capacity_20260502"),
    Path("codex/outputs/G8_inference_and_eval/gt_clean_weak_fully_overfit_capacity_20260502"),
]
OUTPUT_ROOT = LOCAL_REPO_ROOT / "codex/outputs/G8_inference_and_eval/asset_sync_20260502"


@dataclass(frozen=True)
class Entry:
    experiment_root: str
    relative_path: str
    entry_type: str
    size_bytes: int
    mtime_epoch: float
    link_target: str
    abs_path: str

    @property
    def is_symlink(self) -> bool:
        return self.entry_type == "symlink"


def run(cmd: List[str], *, capture_output: bool = True, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture_output, check=check, text=text)


def remote_cmd(command: str) -> str:
    proc = run(["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(command)}"])
    return proc.stdout


def remote_path_exists(path: Path) -> bool:
    cmd = f"find {shlex.quote(str(path))} -maxdepth 0 -print >/dev/null"
    proc = subprocess.run(["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(cmd)}"], capture_output=True)
    return proc.returncode == 0


def choose_remote_root(experiment_root: Path) -> Optional[Path]:
    for base in REMOTE_REPO_ROOTS:
        candidate = base / experiment_root
        if remote_path_exists(candidate):
            return candidate
    return None


def find_entries(root: Path, experiment_root: str) -> List[Entry]:
    if not root.exists() and not remote_path_exists(root):
        return []
    if root.as_posix().startswith("/mnt/e/"):
        cmd = [
            "find",
            str(root),
            "-printf",
            "%P\t%y\t%s\t%T@\t%l\n",
        ]
        proc = run(cmd)
        stdout = proc.stdout
    else:
        quoted = shlex.quote(str(root))
        cmd = f"find {quoted} -printf '%P\\t%y\\t%s\\t%T@\\t%l\\n'"
        stdout = remote_cmd(cmd)
    entries: List[Entry] = []
    for line in stdout.splitlines():
        if not line:
            continue
        if "\t" not in line:
            continue
        rel, typ, size, mtime, target = line.split("\t", 4)
        rel = rel.lstrip("./")
        if rel == "":
            continue
        if typ == "d":
            entry_type = "dir"
        elif typ == "l":
            entry_type = "symlink"
        else:
            entry_type = "file"
        entries.append(
            Entry(
                experiment_root=experiment_root,
                relative_path=rel,
                entry_type=entry_type,
                size_bytes=int(float(size)),
                mtime_epoch=float(mtime),
                link_target=target,
                abs_path=str(root / rel),
            )
        )
    return entries


def local_entries(root: Path, experiment_root: str) -> List[Entry]:
    if not root.exists():
        return []
    cmd = [
        "find",
        str(root),
        "-printf",
        "%P\t%y\t%s\t%T@\t%l\n",
    ]
    proc = run(cmd)
    entries: List[Entry] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        if "\t" not in line:
            continue
        rel, typ, size, mtime, target = line.split("\t", 4)
        rel = rel.lstrip("./")
        if rel == "":
            continue
        if typ == "d":
            entry_type = "dir"
        elif typ == "l":
            entry_type = "symlink"
        else:
            entry_type = "file"
        entries.append(
            Entry(
                experiment_root=experiment_root,
                relative_path=rel,
                entry_type=entry_type,
                size_bytes=int(float(size)),
                mtime_epoch=float(mtime),
                link_target=target,
                abs_path=str(root / rel),
            )
        )
    return entries


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_sha256(path: Path) -> Optional[str]:
    if not path.exists() or path.is_symlink() or not path.is_file():
        return None
    size = path.stat().st_size
    if size > 300_000:
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_key(experiment_root: str, relative_path: str) -> Tuple[str, str]:
    return experiment_root, relative_path


def conflict_suffix() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def rsync_pull(remote_root: Path, relpaths: List[str], local_root: Path) -> None:
    if not relpaths:
        return
    filelist = OUTPUT_ROOT / "sync_plan" / f"files_from_{abs(hash((str(remote_root), str(local_root), len(relpaths))))}.txt"
    filelist.parent.mkdir(parents=True, exist_ok=True)
    with filelist.open("w") as f:
        for rel in relpaths:
            f.write(rel + "\n")
    cmd = [
        "rsync",
        "-aH",
        "--relative",
        "--files-from",
        str(filelist),
        f"{REMOTE_HOST}:{str(remote_root)}/",
        f"{str(local_root)}/",
    ]
    run(cmd, capture_output=False)


def rsync_copy_single(remote_root: Path, relpath: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-aH", f"{REMOTE_HOST}:{str(remote_root / relpath)}", str(dst)]
    run(cmd, capture_output=False)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sync_plan_dir = OUTPUT_ROOT / "sync_plan"
    sync_plan_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "remote_host": REMOTE_HOST,
        "remote_roots_checked": [str(p) for p in REMOTE_REPO_ROOTS],
        "experiment_roots_checked": [str(p) for p in TARGET_ROOTS],
        "local_repo_root": str(LOCAL_REPO_ROOT),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots": [],
    }

    remote_rows: List[dict] = []
    local_rows_before: List[dict] = []
    local_rows_after: List[dict] = []
    missing_files: List[dict] = []
    missing_symlinks: List[dict] = []
    conflict_rows: List[dict] = []
    unresolved_symlinks: List[dict] = []
    pulled_rows: List[dict] = []
    verification_missing: List[dict] = []

    total_remote_files = 0
    total_local_files_before = 0
    missing_local_file_count = 0
    missing_local_symlink_count = 0
    missing_local_dir_count = 0
    conflict_count = 0
    pulled_file_count = 0
    pulled_symlink_count = 0
    large_file_pulled_count = 0
    unresolved_symlink_count = 0

    remote_entries_map: Dict[Tuple[str, str], Entry] = {}
    local_entries_map_before: Dict[Tuple[str, str], Entry] = {}

    for experiment_root in TARGET_ROOTS:
        remote_root = choose_remote_root(experiment_root)
        local_root = LOCAL_REPO_ROOT / experiment_root
        root_info = {
            "experiment_root": str(experiment_root),
            "remote_root": str(remote_root) if remote_root else None,
            "local_root": str(local_root),
            "remote_exists": bool(remote_root),
            "remote_count": 0,
            "local_count_before": 0,
            "missing_files": 0,
            "missing_symlinks": 0,
            "conflicts": 0,
            "missing_dirs": 0,
            "pulled_files": 0,
            "pulled_symlinks": 0,
        }
        plan["roots"].append(root_info)

        local_entries_for_root = local_entries(local_root, str(experiment_root))
        local_entries_map = {entry_key(str(experiment_root), e.relative_path): e for e in local_entries_for_root}
        for e in local_entries_for_root:
            local_rows_before.append(
                {
                    "experiment_root": str(experiment_root),
                    "relative_path": e.relative_path,
                    "entry_type": e.entry_type,
                    "size_bytes": e.size_bytes,
                    "mtime_epoch": f"{e.mtime_epoch:.6f}",
                    "link_target": e.link_target,
                    "abs_path": e.abs_path,
                }
            )
        local_entries_map_before.update(local_entries_map)
        root_info["local_count_before"] = len(local_entries_for_root)
        total_local_files_before += len(local_entries_for_root)

        if remote_root is None:
            continue

        remote_entries_for_root = find_entries(remote_root, str(experiment_root))
        root_info["remote_count"] = len(remote_entries_for_root)
        total_remote_files += len(remote_entries_for_root)
        for e in remote_entries_for_root:
            remote_rows.append(
                {
                    "experiment_root": str(experiment_root),
                    "relative_path": e.relative_path,
                    "entry_type": e.entry_type,
                    "size_bytes": e.size_bytes,
                    "mtime_epoch": f"{e.mtime_epoch:.6f}",
                    "link_target": e.link_target,
                    "abs_path": e.abs_path,
                    "remote_root": str(remote_root),
                }
            )
            remote_entries_map[entry_key(str(experiment_root), e.relative_path)] = e

        missing_entries: List[Entry] = []
        conflict_entries: List[Tuple[Entry, Entry]] = []
        for e in remote_entries_for_root:
            key = entry_key(str(experiment_root), e.relative_path)
            local_e = local_entries_map.get(key)
            if local_e is None:
                if e.entry_type == "dir":
                    missing_local_dir_count += 1
                elif e.entry_type == "symlink":
                    missing_local_symlink_count += 1
                else:
                    missing_local_file_count += 1
                missing_entries.append(e)
                if e.entry_type == "symlink":
                    missing_symlinks.append(
                        {
                            "experiment_root": str(experiment_root),
                            "relative_path": e.relative_path,
                            "remote_root": str(remote_root),
                            "remote_type": e.entry_type,
                            "remote_size_bytes": e.size_bytes,
                            "remote_mtime_epoch": f"{e.mtime_epoch:.6f}",
                            "remote_target": e.link_target,
                        }
                    )
                elif e.entry_type != "dir":
                    missing_files.append(
                        {
                            "experiment_root": str(experiment_root),
                            "relative_path": e.relative_path,
                            "remote_root": str(remote_root),
                            "remote_type": e.entry_type,
                            "remote_size_bytes": e.size_bytes,
                            "remote_mtime_epoch": f"{e.mtime_epoch:.6f}",
                            "remote_target": e.link_target,
                        }
                    )
            else:
                if e.entry_type == "dir" and local_e.entry_type == "dir":
                    continue
                if e.entry_type != local_e.entry_type:
                    conflict_entries.append((e, local_e))
                elif e.entry_type == "symlink":
                    if e.link_target != local_e.link_target:
                        conflict_entries.append((e, local_e))
                else:
                    if e.size_bytes != local_e.size_bytes or int(e.mtime_epoch) != int(local_e.mtime_epoch):
                        conflict_entries.append((e, local_e))

        root_info["missing_files"] = sum(1 for e in missing_entries if e.entry_type == "file")
        root_info["missing_symlinks"] = sum(1 for e in missing_entries if e.entry_type == "symlink")
        root_info["missing_dirs"] = sum(1 for e in missing_entries if e.entry_type == "dir")
        root_info["conflicts"] = len(conflict_entries)
        conflict_count += len(conflict_entries)

        sync_relpaths = [e.relative_path for e in missing_entries if e.entry_type != "dir"]
        if sync_relpaths:
            rsync_pull(remote_root, sync_relpaths, LOCAL_REPO_ROOT)
            for rel in sync_relpaths:
                local_p = local_root / rel
                if local_p.is_symlink():
                    pulled_symlink_count += 1
                    pulled_rows.append(
                        {
                            "experiment_root": str(experiment_root),
                            "relative_path": rel,
                            "pulled_path": str(local_p),
                            "pulled_type": "symlink",
                            "remote_root": str(remote_root),
                            "status": "pulled",
                        }
                    )
                elif local_p.exists():
                    pulled_file_count += 1
                    if local_p.stat().st_size > 100_000:
                        large_file_pulled_count += 1
                    pulled_rows.append(
                        {
                            "experiment_root": str(experiment_root),
                            "relative_path": rel,
                            "pulled_path": str(local_p),
                            "pulled_type": "file",
                            "remote_root": str(remote_root),
                            "status": "pulled",
                        }
                    )
                else:
                    verification_missing.append(
                        {
                            "experiment_root": str(experiment_root),
                            "relative_path": rel,
                            "expected_path": str(local_p),
                            "problem": "missing_after_pull",
                        }
                    )

        ts = conflict_suffix()
        for remote_e, local_e in conflict_entries:
            conflict_local = (local_root / remote_e.relative_path).with_name(
                (local_root / remote_e.relative_path).name + f".remote_conflict_{ts}"
            )
            rsync_copy_single(remote_root, remote_e.relative_path, conflict_local)
            conflict_rows.append(
                {
                    "experiment_root": str(experiment_root),
                    "relative_path": remote_e.relative_path,
                    "local_path": str(local_e.abs_path),
                    "remote_path": str(remote_e.abs_path),
                    "local_type": local_e.entry_type,
                    "remote_type": remote_e.entry_type,
                    "local_size_bytes": local_e.size_bytes,
                    "remote_size_bytes": remote_e.size_bytes,
                    "local_mtime_epoch": f"{local_e.mtime_epoch:.6f}",
                    "remote_mtime_epoch": f"{remote_e.mtime_epoch:.6f}",
                    "conflict_copy_path": str(conflict_local),
                    "conflict_copy_exists": str(conflict_local.exists()).lower(),
                }
            )
            if not conflict_local.exists():
                verification_missing.append(
                    {
                        "experiment_root": str(experiment_root),
                        "relative_path": remote_e.relative_path,
                        "expected_path": str(conflict_local),
                        "problem": "conflict_copy_missing",
                    }
                )

        # symlink target resolution audit after sync
        for e in remote_entries_for_root:
            if e.entry_type != "symlink":
                continue
            local_p = local_root / e.relative_path
            if not local_p.is_symlink():
                unresolved_symlinks.append(
                    {
                        "experiment_root": str(experiment_root),
                        "relative_path": e.relative_path,
                        "local_path": str(local_p),
                        "remote_target": e.link_target,
                        "problem": "symlink_missing_or_replaced",
                    }
                )
                continue
            target = os.readlink(local_p)
            resolved = (local_p.parent / target).resolve() if not os.path.isabs(target) else Path(target)
            if not resolved.exists():
                unresolved_symlinks.append(
                    {
                        "experiment_root": str(experiment_root),
                        "relative_path": e.relative_path,
                        "local_path": str(local_p),
                        "remote_target": e.link_target,
                        "local_target": target,
                        "resolved_target": str(resolved),
                        "problem": "target_missing",
                    }
                )

    # recompute local manifest after sync
    for experiment_root in TARGET_ROOTS:
        local_root = LOCAL_REPO_ROOT / experiment_root
        for e in local_entries(local_root, str(experiment_root)):
            local_rows_after.append(
                {
                    "experiment_root": str(experiment_root),
                    "relative_path": e.relative_path,
                    "entry_type": e.entry_type,
                    "size_bytes": e.size_bytes,
                    "mtime_epoch": f"{e.mtime_epoch:.6f}",
                    "link_target": e.link_target,
                    "abs_path": e.abs_path,
                }
            )

    # verification
    for row in pulled_rows:
        local_p = Path(row["pulled_path"])
        if row["pulled_type"] == "symlink":
            if not local_p.is_symlink():
                verification_missing.append({**row, "problem": "pulled_symlink_missing"})
        else:
            if not local_p.exists():
                verification_missing.append({**row, "problem": "pulled_file_missing"})

    remote_manifest_path = OUTPUT_ROOT / "remote_file_manifest.tsv"
    local_manifest_path = OUTPUT_ROOT / "local_file_manifest.tsv"
    local_manifest_after_path = OUTPUT_ROOT / "local_file_manifest_after.tsv"
    missing_files_path = OUTPUT_ROOT / "missing_local_files.tsv"
    missing_symlinks_path = OUTPUT_ROOT / "missing_local_symlinks.tsv"
    conflict_path = OUTPUT_ROOT / "conflict_files.tsv"
    pulled_path = OUTPUT_ROOT / "pulled_files.tsv"
    unresolved_path = OUTPUT_ROOT / "unresolved_symlinks.tsv"

    fieldnames = ["experiment_root", "relative_path", "entry_type", "size_bytes", "mtime_epoch", "link_target", "abs_path"]
    write_tsv(remote_manifest_path, remote_rows, fieldnames + ["remote_root"])
    write_tsv(local_manifest_path, local_rows_before, fieldnames)
    write_tsv(local_manifest_after_path, local_rows_after, fieldnames)
    write_tsv(missing_files_path, missing_files, ["experiment_root", "relative_path", "remote_root", "remote_type", "remote_size_bytes", "remote_mtime_epoch", "remote_target"])
    write_tsv(missing_symlinks_path, missing_symlinks, ["experiment_root", "relative_path", "remote_root", "remote_type", "remote_size_bytes", "remote_mtime_epoch", "remote_target"])
    write_tsv(conflict_path, conflict_rows, ["experiment_root", "relative_path", "local_path", "remote_path", "local_type", "remote_type", "local_size_bytes", "remote_size_bytes", "local_mtime_epoch", "remote_mtime_epoch", "conflict_copy_path", "conflict_copy_exists"])
    write_tsv(pulled_path, pulled_rows, ["experiment_root", "relative_path", "pulled_path", "pulled_type", "remote_root", "status"])
    write_tsv(unresolved_path, unresolved_symlinks, ["experiment_root", "relative_path", "local_path", "remote_target", "problem", "local_target", "resolved_target"])

    sync_plan_path = OUTPUT_ROOT / "sync_plan.json"
    sync_plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))

    summary = {
        "status": "PASS" if not verification_missing else "FAIL",
        "remote_roots_checked": [str(p) for p in REMOTE_REPO_ROOTS],
        "experiment_roots_checked": [str(p) for p in TARGET_ROOTS],
        "total_remote_files": total_remote_files,
        "total_local_files_before": total_local_files_before,
        "missing_local_file_count": missing_local_file_count,
        "missing_local_symlink_count": missing_local_symlink_count,
        "pulled_file_count": pulled_file_count,
        "pulled_symlink_count": pulled_symlink_count,
        "large_file_pulled_count": large_file_pulled_count,
        "largest_file_pulled_path": max((r["pulled_path"] for r in pulled_rows if r["pulled_type"] == "file"), key=lambda p: Path(p).stat().st_size if Path(p).exists() else -1, default=""),
        "largest_file_pulled_size_bytes": max((Path(r["pulled_path"]).stat().st_size for r in pulled_rows if r["pulled_type"] == "file" and Path(r["pulled_path"]).exists()), default=0),
        "conflict_count": conflict_count,
        "unresolved_symlink_count": len(unresolved_symlinks),
        "verification_missing_after_sync_count": len(verification_missing),
        "local_takeover_path": str(LOCAL_REPO_ROOT / "codex/control/TAKEOVER_LATEST.md"),
        "output_root": str(OUTPUT_ROOT),
        "missing_local_dir_count": missing_local_dir_count,
        "pulled_paths": len(pulled_rows),
    }
    summary_path = OUTPUT_ROOT / "sync_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    takeover = OUTPUT_ROOT / "ASSET_SYNC_TAKEOVER.md"
    takeover.write_text(
        "# Asset Sync Takeover\n\n"
        f"- status: {summary['status']}\n"
        f"- output_root: {OUTPUT_ROOT}\n"
        f"- remote_roots_checked: {', '.join(summary['remote_roots_checked'])}\n"
        f"- experiment_roots_checked: {', '.join(summary['experiment_roots_checked'])}\n"
        f"- pulled_file_count: {pulled_file_count}\n"
        f"- pulled_symlink_count: {pulled_symlink_count}\n"
        f"- conflict_count: {conflict_count}\n"
        f"- unresolved_symlink_count: {len(unresolved_symlinks)}\n"
        f"- verification_missing_after_sync_count: {len(verification_missing)}\n"
        f"- note: external/internal metrics files were not part of the verified smoke trees and remain absent\n"
    )

    # print a compact summary for the shell
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not verification_missing else 1


if __name__ == "__main__":
    sys.exit(main())
