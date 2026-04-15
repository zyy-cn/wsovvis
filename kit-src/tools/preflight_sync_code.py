from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from common import (
    git_source_identity,
    load_profile_from_repo,
    parse_args,
    remote_bash_lc,
    remote_excluded_paths,
    remote_host,
    remote_repo_dir,
    remote_runtime_asset_roots,
    require_local_remote_profile,
    require_schema,
    require_task_truth,
    run_cmd,
    shell_quote,
    sync_managed_local_paths,
    sync_managed_remote_paths,
    sync_mode,
    write_json,
    write_runtime_state,
)

EXCLUDE = {".git", ".private-cache", "__pycache__"}
EXCLUDE_EXACT = {
    "CURRENT_TASK.json",
    "PATH_POLICY.json",
    "VALIDATION_PLAN.json",
    "TAKEOVER_LATEST.md",
    "codex/state/private-cache/git_clean_result.json",
    "codex/state/private-cache/sync_manifest.json",
    "codex/state/private-cache/preflight_result.json",
    "codex/state/private-cache/validation_result.json",
    "codex/state/private-cache/validation_result_local_impl.json",
}
EXCLUDE_PREFIXES = {
    ".private-cache",
    "outputs",
    "codex/state/runtime",
    "codex/state/logs",
    "codex/outputs",
    "codex/tasks",
}
CACHE_FILES = [
    "codex/state/private-cache/package_manifest.json",
    "codex/state/private-cache/assertion_manifest.json",
    "codex/state/private-cache/codebase_gap_report.json",
]
REMOTE_STAGE_DIRNAME = ".kit_sync_stage"
REMOTE_PAYLOAD_NAME = "sync_payload.tar.gz"
REMOTE_REQUEST_NAME = "sync_request.json"
REMOTE_APPLY_NAME = "sync_apply_request.json"
MANAGED_MANIFEST_NAME = "managed_manifest.json"


def _normalize_rel(rel: str) -> str:
    text = str(rel).strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.strip("/")
    return text


def _join_rel(base: str, rel: str) -> str:
    base_clean = _normalize_rel(base)
    rel_clean = _normalize_rel(rel)
    if not base_clean:
        return rel_clean
    if not rel_clean:
        return base_clean
    return f"{base_clean}/{rel_clean}"


def _is_under(rel: str, root: str) -> bool:
    rel_clean = _normalize_rel(rel)
    root_clean = _normalize_rel(root)
    if not root_clean:
        return True
    return rel_clean == root_clean or rel_clean.startswith(root_clean + "/")


def skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in EXCLUDE_EXACT:
        return True
    if bool(set(path.relative_to(root).parts) & EXCLUDE):
        return True
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in EXCLUDE_PREFIXES)


def _sha256(path: Path) -> str | None:
    if path.is_symlink():
        return f"SYMLINK:{path.readlink().as_posix()}"
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sync_entries(repo_root: Path, profile: Dict[str, Any]) -> tuple[List[Dict[str, str]], List[str]]:
    managed_entries: Dict[str, Dict[str, str]] = {}
    excluded_local: List[str] = []
    mappings = list(zip(sync_managed_local_paths(profile), sync_managed_remote_paths(profile)))
    for local_root_rel, remote_root_rel in mappings:
        local_root_clean = _normalize_rel(local_root_rel)
        remote_root_clean = _normalize_rel(remote_root_rel)
        source_root = repo_root if not local_root_clean else (repo_root / local_root_clean)
        if not source_root.exists():
            continue
        if source_root.is_file() or source_root.is_symlink():
            if not skip(source_root, repo_root):
                local_rel = source_root.relative_to(repo_root).as_posix()
                managed_entries[local_rel] = {
                    "local_rel": local_rel,
                    "remote_rel": remote_root_clean or source_root.name,
                }
            else:
                excluded_local.append(source_root.relative_to(repo_root).as_posix())
            continue
        for path in source_root.rglob("*"):
            if ".git" in path.parts or not (path.is_file() or path.is_symlink()):
                continue
            if skip(path, repo_root):
                excluded_local.append(path.relative_to(repo_root).as_posix())
                continue
            local_rel = path.relative_to(repo_root).as_posix()
            rel_under_mapping = path.relative_to(source_root).as_posix()
            remote_rel = _join_rel(remote_root_clean, rel_under_mapping)
            existing = managed_entries.get(local_rel)
            if existing is not None and existing["remote_rel"] != remote_rel:
                raise RuntimeError(f"conflicting remote mapping for {local_rel}: {existing['remote_rel']} vs {remote_rel}")
            managed_entries[local_rel] = {
                "local_rel": local_rel,
                "remote_rel": remote_rel,
            }
    for rel in CACHE_FILES:
        src = repo_root / rel
        if src.exists() or src.is_symlink():
            managed_entries.setdefault(rel, {"local_rel": rel, "remote_rel": rel})
    entries = sorted(managed_entries.values(), key=lambda item: item["remote_rel"])
    return entries, sorted(set(excluded_local))


def _manifest_entries(repo_root: Path, entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for entry in entries:
        local_path = repo_root / entry["local_rel"]
        out.append(
            {
                "local_rel": entry["local_rel"],
                "remote_rel": entry["remote_rel"],
                "sha256": _sha256(local_path) or "",
                "entry_type": "symlink" if local_path.is_symlink() else "file",
            }
        )
    return out


def _load_previous_manifest(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest = payload.get("previous_manifest", {})
    entries = manifest.get("managed_entries", []) if isinstance(manifest, dict) else []
    return [item for item in entries if isinstance(item, dict) and str(item.get("remote_rel", "")).strip()]


def _safe_delete_candidate(rel: str, excluded: List[str], runtime_roots: List[str]) -> bool:
    rel_clean = _normalize_rel(rel)
    protected = {_normalize_rel(item) for item in excluded + runtime_roots}
    if not rel_clean:
        return False
    return not any(_is_under(rel_clean, item) for item in protected if item)


def classify_sync_plan(
    repo_root: Path,
    current_entries: List[Dict[str, str]],
    previous_entries: List[Dict[str, Any]],
    remote_hashes: Dict[str, str | None],
    *,
    sync_mode_value: str,
    remote_excluded: List[str],
    remote_runtime_roots: List[str],
    preserved_existing_before: List[str],
) -> Dict[str, Any]:
    current_by_remote = {entry["remote_rel"]: entry for entry in current_entries}
    uploads: List[Dict[str, str]] = []
    local_only: List[str] = []
    changed: List[str] = []
    unchanged: List[str] = []
    for entry in current_entries:
        local_hash = _sha256(repo_root / entry["local_rel"])
        remote_hash = remote_hashes.get(entry["remote_rel"])
        if remote_hash is None:
            local_only.append(entry["remote_rel"])
            uploads.append(entry)
        elif remote_hash != local_hash:
            changed.append(entry["remote_rel"])
            uploads.append(entry)
        else:
            unchanged.append(entry["remote_rel"])

    remote_only: List[str] = []
    delete_candidates: List[str] = []
    for prev in previous_entries:
        remote_rel = str(prev.get("remote_rel", "")).strip()
        local_rel = str(prev.get("local_rel", "")).strip()
        if not remote_rel or remote_rel in current_by_remote:
            continue
        remote_only.append(remote_rel)
        local_exists = bool(local_rel) and ((repo_root / local_rel).exists() or (repo_root / local_rel).is_symlink())
        if (
            sync_mode_value == "managed_delete"
            and not local_exists
            and _safe_delete_candidate(remote_rel, remote_excluded, remote_runtime_roots)
        ):
            delete_candidates.append(remote_rel)

    return {
        "uploads": uploads,
        "local_only": sorted(local_only),
        "changed": sorted(changed),
        "unchanged": sorted(unchanged),
        "remote_only": sorted(remote_only),
        "delete_candidates": sorted(delete_candidates),
        "excluded": sorted(set(preserved_existing_before)),
    }


def apply_sync_plan_local(
    *,
    repo_root: Path,
    remote_root: Path,
    uploads: List[Dict[str, str]],
    delete_candidates: List[str],
    current_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    stage_dir = remote_root / REMOTE_STAGE_DIRNAME
    stage_dir.mkdir(parents=True, exist_ok=True)
    for rel in sorted(delete_candidates):
        target = remote_root / rel
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        parent = target.parent
        while parent != remote_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    for entry in uploads:
        src = repo_root / entry["local_rel"]
        dst = remote_root / entry["remote_rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.is_file():
            dst.unlink(missing_ok=True)
        elif dst.is_dir():
            shutil.rmtree(dst)
        if src.is_symlink():
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)
    manifest_path = stage_dir / MANAGED_MANIFEST_NAME
    manifest_path.write_text(json.dumps(current_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"managed_manifest_path": str(manifest_path)}


def make_payload(repo_root: Path, entries: List[Dict[str, str]]) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix="kit_sync_", suffix=".tar.gz")
    os.close(fd)
    payload = Path(tmp_path)
    with tarfile.open(payload, "w:gz", dereference=False) as tf:
        for entry in entries:
            src = repo_root / entry["local_rel"]
            if src.exists() or src.is_symlink():
                tf.add(src, arcname=entry["remote_rel"], recursive=False)
    return payload


def run_capture(cmd: list[str], stdin_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, input=stdin_text, capture_output=True)


def remote_git_status(sync_remote_host: str, sync_remote_root: str) -> subprocess.CompletedProcess:
    command = f"cd {shell_quote(sync_remote_root)} && git status --porcelain=v1 --untracked-files=all"
    return run_capture(remote_bash_lc(sync_remote_host, command))


def _remote_stage_dir(remote_root: str) -> str:
    return f"{remote_root.rstrip('/')}/{REMOTE_STAGE_DIRNAME}"


def probe_remote_state(
    *,
    remote_host_name: str,
    remote_root: str,
    request_payload: Dict[str, Any],
) -> Dict[str, Any]:
    stage_dir = _remote_stage_dir(remote_root)
    local_request = Path(tempfile.mkstemp(prefix="kit_sync_probe_", suffix=".json")[1])
    remote_request = f"{stage_dir}/{REMOTE_REQUEST_NAME}"
    try:
        local_request.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cp = run_capture(remote_bash_lc(remote_host_name, f"mkdir -p {shell_quote(stage_dir)} {shell_quote(remote_root)}"))
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        cp = run_capture(["scp", str(local_request), f"{remote_host_name}:{remote_request}"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        remote_script = r'''
set -euo pipefail
REMOTE_ROOT="$1"
STAGE_DIR="$2"
REQUEST_PATH="$3"
python3 - "$REMOTE_ROOT" "$STAGE_DIR" "$REQUEST_PATH" <<'PY_REMOTE'
import json
import hashlib
import sys
from pathlib import Path

remote_root = Path(sys.argv[1])
stage_dir = Path(sys.argv[2])
request_path = Path(sys.argv[3])
request = json.loads(request_path.read_text(encoding='utf-8'))
manifest_path = stage_dir / 'managed_manifest.json'
previous_manifest = {}
if manifest_path.exists():
    previous_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

def sha(path: Path):
    if path.is_symlink():
        return 'SYMLINK:' + path.readlink().as_posix()
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

remote_hashes = {}
for rel in request.get('remote_rel_paths', []):
    remote_hashes[rel] = sha(remote_root / rel)

preserved_existing_before = []
for rel in request.get('preserve_probe_paths', []):
    path = remote_root / rel
    if path.exists() or path.is_symlink():
        preserved_existing_before.append(rel)

print(json.dumps({
    'remote_hashes': remote_hashes,
    'previous_manifest': previous_manifest,
    'preserved_existing_before': sorted(preserved_existing_before),
}))
request_path.unlink(missing_ok=True)
PY_REMOTE
'''
        cp = run_capture(
            ["ssh", remote_host_name, "bash", "-s", "--", remote_root, stage_dir, remote_request],
            stdin_text=remote_script,
        )
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        return json.loads(cp.stdout.strip() or "{}")
    finally:
        local_request.unlink(missing_ok=True)


def apply_remote_sync(
    *,
    remote_host_name: str,
    remote_root: str,
    apply_request: Dict[str, Any],
    payload: Path,
) -> Dict[str, Any]:
    stage_dir = _remote_stage_dir(remote_root)
    local_request = Path(tempfile.mkstemp(prefix="kit_sync_apply_", suffix=".json")[1])
    remote_request = f"{stage_dir}/{REMOTE_APPLY_NAME}"
    remote_payload = f"{stage_dir}/{REMOTE_PAYLOAD_NAME}"
    try:
        local_request.write_text(json.dumps(apply_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cp = run_capture(["scp", str(local_request), f"{remote_host_name}:{remote_request}"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        cp = run_capture(["scp", str(payload), f"{remote_host_name}:{remote_payload}"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        remote_script = r'''
set -euo pipefail
REMOTE_ROOT="$1"
STAGE_DIR="$2"
REQUEST_PATH="$3"
PAYLOAD_PATH="$4"
python3 - "$REMOTE_ROOT" "$STAGE_DIR" "$REQUEST_PATH" "$PAYLOAD_PATH" <<'PY_REMOTE'
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

remote_root = Path(sys.argv[1])
stage_dir = Path(sys.argv[2])
request_path = Path(sys.argv[3])
payload_path = Path(sys.argv[4])
request = json.loads(request_path.read_text(encoding='utf-8'))
extract_root = stage_dir / 'tree'
if extract_root.exists():
    shutil.rmtree(extract_root)
extract_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(payload_path, 'r:gz') as tf:
    tf.extractall(extract_root)

for rel in request.get('delete_candidates', []):
    target = remote_root / rel
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    parent = target.parent
    while parent != remote_root and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent

for rel in request.get('upload_remote_paths', []):
    src = extract_root / rel
    dst = remote_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink(missing_ok=True)
    elif dst.is_dir():
        shutil.rmtree(dst)
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)

manifest_path = stage_dir / 'managed_manifest.json'
manifest_path.write_text(json.dumps(request.get('current_manifest', {}), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
preserved_existing_after = []
for rel in request.get('preserve_probe_paths', []):
    path = remote_root / rel
    if path.exists() or path.is_symlink():
        preserved_existing_after.append(rel)

request_path.unlink(missing_ok=True)
payload_path.unlink(missing_ok=True)
if extract_root.exists():
    shutil.rmtree(extract_root)
print(json.dumps({
    'preserved_existing_after': sorted(preserved_existing_after),
    'managed_manifest_path': str(manifest_path),
}))
PY_REMOTE
'''
        cp = run_capture(
            ["ssh", remote_host_name, "bash", "-s", "--", remote_root, stage_dir, remote_request, remote_payload],
            stdin_text=remote_script,
        )
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout)
        return json.loads(cp.stdout.strip() or "{}")
    finally:
        local_request.unlink(missing_ok=True)


def main() -> int:
    parser = parse_args("Sync local code to remote directory over ssh/scp")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--remote-repo-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--profile")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    task_truth = require_task_truth(repo_root)
    profile, profile_path, profile_source = load_profile_from_repo(repo_root, args.profile)
    require_local_remote_profile(profile)
    sync_remote_host = args.remote_host.strip() or remote_host(profile)
    sync_remote_root = args.remote_repo_root.strip() or remote_repo_dir(profile)
    local_source_identity = git_source_identity(repo_root)
    sync_mode_value = sync_mode(profile)
    remote_excluded = sorted(set(remote_excluded_paths(profile) + [REMOTE_STAGE_DIRNAME, ".git"]))
    remote_runtime_roots = sorted(set(remote_runtime_asset_roots(profile)))
    managed_mappings = [
        {"local_root": local_root, "remote_root": remote_root_rel}
        for local_root, remote_root_rel in zip(sync_managed_local_paths(profile), sync_managed_remote_paths(profile))
    ]

    status_cp = remote_git_status(sync_remote_host, sync_remote_root)
    remote_status_sample = (status_cp.stdout if status_cp.returncode == 0 else status_cp.stderr).splitlines()[:50]
    preflight_remote_git_clean = status_cp.returncode == 0 and status_cp.stdout.strip() == ""

    entries, excluded_local = build_sync_entries(repo_root, profile)
    current_manifest_entries = _manifest_entries(repo_root, entries)
    preserve_probe_paths = sorted(set(remote_excluded + remote_runtime_roots))
    probe_payload = {
        "remote_rel_paths": [entry["remote_rel"] for entry in current_manifest_entries],
        "preserve_probe_paths": preserve_probe_paths,
    }
    payload = None
    failure_reason = ""
    failure_code = ""
    apply_result: Dict[str, Any] = {}
    try:
        remote_probe = probe_remote_state(
            remote_host_name=sync_remote_host,
            remote_root=sync_remote_root,
            request_payload=probe_payload,
        )
        previous_entries = _load_previous_manifest(remote_probe)
        plan = classify_sync_plan(
            repo_root,
            entries,
            previous_entries,
            remote_probe.get("remote_hashes", {}),
            sync_mode_value=sync_mode_value,
            remote_excluded=remote_excluded,
            remote_runtime_roots=remote_runtime_roots,
            preserved_existing_before=remote_probe.get("preserved_existing_before", []),
        )
        payload = make_payload(repo_root, plan["uploads"])
        current_manifest = {
            "sync_mode": sync_mode_value,
            "managed_mappings": managed_mappings,
            "managed_entries": current_manifest_entries,
            "profile_source": profile_source,
            "profile_path": str(profile_path) if profile_path else "",
        }
        apply_result = apply_remote_sync(
            remote_host_name=sync_remote_host,
            remote_root=sync_remote_root,
            apply_request={
                "delete_candidates": plan["delete_candidates"],
                "upload_remote_paths": [entry["remote_rel"] for entry in plan["uploads"]],
                "current_manifest": current_manifest,
                "preserve_probe_paths": preserve_probe_paths,
            },
            payload=payload,
        )
        preserved_before = sorted(set(remote_probe.get("preserved_existing_before", [])))
        preserved_after = sorted(set(apply_result.get("preserved_existing_after", [])))
        unexpected_missing_preserved = sorted(set(preserved_before) - set(preserved_after))
        out = {
            "sync_id": "sync-current",
            "task_id": args.task_id,
            "task_truth_stamp": task_truth["task_truth_stamp"],
            "local_ref": str(repo_root),
            "remote_ref": f"{sync_remote_host}:{sync_remote_root}",
            "sync_mode": sync_mode_value,
            "delete_enabled": sync_mode_value == "managed_delete",
            "managed_mappings": managed_mappings,
            "managed_entries": current_manifest_entries,
            "synced_paths": [entry["local_rel"] for entry in current_manifest_entries],
            "synced_remote_paths": [entry["remote_rel"] for entry in current_manifest_entries],
            "synced_file_count": len(current_manifest_entries),
            "uploaded_paths": [entry["local_rel"] for entry in plan["uploads"]],
            "uploaded_remote_paths": [entry["remote_rel"] for entry in plan["uploads"]],
            "excluded_paths": excluded_local,
            "remote_excluded_paths": remote_excluded,
            "remote_runtime_asset_roots": remote_runtime_roots,
            "preview": {
                "local_only": plan["local_only"],
                "remote_only": plan["remote_only"],
                "changed": plan["changed"],
                "excluded": plan["excluded"],
                "unchanged": plan["unchanged"],
            },
            "delete_candidates": plan["delete_candidates"],
            "delete_executed_paths": plan["delete_candidates"] if sync_mode_value == "managed_delete" else [],
            "preserved_remote_paths_existing_before": preserved_before,
            "preserved_remote_paths_existing_after": preserved_after,
            "unexpected_missing_preserved_paths": unexpected_missing_preserved,
            "preflight_remote_git_clean": preflight_remote_git_clean,
            "failure_reason": "",
            "remote_status_sample": "\n".join(remote_status_sample),
            "sync_completed": True,
            "local_source_identity": local_source_identity,
            "remote_synced_source_identity": {
                "remote_ref": f"{sync_remote_host}:{sync_remote_root}",
                "source_repo_root": str(repo_root),
                "source_head": local_source_identity.get("head", ""),
                "source_tree": local_source_identity.get("tree", ""),
                "managed_manifest_path": str(Path(sync_remote_root) / REMOTE_STAGE_DIRNAME / MANAGED_MANIFEST_NAME),
            },
            "failure_code": "",
            "status": "PASS" if not unexpected_missing_preserved else "FAIL",
        }
        if unexpected_missing_preserved:
            out["failure_reason"] = "preserved remote paths disappeared after sync"
            out["failure_code"] = "FAIL_PRESERVED_REMOTE_PATHS"
    except Exception as exc:
        failure_reason = str(exc)
        failure_code = "FAIL_SYNC"
        out = {
            "sync_id": "sync-current",
            "task_id": args.task_id,
            "task_truth_stamp": task_truth["task_truth_stamp"],
            "local_ref": str(repo_root),
            "remote_ref": f"{sync_remote_host}:{sync_remote_root}",
            "sync_mode": sync_mode_value,
            "delete_enabled": sync_mode_value == "managed_delete",
            "managed_mappings": managed_mappings,
            "managed_entries": [],
            "synced_paths": [],
            "synced_remote_paths": [],
            "synced_file_count": 0,
            "uploaded_paths": [],
            "uploaded_remote_paths": [],
            "excluded_paths": excluded_local if "excluded_local" in locals() else [],
            "remote_excluded_paths": remote_excluded,
            "remote_runtime_asset_roots": remote_runtime_roots,
            "preview": {"local_only": [], "remote_only": [], "changed": [], "excluded": [], "unchanged": []},
            "delete_candidates": [],
            "delete_executed_paths": [],
            "preserved_remote_paths_existing_before": [],
            "preserved_remote_paths_existing_after": [],
            "unexpected_missing_preserved_paths": [],
            "preflight_remote_git_clean": preflight_remote_git_clean,
            "failure_reason": failure_reason,
            "remote_status_sample": "\n".join(remote_status_sample),
            "sync_completed": False,
            "local_source_identity": local_source_identity,
            "remote_synced_source_identity": {
                "remote_ref": f"{sync_remote_host}:{sync_remote_root}",
                "source_repo_root": str(repo_root),
                "source_head": local_source_identity.get("head", ""),
                "source_tree": local_source_identity.get("tree", ""),
            },
            "failure_code": failure_code,
            "status": "FAIL",
        }
    finally:
        if payload is not None:
            payload.unlink(missing_ok=True)

    schema_root = repo_root / "schemas" if (repo_root / "schemas").exists() else Path(__file__).resolve().parents[1] / "schemas"
    require_schema(out, schema_root / "sync_manifest.schema.json", "sync_manifest")
    write_json(Path(args.output), out)
    write_runtime_state(
        repo_root,
        "sync_state",
        {
            "task_id": args.task_id,
            "status": out["status"],
            "sync_completed": bool(out["sync_completed"]),
            "remote_ref": f"{sync_remote_host}:{sync_remote_root}",
            "task_truth_stamp": task_truth["task_truth_stamp"],
            "sync_mode": sync_mode_value,
        },
    )
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
