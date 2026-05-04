#!/usr/bin/env python3
"""Minimal PATH_POLICY authorization helper for A8 true-margin/convergence audit.

This script is intentionally narrow. It only adds the exact paths required for the
A8 true-margin exporter and convergence-probe patch, and writes a report.

Usage from repo root:
  python tools/a8_authorize_true_margin_convergence_paths.py --apply

Safety:
- Creates a timestamped backup of codex/control/PATH_POLICY.json.
- Does not broaden to tools/** or videocutler/**.
- Fails if it cannot locate an allow-path list in the current policy schema.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
from pathlib import Path
from typing import Any, List, Tuple

REQUIRED_PATHS = [
    "videocutler/run_stageb_train_residual_gated_hungarian_matched.py",
    "tools/a8_score_margin_convergence_audit.py",
    "tools/a8_true_margin_export_audit.py",
    "codex/outputs/G8_inference_and_eval/a8_true_margin_convergence_patch_20260504/**",
]

REPORT_DIR = Path("codex/outputs/G8_inference_and_eval/a8_true_margin_convergence_patch_20260504")
POLICY_PATH = Path("codex/control/PATH_POLICY.json")


def _is_allow_path_key(key: str) -> bool:
    k = key.lower()
    return ("allow" in k or "permitted" in k) and "path" in k


def _find_candidate_lists(obj: Any, path: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], list]]:
    out: List[Tuple[Tuple[Any, ...], list]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_allow_path_key(str(k)) and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out.append((path + (k,), v))
            out.extend(_find_candidate_lists(v, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_find_candidate_lists(v, path + (i,)))
    return out


def _set_by_path(obj: Any, path: Tuple[Any, ...], value: Any) -> None:
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(POLICY_PATH))
    ap.add_argument("--apply", action="store_true", help="write the policy update; otherwise dry-run only")
    ap.add_argument("--target-list-index", type=int, default=None, help="candidate allow list index to patch; default: patch all candidate allow-path lists")
    args = ap.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise SystemExit(f"PATH_POLICY not found: {policy_path}")

    raw = policy_path.read_text(encoding="utf-8")
    policy = json.loads(raw)
    new_policy = copy.deepcopy(policy)

    candidates = _find_candidate_lists(new_policy)
    if not candidates:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "FAIL_NO_ALLOW_PATH_LIST_FOUND",
            "policy": str(policy_path),
            "required_paths": REQUIRED_PATHS,
            "top_level_keys": list(policy.keys()) if isinstance(policy, dict) else None,
            "message": "Could not locate a list-valued key like allow_paths/permitted_paths. Refuse to invent schema.",
        }
        (REPORT_DIR / "path_policy_authorization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    selected = []
    if args.target_list_index is not None:
        if args.target_list_index < 0 or args.target_list_index >= len(candidates):
            raise SystemExit(f"target-list-index out of range; candidates={len(candidates)}")
        selected = [candidates[args.target_list_index]]
    else:
        selected = candidates

    changes = []
    for path, current_list in selected:
        before = list(current_list)
        after = list(current_list)
        for p in REQUIRED_PATHS:
            if p not in after:
                after.append(p)
        _set_by_path(new_policy, path, after)
        changes.append({
            "list_path": "/".join(map(str, path)),
            "before_count": len(before),
            "after_count": len(after),
            "added": [p for p in REQUIRED_PATHS if p not in before],
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "DRY_RUN" if not args.apply else "UPDATED",
        "policy": str(policy_path),
        "required_paths": REQUIRED_PATHS,
        "candidate_allow_lists": ["/".join(map(str, p)) for p, _ in candidates],
        "patched_allow_lists": [c["list_path"] for c in changes],
        "changes": changes,
        "next_step": "Run py_compile on tools/a8_score_margin_convergence_audit.py and then let Codex implement the true-margin exporter within the now-authorized paths.",
    }

    if args.apply:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = policy_path.with_suffix(policy_path.suffix + f".bak_a8_true_margin_{ts}")
        backup.write_text(raw, encoding="utf-8")
        policy_path.write_text(json.dumps(new_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["backup"] = str(backup)

    (REPORT_DIR / "path_policy_authorization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = []
    md.append("# A8 true-margin/convergence PATH_POLICY authorization")
    md.append("")
    md.append(f"- status: {report['status']}")
    md.append(f"- policy: {policy_path}")
    if args.apply:
        md.append(f"- backup: {report.get('backup')}")
    md.append("")
    md.append("## Required paths")
    for p in REQUIRED_PATHS:
        md.append(f"- `{p}`")
    md.append("")
    md.append("## Patched allow lists")
    for c in changes:
        md.append(f"- `{c['list_path']}`: +{len(c['added'])}")
    md.append("")
    md.append("## Next step")
    md.append("After this authorization is applied, run the A8 true-margin/convergence implementation patch only within the listed paths. Do not broaden scope.")
    (REPORT_DIR / "A8_TRUE_MARGIN_CONVERGENCE_AUTH_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
