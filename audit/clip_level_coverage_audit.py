#!/usr/bin/env python3
"""
Clip-level coverage audit for WSOVVIS train-side semantics.

Purpose
-------
Given VideoCutLER stage outputs and GT sidecars, compute the clip-level statistics needed to
confirm whether the current train Observed/Unobserved semantics are behaving as expected.

Core definitions
----------------
For each clip v:
- Y'(v): observed class set for the clip, derived from proxy_records.jsonl rows.
- G(v): original GT class set for the clip, derived from trajectory_gt_identity_train_gt.jsonl.
- T_aud(v): auditable trajectories for the clip, derived from trajectory_gt_match_train_mainline.jsonl
  with audit_usable == True.
- y*(tau): matched_gt_raw_id_canonical for an auditable trajectory tau.

The script reports:
1. Distribution of |Y'(v)|, |G(v)|, and |Y'(v)| / |G(v)|.
2. clips_with_any_observed_trajectory / total_clips.
3. covered_observed_class_count / total_observed_class_count.
4. average auditable trajectory count per clip.

It also emits diagnostic counters and samples for problematic clips.

Notes
-----
- This script does NOT assume anything about weak_labels_train.json. It derives Y'(v) directly from
  VideoCutLER stage output rows (proxy_records.jsonl), which is the most stable source for the
  current audit question.
- It is robust to small schema differences by trying multiple field names where possible.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip-level coverage audit for WSOVVIS train semantics")
    p.add_argument("--batch-root", required=True, help="Batch root, e.g. /home/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/g8_param_exploration_20260420T214500Z")
    p.add_argument("--exp-name", required=True, help="Experiment name, e.g. E0_baseline")
    p.add_argument("--proxy-path", default=None, help="Optional explicit proxy_records.jsonl path")
    p.add_argument("--sidecar-match-path", default=None, help="Optional explicit trajectory_gt_match_train_mainline.jsonl path")
    p.add_argument("--sidecar-identity-path", default=None, help="Optional explicit trajectory_gt_identity_train_gt.jsonl path")
    p.add_argument("--output-json", default=None, help="Optional explicit output json path")
    p.add_argument("--output-md", default=None, help="Optional explicit output md path")
    p.add_argument("--topk-problem-clips", type=int, default=30, help="How many problematic clips to keep in the report")
    return p.parse_args()


def jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e


def first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return None


def normalize_int_list(xs: Any) -> List[int]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        return []
    out: List[int] = []
    for x in xs:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float) and float(x).is_integer():
            out.append(int(x))
        elif isinstance(x, str):
            try:
                out.append(int(x))
            except Exception:
                continue
    return out


def normalize_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, str):
        try:
            return int(x)
        except Exception:
            return None
    return None


def clip_key_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    video = first_present(row, ["video_id", "video", "video_name", "video_key"])
    clip = first_present(row, ["clip_id", "clip", "clip_name", "clip_key"])

    if video is None and clip is None:
        join_key = row.get("join_key")
        if isinstance(join_key, str):
            return ("join_key", join_key)
        traj_id = row.get("trajectory_id")
        if isinstance(traj_id, str):
            # Fallback: infer from trajectory_id format videocutler_r50_native:lvvis_train_base:<clip>:<idx>
            parts = traj_id.split(":")
            if len(parts) >= 4:
                return ("traj_inferred", parts[2])
        raise KeyError(f"Could not infer clip key from row keys={sorted(row.keys())}")

    return (str(video), str(clip))


class SummaryStats:
    def __init__(self, values: Sequence[float]) -> None:
        self.count = len(values)
        self.min = min(values) if values else None
        self.max = max(values) if values else None
        self.mean = (sum(values) / len(values)) if values else None
        self.p10 = self._quantile(values, 0.10)
        self.p50 = self._quantile(values, 0.50)
        self.p90 = self._quantile(values, 0.90)

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> Optional[float]:
        if not values:
            return None
        vs = sorted(values)
        idx = max(0, min(len(vs) - 1, int(len(vs) * q) - 1))
        return vs[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "max": self.max,
            "mean": self.mean,
        }


def sha256_of_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_proxy(proxy_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Set[int]], Dict[Tuple[str, str], int], Counter]:
    """
    Returns:
    - proxy_by_tid: trajectory_id -> selected proxy row info
    - yprime_by_clip: clip_key -> union of observed_raw_ids across rows in the clip
    - yprime_variant_count_by_clip: clip_key -> number of distinct observed_raw_ids signatures seen in the clip
    - diagnostics counter
    """
    proxy_by_tid: Dict[str, Dict[str, Any]] = {}
    yprime_by_clip: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    signatures_by_clip: Dict[Tuple[str, str], Set[Tuple[int, ...]]] = defaultdict(set)
    diag = Counter()

    for row in jsonl_iter(proxy_path):
        tid = row.get("trajectory_id")
        if not isinstance(tid, str):
            diag["proxy_missing_trajectory_id"] += 1
            continue
        ck = clip_key_from_row(row)
        obs = normalize_int_list(row.get("observed_raw_ids"))
        sig = tuple(sorted(set(obs)))
        signatures_by_clip[ck].add(sig)
        yprime_by_clip[ck].update(obs)

        proxy_by_tid[tid] = {
            "trajectory_id": tid,
            "clip_key": ck,
            "video_id": first_present(row, ["video_id", "video", "video_name", "video_key"]),
            "clip_id": first_present(row, ["clip_id", "clip", "clip_name", "clip_key"]),
            "observed_raw_ids": obs,
            "join_key": row.get("join_key"),
        }
        diag["proxy_rows"] += 1

    yprime_variant_count_by_clip = {k: len(v) for k, v in signatures_by_clip.items()}
    diag["proxy_unique_trajectories"] = len(proxy_by_tid)
    diag["proxy_unique_clips"] = len(yprime_by_clip)
    diag["proxy_clips_with_multiple_yprime_signatures"] = sum(1 for n in yprime_variant_count_by_clip.values() if n > 1)
    return proxy_by_tid, yprime_by_clip, yprime_variant_count_by_clip, diag


def load_sidecar_match(sidecar_match_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Counter]:
    sidecar_by_tid: Dict[str, Dict[str, Any]] = {}
    diag = Counter()
    for row in jsonl_iter(sidecar_match_path):
        tid = row.get("trajectory_id")
        if not isinstance(tid, str):
            diag["match_missing_trajectory_id"] += 1
            continue
        sidecar_by_tid[tid] = row
        diag["match_rows"] += 1
        if bool(row.get("audit_usable", False)):
            diag["match_audit_usable_true"] += 1
        else:
            diag["match_audit_usable_false"] += 1
    diag["match_unique_trajectories"] = len(sidecar_by_tid)
    return sidecar_by_tid, diag


def load_sidecar_identity(identity_path: Path) -> Tuple[Dict[Tuple[str, str], Set[int]], Counter]:
    gv_by_clip: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    diag = Counter()
    for row in jsonl_iter(identity_path):
        try:
            ck = clip_key_from_row(row)
        except Exception:
            diag["identity_missing_clip_key"] += 1
            continue
        gt_cls = normalize_int(row.get("matched_gt_raw_id_canonical"))
        if gt_cls is None:
            diag["identity_missing_canonical_gt_raw_id"] += 1
            continue
        gv_by_clip[ck].add(gt_cls)
        diag["identity_rows"] += 1
    diag["identity_unique_clips"] = len(gv_by_clip)
    return gv_by_clip, diag


def compute_clip_level_stats(
    proxy_by_tid: Dict[str, Dict[str, Any]],
    yprime_by_clip: Dict[Tuple[str, str], Set[int]],
    yprime_variant_count_by_clip: Dict[Tuple[str, str], int],
    sidecar_by_tid: Dict[str, Dict[str, Any]],
    gv_by_clip: Dict[Tuple[str, str], Set[int]],
    topk_problem_clips: int,
) -> Dict[str, Any]:
    auditable_count_by_clip: Counter = Counter()
    observed_traj_count_by_clip: Counter = Counter()
    unobserved_traj_count_by_clip: Counter = Counter()
    auditable_gt_classes_by_clip: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    diag = Counter()

    for tid, prow in proxy_by_tid.items():
        ck = prow["clip_key"]
        sc = sidecar_by_tid.get(tid)
        if sc is None:
            diag["proxy_rows_missing_sidecar"] += 1
            continue
        if not bool(sc.get("audit_usable", False)):
            diag["proxy_rows_sidecar_not_usable"] += 1
            continue
        gt_cls = normalize_int(sc.get("matched_gt_raw_id_canonical"))
        if gt_cls is None:
            diag["proxy_rows_sidecar_missing_canonical_gt_raw_id"] += 1
            continue

        auditable_count_by_clip[ck] += 1
        auditable_gt_classes_by_clip[ck].add(gt_cls)
        yprime = set(prow.get("observed_raw_ids", []))
        if gt_cls in yprime:
            observed_traj_count_by_clip[ck] += 1
        else:
            unobserved_traj_count_by_clip[ck] += 1

    all_clip_keys: Set[Tuple[str, str]] = set(yprime_by_clip.keys()) | set(gv_by_clip.keys()) | set(auditable_count_by_clip.keys())

    y_sizes = []
    g_sizes = []
    yg_ratios = []
    auditable_per_clip = []
    clips_with_any_observed = 0
    total_observed_class_count = 0
    covered_observed_class_count = 0
    problem_clips: List[Dict[str, Any]] = []

    for ck in sorted(all_clip_keys):
        yv = set(yprime_by_clip.get(ck, set()))
        gv = set(gv_by_clip.get(ck, set()))
        auditable_gt = set(auditable_gt_classes_by_clip.get(ck, set()))
        obs_cov = yv & auditable_gt

        y_sizes.append(len(yv))
        g_sizes.append(len(gv))
        if len(gv) > 0:
            yg_ratios.append(len(yv) / len(gv))

        aud = auditable_count_by_clip.get(ck, 0)
        auditable_per_clip.append(aud)

        obs_traj = observed_traj_count_by_clip.get(ck, 0)
        unobs_traj = unobserved_traj_count_by_clip.get(ck, 0)
        if obs_traj > 0:
            clips_with_any_observed += 1

        total_observed_class_count += len(yv)
        covered_observed_class_count += len(obs_cov)

        # Track clips that are most suspicious for the user's hypothesis.
        if len(yv) > 0 and obs_traj == 0:
            problem_clips.append({
                "clip_key": list(ck),
                "yprime_size": len(yv),
                "g_size": len(gv),
                "auditable_trajectory_count": aud,
                "observed_trajectory_count": obs_traj,
                "unobserved_trajectory_count": unobs_traj,
                "covered_observed_class_count": len(obs_cov),
                "yprime_variant_count": yprime_variant_count_by_clip.get(ck, 0),
                "yprime_head": sorted(list(yv))[:20],
                "auditable_gt_head": sorted(list(auditable_gt))[:20],
            })

    # Sort problem clips by strongest contradiction signal.
    problem_clips.sort(key=lambda x: (-x["yprime_size"], -x["auditable_trajectory_count"], x["clip_key"]))
    problem_clips = problem_clips[:topk_problem_clips]

    summary = {
        "total_clips": len(all_clip_keys),
        "clips_with_any_observed_trajectory": clips_with_any_observed,
        "clips_with_any_observed_ratio": (clips_with_any_observed / len(all_clip_keys)) if all_clip_keys else None,
        "total_observed_class_count": total_observed_class_count,
        "covered_observed_class_count": covered_observed_class_count,
        "covered_observed_class_ratio": (covered_observed_class_count / total_observed_class_count) if total_observed_class_count > 0 else None,
        "avg_auditable_trajectory_per_clip": (sum(auditable_per_clip) / len(auditable_per_clip)) if auditable_per_clip else None,
        "sample_level_observed_trajectory_total": sum(observed_traj_count_by_clip.values()),
        "sample_level_unobserved_trajectory_total": sum(unobserved_traj_count_by_clip.values()),
        "Y_prime_size_distribution": SummaryStats(y_sizes).to_dict(),
        "G_size_distribution": SummaryStats(g_sizes).to_dict(),
        "Y_prime_over_G_ratio_distribution": SummaryStats(yg_ratios).to_dict(),
        "auditable_trajectory_per_clip_distribution": SummaryStats(auditable_per_clip).to_dict(),
        "proxy_rows_total": len(proxy_by_tid),
        "sidecar_rows_total": len(sidecar_by_tid),
        "sidecar_audit_usable_total": sum(1 for row in sidecar_by_tid.values() if bool(row.get("audit_usable", False))),
        "problem_clips_topk": problem_clips,
        "diagnostics": dict(diag),
    }
    return summary


def format_md(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines: List[str] = []
    lines.append("# Clip-level Coverage Audit")
    lines.append("")
    lines.append("## Core results")
    lines.append(f"- total_clips: {s['total_clips']}")
    lines.append(f"- clips_with_any_observed_trajectory: {s['clips_with_any_observed_trajectory']}")
    lines.append(f"- clips_with_any_observed_ratio: {s['clips_with_any_observed_ratio']}")
    lines.append(f"- total_observed_class_count: {s['total_observed_class_count']}")
    lines.append(f"- covered_observed_class_count: {s['covered_observed_class_count']}")
    lines.append(f"- covered_observed_class_ratio: {s['covered_observed_class_ratio']}")
    lines.append(f"- avg_auditable_trajectory_per_clip: {s['avg_auditable_trajectory_per_clip']}")
    lines.append(f"- sample_level_observed_trajectory_total: {s['sample_level_observed_trajectory_total']}")
    lines.append(f"- sample_level_unobserved_trajectory_total: {s['sample_level_unobserved_trajectory_total']}")
    lines.append("")

    def add_dist(title: str, d: Dict[str, Any]) -> None:
        lines.append(f"## {title}")
        for k in ["count", "min", "p10", "p50", "p90", "max", "mean"]:
            lines.append(f"- {k}: {d.get(k)}")
        lines.append("")

    add_dist("|Y'(v)| distribution", s["Y_prime_size_distribution"])
    add_dist("|G(v)| distribution", s["G_size_distribution"])
    add_dist("|Y'(v)| / |G(v)| distribution", s["Y_prime_over_G_ratio_distribution"])
    add_dist("Auditable trajectory per clip distribution", s["auditable_trajectory_per_clip_distribution"])

    lines.append("## Top problematic clips")
    for item in s["problem_clips_topk"]:
        lines.append(f"- clip_key={item['clip_key']} yprime_size={item['yprime_size']} g_size={item['g_size']} auditable={item['auditable_trajectory_count']} observed_traj={item['observed_trajectory_count']} covered_observed_classes={item['covered_observed_class_count']} yprime_variant_count={item['yprime_variant_count']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    batch_root = Path(args.batch_root)
    exp_root = batch_root / args.exp_name

    proxy_path = Path(args.proxy_path) if args.proxy_path else exp_root / "train" / "prealign" / "proxy_records.jsonl"
    sidecar_match_path = Path(args.sidecar_match_path) if args.sidecar_match_path else exp_root / "audit" / "trajectory_gt_match_train_mainline.jsonl"
    sidecar_identity_path = Path(args.sidecar_identity_path) if args.sidecar_identity_path else exp_root / "audit" / "trajectory_gt_identity_train_gt.jsonl"

    out_dir = exp_root / "audit" / "coverage_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else out_dir / "clip_level_coverage_audit.json"
    output_md = Path(args.output_md) if args.output_md else out_dir / "clip_level_coverage_audit.md"

    proxy_by_tid, yprime_by_clip, yprime_variant_count_by_clip, proxy_diag = load_proxy(proxy_path)
    sidecar_by_tid, match_diag = load_sidecar_match(sidecar_match_path)
    gv_by_clip, identity_diag = load_sidecar_identity(sidecar_identity_path)

    summary = compute_clip_level_stats(
        proxy_by_tid=proxy_by_tid,
        yprime_by_clip=yprime_by_clip,
        yprime_variant_count_by_clip=yprime_variant_count_by_clip,
        sidecar_by_tid=sidecar_by_tid,
        gv_by_clip=gv_by_clip,
        topk_problem_clips=args.topk_problem_clips,
    )

    report = {
        "dataset_name": "lvvis_train_base",
        "exp_name": args.exp_name,
        "batch_root": str(batch_root),
        "paths": {
            "proxy_path": str(proxy_path),
            "proxy_sha256": sha256_of_file(proxy_path),
            "sidecar_match_path": str(sidecar_match_path),
            "sidecar_match_sha256": sha256_of_file(sidecar_match_path),
            "sidecar_identity_path": str(sidecar_identity_path),
            "sidecar_identity_sha256": sha256_of_file(sidecar_identity_path),
            "output_json": str(output_json),
            "output_md": str(output_md),
        },
        "proxy_diagnostics": dict(proxy_diag),
        "sidecar_match_diagnostics": dict(match_diag),
        "sidecar_identity_diagnostics": dict(identity_diag),
        "summary": summary,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with output_md.open("w", encoding="utf-8") as f:
        f.write(format_md(report))

    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_md}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
