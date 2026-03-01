#!/usr/bin/env python3
"""Build StageC3 decoder comparison sidecars from protocol log markers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

MARKER_PREFIXES = {
    "C3R3A_REMOTE_META",
    "C3R3A_COHORT",
    "C3R3A_RUN_COMMAND",
    "C3R3A_RUN_METRIC",
    "C3R3A_DETERMINISM",
    "C3R3A_PROTOCOL_DONE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True, help="Path to captured remote protocol log")
    p.add_argument("--output-dir", required=True, help="Output directory for sidecars")
    p.add_argument("--comparison-id", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--local-head", required=True)
    p.add_argument("--intended-commit", required=True)
    p.add_argument("--remote-host", default="gpu4090d")
    p.add_argument("--schema-version", default="stagec3.decoder_comparison.v1")
    p.add_argument(
        "--required-decoders",
        default="independent,coverage_greedy_v1,otlite_v1",
        help="Comma-separated decoder set",
    )
    p.add_argument(
        "--required-tiers",
        default="small,medium",
        help="Comma-separated required tiers",
    )
    p.add_argument(
        "--determinism-required-tiers",
        default="small",
        help="Comma-separated tiers that must include determinism evidence for all required decoders",
    )
    return p.parse_args()


def load_markers(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in MARKER_PREFIXES}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "::" not in line:
            continue
        marker, payload = line.split("::", 1)
        marker = marker.strip()
        if marker not in MARKER_PREFIXES:
            continue
        payload = payload.strip()
        if not payload:
            continue
        out[marker].append(json.loads(payload))
    return out


def run_key(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(entry["tier"]), str(entry["decoder"]), str(entry["run_id"])


def compute_status(
    metrics: Dict[Tuple[str, str, str], Dict[str, Any]],
    determinism: Dict[Tuple[str, str], Dict[str, Any]],
    required_decoders: List[str],
    required_tiers: List[str],
    determinism_required_tiers: List[str],
) -> Tuple[str, List[str]]:
    issues: List[str] = []
    for tier in required_tiers:
        for decoder in required_decoders:
            key = (tier, decoder, "run_a")
            if key not in metrics:
                issues.append(f"missing_run_metric:{tier}:{decoder}:run_a")
    for tier in determinism_required_tiers:
        for decoder in required_decoders:
            dkey = (tier, decoder)
            if dkey not in determinism:
                issues.append(f"missing_determinism:{tier}:{decoder}")
            elif not bool(determinism[dkey].get("cmp_pass", False)):
                issues.append(f"determinism_fail:{tier}:{decoder}")
    if issues:
        return "PARTIAL", issues
    return "PASS", issues


def make_recommendations(
    status: str,
    tiers_seen: List[str],
    cohorts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    insufficient = []
    if "large" not in tiers_seen:
        insufficient.append("large_tier_not_executed")
    if not any(tier.startswith("cross-run") for tier in tiers_seen):
        insufficient.append("cross_run_cohort_not_executed")

    run_roots = sorted(
        {
            str(entry.get("source_run_root", "")).strip()
            for entry in cohorts
            if str(entry.get("source_run_root", "")).strip()
        }
    )
    if run_roots == ["runs/wsovvis_seqformer/18"]:
        insufficient.append("single_run_source_run18_only")
    elif len(run_roots) <= 1:
        insufficient.append("single_run_source_only")

    engineering = {
        "category": "SUPPORTED_NO_DEFAULT_CHANGE",
        "recommendation": "keep_default_independent",
        "rationale": "Comparison evidence supports keeping production default unchanged.",
    }
    if status != "PASS":
        engineering["category"] = "BLOCKED_EVIDENCE"
        engineering["recommendation"] = "keep_default_independent"
        engineering["rationale"] = "Protocol evidence is incomplete or determinism checks failed."
    elif insufficient:
        engineering["category"] = "INSUFFICIENT_EVIDENCE"
        engineering["recommendation"] = "keep_default_independent"
        engineering["rationale"] = (
            "Protocol checks passed for configured cohorts, but evidence gaps remain for default-change decisions."
        )

    research = {
        "category": "EXPLORATION",
        "recommendation": "otlite_v1_for_global_consistency_experiments",
        "rationale": (
            "Use otlite_v1 as research decoder when OT diagnostics and runtime cost are acceptable; "
            "keep independent for production defaults."
        ),
    }
    return {
        "engineering": engineering,
        "research": research,
        "insufficient_evidence": insufficient,
    }


def write_report_md(
    output_path: Path,
    manifest: Dict[str, Any],
    metrics_payload: Dict[str, Any],
) -> None:
    run_metrics = metrics_payload["run_metrics"]
    determinism = metrics_payload["determinism"]
    recs = metrics_payload["recommendations"]

    lines: List[str] = []
    lines.append("# StageC3 Decoder Comparison Report")
    lines.append("")
    lines.append(f"- comparison_id: `{manifest['comparison_id']}`")
    lines.append(f"- schema_version: `{manifest['schema_version']}`")
    lines.append(f"- branch: `{manifest['branch']}`")
    lines.append(f"- local_head: `{manifest['local_head']}`")
    lines.append(f"- remote_head: `{manifest['remote_head']}`")
    lines.append(f"- commit_match: `{manifest['commit_match']}`")
    lines.append("")
    lines.append("## Table A - Run Metadata")
    lines.append("")
    lines.append("| tier | decoder | run_id | videos | tracks | coverage_ratio | num_tracks_bg | fill_bg_count | tie_break_count | wall_seconds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in run_metrics:
        lines.append(
            "| {tier} | {decoder} | {run_id} | {num_videos_total} | {num_tracks_scored} | {coverage_ratio:.6f} | {num_tracks_bg} | {fill_bg_count} | {tie_break_count} | {wall_seconds:.3f} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Table B - Determinism")
    lines.append("")
    lines.append("| tier | decoder | cmp_pass | track_scores sha256 | per_video_summary sha256 | run_summary sha256 |")
    lines.append("|---|---|---|---|---|---|")
    for row in determinism:
        hashes = row.get("hashes_run_a", {})
        lines.append(
            f"| {row['tier']} | {row['decoder']} | {row['cmp_pass']} | "
            f"`{hashes.get('track_scores.jsonl', '')}` | `{hashes.get('per_video_summary.json', '')}` | `{hashes.get('run_summary.json', '')}` |"
        )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"- engineering: **{recs['engineering']['category']}** -> `{recs['engineering']['recommendation']}`; {recs['engineering']['rationale']}"
    )
    lines.append(
        f"- research: **{recs['research']['category']}** -> `{recs['research']['recommendation']}`; {recs['research']['rationale']}"
    )
    lines.append(
        "- insufficient_evidence: "
        + ", ".join(f"`{item}`" for item in recs.get("insufficient_evidence", []))
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markers = load_markers(log_path)

    remote_meta = markers["C3R3A_REMOTE_META"][0] if markers["C3R3A_REMOTE_META"] else {}
    cohorts = markers["C3R3A_COHORT"]
    run_commands = markers["C3R3A_RUN_COMMAND"]

    metric_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        run_key(entry): entry for entry in markers["C3R3A_RUN_METRIC"]
    }
    determinism_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(entry["tier"]), str(entry["decoder"])): entry
        for entry in markers["C3R3A_DETERMINISM"]
    }

    required_decoders = [s.strip() for s in args.required_decoders.split(",") if s.strip()]
    required_tiers = [s.strip() for s in args.required_tiers.split(",") if s.strip()]
    determinism_required_tiers = [
        s.strip() for s in args.determinism_required_tiers.split(",") if s.strip()
    ]

    status, issues = compute_status(
        metric_map,
        determinism_map,
        required_decoders,
        required_tiers,
        determinism_required_tiers,
    )

    run_metrics_sorted = sorted(
        metric_map.values(),
        key=lambda x: (str(x["tier"]), str(x["decoder"]), str(x["run_id"])),
    )
    determinism_sorted = sorted(
        determinism_map.values(),
        key=lambda x: (str(x["tier"]), str(x["decoder"])),
    )

    tiers_seen = sorted({str(c.get("tier")) for c in cohorts if c.get("tier")})
    recommendations = make_recommendations(status, tiers_seen, cohorts)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    remote_head = str(remote_meta.get("remote_head", ""))
    remote_branch = str(remote_meta.get("remote_branch", ""))
    commit_match = remote_head == args.intended_commit and remote_branch == args.branch

    manifest = {
        "comparison_id": args.comparison_id,
        "schema_version": args.schema_version,
        "created_at_utc": now,
        "branch": args.branch,
        "local_head": args.local_head,
        "intended_commit": args.intended_commit,
        "remote_host": args.remote_host,
        "remote_repo_dir": remote_meta.get("remote_repo_dir"),
        "remote_branch": remote_branch,
        "remote_head": remote_head,
        "branch_match": remote_branch == args.branch,
        "commit_match": commit_match,
        "canonical_remote_verified": bool(markers["C3R3A_REMOTE_META"]),
        "scorer_backend": remote_meta.get("scorer_backend", "labelset_proto_v1"),
        "cohorts": cohorts,
        "required_decoders": required_decoders,
        "required_tiers": required_tiers,
        "determinism_required_tiers": determinism_required_tiers,
        "command_log": run_commands,
        "issues": issues,
    }

    metrics_payload = {
        "comparison_id": args.comparison_id,
        "schema_version": args.schema_version,
        "status": status,
        "issues": issues,
        "run_metrics": run_metrics_sorted,
        "determinism": determinism_sorted,
        "recommendations": recommendations,
    }

    manifest_path = output_dir / "comparison_manifest.json"
    metrics_path = output_dir / "comparison_metrics.json"
    report_path = output_dir / "comparison_report.md"

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report_md(report_path, manifest, metrics_payload)


if __name__ == "__main__":
    main()
