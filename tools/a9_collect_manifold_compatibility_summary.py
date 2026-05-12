#!/usr/bin/env python3
"""Collect A9 manifold compatibility audit summaries into compact tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUMMARY_FILENAMES = {
    "textbank_graph_global_summary.csv": "E1_textbank_graph",
    "graph_global_isomorphism_summary.csv": "E1_text_vision_graph",
    "group_graph_isomorphism_summary.csv": "E1_text_vision_graph_group",
    "hub_structure_summary.csv": "E1_hub_structure",
    "class_proto_alignment_summary.csv": "E2_class_proto_holdout",
    "anchor_count_curve_summary.csv": "E2_anchor_curve",
    "projector_distortion_summary.csv": "E3_projector_distortion",
    "row_level_margin_summary.csv": "E4_row_level_margin",
    "visual_proto_rescue_summary.csv": "E4_visual_proto_rescue",
}

ID_COLUMNS = {
    "variant", "mapping", "target_visual", "comparison", "checkpoint_name", "checkpoint_path",
    "projector_type", "feature_dim", "status", "source", "group", "support_bucket", "alpha",
    "candidate_scope", "class_scope", "dataset_name", "split", "name", "method", "target", "audit",
}

PREFERRED_METRICS = [
    "spearman", "spearman_r", "global_spearman", "offdiag_spearman", "top10_overlap", "top20_overlap",
    "neighbor_overlap", "mean_topk_overlap", "random_control_mean", "bootstrap_mean",
    "t2v_rank@1", "t2v_rank@5", "t2v_mean_rank", "v2t_rank@1", "v2t_rank@5", "v2t_mean_rank",
    "t2v_rank@1_mean", "v2t_rank@1_mean", "t2v_mean_rank_mean", "v2t_mean_rank_mean",
    "rank@1", "rank@5", "rank@10", "rank@20", "rank@50", "mean_rank", "mean_normalized_rank",
    "top1_rate", "top5_rate", "top1_person_rate", "positive_margin_gt_vs_top_wrong_rate",
    "mean_margin_gt_vs_top_wrong", "median_margin_gt_vs_top_wrong", "positive_margin_gt_vs_person_rate",
    "baseline_rank@1", "best_rank@1", "delta_rank@1", "best_alpha",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        x = float(s)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def _ident(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ["variant", "mapping", "target_visual", "comparison", "checkpoint_name", "projector_type", "support_bucket", "group", "alpha", "anchor_count", "seed_count", "eval_class_count", "class_count", "candidate_count", "status"]:
        if k in row and str(row.get(k, "")).strip() != "":
            out[k] = row.get(k)
    return out


def _numeric_metric_items(row: Mapping[str, Any]) -> list[tuple[str, float]]:
    keys = list(row.keys())
    preferred_present = [k for k in PREFERRED_METRICS if k in row]
    candidate_keys = preferred_present or [k for k in keys if k not in ID_COLUMNS]
    out = []
    for k in candidate_keys:
        val = _float_or_none(row.get(k))
        if val is not None:
            out.append((k, val))
    return out


def collect(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(analysis_root.rglob("*.csv")):
        audit = SUMMARY_FILENAMES.get(path.name)
        if not audit:
            continue
        rel = path.relative_to(analysis_root)
        rows = _read_csv(path)
        artifact_rows.append({"audit": audit, "artifact": str(rel), "row_count": len(rows)})
        for idx, row in enumerate(rows):
            ident = _ident(row)
            numeric = _numeric_metric_items(row)
            compact = {"audit": audit, "artifact": str(rel), "row_index": idx, **ident}
            for k, v in numeric:
                compact[k] = v
                long_rows.append({"audit": audit, "artifact": str(rel), "row_index": idx, **ident, "metric": k, "value": v})
            if numeric:
                compact_rows.append(compact)
    return artifact_rows, compact_rows, long_rows


def _make_takeover(output_root: Path, artifact_rows: Sequence[Mapping[str, Any]], compact_rows: Sequence[Mapping[str, Any]], long_rows: Sequence[Mapping[str, Any]]) -> None:
    by_audit: dict[str, int] = defaultdict(int)
    for r in artifact_rows:
        by_audit[str(r.get("audit"))] += int(r.get("row_count") or 0)
    lines: list[str] = []
    lines.append("# A9 Manifold Compatibility Audit TAKEOVER\n")
    lines.append("## Status\n")
    lines.append("- audit_type: `read_only`")
    lines.append(f"- artifact_count: `{len(artifact_rows)}`")
    lines.append(f"- compact_summary_rows: `{len(compact_rows)}`")
    lines.append(f"- metric_summary_rows: `{len(long_rows)}`")
    lines.append("\n## Audits collected\n")
    for audit in sorted(by_audit):
        lines.append(f"- {audit}: `{by_audit[audit]}` source rows")
    lines.append("\n## Interpretation checklist\n")
    lines.append("- If E1 graph metrics and E2 holdout metrics are both weak, manifold mismatch is a strong candidate cause.")
    lines.append("- If E1/E2 are acceptable but E4 row-level margins are weak, the blocker is class-level-to-row-level release rather than pure manifold mismatch.")
    lines.append("- If E3 projector distortion is high, the learned projector may be sacrificing class topology for local anchor fitting.")
    lines.append("- This collector does not recompute metrics; it only consolidates lightweight summaries generated by A8/A9 audits.\n")
    lines.append("## Main artifacts\n")
    lines.append(f"- `{output_root / 'analysis' / 'A9_manifold_compatibility_summary.csv'}`")
    lines.append(f"- `{output_root / 'analysis' / 'A9_manifold_compatibility_long_metrics.csv'}`")
    lines.append(f"- `{output_root / 'analysis' / 'A9_manifold_compatibility_artifact_manifest.csv'}`")
    (output_root / "A9_MANIFOLD_COMPATIBILITY_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect A9 manifold compatibility lightweight summaries.")
    p.add_argument("--output_root", required=True)
    p.add_argument("--analysis_root", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    analysis_root = Path(args.analysis_root).expanduser().resolve() if args.analysis_root else output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    artifact_rows, compact_rows, long_rows = collect(analysis_root)
    _write_csv(analysis_root / "A9_manifold_compatibility_artifact_manifest.csv", artifact_rows)
    _write_csv(analysis_root / "A9_manifold_compatibility_summary.csv", compact_rows)
    _write_csv(analysis_root / "A9_manifold_compatibility_long_metrics.csv", long_rows)
    payload = {
        "status": "PASS",
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "artifact_count": len(artifact_rows),
        "compact_summary_rows": len(compact_rows),
        "long_metric_rows": len(long_rows),
        "artifacts": {
            "manifest_csv": str(analysis_root / "A9_manifold_compatibility_artifact_manifest.csv"),
            "summary_csv": str(analysis_root / "A9_manifold_compatibility_summary.csv"),
            "long_metrics_csv": str(analysis_root / "A9_manifold_compatibility_long_metrics.csv"),
            "takeover": str(output_root / "A9_MANIFOLD_COMPATIBILITY_TAKEOVER.md"),
        },
    }
    _write_json(analysis_root / "A9_manifold_compatibility_summary.json", payload)
    _make_takeover(output_root, artifact_rows, compact_rows, long_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
