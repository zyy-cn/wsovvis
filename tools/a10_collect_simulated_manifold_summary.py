#!/usr/bin/env python3
"""Collect A10 simulated-manifold audit artifacts into compact summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if str(k) not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def collect(output_root: Path, analysis_root: Path) -> Dict[str, Any]:
    manifest: List[Dict[str, Any]] = []
    compact: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for p in sorted(analysis_root.rglob("*.csv")):
        if p.name in {"A10_artifact_manifest.csv", "A10_simulated_manifold_summary.csv", "A10_simulated_manifold_long_metrics.csv"}:
            continue
        rel = str(p.relative_to(analysis_root))
        try:
            rows = _read_csv(p)
        except Exception:
            rows = []
        manifest.append({"artifact": rel, "row_count": len(rows)})
        if "per_row" in p.name:
            continue
        for idx, row in enumerate(rows):
            crow = {"artifact": rel, "row_index": idx, **row}
            compact.append(crow)
            context = {kk: row.get(kk, "") for kk in (
                "feature_kind", "feature_name", "transform", "projector", "selected_projector",
                "candidate_scope", "target_visual", "anchor_count", "seed", "oracle_type")}
            for k, v in row.items():
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        long_rows.append({"artifact": rel, "row_index": idx, "metric": k, "value": fv, **context})
                except Exception:
                    continue
    _write_csv(analysis_root / "A10_artifact_manifest.csv", manifest)
    _write_csv(analysis_root / "A10_simulated_manifold_summary.csv", compact)
    _write_csv(analysis_root / "A10_simulated_manifold_long_metrics.csv", long_rows)
    payload = {
        "status": "PASS",
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "artifact_count": len(manifest),
        "compact_summary_rows": len(compact),
        "long_metric_rows": len(long_rows),
        "artifacts": {
            "manifest_csv": str(analysis_root / "A10_artifact_manifest.csv"),
            "summary_csv": str(analysis_root / "A10_simulated_manifold_summary.csv"),
            "long_metrics_csv": str(analysis_root / "A10_simulated_manifold_long_metrics.csv"),
        },
    }
    _write_json(analysis_root / "A10_simulated_manifold_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect A10 simulated-manifold audit outputs.")
    p.add_argument("--output_root", required=True)
    p.add_argument("--analysis_root", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    analysis_root = Path(args.analysis_root).expanduser().resolve() if args.analysis_root else output_root / "analysis"
    payload = collect(output_root, analysis_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
