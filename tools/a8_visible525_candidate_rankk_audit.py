#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_gtceil_module():
    p = REPO_ROOT / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not p.is_file():
        raise FileNotFoundError(f"missing helper: {p}")
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper spec from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


GTCEIL = _load_gtceil_module()

from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # noqa: E402
    load_projector_bundle,
    load_text_vocab_with_names,
    score_infer_rows_matrix,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_int(x: Any, default=None):
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _raw_name(raw_id: int, class_name_map: Mapping[Any, Any]) -> str:
    for k in (raw_id, str(raw_id)):
        if k in class_name_map:
            return str(class_name_map[k])
    return ""


def _load_visible_ids(path: Path) -> set[int]:
    ids = set()
    for r in _read_csv(path):
        rid = _safe_int(r.get("raw_id"))
        if rid is None:
            continue
        if str(r.get("in_row_gap", "0")).strip() == "1":
            ids.add(int(rid))
    if len(ids) != 525:
        raise RuntimeError(f"expected 525 train-visible ids, got {len(ids)} from {path}")
    return ids


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ranks = [int(r["restricted_rank"]) for r in rows]
    if n == 0:
        return {
            "group": "target_visible_525_candidate_visible_525",
            "count": 0,
            "class_count": 0,
            "rank@1": 0.0,
            "rank@5": 0.0,
            "rank@10": 0.0,
            "rank@20": 0.0,
            "rank@50": 0.0,
            "mean_rank": None,
            "median_rank": None,
        }
    return {
        "group": "target_visible_525_candidate_visible_525",
        "count": n,
        "class_count": len({int(r["gt_raw_id"]) for r in rows}),
        "rank@1": sum(x <= 1 for x in ranks) / n,
        "rank@5": sum(x <= 5 for x in ranks) / n,
        "rank@10": sum(x <= 10 for x in ranks) / n,
        "rank@20": sum(x <= 20 for x in ranks) / n,
        "rank@50": sum(x <= 50 for x in ranks) / n,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical A8 visible-525 GT trajectory rank@K audit")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--asset_root", required=True)
    p.add_argument("--gt_carrier_path", required=True)
    p.add_argument("--gt_identity_path", required=True)
    p.add_argument("--gt_trajectory_path", required=True)
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--visible_csv", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_mode", choices=["logit", "prob"], default="logit")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--show_progress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset_name = str(args.dataset_name)
    output_root = Path(args.output_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    gt_carrier_path = Path(args.gt_carrier_path).expanduser().resolve()
    gt_identity_path = Path(args.gt_identity_path).expanduser().resolve()
    gt_trajectory_path = Path(args.gt_trajectory_path).expanduser().resolve()
    annotation_json = Path(args.annotation_json).expanduser().resolve()
    visible_csv = Path(args.visible_csv).expanduser().resolve()

    for p in [checkpoint_path, gt_carrier_path, gt_identity_path, gt_trajectory_path, annotation_json, visible_csv]:
        if not p.is_file():
            raise FileNotFoundError(p)

    visible_ids = _load_visible_ids(visible_csv)

    rows0, source_meta = GTCEIL._candidate_source_rows(
        gt_carrier_path=gt_carrier_path,
        gt_identity_path=gt_identity_path,
        gt_trajectory_path=gt_trajectory_path,
        annotation_json=annotation_json,
        max_rows=int(args.max_rows or 0),
    )
    carrier_matrix, keep_indices, vector_counters = GTCEIL._load_carrier_matrix(
        gt_carrier_path=gt_carrier_path,
        rows=rows0,
    )
    rows = [rows0[i] for i in keep_indices]

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_projector_bundle(checkpoint_path, device=device)
    text_vocab_ids, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_root, dataset_name)

    raw_to_idx = {int(raw): int(i) for i, raw in enumerate(text_vocab_ids)}
    visible_sorted = [rid for rid in sorted(visible_ids) if rid in raw_to_idx]
    if len(visible_sorted) != 525:
        raise RuntimeError(f"visible candidate ids in text vocab expected 525, got {len(visible_sorted)}")

    visible_cols = np.asarray([raw_to_idx[rid] for rid in visible_sorted], dtype=np.int64)
    visible_pos = {rid: i for i, rid in enumerate(visible_sorted)}

    scores = score_infer_rows_matrix(
        carrier_matrix=carrier_matrix,
        bundle=bundle,
        text_matrix=text_matrix,
        show_progress=bool(args.show_progress),
    )
    logits = np.asarray(scores["fused_logits"], dtype=np.float32)
    probs = np.asarray(scores["known_probs"], dtype=np.float32)
    score_mat = probs if args.score_mode == "prob" else logits

    per_rows: List[Dict[str, Any]] = []
    skipped_not_visible = 0
    skipped_missing_gt = 0

    for i, row in enumerate(rows):
        gt_raw = int(row["raw_category_id"])
        if gt_raw not in visible_ids:
            skipped_not_visible += 1
            continue
        if gt_raw not in visible_pos:
            skipped_missing_gt += 1
            continue

        cur_scores = score_mat[i, visible_cols]
        gt_pos = int(visible_pos[gt_raw])
        gt_score = float(cur_scores[gt_pos])
        rank = int(np.sum(cur_scores > cur_scores[gt_pos]) + 1)

        order = np.argsort(-cur_scores, kind="mergesort")
        top1_pos = int(order[0])
        top1_raw = int(visible_sorted[top1_pos])
        top1_score = float(cur_scores[top1_pos])

        per_rows.append({
            "dataset_name": dataset_name,
            "trajectory_id": row.get("trajectory_id"),
            "video_id": int(row.get("video_id", -1)),
            "clip_id": int(row.get("clip_id", row.get("video_id", -1))),
            "gt_raw_id": gt_raw,
            "gt_name": _raw_name(gt_raw, class_name_map),
            "candidate_scope": "train_visible_525",
            "candidate_count": 525,
            "restricted_rank": rank,
            "gt_score": gt_score,
            "top1_raw_id": top1_raw,
            "top1_name": _raw_name(top1_raw, class_name_map),
            "top1_score": top1_score,
            "margin_gt_minus_top1": float(gt_score - top1_score),
            "is_gt_top1": rank <= 1,
            "is_gt_top5": rank <= 5,
            "is_gt_top10": rank <= 10,
            "is_gt_top20": rank <= 20,
            "is_gt_top50": rank <= 50,
        })

    summary_row = _summarize(per_rows)
    summary_rows = [summary_row]

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "visible525_candidate_rankk_per_row.csv", per_rows)
    _write_csv(output_root / "visible525_candidate_rankk_summary_by_group.csv", summary_rows)

    summary = {
        "status": "PASS",
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "visible_csv": str(visible_csv),
        "score_mode": str(args.score_mode),
        "candidate_scope": "train_visible_525",
        "candidate_count": 525,
        "source_row_count_before_vector_load": int(len(rows0)),
        "source_row_count_after_vector_load": int(len(rows)),
        "retained_target_visible_rows": int(len(per_rows)),
        "skipped_not_visible_target_rows": int(skipped_not_visible),
        "skipped_missing_gt_in_visible_vocab": int(skipped_missing_gt),
        "carrier_matrix_shape": [int(x) for x in carrier_matrix.shape],
        "source_meta": source_meta,
        "vector_counters": dict(vector_counters),
        "summary_by_group": summary_rows,
        "primary_metric": {
            "name": "canonical_visible525_rank@1",
            "value": summary_row.get("rank@1"),
            "scope": "target=train_visible_525,candidate=train_visible_525,source=GT trajectory",
        },
        "artifacts": {
            "per_row_csv": str(output_root / "visible525_candidate_rankk_per_row.csv"),
            "summary_by_group_csv": str(output_root / "visible525_candidate_rankk_summary_by_group.csv"),
        },
    }
    _write_json(output_root / "visible525_candidate_rankk_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
