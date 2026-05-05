#!/usr/bin/env python3
"""A8 visual prototype rescue probe.

Read-only oracle/probe scorer for checking whether DINO visual class prototypes
rescue train-good/val-fail classes better than projected CLIP text scoring.

This is diagnostic only: visual prototypes are not a formal weak-supervision
method claim. Train visual prototypes use train GT carriers; val rows are used
only for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _ensure_repo(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_gtceil_module(repo_root: Path):
    path = repo_root / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not path.exists():
        raise FileNotFoundError(f"missing helper module: {path}")
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _norm_id(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def _fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if x in (None, "", "None", "nan"):
            return default
        return float(x)
    except Exception:
        return default


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _zscore_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def _mean(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _median(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x]
    return float(np.median(xs)) if xs else float("nan")


def _load_rows_and_carriers(
    *,
    gtceil: Any,
    gt_carrier_path: Path,
    gt_identity_path: Path,
    gt_trajectory_path: Path,
    annotation_json: Path,
    max_rows: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any], Dict[str, Any]]:
    rows0, source_meta = gtceil._candidate_source_rows(
        gt_carrier_path=gt_carrier_path,
        gt_identity_path=gt_identity_path,
        gt_trajectory_path=gt_trajectory_path,
        annotation_json=annotation_json,
        max_rows=int(max_rows or 0),
    )
    carrier_matrix, keep_indices, vector_counters = gtceil._load_carrier_matrix(
        gt_carrier_path=gt_carrier_path,
        rows=rows0,
    )
    rows = [rows0[int(i)] for i in keep_indices]
    return rows, np.asarray(carrier_matrix, dtype=np.float32), dict(source_meta), dict(vector_counters)


def _build_visual_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Tuple[Dict[str, np.ndarray], Counter[str]]:
    sums: Dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    for i, row in enumerate(rows):
        rid = _norm_id(row.get("raw_category_id", row.get("gt_raw_id", row.get("category_id", ""))))
        if not rid:
            continue
        vec = np.asarray(carrier[i], dtype=np.float32)
        if rid not in sums:
            sums[rid] = np.zeros_like(vec, dtype=np.float32)
        sums[rid] += vec
        counts[rid] += 1
    out: Dict[str, np.ndarray] = {}
    for rid, vec in sums.items():
        if counts[rid] > 0:
            out[rid] = _l2_normalize((vec / float(counts[rid]))[None, :])[0]
    return out, counts


def _class_name(raw_id: str, class_name_map: Mapping[Any, Any]) -> str:
    for key in (raw_id, int(raw_id) if str(raw_id).isdigit() else raw_id):
        if key in class_name_map:
            return str(class_name_map[key])
    return f"raw_id_{raw_id}"


def _summary_for_rows(rows: Sequence[Mapping[str, Any]], group: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "group": group,
            "count": 0,
            "class_count": 0,
            "rank@1": float("nan"),
            "rank@5": float("nan"),
            "rank@10": float("nan"),
            "rank@20": float("nan"),
            "rank@50": float("nan"),
            "mean_rank": float("nan"),
            "median_rank": float("nan"),
            "large_negative_margin_count": 0,
        }
    ranks = [int(float(r["rank"])) for r in rows]
    margins = [_fnum(r.get("gt_vs_top1_margin")) for r in rows]
    return {
        "group": group,
        "count": n,
        "class_count": len({_norm_id(r.get("gt_raw_id")) for r in rows}),
        "rank@1": sum(x <= 1 for x in ranks) / n,
        "rank@5": sum(x <= 5 for x in ranks) / n,
        "rank@10": sum(x <= 10 for x in ranks) / n,
        "rank@20": sum(x <= 20 for x in ranks) / n,
        "rank@50": sum(x <= 50 for x in ranks) / n,
        "mean_rank": _mean(ranks),
        "median_rank": _median(ranks),
        "large_negative_margin_count": sum(1 for m in margins if m == m and m < -0.1),
        "mean_gt_vs_top1_margin": _mean(margins),
    }


def _rank_rows(
    *,
    eval_rows: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    score_mat: np.ndarray,
    class_name_map: Mapping[Any, Any],
    group_by_class: Mapping[str, Mapping[str, Any]],
    score_name: str,
) -> List[Dict[str, Any]]:
    id_to_pos = {str(rid): i for i, rid in enumerate(candidate_ids)}
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(eval_rows):
        gt = _norm_id(row.get("raw_category_id", row.get("gt_raw_id", row.get("category_id", ""))))
        if gt not in id_to_pos:
            continue
        cur = np.asarray(score_mat[i], dtype=np.float64)
        gt_pos = id_to_pos[gt]
        gt_score = float(cur[gt_pos])
        rank = int(np.sum(cur > cur[gt_pos]) + 1)
        order = np.argsort(-cur, kind="mergesort")
        top1_pos = int(order[0])
        top1_raw = str(candidate_ids[top1_pos])
        top1_score = float(cur[top1_pos])
        cls_info = group_by_class.get(gt, {})
        out.append({
            "score_name": score_name,
            "dataset_name": "lvvis_val",
            "trajectory_id": row.get("trajectory_id", ""),
            "video_id": row.get("video_id", ""),
            "clip_id": row.get("clip_id", row.get("video_id", "")),
            "gt_raw_id": gt,
            "gt_name": _class_name(gt, class_name_map),
            "quadrant": cls_info.get("quadrant", ""),
            "support_bucket": cls_info.get("support_bucket", ""),
            "candidate_scope": "train_visible_525",
            "candidate_count": len(candidate_ids),
            "rank": rank,
            "gt_score": gt_score,
            "top1_raw_id": top1_raw,
            "top1_name": _class_name(top1_raw, class_name_map),
            "top1_score": top1_score,
            "gt_vs_top1_margin": float(gt_score - top1_score),
            "is_gt_top1": rank <= 1,
            "is_gt_top5": rank <= 5,
        })
    return out


def _group_summaries(per_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in per_rows:
        groups["all"].append(r)
        quad = str(r.get("quadrant", "")) or "unknown_quadrant"
        groups[quad].append(r)
        sb = str(r.get("support_bucket", "")) or "unknown_support"
        groups[f"support_{sb}"].append(r)
    return [_summary_for_rows(rows, g) for g, rows in sorted(groups.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description="A8 visual prototype rescue probe")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--per_class_join", default="")
    ap.add_argument("--train_dataset_name", default="lvvis_train_base")
    ap.add_argument("--val_dataset_name", default="lvvis_val")
    ap.add_argument("--train_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/train_instances.json")
    ap.add_argument("--val_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/val_instances.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    asset_root = Path(args.asset_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo(repo_root)
    gtceil = _load_gtceil_module(repo_root)

    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else run_root / "outputs/a8_joint_train_time_dynamic_hungarian/lvvis_train_base/D-J3_pre1_dyn1_ep10/train/joint_train_time_dynamic_hungarian/a8_joint_train_time_dynamic_last.pth"
    per_class_path = Path(args.per_class_join) if args.per_class_join else run_root / "analysis/a8_dj3_semantic_boundary_bottleneck_audit/per_class_train_val_525_join.csv"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    if not per_class_path.exists():
        raise FileNotFoundError(f"missing per-class join: {per_class_path}")

    from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # type: ignore
        load_projector_bundle,
        load_text_vocab_with_names,
        score_infer_rows_matrix,
    )

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_projector_bundle(ckpt, device=device)
    text_vocab_ids_raw, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_root, args.train_dataset_name)
    text_ids = [_norm_id(x) for x in text_vocab_ids_raw]
    text_id_to_idx = {rid: i for i, rid in enumerate(text_ids)}

    per_class = _read_csv(per_class_path)
    group_by_class = {_norm_id(r.get("raw_id")): r for r in per_class}
    candidate_ids = [_norm_id(r.get("raw_id")) for r in per_class if _norm_id(r.get("raw_id")) in text_id_to_idx]
    if len(candidate_ids) == 0:
        raise RuntimeError("candidate_ids empty from per_class_join")
    candidate_idx = np.asarray([text_id_to_idx[rid] for rid in candidate_ids], dtype=np.int64)

    train_rows, train_carrier, train_meta, train_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.train_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.train_annotation_json),
        max_rows=args.max_rows,
    )
    val_rows_all, val_carrier_all, val_meta, val_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.val_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.val_annotation_json),
        max_rows=args.max_rows,
    )

    train_proto, train_counts = _build_visual_prototypes(train_rows, train_carrier)
    # Evaluate only target classes in train-visible 525 candidate set.
    keep_rows: List[Mapping[str, Any]] = []
    keep_vecs: List[np.ndarray] = []
    for row, vec in zip(val_rows_all, val_carrier_all):
        gt = _norm_id(row.get("raw_category_id", row.get("gt_raw_id", row.get("category_id", ""))))
        if gt in set(candidate_ids):
            keep_rows.append(row)
            keep_vecs.append(np.asarray(vec, dtype=np.float32))
    if not keep_rows:
        raise RuntimeError("no val rows with target in candidate_ids")
    val_carrier = _l2_normalize(np.stack(keep_vecs, axis=0))

    # Current projected text scorer.
    scores = score_infer_rows_matrix(
        carrier_matrix=val_carrier,
        bundle=bundle,
        text_matrix=text_matrix,
        show_progress=True,
    )
    text_score = np.asarray(scores["fused_logits"], dtype=np.float32)[:, candidate_idx]

    # Train visual prototype scorer; missing prototypes get very low scores.
    dim = val_carrier.shape[1]
    proto_mat = []
    proto_missing = []
    for rid in candidate_ids:
        if rid in train_proto:
            proto_mat.append(train_proto[rid])
            proto_missing.append(False)
        else:
            proto_mat.append(np.zeros((dim,), dtype=np.float32))
            proto_missing.append(True)
    proto_mat_np = _l2_normalize(np.stack(proto_mat, axis=0))
    visual_score = val_carrier @ proto_mat_np.T
    if any(proto_missing):
        visual_score[:, np.asarray(proto_missing, dtype=bool)] = -1e6

    text_z = _zscore_rows(text_score)
    visual_z = _zscore_rows(visual_score)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    all_per_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        mixed = alpha * text_z + (1.0 - alpha) * visual_z
        score_name = f"mix_z_alpha_text_{alpha:g}"
        per_rows = _rank_rows(
            eval_rows=keep_rows,
            candidate_ids=candidate_ids,
            score_mat=mixed,
            class_name_map=class_name_map,
            group_by_class=group_by_class,
            score_name=score_name,
        )
        all_per_rows.extend(per_rows)
        for srow in _group_summaries(per_rows):
            srow["score_name"] = score_name
            srow["alpha_text"] = alpha
            summary_rows.append(srow)

    # Also output pure raw scorers without z-score mixing, for interpretability.
    for score_name, mat in [("projected_text_raw_logit", text_score), ("train_visual_proto_cosine", visual_score)]:
        per_rows = _rank_rows(
            eval_rows=keep_rows,
            candidate_ids=candidate_ids,
            score_mat=mat,
            class_name_map=class_name_map,
            group_by_class=group_by_class,
            score_name=score_name,
        )
        all_per_rows.extend(per_rows)
        for srow in _group_summaries(per_rows):
            srow["score_name"] = score_name
            srow["alpha_text"] = "NA"
            summary_rows.append(srow)

    _write_csv(out_root / "visual_proto_rescue_per_row.csv", all_per_rows)
    _write_csv(out_root / "visual_proto_rescue_summary.csv", summary_rows)

    # Extract headline groups for baseline projected raw and best mixed over overfit.
    def find_summary(score_name: str, group: str) -> Optional[Dict[str, Any]]:
        for r in summary_rows:
            if str(r.get("score_name")) == score_name and str(r.get("group")) == group:
                return dict(r)
        return None

    overfit_rows = [r for r in summary_rows if str(r.get("group")) == "overfit_context_fail" and str(r.get("score_name", "")).startswith("mix_z_alpha")]
    best_overfit = None
    if overfit_rows:
        best_overfit = max(overfit_rows, key=lambda r: _fnum(r.get("rank@1"), -1.0))

    payload = {
        "status": "PASS",
        "output_root": str(out_root),
        "checkpoint_path": str(ckpt),
        "candidate_scope": "train_visible_525_from_per_class_join",
        "candidate_count": len(candidate_ids),
        "val_target_row_count": len(keep_rows),
        "alphas": alphas,
        "source_meta": {
            "train": train_meta,
            "val": val_meta,
            "train_vector_counters": train_vec_counters,
            "val_vector_counters": val_vec_counters,
        },
        "headline": {
            "projected_text_raw_all": find_summary("projected_text_raw_logit", "all"),
            "projected_text_raw_overfit60": find_summary("projected_text_raw_logit", "overfit_context_fail"),
            "train_visual_proto_all": find_summary("train_visual_proto_cosine", "all"),
            "train_visual_proto_overfit60": find_summary("train_visual_proto_cosine", "overfit_context_fail"),
            "best_mixed_overfit60": best_overfit,
        },
        "artifacts": {
            "visual_proto_rescue_per_row": str(out_root / "visual_proto_rescue_per_row.csv"),
            "visual_proto_rescue_summary": str(out_root / "visual_proto_rescue_summary.csv"),
        },
        "notes": [
            "This is an oracle/probe diagnostic, not a formal weak-supervision method.",
            "train_visual_proto_cosine uses train GT carrier class means as visual class prototypes.",
            "mix_z_alpha_text_* mixes row-zscored projected-text logits and train visual prototype cosine scores.",
        ],
    }
    (out_root / "visual_proto_rescue_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# A8 Visual Prototype Rescue Probe TAKEOVER",
        "",
        "- status: PASS",
        f"- output_root: {out_root}",
        f"- candidate_count: {len(candidate_ids)}",
        f"- val_target_row_count: {len(keep_rows)}",
        "",
        "## Headline",
        f"- projected_text_raw_overfit60: {payload['headline']['projected_text_raw_overfit60']}",
        f"- train_visual_proto_overfit60: {payload['headline']['train_visual_proto_overfit60']}",
        f"- best_mixed_overfit60: {payload['headline']['best_mixed_overfit60']}",
        "",
        "## Notes",
    ]
    for n in payload["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## Artifacts")
    for p in payload["artifacts"].values():
        lines.append(f"- {p}")
    (out_root / "VISUAL_PROTO_RESCUE_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
