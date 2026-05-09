#!/usr/bin/env python3
"""Val-side rank@K audit for A8 semantic scorer.

Read-only audit. It scores GT trajectories on lvvis_val with a checkpoint and
reports where each GT class ranks in the full LV-VIS vocabulary, split by
official base/novel ids.

This complements AP: it asks whether the correct class is top-1/top-K before
mask/proposal/NMS effects.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_gtceil_module():
    p = REPO_ROOT / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"missing required helper {p}; deploy/run the GT trajectory semantic ceiling overlay first"
        )
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module spec from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

GTCEIL = _load_gtceil_module()

from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # noqa: E402
    load_projector_bundle,
    load_text_vocab_for_checkpoint,
    score_infer_rows_matrix,
)

Record = Dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(str(key))
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _find_ids(obj: Any, names: Sequence[str]) -> set[int]:
    if isinstance(obj, Mapping):
        for n in names:
            if n in obj and isinstance(obj[n], (list, tuple, set)):
                return {int(x) for x in obj[n]}
        for v in obj.values():
            got = _find_ids(v, names)
            if got:
                return got
    return set()


def _load_split(split_json: Path) -> Tuple[set[int], set[int]]:
    payload = _read_json(split_json)
    base = _find_ids(payload, ["base_raw_ids", "base_category_ids", "base_ids", "base"])
    novel = _find_ids(payload, ["novel_raw_ids", "novel_category_ids", "novel_ids", "novel"])
    if not base or not novel:
        raise RuntimeError(f"failed to locate base/novel raw ids in {split_json}")
    return base, novel


def _topk_indices(scores: np.ndarray, k: int) -> List[int]:
    n = int(scores.shape[0])
    if n <= 0:
        return []
    k = min(max(int(k), 1), n)
    if k == n:
        order = np.argsort(-scores, kind="mergesort")
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="mergesort")]
    return [int(x) for x in order.tolist()]


def _raw_name(raw_id: int, class_name_map: Mapping[Any, Any]) -> str:
    for key in (raw_id, str(raw_id)):
        if key in class_name_map:
            return str(class_name_map[key])
    return ""


def _percent(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _summarize(rows: Sequence[Mapping[str, Any]], *, prefix: str = "") -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {f"{prefix}count" if prefix else "count": n}
    if not n:
        for k in [1, 5, 10, 20, 50]:
            out[f"{prefix}rank@{k}" if prefix else f"rank@{k}"] = 0.0
        out[f"{prefix}mean_rank" if prefix else "mean_rank"] = None
        out[f"{prefix}mean_normalized_rank" if prefix else "mean_normalized_rank"] = None
        return out
    ranks = [int(r["gt_rank"]) for r in rows if r.get("gt_rank") not in (None, "")]
    vocab_sizes = [int(r.get("vocab_size", 0) or 0) for r in rows]
    for k in [1, 5, 10, 20, 50]:
        out[f"{prefix}rank@{k}" if prefix else f"rank@{k}"] = _percent(sum(1 for rr in ranks if rr <= k), n)
    out[f"{prefix}mean_rank" if prefix else "mean_rank"] = float(np.mean(ranks)) if ranks else None
    norm_vals = []
    for r in rows:
        rank = _safe_int(r.get("gt_rank"), None)
        vocab = _safe_int(r.get("vocab_size"), None)
        if rank is None or vocab is None or vocab <= 1:
            continue
        norm_vals.append((rank - 1) / float(vocab - 1))
    out[f"{prefix}mean_normalized_rank" if prefix else "mean_normalized_rank"] = float(np.mean(norm_vals)) if norm_vals else None
    out[f"{prefix}median_rank" if prefix else "median_rank"] = float(np.median(ranks)) if ranks else None
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8 val-side GT-class rank@K audit split by official base/novel ids.")
    p.add_argument("--dataset_name", default="lvvis_val")
    p.add_argument("--output_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--gt_carrier_path", default="")
    p.add_argument("--gt_identity_path", default="")
    p.add_argument("--gt_trajectory_path", default="")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="/mnt/sda/zyy/code/wsovvis/package/reference/lvvis_official_base_novel_split.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_mode", choices=["logit", "prob"], default="logit")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--top_suppressor_limit", type=int, default=50)
    p.add_argument("--show_progress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = str(args.dataset_name)
    output_root = Path(args.output_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    gt_carrier_path = Path(args.gt_carrier_path).expanduser().resolve() if args.gt_carrier_path else asset_root / "carrier_bank_gt" / dataset_name / "carrier_records.jsonl"
    gt_identity_path = Path(args.gt_identity_path).expanduser().resolve() if args.gt_identity_path else asset_root / "carrier_bank_gt" / dataset_name / "gt_carrier_identity_binding.jsonl"
    gt_trajectory_path = Path(args.gt_trajectory_path).expanduser().resolve() if args.gt_trajectory_path else asset_root / "exports_gt" / dataset_name / "trajectory_records.jsonl"
    annotation_json = Path(args.annotation_json).expanduser().resolve() if args.annotation_json else REPO_ROOT / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    split_json = Path(args.split_json).expanduser().resolve()

    for path, desc in [
        (gt_carrier_path, "gt_carrier_path"),
        (gt_identity_path, "gt_identity_path"),
        (gt_trajectory_path, "gt_trajectory_path"),
        (annotation_json, "annotation_json"),
        (checkpoint_path, "checkpoint_path"),
        (split_json, "split_json"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {desc}: {path}")

    base_ids, novel_ids = _load_split(split_json)

    rows0, source_meta = GTCEIL._candidate_source_rows(
        gt_carrier_path=gt_carrier_path,
        gt_identity_path=gt_identity_path,
        gt_trajectory_path=gt_trajectory_path,
        annotation_json=annotation_json,
        max_rows=int(args.max_rows or 0),
    )
    carrier_matrix, keep_indices, vector_counters = GTCEIL._load_carrier_matrix(gt_carrier_path=gt_carrier_path, rows=rows0)
    rows = [rows0[i] for i in keep_indices]

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_projector_bundle(checkpoint_path, device=device)
    text_vocab_ids, _text_records, text_matrix, class_name_map, text_bank_eval_summary = load_text_vocab_for_checkpoint(
        asset_root,
        dataset_name,
        bundle.checkpoint_payload,
    )
    text_bank_eval_summary = dict(text_bank_eval_summary)
    text_bank_eval_summary["loaded_by_val_side_rankk_audit"] = bool(text_bank_eval_summary.get("variant") != "clip_current")
    raw_to_idx = {int(raw): int(i) for i, raw in enumerate(text_vocab_ids)}
    scores = score_infer_rows_matrix(
        carrier_matrix=carrier_matrix,
        bundle=bundle,
        text_matrix=text_matrix,
        show_progress=bool(args.show_progress),
    )
    logits = np.asarray(scores["fused_logits"], dtype=np.float32)
    probs = np.asarray(scores["known_probs"], dtype=np.float32)
    score_mat = probs if args.score_mode == "prob" else logits

    per_rows: List[Record] = []
    missing_gt = 0
    for i, row in enumerate(rows):
        gt_raw = int(row["raw_category_id"])
        group = "base" if gt_raw in base_ids else ("novel" if gt_raw in novel_ids else "unknown")
        gt_idx = raw_to_idx.get(gt_raw)
        if gt_idx is None:
            missing_gt += 1
            continue
        cur_logits = logits[i]
        cur_probs = probs[i]
        cur_scores = score_mat[i]
        gt_score = float(cur_scores[gt_idx])
        gt_logit = float(cur_logits[gt_idx])
        gt_prob = float(cur_probs[gt_idx])
        rank = int(np.sum(cur_scores > cur_scores[gt_idx]) + 1)
        top = _topk_indices(cur_scores, 50)
        top1_idx = int(top[0]) if top else -1
        top1_raw = int(text_vocab_ids[top1_idx]) if top1_idx >= 0 else None
        top1_group = "base" if top1_raw in base_ids else ("novel" if top1_raw in novel_ids else "unknown") if top1_raw is not None else "unknown"
        second_idx = int(top[1]) if len(top) > 1 else -1
        margin_top1_minus_gt = float(cur_scores[top1_idx] - cur_scores[gt_idx]) if top1_idx >= 0 else None
        gt_minus_best_wrong = None
        if rank == 1 and second_idx >= 0:
            gt_minus_best_wrong = float(cur_scores[gt_idx] - cur_scores[second_idx])
        per_rows.append({
            "trajectory_id": row.get("trajectory_id"),
            "video_id": int(row.get("video_id", -1)),
            "clip_id": int(row.get("clip_id", row.get("video_id", -1))),
            "gt_raw_id": gt_raw,
            "gt_name": _raw_name(gt_raw, class_name_map),
            "gt_group": group,
            "gt_rank": rank,
            "gt_normalized_rank": (rank - 1) / float(max(1, len(text_vocab_ids) - 1)),
            "gt_logit": gt_logit,
            "gt_prob": gt_prob,
            "gt_score": gt_score,
            "top1_raw_id": top1_raw,
            "top1_name": _raw_name(int(top1_raw), class_name_map) if top1_raw is not None else "",
            "top1_group": top1_group,
            "top1_score": float(cur_scores[top1_idx]) if top1_idx >= 0 else None,
            "top1_logit": float(cur_logits[top1_idx]) if top1_idx >= 0 else None,
            "top1_prob": float(cur_probs[top1_idx]) if top1_idx >= 0 else None,
            "margin_top1_minus_gt": margin_top1_minus_gt,
            "gt_minus_best_wrong_margin_if_top1": gt_minus_best_wrong,
            "is_gt_top1": rank <= 1,
            "is_gt_top5": rank <= 5,
            "is_gt_top10": rank <= 10,
            "is_gt_top20": rank <= 20,
            "is_gt_top50": rank <= 50,
            "top5_raw_ids": ";".join(str(int(text_vocab_ids[j])) for j in top[:5]),
            "top10_raw_ids": ";".join(str(int(text_vocab_ids[j])) for j in top[:10]),
            "vocab_size": len(text_vocab_ids),
        })

    groups = ["all", "base", "novel", "unknown"]
    group_rows: Dict[str, List[Record]] = {
        "all": list(per_rows),
        "base": [r for r in per_rows if r.get("gt_group") == "base"],
        "novel": [r for r in per_rows if r.get("gt_group") == "novel"],
        "unknown": [r for r in per_rows if r.get("gt_group") == "unknown"],
    }
    summary_rows: List[Record] = []
    for g in groups:
        if g == "unknown" and not group_rows[g]:
            continue
        row = {"group": g}
        row.update(_summarize(group_rows[g]))
        summary_rows.append(row)

    suppressor_rows: List[Record] = []
    for g in groups:
        rows_g = group_rows[g]
        wrong = [r for r in rows_g if int(r.get("gt_rank", 999999)) > 1]
        cnt: Counter = Counter(int(r["top1_raw_id"]) for r in wrong if r.get("top1_raw_id") not in (None, ""))
        for raw, n in cnt.most_common(int(args.top_suppressor_limit)):
            suppressor_rows.append({
                "group": g,
                "top1_raw_id": int(raw),
                "top1_name": _raw_name(int(raw), class_name_map),
                "top1_group": "base" if raw in base_ids else ("novel" if raw in novel_ids else "unknown"),
                "wrong_count": int(n),
                "wrong_rate_within_group": _percent(int(n), len(rows_g)),
                "wrong_rate_within_wrong": _percent(int(n), len(wrong)),
            })

    by_class: Dict[int, List[Record]] = defaultdict(list)
    for r in per_rows:
        by_class[int(r["gt_raw_id"])].append(r)
    class_rows: List[Record] = []
    for raw, rs in sorted(by_class.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out: Record = {
            "gt_raw_id": int(raw),
            "gt_name": _raw_name(int(raw), class_name_map),
            "gt_group": "base" if raw in base_ids else ("novel" if raw in novel_ids else "unknown"),
        }
        out.update(_summarize(rs))
        suppress = Counter(int(r["top1_raw_id"]) for r in rs if int(r.get("gt_rank", 999999)) > 1 and r.get("top1_raw_id") not in (None, ""))
        if suppress:
            top_raw, top_n = suppress.most_common(1)[0]
            out.update({
                "top_suppressor_raw_id": int(top_raw),
                "top_suppressor_name": _raw_name(int(top_raw), class_name_map),
                "top_suppressor_count": int(top_n),
            })
        class_rows.append(out)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "val_side_rankk_per_row.csv", per_rows)
    _write_csv(output_root / "val_side_rankk_summary_by_group.csv", summary_rows)
    _write_csv(output_root / "val_side_rankk_top_suppressors.csv", suppressor_rows)
    _write_csv(output_root / "val_side_rankk_summary_by_class.csv", class_rows)

    summary = {
        "status": "PASS",
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "asset_root": str(asset_root),
        "gt_carrier_path": str(gt_carrier_path),
        "gt_identity_path": str(gt_identity_path),
        "gt_trajectory_path": str(gt_trajectory_path),
        "annotation_json": str(annotation_json),
        "split_json": str(split_json),
        "score_mode": str(args.score_mode),
        "text_bank": text_bank_eval_summary,
        "vocab_size": int(len(text_vocab_ids)),
        "base_id_count": int(len(base_ids)),
        "novel_id_count": int(len(novel_ids)),
        "retained_rows_before_vector_load": int(len(rows0)),
        "retained_rows": int(len(per_rows)),
        "missing_gt_in_vocab_count": int(missing_gt),
        "carrier_matrix_shape": [int(x) for x in carrier_matrix.shape],
        "source_meta": source_meta,
        "vector_load_counters": dict(vector_counters),
        "summary_by_group": summary_rows,
        "artifacts": {
            "per_row_csv": str(output_root / "val_side_rankk_per_row.csv"),
            "summary_by_group_csv": str(output_root / "val_side_rankk_summary_by_group.csv"),
            "top_suppressors_csv": str(output_root / "val_side_rankk_top_suppressors.csv"),
            "summary_by_class_csv": str(output_root / "val_side_rankk_summary_by_class.csv"),
        },
    }
    _write_json(output_root / "val_side_rankk_summary.json", summary)

    md = [
        "# A8 Val-side Rank@K Audit",
        "",
        f"- dataset_name: `{dataset_name}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- score_mode: `{args.score_mode}`",
        f"- retained_rows: `{len(per_rows)}`",
        f"- vocab_size: `{len(text_vocab_ids)}`",
        "",
        "| group | count | rank@1 | rank@5 | rank@10 | rank@20 | rank@50 | mean_rank | mean_normalized_rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r.get('group')} | {r.get('count')} | {r.get('rank@1')} | {r.get('rank@5')} | {r.get('rank@10')} | {r.get('rank@20')} | {r.get('rank@50')} | {r.get('mean_rank')} | {r.get('mean_normalized_rank')} |"
        )
    (output_root / "A8_VAL_SIDE_RANKK_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
