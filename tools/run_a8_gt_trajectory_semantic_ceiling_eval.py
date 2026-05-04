#!/usr/bin/env python3
"""GT-trajectory semantic ceiling evaluation for A8 checkpoints.

Read-only evaluator. It replaces VideoCutLER proposal masks with GT trajectory
masks/carriers, then evaluates how far the current semantic scorer can go when
proposal/mask/trajectory quality is oracle-clean.

It writes three LV-VIS-formatted prediction sets:
  1. oracle_label: GT mask + GT category + score=1.0 (format/evaluator sanity)
  2. model_top1:  GT mask + model top-1 category over the selected vocabulary
  3. model_topK:  GT mask + model top-K categories, one prediction per class

No training, no checkpoint mutation, no GT use for model parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _bootstrap_repo_root() -> Path:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


REPO_ROOT = _bootstrap_repo_root()

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator  # noqa: E402
from videocutler.ext_stageb_ovvis.eval.external_lvvis import (  # noqa: E402
    ExternalLVVISEvalConfig,
    run_external_lvvis_eval,
)
from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # noqa: E402
    G8Paths,
    load_projector_bundle,
    load_text_vocab_with_names,
    load_video_meta,
    score_infer_rows_matrix,
    write_json,
    write_jsonl,
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


def _iter_jsonl(path: Path) -> Iterator[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield dict(obj)


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


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _dataset_to_split(dataset_name: str) -> str:
    text = str(dataset_name)
    if text.endswith("_val") or text == "lvvis_val":
        return "val"
    if text.endswith("_train") or text == "lvvis_train" or text == "lvvis_train_base":
        return "train"
    if "val" in text:
        return "val"
    return "train"


def _default_annotation_json(repo_root: Path, dataset_name: str) -> Path:
    split = _dataset_to_split(dataset_name)
    name = "val_instances.json" if split == "val" else "train_instances.json"
    return repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / name


def _load_video_meta_from_ann(annotation_json: Path) -> Dict[int, Dict[str, int]]:
    payload = _read_json(annotation_json)
    out: Dict[int, Dict[str, int]] = {}
    for video in payload.get("videos", []):
        if not isinstance(video, Mapping):
            continue
        vid = _safe_int(video.get("id"))
        if vid is None:
            continue
        file_names = video.get("file_names") or video.get("filenames") or []
        out[int(vid)] = {
            "video_id": int(vid),
            "clip_id": int(video.get("id", vid)),
            "length": int(video.get("length", len(file_names)) or len(file_names) or 0),
            "height": int(video.get("height", 0) or 0),
            "width": int(video.get("width", 0) or 0),
        }
    return out


def _load_annotation_by_id(annotation_json: Path) -> Dict[str, Record]:
    payload = _read_json(annotation_json)
    out: Dict[str, Record] = {}
    for ann in payload.get("annotations", []):
        if not isinstance(ann, Mapping):
            continue
        ann_id = ann.get("id")
        if ann_id is not None:
            out[str(ann_id)] = dict(ann)
        # Some exported GT trajectories use trajectory_id equal to video_track id.
        for key in ("trajectory_id", "track_id", "instance_id"):
            val = ann.get(key)
            if val is not None and str(val) not in out:
                out[str(val)] = dict(ann)
    return out


def _load_by_tid(path: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for idx, row in enumerate(_iter_jsonl(path)):
        tid = str(row.get("trajectory_id", row.get("carrier_id", ""))).strip()
        if not tid:
            tid = str(row.get("gt_track_id", row.get("id", idx))).strip()
        if tid:
            rec = dict(row)
            rec.setdefault("carrier_row_index", idx)
            out[str(tid)] = rec
    return out


def _normalize_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return arr.astype(np.float32)
    return (arr / norm).astype(np.float32)


def _load_carrier_matrix(
    *,
    gt_carrier_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, List[int], Counter]:
    parent = gt_carrier_path.parent
    vectors: List[np.ndarray] = []
    keep_indices: List[int] = []
    counters: Counter = Counter()
    for idx, row in enumerate(rows):
        carrier = row.get("carrier_record") if isinstance(row.get("carrier_record"), Mapping) else {}
        locator = carrier.get("z_norm_path") or carrier.get("vector_path") or carrier.get("feature_path") or carrier.get("z_path")
        if not locator:
            counters["missing_z_locator"] += 1
            continue
        try:
            vec = _normalize_vec(read_vector_from_locator(parent, str(locator)))
        except Exception:
            counters["carrier_vector_load_failed"] += 1
            continue
        vectors.append(vec)
        keep_indices.append(idx)
    if not vectors:
        raise RuntimeError(f"no GT carrier vectors loaded from {gt_carrier_path}; counters={dict(counters)}")
    return np.stack(vectors, axis=0).astype(np.float32), keep_indices, counters


def _dense_segmentations_from_record(record: Mapping[str, Any], *, video_length: int) -> Optional[List[Any]]:
    # Prefer already dense LV-VIS-style segmentations.
    for key in ("segmentations", "segmentation"):
        seg = record.get(key)
        if isinstance(seg, list) and seg:
            dense = list(seg)
            if video_length > 0 and len(dense) < video_length:
                dense.extend([None] * (video_length - len(dense)))
            return dense[:video_length] if video_length > 0 else dense
    frame_indices = record.get("frame_indices") or record.get("frames") or record.get("frame_ids")
    masks = record.get("masks_rle") or record.get("segmentations_rle") or record.get("rles") or record.get("masks")
    if isinstance(frame_indices, list) and isinstance(masks, list) and len(frame_indices) == len(masks):
        length = int(video_length) if int(video_length or 0) > 0 else (max([int(x) for x in frame_indices] or [-1]) + 1)
        dense: List[Any] = [None for _ in range(max(0, length))]
        for frame_idx, mask in zip(frame_indices, masks):
            idx = int(frame_idx)
            if idx < 0:
                continue
            if idx >= len(dense):
                dense.extend([None] * (idx + 1 - len(dense)))
            dense[idx] = mask
        return dense
    return None


def _raw_id_from_identity(identity: Mapping[str, Any], carrier: Mapping[str, Any], traj: Mapping[str, Any]) -> Optional[int]:
    for source in (identity, carrier, traj):
        for key in (
            "raw_category_id",
            "matched_gt_raw_id",
            "matched_gt_class_id",
            "gt_raw_id",
            "category_id",
            "gt_category_id",
        ):
            val = _safe_int(source.get(key), None)
            if val is not None:
                return int(val)
    # exports_gt trajectory_records historically used zero-based pred_label_raw.
    val = _safe_int(traj.get("pred_label_raw"), None)
    if val is not None:
        return int(val) + 1
    return None


def _candidate_source_rows(
    *,
    gt_carrier_path: Path,
    gt_identity_path: Path,
    gt_trajectory_path: Path,
    annotation_json: Path,
    max_rows: int,
) -> Tuple[List[Record], Dict[str, Any]]:
    carriers = _load_by_tid(gt_carrier_path)
    identities = _load_by_tid(gt_identity_path)
    trajectories = _load_by_tid(gt_trajectory_path) if gt_trajectory_path.is_file() else {}
    ann_by_id = _load_annotation_by_id(annotation_json)
    video_meta = _load_video_meta_from_ann(annotation_json)

    rows: List[Record] = []
    counters: Counter = Counter()
    for tid in sorted(carriers.keys()):
        if max_rows > 0 and len(rows) >= max_rows:
            break
        carrier = carriers.get(tid, {})
        identity = identities.get(tid, {})
        traj = trajectories.get(tid, {})
        if not traj:
            # Last-resort: use an annotation with the same id/track id when available.
            traj = ann_by_id.get(str(identity.get("gt_track_id", tid)), ann_by_id.get(str(tid), {}))
        video_id = _safe_int(identity.get("video_id"), _safe_int(carrier.get("video_id"), _safe_int(traj.get("video_id"), None)))
        if video_id is None:
            counters["missing_video_id"] += 1
            continue
        raw_id = _raw_id_from_identity(identity, carrier, traj)
        if raw_id is None:
            counters["missing_raw_category_id"] += 1
            continue
        video_length = int(video_meta.get(int(video_id), {}).get("length", 0) or 0)
        segs = _dense_segmentations_from_record(traj, video_length=video_length)
        if not segs:
            counters["missing_gt_segmentations"] += 1
            continue
        rows.append({
            "trajectory_id": str(tid),
            "video_id": int(video_id),
            "clip_id": _safe_int(identity.get("clip_id"), _safe_int(carrier.get("clip_id"), _safe_int(traj.get("clip_id"), int(video_id)))) or int(video_id),
            "raw_category_id": int(raw_id),
            "segmentations": segs,
            "carrier_record": dict(carrier),
            "identity_record": dict(identity),
            "trajectory_record": dict(traj),
        })
    return rows, {
        "carrier_count": len(carriers),
        "identity_count": len(identities),
        "trajectory_count": len(trajectories),
        "retained_rows_before_vector_load": len(rows),
        "counters": dict(counters),
    }


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


def _build_pred_rows(
    *,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    fused_logits: Optional[np.ndarray],
    known_probs: Optional[np.ndarray],
    text_vocab_ids: Sequence[int],
    topk: int,
    score_mode: str,
) -> Tuple[List[Record], List[Record], List[Record]]:
    pred: List[Record] = []
    diag: List[Record] = []
    row_scores: List[Record] = []
    raw_to_idx = {int(raw): int(i) for i, raw in enumerate(text_vocab_ids)}
    for row_idx, row in enumerate(rows):
        gt_raw = int(row["raw_category_id"])
        video_id = int(row["video_id"])
        segs = list(row["segmentations"])
        trajectory_id = str(row["trajectory_id"])
        if mode == "oracle_label":
            pred.append({
                "trajectory_id": f"{trajectory_id}::oracle",
                "video_id": video_id,
                "score": 1.0,
                "category_id": gt_raw,
                "segmentations": segs,
            })
            diag.append({
                "trajectory_id": trajectory_id,
                "mode": mode,
                "video_id": video_id,
                "gt_raw_id": gt_raw,
                "pred_raw_id": gt_raw,
                "score": 1.0,
                "gt_rank": 1,
                "is_gt_top1": True,
            })
            row_scores.append(dict(diag[-1]))
            continue
        if fused_logits is None or known_probs is None:
            raise RuntimeError("model modes require score matrices")
        logits = np.asarray(fused_logits[row_idx], dtype=np.float32)
        probs = np.asarray(known_probs[row_idx], dtype=np.float32)
        top_indices = _topk_indices(logits, 1 if mode == "model_top1" else int(topk))
        gt_idx = raw_to_idx.get(gt_raw)
        gt_rank = None
        gt_logit = None
        gt_prob = None
        if gt_idx is not None:
            gt_logit = float(logits[gt_idx])
            gt_prob = float(probs[gt_idx])
            gt_rank = int(np.sum(logits > logits[gt_idx]) + 1)
        for rank, idx in enumerate(top_indices, start=1):
            raw_id = int(text_vocab_ids[int(idx)])
            score_val = float(probs[int(idx)] if score_mode == "prob" else logits[int(idx)])
            pred.append({
                "trajectory_id": f"{trajectory_id}::{mode}::{rank}",
                "video_id": video_id,
                "score": float(score_val),
                "category_id": raw_id,
                "segmentations": segs,
            })
            diag.append({
                "trajectory_id": trajectory_id,
                "mode": mode,
                "video_id": video_id,
                "gt_raw_id": gt_raw,
                "pred_rank": int(rank),
                "pred_raw_id": raw_id,
                "score": float(score_val),
                "logit": float(logits[int(idx)]),
                "prob": float(probs[int(idx)]),
                "gt_rank": gt_rank,
                "gt_logit": gt_logit,
                "gt_prob": gt_prob,
                "is_gt_top1": bool(gt_rank == 1) if gt_rank is not None else False,
                "is_gt_in_topk": bool(gt_idx in top_indices) if gt_idx is not None else False,
            })
        if mode == "model_top1" and diag:
            row_scores.append(dict(diag[-1]))
        elif mode == "model_topk":
            row_scores.append({
                "trajectory_id": trajectory_id,
                "mode": mode,
                "video_id": video_id,
                "gt_raw_id": gt_raw,
                "gt_rank": gt_rank,
                "gt_logit": gt_logit,
                "gt_prob": gt_prob,
                "is_gt_top1": bool(gt_rank == 1) if gt_rank is not None else False,
                "is_gt_in_topk": bool(gt_idx in top_indices) if gt_idx is not None else False,
                "topk_raw_ids": [int(text_vocab_ids[i]) for i in top_indices[: min(len(top_indices), 20)]],
            })
    return pred, diag, row_scores


def _run_mode_eval(
    *,
    mode: str,
    output_root: Path,
    dataset_name: str,
    seed: int,
    smoke: bool,
    rows: Sequence[Mapping[str, Any]],
    fused_logits: Optional[np.ndarray],
    known_probs: Optional[np.ndarray],
    text_vocab_ids: Sequence[int],
    topk: int,
    score_mode: str,
) -> Dict[str, Any]:
    mode_root = output_root / mode
    paths = G8Paths(mode_root, dataset_name)
    pred, diag, row_scores = _build_pred_rows(
        mode=mode,
        rows=rows,
        fused_logits=fused_logits,
        known_probs=known_probs,
        text_vocab_ids=text_vocab_ids,
        topk=topk,
        score_mode=score_mode,
    )
    write_json(paths.pred_main_path, pred)
    write_json(paths.pred_diag_path, diag)
    write_jsonl(mode_root / "predictions" / dataset_name / "row_scores.jsonl", row_scores)
    eval_payload = run_external_lvvis_eval(ExternalLVVISEvalConfig(
        exp_name=f"{output_root.name}_{mode}",
        output_root=mode_root,
        seed=int(seed),
        smoke=bool(smoke),
    ))
    return {
        "mode": mode,
        "output_root": str(mode_root),
        "pred_main_path": str(paths.pred_main_path),
        "pred_diag_path": str(paths.pred_diag_path),
        "prediction_count": int(len(pred)),
        "row_score_count": int(len(row_scores)),
        "metrics": dict(eval_payload.get("metrics", {})),
        "external_metrics_path": str(paths.external_lvvis_metrics_path),
    }


def _parse_modes(text: str) -> List[str]:
    allowed = {"oracle_label", "model_top1", "model_topk"}
    modes: List[str] = []
    for part in str(text).replace(";", ",").split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"unknown mode {mode!r}; allowed={sorted(allowed)}")
        if mode not in modes:
            modes.append(mode)
    return modes or ["oracle_label", "model_top1", "model_topk"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT-trajectory semantic ceiling LV-VIS AP eval.")
    p.add_argument("--dataset_name", default="lvvis_val")
    p.add_argument("--output_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--gt_carrier_path", default="")
    p.add_argument("--gt_identity_path", default="")
    p.add_argument("--gt_trajectory_path", default="")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--score_mode", choices=["prob", "logit"], default="prob")
    p.add_argument("--modes", default="oracle_label,model_top1,model_topk")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
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
    annotation_json = Path(args.annotation_json).expanduser().resolve() if args.annotation_json else _default_annotation_json(REPO_ROOT, dataset_name)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()

    for path, desc in (
        (gt_carrier_path, "gt_carrier_path"),
        (gt_identity_path, "gt_identity_path"),
        (gt_trajectory_path, "gt_trajectory_path"),
        (annotation_json, "annotation_json"),
        (checkpoint_path, "checkpoint_path"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {desc}: {path}")

    modes = _parse_modes(args.modes)
    if args.smoke and int(args.max_rows or 0) <= 0:
        args.max_rows = 32

    rows0, source_meta = _candidate_source_rows(
        gt_carrier_path=gt_carrier_path,
        gt_identity_path=gt_identity_path,
        gt_trajectory_path=gt_trajectory_path,
        annotation_json=annotation_json,
        max_rows=int(args.max_rows or 0),
    )
    carrier_matrix, keep_indices, vector_counters = _load_carrier_matrix(gt_carrier_path=gt_carrier_path, rows=rows0)
    rows = [rows0[i] for i in keep_indices]

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    fused_logits: Optional[np.ndarray] = None
    known_probs: Optional[np.ndarray] = None
    text_vocab_ids: List[int] = []
    vocab_size = 0
    if any(mode.startswith("model_") for mode in modes):
        bundle = load_projector_bundle(checkpoint_path, device=device)
        text_vocab_ids, _text_records, text_matrix, _class_name_map = load_text_vocab_with_names(asset_root, dataset_name)
        scores = score_infer_rows_matrix(
            carrier_matrix=carrier_matrix,
            bundle=bundle,
            text_matrix=text_matrix,
            show_progress=bool(args.show_progress),
        )
        fused_logits = np.asarray(scores["fused_logits"], dtype=np.float32)
        known_probs = np.asarray(scores["known_probs"], dtype=np.float32)
        vocab_size = int(len(text_vocab_ids))
    else:
        # For oracle-only, still load text ids from annotation categories as fallback.
        ann = _read_json(annotation_json)
        text_vocab_ids = [int(c["id"]) for c in ann.get("categories", []) if isinstance(c, Mapping) and "id" in c]
        vocab_size = len(text_vocab_ids)

    mode_summaries: List[Record] = []
    for mode in modes:
        mode_summaries.append(_run_mode_eval(
            mode=mode,
            output_root=output_root,
            dataset_name=dataset_name,
            seed=int(args.seed),
            smoke=bool(args.smoke),
            rows=rows,
            fused_logits=fused_logits,
            known_probs=known_probs,
            text_vocab_ids=text_vocab_ids,
            topk=int(args.topk),
            score_mode=str(args.score_mode),
        ))

    compact_rows = []
    for item in mode_summaries:
        metrics = item.get("metrics", {})
        compact_rows.append({
            "mode": item.get("mode"),
            "prediction_count": item.get("prediction_count"),
            "AP": metrics.get("AP"),
            "AP50": metrics.get("AP50"),
            "AP75": metrics.get("AP75"),
            "mAPb": metrics.get("mAPb"),
            "mAPn": metrics.get("mAPn"),
            "external_metrics_path": item.get("external_metrics_path"),
        })
    _write_csv(output_root / "gt_trajectory_semantic_ceiling_metrics.csv", compact_rows)

    summary = {
        "status": "PASS",
        "dataset_name": dataset_name,
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "asset_root": str(asset_root),
        "gt_carrier_path": str(gt_carrier_path),
        "gt_identity_path": str(gt_identity_path),
        "gt_trajectory_path": str(gt_trajectory_path),
        "annotation_json": str(annotation_json),
        "retained_gt_rows": int(len(rows)),
        "retained_rows_before_vector_load": int(len(rows0)),
        "carrier_matrix_shape": [int(x) for x in carrier_matrix.shape],
        "vocab_size": int(vocab_size),
        "topk": int(args.topk),
        "score_mode": str(args.score_mode),
        "source_meta": source_meta,
        "vector_load_counters": dict(vector_counters),
        "modes": mode_summaries,
        "compact_metrics_csv": str(output_root / "gt_trajectory_semantic_ceiling_metrics.csv"),
    }
    _write_json(output_root / "gt_trajectory_semantic_ceiling_summary.json", summary)

    md_lines = [
        "# A8 GT-Trajectory Semantic Ceiling Eval",
        "",
        f"- dataset_name: `{dataset_name}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- retained_gt_rows: `{len(rows)}`",
        f"- vocab_size: `{vocab_size}`",
        "",
        "| mode | AP | AP50 | AP75 | mAPb | mAPn | predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact_rows:
        md_lines.append(
            f"| {row['mode']} | {row.get('AP')} | {row.get('AP50')} | {row.get('AP75')} | {row.get('mAPb')} | {row.get('mAPn')} | {row.get('prediction_count')} |"
        )
    (output_root / "GT_TRAJECTORY_SEMANTIC_CEILING_EVAL.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
