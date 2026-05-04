#!/usr/bin/env python3
"""VideoCutLER trajectory oracle-class ceiling evaluation.

This is a read-only AP ceiling tool. It keeps the existing VideoCutLER/mainline
trajectory masks and replaces their predicted class with the GT class matched by
an offline GT sidecar. It answers:

  If VideoCutLER trajectories were classified perfectly, how high could LV-VIS AP go?

Modes:
  - oracle_class_score1:      matched VideoCutLER trajectory + GT class + score=1.0
  - oracle_class_iou_score:  matched VideoCutLER trajectory + GT class + score=matched IoU

No training, no checkpoint mutation, no feature extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

Record = Dict[str, Any]


def _bootstrap_repo_root() -> Path:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


REPO_ROOT = _bootstrap_repo_root()

from videocutler.ext_stageb_ovvis.eval.external_lvvis import (  # noqa: E402
    ExternalLVVISEvalConfig,
    run_external_lvvis_eval,
)
from videocutler.ext_stageb_ovvis.eval.g8_bridge import G8Paths, write_json, write_jsonl  # noqa: E402


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


def _dataset_split(dataset_name: str) -> str:
    text = str(dataset_name)
    if "val" in text:
        return "val"
    return "train"


def _default_annotation_json(repo_root: Path, dataset_name: str) -> Path:
    split = _dataset_split(dataset_name)
    name = "val_instances.json" if split == "val" else "train_instances.json"
    return repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / name


def _default_sidecar_candidates(asset_root: Path, dataset_name: str) -> List[Path]:
    split = _dataset_split(dataset_name)
    return [
        asset_root / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_match_{split}_mainline.jsonl",
        asset_root / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_identity_{split}_mainline.jsonl",
        asset_root / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_match_{split}_gt.jsonl",
        asset_root / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_identity_{split}_gt.jsonl",
        REPO_ROOT / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_match_{split}_mainline.jsonl",
        REPO_ROOT / "gt_sidecar_bank" / dataset_name / "mainline" / f"trajectory_gt_identity_{split}_mainline.jsonl",
    ]


def _resolve_sidecar(asset_root: Path, dataset_name: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for p in _default_sidecar_candidates(asset_root, dataset_name):
        if p.is_file():
            return p.resolve()
    tried = "\n".join(str(p) for p in _default_sidecar_candidates(asset_root, dataset_name))
    raise FileNotFoundError(f"could not auto-resolve gt sidecar for {dataset_name}; tried:\n{tried}")


def _load_by_tid(path: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    dup = 0
    for idx, row in enumerate(_iter_jsonl(path)):
        tid = str(row.get("trajectory_id", row.get("carrier_id", row.get("gt_track_id", "")))).strip()
        if not tid:
            tid = str(row.get("id", idx)).strip()
        if not tid:
            continue
        rec = dict(row)
        rec.setdefault("_row_index", idx)
        if tid in out:
            dup += 1
            continue
        out[tid] = rec
    return out


def _load_video_meta(annotation_json: Path) -> Dict[int, Dict[str, int]]:
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
            "length": int(video.get("length", len(file_names)) or len(file_names) or 0),
            "height": int(video.get("height", 0) or 0),
            "width": int(video.get("width", 0) or 0),
        }
    return out


def _dense_segmentations(record: Mapping[str, Any], *, video_length: int) -> Optional[List[Any]]:
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


def _extract_raw_id(sidecar: Mapping[str, Any]) -> Optional[int]:
    # Prefer canonical field when present because current audits use canonical raw-id space.
    for key in (
        "matched_gt_raw_id_canonical",
        "gt_raw_id_canonical",
        "matched_gt_raw_id",
        "matched_gt_class_id",
        "gt_raw_id",
        "raw_category_id",
        "category_id",
        "gt_category_id",
    ):
        val = _safe_int(sidecar.get(key), None)
        if val is not None:
            return int(val)
    return None


def _extract_iou(sidecar: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "match_iou_video",
        "match_iou_mean",
        "mean_iou",
        "best_iou",
        "iou",
        "matched_gt_iou",
        "mask_iou",
        "video_iou",
    ):
        val = _safe_float(sidecar.get(key), None)
        if val is not None:
            return float(max(0.0, min(1.0, val)))
    return None


def _is_usable(sidecar: Mapping[str, Any], *, min_iou: float) -> bool:
    if "audit_usable" in sidecar and not bool(sidecar.get("audit_usable")):
        return False
    if str(sidecar.get("match_quality", "")).strip().lower() in {"bad", "failed", "reject", "rejected"}:
        return False
    raw = _extract_raw_id(sidecar)
    if raw is None:
        return False
    iou = _extract_iou(sidecar)
    if iou is not None and float(iou) < float(min_iou):
        return False
    return True


def _collect_rows(
    *,
    trajectory_path: Path,
    sidecar_path: Path,
    annotation_json: Path,
    min_iou: float,
    max_rows: int,
) -> Tuple[List[Record], Dict[str, Any]]:
    traj_by_tid = _load_by_tid(trajectory_path)
    side_by_tid = _load_by_tid(sidecar_path)
    video_meta = _load_video_meta(annotation_json)
    rows: List[Record] = []
    counters: Counter = Counter()
    iou_values: List[float] = []
    for tid in sorted(traj_by_tid.keys()):
        if max_rows > 0 and len(rows) >= max_rows:
            break
        traj = traj_by_tid[tid]
        side = side_by_tid.get(tid)
        if side is None:
            counters["missing_sidecar"] += 1
            continue
        if not _is_usable(side, min_iou=float(min_iou)):
            counters["sidecar_not_usable_or_no_raw_id"] += 1
            continue
        raw_id = _extract_raw_id(side)
        if raw_id is None:
            counters["missing_raw_id"] += 1
            continue
        video_id = _safe_int(traj.get("video_id"), _safe_int(traj.get("clip_id"), _safe_int(side.get("video_id"), _safe_int(side.get("clip_id"), None))))
        if video_id is None:
            counters["missing_video_id"] += 1
            continue
        video_length = int(video_meta.get(int(video_id), {}).get("length", 0) or 0)
        segs = _dense_segmentations(traj, video_length=video_length)
        if not segs:
            counters["missing_segmentations"] += 1
            continue
        iou = _extract_iou(side)
        if iou is not None:
            iou_values.append(float(iou))
        rows.append({
            "trajectory_id": str(tid),
            "video_id": int(video_id),
            "clip_id": _safe_int(traj.get("clip_id"), _safe_int(side.get("clip_id"), int(video_id))) or int(video_id),
            "oracle_raw_id": int(raw_id),
            "match_iou": float(iou) if iou is not None else None,
            "segmentations": segs,
            "trajectory_record": dict(traj),
            "sidecar_record": dict(side),
        })
    meta = {
        "trajectory_count": int(len(traj_by_tid)),
        "sidecar_count": int(len(side_by_tid)),
        "retained_rows": int(len(rows)),
        "counters": dict(counters),
        "match_iou_count": int(len(iou_values)),
        "match_iou_mean": float(sum(iou_values) / len(iou_values)) if iou_values else None,
        "match_iou_min": float(min(iou_values)) if iou_values else None,
        "match_iou_max": float(max(iou_values)) if iou_values else None,
    }
    return rows, meta


def _score_for_row(row: Mapping[str, Any], mode: str) -> float:
    if mode == "oracle_class_score1":
        return 1.0
    if mode == "oracle_class_iou_score":
        val = _safe_float(row.get("match_iou"), None)
        if val is None:
            # Keep all matched predictions valid but rank missing-IoU rows conservatively.
            return 1e-6
        return float(max(1e-6, min(1.0, val)))
    raise ValueError(f"unknown mode {mode}")


def _build_predictions(rows: Sequence[Mapping[str, Any]], *, mode: str) -> Tuple[List[Record], List[Record]]:
    pred: List[Record] = []
    diag: List[Record] = []
    for row in rows:
        score = _score_for_row(row, mode)
        tid = str(row["trajectory_id"])
        raw = int(row["oracle_raw_id"])
        vid = int(row["video_id"])
        pred.append({
            "trajectory_id": f"{tid}::{mode}",
            "video_id": vid,
            "score": float(score),
            "category_id": raw,
            "segmentations": list(row["segmentations"]),
        })
        diag.append({
            "trajectory_id": tid,
            "mode": mode,
            "video_id": vid,
            "clip_id": row.get("clip_id"),
            "oracle_raw_id": raw,
            "score": float(score),
            "match_iou": row.get("match_iou"),
        })
    return pred, diag


def _run_mode_eval(
    *,
    mode: str,
    output_root: Path,
    dataset_name: str,
    seed: int,
    smoke: bool,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    mode_root = output_root / mode
    paths = G8Paths(mode_root, dataset_name)
    pred, diag = _build_predictions(rows, mode=mode)
    write_json(paths.pred_main_path, pred)
    write_json(paths.pred_diag_path, diag)
    write_jsonl(mode_root / "predictions" / dataset_name / "row_scores.jsonl", diag)
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
        "metrics": dict(eval_payload.get("metrics", {})),
        "external_metrics_path": str(paths.external_lvvis_metrics_path),
    }


def _parse_modes(text: str) -> List[str]:
    allowed = {"oracle_class_score1", "oracle_class_iou_score"}
    modes: List[str] = []
    for part in str(text).replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"unknown mode={item!r}; allowed={sorted(allowed)}")
        if item not in modes:
            modes.append(item)
    return modes or ["oracle_class_score1", "oracle_class_iou_score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VideoCutLER trajectory oracle-class AP ceiling eval.")
    p.add_argument("--dataset_name", default="lvvis_val")
    p.add_argument("--output_root", required=True)
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--trajectory_path", default="")
    p.add_argument("--gt_sidecar_path", default="")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--modes", default="oracle_class_score1,oracle_class_iou_score")
    p.add_argument("--min_iou", type=float, default=0.0)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = str(args.dataset_name)
    output_root = Path(args.output_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    trajectory_path = Path(args.trajectory_path).expanduser().resolve() if args.trajectory_path else asset_root / "exports" / dataset_name / "trajectory_records.jsonl"
    gt_sidecar_path = _resolve_sidecar(asset_root, dataset_name, str(args.gt_sidecar_path or ""))
    annotation_json = Path(args.annotation_json).expanduser().resolve() if args.annotation_json else _default_annotation_json(REPO_ROOT, dataset_name)

    for path, name in (
        (trajectory_path, "trajectory_path"),
        (gt_sidecar_path, "gt_sidecar_path"),
        (annotation_json, "annotation_json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    modes = _parse_modes(args.modes)
    if args.smoke and int(args.max_rows or 0) <= 0:
        args.max_rows = 128

    rows, source_meta = _collect_rows(
        trajectory_path=trajectory_path,
        sidecar_path=gt_sidecar_path,
        annotation_json=annotation_json,
        min_iou=float(args.min_iou),
        max_rows=int(args.max_rows or 0),
    )
    if not rows:
        raise RuntimeError(f"no usable VideoCutLER oracle-class rows; source_meta={source_meta}")

    mode_summaries: List[Record] = []
    for mode in modes:
        mode_summaries.append(_run_mode_eval(
            mode=mode,
            output_root=output_root,
            dataset_name=dataset_name,
            seed=int(args.seed),
            smoke=bool(args.smoke),
            rows=rows,
        ))

    compact_rows: List[Record] = []
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
    _write_csv(output_root / "videocutler_oracle_class_ceiling_metrics.csv", compact_rows)

    summary = {
        "status": "PASS",
        "dataset_name": dataset_name,
        "output_root": str(output_root),
        "asset_root": str(asset_root),
        "trajectory_path": str(trajectory_path),
        "gt_sidecar_path": str(gt_sidecar_path),
        "annotation_json": str(annotation_json),
        "min_iou": float(args.min_iou),
        "source_meta": source_meta,
        "modes": mode_summaries,
        "compact_metrics_csv": str(output_root / "videocutler_oracle_class_ceiling_metrics.csv"),
    }
    _write_json(output_root / "videocutler_oracle_class_ceiling_summary.json", summary)

    lines = [
        "# A8 VideoCutLER Oracle-Class Ceiling Eval",
        "",
        f"- dataset_name: `{dataset_name}`",
        f"- trajectory_path: `{trajectory_path}`",
        f"- gt_sidecar_path: `{gt_sidecar_path}`",
        f"- retained_rows: `{len(rows)}`",
        f"- trajectory_count: `{source_meta.get('trajectory_count')}`",
        f"- sidecar_count: `{source_meta.get('sidecar_count')}`",
        f"- match_iou_mean: `{source_meta.get('match_iou_mean')}`",
        "",
        "| mode | AP | AP50 | AP75 | mAPb | mAPn | predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in mode_summaries:
        m = item.get("metrics", {})
        lines.append(
            f"| {item.get('mode')} | {m.get('AP')} | {m.get('AP50')} | {m.get('AP75')} | {m.get('mAPb')} | {m.get('mAPn')} | {item.get('prediction_count')} |"
        )
    _write_json(output_root / "source_meta.json", source_meta)
    (output_root / "VIDEOCUTLER_ORACLE_CLASS_CEILING_EVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
