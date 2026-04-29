#!/usr/bin/env python3
"""Full row-level failure dump for oracle clean-data D arm.

Read-only analysis. It scores GT-carrier rows against per-clip full-Y candidates
using the same text-side projector/checkpoint convention as the current V2-B
prealign scorer, then writes full row-level rank/margin/top-k diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab
from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig

Record = Dict[str, Any]


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
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
        if value is None:
            return default
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _unique_ints(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = values.replace(";", ",").split(",")
    elif isinstance(values, Mapping):
        parts = list(values.keys())
    elif isinstance(values, Iterable):
        parts = list(values)
    else:
        parts = [values]
    out: List[int] = []
    seen: set[int] = set()
    for item in parts:
        val = _safe_int(item)
        if val is None or val in seen:
            continue
        seen.add(int(val))
        out.append(int(val))
    return out


def _iter_jsonl(path: Path) -> Iterator[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _load_json(path: Path) -> Record:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _mean(vals: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else None


def _median(vals: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(median(clean)) if clean else None


def _normalize_np(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _load_projector(checkpoint_path: Path, *, device: torch.device) -> Tuple[Projector, float, Record]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = dict(checkpoint.get("text_projector_config", {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(cfg.get("input_dim", 512)),
            hidden_dim=int(cfg.get("hidden_dim", 1024)),
            output_dim=int(cfg.get("output_dim", 768)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_layernorm=bool(cfg.get("use_layernorm", True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint["text_projector_state_dict"])
    projector.eval()
    theta_raw = checkpoint.get("theta_T", 0.07)
    try:
        theta_tensor = torch.tensor(float(theta_raw), dtype=torch.float32)
        temp = float(F.softplus(theta_tensor).item())
    except Exception:
        temp = 0.07
    if not math.isfinite(temp) or temp <= 0.0:
        temp = 0.07
    return projector, temp, dict(checkpoint)


def _project_text_matrix(
    projector: Projector,
    matrix: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    projected: List[np.ndarray] = []
    matrix = np.asarray(matrix, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, int(matrix.shape[0]), int(batch_size)):
            batch = torch.from_numpy(matrix[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            input_dim = int(getattr(projector.config, "input_dim", batch.shape[-1]))
            output_dim = int(getattr(projector.config, "output_dim", batch.shape[-1]))
            if int(batch.shape[-1]) == input_dim:
                out = projector(batch)
            elif int(batch.shape[-1]) == output_dim:
                out = F.normalize(batch, p=2.0, dim=-1)
            else:
                raise ValueError(
                    f"text width {batch.shape[-1]} does not match projector input/output dims "
                    f"({input_dim}, {output_dim})"
                )
            projected.append(F.normalize(out, p=2.0, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(projected, axis=0).astype(np.float32)


def _load_full_y_by_clip(path: Path) -> Dict[int, List[int]]:
    payload = _load_json(path)
    out: Dict[int, List[int]] = {}
    for row in payload.get("records", []):
        if not isinstance(row, Mapping):
            continue
        clip_id = _safe_int(row.get("clip_id"), _safe_int(row.get("video_id"), None))
        if clip_id is None:
            continue
        labels = sorted(set(_unique_ints(row.get("full_y_raw_ids"))))
        if labels:
            out[int(clip_id)] = labels
    return out


def _load_identity_by_tid(path: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for idx, row in enumerate(_iter_jsonl(path)):
        tid = str(row.get("trajectory_id", row.get("carrier_id", ""))).strip()
        if not tid:
            tid = str(row.get("carrier_row_index", idx))
        rec = dict(row)
        rec.setdefault("carrier_row_index", idx)
        out[tid] = rec
    return out


def _load_carriers_by_tid(path: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for idx, row in enumerate(_iter_jsonl(path)):
        tid = str(row.get("trajectory_id", row.get("carrier_id", ""))).strip()
        if not tid:
            tid = str(idx)
        rec = dict(row)
        rec.setdefault("carrier_row_index", idx)
        out[tid] = rec
    return out


def _topk(scores: np.ndarray, k: int) -> List[int]:
    n = int(scores.shape[0])
    if n <= 0:
        return []
    k = min(int(k), n)
    if k == n:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
    return [int(i) for i in order.tolist()]


def _rank_of_index(scores: np.ndarray, idx: int) -> int:
    gt_score = float(scores[int(idx)])
    # Stable rank with ties not penalized beyond strict greater scores.
    return int(np.sum(scores > gt_score) + 1)


def _bin_candidate_count(n: int) -> str:
    if n <= 3:
        return "01_<=3"
    if n <= 5:
        return "02_4-5"
    if n <= 10:
        return "03_6-10"
    if n <= 20:
        return "04_11-20"
    if n <= 50:
        return "05_21-50"
    return "06_>50"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full row-level D GT-carrier latent assignment dump.")
    parser.add_argument("--run_root_v2b", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--full_y_path", required=True)
    parser.add_argument("--gt_carrier_path", required=True)
    parser.add_argument("--gt_identity_path", required=True)
    parser.add_argument("--dataset_name", default="lvvis_train_base")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size_text", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--large_wrong_threshold", type=float, default=-0.20)
    parser.add_argument("--near_tie_threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root_v2b)
    output_root = Path(args.output_root)
    out_dir = output_root / "d_failure_decomposition_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    projector, temperature, ckpt = _load_projector(checkpoint_path, device=device)
    text_raw_ids, _text_records, text_matrix = load_text_vocab(run_root)
    projected_text = _project_text_matrix(projector, text_matrix, device=device, batch_size=int(args.batch_size_text))
    raw_to_text_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_raw_ids)}

    full_y_by_clip = _load_full_y_by_clip(Path(args.full_y_path))
    carrier_by_tid = _load_carriers_by_tid(Path(args.gt_carrier_path))
    identity_by_tid = _load_identity_by_tid(Path(args.gt_identity_path))
    carrier_parent = Path(args.gt_carrier_path).parent

    row_scores_path = out_dir / "D_full_row_scores.jsonl"
    examples_path = out_dir / "D_full_failure_examples.jsonl"

    total_rows = 0
    with_label_record = 0
    in_scope = 0
    latent_evaluable = 0
    carrier_load_fail = 0
    candidate_missing_vocab = 0
    malformed = 0

    gt_ranks: List[int] = []
    normalized_ranks: List[float] = []
    margins: List[float] = []
    candidate_sizes: List[int] = []
    entropies: List[float] = []

    rank_bucket = Counter()
    margin_bucket = Counter()
    top1_counts = Counter()
    wrong_top1_counts = Counter()
    confusion_pairs = Counter()
    by_class: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "row_count": 0,
        "rank1_count": 0,
        "top5_count": 0,
        "top20_count": 0,
        "rank_sum": 0.0,
        "norm_rank_sum": 0.0,
        "margin_sum": 0.0,
        "margin_n": 0,
        "wrong_top1_counts": Counter(),
    })
    candidate_bins: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "row_count": 0,
        "rank1_count": 0,
        "wrong_count": 0,
        "margin_sum": 0.0,
        "margin_n": 0,
    })
    failure_examples_written = 0

    sorted_tids = sorted(carrier_by_tid.keys())
    max_rows = int(args.max_rows or 0)
    if max_rows > 0:
        sorted_tids = sorted_tids[:max_rows]

    with row_scores_path.open("w", encoding="utf-8") as row_handle, examples_path.open("w", encoding="utf-8") as ex_handle:
        for row_index, tid in enumerate(sorted_tids):
            total_rows += 1
            carrier = carrier_by_tid.get(tid, {})
            identity = identity_by_tid.get(tid, {})
            clip_id = _safe_int(identity.get("clip_id"), _safe_int(carrier.get("clip_id"), _safe_int(carrier.get("video_id"), None)))
            gt_raw = _safe_int(
                identity.get("raw_category_id"),
                _safe_int(identity.get("matched_gt_raw_id"), _safe_int(carrier.get("raw_category_id"), None)),
            )
            if clip_id is None or gt_raw is None:
                malformed += 1
                continue
            labels = full_y_by_clip.get(int(clip_id))
            if labels is None:
                continue
            with_label_record += 1
            if int(gt_raw) not in set(labels):
                continue
            in_scope += 1
            candidate_raw_ids = [int(x) for x in labels if int(x) in raw_to_text_idx]
            if int(gt_raw) not in candidate_raw_ids:
                candidate_missing_vocab += 1
                continue
            candidate_indices = np.asarray([raw_to_text_idx[int(x)] for x in candidate_raw_ids], dtype=np.int64)
            gt_candidate_idx = candidate_raw_ids.index(int(gt_raw))
            z_path = str(carrier.get("z_norm_path", ""))
            if not z_path:
                carrier_load_fail += 1
                continue
            try:
                carrier_vec = _normalize_np(read_vector_from_locator(carrier_parent, z_path))
            except Exception:
                carrier_load_fail += 1
                continue
            candidate_matrix = projected_text[candidate_indices]
            scores = np.matmul(candidate_matrix, carrier_vec.astype(np.float32)) / float(temperature)
            scores = np.asarray(scores, dtype=np.float32)
            if scores.size <= 0:
                continue
            top_order = _topk(scores, int(args.topk))
            top1_idx = int(top_order[0])
            top1_raw = int(candidate_raw_ids[top1_idx])
            gt_rank = _rank_of_index(scores, int(gt_candidate_idx))
            gt_score = float(scores[int(gt_candidate_idx)])
            if len(candidate_raw_ids) > 1:
                non_gt_scores = np.delete(scores, int(gt_candidate_idx))
                best_non_gt_score = float(np.max(non_gt_scores))
                margin = float(gt_score - best_non_gt_score)
            else:
                best_non_gt_score = None
                margin = None
            rank1 = int(gt_rank == 1)
            top5 = int(gt_rank <= 5)
            top20 = int(gt_rank <= 20)
            norm_rank = float((gt_rank - 1) / max(len(candidate_raw_ids) - 1, 1))
            probs = torch.softmax(torch.from_numpy(scores), dim=0).numpy()
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))

            latent_evaluable += 1
            gt_ranks.append(int(gt_rank))
            normalized_ranks.append(float(norm_rank))
            candidate_sizes.append(int(len(candidate_raw_ids)))
            entropies.append(float(entropy))
            top1_counts[int(top1_raw)] += 1
            if margin is not None:
                margins.append(float(margin))
            if gt_rank == 1:
                rank_bucket["rank1_correct"] += 1
            elif gt_rank <= 5:
                rank_bucket["rank2_5_wrong"] += 1
            elif gt_rank <= 20:
                rank_bucket["rank6_20_wrong"] += 1
            else:
                rank_bucket["rank_gt20_wrong"] += 1

            if rank1:
                if margin is not None:
                    margin_bucket["correct_positive_margin"] += 1
            else:
                wrong_top1_counts[int(top1_raw)] += 1
                confusion_pairs[(int(gt_raw), int(top1_raw))] += 1
                if margin is None:
                    margin_bucket["wrong_no_non_gt_candidate"] += 1
                elif -float(args.near_tie_threshold) <= float(margin) < 0.0:
                    margin_bucket[f"wrong_near_tie_margin_-{args.near_tie_threshold}_0"] += 1
                elif float(margin) < float(args.large_wrong_threshold):
                    margin_bucket[f"wrong_large_negative_margin_lt_{args.large_wrong_threshold}"] += 1
                else:
                    margin_bucket["wrong_other_margin"] += 1

            b = _bin_candidate_count(len(candidate_raw_ids))
            candidate_bins[b]["row_count"] += 1
            candidate_bins[b]["rank1_count"] += rank1
            candidate_bins[b]["wrong_count"] += int(not rank1)
            if margin is not None:
                candidate_bins[b]["margin_sum"] += float(margin)
                candidate_bins[b]["margin_n"] += 1

            cls = by_class[int(gt_raw)]
            cls["row_count"] += 1
            cls["rank1_count"] += rank1
            cls["top5_count"] += top5
            cls["top20_count"] += top20
            cls["rank_sum"] += float(gt_rank)
            cls["norm_rank_sum"] += float(norm_rank)
            if margin is not None:
                cls["margin_sum"] += float(margin)
                cls["margin_n"] += 1
            if not rank1:
                cls["wrong_top1_counts"][int(top1_raw)] += 1

            top_raw_ids = [int(candidate_raw_ids[i]) for i in top_order]
            top_scores = [float(scores[i]) for i in top_order]
            record = {
                "row_index": int(row_index),
                "trajectory_id": str(tid),
                "clip_id": int(clip_id),
                "gt_raw_id": int(gt_raw),
                "candidate_label_count": int(len(candidate_raw_ids)),
                "top1_raw_id": int(top1_raw),
                "gt_rank": int(gt_rank),
                "score_gt": float(gt_score),
                "score_top1": float(scores[top1_idx]),
                "score_best_non_gt": best_non_gt_score,
                "margin_gt_vs_best_non_gt": margin,
                "normalized_gt_rank": float(norm_rank),
                "is_gt_top1": bool(rank1),
                "is_gt_top5": bool(top5),
                "is_gt_top20": bool(top20),
                "assignment_entropy": float(entropy),
                "topk_raw_ids": top_raw_ids,
                "topk_scores": top_scores,
            }
            row_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not rank1 and failure_examples_written < 512:
                ex_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                failure_examples_written += 1

    rank1_count = rank_bucket.get("rank1_correct", 0)
    top5_count = rank_bucket.get("rank1_correct", 0) + rank_bucket.get("rank2_5_wrong", 0)
    top20_count = top5_count + rank_bucket.get("rank6_20_wrong", 0)
    wrong_count = max(latent_evaluable - rank1_count, 0)

    rank_rows = []
    for key in ["rank1_correct", "rank2_5_wrong", "rank6_20_wrong", "rank_gt20_wrong"]:
        cnt = int(rank_bucket.get(key, 0))
        rank_rows.append({"bucket": key, "count": cnt, "rate": float(cnt / latent_evaluable) if latent_evaluable else None})
    _write_csv(out_dir / "D_full_rank_buckets.csv", rank_rows, ["bucket", "count", "rate"])

    margin_rows = []
    margin_total = sum(int(v) for v in margin_bucket.values())
    for key, cnt in margin_bucket.most_common():
        margin_rows.append({"bucket": key, "count": int(cnt), "rate": float(cnt / margin_total) if margin_total else None})
    _write_csv(out_dir / "D_full_margin_buckets.csv", margin_rows, ["bucket", "count", "rate"])

    top1_rows = []
    for raw_id, cnt in top1_counts.most_common(100):
        top1_rows.append({
            "raw_category_id": int(raw_id),
            "top1_count": int(cnt),
            "top1_rate_over_evaluable": float(cnt / latent_evaluable) if latent_evaluable else None,
        })
    _write_csv(out_dir / "D_full_top1_label_concentration.csv", top1_rows, ["raw_category_id", "top1_count", "top1_rate_over_evaluable"])

    wrong_top1_rows = []
    for raw_id, cnt in wrong_top1_counts.most_common(100):
        wrong_top1_rows.append({
            "raw_category_id": int(raw_id),
            "wrong_top1_count": int(cnt),
            "wrong_top1_rate_over_evaluable": float(cnt / latent_evaluable) if latent_evaluable else None,
            "wrong_top1_rate_over_wrong": float(cnt / wrong_count) if wrong_count else None,
        })
    _write_csv(out_dir / "D_full_wrong_top1_label_concentration.csv", wrong_top1_rows, ["raw_category_id", "wrong_top1_count", "wrong_top1_rate_over_evaluable", "wrong_top1_rate_over_wrong"])

    conf_rows = []
    for (gt_raw, top1_raw), cnt in confusion_pairs.most_common(500):
        conf_rows.append({
            "gt_raw_id": int(gt_raw),
            "wrong_top1_raw_id": int(top1_raw),
            "count": int(cnt),
            "rate_over_wrong": float(cnt / wrong_count) if wrong_count else None,
        })
    _write_csv(out_dir / "D_full_confusion_pairs.csv", conf_rows, ["gt_raw_id", "wrong_top1_raw_id", "count", "rate_over_wrong"])

    cand_rows = []
    for key in sorted(candidate_bins.keys()):
        obj = candidate_bins[key]
        n = int(obj["row_count"])
        cand_rows.append({
            "candidate_size_bin": key,
            "row_count": n,
            "rank1_rate": float(obj["rank1_count"] / n) if n else None,
            "wrong_rate": float(obj["wrong_count"] / n) if n else None,
            "mean_margin": float(obj["margin_sum"] / obj["margin_n"]) if obj["margin_n"] else None,
        })
    _write_csv(out_dir / "D_full_candidate_size_effect.csv", cand_rows, ["candidate_size_bin", "row_count", "rank1_rate", "wrong_rate", "mean_margin"])

    class_rows = []
    for gt_raw, obj in by_class.items():
        n = int(obj["row_count"])
        wrong_top = obj["wrong_top1_counts"].most_common(1)
        class_rows.append({
            "raw_category_id": int(gt_raw),
            "row_count": n,
            "gt_rank1_rate": float(obj["rank1_count"] / n) if n else None,
            "gt_top5_rate": float(obj["top5_count"] / n) if n else None,
            "gt_top20_rate": float(obj["top20_count"] / n) if n else None,
            "mean_gt_rank": float(obj["rank_sum"] / n) if n else None,
            "mean_normalized_gt_rank": float(obj["norm_rank_sum"] / n) if n else None,
            "mean_margin": float(obj["margin_sum"] / obj["margin_n"]) if obj["margin_n"] else None,
            "most_common_wrong_top1_raw_id": int(wrong_top[0][0]) if wrong_top else None,
            "most_common_wrong_top1_count": int(wrong_top[0][1]) if wrong_top else 0,
        })
    class_rows_sorted = sorted(class_rows, key=lambda r: (float(r["gt_rank1_rate"]), -int(r["row_count"])))
    _write_csv(
        out_dir / "D_full_by_gt_class.csv",
        class_rows_sorted,
        [
            "raw_category_id",
            "row_count",
            "gt_rank1_rate",
            "gt_top5_rate",
            "gt_top20_rate",
            "mean_gt_rank",
            "mean_normalized_gt_rank",
            "mean_margin",
            "most_common_wrong_top1_raw_id",
            "most_common_wrong_top1_count",
        ],
    )

    wrong_large_key = f"wrong_large_negative_margin_lt_{args.large_wrong_threshold}"
    near_tie_key = f"wrong_near_tie_margin_-{args.near_tie_threshold}_0"
    summary = {
        "status": "PASS",
        "audit_name": "D_full_row_level_failure_dump",
        "scoring_backend": "current_v2b_prealign_projector_text_cosine",
        "run_root_v2b": str(run_root),
        "checkpoint_path": str(checkpoint_path),
        "score_temperature": float(temperature),
        "full_y_path": str(Path(args.full_y_path)),
        "gt_carrier_path": str(Path(args.gt_carrier_path)),
        "gt_identity_path": str(Path(args.gt_identity_path)),
        "row_count_total": int(total_rows),
        "row_count_with_label_record": int(with_label_record),
        "row_count_in_scope": int(in_scope),
        "latent_evaluable_row_count": int(latent_evaluable),
        "carrier_load_fail_count": int(carrier_load_fail),
        "candidate_missing_vocab_count": int(candidate_missing_vocab),
        "malformed_count": int(malformed),
        "candidate_contains_gt_rate": 1.0 if latent_evaluable else None,
        "gt_rank1_rate": float(rank1_count / latent_evaluable) if latent_evaluable else None,
        "gt_top5_rate": float(top5_count / latent_evaluable) if latent_evaluable else None,
        "gt_top20_rate": float(top20_count / latent_evaluable) if latent_evaluable else None,
        "top1_wrong_rate": float(wrong_count / latent_evaluable) if latent_evaluable else None,
        "mean_gt_rank": _mean([float(x) for x in gt_ranks]),
        "median_gt_rank": _median([float(x) for x in gt_ranks]),
        "mean_normalized_gt_rank": _mean(normalized_ranks),
        "mean_gt_margin_vs_best_non_gt": _mean(margins),
        "positive_gt_margin_rate": float(sum(1 for x in margins if x > 0.0) / len(margins)) if margins else None,
        "assignment_entropy_mean": _mean(entropies),
        "top1_top_label": top1_rows[0] if top1_rows else None,
        "wrong_top1_top_label": wrong_top1_rows[0] if wrong_top1_rows else None,
        "wrong_large_negative_margin_rate": float(margin_bucket.get(wrong_large_key, 0) / wrong_count) if wrong_count else None,
        "wrong_near_tie_margin_rate": float(margin_bucket.get(near_tie_key, 0) / wrong_count) if wrong_count else None,
        "rank_buckets": rank_rows,
        "margin_buckets": margin_rows,
        "outputs": {
            "row_scores_jsonl_remote": str(row_scores_path),
            "failure_examples_jsonl": str(examples_path),
            "summary": str(out_dir / "D_full_failure_summary.json"),
            "takeover": str(out_dir / "D_FULL_FAILURE_TAKEOVER.md"),
            "rank_buckets": str(out_dir / "D_full_rank_buckets.csv"),
            "margin_buckets": str(out_dir / "D_full_margin_buckets.csv"),
            "top1_concentration": str(out_dir / "D_full_top1_label_concentration.csv"),
            "wrong_top1_concentration": str(out_dir / "D_full_wrong_top1_label_concentration.csv"),
            "confusion_pairs": str(out_dir / "D_full_confusion_pairs.csv"),
            "candidate_size_effect": str(out_dir / "D_full_candidate_size_effect.csv"),
            "by_gt_class": str(out_dir / "D_full_by_gt_class.csv"),
        },
        "interpretation_hints": {
            "rank_structure": "High top5/top20 with low rank1 means coarse recall but weak top1 discrimination.",
            "margin_structure": "High large-negative wrong margin means wrong classes dominate GT, not just near ties.",
            "hub_structure": "Wrong top1 concentration is data-driven audit evidence only; do not hard-code any class as training prior.",
        },
    }
    _write_json(out_dir / "D_full_failure_summary.json", summary)

    lines: List[str] = []
    lines.append("# D Full Row-Level Failure Dump")
    lines.append("")
    lines.append("Status: PASS")
    lines.append("")
    lines.append("Scope:")
    lines.append("- Read-only D full-Y + GT-carrier latent scorer row dump.")
    lines.append("- No training, no inference, no checkpoint writes.")
    lines.append("- Large row-score JSONL is intentionally remote-only.")
    lines.append("")
    lines.append("## Key metrics")
    for key in [
        "row_count_total",
        "row_count_with_label_record",
        "row_count_in_scope",
        "latent_evaluable_row_count",
        "candidate_contains_gt_rate",
        "gt_rank1_rate",
        "gt_top5_rate",
        "gt_top20_rate",
        "top1_wrong_rate",
        "mean_gt_rank",
        "median_gt_rank",
        "mean_normalized_gt_rank",
        "mean_gt_margin_vs_best_non_gt",
        "positive_gt_margin_rate",
        "wrong_large_negative_margin_rate",
        "wrong_near_tie_margin_rate",
        "assignment_entropy_mean",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.append("")
    lines.append("## Rank buckets")
    for row in rank_rows:
        lines.append(f"- {row['bucket']}: count=`{row['count']}`, rate=`{row['rate']}`")
    lines.append("")
    lines.append("## Margin buckets")
    for row in margin_rows:
        lines.append(f"- {row['bucket']}: count=`{row['count']}`, rate=`{row['rate']}`")
    lines.append("")
    lines.append("## Top1 concentration")
    if top1_rows:
        lines.append(f"- top1 top label: `{top1_rows[0]}`")
    if wrong_top1_rows:
        lines.append(f"- wrong-top1 top label: `{wrong_top1_rows[0]}`")
    lines.append("")
    lines.append("## Outputs")
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    (out_dir / "D_FULL_FAILURE_TAKEOVER.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "output_dir": str(out_dir),
        "latent_evaluable_row_count": latent_evaluable,
        "gt_rank1_rate": summary["gt_rank1_rate"],
        "gt_top5_rate": summary["gt_top5_rate"],
        "mean_margin": summary["mean_gt_margin_vs_best_non_gt"],
        "wrong_large_negative_margin_rate": summary["wrong_large_negative_margin_rate"],
        "wrong_near_tie_margin_rate": summary["wrong_near_tie_margin_rate"],
        "top1_top_label": summary["top1_top_label"],
        "wrong_top1_top_label": summary["wrong_top1_top_label"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
