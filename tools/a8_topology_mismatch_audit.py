#!/usr/bin/env python3
"""A8 Phase-1 topology mismatch audit.

Read-only diagnostic for testing whether train-good/val-fail classes have a
larger CLIP-text <-> DINO-visual topology mismatch than learned-stable classes.

The script intentionally does not train, mutate checkpoints, or touch the
control-plane. It writes CSV/JSON/TAKEOVER artifacts only.
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


def _repo_root_from_arg(repo_root: Optional[str]) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    return Path.cwd().resolve()


def _ensure_repo_on_path(repo_root: Path) -> None:
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
        for r in rows:
            w.writerow(r)


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


def _mean(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _median(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x]
    return float(np.median(xs)) if xs else float("nan")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _rank_of(target: str, ordered: Sequence[str]) -> int:
    for i, x in enumerate(ordered, start=1):
        if str(x) == str(target):
            return i
    return 10**9


def _topk_neighbors(sim: np.ndarray, ids: Sequence[str], idx: int, k: int) -> List[str]:
    row = np.asarray(sim[idx], dtype=np.float64).copy()
    row[idx] = -np.inf
    order = np.argsort(-row, kind="mergesort")[:k]
    return [str(ids[int(i)]) for i in order]


def _class_name(raw_id: str, class_name_map: Mapping[Any, Any]) -> str:
    for key in (raw_id, int(raw_id) if str(raw_id).isdigit() else raw_id):
        if key in class_name_map:
            return str(class_name_map[key])
    return f"raw_id_{raw_id}"


def _default_path(run_root: Path, rel: str) -> Path:
    return run_root / rel


def _resolve_inputs(args: argparse.Namespace, run_root: Path) -> Dict[str, Path]:
    return {
        "per_class_join": Path(args.per_class_join) if args.per_class_join else _default_path(run_root, "analysis/a8_dj3_semantic_boundary_bottleneck_audit/per_class_train_val_525_join.csv"),
        "val_visible_per_row": Path(args.val_visible_per_row) if args.val_visible_per_row else _default_path(run_root, "analysis/a8_visible525_candidate_rankk_audit/lvvis_val/D-J3_train_time_dynamic_ep10_val_target525_candidate525/visible525_candidate_rankk_per_row.csv"),
        "checkpoint": Path(args.checkpoint_path) if args.checkpoint_path else _default_path(run_root, "outputs/a8_joint_train_time_dynamic_hungarian/lvvis_train_base/D-J3_pre1_dyn1_ep10/train/joint_train_time_dynamic_hungarian/a8_joint_train_time_dynamic_last.pth"),
    }


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


def _build_visual_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Dict[str, np.ndarray]:
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
    return out


def _matrix_for_ids(proto: Mapping[str, np.ndarray], ids: Sequence[str]) -> np.ndarray:
    vecs = []
    dim = None
    for v in proto.values():
        dim = int(np.asarray(v).shape[-1])
        break
    if dim is None:
        raise RuntimeError("empty prototype dictionary")
    for rid in ids:
        if rid in proto:
            vecs.append(np.asarray(proto[rid], dtype=np.float32))
        else:
            vecs.append(np.full((dim,), np.nan, dtype=np.float32))
    return np.stack(vecs, axis=0)


def _valid_cosine_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    valid = np.isfinite(mat).all(axis=1)
    normed = np.zeros_like(mat, dtype=np.float32)
    if valid.any():
        normed[valid] = _l2_normalize(mat[valid])
    sim = normed @ normed.T
    for i, ok in enumerate(valid):
        if not ok:
            sim[i, :] = -np.inf
            sim[:, i] = -np.inf
    return sim


def _topk_score_neighbors(score_mat: np.ndarray, ids: Sequence[str], idx: int, k: int) -> List[str]:
    row = np.asarray(score_mat[idx], dtype=np.float64).copy()
    if idx < len(row):
        row[idx] = -np.inf
    order = np.argsort(-row, kind="mergesort")[:k]
    return [str(ids[int(i)]) for i in order]


def _summarize_group(rows: Sequence[Mapping[str, Any]], group_key: str = "group") -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_key, "NA"))].append(r)
    out = []
    metric_keys = [
        "text_trainvis_jaccard@10",
        "text_valvis_jaccard@10",
        "train_proto_score_neighbor_jaccard@10",
        "val_proto_score_neighbor_jaccard@10",
        "train_val_visual_proto_drift",
        "suppressor_text_rank",
        "suppressor_train_visual_rank",
        "suppressor_val_visual_rank",
        "suppressor_train_proto_score_rank",
        "suppressor_val_proto_score_rank",
    ]
    for g, items in sorted(groups.items()):
        rec: Dict[str, Any] = {"group": g, "class_count": len(items)}
        for k in metric_keys:
            vals = [_fnum(x.get(k)) for x in items if str(x.get(k, "")) not in ("", "NA", "None")]
            rec[f"mean_{k}"] = _mean(vals)
            rec[f"median_{k}"] = _median(vals)
        rec["has_train_visual_proto_count"] = sum(str(x.get("has_train_visual_proto")) == "True" for x in items)
        rec["has_val_visual_proto_count"] = sum(str(x.get("has_val_visual_proto")) == "True" for x in items)
        rec["person_suppressed_class_count"] = sum(str(x.get("is_person_suppressed")) == "True" for x in items)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A8 read-only topology mismatch audit")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--per_class_join", default="")
    ap.add_argument("--val_visible_per_row", default="")
    ap.add_argument("--train_dataset_name", default="lvvis_train_base")
    ap.add_argument("--val_dataset_name", default="lvvis_val")
    ap.add_argument("--train_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/train_instances.json")
    ap.add_argument("--val_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/val_instances.json")
    ap.add_argument("--neighbor_k", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--person_raw_id", default="773")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = _repo_root_from_arg(args.repo_root)
    asset_root = Path(args.asset_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo_on_path(repo_root)
    gtceil = _load_gtceil_module(repo_root)

    inputs = _resolve_inputs(args, run_root)
    for name, path in inputs.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing input {name}: {path}")
    per_class = _read_csv(inputs["per_class_join"])
    val_per_rows = _read_csv(inputs["val_visible_per_row"])

    # Import after sys.path setup.
    from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # type: ignore
        load_projector_bundle,
        load_text_vocab_with_names,
        score_infer_rows_matrix,
    )

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_projector_bundle(inputs["checkpoint"], device=device)
    text_vocab_ids_raw, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_root, args.train_dataset_name)
    text_ids = [_norm_id(x) for x in text_vocab_ids_raw]
    text_id_to_idx = {rid: i for i, rid in enumerate(text_ids)}
    text_matrix_np = _l2_normalize(np.asarray(text_matrix, dtype=np.float32))

    # Candidate classes are those in the per-class 525 join.
    class_ids = [_norm_id(r.get("raw_id")) for r in per_class]
    class_ids = [rid for rid in class_ids if rid in text_id_to_idx]
    if not class_ids:
        raise RuntimeError("no class ids found in per_class_join that also exist in text vocab")
    class_idx = np.asarray([text_id_to_idx[rid] for rid in class_ids], dtype=np.int64)
    id_to_pos = {rid: i for i, rid in enumerate(class_ids)}
    text_sub = text_matrix_np[class_idx]
    text_sim = text_sub @ text_sub.T

    # Load train/val visual prototype carriers.
    train_rows, train_carrier, train_meta, train_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.train_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.train_annotation_json),
        max_rows=args.max_rows,
    )
    val_rows, val_carrier, val_meta, val_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.val_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.val_annotation_json),
        max_rows=args.max_rows,
    )
    train_proto = _build_visual_prototypes(train_rows, train_carrier)
    val_proto = _build_visual_prototypes(val_rows, val_carrier)
    train_proto_mat = _matrix_for_ids(train_proto, class_ids)
    val_proto_mat = _matrix_for_ids(val_proto, class_ids)
    train_vis_sim = _valid_cosine_matrix(train_proto_mat)
    val_vis_sim = _valid_cosine_matrix(val_proto_mat)

    # Score visual prototypes against projected-text scorer. This is more robust
    # than assuming a projector API, and directly reflects the current checkpoint's
    # text-to-visual scoring neighborhoods.
    train_valid_mask = np.isfinite(train_proto_mat).all(axis=1)
    val_valid_mask = np.isfinite(val_proto_mat).all(axis=1)
    train_score_sub = np.full((len(class_ids), len(class_ids)), -np.inf, dtype=np.float32)
    val_score_sub = np.full((len(class_ids), len(class_ids)), -np.inf, dtype=np.float32)
    if train_valid_mask.any():
        train_scores = score_infer_rows_matrix(
            carrier_matrix=train_proto_mat[train_valid_mask],
            bundle=bundle,
            text_matrix=text_matrix,
            show_progress=False,
        )
        train_logits = np.asarray(train_scores["fused_logits"], dtype=np.float32)[:, class_idx]
        train_score_sub[np.where(train_valid_mask)[0], :] = train_logits
    if val_valid_mask.any():
        val_scores = score_infer_rows_matrix(
            carrier_matrix=val_proto_mat[val_valid_mask],
            bundle=bundle,
            text_matrix=text_matrix,
            show_progress=False,
        )
        val_logits = np.asarray(val_scores["fused_logits"], dtype=np.float32)[:, class_idx]
        val_score_sub[np.where(val_valid_mask)[0], :] = val_logits

    # Val suppressor by GT class.
    suppressor_by_gt: Dict[str, Counter[str]] = defaultdict(Counter)
    suppressor_margin: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in val_per_rows:
        gt = _norm_id(row.get("gt_raw_id"))
        top1 = _norm_id(row.get("top1_raw_id"))
        rank = _fnum(row.get("restricted_rank"))
        if rank > 1 and gt and top1:
            suppressor_by_gt[gt][top1] += 1
            m = _fnum(row.get("margin_gt_minus_top1"))
            if m == m:
                suppressor_margin[(gt, top1)].append(m)

    k = int(args.neighbor_k)
    metric_rows: List[Dict[str, Any]] = []
    for cls in per_class:
        rid = _norm_id(cls.get("raw_id"))
        if rid not in id_to_pos:
            continue
        i = id_to_pos[rid]
        text_n = _topk_neighbors(text_sim, class_ids, i, k)
        train_vis_n = _topk_neighbors(train_vis_sim, class_ids, i, k) if np.isfinite(train_vis_sim[i]).any() else []
        val_vis_n = _topk_neighbors(val_vis_sim, class_ids, i, k) if np.isfinite(val_vis_sim[i]).any() else []
        train_score_n = _topk_score_neighbors(train_score_sub, class_ids, i, k) if np.isfinite(train_score_sub[i]).any() else []
        val_score_n = _topk_score_neighbors(val_score_sub, class_ids, i, k) if np.isfinite(val_score_sub[i]).any() else []

        top_supp = ""
        top_supp_count = 0
        if rid in suppressor_by_gt and suppressor_by_gt[rid]:
            top_supp, top_supp_count = suppressor_by_gt[rid].most_common(1)[0]

        drift = float("nan")
        if rid in train_proto and rid in val_proto:
            drift = float(1.0 - np.dot(train_proto[rid], val_proto[rid]))

        suppressor_text_rank = _rank_of(top_supp, text_n) if top_supp else ""
        suppressor_train_visual_rank = _rank_of(top_supp, train_vis_n) if top_supp else ""
        suppressor_val_visual_rank = _rank_of(top_supp, val_vis_n) if top_supp else ""
        suppressor_train_score_rank = _rank_of(top_supp, train_score_n) if top_supp else ""
        suppressor_val_score_rank = _rank_of(top_supp, val_score_n) if top_supp else ""

        margins = suppressor_margin.get((rid, top_supp), []) if top_supp else []
        metric_rows.append({
            "raw_id": rid,
            "class_name": cls.get("class_name", _class_name(rid, class_name_map)),
            "quadrant": cls.get("quadrant", ""),
            "group": cls.get("quadrant", ""),
            "support_bucket": cls.get("support_bucket", ""),
            "train_count": cls.get("train_count", ""),
            "val_count": cls.get("val_count", ""),
            "train_rank@1": cls.get("train_rank@1", ""),
            "val_rank@1": cls.get("val_rank@1", ""),
            "has_train_visual_proto": rid in train_proto,
            "has_val_visual_proto": rid in val_proto,
            "train_val_visual_proto_drift": drift,
            "text_visual_train_jaccard@10": _jaccard(text_n, train_vis_n),
            "text_visual_val_jaccard@10": _jaccard(text_n, val_vis_n),
            "train_proto_score_neighbor_jaccard@10": _jaccard(train_score_n, train_vis_n),
            "val_proto_score_neighbor_jaccard@10": _jaccard(val_score_n, val_vis_n),
            "text_neighbors@10": ";".join(text_n),
            "train_visual_neighbors@10": ";".join(train_vis_n),
            "val_visual_neighbors@10": ";".join(val_vis_n),
            "train_proto_score_neighbors@10": ";".join(train_score_n),
            "val_proto_score_neighbors@10": ";".join(val_score_n),
            "top_suppressor_raw_id": top_supp,
            "top_suppressor_name": _class_name(top_supp, class_name_map) if top_supp else "",
            "top_suppressor_count": top_supp_count,
            "is_person_suppressed": top_supp == _norm_id(args.person_raw_id),
            "mean_gt_vs_top_suppressor_margin": _mean(margins),
            "suppressor_text_rank": suppressor_text_rank,
            "suppressor_train_visual_rank": suppressor_train_visual_rank,
            "suppressor_val_visual_rank": suppressor_val_visual_rank,
            "suppressor_train_proto_score_rank": suppressor_train_score_rank,
            "suppressor_val_proto_score_rank": suppressor_val_score_rank,
        })

    group_summary = _summarize_group(metric_rows, "group")
    support_summary = _summarize_group(metric_rows, "support_bucket")

    suppressor_rank_rows = []
    for r in metric_rows:
        if not r.get("top_suppressor_raw_id"):
            continue
        suppressor_rank_rows.append({
            "raw_id": r.get("raw_id"),
            "class_name": r.get("class_name"),
            "group": r.get("group"),
            "support_bucket": r.get("support_bucket"),
            "top_suppressor_raw_id": r.get("top_suppressor_raw_id"),
            "top_suppressor_name": r.get("top_suppressor_name"),
            "top_suppressor_count": r.get("top_suppressor_count"),
            "is_person_suppressed": r.get("is_person_suppressed"),
            "suppressor_text_rank": r.get("suppressor_text_rank"),
            "suppressor_train_visual_rank": r.get("suppressor_train_visual_rank"),
            "suppressor_val_visual_rank": r.get("suppressor_val_visual_rank"),
            "suppressor_train_proto_score_rank": r.get("suppressor_train_proto_score_rank"),
            "suppressor_val_proto_score_rank": r.get("suppressor_val_proto_score_rank"),
            "mean_gt_vs_top_suppressor_margin": r.get("mean_gt_vs_top_suppressor_margin"),
        })

    _write_csv(out_root / "class_topology_metrics.csv", metric_rows)
    _write_csv(out_root / "group_topology_summary.csv", group_summary)
    _write_csv(out_root / "support_vs_topology_bucket_summary.csv", support_summary)
    _write_csv(out_root / "suppressor_neighbor_rank_summary.csv", suppressor_rank_rows)

    def _group_rec(name: str) -> Dict[str, Any]:
        for r in group_summary:
            if r.get("group") == name:
                return dict(r)
        return {"group": name, "class_count": 0}

    learned = _group_rec("learned_stable")
    overfit = _group_rec("overfit_context_fail")
    payload = {
        "status": "PASS",
        "output_root": str(out_root),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "asset_root": str(asset_root),
        "neighbor_k": k,
        "source_meta": {
            "train": train_meta,
            "val": val_meta,
            "train_vector_counters": train_vec_counters,
            "val_vector_counters": val_vec_counters,
        },
        "headline": {
            "learned_stable": learned,
            "overfit_context_fail": overfit,
            "overfit_minus_learned_mean_projected_score_visual_jaccard@10": _fnum(overfit.get("mean_val_proto_score_neighbor_jaccard@10")) - _fnum(learned.get("mean_val_proto_score_neighbor_jaccard@10")),
            "overfit_minus_learned_mean_train_val_visual_proto_drift": _fnum(overfit.get("mean_train_val_visual_proto_drift")) - _fnum(learned.get("mean_train_val_visual_proto_drift")),
        },
        "artifacts": {
            "class_topology_metrics": str(out_root / "class_topology_metrics.csv"),
            "group_topology_summary": str(out_root / "group_topology_summary.csv"),
            "support_vs_topology_bucket_summary": str(out_root / "support_vs_topology_bucket_summary.csv"),
            "suppressor_neighbor_rank_summary": str(out_root / "suppressor_neighbor_rank_summary.csv"),
        },
    }
    (out_root / "topology_mismatch_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# A8 Topology Mismatch Audit TAKEOVER",
        "",
        "- status: PASS",
        f"- output_root: {out_root}",
        f"- neighbor_k: {k}",
        "",
        "## Headline",
        f"- learned_stable_class_count: {learned.get('class_count')}",
        f"- overfit_context_fail_class_count: {overfit.get('class_count')}",
        f"- learned mean val_proto_score_neighbor_jaccard@10: {learned.get('mean_val_proto_score_neighbor_jaccard@10')}",
        f"- overfit mean val_proto_score_neighbor_jaccard@10: {overfit.get('mean_val_proto_score_neighbor_jaccard@10')}",
        f"- learned mean train_val_visual_proto_drift: {learned.get('mean_train_val_visual_proto_drift')}",
        f"- overfit mean train_val_visual_proto_drift: {overfit.get('mean_train_val_visual_proto_drift')}",
        "",
        "## Notes",
        "- This is a read-only topology audit; it does not train or mutate checkpoints.",
        "- *_score_neighbor_jaccard uses the current checkpoint scorer on visual class prototypes against text classes.",
        "- visual_proto_drift is 1 - cosine(train visual prototype, val visual prototype).",
        "",
        "## Artifacts",
    ]
    for p in payload["artifacts"].values():
        lines.append(f"- {p}")
    (out_root / "TOPOLOGY_MISMATCH_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
