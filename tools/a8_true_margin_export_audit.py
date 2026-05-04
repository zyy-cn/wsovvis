#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.run_stageb_analysis_residual_gated_coverage_assignment import (
    _default_row_gap_path,
    _inverse_softplus,
    _load_checkpoint_if_requested,
    _prepare_data,
)
from videocutler.run_stageb_train_residual_gated_hungarian_matched import (
    _evaluate_full_y,
    _example_by_tid,
    _load_eval_rows,
)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return int(default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        _norm_id(row.get("clip_id")),
        str(row.get("trajectory_id", "")).strip(),
        _norm_id(row.get("gt_raw_id")),
    )


def _load_reference_rows(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    rows = _read_csv(path)
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        out[_row_key(row)] = row
    return out


def _recompute_from_export(row: Mapping[str, Any]) -> Dict[str, Any]:
    cand_ids = json.loads(str(row.get("candidate_raw_ids_json", "[]")))
    cand_scores = json.loads(str(row.get("candidate_scores_json", "[]")))
    if len(cand_ids) != len(cand_scores) or not cand_ids:
        raise RuntimeError(f"invalid candidate score export for row_key={_row_key(row)}")
    scores = np.asarray(cand_scores, dtype=np.float64)
    order = np.argsort(-scores)
    top1_raw_id = int(cand_ids[int(order[0])])
    gt_raw_id = int(row["gt_raw_id"])
    try:
        gt_index = next(i for i, raw_id in enumerate(cand_ids) if int(raw_id) == gt_raw_id)
    except StopIteration as exc:
        raise RuntimeError(f"gt_raw_id={gt_raw_id} missing from candidate domain for row_key={_row_key(row)}") from exc
    gt_score = float(scores[int(gt_index)])
    gt_rank = int(1 + np.count_nonzero(scores > gt_score))
    return {
        "top1_raw_id": top1_raw_id,
        "gt_rank": gt_rank,
        "gt_score": gt_score,
    }


def _aggregate_by_class(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_norm_id(row.get("gt_raw_id"))].append(row)
    out: List[Dict[str, Any]] = []
    for raw_id, group in sorted(grouped.items(), key=lambda x: int(x[0])):
        wrong = [row for row in group if _as_int(row.get("top1_hit")) == 0]
        out.append(
            {
                "gt_raw_id": raw_id,
                "gt_class_name": group[0].get("gt_class_name", ""),
                "gt_count": len(group),
                "gt_top1_hit_rate": sum(_as_int(row.get("top1_hit")) for row in group) / max(len(group), 1),
                "mean_gt_rank": _mean([_as_float(row.get("gt_rank")) for row in group]),
                "mean_normalized_gt_rank": _mean([_as_float(row.get("normalized_gt_rank")) for row in group]),
                "mean_gt_score": _mean([_as_float(row.get("gt_score")) for row in group]),
                "mean_top1_score": _mean([_as_float(row.get("top1_score")) for row in group]),
                "mean_score_margin": _mean([_as_float(row.get("score_margin")) for row in group]),
                "wrong_count": len(wrong),
                "mean_wrong_abs_gap": _mean([_as_float(row.get("wrong_abs_gap")) for row in wrong]),
                "median_wrong_abs_gap": _median([_as_float(row.get("wrong_abs_gap")) for row in wrong]),
            }
        )
    return out


def _aggregate_by_edge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _as_int(row.get("top1_hit")) == 0:
            grouped[(_norm_id(row.get("gt_raw_id")), _norm_id(row.get("top1_raw_id")))].append(row)
    out: List[Dict[str, Any]] = []
    for (gt_raw_id, top1_raw_id), group in sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0][0], x[0][1])):
        out.append(
            {
                "gt_raw_id": gt_raw_id,
                "gt_class_name": group[0].get("gt_class_name", ""),
                "wrong_top1_raw_id": top1_raw_id,
                "wrong_count": len(group),
                "mean_wrong_abs_gap": _mean([_as_float(row.get("wrong_abs_gap")) for row in group]),
                "median_wrong_abs_gap": _median([_as_float(row.get("wrong_abs_gap")) for row in group]),
                "mean_score_margin": _mean([_as_float(row.get("score_margin")) for row in group]),
                "candidate_count_mean": _mean([_as_float(row.get("candidate_count")) for row in group]),
            }
        )
    return out


def _aggregate_absorber(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[_norm_id(edge.get("wrong_top1_raw_id"))].append(edge)
    out: List[Dict[str, Any]] = []
    for wrong_top1_raw_id, group in sorted(grouped.items(), key=lambda x: -sum(_as_int(e.get("wrong_count")) for e in x[1])):
        source_rows = sum(_as_int(edge.get("wrong_count")) for edge in group)
        out.append(
            {
                "wrong_top1_raw_id": wrong_top1_raw_id,
                "absorbed_wrong_rows": source_rows,
                "absorbed_source_class_count": len(group),
                "mean_wrong_abs_gap": _mean([_as_float(edge.get("mean_wrong_abs_gap")) for edge in group if edge.get("mean_wrong_abs_gap") not in (None, "")]),
                "top_source_edges_json": [
                    {
                        "gt_raw_id": edge.get("gt_raw_id", ""),
                        "gt_class_name": edge.get("gt_class_name", ""),
                        "wrong_count": _as_int(edge.get("wrong_count")),
                        "mean_wrong_abs_gap": edge.get("mean_wrong_abs_gap"),
                    }
                    for edge in sorted(group, key=lambda row: -_as_int(row.get("wrong_count")))[:20]
                ],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only A8 true-margin export audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--base_out", default="")
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    dataset_name = str(args.dataset_name)
    base_out = Path(args.base_out).expanduser().resolve() if str(args.base_out).strip() else run_root / "outputs" / "a8_hungarian_prealign_ablation" / dataset_name / "baseline_full_y_5ep_base_ce_50ep"
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else Path(args.repo_root).expanduser().resolve() / "codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427/analysis/a8_baseline_full_y_5ep_true_margin_audit" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "export_replay"
    reference_row_csv = base_out / "analysis" / "eval_after_row_predictions.csv"
    reference_by_class_csv = base_out / "analysis" / "eval_after_by_class.csv"
    explicit_checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if str(args.checkpoint_path).strip() else None
    default_checkpoint_path = (base_out / "train" / "a8_hungarian_matched" / "a8_hungarian_last.pth").resolve()
    checkpoint_path = explicit_checkpoint_path if explicit_checkpoint_path is not None else default_checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"A8 checkpoint not found: {checkpoint_path}")

    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.repo_root) / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(Path(args.repo_root) / "package" / "reference" / "lvvis_official_base_novel_split.json")
    args.output_dir = ""
    args.smoke = False
    args.smoke_max_trajectories = 512
    args.subset_fraction = None
    args.seed = 3407

    data = _prepare_data(args)
    example_by_tid = _example_by_tid(data)
    row_gap_csv = Path(args.row_gap_csv).expanduser().resolve() if str(args.row_gap_csv).strip() else _default_row_gap_path(Path(args.repo_root).expanduser().resolve(), dataset_name)
    eval_rows, eval_summary = _load_eval_rows(row_gap_csv, data, max_rows=0, seed=3407)
    if not eval_rows:
        raise RuntimeError("no eval rows available for A8 true-margin audit")

    device = torch.device(str(args.device))
    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(0.07), device=device, dtype=torch.float32))
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, str(checkpoint_path), device)
    if not checkpoint_summary.get("loaded_projector"):
        raise RuntimeError(f"failed to load A8 projector checkpoint: {checkpoint_summary}")

    replay_summary = _evaluate_full_y(
        stage="after",
        rows=eval_rows,
        data=data,
        example_by_tid=example_by_tid,
        projector=projector,
        text_tensor=text_tensor,
        theta_t=theta_t,
        device=device,
        out_dir=export_dir,
    )
    exported_rows = _read_csv(export_dir / "eval_after_row_predictions.csv")
    reference_validation_available = bool(reference_row_csv.is_file()) and checkpoint_path == default_checkpoint_path
    reference_rows = _load_reference_rows(reference_row_csv) if reference_validation_available else {}
    validation_rows: List[Dict[str, Any]] = []
    mismatch_count = 0
    recompute_mismatch_count = 0
    for row in exported_rows:
        key = _row_key(row)
        recomputed = _recompute_from_export(row)
        ref = reference_rows.get(key)
        top1_match_ref = (not reference_validation_available) or (ref is not None and _norm_id(ref.get("top1_raw_id")) == _norm_id(row.get("top1_raw_id")))
        rank_match_ref = (not reference_validation_available) or (ref is not None and _as_int(ref.get("gt_rank")) == _as_int(row.get("gt_rank")))
        top1_match_recomputed = int(recomputed["top1_raw_id"]) == _as_int(row.get("top1_raw_id"))
        rank_match_recomputed = int(recomputed["gt_rank"]) == _as_int(row.get("gt_rank"))
        if reference_validation_available and not (top1_match_ref and rank_match_ref):
            mismatch_count += 1
        if not (top1_match_recomputed and rank_match_recomputed):
            recompute_mismatch_count += 1
        validation_rows.append(
            {
                **row,
                "ref_top1_raw_id": ref.get("top1_raw_id") if ref else "",
                "ref_gt_rank": ref.get("gt_rank") if ref else "",
                "recomputed_top1_raw_id": recomputed["top1_raw_id"],
                "recomputed_gt_rank": recomputed["gt_rank"],
                "top1_match_reference": int(top1_match_ref),
                "gt_rank_match_reference": int(rank_match_ref),
                "top1_match_recomputed": int(top1_match_recomputed),
                "gt_rank_match_recomputed": int(rank_match_recomputed),
            }
        )
    mismatch_rate = float(mismatch_count / max(len(exported_rows), 1))
    recompute_mismatch_rate = float(recompute_mismatch_count / max(len(exported_rows), 1))
    if mismatch_rate > 0.0001 or recompute_mismatch_rate > 0.0001:
        raise RuntimeError(
            f"semantic drift detected: mismatch_rate={mismatch_rate:.6f} recompute_mismatch_rate={recompute_mismatch_rate:.6f}"
        )

    by_class_rows = _aggregate_by_class(validation_rows)
    by_edge_rows = _aggregate_by_edge(validation_rows)
    absorber_rows = _aggregate_absorber(by_edge_rows)

    _write_csv(output_dir / "true_score_margin_row_audit.csv", validation_rows)
    _write_csv(output_dir / "true_score_margin_by_class.csv", by_class_rows)
    _write_csv(output_dir / "true_score_margin_by_confusion_edge.csv", by_edge_rows)
    _write_csv(output_dir / "absorber_true_margin_summary.csv", absorber_rows)

    validation_summary = {
        "status": "PASS",
        "checkpoint_path": str(checkpoint_path),
        "reference_row_csv": str(reference_row_csv),
        "reference_by_class_csv": str(reference_by_class_csv),
        "reference_validation_available": bool(reference_validation_available),
        "exported_row_csv": str(export_dir / "eval_after_row_predictions.csv"),
        "score_domain": "full_y_clip_logits_div_t_dis",
        "row_count": len(exported_rows),
        "eval_summary": replay_summary,
        "eval_row_summary": eval_summary,
        "mismatch_count_vs_existing_csv": mismatch_count,
        "mismatch_rate_vs_existing_csv": mismatch_rate,
        "recompute_mismatch_count_from_exported_scores": recompute_mismatch_count,
        "recompute_mismatch_rate_from_exported_scores": recompute_mismatch_rate,
        "checkpoint_load_summary": checkpoint_summary,
    }
    _write_json(output_dir / "validation_summary.json", validation_summary)
    takeover = [
        "# A8 True Margin Audit",
        "",
        f"- status: PASS",
        f"- checkpoint_path: {checkpoint_path}",
        f"- true_score_margin_available: yes",
        f"- score_domain: full_y_clip_logits_div_t_dis",
        f"- reference_validation_available: {str(reference_validation_available).lower()}",
        f"- mismatch_rate_vs_existing_csv: {mismatch_rate:.6f}",
        f"- recompute_mismatch_rate_from_exported_scores: {recompute_mismatch_rate:.6f}",
        f"- row_count: {len(exported_rows)}",
        "",
        "## Outputs",
        f"- {output_dir / 'true_score_margin_row_audit.csv'}",
        f"- {output_dir / 'true_score_margin_by_class.csv'}",
        f"- {output_dir / 'true_score_margin_by_confusion_edge.csv'}",
        f"- {output_dir / 'absorber_true_margin_summary.csv'}",
        f"- {output_dir / 'validation_summary.json'}",
    ]
    (output_dir / "TRUE_MARGIN_AUDIT_TAKEOVER.md").write_text("\n".join(takeover) + "\n", encoding="utf-8")
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
