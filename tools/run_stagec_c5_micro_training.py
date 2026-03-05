#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wsovvis.track_feature_export import build_normalized_bridge_input_from_real_stageb_sidecar
from wsovvis.training import build_stagec_semantic_plumbing_c3_minimal_coupled


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic C5 Stage C semantic micro-training")
    p.add_argument(
        "--data-mode",
        choices=("real_sidecar_v1", "synthetic_v1"),
        default="real_sidecar_v1",
        help="Input source for micro-training. real_sidecar_v1 is the canonical C5.r1 mode.",
    )
    p.add_argument("--steps", type=int, default=6, help="Number of micro-training steps")
    p.add_argument("--seed", type=int, default=20260305, help="Global random seed")
    p.add_argument("--lr", type=float, default=0.08, help="Optimizer learning rate")
    p.add_argument(
        "--sinkhorn-temperature",
        type=float,
        default=0.10,
        help="C2 Sinkhorn temperature (tau). Primary sweep factor for C6.",
    )
    p.add_argument("--cache-root", type=Path, default=Path("/tmp/wsovvis_stagec_c5_clip_cache"), help="C1 prototype cache root")
    p.add_argument(
        "--real-run-root",
        type=Path,
        default=Path("runs/wsovvis_seqformer/18"),
        help="Stage B real run root used by --data-mode=real_sidecar_v1",
    )
    p.add_argument(
        "--sample-video-limit",
        type=int,
        default=20,
        help="Tiny deterministic sampling limit for sidecar-to-bridge discovery in real mode",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=8,
        help="Max tracks consumed from selected real-backed video",
    )
    p.add_argument(
        "--max-positive-labels",
        type=int,
        default=6,
        help="Cap positive labels from annotation-derived video labels",
    )
    p.add_argument(
        "--min-positive-labels",
        type=int,
        default=1,
        help="Require at least this many positive labels when selecting real-backed sample.",
    )
    p.add_argument(
        "--preferred-video-id",
        type=str,
        default=None,
        help="Optional explicit video_id to select in real mode (must satisfy selection constraints).",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write full run summary JSON")
    return p


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _step_summary(step: int, out: dict[str, Any], loss_value: float) -> dict[str, Any]:
    diag = out["diagnostics"]
    c4 = diag["c4_observability"]
    return {
        "step": int(step),
        "loss_total": float(loss_value),
        "loss_component_alignment": float(diag.get("component_alignment", 0.0)),
        "loss_component_coverage": float(diag.get("component_coverage", 0.0)),
        "loss_component_fg_not_bg": float(diag.get("component_fg_not_bg", 0.0)),
        "assignment_backend": str(out.get("assignment_backend")),
        "candidate_label_ids": list(out["candidate_summary"]["candidate_label_ids"]),
        "c4_bg_mass_fraction": float(c4["assignment_mass"]["bg_mass_fraction"]),
        "c4_unk_fg_mass_fraction": float(c4["assignment_mass"]["unk_fg_mass_fraction"]),
        "c4_non_special_mass_fraction": float(c4["assignment_mass"]["non_special_mass_fraction"]),
        "c4_coverage_ratio_present": float(c4["coverage"]["coverage_ratio_present"]),
        "c4_mean_row_entropy": float(c4["distribution"]["mean_row_entropy"]),
        "c4_mean_top1_mass": float(c4["distribution"]["mean_top1_mass"]),
        "c4_high_obj_bg_dom_frac": float(c4["fg_not_bg_monitor"]["high_objectness_bg_dominant_fraction"]),
        "c4_backend_echo": c4["backend_echo"],
    }


def _resolve_split_annotation_path(run_root: Path) -> Path:
    cfg_path = run_root / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config JSON object: {cfg_path}")
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError(f"missing config.data object: {cfg_path}")
    rel = data_cfg.get("val_json")
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"missing config.data.val_json string: {cfg_path}")
    val_path = Path(rel)
    if val_path.is_absolute():
        return val_path
    repo_root = Path.cwd()
    candidate_paths = [repo_root / val_path]
    probe = run_root
    while probe != probe.parent:
        candidate_paths.append(probe / val_path)
        probe = probe.parent
    candidate_paths.append(Path.cwd() / val_path)
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def _load_annotation_label_lookup(split_json_path: Path) -> tuple[dict[str, list[int]], dict[int, str]]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"annotation JSON must be object: {split_json_path}")
    videos = payload.get("videos")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(videos, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"annotation JSON must contain list keys videos/annotations/categories: {split_json_path}")

    known_video_ids = {str(v["id"]) for v in videos if isinstance(v, dict) and isinstance(v.get("id"), int)}
    labels_by_video: dict[str, set[int]] = {video_id: set() for video_id in known_video_ids}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        vid = ann.get("video_id")
        cid = ann.get("category_id")
        if not isinstance(vid, int) or not isinstance(cid, int):
            continue
        vid_key = str(vid)
        if vid_key in labels_by_video:
            labels_by_video[vid_key].add(cid)

    cat_name_by_id: dict[int, str] = {}
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        cid = cat.get("id")
        if not isinstance(cid, int):
            continue
        name = cat.get("name")
        cat_name_by_id[cid] = name if isinstance(name, str) and name else f"class_{cid}"
    return {k: sorted(v) for k, v in labels_by_video.items()}, cat_name_by_id


def load_real_backed_micro_batch(
    *,
    run_root: Path,
    sample_video_limit: int,
    max_tracks: int,
    max_positive_labels: int,
    min_positive_labels: int = 1,
    preferred_video_id: str | None = None,
) -> dict[str, Any]:
    if sample_video_limit < 1:
        raise ValueError("sample_video_limit must be >= 1")
    if max_tracks < 1:
        raise ValueError("max_tracks must be >= 1")
    if max_positive_labels < 1:
        raise ValueError("max_positive_labels must be >= 1")
    if min_positive_labels < 1:
        raise ValueError("min_positive_labels must be >= 1")

    bridge_payload, bridge_summary = build_normalized_bridge_input_from_real_stageb_sidecar(
        run_root=run_root,
        sample_video_limit=sample_video_limit,
    )
    split_json_path = _resolve_split_annotation_path(run_root)
    labels_by_video, cat_name_by_id = _load_annotation_label_lookup(split_json_path)

    selected_video: dict[str, Any] | None = None
    selected_positive_ids: list[int] = []
    preferred_key = None if preferred_video_id is None else str(preferred_video_id)
    for result in bridge_payload.get("stageb_video_results", ()):
        if not isinstance(result, dict):
            continue
        if result.get("runtime_status") != "success":
            continue
        tracks = result.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            continue
        video_id = str(result.get("video_id"))
        if preferred_key is not None and video_id != preferred_key:
            continue
        positives = labels_by_video.get(video_id, [])
        if len(positives) < min_positive_labels:
            continue
        selected_video = result
        selected_positive_ids = positives[:max_positive_labels]
        break
    if selected_video is None:
        if preferred_key is not None:
            raise ValueError(
                "failed to select preferred video with required labels/tracks; "
                f"run_root={run_root}; preferred_video_id={preferred_key}; min_positive_labels={min_positive_labels}"
            )
        raise ValueError(
            "failed to select real-backed video with success tracks and sufficient annotation labels; "
            f"run_root={run_root}; min_positive_labels={min_positive_labels}"
        )

    selected_tracks = selected_video["tracks"][:max_tracks]
    track_features_np = np.asarray([track["embedding"] for track in selected_tracks], dtype=np.float32)
    track_objectness_np = np.asarray([track["objectness_score"] for track in selected_tracks], dtype=np.float32)
    if track_features_np.ndim != 2 or track_features_np.shape[0] == 0 or track_features_np.shape[1] <= 0:
        raise ValueError("real-backed track feature tensor is invalid")
    if not np.isfinite(track_features_np).all() or not np.isfinite(track_objectness_np).all():
        raise ValueError("real-backed track features/objectness must be finite")

    positive_label_ids = tuple(int(x) for x in selected_positive_ids)
    topk_label_ids = tuple(positive_label_ids)
    label_text = {int(cid): str(cat_name_by_id.get(int(cid), f"class_{int(cid)}")) for cid in positive_label_ids}
    label_text["__bg__"] = "background"
    label_text["__unk_fg__"] = "unknown foreground"

    return {
        "track_features_tensor": torch.nn.Parameter(torch.as_tensor(track_features_np, dtype=torch.float32)),
        "track_objectness_tensor": torch.as_tensor(track_objectness_np, dtype=torch.float32),
        "positive_label_ids": positive_label_ids,
        "topk_label_ids": topk_label_ids,
        "label_text_by_id": label_text,
        "selected_video_id": str(selected_video["video_id"]),
        "selected_num_tracks_total": int(len(selected_video["tracks"])),
        "selected_num_tracks_used": int(track_features_np.shape[0]),
        "selected_num_positive_labels": int(len(positive_label_ids)),
        "selected_positive_label_ids": [int(x) for x in positive_label_ids],
        "split_annotation_json_path": str(split_json_path),
        "bridge_summary": bridge_summary,
    }


def _build_synthetic_batch() -> dict[str, Any]:
    track_features = torch.nn.Parameter(
        torch.tensor(
            [
                [1.6, 0.2, 0.0, 0.1],
                [0.1, 1.5, 0.2, 0.0],
                [0.2, 0.2, 1.2, 0.1],
                [0.4, 0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
    )
    return {
        "track_features_tensor": track_features,
        "track_objectness_tensor": torch.tensor([0.95, 0.90, 0.85, 0.35], dtype=torch.float32),
        "positive_label_ids": (101, 202),
        "topk_label_ids": (101, 202, 303),
        "label_text_by_id": {
            101: "person",
            202: "dog",
            303: "car",
            "__bg__": "background",
            "__unk_fg__": "unknown foreground",
        },
        "selected_video_id": "synthetic",
        "selected_num_tracks_total": 4,
        "selected_num_tracks_used": 4,
        "selected_num_positive_labels": 2,
        "split_annotation_json_path": None,
        "bridge_summary": None,
    }


def main() -> int:
    args = _build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and > 0")

    _set_seed(args.seed)
    if args.data_mode == "real_sidecar_v1":
        batch = load_real_backed_micro_batch(
            run_root=args.real_run_root.resolve(),
            sample_video_limit=int(args.sample_video_limit),
            max_tracks=int(args.max_tracks),
            max_positive_labels=int(args.max_positive_labels),
            min_positive_labels=int(args.min_positive_labels),
            preferred_video_id=args.preferred_video_id,
        )
    else:
        batch = _build_synthetic_batch()
    optimizer = torch.optim.Adam([batch["track_features_tensor"]], lr=float(args.lr))

    config = {
        "enabled": True,
        "loss_key": "loss_stage_c_semantic",
        "loss_weight": 1.0,
        "assignment_backend": "c2_sinkhorn_minimal_v1",
        "sinkhorn_temperature": float(args.sinkhorn_temperature),
        "sinkhorn_iterations": 20,
        "sinkhorn_tolerance": 1e-6,
        "sinkhorn_eps": 1e-12,
        "sinkhorn_bg_capacity_weight": 1.5,
        "sinkhorn_unk_fg_capacity_weight": 1.5,
        "c3_score_temperature": 0.15,
        "c3_alignment_weight": 1.0,
        "c3_coverage_weight": 0.30,
        "c3_fg_not_bg_weight": 0.10,
    }

    step_summaries: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        out = build_stagec_semantic_plumbing_c3_minimal_coupled(
            config,
            track_features_tensor=batch["track_features_tensor"],
            positive_label_ids=batch["positive_label_ids"],
            topk_label_ids=batch["topk_label_ids"],
            merge_mode="yp_plus_topk",
            include_bg=True,
            include_unk_fg=True,
            cache_root=args.cache_root,
            label_text_by_id=batch["label_text_by_id"],
            track_objectness_tensor=batch["track_objectness_tensor"],
        )
        loss = out["semantic_loss_tensor"]
        loss.backward()
        optimizer.step()

        summary = _step_summary(step=step, out=out, loss_value=float(loss.detach().cpu().item()))
        step_summaries.append(summary)
        print("C5_STEP", json.dumps(summary, sort_keys=True))

    final = step_summaries[-1]
    result = {
        "status": "PASS",
        "data_mode": args.data_mode,
        "real_run_root": str(args.real_run_root.resolve()) if args.data_mode == "real_sidecar_v1" else None,
        "selected_video_id": batch["selected_video_id"],
        "selected_num_tracks_total": batch["selected_num_tracks_total"],
        "selected_num_tracks_used": batch["selected_num_tracks_used"],
        "selected_num_positive_labels": batch["selected_num_positive_labels"],
        "selected_positive_label_ids": batch["selected_positive_label_ids"],
        "split_annotation_json_path": batch["split_annotation_json_path"],
        "bridge_summary": batch["bridge_summary"],
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "sinkhorn_temperature": float(args.sinkhorn_temperature),
        "min_positive_labels": int(args.min_positive_labels),
        "preferred_video_id": args.preferred_video_id,
        "backend_locked": final["assignment_backend"] == "c2_sinkhorn_minimal_v1",
        "final": final,
        "all_steps": step_summaries,
    }

    print("C5_SUMMARY", json.dumps(result, sort_keys=True))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"C5_SUMMARY_PATH {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
