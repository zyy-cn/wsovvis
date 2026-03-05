#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wsovvis.training import build_stagec_semantic_plumbing_c3_minimal_coupled


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic C5 Stage C semantic micro-training")
    p.add_argument("--steps", type=int, default=6, help="Number of micro-training steps")
    p.add_argument("--seed", type=int, default=20260305, help="Global random seed")
    p.add_argument("--lr", type=float, default=0.08, help="Optimizer learning rate")
    p.add_argument("--cache-root", type=Path, default=Path("/tmp/wsovvis_stagec_c5_clip_cache"), help="C1 prototype cache root")
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


def main() -> int:
    args = _build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and > 0")

    _set_seed(args.seed)

    # Small deterministic synthetic batch for canonical C5 micro-training.
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
    track_objectness = torch.tensor([0.95, 0.90, 0.85, 0.35], dtype=torch.float32)
    optimizer = torch.optim.Adam([track_features], lr=float(args.lr))

    config = {
        "enabled": True,
        "loss_key": "loss_stage_c_semantic",
        "loss_weight": 1.0,
        "assignment_backend": "c2_sinkhorn_minimal_v1",
        "sinkhorn_temperature": 0.10,
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

    label_text = {
        101: "person",
        202: "dog",
        303: "car",
        "__bg__": "background",
        "__unk_fg__": "unknown foreground",
    }

    step_summaries: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        out = build_stagec_semantic_plumbing_c3_minimal_coupled(
            config,
            track_features_tensor=track_features,
            positive_label_ids=(101, 202),
            topk_label_ids=(101, 202, 303),
            merge_mode="yp_plus_topk",
            include_bg=True,
            include_unk_fg=True,
            cache_root=args.cache_root,
            label_text_by_id=label_text,
            track_objectness_tensor=track_objectness,
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
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
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
