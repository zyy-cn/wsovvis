from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import (  # noqa: E402
    _clip_groups,
    _load_reservoir_checkpoint,
    _prepare_prealign_examples,
    _scope_text_vocab,
    _sinkhorn_collect_responsibility_rows,
    _write_json,
    _write_jsonl,
)
from videocutler.ext_stageb_ovvis.analysis.extra_attribution_probe import (  # noqa: E402
    ExtraAttributionProbeConfig,
    _materialize_valid_samples,
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        if path.is_file():
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, Mapping):
                return obj
    except Exception:
        pass
    return {}


def _allowed_raw_ids_from_run(run_root: Path, checkpoint_payload: Mapping[str, Any], full_text_vocab_ids: Sequence[int]) -> list[int]:
    policy = checkpoint_payload.get("vocab_scope_policy")
    if isinstance(policy, Mapping):
        vals = policy.get("allowed_train_vocab_raw_ids")
        if isinstance(vals, list) and vals:
            return [int(x) for x in vals]
    for p in [run_root / "train" / "prealign" / "stage_summary.json", run_root / "train" / "pipeline_train_summary.json"]:
        obj = _load_json(p)
        pol = obj.get("vocab_scope_policy") if isinstance(obj, Mapping) else None
        if isinstance(pol, Mapping):
            vals = pol.get("allowed_train_vocab_raw_ids")
            if isinstance(vals, list) and vals:
                return [int(x) for x in vals]
    return [int(x) for x in full_text_vocab_ids]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export support-null prealign responsibility rows from an existing checkpoint.")
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--runtime_output_root", required=True, type=Path)
    parser.add_argument("--dataset_name", default="lvvis_train_base")
    parser.add_argument("--trajectory_source_branch", default="mainline")
    parser.add_argument("--stage", default="prealign")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--sinkhorn_tau", default=0.15, type=float)
    parser.add_argument("--sinkhorn_iters", default=5, type=int)
    parser.add_argument("--sinkhorn_row_cap_scale", default=2.0, type=float)
    parser.add_argument("--sinkhorn_null_logit_bias", default=-1.0, type=float)
    parser.add_argument("--sinkhorn_null_residual", default=True, type=_parse_bool)
    parser.add_argument("--sinkhorn_null_demand_cap_ratio", default=0.35, type=float)
    parser.add_argument("--sinkhorn_yprime_demand_mode", default="relative_margin_ema")
    parser.add_argument("--sinkhorn_yprime_demand_min", default=0.20, type=float)
    parser.add_argument("--sinkhorn_yprime_support_topk", default=2, type=int)
    parser.add_argument("--sinkhorn_yprime_support_temp", default=0.25, type=float)
    parser.add_argument("--smoke", default=False, type=_parse_bool)
    parser.add_argument("--smoke_max_trajectories", default=1024, type=int)
    parser.add_argument("--subset_fraction", default=None, type=float)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else run_root / "train" / str(args.stage) / "checkpoints" / f"{args.stage}_last.pth"
    if not checkpoint.is_file() and str(args.stage) == "prealign":
        checkpoint = run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    proxy_config = ExtraAttributionProbeConfig(
        run_root=run_root,
        runtime_output_root=runtime_output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        device=str(args.device),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        stage_scope=(str(args.stage),),
        batch_size=512,
        output_dir=run_root / "analysis" / "support_null_responsibility_export" / str(args.dataset_name) / str(args.stage),
        sidecar_root=run_root,
        show_progress=False,
    )
    materialized = _materialize_valid_samples(proxy_config)
    prepared = _prepare_prealign_examples(
        list(materialized.get("valid_samples", [])),
        output_root=runtime_output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
    )
    examples = list(prepared.get("examples", []))
    if not examples:
        raise RuntimeError("no trainable prealign examples materialized")
    groups = _clip_groups(examples)

    device = torch.device(str(args.device))
    projector, theta_t, _unknown, payload = _load_reservoir_checkpoint(checkpoint, device=device)
    full_text_vocab_ids, _text_records, full_text_vocab_matrix = load_text_vocab(runtime_output_root)
    allowed = _allowed_raw_ids_from_run(run_root, payload, full_text_vocab_ids)
    text_vocab_ids, text_vocab_matrix, text_scope_meta = _scope_text_vocab(full_text_vocab_ids, np.asarray(full_text_vocab_matrix, dtype=np.float32), allowed)
    raw_to_vocab_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    support_state = payload.get("support_null_state_snapshot")
    if not isinstance(support_state, Mapping):
        support_state = {}
    support_cfg = payload.get("support_null_config")
    if not isinstance(support_cfg, Mapping):
        support_cfg = {}

    rows = _sinkhorn_collect_responsibility_rows(
        stage_id=str(args.stage),
        dataset_name=str(args.dataset_name),
        groups=groups,
        output_root=run_root,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        raw_to_vocab_idx=raw_to_vocab_idx,
        mode="prealign",
        sinkhorn_tau=float(args.sinkhorn_tau),
        sinkhorn_iters=int(args.sinkhorn_iters),
        sinkhorn_row_cap_scale=float(args.sinkhorn_row_cap_scale),
        extra_demand=0.0,
        sinkhorn_final_rerank_lambda_r=float(payload.get("sinkhorn_final_rerank_lambda_r", 0.0)),
        vocab_scope_policy={"policy": "weak_label_only", "allowed_train_vocab_raw_ids": [int(x) for x in allowed], **dict(text_scope_meta)},
        enable_null_column=True,
        null_logit_bias=float(args.sinkhorn_null_logit_bias),
        null_residual=bool(args.sinkhorn_null_residual),
        null_demand_cap_ratio=float(args.sinkhorn_null_demand_cap_ratio),
        yprime_demand_mode=str(args.sinkhorn_yprime_demand_mode),
        yprime_demand_min=float(args.sinkhorn_yprime_demand_min),
        yprime_support_topk=int(args.sinkhorn_yprime_support_topk),
        yprime_support_temp=float(args.sinkhorn_yprime_support_temp),
        support_state_snapshot=support_state,
        enable_positive_protection=bool(support_cfg.get("positive_protection_enabled", False)),
        positive_margin_threshold=float(support_cfg.get("positive_margin_threshold", 0.15)),
        positive_margin_temp=float(support_cfg.get("positive_margin_temp", 0.10)),
        positive_null_cap=float(support_cfg.get("positive_null_cap", 0.40)),
        positive_redistribute_mode=str(support_cfg.get("positive_redistribute_mode", "best_y")),
    )
    train_dir = run_root / "train" / str(args.stage)
    _write_jsonl(train_dir / "responsibility_records.jsonl", rows)
    _write_jsonl(train_dir / "proxy_records.jsonl", rows)
    summary = {
        "status": "PASS",
        "stage": str(args.stage),
        "checkpoint": str(checkpoint),
        "record_count_output": int(len(rows)),
        "responsibility_records_path": str(train_dir / "responsibility_records.jsonl"),
        "proxy_records_path": str(train_dir / "proxy_records.jsonl"),
        "null_candidate_row_count": int(sum(1 for row in rows if "-1" in dict(row.get("r_final", {})))),
        "demand_candidate_row_count": int(sum(1 for row in rows if "-1" in dict(row.get("candidate_demand_by_raw_id", {})))),
        "text_scope_meta": dict(text_scope_meta),
        "note": "This is a posthoc export. If checkpoint lacks support_null_state_snapshot, demand uses instantaneous relative-margin confidence rather than final EMA state.",
    }
    out = run_root / "analysis" / "support_null_responsibility_export" / str(args.dataset_name) / str(args.stage)
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
