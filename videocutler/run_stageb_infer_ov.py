from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

import torch

from videocutler.ext_stageb_ovvis.eval.g8_bridge import (
    G8Paths,
    build_cli_contract_summary,
    build_infer_rows,
    build_pred_rows,
    load_projector_bundle,
    load_text_vocab_with_names,
    load_video_meta,
    repo_root,
    require_dataset_name,
    resolve_inference_asset_roots,
    resolve_selected_for_infer,
    score_infer_row,
    validate_json_artifact,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G8 open-vocabulary inference CLI.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_val", "ytvis_2019_val"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--logit_chunk_size", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ckpt_path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_dataset_name(args.dataset_name)
    output_root = Path(args.output_root).expanduser().resolve()
    resolution = resolve_selected_for_infer(output_root, ckpt_path=args.ckpt_path)
    asset_roots = resolve_inference_asset_roots(
        output_root,
        dataset_name=args.dataset_name,
        trajectory_source_branch="mainline",
        resolution=resolution,
    )
    device = torch.device(args.device)
    bundle = load_projector_bundle(resolution.checkpoint_path, device=device)
    text_vocab_ids, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_roots.asset_root, args.dataset_name)
    video_meta = load_video_meta(args.dataset_name)
    infer_rows, skipped, asset_counts = build_infer_rows(asset_roots, dataset_name=args.dataset_name)

    if args.smoke:
        infer_rows = infer_rows[: min(8, len(infer_rows))]

    scored_rows = [
        score_infer_row(
            row,
            bundle=bundle,
            asset_root=asset_roots.asset_root,
            dataset_name=args.dataset_name,
            trajectory_source_branch="mainline",
            text_vocab_ids=text_vocab_ids,
            text_matrix=text_matrix,
            class_name_map=class_name_map,
            logit_chunk_size=args.logit_chunk_size,
        )
        for row in infer_rows
    ]
    pred_main, pred_diag = build_pred_rows(scored_rows, video_meta=video_meta)

    validate_json_artifact(pred_main, "pred_main.schema.json")
    validate_json_artifact(pred_diag, "pred_diag.schema.json")
    paths = G8Paths(output_root, args.dataset_name)
    write_json(paths.pred_main_path, pred_main)
    write_json(paths.pred_diag_path, pred_diag)

    summary = {
        "status": "PASS",
        "cli": build_cli_contract_summary("contracts/cli/run_stageb_infer_ov.cli_contract.json"),
        "selected_for_infer": resolution.selected_for_infer,
        "checkpoint_path": str(resolution.checkpoint_path),
        "resolution_source": resolution.source,
        "train_state_path": None if resolution.train_state_path is None else str(resolution.train_state_path),
        "asset_root": str(asset_roots.asset_root),
        "pred_main_path": str(paths.pred_main_path),
        "pred_diag_path": str(paths.pred_diag_path),
        "trajectory_source_branch": "mainline",
        "dataset_name": args.dataset_name,
        "scored_row_count": len(scored_rows),
        "skipped_trajectory_histogram": skipped,
        "asset_counts": asset_counts,
        "text_vocab_size": len(text_vocab_ids),
        "checkpoint_stage_id": bundle.stage_id,
        "temperature": float(bundle.temperature),
        "unknown_logit": float(bundle.unknown_logit),
        "smoke": bool(args.smoke),
        "logit_chunk_size": int(args.logit_chunk_size),
    }
    contract_path = repo_root() / "codex" / "outputs" / "G8_inference_and_eval" / "infer_contract_step2.json"
    write_json(contract_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
