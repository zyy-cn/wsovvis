from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from tqdm.auto import tqdm
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import GTAttributionRankAuditConfig, run_gt_attribution_rank_audit
from videocutler.ext_stageb_ovvis.eval.external_lvvis import ExternalLVVISEvalConfig, run_external_lvvis_eval
from videocutler.ext_stageb_ovvis.eval.g8_bridge import (
    G8Paths,
    build_carrier_only_infer_pack,
    build_infer_rows,
    build_pred_rows,
    load_projector_bundle,
    load_text_vocab_with_names,
    load_video_meta,
    materialize_scored_rows_from_matrix,
    resolve_inference_asset_roots,
    resolve_selected_for_infer,
    score_infer_rows_matrix,
    validate_json_artifact,
    write_json,
    write_jsonl,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def collect_runtime_metrics(output_root: Path) -> Dict[str, Any]:
    path = output_root / 'train' / 'pipeline_train_summary.json'
    return dict(_read_json(path).get('stages', {})) if path.is_file() else {}


def _selected_audit_stage(selected_for_infer: str) -> str:
    mapping = {
        'prealign_only': 'prealign',
        'base_only': 'softem_base',
        'augmented': 'softem_aug',
    }
    return mapping.get(str(selected_for_infer), 'softem_aug')


def run_inference(*, output_root: Path, dataset_name: str, device: str, seed: int, logit_chunk_size: int, smoke: bool, ckpt_path: Optional[str] = None, show_progress: bool = True) -> Dict[str, Any]:
    stage_bar = tqdm(total=8, desc='inference', unit='stage', leave=True) if show_progress else None
    try:
        resolution = resolve_selected_for_infer(output_root, ckpt_path=ckpt_path)
        asset_roots = resolve_inference_asset_roots(output_root, dataset_name=dataset_name, trajectory_source_branch='mainline', resolution=resolution)
        device_obj = torch.device(device)
        bundle = load_projector_bundle(resolution.checkpoint_path, device=device_obj)
        text_vocab_ids, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_roots.asset_root, dataset_name)
        video_meta = load_video_meta(dataset_name)
        if stage_bar is not None:
            stage_bar.set_postfix_str('resolve assets/model')
            stage_bar.update(1)

        infer_rows, skipped, asset_counts = build_infer_rows(asset_roots, dataset_name=dataset_name)
        infer_rows = infer_rows[:min(8, len(infer_rows))] if smoke else infer_rows
        if stage_bar is not None:
            stage_bar.set_postfix_str(f'rows={len(infer_rows)}')
            stage_bar.update(1)

        pack = build_carrier_only_infer_pack(
            infer_rows,
            asset_root=asset_roots.asset_root,
            dataset_name=dataset_name,
            trajectory_source_branch='mainline',
            show_progress=show_progress,
        )
        if stage_bar is not None:
            stage_bar.set_postfix_str('carrier pack ready')
            stage_bar.update(1)

        scores = score_infer_rows_matrix(
            carrier_matrix=pack['carrier_matrix'],
            bundle=bundle,
            text_matrix=text_matrix,
            show_progress=show_progress,
        )
        if stage_bar is not None:
            stage_bar.set_postfix_str('matrix scored')
            stage_bar.update(1)

        scored_rows = materialize_scored_rows_from_matrix(
            row_manifest=pack['row_manifest'],
            trajectory_records=pack['trajectory_records'],
            text_vocab_ids=text_vocab_ids,
            class_name_map=class_name_map,
            fused_logits=scores['fused_logits'],
            known_probs=scores['known_probs'],
            unknown_probs=scores['unknown_probs'],
            show_progress=show_progress,
        )
        if stage_bar is not None:
            stage_bar.set_postfix_str('rows materialized')
            stage_bar.update(1)

        pred_main, pred_diag = build_pred_rows(scored_rows, video_meta=video_meta)
        if stage_bar is not None:
            stage_bar.set_postfix_str('pred rows built')
            stage_bar.update(1)

        validate_json_artifact(pred_main, 'pred_main.schema.json')
        validate_json_artifact(pred_diag, 'pred_diag.schema.json')
        if stage_bar is not None:
            stage_bar.set_postfix_str('schema checked')
            stage_bar.update(1)

        paths = G8Paths(output_root, dataset_name)
        write_json(paths.pred_main_path, pred_main)
        write_json(paths.pred_diag_path, pred_diag)
        row_manifest_path = output_root / 'predictions' / dataset_name / 'row_manifest.jsonl'
        write_jsonl(row_manifest_path, pack['row_manifest'])
        if stage_bar is not None:
            stage_bar.set_postfix_str('artifacts written')
            stage_bar.update(1)

        return {
            'status': 'PASS',
            'selected_for_infer': resolution.selected_for_infer,
            'checkpoint_path': str(resolution.checkpoint_path),
            'pred_main_path': str(paths.pred_main_path),
            'pred_diag_path': str(paths.pred_diag_path),
            'row_manifest_path': str(row_manifest_path),
            'dataset_name': dataset_name,
            'seed': int(seed),
            'logit_chunk_size': int(logit_chunk_size),
            'smoke': bool(smoke),
            'scored_row_count': len(scored_rows),
            'asset_counts': asset_counts,
            'skipped_trajectory_histogram': skipped,
            'vocab_size': int(len(text_vocab_ids)),
            'carrier_matrix_shape': [int(x) for x in pack['carrier_matrix'].shape],
            'score_matrix_shape': [int(x) for x in scores['fused_logits'].shape],
        }
    finally:
        if stage_bar is not None:
            stage_bar.close()


def collect_gt_attribution_rank(*, output_root: Path, dataset_name: str, device: str, metrics_profile: str, selected_for_infer: str, show_progress: bool = True) -> Dict[str, Any]:
    stage = 'all' if str(metrics_profile) == 'formal' else _selected_audit_stage(selected_for_infer)
    bar = tqdm(total=1, desc=f'gt_attribution_rank[{stage}]', unit='stage', leave=True) if show_progress else None
    try:
        payload = run_gt_attribution_rank_audit(
            GTAttributionRankAuditConfig(
                dataset_name=dataset_name,
                output_root=output_root,
                stage=stage,
                device=torch.device(device),
                all_gt_only=True,
                all_gt_generate_sidecars_if_missing=False,
            )
        )
        if bar is not None:
            bar.update(1)
        return payload
    finally:
        if bar is not None:
            bar.close()


def collect_external_eval(*, output_root: Path, exp_name: str, seed: int, smoke: bool, show_progress: bool = True) -> Dict[str, Any]:
    bar = tqdm(total=1, desc='external_eval', unit='stage', leave=True) if show_progress else None
    try:
        payload = run_external_lvvis_eval(ExternalLVVISEvalConfig(exp_name=exp_name, output_root=output_root, seed=seed, smoke=smoke))
        if bar is not None:
            bar.update(1)
        return payload
    finally:
        if bar is not None:
            bar.close()
