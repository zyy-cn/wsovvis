from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import GTAttributionRankAuditConfig, run_gt_attribution_rank_audit
from videocutler.ext_stageb_ovvis.eval.external_lvvis import ExternalLVVISEvalConfig, run_external_lvvis_eval
from videocutler.ext_stageb_ovvis.eval.g8_bridge import G8Paths, build_pred_rows, load_projector_bundle, load_text_vocab_with_names, load_video_meta, resolve_inference_asset_roots, resolve_selected_for_infer, score_infer_row, validate_json_artifact, write_json

def _read_json(path: Path) -> Dict[str, Any]: return json.loads(path.read_text(encoding='utf-8'))
def collect_runtime_metrics(output_root: Path) -> Dict[str, Any]:
    path=output_root/'train'/'pipeline_train_summary.json'
    return dict(_read_json(path).get('stages', {})) if path.is_file() else {}
def run_inference(*, output_root: Path, dataset_name: str, device: str, seed: int, logit_chunk_size: int, smoke: bool, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
    resolution=resolve_selected_for_infer(output_root, ckpt_path=ckpt_path); asset_roots=resolve_inference_asset_roots(output_root, dataset_name=dataset_name, trajectory_source_branch='mainline', resolution=resolution); device_obj=torch.device(device); bundle=load_projector_bundle(resolution.checkpoint_path, device=device_obj); text_vocab_ids,_text_records,text_matrix,class_name_map=load_text_vocab_with_names(asset_roots.asset_root, dataset_name); video_meta=load_video_meta(dataset_name); from videocutler.ext_stageb_ovvis.eval.g8_bridge import build_infer_rows; infer_rows, skipped, asset_counts = build_infer_rows(asset_roots, dataset_name=dataset_name); infer_rows = infer_rows[:min(8,len(infer_rows))] if smoke else infer_rows; scored_rows=[score_infer_row(row, bundle=bundle, asset_root=asset_roots.asset_root, dataset_name=dataset_name, trajectory_source_branch='mainline', text_vocab_ids=text_vocab_ids, text_matrix=text_matrix, class_name_map=class_name_map, logit_chunk_size=logit_chunk_size) for row in infer_rows]; pred_main,pred_diag=build_pred_rows(scored_rows, video_meta=video_meta); validate_json_artifact(pred_main,'pred_main.schema.json'); validate_json_artifact(pred_diag,'pred_diag.schema.json'); paths=G8Paths(output_root, dataset_name); write_json(paths.pred_main_path, pred_main); write_json(paths.pred_diag_path, pred_diag); return {'status':'PASS','selected_for_infer':resolution.selected_for_infer,'checkpoint_path':str(resolution.checkpoint_path),'pred_main_path':str(paths.pred_main_path),'pred_diag_path':str(paths.pred_diag_path),'dataset_name':dataset_name,'seed':int(seed),'logit_chunk_size':int(logit_chunk_size),'smoke':bool(smoke),'scored_row_count':len(scored_rows),'asset_counts':asset_counts,'skipped_trajectory_histogram':skipped}
def collect_gt_attribution_rank(*, output_root: Path, dataset_name: str, device: str) -> Dict[str, Any]:
    return run_gt_attribution_rank_audit(GTAttributionRankAuditConfig(dataset_name=dataset_name, output_root=output_root, stage='all', device=torch.device(device), all_gt_only=True))
def collect_external_eval(*, output_root: Path, exp_name: str, seed: int, smoke: bool) -> Dict[str, Any]:
    return run_external_lvvis_eval(ExternalLVVISEvalConfig(exp_name=exp_name, output_root=output_root, seed=seed, smoke=smoke))
