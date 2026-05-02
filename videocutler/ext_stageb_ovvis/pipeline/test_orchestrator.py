from __future__ import annotations
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict
from tqdm.auto import tqdm
from videocutler.ext_stageb_ovvis.metrics.collector import collect_external_eval, collect_gt_attribution_rank, collect_runtime_metrics, run_inference
from videocutler.ext_stageb_ovvis.pipeline.plans import TestPlan

REPO_ASSET_LINK_NAMES = ('exports', 'exports_gt', 'carrier_bank', 'carrier_bank_gt', 'frame_bank', 'text_bank', 'gt_sidecar_bank', 'weak_labels', 'weights', 'dataset', 'eval')


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _bootstrap_asset_links(target_root: Path, asset_root: Path) -> None:
    if not asset_root.is_dir():
        return
    target_root.mkdir(parents=True, exist_ok=True)
    for name in REPO_ASSET_LINK_NAMES:
        src = asset_root / name
        dst = target_root / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            try:
                _safe_link(src, dst)
            except Exception:
                pass


def run_test_pipeline(plan: TestPlan) -> Dict[str, Any]:
    _bootstrap_asset_links(Path(plan.repo_root), Path(plan.asset_root))
    _bootstrap_asset_links(Path(plan.output_root), Path(plan.asset_root))
    stage_bar = tqdm(total=3, desc='test stages', unit='stage', leave=True)
    try:
        with _pushd(plan.repo_root):
            stage_bar.set_postfix_str('inference')
            inference_summary = run_inference(
                output_root=plan.output_root,
                dataset_name=plan.dataset_name,
                device=plan.device,
                seed=plan.seed,
                logit_chunk_size=plan.logit_chunk_size,
                smoke=plan.smoke,
                ckpt_path=plan.ckpt_path,
                show_progress=True,
            )
            runtime_metrics = collect_runtime_metrics(plan.output_root)
            stage_bar.update(1)

            stage_bar.set_postfix_str('gt attribution rank')
            gt_attr = collect_gt_attribution_rank(
                output_root=plan.output_root,
                dataset_name=plan.dataset_name,
                device=plan.device,
                metrics_profile=plan.metrics_profile,
                selected_for_infer=str(inference_summary.get('selected_for_infer', 'augmented')),
                ckpt_path=str(plan.ckpt_path) if plan.ckpt_path is not None else None,
                show_progress=True,
            )
            stage_bar.update(1)

            stage_bar.set_postfix_str('external eval')
            external = collect_external_eval(
                output_root=plan.output_root,
                exp_name=plan.exp_name,
                seed=plan.seed,
                smoke=plan.smoke,
                show_progress=True,
            )
            stage_bar.update(1)

        payload = {
            'exp_name': plan.exp_name,
            'pipeline': plan.pipeline,
            'stage_scope': plan.stage_scope,
            'dataset_name': plan.dataset_name,
            'benchmark': plan.benchmark,
            'metrics_profile': plan.metrics_profile,
            'repo_root': str(plan.repo_root),
            'asset_root': str(plan.asset_root),
            'inference': inference_summary,
            'train': runtime_metrics,
            'gt_attribution_rank': gt_attr,
            'external_eval': external,
        }
        path = plan.output_root / 'final_summary.json'
        _write_json(path, payload)
        return {'status': 'PASS', 'summary_path': str(path), 'summary': payload}
    finally:
        stage_bar.close()
