from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from videocutler.ext_stageb_ovvis.metrics.collector import collect_external_eval, collect_gt_attribution_rank, collect_runtime_metrics, run_inference
from videocutler.ext_stageb_ovvis.pipeline.plans import TestPlan

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def run_test_pipeline(plan: TestPlan) -> Dict[str, Any]:
    inference_summary=run_inference(output_root=plan.output_root, dataset_name=plan.dataset_name, device=plan.device, seed=plan.seed, logit_chunk_size=plan.logit_chunk_size, smoke=plan.smoke, ckpt_path=plan.ckpt_path); runtime_metrics=collect_runtime_metrics(plan.output_root); gt_attr=collect_gt_attribution_rank(output_root=plan.output_root, dataset_name=plan.dataset_name, device=plan.device); external=collect_external_eval(output_root=plan.output_root, exp_name=plan.exp_name, seed=plan.seed, smoke=plan.smoke); payload={'exp_name':plan.exp_name,'pipeline':plan.pipeline,'stage_scope':plan.stage_scope,'dataset_name':plan.dataset_name,'benchmark':plan.benchmark,'inference':inference_summary,'train':runtime_metrics,'gt_attribution_rank':gt_attr,'external_eval':external}; path=plan.output_root/'final_summary.json'; _write_json(path,payload); return {'status':'PASS','summary_path':str(path),'summary':payload}
