from __future__ import annotations
import json, random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm as _tqdm_cls
except Exception:
    _tqdm_cls = None

def _maybe_tqdm(iterable, *, enabled: bool, **kwargs):
    if enabled and _tqdm_cls is not None:
        return _tqdm_cls(iterable, **kwargs)
    return iterable
from videocutler.ext_stageb_ovvis.algorithms._training_budget import build_dynamic_microbatches, resolve_default_batch_budget
from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab, score_carrier_logits_torch, observed_mass_loss
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples
from videocutler.ext_stageb_ovvis.algorithms.soft_em import _prepare_examples as _prepare_softem_examples, _build_runtime_extra_cache, _apply_runtime_extra_cache
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.ext_stageb_ovvis.utils.unknown_metrics import UnknownMetricsAccumulator
try:
    from videocutler.ext_stageb_ovvis.training.residual_peeling_schedule import (
        ResidualPeelingSchedule,
        apply_residual_candidate_override_to_groups,
        load_oracle_static_residual_schedule,
    )
except Exception:  # keeps legacy imports robust when overlay helper is absent
    ResidualPeelingSchedule = Any  # type: ignore
    apply_residual_candidate_override_to_groups = None  # type: ignore
    load_oracle_static_residual_schedule = None  # type: ignore
Record=Dict[str,Any]
@dataclass(frozen=True)
class ReservoirPrealignConfig:
    dataset_name:str; trajectory_source_branch:str='mainline'; device:str='cpu'; seed:int=0; smoke:bool=False; lambda_frame:float=0.25; epochs:int=1; learning_rate:float=1e-4; weight_decay:float=1e-2; t_dis_init:float=0.07; projector:ProjectorConfig=ProjectorConfig(); runtime_asset_source:str='local_canonical_assets'; runtime_asset_source_local_incomplete:bool=False; runtime_asset_output_root:str=''; batch_budget:int|None=None; show_progress:bool=True; log_every:int=10; write_runtime_metrics_jsonl:bool=True; print_epoch_summary:bool=True
@dataclass(frozen=True)
class ReservoirSoftEMConfig:
    dataset_name:str; trajectory_source_branch:str='mainline'; mode:str='base_then_aug'; device:str='cpu'; seed:int=0; smoke:bool=False; lambda_frame:float=0.25; t_dis_init:float=0.07; weight_decay:float=1e-2; projector:ProjectorConfig=ProjectorConfig(); base_epochs:int=1; aug_epochs:int=1; base_learning_rate:float=5e-5; aug_learning_rate:float=5e-5; runtime_asset_source:str='local_canonical_assets'; runtime_asset_source_local_incomplete:bool=False; runtime_asset_output_root:str=''; batch_budget:int|None=None; k_extra:int=2; extra_alpha:float=0.25; base_release_margin:float=0.0; ablate_skip_base:bool=False; ablate_no_yprime_reward:bool=False; show_progress:bool=True; log_every:int=10; write_runtime_metrics_jsonl:bool=True; print_epoch_summary:bool=True

def _set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def _inverse_softplus(value:float)->float:
    target=max(float(value),1e-6); return float(np.log(np.expm1(target)))
def _compute_t_dis(theta_t:torch.nn.Parameter)->torch.Tensor:
    return F.softplus(theta_t)+1e-4
def _write_json(path:Path,payload:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def _append_jsonl(path:Path,row:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as h: h.write(json.dumps(dict(row), ensure_ascii=False)+'\n')
def _write_jsonl(path:Path,rows:Iterable[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as h:
        for row in rows: h.write(json.dumps(dict(row), ensure_ascii=False)+'\n')
def _normalize_np(vec:np.ndarray)->np.ndarray:
    arr=np.asarray(vec,dtype=np.float32); denom=float(np.linalg.norm(arr)); return arr if denom<=1e-12 else (arr/denom).astype(np.float32)
def _mean_or_zero(values:Sequence[float])->float:
    return float(np.mean(np.asarray(list(values), dtype=np.float32))) if values else 0.0
def _project_text_matrix(projector:Projector,matrix:np.ndarray,device:torch.device)->torch.Tensor:
    return projector(torch.from_numpy(np.asarray(matrix,dtype=np.float32)).to(device=device, dtype=torch.float32))
def _unknown_score(z:torch.Tensor,u_unknown:torch.nn.Parameter,temperature:torch.Tensor)->torch.Tensor:
    return torch.matmul(F.normalize(z,p=2.0,dim=-1), F.normalize(u_unknown.reshape(1,-1),p=2.0,dim=-1).t()).squeeze(-1)/temperature
def _clip_groups(examples:Sequence[Mapping[str,Any]])->List[List[Mapping[str,Any]]]:
    by_clip:Dict[int,List[Mapping[str,Any]]]={}
    for ex in examples: by_clip.setdefault(int(ex['clip_id']), []).append(ex)
    return [by_clip[k] for k in sorted(by_clip.keys())]
def train_reservoir_prealign(*, output_root:Path, materialized_samples:Sequence[Record], config:ReservoirPrealignConfig)->Dict[str,Any]:
    _set_seed(int(config.seed)); device=torch.device(str(config.device)); prepared=_prepare_prealign_examples(materialized_samples, output_root=output_root, dataset_name=config.dataset_name, trajectory_source_branch=config.trajectory_source_branch); examples=list(prepared['examples']); skipped=dict(prepared['skipped_reason_histogram']);
    if not examples: raise RuntimeError('no trainable examples for reservoir prealign')
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(output_root); projector=Projector(config.projector).to(device); projector.train(); theta_t=torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(config.t_dis_init)-1e-4,1e-6)), device=device, dtype=torch.float32)); unknown_prototype=torch.nn.Parameter(F.normalize(torch.randn(int(config.projector.output_dim), device=device, dtype=torch.float32), p=2.0, dim=0)); optimizer=torch.optim.AdamW([*projector.parameters(), theta_t, unknown_prototype], lr=float(config.learning_rate), weight_decay=float(config.weight_decay)); batch_budget=resolve_default_batch_budget(smoke=bool(config.smoke), explicit=config.batch_budget); groups=_clip_groups(examples); plan=build_dynamic_microbatches(groups, batch_budget=batch_budget, cost_fn=lambda grp: len(text_vocab_ids), bucket_key_fn=lambda grp: (0, len(text_vocab_ids))); runtime_metrics_path=output_root/'train'/'prealign'/'runtime_metrics.jsonl'; losses=[]; batch_losses=[]; unknown_metrics=UnknownMetricsAccumulator(device=device); global_step=0
    for epoch_index in _maybe_tqdm(
        range(int(config.epochs)),
        enabled=bool(config.show_progress),
        desc='prealign epochs',
        leave=True,
    ):
        shuffled_groups=list(groups); random.Random(int(config.seed)+int(epoch_index)).shuffle(shuffled_groups); epoch_plan=build_dynamic_microbatches(shuffled_groups, batch_budget=batch_budget, cost_fn=lambda grp: len(text_vocab_ids), bucket_key_fn=lambda grp: (0, len(text_vocab_ids))); epoch_losses=[]; epoch_batch_losses=[]
        for micro_idx, batch_indices in enumerate(
            _maybe_tqdm(
                epoch_plan.batches,
                enabled=bool(config.show_progress),
                desc=f'prealign epoch {int(epoch_index)+1}',
                total=len(epoch_plan.batches),
                leave=False,
            ),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True); batch_loss_accum=None; clip_count=0
            for batch_index in batch_indices:
                clip_examples=shuffled_groups[int(batch_index)]; z_clip_np=_normalize_np(np.mean(np.stack([np.asarray(ex['carrier_vec'], dtype=np.float32) for ex in clip_examples], axis=0), axis=0)); z_clip=torch.from_numpy(z_clip_np).to(device=device, dtype=torch.float32).unsqueeze(0); temperature=_compute_t_dis(theta_t); text_proj=_project_text_matrix(projector, np.asarray(text_vocab_matrix, dtype=np.float32), device=device); logits=torch.matmul(F.normalize(z_clip,p=2.0,dim=-1), text_proj.t()).squeeze(0)/temperature; positive=[idx for idx, raw_id in enumerate(text_vocab_ids) if int(raw_id) in {int(x) for x in clip_examples[0]['observed_raw_ids']}]
                if not positive: continue
                unknown_logit=_unknown_score(z_clip, unknown_prototype, temperature).reshape(()); sample_loss=observed_mass_loss(logits, positive, unknown_logit=unknown_logit); batch_loss_accum=sample_loss if batch_loss_accum is None else (batch_loss_accum+sample_loss); clip_count+=1; val=float(sample_loss.detach().cpu().item()); losses.append(val); epoch_losses.append(val); unknown_metrics.update_prealign(len(clip_examples))
            if batch_loss_accum is None or clip_count<=0: continue
            batch_loss=batch_loss_accum/float(clip_count); batch_loss.backward(); optimizer.step(); global_step+=1; batch_val=float(batch_loss.detach().cpu().item()); batch_losses.append(batch_val); epoch_batch_losses.append(batch_val)
            if bool(config.write_runtime_metrics_jsonl): _append_jsonl(runtime_metrics_path, {'row_type':'microbatch','timestamp':datetime.now(timezone.utc).isoformat(),'stage':'prealign','epoch':int(epoch_index)+1,'microbatch_idx':int(micro_idx),'microbatch_total':int(epoch_plan.batch_count),'loss':batch_val,'optimization_loss':batch_val,'effective_trajectory_count':int(sum(len(shuffled_groups[int(idx)]) for idx in batch_indices)),'unknown_mass_mean':1.0,'release_gate_mean':0.0})
        if bool(config.write_runtime_metrics_jsonl): _append_jsonl(runtime_metrics_path, {'row_type':'epoch_summary','timestamp':datetime.now(timezone.utc).isoformat(),'stage':'prealign','epoch':int(epoch_index)+1,'microbatch_count':int(len(epoch_batch_losses)),'loss_mean':_mean_or_zero(epoch_losses),'loss_last':float(epoch_losses[-1]) if epoch_losses else 0.0,'optimization_loss_mean':_mean_or_zero(epoch_batch_losses),'optimization_loss_last':float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,'unknown_mass_mean_epoch':1.0,'release_gate_mean_epoch':0.0})
    train_dir=output_root/'train'/'prealign'; ckpt_dir=train_dir/'checkpoints'; ckpt_dir.mkdir(parents=True, exist_ok=True); proxy_path=train_dir/'proxy_records.jsonl'; train_state_path=train_dir/'train_state.json'; ckpt_last_path=ckpt_dir/'prealign_last.pth'; proxy_rows=[{'dataset_name':str(config.dataset_name),'clip_id':int(ex['clip_id']),'video_id':int(ex['video_id']),'trajectory_id':str(ex['trajectory_id']),'observed_raw_ids':[int(x) for x in ex['observed_raw_ids']],'proxy_mass':{'unknown':1.0},'join_key':str(ex['trajectory_id'])} for ex in sorted(examples, key=lambda row: str(row['trajectory_id']))]
    torch.save({'stage_id':'prealign','epoch':int(config.epochs),'text_projector_state_dict':projector.state_dict(),'text_projector_config':{'input_dim':int(config.projector.input_dim),'hidden_dim':int(config.projector.hidden_dim),'output_dim':int(config.projector.output_dim),'dropout':float(config.projector.dropout),'use_layernorm':bool(config.projector.use_layernorm)},'theta_T':float(theta_t.detach().cpu().item()),'b_u':0.0,'unknown_prototype':F.normalize(unknown_prototype.detach().cpu(), p=2.0, dim=0),'seed':int(config.seed),'global_step':int(global_step),'pipeline':'reservoir_v1'}, ckpt_last_path); _write_jsonl(proxy_path, proxy_rows); train_state={'stage_id':'prealign','epoch':int(config.epochs),'selected_for_infer':'prealign_only','selected_for_infer_authority':'explicit_train_state_field','checkpoint_last':'train/prealign/checkpoints/prealign_last.pth','checkpoint_selected':'train/prealign/checkpoints/prealign_last.pth','global_step':int(global_step),'runtime_asset_source':str(config.runtime_asset_source),'runtime_asset_source_local_incomplete':bool(config.runtime_asset_source_local_incomplete),'runtime_asset_output_root':str(config.runtime_asset_output_root),'pipeline':'reservoir_v1'}; _write_json(train_state_path, train_state); unknown_summary=unknown_metrics.finalize(distributed=False); _write_json(train_dir/'stage_summary.json', {'stage_id':'prealign','pipeline':'reservoir_v1','loss_mean':_mean_or_zero(losses),'loss_last':float(losses[-1]) if losses else 0.0,'optimization_loss_mean':_mean_or_zero(batch_losses),'optimization_loss_last':float(batch_losses[-1]) if batch_losses else 0.0,'unknown_metrics':unknown_summary});
    return {'proxy_records_path':proxy_path,'train_state_path':train_state_path,'checkpoint_last_path':ckpt_last_path,'record_count_input':int(len(materialized_samples)),'record_count_trainable':int(len(examples)),'record_count_output':int(len(proxy_rows)),'coverage_ratio_trainable':float(len(examples)/max(len(materialized_samples),1)),'skipped_reason_histogram':skipped,'loss_mean':_mean_or_zero(losses),'loss_last':float(losses[-1]) if losses else 0.0,'optimization_loss_mean':_mean_or_zero(batch_losses),'optimization_loss_last':float(batch_losses[-1]) if batch_losses else 0.0,'batch_budget':int(batch_budget),'micro_batch_count_per_epoch':int(plan.batch_count),'budget_policy':'dynamic_sum_Tv_times_Kv','loss_normalization':'clip_count','train_state':train_state,'unknown_metrics':unknown_summary,'stage_summary_path':train_dir/'stage_summary.json'}
def _load_reservoir_checkpoint(checkpoint_path:Path, *, device:torch.device)->Tuple[Projector,torch.nn.Parameter,torch.nn.Parameter,Dict[str,Any]]:
    checkpoint=torch.load(checkpoint_path, map_location=device); cfg=dict(checkpoint.get('text_projector_config', {})); projector=Projector(ProjectorConfig(input_dim=int(cfg.get('input_dim',512)), hidden_dim=int(cfg.get('hidden_dim',1024)), output_dim=int(cfg.get('output_dim',768)), dropout=float(cfg.get('dropout',0.0)), use_layernorm=bool(cfg.get('use_layernorm',True)))).to(device); projector.load_state_dict(checkpoint['text_projector_state_dict']); theta_t=torch.nn.Parameter(torch.tensor(float(checkpoint.get('theta_T', _inverse_softplus(0.07))), device=device, dtype=torch.float32)); unknown_payload=checkpoint.get('unknown_prototype'); unknown_payload=F.normalize(torch.randn(int(projector.config.output_dim), device=device, dtype=torch.float32), p=2.0, dim=0) if unknown_payload is None else torch.as_tensor(unknown_payload, device=device, dtype=torch.float32); unknown_prototype=torch.nn.Parameter(F.normalize(unknown_payload.reshape(-1), p=2.0, dim=0)); return projector,theta_t,unknown_prototype,dict(checkpoint)
def _stage_output(stage_id:str)->Tuple[str,str,str,str]:
    return ('base_only','softem_base_last.pth','train/softem_base/responsibility_records.jsonl','train/softem_base/train_state.json') if stage_id=='softem_base' else ('augmented','softem_aug_last.pth','train/softem_aug/responsibility_records.jsonl','train/softem_aug/train_state.json')

def _stage_domain_ids(ex: Mapping[str, Any], *, is_aug: bool, ablate_no_yprime_reward: bool) -> Tuple[List[int], List[int], List[int]]:
    known_ids = [int(x) for x in ex['candidate_ids_known']]
    # Corrected E2 keeps the full aug domain; only the previous extra-id drop was wrong.
    extra_ids = [int(x) for x in ex['candidate_ids_extra']]
    domain_ids = list(known_ids) + list(extra_ids)
    return domain_ids, known_ids, extra_ids

def run_reservoir_soft_em(*, output_root:Path, materialized_samples:Sequence[Record], config:ReservoirSoftEMConfig)->Dict[str,Any]:
    _set_seed(int(config.seed)); device=torch.device(str(config.device)); prepared=_prepare_softem_examples(materialized_samples, output_root=output_root, dataset_name=config.dataset_name, trajectory_source_branch=config.trajectory_source_branch); examples=list(prepared['examples']); skipped=dict(prepared['skipped_reason_histogram']);
    if not examples: raise RuntimeError('no trainable examples for reservoir softem')
    projector, theta_t, unknown_prototype, _ckpt = _load_reservoir_checkpoint(output_root/'train'/'prealign'/'checkpoints'/'prealign_last.pth', device=device); projector.train(); optimizer=torch.optim.AdamW([*projector.parameters(), theta_t, unknown_prototype], lr=float(config.base_learning_rate), weight_decay=float(config.weight_decay)); batch_budget=resolve_default_batch_budget(smoke=bool(config.smoke), explicit=config.batch_budget); runtime_root=output_root/'train'; stage_reports=[]; current_unknown_metrics={}; final_examples=examples
    stage_ids = ['softem_aug'] if bool(config.ablate_skip_base) else (['softem_base'] if str(config.mode)=='base_only' else ['softem_base','softem_aug'])
    for stage_id in _maybe_tqdm(
        stage_ids,
        enabled=bool(config.show_progress),
        desc='softem stages',
        leave=True,
    ):
        is_aug=stage_id=='softem_aug'; learning_rate=float(config.aug_learning_rate if is_aug else config.base_learning_rate); epochs=int(config.aug_epochs if is_aug else config.base_epochs); [group.update({'lr':learning_rate}) for group in optimizer.param_groups]; stage_examples=[]; staged_domain_counts=[]
        stage_source_examples=list(examples)
        runtime_extra_cache={}
        if is_aug and not bool(config.ablate_skip_base):
            runtime_extra_cache=_build_runtime_extra_cache(
                examples=examples,
                text_projector=projector,
                theta_t=theta_t,
                output_root=output_root,
                k_extra=int(config.k_extra),
                alpha=float(config.extra_alpha),
                lambda_frame=float(config.lambda_frame),
                device=device,
            )
            if runtime_extra_cache:
                stage_source_examples=_apply_runtime_extra_cache(
                    examples,
                    runtime_extra_cache=runtime_extra_cache,
                    output_root=output_root,
                )
                if bool(config.write_runtime_metrics_jsonl):
                    nonempty_count=int(sum(1 for row in stage_source_examples if len(list(row.get('candidate_ids_extra', []))) > 0))
                    _append_jsonl(
                        runtime_root/'softem_aug'/'runtime_metrics.jsonl',
                        {
                            'row_type':'runtime_extra_cache_summary',
                            'timestamp':datetime.now(timezone.utc).isoformat(),
                            'stage':'softem_aug',
                            'clip_cache_count':int(len(runtime_extra_cache)),
                            'example_count':int(len(stage_source_examples)),
                            'example_with_nonempty_extra_count':nonempty_count,
                            'example_with_nonempty_extra_rate':float(nonempty_count/ max(len(stage_source_examples), 1)),
                            'k_extra':int(config.k_extra),
                            'extra_alpha':float(config.extra_alpha),
                        },
                    )
        for ex in stage_source_examples:
            domain_ids, known_ids, extra_ids = _stage_domain_ids(ex, is_aug=is_aug, ablate_no_yprime_reward=bool(config.ablate_no_yprime_reward))
            if len(domain_ids) <= 0:
                continue
            stage_examples.append((ex, domain_ids, known_ids, extra_ids))
            staged_domain_counts.append(len(domain_ids))
        final_examples=[item[0] for item in stage_examples]; plan=build_dynamic_microbatches(stage_examples, batch_budget=batch_budget, cost_fn=lambda item:max(len(item[1]),1), bucket_key_fn=lambda item:(1, len(item[1]))); runtime_metrics_path=runtime_root/stage_id/'runtime_metrics.jsonl'; unknown_metrics=UnknownMetricsAccumulator(device=device); losses=[]; batch_losses=[]; global_step=0
        for epoch_index in _maybe_tqdm(
            range(epochs),
            enabled=bool(config.show_progress),
            desc=f'{stage_id} epochs',
            leave=True,
        ):
            shuffled_examples=list(stage_examples); random.Random(int(config.seed)+epoch_index).shuffle(shuffled_examples); epoch_plan=build_dynamic_microbatches(shuffled_examples, batch_budget=batch_budget, cost_fn=lambda item:max(len(item[1]),1), bucket_key_fn=lambda item:(1, len(item[1]))); epoch_losses=[]; epoch_batch_losses=[]; epoch_g=[]
            for micro_idx, batch_indices in enumerate(
                _maybe_tqdm(
                    epoch_plan.batches,
                    enabled=bool(config.show_progress),
                    desc=f'{stage_id} epoch {int(epoch_index)+1}',
                    total=len(epoch_plan.batches),
                    leave=False,
                ),
                start=1,
            ):
                optimizer.zero_grad(set_to_none=True); batch_loss_accum=None; eff=0; batch_g=[]
                for batch_index in batch_indices:
                    ex, domain_ids, known_ids, extra_ids = shuffled_examples[int(batch_index)]; domain_count=len(domain_ids); candidate_matrix=np.asarray(ex['candidate_matrix'][:domain_count], dtype=np.float32); temperature=_compute_t_dis(theta_t); logits=score_carrier_logits_torch(projector=projector, carrier_vec=ex['carrier_vec'], candidate_matrix=candidate_matrix, temperature=temperature); z=torch.from_numpy(_normalize_np(np.asarray(ex['carrier_vec'], dtype=np.float32))).to(device=device, dtype=torch.float32).unsqueeze(0); s_u=_unknown_score(z, unknown_prototype, temperature).reshape(()); s_star=torch.max(logits); g_i=torch.softmax(torch.stack([s_star - torch.as_tensor(float(config.base_release_margin), device=device, dtype=torch.float32), s_u]), dim=0)[0]; q=torch.softmax(logits, dim=0); target=torch.cat([(1.0-g_i).reshape(1), g_i.reshape(1)*q], dim=0); stage_logits=torch.cat([s_u.reshape(1), logits], dim=0); sample_loss=-(target*torch.log_softmax(stage_logits, dim=0)).sum(); batch_loss_accum=sample_loss if batch_loss_accum is None else (batch_loss_accum+sample_loss); eff+=1; val=float(sample_loss.detach().cpu().item()); losses.append(val); epoch_losses.append(val); gv=float(g_i.detach().cpu().item()); batch_g.append(gv); epoch_g.append(gv); 
                    if (not is_aug) or bool(config.ablate_skip_base):
                        unknown_metrics.update_base(g_i.reshape(1))
                if batch_loss_accum is None or eff<=0: continue
                batch_loss=batch_loss_accum/float(eff); batch_loss.backward(); optimizer.step(); global_step+=1; batch_val=float(batch_loss.detach().cpu().item()); batch_losses.append(batch_val); epoch_batch_losses.append(batch_val)
                if bool(config.write_runtime_metrics_jsonl): _append_jsonl(runtime_metrics_path, {'row_type':'microbatch','timestamp':datetime.now(timezone.utc).isoformat(),'stage':stage_id,'epoch':int(epoch_index)+1,'microbatch_idx':int(micro_idx),'microbatch_total':int(epoch_plan.batch_count),'loss':_mean_or_zero(epoch_losses[-eff:]),'optimization_loss':batch_val,'effective_trajectory_count':int(eff),'release_gate_mean':_mean_or_zero(batch_g),'unknown_mean_responsibility':float(1.0-_mean_or_zero(batch_g))})
            if bool(config.write_runtime_metrics_jsonl): _append_jsonl(runtime_metrics_path, {'row_type':'epoch_summary','timestamp':datetime.now(timezone.utc).isoformat(),'stage':stage_id,'epoch':int(epoch_index)+1,'microbatch_count':int(len(epoch_batch_losses)),'loss_mean':_mean_or_zero(epoch_losses),'loss_last':float(epoch_losses[-1]) if epoch_losses else 0.0,'optimization_loss_mean':_mean_or_zero(epoch_batch_losses),'optimization_loss_last':float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,'release_gate_mean_epoch':_mean_or_zero(epoch_g),'unknown_mean_responsibility_epoch':float(1.0-_mean_or_zero(epoch_g))})
        selected_for_infer,ckpt_name,resp_rel,train_state_rel=_stage_output(stage_id); stage_dir=output_root/'train'/stage_id; ckpt_dir=stage_dir/'checkpoints'; ckpt_dir.mkdir(parents=True, exist_ok=True); checkpoint_path=ckpt_dir/ckpt_name; rows=[]
        for ex, domain_ids, known_ids, extra_ids in stage_examples:
            candidate_matrix=np.asarray(ex['candidate_matrix'][:len(domain_ids)], dtype=np.float32); temperature=_compute_t_dis(theta_t)
            with torch.no_grad(): logits=score_carrier_logits_torch(projector=projector, carrier_vec=ex['carrier_vec'], candidate_matrix=candidate_matrix, temperature=temperature); z=torch.from_numpy(_normalize_np(np.asarray(ex['carrier_vec'], dtype=np.float32))).to(device=device, dtype=torch.float32).unsqueeze(0); s_u=_unknown_score(z, unknown_prototype, temperature).reshape(()); s_star=torch.max(logits); g_i=torch.softmax(torch.stack([s_star - torch.as_tensor(float(config.base_release_margin), device=device, dtype=torch.float32), s_u]), dim=0)[0]; q=torch.softmax(logits, dim=0); r_final={'unknown':float((1.0-g_i).detach().cpu().item())};
            for raw_id,prob in zip(domain_ids,(g_i*q).detach().cpu().numpy().astype(np.float64).tolist()): r_final[str(int(raw_id))]=float(prob)
            rows.append({'dataset_name':str(config.dataset_name),'clip_id':int(ex['clip_id']),'video_id':int(ex['video_id']),'trajectory_id':str(ex['trajectory_id']),'candidate_ids_known':list(known_ids),'candidate_ids_extra':list(extra_ids),'unknown_slot':'unknown','r_init':{'unknown':1.0},'r_final':r_final,'join_key':str(ex['trajectory_id'])})
        _write_jsonl(output_root/resp_rel, rows); torch.save({'stage_id':stage_id,'epoch':int(epochs),'text_projector_state_dict':projector.state_dict(),'text_projector_config':{'input_dim':int(config.projector.input_dim),'hidden_dim':int(config.projector.hidden_dim),'output_dim':int(config.projector.output_dim),'dropout':float(config.projector.dropout),'use_layernorm':bool(config.projector.use_layernorm)},'theta_T':float(theta_t.detach().cpu().item()),'b_u':0.0,'unknown_prototype':F.normalize(unknown_prototype.detach().cpu(), p=2.0, dim=0),'seed':int(config.seed),'mode':str(config.mode),'global_step':int(global_step),'pipeline':'reservoir_v1'}, checkpoint_path); train_state={'stage_id':stage_id,'epoch':int(epochs),'selected_for_infer':selected_for_infer,'selected_for_infer_authority':'explicit_train_state_field','checkpoint_last':str((Path('train')/stage_id/'checkpoints'/ckpt_name).as_posix()),'checkpoint_selected':str((Path('train')/stage_id/'checkpoints'/ckpt_name).as_posix()),'global_step':int(global_step),'runtime_asset_source':str(config.runtime_asset_source),'runtime_asset_source_local_incomplete':bool(config.runtime_asset_source_local_incomplete),'runtime_asset_output_root':str(config.runtime_asset_output_root),'pipeline':'reservoir_v1'}; _write_json(output_root/train_state_rel, train_state); stage_summary={'stage_id':stage_id,'pipeline':'reservoir_v1','loss_mean':_mean_or_zero(losses),'loss_last':float(losses[-1]) if losses else 0.0,'optimization_loss_mean':_mean_or_zero(batch_losses),'optimization_loss_last':float(batch_losses[-1]) if batch_losses else 0.0};
        if (not is_aug) or bool(config.ablate_skip_base):
            current_unknown_metrics=unknown_metrics.finalize(distributed=False)
            stage_summary['unknown_metrics']=current_unknown_metrics
        if is_aug and not bool(config.ablate_skip_base):
            stage_summary['runtime_extra_cache_metrics']={
                'clip_cache_count':int(len(runtime_extra_cache)),
                'example_count':int(len(stage_source_examples)),
                'example_with_nonempty_extra_count':int(sum(1 for row in stage_source_examples if len(list(row.get('candidate_ids_extra', []))) > 0)),
                'example_with_nonempty_extra_rate':float(sum(1 for row in stage_source_examples if len(list(row.get('candidate_ids_extra', []))) > 0)/max(len(stage_source_examples),1)),
                'k_extra':int(config.k_extra),
                'extra_alpha':float(config.extra_alpha),
            }
        _write_json(stage_dir/'stage_summary.json', stage_summary); stage_reports.append({'stage_id':stage_id,'responsibility_records_path':resp_rel,'train_state_path':train_state_rel,'checkpoint_last_path':str((Path('train')/stage_id/'checkpoints'/ckpt_name).as_posix()),'record_count_output':int(len(rows)),'loss_mean':_mean_or_zero(losses),'loss_last':float(losses[-1]) if losses else 0.0,'optimization_loss_mean':_mean_or_zero(batch_losses),'optimization_loss_last':float(batch_losses[-1]) if batch_losses else 0.0,'batch_budget':int(batch_budget),'loss_normalization':'effective_trajectory_count'})
    return {'stage_reports':stage_reports,'record_count_input':int(len(materialized_samples)),'record_count_trainable':int(len(examples)),'record_count_output':int(len(final_examples)),'coverage_ratio_trainable':float(len(examples)/max(len(materialized_samples),1)),'skipped_reason_histogram':skipped,'selected_checkpoint_path':stage_reports[-1]['checkpoint_last_path'] if stage_reports else '','unknown_metrics':current_unknown_metrics}

# ---------------------------------------------------------------------------
# Optional Sinkhorn/no-unknown experimental branch.
# This branch is intentionally additive and is not called by the legacy or
# reservoir_v1 default paths. It does not alter the semantics of the existing
# prealign/softem functions above.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservoirSinkhornNoUnknownConfig:
    dataset_name: str
    trajectory_source_branch: str = 'mainline'
    device: str = 'cpu'
    seed: int = 0
    smoke: bool = False
    t_dis_init: float = 0.07
    weight_decay: float = 1e-2
    projector: ProjectorConfig = ProjectorConfig()
    prealign_epochs: int = 1
    aug_epochs: int = 1
    prealign_learning_rate: float = 1e-4
    aug_learning_rate: float = 5e-5
    runtime_asset_source: str = 'local_canonical_assets'
    runtime_asset_source_local_incomplete: bool = False
    runtime_asset_output_root: str = ''
    batch_budget: int | None = None
    k_extra: int = 2
    extra_alpha: float = 0.25
    lambda_frame: float = 0.25
    sinkhorn_tau: float = 0.15
    sinkhorn_iters: int = 5
    sinkhorn_row_cap_scale: float = 2.0
    sinkhorn_extra_demand: float = 0.25
    sinkhorn_aug_extra_lambda: float = 0.2
    sinkhorn_assignment_stopgrad: bool = True
    sinkhorn_safe_negatives: bool = False
    sinkhorn_safe_neg_count: int = 64
    sinkhorn_safe_neg_weight: float = 0.25
    sinkhorn_safe_neg_text_sim_threshold: float = 0.50
    sinkhorn_safe_neg_exclude_model_topk: int = 100
    sinkhorn_safe_neg_seed: int = 3407
    sinkhorn_extra_margin_gate: float | None = None
    sinkhorn_final_rerank_lambda_r: float = 0.0
    sinkhorn_vocab_scope_policy: str = 'weak_label_only'
    sinkhorn_vocab_scope_strict_check: bool = True
    train_vocab_raw_ids: Tuple[int, ...] = ()
    train_vocab_source: str = 'weak_labels_train_union'
    show_progress: bool = True
    log_every: int = 10
    write_runtime_metrics_jsonl: bool = True
    print_epoch_summary: bool = True
    # Support-null branch defaults are off to preserve existing sinkhorn behavior.
    sinkhorn_enable_null_column: bool = False
    sinkhorn_null_logit_bias: float = 0.0
    sinkhorn_null_residual: bool = False
    sinkhorn_null_demand_cap_ratio: float = 1.0
    sinkhorn_support_warmup_epochs: int = 0
    sinkhorn_yprime_demand_mode: str = 'fixed'
    sinkhorn_yprime_demand_min: float = 0.10
    sinkhorn_yprime_support_topk: int = 2
    sinkhorn_yprime_support_temp: float = 0.25
    sinkhorn_yprime_support_ema: float = 0.90
    sinkhorn_null_collapse_max: float = 0.85
    sinkhorn_yprime_demand_min_guard: float = 0.20
    # V2-C positive-support protection defaults are off for compatibility.
    sinkhorn_enable_positive_protection: bool = False
    sinkhorn_positive_margin_threshold: float = 0.15
    sinkhorn_positive_margin_temp: float = 0.10
    sinkhorn_positive_null_cap: float = 0.40
    sinkhorn_positive_redistribute_mode: str = 'best_y'
    residual_peeling_mode: str = 'off'
    residual_schedule_csv: str = ''
    residual_variant: str = 'person_aware'
    residual_annotation_json: str = ''
    residual_split_json: str = ''
    residual_round_epoch_plan: str = '5,5,3,2'
    residual_candidate_policy: str = 'base_residual'
    known_null_gate: str = 'off'
    known_null_margin: float = 0.10


def _sinkhorn_observed_ids(group: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted({int(x) for ex in group for x in list(ex.get('observed_raw_ids', []))})


def _sinkhorn_known_extra_ids(group: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
    known = sorted({int(x) for ex in group for x in list(ex.get('candidate_ids_known', []))})
    extra = sorted({int(x) for ex in group for x in list(ex.get('candidate_ids_extra', [])) if int(x) not in set(known)})
    return known, extra



def _resolve_sinkhorn_train_vocab(
    *,
    policy: str,
    train_vocab_raw_ids: Sequence[int],
    text_vocab_ids: Sequence[int],
    strict_check: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    policy = str(policy or 'weak_label_only')
    text_ids = [int(x) for x in text_vocab_ids]
    text_set = set(text_ids)
    if policy == 'legacy_full':
        allowed = sorted(text_set)
        return allowed, {
            'policy': 'legacy_full',
            'status': 'RETIRED_LEGACY_FULL_SCOPE',
            'train_vocab_source': 'full_text_vocab_retired',
            'allowed_train_vocab_count': int(len(allowed)),
            'full_text_vocab_count': int(len(text_ids)),
            'strict_check': bool(strict_check),
        }
    if policy != 'weak_label_only':
        raise ValueError(f'unsupported sinkhorn_vocab_scope_policy: {policy!r}')
    allowed = sorted({int(x) for x in train_vocab_raw_ids})
    missing = [raw_id for raw_id in allowed if raw_id not in text_set]
    if not allowed:
        raise ValueError('weak_label_only policy requires non-empty train_vocab_raw_ids from weak_labels_train.json')
    if missing:
        raise KeyError(f'weak_label_only train vocab ids missing from text bank: {missing[:16]}')
    return allowed, {
        'policy': 'weak_label_only',
        'status': 'PASS',
        'train_vocab_source': 'weak_labels_train_union',
        'allowed_train_vocab_count': int(len(allowed)),
        'full_text_vocab_count': int(len(text_ids)),
        'outside_train_vocab_count': int(len(text_set - set(allowed))),
        'strict_check': bool(strict_check),
    }


def _scope_text_vocab(
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    allowed_raw_ids: Sequence[int],
) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
    raw_to_full_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    scoped_ids = [int(raw_id) for raw_id in allowed_raw_ids if int(raw_id) in raw_to_full_idx]
    scoped_indices = [raw_to_full_idx[int(raw_id)] for raw_id in scoped_ids]
    scoped_matrix = np.asarray(text_vocab_matrix, dtype=np.float32)[scoped_indices]
    return scoped_ids, scoped_matrix, {
        'scoped_text_vocab_count': int(len(scoped_ids)),
        'full_text_vocab_count': int(len(text_vocab_ids)),
        'dropped_text_vocab_count': int(len(text_vocab_ids) - len(scoped_ids)),
    }

def _sinkhorn_group_cost_pre(group: Sequence[Mapping[str, Any]]) -> int:
    return max(1, len(group) * max(1, len(_sinkhorn_observed_ids(group))))


def _sinkhorn_group_cost_aug(group: Sequence[Mapping[str, Any]]) -> int:
    known, extra = _sinkhorn_known_extra_ids(group)
    return max(1, len(group) * max(1, len(known) + len(extra)))


def _sinkhorn_pack_groups(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    raw_to_vocab_idx: Mapping[int, int],
    device: torch.device,
    mode: str,
    extra_demand: float,
) -> Dict[str, Any]:
    packed_groups: List[Sequence[Mapping[str, Any]]] = []
    label_raws_by_group: List[List[int]] = []
    label_kind_by_group: List[List[int]] = []  # 1=Y'/known, 2=extra
    for group in groups:
        if str(mode) == 'prealign':
            known = [raw_id for raw_id in _sinkhorn_observed_ids(group) if int(raw_id) in raw_to_vocab_idx]
            extra: List[int] = []
        else:
            known_raw, extra_raw = _sinkhorn_known_extra_ids(group)
            known = [raw_id for raw_id in known_raw if int(raw_id) in raw_to_vocab_idx]
            extra = [raw_id for raw_id in extra_raw if int(raw_id) in raw_to_vocab_idx and int(raw_id) not in set(known)]
        labels = list(known) + list(extra)
        if not labels:
            continue
        packed_groups.append(group)
        label_raws_by_group.append(labels)
        label_kind_by_group.append([1] * len(known) + [2] * len(extra))
    if not packed_groups:
        return {}
    B = len(packed_groups)
    Qmax = max(len(group) for group in packed_groups)
    Mmax = max(len(labels) for labels in label_raws_by_group)
    D = int(np.asarray(packed_groups[0][0]['carrier_vec'], dtype=np.float32).reshape(-1).shape[0])
    Z = torch.zeros((B, Qmax, D), device=device, dtype=torch.float32)
    q_mask = torch.zeros((B, Qmax), device=device, dtype=torch.bool)
    yidx = torch.zeros((B, Mmax), device=device, dtype=torch.long)
    c_mask = torch.zeros((B, Mmax), device=device, dtype=torch.bool)
    demand = torch.zeros((B, Mmax), device=device, dtype=torch.float32)
    kind = torch.zeros((B, Mmax), device=device, dtype=torch.long)
    raw_ids_tensor = torch.zeros((B, Mmax), device=device, dtype=torch.long)
    for b, group in enumerate(packed_groups):
        for q, ex in enumerate(group):
            vec = _normalize_np(np.asarray(ex['carrier_vec'], dtype=np.float32))
            Z[b, q] = torch.from_numpy(vec).to(device=device, dtype=torch.float32)
            q_mask[b, q] = True
        for m, raw_id in enumerate(label_raws_by_group[b]):
            yidx[b, m] = int(raw_to_vocab_idx[int(raw_id)])
            c_mask[b, m] = True
            raw_ids_tensor[b, m] = int(raw_id)
            label_kind = int(label_kind_by_group[b][m])
            kind[b, m] = label_kind
            demand[b, m] = 1.0 if label_kind == 1 else max(0.0, float(extra_demand))
    residual_meta_by_group = []
    for group in packed_groups:
        first = group[0] if group else {}
        residual_meta_by_group.append({
            'round_id': int(first.get('residual_peeling_round_id', -1)),
            'known_raw_ids': [int(x) for x in list(first.get('residual_known_raw_ids', []))],
            'candidate_raw_ids': [int(x) for x in list(first.get('residual_candidate_raw_ids', []))],
            'candidate_policy': str(first.get('residual_candidate_policy', '')),
        })
    return {
        'groups': packed_groups,
        'Z': Z,
        'q_mask': q_mask,
        'yidx': yidx,
        'c_mask': c_mask,
        'demand': demand,
        'kind': kind,
        'raw_ids': raw_ids_tensor,
        'residual_meta_by_group': residual_meta_by_group,
    }


def _sinkhorn_runtime_extra_cache_metrics(runtime_extra_cache: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    clip_cache_count = int(len(runtime_extra_cache))
    example_with_nonempty_extra_count = int(sum(1 for row in runtime_extra_cache.values() if len(list(row.get('candidate_ids_extra', []))) > 0))
    candidate_count_before_gate = int(sum(int(row.get('candidate_ids_extra_pre_gate_count', len(list(row.get('candidate_ids_extra', []))))) for row in runtime_extra_cache.values()))
    candidate_count_after_gate = int(sum(int(row.get('candidate_ids_extra_retained_count', len(list(row.get('candidate_ids_extra', []))))) for row in runtime_extra_cache.values()))
    return {
        'clip_cache_count': clip_cache_count,
        'example_with_nonempty_extra_count': example_with_nonempty_extra_count,
        'example_with_nonempty_extra_rate': float(example_with_nonempty_extra_count / max(clip_cache_count, 1)),
        'candidate_count_before_gate': candidate_count_before_gate,
        'candidate_count_after_gate': candidate_count_after_gate,
        'candidate_retained_rate': float(candidate_count_after_gate / max(candidate_count_before_gate, 1)),
        'sinkhorn_extra_margin_gate': next((row.get('sinkhorn_extra_margin_gate') for row in runtime_extra_cache.values() if row.get('sinkhorn_extra_margin_gate') is not None), None),
    }


def _sinkhorn_scope_contract_fields(
    vocab_scope_policy: Mapping[str, Any],
    *,
    runtime_extra_cache: Optional[Mapping[int, Mapping[str, Any]]] = None,
    aug_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    allowed_train_vocab_count = int(vocab_scope_policy.get('allowed_train_vocab_count', 0))
    full_text_vocab_count = int(vocab_scope_policy.get('full_text_vocab_count', 0))
    extra_outside_weak_count = int(
        sum(int(row.get('candidate_ids_extra_outside_scope_count', 0)) for row in (runtime_extra_cache or {}).values())
    )
    responsibility_candidate_outside_weak_count = int(
        sum(int(row.get('candidate_outside_train_vocab_count', 0)) for row in (aug_rows or []))
    )
    return {
        'vocab_scope_policy': str(vocab_scope_policy.get('policy', 'unknown')),
        'weak_vocab_count': allowed_train_vocab_count,
        'full_text_vocab_count': full_text_vocab_count,
        'extra_scope': 'weak_union',
        'safe_neg_scope': 'weak_union',
        'model_topk_scope': 'weak_union',
        'extra_outside_weak_count': extra_outside_weak_count,
        'safe_neg_outside_weak_count': 0,
        'model_topk_outside_weak_count': 0,
        'denominator_outside_weak_count': 0,
        'responsibility_candidate_outside_weak_count': responsibility_candidate_outside_weak_count,
    }


def _sinkhorn_scores_from_pack(projector: Projector, text_vocab_tensor: torch.Tensor, pack: Mapping[str, Any], temperature: torch.Tensor) -> torch.Tensor:
    text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
    anchors = text_proj_all[pack['yidx']]
    Z = F.normalize(pack['Z'], p=2.0, dim=-1)
    return torch.bmm(Z, anchors.transpose(1, 2)) / temperature


def _sinkhorn_candidate_and_full_scores_from_pack(projector: Projector, text_vocab_tensor: torch.Tensor, pack: Mapping[str, Any], temperature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
    Z = F.normalize(pack['Z'], p=2.0, dim=-1)
    full_scores = torch.matmul(Z, text_proj_all.transpose(0, 1)) / temperature
    gather_idx = pack['yidx'][:, None, :].expand(-1, full_scores.shape[1], -1)
    scores = full_scores.gather(dim=2, index=gather_idx)
    return scores, full_scores





def _apply_residual_known_null_gate_to_pack(
    pack: Mapping[str, Any],
    scores: torch.Tensor,
    full_scores: torch.Tensor,
    *,
    raw_to_vocab_idx: Mapping[int, int],
    margin: float,
    enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Mask rows that are clearly better explained by already-known classes.

    This is not row-level GT supervision. It uses only the offline class schedule's
    K_{t-1} and current model scores. Masked rows are excluded from the residual
    assignment loss so known/person-like trajectories are not forced into residual
    labels.
    """
    metrics: Dict[str, Any] = {
        'residual_known_null_gate_enabled': bool(enabled),
        'residual_known_null_routed_rate': 0.0,
        'residual_known_null_routed_count': 0,
        'residual_known_null_eligible_count': 0,
        'residual_known_score_mean': 0.0,
        'residual_best_score_mean': 0.0,
    }
    if not bool(enabled):
        return dict(pack), metrics
    if full_scores is None:
        return dict(pack), metrics
    q_mask = pack['q_mask'].bool().clone()
    c_mask = pack['c_mask'].bool()
    B, Q = q_mask.shape
    routed_total = 0
    eligible_total = 0
    known_vals: List[float] = []
    residual_vals: List[float] = []
    for b in range(B):
        meta = (pack.get('residual_meta_by_group') or [{}])[b]
        known_raw = [int(x) for x in list(meta.get('known_raw_ids', [])) if int(x) in raw_to_vocab_idx]
        if not known_raw:
            continue
        known_idx = torch.tensor([int(raw_to_vocab_idx[int(x)]) for x in known_raw], device=full_scores.device, dtype=torch.long)
        known_score = full_scores[b, :, known_idx].max(dim=1).values
        residual_score = scores[b].masked_fill(~c_mask[b][None, :], -1.0e4).max(dim=1).values
        eligible = q_mask[b] & torch.isfinite(known_score) & (residual_score > -1.0e3)
        routed = eligible & (known_score > (residual_score + float(margin)))
        q_mask[b, routed] = False
        eligible_total += int(eligible.detach().sum().cpu().item())
        routed_total += int(routed.detach().sum().cpu().item())
        if bool(eligible.any()):
            known_vals.append(float(known_score[eligible].detach().float().mean().cpu().item()))
            residual_vals.append(float(residual_score[eligible].detach().float().mean().cpu().item()))
    pack2 = dict(pack)
    pack2['q_mask'] = q_mask
    metrics.update({
        'residual_known_null_routed_count': int(routed_total),
        'residual_known_null_eligible_count': int(eligible_total),
        'residual_known_null_routed_rate': float(routed_total / max(eligible_total, 1)),
        'residual_known_score_mean': _mean_or_zero(known_vals),
        'residual_best_score_mean': _mean_or_zero(residual_vals),
    })
    return pack2, metrics

def _apply_support_null_to_pack(
    pack: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    epoch_index: int,
    support_state: Dict[Tuple[int, int], float],
    enable_null_column: bool,
    null_logit_bias: float,
    null_residual: bool,
    null_demand_cap_ratio: float,
    yprime_demand_mode: str,
    yprime_demand_min: float,
    yprime_support_topk: int,
    yprime_support_temp: float,
    yprime_support_ema: float,
) -> Tuple[Dict[str, Any], torch.Tensor, Dict[str, Any]]:
    """Add a non-semantic NULL/dustbin column and optional support-aware Y-prime demand."""
    if not bool(enable_null_column):
        return dict(pack), scores, {
            'support_null_enabled': False,
            'support_null_active': False,
            'null_mass_mean': 0.0,
            'nonnull_mass_mean': 0.0,
            'null_demand_mean': 0.0,
            'yprime_demand_mean': 1.0,
            'yprime_low_demand_rate': 0.0,
            'support_epoch_index': int(epoch_index),
        }
    B, Q, M = scores.shape
    device = scores.device
    c_mask = pack['c_mask'].clone()
    demand = pack['demand'].clone().float()
    kind = pack['kind'].clone()
    yidx = pack['yidx'].clone()
    raw_ids = pack['raw_ids'].clone()
    q_mask = pack['q_mask'].bool()
    yprime_mask = c_mask.bool() & (kind == 1)

    if str(yprime_demand_mode) in ('support_ema', 'relative_margin_ema'):
        with torch.no_grad():
            new_demand = demand.clone()
            min_d = float(max(0.0, min(1.0, yprime_demand_min)))
            temp = float(max(1.0e-6, yprime_support_temp))
            topk = max(1, int(yprime_support_topk))
            ema = float(max(0.0, min(0.999, yprime_support_ema)))
            for b in range(B):
                q_valid = q_mask[b]
                for m in torch.nonzero(yprime_mask[b], as_tuple=False).reshape(-1).tolist():
                    vals = scores[b, q_valid, int(m)].detach().float()
                    if vals.numel() <= 0:
                        conf = min_d
                    else:
                        k = min(int(topk), int(vals.numel()))
                        if str(yprime_demand_mode) == 'relative_margin_ema':
                            # Use relative evidence, not absolute score. A Y' class
                            # only gets high demand when some trajectories prefer it
                            # over competing Y' columns and the fixed NULL bias. This
                            # prevents generic/hub trajectories from lifting every
                            # Y' demand to ~1.0.
                            comp_mask = yprime_mask[b].clone()
                            comp_mask[int(m)] = False
                            if bool(comp_mask.any()):
                                comp = scores[b, q_valid][:, comp_mask].detach().float().amax(dim=1)
                                comp = torch.maximum(comp, torch.full_like(comp, float(null_logit_bias)))
                            else:
                                comp = torch.full_like(vals, float(null_logit_bias))
                            margin_vals = vals - comp
                            support_score = margin_vals.topk(k=k).values.mean()
                        else:
                            support_score = vals.topk(k=k).values.mean()
                        conf = float(torch.sigmoid(support_score / temp).detach().cpu().item())
                        conf = max(min_d, min(1.0, conf))
                    group0 = pack['groups'][b][0]
                    key = (int(group0.get('clip_id', group0.get('video_id', b))), int(raw_ids[b, int(m)].detach().cpu().item()))
                    prev = support_state.get(key, conf)
                    smoothed = ema * float(prev) + (1.0 - ema) * float(conf)
                    support_state[key] = smoothed
                    new_demand[b, int(m)] = float(max(min_d, min(1.0, smoothed)))
            demand = new_demand

    null_scores = torch.full((B, Q, 1), float(null_logit_bias), device=device, dtype=scores.dtype)
    scores2 = torch.cat([scores, null_scores], dim=2)
    c_mask2 = torch.cat([c_mask, torch.ones((B, 1), device=device, dtype=torch.bool)], dim=1)
    kind2 = torch.cat([kind, torch.zeros((B, 1), device=device, dtype=kind.dtype)], dim=1)
    yidx2 = torch.cat([yidx, torch.zeros((B, 1), device=device, dtype=yidx.dtype)], dim=1)
    raw_ids2 = torch.cat([raw_ids, torch.full((B, 1), -1, device=device, dtype=raw_ids.dtype)], dim=1)
    q_count = q_mask.float().sum(dim=1).clamp_min(1.0)
    y_demand_sum = demand.masked_fill(~yprime_mask, 0.0).sum(dim=1)
    if bool(null_residual):
        null_residual_uncapped = (q_count - y_demand_sum).clamp_min(1.0e-6)
    else:
        null_residual_uncapped = torch.ones_like(q_count).clamp_min(1.0e-6)
    cap_ratio = float(max(0.0, null_demand_cap_ratio))
    if cap_ratio > 0.0 and cap_ratio < 1.0e6:
        null_cap = (q_count * cap_ratio).clamp_min(1.0e-6)
        null_demand = torch.minimum(null_residual_uncapped, null_cap)
    else:
        null_cap = torch.full_like(q_count, float('inf'))
        null_demand = null_residual_uncapped
    demand2 = torch.cat([demand, null_demand[:, None]], dim=1)
    pack2 = dict(pack)
    pack2.update({'c_mask': c_mask2, 'kind': kind2, 'yidx': yidx2, 'raw_ids': raw_ids2, 'demand': demand2})
    with torch.no_grad():
        y_d = demand[yprime_mask]
        metrics = {
            'support_null_enabled': True,
            'support_null_active': True,
            'support_epoch_index': int(epoch_index),
            'null_demand_mean': float(null_demand.detach().float().mean().cpu().item()),
            'null_residual_uncapped_mean': float(null_residual_uncapped.detach().float().mean().cpu().item()),
            'null_cap_mean': float(torch.where(torch.isfinite(null_cap), null_cap, torch.zeros_like(null_cap)).detach().float().mean().cpu().item()),
            'null_demand_cap_ratio': float(null_demand_cap_ratio),
            'yprime_demand_mean': float(y_d.detach().float().mean().cpu().item()) if y_d.numel() else 0.0,
            'yprime_demand_min_observed': float(y_d.detach().float().min().cpu().item()) if y_d.numel() else 0.0,
            'yprime_demand_max_observed': float(y_d.detach().float().max().cpu().item()) if y_d.numel() else 0.0,
            'yprime_low_demand_rate': float((y_d.detach().float() <= (float(yprime_demand_min) + 1.0e-6)).float().mean().cpu().item()) if y_d.numel() else 0.0,
        }
    return pack2, scores2, metrics



def _apply_positive_protection_to_assignment(
    P: torch.Tensor,
    scores: torch.Tensor,
    pack: Mapping[str, Any],
    *,
    enable_positive_protection: bool,
    margin_threshold: float,
    margin_temp: float,
    positive_null_cap: float,
    redistribute_mode: str = 'best_y',
    null_logit_bias: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Limit NULL mass for high relative-margin rows.

    This is a non-GT proxy protection: if a trajectory strongly prefers one
    Y' column over competing Y' columns and the fixed NULL bias, keep a minimum
    amount of row mass in non-NULL assignment. NULL remains available for low
    confidence/background rows.
    """
    metrics: Dict[str, Any] = {
        'positive_protection_enabled': bool(enable_positive_protection),
        'positive_protected_row_rate': 0.0,
        'positive_strength_mean': 0.0,
        'positive_margin_mean': 0.0,
        'positive_margin_p50': 0.0,
        'positive_margin_p90': 0.0,
        'positive_null_mass_before_mean': 0.0,
        'positive_null_mass_after_mean': 0.0,
        'positive_null_excess_moved_mean': 0.0,
        'positive_redistributed_mass_mean': 0.0,
        'positive_null_cap': float(positive_null_cap),
    }
    if not bool(enable_positive_protection):
        return P, metrics
    if scores.ndim != 3 or P.ndim != 3 or tuple(scores.shape) != tuple(P.shape):
        return P, metrics
    q_mask = pack['q_mask'].bool()
    c_mask = pack['c_mask'].bool()
    kind = pack['kind']
    yprime_mask = c_mask & (kind == 1)
    null_mask = c_mask & (kind == 0)
    if not bool(yprime_mask.any()) or not bool(null_mask.any()):
        return P, metrics

    eps = 1.0e-8
    B, Q, M = scores.shape
    masked_y = scores.float().masked_fill(~yprime_mask[:, None, :], -1.0e4)
    best_vals, best_idx = masked_y.max(dim=2)  # [B,Q]
    comp = masked_y.clone()
    comp.scatter_(2, best_idx[:, :, None], -1.0e4)
    second_vals = comp.max(dim=2).values
    second_vals = torch.maximum(second_vals, torch.full_like(second_vals, float(null_logit_bias)))
    margin = (best_vals - second_vals).masked_fill(~q_mask, 0.0)
    valid_best = q_mask & (best_vals > -1.0e3)
    temp = max(float(margin_temp), 1.0e-6)
    strength = torch.sigmoid((margin - float(margin_threshold)) / temp).masked_fill(~valid_best, 0.0)

    cap = float(max(0.0, min(1.0, positive_null_cap)))
    row_load = P.float().sum(dim=2).clamp_min(eps)
    null_mass_before = P.float().masked_fill(~null_mask[:, None, :], 0.0).sum(dim=2)
    # Low strength rows have cap ~1.0; high strength rows approach positive_null_cap.
    row_null_cap_fraction = 1.0 - strength * (1.0 - cap)
    max_null_mass = row_load * row_null_cap_fraction
    excess = (null_mass_before - max_null_mass).clamp_min(0.0).masked_fill(~valid_best, 0.0)
    if float(excess.detach().sum().cpu().item()) <= 0.0:
        with torch.no_grad():
            valid_m = margin[valid_best]
            if valid_m.numel():
                metrics.update({
                    'positive_protected_row_rate': float((strength[valid_best] > 0.5).float().mean().cpu().item()),
                    'positive_strength_mean': float(strength[valid_best].float().mean().cpu().item()),
                    'positive_margin_mean': float(valid_m.float().mean().cpu().item()),
                    'positive_margin_p50': float(torch.quantile(valid_m.float(), 0.5).cpu().item()),
                    'positive_margin_p90': float(torch.quantile(valid_m.float(), 0.9).cpu().item()),
                    'positive_null_mass_before_mean': float((null_mass_before[valid_best] / row_load[valid_best]).float().mean().cpu().item()),
                    'positive_null_mass_after_mean': float((null_mass_before[valid_best] / row_load[valid_best]).float().mean().cpu().item()),
                })
        return P, metrics

    P2 = P.clone()
    null_mass_safe = null_mass_before.clamp_min(eps)
    null_share = P2.float().masked_fill(~null_mask[:, None, :], 0.0) / null_mass_safe[:, :, None]
    null_delta = null_share * excess[:, :, None]
    P2 = P2 - null_delta.to(dtype=P2.dtype)
    add = torch.zeros_like(P2)
    # First version intentionally redistributes to best_y only for a clear audit surface.
    add.scatter_add_(2, best_idx[:, :, None], excess[:, :, None].to(dtype=add.dtype))
    P2 = P2 + add
    P2 = P2.clamp_min(0.0)
    null_mass_after = P2.float().masked_fill(~null_mask[:, None, :], 0.0).sum(dim=2)

    with torch.no_grad():
        valid_m = margin[valid_best]
        protected = valid_best & (strength > 0.5)
        metrics.update({
            'positive_protection_enabled': True,
            'positive_protected_row_rate': float(protected.float().sum().cpu().item() / max(float(q_mask.float().sum().cpu().item()), 1.0)),
            'positive_strength_mean': float(strength[valid_best].float().mean().cpu().item()) if bool(valid_best.any()) else 0.0,
            'positive_margin_mean': float(valid_m.float().mean().cpu().item()) if valid_m.numel() else 0.0,
            'positive_margin_p50': float(torch.quantile(valid_m.float(), 0.5).cpu().item()) if valid_m.numel() else 0.0,
            'positive_margin_p90': float(torch.quantile(valid_m.float(), 0.9).cpu().item()) if valid_m.numel() else 0.0,
            'positive_null_mass_before_mean': float((null_mass_before[valid_best] / row_load[valid_best]).float().mean().cpu().item()) if bool(valid_best.any()) else 0.0,
            'positive_null_mass_after_mean': float((null_mass_after[valid_best] / row_load[valid_best]).float().mean().cpu().item()) if bool(valid_best.any()) else 0.0,
            'positive_null_excess_moved_mean': float((excess[valid_best] / row_load[valid_best]).float().mean().cpu().item()) if bool(valid_best.any()) else 0.0,
            'positive_redistributed_mass_mean': float(excess[valid_best].float().mean().cpu().item()) if bool(valid_best.any()) else 0.0,
            'positive_null_cap': float(cap),
            'positive_redistribute_mode_best_y': 1.0 if str(redistribute_mode) == 'best_y' else 0.0,
        })
    return P2, metrics

def _sinkhorn_train_stage(
    *,
    stage_id: str,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    output_root: Path,
    projector: Projector,
    theta_t: torch.nn.Parameter,
    text_vocab_tensor: torch.Tensor,
    raw_to_vocab_idx: Mapping[int, int],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    learning_rate: float,
    batch_budget: int,
    seed: int,
    mode: str,
    sinkhorn_tau: float,
    sinkhorn_iters: int,
    sinkhorn_row_cap_scale: float,
    extra_demand: float,
    extra_lambda: float,
    assignment_stopgrad: bool,
    safe_negatives: bool,
    safe_neg_count: int,
    safe_neg_weight: float,
    safe_neg_text_sim_threshold: float,
    safe_neg_exclude_model_topk: int,
    safe_neg_seed: int,
    sinkhorn_final_rerank_lambda_r: float,
    allowed_vocab_mask: torch.Tensor | None,
    vocab_scope_policy: Mapping[str, Any],
    show_progress: bool,
    write_runtime_metrics_jsonl: bool,
    enable_null_column: bool = False,
    null_logit_bias: float = 0.0,
    null_residual: bool = False,
    null_demand_cap_ratio: float = 1.0,
    support_warmup_epochs: int = 0,
    yprime_demand_mode: str = 'fixed',
    yprime_demand_min: float = 0.10,
    yprime_support_topk: int = 2,
    yprime_support_temp: float = 0.25,
    yprime_support_ema: float = 0.90,
    null_collapse_max: float = 0.85,
    yprime_demand_min_guard: float = 0.20,
    enable_positive_protection: bool = False,
    positive_margin_threshold: float = 0.15,
    positive_margin_temp: float = 0.10,
    positive_null_cap: float = 0.40,
    positive_redistribute_mode: str = 'best_y',
    residual_schedule: Optional[Any] = None,
    residual_candidate_policy: str = 'base_residual',
    known_null_gate: str = 'off',
    known_null_margin: float = 0.10,
) -> Dict[str, Any]:
    from videocutler.ext_stageb_ovvis.algorithms.sinkhorn_assignment import (
        SinkhornAssignmentConfig,
        assignment_metrics,
        capped_sinkhorn_assignment,
        sinkhorn_loss_from_assignment,
        yprime_only_nce_loss_from_assignment,
        yprime_nce_with_safe_negatives_loss_from_assignment,
    )

    for group in optimizer.param_groups:
        group['lr'] = float(learning_rate)
    runtime_metrics_path = output_root / 'train' / stage_id / 'runtime_metrics.jsonl'
    losses: List[float] = []
    batch_losses: List[float] = []
    global_step = 0
    cost_fn = _sinkhorn_group_cost_pre if str(mode) == 'prealign' else _sinkhorn_group_cost_aug
    bucket_fn = lambda grp: (len(grp), max(1, len(_sinkhorn_observed_ids(grp)) if str(mode) == 'prealign' else sum(len(x) for x in _sinkhorn_known_extra_ids(grp))))
    cfg = SinkhornAssignmentConfig(tau=float(sinkhorn_tau), iters=int(sinkhorn_iters), row_cap_scale=float(sinkhorn_row_cap_scale))
    raw_text_norm = F.normalize(text_vocab_tensor.detach().float(), p=2.0, dim=-1)
    raw_text_cos_all = torch.matmul(raw_text_norm, raw_text_norm.transpose(0, 1))
    support_state: Dict[Tuple[int, int], float] = {}
    support_null_epoch_metrics: List[Dict[str, Any]] = []
    for epoch_index in _maybe_tqdm(range(int(epochs)), enabled=bool(show_progress), desc=f'{stage_id} epochs', leave=True):
        shuffled_groups = list(groups)
        random.Random(int(seed) + int(epoch_index)).shuffle(shuffled_groups)
        epoch_plan = build_dynamic_microbatches(shuffled_groups, batch_budget=int(batch_budget), cost_fn=cost_fn, bucket_key_fn=bucket_fn)
        epoch_losses: List[float] = []
        epoch_batch_losses: List[float] = []
        epoch_metric_rows: List[Dict[str, Any]] = []
        for micro_idx, batch_indices in enumerate(_maybe_tqdm(epoch_plan.batches, enabled=bool(show_progress), desc=f'{stage_id} epoch {int(epoch_index)+1}', total=len(epoch_plan.batches), leave=False), start=1):
            selected_groups = [shuffled_groups[int(i)] for i in batch_indices]
            residual_metric_row: Dict[str, Any] = {'residual_peeling_enabled': False}
            if residual_schedule is not None:
                selected_groups, residual_metric_row = apply_residual_candidate_override_to_groups(
                    selected_groups,
                    schedule=residual_schedule,
                    epoch_index=int(epoch_index),
                    candidate_policy=str(residual_candidate_policy),
                )
                if not selected_groups:
                    continue
            pack = _sinkhorn_pack_groups(selected_groups, raw_to_vocab_idx=raw_to_vocab_idx, device=text_vocab_tensor.device, mode=str(mode), extra_demand=float(extra_demand))
            if not pack:
                continue
            optimizer.zero_grad(set_to_none=True)
            temperature = _compute_t_dis(theta_t)
            need_full_scores_for_residual_gate = bool(residual_schedule is not None and str(known_null_gate) == 'margin')
            if bool(safe_negatives) or need_full_scores_for_residual_gate:
                scores, full_scores = _sinkhorn_candidate_and_full_scores_from_pack(projector, text_vocab_tensor, pack, temperature)
            else:
                scores = _sinkhorn_scores_from_pack(projector, text_vocab_tensor, pack, temperature)
                full_scores = None
            residual_gate_metric_row: Dict[str, Any] = {'residual_known_null_gate_enabled': bool(need_full_scores_for_residual_gate)}
            if need_full_scores_for_residual_gate:
                pack, residual_gate_metric_row = _apply_residual_known_null_gate_to_pack(
                    pack, scores, full_scores, raw_to_vocab_idx=raw_to_vocab_idx, margin=float(known_null_margin), enabled=True
                )
            support_null_active = bool(enable_null_column) and str(mode) == 'prealign' and (int(epoch_index) >= int(max(0, support_warmup_epochs)))
            pack, scores, support_metric_row = _apply_support_null_to_pack(
                pack,
                scores,
                epoch_index=int(epoch_index) + 1,
                support_state=support_state,
                enable_null_column=support_null_active,
                null_logit_bias=float(null_logit_bias),
                null_residual=bool(null_residual),
                null_demand_cap_ratio=float(null_demand_cap_ratio),
                yprime_demand_mode=str(yprime_demand_mode),
                yprime_demand_min=float(yprime_demand_min),
                yprime_support_topk=int(yprime_support_topk),
                yprime_support_temp=float(yprime_support_temp),
                yprime_support_ema=float(yprime_support_ema),
            )
            P = capped_sinkhorn_assignment(scores, pack['q_mask'], pack['c_mask'], pack['demand'], config=cfg)
            P, positive_metric_row = _apply_positive_protection_to_assignment(
                P,
                scores,
                pack,
                enable_positive_protection=bool(enable_positive_protection) and bool(support_null_active),
                margin_threshold=float(positive_margin_threshold),
                margin_temp=float(positive_margin_temp),
                positive_null_cap=float(positive_null_cap),
                redistribute_mode=str(positive_redistribute_mode),
                null_logit_bias=float(null_logit_bias),
            )
            support_metric_row.update(positive_metric_row)
            support_metric_row.update(residual_metric_row)
            support_metric_row.update(residual_gate_metric_row)
            with torch.no_grad():
                null_mask = pack['c_mask'].bool() & (pack['kind'] == 0)
                nonnull_mask = pack['c_mask'].bool() & (pack['kind'] > 0)
                null_mass = P.masked_fill(~null_mask[:, None, :], 0.0).sum(dim=(1, 2)) if bool(null_mask.any()) else torch.zeros((P.shape[0],), device=P.device)
                nonnull_mass = P.masked_fill(~nonnull_mask[:, None, :], 0.0).sum(dim=(1, 2))
                total_mass = (null_mass + nonnull_mass).clamp_min(1.0e-6)
                null_fraction = null_mass / total_mass
                support_metric_row.update({
                    'null_mass_mean': float(null_fraction.detach().float().mean().cpu().item()),
                    'nonnull_mass_mean': float((nonnull_mass / total_mass).detach().float().mean().cpu().item()),
                    'null_collapse_guard_triggered': bool(float(null_fraction.detach().float().mean().cpu().item()) > float(null_collapse_max)),
                    'support_demand_guard_triggered': bool(float(support_metric_row.get('yprime_demand_mean', 1.0)) < float(yprime_demand_min_guard)) if bool(support_metric_row.get('support_null_active', False)) else False,
                })
                support_null_epoch_metrics.append(dict(support_metric_row))
            # V3 loss: optional filtered full-vocabulary safe negatives add weak
            # rank pressure to Y'-only soft-label NCE while avoiding naive
            # full-vocab false negatives. With safe negatives disabled, behavior
            # is identical to V2 Y'-only NCE.
            if bool(safe_negatives):
                loss, safe_metric_row = yprime_nce_with_safe_negatives_loss_from_assignment(
                    scores,
                    P,
                    pack['kind'],
                    pack['c_mask'],
                    full_scores=full_scores,
                    yidx=pack['yidx'],
                    raw_text_cos_all=raw_text_cos_all,
                    safe_neg_count=int(safe_neg_count),
                    safe_neg_weight=float(safe_neg_weight),
                    text_sim_exclude_threshold=float(safe_neg_text_sim_threshold),
                    exclude_model_topk=int(safe_neg_exclude_model_topk),
                    generator_seed=int(safe_neg_seed) + int(epoch_index) * 100000 + int(micro_idx),
                    allowed_vocab_mask=allowed_vocab_mask,
                    stopgrad_assignment=bool(assignment_stopgrad),
                )
            else:
                loss = yprime_only_nce_loss_from_assignment(
                    scores,
                    P,
                    pack['kind'],
                    pack['c_mask'],
                    stopgrad_assignment=bool(assignment_stopgrad),
                )
                safe_metric_row = {'safe_neg_enabled': False}
            loss.backward()
            optimizer.step()
            global_step += 1
            batch_val = float(loss.detach().cpu().item())
            losses.append(batch_val)
            epoch_losses.append(batch_val)
            batch_losses.append(batch_val)
            epoch_batch_losses.append(batch_val)
            metric_row = assignment_metrics(P, pack['q_mask'], pack['c_mask'], pack['demand'])
            metric_row.update(safe_metric_row)
            metric_row.update(support_metric_row)
            epoch_metric_rows.append(metric_row)
            if bool(write_runtime_metrics_jsonl):
                _append_jsonl(runtime_metrics_path, {
                    'row_type': 'microbatch',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'stage': str(stage_id),
                    'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(safe_negatives) else 'sinkhorn_yprime_nce_no_unknown',
                    'epoch': int(epoch_index) + 1,
                    'microbatch_idx': int(micro_idx),
                    'microbatch_total': int(epoch_plan.batch_count),
                    'loss': batch_val,
                    'optimization_loss': batch_val,
                    'effective_trajectory_count': int(pack['q_mask'].sum().detach().cpu().item()),
                    'candidate_column_count': int(pack['c_mask'].sum().detach().cpu().item()),
                    'vocab_scope_policy': str(vocab_scope_policy.get('policy', 'unknown')),
                    'allowed_train_vocab_count': int(vocab_scope_policy.get('allowed_train_vocab_count', 0)),
                    **metric_row,
                })
        if bool(write_runtime_metrics_jsonl):
            merged: Dict[str, float] = {}
            if epoch_metric_rows:
                for key in epoch_metric_rows[0].keys():
                    vals = [float(row.get(key, 0.0)) for row in epoch_metric_rows]
                    merged[f'{key}_epoch'] = _mean_or_zero(vals)
            _append_jsonl(runtime_metrics_path, {
                'row_type': 'epoch_summary',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage': str(stage_id),
                'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(safe_negatives) else 'sinkhorn_yprime_nce_no_unknown',
                'epoch': int(epoch_index) + 1,
                'microbatch_count': int(len(epoch_batch_losses)),
                'loss_mean': _mean_or_zero(epoch_losses),
                'loss_last': float(epoch_losses[-1]) if epoch_losses else 0.0,
                'optimization_loss_mean': _mean_or_zero(epoch_batch_losses),
                'optimization_loss_last': float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,
                **merged,
            })
    return {
        'stage_id': str(stage_id),
        'loss_mean': _mean_or_zero(losses),
        'loss_last': float(losses[-1]) if losses else 0.0,
        'optimization_loss_mean': _mean_or_zero(batch_losses),
        'optimization_loss_last': float(batch_losses[-1]) if batch_losses else 0.0,
        'global_step': int(global_step),
        'loss_normalization': 'sum_candidate_demand',
        'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(safe_negatives) else 'sinkhorn_yprime_nce_no_unknown',
        'safe_neg_enabled': bool(safe_negatives),
        'safe_neg_count': int(safe_neg_count),
        'safe_neg_weight': float(safe_neg_weight),
        'safe_neg_text_sim_threshold': float(safe_neg_text_sim_threshold),
        'safe_neg_exclude_model_topk': int(safe_neg_exclude_model_topk),
        'sinkhorn_final_rerank_lambda_r': float(sinkhorn_final_rerank_lambda_r),
        'support_null_enabled': bool(enable_null_column),
        'support_null_warmup_epochs': int(max(0, support_warmup_epochs)),
        'support_null_epoch_metric_count': int(len(support_null_epoch_metrics)),
        'support_null_metric_summary': {
            key: _mean_or_zero([float(row.get(key, 0.0)) for row in support_null_epoch_metrics])
            for key in (
                'null_mass_mean', 'nonnull_mass_mean', 'null_demand_mean', 'null_residual_uncapped_mean', 'null_cap_mean', 'null_demand_cap_ratio', 'yprime_demand_mean',
                'yprime_low_demand_rate', 'null_collapse_guard_triggered', 'support_demand_guard_triggered',
                'positive_protection_enabled', 'positive_protected_row_rate', 'positive_strength_mean',
                'positive_margin_mean', 'positive_margin_p50', 'positive_margin_p90',
                'positive_null_mass_before_mean', 'positive_null_mass_after_mean',
                'positive_null_excess_moved_mean', 'positive_redistributed_mass_mean', 'positive_null_cap'
            )
        } if support_null_epoch_metrics else {},
        'private_support_state_snapshot': {f'{int(k[0])}:{int(k[1])}': float(v) for k, v in support_state.items()},
        'vocab_scope_policy': dict(vocab_scope_policy),
        'residual_peeling_enabled': bool(residual_schedule is not None),
        'residual_candidate_policy': str(residual_candidate_policy),
        'known_null_gate': str(known_null_gate),
    }


def _sinkhorn_collect_responsibility_rows(
    *,
    stage_id: str,
    dataset_name: str,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    output_root: Path,
    projector: Projector,
    theta_t: torch.nn.Parameter,
    text_vocab_tensor: torch.Tensor,
    raw_to_vocab_idx: Mapping[int, int],
    mode: str,
    sinkhorn_tau: float,
    sinkhorn_iters: int,
    sinkhorn_row_cap_scale: float,
    extra_demand: float,
    sinkhorn_final_rerank_lambda_r: float = 0.0,
    vocab_scope_policy: Mapping[str, Any] | None = None,
    enable_null_column: bool = False,
    null_logit_bias: float = 0.0,
    null_residual: bool = False,
    null_demand_cap_ratio: float = 1.0,
    yprime_demand_mode: str = 'fixed',
    yprime_demand_min: float = 0.10,
    yprime_support_topk: int = 2,
    yprime_support_temp: float = 0.25,
    support_state_snapshot: Mapping[str, float] | None = None,
    enable_positive_protection: bool = False,
    positive_margin_threshold: float = 0.15,
    positive_margin_temp: float = 0.10,
    positive_null_cap: float = 0.40,
    positive_redistribute_mode: str = 'best_y',
    residual_schedule: Optional[Any] = None,
    residual_candidate_policy: str = 'base_residual',
    residual_collect_epoch_index: int = 0,
) -> List[Record]:
    from videocutler.ext_stageb_ovvis.algorithms.sinkhorn_assignment import SinkhornAssignmentConfig, capped_sinkhorn_assignment
    rows: List[Record] = []
    cfg = SinkhornAssignmentConfig(tau=float(sinkhorn_tau), iters=int(sinkhorn_iters), row_cap_scale=float(sinkhorn_row_cap_scale))
    support_state: Dict[Tuple[int, int], float] = {}
    if support_state_snapshot:
        for key, value in dict(support_state_snapshot).items():
            try:
                a, b = str(key).split(':', 1)
                support_state[(int(a), int(b))] = float(value)
            except Exception:
                continue
    projector.eval()
    with torch.no_grad():
        for group in groups:
            collect_groups = [group]
            if residual_schedule is not None:
                collect_groups, _residual_collect_metrics = apply_residual_candidate_override_to_groups(
                    collect_groups, schedule=residual_schedule, epoch_index=int(residual_collect_epoch_index), candidate_policy=str(residual_candidate_policy)
                )
                if not collect_groups:
                    continue
            pack = _sinkhorn_pack_groups(collect_groups, raw_to_vocab_idx=raw_to_vocab_idx, device=text_vocab_tensor.device, mode=str(mode), extra_demand=float(extra_demand))
            if not pack:
                continue
            scores = _sinkhorn_scores_from_pack(projector, text_vocab_tensor, pack, _compute_t_dis(theta_t))
            support_metric_row: Dict[str, Any] = {'support_null_enabled': False, 'support_null_active': False}
            if bool(enable_null_column) and str(mode) == 'prealign':
                pack, scores, support_metric_row = _apply_support_null_to_pack(
                    pack,
                    scores,
                    epoch_index=10**9,
                    support_state=support_state,
                    enable_null_column=True,
                    null_logit_bias=float(null_logit_bias),
                    null_residual=bool(null_residual),
                    null_demand_cap_ratio=float(null_demand_cap_ratio),
                    yprime_demand_mode=str(yprime_demand_mode),
                    yprime_demand_min=float(yprime_demand_min),
                    yprime_support_topk=int(yprime_support_topk),
                    yprime_support_temp=float(yprime_support_temp),
                    yprime_support_ema=1.0 if support_state_snapshot else 0.0,
                )
            P = capped_sinkhorn_assignment(scores, pack['q_mask'], pack['c_mask'], pack['demand'], config=cfg)
            P, positive_metric_row = _apply_positive_protection_to_assignment(
                P,
                scores,
                pack,
                enable_positive_protection=bool(enable_positive_protection) and bool(support_metric_row.get('support_null_active', False)),
                margin_threshold=float(positive_margin_threshold),
                margin_temp=float(positive_margin_temp),
                positive_null_cap=float(positive_null_cap),
                redistribute_mode=str(positive_redistribute_mode),
                null_logit_bias=float(null_logit_bias),
            )
            support_metric_row.update(positive_metric_row)
            P = P[0]
            raw_ids = [int(x) for x in pack['raw_ids'][0][pack['c_mask'][0]].detach().cpu().numpy().astype(np.int64).tolist()]
            kind = [int(x) for x in pack['kind'][0][pack['c_mask'][0]].detach().cpu().numpy().astype(np.int64).tolist()]
            demand_vals = [float(x) for x in pack['demand'][0][pack['c_mask'][0]].detach().cpu().numpy().astype(np.float64).tolist()]
            known_ids = [raw for raw, k in zip(raw_ids, kind) if int(k) == 1]
            extra_ids = [raw for raw, k in zip(raw_ids, kind) if int(k) == 2]
            null_ids = [raw for raw, k in zip(raw_ids, kind) if int(k) == 0]
            demand_by_raw_id = {str(int(raw)): float(val) for raw, val in zip(raw_ids, demand_vals)}
            kind_by_raw_id = {str(int(raw)): int(k) for raw, k in zip(raw_ids, kind)}
            for q, ex in enumerate(pack['groups'][0]):
                row_mass = P[q, :len(raw_ids)].detach().cpu().numpy().astype(np.float64)
                total = float(np.sum(row_mass))
                if total <= 1e-12:
                    # Fallback to local softmax when column coverage leaves a row with no mass.
                    local_scores = scores[0, q, :len(raw_ids)]
                    row_mass = torch.softmax(local_scores, dim=0).detach().cpu().numpy().astype(np.float64)
                    total = float(np.sum(row_mass))
                probs = (row_mass / max(total, 1e-12)).astype(np.float64).tolist()
                rerank_lambda = float(sinkhorn_final_rerank_lambda_r)
                if rerank_lambda > 0.0 and len(probs) > 0:
                    local_scores = scores[0, q, :len(raw_ids)].detach().cpu().numpy().astype(np.float64)
                    final_scores = local_scores + rerank_lambda * np.log(np.asarray(probs, dtype=np.float64) + 1.0e-8)
                    final_scores = final_scores - float(np.max(final_scores))
                    final_probs = np.exp(final_scores)
                    final_probs = final_probs / max(float(np.sum(final_probs)), 1.0e-12)
                    probs = final_probs.astype(np.float64).tolist()
                rows.append({
                    'dataset_name': str(dataset_name),
                    'clip_id': int(ex['clip_id']),
                    'video_id': int(ex['video_id']),
                    'trajectory_id': str(ex['trajectory_id']),
                    'candidate_ids_known': list(known_ids),
                    'candidate_ids_extra': list(extra_ids),
                    'candidate_ids_null': list(null_ids),
                    'candidate_scope_policy': dict(vocab_scope_policy or {}),
                    'candidate_demand_by_raw_id': dict(demand_by_raw_id),
                    'candidate_kind_by_raw_id': dict(kind_by_raw_id),
                    'support_null_active': bool(support_metric_row.get('support_null_active', False)),
                    'support_null_metrics': dict(support_metric_row),
                    'candidate_outside_train_vocab_count': int(sum(1 for x in (list(known_ids) + list(extra_ids)) if int(x) not in set((vocab_scope_policy or {}).get('allowed_train_vocab_raw_ids', [])))) if vocab_scope_policy else 0,
                    'unknown_disabled': True,
                    'training_semantics': 'sinkhorn_yprime_nce_no_unknown',
                    'stage_id': str(stage_id),
                    'r_init': {},
                    'r_final': {str(int(raw_id)): float(prob) for raw_id, prob in zip(raw_ids, probs)},
                    'join_key': str(ex['trajectory_id']),
                })
    projector.train()
    return rows


def run_reservoir_sinkhorn_no_unknown(*, output_root: Path, materialized_samples: Sequence[Record], config: ReservoirSinkhornNoUnknownConfig, stage_scope: str = 'sinkhorn_preaug_no_unknown') -> Dict[str, Any]:
    _set_seed(int(config.seed))
    device = torch.device(str(config.device))
    full_text_vocab_ids, _text_records, full_text_vocab_matrix = load_text_vocab(output_root)
    allowed_train_raw_ids, vocab_scope_policy = _resolve_sinkhorn_train_vocab(
        policy=str(config.sinkhorn_vocab_scope_policy),
        train_vocab_raw_ids=tuple(int(x) for x in config.train_vocab_raw_ids),
        text_vocab_ids=full_text_vocab_ids,
        strict_check=bool(config.sinkhorn_vocab_scope_strict_check),
    )
    text_vocab_ids, text_vocab_matrix, text_scope_meta = _scope_text_vocab(full_text_vocab_ids, full_text_vocab_matrix, allowed_train_raw_ids)
    vocab_scope_policy = {**vocab_scope_policy, **text_scope_meta, 'allowed_train_vocab_raw_ids': [int(x) for x in allowed_train_raw_ids]}
    raw_to_vocab_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    allowed_vocab_mask = torch.ones((len(text_vocab_ids),), device=device, dtype=torch.bool)
    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    projector = Projector(config.projector).to(device)
    projector.train()
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(config.t_dis_init)-1e-4,1e-6)), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(config.prealign_learning_rate), weight_decay=float(config.weight_decay))
    batch_budget = resolve_default_batch_budget(smoke=bool(config.smoke), explicit=config.batch_budget)

    prepared_pre = _prepare_prealign_examples(materialized_samples, output_root=output_root, dataset_name=config.dataset_name, trajectory_source_branch=config.trajectory_source_branch)
    pre_examples = list(prepared_pre['examples'])
    if not pre_examples:
        raise RuntimeError('no trainable examples for sinkhorn prealign')
    if bool(config.sinkhorn_vocab_scope_strict_check):
        outside_yprime = sorted({
            int(raw_id)
            for ex in pre_examples
            for raw_id in list(ex.get('observed_raw_ids', []))
            if int(raw_id) not in raw_to_vocab_idx
        })
        if outside_yprime:
            raise RuntimeError(f'weak-label-only scope violation: observed Yprime ids outside train vocab: {outside_yprime[:16]}')
    pre_groups = _clip_groups(pre_examples)
    residual_schedule_obj = None
    residual_schedule_summary: Dict[str, Any] = {'mode': str(config.residual_peeling_mode), 'enabled': False}
    if str(config.residual_peeling_mode) == 'oracle_static':
        if load_oracle_static_residual_schedule is None:
            raise RuntimeError('residual peeling schedule helper is unavailable')
        residual_schedule_obj = load_oracle_static_residual_schedule(
            csv_path=str(config.residual_schedule_csv),
            annotation_json=str(config.residual_annotation_json),
            split_json=str(config.residual_split_json),
            variant=str(config.residual_variant),
            epoch_plan=str(config.residual_round_epoch_plan),
        )
        residual_schedule_summary = {**residual_schedule_obj.public_summary(), 'enabled': True}
        _write_json(output_root / 'train' / 'residual_peeling_schedule_summary.json', residual_schedule_summary)
    pre_stage = _sinkhorn_train_stage(
        stage_id='prealign', groups=pre_groups, output_root=output_root, projector=projector, theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, optimizer=optimizer,
        epochs=int(config.prealign_epochs), learning_rate=float(config.prealign_learning_rate), batch_budget=int(batch_budget), seed=int(config.seed), mode='prealign',
        sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale),
        extra_demand=0.0, extra_lambda=0.0, assignment_stopgrad=bool(config.sinkhorn_assignment_stopgrad), safe_negatives=bool(config.sinkhorn_safe_negatives), safe_neg_count=int(config.sinkhorn_safe_neg_count), safe_neg_weight=float(config.sinkhorn_safe_neg_weight), safe_neg_text_sim_threshold=float(config.sinkhorn_safe_neg_text_sim_threshold), safe_neg_exclude_model_topk=int(config.sinkhorn_safe_neg_exclude_model_topk), safe_neg_seed=int(config.sinkhorn_safe_neg_seed), sinkhorn_final_rerank_lambda_r=float(config.sinkhorn_final_rerank_lambda_r), allowed_vocab_mask=allowed_vocab_mask, vocab_scope_policy=vocab_scope_policy, show_progress=bool(config.show_progress), write_runtime_metrics_jsonl=bool(config.write_runtime_metrics_jsonl),
        enable_null_column=bool(config.sinkhorn_enable_null_column) or str(stage_scope) == 'support_null_prealign_base_only',
        null_logit_bias=float(config.sinkhorn_null_logit_bias),
        null_residual=bool(config.sinkhorn_null_residual) or str(stage_scope) == 'support_null_prealign_base_only',
        null_demand_cap_ratio=float(config.sinkhorn_null_demand_cap_ratio),
        support_warmup_epochs=int(config.sinkhorn_support_warmup_epochs),
        yprime_demand_mode=str(config.sinkhorn_yprime_demand_mode),
        yprime_demand_min=float(config.sinkhorn_yprime_demand_min),
        yprime_support_topk=int(config.sinkhorn_yprime_support_topk),
        yprime_support_temp=float(config.sinkhorn_yprime_support_temp),
        yprime_support_ema=float(config.sinkhorn_yprime_support_ema),
        null_collapse_max=float(config.sinkhorn_null_collapse_max),
        yprime_demand_min_guard=float(config.sinkhorn_yprime_demand_min_guard),
        enable_positive_protection=bool(config.sinkhorn_enable_positive_protection),
        positive_margin_threshold=float(config.sinkhorn_positive_margin_threshold),
        positive_margin_temp=float(config.sinkhorn_positive_margin_temp),
        positive_null_cap=float(config.sinkhorn_positive_null_cap),
        positive_redistribute_mode=str(config.sinkhorn_positive_redistribute_mode),
        residual_schedule=residual_schedule_obj,
        residual_candidate_policy=str(config.residual_candidate_policy),
        known_null_gate=str(config.known_null_gate),
        known_null_margin=float(config.known_null_margin),
    )
    train_dir = output_root / 'train' / 'prealign'
    ckpt_dir = train_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last_path = ckpt_dir / 'prealign_last.pth'
    torch.save({
        'stage_id': 'prealign', 'epoch': int(config.prealign_epochs), 'text_projector_state_dict': projector.state_dict(),
        'text_projector_config': {'input_dim': int(config.projector.input_dim), 'hidden_dim': int(config.projector.hidden_dim), 'output_dim': int(config.projector.output_dim), 'dropout': float(config.projector.dropout), 'use_layernorm': bool(config.projector.use_layernorm)},
        'theta_T': float(theta_t.detach().cpu().item()), 'b_u': 0.0, 'unknown_disabled': True,
        'seed': int(config.seed), 'global_step': int(pre_stage.get('global_step', 0)), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(config.sinkhorn_safe_negatives) else 'sinkhorn_yprime_nce_no_unknown', 'safe_neg_enabled': bool(config.sinkhorn_safe_negatives), 'safe_neg_count': int(config.sinkhorn_safe_neg_count), 'safe_neg_weight': float(config.sinkhorn_safe_neg_weight), 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'support_null_state_snapshot': dict(pre_stage.get('private_support_state_snapshot', {})), 'residual_peeling_schedule': residual_schedule_summary, 'support_null_config': {'enable_null_column': bool(config.sinkhorn_enable_null_column) or str(stage_scope) == 'support_null_prealign_base_only', 'null_logit_bias': float(config.sinkhorn_null_logit_bias), 'null_residual': bool(config.sinkhorn_null_residual) or str(stage_scope) == 'support_null_prealign_base_only', 'null_demand_cap_ratio': float(config.sinkhorn_null_demand_cap_ratio), 'yprime_demand_mode': str(config.sinkhorn_yprime_demand_mode), 'yprime_demand_min': float(config.sinkhorn_yprime_demand_min), 'yprime_support_topk': int(config.sinkhorn_yprime_support_topk), 'yprime_support_temp': float(config.sinkhorn_yprime_support_temp), 'positive_protection_enabled': bool(config.sinkhorn_enable_positive_protection), 'positive_margin_threshold': float(config.sinkhorn_positive_margin_threshold), 'positive_margin_temp': float(config.sinkhorn_positive_margin_temp), 'positive_null_cap': float(config.sinkhorn_positive_null_cap), 'positive_redistribute_mode': str(config.sinkhorn_positive_redistribute_mode)}, 'vocab_scope_policy': vocab_scope_policy
    }, ckpt_last_path)
    pre_proxy_rows = _sinkhorn_collect_responsibility_rows(
        stage_id='prealign', dataset_name=str(config.dataset_name), groups=pre_groups, output_root=output_root, projector=projector, theta_t=theta_t, text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, mode='prealign', sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale), extra_demand=0.0, sinkhorn_final_rerank_lambda_r=float(config.sinkhorn_final_rerank_lambda_r), vocab_scope_policy=vocab_scope_policy,
        enable_null_column=(bool(config.sinkhorn_enable_null_column) or str(stage_scope) == 'support_null_prealign_base_only'),
        null_logit_bias=float(config.sinkhorn_null_logit_bias),
        null_residual=(bool(config.sinkhorn_null_residual) or str(stage_scope) == 'support_null_prealign_base_only'),
        null_demand_cap_ratio=float(config.sinkhorn_null_demand_cap_ratio),
        yprime_demand_mode=str(config.sinkhorn_yprime_demand_mode),
        yprime_demand_min=float(config.sinkhorn_yprime_demand_min),
        yprime_support_topk=int(config.sinkhorn_yprime_support_topk),
        yprime_support_temp=float(config.sinkhorn_yprime_support_temp),
        support_state_snapshot=pre_stage.get('private_support_state_snapshot', {}),
        enable_positive_protection=bool(config.sinkhorn_enable_positive_protection),
        positive_margin_threshold=float(config.sinkhorn_positive_margin_threshold),
        positive_margin_temp=float(config.sinkhorn_positive_margin_temp),
        positive_null_cap=float(config.sinkhorn_positive_null_cap),
        positive_redistribute_mode=str(config.sinkhorn_positive_redistribute_mode),
        residual_schedule=residual_schedule_obj,
        residual_candidate_policy=str(config.residual_candidate_policy),
        residual_collect_epoch_index=max(0, int(config.prealign_epochs) - 1),
    )
    _write_jsonl(train_dir / 'proxy_records.jsonl', pre_proxy_rows)
    _write_jsonl(train_dir / 'responsibility_records.jsonl', pre_proxy_rows)
    pre_train_state = {'stage_id': 'prealign', 'epoch': int(config.prealign_epochs), 'selected_for_infer': 'prealign_only', 'selected_for_infer_authority': 'explicit_train_state_field', 'checkpoint_last': 'train/prealign/checkpoints/prealign_last.pth', 'checkpoint_selected': 'train/prealign/checkpoints/prealign_last.pth', 'global_step': int(pre_stage.get('global_step', 0)), 'runtime_asset_source': str(config.runtime_asset_source), 'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete), 'runtime_asset_output_root': str(config.runtime_asset_output_root), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(config.sinkhorn_safe_negatives) else 'sinkhorn_yprime_nce_no_unknown', 'unknown_disabled': True, 'safe_neg_enabled': bool(config.sinkhorn_safe_negatives), 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy, 'residual_peeling_schedule': residual_schedule_summary}
    _write_json(train_dir / 'train_state.json', pre_train_state)
    pre_scope_contract = _sinkhorn_scope_contract_fields(vocab_scope_policy)
    pre_stage_public = {k: v for k, v in dict(pre_stage).items() if k != 'private_support_state_snapshot'}
    pre_summary = {**pre_stage_public, **pre_scope_contract, 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'unknown_disabled': True, 'record_count_output': int(len(pre_proxy_rows)), 'checkpoint_last_path': 'train/prealign/checkpoints/prealign_last.pth', 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy}
    _write_json(train_dir / 'stage_summary.json', pre_summary)
    if bool(pre_stage.get('support_null_enabled', False)):
        _write_json(train_dir / 'support_null_summary.json', {
            'stage_id': 'prealign',
            'support_null_enabled': bool(pre_stage.get('support_null_enabled', False)),
            'support_null_warmup_epochs': int(pre_stage.get('support_null_warmup_epochs', 0)),
            'support_null_metric_summary': dict(pre_stage.get('support_null_metric_summary', {})),
            'support_null_epoch_metric_count': int(pre_stage.get('support_null_epoch_metric_count', 0)),
            'null_semantics': 'non_semantic_assignment_slack_not_unknown_class',
        })

    stage_reports = [{'stage_id': 'prealign', 'responsibility_records_path': 'train/prealign/responsibility_records.jsonl', 'proxy_records_path': 'train/prealign/proxy_records.jsonl', 'train_state_path': 'train/prealign/train_state.json', 'checkpoint_last_path': 'train/prealign/checkpoints/prealign_last.pth', 'record_count_output': int(len(pre_proxy_rows)), **pre_stage_public}]
    selected_checkpoint_path = 'train/prealign/checkpoints/prealign_last.pth'
    final_count = len(pre_proxy_rows)

    if str(stage_scope) not in ('sinkhorn_prealign_only', 'support_null_prealign_base_only'):
        prepared_aug = _prepare_softem_examples(materialized_samples, output_root=output_root, dataset_name=config.dataset_name, trajectory_source_branch=config.trajectory_source_branch)
        aug_examples0 = list(prepared_aug['examples'])
        runtime_extra_cache = _build_runtime_extra_cache(examples=aug_examples0, text_projector=projector, theta_t=theta_t, output_root=output_root, k_extra=int(config.k_extra), alpha=float(config.extra_alpha), lambda_frame=float(config.lambda_frame), device=device, extra_margin_gate=float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, allowed_extra_raw_ids=allowed_train_raw_ids, extra_vocab_scope_policy=str(vocab_scope_policy.get('policy', 'unknown')), strict_check=bool(config.sinkhorn_vocab_scope_strict_check))
        aug_examples = _apply_runtime_extra_cache(aug_examples0, runtime_extra_cache=runtime_extra_cache, output_root=output_root) if runtime_extra_cache else aug_examples0
        aug_groups = _clip_groups(aug_examples)
        aug_stage = _sinkhorn_train_stage(
            stage_id='softem_aug', groups=aug_groups, output_root=output_root, projector=projector, theta_t=theta_t,
            text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, optimizer=optimizer,
            epochs=int(config.aug_epochs), learning_rate=float(config.aug_learning_rate), batch_budget=int(batch_budget), seed=int(config.seed) + 1000, mode='aug',
            sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale),
            extra_demand=float(config.sinkhorn_extra_demand), extra_lambda=float(config.sinkhorn_aug_extra_lambda), assignment_stopgrad=bool(config.sinkhorn_assignment_stopgrad), safe_negatives=bool(config.sinkhorn_safe_negatives), safe_neg_count=int(config.sinkhorn_safe_neg_count), safe_neg_weight=float(config.sinkhorn_safe_neg_weight), safe_neg_text_sim_threshold=float(config.sinkhorn_safe_neg_text_sim_threshold), safe_neg_exclude_model_topk=int(config.sinkhorn_safe_neg_exclude_model_topk), safe_neg_seed=int(config.sinkhorn_safe_neg_seed) + 1000, sinkhorn_final_rerank_lambda_r=float(config.sinkhorn_final_rerank_lambda_r), allowed_vocab_mask=allowed_vocab_mask, vocab_scope_policy=vocab_scope_policy, show_progress=bool(config.show_progress), write_runtime_metrics_jsonl=bool(config.write_runtime_metrics_jsonl),
        )
        aug_dir = output_root / 'train' / 'softem_aug'
        aug_ckpt_dir = aug_dir / 'checkpoints'
        aug_ckpt_dir.mkdir(parents=True, exist_ok=True)
        aug_ckpt_path = aug_ckpt_dir / 'softem_aug_last.pth'
        torch.save({
            'stage_id': 'softem_aug', 'epoch': int(config.aug_epochs), 'text_projector_state_dict': projector.state_dict(),
            'text_projector_config': {'input_dim': int(config.projector.input_dim), 'hidden_dim': int(config.projector.hidden_dim), 'output_dim': int(config.projector.output_dim), 'dropout': float(config.projector.dropout), 'use_layernorm': bool(config.projector.use_layernorm)},
            'theta_T': float(theta_t.detach().cpu().item()), 'b_u': 0.0, 'unknown_disabled': True,
            'seed': int(config.seed), 'global_step': int(aug_stage.get('global_step', 0)), 'mode': 'sinkhorn_aug_no_unknown', 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_yprime_nce_no_unknown',
            'sinkhorn_extra_demand': float(config.sinkhorn_extra_demand), 'sinkhorn_aug_extra_lambda': float(config.sinkhorn_aug_extra_lambda), 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy,
        }, aug_ckpt_path)
        aug_rows = _sinkhorn_collect_responsibility_rows(stage_id='softem_aug', dataset_name=str(config.dataset_name), groups=aug_groups, output_root=output_root, projector=projector, theta_t=theta_t, text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, mode='aug', sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale), extra_demand=float(config.sinkhorn_extra_demand), sinkhorn_final_rerank_lambda_r=float(config.sinkhorn_final_rerank_lambda_r), vocab_scope_policy=vocab_scope_policy)
        _write_jsonl(aug_dir / 'responsibility_records.jsonl', aug_rows)
        aug_state = {'stage_id': 'softem_aug', 'epoch': int(config.aug_epochs), 'selected_for_infer': 'augmented', 'selected_for_infer_authority': 'explicit_train_state_field', 'checkpoint_last': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'checkpoint_selected': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'global_step': int(aug_stage.get('global_step', 0)), 'runtime_asset_source': str(config.runtime_asset_source), 'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete), 'runtime_asset_output_root': str(config.runtime_asset_output_root), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_yprime_nce_no_unknown', 'unknown_disabled': True, 'softem_base_skipped': True, 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy}
        _write_json(aug_dir / 'train_state.json', aug_state)
        runtime_extra_metrics = {'example_count': int(len(aug_examples)), 'k_extra': int(config.k_extra), 'extra_alpha': float(config.extra_alpha), 'sinkhorn_extra_demand': float(config.sinkhorn_extra_demand), 'sinkhorn_aug_extra_lambda': float(config.sinkhorn_aug_extra_lambda), 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy}
        runtime_extra_metrics.update(_sinkhorn_runtime_extra_cache_metrics(runtime_extra_cache))
        aug_scope_contract = _sinkhorn_scope_contract_fields(vocab_scope_policy, runtime_extra_cache=runtime_extra_cache, aug_rows=aug_rows)
        aug_summary = {**aug_stage, **aug_scope_contract, 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'unknown_disabled': True, 'softem_base_skipped': True, 'record_count_output': int(len(aug_rows)), 'runtime_extra_cache_metrics': runtime_extra_metrics, 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), 'vocab_scope_policy': vocab_scope_policy, 'checkpoint_last_path': 'train/softem_aug/checkpoints/softem_aug_last.pth'}
        _write_json(aug_dir / 'stage_summary.json', aug_summary)
        stage_reports.append({'stage_id': 'softem_aug', 'responsibility_records_path': 'train/softem_aug/responsibility_records.jsonl', 'train_state_path': 'train/softem_aug/train_state.json', 'checkpoint_last_path': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'record_count_output': int(len(aug_rows)), 'sinkhorn_extra_margin_gate': float(config.sinkhorn_extra_margin_gate) if config.sinkhorn_extra_margin_gate is not None else None, 'sinkhorn_final_rerank_lambda_r': float(config.sinkhorn_final_rerank_lambda_r), **aug_stage})
        selected_checkpoint_path = 'train/softem_aug/checkpoints/softem_aug_last.pth'
        final_count = len(aug_rows)

    return {
        'stage_reports': stage_reports,
        'record_count_input': int(len(materialized_samples)),
        'record_count_trainable': int(len(pre_examples)),
        'record_count_output': int(final_count),
        'coverage_ratio_trainable': float(len(pre_examples) / max(len(materialized_samples), 1)),
        'skipped_reason_histogram': dict(prepared_pre.get('skipped_reason_histogram', {})),
        'selected_checkpoint_path': selected_checkpoint_path,
        'unknown_disabled': True,
        'softem_base_skipped': True,
        'pipeline': 'reservoir_v1_sinkhorn_no_unknown',
        'training_semantics': 'sinkhorn_yprime_nce_no_unknown',
        'vocab_scope_policy': vocab_scope_policy,
        **pre_scope_contract,
        **(aug_scope_contract if str(stage_scope) not in ('sinkhorn_prealign_only', 'support_null_prealign_base_only') else pre_scope_contract),
    }
