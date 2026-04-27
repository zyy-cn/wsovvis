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
    show_progress: bool = True
    log_every: int = 10
    write_runtime_metrics_jsonl: bool = True
    print_epoch_summary: bool = True


def _sinkhorn_observed_ids(group: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted({int(x) for ex in group for x in list(ex.get('observed_raw_ids', []))})


def _sinkhorn_known_extra_ids(group: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
    known = sorted({int(x) for ex in group for x in list(ex.get('candidate_ids_known', []))})
    extra = sorted({int(x) for ex in group for x in list(ex.get('candidate_ids_extra', [])) if int(x) not in set(known)})
    return known, extra


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
    return {
        'groups': packed_groups,
        'Z': Z,
        'q_mask': q_mask,
        'yidx': yidx,
        'c_mask': c_mask,
        'demand': demand,
        'kind': kind,
        'raw_ids': raw_ids_tensor,
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
    show_progress: bool,
    write_runtime_metrics_jsonl: bool,
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
    for epoch_index in _maybe_tqdm(range(int(epochs)), enabled=bool(show_progress), desc=f'{stage_id} epochs', leave=True):
        shuffled_groups = list(groups)
        random.Random(int(seed) + int(epoch_index)).shuffle(shuffled_groups)
        epoch_plan = build_dynamic_microbatches(shuffled_groups, batch_budget=int(batch_budget), cost_fn=cost_fn, bucket_key_fn=bucket_fn)
        epoch_losses: List[float] = []
        epoch_batch_losses: List[float] = []
        epoch_metric_rows: List[Dict[str, Any]] = []
        for micro_idx, batch_indices in enumerate(_maybe_tqdm(epoch_plan.batches, enabled=bool(show_progress), desc=f'{stage_id} epoch {int(epoch_index)+1}', total=len(epoch_plan.batches), leave=False), start=1):
            selected_groups = [shuffled_groups[int(i)] for i in batch_indices]
            pack = _sinkhorn_pack_groups(selected_groups, raw_to_vocab_idx=raw_to_vocab_idx, device=text_vocab_tensor.device, mode=str(mode), extra_demand=float(extra_demand))
            if not pack:
                continue
            optimizer.zero_grad(set_to_none=True)
            temperature = _compute_t_dis(theta_t)
            if bool(safe_negatives):
                scores, full_scores = _sinkhorn_candidate_and_full_scores_from_pack(projector, text_vocab_tensor, pack, temperature)
            else:
                scores = _sinkhorn_scores_from_pack(projector, text_vocab_tensor, pack, temperature)
                full_scores = None
            P = capped_sinkhorn_assignment(scores, pack['q_mask'], pack['c_mask'], pack['demand'], config=cfg)
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
) -> List[Record]:
    from videocutler.ext_stageb_ovvis.algorithms.sinkhorn_assignment import SinkhornAssignmentConfig, capped_sinkhorn_assignment
    rows: List[Record] = []
    cfg = SinkhornAssignmentConfig(tau=float(sinkhorn_tau), iters=int(sinkhorn_iters), row_cap_scale=float(sinkhorn_row_cap_scale))
    projector.eval()
    with torch.no_grad():
        for group in groups:
            pack = _sinkhorn_pack_groups([group], raw_to_vocab_idx=raw_to_vocab_idx, device=text_vocab_tensor.device, mode=str(mode), extra_demand=float(extra_demand))
            if not pack:
                continue
            scores = _sinkhorn_scores_from_pack(projector, text_vocab_tensor, pack, _compute_t_dis(theta_t))
            P = capped_sinkhorn_assignment(scores, pack['q_mask'], pack['c_mask'], pack['demand'], config=cfg)[0]
            raw_ids = [int(x) for x in pack['raw_ids'][0][pack['c_mask'][0]].detach().cpu().numpy().astype(np.int64).tolist()]
            kind = [int(x) for x in pack['kind'][0][pack['c_mask'][0]].detach().cpu().numpy().astype(np.int64).tolist()]
            known_ids = [raw for raw, k in zip(raw_ids, kind) if int(k) == 1]
            extra_ids = [raw for raw, k in zip(raw_ids, kind) if int(k) == 2]
            for q, ex in enumerate(pack['groups'][0]):
                row_mass = P[q, :len(raw_ids)].detach().cpu().numpy().astype(np.float64)
                total = float(np.sum(row_mass))
                if total <= 1e-12:
                    # Fallback to local softmax when column coverage leaves a row with no mass.
                    local_scores = scores[0, q, :len(raw_ids)]
                    row_mass = torch.softmax(local_scores, dim=0).detach().cpu().numpy().astype(np.float64)
                    total = float(np.sum(row_mass))
                probs = (row_mass / max(total, 1e-12)).astype(np.float64).tolist()
                rows.append({
                    'dataset_name': str(dataset_name),
                    'clip_id': int(ex['clip_id']),
                    'video_id': int(ex['video_id']),
                    'trajectory_id': str(ex['trajectory_id']),
                    'candidate_ids_known': list(known_ids),
                    'candidate_ids_extra': list(extra_ids),
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
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(output_root)
    raw_to_vocab_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
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
    pre_groups = _clip_groups(pre_examples)
    pre_stage = _sinkhorn_train_stage(
        stage_id='prealign', groups=pre_groups, output_root=output_root, projector=projector, theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, optimizer=optimizer,
        epochs=int(config.prealign_epochs), learning_rate=float(config.prealign_learning_rate), batch_budget=int(batch_budget), seed=int(config.seed), mode='prealign',
        sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale),
        extra_demand=0.0, extra_lambda=0.0, assignment_stopgrad=bool(config.sinkhorn_assignment_stopgrad), safe_negatives=bool(config.sinkhorn_safe_negatives), safe_neg_count=int(config.sinkhorn_safe_neg_count), safe_neg_weight=float(config.sinkhorn_safe_neg_weight), safe_neg_text_sim_threshold=float(config.sinkhorn_safe_neg_text_sim_threshold), safe_neg_exclude_model_topk=int(config.sinkhorn_safe_neg_exclude_model_topk), safe_neg_seed=int(config.sinkhorn_safe_neg_seed), show_progress=bool(config.show_progress), write_runtime_metrics_jsonl=bool(config.write_runtime_metrics_jsonl),
    )
    train_dir = output_root / 'train' / 'prealign'
    ckpt_dir = train_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last_path = ckpt_dir / 'prealign_last.pth'
    torch.save({
        'stage_id': 'prealign', 'epoch': int(config.prealign_epochs), 'text_projector_state_dict': projector.state_dict(),
        'text_projector_config': {'input_dim': int(config.projector.input_dim), 'hidden_dim': int(config.projector.hidden_dim), 'output_dim': int(config.projector.output_dim), 'dropout': float(config.projector.dropout), 'use_layernorm': bool(config.projector.use_layernorm)},
        'theta_T': float(theta_t.detach().cpu().item()), 'b_u': 0.0, 'unknown_disabled': True,
        'seed': int(config.seed), 'global_step': int(pre_stage.get('global_step', 0)), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(config.sinkhorn_safe_negatives) else 'sinkhorn_yprime_nce_no_unknown', 'safe_neg_enabled': bool(config.sinkhorn_safe_negatives), 'safe_neg_count': int(config.sinkhorn_safe_neg_count), 'safe_neg_weight': float(config.sinkhorn_safe_neg_weight)
    }, ckpt_last_path)
    pre_proxy_rows = _sinkhorn_collect_responsibility_rows(stage_id='prealign', dataset_name=str(config.dataset_name), groups=pre_groups, output_root=output_root, projector=projector, theta_t=theta_t, text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, mode='prealign', sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale), extra_demand=0.0)
    _write_jsonl(train_dir / 'proxy_records.jsonl', pre_proxy_rows)
    pre_train_state = {'stage_id': 'prealign', 'epoch': int(config.prealign_epochs), 'selected_for_infer': 'prealign_only', 'selected_for_infer_authority': 'explicit_train_state_field', 'checkpoint_last': 'train/prealign/checkpoints/prealign_last.pth', 'checkpoint_selected': 'train/prealign/checkpoints/prealign_last.pth', 'global_step': int(pre_stage.get('global_step', 0)), 'runtime_asset_source': str(config.runtime_asset_source), 'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete), 'runtime_asset_output_root': str(config.runtime_asset_output_root), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_safe_neg_yprime_nce_no_unknown' if bool(config.sinkhorn_safe_negatives) else 'sinkhorn_yprime_nce_no_unknown', 'unknown_disabled': True, 'safe_neg_enabled': bool(config.sinkhorn_safe_negatives)}
    _write_json(train_dir / 'train_state.json', pre_train_state)
    pre_summary = {**pre_stage, 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'unknown_disabled': True, 'record_count_output': int(len(pre_proxy_rows)), 'checkpoint_last_path': 'train/prealign/checkpoints/prealign_last.pth'}
    _write_json(train_dir / 'stage_summary.json', pre_summary)

    stage_reports = [{'stage_id': 'prealign', 'responsibility_records_path': 'train/prealign/proxy_records.jsonl', 'train_state_path': 'train/prealign/train_state.json', 'checkpoint_last_path': 'train/prealign/checkpoints/prealign_last.pth', 'record_count_output': int(len(pre_proxy_rows)), **pre_stage}]
    selected_checkpoint_path = 'train/prealign/checkpoints/prealign_last.pth'
    final_count = len(pre_proxy_rows)

    if str(stage_scope) != 'sinkhorn_prealign_only':
        prepared_aug = _prepare_softem_examples(materialized_samples, output_root=output_root, dataset_name=config.dataset_name, trajectory_source_branch=config.trajectory_source_branch)
        aug_examples0 = list(prepared_aug['examples'])
        runtime_extra_cache = _build_runtime_extra_cache(examples=aug_examples0, text_projector=projector, theta_t=theta_t, output_root=output_root, k_extra=int(config.k_extra), alpha=float(config.extra_alpha), lambda_frame=float(config.lambda_frame), device=device)
        aug_examples = _apply_runtime_extra_cache(aug_examples0, runtime_extra_cache=runtime_extra_cache, output_root=output_root) if runtime_extra_cache else aug_examples0
        aug_groups = _clip_groups(aug_examples)
        aug_stage = _sinkhorn_train_stage(
            stage_id='softem_aug', groups=aug_groups, output_root=output_root, projector=projector, theta_t=theta_t,
            text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, optimizer=optimizer,
            epochs=int(config.aug_epochs), learning_rate=float(config.aug_learning_rate), batch_budget=int(batch_budget), seed=int(config.seed) + 1000, mode='aug',
            sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale),
            extra_demand=float(config.sinkhorn_extra_demand), extra_lambda=float(config.sinkhorn_aug_extra_lambda), assignment_stopgrad=bool(config.sinkhorn_assignment_stopgrad), safe_negatives=bool(config.sinkhorn_safe_negatives), safe_neg_count=int(config.sinkhorn_safe_neg_count), safe_neg_weight=float(config.sinkhorn_safe_neg_weight), safe_neg_text_sim_threshold=float(config.sinkhorn_safe_neg_text_sim_threshold), safe_neg_exclude_model_topk=int(config.sinkhorn_safe_neg_exclude_model_topk), safe_neg_seed=int(config.sinkhorn_safe_neg_seed) + 1000, show_progress=bool(config.show_progress), write_runtime_metrics_jsonl=bool(config.write_runtime_metrics_jsonl),
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
            'sinkhorn_extra_demand': float(config.sinkhorn_extra_demand), 'sinkhorn_aug_extra_lambda': float(config.sinkhorn_aug_extra_lambda),
        }, aug_ckpt_path)
        aug_rows = _sinkhorn_collect_responsibility_rows(stage_id='softem_aug', dataset_name=str(config.dataset_name), groups=aug_groups, output_root=output_root, projector=projector, theta_t=theta_t, text_vocab_tensor=text_vocab_tensor, raw_to_vocab_idx=raw_to_vocab_idx, mode='aug', sinkhorn_tau=float(config.sinkhorn_tau), sinkhorn_iters=int(config.sinkhorn_iters), sinkhorn_row_cap_scale=float(config.sinkhorn_row_cap_scale), extra_demand=float(config.sinkhorn_extra_demand))
        _write_jsonl(aug_dir / 'responsibility_records.jsonl', aug_rows)
        aug_state = {'stage_id': 'softem_aug', 'epoch': int(config.aug_epochs), 'selected_for_infer': 'augmented', 'selected_for_infer_authority': 'explicit_train_state_field', 'checkpoint_last': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'checkpoint_selected': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'global_step': int(aug_stage.get('global_step', 0)), 'runtime_asset_source': str(config.runtime_asset_source), 'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete), 'runtime_asset_output_root': str(config.runtime_asset_output_root), 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'training_semantics': 'sinkhorn_yprime_nce_no_unknown', 'unknown_disabled': True, 'softem_base_skipped': True}
        _write_json(aug_dir / 'train_state.json', aug_state)
        runtime_extra_metrics = {'clip_cache_count': int(len(runtime_extra_cache)), 'example_count': int(len(aug_examples)), 'example_with_nonempty_extra_count': int(sum(1 for row in aug_examples if len(list(row.get('candidate_ids_extra', []))) > 0)), 'k_extra': int(config.k_extra), 'extra_alpha': float(config.extra_alpha), 'sinkhorn_extra_demand': float(config.sinkhorn_extra_demand), 'sinkhorn_aug_extra_lambda': float(config.sinkhorn_aug_extra_lambda)}
        aug_summary = {**aug_stage, 'pipeline': 'reservoir_v1_sinkhorn_no_unknown', 'unknown_disabled': True, 'softem_base_skipped': True, 'record_count_output': int(len(aug_rows)), 'runtime_extra_cache_metrics': runtime_extra_metrics, 'checkpoint_last_path': 'train/softem_aug/checkpoints/softem_aug_last.pth'}
        _write_json(aug_dir / 'stage_summary.json', aug_summary)
        stage_reports.append({'stage_id': 'softem_aug', 'responsibility_records_path': 'train/softem_aug/responsibility_records.jsonl', 'train_state_path': 'train/softem_aug/train_state.json', 'checkpoint_last_path': 'train/softem_aug/checkpoints/softem_aug_last.pth', 'record_count_output': int(len(aug_rows)), **aug_stage})
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
    }
