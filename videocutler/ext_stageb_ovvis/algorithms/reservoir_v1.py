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
from videocutler.ext_stageb_ovvis.algorithms.soft_em import _prepare_examples as _prepare_softem_examples
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.ext_stageb_ovvis.utils.unknown_metrics import UnknownMetricsAccumulator
Record=Dict[str,Any]
@dataclass(frozen=True)
class ReservoirPrealignConfig:
    dataset_name:str; trajectory_source_branch:str='mainline'; device:str='cpu'; seed:int=0; smoke:bool=False; lambda_frame:float=0.25; epochs:int=1; learning_rate:float=1e-4; weight_decay:float=1e-2; t_dis_init:float=0.07; projector:ProjectorConfig=ProjectorConfig(); runtime_asset_source:str='local_canonical_assets'; runtime_asset_source_local_incomplete:bool=False; runtime_asset_output_root:str=''; batch_budget:int|None=None; show_progress:bool=True; log_every:int=10; write_runtime_metrics_jsonl:bool=True; print_epoch_summary:bool=True
@dataclass(frozen=True)
class ReservoirSoftEMConfig:
    dataset_name:str; trajectory_source_branch:str='mainline'; mode:str='base_then_aug'; device:str='cpu'; seed:int=0; smoke:bool=False; lambda_frame:float=0.25; t_dis_init:float=0.07; weight_decay:float=1e-2; projector:ProjectorConfig=ProjectorConfig(); base_epochs:int=1; aug_epochs:int=1; base_learning_rate:float=5e-5; aug_learning_rate:float=5e-5; runtime_asset_source:str='local_canonical_assets'; runtime_asset_source_local_incomplete:bool=False; runtime_asset_output_root:str=''; batch_budget:int|None=None; base_release_margin:float=0.0; ablate_skip_base:bool=False; ablate_no_yprime_reward:bool=False; show_progress:bool=True; log_every:int=10; write_runtime_metrics_jsonl:bool=True; print_epoch_summary:bool=True

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
        for ex in examples:
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
        _write_json(stage_dir/'stage_summary.json', stage_summary); stage_reports.append({'stage_id':stage_id,'responsibility_records_path':resp_rel,'train_state_path':train_state_rel,'checkpoint_last_path':str((Path('train')/stage_id/'checkpoints'/ckpt_name).as_posix()),'record_count_output':int(len(rows)),'loss_mean':_mean_or_zero(losses),'loss_last':float(losses[-1]) if losses else 0.0,'optimization_loss_mean':_mean_or_zero(batch_losses),'optimization_loss_last':float(batch_losses[-1]) if batch_losses else 0.0,'batch_budget':int(batch_budget),'loss_normalization':'effective_trajectory_count'})
    return {'stage_reports':stage_reports,'record_count_input':int(len(materialized_samples)),'record_count_trainable':int(len(examples)),'record_count_output':int(len(final_examples)),'coverage_ratio_trainable':float(len(examples)/max(len(materialized_samples),1)),'skipped_reason_histogram':skipped,'selected_checkpoint_path':stage_reports[-1]['checkpoint_last_path'] if stage_reports else '','unknown_metrics':current_unknown_metrics}
