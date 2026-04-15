from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.banks.responsibility_cache import ResponsibilityCache
from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    build_stage_domain_indices,
    fuse_carrier_frame_logits,
    fuse_carrier_frame_logits_torch,
    load_combined_evidence,
)
from videocutler.ext_stageb_ovvis.banks.text_bank import resolve_text_prototype
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


Record = Dict[str, Any]


@dataclass(frozen=True)
class SoftEMStageConfig:
    stage_id: str
    selected_for_infer: str
    checkpoint_name: str
    responsibility_relpath: str
    train_state_relpath: str
    learning_rate: float
    epochs: int


@dataclass(frozen=True)
class SoftEMConfig:
    dataset_name: str
    trajectory_source_branch: str = "mainline"
    mode: str = "base_then_aug"
    device: str = "cpu"
    seed: int = 0
    smoke: bool = False
    temperature: float = 0.07
    weight_decay: float = 1e-2
    em_subiterations: int = 1
    projector: ProjectorConfig = ProjectorConfig()
    base_epochs: int = 1
    aug_epochs: int = 1
    base_learning_rate: float = 5e-5
    aug_learning_rate: float = 5e-5


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _carrier_artifact_parent(output_root: Path, dataset_name: str, branch: str) -> Path:
    if branch == "mainline":
        return output_root / "carrier_bank" / dataset_name
    if branch == "gt_upper_bound":
        return output_root / "carrier_bank_gt" / dataset_name
    raise ValueError(f"unsupported trajectory_source_branch: {branch}")


def _prepare_examples(
    materialized_samples: Sequence[Record],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    text_records_path = output_root / "text_bank" / "text_prototype_records.jsonl"

    def bump(reason: str) -> None:
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for sample in materialized_samples:
        if not bool(sample.get("sample_valid", False)):
            bump("sample_not_valid_from_phase1")
            continue
        try:
            carrier_vec, frame_vec, combined_vec = load_combined_evidence(
                sample,
                output_root=output_root,
                dataset_name=dataset_name,
                trajectory_source_branch=trajectory_source_branch,
            )
        except Exception:
            bump("missing_frame_evidence")
            continue
        candidate_records = list(sample.get("candidate_text_prototypes", []))
        if not candidate_records:
            bump("empty_candidate_text_prototypes")
            continue
        candidate_ids_known = [int(x) for x in list(sample.get("candidate_ids_known", []))]
        candidate_ids_extra = [int(x) for x in list(sample.get("candidate_ids_extra", []))]
        if len(candidate_ids_known) + len(candidate_ids_extra) != len(candidate_records):
            bump("candidate_id_vector_length_mismatch")
            continue
        observed_set = {int(x) for x in list(sample.get("observed_raw_ids", []))}
        if not candidate_ids_known:
            bump("empty_candidate_ids_known")
            continue
        try:
            candidate_matrix = [resolve_text_prototype(text_records_path, rec) for rec in candidate_records]
        except Exception:
            bump("invalid_text_prototype_locator")
            continue
        examples.append(
            {
                "trajectory_id": str(sample.get("trajectory_id", "")),
                "clip_id": int(sample.get("clip_id", -1)),
                "video_id": int(sample.get("trajectory_record", {}).get("video_id", -1)),
                "observed_raw_ids": sorted(observed_set),
                "candidate_ids_known": candidate_ids_known,
                "candidate_ids_extra": candidate_ids_extra,
                "candidate_matrix": np.asarray(candidate_matrix, dtype=np.float32),
                "carrier_vec": np.asarray(carrier_vec, dtype=np.float32),
                "frame_vec": np.asarray(frame_vec, dtype=np.float32),
                "combined_vec": np.asarray(combined_vec, dtype=np.float32),
                "candidate_records": candidate_records,
            }
        )
    return {"examples": examples, "skipped_reason_histogram": skipped}


def _load_projector_from_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Tuple[Projector, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_payload = dict(checkpoint.get("projector_config", {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(config_payload.get("input_dim", 768)),
            hidden_dim=int(config_payload.get("hidden_dim", 512)),
            output_dim=int(config_payload.get("output_dim", 512)),
            dropout=float(config_payload.get("dropout", 0.0)),
            use_layernorm=bool(config_payload.get("use_layernorm", True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    return projector, checkpoint


def _normalize_mass(mass: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = 0.0
    for key, value in mass.items():
        v = max(0.0, float(value))
        out[str(key)] = v
        total += v
    if total <= 0.0:
        return {"unknown": 1.0}
    return {key: float(value / total) for key, value in out.items()}


def _stage_cfg(config: SoftEMConfig) -> List[SoftEMStageConfig]:
    if config.mode == "base_only":
        return [
            SoftEMStageConfig(
                stage_id="softem_base",
                selected_for_infer="base_only",
                checkpoint_name="softem_base_last.pth",
                responsibility_relpath="train/softem_base/responsibility_records.jsonl",
                train_state_relpath="train/softem_base/train_state.json",
                learning_rate=float(config.base_learning_rate),
                epochs=int(config.base_epochs),
            )
        ]
    if config.mode == "aug_only":
        return [
            SoftEMStageConfig(
                stage_id="softem_aug",
                selected_for_infer="augmented",
                checkpoint_name="softem_aug_last.pth",
                responsibility_relpath="train/softem_aug/responsibility_records.jsonl",
                train_state_relpath="train/softem_aug/train_state.json",
                learning_rate=float(config.aug_learning_rate),
                epochs=int(config.aug_epochs),
            )
        ]
    if config.mode == "base_then_aug":
        return [
            SoftEMStageConfig(
                stage_id="softem_base",
                selected_for_infer="base_only",
                checkpoint_name="softem_base_last.pth",
                responsibility_relpath="train/softem_base/responsibility_records.jsonl",
                train_state_relpath="train/softem_base/train_state.json",
                learning_rate=float(config.base_learning_rate),
                epochs=int(config.base_epochs),
            ),
            SoftEMStageConfig(
                stage_id="softem_aug",
                selected_for_infer="augmented",
                checkpoint_name="softem_aug_last.pth",
                responsibility_relpath="train/softem_aug/responsibility_records.jsonl",
                train_state_relpath="train/softem_aug/train_state.json",
                learning_rate=float(config.aug_learning_rate),
                epochs=int(config.aug_epochs),
            ),
        ]
    raise ValueError(f"unsupported soft-em mode: {config.mode}")


def _stage_domain_for_stage(stage_id: str, example: Mapping[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    return build_stage_domain_indices(
        example.get("candidate_ids_known", []),
        example.get("candidate_ids_extra", []),
        stage_id=stage_id,
    )


def _stage_mass_from_logits(
    *,
    stage_id: str,
    candidate_ids_known: Sequence[int],
    candidate_ids_extra: Sequence[int],
    initial_mass: Mapping[str, float],
    stage_logits: np.ndarray,
    coverage_bonus: float = 0.15,
    extra_penalty: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, float], List[int]]:
    return _stage_mass_from_logits_iterative(
        stage_id=stage_id,
        candidate_ids_known=candidate_ids_known,
        candidate_ids_extra=candidate_ids_extra,
        initial_mass=initial_mass,
        stage_logits=stage_logits,
        coverage_bonus=coverage_bonus,
        extra_penalty=extra_penalty,
        em_subiterations=1,
    )[:3]


def _stage_mass_from_logits_iterative(
    *,
    stage_id: str,
    candidate_ids_known: Sequence[int],
    candidate_ids_extra: Sequence[int],
    initial_mass: Mapping[str, float],
    stage_logits: np.ndarray,
    coverage_bonus: float = 0.15,
    extra_penalty: float = 0.05,
    em_subiterations: int = 1,
) -> Tuple[Dict[str, float], Dict[str, float], List[int], List[Dict[str, Any]]]:
    domain_ids, known_ids, extra_ids = build_stage_domain_indices(
        candidate_ids_known,
        candidate_ids_extra,
        stage_id=stage_id,
    )
    logits = np.asarray(stage_logits, dtype=np.float64)
    if logits.ndim != 1 or int(logits.shape[0]) != len(domain_ids) + 1:
        raise ValueError("stage logits shape mismatch")
    current_mass = _normalize_mass(initial_mass)
    keys: List[str] = ["unknown", *[str(int(raw_id)) for raw_id in domain_ids]]
    model_probs = torch.softmax(torch.tensor(logits[1:], dtype=torch.float64), dim=0).cpu().numpy().astype(np.float64)
    trace: List[Dict[str, Any]] = []
    coverage_bonus_applied_to: List[int] = []
    final_mass = dict(current_mass)
    init_mass = dict(current_mass)

    for subiter_index in range(max(1, int(em_subiterations))):
        refined_init, refined_final, coverage_bonus_applied_to = refine_responsibilities(
            initial_mass=current_mass,
            model_probs=model_probs,
            candidate_ids_known=known_ids,
            candidate_ids_extra=extra_ids,
            stage_id=stage_id,
            coverage_bonus=coverage_bonus,
            extra_penalty=extra_penalty,
        )
        trace.append(
            {
                "subiteration_index": int(subiter_index),
                "r_init": refined_init,
                "r_final": refined_final,
                "coverage_bonus_applied_to": list(coverage_bonus_applied_to),
            }
        )
        init_mass = refined_init
        final_mass = refined_final
        current_mass = refined_final

    # Ensure the returned dictionaries include all expected keys even when the stage domain is empty.
    init_mass = _normalize_mass({key: float(init_mass.get(key, 0.0)) for key in keys})
    final_mass = _normalize_mass({key: float(final_mass.get(key, 0.0)) for key in keys})
    return init_mass, final_mass, sorted(set(coverage_bonus_applied_to)), trace


def _load_proxy_rows(output_root: Path) -> List[Record]:
    proxy_path = output_root / "train" / "prealign" / "proxy_records.jsonl"
    if not proxy_path.is_file():
        raise FileNotFoundError("missing prealign proxy records: train/prealign/proxy_records.jsonl")
    return _load_jsonl(proxy_path)


def _initial_checkpoint_path(output_root: Path, *, mode: str) -> Path:
    if mode == "aug_only":
        aug_path = output_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth"
        if aug_path.is_file():
            return aug_path
    return output_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"


def run_soft_em(
    *,
    output_root: Path,
    materialized_samples: Sequence[Record],
    config: SoftEMConfig,
    audit_callback: Any = None,
) -> Dict[str, Any]:
    if config.dataset_name != "lvvis_train_base":
        raise ValueError("soft-EM implementation currently supports dataset_name=lvvis_train_base only")
    _set_seed(int(config.seed))
    device = torch.device(str(config.device))
    prepared = _prepare_examples(
        materialized_samples,
        output_root=output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
    )
    examples = list(prepared["examples"])
    skipped = dict(prepared["skipped_reason_histogram"])
    if not examples:
        raise RuntimeError("no trainable examples available for soft-EM")
    proxy_rows = _load_proxy_rows(output_root)
    cache = ResponsibilityCache.from_proxy_records(proxy_rows, stage_id="prealign_proxy")
    stage_reports: List[Dict[str, Any]] = []
    current_checkpoint = _initial_checkpoint_path(output_root, mode=config.mode)
    if not current_checkpoint.is_file():
        raise FileNotFoundError(f"missing prealign checkpoint for soft-EM bootstrap: {current_checkpoint}")

    for stage in _stage_cfg(config):
        projector, _ckpt = _load_projector_from_checkpoint(current_checkpoint, device=device)
        projector.train()
        optimizer = torch.optim.AdamW(
            projector.parameters(),
            lr=float(stage.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        losses: List[float] = []
        iteration_index = 0
        stage_trace_sample: List[Dict[str, Any]] = []

        if audit_callback is not None:
            audit_callback(
                {
                    "dataset_name": str(config.dataset_name),
                    "trajectory_source_branch": str(config.trajectory_source_branch),
                    "stage_id": str(stage.stage_id),
                    "snapshot_id": "stage_start",
                    "phase": "stage_start",
                    "output_root": output_root,
                    "materialized_samples": materialized_samples,
                    "projector": projector,
                    "responsibility_cache": cache,
                    "device": str(device),
                    "temperature": float(config.temperature),
                    "seed": int(config.seed),
                    "mode": str(config.mode),
                }
            )

        for _epoch in range(int(stage.epochs)):
            random.Random(int(config.seed)).shuffle(examples)
            for ex in examples:
                ex_tid = str(ex["trajectory_id"])
                init_mass = cache.get_init_mass(ex_tid)
                domain_ids, known_ids, extra_ids = _stage_domain_for_stage(stage.stage_id, ex)
                _, _, logits_known_extra = fuse_carrier_frame_logits_torch(
                    projector=projector,
                    carrier_vec=ex["carrier_vec"],
                    frame_vec=ex["frame_vec"],
                    candidate_matrix=ex["candidate_matrix"],
                    temperature=float(config.temperature),
                )
                stage_candidate_count = len(domain_ids)
                if stage_candidate_count <= 0:
                    raise RuntimeError(f"empty candidate domain for stage {stage.stage_id}")
                stage_logits_candidates = logits_known_extra[:stage_candidate_count]
                stage_logits = torch.cat([torch.zeros(1, device=device, dtype=torch.float32), stage_logits_candidates], dim=0)
                init_domain = {"unknown": float(init_mass.get("unknown", 1.0))}
                for raw_id in domain_ids:
                    init_domain[str(int(raw_id))] = float(init_mass.get(str(int(raw_id)), 0.0))
                refined_init, refined_final, coverage_bonus_applied_to, trace = _stage_mass_from_logits_iterative(
                    stage_id=stage.stage_id,
                    candidate_ids_known=ex["candidate_ids_known"],
                    candidate_ids_extra=ex["candidate_ids_extra"],
                    initial_mass=init_domain,
                    stage_logits=stage_logits.detach().cpu().numpy(),
                    em_subiterations=max(1, int(config.em_subiterations)),
                )
                if not stage_trace_sample:
                    stage_trace_sample = list(trace)
                target = torch.tensor(
                    [refined_final["unknown"], *[refined_final[str(int(raw_id))] for raw_id in domain_ids]],
                    device=device,
                    dtype=torch.float32,
                )
                loss = -(target * torch.log_softmax(stage_logits, dim=0)).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                row = {
                    "dataset_name": str(config.dataset_name),
                    "clip_id": int(ex["clip_id"]),
                    "video_id": int(ex["video_id"]),
                    "trajectory_id": ex_tid,
                    "candidate_ids_known": [int(x) for x in ex["candidate_ids_known"]],
                    "candidate_ids_extra": [int(x) for x in ex["candidate_ids_extra"]],
                    "unknown_slot": "unknown",
                    "r_init": refined_init,
                    "r_final": refined_final,
                    "coverage_bonus_applied_to": coverage_bonus_applied_to,
                    "em_subiterations": int(max(1, int(config.em_subiterations))),
                    "em_subiteration_count": int(len(trace)),
                    "iteration_index": int(iteration_index),
                    "join_key": str(ex_tid),
                }
                cache.update(ex_tid, row)
                iteration_index += 1
            if audit_callback is not None:
                audit_callback(
                    {
                        "dataset_name": str(config.dataset_name),
                        "trajectory_source_branch": str(config.trajectory_source_branch),
                        "stage_id": str(stage.stage_id),
                        "snapshot_id": f"epoch_{int(_epoch) + 1:03d}",
                        "phase": "epoch_end",
                        "output_root": output_root,
                        "materialized_samples": materialized_samples,
                        "projector": projector,
                        "responsibility_cache": cache,
                        "device": str(device),
                        "temperature": float(config.temperature),
                        "seed": int(config.seed),
                        "mode": str(config.mode),
                    }
                )

        stage_dir = output_root / "train" / stage.stage_id
        ckpt_dir = stage_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        responsibility_path = output_root / stage.responsibility_relpath
        train_state_path = output_root / stage.train_state_relpath
        checkpoint_path = ckpt_dir / stage.checkpoint_name
        cache.stage_id = stage.stage_id
        cache.write_jsonl(responsibility_path)
        train_state = {
            "stage_id": stage.stage_id,
            "epoch": int(stage.epochs),
            "selected_for_infer": stage.selected_for_infer,
            "selected_for_infer_authority": "explicit_train_state_field",
            "checkpoint_last": str((Path("train") / stage.stage_id / "checkpoints" / stage.checkpoint_name).as_posix()),
            "checkpoint_selected": str((Path("train") / stage.stage_id / "checkpoints" / stage.checkpoint_name).as_posix()),
        }
        _write_json(train_state_path, train_state)
        torch.save(
            {
                "stage_id": stage.stage_id,
                "epoch": int(stage.epochs),
                "projector_state_dict": projector.state_dict(),
                "projector_config": {
                    "input_dim": int(config.projector.input_dim),
                    "hidden_dim": int(config.projector.hidden_dim),
                    "output_dim": int(config.projector.output_dim),
                    "dropout": float(config.projector.dropout),
                    "use_layernorm": bool(config.projector.use_layernorm),
                },
                "seed": int(config.seed),
                "mode": str(config.mode),
            },
            checkpoint_path,
        )
        stage_reports.append(
                {
                    "stage_id": stage.stage_id,
                    "responsibility_records_path": stage.responsibility_relpath,
                    "train_state_path": stage.train_state_relpath,
                    "checkpoint_last_path": str((Path("train") / stage.stage_id / "checkpoints" / stage.checkpoint_name).as_posix()),
                    "record_count_output": int(len(cache.by_trajectory_id)),
                    "loss_mean": float(np.mean(losses)) if losses else 0.0,
                    "loss_last": float(losses[-1]) if losses else 0.0,
                    "em_subiterations": int(max(1, int(config.em_subiterations))),
                    "subiteration_trace_sample": stage_trace_sample,
                }
        )
        current_checkpoint = checkpoint_path

        if audit_callback is not None:
            audit_callback(
                {
                    "dataset_name": str(config.dataset_name),
                    "trajectory_source_branch": str(config.trajectory_source_branch),
                    "stage_id": str(stage.stage_id),
                    "snapshot_id": "stage_end",
                    "phase": "stage_end",
                    "output_root": output_root,
                    "materialized_samples": materialized_samples,
                    "projector": projector,
                    "responsibility_cache": cache,
                    "device": str(device),
                    "temperature": float(config.temperature),
                    "seed": int(config.seed),
                    "mode": str(config.mode),
                    "train_state": train_state,
                }
            )

    selected_checkpoint_path = stage_reports[-1]["checkpoint_last_path"] if stage_reports else ""
    return {
        "stage_reports": stage_reports,
        "record_count_input": int(len(materialized_samples)),
        "record_count_trainable": int(len(examples)),
        "record_count_output": int(len(cache.by_trajectory_id)),
        "coverage_ratio_trainable": float(len(examples) / float(len(materialized_samples))) if materialized_samples else 0.0,
        "skipped_reason_histogram": skipped,
        "selected_checkpoint_path": selected_checkpoint_path,
    }
