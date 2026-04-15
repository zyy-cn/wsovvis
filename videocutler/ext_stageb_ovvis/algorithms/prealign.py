from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    fuse_carrier_frame_logits,
    fuse_carrier_frame_logits_torch,
    load_combined_evidence,
    load_text_vocab,
    observed_mass_loss,
)
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


Record = Dict[str, Any]


@dataclass(frozen=True)
class PrealignConfig:
    dataset_name: str
    trajectory_source_branch: str = "mainline"
    device: str = "cpu"
    seed: int = 0
    smoke: bool = False
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    temperature: float = 0.07
    projector: ProjectorConfig = ProjectorConfig()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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

        examples.append(
            {
                "trajectory_id": str(sample["trajectory_id"]),
                "clip_id": int(sample["clip_id"]),
                "video_id": int(sample["trajectory_record"]["video_id"]),
                "observed_raw_ids": sorted({int(x) for x in list(sample.get("observed_raw_ids", []))}),
                "carrier_vec": np.asarray(carrier_vec, dtype=np.float32),
                "frame_vec": np.asarray(frame_vec, dtype=np.float32),
                "combined_vec": np.asarray(combined_vec, dtype=np.float32),
            }
        )
    return {"examples": examples, "skipped_reason_histogram": skipped}


def train_prealign(
    *,
    output_root: Path,
    materialized_samples: Sequence[Record],
    config: PrealignConfig,
    audit_callback: Any = None,
) -> Dict[str, Any]:
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
    total_samples = len(materialized_samples)
    if not examples:
        raise RuntimeError("no valid trainable prealign examples after phase-1 filtering")
    text_vocab_ids, text_vocab_records, text_vocab_matrix = load_text_vocab(output_root)
    text_vocab_matrix_t = torch.from_numpy(text_vocab_matrix.astype(np.float32))

    projector = Projector(config.projector).to(device)
    projector.train()
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    losses: List[float] = []
    text_candidate_matrix = np.asarray(text_vocab_matrix, dtype=np.float32)

    if audit_callback is not None:
        audit_callback(
            {
                "dataset_name": str(config.dataset_name),
                "trajectory_source_branch": str(config.trajectory_source_branch),
                "stage_id": "prealign",
                "snapshot_id": "stage_start",
                "phase": "stage_start",
                "output_root": output_root,
                "materialized_samples": materialized_samples,
                "projector": projector,
                "device": str(device),
                "temperature": float(config.temperature),
                "seed": int(config.seed),
                "mode": "prealign",
            }
        )

    for epoch_index in range(int(config.epochs)):
        random.Random(int(config.seed)).shuffle(examples)
        for ex in examples:
            _, _, logits = fuse_carrier_frame_logits_torch(
                projector=projector,
                carrier_vec=ex["carrier_vec"],
                frame_vec=ex["frame_vec"],
                candidate_matrix=text_candidate_matrix,
                temperature=float(config.temperature),
            )
            observed_raw_ids = [int(x) for x in ex["observed_raw_ids"]]
            positive = [idx for idx, raw_id in enumerate(text_vocab_ids) if int(raw_id) in observed_raw_ids]
            if not positive:
                raise RuntimeError(f"no observed raw ids found in text vocab for trajectory {ex['trajectory_id']}")
            loss = observed_mass_loss(logits, positive, unknown_logit=torch.zeros((), device=device, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if audit_callback is not None:
            audit_callback(
                {
                    "dataset_name": str(config.dataset_name),
                    "trajectory_source_branch": str(config.trajectory_source_branch),
                    "stage_id": "prealign",
                    "snapshot_id": f"epoch_{int(epoch_index) + 1:03d}",
                    "phase": "epoch_end",
                    "output_root": output_root,
                    "materialized_samples": materialized_samples,
                    "projector": projector,
                    "device": str(device),
                    "temperature": float(config.temperature),
                    "seed": int(config.seed),
                    "mode": "prealign",
                }
            )

    projector.eval()
    proxy_rows: List[Record] = []
    with torch.no_grad():
        for ex in sorted(examples, key=lambda row: str(row["trajectory_id"])):
            _, _, logits_np = fuse_carrier_frame_logits(
                projector=projector,
                carrier_vec=ex["carrier_vec"],
                frame_vec=ex["frame_vec"],
                candidate_matrix=text_candidate_matrix,
                temperature=float(config.temperature),
            )
            logits = torch.from_numpy(np.asarray(logits_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
            logits_full = torch.cat([torch.zeros(1, device=device, dtype=torch.float32), logits], dim=0)
            probs = torch.softmax(logits_full, dim=0).detach().cpu().numpy().astype(np.float64)
            proxy_mass = {"unknown": float(probs[0])}
            for idx, raw_id in enumerate(text_vocab_ids):
                proxy_mass[str(int(raw_id))] = float(probs[idx + 1])
            proxy_rows.append(
                {
                    "dataset_name": str(config.dataset_name),
                    "clip_id": int(ex["clip_id"]),
                    "video_id": int(ex["video_id"]),
                    "trajectory_id": str(ex["trajectory_id"]),
                    "observed_raw_ids": [int(x) for x in ex["observed_raw_ids"]],
                    "proxy_mass": proxy_mass,
                    "join_key": str(ex["trajectory_id"]),
                }
            )

    train_dir = output_root / "train" / "prealign"
    ckpt_dir = train_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = train_dir / "proxy_records.jsonl"
    train_state_path = train_dir / "train_state.json"
    ckpt_last_path = ckpt_dir / "prealign_last.pth"
    torch.save(
        {
            "stage_id": "prealign",
            "epoch": int(config.epochs),
            "projector_state_dict": projector.state_dict(),
            "projector_config": {
                "input_dim": int(config.projector.input_dim),
                "hidden_dim": int(config.projector.hidden_dim),
                "output_dim": int(config.projector.output_dim),
                "dropout": float(config.projector.dropout),
                "use_layernorm": bool(config.projector.use_layernorm),
            },
            "seed": int(config.seed),
        },
        ckpt_last_path,
    )
    _write_jsonl(proxy_path, proxy_rows)
    train_state = {
        "stage_id": "prealign",
        "epoch": int(config.epochs),
        "selected_for_infer": "prealign_only",
        "selected_for_infer_authority": "explicit_train_state_field",
        "checkpoint_last": "train/prealign/checkpoints/prealign_last.pth",
        "checkpoint_selected": "train/prealign/checkpoints/prealign_last.pth",
    }
    _write_json(train_state_path, train_state)

    if audit_callback is not None:
        audit_callback(
            {
                "dataset_name": str(config.dataset_name),
                "trajectory_source_branch": str(config.trajectory_source_branch),
                "stage_id": "prealign",
                "snapshot_id": "stage_end",
                "phase": "stage_end",
                "output_root": output_root,
                "materialized_samples": materialized_samples,
                "projector": projector,
                "device": str(device),
                "temperature": float(config.temperature),
                "seed": int(config.seed),
                "mode": "prealign",
                "train_state": train_state,
            }
        )

    return {
        "proxy_records_path": proxy_path,
        "train_state_path": train_state_path,
        "checkpoint_last_path": ckpt_last_path,
        "record_count_input": int(total_samples),
        "record_count_trainable": int(len(examples)),
        "record_count_output": int(len(proxy_rows)),
        "coverage_ratio_trainable": float(len(examples) / float(total_samples)) if total_samples > 0 else 0.0,
        "skipped_reason_histogram": skipped,
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "loss_last": float(losses[-1]) if losses else 0.0,
        "train_state": train_state,
    }
