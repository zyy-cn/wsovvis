from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import _load_or_generate_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import Phase1MaterializationConfig, materialize_phase1_training_samples
from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_projector_bundle
from videocutler.ext_stageb_ovvis.audit._matrix_vocab_scoring import build_carrier_matrix_pack, compute_rank_metrics_batched
from videocutler.ext_stageb_ovvis.banks.text_bank import load_text_vocab

Record = Dict[str, Any]
_STAGE_TO_CHECKPOINT = {
    "prealign": ("train", "prealign", "checkpoints", "prealign_last.pth"),
    "softem_base": ("train", "softem_base", "checkpoints", "softem_base_last.pth"),
    "softem_aug": ("train", "softem_aug", "checkpoints", "softem_aug_last.pth"),
}
_STAGE_ORDER = ("prealign", "softem_base", "softem_aug")


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _unique_ints(values: Sequence[Any]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for value in values:
        item = _as_int(value)
        if item is None or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _checkpoint_path(output_root: Path, stage: str) -> Path:
    rel = _STAGE_TO_CHECKPOINT[stage]
    return output_root.joinpath(*rel)


def _split_label(*, gt_class_id: int, observed_raw_ids: Sequence[int]) -> str:
    return "observed" if int(gt_class_id) in {int(x) for x in observed_raw_ids} else "dropped"


def _rank_and_top1_from_logits(logits: np.ndarray | Sequence[float], *, gt_index: int) -> Tuple[int, float, bool]:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")
    if gt_index < 0 or gt_index >= int(logits.shape[0]):
        raise IndexError("gt_index out of range")
    gt_logit = float(logits[int(gt_index)])
    rank = 1 + int(np.sum(np.asarray(logits, dtype=np.float64) > gt_logit))
    denom = max(1, int(logits.shape[0]) - 1)
    normalized_rank = float((rank - 1) / denom)
    top1 = bool(int(np.argmax(logits)) == int(gt_index))
    return rank, normalized_rank, top1


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, total_prediction_count: int) -> Dict[str, Any]:
    matched_count = len(rows)
    summary: Dict[str, Any] = {
        "total_prediction_count": int(total_prediction_count),
        "matched_prediction_count": int(matched_count),
        "match_rate": float(matched_count / total_prediction_count) if total_prediction_count else 0.0,
        "observed": {
            "mean_normalized_gt_rank": None,
            "gt_top1_hit_rate": None,
        },
        "dropped": {
            "mean_normalized_gt_rank": None,
            "gt_top1_hit_rate": None,
        },
    }
    for split in ("observed", "dropped"):
        split_rows = [row for row in rows if str(row.get("supervision_split", "")) == split]
        if not split_rows:
            continue
        nranks = [float(row["normalized_gt_rank"]) for row in split_rows]
        top1s = [1.0 if bool(row["gt_is_top1"]) else 0.0 for row in split_rows]
        summary[split] = {
            "mean_normalized_gt_rank": float(np.mean(np.asarray(nranks, dtype=np.float64))),
            "gt_top1_hit_rate": float(np.mean(np.asarray(top1s, dtype=np.float64))),
        }
    return summary


def _build_clip_summary(rows: Sequence[Mapping[str, Any]]) -> List[Record]:
    by_key: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("supervision_split", "")), int(row.get("clip_id", -1)))
        by_key.setdefault(key, []).append(row)
    out: List[Record] = []
    for (split, clip_id), group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        nranks = [float(row["normalized_gt_rank"]) for row in group]
        top1s = [1.0 if bool(row["gt_is_top1"]) else 0.0 for row in group]
        out.append(
            {
                "supervision_split": split,
                "clip_id": int(clip_id),
                "matched_prediction_count": int(len(group)),
                "mean_normalized_gt_rank": float(np.mean(np.asarray(nranks, dtype=np.float64))),
                "gt_top1_hit_rate": float(np.mean(np.asarray(top1s, dtype=np.float64))),
            }
        )
    return out


@dataclass(frozen=True)
class TrainGtRankAuditConfig:
    output_root: Path
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    stage: str = "all"
    device: str = "cpu"
    logit_chunk_size: int = 256
    generate_sidecars: bool = True


def _iter_stages(stage: str) -> List[str]:
    if stage == "all":
        return list(_STAGE_ORDER)
    if stage not in _STAGE_TO_CHECKPOINT:
        raise ValueError(f"unsupported stage: {stage}")
    return [stage]


def _materialize_context(output_root: Path, dataset_name: str, trajectory_source_branch: str, generate_sidecars: bool) -> Tuple[List[Record], Dict[str, Record]]:
    materialized = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=str(dataset_name),
            trajectory_source_branch=str(trajectory_source_branch),
            smoke=False,
        ),
    )
    samples = [dict(x) for x in materialized["samples"]]
    clip_ids = sorted({_as_int((sample.get("trajectory_record") or {}).get("clip_id")) for sample in samples if _as_int((sample.get("trajectory_record") or {}).get("clip_id")) is not None})
    sidecar_lookup = _load_or_generate_gt_sidecar_lookup(
        output_root=output_root,
        dataset_name=dataset_name,
        clip_ids=[int(x) for x in clip_ids],
        generate_sidecars=bool(generate_sidecars),
    )
    return samples, sidecar_lookup


def _score_stage_rows(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    stage: str,
    device: torch.device,
    logit_chunk_size: int,
    samples: Sequence[Mapping[str, Any]],
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    checkpoint_path = _checkpoint_path(output_root, stage)
    if not checkpoint_path.is_file():
        return {
            "dataset_name": dataset_name,
            "stage": stage,
            "stage_status": "STAGE_NOT_PRESENT",
            "checkpoint_path": str(checkpoint_path),
            "note": "checkpoint_missing",
        }

    bundle = load_projector_bundle(checkpoint_path, device=device)
    text_vocab_ids, _text_records, text_matrix = load_text_vocab(output_root)
    raw_id_to_index = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}

    ledger_rows: List[Record] = []
    total_prediction_count = len(samples)
    score_rows: List[Record] = []
    score_gt_indices: List[int] = []
    score_meta: List[Record] = []
    for sample in samples:
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        gt_record = dict(gt_sidecar_lookup.get(trajectory_id, {}))
        gt_class_id = _as_int(gt_record.get("matched_gt_class_id"))
        if gt_class_id is None:
            continue
        gt_index = raw_id_to_index.get(int(gt_class_id))
        if gt_index is None:
            continue
        observed_raw_ids = _unique_ints(sample.get("observed_raw_ids", []))
        supervision_split = _split_label(gt_class_id=int(gt_class_id), observed_raw_ids=observed_raw_ids)
        score_rows.append(dict(sample))
        score_gt_indices.append(int(gt_index))
        score_meta.append(
            {
                "stage": stage,
                "trajectory_id": trajectory_id,
                "clip_id": int((sample.get("trajectory_record") or {}).get("clip_id", -1)),
                "gt_class_id": int(gt_class_id),
                "observed_raw_ids": observed_raw_ids,
                "supervision_split": supervision_split,
            }
        )

    if score_rows:
        metrics_batch_size = max(1, int(logit_chunk_size))
        pack = build_carrier_matrix_pack(
            score_rows,
            output_root=output_root,
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
        )
        metrics = compute_rank_metrics_batched(
            carrier_matrix=np.asarray(pack["carrier_matrix"], dtype=np.float32),
            projector=bundle.projector,
            candidate_matrix=np.asarray(text_matrix, dtype=np.float32),
            temperature=float(bundle.temperature),
            gt_indices=score_gt_indices,
            batch_size=metrics_batch_size,
        )
        for row_idx, meta in enumerate(score_meta):
            ledger_rows.append(
                {
                    **meta,
                    "gt_rank": int(metrics["rank"][row_idx]),
                    "normalized_gt_rank": float(metrics["normalized_rank"][row_idx]),
                    "gt_is_top1": bool(metrics["top1"][row_idx]),
                    "audit_status": "ok",
                }
            )

    ok_rows = [row for row in ledger_rows if str(row.get("audit_status")) == "ok"]
    summary = _aggregate_rows(ok_rows, total_prediction_count=total_prediction_count)
    clip_summary = _build_clip_summary(ok_rows)
    audit_dir = output_root / "train" / "audit" / "gt_rank_train" / stage
    ledger_path = audit_dir / "ledger.jsonl"
    clip_summary_path = audit_dir / "clip_summary.jsonl"
    summary_path = audit_dir / "summary.json"
    _write_jsonl(ledger_path, ledger_rows)
    _write_jsonl(clip_summary_path, clip_summary)
    payload = {
        "dataset_name": dataset_name,
        "stage": stage,
        "stage_status": "STAGE_PRESENT",
        "class_space_size": int(len(text_vocab_ids)),
        "checkpoint_path": str(checkpoint_path),
        "ledger_path": str(ledger_path),
        "clip_summary_path": str(clip_summary_path),
        **summary,
    }
    _write_json(summary_path, payload)
    return payload


def run_train_gt_rank_audit(config: TrainGtRankAuditConfig) -> Dict[str, Any]:
    output_root = Path(config.output_root).expanduser().resolve()
    if str(config.dataset_name) != "lvvis_train_base":
        raise ValueError("train GT rank audit currently supports dataset_name=lvvis_train_base only")
    samples, gt_sidecar_lookup = _materialize_context(
        output_root,
        str(config.dataset_name),
        str(config.trajectory_source_branch),
        bool(config.generate_sidecars),
    )
    device = torch.device(str(config.device))
    stages: Dict[str, Any] = {}
    for stage in _iter_stages(str(config.stage)):
        stages[stage] = _score_stage_rows(
            output_root=output_root,
            dataset_name=str(config.dataset_name),
            trajectory_source_branch=str(config.trajectory_source_branch),
            stage=stage,
            device=device,
            logit_chunk_size=int(config.logit_chunk_size),
            samples=samples,
            gt_sidecar_lookup=gt_sidecar_lookup,
        )
    summary_path = output_root / "train" / "audit" / "gt_rank_train" / "summary.json"
    payload = {
        "dataset_name": str(config.dataset_name),
        "output_root": str(output_root),
        "stages": stages,
    }
    _write_json(summary_path, payload)
    return payload
