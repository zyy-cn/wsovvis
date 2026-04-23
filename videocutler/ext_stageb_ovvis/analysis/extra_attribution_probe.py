from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_carrier_evidence, load_text_vocab
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import (
    _compute_t_dis,
    _load_reservoir_checkpoint,
    _normalize_np,
    _project_text_matrix,
    _unknown_score,
)
from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import _candidate_domain_from_sample
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import _canonical_sidecar_gt_raw_id
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_base_and_novel_raw_ids
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)

Record = Dict[str, Any]
TRAIN_SPLIT_ORDER: Tuple[str, ...] = ("base_observed", "base_unobserved")
VAL_SPLIT_ORDER: Tuple[str, ...] = ("base", "novel")
STAGE_ORDER: Tuple[str, ...] = ("prealign", "softem_base", "softem_aug")
TOPK_VALUES: Tuple[int, ...] = (1, 5, 10)
DEFAULT_TOP_FAILURE_CASES = 64
EXTRA_APPLICABLE_STAGES = frozenset({"softem_aug"})


@dataclass(frozen=True)
class ExtraAttributionProbeConfig:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str = "lvvis_val"
    trajectory_source_branch: str = "mainline"
    device: str = "cpu"
    smoke: bool = False
    smoke_max_trajectories: int = 128
    subset_fraction: Optional[float] = None
    stage_scope: Tuple[str, ...] = STAGE_ORDER
    batch_size: int = 512
    extra_candidate_topk: int = 5
    top_failure_cases: int = DEFAULT_TOP_FAILURE_CASES
    output_dir: Optional[Path] = None
    sidecar_root: Optional[Path] = None
    show_progress: bool = True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _make_iter_progress(iterable, *, enabled: bool, desc: str):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, desc=str(desc), dynamic_ncols=True)
    return iterable


def _split_order_for_dataset(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name == "lvvis_train_base":
        return TRAIN_SPLIT_ORDER
    if dataset_name == "lvvis_val":
        return VAL_SPLIT_ORDER
    raise ValueError(f"unsupported dataset_name: {dataset_name}")


def _default_output_dir(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "extra_attribution_probe" / dataset_name


def _resolve_output_dir(config: ExtraAttributionProbeConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() if config.output_dir is not None else _default_output_dir(config.run_root, config.dataset_name)


def _sidecar_root(config: ExtraAttributionProbeConfig) -> Path:
    return Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root is not None else Path(config.run_root).expanduser().resolve()


def _materialize_valid_samples(config: ExtraAttributionProbeConfig) -> Dict[str, Any]:
    return materialize_phase1_training_samples(
        Path(config.runtime_output_root).expanduser().resolve(),
        Phase1MaterializationConfig(
            dataset_name=str(config.dataset_name),
            trajectory_source_branch=str(config.trajectory_source_branch),
            smoke=bool(config.smoke),
            smoke_max_trajectories=int(config.smoke_max_trajectories),
            subset_fraction=None if config.subset_fraction is None else float(config.subset_fraction),
        ),
    )


def _prepare_probe_examples(
    materialized_samples: Sequence[Record],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Dict[str, Any]:
    examples: List[Record] = []
    skipped: Dict[str, int] = {}

    def bump(reason: str) -> None:
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for sample in materialized_samples:
        if not bool(sample.get("sample_valid", False)):
            bump("sample_not_valid_from_phase1")
            continue
        try:
            carrier_vec = load_carrier_evidence(
                sample,
                output_root=output_root,
                dataset_name=dataset_name,
                trajectory_source_branch=trajectory_source_branch,
            )
        except Exception:
            bump("missing_carrier_evidence")
            continue
        candidate_ids_known = [int(x) for x in list(sample.get("candidate_ids_known", []))]
        candidate_ids_extra = [int(x) for x in list(sample.get("candidate_ids_extra", []))]
        if not candidate_ids_known:
            bump("empty_candidate_ids_known")
            continue
        examples.append(
            {
                "trajectory_id": str(sample.get("trajectory_id", "")),
                "clip_id": int(sample.get("clip_id", -1)),
                "video_id": int(sample.get("trajectory_record", {}).get("video_id", -1)),
                "observed_raw_ids": sorted({int(x) for x in list(sample.get("observed_raw_ids", []))}),
                "candidate_ids_known": candidate_ids_known,
                "candidate_ids_extra": candidate_ids_extra,
                "candidate_proposal_source": str(sample.get("candidate_proposal_source", sample.get("candidate_source", ""))),
                "candidate_source": str(sample.get("candidate_source", "")),
                "carrier_vec": np.asarray(carrier_vec, dtype=np.float32),
            }
        )
    return {"examples": examples, "skipped_reason_histogram": skipped}


def _default_checkpoint_path(run_root: Path, stage_id: str) -> Path:
    if stage_id == "prealign":
        return run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if stage_id == "softem_base":
        return run_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth"
    if stage_id == "softem_aug":
        return run_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth"
    raise ValueError(f"unsupported stage_id: {stage_id}")


def _score_batches(
    *,
    examples: Sequence[Mapping[str, Any]],
    projector: torch.nn.Module,
    text_vocab_matrix: np.ndarray,
    unknown_prototype: torch.nn.Parameter,
    temperature: torch.Tensor,
    device: torch.device,
    batch_size: int,
    show_progress: bool,
    stage_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        text_proj = _project_text_matrix(projector, np.asarray(text_vocab_matrix, dtype=np.float32), device=device)
        logits_parts: List[np.ndarray] = []
        unknown_parts: List[np.ndarray] = []
        total = max(1, len(examples))
        for start in _make_iter_progress(range(0, total, max(1, int(batch_size))), enabled=show_progress, desc=f"extra-probe[{stage_id}] score"):
            end = min(total, int(start) + max(1, int(batch_size)))
            batch_examples = list(examples[int(start):int(end)])
            if not batch_examples:
                continue
            carrier_np = np.stack([_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32)) for ex in batch_examples], axis=0)
            carrier_t = torch.from_numpy(carrier_np).to(device=device, dtype=torch.float32)
            logits_vocab_t = torch.matmul(F.normalize(carrier_t, p=2.0, dim=-1), text_proj.t()) / temperature
            logits_unknown_t = _unknown_score(carrier_t, unknown_prototype, temperature).reshape(-1)
            logits_parts.append(np.asarray(logits_vocab_t.detach().cpu().numpy(), dtype=np.float64))
            unknown_parts.append(np.asarray(logits_unknown_t.detach().cpu().numpy(), dtype=np.float64))
    logits_vocab = np.concatenate(logits_parts, axis=0) if logits_parts else np.zeros((0, int(text_vocab_matrix.shape[0])), dtype=np.float64)
    logits_unknown = np.concatenate(unknown_parts, axis=0) if unknown_parts else np.zeros((0,), dtype=np.float64)
    return logits_vocab, logits_unknown


def _rank_of_index(sort_order: np.ndarray, target_index: int) -> int:
    order = np.empty_like(sort_order)
    order[sort_order] = np.arange(len(sort_order), dtype=np.int64)
    return int(order[int(target_index)]) + 1


def _subset_rank(sort_order: np.ndarray, subset_indices: Sequence[int], target_index: int) -> Optional[int]:
    subset = [int(x) for x in subset_indices]
    if int(target_index) not in subset:
        return None
    ordered = [int(idx) for idx in sort_order.tolist() if int(idx) in set(subset)]
    if not ordered:
        return None
    return int(ordered.index(int(target_index))) + 1


def _safe_mass(probs_vocab: np.ndarray, index_list: Sequence[int]) -> float:
    if not index_list:
        return 0.0
    idx = np.asarray([int(x) for x in index_list], dtype=np.int64)
    if idx.size == 0:
        return 0.0
    return float(np.sum(probs_vocab[idx]))


def _topk_contains(sort_order: np.ndarray, target_index: Optional[int], *, k: int) -> bool:
    if target_index is None:
        return False
    return int(target_index) in set(int(x) for x in sort_order[: min(len(sort_order), int(k))].tolist())


def _stage_extra_ids(example: Mapping[str, Any], *, stage_id: str) -> Tuple[List[int], bool]:
    if str(stage_id) not in EXTRA_APPLICABLE_STAGES:
        return [], False
    known_ids, extra_ids, _union, extra_authoritative = _candidate_domain_from_sample(example)
    known_set = {int(x) for x in known_ids}
    filtered = [int(x) for x in extra_ids if int(x) not in known_set]
    return filtered, bool(extra_authoritative and len(filtered) > 0)


def _failure_bucket(
    *,
    stage_id: str,
    extra_authoritative: bool,
    gt_raw_id: Optional[int],
    gt_in_vocab: bool,
    gt_in_extra: bool,
    winner_domain: str,
    winner_is_gt_extra: bool,
) -> str:
    if str(stage_id) not in EXTRA_APPLICABLE_STAGES:
        return "extra_not_applicable_stage"
    if gt_raw_id is None or not bool(gt_in_vocab):
        return "gt_missing_from_vocab_or_sidecar"
    if not bool(extra_authoritative) or not bool(gt_in_extra):
        return "gt_not_in_extra_candidate"
    if bool(winner_is_gt_extra):
        return "success_gt_extra_wins"
    if str(winner_domain) == "extra":
        return "gt_in_extra_but_wrong_extra_wins"
    if str(winner_domain) == "other_nonYprime":
        return "gt_in_extra_but_nonextra_nonYprime_wins"
    if str(winner_domain) == "Yprime":
        return "gt_in_extra_but_Yprime_still_wins"
    if str(winner_domain) == "unknown":
        return "gt_in_extra_but_unknown_wins"
    return "gt_in_extra_other_failure"


def _build_row_metrics(
    *,
    stage_id: str,
    examples: Sequence[Mapping[str, Any]],
    logits_vocab: np.ndarray,
    logits_unknown: np.ndarray,
    text_vocab_ids: Sequence[int],
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    dataset_name: str,
    base_vocab_ids: Sequence[int],
    extra_candidate_topk: int,
) -> List[Record]:
    vocab_ids = [int(x) for x in text_vocab_ids]
    vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(vocab_ids)}
    denom_rank = float(max(len(vocab_ids) - 1, 1))
    rows: List[Record] = []
    base_vocab_set = {int(x) for x in base_vocab_ids}
    for row_index, example in enumerate(examples):
        trajectory_id = str(example.get("trajectory_id", "")).strip()
        sidecar = dict(sidecar_lookup.get(trajectory_id, {})) if trajectory_id else {}
        gt_raw_id = _canonical_sidecar_gt_raw_id(sidecar) if sidecar else None
        gt_raw_id_int = int(gt_raw_id) if gt_raw_id is not None else None
        observed_raw_ids = [int(x) for x in list(example.get("observed_raw_ids", []))]
        split_label = None
        if gt_raw_id_int is not None:
            split_label = _all_gt_split_label(
                dataset_name=str(dataset_name),
                gt_raw_id=int(gt_raw_id_int),
                observed_raw_ids=observed_raw_ids,
                base_vocab_ids=base_vocab_set,
            )
        known_ids = [int(x) for x in list(example.get("candidate_ids_known", []))]
        extra_ids, extra_authoritative = _stage_extra_ids(example, stage_id=stage_id)
        known_set = {int(x) for x in known_ids if int(x) in vocab_index}
        extra_set = {int(x) for x in extra_ids if int(x) in vocab_index and int(x) not in known_set}
        known_index = [int(vocab_index[x]) for x in known_ids if int(x) in vocab_index]
        extra_index = [int(vocab_index[x]) for x in extra_ids if int(x) in vocab_index and int(x) not in known_set]
        known_mask = np.zeros((len(vocab_ids),), dtype=bool)
        extra_mask = np.zeros((len(vocab_ids),), dtype=bool)
        if known_index:
            known_mask[np.asarray(known_index, dtype=np.int64)] = True
        if extra_index:
            extra_mask[np.asarray(extra_index, dtype=np.int64)] = True
        other_mask = (~known_mask) & (~extra_mask)
        vocab_logits = np.asarray(logits_vocab[row_index], dtype=np.float64)
        unknown_logit = float(logits_unknown[row_index])
        max_logit = float(max(float(np.max(vocab_logits)) if vocab_logits.size else -np.inf, unknown_logit))
        exp_vocab = np.exp(vocab_logits - max_logit)
        exp_unknown = float(np.exp(unknown_logit - max_logit))
        denom = float(np.sum(exp_vocab) + exp_unknown)
        probs_vocab = exp_vocab / max(denom, 1e-12)
        prob_unknown = exp_unknown / max(denom, 1e-12)
        sort_order = np.argsort(-vocab_logits, kind="stable")
        gt_in_vocab = gt_raw_id_int is not None and int(gt_raw_id_int) in vocab_index
        gt_index = int(vocab_index[int(gt_raw_id_int)]) if gt_in_vocab else None
        gt_in_known = bool(gt_raw_id_int in known_set) if gt_raw_id_int is not None else False
        gt_in_extra = bool(gt_raw_id_int in extra_set) if gt_raw_id_int is not None else False
        gt_in_extra_topk_candidate = bool(gt_raw_id_int in {int(x) for x in extra_ids[: max(1, int(extra_candidate_topk))]}) if gt_raw_id_int is not None else False
        gt_in_other_nonYprime = bool(gt_in_vocab and (not gt_in_known) and (not gt_in_extra))
        mass_on_Yprime = _safe_mass(probs_vocab, known_index)
        mass_on_extra = _safe_mass(probs_vocab, extra_index)
        mass_on_other_nonYprime = float(np.sum(probs_vocab[other_mask])) if other_mask.any() else 0.0
        mass_on_gt_class = float(probs_vocab[gt_index]) if gt_index is not None else 0.0
        mass_on_gt_extra = float(probs_vocab[gt_index]) if gt_in_extra and gt_index is not None else 0.0
        mass_on_wrong_extra = max(0.0, float(mass_on_extra - mass_on_gt_extra))
        best_vocab_idx = int(np.argmax(vocab_logits)) if vocab_logits.size else -1
        best_vocab_logit = float(vocab_logits[best_vocab_idx]) if best_vocab_idx >= 0 else float("-inf")
        winner_domain = "unknown"
        winner_raw_id: Optional[int] = None
        winner_is_gt = False
        winner_is_gt_extra = False
        if best_vocab_idx >= 0 and best_vocab_logit >= unknown_logit:
            winner_raw_id = int(vocab_ids[best_vocab_idx])
            if known_mask[best_vocab_idx]:
                winner_domain = "Yprime"
            elif extra_mask[best_vocab_idx]:
                winner_domain = "extra"
            else:
                winner_domain = "other_nonYprime"
            winner_is_gt = bool(gt_index is not None and int(best_vocab_idx) == int(gt_index))
            winner_is_gt_extra = bool(winner_is_gt and gt_in_extra)
        best_gt_rank = _rank_of_index(sort_order, gt_index) if gt_index is not None else None
        best_gt_normalized_rank = (float(best_gt_rank - 1) / denom_rank) if best_gt_rank is not None else None
        gt_extra_rank = _subset_rank(sort_order, extra_index, gt_index) if gt_index is not None else None
        gt_extra_normalized_rank = (
            float(gt_extra_rank - 1) / float(max(len(extra_index) - 1, 1))
            if gt_extra_rank is not None and len(extra_index) > 0
            else None
        )
        top1_is_gt = bool(gt_index is not None and _topk_contains(sort_order, gt_index, k=1))
        top5_contains_gt = bool(gt_index is not None and _topk_contains(sort_order, gt_index, k=5))
        top10_contains_gt = bool(gt_index is not None and _topk_contains(sort_order, gt_index, k=10))
        top1_is_gt_extra = bool(gt_in_extra and gt_index is not None and _topk_contains(sort_order, gt_index, k=1))
        top5_contains_gt_extra = bool(gt_in_extra and gt_index is not None and _topk_contains(sort_order, gt_index, k=5))
        top10_contains_gt_extra = bool(gt_in_extra and gt_index is not None and _topk_contains(sort_order, gt_index, k=10))
        failure_bucket = _failure_bucket(
            stage_id=stage_id,
            extra_authoritative=bool(extra_authoritative),
            gt_raw_id=gt_raw_id_int,
            gt_in_vocab=bool(gt_in_vocab),
            gt_in_extra=bool(gt_in_extra),
            winner_domain=str(winner_domain),
            winner_is_gt_extra=bool(winner_is_gt_extra),
        )
        rows.append(
            {
                "stage": str(stage_id),
                "trajectory_id": trajectory_id,
                "clip_id": int(example.get("clip_id", -1)),
                "video_id": int(example.get("video_id", -1)),
                "split": split_label,
                "gt_raw_id_canonical": gt_raw_id_int,
                "observed_raw_ids": observed_raw_ids,
                "candidate_ids_known": known_ids,
                "candidate_ids_extra": extra_ids,
                "extra_authoritative": bool(extra_authoritative),
                "extra_applicable": bool(stage_id in EXTRA_APPLICABLE_STAGES),
                "gt_in_vocab": bool(gt_in_vocab),
                "gt_in_Yprime": bool(gt_in_known),
                "gt_in_extra": bool(gt_in_extra),
                "gt_in_extra_topk_candidate": bool(gt_in_extra_topk_candidate),
                "gt_in_other_nonYprime": bool(gt_in_other_nonYprime),
                "mass_on_Yprime": float(mass_on_Yprime),
                "mass_on_extra": float(mass_on_extra),
                "mass_on_gt_extra": float(mass_on_gt_extra),
                "mass_on_wrong_extra": float(mass_on_wrong_extra),
                "mass_on_nonextra_nonYprime": float(mass_on_other_nonYprime),
                "mass_on_unknown": float(prob_unknown),
                "mass_on_gt_class": float(mass_on_gt_class),
                "winner_domain": str(winner_domain),
                "winner_raw_id": winner_raw_id,
                "winner_is_gt": bool(winner_is_gt),
                "winner_is_gt_extra": bool(winner_is_gt_extra),
                "top1_is_gt": bool(top1_is_gt),
                "top5_contains_gt": bool(top5_contains_gt),
                "top10_contains_gt": bool(top10_contains_gt),
                "top1_is_gt_extra": bool(top1_is_gt_extra),
                "top5_contains_gt_extra": bool(top5_contains_gt_extra),
                "top10_contains_gt_extra": bool(top10_contains_gt_extra),
                "gt_rank": best_gt_rank,
                "gt_normalized_rank": best_gt_normalized_rank,
                "gt_extra_rank": gt_extra_rank,
                "gt_extra_normalized_rank": gt_extra_normalized_rank,
                "unknown_logit": float(unknown_logit),
                "failure_bucket": str(failure_bucket),
            }
        )
    return rows


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rate(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray([1.0 if bool(v) else 0.0 for v in values], dtype=np.float64)))


def _winner_rate(rows: Sequence[Mapping[str, Any]], label: str) -> Optional[float]:
    return _rate([str(row.get("winner_domain", "")) == str(label) for row in rows])


def _failure_bucket_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter = Counter(str(row.get("failure_bucket", "")) for row in rows)
    return {str(key): int(counter[key]) for key in sorted(counter.keys())}


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "trajectory_count": int(len(rows)),
        "gt_in_extra_candidate_rate": _rate([bool(row.get("gt_in_extra", False)) for row in rows]),
        "gt_in_extra_topk_candidate_rate": _rate([bool(row.get("gt_in_extra_topk_candidate", False)) for row in rows]),
        "mass_on_Yprime_mean": _safe_mean([float(row.get("mass_on_Yprime", 0.0)) for row in rows]),
        "mass_on_extra_mean": _safe_mean([float(row.get("mass_on_extra", 0.0)) for row in rows]),
        "mass_on_gt_extra_mean": _safe_mean([float(row.get("mass_on_gt_extra", 0.0)) for row in rows]),
        "mass_on_wrong_extra_mean": _safe_mean([float(row.get("mass_on_wrong_extra", 0.0)) for row in rows]),
        "mass_on_nonextra_nonYprime_mean": _safe_mean([float(row.get("mass_on_nonextra_nonYprime", 0.0)) for row in rows]),
        "mass_on_unknown_mean": _safe_mean([float(row.get("mass_on_unknown", 0.0)) for row in rows]),
        "mass_on_gt_class_mean": _safe_mean([float(row.get("mass_on_gt_class", 0.0)) for row in rows]),
        "winner_is_Yprime_rate": _winner_rate(rows, "Yprime"),
        "winner_is_extra_rate": _winner_rate(rows, "extra"),
        "winner_is_other_nonYprime_rate": _winner_rate(rows, "other_nonYprime"),
        "winner_is_unknown_rate": _winner_rate(rows, "unknown"),
        "winner_is_gt_rate": _rate([bool(row.get("winner_is_gt", False)) for row in rows]),
        "winner_is_gt_extra_rate": _rate([bool(row.get("winner_is_gt_extra", False)) for row in rows]),
        "top1_is_gt_rate": _rate([bool(row.get("top1_is_gt", False)) for row in rows]),
        "top5_contains_gt_rate": _rate([bool(row.get("top5_contains_gt", False)) for row in rows]),
        "top10_contains_gt_rate": _rate([bool(row.get("top10_contains_gt", False)) for row in rows]),
        "top1_is_gt_extra_rate": _rate([bool(row.get("top1_is_gt_extra", False)) for row in rows]),
        "top5_contains_gt_extra_rate": _rate([bool(row.get("top5_contains_gt_extra", False)) for row in rows]),
        "top10_contains_gt_extra_rate": _rate([bool(row.get("top10_contains_gt_extra", False)) for row in rows]),
        "gt_rank_mean": _safe_mean([float(row["gt_rank"]) for row in rows if row.get("gt_rank") is not None]),
        "gt_normalized_rank_mean": _safe_mean([float(row["gt_normalized_rank"]) for row in rows if row.get("gt_normalized_rank") is not None]),
        "gt_extra_rank_mean": _safe_mean([float(row["gt_extra_rank"]) for row in rows if row.get("gt_extra_rank") is not None]),
        "gt_extra_normalized_rank_mean": _safe_mean([float(row["gt_extra_normalized_rank"]) for row in rows if row.get("gt_extra_normalized_rank") is not None]),
        "failure_buckets": _failure_bucket_summary(rows),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, split_order: Sequence[str]) -> Dict[str, Any]:
    overall_rows = list(rows)
    by_split: Dict[str, Any] = {}
    failure_by_split: Dict[str, Dict[str, int]] = {}
    for split_name in split_order:
        split_rows = [row for row in overall_rows if row.get("split") == split_name]
        by_split[str(split_name)] = _summarize_group(split_rows)
        failure_by_split[str(split_name)] = _failure_bucket_summary(split_rows)
    return {
        "overall": _summarize_group(overall_rows),
        "by_split": by_split,
        "split_order": [str(x) for x in split_order],
        "failure_buckets": {
            "overall": _failure_bucket_summary(overall_rows),
            "by_split": failure_by_split,
        },
    }


def _select_top_failure_cases(rows: Sequence[Mapping[str, Any]], *, limit: int) -> List[Record]:
    bad = [dict(row) for row in rows if str(row.get("failure_bucket", "")).startswith("gt_in_extra") or str(row.get("failure_bucket", "")) == "gt_not_in_extra_candidate"]
    ranked = sorted(
        bad,
        key=lambda row: (
            float(row.get("mass_on_wrong_extra", 0.0)) + float(row.get("mass_on_nonextra_nonYprime", 0.0)),
            float(row.get("mass_on_Yprime", 0.0)),
            -(float(row.get("gt_extra_rank") or 1e9)),
            str(row.get("trajectory_id", "")),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(limit))] if ranked else []


def _build_markdown_report(*, config: ExtraAttributionProbeConfig, dataset_name: str, stage_results: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("# Extra Attribution Probe")
    lines.append("")
    lines.append(f"- run_root: `{Path(config.run_root).expanduser().resolve()}`")
    lines.append(f"- runtime_output_root: `{Path(config.runtime_output_root).expanduser().resolve()}`")
    lines.append(f"- dataset_name: `{dataset_name}`")
    lines.append(f"- trajectory_source_branch: `{config.trajectory_source_branch}`")
    lines.append(f"- stage_scope: `{list(config.stage_scope)}`")
    lines.append("")
    for stage_payload in stage_results:
        stage_id = str(stage_payload.get("stage_id", ""))
        summary = dict(stage_payload.get("summary", {}))
        overall = dict(summary.get("overall", {}))
        lines.append(f"## {stage_id}")
        lines.append("")
        for key in (
            "trajectory_count",
            "gt_in_extra_candidate_rate",
            "mass_on_Yprime_mean",
            "mass_on_extra_mean",
            "mass_on_gt_extra_mean",
            "mass_on_wrong_extra_mean",
            "mass_on_nonextra_nonYprime_mean",
            "mass_on_unknown_mean",
            "winner_is_Yprime_rate",
            "winner_is_extra_rate",
            "winner_is_other_nonYprime_rate",
            "winner_is_unknown_rate",
            "winner_is_gt_rate",
            "winner_is_gt_extra_rate",
            "top1_is_gt_rate",
            "top1_is_gt_extra_rate",
            "gt_rank_mean",
            "gt_extra_rank_mean",
        ):
            lines.append(f"- {key}: `{overall.get(key)}`")
        lines.append("")
    return lines


def _comparison_payload(dataset_name: str, stage_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    comparison_by_stage: Dict[str, Any] = {}
    by_split_metric_view: Dict[str, Dict[str, Any]] = {}
    for stage_payload in stage_results:
        stage_id = str(stage_payload.get("stage_id", ""))
        summary = dict(stage_payload.get("summary", {}))
        comparison_by_stage[stage_id] = {
            "summary_path": stage_payload.get("summary_path"),
            "rows_path": stage_payload.get("rows_path"),
            "failure_buckets_path": stage_payload.get("failure_buckets_path"),
            "top_failure_cases_path": stage_payload.get("top_failure_cases_path"),
            "extra_applicable": bool(stage_payload.get("extra_applicable", False)),
            "overall": summary.get("overall", {}),
        }
        for split_name in summary.get("split_order", []):
            split_key = str(split_name)
            by_split_metric_view.setdefault(split_key, {})[stage_id] = dict(summary.get("by_split", {}).get(split_key, {}))
    return {
        "dataset_name": str(dataset_name),
        "stage_scope": [str(row.get("stage_id", "")) for row in stage_results],
        "comparison_by_stage": comparison_by_stage,
        "by_split_metric_view": by_split_metric_view,
    }


def _stage_output_dir(base_dir: Path, stage_id: str) -> Path:
    return base_dir / str(stage_id)


def _run_stage_probe(
    *,
    config: ExtraAttributionProbeConfig,
    stage_id: str,
    examples: Sequence[Mapping[str, Any]],
    materialized_stats: Mapping[str, Any],
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    base_vocab_ids: Sequence[int],
    output_dir: Path,
) -> Dict[str, Any]:
    checkpoint_path = _default_checkpoint_path(Path(config.run_root).expanduser().resolve(), stage_id)
    if not checkpoint_path.is_file():
        payload = {
            "stage_id": str(stage_id),
            "status": "CHECKPOINT_NOT_FOUND",
            "checkpoint_path": str(checkpoint_path),
            "dataset_name": str(config.dataset_name),
            "summary": {},
            "rows_path": None,
            "summary_path": None,
            "failure_buckets_path": None,
            "top_failure_cases_path": None,
            "extra_applicable": bool(stage_id in EXTRA_APPLICABLE_STAGES),
        }
        return payload
    device = torch.device(str(config.device))
    projector, theta_t, unknown_prototype, checkpoint_payload = _load_reservoir_checkpoint(checkpoint_path, device=device)
    projector.eval()
    temperature = _compute_t_dis(theta_t).detach()
    logits_vocab, logits_unknown = _score_batches(
        examples=examples,
        projector=projector,
        text_vocab_matrix=np.asarray(text_vocab_matrix, dtype=np.float32),
        unknown_prototype=unknown_prototype,
        temperature=temperature,
        device=device,
        batch_size=max(1, int(config.batch_size)),
        show_progress=bool(config.show_progress),
        stage_id=str(stage_id),
    )
    rows = _build_row_metrics(
        stage_id=str(stage_id),
        examples=examples,
        logits_vocab=logits_vocab,
        logits_unknown=logits_unknown,
        text_vocab_ids=text_vocab_ids,
        sidecar_lookup=sidecar_lookup,
        dataset_name=str(config.dataset_name),
        base_vocab_ids=base_vocab_ids,
        extra_candidate_topk=max(1, int(config.extra_candidate_topk)),
    )
    summary = _summarize_rows(rows, split_order=_split_order_for_dataset(config.dataset_name))
    failure_buckets = dict(summary.get("failure_buckets", {}))
    top_failure_cases = _select_top_failure_cases(rows, limit=max(1, int(config.top_failure_cases)))
    stage_dir = _stage_output_dir(output_dir, stage_id)
    payload = {
        "run_root": str(Path(config.run_root).expanduser().resolve()),
        "runtime_output_root": str(Path(config.runtime_output_root).expanduser().resolve()),
        "dataset_name": str(config.dataset_name),
        "trajectory_source_branch": str(config.trajectory_source_branch),
        "stage_id": str(stage_id),
        "checkpoint_path": str(checkpoint_path),
        "sidecar_root": str(_sidecar_root(config)),
        "materialization_stats": dict(materialized_stats),
        "trajectory_count": int(len(rows)),
        "text_vocab_size": int(len(text_vocab_ids)),
        "temperature": float(temperature.detach().cpu().item()),
        "extra_applicable": bool(stage_id in EXTRA_APPLICABLE_STAGES),
        "summary": summary,
        "failure_buckets": failure_buckets,
        "checkpoint_payload_meta": {
            "stage_id": checkpoint_payload.get("stage_id"),
            "epoch": checkpoint_payload.get("epoch"),
            "seed": checkpoint_payload.get("seed"),
            "pipeline": checkpoint_payload.get("pipeline"),
        },
    }
    summary_path = stage_dir / "summary.json"
    rows_path = stage_dir / "row_metrics.jsonl"
    failure_path = stage_dir / "failure_buckets.json"
    top_failure_path = stage_dir / "top_failure_cases.jsonl"
    meta_path = stage_dir / "meta.json"
    _write_json(summary_path, payload)
    _write_jsonl(rows_path, rows)
    _write_json(failure_path, failure_buckets)
    _write_jsonl(top_failure_path, top_failure_cases)
    _write_json(
        meta_path,
        {
            "stage_id": str(stage_id),
            "checkpoint_path": str(checkpoint_path),
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
            "failure_buckets_path": str(failure_path),
            "top_failure_cases_path": str(top_failure_path),
        },
    )
    return {
        "stage_id": str(stage_id),
        "status": "PASS",
        "summary": summary,
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "failure_buckets_path": str(failure_path),
        "top_failure_cases_path": str(top_failure_path),
        "extra_applicable": bool(stage_id in EXTRA_APPLICABLE_STAGES),
    }


def run_extra_attribution_probe(config: ExtraAttributionProbeConfig) -> Dict[str, Any]:
    run_root = Path(config.run_root).expanduser().resolve()
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = _materialize_valid_samples(config)
    valid_samples = list(materialized.get("valid_samples", []))
    prepared = _prepare_probe_examples(
        valid_samples,
        output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    examples = list(prepared.get("examples", []))
    if not examples:
        raise RuntimeError("no valid probe examples after phase-1 filtering")
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(runtime_output_root)
    sidecar_lookup = load_gt_sidecar_lookup(
        _sidecar_root(config),
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    base_vocab_ids, _novel_vocab_ids = load_lvvis_base_and_novel_raw_ids()
    stage_results: List[Dict[str, Any]] = []
    for stage_id in config.stage_scope:
        stage_results.append(
            _run_stage_probe(
                config=config,
                stage_id=str(stage_id),
                examples=examples,
                materialized_stats=dict(materialized.get("stats", {})),
                text_vocab_ids=text_vocab_ids,
                text_vocab_matrix=np.asarray(text_vocab_matrix, dtype=np.float32),
                sidecar_lookup=sidecar_lookup,
                base_vocab_ids=base_vocab_ids,
                output_dir=output_dir,
            )
        )
    comparison = _comparison_payload(str(config.dataset_name), stage_results)
    comparison_path = output_dir / "comparison_summary.json"
    report_path = output_dir / "report.md"
    _write_json(comparison_path, comparison)
    _write_markdown(report_path, _build_markdown_report(config=config, dataset_name=str(config.dataset_name), stage_results=stage_results))
    return {
        "status": "PASS",
        "comparison_path": str(comparison_path),
        "report_path": str(report_path),
        "stage_results": stage_results,
    }
