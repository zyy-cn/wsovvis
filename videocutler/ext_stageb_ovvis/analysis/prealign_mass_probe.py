
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import (
    _clip_groups,
    _compute_t_dis,
    _load_reservoir_checkpoint,
    _normalize_np,
    _project_text_matrix,
    _unknown_score,
)
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
TOPK_VALUES: Tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class PrealignMassProbeConfig:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    device: str = "cpu"
    smoke: bool = False
    smoke_max_trajectories: int = 128
    subset_fraction: Optional[float] = None
    checkpoint_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    sidecar_root: Optional[Path] = None
    show_progress: bool = True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _make_stage_progress(enabled: bool, total: int, desc: str):
    if enabled and _tqdm is not None:
        return _tqdm(total=int(total), desc=str(desc), dynamic_ncols=True)
    return None


def _make_iter_progress(iterable, *, enabled: bool, desc: str):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, desc=str(desc), dynamic_ncols=True)
    return iterable


def _progress_update(handle, n: int = 1) -> None:
    if handle is not None:
        handle.update(int(n))


def _progress_close(handle) -> None:
    if handle is not None:
        handle.close()


def _safe_mass_on_indices(probs_vocab: np.ndarray, index_list: Sequence[int]) -> float:
    if not index_list:
        return 0.0
    idx = np.asarray([int(x) for x in index_list], dtype=np.int64)
    if idx.size == 0:
        return 0.0
    return float(np.sum(probs_vocab[idx]))


def _safe_rank_of_indices(ranks: np.ndarray, index_list: Sequence[int]) -> Optional[int]:
    if not index_list:
        return None
    idx = np.asarray([int(x) for x in index_list], dtype=np.int64)
    if idx.size == 0:
        return None
    return int(np.min(ranks[idx])) + 1


def _topk_mass_on_mask(probs_vocab: np.ndarray, sort_order: np.ndarray, mask: np.ndarray, *, k: int) -> float:
    if probs_vocab.size == 0 or int(k) <= 0:
        return 0.0
    topk_idx = sort_order[: min(int(k), len(sort_order))]
    if topk_idx.size == 0:
        return 0.0
    return float(np.sum(probs_vocab[topk_idx[mask[topk_idx]]]))


def _default_checkpoint_path(run_root: Path) -> Path:
    return run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"


def _default_output_dir(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "prealign_mass_probe" / dataset_name


def _split_order_for_dataset(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name == "lvvis_train_base":
        return TRAIN_SPLIT_ORDER
    if dataset_name == "lvvis_val":
        return VAL_SPLIT_ORDER
    raise ValueError(f"unsupported dataset_name: {dataset_name}")


def _resolve_output_dir(config: PrealignMassProbeConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() if config.output_dir is not None else _default_output_dir(config.run_root, config.dataset_name)


def _sidecar_root(config: PrealignMassProbeConfig) -> Path:
    return Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root is not None else Path(config.run_root).expanduser().resolve()


def _materialize_valid_samples(config: PrealignMassProbeConfig) -> Dict[str, Any]:
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


def _prepare_examples(config: PrealignMassProbeConfig, *, valid_samples: Sequence[Record]) -> Sequence[Record]:
    prepared = _prepare_prealign_examples(
        valid_samples,
        output_root=Path(config.runtime_output_root).expanduser().resolve(),
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    return list(prepared["examples"])


def _build_clip_gt_sets(
    examples: Sequence[Mapping[str, Any]],
    *,
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    vocab_index: Mapping[int, int],
) -> Dict[int, List[int]]:
    clip_to_gt: Dict[int, set[int]] = {}
    for ex in examples:
        trajectory_id = str(ex.get("trajectory_id", "")).strip()
        clip_id = int(ex.get("clip_id", -1))
        if not trajectory_id or clip_id < 0:
            continue
        sidecar = dict(sidecar_lookup.get(trajectory_id, {}))
        if not sidecar:
            continue
        gt_raw_id = _canonical_sidecar_gt_raw_id(sidecar)
        if gt_raw_id is None:
            continue
        gt_raw_id = int(gt_raw_id)
        if gt_raw_id not in vocab_index:
            continue
        clip_to_gt.setdefault(clip_id, set()).add(gt_raw_id)
    return {int(k): sorted(int(x) for x in v) for k, v in clip_to_gt.items()}


def _build_clip_probe_rows(
    examples: Sequence[Mapping[str, Any]],
    *,
    text_vocab_ids: Sequence[int],
    logits_vocab: np.ndarray,
    logits_unknown: np.ndarray,
    clip_gt_sets: Mapping[int, Sequence[int]],
    show_progress: bool = True,
) -> Tuple[List[Record], Dict[int, Record]]:
    vocab_ids = [int(x) for x in text_vocab_ids]
    vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(vocab_ids)}
    clip_groups = _clip_groups(examples)
    clip_rows: List[Record] = []
    clip_lookup: Dict[int, Record] = {}
    denom_rank = float(max(len(vocab_ids) - 1, 1))
    for clip_pos, clip_examples in enumerate(_make_iter_progress(clip_groups, enabled=show_progress, desc='probe: clip rows')):
        if not clip_examples:
            continue
        clip_id = int(clip_examples[0]["clip_id"])
        video_id = int(clip_examples[0]["video_id"])
        observed_raw_ids = sorted({int(x) for x in list(clip_examples[0].get("observed_raw_ids", []))})
        observed_index = [int(vocab_index[x]) for x in observed_raw_ids if int(x) in vocab_index]
        observed_mask = np.zeros((len(vocab_ids),), dtype=bool)
        if observed_index:
            observed_mask[np.asarray(observed_index, dtype=np.int64)] = True
        gt_raw_ids = sorted({int(x) for x in list(clip_gt_sets.get(clip_id, [])) if int(x) in vocab_index})
        gt_index = [int(vocab_index[x]) for x in gt_raw_ids]
        gt_mask = np.zeros((len(vocab_ids),), dtype=bool)
        if gt_index:
            gt_mask[np.asarray(gt_index, dtype=np.int64)] = True
        wrong_non_yprime_mask = (~observed_mask) & (~gt_mask)

        vocab_logits = np.asarray(logits_vocab[clip_pos], dtype=np.float64)
        unknown_logit = float(logits_unknown[clip_pos])
        max_logit = float(max(float(np.max(vocab_logits)) if vocab_logits.size else -np.inf, unknown_logit))
        exp_vocab = np.exp(vocab_logits - max_logit)
        exp_unknown = float(np.exp(unknown_logit - max_logit))
        denom = float(np.sum(exp_vocab) + exp_unknown)
        probs_vocab = exp_vocab / max(denom, 1e-12)
        prob_unknown = exp_unknown / max(denom, 1e-12)
        mass_yprime = float(np.sum(probs_vocab[observed_mask])) if observed_mask.any() else 0.0
        mass_non_yprime = float(np.sum(probs_vocab[~observed_mask])) if probs_vocab.size else 0.0
        mass_gt_set = float(np.sum(probs_vocab[gt_mask])) if gt_mask.any() else 0.0
        mass_wrong_non_yprime = float(np.sum(probs_vocab[wrong_non_yprime_mask])) if wrong_non_yprime_mask.any() else 0.0

        best_yprime_logit = float(np.max(vocab_logits[observed_mask])) if observed_mask.any() else float("-inf")
        best_non_yprime_logit = float(np.max(vocab_logits[~observed_mask])) if (~observed_mask).any() else float("-inf")
        best_gt_logit = float(np.max(vocab_logits[gt_mask])) if gt_mask.any() else float("-inf")

        sort_order = np.argsort(-vocab_logits, kind="stable")
        ranks = np.empty_like(sort_order)
        ranks[sort_order] = np.arange(len(sort_order), dtype=np.int64)
        best_gt_rank = _safe_rank_of_indices(ranks, gt_index)
        rank_of_gt_class = best_gt_rank
        best_gt_normalized_rank = (float(best_gt_rank - 1) / denom_rank) if best_gt_rank is not None else None
        mass_on_gt_class = _safe_mass_on_indices(probs_vocab, gt_index)
        mass_on_nonYprime_excluding_gt = mass_wrong_non_yprime
        topk_payload: Dict[int, bool] = {}
        topk_non_yprime_mass_payload: Dict[int, float] = {}
        for k in TOPK_VALUES:
            if gt_mask.any():
                topk_idx = sort_order[: min(int(k), len(sort_order))]
                topk_payload[int(k)] = bool(np.any(gt_mask[topk_idx]))
            else:
                topk_payload[int(k)] = False
            topk_non_yprime_mass_payload[int(k)] = _topk_mass_on_mask(probs_vocab, sort_order, ~observed_mask, k=int(k))
        top1_is_gt_rate = bool(topk_payload[1])
        topk_contains_gt_rate = bool(topk_payload[5])
        mass_on_topk_nonYprime = float(topk_non_yprime_mass_payload[5])

        winner_group = "unknown"
        winner_raw_id: Optional[int] = None
        winner_is_gt = False
        if vocab_logits.size:
            best_vocab_idx = int(np.argmax(vocab_logits))
            best_vocab_logit = float(vocab_logits[best_vocab_idx])
            if best_vocab_logit >= unknown_logit:
                winner_raw_id = int(vocab_ids[best_vocab_idx])
                winner_group = "Yprime" if observed_mask[best_vocab_idx] else "nonYprime"
                winner_is_gt = bool(gt_mask[best_vocab_idx])

        clip_row = {
            "clip_id": clip_id,
            "video_id": video_id,
            "trajectory_count": int(len(clip_examples)),
            "observed_raw_ids": observed_raw_ids,
            "gt_raw_ids": gt_raw_ids,
            "gt_set_size": int(len(gt_raw_ids)),
            "mass_on_Yprime": mass_yprime,
            "mass_on_nonYprime": mass_non_yprime,
            "mass_on_unknown": float(prob_unknown),
            "mass_on_gt_set": mass_gt_set,
            "mass_on_gt_class": mass_on_gt_class,
            "mass_on_wrong_nonYprime": mass_wrong_non_yprime,
            "mass_on_nonYprime_excluding_gt": mass_on_nonYprime_excluding_gt,
            "mass_on_topk_nonYprime": mass_on_topk_nonYprime,
            "winner_group": winner_group,
            "winner_raw_id": winner_raw_id,
            "winner_is_gt": bool(winner_is_gt),
            "unknown_beats_best_Yprime": bool(unknown_logit > best_yprime_logit) if np.isfinite(best_yprime_logit) else True,
            "best_nonYprime_beats_best_Yprime": bool(best_non_yprime_logit > best_yprime_logit) if np.isfinite(best_yprime_logit) else False,
            "best_Yprime_logit": None if not np.isfinite(best_yprime_logit) else best_yprime_logit,
            "best_nonYprime_logit": None if not np.isfinite(best_non_yprime_logit) else best_non_yprime_logit,
            "best_gt_logit": None if not np.isfinite(best_gt_logit) else best_gt_logit,
            "best_gt_rank": best_gt_rank,
            "rank_of_gt_class": rank_of_gt_class,
            "best_gt_normalized_rank": best_gt_normalized_rank,
            "gt_top1_hit": bool(topk_payload[1]),
            "gt_top5_hit": bool(topk_payload[5]),
            "gt_top10_hit": bool(topk_payload[10]),
            "top1_is_gt": top1_is_gt_rate,
            "topk_contains_gt": topk_contains_gt_rate,
            "unknown_logit": unknown_logit,
        }
        clip_rows.append(clip_row)
        clip_lookup[clip_id] = clip_row
    return clip_rows, clip_lookup


def _build_trajectory_rows(
    examples: Sequence[Mapping[str, Any]],
    *,
    clip_lookup: Mapping[int, Mapping[str, Any]],
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    dataset_name: str,
    base_vocab_ids: Sequence[int],
) -> List[Record]:
    rows: List[Record] = []
    for ex in sorted(examples, key=lambda row: str(row.get("trajectory_id", ""))):
        trajectory_id = str(ex.get("trajectory_id", "")).strip()
        clip_id = int(ex.get("clip_id", -1))
        clip_payload = dict(clip_lookup.get(clip_id, {}))
        sidecar = dict(sidecar_lookup.get(trajectory_id, {})) if trajectory_id else {}
        gt_raw_id = _canonical_sidecar_gt_raw_id(sidecar) if sidecar else None
        observed_raw_ids = [int(x) for x in list(ex.get("observed_raw_ids", []))]
        split_label = None
        if gt_raw_id is not None:
            split_label = _all_gt_split_label(
                dataset_name=str(dataset_name),
                gt_raw_id=int(gt_raw_id),
                observed_raw_ids=observed_raw_ids,
                base_vocab_ids=base_vocab_ids,
            )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "clip_id": clip_id,
                "video_id": int(ex.get("video_id", -1)),
                "observed_raw_ids": observed_raw_ids,
                "gt_raw_id_canonical": int(gt_raw_id) if gt_raw_id is not None else None,
                "gt_raw_ids": clip_payload.get("gt_raw_ids", []),
                "gt_set_size": int(clip_payload.get("gt_set_size", 0)),
                "split": split_label,
                "mass_on_Yprime": float(clip_payload.get("mass_on_Yprime", 0.0)),
                "mass_on_nonYprime": float(clip_payload.get("mass_on_nonYprime", 0.0)),
                "mass_on_unknown": float(clip_payload.get("mass_on_unknown", 0.0)),
                "mass_on_gt_set": float(clip_payload.get("mass_on_gt_set", 0.0)),
                "mass_on_gt_class": float(clip_payload.get("mass_on_gt_class", 0.0)),
                "mass_on_wrong_nonYprime": float(clip_payload.get("mass_on_wrong_nonYprime", 0.0)),
                "mass_on_nonYprime_excluding_gt": float(clip_payload.get("mass_on_nonYprime_excluding_gt", 0.0)),
                "mass_on_topk_nonYprime": float(clip_payload.get("mass_on_topk_nonYprime", 0.0)),
                "winner_group": str(clip_payload.get("winner_group", "")),
                "winner_raw_id": clip_payload.get("winner_raw_id"),
                "winner_is_gt": bool(clip_payload.get("winner_is_gt", False)),
                "unknown_beats_best_Yprime": bool(clip_payload.get("unknown_beats_best_Yprime", False)),
                "best_nonYprime_beats_best_Yprime": bool(clip_payload.get("best_nonYprime_beats_best_Yprime", False)),
                "best_Yprime_logit": clip_payload.get("best_Yprime_logit"),
                "best_nonYprime_logit": clip_payload.get("best_nonYprime_logit"),
                "best_gt_logit": clip_payload.get("best_gt_logit"),
                "best_gt_rank": clip_payload.get("best_gt_rank"),
                "rank_of_gt_class": clip_payload.get("rank_of_gt_class"),
                "best_gt_normalized_rank": clip_payload.get("best_gt_normalized_rank"),
                "gt_top1_hit": bool(clip_payload.get("gt_top1_hit", False)),
                "gt_top5_hit": bool(clip_payload.get("gt_top5_hit", False)),
                "gt_top10_hit": bool(clip_payload.get("gt_top10_hit", False)),
                "top1_is_gt": bool(clip_payload.get("top1_is_gt", False)),
                "topk_contains_gt": bool(clip_payload.get("topk_contains_gt", False)),
                "unknown_logit": clip_payload.get("unknown_logit"),
            }
        )
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, split_order: Sequence[str]) -> Dict[str, Any]:
    summary_by_split: Dict[str, Any] = {}
    overall_rows = list(rows)
    for split_name in split_order:
        split_rows = [row for row in overall_rows if row.get("split") == split_name]
        summary_by_split[str(split_name)] = _summarize_group(split_rows)
    return {
        "overall": _summarize_group(overall_rows),
        "by_split": summary_by_split,
        "split_order": [str(x) for x in split_order],
    }


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rate(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray([1.0 if bool(v) else 0.0 for v in values], dtype=np.float64)))


def _winner_rate(rows: Sequence[Mapping[str, Any]], label: str) -> Optional[float]:
    return _rate([str(row.get("winner_group", "")) == str(label) for row in rows])


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "trajectory_count": int(len(rows)),
        "mass_on_Yprime_mean": _safe_mean([float(row.get("mass_on_Yprime", 0.0)) for row in rows]),
        "mass_on_nonYprime_mean": _safe_mean([float(row.get("mass_on_nonYprime", 0.0)) for row in rows]),
        "mass_on_unknown_mean": _safe_mean([float(row.get("mass_on_unknown", 0.0)) for row in rows]),
        "mass_on_gt_set_mean": _safe_mean([float(row.get("mass_on_gt_set", 0.0)) for row in rows]),
        "mass_on_gt_class_mean": _safe_mean([float(row.get("mass_on_gt_class", 0.0)) for row in rows]),
        "mass_on_wrong_nonYprime_mean": _safe_mean([float(row.get("mass_on_wrong_nonYprime", 0.0)) for row in rows]),
        "mass_on_nonYprime_excluding_gt_mean": _safe_mean([float(row.get("mass_on_nonYprime_excluding_gt", 0.0)) for row in rows]),
        "mass_on_topk_nonYprime_mean": _safe_mean([float(row.get("mass_on_topk_nonYprime", 0.0)) for row in rows]),
        "winner_is_Yprime_rate": _winner_rate(rows, "Yprime"),
        "winner_is_nonYprime_rate": _winner_rate(rows, "nonYprime"),
        "winner_is_unknown_rate": _winner_rate(rows, "unknown"),
        "winner_is_gt_rate": _rate([bool(row.get("winner_is_gt", False)) for row in rows]),
        "top1_is_gt_rate": _rate([bool(row.get("top1_is_gt", False)) for row in rows]),
        "probe_probe_gt_top1_hit_rate": _rate([bool(row.get("gt_top1_hit", False)) for row in rows]),
        "topk_contains_gt_rate": _rate([bool(row.get("topk_contains_gt", False)) for row in rows]),
        "probe_probe_gt_top5_hit_rate": _rate([bool(row.get("gt_top5_hit", False)) for row in rows]),
        "probe_probe_gt_top10_hit_rate": _rate([bool(row.get("gt_top10_hit", False)) for row in rows]),
        "unknown_beats_best_Yprime_rate": _rate([bool(row.get("unknown_beats_best_Yprime", False)) for row in rows]),
        "best_nonYprime_beats_best_Yprime_rate": _rate([bool(row.get("best_nonYprime_beats_best_Yprime", False)) for row in rows]),
        "best_Yprime_logit_mean": _safe_mean([float(row["best_Yprime_logit"]) for row in rows if row.get("best_Yprime_logit") is not None]),
        "best_nonYprime_logit_mean": _safe_mean([float(row["best_nonYprime_logit"]) for row in rows if row.get("best_nonYprime_logit") is not None]),
        "best_gt_logit_mean": _safe_mean([float(row["best_gt_logit"]) for row in rows if row.get("best_gt_logit") is not None]),
        "best_gt_rank_mean": _safe_mean([float(row["best_gt_rank"]) for row in rows if row.get("best_gt_rank") is not None]),
        "rank_of_gt_class_mean": _safe_mean([float(row["rank_of_gt_class"]) for row in rows if row.get("rank_of_gt_class") is not None]),
        "best_gt_normalized_rank_mean": _safe_mean([float(row["best_gt_normalized_rank"]) for row in rows if row.get("best_gt_normalized_rank") is not None]),
        "unknown_logit_mean": _safe_mean([float(row.get("unknown_logit", 0.0)) for row in rows]),
    }


def _build_markdown_report(*, config: PrealignMassProbeConfig, summary: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append("# Prealign Mass Probe")
    lines.append("")
    lines.append(f"- run_root: `{Path(config.run_root).expanduser().resolve()}`")
    lines.append(f"- runtime_output_root: `{Path(config.runtime_output_root).expanduser().resolve()}`")
    lines.append(f"- dataset_name: `{config.dataset_name}`")
    lines.append(f"- trajectory_source_branch: `{config.trajectory_source_branch}`")
    lines.append(f"- checkpoint_stage: `prealign`")
    lines.append("")
    overall = dict(summary.get("overall", {}))
    lines.append("## Overall")
    lines.append("")
    for key in (
        "trajectory_count",
        "mass_on_Yprime_mean",
        "mass_on_nonYprime_mean",
        "mass_on_unknown_mean",
        "mass_on_gt_set_mean",
        "mass_on_wrong_nonYprime_mean",
        "winner_is_Yprime_rate",
        "winner_is_nonYprime_rate",
        "winner_is_unknown_rate",
        "winner_is_gt_rate",
        "probe_probe_gt_top1_hit_rate",
        "probe_probe_gt_top5_hit_rate",
        "probe_probe_gt_top10_hit_rate",
        "best_gt_rank_mean",
        "best_gt_normalized_rank_mean",
        "unknown_beats_best_Yprime_rate",
        "best_nonYprime_beats_best_Yprime_rate",
    ):
        lines.append(f"- {key}: `{overall.get(key)}`")
    lines.append("")
    lines.append("## By Split")
    lines.append("")
    by_split = dict(summary.get("by_split", {}))
    for split_name in summary.get("split_order", []):
        payload = dict(by_split.get(str(split_name), {}))
        lines.append(f"### {split_name}")
        lines.append("")
        for key in (
            "trajectory_count",
            "mass_on_Yprime_mean",
            "mass_on_nonYprime_mean",
            "mass_on_unknown_mean",
            "mass_on_gt_set_mean",
            "mass_on_gt_class_mean",
            "mass_on_wrong_nonYprime_mean",
            "mass_on_nonYprime_excluding_gt_mean",
            "mass_on_topk_nonYprime_mean",
            "winner_is_Yprime_rate",
            "winner_is_nonYprime_rate",
            "winner_is_unknown_rate",
            "winner_is_gt_rate",
            "top1_is_gt_rate",
            "topk_contains_gt_rate",
            "probe_probe_gt_top1_hit_rate",
            "probe_probe_gt_top5_hit_rate",
            "probe_probe_gt_top10_hit_rate",
            "best_gt_rank_mean",
            "rank_of_gt_class_mean",
            "best_gt_normalized_rank_mean",
            "unknown_beats_best_Yprime_rate",
            "best_nonYprime_beats_best_Yprime_rate",
        ):
            lines.append(f"- {key}: `{payload.get(key)}`")
        lines.append("")
    return lines


def _table_rows(summary: Mapping[str, Any]) -> List[Record]:
    rows: List[Record] = []
    overall = dict(summary.get("overall", {}))
    rows.append({"split": "overall", **overall})
    for split_name in summary.get("split_order", []):
        rows.append({"split": str(split_name), **dict(summary.get("by_split", {}).get(str(split_name), {}))})
    return rows


def run_prealign_mass_probe(config: PrealignMassProbeConfig) -> Dict[str, Any]:
    run_root = Path(config.run_root).expanduser().resolve()
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve() if config.checkpoint_path is not None else _default_checkpoint_path(run_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"prealign checkpoint not found: {checkpoint_path}")
    output_dir = _resolve_output_dir(config)
    sidecar_root = _sidecar_root(config)

    stage_bar = _make_stage_progress(bool(config.show_progress), total=8, desc='prealign-mass-probe')
    materialized = _materialize_valid_samples(config)
    _progress_update(stage_bar)
    valid_samples = list(materialized.get("valid_samples", []))
    examples = list(_prepare_examples(config, valid_samples=valid_samples))
    _progress_update(stage_bar)
    if not examples:
        _progress_close(stage_bar)
        raise RuntimeError("no prealign examples available for probe")

    device = torch.device(str(config.device))
    projector, theta_t, unknown_prototype, checkpoint_payload = _load_reservoir_checkpoint(checkpoint_path, device=device)
    projector.eval()
    temperature = _compute_t_dis(theta_t).detach()
    _progress_update(stage_bar)
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(runtime_output_root)
    vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    _progress_update(stage_bar)
    sidecar_lookup = load_gt_sidecar_lookup(
        sidecar_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    clip_gt_sets = _build_clip_gt_sets(examples, sidecar_lookup=sidecar_lookup, vocab_index=vocab_index)
    _progress_update(stage_bar)

    with torch.no_grad():
        text_proj = _project_text_matrix(projector, np.asarray(text_vocab_matrix, dtype=np.float32), device=device)
        clip_groups = _clip_groups(examples)
        clip_vectors: List[np.ndarray] = []
        for clip_examples in _make_iter_progress(clip_groups, enabled=bool(config.show_progress), desc='probe: clip encode'):
            carrier_stack = np.stack([np.asarray(ex["carrier_vec"], dtype=np.float32) for ex in clip_examples], axis=0)
            clip_vectors.append(_normalize_np(np.mean(carrier_stack, axis=0)))
        z_clip = torch.from_numpy(np.asarray(clip_vectors, dtype=np.float32)).to(device=device, dtype=torch.float32)
        logits_vocab_t = torch.matmul(F.normalize(z_clip, p=2.0, dim=-1), text_proj.t()) / temperature
        unknown_mode = str(checkpoint_payload.get('unknown_mode', 'prototype'))
        if unknown_mode == 'scalar_bias':
            b_u_value = float(checkpoint_payload.get('b_u', 0.0))
            logits_unknown_t = torch.ones((int(z_clip.shape[0]),), device=z_clip.device, dtype=z_clip.dtype) * b_u_value
        else:
            logits_unknown_t = _unknown_score(z_clip, unknown_prototype, temperature)
    _progress_update(stage_bar)
    logits_vocab = logits_vocab_t.detach().cpu().numpy().astype(np.float64)
    logits_unknown = logits_unknown_t.detach().cpu().numpy().reshape(-1).astype(np.float64)

    clip_rows, clip_lookup = _build_clip_probe_rows(
        examples,
        text_vocab_ids=text_vocab_ids,
        logits_vocab=logits_vocab,
        logits_unknown=logits_unknown,
        clip_gt_sets=clip_gt_sets,
        show_progress=bool(config.show_progress),
    )
    _progress_update(stage_bar)
    base_vocab_ids, _novel_vocab_ids = load_lvvis_base_and_novel_raw_ids()
    trajectory_rows = _build_trajectory_rows(
        examples,
        clip_lookup=clip_lookup,
        sidecar_lookup=sidecar_lookup,
        dataset_name=str(config.dataset_name),
        base_vocab_ids=base_vocab_ids,
    )
    summary = _summarize_rows(trajectory_rows, split_order=_split_order_for_dataset(config.dataset_name))
    _progress_update(stage_bar)
    payload = {
        "run_root": str(run_root),
        "runtime_output_root": str(runtime_output_root),
        "dataset_name": str(config.dataset_name),
        "trajectory_source_branch": str(config.trajectory_source_branch),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_stage": "prealign",
        "sidecar_root": str(sidecar_root),
        "materialization_stats": dict(materialized.get("stats", {})),
        "clip_count": int(len(clip_rows)),
        "trajectory_count": int(len(trajectory_rows)),
        "text_vocab_size": int(len(text_vocab_ids)),
        "temperature": float(temperature.detach().cpu().item()),
        "summary": summary,
        "checkpoint_payload_meta": {
            "stage_id": checkpoint_payload.get("stage_id"),
            "epoch": checkpoint_payload.get("epoch"),
            "seed": checkpoint_payload.get("seed"),
            "pipeline": checkpoint_payload.get("pipeline"),
        },
    }
    _write_json(output_dir / "prealign_mass_probe_summary.json", payload)
    _write_csv(output_dir / "prealign_mass_probe_table.csv", _table_rows(summary))
    _write_csv(output_dir / "prealign_mass_probe_trajectory_rows.csv", trajectory_rows)
    _write_markdown(output_dir / "prealign_mass_probe_report.md", _build_markdown_report(config=config, summary=summary))
    _progress_update(stage_bar)
    _progress_close(stage_bar)
    return {
        "status": "PASS",
        "summary_path": str(output_dir / "prealign_mass_probe_summary.json"),
        "table_path": str(output_dir / "prealign_mass_probe_table.csv"),
        "rows_path": str(output_dir / "prealign_mass_probe_trajectory_rows.csv"),
        "report_path": str(output_dir / "prealign_mass_probe_report.md"),
        "summary": payload,
    }
