from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.analysis.extra_mining_score_registry import (
    ExtraMiningScoreVariant,
    apply_candidate_masks,
    build_default_variants,
    mmr_select_topk,
    parse_variant_names,
    score_variant,
    tau_key,
)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

Record = Dict[str, Any]
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 2, 3, 5, 10)
DEFAULT_EXPECTED_FORMAL_GT_COUNT = 4007
DEFAULT_EXPECTED_EXISTING_RATE = 0.2939855


@dataclass(frozen=True)
class ExtraMiningSimulatorConfig:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    formal_split: str = "base_unobserved"
    device: str = "cpu"
    stage_id: str = "softem_aug"
    checkpoint_stage: str = "softem_aug"
    checkpoint_path: Optional[Path] = None
    sidecar_root: Optional[Path] = None
    formal_row_diagnostics_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    smoke: bool = False
    smoke_max_trajectories: int = 128
    subset_fraction: Optional[float] = None
    batch_size_clips: int = 16
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES
    primary_k: int = 3
    variant_names: Optional[Tuple[str, ...]] = None
    expected_formal_gt_count: Optional[int] = DEFAULT_EXPECTED_FORMAL_GT_COUNT
    expected_existing_gt_in_extra_rate: Optional[float] = DEFAULT_EXPECTED_EXISTING_RATE
    expected_rate_tolerance: float = 0.003
    enforce_expected_baseline: bool = True
    obs_sim_max: float = 0.90
    alpha_obs: float = 0.25
    topm_values: Tuple[int, ...] = (2, 3, 5)
    lse_taus: Tuple[float, ...] = (0.05, 0.10, 0.20)
    hub_prior_topm: int = 20
    density_k_values: Tuple[int, ...] = (10, 20)
    mmr_top_l: int = 50
    show_progress: bool = True
    write_row_level_debug: bool = False


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


def _progress(iterable, *, enabled: bool, desc: str):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, desc=desc, dynamic_ncols=True)
    return iterable


def _default_output_dir(run_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(run_root).expanduser().resolve() / "analysis" / "extra_mining_simulator" / stamp


def _checkpoint_path(run_root: Path, stage: str) -> Path:
    if stage == "prealign":
        return run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if stage == "softem_base":
        return run_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth"
    if stage == "softem_aug":
        return run_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth"
    raise ValueError(f"unsupported checkpoint_stage: {stage}")


def _load_project_imports():
    # Lazy imports keep --self_test and py_compile independent of repo assets.
    from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_carrier_evidence, load_text_vocab
    from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import _compute_t_dis, _load_reservoir_checkpoint, _normalize_np, _project_text_matrix
    from videocutler.ext_stageb_ovvis.analysis.extra_attribution_probe import (
        _apply_stage_candidate_overrides,
        _load_stage_responsibility_candidate_overrides,
        _prepare_probe_examples,
    )
    from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import _canonical_sidecar_gt_raw_id
    from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label
    from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup
    from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_base_and_novel_raw_ids
    from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
        Phase1MaterializationConfig,
        materialize_phase1_training_samples,
    )

    return {
        "load_carrier_evidence": load_carrier_evidence,
        "load_text_vocab": load_text_vocab,
        "_compute_t_dis": _compute_t_dis,
        "_load_reservoir_checkpoint": _load_reservoir_checkpoint,
        "_normalize_np": _normalize_np,
        "_project_text_matrix": _project_text_matrix,
        "_apply_stage_candidate_overrides": _apply_stage_candidate_overrides,
        "_load_stage_responsibility_candidate_overrides": _load_stage_responsibility_candidate_overrides,
        "_prepare_probe_examples": _prepare_probe_examples,
        "_canonical_sidecar_gt_raw_id": _canonical_sidecar_gt_raw_id,
        "_all_gt_split_label": _all_gt_split_label,
        "load_gt_sidecar_lookup": load_gt_sidecar_lookup,
        "load_lvvis_base_and_novel_raw_ids": load_lvvis_base_and_novel_raw_ids,
        "Phase1MaterializationConfig": Phase1MaterializationConfig,
        "materialize_phase1_training_samples": materialize_phase1_training_samples,
    }


def _materialize_examples(config: ExtraMiningSimulatorConfig, imports: Mapping[str, Any]) -> Tuple[List[Record], Dict[str, Any]]:
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    materialized = imports["materialize_phase1_training_samples"](
        runtime_output_root,
        imports["Phase1MaterializationConfig"](
            dataset_name=str(config.dataset_name),
            trajectory_source_branch=str(config.trajectory_source_branch),
            smoke=bool(config.smoke),
            smoke_max_trajectories=int(config.smoke_max_trajectories),
            subset_fraction=None if config.subset_fraction is None else float(config.subset_fraction),
        ),
    )
    valid_samples = list(materialized.get("valid_samples", []))
    prepared = imports["_prepare_probe_examples"](
        valid_samples,
        output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    examples = [dict(ex) for ex in list(prepared.get("examples", []))]
    if not examples:
        raise RuntimeError("no valid examples available for extra mining simulator")
    meta = {
        "materialized_stats": dict(materialized.get("stats", {})),
        "prepare_skipped_reason_histogram": dict(prepared.get("skipped_reason_histogram", {})),
        "example_count": int(len(examples)),
    }
    return examples, meta


def _apply_existing_stage_extras(
    *,
    examples: Sequence[Mapping[str, Any]],
    run_root: Path,
    stage_id: str,
    imports: Mapping[str, Any],
) -> Tuple[List[Record], Dict[str, Any]]:
    overrides, override_meta = imports["_load_stage_responsibility_candidate_overrides"](run_root=run_root, stage_id=str(stage_id))
    effective, apply_meta = imports["_apply_stage_candidate_overrides"](examples, overrides, stage_id=str(stage_id))
    return [dict(ex) for ex in effective], {"override_meta": dict(override_meta), "apply_meta": dict(apply_meta)}


def _formal_rows(
    *,
    examples: Sequence[Mapping[str, Any]],
    existing_examples_by_tid: Mapping[str, Mapping[str, Any]],
    text_vocab_ids: Sequence[int],
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    dataset_name: str,
    formal_split: str,
    imports: Mapping[str, Any],
) -> List[Record]:
    base_vocab_ids, _novel_vocab_ids = imports["load_lvvis_base_and_novel_raw_ids"]()
    base_vocab_set = {int(x) for x in base_vocab_ids}
    vocab_set = {int(x) for x in text_vocab_ids}
    rows: List[Record] = []
    for example in examples:
        tid = str(example.get("trajectory_id", "")).strip()
        if not tid:
            continue
        sidecar = dict(sidecar_lookup.get(tid, {})) if tid else {}
        gt_raw_id = imports["_canonical_sidecar_gt_raw_id"](sidecar) if sidecar else None
        if gt_raw_id is None:
            continue
        gt_raw_id = int(gt_raw_id)
        observed_raw_ids = [int(x) for x in list(example.get("observed_raw_ids", []))]
        split = imports["_all_gt_split_label"](
            dataset_name=str(dataset_name),
            gt_raw_id=int(gt_raw_id),
            observed_raw_ids=observed_raw_ids,
            base_vocab_ids=base_vocab_set,
        )
        if str(split) != str(formal_split):
            continue
        if int(gt_raw_id) not in vocab_set:
            continue
        existing = dict(existing_examples_by_tid.get(tid, {}))
        existing_extra = _unique_int_list(existing.get("candidate_ids_extra", []))
        existing_known = _unique_int_list(existing.get("candidate_ids_known", example.get("candidate_ids_known", [])))
        rows.append(
            {
                "trajectory_id": tid,
                "clip_id": int(example.get("clip_id", -1)),
                "video_id": int(example.get("video_id", -1)),
                "gt_raw_id": int(gt_raw_id),
                "observed_raw_ids": observed_raw_ids,
                "candidate_ids_known": _unique_int_list(example.get("candidate_ids_known", [])),
                "existing_candidate_ids_known": existing_known,
                "existing_candidate_ids_extra": [int(x) for x in existing_extra if int(x) not in {int(y) for y in existing_known}],
                "carrier_vec": np.asarray(example.get("carrier_vec"), dtype=np.float32),
            }
        )
    return rows



def _read_jsonl_records(path: Path) -> List[Record]:
    records: List[Record] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(dict(payload))
    return records


def _trajectory_id_from_formal_record(record: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "trajectory_id",
        "main_trajectory_id",
        "row_trajectory_id",
        "formal_trajectory_id",
        "trajectory_key",
    ):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _record_matches_formal_split(record: Mapping[str, Any], formal_split: str) -> bool:
    for key in ("split", "split_name", "formal_split", "minimal_split", "subset"):
        value = record.get(key)
        if value is not None:
            return str(value) == str(formal_split)
    return True


def _discover_formal_row_diagnostics_path(config: ExtraMiningSimulatorConfig) -> Optional[Path]:
    if config.formal_row_diagnostics_path is not None:
        return Path(config.formal_row_diagnostics_path).expanduser().resolve()
    run_root = Path(config.run_root).expanduser().resolve()
    direct = (
        run_root
        / "analysis"
        / "extra_mining_recall_diagnosis"
        / str(config.dataset_name)
        / str(config.stage_id)
        / "formal_aligned_row_diagnostics.jsonl"
    )
    if direct.exists():
        return direct
    root = run_root / "analysis" / "extra_mining_recall_diagnosis"
    if not root.exists():
        return None
    candidates = sorted(root.rglob("formal_aligned_row_diagnostics.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _apply_formal_authority_row_filter(
    *,
    rows: Sequence[Mapping[str, Any]],
    config: ExtraMiningSimulatorConfig,
) -> Tuple[List[Record], Dict[str, Any]]:
    path = _discover_formal_row_diagnostics_path(config)
    if path is None or not path.exists():
        meta = {
            "status": "NO_FORMAL_ROW_DIAGNOSTICS_FOUND",
            "path": None if path is None else str(path),
            "input_row_count": int(len(rows)),
            "filtered_row_count": int(len(rows)),
        }
        if config.expected_formal_gt_count is not None and int(len(rows)) != int(config.expected_formal_gt_count):
            raise RuntimeError(
                "formal row authority missing: split-derived row count is "
                f"{len(rows)} but expected {config.expected_formal_gt_count}. "
                "Run/provide extra_mining_recall_diagnosis/.../formal_aligned_row_diagnostics.jsonl "
                "or pass --formal_row_diagnostics_path explicitly. Refusing to use the larger probe universe."
            )
        return [dict(r) for r in rows], meta

    authority_records = _read_jsonl_records(path)
    authority_ids = {
        tid
        for rec in authority_records
        if _record_matches_formal_split(rec, str(config.formal_split))
        for tid in [_trajectory_id_from_formal_record(rec)]
        if tid is not None
    }
    if not authority_ids:
        raise RuntimeError(f"formal row diagnostics exists but no usable trajectory_id records were found: {path}")
    row_by_tid = {str(row.get("trajectory_id", "")): dict(row) for row in rows if str(row.get("trajectory_id", "")).strip()}
    filtered = [row_by_tid[tid] for tid in sorted(authority_ids) if tid in row_by_tid]
    missing_ids = sorted(tid for tid in authority_ids if tid not in row_by_tid)
    meta = {
        "status": "PASS" if not missing_ids else "MISSING_JOINED_ROWS",
        "path": str(path),
        "authority_record_count": int(len(authority_records)),
        "authority_id_count": int(len(authority_ids)),
        "input_row_count": int(len(rows)),
        "filtered_row_count": int(len(filtered)),
        "missing_join_count": int(len(missing_ids)),
        "missing_join_examples": missing_ids[:20],
    }
    if missing_ids:
        raise RuntimeError(
            f"formal row diagnostics authority has {len(authority_ids)} ids, "
            f"but {len(missing_ids)} are missing from simulator rows; examples={missing_ids[:10]}"
        )
    return filtered, meta

def _unique_int_list(values: Sequence[Any]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for value in list(values or []):
        try:
            item = int(value)
        except Exception:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_clip_groups(examples: Sequence[Mapping[str, Any]]) -> Tuple[List[int], Dict[int, List[Mapping[str, Any]]]]:
    by_clip: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_clip[int(ex.get("clip_id", -1))].append(ex)
    clip_ids = sorted(int(x) for x in by_clip.keys())
    return clip_ids, dict(by_clip)


def _project_text(
    *,
    checkpoint_path: Path,
    text_vocab_matrix: np.ndarray,
    device: torch.device,
    imports: Mapping[str, Any],
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    projector, theta_t, _unknown, checkpoint = imports["_load_reservoir_checkpoint"](Path(checkpoint_path), device=device)
    projector.eval()
    with torch.no_grad():
        text_proj = imports["_project_text_matrix"](projector, np.asarray(text_vocab_matrix, dtype=np.float32), device=device)
        text_proj = F.normalize(text_proj, p=2.0, dim=-1)
        temperature = float(imports["_compute_t_dis"](theta_t).detach().cpu().item())
    return text_proj, temperature, {"checkpoint_stage_id": str(checkpoint.get("stage_id", "")), "checkpoint_path": str(checkpoint_path)}


def _prepare_tensor_cache(
    *,
    config: ExtraMiningSimulatorConfig,
    examples: Sequence[Mapping[str, Any]],
    text_vocab_ids: Sequence[int],
    text_projected: torch.Tensor,
    temperature: float,
    device: torch.device,
    imports: Mapping[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], List[int]]:
    raw_ids = [int(x) for x in text_vocab_ids]
    raw_to_col = {int(raw_id): int(idx) for idx, raw_id in enumerate(raw_ids)}
    clip_ids, by_clip = _build_clip_groups(examples)
    vocab_size = int(len(raw_ids))
    num_clips = int(len(clip_ids))
    text_sim = torch.matmul(text_projected, text_projected.t()).clamp(min=-1.0, max=1.0)

    agg_max = torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32)
    topm_tensors = {int(m): torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32) for m in config.topm_values}
    lse_tensors = {float(t): torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32) for t in config.lse_taus}
    residual_topm3 = torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32)
    support_p90 = torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32)
    support_p95 = torch.empty((num_clips, vocab_size), device=device, dtype=torch.float32)
    obs_sim = torch.zeros((num_clips, vocab_size), device=device, dtype=torch.float32)
    yprime_mask = torch.zeros((num_clips, vocab_size), device=device, dtype=torch.bool)

    batch_size = max(1, int(config.batch_size_clips))
    for start in _progress(range(0, num_clips, batch_size), enabled=bool(config.show_progress), desc="extra-mining simulator: clip batches"):
        end = min(num_clips, int(start) + batch_size)
        batch_clip_ids = clip_ids[start:end]
        groups = [by_clip[int(cid)] for cid in batch_clip_ids]
        rmax = max(len(group) for group in groups)
        carriers = torch.zeros((len(groups), rmax, int(text_projected.shape[1])), device=device, dtype=torch.float32)
        valid = torch.zeros((len(groups), rmax), device=device, dtype=torch.bool)
        for bi, group in enumerate(groups):
            for ri, ex in enumerate(group):
                arr = imports["_normalize_np"](np.asarray(ex["carrier_vec"], dtype=np.float32))
                carriers[bi, ri] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                valid[bi, ri] = True
        carriers = F.normalize(carriers, p=2.0, dim=-1)
        logits = torch.matmul(carriers, text_projected.t()) / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~valid.reshape(len(groups), rmax, 1), float("-inf"))
        batch_known_mask = torch.zeros((len(groups), vocab_size), device=device, dtype=torch.bool)
        for bi, group in enumerate(groups):
            known_ids: List[int] = []
            for ex in group:
                known_ids.extend(_unique_int_list(ex.get("candidate_ids_known", [])))
                known_ids.extend(_unique_int_list(ex.get("observed_raw_ids", [])))
            known_cols = sorted({raw_to_col[int(rid)] for rid in known_ids if int(rid) in raw_to_col})
            if known_cols:
                cols = torch.as_tensor(known_cols, device=device, dtype=torch.long)
                batch_known_mask[bi, cols] = True
                obs_sim[batch_slice := slice(start, end)][bi, :] = torch.max(text_sim[:, cols], dim=1).values
        yprime_mask[start:end] = batch_known_mask
        agg_max[start:end] = torch.max(logits, dim=1).values
        finite_logits = logits.masked_fill(~torch.isfinite(logits), -1e9)
        valid_counts = valid.float().sum(dim=1).clamp(min=1.0)
        for m in config.topm_values:
            k = min(int(m), rmax)
            vals = torch.topk(finite_logits, k=k, dim=1).values
            vals = torch.where(vals < -1e8, torch.zeros_like(vals), vals)
            denom = torch.clamp(valid_counts, max=float(k)).reshape(len(groups), 1)
            topm_tensors[int(m)][start:end] = vals.sum(dim=1) / denom
        for tau in config.lse_taus:
            tau_f = max(float(tau), 1e-6)
            lse_tensors[float(tau)][start:end] = tau_f * torch.logsumexp(finite_logits / tau_f, dim=1)
        # support: row-level top percentile threshold over classes.
        kth90 = max(1, int(math.ceil(0.10 * vocab_size)))
        kth95 = max(1, int(math.ceil(0.05 * vocab_size)))
        thr90 = torch.topk(finite_logits, k=kth90, dim=2).values[:, :, -1]
        thr95 = torch.topk(finite_logits, k=kth95, dim=2).values[:, :, -1]
        support_p90[start:end] = ((finite_logits >= thr90.unsqueeze(-1)) & valid.unsqueeze(-1)).float().sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        support_p95[start:end] = ((finite_logits >= thr95.unsqueeze(-1)) & valid.unsqueeze(-1)).float().sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        # residual over observed visual explanation.
        best_yprime = torch.full((len(groups), rmax), fill_value=0.0, device=device, dtype=torch.float32)
        for bi in range(len(groups)):
            cols = torch.nonzero(batch_known_mask[bi], as_tuple=False).reshape(-1)
            if int(cols.numel()) > 0:
                best_yprime[bi] = torch.max(finite_logits[bi, :, cols], dim=1).values
        residual = torch.relu(finite_logits - best_yprime.unsqueeze(-1)).masked_fill(~valid.unsqueeze(-1), 0.0)
        k_res = min(3, rmax)
        vals_res = torch.topk(residual, k=k_res, dim=1).values
        denom_res = torch.clamp(valid_counts, max=float(k_res)).reshape(len(groups), 1)
        residual_topm3[start:end] = vals_res.sum(dim=1) / denom_res

    cache: Dict[str, torch.Tensor] = {
        "agg_max": agg_max,
        "obs_sim": obs_sim,
        "yprime_mask": yprime_mask,
        "text_sim": text_sim,
        "support_p90": support_p90,
        "support_p95": support_p95,
        "residual_topm3": residual_topm3,
        "genericity_prior": torch.mean(text_sim, dim=1),
    }
    for m, tensor in topm_tensors.items():
        cache[f"agg_topm{int(m)}"] = tensor
    for tau, tensor in lse_tensors.items():
        cache[f"agg_lse_tau{tau_key(float(tau))}"] = tensor
    _populate_density_and_hub_priors(cache, config=config, device=device)
    return cache, raw_to_col, clip_ids


def _populate_density_and_hub_priors(cache: Dict[str, torch.Tensor], *, config: ExtraMiningSimulatorConfig, device: torch.device) -> None:
    base_for_prior = cache["agg_max"] - float(config.alpha_obs) * cache["obs_sim"]
    masked = base_for_prior.masked_fill(cache["yprime_mask"], float("-inf"))
    topm = min(max(1, int(config.hub_prior_topm)), int(masked.shape[1]))
    top_cols = torch.topk(masked, k=topm, dim=1).indices
    freq = torch.zeros((int(masked.shape[1]),), device=device, dtype=torch.float32)
    freq.scatter_add_(0, top_cols.reshape(-1), torch.ones((top_cols.numel(),), device=device, dtype=torch.float32))
    freq = freq / max(float(masked.shape[0] * topm), 1.0)
    cache["hub_prior_freq"] = freq
    cache["hub_prior_mean"] = torch.mean(cache["agg_max"], dim=0)
    cache["hub_prior_freq_zscore"] = _zscore(freq)
    cache["hub_prior_mean_zscore"] = _zscore(cache["hub_prior_mean"])

    aggregations: Dict[str, torch.Tensor] = {
        "max": cache["agg_max"],
        "topm3": cache.get("agg_topm3", cache["agg_max"]),
        "blend_rho0p5_topm3": 0.5 * cache["agg_max"] + 0.5 * cache.get("agg_topm3", cache["agg_max"]),
        "residual_topm3": cache.get("residual_topm3", cache["agg_max"]),
    }
    for name, matrix in aggregations.items():
        for k in config.density_k_values:
            kk_cls = min(int(k), int(matrix.shape[1]))
            kk_clip = min(int(k), int(matrix.shape[0]))
            cache[f"density_query_{name}_k{int(k)}"] = torch.topk(matrix, k=kk_cls, dim=1).values.mean(dim=1)
            cache[f"density_class_{name}_k{int(k)}"] = torch.topk(matrix, k=kk_clip, dim=0).values.mean(dim=0)
        # class-to-clip rank approximation: lower rank means the class is specific to this clip.
        order = torch.argsort(matrix, dim=0, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        rank_values = torch.arange(1, int(matrix.shape[0]) + 1, device=matrix.device, dtype=torch.float32).reshape(-1, 1).expand_as(order)
        ranks.scatter_(0, order, rank_values)
        cache[f"class_to_clip_rank_{name}"] = ranks


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - torch.mean(x)) / torch.clamp(torch.std(x), min=float(eps))


def _existing_baseline(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gt_in = [int(row["gt_raw_id"]) in {int(x) for x in list(row.get("existing_candidate_ids_extra", []))} for row in rows]
    count = int(sum(1 for x in gt_in if x))
    total = int(len(gt_in))
    return {
        "variant": "existing_stage_extra",
        "formal_gt_count": total,
        "gt_in_extra_count": count,
        "gt_not_in_extra_count": int(total - count),
        "gt_in_extra_rate": float(count / max(total, 1)),
    }


def _evaluate_variant(
    *,
    variant: ExtraMiningScoreVariant,
    cache: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    raw_to_col: Mapping[int, int],
    clip_ids: Sequence[int],
    raw_ids: Sequence[int],
    k_values: Sequence[int],
    primary_k: int,
    device: torch.device,
    person_cols: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[List[Record]]]:
    clip_to_index = {int(cid): int(idx) for idx, cid in enumerate(clip_ids)}
    row_clip_idx = torch.as_tensor([clip_to_index[int(row["clip_id"])] for row in rows], device=device, dtype=torch.long)
    gt_cols = torch.as_tensor([int(raw_to_col[int(row["gt_raw_id"])]) for row in rows], device=device, dtype=torch.long)
    max_k = max(int(k) for k in k_values)
    yprime_mask = cache["yprime_mask"]
    obs_sim = cache["obs_sim"]
    if variant.family == "mmr":
        # MMR uses a blend base score and returns explicit selected top-k columns.
        base_variant = ExtraMiningScoreVariant(name="mmr_base", family="blend", aggregation=variant.aggregation, topm=variant.topm, blend_rho=variant.blend_rho, alpha_obs=variant.alpha_obs)
        base_score = score_variant(cache, base_variant)
        selected_cols = mmr_select_topk(
            base_score,
            yprime_mask=yprime_mask,
            obs_sim=obs_sim,
            obs_sim_max=variant.obs_sim_max,
            text_sim=cache["text_sim"],
            k=max_k,
            top_l=int(variant.mmr_top_l),
            lambda_text=float(variant.mmr_lambda_text),
        )
        masked_score = apply_candidate_masks(base_score, yprime_mask=yprime_mask, obs_sim=obs_sim, obs_sim_max=variant.obs_sim_max)
    else:
        score = score_variant(cache, variant)
        masked_score = apply_candidate_masks(score, yprime_mask=yprime_mask, obs_sim=obs_sim, obs_sim_max=variant.obs_sim_max)
        selected_cols = torch.topk(masked_score, k=max_k, dim=1).indices
    formal_selected = selected_cols[row_clip_idx]
    eval_by_k: Dict[str, Any] = {}
    primary_hits: Optional[torch.Tensor] = None
    for k in k_values:
        kk = int(k)
        hits = torch.any(formal_selected[:, :kk] == gt_cols.reshape(-1, 1), dim=1)
        if kk == int(primary_k):
            primary_hits = hits
        hit_count = int(hits.detach().cpu().sum().item())
        total = int(len(rows))
        eval_by_k[str(kk)] = {
            "gt_in_extra_count": hit_count,
            "gt_not_in_extra_count": int(total - hit_count),
            "gt_in_extra_rate": float(hit_count / max(total, 1)),
        }
    if primary_hits is None:
        primary_hits = torch.any(formal_selected[:, : int(primary_k)] == gt_cols.reshape(-1, 1), dim=1)
    gt_scores = masked_score[row_clip_idx, gt_cols]
    finite_gt = torch.isfinite(gt_scores)
    ranks = 1 + torch.sum(masked_score[row_clip_idx] > gt_scores.reshape(-1, 1), dim=1)
    ranks = torch.where(finite_gt, ranks, torch.full_like(ranks, fill_value=int(masked_score.shape[1]) + 1))
    metrics = {
        "variant": str(variant.name),
        "family": str(variant.family),
        "primary_k": int(primary_k),
        "formal_gt_count": int(len(rows)),
        "gt_in_extra_count": int(primary_hits.detach().cpu().sum().item()),
        "gt_not_in_extra_count": int(len(rows) - int(primary_hits.detach().cpu().sum().item())),
        "gt_in_extra_rate": float(primary_hits.float().mean().detach().cpu().item()) if rows else None,
        "recall_at_k": eval_by_k,
        "mean_gt_mining_rank": float(ranks.float().mean().detach().cpu().item()) if rows else None,
        "median_gt_mining_rank": float(torch.median(ranks.float()).detach().cpu().item()) if rows else None,
        "far_miss_rate_rank_gt_50": float((ranks > 50).float().mean().detach().cpu().item()) if rows else None,
        "far_miss_rate_rank_gt_100": float((ranks > 100).float().mean().detach().cpu().item()) if rows else None,
    }
    hub_report = _hub_report(formal_selected[:, : int(primary_k)], raw_ids=raw_ids, person_cols=person_cols)
    by_class = _by_gt_class_report(rows, primary_hits.detach().cpu().numpy().astype(bool).tolist(), formal_selected[:, 0].detach().cpu().numpy().astype(int).tolist(), raw_ids)
    suppressor = _suppressor_report(rows, primary_hits.detach().cpu().numpy().astype(bool).tolist(), formal_selected[:, 0].detach().cpu().numpy().astype(int).tolist(), raw_ids)
    return metrics, hub_report, by_class, suppressor, None


def _hub_report(selected_cols: torch.Tensor, *, raw_ids: Sequence[int], person_cols: Sequence[int]) -> Dict[str, Any]:
    arr = selected_cols.detach().cpu().numpy().astype(int)
    total_slots = int(arr.size)
    flat = arr.reshape(-1).tolist()
    counts = Counter(int(raw_ids[int(col)]) for col in flat)
    top1_counts = Counter(int(raw_ids[int(col)]) for col in arr[:, 0].tolist()) if arr.size else Counter()
    entropy, eff = _entropy_effective_count(counts)
    person_col_set = {int(x) for x in person_cols}
    person_slots = int(sum(1 for col in flat if int(col) in person_col_set))
    person_top1 = int(sum(1 for col in arr[:, 0].tolist() if int(col) in person_col_set)) if arr.size else 0
    return {
        "selected_slot_count": total_slots,
        "selected_extra_class_freq_top50": _counter_top(counts, 50),
        "selected_extra_distribution_entropy": entropy,
        "effective_num_selected_classes": eff,
        "person_selected_count": person_slots,
        "person_selected_rate": float(person_slots / max(total_slots, 1)),
        "person_top1_extra_count": person_top1,
        "person_top1_extra_rate": float(person_top1 / max(int(arr.shape[0]), 1)) if arr.size else 0.0,
        "top1_extra_class_freq_top50": _counter_top(top1_counts, 50),
    }


def _by_gt_class_report(rows: Sequence[Mapping[str, Any]], hits: Sequence[bool], top1_cols: Sequence[int], raw_ids: Sequence[int]) -> Dict[str, Any]:
    by_gt: Dict[int, Dict[str, Any]] = {}
    for row, hit, top_col in zip(rows, hits, top1_cols):
        gt = int(row["gt_raw_id"])
        bucket = by_gt.setdefault(gt, {"gt_raw_id": gt, "formal_gt_count": 0, "hit_count": 0, "top_suppressor_counter": Counter()})
        bucket["formal_gt_count"] += 1
        bucket["hit_count"] += int(bool(hit))
        if not bool(hit):
            bucket["top_suppressor_counter"][int(raw_ids[int(top_col)])] += 1
    records = []
    for gt, payload in sorted(by_gt.items(), key=lambda kv: (-kv[1]["formal_gt_count"], kv[0])):
        total = int(payload["formal_gt_count"])
        hit_count = int(payload["hit_count"])
        records.append(
            {
                "gt_raw_id": int(gt),
                "formal_gt_count": total,
                "gt_in_extra_count": hit_count,
                "gt_in_extra_rate": float(hit_count / max(total, 1)),
                "top_suppressor_classes": _counter_top(payload["top_suppressor_counter"], 5),
            }
        )
    return {"by_gt_class": records}


def _suppressor_report(rows: Sequence[Mapping[str, Any]], hits: Sequence[bool], top1_cols: Sequence[int], raw_ids: Sequence[int]) -> Dict[str, Any]:
    counter = Counter()
    for hit, top_col in zip(hits, top1_cols):
        if not bool(hit):
            counter[int(raw_ids[int(top_col)])] += 1
    return {"top_suppressor_classes": _counter_top(counter, 50)}


def _counter_top(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    return [{"raw_id": int(key), "count": int(value)} for key, value in counter.most_common(int(limit))]


def _entropy_effective_count(counter: Counter) -> Tuple[float, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0, 0.0
    probs = np.asarray([float(v) / total for v in counter.values()], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return entropy, float(np.exp(entropy))


def _find_person_cols(text_records: Sequence[Mapping[str, Any]], raw_to_col: Mapping[int, int]) -> List[int]:
    cols: List[int] = []
    for record in text_records:
        raw_id = int(record.get("raw_id")) if record.get("raw_id") is not None else None
        if raw_id is None or raw_id not in raw_to_col:
            continue
        names = []
        for key in ("name", "class_name", "category_name", "synset", "label"):
            if record.get(key) is not None:
                names.append(str(record.get(key)).strip().lower())
        if any(name == "person" or name.endswith(" person") or name.startswith("person,") for name in names):
            cols.append(int(raw_to_col[int(raw_id)]))
    return sorted(set(cols))


def run_extra_mining_simulator(config: ExtraMiningSimulatorConfig) -> Dict[str, Any]:
    imports = _load_project_imports()
    run_root = Path(config.run_root).expanduser().resolve()
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir else _default_output_dir(run_root)
    device = torch.device(str(config.device))
    examples, materialization_meta = _materialize_examples(config, imports)
    existing_examples, existing_meta = _apply_existing_stage_extras(examples=examples, run_root=run_root, stage_id=str(config.stage_id), imports=imports)
    existing_by_tid = {str(ex.get("trajectory_id", "")): dict(ex) for ex in existing_examples}
    text_vocab_ids, text_records, text_vocab_matrix = imports["load_text_vocab"](runtime_output_root)
    sidecar_root = Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root else run_root
    sidecar_lookup = imports["load_gt_sidecar_lookup"](sidecar_root, dataset_name=str(config.dataset_name), trajectory_source_branch=str(config.trajectory_source_branch))
    rows = _formal_rows(
        examples=examples,
        existing_examples_by_tid=existing_by_tid,
        text_vocab_ids=text_vocab_ids,
        sidecar_lookup=sidecar_lookup,
        dataset_name=str(config.dataset_name),
        formal_split=str(config.formal_split),
        imports=imports,
    )
    if not rows:
        raise RuntimeError("no formal rows available for simulator")
    rows, formal_row_authority_meta = _apply_formal_authority_row_filter(rows=rows, config=config)
    if not rows:
        raise RuntimeError("formal row authority filtering produced no rows")
    existing = _existing_baseline(rows)
    _validate_existing_baseline(existing, config)
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve() if config.checkpoint_path else _checkpoint_path(run_root, str(config.checkpoint_stage))
    text_projected, temperature, ckpt_meta = _project_text(checkpoint_path=checkpoint_path, text_vocab_matrix=np.asarray(text_vocab_matrix, dtype=np.float32), device=device, imports=imports)
    cache, raw_to_col, clip_ids = _prepare_tensor_cache(
        config=config,
        examples=examples,
        text_vocab_ids=text_vocab_ids,
        text_projected=text_projected,
        temperature=float(temperature),
        device=device,
        imports=imports,
    )
    raw_ids = [int(x) for x in text_vocab_ids]
    person_cols = _find_person_cols(text_records, raw_to_col)
    variants = parse_variant_names(config.variant_names) if config.variant_names else build_default_variants()
    variant_results: List[Dict[str, Any]] = []
    hub_reports: Dict[str, Any] = {}
    by_class_reports: Dict[str, Any] = {}
    suppressor_reports: Dict[str, Any] = {}
    for variant in _progress(variants, enabled=bool(config.show_progress), desc="extra-mining simulator: variants"):
        metrics, hub, by_class, suppressor, row_debug = _evaluate_variant(
            variant=variant,
            cache=cache,
            rows=rows,
            raw_to_col=raw_to_col,
            clip_ids=clip_ids,
            raw_ids=raw_ids,
            k_values=config.k_values,
            primary_k=int(config.primary_k),
            device=device,
            person_cols=person_cols,
        )
        metrics["delta_gt_in_extra_rate_vs_existing"] = None if metrics.get("gt_in_extra_rate") is None else float(metrics["gt_in_extra_rate"] - existing["gt_in_extra_rate"])
        metrics["delta_gt_not_in_extra_count_vs_existing"] = int(metrics["gt_not_in_extra_count"] - existing["gt_not_in_extra_count"])
        variant_results.append(metrics)
        hub_reports[str(variant.name)] = hub
        by_class_reports[str(variant.name)] = by_class
        suppressor_reports[str(variant.name)] = suppressor
        if config.write_row_level_debug and row_debug is not None:
            _write_jsonl(output_dir / "row_debug" / f"{variant.name}.jsonl", row_debug)
    variant_results_sorted = sorted(variant_results, key=lambda row: (-(row.get("gt_in_extra_rate") or -1.0), row.get("variant", "")))
    summary = {
        "status": "PASS",
        "run_root": str(run_root),
        "runtime_output_root": str(runtime_output_root),
        "dataset_name": str(config.dataset_name),
        "formal_split": str(config.formal_split),
        "stage_id": str(config.stage_id),
        "checkpoint": ckpt_meta,
        "temperature": float(temperature),
        "formal_gt_count": int(len(rows)),
        "existing_stage_extra_baseline": existing,
        "variant_count": int(len(variant_results)),
        "variants": variant_results_sorted,
        "materialization_meta": materialization_meta,
        "existing_candidate_meta": existing_meta,
        "formal_row_authority_meta": formal_row_authority_meta,
        "person_cols": [int(x) for x in person_cols],
        "config": _config_payload(config),
    }
    paths = _write_outputs(output_dir, summary, variant_results_sorted, hub_reports, by_class_reports, suppressor_reports)
    return {"status": "PASS", "output_dir": str(output_dir), **paths, "summary": summary}


def _validate_existing_baseline(existing: Mapping[str, Any], config: ExtraMiningSimulatorConfig) -> None:
    if config.expected_formal_gt_count is not None and int(existing.get("formal_gt_count", -1)) != int(config.expected_formal_gt_count):
        msg = f"formal_gt_count mismatch: got {existing.get('formal_gt_count')}, expected {config.expected_formal_gt_count}"
        if bool(config.enforce_expected_baseline):
            raise RuntimeError(msg)
    if config.expected_existing_gt_in_extra_rate is not None:
        got = float(existing.get("gt_in_extra_rate", -1.0))
        expected = float(config.expected_existing_gt_in_extra_rate)
        if abs(got - expected) > float(config.expected_rate_tolerance):
            msg = f"existing gt_in_extra_rate mismatch: got {got:.6f}, expected {expected:.6f}±{config.expected_rate_tolerance}"
            if bool(config.enforce_expected_baseline):
                raise RuntimeError(msg)


def _config_payload(config: ExtraMiningSimulatorConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    hub_reports: Mapping[str, Any],
    by_class_reports: Mapping[str, Any],
    suppressor_reports: Mapping[str, Any],
) -> Dict[str, str]:
    metrics_path = output_dir / "metrics_summary.json"
    variant_table_path = output_dir / "variant_table.md"
    recall_path = output_dir / "recall_at_k.json"
    hub_path = output_dir / "hub_report.json"
    by_class_path = output_dir / "by_gt_class_recall.json"
    suppressor_path = output_dir / "suppressor_report.json"
    manifest_path = output_dir / "simulator_manifest.json"
    failure_path = output_dir / "failure_buckets.json"
    _write_json(metrics_path, summary)
    _write_json(hub_path, hub_reports)
    _write_json(by_class_path, by_class_reports)
    _write_json(suppressor_path, suppressor_reports)
    _write_json(recall_path, {str(row.get("variant")): row.get("recall_at_k", {}) for row in variants})
    _write_json(failure_path, {str(row.get("variant")): {"gt_in_extra_count": row.get("gt_in_extra_count"), "gt_not_in_extra_count": row.get("gt_not_in_extra_count")} for row in variants})
    _write_json(manifest_path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "files": {"metrics_summary": str(metrics_path), "variant_table": str(variant_table_path), "recall_at_k": str(recall_path), "hub_report": str(hub_path), "by_gt_class_recall": str(by_class_path), "suppressor_report": str(suppressor_path), "failure_buckets": str(failure_path)}})
    _write_markdown(variant_table_path, _variant_table_lines(summary, variants))
    return {
        "metrics_summary_path": str(metrics_path),
        "variant_table_path": str(variant_table_path),
        "recall_at_k_path": str(recall_path),
        "hub_report_path": str(hub_path),
        "by_gt_class_recall_path": str(by_class_path),
        "suppressor_report_path": str(suppressor_path),
        "failure_buckets_path": str(failure_path),
        "simulator_manifest_path": str(manifest_path),
    }


def _variant_table_lines(summary: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> List[str]:
    existing = dict(summary.get("existing_stage_extra_baseline", {}))
    lines = [
        "# Formal-aligned Extra Mining Simulator",
        "",
        f"- formal_gt_count: `{summary.get('formal_gt_count')}`",
        f"- existing_stage_extra gt_in_extra_rate: `{existing.get('gt_in_extra_rate')}`",
        f"- existing_stage_extra gt_not_in_extra_count: `{existing.get('gt_not_in_extra_count')}`",
        "",
        "| rank | variant | family | gt_in_extra_rate | delta_rate | gt_not_in_extra_count | delta_not_in | mean_gt_rank | person_selected_rate | effective_selected_classes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(variants, start=1):
        lines.append(
            "| {idx} | {variant} | {family} | {rate} | {delta} | {not_in} | {delta_not} | {rank} | {person} | {eff} |".format(
                idx=idx,
                variant=row.get("variant"),
                family=row.get("family"),
                rate=_fmt(row.get("gt_in_extra_rate")),
                delta=_fmt(row.get("delta_gt_in_extra_rate_vs_existing")),
                not_in=row.get("gt_not_in_extra_count"),
                delta_not=row.get("delta_gt_not_in_extra_count_vs_existing"),
                rank=_fmt(row.get("mean_gt_mining_rank")),
                person="see hub_report.json",
                eff="see hub_report.json",
            )
        )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def synthetic_self_test() -> Dict[str, Any]:
    from videocutler.ext_stageb_ovvis.analysis.extra_mining_score_registry import _synthetic_self_test

    registry = _synthetic_self_test()
    # Minimal evaluator sanity check.
    device = torch.device("cpu")
    raw_ids = list(range(10, 19))
    rows = [
        {"trajectory_id": "t0", "clip_id": 1, "gt_raw_id": 12},
        {"trajectory_id": "t1", "clip_id": 1, "gt_raw_id": 13},
        {"trajectory_id": "t2", "clip_id": 2, "gt_raw_id": 14},
    ]
    cache = {
        "agg_max": torch.randn(2, 9),
        "agg_topm2": torch.randn(2, 9),
        "agg_topm3": torch.randn(2, 9),
        "agg_topm5": torch.randn(2, 9),
        "agg_lse_tau0p05": torch.randn(2, 9),
        "agg_lse_tau0p1": torch.randn(2, 9),
        "agg_lse_tau0p2": torch.randn(2, 9),
        "obs_sim": torch.rand(2, 9),
        "yprime_mask": torch.zeros(2, 9, dtype=torch.bool),
        "hub_prior_freq": torch.rand(9),
        "hub_prior_mean": torch.rand(9),
        "hub_prior_freq_zscore": torch.rand(9),
        "hub_prior_mean_zscore": torch.rand(9),
        "genericity_prior": torch.rand(9),
        "text_sim": torch.eye(9),
        "support_p95": torch.rand(2, 9),
        "residual_topm3": torch.rand(2, 9),
        "density_query_blend_rho0p5_topm3_k10": torch.rand(2),
        "density_class_blend_rho0p5_topm3_k10": torch.rand(9),
        "density_query_blend_rho0p5_topm3_k20": torch.rand(2),
        "density_class_blend_rho0p5_topm3_k20": torch.rand(9),
        "class_to_clip_rank_blend_rho0p5_topm3": torch.ones(2, 9),
    }
    raw_to_col = {rid: idx for idx, rid in enumerate(raw_ids)}
    metrics, *_ = _evaluate_variant(
        variant=build_default_variants()[0],
        cache=cache,
        rows=rows,
        raw_to_col=raw_to_col,
        clip_ids=[1, 2],
        raw_ids=raw_ids,
        k_values=(1, 2, 3),
        primary_k=3,
        device=device,
        person_cols=[],
    )
    assert metrics["formal_gt_count"] == 3
    return {"status": "PASS", "registry": registry, "evaluator": metrics}
