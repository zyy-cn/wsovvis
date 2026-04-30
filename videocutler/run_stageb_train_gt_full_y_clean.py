#!/usr/bin/env python3
"""Clean GT-trajectory + full-Y_base mechanism training.

This standalone entry is intentionally isolated from the normal weak-label / VideoCutLER
training path.  It uses GT carrier trajectories and full clip-level official-base labels
only, and it evaluates three clean protocols:

  * baseline_full_y:     candidate_set(v) = Y_base(v)
  * static_residual:     candidate_set(v,t) = Y_base(v)∩K0 or Y_base(v)\K_{t-1}
  * soft_routing:        candidate_set(v) = Y_base(v), with soft row weighting

No Y-prime, VideoCutLER carrier, extra mining, or row-level GT target is used by training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.sinkhorn_assignment import SinkhornAssignmentConfig, capped_sinkhorn_assignment  # noqa: E402
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import Phase1MaterializationConfig, materialize_phase1_training_samples  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402


REPO_ASSET_LINK_NAMES = (
    "exports", "exports_gt", "carrier_bank", "carrier_bank_gt", "frame_bank",
    "text_bank", "gt_sidecar_bank", "weak_labels", "weights", "dataset", "eval",
)


def _safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _bootstrap_asset_links(target_root: Path, asset_root: Path) -> None:
    if not asset_root.is_dir():
        return
    target_root.mkdir(parents=True, exist_ok=True)
    for name in REPO_ASSET_LINK_NAMES:
        src = asset_root / name
        dst = target_root / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            try:
                _safe_link(src, dst)
            except Exception:
                pass


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return None


    for rec in obj.get("categories", []) or []:
        if not isinstance(rec, Mapping):
            continue
        raw_id = _as_int(rec.get("id"))
        if raw_id is None:
            continue
        name = rec.get("name") or rec.get("category_name") or rec.get("synset") or str(raw_id)
        out[int(raw_id)] = str(name)
    return out


def _class_name_map_from_annotation_json(annotation_json: Path) -> Dict[int, str]:
    """Build raw category id -> class name map from LV-VIS annotation JSON.

    Diagnostic-only helper for absorber logging. It must not affect training loss.
    """
    out: Dict[int, str] = {}
    try:
        obj = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    except Exception:
        return out
    for rec in obj.get("categories", []) or []:
        if not isinstance(rec, Mapping):
            continue
        raw_id = _as_int(rec.get("id"))
        if raw_id is None:
            continue
        name = rec.get("name") or rec.get("category_name") or rec.get("synset") or str(raw_id)
        out[int(raw_id)] = str(name)
    return out


def _class_name_map_from_text_records(records: Sequence[Mapping[str, Any]]) -> Dict[int, str]:
    """Build raw category id -> class name map from text prototype records.

    This is used only for absorber logging / diagnostics. It must not affect loss.
    """
    out: Dict[int, str] = {}
    for rec in records or []:
        if not isinstance(rec, Mapping):
            continue

        raw_id = None
        for key in ("raw_id", "raw_category_id", "category_id", "class_id", "id"):
            raw_id = _as_int(rec.get(key))
            if raw_id is not None:
                break
        if raw_id is None:
            continue

        name = ""
        for key in ("class_name", "category_name", "name", "synset", "label"):
            val = rec.get(key)
            if val is not None and str(val).strip():
                name = str(val).strip()
                break

        out[int(raw_id)] = name or str(raw_id)
    return out


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(np.asarray(list(vals), dtype=np.float32))) if vals else 0.0


def _normalize_np(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return (arr / denom).astype(np.float32, copy=False)


def _inverse_softplus(value: float) -> float:
    target = max(float(value), 1.0e-6)
    return float(math.log(math.expm1(target)))


def _compute_t_dis(theta_t: torch.nn.Parameter) -> torch.Tensor:
    return F.softplus(theta_t) + 1.0e-4


def _iter_progress(iterable: Iterable[Any], *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _maybe_list_of_ids(v: Any) -> Optional[List[int]]:
    if isinstance(v, list):
        out: List[int] = []
        ok = True
        for item in v:
            if isinstance(item, Mapping):
                val = item.get("raw_id", item.get("id", item.get("category_id")))
            else:
                val = item
            ii = _as_int(val)
            if ii is None:
                ok = False
                break
            out.append(int(ii))
        return out if ok else None
    return None


def _extract_split_ids(obj: Any, split_name: str) -> List[int]:
    keys = {
        "base": [
            "base", "base_ids", "base_raw_ids", "base_category_ids", "base_classes",
            "official_base", "base_raw_id_list", "base_categories",
        ],
        "novel": [
            "novel", "novel_ids", "novel_raw_ids", "novel_category_ids", "novel_classes",
            "official_novel", "novel_raw_id_list", "novel_categories",
        ],
    }[split_name]
    found: List[int] = []

    def walk(x: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(x, Mapping):
            for k in keys:
                if k in x:
                    ids = _maybe_list_of_ids(x[k])
                    if ids is not None:
                        found = ids
                        return
            for k, v in x.items():
                if str(k).lower() == split_name:
                    ids = _maybe_list_of_ids(v)
                    if ids is not None:
                        found = ids
                        return
            for v in x.values():
                walk(v)
                if found:
                    return
        elif isinstance(x, list):
            records = [e for e in x if isinstance(e, Mapping)]
            if records and any(str(r.get("split", "")).lower() == split_name for r in records):
                vals: List[int] = []
                for r in records:
                    if str(r.get("split", "")).lower() == split_name:
                        val = r.get("raw_id", r.get("id", r.get("category_id")))
                        ii = _as_int(val)
                        if ii is not None:
                            vals.append(int(ii))
                if vals:
                    found = vals
                    return
            for v in x:
                walk(v)
                if found:
                    return

    walk(obj)
    if not found:
        raise KeyError(f"could not extract {split_name} ids from split json")
    return sorted({int(x) for x in found})


def _load_base_ids(split_json: Path) -> Set[int]:
    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return set(_extract_split_ids(obj, "base"))


def _load_clip_y_base(annotation_json: Path, base_ids: Set[int]) -> Dict[int, Set[int]]:
    with annotation_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    anns = obj.get("annotations", []) if isinstance(obj, Mapping) else []
    out: Dict[int, Set[int]] = {}
    for ann in anns:
        if not isinstance(ann, Mapping):
            continue
        clip = _as_int(ann.get("video_id", ann.get("clip_id", ann.get("image_id"))))
        cat = _as_int(ann.get("category_id", ann.get("raw_id", ann.get("raw_category_id"))))
        if clip is None or cat is None:
            continue
        if int(cat) in base_ids:
            out.setdefault(int(clip), set()).add(int(cat))
    return out


def _parse_epoch_plan(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(max(0, int(float(part))))
    return out or [1]


@dataclass
class ResidualSchedule:
    epoch_plan: List[int]
    k_prev_by_round: Dict[int, Set[int]]
    k_by_round: Dict[int, Set[int]]
    c_by_round: Dict[int, Set[int]]
    class_to_round: Dict[int, int]
    class_to_certificate: Dict[int, str]

    def round_for_epoch(self, epoch_zero_based: int) -> int:
        acc = 0
        for rid, width in enumerate(self.epoch_plan):
            acc += int(width)
            if int(epoch_zero_based) < acc:
                return int(rid)
        return max(0, len(self.epoch_plan) - 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "epoch_plan": list(self.epoch_plan),
            "round_count": len(self.epoch_plan),
            "k_by_round_count": {str(k): len(v) for k, v in sorted(self.k_by_round.items())},
            "c_by_round_count": {str(k): len(v) for k, v in sorted(self.c_by_round.items())},
            "resolved_class_count": len(self.class_to_round),
        }


def _load_residual_schedule(csv_path: Path, *, variant: str, epoch_plan: str, base_ids: Set[int]) -> ResidualSchedule:
    class_to_round: Dict[int, int] = {}
    class_to_certificate: Dict[int, str] = {}
    c_by_round: Dict[int, Set[int]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant", "")) != str(variant):
                continue
            raw_id = _as_int(row.get("raw_id"))
            if raw_id is None or int(raw_id) not in base_ids:
                continue
            if not _truthy(row.get("resolved")):
                continue
            rr = _as_int(row.get("resolved_at_iteration"))
            if rr is None:
                continue
            class_to_round[int(raw_id)] = int(rr)
            class_to_certificate[int(raw_id)] = str(row.get("certificate_type", "unknown"))
            c_by_round[int(rr)].add(int(raw_id))
    if not c_by_round:
        raise RuntimeError(f"no resolved classes found in {csv_path} for variant={variant!r}")
    plan = _parse_epoch_plan(epoch_plan)
    max_round = max(max(c_by_round.keys()), len(plan) - 1)
    known: Set[int] = set()
    k_prev: Dict[int, Set[int]] = {}
    k_by: Dict[int, Set[int]] = {}
    for r in range(max_round + 1):
        if r == 0:
            k_prev[0] = set()
            known = set(c_by_round.get(0, set()))
            k_by[0] = set(known)
        else:
            k_prev[r] = set(known)
            known = set(known) | set(c_by_round.get(r, set()))
            k_by[r] = set(known)
    return ResidualSchedule(plan, k_prev, k_by, dict(c_by_round), class_to_round, class_to_certificate)


def _candidate_set_for_protocol(
    *,
    protocol: str,
    y_base: Set[int],
    schedule: Optional[ResidualSchedule],
    epoch_zero_based: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    proto = str(protocol)
    if proto in {"baseline_full_y", "soft_routing"}:
        return set(y_base), {"protocol_round_id": -1, "known_count": 0, "candidate_source": "Y_base"}
    if proto != "static_residual":
        raise ValueError(f"unsupported protocol: {protocol}")
    if schedule is None:
        raise ValueError("static_residual requires schedule_csv")
    round_id = schedule.round_for_epoch(epoch_zero_based)
    if round_id == 0:
        cand = set(y_base) & set(schedule.k_by_round.get(0, set()))
        known = set()
    else:
        known = set(schedule.k_prev_by_round.get(round_id, set()))
        cand = set(y_base) - known
    return cand, {
        "protocol_round_id": int(round_id),
        "known_count": int(len(known)),
        "candidate_source": "Y_base_intersect_K0" if round_id == 0 else "Y_base_minus_Kprev",
    }


def _group_by_clip(examples: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    by_clip: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_clip[int(ex["clip_id"])].append(ex)
    return [by_clip[k] for k in sorted(by_clip.keys())]


def _iter_microbatches(groups: Sequence[Sequence[Mapping[str, Any]]], *, max_groups_per_batch: int) -> List[List[int]]:
    n = max(1, int(max_groups_per_batch))
    return [list(range(i, min(i + n, len(groups)))) for i in range(0, len(groups), n)]


def _load_materialized_gt_examples(
    *,
    repo_root: Path,
    output_root: Path,
    asset_root: Path,
    dataset_name: str,
    annotation_json: Path,
    split_json: Path,
    smoke: bool,
    smoke_max_trajectories: int,
    subset_fraction: Optional[float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, Set[int]], Set[int], Dict[str, Any]]:
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(output_root, asset_root)
    base_ids = _load_base_ids(split_json)
    clip_y_base = _load_clip_y_base(annotation_json, base_ids)
    with _pushd(repo_root):
        materialized = materialize_phase1_training_samples(
            repo_root,
            Phase1MaterializationConfig(
                dataset_name=str(dataset_name),
                trajectory_source_branch="gt_upper_bound",
                smoke=bool(smoke),
                smoke_max_trajectories=int(smoke_max_trajectories),
                subset_fraction=subset_fraction,
                subset_seed=int(seed),
            ),
        )
    samples_raw = materialized.get("valid_samples") or materialized.get("samples") or []
    samples: List[Dict[str, Any]] = []
    sample_counters = Counter()
    for sample in samples_raw:
        if not bool(sample.get("sample_valid", False)):
            sample_counters["skip_sample_not_valid"] += 1
            continue
        clip = _as_int(sample.get("clip_id"))
        if clip is None:
            sample_counters["skip_no_clip_id"] += 1
            continue
        y_base = sorted(clip_y_base.get(int(clip), set()))
        if not y_base:
            sample_counters["skip_no_y_base"] += 1
            continue
        row = dict(sample)
        row["observed_raw_ids"] = [int(x) for x in y_base]
        row["clean_label_source"] = "full_Y_base_from_GT_annotations"
        samples.append(row)
    prepared = _prepare_prealign_examples(
        samples,
        output_root=output_root,
        dataset_name=str(dataset_name),
        trajectory_source_branch="gt_upper_bound",
    )
    examples = list(prepared.get("examples", []))
    materialization_summary = {
        "materialized_stats": materialized.get("stats", {}),
        "materialized_resolution": materialized.get("resolution", {}),
        "sample_counters": dict(sample_counters),
        "prepare_skipped_reason_histogram": dict(prepared.get("skipped_reason_histogram", {})),
        "sample_count_after_full_y_base_filter": int(len(samples)),
        "trainable_example_count": int(len(examples)),
    }
    return examples, clip_y_base, base_ids, materialization_summary


def _build_response_rows(
    *,
    stage_id: str,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    projector: Projector,
    text_vocab_tensor: torch.Tensor,
    text_vocab_ids: Sequence[int],
    raw_to_idx: Mapping[int, int],
    theta_t: torch.nn.Parameter,
    device: torch.device,
    y_base_by_clip: Mapping[int, Set[int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    projector.eval()
    with torch.no_grad():
        text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
        temperature = _compute_t_dis(theta_t)
        for group in groups:
            clip_id = int(group[0]["clip_id"])
            candidates = sorted(int(x) for x in y_base_by_clip.get(clip_id, set()) if int(x) in raw_to_idx)
            if not candidates:
                continue
            cand_idx = torch.tensor([int(raw_to_idx[int(x)]) for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj_all[cand_idx]
            Z = torch.stack([
                torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32)
                for ex in group
            ], dim=0)
            Z = F.normalize(Z, p=2.0, dim=-1)
            scores = torch.matmul(Z, cand_text.t()) / temperature
            probs = torch.softmax(scores, dim=1).detach().cpu().numpy().astype(np.float64)
            for q, ex in enumerate(group):
                rows.append({
                    "dataset_name": str(ex.get("dataset_name", "")),
                    "clip_id": int(clip_id),
                    "video_id": int(ex.get("video_id", clip_id)),
                    "trajectory_id": str(ex.get("trajectory_id", "")),
                    "candidate_ids_known": list(candidates),
                    "candidate_ids_extra": [],
                    "candidate_ids_null": [],
                    "candidate_scope_policy": {"policy": "GT_FULL_Y_BASE_CLEAN", "label_source": "Y_base"},
                    "candidate_demand_by_raw_id": {str(int(x)): 1.0 for x in candidates},
                    "candidate_kind_by_raw_id": {str(int(x)): 1 for x in candidates},
                    "unknown_disabled": True,
                    "training_semantics": "gt_full_y_clean_prealign",
                    "stage_id": str(stage_id),
                    "r_init": {},
                    "r_final": {str(int(raw_id)): float(prob) for raw_id, prob in zip(candidates, probs[q].tolist())},
                    "join_key": str(ex.get("trajectory_id", "")),
                })
    projector.train()
    return rows


def train_clean(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(output_root, asset_root)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    examples, clip_y_base, base_ids, materialization_summary = _load_materialized_gt_examples(
        repo_root=repo_root,
        output_root=output_root,
        asset_root=asset_root,
        dataset_name=str(args.dataset_name),
        annotation_json=Path(args.annotation_json),
        split_json=Path(args.split_json),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        seed=int(args.seed),
    )
    if not examples:
        raise RuntimeError("no trainable GT-fullY examples")

    schedule: Optional[ResidualSchedule] = None
    if str(args.protocol) == "static_residual" or str(args.soft_floor_scope) in {"residual", "resolved"}:
        if not args.schedule_csv:
            raise ValueError("schedule_csv is required for static_residual or residual soft floor scope")
        schedule = _load_residual_schedule(Path(args.schedule_csv), variant=str(args.residual_variant), epoch_plan=str(args.round_epoch_plan), base_ids=base_ids)

    if bool(args.enable_absorber_logging) and str(args.protocol) != "soft_routing":
        raise ValueError("--enable_absorber_logging is only supported for --protocol soft_routing")

    if str(args.protocol) == "static_residual" and args.epochs is None:
        epochs = int(sum(_parse_epoch_plan(str(args.round_epoch_plan))))
    else:
        epochs = int(args.epochs if args.epochs is not None else 15)

    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(output_root)
    class_name_by_raw = _class_name_map_from_text_records(_text_records if isinstance(_text_records, (list, tuple)) else (list(_text_records.values()) if isinstance(_text_records, dict) else []))
    ann_name_by_raw = _class_name_map_from_annotation_json(Path(args.annotation_json))
    if ann_name_by_raw:
        class_name_by_raw.update(ann_name_by_raw)
    raw_to_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    projector_cfg = ProjectorConfig()
    projector = Projector(projector_cfg).to(device)
    projector.train()
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    sinkhorn_cfg = SinkhornAssignmentConfig(tau=float(args.sinkhorn_tau), iters=int(args.sinkhorn_iters), row_cap_scale=float(args.sinkhorn_row_cap_scale))

    groups = _group_by_clip(examples)
    max_groups_per_batch = max(1, int(args.max_groups_per_batch))
    runtime_metrics_path = output_root / "train" / "prealign" / "runtime_metrics.jsonl"
    if runtime_metrics_path.exists():
        runtime_metrics_path.unlink()
    protocol_metrics_path = output_root / "train" / "prealign" / "gt_full_y_clean_protocol_metrics.jsonl"
    if protocol_metrics_path.exists():
        protocol_metrics_path.unlink()

    global_losses: List[float] = []
    batch_losses: List[float] = []
    global_step = 0
    train_start = datetime.now(timezone.utc).isoformat()

    absorber_logging_enabled = bool(args.enable_absorber_logging)
    absorber_decay = float(args.absorber_ema_decay)
    absorber_ema: Dict[int, Dict[str, float]] = {}
    absorber_ema_rows: List[Dict[str, Any]] = []
    top_absorber_rows: List[Dict[str, Any]] = []
    top_absorbers_k = max(1, int(args.top_absorbers_k))

    for epoch_idx in _iter_progress(range(epochs), enabled=bool(args.show_progress), desc=f"{args.protocol} epochs", leave=True):
        rng = random.Random(int(args.seed) + int(epoch_idx))
        shuffled = list(groups)
        rng.shuffle(shuffled)
        batches = _iter_microbatches(shuffled, max_groups_per_batch=max_groups_per_batch)
        epoch_losses: List[float] = []
        epoch_batch_losses: List[float] = []
        epoch_stats = Counter()
        epoch_float_stats: Dict[str, List[float]] = defaultdict(list)
        round_hist = Counter()
        absorber_epoch_support: Dict[int, float] = defaultdict(float)
        absorber_epoch_mass: Dict[int, float] = defaultdict(float)
        absorber_epoch_top1: Dict[int, float] = defaultdict(float)

        for mb_idx, batch_indices in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
            temperature = _compute_t_dis(theta_t)
            batch_loss_accum: Optional[torch.Tensor] = None
            batch_effective_groups = 0
            batch_effective_traj = 0
            batch_float_stats: Dict[str, List[float]] = defaultdict(list)
            batch_stats = Counter()
            for group_idx in batch_indices:
                group = shuffled[int(group_idx)]
                clip_id = int(group[0]["clip_id"])
                y_base = set(clip_y_base.get(clip_id, set()))
                candidates, cand_meta = _candidate_set_for_protocol(protocol=str(args.protocol), y_base=y_base, schedule=schedule, epoch_zero_based=int(epoch_idx))
                candidates = sorted(int(x) for x in candidates if int(x) in raw_to_idx)
                round_hist[str(cand_meta.get("protocol_round_id", -1))] += 1
                batch_stats["groups_seen"] += 1
                if not candidates:
                    batch_stats["groups_empty_candidate"] += 1
                    continue
                cand_idx = torch.tensor([int(raw_to_idx[int(x)]) for x in candidates], device=device, dtype=torch.long)
                cand_text = text_proj_all[cand_idx]
                Z = torch.stack([
                    torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32)
                    for ex in group
                ], dim=0)
                Z = F.normalize(Z, p=2.0, dim=-1)
                Q = int(Z.shape[0])
                M = int(cand_text.shape[0])
                scores = torch.matmul(Z, cand_text.t()) / temperature
                q_mask = torch.ones((1, Q), device=device, dtype=torch.bool)
                c_mask = torch.ones((1, M), device=device, dtype=torch.bool)
                demand = torch.ones((1, M), device=device, dtype=torch.float32)
                P = capped_sinkhorn_assignment(scores.unsqueeze(0), q_mask, c_mask, demand, config=sinkhorn_cfg)[0, :Q, :M]
                if bool(args.assignment_stopgrad):
                    P_for_loss = P.detach()
                else:
                    P_for_loss = P
                log_probs = torch.log_softmax(scores, dim=1)
                per_row_loss = -(P_for_loss * log_probs).sum(dim=1)
                row_weight = torch.ones((Q,), device=device, dtype=torch.float32)
                soft_row = {
                    "soft_routing_enabled": False,
                    "residual_weight_mean": 1.0,
                    "residual_weight_sum": float(Q),
                    "residual_weight_min": 1.0,
                    "residual_weight_p10": 1.0,
                    "residual_weight_p50": 1.0,
                    "residual_weight_p90": 1.0,
                    "explained_mass_mean": 0.0,
                    "explicit_hub_top1_share": 0.0,
                }
                if str(args.protocol) == "soft_routing":
                    probs = torch.softmax(scores.detach(), dim=1)
                    conf, top_idx = torch.max(probs, dim=1)
                    explained = torch.sigmoid(float(args.soft_gamma) * (conf - float(args.soft_tau)))
                    row_weight = 1.0 - explained
                    cand_raw_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
                    top_raw = cand_raw_tensor[top_idx]

                    explicit_hub_ids = set() if bool(args.disable_explicit_hub_cap) else _parse_optional_id_set(args.hub_raw_ids)
                    if float(args.hub_cap) < 1.0 and explicit_hub_ids:
                        hub_mask = torch.zeros_like(row_weight, dtype=torch.bool)
                        for hid in sorted(explicit_hub_ids):
                            hub_mask |= top_raw == int(hid)
                        row_weight = torch.where(hub_mask, torch.clamp(row_weight, min=float(1.0 - float(args.hub_cap))), row_weight)

                    if float(args.rare_floor) > 0.0:
                        floor_mask = torch.zeros_like(row_weight, dtype=torch.bool)
                        if str(args.soft_floor_scope) == "all":
                            floor_mask = torch.ones_like(row_weight, dtype=torch.bool)
                        elif str(args.soft_floor_scope) in {"residual", "resolved"} and schedule is not None:
                            residual_ids = {rid for rid, rr in schedule.class_to_round.items() if int(rr) >= 1}
                            for rid in residual_ids:
                                floor_mask |= top_raw == int(rid)
                        row_weight = torch.where(floor_mask, torch.clamp(row_weight, min=float(args.rare_floor)), row_weight)

                    if float(args.min_row_weight) > 0.0:
                        row_weight = torch.clamp(row_weight, min=float(args.min_row_weight))
                    row_weight = torch.clamp(row_weight, min=0.0, max=1.0)

                    if absorber_logging_enabled:
                        probs_cpu = probs.detach().cpu().numpy().astype(np.float64)
                        top_raw_list = top_raw.detach().cpu().numpy().astype(np.int64).tolist()
                        top_counter = Counter(int(x) for x in top_raw_list)
                        for j, raw_id in enumerate(candidates):
                            rid = int(raw_id)
                            absorber_epoch_support[rid] += 1.0
                            absorber_epoch_mass[rid] += float(probs_cpu[:, j].sum())
                            absorber_epoch_top1[rid] += float(top_counter.get(rid, 0))

                    denom = torch.clamp(row_weight.sum(), min=1.0)
                    sample_loss = (per_row_loss * row_weight).sum() / denom
                    row_weight_np = row_weight.detach().cpu().numpy().astype(np.float64)
                    explicit_hub_share = 0.0
                    if explicit_hub_ids:
                        top_raw_list_for_hub = top_raw.detach().cpu().numpy().astype(np.int64).tolist()
                        explicit_hub_share = float(sum(1 for x in top_raw_list_for_hub if int(x) in explicit_hub_ids) / max(Q, 1))
                    soft_row = {
                        "soft_routing_enabled": True,
                        "explicit_hub_cap_disabled": bool(args.disable_explicit_hub_cap),
                        "residual_weight_mean": float(row_weight_np.mean()) if row_weight_np.size else 0.0,
                        "residual_weight_sum": float(row_weight_np.sum()) if row_weight_np.size else 0.0,
                        "residual_weight_min": float(row_weight_np.min()) if row_weight_np.size else 0.0,
                        "residual_weight_p10": float(np.percentile(row_weight_np, 10)) if row_weight_np.size else 0.0,
                        "residual_weight_p50": float(np.percentile(row_weight_np, 50)) if row_weight_np.size else 0.0,
                        "residual_weight_p90": float(np.percentile(row_weight_np, 90)) if row_weight_np.size else 0.0,
                        "explained_mass_mean": float((1.0 - row_weight_np).mean()) if row_weight_np.size else 0.0,
                        "explicit_hub_top1_share": explicit_hub_share,
                    }
                else:
                    sample_loss = per_row_loss.mean()
                batch_loss_accum = sample_loss if batch_loss_accum is None else batch_loss_accum + sample_loss
                batch_effective_groups += 1
                batch_effective_traj += Q
                val = float(sample_loss.detach().cpu().item())
                global_losses.append(val)
                epoch_losses.append(val)
                batch_float_stats["candidate_size"].append(float(M))
                batch_float_stats["trajectory_count"].append(float(Q))
                for k, v in soft_row.items():
                    if isinstance(v, (int, float)):
                        batch_float_stats[str(k)].append(float(v))
            if batch_loss_accum is None or batch_effective_groups <= 0:
                continue
            loss = batch_loss_accum / float(batch_effective_groups)
            loss.backward()
            optimizer.step()
            global_step += 1
            bval = float(loss.detach().cpu().item())
            batch_losses.append(bval)
            epoch_batch_losses.append(bval)
            batch_summary = {
                "row_type": "microbatch",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "prealign",
                "protocol": str(args.protocol),
                "epoch": int(epoch_idx) + 1,
                "microbatch_idx": int(mb_idx),
                "microbatch_total": int(len(batches)),
                "loss": bval,
                "optimization_loss": bval,
                "effective_group_count": int(batch_effective_groups),
                "effective_trajectory_count": int(batch_effective_traj),
                "candidate_size_mean": _mean(batch_float_stats.get("candidate_size", [])),
                "trajectory_count_mean": _mean(batch_float_stats.get("trajectory_count", [])),
                "empty_group_rate": float(batch_stats.get("groups_empty_candidate", 0) / max(batch_stats.get("groups_seen", 1), 1)),
                "residual_weight_mean": _mean(batch_float_stats.get("residual_weight_mean", [])),
                "residual_weight_sum_mean": _mean(batch_float_stats.get("residual_weight_sum", [])),
                "residual_weight_min_mean": _mean(batch_float_stats.get("residual_weight_min", [])),
                "residual_weight_p10_mean": _mean(batch_float_stats.get("residual_weight_p10", [])),
                "residual_weight_p50_mean": _mean(batch_float_stats.get("residual_weight_p50", [])),
                "residual_weight_p90_mean": _mean(batch_float_stats.get("residual_weight_p90", [])),
                "explained_mass_mean": _mean(batch_float_stats.get("explained_mass_mean", [])),
                "explicit_hub_top1_share": _mean(batch_float_stats.get("explicit_hub_top1_share", [])),
            }
            _append_jsonl(runtime_metrics_path, batch_summary)
            _append_jsonl(protocol_metrics_path, batch_summary)
            for k, v in batch_summary.items():
                if isinstance(v, (int, float)):
                    epoch_float_stats[k].append(float(v))
            epoch_stats.update(batch_stats)
        epoch_summary = {
            "row_type": "epoch_summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "prealign",
            "protocol": str(args.protocol),
            "epoch": int(epoch_idx) + 1,
            "microbatch_count": int(len(epoch_batch_losses)),
            "loss_mean": _mean(epoch_losses),
            "loss_last": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "optimization_loss_mean": _mean(epoch_batch_losses),
            "optimization_loss_last": float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,
            "effective_group_count_epoch": float(sum(epoch_float_stats.get("effective_group_count", []))),
            "effective_trajectory_count_epoch": float(sum(epoch_float_stats.get("effective_trajectory_count", []))),
            "candidate_size_mean_epoch": _mean(epoch_float_stats.get("candidate_size_mean", [])),
            "empty_group_rate_epoch": _mean(epoch_float_stats.get("empty_group_rate", [])),
            "residual_weight_mean_epoch": _mean(epoch_float_stats.get("residual_weight_mean", [])),
            "residual_weight_sum_mean_epoch": _mean(epoch_float_stats.get("residual_weight_sum_mean", [])),
            "residual_weight_min_mean_epoch": _mean(epoch_float_stats.get("residual_weight_min_mean", [])),
            "residual_weight_p10_mean_epoch": _mean(epoch_float_stats.get("residual_weight_p10_mean", [])),
            "residual_weight_p50_mean_epoch": _mean(epoch_float_stats.get("residual_weight_p50_mean", [])),
            "residual_weight_p90_mean_epoch": _mean(epoch_float_stats.get("residual_weight_p90_mean", [])),
            "explained_mass_mean_epoch": _mean(epoch_float_stats.get("explained_mass_mean", [])),
            "explicit_hub_top1_share_epoch": _mean(epoch_float_stats.get("explicit_hub_top1_share", [])),
            "absorber_logging_enabled": bool(absorber_logging_enabled),
            "round_hist_epoch": dict(round_hist),
        }
        _append_jsonl(runtime_metrics_path, epoch_summary)
        _append_jsonl(protocol_metrics_path, epoch_summary)

        if absorber_logging_enabled:
            all_absorber_ids = set(absorber_ema.keys()) | set(absorber_epoch_support.keys()) | set(absorber_epoch_mass.keys()) | set(absorber_epoch_top1.keys())
            epoch_rows: List[Dict[str, Any]] = []
            for rid in sorted(int(x) for x in all_absorber_ids):
                prev = absorber_ema.get(int(rid), {"label_support_ema": 0.0, "responsibility_mass_ema": 0.0, "top1_count_ema": 0.0})
                support_epoch = float(absorber_epoch_support.get(int(rid), 0.0))
                mass_epoch = float(absorber_epoch_mass.get(int(rid), 0.0))
                top1_epoch = float(absorber_epoch_top1.get(int(rid), 0.0))
                support_ema = absorber_decay * float(prev.get("label_support_ema", 0.0)) + (1.0 - absorber_decay) * support_epoch
                mass_ema = absorber_decay * float(prev.get("responsibility_mass_ema", 0.0)) + (1.0 - absorber_decay) * mass_epoch
                top1_ema = absorber_decay * float(prev.get("top1_count_ema", 0.0)) + (1.0 - absorber_decay) * top1_epoch
                absorber_ema[int(rid)] = {
                    "label_support_ema": float(support_ema),
                    "responsibility_mass_ema": float(mass_ema),
                    "top1_count_ema": float(top1_ema),
                }
                absorber_score = float(mass_ema / max(support_ema, 1.0e-12)) if support_ema > 0 else 0.0
                top1_absorb_score = float(top1_ema / max(support_ema, 1.0e-12)) if support_ema > 0 else 0.0
                epoch_rows.append({
                    "epoch": int(epoch_idx) + 1,
                    "raw_id": int(rid),
                    "class_name": class_name_by_raw.get(int(rid), ""),
                    "label_support_epoch": support_epoch,
                    "responsibility_mass_epoch": mass_epoch,
                    "top1_count_epoch": top1_epoch,
                    "label_support_ema": float(support_ema),
                    "responsibility_mass_ema": float(mass_ema),
                    "top1_count_ema": float(top1_ema),
                    "absorber_score": absorber_score,
                    "top1_absorb_score": top1_absorb_score,
                })
            epoch_rows_sorted = sorted(epoch_rows, key=lambda r: (float(r.get("absorber_score", 0.0)), float(r.get("top1_absorb_score", 0.0))), reverse=True)
            for rank, row in enumerate(epoch_rows_sorted[:top_absorbers_k], start=1):
                out = dict(row)
                out["rank_by_absorber_score"] = int(rank)
                top_absorber_rows.append(out)
            absorber_ema_rows.extend(epoch_rows)

        if bool(args.print_epoch_summary):
            print(json.dumps({k: epoch_summary[k] for k in ["epoch", "protocol", "loss_mean", "effective_trajectory_count_epoch", "candidate_size_mean_epoch", "empty_group_rate_epoch"]}, ensure_ascii=False))

    train_dir = output_root / "train" / "prealign"
    ckpt_dir = train_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "prealign_last.pth"
    torch.save({
        "stage_id": "prealign",
        "epoch": int(epochs),
        "text_projector_state_dict": projector.state_dict(),
        "text_projector_config": {
            "input_dim": int(projector_cfg.input_dim),
            "hidden_dim": int(projector_cfg.hidden_dim),
            "output_dim": int(projector_cfg.output_dim),
            "dropout": float(projector_cfg.dropout),
            "use_layernorm": bool(projector_cfg.use_layernorm),
        },
        "theta_T": float(theta_t.detach().cpu().item()),
        "seed": int(args.seed),
        "protocol": str(args.protocol),
        "pipeline": "gt_full_y_clean",
        "label_source": "full_Y_base",
        "trajectory_source_branch": "gt_upper_bound",
        "global_step": int(global_step),
    }, ckpt_path)

    response_rows = _build_response_rows(
        stage_id="prealign",
        groups=groups,
        projector=projector,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        theta_t=theta_t,
        device=device,
        y_base_by_clip=clip_y_base,
    )
    _write_jsonl(train_dir / "responsibility_records.jsonl", response_rows)

    absorber_outputs: Dict[str, str] = {}
    if absorber_logging_enabled:
        absorber_fields = [
            "epoch", "raw_id", "class_name",
            "label_support_epoch", "responsibility_mass_epoch", "top1_count_epoch",
            "label_support_ema", "responsibility_mass_ema", "top1_count_ema",
            "absorber_score", "top1_absorb_score",
        ]
        top_fields = absorber_fields + ["rank_by_absorber_score"]
        final_rows = []
        for rid, vals in sorted(absorber_ema.items()):
            support = float(vals.get("label_support_ema", 0.0))
            mass = float(vals.get("responsibility_mass_ema", 0.0))
            top1 = float(vals.get("top1_count_ema", 0.0))
            final_rows.append({
                "epoch": int(epochs),
                "raw_id": int(rid),
                "class_name": class_name_by_raw.get(int(rid), ""),
                "label_support_epoch": "",
                "responsibility_mass_epoch": "",
                "top1_count_epoch": "",
                "label_support_ema": support,
                "responsibility_mass_ema": mass,
                "top1_count_ema": top1,
                "absorber_score": float(mass / max(support, 1.0e-12)) if support > 0 else 0.0,
                "top1_absorb_score": float(top1 / max(support, 1.0e-12)) if support > 0 else 0.0,
            })
        final_rows = sorted(final_rows, key=lambda r: (float(r.get("absorber_score", 0.0)), float(r.get("top1_absorb_score", 0.0))), reverse=True)
        for rank, row in enumerate(final_rows, start=1):
            row["rank_by_absorber_score"] = int(rank)
        _write_csv_rows(train_dir / "absorber_ema_by_epoch.csv", absorber_ema_rows, absorber_fields)
        _write_csv_rows(train_dir / "top_absorbers_by_epoch.csv", top_absorber_rows, top_fields)
        _write_csv_rows(train_dir / "final_absorber_scores.csv", final_rows, top_fields)
        absorber_outputs = {
            "absorber_ema_by_epoch": str((Path("train") / "prealign" / "absorber_ema_by_epoch.csv").as_posix()),
            "top_absorbers_by_epoch": str((Path("train") / "prealign" / "top_absorbers_by_epoch.csv").as_posix()),
            "final_absorber_scores": str((Path("train") / "prealign" / "final_absorber_scores.csv").as_posix()),
        }

    stage_summary = {
        "stage_id": "prealign",
        "pipeline": "gt_full_y_clean",
        "protocol": str(args.protocol),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "full_Y_base",
        "epochs": int(epochs),
        "loss_mean": _mean(global_losses),
        "loss_last": float(global_losses[-1]) if global_losses else 0.0,
        "optimization_loss_mean": _mean(batch_losses),
        "optimization_loss_last": float(batch_losses[-1]) if batch_losses else 0.0,
        "global_step": int(global_step),
        "trainable_example_count": int(len(examples)),
        "clip_group_count": int(len(groups)),
        "checkpoint_last": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "runtime_metrics": str((Path("train") / "prealign" / "runtime_metrics.jsonl").as_posix()),
        "protocol_metrics": str((Path("train") / "prealign" / "gt_full_y_clean_protocol_metrics.jsonl").as_posix()),
        "materialization_summary": materialization_summary,
        "schedule_summary": schedule.summary() if schedule is not None else None,
        "soft_routing_config": {
            "tau": float(args.soft_tau),
            "gamma": float(args.soft_gamma),
            "rare_floor": float(args.rare_floor),
            "hub_cap": float(args.hub_cap),
            "hub_raw_ids": str(args.hub_raw_ids),
            "disable_explicit_hub_cap": bool(args.disable_explicit_hub_cap),
            "min_row_weight": float(args.min_row_weight),
            "soft_floor_scope": str(args.soft_floor_scope),
            "enable_absorber_logging": bool(args.enable_absorber_logging),
            "absorber_ema_decay": float(args.absorber_ema_decay),
            "top_absorbers_k": int(args.top_absorbers_k),
            "absorber_outputs": absorber_outputs,
        } if str(args.protocol) == "soft_routing" else None,
    }
    _write_json(train_dir / "stage_summary.json", stage_summary)
    _write_json(train_dir / "train_state.json", {
        "stage_id": "prealign",
        "epoch": int(epochs),
        "checkpoint_last": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "checkpoint_selected": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "selected_for_infer": True,
        "selected_for_infer_authority": "gt_full_y_clean_protocol",
        "pipeline": "gt_full_y_clean",
        "protocol": str(args.protocol),
        "global_step": int(global_step),
    })
    pipeline_summary = {
        "status": "PASS",
        "exp_name": str(args.exp_name),
        "pipeline": "gt_full_y_clean",
        "protocol": str(args.protocol),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "full_Y_base",
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "output_root": str(output_root),
        "train_started_at": train_start,
        "train_finished_at": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint_path": str(ckpt_path),
        "stages": {"prealign": stage_summary},
    }
    _write_json(output_root / "train" / "pipeline_train_summary.json", pipeline_summary)
    _write_json(output_root / "GT_FULL_Y_CLEAN_MECHANISM_TRAIN_TAKEOVER.json", pipeline_summary)
    print(output_root / "train" / "pipeline_train_summary.json")
    return pipeline_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean GT trajectory + full Y_base mechanism trainer.")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--schedule_csv", default="")
    p.add_argument("--residual_variant", default="person_aware")
    p.add_argument("--protocol", required=True, choices=("baseline_full_y", "static_residual", "soft_routing"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--round_epoch_plan", default="5,5,3,2")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--sinkhorn_tau", type=float, default=0.15)
    p.add_argument("--sinkhorn_iters", type=int, default=5)
    p.add_argument("--sinkhorn_row_cap_scale", type=float, default=2.0)
    p.add_argument("--assignment_stopgrad", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--max_groups_per_batch", type=int, default=256)
    p.add_argument("--show_progress", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--print_epoch_summary", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--soft_tau", type=float, default=0.8)
    p.add_argument("--soft_gamma", type=float, default=10.0)
    p.add_argument("--rare_floor", type=float, default=0.25)
    p.add_argument("--hub_cap", type=float, default=0.75)
    p.add_argument("--hub_raw_ids", default="773")
    p.add_argument("--disable_explicit_hub_cap", action="store_true", help="Ignore hub_raw_ids/hub_cap and use no hand-specified hub prior.")
    p.add_argument("--min_row_weight", type=float, default=0.0, help="Global minimum soft-routing row weight; train-usable and does not use GT counts.")
    p.add_argument("--enable_absorber_logging", action="store_true", help="Log observable class absorber EMA statistics without affecting loss.")
    p.add_argument("--absorber_ema_decay", type=float, default=0.95)
    p.add_argument("--top_absorbers_k", type=int, default=50)
    p.add_argument("--soft_floor_scope", default="residual", choices=("none", "all", "residual", "resolved"))
    return p.parse_args()


def main() -> int:
    train_clean(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
