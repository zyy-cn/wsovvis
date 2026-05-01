#!/usr/bin/env python3
"""Read-only counterfactual/global-reference direct evidence audit for GT-fullY clean runs.

This script does not train, does not mutate checkpoints, and does not run mAP.
It evaluates whether the clip-local confidence used by soft_e2e_nohub is more or less
predictive of GT attribution correctness than a fixed-vocabulary global-reference signal.

Phase-1 design:
- local view: softmax over full Y_base(v), matching soft_e2e_nohub's row gate context.
- global view: score the same trajectory against a fixed reference vocabulary.
- direct stability: local top1 is considered more direct/less context-confounded when it
  remains highly ranked under the global reference view.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples  # noqa: E402
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import Phase1MaterializationConfig, materialize_phase1_training_samples  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402

REPO_ASSET_LINK_NAMES = (
    "exports", "exports_gt", "carrier_bank", "carrier_bank_gt", "frame_bank", "text_bank",
    "gt_sidecar_bank", "weak_labels", "weights", "dataset", "eval",
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


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return repr(v)
    return v


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(str(k)); seen.add(k)
        fields = keys
    if not fields:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _maybe_list_of_ids(v: Any) -> Optional[List[int]]:
    if isinstance(v, list):
        vals: List[int] = []
        for item in v:
            val = item.get("raw_id", item.get("id", item.get("category_id"))) if isinstance(item, Mapping) else item
            ii = _as_int(val)
            if ii is None:
                return None
            vals.append(int(ii))
        return vals
    return None


def _extract_split_ids(obj: Any, split_name: str) -> List[int]:
    keys = {
        "base": ["base", "base_ids", "base_raw_ids", "base_category_ids", "base_classes", "official_base", "base_raw_id_list", "base_categories"],
        "novel": ["novel", "novel_ids", "novel_raw_ids", "novel_category_ids", "novel_classes", "official_novel", "novel_raw_id_list", "novel_categories"],
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
                        found = ids; return
            for k, v in x.items():
                if str(k).lower() == split_name:
                    ids = _maybe_list_of_ids(v)
                    if ids is not None:
                        found = ids; return
            for v in x.values():
                walk(v)
                if found:
                    return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if found:
                    return

    walk(obj)
    if not found:
        raise KeyError(f"could not extract {split_name} ids")
    return sorted({int(x) for x in found})


def _load_base_ids(split_json: Path) -> Set[int]:
    with split_json.open("r", encoding="utf-8") as f:
        return set(_extract_split_ids(json.load(f), "base"))


def _load_clip_y_base(annotation_json: Path, base_ids: Set[int]) -> Dict[int, Set[int]]:
    with annotation_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, Set[int]] = {}
    for ann in obj.get("annotations", []):
        if not isinstance(ann, Mapping):
            continue
        clip = _as_int(ann.get("video_id", ann.get("clip_id", ann.get("image_id"))))
        cat = _as_int(ann.get("category_id", ann.get("raw_id", ann.get("raw_category_id"))))
        if clip is not None and cat is not None and int(cat) in base_ids:
            out.setdefault(int(clip), set()).add(int(cat))
    return out


def _load_schedule(csv_path: Path, *, variant: str, base_ids: Set[int]) -> Tuple[Dict[int, int], Dict[int, str]]:
    class_to_round: Dict[int, int] = {}
    class_to_cert: Dict[int, str] = {}
    if not csv_path or not csv_path.is_file():
        return class_to_round, class_to_cert
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "variant" in row and str(row.get("variant", "")) and str(row.get("variant", "")) != str(variant):
                continue
            rid = _as_int(row.get("raw_id"))
            rr = _as_int(row.get("resolved_at_iteration") or row.get("resolved_round"))
            resolved = row.get("resolved", "true")
            if rid is None or int(rid) not in base_ids or rr is None or not _truthy(resolved):
                continue
            class_to_round[int(rid)] = int(rr)
            class_to_cert[int(rid)] = str(row.get("certificate_type", "unknown"))
    return class_to_round, class_to_cert


def _load_class_names(annotation_json: Path) -> Dict[int, str]:
    try:
        obj = json.loads(annotation_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for rec in obj.get("categories", []) or []:
        if not isinstance(rec, Mapping):
            continue
        rid = _as_int(rec.get("id") or rec.get("raw_id") or rec.get("category_id"))
        if rid is None:
            continue
        out[int(rid)] = str(rec.get("name") or rec.get("category_name") or rec.get("synset") or rid)
    return out


def _load_weak_vocab(repo_root: Path, materialized_meta: Mapping[str, Any]) -> Set[int]:
    # Preferred: materialization stats often records weak_label_vocab. Fall back to weak_labels_train.json.
    stats = materialized_meta.get("materialized_stats", {}) if isinstance(materialized_meta, Mapping) else {}
    for root in [stats, materialized_meta]:
        if isinstance(root, Mapping):
            wl = root.get("weak_label_vocab", {})
            if isinstance(wl, Mapping) and isinstance(wl.get("weak_vocab_raw_ids"), list):
                return {int(x) for x in wl.get("weak_vocab_raw_ids", []) if _as_int(x) is not None}
    path = repo_root / "weak_labels" / "weak_labels_train.json"
    if not path.exists():
        return set()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    vals: Set[int] = set()

    def walk(x: Any) -> None:
        if isinstance(x, Mapping):
            for k in ("raw_id", "category_id", "class_id"):
                ii = _as_int(x.get(k))
                if ii is not None:
                    vals.add(int(ii))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return vals


def _normalize_np(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    return arr if denom <= 1e-12 else (arr / denom).astype(np.float32, copy=False)


def _extract_gt_raw_id(sample: Mapping[str, Any]) -> Optional[int]:
    candidate_roots: List[Any] = [sample]
    for k in ("trajectory_record", "carrier_record", "gt_record", "annotation"):
        if isinstance(sample.get(k), Mapping):
            candidate_roots.append(sample[k])
    keys = (
        "matched_gt_raw_id_canonical", "matched_gt_raw_id", "best_gt_raw_id",
        "gt_raw_id", "raw_id", "category_id", "raw_category_id", "class_raw_id",
    )
    for root in candidate_roots:
        if not isinstance(root, Mapping):
            continue
        for k in keys:
            ii = _as_int(root.get(k))
            if ii is not None:
                return int(ii)
    return None


def _load_identity_binding(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            tid = str(row.get("trajectory_id", row.get("gt_trajectory_id", row.get("join_key", ""))))
            rid = _extract_gt_raw_id(row)
            if tid and rid is not None:
                out[tid] = int(rid)
    return out


def _load_examples_with_gt(args: argparse.Namespace, output_root_for_assets: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Set[int]], Set[int], Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(output_root_for_assets, asset_root)
    base_ids = _load_base_ids(Path(args.split_json))
    clip_y_base = _load_clip_y_base(Path(args.annotation_json), base_ids)
    with _pushd(repo_root):
        materialized = materialize_phase1_training_samples(
            repo_root,
            Phase1MaterializationConfig(
                dataset_name=str(args.dataset_name), trajectory_source_branch="gt_upper_bound",
                smoke=bool(args.smoke), smoke_max_trajectories=int(args.smoke_max_trajectories),
                subset_fraction=args.subset_fraction, subset_seed=int(args.seed),
            ),
        )
    raw_samples = materialized.get("valid_samples") or materialized.get("samples") or []
    id_binding = _load_identity_binding(Path(args.asset_root) / "carrier_bank_gt" / str(args.dataset_name) / "gt_carrier_identity_binding.jsonl")
    samples: List[Dict[str, Any]] = []
    gt_by_tid: Dict[str, int] = {}
    counters = Counter()
    for s in raw_samples:
        if not bool(s.get("sample_valid", False)):
            counters["skip_invalid"] += 1; continue
        clip = _as_int(s.get("clip_id"))
        if clip is None:
            counters["skip_no_clip"] += 1; continue
        yb = sorted(clip_y_base.get(int(clip), set()))
        if not yb:
            counters["skip_no_y_base"] += 1; continue
        row = dict(s); row["observed_raw_ids"] = yb
        tid = str(row.get("trajectory_id", ""))
        rid = _extract_gt_raw_id(row)
        if rid is None and tid in id_binding:
            rid = int(id_binding[tid])
        if rid is not None:
            gt_by_tid[tid] = int(rid)
        samples.append(row)
    prepared = _prepare_prealign_examples(samples, output_root=output_root_for_assets, dataset_name=str(args.dataset_name), trajectory_source_branch="gt_upper_bound")
    examples = list(prepared.get("examples", []))
    for ex in examples:
        tid = str(ex.get("trajectory_id", ""))
        ex["matched_gt_raw_id"] = gt_by_tid.get(tid)
    meta = {
        "materialized_stats": materialized.get("stats", {}),
        "sample_counters": dict(counters),
        "prepare_skipped": dict(prepared.get("skipped_reason_histogram", {})),
        "identity_binding_count": len(id_binding),
    }
    return examples, clip_y_base, base_ids, meta


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Projector, torch.Tensor, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    cfg_raw = ckpt.get("text_projector_config", {}) if isinstance(ckpt, Mapping) else {}
    cfg = ProjectorConfig(
        input_dim=int(cfg_raw.get("input_dim", 512)), hidden_dim=int(cfg_raw.get("hidden_dim", 1024)),
        output_dim=int(cfg_raw.get("output_dim", 768)), dropout=float(cfg_raw.get("dropout", 0.0)),
        use_layernorm=bool(cfg_raw.get("use_layernorm", True)),
    )
    projector = Projector(cfg).to(device)
    projector.load_state_dict(ckpt.get("text_projector_state_dict", ckpt.get("state_dict", {})), strict=False)
    projector.eval()
    theta_raw = float(ckpt.get("theta_T", 0.0))
    theta_t = torch.tensor(theta_raw, device=device, dtype=torch.float32)
    temperature = F.softplus(theta_t) + 1.0e-4
    return projector, temperature, dict(ckpt)


def _entropy_norm(probs: torch.Tensor) -> torch.Tensor:
    n = probs.shape[-1]
    if n <= 1:
        return torch.zeros(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    ent = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
    return ent / math.log(float(n))


def _rank_score(rank: int, denom_count: int) -> float:
    denom = max(int(denom_count) - 1, 1)
    return float(1.0 - min(max(rank - 1, 0), denom) / denom)


def _auc_score(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))]
    if len(pairs) < 2:
        return None
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks for ties, ascending; AUC = (rank_sum_pos - n_pos(n_pos+1)/2) / (n_pos*n_neg)
    pairs_sorted = sorted((s, y, i) for i, (s, y) in enumerate(pairs))
    ranks = [0.0] * len(pairs_sorted)
    i = 0
    while i < len(pairs_sorted):
        j = i + 1
        while j < len(pairs_sorted) and pairs_sorted[j][0] == pairs_sorted[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rank_sum_pos = sum(r for r, (_, y, _) in zip(ranks, pairs_sorted) if y == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mx = float(np.mean(xs)); my = float(np.mean(ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def _rankdata(vals: Sequence[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(vals))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
            ranks[idx] = avg
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _quantile(vals: Sequence[float], q: float, default: float = 0.0) -> float:
    arr = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not arr:
        return default
    idx = int(round((len(arr) - 1) * float(q)))
    return arr[min(max(idx, 0), len(arr) - 1)]


class Stats:
    def __init__(self) -> None:
        self.n = 0
        self.sums: Dict[str, float] = defaultdict(float)
    def add(self, **kw: float) -> None:
        self.n += 1
        for k, v in kw.items():
            if v is not None and math.isfinite(float(v)):
                self.sums[k] += float(v)
    def row(self, prefix: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(prefix)
        out["gt_count"] = int(self.n)
        for k, v in self.sums.items():
            out[k] = float(v / max(self.n, 1))
        return out


def _family_from_cert(cert: str, rr: int) -> str:
    cert_l = str(cert).lower()
    if "person" in cert_l:
        return "person_conditioned"
    if "anchor" in cert_l:
        return "anchor_conditioned"
    if rr == 0 or "initial" in cert_l:
        return "initial_context_identifiable"
    if rr >= 0:
        return "resolved_other"
    return "unresolved"


def _choose_global_ref_ids(kind: str, text_vocab_ids: Sequence[int], base_ids: Set[int], weak_ids: Set[int]) -> List[int]:
    text_ids = {int(x) for x in text_vocab_ids}
    k = str(kind).strip().lower()
    if k in {"base", "base_train_vocab", "base_vocab"}:
        ids = sorted(text_ids & set(base_ids))
    elif k in {"weak", "weak_vocab", "yprime"}:
        ids = sorted(text_ids & set(weak_ids))
    elif k in {"full", "full_text", "full_text_vocab"}:
        ids = sorted(text_ids)
    else:
        raise ValueError(f"unsupported global_reference_vocab: {kind}")
    if not ids:
        raise RuntimeError(f"empty global reference vocab for kind={kind}")
    return ids


def _evaluate_direct_evidence(
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    examples: Sequence[Mapping[str, Any]],
    clip_y_base: Mapping[int, Set[int]],
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    global_ref_ids: Sequence[int],
    class_to_round: Mapping[int, int],
    class_to_cert: Mapping[int, str],
    class_names: Mapping[int, str],
    weak_ids: Set[int],
    device: torch.device,
    output_row_level: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_to_idx = {int(r): i for i, r in enumerate(text_vocab_ids)}
    projector, temperature, ckpt = _load_checkpoint(checkpoint_path, device)
    text_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        text_proj = F.normalize(projector(text_tensor), p=2.0, dim=-1)
    global_idx = torch.tensor([raw_to_idx[int(x)] for x in global_ref_ids if int(x) in raw_to_idx], device=device, dtype=torch.long)
    global_ids = [int(x) for x in global_ref_ids if int(x) in raw_to_idx]
    if len(global_ids) <= 1:
        raise RuntimeError("global reference vocabulary too small")
    global_text = text_proj[global_idx]
    global_raw_to_pos = {int(r): i for i, r in enumerate(global_ids)}

    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups[int(ex["clip_id"])].append(ex)

    stats_overall = Stats()
    grouped: Dict[Tuple[str, str], Stats] = defaultdict(Stats)
    class_stats: Dict[int, Stats] = defaultdict(Stats)
    skipped = Counter()
    row_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for clip, group in groups.items():
            candidates = sorted(int(x) for x in clip_y_base.get(int(clip), set()) if int(x) in raw_to_idx)
            if len(candidates) <= 1:
                skipped["no_or_single_candidate"] += len(group); continue
            cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj[cand_idx]
            Z = torch.stack([
                torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32)
                for ex in group
            ], dim=0)
            Z = F.normalize(Z, p=2.0, dim=-1)

            local_logits = torch.matmul(Z, cand_text.t()) / temperature
            local_probs = F.softmax(local_logits, dim=1)
            local_top2_vals, local_top2_pos = torch.topk(local_probs, k=min(2, local_probs.shape[1]), dim=1)
            local_conf = local_top2_vals[:, 0]
            local_margin = local_top2_vals[:, 0] - (local_top2_vals[:, 1] if local_top2_vals.shape[1] > 1 else torch.zeros_like(local_top2_vals[:, 0]))
            local_entropy = _entropy_norm(local_probs)
            local_order = torch.argsort(local_logits, dim=1, descending=True).detach().cpu().numpy()

            global_logits = torch.matmul(Z, global_text.t()) / temperature
            global_probs = F.softmax(global_logits, dim=1)
            global_top2_vals, global_top2_pos = torch.topk(global_probs, k=2, dim=1)
            global_conf = global_top2_vals[:, 0]
            global_margin = global_top2_vals[:, 0] - global_top2_vals[:, 1]
            global_entropy = _entropy_norm(global_probs)
            global_order = torch.argsort(global_logits, dim=1, descending=True).detach().cpu().numpy()

            for qi, ex in enumerate(group):
                gt_raw = _as_int(ex.get("matched_gt_raw_id"))
                if gt_raw is None:
                    skipped["no_gt_raw_id"] += 1; continue
                if int(gt_raw) not in candidates:
                    skipped["gt_not_in_y_base"] += 1; continue
                gt_raw_i = int(gt_raw)
                gt_pos = candidates.index(gt_raw_i)
                local_ranks = local_order[qi].tolist()
                local_gt_rank = int(local_ranks.index(gt_pos)) + 1
                local_top1_pos = int(local_ranks[0])
                local_top1_raw = int(candidates[local_top1_pos])
                local_correct = 1.0 if local_top1_raw == gt_raw_i else 0.0
                local_gt_norm_rank = float((local_gt_rank - 1) / max(len(candidates) - 1, 1))

                global_ranks = global_order[qi].tolist()
                global_top1_pos = int(global_ranks[0])
                global_top1_raw = int(global_ids[global_top1_pos])
                global_rank_of_local_top1 = int(global_ranks.index(global_raw_to_pos[local_top1_raw])) + 1 if local_top1_raw in global_raw_to_pos else len(global_ids) + 1
                global_rank_of_gt = int(global_ranks.index(global_raw_to_pos[gt_raw_i])) + 1 if gt_raw_i in global_raw_to_pos else len(global_ids) + 1
                local_global_agree = 1.0 if local_top1_raw == global_top1_raw else 0.0
                local_top1_global_rank_score = _rank_score(global_rank_of_local_top1, len(global_ids))
                gt_global_rank_score = _rank_score(global_rank_of_gt, len(global_ids))
                global_certainty = 1.0 - float(global_entropy[qi].detach().cpu().item())
                local_certainty = 1.0 - float(local_entropy[qi].detach().cpu().item())

                # Conservative direct-stability score: local top1 must survive in fixed global view.
                # Do not include GT in this score; GT rank is only reported post hoc.
                direct_stability = float(0.50 * local_top1_global_rank_score + 0.30 * local_global_agree + 0.20 * max(global_certainty, 0.0))
                context_sensitivity = float(1.0 - direct_stability)
                high_local_conf = float(local_conf[qi].detach().cpu().item())
                high_local_margin = float(local_margin[qi].detach().cpu().item())

                rr = int(class_to_round.get(gt_raw_i, -99))
                cert = str(class_to_cert.get(gt_raw_i, "unresolved"))
                family = _family_from_cert(cert, rr)
                base_group = "base_observed" if gt_raw_i in weak_ids else "base_unobserved"
                vals = {
                    "local_gt_top1_hit_rate": local_correct,
                    "local_mean_normalized_gt_rank": local_gt_norm_rank,
                    "local_gt_rank_mean": float(local_gt_rank),
                    "local_conf_mean": high_local_conf,
                    "local_margin_mean": high_local_margin,
                    "local_certainty_mean": local_certainty,
                    "global_rank_of_local_top1_mean": float(global_rank_of_local_top1),
                    "global_rank_of_gt_mean": float(global_rank_of_gt),
                    "local_top1_global_rank_score_mean": local_top1_global_rank_score,
                    "gt_global_rank_score_mean": gt_global_rank_score,
                    "local_global_top1_agreement_rate": local_global_agree,
                    "global_conf_mean": float(global_conf[qi].detach().cpu().item()),
                    "global_margin_mean": float(global_margin[qi].detach().cpu().item()),
                    "global_certainty_mean": global_certainty,
                    "direct_stability_mean": direct_stability,
                    "context_sensitivity_mean": context_sensitivity,
                }
                stats_overall.add(**vals)
                grouped[("certificate_family", family)].add(**vals)
                grouped[("certificate_type", cert)].add(**vals)
                grouped[("resolved_round", str(rr))].add(**vals)
                grouped[("base_group", base_group)].add(**vals)
                grouped[("person_conditioned", str(family == "person_conditioned").lower())].add(**vals)
                class_stats[gt_raw_i].add(**vals)
                if output_row_level:
                    row_rows.append({
                        "trajectory_id": ex.get("trajectory_id", ""),
                        "clip_id": int(clip),
                        "gt_raw_id": gt_raw_i,
                        "gt_class_name": class_names.get(gt_raw_i, str(gt_raw_i)),
                        "certificate_family": family,
                        "certificate_type": cert,
                        "resolved_round": rr,
                        "base_group": base_group,
                        "candidate_size": len(candidates),
                        "global_reference_size": len(global_ids),
                        "local_top1_raw_id": local_top1_raw,
                        "local_top1_name": class_names.get(local_top1_raw, str(local_top1_raw)),
                        "global_top1_raw_id": global_top1_raw,
                        "global_top1_name": class_names.get(global_top1_raw, str(global_top1_raw)),
                        "local_correct": int(local_correct),
                        "local_gt_rank": local_gt_rank,
                        "local_gt_norm_rank": local_gt_norm_rank,
                        "local_conf": high_local_conf,
                        "local_margin": high_local_margin,
                        "local_certainty": local_certainty,
                        "global_rank_of_local_top1": global_rank_of_local_top1,
                        "global_rank_of_gt": global_rank_of_gt,
                        "local_top1_global_rank_score": local_top1_global_rank_score,
                        "gt_global_rank_score": gt_global_rank_score,
                        "local_global_top1_agreement": int(local_global_agree),
                        "global_conf": float(global_conf[qi].detach().cpu().item()),
                        "global_margin": float(global_margin[qi].detach().cpu().item()),
                        "global_certainty": global_certainty,
                        "direct_stability": direct_stability,
                        "context_sensitivity": context_sensitivity,
                    })

    group_rows = [stats_overall.row({"checkpoint": checkpoint_name, "group_name": "overall", "group_value": "overall"})]
    for (gname, gval), st in sorted(grouped.items()):
        group_rows.append(st.row({"checkpoint": checkpoint_name, "group_name": gname, "group_value": gval}))

    per_class_rows: List[Dict[str, Any]] = []
    for rid, st in sorted(class_stats.items()):
        rr = int(class_to_round.get(rid, -99))
        cert = str(class_to_cert.get(rid, "unresolved"))
        family = _family_from_cert(cert, rr)
        base_group = "base_observed" if rid in weak_ids else "base_unobserved"
        per_class_rows.append(st.row({
            "raw_id": rid,
            "class_name": class_names.get(rid, str(rid)),
            "certificate_family": family,
            "certificate_type": cert,
            "resolved_round": rr,
            "base_group": base_group,
        }))

    summary = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "status": "PASS" if stats_overall.n > 0 else "FAIL",
        "processed_gt_rows": int(stats_overall.n),
        "skipped": dict(skipped),
        "temperature": float(temperature.detach().cpu().item()),
        "checkpoint_protocol": ckpt.get("protocol", ""),
        "global_reference_size": len(global_ids),
    }
    return summary, group_rows, per_class_rows, row_rows


def _score_predictiveness(row_rows: Sequence[Mapping[str, Any]], high_local_q: float, low_direct_q: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not row_rows:
        return [], {}
    labels = [int(_num(r.get("local_correct"))) for r in row_rows]
    score_specs = [
        ("local_conf", "local_conf"),
        ("local_margin", "local_margin"),
        ("local_certainty", "local_certainty"),
        ("global_conf", "global_conf"),
        ("global_margin", "global_margin"),
        ("global_certainty", "global_certainty"),
        ("local_global_top1_agreement", "local_global_top1_agreement"),
        ("local_top1_global_rank_score", "local_top1_global_rank_score"),
        ("direct_stability", "direct_stability"),
        ("context_sensitivity_negative", "context_sensitivity"),
    ]
    out: List[Dict[str, Any]] = []
    for name, col in score_specs:
        vals = [_num(r.get(col)) for r in row_rows]
        if name == "context_sensitivity_negative":
            vals_auc = [-v for v in vals]
        else:
            vals_auc = vals
        out.append({
            "score_name": name,
            "n": len(vals),
            "auc_predict_local_correct": _auc_score(vals_auc, labels),
            "pearson_with_correct": _pearson(vals_auc, labels),
            "spearman_with_correct": _spearman(vals_auc, labels),
            "mean_score_correct": float(np.mean([v for v, y in zip(vals, labels) if y == 1])) if any(y == 1 for y in labels) else "",
            "mean_score_wrong": float(np.mean([v for v, y in zip(vals, labels) if y == 0])) if any(y == 0 for y in labels) else "",
        })
    local_threshold = _quantile([_num(r.get("local_conf")) for r in row_rows], high_local_q)
    direct_threshold = _quantile([_num(r.get("direct_stability")) for r in row_rows], low_direct_q)
    subset = [r for r in row_rows if _num(r.get("local_conf")) >= local_threshold and _num(r.get("direct_stability")) <= direct_threshold]
    overall_error = float(1.0 - np.mean(labels)) if labels else 0.0
    subset_error = float(1.0 - np.mean([int(_num(r.get("local_correct"))) for r in subset])) if subset else 0.0
    meta = {
        "local_conf_high_quantile": high_local_q,
        "direct_stability_low_quantile": low_direct_q,
        "local_conf_threshold": local_threshold,
        "direct_stability_threshold": direct_threshold,
        "overall_error_rate": overall_error,
        "high_local_low_direct_count": len(subset),
        "high_local_low_direct_error_rate": subset_error,
        "high_local_low_direct_error_enrichment": (subset_error / overall_error) if overall_error > 0 else None,
    }
    return out, meta


def _top_rows(rows: Sequence[Mapping[str, Any]], key: str, *, reverse: bool = True, min_count: int = 1, k: int = 20) -> List[Dict[str, Any]]:
    vals = [dict(r) for r in rows if _num(r.get("gt_count")) >= min_count]
    vals.sort(key=lambda r: _num(r.get(key)), reverse=reverse)
    return vals[:k]


def parse_checkpoint_arg(text: str) -> Tuple[str, Path]:
    if "=" not in str(text):
        raise argparse.ArgumentTypeError("checkpoint must be name=/path/to/prealign_last.pth")
    name, path = str(text).split("=", 1)
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual/global-reference direct evidence audit for clean GT-fullY nohub runs.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--schedule_csv", default="")
    p.add_argument("--residual_variant", default="person_aware")
    p.add_argument("--checkpoint", action="append", type=parse_checkpoint_arg, required=True)
    p.add_argument("--global_reference_vocab", default="base_train_vocab", choices=["base_train_vocab", "base", "weak", "weak_vocab", "full_text", "full_text_vocab"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--write_row_level", action="store_true", help="Write row_level_direct_evidence.csv; useful but can be several MB.")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--min_class_gt_count", type=int, default=5)
    p.add_argument("--high_local_quantile", type=float, default=0.75)
    p.add_argument("--low_direct_quantile", type=float, default=0.25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    _bootstrap_asset_links(out, asset_root)
    examples, clip_y_base, base_ids, mat_meta = _load_examples_with_gt(args, out)
    weak_ids = _load_weak_vocab(repo_root, mat_meta)
    class_to_round, class_to_cert = _load_schedule(Path(args.schedule_csv), variant=str(args.residual_variant), base_ids=base_ids)
    class_names = _load_class_names(Path(args.annotation_json))
    text_vocab_ids, _records, text_vocab_matrix = load_text_vocab(out)
    global_ref_ids = _choose_global_ref_ids(str(args.global_reference_vocab), text_vocab_ids, base_ids, weak_ids)
    device = torch.device(str(args.device))

    all_summary: List[Dict[str, Any]] = []
    all_group_rows: List[Dict[str, Any]] = []
    all_class_rows: List[Dict[str, Any]] = []
    all_row_rows: List[Dict[str, Any]] = []
    for name, ckpt in args.checkpoint:
        summary, group_rows, class_rows, row_rows = _evaluate_direct_evidence(
            checkpoint_name=name,
            checkpoint_path=ckpt,
            examples=examples,
            clip_y_base=clip_y_base,
            text_vocab_ids=text_vocab_ids,
            text_vocab_matrix=text_vocab_matrix,
            global_ref_ids=global_ref_ids,
            class_to_round=class_to_round,
            class_to_cert=class_to_cert,
            class_names=class_names,
            weak_ids=weak_ids,
            device=device,
            output_row_level=True,  # always keep in memory for summaries; file writing is optional
        )
        all_summary.append(summary)
        all_group_rows.extend(group_rows)
        for r in class_rows:
            rr = dict(r); rr["checkpoint"] = name; all_class_rows.append(rr)
        for r in row_rows:
            rr = dict(r); rr["checkpoint"] = name; all_row_rows.append(rr)

    score_rows, threshold_meta = _score_predictiveness(all_row_rows, float(args.high_local_quantile), float(args.low_direct_quantile))

    _write_csv(out / "summary_by_group.csv", all_group_rows)
    _write_csv(out / "score_predictiveness.csv", score_rows)
    _write_csv(out / "per_class_direct_evidence.csv", all_class_rows)
    if args.write_row_level:
        _write_csv(out / "row_level_direct_evidence.csv", all_row_rows)

    # Row-level high-local/low-direct cases, prioritized by wrong cases then local confidence.
    local_thr = threshold_meta.get("local_conf_threshold", 1.0)
    direct_thr = threshold_meta.get("direct_stability_threshold", 0.0)
    high_local_low_direct = [r for r in all_row_rows if _num(r.get("local_conf")) >= float(local_thr) and _num(r.get("direct_stability")) <= float(direct_thr)]
    high_local_low_direct.sort(key=lambda r: (int(_num(r.get("local_correct"))), -_num(r.get("local_conf")), _num(r.get("direct_stability"))))
    _write_csv(out / "high_local_low_direct_error_cases.csv", high_local_low_direct[: max(200, int(args.top_k) * 10)])

    # Class-level rankings.
    top_k = int(args.top_k)
    min_count = int(args.min_class_gt_count)
    # Context sensitivity high means local top1 does not survive global reference.
    _write_csv(out / "top20_context_sensitive_classes.csv", _top_rows(all_class_rows, "context_sensitivity_mean", reverse=True, min_count=min_count, k=top_k))
    _write_csv(out / "top20_direct_stable_classes.csv", _top_rows(all_class_rows, "direct_stability_mean", reverse=True, min_count=min_count, k=top_k))
    wrong_proxy: List[Dict[str, Any]] = []
    for r in all_class_rows:
        rr = dict(r)
        rr["local_conf_wrong_score"] = _num(r.get("local_conf_mean")) * (1.0 - _num(r.get("local_gt_top1_hit_rate")))
        wrong_proxy.append(rr)
    _write_csv(out / "top20_local_conf_wrong_classes.csv", _top_rows(wrong_proxy, "local_conf_wrong_score", reverse=True, min_count=min_count, k=top_k))

    status = "PASS" if all(s.get("status") == "PASS" for s in all_summary) and all_row_rows else "FAIL"
    # Simple decision hints.
    auc_by = {str(r.get("score_name")): _num(r.get("auc_predict_local_correct"), float("nan")) for r in score_rows}
    local_auc = auc_by.get("local_conf")
    direct_auc = auc_by.get("direct_stability")
    auc_gain = None
    if local_auc is not None and direct_auc is not None and math.isfinite(local_auc) and math.isfinite(direct_auc):
        auc_gain = float(direct_auc - local_auc)
    highlow_enrichment = threshold_meta.get("high_local_low_direct_error_enrichment")
    if auc_gain is not None and auc_gain > 0.03:
        interpretation = "STRONG_SIGNAL_DIRECT_STABILITY_OUTPERFORMS_LOCAL_CONF"
    elif highlow_enrichment is not None and _num(highlow_enrichment) > 1.5:
        interpretation = "STRONG_SIGNAL_HIGH_LOCAL_LOW_DIRECT_ERRORS_ARE_ENRICHED"
    elif auc_gain is not None and auc_gain > 0.0:
        interpretation = "WEAK_POSITIVE_SIGNAL_DIRECT_STABILITY"
    else:
        interpretation = "NO_CLEAR_DIRECT_EVIDENCE_ADVANTAGE_YET"

    payload = {
        "status": status,
        "output_dir": str(out),
        "dataset_name": str(args.dataset_name),
        "checkpoint_summaries": all_summary,
        "materialization": mat_meta,
        "gt_example_count": len(examples),
        "evaluated_gt_rows": len(all_row_rows),
        "base_count": len(base_ids),
        "weak_vocab_count": len(weak_ids),
        "global_reference_vocab": str(args.global_reference_vocab),
        "global_reference_count": len(global_ref_ids),
        "schedule_resolved_count": len(class_to_round),
        "predictiveness_threshold_meta": threshold_meta,
        "local_conf_auc": local_auc,
        "direct_stability_auc": direct_auc,
        "direct_minus_local_auc": auc_gain,
        "interpretation": interpretation,
        "outputs": {
            "summary_by_group": str(out / "summary_by_group.csv"),
            "score_predictiveness": str(out / "score_predictiveness.csv"),
            "per_class_direct_evidence": str(out / "per_class_direct_evidence.csv"),
            "row_level_direct_evidence": str(out / "row_level_direct_evidence.csv") if args.write_row_level else "not_written_use_--write_row_level",
            "high_local_low_direct_error_cases": str(out / "high_local_low_direct_error_cases.csv"),
        },
    }
    _write_json(out / "summary.json", payload)

    takeover = out / "COUNTERFACTUAL_DIRECT_EVIDENCE_AUDIT_TAKEOVER.md"
    takeover.write_text(
        "# Counterfactual Direct Evidence Audit Takeover\n\n"
        f"Status: `{status}`\n\n"
        f"Output: `{out}`\n\n"
        "## Scope\n\n"
        "Read-only GT-fullY clean audit. No training, no checkpoint modification, no VideoCutLER/Y′/extra/mAP.\n"
        "GT is used only for post-hoc diagnosis of local/global evidence predictiveness.\n\n"
        "## Compared evidence views\n\n"
        "- local view: full `Y_base(v)` candidate context.\n"
        f"- global reference view: `{args.global_reference_vocab}` with `{len(global_ref_ids)}` classes.\n\n"
        "## Key findings\n\n"
        f"- evaluated GT rows: `{len(all_row_rows)}`\n"
        f"- local_conf_auc: `{local_auc}`\n"
        f"- direct_stability_auc: `{direct_auc}`\n"
        f"- direct_minus_local_auc: `{auc_gain}`\n"
        f"- high-local/low-direct error enrichment: `{highlow_enrichment}`\n"
        f"- interpretation: `{interpretation}`\n\n"
        "## Core outputs\n\n"
        "- summary.json\n"
        "- summary_by_group.csv\n"
        "- score_predictiveness.csv\n"
        "- per_class_direct_evidence.csv\n"
        "- high_local_low_direct_error_cases.csv\n"
        "- top20_context_sensitive_classes.csv\n"
        "- top20_direct_stable_classes.csv\n"
        "- top20_local_conf_wrong_classes.csv\n\n"
        "## Required follow-up\n\n"
        "If direct_stability AUC is meaningfully better than local_conf, or high-local/low-direct rows are error-enriched, implement the training-side direct gate. Otherwise do not replace nohub gating yet.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
