#!/usr/bin/env python3
"""Clean GT-fullY attribution comparison for mechanism-only runs.

This read-only replay compares checkpoints produced by run_stageb_train_gt_full_y_clean.py
on the same GT carrier trajectories and full official-base clip label context.  It reports
only the core attribution metrics and stratifies by residual resolved round/certificate.
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

REPO_ASSET_LINK_NAMES = ("exports", "exports_gt", "carrier_bank", "carrier_bank_gt", "frame_bank", "text_bank", "gt_sidecar_bank", "weak_labels", "weights", "dataset", "eval")


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(str(k)); seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


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
                if found: return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if found: return
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


def _load_annotation_categories(annotation_json: Path) -> Dict[int, str]:
    with annotation_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, str] = {}
    for cat in obj.get("categories", []):
        if not isinstance(cat, Mapping):
            continue
        cid = _as_int(cat.get("id", cat.get("category_id", cat.get("raw_id"))))
        if cid is None:
            continue
        name = str(cat.get("name", cid))
        out[int(cid)] = name
    return out


def _load_schedule(csv_path: Path, *, variant: str, base_ids: Set[int]) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, Dict[str, Any]]]:
    class_to_round: Dict[int, int] = {}
    class_to_cert: Dict[int, str] = {}
    class_rows: Dict[int, Dict[str, Any]] = {}
    if not csv_path or not csv_path.is_file():
        return class_to_round, class_to_cert, class_rows
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant", "")) != str(variant):
                continue
            rid = _as_int(row.get("raw_id"))
            if rid is None or int(rid) not in base_ids:
                continue
            resolved = _truthy(row.get("resolved"))
            rr = _as_int(row.get("resolved_at_iteration"))
            cert = str(row.get("certificate_type", "")).strip()
            class_rows[int(rid)] = {
                "raw_id": int(rid),
                "class_name": str(row.get("class_name", "")),
                "clip_count": _as_int(row.get("clip_count")),
                "instance_count": _as_int(row.get("instance_count")),
                "resolved": bool(resolved),
                "resolved_at_iteration": rr,
                "certificate_type": cert,
                "variant": str(row.get("variant", "")),
                "row": dict(row),
            }
            if resolved and rr is not None:
                class_to_round[int(rid)] = int(rr)
                class_to_cert[int(rid)] = cert or "unknown"
    return class_to_round, class_to_cert, class_rows


def _resolved_round_bucket(schedule_row: Optional[Mapping[str, Any]]) -> str:
    if not schedule_row:
        return "absent_or_unknown"
    if not bool(schedule_row.get("resolved", False)):
        return "unresolved"
    rr = _as_int(schedule_row.get("resolved_at_iteration"))
    if rr is None:
        return "unresolved"
    if int(rr) in {0, 1, 2, 3}:
        return str(int(rr))
    return "unresolved"


def _certificate_family(schedule_row: Optional[Mapping[str, Any]]) -> str:
    if not schedule_row:
        return "absent_or_unknown"
    if not bool(schedule_row.get("resolved", False)):
        return "unresolved"
    cert = str(schedule_row.get("certificate_type", "")).strip().lower()
    if not cert:
        return "other_conditioned"
    if "initial_context_identifiable" in cert or "initial" in cert or "context_identifiable" in cert:
        return "initial_context_identifiable"
    if "person" in cert:
        return "person_conditioned"
    if "anchor" in cert:
        return "anchor_conditioned"
    return "other_conditioned"


def _base_group(raw_id: Optional[int], base_ids: Set[int], weak_vocab_raw_ids: Set[int]) -> str:
    if raw_id is None:
        return "other_or_unknown"
    if int(raw_id) not in base_ids:
        return "other_or_unknown"
    if int(raw_id) in weak_vocab_raw_ids:
        return "base_observed"
    return "base_unobserved"


def _person_conditioned(schedule_row: Optional[Mapping[str, Any]]) -> str:
    if not schedule_row:
        return "unknown"
    cert = str(schedule_row.get("certificate_type", "")).lower()
    if not cert:
        return "unknown"
    if "person" in cert:
        return "true"
    return "false"


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
            if not line: continue
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
    resolution = dict(materialized.get("resolution", {}))
    stats = dict(materialized.get("stats", {}))
    weak_vocab = dict(resolution.get("weak_label_vocab") or stats.get("weak_label_vocab") or {})
    meta = {
        "materialized_stats": stats,
        "materialized_resolution": resolution,
        "weak_label_vocab": weak_vocab,
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


def _evaluate_checkpoint(
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    examples: Sequence[Mapping[str, Any]],
    clip_y_base: Mapping[int, Set[int]],
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    class_to_round: Mapping[int, int],
    class_to_cert: Mapping[int, str],
    class_rows: Mapping[int, Mapping[str, Any]],
    class_names: Mapping[int, str],
    base_ids: Set[int],
    weak_vocab_raw_ids: Set[int],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_to_idx = {int(r): i for i, r in enumerate(text_vocab_ids)}
    projector, temperature, ckpt = _load_checkpoint(checkpoint_path, device)
    text_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        text_proj = F.normalize(projector(text_tensor), p=2.0, dim=-1)
    groups = defaultdict(list)
    for ex in examples:
        groups[int(ex["clip_id"])].append(ex)
    stats_overall = Stats()
    grouped: Dict[Tuple[str, str], Stats] = defaultdict(Stats)
    per_class: Dict[int, Stats] = defaultdict(Stats)
    per_class_rows: Dict[int, Dict[str, Any]] = {}
    skipped = Counter()
    with torch.no_grad():
        for clip, group in groups.items():
            candidates = sorted(int(x) for x in clip_y_base.get(int(clip), set()) if int(x) in raw_to_idx)
            if not candidates:
                skipped["no_candidate"] += len(group); continue
            cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj[cand_idx]
            Z = torch.stack([torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32) for ex in group], dim=0)
            Z = F.normalize(Z, p=2.0, dim=-1)
            scores = torch.matmul(Z, cand_text.t()) / temperature
            order = torch.argsort(scores, dim=1, descending=True).detach().cpu().numpy()
            for qi, ex in enumerate(group):
                gt_raw = _as_int(ex.get("matched_gt_raw_id"))
                if gt_raw is None:
                    skipped["no_gt_raw_id"] += 1; continue
                if int(gt_raw) not in candidates:
                    skipped["gt_not_in_y_base"] += 1; continue
                gt_pos = candidates.index(int(gt_raw))
                ranks = order[qi].tolist()
                rank = int(ranks.index(gt_pos)) + 1
                denom = max(len(candidates) - 1, 1)
                norm_rank = float((rank - 1) / denom)
                top1 = 1.0 if rank == 1 else 0.0
                schedule_row = class_rows.get(int(gt_raw))
                rr_bucket = _resolved_round_bucket(schedule_row)
                cert = str(schedule_row.get("certificate_type", "")).strip() if schedule_row else ""
                family = _certificate_family(schedule_row)
                base_group = _base_group(int(gt_raw), base_ids, weak_vocab_raw_ids)
                person_flag = _person_conditioned(schedule_row)
                vals = {
                    "mean_normalized_gt_rank": norm_rank,
                    "gt_top1_hit_rate": top1,
                    "candidate_size_mean": float(len(candidates)),
                    "gt_rank_mean": float(rank),
                }
                stats_overall.add(**vals)
                per_class[int(gt_raw)].add(**vals)
                per_class_rows.setdefault(int(gt_raw), {
                    "raw_id": int(gt_raw),
                    "class_name": str((schedule_row or {}).get("class_name") or class_names.get(int(gt_raw), str(gt_raw))),
                    "resolved_round": rr_bucket,
                    "certificate_type": cert,
                    "certificate_family": family,
                    "base_group": base_group,
                    "person_conditioned": person_flag,
                })
                grouped[("resolved_round", rr_bucket)].add(**vals)
                grouped[("certificate_type", cert)].add(**vals)
                grouped[("certificate_family", family)].add(**vals)
                grouped[("base_observed_unobserved", base_group)].add(**vals)
                grouped[("person_conditioned", person_flag)].add(**vals)
    rows = [stats_overall.row({"checkpoint": checkpoint_name, "group_name": "overall", "group_value": "overall"})]
    for (gname, gval), st in sorted(grouped.items()):
        rows.append(st.row({"checkpoint": checkpoint_name, "group_name": gname, "group_value": gval}))
    per_class_rows_out: List[Dict[str, Any]] = []
    for rid, st in sorted(per_class.items()):
        base_row = per_class_rows.get(int(rid), {
            "raw_id": int(rid),
            "class_name": str(class_names.get(int(rid), rid)),
            "resolved_round": "absent_or_unknown",
            "certificate_type": "",
            "certificate_family": "absent_or_unknown",
            "base_group": "other_or_unknown",
            "person_conditioned": "unknown",
        })
        per_class_rows_out.append(st.row({"checkpoint": checkpoint_name, **base_row}))
    summary = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "status": "PASS",
        "processed_gt_rows": int(stats_overall.n),
        "skipped": dict(skipped),
        "temperature": float(temperature.detach().cpu().item()),
        "checkpoint_protocol": ckpt.get("protocol", ""),
    }
    if stats_overall.n <= 0:
        summary["status"] = "FAIL"
    return summary, rows, per_class_rows_out


def parse_checkpoint_arg(text: str) -> Tuple[str, Path]:
    if "=" not in str(text):
        raise argparse.ArgumentTypeError("checkpoint must be name=/path/to/prealign_last.pth")
    name, path = str(text).split("=", 1)
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare clean GT-fullY mechanism checkpoints.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--schedule_csv", default="")
    p.add_argument("--residual_variant", default="person_aware")
    p.add_argument("--checkpoint", action="append", type=parse_checkpoint_arg, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Use output_dir as temporary asset-linked root for text/carrier loading.
    _bootstrap_asset_links(out, Path(args.asset_root).expanduser().resolve())
    examples, clip_y_base, base_ids, mat_meta = _load_examples_with_gt(args, out)
    class_to_round, class_to_cert, class_rows = _load_schedule(Path(args.schedule_csv), variant=str(args.residual_variant), base_ids=base_ids)
    class_names = _load_annotation_categories(Path(args.annotation_json))
    weak_vocab_raw_ids = {int(x) for x in (mat_meta.get("weak_label_vocab", {}) or {}).get("weak_vocab_raw_ids", [])}
    text_vocab_ids, _records, text_vocab_matrix = load_text_vocab(out)
    device = torch.device(str(args.device))
    all_summary: List[Dict[str, Any]] = []
    all_group_rows: List[Dict[str, Any]] = []
    all_per_class_rows: List[Dict[str, Any]] = []
    for name, ckpt in args.checkpoint:
        summary, rows, per_class_rows = _evaluate_checkpoint(
            checkpoint_name=name, checkpoint_path=ckpt, examples=examples, clip_y_base=clip_y_base,
            text_vocab_ids=text_vocab_ids, text_vocab_matrix=text_vocab_matrix, class_to_round=class_to_round,
            class_to_cert=class_to_cert, class_rows=class_rows, class_names=class_names,
            base_ids=base_ids, weak_vocab_raw_ids=weak_vocab_raw_ids, device=device,
        )
        all_summary.append(summary)
        all_group_rows.extend(rows)
        all_per_class_rows.extend(per_class_rows)

    def _filter_group(group_name: str) -> List[Dict[str, Any]]:
        return [r for r in all_group_rows if r.get("group_name") == group_name]

    _write_csv(out / "summary_by_group.csv", all_group_rows)
    _write_csv(out / "summary_by_run.csv", _filter_group("overall"))
    _write_csv(out / "summary_by_resolved_round.csv", _filter_group("resolved_round"))
    _write_csv(out / "summary_by_certificate_family.csv", _filter_group("certificate_family"))
    _write_csv(out / "summary_by_certificate_type.csv", _filter_group("certificate_type"))
    _write_csv(out / "summary_by_base_observed_unobserved.csv", _filter_group("base_observed_unobserved"))
    _write_csv(out / "summary_by_person_conditioned.csv", _filter_group("person_conditioned"))
    _write_csv(out / "per_class_attribution.csv", all_per_class_rows)

    # Delta tables vs baseline.
    overall_rows = _filter_group("overall")
    baseline_by_checkpoint: Dict[str, Dict[str, Any]] = {}
    if overall_rows:
        baseline = overall_rows[0]
        baseline_by_checkpoint[baseline.get("checkpoint", "")] = baseline
    for row in overall_rows:
        baseline_by_checkpoint.setdefault(row.get("checkpoint", ""), row)
    group_deltas: List[Dict[str, Any]] = []
    if overall_rows:
        baseline = overall_rows[0]
        baseline_ckpt = str(baseline.get("checkpoint", ""))
        for row in all_group_rows:
            if row.get("checkpoint") == baseline_ckpt:
                continue
            key = (row.get("group_name"), row.get("group_value"))
            base_row = next((r for r in all_group_rows if r.get("checkpoint") == baseline_ckpt and r.get("group_name") == key[0] and r.get("group_value") == key[1]), None)
            if base_row is None:
                continue
            group_deltas.append({
                "baseline": baseline_ckpt,
                "run": row.get("checkpoint"),
                "group_name": row.get("group_name"),
                "group_value": row.get("group_value"),
                "gt_count": row.get("gt_count"),
                "baseline_gt_count": base_row.get("gt_count"),
                "baseline_mean_normalized_gt_rank": base_row.get("mean_normalized_gt_rank"),
                "baseline_gt_top1_hit_rate": base_row.get("gt_top1_hit_rate"),
                "baseline_gt_rank_mean": base_row.get("gt_rank_mean"),
                "baseline_candidate_size_mean": base_row.get("candidate_size_mean"),
                "delta_mean_normalized_gt_rank": float(row.get("mean_normalized_gt_rank", 0.0)) - float(base_row.get("mean_normalized_gt_rank", 0.0)),
                "delta_gt_top1_hit_rate": float(row.get("gt_top1_hit_rate", 0.0)) - float(base_row.get("gt_top1_hit_rate", 0.0)),
                "delta_gt_rank_mean": float(row.get("gt_rank_mean", 0.0)) - float(base_row.get("gt_rank_mean", 0.0)),
            })
    _write_csv(out / "summary_delta_vs_baseline_by_group.csv", group_deltas)

    per_class_by_run: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in all_per_class_rows:
        per_class_by_run[(str(row.get("checkpoint", "")), int(row.get("raw_id", -1)))] = row
    class_delta_rows: List[Dict[str, Any]] = []
    if overall_rows:
        baseline_ckpt = str(overall_rows[0].get("checkpoint", ""))
        baseline_classes = {int(r.get("raw_id", -1)): r for r in all_per_class_rows if r.get("checkpoint") == baseline_ckpt}
        for row in all_per_class_rows:
            ckpt = str(row.get("checkpoint", ""))
            if ckpt == baseline_ckpt:
                continue
            base_row = baseline_classes.get(int(row.get("raw_id", -1)))
            if base_row is None:
                continue
            class_delta_rows.append({
                "baseline": baseline_ckpt,
                "run": ckpt,
                "raw_id": row.get("raw_id"),
                "class_name": row.get("class_name"),
                "resolved_round": row.get("resolved_round"),
                "certificate_type": row.get("certificate_type"),
                "certificate_family": row.get("certificate_family"),
                "base_group": row.get("base_group"),
                "person_conditioned": row.get("person_conditioned"),
                "baseline_mean_normalized_gt_rank": base_row.get("mean_normalized_gt_rank"),
                "baseline_gt_top1_hit_rate": base_row.get("gt_top1_hit_rate"),
                "baseline_gt_rank_mean": base_row.get("gt_rank_mean"),
                "baseline_candidate_size_mean": base_row.get("candidate_size_mean"),
                "mean_normalized_gt_rank": row.get("mean_normalized_gt_rank"),
                "gt_top1_hit_rate": row.get("gt_top1_hit_rate"),
                "gt_rank_mean": row.get("gt_rank_mean"),
                "candidate_size_mean": row.get("candidate_size_mean"),
                "delta_mean_normalized_gt_rank": float(row.get("mean_normalized_gt_rank", 0.0)) - float(base_row.get("mean_normalized_gt_rank", 0.0)),
                "delta_gt_top1_hit_rate": float(row.get("gt_top1_hit_rate", 0.0)) - float(base_row.get("gt_top1_hit_rate", 0.0)),
                "delta_gt_rank_mean": float(row.get("gt_rank_mean", 0.0)) - float(base_row.get("gt_rank_mean", 0.0)),
            })
    _write_csv(out / "per_class_delta_vs_baseline.csv", class_delta_rows)
    _write_csv(out / "summary_delta_vs_first.csv", [{"baseline": baseline_ckpt, "run": r.get("checkpoint"), "delta_mean_normalized_gt_rank": float(r.get("mean_normalized_gt_rank", 0.0)) - float(next((b.get("mean_normalized_gt_rank") for b in overall_rows if b.get("checkpoint") == baseline_ckpt), 0.0)), "delta_gt_top1_hit_rate": float(r.get("gt_top1_hit_rate", 0.0)) - float(next((b.get("gt_top1_hit_rate") for b in overall_rows if b.get("checkpoint") == baseline_ckpt), 0.0)), "gt_count": r.get("gt_count")} for r in overall_rows[1:]] if overall_rows else [])

    status = "PASS" if all(s.get("status") == "PASS" for s in all_summary) else "FAIL"
    skipped_counts = dict(all_summary[0].get("skipped", {})) if all_summary else {}
    payload = {
        "status": status,
        "output_dir": str(out),
        "dataset_name": str(args.dataset_name),
        "checkpoint_summaries": all_summary,
        "materialization": mat_meta,
        "gt_example_count": len(examples),
        "base_count": len(base_ids),
        "weak_vocab_count": len(weak_vocab_raw_ids),
        "schedule_resolved_count": len(class_rows),
        "skipped_counts": skipped_counts,
        "comparable_gt_count": int(overall_rows[0]["gt_count"]) if overall_rows else 0,
        "group_output_files": [
            "summary_by_group.csv",
            "summary_by_run.csv",
            "summary_by_resolved_round.csv",
            "summary_by_certificate_family.csv",
            "summary_by_certificate_type.csv",
            "summary_by_base_observed_unobserved.csv",
            "summary_by_person_conditioned.csv",
            "per_class_attribution.csv",
            "per_class_delta_vs_baseline.csv",
            "summary_delta_vs_baseline_by_group.csv",
        ],
    }
    _write_json(out / "summary.json", payload)
    takeover = out / "GT_FULL_Y_CLEAN_GROUPED_ATTRIBUTION_TAKEOVER.md"
    takeover.write_text(
        "# GT Full-Y Clean Grouped Attribution Diagnosis\n\n"
        + f"Status: `{status}`\n\n"
        + f"Output: `{out}`\n\n"
        + "Core outputs:\n"
        + "- summary.json\n"
        + "- summary_by_run.csv\n"
        + "- summary_by_resolved_round.csv\n"
        + "- summary_by_certificate_family.csv\n"
        + "- summary_by_certificate_type.csv\n"
        + "- summary_by_base_observed_unobserved.csv\n"
        + "- summary_by_person_conditioned.csv\n"
        + "- per_class_attribution.csv\n"
        + "- per_class_delta_vs_baseline.csv\n"
        + "- summary_delta_vs_baseline_by_group.csv\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
