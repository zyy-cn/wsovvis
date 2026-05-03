#!/usr/bin/env python3
"""
Build a class-balanced training manifest from a residual-gated row seed pool.

This script is intentionally read-only with respect to model/code/control-plane state.
It only reads CSV artifacts and writes lightweight CSV/JSON/MD summaries.

Expected input seed CSV columns include:
  clip_id, trajectory_id, gt_raw_id, gt_class_name, policy, seed_type,
  certificate_type, clip_count, instance_count, oracle_top1_is_gt,
  weak_base_top1_is_gt, weak_base_top1_class_name, weak_base_gt_rank,
  weak_base_error_type, weak_nohub_top1_is_gt, weak_nohub_top1_class_name,
  weak_nohub_gt_rank, weak_nohub_error_type, nohub_rescued_baseline_wrong,
  oracle_correct_weak_base_wrong, clip_y_size

Seed types consumed:
  - hard_ce_seed -> hard_ce
  - soft_ce_seed -> soft_ce
  - prototype_seed -> prototype_calibration
  - deferred -> never selected
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", type=str, default="", help="Experiment root used only to locate default seed pool/output paths.")
    p.add_argument("--dataset_name", type=str, default="lvvis_train_base")
    p.add_argument("--seed_pool_csv", type=str, default="", help="Path to residual_gated_row_seed_pool.csv. If omitted, infer from run_root.")
    p.add_argument("--out_dir", type=str, default="", help="Output directory. If omitted, infer from run_root.")

    p.add_argument("--hard_cap_per_class", type=int, default=32)
    p.add_argument("--soft_cap_per_class", type=int, default=32)
    p.add_argument("--prototype_cap_per_class", type=int, default=16)
    p.add_argument("--person_hard_cap", type=int, default=64)
    p.add_argument("--hub_hard_cap", type=int, default=64)
    p.add_argument("--person_raw_ids", type=str, default="773", help="Comma-separated raw ids treated as person/hub-person.")
    p.add_argument("--hub_raw_ids", type=str, default="773", help="Comma-separated raw ids treated as hub classes for stricter caps/weights.")

    p.add_argument("--max_person_hard_share", type=float, default=0.20, help="Gate: selected hard CE person share should not exceed this.")
    p.add_argument("--max_hub_hard_share", type=float, default=0.25, help="Gate: selected hard CE hub share should not exceed this.")
    p.add_argument("--hub_sample_weight", type=float, default=0.25)
    p.add_argument("--default_sample_weight", type=float, default=1.0)
    p.add_argument("--soft_sample_weight", type=float, default=0.5)
    p.add_argument("--prototype_sample_weight", type=float, default=0.25)

    p.add_argument("--prefer_multilabel", action="store_true", default=True, help="Prefer rows with clip_y_size>1 within each class.")
    p.add_argument("--no_prefer_multilabel", dest="prefer_multilabel", action="store_false")
    p.add_argument("--seed", type=int, default=3407, help="Deterministic tie-break seed.")
    p.add_argument("--shuffle_ties", action="store_true", help="Shuffle exact ties deterministically. Default false for stable path order.")

    p.add_argument("--min_selected_hard_classes", type=int, default=20)
    p.add_argument("--min_selected_nonhub_hard_rows", type=int, default=256)
    return p.parse_args()


def ensure_path_args(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.seed_pool_csv:
        seed_pool = Path(args.seed_pool_csv)
    else:
        if not args.run_root:
            raise SystemExit("Either --seed_pool_csv or --run_root is required.")
        seed_pool = Path(args.run_root) / "analysis" / "residual_gated_row_seed_pool" / args.dataset_name / "residual_gated_row_seed_pool.csv"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if not args.run_root:
            raise SystemExit("Either --out_dir or --run_root is required.")
        out_dir = Path(args.run_root) / "analysis" / "residual_gated_training_manifest" / args.dataset_name
    return seed_pool, out_dir


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return [dict(r) for r in reader]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def parse_id_set(s: str) -> set[str]:
    out = set()
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(str(int(float(part))))
        except Exception:
            out.add(part)
    return out


def raw_norm(x: object) -> str:
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()


def truth(x: object) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def fnum(x: object, default: float = math.inf) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def make_priority(row: Dict[str, str], loss_family: str, prefer_multilabel: bool) -> Tuple:
    """Lower tuple sorts earlier. Designed to prefer informative, safe non-trivial rows."""
    clip_y = fnum(row.get("clip_y_size"), 0.0)
    nohub_rescued = truth(row.get("nohub_rescued_baseline_wrong"))
    oracle_correct_weak_base_wrong = truth(row.get("oracle_correct_weak_base_wrong"))
    weak_base_correct = truth(row.get("weak_base_top1_is_gt"))
    weak_nohub_correct = truth(row.get("weak_nohub_top1_is_gt"))
    nohub_rank = fnum(row.get("weak_nohub_gt_rank"), 999999.0)
    base_rank = fnum(row.get("weak_base_gt_rank"), 999999.0)
    oracle_rank = fnum(row.get("oracle_gt_rank"), 999999.0)

    # Multilabel rows expose positive-set ambiguity. Single-label rows are useful but less diagnostic.
    multilabel_key = 0 if (clip_y > 1 and prefer_multilabel) else 1

    if loss_family == "hard_ce":
        return (
            0 if nohub_rescued else 1,
            0 if oracle_correct_weak_base_wrong else 1,
            multilabel_key,
            0 if weak_nohub_correct else 1,
            nohub_rank,
            base_rank,
            oracle_rank,
            str(row.get("clip_id", "")),
            str(row.get("trajectory_id", "")),
        )
    if loss_family == "soft_ce":
        return (
            0 if oracle_correct_weak_base_wrong else 1,
            multilabel_key,
            nohub_rank,
            0 if weak_base_correct else 1,
            base_rank,
            oracle_rank,
            str(row.get("clip_id", "")),
            str(row.get("trajectory_id", "")),
        )
    # Prototype rows should favor oracle-correct but weak-failed rows, because they expose scorer misalignment.
    return (
        0 if oracle_correct_weak_base_wrong else 1,
        multilabel_key,
        nohub_rank,
        base_rank,
        oracle_rank,
        str(row.get("clip_id", "")),
        str(row.get("trajectory_id", "")),
    )


def cap_for(raw_id: str, loss_family: str, args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> int:
    if loss_family == "hard_ce":
        if raw_id in person_ids:
            return max(0, args.person_hard_cap)
        if raw_id in hub_ids:
            return max(0, args.hub_hard_cap)
        return max(0, args.hard_cap_per_class)
    if loss_family == "soft_ce":
        return max(0, args.soft_cap_per_class)
    if loss_family == "prototype_calibration":
        return max(0, args.prototype_cap_per_class)
    return 0


def loss_family_for_seed(seed_type: str) -> str | None:
    if seed_type == "hard_ce_seed":
        return "hard_ce"
    if seed_type == "soft_ce_seed":
        return "soft_ce"
    if seed_type == "prototype_seed":
        return "prototype_calibration"
    return None


def sample_weight_for(raw_id: str, loss_family: str, args: argparse.Namespace, hub_ids: set[str]) -> float:
    if raw_id in hub_ids:
        return args.hub_sample_weight
    if loss_family == "soft_ce":
        return args.soft_sample_weight
    if loss_family == "prototype_calibration":
        return args.prototype_sample_weight
    return args.default_sample_weight


def select_rows(rows: List[Dict[str, str]], args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    rng = random.Random(args.seed)
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)  # (loss_family, raw_id)
    rejected: List[Dict[str, object]] = []

    for r in rows:
        seed_type = str(r.get("seed_type", "")).strip()
        loss_family = loss_family_for_seed(seed_type)
        raw_id = raw_norm(r.get("gt_raw_id"))
        if loss_family is None:
            rr = dict(r)
            rr.update({"reject_reason": "seed_type_not_trainable", "loss_family": ""})
            rejected.append(rr)
            continue
        grouped[(loss_family, raw_id)].append(r)

    selected: List[Dict[str, object]] = []
    by_class_rows: List[Dict[str, object]] = []

    for (loss_family, raw_id), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if args.shuffle_ties:
            rng.shuffle(group)
        group_sorted = sorted(group, key=lambda r: make_priority(r, loss_family, args.prefer_multilabel))
        cap = cap_for(raw_id, loss_family, args, person_ids, hub_ids)
        take = group_sorted[:cap]
        rest = group_sorted[cap:]

        for rank, r in enumerate(take):
            out = dict(r)
            out.update({
                "manifest_use": "train",
                "loss_family": loss_family,
                "selection_rank_within_class_loss": rank + 1,
                "class_cap_used": cap,
                "is_person_raw_id": str(raw_id in person_ids),
                "is_hub_raw_id": str(raw_id in hub_ids),
                "sample_weight": sample_weight_for(raw_id, loss_family, args, hub_ids),
                "selection_reason": f"selected_by_{loss_family}_cap",
            })
            selected.append(out)

        for r in rest:
            rr = dict(r)
            rr.update({
                "reject_reason": f"over_class_cap_{loss_family}",
                "loss_family": loss_family,
                "class_cap_used": cap,
            })
            rejected.append(rr)

        first = group_sorted[0] if group_sorted else {}
        by_class_rows.append({
            "raw_id": raw_id,
            "class_name": first.get("gt_class_name", ""),
            "loss_family": loss_family,
            "available_rows": len(group_sorted),
            "selected_rows": len(take),
            "rejected_over_cap_rows": max(0, len(group_sorted) - len(take)),
            "class_cap_used": cap,
            "policy": first.get("policy", ""),
            "certificate_type": first.get("certificate_type", ""),
            "clip_count": first.get("clip_count", ""),
            "instance_count": first.get("instance_count", ""),
            "is_person_raw_id": str(raw_id in person_ids),
            "is_hub_raw_id": str(raw_id in hub_ids),
        })

    # Stable manifest order: hard CE first, then soft, then prototype; within each, class-balanced order by class id.
    loss_order = {"hard_ce": 0, "soft_ce": 1, "prototype_calibration": 2}
    selected = sorted(selected, key=lambda r: (loss_order.get(str(r.get("loss_family")), 99), raw_norm(r.get("gt_raw_id")), int(r.get("selection_rank_within_class_loss", 999999))))
    for i, r in enumerate(selected):
        r["manifest_row_id"] = i

    stats = {
        "available_trainable_rows": sum(len(v) for v in grouped.values()),
        "group_count": len(grouped),
    }
    return selected, by_class_rows, {"rejected_rows": rejected, **stats}


def summarize(rows: List[Dict[str, object]], by_class_rows: List[Dict[str, object]], rejected_rows: List[Dict[str, object]], args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> Dict[str, object]:
    loss_counts = Counter(str(r.get("loss_family", "")) for r in rows)
    selected_by_seed = Counter(str(r.get("seed_type", "")) for r in rows)
    selected_by_raw = Counter(raw_norm(r.get("gt_raw_id")) for r in rows)
    selected_by_class_name = {}
    for r in rows:
        rid = raw_norm(r.get("gt_raw_id"))
        if rid not in selected_by_class_name:
            selected_by_class_name[rid] = str(r.get("gt_class_name", ""))

    hard_rows = [r for r in rows if r.get("loss_family") == "hard_ce"]
    hard_total = len(hard_rows)
    person_hard = sum(1 for r in hard_rows if raw_norm(r.get("gt_raw_id")) in person_ids)
    hub_hard = sum(1 for r in hard_rows if raw_norm(r.get("gt_raw_id")) in hub_ids)
    nonhub_hard = hard_total - hub_hard
    hard_classes = {raw_norm(r.get("gt_raw_id")) for r in hard_rows}
    nonhub_hard_classes = {raw_norm(r.get("gt_raw_id")) for r in hard_rows if raw_norm(r.get("gt_raw_id")) not in hub_ids}

    top_classes = []
    for rid, cnt in selected_by_raw.most_common(30):
        top_classes.append({"raw_id": rid, "class_name": selected_by_class_name.get(rid, ""), "selected_rows": cnt})

    gate_checks = []
    person_share = (person_hard / hard_total) if hard_total else 0.0
    hub_share = (hub_hard / hard_total) if hard_total else 0.0
    gate_checks.append({
        "name": "person_hard_share_cap",
        "status": "PASS" if person_share <= args.max_person_hard_share + 1e-12 else "FAIL",
        "got": person_share,
        "expected_max": args.max_person_hard_share,
        "hard": True,
    })
    gate_checks.append({
        "name": "hub_hard_share_cap",
        "status": "PASS" if hub_share <= args.max_hub_hard_share + 1e-12 else "FAIL",
        "got": hub_share,
        "expected_max": args.max_hub_hard_share,
        "hard": True,
    })
    gate_checks.append({
        "name": "min_selected_hard_classes",
        "status": "PASS" if len(hard_classes) >= args.min_selected_hard_classes else "WARN",
        "got": len(hard_classes),
        "expected_min": args.min_selected_hard_classes,
        "hard": False,
    })
    gate_checks.append({
        "name": "min_selected_nonhub_hard_rows",
        "status": "PASS" if nonhub_hard >= args.min_selected_nonhub_hard_rows else "WARN",
        "got": nonhub_hard,
        "expected_min": args.min_selected_nonhub_hard_rows,
        "hard": False,
    })
    gate_checks.append({
        "name": "no_deferred_selected",
        "status": "PASS" if all(str(r.get("seed_type")) != "deferred" for r in rows) else "FAIL",
        "bad_count": sum(1 for r in rows if str(r.get("seed_type")) == "deferred"),
        "hard": True,
    })

    status = "PASS" if all(c.get("status") != "FAIL" for c in gate_checks if c.get("hard")) else "FAIL_GATE"

    return {
        "status": status,
        "selected_rows": len(rows),
        "selected_loss_counts": dict(loss_counts),
        "selected_seed_counts": dict(selected_by_seed),
        "selected_class_loss_groups": len(by_class_rows),
        "selected_hard_rows": hard_total,
        "selected_hard_classes": len(hard_classes),
        "selected_nonhub_hard_classes": len(nonhub_hard_classes),
        "person_hard_rows": person_hard,
        "person_hard_share": person_share,
        "hub_hard_rows": hub_hard,
        "hub_hard_share": hub_share,
        "nonhub_hard_rows": nonhub_hard,
        "rejected_rows": len(rejected_rows),
        "rejected_reason_counts": dict(Counter(str(r.get("reject_reason", "")) for r in rejected_rows)),
        "top_selected_classes": top_classes,
        "gate_checks": gate_checks,
        "args": vars(args),
        "interpretation": {
            "balanced_manifest": "Class-capped trainable rows only; deferred rows are never selected.",
            "hard_ce": "Use for strong row-level seeds only, after class caps and hub/person caps.",
            "soft_ce": "Use for uncertain row-level seeds; do not convert to hard labels.",
            "prototype_calibration": "Use for prototype updates only; do not backprop as hard CE.",
            "sample_weight": "Suggested weight if the future trainer supports row weights; caps are the primary balancing mechanism.",
            "gpu_required": False,
            "rowwise_large_matrix_compute": False,
        },
    }


def main() -> None:
    args = parse_args()
    seed_pool, out_dir = ensure_path_args(args)
    person_ids = parse_id_set(args.person_raw_ids)
    hub_ids = parse_id_set(args.hub_raw_ids) | person_ids

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = read_csv(seed_pool)
    except Exception as e:
        fail = {
            "status": "FAIL_MISSING_OR_BAD_SEED_POOL",
            "seed_pool_csv": str(seed_pool),
            "error": repr(e),
        }
        (out_dir / "balanced_manifest_summary.json").write_text(json.dumps(fail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    selected, by_class_rows, aux = select_rows(rows, args, person_ids, hub_ids)
    rejected = aux["rejected_rows"]
    summary = summarize(selected, by_class_rows, rejected, args, person_ids, hub_ids)
    summary.update({
        "seed_pool_csv": str(seed_pool),
        "out_dir": str(out_dir),
        "input_rows": len(rows),
        "available_trainable_rows": aux.get("available_trainable_rows", 0),
        "person_raw_ids": sorted(person_ids),
        "hub_raw_ids": sorted(hub_ids),
    })

    manifest_fields = [
        "manifest_row_id", "manifest_use", "loss_family", "sample_weight", "selection_reason",
        "selection_rank_within_class_loss", "class_cap_used", "clip_id", "trajectory_id",
        "gt_raw_id", "gt_class_name", "policy", "seed_type", "certificate_type", "clip_count",
        "instance_count", "oracle_top1_is_gt", "oracle_gt_rank", "weak_base_top1_is_gt",
        "weak_base_top1_class_name", "weak_base_gt_rank", "weak_base_error_type",
        "weak_nohub_top1_is_gt", "weak_nohub_top1_class_name", "weak_nohub_gt_rank",
        "weak_nohub_error_type", "nohub_rescued_baseline_wrong", "oracle_correct_weak_base_wrong",
        "clip_y_size", "is_person_raw_id", "is_hub_raw_id",
    ]
    write_csv(out_dir / "balanced_training_manifest.csv", selected, manifest_fields)

    by_class_fields = [
        "raw_id", "class_name", "loss_family", "available_rows", "selected_rows",
        "rejected_over_cap_rows", "class_cap_used", "policy", "certificate_type",
        "clip_count", "instance_count", "is_person_raw_id", "is_hub_raw_id",
    ]
    write_csv(out_dir / "balanced_training_manifest_by_class.csv", by_class_rows, by_class_fields)

    rejected_fields = sorted({k for r in rejected for k in r.keys()}) if rejected else ["empty"]
    write_csv(out_dir / "balanced_training_manifest_rejected_rows.csv", rejected, rejected_fields)

    (out_dir / "balanced_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Residual-Gated Balanced Training Manifest",
        "",
        f"- status: {summary['status']}",
        f"- input_rows: {summary['input_rows']}",
        f"- selected_rows: {summary['selected_rows']}",
        f"- selected_loss_counts: {summary['selected_loss_counts']}",
        f"- selected_hard_rows: {summary['selected_hard_rows']}",
        f"- selected_hard_classes: {summary['selected_hard_classes']}",
        f"- person_hard_rows/share: {summary['person_hard_rows']} / {summary['person_hard_share']:.4f}",
        f"- hub_hard_rows/share: {summary['hub_hard_rows']} / {summary['hub_hard_share']:.4f}",
        f"- rejected_rows: {summary['rejected_rows']}",
        "",
        "## Gate checks",
    ]
    for c in summary["gate_checks"]:
        if "got" in c:
            md.append(f"- {c['name']}: {c['status']} got={c.get('got')} expected={c.get('expected_max', c.get('expected_min', ''))}")
        else:
            md.append(f"- {c['name']}: {c['status']} bad_count={c.get('bad_count')}")
    md += [
        "",
        "## Outputs",
        "- balanced_training_manifest.csv",
        "- balanced_training_manifest_by_class.csv",
        "- balanced_training_manifest_rejected_rows.csv",
        "- balanced_manifest_summary.json",
        "",
        "## Notes",
        "- This is a sampling/manifest planner only; it does not train and does not modify control-plane files.",
        "- Deferred rows are intentionally excluded from the manifest.",
        "- Hard CE rows are capped per class and additionally capped for person/hub raw ids.",
    ]
    (out_dir / "RESIDUAL_GATED_BALANCED_MANIFEST_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
