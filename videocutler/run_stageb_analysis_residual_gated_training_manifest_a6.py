#!/usr/bin/env python3
"""
Build an A6 selective soft-preserve training manifest from a residual-gated row seed pool.

A6 is designed after A4/A5 showed a real rescue/forgetting trade-off:
  * A4: strong soft rescue, but still non-trivial old-nohub-correct forgetting;
  * A5: stronger preservation reduced forgetting, but collapsed soft rescue.

A6 keeps soft CE as the dominant rescue signal and uses only selective, high-risk
old-nohub-correct rows as lightweight preservation anchors. It is read-only and
writes lightweight CSV/JSON/MD artifacts only.

Selected loss families:
  * soft_ce_seed -> soft_ce
  * high-risk hard_ce_seed with weak_nohub_top1_is_gt -> preservation_anchor
  * prototype_seed -> excluded, audited only
  * deferred -> excluded

High-risk preservation rows are old-nohub-correct rows that are likely near a
positive-set decision boundary, e.g. weak_base was wrong, nohub rescued base, or
the clip has multiple positive labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", type=str, default="")
    p.add_argument("--dataset_name", type=str, default="lvvis_train_base")
    p.add_argument("--seed_pool_csv", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--soft_cap_per_class", type=int, default=64)
    p.add_argument("--preservation_cap_per_class", type=int, default=24)
    p.add_argument("--person_preservation_cap", type=int, default=12)
    p.add_argument("--hub_preservation_cap", type=int, default=12)
    p.add_argument("--person_raw_ids", type=str, default="773")
    p.add_argument("--hub_raw_ids", type=str, default="773")

    p.add_argument("--soft_sample_weight", type=float, default=1.0)
    p.add_argument("--preservation_sample_weight", type=float, default=0.75)
    p.add_argument("--hub_sample_weight", type=float, default=0.5)

    p.add_argument("--selective_preservation", action="store_true", default=True)
    p.add_argument("--no_selective_preservation", dest="selective_preservation", action="store_false")
    p.add_argument("--include_multilabel_as_high_risk", action="store_true", default=True)
    p.add_argument("--no_include_multilabel_as_high_risk", dest="include_multilabel_as_high_risk", action="store_false")
    p.add_argument("--prefer_multilabel", action="store_true", default=True)
    p.add_argument("--no_prefer_multilabel", dest="prefer_multilabel", action="store_false")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--shuffle_ties", action="store_true")

    p.add_argument("--min_selected_soft_rows", type=int, default=256)
    p.add_argument("--min_selected_preservation_rows", type=int, default=256)
    p.add_argument("--max_person_preservation_share", type=float, default=0.20)
    p.add_argument("--max_hub_preservation_share", type=float, default=0.25)
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
        out_dir = Path(args.run_root) / "analysis" / "residual_gated_training_manifest_a6" / args.dataset_name
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
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


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


def preserve_risk_kind(row: Dict[str, str], args: argparse.Namespace) -> str:
    base_ok = truth(row.get("weak_base_top1_is_gt"))
    nohub_rescued = truth(row.get("nohub_rescued_baseline_wrong"))
    clip_y = fnum(row.get("clip_y_size"), 0.0)
    if nohub_rescued:
        return "nohub_rescued_base_wrong"
    if not base_ok:
        return "weak_base_wrong_nohub_correct"
    if bool(args.include_multilabel_as_high_risk) and clip_y > 1:
        return "multilabel_old_correct"
    return "low_risk_old_correct"


def is_high_risk_preserve(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if not truth(row.get("weak_nohub_top1_is_gt")):
        return False
    if not bool(args.selective_preservation):
        return True
    return preserve_risk_kind(row, args) != "low_risk_old_correct"


def make_soft_priority(row: Dict[str, str], prefer_multilabel: bool) -> Tuple:
    clip_y = fnum(row.get("clip_y_size"), 0.0)
    multilabel_key = 0 if (clip_y > 1 and prefer_multilabel) else 1
    oracle_correct_weak_base_wrong = truth(row.get("oracle_correct_weak_base_wrong"))
    nohub_rescued = truth(row.get("nohub_rescued_baseline_wrong"))
    nohub_rank = fnum(row.get("weak_nohub_gt_rank"), 999999.0)
    base_rank = fnum(row.get("weak_base_gt_rank"), 999999.0)
    oracle_rank = fnum(row.get("oracle_gt_rank"), 999999.0)
    return (
        0 if oracle_correct_weak_base_wrong else 1,
        0 if nohub_rescued else 1,
        multilabel_key,
        nohub_rank,
        base_rank,
        oracle_rank,
        str(row.get("clip_id", "")),
        str(row.get("trajectory_id", "")),
    )


def make_preserve_priority(row: Dict[str, str], prefer_multilabel: bool, args: argparse.Namespace) -> Tuple:
    clip_y = fnum(row.get("clip_y_size"), 0.0)
    risk = preserve_risk_kind(row, args)
    risk_order = {
        "nohub_rescued_base_wrong": 0,
        "weak_base_wrong_nohub_correct": 1,
        "multilabel_old_correct": 2,
        "low_risk_old_correct": 9,
    }.get(risk, 8)
    multilabel_key = 0 if (clip_y > 1 and prefer_multilabel) else 1
    nohub_rank = fnum(row.get("weak_nohub_gt_rank"), 999999.0)
    base_rank = fnum(row.get("weak_base_gt_rank"), 999999.0)
    return (
        risk_order,
        multilabel_key,
        nohub_rank,
        base_rank,
        str(row.get("clip_id", "")),
        str(row.get("trajectory_id", "")),
    )


def cap_for(raw_id: str, loss_family: str, args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> int:
    if loss_family == "soft_ce":
        return max(0, int(args.soft_cap_per_class))
    if loss_family == "preservation_anchor":
        if raw_id in person_ids:
            return max(0, int(args.person_preservation_cap))
        if raw_id in hub_ids:
            return max(0, int(args.hub_preservation_cap))
        return max(0, int(args.preservation_cap_per_class))
    return 0


def sample_weight_for(raw_id: str, loss_family: str, args: argparse.Namespace, hub_ids: set[str]) -> float:
    if raw_id in hub_ids:
        return float(args.hub_sample_weight)
    if loss_family == "soft_ce":
        return float(args.soft_sample_weight)
    if loss_family == "preservation_anchor":
        return float(args.preservation_sample_weight)
    return 1.0


def select_rows(rows: List[Dict[str, str]], args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(int(args.seed))
    soft_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    preserve_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    rejected: List[Dict[str, object]] = []

    for r in rows:
        seed_type = str(r.get("seed_type", "")).strip()
        raw_id = raw_norm(r.get("gt_raw_id"))
        if seed_type == "soft_ce_seed":
            soft_grouped[raw_id].append(r)
        elif seed_type == "hard_ce_seed" and truth(r.get("weak_nohub_top1_is_gt")):
            if is_high_risk_preserve(r, args):
                preserve_grouped[raw_id].append(r)
            else:
                rr = dict(r)
                rr.update({"reject_reason": "low_risk_old_correct_not_selected_in_a6", "loss_family": "preservation_anchor", "preservation_risk_kind": preserve_risk_kind(r, args)})
                rejected.append(rr)
        else:
            rr = dict(r)
            if seed_type == "prototype_seed":
                reason = "prototype_excluded_in_a6"
            elif seed_type == "hard_ce_seed":
                reason = "hard_seed_not_nohub_correct_or_unused"
            elif seed_type == "deferred":
                reason = "seed_type_not_trainable"
            else:
                reason = "seed_type_not_selected_in_a6"
            rr.update({"reject_reason": reason, "loss_family": ""})
            rejected.append(rr)

    selected: List[Dict[str, object]] = []
    by_class_rows: List[Dict[str, object]] = []

    def emit_group(grouped: Dict[str, List[Dict[str, str]]], loss_family: str) -> None:
        nonlocal selected, rejected, by_class_rows
        for raw_id, group in sorted(grouped.items(), key=lambda kv: kv[0]):
            if args.shuffle_ties:
                rng.shuffle(group)
            if loss_family == "soft_ce":
                group_sorted = sorted(group, key=lambda r: make_soft_priority(r, bool(args.prefer_multilabel)))
            else:
                group_sorted = sorted(group, key=lambda r: make_preserve_priority(r, bool(args.prefer_multilabel), args))
            cap = cap_for(raw_id, loss_family, args, person_ids, hub_ids)
            take = group_sorted[:cap]
            rest = group_sorted[cap:]
            for rank, r in enumerate(take):
                out = dict(r)
                risk_kind = preserve_risk_kind(r, args) if loss_family == "preservation_anchor" else ""
                out.update({
                    "manifest_use": "train",
                    "loss_family": loss_family,
                    "selection_rank_within_class_loss": rank + 1,
                    "class_cap_used": cap,
                    "is_person_raw_id": str(raw_id in person_ids),
                    "is_hub_raw_id": str(raw_id in hub_ids),
                    "sample_weight": sample_weight_for(raw_id, loss_family, args, hub_ids),
                    "preservation_risk_kind": risk_kind,
                    "selection_reason": f"a6_selected_by_{loss_family}_cap" if loss_family == "soft_ce" else f"a6_selected_selective_preservation_{risk_kind}",
                })
                selected.append(out)
            for r in rest:
                rr = dict(r)
                rr.update({
                    "reject_reason": f"over_class_cap_{loss_family}",
                    "loss_family": loss_family,
                    "class_cap_used": cap,
                    "preservation_risk_kind": preserve_risk_kind(r, args) if loss_family == "preservation_anchor" else "",
                })
                rejected.append(rr)
            first = group_sorted[0] if group_sorted else {}
            if group_sorted:
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

    emit_group(soft_grouped, "soft_ce")
    emit_group(preserve_grouped, "preservation_anchor")

    order = {"soft_ce": 0, "preservation_anchor": 1}
    selected = sorted(selected, key=lambda r: (order.get(str(r.get("loss_family")), 99), raw_norm(r.get("gt_raw_id")), int(r.get("selection_rank_within_class_loss", 999999))))
    for i, row in enumerate(selected):
        row["manifest_row_id"] = i
    return selected, by_class_rows, rejected


def summarize(selected: List[Dict[str, object]], by_class_rows: List[Dict[str, object]], rejected: List[Dict[str, object]], args: argparse.Namespace, person_ids: set[str], hub_ids: set[str]) -> Dict[str, object]:
    loss_counts = Counter(str(r.get("loss_family", "")) for r in selected)
    seed_counts = Counter(str(r.get("seed_type", "")) for r in selected)
    risk_counts = Counter(str(r.get("preservation_risk_kind", "")) for r in selected if r.get("loss_family") == "preservation_anchor")
    preserve_rows = [r for r in selected if r.get("loss_family") == "preservation_anchor"]
    soft_rows = [r for r in selected if r.get("loss_family") == "soft_ce"]
    person_preserve = sum(1 for r in preserve_rows if raw_norm(r.get("gt_raw_id")) in person_ids)
    hub_preserve = sum(1 for r in preserve_rows if raw_norm(r.get("gt_raw_id")) in hub_ids)
    preserve_total = len(preserve_rows)
    person_share = person_preserve / preserve_total if preserve_total else 0.0
    hub_share = hub_preserve / preserve_total if preserve_total else 0.0
    selected_by_raw = Counter(raw_norm(r.get("gt_raw_id")) for r in selected)
    selected_name: Dict[str, str] = {}
    for r in selected:
        selected_name.setdefault(raw_norm(r.get("gt_raw_id")), str(r.get("gt_class_name", "")))
    top_classes = [{"raw_id": rid, "class_name": selected_name.get(rid, ""), "selected_rows": cnt} for rid, cnt in selected_by_raw.most_common(30)]

    gates = [
        {"name": "min_selected_soft_rows", "status": "PASS" if len(soft_rows) >= int(args.min_selected_soft_rows) else "WARN", "got": len(soft_rows), "expected_min": int(args.min_selected_soft_rows), "hard": False},
        {"name": "min_selected_preservation_rows", "status": "PASS" if preserve_total >= int(args.min_selected_preservation_rows) else "WARN", "got": preserve_total, "expected_min": int(args.min_selected_preservation_rows), "hard": False},
        {"name": "person_preservation_share_cap", "status": "PASS" if person_share <= float(args.max_person_preservation_share) + 1e-12 else "FAIL", "got": person_share, "expected_max": float(args.max_person_preservation_share), "hard": True},
        {"name": "hub_preservation_share_cap", "status": "PASS" if hub_share <= float(args.max_hub_preservation_share) + 1e-12 else "FAIL", "got": hub_share, "expected_max": float(args.max_hub_preservation_share), "hard": True},
        {"name": "no_hard_ce_selected", "status": "PASS" if loss_counts.get("hard_ce", 0) == 0 else "FAIL", "got": loss_counts.get("hard_ce", 0), "expected": 0, "hard": True},
        {"name": "no_prototype_selected", "status": "PASS" if loss_counts.get("prototype_calibration", 0) == 0 else "FAIL", "got": loss_counts.get("prototype_calibration", 0), "expected": 0, "hard": True},
        {"name": "no_deferred_selected", "status": "PASS" if all(str(r.get("seed_type")) != "deferred" for r in selected) else "FAIL", "bad_count": sum(1 for r in selected if str(r.get("seed_type")) == "deferred"), "hard": True},
    ]
    status = "PASS" if all(g.get("status") != "FAIL" for g in gates if g.get("hard")) else "FAIL_GATE"
    return {
        "status": status,
        "selected_rows": len(selected),
        "selected_loss_counts": dict(loss_counts),
        "selected_seed_counts": dict(seed_counts),
        "selected_preservation_risk_counts": dict(risk_counts),
        "selected_class_loss_groups": len(by_class_rows),
        "selected_soft_rows": len(soft_rows),
        "selected_preservation_rows": preserve_total,
        "person_preservation_rows": person_preserve,
        "person_preservation_share": person_share,
        "hub_preservation_rows": hub_preserve,
        "hub_preservation_share": hub_share,
        "rejected_rows": len(rejected),
        "rejected_reason_counts": dict(Counter(str(r.get("reject_reason", "")) for r in rejected)),
        "top_selected_classes": top_classes,
        "gate_checks": gates,
        "args": vars(args),
        "interpretation": {
            "a6_policy": "Soft CE is the dominant positive-set rescue loss. Preservation anchors are selected only from high-risk old-nohub-correct rows and are lightweight. Prototype/deferred rows are excluded.",
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
        fail = {"status": "FAIL_MISSING_OR_BAD_SEED_POOL", "seed_pool_csv": str(seed_pool), "error": repr(e)}
        (out_dir / "balanced_manifest_summary.json").write_text(json.dumps(fail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    selected, by_class_rows, rejected = select_rows(rows, args, person_ids, hub_ids)
    summary = summarize(selected, by_class_rows, rejected, args, person_ids, hub_ids)
    summary.update({"seed_pool_csv": str(seed_pool), "out_dir": str(out_dir), "input_rows": len(rows), "person_raw_ids": sorted(person_ids), "hub_raw_ids": sorted(hub_ids)})

    manifest_fields = [
        "manifest_row_id", "manifest_use", "loss_family", "sample_weight", "selection_reason", "preservation_risk_kind",
        "selection_rank_within_class_loss", "class_cap_used", "clip_id", "trajectory_id",
        "gt_raw_id", "gt_class_name", "policy", "seed_type", "certificate_type", "clip_count",
        "instance_count", "oracle_top1_is_gt", "oracle_gt_rank", "weak_base_top1_is_gt",
        "weak_base_top1_class_name", "weak_base_gt_rank", "weak_base_error_type",
        "weak_nohub_top1_is_gt", "weak_nohub_top1_class_name", "weak_nohub_gt_rank",
        "weak_nohub_error_type", "nohub_rescued_baseline_wrong", "oracle_correct_weak_base_wrong",
        "clip_y_size", "is_person_raw_id", "is_hub_raw_id",
    ]
    write_csv(out_dir / "balanced_training_manifest.csv", selected, manifest_fields)
    by_class_fields = ["raw_id", "class_name", "loss_family", "available_rows", "selected_rows", "rejected_over_cap_rows", "class_cap_used", "policy", "certificate_type", "clip_count", "instance_count", "is_person_raw_id", "is_hub_raw_id"]
    write_csv(out_dir / "balanced_training_manifest_by_class.csv", by_class_rows, by_class_fields)
    rejected_fields = sorted({k for r in rejected for k in r.keys()}) if rejected else ["empty"]
    write_csv(out_dir / "balanced_training_manifest_rejected_rows.csv", rejected, rejected_fields)
    (out_dir / "balanced_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Residual-Gated A6 Selective Soft-Preserve Manifest", "",
        f"- status: {summary['status']}",
        f"- input_rows: {summary['input_rows']}",
        f"- selected_rows: {summary['selected_rows']}",
        f"- selected_loss_counts: {summary['selected_loss_counts']}",
        f"- selected_preservation_risk_counts: {summary['selected_preservation_risk_counts']}",
        f"- selected_soft_rows: {summary['selected_soft_rows']}",
        f"- selected_preservation_rows: {summary['selected_preservation_rows']}",
        f"- person_preservation_rows/share: {summary['person_preservation_rows']} / {summary['person_preservation_share']:.4f}",
        f"- hub_preservation_rows/share: {summary['hub_preservation_rows']} / {summary['hub_preservation_share']:.4f}",
        f"- rejected_rows: {summary['rejected_rows']}", "", "## Gate checks",
    ]
    for c in summary["gate_checks"]:
        if "got" in c:
            md.append(f"- {c['name']}: {c['status']} got={c.get('got')} expected={c.get('expected_max', c.get('expected_min', c.get('expected', '')))}")
        else:
            md.append(f"- {c['name']}: {c['status']} bad_count={c.get('bad_count')}")
    md += [
        "", "## Outputs", "- balanced_training_manifest.csv", "- balanced_training_manifest_by_class.csv", "- balanced_training_manifest_rejected_rows.csv", "- balanced_manifest_summary.json", "",
        "## Notes", "- A6 selects soft_ce + selective lightweight preservation_anchor only.", "- No hard_ce rows are selected.", "- Prototype rows are intentionally excluded from training.", "- Deferred rows are intentionally excluded from the manifest.",
    ]
    (out_dir / "RESIDUAL_GATED_A6_MANIFEST_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
