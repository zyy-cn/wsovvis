#!/usr/bin/env python3
"""A8 Anchor-first drift audit.

Read-only audit over existing true-margin exports/checkpoints. It splits rows by
strict cross-clip/context-identifiable anchor classes and measures whether CE
post-epoch-5 drift primarily corrupts anchor or non-anchor classes, and whether
Hungarian pseudo assignments amplify hub / pseudo-label errors.

This script does NOT train, does NOT modify checkpoints, and does NOT use GT for
method training. It is an audit-only diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def as_float(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def as_int(x, default=0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def truthy(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "pass"}


def first_present(row: dict, names: Sequence[str], default: str = "") -> str:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def norm_id(x: object) -> str:
    s = str(x if x is not None else "").strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def row_key(row: dict) -> Tuple[str, str]:
    clip = first_present(row, ["clip_id", "video_id", "video", "clip"])
    traj = first_present(row, ["trajectory_id", "traj_id", "track_id", "trajectory_key", "row_id"])
    return (clip, traj)


def infer_anchor_from_row(row: dict) -> bool:
    """Heuristic for context-identifiable anchor rows across possible schemas."""
    text_blob = " ".join(str(v).lower() for v in row.values() if v is not None)

    # Strong explicit booleans.
    for k in [
        "is_anchor",
        "anchor",
        "context_identifiable",
        "strict_context_identifiable",
        "cross_clip_identifiable",
        "is_context_identifiable",
        "is_cross_clip_identifiable",
        "certified_context_identifiable",
        "strict_anchor",
    ]:
        if k in row:
            val = str(row.get(k, "")).strip().lower()
            if val in {"1", "true", "yes", "y", "pass"}:
                return True
            if val in {"0", "false", "no", "n", "fail"}:
                # do not early return; another column may carry the bucket
                pass

    # Bucket/status strings.
    for k in [
        "bucket",
        "initial_bucket",
        "status",
        "certification",
        "certificate_type",
        "identifiability_bucket",
        "context_bucket",
        "class_bucket",
    ]:
        v = str(row.get(k, "")).strip().lower()
        if not v:
            continue
        if "context-identifiable" in v or "context_identifiable" in v:
            if "not" not in v and "non" not in v and "entangled" not in v and "low" not in v:
                return True
        if "cross_clip_identifiable" in v or "cross-clip identifiable" in v:
            return True
        if "strict" in v and "anchor" in v:
            return True

    # Numeric context intersection certificate.
    # This covers CSVs that store intersection size / base clip count without explicit bucket labels.
    intersection = None
    for k in ["intersection_size", "context_intersection_size", "gt_context_intersection_size"]:
        if k in row:
            intersection = as_int(row.get(k), default=-1)
            break
    clip_count = None
    for k in ["base_clip_count", "clip_count", "train_clip_count", "video_count", "base_video_count"]:
        if k in row:
            clip_count = as_int(row.get(k), default=-1)
            break
    if intersection == 1 and (clip_count is None or clip_count >= 3):
        return True

    # Last-resort textual match, but avoid residual-only certificates.
    if "context-identifiable anchor" in text_blob or "strict context-identifiable" in text_blob:
        return True
    return False


def load_anchor_set(anchor_csv: Path, anchor_raw_ids: str = "") -> Tuple[set, Dict[str, str], dict]:
    anchors = set()
    names: Dict[str, str] = {}
    counters = Counter()

    if anchor_raw_ids.strip():
        for tok in anchor_raw_ids.split(","):
            rid = norm_id(tok)
            if rid:
                anchors.add(rid)
                counters["manual_anchor_ids"] += 1

    if anchor_csv:
        if not anchor_csv.exists():
            raise FileNotFoundError(f"anchor_csv not found: {anchor_csv}")
        rows = read_csv(anchor_csv)
        counters["anchor_csv_rows"] = len(rows)
        for r in rows:
            rid = norm_id(first_present(r, [
                "raw_category_id", "raw_id", "category_id", "base_raw_id", "class_raw_id", "id"
            ]))
            if not rid:
                counters["rows_without_raw_id"] += 1
                continue
            cname = first_present(r, ["class_name", "name", "category_name", "gt_class_name"], "")
            if cname:
                names[rid] = cname
            if infer_anchor_from_row(r):
                anchors.add(rid)
                counters["inferred_anchor_rows"] += 1
            else:
                counters["non_anchor_rows"] += 1

    meta = {
        "anchor_count": len(anchors),
        "counters": dict(counters),
        "anchor_csv": str(anchor_csv) if anchor_csv else "",
        "anchor_raw_ids_supplied": bool(anchor_raw_ids.strip()),
    }
    return anchors, names, meta


def load_matched_pairs(matched_csv: Path) -> Dict[Tuple[str, str], dict]:
    rows = read_csv(matched_csv)
    out = {}
    for r in rows:
        k = row_key(r)
        if k[0] and k[1]:
            out[k] = r
    return out


def extract_row_semantics(row: dict, matched_by_key: Dict[Tuple[str, str], dict]) -> dict:
    k = row_key(row)
    mr = matched_by_key.get(k, {})
    gt = norm_id(first_present(row, ["gt_raw_id", "audit_gt_raw_id", "raw_gt_id", "gt_category_id"]))
    top1 = norm_id(first_present(row, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id", "row_top1_raw_id_in_full_y"]))
    matched = norm_id(first_present(row, ["matched_raw_id", "pseudo_raw_id", "pseudo_label_raw_id", "oracle_boundary_original_target_raw_id"]))
    if not matched:
        matched = norm_id(first_present(mr, ["matched_raw_id", "pseudo_raw_id", "matched_category_id", "oracle_boundary_original_target_raw_id"]))
    gt_name = first_present(row, ["gt_class_name", "audit_gt_class_name", "gt_name"], "")
    top1_name = first_present(row, ["top1_class_name", "wrong_top1_class_name", "row_top1_class_name_in_full_y"], "")
    matched_name = first_present(row, ["matched_class_name", "pseudo_class_name"], "") or first_present(mr, ["matched_class_name", "pseudo_class_name"], "")

    # hit inference: explicit hit preferred; otherwise compare ids.
    hit_val = first_present(row, ["top1_hit", "gt_top1_hit", "is_top1_gt"], "")
    if hit_val != "":
        is_correct = truthy(hit_val) or as_int(hit_val, -1) == 1
    else:
        is_correct = bool(gt and top1 and gt == top1)

    wrong_abs_gap = as_float(first_present(row, ["wrong_abs_gap", "margin_abs_gap"]), 0.0)
    gt_score = as_float(first_present(row, ["gt_score", "score_gt", "gt_logit"]), math.nan)
    top1_score = as_float(first_present(row, ["top1_score", "score_top1", "top1_logit"]), math.nan)
    pseudo_score = as_float(first_present(row, ["pseudo_score", "matched_score", "score_pseudo"]), math.nan)

    # If matched_score only exists in matched CSV, use it for proxy diagnostics.
    if math.isnan(pseudo_score):
        pseudo_score = as_float(first_present(mr, ["matched_score", "score", "pseudo_score"]), math.nan)

    return {
        "key": "||".join(k),
        "clip_id": k[0],
        "trajectory_id": k[1],
        "gt_raw_id": gt,
        "gt_class_name": gt_name,
        "top1_raw_id": top1,
        "top1_class_name": top1_name,
        "matched_raw_id": matched,
        "matched_class_name": matched_name,
        "is_correct": is_correct,
        "wrong_abs_gap": wrong_abs_gap,
        "gt_score": gt_score,
        "top1_score": top1_score,
        "pseudo_score": pseudo_score,
    }


def margin_bucket(is_correct: bool, gap: float) -> str:
    if is_correct:
        return "correct"
    if gap <= 1.0:
        return "small_le_1"
    if gap >= 3.0:
        return "large_ge_3"
    return "middle_1_to_3"


def read_epoch_rows(drift_dir: Path, epoch: int, matched_by_key: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    csv_path = drift_dir / f"epoch_{epoch:03d}_true_margin" / "true_score_margin_row_audit.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing epoch row audit: {csv_path}")
    out = {}
    for row in read_csv(csv_path):
        sem = extract_row_semantics(row, matched_by_key)
        if sem["key"]:
            out[sem["key"]] = sem
    return out


def group_name(sem: dict, anchor_ids: set, hub_ids: set) -> str:
    gt = sem.get("gt_raw_id", "")
    matched = sem.get("matched_raw_id", "")
    if gt in anchor_ids:
        return "gt_anchor"
    if matched in anchor_ids:
        return "matched_anchor_gt_nonanchor"
    if matched in hub_ids:
        return "matched_hub_gt_nonanchor"
    return "gt_nonanchor"


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize_epoch(epoch: int, rows: Dict[str, dict], base_rows: Dict[str, dict], anchor_ids: set, hub_ids: set) -> Tuple[List[dict], List[dict], List[dict]]:
    groups = defaultdict(list)
    transitions = []
    causes = []

    for key, sem in rows.items():
        base = base_rows.get(key)
        g = group_name(sem, anchor_ids, hub_ids)
        sem["group"] = g
        groups[g].append(sem)
        if base:
            c2w = base["is_correct"] and not sem["is_correct"]
            w2c = (not base["is_correct"]) and sem["is_correct"]
            stable_c = base["is_correct"] and sem["is_correct"]
            stable_w = (not base["is_correct"]) and (not sem["is_correct"])
            pseudo_bad = bool(sem.get("matched_raw_id") and sem.get("gt_raw_id") and sem["matched_raw_id"] != sem["gt_raw_id"])
            top1_is_pseudo = bool(sem.get("top1_raw_id") and sem.get("matched_raw_id") and sem["top1_raw_id"] == sem["matched_raw_id"])
            top1_is_hub = sem.get("top1_raw_id") in hub_ids
            transition = "stable_correct" if stable_c else "stable_wrong" if stable_w else "correct_to_wrong" if c2w else "wrong_to_correct" if w2c else "unknown"
            transitions.append({
                "epoch": epoch,
                "key": key,
                "clip_id": sem["clip_id"],
                "trajectory_id": sem["trajectory_id"],
                "group": g,
                "transition": transition,
                "baseline_top1_raw_id": base.get("top1_raw_id", ""),
                "epoch_top1_raw_id": sem.get("top1_raw_id", ""),
                "gt_raw_id": sem.get("gt_raw_id", ""),
                "matched_raw_id": sem.get("matched_raw_id", ""),
                "matched_is_gt": int(not pseudo_bad) if sem.get("matched_raw_id") else "",
                "top1_is_pseudo": int(top1_is_pseudo),
                "top1_is_hub": int(top1_is_hub),
                "wrong_abs_gap": sem.get("wrong_abs_gap", 0.0),
                "baseline_wrong_abs_gap": base.get("wrong_abs_gap", 0.0),
            })
            if c2w:
                causes.append({
                    "epoch": epoch,
                    "group": g,
                    "gt_raw_id": sem.get("gt_raw_id", ""),
                    "gt_class_name": sem.get("gt_class_name", ""),
                    "matched_raw_id": sem.get("matched_raw_id", ""),
                    "matched_class_name": sem.get("matched_class_name", ""),
                    "new_top1_raw_id": sem.get("top1_raw_id", ""),
                    "new_top1_class_name": sem.get("top1_class_name", ""),
                    "pseudo_bad": int(pseudo_bad),
                    "new_top1_is_pseudo": int(top1_is_pseudo),
                    "new_top1_is_hub": int(top1_is_hub),
                    "wrong_abs_gap": sem.get("wrong_abs_gap", 0.0),
                })

    summary_rows = []
    all_group_names = sorted(set(groups.keys()) | {"gt_anchor", "matched_anchor_gt_nonanchor", "matched_hub_gt_nonanchor", "gt_nonanchor"})
    for g in all_group_names:
        items = groups.get(g, [])
        n = len(items)
        correct = sum(1 for x in items if x["is_correct"])
        wrong_items = [x for x in items if not x["is_correct"]]
        mb = Counter(margin_bucket(x["is_correct"], x.get("wrong_abs_gap", 0.0)) for x in items)
        pseudo_avail = [x for x in items if x.get("matched_raw_id")]
        pseudo_bad = [x for x in pseudo_avail if x.get("matched_raw_id") != x.get("gt_raw_id")]
        pseudo_wrong_top1 = [x for x in pseudo_bad if x.get("top1_raw_id") == x.get("matched_raw_id")]
        top1_hub_wrong = [x for x in wrong_items if x.get("top1_raw_id") in hub_ids]
        c2w = sum(1 for t in transitions if t["group"] == g and t["transition"] == "correct_to_wrong")
        w2c = sum(1 for t in transitions if t["group"] == g and t["transition"] == "wrong_to_correct")
        c2w_pseudo = sum(1 for t in transitions if t["group"] == g and t["transition"] == "correct_to_wrong" and t["top1_is_pseudo"] == 1 and t["matched_is_gt"] == 0)
        c2w_hub = sum(1 for t in transitions if t["group"] == g and t["transition"] == "correct_to_wrong" and t["top1_is_hub"] == 1)
        summary_rows.append({
            "epoch": epoch,
            "group": g,
            "rows": n,
            "correct": correct,
            "wrong": n - correct,
            "micro_top1": safe_div(correct, n),
            "small_wrong": mb.get("small_le_1", 0),
            "middle_wrong": mb.get("middle_1_to_3", 0),
            "large_wrong": mb.get("large_ge_3", 0),
            "top1_hub_wrong": len(top1_hub_wrong),
            "pseudo_label_available": len(pseudo_avail),
            "pseudo_mismatch": len(pseudo_bad),
            "pseudo_wrong_top1": len(pseudo_wrong_top1),
            "pseudo_wrong_top1_rate": safe_div(len(pseudo_wrong_top1), len(pseudo_avail)),
            "correct_to_wrong_vs_baseline": c2w,
            "wrong_to_correct_vs_baseline": w2c,
            "correct_to_wrong_pseudo_bad_top1": c2w_pseudo,
            "correct_to_wrong_to_hub": c2w_hub,
            "mean_wrong_abs_gap_wrong": safe_div(sum(x.get("wrong_abs_gap", 0.0) for x in wrong_items), len(wrong_items)),
        })
    return summary_rows, transitions, causes


def main() -> int:
    ap = argparse.ArgumentParser(description="A8 anchor-first drift audit")
    ap.add_argument("--drift_audit_dir", required=True, help="Existing CE drift onset audit dir containing epoch_XXX_true_margin dirs")
    ap.add_argument("--anchor_csv", required=True, help="per-class context-identifiability CSV")
    ap.add_argument("--anchor_raw_ids", default="", help="Optional comma-separated anchor raw ids to add/override")
    ap.add_argument("--matched_pairs_csv", required=True)
    ap.add_argument("--epochs", default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    ap.add_argument("--baseline_epoch", type=int, default=5)
    ap.add_argument("--hub_raw_ids", default="773,1112,63,936,931,970,173,580,135,1044,1114,868")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    drift_dir = Path(args.drift_audit_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    if args.baseline_epoch not in epochs:
        epochs = [args.baseline_epoch] + epochs
    hub_ids = {norm_id(x) for x in args.hub_raw_ids.split(",") if norm_id(x)}

    anchor_ids, anchor_names, anchor_meta = load_anchor_set(Path(args.anchor_csv), args.anchor_raw_ids)
    if not anchor_ids:
        raise RuntimeError("No anchor ids inferred. Check --anchor_csv schema or pass --anchor_raw_ids.")

    matched_by_key = load_matched_pairs(Path(args.matched_pairs_csv))
    base_rows = read_epoch_rows(drift_dir, args.baseline_epoch, matched_by_key)

    all_group_rows: List[dict] = []
    all_transitions: List[dict] = []
    all_causes: List[dict] = []
    for ep in epochs:
        rows = read_epoch_rows(drift_dir, ep, matched_by_key)
        group_rows, transitions, causes = summarize_epoch(ep, rows, base_rows, anchor_ids, hub_ids)
        all_group_rows.extend(group_rows)
        all_transitions.extend(transitions)
        all_causes.extend(causes)

    write_csv(out_dir / "anchor_epoch_group_drift_table.csv", all_group_rows)
    # Transition file can be large but lightweight enough; keep it row-level for exact drilldown.
    write_csv(out_dir / "anchor_row_transition_events.csv", all_transitions)
    write_csv(out_dir / "anchor_correct_to_wrong_causes.csv", all_causes)

    # Compact top aggregations.
    cause_counter = Counter()
    absorber_counter = Counter()
    gt_counter = Counter()
    for c in all_causes:
        key = (c["epoch"], c["group"], c["matched_raw_id"], c["new_top1_raw_id"], c["pseudo_bad"], c["new_top1_is_pseudo"], c["new_top1_is_hub"])
        cause_counter[key] += 1
        absorber_counter[(c["epoch"], c["group"], c["new_top1_raw_id"], c["new_top1_class_name"])] += 1
        gt_counter[(c["epoch"], c["group"], c["gt_raw_id"], c["gt_class_name"])] += 1
    top_cause_rows = [
        {
            "epoch": ep,
            "group": g,
            "matched_raw_id": m,
            "new_top1_raw_id": t,
            "pseudo_bad": pb,
            "new_top1_is_pseudo": tp,
            "new_top1_is_hub": th,
            "correct_to_wrong_rows": n,
        }
        for (ep, g, m, t, pb, tp, th), n in cause_counter.most_common()
    ]
    write_csv(out_dir / "top_correct_to_wrong_assignment_causes.csv", top_cause_rows)
    top_absorber_rows = [
        {"epoch": ep, "group": g, "new_top1_raw_id": tid, "new_top1_class_name": name, "correct_to_wrong_rows": n}
        for (ep, g, tid, name), n in absorber_counter.most_common()
    ]
    write_csv(out_dir / "top_correct_to_wrong_absorbers_by_group.csv", top_absorber_rows)
    top_gt_rows = [
        {"epoch": ep, "group": g, "gt_raw_id": gid, "gt_class_name": name, "correct_to_wrong_rows": n}
        for (ep, g, gid, name), n in gt_counter.most_common()
    ]
    write_csv(out_dir / "top_correct_to_wrong_gt_classes_by_group.csv", top_gt_rows)

    # Find first drift per group: first epoch after baseline where c2w>0 and micro lower than baseline group micro.
    by_group_epoch = {(r["group"], int(r["epoch"])): r for r in all_group_rows}
    groups = sorted({r["group"] for r in all_group_rows})
    first_drift = []
    for g in groups:
        base = by_group_epoch.get((g, args.baseline_epoch))
        if not base:
            continue
        base_micro = as_float(base["micro_top1"])
        for ep in sorted(e for e in epochs if e != args.baseline_epoch):
            r = by_group_epoch.get((g, ep))
            if not r:
                continue
            if as_float(r["micro_top1"]) < base_micro and as_int(r["correct_to_wrong_vs_baseline"]) > 0:
                first_drift.append({
                    "group": g,
                    "first_drift_epoch": ep,
                    "baseline_micro": base_micro,
                    "epoch_micro": as_float(r["micro_top1"]),
                    "correct_to_wrong_vs_baseline": as_int(r["correct_to_wrong_vs_baseline"]),
                    "correct_to_wrong_pseudo_bad_top1": as_int(r["correct_to_wrong_pseudo_bad_top1"]),
                    "correct_to_wrong_to_hub": as_int(r["correct_to_wrong_to_hub"]),
                })
                break
    write_csv(out_dir / "anchor_first_drift_onset_by_group.csv", first_drift)

    summary = {
        "status": "PASS",
        "drift_audit_dir": str(drift_dir),
        "matched_pairs_csv": args.matched_pairs_csv,
        "anchor_meta": anchor_meta,
        "anchor_count": len(anchor_ids),
        "hub_raw_ids": sorted(hub_ids),
        "epochs": sorted(epochs),
        "baseline_epoch": args.baseline_epoch,
        "first_drift_by_group": first_drift,
        "outputs": {
            "anchor_epoch_group_drift_table": str(out_dir / "anchor_epoch_group_drift_table.csv"),
            "anchor_row_transition_events": str(out_dir / "anchor_row_transition_events.csv"),
            "anchor_correct_to_wrong_causes": str(out_dir / "anchor_correct_to_wrong_causes.csv"),
            "top_correct_to_wrong_assignment_causes": str(out_dir / "top_correct_to_wrong_assignment_causes.csv"),
            "top_correct_to_wrong_absorbers_by_group": str(out_dir / "top_correct_to_wrong_absorbers_by_group.csv"),
            "top_correct_to_wrong_gt_classes_by_group": str(out_dir / "top_correct_to_wrong_gt_classes_by_group.csv"),
            "markdown": str(out_dir / "A8_ANCHOR_FIRST_DRIFT_AUDIT.md"),
        },
    }
    (out_dir / "anchor_first_drift_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report.
    md = []
    md.append("# A8 Anchor-first Drift Audit")
    md.append("")
    md.append(f"- drift_audit_dir: `{drift_dir}`")
    md.append(f"- baseline_epoch: `{args.baseline_epoch}`")
    md.append(f"- inferred_anchor_count: `{len(anchor_ids)}`")
    md.append(f"- anchor_csv: `{args.anchor_csv}`")
    md.append("")
    md.append("## First drift by group")
    if first_drift:
        md.append("| group | first_epoch | baseline_micro | epoch_micro | c2w | c2w_pseudo_bad_top1 | c2w_to_hub |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in first_drift:
            md.append(f"| {r['group']} | {r['first_drift_epoch']} | {r['baseline_micro']:.6f} | {r['epoch_micro']:.6f} | {r['correct_to_wrong_vs_baseline']} | {r['correct_to_wrong_pseudo_bad_top1']} | {r['correct_to_wrong_to_hub']} |")
    else:
        md.append("No group-level drift found under current criterion.")
    md.append("")
    md.append("## Group epoch table")
    md.append("| epoch | group | micro | wrong | large | middle | small | c2w | w2c | pseudo_wrong_top1 | c2w_pseudo_bad | c2w_to_hub |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_group_rows:
        md.append(f"| {r['epoch']} | {r['group']} | {as_float(r['micro_top1']):.6f} | {r['wrong']} | {r['large_wrong']} | {r['middle_wrong']} | {r['small_wrong']} | {r['correct_to_wrong_vs_baseline']} | {r['wrong_to_correct_vs_baseline']} | {r['pseudo_wrong_top1']} | {r['correct_to_wrong_pseudo_bad_top1']} | {r['correct_to_wrong_to_hub']} |")
    md.append("")
    md.append("## Interpretation guide")
    md.append("- If `gt_anchor` drifts at 5->10, CE is corrupting even the safest anchor classes.")
    md.append("- If `gt_anchor` is stable but `gt_nonanchor` drifts, anchor-first curriculum is plausible but expansion needs a drift gate.")
    md.append("- If `correct_to_wrong_pseudo_bad_top1` is high, drift is directly consistent with pseudo-label overfit.")
    md.append("- If `correct_to_wrong_to_hub` is high, drift is hub expansion rather than random noise.")
    (out_dir / "A8_ANCHOR_FIRST_DRIFT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
