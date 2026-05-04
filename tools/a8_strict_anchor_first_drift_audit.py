#!/usr/bin/env python3
"""
Strict-anchor first drift audit for A8 CE drift-onset results.

This is read-only. It reuses per-epoch true-margin CSVs produced by
A8 CE drift-onset audit and compares strict context-identifiable anchors
against non-strict anchors.

Design goals:
- no training
- no evaluator changes
- no Hungarian changes
- robust CSV schema discovery
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def sget(row: dict, names: Sequence[str], default: str = "") -> str:
    for n in names:
        if n in row and str(row.get(n, "")).strip() != "":
            return str(row.get(n, "")).strip()
    return default


def fnum(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def inum(x, default=0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def bval(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "pass"}


def find_raw_id_col(rows: List[dict]) -> str:
    if not rows:
        raise SystemExit("empty anchor csv")
    cols = list(rows[0].keys())
    priorities = [
        "raw_category_id", "raw_id", "category_id", "cat_id", "class_raw_id", "lvvis_raw_id",
        "gt_raw_id", "matched_raw_id",
    ]
    for c in priorities:
        if c in cols:
            return c
    # fallback: any column that looks raw-id-ish and has many numeric values
    candidates = []
    for c in cols:
        lc = c.lower()
        if "id" in lc and ("raw" in lc or "category" in lc or "class" in lc):
            numeric = sum(1 for r in rows if str(r.get(c, "")).strip().replace(".", "", 1).isdigit())
            candidates.append((numeric, c))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]
    raise SystemExit(f"Cannot infer raw id column from columns: {cols}")


def discover_anchor_candidates(rows: List[dict], raw_col: str) -> List[dict]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    out = []
    for c in cols:
        lc = c.lower()
        if c == raw_col:
            continue
        if not any(k in lc for k in ["strict", "unique", "ident", "anchor", "context", "certificate", "resolved", "safe"]):
            continue
        vals = [str(r.get(c, "")).strip().lower() for r in rows]
        bool_like = sum(1 for v in vals if v in {"1", "0", "true", "false", "yes", "no", "pass", "fail", ""})
        if bool_like < max(5, int(0.5 * len(vals))):
            continue
        ids = {str(r.get(raw_col, "")).strip() for r in rows if bval(r.get(c)) and str(r.get(raw_col, "")).strip()}
        out.append({"column": c, "anchor_count": len(ids), "true_rows": sum(1 for r in rows if bval(r.get(c)))})
    return sorted(out, key=lambda x: (abs(int(x["anchor_count"]) - 150), x["column"]))


def choose_anchor_ids(rows: List[dict], raw_col: str, mode: str, explicit_col: Optional[str], target_count: int) -> Tuple[Set[str], dict]:
    meta = {"mode": mode, "raw_id_col": raw_col, "target_count": target_count}
    if explicit_col:
        if explicit_col not in rows[0]:
            raise SystemExit(f"explicit anchor column not found: {explicit_col}; columns={list(rows[0].keys())}")
        ids = {str(r.get(raw_col, "")).strip() for r in rows if bval(r.get(explicit_col)) and str(r.get(raw_col, "")).strip()}
        meta.update({"selected_column": explicit_col, "selection": "explicit_column", "anchor_count": len(ids)})
        return ids, meta

    if mode == "strict_auto":
        priorities = [
            "strict_context_identifiable", "strict_identifiable", "context_identifiable_strict",
            "is_strict_anchor", "strict_anchor", "strict", "cross_clip_context_identifiable",
            "context_identifiable", "is_context_identifiable", "identifiable", "is_anchor", "anchor",
        ]
        cols = list(rows[0].keys())
        for p in priorities:
            if p in cols:
                ids = {str(r.get(raw_col, "")).strip() for r in rows if bval(r.get(p)) and str(r.get(raw_col, "")).strip()}
                if ids:
                    meta.update({"selected_column": p, "selection": "priority_column", "anchor_count": len(ids)})
                    return ids, meta
        cands = discover_anchor_candidates(rows, raw_col)
        if cands:
            c = cands[0]["column"]
            ids = {str(r.get(raw_col, "")).strip() for r in rows if bval(r.get(c)) and str(r.get(raw_col, "")).strip()}
            meta.update({"selected_column": c, "selection": "nearest_bool_candidate", "anchor_count": len(ids), "candidate_columns": cands[:20]})
            return ids, meta
        # fall through to topN
        mode = "topN"
        meta["fallback_from_strict_auto"] = True

    if mode == "topN":
        # Score rows by conservative evidence columns. Prefer unique/strict/certificate/context support, then lower support/ambiguity if present.
        scored: Dict[str, float] = defaultdict(float)
        evidence: Dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            rid = str(r.get(raw_col, "")).strip()
            if not rid:
                continue
            score = 0.0
            for c, v in r.items():
                lc = c.lower()
                if c == raw_col:
                    continue
                if bval(v):
                    if "strict" in lc:
                        score += 5
                    if "unique" in lc:
                        score += 4
                    if "ident" in lc:
                        score += 3
                    if "context" in lc:
                        score += 2
                    if "anchor" in lc or "certificate" in lc or "safe" in lc:
                        score += 2
                    if "resolved" in lc:
                        score += 1
                    evidence[rid][c] += 1
                # numeric heuristic: smaller ambiguity/support rank columns can help if named as such
                if any(k in lc for k in ["support", "clip_count", "contexts", "frequency"]):
                    score += min(fnum(v, 0.0), 10.0) * 0.01
            scored[rid] += score
        if not scored:
            # Last resort: first target_count unique ids in file order.
            ids_order = []
            seen = set()
            for r in rows:
                rid = str(r.get(raw_col, "")).strip()
                if rid and rid not in seen:
                    seen.add(rid); ids_order.append(rid)
            ids = set(ids_order[:target_count])
            meta.update({"selection": "first_unique_ids_fallback", "anchor_count": len(ids)})
            return ids, meta
        ordered = sorted(scored.items(), key=lambda kv: (-kv[1], int(kv[0]) if kv[0].isdigit() else 10**12, kv[0]))
        ids = {rid for rid, _ in ordered[:target_count]}
        meta.update({"selection": "topN_scored", "anchor_count": len(ids), "top_score_min": ordered[min(len(ordered), target_count)-1][1] if ordered else None})
        # Emit small audit of top scores later.
        return ids, meta

    raise SystemExit(f"unsupported mode: {mode}")


def load_matched_pairs(path: Path) -> Dict[str, str]:
    rows = read_csv(path)
    if not rows:
        return {}
    key_cols = ["row_key", "key", "eval_key", "trajectory_key", "video_traj_key", "row_id"]
    match_cols = ["matched_raw_id", "pseudo_raw_id", "pseudo_label_raw_id", "assigned_raw_id", "class_raw_id", "category_id", "matched_category_id"]
    out = {}
    for r in rows:
        k = sget(r, key_cols)
        m = sget(r, match_cols)
        if k and m:
            out[k] = m
    return out


def row_key(r: dict) -> str:
    return sget(r, ["row_key", "key", "eval_key", "trajectory_key", "video_traj_key", "row_id", "trajectory_id"])


def is_top1_hit(r: dict) -> bool:
    return bval(sget(r, ["top1_hit", "gt_top1_hit", "is_top1_gt", "correct"])) or inum(sget(r, ["top1_hit", "gt_top1_hit", "is_top1_gt", "correct"])) == 1


def get_gt(r: dict) -> str:
    return sget(r, ["gt_raw_id", "gt_category_id", "gt_class_raw_id", "raw_category_id", "category_id"])


def get_top1(r: dict) -> str:
    return sget(r, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id", "top1_category_id"])


def get_pseudo(r: dict, matched_by_key: Dict[str, str]) -> str:
    direct = sget(r, ["pseudo_raw_id", "matched_raw_id", "pseudo_label_raw_id", "assigned_raw_id"])
    if direct:
        return direct
    return matched_by_key.get(row_key(r), "")


def group_for(gt: str, pseudo: str, anchors: Set[str], hubs: Set[str]) -> List[str]:
    groups = []
    if gt in anchors:
        groups.append("gt_strict_anchor")
    else:
        groups.append("gt_non_strict_anchor")
    if gt not in anchors and pseudo in anchors:
        groups.append("matched_strict_anchor_gt_nonanchor")
    if gt not in anchors and pseudo in hubs:
        groups.append("matched_hub_gt_nonanchor")
    return groups


def summarize_epoch(epoch: int, rows: List[dict], base_rows_by_key: Dict[str, dict], anchors: Set[str], hubs: Set[str], matched_by_key: Dict[str, str]) -> Tuple[List[dict], List[dict]]:
    stats = defaultdict(Counter)
    c2w_events = []
    for r in rows:
        k = row_key(r)
        gt = get_gt(r)
        top1 = get_top1(r)
        pseudo = get_pseudo(r, matched_by_key)
        hit = is_top1_hit(r)
        base = base_rows_by_key.get(k)
        base_hit = is_top1_hit(base) if base else False
        gap = fnum(sget(r, ["wrong_abs_gap", "margin_abs_gap", "top1_minus_gt", "wrong_gap"]), 0.0)
        mb = "correct" if hit else ("small" if gap <= 1.0 else "large" if gap >= 3.0 else "middle")
        pseudo_bad = bool(pseudo and gt and pseudo != gt)
        pseudo_bad_top1 = pseudo_bad and top1 == pseudo
        top1_hub = top1 in hubs
        groups = group_for(gt, pseudo, anchors, hubs)
        for g in groups:
            s = stats[g]
            s["rows"] += 1
            s["correct"] += int(hit)
            s["wrong"] += int(not hit)
            s[f"{mb}_wrong"] += int(not hit)
            s["top1_hub_wrong"] += int((not hit) and top1_hub)
            s["pseudo_label_available"] += int(bool(pseudo))
            s["pseudo_mismatch"] += int(pseudo_bad)
            s["pseudo_wrong_top1"] += int(pseudo_bad_top1)
            s["correct_to_wrong_vs_baseline"] += int(base_hit and not hit)
            s["wrong_to_correct_vs_baseline"] += int((not base_hit) and hit)
            s["correct_to_wrong_pseudo_bad_top1"] += int(base_hit and not hit and pseudo_bad_top1)
            s["correct_to_wrong_to_hub"] += int(base_hit and not hit and top1_hub)
            if not hit:
                s["wrong_abs_gap_sum_x1e6"] += int(gap * 1_000_000)
            if base_hit and not hit:
                c2w_events.append({
                    "epoch": epoch,
                    "group": g,
                    "row_key": k,
                    "gt_raw_id": gt,
                    "matched_raw_id": pseudo,
                    "new_top1_raw_id": top1,
                    "pseudo_bad": int(pseudo_bad),
                    "new_top1_is_pseudo": int(top1 == pseudo and bool(pseudo)),
                    "new_top1_is_hub": int(top1_hub),
                    "wrong_abs_gap": gap,
                })
    table = []
    for g, s in stats.items():
        wrong = s["wrong"]
        rows_n = s["rows"]
        table.append({
            "epoch": epoch,
            "group": g,
            "rows": rows_n,
            "correct": s["correct"],
            "wrong": wrong,
            "micro_top1": s["correct"] / rows_n if rows_n else 0.0,
            "small_wrong": s["small_wrong"],
            "middle_wrong": s["middle_wrong"],
            "large_wrong": s["large_wrong"],
            "top1_hub_wrong": s["top1_hub_wrong"],
            "pseudo_label_available": s["pseudo_label_available"],
            "pseudo_mismatch": s["pseudo_mismatch"],
            "pseudo_wrong_top1": s["pseudo_wrong_top1"],
            "pseudo_wrong_top1_rate": s["pseudo_wrong_top1"] / s["pseudo_label_available"] if s["pseudo_label_available"] else 0.0,
            "correct_to_wrong_vs_baseline": s["correct_to_wrong_vs_baseline"],
            "wrong_to_correct_vs_baseline": s["wrong_to_correct_vs_baseline"],
            "correct_to_wrong_pseudo_bad_top1": s["correct_to_wrong_pseudo_bad_top1"],
            "correct_to_wrong_to_hub": s["correct_to_wrong_to_hub"],
            "mean_wrong_abs_gap_wrong": (s["wrong_abs_gap_sum_x1e6"] / 1_000_000.0 / wrong) if wrong else 0.0,
        })
    return table, c2w_events


def aggregate_causes(events: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    c1 = Counter()
    c2 = Counter()
    c3 = Counter()
    for e in events:
        c1[(e["epoch"], e["group"], e.get("matched_raw_id", ""), e.get("new_top1_raw_id", ""), e.get("pseudo_bad", 0), e.get("new_top1_is_pseudo", 0), e.get("new_top1_is_hub", 0))] += 1
        c2[(e["epoch"], e["group"], e.get("new_top1_raw_id", ""))] += 1
        c3[(e["epoch"], e["group"], e.get("gt_raw_id", ""))] += 1
    rows1 = [{"epoch": k[0], "group": k[1], "matched_raw_id": k[2], "new_top1_raw_id": k[3], "pseudo_bad": k[4], "new_top1_is_pseudo": k[5], "new_top1_is_hub": k[6], "correct_to_wrong_rows": v} for k, v in c1.items()]
    rows2 = [{"epoch": k[0], "group": k[1], "new_top1_raw_id": k[2], "correct_to_wrong_rows": v} for k, v in c2.items()]
    rows3 = [{"epoch": k[0], "group": k[1], "gt_raw_id": k[2], "correct_to_wrong_rows": v} for k, v in c3.items()]
    rows1.sort(key=lambda r: -r["correct_to_wrong_rows"])
    rows2.sort(key=lambda r: -r["correct_to_wrong_rows"])
    rows3.sort(key=lambda r: -r["correct_to_wrong_rows"])
    return rows1, rows2, rows3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift_audit_dir", required=True)
    ap.add_argument("--anchor_csv", required=True)
    ap.add_argument("--matched_pairs_csv", required=True)
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--baseline_epoch", type=int, default=5)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--anchor_mode", choices=["strict_auto", "topN"], default="strict_auto")
    ap.add_argument("--anchor_column", default=None)
    ap.add_argument("--strict_anchor_count", type=int, default=150)
    ap.add_argument("--hub_raw_ids", default="773,1112,63,936,931,970,173,580,135,1044,1114,868")
    args = ap.parse_args()

    drift_dir = Path(args.drift_audit_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    hubs = {x.strip() for x in args.hub_raw_ids.split(",") if x.strip()}

    anchor_rows = read_csv(Path(args.anchor_csv))
    raw_col = find_raw_id_col(anchor_rows)
    anchors, meta = choose_anchor_ids(anchor_rows, raw_col, args.anchor_mode, args.anchor_column, args.strict_anchor_count)
    meta["anchor_csv"] = str(Path(args.anchor_csv))
    meta["candidate_columns"] = discover_anchor_candidates(anchor_rows, raw_col)[:50]

    (out / "strict_anchor_raw_ids.txt").write_text("\n".join(sorted(anchors, key=lambda x: int(x) if x.isdigit() else 10**12)) + "\n", encoding="utf-8")
    (out / "strict_anchor_selection_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    matched_by_key = load_matched_pairs(Path(args.matched_pairs_csv))
    base_csv = drift_dir / f"epoch_{args.baseline_epoch:03d}_true_margin" / "true_score_margin_row_audit.csv"
    if not base_csv.exists():
        alt = drift_dir / f"epoch_{args.baseline_epoch:03d}_true_margin" / "export_replay" / "eval_after_row_predictions.csv"
        base_csv = alt
    if not base_csv.exists():
        raise SystemExit(f"missing baseline row csv: {base_csv}")
    base_rows = read_csv(base_csv)
    base_by_key = {row_key(r): r for r in base_rows if row_key(r)}

    all_table, all_events = [], []
    for ep in epochs:
        row_csv = drift_dir / f"epoch_{ep:03d}_true_margin" / "true_score_margin_row_audit.csv"
        if not row_csv.exists():
            row_csv = drift_dir / f"epoch_{ep:03d}_true_margin" / "export_replay" / "eval_after_row_predictions.csv"
        if not row_csv.exists():
            raise SystemExit(f"missing epoch row csv for epoch {ep}: {row_csv}")
        rows = read_csv(row_csv)
        table, events = summarize_epoch(ep, rows, base_by_key, anchors, hubs, matched_by_key)
        all_table.extend(table)
        all_events.extend(events)

    fields = [
        "epoch", "group", "rows", "correct", "wrong", "micro_top1", "small_wrong", "middle_wrong", "large_wrong",
        "top1_hub_wrong", "pseudo_label_available", "pseudo_mismatch", "pseudo_wrong_top1", "pseudo_wrong_top1_rate",
        "correct_to_wrong_vs_baseline", "wrong_to_correct_vs_baseline", "correct_to_wrong_pseudo_bad_top1", "correct_to_wrong_to_hub", "mean_wrong_abs_gap_wrong",
    ]
    write_csv(out / "strict_anchor_epoch_group_drift_table.csv", all_table, fields)
    write_csv(out / "strict_anchor_row_transition_events.csv", all_events)
    top_causes, top_abs, top_gt = aggregate_causes(all_events)
    write_csv(out / "top_correct_to_wrong_assignment_causes.csv", top_causes, ["epoch", "group", "matched_raw_id", "new_top1_raw_id", "pseudo_bad", "new_top1_is_pseudo", "new_top1_is_hub", "correct_to_wrong_rows"])
    write_csv(out / "top_correct_to_wrong_absorbers_by_group.csv", top_abs, ["epoch", "group", "new_top1_raw_id", "correct_to_wrong_rows"])
    write_csv(out / "top_correct_to_wrong_gt_classes_by_group.csv", top_gt, ["epoch", "group", "gt_raw_id", "correct_to_wrong_rows"])

    # first drift per group: epoch where micro decreases from baseline and c2w > 0
    by_group_epoch = {(str(r["group"]), int(r["epoch"])): r for r in all_table}
    groups = sorted({str(r["group"]) for r in all_table})
    first = []
    for g in groups:
        b = by_group_epoch.get((g, args.baseline_epoch))
        if not b:
            continue
        bm = fnum(b.get("micro_top1"))
        for ep in epochs:
            if ep == args.baseline_epoch:
                continue
            r = by_group_epoch.get((g, ep))
            if not r:
                continue
            if fnum(r.get("micro_top1")) < bm and inum(r.get("correct_to_wrong_vs_baseline")) > 0:
                first.append({
                    "group": g,
                    "first_drift_epoch": ep,
                    "baseline_micro": bm,
                    "epoch_micro": fnum(r.get("micro_top1")),
                    "correct_to_wrong_vs_baseline": inum(r.get("correct_to_wrong_vs_baseline")),
                    "correct_to_wrong_pseudo_bad_top1": inum(r.get("correct_to_wrong_pseudo_bad_top1")),
                    "correct_to_wrong_to_hub": inum(r.get("correct_to_wrong_to_hub")),
                })
                break

    summary = {
        "status": "PASS",
        "drift_audit_dir": str(drift_dir),
        "matched_pairs_csv": str(Path(args.matched_pairs_csv)),
        "strict_anchor_meta": meta,
        "strict_anchor_count": len(anchors),
        "hub_raw_ids": sorted(hubs, key=lambda x: int(x) if x.isdigit() else 10**12),
        "epochs": epochs,
        "baseline_epoch": args.baseline_epoch,
        "first_drift_by_group": first,
        "outputs": {
            "strict_anchor_epoch_group_drift_table": str(out / "strict_anchor_epoch_group_drift_table.csv"),
            "strict_anchor_selection_meta": str(out / "strict_anchor_selection_meta.json"),
            "strict_anchor_raw_ids": str(out / "strict_anchor_raw_ids.txt"),
            "top_correct_to_wrong_assignment_causes": str(out / "top_correct_to_wrong_assignment_causes.csv"),
            "top_correct_to_wrong_absorbers_by_group": str(out / "top_correct_to_wrong_absorbers_by_group.csv"),
        },
    }
    (out / "strict_anchor_first_drift_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# A8 Strict-Anchor First Drift Audit")
    md.append("")
    md.append(f"- strict_anchor_count: `{len(anchors)}`")
    md.append(f"- anchor_selection: `{meta.get('selection')}`")
    md.append(f"- selected_column: `{meta.get('selected_column', '')}`")
    md.append(f"- anchor_csv: `{args.anchor_csv}`")
    md.append("")
    md.append("## First drift by group")
    md.append("| group | first_epoch | baseline_micro | epoch_micro | c2w | c2w_pseudo_bad_top1 | c2w_to_hub |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in first:
        md.append(f"| {r['group']} | {r['first_drift_epoch']} | {r['baseline_micro']:.6f} | {r['epoch_micro']:.6f} | {r['correct_to_wrong_vs_baseline']} | {r['correct_to_wrong_pseudo_bad_top1']} | {r['correct_to_wrong_to_hub']} |")
    md.append("")
    md.append("## Interpretation")
    md.append("- If `gt_strict_anchor` drifts at 5->10, even the strict anchor set is not protected by continued CE.")
    md.append("- If `matched_strict_anchor_gt_nonanchor` has high pseudo_bad_top1, Hungarian anchor assignments are being overfit when they disagree with GT.")
    md.append("- This is read-only and uses existing drift-audit row exports.")
    (out / "A8_STRICT_ANCHOR_FIRST_DRIFT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
