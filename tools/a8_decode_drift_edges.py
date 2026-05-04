#!/usr/bin/env python3
"""Decode raw-id drift edges into class names and aggregate top boundaries.

Read-only. It joins correct->wrong transition causes/events with class names from
LV-VIS annotations and/or split JSON. It is designed to explain edges such as
63 -> 527, 773, 1112.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_names(annotation_json: Optional[Path], split_json: Optional[Path], extra_json: Optional[Path]) -> Dict[str, str]:
    names = {}
    def add(rid, name):
        if rid is None or name is None:
            return
        rid = str(rid).strip(); name = str(name).strip()
        if rid and name and rid not in names:
            names[rid] = name
    if annotation_json and annotation_json.exists():
        obj = json.loads(annotation_json.read_text(encoding="utf-8"))
        for c in obj.get("categories", []):
            add(c.get("id") or c.get("raw_id") or c.get("category_id"), c.get("name"))
    if split_json and split_json.exists():
        obj = json.loads(split_json.read_text(encoding="utf-8"))
        # support various schemas
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    add(v.get("raw_id") or v.get("id") or k, v.get("name") or v.get("class_name"))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            add(item.get("raw_id") or item.get("id") or item.get("category_id"), item.get("name") or item.get("class_name"))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    add(item.get("raw_id") or item.get("id") or item.get("category_id"), item.get("name") or item.get("class_name"))
    if extra_json and extra_json.exists():
        obj = json.loads(extra_json.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    add(k, v)
                elif isinstance(v, dict):
                    add(v.get("raw_id") or v.get("id") or k, v.get("name") or v.get("class_name"))
    return names


def decode(names: Dict[str, str], rid: str) -> str:
    rid = str(rid or "").strip()
    return names.get(rid, "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True, help="Anchor/strict-anchor audit dir containing transition CSVs")
    ap.add_argument("--drift_audit_dir", default=None, help="Optional original drift audit dir for full row event fallback")
    ap.add_argument("--annotation_json", default=None)
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--extra_name_json", default=None)
    ap.add_argument("--focus_edges", default="63:527,773,1112")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    audit_dir = Path(args.audit_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(Path(args.annotation_json) if args.annotation_json else None, Path(args.split_json) if args.split_json else None, Path(args.extra_name_json) if args.extra_name_json else None)

    cause_csv = audit_dir / "top_correct_to_wrong_assignment_causes.csv"
    event_csv = audit_dir / "strict_anchor_row_transition_events.csv"
    if not event_csv.exists():
        event_csv = audit_dir / "anchor_row_transition_events.csv"
    if not cause_csv.exists():
        raise SystemExit(f"missing cause csv: {cause_csv}")
    causes = read_csv(cause_csv)
    events = read_csv(event_csv) if event_csv.exists() else []

    decoded_causes = []
    for r in causes:
        matched = sget(r, ["matched_raw_id"])
        top1 = sget(r, ["new_top1_raw_id", "top1_raw_id"])
        gt = sget(r, ["gt_raw_id"])
        rr = dict(r)
        rr["matched_class_name"] = decode(names, matched)
        rr["new_top1_class_name"] = decode(names, top1)
        rr["gt_class_name"] = decode(names, gt)
        decoded_causes.append(rr)

    cause_fields = list(decoded_causes[0].keys()) if decoded_causes else []
    # Place useful decode fields near ids.
    preferred = ["epoch", "group", "gt_raw_id", "gt_class_name", "matched_raw_id", "matched_class_name", "new_top1_raw_id", "new_top1_class_name", "pseudo_bad", "new_top1_is_pseudo", "new_top1_is_hub", "correct_to_wrong_rows"]
    fields = preferred + [f for f in cause_fields if f not in preferred]
    write_csv(out / "decoded_top_correct_to_wrong_assignment_causes.csv", decoded_causes, fields)

    # Aggregate full events by GT, matched, top1 if available.
    edge_counter = Counter()
    for e in events:
        gt = sget(e, ["gt_raw_id"])
        matched = sget(e, ["matched_raw_id"])
        top1 = sget(e, ["new_top1_raw_id", "top1_raw_id"])
        group = sget(e, ["group"])
        epoch = sget(e, ["epoch"])
        edge_counter[(epoch, group, gt, matched, top1)] += 1
    edge_rows = []
    for (epoch, group, gt, matched, top1), n in edge_counter.items():
        edge_rows.append({
            "epoch": epoch,
            "group": group,
            "gt_raw_id": gt,
            "gt_class_name": decode(names, gt),
            "matched_raw_id": matched,
            "matched_class_name": decode(names, matched),
            "new_top1_raw_id": top1,
            "new_top1_class_name": decode(names, top1),
            "correct_to_wrong_rows": n,
        })
    edge_rows.sort(key=lambda r: -int(r["correct_to_wrong_rows"]))
    write_csv(out / "decoded_correct_to_wrong_edges_by_gt_matched_top1.csv", edge_rows, ["epoch", "group", "gt_raw_id", "gt_class_name", "matched_raw_id", "matched_class_name", "new_top1_raw_id", "new_top1_class_name", "correct_to_wrong_rows"])

    focus_rows = []
    focus_items = [x.strip() for x in args.focus_edges.split(",") if x.strip()]
    for item in focus_items:
        if ":" in item:
            a, b = item.split(":", 1)
            for r in edge_rows:
                if str(r.get("matched_raw_id")) == a and str(r.get("new_top1_raw_id")) == b:
                    focus_rows.append(dict(r, focus_query=item))
            for r in decoded_causes:
                if str(r.get("matched_raw_id")) == a and str(r.get("new_top1_raw_id")) == b:
                    focus_rows.append(dict(r, focus_query=item))
        else:
            for r in decoded_causes:
                if str(r.get("new_top1_raw_id")) == item or str(r.get("matched_raw_id")) == item or str(r.get("gt_raw_id")) == item:
                    focus_rows.append(dict(r, focus_query=item))
            for r in edge_rows:
                if str(r.get("new_top1_raw_id")) == item or str(r.get("matched_raw_id")) == item or str(r.get("gt_raw_id")) == item:
                    focus_rows.append(dict(r, focus_query=item))
    # de-dupe by json representation
    seen = set(); unique_focus = []
    for r in focus_rows:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key); unique_focus.append(r)
    focus_fields = ["focus_query", "epoch", "group", "gt_raw_id", "gt_class_name", "matched_raw_id", "matched_class_name", "new_top1_raw_id", "new_top1_class_name", "pseudo_bad", "new_top1_is_pseudo", "new_top1_is_hub", "correct_to_wrong_rows"]
    write_csv(out / "decoded_focus_edges.csv", unique_focus, focus_fields)

    summary = {
        "status": "PASS",
        "audit_dir": str(audit_dir),
        "name_sources": {
            "annotation_json": args.annotation_json,
            "split_json": args.split_json,
            "extra_name_json": args.extra_name_json,
            "decoded_name_count": len(names),
        },
        "focus_edges": focus_items,
        "outputs": {
            "decoded_top_correct_to_wrong_assignment_causes": str(out / "decoded_top_correct_to_wrong_assignment_causes.csv"),
            "decoded_correct_to_wrong_edges_by_gt_matched_top1": str(out / "decoded_correct_to_wrong_edges_by_gt_matched_top1.csv"),
            "decoded_focus_edges": str(out / "decoded_focus_edges.csv"),
        },
    }
    (out / "decoded_drift_edge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# A8 Drift Edge Decode Audit", "",
        f"- decoded_name_count: `{len(names)}`",
        f"- focus_edges: `{', '.join(focus_items)}`", "",
        "## Outputs", "",
        f"- `{out / 'decoded_top_correct_to_wrong_assignment_causes.csv'}`",
        f"- `{out / 'decoded_correct_to_wrong_edges_by_gt_matched_top1.csv'}`",
        f"- `{out / 'decoded_focus_edges.csv'}`",
        "", "## Note", "",
        "If class names are blank, the provided annotation/split sources did not contain those raw IDs; rerun with an extra raw-id-to-name JSON.",
    ]
    (out / "A8_DRIFT_EDGE_DECODE_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
