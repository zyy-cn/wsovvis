#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_video_name(video_payload: Mapping[str, Any]) -> str:
    name = video_payload.get("name")
    if isinstance(name, str) and name:
        return name
    file_names = video_payload.get("file_names")
    if isinstance(file_names, list) and file_names:
        first = file_names[0]
        if isinstance(first, str) and first:
            parent_name = Path(first).parent.name
            if parent_name:
                return parent_name
            return Path(first).stem
    raw_id = video_payload.get("id")
    if raw_id is None:
        raise ValueError("video record missing canonical name and id")
    return str(raw_id)


def _row_key(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("annotations"), list):
        return "annotations"
    if isinstance(payload.get("predictions"), list):
        return "predictions"
    raise ValueError("payload must contain annotations or predictions")


def _video_length(video_payload: Mapping[str, Any]) -> int:
    length = video_payload.get("length")
    if isinstance(length, int) and length > 0:
        return int(length)
    file_names = video_payload.get("file_names")
    if isinstance(file_names, list):
        return len(file_names)
    return 0


def _build_video_index(payload: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("payload must contain videos")
    indexed: List[Dict[str, Any]] = []
    id_to_name: Dict[str, str] = {}
    for row in videos:
        if not isinstance(row, dict):
            continue
        name = _canonical_video_name(row)
        video_id = str(row.get("id"))
        id_to_name[video_id] = name
        indexed.append(
            {
                "video_id": video_id,
                "name": name,
                "length": _video_length(row),
                "payload": row,
            }
        )
    return indexed, id_to_name


def _count_rows_by_video(payload: Mapping[str, Any], id_to_name: Mapping[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in payload.get(_row_key(payload), []):
        if not isinstance(row, dict):
            continue
        name = id_to_name.get(str(row.get("video_id")))
        if name is None:
            continue
        counts[name] += 1
    return counts


def _quantile_edges(values: Sequence[int], num_bins: int = 4) -> List[int]:
    if not values:
        return [0] * max(1, num_bins - 1)
    ordered = sorted(int(v) for v in values)
    edges: List[int] = []
    for i in range(1, num_bins):
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * i / num_bins) - 1))
        edges.append(int(ordered[index]))
    return edges


def _bin_index(value: int, edges: Sequence[int]) -> int:
    index = 0
    for edge in edges:
        if int(value) > int(edge):
            index += 1
    return index


def _stable_shuffle(items: Sequence[Dict[str, Any]], *, seed_text: str) -> List[Dict[str, Any]]:
    shuffled = list(sorted(items, key=lambda row: (row["name"], row["video_id"])))
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _select_stratified(
    *,
    records: Sequence[Dict[str, Any]],
    target_count: int,
    seed: int,
    anchor_names: Sequence[str],
    primary_key: str,
    secondary_key: str,
) -> List[Dict[str, Any]]:
    by_name = {row["name"]: row for row in records}
    anchor_list: List[Dict[str, Any]] = []
    anchor_name_set = set()
    for name in anchor_names:
        if not name:
            continue
        if name in anchor_name_set:
            raise ValueError(f"duplicate anchor video name: {name}")
        if name not in by_name:
            raise ValueError(f"required anchor video is missing from selection surface: {name}")
        anchor_name_set.add(name)
        anchor_list.append(by_name[name])
    if target_count < len(anchor_list):
        raise ValueError("target_count is smaller than required anchors")

    pool = [row for row in records if row["name"] not in anchor_name_set]
    remaining_target = target_count - len(anchor_list)
    if remaining_target == 0:
        return list(sorted(anchor_list, key=lambda row: row["name"]))

    primary_edges = _quantile_edges([int(row[primary_key]) for row in pool])
    secondary_edges = _quantile_edges([int(row[secondary_key]) for row in pool])
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in pool:
        grouped[
            (
                _bin_index(int(row[primary_key]), primary_edges),
                _bin_index(int(row[secondary_key]), secondary_edges),
            )
        ].append(row)

    group_keys = sorted(grouped.keys())
    group_sizes = {key: len(grouped[key]) for key in group_keys}
    total_pool = sum(group_sizes.values())
    if total_pool < remaining_target:
        raise ValueError("selection pool is smaller than requested target")

    quotas: Dict[Tuple[int, int], int] = {}
    remainders: List[Tuple[float, Tuple[int, int]]] = []
    assigned = 0
    for key in group_keys:
        exact = group_sizes[key] * remaining_target / total_pool
        floor_value = int(math.floor(exact))
        quotas[key] = min(group_sizes[key], floor_value)
        assigned += quotas[key]
        remainders.append((exact - floor_value, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if assigned >= remaining_target:
            break
        if quotas[key] >= group_sizes[key]:
            continue
        quotas[key] += 1
        assigned += 1

    selected = list(anchor_list)
    leftovers: List[Dict[str, Any]] = []
    for key in group_keys:
        shuffled = _stable_shuffle(grouped[key], seed_text=f"{seed}|{key[0]}|{key[1]}")
        take = quotas[key]
        selected.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])
    if len(selected) < target_count:
        selected.extend(_stable_shuffle(leftovers, seed_text=f"{seed}|leftovers")[: target_count - len(selected)])
    if len(selected) != target_count:
        raise ValueError(f"selection size mismatch: expected {target_count}, got {len(selected)}")
    return list(sorted(selected, key=lambda row: row["name"]))


def _subset_payload(payload: Mapping[str, Any], *, selected_names: Iterable[str]) -> Dict[str, Any]:
    selected = set(str(name) for name in selected_names)
    videos, id_to_name = _build_video_index(payload)
    kept_video_ids = {row["video_id"] for row in videos if row["name"] in selected}
    out = dict(payload)
    out["videos"] = [row["payload"] for row in videos if row["name"] in selected]
    rows_key = _row_key(payload)
    out[rows_key] = [
        row
        for row in payload.get(rows_key, [])
        if isinstance(row, dict)
        and str(row.get("video_id")) in kept_video_ids
        and id_to_name.get(str(row.get("video_id"))) in selected
    ]
    return out


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Route A pilot train/val subset surfaces.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=96)
    parser.add_argument("--anchor-video-name", dest="anchor_video_names", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_payload = _load_json(args.train_json)
    gt_payload = _load_json(args.gt_json)
    raw_payload = _load_json(args.raw_json)

    train_videos, train_id_to_name = _build_video_index(train_payload)
    gt_videos, gt_id_to_name = _build_video_index(gt_payload)
    raw_videos, raw_id_to_name = _build_video_index(raw_payload)

    train_track_counts = _count_rows_by_video(train_payload, train_id_to_name)
    gt_instance_counts = _count_rows_by_video(gt_payload, gt_id_to_name)
    raw_track_counts = _count_rows_by_video(raw_payload, raw_id_to_name)

    train_records = [
        {
            "video_id": row["video_id"],
            "name": row["name"],
            "length": int(row["length"]),
            "track_count": int(train_track_counts.get(row["name"], 0)),
        }
        for row in train_videos
    ]
    val_records = [
        {
            "video_id": row["video_id"],
            "name": row["name"],
            "length": int(row["length"]),
            "gt_count": int(gt_instance_counts.get(row["name"], 0)),
            "raw_count": int(raw_track_counts.get(row["name"], 0)),
        }
        for row in gt_videos
    ]

    selected_train = _select_stratified(
        records=train_records,
        target_count=int(args.train_count),
        seed=int(args.seed),
        anchor_names=(),
        primary_key="track_count",
        secondary_key="length",
    )
    selected_val = _select_stratified(
        records=val_records,
        target_count=int(args.val_count),
        seed=int(args.seed),
        anchor_names=tuple(args.anchor_video_names),
        primary_key="gt_count",
        secondary_key="raw_count",
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_names = [row["name"] for row in selected_train]
    val_names = [row["name"] for row in selected_val]

    _write_lines(output_root / "train_video_names.txt", train_names)
    _write_lines(output_root / "val_video_names.txt", val_names)
    _dump_json(output_root / "train_subset_conf060.json", _subset_payload(train_payload, selected_names=train_names))
    _dump_json(output_root / "val_gt_subset.json", _subset_payload(gt_payload, selected_names=val_names))
    _dump_json(output_root / "val_raw_subset.json", _subset_payload(raw_payload, selected_names=val_names))
    _dump_json(
        output_root / "selection_manifest.json",
        {
            "schema_name": "wsovvis.s1_routea_pilot_surface",
            "schema_version": "1.0.0",
            "seed": int(args.seed),
            "train_count": int(args.train_count),
            "val_count": int(args.val_count),
            "anchor_video_names": list(args.anchor_video_names),
            "inputs": {
                "train_json": str(args.train_json),
                "gt_json": str(args.gt_json),
                "raw_json": str(args.raw_json),
            },
            "selected_train": selected_train,
            "selected_val": selected_val,
        },
    )
    print(json.dumps({"output_root": str(output_root), "train_count": len(train_names), "val_count": len(val_names)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
