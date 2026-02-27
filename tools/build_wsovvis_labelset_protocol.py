#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence


VERSION = "wsovvis-labelset-protocol-v1"


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Input JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a top-level object.")
    return data


def _validate_protocol_args(protocol: str, missing_rate: float, min_labels_per_clip: int) -> None:
    if protocol != "uniform":
        raise ValueError("Unsupported protocol. v1 supports 'uniform' only.")
    if missing_rate < 0.0 or missing_rate > 1.0:
        raise ValueError("Invalid --missing-rate. Expected a float in [0, 1].")
    if min_labels_per_clip < 0:
        raise ValueError("Invalid --min-labels-per-clip. Expected an integer >= 0.")


def _validate_input_shape(data: dict) -> None:
    for key in ("videos", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Malformed input JSON: missing top-level key '{key}'.")
        if not isinstance(data[key], list):
            raise ValueError(f"Malformed input JSON: '{key}' must be a list.")


def _int_field(obj: dict, key: str, ctx: str) -> int:
    if key not in obj:
        raise ValueError(f"Malformed input JSON: missing '{key}' in {ctx}.")
    value = obj[key]
    if not isinstance(value, int):
        raise ValueError(f"Malformed input JSON: '{key}' in {ctx} must be an integer.")
    return value


def _extract_video_info(videos: Sequence[dict]) -> Dict[int, str]:
    video_id_to_name: Dict[int, str] = {}
    for i, video in enumerate(videos):
        if not isinstance(video, dict):
            raise ValueError(f"Malformed input JSON: videos[{i}] must be an object.")
        vid = _int_field(video, "id", f"videos[{i}]")
        if vid in video_id_to_name:
            raise ValueError(f"Malformed input JSON: duplicate video id {vid}.")
        name = video.get("name")
        video_id_to_name[vid] = name if isinstance(name, str) else ""
    return video_id_to_name


def _extract_categories(categories: Sequence[dict]) -> Dict[int, str]:
    category_id_to_name: Dict[int, str] = {}
    for i, cat in enumerate(categories):
        if not isinstance(cat, dict):
            raise ValueError(f"Malformed input JSON: categories[{i}] must be an object.")
        cid = _int_field(cat, "id", f"categories[{i}]")
        if cid in category_id_to_name:
            raise ValueError(f"Malformed input JSON: duplicate category id {cid}.")
        name = cat.get("name")
        category_id_to_name[cid] = name if isinstance(name, str) else ""
    return category_id_to_name


def _build_full_labelsets(
    annotations: Sequence[dict], known_video_ids: set, known_category_ids: set
) -> Dict[int, set]:
    labels_by_video: Dict[int, set] = {}
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise ValueError(f"Malformed input JSON: annotations[{i}] must be an object.")
        vid = _int_field(ann, "video_id", f"annotations[{i}]")
        cid = _int_field(ann, "category_id", f"annotations[{i}]")
        if vid not in known_video_ids:
            raise ValueError(
                f"Invalid annotation at annotations[{i}]: unknown video_id {vid}."
            )
        if cid not in known_category_ids:
            raise ValueError(
                f"Invalid annotation at annotations[{i}]: unknown category_id {cid}."
            )
        labels_by_video.setdefault(vid, set()).add(cid)
    return labels_by_video


def _sample_observed_labelset(full_ids: List[int], missing_rate: float, min_labels_per_clip: int, rng: random.Random) -> List[int]:
    n = len(full_ids)
    keep_floor = max(min_labels_per_clip, math.ceil((1.0 - missing_rate) * n))
    num_keep = min(n, keep_floor)
    if num_keep == 0:
        return []
    return sorted(rng.sample(full_ids, k=num_keep))


def build_outputs(
    data: dict,
    input_json_path: str,
    protocol: str,
    missing_rate: float,
    seed: int,
    min_labels_per_clip: int,
) -> tuple:
    _validate_input_shape(data)
    _validate_protocol_args(protocol, missing_rate, min_labels_per_clip)

    videos = data["videos"]
    annotations = data["annotations"]
    categories = data["categories"]

    video_id_to_name = _extract_video_info(videos)
    category_id_to_name = _extract_categories(categories)
    labels_by_video = _build_full_labelsets(
        annotations, set(video_id_to_name.keys()), set(category_id_to_name.keys())
    )

    rng = random.Random(seed)
    clips = []
    for video_id in sorted(video_id_to_name.keys()):
        full_ids = sorted(labels_by_video.get(video_id, set()))
        observed_ids = _sample_observed_labelset(full_ids, missing_rate, min_labels_per_clip, rng)
        clip = {
            "video_id": video_id,
            "label_set_full_ids": full_ids,
            "label_set_observed_ids": observed_ids,
            "num_full": len(full_ids),
            "num_observed": len(observed_ids),
        }
        if video_id_to_name[video_id]:
            clip["video_name"] = video_id_to_name[video_id]
        clips.append(clip)

    nonempty_full = [c for c in clips if c["num_full"] > 0]
    total_full = sum(c["num_full"] for c in clips)
    total_observed = sum(c["num_observed"] for c in clips)
    denom_nonempty = len(nonempty_full)
    empirical_missing = (
        sum((c["num_full"] - c["num_observed"]) / c["num_full"] for c in nonempty_full)
        / denom_nonempty
        if denom_nonempty > 0
        else 0.0
    )

    main_output = {
        "version": VERSION,
        "protocol": protocol,
        "missing_rate": missing_rate,
        "seed": seed,
        "clips": clips,
    }

    manifest = {
        "version": VERSION,
        "source": {"input_json": input_json_path},
        "params": {
            "protocol": protocol,
            "missing_rate": missing_rate,
            "seed": seed,
            "min_labels_per_clip": min_labels_per_clip,
        },
        "dataset_stats": {
            "num_videos_total": len(videos),
            "num_videos_with_labels": len(nonempty_full),
            "num_categories": len(categories),
        },
        "label_stats": {
            "avg_num_full_labels_per_clip": (total_full / len(clips)) if clips else 0.0,
            "avg_num_observed_labels_per_clip": (total_observed / len(clips)) if clips else 0.0,
            "overall_missing_ratio_empirical": empirical_missing,
        },
        "integrity_checks": {
            "all_observed_subset_of_full": all(
                set(c["label_set_observed_ids"]).issubset(set(c["label_set_full_ids"]))
                for c in clips
            ),
            "all_nonempty_when_full_nonempty": all(
                (c["num_full"] == 0) or (c["num_observed"] > 0)
                for c in clips
            ),
        },
        "category_id_to_name": {str(k): v for k, v in sorted(category_id_to_name.items())},
    }
    return main_output, manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WS-OVVIS clip-level full/observed labelset protocol JSONs."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--protocol", default="uniform")
    parser.add_argument("--missing-rate", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--min-labels-per-clip", default=1, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_path = Path(args.input_json)
        output_path = Path(args.output_json)
        manifest_path = Path(args.manifest_json)

        data = _load_json(input_path)
        main_output, manifest = build_outputs(
            data=data,
            input_json_path=str(input_path),
            protocol=args.protocol,
            missing_rate=args.missing_rate,
            seed=args.seed,
            min_labels_per_clip=args.min_labels_per_clip,
        )
        _write_json(output_path, main_output)
        _write_json(manifest_path, manifest)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
