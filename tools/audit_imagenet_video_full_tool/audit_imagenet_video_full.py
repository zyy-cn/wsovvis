#!/usr/bin/env python3
"""
Full audit for VideoCutLER-style ImageNet synthetic video annotations.

Scans:
- JSON structure consistency
- video/image existence and image sizes
- annotation length consistency vs video_length
- segmentation structure / RLE size mismatches / polygon validity
- frame-level instance-count consistency relevant to official copy_paste assumptions

Outputs under --out-dir:
- summary.json
- findings.jsonl
- report.md

Designed to be dependency-light. Uses PIL to inspect image size. If pycocotools
is installed and --decode-rle is set, compressed/uncompressed RLE can be decoded
for extra validation.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    print(f"PIL is required: {e}", file=sys.stderr)
    sys.exit(2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_image_size(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            w, h = img.size
        return h, w, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def maybe_decode_rle(seg: Dict[str, Any]) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    try:
        from pycocotools import mask as mask_util  # type: ignore
    except Exception:
        return False, None, "pycocotools not available"
    try:
        m = mask_util.decode(seg)
        if m.ndim == 3:
            # Some decoders may return HxWx1
            m = m[..., 0]
        h, w = m.shape[:2]
        return True, (h, w), None
    except Exception as e:
        return True, None, f"{type(e).__name__}: {e}"


def add_finding(findings: List[Dict[str, Any]], counts: Dict[str, int], kind: str, severity: str, **payload: Any) -> None:
    counts[kind] += 1
    item = {"kind": kind, "severity": severity}
    item.update(payload)
    findings.append(item)


def summarize_examples(findings: List[Dict[str, Any]], max_examples: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for f in findings:
        kind = f["kind"]
        if len(grouped[kind]) < max_examples:
            grouped[kind].append(f)
    return dict(grouped)


def audit(json_path: Path, datasets_root: Path, out_dir: Path, decode_rle: bool, max_examples: int) -> Dict[str, Any]:
    data = load_json(json_path)

    videos: List[Dict[str, Any]] = data.get("videos", [])
    annotations: List[Dict[str, Any]] = data.get("annotations", [])
    categories: List[Dict[str, Any]] = data.get("categories", [])

    findings: List[Dict[str, Any]] = []
    counts: Dict[str, int] = collections.Counter()

    # Index videos
    videos_by_id: Dict[int, Dict[str, Any]] = {}
    anns_by_video: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)

    for v in videos:
        vid = int(v["id"])
        if vid in videos_by_id:
            add_finding(findings, counts, "DUPLICATE_VIDEO_ID", "error", video_id=vid)
        videos_by_id[vid] = v

    for ann in annotations:
        vid = int(ann["video_id"])
        anns_by_video[vid].append(ann)
        if vid not in videos_by_id:
            add_finding(findings, counts, "ANN_REFERENCES_MISSING_VIDEO", "error", ann_id=ann.get("id"), video_id=vid)

    # Basic dataset stats
    summary: Dict[str, Any] = {
        "json_path": str(json_path),
        "datasets_root": str(datasets_root),
        "num_videos": len(videos),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "counts": {},
        "examples": {},
    }

    for v in videos:
        vid = int(v["id"])
        file_names = v.get("file_names", [])
        video_len = int(v.get("length", len(file_names)))
        json_h = v.get("height")
        json_w = v.get("width")

        if len(file_names) != video_len:
            add_finding(
                findings,
                counts,
                "VIDEO_LENGTH_FILENAME_MISMATCH",
                "error",
                video_id=vid,
                video_length=video_len,
                file_names_len=len(file_names),
            )

        if len(set(file_names)) != len(file_names):
            add_finding(
                findings,
                counts,
                "VIDEO_HAS_DUPLICATE_FILENAMES",
                "warn",
                video_id=vid,
                duplicate_count=len(file_names) - len(set(file_names)),
            )

        actual_hw_by_frame: List[Tuple[Optional[int], Optional[int]]] = []
        missing_or_bad_frames = 0
        for frame_idx, rel_name in enumerate(file_names):
            img_path = datasets_root / rel_name
            if not img_path.exists():
                add_finding(
                    findings,
                    counts,
                    "MISSING_IMAGE",
                    "error",
                    video_id=vid,
                    frame_idx=frame_idx,
                    file_name=rel_name,
                )
                actual_hw_by_frame.append((None, None))
                missing_or_bad_frames += 1
                continue
            h, w, err = get_image_size(img_path)
            if err is not None:
                add_finding(
                    findings,
                    counts,
                    "UNREADABLE_IMAGE",
                    "error",
                    video_id=vid,
                    frame_idx=frame_idx,
                    file_name=rel_name,
                    error=err,
                )
                actual_hw_by_frame.append((None, None))
                missing_or_bad_frames += 1
                continue
            actual_hw_by_frame.append((h, w))
            if json_h is not None and json_w is not None and (h, w) != (json_h, json_w):
                add_finding(
                    findings,
                    counts,
                    "IMAGE_SIZE_MISMATCH_JSON_VIDEO_META",
                    "error",
                    video_id=vid,
                    frame_idx=frame_idx,
                    file_name=rel_name,
                    json_hw=[json_h, json_w],
                    actual_hw=[h, w],
                )

        # Audit annotations tied to this video
        active_counts = [0 for _ in range(max(video_len, len(file_names)))]
        anns = anns_by_video.get(vid, [])
        for ann in anns:
            ann_id = ann.get("id")
            segs = ann.get("segmentations", [])
            bboxes = ann.get("bboxes", [])
            cat_id = ann.get("category_id")

            if len(segs) != video_len:
                add_finding(
                    findings,
                    counts,
                    "ANN_SEGMENTATIONS_LEN_MISMATCH_VIDEO_LENGTH",
                    "error",
                    video_id=vid,
                    ann_id=ann_id,
                    category_id=cat_id,
                    video_length=video_len,
                    segmentations_len=len(segs),
                )
            if len(bboxes) != video_len:
                add_finding(
                    findings,
                    counts,
                    "ANN_BBOXES_LEN_MISMATCH_VIDEO_LENGTH",
                    "error",
                    video_id=vid,
                    ann_id=ann_id,
                    category_id=cat_id,
                    video_length=video_len,
                    bboxes_len=len(bboxes),
                )

            per_ann_present = []
            for frame_idx in range(min(video_len, len(segs))):
                seg = segs[frame_idx]
                bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else None
                actual_h, actual_w = actual_hw_by_frame[frame_idx] if frame_idx < len(actual_hw_by_frame) else (None, None)
                is_present = seg is not None and seg != []
                per_ann_present.append(bool(is_present))
                if is_present and frame_idx < len(active_counts):
                    active_counts[frame_idx] += 1

                if bbox is not None and bbox != []:
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        add_finding(
                            findings,
                            counts,
                            "ANN_BBOX_INVALID_FORMAT",
                            "error",
                            video_id=vid,
                            ann_id=ann_id,
                            frame_idx=frame_idx,
                            bbox=bbox,
                        )
                    else:
                        x, y, w_box, h_box = bbox
                        if w_box < 0 or h_box < 0:
                            add_finding(
                                findings,
                                counts,
                                "ANN_BBOX_NEGATIVE_SIZE",
                                "error",
                                video_id=vid,
                                ann_id=ann_id,
                                frame_idx=frame_idx,
                                bbox=bbox,
                            )

                if seg is None or seg == []:
                    continue

                if isinstance(seg, dict):
                    rle_size = seg.get("size")
                    if not (isinstance(rle_size, list) and len(rle_size) == 2):
                        add_finding(
                            findings,
                            counts,
                            "ANN_RLE_INVALID_SIZE_FIELD",
                            "error",
                            video_id=vid,
                            ann_id=ann_id,
                            frame_idx=frame_idx,
                            rle_size=rle_size,
                        )
                    else:
                        if actual_h is not None and actual_w is not None and tuple(rle_size) != (actual_h, actual_w):
                            add_finding(
                                findings,
                                counts,
                                "ANN_RLE_SIZE_MISMATCH_IMAGE",
                                "error",
                                video_id=vid,
                                ann_id=ann_id,
                                frame_idx=frame_idx,
                                rle_size=rle_size,
                                actual_hw=[actual_h, actual_w],
                                file_name=file_names[frame_idx] if frame_idx < len(file_names) else None,
                            )
                        if decode_rle:
                            attempted, decoded_hw, err = maybe_decode_rle(seg)
                            if attempted and err is not None:
                                add_finding(
                                    findings,
                                    counts,
                                    "ANN_RLE_DECODE_FAILED",
                                    "error",
                                    video_id=vid,
                                    ann_id=ann_id,
                                    frame_idx=frame_idx,
                                    error=err,
                                )
                            elif attempted and decoded_hw is not None and tuple(rle_size) != tuple(decoded_hw):
                                add_finding(
                                    findings,
                                    counts,
                                    "ANN_RLE_DECODED_SIZE_MISMATCH_RLE_SIZE",
                                    "error",
                                    video_id=vid,
                                    ann_id=ann_id,
                                    frame_idx=frame_idx,
                                    rle_size=rle_size,
                                    decoded_hw=list(decoded_hw),
                                )
                elif isinstance(seg, list):
                    # polygon(s)
                    if len(seg) == 0:
                        add_finding(
                            findings,
                            counts,
                            "ANN_POLYGON_EMPTY_LIST_PRESENT",
                            "warn",
                            video_id=vid,
                            ann_id=ann_id,
                            frame_idx=frame_idx,
                        )
                    for poly_idx, poly in enumerate(seg):
                        if not hasattr(poly, "__len__"):
                            add_finding(findings, counts, "ANN_POLYGON_INVALID_TYPE", "error", video_id=vid, ann_id=ann_id, frame_idx=frame_idx, polygon_index=poly_idx)
                            continue
                        if len(poly) < 6:
                            add_finding(findings, counts, "ANN_POLYGON_TOO_SHORT", "error", video_id=vid, ann_id=ann_id, frame_idx=frame_idx, polygon_index=poly_idx, poly_len=len(poly))
                        if len(poly) % 2 != 0:
                            add_finding(findings, counts, "ANN_POLYGON_ODD_COORD_COUNT", "error", video_id=vid, ann_id=ann_id, frame_idx=frame_idx, polygon_index=poly_idx, poly_len=len(poly))
                else:
                    add_finding(
                        findings,
                        counts,
                        "ANN_SEGMENTATION_UNKNOWN_TYPE",
                        "error",
                        video_id=vid,
                        ann_id=ann_id,
                        frame_idx=frame_idx,
                        seg_type=type(seg).__name__,
                    )

            # Track presence consistency (useful for official copy_paste assumption)
            if per_ann_present and any(per_ann_present) and not all(per_ann_present):
                add_finding(
                    findings,
                    counts,
                    "ANN_TRACK_PRESENCE_VARIES_ACROSS_FRAMES",
                    "warn",
                    video_id=vid,
                    ann_id=ann_id,
                    category_id=cat_id,
                    present_frames=sum(per_ann_present),
                    video_length=video_len,
                )

        if active_counts and max(active_counts) != min(active_counts):
            add_finding(
                findings,
                counts,
                "VIDEO_ACTIVE_INSTANCE_COUNT_VARIES_ACROSS_FRAMES",
                "warn",
                video_id=vid,
                active_counts=active_counts,
                video_length=video_len,
            )

        if active_counts and active_counts[0] > 0 and any(c == 0 for c in active_counts[1:]):
            add_finding(
                findings,
                counts,
                "VIDEO_FRAME0_NONEMPTY_OTHER_FRAME_EMPTY",
                "warn",
                video_id=vid,
                active_counts=active_counts,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    findings_path = out_dir / "findings.jsonl"
    with findings_path.open("w", encoding="utf-8") as f:
        for item in findings:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary["counts"] = dict(sorted(counts.items()))
    summary["examples"] = summarize_examples(findings, max_examples=max_examples)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# ImageNet Synthetic Video Audit Report\n\n")
        f.write(f"- JSON: `{json_path}`\n")
        f.write(f"- Datasets root: `{datasets_root}`\n")
        f.write(f"- Videos: {summary['num_videos']}\n")
        f.write(f"- Annotations: {summary['num_annotations']}\n")
        f.write(f"- Categories: {summary['num_categories']}\n\n")
        f.write("## Counts by issue kind\n\n")
        if summary["counts"]:
            for kind, n in summary["counts"].items():
                f.write(f"- **{kind}**: {n}\n")
        else:
            f.write("No issues found.\n")
        f.write("\n## Example findings\n\n")
        if summary["examples"]:
            for kind, examples in summary["examples"].items():
                f.write(f"### {kind}\n\n")
                for ex in examples:
                    f.write("```json\n")
                    f.write(json.dumps(ex, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")
        else:
            f.write("No example findings.\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("datasets/imagenet/annotations/video_imagenet_train_fixsize480_tau0.15_N3.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("audit_imagenet_video_full_out"))
    parser.add_argument("--decode-rle", action="store_true")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    summary = audit(args.json, args.datasets_root, args.out_dir, decode_rle=args.decode_rle, max_examples=args.max_examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {args.out_dir / 'summary.json'}")
    print(f"Wrote: {args.out_dir / 'findings.jsonl'}")
    print(f"Wrote: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
