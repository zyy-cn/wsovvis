#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full audit for VideoCutLER synthetic ImageNet video annotations.

Fixes vs previous version:
- image files are resolved against an explicit --image-root
- supports multiple file_name styles:
  * n01440764/n01440764_10026.JPEG
  * train/n01440764/n01440764_10026.JPEG
  * imagenet/train/n01440764/n01440764_10026.JPEG
  * datasets/imagenet/train/n01440764/n01440764_10026.JPEG

Outputs:
- summary.json
- findings.jsonl
- report.md
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=str,
        default="videocutler/datasets/imagenet/annotations/video_imagenet_train_fixsize480_tau0.15_N3.json",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default="videocutler/datasets/imagenet/train",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="audit_imagenet_video_full_out_v2",
    )
    p.add_argument("--decode-rle", action="store_true")
    p.add_argument("--max-findings-per-kind", type=int, default=20)
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_rel_path(file_name: str) -> str:
    s = file_name.replace("\\", "/").lstrip("./")
    prefixes = [
        "datasets/imagenet/train/",
        "imagenet/train/",
        "train/",
    ]
    for pref in prefixes:
        if s.startswith(pref):
            return s[len(pref):]
    return s


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return image_root / normalize_rel_path(file_name)


def read_image_size(path: Path):
    with Image.open(path) as im:
        return im.size[1], im.size[0]  # (H, W)


def add_finding(findings_fh, counter, examples, finding, max_examples):
    kind = finding["kind"]
    counter[kind] += 1
    if len(examples[kind]) < max_examples:
        examples[kind].append(finding)
    findings_fh.write(json.dumps(finding, ensure_ascii=False) + "\n")


def audit_video(
    video,
    anns_by_video,
    image_root: Path,
    findings_fh,
    counts,
    examples,
    max_examples,
    decode_rle=False,
):
    video_id = video["id"]
    file_names = video.get("file_names", [])
    vlen = len(file_names)
    json_h = video.get("height")
    json_w = video.get("width")

    if "length" in video and video["length"] != vlen:
        add_finding(
            findings_fh, counts, examples,
            {
                "kind": "VIDEO_LENGTH_MISMATCH",
                "severity": "error",
                "video_id": video_id,
                "length_field": video["length"],
                "file_names_len": vlen,
            },
            max_examples,
        )

    dup_count = vlen - len(set(file_names))
    if dup_count > 0:
        add_finding(
            findings_fh, counts, examples,
            {
                "kind": "VIDEO_HAS_DUPLICATE_FILENAMES",
                "severity": "warn",
                "video_id": video_id,
                "duplicate_count": dup_count,
            },
            max_examples,
        )

    actual_sizes = []
    for frame_idx, fn in enumerate(file_names):
        img_path = resolve_image_path(image_root, fn)
        if not img_path.exists():
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "MISSING_IMAGE",
                    "severity": "error",
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "file_name": fn,
                    "resolved_path": str(img_path),
                },
                max_examples,
            )
            actual_sizes.append(None)
            continue

        try:
            h, w = read_image_size(img_path)
            actual_sizes.append((h, w))
            if json_h is not None and json_w is not None and (h, w) != (json_h, json_w):
                add_finding(
                    findings_fh, counts, examples,
                    {
                        "kind": "IMAGE_SHAPE_MISMATCH",
                        "severity": "error",
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "file_name": fn,
                        "resolved_path": str(img_path),
                        "json_hw": [json_h, json_w],
                        "actual_hw": [h, w],
                    },
                    max_examples,
                )
        except Exception as e:
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "IMAGE_READ_ERROR",
                    "severity": "error",
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "file_name": fn,
                    "resolved_path": str(img_path),
                    "error": repr(e),
                },
                max_examples,
            )
            actual_sizes.append(None)

    anns = anns_by_video.get(video_id, [])
    frame_valid_masks = [0] * vlen

    for ann in anns:
        ann_id = ann.get("id")
        segs = ann.get("segmentations", [])
        bboxes = ann.get("bboxes", [])

        if len(segs) != vlen:
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "SEGMENTATIONS_LENGTH_MISMATCH",
                    "severity": "error",
                    "video_id": video_id,
                    "ann_id": ann_id,
                    "segmentations_len": len(segs),
                    "video_length": vlen,
                },
                max_examples,
            )

        if len(bboxes) != vlen:
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "BBOXES_LENGTH_MISMATCH",
                    "severity": "error",
                    "video_id": video_id,
                    "ann_id": ann_id,
                    "bboxes_len": len(bboxes),
                    "video_length": vlen,
                },
                max_examples,
            )

        presence = []
        for frame_idx in range(min(vlen, len(segs))):
            seg = segs[frame_idx]
            present = seg is not None
            presence.append(bool(present))

            if seg is None:
                continue

            if isinstance(seg, dict):
                size = seg.get("size", None)
                counts_field = seg.get("counts", None)

                if size is None or len(size) != 2:
                    add_finding(
                        findings_fh, counts, examples,
                        {
                            "kind": "BAD_RLE_META",
                            "severity": "error",
                            "video_id": video_id,
                            "ann_id": ann_id,
                            "frame_idx": frame_idx,
                            "seg_size": size,
                            "counts_type": type(counts_field).__name__,
                        },
                        max_examples,
                    )
                    continue

                actual_hw = actual_sizes[frame_idx]
                if actual_hw is not None and tuple(size) != tuple(actual_hw):
                    add_finding(
                        findings_fh, counts, examples,
                        {
                            "kind": "BAD_RLE_SIZE",
                            "severity": "error",
                            "video_id": video_id,
                            "ann_id": ann_id,
                            "frame_idx": frame_idx,
                            "seg_size": list(size),
                            "actual_hw": list(actual_hw),
                            "file_name": file_names[frame_idx],
                        },
                        max_examples,
                    )

                frame_valid_masks[frame_idx] += 1

                if decode_rle and mask_utils is not None:
                    try:
                        decoded = mask_utils.decode(seg)
                        dh, dw = decoded.shape[:2]
                        actual_hw = actual_sizes[frame_idx]
                        if actual_hw is not None and (dh, dw) != tuple(actual_hw):
                            add_finding(
                                findings_fh, counts, examples,
                                {
                                    "kind": "DECODED_RLE_SHAPE_MISMATCH",
                                    "severity": "error",
                                    "video_id": video_id,
                                    "ann_id": ann_id,
                                    "frame_idx": frame_idx,
                                    "decoded_hw": [dh, dw],
                                    "actual_hw": list(actual_hw),
                                    "file_name": file_names[frame_idx],
                                },
                                max_examples,
                            )
                    except Exception as e:
                        add_finding(
                            findings_fh, counts, examples,
                            {
                                "kind": "RLE_DECODE_ERROR",
                                "severity": "error",
                                "video_id": video_id,
                                "ann_id": ann_id,
                                "frame_idx": frame_idx,
                                "error": repr(e),
                            },
                            max_examples,
                        )

            elif isinstance(seg, list):
                if len(seg) == 0:
                    add_finding(
                        findings_fh, counts, examples,
                        {
                            "kind": "EMPTY_POLYGON",
                            "severity": "warn",
                            "video_id": video_id,
                            "ann_id": ann_id,
                            "frame_idx": frame_idx,
                        },
                        max_examples,
                    )
                else:
                    frame_valid_masks[frame_idx] += 1
                    for poly_idx, poly in enumerate(seg):
                        if not hasattr(poly, "__len__") or len(poly) < 6 or (len(poly) % 2 != 0):
                            add_finding(
                                findings_fh, counts, examples,
                                {
                                    "kind": "BAD_POLYGON",
                                    "severity": "error",
                                    "video_id": video_id,
                                    "ann_id": ann_id,
                                    "frame_idx": frame_idx,
                                    "poly_idx": poly_idx,
                                    "poly_len": len(poly) if hasattr(poly, "__len__") else None,
                                },
                                max_examples,
                            )
            else:
                add_finding(
                    findings_fh, counts, examples,
                    {
                        "kind": "UNKNOWN_SEGMENTATION_TYPE",
                        "severity": "error",
                        "video_id": video_id,
                        "ann_id": ann_id,
                        "frame_idx": frame_idx,
                        "seg_type": type(seg).__name__,
                    },
                    max_examples,
                )

        if presence and (any(presence) and not all(presence)):
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "TRACK_PRESENCE_VARIES",
                    "severity": "warn",
                    "video_id": video_id,
                    "ann_id": ann_id,
                    "presence": presence,
                },
                max_examples,
            )

    if vlen > 0:
        nonzero = [x for x in frame_valid_masks if x > 0]
        if len(nonzero) > 1 and min(nonzero) != max(nonzero):
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "FRAME_VALID_INSTANCE_COUNT_VARIES",
                    "severity": "warn",
                    "video_id": video_id,
                    "frame_valid_masks": frame_valid_masks,
                },
                max_examples,
            )

        if frame_valid_masks and frame_valid_masks[0] > 0 and any(x == 0 for x in frame_valid_masks[1:]):
            add_finding(
                findings_fh, counts, examples,
                {
                    "kind": "FRAME0_NONEMPTY_OTHERS_EMPTY",
                    "severity": "warn",
                    "video_id": video_id,
                    "frame_valid_masks": frame_valid_masks,
                },
                max_examples,
            )


def main():
    args = parse_args()
    json_path = Path(args.json)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    anns_by_video = defaultdict(list)
    for ann in annotations:
        anns_by_video[ann["video_id"]].append(ann)

    counts = Counter()
    examples = defaultdict(list)

    findings_path = out_dir / "findings.jsonl"
    with open(findings_path, "w", encoding="utf-8") as findings_fh:
        for i, video in enumerate(videos):
            audit_video(
                video=video,
                anns_by_video=anns_by_video,
                image_root=image_root,
                findings_fh=findings_fh,
                counts=counts,
                examples=examples,
                max_examples=args.max_findings_per_kind,
                decode_rle=args.decode_rle,
            )
            if (i + 1) % 5000 == 0:
                print(f"[audit] processed {i + 1}/{len(videos)} videos", flush=True)

    summary = {
        "json_path": str(json_path),
        "image_root": str(image_root),
        "num_videos": len(videos),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "counts": dict(counts),
        "examples": dict(examples),
    }
    write_json(summary, out_dir / "summary.json")

    lines = []
    lines.append("# ImageNet Synthetic Video Audit Report")
    lines.append("")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- Image root: `{image_root}`")
    lines.append(f"- Videos: {len(videos)}")
    lines.append(f"- Annotations: {len(annotations)}")
    lines.append(f"- Categories: {len(categories)}")
    lines.append("")
    lines.append("## Counts by issue kind")
    lines.append("")
    if counts:
        for k, v in counts.most_common():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- No findings")
    lines.append("")
    lines.append("## Example findings")
    lines.append("")
    for k, exs in examples.items():
        lines.append(f"### {k}")
        lines.append("")
        for ex in exs:
            lines.append("```json")
            lines.append(json.dumps(ex, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] report written to: {out_dir}", flush=True)
    print(f"[done] summary: {out_dir / 'summary.json'}", flush=True)
    print(f"[done] findings: {out_dir / 'findings.jsonl'}", flush=True)
    print(f"[done] markdown: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
