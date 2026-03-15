#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torch.cuda.amp import autocast


def _load_video_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None, env=env)


def _load_existing_empty_videos(path: Path | None) -> set[str]:
    if path is None or (not path.is_file()):
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _write_empty_videos(path: Path | None, names: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{name}\n" for name in sorted(names))
    path.write_text(payload, encoding="utf-8")


def _save_masks(masks: list[np.ndarray], file_path: Path) -> bool:
    if not masks:
        return False
    mask_image = np.zeros(np.asarray(masks[0], dtype=np.uint8).shape, dtype=np.uint8)
    for index, mask in enumerate(masks, start=1):
        arr = np.asarray(mask, dtype=np.uint8)
        mask_image[arr != 0] = index
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]
    image = Image.fromarray(mask_image, mode="P")
    image.putpalette(palette)
    image.save(file_path)
    return True


def _is_video_complete(output_dir: Path, frame_count: int, known_empty: set[str], video_name: str) -> bool:
    if video_name in known_empty:
        return True
    if not output_dir.is_dir():
        return False
    mask_count = len(list(output_dir.glob("mask_*.png")))
    return frame_count > 0 and mask_count == frame_count


def _build_cutler_predictor(
    *,
    repo_root: Path,
    config_file: Path,
    weights: Path,
):
    cutler_root = (repo_root / "third_party" / "CutLER" / "videocutler").resolve()
    demo_root = cutler_root / "demo_video"
    for path in (demo_root, cutler_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from mask2former_video import add_maskformer2_video_config
    from predictor import VideoPredictor

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(str(config_file.resolve()))
    cfg.merge_from_list(["MODEL.WEIGHTS", str(weights.resolve())])
    cfg.freeze()
    return VideoPredictor(cfg)


def _read_frames(
    *,
    repo_root: Path,
    frame_paths: list[Path],
):
    cutler_root = (repo_root / "third_party" / "CutLER" / "videocutler").resolve()
    demo_root = cutler_root / "demo_video"
    for path in (demo_root, cutler_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from detectron2.data.detection_utils import read_image

    return [read_image(str(path), format="BGR") for path in frame_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VideoCutLER on a bounded subset and convert masks to YTVIS JSON.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing <video_name>/*.jpg")
    parser.add_argument("--video-names-file", type=Path, required=True)
    parser.add_argument("--output-mask-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--split-name", default="routea_subset")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("third_party/CutLER/videocutler/configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("third_party/CutLER/videocutler/pretrain/videocutler_m2f_rn50.pth"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="Only generate mask PNGs; skip JSON conversion so callers can shard generation safely.",
    )
    parser.add_argument(
        "--empty-videos-file",
        type=Path,
        default=None,
        help="Optional empty-video manifest path. Defaults to <output-mask-root>/empty_videos.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    output_mask_root = args.output_mask_root.resolve()
    output_json = args.output_json.resolve()
    output_mask_root.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    video_names = _load_video_names(args.video_names_file)
    if not video_names:
        raise ValueError("video_names_file is empty")

    empty_videos_file = args.empty_videos_file.resolve() if args.empty_videos_file else (output_mask_root / "empty_videos.txt")
    known_empty = _load_existing_empty_videos(empty_videos_file)
    predictor = _build_cutler_predictor(
        repo_root=repo_root,
        config_file=args.config_file,
        weights=args.weights,
    )

    for index, video_name in enumerate(video_names, start=1):
        output_dir = output_mask_root / video_name
        frame_paths = sorted((data_root / video_name).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"no frames found for video {video_name}: {data_root / video_name}")
        if args.skip_existing and _is_video_complete(output_dir, len(frame_paths), known_empty, video_name):
            print(f"[skip {index}/{len(video_names)}] {video_name}")
            continue
        print(f"[run {index}/{len(video_names)}] {video_name} conf={args.confidence_threshold:.2f}")
        vid_frames = _read_frames(repo_root=repo_root, frame_paths=frame_paths)
        with autocast():
            predictions = predictor(vid_frames)
        selected_idx = [
            idx for idx, score in enumerate(predictions["pred_scores"])
            if score >= args.confidence_threshold
        ]
        pred_masks = [predictions["pred_masks"][idx] for idx in selected_idx]

        written_any = False
        if pred_masks:
            output_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, frame_path in enumerate(frame_paths):
                frame_masks = [
                    pred_masks[inst_id][frame_index].detach().cpu().numpy()
                    for inst_id in range(len(pred_masks))
                ]
                out_mask_path = output_dir / f"mask_{frame_path.stem}.png"
                written_any = _save_masks(frame_masks, out_mask_path) or written_any

        if written_any:
            known_empty.discard(video_name)
        else:
            known_empty.add(video_name)

    _write_empty_videos(empty_videos_file, known_empty)

    if not args.masks_only:
        _run(
            [
                sys.executable,
                str((repo_root / "tools" / "convert_videocutler_png_to_json.py").resolve()),
                "--mask_root",
                str(output_mask_root),
                "--img_root",
                str(data_root),
                "--out_json",
                str(output_json),
                "--split_name",
                args.split_name,
            ],
            cwd=repo_root,
            env=env,
        )
        print(f"[done] {output_json}")
    else:
        print(f"[done] masks-only {output_mask_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
