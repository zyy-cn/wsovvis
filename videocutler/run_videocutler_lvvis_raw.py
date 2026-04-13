from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


DATASET_CHOICES = ("lvvis_train_base", "lvvis_val")
DEFAULT_CONFIG_REL = "videocutler/configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml"
DEFAULT_CKPT_REL = "weights/videocutler_m2f_rn50.pth"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper to run official VideoCutLER R50 eval-only inference on LV-VIS.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=DATASET_CHOICES,
        help="LV-VIS split to run.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=DEFAULT_CKPT_REL,
        help="Checkpoint path (must resolve under repo/weights/).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root directory; final run dir is output_dir/<dataset>_videocutler_r50.",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for official launcher.")
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:29500",
        help="Distributed init URL passed to official launcher.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Print assembled official invocation details and exit without running inference.",
    )
    parser.add_argument(
        "--smoke_num_videos",
        type=int,
        default=0,
        help="When positive, register a temporary private smoke alias using the first N videos and run that alias.",
    )
    return parser.parse_args()


def _resolve_ckpt(ckpt_path: str, repo_root: Path, *, require_exists: bool) -> Path:
    weights_root = (repo_root / "weights").resolve()
    candidate = Path(ckpt_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).absolute()
    else:
        candidate = candidate.absolute()

    try:
        candidate.relative_to(weights_root)
    except ValueError as exc:
        raise FileNotFoundError(
            f"checkpoint must be under repo weights dir: {weights_root}; got: {candidate}"
        ) from exc

    if require_exists and not candidate.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {candidate}. Expected default: {repo_root / DEFAULT_CKPT_REL}"
        )
    return candidate


def _build_official_args(
    *,
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    dataset_name: str,
    test_dataset: str,
    num_gpus: int,
    dist_url: str,
) -> tuple[SimpleNamespace, str]:
    opts = [
        "MODEL.WEIGHTS",
        str(ckpt_path),
        "OUTPUT_DIR",
        str(output_dir),
    ]
    args = SimpleNamespace(
        config_file=str(config_path),
        eval_only=True,
        resume=False,
        num_gpus=int(num_gpus),
        num_machines=1,
        machine_rank=0,
        dist_url=str(dist_url),
        opts=opts,
        test_dataset=str(test_dataset),
        train_dataset="",
        steps=0,
        wandb_name="",
    )
    cmd = (
        "python videocutler/train_net_video.py "
        f"--config-file {config_path} "
        f"--eval-only --num-gpus {num_gpus} --dist-url {dist_url} "
        f"MODEL.WEIGHTS {ckpt_path} OUTPUT_DIR {output_dir}"
    )
    return args, cmd


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config_path = (repo_root / DEFAULT_CONFIG_REL).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"official config missing: {config_path}")

    ckpt_path = _resolve_ckpt(args.ckpt_path, repo_root, require_exists=not args.print_only)
    run_name = f"{args.dataset_name}_videocutler_r50"
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "WSOVVIS_LVVIS_SANITIZED_ROOT",
        str(output_dir / "_lvvis_sanitized_frames"),
    )

    test_dataset = args.dataset_name
    smoke_annotation_path = None
    smoke_image_root = None
    if args.smoke_num_videos > 0:
        smoke_root = output_dir / "_smoke_subset"
        if args.dataset_name not in DATASET_CHOICES:
            raise ValueError(f"unsupported dataset for smoke subset: {args.dataset_name}")
        if args.print_only:
            test_dataset = f"{args.dataset_name}_smoke_first{int(args.smoke_num_videos)}"
            smoke_annotation_path = smoke_root / "annotations" / f"{'train' if args.dataset_name == 'lvvis_train_base' else 'val'}_instances_smoke_first{int(args.smoke_num_videos)}.json"
            from videocutler.ext_stageb_ovvis.data.datasets.lvvis import resolve_lvvis_root, _SPLITS

            _, _, image_rel = _SPLITS[args.dataset_name]
            smoke_image_root = resolve_lvvis_root() / image_rel
        else:
            from videocutler.ext_stageb_ovvis.data.datasets.lvvis_smoke import register_lvvis_smoke_subset

            test_dataset, smoke_annotation_path, _ = register_lvvis_smoke_subset(
                args.dataset_name,
                int(args.smoke_num_videos),
                smoke_root,
            )

    if not args.print_only:
        from videocutler.ext_stageb_ovvis.data.datasets.lvvis import (
            _SPLITS,
            resolve_lvvis_root,
            summarize_lvvis_sanitization,
        )

        if smoke_annotation_path is not None:
            _, _, image_rel = _SPLITS[args.dataset_name]
            image_root = smoke_image_root or (resolve_lvvis_root() / image_rel)
            summary_json = smoke_annotation_path
        else:
            _, annotation_rel, image_rel = _SPLITS[args.dataset_name]
            image_root = resolve_lvvis_root() / image_rel
            summary_json = resolve_lvvis_root() / annotation_rel

        summary = summarize_lvvis_sanitization(str(summary_json), str(image_root), test_dataset)
        print(
            "[lvvis_raw] lvvis_sanitization_summary: "
            f"videos_checked={summary['videos_checked']} "
            f"videos_sanitized={summary['videos_sanitized']}"
        )
        for sample in summary["mismatch_samples"][:3]:
            print(
                "[lvvis_raw] lvvis_mismatch_sample: "
                f"video_id={sample['video_id']} "
                f"orig=({sample['orig_annot_height']},{sample['orig_annot_width']}) "
                f"actual=({sample['actual_height']},{sample['actual_width']}) "
                f"frame={sample['first_frame']}"
            )

    official_args, cmd = _build_official_args(
        config_path=config_path,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        test_dataset=test_dataset,
        num_gpus=args.num_gpus,
        dist_url=args.dist_url,
    )

    print(f"[lvvis_raw] dataset_name: {args.dataset_name}")
    print(f"[lvvis_raw] config_file: {config_path}")
    print(f"[lvvis_raw] checkpoint: {ckpt_path}")
    print(f"[lvvis_raw] output_dir: {output_dir}")
    if smoke_annotation_path is not None:
        print(f"[lvvis_raw] smoke_alias: {test_dataset}")
        print(f"[lvvis_raw] smoke_annotation_path: {smoke_annotation_path}")
    print(f"[lvvis_raw] expected_results: {output_dir / 'inference' / 'results.json'}")
    print(f"[lvvis_raw] official_eval_command: {cmd}")
    if not ckpt_path.exists():
        print(
            "[lvvis_raw] warning: checkpoint does not exist in print-only mode. "
            "Real execution requires an existing checkpoint under repo/weights/."
        )

    if args.print_only:
        print("[lvvis_raw] print_only=true; skipping official launch")
        return 0

    # Import official runner only for real execution.
    import videocutler.ext_stageb_ovvis.data.datasets.lvvis  # noqa: F401
    import videocutler.train_net_video as train_net_video

    train_net_video.launch(
        train_net_video.main,
        official_args.num_gpus,
        num_machines=official_args.num_machines,
        machine_rank=official_args.machine_rank,
        dist_url=official_args.dist_url,
        args=(official_args,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
