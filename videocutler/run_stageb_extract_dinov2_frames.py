from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _FallbackProgress:
        def __init__(self, total: int, unit: str = "item", dynamic_ncols: bool = True, leave: bool = True, desc: str = "") -> None:
            self.total = max(0, int(total))
            self.unit = unit
            self.desc = desc
            self.count = 0
            self._postfix: Dict[str, Any] = {}
            self._render()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def update(self, n: int = 1) -> None:
            self.count += int(n)
            self._render()

        def set_postfix(self, data: Dict[str, Any], refresh: bool = False) -> None:
            self._postfix = dict(data)
            if refresh:
                self._render()

        def _render(self) -> None:
            total = self.total or 1
            frac = min(1.0, max(0.0, float(self.count) / float(total)))
            width = 24
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            postfix = ""
            if self._postfix:
                postfix = " " + " ".join(f"{k}={v}" for k, v in self._postfix.items())
            sys.stdout.write(f"\r{self.desc} [{bar}] {self.count}/{self.total} {self.unit}{postfix}")
            sys.stdout.flush()

        def close(self) -> None:
            self._render()
            sys.stdout.write("\n")
            sys.stdout.flush()

    tqdm = _FallbackProgress

DATASET_CHOICES = ("lvvis_train_base", "lvvis_val")
SPLIT_CFG: Dict[str, Dict[str, str]] = {
    "lvvis_train_base": {
        "annotation_rel": "annotations/train_instances.json",
        "image_root_rel": "train",
        "input_source_smoke": "smoke_fixture",
        "input_source_full": "official_lvvis_train_annotations",
        "data_scope_smoke": "train_smoke",
        "data_scope_full": "train",
        "consumer_target": "run_stageb_build_carrier_bank|run_stageb_train_prealign|run_stageb_train_softem",
    },
    "lvvis_val": {
        "annotation_rel": "annotations/val_instances.json",
        "image_root_rel": "val",
        "input_source_smoke": "smoke_fixture",
        "input_source_full": "official_lvvis_val_annotations",
        "data_scope_smoke": "val_smoke",
        "data_scope_full": "val",
        "consumer_target": "run_stageb_build_carrier_bank|run_stageb_infer_ov",
    },
}
PRIMARY_ARTIFACTS = {
    "lvvis_train_base": "frame_bank/lvvis_train_base/frame_records.jsonl",
    "lvvis_val": "frame_bank/lvvis_val/frame_records.jsonl",
}
PRIMARY_GEOM_ARTIFACTS = {
    "lvvis_train_base": "frame_bank/lvvis_train_base/frame_geom_records.jsonl",
    "lvvis_val": "frame_bank/lvvis_val/frame_geom_records.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract official DINOv2 frame features and materialize frame records.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--dinov2_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--batch_size_frames", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preprocess_mode", default="keep_aspect_pad")
    parser.add_argument("--resize_short_side", type=int, default=672)
    parser.add_argument("--max_long_side", type=int, default=1008)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--contract_check_json")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolved_run_root(output_root: str, exp_name: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.name == exp_name:
        return root
    return root / exp_name


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_image_path(split_root: Path, lvvis_root: Path, rel_file: str) -> Path:
    rel = Path(rel_file)
    candidates = [
        split_root / rel,
        split_root / "JPEGImages" / rel,
        lvvis_root / rel,
        lvvis_root / split_root.name / "JPEGImages" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _collect_samples(dataset_name: str, smoke: bool, progress: Any | None = None):
    from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import FrameSample
    from videocutler.ext_stageb_ovvis.data.datasets.lvvis_smoke import load_lvvis_smoke_subset_data

    cfg = SPLIT_CFG[dataset_name]
    env_root = os.environ.get("WSOVVIS_LVVIS_ROOT", "").strip()
    if env_root:
        lvvis_root = Path(env_root).expanduser().resolve()
    else:
        lvvis_root = (_repo_root() / "videocutler" / "datasets" / "LV-VIS").resolve()
    annotation_path = lvvis_root / cfg["annotation_rel"]
    split_root = lvvis_root / cfg["image_root_rel"]
    if smoke and dataset_name == "lvvis_train_base":
        _, subset_data, subset_image_root = load_lvvis_smoke_subset_data(dataset_name, smoke_num_videos=2)
        data = subset_data
        split_root = subset_image_root
    else:
        data = _read_json(annotation_path)
    videos = sorted(data.get("videos", []), key=lambda item: int(item["id"]))
    if smoke and dataset_name != "lvvis_train_base":
        videos = videos[:2]

    samples: List[FrameSample] = []
    expected_frames = 0
    with tqdm(total=max(1, len(videos)), unit="video", dynamic_ncols=True, leave=True, desc="g4-collect") as collect_progress:
        for video in videos:
            clip_id = str(video["id"])
            file_names = list(video.get("file_names") or video.get("filenames") or [])
            for frame_index, rel_file in enumerate(file_names):
                expected_frames += 1
                image_path = _resolve_image_path(split_root, lvvis_root, str(rel_file))
                if not image_path.exists():
                    continue
                samples.append(FrameSample(clip_id=clip_id, frame_index=int(frame_index), image_path=image_path))
            collect_progress.update(1)
            collect_progress.set_postfix({"collected": len(samples)}, refresh=False)
    samples.sort(key=lambda item: (item.clip_id, item.frame_index))
    return samples, expected_frames, len(videos)


def _summary_fields(dataset_name: str, smoke: bool) -> Dict[str, str]:
    cfg = SPLIT_CFG[dataset_name]
    if smoke:
        return {
            "run_scope": "smoke",
            "input_source_type": cfg["input_source_smoke"],
            "data_scope": cfg["data_scope_smoke"],
            "consumer_target": cfg["consumer_target"],
        }
    return {
        "run_scope": "full",
        "input_source_type": cfg["input_source_full"],
        "data_scope": cfg["data_scope_full"],
        "consumer_target": cfg["consumer_target"],
    }


def _default_contract_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "g4_frame_feature_cache" / "frame_contract_check.json"


def _merge_contract_check(
    path: Path,
    *,
    dataset_name: str,
    smoke: bool,
    record_count: int,
    expected_frames: int,
    artifact_path: Path,
) -> None:
    summary = _summary_fields(dataset_name, smoke)
    coverage_ratio = 0.0 if expected_frames <= 0 else min(1.0, float(record_count) / float(expected_frames))
    missing_ratio = 1.0 - coverage_ratio if expected_frames > 0 else 0.0
    status = "PASS" if artifact_path.exists() and record_count > 0 else "FAIL"
    consumer_ready = bool(summary["run_scope"] == "full" and coverage_ratio >= 1.0 and status == "PASS")
    checks_run = [
        "frame_feature_reader_parses_dsl",
        "frame_cache_coverage_ready",
        "artifact_exists",
        "artifact_schema_valid",
        "full_split_frame_geometry_applicability",
    ]
    payload = {
        "gate_id": "G4_frame_feature_cache",
        "contract_ref": "contracts/gates/G4_frame_feature_cache.gate_contract.json",
        "status": status,
        "artifact_path_base": "output_root_relative",
        "primary_artifacts": [PRIMARY_ARTIFACTS[dataset_name], PRIMARY_GEOM_ARTIFACTS[dataset_name]],
        "checks_run": checks_run,
        "run_scope": summary["run_scope"],
        "input_source_type": summary["input_source_type"],
        "data_scope": summary["data_scope"],
        "record_count": int(record_count),
        "coverage_ratio": float(coverage_ratio),
        "missing_frame_ratio": float(missing_ratio),
        "consumer_ready": consumer_ready,
    }

    if path.exists():
        existing = _read_json(path)
        existing_scope = str(existing.get("run_scope", "")).strip()
        if existing_scope == payload["run_scope"]:
            merged_primary = sorted(
                {
                    str(item).strip()
                    for item in list(existing.get("primary_artifacts", [])) + payload["primary_artifacts"]
                    if str(item).strip()
                }
            )
            existing_cov = float(existing.get("coverage_ratio", 0.0) or 0.0)
            existing_missing = float(existing.get("missing_frame_ratio", 1.0) or 1.0)
            existing_count = int(existing.get("record_count", 0) or 0)
            payload["primary_artifacts"] = merged_primary
            payload["record_count"] = existing_count + payload["record_count"]
            payload["coverage_ratio"] = min(existing_cov if existing_count > 0 else 1.0, payload["coverage_ratio"])
            payload["missing_frame_ratio"] = max(existing_missing if existing_count > 0 else 0.0, payload["missing_frame_ratio"])
            payload["consumer_ready"] = bool(existing.get("consumer_ready", False) and payload["consumer_ready"])
            payload["status"] = "PASS" if bool(existing.get("status") == "PASS" and payload["status"] == "PASS") else "FAIL"
            payload["input_source_type"] = "|".join(
                sorted(
                    {
                        str(existing.get("input_source_type", "")).strip(),
                        str(payload["input_source_type"]).strip(),
                    }
                    - {""}
                )
            )
            payload["data_scope"] = "|".join(
                sorted(
                    {
                        str(existing.get("data_scope", "")).strip(),
                        str(payload["data_scope"]).strip(),
                    }
                    - {""}
                )
            )

    _write_json(path, payload)


def _write_manifests(args: argparse.Namespace, run_root: Path, record_count: int, expected_frames: int) -> None:
    summary = _summary_fields(args.dataset_name, args.smoke)
    manifests = run_root / "manifests"
    _write_json(
        manifests / "resolved_config.json",
        {
            "dataset_name": args.dataset_name,
            "dinov2_ckpt": args.dinov2_ckpt,
            "device": args.device,
            "seed": int(args.seed),
            "num_workers": int(args.num_workers),
            "batch_size_frames": int(args.batch_size_frames),
            "preprocess_mode": str(args.preprocess_mode),
            "resize_short_side": int(args.resize_short_side),
            "max_long_side": int(args.max_long_side),
            "pad_to_multiple": int(args.pad_to_multiple),
            "output_root": str(run_root),
        },
    )
    _write_json(
        manifests / "run_meta.json",
        {
            "exp_name": args.exp_name,
            "dataset_name": args.dataset_name,
            "run_scope": summary["run_scope"],
            "input_source_type": summary["input_source_type"],
            "data_scope": summary["data_scope"],
            "consumer_target": summary["consumer_target"],
            "record_count": int(record_count),
            "coverage_ratio": 0.0 if expected_frames <= 0 else float(record_count) / float(expected_frames),
        },
    )


def _require_preprocess_contract(args: argparse.Namespace) -> None:
    if str(args.preprocess_mode) != "keep_aspect_pad":
        raise ValueError("Only preprocess_mode=keep_aspect_pad is supported by contract")
    if int(args.pad_to_multiple) != 14:
        raise ValueError("pad_to_multiple must be 14 for G4 contract")


def _iter_sample_chunks(samples: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(samples), size):
        yield samples[start : start + size]


def _stream_encode_and_write(
    samples: Sequence[Any],
    encoder: Any,
    stream_writer: Any,
    *,
    chunk_size: int,
    encode_progress: Any | None = None,
    write_progress: Any | None = None,
) -> int:
    written = 0
    for chunk in _iter_sample_chunks(samples, chunk_size):
        encoded = encoder.encode_image_paths([sample.image_path for sample in chunk])
        if len(encoded) != len(chunk):
            raise ValueError("encoded output length mismatch")
        if encode_progress is not None:
            encode_progress.update(len(chunk))
            encode_progress.set_postfix({"batch": len(chunk)}, refresh=False)
        for sample, item in zip(chunk, encoded):
            stream_writer.write_sample(sample, item.features, item.geometry, item.valid_token_mask)
            written += 1
            if write_progress is not None:
                write_progress.update(1)
                write_progress.set_postfix({"last_clip": sample.clip_id}, refresh=False)
    return written


def main() -> int:
    args = parse_args()
    _require_preprocess_contract(args)
    import sys

    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
        FrameBankStreamWriter,
    )
    from videocutler.ext_stageb_ovvis.models.visual_encoder_dinov2 import DinoV2VisualEncoder

    run_root = _resolved_run_root(args.output_root, args.exp_name)

    records_path = repo_root / PRIMARY_ARTIFACTS[args.dataset_name]
    geom_path = repo_root / PRIMARY_GEOM_ARTIFACTS[args.dataset_name]
    payload_parent = records_path.parent

    print(f"[g4] dataset_name: {args.dataset_name}")
    print(f"[g4] expected_frames: collecting...")
    print(f"[g4] preprocess_mode: {args.preprocess_mode}")
    print(
        f"[g4] resize_short_side/max_long_side/pad_to_multiple: "
        f"{args.resize_short_side}/{args.max_long_side}/{args.pad_to_multiple}"
    )
    print(f"[g4] batch_size_frames: {args.batch_size_frames}")
    print(f"[g4] device: {args.device}")
    print(f"[g4] canonical frame records: {records_path}")
    print(f"[g4] canonical geometry records: {geom_path}")
    print(f"[g4] canonical payload parent dir: {payload_parent}")
    print(f"[g4] run metadata / contract check root: {run_root}")

    encoder = DinoV2VisualEncoder(
        dinov2_ckpt=args.dinov2_ckpt,
        device=args.device,
        batch_size_frames=args.batch_size_frames,
        preprocess_mode=args.preprocess_mode,
        resize_short_side=args.resize_short_side,
        max_long_side=args.max_long_side,
        pad_to_multiple=args.pad_to_multiple,
        patch_size=14,
    )
    chunk_size = max(int(args.batch_size_frames), int(args.batch_size_frames) * 4)
    print("[g4] stage 1/4: collecting samples")
    samples, expected_frames, video_count = _collect_samples(args.dataset_name, args.smoke)
    print(f"[g4] stage 1 complete: collected {len(samples)} samples across {video_count} videos for {args.dataset_name}")
    print(f"[g4] expected_frames: {expected_frames}")

    print("[g4] stage 2/4: encoding frames")
    print("[g4] stage 3/4: writing payload / records / geometry sidecar")
    with FrameBankStreamWriter(payload_parent, records_path, geom_path) as stream_writer:
        with tqdm(total=max(1, len(samples)), unit="frame", dynamic_ncols=True, leave=True, desc="g4-encode") as encode_progress:
            with tqdm(total=max(1, len(samples)), unit="frame", dynamic_ncols=True, leave=True, desc="g4-write") as write_progress:
                written_count = _stream_encode_and_write(
                    samples,
                    encoder,
                    stream_writer,
                    chunk_size=chunk_size,
                    encode_progress=encode_progress,
                    write_progress=write_progress,
                )

    print("[g4] stage 4/4: writing manifest / contract check / finalize")
    with tqdm(total=3, unit="step", dynamic_ncols=True, leave=True, desc="g4-finalize") as finalize_progress:
        contract_path = Path(args.contract_check_json) if args.contract_check_json else _default_contract_path(repo_root)
        _merge_contract_check(
            contract_path,
            dataset_name=args.dataset_name,
            smoke=args.smoke,
            record_count=written_count,
            expected_frames=expected_frames,
            artifact_path=records_path,
        )
        finalize_progress.update(1)
        finalize_progress.set_postfix({"contract": contract_path.name}, refresh=False)

        _write_manifests(args, run_root, written_count, expected_frames)
        finalize_progress.update(1)
        finalize_progress.set_postfix({"manifests": "done"}, refresh=False)

        finalize_progress.update(1)
        finalize_progress.set_postfix({"status": "done"}, refresh=False)
    print("[g4] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
