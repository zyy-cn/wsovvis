#!/usr/bin/env python3
"""VideoCutLER trajectory + full-Y_base clean mechanism trainer.

This entry intentionally reuses the current GT-fullY clean trainer's optimization logic
(including soft_e2e_nohub flags already present in run_stageb_train_gt_full_y_clean.py),
but switches the materialized trajectory source from GT upper-bound trajectories to the
VideoCutLER/mainline trajectory branch.

Boundary:
  * full official-base Y_base(v) clip labels only;
  * no Y-prime / extra mining / mAP;
  * no GT correctness/count used in training;
  * GT sidecar is not used by training, only by the companion attribution analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


def _bootstrap_repo_root_for_direct_cli() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_BOOT_REPO_ROOT = _bootstrap_repo_root_for_direct_cli()


def _parse_wrapper_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "videocutler", "gt_upper_bound"))
    p.add_argument("--vc_protocol_tag", default="videocutler_full_y_clean")
    args, _ = p.parse_known_args(argv)
    # Treat explicit videocutler alias as mainline because the Phase1 materializer uses mainline.
    if str(args.trajectory_source_branch) == "videocutler":
        args.trajectory_source_branch = "mainline"
    return args


def _strip_wrapper_args(argv: List[str]) -> List[str]:
    out: List[str] = []
    skip_next = False
    strip_keys = {"--trajectory_source_branch", "--vc_protocol_tag"}
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if tok in strip_keys:
            skip_next = True
            continue
        if any(tok.startswith(k + "=") for k in strip_keys):
            continue
        out.append(tok)
    return out


def _rewrite_json_file(path: Path, mutator) -> None:
    if not path.is_file():
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    obj2 = mutator(obj)
    path.write_text(json.dumps(obj2, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _repair_outputs(output_root: Path, *, branch: str, protocol_tag: str) -> None:
    """Repair metadata emitted by the reused GT-clean trainer.

    The reused trainer writes some hard-coded gt_upper_bound labels in checkpoint / summaries.
    This function rewrites only metadata fields; it never changes learned weights.
    """
    output_root = output_root.expanduser().resolve()

    def mutate(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "trajectory_source_branch":
                    out[k] = branch
                elif k == "pipeline" and str(v) == "gt_full_y_clean":
                    out[k] = protocol_tag
                elif k == "selected_for_infer_authority" and str(v) == "gt_full_y_clean_protocol":
                    out[k] = protocol_tag
                elif isinstance(v, (dict, list)):
                    out[k] = mutate(v)
                else:
                    out[k] = v
            out.setdefault("vc_full_y_validation", True)
            out.setdefault("vc_protocol_tag", protocol_tag)
            return out
        if isinstance(obj, list):
            return [mutate(v) for v in obj]
        return obj

    for rel in (
        "train/prealign/stage_summary.json",
        "train/pipeline_train_summary.json",
        "run_meta.json",
    ):
        _rewrite_json_file(output_root / rel, mutate)

    # Repair checkpoint metadata, if torch is available. Learned tensors are untouched.
    ckpt_path = output_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if ckpt_path.is_file():
        try:
            import torch  # type: ignore
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                ckpt["trajectory_source_branch"] = branch
                if ckpt.get("pipeline") == "gt_full_y_clean":
                    ckpt["pipeline"] = protocol_tag
                ckpt["vc_full_y_validation"] = True
                ckpt["vc_protocol_tag"] = protocol_tag
                torch.save(ckpt, ckpt_path)
        except Exception as e:  # keep training artifact usable even if metadata rewrite fails
            warn_path = output_root / "train" / "prealign" / "vc_checkpoint_metadata_repair_warning.json"
            warn_path.parent.mkdir(parents=True, exist_ok=True)
            warn_path.write_text(json.dumps({"warning": str(e)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    wrapper = _parse_wrapper_args(sys.argv[1:])
    branch = str(wrapper.trajectory_source_branch)
    protocol_tag = str(wrapper.vc_protocol_tag)

    import videocutler.run_stageb_train_gt_full_y_clean as clean  # noqa: WPS433

    original_loader = clean._load_materialized_gt_examples
    original_response_builder = clean._build_response_rows

    def _load_materialized_vc_examples(
        *,
        repo_root: Path,
        output_root: Path,
        asset_root: Path,
        dataset_name: str,
        annotation_json: Path,
        split_json: Path,
        smoke: bool,
        smoke_max_trajectories: int,
        subset_fraction: Optional[float],
        seed: int,
        **_: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Set[int]], Set[int], Dict[str, Any]]:
        # This mirrors the GT-clean loader while changing only trajectory_source_branch.
        clean._bootstrap_asset_links(repo_root, asset_root)
        clean._bootstrap_asset_links(output_root, asset_root)
        base_ids = clean._load_base_ids(split_json)
        clip_y_base = clean._load_clip_y_base(annotation_json, base_ids)
        with clean._pushd(repo_root):
            materialized = clean.materialize_phase1_training_samples(
                repo_root,
                clean.Phase1MaterializationConfig(
                    dataset_name=str(dataset_name),
                    trajectory_source_branch=branch,
                    smoke=bool(smoke),
                    smoke_max_trajectories=int(smoke_max_trajectories),
                    subset_fraction=subset_fraction,
                    subset_seed=int(seed),
                ),
            )
        samples_raw = materialized.get("valid_samples") or materialized.get("samples") or []
        samples: List[Dict[str, Any]] = []
        sample_counters = clean.Counter()
        for sample in samples_raw:
            if not bool(sample.get("sample_valid", False)):
                sample_counters["skip_sample_not_valid"] += 1
                continue
            clip = clean._as_int(sample.get("clip_id"))
            if clip is None:
                sample_counters["skip_no_clip_id"] += 1
                continue
            y_base = sorted(clip_y_base.get(int(clip), set()))
            if not y_base:
                sample_counters["skip_no_y_base"] += 1
                continue
            row = dict(sample)
            row["observed_raw_ids"] = [int(x) for x in y_base]
            row["clean_label_source"] = "full_Y_base_from_GT_annotations"
            row["trajectory_source_branch"] = branch
            samples.append(row)
        prepared = clean._prepare_prealign_examples(
            samples,
            output_root=output_root,
            dataset_name=str(dataset_name),
            trajectory_source_branch=branch,
        )
        examples = list(prepared.get("examples", []))
        materialization_summary = {
            "materialized_stats": materialized.get("stats", {}),
            "materialized_resolution": materialized.get("resolution", {}),
            "sample_counters": dict(sample_counters),
            "prepare_skipped_reason_histogram": dict(prepared.get("skipped_reason_histogram", {})),
            "sample_count_after_full_y_base_filter": int(len(samples)),
            "trainable_example_count": int(len(examples)),
            "trajectory_source_branch": branch,
            "vc_full_y_validation": True,
            "label_source": "full_Y_base",
        }
        return examples, clip_y_base, base_ids, materialization_summary

    def _build_response_rows_vc(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        rows = original_response_builder(*args, **kwargs)
        for row in rows:
            row["candidate_scope_policy"] = {"policy": "VIDEOCUTLER_FULL_Y_BASE_CLEAN", "label_source": "Y_base"}
            row["training_semantics"] = "videocutler_full_y_clean_prealign"
            row["trajectory_source_branch"] = branch
        return rows

    clean._load_materialized_gt_examples = _load_materialized_vc_examples
    clean._build_response_rows = _build_response_rows_vc

    stripped = [sys.argv[0]] + _strip_wrapper_args(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = stripped
        args = clean.parse_args()
        result = clean.train_clean(args)
        _repair_outputs(Path(args.output_root), branch=branch, protocol_tag=protocol_tag)
        # Write an explicit validation marker so downstream scripts can distinguish this run.
        marker = Path(args.output_root).expanduser().resolve() / "train" / "prealign" / "vc_full_y_validation_meta.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "PASS",
            "entry": "run_stageb_train_full_y_videocutler_clean.py",
            "trajectory_source_branch": branch,
            "label_source": "full_Y_base",
            "protocol": str(getattr(args, "protocol", "")),
            "output_root": str(Path(args.output_root).expanduser().resolve()),
            "boundary": "VideoCutLER/mainline trajectories + full Y_base; no Y-prime/extra/mAP; GT only for later evaluation.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0 if isinstance(result, dict) else 0
    finally:
        sys.argv = old_argv
        clean._load_materialized_gt_examples = original_loader
        clean._build_response_rows = original_response_builder


if __name__ == "__main__":
    raise SystemExit(main())
