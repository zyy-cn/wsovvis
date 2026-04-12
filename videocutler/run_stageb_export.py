from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DATASET_CHOICES = ("lvvis_train_base", "lvvis_val", "ytvis_2019_val")
SPEC_VERSION = "v20.1.3"
CONTRACT_VERSION = "v4.8.8"
TEST_SPEC_VERSION = "v1.3.7"
ASSET_VERSION = "v1.2.6"
SMOKE_FIXTURE_RELS = {
    "lvvis_train_base": "fixtures/tiny_lvvis_pipeline_case/trajectory_records_train_min.jsonl",
    "lvvis_val": "fixtures/tiny_lvvis_pipeline_case/trajectory_records_val_min.jsonl",
}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolved_run_root(output_root: str, exp_name: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.name == exp_name:
        return root
    return root / exp_name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_smoke_fixture(dataset_name: str) -> str:
    fixture_rel = SMOKE_FIXTURE_RELS.get(dataset_name)
    if not fixture_rel:
        raise ValueError(f"unsupported dataset for smoke export: {dataset_name}")
    zip_path = _repo_root() / "package" / "assets" / "WSOVVIS_stageb_test_assets_v1.2.6.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"smoke fixture archive missing: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        try:
            return zf.read(fixture_rel).decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(f"smoke fixture missing in archive: {fixture_rel}") from exc


def _write_smoke_exports(args: argparse.Namespace) -> None:
    fixture_text = _load_smoke_fixture(args.dataset_name)
    export_rel = Path("exports") / args.dataset_name / "trajectory_records.jsonl"
    export_path = _repo_root() / export_rel
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(fixture_text.rstrip("\n") + "\n", encoding="utf-8")


def _lvvis_root_binding() -> Dict[str, str]:
    env_value = os.environ.get("WSOVVIS_LVVIS_ROOT")
    if env_value:
        return {
            "lvvis_root_binding_mode": "env",
            "lvvis_root_resolved": str(Path(env_value).expanduser().resolve()),
        }
    return {
        "lvvis_root_binding_mode": "repo_fallback",
        "lvvis_root_resolved": str((Path(__file__).resolve().parent / "datasets" / "LV-VIS").resolve()),
    }


def build_resolved_config(args: argparse.Namespace, run_root: Path) -> Dict[str, Any]:
    return {
        "runtime": {
            "seed": args.seed,
            "device": args.device,
            "smoke": bool(args.smoke),
        },
        "data": {
            "dataset_name": args.dataset_name,
            "train_dataset": "lvvis_train_base",
            "main_val_dataset": "lvvis_val",
            "aux_val_dataset": "ytvis_2019_val",
        },
        "observation_protocol_id": args.protocol_id,
        "generator_ckpt": args.generator_ckpt,
        "dinov2_ckpt": args.dinov2_ckpt,
        "clip_backend": "openai_clip",
        "clip_ckpt": args.clip_ckpt,
        "output_root": str(run_root),
        "benchmark": {
            "main": "lvvis",
            "aux_validation": "ytvis2019",
        },
        "bindings": {
            "dinov2_source_binding": "third_party/dinov2",
            "clip_source_binding": "third_party/openai_clip",
            "lvvis_evaluator_binding": "third_party/lvvis_official/evaluate",
            "ytvis_evaluator_binding": "videocutler/mask2former_video/data_video/ytvis_eval.py",
            "lvvis_registration_binding": "videocutler/ext_stageb_ovvis/data/datasets/lvvis.py",
            **_lvvis_root_binding(),
        },
        "layout": {
            "output_root_layout_version": "v1",
            "prediction_dirs": {
                "main": "predictions/lvvis_val",
                "aux": "predictions/ytvis_2019_val",
            },
            "eval_dirs": {
                "internal": "eval/internal",
                "main": "eval/lvvis",
                "aux": "eval/ytvis2019",
            },
            "default_selected_for_infer": "softem_aug_last.pth",
        },
    }


def build_run_meta(args: argparse.Namespace, run_root: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "run_id": f"{args.exp_name}_{args.dataset_name}_seed{args.seed}",
        "exp_name": args.exp_name,
        "dataset_name": args.dataset_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_version": SPEC_VERSION,
        "contract_version": CONTRACT_VERSION,
        "test_spec_version": TEST_SPEC_VERSION,
        "asset_version": ASSET_VERSION,
        "main_prediction_path": str(run_root / "predictions" / "lvvis_val" / "pred_main.json"),
        "aux_prediction_path": str(run_root / "predictions" / "ytvis_2019_val" / "pred_main.json"),
        "main_eval_path": str(run_root / "eval" / "lvvis" / "external_metrics.lvvis.json"),
        "aux_eval_path": str(run_root / "eval" / "ytvis2019" / "external_metrics.ytvis2019.json"),
    }
    if args.ckpt_path:
        meta["override_ckpt_path"] = args.ckpt_path
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B LV-VIS export orchestration entrypoint.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--protocol_id",
        default="keep60_seed42",
        choices=("keep80_seed42", "keep60_seed42", "keep40_seed42"),
    )
    parser.add_argument("--dinov2_ckpt", default="vitb14_reg")
    parser.add_argument("--clip_ckpt", default="openai_clip_vit_b16")
    parser.add_argument("--ckpt_path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.smoke and args.dataset_name in {"lvvis_train_base", "lvvis_val"}:
        import videocutler.ext_stageb_ovvis.data.datasets.lvvis  # noqa: F401

    run_root = _resolved_run_root(args.output_root, args.exp_name)
    manifest_root = run_root / "manifests"
    _write_json(manifest_root / "resolved_config.json", build_resolved_config(args, run_root))
    _write_json(manifest_root / "run_meta.json", build_run_meta(args, run_root))
    if args.smoke:
        _write_smoke_exports(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
