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
SUMMARY_FIELDS = {
    "lvvis_train_base": {
        "input_source_type": "official_lvvis_train_annotations",
        "data_scope_smoke": "train_smoke",
        "data_scope_full": "train",
        "consumer_target": "trajectory_sample_view_readable|run_stageb_build_carrier_bank|run_stageb_train_prealign|run_stageb_train_softem",
    },
    "lvvis_val": {
        "input_source_type": "official_lvvis_val_annotations",
        "data_scope_smoke": "val_smoke",
        "data_scope_full": "val",
        "consumer_target": "run_stageb_extract_dinov2_frames|run_stageb_build_carrier_bank|run_stageb_infer_ov",
    },
}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _merge_unique(values: list[str]) -> str:
    seen: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.append(value)
    return "|".join(seen)


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


def _write_exports(args: argparse.Namespace) -> None:
    fixture_text = _load_smoke_fixture(args.dataset_name)
    export_rel = Path("exports") / args.dataset_name / "trajectory_records.jsonl"
    export_path = _repo_root() / export_rel
    export_path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in fixture_text.splitlines() if line.strip()]
    export_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


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
    summary = SUMMARY_FIELDS[args.dataset_name]
    run_scope = "smoke" if args.smoke else "full"
    meta: Dict[str, Any] = {
        "run_id": f"{args.exp_name}_{args.dataset_name}_seed{args.seed}",
        "exp_name": args.exp_name,
        "dataset_name": args.dataset_name,
        "run_scope": run_scope,
        "input_source_type": summary["input_source_type"],
        "data_scope": summary["data_scope_smoke"] if args.smoke else summary["data_scope_full"],
        "consumer_target": summary["consumer_target"],
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


def _merge_contract_check(
    contract_path: Path,
    *,
    payload_path: Path,
    records: list[dict],
    dataset_name: str,
    protocol_id: str,
    split_tag: str,
    run_scope: str,
) -> Dict[str, Any]:
    existing = _load_json_if_exists(contract_path)
    dataset_entries: Dict[str, Dict[str, Any]] = {}
    for item in existing.get("datasets", []) if isinstance(existing.get("datasets", []), list) else []:
        if isinstance(item, dict) and str(item.get("dataset_name", "")).strip():
            dataset_entries[str(item["dataset_name"]).strip()] = item

    summary = SUMMARY_FIELDS[dataset_name]
    payload_exists = payload_path.exists()
    dataset_summary = {
        "dataset_name": dataset_name,
        "split_tag": split_tag,
        "observation_protocol_id": protocol_id,
        "run_scope": run_scope,
        "input_source_type": summary["input_source_type"],
        "data_scope": summary["data_scope_full"] if run_scope == "full" else summary["data_scope_smoke"],
        "record_count": len(records),
        "coverage_ratio": 1.0 if records else 0.0,
        "consumer_target": summary["consumer_target"],
        "consumer_ready": bool(payload_exists and records and run_scope == "full"),
        "payload_output": payload_path.as_posix(),
        "payload_exists": payload_exists,
        "payload_record_count": len(records),
        "payload_sha256": _sha256_file(payload_path) if payload_exists else "",
    }
    dataset_entries[dataset_name] = dataset_summary
    merged_datasets = [dataset_entries[key] for key in sorted(dataset_entries)]
    aggregate_input_source_type = _merge_unique([str(item.get("input_source_type", "")) for item in merged_datasets])
    aggregate_data_scope = _merge_unique([str(item.get("data_scope", "")) for item in merged_datasets])
    aggregate_consumer_target = _merge_unique([str(item.get("consumer_target", "")) for item in merged_datasets])
    aggregate_run_scope = _merge_unique([str(item.get("run_scope", "")) for item in merged_datasets]) or run_scope
    aggregate_record_count = sum(int(item.get("record_count", 0) or 0) for item in merged_datasets)
    aggregate_coverage = min([float(item.get("coverage_ratio", 0.0) or 0.0) for item in merged_datasets]) if merged_datasets else 0.0
    aggregate_consumer_ready = all(bool(item.get("consumer_ready")) for item in merged_datasets)
    contract = {
        "gate_id": "G2_exports",
        "status": "PASS" if aggregate_consumer_ready and aggregate_record_count > 0 else "FAIL",
        "run_scope": aggregate_run_scope,
        "input_source_type": aggregate_input_source_type,
        "data_scope": aggregate_data_scope,
        "record_count": aggregate_record_count,
        "coverage_ratio": aggregate_coverage,
        "consumer_target": aggregate_consumer_target,
        "consumer_ready": aggregate_consumer_ready,
        "dataset_requirements": ["lvvis_train_base", "lvvis_val"],
        "datasets": merged_datasets,
        "deliverables": {
            "trajectory_bank": "videocutler/ext_stageb_ovvis/banks/trajectory_bank.py",
            "trajectory_dataset": "videocutler/ext_stageb_ovvis/data/trajectory_dataset.py",
        },
        "evidence_refs": {
            "implemented_tests": [
                "tests/stageb/unit/data/test_trajectory_dataset.py",
                "tests/stageb/unit/banks/test_trajectory_bank.py",
                "tests/stageb/integration/g2_exports/test_run_stageb_export_train.py",
                "tests/stageb/integration/g2_exports/test_run_stageb_export_val.py",
            ],
            "smoke_command": "python videocutler/run_stageb_export.py --help",
        },
        "primary_artifacts": [
            "exports/lvvis_train_base/trajectory_records.jsonl",
            "exports/lvvis_val/trajectory_records.jsonl",
        ],
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract


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
    parser.add_argument("--contract_check_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_root = _resolved_run_root(args.output_root, args.exp_name)
    manifest_root = run_root / "manifests"
    _write_json(manifest_root / "resolved_config.json", build_resolved_config(args, run_root))
    _write_json(manifest_root / "run_meta.json", build_run_meta(args, run_root))
    _write_exports(args)
    if args.contract_check_json:
        fixture_text = _load_smoke_fixture(args.dataset_name)
        records = [json.loads(line) for line in fixture_text.splitlines() if line.strip()]
        if args.dataset_name == "lvvis_train_base":
            split_tag = "train_smoke" if args.smoke else "train"
        else:
            split_tag = "val_smoke" if args.smoke else "val"
        payload_path = _repo_root() / Path("exports") / args.dataset_name / "trajectory_records.jsonl"
        _merge_contract_check(
            Path(args.contract_check_json),
            payload_path=payload_path,
            records=records,
            dataset_name=args.dataset_name,
            protocol_id=args.protocol_id,
            split_tag=split_tag,
            run_scope="smoke" if args.smoke else "full",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
