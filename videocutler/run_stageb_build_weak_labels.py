from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from videocutler.ext_stageb_ovvis.data.weak_labels import (
    build_label_map_from_class_map,
    build_label_map_from_text_prototypes,
    build_weak_labels_from_fixture,
    read_json,
    read_jsonl,
    write_weak_labels,
)


DATASET_CHOICES = ("lvvis_train_base",)
PROTOCOL_CHOICES = ("keep80_seed42", "keep60_seed42", "keep40_seed42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B weak-label orchestration entrypoint.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--protocol_id", default="keep60_seed42", choices=PROTOCOL_CHOICES)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input_json")
    parser.add_argument("--text_prototypes_jsonl")
    parser.add_argument("--class_map_json")
    parser.add_argument("--split_tag", default="train_smoke", choices=("train", "train_smoke"))
    parser.add_argument("--output_json")
    parser.add_argument("--contract_check_json")
    return parser.parse_args()


def _resolved_run_root(output_root: str, exp_name: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.name == exp_name:
        return root
    return root / exp_name


def _default_output_json(run_root: Path) -> Path:
    return run_root / "weak_labels" / "weak_labels_train.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_contract_check(
    path: str | Path,
    *,
    payload_path: Path,
    records: list[dict],
    dataset_name: str,
    protocol_id: str,
    split_tag: str,
) -> Path:
    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    payload_rel = payload_path.as_posix()
    payload_exists = payload_path.exists()
    sample = records[0] if records and isinstance(records[0], dict) else {}
    payload = {
        "gate_id": "G3_weak_labels",
        "status": "PASS" if payload_exists and records else "FAIL",
        "dataset_name": dataset_name,
        "split_tag": split_tag,
        "observation_protocol_id": protocol_id,
        "run_scope": str(sample.get("run_scope", "")),
        "input_source_type": str(sample.get("input_source_type", "")),
        "data_scope": str(sample.get("data_scope", "")),
        "consumer_target": str(sample.get("consumer_target", "")),
        "record_count": len(records),
        "coverage_ratio": sample.get("coverage_ratio", 0.0),
        "consumer_ready": bool(sample.get("consumer_ready", False)),
        "payload_output": payload_rel,
        "payload_exists": payload_exists,
        "payload_record_count": len(records),
        "payload_sha256": _sha256_file(payload_path) if payload_exists else "",
        "schema_ref": "package/assets/schemas/weak_labels_train.schema.json",
        "output_root_layout_ref": "package/assets/reference/output_root_layout.json",
        "deliverables": {
            "weak_labels_cli": "videocutler/run_stageb_build_weak_labels.py",
            "weak_labels_module": "videocutler/ext_stageb_ovvis/data/weak_labels.py",
            "weak_labels_payload": payload_rel,
        },
    }
    contract_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract_path


def main() -> int:
    args = parse_args()
    if not args.input_json:
        if args.smoke:
            return 0
        raise SystemExit("--input_json is required unless --smoke is set")
    if bool(args.text_prototypes_jsonl) == bool(args.class_map_json):
        raise SystemExit("Provide exactly one of --text_prototypes_jsonl or --class_map_json")

    fixture = read_json(args.input_json)
    if args.text_prototypes_jsonl:
        label_map = build_label_map_from_text_prototypes(read_jsonl(args.text_prototypes_jsonl))
    else:
        label_map = build_label_map_from_class_map(read_json(args.class_map_json))

    records = build_weak_labels_from_fixture(
        fixture,
        protocol_id=args.protocol_id,
        label_map=label_map,
        split_tag=args.split_tag,
    )
    run_root = _resolved_run_root(args.output_root, args.exp_name)
    output_json = Path(args.output_json) if args.output_json else _default_output_json(run_root)
    write_weak_labels(output_json, records)
    if args.contract_check_json:
        write_contract_check(
            args.contract_check_json,
            payload_path=output_json,
            records=records,
            dataset_name=args.dataset_name,
            protocol_id=args.protocol_id,
            split_tag=args.split_tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
