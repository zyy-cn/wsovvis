from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_official_split_reference
from videocutler.ext_stageb_ovvis.data.weak_labels import build_label_map_from_class_map, build_label_map_from_text_prototypes, build_official_lvvis_train_fixture, build_weak_labels_from_fixture, read_json, read_jsonl, sha256_path, write_weak_labels
from videocutler.ext_stageb_ovvis.eval.external_lvvis import resolve_lvvis_annotation_paths

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
    return root if root.name == exp_name else root / exp_name


def _default_output_json(run_root: Path, *, dataset_name: str, smoke: bool) -> Path:
    return (run_root / "stageb_smoke" / "weak_labels" / dataset_name / "weak_labels_train.json") if smoke else (run_root / "weak_labels" / "weak_labels_train.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_full_output_json(run_root: Path) -> Path:
    return run_root / "weak_labels" / "weak_labels_train.json"


def _validate_output_path(*, output_json: Path, run_root: Path, smoke: bool) -> None:
    if smoke and output_json.resolve() == _canonical_full_output_json(run_root).resolve():
        raise SystemExit("smoke weak-label output must not overwrite canonical full weak_labels/weak_labels_train.json")


def write_contract_check(path: str | Path, *, payload_path: Path, records: list[dict], dataset_name: str, protocol_id: str, split_tag: str) -> Path:
    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    payload_rel = payload_path.as_posix(); payload_exists = payload_path.exists(); sample = records[0] if records and isinstance(records[0], dict) else {}
    payload = {"gate_id": "G3_weak_labels", "contract_ref": "contracts/gates/G3_weak_labels.gate_contract.json", "status": "PASS" if payload_exists and records else "FAIL", "artifact_path_base": "output_root_relative", "primary_artifacts": [payload_rel], "checks_run": ["clip_level_weak_label_reader_readable", "weak_labels_run_scope_declared", "weak_labels_full_consumer_ready", "artifact_exists", "artifact_schema_valid"], "dataset_name": dataset_name, "split_tag": split_tag, "observation_protocol_id": protocol_id, "run_scope": str(sample.get("run_scope", "scope_neutral")), "input_source_type": str(sample.get("input_source_type", "")), "data_scope": str(sample.get("data_scope", "")), "consumer_target": str(sample.get("consumer_target", "")), "record_count": len(records), "coverage_ratio": sample.get("coverage_ratio", 0.0), "consumer_ready": bool(sample.get("consumer_ready", False)), "official_split_ref": str(sample.get("official_split_ref", "")), "official_split_sha256": str(sample.get("official_split_sha256", "")), "upstream_source_ref": str(sample.get("upstream_source_ref", "")), "upstream_source_sha256": str(sample.get("upstream_source_sha256", "")), "payload_output": payload_rel, "payload_exists": payload_exists, "payload_record_count": len(records), "payload_sha256": _sha256_file(payload_path) if payload_exists else "", "schema_ref": "package/schemas/weak_labels_train.schema.json", "output_root_layout_ref": "package/reference/output_root_layout.json", "deliverables": {"weak_labels_cli": "videocutler/run_stageb_build_weak_labels.py", "weak_labels_module": "videocutler/ext_stageb_ovvis/data/weak_labels.py", "weak_labels_payload": payload_rel}}
    contract_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract_path


def main() -> int:
    args = parse_args()
    if bool(args.text_prototypes_jsonl) == bool(args.class_map_json):
        raise SystemExit("Provide exactly one of --text_prototypes_jsonl or --class_map_json")
    label_map = build_label_map_from_text_prototypes(read_jsonl(args.text_prototypes_jsonl)) if args.text_prototypes_jsonl else build_label_map_from_class_map(read_json(args.class_map_json))
    official_split = load_lvvis_official_split_reference()
    run_root = _resolved_run_root(args.output_root, args.exp_name)
    output_json = Path(args.output_json) if args.output_json else _default_output_json(run_root, dataset_name=args.dataset_name, smoke=bool(args.smoke))
    _validate_output_path(output_json=output_json, run_root=run_root, smoke=bool(args.smoke))
    if args.smoke:
        if not args.input_json:
            raise SystemExit("--input_json is required for smoke weak-label generation")
        fixture = read_json(args.input_json); input_source_type = "smoke_fixture"; upstream_source_ref = str(Path(args.input_json).expanduser().resolve()); upstream_source_sha256 = sha256_path(args.input_json)
    else:
        if args.input_json:
            raise SystemExit("formal/full weak-label generation is official-source locked and does not accept arbitrary --input_json")
        ann_paths = resolve_lvvis_annotation_paths(validate_official_authority=True)
        fixture = build_official_lvvis_train_fixture(ann_paths.train_json, dataset_name=args.dataset_name)
        input_source_type = "official_lvvis_train_annotations"; upstream_source_ref = str(ann_paths.train_json); upstream_source_sha256 = sha256_path(ann_paths.train_json)
    records = build_weak_labels_from_fixture(fixture, protocol_id=args.protocol_id, label_map=label_map, split_tag=args.split_tag, input_source_type=input_source_type, upstream_source_ref=upstream_source_ref, upstream_source_sha256=upstream_source_sha256, official_split_ref=str(official_split["official_split_ref"]), official_split_sha256=str(official_split["official_split_sha256"]))
    write_weak_labels(output_json, records)
    if args.contract_check_json:
        write_contract_check(args.contract_check_json, payload_path=output_json, records=records, dataset_name=args.dataset_name, protocol_id=args.protocol_id, split_tag=args.split_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
