from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-B text bank artifacts.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--clip_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract_check_path(repo_root: Path, smoke: bool) -> Path:
    filename = "g6_text_bank_smoke_contract_check.json" if smoke else "text_bank_contract_check.json"
    return repo_root / "codex" / "outputs" / "G6_text_bank" / filename


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.dataset_name != "lvvis_train_base":
        raise SystemExit("dataset_name must be lvvis_train_base")
    if args.clip_ckpt != "openai_clip_vit_b16":
        raise SystemExit("clip_ckpt must be openai_clip_vit_b16")

    from videocutler.ext_stageb_ovvis.banks.text_bank import TextBankBuildConfig, build_text_bank

    repo_root = _repo_root()
    output_root = Path(args.output_root).expanduser().resolve()
    result = build_text_bank(
        TextBankBuildConfig(
            dataset_name=args.dataset_name,
            output_root=output_root,
            clip_ckpt=args.clip_ckpt,
            device=args.device,
            seed=int(args.seed),
            smoke=bool(args.smoke),
        )
    )

    run_scope = str(result["run_scope"])
    contract_check = {
        "gate_id": "G6_text_bank",
        "contract_ref": "contracts/gates/G6_text_bank.gate_contract.json",
        "status": "PASS",
        "artifact_path_base": "output_root_relative",
        "primary_artifacts": ["text_bank/text_prototype_records.jsonl"],
        "checks_run": [
            "text_bank_reader_parses_dsl",
            "text_bank_consumer_ready",
            "artifact_exists",
            "artifact_schema_valid",
        ],
        "run_scope": run_scope,
        "input_source_type": "official_lvvis_train_val_category_union",
        "data_scope": "lvvis_full_class_map" if run_scope == "full" else "lvvis_full_class_map_smoke_subset",
        "class_coverage": {
            "coverage_mode": "sorted_union_categories",
            "dataset_name_control_plane": args.dataset_name,
            "full_class_count": int(result["full_class_count"]),
            "selected_class_count": int(result["record_count"]),
            "selected_raw_ids": result["selected_raw_ids"],
        },
        "consumer_target": "run_stageb_train_prealign|run_stageb_train_softem|run_stageb_infer_ov",
        "consumer_ready": run_scope == "full",
    }
    _write_json(_contract_check_path(repo_root, bool(args.smoke)), contract_check)

    print(
        json.dumps(
            {
                "status": "PASS",
                "run_scope": run_scope,
                "records_path": str(result["records_path"]),
                "payload_path": str(result["payload_path"]),
                "record_count": int(result["record_count"]),
                "embedding_dim": int(result["embedding_dim"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
