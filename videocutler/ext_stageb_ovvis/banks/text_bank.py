from __future__ import annotations

import json
import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from videocutler.ext_stageb_ovvis.models.text_encoder_clip import (
    ClipTextEncoderConfig,
    OpenAIClipTextEncoder,
)


Record = Dict[str, Any]
_PROTO_LOCATOR_RE = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+)#protos\[(?P<idx>[0-9]+)\]$")


@dataclass(frozen=True)
class TextBankBuildConfig:
    dataset_name: str
    output_root: Path
    clip_ckpt: str
    device: str
    seed: int
    smoke: bool = False
    smoke_num_classes: int = 8


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_lvvis_root() -> Path:
    env_value = os.environ.get("WSOVVIS_LVVIS_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (_repo_root() / "videocutler" / "datasets" / "LV-VIS").resolve()


def _load_templates() -> List[str]:
    ref_path = _repo_root() / "package" / "reference" / "openai_imagenet_templates_80.json"
    payload = _load_json(ref_path)
    templates = [str(item) for item in payload["templates"]]
    if len(templates) != 80:
        raise ValueError(f"expected 80 templates, found {len(templates)}")
    return templates


def _load_class_map() -> List[Tuple[int, str]]:
    lvvis_root = _resolve_lvvis_root()
    ann_paths = [
        lvvis_root / "annotations" / "train_instances.json",
        lvvis_root / "annotations" / "val_instances.json",
    ]
    class_map: Dict[int, str] = {}
    for ann_path in ann_paths:
        if not ann_path.is_file():
            raise FileNotFoundError(f"LV-VIS annotation json not found: {ann_path}")
        payload = _load_json(ann_path)
        for category in payload.get("categories", []):
            raw_id = int(category["id"])
            name = str(category["name"])
            if raw_id in class_map and class_map[raw_id] != name:
                raise ValueError(
                    f"inconsistent category name for raw_id={raw_id}: {class_map[raw_id]} vs {name}"
                )
            class_map[raw_id] = name
    return sorted(class_map.items(), key=lambda item: int(item[0]))


def _select_raw_id_subset(class_map: Sequence[Tuple[int, str]], *, seed: int, smoke: bool, smoke_num_classes: int) -> List[Tuple[int, str]]:
    ordered = list(class_map)
    if not smoke or len(ordered) <= smoke_num_classes:
        return ordered
    rng = random.Random(int(seed))
    chosen = sorted(rng.sample(range(len(ordered)), k=int(smoke_num_classes)))
    return [ordered[index] for index in chosen]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("encountered zero-norm text feature")
    return (matrix / norms).astype(np.float32, copy=False)


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("encountered zero-norm class prototype")
    return (vector / norm).astype(np.float32, copy=False)


def _payload_paths(output_root: Path) -> Tuple[Path, Path]:
    base_dir = output_root / "text_bank"
    return base_dir / "payload" / "text_prototypes.npz", base_dir / "text_prototype_records.jsonl"


def parse_proto_path(proto_path: str) -> Tuple[Path, int]:
    match = _PROTO_LOCATOR_RE.match(proto_path)
    if not match:
        raise ValueError(f"invalid proto_path: {proto_path}")
    return Path(match.group("path")), int(match.group("idx"))


def read_text_prototype_records(path: Path) -> List[Record]:
    records = _load_jsonl(path)
    return sorted(records, key=lambda record: int(record["raw_id"]))


def resolve_text_prototype(records_path: Path, record: Record) -> np.ndarray:
    if str(record.get("path_base_mode", "")) != "artifact_parent_dir":
        raise ValueError(f"unsupported path_base_mode for raw_id={record.get('raw_id')}")
    rel_path, index = parse_proto_path(str(record["proto_path"]))
    payload_path = records_path.parent / rel_path
    with np.load(payload_path, allow_pickle=False) as payload:
        protos = np.asarray(payload["protos"], dtype=np.float32)
    if protos.ndim != 2:
        raise ValueError(f"payload protos must be rank-2: {payload_path}")
    if index < 0 or index >= int(protos.shape[0]):
        raise IndexError(f"proto index out of range for raw_id={record['raw_id']}")
    return np.asarray(protos[index], dtype=np.float32)


def run_text_bank_reader_sanity(records_path: Path) -> List[Dict[str, Any]]:
    records = read_text_prototype_records(records_path)
    rows: List[Dict[str, Any]] = []
    previous_raw_id = None
    for slot, record in enumerate(records):
        raw_id = int(record["raw_id"])
        if previous_raw_id is not None and raw_id < previous_raw_id:
            raise ValueError("text prototype records are not sorted by raw_id")
        previous_raw_id = raw_id
        vector = resolve_text_prototype(records_path, record)
        rows.append(
            {
                "raw_id": raw_id,
                "slot": slot,
                "dtype": str(vector.dtype),
                "dim": int(vector.shape[0]),
                "proto_path": str(record["proto_path"]),
            }
        )
    return rows


def build_text_bank(config: TextBankBuildConfig) -> Dict[str, Any]:
    if config.dataset_name != "lvvis_train_base":
        raise ValueError(f"unsupported dataset_name: {config.dataset_name}")
    if config.clip_ckpt != "openai_clip_vit_b16":
        raise ValueError(f"unsupported clip_ckpt: {config.clip_ckpt}")

    templates = _load_templates()
    full_class_map = _load_class_map()
    selected_class_map = _select_raw_id_subset(
        full_class_map,
        seed=config.seed,
        smoke=config.smoke,
        smoke_num_classes=config.smoke_num_classes,
    )
    if not selected_class_map:
        raise ValueError("resolved empty LV-VIS class map")

    encoder = OpenAIClipTextEncoder(
        ClipTextEncoderConfig(
            clip_ckpt=config.clip_ckpt,
            device=config.device,
        )
    )

    prototypes: List[np.ndarray] = []
    records: List[Record] = []
    for slot, (raw_id, class_name) in enumerate(selected_class_map):
        prompts = [template.format(class_name) for template in templates]
        prompt_features = encoder.encode_texts(prompts)
        prompt_features = _l2_normalize_rows(prompt_features)
        prototype = _l2_normalize_vector(prompt_features.mean(axis=0, dtype=np.float32))
        prototypes.append(prototype.astype(np.float32, copy=False))
        records.append(
            {
                "raw_id": int(raw_id),
                "proto_path": f"payload/text_prototypes.npz#protos[{slot}]",
                "path_base_mode": "artifact_parent_dir",
            }
        )

    protos = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    if protos.ndim != 2 or int(protos.shape[1]) != 512:
        raise ValueError(f"expected text prototypes with shape [C, 512], found {tuple(protos.shape)}")
    payload_path, records_path = _payload_paths(config.output_root)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(payload_path, protos=protos)
    _write_jsonl(records_path, records)

    return {
        "records_path": records_path,
        "payload_path": payload_path,
        "record_count": len(records),
        "embedding_dim": int(protos.shape[1]),
        "full_class_count": len(full_class_map),
        "selected_raw_ids": [int(raw_id) for raw_id, _ in selected_class_map],
        "run_scope": "smoke" if config.smoke else "full",
    }
