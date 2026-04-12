from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from zipfile import ZipFile


Record = Dict[str, Any]
LabelInfo = Dict[str, Any]

PROTOCOL_RATIOS: Dict[str, float] = {
    "keep80_seed42": 0.8,
    "keep60_seed42": 0.6,
    "keep40_seed42": 0.4,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_from_asset_zip(path: Path) -> str:
    rel = path.as_posix()
    prefix = "package/assets/"
    if not rel.startswith(prefix):
        raise FileNotFoundError(path)
    zip_path = _repo_root() / "package" / "assets" / "WSOVVIS_stageb_test_assets_v1.2.6.zip"
    if not zip_path.exists():
        raise FileNotFoundError(path)
    inner = rel[len(prefix):]
    with ZipFile(zip_path, "r") as zf:
        try:
            data = zf.read(inner)
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
    return data.decode("utf-8")


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(_read_from_asset_zip(p))


def read_jsonl(path: str | Path) -> List[Record]:
    records: List[Record] = []
    p = Path(path)
    if p.exists():
        payload = p.read_text(encoding="utf-8")
    else:
        payload = _read_from_asset_zip(p)
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def protocol_ratio(protocol_id: str) -> float:
    if protocol_id not in PROTOCOL_RATIOS:
        raise ValueError(f"Unsupported observation protocol: {protocol_id}")
    return PROTOCOL_RATIOS[protocol_id]


def exact_k_count(num_classes: int, protocol_id: str) -> int:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return max(1, round(protocol_ratio(protocol_id) * num_classes))


def build_label_map_from_text_prototypes(records: Iterable[Record]) -> Dict[int, LabelInfo]:
    label_map: Dict[int, LabelInfo] = {}
    for record in records:
        raw_id = int(record["raw_id"])
        label_map[raw_id] = {
            "contiguous_id": int(record["contiguous_id"]),
            "class_name": str(record["class_name"]),
        }
    return label_map


def build_label_map_from_class_map(records: Iterable[Record]) -> Dict[int, LabelInfo]:
    sorted_records = sorted(records, key=lambda item: int(item["raw_id"]))
    label_map: Dict[int, LabelInfo] = {}
    for contiguous_id, record in enumerate(sorted_records):
        raw_id = int(record["raw_id"])
        label_map[raw_id] = {
            "contiguous_id": contiguous_id,
            "class_name": str(record["name"]),
        }
    return label_map


def select_observed_raw_ids(
    full_raw_ids: Iterable[int],
    *,
    video_id: int,
    protocol_id: str,
    seed: int = 42,
) -> List[int]:
    unique_sorted_ids = sorted({int(raw_id) for raw_id in full_raw_ids})
    if not unique_sorted_ids:
        raise ValueError("full_raw_ids must not be empty")
    keep_count = exact_k_count(len(unique_sorted_ids), protocol_id)
    rng = random.Random(f"{int(seed)}:{int(video_id)}:{protocol_id}")
    return sorted(rng.sample(unique_sorted_ids, keep_count))


def build_weak_label_record(
    *,
    dataset_name: str,
    split_tag: str,
    video_id: int,
    observed_raw_ids: Iterable[int],
    protocol_id: str,
    label_map: Mapping[int, LabelInfo],
    run_scope: str,
    input_source_type: str,
    data_scope: str,
    consumer_target: str,
    record_count: int,
    coverage_ratio: float,
    consumer_ready: bool,
) -> Record:
    raw_ids = sorted(int(raw_id) for raw_id in observed_raw_ids)
    missing = [raw_id for raw_id in raw_ids if raw_id not in label_map]
    if missing:
        raise KeyError(f"Missing label metadata for raw ids: {missing}")
    return {
        "dataset_name": dataset_name,
        "split_tag": split_tag,
        "clip_id": int(video_id),
        "video_id": int(video_id),
        "observed_raw_ids": raw_ids,
        "observed_contiguous_ids": [int(label_map[raw_id]["contiguous_id"]) for raw_id in raw_ids],
        "observed_class_names": [str(label_map[raw_id]["class_name"]) for raw_id in raw_ids],
        "completeness_status": "unknown",
        "label_source_type": "simulated_from_gt",
        "observation_protocol_id": protocol_id,
        "run_scope": run_scope,
        "input_source_type": input_source_type,
        "data_scope": data_scope,
        "consumer_target": consumer_target,
        "record_count": int(record_count),
        "coverage_ratio": float(coverage_ratio),
        "consumer_ready": bool(consumer_ready),
    }


def build_weak_labels(
    videos: Iterable[Record],
    *,
    dataset_name: str,
    split_tag: str,
    protocol_id: str,
    label_map: Mapping[int, LabelInfo],
    seed: int = 42,
) -> List[Record]:
    run_scope = "smoke" if str(split_tag).endswith("_smoke") else "full"
    input_source_type = "smoke_fixture" if run_scope == "smoke" else "official_lvvis_train_annotations"
    data_scope = "train_smoke" if run_scope == "smoke" else "train"
    consumer_target = "downstream_train"
    output: List[Record] = []
    sorted_videos = sorted(videos, key=lambda item: int(item["video_id"]))
    coverage_ratio = 1.0 if sorted_videos else 0.0
    consumer_ready = run_scope == "full"
    for video in sorted_videos:
        video_id = int(video["video_id"])
        observed_raw_ids = select_observed_raw_ids(
            video.get("full_raw_ids", []),
            video_id=video_id,
            protocol_id=protocol_id,
            seed=seed,
        )
        output.append(
            build_weak_label_record(
                dataset_name=dataset_name,
                split_tag=split_tag,
                video_id=video_id,
                observed_raw_ids=observed_raw_ids,
                protocol_id=protocol_id,
                label_map=label_map,
                run_scope=run_scope,
                input_source_type=input_source_type,
                data_scope=data_scope,
                consumer_target=consumer_target,
                record_count=len(sorted_videos),
                coverage_ratio=coverage_ratio,
                consumer_ready=consumer_ready,
            )
        )
    return output


def build_weak_labels_from_fixture(
    fixture: Mapping[str, Any],
    *,
    protocol_id: str,
    label_map: Mapping[int, LabelInfo],
    split_tag: str = "train_smoke",
) -> List[Record]:
    return build_weak_labels(
        fixture.get("videos", []),
        dataset_name=str(fixture["dataset_name"]),
        split_tag=split_tag,
        protocol_id=protocol_id,
        label_map=label_map,
        seed=int(fixture.get("seed", 42)),
    )


def write_weak_labels(path: str | Path, records: Iterable[Record]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path
