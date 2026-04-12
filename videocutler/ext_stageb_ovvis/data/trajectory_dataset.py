from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


Record = Dict[str, Any]


def read_trajectory_records(path: str | Path) -> List[Record]:
    records: List[Record] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def filter_records_by_dataset(records: Iterable[Record], dataset_name: str) -> List[Record]:
    return [record for record in records if record.get("dataset_name") == dataset_name]


def filter_records_by_split(records: Iterable[Record], split_tag: str) -> List[Record]:
    return [record for record in records if record.get("split_tag") == split_tag]


def valid_carrier_records(records: Iterable[Record]) -> List[Record]:
    return [record for record in records if bool(record.get("valid_carrier"))]


def group_records_by_clip(records: Iterable[Record]) -> Dict[int, List[Record]]:
    grouped: Dict[int, List[Record]] = {}
    for record in records:
        clip_id = int(record["clip_id"])
        grouped.setdefault(clip_id, []).append(record)
    return grouped
