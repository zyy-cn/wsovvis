from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


Record = Dict[str, Any]


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_mass(mass: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = 0.0
    for key, value in mass.items():
        v = max(0.0, float(value))
        out[str(key)] = v
        total += v
    if total <= 0.0:
        return {"unknown": 1.0}
    return {key: float(value / total) for key, value in out.items()}


@dataclass
class ResponsibilityCache:
    stage_id: str
    by_trajectory_id: MutableMapping[str, Record]

    @classmethod
    def empty(cls, *, stage_id: str) -> "ResponsibilityCache":
        return cls(stage_id=str(stage_id), by_trajectory_id={})

    @classmethod
    def from_records(cls, *, stage_id: str, records: Iterable[Record]) -> "ResponsibilityCache":
        by_tid: Dict[str, Record] = {}
        for row in records:
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id:
                continue
            by_tid[trajectory_id] = dict(row)
        return cls(stage_id=str(stage_id), by_trajectory_id=by_tid)

    @classmethod
    def from_proxy_records(cls, proxy_rows: Iterable[Record], *, stage_id: str) -> "ResponsibilityCache":
        by_tid: Dict[str, Record] = {}
        for row in proxy_rows:
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id:
                continue
            mass = _normalize_mass(dict(row.get("proxy_mass", {})))
            by_tid[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "join_key": str(row.get("join_key", trajectory_id)),
                "r_init": mass,
            }
        return cls(stage_id=str(stage_id), by_trajectory_id=by_tid)

    @classmethod
    def load_jsonl(cls, path: Path, *, stage_id: str) -> "ResponsibilityCache":
        if not path.is_file():
            return cls.empty(stage_id=stage_id)
        return cls.from_records(stage_id=stage_id, records=_load_jsonl(path))

    def get_init_mass(self, trajectory_id: str) -> Dict[str, float]:
        row = self.by_trajectory_id.get(str(trajectory_id))
        if row is None:
            return {"unknown": 1.0}
        if "r_final" in row and isinstance(row["r_final"], dict):
            return _normalize_mass(row["r_final"])
        if "r_init" in row and isinstance(row["r_init"], dict):
            return _normalize_mass(row["r_init"])
        if "proxy_mass" in row and isinstance(row["proxy_mass"], dict):
            return _normalize_mass(row["proxy_mass"])
        return {"unknown": 1.0}

    def update(self, trajectory_id: str, row: Record) -> None:
        self.by_trajectory_id[str(trajectory_id)] = dict(row)

    def sorted_rows(self) -> List[Record]:
        return [
            self.by_trajectory_id[tid]
            for tid in sorted(self.by_trajectory_id.keys(), key=lambda item: str(item))
        ]

    def write_jsonl(self, path: Path) -> None:
        _write_jsonl(path, self.sorted_rows())
