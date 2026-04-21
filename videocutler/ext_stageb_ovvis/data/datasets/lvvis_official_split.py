from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_OFFICIAL_SPLIT_REL = Path("package/reference/lvvis_official_base_novel_split.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _official_split_path() -> Path:
    return (_repo_root() / _OFFICIAL_SPLIT_REL).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"official LV-VIS split reference must be a JSON object: {path}")
    return payload


def load_lvvis_official_split_reference() -> Dict[str, Any]:
    path = _official_split_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"official LV-VIS split reference is required but missing: {path}. Refuse to auto-derive from train-seen/val-only."
        )
    payload = _load_json(path)
    base_raw_ids = [int(x) for x in payload.get("base_raw_ids", [])]
    novel_raw_ids = [int(x) for x in payload.get("novel_raw_ids", [])]
    if len(base_raw_ids) != 641 or len(novel_raw_ids) != 555:
        raise ValueError(f"official LV-VIS split reference must contain exactly 641 base and 555 novel raw ids; got base={len(base_raw_ids)} novel={len(novel_raw_ids)}")
    union = set(base_raw_ids) | set(novel_raw_ids)
    if len(union) != 1196:
        raise ValueError(f"official LV-VIS split reference must cover 1196 unique raw ids; got {len(union)}")
    if set(base_raw_ids) & set(novel_raw_ids):
        raise ValueError("official LV-VIS split reference has overlapping base and novel raw ids")
    payload = dict(payload)
    payload["base_raw_ids"] = base_raw_ids
    payload["novel_raw_ids"] = novel_raw_ids
    payload["official_split_ref"] = str(_OFFICIAL_SPLIT_REL.as_posix())
    payload["official_split_sha256"] = _sha256_file(path)
    return payload


def load_lvvis_base_and_novel_raw_ids() -> Tuple[List[int], List[int]]:
    payload = load_lvvis_official_split_reference()
    return list(payload["base_raw_ids"]), list(payload["novel_raw_ids"])


def load_lvvis_official_category_map() -> Dict[int, Dict[str, Any]]:
    payload = load_lvvis_official_split_reference()
    category_map: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("categories", []):
        raw_id = int(row["raw_id"])
        category_map[raw_id] = {"class_name": str(row["class_name"]), "partition": int(row["partition"])}
    return category_map


def validate_lvvis_annotation_categories(*annotation_jsons: Path) -> Dict[str, Any]:
    official = load_lvvis_official_split_reference()
    official_map = load_lvvis_official_category_map()
    official_ids = set(official_map.keys())
    validation_rows: List[Dict[str, Any]] = []
    for ann_path in annotation_jsons:
        if not ann_path.is_file():
            raise FileNotFoundError(f"LV-VIS annotation json not found: {ann_path}")
        payload = _load_json(ann_path)
        categories = payload.get("categories", [])
        category_ids = {int(cat["id"]) for cat in categories}
        if category_ids != official_ids:
            missing = sorted(official_ids - category_ids)
            extra = sorted(category_ids - official_ids)
            raise ValueError(f"LV-VIS annotation categories mismatch official split authority for {ann_path}: missing={missing[:8]} extra={extra[:8]}")
        bad_names: List[int] = []
        for category in categories:
            raw_id = int(category["id"])
            if str(category.get("name", raw_id)) != official_map[raw_id]["class_name"]:
                bad_names.append(raw_id)
                if len(bad_names) >= 8:
                    break
        if bad_names:
            raise ValueError(f"LV-VIS annotation category names mismatch official split authority for {ann_path}: {bad_names}")
        validation_rows.append({"annotation_json": str(ann_path), "annotation_sha256": _sha256_file(ann_path), "category_count": len(categories)})
    return {"official_split_ref": str(official["official_split_ref"]), "official_split_sha256": str(official["official_split_sha256"]), "base_category_count": int(len(official["base_raw_ids"])), "novel_category_count": int(len(official["novel_raw_ids"])), "validated_annotation_jsons": validation_rows}


def filter_raw_ids_to_official_base(raw_ids: Iterable[int]) -> List[int]:
    base_raw_ids, _ = load_lvvis_base_and_novel_raw_ids()
    base_set = {int(x) for x in base_raw_ids}
    return sorted(int(v) for v in {int(x) for x in raw_ids} if int(v) in base_set)
