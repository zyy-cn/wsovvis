from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

Record = Dict[str, Any]


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def unique_ints(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = values.replace(";", ",").split(",")
    elif isinstance(values, Mapping):
        parts = values.keys()
    elif isinstance(values, Iterable):
        parts = values
    else:
        parts = [values]
    out: List[int] = []
    seen: set[int] = set()
    for item in parts:
        ix = safe_int(item)
        if ix is None or ix in seen:
            continue
        seen.add(ix)
        out.append(int(ix))
    return out


def load_json(path: Path) -> Record:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def iter_jsonl(path: Path) -> Iterator[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def load_official_split(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    base_raw_ids = [int(x) for x in payload.get("base_raw_ids", [])]
    novel_raw_ids = [int(x) for x in payload.get("novel_raw_ids", [])]
    if len(base_raw_ids) != 641 or len(novel_raw_ids) != 555:
        raise ValueError(f"official split must contain 641 base and 555 novel raw ids; got base={len(base_raw_ids)} novel={len(novel_raw_ids)}")
    if set(base_raw_ids) & set(novel_raw_ids):
        raise ValueError("official split has overlapping base and novel raw ids")
    return {
        "payload": payload,
        "base_raw_ids": base_raw_ids,
        "novel_raw_ids": novel_raw_ids,
        "base_set": set(base_raw_ids),
        "novel_set": set(novel_raw_ids),
        "base_count": len(base_raw_ids),
        "novel_count": len(novel_raw_ids),
        "union_count": len(set(base_raw_ids) | set(novel_raw_ids)),
        "official_split_ref": str(payload.get("official_split_ref", path.as_posix())),
        "official_split_sha256": str(payload.get("official_split_sha256", "")),
    }


def load_weak_label_records(path: Path) -> List[Record]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"weak label payload must be a JSON list: {path}")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    return rows


def load_text_bank_raw_ids(output_root: Path) -> List[int]:
    from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab

    raw_ids, _records, _matrix = load_text_vocab(output_root)
    return [int(x) for x in raw_ids]


def iter_train_annotation_pairs(annotation_json: Path) -> Iterator[Tuple[int, int]]:
    """Yield (video_id, category_id) from LV-VIS train_instances.json without materializing the full file.

    The file is a single large minified JSON object. We scan until the annotations
    array and then use JSONDecoder.raw_decode to read one annotation object at a
    time.
    """

    decoder = json.JSONDecoder()
    key_token = '"annotations"'
    buffer = ""
    found = False
    with annotation_json.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk and not buffer:
                break
            buffer += chunk
            if not found:
                idx = buffer.find(key_token)
                if idx < 0:
                    if not chunk:
                        break
                    buffer = buffer[-4096:]
                    continue
                arr_start = buffer.find("[", idx)
                if arr_start < 0:
                    if not chunk:
                        break
                    buffer = buffer[idx:]
                    continue
                buffer = buffer[arr_start + 1 :]
                found = True
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                video_id = safe_int(obj.get("video_id"))
                category_id = safe_int(obj.get("category_id"))
                if video_id is not None and category_id is not None:
                    yield int(video_id), int(category_id)
                buffer = buffer[end:]
            if not chunk:
                break


def build_full_y_records(
    *,
    annotation_json: Path,
    weak_labels_json: Path,
    official_split_json: Path,
    text_bank_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    split = load_official_split(official_split_json)
    weak_rows = load_weak_label_records(weak_labels_json)
    weak_by_video: Dict[int, Record] = {}
    weak_align_mismatch_examples: List[Record] = []
    for row in weak_rows:
        video_id = safe_int(row.get("video_id"), safe_int(row.get("clip_id"), None))
        if video_id is None:
            continue
        weak_by_video[int(video_id)] = dict(row)

    full_by_video: Dict[int, set[int]] = defaultdict(set)
    annotation_count = 0
    for video_id, category_id in iter_train_annotation_pairs(annotation_json):
        annotation_count += 1
        full_by_video[int(video_id)].add(int(category_id))

    text_bank_raw_ids: Optional[set[int]] = None
    if text_bank_root is not None:
        try:
            text_bank_raw_ids = set(load_text_bank_raw_ids(text_bank_root))
        except Exception:
            text_bank_raw_ids = None

    records: List[Record] = []
    full_y_union: set[int] = set()
    yprime_union: set[int] = set()
    yprime_subset_violation_count = 0
    missing_full_y_count = 0
    missing_weak_count = 0
    yprime_clip_mismatch_count = 0

    for video_id in sorted(weak_by_video.keys()):
        weak_row = weak_by_video[video_id]
        clip_id = safe_int(weak_row.get("clip_id"), video_id)
        weak_raw_ids = sorted({int(x) for x in unique_ints(weak_row.get("observed_raw_ids"))})
        full_raw_ids = sorted({int(x) for x in full_by_video.get(int(video_id), set())})
        if clip_id is None or int(clip_id) != int(video_id):
            yprime_clip_mismatch_count += 1
        if not full_raw_ids:
            missing_full_y_count += 1
        if not weak_raw_ids:
            missing_weak_count += 1
        if not set(weak_raw_ids).issubset(set(full_raw_ids)):
            yprime_subset_violation_count += 1
            if len(weak_align_mismatch_examples) < 16:
                weak_align_mismatch_examples.append(
                    {
                        "clip_id": int(clip_id) if clip_id is not None else None,
                        "video_id": int(video_id),
                        "yprime_raw_ids": weak_raw_ids[:32],
                        "full_y_raw_ids": full_raw_ids[:32],
                        "missing_from_full_y": sorted(set(weak_raw_ids) - set(full_raw_ids))[:32],
                    }
                )
        full_y_union.update(full_raw_ids)
        yprime_union.update(weak_raw_ids)
        records.append(
            {
                "clip_id": int(clip_id) if clip_id is not None else int(video_id),
                "video_id": int(video_id),
                "full_y_raw_ids": full_raw_ids,
                "yprime_raw_ids": weak_raw_ids,
                "base_raw_ids": [int(x) for x in full_raw_ids if int(x) in split["base_set"]],
                "novel_raw_ids": [int(x) for x in full_raw_ids if int(x) in split["novel_set"]],
            }
        )

    missing_text_bank_raw_id_count = None
    if text_bank_raw_ids is not None:
        missing_text_bank_raw_id_count = int(len([rid for rid in full_y_union if rid not in text_bank_raw_ids]))

    summary: Dict[str, Any] = {
        "dataset_name": "lvvis_train_base",
        "id_space": "raw_category_id",
        "label_scope": "full_gt_raw",
        "source_annotation": str(annotation_json),
        "source_weak_labels": str(weak_labels_json),
        "clip_count": int(len(records)),
        "annotation_count": int(annotation_count),
        "full_y_union_count": int(len(full_y_union)),
        "yprime_union_count": int(len(yprime_union)),
        "yprime_subset_full_y_violation_count": int(yprime_subset_violation_count),
        "full_y_base_count": int(len(full_y_union & split["base_set"])),
        "full_y_novel_count": int(len(full_y_union & split["novel_set"])),
        "missing_full_y_count": int(missing_full_y_count),
        "missing_weak_count": int(missing_weak_count),
        "weak_video_alignment_mismatch_count": int(yprime_clip_mismatch_count),
        "weak_alignment_violation_examples": weak_align_mismatch_examples,
        "missing_text_bank_raw_id_count": missing_text_bank_raw_id_count,
        "official_split_path": str(official_split_json),
        "official_split_base_count": int(split["base_count"]),
        "official_split_novel_count": int(split["novel_count"]),
    }
    asset = {
        "dataset_name": "lvvis_train_base",
        "id_space": "raw_category_id",
        "label_scope": "full_gt_raw",
        "source_annotation": str(annotation_json),
        "source_weak_labels": str(weak_labels_json),
        "official_split_path": str(official_split_json),
        "records": records,
    }
    return asset, summary
