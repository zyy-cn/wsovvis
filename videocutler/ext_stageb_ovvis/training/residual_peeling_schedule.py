from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return None


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_list_of_ids(v: Any) -> Optional[List[int]]:
    if isinstance(v, list):
        out: List[int] = []
        ok = True
        for item in v:
            if isinstance(item, dict):
                val = item.get("raw_id", item.get("id", item.get("category_id")))
            else:
                val = item
            ii = _as_int(val)
            if ii is None:
                ok = False
                break
            out.append(ii)
        return out if ok else None
    return None


def _extract_split_ids(obj: Any, split_name: str) -> List[int]:
    keys = {
        "base": [
            "base", "base_ids", "base_raw_ids", "base_category_ids", "base_classes",
            "official_base", "base_raw_id_list", "base_categories",
        ],
        "novel": [
            "novel", "novel_ids", "novel_raw_ids", "novel_category_ids", "novel_classes",
            "official_novel", "novel_raw_id_list", "novel_categories",
        ],
    }[split_name]
    found: List[int] = []

    def walk(x: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(x, dict):
            for k in keys:
                if k in x:
                    ids = _maybe_list_of_ids(x[k])
                    if ids is not None:
                        found = ids
                        return
            for k, v in x.items():
                if str(k).lower() == split_name:
                    ids = _maybe_list_of_ids(v)
                    if ids is not None:
                        found = ids
                        return
            for v in x.values():
                walk(v)
                if found:
                    return
        elif isinstance(x, list):
            records = [e for e in x if isinstance(e, dict)]
            if records and any(str(r.get("split", "")).lower() == split_name for r in records):
                vals: List[int] = []
                for r in records:
                    if str(r.get("split", "")).lower() == split_name:
                        val = r.get("raw_id", r.get("id", r.get("category_id")))
                        ii = _as_int(val)
                        if ii is not None:
                            vals.append(ii)
                if vals:
                    found = vals
                    return
            for v in x:
                walk(v)
                if found:
                    return

    walk(obj)
    if not found:
        raise KeyError(f"could not extract {split_name} ids from split json")
    return sorted({int(x) for x in found})


def _load_base_ids(split_json: Path) -> Set[int]:
    with Path(split_json).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return set(_extract_split_ids(obj, "base"))


def _load_clip_y_base(annotation_json: Path, base_ids: Set[int]) -> Dict[int, Set[int]]:
    with Path(annotation_json).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    anns = obj.get("annotations", []) if isinstance(obj, dict) else []
    out: Dict[int, Set[int]] = {}
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        clip = _as_int(ann.get("video_id", ann.get("clip_id", ann.get("image_id"))))
        cat = _as_int(ann.get("category_id", ann.get("raw_id", ann.get("raw_category_id"))))
        if clip is None or cat is None:
            continue
        if int(cat) in base_ids:
            out.setdefault(int(clip), set()).add(int(cat))
    return out


def _parse_epoch_plan(text: str) -> List[int]:
    vals = []
    for part in str(text or "").split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(max(0, int(float(part))))
    return vals or [1]


@dataclass(frozen=True)
class ResidualPeelingSchedule:
    mode: str
    variant: str
    csv_path: str
    annotation_json: str
    split_json: str
    epoch_plan: Tuple[int, ...]
    k_prev_by_round: Mapping[int, Set[int]]
    k_by_round: Mapping[int, Set[int]]
    c_by_round: Mapping[int, Set[int]]
    class_to_round: Mapping[int, int]
    class_to_certificate: Mapping[int, str]
    clip_id_to_y_base: Mapping[int, Set[int]]
    base_count: int

    def round_for_epoch(self, epoch_index_zero_based: int) -> int:
        epoch = int(epoch_index_zero_based)
        acc = 0
        for rid, width in enumerate(self.epoch_plan):
            acc += int(width)
            if epoch < acc:
                return rid
        return max(0, len(self.epoch_plan) - 1)

    def y_base_for_clip(self, clip_id: Any) -> Set[int]:
        ci = _as_int(clip_id)
        return set(self.clip_id_to_y_base.get(int(ci), set())) if ci is not None else set()

    def candidate_and_known_for_clip(self, clip_id: Any, round_id: int) -> Tuple[Set[int], Set[int], Dict[str, Any]]:
        y_base = self.y_base_for_clip(clip_id)
        rid = max(0, int(round_id))
        if rid == 0:
            known = set()
            candidates = y_base & set(self.k_by_round.get(0, set()))
        else:
            known = set(self.k_prev_by_round.get(rid, set()))
            candidates = y_base - known
        return candidates, known, {
            "residual_round_id": rid,
            "y_base_count": len(y_base),
            "candidate_count": len(candidates),
            "known_count": len(known),
        }

    def public_summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "variant": self.variant,
            "csv_path": self.csv_path,
            "annotation_json": self.annotation_json,
            "split_json": self.split_json,
            "epoch_plan": list(self.epoch_plan),
            "base_count": int(self.base_count),
            "round_count": int(len(self.epoch_plan)),
            "k_by_round_count": {str(k): len(v) for k, v in self.k_by_round.items()},
            "c_by_round_count": {str(k): len(v) for k, v in self.c_by_round.items()},
            "clip_context_count": int(len(self.clip_id_to_y_base)),
        }


def load_oracle_static_residual_schedule(
    *,
    csv_path: str | Path,
    annotation_json: str | Path,
    split_json: str | Path,
    variant: str = "person_aware",
    epoch_plan: str = "5,5,3,2",
) -> ResidualPeelingSchedule:
    csv_path = Path(csv_path)
    annotation_json = Path(annotation_json)
    split_json = Path(split_json)
    if not csv_path.is_file():
        raise FileNotFoundError(f"residual schedule csv not found: {csv_path}")
    if not annotation_json.is_file():
        raise FileNotFoundError(f"residual annotation json not found: {annotation_json}")
    if not split_json.is_file():
        raise FileNotFoundError(f"residual split json not found: {split_json}")

    base_ids = _load_base_ids(split_json)
    clip_y_base = _load_clip_y_base(annotation_json, base_ids)

    class_to_round: Dict[int, int] = {}
    class_to_certificate: Dict[int, str] = {}
    c_by_round: Dict[int, Set[int]] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant", "")) != str(variant):
                continue
            rid = _as_int(row.get("raw_id"))
            if rid is None or int(rid) not in base_ids:
                continue
            if not _truthy(row.get("resolved")):
                continue
            rr = _as_int(row.get("resolved_at_iteration"))
            if rr is None:
                continue
            class_to_round[int(rid)] = int(rr)
            class_to_certificate[int(rid)] = str(row.get("certificate_type", ""))
            c_by_round.setdefault(int(rr), set()).add(int(rid))

    if not c_by_round:
        raise RuntimeError(f"no resolved classes found for variant={variant!r} in {csv_path}")

    plan = tuple(_parse_epoch_plan(epoch_plan))
    max_round = max(max(c_by_round.keys()), len(plan) - 1)
    k_by_round: Dict[int, Set[int]] = {}
    k_prev_by_round: Dict[int, Set[int]] = {}
    known: Set[int] = set()
    for r in range(0, max_round + 1):
        if r == 0:
            known = set(c_by_round.get(0, set()))
            k_prev_by_round[0] = set()
            k_by_round[0] = set(known)
        else:
            k_prev_by_round[r] = set(known)
            known = set(known) | set(c_by_round.get(r, set()))
            k_by_round[r] = set(known)

    return ResidualPeelingSchedule(
        mode="oracle_static",
        variant=str(variant),
        csv_path=str(csv_path),
        annotation_json=str(annotation_json),
        split_json=str(split_json),
        epoch_plan=plan,
        k_prev_by_round=k_prev_by_round,
        k_by_round=k_by_round,
        c_by_round=c_by_round,
        class_to_round=class_to_round,
        class_to_certificate=class_to_certificate,
        clip_id_to_y_base=clip_y_base,
        base_count=len(base_ids),
    )


def apply_residual_candidate_override_to_groups(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    schedule: ResidualPeelingSchedule,
    epoch_index: int,
    candidate_policy: str = "base_residual",
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    if str(candidate_policy) != "base_residual":
        raise ValueError(f"only base_residual candidate policy is implemented in training overlay, got {candidate_policy!r}")
    rid = schedule.round_for_epoch(int(epoch_index))
    out_groups: List[List[Dict[str, Any]]] = []
    stats = {
        "residual_peeling_enabled": True,
        "residual_round_id": int(rid),
        "residual_group_count_input": int(len(groups)),
        "residual_group_count_output": 0,
        "residual_empty_candidate_group_count": 0,
        "residual_candidate_count_sum": 0,
        "residual_known_count_sum": 0,
    }
    for group in groups:
        if not group:
            continue
        clip_id = group[0].get("clip_id", group[0].get("video_id"))
        cand, known, meta = schedule.candidate_and_known_for_clip(clip_id, rid)
        if not cand:
            stats["residual_empty_candidate_group_count"] += 1
            continue
        cand_sorted = sorted(int(x) for x in cand)
        known_sorted = sorted(int(x) for x in known)
        new_group: List[Dict[str, Any]] = []
        for ex in group:
            row = dict(ex)
            # For prealign mode the packer uses observed_raw_ids. For aug mode it uses candidate_ids_known.
            row["observed_raw_ids"] = list(cand_sorted)
            row["candidate_ids_known"] = list(cand_sorted)
            row["candidate_ids_extra"] = []
            row["residual_peeling_round_id"] = int(rid)
            row["residual_known_raw_ids"] = list(known_sorted)
            row["residual_candidate_raw_ids"] = list(cand_sorted)
            row["residual_candidate_policy"] = str(candidate_policy)
            new_group.append(row)
        out_groups.append(new_group)
        stats["residual_group_count_output"] += 1
        stats["residual_candidate_count_sum"] += len(cand_sorted)
        stats["residual_known_count_sum"] += len(known_sorted)
    denom = max(1, int(stats["residual_group_count_output"]))
    stats["residual_candidate_count_mean"] = float(stats["residual_candidate_count_sum"] / denom)
    stats["residual_known_count_mean"] = float(stats["residual_known_count_sum"] / denom)
    return out_groups, stats
