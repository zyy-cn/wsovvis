#!/usr/bin/env python3
"""Read-only feasibility audit for soft explained-mass routing.

This audit tests whether an end-to-end soft residual routing mechanism would
retain enough residual signal on real VideoCutLER trajectories before training it.
It does not train, change checkpoints, or run LV-VIS eval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator

try:  # torch is required for checkpoint-backed scorer replay.
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from videocutler.ext_stageb_ovvis.banks.text_bank import load_text_vocab
    from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
except Exception as exc:  # pragma: no cover
    load_text_vocab = None  # type: ignore
    Projector = None  # type: ignore
    ProjectorConfig = None  # type: ignore
    _WSOVVIS_IMPORT_ERROR = exc
else:
    _WSOVVIS_IMPORT_ERROR = None


_VECTOR_LOCATOR_RE = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+)#(?P<key>[A-Za-z0-9_]+)\[(?P<idx>[0-9]+)\]$")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl_stream(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _p90(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(0.9 * (len(vals) - 1)))
    return float(vals[idx])


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return (arr / denom).astype(np.float32, copy=False)


def _parse_vector_locator(locator: str) -> Tuple[Path, str, int]:
    match = _VECTOR_LOCATOR_RE.match(str(locator))
    if not match:
        raise ValueError(f"invalid vector locator: {locator}")
    return Path(match.group("path")), str(match.group("key")), int(match.group("idx"))


class VectorReader:
    def __init__(self, artifact_parent_dir: Path) -> None:
        self.artifact_parent_dir = Path(artifact_parent_dir)
        self._reads = 0

    def close(self) -> None:
        self._reads = 0

    def read(self, locator: str) -> np.ndarray:
        self._reads += 1
        return np.asarray(read_vector_from_locator(self.artifact_parent_dir, locator), dtype=np.float32)


def _record_clip_id(record: Mapping[str, Any]) -> str:
    for key in ("clip_id", "video_id", "video_name"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    return ""


def _record_trajectory_id(record: Mapping[str, Any]) -> str:
    for key in ("trajectory_id", "track_id", "traj_id"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    return ""


def _record_vector_locator(record: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "z_norm_path",
        "z_raw_path",
        "traj_vector_locator",
        "trajectory_vector_locator",
        "vector_locator",
        "carrier_locator",
        "z_norm_locator",
        "feature_locator",
        "traj_locator",
    ):
        value = record.get(key)
        if value and "#" in str(value):
            return str(value)
    value = record.get("frame_carriers_norm_paths")
    if isinstance(value, Sequence) and value:
        first = value[0]
        if isinstance(first, str) and "#" in first:
            return first
    # Some records keep a locator dict.
    for key in ("locator", "vector", "traj_vector"):
        value = record.get(key)
        if isinstance(value, str) and "#" in value:
            return value
        if isinstance(value, Mapping):
            for sub in ("path", "locator", "z_norm"):
                v = value.get(sub)
                if isinstance(v, str) and "#" in v:
                    return v
    return None


@dataclass
class ClassInfo:
    raw_id: int
    name: str = ""
    is_base: bool = False
    is_novel: bool = False
    base_observed: Optional[bool] = None
    resolved_round: int = -1
    certificate_type: str = "unknown"
    person_conditioned: bool = False
    anchor_conditioned: bool = False


class StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: Dict[str, float] = defaultdict(float)
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.counters: Counter[str] = Counter()

    def add(self, row: Mapping[str, Any], numeric_fields: Sequence[str], value_fields: Sequence[str] = ()) -> None:
        self.count += 1
        for field in numeric_fields:
            val = _to_float(row.get(field), 0.0)
            if math.isfinite(val):
                self.sum[field] += float(val)
        for field in value_fields:
            val = _to_float(row.get(field), 0.0)
            if math.isfinite(val):
                self.values[field].append(float(val))

    def row(self, prefix: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(prefix)
        out["row_count"] = int(self.count)
        for k, v in sorted(self.sum.items()):
            out[f"{k}_sum"] = float(v)
            out[f"{k}_mean"] = float(v / self.count) if self.count else 0.0
        for k, vals in sorted(self.values.items()):
            out[f"{k}_mean"] = _mean(vals)
            out[f"{k}_median"] = _median(vals)
            out[f"{k}_p90"] = _p90(vals)
        return out


def _load_split(split_json: Path) -> Tuple[set[int], set[int]]:
    payload = _load_json(split_json)
    base = {int(x) for x in payload.get("base_raw_ids", [])}
    novel = {int(x) for x in payload.get("novel_raw_ids", [])}
    return base, novel


def _load_annotation_context(annotation_json: Path, base_ids: set[int]) -> Tuple[Dict[str, set[int]], Dict[int, str], Counter[int]]:
    payload = _load_json(annotation_json)
    cat_name = {int(c["id"]): str(c.get("name", c.get("id"))) for c in payload.get("categories", [])}
    by_clip: Dict[str, set[int]] = defaultdict(set)
    class_clip_counter: Dict[int, set[str]] = defaultdict(set)
    for ann in payload.get("annotations", []):
        raw_id = _to_int(ann.get("category_id"))
        video_id = ann.get("video_id", ann.get("vid_id", ann.get("clip_id")))
        if raw_id is None or video_id is None:
            continue
        clip_id = str(video_id)
        if int(raw_id) in base_ids:
            by_clip[clip_id].add(int(raw_id))
            class_clip_counter[int(raw_id)].add(clip_id)
    return dict(by_clip), cat_name, Counter({rid: len(clips) for rid, clips in class_clip_counter.items()})


def _load_schedule(schedule_csv: Path, class_info: Dict[int, ClassInfo]) -> None:
    if not schedule_csv.is_file():
        return
    with schedule_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = _to_int(row.get("raw_category_id", row.get("raw_id", row.get("category_id"))))
            if raw_id is None:
                continue
            info = class_info.setdefault(int(raw_id), ClassInfo(raw_id=int(raw_id)))
            for field in ("class_name", "name", "category_name"):
                if row.get(field):
                    info.name = str(row[field])
                    break
            rr = _to_int(row.get("resolved_at_iteration", row.get("resolved_round", row.get("iteration"))), default=None)
            if rr is not None:
                info.resolved_round = int(rr)
            cert = row.get("certificate_type", row.get("bucket", row.get("status")))
            if cert:
                info.certificate_type = str(cert)
            for field in ("base_observed", "is_base_observed", "observed"):
                val = _to_bool(row.get(field))
                if val is not None:
                    info.base_observed = bool(val)
                    break
            pc = _to_bool(row.get("person_conditioned"))
            ac = _to_bool(row.get("anchor_conditioned"))
            if pc is not None:
                info.person_conditioned = bool(pc)
            if ac is not None:
                info.anchor_conditioned = bool(ac)
            if "person_conditioned" in info.certificate_type:
                info.person_conditioned = True
            if "anchor_conditioned" in info.certificate_type or "context_identifiable" in info.certificate_type:
                info.anchor_conditioned = True


def _load_class_coverage(class_coverage_csv: Optional[Path], class_info: Dict[int, ClassInfo]) -> None:
    if class_coverage_csv is None or not class_coverage_csv.is_file():
        return
    with class_coverage_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = _to_int(row.get("raw_category_id", row.get("raw_id", row.get("category_id"))))
            if raw_id is None:
                continue
            info = class_info.setdefault(int(raw_id), ClassInfo(raw_id=int(raw_id)))
            bo = _to_bool(row.get("base_observed"))
            bu = _to_bool(row.get("base_unobserved"))
            if bo is not None:
                info.base_observed = bool(bo)
            elif bu is not None:
                info.base_observed = not bool(bu)
            rr = _to_int(row.get("resolved_at_iteration"), default=None)
            if rr is not None:
                info.resolved_round = int(rr)
            cert = row.get("certificate_type")
            if cert:
                info.certificate_type = str(cert)


def _load_gt_match(path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if path is None or not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vid = str(row.get("video_id", row.get("clip_id", ""))).strip()
            tid = str(row.get("trajectory_id", row.get("traj_id", ""))).strip()
            if not vid or not tid:
                continue
            raw_id = _to_int(row.get("best_gt_raw_id", row.get("matched_gt_raw_id", row.get("category_id"))))
            row2 = dict(row)
            row2["best_gt_raw_id_int"] = int(raw_id) if raw_id is not None else None
            row2["best_gt_iou_float"] = _to_float(row.get("best_gt_iou", row.get("iou")), 0.0)
            mapping[(vid, tid)] = row2
    return mapping


def _load_checkpoint_projector(checkpoint_path: Path, device: Any):
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    if Projector is None or ProjectorConfig is None:
        raise RuntimeError(f"WSOVVIS imports failed: {_WSOVVIS_IMPORT_ERROR}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = dict(checkpoint.get("text_projector_config", {}))
    projector = Projector(ProjectorConfig(
        input_dim=int(cfg.get("input_dim", 512)),
        hidden_dim=int(cfg.get("hidden_dim", 1024)),
        output_dim=int(cfg.get("output_dim", 768)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_layernorm=bool(cfg.get("use_layernorm", True)),
    )).to(device)
    projector.load_state_dict(checkpoint["text_projector_state_dict"])
    projector.eval()
    theta_raw = float(checkpoint.get("theta_T", 0.0))
    theta = torch.tensor(theta_raw, device=device, dtype=torch.float32)
    temperature = F.softplus(theta) + 1e-4
    return projector, temperature, checkpoint


def _project_text_matrix(projector: Any, text_matrix: np.ndarray, device: Any, batch: int = 4096) -> Any:
    chunks = []
    with torch.no_grad():
        for start in range(0, int(text_matrix.shape[0]), int(batch)):
            x = torch.from_numpy(np.asarray(text_matrix[start:start+batch], dtype=np.float32)).to(device=device, dtype=torch.float32)
            chunks.append(projector(x).detach())
    return torch.cat(chunks, dim=0)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    denom = float(np.sum(exp))
    if denom <= 1e-30:
        return np.ones_like(arr, dtype=np.float64) / max(1, arr.size)
    return exp / denom


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _variant_weight(conf: float, tau: float, gamma: float, *, variant: str, top1_raw_id: Optional[int], info: Optional[ClassInfo], rare_floor: float, hub_cap: float, hub_ids: set[int]) -> Tuple[float, float]:
    explained = _sigmoid(float(gamma) * (float(conf) - float(tau)))
    if "hubcap" in variant and top1_raw_id is not None and int(top1_raw_id) in hub_ids:
        explained = min(float(explained), float(hub_cap))
    residual_weight = 1.0 - float(explained)
    if "floor" in variant:
        protect = False
        if info is not None:
            # Uses only class-level metadata of model top1, not row-level GT.
            if info.base_observed is False:
                protect = True
            if info.resolved_round >= 1:
                protect = True
            if "unobserved" in str(info.certificate_type).lower() or "low_support" in str(info.certificate_type).lower():
                protect = True
        if protect:
            residual_weight = max(float(residual_weight), float(rare_floor))
    return float(residual_weight), float(explained)


def _group_keys_for_row(row: Mapping[str, Any]) -> Dict[str, str]:
    info: Optional[ClassInfo] = row.get("gt_class_info")  # type: ignore
    top1_info: Optional[ClassInfo] = row.get("top1_class_info")  # type: ignore
    out = {
        "round": str(info.resolved_round if info is not None else "unknown"),
        "certificate": str(info.certificate_type if info is not None else "unknown"),
        "observed_unobserved": "unknown",
        "top1_observed_unobserved": "unknown",
        "iou_bucket": str(row.get("iou_bucket", "unknown")),
    }
    if info is not None and info.base_observed is not None:
        out["observed_unobserved"] = "base_observed" if info.base_observed else "base_unobserved"
    if top1_info is not None and top1_info.base_observed is not None:
        out["top1_observed_unobserved"] = "base_observed" if top1_info.base_observed else "base_unobserved"
    if info is not None:
        if info.person_conditioned:
            out["certificate_family"] = "person_conditioned"
        elif info.anchor_conditioned:
            out["certificate_family"] = "anchor_conditioned"
        else:
            out["certificate_family"] = "other"
    else:
        out["certificate_family"] = "unknown"
    return out


def _iou_bucket(iou: float) -> str:
    if iou >= 0.7:
        return "iou_ge_0.7"
    if iou >= 0.5:
        return "iou_0.5_0.7"
    if iou >= 0.3:
        return "iou_0.3_0.5"
    if iou > 0.0:
        return "iou_0_0.3"
    return "no_iou"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    if load_text_vocab is None:
        raise RuntimeError(f"WSOVVIS imports failed: {_WSOVVIS_IMPORT_ERROR}")

    run_root = Path(args.run_root).resolve()
    asset_root = Path(args.asset_root).resolve()
    dataset_name = str(args.dataset_name)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis" / "soft_explained_mass_routing_feasibility" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ids, novel_ids = _load_split(Path(args.split_json))
    clip_y_base, cat_name, train_clip_count = _load_annotation_context(Path(args.annotation_json), base_ids)
    class_info: Dict[int, ClassInfo] = {}
    for rid in base_ids:
        class_info[int(rid)] = ClassInfo(raw_id=int(rid), name=cat_name.get(int(rid), ""), is_base=True, is_novel=False)
    for rid in novel_ids:
        class_info[int(rid)] = ClassInfo(raw_id=int(rid), name=cat_name.get(int(rid), ""), is_base=False, is_novel=True)
    _load_schedule(Path(args.schedule_csv), class_info)
    if args.class_coverage_csv:
        _load_class_coverage(Path(args.class_coverage_csv), class_info)
    else:
        default_cov = run_root / "analysis" / "rcp_training_dynamics_diagnosis" / dataset_name / "class_level_residual_coverage.csv"
        _load_class_coverage(default_cov, class_info)

    gt_match = _load_gt_match(Path(args.gt_match_csv) if args.gt_match_csv else None)

    text_vocab_ids, _records, text_matrix = load_text_vocab(asset_root)
    text_raw_ids = [int(x) for x in text_vocab_ids]
    raw_to_text_index = {int(rid): idx for idx, rid in enumerate(text_raw_ids)}

    carrier_dir = Path(args.carrier_bank_dir) if args.carrier_bank_dir else asset_root / "carrier_bank" / dataset_name
    carrier_records = carrier_dir / "carrier_records.jsonl"
    if not carrier_records.is_file():
        raise FileNotFoundError(f"carrier_records not found: {carrier_records}")

    device = torch.device(str(args.device))
    checkpoints = []
    for spec in args.checkpoint:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).parents[2].name if len(Path(path).parents) > 2 else Path(path).stem
        checkpoints.append((str(name), Path(path)))
    if not checkpoints:
        raise ValueError("at least one --checkpoint NAME=PATH is required")

    tau_values = [float(x) for x in str(args.tau_values).split(",") if str(x).strip()]
    gamma_values = [float(x) for x in str(args.gamma_values).split(",") if str(x).strip()]
    variants = [str(x).strip() for x in str(args.variants).split(",") if str(x).strip()]
    hub_ids = {int(x) for x in str(args.hub_raw_ids).split(",") if str(x).strip()}

    # Load carrier records into lightweight grouped metadata only; vector payloads remain lazy.
    clip_records: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    carrier_record_sample_keys: List[List[str]] = []
    accepted_locator_fields = [
        "z_norm_path",
        "z_raw_path",
        "traj_vector_locator",
        "trajectory_vector_locator",
        "vector_locator",
        "carrier_locator",
        "z_norm_locator",
        "feature_locator",
        "traj_locator",
        "frame_carriers_norm_paths",
        "locator",
        "vector",
        "traj_vector",
    ]
    for record in _read_jsonl_stream(carrier_records):
        if len(carrier_record_sample_keys) < 5:
            carrier_record_sample_keys.append(sorted(str(k) for k in record.keys()))
        clip_id = _record_clip_id(record)
        traj_id = _record_trajectory_id(record)
        locator = _record_vector_locator(record)
        if not clip_id:
            counters["skip_no_clip_id"] += 1
            continue
        if not traj_id:
            counters["skip_no_trajectory_id"] += 1
            continue
        if not locator:
            counters["skip_no_vector_locator"] += 1
            continue
        if clip_id not in clip_y_base or not clip_y_base.get(clip_id):
            counters["skip_no_y_base"] += 1
            continue
        clip_records[clip_id].append((traj_id, locator))
        counters["carrier_records_usable"] += 1
        if args.max_rows and counters["carrier_records_usable"] >= int(args.max_rows):
            break

    checkpoint_summaries: List[Dict[str, Any]] = []
    rows_by_tau_gamma: List[Dict[str, Any]] = []
    rows_by_checkpoint: List[Dict[str, Any]] = []
    group_accumulators: Dict[Tuple[str, str, str, str, float, float], StreamingStats] = defaultdict(StreamingStats)
    hub_rows: Dict[Tuple[str, str, float, float, str], StreamingStats] = defaultdict(StreamingStats)
    failure_examples_path = output_dir / "soft_routing_failure_examples.jsonl"
    if failure_examples_path.exists():
        failure_examples_path.unlink()

    numeric_fields = [
        "residual_weight", "explained_mass", "conf", "margin", "entropy", "top1_is_gt",
        "gt_rank", "normalized_gt_rank", "is_gt_iou_ge_0.5", "is_gt_base_observed", "is_gt_base_unobserved",
    ]
    value_fields = ["residual_weight", "conf", "margin", "entropy", "normalized_gt_rank"]

    vector_reader = VectorReader(carrier_dir)
    try:
        for checkpoint_name, checkpoint_path in checkpoints:
            projector, temperature, checkpoint_payload = _load_checkpoint_projector(checkpoint_path, device)
            projected_text = _project_text_matrix(projector, np.asarray(text_matrix, dtype=np.float32), device)
            projected_text = F.normalize(projected_text, p=2.0, dim=-1)
            temperature_float = float(temperature.detach().cpu().item())

            stat_by_setting: Dict[Tuple[str, float, float], StreamingStats] = defaultdict(StreamingStats)
            setting_top1_counter: Counter[Tuple[str, float, float, int]] = Counter()
            processed_rows = 0
            skipped_rows = Counter()

            for clip_id, entries in clip_records.items():
                y_base = sorted(int(x) for x in clip_y_base.get(str(clip_id), set()) if int(x) in raw_to_text_index)
                if not y_base:
                    skipped_rows["empty_y_base"] += len(entries)
                    continue
                y_indices = [raw_to_text_index[int(rid)] for rid in y_base]
                y_tensor = projected_text[y_indices]
                vectors: List[np.ndarray] = []
                row_meta: List[Tuple[str, str]] = []
                for traj_id, locator in entries:
                    try:
                        vec = _normalize(vector_reader.read(locator))
                    except Exception as exc:
                        skipped_rows[f"vector_read_error:{type(exc).__name__}"] += 1
                        continue
                    if int(vec.shape[0]) != int(projected_text.shape[1]):
                        skipped_rows[f"dim_mismatch:{vec.shape[0]}_vs_{int(projected_text.shape[1])}"] += 1
                        continue
                    vectors.append(vec)
                    row_meta.append((traj_id, locator))
                if not vectors:
                    continue

                z = torch.from_numpy(np.stack(vectors, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32)
                z = F.normalize(z, p=2.0, dim=-1)
                with torch.no_grad():
                    logits = torch.matmul(z, y_tensor.t()) / temperature
                logits_np = logits.detach().cpu().numpy().astype(np.float32)

                for row_idx, (traj_id, _locator) in enumerate(row_meta):
                    logit_row = logits_np[row_idx]
                    if logit_row.size == 0:
                        continue
                    probs = _softmax_np(logit_row)
                    order = np.argsort(-logit_row)
                    top1_local = int(order[0])
                    top2_local = int(order[1]) if int(order.size) > 1 else int(order[0])
                    top1_raw_id = int(y_base[top1_local])
                    top2_raw_id = int(y_base[top2_local])
                    conf = float(probs[top1_local])
                    margin = float(logit_row[top1_local] - logit_row[top2_local]) if int(order.size) > 1 else float(abs(logit_row[top1_local]))
                    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / max(math.log(max(2, int(probs.size))), 1e-12))
                    top1_info = class_info.get(top1_raw_id)

                    gt = gt_match.get((str(clip_id), str(traj_id)), {})
                    gt_raw_id = gt.get("best_gt_raw_id_int")
                    gt_iou = float(gt.get("best_gt_iou_float", 0.0) or 0.0)
                    gt_info = class_info.get(int(gt_raw_id)) if gt_raw_id is not None else None
                    gt_rank = None
                    norm_rank = None
                    top1_is_gt = 0.0
                    if gt_raw_id is not None and int(gt_raw_id) in y_base:
                        gt_local = y_base.index(int(gt_raw_id))
                        gt_rank = int(np.where(order == gt_local)[0][0]) + 1
                        norm_rank = float((gt_rank - 1) / max(1, int(len(y_base) - 1)))
                        top1_is_gt = 1.0 if int(top1_raw_id) == int(gt_raw_id) else 0.0

                    base_row: Dict[str, Any] = {
                        "checkpoint": checkpoint_name,
                        "clip_id": str(clip_id),
                        "trajectory_id": str(traj_id),
                        "top1_raw_id": int(top1_raw_id),
                        "top2_raw_id": int(top2_raw_id),
                        "top1_class_info": top1_info,
                        "gt_raw_id": int(gt_raw_id) if gt_raw_id is not None else "",
                        "gt_class_info": gt_info,
                        "gt_iou": float(gt_iou),
                        "iou_bucket": _iou_bucket(gt_iou),
                        "conf": conf,
                        "margin": margin,
                        "entropy": entropy,
                        "gt_rank": float(gt_rank or 0),
                        "normalized_gt_rank": float(norm_rank) if norm_rank is not None else 0.0,
                        "top1_is_gt": top1_is_gt,
                        "is_gt_iou_ge_0.5": 1.0 if gt_iou >= 0.5 else 0.0,
                        "is_gt_base_observed": 1.0 if gt_info is not None and gt_info.base_observed is True else 0.0,
                        "is_gt_base_unobserved": 1.0 if gt_info is not None and gt_info.base_observed is False else 0.0,
                    }
                    processed_rows += 1
                    if processed_rows <= int(args.top_examples) and (gt_info is not None and gt_info.base_observed is False):
                        _append_jsonl(failure_examples_path, {
                            "checkpoint": checkpoint_name,
                            "clip_id": clip_id,
                            "trajectory_id": traj_id,
                            "gt_raw_id": int(gt_raw_id) if gt_raw_id is not None else None,
                            "gt_class_name": gt_info.name if gt_info else "",
                            "top1_raw_id": top1_raw_id,
                            "top1_class_name": top1_info.name if top1_info else "",
                            "conf": conf,
                            "margin": margin,
                            "entropy": entropy,
                            "gt_iou": gt_iou,
                        })
                    for tau in tau_values:
                        for gamma in gamma_values:
                            for variant in variants:
                                weight, explained = _variant_weight(
                                    conf, tau, gamma,
                                    variant=variant,
                                    top1_raw_id=top1_raw_id,
                                    info=top1_info,
                                    rare_floor=float(args.rare_floor),
                                    hub_cap=float(args.hub_cap),
                                    hub_ids=hub_ids,
                                )
                                row = dict(base_row)
                                row.update({
                                    "tau": float(tau),
                                    "gamma": float(gamma),
                                    "variant": variant,
                                    "residual_weight": float(weight),
                                    "explained_mass": float(explained),
                                })
                                setting_key = (variant, float(tau), float(gamma))
                                stat_by_setting[setting_key].add(row, numeric_fields, value_fields)
                                setting_top1_counter[(variant, float(tau), float(gamma), int(top1_raw_id))] += 1
                                keys = _group_keys_for_row(row)
                                for group_name, group_val in keys.items():
                                    group_accumulators[(checkpoint_name, variant, group_name, group_val, float(tau), float(gamma))].add(row, numeric_fields, value_fields)
                                if int(top1_raw_id) in hub_ids:
                                    hub_rows[(checkpoint_name, variant, float(tau), float(gamma), str(top1_raw_id))].add(row, numeric_fields, value_fields)

            for (variant, tau, gamma), stats in sorted(stat_by_setting.items()):
                out = stats.row({"checkpoint": checkpoint_name, "variant": variant, "tau": tau, "gamma": gamma})
                row_count = int(out.get("row_count", 0))
                weight_sum = float(out.get("residual_weight_sum", 0.0))
                out["effective_signal_retention_rate"] = float(weight_sum / max(1, row_count))
                out["processed_rows"] = int(processed_rows)
                out["temperature"] = float(temperature_float)
                # hub share under top1 assignments.
                top_total = sum(v for k, v in setting_top1_counter.items() if k[:3] == (variant, tau, gamma))
                hub_total = sum(v for k, v in setting_top1_counter.items() if k[:3] == (variant, tau, gamma) and int(k[3]) in hub_ids)
                out["top1_hub_share"] = float(hub_total / max(1, top_total))
                rows_by_tau_gamma.append(out)
            checkpoint_summaries.append({
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "processed_rows": int(processed_rows),
                "skipped_rows": dict(skipped_rows),
                "temperature": float(temperature_float),
                "projected_text_shape": list(projected_text.shape),
            })

    finally:
        vector_reader.close()

    if checkpoint_summaries and all(int(row.get("processed_rows", 0)) == 0 for row in checkpoint_summaries):
        raise RuntimeError(
            "FAIL: processed_rows were zero for every checkpoint; carrier locator parsing or vector loading is broken"
        )

    # Aggregate by checkpoint using default setting, or first available setting.
    default_variant = args.default_variant
    default_tau = float(args.default_tau)
    default_gamma = float(args.default_gamma)
    selected = [r for r in rows_by_tau_gamma if r.get("variant") == default_variant and float(r.get("tau")) == default_tau and float(r.get("gamma")) == default_gamma]
    if not selected and rows_by_tau_gamma:
        selected = [rows_by_tau_gamma[0]]
    for r in selected:
        rows_by_checkpoint.append({k: v for k, v in r.items() if k in {
            "checkpoint", "variant", "tau", "gamma", "row_count", "residual_weight_mean", "residual_weight_sum",
            "effective_signal_retention_rate", "conf_mean", "margin_mean", "entropy_mean", "top1_is_gt_mean",
            "normalized_gt_rank_mean", "is_gt_base_observed_sum", "is_gt_base_unobserved_sum", "top1_hub_share",
        }})

    group_rows: List[Dict[str, Any]] = []
    for (checkpoint, variant, group_name, group_val, tau, gamma), stats in sorted(group_accumulators.items()):
        # Keep all groups for default setting; also keep other settings for top-level csv if useful.
        if float(tau) == default_tau and float(gamma) == default_gamma and str(variant) == default_variant:
            row = stats.row({"checkpoint": checkpoint, "variant": variant, "tau": tau, "gamma": gamma, "group_name": group_name, "group_value": group_val})
            row["effective_signal_retention_rate"] = float(row.get("residual_weight_sum", 0.0) / max(1, int(row.get("row_count", 0))))
            group_rows.append(row)

    def filter_group(name: str) -> List[Dict[str, Any]]:
        return [r for r in group_rows if r.get("group_name") == name]

    hub_summary_rows: List[Dict[str, Any]] = []
    for (checkpoint, variant, tau, gamma, hub_raw_id), stats in sorted(hub_rows.items()):
        if float(tau) == default_tau and float(gamma) == default_gamma and str(variant) == default_variant:
            info = class_info.get(int(hub_raw_id))
            row = stats.row({"checkpoint": checkpoint, "variant": variant, "tau": tau, "gamma": gamma, "hub_raw_id": hub_raw_id, "hub_name": info.name if info else ""})
            row["effective_signal_retention_rate"] = float(row.get("residual_weight_sum", 0.0) / max(1, int(row.get("row_count", 0))))
            hub_summary_rows.append(row)

    summary = {
        "status": "PASS",
        "run_root": str(run_root),
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "asset_root": str(asset_root),
        "carrier_records": str(carrier_records),
        "carrier_record_sample_keys": carrier_record_sample_keys,
        "accepted_locator_fields": accepted_locator_fields,
        "annotation_json": str(args.annotation_json),
        "split_json": str(args.split_json),
        "schedule_csv": str(args.schedule_csv),
        "gt_match_csv": str(args.gt_match_csv or ""),
        "base_count": len(base_ids),
        "novel_count": len(novel_ids),
        "clip_context_count": len(clip_y_base),
        "carrier_record_counters": dict(counters),
        "processed_rows_by_checkpoint": {row["checkpoint"]: int(row.get("processed_rows", 0)) for row in checkpoint_summaries},
        "skipped_by_reason": {str(k): int(v) for k, v in sorted(dict(counters).items())},
        "checkpoints": checkpoint_summaries,
        "tau_values": tau_values,
        "gamma_values": gamma_values,
        "variants": variants,
        "default_setting": {"variant": default_variant, "tau": default_tau, "gamma": default_gamma},
        "interpretation": {
            "purpose": "Read-only feasibility check for soft explained-mass routing on real VideoCutLER trajectories.",
            "routing_uses_gt": False,
            "gt_use": "GT match is used only for stratified audit statistics, not for routing weights.",
            "pass_criterion": "Prefer settings with non-zero retained signal, better base_unobserved retention, and controlled hub/person explained mass.",
        },
    }

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "summary_by_tau_gamma.csv", rows_by_tau_gamma)
    _write_csv(output_dir / "summary_by_checkpoint.csv", rows_by_checkpoint)
    _write_csv(output_dir / "summary_by_round.csv", filter_group("round"))
    _write_csv(output_dir / "summary_by_certificate.csv", filter_group("certificate"))
    _write_csv(output_dir / "summary_by_certificate_family.csv", filter_group("certificate_family"))
    _write_csv(output_dir / "summary_by_observed_unobserved.csv", filter_group("observed_unobserved"))
    _write_csv(output_dir / "summary_by_top1_observed_unobserved.csv", filter_group("top1_observed_unobserved"))
    _write_csv(output_dir / "summary_by_iou_bucket.csv", filter_group("iou_bucket"))
    _write_csv(output_dir / "summary_by_hub_class.csv", hub_summary_rows)

    takeover = output_dir / "SOFT_EXPLAINED_MASS_ROUTING_FEASIBILITY_TAKEOVER.md"
    takeover.write_text(
        "# Soft Explained-Mass Routing Feasibility\n\n"
        "Status: `PASS`\n\n"
        f"Output: `{output_dir}`\n\n"
        "This is a read-only simulation on real VideoCutLER trajectory carriers. "
        "No training, checkpoint modification, or LV-VIS eval was run.\n\n"
        "Key files:\n"
        "- `summary.json`\n"
        "- `summary_by_tau_gamma.csv`\n"
        "- `summary_by_checkpoint.csv`\n"
        "- `summary_by_round.csv`\n"
        "- `summary_by_certificate.csv`\n"
        "- `summary_by_observed_unobserved.csv`\n"
        "- `summary_by_hub_class.csv`\n\n"
        "Interpretation guide:\n"
        "- If `effective_signal_retention_rate` collapses toward 0, soft routing is still too aggressive.\n"
        "- If `base_unobserved` retained signal stays much lower than `base_observed`, add demand floor/class-balanced sampling before training.\n"
        "- If hub/person top1 mass dominates, use hub cap or IDF prior before training.\n",
        encoding="utf-8",
    )

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soft explained-mass routing feasibility audit")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset_name", default="lvvis_train_base")
    parser.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    parser.add_argument("--carrier_bank_dir", default="")
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--schedule_csv", required=True)
    parser.add_argument("--class_coverage_csv", default="")
    parser.add_argument("--gt_match_csv", default="")
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH; may be repeated")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau_values", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--gamma_values", default="5,10,20")
    parser.add_argument("--variants", default="plain,floor,hubcap,floor_hubcap")
    parser.add_argument("--default_variant", default="floor_hubcap")
    parser.add_argument("--default_tau", type=float, default=0.7)
    parser.add_argument("--default_gamma", type=float, default=10.0)
    parser.add_argument("--rare_floor", type=float, default=0.25)
    parser.add_argument("--hub_cap", type=float, default=0.75)
    parser.add_argument("--hub_raw_ids", default="773")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--top_examples", type=int, default=128)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
