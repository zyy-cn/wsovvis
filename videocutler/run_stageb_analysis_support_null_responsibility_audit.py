from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.analysis.extra_attribution_probe import (  # noqa: E402
    ExtraAttributionProbeConfig,
    _apply_stage_candidate_overrides,
    _load_stage_responsibility_candidate_overrides,
    _materialize_valid_samples,
    _prepare_probe_examples,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label  # noqa: E402
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup  # noqa: E402
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_base_and_novel_raw_ids  # noqa: E402

Record = Dict[str, Any]


@dataclass(frozen=True)
class Config:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str
    trajectory_source_branch: str
    stage: str
    output_dir: Optional[Path]
    sidecar_root: Optional[Path]
    hub_raw_ids: Tuple[int, ...]
    write_rows: bool
    top_examples: int
    smoke: bool
    smoke_max_trajectories: int
    subset_fraction: Optional[float]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def _parse_ints(value: str) -> Tuple[int, ...]:
    out: List[int] = []
    for p in str(value).replace(";", ",").split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return tuple(out)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_json(path: Path) -> Optional[Record]:
    try:
        if path.is_file():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    return None


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _unique_ints(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, Mapping):
        parts = value.keys()
    elif isinstance(value, Iterable):
        parts = value
    else:
        parts = [value]
    out: List[int] = []
    seen = set()
    for x in parts:
        ix = _safe_int(x)
        if ix is None or ix in seen:
            continue
        out.append(ix)
        seen.add(ix)
    return out


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else None


def _median(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(median(vals)) if vals else None


def _rate(n: int, d: int) -> Optional[float]:
    return float(n / d) if d else None


def _default_output_dir(run_root: Path, dataset_name: str, stage: str) -> Path:
    return run_root / "analysis" / "support_null_responsibility" / dataset_name / stage


def _sidecar_root(config: Config) -> Path:
    return Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root is not None else Path(config.run_root).expanduser().resolve()


def _read_responsibility_rows(run_root: Path, stage: str) -> Tuple[Dict[str, Record], Dict[str, Any]]:
    candidates = [
        run_root / "train" / stage / "responsibility_records.jsonl",
        run_root / "train" / stage / "proxy_records.jsonl",
    ]
    chosen = next((p for p in candidates if p.is_file()), candidates[0])
    by_tid: Dict[str, Record] = {}
    total = 0
    null_candidate_count = 0
    demand_candidate_count = 0
    for row in _iter_jsonl(chosen):
        total += 1
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            by_tid[tid] = row
        r = row.get("r_final") if isinstance(row.get("r_final"), Mapping) else {}
        if "-1" in r or -1 in r:
            null_candidate_count += 1
        d = row.get("candidate_demand_by_raw_id")
        if isinstance(d, Mapping) and ("-1" in d or -1 in d):
            demand_candidate_count += 1
    return by_tid, {
        "path": str(chosen),
        "exists": chosen.is_file(),
        "record_count": int(total),
        "by_tid_count": int(len(by_tid)),
        "null_candidate_row_count": int(null_candidate_count),
        "demand_candidate_row_count": int(demand_candidate_count),
        "fallback_used": str(chosen.name) != "responsibility_records.jsonl",
    }


def _r_value(resp_row: Mapping[str, Any], raw_id: int) -> Optional[float]:
    for key in ("r_final", "responsibility_final", "responsibilities", "r", "R_final"):
        obj = resp_row.get(key)
        if isinstance(obj, Mapping):
            for rk in (str(raw_id), raw_id):
                if rk in obj:
                    return _safe_float(obj.get(rk), default=None)
    return None


def _demand_value(resp_row: Mapping[str, Any], raw_id: int) -> Optional[float]:
    for key in ("candidate_demand_by_raw_id", "demand_by_raw_id", "candidate_demands"):
        obj = resp_row.get(key)
        if isinstance(obj, Mapping):
            for rk in (str(raw_id), raw_id):
                if rk in obj:
                    return _safe_float(obj.get(rk), default=None)
    return None


def _top_raw_id(resp_row: Mapping[str, Any]) -> Optional[int]:
    obj = resp_row.get("r_final")
    if not isinstance(obj, Mapping) or not obj:
        return None
    best = None
    best_val = -1.0
    for k, v in obj.items():
        rid = _safe_int(k, None)
        val = _safe_float(v, None)
        if rid is None or val is None:
            continue
        if val > best_val:
            best = int(rid)
            best_val = float(val)
    return best


def _matched_gt_raw_id(sidecar: Mapping[str, Any]) -> Optional[int]:
    for key in ("matched_gt_raw_id_canonical", "gt_raw_id_canonical", "matched_gt_raw_id", "gt_raw_id", "category_id"):
        val = _safe_int(sidecar.get(key), None)
        if val is not None:
            return val
    for key in ("matched_gt", "gt", "match"):
        obj = sidecar.get(key)
        if isinstance(obj, Mapping):
            for kk in ("raw_id", "category_id", "id", "gt_raw_id"):
                val = _safe_int(obj.get(kk), None)
                if val is not None:
                    return val
    return None


def run_audit(config: Config) -> Dict[str, Any]:
    run_root = Path(config.run_root).expanduser().resolve()
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir else _default_output_dir(run_root, config.dataset_name, config.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_config = ExtraAttributionProbeConfig(
        run_root=run_root,
        runtime_output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
        device="cpu",
        smoke=bool(config.smoke),
        smoke_max_trajectories=int(config.smoke_max_trajectories),
        subset_fraction=None if config.subset_fraction is None else float(config.subset_fraction),
        stage_scope=(str(config.stage),),
        batch_size=512,
        output_dir=output_dir,
        sidecar_root=_sidecar_root(config),
        show_progress=False,
    )
    materialized = _materialize_valid_samples(proxy_config)
    prepared = _prepare_probe_examples(
        list(materialized.get("valid_samples", [])),
        output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    examples = list(prepared.get("examples", []))
    if not examples:
        raise RuntimeError("no examples materialized; cannot audit support-null responsibility")

    stage_overrides, override_meta = _load_stage_responsibility_candidate_overrides(run_root=run_root, stage_id=str(config.stage))
    examples, scope_meta = _apply_stage_candidate_overrides(examples, stage_overrides, stage_id=str(config.stage))

    sidecar_lookup = load_gt_sidecar_lookup(_sidecar_root(config), dataset_name=str(config.dataset_name), trajectory_source_branch=str(config.trajectory_source_branch))
    base_vocab_ids, _novel = load_lvvis_base_and_novel_raw_ids()
    base_vocab_set = {int(x) for x in base_vocab_ids}
    resp_by_tid, resp_meta = _read_responsibility_rows(run_root, str(config.stage))
    hub_set = {int(x) for x in config.hub_raw_ids}

    row_meta: List[Record] = []
    by_clip: Dict[int, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        tid = str(ex.get("trajectory_id", "")).strip()
        clip_id = _safe_int(ex.get("clip_id"), -1)
        observed = _unique_ints(ex.get("observed_raw_ids")) or _unique_ints(ex.get("candidate_ids_known"))
        sidecar = sidecar_lookup.get(tid, {}) if tid else {}
        gt = _matched_gt_raw_id(sidecar) if isinstance(sidecar, Mapping) else None
        auditable = bool(gt is not None)
        if isinstance(sidecar, Mapping) and "audit_usable" in sidecar:
            auditable = bool(sidecar.get("audit_usable")) and gt is not None
        split = None
        if gt is not None:
            try:
                split = _all_gt_split_label(
                    dataset_name=str(config.dataset_name),
                    gt_raw_id=int(gt),
                    observed_raw_ids=observed,
                    base_vocab_ids=base_vocab_set,
                )
            except Exception:
                split = None
        resp = resp_by_tid.get(tid, {})
        null_mass = _r_value(resp, -1)
        top_raw = _top_raw_id(resp)
        m = {
            "row_index": int(idx),
            "trajectory_id": tid,
            "clip_id": int(clip_id if clip_id is not None else -1),
            "video_id": _safe_int(ex.get("video_id"), None),
            "observed_raw_ids": observed,
            "gt_raw_id": int(gt) if gt is not None else None,
            "auditable_gt": bool(auditable),
            "split": split,
            "has_responsibility": bool(resp),
            "null_mass": null_mass,
            "nonnull_mass": (1.0 - float(null_mass)) if null_mass is not None else None,
            "top_raw_id": top_raw,
            "top1_is_null": bool(top_raw == -1),
        }
        row_meta.append(m)
        if clip_id is not None:
            by_clip[int(clip_id)].append(int(idx))

    # Per trajectory NULL allocation oracle buckets.
    bucket_rows: Dict[str, List[Record]] = defaultdict(list)
    for m in row_meta:
        observed_set = {int(x) for x in m.get("observed_raw_ids", [])}
        gt = m.get("gt_raw_id")
        if gt is None or not bool(m.get("auditable_gt")):
            bucket = "unmatched_or_no_auditable_gt"
        elif int(gt) in observed_set:
            bucket = "yprime_matched_gt"
        else:
            bucket = "hidden_gt_not_in_yprime"
        row = {"bucket": bucket, **m}
        bucket_rows[bucket].append(row)

    bucket_summary_rows: List[Record] = []
    for bucket, rows in sorted(bucket_rows.items()):
        null_vals = [r.get("null_mass") for r in rows]
        nonnull_vals = [r.get("nonnull_mass") for r in rows]
        resp_rows = [r for r in rows if bool(r.get("has_responsibility")) and r.get("null_mass") is not None]
        bucket_summary_rows.append({
            "bucket": bucket,
            "row_count": int(len(rows)),
            "responsibility_row_count": int(len(resp_rows)),
            "mean_null_mass": _mean(null_vals),
            "median_null_mass": _median(null_vals),
            "mean_nonnull_mass": _mean(nonnull_vals),
            "top1_null_rate": _rate(sum(1 for r in resp_rows if bool(r.get("top1_is_null"))), len(resp_rows)),
        })

    # Per (clip, yprime) support / responsibility / demand calibration.
    pair_rows: List[Record] = []
    for clip_id, idxs in sorted(by_clip.items()):
        yprime: List[int] = []
        seen = set()
        for idx in idxs:
            for y in row_meta[idx].get("observed_raw_ids", []):
                if int(y) not in seen:
                    seen.add(int(y)); yprime.append(int(y))
        if not yprime:
            continue
        auditable_idxs = [idx for idx in idxs if bool(row_meta[idx].get("auditable_gt")) and row_meta[idx].get("gt_raw_id") is not None]
        # One row from this clip is enough to recover column demand because it is clip-level.
        first_resp = None
        for idx in idxs:
            r = resp_by_tid.get(str(row_meta[idx].get("trajectory_id", "")), {})
            if r:
                first_resp = r
                break
        for y in yprime:
            support_idxs = [idx for idx in auditable_idxs if int(row_meta[idx].get("gt_raw_id")) == int(y)]
            has_support = bool(support_idxs)
            total_mass_y = 0.0
            true_support_mass_y = 0.0
            best_tid = None
            best_gt = None
            best_mass = -1.0
            available = False
            for idx in idxs:
                tid = str(row_meta[idx].get("trajectory_id", ""))
                resp = resp_by_tid.get(tid, {})
                rv = _r_value(resp, int(y)) if resp else None
                if rv is None:
                    continue
                available = True
                val = float(rv)
                total_mass_y += val
                if row_meta[idx].get("gt_raw_id") is not None and int(row_meta[idx].get("gt_raw_id")) == int(y):
                    true_support_mass_y += val
                if val > best_mass:
                    best_mass = val
                    best_tid = tid
                    best_gt = row_meta[idx].get("gt_raw_id")
            ratio = float(true_support_mass_y / total_mass_y) if available and total_mass_y > 0 else None
            top1 = bool(available and has_support and best_gt == int(y))
            hub_hijack = bool(available and best_gt in hub_set and int(y) not in hub_set and not top1)
            demand_y = _demand_value(first_resp, int(y)) if first_resp else None
            pair_rows.append({
                "clip_id": int(clip_id),
                "yprime_raw_id": int(y),
                "has_trajectory_support": bool(has_support),
                "support_count": int(len(support_idxs)),
                "responsibility_available_for_y": bool(available),
                "responsibility_total_mass_y": float(total_mass_y) if available else None,
                "responsibility_true_support_mass_y": float(true_support_mass_y) if available else None,
                "responsibility_true_support_mass_ratio": ratio,
                "responsibility_true_support_top1": bool(top1),
                "responsibility_best_mass_trajectory_id": best_tid,
                "responsibility_best_mass_gt_raw_id": best_gt,
                "responsibility_hub_hijack": bool(hub_hijack),
                "yprime_demand": demand_y,
            })

    resp_pair_rows = [r for r in pair_rows if bool(r.get("responsibility_available_for_y"))]
    supported_pairs = [r for r in pair_rows if bool(r.get("has_trajectory_support"))]
    unsupported_pairs = [r for r in pair_rows if not bool(r.get("has_trajectory_support"))]
    demand_supported = [r.get("yprime_demand") for r in supported_pairs]
    demand_unsupported = [r.get("yprime_demand") for r in unsupported_pairs]

    null_rows = [r for rows in bucket_rows.values() for r in rows if r.get("null_mass") is not None]
    summary: Record = {
        "status": "PASS" if resp_meta.get("exists") else "MISSING_RESPONSIBILITY_RECORDS",
        "audit_name": "support_null_responsibility_audit",
        "run_root": str(run_root),
        "dataset_name": str(config.dataset_name),
        "stage": str(config.stage),
        "responsibility_records": dict(resp_meta),
        "scope_override_meta": dict(override_meta),
        "scope_meta": dict(scope_meta),
        "row_count": int(len(row_meta)),
        "row_with_responsibility_count": int(sum(1 for r in row_meta if bool(r.get("has_responsibility")))),
        "row_with_null_mass_count": int(len(null_rows)),
        "overall_null_mass_mean": _mean([r.get("null_mass") for r in null_rows]),
        "overall_top1_null_rate": _rate(sum(1 for r in null_rows if bool(r.get("top1_is_null"))), len(null_rows)),
        "bucket_summary": bucket_summary_rows,
        "yprime_pair_count": int(len(pair_rows)),
        "responsibility_available_pair_count": int(len(resp_pair_rows)),
        "responsibility_available_pair_rate": _rate(len(resp_pair_rows), len(pair_rows)),
        "sinkhorn_yprime_true_support_mass_mean": _mean([r.get("responsibility_true_support_mass_ratio") for r in resp_pair_rows]),
        "sinkhorn_yprime_true_support_top1_rate": _rate(sum(1 for r in resp_pair_rows if bool(r.get("responsibility_true_support_top1"))), len(resp_pair_rows)),
        "sinkhorn_yprime_hub_hijack_rate": _rate(sum(1 for r in resp_pair_rows if bool(r.get("responsibility_hub_hijack"))), len(resp_pair_rows)),
        "supported_yprime_demand_mean": _mean(demand_supported),
        "unsupported_yprime_demand_mean": _mean(demand_unsupported),
        "support_demand_gap_supported_minus_unsupported": (None if _mean(demand_supported) is None or _mean(demand_unsupported) is None else float(_mean(demand_supported) - _mean(demand_unsupported))),
        "supported_yprime_low_demand_rate": _rate(sum(1 for r in supported_pairs if (r.get("yprime_demand") is not None and float(r.get("yprime_demand")) <= 0.200001)), len([r for r in supported_pairs if r.get("yprime_demand") is not None])),
        "unsupported_yprime_low_demand_rate": _rate(sum(1 for r in unsupported_pairs if (r.get("yprime_demand") is not None and float(r.get("yprime_demand")) <= 0.200001)), len([r for r in unsupported_pairs if r.get("yprime_demand") is not None])),
        "interpretation": {
            "use": "Check whether NULL absorbs unmatched/no-auditable rows more than Yprime-matched support rows, whether Yprime true-support mass is high, and whether support-aware demand separates supported from unsupported Yprime pairs.",
            "good_pattern": "unmatched_or_no_auditable_gt top1_null_rate high; yprime_matched_gt null_mass low; sinkhorn_yprime_true_support_mass_mean high; supported_yprime_demand_mean > unsupported_yprime_demand_mean.",
            "bad_pattern": "yprime_matched_gt null_mass high or support_demand_gap near zero/negative or hub_hijack high.",
        },
    }

    _dump_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "bucket_null_allocation_summary.csv", bucket_summary_rows, ["bucket", "row_count", "responsibility_row_count", "mean_null_mass", "median_null_mass", "mean_nonnull_mass", "top1_null_rate"])
    _write_csv(output_dir / "clip_yprime_responsibility_summary.csv", pair_rows, ["clip_id", "yprime_raw_id", "has_trajectory_support", "support_count", "responsibility_available_for_y", "responsibility_total_mass_y", "responsibility_true_support_mass_y", "responsibility_true_support_mass_ratio", "responsibility_true_support_top1", "responsibility_best_mass_trajectory_id", "responsibility_best_mass_gt_raw_id", "responsibility_hub_hijack", "yprime_demand"])
    if bool(config.write_rows):
        with (output_dir / "trajectory_null_allocation_rows.jsonl").open("w", encoding="utf-8") as f:
            for bucket in sorted(bucket_rows):
                for row in bucket_rows[bucket]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    takeover = [
        "# Support-Null Responsibility Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- dataset: `{config.dataset_name}`",
        f"- stage: `{config.stage}`",
        f"- responsibility path: `{resp_meta.get('path')}`",
        f"- row_with_null_mass_count: `{summary['row_with_null_mass_count']}`",
        f"- overall_null_mass_mean: `{summary['overall_null_mass_mean']}`",
        f"- overall_top1_null_rate: `{summary['overall_top1_null_rate']}`",
        f"- sinkhorn_yprime_true_support_mass_mean: `{summary['sinkhorn_yprime_true_support_mass_mean']}`",
        f"- sinkhorn_yprime_true_support_top1_rate: `{summary['sinkhorn_yprime_true_support_top1_rate']}`",
        f"- sinkhorn_yprime_hub_hijack_rate: `{summary['sinkhorn_yprime_hub_hijack_rate']}`",
        f"- supported_yprime_demand_mean: `{summary['supported_yprime_demand_mean']}`",
        f"- unsupported_yprime_demand_mean: `{summary['unsupported_yprime_demand_mean']}`",
        f"- support_demand_gap_supported_minus_unsupported: `{summary['support_demand_gap_supported_minus_unsupported']}`",
        "",
        "## Outputs",
        f"- summary: `{output_dir / 'summary.json'}`",
        f"- bucket_null_allocation_summary: `{output_dir / 'bucket_null_allocation_summary.csv'}`",
        f"- clip_yprime_responsibility_summary: `{output_dir / 'clip_yprime_responsibility_summary.csv'}`",
    ]
    (output_dir / "SUPPORT_NULL_RESPONSIBILITY_TAKEOVER.md").write_text("\n".join(takeover) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir), "row_with_null_mass_count": summary["row_with_null_mass_count"], "sinkhorn_yprime_true_support_mass_mean": summary["sinkhorn_yprime_true_support_mass_mean"]}, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Audit support-null assignment responsibility and demand calibration.")
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--runtime_output_root", required=True, type=Path)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--trajectory_source_branch", default="mainline")
    p.add_argument("--stage", default="prealign")
    p.add_argument("--output_dir", default=None, type=Path)
    p.add_argument("--sidecar_root", default=None, type=Path)
    p.add_argument("--hub_raw_ids", default="773", type=_parse_ints)
    p.add_argument("--write_rows", default=True, type=_parse_bool)
    p.add_argument("--top_examples", default=128, type=int)
    p.add_argument("--smoke", default=False, type=_parse_bool)
    p.add_argument("--smoke_max_trajectories", default=1024, type=int)
    p.add_argument("--subset_fraction", default=None, type=float)
    a = p.parse_args(argv)
    return Config(
        run_root=a.run_root,
        runtime_output_root=a.runtime_output_root,
        dataset_name=str(a.dataset_name),
        trajectory_source_branch=str(a.trajectory_source_branch),
        stage=str(a.stage),
        output_dir=a.output_dir,
        sidecar_root=a.sidecar_root,
        hub_raw_ids=tuple(int(x) for x in a.hub_raw_ids),
        write_rows=bool(a.write_rows),
        top_examples=int(a.top_examples),
        smoke=bool(a.smoke),
        smoke_max_trajectories=int(a.smoke_max_trajectories),
        subset_fraction=a.subset_fraction,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
