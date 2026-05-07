#!/usr/bin/env python3
"""A8 text-bank graph audit for CLIP / CLIP-of-LLM / Llama3 text anchors.

Read-only audit.  It compares category graphs induced by several text-bank
variants against GT-trajectory DINO carrier prototype graphs.  It does not train
and does not mutate any checkpoint or text bank.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_gtceil(repo_root: Path):
    path = repo_root / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if str(k) not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x)))
    except Exception:
        return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _cos(x: np.ndarray) -> np.ndarray:
    z = _l2(x)
    return z @ z.T


def _valid_cos(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x).all(axis=1)
    sim = np.full((x.shape[0], x.shape[0]), np.nan, dtype=np.float32)
    if valid.any():
        idx = np.where(valid)[0]
        z = _l2(x[valid])
        sim[np.ix_(idx, idx)] = z @ z.T
    return sim, valid


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return ranks
    sx = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sx[end] == sx[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra, rb = _rankdata_average(a[m]), _rankdata_average(b[m])
    if float(np.std(ra)) <= 0 or float(np.std(rb)) <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _upper(sim: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(sim.shape[0])
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(sim[iu], dtype=np.float64)
    if valid is not None:
        vm = np.asarray(valid, dtype=bool)
        vals = vals[vm[iu[0]] & vm[iu[1]]]
    return vals


def _topk(sim: np.ndarray, ids: Sequence[int], i: int, k: int) -> List[int]:
    row = np.asarray(sim[i], dtype=np.float64).copy()
    if i < len(row):
        row[i] = -np.inf
    row[~np.isfinite(row)] = -np.inf
    if not np.isfinite(row).any():
        return []
    order = np.argsort(-row, kind="mergesort")[:k]
    return [int(ids[int(j)]) for j in order if np.isfinite(row[int(j)])]


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _load_npz_first(path: Path) -> Tuple[np.ndarray, str]:
    z = np.load(path)
    keys = list(z.keys())
    if not keys:
        raise RuntimeError(f"empty npz: {path}")
    for key in ("protos", "features", "arr_0", "llama_hidden_mean", "clip_of_llm_mean", "llama_direct_concept_mean"):
        if key in z:
            return np.asarray(z[key]), key
    return np.asarray(z[keys[0]]), str(keys[0])


def _load_lvvis_classes(bank_root: Path) -> Tuple[List[int], Dict[int, str]]:
    path = bank_root / "lvvis_class_names.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("classes", []) if isinstance(payload, dict) else payload
    ids: List[int] = []
    names: Dict[int, str] = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        ids.append(int(rid))
        names[int(rid)] = str(r.get("name", r.get("class_name", rid)))
    if not ids or ids != sorted(ids):
        raise RuntimeError(f"invalid LV-VIS class list: {path}")
    return ids, names


def _payload_for_variant(root: Path, variant: str) -> Path:
    table = {
        "clip_of_llm_mean": root / "payload" / "clip_of_llm_mean.fp16.npz",
        "llama_hidden_mean": root / "payload" / "llama_hidden_mean.fp16.npz",
        "llama_direct_concept_mean": root / "payload" / "llama_direct_concept_mean.fp16.npz",
    }
    if variant not in table:
        raise ValueError(f"unsupported text bank variant: {variant}")
    if not table[variant].is_file():
        raise FileNotFoundError(table[variant])
    return table[variant]


def _load_external_text_bank(root: Path, variant: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    ids, names = _load_lvvis_classes(root)
    p = _payload_for_variant(root, variant)
    arr, key = _load_npz_first(p)
    if arr.ndim != 2 or arr.shape[0] != len(ids):
        raise RuntimeError(f"bad text-bank payload shape={arr.shape} ids={len(ids)}")
    arr = _l2(np.asarray(arr, dtype=np.float32))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    meta = {
        "source": "external_text_bank",
        "root": str(root),
        "variant": variant,
        "payload_path": str(p),
        "payload_array_key": key,
        "feature_dim": int(arr.shape[1]),
        "class_count": int(len(ids)),
        "manifest_status": manifest.get("status"),
        "profile_id": manifest.get("profile_id"),
        "token_feature_alignment": manifest.get("token_feature_alignment"),
        "uses_old_corr_feats": manifest.get("uses_old_corr_feats"),
    }
    return ids, arr, names, meta


def _load_current_clip_text_bank(asset_root: Path, dataset_name: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_text_vocab_with_names  # type: ignore
    raw_ids, _records, mat, class_map = load_text_vocab_with_names(asset_root, dataset_name)
    ids = [int(x) for x in raw_ids]
    names = {int(k): str(v) for k, v in dict(class_map).items()}
    mat = _l2(np.asarray(mat, dtype=np.float32))
    return ids, mat, names, {"source": "current_clip_text_bank", "variant": "clip_current", "feature_dim": int(mat.shape[1]), "class_count": len(ids)}


def _load_visible_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.is_file():
        return ids
    for row in _read_csv(path):
        rid = _as_int(row.get("raw_id"))
        if rid is not None and str(row.get("in_row_gap", "0")).strip() == "1":
            ids.add(int(rid))
    return ids


def _load_base_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    obj = json.loads(path.read_text(encoding="utf-8"))
    vals = obj.get("base_raw_ids", obj.get("base", [])) if isinstance(obj, dict) else []
    return {int(x) for x in vals}


def _rows_and_carriers(gtceil: Any, *, asset_root: Path, dataset_name: str, ann: Path, max_rows: int) -> Tuple[List[Mapping[str, Any]], np.ndarray, Dict[str, Any]]:
    rows0, meta = gtceil._candidate_source_rows(
        gt_carrier_path=asset_root / "carrier_bank_gt" / dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / dataset_name / "trajectory_records.jsonl",
        annotation_json=ann,
        max_rows=int(max_rows or 0),
    )
    carrier, keep, counters = gtceil._load_carrier_matrix(
        gt_carrier_path=asset_root / "carrier_bank_gt" / dataset_name / "carrier_records.jsonl",
        rows=rows0,
    )
    rows = [rows0[i] for i in keep]
    return rows, np.asarray(carrier, dtype=np.float32), {"source_meta": meta, "vector_counters": dict(counters), "row_count": len(rows)}


def _visual_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Dict[int, np.ndarray]:
    buf: Dict[int, List[np.ndarray]] = {}
    for i, row in enumerate(rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or i >= carrier.shape[0]:
            continue
        vec = np.asarray(carrier[i], dtype=np.float32)
        if np.isfinite(vec).all():
            buf.setdefault(int(rid), []).append(vec)
    out: Dict[int, np.ndarray] = {}
    for rid, vecs in buf.items():
        out[rid] = _l2(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))[0]
    return out


def _matrix_for_ids(proto: Mapping[int, np.ndarray], ids: Sequence[int], dim: int = 768) -> np.ndarray:
    out = np.full((len(ids), dim), np.nan, dtype=np.float32)
    for i, rid in enumerate(ids):
        if int(rid) in proto:
            v = np.asarray(proto[int(rid)], dtype=np.float32)
            out[i, : v.shape[0]] = v
    return out


def _graph_metrics(text_sim: np.ndarray, visual_sim: np.ndarray, ids: Sequence[int], valid: np.ndarray, k: int) -> Dict[str, Any]:
    jac = []
    person_in_text_topk = 0
    person = 773
    for i, rid in enumerate(ids):
        tn = _topk(text_sim, ids, i, k)
        vn = _topk(visual_sim, ids, i, k)
        if valid[i]:
            jac.append(_jaccard(tn, vn))
        if person in tn:
            person_in_text_topk += 1
    text_deg = Counter()
    visual_deg = Counter()
    for i in range(len(ids)):
        for nb in _topk(text_sim, ids, i, k):
            text_deg[int(nb)] += 1
        for nb in _topk(visual_sim, ids, i, k):
            visual_deg[int(nb)] += 1
    vm = np.asarray(valid, dtype=bool)
    return {
        "valid_class_count": int(vm.sum()),
        "spearman_pairwise": _spearman(_upper(text_sim, vm), _upper(visual_sim, vm)),
        "mean_topk_jaccard": _mean(jac),
        "text_person_in_topk_rate": float(person_in_text_topk / max(len(ids), 1)),
        "text_person_indegree@k": int(text_deg.get(person, 0)),
        "visual_person_indegree@k": int(visual_deg.get(person, 0)),
        "text_max_indegree@k": int(max(text_deg.values()) if text_deg else 0),
        "visual_max_indegree@k": int(max(visual_deg.values()) if visual_deg else 0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare text-bank category graphs against GT visual prototype graphs")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default="")
    p.add_argument("--output_root", required=True)
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--val_annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--visible_csv", default="")
    p.add_argument("--class_scope", choices=["visible525", "base", "all"], default="visible525")
    p.add_argument("--neighbor_k", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--visual_only_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    p.add_argument("--direct_concept_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1")
    p.add_argument("--variants", default="clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo(repo_root)
    gtceil = _load_gtceil(repo_root)

    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
    val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
    split_json = Path(args.split_json).expanduser().resolve() if args.split_json else repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json"
    visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else (Path(args.run_root).expanduser().resolve() / "analysis" / "a8_base_116_visibility_audit" / "lvvis_train_base" / "base_641_visibility_by_class.csv" if str(args.run_root).strip() else Path(""))

    current_ids, current_text, current_names, current_meta = _load_current_clip_text_bank(asset_root, args.train_dataset_name)
    id_set = set(current_ids)
    if args.class_scope == "visible525":
        visible = _load_visible_ids(visible_csv)
        if not visible:
            raise RuntimeError(f"class_scope=visible525 requires valid --visible_csv; got {visible_csv}")
        id_set &= visible
    elif args.class_scope == "base":
        base_ids = _load_base_ids(split_json)
        if not base_ids:
            raise RuntimeError(f"class_scope=base requires valid split json; got {split_json}")
        id_set &= base_ids
    ids = sorted(id_set)
    if not ids:
        raise RuntimeError("empty class id scope")

    train_rows, train_carrier, train_meta = _rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
    val_rows, val_carrier, val_meta = _rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.val_dataset_name, ann=val_ann, max_rows=int(args.max_rows))
    train_proto = _visual_prototypes(train_rows, train_carrier)
    val_proto = _visual_prototypes(val_rows, val_carrier)
    train_mat = _matrix_for_ids(train_proto, ids)
    val_mat = _matrix_for_ids(val_proto, ids)
    train_sim, train_valid = _valid_cos(train_mat)
    val_sim, val_valid = _valid_cos(val_mat)

    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    text_banks: Dict[str, Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]] = {}
    for v in variants:
        if v == "clip_current":
            text_banks[v] = (current_ids, current_text, current_names, current_meta)
        elif v in {"clip_of_llm_mean", "llama_hidden_mean"}:
            text_banks[v] = _load_external_text_bank(Path(args.visual_only_root).expanduser().resolve(), v)
        elif v == "llama_direct_concept_mean":
            text_banks[v] = _load_external_text_bank(Path(args.direct_concept_root).expanduser().resolve(), v)
        else:
            raise ValueError(f"unsupported variant: {v}")

    summary_rows: List[Dict[str, Any]] = []
    hub_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    names = dict(current_names)
    for variant, (t_ids, t_mat, t_names, meta) in text_banks.items():
        names.update(t_names)
        t_idx = {int(r): i for i, r in enumerate(t_ids)}
        missing = [rid for rid in ids if rid not in t_idx]
        if missing:
            raise RuntimeError(f"variant {variant} missing {len(missing)} scoped raw ids; first={missing[:10]}")
        sub = _l2(np.asarray(t_mat[[t_idx[rid] for rid in ids]], dtype=np.float32))
        text_sim = _cos(sub)
        row_train = {"variant": variant, "comparison": "text_vs_vision_train", **_graph_metrics(text_sim, train_sim, ids, train_valid, int(args.neighbor_k)), **{f"meta_{k}": v for k, v in meta.items() if k in {"source", "profile_id", "feature_dim", "token_feature_alignment", "uses_old_corr_feats"}}}
        row_val = {"variant": variant, "comparison": "text_vs_vision_val", **_graph_metrics(text_sim, val_sim, ids, val_valid, int(args.neighbor_k)), **{f"meta_{k}": v for k, v in meta.items() if k in {"source", "profile_id", "feature_dim", "token_feature_alignment", "uses_old_corr_feats"}}}
        summary_rows.extend([row_train, row_val])
        deg = Counter()
        for i in range(len(ids)):
            for nb in _topk(text_sim, ids, i, int(args.neighbor_k)):
                deg[nb] += 1
        for rank, (rid, indeg) in enumerate(deg.most_common(30), start=1):
            hub_rows.append({"variant": variant, "hub_rank": rank, "raw_id": rid, "class_name": names.get(rid, str(rid)), "indegree@k": indeg, "is_person": int(rid) == 773})
        for i, rid in enumerate(ids):
            tn = _topk(text_sim, ids, i, int(args.neighbor_k))
            class_rows.append({
                "variant": variant,
                "raw_id": rid,
                "class_name": names.get(rid, str(rid)),
                "has_train_visual_proto": bool(train_valid[i]),
                "has_val_visual_proto": bool(val_valid[i]),
                "text_vtrain_jaccard@k": _jaccard(tn, _topk(train_sim, ids, i, int(args.neighbor_k))) if train_valid[i] else float("nan"),
                "text_vval_jaccard@k": _jaccard(tn, _topk(val_sim, ids, i, int(args.neighbor_k))) if val_valid[i] else float("nan"),
                "person_in_text_topk": 773 in tn,
                "text_neighbors@k": ";".join(str(x) for x in tn),
            })

    _write_csv(out_root / "textbank_graph_global_summary.csv", summary_rows)
    _write_csv(out_root / "textbank_graph_hub_summary.csv", hub_rows)
    _write_csv(out_root / "textbank_graph_class_metrics.csv", class_rows)
    payload = {
        "status": "PASS",
        "output_root": str(out_root),
        "asset_root": str(asset_root),
        "class_scope": str(args.class_scope),
        "class_count": len(ids),
        "neighbor_k": int(args.neighbor_k),
        "variants": variants,
        "visual_source_meta": {"train": train_meta, "val": val_meta},
        "summary": summary_rows,
        "artifacts": {
            "global_summary": str(out_root / "textbank_graph_global_summary.csv"),
            "hub_summary": str(out_root / "textbank_graph_hub_summary.csv"),
            "class_metrics": str(out_root / "textbank_graph_class_metrics.csv"),
        },
    }
    _write_json(out_root / "textbank_graph_audit_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
