#!/usr/bin/env python3
"""Validate LV-VIS Llama3 / CLIP-of-LLM text-bank assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_norms(name: str, arr: np.ndarray, atol: float = 5e-3) -> Dict[str, Any]:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name}: expected [C,D], got {tuple(x.shape)}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: non-finite values")
    norms = np.linalg.norm(x, axis=1)
    max_abs_err = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    if max_abs_err > atol:
        raise ValueError(f"{name}: L2 norm check failed, max_abs_err={max_abs_err}")
    return {"shape": list(x.shape), "max_l2_norm_abs_err": max_abs_err}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate LV-VIS Llama3 / CLIP-of-LLM text bank")
    p.add_argument("--bank_root", required=True)
    p.add_argument("--expect_class_count", type=int, default=1196)
    p.add_argument("--allow_smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank_root = Path(args.bank_root).expanduser().resolve()
    manifest_path = bank_root / "manifest.json"
    class_path = bank_root / "lvvis_class_names.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not class_path.is_file():
        raise FileNotFoundError(class_path)
    manifest = _load_json(manifest_path)
    classes = _load_json(class_path)["classes"]
    raw_ids = [int(item["raw_id"]) for item in classes]
    if raw_ids != sorted(raw_ids):
        raise ValueError("lvvis_class_names raw_id order is not ascending")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate raw_id in lvvis_class_names")
    if not args.allow_smoke and len(raw_ids) != int(args.expect_class_count):
        raise ValueError(f"class_count mismatch: {len(raw_ids)} != {args.expect_class_count}")
    if manifest.get("uses_old_corr_feats") is not False:
        raise ValueError("manifest does not explicitly set uses_old_corr_feats=false")
    if manifest.get("does_not_use_coco_class_list") is not True:
        raise ValueError("manifest does not confirm LV-VIS class list source")

    artifacts = manifest.get("artifacts", {})
    checked: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if not key.endswith("_mean_path"):
            continue
        p = Path(str(value))
        if not p.is_file():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as payload:
            if "protos" not in payload:
                raise ValueError(f"{p} missing 'protos'")
            protos = np.asarray(payload["protos"], dtype=np.float32)
        if int(protos.shape[0]) != len(raw_ids):
            raise ValueError(f"{key}: class dimension mismatch {protos.shape[0]} vs {len(raw_ids)}")
        checked[key] = _check_norms(key, protos)

    if not checked:
        raise ValueError("no *_mean_path payloads found in manifest artifacts")

    result = {
        "status": "PASS",
        "bank_root": str(bank_root),
        "profile_id": manifest.get("profile_id"),
        "class_count": len(raw_ids),
        "checked_payloads": checked,
        "uses_old_corr_feats": manifest.get("uses_old_corr_feats"),
        "token_feature_alignment": manifest.get("token_feature_alignment"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
