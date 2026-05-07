#!/usr/bin/env python3
"""Build or rebuild CLIP-of-LLM description features from an existing Llama3 text bank.

This is useful when llama3_responses.jsonl already exists and you want to encode the
same generated descriptions with OpenAI CLIP without rerunning Llama3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _l2_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite feature values")
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm feature row")
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32)


def _load_clip_encoder(repo_root: Path, device: str):
    repo_s = str(repo_root)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    from videocutler.ext_stageb_ovvis.models.text_encoder_clip import (  # type: ignore
        ClipTextEncoderConfig,
        OpenAIClipTextEncoder,
    )

    return OpenAIClipTextEncoder(ClipTextEncoderConfig(clip_ckpt="openai_clip_vit_b16", device=device))


def _write_records(path: Path, classes: List[Dict[str, Any]], payload_rel: str) -> None:
    rows = []
    for slot, item in enumerate(classes):
        rows.append(
            {
                "raw_id": int(item["raw_id"]),
                "class_name": str(item.get("name", item.get("class_name", ""))),
                "proto_path": f"{payload_rel}#protos[{slot}]",
                "path_base_mode": "artifact_parent_dir",
            }
        )
    _write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild CLIP-of-LLM features from llama3_responses.jsonl")
    p.add_argument("--repo_root", default=None)
    p.add_argument("--bank_root", required=True, help="Existing profile directory containing llama3_responses.jsonl and lvvis_class_names.json")
    p.add_argument("--clip_device", default="cuda:0")
    p.add_argument("--clip_batch_size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root()
    bank_root = Path(args.bank_root).expanduser().resolve()
    responses_path = bank_root / "llama3_responses.jsonl"
    class_path = bank_root / "lvvis_class_names.json"
    if not responses_path.is_file():
        raise FileNotFoundError(responses_path)
    if not class_path.is_file():
        raise FileNotFoundError(class_path)
    classes_payload = _load_json(class_path)
    classes = list(classes_payload["classes"])
    responses = _read_jsonl(responses_path)
    grouped: Dict[int, List[str]] = {int(item["raw_id"]): [] for item in classes}
    for rec in responses:
        raw_id = int(rec["raw_id"])
        if raw_id in grouped:
            grouped[raw_id].append(str(rec["generated_text"]))
    texts: List[str] = []
    view_counts: List[int] = []
    for item in classes:
        raw_id = int(item["raw_id"])
        cur = grouped[raw_id]
        if not cur:
            raise RuntimeError(f"no generated_text rows for raw_id={raw_id}")
        view_counts.append(len(cur))
        texts.extend(cur)

    encoder = _load_clip_encoder(repo_root, args.clip_device)
    flat = encoder.encode_texts(texts, batch_size=int(args.clip_batch_size)).astype(np.float32)
    flat = _l2_rows(flat)
    views: List[np.ndarray] = []
    cursor = 0
    for count in view_counts:
        views.append(flat[cursor: cursor + count])
        cursor += count
    view_arr = np.stack(views, axis=0).astype(np.float32)
    mean = _l2_rows(view_arr.mean(axis=1))

    payload_dir = bank_root / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    views_path = payload_dir / "clip_of_llm_views.fp16.npz"
    mean_path = payload_dir / "clip_of_llm_mean.fp16.npz"
    if (views_path.exists() or mean_path.exists()) and not args.overwrite:
        raise FileExistsError("CLIP-of-LLM payload exists; pass --overwrite")
    np.savez(views_path, views=view_arr.astype(np.float16))
    np.savez(mean_path, protos=mean.astype(np.float16))
    records_path = bank_root / "records" / "clip_of_llm_mean_text_prototype_records.jsonl"
    _write_records(records_path, classes, "../payload/clip_of_llm_mean.fp16.npz")

    manifest_path = bank_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {"status": "PASS"}
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update(
        {
            "clip_of_llm_views_path": str(views_path),
            "clip_of_llm_mean_path": str(mean_path),
            "clip_of_llm_records_path": str(records_path),
            "clip_of_llm_shape": list(mean.shape),
            "clip_of_llm_views_shape": list(view_arr.shape),
            "clip_of_llm_dim": int(mean.shape[1]),
            "clip_of_llm_mean_sha256": _sha256(mean_path),
        }
    )
    manifest.update(
        {
            "status": "PASS",
            "clip_of_llm_rebuilt_from_existing_responses": True,
            "all_mean_vectors_l2_normalized": True,
            "artifacts": artifacts,
        }
    )
    _write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "bank_root": str(bank_root), "clip_of_llm_mean_path": str(mean_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
