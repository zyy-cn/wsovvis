#!/usr/bin/env python3
"""Build LV-VIS Llama3-base completion-protocol text banks.

This is an A11A-2 diagnostic builder for base Llama models.  It intentionally
avoids the chat/system/user protocol used by tools/build_lvvis_llama3_text_bank.py
so that a pretrained/base Llama model is evaluated under plain completion prompts.

It writes the same external text-bank contract expected by A10C:
  payload/llama_hidden_mean.fp16.npz with key `protos`
  records/llama_hidden_mean_text_prototype_records.jsonl
  lvvis_class_names.json
  manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

Record = Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_s = str(repo_root)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)


def _import_legacy_builder(repo_root: Path):
    _ensure_repo_on_path(repo_root)
    try:
        import tools.build_lvvis_llama3_text_bank as legacy  # type: ignore
    except Exception as exc:  # pragma: no cover - remote repo dependency
        raise RuntimeError(
            "failed to import tools/build_lvvis_llama3_text_bank.py; "
            "run this script from a deployed wsovvis repo"
        ) from exc
    return legacy


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values in feature matrix")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("zero-norm feature row encountered")
    return (arr / np.maximum(norms, eps)).astype(np.float32, copy=False)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _resolve_assert_root(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("WSOVVIS_ASSERT_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("/home/zyy/code/wsovvis_asserts").resolve()


def _default_ann_paths(repo_root: Path, assert_root: Path) -> Tuple[Path, Path]:
    repo_lvvis = repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations"
    assert_lvvis = assert_root / "dataset" / "LV-VIS" / "annotations"
    if (repo_lvvis / "train_instances.json").is_file() and (repo_lvvis / "val_instances.json").is_file():
        return repo_lvvis / "train_instances.json", repo_lvvis / "val_instances.json"
    return assert_lvvis / "train_instances.json", assert_lvvis / "val_instances.json"


def _load_lvvis_class_map(train_ann: Path, val_ann: Path) -> Tuple[List[Tuple[int, str]], Dict[int, Dict[str, bool]]]:
    class_map: Dict[int, str] = {}
    presence: Dict[int, Dict[str, bool]] = {}
    for split, path in (("train", train_ann), ("val", val_ann)):
        if not path.is_file():
            raise FileNotFoundError(f"LV-VIS annotation json missing: {path}")
        payload = _load_json(path)
        for cat in payload.get("categories", []):
            raw_id = int(cat["id"])
            name = str(cat["name"])
            if raw_id in class_map and class_map[raw_id] != name:
                raise ValueError(f"inconsistent class name for raw_id={raw_id}: {class_map[raw_id]} vs {name}")
            class_map[raw_id] = name
            presence.setdefault(raw_id, {"train": False, "val": False})[split] = True
    ordered = sorted(class_map.items(), key=lambda it: int(it[0]))
    if not ordered:
        raise ValueError("empty LV-VIS class map")
    return ordered, presence


def _select_classes(class_map: Sequence[Tuple[int, str]], max_classes: int) -> List[Tuple[int, str]]:
    if max_classes and max_classes > 0:
        return list(class_map)[: int(max_classes)]
    return list(class_map)


def _write_text_records(path: Path, class_items: Sequence[Tuple[int, str]], payload_rel: str) -> None:
    rows: List[Record] = []
    for slot, (raw_id, class_name) in enumerate(class_items):
        rows.append({
            "raw_id": int(raw_id),
            "class_name": str(class_name),
            "proto_path": f"{payload_rel}#protos[{slot}]",
            "path_base_mode": "artifact_parent_dir",
        })
    _write_jsonl(path, rows)


def _decode_tokens(generator: Any, tokens: Sequence[int]) -> str:
    return generator.tokenizer.decode([int(x) for x in tokens])


def _encode_completion_prompt(generator: Any, prompt: str) -> List[int]:
    return [int(x) for x in generator.tokenizer.encode(prompt, bos=True, eos=False)]


@torch.inference_mode()
def _generated_completion_feature(
    generator: Any,
    *,
    prompt: str,
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    prompt_tokens = _encode_completion_prompt(generator, prompt)
    if not prompt_tokens:
        raise RuntimeError("empty prompt tokens")
    model_max = int(generator.model.params.max_seq_len)
    if len(prompt_tokens) >= model_max:
        raise RuntimeError(f"prompt token length exceeds max_seq_len: {len(prompt_tokens)} >= {model_max}")
    if max_gen_len == 0:
        effective_max_gen_len = max(1, model_max - len(prompt_tokens))
    else:
        effective_max_gen_len = min(int(max_gen_len), max(1, model_max - len(prompt_tokens)))
    gen_tokens, _ = generator.generate(
        prompt_tokens=[prompt_tokens],
        max_gen_len=effective_max_gen_len,
        temperature=float(temperature),
        top_p=float(top_p),
        logprobs=False,
        echo=False,
    )
    idx_tokens = [int(x) for x in gen_tokens[0]]
    if not idx_tokens:
        raise RuntimeError(f"empty generated response for prompt={prompt!r}")
    full_tokens = prompt_tokens + idx_tokens
    if len(full_tokens) > model_max:
        raise RuntimeError(f"full token length exceeds model max_seq_len: {len(full_tokens)}")
    token_tensor = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    _, hidden = generator.model.forward_feat(token_tensor, 0)
    gen_start = len(prompt_tokens)
    gen_end = gen_start + len(idx_tokens)
    gen_hidden = hidden[0, gen_start:gen_end].detach().float().cpu().numpy()
    if int(gen_hidden.shape[0]) != len(idx_tokens):
        raise RuntimeError(f"token-feature alignment failure: tokens={len(idx_tokens)} hidden={gen_hidden.shape[0]}")
    pooled = _l2_normalize_rows(gen_hidden.mean(axis=0, dtype=np.float32).reshape(1, -1))[0]
    meta = {
        "prompt": prompt,
        "generated_text": generator.tokenizer.decode(idx_tokens),
        "prompt_tokens": prompt_tokens,
        "idx_tokens": idx_tokens,
        "token_strings": [_decode_tokens(generator, [x]) for x in idx_tokens],
        "prompt_token_count": int(len(prompt_tokens)),
        "gen_start": int(gen_start),
        "gen_end": int(gen_end),
        "feature_count": int(gen_hidden.shape[0]),
        "pooling_mode": "generated_mean",
        "token_feature_alignment": "exact_completion_generated_token_slice",
    }
    return pooled, meta


@torch.inference_mode()
def _class_span_completion_feature(generator: Any, *, template: str, class_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "{cls}" not in template:
        raise ValueError("class-span template must contain {cls}")
    prefix, suffix = template.split("{cls}", 1)
    prefix_tokens = [int(x) for x in generator.tokenizer.encode(prefix, bos=True, eos=False)]
    class_tokens = [int(x) for x in generator.tokenizer.encode(class_name, bos=False, eos=False)]
    suffix_tokens = [int(x) for x in generator.tokenizer.encode(suffix, bos=False, eos=False)]
    if not class_tokens:
        raise RuntimeError(f"empty class token span for {class_name!r}")
    full_tokens = prefix_tokens + class_tokens + suffix_tokens
    model_max = int(generator.model.params.max_seq_len)
    if len(full_tokens) > model_max:
        raise RuntimeError(f"input too long for class-span mode: {len(full_tokens)} > {model_max}")
    token_tensor = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    _, hidden = generator.model.forward_feat(token_tensor, 0)
    start = len(prefix_tokens)
    end = start + len(class_tokens)
    class_hidden = hidden[0, start:end].detach().float().cpu().numpy()
    pooled = _l2_normalize_rows(class_hidden.mean(axis=0, dtype=np.float32).reshape(1, -1))[0]
    meta = {
        "prompt": template.format(cls=class_name),
        "input_text": generator.tokenizer.decode(full_tokens),
        "idx_tokens": full_tokens,
        "class_idx_tokens": class_tokens,
        "class_token_strings": [_decode_tokens(generator, [x]) for x in class_tokens],
        "class_span_start": int(start),
        "class_span_end": int(end),
        "feature_count": int(class_hidden.shape[0]),
        "pooling_mode": "class_span_mean",
        "token_feature_alignment": "exact_completion_class_name_token_span_slice",
    }
    return pooled, meta


_PROTOCOLS: Dict[str, Dict[str, str]] = {
    "completion_visual_generated": {
        "prompt_template": "Object category: {cls}.\nVisual appearance:",
        "pooling_mode": "generated_mean",
        "description": "plain completion prompt; pool generated continuation tokens",
    },
    "completion_visual_class_span": {
        "prompt_template": "Object category: {cls}.\nVisual appearance:",
        "pooling_mode": "class_span_mean",
        "description": "plain completion prompt; pool class-name token span only",
    },
    "natural_visual_generated": {
        "prompt_template": "A {cls} is visually characterized by",
        "pooling_mode": "generated_mean",
        "description": "natural language completion; pool generated continuation tokens",
    },
    "natural_visual_class_span": {
        "prompt_template": "A {cls} is visually characterized by",
        "pooling_mode": "class_span_mean",
        "description": "natural language completion; pool class-name token span only",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=None, help="wsovvis repo root; default: parent of this tools directory")
    p.add_argument("--assert_root", default=None)
    p.add_argument("--output_root", default=None, help="default: <assert_root>/text_bank_llama3/lvvis")
    p.add_argument("--output_name", required=True)
    p.add_argument("--protocol", choices=sorted(_PROTOCOLS), required=True)
    p.add_argument("--train_annotation_json", default=None)
    p.add_argument("--val_annotation_json", default=None)
    p.add_argument("--max_classes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--max_batch_size", type=int, default=16)
    p.add_argument("--master_port", default="56841")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_gen_len", type=int, default=48, help="0 means use remaining context; generated protocols only")

    p.add_argument("--log_every_classes", type=int, default=20)
    p.add_argument("--print_progress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root()
    assert_root = _resolve_assert_root(repo_root, args.assert_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (assert_root / "text_bank_llama3" / "lvvis").resolve()
    protocol = _PROTOCOLS[str(args.protocol)]
    output_dir = output_root / str(args.output_name)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_default, val_default = _default_ann_paths(repo_root, assert_root)
    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else train_default
    val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else val_default
    full_class_map, presence = _load_lvvis_class_map(train_ann, val_ann)
    class_items = _select_classes(full_class_map, int(args.max_classes))
    class_records = [
        {
            "slot": slot,
            "raw_id": int(raw_id),
            "name": class_name,
            "presence": presence.get(int(raw_id), {"train": False, "val": False}),
        }
        for slot, (raw_id, class_name) in enumerate(class_items)
    ]
    class_path = output_dir / "lvvis_class_names.json"
    _write_json(class_path, {
        "class_count": len(class_records),
        "full_class_count": len(full_class_map),
        "raw_id_order": "ascending",
        "classes": class_records,
        "train_annotation_json": str(train_ann),
        "val_annotation_json": str(val_ann),
        "does_not_use_coco_class_list": True,
    })

    _write_json(output_dir / "prompt_profile.json", {
        "profile_id": f"llama3_base_{args.protocol}",
        "profile_type": "base_completion_protocol",
        "protocol": args.protocol,
        "prompt_template": protocol["prompt_template"],
        "pooling_mode": protocol["pooling_mode"],
        "description": protocol["description"],
        "runtime_overrides": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_gen_len": int(args.max_gen_len),
            "max_classes": int(args.max_classes),
        },
    })

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    legacy = _import_legacy_builder(repo_root)
    generator = legacy._build_llama(args, repo_root)  # type: ignore[attr-defined]

    features: List[np.ndarray] = []
    rows: List[Record] = []
    prompt_template = str(protocol["prompt_template"])
    pooling_mode = str(protocol["pooling_mode"])
    for slot, (raw_id, class_name) in enumerate(class_items):
        if pooling_mode == "generated_mean":
            prompt = prompt_template.format(cls=class_name)
            feat, meta = _generated_completion_feature(
                generator,
                prompt=prompt,
                max_gen_len=int(args.max_gen_len),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
        elif pooling_mode == "class_span_mean":
            feat, meta = _class_span_completion_feature(generator, template=prompt_template, class_name=class_name)
        else:
            raise ValueError(f"unsupported pooling_mode={pooling_mode}")
        features.append(feat)
        meta.update({
            "raw_id": int(raw_id),
            "class_name": str(class_name),
            "class_slot": int(slot),
            "protocol": str(args.protocol),
            "profile_id": f"llama3_base_{args.protocol}",
            "uses_chat_formatter": False,
            "uses_system_user_assistant": False,
            "uses_old_corr_feats": False,
        })
        rows.append(meta)
        if args.print_progress and (slot + 1) % max(1, int(args.log_every_classes)) == 0:
            print(f"[llama3-base-completion] protocol={args.protocol} processed {slot + 1}/{len(class_items)} classes", flush=True)

    mat = _l2_normalize_rows(np.stack(features, axis=0).astype(np.float32))
    mean_path = output_dir / "payload" / "llama_hidden_mean.fp16.npz"
    _save_npz(mean_path, protos=mat.astype(np.float16))
    views_path = output_dir / "payload" / "llama_hidden_views.fp16.npz"
    _save_npz(views_path, views=mat[:, None, :].astype(np.float16))
    response_path = output_dir / "llama3_base_completion_records.jsonl"
    _write_jsonl(response_path, rows)
    records_path = output_dir / "records" / "llama_hidden_mean_text_prototype_records.jsonl"
    _write_text_records(records_path, class_items, "../payload/llama_hidden_mean.fp16.npz")

    manifest = {
        "status": "PASS",
        "tool": "tools/build_lvvis_llama3_base_completion_text_bank.py",
        "profile_id": f"llama3_base_{args.protocol}",
        "profile_type": "base_completion_protocol",
        "protocol": str(args.protocol),
        "prompt_template": prompt_template,
        "pooling_mode": pooling_mode,
        "output_dir": str(output_dir),
        "asset_storage_policy": "canonical_text_bank_assets_under_wsovvis_asserts_not_codex_outputs",
        "repo_root": str(repo_root),
        "assert_root": str(assert_root),
        "class_count": len(class_items),
        "full_class_count": len(full_class_map),
        "raw_id_order": "ascending",
        "does_not_use_coco_class_list": True,
        "does_not_overwrite_clip_text_bank": True,
        "uses_chat_formatter": False,
        "uses_system_user_assistant": False,
        "llama3_token_feature_alignment_fixed": True,
        "token_feature_alignment": "exact_completion_generated_token_or_class_span_slice",
        "uses_old_corr_feats": False,
        "all_vectors_finite": True,
        "all_mean_vectors_l2_normalized": True,
        "train_annotation_json": str(train_ann),
        "val_annotation_json": str(val_ann),
        "runtime": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_gen_len": int(args.max_gen_len),
            "seed": int(args.seed),
        },
        "artifacts": {
            "completion_records_path": str(response_path),
            "llama_hidden_views_path": str(views_path),
            "llama_hidden_mean_path": str(mean_path),
            "llama_hidden_records_path": str(records_path),
            "llama_hidden_shape": list(mat.shape),
            "llama_hidden_views_shape": [int(mat.shape[0]), 1, int(mat.shape[1])],
            "llama_hidden_dim": int(mat.shape[1]),
            "llama_hidden_mean_sha256": _sha256_file(mean_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
