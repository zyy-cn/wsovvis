#!/usr/bin/env python3
"""Build LV-VIS Llama3 / CLIP-of-LLM text-bank assets.

This tool migrates the UVLT Llama3 text-feature extraction idea into wsovvis,
while fixing the original token-feature alignment bug:

  old UVLT corr_feats: h(last_prompt_token), h(y1), h(y2), ...
  this tool:           h(y1), h(y2), h(y3), ...

The fix is implemented by two phases:
  1. generate assistant token ids with Llama3;
  2. full-forward prompt_tokens + generated_tokens and slice the hidden states
     exactly at the generated token span.

It writes reusable assets under wsovvis_asserts/text_bank_llama3/lvvis by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


Record = Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _l2_normalize_rows(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values in feature matrix")
    if np.any(norms <= eps):
        raise ValueError("zero-norm feature row encountered")
    return (arr / np.maximum(norms, eps)).astype(np.float32, copy=False)


def _mean_then_l2(views: np.ndarray) -> np.ndarray:
    """views: [C, V, D] -> [C, D] after view mean and L2 normalization."""
    if views.ndim != 3:
        raise ValueError(f"expected views [C,V,D], got {tuple(views.shape)}")
    mean = views.astype(np.float32).mean(axis=1)
    return _l2_normalize_rows(mean)


def _resolve_assert_root(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("WSOVVIS_ASSERT_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    # Remote default used in this project.
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
    for split, path in [("train", train_ann), ("val", val_ann)]:
        if not path.is_file():
            raise FileNotFoundError(f"LV-VIS annotation json missing: {path}")
        payload = _load_json(path)
        for category in payload.get("categories", []):
            raw_id = int(category["id"])
            name = str(category["name"])
            if raw_id in class_map and class_map[raw_id] != name:
                raise ValueError(f"inconsistent class name for raw_id={raw_id}: {class_map[raw_id]} vs {name}")
            class_map[raw_id] = name
            presence.setdefault(raw_id, {"train": False, "val": False})[split] = True
    ordered = sorted(class_map.items(), key=lambda item: int(item[0]))
    if not ordered:
        raise ValueError("empty LV-VIS class map")
    return ordered, presence


def _select_classes(class_map: Sequence[Tuple[int, str]], max_classes: int) -> List[Tuple[int, str]]:
    if max_classes and max_classes > 0:
        return list(class_map)[: int(max_classes)]
    return list(class_map)


def _load_profiles(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "third_party" / "uvlt_llama3" / "prompts" / "prompt_profiles.json"
    payload = _load_json(path)
    return payload


def _resolve_profile(repo_root: Path, profile_id: str) -> Dict[str, Any]:
    profiles = _load_profiles(repo_root)
    if profile_id not in profiles.get("profiles", {}):
        raise KeyError(f"unknown profile_id={profile_id}; available={sorted(profiles.get('profiles', {}).keys())}")
    return dict(profiles["profiles"][profile_id])


def _format_class_in_prompt(prompt: str, class_name: str, style: str) -> str:
    if style == "bracketed_lower":
        return prompt.replace("[cls]", f"[{class_name}]").replace("[CLS]", f"[{class_name}]")
    if style in {"plain_upper", "plain"}:
        return prompt.replace("[CLS]", class_name).replace("[cls]", class_name)
    if style == "plain_format":
        return prompt.format(cls=class_name)
    raise ValueError(f"unsupported class_placeholder_style={style}")


def _system_for_prompt(profile: Dict[str, Any], prompt_template: str) -> str:
    if "system_prompt_template" in profile:
        template = str(profile["system_prompt_template"])
        return template.replace("[prompt]", prompt_template).replace("{prompt}", prompt_template)
    return str(profile.get("system_prompt", ""))


def _copy_prompt_profile(output_dir: Path, profile: Dict[str, Any], overrides: Dict[str, Any]) -> Path:
    payload = dict(profile)
    payload["runtime_overrides"] = overrides
    path = output_dir / "prompt_profile.json"
    _write_json(path, payload)
    return path


def _import_llama(repo_root: Path):
    llama_root = repo_root / "third_party" / "uvlt_llama3"
    if not llama_root.is_dir():
        raise FileNotFoundError(f"missing third_party UVLT Llama3 package: {llama_root}")
    llama_root_s = str(llama_root)
    if llama_root_s not in sys.path:
        sys.path.insert(0, llama_root_s)
    from llama import Llama  # type: ignore

    return Llama


def _build_llama(args: argparse.Namespace, repo_root: Path):
    Llama = _import_llama(repo_root)
    return Llama.build(
        ckpt_dir=str(Path(args.ckpt_dir).expanduser()),
        tokenizer_path=str(Path(args.tokenizer_path).expanduser()),
        max_seq_len=int(args.max_seq_len),
        max_batch_size=int(args.max_batch_size),
        seed=int(args.seed),
        local_rank=int(args.local_rank),
        MASTER_PORT=str(args.master_port),
    )


@dataclass(frozen=True)
class GeneratedFeature:
    generated_text: str
    idx_tokens: List[int]
    token_strings: List[str]
    pooled_feature: np.ndarray
    prompt_token_count: int
    feature_count: int
    gen_start: int
    gen_end: int


@torch.inference_mode()
def _generate_response_feature(
    generator: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> GeneratedFeature:
    dialog = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_tokens = generator.formatter.encode_dialog_prompt(dialog)
    if max_gen_len == 0:
        effective_max_gen_len = max(1, int(generator.model.params.max_seq_len) - len(prompt_tokens))
    else:
        effective_max_gen_len = int(max_gen_len)
    if len(prompt_tokens) + effective_max_gen_len > int(generator.model.params.max_seq_len):
        effective_max_gen_len = max(1, int(generator.model.params.max_seq_len) - len(prompt_tokens))
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
        raise RuntimeError(f"Llama3 generated empty response for user_prompt={user_prompt!r}")

    full_tokens = prompt_tokens + idx_tokens
    if len(full_tokens) > int(generator.model.params.max_seq_len):
        raise RuntimeError(f"full token length exceeds model max_seq_len: {len(full_tokens)}")
    token_tensor = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    _, hidden = generator.model.forward_feat(token_tensor, 0)
    gen_start = len(prompt_tokens)
    gen_end = gen_start + len(idx_tokens)
    gen_hidden = hidden[0, gen_start:gen_end].detach().float().cpu().numpy()
    if int(gen_hidden.shape[0]) != len(idx_tokens):
        raise RuntimeError(f"token-feature alignment failure: tokens={len(idx_tokens)} hidden={gen_hidden.shape[0]}")
    if not np.all(np.isfinite(gen_hidden)):
        raise RuntimeError("non-finite generated token hidden states")
    pooled = gen_hidden.mean(axis=0).astype(np.float32, copy=False)
    pooled = _l2_normalize_rows(pooled.reshape(1, -1))[0]
    token_strings = [generator.tokenizer.decode([int(x)]) for x in idx_tokens]
    decoded = generator.tokenizer.decode(idx_tokens)
    return GeneratedFeature(
        generated_text=decoded,
        idx_tokens=idx_tokens,
        token_strings=token_strings,
        pooled_feature=pooled,
        prompt_token_count=len(prompt_tokens),
        feature_count=int(gen_hidden.shape[0]),
        gen_start=gen_start,
        gen_end=gen_end,
    )


@torch.inference_mode()
def _direct_concept_feature(generator: Any, class_name: str, template: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "{cls}" not in template:
        raise ValueError("direct concept template must contain {cls}")
    prefix, suffix = template.split("{cls}", 1)
    prefix_tokens = generator.tokenizer.encode(prefix, bos=True, eos=False)
    class_tokens = generator.tokenizer.encode(class_name, bos=False, eos=False)
    suffix_tokens = generator.tokenizer.encode(suffix, bos=False, eos=False)
    if not class_tokens:
        raise RuntimeError(f"empty class token span for {class_name!r}")
    full_tokens = prefix_tokens + class_tokens + suffix_tokens
    if len(full_tokens) > int(generator.model.params.max_seq_len):
        raise RuntimeError(f"direct concept input too long for {class_name!r}")
    token_tensor = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    _, hidden = generator.model.forward_feat(token_tensor, 0)
    start = len(prefix_tokens)
    end = start + len(class_tokens)
    class_hidden = hidden[0, start:end].detach().float().cpu().numpy()
    pooled = class_hidden.mean(axis=0).astype(np.float32, copy=False)
    pooled = _l2_normalize_rows(pooled.reshape(1, -1))[0]
    meta = {
        "input_text": generator.tokenizer.decode(full_tokens),
        "idx_tokens": [int(x) for x in full_tokens],
        "class_idx_tokens": [int(x) for x in class_tokens],
        "class_token_strings": [generator.tokenizer.decode([int(x)]) for x in class_tokens],
        "class_span_start": int(start),
        "class_span_end": int(end),
        "feature_count": int(class_hidden.shape[0]),
    }
    return pooled, meta


def _load_clip_encoder(repo_root: Path, device: str):
    repo_s = str(repo_root)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    from videocutler.ext_stageb_ovvis.models.text_encoder_clip import (  # type: ignore
        ClipTextEncoderConfig,
        OpenAIClipTextEncoder,
    )

    return OpenAIClipTextEncoder(ClipTextEncoderConfig(clip_ckpt="openai_clip_vit_b16", device=device))


def _write_text_records(path: Path, class_items: Sequence[Tuple[int, str]], payload_rel: str) -> None:
    records: List[Record] = []
    for slot, (raw_id, class_name) in enumerate(class_items):
        records.append(
            {
                "raw_id": int(raw_id),
                "class_name": str(class_name),
                "proto_path": f"{payload_rel}#protos[{slot}]",
                "path_base_mode": "artifact_parent_dir",
            }
        )
    _write_jsonl(path, records)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _build_description_bank(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
    profile: Dict[str, Any],
    class_items: Sequence[Tuple[int, str]],
    class_presence: Dict[int, Dict[str, bool]],
) -> Dict[str, Any]:
    defaults = dict(profile.get("generation_defaults", {}))
    temperature = float(args.temperature if args.temperature is not None else defaults.get("temperature", 0.0))
    top_p = float(args.top_p if args.top_p is not None else defaults.get("top_p", 1.0))
    max_gen_len = int(args.max_gen_len if args.max_gen_len is not None else defaults.get("max_gen_len", 48))
    repeat = int(args.repeat if args.repeat is not None else defaults.get("repeat", 1))
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    generator = None
    if args.build_llama_hidden:
        generator = _build_llama(args, repo_root)

    responses: List[Record] = []
    llama_views: List[List[np.ndarray]] = [[] for _ in class_items]
    text_views: List[List[str]] = [[] for _ in class_items]
    style = str(profile.get("class_placeholder_style", "plain_upper"))
    prompts = list(profile.get("user_prompts", []))
    if not prompts:
        raise ValueError("description profile has no user_prompts")

    # Reuse existing responses if requested. This supports CLIP-of-LLM generation without rerunning Llama3.
    response_path = output_dir / "llama3_responses.jsonl"
    if args.reuse_responses and response_path.is_file():
        responses = _read_jsonl(response_path)
        by_key: Dict[Tuple[int, str, int], Record] = {}
        for rec in responses:
            by_key[(int(rec["raw_id"]), str(rec["prompt_id"]), int(rec["repeat_index"]))] = rec
        for class_slot, (raw_id, _class_name) in enumerate(class_items):
            for prompt in prompts:
                prompt_id = str(prompt.get("prompt_id"))
                for repeat_index in range(repeat):
                    rec = by_key[(int(raw_id), prompt_id, repeat_index)]
                    text_views[class_slot].append(str(rec["generated_text"]))
        if args.build_llama_hidden:
            raise RuntimeError("--reuse_responses can rebuild CLIP-of-LLM, but Llama hidden extraction still requires rerunning full-forward; rerun without --reuse_responses for llama_hidden")
    else:
        if generator is None:
            raise RuntimeError("description generation requires --build_llama_hidden unless --reuse_responses is used")
        for class_slot, (raw_id, class_name) in enumerate(class_items):
            for prompt_obj in prompts:
                prompt_id = str(prompt_obj.get("prompt_id"))
                prompt_template = str(prompt_obj.get("prompt"))
                system_prompt = _system_for_prompt(profile, prompt_template)
                user_prompt = _format_class_in_prompt(prompt_template, class_name, style)
                for repeat_index in range(repeat):
                    feat = _generate_response_feature(
                        generator,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    view_index = len(llama_views[class_slot])
                    llama_views[class_slot].append(feat.pooled_feature)
                    text_views[class_slot].append(feat.generated_text)
                    responses.append(
                        {
                            "raw_id": int(raw_id),
                            "class_name": class_name,
                            "class_slot": int(class_slot),
                            "view_index": int(view_index),
                            "prompt_profile_id": str(profile["profile_id"]),
                            "prompt_id": prompt_id,
                            "prompt_source": prompt_obj.get("source"),
                            "repeat_index": int(repeat_index),
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "generated_text": feat.generated_text,
                            "idx_tokens": feat.idx_tokens,
                            "token_strings": feat.token_strings,
                            "prompt_token_count": int(feat.prompt_token_count),
                            "gen_start": int(feat.gen_start),
                            "gen_end": int(feat.gen_end),
                            "feature_count": int(feat.feature_count),
                            "token_feature_alignment": "exact_full_forward_generated_token_slice",
                            "uses_old_corr_feats": False,
                        }
                    )
            if args.print_progress and (class_slot + 1) % max(1, args.log_every_classes) == 0:
                print(f"[llama3-text-bank] processed {class_slot + 1}/{len(class_items)} classes", flush=True)
        _write_jsonl(response_path, responses)

    artifacts: Dict[str, Any] = {
        "responses_path": str(response_path),
        "response_count": len(responses),
    }

    if args.build_llama_hidden:
        views = np.stack([np.stack(v, axis=0) for v in llama_views], axis=0).astype(np.float32)
        if views.ndim != 3:
            raise RuntimeError(f"invalid llama views shape: {tuple(views.shape)}")
        mean = _mean_then_l2(views)
        views_path = output_dir / "payload" / "llama_hidden_views.fp16.npz"
        mean_path = output_dir / "payload" / "llama_hidden_mean.fp16.npz"
        _save_npz(views_path, views=views.astype(np.float16))
        _save_npz(mean_path, protos=mean.astype(np.float16))
        records_path = output_dir / "records" / "llama_hidden_mean_text_prototype_records.jsonl"
        _write_text_records(records_path, class_items, "../payload/llama_hidden_mean.fp16.npz")
        artifacts.update(
            {
                "llama_hidden_views_path": str(views_path),
                "llama_hidden_mean_path": str(mean_path),
                "llama_hidden_records_path": str(records_path),
                "llama_hidden_shape": list(mean.shape),
                "llama_hidden_views_shape": list(views.shape),
                "llama_hidden_dim": int(mean.shape[1]),
                "llama_hidden_mean_sha256": _sha256_file(mean_path),
            }
        )

    if args.build_clip_of_llm:
        encoder = _load_clip_encoder(repo_root, args.clip_device)
        flat_texts: List[str] = []
        view_counts: List[int] = []
        for texts in text_views:
            view_counts.append(len(texts))
            flat_texts.extend(texts)
        if not flat_texts:
            raise RuntimeError("no generated texts available for CLIP-of-LLM")
        flat_features = encoder.encode_texts(flat_texts, batch_size=int(args.clip_batch_size)).astype(np.float32)
        flat_features = _l2_normalize_rows(flat_features)
        cursor = 0
        clip_views: List[np.ndarray] = []
        for count in view_counts:
            chunk = flat_features[cursor: cursor + count]
            cursor += count
            clip_views.append(chunk)
        clip_views_arr = np.stack(clip_views, axis=0).astype(np.float32)
        clip_mean = _mean_then_l2(clip_views_arr)
        clip_views_path = output_dir / "payload" / "clip_of_llm_views.fp16.npz"
        clip_mean_path = output_dir / "payload" / "clip_of_llm_mean.fp16.npz"
        _save_npz(clip_views_path, views=clip_views_arr.astype(np.float16))
        _save_npz(clip_mean_path, protos=clip_mean.astype(np.float16))
        records_path = output_dir / "records" / "clip_of_llm_mean_text_prototype_records.jsonl"
        _write_text_records(records_path, class_items, "../payload/clip_of_llm_mean.fp16.npz")
        artifacts.update(
            {
                "clip_of_llm_views_path": str(clip_views_path),
                "clip_of_llm_mean_path": str(clip_mean_path),
                "clip_of_llm_records_path": str(records_path),
                "clip_of_llm_shape": list(clip_mean.shape),
                "clip_of_llm_views_shape": list(clip_views_arr.shape),
                "clip_of_llm_dim": int(clip_mean.shape[1]),
                "clip_of_llm_mean_sha256": _sha256_file(clip_mean_path),
            }
        )

    return {
        "generation_runtime": {
            "temperature": temperature,
            "top_p": top_p,
            "max_gen_len": max_gen_len,
            "repeat": repeat,
        },
        "artifacts": artifacts,
    }


def _build_direct_concept_bank(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
    profile: Dict[str, Any],
    class_items: Sequence[Tuple[int, str]],
) -> Dict[str, Any]:
    generator = _build_llama(args, repo_root)
    template = str(profile.get("concept_template", "visual object category: {cls}."))
    features: List[np.ndarray] = []
    records_meta: List[Record] = []
    for slot, (raw_id, class_name) in enumerate(class_items):
        feat, meta = _direct_concept_feature(generator, class_name, template)
        features.append(feat)
        meta.update(
            {
                "raw_id": int(raw_id),
                "class_name": class_name,
                "class_slot": int(slot),
                "prompt_profile_id": str(profile["profile_id"]),
                "token_feature_alignment": "exact_class_name_token_span_full_forward_slice",
                "uses_generation": False,
                "uses_old_corr_feats": False,
            }
        )
        records_meta.append(meta)
        if args.print_progress and (slot + 1) % max(1, args.log_every_classes) == 0:
            print(f"[llama3-direct-concept] processed {slot + 1}/{len(class_items)} classes", flush=True)
    mat = _l2_normalize_rows(np.stack(features, axis=0).astype(np.float32))
    mean_path = output_dir / "payload" / "llama_direct_concept_mean.fp16.npz"
    _save_npz(mean_path, protos=mat.astype(np.float16))
    meta_path = output_dir / "llama3_direct_concept_records.jsonl"
    _write_jsonl(meta_path, records_meta)
    records_path = output_dir / "records" / "llama_direct_concept_mean_text_prototype_records.jsonl"
    _write_text_records(records_path, class_items, "../payload/llama_direct_concept_mean.fp16.npz")
    return {
        "generation_runtime": {"uses_generation": False, "concept_template": template},
        "artifacts": {
            "direct_concept_records_path": str(meta_path),
            "llama_direct_concept_mean_path": str(mean_path),
            "llama_direct_concept_records_path": str(records_path),
            "llama_direct_concept_shape": list(mat.shape),
            "llama_direct_concept_dim": int(mat.shape[1]),
            "llama_direct_concept_mean_sha256": _sha256_file(mean_path),
        },
    }


def _build_clip_single_template_bank(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
    profile: Dict[str, Any],
    class_items: Sequence[Tuple[int, str]],
) -> Dict[str, Any]:
    template = str(profile.get("clip_template", "a photo of a {cls}."))
    texts = [template.format(cls=class_name) for _, class_name in class_items]
    encoder = _load_clip_encoder(repo_root, args.clip_device)
    features = encoder.encode_texts(texts, batch_size=int(args.clip_batch_size)).astype(np.float32)
    features = _l2_normalize_rows(features)
    mean_path = output_dir / "payload" / "clip_single_template_mean.fp16.npz"
    _save_npz(mean_path, protos=features.astype(np.float16))
    records_path = output_dir / "records" / "clip_single_template_mean_text_prototype_records.jsonl"
    _write_text_records(records_path, class_items, "../payload/clip_single_template_mean.fp16.npz")
    text_records = [
        {"raw_id": int(raw_id), "class_name": class_name, "text": text, "template": template}
        for (raw_id, class_name), text in zip(class_items, texts)
    ]
    text_path = output_dir / "clip_single_template_texts.jsonl"
    _write_jsonl(text_path, text_records)
    return {
        "generation_runtime": {"uses_generation": False, "clip_template": template},
        "artifacts": {
            "clip_single_template_texts_path": str(text_path),
            "clip_single_template_mean_path": str(mean_path),
            "clip_single_template_records_path": str(records_path),
            "clip_single_template_shape": list(features.shape),
            "clip_single_template_dim": int(features.shape[1]),
            "clip_single_template_mean_sha256": _sha256_file(mean_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LV-VIS Llama3 / CLIP-of-LLM text bank assets.")
    parser.add_argument("--repo_root", default=None, help="wsovvis repo root; default: parent of this tools directory")
    parser.add_argument("--assert_root", default=None, help="wsovvis_asserts root; default: $WSOVVIS_ASSERT_ROOT or /home/zyy/code/wsovvis_asserts")
    parser.add_argument("--output_root", default=None, help="default: <assert_root>/text_bank_llama3/lvvis")
    parser.add_argument("--output_name", default=None, help="profile output directory name; default profile_id, with _smokeN suffix when --max_classes is set")
    parser.add_argument("--profile", default="lvvis_visual_only_v1", help="profile id in third_party/uvlt_llama3/prompts/prompt_profiles.json")
    parser.add_argument("--train_annotation_json", default=None)
    parser.add_argument("--val_annotation_json", default=None)
    parser.add_argument("--max_classes", type=int, default=0, help="smoke subset: first N raw-id-sorted classes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse_responses", action="store_true", help="reuse llama3_responses.jsonl for CLIP-of-LLM only")

    # Llama args.
    parser.add_argument("--ckpt_dir", default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tokenizer_path", default="Meta-Llama-3-8B-Instruct/tokenizer.model")
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--master_port", default="56789")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_gen_len", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)

    # What to build.
    parser.add_argument("--build_llama_hidden", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build_clip_of_llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip_device", default="cuda:0")
    parser.add_argument("--clip_batch_size", type=int, default=256)

    parser.add_argument("--log_every_classes", type=int, default=20)
    parser.add_argument("--print_progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root()
    assert_root = _resolve_assert_root(repo_root, args.assert_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (assert_root / "text_bank_llama3" / "lvvis").resolve()
    profile = _resolve_profile(repo_root, args.profile)
    profile_id = str(profile["profile_id"])
    output_name = args.output_name
    if not output_name:
        output_name = profile_id if int(args.max_classes) <= 0 else f"{profile_id}_smoke{int(args.max_classes)}"
    output_dir = output_root / output_name
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
    _write_json(
        class_path,
        {
            "class_count": len(class_records),
            "full_class_count": len(full_class_map),
            "raw_id_order": "ascending",
            "classes": class_records,
            "train_annotation_json": str(train_ann),
            "val_annotation_json": str(val_ann),
            "does_not_use_coco_class_list": True,
        },
    )
    prompt_profile_path = _copy_prompt_profile(
        output_dir,
        profile,
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_gen_len": args.max_gen_len,
            "repeat": args.repeat,
            "max_classes": int(args.max_classes),
        },
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    profile_type = str(profile.get("profile_type"))
    if profile_type == "description_generation":
        result = _build_description_bank(
            args,
            repo_root=repo_root,
            output_dir=output_dir,
            profile=profile,
            class_items=class_items,
            class_presence=presence,
        )
    elif profile_type == "direct_concept_encoding":
        result = _build_direct_concept_bank(
            args,
            repo_root=repo_root,
            output_dir=output_dir,
            profile=profile,
            class_items=class_items,
        )
    elif profile_type == "clip_single_template":
        result = _build_clip_single_template_bank(
            args,
            repo_root=repo_root,
            output_dir=output_dir,
            profile=profile,
            class_items=class_items,
        )
    else:
        raise ValueError(f"unsupported profile_type={profile_type}")

    manifest = {
        "status": "PASS",
        "tool": "tools/build_lvvis_llama3_text_bank.py",
        "profile_id": profile_id,
        "profile_type": profile_type,
        "output_dir": str(output_dir),
        "asset_storage_policy": "canonical_text_bank_assets_under_wsovvis_asserts_not_codex_outputs",
        "repo_root": str(repo_root),
        "assert_root": str(assert_root),
        "class_count": len(class_items),
        "full_class_count": len(full_class_map),
        "raw_id_order": "ascending",
        "does_not_use_coco_class_list": True,
        "does_not_overwrite_clip_text_bank": True,
        "llama3_token_feature_alignment_fixed": True,
        "token_feature_alignment": "exact_full_forward_generated_token_or_class_span_slice",
        "uses_old_corr_feats": False,
        "all_vectors_finite": True,
        "all_mean_vectors_l2_normalized": True,
        "train_annotation_json": str(train_ann),
        "val_annotation_json": str(val_ann),
        "prompt_profile_path": str(prompt_profile_path),
        "lvvis_class_names_path": str(class_path),
        "runtime": result.get("generation_runtime", {}),
        "artifacts": result.get("artifacts", {}),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
