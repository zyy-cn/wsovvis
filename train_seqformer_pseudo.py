#!/usr/bin/env python3
"""Minimal SeqFormer training entry for WS-OVVIS pseudo tubes.

Goals for this stage:
  1) Read pseudo tube json (YTVIS-style, class-agnostic)
  2) Train / fit SeqFormer on that pseudo supervision
  3) Evaluate on LV-VIS ground-truth masks (class-agnostic json)

Experiment management:
  - sacred (supports -F/--file_storage)
Multi-GPU:
  - detectron2.engine.launch (DDP)

Usage example:
  python train_seqformer_pseudo.py with configs/seqformer_pseudo_sacred.yaml
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment("wsovvis_seqformer")

from detectron2.config import CfgNode as CN

import wsovvis.modeling.backbone.dinov2_backbone

def add_dinov2_config(cfg):
    # 只在不存在时添加，避免重复覆盖
    if not hasattr(cfg.MODEL, "DINOV2"):
        cfg.MODEL.DINOV2 = CN()
    cfg.MODEL.DINOV2.MODEL_NAME = "dinov2_vitb14"
    cfg.MODEL.DINOV2.DINO_DIM = 768
    cfg.MODEL.DINOV2.OUT_CHANNELS = 256
    cfg.MODEL.DINOV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DINOV2.FREEZE = True
    cfg.MODEL.DINOV2.REPO_PATH = ""
    cfg.MODEL.DINOV2.WEIGHTS = ""


def _get_file_storage(_run, default_root: str) -> str:
    """Sacred CLI supports `-F <dir>` / `--file_storage <dir>`.

    When not provided, use default_root.
    """

    opts = (_run.meta_info or {}).get("options", {}) or {}
    # sacred stores raw options, keys are like '--file_storage'
    return opts.get("--file_storage", default_root)


def _get_run_dir(_run, output_root: str) -> str:
    """Prefer sacred's FileStorageObserver dir when available."""

    for o in getattr(_run, "observers", []) or []:
        if isinstance(o, FileStorageObserver) and getattr(o, "dir", None):
            return str(o.dir)
    return os.path.join(output_root, str(_run._id))


@ex.config
def _default_cfg():
    # detectron2/vnext config
    d2_cfg_path = "projects/SeqFormer/configs/ytvis_2019/seqformer_ytvis_2019.yaml"
    d2_opts: List[str] = []

    # dataset registration
    data = {
        "train_name": "wsovvis_pseudo_train",
        "train_json": "outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json",
        "train_img_root": "data/LV-VIS/train/JPEGImages",
        "val_name": "lvvis_val_agnostic",
        "val_json": "data/LV-VIS/annotations/lvvis_val_agnostic.json",
        "val_img_root": "data/LV-VIS/val/JPEGImages",
        "num_classes": 1,
    }

    # runtime
    output_root = "runs/wsovvis_seqformer"
    seed = 42
    resume = False
    eval_only = False

    # distributed
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "tcp://127.0.0.1:29500"


def _maybe_set_num_classes(cfg, num_classes: int) -> None:
    """Best-effort set num classes across common keys.

    VNext/SeqFormer configs differ across versions, so we only set keys that exist.
    """

    # helper to safely walk CfgNode
    def has_path(node, path: str) -> bool:
        cur = node
        parts = path.split(".")
        for p in parts:
            if not hasattr(cur, p) and (p not in cur):
                return False
            cur = getattr(cur, p) if hasattr(cur, p) else cur[p]
        return True

    def set_path(node, path: str, value: int) -> None:
        cur = node
        parts = path.split(".")
        for p in parts[:-1]:
            cur = getattr(cur, p) if hasattr(cur, p) else cur[p]
        last = parts[-1]
        if hasattr(cur, last):
            setattr(cur, last, value)
        else:
            cur[last] = value

    candidates = [
        "MODEL.SEQFORMER.NUM_CLASSES",
        "MODEL.MASK_FORMER.NUM_CLASSES",
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES",
        "MODEL.DETR.NUM_CLASSES",
        "MODEL.ROI_HEADS.NUM_CLASSES",
    ]
    for k in candidates:
        if has_path(cfg, k):
            set_path(cfg, k, int(num_classes))


def _setup_cfg(
    *,
    d2_cfg_path: str,
    d2_opts: List[str],
    output_dir: str,
    train_name: str,
    val_name: Optional[str],
    num_classes: int,
    seed: int,
):
    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger

    setup_logger(output=output_dir, name="detectron2")

    cfg = get_cfg()
    # add SeqFormer config
    from detectron2.projects.seqformer import add_seqformer_config

    add_dinov2_config(cfg)
    add_seqformer_config(cfg)
    cfg.merge_from_file(d2_cfg_path)
    if d2_opts:
        cfg.merge_from_list(d2_opts)

    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = seed

    # datasets
    cfg.DATASETS.TRAIN = (train_name,)
    if val_name:
        cfg.DATASETS.TEST = (val_name,)

    cfg.defrost()
    _maybe_set_num_classes(cfg, num_classes)
    cfg.freeze()
    return cfg

def _resolve_seqformer_trainer():
    """
    Resolve SeqFormer Trainer across different VNext/SeqFormer layouts.

    We try:
      1) detectron2.projects.seqformer.{Trainer,SeqFormerTrainer}
      2) detectron2.projects.seqformer.trainer.{Trainer,SeqFormerTrainer}
      3) sibling project-level train_net.py (common detectron2-project style)
    """
    import importlib
    import sys
    from pathlib import Path

    pkg = importlib.import_module("detectron2.projects.seqformer")

    # 1) exported directly
    for name in ("Trainer", "SeqFormerTrainer"):
        if hasattr(pkg, name):
            return getattr(pkg, name)

    # 2) common submodule
    for mod_name in (
        "detectron2.projects.seqformer.trainer",
        "detectron2.projects.seqformer.training",
        "detectron2.projects.seqformer.engine",
    ):
        try:
            m = importlib.import_module(mod_name)
            for name in ("Trainer", "SeqFormerTrainer"):
                if hasattr(m, name):
                    return getattr(m, name)
        except Exception:
            pass

    # 3) project-level train_net.py (often sits at .../projects/SeqFormer/train_net.py)
    # pkg.__file__ is .../projects/SeqFormer/seqformer/__init__.py
    pkg_dir = Path(pkg.__file__).resolve().parent          # .../SeqFormer/seqformer
    project_dir = pkg_dir.parent                           # .../SeqFormer
    train_net = project_dir / "train_net.py"
    if train_net.is_file():
        sys.path.insert(0, str(project_dir))
        try:
            m = importlib.import_module("train_net")
            if hasattr(m, "Trainer"):
                return getattr(m, "Trainer")
            if hasattr(m, "SeqFormerTrainer"):
                return getattr(m, "SeqFormerTrainer")
        finally:
            # remove only the exact inserted entry
            if sys.path and sys.path[0] == str(project_dir):
                sys.path.pop(0)

    raise ImportError(
        "Cannot resolve SeqFormer Trainer. Checked: "
        "detectron2.projects.seqformer.{Trainer,SeqFormerTrainer}, "
        "detectron2.projects.seqformer.trainer, and project-level train_net.py next to the seqformer package."
    )


def _main_worker(*args):
    """Per-process entry (called by detectron2.launch / mp.spawn).

    Notes:
      - Different detectron2 forks may call main_func() as:
          main_func(*user_args) OR main_func(local_rank, *user_args)
      - To be robust, we accept *args and retrieve cfg_dict either from args
        or from an env var WSOVVIS_CFG_JSON written by the parent process.
    """

    # Try to recover cfg_dict from args (supports: (cfg_dict,), (local_rank,cfg_dict), etc.)
    cfg_dict = None
    local_rank = None
    for a in args:
        if isinstance(a, int):
            local_rank = a
        if isinstance(a, dict) and "d2_cfg_path" in a and "data" in a:
            cfg_dict = a

    if cfg_dict is None:
        cfg_json = os.environ.get("WSOVVIS_CFG_JSON", "")
        if not cfg_json or not os.path.exists(cfg_json):
            raise RuntimeError(
                "No cfg_dict passed to _main_worker and WSOVVIS_CFG_JSON is not set/invalid. "
                "This usually means detectron2.launch did not forward args. "
                "Fix: ensure run() writes WSOVVIS_CFG_JSON before launch()."
            )
        with open(cfg_json, "r") as f:
            cfg_dict = json.load(f)

    # Rank info (prefer detectron2 comm once process group is initialized)
    try:
        from detectron2.utils import comm
        rank = comm.get_rank()
        local_rank = comm.get_local_rank() if local_rank is None else local_rank
    except Exception:
        rank = 0
        local_rank = 0 if local_rank is None else local_rank

    # Register datasets (must happen in every process).
    from wsovvis.data import register_wsovvis_datasets

    register_wsovvis_datasets(cfg_dict["data"])

    d2_cfg = _setup_cfg(
        d2_cfg_path=cfg_dict["d2_cfg_path"],
        d2_opts=cfg_dict.get("d2_opts", []),
        output_dir=cfg_dict["output_dir"],
        train_name=cfg_dict["data"]["train_name"],
        val_name=cfg_dict["data"].get("val_name"),
        num_classes=int(cfg_dict["data"].get("num_classes", 1)),
        seed=int(cfg_dict.get("seed", 42)),
    )

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.logger import setup_logger

    setup_logger(output=d2_cfg.OUTPUT_DIR, distributed_rank=rank, name="wsovvis")

    # from detectron2.projects.seqformer import Trainer
    Trainer = _resolve_seqformer_trainer()

    if cfg_dict.get("eval_only", False):
        model = Trainer.build_model(d2_cfg)
        DetectionCheckpointer(model, save_dir=d2_cfg.OUTPUT_DIR).resume_or_load(
            d2_cfg.MODEL.WEIGHTS, resume=cfg_dict.get("resume", False)
        )

        res = Trainer.test(d2_cfg, model)
        return res

    trainer = Trainer(d2_cfg)
    trainer.resume_or_load(resume=cfg_dict.get("resume", False))
    trainer.train()

@ex.main
def run(
    _run,
    d2_cfg_path,
    d2_opts,
    data,
    output_root,
    seed,
    resume,
    eval_only,
    num_gpus,
    num_machines,
    machine_rank,
    dist_url,
):
    # Prefer sacred file storage if user passed -F/--file_storage.
    fs_root = _get_file_storage(_run, output_root)
    run_dir = _get_run_dir(_run, fs_root)
    os.makedirs(run_dir, exist_ok=True)

    output_dir = os.path.join(run_dir, "d2")
    os.makedirs(output_dir, exist_ok=True)

    cfg_dict = {
        "d2_cfg_path": d2_cfg_path,
        "d2_opts": d2_opts,
        "data": data,
        "output_dir": output_dir,
        "seed": seed,
        "resume": resume,
        "eval_only": eval_only,
    }

    # Write cfg_dict to disk and expose via env var so worker processes can load it
    # even if the detectron2 fork does not forward `args` correctly.
    cfg_json_path = os.path.join(run_dir, "cfg_runtime.json")
    with open(cfg_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    os.environ["WSOVVIS_CFG_JSON"] = cfg_json_path

    from detectron2.engine import launch

    return launch(
        _main_worker,
        num_gpus_per_machine=int(num_gpus),
        num_machines=int(num_machines),
        machine_rank=int(machine_rank),
        dist_url=dist_url,
        args=(cfg_dict,),
    )


if __name__ == "__main__":
    # Sacred CLI entry
    ex.run_commandline()