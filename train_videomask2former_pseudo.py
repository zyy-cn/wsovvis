#!/usr/bin/env python3
"""Current-family WS-OVVIS training entry for DINOv2 + VideoMask2Former pseudo tubes."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MINVIS_ROOT = REPO_ROOT / "third_party" / "VNext" / "projects" / "InstMove" / "MinVIS_motion"

# Keep the active MinVIS_motion project importable without reviving the legacy SeqFormer path.
for path in (REPO_ROOT, MINVIS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN, get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config

import train_net_video as minvis_train_net
import wsovvis.modeling.backbone.dinov2_backbone  # noqa: F401
from wsovvis.data import register_wsovvis_datasets


def add_wsovvis_dinov2_vm2f_config(cfg) -> None:
    if not hasattr(cfg.MODEL, "DINOV2"):
        cfg.MODEL.DINOV2 = CN()
    cfg.MODEL.DINOV2.MODEL_NAME = "dinov2_vitb14"
    cfg.MODEL.DINOV2.DINO_DIM = 768
    cfg.MODEL.DINOV2.OUT_CHANNELS = 256
    cfg.MODEL.DINOV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DINOV2.FREEZE = True
    cfg.MODEL.DINOV2.REPO_PATH = "third_party/dinov2"
    cfg.MODEL.DINOV2.WEIGHTS = "weights/DINOv2/dinov2_vitb14_pretrain.pth"

    if not hasattr(cfg, "WSOVVIS"):
        cfg.WSOVVIS = CN()
    if not hasattr(cfg.WSOVVIS, "DATA"):
        cfg.WSOVVIS.DATA = CN()

    cfg.WSOVVIS.DATA.TRAIN_NAME = "ytvis_wsovvis_lvvis_pseudo_train_dinov2_vm2f"
    cfg.WSOVVIS.DATA.TRAIN_JSON = "outputs/videocutler_lvvis_png_dinov2_vm2f/train/pseudo_tube_ytvis.json"
    cfg.WSOVVIS.DATA.TRAIN_IMG_ROOT = "data/LV-VIS/train/JPEGImages"
    cfg.WSOVVIS.DATA.VAL_NAME = ""
    cfg.WSOVVIS.DATA.VAL_JSON = ""
    cfg.WSOVVIS.DATA.VAL_IMG_ROOT = ""


def _maybe_register_datasets(cfg) -> None:
    data_cfg = cfg.WSOVVIS.DATA
    register_wsovvis_datasets(
        {
            "train_name": str(data_cfg.TRAIN_NAME),
            "train_json": str(data_cfg.TRAIN_JSON),
            "train_img_root": str(data_cfg.TRAIN_IMG_ROOT),
            "val_name": str(data_cfg.VAL_NAME),
            "val_json": str(data_cfg.VAL_JSON),
            "val_img_root": str(data_cfg.VAL_IMG_ROOT),
        }
    )


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_wsovvis_dinov2_vm2f_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    _maybe_register_datasets(cfg)
    cfg.DATASETS.TRAIN = (str(cfg.WSOVVIS.DATA.TRAIN_NAME),)
    if cfg.WSOVVIS.DATA.VAL_NAME and cfg.WSOVVIS.DATA.VAL_JSON and cfg.WSOVVIS.DATA.VAL_IMG_ROOT:
        cfg.DATASETS.TEST = (str(cfg.WSOVVIS.DATA.VAL_NAME),)
    else:
        cfg.DATASETS.TEST = ()
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = minvis_train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = minvis_train_net.Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = minvis_train_net.Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
