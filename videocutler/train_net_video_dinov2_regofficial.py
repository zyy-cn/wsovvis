try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from ext_dinov2_regofficial import add_dinov2_vit_adapter_regofficial_config
from ext_dinov2_regofficial.modeling.backbone.d2_dinov2_vit_adapter_regofficial import D2DinoV2ViTAdapterRegOfficial  # noqa: F401
from train_net_video import Trainer


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_dinov2_vit_adapter_regofficial_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    test_dataset = getattr(args, "test_dataset", "")
    train_dataset = getattr(args, "train_dataset", "")
    steps = int(getattr(args, "steps", 0))
    if test_dataset != "":
        cfg.DATASETS.TEST = (test_dataset,)
    if train_dataset != "":
        cfg.DATASETS.TRAIN = (train_dataset,)
    if steps != 0:
        cfg.SOLVER.STEPS = (steps,)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former_video")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
