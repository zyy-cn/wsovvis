import os
import json
import time
import random
import logging
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sacred import Experiment

import detectron2.utils.comm as comm
from detectron2.engine import launch, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results

# SeqFormer project utilities from VNext
from detectron2.projects.seqformer import (
    add_seqformer_config,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.projects.seqformer.data import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    get_detection_dataset_dicts,
    DetrDatasetMapper,
)
from detectron2.solver.build import maybe_add_gradient_clipping


# -------------------------
# Sacred experiment
# -------------------------
ex = Experiment("wsovvis")


def _create_basic_stream_logger(fmt: str):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger


ex.logger = _create_basic_stream_logger("%(levelname)s - %(name)s - %(message)s")
ex.add_config("configs/wsovvis_sacred.yaml")


def _get_file_storage_base(_run):
    # Sacred CLI: --file_storage=BASEDIR / -F BASEDIR
    # In your previous entry, you used _run.meta_info['options']['--file_storage'] :contentReference[oaicite:10]{index=10}
    opts = _run.meta_info.get("options", {})
    return opts.get("--file_storage", None)


def _make_output_dir(_run):
    base = _get_file_storage_base(_run)
    if base and _run._id is not None:
        out = os.path.join(base, str(_run._id))
        Path(out).mkdir(parents=True, exist_ok=True)
        return out

    # Fallback (e.g., if you forgot --file_storage)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join("outputs", f"debug_{ts}")
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _import_project_modules(module_list):
    for m in module_lis:contentReference[oaicite:11]{index=11}import_module(m)


def _build_detectron2_args(run_cfg: dict, output_dir: str):
    d2 = run_cfg["d2"]
    dist = run_cfg["dist"]

    # Detectron2 expects opts as a flat list: ["A.B", "value", "C.D", "value", ...]
    # We'll force OUTPUT_DIR to the sacred run dir.
    opts = list(d2.get("opts", []))
    opts += ["OUTPUT_DIR", output_dir]

    args = SimpleNamespace(
        config_file=d2["config_file"],
        opts=opts,
        resume=bool(d2.get("resume", False)),
        eval_only=bool(d2.get("eval_only", False)),
        num_gpus=int(dist.get("num_gpus", torch.cuda.device_count())),
        num_machines=int(dist.get("num_machines", 1)),
        machine_rank=int(dist.get("machine_rank", 0)),
        dist_url=str(dist.get("dist_url", "auto")),
    )
    return args


# -------------------------
# SeqFormer Trainer (minimal copy from VNext SeqFormer train_net.py structure)
# -------------------------
def _build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    os.makedirs(output_folder, exist_ok=True)

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    evaluator_list = []

    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    elif evaluator_type == "ytvis":
        evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))
    else:
        raise NotImplementedError(f"no Evaluator for dataset {dataset_name} with type {evaluator_type}")

    return evaluator_list[0] if len(evaluator_list) == 1 else DatasetEvaluators(evaluator_list)


class Trainer(torch.nn.Module):
    """
    A thin wrapper to keep the code explicit:
    we will internally use detectron2 DefaultTrainer pattern via composition.
    """

    def __init__(self, cfg):
        from detectron2.engine import DefaultTrainer

        # Create a subclass on-the-fly to override classmethods cleanly
        class _T(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                return _build_evaluator(cfg, dataset_name, output_folder)

            @classmethod
            def build_train_loader(cls, cfg):
                dataset_name = cfg.DATASETS.TRAIN[0]
                if dataset_name.startswith("coco"):
                    mapper = DetrDatasetMapper(cfg, is_train=True)
                else:
                    # treat ytvis/lvvis/wsovvis as ytvis mapper first; you can refine later
                    mapper = YTVISDatasetMapper(cfg, is_train=True)

                dataset_dict = get_detection_dataset_dicts(
                    dataset_name,
                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                    proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
                )
                return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

            @classmethod
            def build_test_loader(cls, cfg, dataset_name):
                if dataset_name.startswith("coco"):
                    mapper = DetrDatasetMapper(cfg, is_train=False)
                else:
                    mapper = YTVISDatasetMapper(cfg, is_train=False)
                return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

            @classmethod
            def build_optimizer(cls, cfg, model):
                params = []
                memo = set()

                for key, value in model.named_parameters(recurse=True):
                    if not value.requires_grad:
                        continue
                    if value in memo:
                        continue
                    memo.add(value)

                    lr = cfg.SOLVER.BASE_LR
                    wd = cfg.SOLVER.WEIGHT_DECAY
                    if "backbone" in key:
                        lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER

                    params.append({"params": [value], "lr": lr, "weight_decay": wd})

                optimizer_type = cfg.SOLVER.OPTIMIZER
                if optimizer_type == "SGD":
                    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
                elif optimizer_type == "ADAMW":
                    optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
                else:
                    raise NotImplementedError(f"no optimizer type {optimizer_type}")

                optimizer = maybe_add_gradient_clipping(cfg, optimizer)
                return optimizer

        self._trainer = _T(cfg)

    @property
    def model(self):
        return self._trainer.model

    def resume_or_load(self, resume: bool):
        self._trainer.resume_or_load(resume=resume)

    def train(self):
        return self._trainer.train()

    @staticmethod
    def test(cfg, model):
        from detectron2.engine import DefaultTrainer
        return DefaultTrainer.test(cfg, model)


def _setup_cfg(args):
    cfg = get_cfg()
    add_seqformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def _main_worker(args):
    cfg = _setup_cfg(args)

    if args.eval_only:
        model = Trainer(cfg).model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def _parse_metrics_json(metrics_path: str):
    """
    Detectron2 JSONWriter writes one json per line to metrics.json. :contentReference[oaicite:12]{index=12}
    We'll compute:
      - last_record
      - best_ap_like (max over keys containing 'AP')
      - best_loss_like (min over keys containing 'loss')
    """
    last = None
    best_ap = {}
    best_loss = {}

    with open(metrics_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            last = rec
            for k, v in rec.items():
                if not isinstance(v, (int, float)):
                    continue
                lk = k.lower()
                if "ap" in lk:
                    best_ap[k] = max(best_ap.get(k, float("-inf")), v)
                if "loss" in lk:
                    best_loss[k] = min(best_loss.get(k, float("inf")), v)

    return {"last": last, "best_ap": best_ap, "best_loss": best_loss}


@ex.automain
def main(_run, _log):
    run_cfg = dict(_run.config)

    # 1) output dir = sacred file storage dir/<run_id>
    output_dir = _make_output_dir(_run)
    _log.info(f"[sacred] output_dir = {output_dir}")

    # 2) save a copy of sacred config for reproducibility
    with open(os.path.join(output_dir, "sacred_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    # 3) seed
    _seed_everything(int(run_cfg.get("seed", 42)))

    # 4) import project modules (dataset registration, custom meta-arch, etc.)
    _import_project_modules(run_cfg.get("project", {}).get("imports", []))

    # 5) build detectron2 args
    args = _build_detectron2_args(run_cfg, output_dir)

    # 6) launch multi-gpu
    launch(
        _main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    # 7) rank0: parse metrics.json and log to sacred
    if comm.is_main_process():
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            m = _parse_metrics_json(metrics_path)
            last = m["last"] or {}
            for k, v in m["best_ap"].items():
                _run.log_scalar(f"best_ap/{k}", float(v))
            for k, v in m["best_loss"].items():
                _run.log_scalar(f"best_loss/{k}", float(v))
            if "iteration" in last:
                _run.log_scalar("last/iteration", float(last["iteration"]))

            _run.add_artifact(metrics_path)
        else:
            _log.warning(f"metrics.json not found at {metrics_path}")
