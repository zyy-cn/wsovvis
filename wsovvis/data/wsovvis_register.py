from __future__ import annotations

from typing import Any, Dict, Optional

from .ytvis import register_ytvis_like_dataset


def register_wsovvis_datasets(cfg: Dict[str, Any]) -> None:
    """Register the datasets needed for WS-OVVIS.

    Expected keys in cfg:
      - train_name, train_json, train_img_root
      - val_name, val_json, val_img_root
    """

    train_name = cfg["train_name"]
    train_json = cfg["train_json"]
    train_img_root = cfg["train_img_root"]

    val_name = cfg.get("val_name")
    val_json = cfg.get("val_json")
    val_img_root = cfg.get("val_img_root")

    register_ytvis_like_dataset(
        name=train_name,
        json_file=train_json,
        image_root=train_img_root,
        evaluator_type="ytvis",
    )

    if val_name and val_json and val_img_root:
        register_ytvis_like_dataset(
            name=val_name,
            json_file=val_json,
            image_root=val_img_root,
            evaluator_type="ytvis",
        )
