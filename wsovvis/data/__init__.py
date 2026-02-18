from __future__ import annotations

from .ytvis import register_ytvis_like_dataset
from .wsovvis_register import register_wsovvis_datasets

__all__ = [
    "register_ytvis_like_dataset",
    "register_wsovvis_datasets",
]
