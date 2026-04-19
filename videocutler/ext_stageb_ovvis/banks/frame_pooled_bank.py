from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict


def _deprecated() -> RuntimeError:
    return RuntimeError(
        "frame_pooled_bank is deprecated: pooled_frame_records.jsonl has been retired from the canonical runtime chain"
    )


def pooled_frame_records_path(output_root: Path, dataset_name: str) -> Path:
    raise _deprecated()


def pooled_frame_payload_rel(clip_id: str) -> str:
    raise _deprecated()


def read_pooled_frame_vector(artifact_parent_dir: Path, locator: str):
    raise _deprecated()


def build_pooled_frame_bank(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise _deprecated()
