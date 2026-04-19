from __future__ import annotations

"""Thin wrapper binding to the vendored LV-VIS evaluator class.

This file preserves the canonical official-LVVIS import surface required by the
package while delegating execution to the snapshot-local YTVIS evaluation stack.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def _load_backend_module(module_filename: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "videocutler" / "mask2former_video" / "data_video" / "datasets" / "ytvis_api" / module_filename
    spec = spec_from_file_location(f"_wsovvis_lvviseval_backend_{module_filename.replace('.', '_')}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load LV-VIS backend module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backend = _load_backend_module("ytvoseval.py")
LVVISEval = _backend.YTVOSeval

__all__ = ["LVVISEval"]
