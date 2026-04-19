from __future__ import annotations

"""G8 inference/evaluation bridge modules.

This package is intentionally add-only. It provides the new G8 canonical
inference/evaluation surfaces required by the package contract without changing
legacy G7 training or legacy evaluator behavior.
"""

__all__ = [
    "g8_bridge",
    "internal_val",
    "external_lvvis",
    "external_ytvis2019",
]
