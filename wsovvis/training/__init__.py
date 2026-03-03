"""Stage D training-side plumbing helpers."""

from .staged_attribution_plumbing_v1 import (
    StageDAttributionPlumbingConfig,
    StageDAttributionPlumbingError,
    build_stage_d_attribution_consumption_boundary,
    build_stage_d_objective_coupling_decision,
    consume_stage_d_attribution_config,
    resolve_stage_d_attribution_plumbing,
)

__all__ = [
    "StageDAttributionPlumbingConfig",
    "StageDAttributionPlumbingError",
    "build_stage_d_attribution_consumption_boundary",
    "build_stage_d_objective_coupling_decision",
    "consume_stage_d_attribution_config",
    "resolve_stage_d_attribution_plumbing",
]
