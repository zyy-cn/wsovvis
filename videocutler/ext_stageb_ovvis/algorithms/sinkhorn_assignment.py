from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class SinkhornAssignmentConfig:
    tau: float = 0.15
    iters: int = 5
    row_cap_scale: float = 2.0
    eps: float = 1e-6
    invalid_logit: float = -1.0e4


def capped_sinkhorn_assignment(
    scores: torch.Tensor,
    q_mask: torch.Tensor,
    c_mask: torch.Tensor,
    column_demand: torch.Tensor,
    *,
    config: SinkhornAssignmentConfig | None = None,
) -> torch.Tensor:
    """Batched capped Sinkhorn over small clip candidate sets.

    Args:
        scores: [B, Q, M] score matrix on GPU.
        q_mask: [B, Q] valid trajectory mask.
        c_mask: [B, M] valid candidate/class mask.
        column_demand: [B, M] desired mass for each candidate column.

    Returns:
        P: [B, Q, M] non-negative assignment mass. Columns approximate the
        requested demand; rows are softly capped by config.row_cap_scale.
    """
    cfg = config or SinkhornAssignmentConfig()
    if scores.ndim != 3:
        raise ValueError(f'scores must be [B,Q,M], got shape={tuple(scores.shape)}')
    if q_mask.ndim != 2 or c_mask.ndim != 2 or column_demand.ndim != 2:
        raise ValueError('q_mask, c_mask, and column_demand must be rank-2 tensors')
    B, Q, M = scores.shape
    if tuple(q_mask.shape) != (B, Q) or tuple(c_mask.shape) != (B, M) or tuple(column_demand.shape) != (B, M):
        raise ValueError('mask/demand shapes must match scores')

    valid = q_mask[:, :, None].bool() & c_mask[:, None, :].bool() & (column_demand[:, None, :] > 0)
    tau = max(float(cfg.tau), float(cfg.eps))
    eps = float(cfg.eps)
    logP = scores.float() / tau
    logP = logP.masked_fill(~valid, float(cfg.invalid_logit))

    demand = column_demand.float().masked_fill(~c_mask.bool(), 0.0).clamp_min(0.0)
    total_demand = demand.sum(dim=1).clamp_min(eps)  # [B]
    q_count = q_mask.float().sum(dim=1).clamp_min(1.0)  # [B]
    row_cap = (float(cfg.row_cap_scale) * total_demand / q_count).clamp_min(eps)  # [B]

    for _ in range(max(1, int(cfg.iters))):
        # Column normalization to requested demand.
        col_lse = torch.logsumexp(logP, dim=1, keepdim=True)
        logP = logP - col_lse + torch.log(demand.clamp_min(eps))[:, None, :]
        logP = logP.masked_fill(~valid, float(cfg.invalid_logit))

        # Row capacity clipping; does not force every trajectory to take mass.
        P = torch.exp(logP).masked_fill(~valid, 0.0)
        load = P.sum(dim=2)  # [B,Q]
        scale = torch.clamp(row_cap[:, None] / (load + eps), max=1.0)
        logP = logP + torch.log(scale.clamp_min(eps))[:, :, None]
        logP = logP.masked_fill(~valid, float(cfg.invalid_logit))

    # Final column normalization, then one final row clipping. This keeps class
    # coverage explicit while still enforcing a conservative hub/load brake.
    col_lse = torch.logsumexp(logP, dim=1, keepdim=True)
    logP = logP - col_lse + torch.log(demand.clamp_min(eps))[:, None, :]
    logP = logP.masked_fill(~valid, float(cfg.invalid_logit))
    P = torch.exp(logP).masked_fill(~valid, 0.0)
    load = P.sum(dim=2)
    scale = torch.clamp(row_cap[:, None] / (load + eps), max=1.0)
    P = (P * scale[:, :, None]).masked_fill(~valid, 0.0)
    return P


def sinkhorn_loss_from_assignment(
    scores: torch.Tensor,
    assignment: torch.Tensor,
    *,
    stopgrad_assignment: bool = True,
    normalize_by_demand: torch.Tensor | None = None,
) -> torch.Tensor:
    P = assignment.detach() if bool(stopgrad_assignment) else assignment
    per_clip = -(P * scores.float()).sum(dim=(1, 2))
    if normalize_by_demand is None:
        denom = P.sum(dim=(1, 2)).clamp_min(1.0)
    else:
        denom = normalize_by_demand.float().clamp_min(1.0)
    return (per_clip / denom).mean()



def yprime_only_nce_loss_from_assignment(
    scores: torch.Tensor,
    assignment: torch.Tensor,
    kind: torch.Tensor,
    c_mask: torch.Tensor,
    *,
    stopgrad_assignment: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft-label InfoNCE/CE over Y' columns only.

    This loss uses the Sinkhorn assignment as soft positive labels, but the
    contrastive denominator is restricted to confirmed observed/known columns
    (kind == 1, i.e. Y'). Extra columns are intentionally excluded from the
    denominator and from positive labels so hidden/unobserved full-vocabulary
    classes are not treated as negatives.

    Args:
        scores: [B,Q,M] candidate scores for Y' plus optional extra columns.
        assignment: [B,Q,M] Sinkhorn assignment mass.
        kind: [B,M], 1 for Y'/known columns and 2 for extra columns.
        c_mask: [B,M] valid candidate mask.

    Returns:
        Scalar loss averaged over clips with non-zero Y' assignment mass.
    """
    if scores.ndim != 3 or assignment.ndim != 3:
        raise ValueError('scores and assignment must be [B,Q,M] tensors')
    if tuple(scores.shape) != tuple(assignment.shape):
        raise ValueError('scores and assignment must have the same shape')
    B, Q, M = scores.shape
    if tuple(kind.shape) != (B, M) or tuple(c_mask.shape) != (B, M):
        raise ValueError('kind and c_mask shapes must match [B,M]')

    yprime_mask = c_mask.bool() & (kind == 1)
    # Invalid columns must not participate in the denominator.
    masked_scores = scores.float().masked_fill(~yprime_mask[:, None, :], -1.0e4)
    log_probs = masked_scores - torch.logsumexp(masked_scores, dim=2, keepdim=True)

    P = assignment.detach() if bool(stopgrad_assignment) else assignment
    P_y = P.float().masked_fill(~yprime_mask[:, None, :], 0.0)
    mass = P_y.sum(dim=(1, 2))
    per_clip = -(P_y * log_probs).sum(dim=(1, 2)) / mass.clamp_min(float(eps))
    valid_clip = mass > float(eps)
    if bool(valid_clip.any()):
        return per_clip[valid_clip].mean()
    # Preserve gradient connectivity for pathological empty microbatches.
    return scores.float().sum() * 0.0


def assignment_metrics(
    assignment: torch.Tensor,
    q_mask: torch.Tensor,
    c_mask: torch.Tensor,
    column_demand: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        P = assignment.detach().float()
        q_valid = q_mask.bool()
        c_valid = c_mask.bool() & (column_demand > 0)
        col_mass = P.sum(dim=1)
        row_load = P.sum(dim=2)
        valid_col_mass = col_mass[c_valid]
        demand = column_demand.float()[c_valid]
        if valid_col_mass.numel() > 0:
            coverage_ratio = valid_col_mass / demand.clamp_min(1e-6)
            column_coverage_mean = float(coverage_ratio.mean().detach().cpu().item())
            column_coverage_min = float(coverage_ratio.min().detach().cpu().item())
        else:
            column_coverage_mean = 0.0
            column_coverage_min = 0.0
        valid_row_load = row_load[q_valid]
        if valid_row_load.numel() > 0:
            max_row_load = float(valid_row_load.max().detach().cpu().item())
            mean_row_load = float(valid_row_load.mean().detach().cpu().item())
            sorted_load, _ = torch.sort(valid_row_load)
            n = sorted_load.numel()
            total = sorted_load.sum().clamp_min(1e-6)
            idx = torch.arange(1, n + 1, device=sorted_load.device, dtype=sorted_load.dtype)
            gini = float(((2 * idx - n - 1) * sorted_load).sum().div(n * total).detach().cpu().item())
            effective = float(((valid_row_load.sum() ** 2) / (valid_row_load.square().sum().clamp_min(1e-6))).detach().cpu().item())
        else:
            max_row_load = 0.0
            mean_row_load = 0.0
            gini = 0.0
            effective = 0.0
    return {
        'sinkhorn_column_coverage_mean': column_coverage_mean,
        'sinkhorn_column_coverage_min': column_coverage_min,
        'sinkhorn_max_row_load_mean': max_row_load,
        'sinkhorn_mean_row_load': mean_row_load,
        'sinkhorn_load_gini': gini,
        'sinkhorn_effective_num_trajectories': effective,
    }
