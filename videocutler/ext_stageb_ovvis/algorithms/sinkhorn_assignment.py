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




def yprime_nce_with_safe_negatives_loss_from_assignment(
    scores: torch.Tensor,
    assignment: torch.Tensor,
    kind: torch.Tensor,
    c_mask: torch.Tensor,
    *,
    full_scores: torch.Tensor,
    yidx: torch.Tensor,
    raw_text_cos_all: torch.Tensor,
    safe_neg_count: int = 64,
    safe_neg_weight: float = 0.25,
    text_sim_exclude_threshold: float = 0.50,
    exclude_model_topk: int = 100,
    generator_seed: int = 0,
    stopgrad_assignment: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Y'-restricted soft-label InfoNCE plus filtered full-vocab safe negatives.

    The positive soft labels remain Sinkhorn mass on Y' columns. The denominator
    contains all Y' columns plus K sampled full-vocabulary negatives that pass
    conservative filters:
      - not in the current clip candidate set (Y' or extra),
      - not a raw-text near neighbor of any Y' class,
      - not in the current trajectory model top-K predictions.

    Hidden/unobserved classes are therefore not exposed to a naive full-vocab
    denominator. The sampled negatives only provide weak cross-vocabulary rank
    pressure, controlled by safe_neg_weight.
    """
    if scores.ndim != 3 or assignment.ndim != 3 or full_scores.ndim != 3:
        raise ValueError('scores, assignment, and full_scores must be [B,Q,*] tensors')
    if tuple(scores.shape) != tuple(assignment.shape):
        raise ValueError('scores and assignment must have the same shape')
    B, Q, M = scores.shape
    if tuple(kind.shape) != (B, M) or tuple(c_mask.shape) != (B, M) or tuple(yidx.shape) != (B, M):
        raise ValueError('kind/c_mask/yidx shapes must match [B,M]')
    if full_scores.shape[0] != B or full_scores.shape[1] != Q:
        raise ValueError('full_scores must match scores on [B,Q]')
    C = int(full_scores.shape[2])
    K = max(0, int(safe_neg_count))
    beta = max(0.0, float(safe_neg_weight))
    if K <= 0 or beta <= 0.0:
        loss = yprime_only_nce_loss_from_assignment(
            scores,
            assignment,
            kind,
            c_mask,
            stopgrad_assignment=bool(stopgrad_assignment),
            eps=eps,
        )
        return loss, {
            'safe_neg_enabled': False,
            'safe_neg_count': 0,
            'safe_neg_valid_mean': 0.0,
            'safe_neg_valid_min': 0.0,
            'safe_neg_fallback_row_count': 0,
        }

    yprime_mask = c_mask.bool() & (kind == 1)
    cand_mask = c_mask.bool()

    # Build per-clip base safe mask [B,C]. Small B loop avoids brittle scatter
    # behavior with padded yidx while preserving the expensive work on GPU.
    base_safe = torch.ones((B, C), device=full_scores.device, dtype=torch.bool)
    for b in range(B):
        cand_idx = yidx[b, cand_mask[b]].long()
        if cand_idx.numel() > 0:
            base_safe[b, cand_idx.clamp(0, C - 1)] = False
        y_idx = yidx[b, yprime_mask[b]].long()
        if y_idx.numel() > 0:
            sims = raw_text_cos_all[y_idx.clamp(0, C - 1)].amax(dim=0)
            base_safe[b] &= sims <= float(text_sim_exclude_threshold)

    safe_mask = base_safe[:, None, :].expand(B, Q, C).clone()
    topk = min(max(0, int(exclude_model_topk)), C)
    if topk > 0:
        top_idx = full_scores.detach().topk(k=topk, dim=2).indices
        safe_mask.scatter_(2, top_idx, False)

    weights = safe_mask.reshape(B * Q, C).float()
    valid_count = weights.sum(dim=1)

    # Fallback 1: keep candidate/raw-text exclusions but ignore model-topK if
    # those filters made a row empty. Fallback 2: sample anywhere if a clip has
    # no safe classes at all (rare; recorded for audit).
    fallback_row = valid_count <= 0
    if bool(fallback_row.any()):
        fallback_weights = base_safe[:, None, :].expand(B, Q, C).reshape(B * Q, C).float()
        weights = torch.where(fallback_row[:, None], fallback_weights, weights)
        valid_count = weights.sum(dim=1)
    fallback_row2 = valid_count <= 0
    if bool(fallback_row2.any()):
        weights = torch.where(fallback_row2[:, None], torch.ones_like(weights), weights)
        valid_count = weights.sum(dim=1)

    gen = torch.Generator(device=full_scores.device)
    gen.manual_seed(int(generator_seed))
    safe_idx = torch.multinomial(weights, num_samples=K, replacement=True, generator=gen)
    safe_logits = full_scores.reshape(B * Q, C).gather(1, safe_idx).reshape(B, Q, K)

    yprime_scores = scores.float().masked_fill(~yprime_mask[:, None, :], -1.0e4)
    safe_logits = safe_logits.float() + torch.log(torch.tensor(beta, device=full_scores.device, dtype=torch.float32))
    concat_logits = torch.cat([yprime_scores, safe_logits], dim=2)
    log_probs = concat_logits - torch.logsumexp(concat_logits, dim=2, keepdim=True)
    log_probs_y = log_probs[:, :, :M]

    P = assignment.detach() if bool(stopgrad_assignment) else assignment
    P_y = P.float().masked_fill(~yprime_mask[:, None, :], 0.0)
    mass = P_y.sum(dim=(1, 2))
    per_clip = -(P_y * log_probs_y).sum(dim=(1, 2)) / mass.clamp_min(float(eps))
    valid_clip = mass > float(eps)
    if bool(valid_clip.any()):
        loss = per_clip[valid_clip].mean()
    else:
        loss = scores.float().sum() * 0.0

    with torch.no_grad():
        metrics = {
            'safe_neg_enabled': True,
            'safe_neg_count': int(K),
            'safe_neg_weight': float(beta),
            'safe_neg_text_sim_threshold': float(text_sim_exclude_threshold),
            'safe_neg_exclude_model_topk': int(topk),
            'safe_neg_valid_mean': float(valid_count.detach().float().mean().cpu().item()),
            'safe_neg_valid_min': float(valid_count.detach().float().min().cpu().item()),
            'safe_neg_fallback_row_count': int((fallback_row | fallback_row2).detach().sum().cpu().item()),
        }
    return loss, metrics


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
