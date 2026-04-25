from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ExtraMiningScoreVariant:
    """Configuration for one extra-mining simulator score.

    The registry is intentionally tensor-only: it does not read checkpoints,
    materialization files, or responsibility records.  This keeps the same score
    functions reusable by the offline simulator and, later, by the training
    backend behind an explicit config switch.
    """

    name: str
    family: str
    aggregation: str = "max"
    topm: int = 3
    lse_tau: float = 0.10
    blend_rho: float = 0.50
    alpha_obs: float = 0.25
    obs_sim_max: Optional[float] = 0.90
    hub_prior: Optional[str] = None
    hub_lambda: float = 0.0
    csls_k: int = 20
    reciprocal_eta: float = 0.0
    residual_topm: int = 3
    generic_lambda: float = 0.0
    consensus_gamma: float = 0.0
    consensus_key: str = "support_p95"
    mmr_lambda_text: float = 0.0
    mmr_top_l: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


def _require(cache: TensorMap, key: str) -> torch.Tensor:
    if key not in cache:
        raise KeyError(f"missing simulator tensor cache key: {key}")
    value = cache[key]
    if not torch.is_tensor(value):
        raise TypeError(f"cache key {key!r} is not a torch.Tensor")
    return value


def _aggregation_score(cache: TensorMap, variant: ExtraMiningScoreVariant) -> torch.Tensor:
    agg = str(variant.aggregation)
    if agg == "max":
        return _require(cache, "agg_max")
    if agg == "topm_mean":
        key = f"agg_topm{int(variant.topm)}"
        return _require(cache, key)
    if agg == "logsumexp":
        key = f"agg_lse_tau{_tau_key(float(variant.lse_tau))}"
        return _require(cache, key)
    if agg == "blend_max_topm":
        max_score = _require(cache, "agg_max")
        topm_score = _require(cache, f"agg_topm{int(variant.topm)}")
        rho = float(variant.blend_rho)
        return (1.0 - rho) * max_score + rho * topm_score
    if agg == "residual_topm":
        return _require(cache, f"residual_topm{int(variant.residual_topm)}")
    raise ValueError(f"unsupported aggregation: {agg}")


def _tau_key(value: float) -> str:
    # Stable key format shared by simulator aggregation cache.
    return str(float(value)).replace(".", "p").replace("-", "m")


def tau_key(value: float) -> str:
    return _tau_key(value)


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - torch.mean(x)) / torch.clamp(torch.std(x), min=float(eps))


def score_variant(cache: TensorMap, variant: ExtraMiningScoreVariant) -> torch.Tensor:
    """Return a [num_clips, vocab_size] score tensor for a non-MMR variant.

    Y' masking and observed-neighbor hard filtering are applied by the simulator
    evaluator after score construction.  This function only computes real-valued
    scores.
    """
    family = str(variant.family)
    base = _aggregation_score(cache, variant).clone()
    obs_sim = _require(cache, "obs_sim")

    if family in {"current", "anti_hub", "topm", "logsumexp", "blend", "consensus"}:
        score = base - float(variant.alpha_obs) * obs_sim
        if family == "consensus":
            score = score + float(variant.consensus_gamma) * _require(cache, str(variant.consensus_key))
        if variant.hub_prior:
            score = score - float(variant.hub_lambda) * _hub_prior_tensor(cache, str(variant.hub_prior)).reshape(1, -1)
        return score

    if family == "csls":
        # Use precomputed local densities matching the requested aggregation.
        agg_name = _aggregation_name(variant)
        query_density = _require(cache, f"density_query_{agg_name}_k{int(variant.csls_k)}").reshape(-1, 1)
        class_density = _require(cache, f"density_class_{agg_name}_k{int(variant.csls_k)}").reshape(1, -1)
        return 2.0 * base - query_density - class_density - float(variant.alpha_obs) * obs_sim

    if family == "reciprocal":
        # Reciprocal penalty approximates class-to-clip specificity.
        agg_name = _aggregation_name(variant)
        class_rank = _require(cache, f"class_to_clip_rank_{agg_name}")
        return base - float(variant.reciprocal_eta) * torch.log1p(class_rank) - float(variant.alpha_obs) * obs_sim

    if family == "residual":
        score = base
        if variant.hub_prior:
            score = score - float(variant.hub_lambda) * _hub_prior_tensor(cache, str(variant.hub_prior)).reshape(1, -1)
        # residual_over_observed already subtracts the observed visual explanation.
        return score

    if family == "genericity":
        generic = _require(cache, "genericity_prior").reshape(1, -1)
        return base - float(variant.alpha_obs) * obs_sim - float(variant.generic_lambda) * generic

    raise ValueError(f"unsupported score variant family: {family}")


def _aggregation_name(variant: ExtraMiningScoreVariant) -> str:
    agg = str(variant.aggregation)
    if agg == "topm_mean":
        return f"topm{int(variant.topm)}"
    if agg == "logsumexp":
        return f"lse_tau{_tau_key(float(variant.lse_tau))}"
    if agg == "blend_max_topm":
        return f"blend_rho{_tau_key(float(variant.blend_rho))}_topm{int(variant.topm)}"
    if agg == "residual_topm":
        return f"residual_topm{int(variant.residual_topm)}"
    return "max"


def _hub_prior_tensor(cache: TensorMap, prior_name: str) -> torch.Tensor:
    key = f"hub_prior_{prior_name}"
    if key in cache:
        return _require(cache, key)
    if prior_name == "freq_zscore":
        return _zscore(_require(cache, "hub_prior_freq"))
    if prior_name == "mean_zscore":
        return _zscore(_require(cache, "hub_prior_mean"))
    raise KeyError(f"unknown hub prior: {prior_name}")


def apply_candidate_masks(
    score: torch.Tensor,
    *,
    yprime_mask: torch.Tensor,
    obs_sim: torch.Tensor,
    obs_sim_max: Optional[float],
) -> torch.Tensor:
    """Apply candidate-domain masks without modifying the input tensor."""
    out = score.clone()
    out = out.masked_fill(yprime_mask.to(dtype=torch.bool), float("-inf"))
    if obs_sim_max is not None:
        out = out.masked_fill(obs_sim > float(obs_sim_max), float("-inf"))
    return out


def mmr_select_topk(
    score: torch.Tensor,
    *,
    yprime_mask: torch.Tensor,
    obs_sim: torch.Tensor,
    obs_sim_max: Optional[float],
    text_sim: torch.Tensor,
    k: int,
    top_l: int = 50,
    lambda_text: float = 0.10,
) -> torch.Tensor:
    """GPU batched MMR selection over a compact top-L pool.

    Returns selected vocabulary column indices with shape [B, k].  The greedy loop
    is over small K only; candidate scoring/ranking remains tensorized.
    """
    if score.ndim != 2:
        raise ValueError("score must be [B, C]")
    bsz, vocab_size = int(score.shape[0]), int(score.shape[1])
    k = max(1, min(int(k), vocab_size))
    top_l = max(k, min(int(top_l), vocab_size))
    masked = apply_candidate_masks(score, yprime_mask=yprime_mask, obs_sim=obs_sim, obs_sim_max=obs_sim_max)
    pool_scores, pool_cols = torch.topk(masked, k=top_l, dim=1)
    selected_cols: List[torch.Tensor] = []
    available = torch.isfinite(pool_scores)
    diversity_penalty = torch.zeros_like(pool_scores)
    for step in range(k):
        current = pool_scores - float(lambda_text) * diversity_penalty
        current = current.masked_fill(~available, float("-inf"))
        choice_pos = torch.argmax(current, dim=1)
        choice_cols = torch.gather(pool_cols, 1, choice_pos.reshape(-1, 1)).reshape(-1)
        selected_cols.append(choice_cols)
        available.scatter_(1, choice_pos.reshape(-1, 1), False)
        # Update max text similarity to any already selected candidate.
        sim_to_choice = text_sim[pool_cols.reshape(-1), choice_cols.repeat_interleave(top_l)].reshape(bsz, top_l)
        diversity_penalty = torch.maximum(diversity_penalty, sim_to_choice)
    return torch.stack(selected_cols, dim=1)


def build_default_variants() -> List[ExtraMiningScoreVariant]:
    """First-pass simulator matrix: compact enough to run, broad enough to diagnose."""
    variants: List[ExtraMiningScoreVariant] = [
        ExtraMiningScoreVariant(name="G0_current_max", family="current", aggregation="max"),
    ]
    for lam in (0.05, 0.10, 0.20, 0.40):
        variants.append(ExtraMiningScoreVariant(name=f"G1_hub_freq_l{_short_float(lam)}", family="anti_hub", aggregation="max", hub_prior="freq", hub_lambda=lam))
    for lam in (0.10, 0.20):
        variants.append(ExtraMiningScoreVariant(name=f"G1_hub_mean_l{_short_float(lam)}", family="anti_hub", aggregation="max", hub_prior="mean_zscore", hub_lambda=lam))
    for m in (2, 3, 5):
        variants.append(ExtraMiningScoreVariant(name=f"G2_topm{m}", family="topm", aggregation="topm_mean", topm=m))
    for tau in (0.05, 0.10, 0.20):
        variants.append(ExtraMiningScoreVariant(name=f"G2_lse_tau{_short_float(tau)}", family="logsumexp", aggregation="logsumexp", lse_tau=tau))
    for rho in (0.25, 0.50, 0.75):
        variants.append(ExtraMiningScoreVariant(name=f"G3_blend_r{_short_float(rho)}_hub_l010", family="blend", aggregation="blend_max_topm", topm=3, blend_rho=rho, hub_prior="freq", hub_lambda=0.10))
    variants.extend(
        [
            ExtraMiningScoreVariant(name="G3_topm3_hub_l010", family="topm", aggregation="topm_mean", topm=3, hub_prior="freq", hub_lambda=0.10),
            ExtraMiningScoreVariant(name="G3_topm3_hub_l020", family="topm", aggregation="topm_mean", topm=3, hub_prior="freq", hub_lambda=0.20),
            ExtraMiningScoreVariant(name="G3_lse_tau010_hub_l010", family="logsumexp", aggregation="logsumexp", lse_tau=0.10, hub_prior="freq", hub_lambda=0.10),
            ExtraMiningScoreVariant(name="G4_csls_k10", family="csls", aggregation="blend_max_topm", topm=3, blend_rho=0.50, csls_k=10),
            ExtraMiningScoreVariant(name="G4_csls_k20", family="csls", aggregation="blend_max_topm", topm=3, blend_rho=0.50, csls_k=20),
            ExtraMiningScoreVariant(name="G4_recip_eta010", family="reciprocal", aggregation="blend_max_topm", topm=3, blend_rho=0.50, reciprocal_eta=0.10),
            ExtraMiningScoreVariant(name="G4_residual_topm3", family="residual", aggregation="residual_topm", residual_topm=3, alpha_obs=0.0, obs_sim_max=None),
            ExtraMiningScoreVariant(name="G4_residual_topm3_hub_l010", family="residual", aggregation="residual_topm", residual_topm=3, alpha_obs=0.0, obs_sim_max=None, hub_prior="freq", hub_lambda=0.10),
            ExtraMiningScoreVariant(name="G4_mmr_text_l010", family="mmr", aggregation="blend_max_topm", topm=3, blend_rho=0.50, mmr_lambda_text=0.10, mmr_top_l=50),
            ExtraMiningScoreVariant(name="G4_generic_prior_l010", family="genericity", aggregation="blend_max_topm", topm=3, blend_rho=0.50, generic_lambda=0.10),
        ]
    )
    return variants


def _short_float(value: float) -> str:
    return f"{float(value):.3f}".replace("0.", "").replace(".", "p")


def parse_variant_names(names: Optional[Sequence[str]]) -> List[ExtraMiningScoreVariant]:
    variants = build_default_variants()
    if not names:
        return variants
    by_name = {v.name: v for v in variants}
    missing = [str(name) for name in names if str(name) not in by_name]
    if missing:
        raise ValueError(f"unknown variant names: {missing}; available={sorted(by_name)}")
    return [by_name[str(name)] for name in names]


def _synthetic_self_test() -> Dict[str, Any]:
    torch.manual_seed(7)
    b, c = 4, 9
    base = torch.randn(b, c)
    cache: Dict[str, torch.Tensor] = {
        "agg_max": base,
        "agg_topm2": base - 0.05,
        "agg_topm3": base - 0.10,
        "agg_topm5": base - 0.20,
        "agg_lse_tau0p05": base + 0.01,
        "agg_lse_tau0p1": base + 0.02,
        "agg_lse_tau0p2": base + 0.03,
        "obs_sim": torch.rand(b, c),
        "hub_prior_freq": torch.linspace(0, 1, c),
        "hub_prior_mean": torch.linspace(1, 0, c),
        "genericity_prior": torch.rand(c),
        "support_p95": torch.rand(b, c),
        "density_query_blend_rho0p5_topm3_k10": torch.rand(b),
        "density_class_blend_rho0p5_topm3_k10": torch.rand(c),
        "density_query_blend_rho0p5_topm3_k20": torch.rand(b),
        "density_class_blend_rho0p5_topm3_k20": torch.rand(c),
        "class_to_clip_rank_blend_rho0p5_topm3": torch.ones(b, c),
        "residual_topm3": torch.rand(b, c),
    }
    yprime_mask = torch.zeros(b, c, dtype=torch.bool)
    yprime_mask[:, 0] = True
    text_sim = torch.eye(c)
    out = {}
    for variant in build_default_variants():
        if variant.family == "mmr":
            score = score_variant(cache, ExtraMiningScoreVariant(name="tmp", family="blend", aggregation="blend_max_topm", topm=3, blend_rho=0.50))
            topk = mmr_select_topk(score, yprime_mask=yprime_mask, obs_sim=cache["obs_sim"], obs_sim_max=variant.obs_sim_max, text_sim=text_sim, k=3, top_l=min(5, c), lambda_text=variant.mmr_lambda_text)
            assert tuple(topk.shape) == (b, 3)
            out[variant.name] = list(topk.shape)
        else:
            score = score_variant(cache, variant)
            masked = apply_candidate_masks(score, yprime_mask=yprime_mask, obs_sim=cache["obs_sim"], obs_sim_max=variant.obs_sim_max)
            assert tuple(masked.shape) == (b, c)
            assert torch.all(~torch.isfinite(masked[:, 0]))
            out[variant.name] = list(masked.shape)
    return {"status": "PASS", "variant_count": len(out), "variants": out}


if __name__ == "__main__":
    import json

    print(json.dumps(_synthetic_self_test(), indent=2, ensure_ascii=False))
