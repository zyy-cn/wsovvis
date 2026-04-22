from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.distributed as dist

@dataclass
class UnknownMetricsAccumulator:
    device: torch.device

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum_unknown_pre = torch.zeros((), device=self.device, dtype=torch.float64)
        self.sum_unknown_base = torch.zeros((), device=self.device, dtype=torch.float64)
        self.sum_release_to_yprime = torch.zeros((), device=self.device, dtype=torch.float64)
        self.sum_valid = torch.zeros((), device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def update_prealign(self, valid_count: int) -> None:
        value = torch.tensor(float(max(int(valid_count), 0)), device=self.device, dtype=torch.float64)
        self.sum_unknown_pre += value
        self.sum_valid += value

    @torch.no_grad()
    def update_base(self, g_i: torch.Tensor) -> None:
        if g_i.numel() <= 0:
            return
        g = g_i.detach().to(self.device, torch.float64).reshape(-1)
        self.sum_release_to_yprime += g.sum()
        self.sum_unknown_base += (1.0 - g).sum()
        self.sum_valid += torch.tensor(float(g.numel()), device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def merge_(self) -> None:
        if dist.is_available() and dist.is_initialized():
            for tensor in (self.sum_unknown_pre, self.sum_unknown_base, self.sum_release_to_yprime, self.sum_valid):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def finalize_prealign(self, distributed: bool = False) -> Dict[str, float]:
        if distributed:
            self.merge_()
        denom = max(float(self.sum_valid.item()), 1.0)
        pre = float(self.sum_unknown_pre.item() / denom)
        return {
            'unknown_mass_mean_prealign': pre,
        }

    @torch.no_grad()
    def finalize_base(self, *, prealign_mass_mean: float = 1.0, distributed: bool = False) -> Dict[str, float]:
        if distributed:
            self.merge_()
        denom = max(float(self.sum_valid.item()), 1.0)
        base = float(self.sum_unknown_base.item() / denom)
        release = float(self.sum_release_to_yprime.item() / denom)
        pre = max(float(prealign_mass_mean), 1e-12)
        return {
            'unknown_mass_mean_base': base,
            'unknown_to_yprime_release_rate': release,
            'unknown_retention_rate': float(base / pre),
        }

    @torch.no_grad()
    def finalize(self, distributed: bool = False) -> Dict[str, float]:
        return self.finalize_base(prealign_mass_mean=1.0, distributed=distributed)
