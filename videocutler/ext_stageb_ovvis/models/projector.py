from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectorConfig:
    input_dim: int = 768
    hidden_dim: int = 512
    output_dim: int = 512
    dropout: float = 0.0
    use_layernorm: bool = True


class Projector(nn.Module):
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        if bool(config.use_layernorm):
            layers.append(nn.LayerNorm(int(config.input_dim)))
        layers.append(nn.Linear(int(config.input_dim), int(config.hidden_dim)))
        layers.append(nn.GELU())
        if float(config.dropout) > 0.0:
            layers.append(nn.Dropout(float(config.dropout)))
        layers.append(nn.Linear(int(config.hidden_dim), int(config.output_dim)))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.net(inputs)
        return F.normalize(outputs, p=2.0, dim=-1)

