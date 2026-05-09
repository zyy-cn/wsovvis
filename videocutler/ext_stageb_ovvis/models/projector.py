from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectorConfig:
    """Historical filename, current authority: text-side projector.

    Under the current G7 authority this module maps CLIP text features into the
    frozen DINO feature space. It must not be interpreted as a visual-side
    projector.
    """

    input_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 768
    dropout: float = 0.0
    use_layernorm: bool = True
    projector_type: str = "mlp"


class Projector(nn.Module):
    """Text-side projector with legacy layouts plus a side-path semi-orthogonal layout.

    Backward compatibility:
      * The default remains ProjectorConfig(projector_type="mlp").
      * Historical "mlp", "linear", and "linear_ln" state dicts keep the exact
        same parameter names and forward semantics.
      * The new "semi_orthogonal_linear" type is opt-in and intended for A8
        text-bank / topology ablations only.
    """

    SUPPORTED_TYPES = {"mlp", "linear", "linear_ln", "semi_orthogonal_linear"}

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        self.projector_type = str(getattr(config, "projector_type", "mlp") or "mlp").strip().lower()
        if self.projector_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"unsupported projector_type={self.projector_type!r}; "
                f"expected one of {sorted(self.SUPPORTED_TYPES)}"
            )
        if int(config.input_dim) <= 0 or int(config.output_dim) <= 0:
            raise ValueError(f"invalid projector dimensions: input_dim={config.input_dim}, output_dim={config.output_dim}")

        self.net: nn.Module
        if self.projector_type == "semi_orthogonal_linear":
            # Free parameter A; forward computes a semi-orthogonal effective W.
            # Shape follows nn.Linear convention: W is [output_dim, input_dim].
            self.orth_raw = nn.Parameter(torch.empty(int(config.output_dim), int(config.input_dim)))
            self.net = nn.Identity()
        else:
            layers: list[nn.Module] = []
            if self.projector_type == "mlp":
                if bool(config.use_layernorm):
                    layers.append(nn.LayerNorm(int(config.input_dim)))
                layers.append(nn.Linear(int(config.input_dim), int(config.hidden_dim)))
                layers.append(nn.GELU())
                if float(config.dropout) > 0.0:
                    layers.append(nn.Dropout(float(config.dropout)))
                layers.append(nn.Linear(int(config.hidden_dim), int(config.output_dim)))
            elif self.projector_type == "linear_ln":
                layers.append(nn.LayerNorm(int(config.input_dim)))
                layers.append(nn.Linear(int(config.input_dim), int(config.output_dim)))
            else:
                layers.append(nn.Linear(int(config.input_dim), int(config.output_dim)))
            self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.projector_type == "semi_orthogonal_linear":
            # Do not QR-project here: the module is constructed on CPU before
            # most training scripts move it to CUDA.  A8 hard-orth training calls
            # project_semi_orthogonal_() after .to(device) and after optimizer
            # steps, avoiding slow CPU QR for 4096-d Llama3 banks.
            nn.init.normal_(self.orth_raw, mean=0.0, std=1.0 / max(float(int(self.config.input_dim)) ** 0.5, 1.0))
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _qr_with_stable_sign(matrix: torch.Tensor) -> torch.Tensor:
        q, r = torch.linalg.qr(matrix, mode="reduced")
        diag = torch.diagonal(r, 0)
        sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
        return q * sign.unsqueeze(0)

    @torch.no_grad()
    def project_semi_orthogonal_(self) -> None:
        """Project the free semi-orthogonal parameter back to the Stiefel set.

        This is intentionally called after optimizer steps for the hard-constraint
        ablation.  Forward then uses orth_raw directly, avoiding a large QR on
        every clip/iteration, which is important for Llama3 4096-d text banks.
        """
        if self.projector_type != "semi_orthogonal_linear":
            return
        output_dim = int(self.config.output_dim)
        input_dim = int(self.config.input_dim)
        a = self.orth_raw.detach()
        if output_dim >= input_dim:
            q = self._qr_with_stable_sign(a)
            self.orth_raw.copy_(q)
        else:
            q = self._qr_with_stable_sign(a.t())
            self.orth_raw.copy_(q.t())

    def effective_linear_weight(self) -> torch.Tensor:
        """Return the effective linear map W with shape [output_dim, input_dim]."""
        if self.projector_type == "semi_orthogonal_linear":
            return self.orth_raw
        if self.projector_type == "linear":
            return self.net[0].weight  # type: ignore[index]
        if self.projector_type == "linear_ln":
            return self.net[1].weight  # type: ignore[index]
        raise RuntimeError("effective_linear_weight is only defined for linear projector layouts")

    def orthogonality_penalty(self) -> torch.Tensor:
        """Squared Frobenius orthogonality error for linear layouts.

        For output_dim >= input_dim, use ||W^T W - I||_F^2.
        For output_dim < input_dim, use ||W W^T - I||_F^2.
        """
        w = self.effective_linear_weight()
        output_dim, input_dim = int(w.shape[0]), int(w.shape[1])
        if output_dim >= input_dim:
            gram = w.t().matmul(w)
            eye = torch.eye(input_dim, device=w.device, dtype=w.dtype)
            return torch.mean((gram - eye) ** 2)
        gram = w.matmul(w.t())
        eye = torch.eye(output_dim, device=w.device, dtype=w.dtype)
        return torch.mean((gram - eye) ** 2)

    def orthogonality_report(self) -> Dict[str, float | str | bool]:
        if self.projector_type not in {"linear", "linear_ln", "semi_orthogonal_linear"}:
            return {
                "orthogonality_applicable": False,
                "projector_type": str(self.projector_type),
                "orthogonality_target": "not_applicable",
                "orthogonality_error_mean_sq": 0.0,
            }
        with torch.no_grad():
            w = self.effective_linear_weight().detach()
            output_dim, input_dim = int(w.shape[0]), int(w.shape[1])
            if output_dim >= input_dim:
                gram = w.t().matmul(w)
                eye = torch.eye(input_dim, device=w.device, dtype=w.dtype)
                target = "W^T W = I"
            else:
                gram = w.matmul(w.t())
                eye = torch.eye(output_dim, device=w.device, dtype=w.dtype)
                target = "W W^T = I"
            err = (gram - eye).float()
            return {
                "orthogonality_applicable": True,
                "projector_type": str(self.projector_type),
                "orthogonality_target": target,
                "orthogonality_error_mean_sq": float(torch.mean(err ** 2).cpu().item()),
                "orthogonality_error_fro": float(torch.linalg.vector_norm(err).cpu().item()),
                "input_dim": int(input_dim),
                "output_dim": int(output_dim),
                "uses_bias": bool(self.projector_type in {"linear", "linear_ln"}),
            }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.projector_type == "semi_orthogonal_linear":
            outputs = F.linear(inputs, self.effective_linear_weight(), bias=None)
        else:
            outputs = self.net(inputs)
        return F.normalize(outputs, p=2.0, dim=-1)
