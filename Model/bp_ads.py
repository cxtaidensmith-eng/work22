"""Boundary-protected AD/sMCI specialist for a frozen C1 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryProtectedADSMCISpecialist(nn.Module):
    """One frozen C1 model plus a tiny AD-vs-sMCI log-odds specialist.

    The specialist operates on the historical *scoring* logits (raw logits
    after the locked logit adjustment).  Therefore a zero delta recomposes the
    exact C1 softmax.  Subjects whose C1 argmax is CN are hard-protected and
    receive the original C1 probability without any arithmetic replacement.
    """

    AD = 0
    CN = 1
    SMCI = 2

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        label_weight: torch.Tensor,
        logit_adjust_tau: float,
        expert_hidden: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.expert_hidden = int(expert_hidden)
        self.logit_adjust_tau = float(logit_adjust_tau)
        if self.expert_hidden != 8:
            raise ValueError("BP-ADS v1 fixes expert_hidden=8")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.y_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.g_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.expert = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.expert_hidden),
            nn.GELU(),
            nn.Linear(self.expert_hidden, 1),
        )
        nn.init.zeros_(self.expert[-1].weight)
        nn.init.zeros_(self.expert[-1].bias)
        self.register_buffer(
            "label_weight",
            label_weight.detach().clone().to(dtype=torch.float32),
            persistent=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The backbone must never enable modal noise or dropout.
        self.backbone.eval()
        return self

    @property
    def expert_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.expert.parameters())

    def scoring_logits(self, raw_logits: torch.Tensor) -> torch.Tensor:
        prior = self.label_weight.to(raw_logits).clamp_min(1e-8)
        return raw_logits - self.logit_adjust_tau * prior.log().view(1, -1)

    @torch.no_grad()
    def frozen_cache(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        raw_logits, _, _, intermediates = self.backbone(
            features,
            return_intermediates=True,
        )
        y = self.y_norm(intermediates["Y"])
        g = self.g_norm(intermediates["G"])
        specialist_input = torch.cat((y, g, (y - g).abs(), y * g), dim=-1)
        z0 = self.scoring_logits(raw_logits)
        p0 = torch.softmax(z0, dim=-1)
        return {
            "specialist_input": specialist_input.detach(),
            "z0": z0.detach(),
            "p0": p0.detach(),
            "protected_cn": (p0.argmax(dim=-1) == self.CN).detach(),
            "raw_logits": raw_logits.detach(),
            "Y": intermediates["Y"].detach(),
            "G": intermediates["G"].detach(),
        }

    def probability_from_cache(
        self,
        cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = self.expert(cache["specialist_input"]).squeeze(-1)
        z0 = cache["z0"]
        p0 = cache["p0"]
        base_boundary_logit = z0[:, self.AD] - z0[:, self.SMCI]
        boundary_logit = base_boundary_logit + delta
        q_ad_given_non_cn = torch.sigmoid(boundary_logit)
        q0_ad_given_non_cn = torch.sigmoid(base_boundary_logit)
        q_cn = p0[:, self.CN]
        zero_candidate = torch.stack(
            (
                (1.0 - q_cn) * q0_ad_given_non_cn,
                q_cn,
                (1.0 - q_cn) * (1.0 - q0_ad_given_non_cn),
            ),
            dim=-1,
        )
        shifted_candidate = torch.stack(
            (
                (1.0 - q_cn) * q_ad_given_non_cn,
                q_cn,
                (1.0 - q_cn) * (1.0 - q_ad_given_non_cn),
            ),
            dim=-1,
        )
        # Residual form makes delta=0 exactly p0 (bitwise), while preserving
        # the shifted candidate's gradient at zero initialization.
        candidate = p0 + (shifted_candidate - zero_candidate)
        probability = torch.where(
            cache["protected_cn"].unsqueeze(-1),
            p0,
            candidate,
        )
        return probability, {
            "delta": delta,
            "boundary_logit": boundary_logit,
            "q_ad_given_non_cn": q_ad_given_non_cn,
            "protected_cn": cache["protected_cn"],
            "p0": p0,
            "candidate": candidate,
        }

    def forward(self, features: torch.Tensor):
        return self.probability_from_cache(self.frozen_cache(features))
