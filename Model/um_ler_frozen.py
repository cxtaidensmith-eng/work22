from __future__ import annotations

import math

import torch

from Model.um_ler import UncertaintyGuidedLocalEvidenceRefinement


class FrozenUncertaintyGuidedLocalEvidenceRefinement(
    UncertaintyGuidedLocalEvidenceRefinement
):
    """UM-LER driven exclusively by cached Original outputs and train quantiles."""

    @staticmethod
    def refine_from_train_quantiles(
        original_probability: torch.Tensor,
        local_probability: torch.Tensor,
        train_mask: torch.Tensor,
        *,
        uncertain_quantile: float,
        neighbor_confidence_quantile: float,
        gate_cap: float,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 < uncertain_quantile < 1.0:
            raise ValueError("uncertain_quantile must be in (0,1)")
        if not 0.0 < neighbor_confidence_quantile < 1.0:
            raise ValueError("neighbor_confidence_quantile must be in (0,1)")
        if not 0.0 <= gate_cap <= 1.0:
            raise ValueError("gate_cap must be in [0,1]")
        if train_mask.dtype is not torch.bool:
            raise ValueError("train_mask must be boolean")

        probability = original_probability.detach()
        top_two = torch.topk(probability, k=2, dim=-1).values
        margin = top_two[:, 0] - top_two[:, 1]
        margin_threshold = torch.quantile(
            margin[train_mask].detach(), float(uncertain_quantile)
        )
        uncertainty = torch.clamp(
            (margin_threshold - margin) / (margin_threshold + 1e-8),
            min=0.0,
            max=1.0,
        )

        local_safe = local_probability.clamp_min(1e-12)
        entropy = -(local_safe * local_safe.log()).sum(dim=-1)
        reliability = torch.clamp(
            1.0 - entropy / math.log(local_probability.size(-1)),
            min=0.0,
            max=1.0,
        )
        local_max = local_probability.max(dim=-1).values
        confidence_threshold = torch.quantile(
            local_max[train_mask].detach(),
            float(neighbor_confidence_quantile),
        )
        eligible = (margin <= margin_threshold) & (
            local_max >= confidence_threshold
        )
        gate = eligible.to(dtype=probability.dtype) * torch.clamp(
            uncertainty * reliability,
            min=0.0,
            max=float(gate_cap),
        )
        final_probability = (
            (1.0 - gate.unsqueeze(-1)) * probability
            + gate.unsqueeze(-1) * local_probability
        )
        final_probability = final_probability / final_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        return {
            "p": probability,
            "q": local_probability,
            "p_final": final_probability,
            "margin": margin,
            "margin_threshold": margin_threshold,
            "uncertainty": uncertainty,
            "local_entropy": entropy,
            "local_reliability": reliability,
            "local_max": local_max,
            "neighbor_confidence_threshold": confidence_threshold,
            "eligible": eligible,
            "gate": gate,
        }

    def forward(
        self,
        modal_tokens: torch.Tensor,
        original_probability: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        *,
        top_k: int = 8,
        temperature: float = 0.2,
        uncertain_quantile: float = 0.25,
        neighbor_confidence_quantile: float = 0.60,
        gate_cap: float = 0.30,
    ) -> dict[str, torch.Tensor]:
        retrieval, modal_reliability = self.encode(modal_tokens)
        local = self.local_probabilities(
            retrieval,
            labels,
            train_mask,
            top_k=top_k,
            temperature=temperature,
            class_count=original_probability.size(-1),
        )
        refined = self.refine_from_train_quantiles(
            original_probability,
            local["q"],
            train_mask,
            uncertain_quantile=uncertain_quantile,
            neighbor_confidence_quantile=neighbor_confidence_quantile,
            gate_cap=gate_cap,
        )
        return {
            **local,
            **refined,
            "retrieval": retrieval,
            "modal_reliability": modal_reliability,
        }
