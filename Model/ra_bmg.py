"""Single-backbone RA-BMG inference wrapper.

The wrapper keeps one A012 backbone and a C1 classifier that has already been
mapped into A012's final hidden coordinates.  The mapped classifier is stored
as persistent buffers: no C1 backbone or representation map participates in
inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_ORDER = ("AD", "CN", "SMCI")


class RABMGA012(nn.Module):
    """CN-first conditional coupling on top of one A012 backbone forward."""

    def __init__(
        self,
        a012_backbone: nn.Module,
        mapped_c1_weight: torch.Tensor,
        mapped_c1_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        classifier = getattr(getattr(a012_backbone, "GCN", None), "classifier", None)
        if not isinstance(classifier, nn.Linear):
            raise TypeError("A012 final classifier must be backbone.GCN.classifier nn.Linear")
        if classifier.out_features != len(CLASS_ORDER):
            raise ValueError("A012 final classifier must use AD/CN/SMCI output order")
        expected_weight = (len(CLASS_ORDER), classifier.in_features)
        if tuple(mapped_c1_weight.shape) != expected_weight:
            raise ValueError(
                f"mapped C1 weight shape {tuple(mapped_c1_weight.shape)} != {expected_weight}"
            )
        if tuple(mapped_c1_bias.shape) != (len(CLASS_ORDER),):
            raise ValueError("mapped C1 bias must have shape [3]")
        if not torch.isfinite(mapped_c1_weight).all() or not torch.isfinite(mapped_c1_bias).all():
            raise ValueError("mapped C1 classifier contains non-finite coefficients")

        self.backbone = a012_backbone
        mapped_c1_weight = mapped_c1_weight.to(
            device=classifier.weight.device, dtype=classifier.weight.dtype
        )
        mapped_c1_bias = mapped_c1_bias.to(
            device=classifier.weight.device, dtype=classifier.weight.dtype
        )
        self.register_buffer(
            "mapped_c1_weight", mapped_c1_weight.detach().clone(), persistent=True
        )
        self.register_buffer(
            "mapped_c1_bias", mapped_c1_bias.detach().clone(), persistent=True
        )
        self._classifier_input: torch.Tensor | None = None
        self._classifier_hook_calls = 0
        self._classifier_hook = classifier.register_forward_pre_hook(
            self._capture_classifier_input
        )
        self.last_probabilities: torch.Tensor | None = None
        self.last_a012_logits: torch.Tensor | None = None
        self.last_mapped_c1_logits: torch.Tensor | None = None

    @property
    def hidden_dim(self) -> int:
        return int(self.mapped_c1_weight.shape[1])

    @property
    def mapped_head_coefficients(self) -> int:
        return int(self.mapped_c1_weight.numel() + self.mapped_c1_bias.numel())

    @property
    def classifier_hook_calls(self) -> int:
        return int(self._classifier_hook_calls)

    def _capture_classifier_input(self, module: nn.Module, inputs: tuple) -> None:
        del module
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Unexpected final-classifier input schema")
        hidden = inputs[0]
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
            raise RuntimeError("Final-classifier hidden representation shape changed")
        self._classifier_input = hidden
        self._classifier_hook_calls += 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._classifier_input = None
        calls_before = self._classifier_hook_calls
        backbone_output = self.backbone(features)
        if not isinstance(backbone_output, tuple) or not backbone_output:
            raise RuntimeError("A012 backbone output schema changed")
        a012_logits = backbone_output[0]
        if self._classifier_hook_calls != calls_before + 1 or self._classifier_input is None:
            raise RuntimeError("A012 final classifier hook must fire exactly once per forward")
        if a012_logits.ndim != 2 or a012_logits.shape[1] != len(CLASS_ORDER):
            raise RuntimeError("A012 logits must have shape [subjects, 3]")

        mapped_c1_logits = F.linear(
            self._classifier_input, self.mapped_c1_weight, self.mapped_c1_bias
        )
        # Couple in float64 so the required probability-simplex and conditional
        # log-odds invariants are tighter than their 1e-7/1e-6 gates.
        q_cn = torch.softmax(mapped_c1_logits.to(torch.float64), dim=-1)[:, 1]
        q_ad_given_non_cn = torch.softmax(
            a012_logits[:, (0, 2)].to(torch.float64), dim=-1
        )[:, 0]
        one_minus_cn = 1.0 - q_cn
        probabilities = torch.stack(
            (
                one_minus_cn * q_ad_given_non_cn,
                q_cn,
                one_minus_cn * (1.0 - q_ad_given_non_cn),
            ),
            dim=-1,
        )
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("RA-BMG probabilities are non-finite")

        self.last_a012_logits = a012_logits
        self.last_mapped_c1_logits = mapped_c1_logits
        self.last_probabilities = probabilities
        return torch.log(probabilities.clamp_min(1e-8))
