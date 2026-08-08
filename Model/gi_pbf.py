"""Gradient-isolated pairwise boundary correction on top of formal C1.

GI-PBF keeps the complete C1 forward path intact and learns two independent
logit corrections for the AD--sMCI and CN--sMCI boundaries.  Every tensor that
enters the correction module from C1 is detached, while the correction output
itself remains differentiable.  Consequently, the historical C1 loss and the
pairwise correction loss can update disjoint parameter sets without hooks or
manual gradient surgery.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
import torch.nn as nn

from .cme_dual_branch import CMEDualBranchModel


C1_EXPECTED_PARAMETERS = 862_971
GI_PBF_ADDED_PARAMETERS_H96_R8 = 3_098
GI_PBF_EXPECTED_PARAMETERS_H96_R8 = 866_069


def _capture_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Capture every RNG stream that module initialization could consume."""

    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return cpu_state, cuda_states


def _restore_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    """Restore RNG streams after constructing the correction-only module."""

    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _canonical_class_name(value: Any) -> str:
    compact = "".join(character for character in str(value).strip().lower() if character.isalnum())
    aliases = {
        "ad": "AD",
        "alzheimersdisease": "AD",
        "cn": "CN",
        "cognitivelynormal": "CN",
        "smci": "sMCI",
        "stablemci": "sMCI",
        "stablemildcognitiveimpairment": "sMCI",
    }
    if compact not in aliases:
        raise ValueError(f"GI-PBF does not recognize class name {value!r}")
    return aliases[compact]


def _resolve_class_indices(class_names: tuple[Any, ...]) -> dict[str, int]:
    if len(class_names) != 3:
        raise ValueError(f"GI-PBF requires exactly three classes, got {class_names!r}")
    canonical = tuple(_canonical_class_name(value) for value in class_names)
    if len(set(canonical)) != 3 or set(canonical) != {"AD", "CN", "sMCI"}:
        raise ValueError(f"GI-PBF requires AD, CN, and sMCI exactly once, got {class_names!r}")
    return {name: canonical.index(name) for name in ("AD", "CN", "sMCI")}


class PairwiseBoundaryCorrection(nn.Module):
    """Two zero-initialized, uncertainty-weighted zero-sum corrections."""

    def __init__(
        self,
        hidden_size: int,
        class_indices: dict[str, int],
        rank: int = 8,
        cap_coefficient: float = 0.10,
    ):
        super().__init__()
        if int(rank) != 8:
            raise ValueError("GI-PBF v1 requires rank=8")
        if abs(float(cap_coefficient) - 0.10) > 1e-12:
            raise ValueError("GI-PBF v1 requires cap_coefficient=0.10")
        if set(class_indices) != {"AD", "CN", "sMCI"}:
            raise ValueError(f"Invalid GI-PBF class mapping: {class_indices!r}")
        if set(class_indices.values()) != {0, 1, 2}:
            raise ValueError(f"GI-PBF class indices must be a permutation of 0,1,2: {class_indices!r}")

        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.cap_coefficient = float(cap_coefficient)
        self.class_indices = dict(class_indices)

        self.branch_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.input_projection = nn.Linear(4 * self.hidden_size, self.rank)
        self.activation = nn.GELU()
        self.output = nn.Linear(self.rank, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        inverse_root_two = 1.0 / math.sqrt(2.0)
        direction_ad_smci = torch.zeros(3)
        direction_ad_smci[self.class_indices["AD"]] = inverse_root_two
        direction_ad_smci[self.class_indices["sMCI"]] = -inverse_root_two
        direction_cn_smci = torch.zeros(3)
        direction_cn_smci[self.class_indices["CN"]] = inverse_root_two
        direction_cn_smci[self.class_indices["sMCI"]] = -inverse_root_two
        self.register_buffer("direction_ad_smci", direction_ad_smci)
        self.register_buffer("direction_cn_smci", direction_cn_smci)

    def forward(
        self,
        category_embedding: torch.Tensor,
        global_embedding: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if category_embedding.shape != global_embedding.shape:
            raise ValueError("Category and Global representations must have matching shapes")
        if category_embedding.ndim != 2 or category_embedding.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected branch representations [N,{self.hidden_size}], got "
                f"{tuple(category_embedding.shape)}"
            )
        if base_logits.ndim != 2 or tuple(base_logits.shape) != (
            category_embedding.size(0),
            3,
        ):
            raise ValueError(f"Expected base logits [N,3], got {tuple(base_logits.shape)}")

        # The only bridge from C1 into GI-PBF is deliberately gradient-free.
        detached_y = category_embedding.detach()
        detached_g = global_embedding.detach()
        detached_base_logits = base_logits.detach()
        normalized_y = self.branch_norm(detached_y)
        normalized_g = self.branch_norm(detached_g)
        correction_input = torch.cat(
            (
                normalized_y,
                normalized_g,
                torch.abs(normalized_y - normalized_g),
                normalized_y * normalized_g,
            ),
            dim=-1,
        )
        hidden = self.activation(self.input_projection(correction_input))
        raw_delta = self.output(hidden)

        base_probability = torch.softmax(detached_base_logits, dim=-1)
        ad_index = self.class_indices["AD"]
        cn_index = self.class_indices["CN"]
        smci_index = self.class_indices["sMCI"]
        p_ad = base_probability[:, ad_index]
        p_cn = base_probability[:, cn_index]
        p_smci = base_probability[:, smci_index]
        uncertainty_ad_smci = (
            4.0 * p_ad * p_smci / (p_ad + p_smci + 1e-8)
        ).detach()
        uncertainty_cn_smci = (
            4.0 * p_cn * p_smci / (p_cn + p_smci + 1e-8)
        ).detach()

        centered_base_logits = detached_base_logits - detached_base_logits.mean(
            dim=-1, keepdim=True
        )
        logit_scale = torch.clamp(
            torch.linalg.vector_norm(centered_base_logits, dim=-1), min=1.0
        ).detach()
        delta_cap = (self.cap_coefficient * logit_scale).detach()
        delta_ad_smci = (
            uncertainty_ad_smci * delta_cap * torch.tanh(raw_delta[:, 0])
        )
        delta_cn_smci = (
            uncertainty_cn_smci * delta_cap * torch.tanh(raw_delta[:, 1])
        )

        correction_vector = (
            delta_ad_smci.unsqueeze(-1) * self.direction_ad_smci.unsqueeze(0)
            + delta_cn_smci.unsqueeze(-1) * self.direction_cn_smci.unsqueeze(0)
        )
        corrected_logits = detached_base_logits + correction_vector

        inverse_root_two = self.direction_ad_smci[ad_index]
        pair_logits_ad_smci = torch.stack(
            (
                detached_base_logits[:, ad_index]
                + inverse_root_two * delta_ad_smci,
                detached_base_logits[:, smci_index]
                - inverse_root_two * delta_ad_smci,
            ),
            dim=-1,
        )
        pair_logits_cn_smci = torch.stack(
            (
                detached_base_logits[:, cn_index]
                + inverse_root_two * delta_cn_smci,
                detached_base_logits[:, smci_index]
                - inverse_root_two * delta_cn_smci,
            ),
            dim=-1,
        )

        return corrected_logits, {
            "normalized_Y": normalized_y,
            "normalized_G": normalized_g,
            "correction_input": correction_input,
            "correction_hidden": hidden,
            "raw_delta": raw_delta,
            "base_probability_detached": base_probability,
            "uncertainty_ad_smci": uncertainty_ad_smci,
            "uncertainty_cn_smci": uncertainty_cn_smci,
            "centered_base_logits_detached": centered_base_logits,
            "logit_scale": logit_scale,
            "delta_cap": delta_cap,
            "delta_ad_smci": delta_ad_smci,
            "delta_cn_smci": delta_cn_smci,
            "direction_ad_smci": self.direction_ad_smci,
            "direction_cn_smci": self.direction_cn_smci,
            "correction_vector": correction_vector,
            "pair_logits_ad_smci": pair_logits_ad_smci,
            "pair_logits_cn_smci": pair_logits_cn_smci,
        }


class GIPBFModel(CMEDualBranchModel):
    """Formal C1 plus a gradient-isolated pairwise boundary correction head."""

    def __init__(
        self,
        dataset_dict: dict,
        *args: Any,
        class_names: tuple[Any, ...] | list[Any] | None = None,
        pair_rank: int = 8,
        cap_coefficient: float = 0.10,
        cme_arm: str = "c1",
        **kwargs: Any,
    ):
        if str(cme_arm).lower() != "c1":
            raise ValueError("GI-PBF v1 must use the formal C1 arm")

        # Construct C1 completely before adding anything.  Its parameter names,
        # initialization, and forward graph are inherited rather than copied.
        super().__init__(dataset_dict, *args, cme_arm="c1", **kwargs)

        dataset_class_names = tuple(dataset_dict.get("Class_Names", ()))
        selected_class_names = (
            dataset_class_names if class_names is None else tuple(class_names)
        )
        if not selected_class_names:
            raise ValueError("GI-PBF requires class names from the loaded dataset")
        if dataset_class_names:
            dataset_canonical = tuple(
                _canonical_class_name(value) for value in dataset_class_names
            )
            selected_canonical = tuple(
                _canonical_class_name(value) for value in selected_class_names
            )
            if selected_canonical != dataset_canonical:
                raise ValueError(
                    "Explicit class_names do not match DATASET_Dict['Class_Names']"
                )
        self.class_names = selected_class_names
        self.class_indices = _resolve_class_indices(self.class_names)

        # Restoring the post-C1 RNG state makes construction observationally
        # identical to constructing C1 alone, including the next dropout/noise
        # draw used by the first training forward.
        rng_state = _capture_rng_state()
        try:
            self.gi_pbf = PairwiseBoundaryCorrection(
                self.Hidden_size,
                self.class_indices,
                rank=pair_rank,
                cap_coefficient=cap_coefficient,
            )
        finally:
            _restore_rng_state(rng_state)

        self.last_base_logits: torch.Tensor | None = None
        self.last_corrected_logits: torch.Tensor | None = None
        self.last_gi_pbf_intermediates: dict[str, torch.Tensor] = {}

    def c1_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate only formal-C1 parameters for the historical optimizer."""

        for name, parameter in self.named_parameters():
            if not name.startswith("gi_pbf."):
                yield parameter

    def correction_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate only GI-PBF parameters for the correction optimizer."""

        yield from self.gi_pbf.parameters()

    def forward(
        self,
        x_raw: torch.Tensor,
        return_intermediates: bool = False,
        counterfactual_modal_index: int | None = None,
        counterfactual_sample_mask: torch.Tensor | None = None,
    ):
        base_logits, label_embeddings, auxiliary_outputs, c1_intermediates = (
            super().forward(
                x_raw,
                return_intermediates=True,
                counterfactual_modal_index=counterfactual_modal_index,
                counterfactual_sample_mask=counterfactual_sample_mask,
            )
        )
        corrected_logits, correction_intermediates = self.gi_pbf(
            c1_intermediates["Y"], c1_intermediates["G"], base_logits
        )
        self.last_base_logits = base_logits.detach()
        self.last_corrected_logits = corrected_logits.detach()
        self.last_gi_pbf_intermediates = {
            key: value.detach() for key, value in correction_intermediates.items()
        }

        if return_intermediates:
            intermediates = {
                **c1_intermediates,
                "base_logits": base_logits,
                "corrected_logits": corrected_logits,
                **correction_intermediates,
            }
            return (
                base_logits,
                corrected_logits,
                label_embeddings,
                auxiliary_outputs,
                intermediates,
            )
        # The ordinary inference/training-compatible three-tuple deliberately
        # exposes corrected logits while retaining historical auxiliary outputs.
        return corrected_logits, label_embeddings, auxiliary_outputs


GI_PBFModel = GIPBFModel


__all__ = [
    "C1_EXPECTED_PARAMETERS",
    "GI_PBF_ADDED_PARAMETERS_H96_R8",
    "GI_PBF_EXPECTED_PARAMETERS_H96_R8",
    "PairwiseBoundaryCorrection",
    "GIPBFModel",
    "GI_PBFModel",
]
