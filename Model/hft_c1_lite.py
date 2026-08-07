"""Hierarchical Fine-Grained Modality Token Residual v1.

This module preserves the complete C1 classification path and adds one
fine-grained residual branch for each non-COG modality.  Every branch tokenizes
the modality's scalar features, pools them with the detached C1 summary token,
and injects a bounded residual into the Category path only.  Its final
projection is zero initialized, so an enabled HFT model is exactly C1 at
initialization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import HeterGraph_Model_Kmeans


class PrivateResidualAdapter(nn.Module):
    """The unchanged rank-eight private residual used by C1."""

    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(token)))


class FineGrainedModalityResidual(nn.Module):
    """One lightweight scalar-feature token branch.

    Cross-attention is deliberately parameter-free after token construction:
    the specification authorizes one shared ``Linear(1, H)`` per modality,
    learnable feature identities, and one zero-initialized output projection,
    but no additional Q/K/V stack.
    """

    def __init__(
        self,
        feature_indices: list[int] | tuple[int, ...],
        hidden_size: int,
    ):
        super().__init__()
        indices = torch.as_tensor(feature_indices, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("A fine-grained modality requires at least one feature")
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError("Fine-grained modality feature indices must be unique")

        self.hidden_size = int(hidden_size)
        self.feature_count = int(indices.numel())
        self.register_buffer("feature_indices", indices)
        self.value_projection = nn.Linear(1, self.hidden_size)
        self.feature_id_embedding = nn.Embedding(
            self.feature_count, self.hidden_size
        )
        nn.init.normal_(self.feature_id_embedding.weight, std=0.02)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        preprocessed_features: torch.Tensor,
        c1_summary: torch.Tensor,
        modal_gate: torch.Tensor,
        *,
        cap: float,
        eps: float,
        attention_temperature: float,
    ) -> dict[str, torch.Tensor]:
        if preprocessed_features.ndim != 2:
            raise ValueError(
                "Expected preprocessed features with shape [N,F], got "
                f"{tuple(preprocessed_features.shape)}"
            )
        if tuple(c1_summary.shape) != (
            preprocessed_features.size(0),
            self.hidden_size,
        ):
            raise ValueError(
                "C1 summary shape mismatch: "
                f"{tuple(c1_summary.shape)}"
            )

        scalar_features = preprocessed_features.index_select(
            1, self.feature_indices
        ).unsqueeze(-1)
        feature_tokens = self.value_projection(scalar_features)
        feature_tokens = feature_tokens + self.feature_id_embedding.weight.unsqueeze(0)
        # The fine-grained branch observes the exact historical modal gate in
        # the forward pass instead of creating a parallel ungated shortcut.
        feature_tokens = feature_tokens * modal_gate.to(feature_tokens).reshape(1, 1, 1)

        # Parameter-free LayerNorm followed by stop-gradient implements the
        # preregistered stop_gradient(layer_norm(c1_summary)) query exactly.
        query = F.layer_norm(c1_summary, (self.hidden_size,)).detach()
        attention_logits = torch.einsum(
            "nd,nfd->nf", query, feature_tokens
        ) / (math.sqrt(self.hidden_size) * float(attention_temperature))
        attention_weights = torch.softmax(attention_logits, dim=-1)
        local_evidence = torch.einsum(
            "nf,nfd->nd", attention_weights, feature_tokens
        )
        raw_delta = self.output_projection(local_evidence)

        # Norms are routing statistics only.  Detaching them prevents the cap
        # itself from introducing a second gradient path into the C1 summary.
        shared_norm = c1_summary.detach().norm(dim=-1, keepdim=True)
        raw_delta_norm = raw_delta.detach().norm(dim=-1, keepdim=True)
        cap_scale = torch.clamp(
            float(cap) * shared_norm / (raw_delta_norm + float(eps)),
            max=1.0,
        )
        delta = raw_delta * cap_scale
        actual_ratio = (
            delta.detach().norm(dim=-1, keepdim=True)
            / (shared_norm + float(eps))
        )
        cap_saturated = cap_scale < (1.0 - 1e-7)

        return {
            "feature_tokens": feature_tokens,
            "query": query,
            "attention_weights": attention_weights,
            "local_evidence": local_evidence,
            "raw_delta": raw_delta,
            "delta": delta,
            "shared_norm": shared_norm,
            "raw_delta_norm": raw_delta_norm,
            "cap_scale": cap_scale,
            "ratio": actual_ratio,
            "cap_saturated": cap_saturated,
        }


class HFTC1LiteModel(HeterGraph_Model_Kmeans):
    """Exact C1 plus bounded fine-grained residuals for five non-COG modalities."""

    HFT_MODALITIES = ("MRI", "PET", "CSF", "Risk", "ROI")
    ALL_CANONICAL_MODALITIES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
    ALLOWED_CAPS = (0.15, 0.25)
    ALLOWED_ATTENTION_TEMPERATURES = (1.0, 0.5)
    EPS = 1e-8

    def __init__(
        self,
        *args,
        cap: float = 0.15,
        attention_temperature: float = 1.0,
        adapter_rank: int = 8,
        hft_enabled: bool = True,
        **kwargs,
    ):
        # Construct every historical Original module first, then C1 adapters,
        # and only then HFT.  New parameters therefore cannot perturb seeded
        # initialization of any C1 parameter.
        super().__init__(*args, **kwargs)
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode == "none"
        ):
            raise ValueError("HFT-C1-Lite requires the locked C1 configuration")
        if self._modal_num != 6:
            raise ValueError(
                f"HFT-C1-Lite requires six modalities, got {self._modal_num}"
            )

        self.adapter_rank = int(adapter_rank)
        if self.adapter_rank != 8:
            raise ValueError("HFT-C1-Lite v1 requires C1 adapter_rank=8")
        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )

        self.cap = float(cap)
        if self.cap not in self.ALLOWED_CAPS:
            raise ValueError(
                "HFT-C1-Lite permits only preregistered cap=0.15 or 0.25"
            )
        self.attention_temperature = float(attention_temperature)
        if self.attention_temperature not in self.ALLOWED_ATTENTION_TEMPERATURES:
            raise ValueError(
                "HFT-C1-Lite permits only preregistered attention temperature "
                "1.0 or 0.5"
            )
        self.hft_enabled = bool(hft_enabled)
        self.eps = float(self.EPS)

        raw_modal_names = [
            str(name).strip() for name in self.DATASET_Dict["Modal_Name"]
        ]
        canonical_names = [self._canonical_modal_name(name) for name in raw_modal_names]
        if len(set(canonical_names)) != len(canonical_names):
            raise ValueError(
                f"Ambiguous canonical modality mapping: {raw_modal_names} -> "
                f"{canonical_names}"
            )
        if set(canonical_names) != set(self.ALL_CANONICAL_MODALITIES):
            raise ValueError(
                f"Unexpected modalities: {raw_modal_names} -> {canonical_names}"
            )
        self.raw_modal_names = tuple(raw_modal_names)
        self.canonical_modal_names = tuple(canonical_names)
        self.modal_index_by_name = {
            name: canonical_names.index(name)
            for name in self.ALL_CANONICAL_MODALITIES
        }
        self.hft_feature_indices_by_modality = {
            name: tuple(
                int(index)
                for index in self._modal_index[self.modal_index_by_name[name]]
            )
            for name in self.HFT_MODALITIES
        }

        self.hft_branches = nn.ModuleDict()
        if self.hft_enabled:
            for modality in self.HFT_MODALITIES:
                self.hft_branches[modality] = FineGrainedModalityResidual(
                    self.hft_feature_indices_by_modality[modality],
                    self.Hidden_size,
                )

    @staticmethod
    def _canonical_modal_name(raw_name: str) -> str:
        name = str(raw_name).strip().upper()
        if name == "MRI" or "UCSFFSX" in name:
            return "MRI"
        if name == "PET" or "UCBERKELEYAV45" in name:
            return "PET"
        if name == "CSF" or "UPENNBIOMK" in name:
            return "CSF"
        if name == "PHS" or "RISK" in name or "GENETIC" in name:
            return "Risk"
        if "COG" in name:
            return "COG"
        if "ROI" in name:
            return "ROI"
        raise ValueError(f"Cannot map dataset modality name: {raw_name!r}")

    def hft_parameter_count(self) -> int:
        """Number of parameters added by the five HFT branches."""

        return sum(parameter.numel() for parameter in self.hft_branches.parameters())

    def c1_parameter_count(self) -> int:
        """Number of parameters in the unchanged C1 backbone."""

        return sum(parameter.numel() for parameter in self.parameters()) - self.hft_parameter_count()

    def inference_parameter_count(self) -> int:
        """All HFT parameters participate in enabled-model inference."""

        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
    ):
        self.last_modal_tokens = None
        feature_modal_output = self.Feature_Modal(X_raw)
        X = feature_modal_output
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (
                per_feature_noise_std * self.noise_scale
            ).view(1, -1)

        modal_gate = torch.sigmoid(self.modal_gate_logit)
        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)

        # Complete historical C1 summary-token path.
        H = self.modal_token_encoder(X)
        H = H * modal_gate.view(1, -1, 1)
        modal_tokens_pre_transformer = H
        for block in self.shared_transformer:
            H = block(H)
        modal_tokens_post_transformer = H
        self.last_modal_tokens = H.detach()

        private_residuals = torch.stack(
            [
                adapter(H[:, modal_index])
                for modal_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )

        hft_outputs: dict[str, dict[str, torch.Tensor]] = {}
        if self.hft_enabled:
            for modality, branch in self.hft_branches.items():
                modal_index = self.modal_index_by_name[modality]
                hft_outputs[modality] = branch(
                    X,
                    H[:, modal_index],
                    modal_gate[modal_index],
                    cap=self.cap,
                    eps=self.eps,
                    attention_temperature=self.attention_temperature,
                )

        category_token_items = []
        for modal_index, canonical_name in enumerate(self.canonical_modal_names):
            category_token = H[:, modal_index] + private_residuals[:, modal_index]
            if canonical_name in hft_outputs:
                category_token = category_token + hft_outputs[canonical_name]["delta"]
            category_token_items.append(category_token)
        category_tokens = torch.stack(category_token_items, dim=1)

        label_embeddings = []
        auxiliary_outputs = []
        for pool, auxiliary_head in zip(self.label_pools, self._Auxi_classifier):
            branch_embedding = pool(category_tokens)
            label_embeddings.append(branch_embedding)
            auxiliary_outputs.append(auxiliary_head(branch_embedding))

        self.last_branch_outputs = label_embeddings
        routed_branch_concat = torch.cat(label_embeddings, dim=-1)
        self.last_routed_branch_concat = routed_branch_concat
        category_embedding = self.Message_MLP(routed_branch_concat)

        # Global is the bit-identical C1 path: it never reads an HFT delta.
        global_embedding = self.Global_Message(X_gated)
        adjacency = torch.eye(
            X_raw.size(0), device=X_raw.device, dtype=X_raw.dtype
        )
        self.last_adj_base = adjacency
        self.last_label_relation = None
        fused_embedding = self._fuse_semantic_branches(
            category_embedding, global_embedding
        )
        raw_logits = self.GCN(fused_embedding, adjacency)

        if return_intermediates:
            hft_shared = {
                modality: H[:, self.modal_index_by_name[modality]]
                for modality in self.HFT_MODALITIES
                if modality in hft_outputs
            }
            intermediates: dict[str, Any] = {
                "feature_modal_output": feature_modal_output,
                "modal_tokens_pre_transformer": modal_tokens_pre_transformer,
                "modal_tokens_post_transformer": modal_tokens_post_transformer,
                "private_residuals": private_residuals,
                "category_tokens": category_tokens,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
                "hft_enabled": self.hft_enabled,
                "hft_cap": self.cap,
                "hft_attention_temperature": self.attention_temperature,
                "hft_shared_by_modality": hft_shared,
                "hft_raw_delta_by_modality": {
                    name: output["raw_delta"] for name, output in hft_outputs.items()
                },
                "hft_delta_by_modality": {
                    name: output["delta"] for name, output in hft_outputs.items()
                },
                "hft_ratio_by_modality": {
                    name: output["ratio"] for name, output in hft_outputs.items()
                },
                "hft_cap_scale_by_modality": {
                    name: output["cap_scale"] for name, output in hft_outputs.items()
                },
                "hft_cap_saturated_by_modality": {
                    name: output["cap_saturated"]
                    for name, output in hft_outputs.items()
                },
                "hft_attention_weights_by_modality": {
                    name: output["attention_weights"]
                    for name, output in hft_outputs.items()
                },
                "hft_local_evidence_by_modality": {
                    name: output["local_evidence"]
                    for name, output in hft_outputs.items()
                },
                "hft_feature_indices_by_modality": self.hft_feature_indices_by_modality,
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs


__all__ = [
    "FineGrainedModalityResidual",
    "HFTC1LiteModel",
    "PrivateResidualAdapter",
]
