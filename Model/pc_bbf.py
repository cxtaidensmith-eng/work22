"""Patient-conditioned bounded fusion on top of the locked C1 model.

The formal C1 checkpoint was produced by a small extension of
``HeterGraph_Model_Kmeans``: six zero-initialized rank-8 private adapters alter
the modal tokens supplied to the three historical query pools.  The target
branch starts from the parent of the historical C1 commit, so the exact C1
replay model is kept here as a self-contained class rather than depending on a
sibling worktree at runtime.

``PCBBFC1Model`` constructs every C1 module first.  Only afterwards does it add
the PC-BBF module, preserving the seeded initialization (and state-dict names)
of all C1 parameters.  PC-BBF leaves the historical ``Y + G`` fusion in place
and adds a per-patient residual whose pre-gate norm cannot exceed a fixed
fraction of ``||Y + G||``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .network import HeterGraph_Model_Kmeans


C1_EXPECTED_PARAMETERS = 862_971
PC_BBF_ADDED_PARAMETERS_H96_R8 = 3_953
PC_BBF_EXPECTED_PARAMETERS = 866_924


class PrivateResidualAdapter(nn.Module):
    """The exact rank-8 private adapter used by formal C1 checkpoints."""

    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(token)))


class C1SharedPrivateControlModel(HeterGraph_Model_Kmeans):
    """Self-contained, strict-load-compatible replay of formal C1.

    The constructor accepts the historical CME-only keyword arguments so the
    same locked builder can be used for both the old checkpoint replay and the
    new model.  C1 itself has no router, so those compatibility arguments do
    not create modules or consume random numbers.
    """

    def __init__(
        self,
        *args: Any,
        cme_arm: str = "c1",
        adapter_rank: int = 8,
        router_hidden: int = 16,
        modality_embedding_dim: int = 8,
        **kwargs: Any,
    ):
        del router_hidden, modality_embedding_dim
        # Construct the complete historical Original Query model first.
        super().__init__(*args, **kwargs)
        if str(cme_arm).lower() != "c1":
            raise ValueError("C1SharedPrivateControlModel only supports cme_arm='c1'")
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode == "none"
        ):
            raise ValueError("C1 requires the locked Original Query configuration")

        self.cme_arm = "c1"
        self.adapter_rank = int(adapter_rank)
        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )

    def _category_token_streams(
        self, tokens: torch.Tensor
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        residuals = torch.stack(
            [
                adapter(tokens[:, modality_index])
                for modality_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        category_tokens = tokens + residuals
        streams = [category_tokens, category_tokens, category_tokens]
        return streams, {
            "private_residuals": residuals,
            "category_token_streams": torch.stack(streams, dim=1),
        }

    def _extra_fusion_intermediates(self) -> dict[str, torch.Tensor]:
        return {}

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
        counterfactual_modal_index: int | None = None,
        counterfactual_sample_mask: torch.Tensor | None = None,
    ):
        """Run the exact formal C1 path, exposing the two pre-add inputs."""

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

        H = self.modal_token_encoder(X)
        if counterfactual_modal_index is not None:
            modal_index = int(counterfactual_modal_index)
            if not 0 <= modal_index < self._modal_num:
                raise ValueError(f"Invalid counterfactual modality: {modal_index}")
            if counterfactual_sample_mask is None:
                raise ValueError(
                    "Counterfactual deletion requires an explicit sample mask"
                )
            sample_mask = counterfactual_sample_mask.to(
                device=H.device, dtype=torch.bool
            )
            if tuple(sample_mask.shape) != (H.size(0),):
                raise ValueError("Counterfactual sample mask shape mismatch")
            keep = torch.ones(
                (H.size(0), self._modal_num), device=H.device, dtype=H.dtype
            )
            keep[sample_mask, modal_index] = 0.0
            H = H * keep.unsqueeze(-1)
        H = H * modal_gate.view(1, -1, 1)
        modal_tokens_pre_transformer = H if return_intermediates else None
        for block in self.shared_transformer:
            H = block(H)
        modal_tokens_post_transformer = H if return_intermediates else None
        self.last_modal_tokens = H.detach()

        pool_token_streams, c1_intermediates = self._category_token_streams(H)
        label_embeddings = []
        auxiliary_outputs = []
        for label_index, (pool, auxiliary_head) in enumerate(
            zip(self.label_pools, self._Auxi_classifier)
        ):
            branch_embedding = pool(pool_token_streams[label_index])
            label_embeddings.append(branch_embedding)
            auxiliary_outputs.append(auxiliary_head(branch_embedding))

        self.last_branch_outputs = label_embeddings
        routed_branch_concat = torch.cat(label_embeddings, dim=-1)
        self.last_routed_branch_concat = routed_branch_concat
        category_embedding = self.Message_MLP(routed_branch_concat)

        # This is the untouched historical Global Message path.  It consumes
        # gated raw modal features, not post-transformer tokens.
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
            intermediates = {
                "feature_modal_output": feature_modal_output,
                "modal_tokens_pre_transformer": modal_tokens_pre_transformer,
                "modal_tokens_post_transformer": modal_tokens_post_transformer,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H0": category_embedding + global_embedding,
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
                "counterfactual_modal_index": counterfactual_modal_index,
                **c1_intermediates,
                **self._extra_fusion_intermediates(),
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs


class PatientConditionedBoundedFusion(nn.Module):
    """Rank-8 patient-conditioned correction bounded relative to ``Y + G``."""

    def __init__(self, hidden_size: int, rank: int = 8, cap_value: float = 0.10):
        super().__init__()
        if int(rank) != 8:
            raise ValueError("PC-BBF v1 requires rank=8")
        if float(cap_value) <= 0.0:
            raise ValueError("cap_value must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.cap_value = float(cap_value)

        # One shared, parameter-free normalization is deliberately applied to
        # both branches.
        self.branch_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False
        )
        self.input_projection = nn.Linear(4 * self.hidden_size, self.rank)
        self.activation = nn.GELU()
        self.residual_output = nn.Linear(self.rank, self.hidden_size)
        self.gate_output = nn.Linear(self.rank, 1)

        # Only the two output layers are zero initialized.  The input
        # projection retains its ordinary nonzero initialization.
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(
        self, category_embedding: torch.Tensor, global_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if category_embedding.shape != global_embedding.shape:
            raise ValueError(
                "Category and Global representations must have matching shapes"
            )
        if category_embedding.ndim != 2 or category_embedding.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected [N,{self.hidden_size}] branch representations, got "
                f"{tuple(category_embedding.shape)}"
            )

        h0 = category_embedding + global_embedding
        y = self.branch_norm(category_embedding)
        g = self.branch_norm(global_embedding)
        fusion_input = torch.cat((y, g, torch.abs(y - g), y * g), dim=-1)
        z = self.activation(self.input_projection(fusion_input))
        residual_raw = self.residual_output(z)
        gate_logit = self.gate_output(z)
        gate = torch.sigmoid(gate_logit)

        shared_norm = h0.norm(dim=-1, keepdim=True)
        residual_raw_norm = residual_raw.norm(dim=-1, keepdim=True)
        cap_scale = torch.clamp(
            self.cap_value * shared_norm / (residual_raw_norm + 1e-8),
            max=1.0,
        )
        residual_capped = residual_raw * cap_scale
        applied_residual = gate * residual_capped
        fused = h0 + applied_residual

        return fused, {
            "pc_bbf_normalized_Y": y,
            "pc_bbf_normalized_G": g,
            "pc_bbf_z": z,
            "pc_bbf_residual_raw": residual_raw,
            "pc_bbf_residual_capped": residual_capped,
            "pc_bbf_gate_logit": gate_logit,
            "pc_bbf_gate": gate,
            "pc_bbf_cap_scale": cap_scale,
            "pc_bbf_applied_residual": applied_residual,
            "pc_bbf_shared_norm": shared_norm,
            "pc_bbf_residual_raw_norm": residual_raw_norm,
        }


class PCBBFC1Model(C1SharedPrivateControlModel):
    """C1 plus Patient-Conditioned Bounded Branch Fusion v1."""

    def __init__(
        self,
        *args: Any,
        fusion_rank: int = 8,
        cap_value: float = 0.10,
        **kwargs: Any,
    ):
        # All C1 modules, including the six private adapters, must be created
        # before PC-BBF so their seed-dependent initialization remains exact.
        super().__init__(*args, **kwargs)
        self.pc_bbf = PatientConditionedBoundedFusion(
            self.Hidden_size, rank=fusion_rank, cap_value=cap_value
        )
        self._last_pc_bbf_intermediates: dict[str, torch.Tensor] = {}

    def _fuse_semantic_branches(
        self, category_embedding: torch.Tensor, global_embedding: torch.Tensor
    ) -> torch.Tensor:
        self.last_category_embedding = category_embedding
        self.last_global_embedding = global_embedding
        fused, intermediates = self.pc_bbf(category_embedding, global_embedding)
        self._last_pc_bbf_intermediates = intermediates
        self.last_global_residual = intermediates["pc_bbf_applied_residual"]
        self.last_global_gate = intermediates["pc_bbf_gate"]
        return fused

    def _extra_fusion_intermediates(self) -> dict[str, torch.Tensor]:
        return self._last_pc_bbf_intermediates


# Concise public alias for runners.
PCBBFModel = PCBBFC1Model


__all__ = [
    "C1_EXPECTED_PARAMETERS",
    "PC_BBF_ADDED_PARAMETERS_H96_R8",
    "PC_BBF_EXPECTED_PARAMETERS",
    "PrivateResidualAdapter",
    "C1SharedPrivateControlModel",
    "PatientConditionedBoundedFusion",
    "PCBBFC1Model",
    "PCBBFModel",
]
