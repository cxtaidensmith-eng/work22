"""Gate-Protected Private Residual v1.

The historical Original/C1 shared and Global paths are preserved.  Only the
input to the six rank-8 private adapters changes: they read clean modal-encoder
tokens before directional noise and the original modal gate.  Six independent
bounded scalars control the applied residual magnitude.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .network import HeterGraph_Model_Kmeans


class PrivateResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(token)))


class GateProtectedPrivateResidualModel(HeterGraph_Model_Kmeans):
    """Original Query with C1 adapters fed by clean, pre-gate modal tokens."""

    def __init__(
        self,
        *args,
        adapter_rank: int = 8,
        gate_protected: bool = True,
        **kwargs,
    ):
        # Construct every historical module before GPPR additions so seeded
        # Original/C1 parameters remain bit-identical.
        super().__init__(*args, **kwargs)
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode == "none"
        ):
            raise ValueError("GPPR requires the locked Original Query configuration")
        self.adapter_rank = int(adapter_rank)
        self.gate_protected = bool(gate_protected)
        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )
        if self.gate_protected:
            # beta_m = 1 + 0.5*tanh(a_m), initialized to exactly one.
            self.private_residual_gate_logits = nn.Parameter(
                torch.zeros(self._modal_num)
            )

    def private_beta(self) -> torch.Tensor:
        if not self.gate_protected:
            return self.modal_gate_logit.new_ones(self._modal_num)
        return 1.0 + 0.5 * torch.tanh(self.private_residual_gate_logits)

    def forward(self, X_raw: torch.Tensor, return_intermediates: bool = False):
        self.last_modal_tokens = None
        feature_modal_output = self.Feature_Modal(X_raw)

        # U: feature-calibrated Modal Encoder output. Modal_Token_Encoder ends
        # with its existing input_norm, so no new normalization is introduced.
        # This pass occurs before directional noise and before modal gating.
        clean_modal_tokens = self.modal_token_encoder(feature_modal_output)

        # Historical shared/global path, kept in the original order.
        X = feature_modal_output
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (
                per_feature_noise_std * self.noise_scale
            ).view(1, -1)
            shared_encoded_tokens = self.modal_token_encoder(X)
        else:
            # In eval mode X equals feature_modal_output, so reusing the clean
            # deterministic encoding is exactly the historical computation.
            shared_encoded_tokens = clean_modal_tokens

        modal_gate = torch.sigmoid(self.modal_gate_logit)
        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)
        shared_pre_transformer = shared_encoded_tokens * modal_gate.view(1, -1, 1)
        shared_tokens = shared_pre_transformer
        for block in self.shared_transformer:
            shared_tokens = block(shared_tokens)
        self.last_modal_tokens = shared_tokens.detach()

        private_input = clean_modal_tokens if self.gate_protected else shared_tokens
        private_residuals = torch.stack(
            [
                adapter(private_input[:, modality_index])
                for modality_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        beta = self.private_beta()
        applied_private_residuals = beta.view(1, -1, 1) * private_residuals
        category_tokens = shared_tokens + applied_private_residuals

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
                "clean_modal_tokens_pre_noise_pre_gate": clean_modal_tokens,
                "shared_modal_tokens_pre_transformer": shared_pre_transformer,
                "shared_modal_tokens_post_transformer": shared_tokens,
                "private_input_tokens": private_input,
                "private_residuals": private_residuals,
                "private_beta": beta,
                "applied_private_residuals": applied_private_residuals,
                "category_tokens": category_tokens,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs

