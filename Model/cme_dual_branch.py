"""Counterfactual marginal-evidence dual-branch extensions.

The legacy Original Query modules are constructed first and are left unchanged.
Only the post-transformer tokens supplied to the three existing query pools are
augmented.  This keeps the historical Global Message, Message_MLP, DIFFormer,
classifier, and auxiliary heads intact.
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


class MarginalEvidenceRouter(nn.Module):
    """One shared scorer applied to all six modalities of each patient."""

    def __init__(
        self,
        hidden_size: int,
        modality_count: int,
        router_hidden: int = 16,
        modality_embedding_dim: int = 8,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.modality_embedding = nn.Embedding(
            modality_count, modality_embedding_dim
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_size + modality_embedding_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [N,M,D] tokens, got {tuple(tokens.shape)}")
        batch_size, modality_count, _ = tokens.shape
        normalized = self.norm(tokens)
        global_context = self.norm(tokens.mean(dim=1)).unsqueeze(1).expand(
            -1, modality_count, -1
        )
        modality_ids = torch.arange(modality_count, device=tokens.device)
        modality_embedding = self.modality_embedding(modality_ids).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        router_input = torch.cat(
            (normalized, global_context, modality_embedding), dim=-1
        )
        scores = self.scorer(router_input).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return weights, scores


class CMEDualBranchModel(HeterGraph_Model_Kmeans):
    """Original Query plus the C1/C2/C3 private-token pathways."""

    VALID_ARMS = {"c1", "c2", "c3"}

    def __init__(
        self,
        *args,
        cme_arm: str,
        adapter_rank: int = 8,
        router_hidden: int = 16,
        modality_embedding_dim: int = 8,
        private_source: str = "post_shared",
        **kwargs,
    ):
        # Construct every historical module first so its seeded initialization
        # is bit-identical to Original Query.
        super().__init__(*args, **kwargs)
        arm = str(cme_arm).lower()
        if arm not in self.VALID_ARMS:
            raise ValueError(f"Unsupported CME arm: {cme_arm}")
        source = str(private_source).lower()
        if source not in {"post_shared", "pre_shared"}:
            raise ValueError(f"Unsupported private source: {private_source}")
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode in {"learned", "fixed", "none"}
        ):
            raise ValueError("CME requires the locked Original Query configuration")

        self.cme_arm = arm
        self.private_source = source
        self.adapter_rank = int(adapter_rank)
        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )
        if arm == "c2":
            self.cme_router = MarginalEvidenceRouter(
                self.Hidden_size,
                self._modal_num,
                router_hidden=router_hidden,
                modality_embedding_dim=modality_embedding_dim,
            )
        elif arm == "c3":
            self.cme_boundary_routers = nn.ModuleDict(
                {
                    "ad_smci": MarginalEvidenceRouter(
                        self.Hidden_size,
                        self._modal_num,
                        router_hidden=router_hidden,
                        modality_embedding_dim=modality_embedding_dim,
                    ),
                    "cn_smci": MarginalEvidenceRouter(
                        self.Hidden_size,
                        self._modal_num,
                        router_hidden=router_hidden,
                        modality_embedding_dim=modality_embedding_dim,
                    ),
                }
            )

    def _category_token_streams(
        self,
        tokens: torch.Tensor,
        private_tokens: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if private_tokens is None:
            private_tokens = tokens
        if private_tokens.shape != tokens.shape:
            raise ValueError(
                "Private-source and post-shared token shapes must match: "
                f"{tuple(private_tokens.shape)} != {tuple(tokens.shape)}"
            )
        residuals = torch.stack(
            [
                adapter(private_tokens[:, modality_index])
                for modality_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        extras: dict[str, torch.Tensor] = {
            "private_adapter_inputs": private_tokens,
            "private_residual_addback_base": tokens,
            "private_residuals": residuals,
        }
        if self.cme_arm == "c1":
            category_tokens = tokens + residuals
            streams = [category_tokens for _ in range(self._Label_num)]
        elif self.cme_arm == "c2":
            weights, scores = self.cme_router(tokens)
            category_tokens = tokens + 6.0 * weights.unsqueeze(-1) * residuals
            streams = [category_tokens, category_tokens, category_tokens]
            extras.update({"router_weights": weights, "router_scores": scores})
        else:
            ad_smci_weights, ad_smci_scores = self.cme_boundary_routers["ad_smci"](
                tokens
            )
            cn_smci_weights, cn_smci_scores = self.cme_boundary_routers["cn_smci"](
                tokens
            )
            streams = [
                tokens + 6.0 * ad_smci_weights.unsqueeze(-1) * residuals,
                tokens + 6.0 * cn_smci_weights.unsqueeze(-1) * residuals,
                tokens
                + 3.0
                * (ad_smci_weights + cn_smci_weights).unsqueeze(-1)
                * residuals,
            ]
            extras.update(
                {
                    "router_weights_ad_smci": ad_smci_weights,
                    "router_scores_ad_smci": ad_smci_scores,
                    "router_weights_cn_smci": cn_smci_weights,
                    "router_scores_cn_smci": cn_smci_scores,
                }
            )
        extras["category_token_streams"] = torch.stack(streams, dim=1)
        return streams, extras

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
        counterfactual_modal_index: int | None = None,
        counterfactual_sample_mask: torch.Tensor | None = None,
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

        H = self.modal_token_encoder(X)
        if counterfactual_modal_index is not None:
            modal_index = int(counterfactual_modal_index)
            if not 0 <= modal_index < self._modal_num:
                raise ValueError(f"Invalid counterfactual modality: {modal_index}")
            if counterfactual_sample_mask is None:
                raise ValueError("Counterfactual deletion requires an explicit sample mask")
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
        pre_shared_tokens = H
        modal_tokens_pre_transformer = pre_shared_tokens if return_intermediates else None
        for block in self.shared_transformer:
            H = block(H)
        modal_tokens_post_transformer = H if return_intermediates else None
        self.last_modal_tokens = H.detach()

        private_tokens = pre_shared_tokens if self.private_source == "pre_shared" else H
        pool_token_streams, cme_intermediates = self._category_token_streams(
            H, private_tokens=private_tokens
        )
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
        # gated raw modal features in Original Query; changing its interface to
        # post-transformer tokens would violate the required initial equality.
        global_embedding = self.Global_Message(X_gated)
        if self.adj_mode == "learned":
            adjacency = self.Adj_Learning(X_gated)
        elif self.adj_mode == "fixed" and self.fixed_adj is not None:
            adjacency = self.fixed_adj.to(device=X_raw.device, dtype=X_raw.dtype)
        else:
            adjacency = torch.eye(
                X_raw.size(0), device=X_raw.device, dtype=X_raw.dtype
            )
        self.last_adj_base = adjacency
        self.last_label_relation = None
        if self.label_graph_alpha > 0 or self.label_graph_reg_lambda > 0:
            label_relation = self._label_relation_graph(label_embeddings)
            self.last_label_relation = label_relation
            alpha = max(0.0, min(1.0, self.label_graph_alpha))
            if alpha > 0:
                adjacency = (
                    (1.0 - alpha) * adjacency + alpha * label_relation
                )
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
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
                "counterfactual_modal_index": counterfactual_modal_index,
                "private_source": self.private_source,
                **cme_intermediates,
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs
