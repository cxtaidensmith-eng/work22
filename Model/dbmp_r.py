"""Dual-Boundary sMCI Multi-Prototype Residual v1.

This module preserves the complete C1 shared/private model.  It constructs
non-parametric, fold-train-only boundary-conditioned sMCI prototypes in the
DIFFormer output space and adds a two-scalar bounded residual to C1 logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import HeterGraph_Model_Kmeans


class PrivateResidualAdapter(nn.Module):
    """The unchanged C1 rank-8 private residual adapter."""

    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(token)))


class DBMPRModel(HeterGraph_Model_Kmeans):
    """C1 plus a train-fold-only dual-boundary prototype logit residual."""

    def __init__(
        self,
        *args,
        adapter_rank: int = 8,
        dbmp_enabled: bool = True,
        tau_assignment: float = 0.25,
        tau_prototype: float = 0.25,
        gamma_initial: float = 0.05,
        **kwargs,
    ):
        # Historical modules are created first so seeded C1 parameters remain
        # bit-identical.  DBMP adds only two scalars after the six C1 adapters.
        super().__init__(*args, **kwargs)
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode == "none"
        ):
            raise ValueError("DBMP-R requires the locked C1 configuration")

        self.adapter_rank = int(adapter_rank)
        self.dbmp_enabled = bool(dbmp_enabled)
        self.tau_assignment = float(tau_assignment)
        self.tau_prototype = float(tau_prototype)
        if not 0.0 < gamma_initial < 0.5:
            raise ValueError("gamma_initial must lie strictly between zero and 0.5")
        self.gamma_initial = float(gamma_initial)
        self.prototype_eps = 1e-8

        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )

        class_names = [str(name) for name in self.DATASET_Dict["Class_Names"]]
        normalized_names = [name.upper() for name in class_names]
        required = ("AD", "CN", "SMCI")
        if sorted(normalized_names) != sorted(required):
            raise ValueError(f"Unexpected class names: {class_names}")
        self.class_names = tuple(class_names)
        self.class_index_by_name = {
            name: normalized_names.index(name) for name in required
        }

        if self.dbmp_enabled:
            # gamma = 0.5*sigmoid(a), so sigmoid(a)=gamma/0.5.
            initial_logit = math.log(
                (self.gamma_initial / 0.5)
                / (1.0 - self.gamma_initial / 0.5)
            )
            self.a_CN = nn.Parameter(torch.tensor(float(initial_logit)))
            self.a_AD = nn.Parameter(torch.tensor(float(initial_logit)))

    def boundary_gamma(self) -> torch.Tensor:
        if not self.dbmp_enabled:
            return self.modal_gate_logit.new_zeros(2)
        return 0.5 * torch.sigmoid(torch.stack((self.a_CN, self.a_AD)))

    def _category_tokens(
        self, shared_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        private_residuals = torch.stack(
            [
                adapter(shared_tokens[:, modality_index])
                for modality_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        return shared_tokens + private_residuals, private_residuals

    def _prototype_terms(
        self,
        final_representation: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        residual_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        if labels.ndim != 1 or labels.size(0) != final_representation.size(0):
            raise ValueError("Label shape mismatch")
        train_mask = train_mask.to(
            device=final_representation.device, dtype=torch.bool
        )
        if tuple(train_mask.shape) != (final_representation.size(0),):
            raise ValueError("Train-mask shape mismatch")

        ad_index = self.class_index_by_name["AD"]
        cn_index = self.class_index_by_name["CN"]
        smci_index = self.class_index_by_name["SMCI"]
        ad_train = train_mask & (labels == ad_index)
        cn_train = train_mask & (labels == cn_index)
        smci_train = train_mask & (labels == smci_index)
        if not bool(ad_train.any() and cn_train.any() and smci_train.any()):
            raise ValueError("Every DBMP prototype class must occur in fold-train")

        # Only prototype construction is detached. Query z remains connected
        # so both the final representation and the C1 backbone receive gradient.
        z = F.normalize(final_representation, dim=-1, eps=self.prototype_eps)
        z_prototype = z.detach()
        center_ad = F.normalize(
            z_prototype[ad_train].mean(dim=0), dim=0, eps=self.prototype_eps
        )
        center_cn = F.normalize(
            z_prototype[cn_train].mean(dim=0), dim=0, eps=self.prototype_eps
        )

        z_smci = z_prototype[smci_train]
        similarity_cn = z_smci @ center_cn
        similarity_ad = z_smci @ center_ad
        weight_cn = torch.sigmoid(
            (similarity_cn - similarity_ad) / self.tau_assignment
        ).clamp(0.05, 0.95)
        weight_ad = 1.0 - weight_cn
        prototype_smci_cn = F.normalize(
            (weight_cn.unsqueeze(-1) * z_smci).sum(dim=0)
            / (weight_cn.sum() + self.prototype_eps),
            dim=0,
            eps=self.prototype_eps,
        )
        prototype_smci_ad = F.normalize(
            (weight_ad.unsqueeze(-1) * z_smci).sum(dim=0)
            / (weight_ad.sum() + self.prototype_eps),
            dim=0,
            eps=self.prototype_eps,
        )

        q_cn = (
            z @ prototype_smci_cn - z @ center_cn
        ) / self.tau_prototype
        q_ad = (
            z @ center_ad - z @ prototype_smci_ad
        ) / self.tau_prototype

        gamma = self.boundary_gamma()
        gamma_effective = gamma if residual_enabled else torch.zeros_like(gamma)
        delta_cn = gamma_effective[0] * torch.tanh(q_cn)
        delta_ad = gamma_effective[1] * torch.tanh(q_ad)
        r_cn = -(2.0 * delta_cn + delta_ad) / 3.0
        r_smci = (delta_cn - delta_ad) / 3.0
        r_ad = (delta_cn + 2.0 * delta_ad) / 3.0
        residual = torch.zeros(
            (final_representation.size(0), self._Label_num),
            device=final_representation.device,
            dtype=final_representation.dtype,
        )
        residual[:, cn_index] = r_cn
        residual[:, smci_index] = r_smci
        residual[:, ad_index] = r_ad

        cn_term = 0.5 * F.softplus(q_cn[cn_train]).mean()
        cn_smci_term = 0.5 * (
            weight_cn * F.softplus(-q_cn[smci_train])
        ).sum() / (weight_cn.sum() + self.prototype_eps)
        ad_term = 0.5 * F.softplus(-q_ad[ad_train]).mean()
        ad_smci_term = 0.5 * (
            weight_ad * F.softplus(q_ad[smci_train])
        ).sum() / (weight_ad.sum() + self.prototype_eps)
        loss_cn = cn_term + cn_smci_term
        loss_ad = ad_term + ad_smci_term
        loss_prototype = 0.5 * (loss_cn + loss_ad)

        return {
            "z": z,
            "center_AD": center_ad,
            "center_CN": center_cn,
            "prototype_sMCI_CN": prototype_smci_cn,
            "prototype_sMCI_AD": prototype_smci_ad,
            "weight_CN": weight_cn,
            "weight_AD": weight_ad,
            "q_CN": q_cn,
            "q_AD": q_ad,
            "gamma": gamma,
            "gamma_effective": gamma_effective,
            "delta_CN": delta_cn,
            "delta_AD": delta_ad,
            "logit_residual": residual,
            "loss_CN": loss_cn,
            "loss_AD": loss_ad,
            "loss_prototype": loss_prototype,
            "train_mask": train_mask,
            "train_AD_mask": ad_train,
            "train_CN_mask": cn_train,
            "train_sMCI_mask": smci_train,
        }

    def forward(
        self,
        X_raw: torch.Tensor,
        labels: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        residual_enabled: bool = True,
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
        shared_tokens = self.modal_token_encoder(X)
        shared_tokens = shared_tokens * modal_gate.view(1, -1, 1)
        modal_tokens_pre_transformer = shared_tokens
        for block in self.shared_transformer:
            shared_tokens = block(shared_tokens)
        self.last_modal_tokens = shared_tokens.detach()

        category_tokens, private_residuals = self._category_tokens(shared_tokens)
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
        base_logits = self.GCN(fused_embedding, adjacency)
        final_representation = self.GCN.GCN_feature_1
        if final_representation is None:
            raise RuntimeError("DIFFormer did not expose its classifier input")

        if self.dbmp_enabled:
            if labels is None or train_mask is None:
                raise ValueError("DBMP requires fold-train labels and mask")
            prototype_terms = self._prototype_terms(
                final_representation,
                labels,
                train_mask,
                bool(residual_enabled),
            )
            final_logits = base_logits + prototype_terms["logit_residual"]
        else:
            prototype_terms = {}
            final_logits = base_logits

        if return_intermediates:
            intermediates = {
                "feature_modal_output": feature_modal_output,
                "modal_tokens_pre_transformer": modal_tokens_pre_transformer,
                "modal_tokens_post_transformer": shared_tokens,
                "private_residuals": private_residuals,
                "category_tokens": category_tokens,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H_fused": fused_embedding,
                "final_representation": final_representation,
                "base_logits": base_logits,
                "raw_logits": final_logits,
                "residual_enabled": bool(residual_enabled),
                **prototype_terms,
            }
            return final_logits, label_embeddings, auxiliary_outputs, intermediates
        return final_logits, label_embeddings, auxiliary_outputs
