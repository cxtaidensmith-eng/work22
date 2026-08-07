"""Boundary-Conditioned Disentangled Gradient Learning v1.

The model keeps the complete C1 inference path and adds twelve training-only
boundary heads.  The historical model injects directional noise into features
*before* the modal encoder.  Consequently, two encoder passes are required:

* a clean, pre-noise pass used only by the boundary heads; and
* the unchanged noisy C1 pass used by the fusion network.

The fusion pass uses a stop-gradient bridge.  Its forward value is bitwise the
historical noisy token, while its derivative with respect to that token is
``rho``.  Boundary-head inputs are computed from a detached calibrated-feature
tensor, so the boundary objective updates only the modal encoder and the twelve
heads, never Feature_Modal or any downstream C1 module.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import HeterGraph_Model_Kmeans


class PrivateResidualAdapter(nn.Module):
    """The unchanged C1 rank-reduced private residual adapter."""

    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(token)))


class BoundaryHead(nn.Module):
    """One fixed LayerNorm-without-affine plus linear binary head."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.classifier = nn.Linear(token_dim, 2)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(token))


class BCDGLModel(HeterGraph_Model_Kmeans):
    """C1 with training-only adjacent-boundary supervision.

    Parameters
    ----------
    rho:
        Fraction of the historical C1 fusion gradient allowed to enter the
        modal encoder.  V1 permits only the two preregistered values 0 and 0.25.
    boundary_enabled:
        When false, construct an exact C1 reference without the twelve heads.
        Formal BC-DGL models always leave this true.
    """

    BOUNDARY_NAMES = ("AD_SMCI", "CN_SMCI")

    def __init__(
        self,
        *args,
        rho: float,
        adapter_rank: int = 8,
        boundary_enabled: bool = True,
        **kwargs,
    ):
        # Construct every historical Original module first.  C1 adapters are
        # then created in their historical order, and only then are the twelve
        # new heads initialized.  Thus adding BC-DGL cannot perturb any C1
        # parameter's seeded initialization.
        super().__init__(*args, **kwargs)
        if not (
            self.category_branch_variant == "original"
            and self.query_pool_variant == "independent"
            and self.category_branch_fusion == "concat"
            and self.semantic_branch == "both"
            and self.semantic_fusion == "add"
            and self.adj_mode == "none"
        ):
            raise ValueError("BC-DGL requires the locked C1 configuration")
        if self._modal_num != 6:
            raise ValueError(f"BC-DGL requires six modalities, got {self._modal_num}")

        self.rho = float(rho)
        if self.rho not in (0.0, 0.25):
            raise ValueError("BC-DGL v1 permits only rho=0.0 or rho=0.25")
        self.adapter_rank = int(adapter_rank)
        self.boundary_enabled = bool(boundary_enabled)

        self.private_adapters = nn.ModuleList(
            [
                PrivateResidualAdapter(self.Hidden_size, self.adapter_rank)
                for _ in range(self._modal_num)
            ]
        )

        class_names = [str(name).strip() for name in self.DATASET_Dict["Class_Names"]]
        normalized_names = [name.upper() for name in class_names]
        required_names = ("AD", "CN", "SMCI")
        if sorted(normalized_names) != sorted(required_names):
            raise ValueError(f"Unexpected class names: {class_names}")
        self.class_names = tuple(class_names)
        self.class_index_by_name = {
            name: normalized_names.index(name) for name in required_names
        }

        if self.boundary_enabled:
            # Creation order is fixed and independent of rho, ensuring that
            # strict and mild arms receive identical auxiliary initialization.
            self.boundary_heads_ads = nn.ModuleList(
                [BoundaryHead(self.Hidden_size) for _ in range(self._modal_num)]
            )
            self.boundary_heads_cns = nn.ModuleList(
                [BoundaryHead(self.Hidden_size) for _ in range(self._modal_num)]
            )
        else:
            self.boundary_heads_ads = nn.ModuleList()
            self.boundary_heads_cns = nn.ModuleList()

    def boundary_head_parameter_count(self) -> int:
        """Number of training-only parameters excluded from inference."""

        return sum(
            parameter.numel()
            for module in (self.boundary_heads_ads, self.boundary_heads_cns)
            for parameter in module.parameters()
        )

    def inference_parameter_count(self) -> int:
        """C1 parameters used by the classification inference path."""

        total = sum(parameter.numel() for parameter in self.parameters())
        return total - self.boundary_head_parameter_count()

    def _boundary_logits(
        self, clean_modal_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.boundary_enabled:
            raise RuntimeError("Boundary heads are disabled for this C1 reference")
        if tuple(clean_modal_tokens.shape[1:]) != (
            self._modal_num,
            self.Hidden_size,
        ):
            raise ValueError(
                "Unexpected clean-token shape: "
                f"{tuple(clean_modal_tokens.shape)}"
            )
        logits_ads = torch.stack(
            [
                head(clean_modal_tokens[:, modal_index])
                for modal_index, head in enumerate(self.boundary_heads_ads)
            ],
            dim=1,
        )
        logits_cns = torch.stack(
            [
                head(clean_modal_tokens[:, modal_index])
                for modal_index, head in enumerate(self.boundary_heads_cns)
            ],
            dim=1,
        )
        return logits_ads, logits_cns

    def boundary_loss(
        self,
        boundary_logits_ads: torch.Tensor,
        boundary_logits_cns: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Balanced train-fold-only AD/sMCI and CN/sMCI losses.

        Binary target 0 is AD (or CN) and target 1 is sMCI.  Global numeric
        class indices are never assumed; all masks use the dataset names.
        """

        expected_shape = (labels.numel(), self._modal_num, 2)
        if tuple(boundary_logits_ads.shape) != expected_shape:
            raise ValueError(
                "AD-sMCI boundary-logit shape mismatch: "
                f"expected {expected_shape}, got {tuple(boundary_logits_ads.shape)}"
            )
        if tuple(boundary_logits_cns.shape) != expected_shape:
            raise ValueError(
                "CN-sMCI boundary-logit shape mismatch: "
                f"expected {expected_shape}, got {tuple(boundary_logits_cns.shape)}"
            )
        if labels.ndim != 1:
            raise ValueError(f"Expected one-dimensional labels, got {labels.shape}")
        train_mask = train_mask.to(device=labels.device, dtype=torch.bool)
        if tuple(train_mask.shape) != tuple(labels.shape):
            raise ValueError("Train-mask shape mismatch")

        ad_index = self.class_index_by_name["AD"]
        cn_index = self.class_index_by_name["CN"]
        smci_index = self.class_index_by_name["SMCI"]
        train_ad = train_mask & (labels == ad_index)
        train_cn = train_mask & (labels == cn_index)
        train_smci = train_mask & (labels == smci_index)
        if not bool(train_ad.any() and train_cn.any() and train_smci.any()):
            raise ValueError("Every boundary class must occur in fold-train")

        target_ad_or_cn = torch.zeros(
            int(labels.size(0)), device=labels.device, dtype=torch.long
        )
        target_smci = torch.ones(
            int(labels.size(0)), device=labels.device, dtype=torch.long
        )
        losses_ads = []
        losses_cns = []
        losses_by_modality = []
        for modal_index in range(self._modal_num):
            logits_ads = boundary_logits_ads[:, modal_index]
            logits_cns = boundary_logits_cns[:, modal_index]
            loss_ads = 0.5 * F.cross_entropy(
                logits_ads[train_ad], target_ad_or_cn[train_ad]
            ) + 0.5 * F.cross_entropy(
                logits_ads[train_smci], target_smci[train_smci]
            )
            loss_cns = 0.5 * F.cross_entropy(
                logits_cns[train_cn], target_ad_or_cn[train_cn]
            ) + 0.5 * F.cross_entropy(
                logits_cns[train_smci], target_smci[train_smci]
            )
            losses_ads.append(loss_ads)
            losses_cns.append(loss_cns)
            losses_by_modality.append(0.5 * (loss_ads + loss_cns))

        stacked_ads = torch.stack(losses_ads)
        stacked_cns = torch.stack(losses_cns)
        stacked_by_modality = torch.stack(losses_by_modality)
        loss_boundary = stacked_by_modality.mean()
        details = {
            "loss_AD_SMCI_by_modality": stacked_ads,
            "loss_CN_SMCI_by_modality": stacked_cns,
            "loss_boundary_by_modality": stacked_by_modality,
            "loss_boundary": loss_boundary,
            "train_AD_mask": train_ad,
            "train_CN_mask": train_cn,
            "train_sMCI_mask": train_smci,
            "train_mask": train_mask,
        }
        return loss_boundary, details

    # Explicit alias for runners that prefer a verb-oriented helper name.
    compute_boundary_loss = boundary_loss

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
        compute_boundary: bool = False,
    ):
        self.last_modal_tokens = None
        feature_modal_output = self.Feature_Modal(X_raw)

        clean_modal_tokens = None
        boundary_logits_ads = None
        boundary_logits_cns = None
        if compute_boundary:
            if not self.boundary_enabled:
                raise ValueError("Cannot compute boundary logits with disabled heads")
            # Detaching calibrated features confines this objective to the
            # shared modal encoder and the twelve heads.
            clean_modal_tokens = self.modal_token_encoder(
                feature_modal_output.detach()
            )
            boundary_logits_ads, boundary_logits_cns = self._boundary_logits(
                clean_modal_tokens
            )

        # Historical C1 feature-noise path.  Its order and forward values are
        # deliberately unchanged.
        X = feature_modal_output
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (
                per_feature_noise_std * self.noise_scale
            ).view(1, -1)

        modal_gate = torch.sigmoid(self.modal_gate_logit)
        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)

        noisy_modal_tokens = self.modal_token_encoder(X)
        noisy_modal_tokens_detached = noisy_modal_tokens.detach()
        # Forward is exactly noisy_modal_tokens; only the backward Jacobian is
        # scaled.  This algebraic form avoids rounding from rho*x+(1-rho)*x.
        fusion_modal_tokens = noisy_modal_tokens_detached + self.rho * (
            noisy_modal_tokens - noisy_modal_tokens_detached
        )
        H = fusion_modal_tokens * modal_gate.view(1, -1, 1)
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
        category_tokens = H + private_residuals
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

        # The original Global path consumes gated noisy feature values and is
        # independent of the rho bridge.
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
            intermediates: dict[str, Any] = {
                "feature_modal_output": feature_modal_output,
                "clean_modal_tokens": clean_modal_tokens,
                "noisy_modal_tokens": noisy_modal_tokens,
                "u_fusion": fusion_modal_tokens,
                "fusion_modal_tokens": fusion_modal_tokens,
                "fusion_forward_max_abs_error": (
                    fusion_modal_tokens - noisy_modal_tokens
                ).abs().max(),
                "modal_tokens_pre_transformer": modal_tokens_pre_transformer,
                "modal_tokens_post_transformer": modal_tokens_post_transformer,
                "private_residuals": private_residuals,
                "category_tokens": category_tokens,
                "boundary_logits_ads": boundary_logits_ads,
                "boundary_logits_cns": boundary_logits_cns,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
                "rho": self.rho,
                "boundary_computed": bool(compute_boundary),
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs
