"""Class-balanced train-only retrieval residual for the locked C1 model.

The retrieval memory is supplied per fold as training indices plus their labels.
Every query retrieves four cases from each training class, while test subjects
can never enter the memory.  The bounded residual is inserted after C1's
Category + Global fusion and before the untouched DIFFormer/classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cme_dual_branch import CMEDualBranchModel


C1_PARAMETER_COUNT = 862_971
CBTR_ADDED_PARAMETERS_H96_D16 = 3_376
CBTR_TOTAL_PARAMETERS_H96_D16 = 866_347


class ClassBalancedTrainOnlyRetrieval(nn.Module):
    """Three-class, equal-candidate-count retrieval using train-only memory."""

    def __init__(
        self,
        hidden_size: int,
        *,
        key_dim: int = 16,
        value_dim: int = 16,
        per_class_k: int = 4,
        temperature: float = 0.2,
        residual_cap: float = 0.10,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.per_class_k = int(per_class_k)
        self.temperature = float(temperature)
        self.residual_cap = float(residual_cap)

        if self.key_dim != 16 or self.value_dim != 16:
            raise ValueError("CBTR-C1 v1 fixes key_dim=value_dim=16")
        if self.per_class_k != 4:
            raise ValueError("CBTR-C1 v1 fixes four candidates per class")
        if abs(self.temperature - 0.2) > 1e-12:
            raise ValueError("CBTR-C1 v1 fixes retrieval temperature=0.2")
        if self.residual_cap not in (0.10, 0.15):
            raise ValueError("CBTR supports only the v1/v1.1 residual caps")

        self.key_projection = nn.Linear(
            self.hidden_size, self.key_dim, bias=False
        )
        self.label_embedding = nn.Embedding(3, self.value_dim)
        self.diff_projection = nn.Linear(
            self.key_dim, self.value_dim, bias=False
        )
        self.output_projection = nn.Linear(
            self.value_dim, self.hidden_size, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _canonical_train_memory(
        train_idx: torch.Tensor,
        train_y: torch.Tensor,
        subject_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(train_idx, torch.Tensor) or train_idx.ndim != 1:
            raise ValueError("train_idx must be a one-dimensional tensor")
        if train_idx.dtype == torch.bool:
            if train_idx.numel() != subject_count:
                raise ValueError("Boolean train_idx must span all subjects")
            train_indices = train_idx.to(device=device).nonzero(as_tuple=False).flatten()
        else:
            train_indices = train_idx.to(device=device, dtype=torch.long)
        if train_indices.numel() == 0:
            raise ValueError("The train-only retrieval memory is empty")
        if int(train_indices.min()) < 0 or int(train_indices.max()) >= subject_count:
            raise ValueError("train_idx contains an out-of-range subject")
        if torch.unique(train_indices).numel() != train_indices.numel():
            raise ValueError("train_idx contains duplicate subjects")

        if not isinstance(train_y, torch.Tensor) or train_y.ndim != 1:
            raise ValueError("train_y must be a one-dimensional tensor")
        memory_labels = train_y.to(device=device, dtype=torch.long)
        if memory_labels.numel() != train_indices.numel():
            raise ValueError(
                "train_y must contain only the labels aligned with train_idx"
            )
        if int(memory_labels.min()) < 0 or int(memory_labels.max()) > 2:
            raise ValueError("CBTR memory labels must use class indices 0, 1, 2")
        return train_indices, memory_labels

    def forward(
        self,
        h0: torch.Tensor,
        train_idx: torch.Tensor,
        train_y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
        """Return a bounded residual and retrieval audit intermediates.

        ``train_y`` is deliberately aligned only to ``train_idx``.  A complete
        label vector is neither accepted nor needed.
        """

        if h0.ndim != 2 or h0.size(1) != self.hidden_size:
            raise ValueError(
                f"Expected H0 [N,{self.hidden_size}], got {tuple(h0.shape)}"
            )
        subject_count = int(h0.size(0))
        train_indices, memory_labels = self._canonical_train_memory(
            train_idx, train_y, subject_count, h0.device
        )

        normalized_h0 = F.layer_norm(
            h0.detach(),
            (self.hidden_size,),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        keys = F.normalize(
            self.key_projection(normalized_h0), p=2, dim=-1, eps=1e-8
        )

        selected_indices_by_class = []
        selected_labels_by_class = []
        selected_scores_by_class = []
        query_indices = torch.arange(subject_count, device=h0.device)
        for class_index in range(3):
            class_memory = train_indices[memory_labels == class_index]
            # A memory query from this class must still have four candidates
            # after its own entry is excluded.
            if class_memory.numel() <= self.per_class_k:
                raise ValueError(
                    f"Class {class_index} has too few train-only cases for self-excluded top-4"
                )

            # Selection uses a two-dimensional N_query x N_train_class matrix.
            # No N x N x D difference tensor is formed.
            detached_query = keys.detach()
            detached_memory = keys[class_memory].detach()
            detached_similarity = detached_query @ detached_memory.T
            detached_distance_sq = (
                detached_query.square().sum(dim=1, keepdim=True)
                + detached_memory.square().sum(dim=1).unsqueeze(0)
                - 2.0 * detached_similarity
            ).clamp_min_(0.0)
            detached_score = -detached_distance_sq / (2.0 * self.temperature)
            self_mask = query_indices[:, None] == class_memory[None, :]
            detached_score = detached_score.masked_fill(self_mask, float("-inf"))
            local_topk = torch.topk(
                detached_score,
                k=self.per_class_k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            selected_indices = class_memory[local_topk]

            # Recompute scores from live keys only for the twelve selected
            # neighbors, allowing gradients into W_key while top-k stays discrete.
            selected_keys = keys[selected_indices]
            live_distance_sq = (
                keys[:, None, :] - selected_keys
            ).square().sum(dim=-1)
            selected_score = -live_distance_sq / (2.0 * self.temperature)

            selected_indices_by_class.append(selected_indices)
            selected_labels_by_class.append(
                torch.full_like(selected_indices, class_index)
            )
            selected_scores_by_class.append(selected_score)

        candidate_indices = torch.cat(selected_indices_by_class, dim=1)
        candidate_labels = torch.cat(selected_labels_by_class, dim=1)
        selected_score = torch.cat(selected_scores_by_class, dim=1)
        attention = torch.softmax(selected_score, dim=1)

        centered_embedding = self.label_embedding.weight - self.label_embedding.weight.mean(
            dim=0, keepdim=True
        )
        candidate_keys = keys[candidate_indices]
        query_minus_candidate = keys[:, None, :] - candidate_keys
        values = centered_embedding[candidate_labels] + F.gelu(
            self.diff_projection(query_minus_candidate)
        )
        context = (attention.unsqueeze(-1) * values).sum(dim=1)

        raw_residual = self.output_projection(context)
        h0_norm = h0.detach().norm(dim=-1, keepdim=True)
        raw_norm = raw_residual.norm(dim=-1, keepdim=True)
        cap_scale = torch.clamp(
            self.residual_cap * h0_norm / (raw_norm + 1e-8), max=1.0
        )
        residual = raw_residual * cap_scale
        residual_ratio = residual.norm(dim=-1) / (h0_norm.squeeze(-1) + 1e-8)

        memory_mask = torch.zeros(
            subject_count, dtype=torch.bool, device=h0.device
        )
        memory_mask[train_indices] = True
        candidate_is_train = memory_mask[candidate_indices]
        train_self_hits = (
            candidate_indices[train_indices] == train_indices[:, None]
        ).sum()
        candidate_class_counts = torch.stack(
            [(candidate_labels == class_index).sum(dim=1) for class_index in range(3)],
            dim=1,
        )
        attention_class_mass = torch.stack(
            [
                attention.masked_fill(candidate_labels != class_index, 0.0).sum(dim=1)
                for class_index in range(3)
            ],
            dim=1,
        )
        attention_entropy = -(
            attention * attention.clamp_min(1e-12).log()
        ).sum(dim=1)

        intermediates: dict[str, torch.Tensor | int | bool] = {
            "memory_train_indices": train_indices,
            "candidate_indices": candidate_indices,
            "candidate_labels": candidate_labels,
            "candidate_class_counts": candidate_class_counts,
            "candidate_is_train": candidate_is_train,
            "train_self_hit_count": train_self_hits,
            "test_candidate_count": (~candidate_is_train).sum(),
            "keys": keys,
            "selected_score": selected_score,
            "attention": attention,
            "attention_class_mass": attention_class_mass,
            "attention_entropy": attention_entropy,
            "effective_neighbor_count": attention_entropy.exp(),
            "retrieval_context": context,
            "R_raw": raw_residual,
            "R_retrieval": residual,
            "retrieval_cap_scale": cap_scale.squeeze(-1),
            "retrieval_cap_saturated": cap_scale.squeeze(-1) < (1.0 - 1e-7),
            "retrieval_residual_h0_ratio": residual_ratio,
        }
        return residual, intermediates


class CBTRC1Model(CMEDualBranchModel):
    """Exact C1 with a train-only retrieval residual before DIFFormer."""

    def __init__(
        self,
        *args,
        key_dim: int = 16,
        value_dim: int = 16,
        retrieval_k: int = 4,
        retrieval_temperature: float = 0.2,
        retrieval_residual_cap: float = 0.10,
        **kwargs,
    ):
        kwargs.pop("cme_arm", None)
        super().__init__(*args, cme_arm="c1", **kwargs)

        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else None
        )
        try:
            self.cbtr = ClassBalancedTrainOnlyRetrieval(
                self.Hidden_size,
                key_dim=key_dim,
                value_dim=value_dim,
                per_class_k=retrieval_k,
                temperature=retrieval_temperature,
                residual_cap=retrieval_residual_cap,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._active_train_idx: torch.Tensor | None = None
        self._active_train_y: torch.Tensor | None = None
        self._retrieval_enabled = True
        self._last_retrieval_intermediates: dict | None = None

    def c1_parameters(self):
        retrieval_ids = {id(parameter) for parameter in self.cbtr.parameters()}
        return (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in retrieval_ids
        )

    def retrieval_parameters(self):
        return self.cbtr.parameters()

    def _fuse_semantic_branches(self, category_embedding, global_embedding):
        h0 = super()._fuse_semantic_branches(
            category_embedding, global_embedding
        )
        if self._active_train_idx is None or self._active_train_y is None:
            raise RuntimeError("CBTR forward requires train_idx and aligned train_y")
        residual, retrieval_intermediates = self.cbtr(
            h0, self._active_train_idx, self._active_train_y
        )
        if self._retrieval_enabled:
            applied_residual = residual
        else:
            applied_residual = torch.zeros_like(residual)
            retrieval_intermediates = dict(retrieval_intermediates)
            retrieval_intermediates["R_retrieval_unforced"] = residual
            retrieval_intermediates["R_retrieval"] = applied_residual
            retrieval_intermediates["retrieval_residual_h0_ratio"] = torch.zeros_like(
                retrieval_intermediates["retrieval_residual_h0_ratio"]
            )
            retrieval_intermediates["retrieval_cap_saturated"] = torch.zeros_like(
                retrieval_intermediates["retrieval_cap_saturated"]
            )
        self._last_retrieval_intermediates = {
            "H0": h0,
            "retrieval_enabled": bool(self._retrieval_enabled),
            **retrieval_intermediates,
        }
        return h0 + applied_residual

    def forward(
        self,
        X_raw: torch.Tensor,
        train_idx: torch.Tensor,
        train_y: torch.Tensor,
        return_intermediates: bool = False,
        retrieval_enabled: bool = True,
        counterfactual_modal_index: int | None = None,
        counterfactual_sample_mask: torch.Tensor | None = None,
    ):
        previous_idx = self._active_train_idx
        previous_y = self._active_train_y
        previous_enabled = self._retrieval_enabled
        self._active_train_idx = train_idx
        self._active_train_y = train_y
        self._retrieval_enabled = bool(retrieval_enabled)
        try:
            output = super().forward(
                X_raw,
                return_intermediates=return_intermediates,
                counterfactual_modal_index=counterfactual_modal_index,
                counterfactual_sample_mask=counterfactual_sample_mask,
            )
        finally:
            self._active_train_idx = previous_idx
            self._active_train_y = previous_y
            self._retrieval_enabled = previous_enabled

        if not return_intermediates:
            self._last_retrieval_intermediates = None
            return output
        raw_logits, label_embeddings, auxiliary_outputs, intermediates = output
        if self._last_retrieval_intermediates is None:
            raise RuntimeError("CBTR intermediates are missing")
        intermediates.update(self._last_retrieval_intermediates)
        self._last_retrieval_intermediates = None
        return raw_logits, label_embeddings, auxiliary_outputs, intermediates
