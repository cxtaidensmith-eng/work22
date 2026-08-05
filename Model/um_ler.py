from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGuidedLocalEvidenceRefinement(nn.Module):
    """Train-only local evidence refinement over detached modal tokens."""

    def __init__(self, hidden_size: int, projection_size: int = 32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.modal_norm = nn.LayerNorm(self.hidden_size)
        self.modal_projector = nn.Linear(self.hidden_size, self.projection_size)
        self.modal_reliability = nn.Linear(self.hidden_size, 1)

    def encode(self, modal_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if modal_tokens.ndim != 3:
            raise ValueError(
                f"Expected modal tokens [N,M,D], got {tuple(modal_tokens.shape)}"
            )
        if modal_tokens.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {modal_tokens.size(-1)}"
            )
        normalized = self.modal_norm(modal_tokens.detach())
        projected = F.normalize(
            self.modal_projector(normalized), dim=-1, eps=1e-8
        )
        reliability = torch.softmax(
            self.modal_reliability(normalized).squeeze(-1), dim=1
        )
        weighted = torch.sqrt(reliability.clamp_min(1e-12)).unsqueeze(-1) * projected
        retrieval = F.normalize(weighted.flatten(start_dim=1), dim=-1, eps=1e-8)
        return retrieval, reliability

    def local_probabilities(
        self,
        retrieval: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
        class_count: int,
    ) -> dict[str, torch.Tensor]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        train_indices = torch.where(train_mask)[0]
        if train_indices.numel() <= top_k:
            raise ValueError("Training fold is too small for leave-one-out top-k")

        # The only label memory constructed by this module is train_labels.
        train_labels = labels.index_select(0, train_indices)
        train_memory = retrieval.index_select(0, train_indices)
        similarities = retrieval @ train_memory.transpose(0, 1)

        # train_indices are ordered exactly like train_memory, so this masks
        # every training query's own memory slot and nothing else.
        memory_positions = torch.arange(
            train_indices.numel(), device=retrieval.device
        )
        similarities = similarities.clone()
        similarities[train_indices, memory_positions] = -torch.inf

        top_values, top_positions = torch.topk(
            similarities, k=int(top_k), dim=1, largest=True, sorted=True
        )
        neighbor_weights = torch.softmax(top_values / float(temperature), dim=1)
        neighbor_global_indices = train_indices.index_select(
            0, top_positions.reshape(-1)
        ).reshape_as(top_positions)
        neighbor_labels = train_labels.index_select(
            0, top_positions.reshape(-1)
        ).reshape_as(top_positions)

        train_counts = torch.bincount(
            train_labels, minlength=int(class_count)
        ).to(dtype=retrieval.dtype)
        if bool((train_counts <= 0).any()):
            raise ValueError("Every class must be present in the training fold")
        train_prior = train_counts / train_counts.sum()
        one_hot = F.one_hot(
            neighbor_labels, num_classes=int(class_count)
        ).to(dtype=retrieval.dtype)
        weighted_mass = (
            neighbor_weights.unsqueeze(-1) * one_hot
        ).sum(dim=1)
        evidence = weighted_mass / torch.sqrt(train_prior).view(1, -1)
        # A tiny smoothing floor keeps NCA finite and differentiable when a
        # hard top-k neighborhood happens to omit the query's true class.
        evidence = evidence + 1e-8
        local_probability = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        train_neighbor_labels = neighbor_labels.index_select(0, train_indices)
        train_query_labels = train_labels.view(-1, 1)
        train_neighbor_agreement = (
            train_neighbor_labels == train_query_labels
        ).to(dtype=retrieval.dtype)
        weighted_train_agreement = (
            neighbor_weights.index_select(0, train_indices)
            * train_neighbor_agreement
        ).sum(dim=1)
        unweighted_train_agreement = train_neighbor_agreement.mean(dim=1)

        return {
            "q": local_probability,
            "train_indices": train_indices,
            "train_labels": train_labels,
            "train_prior": train_prior,
            "neighbor_positions": top_positions,
            "neighbor_indices": neighbor_global_indices,
            "neighbor_labels": neighbor_labels,
            "neighbor_weights": neighbor_weights,
            "neighbor_similarities": top_values,
            "weighted_train_neighbor_agreement": weighted_train_agreement,
            "unweighted_train_neighbor_agreement": unweighted_train_agreement,
        }

    @staticmethod
    def refine(
        original_probability: torch.Tensor,
        local_probability: torch.Tensor,
        *,
        margin_threshold: float,
        neighbor_confidence_threshold: float,
        gate_cap: float,
    ) -> dict[str, torch.Tensor]:
        if margin_threshold <= 0:
            raise ValueError("margin_threshold must be positive")
        if not 0 <= neighbor_confidence_threshold <= 1:
            raise ValueError("neighbor_confidence_threshold must be in [0,1]")
        if not 0 <= gate_cap <= 1:
            raise ValueError("gate_cap must be in [0,1]")

        probability = original_probability.detach()
        top_two = torch.topk(probability, k=2, dim=-1).values
        margin = top_two[:, 0] - top_two[:, 1]
        uncertainty = torch.clamp(
            (float(margin_threshold) - margin) / float(margin_threshold),
            min=0.0,
            max=1.0,
        )
        entropy = -(
            local_probability.clamp_min(1e-12)
            * local_probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        reliability = torch.clamp(
            1.0 - entropy / math.log(local_probability.size(-1)),
            min=0.0,
            max=1.0,
        )
        local_max = local_probability.max(dim=-1).values
        eligible = local_max >= float(neighbor_confidence_threshold)
        reliability = torch.where(
            eligible, reliability, torch.zeros_like(reliability)
        )
        gate = torch.clamp(
            uncertainty * reliability, min=0.0, max=float(gate_cap)
        )
        final_probability = (
            (1.0 - gate.unsqueeze(-1)) * probability
            + gate.unsqueeze(-1) * local_probability
        )
        return {
            "p": probability,
            "q": local_probability,
            "p_final": final_probability,
            "margin": margin,
            "uncertainty": uncertainty,
            "local_entropy": entropy,
            "local_reliability": reliability,
            "local_max": local_max,
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
        margin_threshold: float = 0.30,
        neighbor_confidence_threshold: float = 0.70,
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
        refined = self.refine(
            original_probability,
            local["q"],
            margin_threshold=margin_threshold,
            neighbor_confidence_threshold=neighbor_confidence_threshold,
            gate_cap=gate_cap,
        )
        return {
            **local,
            **refined,
            "retrieval": retrieval,
            "modal_reliability": modal_reliability,
        }
