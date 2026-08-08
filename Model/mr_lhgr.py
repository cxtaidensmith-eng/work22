"""Modality-specific multi-relation low/high-frequency residual for C1.

The complete historical C1 model is constructed first.  Six fixed, label-free
patient graphs then filter the C1 category tokens.  Only the resulting bounded
residual is added to H0=Y+G before the untouched DIFFormer/classifier.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cme_dual_branch import CMEDualBranchModel


C1_PARAMETER_COUNT = 862_971
MR_LHGR_ADDED_PARAMETERS_H96_R8 = 10_752
MR_LHGR_TOTAL_PARAMETERS_H96_R8 = 873_723


class IdentityCenteredRelationResidual(nn.Module):
    """Six equal-weight, complementary low/high relation channels."""

    def __init__(
        self,
        hidden_size: int,
        relation_graphs: Sequence[torch.Tensor],
        rank: int = 8,
        residual_cap: float = 0.10,
    ):
        super().__init__()
        if len(relation_graphs) != 6:
            raise ValueError(f"Expected six modality graphs, got {len(relation_graphs)}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.residual_cap = float(residual_cap)
        if self.rank != 8:
            raise ValueError("MR-LHGR v1 fixes rank=8")
        if not 0.0 < self.residual_cap <= 1.0:
            raise ValueError("Invalid graph residual cap")

        self._graph_buffer_names: list[str] = []
        graph_size = None
        for index, graph in enumerate(relation_graphs):
            if graph.layout != torch.sparse_coo:
                raise ValueError("Relation graphs must be sparse COO tensors")
            graph = graph.coalesce().detach()
            if graph.ndim != 2 or graph.size(0) != graph.size(1):
                raise ValueError(f"Invalid graph shape: {tuple(graph.shape)}")
            graph_size = graph.size(0) if graph_size is None else graph_size
            if graph.size(0) != graph_size:
                raise ValueError("All relation graphs must have the same node count")
            name = f"relation_graph_{index}"
            # Graphs are deterministic fixed inputs, not checkpoint state.  Making
            # them non-persistent also keeps ModelEMA away from sparse buffers.
            self.register_buffer(name, graph, persistent=False)
            self._graph_buffer_names.append(name)
        self.graph_size = int(graph_size)

        self.low_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.rank, bias=False) for _ in range(6)]
        )
        self.high_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.rank, bias=False) for _ in range(6)]
        )
        self.output_projection = nn.Linear(
            2 * self.rank, self.hidden_size, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    @property
    def relation_graphs(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, name) for name in self._graph_buffer_names)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        h0: torch.Tensor,
        *,
        identity_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        if modality_tokens.ndim != 3:
            raise ValueError(
                f"Expected modality tokens [N,6,D], got {tuple(modality_tokens.shape)}"
            )
        batch_size, modality_count, hidden_size = modality_tokens.shape
        if batch_size != self.graph_size:
            raise ValueError(
                f"Graph/token subject mismatch: graph={self.graph_size}, tokens={batch_size}"
            )
        if modality_count != 6 or hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected [N,6,{self.hidden_size}], got {tuple(modality_tokens.shape)}"
            )
        if tuple(h0.shape) != (batch_size, self.hidden_size):
            raise ValueError(f"Unexpected H0 shape: {tuple(h0.shape)}")

        delta_low, delta_high = [], []
        low_response, high_response = [], []
        for modality_index, (low_projection, high_projection) in enumerate(
            zip(self.low_projections, self.high_projections)
        ):
            token = modality_tokens[:, modality_index]
            normalized = F.layer_norm(
                token, (self.hidden_size,), weight=None, bias=None, eps=1e-5
            )
            if identity_graph:
                neighbor = normalized
            else:
                graph = self.relation_graphs[modality_index]
                neighbor = torch.sparse.mm(graph, normalized)
            low = 0.5 * (normalized + neighbor)
            high = 0.5 * (normalized - neighbor)
            low_increment = F.gelu(low_projection(low)) - F.gelu(
                low_projection(normalized)
            )
            zeros = torch.zeros_like(normalized)
            high_increment = F.gelu(high_projection(high)) - F.gelu(
                high_projection(zeros)
            )
            delta_low.append(low_increment)
            delta_high.append(high_increment)
            low_response.append(low)
            high_response.append(high)

        delta_low_tensor = torch.stack(delta_low, dim=1)
        delta_high_tensor = torch.stack(delta_high, dim=1)
        low_mean = delta_low_tensor.mean(dim=1)
        high_mean = delta_high_tensor.mean(dim=1)
        relation_code = torch.cat((low_mean, high_mean), dim=-1)
        raw_residual = self.output_projection(relation_code)

        h_norm = h0.detach().norm(dim=-1, keepdim=True)
        raw_norm = raw_residual.norm(dim=-1, keepdim=True)
        cap_scale = torch.clamp(
            self.residual_cap * h_norm / (raw_norm + 1e-8), max=1.0
        )
        graph_residual = cap_scale * raw_residual
        ratio = graph_residual.norm(dim=-1) / (h_norm.squeeze(-1) + 1e-8)
        # Report saturation from the scale that is actually applied.  The
        # small tolerance avoids counting float32 round-off at exactly one.
        cap_saturated = cap_scale.squeeze(-1) < (1.0 - 1e-7)

        intermediates: dict[str, torch.Tensor | bool] = {
            "graph_identity_mode": bool(identity_graph),
            "relation_tokens": modality_tokens,
            "relation_low_response": torch.stack(low_response, dim=1),
            "relation_high_response": torch.stack(high_response, dim=1),
            "relation_delta_low": delta_low_tensor,
            "relation_delta_high": delta_high_tensor,
            "relation_code": relation_code,
            "R_raw": raw_residual,
            "R_graph": graph_residual,
            "graph_cap_scale": cap_scale.squeeze(-1),
            "graph_cap_saturated": cap_saturated,
            "graph_residual_h0_ratio": ratio,
        }
        return graph_residual, intermediates


class MRLHGRC1Model(CMEDualBranchModel):
    """Exact C1 plus an identity-centered fixed-relation residual."""

    def __init__(
        self,
        *args,
        relation_graphs: Sequence[torch.Tensor],
        graph_rank: int = 8,
        graph_residual_cap: float = 0.10,
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
            self.mr_lhgr = IdentityCenteredRelationResidual(
                self.Hidden_size,
                relation_graphs,
                rank=graph_rank,
                residual_cap=graph_residual_cap,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._pending_relation_tokens: torch.Tensor | None = None
        self._last_relation_intermediates: dict | None = None
        self._identity_graph_override = False

    def c1_parameters(self):
        graph_parameter_ids = {id(parameter) for parameter in self.mr_lhgr.parameters()}
        return (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in graph_parameter_ids
        )

    def graph_parameters(self):
        return self.mr_lhgr.parameters()

    def _category_token_streams(self, tokens: torch.Tensor):
        streams, extras = super()._category_token_streams(tokens)
        relation_tokens = streams[0]
        if relation_tokens.ndim != 3 or relation_tokens.size(1) != 6:
            raise RuntimeError(
                f"Invalid C1 relation-token shape: {tuple(relation_tokens.shape)}"
            )
        self._pending_relation_tokens = relation_tokens
        return streams, extras

    def _fuse_semantic_branches(self, category_embedding, global_embedding):
        h0 = super()._fuse_semantic_branches(category_embedding, global_embedding)
        relation_tokens = self._pending_relation_tokens
        self._pending_relation_tokens = None
        if relation_tokens is None:
            raise RuntimeError("C1 relation tokens were not produced before semantic fusion")
        graph_residual, graph_intermediates = self.mr_lhgr(
            relation_tokens,
            h0,
            identity_graph=self._identity_graph_override,
        )
        self._last_relation_intermediates = {
            "H0": h0,
            **graph_intermediates,
        }
        return h0 + graph_residual

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
        counterfactual_modal_index: int | None = None,
        counterfactual_sample_mask: torch.Tensor | None = None,
        identity_graph: bool = False,
    ):
        previous_identity = self._identity_graph_override
        self._identity_graph_override = bool(identity_graph)
        try:
            output = super().forward(
                X_raw,
                return_intermediates=return_intermediates,
                counterfactual_modal_index=counterfactual_modal_index,
                counterfactual_sample_mask=counterfactual_sample_mask,
            )
        finally:
            self._identity_graph_override = previous_identity

        if not return_intermediates:
            self._last_relation_intermediates = None
            return output
        raw_logits, label_embeddings, auxiliary_outputs, intermediates = output
        if self._last_relation_intermediates is None:
            raise RuntimeError("MR-LHGR intermediates are missing")
        intermediates.update(self._last_relation_intermediates)
        self._last_relation_intermediates = None
        return raw_logits, label_embeddings, auxiliary_outputs, intermediates
