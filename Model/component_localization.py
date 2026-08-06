from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int = 96, classes: int = 3):
        super().__init__()
        self.classifier = nn.Linear(input_dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class ResidualMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 48,
        classes: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_dropout(F.gelu(self.input_norm(self.input_projection(features))))
        hidden = hidden + self.residual_ffn(self.residual_norm(hidden))
        return self.classifier(self.output_norm(hidden))


class BatchEnsembleLinear(nn.Module):
    """Shared linear weights with member-specific rank-one scales and bias."""

    def __init__(self, input_dim: int, output_dim: int, members: int = 4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.members = int(members)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.r = nn.Parameter(torch.ones(members, input_dim))
        self.s = nn.Parameter(torch.ones(members, output_dim))
        self.bias = nn.Parameter(torch.zeros(members, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.r.fill_(1.0)
            self.s.fill_(1.0)
            if self.members > 1:
                self.r[1:].normal_(mean=1.0, std=0.02)
                self.s[1:].normal_(mean=1.0, std=0.02)
            self.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).expand(self.members, -1, -1)
        if inputs.ndim != 3 or inputs.shape[0] != self.members:
            raise ValueError(
                f"Expected [N,{self.input_dim}] or [{self.members},N,{self.input_dim}], "
                f"got {tuple(inputs.shape)}"
            )
        scaled_inputs = inputs * self.r[:, None, :]
        outputs = torch.einsum("mni,oi->mno", scaled_inputs, self.weight)
        return outputs * self.s[:, None, :] + self.bias[:, None, :]


class PEEHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        members: int = 4,
        classes: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.members = int(members)
        self.layer1 = BatchEnsembleLinear(input_dim, 64, members)
        self.norm1 = nn.LayerNorm(64)
        self.layer2 = BatchEnsembleLinear(64, 32, members)
        self.norm2 = nn.LayerNorm(32)
        self.layer3 = BatchEnsembleLinear(32, classes, members)
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(features)
        hidden = F.dropout(F.gelu(self.norm1(hidden)), p=self.dropout, training=self.training)
        hidden = self.layer2(hidden)
        hidden = F.dropout(F.gelu(self.norm2(hidden)), p=self.dropout, training=self.training)
        return self.layer3(hidden)

    @staticmethod
    def ensemble_probability(member_logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(member_logits, dim=-1).mean(dim=0)


def build_mutual_knn_graph(features: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Build the fixed non-negative, symmetric mutual-kNN normalized graph."""
    if features.ndim != 2:
        raise ValueError(f"Expected two-dimensional features, got {tuple(features.shape)}")
    node_count = int(features.shape[0])
    if not 0 < top_k < node_count:
        raise ValueError(f"top_k must be in [1,{node_count - 1}], got {top_k}")
    normalized = F.normalize(F.layer_norm(features, (features.shape[-1],)), dim=-1)
    similarity = normalized @ normalized.T
    similarity = similarity.clone()
    similarity.fill_diagonal_(-torch.inf)
    _, neighbor_indices = torch.topk(similarity, k=top_k, dim=1)
    directed = torch.zeros_like(similarity, dtype=torch.bool)
    directed.scatter_(1, neighbor_indices, True)
    mutual = directed & directed.T
    weights = torch.where(mutual, similarity.clamp_min(0.0), torch.zeros_like(similarity))
    weights = 0.5 * (weights + weights.T)
    weights = weights + torch.eye(node_count, device=features.device, dtype=features.dtype)
    degree = weights.sum(dim=1).clamp_min(1e-12)
    inv_sqrt = degree.rsqrt()
    graph = inv_sqrt[:, None] * weights * inv_sqrt[None, :]
    if not torch.allclose(graph, graph.T, atol=1e-6, rtol=0.0):
        raise RuntimeError("Mutual-kNN normalized adjacency is not symmetric")
    return graph.detach()


class SparseResidualGCNHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 48,
        classes: int = 3,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gamma_parameter = nn.Parameter(torch.zeros(()))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, classes)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.clamp(self.gamma_parameter, 0.0, 0.5)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_norm(self.input_projection(features)))
        graph_hidden = adjacency @ self.graph_projection(hidden)
        refined = self.output_norm(hidden + self.gamma * graph_hidden)
        return self.classifier(refined)


class RawFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        classes: int = 3,
        dropout: float = 0.67,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def pee_pairwise_diagnostics(member_probability: torch.Tensor) -> dict:
    if member_probability.ndim != 3:
        raise ValueError("member_probability must be [members,samples,classes]")
    predictions = member_probability.argmax(dim=-1)
    disagreement = []
    cosine = []
    symmetric_kl = []
    for left, right in itertools.combinations(range(member_probability.shape[0]), 2):
        disagreement.append((predictions[left] != predictions[right]).float().mean())
        cosine.append(
            F.cosine_similarity(member_probability[left], member_probability[right], dim=-1).mean()
        )
        p = member_probability[left].clamp_min(1e-12)
        q = member_probability[right].clamp_min(1e-12)
        kl_pq = (p * (p.log() - q.log())).sum(dim=-1).mean()
        kl_qp = (q * (q.log() - p.log())).sum(dim=-1).mean()
        symmetric_kl.append(0.5 * (kl_pq + kl_qp))
    return {
        "pairwise_prediction_disagreement": float(torch.stack(disagreement).mean()),
        "pairwise_probability_cosine": float(torch.stack(cosine).mean()),
        "pairwise_symmetric_kl": float(torch.stack(symmetric_kl).mean()),
    }
