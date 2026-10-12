from torch import nn
import torch
import torch.nn.functional as F

from .layers import ChebGraphConv


class Cheb_GCN(nn.Module):
    """Two-layer Chebyshev graph classifier used by MSCQ-GN."""

    def __init__(self, Dim_emb, hidden, out_channels, P, K):
        super().__init__()
        self.conv1 = ChebGraphConv(in_features=Dim_emb, out_features=hidden, K=K)
        self.dropout = nn.Dropout(p=P)
        self.conv2 = ChebGraphConv(in_features=hidden, out_features=out_channels, K=K)

    def forward(self, x, gso):
        gso = self.torch_compute_gso(gso)
        x = self.conv1(x=x, gso=gso)
        self.GCN_feature_1 = x
        x = F.relu(self.dropout(x))
        x = self.conv2(x=x, gso=gso)
        self.GCN_feature_2 = x
        return x

    @staticmethod
    def torch_compute_gso(adj):
        num_nodes = adj.shape[0]
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree.clamp(min=1e-8), -0.5)
        norm = torch.diag(degree_inv_sqrt)
        identity = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
        laplacian = identity - norm @ adj @ norm
        laplacian = torch.nan_to_num(laplacian, nan=0.0)
        return laplacian - identity


class graph_learning(nn.Module):
    """Learn the dense subject graph from gated subject features."""

    def __init__(self, DATASET_Dict, Hidden_size, rate=0.1, use_raw_x=True):
        super().__init__()
        self.rate = rate
        prior = DATASET_Dict['Adj'] * 0.9 + (1 - DATASET_Dict['Adj']) * 0.1
        upper = nn.Parameter(prior.triu(1), requires_grad=True)
        diagonal = nn.Parameter(prior.diag(), requires_grad=True)
        self.Ws_Parameter = upper + upper.t() + diagonal.diag()
        self.layer_ = (
            nn.Linear(DATASET_Dict['In_Channels'], Hidden_size)
            if use_raw_x else nn.Identity()
        )
        self.relu = nn.ReLU() if rate == 0 else nn.LeakyReLU(negative_slope=rate)

    def forward(self, x):
        projected = F.normalize(self.layer_(x), dim=-1)
        cosine = self.relu(projected @ projected.T)
        scaled = (cosine + self.rate) / (1.0 + self.rate)
        prior = torch.clamp(self.Ws_Parameter, min=0, max=1)
        return torch.sigmoid(scaled * prior)
