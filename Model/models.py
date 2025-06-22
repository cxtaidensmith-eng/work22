from .layers import ChebGraphConv, GraphConvolution
from torch import nn
import torch
import torch.nn.functional as F

class Cheb_GCN(nn.Module):
    def __init__(self, Dim_emb, hidden, out_channels, P, K):
        super(Cheb_GCN, self).__init__()
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

    def torch_compute_gso(self, adj):
        num_nodes = adj.shape[0]
        d = torch.sum(adj, axis=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = torch.eye(num_nodes, device=adj.device) - torch.matmul(torch.matmul(d_mat_inv_sqrt, adj),
                                                                           d_mat_inv_sqrt)
        if torch.isnan(laplacian).any():
            laplacian = torch.where(torch.isnan(laplacian), torch.zeros_like(laplacian), laplacian)

        eigv_max = torch.max(torch.linalg.eigvals(laplacian).abs())
        gso = (2.0 / eigv_max) * laplacian - torch.eye(laplacian.shape[0], device=laplacian.device)

        return gso
    
class graph_learning(nn.Module):
    def __init__(self, DATASET_Dict, Hidden_size, rate=0.1):
        super(graph_learning, self).__init__()

        self.adj_ = DATASET_Dict['Adj']
        self.Hidden_size = Hidden_size
        self.rate = rate
        self.sample_num = DATASET_Dict['Sample_Num']
        self.In_channels = DATASET_Dict['In_Channels']

        self.W = self.adj_ * 0.9 + (1 - self.adj_) * 0.1
        W_upper_triangle = torch.nn.Parameter(self.W.triu(1), requires_grad=True)
        W_diag = torch.nn.Parameter(self.W.diag(), requires_grad=True)
        self.Ws_Parameter = W_upper_triangle + W_upper_triangle.t() + W_diag.diag()

        # self.attention_adj = nn.MultiheadAttention(self.Hidden_size, 1, batch_first=True)
        self.layer_ = nn.Linear(self.In_channels, self.Hidden_size)

        if rate == 0:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(negative_slope=self.rate)

    def forward(self, X):

        # X_ = self.layer_(X)
        # X_T = torch.unsqueeze(X_, 0)
        # _, attention_sorce = self.attention_adj(X_T, X_T, X_T)
        # attention_sorce = torch.squeeze(attention_sorce, 0)
        # attention_sorce = (attention_sorce + attention_sorce.T) / 2
        # Adj_ = torch.squeeze(attention_sorce, 0) * torch.clamp(self.Ws_Parameter, min=0, max=1)

        X_ = self.layer_(X)
        x_norm = F.normalize(X_, dim=-1)
        Cos_sorce = self.relu(torch.mm(x_norm, x_norm.T))

        Adj_ = self.Tensor_To_0_1(Cos_sorce) * torch.clamp(self.Ws_Parameter, min=0, max=1)

        return torch.sigmoid(Adj_)

    def Tensor_To_0_1(self, tensor):

        return self.map_values_torch(tensor, [-1 * self.rate, 1], [0, 1])

    def map_values_torch(self, tensor, from_range, to_range):
        (a, b) = from_range
        (c, d) = to_range
        result = (tensor - a) / (b - a) * (d - c) + c
        return result

    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix, recurse)