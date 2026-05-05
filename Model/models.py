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
        """归一化 Laplacian: L = I - D^(-1/2) A D^(-1/2)
        其特征值上界为 2（Defferrard 2016 标准近似），直接令 eigv_max=2.0
        避免训练中 torch.linalg.eigvals 的反向传播数值不稳定（在 modal_gate 极端时会触发 singular solve）。
        Cheb 多项式期望 gso ∈ [-1, 1]，所以 gso = (2/2) L - I = L - I.
        """
        num_nodes = adj.shape[0]
        d = torch.sum(adj, axis=1)
        d_inv_sqrt = torch.pow(d.clamp(min=1e-8), -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = torch.eye(num_nodes, device=adj.device) - torch.matmul(torch.matmul(d_mat_inv_sqrt, adj),
                                                                           d_mat_inv_sqrt)
        if torch.isnan(laplacian).any():
            laplacian = torch.where(torch.isnan(laplacian), torch.zeros_like(laplacian), laplacian)

        gso = laplacian - torch.eye(laplacian.shape[0], device=laplacian.device)
        return gso
    
class graph_learning(nn.Module):
    """学习样本邻接矩阵。
    use_raw_x=True 时（旧行为）：内部 Linear(In_channels, Hidden_size) 把 raw X 投影后算 cosine 相似度。
    use_raw_x=False 时（v4 新行为）：直接接收已学到的 sample embedding (B, Hidden_size)，不再用 raw X。
        理由：raw X (360 维) 中 MRI/PET 弱信号特征 (288 维) 数量上压倒强信号 modal (60 维)，
        cos 相似度被弱信号主导。改用模型学到的 sample embedding（已经吸收了 modal 重要性）算 sim 更合理。
    """

    def __init__(self, DATASET_Dict, Hidden_size, rate=0.1, use_raw_x=True):
        super(graph_learning, self).__init__()

        self.adj_ = DATASET_Dict['Adj']
        self.Hidden_size = Hidden_size
        self.rate = rate
        self.sample_num = DATASET_Dict['Sample_Num']
        self.In_channels = DATASET_Dict['In_Channels']
        self.use_raw_x = use_raw_x

        self.W = self.adj_ * 0.9 + (1 - self.adj_) * 0.1
        W_upper_triangle = torch.nn.Parameter(self.W.triu(1), requires_grad=True)
        W_diag = torch.nn.Parameter(self.W.diag(), requires_grad=True)
        self.Ws_Parameter = W_upper_triangle + W_upper_triangle.t() + W_diag.diag()

        if use_raw_x:
            self.layer_ = nn.Linear(self.In_channels, self.Hidden_size)
        else:
            self.layer_ = nn.Identity()

        if rate == 0:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(negative_slope=self.rate)

    def forward(self, X):
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

