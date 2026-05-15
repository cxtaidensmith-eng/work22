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


def _row_normalize_adj(adj, eps=1e-12):
    adj = adj.clamp_min(0.0)
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / degree


class DenseAdvDIFFormerConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, beta=0.5, K_order=3):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.beta = float(beta)
        self.K_order = int(K_order)
        self.Wq = nn.Linear(in_dim, in_dim * num_heads)
        self.Wk = nn.Linear(in_dim, in_dim * num_heads)
        self.Wo = nn.Linear(in_dim * num_heads * (self.K_order + 1), out_dim)

    @staticmethod
    def _gcn(x, adj_norm):
        return torch.einsum("nm,mhd->nhd", adj_norm, x)

    @staticmethod
    def _attn(qs, ks, vs, eps=1e-12):
        n = qs.shape[0]
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        qkvs = torch.einsum("nhm,hmd->nhd", qs, kvs)
        vs_sum = vs.sum(dim=0).unsqueeze(0).expand(n, -1, -1)
        num = qkvs + vs_sum
        ks_sum = ks.sum(dim=0)
        den = torch.einsum("nhm,hm->nh", qs, ks_sum).unsqueeze(-1) + n
        return num / den.clamp_min(eps)

    def forward(self, x, adj_norm, eps=1e-12):
        q = self.Wq(x).reshape(-1, self.num_heads, self.in_dim)
        k = self.Wk(x).reshape(-1, self.num_heads, self.in_dim)
        qs = q / torch.norm(q, p=2, dim=2, keepdim=True).clamp_min(eps)
        ks = k / torch.norm(k, p=2, dim=2, keepdim=True).clamp_min(eps)

        x_in = x.unsqueeze(1).expand(-1, self.num_heads, -1)
        x_list = [x_in]
        for _ in range(self.K_order):
            attn_i = self._attn(qs, ks, x_list[-1])
            gcn_i = self._gcn(x_list[-1], adj_norm)
            x_list.append(self.beta * gcn_i + attn_i)

        x_concat = torch.cat(x_list, dim=-1).reshape(
            -1, self.num_heads * self.in_dim * (self.K_order + 1)
        )
        return self.Wo(x_concat) / self.num_heads


class DenseDIFFormerConv(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=1,
        kernel='simple',
        graph_weight=0.5,
        use_graph=True,
        use_source=False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.kernel = kernel
        self.graph_weight = float(graph_weight)
        self.use_graph = use_graph
        self.use_source = use_source
        self.Wq = nn.Linear(in_dim, out_dim * num_heads)
        self.Wk = nn.Linear(in_dim, out_dim * num_heads)
        self.Wv = nn.Linear(in_dim, out_dim * num_heads)

    @staticmethod
    def _gcn(x, adj_norm):
        return torch.einsum("nm,mhd->nhd", adj_norm, x)

    @staticmethod
    def _simple_attn(qs, ks, vs, eps=1e-12):
        n = qs.shape[0]
        qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True).clamp_min(eps)
        ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True).clamp_min(eps)
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        qkvs = torch.einsum("nhm,hmd->nhd", qs, kvs)
        vs_sum = vs.sum(dim=0).unsqueeze(0).expand(n, -1, -1)
        ks_sum = ks.sum(dim=0)
        den = torch.einsum("nhm,hm->nh", qs, ks_sum).unsqueeze(-1) + n
        return (qkvs + vs_sum) / den.clamp_min(eps)

    @staticmethod
    def _sigmoid_attn(qs, ks, vs, eps=1e-12):
        attn = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))
        den = attn.sum(dim=1, keepdim=True).clamp_min(eps)
        attn = attn / den
        return torch.einsum("nlh,lhd->nhd", attn, vs)

    def forward(self, x, adj_norm):
        q = self.Wq(x).reshape(-1, self.num_heads, self.out_dim)
        k = self.Wk(x).reshape(-1, self.num_heads, self.out_dim)
        v = self.Wv(x).reshape(-1, self.num_heads, self.out_dim)

        if self.kernel == 'sigmoid':
            attn_out = self._sigmoid_attn(q, k, v)
        else:
            attn_out = self._simple_attn(q, k, v)

        if self.use_graph:
            gcn_out = self._gcn(v, adj_norm)
            if self.graph_weight >= 0:
                out = (1.0 - self.graph_weight) * attn_out + self.graph_weight * gcn_out
            else:
                out = attn_out + gcn_out
        else:
            out = attn_out

        out = out.mean(dim=1)
        if self.use_source:
            out = out + x
        return out


class DIFFormer_GraphHead(nn.Module):
    def __init__(
        self,
        Dim_emb,
        hidden,
        out_channels,
        P,
        num_layers=1,
        num_heads=2,
        graph_weight=0.5,
        alpha=0.5,
        kernel='simple',
        use_graph=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(Dim_emb, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            DenseDIFFormerConv(
                hidden,
                hidden,
                num_heads=num_heads,
                kernel=kernel,
                graph_weight=graph_weight,
                use_graph=use_graph,
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden, out_channels)
        self.dropout = P
        self.alpha = float(alpha)
        self.GCN_feature_1 = None
        self.GCN_feature_2 = None
        self.last_adj = None

    def forward(self, x, gso):
        adj_norm = _row_normalize_adj(gso)
        self.last_adj = adj_norm
        h = F.relu(self.input_norm(self.input_proj(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        layer_cache = [h]
        for idx, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            h_new = layer(h, adj_norm)
            h = self.alpha * h_new + (1.0 - self.alpha) * layer_cache[idx]
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_cache.append(h)

        self.GCN_feature_1 = h
        logits = self.classifier(h)
        self.GCN_feature_2 = logits
        return logits


class AdvDIFFormer_GraphHead(nn.Module):
    def __init__(
        self,
        Dim_emb,
        hidden,
        out_channels,
        P,
        num_layers=1,
        num_heads=2,
        beta=0.5,
        K_order=3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(Dim_emb, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [
                DenseAdvDIFFormerConv(
                    hidden, hidden, num_heads=num_heads, beta=beta, K_order=K_order
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = P
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(P),
            nn.Linear(hidden, out_channels),
        )
        self.GCN_feature_1 = None
        self.GCN_feature_2 = None
        self.last_adj = None

    def forward(self, x, gso):
        adj_norm = _row_normalize_adj(gso)
        self.last_adj = adj_norm
        h = F.gelu(self.input_norm(self.input_proj(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer, norm in zip(self.layers, self.layer_norms):
            h = norm(h + layer(h, adj_norm))
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        self.GCN_feature_1 = h
        logits = self.classifier(h)
        self.GCN_feature_2 = logits
        return logits
     
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
