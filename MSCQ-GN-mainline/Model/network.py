from .models import Cheb_GCN, graph_learning
import torch
from torch import nn
import torch.nn.functional as F


class Feature_Modal_Embedding(nn.Module):
    """对原始 X 做轻量加性 modal-aware 调节（gain & bias 按特征所属模态共享）。
    输出形状仍为 (B, F)，对下游所有按 Modal_Index 取索引的 Linear 透明。
    """

    def __init__(self, Modal_Index, modal_num, feature_num):
        super(Feature_Modal_Embedding, self).__init__()

        feature_to_modal = torch.zeros(feature_num, dtype=torch.long)
        for modal_idx, feat_indices in enumerate(Modal_Index):
            feature_to_modal[feat_indices] = modal_idx
        self.register_buffer('feature_to_modal', feature_to_modal)

        self.modal_gain = nn.Embedding(modal_num, 1)
        self.modal_bias = nn.Embedding(modal_num, 1)
        nn.init.ones_(self.modal_gain.weight)
        nn.init.zeros_(self.modal_bias.weight)

    def forward(self, X):
        gain = self.modal_gain(self.feature_to_modal).squeeze(-1)
        bias = self.modal_bias(self.feature_to_modal).squeeze(-1)
        return X * gain + bias


def _modal_intermediate_dim(modal_size, hidden_size):
    """根据模态特征数选合适的中间瓶颈维度。
    避免小模态（如 CSF=3）被强制 expand 到 H=96 后又 collapse 回 H 造成的过参数化与噪声放大。
        modal_size 1-8   → hidden_size // 6   (e.g. 16 for H=96)
        modal_size 9-30  → hidden_size // 3   (e.g. 32)
        modal_size 31-100 → hidden_size * 2 / 3 (e.g. 64)
        modal_size > 100 → hidden_size       (e.g. 96)
    """
    if modal_size <= 8:
        return max(8, hidden_size // 6)
    if modal_size <= 30:
        return max(16, hidden_size // 3)
    if modal_size <= 100:
        return max(32, (hidden_size * 2) // 3)
    return hidden_size


class Modal_Token_Encoder(nn.Module):
    """每模态压缩成 1 个 hidden token。
    v4 改进：中间瓶颈维度 modal-adaptive（小模态用小 hidden，大模态用大 hidden），
    最终统一投影到 H 以便后续 attention 在统一维度上进行。
    """

    def __init__(self, Modal_Index, modal_num, hidden_size):
        super(Modal_Token_Encoder, self).__init__()
        self.Modal_Index = Modal_Index
        self.modal_num = modal_num
        self.hidden_size = hidden_size

        self.modal_proj = nn.ModuleList()
        for idx in Modal_Index:
            modal_size = len(idx)
            mid_dim = _modal_intermediate_dim(modal_size, hidden_size)
            self.modal_proj.append(nn.Sequential(
                nn.Linear(modal_size, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, hidden_size),
            ))

        self.modal_id_embedding = nn.Embedding(modal_num, hidden_size)
        nn.init.normal_(self.modal_id_embedding.weight, std=0.02)

        self.input_norm = nn.LayerNorm(hidden_size)

    def forward(self, X):
        device = X.device
        modal_tokens = []
        for m_idx, feat_indices in enumerate(self.Modal_Index):
            idx_t = torch.as_tensor(feat_indices, device=device, dtype=torch.long)
            x_m = X.index_select(1, idx_t)
            tok = self.modal_proj[m_idx](x_m)
            modal_tokens.append(tok)

        tokens = torch.stack(modal_tokens, dim=1)

        ids = torch.arange(self.modal_num, device=device)
        modal_emb = self.modal_id_embedding(ids).unsqueeze(0)

        return self.input_norm(tokens + modal_emb)


def _modal_structure_parameter_budget(modal_index, hidden_size, num_layers):
    """Parameter count of Modal_Token_Encoder + shared Transformer blocks."""
    encoder_params = 0
    for indices in modal_index:
        modal_size = len(indices)
        mid_dim = _modal_intermediate_dim(modal_size, hidden_size)
        encoder_params += mid_dim * (modal_size + hidden_size + 1) + hidden_size
    encoder_params += len(modal_index) * hidden_size  # modal ID embedding
    encoder_params += 2 * hidden_size  # input LayerNorm

    block_params = 8 * hidden_size * hidden_size + 11 * hidden_size
    return encoder_params + num_layers * block_params


class Flat_Feature_Token_Encoder(nn.Module):
    """Parameter-matched flat-feature baseline for the modal-structure ablation.

    All input features are treated as one concatenated vector and mapped to a
    single semantic token. No modality-specific encoder, modality ID embedding,
    or cross-modal Transformer is used on this path.
    """

    def __init__(self, feature_num, hidden_size, target_param_count, dropout=0.0):
        super().__init__()
        denominator = feature_num + hidden_size + 1
        mid_dim = max(hidden_size, round((target_param_count - 3 * hidden_size) / denominator))
        self.mid_dim = mid_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_num, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, X):
        return self.encoder(X).unsqueeze(1)


def _drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class HeterAttentionBlock(nn.Module):
    """Pre-LN transformer block over modal tokens. M=6 个 token，无需 mask."""

    def __init__(self, hidden_size, num_heads, dropout=0.3, ffn_mult=2, drop_path=0.0):
        super(HeterAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ffn_mult, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = drop_path

    def forward(self, X):
        h = self.norm1(X)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        X = X + _drop_path(self.dropout1(attn_out), self.drop_path, self.training)

        h = self.norm2(X)
        X = X + _drop_path(self.dropout2(self.ffn(h)), self.drop_path, self.training)
        return X


class Per_Label_Pool(nn.Module):
    """每个 label 一个 learnable query，cross-attend 到共享的 modal tokens 上。
    参数：1 × query (H,) + 1 × MultiheadAttention(H, h)。
    """

    def __init__(self, hidden_size, num_heads, dropout=0.3):
        super(Per_Label_Pool, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm_out = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, modal_tokens):
        B = modal_tokens.size(0)
        q = self.query.expand(B, -1, -1)
        kv = self.norm_kv(modal_tokens)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        pooled = pooled.squeeze(1)
        pooled = pooled + self.ffn(self.norm_out(pooled))
        return pooled


class _Global_Message_Model(nn.Module):
    """模态级全局类别无关表征。每模态 Linear → modal-level self-attention → squeeze。
    与 label_pools 互补：前者类别无关全局信号，后者 label-specific 信号。
    """

    def __init__(self, DATASET_Dict, Hidden_size=32, word_emb=32, num_heads=4):
        super(_Global_Message_Model, self).__init__()

        self.Modal_Index = DATASET_Dict['Modal_Index']
        self.Embedding_dict = nn.ModuleDict()

        self.Hidden_size = Hidden_size
        self.num_heads = num_heads
        self.word_emb = word_emb
        self.seq_len = len(self.Modal_Index)

        for index_, modal_index in enumerate(self.Modal_Index):
            self.Embedding_dict[f'Modal_{index_}'] = nn.Linear(len(modal_index), self.word_emb)

        self.Fusion_modal_0 = nn.MultiheadAttention(self.word_emb, self.num_heads, batch_first=True)
        self.layer_norm_0 = nn.LayerNorm(self.word_emb)
        self.Fusion_modal_1 = nn.MultiheadAttention(self.word_emb, self.num_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(self.word_emb)

        self.sequeeze_layer = nn.Sequential(
            nn.Linear(self.word_emb * self.seq_len, self.word_emb * (self.seq_len // 2)),
            nn.ReLU(),
            nn.Linear(self.word_emb * (self.seq_len // 2), self.Hidden_size),
            nn.ReLU(),
        )

    def forward(self, X):
        modal_embeddings = []
        for index_, modal_index in enumerate(self.Modal_Index):
            embedding = self.Embedding_dict[f'Modal_{index_}'](X[:, modal_index])
            modal_embeddings.append(embedding)

        Modal_embedding = torch.stack(modal_embeddings, 1)

        Modal_embedding_0, _ = self.Fusion_modal_0(Modal_embedding, Modal_embedding, Modal_embedding)
        Modal_embedding_0 = self.layer_norm_0(Modal_embedding + Modal_embedding_0)

        Modal_embedding_1, _ = self.Fusion_modal_1(Modal_embedding_0, Modal_embedding_0, Modal_embedding_0)
        Modal_embedding = self.layer_norm_1(Modal_embedding_0 + Modal_embedding_1)

        Y = self.sequeeze_layer(Modal_embedding.reshape(-1, self.word_emb * self.seq_len))
        return Y


class OrdinaryAttentionPool(Per_Label_Pool):
    """One class-agnostic query; active FFN capacity matches the removed path."""

    def __init__(self, hidden_size, num_heads, target_params, dropout):
        super().__init__(hidden_size, num_heads, dropout)
        fixed = sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.ffn.parameters())
        width = max(1, round((target_params - fixed - hidden_size) / (2 * hidden_size + 1)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, hidden_size),
        )


class MSCQGN(nn.Module):
    """Modality-Structured Class-Semantic Query Graph Network.

        X (B, F)
        ├── Feature_Modal: (B, F) modal-aware 加性增益
        │     ↓
        │     ├── Modal_Token_Encoder → (B, M=6, H) 模态级 token（每模态 1 个 hidden）
        │     │     ↓
        │     │     Shared Modal Transformer (L 层 Pre-LN block over M tokens)
        │     │     ↓
        │     │     × _Label_num 次：Per_Label_Pool(modal_tokens) → (B, H) per label
        │     │     Auxi_classifier_l: Linear(H, 2)
        │     │     ↓ cat → Message_MLP → (B, H)
        │     │
        │     └── Global_Message → (B, H)
        │           +
        │     Adj_Learning(X) → (B, B)
        │           ↓
        │     Cheb_GCN → (B, Class_Num)

    The model preserves modality-level tokens, extracts class-specific semantic
    representations with label queries, complements them with a global branch,
    and performs subject-level propagation on a learned population graph.
    """

    def __init__(self, DATASET_Dict, Hidden_size, Drop_rate, K,
                 num_layers=3, num_heads=4, input_noise_std=0.05, drop_path=0.05,
                 global_word_emb=None,
                 semantic_branch='both', adj_mode='learned',
                 modal_structure='modal', readout_mode='class_query'):
        super().__init__()
        if readout_mode not in {'class_query', 'ordinary_attention', 'ordinary_multi_query', 'generic_multi_query'}:
            raise ValueError(f'Unknown readout mode: {readout_mode}')
        self.readout_mode = readout_mode

        self.Hidden_size = Hidden_size
        self.DATASET_Dict = DATASET_Dict
        self._Label_num = self.DATASET_Dict['Class_Num']
        self._feature_num = self.DATASET_Dict['Feature_Num']
        self._modal_num = len(DATASET_Dict['Modal_Name'])
        self._modal_index = DATASET_Dict['Modal_Index']
        self.input_noise_std = input_noise_std
        self.noise_scale = input_noise_std / 0.1 if input_noise_std > 0 else 0.0
        modal_structure_alias = {
            'modal': 'modal',
            'modal_tokens': 'modal',
            'full': 'modal',
            'flat': 'flat',
            'none': 'flat',
            'no_modal': 'flat',
        }
        self.modal_structure = modal_structure_alias.get(
            str(modal_structure).lower(), str(modal_structure).lower(),
        )
        if self.modal_structure not in {'modal', 'flat'}:
            raise ValueError(f'Unknown modal_structure: {modal_structure}')
        branch_alias = {
            'both': 'both',
            'per_label_global': 'both',
            'per-label+global': 'both',
            'per_label': 'per_label',
            'per-label': 'per_label',
            'label': 'per_label',
            'global': 'global',
            'global_only': 'global',
        }
        adj_alias = {
            'learned': 'learned',
            'learn': 'learned',
            'learning': 'learned',
            'fixed': 'fixed',
            'static': 'fixed',
            'none': 'none',
            'no': 'none',
            'identity': 'none',
        }
        self.semantic_branch = branch_alias.get(str(semantic_branch).lower(), str(semantic_branch).lower())
        self.adj_mode = adj_alias.get(str(adj_mode).lower(), str(adj_mode).lower())
        if self.semantic_branch not in {'both', 'per_label', 'global'}:
            raise ValueError(f'Unknown semantic_branch: {semantic_branch}')
        if self.adj_mode not in {'learned', 'fixed', 'none'}:
            raise ValueError(f'Unknown adj_mode: {adj_mode}')
        fixed_adj = self.DATASET_Dict.get('Adj')
        if fixed_adj is not None:
            self.register_buffer('fixed_adj', fixed_adj.float().clone().detach())
        else:
            self.fixed_adj = None

        if Hidden_size % num_heads != 0:
            for nh in [4, 2, 1]:
                if Hidden_size % nh == 0:
                    num_heads = nh
                    break

        self.Feature_Modal = Feature_Modal_Embedding(
            self._modal_index, self._modal_num, self._feature_num,
        )

        if self.modal_structure == 'modal':
            self.modal_token_encoder = Modal_Token_Encoder(
                self._modal_index, self._modal_num, Hidden_size,
            )
            self.flat_feature_encoder = None
        else:
            parameter_budget = _modal_structure_parameter_budget(
                self._modal_index, Hidden_size, num_layers,
            )
            self.modal_token_encoder = None
            self.flat_feature_encoder = Flat_Feature_Token_Encoder(
                self._feature_num, Hidden_size, parameter_budget,
                dropout=min(0.5, Drop_rate * 0.5),
            )

        modal_names = self.DATASET_Dict['Modal_Name']
        prior_logits = []
        modal_noise_std_list = []
        for n in modal_names:
            up = n.upper()
            if 'COG' in up or 'COGNITIVE' in up:
                prior_logits.append(4.0)
                modal_noise_std_list.append(0.0)
            elif 'ROI' in up:
                prior_logits.append(2.0)
                modal_noise_std_list.append(0.05)
            elif 'UCSFFSX' in up or 'MRI' in up:
                prior_logits.append(-1.0)
                modal_noise_std_list.append(0.3)
            else:
                prior_logits.append(-2.0)
                modal_noise_std_list.append(0.3)
        prior_t = torch.tensor(prior_logits, dtype=torch.float32)
        self.modal_gate_logit = nn.Parameter(prior_t.clone())
        self.register_buffer('_noise_modal_mask', (prior_t < 0).float())
        self.register_buffer('_modal_noise_std', torch.tensor(modal_noise_std_list, dtype=torch.float32))

        feature_to_modal = torch.zeros(self._feature_num, dtype=torch.long)
        for modal_idx, feat_indices in enumerate(self._modal_index):
            feature_to_modal[feat_indices] = modal_idx
        self.register_buffer('feature_to_modal', feature_to_modal)

        attn_dropout = min(0.5, Drop_rate * 0.5)

        if self.modal_structure == 'modal':
            self.shared_transformer = nn.ModuleList([
                HeterAttentionBlock(Hidden_size, num_heads, dropout=attn_dropout, drop_path=drop_path)
                for _ in range(num_layers)
            ])
        else:
            self.shared_transformer = nn.ModuleList()

        self.label_pools = nn.ModuleList([
            Per_Label_Pool(Hidden_size, num_heads, dropout=attn_dropout)
            for _ in range(self._Label_num)
        ])
        self._Auxi_classifier = nn.ModuleList([
            nn.Linear(Hidden_size, 2) for _ in range(self._Label_num)
        ])

        global_word_emb = Hidden_size if global_word_emb is None else global_word_emb
        self.Global_Message = _Global_Message_Model(
            self.DATASET_Dict, Hidden_size=Hidden_size, word_emb=global_word_emb,
        )
        self.Adj_Learning = graph_learning(
            self.DATASET_Dict, Hidden_size, rate=0.1, use_raw_x=True,
        )
        self.GCN = Cheb_GCN(
            Dim_emb=Hidden_size,
            hidden=Hidden_size // 2,
            out_channels=self._Label_num,
            P=Drop_rate,
            K=K,
        )

        self.Message_MLP = nn.Sequential(
            nn.Linear(self._Label_num * Hidden_size, (self._Label_num * Hidden_size) // 2),
            nn.ReLU(),
            nn.Dropout(Drop_rate),
            nn.Linear((self._Label_num * Hidden_size) // 2, Hidden_size),
            nn.ReLU(),
        )

        if self.readout_mode == 'ordinary_attention':
            if self.semantic_branch != 'both':
                raise ValueError('Ordinary pooling ablation requires both semantic paths')
            budget = sum(p.numel() for module in (self.label_pools, self._Auxi_classifier, self.Message_MLP)
                         for p in module.parameters())
            # Initialize common components exactly as baseline; preserve training RNG state.
            with torch.random.fork_rng(devices=[]):
                self.ordinary_pool = OrdinaryAttentionPool(Hidden_size, num_heads, budget, attn_dropout)
            del self.label_pools, self._Auxi_classifier, self.Message_MLP
        elif self.readout_mode == 'generic_multi_query':
            if self.semantic_branch != 'both':
                raise ValueError('Generic multi-query ablation requires both semantic paths')
            with torch.random.fork_rng(devices=[]):
                self._Auxi_classifier = nn.ModuleList([
                    nn.Linear(Hidden_size, self._Label_num) for _ in range(self._Label_num)
                ])
        elif self.readout_mode == 'ordinary_multi_query':
            if self.semantic_branch != 'both':
                raise ValueError('Ordinary multi-query ablation requires both semantic paths')
            # Keep query/readout initialization and regularization identical;
            # remove only the explicit query-to-diagnosis auxiliary heads.
            del self._Auxi_classifier

    def gate_sparsity_loss(self):
        """L1 push-to-0 penalty on noise-modal gates only.
        防止训练中弱模态 gate 上升（即 model 试图通过激活 noise modal 来过拟合 train）。
        """
        gate = torch.sigmoid(self.modal_gate_logit)
        return (gate * self._noise_modal_mask).sum()

    def forward(self, X_raw):
        X = self.Feature_Modal(X_raw)
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (per_feature_noise_std * self.noise_scale).view(1, -1)

        modal_gate = torch.sigmoid(self.modal_gate_logit)

        if self.semantic_branch == 'global':
            Label_embedding = [
                X_raw.new_zeros((X_raw.size(0), self.Hidden_size))
                for _ in range(self._Label_num)
            ]
            Auxi_classifier_output = [
                X_raw.new_zeros((X_raw.size(0), 2))
                for _ in range(self._Label_num)
            ]
            Y = X_raw.new_zeros((X_raw.size(0), self.Hidden_size))
        else:
            if self.modal_structure == 'modal':
                H = self.modal_token_encoder(X)
                H = H * modal_gate.view(1, -1, 1)

                for blk in self.shared_transformer:
                    H = blk(H)
            else:
                H = self.flat_feature_encoder(X_raw)

            Label_embedding = []
            Auxi_classifier_output = []
            if self.readout_mode == 'ordinary_attention':
                Y = self.ordinary_pool(H)
            else:
                for label_idx in range(self._Label_num):
                    label_emb = self.label_pools[label_idx](H)
                    Label_embedding.append(label_emb)
                    if self.readout_mode in {'class_query', 'generic_multi_query'}:
                        aux_out = self._Auxi_classifier[label_idx](label_emb)
                        Auxi_classifier_output.append(aux_out)
                Y = self.Message_MLP(torch.cat(Label_embedding, dim=-1))

        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)

        if self.semantic_branch == 'per_label':
            Global_Embedding = X_raw.new_zeros((X_raw.size(0), self.Hidden_size))
        else:
            Global_Embedding = self.Global_Message(X_gated)

        if self.adj_mode == 'learned':
            Adj = self.Adj_Learning(X_gated)
        elif self.adj_mode == 'fixed' and self.fixed_adj is not None:
            Adj = self.fixed_adj.to(device=X_raw.device, dtype=X_raw.dtype)
        else:
            Adj = torch.eye(X_raw.size(0), device=X_raw.device, dtype=X_raw.dtype)
        if self.semantic_branch == 'per_label':
            Fused_Embedding = Y
        elif self.semantic_branch == 'global':
            Fused_Embedding = Global_Embedding
        else:
            Fused_Embedding = Y + Global_Embedding

        Y = self.GCN(Fused_Embedding, Adj)

        return Y, Label_embedding, Auxi_classifier_output
