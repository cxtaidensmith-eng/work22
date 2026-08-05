from .models import AdvDIFFormer_GraphHead, Cheb_GCN, DIFFormer_GraphHead, graph_learning
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

    def forward(self, modal_tokens, query_override=None):
        B = modal_tokens.size(0)
        query = self.query if query_override is None else query_override
        if query.dim() == 2:
            query = query.unsqueeze(0)
        if query.dim() != 3 or query.size(1) != 1 or query.size(2) != self.query.size(2):
            raise ValueError(
                'query_override must have shape [1, 1, H], [1, H], or [B, 1, H]'
            )
        if query.size(0) not in {1, B}:
            raise ValueError(
                f'query_override batch dimension must be 1 or {B}, got {query.size(0)}'
            )
        q = query.expand(B, -1, -1)
        kv = self.norm_kv(modal_tokens)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        pooled = pooled.squeeze(1)
        pooled = pooled + self.ffn(self.norm_out(pooled))
        return pooled


class ComponentSharedLabelPool(nn.Module):
    """Factorized label pooling with private queries/FFNs and shared attention.

    The module is assembled from fully initialized legacy label pools so the
    random-number consumption, and therefore all downstream initialization,
    remains identical to the independent-pool baseline under the same seed.
    Only one ``norm_kv`` and one ``attn`` module are retained; query tokens,
    output norms, and FFNs remain label-specific.
    """

    def __init__(self, initialized_label_pools):
        super(ComponentSharedLabelPool, self).__init__()
        if not initialized_label_pools:
            raise ValueError('initialized_label_pools must not be empty')

        self.queries = nn.ParameterList([
            pool.query for pool in initialized_label_pools
        ])
        self.norm_kv = initialized_label_pools[0].norm_kv
        self.attn = initialized_label_pools[0].attn
        self.norm_out = nn.ModuleList([
            pool.norm_out for pool in initialized_label_pools
        ])
        self.ffn = nn.ModuleList([
            pool.ffn for pool in initialized_label_pools
        ])
        self.last_attention_outputs = None
        self.last_outputs = None

    def forward(self, modal_tokens):
        batch_size = modal_tokens.size(0)
        kv = self.norm_kv(modal_tokens)
        attention_outputs = []
        pooled_outputs = []
        for query, norm_out, ffn in zip(self.queries, self.norm_out, self.ffn):
            q = query.expand(batch_size, -1, -1)
            pooled, _ = self.attn(q, kv, kv, need_weights=False)
            pooled = pooled.squeeze(1)
            attention_outputs.append(pooled)
            pooled_outputs.append(pooled + ffn(norm_out(pooled)))
        self.last_attention_outputs = torch.stack(attention_outputs, dim=1).detach()
        self.last_outputs = torch.stack(pooled_outputs, dim=1).detach()
        return pooled_outputs


class LatentBranchPool(nn.Module):
    """Query-free semantic branch over the shared modal tokens.

    The branch owns an independent value projection and FFN.  It deliberately
    contains no learnable query, attention layer, or class-specific logits.
    """

    def __init__(self, hidden_size, dropout=0.3):
        super(LatentBranchPool, self).__init__()
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, modal_tokens):
        value = self.value_projection(modal_tokens).mean(dim=1)
        return self.output_norm(value + self.ffn(value))


class PatientBoundaryRouter(nn.Module):
    """Minimal patient-specific router over the existing global representation."""

    def __init__(self, hidden_size, router_hidden=16, branch_count=3):
        super(PatientBoundaryRouter, self).__init__()
        self.input_layer = nn.Linear(hidden_size, router_hidden)
        self.activation = nn.GELU()
        self.output_layer = nn.Linear(router_hidden, branch_count)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, global_repr):
        return self.output_layer(self.activation(self.input_layer(global_repr)))


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
            modal_embeddings.append(self.Embedding_dict[f'Modal_{index_}'](X[:, modal_index]))

        Modal_embedding = torch.stack(modal_embeddings, 1)

        Modal_embedding_0, _ = self.Fusion_modal_0(Modal_embedding, Modal_embedding, Modal_embedding)
        Modal_embedding_0 = self.layer_norm_0(Modal_embedding + Modal_embedding_0)

        Modal_embedding_1, _ = self.Fusion_modal_1(Modal_embedding_0, Modal_embedding_0, Modal_embedding_0)
        Modal_embedding = self.layer_norm_1(Modal_embedding_0 + Modal_embedding_1)

        Y = self.sequeeze_layer(Modal_embedding.reshape(-1, self.word_emb * self.seq_len))
        return Y


class HeterGraph_Model_Kmeans(nn.Module):
    """v3: modal-token + shared transformer + per-label query pool.

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

    与 v2 关键差异（结构性）：
      - Token 粒度从 per-feature (F=360) 降到 per-modal (M=6)：参数量降，归纳偏置紧
      - Transformer stack 跨 label 共享：避免每 label 重复学特征关系
      - 去掉 K-means cluster mask：M=6 token 直接全连接 attention，无须 mask
      - 保留 _Global_Message_Model 作为类别无关的全局信号通道（与 label_pools 互补）
      - Herter_Graph 参数依然接收以保持接口兼容，但内部不使用
    """

    def __init__(self, DATASET_Dict, Herter_Graph, Hidden_size, Drop_rate, K,
                 num_layers=3, num_heads=4, input_noise_std=0.05, drop_path=0.05,
                 graph_head='cheb', graph_layers=1, graph_heads=2,
                 graph_beta=0.5, graph_k_order=3, global_word_emb=None,
                 graph_alpha=0.5, graph_kernel='simple', graph_use_graph=True,
                 graph_dropout=None, graph_hidden=None,
                 semantic_branch='both', adj_mode='learned',
                 semantic_fusion='add',
                 category_branch_variant='original',
                 query_pool_variant='independent',
                 osfq_anchor_mode='none',
                 osfq_ema_decay=0.9,
                 latent_auxiliary_mode='multiclass',
                 category_branch_fusion='concat',
                 label_graph_alpha=0.0, label_graph_topk=0,
                 label_graph_reg_lambda=0.0):
        super(HeterGraph_Model_Kmeans, self).__init__()

        self.Hidden_size = Hidden_size
        self.DATASET_Dict = DATASET_Dict
        self.Herter_Graph = Herter_Graph

        self._Label_num = self.DATASET_Dict['Class_Num']
        self._feature_num = self.DATASET_Dict['Feature_Num']
        self._modal_num = len(DATASET_Dict['Modal_Name'])
        self._modal_index = DATASET_Dict['Modal_Index']
        self.input_noise_std = input_noise_std
        self.noise_scale = input_noise_std / 0.1 if input_noise_std > 0 else 0.0
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
        fusion_alias = {
            'add': 'add',
            'sum': 'add',
            'legacy': 'add',
            'safe_residual': 'safe_residual',
            'gated_residual': 'safe_residual',
            'zero_init_residual': 'safe_residual',
            'norm_matched_residual': 'norm_matched_residual',
            'safe_residual_v2': 'norm_matched_residual',
        }
        self.semantic_fusion = fusion_alias.get(str(semantic_fusion).lower(), str(semantic_fusion).lower())
        category_branch_alias = {
            'original': 'original',
            'query': 'original',
            'query_pool': 'original',
            'legacy_query': 'original',
            'query_free_multibranch': 'query_free_multibranch',
            'query-free-multibranch': 'query_free_multibranch',
            'latent_multibranch': 'query_free_multibranch',
        }
        self.category_branch_variant = category_branch_alias.get(
            str(category_branch_variant).lower(), str(category_branch_variant).lower()
        )
        query_pool_alias = {
            'independent': 'independent',
            'original': 'independent',
            'shared': 'shared',
            'osfq': 'shared',
            'ovr_aligned_shared_query': 'shared',
            'component_shared': 'component_shared',
            'component_sharing': 'component_shared',
            'component-sharing': 'component_shared',
            'shared_attention': 'component_shared',
            'query_pool_component_sharing': 'component_shared',
        }
        self.query_pool_variant = query_pool_alias.get(
            str(query_pool_variant).lower(), str(query_pool_variant).lower()
        )
        anchor_mode_alias = {
            'none': 'none',
            'no_anchor': 'none',
            'correct': 'correct',
            'correct_anchor': 'correct',
            'cyclic_mismatch': 'cyclic_mismatch',
            'wrong': 'cyclic_mismatch',
            'wrong_anchor': 'cyclic_mismatch',
        }
        self.osfq_anchor_mode = anchor_mode_alias.get(
            str(osfq_anchor_mode).lower(), str(osfq_anchor_mode).lower()
        )
        self.osfq_ema_decay = float(osfq_ema_decay)
        self.osfq_anchor_active = False
        latent_auxiliary_alias = {
            'multiclass': 'multiclass',
            'three_class': 'multiclass',
            'full': 'multiclass',
            'one_vs_rest': 'one_vs_rest',
            'one-vs-rest': 'one_vs_rest',
            'ovr': 'one_vs_rest',
            'binary': 'one_vs_rest',
        }
        self.latent_auxiliary_mode = latent_auxiliary_alias.get(
            str(latent_auxiliary_mode).lower(), str(latent_auxiliary_mode).lower()
        )
        category_branch_fusion_alias = {
            'concat': 'concat',
            'standard': 'concat',
            'global_weight': 'global_weight',
            'global_weighted': 'global_weight',
            'boundary_global_weight': 'global_weight',
            'patient_router': 'patient_router',
            'patient_routing': 'patient_router',
            'patient_boundary_router': 'patient_router',
        }
        self.category_branch_fusion = category_branch_fusion_alias.get(
            str(category_branch_fusion).lower(), str(category_branch_fusion).lower()
        )
        self.adj_mode = adj_alias.get(str(adj_mode).lower(), str(adj_mode).lower())
        if self.semantic_branch not in {'both', 'per_label', 'global'}:
            raise ValueError(f'Unknown semantic_branch: {semantic_branch}')
        if self.semantic_fusion not in {'add', 'safe_residual', 'norm_matched_residual'}:
            raise ValueError(f'Unknown semantic_fusion: {semantic_fusion}')
        if self.semantic_fusion in {'safe_residual', 'norm_matched_residual'} and self.semantic_branch != 'both':
            raise ValueError(f'{self.semantic_fusion} fusion requires semantic_branch=both')
        if self.category_branch_variant not in {'original', 'query_free_multibranch'}:
            raise ValueError(f'Unknown category_branch_variant: {category_branch_variant}')
        if self.query_pool_variant not in {'independent', 'shared', 'component_shared'}:
            raise ValueError(f'Unknown query_pool_variant: {query_pool_variant}')
        if self.osfq_anchor_mode not in {'none', 'correct', 'cyclic_mismatch'}:
            raise ValueError(f'Unknown osfq_anchor_mode: {osfq_anchor_mode}')
        if not 0.0 <= self.osfq_ema_decay < 1.0:
            raise ValueError('osfq_ema_decay must be in [0, 1)')
        if self.query_pool_variant in {'shared', 'component_shared'} and self.category_branch_variant != 'original':
            raise ValueError('shared query-pool variants require category_branch_variant=original')
        if self.query_pool_variant != 'shared' and self.osfq_anchor_mode != 'none':
            raise ValueError('OVR anchors require query_pool_variant=shared')
        if self.latent_auxiliary_mode not in {'multiclass', 'one_vs_rest'}:
            raise ValueError(f'Unknown latent_auxiliary_mode: {latent_auxiliary_mode}')
        if self.category_branch_fusion not in {'concat', 'global_weight', 'patient_router'}:
            raise ValueError(f'Unknown category_branch_fusion: {category_branch_fusion}')
        if self.category_branch_fusion in {'global_weight', 'patient_router'} and self.category_branch_variant != 'query_free_multibranch':
            raise ValueError(f'{self.category_branch_fusion} category fusion requires query_free_multibranch')
        if self.category_branch_fusion == 'patient_router' and self.semantic_branch != 'both':
            raise ValueError('patient_router category fusion requires semantic_branch=both')
        if self.adj_mode not in {'learned', 'fixed', 'none'}:
            raise ValueError(f'Unknown adj_mode: {adj_mode}')
        self.label_graph_alpha = float(label_graph_alpha)
        self.label_graph_topk = int(label_graph_topk)
        self.label_graph_reg_lambda = float(label_graph_reg_lambda)
        self.last_adj_base = None
        self.last_label_relation = None

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

        self.modal_token_encoder = Modal_Token_Encoder(
            self._modal_index, self._modal_num, Hidden_size,
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

        self.shared_transformer = nn.ModuleList([
            HeterAttentionBlock(Hidden_size, num_heads, dropout=attn_dropout, drop_path=drop_path)
            for _ in range(num_layers)
        ])

        if self.category_branch_variant == 'original':
            constructed_label_pools = [
                Per_Label_Pool(Hidden_size, num_heads, dropout=attn_dropout)
                for _ in range(self._Label_num)
            ]
            # The shared-pool variant deliberately constructs all three legacy
            # pools before retaining the first. This preserves the exact random
            # number consumption and therefore the initialization of every
            # downstream baseline module under the same seed.
            if self.query_pool_variant == 'shared':
                self.label_pools = nn.ModuleList([constructed_label_pools[0]])
            elif self.query_pool_variant == 'component_shared':
                self.component_shared_pool = ComponentSharedLabelPool(
                    constructed_label_pools
                )
            else:
                self.label_pools = nn.ModuleList(constructed_label_pools)
            self._Auxi_classifier = nn.ModuleList([
                nn.Linear(Hidden_size, 2) for _ in range(self._Label_num)
            ])
        else:
            self.latent_branches = nn.ModuleDict({
                f'latent_branch_{idx + 1}': LatentBranchPool(Hidden_size, dropout=attn_dropout)
                for idx in range(self._Label_num)
            })
            latent_auxiliary_classes = (
                self._Label_num if self.latent_auxiliary_mode == 'multiclass' else 2
            )
            self.latent_aux_classifiers = nn.ModuleDict({
                f'latent_branch_{idx + 1}': nn.Linear(Hidden_size, latent_auxiliary_classes)
                for idx in range(self._Label_num)
            })
            if self.category_branch_fusion == 'global_weight':
                self.global_boundary_fusion_weights = nn.Parameter(
                    torch.ones(self._Label_num)
                )

        global_word_emb = Hidden_size if global_word_emb is None else global_word_emb
        self.Global_Message = _Global_Message_Model(
            self.DATASET_Dict, Hidden_size=Hidden_size, word_emb=global_word_emb,
        )
        self.Adj_Learning = graph_learning(
            self.DATASET_Dict, Hidden_size, rate=0.1, use_raw_x=True,
        )
        graph_head = graph_head.lower()
        if graph_head in {'cheb', 'chebgcn', 'cheb_gcn'}:
            self.GCN = Cheb_GCN(
                Dim_emb=Hidden_size, hidden=Hidden_size // 2,
                out_channels=self._Label_num, P=Drop_rate, K=K,
            )
        elif graph_head in {'advdif', 'advdifformer', 'adv_difformer'}:
            self.GCN = AdvDIFFormer_GraphHead(
                Dim_emb=Hidden_size, hidden=Hidden_size // 2,
                out_channels=self._Label_num, P=Drop_rate,
                num_layers=graph_layers, num_heads=graph_heads,
                beta=graph_beta, K_order=graph_k_order,
            )
        elif graph_head in {'dif', 'difformer', 'dif_former'}:
            graph_hidden = Hidden_size // 2 if graph_hidden is None else graph_hidden
            graph_dropout = Drop_rate if graph_dropout is None else graph_dropout
            self.GCN = DIFFormer_GraphHead(
                Dim_emb=Hidden_size, hidden=graph_hidden,
                out_channels=self._Label_num, P=graph_dropout,
                num_layers=graph_layers, num_heads=graph_heads,
                graph_weight=graph_beta,
                alpha=graph_alpha,
                kernel=graph_kernel,
                use_graph=graph_use_graph,
            )
        else:
            raise ValueError(f'Unknown graph_head: {graph_head}')

        self.Message_MLP = nn.Sequential(
            nn.Linear(self._Label_num * Hidden_size, (self._Label_num * Hidden_size) // 2),
            nn.ReLU(),
            nn.Dropout(Drop_rate),
            nn.Linear((self._Label_num * Hidden_size) // 2, Hidden_size),
            nn.ReLU(),
        )

        # Optional safe global branch. It is intentionally created after all
        # legacy modules so enabling it cannot change their random
        # initialization. The zero-initialized projection makes the initial
        # fused representation exactly equal to the category-only pathway.
        if self.semantic_fusion == 'safe_residual':
            self.global_residual_norm = nn.LayerNorm(Hidden_size, elementwise_affine=False)
            self.global_residual_projection = nn.Linear(Hidden_size, Hidden_size)
            nn.init.zeros_(self.global_residual_projection.weight)
            nn.init.zeros_(self.global_residual_projection.bias)
            # Start almost closed (sigmoid(-4) ~= 0.018). Together with the
            # zero projection this preserves the category-only output exactly
            # at initialization and prevents a one-step residual-scale jump.
            self.global_residual_gate_logit = nn.Parameter(torch.tensor(-4.0))
        elif self.semantic_fusion == 'norm_matched_residual':
            self.global_residual_norm = nn.LayerNorm(Hidden_size, elementwise_affine=False)
            self.global_residual_projection = nn.Linear(Hidden_size, Hidden_size)
            # tanh(0)=0 preserves the category-only representation exactly,
            # while the scalar receives a gradient on the first update. The
            # projected global direction is L2-normalized and then matched to
            # the category representation norm, making the scalar the actual
            # residual-to-category ratio.
            self.global_residual_scale = nn.Parameter(torch.zeros(()))

        # Created after every legacy module so adding the router cannot shift
        # their seed-dependent initialization. Its zero-initialized output
        # layer gives alpha=1/3 and the applied gain 3*alpha=1 at startup.
        if self.category_branch_fusion == 'patient_router':
            self.patient_boundary_router = PatientBoundaryRouter(
                Hidden_size, router_hidden=16, branch_count=self._Label_num,
            )

        # OVR-aligned shared-query state is appended after every legacy module,
        # so enabling it cannot perturb any common parameter initialization.
        if self.query_pool_variant == 'shared':
            self.osfq_alpha = nn.Parameter(torch.zeros(()))
            self.register_buffer(
                'osfq_anchor_ema',
                torch.zeros(self._Label_num, Hidden_size),
                persistent=True,
            )
            self.register_buffer(
                'osfq_anchor_initialized',
                torch.tensor(False, dtype=torch.bool),
                persistent=True,
            )
            self.register_buffer(
                'osfq_anchor_updates',
                torch.tensor(0, dtype=torch.long),
                persistent=True,
            )

        self.last_category_embedding = None
        self.last_global_embedding = None
        self.last_global_residual = None
        self.last_global_gate = None
        self.last_branch_outputs = None
        self.last_boundary_fusion_alpha = None
        self.last_patient_router_input = None
        self.last_patient_router_logits = None
        self.last_patient_routing_alpha = None
        self.last_patient_routing_gain = None
        self.last_routed_branch_concat = None
        self.last_osfq_raw_directions = None
        self.last_osfq_ema_directions = None
        self.last_osfq_centered_directions = None
        self.last_osfq_effective_queries = None
        self.last_osfq_anchor_contribution_norm = None

    @torch.no_grad()
    def _current_ovr_directions(self):
        """Return normalized OVR classifier normals (positive minus rest)."""
        if self.category_branch_variant != 'original':
            raise RuntimeError('OVR directions require the original binary auxiliary heads')
        directions = torch.stack([
            classifier.weight[1] - classifier.weight[0]
            for classifier in self._Auxi_classifier
        ])
        return F.normalize(directions, dim=-1, eps=1e-6)

    @torch.no_grad()
    def update_osfq_anchor_ema(self):
        """Update the detached EMA anchors after an optimizer step."""
        if self.query_pool_variant != 'shared':
            raise RuntimeError('OVR anchor EMA is available only for the shared query pool')
        directions = self._current_ovr_directions()
        if not bool(self.osfq_anchor_initialized.item()):
            self.osfq_anchor_ema.copy_(directions)
            self.osfq_anchor_initialized.fill_(True)
        else:
            self.osfq_anchor_ema.mul_(self.osfq_ema_decay).add_(
                directions, alpha=1.0 - self.osfq_ema_decay
            )
            self.osfq_anchor_ema.copy_(
                F.normalize(self.osfq_anchor_ema, dim=-1, eps=1e-6)
            )
        self.osfq_anchor_updates.add_(1)
        return self.osfq_anchor_ema

    def set_osfq_anchor_active(self, active):
        """Enable anchors only after warm-up; S0 remains anchor-free."""
        requested = bool(active)
        self.osfq_anchor_active = bool(
            requested
            and self.query_pool_variant == 'shared'
            and self.osfq_anchor_mode != 'none'
            and bool(self.osfq_anchor_initialized.item())
        )

    def _osfq_centered_directions(self):
        if not self.osfq_anchor_active:
            return self.osfq_anchor_ema.new_zeros(self.osfq_anchor_ema.shape)
        direction = F.normalize(self.osfq_anchor_ema.detach(), dim=-1, eps=1e-6)
        direction = direction - direction.mean(dim=0, keepdim=True)
        rms = torch.sqrt(direction.pow(2).sum(dim=-1).mean() + 1e-6)
        direction = direction / rms
        if self.osfq_anchor_mode == 'cyclic_mismatch':
            direction = direction[[1, 2, 0]]
        return direction

    def _osfq_effective_queries(self):
        if self.query_pool_variant != 'shared':
            raise RuntimeError('Effective OVR queries require the shared query pool')
        base_query = self.label_pools[0].query
        directions = self._osfq_centered_directions()
        effective = base_query.expand(self._Label_num, -1, -1) + (
            self.osfq_alpha * directions[:, None, :]
        )
        self.last_osfq_raw_directions = self._current_ovr_directions().detach().clone()
        self.last_osfq_ema_directions = self.osfq_anchor_ema.detach().clone()
        self.last_osfq_centered_directions = directions.detach().clone()
        self.last_osfq_effective_queries = effective.detach().clone()
        self.last_osfq_anchor_contribution_norm = (
            self.osfq_alpha.detach().abs() * directions.detach().norm(dim=-1)
        )
        return effective

    @torch.no_grad()
    def osfq_anchor_snapshot(self):
        if self.query_pool_variant != 'shared':
            return None
        directions = self._osfq_centered_directions()
        effective = self._osfq_effective_queries()
        current = self._current_ovr_directions()
        ema = self.osfq_anchor_ema.detach()
        raw_to_ema_cosine = F.cosine_similarity(current, ema, dim=-1, eps=1e-6)
        mapping = [0, 1, 2]
        if self.osfq_anchor_mode == 'cyclic_mismatch':
            mapping = [1, 2, 0]
        return {
            'mode': self.osfq_anchor_mode,
            'active': bool(self.osfq_anchor_active),
            'initialized': bool(self.osfq_anchor_initialized.item()),
            'updates': int(self.osfq_anchor_updates.item()),
            'ema_decay': float(self.osfq_ema_decay),
            'alpha': float(self.osfq_alpha.item()),
            'source_mapping': mapping,
            'current_ovr_direction_norms': current.norm(dim=-1).detach().cpu().tolist(),
            'ema_direction_norms': ema.norm(dim=-1).detach().cpu().tolist(),
            'current_to_ema_cosine': raw_to_ema_cosine.detach().cpu().tolist(),
            'direction_norms': directions.norm(dim=-1).detach().cpu().tolist(),
            'centered_direction_zero_sum_error': float(
                directions.sum(dim=0).norm().detach().cpu().item()
            ),
            'centered_direction_global_rms': float(
                torch.sqrt(directions.pow(2).sum(dim=-1).mean()).detach().cpu().item()
            ),
            'anchor_contribution_norms': (
                self.osfq_alpha.abs() * directions.norm(dim=-1)
            ).detach().cpu().tolist(),
            'effective_query_norms': effective[:, 0].norm(dim=-1).detach().cpu().tolist(),
        }

    def _fuse_semantic_branches(self, category_embedding, global_embedding):
        self.last_category_embedding = category_embedding
        self.last_global_embedding = global_embedding
        if self.semantic_branch == 'per_label':
            self.last_global_residual = torch.zeros_like(category_embedding)
            self.last_global_gate = category_embedding.new_zeros(())
            return category_embedding
        if self.semantic_branch == 'global':
            self.last_global_residual = global_embedding
            self.last_global_gate = global_embedding.new_ones(())
            return global_embedding
        if self.semantic_fusion == 'safe_residual':
            gate = torch.sigmoid(self.global_residual_gate_logit)
            residual = gate * self.global_residual_projection(self.global_residual_norm(global_embedding))
            self.last_global_residual = residual
            self.last_global_gate = gate
            return category_embedding + residual
        if self.semantic_fusion == 'norm_matched_residual':
            scale = torch.tanh(self.global_residual_scale)
            direction = F.normalize(
                self.global_residual_projection(self.global_residual_norm(global_embedding)),
                dim=-1,
            )
            category_scale = category_embedding.norm(dim=-1, keepdim=True).detach()
            residual = scale * category_scale * direction
            self.last_global_residual = residual
            self.last_global_gate = scale
            return category_embedding + residual
        self.last_global_residual = global_embedding
        self.last_global_gate = global_embedding.new_ones(())
        return category_embedding + global_embedding

    def gate_sparsity_loss(self):
        """L1 push-to-0 penalty on noise-modal gates only.
        防止训练中弱模态 gate 上升（即 model 试图通过激活 noise modal 来过拟合 train）。
        """
        gate = torch.sigmoid(self.modal_gate_logit)
        return (gate * self._noise_modal_mask).sum()

    def _label_relation_graph(self, Label_embedding):
        relation = None
        for label_emb in Label_embedding:
            z = F.normalize(label_emb, dim=-1)
            sim = torch.mm(z, z.t())
            sim = (sim + 1.0) * 0.5
            relation = sim if relation is None else relation + sim
        relation = relation / max(1, len(Label_embedding))
        relation = torch.nan_to_num(relation, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if 0 < self.label_graph_topk < relation.size(1):
            values, indices = torch.topk(relation, k=self.label_graph_topk, dim=1)
            sparse_relation = torch.zeros_like(relation)
            relation = sparse_relation.scatter(1, indices, values)
        eye = torch.eye(relation.size(0), device=relation.device, dtype=relation.dtype)
        return relation * (1.0 - eye) + eye

    def label_relation_loss(self, train_mask=None):
        if self.last_adj_base is None or self.last_label_relation is None:
            device = self.modal_gate_logit.device
            return torch.zeros((), device=device)

        adj = self.last_adj_base
        relation = self.last_label_relation.detach()
        if train_mask is not None:
            idx = torch.where(train_mask)[0]
            adj = adj.index_select(0, idx).index_select(1, idx)
            relation = relation.index_select(0, idx).index_select(1, idx)

        off_diag = 1.0 - torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        return F.mse_loss(adj * off_diag, relation * off_diag)

    def forward(self, X_raw):
        X = self.Feature_Modal(X_raw)
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (per_feature_noise_std * self.noise_scale).view(1, -1)

        modal_gate = torch.sigmoid(self.modal_gate_logit)
        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)
        Global_Embedding = None
        self.last_patient_router_input = None
        self.last_patient_router_logits = None
        self.last_patient_routing_alpha = None
        self.last_patient_routing_gain = None
        self.last_routed_branch_concat = None

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
            H = self.modal_token_encoder(X)
            H = H * modal_gate.view(1, -1, 1)

            for blk in self.shared_transformer:
                H = blk(H)

            Label_embedding = []
            Auxi_classifier_output = []
            if self.query_pool_variant == 'component_shared':
                Label_embedding = self.component_shared_pool(H)
                Auxi_classifier_output = [
                    self._Auxi_classifier[label_idx](branch_embedding)
                    for label_idx, branch_embedding in enumerate(Label_embedding)
                ]
                branch_items = None
            elif (
                self.category_branch_variant == 'original'
                and self.query_pool_variant == 'shared'
            ):
                effective_queries = self._osfq_effective_queries()
                branch_items = (
                    (
                        self.label_pools[0],
                        self._Auxi_classifier[label_idx],
                        effective_queries[label_idx:label_idx + 1],
                    )
                    for label_idx in range(self._Label_num)
                )
            elif self.category_branch_variant == 'original':
                branch_items = (
                    (self.label_pools[label_idx], self._Auxi_classifier[label_idx], None)
                    for label_idx in range(self._Label_num)
                )
            else:
                branch_items = (
                    (self.latent_branches[name], self.latent_aux_classifiers[name], None)
                    for name in self.latent_branches
                )

            if branch_items is not None:
                for branch_pool, aux_classifier, query_override in branch_items:
                    if query_override is None:
                        branch_emb = branch_pool(H)
                    else:
                        branch_emb = branch_pool(H, query_override=query_override)
                    aux_out = aux_classifier(branch_emb)
                    Label_embedding.append(branch_emb)
                    Auxi_classifier_output.append(aux_out)

            self.last_branch_outputs = Label_embedding
            branch_embeddings = Label_embedding
            if self.category_branch_fusion == 'global_weight':
                fusion_alpha = torch.softmax(
                    self.global_boundary_fusion_weights, dim=0
                )
                branch_embeddings = [
                    fusion_alpha[idx] * branch_embedding
                    for idx, branch_embedding in enumerate(Label_embedding)
                ]
                self.last_boundary_fusion_alpha = fusion_alpha
            elif self.category_branch_fusion == 'patient_router':
                Global_Embedding = self.Global_Message(X_gated)
                router_input = Global_Embedding.detach()
                router_logits = self.patient_boundary_router(router_input)
                routing_alpha = torch.softmax(router_logits, dim=-1)
                routing_gain = self._Label_num * routing_alpha
                branch_embeddings = [
                    routing_gain[:, idx:idx + 1] * branch_embedding
                    for idx, branch_embedding in enumerate(Label_embedding)
                ]
                self.last_boundary_fusion_alpha = None
                self.last_patient_router_input = router_input
                self.last_patient_router_logits = router_logits
                self.last_patient_routing_alpha = routing_alpha
                self.last_patient_routing_gain = routing_gain
            else:
                self.last_boundary_fusion_alpha = None
            routed_branch_concat = torch.cat(branch_embeddings, dim=-1)
            self.last_routed_branch_concat = routed_branch_concat
            Y = self.Message_MLP(routed_branch_concat)

        if self.semantic_branch == 'per_label':
            Global_Embedding = X_raw.new_zeros((X_raw.size(0), self.Hidden_size))
        elif Global_Embedding is None:
            Global_Embedding = self.Global_Message(X_gated)

        if self.adj_mode == 'learned':
            Adj = self.Adj_Learning(X_gated)
        elif self.adj_mode == 'fixed' and self.fixed_adj is not None:
            Adj = self.fixed_adj.to(device=X_raw.device, dtype=X_raw.dtype)
        else:
            Adj = torch.eye(X_raw.size(0), device=X_raw.device, dtype=X_raw.dtype)
        self.last_adj_base = Adj
        self.last_label_relation = None
        if (self.label_graph_alpha > 0 or self.label_graph_reg_lambda > 0) and self.semantic_branch != 'global':
            R_label = self._label_relation_graph(Label_embedding)
            self.last_label_relation = R_label
            alpha = max(0.0, min(1.0, self.label_graph_alpha))
            if alpha > 0:
                Adj = (1.0 - alpha) * Adj + alpha * R_label
        Y = self.GCN(self._fuse_semantic_branches(Y, Global_Embedding), Adj)

        return Y, Label_embedding, Auxi_classifier_output
