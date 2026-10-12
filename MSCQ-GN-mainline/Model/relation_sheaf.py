"""Relation/content-conditioned cooperative propagation, using native PyTorch.

CSNN paper (arXiv:2507.00647v3), Eqs. 7--8: Delta_in^T Delta_out.
This is a modality-conditioned extension, not an exact author-code reproduction.
Each content token is reshaped into a 2D stalk with H/2 channels. Candidate
graphs are symmetric; row i receives from column j. No labels enter the graphs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .network import MSCQGN


@torch.no_grad()
def modal_candidates(x, modal_index, k=10):
    """CPU stable cosine top-k, followed by union with reverse edges.

    Exactly k outgoing selections BEFORE symmetrization; resulting degrees can
    exceed k. Ties use dataset row order. All three variants reuse these edges.
    """
    x = x.detach().cpu()
    if not 0 < k < len(x) or not torch.isfinite(x).all():
        raise ValueError('Require finite features and 0 < k < sample count')
    graphs = []
    for indices in modal_index:
        z = F.normalize(x[:, indices], dim=-1)
        similarity = z @ z.T
        ranked = similarity.clone()
        ranked.fill_diagonal_(-torch.inf)
        neighbors = ranked.argsort(dim=-1, descending=True, stable=True)[:, :k]
        support = torch.zeros_like(similarity, dtype=torch.bool)
        support.scatter_(1, neighbors, True)
        support = support | support.T
        edge = support.nonzero().T.contiguous()
        graphs.append(dict(edge=edge, similarity=similarity[edge[0], edge[1]],
                           neighbors=neighbors, degree=support.sum(1)))
    return graphs


class _SparseAggregate(torch.autograd.Function):
    """Sparse multiplication with an edgewise, memory-bounded value gradient.

    Avoids constructing a (M*M*N)^2 dense gradient on a small GPU. Chunking is
    only in backward; it does not change the graph or truncate any messages.
    """
    @staticmethod
    def forward(ctx, edge, weight, x):
        adjacency = torch.sparse_coo_tensor(edge, weight, (len(x), len(x)))
        ctx.save_for_backward(edge, weight, x)
        return torch.sparse.mm(adjacency, x)

    @staticmethod
    def backward(ctx, grad):
        edge, weight, x = ctx.saved_tensors
        dx = dw = None
        if ctx.needs_input_grad[2]:
            transpose = torch.sparse_coo_tensor(edge.flip(0), weight, (len(x), len(x)))
            dx = torch.sparse.mm(transpose, grad)
        if ctx.needs_input_grad[1]:
            parts = []
            for start in range(0, edge.shape[1], 16384):
                i, j = edge[:, start:start + 16384]
                parts.append((grad[i] * x[j]).sum(-1))
            dw = torch.cat(parts)
        return None, dw, dx


def sparse_aggregate(edge, weight, x):
    return _SparseAggregate.apply(edge, weight, x)


def conformal_maps(parameters):
    """Four numbers: send/receive angle and send/receive scale logits.

    Positive scales in (0,2), initially near 1; rotations have determinant +1.
    This parametrization approximates suppression but never produces exact zero.
    """
    angle = math.pi * torch.tanh(parameters[..., :2])
    scale = 2 * torch.sigmoid(parameters[..., 2:])
    cos, sin = angle.cos(), angle.sin()
    rotation = torch.stack((cos, -sin, sin, cos), -1).reshape(*angle.shape, 2, 2)
    maps = scale[..., None, None] * rotation
    return maps[..., 0, :, :], maps[..., 1, :, :], scale[..., 0], scale[..., 1]


def cooperative_operator(x, edge, weight, send, receive, send_scale, receive_scale):
    """Weighted (Delta_in)^T Delta_out, with D+I normalization.

    x: [group*N,2,H/2], maps: [group*N,2,2]. Symmetric edge weights required.
    Both Laplacians have off-diagonal -w_ij T_i^T S_j; their diagonals differ.
    One cooperative layer therefore contains TWO graph passes.
    """
    degree = x.new_zeros(len(x)).index_add(0, edge[0], weight)

    def apply(values, scale):
        diagonal = degree * scale.square()
        inv = (diagonal + 1).rsqrt()[:, None, None]
        normalized = values * inv
        messages = torch.bmm(send, normalized)
        aggregated = sparse_aggregate(edge, weight, messages.flatten(1)).reshape_as(values)
        received = torch.bmm(receive.transpose(-1, -2), aggregated)
        return (diagonal[:, None, None] * normalized - received) * inv

    return apply(apply(x, send_scale), receive_scale)


class RelationContentPropagation(nn.Module):
    def __init__(self, hidden, modalities, samples, graphs, mode):
        super().__init__()
        if mode not in ('shared', 'content', 'static', 'dynamic') or hidden % 2:
            raise ValueError('Invalid mode or odd hidden dimension')
        self.hidden, self.modalities, self.samples, self.mode = hidden, modalities, samples, mode
        # All shared components are constructed before the mode-specific maps.
        self.input_norm = nn.LayerNorm(hidden)
        self.edge_query = nn.Linear(hidden, 16, bias=False)
        self.edge_similarity = nn.Parameter(torch.ones(modalities, modalities))
        self.edge_bias = nn.Parameter(torch.zeros(modalities, modalities))
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.output_norm = nn.LayerNorm(hidden)
        self.step_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))
        edges, similarities, groups = [], [], []
        for m, graph in enumerate(graphs):
            for n in range(modalities):
                group = m * modalities + n
                edges.append(graph['edge'] + group * samples)
                similarities.append(graph['similarity'])
                groups.append(torch.full((graph['edge'].shape[1],), group, dtype=torch.long))
        self.register_buffer('edge', torch.cat(edges, 1), persistent=False)
        self.register_buffer('similarity', torch.cat(similarities), persistent=False)
        self.register_buffer('edge_group', torch.cat(groups), persistent=False)
        if mode == 'shared':
            self.map_parameters = nn.Parameter(torch.zeros(4))
        else:
            if mode == 'content':
                # Same active parameter budget as two M-by-8 type embeddings,
                # but all 16 coordinates encode content n, never relation m.
                self.content_id = nn.Embedding(modalities, 16)
            else:
                self.relation_id = nn.Embedding(modalities, 8)
                self.content_id = nn.Embedding(modalities, 8)
            if mode == 'dynamic':
                self.state_project = nn.Linear(hidden, 16)
            self.conditioner = nn.Sequential(nn.Linear(32 if mode == 'dynamic' else 16, 32),
                                             nn.GELU(), nn.Linear(32, 4))
            # Near-identity initial maps; no permanently inactive parameters.
            nn.init.normal_(self.conditioner[-1].weight, std=.002)
            nn.init.zeros_(self.conditioner[-1].bias)

    def map_values(self, z):
        """Return [relation,content,patient,4]; each endpoint gets its own map."""
        m, count = self.modalities, self.samples
        if self.mode == 'shared':
            return self.map_parameters.expand(m, m, count, 4)
        ids = torch.arange(m, device=z.device)
        if self.mode == 'content':
            content = self.content_id(ids)[None, :, None, :].expand(m, m, count, -1)
            return self.conditioner(content)
        relation = self.relation_id(ids)[:, None, None, :].expand(m, m, count, -1)
        content = self.content_id(ids)[None, :, None, :].expand(m, m, count, -1)
        inputs = [relation, content]
        if self.mode == 'dynamic':
            inputs.append(self.state_project(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1))
        return self.conditioner(torch.cat(inputs, -1))

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('This transductive layer requires the saved full cohort and row order')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = self.edge
        affinity = (q[i] * q[j]).sum(-1)
        weight = torch.sigmoid(affinity + self.similarity * self.edge_similarity.flatten()[self.edge_group]
                               + self.edge_bias.flatten()[self.edge_group])
        send, receive, cs, ct = conformal_maps(self.map_values(z).reshape(-1, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = cooperative_operator(values.reshape(-1, 2, h//2), self.edge, weight,
                                        send, receive, cs, ct)
        delta = diffused.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        return self.output_norm(tokens - self.step_logit.sigmoid() * self.output(F.gelu(delta)))


class PatientOnlyPropagation(nn.Module):
    """From-training graph ablation: retain exactly the same token LayerNorm."""
    def __init__(self, output_norm):
        super().__init__()
        self.output_norm = output_norm

    def forward(self, tokens):
        return self.output_norm(tokens)


class RelationSheafModel(MSCQGN):
    """Existing encoder/readout with early cooperative propagation and an MLP."""
    def __init__(self, dd, cfg, graphs, mode, training_seed=None):
        super().__init__(dd, cfg.Hidden_size, cfg.Drop_rate, cfg.ChebGCN_K,
                         num_layers=cfg.num_layers, num_heads=cfg.num_heads,
                         input_noise_std=cfg.input_noise_std, drop_path=cfg.drop_path,
                         global_word_emb=cfg.global_word_emb, semantic_branch=cfg.semantic_branch,
                         adj_mode=cfg.adj_mode, modal_structure=cfg.modal_structure,
                         readout_mode=cfg.readout_mode)
        if (self.semantic_branch, self.modal_structure, self.readout_mode) != ('both', 'modal', 'class_query'):
            raise ValueError('First-round extension requires the original modal/class-query readout')
        del self.Adj_Learning, self.GCN, self.fixed_adj
        # Forking preserves the training RNG stream across the three variants.
        extra_seed = cfg.seed if training_seed is None else training_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2901 + extra_seed)
            self.propagation = RelationContentPropagation(cfg.Hidden_size, self._modal_num,
                                                         dd['Sample_Num'], graphs,
                                                         'static' if mode == 'self_only' else mode)
            if mode == 'self_only':
                # No inactive graph weights remain in the optimizer/checkpoint.
                # Construction first preserves all common initialization/RNG.
                self.propagation = PatientOnlyPropagation(self.propagation.output_norm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2902 + extra_seed)
            self.classifier = nn.Sequential(nn.Linear(cfg.Hidden_size, cfg.Hidden_size//2),
                                             nn.ReLU(), nn.Dropout(cfg.Drop_rate),
                                             nn.Linear(cfg.Hidden_size//2, self._Label_num))

    def forward(self, x_raw):
        x = self.Feature_Modal(x_raw)
        if self.training:
            std = self._modal_noise_std[self.feature_to_modal] * self.noise_scale
            x = x + torch.randn_like(x) * std.view(1, -1)
        gate = self.modal_gate_logit.sigmoid()
        tokens = self.modal_token_encoder(x) * gate.view(1, -1, 1)
        tokens = self.propagation(tokens)
        for block in self.shared_transformer:
            tokens = block(tokens)
        embeddings = [pool(tokens) for pool in self.label_pools]
        auxiliary = [head(z) for head, z in zip(self._Auxi_classifier, embeddings)]
        semantic = self.Message_MLP(torch.cat(embeddings, -1))
        own = self.Global_Message(x * gate[self.feature_to_modal].view(1, -1))
        return self.classifier(semantic + own), embeddings, auxiliary
