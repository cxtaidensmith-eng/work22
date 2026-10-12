"""Joint source-patient/source-modality routing on the original C backbone.

Candidate support comes exclusively from the original per-modality graphs.
Every receiving token competes over its own patient's modal tokens, eligible
source-patient modal tokens, and a learned-key/zero-value null source. No labels
or stochastic routing enter this layer. ``own`` and ``joint`` have identical
trainable parameters; only their non-persistent source-support buffers differ.
"""
import math

import torch
from torch import nn

from .relation_sheaf import RelationSheafModel, sparse_aggregate


class JointEvidenceRouting(nn.Module):
    def __init__(self, hidden, modalities, samples, graphs, mode, query_size=16):
        super().__init__()
        if mode not in ('own', 'joint'):
            raise ValueError('Routing mode must be own or joint')
        if min(hidden, modalities, samples, query_size) < 1 or len(graphs) != modalities:
            raise ValueError('Positive dimensions and one candidate graph per modality required')
        self.hidden, self.modalities, self.samples = hidden, modalities, samples
        self.mode, self.query_size = mode, query_size
        self.input_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, query_size, bias=False)
        self.key = nn.Linear(hidden, query_size, bias=False)
        self.receiver_id = nn.Embedding(modalities, query_size)
        self.source_id = nn.Embedding(modalities, query_size)
        nn.init.normal_(self.receiver_id.weight, std=.02)
        nn.init.normal_(self.source_id.weight, std=.02)
        self.null_key = nn.Parameter(torch.empty(query_size))
        nn.init.normal_(self.null_key, std=.02)
        # Last column is the null source. All entries are used in both arms.
        self.modal_bias = nn.Parameter(torch.zeros(modalities, modalities + 1))
        self.values = nn.ModuleList(nn.Linear(hidden, hidden, bias=False)
                                    for _ in range(modalities))
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.output_norm = nn.LayerNorm(hidden)
        self.step_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))

        # Flatten in patient-major order: token(i,m) = i*M + m.
        patient = torch.arange(samples, dtype=torch.long)
        receiver = torch.arange(modalities, dtype=torch.long)
        source = torch.arange(modalities, dtype=torch.long)
        own_row = (patient[:, None, None] * modalities +
                   receiver[None, :, None]).expand(samples, modalities, modalities)
        own_col = (patient[:, None, None] * modalities +
                   source[None, None, :]).expand(samples, modalities, modalities)
        edges = [torch.stack((own_row.reshape(-1), own_col.reshape(-1)))]
        # Graph n determines eligibility for the source token (j,n), regardless
        # of the receiving modality m. Graph orientation: row i receives j.
        for n, graph in enumerate(graphs):
            edge = graph['edge'].detach().to(device='cpu', dtype=torch.long)
            if edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError('Each graph edge must have shape [2,E]')
            if edge.numel() and (edge.min() < 0 or edge.max() >= samples):
                raise ValueError('Candidate graph index outside the cohort')
            if edge.shape[1] != torch.unique(edge, dim=1).shape[1]:
                raise ValueError('Duplicate candidate edges would duplicate attention sources')
            if bool((edge[0] == edge[1]).any()):
                raise ValueError('Candidate supports must exclude self edges; own tokens are explicit')
            if mode == 'joint':
                row = edge[0, :, None] * modalities + receiver[None, :]
                col = (edge[1, :, None] * modalities + n).expand_as(row)
                edges.append(torch.stack((row.reshape(-1), col.reshape(-1))))
        count = samples * modalities
        edges.append(torch.stack((torch.arange(count), torch.full((count,), count))))
        edge = torch.cat(edges, dim=1)
        self.register_buffer('edge', edge, persistent=False)
        self.register_buffer('receiver_modality', edge[0].remainder(modalities), persistent=False)
        source_modality = edge[1].remainder(modalities)
        source_modality[edge[1] == count] = modalities
        self.register_buffer('source_modality', source_modality, persistent=False)

    def _check_tokens(self, tokens):
        if tuple(tokens.shape) != (self.samples, self.modalities, self.hidden):
            raise ValueError('This transductive layer requires the saved full cohort and row order')

    def projected_evidence(self, tokens):
        """Return flattened queries, keys including null, and source values."""
        self._check_tokens(tokens)
        z = self.input_norm(tokens)
        query = (self.query(z) + self.receiver_id.weight[None]).flatten(0, 1)
        key = (self.key(z) + self.source_id.weight[None]).flatten(0, 1)
        key = torch.cat((key, self.null_key[None]), dim=0)
        values = torch.stack([layer(z[:, n]) for n, layer in enumerate(self.values)], dim=1)
        values = values.flatten(0, 1)
        values = torch.cat((values, values.new_zeros(1, self.hidden)), dim=0)
        return query, key, values

    def attention(self, tokens):
        """Sparse weights, normalized jointly over every candidate for a token."""
        query, key, values = self.projected_evidence(tokens)
        row, col = self.edge
        score = (query[row] * key[col]).sum(-1) / math.sqrt(self.query_size)
        score = score + self.modal_bias[self.receiver_modality, self.source_modality]
        # Detaching the stabilizing maximum preserves the softmax derivative;
        # no E-by-H messages or quadratic dense attention matrix are formed.
        maxima = score.new_full((len(query),), -torch.inf)
        maxima.scatter_reduce_(0, row, score.detach(), reduce='amax', include_self=True)
        exponential = (score - maxima[row]).exp()
        denominator = score.new_zeros(len(query)).index_add(0, row, exponential)
        weight = exponential / denominator[row]
        return weight, values

    def forward(self, tokens):
        weight, values = self.attention(tokens)
        aggregated = sparse_aggregate(self.edge, weight, values)[:-1].reshape_as(tokens)
        return self.output_norm(tokens + self.step_logit.sigmoid() * self.output(aggregated))

    @torch.no_grad()
    def attention_diagnostics(self, tokens):
        """Mean attention masses; overlapping and exclusive categories labelled.

        ``own + neighbor + null == 1``; ``crossmodal`` overlaps own/neighbor.
        Each ``by_receiver_modality`` value averages patients, preserving the
        receiving-modality axis. This routine leaves the forward API unchanged.
        """
        weight, _ = self.attention(tokens)
        row, col = self.edge
        null = col == self.samples * self.modalities
        own = (~null) & (row.div(self.modalities, rounding_mode='floor') ==
                         col.div(self.modalities, rounding_mode='floor'))
        neighbor = (~null) & (~own)
        cross = (~null) & (self.receiver_modality != self.source_modality)
        masks = dict(own=own, neighbor=neighbor, null=null, crossmodal=cross,
                     own_same=own & ~cross, own_cross=own & cross,
                     neighbor_same=neighbor & ~cross, neighbor_cross=neighbor & cross)
        result = {}
        for name, mask in masks.items():
            mass = weight.new_zeros(self.samples * self.modalities)
            mass.index_add_(0, row, weight * mask.to(weight.dtype))
            mass = mass.reshape(self.samples, self.modalities)
            result[name] = dict(mean=float(mass.mean()),
                                by_receiver_modality=mass.mean(0).cpu().tolist())
        return result


class JointEvidenceRoutingModel(RelationSheafModel):
    """Original C encoders/readout/objective with only propagation replaced."""
    def __init__(self, dd, cfg, graphs, mode, training_seed=None):
        if mode not in ('own', 'joint'):
            raise ValueError('Routing mode must be own or joint')
        super().__init__(dd, cfg, graphs, mode='content', training_seed=training_seed)
        old_propagation = self.propagation
        extra_seed = cfg.seed if training_seed is None else training_seed
        # C's legacy constructor affects the CUDA RNG while initializing inside
        # CPU-only forks. Preserve the resulting stream exactly: replacing the
        # layer must neither repeat that behavior nor consume extra RNG draws.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2903 + extra_seed)
            self.propagation = JointEvidenceRouting(cfg.Hidden_size, self._modal_num,
                                                   dd['Sample_Num'], graphs, mode)
        self.propagation.input_norm.load_state_dict(old_propagation.input_norm.state_dict())
        self.propagation.output_norm.load_state_dict(old_propagation.output_norm.state_dict())
        self.routing_mode = mode
