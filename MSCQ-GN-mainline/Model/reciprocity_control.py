"""Symmetric reciprocal-neighbor metadata control for the frozen C architecture.

An edge keeps its original endpoints, prior similarity and two-pass operator.
Only its logit gains tanh(beta[relation, content]) * (mutual - relation_mean).
The shuffled arm permutes labels over unordered edges with a private CPU RNG,
preserving relation-wise counts and reverse-edge equality, not node-wise counts.
No diagnostic labels or predictions are used to construct this metadata.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import (
    RelationSheafModel, RelationContentPropagation, conformal_maps,
    cooperative_operator,
)

SHUFFLE_SEED = 43017


def relation_marks(graph, samples, shuffled=False, seed=SHUFFLE_SEED):
    edge = graph['edge'].cpu()
    neighbors = graph['neighbors'].cpu()
    selected = torch.zeros(samples, samples, dtype=torch.bool)
    selected.scatter_(1, neighbors, True)
    union = selected | selected.T
    expected = union.nonzero().T.contiguous()
    if not torch.equal(edge, expected) or bool(union.diagonal().any()):
        raise ValueError('Expected the original sorted symmetric top-k union')
    i, j = edge
    keys, inverse = torch.unique(torch.minimum(i, j)*samples + torch.maximum(i, j),
                                 sorted=True, return_inverse=True)
    marks = (selected & selected.T)[keys//samples, keys%samples].float()
    if shuffled:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        marks = marks[torch.randperm(len(marks), generator=generator)]
    expanded = marks[inverse]
    return expanded - marks.mean(), dict(
        unordered_edges=len(marks), mutual_edges=int(marks.sum()),
        unilateral_edges=int(len(marks)-marks.sum()), mutual_fraction=float(marks.mean()),
    )


class ReciprocityPropagation(RelationContentPropagation):
    def edge_weights(self, z):
        count, m, _ = z.shape
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = self.edge
        affinity = (q[i] * q[j]).sum(-1)
        original = (affinity + self.similarity * self.edge_similarity.flatten()[self.edge_group]
                    + self.edge_bias.flatten()[self.edge_group])
        adjustment = self.reciprocity_bias.tanh().flatten()[self.edge_group] * self.reciprocity_code
        return torch.sigmoid(original + adjustment)

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('Reciprocity control requires the saved cohort and row order')
        z = self.input_norm(tokens)
        weight = self.edge_weights(z)
        send, receive, cs, ct = conformal_maps(self.map_values(z).reshape(-1, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = cooperative_operator(values.reshape(-1, 2, h//2), self.edge, weight,
                                       send, receive, cs, ct)
        delta = diffused.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        return self.output_norm(tokens - self.step_logit.sigmoid() * self.output(F.gelu(delta)))


class ReciprocityControlModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, mode='REAL', training_seed=0):
        if mode not in ('C', 'REAL', 'SHUFFLED'):
            raise ValueError(mode)
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.reciprocity_mode = mode
        if mode == 'C':
            return
        layer = self.propagation
        if type(layer) is not RelationContentPropagation or layer.mode != 'content':
            raise ValueError('Expected native content-conditioned C')
        codes, statistics = [], []
        for relation, graph in enumerate(graphs):
            code, stats = relation_marks(graph, layer.samples, mode == 'SHUFFLED',
                                         SHUFFLE_SEED + relation)
            codes.extend([code] * layer.modalities)
            statistics.append(stats)
        # Zero initialization consumes no random numbers and exactly recovers C.
        layer.reciprocity_bias = nn.Parameter(torch.zeros(layer.modalities, layer.modalities))
        layer.register_buffer('reciprocity_code', torch.cat(codes), persistent=True)
        layer.__class__ = ReciprocityPropagation
        self.reciprocity_statistics = statistics
