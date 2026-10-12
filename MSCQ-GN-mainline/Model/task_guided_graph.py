"""Independent multi-head graph metric with optional TRAIN-pair supervision.

IDGL-inspired metric; SPromptGL-inspired graph target, not full reproductions.
The final selection score is supervised, so the objective acts on the scores
actually used for top-k. C edge weights, maps and propagation remain unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .relation_sheaf import RelationSheafModel, conformal_maps
from .selection_only import SelectionOnlyPropagation
from .learned_topology import select_neighbors, dense_cooperative_operator


def balanced_relation_loss(score, labels, fit):
    """Class-balanced MSE over unordered FIT-FIT pairs only; exclude diagonal.

    Each within-class group gets equal positive mass; each between-class group
    gets equal negative mass. Positives and negatives each carry half the loss.
    Only labels[fit] are ever read, including when constructing class groups.
    """
    indices = fit.nonzero().flatten()
    y = labels[indices]
    values = score[:, indices][:, :, indices]
    upper = torch.ones(len(y), len(y), dtype=torch.bool, device=y.device).triu(1)
    classes = y.unique(sorted=True)
    positive, negative = [], []
    for i, left in enumerate(classes):
        same = (y[:, None] == left) & (y[None, :] == left) & upper
        if bool(same.any()):
            positive.append((values[:, same]-1).square().mean())
        for right in classes[i+1:]:
            different = (((y[:, None] == left) & (y[None, :] == right)) |
                         ((y[:, None] == right) & (y[None, :] == left))) & upper
            if bool(different.any()):
                negative.append(values[:, different].square().mean())
    if not positive or not negative:
        raise ValueError('Need within-class and between-class training pairs')
    return .5 * (torch.stack(positive).mean() + torch.stack(negative).mean())


class TaskGuidedPropagation(SelectionOnlyPropagation):
    def __init__(self, hidden, modalities, samples, graphs, raw_x, modal_index,
                 k=10, temperature=.1, heads=4, metric_dim=16):
        super().__init__(hidden, modalities, samples, graphs, raw_x, modal_index,
                         'learned', k, temperature)
        self.graph_query = nn.Linear(hidden, metric_dim, bias=False)
        if self.graph_query.weight.shape != self.edge_query.weight.shape:
            raise ValueError('This pilot initializes the independent query from C query')
        with torch.no_grad(): self.graph_query.weight.copy_(self.edge_query.weight)
        self.metric_weights = nn.Parameter(torch.ones(modalities, heads, metric_dim))
        nn.init.normal_(self.metric_weights, mean=1., std=.05)
        self.last_selection_score = None

    def selection_scores(self, z):
        projected = self.graph_query(z).permute(1, 0, 2)
        query = F.normalize(projected[:, None] * self.metric_weights[:, :, None], dim=-1)
        affinity = (query @ query.transpose(-1, -2)).mean(1)
        return self.raw_similarity + self.relation_blend.tanh()[:, None, None] * affinity

    def forward(self, tokens):
        n, m, h = tokens.shape
        if (n, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('Original full-cohort transductive order required')
        z = self.input_norm(tokens)
        query = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        affinity = query @ query.transpose(-1, -2)
        score = self.selection_scores(z)
        self.last_selection_score = score
        if self.topology_mode == 'learned':
            mask, support, neighbors = select_neighbors(score, self.k, self.temperature, self.training)
        else:
            support, neighbors = self.fixed_support, self.fixed_neighbors
            mask = support.to(tokens.dtype)
        self.last_support, self.last_neighbors = support.detach(), neighbors.detach()
        adjacency = (self.edge_weights(affinity) * mask[:, None]).reshape(m*m, n, n)
        send, receive, cs, ct = conformal_maps(self.map_values(z).reshape(m*m, n, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = dense_cooperative_operator(values.reshape(m*m, n, 2, h//2),
                                              adjacency, send, receive, cs, ct)
        delta = diffused.reshape(m, m, n, h).mean(0).permute(1, 0, 2)
        return self.output_norm(tokens - self.step_logit.sigmoid() * self.output(F.gelu(delta)))


class TaskGuidedGraphModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, raw_x, seed, k=10, temperature=.1, heads=4):
        super().__init__(dd, cfg, graphs, 'content', training_seed=seed)
        old_state = self.propagation.state_dict()
        # manual_seed also affects CUDA; preserve every device's RNG stream.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(2901 + seed)
            self.propagation = TaskGuidedPropagation(cfg.Hidden_size, self._modal_num, dd['Sample_Num'],
                graphs, raw_x, dd['Modal_Index'], k, temperature, heads)
        incompatible = self.propagation.load_state_dict(old_state, strict=False)
        assert set(incompatible.missing_keys) == {'relation_blend','graph_query.weight','metric_weights'}
        assert not incompatible.unexpected_keys
        with torch.no_grad():
            self.propagation.graph_query.weight.copy_(self.propagation.edge_query.weight)
