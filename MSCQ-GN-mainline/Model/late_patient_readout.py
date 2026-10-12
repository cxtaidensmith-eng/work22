"""Matched late readout controls for the ABIDE C diagnosis."""
import torch
from torch import nn
from torch.nn import functional as F
from .relation_sheaf import RelationSheafModel


class LatePatientReadout(nn.Module):
    """Same active weights; only P changes from I to .5 I + .5 D^-1 A.

    A is the binary union of existing modality candidates, without self edges.
    z_out = z + .1 W_out GELU(W_in P LayerNorm(z)). No learned graph or labels.
    """
    def __init__(self, hidden, samples, graphs, mode):
        super().__init__()
        if mode not in ('self', 'population'):
            raise ValueError(mode)
        self.mode = mode
        support = torch.zeros(samples, samples, dtype=torch.bool)
        for graph in graphs:
            support[graph['edge'][0], graph['edge'][1]] = True
        support.fill_diagonal_(False)
        edge = support.nonzero().T.contiguous()
        degree = support.sum(1)
        if bool((degree == 0).any()):
            raise ValueError('Late readout requires a nonempty candidate neighborhood')
        self.register_buffer('edge', edge, persistent=False)
        self.register_buffer('weight', degree[edge[0]].float().reciprocal(), persistent=False)
        self.samples = samples
        self.norm = nn.LayerNorm(hidden)
        self.input = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.scale = .1

    def forward(self, z):
        h = self.norm(z)
        if self.mode == 'population':
            adjacency = torch.sparse_coo_tensor(self.edge, self.weight, (self.samples, self.samples))
            h = .5 * h + .5 * torch.sparse.mm(adjacency, h)
        return z + self.scale * self.output(F.gelu(self.input(h)))


class LateReadoutModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, late_mode, training_seed=0):
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        original_classifier = self.classifier
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2903 + training_seed)
            late = LatePatientReadout(cfg.Hidden_size, dd['Sample_Num'], graphs, late_mode)
        # The base forward supplies exactly semantic + own to this wrapper.
        self.classifier = nn.Sequential(late, original_classifier)
