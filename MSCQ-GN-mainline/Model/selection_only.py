"""Learn neighbor support while retaining C's original edge-weight formula."""
import torch
from torch.nn import functional as F
from .relation_sheaf import RelationSheafModel, conformal_maps
from .learned_topology import (LearnedTopologyPropagation, select_neighbors,
                               dense_cooperative_operator)


class SelectionOnlyPropagation(LearnedTopologyPropagation):
    def __init__(self, hidden, modalities, samples, graphs, raw_x, modal_index,
                 topology_mode, k=10, temperature=.1):
        super().__init__(hidden, modalities, samples, graphs, raw_x, modal_index,
                         topology_mode, k, temperature)
        if topology_mode == 'fixed':
            # Fixed arm is C itself: no unused trainable selector coefficients.
            del self.relation_blend
            self.register_buffer('relation_blend', torch.zeros(modalities), persistent=False)

    def edge_weights(self, affinity):
        # Relation m supplies raw cosine; content n supplies query affinity.
        # Selector coefficients NEVER enter this edge-weight expression.
        return torch.sigmoid(affinity[None] +
            self.edge_similarity[:, :, None, None] * self.raw_similarity[:, None] +
            self.edge_bias[:, :, None, None])

    def forward(self, tokens):
        n, m, h = tokens.shape
        if (n, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('Original full-cohort transductive order required')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        affinity = q @ q.transpose(-1, -2)
        if self.topology_mode == 'learned':
            selection_score = self.raw_similarity + self.relation_blend.tanh()[:, None, None] * affinity
            mask, support, neighbors = select_neighbors(selection_score, self.k,
                                                        self.temperature, self.training)
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


class SelectionOnlyModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, raw_x, seed, topology_mode, k=10, temperature=.1):
        super().__init__(dd, cfg, graphs, 'content', training_seed=seed)
        old_state = self.propagation.state_dict()
        # manual_seed touches CUDA too. Restore every device after rebuilding
        # propagation, preserving the exact training RNG left by original C.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(2901 + seed)
            self.propagation = SelectionOnlyPropagation(cfg.Hidden_size, self._modal_num,
                dd['Sample_Num'], graphs, raw_x, dd['Modal_Index'], topology_mode, k, temperature)
        incompatible = self.propagation.load_state_dict(old_state, strict=False)
        expected = ['relation_blend'] if topology_mode == 'learned' else []
        assert incompatible.missing_keys == expected and not incompatible.unexpected_keys
