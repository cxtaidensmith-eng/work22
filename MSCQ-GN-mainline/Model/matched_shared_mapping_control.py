"""Complete the original D1/shared-map factorial without changing old sources.

The graph support, compact six active relation/content groups, scalar edge
weights and one-active-relation normalization are exactly the original D1.
Only the content ID used by the existing 16->32->4 map generator is shared.
Sending and receiving maps remain distinct, learnable outputs.
"""
import torch
from torch import nn

from .c_mechanism_controls import MechanismControlModel


class SharedContentEmbedding(nn.Embedding):
    """One active learned row; every modality ID addresses that shared row."""

    def forward(self, indices):
        return super().forward(torch.zeros_like(indices))


class MatchedSharedMappingControlModel(MechanismControlModel):
    """Original D1, optionally with its six map IDs tied to one learned ID."""

    def __init__(self, dd, cfg, graphs, mode='D1_SHARED_MAP', training_seed=0):
        if mode not in ('D1', 'D1_SHARED_MAP'):
            raise ValueError(f'Unknown matched shared-map mode: {mode}')
        # Native construction first preserves all public parameters and RNG.
        super().__init__(dd, cfg, graphs, 'D1', training_seed=training_seed)
        self.matched_shared_mapping_mode = mode
        if mode == 'D1_SHARED_MAP':
            propagation = self.propagation
            if propagation.control != 'D1' or propagation.content_id.embedding_dim != 16:
                raise ValueError('Requires the original D1 content mapping generator')
            with torch.no_grad():
                initial = propagation.content_id.weight.mean(0, keepdim=True).clone()
            # Construction is on CPU, before .to(device), as in SHARED_MAP.
            # _weight avoids random initialization; freeze=False keeps it active.
            propagation.content_id = SharedContentEmbedding.from_pretrained(initial, freeze=False)
            # Keep MechanismPropagation.forward unchanged: it calls this shared
            # lookup with arange(M), then runs the original compact D1 operator.
