"""Learned shared-map control for the current content-conditioned C model.

Only the input of the send/receive map conditioner is shared across content
modalities. The original conditioner, separate sending/receiving outputs,
relation/content edge weights, 36 logical propagation paths, normalization,
residual, fusion, readout and classifier are retained.

The shared embedding starts at the deterministic mean of the native content
embeddings. This preserves all OTHER initial parameter values, their order and
the construction RNG stream; it does not claim the initial function equals C
with its original, untied embeddings. There is no capacity padding: a cohort
with M modalities loses exactly (M-1)*16 active parameters (80 for TAD's six).
"""
import torch
from torch import nn

from .relation_sheaf import RelationContentPropagation, RelationSheafModel


class SharedMappingPropagation(RelationContentPropagation):
    """Same cooperative operator, with one learned S/T rule for all contents.

S and T remain separate outputs of the original four-output conditioner.
Both are shared across relation sources and patients, as well as contents.
Current patient states continue to influence edge weights through the inherited
forward method; they do not condition these sending/receiving maps.
"""

    def map_values(self, z):
        m, count = self.modalities, self.samples
        # Replacing the embedding retains its original registration position.
        # Expanding BEFORE the unchanged conditioner preserves the native map
        # evaluation layout and gives every logical path the same learned rule.
        shared = self.content_id.weight[None, :, None, :].expand(m, m, count, -1)
        return self.conditioner(shared)


class SharedMappingControlModel(RelationSheafModel):
    """Native C or its learned shared-map ablation; no original source changes."""

    def __init__(self, dd, cfg, graphs, mode='SHARED_MAP', training_seed=0):
        if mode not in ('C', 'SHARED_MAP'):
            raise ValueError(f'Unknown shared-mapping control mode: {mode}')
        # Construct native content C first. Its forked initialization and every
        # common parameter/module order are retained in both experimental arms.
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.shared_mapping_mode = mode
        if mode == 'SHARED_MAP':
            propagation = self.propagation
            if (type(propagation) is not RelationContentPropagation
                    or propagation.mode != 'content'
                    or propagation.content_id.embedding_dim != 16):
                raise ValueError('Shared-map control requires the native content C propagation')
            with torch.no_grad():
                initial_shared = propagation.content_id.weight.mean(dim=0, keepdim=True).clone()
            # Supplying _weight via from_pretrained avoids a random initializer;
            # freeze=False makes all 16 shared coordinates trainable.
            propagation.content_id = nn.Embedding.from_pretrained(initial_shared, freeze=False)
            # Only map_values is overridden. No duplicate operator or alternate
            # forward path can drift away from the original C implementation.
            propagation.__class__ = SharedMappingPropagation
