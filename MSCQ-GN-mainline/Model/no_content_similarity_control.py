"""Matched C ablation: omit only the current-content edge affinity.

Every relation/content path, learned source/content coefficient, map, and
cooperative normalization is retained. This does not remove all content
dependence: edge_similarity and edge_bias still have one entry per (r, c).
"""
import torch
from torch.nn import functional as F

from .relation_sheaf import (
    RelationContentPropagation, RelationSheafModel, conformal_maps,
    cooperative_operator,
)


class NoContentSimilarityPropagation(RelationContentPropagation):
    """Original content propagation without the learned current-content cosine."""

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('This transductive layer requires the saved full cohort and row order')
        z = self.input_norm(tokens)
        weight = torch.sigmoid(self.similarity * self.edge_similarity.flatten()[self.edge_group]
                               + self.edge_bias.flatten()[self.edge_group])
        send, receive, cs, ct = conformal_maps(self.map_values(z).reshape(-1, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = cooperative_operator(values.reshape(-1, 2, h//2), self.edge, weight,
                                        send, receive, cs, ct)
        delta = diffused.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        return self.output_norm(tokens - self.step_logit.sigmoid() * self.output(F.gelu(delta)))


class NoContentSimilarityControlModel(RelationSheafModel):
    """Construct native C first; remove only its exclusively used query layer.

    Native construction precedes the mutation so initialization of every
    common tensor and the CPU/CUDA random streams remain unchanged. No trained
    C weights are used. Omitting the unused query layer keeps every remaining
    optimizer parameter active and reduces the H=96 budget by 1,536 weights.
    """

    def __init__(self, dd, cfg, graphs, mode='NO_CONTENT_SIMILARITY', training_seed=0):
        if mode not in ('C', 'NO_CONTENT_SIMILARITY'):
            raise ValueError(f'Unknown content-similarity control mode: {mode}')
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.no_content_similarity_mode = mode
        if mode == 'NO_CONTENT_SIMILARITY':
            propagation = self.propagation
            if propagation.mode != 'content':
                raise ValueError('Requires the original content-conditioned C propagation')
            propagation.__class__ = NoContentSimilarityPropagation
            del propagation.edge_query
