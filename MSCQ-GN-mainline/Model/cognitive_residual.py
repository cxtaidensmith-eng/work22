"""Explicit cognitive prediction plus a zero-initialized joint supplement.

Both paths are trained from scratch, end to end, with final CE only. A direct
path is retained architecturally; its accuracy is not guaranteed preserved.
The supplement may learn cognitive effects too: it is not a pure interaction.
"""
from copy import deepcopy

import torch
from torch import nn

from .subset_fusion import SubsetFusionModel


class CognitiveResidualModel(SubsetFusionModel):
    def __init__(self, dd, cfg, companion='PET', training_seed=0):
        if companion == 'COGNITIVE_TEST':
            raise ValueError('Companion must differ from the cognitive modality')
        super().__init__(dd, cfg, ('COGNITIVE_TEST', companion), training_seed)
        cognitive = SubsetFusionModel(dd, cfg, ('COGNITIVE_TEST',), training_seed)
        self.cognitive_head = deepcopy(cognitive.head)
        self.cognitive_index = self.modalities.index('COGNITIVE_TEST')
        self.companion = companion
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_raw):
        output = super().forward(x_raw)
        supplement = output['logits']
        direct = self.cognitive_head(output['tokens'][:, self.cognitive_index])
        output.update(logits=direct + supplement,
                      unimodal_logits=direct[:, None], pair_logits=supplement[:, None])
        return output

    def parameter_counts(self):
        counts = super().parameter_counts()
        counts.update(cognitive_head=sum(p.numel() for p in self.cognitive_head.parameters()),
                      supplement_head=counts['head'], auxiliary_heads=0,
                      inference_parameters=counts['total'])
        return counts
