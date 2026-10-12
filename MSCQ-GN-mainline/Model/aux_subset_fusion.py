"""Training-only unimodal heads on the unchanged subset fusion backbone."""
import torch
from torch import nn
from torch.nn import functional as F

from .subset_fusion import SubsetFusionModel


class AuxiliarySubsetFusionModel(SubsetFusionModel):
    def __init__(self, dd, cfg, modalities, training_seed=0):
        super().__init__(dd, cfg, modalities, training_seed)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(4100 + int(training_seed))
            self.auxiliary_heads = nn.ModuleList(
                nn.Linear(self.hidden, dd['Class_Num']) for _ in self.modalities)

    def forward(self, x_raw):
        output = super().forward(x_raw)
        output['unimodal_logits'] = torch.stack(
            [head(output['tokens'][:, i]) for i, head in enumerate(self.auxiliary_heads)], dim=1)
        return output

    def parameter_counts(self):
        counts = super().parameter_counts()
        auxiliary = sum(p.numel() for p in self.auxiliary_heads.parameters())
        counts.update(auxiliary_heads=auxiliary, inference_parameters=counts['total']-auxiliary)
        return counts


def classification_objective(output, labels, train_mask, class_weights, auxiliary_weight):
    """Final CE + weight * mean(per-modality CE); train labels only."""
    if auxiliary_weight < 0:
        raise ValueError('Auxiliary weight must be nonnegative')
    target = labels[train_mask]
    main = F.cross_entropy(output['logits'][train_mask], target,
                           weight=class_weights, label_smoothing=.05)
    auxiliary = torch.stack([
        F.cross_entropy(output['unimodal_logits'][train_mask, i], target,
                        weight=class_weights, label_smoothing=.05)
        for i in range(output['unimodal_logits'].shape[1])])
    return dict(total=main + auxiliary_weight * auxiliary.mean(),
                main=main, auxiliary=auxiliary, auxiliary_mean=auxiliary.mean())
