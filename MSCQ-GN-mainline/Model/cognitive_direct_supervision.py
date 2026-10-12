"""Train-mask-only supervision of the existing cognitive prediction path."""
import math

from torch.nn import functional as F


def classification_objective(output, labels, train_mask, class_weights,
                             direct_weight=.05):
    """Return final CE, direct-path CE and their predeclared weighted sum.

    This loss introduces no prediction heads or forward operations. The direct
    output is the cognitive path already used by CognitiveResidualModel.
    """
    if not math.isfinite(direct_weight) or direct_weight < 0:
        raise ValueError('direct_weight must be finite and nonnegative')
    main = F.cross_entropy(output['logits'][train_mask], labels[train_mask],
                           weight=class_weights, label_smoothing=.05)
    direct = F.cross_entropy(output['unimodal_logits'][train_mask, 0],
                             labels[train_mask], weight=class_weights,
                             label_smoothing=.05)
    total = main if direct_weight == 0 else main + direct_weight * direct
    return dict(total=total, main=main, direct=direct)
