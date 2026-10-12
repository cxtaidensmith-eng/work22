"""Cognitive prediction plus a token-centered joint supplement.

The shared joint head is evaluated with the observed tokens and with the
companion token set to zero, after normalization, gating and dropout. Both
evaluations retain gradients. This only anchors the supplement at a zero
companion token; it does not identify a pure or causal PET contribution.
The final joint-head bias cancels, but is retained for identical parameters
and initialization to CognitiveResidualModel.
"""
from .cognitive_residual import CognitiveResidualModel


class CenteredCognitiveResidualModel(CognitiveResidualModel):
    def __init__(self, dd, cfg, companion='PET', training_seed=0):
        super().__init__(dd, cfg, companion, training_seed)
        self.companion_index = self.modalities.index(companion)

    def reference_joint_from_tokens(self, tokens):
        """Evaluate the same head with just the companion token zeroed."""
        reference = tokens.clone()
        reference[:, self.companion_index] = 0
        return self.head(reference.flatten(start_dim=1))

    def supplement_from_tokens(self, tokens):
        """Expose the centered supplement for token-level diagnostics."""
        return (self.head(tokens.flatten(start_dim=1))
                - self.reference_joint_from_tokens(tokens))

    def forward(self, x_raw):
        # The parent performs the sole encoder/noise/dropout pass, and the
        # first joint-head evaluation. Only the reference head is added here.
        output = super().forward(x_raw)
        raw_joint = output['pair_logits'][:, 0]
        reference_joint = self.reference_joint_from_tokens(output['tokens'])
        supplement = raw_joint - reference_joint
        direct = output['unimodal_logits'][:, 0]
        output.update(logits=direct + supplement,
                      pair_logits=supplement[:, None],
                      raw_joint_logits=raw_joint,
                      reference_joint_logits=reference_joint)
        return output
