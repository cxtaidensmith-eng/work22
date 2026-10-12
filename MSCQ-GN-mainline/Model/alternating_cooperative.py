"""Two shared C propagation calls, with a fixed 2 x 2 token schedule.

No new parameters, topology, loss, or node-conditioned maps. C is a test-only
single-call mode for equivalence checks against the frozen implementation.
"""
import torch
from torch.nn import functional as F

from .relation_sheaf import RelationSheafModel, conformal_maps, cooperative_operator


def propagate(layer, tokens, weight=None, diagnostics=False):
    """Apply exactly C's operator; optionally reuse only live edge weights.

    Values and maps are evaluated on this call. An external weight is never
    detached. Both internal diffusion passes use the same weights and maps.
    """
    count, m, h = tokens.shape
    if (count, m, h) != (layer.samples, layer.modalities, layer.hidden):
        raise ValueError('Requires the original full cohort and modal token shape')
    z = layer.input_norm(tokens)
    if weight is None:
        q = F.normalize(layer.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = layer.edge
        affinity = (q[i] * q[j]).sum(-1)
        weight = torch.sigmoid(affinity + layer.similarity * layer.edge_similarity.flatten()[layer.edge_group]
                               + layer.edge_bias.flatten()[layer.edge_group])
    send, receive, cs, ct = conformal_maps(layer.map_values(z).reshape(-1, 4))
    values = layer.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
    diffused = cooperative_operator(values.reshape(-1, 2, h//2), layer.edge, weight,
                                    send, receive, cs, ct)
    delta = diffused.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
    update = layer.step_logit.sigmoid() * layer.output(F.gelu(delta))
    result = layer.output_norm(tokens - update)
    trace = None
    if diagnostics:
        with torch.no_grad():
            trace = dict(graph_update_over_input_norm=float(update.norm() / tokens.norm().clamp_min(1e-12)),
                         per_modality_update_over_input_norm=(
                             update.norm(dim=(0, 2)) / tokens.norm(dim=(0, 2)).clamp_min(1e-12)).tolist())
    return result, weight, trace


class AlternatingCooperativeModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, mode, training_seed=None):
        if mode not in ('C', 'N0', 'N1', 'A0', 'A1'):
            raise ValueError('Expected C, N0, N1, A0 or A1')
        if cfg.num_layers != 5:
            raise ValueError('This fixed experiment requires exactly five modal layers')
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.schedule = mode
        self.capture_diagnostics = False
        self.last_diagnostics = None

    def forward(self, x_raw):
        if self.capture_diagnostics and self.training:
            raise ValueError('Diagnostics are restricted to noise-free evaluation')
        self.last_diagnostics = None
        x = self.Feature_Modal(x_raw)
        if self.training:
            std = self._modal_noise_std[self.feature_to_modal] * self.noise_scale
            x = x + torch.randn_like(x) * std.view(1, -1)
        gate = self.modal_gate_logit.sigmoid()
        tokens = self.modal_token_encoder(x) * gate.view(1, -1, 1)
        tokens, w0, first = propagate(self.propagation, tokens, diagnostics=self.capture_diagnostics)
        if self.schedule.startswith('A'):
            for block in self.shared_transformer[:2]:
                tokens = block(tokens)
        if self.schedule != 'C':
            reuse = w0 if self.schedule.endswith('0') else None
            tokens, w1, second = propagate(self.propagation, tokens, reuse, self.capture_diagnostics)
            if self.capture_diagnostics:
                self.last_diagnostics = dict(first=first, second=second,
                    mean_absolute_edge_weight_change=float((w1 - w0).abs().mean()),
                    reused_same_weight_tensor=w1 is w0)
        blocks = self.shared_transformer[2:] if self.schedule.startswith('A') else self.shared_transformer
        for block in blocks:
            tokens = block(tokens)
        embeddings = [pool(tokens) for pool in self.label_pools]
        auxiliary = [head(z) for head, z in zip(self._Auxi_classifier, embeddings)]
        semantic = self.Message_MLP(torch.cat(embeddings, -1))
        own = self.Global_Message(x * gate[self.feature_to_modal].view(1, -1))
        return self.classifier(semantic + own), embeddings, auxiliary
