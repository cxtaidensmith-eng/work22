"""C with receiver-by-content control of off-diagonal input contributions.

For fixed edge weights/maps, split the COMPOSED two-pass operator into its
diagonal blocks and all other blocks, including two-hop messages. This is not
the independently trained self-only baseline: the diagonal coefficients still
depend on the graph. The original C module is left unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .relation_sheaf import (RelationSheafModel, RelationContentPropagation,
                            conformal_maps, cooperative_operator)


def composed_self_part(x, edge, weight, send, receive, send_scale, receive_scale, constant_maps=False):
    """Diagonal blocks of L_receive @ L_send, on symmetric loop-free edges."""
    count = len(x)
    i, j = edge
    degree = x.new_zeros(count).index_add(0, i, weight)
    ds, dt = degree * send_scale.square(), degree * receive_scale.square()
    ins, int_ = (ds + 1).rsqrt(), (dt + 1).rsqrt()
    direct = ds / (ds + 1) * dt / (dt + 1)
    coefficient = weight.square() * ins[i] * int_[i] * ins[j] * int_[j]
    if constant_maps:
        # C uses the same content maps at both endpoints of each group. Factor
        # their common 2x2 matrix out of the edge sum; keep the exact operator.
        transform = torch.bmm(receive.transpose(-1, -2), send)
        mass = x.new_zeros(count).index_add(0, i, coefficient)
        cycles = mass[:, None, None] * torch.bmm(transform, transform)
    else:
        tij = torch.bmm(receive[i].transpose(-1, -2), send[j])
        tji = torch.bmm(receive[j].transpose(-1, -2), send[i])
        cycles = x.new_zeros(count, 2, 2).index_add(
            0, i, coefficient[:, None, None] * torch.bmm(tij, tji))
    return direct[:, None, None] * x + torch.bmm(cycles, x)


class SelectiveContentPropagation(RelationContentPropagation):
    def __init__(self, hidden, modalities, samples, graphs, learned=True):
        super().__init__(hidden, modalities, samples, graphs, 'content')
        self.reference_mask = None
        self.last_gate = self.last_gate_logits = None
        if learned:
            self.selector_id = nn.Embedding(modalities, 8)
            self.selector = nn.Sequential(nn.Linear(hidden * 3 + 8, 32),
                                          nn.GELU(), nn.Linear(32, 1))
            nn.init.normal_(self.selector[-1].weight, std=.002)
            nn.init.zeros_(self.selector[-1].bias)
        else:
            self.selector_id = self.selector = None

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('Fixed transductive cohort required')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = self.edge
        weights = torch.sigmoid((q[i]*q[j]).sum(-1)
            + self.similarity * self.edge_similarity.flatten()[self.edge_group]
            + self.edge_bias.flatten()[self.edge_group])
        maps = conformal_maps(self.map_values(z).reshape(-1, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        values = values.reshape(-1, 2, h//2)
        full = cooperative_operator(values, self.edge, weights, *maps)
        own = composed_self_part(values, self.edge, weights, *maps, constant_maps=True)
        delta = full.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        own = own.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        neighbor = delta - own
        if self.selector is not None:
            modal_ids = self.selector_id(torch.arange(m, device=z.device))[None].expand(count, -1, -1)
            # Both gate variants use identical stop-gradient inputs. The extra
            # gain loss trains the selector, not the encoder to mimic a teacher.
            features = torch.cat((z.detach(), z.detach().mean(1, keepdim=True).expand(-1, m, -1),
                                   F.layer_norm(neighbor.detach(), (h,)), modal_ids), -1)
            logits = self.selector(features).squeeze(-1)
            # Stretched sigmoid permits exactly zero/one without a stochastic
            # gate. The two variants have identical initialization and masks.
            gates = (1.2 * logits.sigmoid() - .1).clamp(0, 1)
            self.last_gate_logits = logits
        else:
            gates = tokens.new_ones(count, m)
            self.last_gate_logits = None
        if self.reference_mask is not None:
            gates = torch.as_tensor(self.reference_mask, device=z.device, dtype=z.dtype).expand(count, m)
            if not torch.isfinite(gates).all() or bool(((gates < 0) | (gates > 1)).any()):
                raise ValueError('Reference gates must be finite and within [0,1]')
        self.last_gate = gates.detach()
        self.last_neighbor_norm = neighbor.detach().norm(dim=-1)
        # At gate=1 this is numerically the original delta, not own+neighbor.
        mixed = delta + (gates[..., None] - 1) * neighbor
        return self.output_norm(tokens - self.step_logit.sigmoid() * self.output(F.gelu(mixed)))


class ReferenceSelectionModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, training_seed=0, learned=True):
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        original = self.propagation
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4201 + training_seed)
            replacement = SelectiveContentPropagation(cfg.Hidden_size, len(dd['Modal_Name']),
                dd['Sample_Num'], graphs, learned=learned)
            incompat = replacement.load_state_dict(original.state_dict(), strict=False)
            if incompat.unexpected_keys or any(not k.startswith('selector') for k in incompat.missing_keys):
                raise RuntimeError('C initialization was not preserved')
        self.propagation = replacement

    def forward(self, x_raw, reference_mask=None):
        self.propagation.reference_mask = reference_mask
        return super().forward(x_raw)
