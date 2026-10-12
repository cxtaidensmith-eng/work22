"""Small channel-group extension of C's content-conditioned transport.

Grouping partitions the second axis of the original [2,H/2] stalk layout.
The graph, scalar edge weights, value/output projections and final readout
remain unchanged. Groups are latent channel groups, not anatomical regions.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import (RelationSheafModel, RelationContentPropagation,
                             conformal_maps, cooperative_operator)


class GroupedContentPropagation(RelationContentPropagation):
    def __init__(self, base, groups=1, conditioner_width=32, training_seed=0):
        nn.Module.__init__(self)
        if base.mode != 'content' or groups < 1 or base.hidden % (2 * groups):
            raise ValueError('Content mode and a divisor of H/2 are required')
        if conditioner_width < 32 or (groups != 1 and conditioner_width != 32):
            raise ValueError('Only grouped width32 or widened single-group controls are supported')
        for name in ('hidden', 'modalities', 'samples', 'mode'):
            setattr(self, name, getattr(base, name))
        for name, value in base.named_parameters(recurse=False):
            self.register_parameter(name, value)
        for name, value in base.named_buffers(recurse=False):
            self.register_buffer(name, value, persistent=name not in base._non_persistent_buffers_set)
        for name, value in base.named_children():
            self.add_module(name, value)
        self.groups = groups
        self.conditioner_width = conditioner_width
        self.intervention = 'normal'
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        old = self.conditioner
        if groups != 1 or conditioner_width != 32:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(29101 + training_seed)
                self.conditioner = nn.Sequential(
                    nn.Linear(16, conditioner_width), nn.GELU(),
                    nn.Linear(conditioner_width, 4 * groups)).to(
                        device=old[0].weight.device, dtype=old[0].weight.dtype)
            with torch.no_grad():
                self.conditioner[0].weight[:32].copy_(old[0].weight)
                self.conditioner[0].bias[:32].copy_(old[0].bias)
                self.conditioner[2].weight.zero_()
                for group in range(groups):
                    self.conditioner[2].weight[4*group:4*(group+1), :32].copy_(old[2].weight)
                    self.conditioner[2].bias[4*group:4*(group+1)].copy_(old[2].bias)

    def _apply_intervention(self, parameters):
        if self.intervention == 'collapse':
            return parameters.mean(-2, keepdim=True).expand_as(parameters)
        if self.intervention != 'normal':
            raise ValueError(self.intervention)
        return parameters

    def group_map_parameters(self):
        ids = torch.arange(self.modalities, device=self.content_id.weight.device)
        raw = self.conditioner(self.content_id(ids)).reshape(self.modalities, self.groups, 4)
        return self._apply_intervention(raw)

    def map_values(self, z):
        m, count = self.modalities, self.samples
        ids = torch.arange(m, device=z.device)
        content = self.content_id(ids)[None, :, None, :].expand(m, m, count, -1)
        raw = self.conditioner(content).reshape(m, m, count, self.groups, 4)
        return self._apply_intervention(raw)

    def grouped_diffusion(self, values, weight, parameters):
        """Preserve C's channel pairing while giving each group its own maps.

        values: [R*M*N,2,H/2]; parameters: [R*M*N,G,4].
        Concatenation restores the original second-axis ordering exactly.
        """
        if values.shape[-2:] != (2, self.hidden//2):
            raise ValueError('Expected the original [2,H/2] channel pairing')
        if parameters.shape != (len(values), self.groups, 4):
            raise ValueError('Expected one four-parameter map per node and group')
        chunks = values.chunk(self.groups, dim=-1)
        outputs = []
        for group, chunk in enumerate(chunks):
            send, receive, cs, ct = conformal_maps(parameters[:, group])
            outputs.append(cooperative_operator(chunk, self.edge, weight, send, receive, cs, ct))
        return torch.cat(outputs, dim=-1) if self.groups > 1 else outputs[0]

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('This transductive layer requires the saved full cohort and row order')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = self.edge
        affinity = (q[i] * q[j]).sum(-1)
        weight = torch.sigmoid(affinity + self.similarity * self.edge_similarity.flatten()[self.edge_group]
                               + self.edge_bias.flatten()[self.edge_group])
        parameters = self.map_values(z).reshape(-1, self.groups, 4)
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = self.grouped_diffusion(values.reshape(-1, 2, h//2), weight, parameters)
        delta = diffused.reshape(m, m, count, h).mean(0).permute(1, 0, 2)
        update = self.step_logit.sigmoid() * self.output(F.gelu(delta))
        if self.capture_diagnostics:
            with torch.no_grad():
                maps = parameters.reshape(m, m, count, self.groups, 4)[0, :, 0]
                send, receive, cs, ct = conformal_maps(maps)
                self.last_diagnostics = dict(
                    groups=self.groups,
                    map_parameters=maps.detach().cpu().tolist(),
                    mean_group_parameter_std=float(maps.std(1, unbiased=False).mean()),
                    send_map_group_std=float(send.std(1, unbiased=False).mean()),
                    receive_map_group_std=float(receive.std(1, unbiased=False).mean()),
                    send_scales=cs.detach().cpu().tolist(),
                    receive_scales=ct.detach().cpu().tolist(),
                    update_ratio=float((update.norm(dim=-1)/tokens.norm(dim=-1).clamp_min(1e-12)).mean()),
                    step=float(self.step_logit.sigmoid()))
        return self.output_norm(tokens - update)


class GroupedContentModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, variant, training_seed=0):
        if variant not in ('C', 'G1', 'G2', 'W38'):
            raise ValueError(variant)
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.variant = variant
        if variant != 'C':
            self.propagation = GroupedContentPropagation(
                self.propagation, groups=2 if variant == 'G2' else 1,
                conditioner_width=38 if variant == 'W38' else 32,
                training_seed=training_seed)
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, x_raw):
        self.propagation.capture_diagnostics = self.capture_diagnostics
        output = super().forward(x_raw)
        if self.capture_diagnostics and self.variant != 'C':
            self.last_diagnostics = dict(self.propagation.last_diagnostics)
        return output
