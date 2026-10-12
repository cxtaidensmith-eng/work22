"""Change only C's reduction over relation sources; retain the cooperative operator."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import (RelationSheafModel, RelationContentPropagation,
                             conformal_maps, cooperative_operator)


class RelationAggregationPropagation(RelationContentPropagation):
    def __init__(self, base, aggregation, training_seed):
        # Reuse the already initialized C modules without another random draw.
        nn.Module.__init__(self)
        if aggregation not in ('R1', 'R2'):
            raise ValueError(aggregation)
        for name in ('hidden', 'modalities', 'samples', 'mode'):
            setattr(self, name, getattr(base, name))
        for name, value in base.named_parameters(recurse=False):
            self.register_parameter(name, value)
        for name, value in base.named_buffers(recurse=False):
            self.register_buffer(name, value, persistent=name not in base._non_persistent_buffers_set)
        for name, value in base.named_children():
            self.add_module(name, value)
        self.aggregation = aggregation
        self.intervention = 'normal'
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        self.last_weights = None
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(9101 + training_seed)
            self.relation_score = nn.Sequential(
                nn.Linear(2*self.hidden, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.relation_score[-1].weight)
            self.relation_bias = nn.Parameter(torch.zeros(self.modalities, self.modalities))

    def aggregate(self, z, updates):
        """z: [N,content,H]; updates: [relation,content,N,H]."""
        d = updates.permute(2, 0, 1, 3)
        state = F.layer_norm(z, (self.hidden,))[:, None].expand_as(d)
        message = F.layer_norm(d, (self.hidden,))
        score = self.relation_score(torch.cat((state, message), -1)).squeeze(-1)
        score = score + self.relation_bias[None]
        if self.aggregation == 'R1':
            score = score.mean(0, keepdim=True).expand_as(score)
        alpha = score.softmax(dim=1)
        if self.intervention == 'patient_mean':
            # Fixed-checkpoint intervention: average POST-softmax weights.
            alpha = alpha.mean(0, keepdim=True).expand_as(alpha)
        elif self.intervention == 'uniform':
            alpha = torch.full_like(alpha, 1/self.modalities)
        elif self.intervention != 'normal':
            raise ValueError(self.intervention)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_weights = alpha.detach().clone()
                self.last_diagnostics = dict(
                    mean_abs_weight_deviation=float((alpha-1/self.modalities).abs().mean()),
                    patient_weight_std=float(alpha.std(0, unbiased=False).mean()),
                    normalized_weight_entropy=float((-(alpha*alpha.clamp_min(1e-12).log()).sum(1)/math.log(self.modalities)).mean()),
                    min_weight=float(alpha.min()), max_weight=float(alpha.max()),
                    score_output_norm=float(self.relation_score[-1].weight.norm()))
        return (alpha[..., None]*d).sum(1)

    def forward(self, tokens):
        count, m, h = tokens.shape
        if (count, m, h) != (self.samples, self.modalities, self.hidden):
            raise ValueError('Full saved cohort and row order required')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(m, -1, -1, -1).reshape(m*m*count, -1)
        i, j = self.edge
        affinity = (q[i]*q[j]).sum(-1)
        weight = torch.sigmoid(affinity + self.similarity*self.edge_similarity.flatten()[self.edge_group]
                               + self.edge_bias.flatten()[self.edge_group])
        send, receive, cs, ct = conformal_maps(self.map_values(z).reshape(-1, 4))
        values = self.value(z).permute(1, 0, 2)[None].expand(m, -1, -1, -1)
        diffused = cooperative_operator(values.reshape(-1, 2, h//2), self.edge, weight,
                                        send, receive, cs, ct)
        delta = self.aggregate(z, diffused.reshape(m, m, count, h))
        return self.output_norm(tokens-self.step_logit.sigmoid()*self.output(F.gelu(delta)))


class RelationAggregationModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, aggregation, training_seed=0):
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        if aggregation not in ('C', 'R1', 'R2'):
            raise ValueError(aggregation)
        self.aggregation = aggregation
        if aggregation != 'C':
            self.propagation = RelationAggregationPropagation(self.propagation, aggregation, training_seed)
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, x_raw):
        self.propagation.capture_diagnostics = self.capture_diagnostics
        result = super().forward(x_raw)
        if self.capture_diagnostics and self.aggregation != 'C':
            self.last_diagnostics = dict(self.propagation.last_diagnostics)
        return result
