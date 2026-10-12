"""D1 matched relation/content and D2 scalar diffusion controls for frozen C."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .relation_sheaf import (RelationSheafModel, RelationContentPropagation,
                             cooperative_operator, conformal_maps, sparse_aggregate)


def scalar_diffusion(x, edge, weight):
    """L^2 x, L=(D+I)^(-1/2)(D-A)(D+I)^(-1/2).

    Exactly the existing cooperative operator with identity send/receive maps.
    Retains two graph passes, scalar affinity, normalization and residual sign.
    This is a scalar Laplacian control, not a reproduction of a vanilla GCN.
    """
    degree = x.new_zeros(len(x)).index_add(0, edge[0], weight)
    inv = (degree + 1).rsqrt()[:, None]
    def apply(v):
        z = v * inv
        return (degree[:, None] * z - sparse_aggregate(edge, weight, z)) * inv
    return apply(apply(x))


class MechanismPropagation(RelationContentPropagation):
    @classmethod
    def from_original(cls, original, graphs, control):
        if control not in ('C', 'D1', 'D2'):
            raise ValueError(control)
        layer = copy.deepcopy(original)
        layer.__class__ = cls
        layer.control = control
        if control == 'D1':
            # Compact six active groups; no unused off-diagonal parameters.
            layer.edge_similarity = nn.Parameter(original.edge_similarity.diagonal().clone())
            layer.edge_bias = nn.Parameter(original.edge_bias.diagonal().clone())
            layer.edge = torch.cat([g['edge'] + n*layer.samples for n,g in enumerate(graphs)], 1)
            layer.similarity = torch.cat([g['similarity'] for g in graphs])
            layer.edge_group = torch.cat([torch.full((g['edge'].shape[1],), n, dtype=torch.long)
                                          for n,g in enumerate(graphs)])
        elif control == 'D2':
            del layer.content_id, layer.conditioner
        return layer

    def forward(self, tokens):
        if self.control == 'C':
            return super().forward(tokens)
        count,m,h = tokens.shape
        if (count,m,h) != (self.samples,self.modalities,self.hidden):
            raise ValueError('Requires the original full cohort and row order')
        z = self.input_norm(tokens)
        q = F.normalize(self.edge_query(z),dim=-1).permute(1,0,2)
        values = self.value(z).permute(1,0,2)
        if self.control == 'D2':
            q = q[None].expand(m,-1,-1,-1)
            values = values[None].expand(m,-1,-1,-1)
        q = q.reshape(-1,q.shape[-1])
        i,j = self.edge
        weight = torch.sigmoid((q[i]*q[j]).sum(-1)
            + self.similarity*self.edge_similarity.flatten()[self.edge_group]
            + self.edge_bias.flatten()[self.edge_group])
        if self.control == 'D1':
            ids = torch.arange(m,device=z.device)
            params = self.conditioner(self.content_id(ids))[:,None,:].expand(-1,count,-1)
            delta = cooperative_operator(values.reshape(-1,2,h//2),self.edge,weight,
                        *conformal_maps(params.reshape(-1,4))).reshape(m,count,h).permute(1,0,2)
            # Mean over ONE active relation per content, not zero-padding / 6.
        else:
            delta = scalar_diffusion(values.reshape(-1,h),self.edge,weight)
            delta = delta.reshape(m,m,count,h).mean(0).permute(1,0,2)
        return self.output_norm(tokens-self.step_logit.sigmoid()*self.output(F.gelu(delta)))


class MechanismControlModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, control, training_seed=0):
        super().__init__(dd,cfg,graphs,'content',training_seed=training_seed)
        self.propagation = MechanismPropagation.from_original(self.propagation,graphs,control)
