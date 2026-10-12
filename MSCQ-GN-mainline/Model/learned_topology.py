"""Fixed-budget topology pilot; C operator/readout are retained.

This is a DGG-inspired straight-through selector, not a DGG reproduction.
Forward uses exactly k outgoing choices, symmetrized by union. Backward uses
an implicit-gradient, fixed-mass logistic relaxation, including absent edges.
Both arms share the same six additional relation-blend parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import RelationContentPropagation, RelationSheafModel, conformal_maps


class _SoftBudget(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, valid, k, temperature):
        if not 0 < k < score.shape[-1]-1 or temperature <= 0:
            raise ValueError('Require 0 < k < N-1 and positive temperature')
        lo=score.masked_fill(~valid,torch.inf).amin(-1,keepdim=True)-30*temperature
        hi=score.masked_fill(~valid,-torch.inf).amax(-1,keepdim=True)+30*temperature
        for _ in range(48):
            threshold=(lo+hi)/2
            p=torch.sigmoid((score-threshold)/temperature)*valid
            too_many=p.sum(-1,keepdim=True)>k
            lo=torch.where(too_many,threshold,lo)
            hi=torch.where(too_many,hi,threshold)
        p=torch.sigmoid((score-(lo+hi)/2)/temperature)*valid
        ctx.save_for_backward(p)
        ctx.temperature=temperature
        return p

    @staticmethod
    def backward(ctx, grad):
        p,=ctx.saved_tensors
        slope=p*(1-p)/ctx.temperature
        # Implicit derivative of the row threshold under sum_j p_ij = k.
        center=(grad*slope).sum(-1,keepdim=True)/slope.sum(-1,keepdim=True).clamp_min(1e-20)
        return slope*(grad-center),None,None,None


def select_neighbors(score,k,temperature=.1,relaxed=True):
    n=score.shape[-1]
    valid=~torch.eye(n,device=score.device,dtype=torch.bool)
    with torch.no_grad():
        neighbors=score.masked_fill(~valid,-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:k]
        directed=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,neighbors,True)
        hard=directed|directed.transpose(-1,-2)
    if relaxed and torch.is_grad_enabled():
        soft=_SoftBudget.apply(score,valid,k,temperature)
        union=soft+soft.transpose(-1,-2)-soft*soft.transpose(-1,-2)
        mask=hard.to(score.dtype)+(union-union.detach())
    else:
        mask=hard.to(score.dtype)
    return mask,hard,neighbors


def dense_cooperative_operator(x,adjacency,send,receive,send_scale,receive_scale):
    """Same two-pass D+I-normalized operator, batched by relation/content."""
    degree=adjacency.sum(-1)
    def apply(values,scale):
        diagonal=degree*scale.square()
        inv=(diagonal+1).rsqrt()[...,None,None]
        normalized=values*inv
        messages=torch.matmul(send,normalized)
        aggregated=torch.bmm(adjacency,messages.flatten(-2)).reshape_as(values)
        received=torch.matmul(receive.transpose(-1,-2),aggregated)
        return (diagonal[...,None,None]*normalized-received)*inv
    return apply(apply(x,send_scale),receive_scale)


class LearnedTopologyPropagation(RelationContentPropagation):
    def __init__(self,hidden,modalities,samples,graphs,raw_x,modal_index,topology_mode,k=10,temperature=.1):
        super().__init__(hidden,modalities,samples,graphs,'content')
        if topology_mode not in ('fixed','learned'):
            raise ValueError(topology_mode)
        self.topology_mode,self.k,self.temperature=topology_mode,k,temperature
        self.relation_blend=nn.Parameter(torch.zeros(modalities))
        raw=[];supports=[]
        for indices,graph in zip(modal_index,graphs):
            features=F.normalize(raw_x.detach().cpu()[:,indices],dim=-1)
            raw.append(features@features.T)
            mask=torch.zeros(samples,samples,dtype=torch.bool)
            mask[tuple(graph['edge'])]=True
            supports.append(mask)
        self.register_buffer('raw_similarity',torch.stack(raw),persistent=False)
        self.register_buffer('fixed_support',torch.stack(supports),persistent=False)
        self.register_buffer('fixed_neighbors',torch.stack([g['neighbors'] for g in graphs]),persistent=False)
        self.last_support=None
        self.last_neighbors=None

    def forward(self,tokens):
        n,m,h=tokens.shape
        if (n,m,h)!=(self.samples,self.modalities,self.hidden):
            raise ValueError('Original full-cohort transductive order required')
        z=self.input_norm(tokens)
        q=F.normalize(self.edge_query(z),dim=-1).permute(1,0,2)
        affinity=q@q.transpose(-1,-2)
        # At blend=0 both arms have the exact original cosine support/weights.
        relation=self.raw_similarity+self.relation_blend.tanh()[:,None,None]*affinity
        if self.topology_mode=='learned':
            mask,support,neighbors=select_neighbors(relation,self.k,self.temperature,self.training)
        else:
            support,neighbors=self.fixed_support,self.fixed_neighbors
            mask=support.to(tokens.dtype)
        self.last_support,self.last_neighbors=support.detach(),neighbors.detach()
        weight=torch.sigmoid(affinity[None,:,:,:]+
            self.edge_similarity[:,:,None,None]*relation[:,None,:,:]+self.edge_bias[:,:,None,None])
        adjacency=(weight*mask[:,None,:,:]).reshape(m*m,n,n)
        send,receive,cs,ct=conformal_maps(self.map_values(z).reshape(m*m,n,4))
        values=self.value(z).permute(1,0,2)[None].expand(m,-1,-1,-1)
        diffused=dense_cooperative_operator(values.reshape(m*m,n,2,h//2),adjacency,send,receive,cs,ct)
        delta=diffused.reshape(m,m,n,h).mean(0).permute(1,0,2)
        return self.output_norm(tokens-self.step_logit.sigmoid()*self.output(F.gelu(delta)))

    @torch.no_grad()
    def topology_diagnostics(self,previous=None):
        support=self.last_support
        neighbors=self.last_neighbors
        if support is None:
            raise RuntimeError('Run forward before diagnosis')
        per_modality=[]
        for m in range(self.modalities):
            degree=support[m].sum(-1)
            reference=torch.zeros_like(support[m]).scatter_(-1,self.fixed_neighbors[m],True)
            chosen=torch.zeros_like(support[m]).scatter_(-1,neighbors[m],True)
            per_modality.append(dict(degree_min=int(degree.min()),degree_mean=float(degree.float().mean()),
                degree_max=int(degree.max()),outgoing_k=int(chosen.sum(-1).min()),
                new_choice_fraction=float((chosen&~reference).sum()/chosen.sum()),
                edges_outside_original_fraction=float((support[m]&~self.fixed_support[m]).sum()/support[m].sum()),
                changed_since_probe_fraction=(float((support[m]^previous[m]).sum()/(support[m]|previous[m]).sum())
                    if previous is not None else None)))
        return dict(relation_blend=self.relation_blend.tanh().cpu().tolist(),modalities=per_modality,
            symmetric=bool(torch.equal(support,support.transpose(-1,-2))),
            no_self_edges=not bool(support.diagonal(dim1=-2,dim2=-1).any()))


class LearnedTopologyModel(RelationSheafModel):
    def __init__(self,dd,cfg,graphs,raw_x,seed,topology_mode,k=10,temperature=.1):
        super().__init__(dd,cfg,graphs,'content',training_seed=seed)
        old_state=self.propagation.state_dict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2901+seed)
            self.propagation=LearnedTopologyPropagation(cfg.Hidden_size,self._modal_num,dd['Sample_Num'],
                graphs,raw_x,dd['Modal_Index'],topology_mode,k,temperature)
        incompatible=self.propagation.load_state_dict(old_state,strict=False)
        assert incompatible.missing_keys==['relation_blend'] and not incompatible.unexpected_keys
