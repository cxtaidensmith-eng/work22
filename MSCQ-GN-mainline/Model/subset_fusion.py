"""Pruned, patient-local fusion controls with a fixed historical head width.

Retain only selected encoders and head input columns. Initialization, modality
priors, full-shape noise/dropout draws and shared LayerNorm come from the same
historical full-fusion model. This is head-shape controlled, not equal-parameter.
"""
from copy import deepcopy

import torch
from torch import nn

from .modal_evidence import ModalEvidenceModel


class SubsetFusionModel(nn.Module):
    def __init__(self, dd, cfg, modalities, training_seed=0):
        super().__init__()
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError('Select at least one distinct modality')
        if any(name not in dd['Modal_Name'] for name in modalities):
            raise ValueError('Unknown modality')
        # Canonical source order makes concatenation independent of argument order.
        chosen = [i for i, name in enumerate(dd['Modal_Name']) if name in modalities]
        template = ModalEvidenceModel(dd, cfg, 'fusion', training_seed, rank=8)
        self.modalities = tuple(dd['Modal_Name'][i] for i in chosen)
        self.feature_num = int(dd['Feature_Num'])
        self.full_modal_num = len(dd['Modal_Name'])
        self.hidden = int(cfg.Hidden_size)
        self.fusion_width = template.fusion_width
        self.dropout = float(cfg.Drop_rate)
        self.noise_scale = template.noise_scale
        self.register_buffer('selected', torch.tensor(chosen, dtype=torch.long))
        self.register_buffer('noise_std', template._modal_noise_std[chosen].clone())
        self.gain = nn.Parameter(template.Feature_Modal.modal_gain.weight[chosen].detach().clone())
        self.bias = nn.Parameter(template.Feature_Modal.modal_bias.weight[chosen].detach().clone())
        self.encoders = nn.ModuleList(deepcopy(template.modal_token_encoder.modal_proj[i]) for i in chosen)
        self.modality_id = nn.Parameter(template.modal_token_encoder.modal_id_embedding.weight[chosen].detach().clone())
        self.norm = deepcopy(template.modal_token_encoder.input_norm)
        self.gate_logit = nn.Parameter(template.modal_gate_logit[chosen].detach().clone())
        for local, original in enumerate(chosen):
            self.register_buffer(f'indices_{local}', torch.tensor(dd['Modal_Index'][original], dtype=torch.long))
        self.head = deepcopy(template.fusion_head)
        columns = torch.cat([torch.arange(i * self.hidden, (i + 1) * self.hidden) for i in chosen])
        self.head[0].weight = nn.Parameter(self.head[0].weight.detach().index_select(1, columns).clone())
        self.head[0].in_features = len(chosen) * self.hidden

    def forward(self, x_raw):
        if x_raw.ndim != 2 or x_raw.shape[1] != self.feature_num:
            raise ValueError('Expected unchanged full-width feature tensor')
        noise = torch.randn_like(x_raw) if self.training else None
        tokens = []
        for local, encoder in enumerate(self.encoders):
            indices = getattr(self, f'indices_{local}')
            x = x_raw.index_select(1, indices) * self.gain[local] + self.bias[local]
            if noise is not None:
                x = x + noise.index_select(1, indices) * (self.noise_std[local] * self.noise_scale)
            tokens.append(encoder(x))
        tokens = self.norm(torch.stack(tokens, dim=1) + self.modality_id)
        tokens = tokens * self.gate_logit.sigmoid().view(1, -1, 1)
        padded = tokens.new_zeros((len(x_raw), self.full_modal_num, self.hidden))
        padded[:, self.selected] = tokens
        tokens = nn.functional.dropout(padded, p=self.dropout, training=self.training).index_select(1, self.selected)
        return dict(logits=self.head(tokens.flatten(start_dim=1)), tokens=tokens,
                    unimodal_logits=None, pair_logits=None)

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        return dict(total=count(self), encoders=count(self.encoders), head=count(self.head),
                    normalization=count(self.norm), modality_embedding=self.modality_id.numel(),
                    feature_embedding=self.gain.numel() + self.bias.numel(), gate=self.gate_logit.numel(),
                    modalities=list(self.modalities), fusion_width=self.fusion_width,
                    input_features=sum(getattr(self, f'indices_{i}').numel() for i in range(len(self.modalities))),
                    head_input=len(self.modalities) * self.hidden)
