"""Independent unimodal controls with the historical encoder initialization.

Only the selected modality has parameters or a path to the output. Full-shape
random draws preserve the historical input-noise and token-dropout streams;
values of unselected input features are never used.
"""
from copy import deepcopy

import torch
from torch import nn

from .modal_evidence import ModalEvidenceModel


class SingleModalEvidenceModel(nn.Module):
    def __init__(self, dd, cfg, modality, training_seed=0):
        super().__init__()
        index = dd['Modal_Name'].index(modality)
        template = ModalEvidenceModel(dd, cfg, 'additive', training_seed, rank=8)
        self.modality = modality
        self.index = index
        self.feature_num = dd['Feature_Num']
        self.modal_num = len(dd['Modal_Name'])
        self.hidden = int(cfg.Hidden_size)
        self.noise_scale = template.noise_scale
        self.dropout = float(cfg.Drop_rate)
        self.register_buffer('indices', torch.tensor(dd['Modal_Index'][index], dtype=torch.long))
        self.register_buffer('noise_std', template._modal_noise_std[index].clone())
        self.gain = nn.Parameter(template.Feature_Modal.modal_gain.weight[index].detach().clone())
        self.bias = nn.Parameter(template.Feature_Modal.modal_bias.weight[index].detach().clone())
        self.encoder = deepcopy(template.modal_token_encoder.modal_proj[index])
        self.modality_id = nn.Parameter(template.modal_token_encoder.modal_id_embedding.weight[index].detach().clone())
        self.norm = deepcopy(template.modal_token_encoder.input_norm)
        self.gate_logit = nn.Parameter(template.modal_gate_logit[index].detach().clone())
        self.head = deepcopy(template.unimodal_heads[index])

    def forward(self, x_raw):
        if x_raw.ndim != 2 or x_raw.shape[1] != self.feature_num:
            raise ValueError('Expected the unchanged full-width feature tensor')
        x = x_raw.index_select(1, self.indices) * self.gain + self.bias
        if self.training:
            noise = torch.randn_like(x_raw).index_select(1, self.indices)
            x = x + noise * (self.noise_std * self.noise_scale)
        token = self.norm(self.encoder(x) + self.modality_id) * self.gate_logit.sigmoid()
        # The same full-shaped dropout call as the historical additive model.
        padded = token.new_zeros((len(x), self.modal_num, self.hidden))
        padded[:, self.index] = token
        token = nn.functional.dropout(padded, p=self.dropout, training=self.training)[:, self.index]
        logits = self.head(token)
        return dict(logits=logits, unimodal_logits=logits[:, None],
                    pair_logits=None, tokens=token[:, None])

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        return dict(total=count(self), encoder=count(self.encoder),
                    head=count(self.head), feature_embedding=2, gate=1,
                    modality=self.modality, features=int(self.indices.numel()))
