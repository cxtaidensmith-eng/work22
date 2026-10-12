"""Own-branch gate removal and a shared token-wise residual FFN.

The original C implementation remains unchanged. Gate removal affects only
the input of Global_Message; the population path still uses the original gate.
"""
import torch
from torch import nn

from .relation_sheaf import RelationSheafModel


MODES = ('C', 'NO_GATE', 'FFN', 'NO_GATE_FFN')


class OwnTokenEncoder(nn.Module):
    def __init__(self, original, use_ffn):
        super().__init__()
        for name in ('Modal_Index', 'word_emb', 'seq_len', 'Hidden_size', 'num_heads'):
            setattr(self, name, getattr(original, name))
        for name in ('Embedding_dict', 'Fusion_modal_0', 'layer_norm_0',
                     'Fusion_modal_1', 'layer_norm_1', 'sequeeze_layer'):
            setattr(self, name, getattr(original, name))
        self.use_ffn = use_ffn
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        if use_ffn:
            d = self.word_emb
            self.token_ffn = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
            nn.init.zeros_(self.token_ffn[2].weight)
            nn.init.zeros_(self.token_ffn[2].bias)

    def forward(self, x):
        tokens = torch.stack([self.Embedding_dict[f'Modal_{i}'](x[:, idx])
                              for i, idx in enumerate(self.Modal_Index)], dim=1)
        update, _ = self.Fusion_modal_0(tokens, tokens, tokens)
        tokens = self.layer_norm_0(tokens + update)
        update, _ = self.Fusion_modal_1(tokens, tokens, tokens)
        tokens = self.layer_norm_1(tokens + update)
        if self.use_ffn:
            delta = self.token_ffn(tokens)
            if self.capture_diagnostics:
                with torch.no_grad():
                    self.last_diagnostics = dict(
                        ffn_relative_update=float(delta.norm()/tokens.norm().clamp_min(1e-12)),
                        per_modal_relative_update=(delta.norm(dim=(0, 2))/
                            tokens.norm(dim=(0, 2)).clamp_min(1e-12)).cpu().tolist(),
                        ffn_input_weight_norm=float(self.token_ffn[0].weight.norm()),
                        ffn_output_weight_norm=float(self.token_ffn[2].weight.norm()))
            tokens = tokens + delta
        else:
            self.last_diagnostics = {}
        return self.sequeeze_layer(tokens.reshape(-1, self.word_emb*self.seq_len))


class OwnBranchOptimizationModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, mode, training_seed=0):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.own_mode = mode
        self.use_own_gate = mode in ('C', 'FFN')
        self.use_own_ffn = mode in ('FFN', 'NO_GATE_FFN')
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(10401 + training_seed)
            self.Global_Message = OwnTokenEncoder(self.Global_Message, self.use_own_ffn)
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, x_raw, branch='both'):
        if branch not in ('both', 'semantic_only', 'own_only'):
            raise ValueError(branch)
        x = self.Feature_Modal(x_raw)
        if self.training:
            std = self._modal_noise_std[self.feature_to_modal] * self.noise_scale
            x = x + torch.randn_like(x) * std.view(1, -1)
        gate = self.modal_gate_logit.sigmoid()
        tokens = self.modal_token_encoder(x) * gate.view(1, -1, 1)
        tokens = self.propagation(tokens)
        for block in self.shared_transformer:
            tokens = block(tokens)
        embeddings = [pool(tokens) for pool in self.label_pools]
        auxiliary = [head(z) for head, z in zip(self._Auxi_classifier, embeddings)]
        semantic = self.Message_MLP(torch.cat(embeddings, -1))
        own_input = x * gate[self.feature_to_modal].view(1, -1) if self.use_own_gate else x
        self.Global_Message.capture_diagnostics = self.capture_diagnostics
        own = self.Global_Message(own_input)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = dict(
                    shared_gate=gate.detach().cpu().tolist(),
                    semantic_norm_mean=float(semantic.norm(dim=-1).mean()),
                    own_norm_mean=float(own.norm(dim=-1).mean()),
                    **self.Global_Message.last_diagnostics)
        if branch == 'semantic_only':
            own = torch.zeros_like(own)
        elif branch == 'own_only':
            semantic = torch.zeros_like(semantic)
        return self.classifier(semantic + own), embeddings, auxiliary
