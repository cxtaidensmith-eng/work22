"""Frozen C backbone, with concatenation or a near-matched additive head.

No graph/encoder/readout/loss changes. The wide head's extra units have a
private dropout stream, so enlarging the classifier does not shift backbone
noise/dropout draws. Training restarts from scratch for each fold.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import RelationSheafModel


class ExpandedDropout(nn.Module):
    def __init__(self, p, common_width, seed):
        super().__init__()
        self.p, self.common_width, self.seed = p, common_width, seed
        self.generators = {}

    def forward(self, x):
        if not self.training or not self.p:
            return x
        common = F.dropout(x[:, :self.common_width].contiguous(), self.p, True)
        extra = x[:, self.common_width:]
        device = str(x.device)
        if device not in self.generators:
            self.generators[device] = torch.Generator(device=x.device).manual_seed(self.seed)
        mask = torch.empty_like(extra).bernoulli_(1-self.p, generator=self.generators[device])
        return torch.cat((common, extra * mask / (1-self.p)), -1)


class BranchFusionModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, fusion, training_seed=0):
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        if fusion not in ('C', 'concat', 'wide'):
            raise ValueError(fusion)
        self.fusion = fusion
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        old = self.classifier
        hidden, width, classes = old[0].in_features, old[0].out_features, old[3].out_features
        if fusion != 'C':
            # Protect both CPU and CUDA states, unlike a CPU-only RNG fork.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(7101 + training_seed)
                target_width = width if fusion == 'concat' else round(
                    (2*hidden+1+classes)*width/(hidden+1+classes))
                self.classifier = nn.Sequential(
                    nn.Linear(2*hidden if fusion == 'concat' else hidden, target_width),
                    nn.ReLU(),
                    nn.Dropout(cfg.Drop_rate) if fusion == 'concat' else
                    ExpandedDropout(cfg.Drop_rate, width, 8101+training_seed),
                    nn.Linear(target_width, classes))
            with torch.no_grad():
                if fusion == 'concat':
                    self.classifier[0].weight.copy_(torch.cat((old[0].weight, old[0].weight), -1))
                    self.classifier[0].bias.copy_(old[0].bias)
                    self.classifier[3].load_state_dict(old[3].state_dict())
                else:
                    self.classifier[0].weight[:width].copy_(old[0].weight)
                    self.classifier[0].bias[:width].copy_(old[0].bias)
                    self.classifier[3].weight[:, :width].copy_(old[3].weight)
                    # Extra units become active through their output gradients;
                    # random incoming weights prevent permanently dead units.
                    self.classifier[3].weight[:, width:].zero_()
                    self.classifier[3].bias.copy_(old[3].bias)

    def classify(self, semantic, own):
        value = torch.cat((semantic, own), -1) if self.fusion == 'concat' else semantic + own
        return self.classifier(value)

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
        own = self.Global_Message(x * gate[self.feature_to_modal].view(1, -1))
        if self.capture_diagnostics:
            with torch.no_grad():
                sn, on = semantic.norm(dim=-1), own.norm(dim=-1)
                self.last_diagnostics = dict(
                    semantic_norm_mean=float(sn.mean()), own_norm_mean=float(on.mean()),
                    own_over_semantic_norm_mean=float((on/sn.clamp_min(1e-12)).mean()),
                    branch_cosine_mean=float(F.cosine_similarity(semantic, own).mean()))
                if self.fusion == 'concat':
                    a, b = self.classifier[0].weight.chunk(2, -1)
                    self.last_diagnostics['relative_head_split_difference'] = float(
                        (a-b).norm()/(.5*(a.norm()+b.norm())).clamp_min(1e-12))
        if branch == 'semantic_only':
            own = torch.zeros_like(own)
        elif branch == 'own_only':
            semantic = torch.zeros_like(semantic)
        return self.classify(semantic, own), embeddings, auxiliary
