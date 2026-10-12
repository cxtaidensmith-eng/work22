"""Retained BB model: own prediction plus a fixed 10-neighbor modal reference.

Same computation/initialization as the historical BB implementation, without
depending on the exploratory Stage A/Stage B/AA/AB runners.
"""
import math
import torch
from torch import nn
from .network import MSCQGN


@torch.no_grad()
def fixed_candidates(x, k=10):
    z = torch.nn.functional.normalize(x, dim=-1)
    scores = z @ z.T
    scores.fill_diagonal_(-torch.inf)
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]


class FixedNeighborReference(MSCQGN):
    def __init__(self, dd, cfg, candidates):
        super().__init__(dd, cfg.Hidden_size, cfg.Drop_rate, cfg.ChebGCN_K,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            input_noise_std=cfg.input_noise_std, drop_path=cfg.drop_path,
            global_word_emb=cfg.global_word_emb, semantic_branch=cfg.semantic_branch,
            adj_mode=cfg.adj_mode, modal_structure=cfg.modal_structure, readout_mode=cfg.readout_mode)
        assert self.semantic_branch == 'both' and self.readout_mode == 'class_query'
        assert self.modal_structure == 'modal' and candidates.shape == (dd['Sample_Num'], 10)
        # Construct/delete the old graph modules to preserve the encoder RNG stream.
        del self.Adj_Learning, self.GCN
        h, c = cfg.Hidden_size, dd['Class_Num']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1901)
            self.self_head = nn.Sequential(nn.Linear(h, h//2), nn.Dropout(cfg.Drop_rate), nn.ReLU(), nn.Linear(h//2, c))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1902)
            self.reference_norm = nn.LayerNorm(h)
            self.query = nn.Linear(h, 32, bias=False)
            self.key = nn.Linear(h, 32, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1903)
            self.message = nn.Sequential(nn.Linear(2*h, h//2), nn.GELU(), nn.Linear(h//2, c))
        self.register_buffer('candidates', candidates.detach().clone())

    def load_historical_state(self, state):
        # Old BB inherited an unused diagnosis-pair buffer; no learned tensor is dropped.
        state = dict(state)
        if 'pair_basis' in state:
            expected = torch.tensor([[1.,-1.,0.], [1.,0.,-1.], [0.,1.,-1.]], device=state['pair_basis'].device)
            assert torch.equal(state.pop('pair_basis'), expected)
        return self.load_state_dict(state, strict=True)

    def forward(self, x_raw):
        x = self.Feature_Modal(x_raw)
        if self.training:
            noise_std = self._modal_noise_std[self.feature_to_modal]
            x = x + torch.randn_like(x) * (noise_std * self.noise_scale).view(1, -1)
        gate = torch.sigmoid(self.modal_gate_logit)
        tokens = self.modal_token_encoder(x) * gate.view(1, -1, 1)
        h = tokens
        for block in self.shared_transformer:
            h = block(h)
        embeddings = [pool(h) for pool in self.label_pools]
        auxiliary = [head(z) for head, z in zip(self._Auxi_classifier, embeddings)]
        class_embedding = self.Message_MLP(torch.cat(embeddings, dim=-1))
        global_embedding = self.Global_Message(x * gate[self.feature_to_modal].view(1, -1))
        own = self.self_head(class_embedding + global_embedding)
        z = self.reference_norm(tokens)
        products = self.query(z)[:, None] * self.key(z)[self.candidates]
        groups = torch.tensor_split(products, 3, dim=-1)
        scores = torch.stack([g.sum(-1)/math.sqrt(g.shape[-1]) for g in groups], dim=-1)
        weights = scores.flatten(1, 2).softmax(1).reshape(scores.shape)
        # Keep the historical expanded message operation to reproduce floating point outputs.
        source = z[self.candidates]
        messages = self.message(torch.cat((source, source), dim=-1))
        messages = messages - messages.mean(-1, keepdim=True)
        contributions = weights.mean(-1)[..., None] * messages
        residual = contributions.sum(2).sum(1)
        return dict(logits=own+residual, self_logits=own, residual=residual,
            label_embeddings=embeddings, auxiliary=auxiliary, modal_embedding=tokens,
            channel_weights=weights, message_values=messages, channel_contributions=contributions)


def reference_loss(criterion, result, labels, train_mask):
    loss = criterion(result['logits'], labels, train_mask, result['label_embeddings'], result['auxiliary'])
    final_ce = criterion.CE_loss(result['logits'][train_mask], labels[train_mask])
    own_ce = criterion.CE_loss(result['self_logits'][train_mask], labels[train_mask])
    return loss + .5*(own_ce-final_ce)
