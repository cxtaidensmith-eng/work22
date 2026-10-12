"""D3: move C propagation after its existing patient fusion/readout.

The same fused patient vector is broadcast into all six filter channels.
Channels no longer carry different input modality states. The existing six
content-conditioned maps are retained as filter-bank parameters for capacity
matching, not advertised as modality-specific content in this control.
All 36 relation/filter groups are averaged into one patient update/output.
"""
import torch
from .relation_sheaf import RelationSheafModel


class FuseFirstModel(RelationSheafModel):
    def __init__(self, dd, cfg, graphs, training_seed=0):
        super().__init__(dd,cfg,graphs,'content',training_seed=training_seed)

    def forward(self,x_raw):
        x=self.Feature_Modal(x_raw)
        if self.training:
            std=self._modal_noise_std[self.feature_to_modal]*self.noise_scale
            x=x+torch.randn_like(x)*std.view(1,-1)
        gate=self.modal_gate_logit.sigmoid()
        tokens=self.modal_token_encoder(x)*gate.view(1,-1,1)
        for block in self.shared_transformer:
            tokens=block(tokens)
        embeddings=[pool(tokens) for pool in self.label_pools]
        auxiliary=[head(z) for head,z in zip(self._Auxi_classifier,embeddings)]
        semantic=self.Message_MLP(torch.cat(embeddings,-1))
        # No original modal token bypasses this common patient representation.
        fused_channels=semantic[:,None,:].expand(-1,self._modal_num,-1)
        semantic=self.propagation(fused_channels).mean(1)
        own=self.Global_Message(x*gate[self.feature_to_modal].view(1,-1))
        return self.classifier(semantic+own),embeddings,auxiliary
