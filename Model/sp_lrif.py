"""Sum-preserving low-rank interaction fusion for the registered ABIDE run."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cme_dual_branch import CMEDualBranchModel


class SumPreservingLowRankInteraction(nn.Module):
    """The one preregistered SP-LRIF formula; no gate, alpha, or extra dropout."""

    def __init__(self, dimension: int, rank: int = 4) -> None:
        super().__init__()
        if dimension <= 0 or rank != 4:
            raise ValueError("SP-LRIF requires a positive dimension and fixed rank=4")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.proj_c = nn.Linear(dimension, rank, bias=False)
        self.proj_g = nn.Linear(dimension, rank, bias=False)
        self.proj_diff = nn.Linear(dimension, rank, bias=False)
        self.proj_out = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.proj_out.weight)
        self.enabled = True
        self.last_agreement: torch.Tensor | None = None
        self.last_disagreement: torch.Tensor | None = None
        self.last_delta: torch.Tensor | None = None
        self.last_sum: torch.Tensor | None = None

    def forward(self, category: torch.Tensor, global_message: torch.Tensor) -> torch.Tensor:
        if category.shape != global_message.shape or category.ndim != 2:
            raise RuntimeError("SP-LRIF expects matching [N,d] Category/Global tensors")
        if category.shape[-1] != self.dimension:
            raise RuntimeError("SP-LRIF feature dimension changed")
        direct_sum = category + global_message
        c = F.layer_norm(category, (category.shape[-1],))
        g = F.layer_norm(global_message, (global_message.shape[-1],))
        agreement = self.proj_c(c) * self.proj_g(g)
        disagreement = self.proj_diff(c - g)
        delta = self.proj_out(F.gelu(agreement + disagreement))
        if not self.enabled:
            delta = torch.zeros_like(delta)
        self.last_agreement = agreement
        self.last_disagreement = disagreement
        self.last_delta = delta
        self.last_sum = direct_sum
        return direct_sum + delta


class SPLRIFDualBranchModel(CMEDualBranchModel):
    """D3 backbone with an attach-after-construction SP-LRIF module."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sp_lrif: SumPreservingLowRankInteraction | None = None

    def attach_sp_lrif(self, rank: int = 4) -> None:
        if self.sp_lrif is not None:
            raise RuntimeError("SP-LRIF is already attached")
        output_linear = next(
            (module for module in reversed(list(self.Message_MLP.modules())) if isinstance(module, nn.Linear)),
            None,
        )
        if output_linear is None:
            raise RuntimeError("Unable to infer final Category/Global dimension")
        reference = next(self.parameters())
        self.sp_lrif = SumPreservingLowRankInteraction(output_linear.out_features, rank).to(
            device=reference.device, dtype=reference.dtype
        )

    def _fuse_semantic_branches(
        self, category_embedding: torch.Tensor, global_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.sp_lrif is None:
            return super()._fuse_semantic_branches(category_embedding, global_embedding)
        return self.sp_lrif(category_embedding, global_embedding)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if kwargs.get("return_intermediates", False) and self.sp_lrif is not None:
            raw_logits, label_embeddings, auxiliary, intermediates = output
            intermediates = dict(intermediates)
            intermediates.update(
                {
                    "sp_lrif_agreement": self.sp_lrif.last_agreement,
                    "sp_lrif_disagreement": self.sp_lrif.last_disagreement,
                    "sp_lrif_delta": self.sp_lrif.last_delta,
                    "sp_lrif_sum": self.sp_lrif.last_sum,
                }
            )
            return raw_logits, label_embeddings, auxiliary, intermediates
        return output
