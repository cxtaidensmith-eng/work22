"""Training-only non-destructive comparison for PC-BBF v1.

The formal model, parameters, initialization, and ordinary ``forward`` path
remain exactly those of :class:`Model.pc_bbf.PCBBFModel`.  The additional
``forward_base_safe`` method evaluates the unchanged PC-BBF output and the
counterfactual ``H0 = Y + G`` output with identical post-fusion RNG state.
The counterfactual output is detached because it is only the stop-gradient
reference used by the PC-BBF-Safe v1.1 training loss.
"""

from __future__ import annotations

from typing import Any

import torch

from Model.pc_bbf import PCBBFModel


def _capture_rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class PCBBFSafeModel(PCBBFModel):
    """PC-BBF v1 with a training-only, shared-RNG base-logit probe."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._capture_backend_rng = False
        self._backend_rng_before: dict[str, Any] | None = None

    def _fuse_semantic_branches(
        self, category_embedding: torch.Tensor, global_embedding: torch.Tensor
    ) -> torch.Tensor:
        fused = super()._fuse_semantic_branches(
            category_embedding, global_embedding
        )
        if self._capture_backend_rng:
            # The next operation in the inherited forward is the DIFFormer
            # graph head.  Capturing here makes its dropout masks replayable.
            self._backend_rng_before = _capture_rng_state()
        return fused

    def forward_base_safe(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
    ):
        """Return safe and detached base logits with identical backend masks.

        The safe path is evaluated first and therefore has exactly the same
        values, gradients, and RNG consumption as ordinary PC-BBF inference.
        The base path then replays the captured backend RNG state under
        ``no_grad``.  Finally, the post-safe RNG state and graph-head caches are
        restored, so this method advances randomness exactly once.
        """

        self._backend_rng_before = None
        self._capture_backend_rng = True
        try:
            safe_logits, branches, auxiliary, intermediates = super().forward(
                X_raw, return_intermediates=True
            )
        finally:
            self._capture_backend_rng = False

        if self._backend_rng_before is None:
            raise RuntimeError("Failed to capture the PC-BBF backend RNG state")

        rng_after_safe = _capture_rng_state()
        graph_cache = {
            "feature_1": self.GCN.GCN_feature_1,
            "feature_2": self.GCN.GCN_feature_2,
            "last_adj": self.GCN.last_adj,
        }
        _restore_rng_state(self._backend_rng_before)
        try:
            with torch.no_grad():
                base_logits = self.GCN(
                    intermediates["H0"], self.last_adj_base
                ).detach()
        finally:
            _restore_rng_state(rng_after_safe)
            self.GCN.GCN_feature_1 = graph_cache["feature_1"]
            self.GCN.GCN_feature_2 = graph_cache["feature_2"]
            self.GCN.last_adj = graph_cache["last_adj"]

        intermediates["safe_logits"] = safe_logits
        intermediates["base_logits"] = base_logits
        intermediates["base_logits_stop_gradient"] = torch.tensor(
            True, device=safe_logits.device
        )
        intermediates["backend_rng_replayed"] = torch.tensor(
            True, device=safe_logits.device
        )

        if return_intermediates:
            return safe_logits, base_logits, branches, auxiliary, intermediates
        return safe_logits, base_logits, branches, auxiliary


__all__ = ["PCBBFSafeModel"]
