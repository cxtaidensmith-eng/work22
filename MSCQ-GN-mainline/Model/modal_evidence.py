"""End-to-end patient-only controls for additive and pairwise modal evidence.

All three modes share the same original per-modality encoder, gate priors,
input noise and one token dropout. No patient propagation or cross-modal
Transformer is used. Pairwise branches are bilinear supplements, not an
identified decomposition into statistically pure interactions.
"""

import math
from itertools import combinations

import torch
from torch import nn

from .network import Feature_Modal_Embedding, Modal_Token_Encoder


class _BilinearSupplement(nn.Module):
    """Independent low-rank branch for one unordered pair of modalities."""

    def __init__(self, hidden_size, rank, num_classes):
        super().__init__()
        self.left = nn.Linear(hidden_size, rank, bias=False)
        self.right = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, num_classes, bias=False)
        self.scale = 1.0 / math.sqrt(rank)
        # Initially B is exactly A. The projection gradients become nonzero
        # after the output weights have received their first learning update.
        nn.init.zeros_(self.output.weight)

    def forward(self, left, right):
        return self.output(self.left(left) * self.right(right)) * self.scale


class ModalEvidenceModel(nn.Module):
    """Patient-only additive, pairwise, or capacity-matched fusion classifier.

    ``tokens`` in the returned dictionary are the actual gated, dropped-out
    tokens used by the heads. Each token contains only its own modality's
    patient features. The constructor preserves the caller's CPU and all CUDA
    random states; common parameters match across modes for a training seed.
    No individual modality is assigned an auxiliary classification loss here.
    """

    MODES = ("additive", "pairwise", "fusion")

    def __init__(self, dd, cfg, mode, training_seed=0, rank=8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown modal evidence mode: {mode}")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        self.mode = mode
        self.rank = rank
        self.Hidden_size = int(cfg.Hidden_size)
        self._feature_num = int(dd["Feature_Num"])
        self._Label_num = int(dd["Class_Num"])
        self.modal_names = tuple(dd["Modal_Name"])
        self._modal_num = len(self.modal_names)
        self._modal_index = [list(indices) for indices in dd["Modal_Index"]]
        if self._modal_num < 1 or len(self._modal_index) != self._modal_num:
            raise ValueError("Modal_Name and Modal_Index must describe the same modalities")
        if min(self.Hidden_size, self._feature_num, self._Label_num) < 1:
            raise ValueError("Hidden size, feature count, and class count must be positive")
        if any(not indices for indices in self._modal_index):
            raise ValueError("Each modality must contain at least one feature")
        if any(index < 0 or index >= self._feature_num
               for indices in self._modal_index for index in indices):
            raise ValueError("Modality feature indices exceed Feature_Num")

        self.input_noise_std = float(cfg.input_noise_std)
        self.noise_scale = self.input_noise_std / 0.1 if self.input_noise_std > 0 else 0.0
        self.pairs = tuple(combinations(range(self._modal_num), 2))
        self.pair_indices = self.pairs
        self.fusion_width = None
        h, c, m = self.Hidden_size, self._Label_num, self._modal_num
        self.pairwise_head_budget = m * (h * c + c) + len(self.pairs) * (2 * h * rank + rank * c)

        # torch.manual_seed also affects CUDA: save every CUDA generator even
        # though the modules are constructed on CPU before an eventual .to().
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(3100 + int(training_seed))
            self.Feature_Modal = Feature_Modal_Embedding(self._modal_index, m, self._feature_num)
            self.modal_token_encoder = Modal_Token_Encoder(self._modal_index, m, h)

            # Preserve the current C model's modality-specific priors exactly.
            prior_logits, noise_std = [], []
            for name in self.modal_names:
                upper = name.upper()
                if "COG" in upper or "COGNITIVE" in upper:
                    prior_logits.append(4.0)
                    noise_std.append(0.0)
                elif "ROI" in upper:
                    prior_logits.append(2.0)
                    noise_std.append(0.05)
                elif "UCSFFSX" in upper or "MRI" in upper:
                    prior_logits.append(-1.0)
                    noise_std.append(0.3)
                else:
                    prior_logits.append(-2.0)
                    noise_std.append(0.3)
            self.modal_gate_logit = nn.Parameter(torch.tensor(prior_logits, dtype=torch.float32))
            self.register_buffer("_modal_noise_std", torch.tensor(noise_std, dtype=torch.float32))
            self.register_buffer("feature_to_modal", self.Feature_Modal.feature_to_modal.clone())

            # A single same-shaped stochastic operation in every mode. F has
            # no additional hidden dropout, so it cannot shift the next input
            # noise draw relative to A/B when random states are matched.
            self.token_dropout = nn.Dropout(float(cfg.Drop_rate))
            self.unimodal_heads = nn.ModuleList()
            self.pair_heads = nn.ModuleList()
            self.fusion_head = None
            if mode in ("additive", "pairwise"):
                self.unimodal_heads.extend(nn.Linear(h, c) for _ in range(m))
                if mode == "pairwise":
                    self.pair_heads.extend(
                        _BilinearSupplement(h, rank, c) for _ in self.pairs
                    )
            else:
                # F parameters: width * (M*H + 1 + C) + C. Match the budget
                # of all B classification heads, including unimodal biases.
                target_width = (self.pairwise_head_budget - c) / (m * h + 1 + c)
                candidates = {max(1, math.floor(target_width)), max(1, math.ceil(target_width))}
                self.fusion_width = min(
                    candidates,
                    key=lambda width: (abs(width * (m * h + 1 + c) + c
                                           - self.pairwise_head_budget), width),
                )
                self.fusion_head = nn.Sequential(
                    nn.Linear(m * h, self.fusion_width),
                    nn.GELU(),
                    nn.Linear(self.fusion_width, c),
                )

    def forward(self, x_raw):
        if x_raw.ndim != 2 or x_raw.shape[1] != self._feature_num:
            raise ValueError(f"Expected [N, {self._feature_num}] features, got {tuple(x_raw.shape)}")
        x = self.Feature_Modal(x_raw)
        if self.training:
            std = self._modal_noise_std[self.feature_to_modal] * self.noise_scale
            x = x + torch.randn_like(x) * std.view(1, -1)
        tokens = self.modal_token_encoder(x)
        tokens = tokens * self.modal_gate_logit.sigmoid().view(1, -1, 1)
        tokens = self.token_dropout(tokens)

        unimodal_logits, pair_logits = None, None
        if self.mode == "fusion":
            logits = self.fusion_head(tokens.flatten(start_dim=1))
        else:
            unimodal_logits = torch.stack(
                [head(tokens[:, modality]) for modality, head in enumerate(self.unimodal_heads)],
                dim=1,
            )
            logits = unimodal_logits.sum(dim=1)
            if self.mode == "pairwise":
                if self.pairs:
                    pair_logits = torch.stack(
                        [head(tokens[:, left], tokens[:, right])
                         for (left, right), head in zip(self.pairs, self.pair_heads)],
                        dim=1,
                    )
                else:
                    pair_logits = tokens.new_zeros((tokens.shape[0], 0, self._Label_num))
                logits = logits + pair_logits.sum(dim=1)
        return {"logits": logits, "unimodal_logits": unimodal_logits,
                "pair_logits": pair_logits, "tokens": tokens}

    def parameter_counts(self):
        """Actual trainable parameter counts plus the B-head matching target."""
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters()
                       if parameter.requires_grad) if module is not None else 0

        feature_embedding = count(self.Feature_Modal)
        encoder = count(self.modal_token_encoder)
        gate = self.modal_gate_logit.numel() if self.modal_gate_logit.requires_grad else 0
        unimodal = count(self.unimodal_heads)
        pairwise = count(self.pair_heads)
        fusion = count(self.fusion_head)
        return {"feature_embedding": feature_embedding, "encoder": encoder,
                "gate": gate, "common": feature_embedding + encoder + gate,
                "unimodal_heads": unimodal, "pair_heads": pairwise,
                "fusion_head": fusion, "heads": unimodal + pairwise + fusion,
                "total": count(self), "pairwise_head_budget": self.pairwise_head_budget,
                "fusion_width": self.fusion_width}
