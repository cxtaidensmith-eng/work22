"""MG-JEPA pretraining components for the exact C1 representation backbone.

This module deliberately does not define a replacement classifier.  It copies
only the three trainable parts of C1 named by the protocol -- the real modal
token encoder, shared modal Transformer, and six C1 private adapters -- and
exports those weights back under their original C1 state-dict names.  REG
tokens, modality mask tokens, predictors, and the EMA teacher therefore cannot
enter the supervised C1 model or change its inference parameter count.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


C1_PARAMETER_COUNT = 862_971
C1_HIDDEN_SIZE = 96
C1_PRETRAIN_BACKBONE_PARAMETER_COUNT = 224_400
STUDENT_WITH_REG_PARAMETER_COUNT = 224_784
PRETRAIN_TRAINABLE_PARAMETER_COUNT = 300_240
TEACHER_FROZEN_PARAMETER_COUNT = 224_784
PRETRAIN_ALL_PARAMETER_COUNT = 525_024
MODALITY_COUNT = 6
REG_TOKEN_COUNT = 4
PREDICTOR_HIDDEN_MULTIPLIER = 2
VISIBLE_FEATURE_MASK_RATE = 0.15
TEACHER_DECAY = 0.99

_C1_BACKBONE_PREFIXES = (
    "modal_token_encoder.",
    "shared_transformer.",
    "private_adapters.",
)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def make_mask_generator(
    device: torch.device | str,
    seed: int,
) -> torch.Generator:
    """Create the explicit generator used only by MG-JEPA feature masks."""

    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def masked_modality_for_epoch(epoch: int) -> int:
    """Return the locked cyclic whole-modality mask for a zero-based epoch."""

    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("MG-JEPA epoch must be non-negative")
    return epoch % MODALITY_COUNT


def capture_rng_state() -> dict[str, object]:
    """Capture global RNG state so formal C1 initialization can be restored."""

    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else None
        ),
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    torch.random.set_rng_state(state["cpu"])
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


class TwoLayerPredictor(nn.Module):
    """The fixed two-layer GELU predictor used only during pretraining."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MGJEPABackbone(nn.Module):
    """An exact copy of C1's representation-producing trainable modules.

    ``private`` is C1's actual per-modality category representation,
    ``shared + private_adapter(shared)``.  Using the residual alone would make
    every initial teacher target exactly zero because C1 intentionally
    zero-initializes each adapter's output projection, leaving the locked
    cosine private objective with no gradient.
    """

    def __init__(
        self,
        modal_token_encoder: nn.Module,
        shared_transformer: nn.ModuleList,
        private_adapters: nn.ModuleList,
        modal_gate: torch.Tensor,
        modal_indices,
        *,
        hidden_size: int,
        reg_token_count: int = REG_TOKEN_COUNT,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.modality_count = len(modal_indices)
        self.reg_token_count = int(reg_token_count)
        if self.modality_count != MODALITY_COUNT:
            raise ValueError(f"MG-JEPA requires six modalities, got {self.modality_count}")
        if self.reg_token_count != REG_TOKEN_COUNT:
            raise ValueError("MG-JEPA v1 fixes four REG tokens")
        if tuple(modal_gate.shape) != (self.modality_count,):
            raise ValueError("C1 modal gate shape changed")

        # These are deep copies of the real C1 children, not reimplementations.
        self.modal_token_encoder = modal_token_encoder
        self.shared_transformer = shared_transformer
        self.private_adapters = private_adapters
        self.modal_indices = tuple(tuple(int(index) for index in values) for values in modal_indices)
        self.register_buffer("modal_gate", modal_gate.detach().clone(), persistent=True)

        device, dtype = _module_device_dtype(self.modal_token_encoder)
        self.reg_tokens = nn.Parameter(
            torch.empty(
                self.reg_token_count,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.reg_tokens, std=0.02)

    @classmethod
    def from_c1(
        cls,
        c1_model: nn.Module,
        *,
        reg_token_count: int = REG_TOKEN_COUNT,
    ) -> "MGJEPABackbone":
        required = (
            "modal_token_encoder",
            "shared_transformer",
            "private_adapters",
            "modal_gate_logit",
            "Feature_Modal",
            "_modal_index",
            "Hidden_size",
        )
        missing = [name for name in required if not hasattr(c1_model, name)]
        if missing:
            raise TypeError(f"Not a real C1 model; missing {missing}")
        if getattr(c1_model, "cme_arm", None) != "c1":
            raise ValueError("MG-JEPA must be initialized from the C1 arm")
        if int(c1_model.Hidden_size) != C1_HIDDEN_SIZE:
            raise ValueError(f"MG-JEPA-C1 v1 fixes hidden size={C1_HIDDEN_SIZE}")
        if len(c1_model.private_adapters) != MODALITY_COUNT:
            raise ValueError("C1 private adapter count changed")

        # Feature_Modal is excluded by the protocol.  At the saved original C1
        # initialization it is exactly identity, so raw preprocessed features
        # are exactly the encoder input.  Refuse a later/trained C1 silently.
        feature_modal = c1_model.Feature_Modal
        if not (
            torch.equal(
                feature_modal.modal_gain.weight.detach(),
                torch.ones_like(feature_modal.modal_gain.weight),
            )
            and torch.equal(
                feature_modal.modal_bias.weight.detach(),
                torch.zeros_like(feature_modal.modal_bias.weight),
            )
        ):
            raise ValueError("MG-JEPA requires the original identity Feature_Modal initialization")

        return cls(
            deepcopy(c1_model.modal_token_encoder),
            deepcopy(c1_model.shared_transformer),
            deepcopy(c1_model.private_adapters),
            torch.sigmoid(c1_model.modal_gate_logit.detach()),
            c1_model._modal_index,
            hidden_size=int(c1_model.Hidden_size),
            reg_token_count=reg_token_count,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        masked_modality: int | None = None,
        modality_mask_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"Expected [N,F] features, got {tuple(features.shape)}")
        if masked_modality is not None:
            masked_modality = int(masked_modality)
            if not 0 <= masked_modality < self.modality_count:
                raise ValueError(f"Invalid masked modality: {masked_modality}")
            if modality_mask_tokens is None or tuple(modality_mask_tokens.shape) != (
                self.modality_count,
                self.hidden_size,
            ):
                raise ValueError("Student masking requires six hidden-size modality mask tokens")

        encoder_tokens = self.modal_token_encoder(features)
        transformer_input = encoder_tokens * self.modal_gate.view(1, -1, 1)
        if masked_modality is not None:
            # Replacement occurs at the exact tensor entering C1's shared
            # Transformer, after the historical modal-gate multiplication.
            transformer_input = transformer_input.clone()
            transformer_input[:, masked_modality] = modality_mask_tokens[
                masked_modality
            ].to(transformer_input)

        batch_size = features.size(0)
        reg = self.reg_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        contextual_with_reg = torch.cat((transformer_input, reg), dim=1)
        for block in self.shared_transformer:
            contextual_with_reg = block(contextual_with_reg)
        shared = contextual_with_reg[:, : self.modality_count]
        reg_output = contextual_with_reg[:, self.modality_count :]
        private_residual = torch.stack(
            [
                adapter(shared[:, modality])
                for modality, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        private = shared + private_residual
        return {
            "encoder_tokens": encoder_tokens,
            "transformer_input": transformer_input,
            "shared": shared,
            "private_residual": private_residual,
            "private": private,
            "reg_output": reg_output,
        }

    def export_c1_state(self, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        exported: dict[str, torch.Tensor] = {}
        modules = (
            ("modal_token_encoder.", self.modal_token_encoder),
            ("shared_transformer.", self.shared_transformer),
            ("private_adapters.", self.private_adapters),
        )
        for prefix, module in modules:
            for name, value in module.state_dict().items():
                clone = value.detach().clone()
                exported[prefix + name] = clone.cpu() if cpu else clone
        return exported

    def c1_backbone_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (
                self.modal_token_encoder,
                self.shared_transformer,
                self.private_adapters,
            )
            for parameter in module.parameters()
        )


def _split_exported_state(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }


def load_pretrained_backbone_into_c1(
    c1_model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly overlay only exported MG-JEPA Student weights onto pure C1."""

    modules = (
        ("modal_token_encoder.", c1_model.modal_token_encoder),
        ("shared_transformer.", c1_model.shared_transformer),
        ("private_adapters.", c1_model.private_adapters),
    )
    expected = {
        prefix + name
        for prefix, module in modules
        for name in module.state_dict()
    }
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"MG-JEPA/C1 backbone state mismatch; missing={missing}, unexpected={unexpected}"
        )
    for prefix, module in modules:
        module.load_state_dict(_split_exported_state(state, prefix), strict=True)


class MGJEPAPretrainer(nn.Module):
    """Student/EMA-Teacher latent prediction system used before C1 fine-tuning."""

    def __init__(
        self,
        c1_model: nn.Module,
        *,
        teacher_decay: float = TEACHER_DECAY,
        visible_feature_mask_rate: float = VISIBLE_FEATURE_MASK_RATE,
        reg_token_count: int = REG_TOKEN_COUNT,
    ):
        super().__init__()
        self.teacher_decay = float(teacher_decay)
        self.visible_feature_mask_rate = float(visible_feature_mask_rate)
        if abs(self.teacher_decay - TEACHER_DECAY) > 1e-12:
            raise ValueError("MG-JEPA v1 fixes EMA Teacher decay=0.99")
        if abs(self.visible_feature_mask_rate - VISIBLE_FEATURE_MASK_RATE) > 1e-12:
            raise ValueError("MG-JEPA v1 fixes visible-feature mask rate=0.15")

        self.student = MGJEPABackbone.from_c1(
            c1_model, reg_token_count=reg_token_count
        )
        self.teacher = deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

        hidden_size = self.student.hidden_size
        device, dtype = _module_device_dtype(self.student)
        self.modality_mask_tokens = nn.Parameter(
            torch.empty(
                MODALITY_COUNT,
                hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.modality_mask_tokens, std=0.02)
        predictor_hidden = PREDICTOR_HIDDEN_MULTIPLIER * hidden_size
        self.shared_predictor = TwoLayerPredictor(
            hidden_size, predictor_hidden, hidden_size
        )
        self.private_modality_embedding = nn.Embedding(
            MODALITY_COUNT, hidden_size
        )
        self.private_predictor = TwoLayerPredictor(
            hidden_size, predictor_hidden, hidden_size
        )
        self.to(device=device, dtype=dtype)
        self._assert_teacher_frozen_and_equal()
        if self.student.c1_backbone_parameter_count() != C1_PRETRAIN_BACKBONE_PARAMETER_COUNT:
            raise RuntimeError("C1 pretraining backbone parameter count changed")
        trainable_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        teacher_count = sum(parameter.numel() for parameter in self.teacher.parameters())
        all_count = sum(parameter.numel() for parameter in self.parameters())
        if trainable_count != PRETRAIN_TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError(f"MG-JEPA trainable parameter count changed: {trainable_count}")
        if teacher_count != TEACHER_FROZEN_PARAMETER_COUNT:
            raise RuntimeError(f"MG-JEPA Teacher parameter count changed: {teacher_count}")
        if all_count != PRETRAIN_ALL_PARAMETER_COUNT:
            raise RuntimeError(f"MG-JEPA wrapper parameter count changed: {all_count}")

    @classmethod
    def from_c1(cls, c1_model: nn.Module, **kwargs) -> "MGJEPAPretrainer":
        return cls(c1_model, **kwargs)

    def _assert_teacher_frozen_and_equal(self) -> None:
        student_state = self.student.state_dict()
        teacher_state = self.teacher.state_dict()
        if student_state.keys() != teacher_state.keys():
            raise RuntimeError("Student/Teacher backbone state keys differ")
        for name in student_state:
            if not torch.equal(student_state[name], teacher_state[name]):
                raise RuntimeError(f"Teacher is not an exact Student copy: {name}")
        if any(parameter.requires_grad for parameter in self.teacher.parameters()):
            raise RuntimeError("Teacher unexpectedly requires gradients")

    def train(self, mode: bool = True):
        super().train(mode)
        # The complete-view Teacher is deterministic and never uses dropout or
        # stochastic depth, even while the Student is training.
        self.teacher.eval()
        return self

    def pretrain_parameters(self) -> Iterator[nn.Parameter]:
        for module in (
            self.student,
            self.shared_predictor,
            self.private_modality_embedding,
            self.private_predictor,
        ):
            yield from module.parameters()
        yield self.modality_mask_tokens

    student_parameters = pretrain_parameters

    def _draw_uniform(
        self,
        shape,
        *,
        generator: torch.Generator,
        target_device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(generator, torch.Generator):
            raise TypeError("MG-JEPA masking requires an explicit torch.Generator")
        generator_device = torch.device(generator.device)
        return torch.rand(
            shape,
            generator=generator,
            device=generator_device,
            dtype=torch.float32,
        ).to(device=target_device)

    def corrupt_student_view(
        self,
        complete_features: torch.Tensor,
        *,
        masked_modality: int,
        mask_generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if complete_features.ndim != 2:
            raise ValueError("MG-JEPA expects a two-dimensional train-fold tensor")
        masked_modality = int(masked_modality)
        if not 0 <= masked_modality < MODALITY_COUNT:
            raise ValueError(f"Invalid masked modality: {masked_modality}")
        corrupted = complete_features.clone()
        value_mask = torch.zeros_like(complete_features, dtype=torch.bool)
        for modality, indices in enumerate(self.student.modal_indices):
            index = torch.as_tensor(
                indices, device=complete_features.device, dtype=torch.long
            )
            if modality == masked_modality:
                corrupted.index_fill_(1, index, 0.0)
                value_mask[:, index] = True
            else:
                visible_mask = self._draw_uniform(
                    (complete_features.size(0), len(indices)),
                    generator=mask_generator,
                    target_device=complete_features.device,
                ) < self.visible_feature_mask_rate
                selected = corrupted.index_select(1, index).masked_fill(
                    visible_mask, 0.0
                )
                corrupted[:, index] = selected
                value_mask[:, index] = visible_mask
        return corrupted, value_mask

    @staticmethod
    def _mean_cosine_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction, p=2, dim=-1, eps=1e-8)
        target = F.normalize(target.detach(), p=2, dim=-1, eps=1e-8)
        return (1.0 - (prediction * target).sum(dim=-1)).mean()

    @staticmethod
    def _mean_raw_cosine(
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        return (
            F.normalize(student, p=2, dim=-1, eps=1e-8)
            * F.normalize(teacher.detach(), p=2, dim=-1, eps=1e-8)
        ).sum(dim=-1).mean()

    def forward(
        self,
        complete_train_features: torch.Tensor,
        *,
        masked_modality: int,
        mask_generator: torch.Generator,
    ) -> dict[str, torch.Tensor | int]:
        """Compute the label-free full-batch MG-JEPA objective for one fold."""

        masked_modality = int(masked_modality)
        corrupted, feature_mask = self.corrupt_student_view(
            complete_train_features,
            masked_modality=masked_modality,
            mask_generator=mask_generator,
        )
        student = self.student(
            corrupted,
            masked_modality=masked_modality,
            modality_mask_tokens=self.modality_mask_tokens,
        )
        self.teacher.eval()
        with torch.no_grad():
            teacher = self.teacher(complete_train_features)

        shared_predictions = self.shared_predictor(student["shared"])
        shared_modal_losses = torch.stack(
            [
                self._mean_cosine_loss(
                    shared_predictions[:, modality],
                    teacher["shared"][:, modality],
                )
                for modality in range(MODALITY_COUNT)
            ]
        )
        loss_shared = shared_modal_losses.mean()

        modality_ids = torch.arange(
            MODALITY_COUNT, device=complete_train_features.device
        )
        modality_embedding = self.private_modality_embedding(modality_ids)
        private_input = student["private"] + modality_embedding.unsqueeze(0).expand(
            complete_train_features.size(0), -1, -1
        )
        private_predictions = self.private_predictor(private_input)
        visible_modalities = [
            modality
            for modality in range(MODALITY_COUNT)
            if modality != masked_modality
        ]
        private_modal_losses = torch.stack(
            [
                self._mean_cosine_loss(
                    private_predictions[:, modality],
                    teacher["private"][:, modality],
                )
                for modality in visible_modalities
            ]
        )
        loss_private = private_modal_losses.mean()
        loss = loss_shared + 0.5 * loss_private

        shared_raw_cosine = torch.stack(
            [
                self._mean_raw_cosine(
                    student["shared"][:, modality],
                    teacher["shared"][:, modality],
                )
                for modality in range(MODALITY_COUNT)
            ]
        )
        private_raw_cosine = torch.stack(
            [
                self._mean_raw_cosine(
                    student["private"][:, modality],
                    teacher["private"][:, modality],
                )
                for modality in range(MODALITY_COUNT)
            ]
        )
        return {
            "loss": loss,
            "loss_shared": loss_shared,
            "loss_private": loss_private,
            "shared_loss_by_modality": shared_modal_losses,
            "private_loss_by_visible_modality": private_modal_losses,
            "visible_modalities": torch.tensor(
                visible_modalities,
                device=complete_train_features.device,
                dtype=torch.long,
            ),
            "masked_modality": masked_modality,
            "feature_mask": feature_mask,
            "corrupted_features": corrupted,
            "student_shared": student["shared"],
            "student_private": student["private"],
            "student_private_residual": student["private_residual"],
            "teacher_shared": teacher["shared"].detach(),
            "teacher_private": teacher["private"].detach(),
            "teacher_private_residual": teacher["private_residual"].detach(),
            "shared_prediction": shared_predictions,
            "private_prediction": private_predictions,
            "teacher_student_shared_cosine_by_modality": shared_raw_cosine,
            "teacher_student_private_cosine_by_modality": private_raw_cosine,
            "student_shared_batch_std": student["shared"].std(
                dim=0, unbiased=False
            ).mean(),
            "student_private_batch_std": student["private"].std(
                dim=0, unbiased=False
            ).mean(),
            "teacher_shared_batch_std": teacher["shared"].std(
                dim=0, unbiased=False
            ).mean().detach(),
            "teacher_private_batch_std": teacher["private"].std(
                dim=0, unbiased=False
            ).mean().detach(),
        }

    @torch.no_grad()
    def update_teacher(self, decay: float | None = None) -> float:
        decay = self.teacher_decay if decay is None else float(decay)
        if abs(decay - self.teacher_decay) > 1e-12:
            raise ValueError("MG-JEPA v1 Teacher decay cannot be changed")
        student_parameters = dict(self.student.named_parameters())
        teacher_parameters = dict(self.teacher.named_parameters())
        if student_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError("Student/Teacher parameter keys changed")
        maximum_change = 0.0
        for name, teacher_parameter in teacher_parameters.items():
            before = teacher_parameter.detach().clone()
            teacher_parameter.mul_(decay).add_(
                student_parameters[name].detach(), alpha=1.0 - decay
            )
            maximum_change = max(
                maximum_change,
                float((teacher_parameter - before).abs().max().cpu()),
            )
        student_buffers = dict(self.student.named_buffers())
        teacher_buffers = dict(self.teacher.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise RuntimeError("Student/Teacher buffer keys changed")
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name].detach())
        self.teacher.eval()
        return maximum_change

    @torch.no_grad()
    def complete_view_diagnostics(
        self,
        complete_train_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate post-pretraining Student and Teacher on the same full view.

        This is the canonical source for final cosine/std/collapse reporting;
        diagnostics must not reuse the final epoch's modality-masked Student
        batch.  No labels or test subjects are accepted by this interface.
        """

        was_training = self.training
        self.eval()
        student = self.student(complete_train_features)
        teacher = self.teacher(complete_train_features)
        shared_cosine = torch.stack(
            [
                self._mean_raw_cosine(
                    student["shared"][:, modality],
                    teacher["shared"][:, modality],
                )
                for modality in range(MODALITY_COUNT)
            ]
        )
        private_cosine = torch.stack(
            [
                self._mean_raw_cosine(
                    student["private"][:, modality],
                    teacher["private"][:, modality],
                )
                for modality in range(MODALITY_COUNT)
            ]
        )
        payload = {
            "student_shared": student["shared"].detach(),
            "student_private": student["private"].detach(),
            "student_private_residual": student["private_residual"].detach(),
            "teacher_shared": teacher["shared"].detach(),
            "teacher_private": teacher["private"].detach(),
            "teacher_private_residual": teacher["private_residual"].detach(),
            "teacher_student_shared_cosine_by_modality": shared_cosine.detach(),
            "teacher_student_private_cosine_by_modality": private_cosine.detach(),
            "cosines": {
                "shared": shared_cosine.detach(),
                "private": private_cosine.detach(),
            },
            "student_shared_batch_std": student["shared"].std(
                dim=0, unbiased=False
            ).mean().detach(),
            "student_private_batch_std": student["private"].std(
                dim=0, unbiased=False
            ).mean().detach(),
            "teacher_shared_batch_std": teacher["shared"].std(
                dim=0, unbiased=False
            ).mean().detach(),
            "teacher_private_batch_std": teacher["private"].std(
                dim=0, unbiased=False
            ).mean().detach(),
        }
        if was_training:
            self.train(True)
        return payload

    def export_student_c1_state(self, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        return self.student.export_c1_state(cpu=cpu)

    def load_student_into_c1(self, c1_model: nn.Module) -> None:
        load_pretrained_backbone_into_c1(
            c1_model, self.export_student_c1_state(cpu=True)
        )

    def teacher_has_any_gradient(self) -> bool:
        return any(parameter.grad is not None for parameter in self.teacher.parameters())


def build_pretrainer_preserving_rng(
    c1_model: nn.Module,
    **kwargs,
) -> MGJEPAPretrainer:
    """Construct MG-JEPA modules without advancing formal C1 global RNG."""

    rng = capture_rng_state()
    try:
        pretrainer = MGJEPAPretrainer.from_c1(c1_model, **kwargs)
    finally:
        restore_rng_state(rng)
    return pretrainer


def exported_state_is_c1_only(state: Mapping[str, torch.Tensor]) -> bool:
    return bool(state) and all(
        any(name.startswith(prefix) for prefix in _C1_BACKBONE_PREFIXES)
        for name in state
    )
