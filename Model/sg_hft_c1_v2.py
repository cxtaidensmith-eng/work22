"""Structured Group HFT-C1 v2.

The model in this module is the historical C1 model with a small structured
group residual on MRI, PET, and ROI only.  The C1 modules are constructed
before every SG-HFT parameter, so resetting the random seed before building a
C1 reference and an SG-HFT model preserves identical C1 initialization.

Grouping is supplied explicitly through a serializable manifest.  The helper
``build_semantic_group_manifest`` implements the deterministic grouping used
for the fixed TADPOLE feature names; checkpoints retain every feature-index
assignment as persistent buffers, while the runner saves the full manifest.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hft_c1_lite import HFTC1LiteModel


SG_MODALITIES = ("MRI", "PET", "ROI")
ALL_CANONICAL_MODALITIES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")


def _canonical_modal_name(raw_name: str) -> str:
    """Map the fixed dataset aliases without relying on modality order."""

    name = str(raw_name).strip().upper()
    if name == "MRI" or "UCSFFSX" in name:
        return "MRI"
    if name == "PET" or "UCBERKELEYAV45" in name:
        return "PET"
    if name == "CSF" or "UPENNBIOMK" in name:
        return "CSF"
    if name == "PHS" or "RISK" in name or "GENETIC" in name:
        return "Risk"
    if "COG" in name:
        return "COG"
    if "ROI" in name:
        return "ROI"
    raise ValueError(f"Cannot map dataset modality name: {raw_name!r}")


def _manifest_groups(
    modality: str,
    group_names: Sequence[str],
    local_assignments: Sequence[Sequence[int]],
    feature_names: Sequence[str],
    global_indices: Sequence[int],
) -> list[dict[str, Any]]:
    if len(feature_names) != len(global_indices):
        raise ValueError(f"{modality}: feature-name/index count mismatch")
    groups: list[dict[str, Any]] = []
    seen: list[int] = []
    for group_name, local_indices in zip(group_names, local_assignments):
        local = [int(index) for index in local_indices]
        if not local:
            raise ValueError(f"{modality}/{group_name}: empty semantic group")
        if any(index < 0 or index >= len(feature_names) for index in local):
            raise ValueError(f"{modality}/{group_name}: invalid local index")
        seen.extend(local)
        groups.append(
            {
                "group_name": str(group_name),
                "local_feature_indices": local,
                "global_feature_indices": [int(global_indices[index]) for index in local],
                "feature_names": [str(feature_names[index]) for index in local],
            }
        )
    if sorted(seen) != list(range(len(feature_names))):
        raise ValueError(f"{modality}: semantic groups are not an exact partition")
    if not 4 <= len(groups) <= 8:
        raise ValueError(f"{modality}: expected 4..8 groups, got {len(groups)}")
    return groups


def build_semantic_group_manifest(
    dataset_dict: Mapping[str, Any],
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the deterministic TADPOLE MRI/PET/ROI grouping.

    MRI is partitioned by the explicit FreeSurfer measurement suffix.  PET is
    partitioned by cortical laterality and the explicit ``SIZE`` measurement
    marker, with non-cortical/aggregate measurements kept separate.  The nine
    ROI averages are partitioned by named measurement family.  No label,
    subject outcome, or original column ordering is used to choose a group.
    """

    raw_modalities = [str(value) for value in dataset_dict["Modal_Name"]]
    canonical = [_canonical_modal_name(value) for value in raw_modalities]
    if set(canonical) != set(ALL_CANONICAL_MODALITIES):
        raise ValueError(f"Unexpected modality set: {raw_modalities!r}")
    modal_indices = {
        name: [int(index) for index in dataset_dict["Modal_Index"][canonical.index(name)]]
        for name in SG_MODALITIES
    }

    mri_names = [str(value) for value in feature_names_by_modality["MRI"]]
    mri_buckets: dict[str, list[int]] = {
        "cortical_volume_CV": [],
        "surface_area_SA": [],
        "cortical_thickness_TA_TS": [],
        "subcortical_volume_SV": [],
    }
    for index, feature_name in enumerate(mri_names):
        prefix = feature_name.split("_", 1)[0].upper()
        match = re.fullmatch(r"ST\d+(CV|SA|TA|TS|SV)", prefix)
        if match is None:
            raise ValueError(f"MRI feature lacks a recognized measurement suffix: {feature_name}")
        suffix = match.group(1)
        if suffix == "CV":
            bucket = "cortical_volume_CV"
        elif suffix == "SA":
            bucket = "surface_area_SA"
        elif suffix in {"TA", "TS"}:
            bucket = "cortical_thickness_TA_TS"
        else:
            bucket = "subcortical_volume_SV"
        mri_buckets[bucket].append(index)

    pet_names = [str(value) for value in feature_names_by_modality["PET"]]
    pet_group_names = (
        "cortical_left_uptake",
        "cortical_left_size",
        "cortical_right_uptake",
        "cortical_right_size",
        "noncortical_aggregate_uptake",
        "noncortical_aggregate_size",
    )
    pet_buckets = {name: [] for name in pet_group_names}
    for index, feature_name in enumerate(pet_names):
        upper = feature_name.upper()
        is_size = "_SIZE_" in upper
        if upper.startswith("CTX_LH_"):
            bucket = "cortical_left_size" if is_size else "cortical_left_uptake"
        elif upper.startswith("CTX_RH_"):
            bucket = "cortical_right_size" if is_size else "cortical_right_uptake"
        else:
            bucket = (
                "noncortical_aggregate_size"
                if is_size
                else "noncortical_aggregate_uptake"
            )
        pet_buckets[bucket].append(index)

    roi_names = [str(value) for value in feature_names_by_modality["ROI"]]
    roi_name_to_group = {
        "MIDTEMP": "medial_temporal_structure",
        "FUSIFORM": "medial_temporal_structure",
        "HIPPOCAMPUS": "medial_temporal_structure",
        "ENTORHINAL": "medial_temporal_structure",
        "ICV": "global_structure",
        "WHOLEBRAIN": "global_structure",
        "VENTRICLES": "ventricular_structure",
        "AV45": "amyloid_AV45",
        "FDG": "glucose_metabolism_FDG",
    }
    roi_group_names = (
        "medial_temporal_structure",
        "global_structure",
        "ventricular_structure",
        "amyloid_AV45",
        "glucose_metabolism_FDG",
    )
    roi_buckets = {name: [] for name in roi_group_names}
    for index, feature_name in enumerate(roi_names):
        key = feature_name.strip().upper()
        if key not in roi_name_to_group:
            raise ValueError(f"ROI feature lacks a recognized semantic family: {feature_name}")
        roi_buckets[roi_name_to_group[key]].append(index)

    return {
        "MRI": _manifest_groups(
            "MRI",
            tuple(mri_buckets),
            tuple(mri_buckets.values()),
            mri_names,
            modal_indices["MRI"],
        ),
        "PET": _manifest_groups(
            "PET",
            pet_group_names,
            tuple(pet_buckets[name] for name in pet_group_names),
            pet_names,
            modal_indices["PET"],
        ),
        "ROI": _manifest_groups(
            "ROI",
            roi_group_names,
            tuple(roi_buckets[name] for name in roi_group_names),
            roi_names,
            modal_indices["ROI"],
        ),
    }


class StructuredGroupResidual(nn.Module):
    """One modality's group encoders, conditional gate, and rank-four delta."""

    def __init__(
        self,
        groups: Sequence[Mapping[str, Any]],
        hidden_size: int,
        *,
        group_hidden_size: int = 8,
        gate_hidden_size: int = 8,
        adapter_rank: int = 4,
    ):
        super().__init__()
        if not 4 <= len(groups) <= 8:
            raise ValueError(f"Expected 4..8 groups, got {len(groups)}")
        if group_hidden_size != 8 or gate_hidden_size != 8 or adapter_rank != 4:
            raise ValueError("SG-HFT v2 fixes group/gate width=8 and adapter rank=4")

        self.hidden_size = int(hidden_size)
        self.group_names = tuple(str(group["group_name"]) for group in groups)
        if len(set(self.group_names)) != len(self.group_names):
            raise ValueError("Group names must be unique within a modality")

        # ``group_encoders`` intentionally contains the group-specific first
        # layers; ``shared_group_output`` is the shared second layer required
        # by the parameter budget.
        self.group_encoders = nn.ModuleList()
        for group_index, group in enumerate(groups):
            indices = torch.as_tensor(
                group.get("global_feature_indices", group.get("feature_indices")),
                dtype=torch.long,
            )
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError(f"Group {self.group_names[group_index]} is empty")
            if int(torch.unique(indices).numel()) != int(indices.numel()):
                raise ValueError(f"Group {self.group_names[group_index]} has duplicate indices")
            self.register_buffer(f"group_feature_indices_{group_index}", indices)
            self.group_encoders.append(nn.Linear(int(indices.numel()), 8))

        # Sharing the second layer within a modality keeps all groups in the
        # same token space and reduces the complete SG addition below 20k.
        self.shared_group_output = nn.Linear(8, self.hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[-1].bias)

        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, 4),
            nn.GELU(),
            nn.Linear(4, self.hidden_size),
        )
        nn.init.xavier_uniform_(self.adapter[-1].weight, gain=0.05)
        nn.init.zeros_(self.adapter[-1].bias)

    def group_feature_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in getattr(self, f"group_feature_indices_{index}").tolist())
            for index in range(len(self.group_names))
        )

    def forward(
        self,
        preprocessed_features: torch.Tensor,
        c1_summary: torch.Tensor,
        c1_category_token: torch.Tensor,
        *,
        alpha: float,
        cap: float,
        eps: float,
    ) -> dict[str, torch.Tensor]:
        if preprocessed_features.ndim != 2:
            raise ValueError("Expected preprocessed features with shape [N,F]")
        expected = (preprocessed_features.size(0), self.hidden_size)
        if tuple(c1_summary.shape) != expected or tuple(c1_category_token.shape) != expected:
            raise ValueError("C1 token shape does not match the SG hidden size")

        group_tokens = []
        for index, input_layer in enumerate(self.group_encoders):
            feature_indices = getattr(self, f"group_feature_indices_{index}")
            group_values = preprocessed_features.index_select(1, feature_indices)
            hidden = F.gelu(input_layer(group_values))
            group_tokens.append(self.shared_group_output(hidden))
        tokens = torch.stack(group_tokens, dim=1)

        context = F.layer_norm(c1_summary, (self.hidden_size,)).detach()
        expanded_context = context.unsqueeze(1).expand(-1, tokens.size(1), -1)
        gate_values = torch.sigmoid(
            self.gate(torch.cat((tokens, expanded_context), dim=-1)).squeeze(-1)
        )
        gate_sum = gate_values.sum(dim=1, keepdim=True)
        normalized_gate_weights = gate_values / (gate_sum + float(eps))
        local_evidence = torch.einsum("nk,nkh->nh", normalized_gate_weights, tokens)

        raw_delta = self.adapter(local_evidence)
        shared_norm = c1_category_token.detach().norm(dim=-1, keepdim=True)
        raw_delta_norm = raw_delta.detach().norm(dim=-1, keepdim=True)
        cap_scale = torch.clamp(
            float(cap) * shared_norm / (raw_delta_norm + float(eps)),
            max=1.0,
        )
        delta = float(alpha) * raw_delta * cap_scale
        ratio = delta.detach().norm(dim=-1, keepdim=True) / (
            shared_norm + float(eps)
        )

        return {
            "group_tokens": tokens,
            "context": context,
            "gate_values": gate_values,
            "normalized_gate_weights": normalized_gate_weights,
            "local_evidence": local_evidence,
            "raw_delta": raw_delta,
            "delta": delta,
            "shared_norm": shared_norm,
            "raw_delta_norm": raw_delta_norm,
            "cap_scale": cap_scale,
            "ratio": ratio,
            "cap_saturated": cap_scale < (1.0 - 1e-7),
        }


class SGHFTC1V2Model(HFTC1LiteModel):
    """Exact C1 plus structured group residuals on MRI, PET, and ROI."""

    SG_MODALITIES = SG_MODALITIES
    CAP = 0.08
    EPS = 1e-8

    def __init__(
        self,
        *args,
        group_manifest: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        cap: float = 0.08,
        sg_enabled: bool = True,
        **kwargs,
    ):
        if "hft_enabled" in kwargs:
            raise ValueError("Use sg_enabled; the HFT-v1 branches are forbidden")
        # HFTC1LiteModel with hft_enabled=False is exactly the locked C1 model.
        # Its historical cap is unused; the fixed SG cap is installed below.
        super().__init__(
            *args,
            cap=0.15,
            attention_temperature=1.0,
            adapter_rank=8,
            hft_enabled=False,
            **kwargs,
        )
        if abs(float(cap) - self.CAP) > 1e-12:
            raise ValueError("SG-HFT-C1 v2 fixes cap=0.08")
        self.cap = float(cap)
        self.eps = float(self.EPS)
        self.sg_enabled = bool(sg_enabled)
        self.sg_branches = nn.ModuleDict()
        self._sg_group_manifest: dict[str, list[dict[str, Any]]] = {}

        # SG modules are appended after the complete C1 construction, but their
        # initialization must not advance the random stream used by historical
        # C1 input noise and dropout during epochs 1--20.  Preserve the exact
        # post-C1 CPU RNG state while still retaining the sampled SG weights.
        c1_rng_state = torch.random.get_rng_state()
        try:
            if self.sg_enabled:
                if group_manifest is None:
                    raise ValueError("Enabled SG-HFT requires an explicit group manifest")
                if set(group_manifest) != set(self.SG_MODALITIES):
                    raise ValueError(
                        f"SG manifest must contain exactly {self.SG_MODALITIES}, "
                        f"got {tuple(group_manifest)}"
                    )
                for modality in self.SG_MODALITIES:
                    normalized = self._normalize_groups(modality, group_manifest[modality])
                    self._sg_group_manifest[modality] = normalized
                    self.sg_branches[modality] = StructuredGroupResidual(
                        normalized, self.Hidden_size
                    )
        finally:
            torch.random.set_rng_state(c1_rng_state)
        if self.sg_parameter_count() > 20_000:
            raise ValueError(
                f"SG-HFT adds {self.sg_parameter_count()} parameters (>20,000)"
            )

    def _normalize_groups(
        self,
        modality: str,
        groups: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        expected = tuple(int(value) for value in self.hft_feature_indices_by_modality[modality])
        normalized: list[dict[str, Any]] = []
        seen: list[int] = []
        for index, group in enumerate(groups):
            group_name = str(group.get("group_name", f"group_{index:02d}"))
            raw_indices = group.get("global_feature_indices", group.get("feature_indices"))
            if raw_indices is None:
                raise ValueError(f"{modality}/{group_name}: missing feature indices")
            indices = [int(value) for value in raw_indices]
            if not indices:
                raise ValueError(f"{modality}/{group_name}: empty group")
            seen.extend(indices)
            normalized_group: dict[str, Any] = {
                "group_name": group_name,
                "global_feature_indices": indices,
            }
            for optional_key in ("local_feature_indices", "feature_names"):
                if optional_key in group:
                    normalized_group[optional_key] = list(group[optional_key])
            normalized.append(normalized_group)
        if not 4 <= len(normalized) <= 8:
            raise ValueError(f"{modality}: expected 4..8 groups, got {len(normalized)}")
        if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
            raise ValueError(f"{modality}: groups must partition the modality exactly once")
        return normalized

    @staticmethod
    def activation_alpha(epoch: int | None) -> float:
        """Return the preregistered delayed activation multiplier.

        ``epoch=None`` is inference mode and uses the fully active branch.
        Training and best-epoch evaluation should pass the epoch explicitly.
        """

        if epoch is None:
            return 1.0
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("Epoch numbering is one-based")
        if epoch <= 20:
            return 0.0
        if epoch <= 40:
            return float(epoch - 20) / 20.0
        return 1.0

    def group_manifest(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-serializable copy used by fold manifests/checkpoints."""

        return {
            modality: [dict(group) for group in groups]
            for modality, groups in self._sg_group_manifest.items()
        }

    def sg_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.sg_branches.parameters())

    def c1_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) - self.sg_parameter_count()

    def inference_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        X_raw: torch.Tensor,
        return_intermediates: bool = False,
        *,
        activation_epoch: int | None = None,
        epoch: int | None = None,
    ):
        if activation_epoch is not None and epoch is not None:
            raise ValueError("Pass activation_epoch or epoch, not both")
        effective_epoch = activation_epoch if activation_epoch is not None else epoch
        self.last_modal_tokens = None
        feature_modal_output = self.Feature_Modal(X_raw)
        X = feature_modal_output
        if self.training:
            per_feature_noise_std = self._modal_noise_std[self.feature_to_modal]
            X = X + torch.randn_like(X) * (
                per_feature_noise_std * self.noise_scale
            ).view(1, -1)

        modal_gate = torch.sigmoid(self.modal_gate_logit)
        feature_gate = modal_gate[self.feature_to_modal]
        X_gated = X * feature_gate.view(1, -1)

        H = self.modal_token_encoder(X)
        H = H * modal_gate.view(1, -1, 1)
        modal_tokens_pre_transformer = H
        for block in self.shared_transformer:
            H = block(H)
        modal_tokens_post_transformer = H
        self.last_modal_tokens = H.detach()

        private_residuals = torch.stack(
            [
                adapter(H[:, modal_index])
                for modal_index, adapter in enumerate(self.private_adapters)
            ],
            dim=1,
        )
        c1_category_tokens = H + private_residuals
        alpha = self.activation_alpha(effective_epoch)

        sg_outputs: dict[str, dict[str, torch.Tensor]] = {}
        if self.sg_enabled:
            # During the preregistered inactive warmup, the SG contribution is
            # exactly zero.  Keep computing the diagnostic tensors, but do not
            # attach the SG parameters to the loss graph: a zero-valued gradient
            # would still make Adam apply coupled weight decay and collapse the
            # branch before its epoch-21 activation.  Respect an outer no-grad
            # context during evaluation as well.
            track_sg_grad = torch.is_grad_enabled() and alpha != 0.0
            with torch.set_grad_enabled(track_sg_grad):
                for modality, branch in self.sg_branches.items():
                    modal_index = self.modal_index_by_name[modality]
                    sg_outputs[modality] = branch(
                        X,
                        H[:, modal_index],
                        c1_category_tokens[:, modal_index],
                        alpha=alpha,
                        cap=self.cap,
                        eps=self.eps,
                    )

        category_token_items = []
        for modal_index, canonical_name in enumerate(self.canonical_modal_names):
            category_token = c1_category_tokens[:, modal_index]
            if canonical_name in sg_outputs:
                category_token = category_token + sg_outputs[canonical_name]["delta"]
            category_token_items.append(category_token)
        category_tokens = torch.stack(category_token_items, dim=1)

        label_embeddings = []
        auxiliary_outputs = []
        for pool, auxiliary_head in zip(self.label_pools, self._Auxi_classifier):
            branch_embedding = pool(category_tokens)
            label_embeddings.append(branch_embedding)
            auxiliary_outputs.append(auxiliary_head(branch_embedding))
        self.last_branch_outputs = label_embeddings
        routed_branch_concat = torch.cat(label_embeddings, dim=-1)
        self.last_routed_branch_concat = routed_branch_concat
        category_embedding = self.Message_MLP(routed_branch_concat)

        # Global, DIFFormer, and classifier are the unmodified C1 paths.
        global_embedding = self.Global_Message(X_gated)
        adjacency = torch.eye(
            X_raw.size(0), device=X_raw.device, dtype=X_raw.dtype
        )
        self.last_adj_base = adjacency
        self.last_label_relation = None
        fused_embedding = self._fuse_semantic_branches(
            category_embedding, global_embedding
        )
        raw_logits = self.GCN(fused_embedding, adjacency)

        if return_intermediates:
            intermediates: dict[str, Any] = {
                "feature_modal_output": feature_modal_output,
                "modal_tokens_pre_transformer": modal_tokens_pre_transformer,
                "modal_tokens_post_transformer": modal_tokens_post_transformer,
                "private_residuals": private_residuals,
                "c1_category_tokens": c1_category_tokens,
                "category_tokens": category_tokens,
                "label_embeddings": label_embeddings,
                "Y": category_embedding,
                "G": global_embedding,
                "H_fused": fused_embedding,
                "raw_logits": raw_logits,
                "sg_enabled": self.sg_enabled,
                "sg_alpha": alpha,
                "sg_cap": self.cap,
                "sg_group_manifest": self.group_manifest(),
                "sg_shared_by_modality": {
                    modality: c1_category_tokens[:, self.modal_index_by_name[modality]]
                    for modality in self.SG_MODALITIES
                    if modality in sg_outputs
                },
                "sg_raw_delta_by_modality": {
                    name: output["raw_delta"] for name, output in sg_outputs.items()
                },
                "sg_delta_by_modality": {
                    name: output["delta"] for name, output in sg_outputs.items()
                },
                "sg_ratio_by_modality": {
                    name: output["ratio"] for name, output in sg_outputs.items()
                },
                "sg_cap_scale_by_modality": {
                    name: output["cap_scale"] for name, output in sg_outputs.items()
                },
                "sg_cap_saturated_by_modality": {
                    name: output["cap_saturated"] for name, output in sg_outputs.items()
                },
                "sg_gate_values_by_modality": {
                    name: output["gate_values"] for name, output in sg_outputs.items()
                },
                "sg_group_gates_by_modality": {
                    name: output["gate_values"] for name, output in sg_outputs.items()
                },
                "sg_normalized_gate_weights_by_modality": {
                    name: output["normalized_gate_weights"]
                    for name, output in sg_outputs.items()
                },
                "sg_local_evidence_by_modality": {
                    name: output["local_evidence"] for name, output in sg_outputs.items()
                },
            }
            return raw_logits, label_embeddings, auxiliary_outputs, intermediates
        return raw_logits, label_embeddings, auxiliary_outputs


__all__ = [
    "SG_MODALITIES",
    "SGHFTC1V2Model",
    "StructuredGroupResidual",
    "build_semantic_group_manifest",
]
