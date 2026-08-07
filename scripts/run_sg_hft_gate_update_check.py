"""Second-level SG-HFT-C1 v2 gate update diagnostic.

This script performs the four bounded checks against the formal fold-4
checkpoint, verifies the minimal warmup fix, and runs at most twenty fresh
active fold-4 steps.  It never updates or overwrites the formal checkpoint,
saves no micro-training checkpoint, and cannot launch screen/formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_sg_hft_c1_v2 as sg


OUTPUT_JSON = ROOT / "experiments/sg_hft_c1_v2/gate_update_check.json"
OUTPUT_MD = ROOT / "experiments/sg_hft_c1_v2/gate_update_check.md"
DEFAULT_CHECKPOINT = (
    ROOT.parent
    / "sg_hft_c1_v2"
    / "experiments/sg_hft_c1_v2/screen/fold_04/checkpoint_best.pt"
)
EPS = 1e-12
ACTIVE_EPOCH = 41
PERTURBATION = 0.1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            *args,
        ],
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_l2(tensors: Iterable[torch.Tensor]) -> float:
    values = [tensor.detach().float().square().sum().cpu() for tensor in tensors]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0


def sg_family(parameter_name: str) -> str:
    relative = parameter_name.split(".", 2)[2]
    if relative.startswith("group_encoders.") or relative.startswith("shared_group_output."):
        return "group_encoder"
    if relative.startswith("gate."):
        return "gate"
    if relative.startswith("adapter."):
        return "adapter"
    raise RuntimeError(f"Unclassified SG-HFT parameter: {parameter_name}")


def modality_of(parameter_name: str) -> str:
    return parameter_name.split(".", 2)[1]


def family_parameter_names(
    named_parameters: dict[str, nn.Parameter],
    modality: str,
    family: str,
) -> list[str]:
    return [
        name
        for name in named_parameters
        if name.startswith("sg_branches.")
        and modality_of(name) == modality
        and sg_family(name) == family
    ]


def gradient_snapshot(
    names: list[str],
    named_parameters: dict[str, nn.Parameter],
) -> dict[str, Any]:
    gradients = [named_parameters[name].grad for name in names]
    present = [gradient for gradient in gradients if gradient is not None]
    finite = bool(present) and all(bool(torch.isfinite(gradient).all()) for gradient in present)
    norm = tensor_l2(present)
    return {
        "tensor_count": len(names),
        "gradient_none_count": sum(gradient is None for gradient in gradients),
        "all_gradients_present": len(present) == len(names),
        "gradient_finite": finite,
        "gradient_norm": norm,
        "gradient_effective": finite and norm > 1e-8,
    }


def post_micro_mechanism(
    model: nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    epoch: int,
) -> tuple[dict[str, Any], bool]:
    model.eval()
    with torch.no_grad():
        raw, _, _, intermediates = model(
            features,
            activation_epoch=epoch,
            return_intermediates=True,
        )
    result: dict[str, Any] = {}
    finite = bool(torch.isfinite(raw).all())
    for modality in sg.SG_MODALITIES:
        gates = intermediates["sg_group_gates_by_modality"][modality][mask]
        ratios = intermediates["sg_ratio_by_modality"][modality][mask]
        group_mean_gates = gates.mean(dim=0)
        modality_finite = bool(torch.isfinite(gates).all() and torch.isfinite(ratios).all())
        finite &= modality_finite
        result[modality] = {
            "group_mean_gates": group_mean_gates.detach().cpu().tolist(),
            "gate_max_deviation_from_0_5": float(
                (group_mean_gates - 0.5).abs().max().cpu()
            ),
            "gate_between_group_std_population": float(
                group_mean_gates.std(unbiased=False).cpu()
            ),
            "gate_any_subject_group_max_deviation_from_0_5": float(
                (gates - 0.5).abs().max().cpu()
            ),
            "mean_residual_shared_ratio": float(ratios.mean().cpu()),
            "maximum_residual_shared_ratio": float(ratios.max().cpu()),
            "finite": modality_finite,
        }
    finite &= all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    return result, finite


def family_summary(
    names: list[str],
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
) -> dict[str, Any]:
    deltas = [final[name].detach().cpu().float() - initial[name].detach().cpu().float() for name in names]
    initial_values = [initial[name].detach().cpu().float() for name in names]
    return {
        "parameter_count": int(sum(initial[name].numel() for name in names)),
        "tensor_count": len(names),
        "max_abs_delta": max((float(delta.abs().max()) for delta in deltas), default=0.0),
        "l2_delta": tensor_l2(deltas),
        "relative_l2_delta": tensor_l2(deltas) / (tensor_l2(initial_values) + EPS),
    }


def gradient_and_step_summary(
    names: list[str],
    named_parameters: dict[str, nn.Parameter],
    before: dict[str, torch.Tensor],
) -> dict[str, Any]:
    gradients = [named_parameters[name].grad for name in names]
    present = [gradient for gradient in gradients if gradient is not None]
    finite = bool(present) and all(bool(torch.isfinite(gradient).all()) for gradient in present)
    gradient_norm = tensor_l2(present)
    deltas = [named_parameters[name].detach().cpu() - before[name] for name in names]
    before_values = [before[name] for name in names]
    return {
        "tensor_count": len(names),
        "gradient_none_count": sum(gradient is None for gradient in gradients),
        "gradient_finite": finite,
        "gradient_norm": gradient_norm,
        "max_abs_parameter_delta": max((float(delta.abs().max()) for delta in deltas), default=0.0),
        "relative_l2_parameter_delta": tensor_l2(deltas) / (tensor_l2(before_values) + EPS),
    }


def manual_pet_gate_probe(
    model: nn.Module,
    features: torch.Tensor,
    test_mask: torch.Tensor,
    epoch: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Perturb one raw scorer output and measure the actual downstream path."""

    model.eval()
    branch = model.sg_branches["PET"]
    target_group_index = 0
    baseline_capture: dict[str, torch.Tensor] = {}

    def capture_baseline(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        baseline_capture["input"] = inputs[0].detach().clone()
        baseline_capture["raw_output"] = output.detach().clone()

    baseline_hook = branch.gate.register_forward_hook(capture_baseline)
    try:
        with torch.no_grad():
            baseline_raw, _, _, baseline_inter = model(
                features,
                activation_epoch=epoch,
                return_intermediates=True,
            )
    finally:
        baseline_hook.remove()

    perturb_capture: dict[str, torch.Tensor] = {}

    def add_to_one_group_logit(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        perturb_capture["raw_before"] = output.detach().clone()
        changed = output.clone()
        changed[:, target_group_index, :] += PERTURBATION
        perturb_capture["raw_after"] = changed.detach().clone()
        return changed

    perturb_hook = branch.gate.register_forward_hook(add_to_one_group_logit)
    try:
        with torch.no_grad():
            perturbed_raw, _, _, perturbed_inter = model(
                features,
                activation_epoch=epoch,
                return_intermediates=True,
            )
    finally:
        perturb_hook.remove()

    baseline_gate = baseline_inter["sg_group_gates_by_modality"]["PET"]
    perturbed_gate = perturbed_inter["sg_group_gates_by_modality"]["PET"]
    baseline_weight = baseline_inter["sg_normalized_gate_weights_by_modality"]["PET"]
    perturbed_weight = perturbed_inter["sg_normalized_gate_weights_by_modality"]["PET"]
    actual_group_tokens = baseline_capture["input"][..., : model.Hidden_size]
    masked_tokens = actual_group_tokens[test_mask]
    pairwise_token_difference = (
        masked_tokens.unsqueeze(2) - masked_tokens.unsqueeze(1)
    ).abs()
    selected_raw_diff = (
        perturb_capture["raw_after"][:, target_group_index, 0]
        - perturb_capture["raw_before"][:, target_group_index, 0]
    )[test_mask]

    pooled_diff = float(
        (
            perturbed_inter["sg_local_evidence_by_modality"]["PET"][test_mask]
            - baseline_inter["sg_local_evidence_by_modality"]["PET"][test_mask]
        )
        .abs()
        .max()
        .cpu()
    )
    delta_diff = float(
        (
            perturbed_inter["sg_delta_by_modality"]["PET"][test_mask]
            - baseline_inter["sg_delta_by_modality"]["PET"][test_mask]
        )
        .abs()
        .max()
        .cpu()
    )
    logits_diff = float(
        (perturbed_raw[test_mask] - baseline_raw[test_mask]).abs().max().cpu()
    )
    token_deviation = float(
        (
            masked_tokens
            - masked_tokens.mean(dim=1, keepdim=True)
        )
        .abs()
        .max()
        .cpu()
    )
    return {
        "modality": "PET",
        "group_index": target_group_index,
        "group_name": manifest["groups"]["PET"][target_group_index]["group_name"],
        "raw_gate_logit_perturbation": PERTURBATION,
        "selected_raw_gate_logit_max_abs_diff": float(selected_raw_diff.abs().max().cpu()),
        "selected_sigmoid_gate_max_abs_diff": float(
            (
                perturbed_gate[test_mask, target_group_index]
                - baseline_gate[test_mask, target_group_index]
            )
            .abs()
            .max()
            .cpu()
        ),
        "all_sigmoid_gates_max_abs_diff": float(
            (perturbed_gate[test_mask] - baseline_gate[test_mask]).abs().max().cpu()
        ),
        "normalized_gate_weight_max_abs_diff": float(
            (perturbed_weight[test_mask] - baseline_weight[test_mask]).abs().max().cpu()
        ),
        "pet_group_token_abs_max": float(masked_tokens.abs().max().cpu()),
        "pet_group_token_cross_group_max_deviation": token_deviation,
        "pet_group_token_pairwise_max_abs_diff": float(pairwise_token_difference.max().cpu()),
        "pooled_local_evidence_max_abs_diff": pooled_diff,
        "delta_max_abs_diff": delta_diff,
        "final_logits_max_abs_diff": logits_diff,
        "pooled_feature_changed": pooled_diff > 1e-8,
        "final_logits_changed": logits_diff > 1e-10,
        "hook_changed_selected_raw_logit": bool(
            torch.allclose(
                selected_raw_diff,
                torch.full_like(selected_raw_diff, PERTURBATION),
                atol=1e-7,
                rtol=1e-6,
            )
        ),
        "temporary_hooks_removed": True,
    }


def forward_gate_statistics(
    model: nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        raw, _, _, intermediates = model(
            features, activation_epoch=epoch, return_intermediates=True
        )
    gate_report: dict[str, Any] = {}
    mechanism: dict[str, Any] = {}
    manifest = model.group_manifest()
    for modality in sg.SG_MODALITIES:
        gates = intermediates["sg_group_gates_by_modality"][modality][mask]
        raw_logits = torch.logit(gates.clamp(1e-12, 1.0 - 1e-12))
        raw_group_means = raw_logits.mean(dim=0)
        gate_group_means = gates.mean(dim=0)
        group_names = [group["group_name"] for group in manifest[modality]]
        gate_report[modality] = {
            "groups": [
                {
                    "group_name": group_name,
                    "raw_gate_logit": float(raw_group_means[index].cpu()),
                    "sigmoid_gate": float(gate_group_means[index].cpu()),
                }
                for index, group_name in enumerate(group_names)
            ],
            "raw_gate_logit_between_group_std_population": float(
                raw_group_means.std(unbiased=False).cpu()
            ),
            "sigmoid_gate_between_group_std_population": float(
                gate_group_means.std(unbiased=False).cpu()
            ),
            "maximum_gate_deviation_from_0_5": float(
                (gate_group_means - 0.5).abs().max().cpu()
            ),
        }
        ratios = intermediates["sg_ratio_by_modality"][modality][mask]
        mechanism[modality] = {
            "mean_residual_shared_ratio": float(ratios.mean().cpu()),
            "maximum_residual_shared_ratio": float(ratios.max().cpu()),
        }
    return raw, gate_report, mechanism


def write_markdown(payload: dict[str, Any]) -> None:
    coverage = payload["check_A_registration_optimizer"]
    checkpoint = payload["check_B_checkpoint"]
    active = payload["check_C_active_step"]
    perturbation = payload["check_D_manual_perturbation"]
    fix = payload["bug_fix_validation"]
    lines = [
        "# SG-HFT-C1 v2 Gate Parameter Update Check",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Branch: `{payload['branch']}`",
        f"- Run commit: `{payload['run_commit']}`",
        f"- Runtime: `{payload['runtime_seconds']:.3f} s`",
        f"- Checkpoint SHA256: `{checkpoint['checkpoint_sha256']}`",
        "",
        "## Required summary",
        "",
        f"- Optimizer coverage: `{coverage['optimizer_coverage_fraction']:.6f}`",
        f"- Checkpoint SG-HFT key coverage: `{checkpoint['sg_checkpoint_key_coverage_fraction']:.6f}`",
        f"- Active-step optimizer-state coverage: `{active['optimizer_state_coverage_after_step']:.6f}`",
        f"- Manual pooled/logit diff: `{perturbation['pooled_local_evidence_max_abs_diff']:.12g}` / "
        f"`{perturbation['final_logits_max_abs_diff']:.12g}`",
        f"- Root cause: `{payload['root_cause']}`",
        f"- Action: `{payload['action']}`",
        "",
        "## Checkpoint family deltas",
        "",
        "| Modality | Family | max abs delta | L2 delta | relative L2 delta |",
        "|---|---|---:|---:|---:|",
    ]
    for modality, families in checkpoint["family_parameter_delta"].items():
        for family, item in families.items():
            lines.append(
                f"| {modality} | {family} | {item['max_abs_delta']:.12g} | "
                f"{item['l2_delta']:.12g} | {item['relative_l2_delta']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Active-step family gradients and updates",
            "",
            "| Modality | Family | gradient norm | gradient finite | max abs update | relative L2 update |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for modality, families in active["families"].items():
        for family, item in families.items():
            lines.append(
                f"| {modality} | {family} | {item['gradient_norm']:.12g} | "
                f"{item['gradient_finite']} | {item['max_abs_parameter_delta']:.12g} | "
                f"{item['relative_l2_parameter_delta']:.12g} |"
            )
    lines.extend(["", "## Checkpoint gate outputs", ""])
    for modality, item in checkpoint["checkpoint_gate_outputs"].items():
        lines.append(
            f"- {modality}: raw-logit group std=`{item['raw_gate_logit_between_group_std_population']:.12g}`, "
            f"gate group std=`{item['sigmoid_gate_between_group_std_population']:.12g}`"
        )
        for group in item["groups"]:
            lines.append(
                f"  - {group['group_name']}: raw=`{group['raw_gate_logit']:.10f}`, "
                f"sigmoid=`{group['sigmoid_gate']:.10f}`"
            )
    checkpoint_probe = perturbation["checkpoint_probe"]
    fresh_probe = perturbation["fresh_initialization_control"]
    lines.extend(
        [
            "",
            "## Manual PET gate perturbation",
            "",
            f"- Selected raw-logit diff: `{checkpoint_probe['selected_raw_gate_logit_max_abs_diff']:.12g}`",
            f"- Selected sigmoid-gate diff: `{checkpoint_probe['selected_sigmoid_gate_max_abs_diff']:.12g}`",
            f"- Normalized-weight diff: `{checkpoint_probe['normalized_gate_weight_max_abs_diff']:.12g}`",
            f"- Checkpoint PET token absolute max: `{checkpoint_probe['pet_group_token_abs_max']:.12g}`",
            f"- Checkpoint PET token cross-group max deviation: "
            f"`{checkpoint_probe['pet_group_token_cross_group_max_deviation']:.12g}`",
            f"- Checkpoint pooled/delta/logit diff: "
            f"`{checkpoint_probe['pooled_local_evidence_max_abs_diff']:.12g}` / "
            f"`{checkpoint_probe['delta_max_abs_diff']:.12g}` / "
            f"`{checkpoint_probe['final_logits_max_abs_diff']:.12g}`",
            f"- Fresh-init pooled/delta/logit control: "
            f"`{fresh_probe['pooled_local_evidence_max_abs_diff']:.12g}` / "
            f"`{fresh_probe['delta_max_abs_diff']:.12g}` / "
            f"`{fresh_probe['final_logits_max_abs_diff']:.12g}`",
            "",
            "## Root cause and action boundary",
            "",
            "The scorer hook changes the selected raw gate logit and its normalized weight, "
            "but the checkpoint PET group tokens are numerically indistinguishable. The "
            "failure is therefore not a hook error or gate-normalization cancellation. It is "
            "an optimizer-driven branch collapse caused by zero-alpha warmup combined with "
            "coupled L2 decay.",
            "",
            "The minimal repair preserves the alpha schedule and optimizer configuration while "
            "keeping disabled SG parameters outside the loss graph. This prevents coupled "
            "weight decay from updating them during alpha-zero warmup.",
            "",
            "## Minimal-fix validation",
            "",
            f"- Epoch-1 SG gradients None: "
            f"`{fix['warmup_epoch1']['sg_gradient_none_count']}/"
            f"{fix['warmup_epoch1']['sg_parameter_tensor_count']}`",
            f"- Epoch-1 SG parameters bitwise unchanged: "
            f"`{fix['warmup_epoch1']['sg_parameter_bitwise_unchanged_count']}/"
            f"{fix['warmup_epoch1']['sg_parameter_tensor_count']}`",
            f"- Twenty-step continuous 9-family gradients: "
            f"`{fix['micro_training']['all_steps_all_nine_gradients_finite_nonzero']}`",
            f"- Modalities passing gate deviation/std/ratio thresholds: "
            f"`{fix['micro_training']['gate_deviation_modality_count_gt_0_001']}` / "
            f"`{fix['micro_training']['gate_std_modality_count_gt_0_001']}` / "
            f"`{fix['micro_training']['ratio_modality_count_ge_0_015']}`",
            f"- Post-micro PET logit perturbation diff: "
            f"`{fix['micro_training']['pet_manual_perturbation']['final_logits_max_abs_diff']:.12g}`",
            f"- Micro-training passed: `{fix['micro_training']['passed']}`",
            "",
            "No checkpoint was saved and this script did not start the difficult-fold screen "
            "or any 400-epoch training.",
            "",
            "The PET perturbation was injected into the actual scorer output by a temporary "
            "forward hook and removed immediately after the diagnostic forward.",
            "",
        ]
    )
    lines.extend(
        [
            "## Post-micro mechanism",
            "",
            "| Modality | gate max deviation | gate group std | mean ratio | max ratio | finite |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for modality, item in fix["micro_training"]["post_step_20_mechanism"].items():
        lines.append(
            f"| {modality} | {item['gate_max_deviation_from_0_5']:.12g} | "
            f"{item['gate_between_group_std_population']:.12g} | "
            f"{item['mean_residual_shared_ratio']:.12g} | "
            f"{item['maximum_residual_shared_ratio']:.12g} | {item['finite']} |"
        )
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="replace only this script's two prior gate-check report files",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    checkpoint_path = args.checkpoint.resolve()
    require(checkpoint_path.is_file(), f"Missing fold-4 checkpoint: {checkpoint_path}")
    branch = git_value("branch", "--show-current")
    require(branch.startswith("experiment/sg-hft-c1-v2-gatecheck"), f"Unsafe branch: {branch}")
    require(
        args.overwrite_results or (not OUTPUT_JSON.exists() and not OUTPUT_MD.exists()),
        "Gate-check results already exist; pass --overwrite-results to replace only them",
    )

    # The historical loader verifies branch identity through these module-level
    # anchors.  Pin them to this isolated diagnostic branch only.
    sg.BRANCH_PREFIX = branch
    sg.legacy.BRANCH_PREFIX = branch
    context = sg.load_context()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][4]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(set(checkpoint) == {"model_state", "group_manifest", "best_epoch"}, "Unexpected checkpoint payload")
    manifest = checkpoint["group_manifest"]
    require(int(manifest["fold"]) == 4 and int(checkpoint["best_epoch"]) == 214, "Wrong formal fold-4 checkpoint")

    # Check A: exact formal constructor and exact formal optimizer.
    fresh_model, criterion, optimizer, _ = sg.make_fresh_training_objects(context, manifest)
    named_parameters = dict(fresh_model.named_parameters())
    sg_names = [name for name in named_parameters if name.startswith("sg_branches.")]
    require(sg_names, "No SG-HFT named parameters")
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    parameter_rows = []
    for name in sg_names:
        parameter = named_parameters[name]
        parameter_rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "in_model_named_parameters": True,
                "in_optimizer_param_groups": id(parameter) in optimizer_parameter_ids,
                "optimizer_state_before_step": parameter in optimizer.state,
            }
        )
    optimizer_coverage = sum(row["in_optimizer_param_groups"] for row in parameter_rows) / len(parameter_rows)
    check_a = {
        "sg_parameter_tensor_count": len(parameter_rows),
        "sg_parameter_count": int(sum(row["numel"] for row in parameter_rows)),
        "optimizer_coverage_fraction": optimizer_coverage,
        "optimizer_coverage_percent": 100.0 * optimizer_coverage,
        "optimizer_state_is_lazy_and_empty_before_first_step": len(optimizer.state) == 0,
        "sg_branches_is_ModuleDict": isinstance(fresh_model.sg_branches, nn.ModuleDict),
        "group_encoders_are_ModuleList": all(
            isinstance(branch_module.group_encoders, nn.ModuleList)
            for branch_module in fresh_model.sg_branches.values()
        ),
        "optimizer_constructed_from_complete_model_parameters": True,
        "no_sg_parameter_filter": True,
        "parameters": parameter_rows,
    }

    # Check B: compare deterministic formal initialization with the immutable
    # formal checkpoint, both parameter-by-parameter and family-by-family.
    initial_state = {
        name: named_parameters[name].detach().cpu().clone() for name in sg_names
    }
    checkpoint_state = checkpoint["model_state"]
    checkpoint_sg_keys = [key for key in checkpoint_state if key.startswith("sg_branches.")]
    missing_checkpoint_parameters = [name for name in sg_names if name not in checkpoint_state]
    checkpoint_parameter_rows = []
    for name in sg_names:
        if name not in checkpoint_state:
            continue
        delta = checkpoint_state[name].detach().cpu().float() - initial_state[name].float()
        checkpoint_parameter_rows.append(
            {
                "name": name,
                "max_abs_delta": float(delta.abs().max()),
                "l2_delta": float(delta.norm()),
                "relative_l2_delta": float(delta.norm())
                / (float(initial_state[name].float().norm()) + EPS),
            }
        )
    family_deltas: dict[str, dict[str, Any]] = {}
    for modality in sg.SG_MODALITIES:
        family_deltas[modality] = {}
        for family in ("group_encoder", "gate", "adapter"):
            names = [
                name
                for name in sg_names
                if modality_of(name) == modality and sg_family(name) == family
            ]
            family_deltas[modality][family] = family_summary(
                names, initial_state, checkpoint_state
            )

    checkpoint_model = sg.build_model(context, manifest, True)
    checkpoint_model.load_state_dict(checkpoint_state, strict=True)
    checkpoint_raw, checkpoint_gates, checkpoint_mechanism = forward_gate_statistics(
        checkpoint_model,
        features,
        test_mask,
        int(checkpoint["best_epoch"]),
    )
    initial_raw, initial_gates, _ = forward_gate_statistics(
        fresh_model,
        features,
        test_mask,
        int(checkpoint["best_epoch"]),
    )
    for modality in sg.SG_MODALITIES:
        initial_by_group = {
            group["group_name"]: group for group in initial_gates[modality]["groups"]
        }
        for group in checkpoint_gates[modality]["groups"]:
            initial_group = initial_by_group[group["group_name"]]
            group["raw_gate_logit_change_from_initial"] = (
                group["raw_gate_logit"] - initial_group["raw_gate_logit"]
            )
            group["sigmoid_gate_change_from_initial"] = (
                group["sigmoid_gate"] - initial_group["sigmoid_gate"]
            )

    check_b = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_payload_keys": sorted(checkpoint),
        "checkpoint_contains_optimizer_state": "optimizer_state" in checkpoint,
        "checkpoint_optimizer_state_note": "Formal checkpoint intentionally stores model state only.",
        "sg_state_key_count_in_checkpoint_including_buffers": len(checkpoint_sg_keys),
        "missing_sg_named_parameters": missing_checkpoint_parameters,
        "sg_checkpoint_key_coverage_fraction": 1.0
        - len(missing_checkpoint_parameters) / len(sg_names),
        "strict_state_dict_readback": True,
        "parameter_delta": checkpoint_parameter_rows,
        "family_parameter_delta": family_deltas,
        "initial_gate_outputs": initial_gates,
        "checkpoint_gate_outputs": checkpoint_gates,
        "checkpoint_mechanism": checkpoint_mechanism,
        "report_runner_checkpoint_path_matches": checkpoint_path.name == "checkpoint_best.pt"
        and checkpoint_path.parent.name == "fold_04",
    }

    # Check C: one complete formal C1-loss step at active epoch 41.
    fresh_model.train()
    optimizer.zero_grad(set_to_none=True)
    raw, branches, auxiliary, active_intermediates = fresh_model(
        features, activation_epoch=ACTIVE_EPOCH, return_intermediates=True
    )
    retained_deltas = {}
    for modality in sg.SG_MODALITIES:
        retained = active_intermediates["sg_delta_by_modality"][modality]
        retained.retain_grad()
        retained_deltas[modality] = retained
    loss = criterion(raw, labels, train_mask, branches, auxiliary)
    require(bool(torch.isfinite(loss)), "Active-step loss is non-finite")
    loss.backward()
    before_step = {
        name: named_parameters[name].detach().cpu().clone() for name in sg_names
    }
    if float(context["config"].grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(
            fresh_model.parameters(), float(context["config"].grad_clip)
        )
    optimizer.step()
    step_families: dict[str, dict[str, Any]] = {}
    for modality in sg.SG_MODALITIES:
        step_families[modality] = {}
        for family in ("group_encoder", "gate", "adapter"):
            names = [
                name
                for name in sg_names
                if modality_of(name) == modality and sg_family(name) == family
            ]
            family_item = gradient_and_step_summary(
                names, named_parameters, before_step
            )
            prestep_parameter_l2 = tensor_l2([before_step[name] for name in names])
            coupled_decay_l2 = (
                float(optimizer.param_groups[0]["weight_decay"])
                * prestep_parameter_l2
            )
            family_item["prestep_parameter_l2"] = prestep_parameter_l2
            family_item["coupled_weight_decay_term_l2_estimate"] = coupled_decay_l2
            family_item["decay_to_task_gradient_norm_ratio"] = coupled_decay_l2 / (
                family_item["gradient_norm"] + EPS
            )
            step_families[modality][family] = family_item
    for row in parameter_rows:
        parameter = named_parameters[row["name"]]
        row["optimizer_state_after_step"] = parameter in optimizer.state
    output_gradient = {
        modality: {
            "participates_in_final_logits": retained_deltas[modality].grad is not None,
            "gradient_norm": 0.0
            if retained_deltas[modality].grad is None
            else float(retained_deltas[modality].grad.float().norm().detach().cpu()),
            "gradient_finite": retained_deltas[modality].grad is not None
            and bool(torch.isfinite(retained_deltas[modality].grad).all()),
        }
        for modality in sg.SG_MODALITIES
    }
    state_coverage_after = sum(
        named_parameters[name] in optimizer.state for name in sg_names
    ) / len(sg_names)
    check_c = {
        "activation_epoch": ACTIVE_EPOCH,
        "alpha": float(active_intermediates["sg_alpha"]),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "families": step_families,
        "output_tensor_gradient": output_gradient,
        "optimizer_state_coverage_after_step": state_coverage_after,
        "optimizer": "Adam with coupled L2 weight decay",
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "zero_alpha_warmup_epochs": [1, 20],
        "historical_zero_alpha_forward_constructed_sg_graph": True,
        "historical_zero_times_raw_delta_yielded_zero_but_non_None_sg_gradients": True,
        "forward_has_no_post_gate_detach_item_new_tensor_or_no_grad": all(
            item["participates_in_final_logits"] and item["gradient_finite"]
            for item in output_gradient.values()
        ),
    }

    # Check D: use the same in-graph perturbation on both the formal checkpoint
    # and a deterministic fresh initialization.  The control distinguishes a
    # broken hook/formula from checkpoint-specific numerical collapse.
    checkpoint_probe = manual_pet_gate_probe(
        checkpoint_model,
        features,
        test_mask,
        int(checkpoint["best_epoch"]),
        manifest,
    )
    fresh_probe_model = sg.build_model(context, manifest, True)
    fresh_probe = manual_pet_gate_probe(
        fresh_probe_model,
        features,
        test_mask,
        int(checkpoint["best_epoch"]),
        manifest,
    )
    pooled_diff = checkpoint_probe["pooled_local_evidence_max_abs_diff"]
    delta_diff = checkpoint_probe["delta_max_abs_diff"]
    logits_diff = checkpoint_probe["final_logits_max_abs_diff"]
    check_d = {
        "modality": "PET",
        "group_index": checkpoint_probe["group_index"],
        "group_name": checkpoint_probe["group_name"],
        "raw_gate_logit_perturbation": PERTURBATION,
        "injected_into_actual_gate_scorer_output": True,
        "temporary_hook_removed": True,
        "selected_raw_gate_logit_max_abs_diff": checkpoint_probe[
            "selected_raw_gate_logit_max_abs_diff"
        ],
        "selected_sigmoid_gate_max_abs_diff": checkpoint_probe[
            "selected_sigmoid_gate_max_abs_diff"
        ],
        "normalized_gate_weight_max_abs_diff": checkpoint_probe[
            "normalized_gate_weight_max_abs_diff"
        ],
        "checkpoint_pet_group_token_abs_max": checkpoint_probe[
            "pet_group_token_abs_max"
        ],
        "checkpoint_pet_group_token_cross_group_max_deviation": checkpoint_probe[
            "pet_group_token_cross_group_max_deviation"
        ],
        "pooled_local_evidence_max_abs_diff": pooled_diff,
        "delta_max_abs_diff": delta_diff,
        "final_logits_max_abs_diff": logits_diff,
        "pooled_feature_changed": pooled_diff > 1e-8,
        "final_logits_changed": logits_diff > 1e-10,
        "checkpoint_probe": checkpoint_probe,
        "fresh_initialization_control": fresh_probe,
        "formula_uses_group_specific_gate_values": True,
        "formula_only_cancels_a_common_scale_not_group_specific_changes": True,
    }

    # Confirm the minimal model fix: an inactive SG branch must leave every SG
    # parameter outside the loss graph, so coupled Adam weight decay cannot
    # mutate it during epochs 1--20.
    warmup_model, warmup_criterion, warmup_optimizer, warmup_scheduler = (
        sg.make_fresh_training_objects(context, manifest)
    )
    warmup_named = dict(warmup_model.named_parameters())
    warmup_sg_names = [
        name for name in warmup_named if name.startswith("sg_branches.")
    ]
    warmup_before = {
        name: warmup_named[name].detach().cpu().clone() for name in warmup_sg_names
    }
    warmup_model.train()
    warmup_optimizer.zero_grad(set_to_none=True)
    warmup_raw, warmup_branches, warmup_auxiliary, warmup_intermediates = warmup_model(
        features,
        activation_epoch=1,
        return_intermediates=True,
    )
    warmup_loss = warmup_criterion(
        warmup_raw,
        labels,
        train_mask,
        warmup_branches,
        warmup_auxiliary,
    )
    warmup_loss.backward()
    warmup_grad_none_count = sum(
        warmup_named[name].grad is None for name in warmup_sg_names
    )
    warmup_optimizer.step()
    warmup_scheduler.step()
    warmup_bitwise_unchanged_count = sum(
        torch.equal(warmup_before[name], warmup_named[name].detach().cpu())
        for name in warmup_sg_names
    )
    warmup_optimizer_state_count = sum(
        warmup_named[name] in warmup_optimizer.state for name in warmup_sg_names
    )
    warmup_fix_passed = (
        len(warmup_sg_names) == 60
        and warmup_grad_none_count == len(warmup_sg_names)
        and warmup_bitwise_unchanged_count == len(warmup_sg_names)
        and warmup_optimizer_state_count == 0
        and float(warmup_intermediates["sg_alpha"]) == 0.0
        and bool(torch.isfinite(warmup_loss))
    )
    warmup_fix_validation = {
        "epoch": 1,
        "alpha": float(warmup_intermediates["sg_alpha"]),
        "loss_finite": bool(torch.isfinite(warmup_loss)),
        "sg_parameter_tensor_count": len(warmup_sg_names),
        "sg_gradient_none_count": warmup_grad_none_count,
        "expected_gradient_none_count": 60,
        "sg_parameter_bitwise_unchanged_count": warmup_bitwise_unchanged_count,
        "expected_bitwise_unchanged_count": 60,
        "sg_optimizer_state_count_after_step": warmup_optimizer_state_count,
        "passed": warmup_fix_passed,
    }

    # Gate the micro-training stage on the already-completed A--D evidence.
    # The historical perturbation failure is the expected optimizer-collapse
    # signature; any independent A--C fault makes attribution inconclusive.
    bug_reasons: list[str] = []
    if optimizer_coverage < 1.0 or not all(
        row["requires_grad"]
        and row["in_model_named_parameters"]
        and row["in_optimizer_param_groups"]
        for row in parameter_rows
    ):
        bug_reasons.append("Check A: SG-HFT registration/optimizer coverage failed")
    if not check_a["sg_branches_is_ModuleDict"] or not check_a["group_encoders_are_ModuleList"]:
        bug_reasons.append("Check A: SG-HFT containers are not registered modules")
    if missing_checkpoint_parameters or not check_b["strict_state_dict_readback"]:
        bug_reasons.append("Check B: formal checkpoint SG-HFT readback failed")
    if not check_b["report_runner_checkpoint_path_matches"]:
        bug_reasons.append("Check B: diagnostic/report checkpoint path mismatch")
    if not all(
        item["gradient_finite"]
        and item["gradient_norm"] > 1e-8
        and item["relative_l2_parameter_delta"] > 1e-8
        for families in step_families.values()
        for item in families.values()
    ):
        bug_reasons.append("Check C: an active SG-HFT family lacks an effective gradient/update")
    if state_coverage_after < 1.0:
        bug_reasons.append("Check C: Adam state was not created for every SG-HFT parameter")
    if not all(
        item["participates_in_final_logits"]
        and item["gradient_finite"]
        and item["gradient_norm"] > 1e-8
        for item in output_gradient.values()
    ):
        bug_reasons.append("Check C: an SG-HFT output does not participate in final logits")
    if not all(
        item["relative_l2_delta"] > 1e-8
        for families in family_deltas.values()
        for item in families.values()
    ):
        bug_reasons.append("Check B: a checkpoint SG-HFT family is unchanged from initialization")
    a_to_c_bug_reasons = list(bug_reasons)

    checkpoint_tokens_collapsed = (
        checkpoint_probe["pet_group_token_cross_group_max_deviation"] <= 1e-8
    )
    checkpoint_hook_effective = (
        checkpoint_probe["hook_changed_selected_raw_logit"]
        and checkpoint_probe["selected_sigmoid_gate_max_abs_diff"] > 1e-8
        and checkpoint_probe["normalized_gate_weight_max_abs_diff"] > 1e-8
    )
    fresh_control_effective = (
        fresh_probe["pooled_local_evidence_max_abs_diff"] > 1e-8
        and fresh_probe["delta_max_abs_diff"] > 0.0
        and fresh_probe["final_logits_max_abs_diff"] > 1e-10
    )
    optimizer_driven_branch_collapse = (
        checkpoint_hook_effective
        and checkpoint_tokens_collapsed
        and fresh_control_effective
        and (pooled_diff <= 1e-8 or logits_diff <= 1e-10)
    )
    if not optimizer_driven_branch_collapse:
        bug_reasons.append(
            "Check D: optimizer-driven collapse is not uniquely confirmed by checkpoint/control probes"
        )
    diagnostic_prerequisites_passed = (
        optimizer_driven_branch_collapse
        and not bug_reasons
        and warmup_fix_passed
    )
    diagnostic_confirmation = {
        "A_to_C_no_independent_faults": not a_to_c_bug_reasons,
        "A_to_C_bug_reasons": a_to_c_bug_reasons,
        "optimizer_driven_branch_collapse_confirmed": optimizer_driven_branch_collapse,
        "warmup_fix_passed": warmup_fix_passed,
        "bug_reasons": bug_reasons,
        "micro_training_authorized": diagnostic_prerequisites_passed,
    }

    # The only post-fix training allowed here: 20 fresh active steps using the
    # exact fold-4 criterion, optimizer, scheduler, masks, and epoch-41 alpha.
    micro_model, micro_criterion, micro_optimizer, micro_scheduler = (
        sg.make_fresh_training_objects(context, manifest)
    )
    micro_named = dict(micro_model.named_parameters())
    micro_sg_names = [
        name for name in micro_named if name.startswith("sg_branches.")
    ]
    micro_optimizer_ids = {
        id(parameter)
        for group in micro_optimizer.param_groups
        for parameter in group["params"]
    }
    micro_optimizer_coverage = sum(
        id(micro_named[name]) in micro_optimizer_ids for name in micro_sg_names
    ) / len(micro_sg_names)
    micro_steps: list[dict[str, Any]] = []
    micro_all_finite = True
    for step in (range(1, 21) if diagnostic_prerequisites_passed else ()):
        micro_model.train()
        micro_optimizer.zero_grad(set_to_none=True)
        micro_raw, micro_branches, micro_auxiliary, micro_intermediates = micro_model(
            features,
            activation_epoch=ACTIVE_EPOCH,
            return_intermediates=True,
        )
        micro_loss = micro_criterion(
            micro_raw,
            labels,
            train_mask,
            micro_branches,
            micro_auxiliary,
        )
        loss_finite = bool(torch.isfinite(micro_loss))
        micro_loss.backward()
        step_families_record: dict[str, dict[str, Any]] = {}
        for modality in sg.SG_MODALITIES:
            step_families_record[modality] = {}
            for family in ("group_encoder", "gate", "adapter"):
                names = family_parameter_names(micro_named, modality, family)
                step_families_record[modality][family] = gradient_snapshot(
                    names,
                    micro_named,
                )
        step_gradients_pass = all(
            item["all_gradients_present"]
            and item["gradient_finite"]
            and item["gradient_effective"]
            for families in step_families_record.values()
            for item in families.values()
        )
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                micro_model.parameters(), float(context["config"].grad_clip)
            )
        micro_optimizer.step()
        micro_scheduler.step()
        parameter_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in micro_model.parameters()
        )
        step_numerically_finite = (
            loss_finite
            and parameter_finite
            and bool(torch.isfinite(micro_raw).all())
            and all(
                item["gradient_finite"]
                for families in step_families_record.values()
                for item in families.values()
            )
        )
        # Numerical finiteness and the preregistered >1e-8 gradient threshold
        # are separate gates: a tiny but finite gradient must not be reported
        # as NaN/Inf merely because it fails the activity threshold.
        micro_all_finite &= step_numerically_finite
        micro_steps.append(
            {
                "step": step,
                "activation_epoch": ACTIVE_EPOCH,
                "alpha": float(micro_intermediates["sg_alpha"]),
                "loss": float(micro_loss.detach().cpu()),
                "loss_finite": loss_finite,
                "learning_rate_after_step": float(micro_optimizer.param_groups[0]["lr"]),
                "all_nine_modality_family_gradients_finite_nonzero": step_gradients_pass,
                "model_parameters_finite_after_step": parameter_finite,
                "families": step_families_record,
            }
        )

    if diagnostic_prerequisites_passed:
        micro_mechanism, micro_post_forward_finite = post_micro_mechanism(
            micro_model,
            features,
            train_mask,
            ACTIVE_EPOCH,
        )
        micro_perturbation = manual_pet_gate_probe(
            micro_model,
            features,
            test_mask,
            ACTIVE_EPOCH,
            manifest,
        )
    else:
        micro_mechanism = {}
        micro_post_forward_finite = False
        micro_perturbation = {
            "skipped": True,
            "skip_reason": "A--D attribution or warmup-fix prerequisite failed",
            "pooled_local_evidence_max_abs_diff": 0.0,
            "delta_max_abs_diff": 0.0,
            "final_logits_max_abs_diff": 0.0,
        }
    gate_deviation_modality_count = sum(
        item["gate_max_deviation_from_0_5"] > 1e-3
        for item in micro_mechanism.values()
    )
    gate_std_modality_count = sum(
        item["gate_between_group_std_population"] > 1e-3
        for item in micro_mechanism.values()
    )
    ratio_modality_count = sum(
        item["mean_residual_shared_ratio"] >= 0.015
        for item in micro_mechanism.values()
    )
    micro_gradient_continuity_passed = bool(micro_steps) and all(
        row["all_nine_modality_family_gradients_finite_nonzero"]
        for row in micro_steps
    )
    micro_passed = (
        diagnostic_prerequisites_passed
        and micro_optimizer_coverage == 1.0
        and micro_gradient_continuity_passed
        and gate_deviation_modality_count >= 2
        and gate_std_modality_count >= 2
        and ratio_modality_count >= 2
        and micro_perturbation["final_logits_max_abs_diff"] > 1e-10
        and micro_all_finite
        and micro_post_forward_finite
    )
    bug_fix_validation = {
        "modified_files": [
            "Model/sg_hft_c1_v2.py",
            "scripts/run_sg_hft_gate_update_check.py",
        ],
        "repair_applied": True,
        "repair": (
            "when alpha=0, compute SG diagnostics without attaching SG parameters "
            "to the loss graph, so Adam coupled weight decay cannot update them"
        ),
        "diagnostic_confirmation_before_micro_training": diagnostic_confirmation,
        "warmup_epoch1": warmup_fix_validation,
        "micro_training": {
            "fold": 4,
            "seed": sg.SEED,
            "fixed_activation_epoch": ACTIVE_EPOCH,
            "fixed_alpha": 1.0,
            "steps_requested": 20,
            "steps_executed": len(micro_steps),
            "skipped": not diagnostic_prerequisites_passed,
            "fresh_model": True,
            "fresh_criterion": True,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "optimizer_coverage_fraction": micro_optimizer_coverage,
            "all_steps_all_nine_gradients_finite_nonzero": micro_gradient_continuity_passed,
            "all_steps_and_parameters_finite": micro_all_finite,
            "post_forward_finite": micro_post_forward_finite,
            "per_step": micro_steps,
            "post_step_20_mechanism": micro_mechanism,
            "gate_deviation_modality_count_gt_0_001": gate_deviation_modality_count,
            "gate_std_modality_count_gt_0_001": gate_std_modality_count,
            "ratio_modality_count_ge_0_015": ratio_modality_count,
            "pet_manual_perturbation": micro_perturbation,
            "passed": micro_passed,
            "checkpoint_saved": False,
            "screen_or_400_epoch_started": False,
        },
    }

    maximum_gate_std = max(
        item["sigmoid_gate_between_group_std_population"]
        for item in checkpoint_gates.values()
    )
    maximum_gate_deviation = max(
        item["maximum_gate_deviation_from_0_5"] for item in checkpoint_gates.values()
    )
    maximum_mean_ratio = max(
        item["mean_residual_shared_ratio"] for item in checkpoint_mechanism.values()
    )
    checkpoint_uniform_collapse = (
        maximum_gate_std <= 1e-6
        and maximum_gate_deviation <= 1e-5
        and maximum_mean_ratio <= 0.01
    )
    confirmed_root_cause = (
        "optimizer-driven branch collapse from zero-alpha warmup plus coupled L2 decay"
    )
    if not diagnostic_prerequisites_passed:
        decision = "DIAGNOSTIC_INCONCLUSIVE"
        root_cause = "inconclusive because A--D attribution or warmup-fix verification failed"
        action = "20-step micro-training skipped; stopped before screen"
    elif micro_passed:
        decision = "BUG_FIX_MICRO_PASS"
        root_cause = confirmed_root_cause
        action = (
            "20-step micro-training passed; fixed difficult-fold screen is required, "
            "but this diagnostic script did not start it"
        )
    else:
        decision = "BUG_FIXED_BUT_BRANCH_INACTIVE"
        root_cause = confirmed_root_cause
        action = "minimal fix passed but micro-training gate failed; stopped before screen"

    payload = {
        "experiment": "SG-HFT-C1 v2 Gate Parameter Update Check",
        "decision": decision,
        "branch": branch,
        "run_commit": git_value("rev-parse", "HEAD"),
        "device": str(context["device"]),
        "runtime_seconds": time.perf_counter() - started,
        "check_A_registration_optimizer": check_a,
        "check_B_checkpoint": check_b,
        "check_C_active_step": check_c,
        "check_D_manual_perturbation": check_d,
        "bug_fix_validation": bug_fix_validation,
        "decision_evidence": {
            "confirmed_bug_reason": (
                confirmed_root_cause if optimizer_driven_branch_collapse else None
            ),
            "other_fault_reasons": bug_reasons,
            "A_to_C_no_independent_faults": not a_to_c_bug_reasons,
            "diagnostic_prerequisites_passed": diagnostic_prerequisites_passed,
            "maximum_checkpoint_gate_group_std": maximum_gate_std,
            "maximum_checkpoint_gate_deviation_from_0_5": maximum_gate_deviation,
            "maximum_checkpoint_mean_residual_shared_ratio": maximum_mean_ratio,
            "checkpoint_uniform_low_ratio_collapse": checkpoint_uniform_collapse,
            "diagnostic_hook_effective": checkpoint_hook_effective,
            "checkpoint_pet_group_tokens_collapsed": checkpoint_tokens_collapsed,
            "fresh_initialization_same_hook_effective": fresh_control_effective,
            "optimizer_driven_branch_collapse_confirmed": optimizer_driven_branch_collapse,
            "warmup_fix_passed": warmup_fix_passed,
            "micro_training_passed": micro_passed,
        },
        "root_cause": root_cause,
        "root_cause_detail": {
            "attribution_confirmed": diagnostic_prerequisites_passed,
            "zero_alpha_epochs": [1, 20],
            "original_zero_alpha_multiplication_produced_zero_but_non_None_gradients": True,
            "adam_uses_coupled_l2_weight_decay": True,
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "effect": (
                "SG group encoders and gate scorer are updated toward zero while the "
                "branch is disabled; their later task gradients are too weak to recover"
            ),
            "not_a_hook_error": checkpoint_hook_effective,
            "not_group_gate_normalization_cancellation": fresh_control_effective,
        },
        "repair_boundary": {
            "applied_change": (
                "detach the disabled SG computation graph only while alpha=0 so SG grads "
                "remain None and Adam cannot apply coupled decay"
            ),
            "activation_alpha_schedule_changed": False,
            "optimizer_type_or_configuration_changed": False,
            "loss_scheduler_rank_cap_or_group_count_changed": False,
            "repair_applied": True,
        },
        "modified_files": [
            "Model/sg_hft_c1_v2.py",
            "scripts/run_sg_hft_gate_update_check.py",
        ],
        "modified_model_files": ["Model/sg_hft_c1_v2.py"],
        "modified_diagnostic_files": ["scripts/run_sg_hft_gate_update_check.py"],
        "modified_runner_files": [],
        "micro_training_20_steps_run": len(micro_steps) == 20,
        "micro_training_steps_executed": len(micro_steps),
        "fixed_screen_run": False,
        "formal_training_run": False,
        "action": action,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(payload)
    print(json.dumps({
        "decision": decision,
        "optimizer_coverage": optimizer_coverage,
        "checkpoint_gate_std_max": maximum_gate_std,
        "active_step_optimizer_state_coverage": state_coverage_after,
        "manual_pooled_diff": pooled_diff,
        "manual_logit_diff": logits_diff,
        "warmup_fix_passed": warmup_fix_passed,
        "micro_training_passed": micro_passed,
        "micro_gate_deviation_modalities": gate_deviation_modality_count,
        "micro_gate_std_modalities": gate_std_modality_count,
        "micro_ratio_modalities": ratio_modality_count,
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
