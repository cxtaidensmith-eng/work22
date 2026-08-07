"""Staged runner for Structured Group HFT-C1 v2.

The three stages are intentionally separate so implementation/smoke, hard-fold
screen, and the conditionally-authorized formal run can be committed
independently::

    --smoke   fold-4 epoch-1 equivalence plus simulated epochs 21--23
    --screen  fresh folds 4/7/8 and the preregistered SG_HFT decision
    --formal  fresh folds 0--9, permitted only after committed SG_HFT_GO
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_hft_c1_lite_v1 as legacy
from Loss import criterion_query_pool_no_orth
from Model.sg_hft_c1_v2 import SGHFTC1V2Model, build_semantic_group_manifest
from Utils import CustomCosineAnnealingLR, SET_Random


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/sg_hft_c1_v2")
HFT_V1_BASE_COMMIT = "0dd9161bdec9c91bb99f67325766f9361cba9b91"
C1_ANCHOR_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
C1_SOURCE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
C1_REMOTE_REF = "refs/remotes/origin/experiment/cme-dual-branch-v1"
C1_OOF_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
C1_REPORT_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/report.json"
C1_BLOB_SHA256 = "6f4ef505641f75236b01ae128f2836bf2f126ced43158f9c4ee21cc090898018"
BRANCH_PREFIX = "experiment/sg-hft-c1-v2"
FOLDS = tuple(range(10))
SCREEN_FOLDS = (4, 7, 8)
SEED = 0
EPOCHS = 400
SMOKE_ACTIVATION_EPOCHS = (21, 22, 23)
CAP = 0.08
EXPECTED_ADDED_PARAMETERS = 12_351
C1_PARAMETERS = 862_971
TOTAL_PARAMETERS = C1_PARAMETERS + EXPECTED_ADDED_PARAMETERS
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
SG_MODALITIES = ("MRI", "PET", "ROI")
EXPECTED_FEATURE_COUNTS = {"MRI": 138, "PET": 150, "CSF": 3, "Risk": 36, "COG": 24, "ROI": 9}

C1 = deepcopy(legacy.C1)
C1_SCREEN = deepcopy(legacy.C1_SCREEN)

# Reuse the proven data/probability/C1-OOF implementation while pinning it to
# immutable experiment anchors and the SG branch.  No HFT-v1 training is run.
legacy.BASE_COMMIT = HFT_V1_BASE_COMMIT
legacy.BRANCH_PREFIX = BRANCH_PREFIX
legacy.C1_GIT_REF = C1_SOURCE_COMMIT
legacy.C1_OOF_REL = C1_OOF_REL
legacy.C1_REPORT_REL = C1_REPORT_REL
legacy.C1_BLOB_SHA256 = C1_BLOB_SHA256

require = legacy.require
write_json = legacy.write_json
write_csv = legacy.write_csv
read_csv = legacy.read_csv
git_value = legacy.git_value
parameter_count = legacy.parameter_count
clone_cpu_state = legacy.clone_cpu_state
gradients_finite = legacy.gradients_finite
module_gradient_norm = legacy.module_gradient_norm
selection_metrics = legacy.selection_metrics
prediction_rows = legacy.prediction_rows
metrics_from_rows = legacy.metrics_from_rows
boundary_counts = legacy.boundary_counts
paired_comparison = legacy.paired_comparison


def validate_git_context() -> dict:
    branch = git_value("branch", "--show-current")
    require(
        branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-r"),
        f"Unexpected experiment branch: {branch}",
    )
    for commit in (HFT_V1_BASE_COMMIT, C1_ANCHOR_COMMIT, C1_SOURCE_COMMIT):
        subprocess.check_call(
            [
                "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
                "cat-file", "-e", f"{commit}^{{commit}}",
            ]
        )
    for ancestor, descendant, label in (
        (C1_ANCHOR_COMMIT, HFT_V1_BASE_COMMIT, "C1 anchor is not an ancestor of HFT-v1"),
        (HFT_V1_BASE_COMMIT, "HEAD", "HFT-v1 implementation base is not an ancestor of HEAD"),
    ):
        result = subprocess.run(
            [
                "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
                "merge-base", "--is-ancestor", ancestor, descendant,
            ],
            check=False,
        )
        require(result.returncode == 0, label)
    return {
        "branch": branch,
        "head": git_value("rev-parse", "HEAD"),
        "base_commit": HFT_V1_BASE_COMMIT,
        "c1_anchor_commit": C1_ANCHOR_COMMIT,
    }


def load_context() -> dict:
    # legacy.load_context performs the fixed TADPOLE shape, fold, class-order,
    # modality-order, CUDA, and immutable C1 OOF checks.
    legacy.validate_git_context = validate_git_context
    context = legacy.load_context()
    require(context["git"]["base_commit"] == HFT_V1_BASE_COMMIT, "Experiment base changed")
    return context


def build_group_manifest(context: dict, fold: int) -> dict:
    train_mask, _ = context["dataset_data"]["Mask"][fold]
    groups = build_semantic_group_manifest(
        context["dataset_dict"], context["feature_names_by_modality"]
    )
    require(set(groups) == set(SG_MODALITIES), "SG modality set changed")
    require(
        {modality: len(modality_groups) for modality, modality_groups in groups.items()}
        == {"MRI": 4, "PET": 6, "ROI": 5},
        "Semantic group counts changed",
    )
    for modality, modality_groups in groups.items():
        flattened = [
            int(index)
            for group in modality_groups
            for index in group["global_feature_indices"]
        ]
        modal_position = MODALITY_NAMES.index(modality)
        expected = [
            int(index)
            for index in context["dataset_dict"]["Modal_Index"][modal_position]
        ]
        require(
            sorted(flattened) == sorted(expected) and len(flattened) == len(set(flattened)),
            f"{modality} groups are not an exact partition",
        )
    return {
        "version": 1,
        "fold": int(fold),
        "method": "deterministic_feature_name_semantics",
        "uses_labels": False,
        "uses_feature_values": False,
        "fit_scope": "feature names only; fold train/test labels and values are not read",
        "train_subject_count": int(train_mask.sum()),
        "groups": groups,
    }


def alpha_for_epoch(epoch: int) -> float:
    if epoch <= 20:
        return 0.0
    if epoch <= 40:
        return float(epoch - 20) / 20.0
    return 1.0


def build_model(context: dict, manifest: dict, sg_enabled: bool = True) -> SGHFTC1V2Model:
    config = context["config"]
    SET_Random(SEED)
    model = SGHFTC1V2Model(
        context["dataset_dict"],
        Herter_Graph=None,
        Hidden_size=config.Hidden_size,
        Drop_rate=config.Drop_rate,
        K=config.ChebGCN_K,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        input_noise_std=config.input_noise_std,
        drop_path=config.drop_path,
        graph_head=config.Graph_head,
        graph_layers=config.graph_layers,
        graph_heads=config.graph_heads,
        graph_beta=config.graph_beta,
        graph_k_order=config.graph_k_order,
        graph_alpha=config.graph_alpha,
        graph_kernel=config.graph_kernel,
        graph_use_graph=False,
        graph_dropout=config.graph_dropout,
        graph_hidden=config.graph_hidden,
        global_word_emb=config.global_word_emb,
        semantic_branch="both",
        semantic_fusion="add",
        category_branch_variant="original",
        query_pool_variant="independent",
        category_branch_fusion="concat",
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
        group_manifest=manifest["groups"] if sg_enabled else None,
        cap=CAP,
        sg_enabled=sg_enabled,
    ).to(context["device"])
    if sg_enabled:
        require(set(model.sg_branches) == set(SG_MODALITIES), "SG branch set changed")
        require(model.sg_parameter_count() == EXPECTED_ADDED_PARAMETERS, "SG parameter count changed")
        require(model.c1_parameter_count() == C1_PARAMETERS, "C1 backbone parameter count changed")
        require(parameter_count(model) == TOTAL_PARAMETERS, "SG total parameter count changed")
        require(model.inference_parameter_count() == TOTAL_PARAMETERS, "SG inference parameter count changed")
    else:
        require(parameter_count(model) == C1_PARAMETERS, "C1 reference parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def make_fresh_training_objects(context: dict, manifest: dict):
    model = build_model(context, manifest, True)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=float(context["config"].Lr_Min),
    )
    require(len(optimizer.state) == 0 and int(scheduler.T_max) == EPOCHS, "Fresh optimizer/scheduler changed")
    return model, criterion, optimizer, scheduler


def common_c1_state(model: SGHFTC1V2Model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("sg_branches.")
    }


def gradient_family_norm(branch: torch.nn.Module, family: str) -> float:
    if family == "group_encoder":
        selected = lambda name: "group_encoder" in name or "group_input" in name or "shared_group" in name
    elif family == "gate":
        selected = lambda name: "gate" in name
    elif family == "adapter":
        selected = lambda name: "adapter" in name
    else:
        raise ValueError(f"Unknown gradient family: {family}")
    values = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in branch.named_parameters()
        if selected(name) and parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu())


def fold_config(context: dict, fold: int, scope: str, manifest: dict) -> dict:
    return {
        "experiment": "Structured Group HFT-C1 v2",
        "short_name": "SG-HFT-C1 v2",
        "scope": scope,
        "fold": int(fold),
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "subjects": 598,
        "features": 360,
        "class_order": list(CLASS_NAMES),
        "modality_order": list(MODALITY_NAMES),
        "SG_modalities": list(SG_MODALITIES),
        "group_counts": {modality: len(groups) for modality, groups in manifest["groups"].items()},
        "group_construction": manifest["method"],
        "grouping_uses_labels": False,
        "grouping_uses_feature_values": False,
        "group_encoder": "Linear(group_width,8)->GELU->shared-per-modality Linear(8,hidden_dim)",
        "gate": "shared-per-modality Linear(2*hidden_dim,8)->GELU->Linear(8,1); sigmoid gated mean",
        "gate_context_detached": True,
        "adapter_rank": 4,
        "adapter_final_initialization": "Xavier gain=0.05; bias=0",
        "activation_schedule": {"epochs_1_20": 0.0, "epochs_21_40": "linear 0->1", "epochs_41_400": 1.0},
        "cap": CAP,
        "cap_learnable": False,
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "graph_enabled": False,
        "orthogonality": False,
        "optimizer": "Adam",
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "loss": "complete historical C1 weighted CE plus three OVR auxiliary losses; no SG auxiliary loss",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "parameters": {"C1": C1_PARAMETERS, "added_SG": EXPECTED_ADDED_PARAMETERS, "total": TOTAL_PARAMETERS},
    }


def experiment_config(context: dict) -> dict:
    return {
        "experiment": "Structured Group HFT-C1 v2",
        "base_commit": HFT_V1_BASE_COMMIT,
        "c1_anchor_commit": C1_ANCHOR_COMMIT,
        "c1_source_commit": C1_SOURCE_COMMIT,
        "c1_oof_path": C1_OOF_REL,
        "c1_oof_blob_sha256": C1_BLOB_SHA256,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(FOLDS),
        "screen_folds": list(SCREEN_FOLDS),
        "formal_folds_are_fresh": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "smoke_activation_epochs": list(SMOKE_ACTIVATION_EPOCHS),
        "device": "cuda:0",
        "cap": CAP,
        "feature_counts": EXPECTED_FEATURE_COUNTS,
        "C1_parameters": C1_PARAMETERS,
        "added_parameters": EXPECTED_ADDED_PARAMETERS,
        "training_parameters": TOTAL_PARAMETERS,
        "inference_parameters": TOTAL_PARAMETERS,
        "screen_C1_anchor": C1_SCREEN,
        "formal_C1_anchor": C1,
        "screen_rules": {
            "SG_HFT_GO": "correct>=167, repairs>damages, no AD-CN, boundaries<=C1, F1/BACC within .005, >=2 ratio>=.01, >=2 entropy<.98",
            "SG_HFT_NEAR": "correct=166, repairs>=damages, no AD-CN, boundaries<=C1, >=2 ratio>=.01, >=2 entropy<.98",
            "SG_HFT_STOP": "any preregistered stop condition",
            "adjustments": 0,
        },
        "formal_rules": {
            "FORMAL_GO": "correct>=561, F1/BACC within .003, no AD-CN",
            "FORMAL_NEUTRAL": "correct=560 and AUC or BACC improves",
            "FORMAL_STOP": "correct<560 or material F1/BACC damage",
        },
    }


def write_environment(output_root: Path, context: dict) -> None:
    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "sklearn": sklearn.__version__,
            "branch": context["git"]["branch"],
            "run_commit": context["git"]["head"],
            "base_commit": HFT_V1_BASE_COMMIT,
            "c1_anchor_commit": C1_ANCHOR_COMMIT,
        },
    )


def mechanism_diagnostics(intermediates: dict, mask: torch.Tensor, manifest: dict) -> dict:
    output = {}
    require(abs(float(intermediates["sg_cap"]) - CAP) <= 1e-12, "SG cap changed")
    for modality in SG_MODALITIES:
        shared = intermediates["sg_shared_by_modality"][modality][mask]
        delta = intermediates["sg_delta_by_modality"][modality][mask]
        ratio = intermediates["sg_ratio_by_modality"][modality][mask].squeeze(-1)
        saturated = intermediates["sg_cap_saturated_by_modality"][modality][mask].squeeze(-1)
        gates = intermediates["sg_group_gates_by_modality"][modality][mask]
        require(gates.ndim == 2, f"{modality} group-gate shape changed")
        groups = manifest["groups"][modality]
        require(gates.size(1) == len(groups), f"{modality} group count changed")
        normalized = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(normalized.clamp_min(1e-12) * normalized.clamp_min(1e-12).log()).sum(dim=1)
        entropy = entropy / math.log(gates.size(1))
        require(
            bool(torch.isfinite(ratio).all() and torch.isfinite(gates).all() and torch.isfinite(entropy).all()),
            f"{modality} diagnostic non-finite",
        )
        maximum_ratio = float(ratio.max().detach().cpu())
        require(maximum_ratio <= CAP + 1e-6, f"{modality} cap violated: {maximum_ratio}")
        output[modality] = {
            "count": int(mask.sum()),
            "ratio_sum": float(ratio.sum().detach().cpu()),
            "ratio_max": maximum_ratio,
            "cap_saturation_count": int(saturated.sum().detach().cpu()),
            "entropy_sum": float(entropy.sum().detach().cpu()),
            "gate_sum": gates.sum(dim=0).detach().cpu().tolist(),
            "gate_range": float((gates.max() - gates.min()).detach().cpu()),
            "shared_norm_mean": float(shared.norm(dim=-1).mean().detach().cpu()),
            "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu()),
            "group_names": [str(group["group_name"]) for group in groups],
        }
    return output


def summarize_mechanism(fold_diagnostics: list[dict], fold_manifests: list[dict]) -> dict:
    result = {}
    for modality in SG_MODALITIES:
        count = sum(item[modality]["count"] for item in fold_diagnostics)
        names = fold_diagnostics[0][modality]["group_names"]
        require(
            all(item[modality]["group_names"] == names for item in fold_diagnostics),
            f"{modality} semantic group names changed across folds",
        )
        gate_sum = np.sum(
            [np.asarray(item[modality]["gate_sum"], dtype=np.float64) for item in fold_diagnostics],
            axis=0,
        )
        mean_gate = gate_sum / max(1, count)
        mean_ratio = sum(item[modality]["ratio_sum"] for item in fold_diagnostics) / max(1, count)
        mean_entropy = sum(item[modality]["entropy_sum"] for item in fold_diagnostics) / max(1, count)
        group_weights = [
            {"group_name": name, "mean_gate": float(mean_gate[index])}
            for index, name in enumerate(names)
        ]
        modality_result = {
            "mean_residual_shared_ratio": float(mean_ratio),
            "maximum_ratio": float(max(item[modality]["ratio_max"] for item in fold_diagnostics)),
            "cap_saturation_fraction": float(
                sum(item[modality]["cap_saturation_count"] for item in fold_diagnostics) / max(1, count)
            ),
            "normalized_group_gate_entropy": float(mean_entropy),
            "mean_group_gates": group_weights,
            "gate_range_maximum": float(max(item[modality]["gate_range"] for item in fold_diagnostics)),
        }
        if mean_entropy < 0.98:
            highest = int(np.argmax(mean_gate))
            feature_names = fold_manifests[0]["groups"][modality][highest]["feature_names"]
            modality_result["highest_weight_group"] = {
                "group_name": names[highest],
                "mean_gate": float(mean_gate[highest]),
                "feature_names": feature_names,
            }
        else:
            modality_result["selection_interpretation"] = "group gate did not form reliable selection"
        result[modality] = modality_result
    ratios = [result[modality]["mean_residual_shared_ratio"] for modality in SG_MODALITIES]
    entropies = [result[modality]["normalized_group_gate_entropy"] for modality in SG_MODALITIES]
    result["overall"] = {
        "modalities_ratio_at_least_0_01": int(sum(value >= 0.01 for value in ratios)),
        "modalities_entropy_below_0_98": int(sum(value < 0.98 for value in entropies)),
        "all_three_branches_near_zero": bool(all(value < 0.01 for value in ratios)),
        "all_three_gate_entropies_near_one": bool(all(value >= 0.98 for value in entropies)),
        "all_three_gates_constant": bool(
            all(result[modality]["gate_range_maximum"] <= 1e-8 for modality in SG_MODALITIES)
        ),
    }
    return result


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke output: {smoke_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True)
    write_json(output_root / "config.json", experiment_config(context))
    write_environment(output_root, context)
    shutil.copyfile(ROOT / CONFIG_REL, output_root / "historical_c1_config.ini")

    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][4]
    manifest = build_group_manifest(context, 4)
    write_json(smoke_root / "group_manifest.json", manifest)

    c1_model = build_model(context, manifest, False)
    sg_model = build_model(context, manifest, True)
    c1_state = common_c1_state(c1_model)
    sg_state = common_c1_state(sg_model)
    require(c1_state.keys() == sg_state.keys(), "C1/SG common state keys differ")
    require(all(torch.equal(c1_state[key], sg_state[key]) for key in c1_state), "SG construction perturbed C1 initialization")

    # Alpha=0 must preserve not only deterministic evaluation but also the
    # stochastic C1 training trajectory (input noise/dropout).  Replay the
    # exact CPU/CUDA generator states for both forwards.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    c1_model.train()
    sg_model.train()
    with torch.no_grad():
        c1_train_raw, _, _ = c1_model(features, activation_epoch=1)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_states)
    with torch.no_grad():
        sg_train_raw, _, _ = sg_model(features, activation_epoch=1)
    train_mode_logit_diff = float(
        (c1_train_raw - sg_train_raw).abs().max().detach().cpu()
    )
    require(
        train_mode_logit_diff <= 1e-6,
        f"Epoch-1 train-mode RNG equivalence failed: {train_mode_logit_diff}",
    )
    c1_model.eval()
    sg_model.eval()
    with torch.no_grad():
        c1_raw, _, _, c1_inter = c1_model(features, activation_epoch=1, return_intermediates=True)
        sg_raw, _, _, sg_inter = sg_model(features, activation_epoch=1, return_intermediates=True)
    logit_diff = float((c1_raw - sg_raw).abs().max().detach().cpu())
    global_diff = float((c1_inter["G"] - sg_inter["G"]).abs().max().detach().cpu())
    require(logit_diff <= 1e-6, f"Epoch-1 C1 equivalence failed: {logit_diff}")
    require(global_diff <= 1e-7, f"Global path changed: {global_diff}")
    require(float(sg_inter["sg_alpha"]) == 0.0, "Epoch-1 alpha is not zero")
    require(
        max(float(value.abs().max().cpu()) for value in sg_inter["sg_delta_by_modality"].values()) == 0.0,
        "Epoch-1 SG delta is not exactly zero",
    )
    del c1_model, sg_model, c1_raw, sg_raw, c1_inter, sg_inter, c1_train_raw, sg_train_raw
    torch.cuda.empty_cache()

    model, criterion, optimizer, scheduler = make_fresh_training_objects(context, manifest)
    families = ("group_encoder", "gate", "adapter")
    gradient_seen = {
        modality: {family: False for family in families} for modality in SG_MODALITIES
    }
    gates_nonconstant = {modality: False for modality in SG_MODALITIES}
    epoch_rows = []
    final_diagnostics = None
    started = time.perf_counter()
    for simulated_epoch in SMOKE_ACTIVATION_EPOCHS:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features, activation_epoch=simulated_epoch, return_intermediates=True
        )
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Smoke epoch {simulated_epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"Smoke epoch {simulated_epoch}: non-finite gradient")
        for modality, branch in model.sg_branches.items():
            for family in families:
                gradient_seen[modality][family] |= gradient_family_norm(branch, family) > 0.0
            gates = intermediates["sg_group_gates_by_modality"][modality]
            gates_nonconstant[modality] |= float((gates.max() - gates.min()).detach().cpu()) > 1e-8
        final_diagnostics = mechanism_diagnostics(intermediates, test_mask, manifest)
        require(abs(float(intermediates["sg_alpha"]) - alpha_for_epoch(simulated_epoch)) <= 1e-12, "Smoke alpha changed")
        epoch_rows.append(
            {
                "simulated_epoch": simulated_epoch,
                "alpha": float(intermediates["sg_alpha"]),
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
    require(all(all(values.values()) for values in gradient_seen.values()), f"Smoke SG gradient missing: {gradient_seen}")
    require(all(gates_nonconstant.values()), f"Smoke group gates constant: {gates_nonconstant}")
    require(final_diagnostics is not None, "Smoke diagnostics missing")

    model.eval()
    with torch.no_grad():
        before_raw, _, _, before_inter = model(features, activation_epoch=23, return_intermediates=True)
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    checkpoint = {
        "model_state": clone_cpu_state(model),
        "group_manifest": manifest,
        "activation_epoch": 23,
    }
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=context["device"], weights_only=True)
    require(loaded["group_manifest"] == manifest and int(loaded["activation_epoch"]) == 23, "Smoke checkpoint manifest/epoch changed")
    reloaded = build_model(context, loaded["group_manifest"], True)
    reloaded.load_state_dict(loaded["model_state"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        after_raw, _, _, after_inter = reloaded(features, activation_epoch=23, return_intermediates=True)
    reload_diff = float((before_raw - after_raw).abs().max().detach().cpu())
    require(reload_diff <= 1e-6, f"Smoke checkpoint logit mismatch: {reload_diff}")
    mechanism_diagnostics(after_inter, test_mask, manifest)
    elapsed = time.perf_counter() - started
    payload = {
        "passed": True,
        "fold": 4,
        "activation_epochs": list(SMOKE_ACTIVATION_EPOCHS),
        "epoch1_max_abs_logit_diff_vs_C1": logit_diff,
        "epoch1_train_mode_rng_logit_diff": train_mode_logit_diff,
        "epoch1_global_max_abs_diff_vs_C1": global_diff,
        "gradient_nonzero": gradient_seen,
        "group_gates_nonconstant": gates_nonconstant,
        "finite": True,
        "cap_respected": True,
        "parameters": {"C1": C1_PARAMETERS, "added_SG": EXPECTED_ADDED_PARAMETERS, "total": TOTAL_PARAMETERS},
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_manifest_restored": True,
        "checkpoint_readback_max_abs_logit_diff": reload_diff,
        "final_mechanism": summarize_mechanism([final_diagnostics], [manifest]),
        "elapsed_seconds": elapsed,
        "run_commit": context["git"]["head"],
    }
    write_csv(smoke_root / "epoch_metrics.csv", epoch_rows)
    write_json(smoke_root / "summary.json", payload)
    write_json(output_root / "summary.json", {"decision": "SMOKE_PASS", "smoke": payload})
    (output_root / "summary.md").write_text(
        "# SG-HFT-C1 v2\n\nDecision: `SMOKE_PASS`\n\n"
        f"Epoch-1 maximum C1 logit difference: `{logit_diff:.3e}`. "
        f"Checkpoint readback difference: `{reload_diff:.3e}`.\n",
        encoding="utf-8",
    )
    print("SMOKE PASS", flush=True)
    del model, criterion, optimizer, scheduler, reloaded, before_raw, after_raw, before_inter, after_inter
    torch.cuda.empty_cache()
    return payload


def train_fold(
    context: dict,
    fold: int,
    scope_root: Path,
    scope: str,
) -> tuple[dict, list[dict], dict, dict]:
    final_dir = scope_root / f"fold_{fold:02d}"
    staging_dir = scope_root / f".fold_{fold:02d}_in_progress"
    require(not final_dir.exists() and not staging_dir.exists(), f"Refusing to overwrite {scope} fold {fold}")
    staging_dir.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool((train_mask & test_mask).any()), f"Fold {fold} train/test masks overlap")
    require(int(train_mask.sum() + test_mask.sum()) == 598, f"Fold {fold} masks incomplete")
    manifest = build_group_manifest(context, fold)
    model, criterion, optimizer, scheduler = make_fresh_training_objects(context, manifest)
    config_payload = fold_config(context, fold, scope, manifest)
    best = None
    best_state = None
    epoch_rows = []
    families = ("group_encoder", "gate", "adapter")
    gradient_seen = {
        modality: {family: False for family in families} for modality in SG_MODALITIES
    }
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features, activation_epoch=epoch)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"{scope} fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"{scope} fold{fold} epoch{epoch}: non-finite gradient")
        if epoch >= 21:
            for modality, branch in model.sg_branches.items():
                for family in families:
                    gradient_seen[modality][family] |= gradient_family_norm(branch, family) > 0.0
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _, eval_inter = model(
                features, activation_epoch=epoch, return_intermediates=True
            )
            test_metrics, selection_tuple = selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or selection_tuple > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": selection_tuple,
                "metrics": deepcopy(test_metrics),
            }
            best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "alpha": float(eval_inter["sg_alpha"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "correct": test_metrics["correct"],
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
                **{
                    f"cap_saturation_fraction_{modality}": float(
                        eval_inter["sg_cap_saturated_by_modality"][modality][test_mask].float().mean().cpu()
                    )
                    for modality in SG_MODALITIES
                },
                **{
                    f"mean_residual_shared_ratio_{modality}": float(
                        eval_inter["sg_ratio_by_modality"][modality][test_mask].mean().cpu()
                    )
                    for modality in SG_MODALITIES
                },
            }
        )
        del eval_inter

    require(best is not None and best_state is not None, f"{scope} fold{fold}: no best state")
    require(all(all(values.values()) for values in gradient_seen.values()), f"{scope} fold{fold}: SG gradient missing")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_inter = model(
            features, activation_epoch=int(best["epoch"]), return_intermediates=True
        )
        best_metrics, _ = selection_metrics(
            best_raw,
            labels,
            test_mask,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
        rows = prediction_rows(
            fold, best_raw, labels, test_mask, context["dataset_dict"], context["config"]
        )
        diagnostics = mechanism_diagnostics(best_inter, test_mask, manifest)
    require(best_metrics == best["metrics"], f"{scope} fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"{scope} fold{fold}: saved predictions changed metrics")
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "scope": scope,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "fresh_model": True,
        "fresh_criterion": True,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "best_epoch": int(best["epoch"]),
        "best_alpha": alpha_for_epoch(int(best["epoch"])),
        "best_metrics": best_metrics,
        "boundary_counts": boundary_counts(best_metrics),
        "training_parameters": parameter_count(model),
        "inference_parameters": model.inference_parameter_count(),
        "added_parameters": model.sg_parameter_count(),
        "gradient_nonzero_after_activation": gradient_seen,
        "post_activation_saturation_epoch_fraction_ge_0_50": {
            modality: float(
                np.mean([
                    row[f"cap_saturation_fraction_{modality}"] >= 0.50
                    for row in epoch_rows[40:]
                ])
            )
            for modality in SG_MODALITIES
        },
        "elapsed_seconds": elapsed,
        "config": config_payload,
    }
    torch.save(
        {"model_state": best_state, "group_manifest": manifest, "best_epoch": int(best["epoch"])},
        staging_dir / "checkpoint_best.pt",
    )
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_json(staging_dir / "config.json", config_payload)
    write_json(staging_dir / "group_manifest.json", manifest)
    write_json(staging_dir / "summary.json", summary)
    write_json(staging_dir / "mechanism.json", diagnostics)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    write_csv(
        staging_dir / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(best_metrics["confusion_matrix"])
        ],
    )
    staging_dir.rename(final_dir)
    print(
        f"fold={fold} best_epoch={best['epoch']} correct={best_metrics['correct']} "
        f"ACC={best_metrics['acc']:.7f} Macro-F1={best_metrics['macro_f1']:.7f} "
        f"BACC={best_metrics['bacc']:.7f} Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler, best_raw, best_inter
    torch.cuda.empty_cache()
    return summary, rows, diagnostics, manifest


def aggregate_fold_outputs(
    context: dict,
    fold_summaries: list[dict],
    rows: list[dict],
    fold_diagnostics: list[dict],
    fold_manifests: list[dict],
    expected_folds: tuple[int, ...],
    expected_subjects: int,
) -> dict:
    rows.sort(key=lambda row: int(row["subject_index"]))
    require({int(summary["fold"]) for summary in fold_summaries} == set(expected_folds), "Fold summaries changed")
    require(len(rows) == expected_subjects, f"Expected {expected_subjects} OOF rows")
    require(len({int(row["subject_index"]) for row in rows}) == expected_subjects, "OOF subjects not unique")
    require({int(row["fold"]) for row in rows} == set(expected_folds), "OOF folds changed")
    metrics = metrics_from_rows(rows)
    fold_acc = np.asarray([summary["best_metrics"]["acc"] for summary in fold_summaries], dtype=np.float64)
    return {
        "metrics": metrics,
        **boundary_counts(metrics),
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_sd": float(fold_acc.std(ddof=1)),
        "fold_results": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "correct": summary["best_metrics"]["correct"],
                "acc": summary["best_metrics"]["acc"],
                "macro_f1": summary["best_metrics"]["macro_f1"],
                "bacc": summary["best_metrics"]["bacc"],
                "macro_auc": summary["best_metrics"]["macro_auc"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "fresh_run": True,
            }
            for summary in sorted(fold_summaries, key=lambda item: item["fold"])
        ],
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in fold_summaries)),
        "training_parameters": TOTAL_PARAMETERS,
        "inference_parameters": TOTAL_PARAMETERS,
        "added_parameters": EXPECTED_ADDED_PARAMETERS,
        "mechanism": summarize_mechanism(fold_diagnostics, fold_manifests),
        "paired_vs_C1": paired_comparison(rows, context["c1_by_subject"]),
        "post_activation_saturation_epoch_fraction_ge_0_50": {
            modality: float(np.mean([
                summary["post_activation_saturation_epoch_fraction_ge_0_50"][modality]
                for summary in fold_summaries
            ]))
            for modality in SG_MODALITIES
        },
    }


def save_aggregate(scope_root: Path, rows: list[dict], report: dict) -> None:
    write_csv(scope_root / "oof_predictions.csv", rows)
    write_csv(scope_root / "per_fold_metrics.csv", report["fold_results"])
    write_json(scope_root / "metrics.json", report["metrics"])
    write_json(scope_root / "mechanism.json", report["mechanism"])
    write_json(scope_root / "report.json", report)
    write_csv(
        scope_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(report["metrics"]["confusion_matrix"])
        ],
    )


def checkpoint_readback(context: dict, fold_dir: Path) -> dict:
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(fold_dir / "best_predictions.csv")
    disk_manifest = json.loads((fold_dir / "group_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        fold_dir / "checkpoint_best.pt", map_location=context["device"], weights_only=True
    )
    require(checkpoint["group_manifest"] == disk_manifest, "Checkpoint/disk group manifest mismatch")
    require(int(checkpoint["best_epoch"]) == int(summary["best_epoch"]), "Checkpoint best epoch changed")
    model = build_model(context, disk_manifest, True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    fold = int(summary["fold"])
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    with torch.no_grad():
        raw, _, _, inter = model(
            features, activation_epoch=int(summary["best_epoch"]), return_intermediates=True
        )
    reloaded_rows = prediction_rows(
        fold, raw, labels, test_mask, context["dataset_dict"], context["config"]
    )
    require([int(row["subject_index"]) for row in rows] == [int(row["subject_index"]) for row in reloaded_rows], "Checkpoint subject order mismatch")
    require([int(row["prediction"]) for row in rows] == [int(row["prediction"]) for row in reloaded_rows], "Checkpoint predictions mismatch")
    probability_diff = max(
        abs(float(left[f"probability_{name}"]) - float(right[f"probability_{name}"]))
        for left, right in zip(rows, reloaded_rows)
        for name in CLASS_NAMES
    )
    require(probability_diff <= 1e-6, f"Checkpoint probability mismatch: {probability_diff}")
    mechanism_diagnostics(inter, test_mask, disk_manifest)
    del model, raw, inter
    torch.cuda.empty_cache()
    return {"passed": True, "fold": fold, "maximum_probability_difference": probability_diff}


def screen_stop_reasons(report: dict) -> list[str]:
    metrics = report["metrics"]
    paired = report["paired_vs_C1"]
    mechanism = report["mechanism"]["overall"]
    reasons = []
    if metrics["correct"] <= 165:
        reasons.append("correct_at_most_165")
    if paired["repairs"] < paired["damages"]:
        reasons.append("repairs_less_than_damages")
    if report["ad_cn_errors"] > C1_SCREEN["ad_cn_errors"]:
        reasons.append("new_AD_CN_error")
    if report["ad_smci_errors"] >= C1_SCREEN["ad_smci_errors"] + 2:
        reasons.append("AD_SMCI_errors_increased_by_at_least_2")
    if report["cn_smci_errors"] >= C1_SCREEN["cn_smci_errors"] + 2:
        reasons.append("CN_SMCI_errors_increased_by_at_least_2")
    if mechanism["all_three_branches_near_zero"]:
        reasons.append("all_three_SG_branches_near_zero")
    if mechanism["all_three_gate_entropies_near_one"]:
        reasons.append("all_three_group_gate_entropies_near_one")
    persistent_cap = any(
        value >= 0.50
        for value in report["post_activation_saturation_epoch_fraction_ge_0_50"].values()
    )
    performance_declined = bool(
        metrics["correct"] < C1_SCREEN["correct"]
        or metrics["macro_f1"] < C1_SCREEN["macro_f1"]
        or metrics["bacc"] < C1_SCREEN["bacc"]
    )
    if persistent_cap and performance_declined:
        reasons.append("performance_declined_with_persistent_cap_saturation")
    return reasons


def classify_screen(report: dict) -> tuple[str, list[str]]:
    stop = screen_stop_reasons(report)
    if stop:
        return "SG_HFT_STOP", stop
    metrics = report["metrics"]
    paired = report["paired_vs_C1"]
    mechanism = report["mechanism"]["overall"]
    common = {
        "no_AD_CN_errors": report["ad_cn_errors"] == 0,
        "adjacent_boundary_errors_not_above_C1": report["boundary_errors"] <= C1_SCREEN["boundary_errors"],
        "at_least_two_active_modalities": mechanism["modalities_ratio_at_least_0_01"] >= 2,
        "at_least_two_selective_modalities": mechanism["modalities_entropy_below_0_98"] >= 2,
        "branches_not_all_closed": not mechanism["all_three_branches_near_zero"],
    }
    go = {
        **common,
        "correct_at_least_167": metrics["correct"] >= 167,
        "repairs_greater_than_damages": paired["repairs"] > paired["damages"],
        "macro_f1_drop_within_0_005": metrics["macro_f1"] >= C1_SCREEN["macro_f1"] - 0.005,
        "bacc_drop_within_0_005": metrics["bacc"] >= C1_SCREEN["bacc"] - 0.005,
    }
    report["SG_HFT_GO_checks"] = go
    if all(go.values()):
        return "SG_HFT_GO", []
    near = {
        **common,
        "correct_exactly_166": metrics["correct"] == 166,
        "repairs_not_less_than_damages": paired["repairs"] >= paired["damages"],
    }
    report["SG_HFT_NEAR_checks"] = near
    if all(near.values()):
        return "SG_HFT_NEAR", []
    return "SG_HFT_STOP", ["did_not_meet_preregistered_GO_or_NEAR_rule"]


def formal_decision(report: dict) -> tuple[str, list[str]]:
    metrics = report["metrics"]
    stop = []
    if metrics["correct"] < 560:
        stop.append("correct_below_560")
    if metrics["macro_f1"] < C1["macro_f1"] - 0.003:
        stop.append("Macro-F1_below_C1_by_more_than_0.003")
    if metrics["bacc"] < C1["bacc"] - 0.003:
        stop.append("BACC_below_C1_by_more_than_0.003")
    if stop:
        return "FORMAL_STOP", stop
    go = {
        "correct_at_least_561": metrics["correct"] >= 561,
        "macro_f1_safe": metrics["macro_f1"] >= C1["macro_f1"] - 0.003,
        "bacc_safe": metrics["bacc"] >= C1["bacc"] - 0.003,
        "no_AD_CN_errors": report["ad_cn_errors"] == 0,
    }
    report["FORMAL_GO_checks"] = go
    if all(go.values()):
        return "FORMAL_GO", []
    if metrics["correct"] == 560 and (metrics["macro_auc"] > C1["macro_auc"] or metrics["bacc"] > C1["bacc"]):
        return "FORMAL_NEUTRAL", ["correct_tied_C1_and_AUC_or_BACC_improved"]
    return "FORMAL_STOP", ["did_not_meet_preregistered_FORMAL_GO_or_NEUTRAL_rule"]


def require_committed_stage_files(paths: list[Path], stage_name: str) -> None:
    legacy.require_committed_stage_files(paths, stage_name)


def validate_smoke_for_screen(output_root: Path) -> dict:
    smoke_path = output_root / "smoke" / "summary.json"
    manifest_path = output_root / "smoke" / "group_manifest.json"
    require(smoke_path.is_file() and manifest_path.is_file(), "Run --smoke and commit implementation before --screen")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke.get("passed") is True, "Smoke did not pass")
    require(smoke.get("activation_epochs") == list(SMOKE_ACTIVATION_EPOCHS), "Smoke activation protocol changed")
    require(smoke.get("epoch1_max_abs_logit_diff_vs_C1", 1.0) <= 1e-6, "Smoke C1 equivalence missing")
    require(smoke.get("epoch1_train_mode_rng_logit_diff", 1.0) <= 1e-6, "Smoke train-mode RNG equivalence missing")
    require(smoke.get("checkpoint_readback_max_abs_logit_diff", 1.0) <= 1e-6, "Smoke checkpoint readback missing")
    require_committed_stage_files(
        [
            ROOT / "Model" / "sg_hft_c1_v2.py",
            ROOT / "scripts" / "run_sg_hft_c1_v2.py",
            output_root / "config.json",
            smoke_path,
            manifest_path,
        ],
        "Screen",
    )
    return smoke


def run_screen(context: dict, output_root: Path) -> dict:
    validate_smoke_for_screen(output_root)
    screen_root = output_root / "screen"
    require(not screen_root.exists(), f"Refusing to overwrite screen output: {screen_root}")
    screen_root.mkdir(parents=True)
    summaries, rows, diagnostics, manifests = [], [], [], []
    started = time.perf_counter()
    for fold in SCREEN_FOLDS:
        summary, fold_rows, fold_diagnostics, manifest = train_fold(
            context, fold, screen_root, "screen"
        )
        summaries.append(summary)
        rows.extend(fold_rows)
        diagnostics.append(fold_diagnostics)
        manifests.append(manifest)
    report = aggregate_fold_outputs(
        context, summaries, rows, diagnostics, manifests, SCREEN_FOLDS, C1_SCREEN["subjects"]
    )
    report["C1_anchor"] = C1_SCREEN
    report["metric_delta_vs_C1"] = {
        key: report["metrics"][key] - C1_SCREEN[key]
        for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    decision, reasons = classify_screen(report)
    report["decision"] = decision
    report["decision_reasons"] = reasons
    report["checkpoint_readback"] = checkpoint_readback(context, screen_root / "fold_04")
    save_aggregate(screen_root, rows, report)
    payload = {
        "decision": decision,
        "screen": report,
        "formal_authorized": decision == "SG_HFT_GO",
        "runtime_seconds": time.perf_counter() - started,
        "branch": context["git"]["branch"],
        "base_commit": HFT_V1_BASE_COMMIT,
        "c1_anchor_commit": C1_ANCHOR_COMMIT,
        "run_commit": context["git"]["head"],
        "worktree": str(ROOT),
        "device": torch.cuda.get_device_name(0),
        "parameters": {"C1": C1_PARAMETERS, "added_SG": EXPECTED_ADDED_PARAMETERS, "total": TOTAL_PARAMETERS},
        "C1_source": {
            "commit": C1_SOURCE_COMMIT,
            "remote_ref_at_implementation": C1_REMOTE_REF,
            "path": C1_OOF_REL,
            "sha256": C1_BLOB_SHA256,
            "C1_not_retrained": True,
        },
    }
    write_json(screen_root / "summary.json", payload)
    write_json(output_root / "summary.json", payload)
    (output_root / "summary.md").write_text(render_report(payload), encoding="utf-8")
    print(decision, flush=True)
    return payload


def validate_screen_for_formal(output_root: Path) -> dict:
    screen_path = output_root / "screen" / "summary.json"
    require(screen_path.is_file(), "Run --screen and commit its result before --formal")
    payload = json.loads(screen_path.read_text(encoding="utf-8"))
    require(payload.get("decision") == "SG_HFT_GO", "Formal forbidden without SG_HFT_GO")
    require(payload.get("formal_authorized") is True, "Screen did not authorize formal")
    require_committed_stage_files(
        [
            ROOT / "Model" / "sg_hft_c1_v2.py",
            ROOT / "scripts" / "run_sg_hft_c1_v2.py",
            screen_path,
            output_root / "screen" / "report.json",
            output_root / "screen" / "per_fold_metrics.csv",
            *[
                output_root / "screen" / f"fold_{fold:02d}" / "group_manifest.json"
                for fold in SCREEN_FOLDS
            ],
        ],
        "Formal",
    )
    return payload


def run_formal(context: dict, output_root: Path) -> dict:
    screen = validate_screen_for_formal(output_root)
    formal_root = output_root / "formal"
    require(not formal_root.exists(), f"Refusing to overwrite formal output: {formal_root}")
    formal_root.mkdir(parents=True)
    summaries, rows, diagnostics, manifests = [], [], [], []
    started = time.perf_counter()
    # Screen fold checkpoints are deliberately not reused.
    for fold in FOLDS:
        summary, fold_rows, fold_diagnostics, manifest = train_fold(
            context, fold, formal_root, "formal_fresh_10fold"
        )
        summaries.append(summary)
        rows.extend(fold_rows)
        diagnostics.append(fold_diagnostics)
        manifests.append(manifest)
    report = aggregate_fold_outputs(
        context, summaries, rows, diagnostics, manifests, FOLDS, 598
    )
    report.update(
        {
            "formal_folds_fresh": True,
            "screen_fold_checkpoints_reused": False,
            "C1_anchor": C1,
            "metric_delta_vs_C1": {
                key: report["metrics"][key] - C1[key]
                for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
            },
        }
    )
    decision, reasons = formal_decision(report)
    report["decision"] = decision
    report["decision_reasons"] = reasons
    report["checkpoint_readback"] = checkpoint_readback(context, formal_root / "fold_00")
    report["total_runtime_seconds"] = time.perf_counter() - started
    save_aggregate(formal_root, rows, report)
    payload = {
        "decision": decision,
        "branch": context["git"]["branch"],
        "base_commit": HFT_V1_BASE_COMMIT,
        "c1_anchor_commit": C1_ANCHOR_COMMIT,
        "run_commit": context["git"]["head"],
        "worktree": str(ROOT),
        "device": torch.cuda.get_device_name(0),
        "parameters": {"C1": C1_PARAMETERS, "added_SG": EXPECTED_ADDED_PARAMETERS, "total": TOTAL_PARAMETERS},
        "screen": screen,
        "formal": report,
        "C1_source": {"commit": C1_SOURCE_COMMIT, "path": C1_OOF_REL, "sha256": C1_BLOB_SHA256, "C1_not_retrained": True},
    }
    write_json(formal_root / "summary.json", payload)
    write_json(output_root / "summary.json", payload)
    (output_root / "summary.md").write_text(render_report(payload), encoding="utf-8")
    print(decision, flush=True)
    return payload


def metric_line(metrics: dict, denominator: int) -> str:
    return (
        f"Correct `{metrics['correct']}/{denominator}`; ACC `{metrics['acc']:.7f}`; "
        f"Macro-F1 `{metrics['macro_f1']:.7f}`; BACC `{metrics['bacc']:.7f}`; "
        f"Probability Macro-AUC `{metrics['macro_auc']:.7f}`; Weighted-F1 `{metrics['weighted_f1']:.7f}`."
    )


def mechanism_markdown(mechanism: dict) -> list[str]:
    lines = [
        "| Modality | Mean ratio | Max ratio | Cap saturation | Gate entropy | Group weights |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for modality in SG_MODALITIES:
        item = mechanism[modality]
        weights = ", ".join(
            f"{entry['group_name']}={entry['mean_gate']:.4f}"
            for entry in item["mean_group_gates"]
        )
        lines.append(
            f"| {modality} | {item['mean_residual_shared_ratio']:.6f} | "
            f"{item['maximum_ratio']:.6f} | {item['cap_saturation_fraction']:.6f} | "
            f"{item['normalized_group_gate_entropy']:.6f} | {weights} |"
        )
        if item["normalized_group_gate_entropy"] >= 0.98:
            lines.append(f"\n{modality}: group gate did not form reliable selection.")
        elif "highest_weight_group" in item:
            highest = item["highest_weight_group"]
            lines.append(
                f"\n{modality} highest-weight group: `{highest['group_name']}` "
                f"(mean gate `{highest['mean_gate']:.4f}`); features: "
                + ", ".join(f"`{name}`" for name in highest["feature_names"])
                + "."
            )
    return lines


def render_report(payload: dict) -> str:
    decision = payload["decision"]
    report = payload.get("formal") or payload.get("screen")
    denominator = 598 if "formal" in payload else 179
    lines = [
        "# Structured Group HFT-C1 v2",
        "",
        f"Decision: `{decision}`",
        "",
        f"Branch: `{payload['branch']}`",
        f"Base commit: `{payload['base_commit']}`",
        f"Run commit: `{payload['run_commit']}`",
        f"Device: `{payload['device']}`",
        f"Parameters: `{payload['parameters']['total']}` (added `{payload['parameters']['added_SG']}`).",
        "",
        "## Results",
        "",
        metric_line(report["metrics"], denominator),
        "",
        f"Confusion matrix: `{report['metrics']['confusion_matrix']}`.",
        f"Repairs/damages: `{report['paired_vs_C1']['repairs']}/{report['paired_vs_C1']['damages']}`.",
        f"AD-sMCI / CN-sMCI / AD-CN errors: `{report['ad_smci_errors']}/{report['cn_smci_errors']}/{report['ad_cn_errors']}`.",
        "",
        "| Fold | Best epoch | Correct | ACC |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["fold_results"]:
        lines.append(f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} |")
    lines.extend(["", "## Mechanism", "", "Grouping: deterministic feature-name semantics; no labels or feature values used.", ""])
    lines.extend(mechanism_markdown(report["mechanism"]))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"Exceeded the corresponding C1 correct count: `{'yes' if report['metrics']['correct'] > (C1['correct'] if denominator == 598 else C1_SCREEN['correct']) else 'no'}`.",
            f"Proceed to formal ten-fold training: `{'yes' if decision == 'SG_HFT_GO' else 'no'}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--smoke", action="store_true", help="run fold-4 SG activation smoke")
    stage.add_argument("--screen", action="store_true", help="run fixed folds 4/7/8")
    stage.add_argument("--formal", action="store_true", help="run fresh ten folds after committed SG_HFT_GO")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / OUTPUT_REL,
        help="stage artifact root (default: experiments/sg_hft_c1_v2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    require(ROOT.resolve() in output_root.parents, "Output root must remain inside isolated worktree")
    context = load_context()
    stage = "smoke" if args.smoke else "screen" if args.screen else "formal"
    started = time.perf_counter()
    if args.smoke:
        result = run_smoke(context, output_root)
        decision = "SMOKE_PASS"
    elif args.screen:
        result = run_screen(context, output_root)
        decision = result["decision"]
    else:
        result = run_formal(context, output_root)
        decision = result["decision"]
    write_json(
        output_root / "status.json",
        {
            "stage": stage,
            "decision": decision,
            "completed": True,
            "branch": context["git"]["branch"],
            "base_commit": HFT_V1_BASE_COMMIT,
            "c1_anchor_commit": C1_ANCHOR_COMMIT,
            "run_commit": context["git"]["head"],
            "stage_runtime_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
