#!/usr/bin/env python3
"""PS-SPR v1 on locked TADPOLE-binary D5 and ABIDE-5 D3 tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_cross_dataset_a012_structure_v1 as engine  # noqa: E402
from Loss import criterion_lossv2  # noqa: E402
from Model.cme_dual_branch import CMEDualBranchModel  # noqa: E402
from Utils import SET_Random  # noqa: E402


EXPERIMENT_ID = "pre_shared_source_residual_v1"
BRANCH = "experiment/pre-shared-source-residual-v1"
BASE_COMMIT = "a23134cf23e3c95cb237931d92c327c001545a06"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_DIR = RESULT_DIR / "protocols"
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
FORMAL_DIR = RESULT_DIR / "formal"
TASK_IDS = ("tadpole_smci_pmci", "abide5_ads_cn")
FOLDS = tuple(range(10))
SEED = 0
SMOKE_EPOCHS = 3
FORMAL_EPOCHS = 400
MAX_EXPERIMENT_BYTES = 1_500_000_000
EXPECTED_TOTAL_PARAMETERS = {
    "tadpole_smci_pmci": 617197,
    "abide5_ads_cn": 383616,
}

IMPLEMENTATION_PATHS = (
    ".gitignore",
    "Model/cme_dual_branch.py",
    "scripts/run_pre_shared_source_residual_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocols/tadpole_smci_pmci.json",
    f"experiments/{EXPERIMENT_ID}/protocols/abide5_ads_cn.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
)
SMOKE_PATHS = (
    f"experiments/{EXPERIMENT_ID}/smoke/smoke_config.json",
    f"experiments/{EXPERIMENT_ID}/smoke/smoke_report.json",
    f"experiments/{EXPERIMENT_ID}/smoke/tadpole_smci_pmci_source_token_diagnostic.json",
    f"experiments/{EXPERIMENT_ID}/smoke/abide5_ads_cn_source_token_diagnostic.json",
)
LOCKED_DEPENDENCIES = (
    "Model/network.py",
    "Model/models.py",
    "Loss/loss_fn.py",
    "Utils/data_load.py",
    "Utils/graph_load.py",
    "Utils/utils.py",
    "config_re.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
)


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def current_head() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def protocol_paths() -> dict[str, Path]:
    return {task: PROTOCOL_DIR / f"{task}.json" for task in TASK_IDS}


def load_protocols() -> dict[str, dict[str, Any]]:
    protocols = {task: engine.read_json(path) for task, path in protocol_paths().items()}
    for task, protocol in protocols.items():
        require(protocol["task_id"] == task, f"Protocol owner changed: {task}")
        require(protocol["training"]["epochs"] == FORMAL_EPOCHS, f"Epoch lock changed: {task}")
        require(protocol["training"]["ema"] is False, f"EMA enabled: {task}")
        require(protocol["training"]["graph_use_graph"] is False, f"Graph enabled: {task}")
        require(len(protocol["modalities"]) == 5, f"Expected five modalities: {task}")
        require(protocol["private_source"] == "pre_shared", f"Private-source switch changed: {task}")
        require(protocol["prior_free_modality_prior"] is False, f"Prior-free state leaked: {task}")
        require(protocol["no_orthogonality"] is False, f"NO-ORTH state leaked: {task}")
        training = protocol["training"]
        require(training["optimizer"] == "Adam", f"Optimizer changed: {task}")
        require(training["scheduler"] == "CustomCosineAnnealingLR", f"Scheduler changed: {task}")
        require(training["best_epoch_rule"] == ["ACC", "ROC-AUC", "Macro-F1", "earliest"], f"Selection changed: {task}")
        require(training["full_batch_transductive"] is True, f"Full-batch protocol changed: {task}")
        expected = 0.0001 if task == "tadpole_smci_pmci" else 0.0
        require(math.isclose(float(training["loss_rate"]), expected, abs_tol=0.0), f"Historical loss rate changed: {task}")
        expected_values = {
            "tadpole_smci_pmci": {"lr": 0.0125, "weight_decay": 0.00025, "drop_rate": 0.67, "rank": 8, "adapter_lr_multiplier": 1.0, "input_noise_std": 0.05},
            "abide5_ads_cn": {"lr": 0.005, "weight_decay": 0.0005, "drop_rate": 0.45, "rank": 4, "adapter_lr_multiplier": 2.0, "input_noise_std": 0.0},
        }[task]
        for key in ("lr", "weight_decay", "drop_rate", "input_noise_std"):
            require(math.isclose(float(training[key]), expected_values[key], abs_tol=0.0), f"Historical {key} changed: {task}")
        for key in ("rank", "adapter_lr_multiplier"):
            require(math.isclose(float(protocol["reference"][key]), expected_values[key], abs_tol=0.0), f"Historical {key} changed: {task}")
        reference = protocol["reference"]
        commits = [reference["source_commit"], reference["result_commit"]]
        if "source_fix_commit" in reference:
            commits.append(reference["source_fix_commit"])
        for commit in commits:
            require(git("cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0, f"Reference commit missing: {task}:{commit}")
            require(git("merge-base", "--is-ancestor", commit, reference["result_commit"], check=False).returncode == 0, f"Reference commit lineage changed: {task}:{commit}")
    return protocols


def source_hashes() -> dict[str, str]:
    return {
        path: engine.file_sha256(ROOT / path)
        for path in (*IMPLEMENTATION_PATHS, *LOCKED_DEPENDENCIES)
    }


def reference_root(protocol: dict[str, Any]) -> Path:
    return ROOT.parent / protocol["reference"]["worktree"] / protocol["reference"]["experiment_dir"]


def reference_paths(protocol: dict[str, Any]) -> dict[str, Any]:
    root = reference_root(protocol)
    trial = root / protocol["reference"]["work_trial"]
    return {
        "root": root,
        "oof": root / protocol["reference"]["oof"],
        "fold_manifest": root / "fold_manifest.json",
        "trial_config": root / protocol["reference"]["trial_config"],
        "trial_summary": root / protocol["reference"]["trial_summary"],
        "checkpoints": [trial / f"fold_{fold:02d}" / "checkpoint_best.pt" for fold in FOLDS],
    }


def typed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in engine.read_csv_rows(path):
        rows.append(
            {
                "task_id": row["task_id"],
                "fold": int(row["fold"]),
                "subject_id": row["subject_id"],
                "original_csv_index": int(row["original_csv_index"]),
                "feature_sha256": row["feature_sha256"],
                "truth": int(row["truth"]),
                "prediction": int(row["prediction"]),
                "logit_0": float(row["logit_0"]),
                "logit_1": float(row["logit_1"]),
                "probability_0": float(row["probability_0"]),
                "probability_1": float(row["probability_1"]),
                "positive_probability": float(row["positive_probability"]),
            }
        )
    return rows


def reference_metrics(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth, _, probabilities = engine.rows_arrays(rows)
    metrics = engine.binary_metrics(truth, probabilities, protocol)
    reference = protocol["reference"]
    require(metrics["correct"] == reference["correct"], "Reference Correct changed")
    require(metrics["confusion_matrix"] == reference["confusion_matrix"], "Reference confusion changed")
    require(metrics["predicted_counts"] == reference["predicted_counts"], "Reference prediction counts changed")
    for metric in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe"):
        require(math.isclose(metrics[metric], reference[metric], abs_tol=5e-7), f"Reference {metric} changed")
    return metrics


def total_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def private_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    )


def build_model(
    context: dict[str, Any], private_source: str = "pre_shared", seed: int = SEED
) -> CMEDualBranchModel:
    SET_Random(seed)
    rank = int(context["protocol"]["reference"]["rank"])
    model = CMEDualBranchModel(
        **engine.model_kwargs(context),
        cme_arm="c1",
        adapter_rank=rank,
        router_hidden=16,
        modality_embedding_dim=8,
        private_source=private_source,
    ).to(context["device"])
    require(len(model.private_adapters) == len(context["protocol"]["modalities"]), "Adapter count changed")
    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary query/OVR schema changed")
    require(total_parameter_count(model) == EXPECTED_TOTAL_PARAMETERS[context["task_id"]], "Total parameter count changed")
    require(model.private_source == private_source, "Private source construction changed")
    require(model.modal_gate_logit.requires_grad, "Historical modality gate was frozen")
    require(bool((model._modal_noise_std >= 0).all()), "Historical noise buffer invalid")
    return model


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def make_training_objects(
    context: dict[str, Any], seed: int = SEED
) -> tuple[CMEDualBranchModel, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    model = build_model(context, "pre_shared", seed)
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(
        context["dataset_dict"],
        context["device"],
        rate=float(training["loss_rate"]),
        label_smoothing=float(training["label_smoothing"]),
    )
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    adapters = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
    common = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
    expected_adapter_tensors = 4 * len(context["protocol"]["modalities"])
    require(len(adapters) == expected_adapter_tensors, "Adapter tensor count changed")
    all_ids = {id(parameter) for _, parameter in named}
    adapter_ids = {id(parameter) for _, parameter in adapters}
    common_ids = {id(parameter) for _, parameter in common}
    require(not adapter_ids & common_ids and adapter_ids | common_ids == all_ids, "Optimizer partition changed")
    multiplier = float(context["protocol"]["reference"]["adapter_lr_multiplier"])
    base_lr = float(training["lr"])
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter for _, parameter in common], "lr": base_lr, "weight_decay": float(training["weight_decay"]), "group_name": "common"},
            {"params": [parameter for _, parameter in adapters], "lr": base_lr * multiplier, "weight_decay": float(training["weight_decay"]), "group_name": "private"},
        ]
    )
    scheduler = engine.RatioPreservingCustomCosineAnnealingLR(
        optimizer,
        T_max=int(training["scheduler_t_max"]),
        eta_min=float(training["scheduler_eta_min"]),
        multiplier=multiplier,
    )
    require(id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer), "Historical gate missing from optimizer")
    audit = {
        "total_parameter_count": total_parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "frozen_parameter_count": total_parameter_count(model) - trainable_parameter_count(model),
        "private_parameter_count": private_parameter_count(model),
        "common_parameter_tensors": len(common),
        "adapter_parameter_tensors": len(adapters),
        "legacy_gate_parameter_count": int(model.modal_gate_logit.numel()),
        "historical_gate_trainable": bool(model.modal_gate_logit.requires_grad),
        "historical_gate_in_optimizer": id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer),
        "new_parameter_count": 0,
        "base_lr": base_lr,
        "adapter_lr": base_lr * multiplier,
        "adapter_lr_multiplier": multiplier,
    }
    return model, criterion, optimizer, scheduler, audit


def state_max_diff(left: torch.nn.Module, right: torch.nn.Module, exclude: set[str] | None = None) -> float:
    excluded = exclude or set()
    left_state, right_state = left.state_dict(), right.state_dict()
    require(left_state.keys() == right_state.keys(), "Model state schema changed")
    values = []
    for name in left_state:
        if name in excluded:
            continue
        require(left_state[name].shape == right_state[name].shape and left_state[name].dtype == right_state[name].dtype, f"State schema changed: {name}")
        values.append(float((left_state[name].detach().cpu() - right_state[name].detach().cpu()).abs().max()))
    return max(values, default=0.0)


def initialization_audit(context: dict[str, Any]) -> dict[str, Any]:
    historical = build_model(context, "post_shared")
    ps_spr = build_model(context, "pre_shared")
    full_diff = state_max_diff(historical, ps_spr)
    require(full_diff == 0.0, "Seeded initial model state changed")
    historical.eval()
    ps_spr.eval()
    with torch.no_grad():
        old_logits = historical(context["dataset_data"]["Feature"])[0]
        old_outputs = historical(context["dataset_data"]["Feature"], return_intermediates=True)
        new_outputs = ps_spr(context["dataset_data"]["Feature"], return_intermediates=True)
        old_logits, new_logits = old_outputs[0], new_outputs[0]
        old_i, new_i = old_outputs[3], new_outputs[3]
    logits_diff = float((old_logits - new_logits).abs().max().cpu())
    historical_named = [(name, tuple(parameter.shape), parameter.requires_grad) for name, parameter in historical.named_parameters()]
    ps_spr_named = [(name, tuple(parameter.shape), parameter.requires_grad) for name, parameter in ps_spr.named_parameters()]
    historical_groups = {
        "common": [name for name, _shape, trainable in historical_named if trainable and not name.startswith("private_adapters.")],
        "private": [name for name, _shape, trainable in historical_named if trainable and name.startswith("private_adapters.")],
    }
    ps_spr_groups = {
        "common": [name for name, _shape, trainable in ps_spr_named if trainable and not name.startswith("private_adapters.")],
        "private": [name for name, _shape, trainable in ps_spr_named if trainable and name.startswith("private_adapters.")],
    }
    audit = {
        "all_state_initialization_max_abs_diff": full_diff,
        "common_trainable_initialization_max_abs_diff": full_diff,
        "adapter_state_initialization_max_abs_diff": max(
            float((historical.state_dict()[name] - ps_spr.state_dict()[name]).abs().max().cpu())
            for name in historical.state_dict() if name.startswith("private_adapters.")
        ),
        "step0_logits_max_abs_diff": logits_diff,
        "step0_probability_max_abs_diff": float((torch.softmax(old_logits, -1) - torch.softmax(new_logits, -1)).abs().max().cpu()),
        "step0_private_residual_max_abs": float(new_i["private_residuals"].abs().max().cpu()),
        "step0_final_category_token_max_abs_diff": float((old_i["category_token_streams"] - new_i["category_token_streams"]).abs().max().cpu()),
        "post_adapter_input_matches_H_max_abs_diff": float((old_i["private_adapter_inputs"] - old_i["modal_tokens_post_transformer"]).abs().max().cpu()),
        "pre_adapter_input_matches_E_max_abs_diff": float((new_i["private_adapter_inputs"] - new_i["modal_tokens_pre_transformer"]).abs().max().cpu()),
        "only_forward_difference": "private adapter input: H_m versus E_m",
        "parameter_key_shape_trainability_identical": historical_named == ps_spr_named,
        "optimizer_parameter_groups_identical": historical_groups == ps_spr_groups,
        "scheduler_class_and_config_identical": True,
        "scheduler_class": "RatioPreservingCustomCosineAnnealingLR",
        "scheduler_config": {
            "T_max": int(context["protocol"]["training"]["scheduler_t_max"]),
            "eta_min": float(context["protocol"]["training"]["scheduler_eta_min"]),
            "adapter_lr_multiplier": float(context["protocol"]["reference"]["adapter_lr_multiplier"]),
        },
    }
    require(logits_diff < 1e-7, "Step-0 logits changed")
    require(audit["step0_probability_max_abs_diff"] < 1e-7, "Step-0 probabilities changed")
    require(audit["step0_private_residual_max_abs"] == 0.0, "Zero-output adapter initialization changed")
    require(audit["step0_final_category_token_max_abs_diff"] < 1e-7, "Step-0 category tokens changed")
    require(audit["post_adapter_input_matches_H_max_abs_diff"] == 0.0, "Historical adapter input is not H_m")
    require(audit["pre_adapter_input_matches_E_max_abs_diff"] == 0.0, "PS-SPR adapter input is not E_m")
    require(audit["parameter_key_shape_trainability_identical"], "Parameter schema changed")
    require(audit["optimizer_parameter_groups_identical"], "Optimizer groups changed")
    del historical, ps_spr
    return audit


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("best_model", "model_state", "model"):
        value = payload.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(item) for item in value.values()):
            return value
    raise InvariantError("Checkpoint model state missing")


def validate_reference_checkpoint(context: dict[str, Any], path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Reference checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, "post_shared")
    state = checkpoint_state(payload)
    model.load_state_dict(state, strict=True)
    audit = {
        "path": path.as_posix(),
        "sha256": engine.file_sha256(path),
        "best_epoch": int(payload.get("best_epoch", -1)),
        "state_tensor_count": len(state),
    }
    del model
    return audit


def reference_file_hash(path: Path, canonical_sha256: str) -> dict[str, Any]:
    """Lock both the checkout bytes and the canonical LF Git-blob bytes."""
    require(path.is_file(), f"Reference file missing: {path}")
    checkout = engine.file_sha256(path)
    canonical = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    require(canonical == canonical_sha256, f"Reference canonical SHA changed: {path}")
    return {
        "path": path.as_posix(),
        "checkout_sha256": checkout,
        "canonical_lf_sha256": canonical,
    }


def reference_summary(protocol: dict[str, Any]) -> dict[str, Any]:
    path = reference_paths(protocol)["trial_summary"]
    require(engine.file_sha256(path) == protocol["reference"]["trial_summary_sha256"], "Reference summary SHA changed")
    value = engine.read_json(path)
    reference = protocol["reference"]
    require(value["trial_id"] == reference["trial_id"], "Reference summary owner changed")
    require(int(value["parameter_count"]) == EXPECTED_TOTAL_PARAMETERS[protocol["task_id"]], "Reference parameter count changed")
    require(int(value["metrics"]["correct"]) == int(reference["correct"]), "Reference summary Correct changed")
    return value


def experiment_config_payload() -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "tasks": list(TASK_IDS),
        "folds": list(FOLDS),
        "seed": SEED,
        "smoke_epochs": SMOKE_EPOCHS,
        "formal_epochs": FORMAL_EPOCHS,
        "model_switch": {"private_source": "pre_shared"},
        "reference_switch": {"private_source": "post_shared"},
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
        "residual_formula": "Z_m = H_m + A_m(E_m)",
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "single_model": True,
        "ema": False,
        "ensemble": False,
        "search": False,
        "decision_tokens": ["PS_SPR_PASS", "PS_SPR_NEAR", "PS_SPR_STOP"],
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def make_inspect_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    free = shutil.disk_usage(ROOT).free
    require(free >= 4_000_000_000, f"D drive free space below 4 GB: {free}")
    protocols = load_protocols()
    task_payload: dict[str, Any] = {}
    fold_tasks: dict[str, Any] = {}
    for task_id, protocol in protocols.items():
        context = engine.build_context(protocol, torch.device("cpu"))
        paths = reference_paths(protocol)
        require(all(paths[name].is_file() for name in ("oof", "fold_manifest", "trial_config", "trial_summary")), f"Reference evidence missing: {task_id}")
        reference_oof_hash = reference_file_hash(paths["oof"], protocol["reference"]["oof_sha256"])
        reference_fold_hash = reference_file_hash(paths["fold_manifest"], protocol["reference"]["fold_manifest_sha256"])
        require(engine.file_sha256(paths["trial_config"]) == protocol["reference"]["trial_config_sha256"], f"Reference config SHA changed: {task_id}")
        require(engine.file_sha256(paths["trial_summary"]) == protocol["reference"]["trial_summary_sha256"], f"Reference summary SHA changed: {task_id}")
        locked_reference_summary = reference_summary(protocol)
        rows = typed_rows(paths["oof"])
        require(len(rows) == protocol["sample_count"] and len({row["subject_id"] for row in rows}) == len(rows), "Reference OOF coverage changed")
        metrics = reference_metrics(protocol, rows)
        reference_fold = engine.read_json(paths["fold_manifest"])["fold_manifest"]
        require(reference_fold == context["fold_manifest"], f"Reference fold manifest changed: {task_id}")
        checkpoints = [validate_reference_checkpoint(context, path) for path in paths["checkpoints"]]
        historical = build_model(context, "post_shared")
        ps_spr = build_model(context, "pre_shared")
        modalities = [item["name"] for item in protocol["modalities"]]
        historical.eval()
        ps_spr.eval()
        with torch.no_grad():
            old_out = historical(context["dataset_data"]["Feature"], return_intermediates=True)
            new_out = ps_spr(context["dataset_data"]["Feature"], return_intermediates=True)
        old_i, new_i = old_out[3], new_out[3]
        gate = torch.sigmoid(historical.modal_gate_logit.detach().cpu())
        noise = historical._modal_noise_std.detach().cpu() * float(historical.noise_scale)
        init = initialization_audit(context)
        task_payload[task_id] = {
            "dataset": protocol["dataset"],
            "task": protocol["task"],
            "sample_count": protocol["sample_count"],
            "class_names": protocol["class_names"],
            "class_counts": protocol["class_counts"],
            "positive_class": protocol["positive_class"],
            "positive_index": protocol["positive_index"],
            "modalities": protocol["modalities"],
            "reference": protocol["reference"],
            "reference_oof_hash": reference_oof_hash,
            "reference_fold_hash": reference_fold_hash,
            "reference_trial_config_sha256": engine.file_sha256(paths["trial_config"]),
            "reference_trial_summary_sha256": engine.file_sha256(paths["trial_summary"]),
            "reference_timing_and_parameters": {
                "parameter_count": int(locked_reference_summary["parameter_count"]),
                "training_time_seconds": float(locked_reference_summary["training_time_seconds"]),
                "inference_time_seconds": float(locked_reference_summary["inference_time_seconds"]),
            },
            "reference_metrics_recomputed": metrics,
            "reference_checkpoint_audit": checkpoints,
            "reference_fold_manifest_sha256": context["fold_manifest"]["sha256"],
            "historical_gate": {"formula": "sigmoid(modal_gate_logit)", "trainable": True, "effective_by_modality": dict(zip(modalities, gate.tolist()))},
            "historical_noise": {"formula": "_modal_noise_std * noise_scale", "effective_std_by_modality": dict(zip(modalities, noise.tolist()))},
            "information_flow": {
                "modal_encoder_output_shape": list(old_i["modal_tokens_pre_transformer"].shape),
                "pre_shared_token": "last token after all historical single-modality processing and immediately before first cross-modal shared-transformer block",
                "pre_shared_shape": list(new_i["modal_tokens_pre_transformer"].shape),
                "shared_transformer_input_shape": list(new_i["modal_tokens_pre_transformer"].shape),
                "shared_transformer_output_shape": list(new_i["modal_tokens_post_transformer"].shape),
                "post_shared_adapter_input_shape": list(old_i["private_adapter_inputs"].shape),
                "pre_shared_adapter_input_shape": list(new_i["private_adapter_inputs"].shape),
                "residual_addback_base_shape": list(new_i["private_residual_addback_base"].shape),
                "category_query_input_shape": list(new_i["category_token_streams"].shape),
                "global_path": "historical Global_Message(X_gated), unchanged and independent of private_source",
            },
            "initialization_audit": init,
            "parameter_audit": {
                "old_total": total_parameter_count(historical),
                "old_trainable": trainable_parameter_count(historical),
                "new_total": total_parameter_count(ps_spr),
                "new_trainable": trainable_parameter_count(ps_spr),
                "trainable_delta": trainable_parameter_count(ps_spr) - trainable_parameter_count(historical),
            },
        }
        fold_tasks[task_id] = context["fold_manifest"]
        del historical, ps_spr
    fold_core = {"tasks": fold_tasks}
    fold_payload = {**fold_core, "sha256": engine.payload_sha256(fold_core)}
    core = {
        "experiment": EXPERIMENT_ID,
        "tasks": task_payload,
        "disk_free_bytes": int(free),
        "disk_minimum_bytes": 4_000_000_000,
        "only_allowed_change": "P_m=A_m(H_m) -> P_m=A_m(E_m); Z_m=H_m+P_m unchanged",
        "historical_gate_noise_restored": True,
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
        "references_reused_without_retraining": True,
        "fold_manifest_sha256": fold_payload["sha256"],
    }
    return {**core, "sha256": engine.payload_sha256(core)}, fold_payload


def run_inspect() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    engine.atomic_write_json(CONFIG_PATH, experiment_config_payload())
    inspect, folds = make_inspect_payload()
    engine.atomic_write_json(INSPECT_PATH, inspect)
    engine.atomic_write_json(FOLD_MANIFEST_PATH, folds)
    print(json.dumps({"inspect": inspect["sha256"], "fold_manifest": folds["sha256"], "free_bytes": inspect["disk_free_bytes"]}, indent=2))


def source_gate(include_smoke: bool) -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked source is dirty")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index is dirty")
    head = current_head()
    require(head != BASE_COMMIT, "Implementation must be committed before smoke")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Base is not an ancestor")
    expected = set(IMPLEMENTATION_PATHS) | (set(SMOKE_PATHS) if include_smoke else set())
    changed = {
        line.strip().replace("\\", "/")
        for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines()
        if line.strip()
    }
    require(changed == expected, f"Committed source scope changed: {sorted(changed)}")
    for relative in expected:
        require(git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0, f"Untracked source: {relative}")
        require((ROOT / relative).is_file(), f"Source file missing: {relative}")
    for relative in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", relative, check=False).returncode == 0, f"Locked dependency changed: {relative}")
    require(engine.read_json(CONFIG_PATH) == experiment_config_payload(), "Experiment config drifted")
    expected_inspect, expected_folds = make_inspect_payload()
    saved_inspect = engine.read_json(INSPECT_PATH)
    saved_static = {key: value for key, value in saved_inspect.items() if key not in {"sha256", "disk_free_bytes"}}
    expected_static = {key: value for key, value in expected_inspect.items() if key not in {"sha256", "disk_free_bytes"}}
    require(saved_static == expected_static, "Inspect evidence drifted")
    require(engine.read_json(FOLD_MANIFEST_PATH) == expected_folds, "Fold manifest drifted")
    return head


def capture_target_gradients(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "private": "private_adapters.",
        "shared_modal": "shared_transformer.",
        "global": "Global_Message.",
        "query": "label_pools.",
        "difformer": "GCN.",
    }
    values: dict[str, float] = {}
    for group, prefix in groups.items():
        candidates = [
            float(parameter.grad.detach().abs().max().cpu())
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        values[group] = max(candidates, default=0.0)
    return values


def train_epoch(
    model: CMEDualBranchModel,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    context: dict[str, Any],
    train_mask: torch.Tensor,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, embeddings, auxiliary = model(context["dataset_data"]["Feature"])
    loss = criterion(logits, context["dataset_data"]["Label"], train_mask, embeddings, auxiliary)
    require(bool(torch.isfinite(loss)), "Training loss is non-finite")
    loss.backward()
    adapter_gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid adapter gradient: {name}")
            adapter_gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    require(model.modal_gate_logit.grad is not None and bool(torch.isfinite(model.modal_gate_logit.grad).all()), "Historical gate gradient missing")
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Model gradient is non-finite")
    target_gradients = capture_target_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
    optimizer.step()
    return float(loss.detach().cpu()), adapter_gradients, target_gradients


def reference_fold_replay(context: dict[str, Any], fold: int = 0) -> dict[str, Any]:
    path = reference_paths(context["protocol"])["checkpoints"][fold]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, "post_shared")
    model.load_state_dict(checkpoint_state(payload), strict=True)
    _, test_mask, _ = engine.fold_positions(context, fold)
    logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    rows = sorted(
        [row for row in typed_rows(reference_paths(context["protocol"])["oof"]) if row["fold"] == fold],
        key=lambda row: row["original_csv_index"],
    )
    expected = sorted(context["fold_manifest"]["folds"][fold]["test_rows"], key=lambda row: row["original_csv_index"])
    order = np.argsort([int(row["original_csv_index"]) for row in context["fold_manifest"]["folds"][fold]["test_rows"]])
    logits = logits[order]
    probabilities = probabilities[order]
    _, saved_logits, saved_probabilities = engine.rows_arrays(rows)
    require([row["subject_id"] for row in rows] == [row["subject_id"] for row in expected], "Reference subjects changed")
    logits_diff = float(np.max(np.abs(logits - saved_logits)))
    probability_diff = float(np.max(np.abs(probabilities - saved_probabilities)))
    require(logits_diff <= 1e-6 and probability_diff <= 1e-6, "Reference checkpoint replay changed")
    del model
    return {
        "fold": fold,
        "checkpoint_sha256": engine.file_sha256(path),
        "logits_max_abs_diff": logits_diff,
        "probabilities_max_abs_diff": probability_diff,
    }


def source_preserving_diagnostic(
    context: dict[str, Any], model: CMEDualBranchModel
) -> dict[str, Any]:
    """Isolated graph-origin diagnostic; it never calls backward or optimizer.step."""
    model.eval()
    features = context["dataset_data"]["Feature"]
    outputs = model(features, return_intermediates=True)
    intermediates = outputs[3]
    source = intermediates["modal_tokens_pre_transformer"]
    shared = intermediates["modal_tokens_post_transformer"]
    require(source.requires_grad, "Pre-shared token lost autograd connection")
    require(tuple(source.shape) == tuple(shared.shape), "Pre/post token shapes differ")
    base_private = torch.stack(
        [adapter(source[:, index]) for index, adapter in enumerate(model.private_adapters)], dim=1
    )
    perturbations: list[dict[str, Any]] = []
    for modality in range(source.size(1)):
        other = (modality + 1) % source.size(1)
        perturbation = torch.zeros_like(source)
        perturbation[:, other] = torch.linspace(
            1e-3, 2e-3, source.size(0), device=source.device, dtype=source.dtype
        ).unsqueeze(-1)
        perturbed_private = torch.stack(
            [adapter((source + perturbation)[:, index]) for index, adapter in enumerate(model.private_adapters)], dim=1
        )
        pm_diff = float((base_private[:, modality] - perturbed_private[:, modality]).abs().max().detach().cpu())
        pn_diff = float((base_private[:, other] - perturbed_private[:, other]).abs().max().detach().cpu())
        require(pm_diff < 1e-7, "Changing E_n directly changed P_m")
        require(pn_diff > 0.0, "Changing E_n did not change trained P_n")
        perturbations.append({"modality_m": modality, "modality_n": other, "p_m_max_abs_diff": pm_diff, "p_n_max_abs_diff": pn_diff})

    encoder_gradient_by_modality: dict[str, float] = {}
    cross_encoder_gradient_by_modality: dict[str, float] = {}
    modality_names = [entry["name"] for entry in context["protocol"]["modalities"]]
    modal_projections = list(model.modal_token_encoder.modal_proj)
    for modality, name in enumerate(modality_names):
        own_params = [parameter for parameter in modal_projections[modality].parameters() if parameter.requires_grad]
        other_params = [
            parameter
            for other_index, projection in enumerate(modal_projections)
            if other_index != modality
            for parameter in projection.parameters()
            if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            base_private[:, modality].square().mean(),
            own_params + other_params,
            retain_graph=True,
            allow_unused=True,
        )
        own_gradients = gradients[: len(own_params)]
        other_gradients = gradients[len(own_params) :]
        own_max = max((float(gradient.abs().max().detach().cpu()) for gradient in own_gradients if gradient is not None), default=0.0)
        other_max = max((float(gradient.abs().max().detach().cpu()) for gradient in other_gradients if gradient is not None), default=0.0)
        require(own_max > 0.0 and math.isfinite(own_max), f"Private path did not reach corresponding Modal Encoder: {name}")
        require(other_max == 0.0, f"Private P_m reached a non-corresponding Modal Encoder: {name}")
        encoder_gradient_by_modality[name] = own_max
        cross_encoder_gradient_by_modality[name] = other_max

    private_scalar = base_private.square().mean()
    shared_params = [parameter for parameter in model.shared_transformer.parameters() if parameter.requires_grad]
    shared_grads = torch.autograd.grad(private_scalar, shared_params, retain_graph=True, allow_unused=True)
    shared_max = max((float(gradient.abs().max().detach().cpu()) for gradient in shared_grads if gradient is not None), default=0.0)
    require(shared_max == 0.0, "Private-only path unexpectedly traversed Shared Transformer")
    main_grads = torch.autograd.grad(outputs[0].square().mean(), shared_params, retain_graph=False, allow_unused=True)
    shared_main_gradient = max((float(gradient.abs().max().detach().cpu()) for gradient in main_grads if gradient is not None), default=0.0)
    require(shared_main_gradient > 0.0, "Shared main path gradient missing")
    return {
        "per_modality_perturbations": perturbations,
        "p_m_max_abs_diff_after_e_n_perturbation": max(row["p_m_max_abs_diff"] for row in perturbations),
        "p_n_min_abs_diff_after_e_n_perturbation": min(row["p_n_max_abs_diff"] for row in perturbations),
        "modal_encoder_private_only_gradient_max": max(encoder_gradient_by_modality.values()),
        "modal_encoder_private_only_gradient_by_modality": encoder_gradient_by_modality,
        "non_corresponding_modal_encoder_private_only_gradient_by_modality": cross_encoder_gradient_by_modality,
        "shared_transformer_private_only_gradient_max": shared_max,
        "pre_shared_requires_grad": bool(source.requires_grad),
        "private_path_detached": False,
        "private_path_traverses_shared_transformer": False,
        "shared_main_path_traverses_shared_transformer": True,
        "shared_main_path_gradient_max": shared_main_gradient,
        "interpretation": "graph-origin check only; it does not establish statistical independence",
    }


def ps_spr_diagnostics(
    context: dict[str, Any],
    model: CMEDualBranchModel,
    test_mask: torch.Tensor,
    cumulative_gradient: dict[str, float],
    initial_private: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    features = context["dataset_data"]["Feature"]
    with torch.no_grad():
        logits, _embeddings, _auxiliary, values = model(features, return_intermediates=True)
    source = values["modal_tokens_pre_transformer"][test_mask]
    shared = values["modal_tokens_post_transformer"][test_mask]
    residual = values["private_residuals"][test_mask]
    require(tuple(source.shape) == tuple(shared.shape) == tuple(residual.shape), "Diagnostic token shapes changed")
    source_norm = source.norm(dim=-1)
    shared_norm = shared.norm(dim=-1)
    residual_norm = residual.norm(dim=-1)
    private_source_ratio = residual_norm / source_norm.clamp_min(1e-8)
    private_shared_ratio = residual_norm / shared_norm.clamp_min(1e-8)
    modalities = [entry["name"] for entry in context["protocol"]["modalities"]]

    saved_up = [
        (adapter.up.weight.detach().clone(), adapter.up.bias.detach().clone())
        for adapter in model.private_adapters
    ]
    try:
        with torch.no_grad():
            for adapter in model.private_adapters:
                adapter.up.weight.zero_()
                adapter.up.bias.zero_()
            off_logits = engine.infer(model, features)[0][test_mask]
    finally:
        with torch.no_grad():
            for adapter, (weight, bias) in zip(model.private_adapters, saved_up):
                adapter.up.weight.copy_(weight)
                adapter.up.bias.copy_(bias)
    probability = torch.softmax(logits[test_mask], dim=-1)
    off_probability = torch.softmax(off_logits, dim=-1)
    probability_change = (probability - off_probability).abs()
    argmax_changes = int((probability.argmax(dim=-1) != off_probability.argmax(dim=-1)).sum().cpu())

    parameter_delta: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(name in initial_private, f"Initial private state missing: {name}")
            parameter_delta[name] = float((parameter.detach().cpu() - initial_private[name]).abs().max())
    residual_by_modality = dict(zip(modalities, residual_norm.mean(dim=0).cpu().tolist()))
    private_trained = bool(
        cumulative_gradient
        and set(cumulative_gradient) == set(parameter_delta)
        and all(math.isfinite(value) and value > 0.0 for value in cumulative_gradient.values())
        and all(math.isfinite(value) and value > 0.0 for value in parameter_delta.values())
        and min(residual_by_modality.values(), default=0.0) >= 1e-6
    )
    gradient_by_modality = {
        modality: max(
            value
            for name, value in cumulative_gradient.items()
            if name.startswith(f"private_adapters.{index}.")
        )
        for index, modality in enumerate(modalities)
    }
    return {
        "pre_shared_shape": list(source.shape),
        "post_shared_shape": list(shared.shape),
        "source_mean_norm_by_modality": dict(zip(modalities, source_norm.mean(dim=0).cpu().tolist())),
        "shared_mean_norm_by_modality": dict(zip(modalities, shared_norm.mean(dim=0).cpu().tolist())),
        "residual_mean_norm_by_modality": residual_by_modality,
        "private_source_ratio_mean_by_modality": dict(zip(modalities, private_source_ratio.mean(dim=0).cpu().tolist())),
        "private_shared_ratio_mean_by_modality": dict(zip(modalities, private_shared_ratio.mean(dim=0).cpu().tolist())),
        "private_shared_ratio_max_by_modality": dict(zip(modalities, private_shared_ratio.amax(dim=0).cpu().tolist())),
        "adapter_max_gradient_by_modality": gradient_by_modality,
        "adapter_cumulative_max_gradient": dict(cumulative_gradient),
        "adapter_parameter_max_delta": parameter_delta,
        "category_global_cosine_mean": float(torch.nn.functional.cosine_similarity(values["Y"][test_mask], values["G"][test_mask], dim=-1).mean().cpu()),
        "private_off_probability_mean_abs_change": float(probability_change.mean().cpu()),
        "private_off_probability_max_abs_change": float(probability_change.max().cpu()),
        "private_off_argmax_change_count": argmax_changes,
        "private_trained": private_trained,
        "private_collapse": not private_trained,
    }


def run_smoke(device_text: str) -> None:
    source_commit = source_gate(include_smoke=False)
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_text)
    reports: dict[str, Any] = {}
    for task_id, protocol in load_protocols().items():
        context = engine.build_context(protocol, device)
        context["fold_manifest"] = engine.read_json(FOLD_MANIFEST_PATH)["tasks"][task_id]
        replay = reference_fold_replay(context, 0)
        init_audit = initialization_audit(context)
        model, criterion, optimizer, scheduler, object_audit = make_training_objects(context)
        train_mask, test_mask, _ = engine.fold_positions(context, 0)
        initial_private = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
        # The historical gate is trainable, so its value is expected to change
        # during the three smoke updates.  Lock the restored historical
        # initialization before training; after training, audit participation,
        # finite gradients, and finite values instead of requiring immutability.
        initial_effective_gate = torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist()
        initial_effective_noise = (
            model._modal_noise_std.detach() * float(model.noise_scale)
        ).cpu().tolist()
        cumulative_adapter = {name: 0.0 for name in initial_private}
        cumulative_targets = {name: 0.0 for name in ("private", "shared_modal", "global", "query", "difformer")}
        losses = []
        for _epoch in range(1, SMOKE_EPOCHS + 1):
            loss, adapter_gradients, target_gradients = train_epoch(model, criterion, optimizer, context, train_mask)
            scheduler.step()
            losses.append(loss)
            for name, value in adapter_gradients.items():
                cumulative_adapter[name] = max(cumulative_adapter[name], value)
            for name, value in target_gradients.items():
                cumulative_targets[name] = max(cumulative_targets[name], value)
        require(all(math.isfinite(value) and value > 0.0 for value in cumulative_adapter.values()), "Smoke adapter gradients invalid")
        require(all(math.isfinite(value) and value > 0.0 for value in cumulative_targets.values()), "Smoke target gradients invalid")
        private_delta = {
            name: float((parameter.detach().cpu() - initial_private[name]).abs().max())
            for name, parameter in model.named_parameters()
            if name in initial_private
        }
        require(all(math.isfinite(value) and value > 0.0 for value in private_delta.values()), "Smoke private parameters did not update")
        source_diagnostic = source_preserving_diagnostic(context, model)
        logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask]
        probabilities = torch.softmax(logits, dim=-1)
        simplex = float((probabilities.sum(dim=1) - 1.0).abs().max().cpu())
        require(simplex <= 1e-6 and bool(torch.isfinite(probabilities).all()), "Smoke probability simplex failed")
        checkpoint_path = SMOKE_DIR / f"{task_id}_checkpoint_roundtrip.pt"
        payload = {
            "schema": 1,
            "source_commit": source_commit,
            "task_id": task_id,
            "fold": 0,
            "epoch": SMOKE_EPOCHS,
            "model": engine.clone_cpu_state(model),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
        }
        engine.atomic_torch_save(checkpoint_path, payload)
        fresh, _criterion, fresh_optimizer, fresh_scheduler, _audit = make_training_objects(context)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fresh.load_state_dict(loaded["model"], strict=True)
        fresh_optimizer.load_state_dict(loaded["optimizer"])
        fresh_scheduler.load_state_dict(loaded["scheduler"])
        replay_logits = engine.infer(fresh, context["dataset_data"]["Feature"])[0][test_mask]
        reload_diff = float((replay_logits.detach().cpu() - logits.detach().cpu()).abs().max())
        require(reload_diff <= 1e-6 and int(fresh_scheduler.last_epoch) == SMOKE_EPOCHS, "Smoke checkpoint reload failed")
        inspect_task = engine.read_json(INSPECT_PATH)["tasks"][task_id]
        expected_gate = list(inspect_task["historical_gate"]["effective_by_modality"].values())
        expected_noise = list(inspect_task["historical_noise"]["effective_std_by_modality"].values())
        effective_gate = torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist()
        effective_noise = (model._modal_noise_std.detach() * float(model.noise_scale)).cpu().tolist()
        gate_gradient_is_finite = bool(
            model.modal_gate_logit.grad is not None
            and torch.isfinite(model.modal_gate_logit.grad).all()
        )
        gate_values_are_finite = all(math.isfinite(value) for value in effective_gate)
        gate_in_optimizer = id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer)
        reports[task_id] = {
            "reference_checkpoint_replay": replay,
            "initialization_audit": init_audit,
            "source_preserving_diagnostic": source_diagnostic,
            "object_audit": object_audit,
            "losses": losses,
            "adapter_cumulative_max_gradient": cumulative_adapter,
            "target_group_cumulative_max_gradient": cumulative_targets,
            "adapter_parameter_max_delta": private_delta,
            "initial_effective_gate": initial_effective_gate,
            "effective_gate": effective_gate,
            "initial_effective_noise_std": initial_effective_noise,
            "effective_noise_std": effective_noise,
            "historical_gate_matches_inspect": initial_effective_gate == expected_gate,
            "historical_noise_matches_inspect": (
                initial_effective_noise == expected_noise and effective_noise == expected_noise
            ),
            "historical_gate_gradient_is_finite": gate_gradient_is_finite,
            "historical_gate_values_are_finite": gate_values_are_finite,
            "historical_gate_in_optimizer": gate_in_optimizer,
            "optimizer_steps": SMOKE_EPOCHS,
            "scheduler_steps": SMOKE_EPOCHS,
            "probability_simplex_max_abs_error": simplex,
            "checkpoint_roundtrip_logits_max_abs_diff": reload_diff,
            "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        }
        engine.atomic_write_json(
            SMOKE_DIR / f"{task_id}_source_token_diagnostic.json",
            source_diagnostic,
        )
        del model, fresh, criterion, optimizer, fresh_optimizer, scheduler, fresh_scheduler
        torch.cuda.empty_cache()
    config_core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "device": device_text,
        "epochs": SMOKE_EPOCHS,
        "inspect_sha256": engine.file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": engine.file_sha256(FOLD_MANIFEST_PATH),
    }
    smoke_config = {**config_core, "sha256": engine.payload_sha256(config_core)}
    historical_gate_restored = all(
        report["historical_gate_matches_inspect"]
        and report["historical_gate_gradient_is_finite"]
        and report["historical_gate_values_are_finite"]
        and report["historical_gate_in_optimizer"]
        for report in reports.values()
    )
    historical_noise_restored = all(
        report["historical_noise_matches_inspect"] for report in reports.values()
    )
    source_preserving_all_passed = all(
        report["source_preserving_diagnostic"]["p_m_max_abs_diff_after_e_n_perturbation"] < 1e-7
        and report["source_preserving_diagnostic"]["p_n_min_abs_diff_after_e_n_perturbation"] > 0.0
        and report["source_preserving_diagnostic"]["modal_encoder_private_only_gradient_max"] > 0.0
        and report["source_preserving_diagnostic"]["shared_transformer_private_only_gradient_max"] == 0.0
        and report["source_preserving_diagnostic"]["shared_main_path_gradient_max"] > 0.0
        for report in reports.values()
    )
    formal_go = historical_gate_restored and historical_noise_restored and source_preserving_all_passed
    report_core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "smoke_config_sha256": smoke_config["sha256"],
        "tasks": reports,
        "source_token_diagnostic_sha256": {
            task_id: engine.file_sha256(SMOKE_DIR / f"{task_id}_source_token_diagnostic.json")
            for task_id in TASK_IDS
        },
        "historical_gate_restored": historical_gate_restored,
        "historical_noise_restored": historical_noise_restored,
        "source_preserving_all_passed": source_preserving_all_passed,
        "formal_go": formal_go,
    }
    smoke_report = {**report_core, "sha256": engine.payload_sha256(report_core)}
    engine.atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    engine.atomic_write_json(SMOKE_DIR / "smoke_report.json", smoke_report)
    print(json.dumps({"formal_go": formal_go, "source_commit": source_commit, "report_sha256": smoke_report["sha256"]}, indent=2))


def validate_smoke_and_source() -> tuple[str, dict[str, Any]]:
    source_commit = source_gate(include_smoke=True)
    config = engine.read_json(SMOKE_DIR / "smoke_config.json")
    report = engine.read_json(SMOKE_DIR / "smoke_report.json")
    require(config["sha256"] == engine.payload_sha256({key: value for key, value in config.items() if key != "sha256"}), "Smoke config digest changed")
    require(report["sha256"] == engine.payload_sha256({key: value for key, value in report.items() if key != "sha256"}), "Smoke report digest changed")
    implementation_commit = config["source_commit"]
    require(git("merge-base", "--is-ancestor", implementation_commit, source_commit, check=False).returncode == 0, "Smoke implementation is not an ancestor")
    require(report["source_commit"] == implementation_commit, "Smoke source changed")
    require(report["formal_go"] is True and report["historical_gate_restored"] and report["historical_noise_restored"] and report["source_preserving_all_passed"], "Smoke formal gate failed")
    return source_commit, report


def runtime_lock(source_commit: str, device_text: str) -> dict[str, Any]:
    protocols = load_protocols()
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "experiment_config_sha256": engine.file_sha256(CONFIG_PATH),
        "protocol_sha256": {task: engine.file_sha256(protocol_paths()[task]) for task in TASK_IDS},
        "inspect_sha256": engine.file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": engine.file_sha256(FOLD_MANIFEST_PATH),
        "smoke_config_sha256": engine.file_sha256(SMOKE_DIR / "smoke_config.json"),
        "smoke_report_sha256": engine.file_sha256(SMOKE_DIR / "smoke_report.json"),
        "tasks": list(TASK_IDS),
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs": FORMAL_EPOCHS,
        "device": device_text,
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "model_switch": {"private_source": "pre_shared"},
        "reference_switch": {"private_source": "post_shared"},
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
        "training": {task: protocols[task]["training"] for task in TASK_IDS},
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {
        "runtime_sha256": runtime["sha256"],
        "task_id": context["task_id"],
        "fold": int(fold),
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "private_source": "pre_shared",
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_paths(task_id: str, fold: int) -> dict[str, Path]:
    root = FORMAL_DIR / task_id / f"fold_{fold:02d}"
    return {
        "root": root,
        "resume": root / "resume.pt",
        "checkpoint": root / "checkpoint_best.pt",
        "oof": root / "oof_predictions.csv",
        "history": root / "epoch_metrics.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
    }


def experiment_size() -> int:
    return sum(path.stat().st_size for path in RESULT_DIR.rglob("*") if path.is_file())


def save_resume(
    path: Path,
    lock: dict[str, Any],
    model: CMEDualBranchModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best: dict[str, Any] | None,
    history: list[dict[str, Any]],
    cumulative_gradient: dict[str, float],
    initial_private: dict[str, torch.Tensor],
    elapsed_seconds: float,
) -> None:
    engine.atomic_torch_save(
        path,
        {
            "schema": 1,
            "lock": lock,
            "epoch": int(epoch),
            "optimizer_steps": int(epoch),
            "scheduler_steps": int(epoch),
            "model": engine.clone_cpu_state(model),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": engine.capture_rng_state(),
            "best": copy.deepcopy(best),
            "history": copy.deepcopy(history),
            "cumulative_gradient": dict(cumulative_gradient),
            "initial_private": {name: tensor.detach().cpu().clone() for name, tensor in initial_private.items()},
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def train_fold(
    context: dict[str, Any], fold: int, runtime: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context["task_id"], fold)
    if paths["complete"].is_file():
        return load_completed_fold(context, fold, runtime)
    paths["root"].mkdir(parents=True, exist_ok=True)
    known = {path.name for name, path in paths.items() if name != "root"}
    known |= {name + ".tmp" for name in known}
    unknown = [path for path in paths["root"].iterdir() if path.name not in known]
    require(not unknown, f"Unknown incomplete artifacts: {unknown}")
    lock = fold_lock(runtime, context, fold)
    model, criterion, optimizer, scheduler, object_audit = make_training_objects(context)
    train_mask, test_mask, _ = engine.fold_positions(context, fold)
    initial_private = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    cumulative_gradient = {name: 0.0 for name in initial_private}
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_epoch = 1
    resumed_from_epoch = 0
    elapsed_before = 0.0
    if paths["resume"].is_file():
        resumed = torch.load(paths["resume"], map_location="cpu", weights_only=False)
        require(resumed.get("schema") == 1 and resumed.get("lock") == lock, "Resume lock changed")
        epoch = int(resumed["epoch"])
        require(0 < epoch <= FORMAL_EPOCHS, "Resume epoch invalid")
        require(resumed["optimizer_steps"] == epoch and resumed["scheduler_steps"] == epoch, "Resume step count changed")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        require(int(scheduler.last_epoch) == epoch, "Resume scheduler epoch changed")
        scheduler.assert_ratio()
        engine.restore_rng_state(resumed["rng"])
        start_epoch = epoch + 1
        resumed_from_epoch = epoch
        best = resumed["best"]
        history = resumed["history"]
        cumulative_gradient = {name: float(value) for name, value in resumed["cumulative_gradient"].items()}
        initial_private = {name: tensor.detach().cpu().clone() for name, tensor in resumed["initial_private"].items()}
        elapsed_before = float(resumed["elapsed_seconds"])
        require([int(row["epoch"]) for row in history] == list(range(1, start_epoch)), "Resume history changed")
    started = time.perf_counter()
    labels = context["dataset_data"]["Label"]
    for epoch in range(start_epoch, FORMAL_EPOCHS + 1):
        loss, gradients, _targets = train_epoch(model, criterion, optimizer, context, train_mask)
        for name, value in gradients.items():
            cumulative_gradient[name] = max(cumulative_gradient.get(name, 0.0), value)
        logits_t = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu()
        probabilities_t = torch.softmax(logits_t, dim=-1)
        truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        logits = logits_t.numpy().astype(np.float64)
        probabilities = probabilities_t.numpy().astype(np.float64)
        metrics = engine.binary_metrics(truth, probabilities, context["protocol"])
        row = {
            "epoch": epoch,
            "loss": loss,
            "correct": metrics["correct"],
            "acc": metrics["acc"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "macro_f1": metrics["macro_f1"],
            "bacc": metrics["bacc"],
            "sen": metrics["sen"],
            "spe": metrics["spe"],
        }
        history.append(row)
        key = engine.selection_key(metrics, epoch)
        if best is None or key > tuple(best["selection_key"]):
            best = {
                "epoch": epoch,
                "selection_key": list(key),
                "metrics": metrics,
                "model": engine.clone_cpu_state(model),
                "test_logits": logits,
                "test_probabilities": probabilities,
                "test_truth": truth,
            }
        scheduler.step()
        elapsed = elapsed_before + time.perf_counter() - started
        if epoch % 20 == 0 or epoch == FORMAL_EPOCHS:
            save_resume(paths["resume"], lock, model, optimizer, scheduler, epoch, best, history, cumulative_gradient, initial_private, elapsed)
    require(best is not None and len(history) == FORMAL_EPOCHS, "Formal fold incomplete")
    model.load_state_dict(best["model"], strict=True)
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    inference_started = time.perf_counter()
    replay_logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - inference_started
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - best["test_logits"]))) <= 1e-6, "Best logits replay changed")
    require(float(np.max(np.abs(replay_probabilities - best["test_probabilities"]))) <= 1e-6, "Best probabilities replay changed")
    rows = engine.prediction_rows(context, fold, best["test_logits"], best["test_probabilities"], best["test_truth"])
    for row in rows:
        row["arm"] = "PS_SPR"
    diagnostics = ps_spr_diagnostics(context, model, test_mask, cumulative_gradient, initial_private)
    require(diagnostics["private_trained"] and not diagnostics["private_collapse"], "Private residual collapse")
    require(all(value > 0.0 and math.isfinite(value) for value in diagnostics["adapter_parameter_max_delta"].values()), "Adapter parameter update audit failed")
    elapsed_total = elapsed_before + time.perf_counter() - started
    checkpoint = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "fold": fold,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "best_model": best["model"],
        "formal_epochs": FORMAL_EPOCHS,
        "private_source": "pre_shared",
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
    }
    engine.atomic_torch_save(paths["checkpoint"], checkpoint)
    engine.atomic_write_csv(paths["oof"], rows)
    engine.atomic_write_csv(paths["history"], history)
    summary = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "fold": fold,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "object_audit": object_audit,
        "diagnostics": diagnostics,
        "effective_gate": torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist(),
        "effective_noise_std": (model._modal_noise_std.detach() * float(model.noise_scale)).cpu().tolist(),
        "historical_gate_trainable": model.modal_gate_logit.requires_grad,
        "historical_gate_in_optimizer": id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer),
        "optimizer_steps": FORMAL_EPOCHS,
        "scheduler_steps": FORMAL_EPOCHS,
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds": elapsed_total,
        "inference_seconds_full_graph": inference_seconds,
        "oof_count": len(rows),
        "loss_first": float(history[0]["loss"]),
        "loss_last": float(history[-1]["loss"]),
    }
    engine.atomic_write_json(paths["summary"], summary)
    marker_core = {
        "complete": True,
        "lock": lock,
        "checkpoint_sha256": engine.file_sha256(paths["checkpoint"]),
        "oof_sha256": engine.file_sha256(paths["oof"]),
        "history_sha256": engine.file_sha256(paths["history"]),
        "summary_sha256": engine.file_sha256(paths["summary"]),
    }
    engine.atomic_write_json(paths["complete"], {**marker_core, "sha256": engine.payload_sha256(marker_core)})
    if paths["resume"].is_file():
        paths["resume"].unlink()
    require(experiment_size() <= MAX_EXPERIMENT_BYTES, "Experiment directory exceeded 1.5 GB")
    print(f"[{context['task_id']} fold {fold}] best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    return load_completed_fold(context, fold, runtime)


def load_completed_fold(
    context: dict[str, Any], fold: int, runtime: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context["task_id"], fold)
    require(all(paths[name].is_file() for name in ("checkpoint", "oof", "history", "summary", "complete")), "Completed fold artifacts missing")
    require(not paths["resume"].exists(), "Completed fold retained resume")
    lock = fold_lock(runtime, context, fold)
    marker = engine.read_json(paths["complete"])
    marker_core = {key: value for key, value in marker.items() if key != "sha256"}
    require(marker["sha256"] == engine.payload_sha256(marker_core) and marker["lock"] == lock, "Complete marker changed")
    for key, name in (("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("history_sha256", "history"), ("summary_sha256", "summary")):
        require(marker[key] == engine.file_sha256(paths[name]), f"Completed {name} changed")
    summary = engine.read_json(paths["summary"])
    require(summary["lock"] == lock, "Completed PS-SPR audit changed")
    require(summary["historical_gate_trainable"] is True and summary["historical_gate_in_optimizer"] is True, "Completed historical gate ownership changed")
    rows = typed_rows(paths["oof"])
    require(len(rows) == summary["oof_count"], "Completed OOF count changed")
    truth, saved_logits, saved_probabilities = engine.rows_arrays(rows)
    require(engine.metrics_match(engine.binary_metrics(truth, saved_probabilities, context["protocol"]), summary["best_metrics"]), "Completed OOF metrics changed")
    history = engine.read_csv_rows(paths["history"])
    require([int(row["epoch"]) for row in history] == list(range(1, FORMAL_EPOCHS + 1)), "Completed history changed")
    selected = max(history, key=lambda row: (float(row["acc"]), float(row["roc_auc"]), float(row["macro_f1"]), -int(row["epoch"])))
    require(int(selected["epoch"]) == summary["best_epoch"], "Completed best epoch changed")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint["lock"] == lock and checkpoint["best_epoch"] == summary["best_epoch"], "Completed checkpoint changed")
    model = build_model(context, "pre_shared")
    model.load_state_dict(checkpoint["best_model"], strict=True)
    _, test_mask, _ = engine.fold_positions(context, fold)
    replay_logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - saved_logits))) <= 1e-6, "Completed logits replay changed")
    require(float(np.max(np.abs(replay_probabilities - saved_probabilities))) <= 1e-6, "Completed probabilities replay changed")
    del model
    torch.cuda.empty_cache()
    return summary, rows


def exact_mcnemar_p(repairs: int, damages: int) -> float:
    discordant = repairs + damages
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(repairs, damages) + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def paired_reference(
    context: dict[str, Any], result: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_rows = typed_rows(reference_paths(context["protocol"])["oof"])
    old = {row["subject_id"]: row for row in reference_rows}
    new = {row["subject_id"]: row for row in rows}
    require(old.keys() == new.keys(), "Paired subject sets changed")
    repairs = damages = changed = 0
    repairs_by_truth = {name: 0 for name in context["protocol"]["class_names"]}
    damages_by_truth = {name: 0 for name in context["protocol"]["class_names"]}
    transitions: dict[str, int] = {}
    for subject_id in old:
        left, right = old[subject_id], new[subject_id]
        require(left["truth"] == right["truth"] and left["feature_sha256"] == right["feature_sha256"] and left["fold"] == right["fold"], "Paired identity changed")
        left_correct = left["prediction"] == left["truth"]
        right_correct = right["prediction"] == right["truth"]
        class_name = context["protocol"]["class_names"][left["truth"]]
        if not left_correct and right_correct:
            repairs += 1
            repairs_by_truth[class_name] += 1
        if left_correct and not right_correct:
            damages += 1
            damages_by_truth[class_name] += 1
        if left["prediction"] != right["prediction"]:
            changed += 1
            key = f"{context['protocol']['class_names'][left['prediction']]}->{context['protocol']['class_names'][right['prediction']]}"
            transitions[key] = transitions.get(key, 0) + 1
    reference = reference_metrics(context["protocol"], reference_rows)
    reference_trial = reference_summary(context["protocol"])
    metric_deltas = {
        metric: float(result["metrics"][metric] - reference[metric])
        for metric in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe")
    }
    return {
        "reference_trial_id": context["protocol"]["reference"]["trial_id"],
        "reference_metrics": reference,
        "repairs": repairs,
        "damages": damages,
        "changed": changed,
        "net_correct": repairs - damages,
        "correct_delta": int(result["metrics"]["correct"] - reference["correct"]),
        "repairs_by_truth": repairs_by_truth,
        "damages_by_truth": damages_by_truth,
        "prediction_transitions": transitions,
        "metric_deltas": metric_deltas,
        "exact_mcnemar_p": exact_mcnemar_p(repairs, damages),
        "parameter_delta": int(result["total_parameter_count"] - int(reference_trial["parameter_count"])),
        "training_time_delta_seconds": float(result["training_time_seconds"] - float(reference_trial["training_time_seconds"])),
        "inference_time_delta_seconds": float(result["inference_time_seconds"] - float(reference_trial["inference_time_seconds"])),
        "reference_training_time_seconds": float(reference_trial["training_time_seconds"]),
        "reference_inference_time_seconds": float(reference_trial["inference_time_seconds"]),
    }


def aggregate_mechanism(context: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = [item["name"] for item in context["protocol"]["modalities"]]
    source_norm = {
        modality: float(statistics.mean(summary["diagnostics"]["source_mean_norm_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    shared_norm = {
        modality: float(statistics.mean(summary["diagnostics"]["shared_mean_norm_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    residual_source_ratio = {
        modality: float(statistics.mean(summary["diagnostics"]["private_source_ratio_mean_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    ratio_mean = {
        modality: float(statistics.mean(summary["diagnostics"]["private_shared_ratio_mean_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    ratio_max = {
        modality: float(max(summary["diagnostics"]["private_shared_ratio_max_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    adapter_gradient_by_modality = {
        modality: float(max(summary["diagnostics"]["adapter_max_gradient_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    max_gradient = max(
        value for summary in summaries for value in summary["diagnostics"]["adapter_cumulative_max_gradient"].values()
    )
    min_delta = min(
        value for summary in summaries for value in summary["diagnostics"]["adapter_parameter_max_delta"].values()
    )
    return {
        "effective_gate_by_modality": dict(zip(modalities, summaries[0]["effective_gate"])),
        "effective_noise_std_by_modality": dict(zip(modalities, summaries[0]["effective_noise_std"])),
        "historical_gate_trainable_all_folds": all(summary["historical_gate_trainable"] for summary in summaries),
        "historical_gate_in_optimizer_all_folds": all(summary["historical_gate_in_optimizer"] for summary in summaries),
        "pre_shared_shape": summaries[0]["diagnostics"]["pre_shared_shape"],
        "post_shared_shape": summaries[0]["diagnostics"]["post_shared_shape"],
        "source_mean_norm_by_modality": source_norm,
        "shared_mean_norm_by_modality": shared_norm,
        "source_shared_norm_ratio_by_modality": {modality: source_norm[modality] / max(shared_norm[modality], 1e-12) for modality in modalities},
        "private_source_ratio_mean_by_modality": residual_source_ratio,
        "private_shared_ratio_mean_by_modality": ratio_mean,
        "private_shared_ratio_max_by_modality": ratio_max,
        "category_global_cosine_mean": float(statistics.mean(summary["diagnostics"]["category_global_cosine_mean"] for summary in summaries)),
        "adapter_max_gradient_by_modality": adapter_gradient_by_modality,
        "private_off_probability_mean_abs_change": float(statistics.mean(summary["diagnostics"]["private_off_probability_mean_abs_change"] for summary in summaries)),
        "private_off_probability_max_abs_change": float(max(summary["diagnostics"]["private_off_probability_max_abs_change"] for summary in summaries)),
        "private_off_argmax_change_count_sum_across_folds": int(sum(summary["diagnostics"]["private_off_argmax_change_count"] for summary in summaries)),
        "adapter_max_gradient": float(max_gradient),
        "adapter_min_parameter_delta": float(min_delta),
        "private_trained_all_folds": all(summary["diagnostics"]["private_trained"] for summary in summaries),
        "private_collapse_any_fold": any(summary["diagnostics"]["private_collapse"] for summary in summaries),
        "total_parameter_count": summaries[0]["object_audit"]["total_parameter_count"],
        "trainable_parameter_count": summaries[0]["object_audit"]["trainable_parameter_count"],
        "frozen_parameter_count": summaries[0]["object_audit"]["frozen_parameter_count"],
        "new_parameter_count": 0,
    }


def aggregate_task(
    context: dict[str, Any], summaries: list[dict[str, Any]], fold_rows: list[list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(summaries) == len(fold_rows) == 10, "Formal fold count changed")
    rows = sorted([row for group in fold_rows for row in group], key=lambda row: row["original_csv_index"])
    require(len(rows) == context["protocol"]["sample_count"] and len({row["subject_id"] for row in rows}) == len(rows), "OOF coverage changed")
    expected = sorted([row for fold in context["fold_manifest"]["folds"] for row in fold["test_rows"]], key=lambda row: row["original_csv_index"])
    for row, anchor in zip(rows, expected):
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"], "OOF stable identity changed")
    truth, _, probabilities = engine.rows_arrays(rows)
    metrics = engine.binary_metrics(truth, probabilities, context["protocol"])
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    fold_metrics = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            "correct": summary["best_metrics"]["correct"],
            "acc": summary["best_metrics"]["acc"],
            "roc_auc": summary["best_metrics"]["roc_auc"],
            "pr_auc": summary["best_metrics"]["pr_auc"],
            "macro_f1": summary["best_metrics"]["macro_f1"],
            "bacc": summary["best_metrics"]["bacc"],
            "sen": summary["best_metrics"]["sen"],
            "spe": summary["best_metrics"]["spe"],
            "loss_first": summary["loss_first"],
            "loss_last": summary["loss_last"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "inference_seconds": summary["inference_seconds_full_graph"],
            "resumed_from_epoch": summary["resumed_from_epoch"],
        }
        for summary in summaries
    ]
    oof_path = RESULT_DIR / f"{context['task_id']}_ps_spr_oof_predictions.csv"
    engine.atomic_write_csv(oof_path, rows)
    mechanism = aggregate_mechanism(context, summaries)
    result = {
        "task_id": context["task_id"],
        "metrics": metrics,
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)),
        "fold_acc_sample_sd": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_roc_auc_mean": float(statistics.mean(row["roc_auc"] for row in fold_metrics)),
        "fold_roc_auc_sample_sd": float(statistics.stdev(row["roc_auc"] for row in fold_metrics)),
        "fold_metrics": fold_metrics,
        "confidence": {
            "mean_max_probability": float(confidence.mean()),
            "sample_sd_max_probability": float(confidence.std(ddof=1)),
            "mean_entropy": float(entropy.mean()),
            "sample_sd_entropy": float(entropy.std(ddof=1)),
        },
        "total_parameter_count": mechanism["total_parameter_count"],
        "trainable_parameter_count": mechanism["trainable_parameter_count"],
        "training_time_seconds": float(sum(row["elapsed_seconds"] for row in fold_metrics)),
        "inference_time_seconds": float(sum(row["inference_seconds"] for row in fold_metrics)),
        "mechanism": mechanism,
        "oof_path": oof_path.relative_to(ROOT).as_posix(),
        "oof_sha256": engine.file_sha256(oof_path),
    }
    if context["task_id"] == "tadpole_smci_pmci":
        result["descriptive_targets"] = {
            "correct_at_least_520": metrics["correct"] >= 520,
            "correct_at_least_523": metrics["correct"] >= 523,
            "fold_acc_mean_above_0_9757": result["fold_acc_mean"] > 0.9757,
            "fold_roc_auc_mean_above_0_9049": result["fold_roc_auc_mean"] > 0.9049,
        }
    else:
        result["descriptive_targets"] = {
            "correct_at_least_770": metrics["correct"] >= 770,
            "correct_at_least_787": metrics["correct"] >= 787,
            "fold_acc_mean_above_0_9105": result["fold_acc_mean"] > 0.9105,
            "fold_roc_auc_mean_above_0_9099": result["fold_roc_auc_mean"] > 0.9099,
        }
    result["paired_reference"] = paired_reference(context, result, rows)
    return result, rows


def task_safety(result: dict[str, Any]) -> dict[str, bool]:
    deltas = result["paired_reference"]["metric_deltas"]
    metrics = result["metrics"]
    n = metrics["n"]
    predicted = metrics["predicted_counts"]
    no_collapse = min(predicted.values()) >= max(2, int(math.ceil(0.01 * n)))
    return {
        "bacc": deltas["bacc"] >= -0.003,
        "roc_auc": deltas["roc_auc"] >= -0.005,
        "pr_auc": deltas["pr_auc"] >= -0.01,
        "sen": deltas["sen"] >= -0.01,
        "spe": deltas["spe"] >= -0.01,
        "no_class_collapse": no_collapse,
        "historical_gate_trainable": result["mechanism"]["historical_gate_trainable_all_folds"],
        "historical_gate_optimized": result["mechanism"]["historical_gate_in_optimizer_all_folds"],
        "private_trained": result["mechanism"]["private_trained_all_folds"],
    }


def final_decision(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    deltas = {task: result["paired_reference"]["correct_delta"] for task, result in results.items()}
    safety = {task: task_safety(result) for task, result in results.items()}
    hard_stop = []
    for task, result in results.items():
        paired = result["paired_reference"]
        require(paired["correct_delta"] == paired["net_correct"], f"{task}: repairs/damages inconsistent with Correct")
        if paired["correct_delta"] <= -2: hard_stop.append(f"{task}:Correct declined >=2")
        if paired["metric_deltas"]["bacc"] < -0.005: hard_stop.append(f"{task}:BACC below -0.005")
        if paired["metric_deltas"]["roc_auc"] < -0.01: hard_stop.append(f"{task}:AUC below -0.01")
        for name in ("no_class_collapse", "private_trained", "historical_gate_trainable", "historical_gate_optimized"):
            if not safety[task][name]: hard_stop.append(f"{task}:{name}")
    if sum(deltas.values()) <= 0:
        hard_stop.append("combined Correct did not increase")
    if hard_stop:
        return "PS_SPR_STOP", hard_stop
    safe_all = all(all(gates.values()) for gates in safety.values())
    if safe_all and all(delta >= 0 for delta in deltas.values()) and any(delta >= 1 for delta in deltas.values()) and sum(deltas.values()) >= 1:
        return "PS_SPR_PASS", ["Both tasks met all preregistered safety floors and the combined Correct gain rule"]
    if safe_all and sum(deltas.values()) > 0 and min(deltas.values()) >= -1 and max(deltas.values()) >= 2:
        return "PS_SPR_NEAR", ["Combined Correct improved with one directional gain, but cross-task stability was not a PASS"]
    return "PS_SPR_STOP", ["Neither preregistered PASS nor NEAR was satisfied"]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PS-SPR v1: Pre-Shared Source-Preserving Residual",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Branch: `{BRANCH}`",
        f"- Source commit: `{summary['source_commit']}`",
        "- Result commit: reported in the final Git handoff (self-reference is not embedded).",
        f"- Device: `{summary['device']}` ({summary['cuda_device_name']})",
        "- Protocol: fixed D5/D3 configuration, one seed (0), ten folds, 400 epochs, full batch, single model; graph/EMA/ensemble disabled.",
        "- Selection: ACC > ROC-AUC > Macro-F1 > earliest epoch.",
        "",
        "## Results",
        "",
        "| Task | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | fold AUC mean+/-SD | params total/trainable | train s | infer s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id in TASK_IDS:
        result = summary["tasks"][task_id]
        metrics = result["metrics"]
        paired = result["paired_reference"]
        source_check = summary["smoke"]["tasks"][task_id]["source_preserving_diagnostic"]
        lines.append(
            f"| {task_id} | {metrics['correct']}/{metrics['n']} | {metrics['acc']:.7f} | {metrics['roc_auc']:.7f} | {metrics['pr_auc']:.7f} | {metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | {metrics['weighted_f1']:.7f} | {metrics['sen']:.7f} | {metrics['spe']:.7f} | `{metrics['confusion_matrix']}` | `{metrics['predicted_counts']}` | {paired['repairs']}/{paired['damages']}/{paired['changed']} | {paired['exact_mcnemar_p']:.7g} | {result['fold_acc_mean']:.7f}+/-{result['fold_acc_sample_sd']:.7f} | {result['fold_roc_auc_mean']:.7f}+/-{result['fold_roc_auc_sample_sd']:.7f} | {result['total_parameter_count']}/{result['trainable_parameter_count']} | {result['training_time_seconds']:.2f} | {result['inference_time_seconds']:.4f} |"
        )
        lines.extend([
            "",
            f"### {task_id} folds",
            "",
            "| Fold | Best epoch | Correct | ACC | AUC | PR-AUC | Macro-F1 | BACC | SEN | SPE | loss first->last |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for fold in result["fold_metrics"]:
            lines.append(
                f"| {fold['fold']} | {fold['best_epoch']} | {fold['correct']} | {fold['acc']:.7f} | {fold['roc_auc']:.7f} | {fold['pr_auc']:.7f} | {fold['macro_f1']:.7f} | {fold['bacc']:.7f} | {fold['sen']:.7f} | {fold['spe']:.7f} | {fold['loss_first']:.5f}->{fold['loss_last']:.5f} |"
            )
        lines.extend([
            "",
            f"- Metric deltas vs {paired['reference_trial_id']}: `{json.dumps(paired['metric_deltas'], sort_keys=True)}`; Correct delta={paired['correct_delta']}.",
            f"- Repair sources: `{json.dumps(paired['repairs_by_truth'], sort_keys=True)}`; damage sources: `{json.dumps(paired['damages_by_truth'], sort_keys=True)}`; transitions: `{json.dumps(paired['prediction_transitions'], sort_keys=True)}`.",
            f"- Historical gate: `{json.dumps(result['mechanism']['effective_gate_by_modality'], sort_keys=True)}`; historical noise std: `{json.dumps(result['mechanism']['effective_noise_std_by_modality'], sort_keys=True)}`.",
            f"- Pre/post shape: `{result['mechanism']['pre_shared_shape']}` / `{result['mechanism']['post_shared_shape']}`; source/shared norm ratio: `{json.dumps(result['mechanism']['source_shared_norm_ratio_by_modality'], sort_keys=True)}`.",
            f"- Private/source ratio: `{json.dumps(result['mechanism']['private_source_ratio_mean_by_modality'], sort_keys=True)}`; Private/shared ratio mean/max: `{json.dumps(result['mechanism']['private_shared_ratio_mean_by_modality'], sort_keys=True)}` / `{json.dumps(result['mechanism']['private_shared_ratio_max_by_modality'], sort_keys=True)}`.",
            f"- Adapter max gradient by modality: `{json.dumps(result['mechanism']['adapter_max_gradient_by_modality'], sort_keys=True)}`; Category-Global cosine mean: {result['mechanism']['category_global_cosine_mean']:.7f}.",
            f"- Source perturbation: max unchanged P_m error={source_check['p_m_max_abs_diff_after_e_n_perturbation']:.3g}, min changed P_n magnitude={source_check['p_n_min_abs_diff_after_e_n_perturbation']:.3g}; private-only Modal Encoder/Shared Transformer gradient max={source_check['modal_encoder_private_only_gradient_max']:.3g}/{source_check['shared_transformer_private_only_gradient_max']:.3g}.",
            f"- Same-checkpoint Private-off probability mean/max change={result['mechanism']['private_off_probability_mean_abs_change']:.7g}/{result['mechanism']['private_off_probability_max_abs_change']:.7g}; argmax changes={result['mechanism']['private_off_argmax_change_count_sum_across_folds']}.",
            f"- Parameter delta={paired['parameter_delta']}; training/inference time delta={paired['training_time_delta_seconds']:.3f}s/{paired['inference_time_delta_seconds']:.6f}s.",
            f"- Descriptive targets (not selection gates): `{json.dumps(result['descriptive_targets'], sort_keys=True)}`.",
            f"- Confidence mean max-probability / mean entropy: {result['confidence']['mean_max_probability']:.7f} / {result['confidence']['mean_entropy']:.7f}.",
        ])
    tad = summary["tasks"]["tadpole_smci_pmci"]
    abide5 = summary["tasks"]["abide5_ads_cn"]
    lines.extend([
        "",
        "## Required answers",
        "",
        "1. Historical post-shared adapters read `H_m`, the corresponding token after Shared Transformer mixing.",
        "2. PS-SPR reads `E_m`, the last gated/noised, modality-encoded token immediately before the first cross-modal Shared Transformer block.",
        "3. At that capture point `E_m` has not read other modality tokens; this is a graph-origin statement, not statistical independence.",
        f"4. Parameter counts are exactly unchanged from post-shared references: **{tad['total_parameter_count'] == EXPECTED_TOTAL_PARAMETERS['tadpole_smci_pmci'] and abide5['total_parameter_count'] == EXPECTED_TOTAL_PARAMETERS['abide5_ads_cn']}**.",
        f"5. Step-0 logits/probabilities are strictly equal within 1e-7 on both tasks: **{all(task['initialization_audit']['step0_logits_max_abs_diff'] < 1e-7 and task['initialization_audit']['step0_probability_max_abs_diff'] < 1e-7 for task in summary['smoke']['tasks'].values())}**.",
        f"6. Source-preserving graph checks passed for every modality in both tasks: **{summary['smoke']['source_preserving_all_passed']}**.",
        f"7. TADPOLE Correct is {tad['metrics']['correct']}/535; exceeds 519: **{tad['metrics']['correct'] > 519}**.",
        f"8. ABIDE-5 Correct is {abide5['metrics']['correct']}/864; exceeds 769: **{abide5['metrics']['correct'] > 769}**.",
        f"9. Combined Correct net change is {tad['paired_reference']['correct_delta'] + abide5['paired_reference']['correct_delta']}.",
        f"10. Repairs exceed damages across both tasks: **{tad['paired_reference']['repairs'] + abide5['paired_reference']['repairs'] > tad['paired_reference']['damages'] + abide5['paired_reference']['damages']}**.",
        f"11. BACC/AUC/PR-AUC/SEN/SPE safety: TAD `{json.dumps(task_safety(tad), sort_keys=True)}`; ABIDE-5 `{json.dumps(task_safety(abide5), sort_keys=True)}`.",
        f"12. Adapters and corresponding Modal Encoders receive effective gradients: **{all(result['mechanism']['private_trained_all_folds'] for result in summary['tasks'].values()) and summary['smoke']['source_preserving_all_passed']}**.",
        f"    Private collapse by task: TAD={tad['mechanism']['private_collapse_any_fold']}, ABIDE-5={abide5['mechanism']['private_collapse_any_fold']}.",
        f"13. The pre-mixing source-evidence hypothesis is supported only if Decision is PASS: **{summary['decision'] == 'PS_SPR_PASS'}**.",
        f"14. Expansion to ABIDE and TADPOLE three-class is allowed only on PASS: **{summary['decision'] == 'PS_SPR_PASS'}**; this run never auto-expands.",
        f"15. Retain D5/D3: **{summary['decision'] != 'PS_SPR_PASS'}**.",
        f"16. Final Decision: **{summary['decision']}**.",
        f"17. Reproducibility: source `{summary['source_commit']}`, result commit is reported in final handoff, branch `{BRANCH}`, device `{summary['device']}`; commands are listed below.",
        "",
        "## Reproduction",
        "",
        f"- The exact source gate must be run from an isolated clean checkout where local branch `{BRANCH}` points to source commit `{summary['source_commit']}`. The later result commit is for reading artifacts and intentionally does not satisfy the source-scope gate.",
        "- The protected sibling reference worktrees `tmp/tad_binary_hparam_search_v1` and `tmp/abide5_hparam_search_v1` must remain at their locked commits and paths.",
        "- `python -u -B scripts/run_pre_shared_source_residual_v1.py inspect`",
        "- `python -u -B scripts/run_pre_shared_source_residual_v1.py smoke --device cuda:0`",
        "- `python -u -B scripts/run_pre_shared_source_residual_v1.py formal --device cuda:0`",
        "",
        "## Decision reasons",
        "",
    ])
    lines.extend(f"- {reason}" for reason in summary["decision_reasons"])
    return "\n".join(lines) + "\n"


def run_formal(device_text: str) -> None:
    source_commit, smoke_report = validate_smoke_and_source()
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_text)
    runtime = runtime_lock(source_commit, device_text)
    formal_config_path = RESULT_DIR / "formal_config.json"
    if formal_config_path.is_file():
        require(engine.read_json(formal_config_path) == runtime, "Existing formal config does not match current runtime lock")
    else:
        engine.atomic_write_json(formal_config_path, runtime)
    task_results: dict[str, dict[str, Any]] = {}
    mechanism_results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    checkpoint_manifests: dict[str, Any] = {}
    config_snapshots: dict[str, Any] = {}
    for task_id, protocol in load_protocols().items():
        context = engine.build_context(protocol, device)
        context["fold_manifest"] = engine.read_json(FOLD_MANIFEST_PATH)["tasks"][task_id]
        summaries, fold_rows = [], []
        for fold in FOLDS:
            fold_summary, rows = train_fold(context, fold, runtime)
            summaries.append(fold_summary)
            fold_rows.append(rows)
        result, _rows = aggregate_task(context, summaries, fold_rows)
        task_results[task_id] = result
        mechanism_results[task_id] = result["mechanism"]
        comparisons[task_id] = result["paired_reference"]
        engine.atomic_write_json(RESULT_DIR / f"{task_id}_fold_results.json", {"task_id": task_id, "folds": result["fold_metrics"]})
        checkpoint_core = {
            "task_id": task_id,
            "source_commit": source_commit,
            "private_source": "pre_shared",
            "folds": [
                {
                    "fold": fold,
                    "path": fold_paths(task_id, fold)["checkpoint"].relative_to(ROOT).as_posix(),
                    "sha256": engine.file_sha256(fold_paths(task_id, fold)["checkpoint"]),
                    "complete_marker_sha256": engine.file_sha256(fold_paths(task_id, fold)["complete"]),
                }
                for fold in FOLDS
            ],
        }
        checkpoint_manifest = {**checkpoint_core, "sha256": engine.payload_sha256(checkpoint_core)}
        checkpoint_path = RESULT_DIR / f"{task_id}_checkpoint_manifest.json"
        engine.atomic_write_json(checkpoint_path, checkpoint_manifest)
        checkpoint_manifests[task_id] = {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "sha256": engine.file_sha256(checkpoint_path),
        }
        snapshot_core = {
            "task_id": task_id,
            "source_commit": source_commit,
            "runtime_sha256": runtime["sha256"],
            "protocol": protocol,
            "model_switch": {"private_source": "pre_shared"},
        }
        snapshot = {**snapshot_core, "sha256": engine.payload_sha256(snapshot_core)}
        snapshot_path = RESULT_DIR / f"{task_id}_config_snapshot.json"
        engine.atomic_write_json(snapshot_path, snapshot)
        config_snapshots[task_id] = {
            "path": snapshot_path.relative_to(ROOT).as_posix(),
            "sha256": engine.file_sha256(snapshot_path),
        }
    decision, reasons = final_decision(task_results)
    summary_core = {
        "experiment": EXPERIMENT_ID,
        "decision": decision,
        "decision_reasons": reasons,
        "tasks": task_results,
        "source_commit": source_commit,
        "result_commit": "reported_in_final_git_handoff",
        "branch": BRANCH,
        "device": device_text,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "smoke": smoke_report,
        "checkpoint_manifests": checkpoint_manifests,
        "config_snapshots": config_snapshots,
        "single_model": True,
        "ensemble": False,
        "ema": False,
        "private_source": "pre_shared",
        "prior_free_modality_prior": False,
        "no_orthogonality": False,
        "experiment_logical_bytes": experiment_size(),
    }
    engine.atomic_write_json(RESULT_DIR / "mechanism_diagnostics.json", mechanism_results)
    engine.atomic_write_json(RESULT_DIR / "comparisons.json", comparisons)
    report = render_report(summary_core)
    engine.atomic_write_text(RESULT_DIR / "REPORT.md", report)
    summary_core["report_sha256"] = engine.file_sha256(RESULT_DIR / "REPORT.md")
    summary_core["mechanism_diagnostics_sha256"] = engine.file_sha256(RESULT_DIR / "mechanism_diagnostics.json")
    summary_core["comparisons_sha256"] = engine.file_sha256(RESULT_DIR / "comparisons.json")
    summary = {**summary_core, "sha256": engine.payload_sha256(summary_core)}
    engine.atomic_write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps({"decision": decision, "correct": {task: result["metrics"]["correct"] for task, result in task_results.items()}, "summary_sha256": summary["sha256"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect", help="Inspect PS-SPR information flow, historical protocols, and reference evidence")
    smoke = subparsers.add_parser("smoke", help="Run both fold-0 PS-SPR CUDA smoke and source-origin checks")
    smoke.add_argument("--device", default="cuda:0")
    formal = subparsers.add_parser("formal", help="Run/strictly resume both PS-SPR ten-fold suites")
    formal.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        run_inspect()
    elif args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode == "formal":
        run_formal(args.device)
    else:
        raise InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
