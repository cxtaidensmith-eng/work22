#!/usr/bin/env python3
"""Prior-Free Modality Prior v1 on locked TADPOLE-binary and ABIDE-5 tasks."""

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


EXPERIMENT_ID = "prior_free_modality_prior_v1"
BRANCH = "experiment/prior-free-modality-prior-v1"
BASE_COMMIT = "e7d18205ddfefaf898a81c999cce15b348331206"
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
    "scripts/run_prior_free_modality_prior_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocols/tadpole_smci_pmci.json",
    f"experiments/{EXPERIMENT_ID}/protocols/abide5_ads_cn.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
)
SMOKE_PATHS = (
    f"experiments/{EXPERIMENT_ID}/smoke/smoke_config.json",
    f"experiments/{EXPERIMENT_ID}/smoke/smoke_report.json",
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
        require(protocol["prior_free_modality_prior"] is True, f"Prior-free switch missing: {task}")
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
    for metric in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "sen", "spe"):
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
    context: dict[str, Any], prior_free: bool, seed: int = SEED
) -> CMEDualBranchModel:
    SET_Random(seed)
    rank = int(context["protocol"]["reference"]["rank"])
    model = CMEDualBranchModel(
        **engine.model_kwargs(context),
        cme_arm="c1",
        adapter_rank=rank,
        router_hidden=16,
        modality_embedding_dim=8,
        prior_free_modality_prior=prior_free,
    ).to(context["device"])
    require(len(model.private_adapters) == len(context["protocol"]["modalities"]), "Adapter count changed")
    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary query/OVR schema changed")
    require(total_parameter_count(model) == EXPECTED_TOTAL_PARAMETERS[context["task_id"]], "Total parameter count changed")
    if prior_free:
        require(not model.modal_gate_logit.requires_grad, "Legacy gate remains trainable")
        require(bool(torch.equal(model.effective_modal_gate(), torch.ones_like(model.modal_gate_logit))), "Effective gate is not one")
        expected_noise = torch.full_like(model._modal_noise_std, float(context["protocol"]["training"]["input_noise_std"]))
        require(bool(torch.equal(model.effective_modal_noise_std(), expected_noise)), "Effective noise is not uniform")
    return model


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def make_training_objects(
    context: dict[str, Any], seed: int = SEED
) -> tuple[CMEDualBranchModel, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    model = build_model(context, True, seed)
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
    require(id(model.modal_gate_logit) not in optimizer_parameter_ids(optimizer), "Frozen gate entered optimizer")
    audit = {
        "total_parameter_count": total_parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "frozen_parameter_count": total_parameter_count(model) - trainable_parameter_count(model),
        "private_parameter_count": private_parameter_count(model),
        "common_parameter_tensors": len(common),
        "adapter_parameter_tensors": len(adapters),
        "legacy_gate_parameter_count": int(model.modal_gate_logit.numel()),
        "legacy_gate_trainable": bool(model.modal_gate_logit.requires_grad),
        "legacy_gate_in_optimizer": id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer),
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
    historical = build_model(context, False)
    prior_free = build_model(context, True)
    full_diff = state_max_diff(historical, prior_free)
    require(full_diff == 0.0, "Seeded initial model state changed")
    historical.eval()
    prior_free.eval()
    with torch.no_grad():
        old_logits = historical(context["dataset_data"]["Feature"])[0]
        new_logits = prior_free(context["dataset_data"]["Feature"])[0]
    logits_diff = float((old_logits - new_logits).abs().max().cpu())
    audit = {
        "all_state_initialization_max_abs_diff": full_diff,
        "common_trainable_initialization_max_abs_diff": full_diff,
        "step0_logits_max_abs_diff_expected_due_to_prior_removal": logits_diff,
        "comparison_scope": "all state tensors; modal_gate_logit retained with identical seeded value but frozen and bypassed",
    }
    del historical, prior_free
    return audit


def renamed_modality_audit(context: dict[str, Any]) -> dict[str, Any]:
    renamed_context = dict(context)
    renamed_dataset = dict(context["dataset_dict"])
    renamed_dataset["Modal_Name"] = [f"ANON_{index}" for index in range(len(context["protocol"]["modalities"]))]
    renamed_context["dataset_dict"] = renamed_dataset
    original = build_model(context, True)
    renamed = build_model(renamed_context, True)
    trainable_left = {name: value for name, value in original.named_parameters() if value.requires_grad}
    trainable_right = {name: value for name, value in renamed.named_parameters() if value.requires_grad}
    require(trainable_left.keys() == trainable_right.keys(), "Renaming changed trainable schema")
    trainable_diff = max(
        (float((trainable_left[name].detach().cpu() - trainable_right[name].detach().cpu()).abs().max()) for name in trainable_left),
        default=0.0,
    )
    require(trainable_diff == 0.0, "Modality names changed trainable initialization")
    original.eval()
    renamed.eval()
    with torch.no_grad():
        original_logits = original(context["dataset_data"]["Feature"])[0]
        renamed_logits = renamed(context["dataset_data"]["Feature"])[0]
    logits_diff = float((original_logits - renamed_logits).abs().max().cpu())
    require(logits_diff == 0.0, "Modality names changed prior-free logits")
    result = {
        "renamed_modalities": renamed_dataset["Modal_Name"],
        "trainable_state_max_abs_diff": trainable_diff,
        "eval_logits_max_abs_diff": logits_diff,
        "effective_gate_original": original.effective_modal_gate().detach().cpu().tolist(),
        "effective_gate_renamed": renamed.effective_modal_gate().detach().cpu().tolist(),
        "effective_noise_original": original.effective_modal_noise_std().detach().cpu().tolist(),
        "effective_noise_renamed": renamed.effective_modal_noise_std().detach().cpu().tolist(),
    }
    del original, renamed
    return result


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("best_model", "model_state", "model"):
        value = payload.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(item) for item in value.values()):
            return value
    raise InvariantError("Checkpoint model state missing")


def validate_reference_checkpoint(context: dict[str, Any], path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Reference checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, False)
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
        "model_switch": {"prior_free_modality_prior": True},
        "effective_gate_formula": "ones(modality_count), direct identity multiplication",
        "effective_noise_formula": "input_noise_std * ones(modality_count)",
        "legacy_gate_compatibility": "modal_gate_logit retained, requires_grad=False, bypassed, excluded from optimizer",
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "single_model": True,
        "ema": False,
        "ensemble": False,
        "search": False,
        "decision_tokens": ["PRIOR_FREE_PASS", "PRIOR_FREE_NEUTRAL", "PRIOR_FREE_STOP"],
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
        require(paths["oof"].is_file() and paths["fold_manifest"].is_file(), f"Reference evidence missing: {task_id}")
        rows = typed_rows(paths["oof"])
        require(len(rows) == protocol["sample_count"] and len({row["subject_id"] for row in rows}) == len(rows), "Reference OOF coverage changed")
        metrics = reference_metrics(protocol, rows)
        reference_fold = engine.read_json(paths["fold_manifest"])["fold_manifest"]
        require(reference_fold == context["fold_manifest"], f"Reference fold manifest changed: {task_id}")
        checkpoints = [validate_reference_checkpoint(context, path) for path in paths["checkpoints"]]
        historical = build_model(context, False)
        prior_free = build_model(context, True)
        modalities = [item["name"] for item in protocol["modalities"]]
        old_gate_logits = historical.modal_gate_logit.detach().cpu()
        old_gate = historical.effective_modal_gate().detach().cpu()
        old_noise_factors = historical._modal_noise_std.detach().cpu()
        old_effective_noise = historical.effective_modal_noise_std().detach().cpu()
        new_gate = prior_free.effective_modal_gate().detach().cpu()
        new_noise = prior_free.effective_modal_noise_std().detach().cpu()
        require(bool(torch.equal(new_gate, torch.ones_like(new_gate))), "Inspect effective gate is not one")
        require(bool(torch.all(new_noise == float(protocol["training"]["input_noise_std"]))), "Inspect effective noise is not uniform")
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
            "reference_oof_sha256": engine.file_sha256(paths["oof"]),
            "reference_metrics_recomputed": metrics,
            "reference_checkpoint_audit": checkpoints,
            "reference_fold_manifest_sha256": context["fold_manifest"]["sha256"],
            "old_prior": {
                "gate_formula": "sigmoid(modal_gate_logit), applied to category tokens and Global/Adj features",
                "gate_trainable": True,
                "gate_logit_by_modality": dict(zip(modalities, old_gate_logits.tolist())),
                "effective_gate_by_modality": dict(zip(modalities, old_gate.tolist())),
                "noise_factor_by_modality": dict(zip(modalities, old_noise_factors.tolist())),
                "effective_noise_std_by_modality": dict(zip(modalities, old_effective_noise.tolist())),
            },
            "new_prior": {
                "gate_formula": "identity: one for every real modality; no sigmoid or softmax",
                "effective_gate_by_modality": dict(zip(modalities, new_gate.tolist())),
                "noise_formula": "task input_noise_std multiplied by one for every real modality",
                "effective_noise_std_by_modality": dict(zip(modalities, new_noise.tolist())),
                "legacy_gate_retained": True,
                "legacy_gate_trainable": False,
                "legacy_gate_bypassed": True,
                "new_parameter_count": 0,
            },
            "parameter_audit": {
                "old_total": total_parameter_count(historical),
                "old_trainable": trainable_parameter_count(historical),
                "new_total": total_parameter_count(prior_free),
                "new_trainable": trainable_parameter_count(prior_free),
                "trainable_delta": trainable_parameter_count(prior_free) - trainable_parameter_count(historical),
            },
        }
        fold_tasks[task_id] = context["fold_manifest"]
        del historical, prior_free
    fold_core = {"tasks": fold_tasks}
    fold_payload = {**fold_core, "sha256": engine.payload_sha256(fold_core)}
    core = {
        "experiment": EXPERIMENT_ID,
        "tasks": task_payload,
        "disk_free_bytes": int(free),
        "disk_minimum_bytes": 4_000_000_000,
        "hard_coded_modality_priors_found": True,
        "old_gate_source": "Model/network.py modal-name conditional -> modal_gate_logit; sigmoid in CMEDualBranchModel.forward",
        "old_noise_source": "Model/network.py modal-name conditional -> _modal_noise_std; scaled in CMEDualBranchModel.forward",
        "only_allowed_change": "prior_free_modality_prior=True",
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
    require(model.modal_gate_logit.grad is None, "Frozen legacy gate received gradient")
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Model gradient is non-finite")
    target_gradients = capture_target_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
    optimizer.step()
    return float(loss.detach().cpu()), adapter_gradients, target_gradients


def reference_fold_replay(context: dict[str, Any], fold: int = 0) -> dict[str, Any]:
    path = reference_paths(context["protocol"])["checkpoints"][fold]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, False)
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


def run_smoke(device_text: str) -> None:
    source_commit = source_gate(include_smoke=False)
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_text)
    reports: dict[str, Any] = {}
    for task_id, protocol in load_protocols().items():
        context = engine.build_context(protocol, device)
        context["fold_manifest"] = engine.read_json(FOLD_MANIFEST_PATH)["tasks"][task_id]
        replay = reference_fold_replay(context, 0)
        init_audit = initialization_audit(context) if task_id == TASK_IDS[0] else {"scope": "full comparison performed once on TADPOLE fold0"}
        name_audit = renamed_modality_audit(context)
        model, criterion, optimizer, scheduler, object_audit = make_training_objects(context)
        train_mask, test_mask, _ = engine.fold_positions(context, 0)
        initial_private = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
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
        reports[task_id] = {
            "reference_checkpoint_replay": replay,
            "initialization_audit": init_audit,
            "renamed_modality_audit": name_audit,
            "object_audit": object_audit,
            "losses": losses,
            "adapter_cumulative_max_gradient": cumulative_adapter,
            "target_group_cumulative_max_gradient": cumulative_targets,
            "adapter_parameter_max_delta": private_delta,
            "effective_gate": model.effective_modal_gate().detach().cpu().tolist(),
            "effective_noise_std": model.effective_modal_noise_std().detach().cpu().tolist(),
            "legacy_gate_gradient_is_none": model.modal_gate_logit.grad is None,
            "optimizer_steps": SMOKE_EPOCHS,
            "scheduler_steps": SMOKE_EPOCHS,
            "probability_simplex_max_abs_error": simplex,
            "checkpoint_roundtrip_logits_max_abs_diff": reload_diff,
            "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        }
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
    report_core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "smoke_config_sha256": smoke_config["sha256"],
        "tasks": reports,
        "all_gates_one": all(all(value == 1.0 for value in report["effective_gate"]) for report in reports.values()),
        "all_noise_uniform": all(len(set(report["effective_noise_std"])) == 1 for report in reports.values()),
        "name_independent": all(report["renamed_modality_audit"]["eval_logits_max_abs_diff"] == 0.0 for report in reports.values()),
        "formal_go": True,
    }
    smoke_report = {**report_core, "sha256": engine.payload_sha256(report_core)}
    engine.atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    engine.atomic_write_json(SMOKE_DIR / "smoke_report.json", smoke_report)
    print(json.dumps({"formal_go": True, "source_commit": source_commit, "report_sha256": smoke_report["sha256"]}, indent=2))


def validate_smoke_and_source() -> tuple[str, dict[str, Any]]:
    source_commit = source_gate(include_smoke=True)
    config = engine.read_json(SMOKE_DIR / "smoke_config.json")
    report = engine.read_json(SMOKE_DIR / "smoke_report.json")
    require(config["sha256"] == engine.payload_sha256({key: value for key, value in config.items() if key != "sha256"}), "Smoke config digest changed")
    require(report["sha256"] == engine.payload_sha256({key: value for key, value in report.items() if key != "sha256"}), "Smoke report digest changed")
    implementation_commit = config["source_commit"]
    require(git("merge-base", "--is-ancestor", implementation_commit, source_commit, check=False).returncode == 0, "Smoke implementation is not an ancestor")
    require(report["source_commit"] == implementation_commit, "Smoke source changed")
    require(report["formal_go"] is True and report["all_gates_one"] and report["all_noise_uniform"] and report["name_independent"], "Smoke formal gate failed")
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
        "model_switch": {"prior_free_modality_prior": True},
        "training": {task: protocols[task]["training"] for task in TASK_IDS},
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {
        "runtime_sha256": runtime["sha256"],
        "task_id": context["task_id"],
        "fold": int(fold),
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "prior_free_modality_prior": True,
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
        row["arm"] = "PRIOR_FREE"
    diagnostics = engine.private_diagnostics(context, model, test_mask, cumulative_gradient, initial_private)
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
        "prior_free_modality_prior": True,
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
        "effective_gate": model.effective_modal_gate().detach().cpu().tolist(),
        "effective_noise_std": model.effective_modal_noise_std().detach().cpu().tolist(),
        "legacy_gate_trainable": model.modal_gate_logit.requires_grad,
        "legacy_gate_in_optimizer": id(model.modal_gate_logit) in optimizer_parameter_ids(optimizer),
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
    require(summary["lock"] == lock and summary["effective_gate"] == [1.0] * len(context["protocol"]["modalities"]), "Completed prior-free audit changed")
    require(summary["legacy_gate_trainable"] is False and summary["legacy_gate_in_optimizer"] is False, "Completed gate ownership changed")
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
    model = build_model(context, True)
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
    }


def aggregate_mechanism(context: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = [item["name"] for item in context["protocol"]["modalities"]]
    ratio_mean = {
        modality: float(statistics.mean(summary["diagnostics"]["private_shared_ratio_mean_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    ratio_max = {
        modality: float(max(summary["diagnostics"]["private_shared_ratio_max_by_modality"][modality] for summary in summaries))
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
        "legacy_gate_trainable_any_fold": any(summary["legacy_gate_trainable"] for summary in summaries),
        "legacy_gate_in_optimizer_any_fold": any(summary["legacy_gate_in_optimizer"] for summary in summaries),
        "private_shared_ratio_mean_by_modality": ratio_mean,
        "private_shared_ratio_max_by_modality": ratio_max,
        "category_global_cosine_mean": float(statistics.mean(summary["diagnostics"]["category_global_cosine_mean"] for summary in summaries)),
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
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"] and row["fold"] == anchor["fold"], "OOF stable identity changed")
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
    oof_path = RESULT_DIR / f"{context['task_id']}_prior_free_oof_predictions.csv"
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
    result["paired_reference"] = paired_reference(context, result, rows)
    return result, rows


def task_safety(result: dict[str, Any]) -> dict[str, bool]:
    deltas = result["paired_reference"]["metric_deltas"]
    metrics = result["metrics"]
    n = metrics["n"]
    predicted = metrics["predicted_counts"]
    no_collapse = min(predicted.values()) >= max(2, int(math.ceil(0.01 * n)))
    return {
        "correct": result["paired_reference"]["correct_delta"] >= 0,
        "bacc": deltas["bacc"] >= -0.003,
        "roc_auc": deltas["roc_auc"] >= -0.005,
        "sen": deltas["sen"] >= -0.01,
        "spe": deltas["spe"] >= -0.01,
        "no_class_collapse": no_collapse,
        "gate_identity": all(value == 1.0 for value in result["mechanism"]["effective_gate_by_modality"].values()),
        "noise_uniform": len(set(result["mechanism"]["effective_noise_std_by_modality"].values())) == 1,
        "gate_not_trainable": not result["mechanism"]["legacy_gate_trainable_any_fold"],
        "gate_not_optimized": not result["mechanism"]["legacy_gate_in_optimizer_any_fold"],
        "private_trained": result["mechanism"]["private_trained_all_folds"],
    }


def final_decision(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    safety = {task: task_safety(result) for task, result in results.items()}
    failed = [f"{task}:{name}" for task, gates in safety.items() for name, passed in gates.items() if not passed]
    if failed:
        return "PRIOR_FREE_STOP", failed
    deltas = {task: result["paired_reference"]["correct_delta"] for task, result in results.items()}
    if all(delta >= 0 for delta in deltas.values()) and any(delta >= 1 for delta in deltas.values()):
        return "PRIOR_FREE_PASS", ["Both tasks met reference Correct and safety floors; at least one improved"]
    if all(delta == 0 for delta in deltas.values()):
        return "PRIOR_FREE_NEUTRAL", ["Both tasks tied reference Correct and all safety floors passed"]
    return "PRIOR_FREE_STOP", ["Neither the pre-registered PASS nor NEUTRAL rule was satisfied"]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Prior-Free Modality Prior v1",
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
        "| Task | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | params total/trainable | train s | infer s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for task_id in TASK_IDS:
        result = summary["tasks"][task_id]
        metrics = result["metrics"]
        paired = result["paired_reference"]
        lines.append(
            f"| {task_id} | {metrics['correct']}/{metrics['n']} | {metrics['acc']:.7f} | {metrics['roc_auc']:.7f} | {metrics['pr_auc']:.7f} | {metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | {metrics['weighted_f1']:.7f} | {metrics['sen']:.7f} | {metrics['spe']:.7f} | `{metrics['confusion_matrix']}` | `{metrics['predicted_counts']}` | {paired['repairs']}/{paired['damages']}/{paired['changed']} | {paired['exact_mcnemar_p']:.7g} | {result['fold_acc_mean']:.7f}+/-{result['fold_acc_sample_sd']:.7f} | {result['total_parameter_count']}/{result['trainable_parameter_count']} | {result['training_time_seconds']:.2f} | {result['inference_time_seconds']:.4f} |"
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
            f"- Effective gate: `{json.dumps(result['mechanism']['effective_gate_by_modality'], sort_keys=True)}`; effective noise std: `{json.dumps(result['mechanism']['effective_noise_std_by_modality'], sort_keys=True)}`.",
            f"- Private/shared ratio mean/max: `{json.dumps(result['mechanism']['private_shared_ratio_mean_by_modality'], sort_keys=True)}` / `{json.dumps(result['mechanism']['private_shared_ratio_max_by_modality'], sort_keys=True)}`.",
            f"- Confidence mean max-probability / mean entropy: {result['confidence']['mean_max_probability']:.7f} / {result['confidence']['mean_entropy']:.7f}.",
        ])
    tad = summary["tasks"]["tadpole_smci_pmci"]
    abide5 = summary["tasks"]["abide5_ads_cn"]
    lines.extend([
        "",
        "## Required answers",
        "",
        "1. Old gate formula/values: inspect records the modal-name conditional logits and `sigmoid(modal_gate_logit)` values for every modality.",
        "2. Old noise factors: inspect records `_modal_noise_std` and its effective scaled standard deviation for every modality.",
        "3. New gate formula: exact identity, one for every real modality, with no sigmoid/softmax and no learned gate path.",
        "4. New noise formula: the locked task `input_noise_std` is used uniformly for every modality; zero remains exactly zero.",
        f"5. Legacy gate trainability/optimizer: frozen and absent from optimizer on every fold: **{all(not result['mechanism']['legacy_gate_trainable_any_fold'] and not result['mechanism']['legacy_gate_in_optimizer_any_fold'] for result in summary['tasks'].values())}**.",
        f"6. TADPOLE vs D5: {tad['metrics']['correct']}/535 vs 519; delta={tad['paired_reference']['correct_delta']}, repairs/damages={tad['paired_reference']['repairs']}/{tad['paired_reference']['damages']}.",
        f"7. ABIDE-5 vs D3: {abide5['metrics']['correct']}/864 vs 769; delta={abide5['paired_reference']['correct_delta']}, repairs/damages={abide5['paired_reference']['repairs']}/{abide5['paired_reference']['damages']}.",
        f"8. BACC safety: TAD delta {tad['paired_reference']['metric_deltas']['bacc']:.7f}; ABIDE-5 {abide5['paired_reference']['metric_deltas']['bacc']:.7f} (floor -0.003).",
        f"9. AUC safety: TAD delta {tad['paired_reference']['metric_deltas']['roc_auc']:.7f}; ABIDE-5 {abide5['paired_reference']['metric_deltas']['roc_auc']:.7f} (floor -0.005).",
        f"10. SEN/SPE safety: TAD {tad['paired_reference']['metric_deltas']['sen']:.7f}/{tad['paired_reference']['metric_deltas']['spe']:.7f}; ABIDE-5 {abide5['paired_reference']['metric_deltas']['sen']:.7f}/{abide5['paired_reference']['metric_deltas']['spe']:.7f} (each floor -0.01).",
        f"11. Class collapse/bias: TAD predicted `{tad['metrics']['predicted_counts']}`; ABIDE-5 `{abide5['metrics']['predicted_counts']}`; both safety gates `{task_safety(tad)['no_class_collapse']}/{task_safety(abide5)['no_class_collapse']}`.",
        f"12. Optimization/convergence: fold best epochs and first/last losses are reported above; private adapters trained on every fold: **{all(result['mechanism']['private_trained_all_folds'] for result in summary['tasks'].values())}**.",
        f"13. Parameter change: total counts stay {tad['total_parameter_count']} and {abide5['total_parameter_count']}; trainable counts are {tad['trainable_parameter_count']} and {abide5['trainable_parameter_count']}; new parameters=0.",
        f"14. Reproducibility: source `{summary['source_commit']}`, branch `{BRANCH}`, device `{summary['device']}`, runner `scripts/run_prior_free_modality_prior_v1.py`; result commit is reported in the final Git handoff.",
        "",
        "## Decision reasons",
        "",
    ])
    lines.extend(f"- {reason}" for reason in summary["decision_reasons"])
    return "\n".join(lines) + "\n"


def run_formal(device_text: str) -> None:
    source_commit, _smoke = validate_smoke_and_source()
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_text)
    runtime = runtime_lock(source_commit, device_text)
    engine.atomic_write_json(RESULT_DIR / "formal_config.json", runtime)
    task_results: dict[str, dict[str, Any]] = {}
    mechanism_results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
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
        "single_model": True,
        "ensemble": False,
        "ema": False,
        "prior_free_modality_prior": True,
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
    subparsers.add_parser("inspect", help="Inspect priors, data, reference OOF/checkpoints, and fold manifests")
    smoke = subparsers.add_parser("smoke", help="Run both fold-0 prior-free CUDA smoke checks")
    smoke.add_argument("--device", default="cuda:0")
    formal = subparsers.add_parser("formal", help="Run/strictly resume both prior-free ten-fold suites")
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
