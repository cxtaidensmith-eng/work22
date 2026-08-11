#!/usr/bin/env python3
"""MA-ARA pressure validation on locked TADPOLE-binary and ABIDE-5 tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
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


EXPERIMENT_ID = "ma_ara_pressure_v1"
BRANCH = "experiment/ma-ara-pressure-v1"
BASE_COMMIT = "e7d18205ddfefaf898a81c999cce15b348331206"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_DIR = RESULT_DIR / "protocols"
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
WORK_DIR = RESULT_DIR / "work"
TASK_IDS = ("tadpole_smci_pmci", "abide5_ads_cn")
FOLDS = tuple(range(10))
SEED = 0
SCOUT_EPOCHS = 60
FORMAL_EPOCHS = 400
SMOKE_EPOCHS = 3
EMA_BETA = 0.9
MAX_EXPERIMENT_BYTES = 2_000_000_000

IMPLEMENTATION_PATHS = (
    ".gitignore",
    "Model/cme_dual_branch.py",
    "scripts/run_ma_ara_pressure_v1.py",
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


def source_hashes() -> dict[str, str]:
    return {
        path: engine.file_sha256(ROOT / path)
        for path in (*IMPLEMENTATION_PATHS, *LOCKED_DEPENDENCIES)
    }


def protocol_paths() -> dict[str, Path]:
    return {task: PROTOCOL_DIR / f"{task}.json" for task in TASK_IDS}


def load_protocols() -> dict[str, dict[str, Any]]:
    protocols = {task: engine.read_json(path) for task, path in protocol_paths().items()}
    for task, protocol in protocols.items():
        require(protocol["task_id"] == task, f"Protocol ownership changed: {task}")
        require(protocol["training"]["epochs"] == FORMAL_EPOCHS, f"Epoch lock changed: {task}")
        require(protocol["training"]["graph_use_graph"] is False, f"Graph enabled: {task}")
        require(protocol["training"]["ema"] is False, f"EMA enabled: {task}")
        require(len(protocol["modalities"]) == 5, f"Expected five modalities: {task}")
    return protocols


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
        "summaries": [trial / f"fold_{fold:02d}" / "summary.json" for fold in FOLDS],
    }


def typed_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    for metric in ("roc_auc", "pr_auc", "macro_f1", "bacc"):
        require(math.isclose(metrics[metric], reference[metric], abs_tol=5e-7), f"Reference {metric} changed")
    return metrics


def private_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    )


def build_model(context: dict[str, Any], ranks: list[int], seed: int = SEED) -> CMEDualBranchModel:
    SET_Random(seed)
    model = CMEDualBranchModel(
        **engine.model_kwargs(context),
        cme_arm="c1",
        adapter_rank=int(ranks[0]),
        adapter_ranks=ranks,
        router_hidden=16,
        modality_embedding_dim=8,
    ).to(context["device"])
    require(list(model.adapter_ranks) == list(ranks), "Adapter rank allocation changed")
    require(len(model.private_adapters) == len(ranks) == 5, "Adapter count changed")
    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary query/OVR schema changed")
    return model


def make_training_objects(
    context: dict[str, Any], ranks: list[int], seed: int = SEED
) -> tuple[CMEDualBranchModel, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    model = build_model(context, ranks, seed)
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
    require(len(adapters) == 20, "Every five-modality adapter must contribute four tensors")
    all_ids = {id(parameter) for _, parameter in named}
    common_ids = {id(parameter) for _, parameter in common}
    adapter_ids = {id(parameter) for _, parameter in adapters}
    require(not common_ids & adapter_ids and common_ids | adapter_ids == all_ids, "Optimizer groups are not exhaustive/disjoint")
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
    audit = {
        "parameter_count": engine.parameter_count(model),
        "private_parameter_count": private_parameter_count(model),
        "common_parameter_count": engine.parameter_count(model) - private_parameter_count(model),
        "common_parameter_tensors": len(common),
        "adapter_parameter_tensors": len(adapters),
        "base_lr": base_lr,
        "adapter_lr": base_lr * multiplier,
        "adapter_lr_multiplier": multiplier,
    }
    return model, criterion, optimizer, scheduler, audit


def common_state_diff(left: torch.nn.Module, right: torch.nn.Module) -> float:
    left_state = {name: value for name, value in left.state_dict().items() if not name.startswith("private_adapters.")}
    right_state = {name: value for name, value in right.state_dict().items() if not name.startswith("private_adapters.")}
    require(left_state.keys() == right_state.keys(), "Common state schema changed")
    return max(
        (float((left_state[name].detach().cpu() - right_state[name].detach().cpu()).abs().max()) for name in left_state),
        default=0.0,
    )


def private_disabled_logits(model: CMEDualBranchModel, features: torch.Tensor) -> torch.Tensor:
    handles = [adapter.register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output)) for adapter in model.private_adapters]
    try:
        return engine.infer(model, features)[0].detach().cpu()
    finally:
        for handle in handles:
            handle.remove()


def fairness_audit(
    context: dict[str, Any], reference: CMEDualBranchModel, formal: CMEDualBranchModel
) -> dict[str, Any]:
    common_diff = common_state_diff(reference, formal)
    off_diff = float((private_disabled_logits(reference, context["dataset_data"]["Feature"]) - private_disabled_logits(formal, context["dataset_data"]["Feature"])).abs().max())
    reference_on = engine.infer(reference, context["dataset_data"]["Feature"])[0].detach().cpu()
    formal_on = engine.infer(formal, context["dataset_data"]["Feature"])[0].detach().cpu()
    on_diff = float((reference_on - formal_on).abs().max())
    require(common_diff == 0.0 and off_diff == 0.0, "Common/private-off initialization fairness failed")
    require(on_diff <= 1e-7, "Zero-output private-on initialization fairness failed")
    return {
        "common_state_max_abs_diff": common_diff,
        "private_disabled_logits_max_abs_diff": off_diff,
        "private_enabled_zero_output_logits_max_abs_diff": on_diff,
        "trainable_alpha_used": False,
    }


def validate_reference_checkpoint(context: dict[str, Any], path: Path, ranks: list[int]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, ranks)
    model.load_state_dict(payload["best_model"], strict=True)
    del model
    return {
        "path": path.as_posix(),
        "sha256": engine.file_sha256(path),
        "best_epoch": int(payload["best_epoch"]),
        "state_tensor_count": len(payload["best_model"]),
    }


def make_inspect_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    free = shutil.disk_usage(ROOT).free
    require(free >= 6_000_000_000, f"D drive free space below 6 GB: {free}")
    protocols = load_protocols()
    task_payload: dict[str, Any] = {}
    fold_tasks: dict[str, Any] = {}
    for task_id, protocol in protocols.items():
        context = engine.build_context(protocol, torch.device("cpu"))
        paths = reference_paths(protocol)
        require(paths["oof"].is_file() and paths["fold_manifest"].is_file(), f"Reference artifacts missing: {task_id}")
        rows = typed_rows(paths["oof"])
        metrics = reference_metrics(protocol, rows)
        reference_fold = engine.read_json(paths["fold_manifest"])["fold_manifest"]
        require(reference_fold == context["fold_manifest"], f"Reference fold manifest changed: {task_id}")
        reference_rank = int(protocol["reference"]["rank"])
        uniform = [reference_rank] * 5
        checkpoint_audit = [validate_reference_checkpoint(context, path, uniform) for path in paths["checkpoints"]]
        reference_model = build_model(context, uniform)
        per_rank_cost = 2 * int(protocol["training"]["hidden_size"]) + 1
        expected_private = 5 * (reference_rank * per_rank_cost + int(protocol["training"]["hidden_size"]))
        require(private_parameter_count(reference_model) == expected_private, "Private parameter formula changed")
        task_payload[task_id] = {
            "dataset": protocol["dataset"],
            "task": protocol["task"],
            "sample_count": protocol["sample_count"],
            "class_names": protocol["class_names"],
            "class_counts": protocol["class_counts"],
            "positive_class": protocol["positive_class"],
            "positive_index": protocol["positive_index"],
            "modalities": protocol["modalities"],
            "fold_manifest_sha256": context["fold_manifest"]["sha256"],
            "reference": protocol["reference"],
            "reference_oof_sha256": engine.file_sha256(paths["oof"]),
            "reference_metrics_recomputed": metrics,
            "reference_checkpoint_audit": checkpoint_audit,
            "reference_parameter_count": engine.parameter_count(reference_model),
            "reference_private_parameter_count": expected_private,
            "per_rank_parameter_cost": per_rank_cost,
            "equal_rank_cost_across_modalities": True,
            "rank_budget": 5 * reference_rank,
            "scout_rank_per_modality": 2 * reference_rank,
        }
        fold_tasks[task_id] = context["fold_manifest"]
        del reference_model
    fold_core = {"tasks": fold_tasks}
    fold_payload = {**fold_core, "sha256": engine.payload_sha256(fold_core)}
    core = {
        "experiment": EXPERIMENT_ID,
        "tasks": task_payload,
        "disk_free_bytes": int(free),
        "disk_minimum_bytes": 6_000_000_000,
        "data_protocol_compatible": True,
        "checkpoint_compatible": True,
        "scout_uses_train_mask_only": True,
        "test_labels_used_for_allocation": False,
        "fold_manifest_sha256": fold_payload["sha256"],
    }
    return {**core, "sha256": engine.payload_sha256(core)}, fold_payload


def run_inspect() -> None:
    inspect, folds = make_inspect_payload()
    engine.atomic_write_json(INSPECT_PATH, inspect)
    engine.atomic_write_json(FOLD_MANIFEST_PATH, folds)
    print(json.dumps(inspect, indent=2, ensure_ascii=False))


def source_gate(include_smoke: bool) -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked source is dirty")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Index is staged")
    head = current_head()
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Base is not ancestor")
    expected = set(IMPLEMENTATION_PATHS) | (set(SMOKE_PATHS) if include_smoke else set())
    changed = {line.replace("\\", "/") for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines() if line}
    require(changed == expected, f"Committed source scope changed: {sorted(changed)}")
    for path in expected:
        require(git("ls-files", "--error-unmatch", "--", path, check=False).returncode == 0, f"Source file untracked: {path}")
    for path in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", path, check=False).returncode == 0, f"Locked dependency changed: {path}")
    expected_inspect, expected_folds = make_inspect_payload()
    saved_inspect = engine.read_json(INSPECT_PATH)
    require(saved_inspect["sha256"] == engine.payload_sha256({key: value for key, value in saved_inspect.items() if key != "sha256"}), "Inspect manifest digest changed")
    # Free space is intentionally a point-in-time inspect observation.  Recheck
    # the >=6 GB guard above, but do not require the byte count to stay frozen.
    saved_static = {key: value for key, value in saved_inspect.items() if key not in {"sha256", "disk_free_bytes"}}
    expected_static = {key: value for key, value in expected_inspect.items() if key not in {"sha256", "disk_free_bytes"}}
    require(saved_static == expected_static, "Inspect manifest drifted")
    require(engine.read_json(FOLD_MANIFEST_PATH) == expected_folds, "Fold manifest drifted")
    return head


class ImportanceCapture:
    def __init__(self, model: CMEDualBranchModel):
        self.outputs: list[torch.Tensor | None] = [None] * len(model.private_adapters)
        self.handles = []
        for index, adapter in enumerate(model.private_adapters):
            def hook(_module, _inputs, output, slot=index):
                output.retain_grad()
                self.outputs[slot] = output
            self.handles.append(adapter.activation.register_forward_hook(hook))

    def clear(self) -> None:
        self.outputs = [None] * len(self.outputs)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def train_loss(
    model: CMEDualBranchModel,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    context: dict[str, Any],
    train_mask: torch.Tensor,
    label_override: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, embeddings, auxiliary = model(context["dataset_data"]["Feature"])
    labels = context["dataset_data"]["Label"] if label_override is None else label_override
    loss = criterion(logits, labels, train_mask, embeddings, auxiliary)
    require(bool(torch.isfinite(loss)), "Training loss is non-finite")
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid adapter gradient: {name}")
            gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Non-finite model gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
    optimizer.step()
    return float(loss.detach().cpu()), gradients


def run_scout(
    context: dict[str, Any], fold: int, epochs: int, fold_start_rng: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    reference_rank = int(context["protocol"]["reference"]["rank"])
    scout_ranks = [2 * reference_rank] * 5
    engine.restore_rng_state(fold_start_rng)
    model, criterion, optimizer, scheduler, audit = make_training_objects(context, scout_ranks)
    train_mask, test_mask, _ = engine.fold_positions(context, fold)
    safe_labels = context["dataset_data"]["Label"].clone()
    safe_labels[test_mask] = 0
    require(torch.equal(safe_labels[train_mask], context["dataset_data"]["Label"][train_mask]), "Train labels changed")
    capture = ImportanceCapture(model)
    ema = [torch.zeros(rank, device=context["device"], dtype=torch.float64) for rank in scout_ranks]
    losses = []
    try:
        for epoch in range(1, epochs + 1):
            capture.clear()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(context["dataset_data"]["Feature"])
            loss = criterion(logits, safe_labels, train_mask, embeddings, auxiliary)
            require(bool(torch.isfinite(loss)), "Scout loss is non-finite")
            loss.backward()
            for modality, activation in enumerate(capture.outputs):
                require(activation is not None and activation.grad is not None, "Scout bottleneck gradient missing")
                score = (activation[train_mask] * activation.grad[train_mask]).abs().mean(dim=0).to(torch.float64)
                require(bool(torch.isfinite(score).all()), "Scout importance is non-finite")
                ema[modality] = EMA_BETA * ema[modality] + (1.0 - EMA_BETA) * score
            require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Scout model gradient is non-finite")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
    finally:
        capture.close()
    correction = 1.0 - EMA_BETA ** epochs
    importance = [(values / correction).detach().cpu().tolist() for values in ema]
    flat = [float(value) for row in importance for value in row]
    require(flat and all(math.isfinite(value) and value >= 0.0 for value in flat), "Scout importance invalid")
    require(any(value > 0.0 for value in flat), "Scout importance is entirely zero")
    report = {
        "fold": fold,
        "epochs": epochs,
        "scout_ranks": scout_ranks,
        "importance_ema_beta": EMA_BETA,
        "importance_bias_correction": correction,
        "importance": importance,
        "losses": losses,
        "test_labels_used": False,
        "test_label_reads_for_importance": 0,
        "train_rows": int(train_mask.sum()),
        "test_rows_excluded": int(test_mask.sum()),
        "scheduler_t_max": int(context["protocol"]["training"]["scheduler_t_max"]),
        "scout_checkpoint_saved": False,
        "scout_parameter_count": audit["parameter_count"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return report


def allocate_ranks(context: dict[str, Any], scout: dict[str, Any]) -> dict[str, Any]:
    reference_rank = int(context["protocol"]["reference"]["rank"])
    rank_budget = 5 * reference_rank
    scout_rank = 2 * reference_rank
    ranks = [1] * 5
    candidates = []
    for modality, values in enumerate(scout["importance"]):
        require(len(values) == scout_rank, "Scout channel count changed")
        for channel, importance in enumerate(values):
            candidates.append((float(importance), modality, channel))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    remaining = rank_budget - 5
    selected = []
    for importance, modality, channel in candidates:
        if remaining == 0:
            break
        if ranks[modality] >= scout_rank:
            continue
        ranks[modality] += 1
        remaining -= 1
        selected.append({"modality_index": modality, "channel_index": channel, "importance": importance})
    require(remaining == 0 and sum(ranks) == rank_budget, "Rank budget was not fully allocated")
    hidden = int(context["protocol"]["training"]["hidden_size"])
    per_rank_cost = 2 * hidden + 1
    private_parameters = sum(rank * per_rank_cost + hidden for rank in ranks)
    reference_private = 5 * (reference_rank * per_rank_cost + hidden)
    require(private_parameters <= reference_private, "Private parameter budget exceeded")
    core = {
        "fold": int(scout["fold"]),
        "modality_names": [item["name"] for item in context["protocol"]["modalities"]],
        "ranks": ranks,
        "total_rank": sum(ranks),
        "rank_budget": rank_budget,
        "minimum_rank": 1,
        "per_rank_parameter_cost": per_rank_cost,
        "private_parameter_count": private_parameters,
        "reference_private_parameter_count": reference_private,
        "budget_compliant": private_parameters <= reference_private,
        "importance": scout["importance"],
        "selected_additional_channels": selected,
        "tie_break": "importance descending, modality index ascending, channel index ascending",
        "allocation_train_only": True,
    }
    return {**core, "allocation_hash": engine.payload_sha256(core)}


def fold_start_objects(
    context: dict[str, Any], allocation: dict[str, Any], audit_fairness: bool
) -> tuple[CMEDualBranchModel, Any, torch.optim.Optimizer, Any, dict[str, Any], dict[str, Any]]:
    reference_rank = int(context["protocol"]["reference"]["rank"])
    reference_ranks = [reference_rank] * 5
    SET_Random(SEED)
    fold_start_rng = engine.capture_rng_state()
    engine.restore_rng_state(fold_start_rng)
    reference = build_model(context, reference_ranks)
    post_reference_rng = engine.capture_rng_state()
    engine.restore_rng_state(fold_start_rng)
    model, criterion, optimizer, scheduler, object_audit = make_training_objects(
        context, list(allocation["ranks"])
    )
    fairness = fairness_audit(context, reference, model) if audit_fairness else {
        "common_state_pointwise_audit": "smoke_fold0_only"
    }
    require(object_audit["private_parameter_count"] == allocation["private_parameter_count"], "Formal private parameter count changed")
    require(object_audit["private_parameter_count"] <= allocation["reference_private_parameter_count"], "Formal budget exceeded")
    del reference
    engine.restore_rng_state(post_reference_rng)
    return model, criterion, optimizer, scheduler, object_audit, fairness


def reference_fold_replay(context: dict[str, Any], fold: int = 0) -> dict[str, Any]:
    protocol = context["protocol"]
    ranks = [int(protocol["reference"]["rank"])] * 5
    path = reference_paths(protocol)["checkpoints"][fold]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(context, ranks)
    model.load_state_dict(payload["best_model"], strict=True)
    train_mask, test_mask, _ = engine.fold_positions(context, fold)
    del train_mask
    logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy()
    rows = [row for row in typed_rows(reference_paths(protocol)["oof"]) if row["fold"] == fold]
    rows.sort(key=lambda row: int(row["original_csv_index"]))
    expected = sorted(context["fold_manifest"]["folds"][fold]["test_rows"], key=lambda row: int(row["original_csv_index"]))
    _, saved_logits, saved_probabilities = engine.rows_arrays(rows)
    order = np.argsort([int(row["original_csv_index"]) for row in context["fold_manifest"]["folds"][fold]["test_rows"]])
    logits = logits[order]
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    require([row["subject_id"] for row in rows] == [row["subject_id"] for row in expected], "Reference fold subjects changed")
    logits_diff = float(np.max(np.abs(logits - saved_logits)))
    probabilities_diff = float(np.max(np.abs(probabilities - saved_probabilities)))
    require(logits_diff <= 1e-6 and probabilities_diff <= 1e-6, "Reference checkpoint no longer reproduces OOF")
    del model
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "checkpoint_sha256": engine.file_sha256(path),
        "logits_max_abs_diff": logits_diff,
        "probabilities_max_abs_diff": probabilities_diff,
    }


def run_smoke(device_text: str) -> None:
    implementation_commit = source_gate(include_smoke=False)
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_text)
    protocols = load_protocols()
    task_reports = {}
    for task_id in TASK_IDS:
        context = engine.build_context(protocols[task_id], device)
        context["fold_manifest"] = engine.read_json(FOLD_MANIFEST_PATH)["tasks"][task_id]
        reference_replay = reference_fold_replay(context, 0)
        SET_Random(SEED)
        fold_start_rng = engine.capture_rng_state()
        scout = run_scout(context, 0, SMOKE_EPOCHS, fold_start_rng)
        allocation = allocate_ranks(context, scout)
        model, criterion, optimizer, scheduler, object_audit, fairness = fold_start_objects(
            context, allocation, audit_fairness=True
        )
        train_mask, test_mask, _ = engine.fold_positions(context, 0)
        cumulative = {name: 0.0 for name, _ in model.named_parameters() if name.startswith("private_adapters.")}
        losses = []
        for _epoch in range(1, SMOKE_EPOCHS + 1):
            loss, gradients = train_loss(model, criterion, optimizer, context, train_mask)
            scheduler.step()
            losses.append(loss)
            for name, value in gradients.items():
                cumulative[name] = max(cumulative[name], value)
        require(cumulative and all(math.isfinite(value) and value > 0.0 for value in cumulative.values()), "Smoke adapter gradients are invalid")
        logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask]
        probabilities = torch.softmax(logits, dim=-1)
        require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probabilities).all()), "Smoke outputs invalid")
        simplex_error = float((probabilities.sum(dim=1) - 1.0).abs().max().cpu())
        require(simplex_error <= 1e-6, "Smoke probability simplex failed")
        checkpoint_path = SMOKE_DIR / f"{task_id}_checkpoint_roundtrip.pt"
        checkpoint = {
            "implementation_commit": implementation_commit,
            "task_id": task_id,
            "fold": 0,
            "epoch": SMOKE_EPOCHS,
            "allocation": allocation,
            "model": engine.clone_cpu_state(model),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": engine.capture_rng_state(),
        }
        engine.atomic_torch_save(checkpoint_path, checkpoint)
        fresh_model, _, fresh_optimizer, fresh_scheduler, _ = make_training_objects(context, allocation["ranks"])
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fresh_model.load_state_dict(loaded["model"], strict=True)
        fresh_optimizer.load_state_dict(loaded["optimizer"])
        fresh_scheduler.load_state_dict(loaded["scheduler"])
        replay = engine.infer(fresh_model, context["dataset_data"]["Feature"])[0][test_mask]
        reload_diff = float((replay.detach().cpu() - logits.detach().cpu()).abs().max())
        require(reload_diff <= 1e-6 and int(fresh_scheduler.last_epoch) == SMOKE_EPOCHS, "Smoke strict reload failed")
        task_reports[task_id] = {
            "reference_checkpoint_replay": reference_replay,
            "scout": scout,
            "allocation": allocation,
            "fairness": fairness,
            "object_audit": object_audit,
            "formal_losses": losses,
            "adapter_cumulative_max_gradient": cumulative,
            "formal_steps": SMOKE_EPOCHS,
            "scout_weight_transfer_count": 0,
            "scout_state_used_by_formal": False,
            "simplex_max_abs_error": simplex_error,
            "checkpoint_roundtrip_logits_max_abs_diff": reload_diff,
            "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        }
        del model, fresh_model, optimizer, fresh_optimizer, scheduler, fresh_scheduler, criterion
        torch.cuda.empty_cache()
    smoke_config_core = {
        "experiment": EXPERIMENT_ID,
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes(),
        "inspect_sha256": engine.file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": engine.file_sha256(FOLD_MANIFEST_PATH),
        "device": device_text,
        "scout_epochs": SMOKE_EPOCHS,
        "formal_epochs": SMOKE_EPOCHS,
    }
    smoke_config = {**smoke_config_core, "sha256": engine.payload_sha256(smoke_config_core)}
    engine.atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    report_core = {
        "passed": True,
        "config_sha256": smoke_config["sha256"],
        "tasks": task_reports,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    engine.atomic_write_json(SMOKE_DIR / "smoke_report.json", {**report_core, "sha256": engine.payload_sha256(report_core)})
    print(json.dumps({"passed": True, "tasks": {task: task_reports[task]["allocation"]["ranks"] for task in TASK_IDS}}, indent=2))


def validate_smoke_and_source() -> tuple[str, dict[str, Any]]:
    source_commit = source_gate(include_smoke=True)
    config = engine.read_json(SMOKE_DIR / "smoke_config.json")
    report = engine.read_json(SMOKE_DIR / "smoke_report.json")
    config_core = {key: value for key, value in config.items() if key != "sha256"}
    report_core = {key: value for key, value in report.items() if key != "sha256"}
    require(config["sha256"] == engine.payload_sha256(config_core), "Smoke config digest changed")
    require(report["sha256"] == engine.payload_sha256(report_core), "Smoke report digest changed")
    require(report["passed"] is True and report["config_sha256"] == config["sha256"], "Smoke did not pass")
    require(git("merge-base", "--is-ancestor", config["implementation_commit"], source_commit, check=False).returncode == 0, "Smoke implementation commit is not an ancestor")
    for task in TASK_IDS:
        require(report["tasks"][task]["scout_state_used_by_formal"] is False, "Scout state transfer detected")
        require(report["tasks"][task]["allocation"]["budget_compliant"] is True, "Smoke allocation budget failed")
    return source_commit, report


def runtime_lock(source_commit: str, device_text: str) -> dict[str, Any]:
    protocols = load_protocols()
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "experiment_config_sha256": engine.file_sha256(CONFIG_PATH),
        "protocol_sha256": {task: engine.file_sha256(protocol_paths()[task]) for task in TASK_IDS},
        "inspect_file_sha256": engine.file_sha256(INSPECT_PATH),
        "fold_manifest_file_sha256": engine.file_sha256(FOLD_MANIFEST_PATH),
        "smoke_config_sha256": engine.file_sha256(SMOKE_DIR / "smoke_config.json"),
        "smoke_report_sha256": engine.file_sha256(SMOKE_DIR / "smoke_report.json"),
        "tasks": list(TASK_IDS),
        "folds": list(FOLDS),
        "seed": SEED,
        "scout_epochs": SCOUT_EPOCHS,
        "formal_epochs": FORMAL_EPOCHS,
        "device": device_text,
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "training": {task: protocols[task]["training"] for task in TASK_IDS},
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {
        "runtime_sha256": runtime["sha256"],
        "task_id": context["task_id"],
        "fold": fold,
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_paths(task_id: str, fold: int) -> dict[str, Path]:
    root = WORK_DIR / task_id / f"fold_{fold:02d}"
    return {
        "root": root,
        "resume": root / "resume.pt",
        "allocation": root / "allocation.json",
        "scout": root / "scout_summary.json",
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
    allocation: dict[str, Any],
    best: dict[str, Any],
    history: list[dict[str, Any]],
    gradients: dict[str, float],
    initial_private: dict[str, torch.Tensor],
    elapsed: float,
) -> None:
    engine.atomic_torch_save(
        path,
        {
            "schema": 1,
            "lock": lock,
            "epoch": epoch,
            "allocation": allocation,
            "model": engine.clone_cpu_state(model),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "best": best,
            "history": history,
            "cumulative_gradient": gradients,
            "initial_private": initial_private,
            "rng": engine.capture_rng_state(),
            "elapsed_seconds": elapsed,
        },
    )


def formal_private_diagnostics(
    context: dict[str, Any], model: CMEDualBranchModel, test_mask: torch.Tensor,
    cumulative_gradient: dict[str, float], initial_private: dict[str, torch.Tensor]
) -> dict[str, Any]:
    return engine.private_diagnostics(context, model, test_mask, cumulative_gradient, initial_private)


def materialize_fold(
    context: dict[str, Any], fold: int, runtime: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context["task_id"], fold)
    if paths["complete"].is_file():
        return load_completed_fold(context, fold, runtime)
    paths["root"].mkdir(parents=True, exist_ok=True)
    lock = fold_lock(runtime, context, fold)
    train_mask, test_mask, _ = engine.fold_positions(context, fold)
    start_epoch = 1
    resumed_from_epoch = 0
    elapsed_before = 0.0
    fairness: dict[str, Any] = {"common_state_pointwise_audit": "smoke_fold0_only"}
    if paths["resume"].is_file():
        require(paths["allocation"].is_file() and paths["scout"].is_file(), "Resume allocation/scout evidence missing")
        allocation = engine.read_json(paths["allocation"])
        scout = engine.read_json(paths["scout"])
        resumed = torch.load(paths["resume"], map_location="cpu", weights_only=False)
        require(resumed["lock"] == lock and resumed["allocation"] == allocation, "Resume lock/allocation changed")
        model, criterion, optimizer, scheduler, object_audit = make_training_objects(context, allocation["ranks"])
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"]) + 1
        resumed_from_epoch = int(resumed["epoch"])
        best = resumed["best"]
        history = resumed["history"]
        cumulative_gradient = {name: float(value) for name, value in resumed["cumulative_gradient"].items()}
        initial_private = {name: value.clone() for name, value in resumed["initial_private"].items()}
        elapsed_before = float(resumed["elapsed_seconds"])
        engine.restore_rng_state(resumed["rng"])
        require([row["epoch"] for row in history] == list(range(1, start_epoch)), "Resume history changed")
    else:
        SET_Random(SEED)
        fold_start_rng = engine.capture_rng_state()
        scout = run_scout(context, fold, SCOUT_EPOCHS, fold_start_rng)
        allocation = allocate_ranks(context, scout)
        engine.atomic_write_json(paths["scout"], scout)
        engine.atomic_write_json(paths["allocation"], allocation)
        model, criterion, optimizer, scheduler, object_audit, fairness = fold_start_objects(
            context, allocation, audit_fairness=(fold == 0)
        )
        best = None
        history = []
        cumulative_gradient = {name: 0.0 for name, _ in model.named_parameters() if name.startswith("private_adapters.")}
        initial_private = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
    require(allocation["allocation_hash"] == engine.payload_sha256({key: value for key, value in allocation.items() if key != "allocation_hash"}), "Allocation digest changed")
    session_start = time.perf_counter()
    for epoch in range(start_epoch, FORMAL_EPOCHS + 1):
        loss, gradients = train_loss(model, criterion, optimizer, context, train_mask)
        for name, value in gradients.items():
            cumulative_gradient[name] = max(cumulative_gradient.get(name, 0.0), value)
        logits_t = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu()
        probabilities_t = torch.softmax(logits_t, dim=-1)
        truth = context["dataset_data"]["Label"][test_mask].detach().cpu().numpy().astype(np.int64)
        logits = logits_t.numpy().astype(np.float64)
        probabilities = probabilities_t.numpy().astype(np.float64)
        metrics = engine.binary_metrics(truth, probabilities, context["protocol"])
        row = {
            "epoch": epoch, "loss": loss, "correct": metrics["correct"], "acc": metrics["acc"],
            "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
            "macro_f1": metrics["macro_f1"], "bacc": metrics["bacc"],
        }
        history.append(row)
        if best is None or engine.selection_key(metrics, epoch) > tuple(best["selection_key"]):
            best = {
                "epoch": epoch,
                "selection_key": list(engine.selection_key(metrics, epoch)),
                "metrics": metrics,
                "model": engine.clone_cpu_state(model),
                "test_logits": logits,
                "test_probabilities": probabilities,
                "test_truth": truth,
            }
        scheduler.step()
        elapsed = elapsed_before + time.perf_counter() - session_start
        if epoch % 20 == 0 or epoch == FORMAL_EPOCHS:
            save_resume(paths["resume"], lock, model, optimizer, scheduler, epoch, allocation, best, history, cumulative_gradient, initial_private, elapsed)
    require(best is not None and len(history) == FORMAL_EPOCHS, "Formal training did not complete")
    model.load_state_dict(best["model"], strict=True)
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    infer_start = time.perf_counter()
    replay_logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - infer_start
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - best["test_logits"]))) <= 1e-6, "Best checkpoint logits changed")
    require(float(np.max(np.abs(replay_probabilities - best["test_probabilities"]))) <= 1e-6, "Best checkpoint probabilities changed")
    rows = engine.prediction_rows(context, fold, best["test_logits"], best["test_probabilities"], best["test_truth"])
    for row in rows:
        row["arm"] = "MA_ARA"
    diagnostics = formal_private_diagnostics(context, model, test_mask, cumulative_gradient, initial_private)
    require(all(value > 0.0 and math.isfinite(value) for value in diagnostics["adapter_cumulative_max_gradient"].values()), "Formal adapter gradient audit failed")
    require(all(value > 0.0 and math.isfinite(value) for value in diagnostics["adapter_parameter_max_delta"].values()), "Formal adapter parameter-update audit failed")
    require(diagnostics["private_trained"] and not diagnostics["private_collapse"], "Private adapter collapse")
    elapsed_total = elapsed_before + time.perf_counter() - session_start
    checkpoint = {
        "schema": 1, "lock": lock, "task_id": context["task_id"], "fold": fold,
        "allocation_hash": allocation["allocation_hash"], "ranks": allocation["ranks"],
        "best_epoch": best["epoch"], "best_metrics": best["metrics"], "best_model": best["model"],
        "formal_epochs": FORMAL_EPOCHS, "scout_weight_transfer_count": 0,
    }
    engine.atomic_torch_save(paths["checkpoint"], checkpoint)
    engine.atomic_write_csv(paths["oof"], rows)
    engine.atomic_write_csv(paths["history"], history)
    summary = {
        "schema": 1, "lock": lock, "task_id": context["task_id"], "fold": fold,
        "best_epoch": best["epoch"], "best_metrics": best["metrics"],
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "allocation": allocation, "scout_summary_sha256": engine.file_sha256(paths["scout"]),
        "object_audit": object_audit, "fairness": fairness, "diagnostics": diagnostics,
        "scout_state_used_by_formal": False, "scout_weight_transfer_count": 0,
        "allocation_train_only": True, "optimizer_steps": FORMAL_EPOCHS,
        "scheduler_steps": FORMAL_EPOCHS, "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds": elapsed_total, "inference_seconds_full_graph": inference_seconds,
        "oof_count": len(rows),
    }
    engine.atomic_write_json(paths["summary"], summary)
    marker_core = {
        "complete": True, "lock": lock,
        "allocation_sha256": engine.file_sha256(paths["allocation"]),
        "scout_sha256": engine.file_sha256(paths["scout"]),
        "checkpoint_sha256": engine.file_sha256(paths["checkpoint"]),
        "oof_sha256": engine.file_sha256(paths["oof"]),
        "history_sha256": engine.file_sha256(paths["history"]),
        "summary_sha256": engine.file_sha256(paths["summary"]),
    }
    engine.atomic_write_json(paths["complete"], {**marker_core, "sha256": engine.payload_sha256(marker_core)})
    if paths["resume"].is_file():
        paths["resume"].unlink()
    require(experiment_size() <= MAX_EXPERIMENT_BYTES, "Experiment directory exceeded 2 GB")
    print(f"[{context['task_id']} fold {fold}] ranks={allocation['ranks']} best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    return load_completed_fold(context, fold, runtime)


def load_completed_fold(
    context: dict[str, Any], fold: int, runtime: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context["task_id"], fold)
    require(all(paths[name].is_file() for name in ("allocation", "scout", "checkpoint", "oof", "history", "summary", "complete")), "Completed fold artifacts missing")
    require(not paths["resume"].exists(), "Completed fold retained resume.pt")
    lock = fold_lock(runtime, context, fold)
    marker = engine.read_json(paths["complete"])
    marker_core = {key: value for key, value in marker.items() if key != "sha256"}
    require(marker["sha256"] == engine.payload_sha256(marker_core) and marker["lock"] == lock, "Complete marker changed")
    for key, name in (("allocation_sha256", "allocation"), ("scout_sha256", "scout"), ("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("history_sha256", "history"), ("summary_sha256", "summary")):
        require(marker[key] == engine.file_sha256(paths[name]), f"Completed {name} changed")
    allocation = engine.read_json(paths["allocation"])
    require(allocation["allocation_hash"] == engine.payload_sha256({key: value for key, value in allocation.items() if key != "allocation_hash"}), "Allocation hash changed")
    require(allocation["budget_compliant"] and allocation["allocation_train_only"], "Completed allocation invalid")
    summary = engine.read_json(paths["summary"])
    require(summary["lock"] == lock and summary["allocation"] == allocation, "Completed summary changed")
    rows = typed_rows(paths["oof"])
    require(len(rows) == summary["oof_count"], "Completed OOF count changed")
    truth, saved_logits, saved_probabilities = engine.rows_arrays(rows)
    require(engine.metrics_match(engine.binary_metrics(truth, saved_probabilities, context["protocol"]), summary["best_metrics"]), "Completed OOF metrics changed")
    history = engine.read_csv_rows(paths["history"])
    require([int(row["epoch"]) for row in history] == list(range(1, FORMAL_EPOCHS + 1)), "Formal history changed")
    selected = max(history, key=lambda row: (float(row["acc"]), float(row["roc_auc"]), float(row["macro_f1"]), -int(row["epoch"])))
    require(int(selected["epoch"]) == summary["best_epoch"], "Best epoch no longer reproduces")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint["lock"] == lock and checkpoint["allocation_hash"] == allocation["allocation_hash"], "Best checkpoint changed")
    model = build_model(context, allocation["ranks"])
    model.load_state_dict(checkpoint["best_model"], strict=True)
    _, test_mask, _ = engine.fold_positions(context, fold)
    replay_logits = engine.infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - saved_logits))) <= 1e-6, "Completed checkpoint logits changed")
    require(float(np.max(np.abs(replay_probabilities - saved_probabilities))) <= 1e-6, "Completed checkpoint probabilities changed")
    del model
    torch.cuda.empty_cache()
    return summary, rows


def exact_mcnemar_p(repairs: int, damages: int) -> float:
    discordant = repairs + damages
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(0, min(repairs, damages) + 1)) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_reference(
    context: dict[str, Any], result: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_rows = typed_rows(reference_paths(context["protocol"])["oof"])
    left = {row["subject_id"]: row for row in reference_rows}
    right = {row["subject_id"]: row for row in rows}
    require(left.keys() == right.keys(), "Reference/MA-ARA subject sets changed")
    repairs = damages = changed = 0
    for subject_id in left:
        old, new = left[subject_id], right[subject_id]
        require(old["truth"] == new["truth"] and old["feature_sha256"] == new["feature_sha256"], "Paired truth/features changed")
        old_correct = old["prediction"] == old["truth"]
        new_correct = new["prediction"] == new["truth"]
        repairs += int(not old_correct and new_correct)
        damages += int(old_correct and not new_correct)
        changed += int(old["prediction"] != new["prediction"])
    reference = reference_metrics(context["protocol"], reference_rows)
    deltas = {
        key: float(result["metrics"][key] - reference[key])
        for key in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe")
    }
    return {
        "reference_trial_id": context["protocol"]["reference"]["trial_id"],
        "reference_metrics": reference,
        "repairs": repairs,
        "damages": damages,
        "changed": changed,
        "net_correct": repairs - damages,
        "correct_delta": int(result["metrics"]["correct"] - reference["correct"]),
        "metric_deltas": deltas,
        "exact_mcnemar_p": exact_mcnemar_p(repairs, damages),
    }


def aggregate_allocations(context: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = [entry["name"] for entry in context["protocol"]["modalities"]]
    rank_matrix = np.asarray([summary["allocation"]["ranks"] for summary in summaries], dtype=np.int64)
    importance = np.asarray([summary["allocation"]["importance"] for summary in summaries], dtype=np.float64)
    frequency = {}
    for modality_index, modality in enumerate(modalities):
        values, counts = np.unique(rank_matrix[:, modality_index], return_counts=True)
        frequency[modality] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    pairwise_l1 = [
        float(np.abs(rank_matrix[left] - rank_matrix[right]).sum())
        for left in range(len(rank_matrix))
        for right in range(left + 1, len(rank_matrix))
    ]
    ratio_mean = {
        modality: float(statistics.mean(summary["diagnostics"]["private_shared_ratio_mean_by_modality"][modality] for summary in summaries))
        for modality in modalities
    }
    max_gradient = max(
        value
        for summary in summaries
        for value in summary["diagnostics"]["adapter_cumulative_max_gradient"].values()
    )
    return {
        "fold_allocations": [summary["allocation"] for summary in summaries],
        "fold_ranks": [summary["allocation"]["ranks"] for summary in summaries],
        "allocation_hashes": [summary["allocation"]["allocation_hash"] for summary in summaries],
        "rank_mean_by_modality": {modality: float(rank_matrix[:, index].mean()) for index, modality in enumerate(modalities)},
        "rank_sample_sd_by_modality": {modality: float(rank_matrix[:, index].std(ddof=1)) for index, modality in enumerate(modalities)},
        "rank_frequency_by_modality": frequency,
        "fold_pairwise_rank_l1_mean": float(statistics.mean(pairwise_l1)),
        "fold_pairwise_rank_l1_max": float(max(pairwise_l1)),
        "importance_mean_by_modality": {modality: float(importance[:, index, :].mean()) for index, modality in enumerate(modalities)},
        "importance_max_by_modality": {modality: float(importance[:, index, :].max()) for index, modality in enumerate(modalities)},
        "private_shared_norm_ratio_mean_by_modality": ratio_mean,
        "adapter_max_gradient": float(max_gradient),
        "private_collapse_any": any(summary["diagnostics"]["private_collapse"] for summary in summaries),
        "parameter_budget_all_folds": all(summary["allocation"]["budget_compliant"] for summary in summaries),
        "allocation_train_only_all_folds": all(summary["allocation"]["allocation_train_only"] for summary in summaries),
        "scout_weight_transfer_count": sum(summary["scout_weight_transfer_count"] for summary in summaries),
    }


def aggregate_task(
    context: dict[str, Any], summaries: list[dict[str, Any]], fold_rows: list[list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(summaries) == len(fold_rows) == 10, "Formal fold count changed")
    rows = sorted([row for group in fold_rows for row in group], key=lambda row: row["original_csv_index"])
    require(len(rows) == context["protocol"]["sample_count"] and len({row["subject_id"] for row in rows}) == len(rows), "OOF coverage changed")
    expected = sorted([row for fold in context["fold_manifest"]["folds"] for row in fold["test_rows"]], key=lambda row: row["original_csv_index"])
    for row, anchor in zip(rows, expected):
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"], "OOF stable-ID alignment changed")
    truth, _, probabilities = engine.rows_arrays(rows)
    metrics = engine.binary_metrics(truth, probabilities, context["protocol"])
    fold_metrics = [
        {
            "fold": summary["fold"], "best_epoch": summary["best_epoch"],
            "correct": summary["best_metrics"]["correct"], "acc": summary["best_metrics"]["acc"],
            "roc_auc": summary["best_metrics"]["roc_auc"], "pr_auc": summary["best_metrics"]["pr_auc"],
            "macro_f1": summary["best_metrics"]["macro_f1"], "bacc": summary["best_metrics"]["bacc"],
            "ranks": summary["allocation"]["ranks"], "allocation_hash": summary["allocation"]["allocation_hash"],
            "scout_seconds": engine.read_json(fold_paths(context["task_id"], summary["fold"])["scout"])["elapsed_seconds"],
            "formal_seconds": summary["elapsed_seconds"], "inference_seconds": summary["inference_seconds_full_graph"],
            "resumed_from_epoch": summary["resumed_from_epoch"],
        }
        for summary in summaries
    ]
    oof_path = RESULT_DIR / f"{context['task_id']}_oof_predictions.csv"
    engine.atomic_write_csv(oof_path, rows)
    allocation_path = RESULT_DIR / f"{context['task_id']}_allocations.json"
    allocation_summary = aggregate_allocations(context, summaries)
    engine.atomic_write_json(allocation_path, allocation_summary)
    result = {
        "task_id": context["task_id"],
        "metrics": metrics,
        "fold_roc_auc_mean": float(statistics.mean(row["roc_auc"] for row in fold_metrics)),
        "fold_roc_auc_sample_sd": float(statistics.stdev(row["roc_auc"] for row in fold_metrics)),
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)),
        "fold_acc_sample_sd": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_metrics": fold_metrics,
        "parameter_count": summaries[0]["object_audit"]["parameter_count"],
        "private_parameter_count": summaries[0]["object_audit"]["private_parameter_count"],
        "reference_parameter_count": engine.read_json(INSPECT_PATH)["tasks"][context["task_id"]]["reference_parameter_count"],
        "reference_private_parameter_count": engine.read_json(INSPECT_PATH)["tasks"][context["task_id"]]["reference_private_parameter_count"],
        "training_time_seconds": float(sum(row["scout_seconds"] + row["formal_seconds"] for row in fold_metrics)),
        "scout_time_seconds": float(sum(row["scout_seconds"] for row in fold_metrics)),
        "formal_time_seconds": float(sum(row["formal_seconds"] for row in fold_metrics)),
        "inference_time_seconds": float(sum(row["inference_seconds"] for row in fold_metrics)),
        "mechanism": allocation_summary,
        "oof_path": oof_path.relative_to(ROOT).as_posix(),
        "oof_sha256": engine.file_sha256(oof_path),
        "allocation_path": allocation_path.relative_to(ROOT).as_posix(),
        "allocation_sha256": engine.file_sha256(allocation_path),
    }
    result["paired_reference"] = paired_reference(context, result, rows)
    return result, rows


def performance_safety(task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    comparison = result["paired_reference"]
    deltas = comparison["metric_deltas"]
    pr_limit = -0.010 if task_id == "tadpole_smci_pmci" else -0.005
    return {
        "bacc": deltas["bacc"] >= -0.003,
        "roc_auc": deltas["roc_auc"] >= -0.005,
        "pr_auc": deltas["pr_auc"] >= pr_limit,
        "sen": deltas["sen"] >= -0.01,
        "spe": deltas["spe"] >= -0.01,
    }


def final_decision(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = []
    deltas = {task: result["paired_reference"]["correct_delta"] for task, result in results.items()}
    safeties = {task: performance_safety(task, result) for task, result in results.items()}
    stop = False
    for task, result in results.items():
        comparison = result["paired_reference"]
        metric_deltas = comparison["metric_deltas"]
        mechanism = result["mechanism"]
        if deltas[task] < 0:
            stop, reasons = True, reasons + [f"{task}: Correct below reference"]
        if metric_deltas["bacc"] < -0.005:
            stop, reasons = True, reasons + [f"{task}: BACC drop exceeds .005"]
        if metric_deltas["roc_auc"] < -0.01:
            stop, reasons = True, reasons + [f"{task}: ROC-AUC drop exceeds .01"]
        if mechanism["private_collapse_any"]:
            stop, reasons = True, reasons + [f"{task}: private collapse"]
        if not mechanism["parameter_budget_all_folds"]:
            stop, reasons = True, reasons + [f"{task}: parameter budget failure"]
        if not mechanism["allocation_train_only_all_folds"]:
            stop, reasons = True, reasons + [f"{task}: allocation leakage"]
        if mechanism["scout_weight_transfer_count"] != 0:
            stop, reasons = True, reasons + [f"{task}: Scout weight transfer"]
    if stop:
        return "MA_ARA_PRESSURE_STOP", reasons
    pass_ok = (
        results["tadpole_smci_pmci"]["metrics"]["correct"] >= 520
        and results["abide5_ads_cn"]["metrics"]["correct"] >= 770
        and all(delta >= 1 for delta in deltas.values())
        and all(result["paired_reference"]["repairs"] > result["paired_reference"]["damages"] for result in results.values())
        and all(all(safety.values()) for safety in safeties.values())
    )
    if pass_ok:
        return "MA_ARA_PRESSURE_PASS", ["Both tasks improved and all safety/budget/leakage gates passed"]
    near_ok = sorted(deltas.values()) == [0, 1] and all(all(safety.values()) for safety in safeties.values())
    if near_ok:
        return "MA_ARA_PRESSURE_NEAR", ["One task improved by one and the other tied; all safety gates passed"]
    return "MA_ARA_PRESSURE_STOP", ["Neither the pre-registered PASS nor NEAR rule was satisfied"]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MA-ARA Pressure Validation v1",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Branch: `{BRANCH}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Result commit: reported in the final Git handoff (self-reference is not embedded).",
        f"- Device: `{summary['device']}`",
        "- Protocol: one seed (0), ten folds, Scout 60 epochs, fresh Formal 400 epochs, single model, graph/EMA/ensemble disabled.",
        "- Selection: ACC > ROC-AUC > Macro-F1 > earliest epoch.",
        "",
        "## Results",
        "",
        "| Task | Correct | ACC | ROC-AUC | fold AUC mean+/-SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | params/private | train s | infer s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASK_IDS:
        result = summary["tasks"][task]
        metrics = result["metrics"]
        paired = result["paired_reference"]
        lines.append(
            f"| {task} | {metrics['correct']}/{metrics['n']} | {metrics['acc']:.7f} | {metrics['roc_auc']:.7f} | "
            f"{result['fold_roc_auc_mean']:.7f}+/-{result['fold_roc_auc_sample_sd']:.7f} | {metrics['pr_auc']:.7f} | "
            f"{metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | {metrics['weighted_f1']:.7f} | {metrics['sen']:.7f} | {metrics['spe']:.7f} | "
            f"`{metrics['confusion_matrix']}` | `{metrics['predicted_counts']}` | {paired['repairs']}/{paired['damages']}/{paired['changed']} | "
            f"{paired['exact_mcnemar_p']:.7g} | {result['fold_acc_mean']:.7f}+/-{result['fold_acc_sample_sd']:.7f} | {result['parameter_count']}/{result['private_parameter_count']} | {result['training_time_seconds']:.2f} | {result['inference_time_seconds']:.4f} |"
        )
        lines.extend([
            "",
            f"### {task} folds and mechanism",
            "",
            "| Fold | Best epoch | Correct | ACC | AUC | ranks | allocation hash |",
            "|---:|---:|---:|---:|---:|---|---|",
        ])
        for fold in result["fold_metrics"]:
            lines.append(f"| {fold['fold']} | {fold['best_epoch']} | {fold['correct']} | {fold['acc']:.7f} | {fold['roc_auc']:.7f} | `{fold['ranks']}` | `{fold['allocation_hash']}` |")
        mechanism = result["mechanism"]
        lines.extend([
            "",
            f"- Rank mean by modality: `{json.dumps(mechanism['rank_mean_by_modality'], sort_keys=True)}`",
            f"- Rank sample SD by modality: `{json.dumps(mechanism['rank_sample_sd_by_modality'], sort_keys=True)}`",
            f"- Rank frequency: `{json.dumps(mechanism['rank_frequency_by_modality'], sort_keys=True)}`",
            f"- Fold allocation stability (pairwise L1 mean/max): {mechanism['fold_pairwise_rank_l1_mean']:.3f}/{mechanism['fold_pairwise_rank_l1_max']:.3f}",
            f"- Importance mean/max by modality: `{json.dumps(mechanism['importance_mean_by_modality'], sort_keys=True)}` / `{json.dumps(mechanism['importance_max_by_modality'], sort_keys=True)}`",
            f"- Private/shared norm ratio mean: `{json.dumps(mechanism['private_shared_norm_ratio_mean_by_modality'], sort_keys=True)}`",
            f"- Adapter maximum gradient: {mechanism['adapter_max_gradient']:.8g}; private collapse: {mechanism['private_collapse_any']}",
            f"- Budget all folds: {mechanism['parameter_budget_all_folds']}; train-only allocation all folds: {mechanism['allocation_train_only_all_folds']}; Scout weight transfers: {mechanism['scout_weight_transfer_count']}",
            f"- Metric deltas vs reference: `{json.dumps(paired['metric_deltas'], sort_keys=True)}`; Correct delta={paired['correct_delta']}",
        ])
    tad = summary["tasks"]["tadpole_smci_pmci"]
    abide5 = summary["tasks"]["abide5_ads_cn"]
    lines.extend([
        "",
        "## Required answers",
        "",
        f"1. TADPOLE exceeds D5 519/535: **{tad['metrics']['correct'] > 519}** ({tad['metrics']['correct']}/535).",
        f"2. ABIDE-5 exceeds D3 769/864: **{abide5['metrics']['correct'] > 769}** ({abide5['metrics']['correct']}/864).",
        f"3. Repairs exceed damages on both tasks: **{all(result['paired_reference']['repairs'] > result['paired_reference']['damages'] for result in summary['tasks'].values())}**.",
        f"4. Main modality allocations are given by rank means: TAD `{json.dumps(tad['mechanism']['rank_mean_by_modality'], sort_keys=True)}`; ABIDE-5 `{json.dumps(abide5['mechanism']['rank_mean_by_modality'], sort_keys=True)}`.",
        f"5. Ten-fold stability: TAD pairwise L1 mean {tad['mechanism']['fold_pairwise_rank_l1_mean']:.3f}; ABIDE-5 {abide5['mechanism']['fold_pairwise_rank_l1_mean']:.3f}; full frequencies are above.",
        f"6. Parameter counts do not exceed references: **{all(result['parameter_count'] <= result['reference_parameter_count'] and result['private_parameter_count'] <= result['reference_private_parameter_count'] for result in summary['tasks'].values())}**.",
        f"7. Scout weights were not used by Formal or inference: **{all(result['mechanism']['scout_weight_transfer_count'] == 0 for result in summary['tasks'].values())}**; only train-derived allocations were carried over.",
        f"8. MA_ARA_PRESSURE_PASS satisfied: **{summary['decision'] == 'MA_ARA_PRESSURE_PASS'}**.",
        f"9. Expansion to ABIDE/TADPOLE-three-class is recommended only on PASS: **{summary['decision'] == 'MA_ARA_PRESSURE_PASS'}**; it was not run here.",
        f"10. Source `{summary['source_commit']}`, result commit in final Git handoff, branch `{BRANCH}`, device `{summary['device']}`. Reproduction: `python -u -B scripts/run_ma_ara_pressure_v1.py inspect`; then `... smoke --device {summary['device']}`; then `... formal --device {summary['device']}` from the recorded source commit.",
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
    protocols = load_protocols()
    task_results = {}
    for task_id in TASK_IDS:
        context = engine.build_context(protocols[task_id], device)
        context["fold_manifest"] = engine.read_json(FOLD_MANIFEST_PATH)["tasks"][task_id]
        summaries, fold_rows = [], []
        for fold in FOLDS:
            summary, rows = materialize_fold(context, fold, runtime)
            summaries.append(summary)
            fold_rows.append(rows)
        task_result, _ = aggregate_task(context, summaries, fold_rows)
        task_results[task_id] = task_result
        engine.atomic_write_json(RESULT_DIR / f"{task_id}_fold_results.json", {"task_id": task_id, "folds": task_result["fold_metrics"]})
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
        "scout_checkpoint_saved": False,
        "experiment_logical_bytes": experiment_size(),
    }
    report_text = render_report(summary_core)
    engine.atomic_write_text(RESULT_DIR / "REPORT.md", report_text)
    summary_core["report_sha256"] = engine.file_sha256(RESULT_DIR / "REPORT.md")
    summary = {**summary_core, "sha256": engine.payload_sha256(summary_core)}
    engine.atomic_write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps({"decision": decision, "correct": {task: result["metrics"]["correct"] for task, result in task_results.items()}, "summary_sha256": summary["sha256"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect", help="Validate data, references, folds, checkpoints, budget, and disk")
    smoke = subparsers.add_parser("smoke", help="Run fold-0 3+3 epoch mechanism smoke on both tasks")
    smoke.add_argument("--device", default="cuda:0")
    formal = subparsers.add_parser("formal", help="Run/strictly resume both formal ten-fold suites")
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
