"""PG-PAL-ABIDE5 v1: protected-gradient training for the fixed D3 model."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_abide5_hparam_search_v1 as abide5

# Install the verified ABIDE-5 profile into the shared binary-task engine.
abide5.install_profile()
engine = abide5.engine
historical = engine.historical

EXPERIMENT_ID = "pg_pal_abide5_v1"
BRANCH = "experiment/pg-pal-abide5-v1"
BASE_COMMIT = "a23134cf23e3c95cb237931d92c327c001545a06"
D3_SOURCE_COMMIT = "111f5dcd8350729bfbd227f8116ce158a259bbea"
D3_RESULT_COMMIT = BASE_COMMIT
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
FORMAL_DIR = RESULT_DIR / "formal"
COMPACT_FOLDS_DIR = RESULT_DIR / "folds"
D3_DIR = ROOT / "experiments" / "abide5_hparam_search_v1" / "trials" / "D3"
D3_OOF_PATH = D3_DIR / "oof_predictions.csv"
D3_SUMMARY_PATH = D3_DIR / "summary.json"
D3_CONFIG_PATH = D3_DIR / "config.json"
EPSILON = 1e-12
EPOCHS = 400
FOLDS = tuple(range(10))
SMOKE_EPOCHS = 3
EXPECTED_PARAMETERS = 383_616
EXPECTED_PRIVATE_TENSORS = 20
EXPECTED_MODALITIES = 5
COMMON_NONE_WHITELIST = {
    "Adj_Learning.layer_.weight": "graph=False makes learned adjacency irrelevant to classification",
    "Adj_Learning.layer_.bias": "graph=False makes learned adjacency irrelevant to classification",
}
REQUIRED_SOURCE = (
    ".gitignore",
    "scripts/run_pg_pal_abide5_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
)
LOCKED_DEPENDENCIES = (
    "Model/cme_dual_branch.py",
    "Model/network.py",
    "Model/models.py",
    "Loss/loss_fn.py",
    "Utils/data_load.py",
    "Utils/graph_load.py",
    "Utils/utils.py",
    "config_re.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
    "scripts/run_tad_binary_hparam_search_v1.py",
    "scripts/run_abide_hparam_search_v1.py",
    "scripts/run_abide5_hparam_search_v1.py",
    "experiments/abide5_hparam_search_v1/protocol.json",
    "experiments/abide5_hparam_search_v1/experiment_config.json",
    "experiments/abide5_hparam_search_v1/fold_manifest.json",
    "experiments/abide5_hparam_search_v1/trials/D3/config.json",
    "experiments/abide5_hparam_search_v1/trials/D3/summary.json",
    "experiments/abide5_hparam_search_v1/trials/D3/oof_predictions.csv",
)


def require(condition: bool, message: str) -> None:
    engine.require(condition, message)


def spec() -> dict[str, Any]:
    return engine.trial_spec(
        "PG_PAL", "formal", "B1", 0.005, 0.0005, 0.45,
        rank=4, multiplier=2.0, ordinal=0,
    )


def read_d3_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    require(D3_OOF_PATH.is_file(), "Verified D3 OOF is missing")
    rows = engine.typed_rows(engine.read_csv(D3_OOF_PATH))
    abide5.validate_oof(rows, context, "B1")
    metrics = engine.metrics(rows)
    expected = {
        "correct": 769,
        "acc": 0.8900462962962963,
        "roc_auc": 0.8862642193323589,
        "pr_auc": 0.8505649758766733,
        "macro_f1": 0.8893405326700237,
        "bacc": 0.8894141823850182,
        "weighted_f1": 0.8900565247546481,
        "sen": 0.8816120906801007,
        "spe": 0.8972162740899358,
    }
    for key, value in expected.items():
        require(math.isclose(float(metrics[key]), float(value), rel_tol=0, abs_tol=1e-9), f"D3 metric drifted: {key}")
    require(metrics["confusion_matrix"] == [[350, 47], [48, 419]], "D3 confusion changed")
    require(metrics["predicted_counts"] == {"ADS": 398, "CN": 466}, "D3 prediction counts changed")
    return rows


def parameter_partition(model: torch.nn.Module) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    private = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
    common = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
    private_ids = {id(parameter) for _, parameter in private}
    common_ids = {id(parameter) for _, parameter in common}
    all_ids = {id(parameter) for _, parameter in named}
    require(len(private) == EXPECTED_PRIVATE_TENSORS, "D3 must have five rank-4 adapters / 20 parameter tensors")
    require(not (private_ids & common_ids), "Private/common parameter groups overlap")
    require(private_ids | common_ids == all_ids, "Private/common parameter groups are not exhaustive")
    return common, private


def make_objects(context: dict[str, Any], audit_common: bool = False):
    model, criterion, optimizer, scheduler, audit = abide5.make_training_objects(context, spec(), audit_common=audit_common)
    common, private = parameter_partition(model)
    require(engine.parameter_count(model) == EXPECTED_PARAMETERS, "D3 parameter count changed")
    require(len(model.private_adapters) == EXPECTED_MODALITIES, "D3 real modality/adapter count changed")
    require(len(optimizer.param_groups) == 2, "D3 must use one Adam with two parameter groups")
    require(math.isclose(float(optimizer.param_groups[0]["lr"]), 0.005, rel_tol=0, abs_tol=1e-14), "Base LR changed")
    require(math.isclose(float(optimizer.param_groups[1]["lr"]), 0.010, rel_tol=0, abs_tol=1e-14), "Adapter LR changed")
    require(all(math.isclose(float(group["weight_decay"]), 0.0005, rel_tol=0, abs_tol=1e-14) for group in optimizer.param_groups), "Weight decay changed")
    require(sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()) == 0, "Unexpected BatchNorm requires a buffer-restoration policy")
    return model, criterion, optimizer, scheduler, audit, common, private


def inspect_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    context = engine.build_context(torch.device("cpu"))
    protocol = abide5.protocol()
    require(protocol["dataset"] == "ABIDE-5" and protocol["task"] == "ADS_CN", "Task changed")
    require(protocol["sample_count"] == 864 and protocol["class_counts"] == {"ADS": 397, "CN": 467}, "Sample/class counts changed")
    require(protocol["positive_index"] == 0 and protocol["class_names"] == ["ADS", "CN"], "ASD positive direction changed")
    require(len(protocol["modalities"]) == EXPECTED_MODALITIES and sum(item["feature_count"] for item in protocol["modalities"]) == 470, "Five-modality feature manifest changed")
    d3_rows = read_d3_rows(context)
    d3_summary = engine.read_json(D3_SUMMARY_PATH)
    d3_config = engine.read_json(D3_CONFIG_PATH)
    require(d3_config["trial_id"] == "D3" and d3_config["rank"] == 4 and d3_config["adapter_lr_multiplier"] == 2.0, "D3 config changed")
    require(d3_summary["metrics"] == engine.metrics(d3_rows), "D3 summary/OOF disagreement")
    model, _, optimizer, _, audit, common, private = make_objects(context, audit_common=True)
    config_core = {
        "experiment": EXPERIMENT_ID,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "d3_source_commit": D3_SOURCE_COMMIT,
        "d3_result_commit": D3_RESULT_COMMIT,
        "base_selection_reason": "a23134c is the nearest verified result commit after 111f5dc and contains the exact registered D3 config, metrics, and subject-level OOF",
        "dataset": "ABIDE-5",
        "task": "ADS_CN",
        "subjects": 864,
        "class_names": ["ADS", "CN"],
        "positive_class": "ADS/ASD",
        "positive_index": 0,
        "modalities": EXPECTED_MODALITIES,
        "folds": list(FOLDS),
        "seed_per_fold": 0,
        "epochs": EPOCHS,
        "d3_spec": spec(),
        "epsilon": EPSILON,
        "gradient_definition": "global common-parameter projection of r=g_f-g_s against g_s, followed by global norm preservation; private gradients are grad(L_f)",
        "training_protocol": {
            "full_batch_transductive": True,
            "class_weights": "historical full-dataset weights",
            "loss": "criterion_lossv2 complete historical loss",
            "label_smoothing": 0.05,
            "orthogonality": 0.0,
            "optimizer": "Adam, one optimizer, base/private groups",
            "gradient_clip": 1.0,
            "scheduler": "historical ratio-preserving CustomCosineAnnealingLR(T_max=400, eta_min base/private=0.0001/0.0002)",
            "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
            "graph": False,
            "ema": False,
            "ensemble": False,
            "single_model": True,
        },
        "source_dependency_sha256": {path: engine.file_sha256(ROOT / path) for path in ("scripts/run_pg_pal_abide5_v1.py", *LOCKED_DEPENDENCIES)},
    }
    config = {**config_core, "sha256": engine.payload_sha256(config_core)}
    inspect_core = {
        "experiment": EXPERIMENT_ID,
        "dataset": "ABIDE-5",
        "task": "ADS_CN",
        "sample_count": 864,
        "class_counts": {"ADS": 397, "CN": 467},
        "positive_index": 0,
        "modality_count": EXPECTED_MODALITIES,
        "feature_count": 470,
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "subject_count": len(d3_rows),
        "subject_id_unique": len({row["subject_id"] for row in d3_rows}) == 864,
        "d3_config_sha256": engine.file_sha256(D3_CONFIG_PATH),
        "d3_summary_sha256": engine.file_sha256(D3_SUMMARY_PATH),
        "d3_oof_sha256": engine.file_sha256(D3_OOF_PATH),
        "d3_metrics": engine.metrics(d3_rows),
        "d3_parameter_count": int(d3_summary["parameter_count"]),
        "runtime_parameter_count": engine.parameter_count(model),
        "common_parameter_tensors": len(common),
        "private_parameter_tensors": len(private),
        "optimizer_parameter_groups": len(optimizer.param_groups),
        "common_none_gradient_whitelist": COMMON_NONE_WHITELIST,
        "batchnorm_modules": 0,
        "private_toggle": "temporary output-zero forward hooks on each private adapter; common trunk remains attached",
    }
    inspect = {**inspect_core, "sha256": engine.payload_sha256(inspect_core)}
    fold_core = {"task_id": context["task_id"], "fold_manifest": context["fold_manifest"]}
    folds = {**fold_core, "sha256": engine.payload_sha256(fold_core)}
    del model, optimizer, context
    return config, inspect, folds


def run_inspect() -> None:
    config, inspect, folds = inspect_payloads()
    engine.atomic_write_json(CONFIG_PATH, config)
    engine.atomic_write_json(INSPECT_PATH, inspect)
    engine.atomic_write_json(FOLD_MANIFEST_PATH, folds)
    print(json.dumps({"inspect": "PASS", "d3_correct": inspect["d3_metrics"]["correct"], "parameters": inspect["runtime_parameter_count"]}, indent=2))


def validate_inspect() -> None:
    require(CONFIG_PATH.is_file() and INSPECT_PATH.is_file() and FOLD_MANIFEST_PATH.is_file(), "Run inspect first")
    expected = inspect_payloads()
    require(engine.read_json(CONFIG_PATH) == expected[0], "Experiment config drifted")
    require(engine.read_json(INSPECT_PATH) == expected[1], "Inspect manifest drifted")
    require(engine.read_json(FOLD_MANIFEST_PATH) == expected[2], "Fold manifest drifted")


def source_gate() -> str:
    require(engine.git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(engine.git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree drifted")
    require(engine.git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index drifted")
    head = engine.current_commit()
    require(head != BASE_COMMIT and engine.git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Source commit lineage changed")
    changed = {line.strip().replace("\\", "/") for line in engine.git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines() if line.strip()}
    require(changed == set(REQUIRED_SOURCE), f"Source commit scope changed: {sorted(changed)}")
    for path in REQUIRED_SOURCE:
        require(engine.git("ls-files", "--error-unmatch", "--", path, check=False).returncode == 0, f"Untracked source: {path}")
    for path in LOCKED_DEPENDENCIES:
        require(engine.git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", path, check=False).returncode == 0, f"D3 dependency changed: {path}")
    validate_inspect()
    return head


def runtime_lock(source_commit: str) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "config_sha256": engine.file_sha256(CONFIG_PATH),
        "inspect_sha256": engine.file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": engine.file_sha256(FOLD_MANIFEST_PATH),
        "d3_oof_sha256": engine.file_sha256(D3_OOF_PATH),
        "runner_sha256": engine.file_sha256(ROOT / "scripts" / "run_pg_pal_abide5_v1.py"),
        "folds": list(FOLDS),
        "epochs": EPOCHS,
        "device": "cuda:0",
    }
    return {**core, "sha256": engine.payload_sha256(core)}


@contextlib.contextmanager
def private_outputs_disabled(model: torch.nn.Module) -> Iterator[dict[str, int]]:
    calls = {str(index): 0 for index in range(len(model.private_adapters))}
    handles = []
    for index, adapter in enumerate(model.private_adapters):
        key = str(index)
        def replace(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor, *, _key: str = key) -> torch.Tensor:
            calls[_key] += 1
            return torch.zeros_like(output)
        handles.append(adapter.register_forward_hook(replace))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def torch_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_torch_rng(value: dict[str, Any]) -> None:
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return torch.equal(left["torch_cpu"], right["torch_cpu"]) and len(left["torch_cuda"]) == len(right["torch_cuda"]) and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def pg_pal_update(
    model: torch.nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    features: torch.Tensor,
    labels_for_loss: torch.Tensor,
    train_mask: torch.Tensor,
    grad_clip: float,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    common, private = parameter_partition(model)
    before = torch_rng_state()
    with private_outputs_disabled(model) as hook_calls:
        shared_logits, shared_embeddings, shared_auxiliary = model(features)
    after_shared = torch_rng_state()
    require(all(value == 1 for value in hook_calls.values()), "Each private adapter must be disabled exactly once in shared-only forward")
    restore_torch_rng(before)
    full_logits, full_embeddings, full_auxiliary = model(features)
    after_full = torch_rng_state()
    require(rng_states_equal(after_shared, after_full), "Shared/full forwards did not consume identical stochastic draws")
    require(tuple(shared_logits.shape) == tuple(full_logits.shape) == (features.size(0), 2), "PG-PAL logits shape changed")
    shared_loss = criterion(shared_logits, labels_for_loss, train_mask, shared_embeddings, shared_auxiliary)
    full_loss = criterion(full_logits, labels_for_loss, train_mask, full_embeddings, full_auxiliary)
    require(finite_tensor(shared_loss) and finite_tensor(full_loss), "PG-PAL loss is non-finite")

    common_parameters = [parameter for _, parameter in common]
    private_parameters = [parameter for _, parameter in private]
    shared_grads = torch.autograd.grad(shared_loss, common_parameters, allow_unused=True)
    full_grads_all = torch.autograd.grad(full_loss, [*common_parameters, *private_parameters], allow_unused=True)
    full_common_grads = full_grads_all[:len(common_parameters)]
    private_grads = full_grads_all[len(common_parameters):]

    active_names: list[str] = []
    active_common: list[torch.nn.Parameter] = []
    gs: list[torch.Tensor] = []
    gf: list[torch.Tensor] = []
    none_names: list[str] = []
    for (name, parameter), left, right in zip(common, shared_grads, full_common_grads):
        if left is None or right is None:
            require(left is None and right is None and name in COMMON_NONE_WHITELIST, f"Unexpected missing common gradient: {name}")
            parameter.grad = None
            none_names.append(name)
            continue
        require(finite_tensor(left) and finite_tensor(right), f"Non-finite common gradient: {name}")
        active_names.append(name); active_common.append(parameter); gs.append(left); gf.append(right)
    require(set(none_names) == set(COMMON_NONE_WHITELIST), "Common None-gradient whitelist changed")
    require(active_common, "No active common gradients")

    for (name, parameter), gradient in zip(private, private_grads):
        require(gradient is not None and finite_tensor(gradient), f"Invalid private gradient: {name}")
        parameter.grad = gradient.detach().clone()

    residual = [right - left for left, right in zip(gs, gf)]
    dot = sum((item.double() * base.double()).sum() for item, base in zip(residual, gs))
    shared_norm_sq = sum((base.double() * base.double()).sum() for base in gs)
    residual_norm_sq = sum((item.double() * item.double()).sum() for item in residual)
    require(finite_tensor(dot) and finite_tensor(shared_norm_sq) and finite_tensor(residual_norm_sq) and float(shared_norm_sq) > 0.0, "Invalid global PG-PAL statistics")
    projection_applied = bool(float(dot) < 0.0)
    if projection_applied:
        coefficient = dot / (shared_norm_sq + EPSILON)
        projected_residual = [item - coefficient.to(dtype=item.dtype) * base for item, base in zip(residual, gs)]
    else:
        projected_residual = residual
    q = [base + item for base, item in zip(gs, projected_residual)]
    q_norm_sq = sum((item.double() * item.double()).sum() for item in q)
    require(finite_tensor(q_norm_sq) and float(q_norm_sq) > 0.0, "Projected common gradient has zero/non-finite norm")
    scale = torch.sqrt(shared_norm_sq) / (torch.sqrt(q_norm_sq) + EPSILON)
    projected = [item * scale.to(dtype=item.dtype) for item in q]
    projected_norm_sq = sum((item.double() * item.double()).sum() for item in projected)
    projected_dot = sum((item.double() * base.double()).sum() for item, base in zip(projected, gs))
    norm_ratio = float(torch.sqrt(projected_norm_sq) / (torch.sqrt(shared_norm_sq) + EPSILON))
    require(float(projected_dot) >= -1e-8, "Projected common gradient conflicts with shared baseline")
    require(abs(norm_ratio - 1.0) <= 1e-5, "Global norm preservation failed")
    for parameter, gradient in zip(active_common, projected):
        require(finite_tensor(gradient), "Projected gradient is non-finite")
        parameter.grad = gradient.detach().clone()

    private_max = {name: float(parameter.grad.detach().abs().max().cpu()) for name, parameter in private}
    require(all(math.isfinite(value) for value in private_max.values()), "Private gradient audit is non-finite")
    raw_cosine = float(dot / (torch.sqrt(residual_norm_sq) * torch.sqrt(shared_norm_sq) + EPSILON)) if float(residual_norm_sq) > 0 else 0.0
    q_to_shared_ratio = float(torch.sqrt(q_norm_sq) / (torch.sqrt(shared_norm_sq) + EPSILON))
    clip_value = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    require(math.isfinite(float(clip_value)), "Gradient clipping norm is non-finite")
    optimizer.step()
    scheduler.step()
    scheduler.assert_ratio()
    return {
        "shared_loss": float(shared_loss.detach().cpu()),
        "full_loss": float(full_loss.detach().cpu()),
        "projection_applied": projection_applied,
        "raw_marginal_shared_dot": float(dot.detach().cpu()),
        "raw_marginal_shared_cosine": raw_cosine,
        "shared_gradient_norm": float(torch.sqrt(shared_norm_sq).detach().cpu()),
        "pre_norm_preservation_q_to_shared_ratio": q_to_shared_ratio,
        "post_norm_preservation_ratio": norm_ratio,
        "projected_common_shared_dot": float(projected_dot.detach().cpu()),
        "private_gradient_max_by_tensor": private_max,
        "shared_full_rng_equal": True,
        "shared_hook_calls": hook_calls,
        "active_common_gradient_tensors": len(active_names),
        "whitelisted_none_common_tensors": sorted(none_names),
        "optimizer_steps": 1,
        "scheduler_steps": 1,
    }


def smoke_paths() -> tuple[Path, Path, Path]:
    return SMOKE_DIR / "smoke_config.json", SMOKE_DIR / "smoke_report.json", SMOKE_DIR / "checkpoint_roundtrip.pt"


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = source_gate(); runtime = runtime_lock(source)
    context = engine.trial_context(engine.build_context(torch.device(device_text)), spec())
    model, criterion, optimizer, scheduler, object_audit, _, private = make_objects(context, audit_common=True)
    features, labels = context["dataset_data"]["Feature"], context["dataset_data"]["Label"]
    train_mask, test_mask, _ = historical.fold_positions(context, 0)
    require(not bool(torch.any(train_mask & test_mask)), "Smoke train/test masks overlap")
    labels_for_loss = labels.clone(); labels_for_loss[~train_mask] = 0
    initial_private = {name: parameter.detach().cpu().clone() for name, parameter in private}
    config_path, report_path, checkpoint_path = smoke_paths()
    config_core = {"runtime_lock": runtime, "spec": spec(), "fold": 0, "epochs": SMOKE_EPOCHS}
    smoke_config = {**config_core, "sha256": engine.payload_sha256(config_core)}
    engine.atomic_write_json(config_path, smoke_config)
    audits = []
    cumulative_private = {name: 0.0 for name in initial_private}
    for epoch in range(1, SMOKE_EPOCHS + 1):
        audit = pg_pal_update(model, criterion, optimizer, scheduler, features, labels_for_loss, train_mask, 1.0)
        audit["epoch"] = epoch; audits.append(audit)
        for name, value in audit["private_gradient_max_by_tensor"].items():
            cumulative_private[name] = max(cumulative_private[name], value)
    require(all(value > 0.0 for value in cumulative_private.values()), "Every private tensor must receive nonzero gradient within smoke")
    private_delta = {name: float((parameter.detach().cpu() - initial_private[name]).abs().max()) for name, parameter in model.named_parameters() if name in initial_private}
    require(all(math.isfinite(value) and value > 0.0 for value in private_delta.values()), "Every private tensor must update within smoke")
    model.eval()
    with torch.no_grad():
        logits = model(features)[0]; probability = torch.softmax(logits, dim=-1)
    require(tuple(logits.shape) == (864, 2), "Smoke logits shape changed")
    simplex_error = float((probability.sum(1) - 1.0).abs().max().cpu())
    require(simplex_error <= 2e-6, "Smoke probability simplex failed")
    checkpoint = {
        "schema": 1, "runtime_lock": runtime, "smoke_config": smoke_config,
        "epoch": SMOKE_EPOCHS, "model": engine.clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": engine.capture_rng(), "optimizer_steps": SMOKE_EPOCHS, "scheduler_steps": SMOKE_EPOCHS,
    }
    engine.atomic_torch_save(checkpoint_path, checkpoint)
    restored, _, restored_optimizer, restored_scheduler, _, _, _ = make_objects(context)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(loaded["runtime_lock"] == runtime and loaded["smoke_config"] == smoke_config, "Smoke checkpoint ownership changed")
    restored.load_state_dict(loaded["model"], strict=True); restored_optimizer.load_state_dict(loaded["optimizer"]); restored_scheduler.load_state_dict(loaded["scheduler"]); restored_scheduler.assert_ratio()
    restored.eval()
    with torch.no_grad(): reload_diff = float((restored(features)[0] - logits).abs().max().cpu())
    require(reload_diff == 0.0, "Smoke strict reload changed logits")
    require(engine.parameter_count(restored) == EXPECTED_PARAMETERS, "Smoke parameter count changed")
    report_core = {
        "runtime_lock": runtime,
        "smoke_config_sha256": engine.file_sha256(config_path),
        "fold": 0, "epochs": SMOKE_EPOCHS,
        "epoch_audits": audits,
        "cumulative_private_gradient_max_by_tensor": cumulative_private,
        "private_parameter_delta_by_tensor": private_delta,
        "probability_sum_max_abs_error": simplex_error,
        "parameter_count": engine.parameter_count(model),
        "object_audit": object_audit,
        "batchnorm_module_count": 0,
        "private_disable_method": "adapter output-zero forward hooks; no trunk detach",
        "test_label_use": {"model_input": False, "loss": False, "gradient_projection": False, "best_epoch_metrics_only": True},
        "optimizer_steps": SMOKE_EPOCHS, "scheduler_steps": SMOKE_EPOCHS,
        "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        "checkpoint_reload_logits_max_abs_diff": reload_diff,
        "device": torch.cuda.get_device_name(torch.device(device_text)),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    report = {**report_core, "sha256": engine.payload_sha256(report_core)}
    engine.atomic_write_json(report_path, report)
    print(json.dumps({"smoke": "PASS", "params": report["parameter_count"], "projection_epochs": sum(row["projection_applied"] for row in audits)}, indent=2))


def validate_smoke(runtime: dict[str, Any]) -> dict[str, Any]:
    config_path, report_path, checkpoint_path = smoke_paths()
    require(config_path.is_file() and report_path.is_file() and checkpoint_path.is_file(), "Smoke artifacts missing")
    config, report = engine.read_json(config_path), engine.read_json(report_path)
    require(config["runtime_lock"] == runtime and report["runtime_lock"] == runtime, "Smoke runtime changed")
    require(config["sha256"] == engine.payload_sha256({key: value for key, value in config.items() if key != "sha256"}), "Smoke config digest changed")
    require(report["sha256"] == engine.payload_sha256({key: value for key, value in report.items() if key != "sha256"}), "Smoke report digest changed")
    require(report["smoke_config_sha256"] == engine.file_sha256(config_path) and report["checkpoint_sha256"] == engine.file_sha256(checkpoint_path), "Smoke artifact hash changed")
    require(report["optimizer_steps"] == report["scheduler_steps"] == SMOKE_EPOCHS and report["parameter_count"] == EXPECTED_PARAMETERS, "Smoke core gates failed")
    require(all(row["shared_full_rng_equal"] and row["projected_common_shared_dot"] >= -1e-8 and abs(row["post_norm_preservation_ratio"] - 1.0) <= 1e-5 for row in report["epoch_audits"]), "Smoke projection/RNG gates failed")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(loaded["runtime_lock"] == runtime and loaded["epoch"] == SMOKE_EPOCHS, "Smoke checkpoint changed")
    return report


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {"runtime_sha256": runtime["sha256"], "spec": spec(), "fold": int(fold), "fold_manifest_sha256": context["fold_manifest"]["sha256"]}
    return {**core, "sha256": engine.payload_sha256(core)}


def fold_paths(fold: int) -> dict[str, Path]:
    root = FORMAL_DIR / f"fold_{fold:02d}"
    compact = COMPACT_FOLDS_DIR
    return {
        "root": root,
        "resume": root / "resume.pt",
        "checkpoint": root / "checkpoint_best.pt",
        "oof": root / "oof_predictions.csv",
        "history": root / "epoch_metrics.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
        "compact_oof": compact / f"fold_{fold:02d}_oof_predictions.csv",
        "compact_summary": compact / f"fold_{fold:02d}.json",
        "compact_complete": compact / f"fold_{fold:02d}.complete.json",
    }


def capture_resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, lock: dict[str, Any], fold: int, epoch: int, best: dict[str, Any] | None, mechanism: list[dict[str, Any]], cumulative_private: dict[str, float], initial_private: dict[str, torch.Tensor], history: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    return {
        "schema": 1, "lock": lock, "fold": int(fold), "epoch": int(epoch),
        "optimizer_steps": int(epoch), "scheduler_steps": int(epoch),
        "model": engine.clone_cpu_state(model), "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": engine.capture_rng(),
        "best": copy.deepcopy(best), "mechanism": copy.deepcopy(mechanism),
        "cumulative_private": dict(cumulative_private),
        "initial_private": {name: tensor.detach().cpu().clone() for name, tensor in initial_private.items()},
        "history": copy.deepcopy(history), "elapsed_seconds": float(elapsed),
    }


def parse_rows(path: Path) -> list[dict[str, Any]]:
    return engine.typed_rows(engine.read_csv(path))


def selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return float(metrics["acc"]), float(metrics["roc_auc"]), float(metrics["macro_f1"]), -int(epoch)


def train_fold(base_context: dict[str, Any], runtime: dict[str, Any], fold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = engine.trial_context(base_context, spec()); paths = fold_paths(fold); lock = fold_lock(runtime, context, fold)
    if paths["complete"].is_file() and paths["compact_complete"].is_file():
        return load_completed_fold(base_context, runtime, fold)
    paths["root"].mkdir(parents=True, exist_ok=True)
    allowed = {path.name for key, path in paths.items() if key not in {"root", "compact_oof", "compact_summary", "compact_complete"}}
    allowed |= {name + ".tmp" for name in allowed}
    require(not [path for path in paths["root"].iterdir() if path.name not in allowed], "Unknown partial formal artifacts")
    model, criterion, optimizer, scheduler, object_audit, _, private = make_objects(context)
    features, labels = context["dataset_data"]["Feature"], context["dataset_data"]["Label"]
    train_mask, test_mask, _ = historical.fold_positions(context, fold)
    labels_for_loss = labels.clone(); labels_for_loss[~train_mask] = 0
    initial_private = {name: parameter.detach().cpu().clone() for name, parameter in private}
    cumulative_private = {name: 0.0 for name in initial_private}
    mechanism: list[dict[str, Any]] = []; history: list[dict[str, Any]] = []; best = None
    start_epoch, resumed_from, elapsed_before = 1, 0, 0.0
    if paths["resume"].is_file():
        payload = torch.load(paths["resume"], map_location="cpu", weights_only=False)
        require(payload.get("schema") == 1 and payload.get("lock") == lock and payload.get("fold") == fold, "Resume ownership changed")
        resumed_from = int(payload["epoch"]); require(0 < resumed_from <= EPOCHS, "Resume epoch invalid")
        require(payload["optimizer_steps"] == payload["scheduler_steps"] == resumed_from, "Resume step count changed")
        model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"]); scheduler.assert_ratio()
        require(int(scheduler.last_epoch) == resumed_from, "Resume scheduler epoch changed")
        engine.restore_rng(payload["rng"])
        best = payload["best"]; mechanism = copy.deepcopy(payload["mechanism"]); cumulative_private = dict(payload["cumulative_private"])
        initial_private = {name: tensor.detach().cpu().clone() for name, tensor in payload["initial_private"].items()}
        history = copy.deepcopy(payload["history"]); elapsed_before = float(payload["elapsed_seconds"]); start_epoch = resumed_from + 1
        require([int(row["epoch"]) for row in history] == list(range(1, start_epoch)) and len(mechanism) == resumed_from, "Resume history changed")
    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        audit = pg_pal_update(model, criterion, optimizer, scheduler, features, labels_for_loss, train_mask, 1.0)
        mechanism.append({"epoch": epoch, **{key: value for key, value in audit.items() if key not in {"private_gradient_max_by_tensor", "shared_hook_calls"}}})
        for name, value in audit["private_gradient_max_by_tensor"].items(): cumulative_private[name] = max(cumulative_private[name], value)
        with torch.no_grad():
            full_logits = historical.infer(model, features)[0]
            test_logits_t = full_logits[test_mask].detach().cpu(); test_probability_t = torch.softmax(test_logits_t, dim=-1)
        truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        logits = test_logits_t.numpy().astype(np.float64); probability = test_probability_t.numpy().astype(np.float64)
        metrics = historical.binary_metrics(truth, probability, context["protocol"])
        row = {"epoch": epoch, "shared_loss": audit["shared_loss"], "full_loss": audit["full_loss"], "projection_applied": audit["projection_applied"], "raw_marginal_shared_cosine": audit["raw_marginal_shared_cosine"], "pre_norm_ratio": audit["pre_norm_preservation_q_to_shared_ratio"], "post_norm_ratio": audit["post_norm_preservation_ratio"], "projected_dot": audit["projected_common_shared_dot"], "correct": metrics["correct"], "acc": metrics["acc"], "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"], "macro_f1": metrics["macro_f1"], "bacc": metrics["bacc"]}
        history.append(row)
        key = selection_key(metrics, epoch)
        if best is None or key > tuple(best["selection_key"]):
            best = {"epoch": epoch, "selection_key": list(key), "metrics": metrics, "model": engine.clone_cpu_state(model), "test_logits": logits, "test_probabilities": probability, "test_truth": truth}
        if epoch % 20 == 0 or epoch == EPOCHS:
            elapsed = elapsed_before + time.perf_counter() - started
            engine.atomic_torch_save(paths["resume"], capture_resume(model, optimizer, scheduler, lock, fold, epoch, best, mechanism, cumulative_private, initial_private, history, elapsed))
    require(best is not None and len(history) == len(mechanism) == EPOCHS and int(scheduler.last_epoch) == EPOCHS, "Formal fold is incomplete")
    model.load_state_dict(best["model"], strict=True)
    torch.cuda.synchronize(context["device"]); inference_start = time.perf_counter()
    with torch.no_grad(): replay = historical.infer(model, features)[0]
    torch.cuda.synchronize(context["device"]); inference_seconds = time.perf_counter() - inference_start
    replay_test = replay[test_mask].detach().cpu().numpy().astype(np.float64)
    require(float(np.max(np.abs(replay_test - best["test_logits"]))) <= 1e-6, "Best checkpoint replay changed logits")
    rows = abide5.prediction_rows(context, spec(), fold, best["test_logits"], best["test_probabilities"], best["test_truth"])
    require(engine.metrics_close(engine.metrics(rows), best["metrics"]), "Fold OOF metrics changed")
    diagnostics = historical.private_diagnostics(context, model, test_mask, cumulative_private, initial_private)
    require(diagnostics["private_trained"] and not diagnostics["private_collapse"], "Private adapter did not train")
    elapsed_total = elapsed_before + time.perf_counter() - started
    checkpoint = {"schema": 1, "lock": lock, "fold": fold, "best_epoch": best["epoch"], "best_metrics": best["metrics"], "best_model": best["model"], "optimizer": copy.deepcopy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()), "final_epoch": EPOCHS, "optimizer_steps": EPOCHS, "scheduler_steps": EPOCHS, "elapsed_seconds": elapsed_total}
    engine.atomic_torch_save(paths["checkpoint"], checkpoint); engine.atomic_write_csv(paths["oof"], rows); engine.atomic_write_csv(paths["history"], history)
    mechanism_summary = summarize_fold_mechanism(mechanism, diagnostics)
    summary = {"schema": 1, "lock": lock, "fold": fold, "best_epoch": best["epoch"], "best_metrics": best["metrics"], "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch", "selection_uses_test_truth_after_probability_output": True, "object_audit": object_audit, "mechanism": mechanism_summary, "private_diagnostics": diagnostics, "optimizer_steps": EPOCHS, "scheduler_steps": EPOCHS, "resumed_from_epoch": resumed_from, "elapsed_seconds": elapsed_total, "inference_seconds_full_graph": inference_seconds, "oof_count": len(rows), "checkpoint_sha256": engine.file_sha256(paths["checkpoint"]), "oof_sha256": engine.file_sha256(paths["oof"]), "history_sha256": engine.file_sha256(paths["history"])}
    engine.atomic_write_json(paths["summary"], summary)
    complete_core = {"complete": True, "lock": lock, "checkpoint_sha256": engine.file_sha256(paths["checkpoint"]), "oof_sha256": engine.file_sha256(paths["oof"]), "history_sha256": engine.file_sha256(paths["history"]), "summary_sha256": engine.file_sha256(paths["summary"])}
    complete = {**complete_core, "sha256": engine.payload_sha256(complete_core)}; engine.atomic_write_json(paths["complete"], complete)
    engine.atomic_write_csv(paths["compact_oof"], rows); engine.atomic_write_json(paths["compact_summary"], summary); engine.atomic_write_json(paths["compact_complete"], complete)
    print(f"[fold {fold}] best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    return load_completed_fold(base_context, runtime, fold)


def summarize_fold_mechanism(rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    require(len(rows) == EPOCHS, "Mechanism history incomplete")
    return {
        "projection_applied_epochs": int(sum(bool(row["projection_applied"]) for row in rows)),
        "projection_applied_fraction": float(statistics.mean(bool(row["projection_applied"]) for row in rows)),
        "raw_private_marginal_shared_cosine_mean": float(statistics.mean(float(row["raw_marginal_shared_cosine"]) for row in rows)),
        "pre_norm_preservation_gradient_norm_ratio_mean": float(statistics.mean(float(row["pre_norm_preservation_q_to_shared_ratio"]) for row in rows)),
        "post_norm_preservation_gradient_norm_ratio_mean": float(statistics.mean(float(row["post_norm_preservation_ratio"]) for row in rows)),
        "post_projection_common_shared_dot_min": float(min(float(row["projected_common_shared_dot"]) for row in rows)),
        "private_adapter_max_gradient": float(max(diagnostics["adapter_cumulative_max_gradient"].values())),
        "private_trained": bool(diagnostics["private_trained"]),
        "private_collapse": bool(diagnostics["private_collapse"]),
    }


def load_completed_fold(base_context: dict[str, Any], runtime: dict[str, Any], fold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = engine.trial_context(base_context, spec()); paths = fold_paths(fold); lock = fold_lock(runtime, context, fold)
    for key in ("resume", "checkpoint", "oof", "history", "summary", "complete", "compact_oof", "compact_summary", "compact_complete"):
        require(paths[key].is_file(), f"Completed fold artifact missing: {fold}/{key}")
    summary, complete = engine.read_json(paths["summary"]), engine.read_json(paths["complete"])
    require(summary == engine.read_json(paths["compact_summary"]) and complete == engine.read_json(paths["compact_complete"]), "Compact fold artifacts drifted")
    require(complete["sha256"] == engine.payload_sha256({key: value for key, value in complete.items() if key != "sha256"}) and complete["lock"] == lock, "Fold completion lock changed")
    for field, key in (("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("history_sha256", "history"), ("summary_sha256", "summary")):
        require(complete[field] == engine.file_sha256(paths[key]), f"Fold {key} hash changed")
    rows = parse_rows(paths["oof"]); compact_rows = parse_rows(paths["compact_oof"])
    require(rows == compact_rows and len(rows) == summary["oof_count"], "Fold OOF compact copy changed")
    history = engine.read_csv(paths["history"]); require([int(row["epoch"]) for row in history] == list(range(1, EPOCHS + 1)), "Fold history changed")
    selected = max(history, key=lambda row: (float(row["acc"]), float(row["roc_auc"]), float(row["macro_f1"]), -int(row["epoch"])))
    require(int(selected["epoch"]) == int(summary["best_epoch"]), "Fold best epoch changed")
    require(engine.metrics_close(engine.metrics(rows), summary["best_metrics"]), "Fold OOF metrics changed")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint["lock"] == lock and checkpoint["best_epoch"] == summary["best_epoch"] and checkpoint["optimizer_steps"] == checkpoint["scheduler_steps"] == EPOCHS, "Fold checkpoint changed")
    model, _, _, _, _, _, _ = make_objects(context); model.load_state_dict(checkpoint["best_model"], strict=True)
    _, test_mask, _ = historical.fold_positions(context, fold)
    with torch.no_grad(): probability = torch.softmax(historical.infer(model, context["dataset_data"]["Feature"])[0][test_mask], dim=-1).detach().cpu().numpy()
    saved = np.asarray([[row["probability_0"], row["probability_1"]] for row in rows], dtype=float)
    require(float(np.max(np.abs(probability - saved))) <= 1e-6 and np.array_equal(probability.argmax(1), np.asarray([row["prediction"] for row in rows])), "Fold checkpoint does not reproduce OOF")
    require(summary["mechanism"]["post_projection_common_shared_dot_min"] >= -1e-8 and abs(summary["mechanism"]["post_norm_preservation_gradient_norm_ratio_mean"] - 1.0) <= 1e-5, "Fold mechanism gate changed")
    del model; torch.cuda.empty_cache()
    return summary, rows


def exact_mcnemar(repairs: int, damages: int) -> float:
    n = repairs + damages
    if n == 0: return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(repairs, damages) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_comparison(d3_rows: list[dict[str, Any]], pg_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["subject_id"]: row for row in d3_rows}; right = {row["subject_id"]: row for row in pg_rows}
    require(set(left) == set(right) and len(left) == 864, "D3/PG-PAL subject alignment changed")
    repairs = damages = changed = 0
    for subject_id in sorted(left):
        a, b = left[subject_id], right[subject_id]
        require(a["truth"] == b["truth"] and a["fold"] == b["fold"] and a["feature_sha256"] == b["feature_sha256"], "Paired truth/fold anchor changed")
        a_ok, b_ok = a["prediction"] == a["truth"], b["prediction"] == b["truth"]
        repairs += int((not a_ok) and b_ok); damages += int(a_ok and (not b_ok)); changed += int(a["prediction"] != b["prediction"])
    d3_metrics, pg_metrics = engine.metrics(d3_rows), engine.metrics(pg_rows)
    return {
        "repairs": repairs, "damages": damages, "changed": changed,
        "correct_delta": pg_metrics["correct"] - d3_metrics["correct"],
        "metric_deltas": {key: float(pg_metrics[key]) - float(d3_metrics[key]) for key in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "sen", "spe")},
        "exact_two_sided_mcnemar_p": exact_mcnemar(repairs, damages),
    }


def aggregate_mechanism(folds: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["mechanism"] for row in folds]
    require(len(values) == 10 and all(row["private_trained"] and not row["private_collapse"] for row in values), "Private training mechanism failed")
    return {
        "projection_applied_fraction_mean": float(statistics.mean(row["projection_applied_fraction"] for row in values)),
        "projection_applied_fraction_by_fold": [row["projection_applied_fraction"] for row in values],
        "raw_private_marginal_shared_cosine_mean": float(statistics.mean(row["raw_private_marginal_shared_cosine_mean"] for row in values)),
        "pre_norm_preservation_gradient_norm_ratio_mean": float(statistics.mean(row["pre_norm_preservation_gradient_norm_ratio_mean"] for row in values)),
        "post_norm_preservation_gradient_norm_ratio_mean": float(statistics.mean(row["post_norm_preservation_gradient_norm_ratio_mean"] for row in values)),
        "post_projection_common_shared_dot_min": float(min(row["post_projection_common_shared_dot_min"] for row in values)),
        "private_adapter_max_gradient": float(max(row["private_adapter_max_gradient"] for row in values)),
        "private_trained_all_folds": True,
        "private_collapse_any_fold": False,
        "parameter_count": EXPECTED_PARAMETERS,
    }


def decision(metrics: dict[str, Any], comparison: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    collapse = min(metrics["predicted_counts"].values()) < math.ceil(0.10 * metrics["n"])
    hard_fail = {
        "correct_le_769": metrics["correct"] <= 769,
        "repairs_le_damages": comparison["repairs"] <= comparison["damages"],
        "private_not_trained": not mechanism["private_trained_all_folds"],
        "parameter_count_changed": mechanism["parameter_count"] != EXPECTED_PARAMETERS,
        "class_prediction_collapse": collapse,
    }
    safety = {
        "correct_ge_770": metrics["correct"] >= 770,
        "repairs_gt_damages": comparison["repairs"] > comparison["damages"],
        "bacc_ge_min": metrics["bacc"] >= 0.8864142,
        "macro_f1_ge_min": metrics["macro_f1"] >= 0.8863405,
        "roc_auc_ge_min": metrics["roc_auc"] >= 0.8812642,
        "pr_auc_ge_min": metrics["pr_auc"] >= 0.8455650,
        "asd_sen_ge_min": metrics["sen"] >= 0.8716121,
        "cn_spe_ge_min": metrics["spe"] >= 0.8872163,
        "no_collapse": not collapse,
        "private_trained": mechanism["private_trained_all_folds"],
        "parameter_count_unchanged": mechanism["parameter_count"] == EXPECTED_PARAMETERS,
        "single_model": True,
    }
    if any(hard_fail.values()):
        token = "PG_PAL_ABIDE5_NO_GAIN"; allow = False; next_action = "DGL-Lite: encoder-fusion disentangled gradient learning (record only; not run)"
    elif all(safety.values()):
        token = "PG_PAL_ABIDE5_GO"; allow = True; next_action = "PG-PAL Cross-Task Validation v1 (record only; not run)"
    else:
        token = "PG_PAL_ABIDE5_UNSAFE"; allow = False; next_action = "stop PG-PAL; do not run v1.1"
    return {"decision": token, "hard_fail_checks": hard_fail, "safety_checks": safety, "allow_cross_task_validation": allow, "next_recommendation": next_action, "v1_1_ran": False}


def render_report(summary: dict[str, Any]) -> str:
    m, d3, pair, mech, dec = summary["metrics"], summary["d3_reference"], summary["paired_vs_d3"], summary["mechanism"], summary["decision"]
    lines = [
        "# PG-PAL-ABIDE5 v1", "",
        "## Outcome", "",
        f"- Decision: **{dec['decision']}**",
        f"- Source commit: `{summary['runtime_lock']['source_commit']}`",
        "- Result commit: reported in final Git handoff (a commit cannot contain its own SHA)",
        f"- Branch: `{BRANCH}`", f"- Device: `{summary['device']}`", "",
        "PG-PAL did not add inference parameters, an inference branch, or a second model. It is not an ensemble. Only ABIDE-5 was run; no second version or v1.1 was run.", "",
        "## Purpose and formula", "",
        "For the fixed D3 model, shared-only and full-private losses use identical CPU/CUDA stochastic states. On all common parameters, `r = g_f - g_s`; when `<r,g_s> < 0`, PG-PAL replaces `r` by `r - <r,g_s>/(||g_s||^2+1e-12) g_s`. It then forms `q=g_s+r_tilde` and globally rescales q to `||g_s||`. Private parameters receive only `grad(L_f)`. One Adam step and one scheduler step follow historical global gradient clipping.", "",
        "PG-PAL differs from ordinary PCGrad because it protects a shared-only baseline against the private marginal rather than symmetrically projecting task gradients; it differs from DGL because it adds no encoder/fusion gradient-learning modules; it differs from SP-LRIF because inference is the unchanged D3 private-enabled path and no interaction fusion is added.", "",
        "This only removes a first-order conflict component before Adam. It does not guarantee a lossless Adam parameter step, test improvement, balanced modality use, total multimodal conflict removal, or strict shared/private disentanglement.", "",
        "## Locked protocol", "",
        "ABIDE-5 ADS_CN; 864 subjects (397 ADS/ASD positive class index 0, 467 CN class index 1); five real modalities; folds 0..9; seed 0; 400 full-batch transductive epochs; historical global class weights; complete criterion_lossv2 with label smoothing 0.05 and orthogonality 0; Adam; grad clip 1; historical ratio-preserving cosine scheduler; ACC > ROC-AUC > Macro-F1 > earliest checkpoint selection; graph/EMA/ensemble off; single model. Test labels enter only post-probability metric/checkpoint selection, not model inputs, loss, or gradient projection.", "",
        "## D3 reference and PG-PAL results", "",
        "| Model | Correct | ACC | pooled AUC | fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Confusion [ADS,CN] | Pred ADS/CN | Params | Train sec | Infer sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        f"| D3 | {d3['correct']}/864 | {d3['acc']:.7f} | {d3['roc_auc']:.7f} | {summary['d3_fold_roc_auc_mean']:.7f} +/- {summary['d3_fold_roc_auc_sample_std']:.7f} | {d3['pr_auc']:.7f} | {d3['macro_f1']:.7f} | {d3['bacc']:.7f} | {d3['weighted_f1']:.7f} | {d3['sen']:.7f} | {d3['spe']:.7f} | {d3['confusion_matrix']} | {d3['predicted_counts']['ADS']}/{d3['predicted_counts']['CN']} | {EXPECTED_PARAMETERS} | historical | historical |",
        f"| PG-PAL | {m['correct']}/864 | {m['acc']:.7f} | {m['roc_auc']:.7f} | {summary['fold_roc_auc_mean']:.7f} +/- {summary['fold_roc_auc_sample_std']:.7f} | {m['pr_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {m['weighted_f1']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {m['confusion_matrix']} | {m['predicted_counts']['ADS']}/{m['predicted_counts']['CN']} | {summary['parameter_count']} | {summary['training_time_seconds']:.1f} | {summary['inference_time_seconds']:.5f} |", "",
        f"PG-PAL 10-fold ACC mean +/- sample SD: {summary['fold_acc_mean']:.7f} +/- {summary['fold_acc_sample_std']:.7f}.", "",
        "## Subject-aligned comparison with D3", "",
        f"- repairs/damages/changed: {pair['repairs']}/{pair['damages']}/{pair['changed']}",
        f"- Correct delta: {pair['correct_delta']:+d}",
        f"- ACC/AUC/PR-AUC/Macro-F1/BACC/SEN/SPE deltas: {pair['metric_deltas']['acc']:+.7f} / {pair['metric_deltas']['roc_auc']:+.7f} / {pair['metric_deltas']['pr_auc']:+.7f} / {pair['metric_deltas']['macro_f1']:+.7f} / {pair['metric_deltas']['bacc']:+.7f} / {pair['metric_deltas']['sen']:+.7f} / {pair['metric_deltas']['spe']:+.7f}",
        f"- exact two-sided McNemar p: {pair['exact_two_sided_mcnemar_p']:.9f}", "",
        "## Per-fold results", "",
        "| Fold | Best epoch | Correct | ACC | AUC | Projection fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fold_metrics"]:
        lines.append(f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} | {row['roc_auc']:.7f} | {row['projection_fraction']:.5f} |")
    lines += ["", "## Minimal gradient mechanism", "", f"- Projection applied epoch fraction, mean across folds: {mech['projection_applied_fraction_mean']:.7f}", f"- Per-fold projection fractions: {mech['projection_applied_fraction_by_fold']}", f"- Raw private-marginal/shared gradient cosine mean: {mech['raw_private_marginal_shared_cosine_mean']:.7f}", f"- Pre/post norm-preservation ratios: {mech['pre_norm_preservation_gradient_norm_ratio_mean']:.7f} / {mech['post_norm_preservation_gradient_norm_ratio_mean']:.7f}", f"- Minimum projected common/shared dot product: {mech['post_projection_common_shared_dot_min']:.9g}", f"- Maximum private adapter gradient: {mech['private_adapter_max_gradient']:.9g}", f"- Private trained all folds / collapse: {mech['private_trained_all_folds']} / {mech['private_collapse_any_fold']}", "", "## External descriptive targets", "", f"- Correct >=787: `{m['correct'] >= 787}`", f"- Fold mean ACC >91.05%: `{summary['fold_acc_mean'] > 0.9105}`", f"- Fold mean ROC-AUC >90.99%: `{summary['fold_roc_auc_mean'] > 0.9099}`", "", "These are descriptive cross-paper numerical targets only, not same-protocol SOTA claims and not part of the PG-PAL decision.", "", "## Required answers", "", f"1. Exceeded D3 769/864: `{m['correct'] > 769}`.", f"2. Repairs > damages: `{pair['repairs'] > pair['damages']}`.", f"3. AUC/PR-AUC/Macro-F1/BACC all safe: `{all(dec['safety_checks'][key] for key in ('roc_auc_ge_min','pr_auc_ge_min','macro_f1_ge_min','bacc_ge_min'))}`.", f"4. ASD SEN and CN SPE both safe: `{dec['safety_checks']['asd_sen_ge_min'] and dec['safety_checks']['cn_spe_ge_min']}`.", f"5. Private adapters trained in all folds: `{mech['private_trained_all_folds']}`.", f"6. Conflict/projection epoch fraction: `{mech['projection_applied_fraction_mean']:.7f}`.", f"7. First-order non-conflict condition held: `{mech['post_projection_common_shared_dot_min'] >= -1e-8}`.", "8. Added inference parameters/path: `False`; the normal single D3 model is used.", f"9. Reached 787/864: `{m['correct'] >= 787}`.", f"10. Cross-task validation allowed: `{dec['allow_cross_task_validation']}`.", f"11. Final Decision: `{dec['decision']}`.", f"12. Strictly stopped without v1.1 if failed: `{(dec['decision'] == 'PG_PAL_ABIDE5_GO') or (not dec['v1_1_ran'])}`.", "", "## Reproduction", "", f"Run from a clean local branch named `{BRANCH}` at source commit `{summary['runtime_lock']['source_commit']}` using:", "", "```text", f'"{summary["python_executable"]}" -u -B scripts/run_pg_pal_abide5_v1.py inspect', f'"{summary["python_executable"]}" -u -B scripts/run_pg_pal_abide5_v1.py smoke --device cuda:0', f'"{summary["python_executable"]}" -u -B scripts/run_pg_pal_abide5_v1.py formal --device cuda:0', "```", ""]
    return "\n".join(lines)


def run_formal(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    source = source_gate(); runtime = runtime_lock(source); smoke = validate_smoke(runtime)
    base = engine.build_context(torch.device(device_text)); d3_rows = read_d3_rows(base)
    fold_summaries: list[dict[str, Any]] = []; rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for fold in FOLDS:
        summary, fold_rows = train_fold(base, runtime, fold); fold_summaries.append(summary); rows.extend(fold_rows)
    rows = sorted(rows, key=lambda row: int(row["original_csv_index"])); abide5.validate_oof(rows, base, "B1")
    require(len(rows) == 864 and len({row["subject_id"] for row in rows}) == 864, "Formal OOF coverage changed")
    metrics = engine.metrics(rows); comparison = paired_comparison(d3_rows, rows); mechanism = aggregate_mechanism(fold_summaries); outcome = decision(metrics, comparison, mechanism)
    fold_metrics = [{"fold": summary["fold"], "best_epoch": summary["best_epoch"], "correct": summary["best_metrics"]["correct"], "acc": summary["best_metrics"]["acc"], "roc_auc": summary["best_metrics"]["roc_auc"], "pr_auc": summary["best_metrics"]["pr_auc"], "macro_f1": summary["best_metrics"]["macro_f1"], "bacc": summary["best_metrics"]["bacc"], "projection_fraction": summary["mechanism"]["projection_applied_fraction"], "elapsed_seconds": summary["elapsed_seconds"], "inference_seconds": summary["inference_seconds_full_graph"], "resumed_from_epoch": summary["resumed_from_epoch"]} for summary in fold_summaries]
    d3_summary = engine.read_json(D3_SUMMARY_PATH); python_executable = sys.executable
    summary_core = {
        "experiment": EXPERIMENT_ID, "runtime_lock": runtime, "smoke_report_sha256": engine.file_sha256(SMOKE_DIR / "smoke_report.json"),
        "d3_reference": engine.metrics(d3_rows), "d3_source_commit": D3_SOURCE_COMMIT, "d3_result_commit": D3_RESULT_COMMIT,
        "d3_fold_roc_auc_mean": d3_summary["fold_roc_auc_mean"], "d3_fold_roc_auc_sample_std": d3_summary["fold_roc_auc_sample_std"],
        "metrics": metrics, "fold_metrics": fold_metrics,
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)), "fold_acc_sample_std": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_roc_auc_mean": float(statistics.mean(row["roc_auc"] for row in fold_metrics)), "fold_roc_auc_sample_std": float(statistics.stdev(row["roc_auc"] for row in fold_metrics)),
        "paired_vs_d3": comparison, "mechanism": mechanism, "decision": outcome,
        "parameter_count": EXPECTED_PARAMETERS, "single_model": True, "ensemble": False,
        "training_time_seconds": float(sum(row["elapsed_seconds"] for row in fold_metrics)), "inference_time_seconds": float(sum(row["inference_seconds"] for row in fold_metrics)), "formal_wall_seconds": float(time.perf_counter() - wall_start),
        "device": torch.cuda.get_device_name(torch.device(device_text)), "python_executable": python_executable, "python_version": sys.version, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "test_label_use": smoke["test_label_use"], "v1_1_ran": False,
    }
    summary = {**summary_core, "sha256": engine.payload_sha256(summary_core)}
    engine.atomic_write_csv(RESULT_DIR / "oof_predictions.csv", rows); engine.atomic_write_json(RESULT_DIR / "fold_metrics.json", fold_metrics); engine.atomic_write_json(RESULT_DIR / "mechanism_statistics.json", mechanism); engine.atomic_write_json(RESULT_DIR / "comparison_vs_d3.json", comparison); engine.atomic_write_json(RESULT_DIR / "formal_config.json", {"runtime_lock": runtime, "smoke_report_sha256": summary["smoke_report_sha256"], "spec": spec(), "sha256": engine.payload_sha256({"runtime_lock": runtime, "smoke_report_sha256": summary["smoke_report_sha256"], "spec": spec()})}); engine.atomic_write_json(RESULT_DIR / "summary.json", summary); engine.atomic_write_text(RESULT_DIR / "REPORT.md", render_report(summary))
    print(json.dumps({"formal": "COMPLETE", "decision": outcome["decision"], "metrics": metrics, "repairs": comparison["repairs"], "damages": comparison["damages"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="mode", required=True); sub.add_parser("inspect")
    for mode in ("smoke", "formal"):
        child = sub.add_parser(mode); child.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inspect": run_inspect()
    elif args.mode == "smoke": run_smoke(args.device)
    elif args.mode == "formal": run_formal(args.device)
    else: raise RuntimeError(args.mode)


if __name__ == "__main__":
    main()
