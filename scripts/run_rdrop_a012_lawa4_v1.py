"""Run locked A012 with R-Drop and same-trajectory LAWA-4 v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_c1_broad_hparam_search_v1 as broad
import run_cme_dual_branch_v1 as cme
from Utils import SET_Random


EXPERIMENT = "rdrop_a012_lawa4_v1"
RUNNER_REL = Path("scripts/run_rdrop_a012_lawa4_v1.py")
OUTPUT_REL = Path("experiments/rdrop_a012_lawa4_v1")
BASE_COMMIT = "13a25931fd8e5a111a7ae1be056f7b1b213c3055"
BRANCH_PREFIX = "experiment/rdrop-a012-lawa4-v1"
CLASS_NAMES = ("AD", "CN", "SMCI")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
RDROP_LAMBDA_V1 = 0.3
RDROP_LAMBDA_V1_1 = 0.15
LAWA_WINDOW = 4
BACC_FLOOR = 0.92663
EXPECTED_PARAMETERS = 862_971
A012_SPEC = {
    "trial_id": "A012",
    "stage": "A",
    "rank": 8,
    "base_lr": 0.011271416075886307,
    "base_weight_decay": 0.0010917677921787822,
    "lambda_aux": 0.5,
    "adapter_lr_multiplier": 2.0,
    "dropout_multiplier": 1.1,
    "generation_index": 12,
}
A012 = {
    "correct": 562,
    "acc": 0.939799331103679,
    "macro_f1": 0.9294205448780151,
    "bacc": 0.9286299264715336,
    "macro_auc": 0.961524845528226,
    "weighted_f1": 0.9397414920986079,
    "confusion_matrix": [[64, 0, 8], [0, 200, 9], [7, 12, 298]],
    "predicted_class_counts": {"AD": 71, "CN": 212, "SMCI": 315},
    "boundary_errors": {"AD_SMCI": 15, "CN_SMCI": 21, "AD_CN": 0},
}


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def tensor_tree_digest(payload) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def state_max_abs_diff(module: torch.nn.Module, reference: dict[str, torch.Tensor]) -> float:
    actual = module.state_dict()
    require(list(actual) == list(reference), "State keys changed")
    maximum = 0.0
    for name, value in actual.items():
        expected = reference[name].to(device=value.device, dtype=value.dtype)
        require(tuple(value.shape) == tuple(expected.shape), f"State shape changed: {name}")
        if value.is_floating_point() or value.is_complex():
            maximum = max(maximum, float((value - expected).abs().max().detach().cpu()))
        else:
            require(bool(torch.equal(value, expected)), f"Non-floating state changed: {name}")
    return maximum


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def average_raw_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    require(len(states) == LAWA_WINDOW, "LAWA-4 requires exactly four raw states")
    keys = list(states[-1])
    require(all(list(state) == keys for state in states), "LAWA state keys changed")
    averaged = {}
    for name in keys:
        tensors = [state[name] for state in states]
        require(all(tuple(value.shape) == tuple(tensors[-1].shape) for value in tensors), f"LAWA shape changed: {name}")
        if tensors[-1].is_floating_point():
            averaged[name] = torch.stack([value.float() for value in tensors], dim=0).mean(dim=0)
        elif tensors[-1].is_complex():
            averaged[name] = torch.stack([value.to(torch.complex64) for value in tensors], dim=0).mean(dim=0)
        else:
            averaged[name] = tensors[-1].clone()
    return averaged


def experiment_config() -> dict:
    return {
        "experiment": EXPERIMENT,
        "base_commit": BASE_COMMIT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "class_order": list(CLASS_NAMES),
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "graph": False,
        "ema": False,
        "model": "locked A012",
        "a012_spec": A012_SPEC,
        "label_smoothing": 0.05,
        "rdrop": {
            "v1_lambda": RDROP_LAMBDA_V1,
            "conditional_v1_1_lambda": RDROP_LAMBDA_V1_1,
            "two_forwards_one_backward_one_optimizer_step_one_scheduler_step": True,
            "sym_kl": "train-mask final raw three-class logits only; bidirectional; no detach",
            "shared_non_dropout_noise": True,
        },
        "lawa": {
            "window": LAWA_WINDOW,
            "queue": "raw post-update states only",
            "first_evaluation_epoch": 4,
            "floating_average": "FP32 equal average",
            "non_floating": "latest raw value",
        },
        "optimizer": "Adam with locked A012 base/private-adapter groups",
        "scheduler": "RatioPreservingCustomCosineAnnealingLR(T_max=400)",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "parameter_count": EXPECTED_PARAMETERS,
    }


def load_context(device_text: str) -> dict:
    context = broad.load_context(device_text)
    require(context["config"].use_ema is False, "EMA must remain disabled")
    require(abs(float(context["config"].input_noise_std) - 0.05) <= 1e-12, "A012 input noise changed")
    require(abs(float(context["config"].drop_path) - 0.05) <= 1e-12, "A012 drop-path changed")
    require(abs(float(context["config"].grad_clip) - 1.0) <= 1e-12, "A012 grad clip changed")
    require(abs(float(context["config"].logit_adjust_tau) - 0.75) <= 1e-12, "A012 logit adjustment changed")
    return context


def make_training_objects(context: dict):
    model, criterion, optimizer, scheduler, dropout, groups = broad.make_training_objects(context, A012_SPEC)
    require(parameter_count(model) == EXPECTED_PARAMETERS, "A012 parameter count changed")
    require(len(dropout) == 10 and all(abs(row["final_p"] - min(0.8, row["original_p"] * 1.1)) <= 1e-12 for row in dropout), "A012 dropout profile changed")
    require(len(optimizer.param_groups) == 2, "A012 optimizer groups changed")
    require(abs(float(optimizer.param_groups[0]["lr"]) - A012_SPEC["base_lr"]) <= 1e-15, "A012 base LR changed")
    require(abs(float(optimizer.param_groups[1]["lr"]) - 2.0 * A012_SPEC["base_lr"]) <= 1e-15, "A012 adapter LR changed")
    require(all(abs(float(group["weight_decay"]) - A012_SPEC["base_weight_decay"]) <= 1e-15 for group in optimizer.param_groups), "A012 weight decay changed")
    require(type(scheduler).__name__ == "RatioPreservingCustomCosineAnnealingLR", "Ratio-preserving scheduler changed")
    require(abs(float(getattr(criterion.main_loss, "label_smoothing", -1.0)) - 0.05) <= 1e-12, "A012 label smoothing changed")
    require(abs(float(scheduler.eta_min) - 0.0001) <= 1e-12 and abs(float(scheduler.adapter_multiplier) - 2.0) <= 1e-12, "A012 scheduler eta-min/ratio changed")
    require(len([module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]) == 0, "Unexpected BatchNorm requires a train-only statistic policy")
    return model, criterion, optimizer, scheduler, dropout, groups


def paired_forward_shared_input_noise(model: torch.nn.Module, features: torch.Tensor):
    """Two stochastic forwards sharing A012 Gaussian feature noise.

    The original forward adds Gaussian noise immediately after Feature_Modal.
    A forward hook adds one explicitly sampled tensor at that same point while
    only the root Gaussian guard is suppressed. Child Dropout/MHA/drop-path
    modules remain in train mode and consume independent masks per forward.
    """
    require(model.training, "R-Drop paired forward requires train mode")
    per_feature_std = model._modal_noise_std[model.feature_to_modal].to(features)
    effective_std = per_feature_std * float(model.noise_scale)
    shared_noise = torch.randn_like(features) * effective_std.view(1, -1)
    require(bool(torch.isfinite(shared_noise).all()) and float(shared_noise.abs().max().detach().cpu()) > 0.0, "Shared A012 noise is invalid")
    original_root_training = bool(model.training)
    stochastic_children = [
        module
        for module in model.modules()
        if module is not model
        and isinstance(module, (torch.nn.Dropout, torch.nn.MultiheadAttention))
    ]
    require(stochastic_children and all(module.training for module in stochastic_children), "A012 dropout-family modules are not in train mode")
    calls = []

    def add_shared_noise(_module, _inputs, output):
        calls.append(shared_noise)
        return output + shared_noise

    handle = model.Feature_Modal.register_forward_hook(add_shared_noise)
    # Only the root flag guards the built-in Gaussian draw.  Directly
    # suppressing that flag avoids even consuming a discarded randn_like,
    # while every child Dropout/MHA/DIFFormer/drop-path module stays in train
    # mode and therefore draws an independent mask in the two forwards.
    model.training = False
    try:
        first = model(features)
        second = model(features)
    finally:
        model.training = original_root_training
        handle.remove()
    require(len(calls) == 2 and calls[0].data_ptr() == calls[1].data_ptr(), "The two forwards did not reuse identical input noise")
    return first, second, {
        "hook_calls": len(calls),
        "shared_noise_same_storage": True,
        "shared_noise_reuse_max_abs_diff": 0.0,
        "shared_noise_max_abs": float(shared_noise.abs().max().detach().cpu()),
        "root_training_suppressed_only": True,
        "dropout_family_children_kept_training": all(module.training for module in stochastic_children),
        "restored_root_training": bool(model.training),
        "noise_scale_unchanged": float(model.noise_scale),
    }


def rdrop_loss(
    criterion,
    first,
    second,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    rdrop_lambda: float,
) -> dict[str, torch.Tensor | int]:
    raw1, branches1, auxiliary1 = first
    raw2, branches2, auxiliary2 = second
    supervised1 = broad.loss_components(criterion, raw1, labels, train_mask, auxiliary1, A012_SPEC["lambda_aux"])
    supervised2 = broad.loss_components(criterion, raw2, labels, train_mask, auxiliary2, A012_SPEC["lambda_aux"])
    z1, z2 = raw1[train_mask], raw2[train_mask]
    require(z1.shape == z2.shape and z1.ndim == 2 and z1.size(1) == 3, "R-Drop KL logits changed")
    logp1, logp2 = F.log_softmax(z1, dim=-1), F.log_softmax(z2, dim=-1)
    p1, p2 = logp1.exp(), logp2.exp()
    kl_1_to_2 = (p1 * (logp1 - logp2)).sum(dim=-1).mean()
    kl_2_to_1 = (p2 * (logp2 - logp1)).sum(dim=-1).mean()
    symmetric_kl = 0.5 * (kl_1_to_2 + kl_2_to_1)
    supervised = 0.5 * (supervised1["total"] + supervised2["total"])
    total = supervised + float(rdrop_lambda) * symmetric_kl
    return {
        "total": total,
        "supervised": supervised,
        "symmetric_kl": symmetric_kl,
        "kl_1_to_2": kl_1_to_2,
        "kl_2_to_1": kl_2_to_1,
        "a012_loss_1": supervised1["total"],
        "a012_loss_2": supervised2["total"],
        "main_1": supervised1["main"],
        "main_2": supervised2["main"],
        "auxiliary_1": supervised1["auxiliary"],
        "auxiliary_2": supervised2["auxiliary"],
        "kl_rows": int(z1.size(0)),
    }


@torch.no_grad()
def evaluate_state(context: dict, model: torch.nn.Module, fold: int):
    model.eval()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    raw, _, _ = model(features)
    metrics, selection = cme.selection_metrics(
        raw,
        labels,
        test_mask,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )
    return raw, metrics, selection


def update_best(best: dict | None, epoch: int, metrics: dict, selection: tuple, state: dict, symmetric_kl: float):
    if best is None or selection > tuple(best["selection"]):
        return {
            "epoch": int(epoch),
            "selection": list(selection),
            "metrics": deepcopy(metrics),
            "state": {name: value.clone() for name, value in state.items()},
            "symmetric_kl": float(symmetric_kl),
        }
    return best


def best_metadata(best: dict | None):
    if best is None:
        return None
    return {key: deepcopy(value) for key, value in best.items() if key != "state"}


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # torch 2.5's serializer does not support uint32 storage; int64
            # exactly preserves every MT19937 uint32 word.
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(payload: dict) -> None:
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["torch_cuda_all"])


def training_checkpoint(
    lock: dict,
    fold: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    raw_best,
    lawa_best,
    lawa_queue,
    optimizer_steps: int,
    scheduler_steps: int,
    extra: dict | None = None,
) -> dict:
    require(len(lawa_queue) == min(epoch, LAWA_WINDOW), "LAWA queue depth changed")
    expected_queue_epochs = list(range(max(1, epoch - LAWA_WINDOW + 1), epoch + 1))
    require([int(entry["epoch"]) for entry in lawa_queue] == expected_queue_epochs, "LAWA queue is not the latest raw epoch sequence")
    current = clone_cpu_state(model)
    require(all(torch.equal(current[name], lawa_queue[-1]["state"][name]) for name in current), "Current raw model is not the newest LAWA queue state")
    payload = {
        "schema_version": 1,
        "fold_lock": lock,
        "fold": int(fold),
        "epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "model_state": current,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "raw_best": best_metadata(raw_best),
        "raw_best_state": None if raw_best is None else raw_best["state"],
        "lawa_best": best_metadata(lawa_best),
        "lawa_best_state": None if lawa_best is None else lawa_best["state"],
        "lawa_queue": [
            {
                "epoch": int(entry["epoch"]),
                "state": {name: value.clone() for name, value in entry["state"].items()},
            }
            for entry in lawa_queue
        ],
        "optimizer_steps": int(optimizer_steps),
        "scheduler_steps": int(scheduler_steps),
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }
    return payload


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(path: Path, expected_lock: dict, model, optimizer, scheduler):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("schema_version") == 1 and payload.get("fold_lock") == expected_lock, "Training checkpoint lock/schema changed")
    require(int(payload["next_epoch"]) == int(payload["epoch"]) + 1, "Training checkpoint epoch changed")
    require(int(payload["optimizer_steps"]) == int(payload["scheduler_steps"]) == int(payload["epoch"]), "Checkpoint update counts changed")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    require(int(scheduler.last_epoch) == int(payload["epoch"]), "Checkpoint scheduler epoch changed")
    base_lr, adapter_lr = [float(group["lr"]) for group in optimizer.param_groups]
    require(abs(adapter_lr - 2.0 * base_lr) <= max(1e-14, abs(adapter_lr) * 1e-12), "Reloaded optimizer LR ratio changed")
    raw_best = None if payload["raw_best"] is None else {**payload["raw_best"], "state": payload["raw_best_state"]}
    lawa_best = None if payload["lawa_best"] is None else {**payload["lawa_best"], "state": payload["lawa_best_state"]}
    checkpoint_epoch = int(payload["epoch"])
    require(raw_best is not None and 1 <= int(raw_best["epoch"]) <= checkpoint_epoch, "Loaded raw best state/epoch changed")
    if checkpoint_epoch < LAWA_WINDOW:
        require(lawa_best is None and payload["lawa_best_state"] is None, "Loaded LAWA best became eligible before epoch4")
    else:
        require(lawa_best is not None and LAWA_WINDOW <= int(lawa_best["epoch"]) <= checkpoint_epoch, "Loaded LAWA best state/epoch changed")
    queue = deque(payload["lawa_queue"], maxlen=LAWA_WINDOW)
    require(len(queue) == min(int(payload["epoch"]), LAWA_WINDOW), "Loaded LAWA queue depth changed")
    expected_queue_epochs = list(range(max(1, int(payload["epoch"]) - LAWA_WINDOW + 1), int(payload["epoch"]) + 1))
    require([int(entry["epoch"]) for entry in queue] == expected_queue_epochs, "Loaded LAWA queue epoch sequence changed")
    require(state_max_abs_diff(model, queue[-1]["state"]) == 0.0, "Loaded model differs from newest raw queue state")
    restore_rng_state(payload["rng_state"])
    return payload, raw_best, lawa_best, queue


def variant_config(context: dict, variant: str, rdrop_lambda: float) -> dict:
    require((variant, float(rdrop_lambda)) in {("v1", RDROP_LAMBDA_V1), ("v1_1", RDROP_LAMBDA_V1_1)}, "Unsupported R-Drop variant")
    payload = {
        "experiment": EXPERIMENT,
        "variant": variant,
        "rdrop_lambda": float(rdrop_lambda),
        "lawa_window": LAWA_WINDOW,
        "a012_spec": A012_SPEC,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / RUNNER_REL),
        "broad_runner_sha256": file_sha256(ROOT / "scripts/run_c1_broad_hparam_search_v1.py"),
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "parameter_count": EXPECTED_PARAMETERS,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "logit_adjust_tau": 0.75,
        "scheduler": "RatioPreservingCustomCosineAnnealingLR(T_max=400,base_eta_min=0.0001,adapter_eta_min=0.0002)",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "graph": False,
        "ema": False,
    }
    payload["config_sha256"] = payload_sha256(payload)
    return payload


def fold_lock(config: dict, fold: int) -> dict:
    return {
        "variant": config["variant"],
        "rdrop_lambda": config["rdrop_lambda"],
        "fold": int(fold),
        "source_commit": config["source_commit"],
        "config_sha256": config["config_sha256"],
        "fold_manifest_sha256": config["fold_manifest_sha256"],
    }


def train_one_epoch(
    context: dict,
    model,
    criterion,
    optimizer,
    scheduler,
    train_mask: torch.Tensor,
    rdrop_lambda: float,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    first, second, noise_audit = paired_forward_shared_input_noise(
        model, context["dataset_data"]["Feature"]
    )
    logit_difference = float(
        (first[0] - second[0]).abs().max().detach().cpu()
    )
    components = rdrop_loss(
        criterion,
        first,
        second,
        context["dataset_data"]["Label"],
        train_mask,
        rdrop_lambda,
    )
    tensor_components = {
        name: value for name, value in components.items() if isinstance(value, torch.Tensor)
    }
    require(
        all(bool(torch.isfinite(value)) for value in tensor_components.values()),
        "R-Drop loss contains NaN/Inf",
    )
    require(float(components["symmetric_kl"].detach().cpu()) > 0.0, "R-Drop symmetric KL is not positive")
    require(int(components["kl_rows"]) == int(train_mask.sum()), "R-Drop KL did not use exactly train_mask rows")
    components["total"].backward()
    require(gradients_finite(model), "R-Drop gradient contains NaN/Inf")
    if float(context["config"].grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
    optimizer.step()
    scheduler.step()
    base_lr, adapter_lr = [float(group["lr"]) for group in optimizer.param_groups]
    require(abs(adapter_lr - 2.0 * base_lr) <= max(1e-14, abs(adapter_lr) * 1e-12), "A012 scheduler lost adapter/base ratio")
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "R-Drop parameter contains NaN/Inf")
    return {
        "loss": {name: float(value.detach().cpu()) for name, value in tensor_components.items()},
        "kl_rows": int(components["kl_rows"]),
        "train_logit_max_abs_diff": logit_difference,
        "noise_audit": noise_audit,
        "base_lr": base_lr,
        "adapter_lr": adapter_lr,
    }


def optimizer_step_values(optimizer) -> list[int]:
    values = []
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            values.append(int(value.item() if isinstance(value, torch.Tensor) else value))
    return values


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    config = variant_config(context, "v1", RDROP_LAMBDA_V1)
    lock = fold_lock(config, 0)
    model, criterion, optimizer, scheduler, dropout, groups = make_training_objects(context)
    train_mask, _ = context["dataset_data"]["Mask"][0]
    initial_raw_state = clone_cpu_state(model)
    queue = deque(maxlen=LAWA_WINDOW)
    raw_best, lawa_best = None, None
    epoch_audits = []
    optimizer_steps = scheduler_steps = 0
    started = time.perf_counter()
    for epoch in range(1, 4):
        audit = train_one_epoch(
            context,
            model,
            criterion,
            optimizer,
            scheduler,
            train_mask,
            RDROP_LAMBDA_V1,
        )
        optimizer_steps += 1
        scheduler_steps += 1
        raw_state = clone_cpu_state(model)
        queue.append({"epoch": epoch, "state": raw_state})
        _, raw_metrics, raw_selection = evaluate_state(context, model, 0)
        raw_best = update_best(
            raw_best,
            epoch,
            raw_metrics,
            raw_selection,
            raw_state,
            audit["loss"]["symmetric_kl"],
        )
        require(lawa_best is None, "Formal LAWA must not be eligible before epoch4")
        audit.update({"epoch": epoch, "raw_metrics": raw_metrics, "formal_lawa_eligible": False})
        epoch_audits.append(audit)
    require([entry["epoch"] for entry in queue] == [1, 2, 3], "Smoke formal LAWA queue must be epochs1..3")
    require(lawa_best is None, "Smoke LAWA best must be explicitly unavailable before epoch4")
    require(optimizer_steps == scheduler_steps == 3, "Smoke update counters changed")
    steps = optimizer_step_values(optimizer)
    require(bool(steps) and set(steps) == {3}, f"Adam did not update exactly once per epoch: {set(steps)}")

    model.eval()
    with torch.no_grad():
        eval_first = model(context["dataset_data"]["Feature"])[0]
        eval_second = model(context["dataset_data"]["Feature"])[0]
    eval_repeat_diff = float((eval_first - eval_second).abs().max().cpu())
    require(eval_repeat_diff == 0.0, "Eval mode is not deterministic")
    require(all(item["train_logit_max_abs_diff"] > 0.0 for item in epoch_audits), "Train paired logits did not differ")
    require(all(item["loss"]["symmetric_kl"] > 0.0 for item in epoch_audits), "Smoke symmetric KL is not positive")

    raw_before_diagnostic = clone_cpu_state(model)
    optimizer_before = tensor_tree_digest(optimizer.state_dict())
    scheduler_before = tensor_tree_digest(scheduler.state_dict())
    diagnostic_states = [initial_raw_state] + [entry["state"] for entry in queue]
    require(len(diagnostic_states) == 4, "Smoke diagnostic LAWA needs theta0..theta3")
    diagnostic_average = average_raw_states(diagnostic_states)
    model.load_state_dict(diagnostic_average, strict=True)
    evaluate_state(context, model, 0)
    model.load_state_dict(raw_before_diagnostic, strict=True)
    diagnostic_restore_diff = state_max_abs_diff(model, raw_before_diagnostic)
    require(diagnostic_restore_diff == 0.0, "Diagnostic LAWA did not exactly restore raw parameters")
    require(tensor_tree_digest(optimizer.state_dict()) == optimizer_before, "Diagnostic LAWA changed optimizer state")
    require(tensor_tree_digest(scheduler.state_dict()) == scheduler_before, "Diagnostic LAWA changed scheduler state")

    checkpoint_path = smoke_root / "continuation_checkpoint.pt"
    checkpoint = training_checkpoint(
        lock,
        0,
        3,
        model,
        optimizer,
        scheduler,
        raw_best,
        lawa_best,
        queue,
        optimizer_steps,
        scheduler_steps,
        extra={
            "lawa_best_unavailable_reason": "not_eligible_before_epoch4",
            "smoke_diagnostic_only": True,
            "smoke_diagnostic_lawa_state": diagnostic_average,
            "smoke_diagnostic_state_epochs": [0, 1, 2, 3],
        },
    )
    save_checkpoint_atomic(checkpoint_path, checkpoint)
    reloaded_model, reloaded_criterion, reloaded_optimizer, reloaded_scheduler, _, _ = make_training_objects(context)
    loaded, loaded_raw_best, loaded_lawa_best, loaded_queue = load_training_checkpoint(
        checkpoint_path,
        lock,
        reloaded_model,
        reloaded_optimizer,
        reloaded_scheduler,
    )
    require(loaded["next_epoch"] == 4 and [entry["epoch"] for entry in loaded_queue] == [1, 2, 3], "Smoke checkpoint is not ready to continue at epoch4")
    require(loaded_raw_best is not None and loaded_lawa_best is None, "Smoke best-state continuation schema changed")
    require(loaded["extra"]["smoke_diagnostic_only"] is True, "Smoke diagnostic state lost its isolation marker")
    require(state_max_abs_diff(reloaded_model, queue[-1]["state"]) == 0.0, "Reloaded raw model changed")
    report = {
        "passed": True,
        "fold": 0,
        "optimizer_epochs": 3,
        "optimizer_steps": optimizer_steps,
        "scheduler_steps": scheduler_steps,
        "epoch_audits": epoch_audits,
        "eval_repeat_max_abs_diff": eval_repeat_diff,
        "formal_lawa_queue_epochs": [1, 2, 3],
        "formal_lawa_best_state": None,
        "formal_lawa_best_reason": "not_eligible_before_epoch4",
        "diagnostic_lawa": {
            "diagnostic_only": True,
            "state_epochs": [0, 1, 2, 3],
            "raw_restore_max_abs_diff": diagnostic_restore_diff,
            "optimizer_state_unchanged": True,
            "scheduler_state_unchanged": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "required_keys": sorted(checkpoint),
            "strict_fresh_object_load": True,
            "continuation_next_epoch": loaded["next_epoch"],
            "continuation_queue_epochs": [entry["epoch"] for entry in loaded_queue],
        },
        "parameter_count": parameter_count(model),
        "single_model": True,
        "ensemble": False,
        "source_commit": config["source_commit"],
        "runner_sha256": config["runner_sha256"],
        "wall_seconds": float(time.perf_counter() - started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_rdrop_a012_lawa4_v1.py smoke --device cuda:0',
        "config": config,
    }
    write_json(smoke_root / "smoke_config.json", config)
    write_json(smoke_root / "smoke_report.json", report)
    del model, criterion, optimizer, scheduler
    del reloaded_model, reloaded_criterion, reloaded_optimizer, reloaded_scheduler
    torch.cuda.empty_cache()
    return report


def validate_fold_rows(context: dict, rows: list[dict], fold: int) -> list[dict]:
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(subjects) == len(set(subjects)), f"fold{fold}: duplicate OOF subjects")
    expected_item = context["fold_manifest"]["folds"][fold]
    expected = dict(zip(expected_item["test_subject_indices"], expected_item["test_truth"]))
    require({int(row["subject_index"]): int(row["truth"]) for row in rows} == expected, f"fold{fold}: OOF manifest changed")
    broad.validate_prediction_rows(rows, expected_fold=fold)
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def validate_full_oof(rows: list[dict]) -> list[dict]:
    require(len(rows) == 598, "OOF row count changed")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == 598 and set(subjects) == set(range(598)), "OOF subject coverage changed")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "OOF fold coverage changed")
    broad.validate_prediction_rows(rows)
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def require_rows_match(actual: list[dict], expected: list[dict], context_text: str, tolerance: float = 1e-7) -> None:
    actual = sorted(actual, key=lambda row: int(row["subject_index"]))
    expected = sorted(expected, key=lambda row: int(row["subject_index"]))
    require([int(row["subject_index"]) for row in actual] == [int(row["subject_index"]) for row in expected], f"{context_text}: subject IDs changed")
    for new, old in zip(actual, expected):
        for key in ("fold", "subject_index", "truth", "prediction"):
            require(int(new[key]) == int(old[key]), f"{context_text}: {key} changed")
        difference = max(
            abs(float(new[f"probability_{name}"]) - float(old[f"probability_{name}"]))
            for name in CLASS_NAMES
        )
        require(difference <= tolerance, f"{context_text}: probability changed by {difference}")


def load_completed_fold(context: dict, config: dict, fold: int, final_dir: Path):
    if not final_dir.exists():
        return None
    required = [
        final_dir / name
        for name in (
            "checkpoint_best.pt",
            "summary.json",
            "raw_rdrop_oof.csv",
            "rdrop_lawa4_oof.csv",
            "config.json",
            "complete.json",
        )
    ]
    require(all(path.is_file() for path in required), f"{config['variant']} fold{fold}: completed artifact is incomplete")
    lock = fold_lock(config, fold)
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == lock, f"{config['variant']} fold{fold}: config/source changed")
    marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
    require(marker == {"complete": True, **lock}, f"{config['variant']} fold{fold}: completion marker changed")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary["fold_lock"] == lock, f"{config['variant']} fold{fold}: summary lock changed")
    raw_rows = validate_fold_rows(context, read_csv(final_dir / "raw_rdrop_oof.csv"), fold)
    lawa_rows = validate_fold_rows(context, read_csv(final_dir / "rdrop_lawa4_oof.csv"), fold)
    require(cme.metrics_from_rows(raw_rows) == summary["raw_best"]["metrics"], f"{config['variant']} fold{fold}: raw OOF metrics changed")
    require(cme.metrics_from_rows(lawa_rows) == summary["lawa_best"]["metrics"], f"{config['variant']} fold{fold}: LAWA OOF metrics changed")
    checkpoint = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(checkpoint.get("fold_lock") == lock, f"{config['variant']} fold{fold}: best checkpoint lock changed")
    require(checkpoint.get("raw_best") == summary["raw_best"] and checkpoint.get("lawa_best") == summary["lawa_best"], f"{config['variant']} fold{fold}: best checkpoint metadata changed")
    require(checkpoint.get("raw_best_state") is not None and checkpoint.get("lawa_best_state") is not None, f"{config['variant']} fold{fold}: best state missing")
    model = broad.build_model(context, 8)
    model.load_state_dict(checkpoint["raw_best_state"], strict=True)
    model.eval()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    with torch.no_grad():
        raw_logits, _, _ = model(features)
    raw_readback = cme.prediction_rows(fold, raw_logits, labels, test_mask, context["dataset_dict"], context["config"])
    require_rows_match(raw_readback, raw_rows, f"{config['variant']} fold{fold} raw checkpoint readback")
    model.load_state_dict(checkpoint["lawa_best_state"], strict=True)
    model.eval()
    with torch.no_grad():
        lawa_logits, _, _ = model(features)
    lawa_readback = cme.prediction_rows(fold, lawa_logits, labels, test_mask, context["dataset_dict"], context["config"])
    require_rows_match(lawa_readback, lawa_rows, f"{config['variant']} fold{fold} LAWA checkpoint readback")
    del model
    torch.cuda.empty_cache()
    print(f"RESUME {config['variant']} fold={fold} raw_best={summary['raw_best']['epoch']} lawa_best={summary['lawa_best']['epoch']}", flush=True)
    return summary, raw_rows, lawa_rows


def train_fold(context: dict, output_root: Path, config: dict, fold: int):
    variant_root = output_root / config["variant"]
    final_dir = variant_root / f"fold_{fold:02d}"
    completed = load_completed_fold(context, config, fold, final_dir)
    if completed is not None:
        return completed
    staging = variant_root / f".fold_{fold:02d}_in_progress"
    lock = fold_lock(config, fold)
    resume_path = staging / "resume_checkpoint.pt"
    if staging.exists():
        require((staging / "config.json").is_file(), f"{config['variant']} fold{fold}: in-progress config missing")
        require(json.loads((staging / "config.json").read_text(encoding="utf-8")) == lock, f"{config['variant']} fold{fold}: in-progress config/source changed")
        if not resume_path.is_file():
            resolved = staging.resolve()
            require(resolved.parent == variant_root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe incomplete-fold restart target")
            shutil.rmtree(resolved)
    if not staging.exists():
        staging.mkdir(parents=True)
        write_json(staging / "config.json", lock)

    model, criterion, optimizer, scheduler, dropout, groups = make_training_objects(context)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool((train_mask & test_mask).any()), f"{config['variant']} fold{fold}: train/test overlap")
    raw_best = lawa_best = None
    queue = deque(maxlen=LAWA_WINDOW)
    optimizer_steps = scheduler_steps = 0
    start_epoch = 1
    resumed_from_epoch = 0
    elapsed_before_resume = 0.0
    if resume_path.is_file():
        loaded, raw_best, lawa_best, queue = load_training_checkpoint(
            resume_path, lock, model, optimizer, scheduler
        )
        resumed_from_epoch = int(loaded["epoch"])
        start_epoch = int(loaded["next_epoch"])
        optimizer_steps = int(loaded["optimizer_steps"])
        scheduler_steps = int(loaded["scheduler_steps"])
        elapsed_before_resume = float(loaded.get("extra", {}).get("elapsed_seconds_accumulated", 0.0))
        last_epoch_audit = {
            "loss": {
                "symmetric_kl": float(loaded.get("extra", {}).get("last_epoch_symmetric_kl", float("nan")))
            }
        }
        require(math.isfinite(last_epoch_audit["loss"]["symmetric_kl"]), f"{config['variant']} fold{fold}: resume KL audit missing")
        print(f"RESUME_IN_PROGRESS {config['variant']} fold={fold} next_epoch={start_epoch}", flush=True)

    maximum_restore_diff = 0.0
    first_lawa_epoch = 4 if resumed_from_epoch >= 4 else None
    if resumed_from_epoch == 0:
        last_epoch_audit = None
    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        audit = train_one_epoch(
            context,
            model,
            criterion,
            optimizer,
            scheduler,
            train_mask,
            float(config["rdrop_lambda"]),
        )
        optimizer_steps += 1
        scheduler_steps += 1
        require(optimizer_steps == scheduler_steps == epoch, f"{config['variant']} fold{fold}: update count changed")
        raw_state = clone_cpu_state(model)
        queue.append({"epoch": epoch, "state": raw_state})
        expected_epochs = list(range(max(1, epoch - LAWA_WINDOW + 1), epoch + 1))
        require([entry["epoch"] for entry in queue] == expected_epochs, f"{config['variant']} fold{fold}: raw LAWA queue changed")
        _, raw_metrics, raw_selection = evaluate_state(context, model, fold)
        raw_best = update_best(
            raw_best,
            epoch,
            raw_metrics,
            raw_selection,
            raw_state,
            audit["loss"]["symmetric_kl"],
        )
        if epoch < LAWA_WINDOW:
            require(lawa_best is None, f"{config['variant']} fold{fold}: LAWA became eligible before epoch4")
        else:
            if first_lawa_epoch is None:
                first_lawa_epoch = epoch
                require(epoch == 4 and [entry["epoch"] for entry in queue] == [1, 2, 3, 4], f"{config['variant']} fold{fold}: first formal LAWA queue changed")
            pre_last_epoch = int(scheduler.last_epoch)
            pre_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            averaged = average_raw_states([entry["state"] for entry in queue])
            model.load_state_dict(averaged, strict=True)
            _, lawa_metrics, lawa_selection = evaluate_state(context, model, fold)
            lawa_best = update_best(
                lawa_best,
                epoch,
                lawa_metrics,
                lawa_selection,
                averaged,
                audit["loss"]["symmetric_kl"],
            )
            model.load_state_dict(raw_state, strict=True)
            restore_diff = state_max_abs_diff(model, raw_state)
            maximum_restore_diff = max(maximum_restore_diff, restore_diff)
            require(restore_diff == 0.0, f"{config['variant']} fold{fold}: LAWA evaluation contaminated raw trajectory")
            require(int(scheduler.last_epoch) == pre_last_epoch and [float(group["lr"]) for group in optimizer.param_groups] == pre_lrs, f"{config['variant']} fold{fold}: LAWA evaluation changed optimizer/scheduler")
        last_epoch_audit = audit
        if epoch % 20 == 0 or epoch == EPOCHS:
            checkpoint = training_checkpoint(
                lock,
                fold,
                epoch,
                model,
                optimizer,
                scheduler,
                raw_best,
                lawa_best,
                queue,
                optimizer_steps,
                scheduler_steps,
                extra={
                    "last_epoch_symmetric_kl": audit["loss"]["symmetric_kl"],
                    "elapsed_seconds_accumulated": elapsed_before_resume + float(time.perf_counter() - started),
                },
            )
            save_checkpoint_atomic(resume_path, checkpoint)

    require(raw_best is not None and lawa_best is not None, f"{config['variant']} fold{fold}: best state missing")
    require(first_lawa_epoch in {None, 4}, f"{config['variant']} fold{fold}: LAWA first epoch changed")
    labels = context["dataset_data"]["Label"]
    features = context["dataset_data"]["Feature"]
    model.load_state_dict(raw_best["state"], strict=True)
    model.eval()
    with torch.no_grad():
        raw_logits, _, _ = model(features)
    raw_rows = validate_fold_rows(
        context,
        cme.prediction_rows(fold, raw_logits, labels, test_mask, context["dataset_dict"], context["config"]),
        fold,
    )
    require(cme.metrics_from_rows(raw_rows) == raw_best["metrics"], f"{config['variant']} fold{fold}: raw best readback changed")
    model.load_state_dict(lawa_best["state"], strict=True)
    model.eval()
    with torch.no_grad():
        lawa_logits, _, _ = model(features)
    lawa_rows = validate_fold_rows(
        context,
        cme.prediction_rows(fold, lawa_logits, labels, test_mask, context["dataset_dict"], context["config"]),
        fold,
    )
    require(cme.metrics_from_rows(lawa_rows) == lawa_best["metrics"], f"{config['variant']} fold{fold}: LAWA best readback changed")
    summary = {
        "variant": config["variant"],
        "fold": fold,
        "raw_best": best_metadata(raw_best),
        "lawa_best": best_metadata(lawa_best),
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "parameter_count": parameter_count(model),
        "optimizer_steps": optimizer_steps,
        "scheduler_steps": scheduler_steps,
        "formal_lawa_first_epoch": 4,
        "final_lawa_queue_epochs": [entry["epoch"] for entry in queue],
        "maximum_lawa_raw_restore_diff": maximum_restore_diff,
        "last_epoch_symmetric_kl": last_epoch_audit["loss"]["symmetric_kl"],
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds_this_session": float(time.perf_counter() - started),
        "elapsed_seconds_total": elapsed_before_resume + float(time.perf_counter() - started),
        "fold_lock": lock,
    }
    best_checkpoint = {
        "fold_lock": lock,
        "raw_best": summary["raw_best"],
        "raw_best_state": raw_best["state"],
        "lawa_best": summary["lawa_best"],
        "lawa_best_state": lawa_best["state"],
    }
    torch.save(best_checkpoint, staging / "checkpoint_best.pt")
    write_json(staging / "summary.json", summary)
    write_csv(staging / "raw_rdrop_oof.csv", raw_rows)
    write_csv(staging / "rdrop_lawa4_oof.csv", lawa_rows)
    write_json(staging / "complete.json", {"complete": True, **lock})
    for path in (resume_path, resume_path.with_suffix(resume_path.suffix + ".tmp")):
        if path.exists():
            path.unlink()
    staging.rename(final_dir)
    print(f"{config['variant']} fold={fold} raw_best={raw_best['epoch']}:{raw_best['metrics']['correct']} lawa_best={lawa_best['epoch']}:{lawa_best['metrics']['correct']}", flush=True)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, raw_rows, lawa_rows


def boundary_errors(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        "AD_SMCI": int(matrix[0, 2] + matrix[2, 0]),
        "CN_SMCI": int(matrix[1, 2] + matrix[2, 1]),
        "AD_CN": int(matrix[0, 1] + matrix[1, 0]),
    }


def candidate_rank(candidate: dict) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["correct"]),
        float(metrics["macro_auc"]),
        float(metrics["macro_f1"]),
    )


def candidate_report(name: str, rows: list[dict], summaries: list[dict], best_key: str) -> dict:
    rows = validate_full_oof(rows)
    metrics = cme.metrics_from_rows(rows)
    boundaries = boundary_errors(metrics)
    predicted = {
        CLASS_NAMES[index]: sum(int(row["prediction"]) == index for row in rows)
        for index in range(3)
    }
    fold_acc = np.asarray(
        [float(summary[best_key]["metrics"]["acc"]) for summary in summaries],
        dtype=np.float64,
    )
    return {
        "name": name,
        "metrics": metrics,
        "metric_delta_vs_a012": {
            key: float(metrics[key]) - float(A012[key])
            for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
        "comparison_vs_a012": {"available": False},
        "boundary_errors": boundaries,
        "adjacent_boundary_errors": int(boundaries["AD_SMCI"] + boundaries["CN_SMCI"]),
        "predicted_class_counts": predicted,
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_std": float(fold_acc.std(ddof=1)),
        "best_epoch_mean_symmetric_kl": float(
            np.mean([float(summary[best_key]["symmetric_kl"]) for summary in summaries])
        ),
        "folds": [
            {
                "fold": int(summary["fold"]),
                "best_epoch": int(summary[best_key]["epoch"]),
                "correct": int(summary[best_key]["metrics"]["correct"]),
                "acc": float(summary[best_key]["metrics"]["acc"]),
                "symmetric_kl": float(summary[best_key]["symmetric_kl"]),
            }
            for summary in summaries
        ],
        "parameter_count": EXPECTED_PARAMETERS,
        "single_model": True,
        "ensemble": False,
    }


def run_variant(context: dict, output_root: Path, variant: str, rdrop_lambda: float):
    variant_root = output_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    config = variant_config(context, variant, rdrop_lambda)
    config_path = variant_root / "config.json"
    if config_path.is_file():
        require(json.loads(config_path.read_text(encoding="utf-8")) == config, f"{variant}: formal config/source changed")
    else:
        write_json(config_path, config)
    summaries, raw_rows, lawa_rows = [], [], []
    for fold in FOLDS:
        summary, fold_raw, fold_lawa = train_fold(context, output_root, config, fold)
        summaries.append(summary)
        raw_rows.extend(fold_raw)
        lawa_rows.extend(fold_lawa)
    raw_rows = validate_full_oof(raw_rows)
    lawa_rows = validate_full_oof(lawa_rows)
    raw = candidate_report("raw_rdrop", raw_rows, summaries, "raw_best")
    lawa = candidate_report("rdrop_lawa4", lawa_rows, summaries, "lawa_best")
    report = {
        "variant": variant,
        "rdrop_lambda": float(rdrop_lambda),
        "lawa_window": LAWA_WINDOW,
        "candidates": {"raw_rdrop": raw, "rdrop_lawa4": lawa},
        "ranking": [
            candidate["name"]
            for candidate in sorted((raw, lawa), key=candidate_rank, reverse=True)
        ],
        "config": config,
        "folds": summaries,
        "training_seconds_this_session": float(
            sum(summary["elapsed_seconds_this_session"] for summary in summaries)
        ),
        "training_seconds_total": float(
            sum(summary["elapsed_seconds_total"] for summary in summaries)
        ),
    }
    write_csv(variant_root / "raw_rdrop_oof.csv", raw_rows)
    write_csv(variant_root / "rdrop_lawa4_oof.csv", lawa_rows)
    write_json(variant_root / "report.json", report)
    write_json(
        variant_root / "complete.json",
        {
            "complete": True,
            "variant": variant,
            "source_commit": config["source_commit"],
            "config_sha256": config["config_sha256"],
            "fold_manifest_sha256": config["fold_manifest_sha256"],
        },
    )
    return report, {"raw_rdrop": raw_rows, "rdrop_lawa4": lawa_rows}


def a012_reference_path() -> Path | None:
    candidates = [
        ROOT.parent / "c1_broad_hparam_search_v1/experiments/c1_broad_hparam_search_v1/stage_a/trial_A012/oof_predictions.csv",
        ROOT.parent / "bp_ads_v1/experiments/bp_ads_v1/checkpoint_materialization/a012/oof_predictions.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_a012_reference() -> tuple[dict[int, dict] | None, dict]:
    path = a012_reference_path()
    if path is None:
        return None, {"available": False, "reason": "A012 subject-level OOF unavailable; A012 was not rerun"}
    rows = validate_full_oof(read_csv(path))
    metrics = cme.metrics_from_rows(rows)
    require(metrics["correct"] == A012["correct"] and metrics["confusion_matrix"] == A012["confusion_matrix"], "A012 OOF anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(float(metrics[key]) - float(A012[key])) <= 5e-7, f"A012 OOF metric changed: {key}")
    return {int(row["subject_index"]): row for row in rows}, {
        "available": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "subjects": 598,
        "explicit_alignment_key": "subject_index",
    }


def attach_a012_comparison(candidate: dict, rows: list[dict], reference: dict[int, dict] | None) -> None:
    if reference is None:
        candidate["comparison_vs_a012"] = {"available": False}
        return
    repairs = damages = changed = 0
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in reference, f"A012 missing subject {subject}")
        old = reference[subject]
        require(int(old["truth"]) == int(row["truth"]) and int(old["fold"]) == int(row["fold"]), "A012/candidate subject-fold-truth alignment changed")
        truth, old_prediction, new_prediction = int(row["truth"]), int(old["prediction"]), int(row["prediction"])
        repairs += int(old_prediction != truth and new_prediction == truth)
        damages += int(old_prediction == truth and new_prediction != truth)
        changed += int(old_prediction != new_prediction)
    require(repairs - damages == int(candidate["metrics"]["correct"]) - A012["correct"], "A012 paired comparison net changed")
    candidate["comparison_vs_a012"] = {
        "available": True,
        "repairs": repairs,
        "damages": damages,
        "net_repairs": repairs - damages,
        "changed": changed,
    }


def candidate_success(candidate: dict) -> bool:
    comparison = candidate["comparison_vs_a012"]
    net_repairs = comparison.get("net_repairs", int(candidate["metrics"]["correct"]) - A012["correct"])
    return (
        int(candidate["metrics"]["correct"]) >= 564
        and int(candidate["boundary_errors"]["AD_CN"]) == 0
        and float(candidate["metrics"]["bacc"]) >= BACC_FLOOR
        and int(net_repairs) >= 2
        and int(candidate["adjacent_boundary_errors"]) <= 34
    )


def candidate_weak(candidate: dict) -> bool:
    comparison = candidate["comparison_vs_a012"]
    net_repairs = comparison.get("net_repairs", int(candidate["metrics"]["correct"]) - A012["correct"])
    return (
        int(candidate["metrics"]["correct"]) == 563
        and int(candidate["boundary_errors"]["AD_CN"]) == 0
        and float(candidate["metrics"]["bacc"]) >= BACC_FLOOR
        and int(net_repairs) > 0
    )


def preliminary_variant_outcome(report: dict) -> dict:
    candidates = list(report["candidates"].values())
    # At this point A012 OOF has deliberately not been loaded. Net repairs is
    # nevertheless fixed by Correct(candidate)-Correct(A012) on the same 598 truths.
    require(all(candidate["comparison_vs_a012"].get("available") is False for candidate in candidates), "A012 OOF entered pre-v1.1 training decision")
    successful = sorted(
        (candidate for candidate in candidates if candidate_success(candidate)),
        key=candidate_rank,
        reverse=True,
    )
    weak = sorted(
        (candidate for candidate in candidates if candidate_weak(candidate)),
        key=candidate_rank,
        reverse=True,
    )
    if successful:
        selected = successful[0]
        decision = "R_DROP_SUCCESS"
    elif weak:
        selected = weak[0]
        decision = "R_DROP_WEAK_GAIN"
    else:
        selected = sorted(candidates, key=candidate_rank, reverse=True)[0]
        decision = "R_DROP_NO_GAIN"
    return {"decision": decision, "selected": selected}


def final_variant_outcome(report: dict) -> dict:
    candidates = list(report["candidates"].values())
    successful = sorted(
        (candidate for candidate in candidates if candidate_success(candidate)),
        key=candidate_rank,
        reverse=True,
    )
    weak = sorted(
        (candidate for candidate in candidates if candidate_weak(candidate)),
        key=candidate_rank,
        reverse=True,
    )
    for candidate in candidates:
        candidate["meets_success"] = candidate_success(candidate)
        candidate["meets_weak_gain"] = candidate_weak(candidate)
    if successful:
        return {"decision": "R_DROP_SUCCESS", "selected": successful[0]}
    if weak:
        return {"decision": "R_DROP_WEAK_GAIN", "selected": weak[0]}
    return {
        "decision": "R_DROP_NO_GAIN",
        "selected": sorted(candidates, key=candidate_rank, reverse=True)[0],
    }


def v1_1_can_replace(v1: dict, v1_1: dict) -> tuple[bool, str]:
    new = v1_1["selected"]
    old = v1["selected"]
    new_correct, old_correct = int(new["metrics"]["correct"]), int(old["metrics"]["correct"])
    if v1_1["decision"] == "R_DROP_NO_GAIN":
        return False, "v1.1 outcome was R_DROP_NO_GAIN"
    eligible = candidate_success(new) if new_correct >= 564 else candidate_weak(new)
    if not eligible:
        return False, "v1.1 failed its full SUCCESS/WEAK safety gate"
    if float(new["metrics"]["bacc"]) < float(old["metrics"]["bacc"]) - 0.002:
        return False, "v1.1 BACC fell more than 0.002 below v1"
    if new_correct > old_correct:
        return True, "v1.1 Correct strictly exceeded v1"
    if new_correct < old_correct:
        return False, "v1.1 Correct was lower than v1"
    new_tie = (float(new["metrics"]["macro_auc"]), float(new["metrics"]["macro_f1"]))
    old_tie = (float(old["metrics"]["macro_auc"]), float(old["metrics"]["macro_f1"]))
    return (new_tie > old_tie, "v1.1 tied Correct and won AUC>Macro-F1" if new_tie > old_tie else "v1.1 tied Correct without winning AUC>Macro-F1")


def metric_line(candidate: dict) -> str:
    metrics = candidate["metrics"]
    comparison = candidate["comparison_vs_a012"]
    paired = (
        f"repairs/damages/changed={comparison['repairs']}/{comparison['damages']}/{comparison['changed']}"
        if comparison.get("available")
        else "repairs/damages/changed=unavailable"
    )
    return (
        f"Correct={metrics['correct']}/598; ACC={metrics['acc']:.7f}; Macro-F1={metrics['macro_f1']:.7f}; "
        f"BACC={metrics['bacc']:.7f}; Probability Macro-AUC={metrics['macro_auc']:.7f}; "
        f"Weighted-F1={metrics['weighted_f1']:.7f}; confusion={metrics['confusion_matrix']}; "
        f"predicted={candidate['predicted_class_counts']}; {paired}; "
        f"AD–sMCI/CN–sMCI/AD–CN={candidate['boundary_errors']['AD_SMCI']}/{candidate['boundary_errors']['CN_SMCI']}/{candidate['boundary_errors']['AD_CN']}"
    )


def render_report(summary: dict) -> str:
    lines = [
        "# R-Drop A012 + same-trajectory LAWA-4 v1",
        "",
        f"Decision: `{summary['decision']}`; selected: `{summary['selected']['variant']}/{summary['selected']['candidate']['name']}`.",
        f"Branch=`{summary['branch']}`; source commit=`{summary['source_commit']}`; device={summary['device']} ({summary['device_name']}); formal invocation wall time={summary['formal_wall_seconds']:.1f}s; summed fold training time={summary['total_training_seconds']:.1f}s.",
        "",
    ]
    for variant_name in ("v1", "v1_1"):
        if variant_name not in summary["variants"]:
            continue
        variant = summary["variants"][variant_name]
        lines.extend([f"## {variant_name} (rdrop_lambda={variant['rdrop_lambda']})", ""])
        for candidate_name in ("raw_rdrop", "rdrop_lawa4"):
            candidate = variant["candidates"][candidate_name]
            lines.extend(
                [
                    f"### {candidate_name}",
                    "",
                    metric_line(candidate),
                    f"Metric delta vs A012: {candidate['metric_delta_vs_a012']}.",
                    f"Fold ACC mean ± sample SD={candidate['fold_acc_mean']:.7f} ± {candidate['fold_acc_sample_std']:.7f}; best-epoch mean L_symKL={candidate['best_epoch_mean_symmetric_kl']:.8g}.",
                    "Fold best epoch/ACC: " + ", ".join(f"{fold['fold']}:{fold['best_epoch']}/{fold['acc']:.7f}" for fold in candidate["folds"]) + ".",
                    f"Parameters={candidate['parameter_count']}; single_model={candidate['single_model']}; ensemble={candidate['ensemble']}.",
                    "",
                ]
            )
    lines.extend(
        [
            "## v1.1 gate",
            "",
            f"Triggered={summary['v1_1']['triggered']}; reason={summary['v1_1']['trigger_reason']}; replacement={summary['v1_1']['replaced_v1']}; replacement reason={summary['v1_1']['replacement_reason']}.",
            "",
            "## Final",
            "",
            f"Selected result: {metric_line(summary['selected']['candidate'])}.",
            f"Next recommendation: `{summary['next_recommendation']}`. It was not implemented or run.",
            "",
            "## Reproduction",
            "",
            "```text",
            summary["run_commands"]["smoke"],
            summary["run_commands"]["formal"],
            "```",
        ]
    )
    return "\n".join(lines)


def write_final_artifacts(
    output_root: Path,
    variants: dict,
    rows_by_variant: dict,
    a012_reference_info: dict,
    v1_1_trigger: dict,
    final_selected: dict,
    decision: str,
    formal_started: float,
) -> dict:
    branch = git("branch", "--show-current").stdout.strip()
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    selected_variant = final_selected["variant"]
    selected_candidate = final_selected["candidate"]
    summary = {
        "experiment": EXPERIMENT,
        "decision": decision,
        "selected": final_selected,
        "variants": variants,
        "a012_reference": {"metrics": A012, **a012_reference_info},
        "v1_1": v1_1_trigger,
        "branch": branch,
        "source_commit": source_commit,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "formal_wall_seconds": float(time.perf_counter() - formal_started),
        "total_training_seconds": float(
            sum(report["training_seconds_total"] for report in variants.values())
        ),
        "next_recommendation": "A012-SAM-ON" if decision == "R_DROP_NO_GAIN" else "none; stop after this experiment",
        "run_commands": {
            "smoke": f'"{sys.executable}" -u -B scripts/run_rdrop_a012_lawa4_v1.py smoke --device cuda:0',
            "formal": f'"{sys.executable}" -u -B scripts/run_rdrop_a012_lawa4_v1.py formal --device cuda:0',
        },
    }
    fold_results = {
        variant_name: report["folds"] for variant_name, report in variants.items()
    }
    write_csv(output_root / "raw_rdrop_oof.csv", rows_by_variant["v1"]["raw_rdrop"])
    write_csv(output_root / "rdrop_lawa4_oof.csv", rows_by_variant["v1"]["rdrop_lawa4"])
    if "v1_1" in rows_by_variant:
        write_csv(output_root / "raw_rdrop_v1_1_oof.csv", rows_by_variant["v1_1"]["raw_rdrop"])
        write_csv(output_root / "rdrop_lawa4_v1_1_oof.csv", rows_by_variant["v1_1"]["rdrop_lawa4"])
    write_json(output_root / "fold_results.json", fold_results)
    write_json(output_root / "formal_summary.json", summary)
    (output_root / "REPORT.md").write_text(render_report(summary) + "\n", encoding="utf-8")
    return summary


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode == 0, "Source/config files must be committed before smoke/formal")
    require(git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode == 0, "Committed source/config files changed")


def locked_dependencies() -> list[Path]:
    return [
        ROOT / cme.CONFIG_REL,
        ROOT / "Model/cme_dual_branch.py",
        ROOT / "Model/network.py",
        ROOT / "Loss/loss_fn.py",
        ROOT / "Utils/utils.py",
        ROOT / "scripts/run_cme_dual_branch_v1.py",
        ROOT / "scripts/run_c1_broad_hparam_search_v1.py",
    ]


def validate_source_commit(output_root: Path) -> None:
    current = git("rev-parse", "HEAD").stdout.strip()
    require(current != BASE_COMMIT, "A source commit is required before smoke/formal")
    required = [
        ROOT / RUNNER_REL,
        ROOT / ".gitignore",
        output_root / "experiment_config.json",
        *locked_dependencies(),
    ]
    require_committed_unchanged(required)
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in locked_dependencies()]
    require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", *relative, check=False).returncode == 0, "Locked A012 implementation/dependency changed")


def validate_formal_gate(output_root: Path) -> None:
    validate_source_commit(output_root)
    report_path = output_root / "smoke/smoke_report.json"
    config_path = output_root / "smoke/smoke_config.json"
    require(report_path.is_file() and config_path.is_file(), "Passing CUDA smoke is required before formal")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(report.get("passed") is True and report.get("optimizer_epochs") == 3, "Smoke report is invalid")
    require(report.get("config") == config, "Smoke report/config changed")
    require(config.get("source_commit") == git("rev-parse", "HEAD").stdout.strip(), "Formal source differs from smoke source commit")
    require(config.get("runner_sha256") == file_sha256(ROOT / RUNNER_REL), "Runner differs from smoke-tested source")
    require(report.get("optimizer_steps") == report.get("scheduler_steps") == 3, "Smoke update count gate failed")
    require(report.get("eval_repeat_max_abs_diff") == 0.0, "Smoke deterministic-eval gate failed")
    require(report.get("diagnostic_lawa", {}).get("raw_restore_max_abs_diff") == 0.0, "Smoke LAWA restore gate failed")
    require(report.get("checkpoint", {}).get("continuation_next_epoch") == 4, "Smoke continuation gate failed")


def run_formal(context: dict, output_root: Path) -> dict:
    validate_formal_gate(output_root)
    formal_started = time.perf_counter()
    v1_report, v1_rows = run_variant(context, output_root, "v1", RDROP_LAMBDA_V1)
    preliminary = preliminary_variant_outcome(v1_report)
    trigger_v1_1 = preliminary["decision"] == "R_DROP_WEAK_GAIN"
    variants = {"v1": v1_report}
    rows_by_variant = {"v1": v1_rows}
    trigger_reason = (
        "v1 safety-filtered best candidate was exactly 563 with AD-CN=0, BACC>=0.92663, and net repairs>0"
        if trigger_v1_1
        else f"v1 preliminary decision was {preliminary['decision']}; the only allowed v1.1 gate did not open"
    )
    if trigger_v1_1:
        v1_1_report, v1_1_rows = run_variant(
            context, output_root, "v1_1", RDROP_LAMBDA_V1_1
        )
        variants["v1_1"] = v1_1_report
        rows_by_variant["v1_1"] = v1_1_rows

    # A012 subject-level OOF is intentionally loaded only after every allowed
    # training run is complete. It cannot influence loss, best epochs or LAWA.
    a012_reference, a012_info = load_a012_reference()
    for variant_name, report in variants.items():
        for candidate_name, candidate in report["candidates"].items():
            attach_a012_comparison(
                candidate,
                rows_by_variant[variant_name][candidate_name],
                a012_reference,
            )
    v1_outcome = final_variant_outcome(v1_report)
    selected_variant = "v1"
    selected_outcome = v1_outcome
    replaced_v1 = False
    replacement_reason = "v1.1 was not triggered"
    if trigger_v1_1:
        v1_1_outcome = final_variant_outcome(variants["v1_1"])
        replaced_v1, replacement_reason = v1_1_can_replace(v1_outcome, v1_1_outcome)
        if replaced_v1:
            selected_variant = "v1_1"
            selected_outcome = v1_1_outcome
    selected_candidate = selected_outcome["selected"]
    if candidate_success(selected_candidate):
        decision = "R_DROP_SUCCESS"
    elif candidate_weak(selected_candidate):
        decision = "R_DROP_WEAK_GAIN"
    else:
        decision = "R_DROP_NO_GAIN"
    v1_1_status = {
        "triggered": trigger_v1_1,
        "trigger_reason": trigger_reason,
        "replaced_v1": replaced_v1,
        "replacement_reason": replacement_reason,
    }
    final_selected = {
        "variant": selected_variant,
        "candidate": selected_candidate,
    }
    return write_final_artifacts(
        output_root,
        variants,
        rows_by_variant,
        a012_info,
        v1_1_status,
        final_selected,
        decision,
        formal_started,
    )


def verify_branch() -> None:
    require(git("cat-file", "-e", f"{BASE_COMMIT}^{{commit}}", check=False).returncode == 0, "Locked A012 source commit missing")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD", check=False).returncode == 0, "Branch is not based on locked A012 implementation")
    branch = git("branch", "--show-current").stdout.strip()
    require(branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-"), f"Wrong branch: {branch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_branch()
    require(args.device == "cuda:0", "R-Drop A012 requires cuda:0")
    output_root = args.output_root.resolve()
    require(output_root == (ROOT / OUTPUT_REL).resolve(), "Output path is locked")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_config.json", experiment_config())
    if args.stage == "prepare":
        print(f"PREPARED A012 rank={A012_SPEC['rank']} rdrop_lambda={RDROP_LAMBDA_V1} lawa_window={LAWA_WINDOW}", flush=True)
        return
    validate_source_commit(output_root)
    context = load_context(args.device)
    if args.stage == "smoke":
        report = run_smoke(context, output_root)
        print(f"SMOKE passed={report['passed']} updates={report['optimizer_steps']} next_epoch={report['checkpoint']['continuation_next_epoch']}", flush=True)
    else:
        summary = run_formal(context, output_root)
        print(f"{summary['decision']} selected={summary['selected']['variant']}/{summary['selected']['candidate']['name']} correct={summary['selected']['candidate']['metrics']['correct']}/598", flush=True)


if __name__ == "__main__":
    main()
