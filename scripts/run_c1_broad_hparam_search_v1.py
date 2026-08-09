"""Run the locked C1 Broad Hyperparameter Search v1 experiment.

Stage A evaluates 24 deterministic search configurations at adapter rank 8.
Stage B evaluates ranks 4/12/16 for the three protected Stage A leaders.
The tracked C1 architecture, data, folds, objective internals, evaluation and
EMA-disabled protocol are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Loss import criterion_query_pool_no_orth
from Model.cme_dual_branch import CMEDualBranchModel
from Utils import CustomCosineAnnealingLR, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "c1_broad_hparam_search_v1"
RUNNER_REL = Path("scripts/run_c1_broad_hparam_search_v1.py")
OUTPUT_REL = Path("experiments/c1_broad_hparam_search_v1")
BASE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
BRANCH_PREFIX = "experiment/c1-broad-hparam-search-v1"
C1_OOF_REL = Path("experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv")
CLASS_NAMES = ("AD", "CN", "SMCI")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SEARCH_SEED = 20260808
STAGE_A_TRIALS = 24
STAGE_B_RANKS = (4, 12, 16)
BACC_FLOOR = 0.9113359
EXPECTED_PARAMETERS = {4: 858_339, 8: 862_971, 12: 867_603, 16: 872_235}
EXPECTED_DROPOUT_COUNT = 10
SEARCH_SPACE = {
    "base_lr": {"distribution": "log_uniform", "low": 0.008, "high": 0.013},
    "base_weight_decay": {"distribution": "log_uniform", "low": 0.0003, "high": 0.0012},
    "lambda_aux": [0.25, 0.50, 0.75, 1.00],
    "adapter_lr_multiplier": [0.5, 1.0, 1.5, 2.0],
    "dropout_multiplier": [0.90, 1.00, 1.10],
}
C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
REFERENCE_RESULTS = {
    "C1": C1,
    "PC-BBF v1": {"correct": 561, "acc": 0.9381270903, "macro_f1": 0.9266945217, "bacc": 0.9152139721, "macro_auc": 0.9701189478, "weighted_f1": 0.9378308464},
    "T1": {"correct": 550, "acc": 0.9197324415, "macro_f1": 0.9029118810, "bacc": 0.8926873891, "macro_auc": 0.9546116855, "weighted_f1": 0.9194786062},
    "T2": {"correct": 552, "acc": 0.9230769231, "macro_f1": 0.9005424037, "bacc": 0.8930679449, "macro_auc": 0.9640011101, "weighted_f1": 0.9224997547},
    "T3": {"correct": 546, "acc": 0.9130434783, "macro_f1": 0.9006004316, "bacc": 0.8824118237, "macro_auc": 0.9582175308, "weighted_f1": 0.9124829270},
}


class InvariantError(RuntimeError):
    """A common runner/protocol/artifact error that must stop the suite."""


class TrialFailure(RuntimeError):
    """A numerical/runtime failure isolated to one trial."""


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args], check=check, text=True, encoding="utf-8", errors="replace", capture_output=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default).encode("utf-8")).hexdigest()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def module_gradient_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(sum(values)).cpu()) if values else 0.0


class RatioPreservingCustomCosineAnnealingLR(CustomCosineAnnealingLR):
    """Historical hold+cosine scheduler extended safely to two LR groups."""

    def __init__(self, optimizer, T_max: int, eta_min: float, adapter_multiplier: float, last_epoch: int = -1):
        require(len(optimizer.param_groups) == 2, "Ratio scheduler requires two groups")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.hold_epoch = 20
        self.adapter_multiplier = float(adapter_multiplier)
        self.initial_lrs_locked = [float(group["lr"]) for group in optimizer.param_groups]
        require(abs(self.initial_lrs_locked[1] - self.initial_lrs_locked[0] * self.adapter_multiplier) <= 1e-14, "Initial LR ratio changed")
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch)
        self._assert_ratio()

    def get_lr(self):
        if self.last_epoch < self.hold_epoch:
            base_lr = self.initial_lrs_locked[0]
        else:
            t_cur = self.last_epoch - self.hold_epoch
            base_lr = self.eta_min + (self.initial_lrs_locked[0] - self.eta_min) * (1.0 + math.cos(math.pi * t_cur / (self.T_max - self.hold_epoch))) / 2.0
        return [base_lr, base_lr * self.adapter_multiplier]

    def _assert_ratio(self) -> None:
        base_lr, adapter_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
        require(abs(adapter_lr - base_lr * self.adapter_multiplier) <= max(1e-14, abs(adapter_lr) * 1e-12), "Scheduler lost adapter/base LR ratio")

    def step(self, epoch=None):
        super().step(epoch)
        self._assert_ratio()


def generate_stage_a_specs() -> list[dict]:
    rng = random.Random(SEARCH_SEED)
    specs, seen = [], set()
    while len(specs) < STAGE_A_TRIALS:
        lr = math.exp(rng.uniform(math.log(SEARCH_SPACE["base_lr"]["low"]), math.log(SEARCH_SPACE["base_lr"]["high"])))
        wd = math.exp(rng.uniform(math.log(SEARCH_SPACE["base_weight_decay"]["low"]), math.log(SEARCH_SPACE["base_weight_decay"]["high"])))
        values = (
            lr,
            wd,
            rng.choice(SEARCH_SPACE["lambda_aux"]),
            rng.choice(SEARCH_SPACE["adapter_lr_multiplier"]),
            rng.choice(SEARCH_SPACE["dropout_multiplier"]),
        )
        if values in seen:
            continue
        seen.add(values)
        index = len(specs)
        specs.append({
            "trial_id": f"A{index:03d}", "stage": "A", "rank": 8,
            "base_lr": values[0], "base_weight_decay": values[1],
            "lambda_aux": values[2], "adapter_lr_multiplier": values[3],
            "dropout_multiplier": values[4], "generation_index": index,
        })
    require(len({payload_sha256(spec) for spec in specs}) == STAGE_A_TRIALS, "Fallback configs are not unique")
    return specs


def manifest_core() -> dict:
    specs = generate_stage_a_specs()
    optuna_available = importlib.util.find_spec("optuna") is not None
    require(not optuna_available, "Optuna became available; regenerate this locked source using the required TPE path before training")
    return {
        "experiment": EXPERIMENT,
        "base_commit": BASE_COMMIT,
        "protocol": {"dataset": "TADPOLE", "task": "AD_CN_SMCI", "folds": list(FOLDS), "seed_per_fold": SEED, "epochs_per_fold": EPOCHS, "full_batch_transductive": True, "single_model": True, "ensemble": False, "orthogonality": False, "graph": False, "optimizer": "Adam", "scheduler": "CustomCosineAnnealingLR(T_max=400), ratio-preserving two-group compatibility", "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"], "ema": False},
        "sampler": {"requested": "Optuna TPESampler(seed=20260808,n_startup_trials=8)", "optuna_detected": optuna_available, "effective": "python_stdlib_deterministic_fallback", "reason": "Optuna unavailable; dependency installation forbidden", "seed": SEARCH_SEED},
        "search_space": SEARCH_SPACE,
        "stage_a": {"n_trials": STAGE_A_TRIALS, "rank": 8, "configs": specs},
        "stage_b": {"parent_slots": 3, "ranks": list(STAGE_B_RANKS), "n_trials": 9, "selection": "protected Stage A top 3 by Correct>AUC>Macro-F1; fill from overall ranking only if fewer than 3 protected", "protected_condition": {"AD_CN_errors": 0, "BACC_minimum": BACC_FLOOR}, "deferred_slots": [{"parent_slot": parent, "rank": rank} for parent in range(1, 4) for rank in STAGE_B_RANKS]},
        "objective_scalar": "correct + 1e-3 * macro_auc + 1e-6 * macro_f1",
        "ranking": ["Correct", "Probability Macro-AUC", "Macro-F1"],
    }


def create_or_validate_manifest(output_root: Path) -> dict:
    path = output_root / "search_manifest.json"
    core = manifest_core()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("manifest_core") == core, "Existing search manifest changed")
        require(payload.get("runner_sha256") == file_sha256(ROOT / RUNNER_REL), "Manifest runner hash changed")
        return payload
    payload = {
        "manifest_version": 1,
        "created_before_any_training": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "creation_head": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / RUNNER_REL),
        "manifest_core_sha256": payload_sha256(core),
        "manifest_core": core,
    }
    write_json(path, payload)
    return payload


def fold_manifest(context: dict) -> dict:
    labels = context["dataset_data"]["Label"].detach().cpu().numpy().astype(int)
    indices = np.asarray(context["dataset_dict"]["Index"], dtype=int)
    folds = []
    for fold, (train_mask, test_mask) in enumerate(context["dataset_data"]["Mask"]):
        train = train_mask.detach().cpu().numpy().astype(bool)
        test = test_mask.detach().cpu().numpy().astype(bool)
        folds.append({"fold": fold, "train_subject_indices": indices[train].tolist(), "test_subject_indices": indices[test].tolist(), "test_truth": labels[test].tolist()})
    return {"folds": folds, "sha256": payload_sha256(folds)}


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "Broad search requires cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "Historical device changed")
    config = context["config"]
    require(tuple(context["dataset_dict"]["Class_Names"]) == CLASS_NAMES, "Class order changed")
    require(int(config.epochs) == EPOCHS and int(config.T_max) == EPOCHS, "Epoch protocol changed")
    require(abs(float(config.lr) - 0.01) < 1e-12 and abs(float(config.weight_decay) - 0.0005) < 1e-12, "Tracked C1 optimizer defaults changed")
    require(config.use_ema is False, "EMA must remain disabled")
    require(abs(float(config.Lr_Min) - 0.0001) < 1e-12, "Historical eta_min changed")
    require(abs(float(config.logit_adjust_tau) - 0.75) < 1e-12, "Logit adjustment changed")
    c1_path = ROOT / C1_OOF_REL
    require(c1_path.is_file(), f"C1 OOF missing: {c1_path}")
    c1_rows = read_csv(c1_path)
    require(len(c1_rows) == 598, "C1 OOF row count changed")
    metrics = cme.metrics_from_rows(c1_rows)
    require(metrics["correct"] == C1["correct"] and metrics["confusion_matrix"] == C1["confusion_matrix"], "C1 anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    context["c1_rows"] = c1_rows
    context["c1_by_subject"] = {int(row["subject_index"]): row for row in c1_rows}
    context["fold_manifest"] = fold_manifest(context)
    require(len(context["c1_by_subject"]) == 598, "C1 subjects duplicated")
    return context


def build_model(context: dict, rank: int) -> CMEDualBranchModel:
    require(rank in EXPECTED_PARAMETERS, f"Unsupported adapter rank: {rank}")
    config = context["config"]
    SET_Random(SEED)
    model = CMEDualBranchModel(
        context["dataset_dict"], Herter_Graph=None, Hidden_size=config.Hidden_size,
        Drop_rate=config.Drop_rate, K=config.ChebGCN_K, num_layers=config.num_layers,
        num_heads=config.num_heads, input_noise_std=config.input_noise_std,
        drop_path=config.drop_path, graph_head=config.Graph_head,
        graph_layers=config.graph_layers, graph_heads=config.graph_heads,
        graph_beta=config.graph_beta, graph_k_order=config.graph_k_order,
        graph_alpha=config.graph_alpha, graph_kernel=config.graph_kernel,
        graph_use_graph=False, graph_dropout=config.graph_dropout,
        graph_hidden=config.graph_hidden, global_word_emb=config.global_word_emb,
        semantic_branch="both", semantic_fusion="add",
        category_branch_variant="original", query_pool_variant="independent",
        category_branch_fusion="concat", adj_mode="none", label_graph_alpha=0.0,
        label_graph_topk=0, label_graph_reg_lambda=0.0, cme_arm="c1",
        adapter_rank=rank, router_hidden=16, modality_embedding_dim=8,
    ).to(context["device"])
    require(parameter_count(model) == EXPECTED_PARAMETERS[rank], f"rank {rank} parameter count changed")
    require(len(model.private_adapters) == 6, "Expected exactly six private adapters")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    return model


def apply_dropout_multiplier(model: torch.nn.Module, multiplier: float) -> list[dict]:
    records = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            original = float(module.p)
            final = min(0.8, original * float(multiplier))
            module.p = final
            records.append({"module": name, "original_p": original, "final_p": final})
    require(len(records) == EXPECTED_DROPOUT_COUNT, f"Expected 10 nn.Dropout modules, found {len(records)}")
    originals = sorted(round(row["original_p"], 12) for row in records)
    require(originals == sorted([0.335] * 9 + [0.67]), f"C1 Dropout profile changed: {originals}")
    return records


def split_parameter_groups(model: CMEDualBranchModel) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    adapter_named = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
    base_named = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
    require(len(adapter_named) == 24, f"Six private adapters should expose 24 tensors, found {len(adapter_named)}")
    all_ids, base_ids, adapter_ids = {id(p) for _, p in named}, {id(p) for _, p in base_named}, {id(p) for _, p in adapter_named}
    require(not (base_ids & adapter_ids), "Optimizer groups overlap")
    require(base_ids | adapter_ids == all_ids, "Optimizer groups do not cover all trainable parameters")
    require(sum(len(list(adapter.parameters())) for adapter in model.private_adapters) == 24, "Adapter identity partition changed")
    audit = {"base_parameter_tensors": len(base_named), "adapter_parameter_tensors": len(adapter_named), "adapter_module_count": len(model.private_adapters), "base_parameter_names": [name for name, _ in base_named], "adapter_parameter_names": [name for name, _ in adapter_named]}
    return [p for _, p in base_named], [p for _, p in adapter_named], audit


def make_training_objects(context: dict, spec: dict):
    model = build_model(context, int(spec["rank"]))
    dropout = apply_dropout_multiplier(model, float(spec["dropout_multiplier"]))
    criterion = criterion_query_pool_no_orth(context["dataset_dict"], context["device"], label_smoothing=0.05)
    base, adapter, groups = split_parameter_groups(model)
    optimizer = torch.optim.Adam([
        {"params": base, "lr": float(spec["base_lr"]), "weight_decay": float(spec["base_weight_decay"]), "group_name": "base"},
        {"params": adapter, "lr": float(spec["base_lr"]) * float(spec["adapter_lr_multiplier"]), "weight_decay": float(spec["base_weight_decay"]), "group_name": "private_residual_adapters"},
    ])
    scheduler = RatioPreservingCustomCosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min), adapter_multiplier=float(spec["adapter_lr_multiplier"]))
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler, dropout, groups


def loss_components(criterion, output, labels, mask, auxiliary_outputs, lambda_aux: float) -> dict[str, torch.Tensor]:
    require(len(auxiliary_outputs) == 3 and len(criterion.aux_losses) == 3, "OVR head count changed")
    main = criterion.main_loss(output[mask], labels[mask])
    targets = F.one_hot(labels, num_classes=criterion.Label_num).transpose(0, 1)
    auxiliary = output.new_zeros(())
    individual = []
    for label_index, (loss_fn, aux_output) in enumerate(zip(criterion.aux_losses, auxiliary_outputs)):
        value = loss_fn(aux_output[mask], targets[label_index][mask])
        auxiliary = auxiliary + value
        individual.append(value)
    total = main + float(lambda_aux) * auxiliary
    return {"main": main, "auxiliary": auxiliary, "total": total, "aux_0": individual[0], "aux_1": individual[1], "aux_2": individual[2]}


def runtime_config(context: dict, spec: dict) -> dict:
    model, criterion, optimizer, scheduler, dropout, groups = make_training_objects(context, spec)
    payload = {
        "experiment": EXPERIMENT, "trial": spec, "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / RUNNER_REL), "base_commit": BASE_COMMIT,
        "fold_manifest_sha256": context["fold_manifest"]["sha256"], "folds": list(FOLDS),
        "seed_per_fold": SEED, "epochs_per_fold": EPOCHS, "device": "cuda:0",
        "parameter_count": parameter_count(model), "optimizer": "Adam(two disjoint exhaustive groups)",
        "scheduler": "CustomCosineAnnealingLR hold20+cosine; ratio-preserving compatible subclass", "scheduler_t_max": EPOCHS,
        "scheduler_eta_min_base": float(context["config"].Lr_Min), "scheduler_eta_min_adapter": float(context["config"].Lr_Min) * float(spec["adapter_lr_multiplier"]),
        "loss": "main weighted CE + lambda_aux * (three original OVR weighted CEs)", "label_smoothing": 0.05,
        "logit_adjust_tau": float(context["config"].logit_adjust_tau), "gradient_clip": float(context["config"].grad_clip),
        "dropout_modules": dropout, "dropout_module_count": len(dropout), "dropout_modified_count": sum(abs(row["final_p"] - row["original_p"]) > 0 for row in dropout),
        "parameter_group_audit": groups, "full_batch_transductive": True, "single_model": True, "ensemble": False,
        "orthogonality": False, "graph": False, "ema": False, "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
    }
    payload["canonical_trial_config_sha256"] = payload_sha256(payload)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return payload


def validate_prediction_rows(rows: list[dict], expected_fold: int | None = None) -> None:
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(subjects) == len(set(subjects)), "Prediction rows contain duplicate subjects")
    for row in rows:
        if expected_fold is not None:
            require(int(row["fold"]) == expected_fold, "Prediction row has wrong fold")
        truth, prediction = int(row["truth"]), int(row["prediction"])
        require(0 <= truth < 3 and 0 <= prediction < 3, "Invalid class index")
        probability = np.asarray([float(row[f"probability_{name}"]) for name in CLASS_NAMES])
        require(bool(np.isfinite(probability).all()), "Non-finite OOF probability")
        require(abs(float(probability.sum()) - 1.0) <= 1e-5, "OOF probability does not sum to one")
        require(int(probability.argmax()) == prediction, "OOF prediction is not probability argmax")


def validate_oof(rows: list[dict]) -> list[dict]:
    require(len(rows) == 598, "OOF row count changed")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == 598 and set(subjects) == set(range(598)), "OOF subject coverage changed")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "OOF fold coverage changed")
    validate_prediction_rows(rows)
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def paired_comparison(rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"C1 missing subject {subject}")
        reference = reference_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth and int(reference["fold"]) == int(row["fold"]), "C1/candidate OOF alignment changed")
        old, new = int(reference["prediction"]), int(row["prediction"])
        if old != truth and new == truth:
            repairs.append(subject)
        if old == truth and new != truth:
            damages.append(subject)
        if old != new:
            changed.append(subject)
    return {"repairs": len(repairs), "damages": len(damages), "net_repairs": len(repairs) - len(damages), "changed_predictions": len(changed), "repair_subject_indices": repairs, "damage_subject_indices": damages, "changed_subject_indices": changed}


def boundary_errors(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {"AD_SMCI": int(matrix[0, 2] + matrix[2, 0]), "CN_SMCI": int(matrix[1, 2] + matrix[2, 1]), "AD_CN": int(matrix[0, 1] + matrix[1, 0])}


def ranking_key(report: dict) -> tuple[float, float, float]:
    metrics = report["metrics"]
    return int(metrics["correct"]), float(metrics["macro_auc"]), float(metrics["macro_f1"])


def is_protected(report: dict) -> bool:
    return report["boundary_errors"]["AD_CN"] == 0 and float(report["metrics"]["bacc"]) >= BACC_FLOOR


def fold_expected_subjects(context: dict, fold: int) -> dict[int, int]:
    item = context["fold_manifest"]["folds"][fold]
    return dict(zip(item["test_subject_indices"], item["test_truth"]))


def smoke_probe_roundtrip(context: dict, fold: int, rows: list[dict], smoke_root: Path, probe_config: dict) -> dict:
    probe = smoke_root / "recovery_probe" / f"fold_{fold:02d}"
    require(not probe.exists(), f"Refusing to overwrite smoke recovery probe: {probe}")
    staging = probe.parent / f".fold_{fold:02d}_in_progress"
    staging.mkdir(parents=True)
    write_json(staging / "config.json", probe_config)
    write_csv(staging / "oof_predictions.csv", rows)
    marker = {"complete": True, "fold": fold, "source_commit": probe_config["source_commit"], "canonical_trial_config_sha256": probe_config["canonical_trial_config_sha256"], "fold_manifest_sha256": probe_config["fold_manifest_sha256"]}
    write_json(staging / "complete.json", marker)
    staging.rename(probe)
    require(json.loads((probe / "config.json").read_text(encoding="utf-8")) == probe_config, "Smoke recovery config roundtrip failed")
    require(json.loads((probe / "complete.json").read_text(encoding="utf-8")) == marker, "Smoke completion-marker roundtrip failed")
    loaded = read_csv(probe / "oof_predictions.csv")
    validate_prediction_rows(loaded, expected_fold=fold)
    require({int(row["subject_index"]): int(row["truth"]) for row in loaded} == fold_expected_subjects(context, fold), "Smoke recovery OOF manifest mismatch")
    return {"path": str(probe.relative_to(ROOT)).replace("\\", "/"), "rows": len(loaded), "strict_config_marker_oof_readback": True}


def scheduler_preview(context: dict, spec: dict) -> list[dict]:
    first = torch.nn.Parameter(torch.zeros((), device=context["device"]))
    second = torch.nn.Parameter(torch.zeros((), device=context["device"]))
    multiplier = float(spec["adapter_lr_multiplier"])
    base_initial = float(spec["base_lr"])
    optimizer = torch.optim.Adam([
        {"params": [first], "lr": base_initial},
        {"params": [second], "lr": base_initial * multiplier},
    ])
    scheduler = RatioPreservingCustomCosineAnnealingLR(optimizer, EPOCHS, float(context["config"].Lr_Min), multiplier)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.step()
        scheduler.step()
        if epoch in {1, 19, 20, 21, EPOCHS}:
            base_lr, adapter_lr = [float(group["lr"]) for group in optimizer.param_groups]
            rows.append({"epoch": epoch, "base_lr": base_lr, "adapter_lr": adapter_lr, "ratio": adapter_lr / base_lr})
    require(rows[2]["base_lr"] == base_initial, "Scheduler hold through epoch20 changed")
    require(rows[3]["base_lr"] < base_initial, "Scheduler did not begin cosine decay after epoch20")
    require(abs(rows[-1]["base_lr"] - float(context["config"].Lr_Min)) <= 1e-12, "Scheduler base eta_min changed")
    require(abs(rows[-1]["adapter_lr"] - float(context["config"].Lr_Min) * multiplier) <= 1e-12, "Scheduler adapter eta_min scaling changed")
    require(all(abs(row["ratio"] - multiplier) <= 1e-12 for row in rows), "Scheduler preview lost LR ratio")
    return rows


def run_smoke(context: dict, output_root: Path, manifest: dict) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    specs = manifest["manifest_core"]["stage_a"]["configs"]
    eligible = [spec for spec in specs if float(spec["lambda_aux"]) != 1.0 and float(spec["adapter_lr_multiplier"]) != 1.0 and float(spec["dropout_multiplier"]) != 1.0]
    require(bool(eligible), "Manifest lacks a non-default smoke configuration")
    spec = eligible[0]
    config = runtime_config(context, spec)
    started = time.perf_counter()
    model, criterion, optimizer, scheduler, dropout, groups = make_training_objects(context, spec)
    require(config["dropout_modules"] == dropout and config["parameter_group_audit"] == groups, "Smoke runtime profile changed")
    require(config["dropout_modified_count"] == EXPECTED_DROPOUT_COUNT, "Smoke dropout multiplier did not modify all existing Dropout modules")
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    losses, group_lrs, adapter_gradients = [], [], []
    historical_equivalence_delta = None
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            raw, branches, auxiliary = model(features)
            components = loss_components(criterion, raw, labels, train_mask, auxiliary, float(spec["lambda_aux"]))
        except RuntimeError as exc:
            raise TrialFailure(f"Smoke forward failed: {exc}") from exc
        loss = components["total"]
        require(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(value)) for value in components.values()), f"Smoke epoch{epoch}: non-finite loss")
        formula = components["main"] + float(spec["lambda_aux"]) * components["auxiliary"]
        require(float((loss - formula).abs().detach().cpu()) <= 1e-7, "lambda_aux loss formula mismatch")
        if epoch == 1:
            historical = criterion(raw, labels, train_mask, branches, auxiliary)
            lambda_one = loss_components(criterion, raw, labels, train_mask, auxiliary, 1.0)["total"]
            historical_equivalence_delta = float((historical - lambda_one).abs().detach().cpu())
            require(historical_equivalence_delta <= 1e-7, "lambda_aux=1 is not historically equivalent")
            require(float(components["auxiliary"].detach().cpu()) > 0.0, "Auxiliary loss is zero")
        loss.backward()
        require(gradients_finite(model), f"Smoke epoch{epoch}: non-finite gradient")
        norms = [module_gradient_norm(adapter) for adapter in model.private_adapters]
        require(all(math.isfinite(value) and value > 0.0 for value in norms), f"Smoke epoch{epoch}: six adapters lack finite nonzero gradients")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        base_lr, adapter_lr = [float(group["lr"]) for group in optimizer.param_groups]
        expected_adapter = base_lr * float(spec["adapter_lr_multiplier"])
        require(abs(adapter_lr - expected_adapter) <= max(1e-14, abs(expected_adapter) * 1e-12), "Smoke scheduler lost LR ratio")
        losses.append({name: float(value.detach().cpu()) for name, value in components.items()})
        group_lrs.append({"epoch": epoch, "base": base_lr, "adapter": adapter_lr, "ratio": adapter_lr / base_lr})
        adapter_gradients.append(norms)
    model.eval()
    with torch.no_grad():
        final_raw, _, _ = model(features)
    rows = cme.prediction_rows(0, final_raw, labels, test_mask, context["dataset_dict"], context["config"])
    probe = smoke_probe_roundtrip(context, 0, rows, smoke_root, config)
    schedule_preview = scheduler_preview(context, spec)
    report = {
        "passed": True, "fold": 0, "epochs": 3, "trial_id": spec["trial_id"], "trial": spec,
        "manifest_created_before_training": manifest["created_before_any_training"], "manifest_core_sha256": manifest["manifest_core_sha256"],
        "losses": losses, "lambda_one_historical_max_abs_delta": historical_equivalence_delta,
        "optimizer_group_audit": groups, "group_lrs": group_lrs, "scheduler_preview": schedule_preview, "adapter_gradient_norms": adapter_gradients,
        "dropout_modules": dropout, "dropout_modified_count": config["dropout_modified_count"],
        "recovery_probe": probe, "parameter_count": parameter_count(model), "ema": False,
        "runner_sha256": file_sha256(ROOT / RUNNER_REL), "run_head_at_smoke": git("rev-parse", "HEAD").stdout.strip(),
        "wall_seconds": float(time.perf_counter() - started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_c1_broad_hparam_search_v1.py smoke --device cuda:0',
    }
    write_json(smoke_root / "smoke_config.json", config)
    write_json(smoke_root / "smoke_report.json", report)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return report


def fold_lock(spec: dict, trial_config: dict, fold: int) -> dict:
    return {"trial_id": spec["trial_id"], "trial": spec, "fold": fold, "source_commit": trial_config["source_commit"], "canonical_trial_config_sha256": trial_config["canonical_trial_config_sha256"], "fold_manifest_sha256": trial_config["fold_manifest_sha256"]}


def completed_fold(context: dict, spec: dict, trial_config: dict, fold: int, final_dir: Path):
    if not final_dir.exists():
        return None
    required = [final_dir / name for name in ("config.json", "summary.json", "oof_predictions.csv", "complete.json")]
    require(all(path.is_file() for path in required), f"{spec['trial_id']} fold{fold}: incomplete completed artifact")
    lock = fold_lock(spec, trial_config, fold)
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == lock, f"{spec['trial_id']} fold{fold}: source/config changed")
    marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
    expected_marker = {"complete": True, "trial_id": spec["trial_id"], "fold": fold, "source_commit": trial_config["source_commit"], "canonical_trial_config_sha256": trial_config["canonical_trial_config_sha256"], "fold_manifest_sha256": trial_config["fold_manifest_sha256"]}
    require(marker == expected_marker, f"{spec['trial_id']} fold{fold}: completion marker changed")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(final_dir / "oof_predictions.csv")
    validate_prediction_rows(rows, expected_fold=fold)
    require({int(row["subject_index"]): int(row["truth"]) for row in rows} == fold_expected_subjects(context, fold), f"{spec['trial_id']} fold{fold}: fold manifest mismatch")
    metrics = cme.metrics_from_rows(rows)
    require(metrics == summary["best_metrics"], f"{spec['trial_id']} fold{fold}: saved metrics changed")
    require(summary["trial_config_lock"] == lock and summary["best_epoch"] in range(1, EPOCHS + 1), f"{spec['trial_id']} fold{fold}: summary changed")
    print(f"RESUME {spec['trial_id']} fold={fold} best_epoch={summary['best_epoch']} correct={metrics['correct']}", flush=True)
    return summary, rows


def train_fold(context: dict, spec: dict, trial_config: dict, fold: int, trial_root: Path):
    final_dir = trial_root / f"fold_{fold:02d}"
    resumed = completed_fold(context, spec, trial_config, fold, final_dir)
    if resumed is not None:
        return resumed
    staging = trial_root / f".fold_{fold:02d}_in_progress"
    lock = fold_lock(spec, trial_config, fold)
    if staging.exists():
        require((staging / "config.json").is_file(), f"{spec['trial_id']} fold{fold}: in-progress config missing")
        require(json.loads((staging / "config.json").read_text(encoding="utf-8")) == lock, f"{spec['trial_id']} fold{fold}: in-progress source/config changed")
        resolved = staging.resolve()
        require(resolved.parent == trial_root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe in-progress restart target")
        shutil.rmtree(resolved)
        print(f"RESTART_INCOMPLETE {spec['trial_id']} fold={fold}", flush=True)
    staging.mkdir(parents=True)
    write_json(staging / "config.json", lock)
    model, criterion, optimizer, scheduler, dropout, groups = make_training_objects(context, spec)
    require(parameter_count(model) == trial_config["parameter_count"] and dropout == trial_config["dropout_modules"] and groups == trial_config["parameter_group_audit"], "Fold runtime profile differs from trial config")
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    best, best_state = None, None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            raw, branches, auxiliary = model(features)
            components = loss_components(criterion, raw, labels, train_mask, auxiliary, float(spec["lambda_aux"]))
            loss = components["total"]
            if not bool(torch.isfinite(loss)):
                raise TrialFailure(f"{spec['trial_id']} fold{fold} epoch{epoch}: non-finite loss")
            loss.backward()
        except TrialFailure:
            raise
        except InvariantError:
            raise
        except (RuntimeError, FloatingPointError) as exc:
            raise TrialFailure(f"{spec['trial_id']} fold{fold} epoch{epoch}: {type(exc).__name__}: {exc}") from exc
        if not gradients_finite(model):
            raise TrialFailure(f"{spec['trial_id']} fold{fold} epoch{epoch}: non-finite gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        base_lr, adapter_lr = [float(group["lr"]) for group in optimizer.param_groups]
        require(abs(adapter_lr - base_lr * float(spec["adapter_lr_multiplier"])) <= max(1e-14, abs(adapter_lr) * 1e-12), "LR group ratio changed")
        model.eval()
        with torch.no_grad():
            evaluated, _, _ = model(features)
            metrics, selection = cme.selection_metrics(evaluated, labels, test_mask, context["dataset_dict"]["Label_Weight"], float(context["config"].logit_adjust_tau))
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = clone_cpu_state(model)
    require(best is not None and best_state is not None, f"{spec['trial_id']} fold{fold}: best state missing")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _ = model(features)
    rows = cme.prediction_rows(fold, best_raw, labels, test_mask, context["dataset_dict"], context["config"])
    validate_prediction_rows(rows, expected_fold=fold)
    require(cme.metrics_from_rows(rows) == best["metrics"], f"{spec['trial_id']} fold{fold}: best OOF readback changed")
    summary = {"trial_id": spec["trial_id"], "fold": fold, "train_size": int(train_mask.sum()), "test_size": int(test_mask.sum()), "best_epoch": best["epoch"], "best_metrics": best["metrics"], "parameter_count": parameter_count(model), "last_group_lrs": {"base": base_lr, "adapter": adapter_lr}, "elapsed_seconds": float(time.perf_counter() - started), "trial_config_lock": lock}
    marker = {"complete": True, "trial_id": spec["trial_id"], "fold": fold, "source_commit": trial_config["source_commit"], "canonical_trial_config_sha256": trial_config["canonical_trial_config_sha256"], "fold_manifest_sha256": trial_config["fold_manifest_sha256"]}
    write_json(staging / "summary.json", summary)
    write_csv(staging / "oof_predictions.csv", rows)
    write_json(staging / "complete.json", marker)
    staging.rename(final_dir)
    print(f"{spec['trial_id']} fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']} ACC={best['metrics']['acc']:.7f} AUC={best['metrics']['macro_auc']:.7f} F1={best['metrics']['macro_f1']:.7f}", flush=True)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def trial_directory(output_root: Path, spec: dict) -> Path:
    stage = "stage_a" if spec["stage"] == "A" else "stage_b"
    return output_root / stage / f"trial_{spec['trial_id']}"


def run_trial(context: dict, spec: dict, output_root: Path) -> tuple[dict, list[dict]]:
    root = trial_directory(output_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    config = runtime_config(context, spec)
    config_path = root / "config.json"
    if config_path.is_file():
        require(json.loads(config_path.read_text(encoding="utf-8")) == config, f"{spec['trial_id']}: source/trial config changed")
    else:
        write_json(config_path, config)
    report_path, complete_path = root / "report.json", root / "complete.json"
    if report_path.is_file() and complete_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        require(marker == {"complete": True, "trial_id": spec["trial_id"], "source_commit": config["source_commit"], "canonical_trial_config_sha256": config["canonical_trial_config_sha256"], "fold_manifest_sha256": config["fold_manifest_sha256"]}, f"{spec['trial_id']}: trial marker changed")
        require(report["trial_config"] == config, f"{spec['trial_id']}: completed trial config changed")
        pooled = validate_oof(read_csv(root / "oof_predictions.csv"))
        require(cme.metrics_from_rows(pooled) == report["metrics"], f"{spec['trial_id']}: pooled OOF changed")
        for fold in FOLDS:
            require(completed_fold(context, spec, config, fold, root / f"fold_{fold:02d}") is not None, f"{spec['trial_id']}: completed fold missing")
        return report, pooled
    require(not complete_path.exists(), f"{spec['trial_id']}: complete marker without report")
    summaries, rows = [], []
    for fold in FOLDS:
        summary, fold_rows = train_fold(context, spec, config, fold, root)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = validate_oof(rows)
    metrics = cme.metrics_from_rows(rows)
    paired = paired_comparison(rows, context["c1_by_subject"])
    require(paired["net_repairs"] == metrics["correct"] - C1["correct"], f"{spec['trial_id']}: paired comparison mismatch")
    boundaries = boundary_errors(metrics)
    predicted = {CLASS_NAMES[index]: sum(int(row["prediction"]) == index for row in rows) for index in range(3)}
    fold_acc = np.asarray([summary["best_metrics"]["acc"] for summary in summaries], dtype=float)
    report = {
        "status": "complete", "trial_id": spec["trial_id"], "stage": spec["stage"], "trial": spec,
        "metrics": metrics, "objective": int(metrics["correct"]) + 1e-3 * float(metrics["macro_auc"]) + 1e-6 * float(metrics["macro_f1"]),
        "protected_for_stage_b": boundaries["AD_CN"] == 0 and metrics["bacc"] >= BACC_FLOOR,
        "protection_risks": {"ad_cn_error": boundaries["AD_CN"] > 0, "bacc_below_0p9113359": metrics["bacc"] < BACC_FLOOR},
        "comparison_vs_c1": paired, "boundary_errors": boundaries, "predicted_class_counts": predicted,
        "ten_fold_acc_mean": float(fold_acc.mean()), "ten_fold_acc_sample_std": float(fold_acc.std(ddof=1)),
        "parameter_count": config["parameter_count"], "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "folds": [{"fold": summary["fold"], "best_epoch": summary["best_epoch"], "correct": summary["best_metrics"]["correct"], "acc": summary["best_metrics"]["acc"]} for summary in summaries],
        "trial_config": config,
    }
    write_csv(root / "oof_predictions.csv", rows)
    write_json(report_path, report)
    marker = {"complete": True, "trial_id": spec["trial_id"], "source_commit": config["source_commit"], "canonical_trial_config_sha256": config["canonical_trial_config_sha256"], "fold_manifest_sha256": config["fold_manifest_sha256"]}
    write_json(complete_path, marker)
    return report, rows


def record_trial_failure(spec: dict, output_root: Path, exc: Exception) -> dict:
    path = trial_directory(output_root, spec) / "failure.json"
    history = []
    if path.is_file():
        history = json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
    attempt = {"attempt": len(history) + 1, "exception_type": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc(), "source_commit": git("rev-parse", "HEAD").stdout.strip(), "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    payload = {"status": "failed", "trial_id": spec["trial_id"], "stage": spec["stage"], "trial": spec, "attempts": history + [attempt], "retry_on_resume": False, "terminal_for_source_and_config": True}
    write_json(path, payload)
    print(f"TRIAL_FAILED {spec['trial_id']}: {type(exc).__name__}: {exc}", flush=True)
    return payload


def select_stage_b_parents(stage_a_reports: list[dict]) -> list[dict]:
    ranked = sorted(stage_a_reports, key=ranking_key, reverse=True)
    require(len(ranked) >= 3, "Fewer than three successful Stage A trials")
    selected = [report for report in ranked if is_protected(report)][:3]
    used = {report["trial_id"] for report in selected}
    if len(selected) < 3:
        selected.extend(report for report in ranked if report["trial_id"] not in used and len(selected) < 3)
    require(len(selected) == 3 and len({report["trial_id"] for report in selected}) == 3, "Stage B parent selection failed")
    return selected


def stage_b_specs(parents: list[dict]) -> list[dict]:
    output = []
    for slot, parent in enumerate(parents, start=1):
        inherited = parent["trial"]
        for rank in STAGE_B_RANKS:
            output.append({
                "trial_id": f"B{slot:02d}R{rank:02d}", "stage": "B", "rank": rank,
                "base_lr": inherited["base_lr"], "base_weight_decay": inherited["base_weight_decay"],
                "lambda_aux": inherited["lambda_aux"], "adapter_lr_multiplier": inherited["adapter_lr_multiplier"],
                "dropout_multiplier": inherited["dropout_multiplier"], "parent_stage_a_trial_id": parent["trial_id"],
                "parent_slot": slot, "parent_was_protected": is_protected(parent),
            })
    require(len(output) == 9 and len({spec["trial_id"] for spec in output}) == 9, "Stage B plan changed")
    return output


def flat_trial(report: dict) -> dict:
    spec = report["trial"]
    row = {
        "trial_id": report["trial_id"], "stage": report["stage"], "status": report["status"],
        "parent_stage_a_trial_id": spec.get("parent_stage_a_trial_id", ""), "rank": spec["rank"],
        "base_lr": spec["base_lr"], "base_weight_decay": spec["base_weight_decay"],
        "lambda_aux": spec["lambda_aux"], "adapter_lr_multiplier": spec["adapter_lr_multiplier"],
        "dropout_multiplier": spec["dropout_multiplier"],
    }
    if report["status"] == "complete":
        metrics = report["metrics"]
        row.update({
            "correct": metrics["correct"], "acc": metrics["acc"], "macro_f1": metrics["macro_f1"],
            "bacc": metrics["bacc"], "macro_auc": metrics["macro_auc"], "weighted_f1": metrics["weighted_f1"],
            "ad_smci_errors": report["boundary_errors"]["AD_SMCI"], "cn_smci_errors": report["boundary_errors"]["CN_SMCI"],
            "ad_cn_errors": report["boundary_errors"]["AD_CN"], "protected": is_protected(report),
            "parameters": report["parameter_count"], "training_seconds": report["training_seconds"],
        })
    else:
        last = report.get("attempts", [{}])[-1]
        row.update({"failure_type": last.get("exception_type", ""), "failure_reason": last.get("reason", "")})
    return row


def metric_delta(candidate: dict, baseline: dict) -> dict:
    return {"correct": int(candidate["correct"] - baseline["correct"]), **{key: float(candidate[key] - baseline[key]) for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")}}


def capacity_sensitivity(reports: list[dict], parents: list[dict]) -> dict:
    by_rank = {}
    parent_ids = {parent["trial_id"] for parent in parents}
    for rank in (4, 8, 12, 16):
        selected = [
            report for report in reports
            if int(report["trial"]["rank"]) == rank
            and (rank != 8 or report["trial_id"] in parent_ids)
        ]
        if selected:
            by_rank[str(rank)] = {
                "n": len(selected), "parameter_count": sorted({report["parameter_count"] for report in selected}),
                "correct_mean": float(np.mean([report["metrics"]["correct"] for report in selected])),
                "correct_max": int(max(report["metrics"]["correct"] for report in selected)),
                "auc_mean": float(np.mean([report["metrics"]["macro_auc"] for report in selected])),
                "bacc_mean": float(np.mean([report["metrics"]["bacc"] for report in selected])),
            }
    parent_curves = {}
    by_id = {report["trial_id"]: report for report in reports}
    for slot, parent in enumerate(parents, start=1):
        points = {"8": {"trial_id": parent["trial_id"], "correct": parent["metrics"]["correct"], "macro_auc": parent["metrics"]["macro_auc"], "bacc": parent["metrics"]["bacc"], "parameters": parent["parameter_count"]}}
        for rank in STAGE_B_RANKS:
            trial_id = f"B{slot:02d}R{rank:02d}"
            if trial_id in by_id:
                report = by_id[trial_id]
                points[str(rank)] = {"trial_id": trial_id, "correct": report["metrics"]["correct"], "macro_auc": report["metrics"]["macro_auc"], "bacc": report["metrics"]["bacc"], "parameters": report["parameter_count"]}
        parent_curves[parent["trial_id"]] = points
    all_stage_a = [report for report in reports if report["stage"] == "A"]
    return {
        "matched_parent_aggregate_by_rank": by_rank,
        "all_stage_a_rank8": {
            "n": len(all_stage_a),
            "correct_mean": float(np.mean([report["metrics"]["correct"] for report in all_stage_a])),
            "correct_max": int(max(report["metrics"]["correct"] for report in all_stage_a)),
            "auc_mean": float(np.mean([report["metrics"]["macro_auc"] for report in all_stage_a])),
            "bacc_mean": float(np.mean([report["metrics"]["bacc"] for report in all_stage_a])),
        },
        "matched_parent_curves": parent_curves,
    }


def parameter_relationships(stage_a: list[dict]) -> dict:
    require(bool(stage_a), "No Stage A reports for sensitivity summary")
    output = {}
    for parameter in ("base_lr", "base_weight_decay", "lambda_aux", "adapter_lr_multiplier", "dropout_multiplier"):
        values = np.asarray([float(report["trial"][parameter]) for report in stage_a], dtype=float)
        entry = {"group_means": {}}
        for value in sorted(set(values.tolist())):
            group = [report for report in stage_a if float(report["trial"][parameter]) == value]
            entry["group_means"][f"{value:.12g}"] = {"n": len(group), "correct_mean": float(np.mean([item["metrics"]["correct"] for item in group])), "auc_mean": float(np.mean([item["metrics"]["macro_auc"] for item in group])), "bacc_mean": float(np.mean([item["metrics"]["bacc"] for item in group]))}
        entry["pearson"] = {}
        for metric in ("correct", "macro_auc", "bacc"):
            targets = np.asarray([float(report["metrics"][metric]) for report in stage_a], dtype=float)
            entry["pearson"][metric] = float(np.corrcoef(values, targets)[0, 1]) if len(set(values)) > 1 and float(targets.std()) > 0 else None
        output[parameter] = entry
    return output


def final_decision(report: dict) -> tuple[str, dict]:
    metrics = report["metrics"]
    secondary = sum(metrics[key] > C1[key] for key in ("macro_f1", "bacc", "macro_auc"))
    checks = {"correct_at_least_563": metrics["correct"] >= 563, "correct_equals_562": metrics["correct"] == 562, "correct_equals_561": metrics["correct"] == 561, "secondary_metrics_strictly_above_c1": secondary, "selected_bacc_or_ad_cn_risk": not is_protected(report)}
    if metrics["correct"] >= 563:
        return "C1_SEARCH_SUCCESS", checks
    if metrics["correct"] == 562:
        return "C1_SEARCH_POSITIVE", checks
    if metrics["correct"] == 561:
        return ("C1_SEARCH_WEAK_GAIN" if secondary >= 2 else "C1_SEARCH_NO_RELIABLE_GAIN"), checks
    return "C1_SEARCH_NO_GAIN", checks


def render_report(summary: dict) -> str:
    best = summary["selected_report"]
    metrics = best["metrics"]
    paired, boundary, predicted = best["comparison_vs_c1"], best["boundary_errors"], best["predicted_class_counts"]
    spec = best["trial"]
    lines = [
        "# C1 Broad Hyperparameter Search v1", "",
        f"Decision: **{summary['decision']}**", f"Selected trial across all completed experiments: **{best['trial_id']}** (Correct > AUC > Macro-F1)",
        f"Reached 563/598: **{metrics['correct'] >= 563}**", "",
        "## Best configuration and result", "",
        f"- rank/base lr/weight decay: {spec['rank']} / {spec['base_lr']:.12g} / {spec['base_weight_decay']:.12g}",
        f"- lambda_aux/adapter LR multiplier/dropout multiplier: {spec['lambda_aux']} / {spec['adapter_lr_multiplier']} / {spec['dropout_multiplier']}",
        f"- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: {metrics['correct']}/598 / {metrics['acc']:.7f} / {metrics['macro_f1']:.7f} / {metrics['bacc']:.7f} / {metrics['macro_auc']:.7f} / {metrics['weighted_f1']:.7f}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        f"- Repairs/damages/changed vs C1: {paired['repairs']} / {paired['damages']} / {paired['changed_predictions']}",
        f"- AD-sMCI/CN-sMCI/AD-CN errors: {boundary['AD_SMCI']} / {boundary['CN_SMCI']} / {boundary['AD_CN']}",
        f"- Predicted AD/CN/sMCI: {predicted['AD']} / {predicted['CN']} / {predicted['SMCI']}",
        f"- Parameters: {best['parameter_count']}; AD-CN/BACC category-risk flag: {not is_protected(best)}",
        f"- Best protected trial for reference: {summary['best_protected_trial_id']}", "",
        "## Fixed-reference comparison", "",
        "| Reference | Correct | ACC | Macro-F1 | BACC | AUC | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, reference in REFERENCE_RESULTS.items():
        lines.append(f"| {name} | {reference['correct']}/598 | {reference['acc']:.7f} | {reference['macro_f1']:.7f} | {reference['bacc']:.7f} | {reference['macro_auc']:.7f} | {reference['weighted_f1']:.7f} |")
    lines.extend([
        f"| Best search | {metrics['correct']}/598 | {metrics['acc']:.7f} | {metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | {metrics['macro_auc']:.7f} | {metrics['weighted_f1']:.7f} |", "",
        "T1/T2/T3 are comparison-only failed tune arms; T2 was not used as a baseline.", "",
        "## Ten folds", "",
        ", ".join(f"fold{fold['fold']}={fold['best_epoch']}/{fold['acc']:.7f}" for fold in best["folds"]), "",
        "## Stage A top 10", "",
        "| Rank | Trial | Correct | AUC | Macro-F1 | BACC | lr | wd | lambda | adapter mult | dropout mult | protected |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for index, report in enumerate(summary["stage_a_top10"], start=1):
        item, m = report["trial"], report["metrics"]
        lines.append(f"| {index} | {report['trial_id']} | {m['correct']} | {m['macro_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {item['base_lr']:.8g} | {item['base_weight_decay']:.8g} | {item['lambda_aux']} | {item['adapter_lr_multiplier']} | {item['dropout_multiplier']} | {is_protected(report)} |")
    lines.extend(["", "## Rank capacity sensitivity", ""])
    for rank, value in summary["capacity_sensitivity"]["matched_parent_aggregate_by_rank"].items():
        lines.append(f"- rank {rank}: n={value['n']}, parameters={value['parameter_count']}, Correct mean/max={value['correct_mean']:.3f}/{value['correct_max']}, AUC mean={value['auc_mean']:.7f}, BACC mean={value['bacc_mean']:.7f}")
    lines.extend(["", "Matched rank 4/8/12/16 curves for each Stage B parent are stored in `all_trials.json`.", "", "## Parameter relationships (Stage A)", ""])
    for parameter, relation in summary["parameter_relationships"].items():
        pearson = relation["pearson"]
        lines.append(f"- {parameter}: Pearson with Correct/AUC/BACC = {pearson['correct']} / {pearson['macro_auc']} / {pearson['bacc']}; grouped means are stored in `all_trials.json`.")
    lines.extend([
        "", "## Provenance and runtime", "",
        f"- GPU: {summary['device_name']}", f"- source commit: {summary['source_commit']}",
        "- result commit: the Git commit containing this report (self-referential SHA is intentionally not embedded)",
        f"- completed/failed experiments: {summary['completed_trials']} / {summary['failed_trials']}",
        f"- summed fold training seconds / formal-session wall seconds: {summary['total_training_seconds']:.3f} / {summary['formal_session_wall_seconds']:.3f}",
        f"- sampler: {summary['sampler_effective']} (Optuna detected: {summary['optuna_detected']})", "",
    ])
    return "\n".join(lines)


def write_final_artifacts(output_root: Path, reports: list[dict], failures: list[dict], stage_a: list[dict], stage_b: list[dict], parents: list[dict], rows_by_trial: dict[str, list[dict]], started: float) -> dict:
    ranked_all = sorted(reports, key=ranking_key, reverse=True)
    require(bool(ranked_all), "No completed experiments")
    valid = [report for report in ranked_all if is_protected(report)]
    best = ranked_all[0]
    best_protected = valid[0] if valid else None
    decision, checks = final_decision(best)
    capacity = capacity_sensitivity(reports, parents)
    relationships = parameter_relationships(stage_a)
    summary = {
        "decision": decision, "decision_checks": checks, "selection_rule": ["Correct", "Probability Macro-AUC", "Macro-F1"],
        "selection_policy": "all completed Stage A and Stage B experiments ranked by Correct>AUC>Macro-F1; protection gate applies only to Stage B parent admission",
        "selected_trial_id": best["trial_id"], "selected_report": best,
        "selected_category_performance_risk": not is_protected(best),
        "best_protected_trial_id": best_protected["trial_id"] if best_protected else None,
        "best_protected_report": best_protected,
        "completed_trials": len(reports), "failed_trials": len(failures), "stage_a_completed": len(stage_a), "stage_b_completed": len(stage_b),
        "stage_b_parents": [{"trial_id": parent["trial_id"], "protected": is_protected(parent), "ranking_key": ranking_key(parent)} for parent in parents],
        "stage_a_top10": sorted(stage_a, key=ranking_key, reverse=True)[:10], "capacity_sensitivity": capacity,
        "parameter_relationships": relationships, "references": REFERENCE_RESULTS,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(), "result_commit_semantics": "commit containing these final artifacts",
        "device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
        "total_training_seconds": float(sum(report["training_seconds"] for report in reports)), "formal_session_wall_seconds": float(time.perf_counter() - started),
        "sampler_effective": "python_stdlib_deterministic_fallback", "optuna_detected": False,
        "run_command": f'"{sys.executable}" -u -B scripts/run_c1_broad_hparam_search_v1.py formal --device cuda:0',
    }
    stage_a_rows = [flat_trial(report) for report in sorted(stage_a, key=ranking_key, reverse=True)]
    stage_b_rows = [flat_trial(report) for report in sorted(stage_b, key=ranking_key, reverse=True)]
    failed_rows = [flat_trial(failure) for failure in failures]
    write_csv(output_root / "stage_a_trials.csv", stage_a_rows + [row for row in failed_rows if row["stage"] == "A"])
    write_csv(output_root / "stage_b_rank_trials.csv", stage_b_rows + [row for row in failed_rows if row["stage"] == "B"])
    write_json(output_root / "all_trials.json", {"summary": summary, "completed_trials": sorted(reports, key=ranking_key, reverse=True), "failed_trials": failures})
    write_csv(output_root / "top10_trials.csv", [flat_trial(report) for report in ranked_all[:10]])
    write_json(output_root / "best_config.json", {"trial_id": best["trial_id"], "trial": best["trial"], "runtime_config": best["trial_config"], "metrics": best["metrics"], "protected": is_protected(best), "decision": decision})
    best_rows = validate_oof(rows_by_trial[best["trial_id"]])
    write_csv(output_root / "best_oof_predictions.csv", [{"fold": row["fold"], "subject_index": row["subject_index"], "truth": row["truth"], "prediction": row["prediction"]} for row in best_rows])
    write_csv(output_root / "best_oof_probabilities.csv", [{"fold": row["fold"], "subject_index": row["subject_index"], "truth": row["truth"], **{f"probability_{name}": row[f"probability_{name}"] for name in CLASS_NAMES}} for row in best_rows])
    (output_root / "REPORT.md").write_text(render_report(summary) + "\n", encoding="utf-8")
    return summary


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode == 0, "Source/manifest/smoke files must be committed before formal search")
    require(git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode == 0, "Committed source/manifest/smoke files changed")


def validate_formal_gate(output_root: Path, manifest: dict) -> None:
    report_path = output_root / "smoke/smoke_report.json"
    config_path = output_root / "smoke/smoke_config.json"
    require(report_path.is_file() and config_path.is_file(), "Passing smoke is required before formal search")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("passed") is True and report.get("epochs") == 3 and report.get("fold") == 0, "Invalid smoke report")
    current_hash = file_sha256(ROOT / RUNNER_REL)
    require(manifest["runner_sha256"] == current_hash and report["runner_sha256"] == current_hash, "Runner differs from manifest/smoke-tested source")
    require(report["manifest_core_sha256"] == manifest["manifest_core_sha256"], "Smoke/manifest mismatch")
    require(git("rev-parse", "HEAD").stdout.strip() != BASE_COMMIT, "Formal search requires a source commit after smoke")
    locked_base_dependencies = [
        ROOT / cme.CONFIG_REL,
        ROOT / "Model/cme_dual_branch.py",
        ROOT / "Model/network.py",
        ROOT / "Loss/loss_fn.py",
        ROOT / "Utils/utils.py",
        ROOT / "scripts/run_cme_dual_branch_v1.py",
    ]
    require_committed_unchanged([ROOT / RUNNER_REL, *locked_base_dependencies, output_root / "search_manifest.json", report_path, config_path])
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in locked_base_dependencies]
    require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", *relative, check=False).returncode == 0, "Locked C1 transitive dependency differs from base commit")


def load_or_freeze_stage_b_plan(output_root: Path, stage_a_reports: list[dict]) -> tuple[list[dict], list[dict]]:
    path = output_root / "stage_b_plan.json"
    by_id = {report["trial_id"]: report for report in stage_a_reports}
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("source_commit") == source_commit, "Frozen Stage B plan source commit changed")
        specs = payload.get("configs", [])
        parent_ids = payload.get("parent_trial_ids", [])
        require(len(specs) == 9 and len(parent_ids) == 3 and len(set(parent_ids)) == 3, "Frozen Stage B plan is invalid")
        require({spec["trial_id"] for spec in specs} == {f"B{slot:02d}R{rank:02d}" for slot in range(1, 4) for rank in STAGE_B_RANKS}, "Frozen Stage B trial IDs changed")
        require(all(parent_id in by_id for parent_id in parent_ids), "Frozen Stage B parent is not a completed Stage A trial")
        parents = [by_id[parent_id] for parent_id in parent_ids]
        require(specs == stage_b_specs(parents), "Frozen Stage B plan/config inheritance changed")
        return parents, specs
    parents = select_stage_b_parents(stage_a_reports)
    specs = stage_b_specs(parents)
    payload = {
        "source_commit": source_commit,
        "parent_trial_ids": [parent["trial_id"] for parent in parents],
        "parents": [{"trial_id": parent["trial_id"], "protected": is_protected(parent), "ranking_key": ranking_key(parent)} for parent in parents],
        "configs": specs,
    }
    write_json(path, payload)
    return parents, specs


def run_formal(context: dict, output_root: Path, manifest: dict) -> dict:
    validate_formal_gate(output_root, manifest)
    started = time.perf_counter()
    stage_a_reports, stage_b_reports, failures = [], [], []
    rows_by_trial: dict[str, list[dict]] = {}
    common_failures: dict[tuple[str, str], int] = {}

    def execute(spec: dict, destination: list[dict]) -> None:
        failure_path = trial_directory(output_root, spec) / "failure.json"
        complete_path = trial_directory(output_root, spec) / "complete.json"
        if failure_path.is_file() and not complete_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            require(failure.get("trial") == spec and failure.get("retry_on_resume") is False, f"{spec['trial_id']}: terminal failure/config changed")
            attempts = failure.get("attempts", [])
            require(bool(attempts) and attempts[-1].get("source_commit") == git("rev-parse", "HEAD").stdout.strip(), f"{spec['trial_id']}: terminal failure source changed")
            failures.append(failure)
            print(f"RESUME_TERMINAL_FAILURE {spec['trial_id']}: {attempts[-1].get('reason')}", flush=True)
            return
        try:
            report, rows = run_trial(context, spec, output_root)
            destination.append(report)
            rows_by_trial[spec["trial_id"]] = rows
        except InvariantError:
            raise
        except Exception as exc:
            failure = record_trial_failure(spec, output_root, exc)
            failures.append(failure)
            torch.cuda.empty_cache()
            signature = (type(exc).__name__, str(exc))
            common_failures[signature] = common_failures.get(signature, 0) + 1
            require(common_failures[signature] < 3, f"Same runtime failure repeated across three trials; common runner failure suspected: {signature}")

    for spec in manifest["manifest_core"]["stage_a"]["configs"]:
        execute(spec, stage_a_reports)
    parents, specs_b = load_or_freeze_stage_b_plan(output_root, stage_a_reports)
    for spec in specs_b:
        execute(spec, stage_b_reports)
    return write_final_artifacts(output_root, stage_a_reports + stage_b_reports, failures, stage_a_reports, stage_b_reports, parents, rows_by_trial, started)


def verify_branch() -> None:
    require(git("cat-file", "-e", f"{BASE_COMMIT}^{{commit}}", check=False).returncode == 0, "Base C1 commit missing")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD", check=False).returncode == 0, "Branch is not based on locked C1 commit")
    branch = git("branch", "--show-current").stdout.strip()
    require(branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-"), f"Wrong branch: {branch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_branch()
    output_root = args.output_root.resolve()
    require(output_root == (ROOT / OUTPUT_REL).resolve(), "Broad search output path is locked")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = create_or_validate_manifest(output_root)
    if args.stage == "manifest":
        print(f"MANIFEST_READY configs={len(manifest['manifest_core']['stage_a']['configs'])} sampler={manifest['manifest_core']['sampler']['effective']}", flush=True)
        return
    context = load_context(args.device)
    if args.stage == "smoke":
        report = run_smoke(context, output_root, manifest)
        print(f"SMOKE passed={report['passed']} trial={report['trial_id']}", flush=True)
    else:
        summary = run_formal(context, output_root, manifest)
        print(f"{summary['decision']} selected={summary['selected_trial_id']} correct={summary['selected_report']['metrics']['correct']}/598", flush=True)


if __name__ == "__main__":
    main()
