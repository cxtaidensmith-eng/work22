"""Nested binary task-adapter calibration with corrected two-OVR loss."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_cross_dataset_a012_structure_v1 as cross  # noqa: E402
from Loss.loss_fn import orthogonality_lossv2  # noqa: E402


EXPERIMENT_ID = "binary_task_adapter_calibration_v1"
BRANCH = "experiment/binary-task-adapter-calibration-v1"
BASE_COMMIT = "17004d29714471b5e1548e04bac0072e854e7e18"
LEGACY_RESULT_COMMIT = "e7d18205ddfefaf898a81c999cce15b348331206"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
SELECTION_PATH = RESULT_DIR / "selection_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
SEARCH_DIR = RESULT_DIR / "search"
FORMAL_DIR = RESULT_DIR / "formal"
FOLD_RESULT_DIR = RESULT_DIR / "fold_results"
SEARCH_RESULTS_PATH = RESULT_DIR / "search_results.json"

TASK_IDS = ("tadpole_smci_pmci", "abide_ads_cn", "abide5_ads_cn")
FOLDS = tuple(range(10))
SEED = 0
MAX_EPOCHS = 400
INNER_VALIDATION_FRACTION = 0.2
LAMBDA_AUX = 0.2
RANKS = (4, 8)
MULTIPLIERS = (0.5, 1.0, 2.0)
CANDIDATES = tuple(
    {"rank": rank, "adapter_lr_multiplier": multiplier, "candidate_id": f"r{rank}_m{str(multiplier).replace('.', 'p')}"}
    for rank in RANKS
    for multiplier in MULTIPLIERS
)

REQUIRED_TRACKED = (
    ".gitignore",
    "scripts/run_binary_task_adapter_calibration_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/selection_manifest.json",
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
    "scripts/run_c1_broad_hparam_search_v1.py",
    "experiments/cross_dataset_a012_structure_v1/protocols/tadpole_smci_pmci.json",
    "experiments/cross_dataset_a012_structure_v1/protocols/abide_ads_cn.json",
    "experiments/cross_dataset_a012_structure_v1/protocols/abide5_ads_cn.json",
)

InvariantError = cross.InvariantError
require = cross.require
canonical_json = cross.canonical_json
payload_sha256 = cross.payload_sha256
file_sha256 = cross.file_sha256
atomic_write_text = cross.atomic_write_text
atomic_write_json = cross.atomic_write_json
atomic_write_csv = cross.atomic_write_csv
atomic_torch_save = cross.atomic_torch_save
read_json = cross.read_json
read_csv_rows = cross.read_csv_rows
clone_cpu_state = cross.clone_cpu_state
capture_rng_state = cross.capture_rng_state
restore_rng_state = cross.restore_rng_state


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def current_head() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def implementation_commit() -> str:
    commits = {
        git("log", "-1", "--format=%H", "--", relative).stdout.strip()
        for relative in REQUIRED_TRACKED
    }
    require(len(commits) == 1 and "" not in commits, f"Implementation source spans commits: {sorted(commits)}")
    return next(iter(commits))


def source_hashes() -> dict[str, str]:
    return {
        relative: file_sha256(ROOT / relative)
        for relative in (*REQUIRED_TRACKED, *LOCKED_DEPENDENCIES)
    }


def json_with_digest(path: Path, core: dict[str, Any]) -> dict[str, Any]:
    payload = {**core, "sha256": payload_sha256(core)}
    atomic_write_json(path, payload)
    return payload


def combine_corrected_losses(
    main_loss: torch.Tensor,
    auxiliary_losses: list[torch.Tensor],
    orthogonality_loss: torch.Tensor,
    orthogonality_rate: float,
) -> torch.Tensor:
    require(len(auxiliary_losses) == 2, "Corrected binary loss requires exactly two OVR losses")
    auxiliary_mean = torch.stack(auxiliary_losses).mean()
    return main_loss + LAMBDA_AUX * auxiliary_mean + float(orthogonality_rate) * orthogonality_loss


def corrected_loss_unit_check() -> dict[str, Any]:
    main = torch.tensor(1.25, dtype=torch.float64)
    auxiliary = [torch.tensor(0.4, dtype=torch.float64), torch.tensor(0.8, dtype=torch.float64)]
    orthogonal = torch.tensor(0.3, dtype=torch.float64)
    rate = 0.01
    actual = combine_corrected_losses(main, auxiliary, orthogonal, rate)
    expected = main + 0.2 * ((auxiliary[0] + auxiliary[1]) / 2.0) + rate * orthogonal
    error = float((actual - expected).abs())
    require(error < 1e-8, f"Corrected loss unit check failed: {error}")
    return {
        "main_loss": float(main),
        "ovr_losses": [float(value) for value in auxiliary],
        "ovr_mean": float(torch.stack(auxiliary).mean()),
        "lambda_aux": LAMBDA_AUX,
        "orthogonality_loss": float(orthogonal),
        "orthogonality_rate": rate,
        "expected_total": float(expected),
        "actual_total": float(actual),
        "absolute_error": error,
        "passed": True,
    }


class CorrectedBinaryLoss:
    """Training-mask weighted main CE plus 0.2 times mean(two OVR CE)."""

    def __init__(
        self,
        labels: torch.Tensor,
        training_mask: torch.Tensor,
        device: torch.device,
        orthogonality_rate: float,
        label_smoothing: float,
    ) -> None:
        require(labels.ndim == 1 and training_mask.dtype == torch.bool, "Invalid loss mask")
        selected = labels[training_mask]
        require(selected.numel() > 0 and int(selected.min()) == 0 and int(selected.max()) == 1, "Training mask lost a binary class")
        counts = torch.bincount(selected, minlength=2).to(device=device, dtype=torch.float32)
        total = float(selected.numel())
        self.class_weight = (total - counts) / total
        self.main_loss = nn.CrossEntropyLoss(
            weight=self.class_weight,
            label_smoothing=float(label_smoothing),
        ).to(device)
        self.aux_losses = nn.ModuleList()
        for class_index in range(2):
            binary_weight = torch.empty(2, device=device)
            binary_weight[1] = self.class_weight[class_index]
            binary_weight[0] = self.class_weight.sum() - binary_weight[1]
            self.aux_losses.append(
                nn.CrossEntropyLoss(
                    weight=binary_weight,
                    label_smoothing=float(label_smoothing),
                ).to(device)
            )
        self.orthogonality_rate = float(orthogonality_rate)
        self.training_count = int(selected.numel())
        self.training_class_counts = counts.detach().cpu().to(torch.int64).tolist()

    def components(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        label_embeddings: torch.Tensor,
        auxiliary_outputs: list[torch.Tensor],
    ) -> dict[str, Any]:
        require(len(auxiliary_outputs) == 2, "Expected two OVR outputs")
        one_hot = F.one_hot(labels, num_classes=2).transpose(0, 1).reshape(2, -1)
        main = self.main_loss(output[mask], labels[mask])
        auxiliary = [
            loss_fn(auxiliary_outputs[index][mask], one_hot[index][mask])
            for index, loss_fn in enumerate(self.aux_losses)
        ]
        orthogonal = orthogonality_lossv2(label_embeddings)
        total = combine_corrected_losses(
            main,
            auxiliary,
            orthogonal,
            self.orthogonality_rate,
        )
        return {
            "total": total,
            "main": main,
            "auxiliary": auxiliary,
            "auxiliary_mean": torch.stack(auxiliary).mean(),
            "orthogonality": orthogonal,
        }

    def __call__(self, output, labels, mask, label_embeddings, auxiliary_outputs):
        return self.components(output, labels, mask, label_embeddings, auxiliary_outputs)["total"]


def legacy_reference() -> dict[str, Any]:
    relative = "experiments/cross_dataset_a012_structure_v1/summary.json"
    completed = git("show", f"{LEGACY_RESULT_COMMIT}:{relative}", check=False)
    require(completed.returncode == 0, "Verified cross-dataset legacy summary is unavailable")
    payload = json.loads(completed.stdout)
    return {
        "commit": LEGACY_RESULT_COMMIT,
        "summary_sha256": payload["sha256"],
        "task_results": payload["task_results"],
        "decision": payload["decision"],
        "usage": "reference_only_not_used_for_current_decision",
    }


def stable_row(context: dict[str, Any], dataset_position: int) -> dict[str, Any]:
    original_indices = np.asarray(context["dataset_dict"]["Index"], dtype=int)
    original = int(original_indices[int(dataset_position)])
    label = int(context["dataset_data"]["Label"][int(dataset_position)].detach().cpu())
    feature_hash = context["feature_hashes"][original]
    return {
        "subject_id": cross.stable_subject_id(context["task_id"], original, feature_hash),
        "original_csv_index": original,
        "feature_sha256": feature_hash,
        "truth": label,
    }


def make_selection_manifest() -> dict[str, Any]:
    protocols = cross.load_protocols()
    tasks: dict[str, Any] = {}
    for task_id in TASK_IDS:
        context = cross.build_context(protocols[task_id], torch.device("cpu"))
        labels = context["dataset_data"]["Label"].detach().cpu().numpy().astype(int)
        folds: list[dict[str, Any]] = []
        for fold in FOLDS:
            outer_train, outer_test, _ = cross.fold_positions(context, fold)
            outer_positions = np.flatnonzero(outer_train.detach().cpu().numpy())
            train_positions, validation_positions = train_test_split(
                outer_positions,
                test_size=INNER_VALIDATION_FRACTION,
                random_state=SEED,
                shuffle=True,
                stratify=labels[outer_positions],
            )
            train_positions = np.sort(np.asarray(train_positions, dtype=int))
            validation_positions = np.sort(np.asarray(validation_positions, dtype=int))
            test_positions = np.flatnonzero(outer_test.detach().cpu().numpy())
            require(not (set(train_positions) & set(validation_positions)), "Inner split overlaps")
            require(set(train_positions) | set(validation_positions) == set(outer_positions), "Inner split does not cover outer train")
            require(not ((set(train_positions) | set(validation_positions)) & set(test_positions)), "Outer test leaked into inner split")
            train_counts = np.bincount(labels[train_positions], minlength=2).astype(int).tolist()
            validation_counts = np.bincount(labels[validation_positions], minlength=2).astype(int).tolist()
            if task_id == "tadpole_smci_pmci":
                require(validation_counts[int(protocols[task_id]["positive_index"])] >= 8, "TADPOLE inner validation lacks pMCI")
            fold_core = {
                "fold": fold,
                "outer_train_count": int(len(outer_positions)),
                "outer_test_count": int(len(test_positions)),
                "inner_train_class_counts": train_counts,
                "inner_validation_class_counts": validation_counts,
                "inner_train": [stable_row(context, position) for position in train_positions],
                "inner_validation": [stable_row(context, position) for position in validation_positions],
                "outer_test": [stable_row(context, position) for position in test_positions],
            }
            folds.append({**fold_core, "sha256": payload_sha256(fold_core)})
        task_core = {"task_id": task_id, "folds": folds}
        tasks[task_id] = {**task_core, "sha256": payload_sha256(task_core)}
    core = {
        "experiment": EXPERIMENT_ID,
        "seed": SEED,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "candidates": list(CANDIDATES),
        "tasks": tasks,
        "outer_test_used_for_candidate_or_epoch_selection": False,
    }
    return {**core, "sha256": payload_sha256(core)}


def make_inspect_manifest(historical_root: Path) -> dict[str, Any]:
    cross_inspect, cross_folds = cross.make_inspect_payload(historical_root.resolve())
    legacy_source = inspect.getsource(cross.criterion_lossv2.__call__)
    require("ce_loss + aux_loss" in legacy_source, "Legacy OVR execution path changed")
    core = {
        "experiment": EXPERIMENT_ID,
        "base_source_commit": BASE_COMMIT,
        "cross_dataset_input_inspect_sha256": cross_inspect["sha256"],
        "cross_dataset_fold_manifest_sha256": cross_folds["sha256"],
        "tasks": cross_inspect["tasks"],
        "abide_overlap": cross_inspect["abide_overlap"],
        "legacy_loss_audit": {
            "implementation": "criterion_lossv2.__call__",
            "finding": "legacy code accumulates both OVR CE values and returns main + unnormalized auxiliary sum",
            "declared_abide_configuration": "aux_loss_weight=0.2, reduction=mean",
            "effective_legacy_formula": "main + OVR_0 + OVR_1 + historical_orthogonality",
            "corrected_formula": "main + 0.2 * ((OVR_0 + OVR_1) / 2) + historical_orthogonality",
            "loss_file_sha256": file_sha256(ROOT / "Loss/loss_fn.py"),
            "historical_default_modified": False,
        },
        "loss_unit_check": corrected_loss_unit_check(),
        "legacy_cross_dataset_reference": legacy_reference(),
        "selection_manifest_sha256": make_selection_manifest()["sha256"],
    }
    return {**core, "sha256": payload_sha256(core)}


def run_inspect(historical_root: Path) -> None:
    selection = make_selection_manifest()
    atomic_write_json(SELECTION_PATH, selection)
    payload = make_inspect_manifest(historical_root)
    atomic_write_json(INSPECT_PATH, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_config() -> dict[str, Any]:
    payload = read_json(CONFIG_PATH)
    require(payload["base_source_commit"] == BASE_COMMIT, "Experiment base commit changed")
    require(payload["rank_candidates"] == list(RANKS), "Rank grid changed")
    require(payload["adapter_lr_multiplier_candidates"] == list(MULTIPLIERS), "Adapter LR grid changed")
    require(float(payload["lambda_aux"]) == LAMBDA_AUX, "Auxiliary weight changed")
    require(int(payload["max_search_epochs"]) == MAX_EPOCHS, "Epoch ceiling changed")
    return payload


def validate_manifests(historical_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require(INSPECT_PATH.is_file() and SELECTION_PATH.is_file(), "Run inspect before continuing")
    selection = read_json(SELECTION_PATH)
    inspection = read_json(INSPECT_PATH)
    require(selection == make_selection_manifest(), "Selection manifest drifted")
    require(inspection == make_inspect_manifest(historical_root.resolve()), "Inspect manifest drifted")
    return inspection, selection


def source_gate(historical_root: Path) -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong experiment branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree drifted")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index has staged drift")
    source = implementation_commit()
    require(source != BASE_COMMIT, "Implementation must be committed before smoke")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, source, check=False).returncode == 0, "Base is not source ancestor")
    changed = {
        line.strip().replace("\\", "/")
        for line in git("diff", "--name-only", BASE_COMMIT, source).stdout.splitlines()
        if line.strip()
    }
    require(changed == set(REQUIRED_TRACKED), f"Implementation scope changed: {sorted(changed)}")
    for relative in REQUIRED_TRACKED:
        require(git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0, f"Untracked source: {relative}")
        require(git("diff", "--quiet", source, "--", relative, check=False).returncode == 0, f"Source drift: {relative}")
    for relative in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, source, "--", relative, check=False).returncode == 0, f"Locked dependency changed: {relative}")
        require(git("diff", "--quiet", source, "--", relative, check=False).returncode == 0, f"Dependency worktree drift: {relative}")
    validate_manifests(historical_root)
    return source


def runtime_lock(protocols: dict[str, dict[str, Any]], source_commit: str) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "base_commit": BASE_COMMIT,
        "runner_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "inspect_sha256": file_sha256(INSPECT_PATH),
        "selection_manifest_sha256": file_sha256(SELECTION_PATH),
        "protocol_sha256": {task_id: payload_sha256(protocols[task_id]) for task_id in TASK_IDS},
        "dependency_sha256": {relative: file_sha256(ROOT / relative) for relative in LOCKED_DEPENDENCIES},
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "lambda_aux": LAMBDA_AUX,
        "class_weight_scope": "current_training_mask_only",
        "outer_test_used_for_selection": False,
    }
    return {**core, "sha256": payload_sha256(core)}


def context_positions(context: dict[str, Any]) -> dict[int, int]:
    original = np.asarray(context["dataset_dict"]["Index"], dtype=int)
    require(len(set(original.tolist())) == len(original), "Dataset row mapping is not unique")
    return {int(value): int(position) for position, value in enumerate(original)}


def mask_from_manifest(context: dict[str, Any], rows: list[dict[str, Any]]) -> torch.Tensor:
    mapping = context_positions(context)
    mask = torch.zeros(context["protocol"]["sample_count"], dtype=torch.bool, device=context["device"])
    for row in rows:
        original = int(row["original_csv_index"])
        require(original in mapping, "Manifest row is absent from dataset")
        position = mapping[original]
        anchor = stable_row(context, position)
        require(anchor == row, "Stable subject manifest drifted")
        mask[position] = True
    require(int(mask.sum()) == len(rows), "Manifest mask count changed")
    return mask


def split_masks(context: dict[str, Any], selection: dict[str, Any], fold: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = selection["tasks"][context["task_id"]]["folds"][fold]
    require(payload["fold"] == fold, "Selection fold order changed")
    inner_train = mask_from_manifest(context, payload["inner_train"])
    inner_validation = mask_from_manifest(context, payload["inner_validation"])
    outer_test = mask_from_manifest(context, payload["outer_test"])
    require(not bool(torch.any(inner_train & inner_validation)), "Inner masks overlap")
    require(not bool(torch.any((inner_train | inner_validation) & outer_test)), "Outer-test leaked into selection masks")
    outer_train, expected_test, _ = cross.fold_positions(context, fold)
    require(torch.equal(inner_train | inner_validation, outer_train), "Inner split no longer covers outer train")
    require(torch.equal(outer_test, expected_test), "Outer-test ownership changed")
    return inner_train, inner_validation, outer_test


class RatioScheduler(cross.CustomCosineAnnealingLR):
    def __init__(self, optimizer: torch.optim.Optimizer, t_max: int, eta_min: float, multiplier: float, last_epoch: int = -1):
        require(len(optimizer.param_groups) == 2, "B1 requires exactly two optimizer groups")
        self.T_max = int(t_max)
        self.eta_min = float(eta_min)
        self.hold_epoch = 20
        self.multiplier = float(multiplier)
        self.initial_lrs_locked = [float(group["lr"]) for group in optimizer.param_groups]
        require(abs(self.initial_lrs_locked[1] - self.initial_lrs_locked[0] * self.multiplier) <= 1e-14, "Initial adapter LR changed")
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch)
        self.assert_ratio()

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.hold_epoch:
            base = self.initial_lrs_locked[0]
        else:
            current = self.last_epoch - self.hold_epoch
            base = self.eta_min + (self.initial_lrs_locked[0] - self.eta_min) * (
                1.0 + math.cos(math.pi * current / (self.T_max - self.hold_epoch))
            ) / 2.0
        return [base, base * self.multiplier]

    def assert_ratio(self) -> None:
        base, adapter = [float(group["lr"]) for group in self.optimizer.param_groups]
        require(abs(adapter - base * self.multiplier) <= max(1e-14, abs(adapter) * 1e-12), "Adapter LR ratio changed")

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        self.assert_ratio()


def common_state_max_diff(reference: nn.Module, model: nn.Module) -> float:
    left, right = reference.state_dict(), model.state_dict()
    require(set(left).issubset(right), "B1 lost a common parameter")
    maximum = 0.0
    for name, tensor in left.items():
        require(tensor.shape == right[name].shape and tensor.dtype == right[name].dtype, f"Common schema changed: {name}")
        maximum = max(maximum, float((tensor.detach().cpu() - right[name].detach().cpu()).abs().max()))
    return maximum


def make_objects(
    context: dict[str, Any],
    arm: str,
    training_mask: torch.Tensor,
    rank: int | None = None,
    multiplier: float | None = None,
    audit_common: bool = False,
) -> tuple[nn.Module, CorrectedBinaryLoss, torch.optim.Optimizer, Any, dict[str, Any]]:
    require(arm in {"B0", "B1"}, "Unknown arm")
    kwargs = cross.model_kwargs(context)
    training = context["protocol"]["training"]
    cross.SET_Random(SEED)
    if arm == "B0":
        model = cross.HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = capture_rng_state()
        common_audit: dict[str, Any] = {"common_state_pointwise_audit": "not_applicable"}
    else:
        require(rank in RANKS and multiplier in MULTIPLIERS, "B1 candidate is outside fixed grid")
        reference = cross.HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = capture_rng_state()
        cross.SET_Random(SEED)
        model = cross.CMEDualBranchModel(
            **kwargs,
            cme_arm="c1",
            adapter_rank=int(rank),
            router_hidden=16,
            modality_embedding_dim=8,
        ).to(context["device"])
        difference = common_state_max_diff(reference, model) if audit_common else None
        if audit_common:
            require(difference == 0.0, "B0/B1 common initialization changed")
        common_audit = {"common_state_max_abs_diff": difference, "audited": bool(audit_common)}
        del reference

    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary Query/OVR count changed")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    base_lr = float(training["lr"])
    weight_decay = float(training["weight_decay"])
    if arm == "B0":
        optimizer = torch.optim.Adam([parameter for _, parameter in named], lr=base_lr, weight_decay=weight_decay)
        scheduler = cross.CustomCosineAnnealingLR(
            optimizer,
            T_max=int(training["scheduler_t_max"]),
            eta_min=float(training["scheduler_eta_min"]),
        )
        group_audit = {"base_parameter_tensors": len(named), "adapter_parameter_tensors": 0, "base_lr": base_lr}
    else:
        adapter_named = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
        base_named = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
        expected_adapter_tensors = len(context["protocol"]["modalities"]) * 4
        require(len(model.private_adapters) == len(context["protocol"]["modalities"]), "Adapter count differs from real modalities")
        require(len(adapter_named) == expected_adapter_tensors, "Adapter parameter tensor count changed")
        all_ids = {id(parameter) for _, parameter in named}
        base_ids = {id(parameter) for _, parameter in base_named}
        adapter_ids = {id(parameter) for _, parameter in adapter_named}
        require(not base_ids.intersection(adapter_ids) and base_ids.union(adapter_ids) == all_ids, "Optimizer partition is not exhaustive")
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter for _, parameter in base_named], "lr": base_lr, "weight_decay": weight_decay, "group_name": "base"},
                {"params": [parameter for _, parameter in adapter_named], "lr": base_lr * float(multiplier), "weight_decay": weight_decay, "group_name": "private_adapters"},
            ]
        )
        scheduler = RatioScheduler(
            optimizer,
            int(training["scheduler_t_max"]),
            float(training["scheduler_eta_min"]),
            float(multiplier),
        )
        group_audit = {
            "base_parameter_tensors": len(base_named),
            "adapter_parameter_tensors": len(adapter_named),
            "adapter_parameter_names": [name for name, _ in adapter_named],
            "base_lr": base_lr,
            "adapter_lr": base_lr * float(multiplier),
            "adapter_lr_multiplier": float(multiplier),
        }
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    criterion = CorrectedBinaryLoss(
        context["dataset_data"]["Label"], training_mask, context["device"],
        float(training["loss_rate"]), float(training["label_smoothing"]),
    )
    restore_rng_state(post_common_rng)
    return model, criterion, optimizer, scheduler, {
        **common_audit,
        **group_audit,
        "parameter_count": parameter_count(model),
        "query_count": len(model.label_pools),
        "ovr_head_count": len(model._Auxi_classifier),
        "adapter_count": len(model.private_adapters) if arm == "B1" else 0,
        "rank": int(rank) if rank is not None else None,
        "training_class_counts": criterion.training_class_counts,
        "class_weight": criterion.class_weight.detach().cpu().tolist(),
    }


def metrics_for(model: nn.Module, context: dict[str, Any], mask: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(context["dataset_data"]["Feature"])[0][mask]
        probabilities = torch.softmax(logits, dim=-1)
    require(tuple(logits.shape)[1:] == (2,), "Binary logits shape changed")
    require(bool(torch.isfinite(probabilities).all()), "Probability is non-finite")
    truth = context["dataset_data"]["Label"][mask].detach().cpu().numpy().astype(np.int64)
    metrics = cross.binary_metrics(truth, probabilities.detach().cpu().numpy().astype(np.float64), context["protocol"])
    return metrics, logits.detach().cpu(), probabilities.detach().cpu()


def candidate_selection_key(task_id: str, metrics: dict[str, Any], epoch: int, candidate_order: int = 0) -> tuple[float, ...]:
    if task_id == "tadpole_smci_pmci":
        values = (metrics["bacc"], metrics["pr_auc"], metrics["macro_f1"], metrics["sen"], metrics["acc"])
    else:
        values = (metrics["bacc"], metrics["roc_auc"], metrics["macro_f1"], metrics["acc"])
    return tuple(float(value) for value in values) + (-int(epoch), -int(candidate_order))


def train_step(
    model: nn.Module,
    criterion: CorrectedBinaryLoss,
    optimizer: torch.optim.Optimizer,
    context: dict[str, Any],
    train_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, embeddings, auxiliary = model(context["dataset_data"]["Feature"])
    components = criterion.components(output, context["dataset_data"]["Label"], train_mask, embeddings, auxiliary)
    require(all(bool(torch.isfinite(components[key])) for key in ("total", "main", "auxiliary_mean", "orthogonality")), "Loss component is non-finite")
    components["total"].backward()
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid adapter gradient: {name}")
            gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Gradient is non-finite")
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
    optimizer.step()
    losses = {
        "loss": float(components["total"].detach().cpu()),
        "main_loss": float(components["main"].detach().cpu()),
        "auxiliary_mean": float(components["auxiliary_mean"].detach().cpu()),
        "orthogonality_loss": float(components["orthogonality"].detach().cpu()),
    }
    return losses, gradients


def trial_lock(
    runtime: dict[str, Any],
    context: dict[str, Any],
    selection: dict[str, Any],
    fold: int,
    arm: str,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    split = selection["tasks"][context["task_id"]]["folds"][fold]
    core = {
        "runtime_lock_sha256": runtime["sha256"],
        "source_commit": runtime["source_commit"],
        "task_id": context["task_id"],
        "task_protocol_sha256": runtime["protocol_sha256"][context["task_id"]],
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "selection_fold_sha256": split["sha256"],
        "fold": int(fold),
        "arm": arm,
        "candidate": copy.deepcopy(candidate),
        "seed": SEED,
        "epochs": MAX_EPOCHS,
        "selection_uses_outer_test": False,
    }
    return {**core, "sha256": payload_sha256(core)}


def trial_paths(task_id: str, fold: int, arm: str, candidate_id: str) -> dict[str, Path]:
    root = SEARCH_DIR / task_id / f"fold_{fold:02d}" / arm / candidate_id
    return {
        "root": root,
        "resume": root / "resume.pt",
        "history": root / "validation_epoch_metrics.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
    }


def search_resume_payload(
    lock: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best: dict[str, Any] | None,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "lock": lock,
        "epoch": int(epoch),
        "optimizer_steps": int(epoch),
        "scheduler_steps": int(epoch),
        "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
        "best": copy.deepcopy(best),
        "history": copy.deepcopy(history),
        "elapsed_seconds": float(elapsed_seconds),
    }


def load_search_resume(
    path: Path,
    lock: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("schema") == 1 and payload.get("lock") == lock, f"Search resume lock changed: {path}")
    epoch = int(payload.get("epoch", -1))
    require(0 < epoch <= MAX_EPOCHS, "Search resume epoch is invalid")
    require(payload.get("optimizer_steps") == epoch and payload.get("scheduler_steps") == epoch, "Search resume counters changed")
    require([int(row["epoch"]) for row in payload["history"]] == list(range(1, epoch + 1)), "Search resume history is not contiguous")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    require(int(scheduler.last_epoch) == epoch, "Search scheduler epoch changed")
    if isinstance(scheduler, RatioScheduler):
        scheduler.assert_ratio()
    restore_rng_state(payload["rng"])
    return payload


def typed_epoch_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    numeric_float = {"loss", "main_loss", "auxiliary_mean", "bacc", "roc_auc", "pr_auc", "macro_f1", "sen", "spe", "acc"}
    typed: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key, value in row.items():
            if key == "epoch" or key == "correct":
                output[key] = int(value)
            elif key in numeric_float:
                output[key] = float(value)
            else:
                output[key] = value
        typed.append(output)
    return typed


def select_history(task_id: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    require([int(row["epoch"]) for row in history] == list(range(1, MAX_EPOCHS + 1)), "Search history is not 1..400")
    return max(history, key=lambda row: candidate_selection_key(task_id, row, int(row["epoch"])))


def validate_search_trial(
    context: dict[str, Any],
    selection: dict[str, Any],
    runtime: dict[str, Any],
    fold: int,
    arm: str,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = "b0" if candidate is None else str(candidate["candidate_id"])
    paths = trial_paths(context["task_id"], fold, arm, candidate_id)
    require(all(paths[name].is_file() for name in ("resume", "history", "summary", "complete")), "Completed search trial is incomplete")
    lock = trial_lock(runtime, context, selection, fold, arm, candidate)
    marker = read_json(paths["complete"])
    marker_core = {key: value for key, value in marker.items() if key != "sha256"}
    require(marker.get("sha256") == payload_sha256(marker_core), "Search complete digest changed")
    require(marker.get("complete") is True and marker.get("lock") == lock, "Search complete lock changed")
    for field, name in (("resume_sha256", "resume"), ("history_sha256", "history"), ("summary_sha256", "summary")):
        require(marker[field] == file_sha256(paths[name]), f"Search {name} changed")
    history = typed_epoch_rows(read_csv_rows(paths["history"]))
    require(len(history) == MAX_EPOCHS and all(math.isfinite(float(row["loss"])) for row in history), "Search history changed")
    selected = select_history(context["task_id"], history)
    summary = read_json(paths["summary"])
    require(summary.get("lock") == lock and int(summary.get("fold", -1)) == fold and summary.get("arm") == arm, "Search summary ownership changed")
    require(summary.get("candidate") == candidate and int(summary.get("best_epoch", -1)) == int(selected["epoch"]), "Search best selection changed")
    for key in ("bacc", "roc_auc", "pr_auc", "macro_f1", "sen", "spe", "acc"):
        require(math.isclose(float(summary["best_validation_metrics"][key]), float(selected[key]), rel_tol=0.0, abs_tol=1e-10), f"Search best metric changed: {key}")
    resume = torch.load(paths["resume"], map_location="cpu", weights_only=False)
    require(resume.get("lock") == lock and int(resume.get("epoch", -1)) == MAX_EPOCHS, "Completed search resume changed")
    require(resume.get("optimizer_steps") == MAX_EPOCHS and resume.get("scheduler_steps") == MAX_EPOCHS, "Completed search counters changed")
    return summary


def run_search_trial(
    context: dict[str, Any],
    selection: dict[str, Any],
    runtime: dict[str, Any],
    fold: int,
    arm: str,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = "b0" if candidate is None else str(candidate["candidate_id"])
    paths = trial_paths(context["task_id"], fold, arm, candidate_id)
    if paths["complete"].is_file():
        return validate_search_trial(context, selection, runtime, fold, arm, candidate)
    paths["root"].mkdir(parents=True, exist_ok=True)
    known = {path.name for name, path in paths.items() if name != "root"}
    known |= {name + ".tmp" for name in known}
    require(not [path for path in paths["root"].iterdir() if path.name not in known], "Unknown incomplete search artifact")
    inner_train, inner_validation, _ = split_masks(context, selection, fold)
    rank = None if candidate is None else int(candidate["rank"])
    multiplier = None if candidate is None else float(candidate["adapter_lr_multiplier"])
    model, criterion, optimizer, scheduler, object_audit = make_objects(
        context, arm, inner_train, rank, multiplier, audit_common=False,
    )
    lock = trial_lock(runtime, context, selection, fold, arm, candidate)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    elapsed_before = 0.0
    resumed_from_epoch = 0
    if paths["resume"].is_file():
        resumed = load_search_resume(paths["resume"], lock, model, optimizer, scheduler)
        start_epoch = int(resumed["epoch"]) + 1
        resumed_from_epoch = int(resumed["epoch"])
        history = list(resumed["history"])
        best = copy.deepcopy(resumed["best"])
        elapsed_before = float(resumed.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        losses, _ = train_step(model, criterion, optimizer, context, inner_train)
        validation_metrics, _, _ = metrics_for(model, context, inner_validation)
        row = {
            "epoch": int(epoch),
            **losses,
            "correct": int(validation_metrics["correct"]),
            "acc": float(validation_metrics["acc"]),
            "bacc": float(validation_metrics["bacc"]),
            "roc_auc": float(validation_metrics["roc_auc"]),
            "pr_auc": float(validation_metrics["pr_auc"]),
            "macro_f1": float(validation_metrics["macro_f1"]),
            "sen": float(validation_metrics["sen"]),
            "spe": float(validation_metrics["spe"]),
        }
        history.append(row)
        if best is None or candidate_selection_key(context["task_id"], row, epoch) > tuple(best["selection_key"]):
            best = {
                "epoch": int(epoch),
                "selection_key": list(candidate_selection_key(context["task_id"], row, epoch)),
                "validation_metrics": copy.deepcopy(validation_metrics),
            }
        scheduler.step()
        if isinstance(scheduler, RatioScheduler):
            scheduler.assert_ratio()
        if epoch % 20 == 0 or epoch == MAX_EPOCHS:
            atomic_torch_save(
                paths["resume"],
                search_resume_payload(
                    lock, model, optimizer, scheduler, epoch, best, history,
                    elapsed_before + time.perf_counter() - started,
                ),
            )
    require(best is not None and len(history) == MAX_EPOCHS, "Search trial did not finish")
    independently_selected = select_history(context["task_id"], history)
    require(int(best["epoch"]) == int(independently_selected["epoch"]), "Search online/offline selection differs")
    atomic_write_csv(paths["history"], history)
    summary = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "fold": int(fold),
        "arm": arm,
        "candidate": copy.deepcopy(candidate),
        "best_epoch": int(best["epoch"]),
        "best_validation_metrics": best["validation_metrics"],
        "selection_key": best["selection_key"],
        "selection_order": "BACC>PR-AUC>Macro-F1>SEN>ACC>earlier" if context["task_id"] == "tadpole_smci_pmci" else "BACC>ROC-AUC>Macro-F1>ACC>earlier",
        "outer_test_used": False,
        "object_audit": object_audit,
        "elapsed_seconds": float(elapsed_before + time.perf_counter() - started),
        "resumed_from_epoch": resumed_from_epoch,
        "history_sha256": file_sha256(paths["history"]),
    }
    atomic_write_json(paths["summary"], summary)
    complete_core = {
        "complete": True,
        "lock": lock,
        "resume_sha256": file_sha256(paths["resume"]),
        "history_sha256": file_sha256(paths["history"]),
        "summary_sha256": file_sha256(paths["summary"]),
    }
    atomic_write_json(paths["complete"], {**complete_core, "sha256": payload_sha256(complete_core)})
    print(f"[search {context['task_id']} fold{fold:02d} {candidate_id}] best={best['epoch']} bacc={best['validation_metrics']['bacc']:.6f}")
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return validate_search_trial(context, selection, runtime, fold, arm, candidate)


def aggregate_search(protocols: dict[str, dict[str, Any]], selection: dict[str, Any], runtime: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id in TASK_IDS:
        context = cross.build_context(protocols[task_id], device)
        folds: list[dict[str, Any]] = []
        for fold in FOLDS:
            b0 = validate_search_trial(context, selection, runtime, fold, "B0", None)
            trials = [validate_search_trial(context, selection, runtime, fold, "B1", candidate) for candidate in CANDIDATES]
            selected_index, selected_trial = max(
                enumerate(trials),
                key=lambda pair: candidate_selection_key(
                    task_id,
                    pair[1]["best_validation_metrics"],
                    int(pair[1]["best_epoch"]),
                    pair[0],
                ),
            )
            folds.append(
                {
                    "fold": fold,
                    "selection_fold_sha256": selection["tasks"][task_id]["folds"][fold]["sha256"],
                    "b0": {
                        "best_epoch": b0["best_epoch"],
                        "validation_metrics": b0["best_validation_metrics"],
                        "elapsed_seconds": b0["elapsed_seconds"],
                    },
                    "b1_candidates": [
                        {
                            "candidate": trial["candidate"],
                            "best_epoch": trial["best_epoch"],
                            "validation_metrics": trial["best_validation_metrics"],
                            "selection_key": trial["selection_key"],
                            "elapsed_seconds": trial["elapsed_seconds"],
                        }
                        for trial in trials
                    ],
                    "selected_b1": {
                        "candidate": selected_trial["candidate"],
                        "best_epoch": selected_trial["best_epoch"],
                        "validation_metrics": selected_trial["best_validation_metrics"],
                        "candidate_order": int(selected_index),
                        "elapsed_seconds": selected_trial["elapsed_seconds"],
                    },
                    "outer_test_used": False,
                }
            )
        task_core = {
            "task_id": task_id,
            "folds": folds,
            "search_training_time_seconds": float(sum(
                fold_row["b0"]["elapsed_seconds"]
                + sum(candidate["elapsed_seconds"] for candidate in fold_row["b1_candidates"])
                for fold_row in folds
            )),
        }
        tasks[task_id] = {**task_core, "sha256": payload_sha256(task_core)}
    core = {
        "experiment": EXPERIMENT_ID,
        "runtime_lock": runtime,
        "selection_manifest_sha256": selection["sha256"],
        "candidate_grid": list(CANDIDATES),
        "tasks": tasks,
        "outer_test_used_for_selection": False,
    }
    return {**core, "sha256": payload_sha256(core)}


def validate_smoke(runtime: dict[str, Any]) -> None:
    config_path = SMOKE_DIR / "smoke_config.json"
    report_path = SMOKE_DIR / "smoke_report.json"
    require(config_path.is_file() and report_path.is_file(), "Smoke artifacts are missing")
    config, report = read_json(config_path), read_json(report_path)
    require(config.get("runtime_lock") == runtime and report.get("runtime_lock") == runtime, "Smoke source/config lock changed")
    require(report.get("passed") is True and report.get("loss_unit_check", {}).get("passed") is True, "Smoke did not pass")
    require(set(report.get("tasks", {})) == set(TASK_IDS), "Smoke task coverage changed")
    require(report.get("three_class_compatibility", {}).get("passed") is True, "Three-class compatibility smoke is missing")


def run_search(device_text: str, historical_root: Path) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Search requires cuda:0")
    source = source_gate(historical_root)
    protocols = cross.load_protocols()
    runtime = runtime_lock(protocols, source)
    validate_smoke(runtime)
    selection = read_json(SELECTION_PATH)
    for task_id in TASK_IDS:
        context = cross.build_context(protocols[task_id], torch.device(device_text))
        for fold in FOLDS:
            run_search_trial(context, selection, runtime, fold, "B0", None)
            for candidate in CANDIDATES:
                run_search_trial(context, selection, runtime, fold, "B1", candidate)
    results = aggregate_search(protocols, selection, runtime, torch.device(device_text))
    atomic_write_json(SEARCH_RESULTS_PATH, results)
    print(f"SEARCH COMPLETE sha256={results['sha256']}")


def run_smoke(device_text: str, historical_root: Path) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = source_gate(historical_root)
    protocols = cross.load_protocols()
    runtime = runtime_lock(protocols, source)
    selection = read_json(SELECTION_PATH)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    smoke_config = {
        "runtime_lock": runtime,
        "epochs": 3,
        "fold": 0,
        "candidate": CANDIDATES[0],
        "loss_formula": "main + 0.2 * mean(two OVR CE) + historical orthogonality",
        "training_mask": "inner_train_only",
    }
    atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    reports: dict[str, Any] = {}
    for task_index, task_id in enumerate(TASK_IDS):
        context = cross.build_context(protocols[task_id], torch.device(device_text))
        inner_train, _, outer_test = split_masks(context, selection, 0)
        candidate = CANDIDATES[0]
        model, criterion, optimizer, scheduler, audit = make_objects(
            context,
            "B1",
            inner_train,
            int(candidate["rank"]),
            float(candidate["adapter_lr_multiplier"]),
            audit_common=(task_index == 0),
        )
        model.eval()
        with torch.no_grad():
            initial_output, initial_embeddings, initial_auxiliary = model(context["dataset_data"]["Feature"])
        initial_b0_diff = None
        if task_index == 0:
            base_model, _, _, _, _ = make_objects(context, "B0", inner_train)
            base_model.eval()
            with torch.no_grad():
                base_output = base_model(context["dataset_data"]["Feature"])[0]
            initial_b0_diff = float((base_output - initial_output).abs().max().cpu())
            require(initial_b0_diff <= 1e-6, "Zero-output adapters changed initial logits")
            del base_model

        original_labels = context["dataset_data"]["Label"]
        permuted_labels = original_labels.clone()
        test_positions = torch.nonzero(outer_test, as_tuple=False).flatten()
        permuted_labels[test_positions] = 1 - permuted_labels[test_positions]
        original_components = criterion.components(initial_output, original_labels, inner_train, initial_embeddings, initial_auxiliary)
        permuted_components = criterion.components(initial_output, permuted_labels, inner_train, initial_embeddings, initial_auxiliary)
        test_label_loss_diff = float((original_components["total"] - permuted_components["total"]).abs().cpu())
        require(test_label_loss_diff == 0.0, "Outer-test labels entered smoke loss")
        formula_error = float(
            (
                original_components["total"]
                - (
                    original_components["main"]
                    + LAMBDA_AUX * original_components["auxiliary_mean"]
                    + float(context["protocol"]["training"]["loss_rate"]) * original_components["orthogonality"]
                )
            ).abs().cpu()
        )
        require(formula_error < 1e-7, "Smoke loss formula changed")

        initial_adapter = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
        cumulative = {name: 0.0 for name in initial_adapter}
        losses: list[dict[str, float]] = []
        lr_trace: list[list[float]] = []
        for _epoch in range(1, 4):
            loss_values, gradients = train_step(model, criterion, optimizer, context, inner_train)
            losses.append(loss_values)
            for name, value in gradients.items():
                cumulative[name] = max(cumulative[name], value)
            scheduler.step()
            scheduler.assert_ratio()
            lr_trace.append([float(group["lr"]) for group in optimizer.param_groups])
        parameter_delta = {
            name: float((parameter.detach().cpu() - initial_adapter[name]).abs().max())
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
        require(set(cumulative) == set(parameter_delta) and all(math.isfinite(value) and value > 0.0 for value in cumulative.values()), "Smoke adapter gradient inactive")
        require(all(math.isfinite(value) and value > 0.0 for value in parameter_delta.values()), "Smoke adapter parameter unchanged")
        model.eval()
        with torch.no_grad():
            logits = model(context["dataset_data"]["Feature"])[0][outer_test].detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
        simplex_error = float((probabilities.sum(dim=1) - 1.0).abs().max())
        require(simplex_error <= 1e-6, "Smoke probability simplex changed")
        checkpoint_path = SMOKE_DIR / f"checkpoint_roundtrip_{task_id}.pt"
        checkpoint = {
            "schema": 1,
            "runtime_lock": runtime,
            "task_id": task_id,
            "candidate": candidate,
            "epoch": 3,
            "model": clone_cpu_state(model),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": capture_rng_state(),
        }
        atomic_torch_save(checkpoint_path, checkpoint)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fresh, _, fresh_optimizer, fresh_scheduler, _ = make_objects(
            context, "B1", inner_train, int(candidate["rank"]), float(candidate["adapter_lr_multiplier"])
        )
        require(loaded["runtime_lock"] == runtime and loaded["task_id"] == task_id and loaded["epoch"] == 3, "Smoke checkpoint ownership changed")
        fresh.load_state_dict(loaded["model"], strict=True)
        fresh_optimizer.load_state_dict(loaded["optimizer"])
        fresh_scheduler.load_state_dict(loaded["scheduler"])
        require(int(fresh_scheduler.last_epoch) == 3, "Smoke scheduler state changed")
        fresh_scheduler.assert_ratio()
        fresh.eval()
        with torch.no_grad():
            reload_logits = fresh(context["dataset_data"]["Feature"])[0][outer_test].detach().cpu()
            reload_probabilities = torch.softmax(reload_logits, dim=-1)
        reload_diff = float((reload_logits - logits).abs().max())
        require(reload_diff <= 1e-6 and float((reload_probabilities - probabilities).abs().max()) <= 1e-6, "Smoke strict reload changed inference")
        reports[task_id] = {
            "passed": True,
            "object_audit": audit,
            "output_shape": [int(context["protocol"]["sample_count"]), 2],
            "losses": losses,
            "formula_absolute_error": formula_error,
            "outer_test_label_permutation_loss_max_diff": test_label_loss_diff,
            "outer_test_labels_used_by_loss": False,
            "adapter_cumulative_max_gradient": cumulative,
            "adapter_parameter_max_delta": parameter_delta,
            "lr_trace": lr_trace,
            "simplex_max_error": simplex_error,
            "strict_reload_logits_max_diff": reload_diff,
            "initial_b0_b1_logits_max_diff": initial_b0_diff,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "accuracy_evaluated": False,
        }
        del model, fresh, optimizer, fresh_optimizer, scheduler, fresh_scheduler
        torch.cuda.empty_cache()
    report_core = {
        "experiment": EXPERIMENT_ID,
        "passed": True,
        "runtime_lock": runtime,
        "loss_unit_check": corrected_loss_unit_check(),
        "three_class_compatibility": cross.three_class_compatibility_regression(device_text),
        "tasks": reports,
        "device": torch.cuda.get_device_name(torch.device(device_text)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    report = {**report_core, "sha256": payload_sha256(report_core)}
    atomic_write_json(SMOKE_DIR / "smoke_report.json", report)
    validate_smoke(runtime)
    print("SMOKE PASS")


def validate_search_results(
    protocols: dict[str, dict[str, Any]], selection: dict[str, Any], runtime: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    require(SEARCH_RESULTS_PATH.is_file(), "search_results.json missing; run search first")
    saved = read_json(SEARCH_RESULTS_PATH)
    rebuilt = aggregate_search(protocols, selection, runtime, device)
    require(saved == rebuilt, "Search results drifted")
    require(saved.get("outer_test_used_for_selection") is False, "Search used outer-test labels")
    return saved


def formal_spec(search_results: dict[str, Any], task_id: str, variant: str, fold: int) -> dict[str, Any]:
    require(variant in {"B0", "B1_Tuned", "B1_Transfer"}, "Unknown formal variant")
    if variant == "B0":
        selected = search_results["tasks"][task_id]["folds"][fold]["b0"]
        return {
            "variant": variant,
            "arm": "B0",
            "selection_source_task": task_id,
            "rank": None,
            "adapter_lr_multiplier": None,
            "training_epochs": int(selected["best_epoch"]),
            "inner_validation_metrics": selected["validation_metrics"],
        }
    source_task = "abide_ads_cn" if variant == "B1_Transfer" else task_id
    require(variant != "B1_Transfer" or task_id == "abide5_ads_cn", "Transfer is only defined for ABIDE-5")
    selected = search_results["tasks"][source_task]["folds"][fold]["selected_b1"]
    candidate = selected["candidate"]
    return {
        "variant": variant,
        "arm": "B1",
        "selection_source_task": source_task,
        "rank": int(candidate["rank"]),
        "adapter_lr_multiplier": float(candidate["adapter_lr_multiplier"]),
        "candidate_id": candidate["candidate_id"],
        "training_epochs": int(selected["best_epoch"]),
        "inner_validation_metrics": selected["validation_metrics"],
    }


def formal_lock(
    runtime: dict[str, Any],
    context: dict[str, Any],
    selection: dict[str, Any],
    search_results: dict[str, Any],
    variant: str,
    fold: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "runtime_lock_sha256": runtime["sha256"],
        "source_commit": runtime["source_commit"],
        "task_id": context["task_id"],
        "task_protocol_sha256": runtime["protocol_sha256"][context["task_id"]],
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "selection_fold_sha256": selection["tasks"][context["task_id"]]["folds"][fold]["sha256"],
        "search_results_sha256": search_results["sha256"],
        "variant": variant,
        "fold": int(fold),
        "spec": copy.deepcopy(spec),
        "outer_test_used_for_training_or_selection": False,
    }
    return {**core, "sha256": payload_sha256(core)}


def formal_paths(task_id: str, variant: str, fold: int) -> dict[str, Path]:
    root = FORMAL_DIR / task_id / variant / f"fold_{fold:02d}"
    compact_root = FOLD_RESULT_DIR / task_id / variant / f"fold_{fold:02d}"
    return {
        "root": root,
        "resume": root / "resume.pt",
        "checkpoint": root / "checkpoint_final.pt",
        "oof": root / "oof_predictions.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
        "compact_root": compact_root,
        "compact_summary": compact_root / "summary.json",
        "compact_complete": compact_root / "COMPLETE.json",
    }


def formal_resume_payload(
    lock: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    cumulative_gradient: dict[str, float],
    initial_adapter_state: dict[str, torch.Tensor],
    loss_tail: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "lock": lock,
        "epoch": int(epoch),
        "optimizer_steps": int(epoch),
        "scheduler_steps": int(epoch),
        "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
        "cumulative_gradient": dict(cumulative_gradient),
        "initial_adapter_state": {name: tensor.detach().cpu().clone() for name, tensor in initial_adapter_state.items()},
        "loss_tail": copy.deepcopy(loss_tail[-20:]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def load_formal_resume(
    path: Path, lock: dict[str, Any], target_epoch: int, model: nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("schema") == 1 and payload.get("lock") == lock, "Formal resume lock changed")
    epoch = int(payload.get("epoch", -1))
    require(0 < epoch <= target_epoch, "Formal resume epoch changed")
    require(payload.get("optimizer_steps") == epoch and payload.get("scheduler_steps") == epoch, "Formal resume counters changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    require(int(scheduler.last_epoch) == epoch, "Formal resume scheduler epoch changed")
    if isinstance(scheduler, RatioScheduler):
        scheduler.assert_ratio()
    restore_rng_state(payload["rng"])
    return payload


def private_off_inference(model: nn.Module, context: dict[str, Any], test_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require(hasattr(model, "private_adapters"), "Private-off diagnostic requires B1")
    handles = []
    for adapter in model.private_adapters:
        handles.append(adapter.register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output)))
    try:
        model.eval()
        with torch.no_grad():
            logits = model(context["dataset_data"]["Feature"])[0][test_mask]
            probabilities = torch.softmax(logits, dim=-1)
    finally:
        for handle in handles:
            handle.remove()
    return logits.detach().cpu(), probabilities.detach().cpu()


def private_fold_diagnostics(
    context: dict[str, Any], model: nn.Module, test_mask: torch.Tensor,
    normal_probabilities: torch.Tensor, cumulative_gradient: dict[str, float],
    initial_adapter_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, _, _, intermediates = model(context["dataset_data"]["Feature"], return_intermediates=True)
    residuals = intermediates["private_residuals"][test_mask]
    shared = intermediates["modal_tokens_post_transformer"][test_mask]
    ratio = residuals.norm(dim=-1) / shared.norm(dim=-1).clamp_min(1e-8)
    cosine = F.cosine_similarity(intermediates["Y"][test_mask], intermediates["G"][test_mask], dim=-1)
    modalities = [entry["name"] for entry in context["protocol"]["modalities"]]
    parameter_delta = {
        name: float((parameter.detach().cpu() - initial_adapter_state[name]).abs().max())
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    off_logits, off_probabilities = private_off_inference(model, context, test_mask)
    truth = context["dataset_data"]["Label"][test_mask].detach().cpu().numpy().astype(int)
    normal_prediction = normal_probabilities.numpy().argmax(axis=1)
    off_prediction = off_probabilities.numpy().argmax(axis=1)
    normal_correct = normal_prediction == truth
    off_correct = off_prediction == truth
    active = bool(
        cumulative_gradient
        and max(cumulative_gradient.values(), default=0.0) > 0.0
        and max(parameter_delta.values(), default=0.0) > 0.0
        and float(residuals.norm(dim=-1).max().cpu()) >= 1e-6
    )
    return {
        "private_shared_ratio_mean_by_modality": dict(zip(modalities, ratio.mean(dim=0).detach().cpu().tolist())),
        "private_shared_ratio_max_by_modality": dict(zip(modalities, ratio.amax(dim=0).detach().cpu().tolist())),
        "private_shared_ratio_mean": float(ratio.mean().cpu()),
        "private_shared_ratio_max": float(ratio.max().cpu()),
        "category_global_cosine_mean": float(cosine.mean().cpu()),
        "adapter_cumulative_max_gradient": cumulative_gradient,
        "adapter_parameter_max_delta": parameter_delta,
        "adapter_max_gradient": max(cumulative_gradient.values(), default=0.0),
        "private_active": active,
        "private_collapse": not active,
        "private_off": {
            "probability_max_abs_difference": float((normal_probabilities - off_probabilities).abs().max()),
            "argmax_changed": int((normal_prediction != off_prediction).sum()),
            "direct_repairs": int((~off_correct & normal_correct).sum()),
            "direct_damages": int((off_correct & ~normal_correct).sum()),
            "off_logits_shape": list(off_logits.shape),
        },
    }


def formal_prediction_rows(
    context: dict[str, Any], variant: str, fold: int, logits: torch.Tensor, probabilities: torch.Tensor
) -> list[dict[str, Any]]:
    anchors = context["fold_manifest"]["folds"][fold]["test_rows"]
    truth = context["dataset_data"]["Label"][cross.fold_positions(context, fold)[1]].detach().cpu().numpy().astype(int)
    require(len(anchors) == len(truth) == logits.shape[0] == probabilities.shape[0], "Formal OOF count changed")
    prediction = probabilities.numpy().argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        require(int(anchor["truth"]) == int(truth[index]), "Formal OOF truth alignment changed")
        rows.append(
            {
                "task_id": context["task_id"],
                "variant": variant,
                "fold": int(fold),
                "subject_id": anchor["subject_id"],
                "original_csv_index": int(anchor["original_csv_index"]),
                "feature_sha256": anchor["feature_sha256"],
                "truth": int(truth[index]),
                "prediction": int(prediction[index]),
                "logit_0": float(logits[index, 0]),
                "logit_1": float(logits[index, 1]),
                "probability_0": float(probabilities[index, 0]),
                "probability_1": float(probabilities[index, 1]),
                "positive_probability": float(probabilities[index, int(context["protocol"]["positive_index"])]),
            }
        )
    return rows


def typed_formal_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    for row in rows:
        typed.append(
            {
                "task_id": row["task_id"], "variant": row["variant"], "fold": int(row["fold"]),
                "subject_id": row["subject_id"], "original_csv_index": int(row["original_csv_index"]),
                "feature_sha256": row["feature_sha256"], "truth": int(row["truth"]),
                "prediction": int(row["prediction"]), "logit_0": float(row["logit_0"]),
                "logit_1": float(row["logit_1"]), "probability_0": float(row["probability_0"]),
                "probability_1": float(row["probability_1"]), "positive_probability": float(row["positive_probability"]),
            }
        )
    return typed


def formal_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray([row["truth"] for row in rows], dtype=int)
    logits = np.asarray([[row["logit_0"], row["logit_1"]] for row in rows], dtype=np.float64)
    probabilities = np.asarray([[row["probability_0"], row["probability_1"]] for row in rows], dtype=np.float64)
    return truth, logits, probabilities


def validate_formal_fold(
    context: dict[str, Any], selection: dict[str, Any], search_results: dict[str, Any],
    runtime: dict[str, Any], variant: str, fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = formal_spec(search_results, context["task_id"], variant, fold)
    lock = formal_lock(runtime, context, selection, search_results, variant, fold, spec)
    paths = formal_paths(context["task_id"], variant, fold)
    require(all(paths[name].is_file() for name in ("resume", "checkpoint", "oof", "summary", "complete")), "Completed formal fold is incomplete")
    marker = read_json(paths["complete"])
    marker_core = {key: value for key, value in marker.items() if key != "sha256"}
    require(marker.get("sha256") == payload_sha256(marker_core), "Formal complete digest changed")
    require(marker.get("complete") is True and marker.get("lock") == lock, "Formal complete lock changed")
    for field, name in (("resume_sha256", "resume"), ("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("summary_sha256", "summary")):
        require(marker[field] == file_sha256(paths[name]), f"Formal {name} changed")
    summary = read_json(paths["summary"])
    require(summary.get("lock") == lock and summary.get("spec") == spec, "Formal summary lock/spec changed")
    require(summary.get("task_id") == context["task_id"] and summary.get("variant") == variant and int(summary.get("fold", -1)) == fold, "Formal summary ownership changed")
    require(summary.get("outer_test_used_for_training_or_selection") is False and summary.get("outer_test_inference_after_training") is True, "Outer-test protocol changed")
    rows = typed_formal_rows(read_csv_rows(paths["oof"]))
    anchors = context["fold_manifest"]["folds"][fold]["test_rows"]
    require(len(rows) == len(anchors), "Formal fold OOF count changed")
    for row, anchor in zip(rows, anchors):
        require(row["task_id"] == context["task_id"] and row["variant"] == variant and row["fold"] == fold, "Formal OOF ownership changed")
        require(row["subject_id"] == anchor["subject_id"] and row["original_csv_index"] == anchor["original_csv_index"], "Formal OOF stable ID changed")
        require(row["feature_sha256"] == anchor["feature_sha256"] and row["truth"] == anchor["truth"], "Formal OOF feature/truth changed")
        require(row["prediction"] == int(np.argmax([row["probability_0"], row["probability_1"]])), "Formal OOF prediction changed")
    truth, saved_logits, saved_probabilities = formal_arrays(rows)
    recomputed = cross.binary_metrics(truth, saved_probabilities, context["protocol"])
    require(cross.metrics_match(recomputed, summary["metrics"]), "Formal OOF metrics changed")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    target_epoch = int(spec["training_epochs"])
    require(checkpoint.get("schema") == 1 and checkpoint.get("lock") == lock, "Formal checkpoint lock changed")
    require(checkpoint.get("epoch") == target_epoch and checkpoint.get("optimizer_steps") == target_epoch and checkpoint.get("scheduler_steps") == target_epoch, "Formal checkpoint step counts changed")
    outer_train, outer_test, _ = cross.fold_positions(context, fold)
    model, _, optimizer, scheduler, _ = make_objects(
        context, spec["arm"], outer_train, spec.get("rank"), spec.get("adapter_lr_multiplier")
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    require(int(scheduler.last_epoch) == target_epoch, "Formal checkpoint scheduler epoch changed")
    if isinstance(scheduler, RatioScheduler):
        scheduler.assert_ratio()
    model.eval()
    with torch.no_grad():
        replay_logits = model(context["dataset_data"]["Feature"])[0][outer_test].detach().cpu().numpy().astype(np.float64)
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - saved_logits))) <= 1e-6, "Formal checkpoint logits do not replay OOF")
    require(float(np.max(np.abs(replay_probabilities - saved_probabilities))) <= 1e-6, "Formal checkpoint probabilities do not replay OOF")
    if spec["arm"] == "B1":
        diagnostics = summary.get("diagnostics", {})
        require(diagnostics.get("private_active") is True and diagnostics.get("private_collapse") is False, "Formal private adapter is inactive")
    compact_summary = {**summary, "formal_summary_sha256": file_sha256(paths["summary"]), "formal_complete_sha256": file_sha256(paths["complete"])}
    paths["compact_root"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["compact_summary"], compact_summary)
    compact_core = {"complete": True, "lock": lock, "summary_sha256": file_sha256(paths["compact_summary"])}
    atomic_write_json(paths["compact_complete"], {**compact_core, "sha256": payload_sha256(compact_core)})
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def run_formal_fold(
    context: dict[str, Any], selection: dict[str, Any], search_results: dict[str, Any],
    runtime: dict[str, Any], variant: str, fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = formal_paths(context["task_id"], variant, fold)
    if paths["complete"].is_file():
        return validate_formal_fold(context, selection, search_results, runtime, variant, fold)
    paths["root"].mkdir(parents=True, exist_ok=True)
    known = {path.name for name, path in paths.items() if name not in {"root", "compact_root", "compact_summary", "compact_complete"}}
    known |= {name + ".tmp" for name in known}
    require(not [path for path in paths["root"].iterdir() if path.name not in known], "Unknown incomplete formal artifact")
    spec = formal_spec(search_results, context["task_id"], variant, fold)
    lock = formal_lock(runtime, context, selection, search_results, variant, fold, spec)
    outer_train, outer_test, _ = cross.fold_positions(context, fold)
    model, criterion, optimizer, scheduler, object_audit = make_objects(
        context, spec["arm"], outer_train, spec.get("rank"), spec.get("adapter_lr_multiplier"), audit_common=False,
    )
    initial_adapter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    cumulative_gradient = {name: 0.0 for name in initial_adapter_state}
    target_epoch = int(spec["training_epochs"])
    require(1 <= target_epoch <= MAX_EPOCHS, "Selected formal epoch is invalid")
    start_epoch = 1
    elapsed_before = 0.0
    resumed_from_epoch = 0
    loss_tail: list[dict[str, Any]] = []
    if paths["resume"].is_file():
        resumed = load_formal_resume(paths["resume"], lock, target_epoch, model, optimizer, scheduler)
        start_epoch = int(resumed["epoch"]) + 1
        resumed_from_epoch = int(resumed["epoch"])
        cumulative_gradient = {name: float(value) for name, value in resumed["cumulative_gradient"].items()}
        initial_adapter_state = {name: tensor.detach().cpu().clone() for name, tensor in resumed["initial_adapter_state"].items()}
        loss_tail = list(resumed.get("loss_tail", []))
        elapsed_before = float(resumed.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    for epoch in range(start_epoch, target_epoch + 1):
        losses, gradients = train_step(model, criterion, optimizer, context, outer_train)
        for name, value in gradients.items():
            cumulative_gradient[name] = max(cumulative_gradient.get(name, 0.0), float(value))
        loss_tail.append({"epoch": epoch, **losses})
        loss_tail = loss_tail[-20:]
        scheduler.step()
        if isinstance(scheduler, RatioScheduler):
            scheduler.assert_ratio()
        if epoch % 20 == 0 or epoch == target_epoch:
            atomic_torch_save(
                paths["resume"],
                formal_resume_payload(
                    lock, model, optimizer, scheduler, epoch, cumulative_gradient,
                    initial_adapter_state, loss_tail, elapsed_before + time.perf_counter() - started,
                ),
            )
    require(int(scheduler.last_epoch) == target_epoch, "Formal training did not stop at selected epoch")
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    inference_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        full_logits = model(context["dataset_data"]["Feature"])[0]
        test_logits = full_logits[outer_test].detach().cpu()
        test_probabilities = torch.softmax(test_logits, dim=-1)
    if context["device"].type == "cuda":
        torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - inference_started
    rows = formal_prediction_rows(context, variant, fold, test_logits, test_probabilities)
    truth, _, probability_array = formal_arrays(rows)
    metrics = cross.binary_metrics(truth, probability_array, context["protocol"])
    diagnostics = None
    if spec["arm"] == "B1":
        diagnostics = private_fold_diagnostics(
            context, model, outer_test, test_probabilities, cumulative_gradient, initial_adapter_state
        )
        require(diagnostics["private_active"] is True, "Selected private adapter did not activate")
    elapsed_total = elapsed_before + time.perf_counter() - started
    checkpoint = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "variant": variant,
        "fold": int(fold),
        "spec": spec,
        "epoch": target_epoch,
        "optimizer_steps": target_epoch,
        "scheduler_steps": target_epoch,
        "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
    }
    atomic_torch_save(paths["checkpoint"], checkpoint)
    atomic_write_csv(paths["oof"], rows)
    summary = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "variant": variant,
        "fold": int(fold),
        "spec": spec,
        "metrics": metrics,
        "object_audit": object_audit,
        "diagnostics": diagnostics,
        "training_epochs": target_epoch,
        "optimizer_steps": target_epoch,
        "scheduler_steps": target_epoch,
        "resumed_from_epoch": resumed_from_epoch,
        "loss_tail": loss_tail,
        "training_time_seconds": float(elapsed_total),
        "inference_time_seconds_full_graph": float(inference_seconds),
        "outer_test_used_for_training_or_selection": False,
        "outer_test_inference_after_training": True,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
    }
    atomic_write_json(paths["summary"], summary)
    complete_core = {
        "complete": True, "lock": lock,
        "resume_sha256": file_sha256(paths["resume"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "summary_sha256": file_sha256(paths["summary"]),
    }
    atomic_write_json(paths["complete"], {**complete_core, "sha256": payload_sha256(complete_core)})
    print(f"[formal {context['task_id']} {variant} fold{fold:02d}] epochs={target_epoch} correct={metrics['correct']}/{len(rows)}")
    return validate_formal_fold(context, selection, search_results, runtime, variant, fold)


def aggregate_variant(
    context: dict[str, Any], variant: str,
    fold_summaries: list[dict[str, Any]], fold_rows: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(fold_summaries) == len(fold_rows) == 10, "Formal fold count changed")
    rows = sorted([row for group in fold_rows for row in group], key=lambda row: row["original_csv_index"])
    expected = sorted(
        [row for fold in context["fold_manifest"]["folds"] for row in fold["test_rows"]],
        key=lambda row: row["original_csv_index"],
    )
    require(len(rows) == len(expected) == context["protocol"]["sample_count"], "Formal OOF coverage changed")
    require(len({row["subject_id"] for row in rows}) == len(rows), "Formal OOF subject IDs are not unique")
    for row, anchor in zip(rows, expected):
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"] and row["feature_sha256"] == anchor["feature_sha256"], "Formal OOF alignment changed")
    truth, _, probabilities = formal_arrays(rows)
    metrics = cross.binary_metrics(truth, probabilities, context["protocol"])
    fold_metrics = [
        {
            "fold": int(summary["fold"]),
            "training_epochs": int(summary["training_epochs"]),
            "correct": int(summary["metrics"]["correct"]),
            "acc": float(summary["metrics"]["acc"]),
            "bacc": float(summary["metrics"]["bacc"]),
            "roc_auc": float(summary["metrics"]["roc_auc"]),
            "pr_auc": float(summary["metrics"]["pr_auc"]),
            "macro_f1": float(summary["metrics"]["macro_f1"]),
            "sen": float(summary["metrics"]["sen"]),
            "spe": float(summary["metrics"]["spe"]),
            "training_time_seconds": float(summary["training_time_seconds"]),
            "inference_time_seconds_full_graph": float(summary["inference_time_seconds_full_graph"]),
            "resumed_from_epoch": int(summary["resumed_from_epoch"]),
            "rank": summary["spec"].get("rank"),
            "adapter_lr_multiplier": summary["spec"].get("adapter_lr_multiplier"),
            "selection_source_task": summary["spec"]["selection_source_task"],
            "parameter_count": int(summary["object_audit"]["parameter_count"]),
        }
        for summary in fold_summaries
    ]
    require([row["fold"] for row in fold_metrics] == list(FOLDS), "Formal fold order changed")
    output_path = RESULT_DIR / f"{context['task_id']}_{variant}_oof_predictions.csv"
    atomic_write_csv(output_path, rows)
    configuration_frequency: dict[str, int] = {}
    for row in fold_metrics:
        if row["rank"] is not None:
            key = f"rank={row['rank']},multiplier={row['adapter_lr_multiplier']}"
            configuration_frequency[key] = configuration_frequency.get(key, 0) + 1
    result = {
        "task_id": context["task_id"],
        "variant": variant,
        "metrics": metrics,
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)),
        "fold_acc_sample_std": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_metrics": fold_metrics,
        "selected_epochs": [row["training_epochs"] for row in fold_metrics],
        "configuration_frequency": configuration_frequency,
        "parameter_count_by_fold": [row["parameter_count"] for row in fold_metrics],
        "training_time_seconds": float(sum(row["training_time_seconds"] for row in fold_metrics)),
        "inference_time_seconds_full_graph": float(sum(row["inference_time_seconds_full_graph"] for row in fold_metrics)),
        "oof_path": output_path.relative_to(ROOT).as_posix(),
        "oof_sha256": file_sha256(output_path),
    }
    if variant != "B0":
        diagnostics = [summary["diagnostics"] for summary in fold_summaries]
        modalities = [entry["name"] for entry in context["protocol"]["modalities"]]
        result["mechanism"] = {
            "private_shared_ratio_mean_by_modality": {
                modality: float(statistics.mean(item["private_shared_ratio_mean_by_modality"][modality] for item in diagnostics))
                for modality in modalities
            },
            "private_shared_ratio_max_by_modality": {
                modality: float(max(item["private_shared_ratio_max_by_modality"][modality] for item in diagnostics))
                for modality in modalities
            },
            "private_shared_ratio_mean": float(statistics.mean(item["private_shared_ratio_mean"] for item in diagnostics)),
            "private_shared_ratio_max": float(max(item["private_shared_ratio_max"] for item in diagnostics)),
            "category_global_cosine_mean": float(statistics.mean(item["category_global_cosine_mean"] for item in diagnostics)),
            "adapter_max_gradient": float(max(item["adapter_max_gradient"] for item in diagnostics)),
            "all_folds_private_active": all(item["private_active"] for item in diagnostics),
            "any_private_collapse": any(item["private_collapse"] for item in diagnostics),
            "private_off_probability_max_abs_difference": float(max(item["private_off"]["probability_max_abs_difference"] for item in diagnostics)),
            "private_off_argmax_changed": int(sum(item["private_off"]["argmax_changed"] for item in diagnostics)),
            "private_off_direct_repairs": int(sum(item["private_off"]["direct_repairs"] for item in diagnostics)),
            "private_off_direct_damages": int(sum(item["private_off"]["direct_damages"] for item in diagnostics)),
        }
    return result, rows


def exact_mcnemar_p(repairs: int, damages: int) -> float:
    discordant = int(repairs + damages)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(repairs, damages) + 1)) / (2.0 ** discordant)
    return float(min(1.0, 2.0 * tail))


def paired_result(
    context: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any],
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    left = {row["subject_id"]: row for row in baseline_rows}
    right = {row["subject_id"]: row for row in candidate_rows}
    require(set(left) == set(right), "Paired OOF subject sets changed")
    repairs = damages = changed = 0
    for subject_id in left:
        old, new = left[subject_id], right[subject_id]
        require(old["truth"] == new["truth"] and old["feature_sha256"] == new["feature_sha256"], "Paired truth/features changed")
        old_correct = old["prediction"] == old["truth"]
        new_correct = new["prediction"] == new["truth"]
        repairs += int(not old_correct and new_correct)
        damages += int(old_correct and not new_correct)
        changed += int(old["prediction"] != new["prediction"])
    deltas = {
        key: float(candidate["metrics"][key] - baseline["metrics"][key])
        for key in ("acc", "bacc", "roc_auc", "pr_auc", "macro_f1", "weighted_f1", "sen", "spe")
    }
    return {
        "task_id": context["task_id"],
        "baseline": baseline["variant"],
        "candidate": candidate["variant"],
        "repairs": repairs,
        "damages": damages,
        "changed": changed,
        "net_correct": repairs - damages,
        "correct_delta": int(candidate["metrics"]["correct"] - baseline["metrics"]["correct"]),
        "metric_deltas": deltas,
        "positive_tp_delta": int(candidate["metrics"]["tp"] - baseline["metrics"]["tp"]),
        "positive_fn_delta": int(candidate["metrics"]["fn"] - baseline["metrics"]["fn"]),
        "exact_mcnemar_p": exact_mcnemar_p(repairs, damages),
    }


def class_collapse(metrics: dict[str, Any]) -> bool:
    predicted = list(metrics["predicted_counts"].values())
    true_counts = [0, 0]
    true_counts[int(metrics["positive_index"])] = int(metrics["tp"] + metrics["fn"])
    true_counts[int(metrics["negative_index"])] = int(metrics["tn"] + metrics["fp"])
    return bool(
        any(predicted[index] < 0.25 * true_counts[index] for index in range(2))
        or min(float(metrics["sen"]), float(metrics["spe"])) < 0.50
        or abs(float(metrics["sen"]) - float(metrics["spe"])) > 0.35
    )


def sen_spe_exchange(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    sen_delta = float(candidate["sen"] - baseline["sen"])
    spe_delta = float(candidate["spe"] - baseline["spe"])
    old_gap = abs(float(baseline["sen"] - baseline["spe"]))
    new_gap = abs(float(candidate["sen"] - candidate["spe"]))
    return bool(sen_delta * spe_delta < 0.0 and min(sen_delta, spe_delta) < -0.05 and new_gap > old_gap + 0.05)


def final_decision(task_results: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    tad = comparisons["tadpole_smci_pmci"]["B1_Tuned_vs_B0"]
    abide = comparisons["abide_ads_cn"]["B1_Tuned_vs_B0"]
    transfer = comparisons["abide5_ads_cn"]["B1_Transfer_vs_B0"]
    tuned5 = comparisons["abide5_ads_cn"]["B1_Tuned_vs_B0"]
    tad0 = task_results["tadpole_smci_pmci"]["B0"]["metrics"]
    tad1 = task_results["tadpole_smci_pmci"]["B1_Tuned"]["metrics"]
    ab0 = task_results["abide_ads_cn"]["B0"]["metrics"]
    ab1 = task_results["abide_ads_cn"]["B1_Tuned"]["metrics"]
    tr1 = task_results["abide5_ads_cn"]["B1_Transfer"]["metrics"]
    all_private = [
        task_results["tadpole_smci_pmci"]["B1_Tuned"],
        task_results["abide_ads_cn"]["B1_Tuned"],
        task_results["abide5_ads_cn"]["B1_Transfer"],
        task_results["abide5_ads_cn"]["B1_Tuned"],
    ]
    no_collapse = all(not class_collapse(result["metrics"]) and not result["mechanism"]["any_private_collapse"] for result in all_private)
    all_active = all(result["mechanism"]["all_folds_private_active"] for result in all_private)
    no_exchange = not sen_spe_exchange(ab0, ab1) and not sen_spe_exchange(task_results["abide5_ads_cn"]["B0"]["metrics"], tr1)
    go_gates = {
        "tad_correct_not_lower": tad["correct_delta"] >= 0,
        "tad_bacc_not_lower": tad["metric_deltas"]["bacc"] >= 0.0,
        "tad_pr_auc_not_lower": tad["metric_deltas"]["pr_auc"] >= 0.0,
        "tad_positive_tp_at_least_plus_2": tad["positive_tp_delta"] >= 2,
        "tad_roc_auc_drop_at_most_0p02": tad["metric_deltas"]["roc_auc"] >= -0.02,
        "abide_correct_not_lower": abide["correct_delta"] >= 0,
        "abide_repairs_greater_than_damages": abide["repairs"] > abide["damages"],
        "abide_bacc_drop_at_most_0p002": abide["metric_deltas"]["bacc"] >= -0.002,
        "abide_roc_auc_drop_at_most_0p005": abide["metric_deltas"]["roc_auc"] >= -0.005,
        "no_sen_spe_exchange_or_collapse": no_exchange and no_collapse,
        "abide5_transfer_correct_drop_at_most_1": transfer["correct_delta"] >= -1,
        "abide5_transfer_bacc_drop_at_most_0p003": transfer["metric_deltas"]["bacc"] >= -0.003,
    }
    tad_clear = bool(tad["correct_delta"] > 0 and tad["metric_deltas"]["bacc"] > 0 and tad["metric_deltas"]["pr_auc"] > 0 and tad["positive_tp_delta"] >= 2)
    abide_improves = bool(abide["correct_delta"] > 0 and abide["repairs"] > abide["damages"] and abide["metric_deltas"]["bacc"] >= 0)
    transfer_improves = bool(transfer["correct_delta"] > 0 and transfer["repairs"] > transfer["damages"] and transfer["metric_deltas"]["bacc"] >= 0)
    tuned5_improves = bool(tuned5["correct_delta"] > 0 and tuned5["repairs"] > tuned5["damages"] and tuned5["metric_deltas"]["bacc"] >= 0)
    severe = any(
        comparison["metric_deltas"]["bacc"] < -0.02 or comparison["correct_delta"] <= -10
        for comparison in (tad, abide, transfer, tuned5)
    )
    stop_gates = {
        "tad_and_abide_both_no_gain": tad["correct_delta"] <= 0 and abide["correct_delta"] <= 0,
        "tad_and_abide_repairs_not_above_damages": tad["repairs"] <= tad["damages"] and abide["repairs"] <= abide["damages"],
        "class_or_private_collapse": not no_collapse,
        "adapter_inactive": not all_active,
        "severe_degradation": severe,
        "protocol_leakage": False,
    }
    if all(go_gates.values()) and all_active:
        decision = "TASK_ADAPTATION_GO"
        conclusion = "Task-calibrated private residual is cross-task usable; SP-LRIF may be considered next, but is not run here."
    elif any(stop_gates.values()):
        decision = "PRIVATE_RESIDUAL_STOP"
        conclusion = "Stop the cross-dataset private-residual route."
    elif tad_clear and not abide_improves and all_active:
        decision = "TASK_SPECIFIC_ONLY"
        conclusion = "Keep private residual only for the progressing TADPOLE task; retain Original for ABIDE."
    elif (abide_improves and not tad_clear) or (tuned5_improves and not transfer_improves):
        decision = "ABIDE_ONLY_MIXED"
        conclusion = "The structure depends on task-specific tuning and lacks natural transfer; do not enter unified fusion."
    else:
        decision = "PRIVATE_RESIDUAL_STOP"
        conclusion = "No pre-registered positive category was met; stop the cross-dataset private-residual route."
    return {
        "decision": decision,
        "conclusion": conclusion,
        "go_gates": go_gates,
        "stop_gates": stop_gates,
        "derived": {
            "tad_clear_improvement": tad_clear,
            "abide_improves": abide_improves,
            "abide5_transfer_improves": transfer_improves,
            "abide5_tuned_improves": tuned5_improves,
            "all_private_adapters_active": all_active,
            "no_collapse": no_collapse,
            "no_sen_spe_exchange": no_exchange,
        },
    }


def most_selected(result: dict[str, Any]) -> list[str]:
    frequency = result.get("configuration_frequency", {})
    if not frequency:
        return []
    maximum = max(frequency.values())
    return sorted(key for key, value in frequency.items() if value == maximum)


def metric_line(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"{metrics['correct']}/{metrics['n']} | {metrics['acc']:.6f} | {metrics['bacc']:.6f} | "
        f"{metrics['macro_f1']:.6f} | {metrics['roc_auc']:.6f} | {metrics['pr_auc']:.6f} | "
        f"{metrics['sen']:.6f} | {metrics['spe']:.6f} | {metrics['weighted_f1']:.6f} | "
        f"{metrics['confusion_matrix']} | {metrics['predicted_counts']}"
    )


def make_report(
    summary_core: dict[str, Any], historical_root: Path, source_commit: str,
) -> str:
    tasks = summary_core["task_results"]
    comparisons = summary_core["comparisons"]
    decision = summary_core["decision"]
    legacy = summary_core["legacy_reference"]
    tad = tasks["tadpole_smci_pmci"]
    abide = tasks["abide_ads_cn"]
    abide5 = tasks["abide5_ads_cn"]
    tad_cmp = comparisons["tadpole_smci_pmci"]["B1_Tuned_vs_B0"]
    abide_cmp = comparisons["abide_ads_cn"]["B1_Tuned_vs_B0"]
    transfer_cmp = comparisons["abide5_ads_cn"]["B1_Transfer_vs_B0"]
    tuned_cmp = comparisons["abide5_ads_cn"]["B1_Tuned_vs_B0"]
    tuned_transfer = comparisons["abide5_ads_cn"]["B1_Tuned_vs_B1_Transfer"]
    lines = [
        "# Binary Task-Adaptive Private Residual Calibration v1",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        decision["conclusion"],
        "",
        "## Stage 0: corrected binary objective",
        "",
        "The historical `criterion_lossv2` accumulated both OVR CE terms into an unnormalized sum and did not consume the declared ABIDE `0.2/mean` setting. This experiment leaves historical runners untouched and uses `L_main + 0.2 * mean(L_ovr0,L_ovr1) + historical_orthogonality` in its own runner. Class weights are computed only from the current training mask. The deterministic float64 unit check has absolute error below 1e-8.",
        "",
        "## Formal results",
        "",
        "| Task / variant | Correct | ACC | BACC | Macro-F1 | ROC-AUC | PR-AUC | SEN | SPE | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id, variants in tasks.items():
        for variant, result in variants.items():
            m = result["metrics"]
            lines.append(
                f"| {task_id} / {variant} | {m['correct']}/{m['n']} | {m['acc']:.7f} | {m['bacc']:.7f} | "
                f"{m['macro_f1']:.7f} | {m['roc_auc']:.7f} | {m['pr_auc']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {m['weighted_f1']:.7f} |"
            )
    lines += ["", "Confusion matrices, predicted counts, fold statistics, selected epochs, parameters and timing:", ""]
    for task_id, variants in tasks.items():
        for variant, result in variants.items():
            lines += [
                f"- `{task_id}/{variant}`: confusion={result['metrics']['confusion_matrix']}; predicted={result['metrics']['predicted_counts']}; "
                f"fold ACC={result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f}; epochs={result['selected_epochs']}; "
                f"params={result['parameter_count_by_fold']}; train={result['training_time_seconds']:.3f}s; inference={result['inference_time_seconds_full_graph']:.6f}s."
            ]
    lines += ["", "## Paired corrected-loss comparisons", ""]
    for task_id, task_comparisons in comparisons.items():
        for name, item in task_comparisons.items():
            lines.append(
                f"- `{task_id}/{name}`: Correct delta={item['correct_delta']:+d}; repairs/damages/changed="
                f"{item['repairs']}/{item['damages']}/{item['changed']}; BACC={item['metric_deltas']['bacc']:+.7f}; "
                f"Macro-F1={item['metric_deltas']['macro_f1']:+.7f}; ROC-AUC={item['metric_deltas']['roc_auc']:+.7f}; "
                f"PR-AUC={item['metric_deltas']['pr_auc']:+.7f}; exact McNemar p={item['exact_mcnemar_p']:.8g}."
            )
    lines += ["", "## Nested selections and mechanisms", ""]
    for task_id, variants in tasks.items():
        for variant, result in variants.items():
            if variant == "B0":
                continue
            mechanism = result["mechanism"]
            lines.append(
                f"- `{task_id}/{variant}`: frequency={result['configuration_frequency']}; ratio mean/max="
                f"{mechanism['private_shared_ratio_mean']:.7f}/{mechanism['private_shared_ratio_max']:.7f}; "
                f"per-modality mean={mechanism['private_shared_ratio_mean_by_modality']}; per-modality max={mechanism['private_shared_ratio_max_by_modality']}; "
                f"Category-Global cosine={mechanism['category_global_cosine_mean']:.7f}; max adapter grad={mechanism['adapter_max_gradient']:.7g}; "
                f"private-off probability diff={mechanism['private_off_probability_max_abs_difference']:.7g}; "
                f"direct changed/repairs/damages={mechanism['private_off_argmax_changed']}/{mechanism['private_off_direct_repairs']}/{mechanism['private_off_direct_damages']}; "
                f"active={mechanism['all_folds_private_active']}; collapse={mechanism['any_private_collapse']}."
            )
    legacy_tasks = legacy["task_results"]
    lines += ["", "## Legacy reference (not used for this decision)", ""]
    for task_id in TASK_IDS:
        old_b0 = legacy_tasks[task_id]["B0"]["metrics"]
        new_b0 = tasks[task_id]["B0"]["metrics"]
        lines.append(
            f"- `{task_id}` legacy B0 Correct={old_b0['correct']}, BACC={old_b0['bacc']:.7f}, Macro-F1={old_b0['macro_f1']:.7f}; "
            f"corrected B0 delta Correct={new_b0['correct']-old_b0['correct']:+d}, BACC={new_b0['bacc']-old_b0['bacc']:+.7f}, Macro-F1={new_b0['macro_f1']-old_b0['macro_f1']:+.7f}."
        )
    q11 = "Yes, only SP-LRIF is worth considering next; it was not run here." if decision["decision"] == "TASK_ADAPTATION_GO" else "No unified Category-Global fusion experiment is authorized by this result."
    lines += [
        "",
        "## Required answers",
        "",
        "1. **Why did OVR not follow the declaration, and how was it fixed?** The historical loss hard-coded a sum of two OVR losses at coefficient 1. The new experiment-local loss explicitly averages the two losses and multiplies by 0.2; historical defaults remain unchanged.",
        f"2. **Did B0 change after correction?** Yes/no by task is quantified above against commit `{legacy['commit']}`; current corrected B0 Correct values are TAD={tad['B0']['metrics']['correct']}, ABIDE={abide['B0']['metrics']['correct']}, ABIDE-5={abide5['B0']['metrics']['correct']}.",
        f"3. **Most common TAD configuration?** {most_selected(tad['B1_Tuned'])}; full frequency={tad['B1_Tuned']['configuration_frequency']}.",
        f"4. **Most common ABIDE configuration?** {most_selected(abide['B1_Tuned'])}; full frequency={abide['B1_Tuned']['configuration_frequency']}.",
        f"5. **Did ABIDE turn from negative to net gain?** Correct delta={abide_cmp['correct_delta']:+d}, repairs/damages={abide_cmp['repairs']}/{abide_cmp['damages']}; {'yes' if abide_cmp['net_correct']>0 else 'no'}.",
        f"6. **Did ABIDE configuration transfer directly to ABIDE-5?** Transfer Correct delta={transfer_cmp['correct_delta']:+d}, BACC delta={transfer_cmp['metric_deltas']['bacc']:+.7f}; {'yes' if transfer_cmp['net_correct']>0 else 'no'}.",
        f"7. **Was ABIDE-5 Tuned clearly better than Transfer?** Tuned-vs-Transfer Correct delta={tuned_transfer['correct_delta']:+d}, BACC delta={tuned_transfer['metric_deltas']['bacc']:+.7f}; {'yes' if tuned_transfer['correct_delta']>0 and tuned_transfer['metric_deltas']['bacc']>0 else 'no'}.",
        f"8. **Did the TAD pMCI benefit remain?** TP delta={tad_cmp['positive_tp_delta']:+d}, sensitivity delta={tad_cmp['metric_deltas']['sen']:+.7f}, PR-AUC delta={tad_cmp['metric_deltas']['pr_auc']:+.7f}.",
        f"9. **Direct inference versus trajectory effect?** TAD direct private-off changed/repairs/damages={tad['B1_Tuned']['mechanism']['private_off_argmax_changed']}/{tad['B1_Tuned']['mechanism']['private_off_direct_repairs']}/{tad['B1_Tuned']['mechanism']['private_off_direct_damages']}, while full B0-to-B1 trajectory repairs/damages/changed={tad_cmp['repairs']}/{tad_cmp['damages']}/{tad_cmp['changed']}; analogous values are recorded for every task.",
        f"10. **Final Decision?** `{decision['decision']}`.",
        f"11. **Continue Category + Global fusion?** {q11}",
        "",
        "## Reproduction and Git handoff",
        "",
        f"Training source commit: `{source_commit}`. The eventual result commit is for reading results; strict reruns must use a clean local branch named `{BRANCH}` pointing at the source commit.",
        f"Historical evidence root: `{historical_root.resolve()}`.",
        f"Python: `{summary_core['environment']['python_executable']}`; torch={summary_core['environment']['torch_version']}; CUDA={summary_core['environment']['cuda_version']}; GPU={summary_core['environment']['gpu_name']}.",
        "",
        "Commands (from the clean source worktree):",
        "```powershell",
        f"& '{summary_core['environment']['python_executable']}' -u -B scripts\\run_binary_task_adapter_calibration_v1.py inspect --historical-root '{historical_root.resolve()}'",
        f"& '{summary_core['environment']['python_executable']}' -u -B scripts\\run_binary_task_adapter_calibration_v1.py smoke --device cuda:0 --historical-root '{historical_root.resolve()}'",
        f"& '{summary_core['environment']['python_executable']}' -u -B scripts\\run_binary_task_adapter_calibration_v1.py search --device cuda:0 --historical-root '{historical_root.resolve()}'",
        f"& '{summary_core['environment']['python_executable']}' -u -B scripts\\run_binary_task_adapter_calibration_v1.py formal --device cuda:0 --historical-root '{historical_root.resolve()}'",
        "```",
        "",
        "Result commit SHA is reported in the final Git handoff because a commit cannot contain its own SHA.",
    ]
    return "\n".join(lines) + "\n"


def run_formal(device_text: str, historical_root: Path) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    source = source_gate(historical_root)
    protocols = cross.load_protocols()
    runtime = runtime_lock(protocols, source)
    validate_smoke(runtime)
    selection = read_json(SELECTION_PATH)
    search_results = validate_search_results(protocols, selection, runtime, torch.device(device_text))
    formal_config_core = {
        "experiment": EXPERIMENT_ID,
        "runtime_lock": runtime,
        "search_results_sha256": search_results["sha256"],
        "variants": {
            "tadpole_smci_pmci": ["B0", "B1_Tuned"],
            "abide_ads_cn": ["B0", "B1_Tuned"],
            "abide5_ads_cn": ["B0", "B1_Transfer", "B1_Tuned"],
        },
        "fresh_outer_retrain": True,
        "outer_test_used_for_training_or_selection": False,
        "outer_test_inference_only_after_training": True,
        "abide5_transfer_source": "corresponding ABIDE fold rank, multiplier, and epoch",
    }
    formal_config = {**formal_config_core, "sha256": payload_sha256(formal_config_core)}
    atomic_write_json(RESULT_DIR / "formal_config.json", formal_config)
    started = time.perf_counter()
    contexts = {task_id: cross.build_context(protocols[task_id], torch.device(device_text)) for task_id in TASK_IDS}
    variant_map = {
        "tadpole_smci_pmci": ("B0", "B1_Tuned"),
        "abide_ads_cn": ("B0", "B1_Tuned"),
        "abide5_ads_cn": ("B0", "B1_Transfer", "B1_Tuned"),
    }
    task_results: dict[str, dict[str, Any]] = {}
    task_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_id in TASK_IDS:
        context = contexts[task_id]
        task_results[task_id] = {}
        task_rows[task_id] = {}
        for variant in variant_map[task_id]:
            summaries: list[dict[str, Any]] = []
            rows_by_fold: list[list[dict[str, Any]]] = []
            for fold in FOLDS:
                summary, rows = run_formal_fold(context, selection, search_results, runtime, variant, fold)
                summaries.append(summary)
                rows_by_fold.append(rows)
            result, rows = aggregate_variant(context, variant, summaries, rows_by_fold)
            task_results[task_id][variant] = result
            task_rows[task_id][variant] = rows

    comparisons: dict[str, dict[str, Any]] = {
        "tadpole_smci_pmci": {
            "B1_Tuned_vs_B0": paired_result(
                contexts["tadpole_smci_pmci"],
                task_results["tadpole_smci_pmci"]["B0"], task_results["tadpole_smci_pmci"]["B1_Tuned"],
                task_rows["tadpole_smci_pmci"]["B0"], task_rows["tadpole_smci_pmci"]["B1_Tuned"],
            )
        },
        "abide_ads_cn": {
            "B1_Tuned_vs_B0": paired_result(
                contexts["abide_ads_cn"],
                task_results["abide_ads_cn"]["B0"], task_results["abide_ads_cn"]["B1_Tuned"],
                task_rows["abide_ads_cn"]["B0"], task_rows["abide_ads_cn"]["B1_Tuned"],
            )
        },
        "abide5_ads_cn": {
            "B1_Transfer_vs_B0": paired_result(
                contexts["abide5_ads_cn"],
                task_results["abide5_ads_cn"]["B0"], task_results["abide5_ads_cn"]["B1_Transfer"],
                task_rows["abide5_ads_cn"]["B0"], task_rows["abide5_ads_cn"]["B1_Transfer"],
            ),
            "B1_Tuned_vs_B0": paired_result(
                contexts["abide5_ads_cn"],
                task_results["abide5_ads_cn"]["B0"], task_results["abide5_ads_cn"]["B1_Tuned"],
                task_rows["abide5_ads_cn"]["B0"], task_rows["abide5_ads_cn"]["B1_Tuned"],
            ),
            "B1_Tuned_vs_B1_Transfer": paired_result(
                contexts["abide5_ads_cn"],
                task_results["abide5_ads_cn"]["B1_Transfer"], task_results["abide5_ads_cn"]["B1_Tuned"],
                task_rows["abide5_ads_cn"]["B1_Transfer"], task_rows["abide5_ads_cn"]["B1_Tuned"],
            ),
        },
    }
    decision = final_decision(task_results, comparisons)
    mechanism = {
        task_id: {
            variant: result["mechanism"]
            for variant, result in variants.items()
            if variant != "B0"
        }
        for task_id, variants in task_results.items()
    }
    atomic_write_json(RESULT_DIR / "comparisons.json", comparisons)
    atomic_write_json(RESULT_DIR / "mechanism_diagnostics.json", mechanism)
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.device(device_text)),
        "device": device_text,
    }
    legacy = legacy_reference()
    summary_core = {
        "experiment": EXPERIMENT_ID,
        "runtime_lock": runtime,
        "formal_config": formal_config,
        "legacy_reference": legacy,
        "task_results": task_results,
        "comparisons": comparisons,
        "mechanism_diagnostics": mechanism,
        "decision": decision,
        "search_training_time_seconds": {
            task_id: float(search_results["tasks"][task_id]["search_training_time_seconds"])
            for task_id in TASK_IDS
        },
        "formal_wall_seconds": float(time.perf_counter() - started),
        "environment": environment,
        "result_commit": "reported_in_final_git_handoff",
    }
    report_path = RESULT_DIR / "REPORT.md"
    atomic_write_text(report_path, make_report(summary_core, historical_root, source))
    artifacts = [
        report_path,
        SEARCH_RESULTS_PATH,
        RESULT_DIR / "formal_config.json",
        RESULT_DIR / "comparisons.json",
        RESULT_DIR / "mechanism_diagnostics.json",
        SMOKE_DIR / "smoke_config.json",
        SMOKE_DIR / "smoke_report.json",
    ] + [
        RESULT_DIR / f"{task_id}_{variant}_oof_predictions.csv"
        for task_id, variants in variant_map.items()
        for variant in variants
    ]
    summary_core["artifact_sha256"] = {
        path.relative_to(ROOT).as_posix(): file_sha256(path) for path in artifacts
    }
    final_summary = {**summary_core, "sha256": payload_sha256(summary_core)}
    atomic_write_json(RESULT_DIR / "summary.json", final_summary)
    print(f"FORMAL COMPLETE decision={decision['decision']} summary_sha256={final_summary['sha256']}")


def run_resume(device_text: str, historical_root: Path) -> None:
    run_search(device_text, historical_root)
    run_formal(device_text, historical_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Freeze verified task and nested split manifests")
    inspect_parser.add_argument("--historical-root", default=str(cross.DEFAULT_HISTORICAL_ROOT))
    for mode in ("smoke", "search", "formal", "resume"):
        mode_parser = subparsers.add_parser(mode)
        mode_parser.add_argument("--device", default="cuda:0")
        mode_parser.add_argument("--historical-root", default=str(cross.DEFAULT_HISTORICAL_ROOT))
    arguments = parser.parse_args()
    historical_root = Path(arguments.historical_root).resolve()
    if arguments.mode == "inspect":
        run_inspect(historical_root)
    elif arguments.mode == "smoke":
        run_smoke(arguments.device, historical_root)
    elif arguments.mode == "search":
        run_search(arguments.device, historical_root)
    elif arguments.mode == "formal":
        run_formal(arguments.device, historical_root)
    elif arguments.mode == "resume":
        run_resume(arguments.device, historical_root)


if __name__ == "__main__":
    main()
