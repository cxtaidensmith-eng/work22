from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Loss import criterion_boundary_aware_multibranch
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
from run_boundary_aware_multibranch_v1 import (
    BOUNDARY_BRANCHES,
    CLASS_NAMES,
    EXPECTED_V2_COMMON_HASH,
    EXPECTED_V2_PARAMETER_COUNT,
    boundary_metadata,
    boundary_metric_bundle,
    copy_and_audit_full_initialization,
    scalar_components,
    structure_audit,
    tensors_are_finite,
)
from run_query_free_multibranch import (
    branch_similarity,
    clean_metrics,
    file_hash,
    metric_bundle,
    module_gradient_norm,
    split_hash,
    tensor_hash,
)
from run_query_free_multibranch_v2 import (
    build_model,
    copy_and_audit_v1_initialization,
)


MODEL_NAME = "Selective Boundary-aware Fixed Mapping"
SCHEMA_VERSION = "selective-boundary-fixed-10fold-smoke-v1"
EXPECTED_BRANCH = "experiment/selective-boundary-fixed-10fold-seed0"
SEED = 0
SMOKE_EPOCHS = 3
FOLDS = tuple(range(10))
ACTIVE_BOUNDARIES = ("SMCI_AD", "CN_AD")
INACTIVE_BOUNDARY = "CN_SMCI"
EXPECTED_FULL_INIT_HASH = (
    "d3dc751d979c3623d04c7703e9f3bd0b50481c857d04826a911792c302321b0a"
)
EXPECTED_CONFIG_HASH = (
    "5f741494141e478a49c0aa6af13e332a2cb98f15a5f29d90d0c057c7804e0cc2"
)
EXPECTED_DATA_HASH = (
    "2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042"
)
EXPECTED_MODAL_DICT_HASH = (
    "5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273"
)
EXPECTED_SAMPLE_ORDER_HASH = (
    "7ef70c38282cd3c3fa2fa95f41d5d7dfd3ee456d7ac58a25f80e2d7c378df668"
)
EXPECTED_LABEL_ORDER_HASH = (
    "40aa3114eeb832ce1b80a8706b47e50f99da7a48030d40560ef56164e109662a"
)
EXPECTED_SPLIT_HASHES = (
    "1adda8298733c24259acfa957378b356f52db9545ae9c40ef8889ee97bf3ff04",
    "529f52f8102c68c35ce93fe397fd1e8be83edc6ddf1ae9c2981b099e5e4b638e",
    "45901ce4e5beb6a5c30ca6e7dd63f69f3bef8cf9cd26bd9ad13c805a834d98a1",
    "af1cd9f0c1c662e18e4827e01525e573ca7755e762858c9423f36c467c9ad727",
    "f927bc45ad33595185b6ace11d1b1deb0fbbb43f2fe4239056b3b4c1097ef7a0",
    "7b18387f37130eb297c904390f0a762788d3ec34d59820dff40e5126e2cddf49",
    "7edb8fbd66273445117ecbf87b01106f796bcf571452e0335d2fef069155d4a4",
    "cc05b052a3c49b11ffc913a2d2a548dd400265e1cb8088c37e5378eee7767aa6",
    "b50979aea7b950e281ec0e4147d698a508f32c0f1314b39a2132117755234795",
    "f1395a882a4f6c518e17eeac3d37500a94d855534d5b8966eaa8421c3162ff12",
)
CLASS_ORDER = ("AD", "CN", "SMCI")
BRANCH_ORDER = (
    "latent_branch_1",
    "latent_branch_2",
    "latent_branch_3",
)
FIXED_MAPPING = {
    "latent_branch_1": {
        "boundary": "CN_SMCI",
        "auxiliary_active": False,
        "loss_contribution": 0.0,
    },
    "latent_branch_2": {
        "boundary": "SMCI_AD",
        "auxiliary_active": True,
        "loss_contribution": 1.0,
    },
    "latent_branch_3": {
        "boundary": "CN_AD",
        "auxiliary_active": True,
        "loss_contribution": 1.0,
    },
}
REQUIRED_FOLD_FILES = (
    "summary.json",
    "config.ini",
    "epoch_metrics.csv",
    "best_predictions.csv",
    "checkpoint_best.pt",
    "checkpoint_epoch3.pt",
    "branch_diagnostics.json",
    "boundary_metrics.json",
    "structure_audit_report.txt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def raw_array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    require(bool(rows), f"Cannot write empty CSV: {path}")
    columns = fieldnames if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def git_command(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def git_snapshot() -> dict:
    return {
        "branch": git_command("branch", "--show-current"),
        "commit": git_command("rev-parse", "HEAD"),
        "dirty": bool(git_command("status", "--porcelain")),
        "dirty_paths": git_command("status", "--porcelain").splitlines(),
    }


def environment_snapshot(device: torch.device) -> dict:
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    stable_environment = {
        key: environment[key]
        for key in (
            "python",
            "python_implementation",
            "platform",
            "torch",
            "numpy",
            "pandas",
            "scikit_learn",
            "scipy",
            "cuda_available",
            "torch_cuda",
            "cudnn",
            "device",
            "device_name",
        )
    }
    environment["core_environment_sha256"] = canonical_hash(stable_environment)
    return environment


def source_hashes() -> dict[str, str]:
    paths = (
        "Model/network.py",
        "Model/models.py",
        "Model/layers.py",
        "Model/__init__.py",
        "Loss/loss_fn.py",
        "Loss/__init__.py",
        "Utils/data_load.py",
        "Utils/graph_load.py",
        "Utils/utils.py",
        "Utils/__init__.py",
        "scripts/run_boundary_ablation.py",
        "scripts/run_boundary_aware_multibranch_v1.py",
        "scripts/run_query_free_multibranch.py",
        "scripts/run_query_free_multibranch_v2.py",
        "scripts/run_selective_boundary_fixed_10fold.py",
        "experiments/boundary_ablation/boundary_wo_cn_smci/summary.json",
    )
    result = {}
    for relative in paths:
        path = ROOT / relative
        require(path.is_file(), f"Required source file missing: {relative}")
        result[relative] = file_hash(path)
    return result


def validate_config(config: Config_, config_path: Path) -> dict:
    checks = {
        "dataset": config.DATA_SET == "TADPOLE",
        "task": config.Task == "AD_CN_SMCI",
        "shuffle": bool(config.Shuffle),
        "train_size": float(config.train_size) == 0.0,
        "seed": int(config.seed) == SEED,
        "base_config_epochs": int(config.epochs) == 400,
        "learning_rate": float(config.lr) == 0.01,
        "weight_decay": float(config.weight_decay) == 0.0005,
        "scheduler_enabled": bool(config.Use_scheduler),
        "scheduler_t_max": int(config.T_max) == 400,
        "scheduler_eta_min": float(config.Lr_Min) == 0.0001,
        "semantic_branch": config.semantic_branch == "both",
        "semantic_fusion": config.semantic_fusion == "add",
        "adj_mode": config.adj_mode == "none",
        "graph_use_graph": not bool(config.graph_use_graph),
        "logit_adjust_tau": float(config.logit_adjust_tau) == 0.75,
        "grad_clip": float(config.grad_clip) == 1.0,
        "config_hash": file_hash(config_path) == EXPECTED_CONFIG_HASH,
    }
    require(all(checks.values()), f"Fixed protocol config gate failed: {checks}")
    return {
        "checks": checks,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": file_hash(config_path),
        "declared_n_seeds": int(config.n_seeds),
        "effective_seed_list": [SEED],
        "executed_model_count_per_fold": 1,
        "resolved_protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "folds": list(FOLDS),
            "seed": SEED,
            "epochs_per_fold": SMOKE_EPOCHS,
            "single_model": True,
            "ensemble": False,
            "transductive_full_batch": True,
            "semantic_branch": "both",
            "semantic_fusion": "add",
            "category_branch_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "orthogonality_loss": False,
            "optimizer": "Adam",
            "learning_rate": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": int(config.T_max),
            "scheduler_eta_min": float(config.Lr_Min),
            "logit_adjust_tau": float(config.logit_adjust_tau),
            "grad_clip": float(config.grad_clip),
            "auxiliary_weight": 1.0,
            "label_smoothing": 0.05,
            "best_selection": "ACC, then Macro-AUC, then Macro-F1",
            "performance_reportable": False,
            "formal_training_authorized": False,
        },
    }


def class_count(labels: np.ndarray, positions: np.ndarray) -> dict[str, int]:
    return {
        name: int((labels[positions] == class_id).sum())
        for class_id, name in enumerate(CLASS_ORDER)
    }


def boundary_count(labels: np.ndarray, positions: np.ndarray) -> dict:
    result = {}
    for name, lower, higher in criterion_boundary_aware_multibranch.BOUNDARIES:
        selected = labels[positions]
        result[name] = {
            CLASS_NAMES[lower]: int((selected == lower).sum()),
            CLASS_NAMES[higher]: int((selected == higher).sum()),
            "contains_both_targets": bool(
                (selected == lower).any() and (selected == higher).any()
            ),
        }
    return result


def build_fold_manifest(dataset_dict: dict, dataset_data: dict) -> dict:
    sample_count = int(dataset_data["Feature"].size(0))
    require(sample_count == 598, f"Expected 598 samples, got {sample_count}")
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
    labels = dataset_data["Label"].detach().cpu().numpy().astype(np.int64)
    require(source_indices.shape == (sample_count,), "Source index shape mismatch")
    require(labels.shape == (sample_count,), "Label shape mismatch")
    require(len(np.unique(source_indices)) == sample_count, "Source indices are not unique")
    require(set(source_indices.tolist()) == set(range(sample_count)), "Unexpected source indices")
    require(raw_array_hash(source_indices) == EXPECTED_SAMPLE_ORDER_HASH, "Sample order hash mismatch")
    require(raw_array_hash(labels) == EXPECTED_LABEL_ORDER_HASH, "Label order hash mismatch")

    test_occurrence = np.zeros(sample_count, dtype=np.int64)
    train_occurrence = np.zeros(sample_count, dtype=np.int64)
    test_sets: list[set[int]] = []
    folds = []
    for fold in FOLDS:
        train_mask_t, test_mask_t = dataset_data["Mask"][fold]
        train_mask = train_mask_t.detach().cpu().numpy().astype(bool)
        test_mask = test_mask_t.detach().cpu().numpy().astype(bool)
        require(train_mask.shape == (sample_count,), f"fold {fold}: train mask shape")
        require(test_mask.shape == (sample_count,), f"fold {fold}: test mask shape")
        require(not np.any(train_mask & test_mask), f"fold {fold}: train/test overlap")
        require(np.array_equal(test_mask, ~train_mask), f"fold {fold}: masks not complementary")
        train_positions = np.flatnonzero(train_mask).astype(np.int64)
        test_positions = np.flatnonzero(test_mask).astype(np.int64)
        train_source_indices = source_indices[train_positions]
        test_source_indices = source_indices[test_positions]
        current_hash = split_hash(source_indices, train_mask_t, test_mask_t)
        require(current_hash == EXPECTED_SPLIT_HASHES[fold], f"fold {fold}: split hash mismatch")
        train_counts = class_count(labels, train_positions)
        test_counts = class_count(labels, test_positions)
        require(all(value > 0 for value in train_counts.values()), f"fold {fold}: train class missing")
        require(all(value > 0 for value in test_counts.values()), f"fold {fold}: test class missing")
        train_boundaries = boundary_count(labels, train_positions)
        test_boundaries = boundary_count(labels, test_positions)
        require(
            all(item["contains_both_targets"] for item in train_boundaries.values()),
            f"fold {fold}: train boundary target missing",
        )
        require(
            all(item["contains_both_targets"] for item in test_boundaries.values()),
            f"fold {fold}: test boundary target missing",
        )
        test_occurrence[test_positions] += 1
        train_occurrence[train_positions] += 1
        test_sets.append(set(test_positions.tolist()))
        folds.append({
            "fold": fold,
            "train_size": int(train_positions.size),
            "test_size": int(test_positions.size),
            "train_positions": train_positions.tolist(),
            "test_positions": test_positions.tolist(),
            "train_indices": train_source_indices.tolist(),
            "test_indices": test_source_indices.tolist(),
            "split_hash": current_hash,
            "train_mask_sha256": raw_array_hash(train_mask.astype(np.uint8)),
            "test_mask_sha256": raw_array_hash(test_mask.astype(np.uint8)),
            "train_class_count": train_counts,
            "test_class_count": test_counts,
            "train_boundary_count": train_boundaries,
            "test_boundary_count": test_boundaries,
            "train_test_disjoint": True,
            "train_test_complementary": True,
        })

    pairwise_overlap = {
        f"fold_{left:02d}_vs_fold_{right:02d}": len(test_sets[left] & test_sets[right])
        for left in FOLDS
        for right in FOLDS
        if left < right
    }
    require(all(value == 0 for value in pairwise_overlap.values()), "Test folds overlap")
    require(np.all(test_occurrence == 1), "Each sample must be test exactly once")
    require(np.all(train_occurrence == 9), "Each sample must be train exactly nine times")
    assignment = np.full(sample_count, -1, dtype=np.int64)
    for fold, positions in enumerate(test_sets):
        assignment[list(positions)] = fold
    require(np.all(assignment >= 0), "Unassigned test samples")
    validation = {
        "fold_count_is_10": len(folds) == 10,
        "all_train_test_disjoint": all(item["train_test_disjoint"] for item in folds),
        "all_train_test_complementary": all(item["train_test_complementary"] for item in folds),
        "test_sets_pairwise_disjoint": all(value == 0 for value in pairwise_overlap.values()),
        "test_union_count": int((test_occurrence > 0).sum()),
        "test_union_is_598": int((test_occurrence > 0).sum()) == sample_count,
        "each_sample_test_once": bool(np.all(test_occurrence == 1)),
        "each_sample_train_nine_times": bool(np.all(train_occurrence == 9)),
        "all_folds_contain_all_classes": all(
            all(value > 0 for value in item["test_class_count"].values())
            for item in folds
        ),
        "all_boundaries_contain_both_targets": all(
            all(value["contains_both_targets"] for value in item[split].values())
            for item in folds
            for split in ("train_boundary_count", "test_boundary_count")
        ),
        "split_hashes_unique": len({item["split_hash"] for item in folds}) == 10,
        "split_hashes_match_expected": tuple(item["split_hash"] for item in folds)
        == EXPECTED_SPLIT_HASHES,
    }
    require(
        all(value is True or (key == "test_union_count" and value == 598)
            for key, value in validation.items()),
        f"Fold manifest validation failed: {validation}",
    )
    assignment_hash_view = assignment.astype(np.uint8)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity_kind": "source CSV row index; not a verified patient/RID identifier",
        "sample_count": sample_count,
        "class_order": list(CLASS_ORDER),
        "class_count": class_count(labels, np.arange(sample_count)),
        "sample_order_sha256": raw_array_hash(source_indices),
        "label_order_sha256": raw_array_hash(labels),
        "test_fold_assignment_by_shuffled_position": assignment.tolist(),
        "test_fold_assignment_hash_dtype": "uint8",
        "test_fold_assignment_sha256": raw_array_hash(assignment_hash_view),
        "folds": folds,
        "pairwise_test_overlap": pairwise_overlap,
        "validation": validation,
    }
    manifest["canonical_sha256"] = canonical_hash(manifest)
    return manifest


def extended_structure_audit(model) -> dict:
    audit = structure_audit(model)
    state = model.state_dict()
    branch_names = tuple(model.latent_branches.keys())
    auxiliary_names = tuple(model.latent_aux_classifiers.keys())
    patient_state_names = sorted(key for key in state if key.startswith("patient_boundary_router."))
    global_weight_names = sorted(key for key in state if "global_boundary_fusion_weights" in key)
    audit.update({
        "state_tensor_count": len(state),
        "full_state_hash": tensor_hash(state),
        "branch_order": list(branch_names),
        "auxiliary_head_order": list(auxiliary_names),
        "category_branch_fusion": model.category_branch_fusion,
        "message_mlp_input_features": int(model.Message_MLP[0].in_features),
        "patient_router_module_present": hasattr(model, "patient_boundary_router"),
        "patient_router_state_names": patient_state_names,
        "global_weight_parameter_present": hasattr(model, "global_boundary_fusion_weights"),
        "global_weight_state_names": global_weight_names,
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
    })
    gates = {
        "parameter_count": audit["parameter_count"] == EXPECTED_V2_PARAMETER_COUNT,
        "state_tensor_count": audit["state_tensor_count"] == 145,
        "full_state_hash": audit["full_state_hash"] == EXPECTED_FULL_INIT_HASH,
        "no_legacy_query_pool": not audit["legacy_query_pool_present"],
        "no_patient_router": not audit["patient_router_module_present"] and not patient_state_names,
        "no_global_weight_fusion": not audit["global_weight_parameter_present"] and not global_weight_names,
        "category_branch_variant": audit["category_branch_variant"] == "query_free_multibranch",
        "latent_auxiliary_mode": audit["latent_auxiliary_mode"] == "one_vs_rest",
        "category_branch_fusion": audit["category_branch_fusion"] == "concat",
        "semantic_branch": audit["semantic_branch"] == "both",
        "semantic_fusion": audit["semantic_fusion"] == "add",
        "adj_mode": audit["adj_mode"] == "none",
        "difformer_graph_disabled": not any(audit["difformer_graph_use_flags"]),
        "latent_branch_count": audit["latent_branch_count"] == 3,
        "branch_order": branch_names == BRANCH_ORDER,
        "auxiliary_head_order": auxiliary_names == BRANCH_ORDER,
        "branch_parameters_independent": audit["latent_branch_parameters_independent"],
        "auxiliary_parameters_independent": audit["auxiliary_head_parameters_independent"],
        "message_mlp_input": audit["message_mlp_input_features"] == 288,
        "all_parameters_trainable": audit["all_parameters_trainable"],
    }
    audit["gates"] = gates
    require(all(gates.values()), f"Model structure gate failed: {gates}")
    return audit


def build_fresh_fold_objects(config, dataset_dict, device):
    SET_Random(SEED)
    v1_reference = build_model(config, dataset_dict, "multiclass", device)
    SET_Random(SEED)
    v2_reference = build_model(config, dataset_dict, "one_vs_rest", device)
    reconstruction = copy_and_audit_v1_initialization(v1_reference, v2_reference)
    require(reconstruction["v2_hash"] == EXPECTED_V2_COMMON_HASH, "Common reconstruction hash mismatch")
    require(reconstruction["max_abs_diff"] == 0.0, "Common reconstruction mismatch")
    require(not reconstruction["mismatch_names"], "Common reconstruction mismatch names")

    SET_Random(SEED)
    model = build_model(config, dataset_dict, "one_vs_rest", device)
    initialization = copy_and_audit_full_initialization(v2_reference, model)
    require(initialization["v2_full_hash"] == EXPECTED_FULL_INIT_HASH, "V2 full hash mismatch")
    require(initialization["boundary_full_hash"] == EXPECTED_FULL_INIT_HASH, "Model full hash mismatch")
    require(initialization["max_abs_diff"] == 0.0, "Full initialization max diff")
    require(not initialization["mismatch_names"], "Full initialization mismatch names")
    require(not initialization["missing_keys"], "Full initialization missing keys")
    require(not initialization["unexpected_keys"], "Full initialization unexpected keys")
    structure = extended_structure_audit(model)

    criterion = criterion_boundary_aware_multibranch(
        dataset_dict,
        device,
        label_smoothing=0.05,
        aux_weight=1.0,
        active_boundaries=ACTIVE_BOUNDARIES,
    )
    boundary_order = tuple(name for name, _, _ in criterion.BOUNDARIES)
    require(boundary_order == ("CN_SMCI", "SMCI_AD", "CN_AD"), "Boundary order changed")
    require(tuple(criterion.active_boundaries) == ACTIVE_BOUNDARIES, "Active boundaries changed")
    require(BOUNDARY_BRANCHES == {
        "CN_SMCI": "latent_branch_1",
        "SMCI_AD": "latent_branch_2",
        "CN_AD": "latent_branch_3",
    }, "Fixed physical boundary mapping changed")

    del v1_reference, v2_reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    SET_Random(SEED)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")
    require(scheduler.T_max == 400, "Scheduler T_max changed")
    require(float(scheduler.eta_min) == 0.0001, "Scheduler eta_min changed")
    return model, criterion, optimizer, scheduler, reconstruction, initialization, structure


def shape_audit(main_logits, branches, auxiliary, node_count: int) -> dict:
    result = {
        "main_logits_shape": list(main_logits.shape),
        "branch_shapes": [list(value.shape) for value in branches],
        "auxiliary_shapes": [list(value.shape) for value in auxiliary],
    }
    result.update({
        "main_logits_ok": tuple(main_logits.shape) == (node_count, 3),
        "branch_outputs_ok": len(branches) == 3 and all(
            tuple(value.shape) == (node_count, 96) for value in branches
        ),
        "auxiliary_outputs_ok": len(auxiliary) == 3 and all(
            tuple(value.shape) == (node_count, 2) for value in auxiliary
        ),
    })
    result["all_shapes_ok"] = all(
        result[key]
        for key in ("main_logits_ok", "branch_outputs_ok", "auxiliary_outputs_ok")
    )
    return result


def all_parameter_gradients_finite(model) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def score_tensors(raw_logits: torch.Tensor, label_weight: torch.Tensor, tau: float):
    adjusted = raw_logits - tau * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    probabilities = torch.softmax(adjusted, dim=-1)
    predictions = adjusted.argmax(dim=-1)
    return adjusted, probabilities, predictions


def make_prediction_rows(
    fold: int,
    source_indices: np.ndarray,
    truth: np.ndarray,
    raw_logits: np.ndarray,
    adjusted_scores: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[dict]:
    count = len(source_indices)
    require(truth.shape == (count,), "Prediction truth length mismatch")
    require(predictions.shape == (count,), "Prediction label length mismatch")
    require(raw_logits.shape == (count, 3), "Raw logits shape mismatch")
    require(adjusted_scores.shape == (count, 3), "Adjusted scores shape mismatch")
    require(probabilities.shape == (count, 3), "Probability shape mismatch")
    require(np.isfinite(raw_logits).all(), "Raw logits contain NaN/Inf")
    require(np.isfinite(adjusted_scores).all(), "Adjusted scores contain NaN/Inf")
    require(np.isfinite(probabilities).all(), "Probabilities contain NaN/Inf")
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0), "Probability sum mismatch")
    require(np.array_equal(adjusted_scores.argmax(axis=1), predictions), "Adjusted argmax mismatch")
    require(np.array_equal(probabilities.argmax(axis=1), predictions), "Probability argmax mismatch")
    require(len(np.unique(source_indices)) == count, "Prediction subject indices not unique")
    rows = []
    for idx in range(count):
        rows.append({
            "fold": fold,
            "subject_index": int(source_indices[idx]),
            "truth": int(truth[idx]),
            "prediction": int(predictions[idx]),
            "raw_logit_AD": float(raw_logits[idx, 0]),
            "raw_logit_CN": float(raw_logits[idx, 1]),
            "raw_logit_SMCI": float(raw_logits[idx, 2]),
            "adjusted_score_AD": float(adjusted_scores[idx, 0]),
            "adjusted_score_CN": float(adjusted_scores[idx, 1]),
            "adjusted_score_SMCI": float(adjusted_scores[idx, 2]),
            "probability_AD": float(probabilities[idx, 0]),
            "probability_CN": float(probabilities[idx, 1]),
            "probability_SMCI": float(probabilities[idx, 2]),
        })
    return rows


def metrics_from_prediction_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    adjusted = np.asarray([
        [
            float(row["adjusted_score_AD"]),
            float(row["adjusted_score_CN"]),
            float(row["adjusted_score_SMCI"]),
        ]
        for row in rows
    ])
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "acc": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(onehot, adjusted)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(3)).tolist(),
    }


def validate_prediction_metrics(rows: list[dict], expected_metrics: dict) -> dict:
    recomputed = metrics_from_prediction_rows(rows)
    deltas = {
        name: abs(float(recomputed[name]) - float(expected_metrics[name]))
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    confusion_equal = recomputed["confusion_matrix"] == expected_metrics["confusion_matrix"]
    passed = max(deltas.values(), default=0.0) <= 1e-10 and confusion_equal
    require(passed, f"Prediction metrics do not reproduce summary: {deltas}")
    return {
        "row_count": len(rows),
        "recomputed_metrics": recomputed,
        "metric_abs_deltas": deltas,
        "confusion_matrix_equal": confusion_equal,
        "passed": passed,
    }


def read_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def checkpoint_roundtrip(path: Path, expected_state: dict[str, torch.Tensor]) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    key_set_equal = set(loaded) == set(expected_state)
    values_equal = key_set_equal and all(
        torch.equal(loaded[key], expected_state[key]) for key in expected_state
    )
    audit = {
        "path": path.name,
        "sha256": file_hash(path),
        "state_tensor_count": len(loaded),
        "key_set_equal": key_set_equal,
        "tensor_values_equal": values_equal,
        "passed": key_set_equal and values_equal,
    }
    require(audit["passed"], f"Checkpoint roundtrip failed: {path}")
    return audit


def branch_output_norms(branches: list[torch.Tensor], names: list[str]) -> dict:
    return {
        name: {
            "mean_l2": float(output.norm(dim=-1).mean()),
            "std_l2": float(output.norm(dim=-1).std(unbiased=False)),
        }
        for name, output in zip(names, branches)
    }


def render_structure_report(summary: dict) -> str:
    structure = summary["structure_audit"]
    result = summary["result"]
    checks = summary["hard_gates"]
    lines = [
        MODEL_NAME,
        f"10-fold smoke fold {summary['fold']:02d} structure audit",
        "",
        f"fold={summary['fold']}",
        f"seed={summary['protocol']['seed']}",
        f"epochs={summary['protocol']['epochs']}",
        f"train/test={summary['protocol']['train_size']}/{summary['protocol']['test_size']}",
        f"split_hash={summary['protocol']['split_hash']}",
        f"parameter_count={structure['parameter_count']}",
        f"state_tensor_count={structure['state_tensor_count']}",
        f"full_state_hash={summary['initialization_audit']['boundary_full_hash']}",
        f"common_reconstruction_hash={summary['v2_reconstruction_audit']['v2_hash']}",
        f"branch_order={structure['branch_order']}",
        f"fixed_mapping={summary['fixed_mapping']}",
        "loss=L_main + L_SMCI_AD + L_CN_AD",
        f"best_epoch={result['best_epoch']}",
        f"best_metrics={result['best_main_metrics']}",
        "",
        "Hard gates:",
    ]
    lines.extend(f"{name}={value}" for name, value in checks.items())
    lines.extend([
        "",
        f"smoke_passed={summary['smoke_passed']}",
        "formal_400_epoch_run_started=False",
        "formal_400_epoch_run_completed=False",
    ])
    return "\n".join(lines) + "\n"


def run_fold(
    fold: int,
    config,
    config_path: Path,
    dataset_dict: dict,
    dataset_data: dict,
    fold_manifest_entry: dict,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    (
        model,
        criterion,
        optimizer,
        scheduler,
        reconstruction,
        initialization,
        structure,
    ) = build_fresh_fold_objects(config, dataset_dict, device)
    train_mask, test_mask = dataset_data["Mask"][fold]
    labels = dataset_data["Label"]
    features = dataset_data["Feature"]
    node_count = int(features.size(0))
    branch_modules = dict(model.latent_branches.items())
    auxiliary_modules = dict(model.latent_aux_classifiers.items())
    branch_names = list(branch_modules)
    require(tuple(branch_names) == BRANCH_ORDER, "Physical branch order changed")
    require(tuple(auxiliary_modules) == BRANCH_ORDER, "Auxiliary head order changed")
    require(int(train_mask.sum()) == fold_manifest_entry["train_size"], "Train size mismatch")
    require(int(test_mask.sum()) == fold_manifest_entry["test_size"], "Test size mismatch")
    metadata = boundary_metadata(criterion, labels, train_mask, test_mask)
    for boundary_name, entry in metadata.items():
        entry["active_auxiliary_loss"] = boundary_name in ACTIVE_BOUNDARIES
    require(
        all(
            entry[split]["is_subset_of_split"] and entry[split]["contains_both_targets"]
            for entry in metadata.values()
            for split in ("train", "test")
        ),
        f"fold {fold}: boundary mask audit failed",
    )

    fold_dir = output_dir / f"fold_{fold:02d}"
    require(not fold_dir.exists(), f"Refusing to overwrite fold directory: {fold_dir}")
    fold_dir.mkdir(parents=False)
    shutil.copyfile(config_path, fold_dir / "config.ini")

    rows = []
    branch_diagnostics = []
    boundary_diagnostics = []
    best = None
    best_state = None
    best_prediction_rows = None
    final_state = None
    first_shape_audit = None
    all_shape_checks = True
    all_loss_checks = True
    all_branch_gradients_finite = True
    all_branch_gradients_nonzero = True
    inactive_auxiliary_gradient_zero = True
    active_auxiliary_gradients_nonzero_finite = True
    all_gradients_finite = True
    no_nan_or_inf = True
    started = time.perf_counter()

    for epoch in range(1, SMOKE_EPOCHS + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        main_logits, branch_outputs, auxiliary_outputs = model(features)
        current_shapes = shape_audit(main_logits, branch_outputs, auxiliary_outputs, node_count)
        if first_shape_audit is None:
            first_shape_audit = current_shapes
        all_shape_checks = all_shape_checks and current_shapes["all_shapes_ok"]
        require(current_shapes["all_shapes_ok"], f"fold {fold} epoch {epoch}: shape gate")
        forward_finite = tensors_are_finite([main_logits, *branch_outputs, *auxiliary_outputs])
        require(forward_finite, f"fold {fold} epoch {epoch}: NaN/Inf in forward")
        loss, train_components_t = criterion.compute(
            main_logits,
            labels,
            train_mask,
            branch_outputs,
            auxiliary_outputs,
        )
        component_finite = tensors_are_finite(list(train_components_t.values()))
        require(component_finite, f"fold {fold} epoch {epoch}: NaN/Inf in loss")
        expected_auxiliary = train_components_t["SMCI_AD"] + train_components_t["CN_AD"]
        expected_total = train_components_t["main"] + expected_auxiliary
        loss_check = (
            float((train_components_t["auxiliary_sum"] - expected_auxiliary).detach().abs()) <= 1e-7
            and float((loss - expected_total).detach().abs()) <= 1e-7
        )
        require(loss_check, f"fold {fold} epoch {epoch}: loss composition changed")
        all_loss_checks = all_loss_checks and loss_check
        loss.backward()
        branch_gradients = {
            name: module_gradient_norm(module) for name, module in branch_modules.items()
        }
        auxiliary_gradients = {
            name: module_gradient_norm(module) for name, module in auxiliary_modules.items()
        }
        branch_finite = all(np.isfinite(value) for value in branch_gradients.values())
        branch_nonzero = all(value > 0.0 for value in branch_gradients.values())
        inactive_zero = auxiliary_gradients["latent_branch_1"] == 0.0
        active_ok = all(
            auxiliary_gradients[name] > 0.0 and np.isfinite(auxiliary_gradients[name])
            for name in ("latent_branch_2", "latent_branch_3")
        )
        parameter_gradients_finite = all_parameter_gradients_finite(model)
        require(branch_finite, f"fold {fold} epoch {epoch}: non-finite branch gradient")
        require(branch_nonzero, f"fold {fold} epoch {epoch}: zero latent branch gradient")
        require(inactive_zero, f"fold {fold} epoch {epoch}: inactive auxiliary got gradient")
        require(active_ok, f"fold {fold} epoch {epoch}: active auxiliary gradient invalid")
        require(parameter_gradients_finite, f"fold {fold} epoch {epoch}: non-finite parameter gradient")
        all_branch_gradients_finite = all_branch_gradients_finite and branch_finite
        all_branch_gradients_nonzero = all_branch_gradients_nonzero and branch_nonzero
        inactive_auxiliary_gradient_zero = inactive_auxiliary_gradient_zero and inactive_zero
        active_auxiliary_gradients_nonzero_finite = (
            active_auxiliary_gradients_nonzero_finite and active_ok
        )
        all_gradients_finite = all_gradients_finite and parameter_gradients_finite
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            eval_logits, eval_branches, eval_auxiliary = model(features)
            eval_shapes = shape_audit(eval_logits, eval_branches, eval_auxiliary, node_count)
            require(eval_shapes["all_shapes_ok"], f"fold {fold} epoch {epoch}: eval shape gate")
            test_loss, test_components_t = criterion.compute(
                eval_logits,
                labels,
                test_mask,
                eval_branches,
                eval_auxiliary,
            )
            eval_finite = tensors_are_finite([
                eval_logits,
                *eval_branches,
                *eval_auxiliary,
                test_loss,
                *test_components_t.values(),
            ])
            require(eval_finite, f"fold {fold} epoch {epoch}: NaN/Inf in eval")
            train_metrics = metric_bundle(
                eval_logits,
                labels,
                train_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            test_metrics = metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            require(
                all(np.isfinite(float(test_metrics[name])) for name in (
                    "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"
                )),
                f"fold {fold} epoch {epoch}: non-finite main metric",
            )
            train_boundary_metrics = boundary_metric_bundle(
                eval_auxiliary, labels, train_mask, criterion
            )
            test_boundary_metrics = boundary_metric_bundle(
                eval_auxiliary, labels, test_mask, criterion
            )
            similarity = branch_similarity(eval_branches, branch_names)
            output_norms = branch_output_norms(eval_branches, branch_names)
            raw_test = eval_logits[test_mask]
            adjusted_test, probability_test, prediction_test = score_tensors(
                raw_test,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            require(
                prediction_test.detach().cpu().tolist() == test_metrics["predictions"],
                f"fold {fold} epoch {epoch}: prediction definition mismatch",
            )

        no_nan_or_inf = no_nan_or_inf and forward_finite and component_finite and eval_finite
        train_components = scalar_components(train_components_t)
        test_components = scalar_components(test_components_t)
        row = {
            "epoch": epoch,
            "lr_used": lr_used,
            "lr_next": lr_next,
            "train_total_loss": float(loss.detach()),
            "train_main_loss": train_components["main"],
            "train_CN_SMCI_loss": train_components["CN_SMCI"],
            "train_SMCI_AD_loss": train_components["SMCI_AD"],
            "train_CN_AD_loss": train_components["CN_AD"],
            "train_CN_SMCI_contribution": 0.0,
            "train_SMCI_AD_contribution": train_components["SMCI_AD"],
            "train_CN_AD_contribution": train_components["CN_AD"],
            "train_acc": train_metrics["acc"],
            "test_total_loss": float(test_loss),
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_bacc": test_metrics["bacc"],
            "test_macro_auc": test_metrics["macro_auc"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "latent_branch_1_grad_norm": branch_gradients["latent_branch_1"],
            "latent_branch_2_grad_norm": branch_gradients["latent_branch_2"],
            "latent_branch_3_grad_norm": branch_gradients["latent_branch_3"],
            "latent_branch_1_aux_grad_norm": auxiliary_gradients["latent_branch_1"],
            "latent_branch_2_aux_grad_norm": auxiliary_gradients["latent_branch_2"],
            "latent_branch_3_aux_grad_norm": auxiliary_gradients["latent_branch_3"],
            "branch_similarity_off_diagonal_mean": similarity["off_diagonal_mean"],
        }
        rows.append(row)
        branch_diagnostics.append({
            "epoch": epoch,
            "fixed_mapping": FIXED_MAPPING,
            "branch_gradient_norms": branch_gradients,
            "auxiliary_head_gradient_norms": auxiliary_gradients,
            "branch_output_norms": output_norms,
            "branch_similarity": similarity,
            "all_parameter_gradients_finite": parameter_gradients_finite,
        })
        boundary_diagnostics.append({
            "epoch": epoch,
            "active_auxiliary_losses": list(ACTIVE_BOUNDARIES),
            "inactive_auxiliary_loss": INACTIVE_BOUNDARY,
            "train_loss_components": train_components,
            "train_loss_contributions": {
                "CN_SMCI": 0.0,
                "SMCI_AD": train_components["SMCI_AD"],
                "CN_AD": train_components["CN_AD"],
            },
            "test_loss_components": test_components,
            "train_boundary_metrics": train_boundary_metrics,
            "test_boundary_metrics": test_boundary_metrics,
        })

        score = (
            test_metrics["acc"],
            test_metrics["macro_auc"],
            test_metrics["macro_f1"],
        )
        if best is None or score > best["score"]:
            source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
                test_mask.detach().cpu().numpy().astype(bool)
            ]
            truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
            best_rows = make_prediction_rows(
                fold,
                source_indices,
                truth,
                raw_test.detach().cpu().numpy(),
                adjusted_test.detach().cpu().numpy(),
                probability_test.detach().cpu().numpy(),
                prediction_test.detach().cpu().numpy().astype(np.int64),
            )
            best = {
                "epoch": epoch,
                "score": score,
                "main_metrics": deepcopy(test_metrics),
                "train_loss_components": deepcopy(train_components),
                "test_boundary_metrics": deepcopy(test_boundary_metrics),
                "branch_gradient_norms": deepcopy(branch_gradients),
                "auxiliary_head_gradient_norms": deepcopy(auxiliary_gradients),
                "branch_similarity": deepcopy(similarity),
            }
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_prediction_rows = best_rows
        print(
            f"[fixed-cv10 smoke] fold={fold:02d} epoch={epoch}/{SMOKE_EPOCHS} "
            f"loss={float(loss.detach()):.4f} test_acc={test_metrics['acc']:.4f}",
            flush=True,
        )

    require(best is not None and best_state is not None, f"fold {fold}: no best state")
    require(best_prediction_rows is not None, f"fold {fold}: no best predictions")
    final_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    write_csv(fold_dir / "epoch_metrics.csv", rows)
    write_json(fold_dir / "branch_diagnostics.json", branch_diagnostics)
    write_json(fold_dir / "boundary_metrics.json", boundary_diagnostics)
    write_csv(fold_dir / "best_predictions.csv", best_prediction_rows)
    prediction_path = fold_dir / "best_predictions.csv"
    reloaded_rows = read_prediction_rows(prediction_path)
    require(len(reloaded_rows) == int(test_mask.sum()), f"fold {fold}: prediction row count")
    prediction_audit = validate_prediction_metrics(reloaded_rows, best["main_metrics"])
    reloaded_subject_indices = [int(row["subject_index"]) for row in reloaded_rows]
    expected_subject_indices = [int(value) for value in fold_manifest_entry["test_indices"]]
    reloaded_truth = [int(row["truth"]) for row in reloaded_rows]
    expected_truth = [int(value) for value in labels[test_mask].detach().cpu().tolist()]
    prediction_audit.update({
        "csv_sha256": file_hash(prediction_path),
        "subject_indices_unique": len(set(reloaded_subject_indices))
        == len(reloaded_rows),
        "subject_indices_match_fold_manifest_order": (
            reloaded_subject_indices == expected_subject_indices
        ),
        "subject_index_set_matches_fold_manifest": (
            set(reloaded_subject_indices) == set(expected_subject_indices)
        ),
        "fold_column_matches": all(int(row["fold"]) == fold for row in reloaded_rows),
        "truth_matches_fold_labels": reloaded_truth == expected_truth,
        "row_count_matches_test_size": len(reloaded_rows) == int(test_mask.sum()),
        "contains_raw_logits": all(
            f"raw_logit_{name}" in reloaded_rows[0] for name in CLASS_ORDER
        ),
        "contains_adjusted_scores": all(
            f"adjusted_score_{name}" in reloaded_rows[0] for name in CLASS_ORDER
        ),
        "contains_probabilities": all(
            f"probability_{name}" in reloaded_rows[0] for name in CLASS_ORDER
        ),
    })
    require(all(value for key, value in prediction_audit.items() if key in {
        "passed",
        "subject_indices_unique",
        "subject_indices_match_fold_manifest_order",
        "subject_index_set_matches_fold_manifest",
        "fold_column_matches",
        "truth_matches_fold_labels",
        "row_count_matches_test_size",
        "contains_raw_logits",
        "contains_adjusted_scores",
        "contains_probabilities",
    }), f"fold {fold}: prediction contract failed")

    best_checkpoint = fold_dir / "checkpoint_best.pt"
    final_checkpoint = fold_dir / "checkpoint_epoch3.pt"
    torch.save(best_state, best_checkpoint)
    torch.save(final_state, final_checkpoint)
    checkpoint_audit = {
        "best": checkpoint_roundtrip(best_checkpoint, best_state),
        "epoch3": checkpoint_roundtrip(final_checkpoint, final_state),
    }

    hard_gates = {
        "train_test_size_and_mask": (
            int(train_mask.sum()) == fold_manifest_entry["train_size"]
            and int(test_mask.sum()) == fold_manifest_entry["test_size"]
        ),
        "split_hash": split_hash(
            np.asarray(dataset_dict["Index"], dtype=np.int64), train_mask, test_mask
        ) == fold_manifest_entry["split_hash"],
        "model_structure": all(structure["gates"].values()),
        "initialization": (
            initialization["max_abs_diff"] == 0.0
            and initialization["boundary_full_hash"] == EXPECTED_FULL_INIT_HASH
            and reconstruction["v2_hash"] == EXPECTED_V2_COMMON_HASH
        ),
        "output_shapes": all_shape_checks,
        "loss_formula": all_loss_checks,
        "cn_smci_contribution_zero": all(
            item["train_loss_contributions"]["CN_SMCI"] == 0.0
            for item in boundary_diagnostics
        ),
        "all_latent_branch_gradients_finite": all_branch_gradients_finite,
        "all_latent_branch_gradients_nonzero": all_branch_gradients_nonzero,
        "inactive_branch1_auxiliary_gradient_zero": inactive_auxiliary_gradient_zero,
        "active_branch2_branch3_auxiliary_gradients_nonzero_finite": (
            active_auxiliary_gradients_nonzero_finite
        ),
        "all_parameter_gradients_finite": all_gradients_finite,
        "no_nan_or_inf": no_nan_or_inf,
        "best_checkpoint_roundtrip": checkpoint_audit["best"]["passed"],
        "epoch3_checkpoint_roundtrip": checkpoint_audit["epoch3"]["passed"],
        "prediction_contract": prediction_audit["passed"],
        "prediction_rows_bound_to_fold_manifest": all(
            prediction_audit[key]
            for key in (
                "subject_indices_match_fold_manifest_order",
                "subject_index_set_matches_fold_manifest",
                "fold_column_matches",
                "truth_matches_fold_labels",
            )
        ),
        "prediction_rows_match_test_size": prediction_audit["row_count_matches_test_size"],
    }
    smoke_passed = all(hard_gates.values())
    require(smoke_passed, f"fold {fold}: smoke hard gate failed: {hard_gates}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "run_kind": "10fold_3epoch_smoke",
        "fold": fold,
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": fold,
            "seed": SEED,
            "epochs": SMOKE_EPOCHS,
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "split_hash": fold_manifest_entry["split_hash"],
            "transductive_full_batch": True,
            "single_model": True,
            "ensemble": False,
            "semantic_branch": "both",
            "semantic_fusion": "add",
            "category_branch_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "orthogonality_loss": False,
            "optimizer": "Adam",
            "learning_rate": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": int(config.T_max),
            "scheduler_eta_min": float(config.Lr_Min),
            "logit_adjust_tau": float(config.logit_adjust_tau),
            "grad_clip": float(config.grad_clip),
            "best_selection": "ACC, then Macro-AUC, then Macro-F1",
        },
        "fixed_mapping": FIXED_MAPPING,
        "active_auxiliary_losses": list(ACTIVE_BOUNDARIES),
        "inactive_auxiliary_loss": INACTIVE_BOUNDARY,
        "loss_formula": "L_main + L_SMCI_AD + L_CN_AD",
        "boundary_class_and_masks": metadata,
        "v2_reconstruction_audit": reconstruction,
        "initialization_audit": initialization,
        "structure_audit": structure,
        "result": {
            "epochs": SMOKE_EPOCHS,
            "first_forward_shape_audit": first_shape_audit,
            "best_epoch": best["epoch"],
            "best_main_metrics": clean_metrics(best["main_metrics"]),
            "best_train_loss_components": best["train_loss_components"],
            "best_test_boundary_metrics": best["test_boundary_metrics"],
            "best_branch_gradient_norms": best["branch_gradient_norms"],
            "best_auxiliary_head_gradient_norms": best["auxiliary_head_gradient_norms"],
            "best_branch_similarity": best["branch_similarity"],
            "final_main_metrics": clean_metrics(test_metrics),
            "elapsed_seconds": time.perf_counter() - started,
            "prediction_audit": prediction_audit,
            "checkpoint_audit": checkpoint_audit,
        },
        "hard_gates": hard_gates,
        "smoke_passed": smoke_passed,
        "performance_reportable": False,
        "formal_400_epoch_run_started": False,
        "formal_400_epoch_run_completed": False,
    }
    (fold_dir / "structure_audit_report.txt").write_text(
        render_structure_report(summary), encoding="utf-8"
    )
    write_json(fold_dir / "summary.json", summary)
    missing = [name for name in REQUIRED_FOLD_FILES if not (fold_dir / name).is_file()]
    require(not missing, f"fold {fold}: required artifacts missing: {missing}")
    del model, criterion, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, best_prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-only 10-fold runner for Selective Boundary Fixed Mapping"
    )
    parser.add_argument(
        "--config",
        default="Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini",
    )
    parser.add_argument("--epochs", type=int, default=SMOKE_EPOCHS)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/selective_boundary_fixed_10fold_seed0/smoke",
    )
    args = parser.parse_args()
    if args.formal:
        raise ValueError("Formal mode is intentionally disabled; 400 epoch is not authorized")
    if args.epochs != SMOKE_EPOCHS:
        raise ValueError("This runner is smoke-only and requires exactly 3 epochs")

    output_dir = (ROOT / args.output).resolve()
    experiments_root = (ROOT / "experiments").resolve()
    try:
        output_dir.relative_to(experiments_root)
    except ValueError as error:
        raise ValueError("Output must stay under experiments/") from error
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")

    repository = git_snapshot()
    require(repository["branch"] == EXPECTED_BRANCH, f"Wrong Git branch: {repository['branch']}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = (ROOT / args.config).resolve()
    require(config_path.is_file(), f"Config missing: {config_path}")
    config = Config_(str(ROOT), str(config_path), 0 if device.type == "cuda" else None)
    config.Device = device
    config_audit = validate_config(config, config_path)

    feature_path_value, dict_path_value, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    feature_path = Path(feature_path_value)
    dict_path = Path(dict_path_value)
    require(file_hash(feature_path) == EXPECTED_DATA_HASH, "Dataset hash mismatch")
    require(file_hash(dict_path) == EXPECTED_MODAL_DICT_HASH, "Modal dictionary hash mismatch")
    require(list(class_names) == ["AD", "CN", "SMCI"], "Class order changed")

    SET_Random(SEED)
    dataset_dict, dataset_data = load_dataset(
        str(feature_path),
        str(dict_path),
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    fold_manifest = build_fold_manifest(dataset_dict, dataset_data)
    initial_source_hashes = source_hashes()
    source_aggregate_hash = canonical_hash(initial_source_hashes)
    resolved_protocol = config_audit["resolved_protocol"]
    protocol_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": MODEL_NAME,
            "run_kind": "10fold_3epoch_smoke",
            "authorized_epochs": SMOKE_EPOCHS,
            "formal_training_authorized": False,
            "seed": SEED,
            "folds": list(FOLDS),
        },
        "repository": repository,
        "source_integrity": {
            "files": initial_source_hashes,
            "aggregate_sha256": source_aggregate_hash,
            "unchanged_at_completion": None,
        },
        "data": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "csv": feature_path.relative_to(ROOT).as_posix(),
            "csv_sha256": file_hash(feature_path),
            "modal_dict": dict_path.relative_to(ROOT).as_posix(),
            "modal_dict_sha256": file_hash(dict_path),
            "num_samples": int(dataset_data["Feature"].size(0)),
            "num_features": int(dataset_data["Feature"].size(1)),
            "class_mapping": {"AD": 0, "CN": 1, "SMCI": 2},
            "class_counts": fold_manifest["class_count"],
            "sample_order_sha256": fold_manifest["sample_order_sha256"],
            "label_order_sha256": fold_manifest["label_order_sha256"],
            "identity_kind": fold_manifest["identity_kind"],
        },
        "configuration": {
            **config_audit,
            "resolved_protocol_sha256": canonical_hash(resolved_protocol),
        },
        "environment": environment_snapshot(device),
        "model_contract": {
            "parameter_count": EXPECTED_V2_PARAMETER_COUNT,
            "state_tensor_count": 145,
            "full_state_sha256": EXPECTED_FULL_INIT_HASH,
            "common_reconstruction_sha256": EXPECTED_V2_COMMON_HASH,
            "category_branch_fusion": "concat",
            "legacy_query_pool": False,
            "patient_router": False,
            "global_boundary_weight": False,
            "independent_latent_branches": 3,
        },
        "fixed_mapping": FIXED_MAPPING,
        "execution_contract": {
            "dataset_load_count": 1,
            "fold_order": list(FOLDS),
            "parallel_folds": False,
            "fresh_model_per_fold": True,
            "fresh_criterion_per_fold": True,
            "fresh_optimizer_per_fold": True,
            "fresh_scheduler_per_fold": True,
            "checkpoint_initialization": False,
        },
        "fold_manifest_sha256": fold_manifest["canonical_sha256"],
        "validation": {
            "static_preflight_passed": True,
            "static_preflight_scope": "configuration, data, fold manifest, and source hashes",
            "per_fold_model_preflight_passed": False,
            "completion_passed": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "fold_manifest.json", fold_manifest)
    write_json(output_dir / "protocol_manifest.json", protocol_manifest)

    fold_summaries = []
    all_best_rows = []
    executed_folds = []
    total_started = time.perf_counter()
    for fold in FOLDS:
        summary, best_rows = run_fold(
            fold,
            config,
            config_path,
            dataset_dict,
            dataset_data,
            fold_manifest["folds"][fold],
            output_dir,
            device,
        )
        fold_summaries.append(summary)
        all_best_rows.extend(best_rows)
        executed_folds.append(fold)

    require(executed_folds == list(FOLDS), "Folds did not execute sequentially 0..9")
    require(len(all_best_rows) == 598, "OOF smoke prediction count must be 598")
    all_subject_indices = [int(row["subject_index"]) for row in all_best_rows]
    require(len(set(all_subject_indices)) == 598, "OOF subject indices overlap")
    require(set(all_subject_indices) == set(range(598)), "OOF subject coverage mismatch")
    write_csv(output_dir / "oof_best_predictions.csv", all_best_rows)

    fold_metric_rows = []
    for summary in fold_summaries:
        best = summary["result"]["best_main_metrics"]
        final = summary["result"]["final_main_metrics"]
        fold_metric_rows.append({
            "fold": summary["fold"],
            "seed": SEED,
            "train_size": summary["protocol"]["train_size"],
            "test_size": summary["protocol"]["test_size"],
            "split_hash": summary["protocol"]["split_hash"],
            "best_epoch": summary["result"]["best_epoch"],
            "best_acc": best["acc"],
            "best_macro_f1": best["macro_f1"],
            "best_bacc": best["bacc"],
            "best_macro_auc": best["macro_auc"],
            "best_weighted_f1": best["weighted_f1"],
            "final_acc": final["acc"],
            "final_macro_f1": final["macro_f1"],
            "final_bacc": final["bacc"],
            "final_macro_auc": final["macro_auc"],
            "final_weighted_f1": final["weighted_f1"],
            "parameter_count": summary["structure_audit"]["parameter_count"],
            "elapsed_seconds": summary["result"]["elapsed_seconds"],
            "smoke_passed": summary["smoke_passed"],
        })
    write_csv(output_dir / "fold_metrics.csv", fold_metric_rows)

    final_source_hashes = source_hashes()
    sources_unchanged = final_source_hashes == initial_source_hashes
    require(sources_unchanged, "Locked source files changed during smoke")
    required_artifacts_present = all(
        all((output_dir / f"fold_{fold:02d}" / name).is_file() for name in REQUIRED_FOLD_FILES)
        for fold in FOLDS
    )
    require(required_artifacts_present, "Required fold artifacts missing")
    all_folds_passed = all(summary["smoke_passed"] for summary in fold_summaries)
    require(all_folds_passed, "At least one fold failed smoke")
    oof_rows = read_prediction_rows(output_dir / "oof_best_predictions.csv")
    oof_unique = len({row["subject_index"] for row in oof_rows})
    smoke_summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "run_kind": "10fold_3epoch_smoke",
        "expected_folds": len(FOLDS),
        "completed_folds": len(fold_summaries),
        "passed_folds": sum(int(summary["smoke_passed"]) for summary in fold_summaries),
        "failed_folds": [summary["fold"] for summary in fold_summaries if not summary["smoke_passed"]],
        "executed_fold_order": executed_folds,
        "epochs_per_fold": SMOKE_EPOCHS,
        "total_epochs_executed": len(FOLDS) * SMOKE_EPOCHS,
        "fold_manifest_validation": fold_manifest["validation"],
        "fold_status": [
            {
                "fold": summary["fold"],
                "train_size": summary["protocol"]["train_size"],
                "test_size": summary["protocol"]["test_size"],
                "split_hash": summary["protocol"]["split_hash"],
                "best_epoch": summary["result"]["best_epoch"],
                "best_main_metrics": summary["result"]["best_main_metrics"],
                "hard_gates": summary["hard_gates"],
                "smoke_passed": summary["smoke_passed"],
            }
            for summary in fold_summaries
        ],
        "all_model_hard_gates_passed": all_folds_passed,
        "artifact_contract_passed": required_artifacts_present,
        "source_integrity_unchanged": sources_unchanged,
        "oof_prediction_rows": len(oof_rows),
        "oof_unique_subject_indices": oof_unique,
        "oof_duplicate_subject_indices": len(oof_rows) - oof_unique,
        "oof_missing_subject_indices": 598 - oof_unique,
        "oof_coverage_passed": len(oof_rows) == 598 and oof_unique == 598,
        "elapsed_seconds": time.perf_counter() - total_started,
        "ready_for_formal_runner_authorization": True,
        "performance_reportable": False,
        "formal_400_epoch_run_started": False,
        "formal_400_epoch_run_completed": False,
    }
    write_json(output_dir / "smoke_summary.json", smoke_summary)
    protocol_manifest["source_integrity"]["unchanged_at_completion"] = sources_unchanged
    protocol_manifest["source_integrity"]["completion_files"] = final_source_hashes
    protocol_manifest["validation"]["per_fold_model_preflight_passed"] = all_folds_passed
    protocol_manifest["validation"]["completion_passed"] = (
        all_folds_passed
        and required_artifacts_present
        and sources_unchanged
        and smoke_summary["oof_coverage_passed"]
    )
    protocol_manifest["completion"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_folds": len(fold_summaries),
        "total_epochs_executed": len(FOLDS) * SMOKE_EPOCHS,
        "smoke_summary_sha256": file_hash(output_dir / "smoke_summary.json"),
        "formal_400_epoch_run_started": False,
        "formal_400_epoch_run_completed": False,
    }
    write_json(output_dir / "protocol_manifest.json", protocol_manifest)
    print(json.dumps({
        "output": output_dir.relative_to(ROOT).as_posix(),
        "completed_folds": len(fold_summaries),
        "passed_folds": smoke_summary["passed_folds"],
        "total_epochs_executed": smoke_summary["total_epochs_executed"],
        "oof_coverage_passed": smoke_summary["oof_coverage_passed"],
        "ready_for_formal_runner_authorization": smoke_summary["ready_for_formal_runner_authorization"],
        "formal_400_epoch_run_started": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
