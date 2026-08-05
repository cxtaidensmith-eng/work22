from __future__ import annotations

import argparse
import json
import math
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query
import run_selective_boundary_fixed_10fold as cv


MODEL_NAME = "Original Query Baseline (Q-noOrth)"
SCHEMA_VERSION = "query-baseline-gpu-10fold-seed0-v1"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/query_baseline_gpu_10fold_seed0")
REFERENCE_FOLD_MANIFEST_REL = Path(
    "experiments/boundary_selective_fixed_mapping_10fold_seed0_v2/formal/fold_manifest.json"
)
REPORT_REL = Path("reports/query_baseline_gpu_10fold_seed0_report.txt")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
CLASS_ORDER = ("AD", "CN", "SMCI")
EXPECTED_PARAMETER_COUNT = 853_131
EXPECTED_STATE_TENSOR_COUNT = 160
EXPECTED_INITIAL_STATE_HASH = (
    "7d5450724928805a255dccf80c4772e2672cbd80c7a14e2fbd8eb73b68671bdc"
)
EXPECTED_CONFIG_HASH = (
    "5f741494141e478a49c0aa6af13e332a2cb98f15a5f29d90d0c057c7804e0cc2"
)
EXPECTED_FOLD_MANIFEST_CANONICAL_HASH = (
    "be020ede2a03d59dbd5e8bbf715cb4d097f8398f9824c122521c3240e00b6515"
)
EXPECTED_FOLD_ASSIGNMENT_HASH = (
    "8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9"
)
SOURCE_PATHS = (
    Path("Model/__init__.py"),
    Path("Model/network.py"),
    Path("Model/models.py"),
    Path("Model/layers.py"),
    Path("Loss/__init__.py"),
    Path("Loss/loss_fn.py"),
    Path("Utils/__init__.py"),
    Path("Utils/utils.py"),
    Path("Utils/data_load.py"),
    Path("Utils/graph_load.py"),
    CONFIG_REL,
    Path("scripts/run_query_free_multibranch.py"),
    Path("scripts/run_selective_boundary_fixed_10fold.py"),
    Path("scripts/run_query_baseline_gpu_10fold.py"),
)
REQUIRED_FOLD_FILES = (
    "summary.json",
    "config.ini",
    "epoch_metrics.csv",
    "best_predictions.csv",
    "final_predictions.csv",
    "best_confusion_matrix.csv",
    "final_confusion_matrix.csv",
    "query_pool_diagnostics.json",
    "auxiliary_metrics.json",
    "checkpoint_best.pt",
    "checkpoint_epoch400.pt",
    "structure_audit_report.txt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_scalar_metrics(metrics: dict) -> dict:
    return {
        name: float(metrics[name])
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    } | {"confusion_matrix": metrics["confusion_matrix"]}


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file(), f"Missing locked source file: {relative.as_posix()}")
        result[relative.as_posix()] = query.file_hash(path)
    return result


def assert_source_integrity(expected: dict[str, str], stage: str) -> None:
    current = source_hashes()
    require(current == expected, f"Locked source/config hash changed at {stage}")


def git_info() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()

    try:
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": "unknown", "commit": "unknown", "dirty": None}


def environment_snapshot(device: torch.device) -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_index": device.index,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
    }


def tensor_device_name(tensor: torch.Tensor) -> str:
    index = tensor.device.index if tensor.device.index is not None else 0
    return f"{tensor.device.type}:{index}" if tensor.device.type == "cuda" else tensor.device.type


def dataset_device_audit(dataset_dict: dict, dataset_data: dict, device: torch.device) -> dict:
    tensors = {
        "Feature": dataset_data["Feature"],
        "Label": dataset_data["Label"],
        "Adj": dataset_dict["Adj"],
        "Label_Weight": dataset_dict["Label_Weight"],
    }
    for fold, (train_mask, test_mask) in enumerate(dataset_data["Mask"]):
        tensors[f"fold_{fold:02d}_train_mask"] = train_mask
        tensors[f"fold_{fold:02d}_test_mask"] = test_mask
    device_map = {name: tensor_device_name(value) for name, value in tensors.items()}
    expected = f"cuda:{device.index if device.index is not None else 0}"
    all_cuda = all(value.is_cuda and tensor_device_name(value) == expected for value in tensors.values())
    require(all_cuda, f"Dataset/input tensor remained off GPU: {device_map}")
    return {"expected_device": expected, "tensor_devices": device_map, "all_cuda": all_cuda}


def model_device_audit(model: torch.nn.Module, device: torch.device) -> dict:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    require(bool(parameters), "Model has no parameters")
    expected = f"cuda:{device.index if device.index is not None else 0}"
    parameter_devices = {name: tensor_device_name(value) for name, value in parameters.items()}
    buffer_devices = {name: tensor_device_name(value) for name, value in buffers.items()}
    all_cuda = all(
        value.is_cuda and tensor_device_name(value) == expected
        for value in [*parameters.values(), *buffers.values()]
    )
    require(all_cuda, "A model parameter or buffer remained off GPU")
    return {
        "expected_device": expected,
        "parameter_count_tensors": len(parameters),
        "buffer_count_tensors": len(buffers),
        "unique_parameter_devices": sorted(set(parameter_devices.values())),
        "unique_buffer_devices": sorted(set(buffer_devices.values())),
        "all_parameters_and_buffers_cuda": all_cuda,
    }


def criterion_device_audit(criterion, device: torch.device) -> dict:
    tensors = {"criterion.weight": criterion.weight}
    if criterion.main_loss.weight is not None:
        tensors["criterion.main_loss.weight"] = criterion.main_loss.weight
    for idx, auxiliary_loss in enumerate(criterion.aux_losses, start=1):
        if auxiliary_loss.weight is not None:
            tensors[f"criterion.aux_losses.{idx}.weight"] = auxiliary_loss.weight
    expected = f"cuda:{device.index if device.index is not None else 0}"
    device_map = {name: tensor_device_name(value) for name, value in tensors.items()}
    all_cuda = all(value.is_cuda and tensor_device_name(value) == expected for value in tensors.values())
    require(all_cuda, f"Criterion weight remained off GPU: {device_map}")
    return {"expected_device": expected, "tensor_devices": device_map, "all_cuda": all_cuda}


def query_structure_audit(model) -> dict:
    audit = query.baseline_structure_audit(model)
    state = model.state_dict()
    audit.update(
        {
            "category_branch_variant": model.category_branch_variant,
            "category_branch_fusion": model.category_branch_fusion,
            "message_mlp_input_features": int(model.Message_MLP[0].in_features),
            "state_tensor_count": len(state),
            "initial_state_hash": query.tensor_hash(state),
            "latent_branches_present": hasattr(model, "latent_branches"),
            "patient_router_present": hasattr(model, "patient_boundary_router"),
            "global_weight_parameter_present": hasattr(model, "global_boundary_fusion_weights"),
            "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        }
    )
    gates = {
        "parameter_count": audit["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        "state_tensor_count": audit["state_tensor_count"] == EXPECTED_STATE_TENSOR_COUNT,
        "initial_state_hash": audit["initial_state_hash"] == EXPECTED_INITIAL_STATE_HASH,
        "original_variant": audit["category_branch_variant"] == "original",
        "three_query_pools": audit["query_pool_count"] == 3,
        "query_pool_parameters_independent": audit["query_pool_parameters_independent"],
        "auxiliary_parameters_independent": audit["auxiliary_head_parameters_independent"],
        "all_pools_have_attention": all(
            details["contains_multihead_attention"] for details in audit["query_pools"].values()
        ),
        "all_pools_have_learnable_query": all(
            details["contains_learnable_query"] for details in audit["query_pools"].values()
        ),
        "all_auxiliary_heads_binary": all(
            details["auxiliary_output_classes"] == 2 for details in audit["query_pools"].values()
        ),
        "no_latent_branch": not audit["latent_branches_present"],
        "no_patient_router": not audit["patient_router_present"],
        "no_global_weight_fusion": not audit["global_weight_parameter_present"],
        "concat_category_fusion": audit["category_branch_fusion"] == "concat",
        "message_mlp_input_288": audit["message_mlp_input_features"] == 288,
        "semantic_branch_both": audit["semantic_branch"] == "both",
        "semantic_fusion_add": audit["semantic_fusion"] == "add",
        "adjacency_none": audit["adj_mode"] == "none",
        "difformer_graph_disabled": not any(audit["difformer_graph_use_flags"]),
        "orthogonality_disabled": audit["orthogonality_loss"] is False,
        "all_parameters_trainable": audit["all_parameters_trainable"],
    }
    audit["gates"] = gates
    require(all(gates.values()), f"Original Query structure gate failed: {gates}")
    return audit


def criterion_components(criterion, logits, labels, mask, auxiliary_outputs) -> dict:
    main = criterion.main_loss(logits[mask], labels[mask])
    targets = F.one_hot(labels, num_classes=criterion.Label_num).transpose(0, 1)
    auxiliary = []
    for label_idx, (loss_fn, output) in enumerate(zip(criterion.aux_losses, auxiliary_outputs)):
        auxiliary.append(loss_fn(output[mask], targets[label_idx][mask]))
    # Match criterion_query_pool_no_orth's left-to-right accumulation exactly.
    # A reduction kernel can differ by a few float32 ULPs and must not trigger
    # the loss-formula integrity gate.
    auxiliary_sum = logits.new_zeros(())
    for value in auxiliary:
        auxiliary_sum = auxiliary_sum + value
    total = main + auxiliary_sum
    return {
        "main": main,
        "query_1_aux": auxiliary[0],
        "query_2_aux": auxiliary[1],
        "query_3_aux": auxiliary[2],
        "auxiliary_sum": auxiliary_sum,
        "total": total,
    }


def scalar_components(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in components.items()}


def outputs_shape_and_device_audit(logits, representations, auxiliary_outputs, node_count, device):
    tensors = [logits, *representations, *auxiliary_outputs]
    expected = f"cuda:{device.index if device.index is not None else 0}"
    result = {
        "main_logits_shape": list(logits.shape),
        "representation_shapes": [list(value.shape) for value in representations],
        "auxiliary_shapes": [list(value.shape) for value in auxiliary_outputs],
        "tensor_devices": [tensor_device_name(value) for value in tensors],
        "main_logits_ok": tuple(logits.shape) == (node_count, 3),
        "representations_ok": len(representations) == 3
        and all(tuple(value.shape) == (node_count, 96) for value in representations),
        "auxiliary_ok": len(auxiliary_outputs) == 3
        and all(tuple(value.shape) == (node_count, 2) for value in auxiliary_outputs),
        "all_cuda": all(value.is_cuda and tensor_device_name(value) == expected for value in tensors),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in tensors),
    }
    result["passed"] = all(
        result[name]
        for name in ("main_logits_ok", "representations_ok", "auxiliary_ok", "all_cuda", "all_finite")
    )
    require(result["passed"], f"Forward output GPU/shape gate failed: {result}")
    return result


def build_fresh_objects(config, dataset_dict, device):
    SET_Random(SEED)
    model = query.build_model(config, dataset_dict, "original", device)
    structure = query_structure_audit(model)
    criterion = criterion_query_pool_no_orth(
        dataset_dict, device, label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly contains state")
    require(int(scheduler.T_max) == EPOCHS, "Scheduler T_max changed")
    return model, criterion, optimizer, scheduler, structure


def gpu_preflight(config, dataset_dict, dataset_data, device) -> dict:
    require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    require(device.type == "cuda", "Configured device is not CUDA")
    data_audit = dataset_device_audit(dataset_dict, dataset_data, device)
    model, criterion, _, _, structure = build_fresh_objects(config, dataset_dict, device)
    model_audit = model_device_audit(model, device)
    criterion_audit = criterion_device_audit(criterion, device)
    model.eval()
    with torch.no_grad():
        logits, representations, auxiliary_outputs = model(dataset_data["Feature"])
        output_audit = outputs_shape_and_device_audit(
            logits,
            representations,
            auxiliary_outputs,
            int(dataset_data["Feature"].size(0)),
            device,
        )
        loss = criterion(
            logits,
            dataset_data["Label"],
            dataset_data["Mask"][0][0],
            representations,
            auxiliary_outputs,
        )
        components = criterion_components(
            criterion,
            logits,
            dataset_data["Label"],
            dataset_data["Mask"][0][0],
            auxiliary_outputs,
        )
    formula_delta = float((loss - components["total"]).detach().abs().cpu().item())
    loss_on_cuda = loss.is_cuda and tensor_device_name(loss) == data_audit["expected_device"]
    require(loss_on_cuda, "Criterion computation did not remain on CUDA")
    require(bool(torch.isfinite(loss)), "Preflight criterion returned NaN/Inf")
    require(formula_delta <= 1e-7, "Q-noOrth loss formula gate failed")
    result = {
        "passed": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "config_device": str(config.Device),
        "dataset_device_audit": data_audit,
        "model_device_audit": model_audit,
        "criterion_device_audit": criterion_audit,
        "forward_output_audit": output_audit,
        "criterion_loss_device": tensor_device_name(loss),
        "criterion_loss_finite": bool(torch.isfinite(loss)),
        "criterion_formula_delta": formula_delta,
        "criterion_components": scalar_components(components),
        "structure_audit": structure,
    }
    del model, criterion, logits, representations, auxiliary_outputs, loss, components
    torch.cuda.empty_cache()
    return result


def auxiliary_metric_bundle(auxiliary_outputs, labels, mask) -> dict:
    targets = F.one_hot(labels, num_classes=3).transpose(0, 1)
    result = {}
    for idx, (name, output) in enumerate(zip(CLASS_ORDER, auxiliary_outputs)):
        true = targets[idx][mask].detach().cpu().numpy().astype(np.int64)
        scores = output[mask].detach().cpu().numpy()
        prediction = scores.argmax(axis=-1)
        try:
            auc = float(sklearn.metrics.roc_auc_score(true, scores[:, 1]))
        except ValueError:
            auc = float("nan")
        result[f"query_{idx + 1}_{name}_vs_rest"] = {
            "acc": float(sklearn.metrics.accuracy_score(true, prediction)),
            "auc": auc,
            "positive_count": int(true.sum()),
            "negative_count": int((true == 0).sum()),
        }
    return result


def confusion_rows(matrix: list[list[int]]) -> list[dict]:
    return [
        {
            "actual": CLASS_ORDER[row_idx],
            **{f"predicted_{CLASS_ORDER[col_idx]}": int(matrix[row_idx][col_idx]) for col_idx in range(3)},
        }
        for row_idx in range(3)
    ]


def clone_cpu_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def prediction_audit(path: Path, expected_metrics: dict, fold_entry: dict) -> dict:
    rows = cv.read_prediction_rows(path)
    validation = cv.validate_prediction_metrics(rows, expected_metrics)
    source_indices = [int(row["subject_index"]) for row in rows]
    folds = {int(row["fold"]) for row in rows}
    gates = {
        "row_count": len(rows) == int(fold_entry["test_size"]),
        "fold_id": folds == {int(fold_entry["fold"])},
        "source_indices": source_indices == [int(value) for value in fold_entry["test_indices"]],
        "unique_source_indices": len(source_indices) == len(set(source_indices)),
        "metrics_reproduce": validation["passed"],
    }
    require(all(gates.values()), f"Prediction artifact gate failed for {path}: {gates}")
    return {"path": path.name, "sha256": query.file_hash(path), "gates": gates, **validation}


def run_fold(
    fold: int,
    output_dir: Path,
    config_path: Path,
    config,
    dataset_dict: dict,
    dataset_data: dict,
    device: torch.device,
    fold_manifest: dict,
    locked_sources: dict[str, str],
    preflight: dict,
) -> dict:
    assert_source_integrity(locked_sources, f"before fold {fold}")
    fold_entry = fold_manifest["folds"][fold]
    train_mask, test_mask = dataset_data["Mask"][fold]
    require(int(train_mask.sum().item()) == int(fold_entry["train_size"]), "Train size mismatch")
    require(int(test_mask.sum().item()) == int(fold_entry["test_size"]), "Test size mismatch")
    require(
        query.split_hash(dataset_dict["Index"], train_mask, test_mask) == fold_entry["split_hash"],
        "Split hash mismatch",
    )

    model, criterion, optimizer, scheduler, structure = build_fresh_objects(
        config, dataset_dict, device
    )
    require(
        structure["initial_state_hash"] == preflight["structure_audit"]["initial_state_hash"],
        "Fresh-fold initialization differs from GPU preflight",
    )
    model_device = model_device_audit(model, device)
    criterion_device = criterion_device_audit(criterion, device)
    fold_dir = output_dir / f"fold_{fold:02d}"
    require(not fold_dir.exists(), f"Refusing to overwrite fold directory: {fold_dir}")
    fold_dir.mkdir()
    shutil.copyfile(config_path, fold_dir / "config.ini")

    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    node_count = int(features.size(0))
    pool_modules = {f"query_pool_{idx + 1}": pool for idx, pool in enumerate(model.label_pools)}
    auxiliary_modules = {
        f"query_auxiliary_{idx + 1}": head for idx, head in enumerate(model._Auxi_classifier)
    }
    pool_names = list(pool_modules)
    epoch_rows = []
    query_diagnostics = []
    auxiliary_diagnostics = []
    best = None
    best_state = None
    best_rows = None
    final_state = None
    final_rows = None
    final_metrics = None
    final_auxiliary = None
    first_forward_audit = None
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, representations, auxiliary_outputs = model(features)
        forward_audit = outputs_shape_and_device_audit(
            logits, representations, auxiliary_outputs, node_count, device
        )
        if first_forward_audit is None:
            first_forward_audit = forward_audit
        loss = criterion(logits, labels, train_mask, representations, auxiliary_outputs)
        require(loss.is_cuda, f"fold {fold} epoch {epoch}: criterion loss left CUDA")
        require(bool(torch.isfinite(loss)), f"fold {fold} epoch {epoch}: non-finite train loss")
        with torch.no_grad():
            train_components_t = criterion_components(
                criterion, logits, labels, train_mask, auxiliary_outputs
            )
            formula_delta = float((loss - train_components_t["total"]).abs().cpu().item())
        require(formula_delta <= 1e-7, f"fold {fold} epoch {epoch}: loss formula changed")
        loss.backward()
        pool_gradients = {
            name: query.module_gradient_norm(module) for name, module in pool_modules.items()
        }
        auxiliary_gradients = {
            name: query.module_gradient_norm(module) for name, module in auxiliary_modules.items()
        }
        query_parameter_gradients = {
            name: float(module.query.grad.detach().norm().cpu().item())
            if module.query.grad is not None
            else 0.0
            for name, module in pool_modules.items()
        }
        require(
            all(np.isfinite(value) and value > 0.0 for value in pool_gradients.values()),
            f"fold {fold} epoch {epoch}: invalid Query Pool gradient",
        )
        require(
            all(np.isfinite(value) and value > 0.0 for value in auxiliary_gradients.values()),
            f"fold {fold} epoch {epoch}: invalid auxiliary gradient",
        )
        require(
            all(np.isfinite(value) for value in query_parameter_gradients.values()),
            f"fold {fold} epoch {epoch}: non-finite learnable Query gradient",
        )
        require(
            cv.all_parameter_gradients_finite(model),
            f"fold {fold} epoch {epoch}: non-finite parameter gradient",
        )
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            eval_logits, eval_representations, eval_auxiliary = model(features)
            outputs_shape_and_device_audit(
                eval_logits, eval_representations, eval_auxiliary, node_count, device
            )
            test_loss = criterion(
                eval_logits, labels, test_mask, eval_representations, eval_auxiliary
            )
            test_components_t = criterion_components(
                criterion, eval_logits, labels, test_mask, eval_auxiliary
            )
            require(test_loss.is_cuda, f"fold {fold} epoch {epoch}: test loss left CUDA")
            require(bool(torch.isfinite(test_loss)), f"fold {fold} epoch {epoch}: non-finite test loss")
            train_metrics_full = query.metric_bundle(
                eval_logits,
                labels,
                train_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            test_metrics_full = query.metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            test_metrics = clean_scalar_metrics(test_metrics_full)
            require(
                all(math.isfinite(float(test_metrics[name])) for name in (
                    "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"
                )),
                f"fold {fold} epoch {epoch}: non-finite metric",
            )
            similarity = query.branch_similarity(eval_representations, pool_names)
            train_auxiliary_metrics = auxiliary_metric_bundle(
                eval_auxiliary, labels, train_mask
            )
            test_auxiliary_metrics = auxiliary_metric_bundle(eval_auxiliary, labels, test_mask)

        train_components = scalar_components(train_components_t)
        test_components = scalar_components(test_components_t)
        scalar_train_metrics = clean_scalar_metrics(train_metrics_full)
        row = {
            "epoch": epoch,
            "lr_used": lr_used,
            "lr_next": lr_next,
            "train_loss": float(loss.detach().cpu().item()),
            "test_loss": float(test_loss.detach().cpu().item()),
            **{f"train_loss_{name}": value for name, value in train_components.items()},
            **{f"test_loss_{name}": value for name, value in test_components.items()},
            **{
                f"train_{name}": float(scalar_train_metrics[name])
                for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
            },
            **{
                f"test_{name}": float(test_metrics[name])
                for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
            },
            **{f"{name}_grad_norm": value for name, value in pool_gradients.items()},
            **{f"{name}_grad_norm": value for name, value in auxiliary_gradients.items()},
            **{f"{name}_query_grad_norm": value for name, value in query_parameter_gradients.items()},
            "representation_similarity_off_diagonal_mean": similarity["off_diagonal_mean"],
            "representation_similarity_off_diagonal_std": similarity["off_diagonal_std"],
        }
        epoch_rows.append(row)
        query_diagnostics.append(
            {
                "epoch": epoch,
                "query_pool_gradient_norms": pool_gradients,
                "learnable_query_gradient_norms": query_parameter_gradients,
                "auxiliary_head_gradient_norms": auxiliary_gradients,
                "representation_similarity": similarity,
            }
        )
        auxiliary_diagnostics.append(
            {
                "epoch": epoch,
                "train": train_auxiliary_metrics,
                "test": test_auxiliary_metrics,
            }
        )

        score = (
            float(test_metrics["acc"]),
            float(test_metrics["macro_auc"]),
            float(test_metrics["macro_f1"]),
        )
        is_best = best is None or score > best["selection_tuple"]
        needs_predictions = is_best or epoch == EPOCHS
        current_rows = None
        if needs_predictions:
            with torch.no_grad():
                raw_test = eval_logits[test_mask]
                adjusted, probabilities, predictions = cv.score_tensors(
                    raw_test,
                    dataset_dict["Label_Weight"],
                    float(config.logit_adjust_tau),
                )
                source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
                    test_mask.detach().cpu().numpy().astype(bool)
                ]
                truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
                current_rows = cv.make_prediction_rows(
                    fold,
                    source_indices,
                    truth,
                    raw_test.detach().cpu().numpy(),
                    adjusted.detach().cpu().numpy(),
                    probabilities.detach().cpu().numpy(),
                    predictions.detach().cpu().numpy().astype(np.int64),
                )
        if is_best:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
                "train_loss_components": deepcopy(train_components),
                "test_loss_components": deepcopy(test_components),
                "test_auxiliary_metrics": deepcopy(test_auxiliary_metrics),
                "query_pool_gradient_norms": deepcopy(pool_gradients),
                "learnable_query_gradient_norms": deepcopy(query_parameter_gradients),
                "representation_similarity": deepcopy(similarity),
            }
            best_state = clone_cpu_state(model)
            best_rows = deepcopy(current_rows)
        if epoch == EPOCHS:
            final_state = clone_cpu_state(model)
            final_rows = deepcopy(current_rows)
            final_metrics = deepcopy(test_metrics)
            final_auxiliary = deepcopy(test_auxiliary_metrics)

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"[fold {fold}] epoch={epoch:03d}/{EPOCHS} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"ACC={float(test_metrics['acc']):.4f} "
                f"Macro-AUC={float(test_metrics['macro_auc']):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    require(best is not None and best_state is not None and best_rows is not None, "No best state")
    require(final_state is not None and final_rows is not None, "No final state")
    cv.write_csv(fold_dir / "epoch_metrics.csv", epoch_rows)
    cv.write_csv(fold_dir / "best_predictions.csv", best_rows)
    cv.write_csv(fold_dir / "final_predictions.csv", final_rows)
    cv.write_csv(
        fold_dir / "best_confusion_matrix.csv",
        confusion_rows(best["metrics"]["confusion_matrix"]),
    )
    cv.write_csv(
        fold_dir / "final_confusion_matrix.csv",
        confusion_rows(final_metrics["confusion_matrix"]),
    )
    cv.write_json(
        fold_dir / "query_pool_diagnostics.json",
        {
            "query_pool_names": pool_names,
            "best_epoch": best["epoch"],
            "best": {
                "query_pool_gradient_norms": best["query_pool_gradient_norms"],
                "learnable_query_gradient_norms": best["learnable_query_gradient_norms"],
                "representation_similarity": best["representation_similarity"],
            },
            "epochs": query_diagnostics,
        },
    )
    cv.write_json(
        fold_dir / "auxiliary_metrics.json",
        {
            "definition": "three original one-vs-rest binary auxiliary heads",
            "best_epoch": best["epoch"],
            "best_test": best["test_auxiliary_metrics"],
            "final_test": final_auxiliary,
            "epochs": auxiliary_diagnostics,
        },
    )
    best_checkpoint = fold_dir / "checkpoint_best.pt"
    final_checkpoint = fold_dir / "checkpoint_epoch400.pt"
    torch.save(best_state, best_checkpoint)
    torch.save(final_state, final_checkpoint)
    checkpoint_audit = {
        "best": cv.checkpoint_roundtrip(best_checkpoint, best_state),
        "epoch400": cv.checkpoint_roundtrip(final_checkpoint, final_state),
    }
    best_prediction_audit = prediction_audit(
        fold_dir / "best_predictions.csv", best["metrics"], fold_entry
    )
    final_prediction_audit = prediction_audit(
        fold_dir / "final_predictions.csv", final_metrics, fold_entry
    )
    assert_source_integrity(locked_sources, f"after fold {fold}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "fold": fold,
        "formal_fold_passed": True,
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": fold,
            "seed": SEED,
            "epochs": EPOCHS,
            "single_model": True,
            "ensemble": False,
            "transductive_full_batch": True,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "train_size": int(train_mask.sum().item()),
            "test_size": int(test_mask.sum().item()),
            "split_hash": fold_entry["split_hash"],
            "semantic_branch": "both",
            "semantic_fusion": "add",
            "category_branch_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": EPOCHS,
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "label_smoothing": 0.05,
            "orthogonality_loss": False,
            "auxiliary_supervision": "original three one-vs-rest binary heads",
        },
        "initialization": {
            "fresh_seed": SEED,
            "fresh_model": True,
            "fresh_criterion": True,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "loaded_checkpoint": False,
            "model_state_hash": structure["initial_state_hash"],
            "optimizer_initial_state_entries": 0,
        },
        "structure_audit": structure,
        "device_audit": {"model": model_device, "criterion": criterion_device},
        "first_forward_audit": first_forward_audit,
        "result": {
            "best_epoch": best["epoch"],
            "best_selection_tuple": list(best["selection_tuple"]),
            "best_main_metrics": best["metrics"],
            "best_train_loss_components": best["train_loss_components"],
            "best_test_loss_components": best["test_loss_components"],
            "best_auxiliary_metrics": best["test_auxiliary_metrics"],
            "epoch400_main_metrics": final_metrics,
            "epoch400_auxiliary_metrics": final_auxiliary,
        },
        "runtime": {"elapsed_seconds": elapsed, "completed_utc": utc_now()},
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": {
            "best": best_prediction_audit,
            "epoch400": final_prediction_audit,
        },
        "source_integrity_after_fold": True,
    }
    report_lines = [
        MODEL_NAME,
        f"Fold {fold} formal structure audit",
        "",
        f"formal_fold_passed=True",
        f"device={device}",
        f"gpu_name={torch.cuda.get_device_name(device)}",
        f"parameter_count={structure['parameter_count']}",
        f"state_tensor_count={structure['state_tensor_count']}",
        f"initial_state_hash={structure['initial_state_hash']}",
        f"query_pool_count={structure['query_pool_count']}",
        f"auxiliary_supervision={structure['auxiliary_supervision']}",
        f"orthogonality_loss=False",
        f"semantic_fusion={structure['semantic_fusion']}",
        f"category_branch_fusion={structure['category_branch_fusion']}",
        f"DIFFormer graph flags={structure['difformer_graph_use_flags']}",
        f"best_epoch={best['epoch']}",
        f"best_metrics={best['metrics']}",
    ]
    (fold_dir / "structure_audit_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    cv.write_json(fold_dir / "summary.json", summary)
    missing = [name for name in REQUIRED_FOLD_FILES if not (fold_dir / name).is_file()]
    require(not missing, f"fold {fold}: missing formal artifacts {missing}")
    summary["artifact_integrity"] = {
        "required_count": len(REQUIRED_FOLD_FILES),
        "present_count": len(REQUIRED_FOLD_FILES),
        "missing": [],
        "all_present": True,
        "sha256": {
            name: query.file_hash(fold_dir / name)
            for name in REQUIRED_FOLD_FILES
            if name != "summary.json"
        },
    }
    cv.write_json(fold_dir / "summary.json", summary)
    print(
        f"[fold {fold}] PASS best_epoch={best['epoch']} "
        f"ACC={best['metrics']['acc']:.4f} Macro-F1={best['metrics']['macro_f1']:.4f} "
        f"BACC={best['metrics']['bacc']:.4f} Macro-AUC={best['metrics']['macro_auc']:.4f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary


def descriptive(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def aggregate_results(output_dir: Path, fold_manifest: dict, locked_sources: dict) -> dict:
    assert_source_integrity(locked_sources, "before aggregation")
    summaries = []
    fold_metric_rows = []
    oof_rows = []
    for fold in FOLDS:
        fold_dir = output_dir / f"fold_{fold:02d}"
        missing = [name for name in REQUIRED_FOLD_FILES if not (fold_dir / name).is_file()]
        require(not missing, f"fold {fold}: aggregate missing artifacts {missing}")
        summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
        require(summary.get("formal_fold_passed") is True, f"fold {fold}: not formally passed")
        summaries.append(summary)
        metrics = summary["result"]["best_main_metrics"]
        fold_metric_rows.append(
            {
                "fold": fold,
                "best_epoch": summary["result"]["best_epoch"],
                "ACC": metrics["acc"],
                "Macro-F1": metrics["macro_f1"],
                "BACC": metrics["bacc"],
                "Macro-AUC": metrics["macro_auc"],
                "Weighted-F1": metrics["weighted_f1"],
                "Params": summary["structure_audit"]["parameter_count"],
            }
        )
        oof_rows.extend(cv.read_prediction_rows(fold_dir / "best_predictions.csv"))

    require(len(oof_rows) == int(fold_manifest["sample_count"]), "OOF row count mismatch")
    oof_indices = [int(row["subject_index"]) for row in oof_rows]
    require(len(oof_indices) == len(set(oof_indices)), "OOF subject index repeated")
    require(
        set(oof_indices) == set(int(value) for value in fold_manifest["folds"][0]["train_indices"])
        | set(int(value) for value in fold_manifest["folds"][0]["test_indices"]),
        "OOF source-index coverage mismatch",
    )
    oof_metrics = cv.metrics_from_prediction_rows(oof_rows)
    fold_confusion_sum = np.sum(
        [np.asarray(summary["result"]["best_main_metrics"]["confusion_matrix"], dtype=np.int64) for summary in summaries],
        axis=0,
    )
    require(
        np.array_equal(fold_confusion_sum, np.asarray(oof_metrics["confusion_matrix"], dtype=np.int64)),
        "Aggregate confusion matrix differs from pooled OOF confusion",
    )
    metric_key_map = {
        "ACC": "ACC",
        "Macro-F1": "Macro-F1",
        "BACC": "BACC",
        "Macro-AUC": "Macro-AUC",
        "Weighted-F1": "Weighted-F1",
        "Best epoch": "best_epoch",
    }
    statistics = {
        name: descriptive([float(row[column]) for row in fold_metric_rows])
        for name, column in metric_key_map.items()
    }
    metrics_summary_rows = [
        {"Metric": name, **values} for name, values in statistics.items()
    ]
    cv.write_csv(output_dir / "fold_metrics.csv", fold_metric_rows)
    cv.write_csv(output_dir / "metrics_summary.csv", metrics_summary_rows)
    cv.write_csv(output_dir / "oof_predictions.csv", oof_rows)
    cv.write_json(
        output_dir / "oof_metrics.json",
        {
            "scope": "pooled predictions from ten per-fold selected best epochs",
            "row_count": len(oof_rows),
            "unique_subject_indices": len(set(oof_indices)),
            "metrics": oof_metrics,
        },
    )
    cv.write_csv(
        output_dir / "aggregate_confusion_matrix.csv",
        confusion_rows(fold_confusion_sum.tolist()),
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "completed_folds": list(FOLDS),
        "all_folds_passed": True,
        "protocol": {
            "dataset": "TADPOLE",
            "task": "AD_CN_SMCI",
            "seed_per_fold": SEED,
            "epochs_per_fold": EPOCHS,
            "single_model": True,
            "ensemble": False,
            "device": "cuda:0",
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "loss": "main weighted CE + three original one-vs-rest auxiliary CE",
            "orthogonality_loss": False,
        },
        "per_fold_statistics": statistics,
        "pooled_oof_performance": {
            "scope": "pooled predictions from ten per-fold selected best epochs",
            "metrics": oof_metrics,
        },
        "aggregate_confusion_matrix": fold_confusion_sum.tolist(),
        "fairness": {
            "fold_manifest_canonical_sha256": fold_manifest["canonical_sha256"],
            "fold_assignment_sha256": fold_manifest["test_fold_assignment_sha256"],
            "same_split_as_selective_boundary": True,
            "fresh_seed0_model_optimizer_scheduler_each_fold": True,
            "no_checkpoint_loading": True,
            "hardware_disclosure": (
                "This Query baseline was explicitly run on GPU. The existing Selective Boundary "
                "10-fold reference was run in a CPU PyTorch environment; split/config/protocol are "
                "matched, but compute environments differ."
            ),
        },
        "validations": {
            "fold_count_is_10": len(summaries) == 10,
            "all_fold_artifacts_complete": True,
            "oof_row_count_is_598": len(oof_rows) == 598,
            "oof_subject_indices_unique": len(set(oof_indices)) == 598,
            "aggregate_confusion_equals_oof": True,
            "source_integrity_unchanged": True,
        },
        "completed_utc": utc_now(),
    }
    require(all(aggregate["validations"].values()), "Aggregate validation failed")
    cv.write_json(output_dir / "aggregate_summary.json", aggregate)
    assert_source_integrity(locked_sources, "after aggregation")
    return aggregate


def render_final_report(aggregate: dict, manifest: dict, output_dir: Path) -> str:
    stats = aggregate["per_fold_statistics"]
    oof = aggregate["pooled_oof_performance"]["metrics"]
    lines = [
        "Query Baseline 10-Fold Single-Seed GPU Experiment",
        f"Date (UTC): {aggregate['completed_utc']}",
        "",
        "Status: PASS — all 10 folds completed; no ensemble or hyperparameter search.",
        f"GPU: {manifest['environment']['gpu_name']}",
        f"Python: {manifest['environment']['python_executable']}",
        f"PyTorch/CUDA: {manifest['environment']['torch_version']} / {manifest['environment']['torch_cuda_version']}",
        "",
        "Model and loss",
        "- Original Per_Label_Pool ×3 with learnable Query and MultiHeadAttention",
        "- Original three one-vs-rest binary auxiliary heads",
        "- Main weighted CE + all three auxiliary CE; orthogonality disabled",
        "- concat → original Message_MLP → Global_Message add → unchanged DIFFormer/classifier",
        f"- Parameters: {manifest['gpu_preflight']['structure_audit']['parameter_count']}",
        "",
        "Protocol",
        "- TADPOLE / AD_CN_SMCI; folds 0–9; fresh seed=0 per fold; 400 epochs",
        "- Single model; no ensemble; no hyperparameter search; transductive full batch",
        "- Best epoch: ACC > Macro-AUC > Macro-F1",
        f"- Fold manifest: {manifest['fold_fairness']['canonical_sha256']}",
        "",
        "Ten-fold mean ± sample SD",
        f"- ACC: {stats['ACC']['mean']:.6f} ± {stats['ACC']['std']:.6f}",
        f"- Macro-F1: {stats['Macro-F1']['mean']:.6f} ± {stats['Macro-F1']['std']:.6f}",
        f"- BACC: {stats['BACC']['mean']:.6f} ± {stats['BACC']['std']:.6f}",
        f"- Macro-AUC: {stats['Macro-AUC']['mean']:.6f} ± {stats['Macro-AUC']['std']:.6f}",
        f"- Weighted-F1: {stats['Weighted-F1']['mean']:.6f} ± {stats['Weighted-F1']['std']:.6f}",
        "",
        "Pooled OOF metrics",
        f"- ACC: {oof['acc']:.6f}",
        f"- Macro-F1: {oof['macro_f1']:.6f}",
        f"- BACC: {oof['bacc']:.6f}",
        f"- Macro-AUC: {oof['macro_auc']:.6f}",
        f"- Weighted-F1: {oof['weighted_f1']:.6f}",
        f"- Confusion matrix (AD, CN, sMCI): {oof['confusion_matrix']}",
        "",
        "Environment disclosure",
        "- This run was mandated to use GPU and all model/input/criterion device gates passed.",
        "- The completed Selective Boundary ten-fold reference used a CPU PyTorch environment.",
        "  Splits, config and protocol are matched, but exact hardware/software arithmetic is not.",
        "",
        f"Output directory: {output_dir}",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run all read-only CUDA/model/loss/split gates, then stop before creating outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.device == "cuda:0", "This experiment is locked to --device cuda:0")
    require(torch.cuda.is_available(), "torch.cuda.is_available() is False; stopping")
    require(torch.cuda.device_count() > 0, "No CUDA device is visible; stopping")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"GPU name={gpu_name}", flush=True)

    config_path = ROOT / CONFIG_REL
    output_dir = ROOT / OUTPUT_REL
    reference_manifest_path = ROOT / REFERENCE_FOLD_MANIFEST_REL
    report_path = ROOT / REPORT_REL
    require(config_path.is_file(), f"Missing config: {config_path}")
    require(query.file_hash(config_path) == EXPECTED_CONFIG_HASH, "Config hash changed")
    require(reference_manifest_path.is_file(), "Missing locked Selective fold manifest")
    require(not output_dir.exists(), f"Refusing to overwrite output directory: {output_dir}")

    config = Config_(str(ROOT), str(config_path), 0)
    config.Device = device
    require(config.Device.type == "cuda", "Config device is not CUDA")
    require(config.DATA_SET == "TADPOLE" and config.Task == "AD_CN_SMCI", "Dataset/task changed")
    require(int(config.T_max) == EPOCHS, "T_max changed")
    feature_path, dict_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    SET_Random(SEED)
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    require(tuple(class_names) == CLASS_ORDER, f"Class order changed: {class_names}")

    generated_fold_manifest = cv.build_fold_manifest(dataset_dict, dataset_data)
    reference_fold_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    require(
        generated_fold_manifest == reference_fold_manifest,
        "GPU-rebuilt fold manifest differs from the locked Selective Boundary manifest",
    )
    require(
        generated_fold_manifest["canonical_sha256"] == EXPECTED_FOLD_MANIFEST_CANONICAL_HASH,
        "Fold manifest canonical hash changed",
    )
    require(
        generated_fold_manifest["test_fold_assignment_sha256"] == EXPECTED_FOLD_ASSIGNMENT_HASH,
        "Fold assignment hash changed",
    )
    locked_sources = source_hashes()
    preflight = gpu_preflight(config, dataset_dict, dataset_data, device)
    require(preflight["passed"], "GPU preflight failed")
    assert_source_integrity(locked_sources, "after GPU preflight")
    print("GPU preflight PASS: model parameters, inputs and criterion computation are cuda:0", flush=True)
    if args.preflight_only:
        print("Preflight-only mode complete; no formal output directory was created.", flush=True)
        return

    output_dir.mkdir(parents=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cv.write_json(output_dir / "fold_manifest.json", generated_fold_manifest)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "model": MODEL_NAME,
        "git": git_info(),
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "folds": list(FOLDS),
            "fresh_seed_per_fold": SEED,
            "epochs_per_fold": EPOCHS,
            "single_model": True,
            "ensemble": False,
            "hyperparameter_search": False,
            "device": str(device),
            "transductive_full_batch": True,
            "semantic_branch": "both",
            "semantic_fusion": "add",
            "category_branch_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "optimizer": "Adam",
            "learning_rate": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": int(config.T_max),
            "scheduler_eta_min": float(config.Lr_Min),
            "logit_adjust_tau": float(config.logit_adjust_tau),
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "loss": "main weighted CE + three original one-vs-rest auxiliary CE",
            "label_smoothing": 0.05,
            "orthogonality_loss": False,
        },
        "environment": environment_snapshot(device),
        "config": {
            "path": CONFIG_REL.as_posix(),
            "sha256": query.file_hash(config_path),
        },
        "data": {
            "feature_csv": str(feature_path),
            "feature_csv_sha256": query.file_hash(Path(feature_path)),
            "modal_dict": str(dict_path),
            "modal_dict_sha256": query.file_hash(Path(dict_path)),
            "sample_count": int(dataset_data["Feature"].size(0)),
            "class_order": list(CLASS_ORDER),
        },
        "fold_fairness": {
            "reference_manifest": REFERENCE_FOLD_MANIFEST_REL.as_posix(),
            "reference_manifest_file_sha256": query.file_hash(reference_manifest_path),
            "payload_exactly_equal": True,
            "canonical_sha256": generated_fold_manifest["canonical_sha256"],
            "assignment_sha256": generated_fold_manifest["test_fold_assignment_sha256"],
        },
        "source_integrity": {
            "files": locked_sources,
            "canonical_sha256": cv.canonical_hash(locked_sources),
        },
        "gpu_preflight": preflight,
    }
    manifest["canonical_sha256"] = cv.canonical_hash(manifest)
    cv.write_json(output_dir / "protocol_manifest.json", manifest)

    for fold in FOLDS:
        run_fold(
            fold,
            output_dir,
            config_path,
            config,
            dataset_dict,
            dataset_data,
            device,
            generated_fold_manifest,
            locked_sources,
            preflight,
        )

    aggregate = aggregate_results(output_dir, generated_fold_manifest, locked_sources)
    report = render_final_report(aggregate, manifest, output_dir)
    (output_dir / "paper_report.txt").write_text(report, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
