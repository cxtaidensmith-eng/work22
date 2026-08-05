from __future__ import annotations

import argparse
import csv
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
import sklearn
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Model import HeterGraph_Model_Kmeans
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query


SCHEMA_VERSION = "ovr-aligned-shared-query-pool-v1-smoke"
MODEL_NAME = "OVR-Aligned Shared-Query Pool v1"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_ROOT_REL = Path("experiments/ovr_aligned_shared_query_v1/smoke")
REPORT_REL = Path("reports/osfq_id_v1_implementation_smoke_20260805.txt")
FOLD = 0
SEED = 0
SMOKE_EPOCHS = 3
SMOKE_WARMUP_EPOCHS = 1
FORMAL_WARMUP_EPOCHS = 20
EMA_DECAY = 0.9
CLASS_ORDER = ("AD", "CN", "SMCI")
EXPECTED_BASELINE_PARAMETERS = 853_131
EXPECTED_BASELINE_STATE_TENSORS = 160
EXPECTED_BASELINE_INITIAL_HASH = (
    "7d5450724928805a255dccf80c4772e2672cbd80c7a14e2fbd8eb73b68671bdc"
)
EXPECTED_CANDIDATE_PARAMETERS = 703_372
EXPECTED_CANDIDATE_NAMED_PARAMETERS = 130
EXPECTED_CANDIDATE_STATE_TENSORS = 138
EXPECTED_CANDIDATE_INITIAL_HASH = (
    "5fe4022cbb7af16d2e0b54aa664528ab42710798014b4dd46a6932cdf75696a2"
)
EXPECTED_COMMON_STATE_TENSORS = 134
EXPECTED_CONFIG_HASH = (
    "5f741494141e478a49c0aa6af13e332a2cb98f15a5f29d90d0c057c7804e0cc2"
)
EXPECTED_FOLD0_SPLIT_HASH = (
    "1adda8298733c24259acfa957378b356f52db9545ae9c40ef8889ee97bf3ff04"
)
HISTORICAL_BASELINE_MANIFEST_FILE_HASH = (
    "78114ce1b9cdff40d2c00c9546cb92b79d446d6212b19c0567c989c43a17327d"
)
HISTORICAL_BASELINE_MANIFEST_CANONICAL_HASH = (
    "930288891ead11b143c3485424c5f141cbcd69a3ec7e4a5d72bd7d8076398926"
)
ARMS = {
    "s0": {
        "directory": "no_anchor",
        "anchor_mode": "none",
        "label": "S0 shared Query Pool without OVR anchor",
    },
    "s1": {
        "directory": "wrong_anchor",
        "anchor_mode": "cyclic_mismatch",
        "label": "S1 shared Query Pool with cyclicly mismatched OVR anchors",
    },
    "s2": {
        "directory": "correct_anchor",
        "anchor_mode": "correct",
        "label": "S2 shared Query Pool with correctly aligned OVR anchors",
    },
}
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
    Path("scripts/run_osfq_id_v1.py"),
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "protocol_manifest.json",
    "config.ini",
    "initialization_audit.json",
    "epoch_metrics.csv",
    "best_predictions.csv",
    "final_predictions.csv",
    "best_confusion_matrix.csv",
    "final_confusion_matrix.csv",
    "checkpoint_best.pt",
    "checkpoint_epoch3.pt",
    "auxiliary_metrics.json",
    "query_pool_diagnostics.json",
    "anchor_diagnostics.json",
    "structure_audit_report.txt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    rows = []
    for idx in range(count):
        rows.append(
            {
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
            }
        )
    return rows


def metrics_from_prediction_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    adjusted = np.asarray(
        [
            [
                float(row["adjusted_score_AD"]),
                float(row["adjusted_score_CN"]),
                float(row["adjusted_score_SMCI"]),
            ]
            for row in rows
        ]
    )
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(sklearn.metrics.f1_score(truth, prediction, average="macro")),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(sklearn.metrics.roc_auc_score(onehot, adjusted)),
        "weighted_f1": float(sklearn.metrics.f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
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


def checkpoint_roundtrip(path: Path, expected_state: dict[str, torch.Tensor]) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    key_set_equal = set(loaded) == set(expected_state)
    values_equal = key_set_equal and all(
        torch.equal(loaded[key], expected_state[key]) for key in expected_state
    )
    result = {
        "path": path.name,
        "sha256": query.file_hash(path),
        "state_tensor_count": len(loaded),
        "key_set_equal": key_set_equal,
        "tensor_values_equal": values_equal,
        "passed": key_set_equal and values_equal,
    }
    require(result["passed"], f"Checkpoint roundtrip failed: {path}")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file(), f"Missing source file: {relative.as_posix()}")
        result[relative.as_posix()] = query.file_hash(path)
    return result


def assert_source_integrity(expected: dict[str, str], stage: str) -> None:
    require(source_hashes() == expected, f"Source/config hash changed at {stage}")


def environment_snapshot(device: torch.device) -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "sklearn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
    }


def build_model(config, dataset_dict: dict, device: torch.device, shared: bool, anchor_mode: str):
    return HeterGraph_Model_Kmeans(
        dataset_dict,
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
        query_pool_variant="shared" if shared else "independent",
        osfq_anchor_mode=anchor_mode if shared else "none",
        osfq_ema_decay=EMA_DECAY,
        category_branch_fusion="concat",
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
    ).to(device)


def component_hashes(state: dict[str, torch.Tensor], keys: list[str]) -> str:
    return query.tensor_hash({key: state[key] for key in keys})


def initialization_audit(baseline, candidate) -> dict:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    common = sorted(set(baseline_state) & set(candidate_state))
    differences = {
        name: float((baseline_state[name] - candidate_state[name]).abs().max().cpu())
        for name in common
    }
    mismatch_names = [name for name, value in differences.items() if value != 0.0]
    candidate_only = sorted(set(candidate_state) - set(baseline_state))
    baseline_only = sorted(set(baseline_state) - set(candidate_state))
    audit = {
        "fresh_seed": SEED,
        "state_copy_used": False,
        "checkpoint_loaded": False,
        "baseline_parameter_count": sum(p.numel() for p in baseline.parameters()),
        "baseline_named_parameter_tensors": len(list(baseline.named_parameters())),
        "baseline_state_tensor_count": len(baseline_state),
        "baseline_full_state_hash": query.tensor_hash(baseline_state),
        "candidate_parameter_count": sum(p.numel() for p in candidate.parameters()),
        "candidate_named_parameter_tensors": len(list(candidate.named_parameters())),
        "candidate_state_tensor_count": len(candidate_state),
        "candidate_full_state_hash": query.tensor_hash(candidate_state),
        "common_tensor_count": len(common),
        "baseline_common_hash": component_hashes(baseline_state, common),
        "candidate_common_hash": component_hashes(candidate_state, common),
        "max_abs_diff": max(differences.values()) if differences else 0.0,
        "mismatch_names": mismatch_names,
        "candidate_only_names": candidate_only,
        "baseline_only_names": baseline_only,
        "new_parameter_values": {"osfq_alpha": float(candidate.osfq_alpha.item())},
        "new_buffer_values": {
            "osfq_anchor_initialized": bool(candidate.osfq_anchor_initialized.item()),
            "osfq_anchor_updates": int(candidate.osfq_anchor_updates.item()),
            "osfq_anchor_ema_abs_max": float(candidate.osfq_anchor_ema.abs().max().item()),
        },
    }
    gates = {
        "baseline_parameter_count": audit["baseline_parameter_count"] == EXPECTED_BASELINE_PARAMETERS,
        "baseline_state_tensor_count": audit["baseline_state_tensor_count"] == EXPECTED_BASELINE_STATE_TENSORS,
        "baseline_initial_hash": audit["baseline_full_state_hash"] == EXPECTED_BASELINE_INITIAL_HASH,
        "candidate_parameter_count": audit["candidate_parameter_count"] == EXPECTED_CANDIDATE_PARAMETERS,
        "candidate_named_parameters": audit["candidate_named_parameter_tensors"] == EXPECTED_CANDIDATE_NAMED_PARAMETERS,
        "candidate_state_tensors": audit["candidate_state_tensor_count"] == EXPECTED_CANDIDATE_STATE_TENSORS,
        "candidate_initial_hash": audit["candidate_full_state_hash"] == EXPECTED_CANDIDATE_INITIAL_HASH,
        "common_tensor_count": audit["common_tensor_count"] == EXPECTED_COMMON_STATE_TENSORS,
        "common_hash_equal": audit["baseline_common_hash"] == audit["candidate_common_hash"],
        "max_abs_diff_zero": audit["max_abs_diff"] == 0.0,
        "mismatch_names_empty": not audit["mismatch_names"],
        "candidate_only_exact": candidate_only == [
            "osfq_alpha",
            "osfq_anchor_ema",
            "osfq_anchor_initialized",
            "osfq_anchor_updates",
        ],
        "fresh_zero_alpha": audit["new_parameter_values"]["osfq_alpha"] == 0.0,
        "fresh_zero_ema": audit["new_buffer_values"]["osfq_anchor_ema_abs_max"] == 0.0,
    }
    audit["gates"] = gates
    audit["passed"] = all(gates.values())
    require(audit["passed"], f"Initialization fairness gate failed: {gates}")
    return audit


def structure_audit(model, arm: str) -> dict:
    state = model.state_dict()
    pool = model.label_pools[0]
    auxiliary_parameter_ids = [
        id(parameter)
        for head in model._Auxi_classifier
        for parameter in head.parameters()
    ]
    difformer_flags = [
        bool(layer.use_graph) for layer in getattr(model.GCN, "layers", [])
    ]
    audit = {
        "arm": arm,
        "arm_label": ARMS[arm]["label"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_delta_vs_baseline": sum(p.numel() for p in model.parameters()) - EXPECTED_BASELINE_PARAMETERS,
        "named_parameter_tensor_count": len(list(model.named_parameters())),
        "state_tensor_count": len(state),
        "initial_state_hash": query.tensor_hash(state),
        "query_pool_variant": model.query_pool_variant,
        "registered_query_pool_count": len(model.label_pools),
        "shared_pool_parameter_count": sum(p.numel() for p in pool.parameters()),
        "pool_contains_mha": any(isinstance(module, torch.nn.MultiheadAttention) for module in pool.modules()),
        "pool_contains_ffn": hasattr(pool, "ffn"),
        "learnable_query_count": sum(1 for name, _ in model.named_parameters() if name.endswith(".query")),
        "auxiliary_head_count": len(model._Auxi_classifier),
        "auxiliary_output_sizes": [head.out_features for head in model._Auxi_classifier],
        "auxiliary_parameters_independent": len(auxiliary_parameter_ids) == len(set(auxiliary_parameter_ids)),
        "anchor_mode": model.osfq_anchor_mode,
        "anchor_ema_decay": model.osfq_ema_decay,
        "alpha_shape": list(model.osfq_alpha.shape),
        "alpha_initial": float(model.osfq_alpha.item()),
        "anchor_buffer_shape": list(model.osfq_anchor_ema.shape),
        "semantic_branch": model.semantic_branch,
        "semantic_fusion": model.semantic_fusion,
        "category_branch_fusion": model.category_branch_fusion,
        "message_mlp_input_features": int(model.Message_MLP[0].in_features),
        "adj_mode": model.adj_mode,
        "difformer_class": type(model.GCN).__name__,
        "difformer_graph_use_flags": difformer_flags,
        "global_branch_class": type(model.Global_Message).__name__,
        "orthogonality_loss": False,
        "auxiliary_supervision": "original three one-vs-rest binary heads",
        "new_adapter_present": any("adapter" in name.lower() for name, _ in model.named_modules()),
    }
    gates = {
        "one_shared_pool": audit["registered_query_pool_count"] == 1,
        "one_learnable_query": audit["learnable_query_count"] == 1,
        "shared_mha_retained": audit["pool_contains_mha"],
        "shared_ffn_retained": audit["pool_contains_ffn"],
        "three_binary_ovr_heads": audit["auxiliary_head_count"] == 3 and audit["auxiliary_output_sizes"] == [2, 2, 2],
        "auxiliary_heads_independent": audit["auxiliary_parameters_independent"],
        "no_adapter": not audit["new_adapter_present"],
        "concat_288": audit["category_branch_fusion"] == "concat" and audit["message_mlp_input_features"] == 288,
        "global_add_unchanged": audit["semantic_branch"] == "both" and audit["semantic_fusion"] == "add",
        "adjacency_none": audit["adj_mode"] == "none",
        "difformer_graph_disabled": not any(difformer_flags),
        "no_orthogonality": audit["orthogonality_loss"] is False,
        "correct_arm_mode": audit["anchor_mode"] == ARMS[arm]["anchor_mode"],
    }
    audit["gates"] = gates
    audit["passed"] = all(gates.values())
    require(audit["passed"], f"Structure gate failed: {gates}")
    return audit


def tensor_device_name(tensor: torch.Tensor) -> str:
    index = tensor.device.index if tensor.device.index is not None else 0
    return f"cuda:{index}" if tensor.is_cuda else tensor.device.type


def device_audit(model, criterion, dataset_dict, dataset_data, device) -> dict:
    expected = f"cuda:{device.index if device.index is not None else 0}"
    data_tensors = {
        "Feature": dataset_data["Feature"],
        "Label": dataset_data["Label"],
        "Adj": dataset_dict["Adj"],
        "Label_Weight": dataset_dict["Label_Weight"],
        "train_mask": dataset_data["Mask"][FOLD][0],
        "test_mask": dataset_data["Mask"][FOLD][1],
    }
    model_tensors = {
        **dict(model.named_parameters()),
        **{f"buffer:{name}": value for name, value in model.named_buffers()},
    }
    criterion_tensors = {"weight": criterion.weight}
    if criterion.main_loss.weight is not None:
        criterion_tensors["main_loss.weight"] = criterion.main_loss.weight
    for idx, loss_fn in enumerate(criterion.aux_losses):
        if loss_fn.weight is not None:
            criterion_tensors[f"aux_losses.{idx}.weight"] = loss_fn.weight
    groups = {
        "data": {name: tensor_device_name(value) for name, value in data_tensors.items()},
        "model": sorted(set(tensor_device_name(value) for value in model_tensors.values())),
        "criterion": {name: tensor_device_name(value) for name, value in criterion_tensors.items()},
    }
    passed = (
        all(value.is_cuda and tensor_device_name(value) == expected for value in data_tensors.values())
        and all(value.is_cuda and tensor_device_name(value) == expected for value in model_tensors.values())
        and all(value.is_cuda and tensor_device_name(value) == expected for value in criterion_tensors.values())
    )
    require(passed, f"CUDA device gate failed: {groups}")
    return {"expected": expected, "groups": groups, "passed": True}


def loss_components(criterion, logits, labels, mask, auxiliary_outputs) -> dict[str, torch.Tensor]:
    main = criterion.main_loss(logits[mask], labels[mask])
    targets = F.one_hot(labels, num_classes=criterion.Label_num).transpose(0, 1)
    auxiliary = [
        loss_fn(output[mask], targets[idx][mask])
        for idx, (loss_fn, output) in enumerate(zip(criterion.aux_losses, auxiliary_outputs))
    ]
    auxiliary_sum = logits.new_zeros(())
    for value in auxiliary:
        auxiliary_sum = auxiliary_sum + value
    return {
        "main": main,
        "AD_vs_rest": auxiliary[0],
        "CN_vs_rest": auxiliary[1],
        "SMCI_vs_rest": auxiliary[2],
        "auxiliary_sum": auxiliary_sum,
        "total": main + auxiliary_sum,
    }


def scalar_components(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in values.items()}


def clean_metrics(metrics: dict) -> dict:
    return {
        name: float(metrics[name])
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    } | {"confusion_matrix": metrics["confusion_matrix"]}


def auxiliary_metrics(outputs, labels, mask) -> dict:
    targets = F.one_hot(labels, num_classes=3).transpose(0, 1)
    result = {}
    for idx, (class_name, output) in enumerate(zip(CLASS_ORDER, outputs)):
        true = targets[idx][mask].detach().cpu().numpy().astype(np.int64)
        score = output[mask].detach().cpu().numpy()
        prediction = score.argmax(axis=-1)
        try:
            auc = float(sklearn.metrics.roc_auc_score(true, score[:, 1]))
        except ValueError:
            auc = float("nan")
        result[f"query_{idx + 1}_{class_name}_vs_rest"] = {
            "acc": float(sklearn.metrics.accuracy_score(true, prediction)),
            "auc": auc,
            "positive_count": int(true.sum()),
            "negative_count": int((true == 0).sum()),
        }
    return result


def pairwise_vectors(vectors: torch.Tensor, names: tuple[str, ...] = CLASS_ORDER) -> dict:
    detached = vectors.detach().float()
    normalized = F.normalize(detached, dim=-1, eps=1e-6)
    matrix = normalized @ normalized.t()
    pairs = {}
    for left in range(detached.size(0)):
        for right in range(left + 1, detached.size(0)):
            pairs[f"{names[left]}_vs_{names[right]}"] = {
                "cosine": float(matrix[left, right].cpu().item()),
                "l2": float((detached[left] - detached[right]).norm().cpu().item()),
            }
    return {
        "norms": detached.norm(dim=-1).cpu().tolist(),
        "cosine_matrix": matrix.cpu().tolist(),
        "pairs": pairs,
    }


def output_audit(logits, representations, auxiliary_outputs, node_count: int, device) -> dict:
    all_tensors = [logits, *representations, *auxiliary_outputs]
    result = {
        "main_logits_shape": list(logits.shape),
        "representation_shapes": [list(value.shape) for value in representations],
        "auxiliary_shapes": [list(value.shape) for value in auxiliary_outputs],
        "all_cuda": all(value.is_cuda for value in all_tensors),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in all_tensors),
        "devices": [tensor_device_name(value) for value in all_tensors],
    }
    result["passed"] = (
        tuple(logits.shape) == (node_count, 3)
        and len(representations) == 3
        and all(tuple(value.shape) == (node_count, 96) for value in representations)
        and len(auxiliary_outputs) == 3
        and all(tuple(value.shape) == (node_count, 2) for value in auxiliary_outputs)
        and result["all_cuda"]
        and result["all_finite"]
    )
    require(result["passed"], f"Forward output gate failed: {result}")
    return result


def clone_cpu_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def confusion_rows(matrix: list[list[int]]) -> list[dict]:
    return [
        {
            "actual": CLASS_ORDER[row],
            **{f"predicted_{CLASS_ORDER[col]}": int(matrix[row][col]) for col in range(3)},
        }
        for row in range(3)
    ]


def prediction_rows(logits, labels, mask, dataset_dict, config) -> list[dict]:
    raw = logits[mask]
    adjusted, probabilities, predictions = score_tensors(
        raw,
        dataset_dict["Label_Weight"],
        float(config.logit_adjust_tau),
    )
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    return make_prediction_rows(
        FOLD,
        source_indices,
        truth,
        raw.detach().cpu().numpy(),
        adjusted.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
        predictions.detach().cpu().numpy().astype(np.int64),
    )


def forced_path_audit(model) -> dict:
    """Non-training audit proving no/wrong/correct routing paths are distinct."""
    original_mode = model.osfq_anchor_mode
    original_active = model.osfq_anchor_active
    saved_state = clone_cpu_state(model)
    with torch.no_grad():
        model.update_osfq_anchor_ema()
        model.osfq_alpha.fill_(0.1)
        result = {}
        for name, mode in (
            ("no_anchor", "none"),
            ("wrong_anchor", "cyclic_mismatch"),
            ("correct_anchor", "correct"),
        ):
            model.osfq_anchor_mode = mode
            model.set_osfq_anchor_active(True)
            effective = model._osfq_effective_queries()[:, 0]
            result[name] = {
                "effective_query_hash": query.tensor_hash({"effective_query": effective}),
                "pairwise": pairwise_vectors(effective),
                "anchor_active": bool(model.osfq_anchor_active),
            }
    model.load_state_dict(saved_state, strict=True)
    model.osfq_anchor_mode = original_mode
    model.osfq_anchor_active = original_active
    no_pairs = result["no_anchor"]["pairwise"]["pairs"]
    gates = {
        "no_anchor_queries_identical": all(values["l2"] == 0.0 for values in no_pairs.values()),
        "wrong_differs_from_correct": result["wrong_anchor"]["effective_query_hash"] != result["correct_anchor"]["effective_query_hash"],
        "no_differs_from_wrong": result["no_anchor"]["effective_query_hash"] != result["wrong_anchor"]["effective_query_hash"],
        "no_differs_from_correct": result["no_anchor"]["effective_query_hash"] != result["correct_anchor"]["effective_query_hash"],
        "state_restored": query.tensor_hash(model.state_dict()) == EXPECTED_CANDIDATE_INITIAL_HASH,
    }
    result["gates"] = gates
    result["passed"] = all(gates.values())
    require(result["passed"], f"Forced anchor-path audit failed: {gates}")
    return result


def render_arm_report(summary: dict) -> str:
    protocol = summary["protocol"]
    structure = summary["structure_audit"]
    init = summary["initialization_audit"]
    result = summary["result"]
    best = result["best_main_metrics"]
    final = result["epoch3_main_metrics"]
    smoke = summary["smoke_gates"]
    lines = [
        f"{MODEL_NAME} — {summary['arm'].upper()} 3-Epoch Smoke Report",
        "",
        "状态：smoke test 完成；没有运行 400 epoch 正式实验。",
        "",
        "1. 隔离与协议",
        f"branch={summary['git']['branch']}",
        f"output={summary['output_directory']}",
        f"dataset={protocol['dataset']}",
        f"task={protocol['task']}",
        f"fold={protocol['fold']}",
        f"seed={protocol['seed']}",
        f"device={protocol['device']}",
        f"epochs={protocol['epochs']}",
        f"smoke_warmup_epochs={protocol['anchor_warmup_epochs']}",
        f"formal_warmup_epochs_locked_for_future={protocol['formal_anchor_warmup_epochs']}",
        f"split_hash={protocol['split_hash']}",
        "best epoch rule=ACC > Macro-AUC > Macro-F1（沿用历史协议）",
        "",
        "2. 结构",
        f"arm={summary['arm']} ({summary['arm_label']})",
        f"shared Query Pool count={structure['registered_query_pool_count']}",
        f"learnable Query count={structure['learnable_query_count']}",
        f"OVR auxiliary heads={structure['auxiliary_head_count']}",
        f"anchor_mode={structure['anchor_mode']}",
        f"parameter_count={structure['parameter_count']}",
        f"parameter_delta_vs_baseline={structure['parameter_delta_vs_baseline']}",
        f"state_tensor_count={structure['state_tensor_count']}",
        f"Global_Message={structure['global_branch_class']} (unchanged)",
        f"DIFFormer={structure['difformer_class']} (unchanged, graph disabled)",
        f"fusion={structure['semantic_fusion']} + concat into Message_MLP (unchanged)",
        "",
        "3. 初始化公平性",
        f"baseline_initial_hash={init['baseline_full_state_hash']}",
        f"candidate_initial_hash={init['candidate_full_state_hash']}",
        f"common_tensor_count={init['common_tensor_count']}",
        f"baseline_common_hash={init['baseline_common_hash']}",
        f"candidate_common_hash={init['candidate_common_hash']}",
        f"max_abs_diff={init['max_abs_diff']}",
        f"mismatch_names={init['mismatch_names']}",
        f"state_copy_used={init['state_copy_used']}",
        "",
        "4. Smoke 结果（只用于链路验证，不用于论文性能结论）",
        f"best_epoch={result['best_epoch']}",
        f"best_ACC={best['acc']}",
        f"best_Macro-F1={best['macro_f1']}",
        f"best_BACC={best['bacc']}",
        f"best_Macro-AUC={best['macro_auc']}",
        f"best_Weighted-F1={best['weighted_f1']}",
        f"epoch3_ACC={final['acc']}",
        f"epoch3_Macro-AUC={final['macro_auc']}",
        "",
        "5. Smoke 门禁",
        *[f"{name}={value}" for name, value in smoke.items()],
        "",
        "结论：forward/loss/backward/EMA/anchor/checkpoint/prediction 链路均已按该 arm 定义检查。",
        "3 epoch 指标不能用于判断正确 anchor 是否优于错误 anchor；需人工确认后才能进入 fold0 400 epoch。",
    ]
    return "\n".join(lines) + "\n"


def run_smoke(arm: str) -> dict:
    arm_spec = ARMS[arm]
    final_output_dir = ROOT / OUTPUT_ROOT_REL / arm_spec["directory"]
    output_dir = ROOT / OUTPUT_ROOT_REL / f".{arm_spec['directory']}_in_progress"
    require(not final_output_dir.exists(), f"Refusing to overwrite existing output: {final_output_dir}")
    require(not output_dir.exists(), f"A retained in-progress directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    locked_sources = source_hashes()
    require(query.file_hash(ROOT / CONFIG_REL) == EXPECTED_CONFIG_HASH, "Historical config hash changed")

    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    config = Config_(str(ROOT), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(int(config.T_max) == 400, "Historical scheduler T_max must remain 400")
    SET_Random(SEED)
    feature_path, dict_path, _, class_names = load_path(str(ROOT), config.DATA_SET, config.Task)
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    train_mask, test_mask = dataset_data["Mask"][FOLD]
    current_split_hash = query.split_hash(dataset_dict["Index"], train_mask, test_mask)
    require(current_split_hash == EXPECTED_FOLD0_SPLIT_HASH, "fold0 split hash changed")

    SET_Random(SEED)
    baseline = build_model(config, dataset_dict, device, shared=False, anchor_mode="none")
    SET_Random(SEED)
    model = build_model(
        config,
        dataset_dict,
        device,
        shared=True,
        anchor_mode=arm_spec["anchor_mode"],
    )
    init_audit = initialization_audit(baseline, model)
    structure = structure_audit(model, arm)
    forced_audit = forced_path_audit(model)
    require(query.tensor_hash(model.state_dict()) == EXPECTED_CANDIDATE_INITIAL_HASH, "Forced audit altered initial state")
    del baseline
    torch.cuda.empty_cache()

    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    device_checks = device_audit(model, criterion, dataset_dict, dataset_data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.Lr_Min)
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")

    shutil.copyfile(ROOT / CONFIG_REL, output_dir / "config.ini")
    write_json(output_dir / "initialization_audit.json", init_audit)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "git": git_info(),
        "arm": arm,
        "arm_definition": arm_spec,
        "phase": "smoke",
        "no_formal_authorization_required_or_used": True,
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": FOLD,
            "seed": SEED,
            "epochs": SMOKE_EPOCHS,
            "anchor_warmup_epochs": SMOKE_WARMUP_EPOCHS,
            "formal_anchor_warmup_epochs_for_future": FORMAL_WARMUP_EPOCHS,
            "ema_decay": EMA_DECAY,
            "transductive_full_batch": True,
            "single_model": True,
            "ensemble": False,
            "semantic_fusion": "add",
            "category_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": 400,
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "loss": "weighted main CE + AD-vs-rest CE + CN-vs-rest CE + SMCI-vs-rest CE",
            "label_smoothing": 0.05,
            "orthogonality_loss": False,
            "split_hash": current_split_hash,
            "train_size": int(train_mask.sum().item()),
            "test_size": int(test_mask.sum().item()),
        },
        "historical_baseline_reference": {
            "manifest_file_sha256": HISTORICAL_BASELINE_MANIFEST_FILE_HASH,
            "manifest_canonical_sha256": HISTORICAL_BASELINE_MANIFEST_CANONICAL_HASH,
            "initial_state_hash": EXPECTED_BASELINE_INITIAL_HASH,
            "parameter_count": EXPECTED_BASELINE_PARAMETERS,
            "state_tensor_count": EXPECTED_BASELINE_STATE_TENSORS,
        },
        "environment": environment_snapshot(device),
        "source_hashes": locked_sources,
        "structure_audit": structure,
        "initialization_audit_sha256": query.file_hash(output_dir / "initialization_audit.json"),
    }
    write_json(output_dir / "protocol_manifest.json", manifest)

    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    node_count = int(features.size(0))
    epoch_rows = []
    query_diagnostics = []
    anchor_diagnostics = []
    auxiliary_diagnostics = []
    best = None
    best_state = None
    best_rows = None
    best_logits = None
    final_state = None
    final_rows = None
    final_metrics = None
    final_logits = None
    first_forward = None
    alpha_gradients = []
    shared_pool_gradients = []
    auxiliary_head_gradients = []
    branch_output_gradients = []
    started = time.perf_counter()

    for epoch in range(1, SMOKE_EPOCHS + 1):
        assert_source_integrity(locked_sources, f"before {arm} epoch {epoch}")
        anchor_requested = arm != "s0" and epoch > SMOKE_WARMUP_EPOCHS
        model.set_osfq_anchor_active(anchor_requested)
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, representations, auxiliary_outputs = model(features)
        for representation in representations:
            representation.retain_grad()
        current_forward = output_audit(logits, representations, auxiliary_outputs, node_count, device)
        if first_forward is None:
            first_forward = current_forward
        loss = criterion(logits, labels, train_mask, representations, auxiliary_outputs)
        components_t = loss_components(criterion, logits, labels, train_mask, auxiliary_outputs)
        formula_delta = float((loss - components_t["total"]).detach().abs().cpu().item())
        require(formula_delta <= 1e-7, f"{arm} epoch {epoch}: loss formula changed")
        require(loss.is_cuda and bool(torch.isfinite(loss)), f"{arm} epoch {epoch}: invalid CUDA loss")
        loss.backward()
        pool_gradient = query.module_gradient_norm(model.label_pools[0])
        head_gradients = {
            CLASS_ORDER[idx]: query.module_gradient_norm(head)
            for idx, head in enumerate(model._Auxi_classifier)
        }
        output_gradients = {
            CLASS_ORDER[idx]: float(representation.grad.detach().norm().cpu().item())
            for idx, representation in enumerate(representations)
        }
        query_gradient = (
            float(model.label_pools[0].query.grad.detach().norm().cpu().item())
            if model.label_pools[0].query.grad is not None
            else 0.0
        )
        alpha_gradient = (
            float(model.osfq_alpha.grad.detach().cpu().item())
            if model.osfq_alpha.grad is not None
            else 0.0
        )
        require(math.isfinite(pool_gradient) and pool_gradient > 0.0, "Shared Query Pool gradient invalid")
        require(all(math.isfinite(value) and value > 0.0 for value in head_gradients.values()), "Auxiliary head gradient invalid")
        require(all(math.isfinite(value) and value > 0.0 for value in output_gradients.values()), "Per-readout output gradient invalid")
        require(math.isfinite(query_gradient) and query_gradient > 0.0, "Shared learnable Query gradient invalid")
        require(math.isfinite(alpha_gradient), "alpha gradient is non-finite")
        require(all_parameter_gradients_finite(model), "A model gradient is NaN/Inf")
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model.update_osfq_anchor_ema()
        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            eval_logits, eval_representations, eval_auxiliary = model(features)
            output_audit(eval_logits, eval_representations, eval_auxiliary, node_count, device)
            test_loss = criterion(eval_logits, labels, test_mask, eval_representations, eval_auxiliary)
            test_components_t = loss_components(criterion, eval_logits, labels, test_mask, eval_auxiliary)
            require(test_loss.is_cuda and bool(torch.isfinite(test_loss)), "Invalid test loss")
            train_metrics_full = query.metric_bundle(
                eval_logits, labels, train_mask, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
            )
            test_metrics_full = query.metric_bundle(
                eval_logits, labels, test_mask, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
            )
            train_metrics = clean_metrics(train_metrics_full)
            test_metrics = clean_metrics(test_metrics_full)
            require(
                all(math.isfinite(test_metrics[name]) for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")),
                "A smoke metric is NaN/Inf",
            )
            similarity = query.branch_similarity(eval_representations, list(CLASS_ORDER))
            query_vectors = model.last_osfq_effective_queries[:, 0]
            centered_directions = model.last_osfq_centered_directions
            query_stats = pairwise_vectors(query_vectors)
            direction_stats = pairwise_vectors(centered_directions)
            snapshot = model.osfq_anchor_snapshot()
            train_aux = auxiliary_metrics(eval_auxiliary, labels, train_mask)
            test_aux = auxiliary_metrics(eval_auxiliary, labels, test_mask)

        components = scalar_components(components_t)
        test_components = scalar_components(test_components_t)
        epoch_rows.append(
            {
                "epoch": epoch,
                "anchor_requested": anchor_requested,
                "anchor_active": bool(model.osfq_anchor_active),
                "anchor_updates": int(model.osfq_anchor_updates.item()),
                "lr_used": lr_used,
                "lr_next": lr_next,
                "train_loss": float(loss.detach().cpu().item()),
                "test_loss": float(test_loss.detach().cpu().item()),
                **{f"train_loss_{name}": value for name, value in components.items()},
                **{f"test_loss_{name}": value for name, value in test_components.items()},
                **{f"train_{name}": train_metrics[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
                **{f"test_{name}": test_metrics[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
                "shared_pool_grad_norm": pool_gradient,
                "shared_query_grad_norm": query_gradient,
                "alpha_grad": alpha_gradient,
                "alpha_value": float(model.osfq_alpha.detach().cpu().item()),
                "AD_aux_head_grad_norm": head_gradients["AD"],
                "CN_aux_head_grad_norm": head_gradients["CN"],
                "SMCI_aux_head_grad_norm": head_gradients["SMCI"],
                "AD_readout_grad_norm": output_gradients["AD"],
                "CN_readout_grad_norm": output_gradients["CN"],
                "SMCI_readout_grad_norm": output_gradients["SMCI"],
                "representation_similarity_mean": similarity["off_diagonal_mean"],
                "effective_query_AD_CN_l2": query_stats["pairs"]["AD_vs_CN"]["l2"],
                "effective_query_AD_SMCI_l2": query_stats["pairs"]["AD_vs_SMCI"]["l2"],
                "effective_query_CN_SMCI_l2": query_stats["pairs"]["CN_vs_SMCI"]["l2"],
                "anchor_contribution_mean": float(np.mean(snapshot["anchor_contribution_norms"])),
            }
        )
        query_diagnostics.append(
            {
                "epoch": epoch,
                "shared_pool_gradient_norm": pool_gradient,
                "shared_query_gradient_norm": query_gradient,
                "alpha_gradient": alpha_gradient,
                "auxiliary_head_gradient_norms": head_gradients,
                "readout_output_gradient_norms": output_gradients,
                "effective_queries": query_stats,
                "readout_representation_similarity": similarity,
            }
        )
        anchor_diagnostics.append(
            {
                "epoch": epoch,
                "requested": anchor_requested,
                "snapshot": snapshot,
                "centered_direction_stats": direction_stats,
                "raw_ema_direction_stats": pairwise_vectors(model.osfq_anchor_ema),
            }
        )
        auxiliary_diagnostics.append({"epoch": epoch, "train": train_aux, "test": test_aux})
        alpha_gradients.append(alpha_gradient)
        shared_pool_gradients.append(pool_gradient)
        auxiliary_head_gradients.extend(head_gradients.values())
        branch_output_gradients.extend(output_gradients.values())

        score = (test_metrics["acc"], test_metrics["macro_auc"], test_metrics["macro_f1"])
        current_rows = prediction_rows(eval_logits, labels, test_mask, dataset_dict, config)
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
                "train_loss_components": deepcopy(components),
                "test_loss_components": deepcopy(test_components),
                "auxiliary_metrics": deepcopy(test_aux),
                "anchor_snapshot": deepcopy(snapshot),
            }
            best_state = clone_cpu_state(model)
            best_rows = deepcopy(current_rows)
            best_logits = eval_logits.detach().cpu().clone()
        if epoch == SMOKE_EPOCHS:
            final_state = clone_cpu_state(model)
            final_rows = deepcopy(current_rows)
            final_metrics = deepcopy(test_metrics)
            final_logits = eval_logits.detach().cpu().clone()

        assert_source_integrity(locked_sources, f"after {arm} epoch {epoch}")
        print(
            f"[{arm}] epoch={epoch}/{SMOKE_EPOCHS} active={model.osfq_anchor_active} "
            f"alpha={float(model.osfq_alpha.detach().cpu()):+.6f} "
            f"loss={float(loss.detach().cpu()):.4f} ACC={test_metrics['acc']:.4f} "
            f"Macro-AUC={test_metrics['macro_auc']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    require(
        best is not None and best_state is not None and best_rows is not None and best_logits is not None,
        "Best state missing",
    )
    require(
        final_state is not None and final_rows is not None and final_metrics is not None and final_logits is not None,
        "Final state missing",
    )
    alpha_gate = (
        all(abs(value) <= 1e-12 for value in alpha_gradients)
        if arm == "s0"
        else any(abs(value) > 1e-12 for value in alpha_gradients[SMOKE_WARMUP_EPOCHS:])
    )
    final_query_pairs = query_diagnostics[-1]["effective_queries"]["pairs"]
    query_separation_gate = (
        all(values["l2"] == 0.0 for values in final_query_pairs.values())
        if arm == "s0"
        else all(values["l2"] > 0.0 for values in final_query_pairs.values())
    )
    smoke_gates = {
        "forward_shapes_and_cuda": bool(first_forward["passed"]),
        "loss_formula_exact": all(row["train_loss"] == row["train_loss_total"] for row in epoch_rows),
        "all_losses_and_metrics_finite": all(
            math.isfinite(float(value))
            for row in epoch_rows
            for key, value in row.items()
            if isinstance(value, (float, int)) and key not in {"epoch"}
        ),
        "shared_pool_gradient_nonzero": all(value > 0.0 for value in shared_pool_gradients),
        "all_auxiliary_head_gradients_nonzero": all(value > 0.0 for value in auxiliary_head_gradients),
        "all_readout_output_gradients_nonzero": all(value > 0.0 for value in branch_output_gradients),
        "alpha_gradient_matches_arm": alpha_gate,
        "effective_query_separation_matches_arm": query_separation_gate,
        "ema_initialized": bool(model.osfq_anchor_initialized.item()),
        "ema_update_count_three": int(model.osfq_anchor_updates.item()) == SMOKE_EPOCHS,
        "anchor_activation_schedule": (
            all(not item["snapshot"]["active"] for item in anchor_diagnostics)
            if arm == "s0"
            else (
                not anchor_diagnostics[0]["snapshot"]["active"]
                and all(item["snapshot"]["active"] for item in anchor_diagnostics[1:])
            )
        ),
        "forced_path_audit": bool(forced_audit["passed"]),
    }
    require(all(smoke_gates.values()), f"Smoke gate failed for {arm}: {smoke_gates}")

    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(output_dir / "best_predictions.csv", best_rows)
    write_csv(output_dir / "final_predictions.csv", final_rows)
    write_csv(output_dir / "best_confusion_matrix.csv", confusion_rows(best["metrics"]["confusion_matrix"]))
    write_csv(output_dir / "final_confusion_matrix.csv", confusion_rows(final_metrics["confusion_matrix"]))
    write_json(
        output_dir / "query_pool_diagnostics.json",
        {
            "definition": "one shared Per_Label_Pool called three times; OVR heads remain independent",
            "forced_path_audit": forced_audit,
            "best_epoch": best["epoch"],
            "epochs": query_diagnostics,
        },
    )
    write_json(
        output_dir / "anchor_diagnostics.json",
        {
            "definition": "stop-gradient EMA of normalized (positive-row minus rest-row) OVR head normals",
            "centering": "subtract three-class mean",
            "normalization": "global RMS across three centered direction norms",
            "effective_query": "q_shared + alpha * centered_ema_direction",
            "arm_mapping": {
                "s0": "zero anchor contribution",
                "s1": "cyclic mapping [1,2,0]",
                "s2": "correct mapping [0,1,2]",
            },
            "smoke_warmup_epochs": SMOKE_WARMUP_EPOCHS,
            "formal_warmup_epochs_for_future": FORMAL_WARMUP_EPOCHS,
            "ema_decay": EMA_DECAY,
            "epochs": anchor_diagnostics,
        },
    )
    write_json(
        output_dir / "auxiliary_metrics.json",
        {
            "definition": "unchanged three-class-indexed one-vs-rest binary auxiliary supervision",
            "best_epoch": best["epoch"],
            "best_test": best["auxiliary_metrics"],
            "epochs": auxiliary_diagnostics,
        },
    )
    best_checkpoint = output_dir / "checkpoint_best.pt"
    final_checkpoint = output_dir / "checkpoint_epoch3.pt"
    torch.save(best_state, best_checkpoint)
    torch.save(final_state, final_checkpoint)
    state_roundtrip = {
        "best": checkpoint_roundtrip(best_checkpoint, best_state),
        "epoch3": checkpoint_roundtrip(final_checkpoint, final_state),
    }

    def functional_roundtrip(path: Path, expected_logits: torch.Tensor, checkpoint_epoch: int) -> dict:
        SET_Random(SEED)
        restored = build_model(
            config,
            dataset_dict,
            device,
            shared=True,
            anchor_mode=arm_spec["anchor_mode"],
        )
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        expected_active = arm != "s0" and checkpoint_epoch > SMOKE_WARMUP_EPOCHS
        restored.set_osfq_anchor_active(expected_active)
        restored.eval()
        with torch.no_grad():
            restored_logits, restored_representations, restored_auxiliary = restored(features)
            restored_forward = output_audit(
                restored_logits,
                restored_representations,
                restored_auxiliary,
                node_count,
                device,
            )
        max_abs_diff = float(
            (restored_logits.detach().cpu() - expected_logits).abs().max().item()
        )
        result = {
            "checkpoint_epoch": checkpoint_epoch,
            "expected_anchor_active": expected_active,
            "restored_anchor_active": bool(restored.osfq_anchor_active),
            "max_abs_logit_diff": max_abs_diff,
            "forward_audit": restored_forward,
            "passed": bool(restored.osfq_anchor_active) == expected_active and max_abs_diff == 0.0,
        }
        del restored
        require(result["passed"], f"Functional checkpoint roundtrip failed: {result}")
        return result

    checkpoint_audit = {
        "state_roundtrip": state_roundtrip,
        "functional_roundtrip": {
            "best": functional_roundtrip(best_checkpoint, best_logits, best["epoch"]),
            "epoch3": functional_roundtrip(final_checkpoint, final_logits, SMOKE_EPOCHS),
        },
    }
    require(
        state_roundtrip["best"]["passed"] and state_roundtrip["epoch3"]["passed"],
        "Checkpoint state roundtrip failed",
    )
    best_prediction_audit = validate_prediction_metrics(best_rows, best["metrics"])
    final_prediction_audit = validate_prediction_metrics(final_rows, final_metrics)
    require(best_prediction_audit["passed"] and final_prediction_audit["passed"], "Prediction metrics do not reproduce")
    require(len(best_rows) == int(test_mask.sum().item()) and len(final_rows) == int(test_mask.sum().item()), "Prediction row count mismatch")
    assert_source_integrity(locked_sources, f"after {arm} smoke")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "arm": arm,
        "arm_label": arm_spec["label"],
        "smoke_passed": True,
        "formal_400_epoch_run": False,
        "git": git_info(),
        "output_directory": str(final_output_dir.relative_to(ROOT)).replace("\\", "/"),
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": FOLD,
            "seed": SEED,
            "epochs": SMOKE_EPOCHS,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "transductive_full_batch": True,
            "single_model": True,
            "ensemble": False,
            "anchor_warmup_epochs": SMOKE_WARMUP_EPOCHS,
            "formal_anchor_warmup_epochs": FORMAL_WARMUP_EPOCHS,
            "ema_decay": EMA_DECAY,
            "split_hash": current_split_hash,
            "train_size": int(train_mask.sum().item()),
            "test_size": int(test_mask.sum().item()),
            "semantic_fusion": "add",
            "category_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": 400,
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "orthogonality_loss": False,
        },
        "structure_audit": structure,
        "initialization_audit": init_audit,
        "device_audit": device_checks,
        "first_forward_audit": first_forward,
        "smoke_gates": smoke_gates,
        "result": {
            "best_epoch": best["epoch"],
            "best_selection_tuple": list(best["selection_tuple"]),
            "best_main_metrics": best["metrics"],
            "best_loss_components": best["test_loss_components"],
            "best_anchor_snapshot": best["anchor_snapshot"],
            "best_anchor_active": bool(best["anchor_snapshot"]["active"]),
            "epoch3_main_metrics": final_metrics,
        },
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": {
            "best": best_prediction_audit,
            "epoch3": final_prediction_audit,
            "row_count": len(best_rows),
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "completed_utc": utc_now()},
        "source_integrity_after_smoke": True,
    }
    (output_dir / "structure_audit_report.txt").write_text(render_arm_report(summary), encoding="utf-8")
    write_json(output_dir / "summary.json", summary)
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    require(not missing, f"Missing required artifacts for {arm}: {missing}")
    summary["artifact_integrity"] = {
        "required_count": len(REQUIRED_ARTIFACTS),
        "present_count": len(REQUIRED_ARTIFACTS),
        "missing": [],
        "all_present": True,
        "sha256": {
            name: query.file_hash(output_dir / name)
            for name in REQUIRED_ARTIFACTS
            if name != "summary.json"
        },
    }
    write_json(output_dir / "summary.json", summary)
    output_dir.rename(final_output_dir)
    print(
        f"[{arm}] PASS best_epoch={best['epoch']} ACC={best['metrics']['acc']:.4f} "
        f"Macro-F1={best['metrics']['macro_f1']:.4f} BACC={best['metrics']['bacc']:.4f} "
        f"Macro-AUC={best['metrics']['macro_auc']:.4f} artifacts={len(REQUIRED_ARTIFACTS)}/{len(REQUIRED_ARTIFACTS)}",
        flush=True,
    )
    return summary


def render_combined_report(summaries: dict[str, dict]) -> str:
    first = summaries["s0"]
    init = first["initialization_audit"]
    structure = first["structure_audit"]
    lines = [
        "OVR-Aligned Shared-Query Pool v1 实现与 3 Epoch Smoke Test 总结",
        "日期：2026-08-05",
        "",
        "状态：S0、S1、S2 均完成 CUDA 3 epoch smoke；未运行 400 epoch 正式实验。",
        "",
        "一、实验隔离",
        f"branch={first['git']['branch']}",
        f"worktree={ROOT}",
        f"output_root={OUTPUT_ROOT_REL.as_posix()}",
        "历史分支、历史模型和历史实验结果均未覆盖。",
        "",
        "二、最终实现",
        "1. 三个独立 Per_Label_Pool 替换为一个共享 Per_Label_Pool；共享 learnable Query、MHA、输出投影和 FFN。",
        "2. 三个原始 OVR auxiliary heads 保持独立，loss 仍为 L_main + L_AD-rest + L_CN-rest + L_sMCI-rest。",
        "3. OVR方向 p_k=normalize(stopgrad(W_positive-W_rest))；训练后更新 EMA。",
        "4. EMA方向先跨类别中心化，再以三方向的全局 RMS 归一化。",
        "5. q_effective,k=q_shared + alpha*d_k；alpha 是唯一新增可学习标量，初始化为0。",
        "6. 没有 adapter、新 attention、新 graph、新 loss、routing 或 ensemble。",
        "",
        "三、三臂可辨识对照",
        "S0=no anchor：检验共享 Query Pool 本身。",
        "S1=cyclic wrong anchor [1,2,0]：参数完全相同，只打乱类别对应关系。",
        "S2=correct anchor [0,1,2]：检验正确 OVR 对齐是否优于错误对齐。",
        "smoke 为覆盖激活路径使用 warmup=1；未来 formal 固定 warmup=20。",
        "",
        "四、参数与初始化公平性",
        f"baseline parameters={init['baseline_parameter_count']}",
        f"candidate parameters={init['candidate_parameter_count']}",
        f"delta={structure['parameter_delta_vs_baseline']} ({100.0 * structure['parameter_delta_vs_baseline'] / init['baseline_parameter_count']:.2f}%)",
        f"candidate state tensors={init['candidate_state_tensor_count']}",
        f"common tensor count={init['common_tensor_count']}",
        f"baseline common hash={init['baseline_common_hash']}",
        f"candidate common hash={init['candidate_common_hash']}",
        f"max_abs_diff={init['max_abs_diff']}",
        f"mismatch_names={init['mismatch_names']}",
        "fresh construction only；未复制 state，未加载 checkpoint。",
        "",
        "五、Smoke 概要（指标仅用于运行检查）",
    ]
    for arm in ("s0", "s1", "s2"):
        summary = summaries[arm]
        result = summary["result"]
        metrics = result["best_main_metrics"]
        final_anchor = json.loads(
            (ROOT / summary["output_directory"] / "anchor_diagnostics.json").read_text(encoding="utf-8")
        )["epochs"][-1]["snapshot"]
        lines.extend(
            [
                f"{arm.upper()} ({summary['arm_label']}):",
                f"  PASS={summary['smoke_passed']}; best_epoch={result['best_epoch']}; ACC={metrics['acc']}; Macro-F1={metrics['macro_f1']}; BACC={metrics['bacc']}; Macro-AUC={metrics['macro_auc']}",
                f"  final alpha={final_anchor['alpha']}; active={final_anchor['active']}; contribution_norms={final_anchor['anchor_contribution_norms']}",
                f"  artifacts={summary['artifact_integrity']['present_count']}/{summary['artifact_integrity']['required_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "六、Smoke 判定",
            "- 三臂 forward、原 OVR loss、backward、EMA、checkpoint 回读和 prediction 重算全部通过。",
            "- S0 的 alpha 梯度与 Query 差异保持为0；S1/S2 在 warm-up 后 alpha 梯度非零且三个有效 Query 可分。",
            "- 强制非训练审计证明 no/wrong/correct 三条计算路径可区分，且审计后初始 state hash 完全恢复。",
            "- 3 epoch 指标不能用于选择 S2，也不能形成论文性能结论。",
            "",
            "七、进入正式实验前的建议",
            "建议可以进入 fold0/seed0/400 epoch 的 S0→S1→S2 顺序验证，但必须先获得人工确认。",
            "正式实验使用 warmup=20，历史 best epoch 规则 ACC > Macro-AUC > Macro-F1 保持不变。",
            "只有当 S2 明显优于参数匹配的 S1，且 alpha/anchor contribution 非零时，才能支持‘正确 OVR 对齐有效’。",
            "若 S2≈S1，只能说明共享池或参数共享有效，不能声称类别语义绑定。",
            "若最佳 epoch≤20，则该最佳结果来自 warm-up 阶段，也不能归因于 OVR anchor。",
            "",
            "停止：未启动 400 epoch、10-fold、ensemble 或后续模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_combined_report() -> dict:
    summaries = {}
    for arm, spec in ARMS.items():
        summary_path = ROOT / OUTPUT_ROOT_REL / spec["directory"] / "summary.json"
        require(summary_path.is_file(), f"Missing completed smoke summary: {summary_path}")
        summaries[arm] = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summaries[arm].get("smoke_passed") is True, f"{arm} did not pass smoke")
    initial_hashes = {
        summary["initialization_audit"]["candidate_full_state_hash"]
        for summary in summaries.values()
    }
    split_hashes = {summary["protocol"]["split_hash"] for summary in summaries.values()}
    require(initial_hashes == {EXPECTED_CANDIDATE_INITIAL_HASH}, "S0/S1/S2 initialization differs")
    require(split_hashes == {EXPECTED_FOLD0_SPLIT_HASH}, "S0/S1/S2 split differs")
    report = render_combined_report(summaries)
    report_path = ROOT / REPORT_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    combined = {
        "schema_version": SCHEMA_VERSION,
        "smoke_suite_passed": True,
        "arms": {
            arm: {
                "summary": str((ROOT / OUTPUT_ROOT_REL / spec["directory"] / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                "best_epoch": summaries[arm]["result"]["best_epoch"],
                "best_metrics": summaries[arm]["result"]["best_main_metrics"],
                "artifacts_complete": summaries[arm]["artifact_integrity"]["all_present"],
            }
            for arm, spec in ARMS.items()
        },
        "fairness": {
            "candidate_initial_hash_same": True,
            "split_hash_same": True,
            "parameter_count_same": True,
            "same_loss": True,
        },
        "formal_400_epoch_run": False,
        "report": str(REPORT_REL).replace("\\", "/"),
        "report_sha256": query.file_hash(report_path),
    }
    combined_path = ROOT / OUTPUT_ROOT_REL / "smoke_suite_summary.json"
    write_json(combined_path, combined)
    print(f"[suite] PASS report={REPORT_REL.as_posix()}", flush=True)
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--phase", choices=("smoke", "report"), default="smoke")
    parser.add_argument("--arm", choices=tuple(ARMS), help="Required for --phase smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "report":
        require(args.arm is None, "--arm is not used with --phase report")
        build_combined_report()
        return
    require(args.arm is not None, "--phase smoke requires --arm s0|s1|s2")
    run_smoke(args.arm)


if __name__ == "__main__":
    main()
