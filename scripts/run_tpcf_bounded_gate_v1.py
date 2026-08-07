#!/usr/bin/env python
"""Run leakage-controlled bounded gates for TPCF v1.

The expensive inner Original/TabPFN expert predictions are generated once per
outer fold and shared by the two independently trained gate-cap arms.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

import run_query_free_multibranch as original_query
import run_tabpfn_baseline_v1 as tab_base
import run_tabpfn_bounded_tuning_v1 as tab_e4
import tpcf_common as common
from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


ROOT = common.ROOT
CONFIG_PATH = ROOT / "Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini"
RESULT_ROOT = ROOT / "experiments/tpcf_v1"
BOUNDED_ROOT = RESULT_ROOT / "bounded_gate"
CACHE_ROOT = BOUNDED_ROOT / "inner_oof_cache"
SMOKE_ROOT = BOUNDED_ROOT / "smoke"
CAP_ARMS = (("tpcf_v1_cap020", 0.20), ("tpcf_v2_cap035", 0.35))
CHECKPOINT = Path(
    r"C:\Users\cxt10\AppData\Roaming\tabpfn\tabpfn-v3-classifier-v3_default.ckpt"
)
CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
CHECKPOINT_SIZE = 212_804_803
FOLDS = tuple(range(10))
INNER_FOLDS = 3
SEED = 0
ORIGINAL_EPOCHS = 400
GATE_EPOCHS = 400
EXPECTED_ORIGINAL_PARAMETERS = 853_131
FIXED_HISTORICAL_LABEL_WEIGHT = np.asarray([526, 389, 281], dtype=np.float64) / 598.0
CONTINUOUS_GATE_FEATURES = (
    "entropy_original",
    "entropy_tabpfn",
    "margin_original",
    "margin_tabpfn",
    "js_divergence",
)
ALL_GATE_FEATURES = (*CONTINUOUS_GATE_FEATURES, "prediction_disagreement")


def log(message: str) -> None:
    print(message, flush=True)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    value = float(value)
    common.require(math.isfinite(value), "Expected a finite scalar")
    return value


def set_seed() -> None:
    SET_Random(SEED)


def all_parameter_gradients_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def score_tensors(
    raw_logits: torch.Tensor, label_weight: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjusted = raw_logits - tau * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    probability = torch.softmax(adjusted, dim=-1)
    prediction = adjusted.argmax(dim=-1)
    return adjusted, probability, prediction


def build_fresh_original(
    config: Any, dataset_dict: dict[str, Any], device: torch.device
) -> tuple[nn.Module, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    set_seed()
    model = original_query.build_model(config, dataset_dict, "original", device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = criterion_query_pool_no_orth(
        dataset_dict, device, label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    common.require(len(optimizer.state) == 0, "Fresh Original optimizer contains state")
    common.require(int(scheduler.T_max) == ORIGINAL_EPOCHS, "Original scheduler T_max changed")
    return model, criterion, optimizer, scheduler, {"parameter_count": parameter_count}


class GateMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, cap: float) -> torch.Tensor:
        return float(cap) * self.network(features)


@dataclass
class DeepContext:
    protocol: dict[str, Any]
    config: Any
    dataset_dict: dict[str, Any]
    dataset_data: dict[str, Any]
    device: torch.device


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tabpfn_version": tab_base.importlib.metadata.version("tabpfn"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
    }


def load_context() -> DeepContext:
    common.require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    common.require(CONFIG_PATH.is_file(), f"Missing Original config: {CONFIG_PATH}")
    protocol = tab_base.load_protocol()
    runtime_root = Path(tempfile.gettempdir()) / "work22_tpcf_v1_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    config = Config_(str(runtime_root), str(CONFIG_PATH), 0)
    config.Device = device
    common.require(config.DATA_SET == "TADPOLE" and config.Task == "AD_CN_SMCI", "Dataset/task changed")
    common.require(int(config.T_max) == ORIGINAL_EPOCHS, "Original scheduler T_max changed")
    common.require(float(config.logit_adjust_tau) == 0.75, "Original logit adjustment changed")
    feature_path, dict_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    common.require(tuple(class_names) == common.CLASS_ORDER, "Original class order changed")
    set_seed()
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    source_order = np.asarray(dataset_dict["Index"], dtype=np.int64)
    labels = dataset_data["Label"].detach().cpu().numpy().astype(np.int64)
    common.require(np.array_equal(source_order, protocol["source_order"]), "Original/TabPFN sample order mismatch")
    common.require(np.array_equal(labels, protocol["y"]), "Original/TabPFN labels mismatch")
    common.require(
        np.allclose(
            dataset_data["Feature"].detach().cpu().numpy(),
            protocol["X"],
            rtol=0.0,
            atol=2e-6,
        ),
        "Original/TabPFN feature matrix mismatch",
    )
    actual_weight = dataset_dict["Label_Weight"].detach().cpu().numpy().astype(np.float64)
    common.require(
        np.allclose(actual_weight, FIXED_HISTORICAL_LABEL_WEIGHT, rtol=0.0, atol=5e-8),
        "Historical Original class weights changed",
    )
    # Treat these as locked historical model hyperparameters. Gate weights are
    # independently recomputed from each current outer-train fold.
    dataset_dict["Label_Weight"] = torch.as_tensor(
        FIXED_HISTORICAL_LABEL_WEIGHT, dtype=torch.float32, device=device
    )
    return DeepContext(protocol, config, dataset_dict, dataset_data, device)


def fold_positions(protocol: dict[str, Any], outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
    entry = protocol["fold_manifest"]["folds"][outer_fold]
    train = np.asarray(entry["train_positions"], dtype=np.int64)
    test = np.asarray(entry["test_positions"], dtype=np.int64)
    common.require(not set(train.tolist()) & set(test.tolist()), "Outer split overlap")
    common.require(set(train.tolist()) | set(test.tolist()) == set(range(598)), "Outer split incomplete")
    return train, test


def inner_splits(
    protocol: dict[str, Any], outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    outer_train, outer_test = fold_positions(protocol, outer_fold)
    y = protocol["y"]
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    result = []
    validation_sets: list[set[int]] = []
    for inner_fold, (relative_train, relative_validation) in enumerate(
        splitter.split(outer_train, y[outer_train])
    ):
        relative_train = np.asarray(relative_train, dtype=np.int64)
        relative_validation = np.asarray(relative_validation, dtype=np.int64)
        inner_train = outer_train[relative_train]
        inner_validation = outer_train[relative_validation]
        common.require(not set(inner_train.tolist()) & set(inner_validation.tolist()), "Inner split overlap")
        common.require(
            not set(outer_test.tolist()) & (set(inner_train.tolist()) | set(inner_validation.tolist())),
            "Outer-test leakage into inner split",
        )
        validation_sets.append(set(inner_validation.tolist()))
        result.append((relative_train, relative_validation, inner_train, inner_validation))
    common.require(set().union(*validation_sets) == set(outer_train.tolist()), "Inner validation union mismatch")
    common.require(
        all(
            not (validation_sets[left] & validation_sets[right])
            for left in range(INNER_FOLDS)
            for right in range(left + 1, INNER_FOLDS)
        ),
        "Inner validation folds overlap",
    )
    return result


def train_original_inner(
    context: DeepContext,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    epochs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed()
    model, criterion, optimizer, scheduler, structure = build_fresh_original(
        context.config, context.dataset_dict, context.device
    )
    common.require(structure["parameter_count"] == EXPECTED_ORIGINAL_PARAMETERS, "Original parameter count changed")
    features = context.dataset_data["Feature"]
    labels = context.dataset_data["Label"]
    train_mask = torch.zeros(598, dtype=torch.bool, device=context.device)
    validation_mask = torch.zeros(598, dtype=torch.bool, device=context.device)
    train_mask[torch.as_tensor(inner_train, dtype=torch.long, device=context.device)] = True
    validation_mask[torch.as_tensor(inner_validation, dtype=torch.long, device=context.device)] = True
    validation_index = torch.as_tensor(inner_validation, dtype=torch.long, device=context.device)
    common.require(not bool((train_mask & validation_mask).any()), "Original inner masks overlap")
    loss_labels = torch.zeros_like(labels)
    loss_labels[train_mask] = labels[train_mask]
    best_score: tuple[float, float, float] | None = None
    best_probability: np.ndarray | None = None
    best_epoch = -1
    first_loss = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, representations, auxiliary = model(features)
        loss = criterion(logits, loss_labels, train_mask, representations, auxiliary)
        common.require(bool(torch.isfinite(loss)), "Original inner loss is NaN/Inf")
        loss.backward()
        common.require(all_parameter_gradients_finite(model), "Original inner gradient NaN/Inf")
        if context.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), context.config.grad_clip)
        optimizer.step()
        scheduler.step()
        if first_loss is None:
            first_loss = safe_float(loss)
        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(features)
            metrics = original_query.metric_bundle(
                eval_logits,
                labels,
                validation_mask,
                context.dataset_dict["Label_Weight"],
                float(context.config.logit_adjust_tau),
            )
            score = (
                float(metrics["acc"]),
                float(metrics["macro_auc"]),
                float(metrics["macro_f1"]),
            )
            common.require(all(math.isfinite(value) for value in score), "Original selection metric is NaN/Inf")
            if best_score is None or score > best_score:
                _, probability, _ = score_tensors(
                    eval_logits[validation_index],
                    context.dataset_dict["Label_Weight"],
                    float(context.config.logit_adjust_tau),
                )
                best_probability = probability.detach().cpu().numpy().astype(np.float64)
                best_score = score
                best_epoch = epoch
    common.require(best_probability is not None and best_score is not None, "Original inner model has no best epoch")
    common.validate_probability(best_probability, "Original inner", rows=len(inner_validation))
    runtime = time.perf_counter() - started
    audit = {
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_selection_tuple": list(best_score),
        "first_train_loss": first_loss,
        "runtime_seconds": runtime,
        "parameter_count": structure["parameter_count"],
        "historical_selection_uses_adjusted_logits": True,
        "saved_probability_is_softmax_of_selected_adjusted_logits": True,
        "train_size": int(len(inner_train)),
        "validation_size": int(len(inner_validation)),
    }
    del model, criterion, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return best_probability, audit


def train_tabpfn_inner(
    context: DeepContext,
    checkpoint: Path,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    tab_e4.configure_estimators(4, 4)
    tab_base.set_reproducibility()
    gc.collect()
    torch.cuda.empty_cache()
    classifier = tab_e4.make_classifier(checkpoint)
    instance_uuid = str(uuid.uuid4())
    columns = context.protocol["feature_names"]
    X = context.protocol["X"]
    y = context.protocol["y"]
    started = time.perf_counter()
    classifier.fit(pd.DataFrame(X[inner_train], columns=columns), y[inner_train])
    torch.cuda.synchronize(0)
    fit_seconds = time.perf_counter() - started
    audit_full = tab_e4.model_audit(classifier, checkpoint, instance_uuid)
    predict_started = time.perf_counter()
    probability = np.asarray(
        classifier.predict_proba(pd.DataFrame(X[inner_validation], columns=columns)),
        dtype=np.float64,
    )
    torch.cuda.synchronize(0)
    predict_seconds = time.perf_counter() - predict_started
    common.validate_probability(probability, "TabPFN inner", rows=len(inner_validation))
    audit = {
        "instance_uuid": instance_uuid,
        "requested_n_estimators": int(audit_full["requested_n_estimators"]),
        "effective_n_estimators": int(audit_full["effective_n_estimators"]),
        "effective_classes": audit_full["effective_classes"],
        "feature_union_count": int(audit_full["feature_coverage"]["member_feature_set_union_count"]),
        "uncovered_feature_count": int(audit_full["feature_coverage"]["uncovered_feature_count"]),
        "loaded_checkpoint_model_count": int(audit_full["loaded_checkpoint_model_count"]),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "train_size": int(len(inner_train)),
        "validation_size": int(len(inner_validation)),
    }
    common.require(audit["effective_n_estimators"] == 4, "TabPFN effective estimator count changed")
    common.require(audit["effective_classes"] == [0, 1, 2], "TabPFN class order changed")
    common.require(audit["feature_union_count"] == 360 and audit["uncovered_feature_count"] == 0, "TabPFN feature coverage changed")
    del classifier, audit_full
    gc.collect()
    torch.cuda.empty_cache()
    return probability, audit


def cache_paths(outer_fold: int) -> tuple[Path, Path]:
    return (
        CACHE_ROOT / f"outer_{outer_fold:02d}_inner_oof.csv",
        CACHE_ROOT / f"outer_{outer_fold:02d}_manifest.json",
    )


def validate_cache(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    context: DeepContext,
    outer_fold: int,
    csv_path: Path,
) -> None:
    outer_train, outer_test = fold_positions(context.protocol, outer_fold)
    required = {
        "outer_fold", "shuffled_position", "subject_index", "truth", "inner_fold",
        *[f"original_probability_{name}" for name in common.CLASS_ORDER],
        *[f"tabpfn_probability_{name}" for name in common.CLASS_ORDER],
    }
    common.require(required.issubset(frame.columns), f"outer {outer_fold}: cache columns missing")
    common.require(len(frame) == len(outer_train), f"outer {outer_fold}: cache row count")
    frame = frame.sort_values("shuffled_position").reset_index(drop=True)
    common.require(np.array_equal(frame["shuffled_position"].to_numpy(dtype=np.int64), np.sort(outer_train)), "Cache positions mismatch")
    positions = frame["shuffled_position"].to_numpy(dtype=np.int64)
    common.require(not set(positions.tolist()) & set(outer_test.tolist()), "Outer-test appears in cache")
    common.require(
        np.array_equal(frame["subject_index"].to_numpy(dtype=np.int64), context.protocol["source_order"][positions]),
        "Cache subject mapping mismatch",
    )
    common.require(
        np.array_equal(frame["truth"].to_numpy(dtype=np.int64), context.protocol["y"][positions]),
        "Cache truth mismatch",
    )
    common.validate_probability(
        frame[[f"original_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
        "cached Original", rows=len(frame),
    )
    common.validate_probability(
        frame[[f"tabpfn_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
        "cached TabPFN", rows=len(frame),
    )
    common.require(set(frame["inner_fold"].astype(int)) == {0, 1, 2}, "Cache inner-fold coverage mismatch")
    common.require(manifest.get("complete") is True, "Cache manifest is not complete")
    common.require(int(manifest.get("outer_fold", -1)) == outer_fold, "Cache manifest outer fold mismatch")
    common.require(manifest.get("csv_sha256") == common.file_sha256(csv_path), "Cache CSV SHA256 mismatch")
    common.require(manifest.get("original_config_sha256") == common.file_sha256(CONFIG_PATH), "Cache Original config mismatch")
    common.require(manifest.get("tabpfn_checkpoint_sha256") == CHECKPOINT_SHA256, "Cache checkpoint provenance mismatch")
    common.require(manifest.get("outer_test_absent_from_all_inner_splits") is True, "Cache leakage audit missing")
    common.require(manifest.get("each_outer_train_sample_predicted_once") is True, "Cache coverage audit missing")
    expected_assignment: dict[int, int] = {}
    expected_splits = inner_splits(context.protocol, outer_fold)
    common.require(len(manifest.get("folds", [])) == INNER_FOLDS, "Cache split manifest count mismatch")
    for inner_fold, (_, _, inner_train, inner_validation) in enumerate(expected_splits):
        record = manifest["folds"][inner_fold]
        common.require(int(record.get("inner_fold", -1)) == inner_fold, "Cache inner-fold id mismatch")
        common.require(record.get("train_positions") == inner_train.tolist(), "Cache inner-train split mismatch")
        common.require(record.get("validation_positions") == inner_validation.tolist(), "Cache inner-validation split mismatch")
        common.require(int(record.get("outer_test_overlap_count", -1)) == 0, "Cache inner split leaks outer-test")
        for position in inner_validation:
            expected_assignment[int(position)] = inner_fold
    actual_assignment = {
        int(row.shuffled_position): int(row.inner_fold)
        for row in frame.itertuples(index=False)
    }
    common.require(actual_assignment == expected_assignment, "Cached inner-fold assignment mismatch")


def load_or_generate_cache(
    context: DeepContext,
    checkpoint: Path,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = cache_paths(outer_fold)
    if csv_path.exists() or manifest_path.exists():
        common.require(csv_path.is_file() and manifest_path.is_file(), f"outer {outer_fold}: partial cache exists")
        frame = pd.read_csv(csv_path, float_precision="round_trip")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_cache(frame, manifest, context, outer_fold, csv_path)
        log(f"outer={outer_fold} inner-expert-cache=REUSED rows={len(frame)}")
        return frame.sort_values("shuffled_position").reset_index(drop=True), manifest

    outer_train, outer_test = fold_positions(context.protocol, outer_fold)
    source_order = context.protocol["source_order"]
    y = context.protocol["y"]
    original_probability = np.full((len(outer_train), 3), np.nan, dtype=np.float64)
    tabpfn_probability = np.full((len(outer_train), 3), np.nan, dtype=np.float64)
    assignment = np.full(len(outer_train), -1, dtype=np.int64)
    split_records = []
    started = time.perf_counter()
    for inner_fold, (relative_train, relative_validation, inner_train, inner_validation) in enumerate(
        inner_splits(context.protocol, outer_fold)
    ):
        log(
            f"outer={outer_fold} inner={inner_fold} train={len(inner_train)} "
            f"validation={len(inner_validation)} Original=400ep TabPFN=E4"
        )
        original_p, original_audit = train_original_inner(
            context, inner_train, inner_validation, ORIGINAL_EPOCHS
        )
        tabpfn_p, tabpfn_audit = train_tabpfn_inner(
            context, checkpoint, inner_train, inner_validation
        )
        original_probability[relative_validation] = original_p
        tabpfn_probability[relative_validation] = tabpfn_p
        assignment[relative_validation] = inner_fold
        split_payload = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "train_positions": inner_train.tolist(),
            "validation_positions": inner_validation.tolist(),
            "train_subject_indices": source_order[inner_train].astype(int).tolist(),
            "validation_subject_indices": source_order[inner_validation].astype(int).tolist(),
        }
        split_records.append(
            {
                **split_payload,
                "split_sha256": tab_base.canonical_sha256(split_payload),
                "outer_test_overlap_count": int(
                    len(set(outer_test.tolist()) & (set(inner_train.tolist()) | set(inner_validation.tolist())))
                ),
                "original": original_audit,
                "tabpfn": tabpfn_audit,
            }
        )
        log(
            f"outer={outer_fold} inner={inner_fold} complete "
            f"Original_best_epoch={original_audit['best_epoch']}"
        )
    common.require(np.isfinite(original_probability).all(), "Original inner OOF incomplete")
    common.require(np.isfinite(tabpfn_probability).all(), "TabPFN inner OOF incomplete")
    common.require(np.all(assignment >= 0), "Inner assignment incomplete")
    frame = pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "shuffled_position": outer_train,
            "subject_index": source_order[outer_train],
            "truth": y[outer_train],
            "inner_fold": assignment,
        }
    )
    for class_id, class_name in enumerate(common.CLASS_ORDER):
        frame[f"original_probability_{class_name}"] = original_probability[:, class_id]
        frame[f"tabpfn_probability_{class_name}"] = tabpfn_probability[:, class_id]
    frame = frame.sort_values("shuffled_position").reset_index(drop=True)
    manifest = {
        "schema_version": "tpcf-inner-expert-cache-v1",
        "complete": True,
        "outer_fold": outer_fold,
        "outer_train_size": int(len(outer_train)),
        "outer_test_size": int(len(outer_test)),
        "inner_splitter": {"class": "StratifiedKFold", "n_splits": 3, "shuffle": True, "random_state": 0},
        "outer_test_positions": outer_test.tolist(),
        "outer_test_absent_from_all_inner_splits": True,
        "inner_validation_union_equals_outer_train": True,
        "inner_validation_sets_pairwise_disjoint": True,
        "each_outer_train_sample_predicted_once": True,
        "outer_test_labels_used_for_training_or_selection": False,
        "transductive_unlabelled_full_features_visible_to_original": True,
        "original_historical_fixed_label_weight": FIXED_HISTORICAL_LABEL_WEIGHT.tolist(),
        "original_historical_weight_provenance": "locked model hyperparameter from the historical 598-subject protocol",
        "original_config_sha256": common.file_sha256(CONFIG_PATH),
        "tabpfn_checkpoint_sha256": CHECKPOINT_SHA256,
        "tabpfn_requested_n_estimators": 4,
        "tabpfn_effective_n_estimators": 4,
        "folds": split_records,
        "runtime_seconds": time.perf_counter() - started,
    }
    frame.to_csv(csv_path, index=False, float_format="%.17g")
    manifest["csv_sha256"] = common.file_sha256(csv_path)
    common.write_json(manifest_path, manifest)
    readback = pd.read_csv(csv_path, float_precision="round_trip")
    validate_cache(readback, manifest, context, outer_fold, csv_path)
    log(f"outer={outer_fold} inner-expert-cache=CREATED rows={len(frame)}")
    return readback.sort_values("shuffled_position").reset_index(drop=True), manifest


def standardize_gate_features(
    train_raw: np.ndarray, test_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_raw[:, :5].mean(axis=0)
    std = train_raw[:, :5].std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    train = train_raw.copy()
    test = test_raw.copy()
    train[:, :5] = (train[:, :5] - mean) / std
    test[:, :5] = (test[:, :5] - mean) / std
    common.require(np.array_equal(train[:, 5], train_raw[:, 5]), "Disagreement feature was standardized")
    common.require(np.array_equal(test[:, 5], test_raw[:, 5]), "Test disagreement feature was standardized")
    common.require(np.isfinite(train).all() and np.isfinite(test).all(), "Gate feature NaN/Inf")
    return train, test, mean, std


def gate_logits(
    model: GateMLP,
    features: torch.Tensor,
    original_probability: torch.Tensor,
    tabpfn_probability: torch.Tensor,
    cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = model(features, cap)
    z_original = torch.log(original_probability.clamp_min(1e-12))
    z_tabpfn = torch.log(tabpfn_probability.clamp_min(1e-12))
    residual = torch.clamp(z_tabpfn - z_original, min=-2.0, max=2.0)
    final_logits = z_original + gate * residual
    probability = torch.softmax(final_logits, dim=-1)
    return gate, final_logits, probability


def arm_fold_paths(arm_name: str, outer_fold: int) -> tuple[Path, Path]:
    fold_dir = BOUNDED_ROOT / arm_name / f"fold_{outer_fold:02d}"
    return fold_dir / "predictions.csv", fold_dir / "metrics.json"


def validate_arm_fold(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    context: DeepContext,
    aligned: pd.DataFrame,
    outer_fold: int,
    cap: float,
    predictions_path: Path,
) -> None:
    _, outer_test = fold_positions(context.protocol, outer_fold)
    frame = frame.sort_values("shuffled_position").reset_index(drop=True)
    common.require(len(frame) == len(outer_test), "Gate fold row count mismatch")
    common.require(np.array_equal(frame["shuffled_position"].to_numpy(dtype=np.int64), np.sort(outer_test)), "Gate outer-test positions mismatch")
    positions = frame["shuffled_position"].to_numpy(dtype=np.int64)
    common.require(
        np.array_equal(frame["subject_index"].to_numpy(dtype=np.int64), context.protocol["source_order"][positions]),
        "Gate subject mapping mismatch",
    )
    common.require(np.array_equal(frame["fold"].to_numpy(dtype=np.int64), np.full(len(frame), outer_fold)), "Gate fold column mismatch")
    common.require(
        np.array_equal(frame["truth"].to_numpy(dtype=np.int64), context.protocol["y"][positions]),
        "Gate truth mapping mismatch",
    )
    probability = frame[[f"probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    common.validate_probability(probability, "Gate fold", rows=len(frame))
    aligned_by_subject = aligned.set_index("subject_index")
    expected = aligned_by_subject.loc[frame["subject_index"].to_numpy(dtype=np.int64)]
    for prefix in ("original", "tabpfn"):
        columns = [f"{prefix}_probability_{name}" for name in common.CLASS_ORDER]
        actual_probability = frame[columns].to_numpy(dtype=np.float64)
        expected_probability = expected[columns].to_numpy(dtype=np.float64)
        common.require(np.allclose(actual_probability, expected_probability, rtol=0.0, atol=2e-12), f"Gate {prefix} expert probability mismatch")
        common.require(
            np.array_equal(frame[f"{prefix}_prediction"].to_numpy(dtype=np.int64), np.argmax(actual_probability, axis=1)),
            f"Gate {prefix} prediction mismatch",
        )
    common.require(
        np.array_equal(frame["prediction"].to_numpy(dtype=np.int64), np.argmax(probability, axis=1)),
        "Gate final prediction mismatch",
    )
    gate = frame["gate"].to_numpy(dtype=np.float64)
    common.require(np.all((gate >= 0.0) & (gate <= cap + 1e-7)), "Gate exceeds cap")
    common.require(metrics.get("complete") is True and int(metrics["epoch"]) == GATE_EPOCHS, "Gate fold incomplete")
    common.require(metrics.get("arm") in {name for name, _ in CAP_ARMS}, "Gate arm metadata invalid")
    common.require(int(metrics.get("outer_fold", -1)) == outer_fold, "Gate metric fold mismatch")
    common.require(math.isclose(float(metrics.get("gate_cap", -1.0)), cap, rel_tol=0.0, abs_tol=1e-12), "Gate cap metadata mismatch")
    common.require(metrics.get("predictions_sha256") == common.file_sha256(predictions_path), "Gate prediction SHA256 mismatch")
    current_cache_path, _ = cache_paths(outer_fold)
    common.require(current_cache_path.is_file(), "Gate fold inner cache is missing")
    common.require(
        metrics.get("inner_cache_csv_sha256") == common.file_sha256(current_cache_path),
        "Gate fold was trained from a different inner cache",
    )
    state_payload = metrics.get("gate_state")
    common.require(isinstance(state_payload, dict), "Gate state is missing")
    replay_model = GateMLP()
    replay_model.load_state_dict(
        {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in state_payload.items()
        },
        strict=True,
    )
    replay_model.eval()
    raw_features = common.gate_features(
        frame[[f"original_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
        frame[[f"tabpfn_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
    )
    replay_features = raw_features.copy()
    replay_mean = np.asarray(metrics["continuous_feature_mean_outer_train"], dtype=np.float64)
    replay_std = np.asarray(metrics["continuous_feature_std_outer_train_ddof0"], dtype=np.float64)
    replay_features[:, :5] = (replay_features[:, :5] - replay_mean) / replay_std
    with torch.no_grad():
        replay_gate, _, replay_probability = gate_logits(
            replay_model,
            torch.as_tensor(replay_features, dtype=torch.float32),
            torch.as_tensor(
                frame[[f"original_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
                dtype=torch.float32,
            ),
            torch.as_tensor(
                frame[[f"tabpfn_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64),
                dtype=torch.float32,
            ),
            cap,
        )
    common.require(
        np.allclose(replay_gate.numpy().reshape(-1), gate, rtol=0.0, atol=2e-7),
        "Saved gate cannot be replayed from its state",
    )
    common.require(
        np.allclose(replay_probability.numpy(), probability, rtol=0.0, atol=2e-7),
        "Saved probability cannot be replayed from its gate state",
    )
    recomputed = common.metric_bundle(frame["truth"].to_numpy(dtype=np.int64), probability)
    stored = metrics.get("metrics", {})
    common.require(recomputed["correct"] == stored.get("correct"), "Gate fold correct metric mismatch")
    common.require(recomputed["confusion_matrix"] == stored.get("confusion_matrix"), "Gate fold confusion mismatch")
    for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        common.require(math.isclose(recomputed[name], float(stored.get(name, float("nan"))), rel_tol=0.0, abs_tol=2e-12), f"Gate fold {name} metric mismatch")


def train_or_load_gate_fold(
    context: DeepContext,
    aligned: pd.DataFrame,
    cache: pd.DataFrame,
    outer_fold: int,
    arm_name: str,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions_path, metrics_path = arm_fold_paths(arm_name, outer_fold)
    if predictions_path.exists() or metrics_path.exists():
        common.require(predictions_path.is_file() and metrics_path.is_file(), f"{arm_name} fold {outer_fold}: partial result exists")
        frame = pd.read_csv(predictions_path, float_precision="round_trip")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        common.require(metrics.get("arm") == arm_name, f"{arm_name} fold {outer_fold}: arm mismatch")
        validate_arm_fold(frame, metrics, context, aligned, outer_fold, cap, predictions_path)
        log(f"outer={outer_fold} arm={arm_name} gate=REUSED")
        return frame, metrics

    fold_dir = predictions_path.parent
    original_columns = [f"original_probability_{name}" for name in common.CLASS_ORDER]
    tabpfn_columns = [f"tabpfn_probability_{name}" for name in common.CLASS_ORDER]
    train_original = cache[original_columns].to_numpy(dtype=np.float64)
    train_tabpfn = cache[tabpfn_columns].to_numpy(dtype=np.float64)
    train_truth = cache["truth"].to_numpy(dtype=np.int64, copy=True)
    _, outer_test = fold_positions(context.protocol, outer_fold)
    test_subjects = context.protocol["source_order"][outer_test]
    aligned_by_subject = aligned.set_index("subject_index", drop=False)
    test_source = aligned_by_subject.loc[test_subjects]
    common.require(np.array_equal(test_source["fold"].to_numpy(dtype=np.int64), np.full(len(outer_test), outer_fold)), "Outer OOF fold mapping mismatch")
    test_original = test_source[original_columns].to_numpy(dtype=np.float64)
    test_tabpfn = test_source[tabpfn_columns].to_numpy(dtype=np.float64)
    train_raw = common.gate_features(train_original, train_tabpfn)
    test_raw = common.gate_features(test_original, test_tabpfn)
    train_features, test_features, feature_mean, feature_std = standardize_gate_features(train_raw, test_raw)

    set_seed()
    model = GateMLP().to(context.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    x_train = torch.as_tensor(train_features, dtype=torch.float32, device=context.device)
    p_original_train = torch.as_tensor(train_original, dtype=torch.float32, device=context.device)
    p_tabpfn_train = torch.as_tensor(train_tabpfn, dtype=torch.float32, device=context.device)
    y_train = torch.as_tensor(train_truth, dtype=torch.long, device=context.device)
    counts = np.bincount(train_truth, minlength=3).astype(np.float64)
    weights_np = (len(train_truth) - counts) / len(train_truth)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=context.device)
    first_loss = None
    last_ce = None
    last_penalty = None
    started = time.perf_counter()
    for epoch in range(1, GATE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        gate, final_logits, probability = gate_logits(
            model, x_train, p_original_train, p_tabpfn_train, cap
        )
        weighted_ce = F.cross_entropy(final_logits, y_train, weight=weights)
        penalty = 0.01 * gate.mean()
        loss = weighted_ce + penalty
        common.require(bool(torch.isfinite(loss)), "Gate loss is NaN/Inf")
        loss.backward()
        common.require(
            all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ),
            "Gate gradient missing or NaN/Inf",
        )
        optimizer.step()
        common.require(bool(torch.isfinite(probability).all()), "Gate train probability NaN/Inf")
        if first_loss is None:
            first_loss = safe_float(loss)
        last_ce = safe_float(weighted_ce)
        last_penalty = safe_float(penalty)

    # Compute every outer-test probability before consulting any outer-test label.
    model.eval()
    with torch.no_grad():
        x_test = torch.as_tensor(test_features, dtype=torch.float32, device=context.device)
        p_original_test = torch.as_tensor(test_original, dtype=torch.float32, device=context.device)
        p_tabpfn_test = torch.as_tensor(test_tabpfn, dtype=torch.float32, device=context.device)
        gate_test, _, probability_test = gate_logits(
            model, x_test, p_original_test, p_tabpfn_test, cap
        )
        gate_np = gate_test.detach().cpu().numpy().reshape(-1).astype(np.float64)
        probability_np = probability_test.detach().cpu().numpy().astype(np.float64)
    common.validate_probability(probability_np, "Gate outer-test", rows=len(outer_test))
    # Final evaluation starts here; labels were not inputs to the gate or epoch selection.
    test_truth = context.protocol["y"][outer_test].astype(np.int64)
    common.require(
        np.array_equal(test_source["truth"].to_numpy(dtype=np.int64), test_truth),
        "Outer OOF truth mapping mismatch",
    )
    prediction = np.argmax(probability_np, axis=1).astype(np.int64)
    original_prediction = np.argmax(test_original, axis=1).astype(np.int64)
    tabpfn_prediction = np.argmax(test_tabpfn, axis=1).astype(np.int64)
    original_correct = original_prediction == test_truth
    final_correct = prediction == test_truth
    frame = pd.DataFrame(
        {
            "fold": outer_fold,
            "shuffled_position": outer_test,
            "subject_index": test_subjects,
            "truth": test_truth,
            "original_prediction": original_prediction,
            "tabpfn_prediction": tabpfn_prediction,
            "prediction": prediction,
            "gate": gate_np,
            "changed_prediction": (prediction != original_prediction).astype(np.int64),
            "repair": ((~original_correct) & final_correct).astype(np.int64),
            "damage": (original_correct & (~final_correct)).astype(np.int64),
        }
    )
    for class_id, class_name in enumerate(common.CLASS_ORDER):
        frame[f"original_probability_{class_name}"] = test_original[:, class_id]
        frame[f"tabpfn_probability_{class_name}"] = test_tabpfn[:, class_id]
        frame[f"probability_{class_name}"] = probability_np[:, class_id]
    fold_metrics = common.metric_bundle(test_truth, probability_np)
    state = {
        name: value.detach().cpu().numpy().astype(np.float64).tolist()
        for name, value in model.state_dict().items()
    }
    metrics = {
        "schema_version": "tpcf-bounded-gate-fold-v1",
        "complete": True,
        "arm": arm_name,
        "outer_fold": outer_fold,
        "gate_cap": cap,
        "epoch": GATE_EPOCHS,
        "optimizer": {"name": "Adam", "learning_rate": 0.01, "weight_decay": 0.0001},
        "loss": "outer-train weighted CE(z_final) + 0.01*mean(capped_gate)",
        "class_weight_formula": "(N_outer_train - n_c) / N_outer_train",
        "class_counts": counts.astype(int).tolist(),
        "class_weights": weights_np.tolist(),
        "feature_names": list(ALL_GATE_FEATURES),
        "continuous_feature_mean_outer_train": feature_mean.tolist(),
        "continuous_feature_std_outer_train_ddof0": feature_std.tolist(),
        "disagreement_feature_standardized": False,
        "first_loss": first_loss,
        "epoch400_weighted_ce": last_ce,
        "epoch400_gate_penalty": last_penalty,
        "metrics": fold_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "outer_test_labels_used_for_training_or_epoch_selection": False,
        "outer_test_predictions_computed_before_evaluation": True,
        "inner_cache_csv_sha256": common.file_sha256(cache_paths(outer_fold)[0]),
        "gate_state": state,
    }
    frame = frame.sort_values("shuffled_position").reset_index(drop=True)
    fold_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(predictions_path, index=False, float_format="%.17g")
    metrics["predictions_sha256"] = common.file_sha256(predictions_path)
    common.write_json(metrics_path, metrics)
    readback = pd.read_csv(predictions_path, float_precision="round_trip")
    validate_arm_fold(readback, metrics, context, aligned, outer_fold, cap, predictions_path)
    log(
        f"outer={outer_fold} arm={arm_name} epoch=400 correct={fold_metrics['correct']}/{len(test_truth)} "
        f"ACC={fold_metrics['acc']:.6f} Macro-F1={fold_metrics['macro_f1']:.6f} "
        f"BACC={fold_metrics['bacc']:.6f} Macro-AUC={fold_metrics['macro_auc']:.6f}"
    )
    del model, optimizer, x_train, p_original_train, p_tabpfn_train, y_train
    gc.collect()
    torch.cuda.empty_cache()
    return frame, metrics


def boundary_diagnostics(
    truth: np.ndarray, original_prediction: np.ndarray, final_prediction: np.ndarray
) -> dict[str, Any]:
    original_correct = original_prediction == truth
    final_correct = final_prediction == truth
    repair = (~original_correct) & final_correct
    damage = original_correct & (~final_correct)
    boundaries = {"AD_SMCI": {0, 2}, "CN_SMCI": {1, 2}}
    result: dict[str, Any] = {}
    for name, pair in boundaries.items():
        repair_mask = np.asarray(
            [bool(repair[i]) and {int(original_prediction[i]), int(truth[i])} == pair for i in range(len(truth))]
        )
        damage_mask = np.asarray(
            [bool(damage[i]) and {int(final_prediction[i]), int(truth[i])} == pair for i in range(len(truth))]
        )
        result[name] = {
            "repairs": int(repair_mask.sum()),
            "damages": int(damage_mask.sum()),
            "net": int(repair_mask.sum() - damage_mask.sum()),
        }
    result["other"] = {
        "repairs": int(repair.sum() - sum(item["repairs"] for item in result.values())),
        "damages": int(damage.sum() - sum(item["damages"] for item in result.values())),
    }
    return result


def aggregate_arm(
    context: DeepContext,
    aligned: pd.DataFrame,
    arm_name: str,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    fold_rows = []
    runtime = 0.0
    for outer_fold in FOLDS:
        predictions_path, metrics_path = arm_fold_paths(arm_name, outer_fold)
        common.require(predictions_path.is_file() and metrics_path.is_file(), f"{arm_name}: fold {outer_fold} missing")
        frame = pd.read_csv(predictions_path, float_precision="round_trip")
        fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        common.require(fold_metrics.get("arm") == arm_name, f"{arm_name}: saved arm mismatch")
        validate_arm_fold(frame, fold_metrics, context, aligned, outer_fold, cap, predictions_path)
        frames.append(frame)
        runtime += float(fold_metrics["runtime_seconds"])
        values = fold_metrics["metrics"]
        fold_rows.append(
            {
                "fold": outer_fold,
                "epoch": GATE_EPOCHS,
                "correct": values["correct"],
                "ACC": values["acc"],
                "Macro-F1": values["macro_f1"],
                "BACC": values["bacc"],
                "Probability Macro-AUC": values["macro_auc"],
                "Weighted-F1": values["weighted_f1"],
            }
        )
    oof = pd.concat(frames, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    common.require(len(oof) == 598 and oof["subject_index"].nunique() == 598, f"{arm_name}: OOF coverage")
    common.require(np.array_equal(oof["subject_index"].to_numpy(dtype=np.int64), np.arange(598)), f"{arm_name}: OOF subject order")
    truth = oof["truth"].to_numpy(dtype=np.int64)
    probability = oof[[f"probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    original_probability = oof[[f"original_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    tabpfn_probability = oof[[f"tabpfn_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    metrics = common.metric_bundle(truth, probability)
    comparison = common.comparison_with_original(truth, original_probability, probability)
    original_prediction = np.argmax(original_probability, axis=1)
    tabpfn_prediction = np.argmax(tabpfn_probability, axis=1)
    prediction = np.argmax(probability, axis=1)
    original_correct = original_prediction == truth
    tabpfn_correct = tabpfn_prediction == truth
    gate = oof["gate"].to_numpy(dtype=np.float64)
    agree = original_prediction == tabpfn_prediction
    diagnostics = {
        **comparison,
        "exact_mcnemar_p_value": common.exact_mcnemar(
            comparison["original_only_correct"], comparison["candidate_only_correct"]
        ),
        "mean_gate": float(gate.mean()),
        "maximum_gate": float(gate.max()),
        "gate_gt_0_05_proportion": float((gate > 0.05).mean()),
        "agree_mean_gate": float(gate[agree].mean()) if bool(agree.any()) else None,
        "disagree_mean_gate": float(gate[~agree].mean()) if bool((~agree).any()) else None,
        "mean_gate_by_true_class": {
            name: float(gate[truth == class_id].mean())
            for class_id, name in enumerate(common.CLASS_ORDER)
        },
        "tabpfn_only_correct_mean_gate": float(gate[(~original_correct) & tabpfn_correct].mean()),
        "original_only_correct_mean_gate": float(gate[original_correct & (~tabpfn_correct)].mean()),
        "boundary_repairs_damages": boundary_diagnostics(truth, original_prediction, prediction),
    }
    payload = {
        "arm": arm_name,
        "gate_cap": cap,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "folds": fold_rows,
        "gate_training_runtime_seconds": runtime,
        "probability_recomputed_from_saved_oof": True,
    }
    arm_dir = BOUNDED_ROOT / arm_name
    oof.to_csv(arm_dir / "oof_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(fold_rows).to_csv(arm_dir / "fold_metrics.csv", index=False, float_format="%.17g")
    common.write_json(arm_dir / "metrics.json", payload)
    report = [
        f"# {arm_name}", "",
        f"Correct={metrics['correct']}/598; ACC={metrics['acc']:.10f}; Macro-F1={metrics['macro_f1']:.10f}; "
        f"BACC={metrics['bacc']:.10f}; Probability Macro-AUC={metrics['macro_auc']:.10f}; "
        f"Weighted-F1={metrics['weighted_f1']:.10f}; confusion={metrics['confusion_matrix']}.",
        f"Changed={diagnostics['changed_predictions']}; repairs={diagnostics['repairs']}; damages={diagnostics['damages']}; "
        f"Original-only={diagnostics['original_only_correct']}; TPCF-only={diagnostics['candidate_only_correct']}; "
        f"McNemar p={diagnostics['exact_mcnemar_p_value']:.10g}.",
        f"Mean gate={diagnostics['mean_gate']:.6f}; max gate={diagnostics['maximum_gate']:.6f}; "
        f"gate>0.05={diagnostics['gate_gt_0_05_proportion']:.4%}; "
        f"agree/disagree mean={diagnostics['agree_mean_gate']:.6f}/{diagnostics['disagree_mean_gate']:.6f}.",
        f"Gate by true class={diagnostics['mean_gate_by_true_class']}; "
        f"TabPFN-only/Original-only mean gate={diagnostics['tabpfn_only_correct_mean_gate']:.6f}/"
        f"{diagnostics['original_only_correct_mean_gate']:.6f}.",
    ]
    (arm_dir / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    return oof, payload


def comparison_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        "correct": metrics["correct"],
        "ACC": metrics["acc"],
        "Macro-F1": metrics["macro_f1"],
        "BACC": metrics["bacc"],
        "Probability Macro-AUC": metrics["macro_auc"],
        "Weighted-F1": metrics["weighted_f1"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
    }


def confidence_pattern(
    truth: np.ndarray, original_probability: np.ndarray, tabpfn_probability: np.ndarray
) -> dict[str, Any]:
    original_prediction = np.argmax(original_probability, axis=1)
    tabpfn_prediction = np.argmax(tabpfn_probability, axis=1)
    original_correct = original_prediction == truth
    tabpfn_correct = tabpfn_prediction == truth
    masks = {
        "tabpfn_only_correct": (~original_correct) & tabpfn_correct,
        "original_only_correct": original_correct & (~tabpfn_correct),
    }
    result = {}
    for name, mask in masks.items():
        result[name] = {
            "count": int(mask.sum()),
            "tabpfn_max_probability_mean": float(tabpfn_probability[mask].max(axis=1).mean()),
            "tabpfn_margin_mean": float(common.probability_margin(tabpfn_probability[mask]).mean()),
            "tabpfn_entropy_mean": float(common.probability_entropy(tabpfn_probability[mask]).mean()),
            "original_margin_mean": float(common.probability_margin(original_probability[mask]).mean()),
        }
    return result


def finalize(
    context: DeepContext,
    aligned: pd.DataFrame,
    provenance: dict[str, Any],
    formal_started: float,
) -> dict[str, Any]:
    arm_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    for arm_name, cap in CAP_ARMS:
        arm_results[arm_name] = aggregate_arm(context, aligned, arm_name, cap)
    zero_payload = json.loads(
        (RESULT_ROOT / "zero_training/best_zero_training_metrics.json").read_text(encoding="utf-8")
    )
    zero_metrics = zero_payload["best_candidate"]["metrics"]
    truth, original_probability, tabpfn_probability = common.aligned_probabilities(aligned)
    rows = [
        comparison_row("Original Query", provenance["original_metrics"]),
        comparison_row("TabPFN E4", provenance["tabpfn_metrics"]),
        comparison_row(zero_payload["best_candidate"]["name"], zero_metrics),
    ]
    for arm_name, _ in CAP_ARMS:
        rows.append(comparison_row(arm_name, arm_results[arm_name][1]["metrics"]))
    comparison_frame = pd.DataFrame(rows)
    comparison_frame.to_csv(RESULT_ROOT / "final_comparison.csv", index=False, float_format="%.17g")
    table_columns = [
        "model", "correct", "ACC", "Macro-F1", "BACC",
        "Probability Macro-AUC", "Weighted-F1", "confusion_matrix",
    ]
    markdown_table = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for row in rows:
        markdown_table.append(
            "| " + " | ".join(str(row[column]) for column in table_columns) + " |"
        )

    gate_ranked = sorted(
        CAP_ARMS,
        key=lambda item: (
            -int(arm_results[item[0]][1]["metrics"]["correct"]),
            -float(arm_results[item[0]][1]["metrics"]["macro_auc"]),
            -float(arm_results[item[0]][1]["metrics"]["macro_f1"]),
            float(item[1]),
        ),
    )
    best_gate_name, best_cap = gate_ranked[0]
    best_gate_oof, best_gate = arm_results[best_gate_name]
    gate_metrics = best_gate["metrics"]
    original_metrics = provenance["original_metrics"]
    bacc_ok = gate_metrics["bacc"] >= original_metrics["bacc"] - 0.002
    if gate_metrics["correct"] >= 557 and bacc_ok:
        decision = "GO"
    elif (
        gate_metrics["correct"] == 556
        and gate_metrics["macro_f1"] > original_metrics["macro_f1"]
        and gate_metrics["macro_auc"] > original_metrics["macro_auc"]
        and bacc_ok
    ):
        decision = "CONDITIONAL"
    else:
        decision = "NO_NET_IMPROVEMENT"

    candidates: list[tuple[str, dict[str, Any], pd.DataFrame | None, float]] = [
        ("Original Query", original_metrics, None, 0.0),
        ("TabPFN E4", provenance["tabpfn_metrics"], None, 1.0),
        (zero_payload["best_candidate"]["name"], zero_metrics, None, float(zero_payload["best_candidate"]["alpha"])),
        *[(name, arm_results[name][1]["metrics"], arm_results[name][0], cap) for name, cap in CAP_ARMS],
    ]
    overall = sorted(
        candidates,
        key=lambda item: (-int(item[1]["correct"]), -float(item[1]["macro_auc"]), -float(item[1]["macro_f1"]), item[3]),
    )[0]
    overall_name = overall[0]
    if overall_name == "Original Query":
        final_oof = aligned[["subject_index", "fold", "truth"]].copy()
        final_probability = original_probability
    elif overall_name == "TabPFN E4":
        final_oof = aligned[["subject_index", "fold", "truth"]].copy()
        final_probability = tabpfn_probability
    elif overall_name == zero_payload["best_candidate"]["name"]:
        final_oof = pd.read_csv(RESULT_ROOT / "zero_training/best_zero_training_predictions.csv", float_precision="round_trip")
        final_probability = final_oof[[f"fused_probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    else:
        final_oof = overall[2].copy()  # type: ignore[union-attr]
        final_probability = final_oof[[f"probability_{name}" for name in common.CLASS_ORDER]].to_numpy(dtype=np.float64)
    for class_id, class_name in enumerate(common.CLASS_ORDER):
        final_oof[f"selected_probability_{class_name}"] = final_probability[:, class_id]
    final_oof["selected_prediction"] = np.argmax(final_probability, axis=1)
    final_oof["selected_model"] = overall_name
    final_oof_path = RESULT_ROOT / "final_oof_predictions.csv"
    final_oof.to_csv(final_oof_path, index=False, float_format="%.17g")
    final_readback = pd.read_csv(final_oof_path, float_precision="round_trip")
    common.require(len(final_readback) == 598 and final_readback["subject_index"].nunique() == 598, "Final OOF coverage mismatch")
    final_readback_probability = final_readback[
        [f"selected_probability_{name}" for name in common.CLASS_ORDER]
    ].to_numpy(dtype=np.float64)
    selected_model_metrics = common.metric_bundle(
        final_readback["truth"].to_numpy(dtype=np.int64), final_readback_probability
    )
    common.require(selected_model_metrics["correct"] == overall[1]["correct"], "Final OOF correct mismatch")
    common.require(selected_model_metrics["confusion_matrix"] == overall[1]["confusion_matrix"], "Final OOF confusion mismatch")
    for metric_name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        common.require(
            math.isclose(selected_model_metrics[metric_name], overall[1][metric_name], rel_tol=0.0, abs_tol=2e-12),
            f"Final OOF {metric_name} mismatch",
        )

    cache_runtime = 0.0
    for outer_fold in FOLDS:
        _, manifest_path = cache_paths(outer_fold)
        cache_runtime += float(json.loads(manifest_path.read_text(encoding="utf-8"))["runtime_seconds"])
    confidence = confidence_pattern(truth, original_probability, tabpfn_probability)
    experiment_runtime = time.perf_counter() - formal_started
    gate_runtime = sum(
        arm_results[name][1]["gate_training_runtime_seconds"] for name, _ in CAP_ARMS
    )
    training_runtime = cache_runtime + gate_runtime
    payload = {
        "schema_version": "tpcf-v1-final-v1",
        "experiment": "TabPFN–Deep Conservative Fusion v1",
        "branch": git_value("branch", "--show-current"),
        "run_source_commit": git_value("rev-parse", "HEAD"),
        "environment": environment_snapshot(context.device),
        "input_provenance": provenance,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE},
        "comparison": rows,
        "best_fixed_fusion": zero_payload["best_candidate"],
        "best_gate": best_gate_name,
        "best_gate_cap": best_cap,
        "best_gate_result": best_gate,
        "gate_results": {
            name: arm_results[name][1]
            for name, _ in CAP_ARMS
        },
        "best_overall_by_fixed_ranking": overall_name,
        "selected_model_metrics_recomputed_from_final_oof": selected_model_metrics,
        "final_oof_predictions_sha256": common.file_sha256(final_oof_path),
        "decision": decision,
        "confidence_pattern": confidence,
        "runtime_seconds": {
            "inner_expert_cache_sum": cache_runtime,
            "gate_training_sum": gate_runtime,
            "formal_training_sum": training_runtime,
            "current_aggregation_invocation": experiment_runtime,
        },
        "integrity": {
            "inner_expert_cache_shared_by_both_gate_caps": True,
            "gate_caps_trained_independently_from_same_seed": True,
            "outer_test_labels_used_only_after_predictions": True,
            "final_oof_metrics_recomputed_from_readback": True,
        },
    }
    common.write_json(RESULT_ROOT / "final_metrics.json", payload)
    diagnostics = best_gate["diagnostics"]
    cap20 = arm_results["tpcf_v1_cap020"][1]
    cap35 = arm_results["tpcf_v2_cap035"][1]
    best_boundary = diagnostics["boundary_repairs_damages"]
    identifiable = (
        confidence["tabpfn_only_correct"]["tabpfn_margin_mean"]
        > confidence["original_only_correct"]["tabpfn_margin_mean"]
    )
    if decision != "NO_NET_IMPROVEMENT":
        failure_reason = "not applicable because the bounded-gate decision was not a failure"
    elif gate_metrics["correct"] >= 557 and not bacc_ok:
        failure_reason = "correct count improved, but the BACC safety condition failed"
    elif diagnostics["gate_gt_0_05_proportion"] < 0.02:
        failure_reason = "gate activation was too weak to change enough predictions"
    elif diagnostics["changed_predictions"] == 0:
        failure_reason = (
            "the gate was highly active, but the bounded log-residual crossed no Original decision boundary; "
            "failure was neither under-activation nor damage-heavy correction"
        )
    elif diagnostics["damages"] >= diagnostics["repairs"]:
        failure_reason = "activated corrections did not yield more repairs than damages"
    else:
        failure_reason = "the net repair count remained below the required accuracy threshold"
    gate_diagnostic_columns = [
        "arm", "cap", "correct", "Weighted-F1", "changed", "repairs", "damages",
        "Original-only", "TPCF-only", "McNemar p", "mean gate", "max gate", "gate>0.05",
        "agree gate", "disagree gate", "AD gate", "CN gate", "SMCI gate",
        "TabPFN-only gate", "Original-only gate", "confusion",
    ]
    gate_diagnostic_rows = []
    for arm_name, arm_cap in CAP_ARMS:
        arm_payload = arm_results[arm_name][1]
        arm_metrics = arm_payload["metrics"]
        arm_diag = arm_payload["diagnostics"]
        gate_diagnostic_rows.append(
            {
                "arm": arm_name,
                "cap": arm_cap,
                "correct": arm_metrics["correct"],
                "Weighted-F1": arm_metrics["weighted_f1"],
                "changed": arm_diag["changed_predictions"],
                "repairs": arm_diag["repairs"],
                "damages": arm_diag["damages"],
                "Original-only": arm_diag["original_only_correct"],
                "TPCF-only": arm_diag["candidate_only_correct"],
                "McNemar p": arm_diag["exact_mcnemar_p_value"],
                "mean gate": arm_diag["mean_gate"],
                "max gate": arm_diag["maximum_gate"],
                "gate>0.05": arm_diag["gate_gt_0_05_proportion"],
                "agree gate": arm_diag["agree_mean_gate"],
                "disagree gate": arm_diag["disagree_mean_gate"],
                "AD gate": arm_diag["mean_gate_by_true_class"]["AD"],
                "CN gate": arm_diag["mean_gate_by_true_class"]["CN"],
                "SMCI gate": arm_diag["mean_gate_by_true_class"]["SMCI"],
                "TabPFN-only gate": arm_diag["tabpfn_only_correct_mean_gate"],
                "Original-only gate": arm_diag["original_only_correct_mean_gate"],
                "confusion": json.dumps(arm_metrics["confusion_matrix"], separators=(",", ":")),
            }
        )
    gate_diagnostic_table = [
        "| " + " | ".join(gate_diagnostic_columns) + " |",
        "| " + " | ".join(["---"] * len(gate_diagnostic_columns)) + " |",
    ]
    for row in gate_diagnostic_rows:
        gate_diagnostic_table.append(
            "| " + " | ".join(str(row[column]) for column in gate_diagnostic_columns) + " |"
        )
    report = [
        "# TabPFN–Deep Conservative Fusion v1",
        "",
        f"Branch: `{payload['branch']}`; run source commit: `{payload['run_source_commit']}`; "
        f"device: {payload['environment']['gpu_name']}; formal training sum: {training_runtime:.1f}s.",
        "",
        "## Results",
        "",
        "\n".join(markdown_table),
        "",
        f"Best bounded gate: **{best_gate_name}** (cap={best_cap:.2f}), "
        f"Correct={gate_metrics['correct']}/598, ACC={gate_metrics['acc']:.10f}, "
        f"Macro-F1={gate_metrics['macro_f1']:.10f}, BACC={gate_metrics['bacc']:.10f}, "
        f"Probability Macro-AUC={gate_metrics['macro_auc']:.10f}. Decision: **{decision}**.",
        "",
        f"Repairs={diagnostics['repairs']}; damages={diagnostics['damages']}; "
        f"changed={diagnostics['changed_predictions']}; exact McNemar p={diagnostics['exact_mcnemar_p_value']:.10g}. "
        f"Mean/max gate={diagnostics['mean_gate']:.6f}/{diagnostics['maximum_gate']:.6f}; "
        f"gate>0.05={diagnostics['gate_gt_0_05_proportion']:.4%}.",
        "",
        "## Gate diagnostics by cap",
        "",
        "\n".join(gate_diagnostic_table),
        "",
        "## Required conclusions",
        "",
        f"1. The best zero-training fusion did **not** produce a net accuracy gain: "
        f"{zero_metrics['correct']}/598, with {zero_payload['best_candidate']['comparison']['repairs']} repairs and "
        f"{zero_payload['best_candidate']['comparison']['damages']} damages; its probability AUC rose to {zero_metrics['macro_auc']:.10f}.",
        f"2. TabPFN-only-correct samples show {'a modest' if identifiable else 'no clear'} confidence separation: "
        f"TabPFN margin mean {confidence['tabpfn_only_correct']['tabpfn_margin_mean']:.4f} versus "
        f"{confidence['original_only_correct']['tabpfn_margin_mean']:.4f} on Original-only-correct samples, while "
        f"Original remains highly confident on many TabPFN-only repairs (Original margin mean "
        f"{confidence['tabpfn_only_correct']['original_margin_mean']:.4f}).",
        f"3. The bounded gate {'was' if gate_metrics['correct'] > zero_metrics['correct'] else 'was not'} more effective than the best fixed fusion by correct count "
        f"({gate_metrics['correct']} versus {zero_metrics['correct']}).",
        f"4. cap={best_cap:.2f} is preferred by Correct > AUC > Macro-F1; cap=.20/.35 correct counts were "
        f"{cap20['metrics']['correct']}/{cap35['metrics']['correct']}.",
        f"5. Boundary net changes: AD/sMCI {best_boundary['AD_SMCI']['repairs']} repairs and "
        f"{best_boundary['AD_SMCI']['damages']} damages; CN/sMCI {best_boundary['CN_SMCI']['repairs']} repairs and "
        f"{best_boundary['CN_SMCI']['damages']} damages.",
        f"6. TPCF {'reached' if gate_metrics['correct'] >= 557 else 'did not reach'} 557 correct subjects.",
        f"7. If unsuccessful, the observed reason is: {failure_reason}.",
        "",
        "No larger cap, class-wise gate, new feature input, new loss, ensemble, or graph branch was tested. "
        "If both bounded gates fail, the next recommended route is DIFFormer global relations plus Sparse GCN local relations.",
    ]
    (RESULT_ROOT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def run_smoke(context: DeepContext, checkpoint: Path) -> None:
    output = common.unique_directory(SMOKE_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    outer_train, outer_test = fold_positions(context.protocol, 0)
    _, relative_validation, inner_train, inner_validation = inner_splits(context.protocol, 0)[0]
    common.require(not set(outer_test.tolist()) & (set(inner_train.tolist()) | set(inner_validation.tolist())), "Smoke leakage")
    original_probability, original_audit = train_original_inner(
        context, inner_train, inner_validation, epochs=3
    )
    tabpfn_probability, tabpfn_audit = train_tabpfn_inner(
        context, checkpoint, inner_train, inner_validation
    )
    raw = common.gate_features(original_probability, tabpfn_probability)
    standardized, _, mean, std = standardize_gate_features(raw, raw)
    set_seed()
    model = GateMLP().to(context.device)
    features = torch.as_tensor(standardized, dtype=torch.float32, device=context.device)
    p_original = torch.as_tensor(original_probability, dtype=torch.float32, device=context.device)
    p_tabpfn = torch.as_tensor(tabpfn_probability, dtype=torch.float32, device=context.device)
    y = torch.as_tensor(context.protocol["y"][inner_validation], dtype=torch.long, device=context.device)
    gate, logits, probability = gate_logits(model, features, p_original, p_tabpfn, 0.20)
    loss = F.cross_entropy(logits, y) + 0.01 * gate.mean()
    common.require(bool(torch.isfinite(loss)), "Smoke gate loss NaN/Inf")
    loss.backward()
    common.require(
        all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()),
        "Smoke gate gradient failed",
    )
    common.require(bool(torch.isfinite(probability).all()), "Smoke probability NaN/Inf")
    common.require(bool(torch.allclose(probability.sum(dim=1), torch.ones(len(probability), device=context.device), atol=1e-6, rtol=0.0)), "Smoke probability sum")
    report = {
        "passed": True,
        "outer_fold": 0,
        "inner_fold": 0,
        "outer_train_size": int(len(outer_train)),
        "inner_train_size": int(len(inner_train)),
        "inner_validation_size": int(len(inner_validation)),
        "outer_test_overlap_count": 0,
        "original": original_audit,
        "tabpfn": tabpfn_audit,
        "gate": {
            "cap": 0.20,
            "loss": safe_float(loss),
            "all_gradients_finite": True,
            "probability_shape": list(probability.shape),
            "probability_rows_sum_to_one": True,
            "cuda": bool(probability.is_cuda),
            "continuous_feature_mean": mean.tolist(),
            "continuous_feature_std": std.tolist(),
        },
    }
    common.write_json(output / "smoke_report.json", report)
    log("TPCF smoke PASS: Original 3 epochs, TabPFN E4, CUDA gate forward/backward")


def validate_checkpoint(checkpoint: Path) -> None:
    common.require(checkpoint.is_file(), f"Checkpoint missing: {checkpoint}")
    common.require(checkpoint.stat().st_size == CHECKPOINT_SIZE, "Checkpoint size mismatch")
    common.require(common.file_sha256(checkpoint) == CHECKPOINT_SHA256, "Checkpoint SHA256 mismatch")


def run_formal(context: DeepContext, checkpoint: Path) -> None:
    common.require((RESULT_ROOT / "zero_training/complementarity.json").is_file(), "Stage A result missing")
    smoke_reports = sorted(BOUNDED_ROOT.glob("smoke*/smoke_report.json")) if BOUNDED_ROOT.exists() else []
    common.require(smoke_reports, "Passing bounded-gate smoke report is required before formal training")
    common.require(
        json.loads(smoke_reports[-1].read_text(encoding="utf-8")).get("passed") is True,
        "Latest bounded-gate smoke did not pass",
    )
    complement = json.loads((RESULT_ROOT / "zero_training/complementarity.json").read_text(encoding="utf-8"))
    common.require(complement.get("run_gate") is True, "RUN_GATE is false")
    aligned, provenance = common.load_aligned_outer_oof()
    formal_started = time.perf_counter()
    BOUNDED_ROOT.mkdir(parents=True, exist_ok=True)
    common.write_json(
        BOUNDED_ROOT / "config.json",
        {
            "experiment": "TabPFN–Deep Conservative Fusion v1",
            "outer_folds": list(FOLDS),
            "inner_folds": 3,
            "inner_splitter": "StratifiedKFold(shuffle=True, random_state=0)",
            "original_epochs": ORIGINAL_EPOCHS,
            "original_selection_rule": ["ACC", "adjusted-logit Macro-AUC", "Macro-F1"],
            "tabpfn": {"arm": "E4 uncalibrated", "n_estimators": 4, "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256},
            "gate": {"architecture": "Linear(6,8)-ReLU-Linear(8,1)-Sigmoid", "caps": [0.20, 0.35], "epochs": 400, "optimizer": "Adam", "learning_rate": 0.01, "weight_decay": 0.0001, "penalty": "0.01*mean(capped_gate)"},
            "gate_class_weight_formula": "(N_outer_train - n_c) / N_outer_train",
            "continuous_standardization": "outer-train inner-OOF mean/std(ddof=0); zero std replaced by one",
            "disagreement_standardized": False,
            "outer_test_labels_used_only_after_predictions": True,
            "input_provenance": provenance,
        },
    )
    for outer_fold in FOLDS:
        cache, _ = load_or_generate_cache(context, checkpoint, outer_fold)
        for arm_name, cap in CAP_ARMS:
            train_or_load_gate_fold(context, aligned, cache, outer_fold, arm_name, cap)
    payload = finalize(context, aligned, provenance, formal_started)
    log(
        f"TPCF formal complete best_gate={payload['best_gate']} "
        f"correct={payload['best_gate_result']['metrics']['correct']}/598 decision={payload['decision']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPCF v1 bounded reliability gate")
    parser.add_argument("--smoke", action="store_true", help="Run the single outer0/inner0 minimal smoke only")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    validate_checkpoint(checkpoint)
    context = load_context()
    log(f"CUDA={torch.cuda.get_device_name(context.device)} branch={git_value('branch', '--show-current')}")
    if args.smoke:
        run_smoke(context, checkpoint)
    else:
        run_formal(context, checkpoint)


if __name__ == "__main__":
    main()
