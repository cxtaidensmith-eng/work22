from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "DATASET/TADPOLE/AD_CN_SMCI_ADNI_processed_standard_data.csv"
MODAL_DICT_PATH = ROOT / "DATASET/TADPOLE/AD_CN_SMCI_ADNI_modal_feat_dict.npy"
FOLD_MANIFEST_PATH = (
    ROOT
    / "experiments/ovr_aligned_shared_query_v1/formal/reference_fold_manifest.json"
)
SEPS_OOF_PATH = (
    ROOT
    / "experiments/query_pool_component_sharing_v1/tenfold_seed0/oof_predictions.csv"
)
DEFAULT_OUTPUT_PARENT = ROOT / "experiments/tabpfn_baseline_v1"

EXPECTED_BRANCH = "experiment/tabpfn-baseline-v1"
BASE_COMMIT = "765720c1c0a3b2263502e9bf662fe33de7e45428"
SEED = 0
FOLDS = tuple(range(10))
CLASS_ORDER = ("AD", "CN", "SMCI")
CLASS_IDS = np.arange(len(CLASS_ORDER), dtype=np.int64)

EXPECTED_DATA_SHA256 = (
    "2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042"
)
EXPECTED_MODAL_SHA256 = (
    "5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106"
)
EXPECTED_MANIFEST_CANONICAL_SHA256 = (
    "be020ede2a03d59dbd5e8bbf715cb4d097f8398f9824c122521c3240e00b6515"
)
EXPECTED_SAMPLE_ORDER_SHA256 = (
    "7ef70c38282cd3c3fa2fa95f41d5d7dfd3ee456d7ac58a25f80e2d7c378df668"
)
EXPECTED_LABEL_ORDER_SHA256 = (
    "40aa3114eeb832ce1b80a8706b47e50f99da7a48030d40560ef56164e109662a"
)
EXPECTED_ASSIGNMENT_SHA256 = (
    "8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9"
)
EXPECTED_SPLIT_SHA256 = (
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

EXPECTED_CHECKPOINT_FILENAME = "tabpfn-v3-classifier-v3_default.ckpt"
EXPECTED_CHECKPOINT_SHA256 = (
    "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
)
EXPECTED_CHECKPOINT_SIZE = 212_804_803
EXPECTED_SEPS_OOF_SHA256 = (
    "1286231fd513d811e4828319ca1210b6c76635c029107f497356e7e577a94e68"
)
EXPECTED_ORIGINAL_OOF_SHA256 = (
    "86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565"
)

HEADLINE_BASELINES = {
    "original_query": {
        "acc": 0.9297658862876255,
        "macro_f1": 0.9140778251271361,
        "bacc": 0.9140778251271361,
        "weighted_f1": 0.9297658862876255,
        "macro_auc_historical_adjusted_score": 0.9500701795923208,
        "macro_auc_probability_recomputed": 0.9560490697782983,
    },
    "seps_q_v1": {
        "acc": 0.9247491638795987,
        "macro_f1": 0.9138656552532017,
        "bacc": 0.9082063928901053,
        "weighted_f1": 0.9246314265285625,
        "macro_auc_historical_adjusted_score": 0.9571291149332602,
        "macro_auc_probability_recomputed": 0.9532899410556896,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        require(np.isfinite(value), f"Non-finite value cannot be serialized: {value}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return safe_json(vars(value))
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            safe_json(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def git_snapshot() -> dict[str, Any]:
    def git(*args: str) -> str:
        return command_output(["git", "-C", str(ROOT), *args])

    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "status_short": git("status", "--short"),
        "base_commit": BASE_COMMIT,
        "remote_branch": "origin/experiment/query-pool-component-sharing-v1",
    }


def split_sha256(
    source_indices: np.ndarray, train_positions: np.ndarray, test_positions: np.ndarray
) -> str:
    train_mask = np.zeros(source_indices.size, dtype=np.uint8)
    test_mask = np.zeros(source_indices.size, dtype=np.uint8)
    train_mask[train_positions] = 1
    test_mask[test_positions] = 1
    digest = hashlib.sha256()
    digest.update(np.asarray(source_indices, dtype=np.int64).tobytes())
    digest.update(train_mask.tobytes())
    digest.update(test_mask.tobytes())
    return digest.hexdigest()


def reconstruct_source_order(fold_manifest: dict[str, Any]) -> np.ndarray:
    sample_count = int(fold_manifest["sample_count"])
    source_order = np.full(sample_count, -1, dtype=np.int64)
    for fold in fold_manifest["folds"]:
        for positions_key, indices_key in (
            ("train_positions", "train_indices"),
            ("test_positions", "test_indices"),
        ):
            positions = np.asarray(fold[positions_key], dtype=np.int64)
            indices = np.asarray(fold[indices_key], dtype=np.int64)
            require(positions.shape == indices.shape, f"fold {fold['fold']} alignment")
            existing = source_order[positions]
            require(
                np.all((existing == -1) | (existing == indices)),
                f"fold {fold['fold']} has inconsistent source-index mapping",
            )
            source_order[positions] = indices
    require(np.all(source_order >= 0), "Fold manifest leaves source positions unmapped")
    require(np.unique(source_order).size == sample_count, "Source indices are not unique")
    require(
        set(source_order.tolist()) == set(range(sample_count)),
        "Source indices are not exactly 0..597",
    )
    return source_order


def load_protocol() -> dict[str, Any]:
    for path in (DATA_PATH, MODAL_DICT_PATH, FOLD_MANIFEST_PATH):
        require(path.is_file(), f"Required protocol file missing: {path}")
    require(file_sha256(DATA_PATH) == EXPECTED_DATA_SHA256, "Dataset SHA256 mismatch")
    require(file_sha256(MODAL_DICT_PATH) == EXPECTED_MODAL_SHA256, "Modal dictionary SHA256 mismatch")
    require(
        file_sha256(FOLD_MANIFEST_PATH) == EXPECTED_MANIFEST_FILE_SHA256,
        "Fold-manifest file SHA256 mismatch",
    )

    fold_manifest = read_json(FOLD_MANIFEST_PATH)
    manifest_without_hash = deepcopy(fold_manifest)
    stored_canonical = manifest_without_hash.pop("canonical_sha256")
    recomputed_canonical = canonical_sha256(manifest_without_hash)
    require(stored_canonical == EXPECTED_MANIFEST_CANONICAL_SHA256, "Stored canonical manifest SHA mismatch")
    require(recomputed_canonical == stored_canonical, "Canonical manifest SHA recomputation failed")
    require(fold_manifest["class_order"] == list(CLASS_ORDER), "Class order mismatch")
    require(int(fold_manifest["sample_count"]) == 598, "Expected 598 samples")
    require(len(fold_manifest["folds"]) == 10, "Expected ten folds")

    frame = pd.read_csv(DATA_PATH, low_memory=False)
    require(frame.shape == (598, 361), f"Unexpected CSV shape: {frame.shape}")
    require(not frame.columns.duplicated().any(), "Duplicate CSV columns")
    feature_names = frame.columns[:-1].astype(str).tolist()
    require(len(feature_names) == 360, "Expected 360 feature columns")
    X_source = frame.iloc[:, :-1].to_numpy(dtype=np.float64, copy=True)
    raw_labels = frame.iloc[:, -1].to_numpy(dtype=np.float64, copy=True)
    require(np.isfinite(X_source).all(), "Dataset features contain NaN/Inf")
    require(np.isfinite(raw_labels).all(), "Dataset labels contain NaN/Inf")
    require(np.array_equal(raw_labels, np.rint(raw_labels)), "Labels are non-integral")
    y_source = raw_labels.astype(np.int64) - 1
    require(set(y_source.tolist()) == {0, 1, 2}, "Unexpected numeric labels")

    modal_dict = np.load(MODAL_DICT_PATH, allow_pickle=True).item()
    require(isinstance(modal_dict, dict), "Modal dictionary is not a dict")
    flattened_modal_features = [
        str(column) for columns in modal_dict.values() for column in columns
    ]
    require(len(flattened_modal_features) == 360, "Modal dictionary does not contain 360 entries")
    require(len(set(flattened_modal_features)) == 360, "Modal dictionary contains duplicate features")
    require(flattened_modal_features == feature_names, "Modal dictionary order differs from CSV")

    source_order = reconstruct_source_order(fold_manifest)
    X = X_source[source_order]
    y = y_source[source_order]
    require(raw_array_sha256(source_order) == EXPECTED_SAMPLE_ORDER_SHA256, "Sample order SHA mismatch")
    require(raw_array_sha256(y) == EXPECTED_LABEL_ORDER_SHA256, "Label order SHA mismatch")
    require(fold_manifest["sample_order_sha256"] == EXPECTED_SAMPLE_ORDER_SHA256, "Manifest sample SHA mismatch")
    require(fold_manifest["label_order_sha256"] == EXPECTED_LABEL_ORDER_SHA256, "Manifest label SHA mismatch")
    assignment = np.asarray(
        fold_manifest["test_fold_assignment_by_shuffled_position"], dtype=np.uint8
    )
    require(raw_array_sha256(assignment) == EXPECTED_ASSIGNMENT_SHA256, "Fold assignment SHA mismatch")

    test_occurrence = np.zeros(598, dtype=np.int64)
    train_occurrence = np.zeros(598, dtype=np.int64)
    fold_audits: list[dict[str, Any]] = []
    test_sets: list[set[int]] = []
    for expected_fold, fold in enumerate(fold_manifest["folds"]):
        require(int(fold["fold"]) == expected_fold, "Fold order mismatch")
        train_positions = np.asarray(fold["train_positions"], dtype=np.int64)
        test_positions = np.asarray(fold["test_positions"], dtype=np.int64)
        train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
        test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
        require(np.array_equal(source_order[train_positions], train_indices), f"fold {expected_fold} train indices mismatch")
        require(np.array_equal(source_order[test_positions], test_indices), f"fold {expected_fold} test indices mismatch")
        require(not set(train_positions.tolist()) & set(test_positions.tolist()), f"fold {expected_fold} overlap")
        require(
            set(train_positions.tolist()) | set(test_positions.tolist()) == set(range(598)),
            f"fold {expected_fold} is not complementary",
        )
        actual_split_sha = split_sha256(source_order, train_positions, test_positions)
        require(actual_split_sha == EXPECTED_SPLIT_SHA256[expected_fold], f"fold {expected_fold} split SHA mismatch")
        require(actual_split_sha == fold["split_hash"], f"fold {expected_fold} manifest split SHA mismatch")
        train_occurrence[train_positions] += 1
        test_occurrence[test_positions] += 1
        test_sets.append(set(test_positions.tolist()))
        train_counts = np.bincount(y[train_positions], minlength=3).astype(int)
        test_counts = np.bincount(y[test_positions], minlength=3).astype(int)
        require(np.all(train_counts > 0) and np.all(test_counts > 0), f"fold {expected_fold} missing class")
        train_constant = np.flatnonzero(np.ptp(X[train_positions], axis=0) == 0)
        fold_audits.append(
            {
                "fold": expected_fold,
                "train_size": int(train_positions.size),
                "test_size": int(test_positions.size),
                "train_class_count": dict(zip(CLASS_ORDER, train_counts.tolist())),
                "test_class_count": dict(zip(CLASS_ORDER, test_counts.tolist())),
                "train_constant_feature_count": int(train_constant.size),
                "train_constant_feature_indices": train_constant.tolist(),
                "split_sha256": actual_split_sha,
            }
        )

    require(np.all(test_occurrence == 1), "Each shuffled row must be test exactly once")
    require(np.all(train_occurrence == 9), "Each shuffled row must be train exactly nine times")
    require(
        all(not (test_sets[left] & test_sets[right]) for left in FOLDS for right in FOLDS if left < right),
        "Test folds are not pairwise disjoint",
    )
    class_counts = np.bincount(y, minlength=3).astype(int)
    require(class_counts.tolist() == [72, 209, 317], "Class count mismatch")
    global_constant = np.flatnonzero(np.ptp(X, axis=0) == 0)
    require(global_constant.tolist() == [299, 318, 326], "Global constant-feature audit changed")
    duplicate_feature_rows = int(pd.DataFrame(X_source).duplicated(keep=False).sum())
    require(duplicate_feature_rows == 0, "Duplicate feature rows detected")

    return {
        "X": X,
        "y": y,
        "source_order": source_order,
        "feature_names": feature_names,
        "fold_manifest": fold_manifest,
        "audit": {
            "dataset_path": DATA_PATH.relative_to(ROOT).as_posix(),
            "dataset_sha256": EXPECTED_DATA_SHA256,
            "modal_dictionary_path": MODAL_DICT_PATH.relative_to(ROOT).as_posix(),
            "modal_dictionary_sha256": EXPECTED_MODAL_SHA256,
            "fold_manifest_path": FOLD_MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "fold_manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
            "fold_manifest_canonical_sha256": EXPECTED_MANIFEST_CANONICAL_SHA256,
            "sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
            "label_order_sha256": EXPECTED_LABEL_ORDER_SHA256,
            "test_fold_assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
            "shape": [598, 360],
            "label_shape": [598],
            "input_dtype": str(X.dtype),
            "class_mapping": {name: index for index, name in enumerate(CLASS_ORDER)},
            "class_counts": dict(zip(CLASS_ORDER, class_counts.tolist())),
            "nan_count": int(np.isnan(X).sum()),
            "positive_inf_count": int(np.isposinf(X).sum()),
            "negative_inf_count": int(np.isneginf(X).sum()),
            "global_constant_feature_count": int(global_constant.size),
            "global_constant_feature_indices": global_constant.tolist(),
            "global_constant_feature_names": [feature_names[index] for index in global_constant],
            "duplicate_feature_row_count": duplicate_feature_rows,
            "identity_kind": fold_manifest["identity_kind"],
            "identity_limitation": "The committed protocol proves 598 unique source CSV rows; no RID column is present, so person-level uniqueness is not independently verifiable.",
            "modalities": {str(key): len(value) for key, value in modal_dict.items()},
            "folds": fold_audits,
            "test_sets_pairwise_disjoint": True,
            "test_union_count": 598,
            "each_source_row_test_once": True,
            "each_source_row_train_nine_times": True,
        },
    }


def checkpoint_audit(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    require(checkpoint.is_file(), f"Checkpoint missing: {checkpoint}")
    require(checkpoint.name == EXPECTED_CHECKPOINT_FILENAME, "Unexpected checkpoint filename")
    actual_size = checkpoint.stat().st_size
    actual_sha = file_sha256(checkpoint)
    require(actual_size == EXPECTED_CHECKPOINT_SIZE, "Checkpoint size mismatch")
    require(actual_sha == EXPECTED_CHECKPOINT_SHA256, "Checkpoint SHA256 mismatch")
    try:
        checkpoint.relative_to(ROOT)
    except ValueError:
        outside_repository = True
    else:
        outside_repository = False
    require(outside_repository, "Checkpoint must remain outside the repository")
    return {
        "path": str(checkpoint),
        "filename": checkpoint.name,
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "outside_repository": outside_repository,
        "model_version": "TabPFN-3 classifier v3_default",
        "official_repository": "Prior-Labs/tabpfn_3",
        "official_revision": "24a16a89d245878b846555110985634aa2e656d7",
        "license": "Prior Labs TabPFN-3 license accepted by the user; verified before execution",
        "credential_material_recorded": False,
    }


def environment_audit() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is not available")
    properties = torch.cuda.get_device_properties(0)
    packages = [
        "tabpfn",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    require(versions["tabpfn"] == "8.2.0", "tabpfn version is not 8.2.0")
    return {
        "captured_at_utc": utc_now(),
        "conda_environment_path": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": versions,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "determinism": {
            "seed": SEED,
            "torch_cudnn_deterministic": True,
            "torch_cudnn_benchmark": False,
            "torch_deterministic_algorithms_forced": False,
            "note": "TabPFN documents that fixed seeds do not guarantee bitwise reproducibility across hardware.",
        },
    }


def metric_bundle(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    require(y_true.ndim == 1, "Truth must be one-dimensional")
    require(probability.shape == (y_true.size, 3), "Probability shape mismatch")
    require(np.isfinite(probability).all(), "Probability contains NaN/Inf")
    require(np.all(probability >= 0.0), "Probability contains negative values")
    row_sums = probability.sum(axis=1)
    require(np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-7), "Probability rows do not sum to one")
    prediction = CLASS_IDS[np.argmax(probability, axis=1)]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=CLASS_IDS,
        zero_division=0,
    )
    auc = roc_auc_score(
        y_true,
        probability,
        labels=CLASS_IDS,
        multi_class="ovr",
        average="macro",
    )
    return {
        "acc": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, labels=CLASS_IDS, average="macro", zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_true, prediction)),
        "macro_auc": float(auc),
        "macro_auc_definition": "sklearn roc_auc_score; multiclass OVR; macro average; predicted probabilities; class order AD,CN,SMCI",
        "weighted_f1": float(f1_score(y_true, prediction, labels=CLASS_IDS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=CLASS_IDS).astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_ORDER)
        },
        "probability_validation": {
            "shape": list(probability.shape),
            "finite": True,
            "minimum": float(probability.min()),
            "maximum": float(probability.max()),
            "row_sum_minimum": float(row_sums.min()),
            "row_sum_maximum": float(row_sums.max()),
            "maximum_absolute_row_sum_error": float(np.abs(row_sums - 1.0).max()),
        },
    }


def set_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_audit(classifier: Any, checkpoint: Path, instance_uuid: str) -> dict[str, Any]:
    params = classifier.get_params(deep=False)
    strict_params = {
        "model_path": str(checkpoint),
        "n_estimators": 1,
        "auto_scale_n_estimators": False,
        "random_state": 0,
        "device": "cuda:0",
        "fit_mode": "low_memory",
        "memory_saving_mode": True,
        "inference_precision": "auto",
    }
    for name, expected in strict_params.items():
        actual = params[name]
        if name == "model_path":
            require(Path(actual).resolve() == checkpoint.resolve(), "Effective model_path changed")
        else:
            require(actual == expected, f"Effective constructor parameter changed: {name}={actual!r}")
    require(classifier.n_estimators == 1, "Requested estimator count changed")
    require(classifier.auto_scale_n_estimators is False, "Auto estimator scaling enabled")
    require(classifier.random_state == 0, "Random state changed")
    require(classifier.device == "cuda:0", "Device changed")
    require(classifier.n_estimators_ == 1, "Effective estimator count is not one")
    require(len(classifier.ensemble_configs_) == 1, "Multiple ensemble configs detected")
    require(len(classifier.models_) == 1, "Multiple checkpoint models detected")
    require(tuple(str(device) for device in classifier.devices_) == ("cuda:0",), "Effective device is not cuda:0")
    require(np.array_equal(np.asarray(classifier.classes_, dtype=np.int64), CLASS_IDS), "Effective class order changed")
    require(int(classifier.n_features_in_) == 360, "Model did not receive 360 input features")

    ensemble_config = classifier.ensemble_configs_[0]
    preprocess_config = ensemble_config.preprocess_config
    raw_indices = classifier.ensemble_preprocessor_.subsample_feature_indices[0]
    if raw_indices is None:
        selected_indices = np.arange(classifier.n_features_in_, dtype=np.int64)
    else:
        selected_indices = np.asarray(raw_indices, dtype=np.int64)
    require(selected_indices.ndim == 1, "Internal feature index is not one-dimensional")
    require(np.all((selected_indices >= 0) & (selected_indices < 360)), "Internal feature index out of range")
    require(np.unique(selected_indices).size == selected_indices.size, "Internal feature index contains duplicates")
    feature_names = np.asarray(classifier.feature_names_in_, dtype=object)
    selected_names = feature_names[selected_indices].astype(str).tolist()
    transforms = [
        {
            "name": config.name,
            "categorical_name": config.categorical_name,
            "append_original": config.append_original,
            "max_features_per_estimator": int(config.max_features_per_estimator),
            "global_transformer_name": config.global_transformer_name,
        }
        for config in classifier.inference_config_.PREPROCESS_TRANSFORMS
    ]
    require(all(item["max_features_per_estimator"] == 200 for item in transforms), "Checkpoint preprocessing limit changed")
    require(int(preprocess_config.max_features_per_estimator) == 200, "Selected preprocessor feature limit changed")

    return {
        "instance_uuid": instance_uuid,
        "classifier_class": type(classifier).__name__,
        "model_architecture_class": type(classifier.model_).__name__,
        "constructor_parameters": safe_json(params),
        "strict_parameters": strict_params,
        "requested_n_estimators": 1,
        "effective_n_estimators": int(classifier.n_estimators_),
        "auto_scale_n_estimators": bool(classifier.auto_scale_n_estimators),
        "ensemble_config_count": len(classifier.ensemble_configs_),
        "loaded_model_count": len(classifier.models_),
        "effective_devices": [str(device) for device in classifier.devices_],
        "effective_classes": np.asarray(classifier.classes_, dtype=np.int64).tolist(),
        "n_features_received": int(classifier.n_features_in_),
        "n_train_samples": int(classifier.n_train_samples_),
        "use_autocast": bool(classifier.use_autocast_),
        "forced_inference_dtype": safe_json(classifier.forced_inference_dtype_),
        "all_checkpoint_preprocess_transforms": transforms,
        "selected_preprocessor": {
            "name": preprocess_config.name,
            "repr": str(preprocess_config),
            "categorical_name": preprocess_config.categorical_name,
            "max_features_per_estimator": int(preprocess_config.max_features_per_estimator),
            "feature_shift_count": int(ensemble_config.feature_shift_count),
            "feature_shift_decoder": ensemble_config.feature_shift_decoder,
        },
        "internal_original_feature_subsampling": {
            "external_feature_selection_performed": False,
            "full_input_feature_count": 360,
            "selected_original_feature_count": int(selected_indices.size),
            "selected_original_feature_indices": selected_indices.tolist(),
            "selected_original_feature_names": selected_names,
            "selected_original_feature_indices_sha256": raw_array_sha256(selected_indices),
            "disclosure": "The complete 360-column matrix was passed to TabPFN. The official checkpoint internally subsamples original columns because one estimator is capped at 200; auto-scaling was disabled by protocol.",
        },
        "single_estimator_gates": {
            "requested_n_estimators_is_one": classifier.n_estimators == 1,
            "auto_scale_disabled": classifier.auto_scale_n_estimators is False,
            "effective_n_estimators_is_one": classifier.n_estimators_ == 1,
            "one_ensemble_config": len(classifier.ensemble_configs_) == 1,
            "one_loaded_model": len(classifier.models_) == 1,
            "cuda_0_only": tuple(str(device) for device in classifier.devices_) == ("cuda:0",),
            "class_order_fixed": np.array_equal(np.asarray(classifier.classes_, dtype=np.int64), CLASS_IDS),
        },
    }


def make_classifier(checkpoint: Path) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(checkpoint),
        n_estimators=1,
        auto_scale_n_estimators=False,
        random_state=0,
        device="cuda:0",
        fit_mode="low_memory",
        memory_saving_mode=True,
        inference_precision="auto",
        show_progress_bar=False,
    )


def write_fold_predictions(
    fold_dir: Path,
    fold: int,
    positions: np.ndarray,
    source_indices: np.ndarray,
    truth: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    prediction = CLASS_IDS[np.argmax(probability, axis=1)]
    frame = pd.DataFrame(
        {
            "fold": fold,
            "shuffled_position": positions,
            "subject_index": source_indices,
            "truth": truth,
            "prediction": prediction,
            "probability_AD": probability[:, 0],
            "probability_CN": probability[:, 1],
            "probability_SMCI": probability[:, 2],
        }
    )
    frame.to_csv(fold_dir / "predictions.csv", index=False, float_format="%.17g")
    return frame


def validate_fold_readback(
    fold_dir: Path,
    expected_frame: pd.DataFrame,
    expected_metrics: dict[str, Any],
) -> dict[str, Any]:
    actual = pd.read_csv(
        fold_dir / "predictions.csv", float_precision="round_trip"
    )
    integer_columns = ["fold", "shuffled_position", "subject_index", "truth", "prediction"]
    probability_columns = [f"probability_{name}" for name in CLASS_ORDER]
    require(actual.columns.tolist() == expected_frame.columns.tolist(), "Fold CSV columns changed on readback")
    require(np.array_equal(actual[integer_columns].to_numpy(), expected_frame[integer_columns].to_numpy()), "Fold integer values changed on readback")
    require(np.array_equal(actual[probability_columns].to_numpy(), expected_frame[probability_columns].to_numpy()), "Fold probabilities changed on readback")
    recomputed = metric_bundle(
        actual["truth"].to_numpy(dtype=np.int64),
        actual[probability_columns].to_numpy(dtype=np.float64),
    )
    for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(recomputed[name] == expected_metrics[name], f"Fold metric changed on readback: {name}")
    require(recomputed["confusion_matrix"] == expected_metrics["confusion_matrix"], "Fold confusion matrix changed on readback")
    return {
        "passed": True,
        "row_count": int(actual.shape[0]),
        "integer_values_bit_exact": True,
        "probabilities_bit_exact_after_csv_roundtrip": True,
        "metrics_exactly_recomputed": True,
        "predictions_sha256": file_sha256(fold_dir / "predictions.csv"),
    }


def run_fold(
    protocol: dict[str, Any],
    checkpoint: Path,
    fold: int,
    fold_dir: Path,
    log,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    require(fold in FOLDS, f"Invalid fold: {fold}")
    fold_entry = protocol["fold_manifest"]["folds"][fold]
    train_positions = np.asarray(fold_entry["train_positions"], dtype=np.int64)
    test_positions = np.asarray(fold_entry["test_positions"], dtype=np.int64)
    X = protocol["X"]
    y = protocol["y"]
    source_order = protocol["source_order"]
    feature_names = protocol["feature_names"]
    X_train = pd.DataFrame(X[train_positions], columns=feature_names)
    X_test = pd.DataFrame(X[test_positions], columns=feature_names)
    y_train = y[train_positions]
    y_test = y[test_positions]
    require(X_train.shape == (int(fold_entry["train_size"]), 360), "Train shape mismatch")
    require(X_test.shape == (int(fold_entry["test_size"]), 360), "Test shape mismatch")
    require(np.isfinite(X_train.to_numpy()).all() and np.isfinite(X_test.to_numpy()).all(), "Fold input contains NaN/Inf")

    fold_dir.mkdir(parents=True, exist_ok=False)
    set_reproducibility()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    instance_uuid = str(uuid.uuid4())
    classifier = make_classifier(checkpoint)
    log(f"fold={fold} fresh_classifier_uuid={instance_uuid} train={len(train_positions)} test={len(test_positions)}")
    started = time.perf_counter()
    fit_started = time.perf_counter()
    classifier.fit(X_train, y_train)
    torch.cuda.synchronize(0)
    fit_seconds = time.perf_counter() - fit_started
    audit = model_audit(classifier, checkpoint, instance_uuid)
    predict_started = time.perf_counter()
    probability = np.asarray(classifier.predict_proba(X_test), dtype=np.float64)
    torch.cuda.synchronize(0)
    predict_seconds = time.perf_counter() - predict_started
    total_seconds = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(0))
    require(probability.shape == (test_positions.size, 3), "predict_proba output shape mismatch")
    metrics = metric_bundle(y_test, probability)
    metrics.update(
        {
            "fold": fold,
            "train_size": int(train_positions.size),
            "test_size": int(test_positions.size),
            "split_sha256": fold_entry["split_hash"],
            "runtime_seconds": {
                "fit": fit_seconds,
                "predict_proba": predict_seconds,
                "total": total_seconds,
            },
            "cuda_peak_memory_allocated_bytes": peak_memory,
        }
    )
    predictions = write_fold_predictions(
        fold_dir,
        fold,
        test_positions,
        source_order[test_positions],
        y_test,
        probability,
    )
    write_json(fold_dir / "metrics.json", metrics)
    write_json(fold_dir / "model_audit.json", audit)
    validation = validate_fold_readback(fold_dir, predictions, metrics)
    write_json(fold_dir / "readback_validation.json", validation)
    log(
        f"fold={fold} complete acc={metrics['acc']:.9f} macro_auc={metrics['macro_auc']:.9f} "
        f"n_estimators_={audit['effective_n_estimators']} selected_features="
        f"{audit['internal_original_feature_subsampling']['selected_original_feature_count']} "
        f"seconds={total_seconds:.3f}"
    )
    del classifier, X_train, X_test, probability
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, metrics, instance_uuid


def fold_metric_table(fold_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": item["fold"],
                "train_size": item["train_size"],
                "test_size": item["test_size"],
                "split_sha256": item["split_sha256"],
                "acc": item["acc"],
                "macro_f1": item["macro_f1"],
                "bacc": item["bacc"],
                "macro_auc": item["macro_auc"],
                "weighted_f1": item["weighted_f1"],
                "fit_seconds": item["runtime_seconds"]["fit"],
                "predict_proba_seconds": item["runtime_seconds"]["predict_proba"],
                "total_seconds": item["runtime_seconds"]["total"],
                "cuda_peak_memory_allocated_bytes": item["cuda_peak_memory_allocated_bytes"],
            }
            for item in fold_metrics
        ]
    )


def fold_metric_summary(fold_table: pd.DataFrame, oof_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        values = fold_table[name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": name,
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "pooled_oof": float(oof_metrics[name]),
            }
        )
    return pd.DataFrame(rows)


def recompute_historical_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["truth"].to_numpy(dtype=np.int64)
    prediction = frame["prediction"].to_numpy(dtype=np.int64)
    probability_columns = [f"probability_{name}" for name in CLASS_ORDER]
    probability = frame[probability_columns].to_numpy(dtype=np.float64)
    base = metric_bundle(y_true, probability)
    base["prediction_matches_probability_argmax"] = bool(
        np.array_equal(prediction, CLASS_IDS[np.argmax(probability, axis=1)])
    )
    if all(f"adjusted_score_{name}" in frame for name in CLASS_ORDER):
        one_hot = np.eye(3, dtype=np.float64)[y_true]
        adjusted = frame[[f"adjusted_score_{name}" for name in CLASS_ORDER]].to_numpy(dtype=np.float64)
        base["historical_adjusted_score_macro_auc"] = float(
            roc_auc_score(one_hot, adjusted, average="macro")
        )
    return base


def paired_comparison(
    tabpfn_oof: pd.DataFrame,
    other_path: Path,
    other_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    require(other_path.is_file(), f"Historical OOF missing: {other_path}")
    expected_sha = {
        "original_query": EXPECTED_ORIGINAL_OOF_SHA256,
        "seps_q_v1": EXPECTED_SEPS_OOF_SHA256,
    }[other_name]
    actual_sha = file_sha256(other_path)
    require(actual_sha == expected_sha, f"{other_name} OOF SHA256 mismatch")
    other = (
        pd.read_csv(other_path, float_precision="round_trip")
        .sort_values("subject_index")
        .reset_index(drop=True)
    )
    tab = tabpfn_oof.sort_values("subject_index").reset_index(drop=True)
    required = {"fold", "subject_index", "truth", "prediction"}
    require(required.issubset(other.columns), f"{other_name} OOF schema incomplete")
    require(other.shape[0] == 598, f"{other_name} OOF row count is not 598")
    require(other["subject_index"].nunique() == 598, f"{other_name} OOF subject indices are not unique")
    require(np.array_equal(tab["subject_index"], other["subject_index"]), f"{other_name} subject indices do not align")
    require(np.array_equal(tab["truth"], other["truth"]), f"{other_name} truths do not align")
    require(np.array_equal(tab["fold"], other["fold"]), f"{other_name} fold assignments do not align")
    truth = tab["truth"].to_numpy(dtype=np.int64)
    tab_prediction = tab["prediction"].to_numpy(dtype=np.int64)
    other_prediction = other["prediction"].to_numpy(dtype=np.int64)
    tab_correct = tab_prediction == truth
    other_correct = other_prediction == truth
    tab_only = tab_correct & ~other_correct
    other_only = ~tab_correct & other_correct
    discordant = int(tab_only.sum() + other_only.sum())
    p_value = float(binomtest(int(tab_only.sum()), discordant, p=0.5).pvalue) if discordant else 1.0
    rows = pd.DataFrame(
        {
            "fold": tab["fold"].to_numpy(dtype=np.int64),
            "subject_index": tab["subject_index"].to_numpy(dtype=np.int64),
            "truth": truth,
            "tabpfn_prediction": tab_prediction,
            f"{other_name}_prediction": other_prediction,
            "tabpfn_correct": tab_correct.astype(np.int64),
            f"{other_name}_correct": other_correct.astype(np.int64),
            "tabpfn_corrects_other_error": tab_only.astype(np.int64),
            "tabpfn_breaks_other_correct": other_only.astype(np.int64),
        }
    )
    rows.to_csv(output_dir / f"paired_rows_{other_name}.csv", index=False)
    by_class = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {
            "support": int(mask.sum()),
            "tabpfn_correct": int((tab_correct & mask).sum()),
            f"{other_name}_correct": int((other_correct & mask).sum()),
            "tabpfn_only_correct": int((tab_only & mask).sum()),
            f"{other_name}_only_correct": int((other_only & mask).sum()),
            "both_wrong": int((~tab_correct & ~other_correct & mask).sum()),
        }
    other_metrics = recompute_historical_metrics(other)
    result = {
        "other_name": other_name,
        "other_oof_path": str(other_path),
        "other_oof_sha256": actual_sha,
        "other_oof_expected_sha256": expected_sha,
        "alignment": {
            "row_count": 598,
            "unique_subject_indices": 598,
            "subject_index_sets_equal": True,
            "truths_equal": True,
            "fold_assignments_equal": True,
            "alignment_key": "subject_index",
        },
        "counts": {
            "both_correct": int((tab_correct & other_correct).sum()),
            "tabpfn_only_correct": int(tab_only.sum()),
            f"{other_name}_only_correct": int(other_only.sum()),
            "both_wrong": int((~tab_correct & ~other_correct).sum()),
            "discordant": discordant,
        },
        "exact_mcnemar_two_sided_p": p_value,
        "by_true_class": by_class,
        "other_metrics_recomputed": other_metrics,
        "corrected_subject_indices": rows.loc[tab_only, "subject_index"].astype(int).tolist(),
        "broken_subject_indices": rows.loc[other_only, "subject_index"].astype(int).tolist(),
    }
    write_json(output_dir / f"paired_summary_{other_name}.json", result)
    return result


def aggregate_formal(
    staging_dir: Path,
    predictions: list[pd.DataFrame],
    fold_metrics: list[dict[str, Any]],
    instance_uuids: list[str],
    original_oof: Path | None,
) -> dict[str, Any]:
    require(len(predictions) == len(fold_metrics) == len(instance_uuids) == 10, "Formal aggregation requires ten folds")
    require(len(set(instance_uuids)) == 10, "Classifier instance UUIDs are not unique")
    oof = pd.concat(predictions, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    require(oof.shape[0] == 598, "OOF row count is not 598")
    require(oof["subject_index"].nunique() == 598, "OOF subject indices are not unique")
    require(set(oof["subject_index"].astype(int)) == set(range(598)), "OOF subject indices are not 0..597")
    require(oof["shuffled_position"].nunique() == 598, "OOF shuffled positions are not unique")
    require(set(oof["fold"].astype(int)) == set(FOLDS), "OOF does not contain all folds")
    probability_columns = [f"probability_{name}" for name in CLASS_ORDER]
    require(
        np.array_equal(
            oof["prediction"].to_numpy(dtype=np.int64),
            CLASS_IDS[np.argmax(oof[probability_columns].to_numpy(dtype=np.float64), axis=1)],
        ),
        "OOF prediction is not the fixed-class-order probability argmax",
    )
    oof_metrics = metric_bundle(
        oof["truth"].to_numpy(dtype=np.int64),
        oof[probability_columns].to_numpy(dtype=np.float64),
    )
    oof.to_csv(staging_dir / "oof_predictions.csv", index=False, float_format="%.17g")
    write_json(staging_dir / "oof_metrics.json", oof_metrics)
    pd.DataFrame(oof_metrics["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        staging_dir / "oof_confusion_matrix.csv", index_label="truth\\prediction"
    )
    fold_table = fold_metric_table(fold_metrics)
    fold_table.to_csv(staging_dir / "fold_metrics.csv", index=False, float_format="%.17g")
    summary_table = fold_metric_summary(fold_table, oof_metrics)
    summary_table.to_csv(staging_dir / "metrics_summary.csv", index=False, float_format="%.17g")

    comparisons: dict[str, Any] = {}
    comparisons["seps_q_v1"] = paired_comparison(oof, SEPS_OOF_PATH, "seps_q_v1", staging_dir)
    if original_oof is not None:
        comparisons["original_query"] = paired_comparison(
            oof, original_oof.resolve(), "original_query", staging_dir
        )

    result = {
        "oof_metrics": oof_metrics,
        "fold_metric_summary": summary_table.to_dict(orient="records"),
        "runtime_seconds": {
            "fit_sum": float(fold_table["fit_seconds"].sum()),
            "predict_proba_sum": float(fold_table["predict_proba_seconds"].sum()),
            "fold_total_sum": float(fold_table["total_seconds"].sum()),
        },
        "oof_integrity": {
            "row_count": 598,
            "unique_subject_indices": 598,
            "subject_index_set_is_0_to_597": True,
            "unique_shuffled_positions": 598,
            "all_ten_folds_present": True,
            "each_source_row_predicted_once": True,
            "identity_limitation": "subject_index is a source CSV row index, not a verified RID",
        },
        "fresh_classifier_instances": {
            "count": 10,
            "unique_count": len(set(instance_uuids)),
            "instance_uuids": instance_uuids,
            "all_unique": len(set(instance_uuids)) == 10,
        },
        "comparisons": comparisons,
    }
    write_json(staging_dir / "aggregate_summary.json", result)
    return result


def validate_formal_readback(formal_dir: Path) -> dict[str, Any]:
    probability_columns = [f"probability_{name}" for name in CLASS_ORDER]
    oof = pd.read_csv(
        formal_dir / "oof_predictions.csv", float_precision="round_trip"
    )
    require(set(oof["truth"].astype(int)) <= set(CLASS_IDS.tolist()), "OOF truth is out of range")
    require(set(oof["prediction"].astype(int)) <= set(CLASS_IDS.tolist()), "OOF prediction is out of range")
    require(
        np.array_equal(
            oof["prediction"].to_numpy(dtype=np.int64),
            CLASS_IDS[np.argmax(oof[probability_columns].to_numpy(dtype=np.float64), axis=1)],
        ),
        "OOF prediction is not probability argmax",
    )
    stored_metrics = read_json(formal_dir / "oof_metrics.json")
    recomputed = metric_bundle(
        oof["truth"].to_numpy(dtype=np.int64),
        oof[probability_columns].to_numpy(dtype=np.float64),
    )
    metric_exact = {
        name: recomputed[name] == stored_metrics[name]
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    require(all(metric_exact.values()), f"OOF metric readback mismatch: {metric_exact}")
    require(recomputed["confusion_matrix"] == stored_metrics["confusion_matrix"], "OOF confusion matrix readback mismatch")
    fold_checks = {}
    concat = []
    model_uuids = []
    for fold in FOLDS:
        fold_dir = formal_dir / f"fold_{fold:02d}"
        frame = pd.read_csv(
            fold_dir / "predictions.csv", float_precision="round_trip"
        )
        require(set(frame["truth"].astype(int)) <= set(CLASS_IDS.tolist()), f"fold {fold} truth out of range")
        require(set(frame["prediction"].astype(int)) <= set(CLASS_IDS.tolist()), f"fold {fold} prediction out of range")
        require(
            np.array_equal(
                frame["prediction"].to_numpy(dtype=np.int64),
                CLASS_IDS[np.argmax(frame[probability_columns].to_numpy(dtype=np.float64), axis=1)],
            ),
            f"fold {fold} prediction is not probability argmax",
        )
        concat.append(frame)
        fold_metrics = read_json(fold_dir / "metrics.json")
        fold_recomputed = metric_bundle(
            frame["truth"].to_numpy(dtype=np.int64),
            frame[probability_columns].to_numpy(dtype=np.float64),
        )
        exact = all(
            fold_recomputed[name] == fold_metrics[name]
            for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        ) and fold_recomputed["confusion_matrix"] == fold_metrics["confusion_matrix"]
        require(exact, f"fold {fold} metric readback mismatch")
        audit = read_json(fold_dir / "model_audit.json")
        require(all(audit["single_estimator_gates"].values()), f"fold {fold} single-estimator gate failed")
        model_uuids.append(audit["instance_uuid"])
        fold_checks[f"fold_{fold:02d}"] = {
            "passed": True,
            "rows": int(frame.shape[0]),
            "metrics_exact": True,
            "single_estimator_gates_passed": True,
            "prediction_sha256": file_sha256(fold_dir / "predictions.csv"),
        }
    concatenated = pd.concat(concat, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    require(np.array_equal(concatenated.to_numpy(), oof.to_numpy()), "Pooled OOF differs from fold concatenation")
    require(len(set(model_uuids)) == 10, "Formal folds did not use ten fresh classifiers")
    result = {
        "validated_at_utc": utc_now(),
        "passed": True,
        "oof_rows": int(oof.shape[0]),
        "oof_unique_subject_indices": int(oof["subject_index"].nunique()),
        "oof_unique_shuffled_positions": int(oof["shuffled_position"].nunique()),
        "oof_metric_exactness": metric_exact,
        "oof_confusion_matrix_exact": True,
        "oof_equals_fold_concatenation": True,
        "fresh_classifier_uuid_count": len(set(model_uuids)),
        "all_single_estimator_gates_passed": True,
        "folds": fold_checks,
        "artifact_sha256": {
            path.relative_to(formal_dir).as_posix(): file_sha256(path)
            for path in sorted(formal_dir.rglob("*"))
            if path.is_file() and path.name != "readback_validation.json"
        },
    }
    write_json(formal_dir / "readback_validation.json", result)
    return result


def preflight_payload(protocol: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    git = git_snapshot()
    require(git["branch"] == EXPECTED_BRANCH, f"Wrong branch: {git['branch']}")
    require(git["head"] == BASE_COMMIT, f"Unexpected base HEAD: {git['head']}")
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment": "TabPFN Single-Estimator Baseline v1",
        "protocol": {
            "dataset": "TADPOLE",
            "task": "AD_CN_SMCI",
            "subjects_or_source_rows": 598,
            "features": 360,
            "classes": 3,
            "folds": list(FOLDS),
            "seed": SEED,
            "external_standardization": False,
            "external_imputation": False,
            "external_pca": False,
            "external_feature_selection": False,
            "external_one_hot_encoding": False,
            "hyperparameter_search": False,
            "extensions": False,
            "phe": False,
            "auto_tabpfn": False,
            "probability_fusion": False,
            "test_labels_passed_to_model": False,
            "allowed_calls_per_fold": ["fit(X_train, y_train)", "predict_proba(X_test)"],
            "prediction_rule": "argmax of predict_proba in fixed class order AD,CN,SMCI",
            "fresh_classifier_per_fold": True,
            "smoke_excluded_from_formal_results": True,
        },
        "git": git,
        "data": protocol["audit"],
        "checkpoint": checkpoint_audit(checkpoint),
        "environment": environment_audit(),
        "constructor": {
            "model_path": str(checkpoint),
            "n_estimators": 1,
            "auto_scale_n_estimators": False,
            "random_state": 0,
            "device": "cuda:0",
            "fit_mode": "low_memory",
            "memory_saving_mode": True,
            "inference_precision": "auto",
            "show_progress_bar": False,
            "all_other_parameters": "tabpfn==8.2.0 defaults",
        },
        "historical_headline_metrics": HEADLINE_BASELINES,
        "auc_comparison_note": {
            "tabpfn_definition": "multiclass OVR macro AUC from predict_proba",
            "fair_common_definition": "multiclass OVR macro AUC from probability columns for all three models",
            "historical_definition": "Original Query and SEPS-Q reports used per-class adjusted logits; retained separately and not treated as a same-scale TabPFN comparison",
        },
        "source_files": {
            "runner": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "recompute": {
                "path": "scripts/recompute_tabpfn_baseline_v1.py",
                "sha256": file_sha256(ROOT / "scripts/recompute_tabpfn_baseline_v1.py"),
            },
        },
        "static_preflight_passed": True,
    }


def run_experiment(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    protocol = load_protocol()
    preflight = preflight_payload(protocol, checkpoint)
    if args.mode == "preflight":
        print(json.dumps(safe_json(preflight), ensure_ascii=False, indent=2, sort_keys=True))
        return

    run_kind = args.mode
    folds = (0,) if run_kind == "smoke" else FOLDS
    output_parent = Path(args.output_parent).resolve()
    if run_kind == "formal":
        require(args.original_oof is not None, "Formal mode requires --original-oof")
        original_oof = Path(args.original_oof).resolve()
        require(original_oof.is_file(), f"Original Query OOF missing: {original_oof}")
        require(
            file_sha256(original_oof) == EXPECTED_ORIGINAL_OOF_SHA256,
            "Original Query OOF SHA256 mismatch",
        )
        smoke_dir = output_parent / "smoke"
        require(smoke_dir.is_dir(), "Formal mode requires a completed smoke directory")
        smoke_summary = read_json(smoke_dir / "smoke_summary.json")
        require(smoke_summary["smoke_passed"] is True, "Smoke did not pass")
        smoke_audit = read_json(smoke_dir / "fold_00/model_audit.json")
        require(
            all(smoke_audit["single_estimator_gates"].values()),
            "Smoke single-estimator gates did not all pass",
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    final_dir = output_parent / run_kind
    staging_dir = output_parent / f".{run_kind}.staging"
    require(not final_dir.exists(), f"Refusing to overwrite existing output: {final_dir}")
    require(not staging_dir.exists(), f"Retained staging output exists: {staging_dir}")
    staging_dir.mkdir(parents=False)
    log_path = staging_dir / "run.txt"

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    started = time.perf_counter()
    preflight["experiment_run"] = {
        "kind": run_kind,
        "folds": list(folds),
        "started_at_utc": utc_now(),
        "formal_reportable": run_kind == "formal",
    }
    write_json(staging_dir / "manifest.json", preflight)
    write_json(staging_dir / "environment.json", preflight["environment"])
    write_json(staging_dir / "data_preflight.json", protocol["audit"])
    log(f"start kind={run_kind} branch={preflight['git']['branch']} head={preflight['git']['head']}")
    log(f"checkpoint_sha256={preflight['checkpoint']['sha256']} cuda={preflight['environment']['gpu_name']}")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    instance_uuids: list[str] = []
    try:
        for fold in folds:
            frame, fold_metrics, instance_uuid = run_fold(
                protocol,
                checkpoint,
                fold,
                staging_dir / f"fold_{fold:02d}",
                log,
            )
            predictions.append(frame)
            metrics.append(fold_metrics)
            instance_uuids.append(instance_uuid)

        if run_kind == "formal":
            aggregate = aggregate_formal(
                staging_dir,
                predictions,
                metrics,
                instance_uuids,
                Path(args.original_oof).resolve() if args.original_oof else None,
            )
            log(
                f"formal_oof acc={aggregate['oof_metrics']['acc']:.9f} "
                f"macro_auc={aggregate['oof_metrics']['macro_auc']:.9f}"
            )
        else:
            smoke_summary = {
                "smoke_passed": True,
                "excluded_from_formal_results": True,
                "fold": 0,
                "metrics": metrics[0],
                "fresh_classifier_uuid": instance_uuids[0],
                "checks": {
                    "cuda_available": True,
                    "checkpoint_loaded": True,
                    "single_estimator": True,
                    "predict_proba_shape": metrics[0]["probability_validation"]["shape"],
                    "probability_rows_sum_to_one": True,
                    "fixed_class_order": list(CLASS_ORDER),
                    "no_nan_or_inf": True,
                    "artifacts_saved_and_read_back": True,
                    "metrics_exactly_recomputed": True,
                },
            }
            write_json(staging_dir / "smoke_summary.json", smoke_summary)

        preflight["experiment_run"].update(
            {
                "completed_at_utc": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "completed": True,
                "fresh_classifier_instance_uuids": instance_uuids,
            }
        )
        write_json(staging_dir / "manifest.json", preflight)
        log(f"complete kind={run_kind} wall_seconds={preflight['experiment_run']['wall_seconds']:.3f}")
        if run_kind == "formal":
            validation = validate_formal_readback(staging_dir)
            require(validation["passed"], "Formal readback validation failed")
        else:
            require(read_json(staging_dir / "smoke_summary.json")["smoke_passed"], "Smoke summary failed")
        staging_dir.rename(final_dir)
        if run_kind == "formal":
            print(f"FORMAL_READBACK_VALIDATION={validation['passed']}", flush=True)
        else:
            print("SMOKE_PASSED=True", flush=True)
    except BaseException as exc:
        write_json(
            staging_dir / "failure.json",
            {
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "credentials_recorded": False,
            },
        )
        log(f"failed kind={run_kind} exception={type(exc).__name__} message={exc}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked TabPFN-3 single-estimator TADPOLE baseline."
    )
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True, help="Explicit official TabPFN-3 checkpoint path")
    parser.add_argument(
        "--output-parent",
        default=str(DEFAULT_OUTPUT_PARENT),
        help="Parent directory for isolated smoke/formal artifacts",
    )
    parser.add_argument(
        "--original-oof",
        default=None,
        help="Optional historical Original Query OOF CSV used only for aligned comparison",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
