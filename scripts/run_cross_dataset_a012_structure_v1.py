#!/usr/bin/env python3
"""Paired cross-dataset Original versus A012-Struct experiment.

The runner intentionally preserves each task's last formal binary training
configuration while rerunning both arms with one fresh seed-0 model per fold.
Legacy three-seed artifacts are inspected as provenance only and are never used
as B0 predictions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2  # noqa: E402
from Model import HeterGraph_Model_Kmeans  # noqa: E402
from Model.cme_dual_branch import CMEDualBranchModel  # noqa: E402
from Utils import (  # noqa: E402
    CustomCosineAnnealingLR,
    SET_Random,
    load_dataset,
    load_path,
)


EXPERIMENT_ID = "cross_dataset_a012_structure_v1"
BRANCH = "experiment/cross-dataset-a012-structure-v1"
BASE_COMMIT = "c37c336eff5b5b7d9b62e98f84587dccf02b61e2"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_DIR = RESULT_DIR / "protocols"
EXPERIMENT_CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_MANIFEST_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
FORMAL_DIR = RESULT_DIR / "formal"
DEFAULT_HISTORICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "RESULT").is_dir() else ROOT

TASK_IDS = (
    "tadpole_smci_pmci",
    "abide_ads_cn",
    "abide5_ads_cn",
)
ARMS = ("B0", "B1")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 3
ADAPTER_RANK = 8
ADAPTER_LR_MULTIPLIER = 2.0

REQUIRED_TRACKED = (
    ".gitignore",
    "Model/cme_dual_branch.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocols/tadpole_smci_pmci.json",
    f"experiments/{EXPERIMENT_ID}/protocols/abide_ads_cn.json",
    f"experiments/{EXPERIMENT_ID}/protocols/abide5_ads_cn.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
)
LOCKED_BASE_DEPENDENCIES = (
    "Model/network.py",
    "Model/models.py",
    "Loss/loss_fn.py",
    "Utils/data_load.py",
    "Utils/graph_load.py",
    "Utils/utils.py",
    "config_re.py",
    "scripts/run_c1_broad_hparam_search_v1.py",
    "scripts/run_cme_dual_branch_v1.py",
    "Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini",
)
THREE_CLASS_A012_SCHEMA_SHA256 = "66d73f7f7a5a8eca4e39806fb814fbb738710bfe85238b88b8b2a10e0947737b"


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def current_source_commit() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def source_hashes() -> dict[str, str]:
    paths = list(REQUIRED_TRACKED) + list(LOCKED_BASE_DEPENDENCIES)
    return {path: file_sha256(ROOT / path) for path in paths}


def protocol_paths() -> dict[str, Path]:
    return {task_id: PROTOCOL_DIR / f"{task_id}.json" for task_id in TASK_IDS}


def load_protocols() -> dict[str, dict[str, Any]]:
    protocols: dict[str, dict[str, Any]] = {}
    for task_id, path in protocol_paths().items():
        require(path.is_file(), f"Missing protocol snapshot: {path}")
        protocol = read_json(path)
        require(protocol["task_id"] == task_id, f"Protocol task id mismatch: {path}")
        training = protocol["training"]
        require(training["epochs"] == EPOCHS, f"Epoch lock changed for {task_id}")
        require(training["seed"] == SEED, f"Seed lock changed for {task_id}")
        require(training["graph_use_graph"] is False, f"Graph must remain disabled: {task_id}")
        require(training["ema"] is False, f"EMA must remain disabled: {task_id}")
        require(training["logit_adjust_tau"] == 0.0, f"Logit adjustment changed: {task_id}")
        require(len(protocol["class_names"]) == 2, f"Expected binary task: {task_id}")
        require(protocol["positive_index"] != protocol["negative_index"], f"Class direction invalid: {task_id}")
        protocols[task_id] = protocol
    return protocols


def raw_csv_feature_hashes(path: Path, expected_feature_count: int) -> list[str]:
    """Hash only canonical feature values; the outcome must never enter an ID."""
    frame = pd.read_csv(path)
    require(frame.shape[1] == expected_feature_count + 1, f"CSV width changed: {path}")
    features = frame.iloc[:, :expected_feature_count].to_numpy(dtype=np.float64, copy=True)
    hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in features]
    require(len(set(hashes)) == len(hashes), f"Feature-only row hashes are not unique: {path}")
    return hashes


def stable_subject_id(task_id: str, original_index: int, feature_hash: str) -> str:
    return f"{task_id}:{original_index}:{feature_hash[:20]}"


def task_fold_manifest(
    task_id: str,
    protocol: dict[str, Any],
    dataset_dict: dict[str, Any],
    dataset_data: dict[str, Any],
    feature_hashes: list[str],
) -> dict[str, Any]:
    original_indices = np.asarray(dataset_dict["Index"], dtype=int)
    labels = dataset_data["Label"].detach().cpu().numpy().astype(int)
    require(len(feature_hashes) == protocol["sample_count"], f"Feature hash count changed: {task_id}")
    require(len(set(feature_hashes)) == len(feature_hashes), f"Feature hashes are not unique: {task_id}")
    folds: list[dict[str, Any]] = []
    seen: list[int] = []
    for fold, (train_mask_t, test_mask_t) in enumerate(dataset_data["Mask"]):
        train_mask = train_mask_t.detach().cpu().numpy().astype(bool)
        test_mask = test_mask_t.detach().cpu().numpy().astype(bool)
        require(not bool(np.any(train_mask & test_mask)), f"Fold overlap: {task_id}/{fold}")
        require(bool(np.all(train_mask | test_mask)), f"Fold coverage gap: {task_id}/{fold}")
        test_positions = np.flatnonzero(test_mask)
        test_rows = []
        for position in test_positions:
            original_index = int(original_indices[position])
            feature_hash = feature_hashes[original_index]
            test_rows.append(
                {
                    "subject_id": stable_subject_id(task_id, original_index, feature_hash),
                    "original_csv_index": original_index,
                    "feature_sha256": feature_hash,
                    "truth": int(labels[position]),
                }
            )
            seen.append(original_index)
        folds.append(
            {
                "fold": fold,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "test_rows": test_rows,
            }
        )
    require(len(seen) == protocol["sample_count"], f"OOF count changed: {task_id}")
    require(sorted(seen) == list(range(protocol["sample_count"])), f"OOF IDs changed: {task_id}")
    core = {"task_id": task_id, "folds": folds}
    return {**core, "sha256": payload_sha256(core)}


def build_context(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    task_id = protocol["task_id"]
    data_path = ROOT / protocol["data_csv"]
    modality_path = ROOT / protocol["modality_manifest"]
    require(data_path.is_file(), f"Missing data: {data_path}")
    require(modality_path.is_file(), f"Missing modality manifest: {modality_path}")
    require(file_sha256(data_path) == protocol["data_sha256"], f"Data hash changed: {task_id}")
    require(file_sha256(modality_path) == protocol["modality_sha256"], f"Modality hash changed: {task_id}")

    resolved_data, resolved_modal, _, class_names = load_path(
        str(ROOT), protocol["dataset"], protocol["task"]
    )
    require(Path(resolved_data).resolve() == data_path.resolve(), f"Data routing changed: {task_id}")
    require(Path(resolved_modal).resolve() == modality_path.resolve(), f"Modality routing changed: {task_id}")
    require(list(class_names) == protocol["class_names"], f"Class mapping changed: {task_id}")

    SET_Random(SEED)
    dataset_dict, dataset_data = load_dataset(
        str(data_path),
        str(modality_path),
        device,
        class_names,
        bool(protocol["training"]["shuffle"]),
        SEED,
        train_size=0,
    )
    require(dataset_dict["Sample_Num"] == protocol["sample_count"], f"Sample count changed: {task_id}")
    require(dataset_dict["Feature_Num"] == protocol["feature_count"], f"Feature count changed: {task_id}")
    actual_modalities = [
        {"name": name, "feature_count": len(indices)}
        for name, indices in zip(dataset_dict["Modal_Name"], dataset_dict["Modal_Index"])
    ]
    require(actual_modalities == protocol["modalities"], f"Modality manifest changed: {task_id}")
    labels = dataset_data["Label"].detach().cpu().numpy().astype(int)
    actual_counts = {
        protocol["class_names"][index]: int((labels == index).sum())
        for index in range(2)
    }
    require(actual_counts == protocol["class_counts"], f"Class counts changed: {task_id}")
    feature_hashes = raw_csv_feature_hashes(data_path, int(protocol["feature_count"]))
    fold_manifest = task_fold_manifest(
        task_id, protocol, dataset_dict, dataset_data, feature_hashes
    )
    return {
        "task_id": task_id,
        "protocol": protocol,
        "device": device,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "feature_hashes": feature_hashes,
        "fold_manifest": fold_manifest,
    }


def historical_evidence_inventory(protocol: dict[str, Any], historical_root: Path) -> dict[str, Any]:
    evidence = protocol["historical_b0_evidence"]
    result_path = historical_root / evidence["result_json"]
    logits_path = historical_root / evidence["raw_logits_npz"]
    result_ok = result_path.is_file() and file_sha256(result_path) == evidence["result_json_sha256"]
    logits_ok = logits_path.is_file() and file_sha256(logits_path) == evidence["raw_logits_sha256"]
    structure_ok = False
    subject_coverage_ok = False
    if logits_ok:
        with np.load(logits_path, allow_pickle=False) as archive:
            structure_ok = all(
                archive[f"fold_{fold}_main"].shape[0:2] == (3, EPOCHS)
                for fold in range(1, 11)
            )
            indices = np.concatenate(
                [archive[f"fold_{fold}_sample_index"] for fold in range(1, 11)]
            ).astype(int)
            subject_coverage_ok = np.array_equal(
                np.sort(indices), np.arange(protocol["sample_count"])
            )
    return {
        "result_json": evidence["result_json"],
        "result_json_sha256": evidence["result_json_sha256"],
        "result_json_verified": result_ok,
        "raw_logits_npz": evidence["raw_logits_npz"],
        "raw_logits_sha256": evidence["raw_logits_sha256"],
        "raw_logits_verified": logits_ok,
        "legacy_shape_three_seeds_by_400_epochs": structure_ok,
        "legacy_original_row_index_coverage": subject_coverage_ok,
        "eligible_for_direct_reuse": False,
        "rejection_reasons": [
            "legacy artifact used three seeds while this experiment locks one seed-0 model",
            "legacy artifact does not lock its training source commit",
            "legacy artifact stores original CSV row indices, not external durable subject identifiers",
            "legacy selection contained historical multi-seed ensemble behavior",
        ],
    }


def shared_abide_overlap(protocols: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = pd.read_csv(ROOT / protocols["abide_ads_cn"]["data_csv"], low_memory=False)
    right = pd.read_csv(ROOT / protocols["abide5_ads_cn"]["data_csv"], low_memory=False)
    common = [column for column in left.columns if column in right.columns and column != "label"]
    require(len(common) == 64, f"ABIDE shared feature schema changed: {len(common)}")

    def signatures(frame: pd.DataFrame, include_label: bool) -> list[str]:
        columns = common + (["label"] if include_label else [])
        values = frame[columns].to_numpy(dtype="<f8", copy=True)
        return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]

    left_features = signatures(left, False)
    right_features = signatures(right, False)
    left_with_label = signatures(left, True)
    right_with_label = signatures(right, True)
    require(len(set(left_features)) == len(left_features), "Duplicate ABIDE rows in shared feature space")
    require(len(set(right_features)) == len(right_features), "Duplicate ABIDE-5 rows in shared feature space")
    overlap_features = len(set(left_features) & set(right_features))
    overlap_with_label = len(set(left_with_label) & set(right_with_label))
    require(overlap_features == 841 and overlap_with_label == 841, "ABIDE overlap anchor changed")
    return {
        "shared_feature_count": len(common),
        "shared_feature_schema_sha256": payload_sha256(common),
        "abide_unique_shared_rows": len(set(left_features)),
        "abide5_unique_shared_rows": len(set(right_features)),
        "cross_dataset_shared_feature_overlap": overlap_features,
        "cross_dataset_shared_feature_and_label_overlap": overlap_with_label,
        "interpretation": "ABIDE-5 is a highly overlapping task view; tasks remain separately trained and reported.",
    }


def make_inspect_payload(historical_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocols = load_protocols()
    task_rows: dict[str, Any] = {}
    fold_manifests: dict[str, Any] = {}
    for task_id in TASK_IDS:
        context = build_context(protocols[task_id], torch.device("cpu"))
        protocol = protocols[task_id]
        task_rows[task_id] = {
            "dataset": protocol["dataset"],
            "task": protocol["task"],
            "sample_count": protocol["sample_count"],
            "class_names": protocol["class_names"],
            "class_counts": protocol["class_counts"],
            "positive_class": protocol["positive_class"],
            "positive_class_semantics": protocol.get("positive_class_semantics"),
            "positive_index": protocol["positive_index"],
            "negative_class": protocol["negative_class"],
            "negative_class_semantics": protocol.get("negative_class_semantics"),
            "negative_index": protocol["negative_index"],
            "feature_count": protocol["feature_count"],
            "modalities": protocol["modalities"],
            "data_csv": protocol["data_csv"],
            "data_sha256": protocol["data_sha256"],
            "modality_manifest": protocol["modality_manifest"],
            "modality_sha256": protocol["modality_sha256"],
            "fold_manifest_sha256": context["fold_manifest"]["sha256"],
            "stable_subject_proxy": "dataset + original CSV index + feature-only canonical float64 row SHA256 (label excluded)",
            "durable_external_subject_id_available": False,
            "historical_protocol": protocol["training"],
            "historical_b0": historical_evidence_inventory(protocol, historical_root),
        }
        fold_manifests[task_id] = context["fold_manifest"]
    fold_core = {"tasks": fold_manifests}
    fold_payload = {**fold_core, "sha256": payload_sha256(fold_core)}
    inspect_core = {
        "experiment": EXPERIMENT_ID,
        "tasks": task_rows,
        "abide_overlap": shared_abide_overlap(protocols),
        "historical_b0_reused": False,
        "paired_training_required": True,
        "legacy_limitations": {
            "offline_preprocessed_inputs": True,
            "global_class_weights": True,
            "test_fold_checkpoint_selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
            "no_additional_test_driven_search": True,
        },
        "protocol_snapshot_sha256": {
            task_id: file_sha256(protocol_paths()[task_id]) for task_id in TASK_IDS
        },
        "fold_manifest_sha256": fold_payload["sha256"],
    }
    inspect_payload = {**inspect_core, "sha256": payload_sha256(inspect_core)}
    return inspect_payload, fold_payload


def run_inspect(historical_root: Path) -> None:
    inspect_payload, fold_payload = make_inspect_payload(historical_root.resolve())
    atomic_write_json(INSPECT_MANIFEST_PATH, inspect_payload)
    atomic_write_json(FOLD_MANIFEST_PATH, fold_payload)
    print(json.dumps(inspect_payload, indent=2, ensure_ascii=False))


EXPECTED_PARAMETERS = {
    "tadpole_smci_pmci": {"B0": 608997, "B1": 617197, "delta": 8200},
    "abide_ads_cn": {"B0": 333033, "B1": 337417, "delta": 4384},
    "abide5_ads_cn": {"B0": 380716, "B1": 386196, "delta": 5480},
}


def runtime_lock(protocols: dict[str, dict[str, Any]], source_commit: str) -> dict[str, Any]:
    inspect_manifest = read_json(INSPECT_MANIFEST_PATH)
    fold_manifest = read_json(FOLD_MANIFEST_PATH)
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "experiment_config_sha256": file_sha256(EXPERIMENT_CONFIG_PATH),
        "protocol_sha256": {
            task_id: file_sha256(protocol_paths()[task_id]) for task_id in TASK_IDS
        },
        "inspect_manifest_sha256": file_sha256(INSPECT_MANIFEST_PATH),
        "inspect_payload_sha256": inspect_manifest["sha256"],
        "fold_manifest_file_sha256": file_sha256(FOLD_MANIFEST_PATH),
        "fold_manifest_payload_sha256": fold_manifest["sha256"],
        "tasks": list(TASK_IDS),
        "arms": list(ARMS),
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs": EPOCHS,
        "device": "cuda:0",
        "adapter_rank": ADAPTER_RANK,
        "adapter_lr_multiplier": ADAPTER_LR_MULTIPLIER,
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "protocol_training": {
            task_id: protocols[task_id]["training"] for task_id in TASK_IDS
        },
    }
    return {**core, "sha256": payload_sha256(core)}


def validate_inspect_artifacts(historical_root: Path) -> None:
    require(INSPECT_MANIFEST_PATH.is_file(), "inspect_manifest.json missing; run inspect first")
    require(FOLD_MANIFEST_PATH.is_file(), "fold_manifest.json missing; run inspect first")
    expected_inspect, expected_folds = make_inspect_payload(historical_root.resolve())
    require(read_json(INSPECT_MANIFEST_PATH) == expected_inspect, "Inspect manifest drifted")
    require(read_json(FOLD_MANIFEST_PATH) == expected_folds, "Fold manifest drifted")


def source_gate(historical_root: Path) -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree has uncommitted drift")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index has staged drift")
    head = current_source_commit()
    require(head != BASE_COMMIT, "Implementation must be committed before smoke")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Base commit is not an ancestor")
    changed_from_base = {
        line.strip().replace("\\", "/")
        for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines()
        if line.strip()
    }
    require(changed_from_base == set(REQUIRED_TRACKED), f"Source commit scope changed: {sorted(changed_from_base)}")
    for relative in REQUIRED_TRACKED:
        require((ROOT / relative).is_file(), f"Required source missing: {relative}")
        require(git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0, f"Required source is untracked: {relative}")
        require(git("diff", "--quiet", "HEAD", "--", relative, check=False).returncode == 0, f"Required source has uncommitted drift: {relative}")
    for relative in LOCKED_BASE_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", relative, check=False).returncode == 0, f"Locked dependency changed from base: {relative}")
        require(git("diff", "--quiet", "HEAD", "--", relative, check=False).returncode == 0, f"Locked dependency has worktree drift: {relative}")
    validate_inspect_artifacts(historical_root)
    return head


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def model_kwargs(context: dict[str, Any]) -> dict[str, Any]:
    training = context["protocol"]["training"]
    return {
        "DATASET_Dict": context["dataset_dict"],
        "Herter_Graph": None,
        "Hidden_size": int(training["hidden_size"]),
        "Drop_rate": float(training["drop_rate"]),
        "K": int(training["cheb_k"]),
        "num_layers": int(training["modal_layers"]),
        "num_heads": int(training["modal_heads"]),
        "input_noise_std": float(training["input_noise_std"]),
        "drop_path": float(training["drop_path"]),
        "graph_head": training["graph_head"],
        "graph_layers": int(training["graph_layers"]),
        "graph_heads": int(training["graph_heads"]),
        "graph_beta": float(training["graph_beta"]),
        "graph_k_order": int(training["graph_k_order"]),
        "graph_alpha": float(training["graph_alpha"]),
        "graph_kernel": training["graph_kernel"],
        "graph_use_graph": bool(training["graph_use_graph"]),
        "graph_dropout": float(training["graph_dropout"]),
        "graph_hidden": int(training["graph_hidden"]),
        "global_word_emb": int(training["global_word_emb"]),
        "semantic_branch": training["semantic_branch"],
        "semantic_fusion": training["semantic_fusion"],
        "category_branch_variant": "original",
        "query_pool_variant": "independent",
        "category_branch_fusion": "concat",
        "adj_mode": training["adj_mode"],
        "label_graph_alpha": 0.0,
        "label_graph_topk": 0,
        "label_graph_reg_lambda": 0.0,
    }


def common_state_max_diff(b0: torch.nn.Module, b1: torch.nn.Module) -> float:
    left = b0.state_dict()
    right = b1.state_dict()
    require(set(left).issubset(set(right)), "B1 lost a B0 state key")
    values = []
    for name, tensor in left.items():
        require(tensor.shape == right[name].shape and tensor.dtype == right[name].dtype, f"Common state schema changed: {name}")
        values.append(float((tensor.detach().cpu() - right[name].detach().cpu()).abs().max()))
    return max(values, default=0.0)


def build_model(context: dict[str, Any], arm: str, seed: int = SEED, audit_common: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    require(arm in ARMS, f"Unknown arm: {arm}")
    kwargs = model_kwargs(context)
    initialization_audit: dict[str, Any] = {}
    if arm == "B0":
        SET_Random(seed)
        model = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = capture_rng_state()
    else:
        SET_Random(seed)
        reference = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = capture_rng_state()
        SET_Random(seed)
        model = CMEDualBranchModel(
            **kwargs,
            cme_arm="c1",
            adapter_rank=ADAPTER_RANK,
            router_hidden=16,
            modality_embedding_dim=8,
        ).to(context["device"])
        if audit_common:
            initialization_audit["common_state_max_abs_diff"] = common_state_max_diff(reference, model)
            require(initialization_audit["common_state_max_abs_diff"] == 0.0, "Common initialization changed")
        else:
            initialization_audit["common_state_pointwise_audit"] = "not_repeated"
        del reference
        restore_rng_state(post_common_rng)

    expected = EXPECTED_PARAMETERS[context["task_id"]][arm]
    require(parameter_count(model) == expected, f"Parameter count changed: {context['task_id']}/{arm}")
    require(len(model.label_pools) == 2, "Binary model must have exactly two category queries")
    require(len(model._Auxi_classifier) == 2, "Binary model must have exactly two OVR heads")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    if arm == "B1":
        expected_modalities = len(context["protocol"]["modalities"])
        require(len(model.private_adapters) == expected_modalities, "Adapter count does not match real modalities")
    initialization_audit.update(
        {
            "parameter_count": parameter_count(model),
            "query_count": len(model.label_pools),
            "ovr_head_count": len(model._Auxi_classifier),
            "adapter_count": len(model.private_adapters) if arm == "B1" else 0,
        }
    )
    return model, initialization_audit


class RatioPreservingCustomCosineAnnealingLR(CustomCosineAnnealingLR):
    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int, eta_min: float, multiplier: float, last_epoch: int = -1):
        require(len(optimizer.param_groups) == 2, "Ratio scheduler requires two parameter groups")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.hold_epoch = 20
        self.multiplier = float(multiplier)
        self.initial_lrs_locked = [float(group["lr"]) for group in optimizer.param_groups]
        require(abs(self.initial_lrs_locked[1] - self.initial_lrs_locked[0] * self.multiplier) <= 1e-14, "Initial LR ratio changed")
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch)
        self.assert_ratio()

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.hold_epoch:
            base_lr = self.initial_lrs_locked[0]
        else:
            current = self.last_epoch - self.hold_epoch
            base_lr = self.eta_min + (self.initial_lrs_locked[0] - self.eta_min) * (
                1.0 + math.cos(math.pi * current / (self.T_max - self.hold_epoch))
            ) / 2.0
        return [base_lr, base_lr * self.multiplier]

    def assert_ratio(self) -> None:
        base_lr, adapter_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
        require(abs(adapter_lr - base_lr * self.multiplier) <= max(1e-14, abs(adapter_lr) * 1e-12), "Adapter/base LR ratio changed")

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        self.assert_ratio()


def make_training_objects(context: dict[str, Any], arm: str, audit_common: bool = False) -> tuple[torch.nn.Module, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    model, initialization = build_model(context, arm, SEED, audit_common=audit_common)
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(
        context["dataset_dict"],
        context["device"],
        rate=float(training["loss_rate"]),
        label_smoothing=float(training["label_smoothing"]),
    )
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if arm == "B0":
        optimizer = torch.optim.Adam(
            [parameter for _, parameter in named],
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = CustomCosineAnnealingLR(
            optimizer,
            T_max=int(training["scheduler_t_max"]),
            eta_min=float(training["scheduler_eta_min"]),
        )
        group_audit = {
            "base_parameter_tensors": len(named),
            "adapter_parameter_tensors": 0,
            "base_lr": float(training["lr"]),
        }
    else:
        adapter_named = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
        base_named = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
        all_ids = {id(parameter) for _, parameter in named}
        adapter_ids = {id(parameter) for _, parameter in adapter_named}
        base_ids = {id(parameter) for _, parameter in base_named}
        expected_tensors = len(context["protocol"]["modalities"]) * 4
        require(len(adapter_named) == expected_tensors, "Adapter parameter tensor count changed")
        require(not (base_ids & adapter_ids) and base_ids | adapter_ids == all_ids, "Optimizer partition invalid")
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in base_named],
                    "lr": float(training["lr"]),
                    "weight_decay": float(training["weight_decay"]),
                    "group_name": "base",
                },
                {
                    "params": [parameter for _, parameter in adapter_named],
                    "lr": float(training["lr"]) * ADAPTER_LR_MULTIPLIER,
                    "weight_decay": float(training["weight_decay"]),
                    "group_name": "private_adapters",
                },
            ]
        )
        scheduler = RatioPreservingCustomCosineAnnealingLR(
            optimizer,
            T_max=int(training["scheduler_t_max"]),
            eta_min=float(training["scheduler_eta_min"]),
            multiplier=ADAPTER_LR_MULTIPLIER,
        )
        group_audit = {
            "base_parameter_tensors": len(base_named),
            "adapter_parameter_tensors": len(adapter_named),
            "adapter_parameter_names": [name for name, _ in adapter_named],
            "base_lr": float(training["lr"]),
            "adapter_lr": float(training["lr"]) * ADAPTER_LR_MULTIPLIER,
        }
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler, {**initialization, **group_audit}


def binary_metrics(truth: np.ndarray, probabilities: np.ndarray, protocol: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(probabilities.shape == (len(truth), 2), "Binary probability shape changed")
    require(bool(np.isfinite(probabilities).all()), "Probabilities are non-finite")
    require(float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) <= 1e-6, "Probability simplex changed")
    prediction = probabilities.argmax(axis=1).astype(int)
    positive = int(protocol["positive_index"])
    truth_positive = (truth == positive).astype(int)
    prediction_positive = (prediction == positive).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth_positive, prediction_positive, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(truth_positive, probabilities[:, positive]))
    pr_auc = float(average_precision_score(truth_positive, probabilities[:, positive]))
    return {
        "n": int(len(truth)),
        "correct": int((prediction == truth).sum()),
        "acc": float((prediction == truth).mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "sen": float(tp / (tp + fn)),
        "spe": float(tn / (tn + fp)),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "confusion_matrix_class_order": protocol["class_names"],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).astype(int).tolist(),
        "predicted_counts": {
            protocol["class_names"][index]: int((prediction == index).sum())
            for index in range(2)
        },
        "positive_class": protocol["positive_class"],
        "positive_index": positive,
        "negative_class": protocol["negative_class"],
        "negative_index": int(protocol["negative_index"]),
    }


def selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(metrics["acc"]),
        float(metrics["roc_auc"]),
        float(metrics["macro_f1"]),
        -int(epoch),
    )


@torch.no_grad()
def infer(model: torch.nn.Module, features: torch.Tensor, return_intermediates: bool = False):
    model.eval()
    return model(features, return_intermediates=return_intermediates)


def train_update(
    model: torch.nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    grad_clip: float,
) -> tuple[float, dict[str, float], tuple[torch.Tensor, Any, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, embeddings, auxiliary = model(features)
    require(tuple(output.shape) == (features.size(0), 2), "Training logits must be [N,2]")
    loss = criterion(output, labels, train_mask, embeddings, auxiliary)
    require(bool(torch.isfinite(loss)), "Loss is non-finite")
    loss.backward()
    gradient_max: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid adapter gradient: {name}")
            gradient_max[name] = float(parameter.grad.detach().abs().max().cpu())
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), "Model gradient is non-finite")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu()), gradient_max, (output, embeddings, auxiliary)


def fold_positions(context: dict[str, Any], fold: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool(torch.any(train_mask & test_mask)), "Train/test masks overlap")
    original_indices = np.asarray(context["dataset_dict"]["Index"], dtype=int)
    return train_mask, test_mask, original_indices[test_mask.detach().cpu().numpy().astype(bool)]


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    lock: dict[str, Any],
    task_id: str,
    epoch: int,
) -> dict[str, Any]:
    return {
        "lock": lock,
        "task_id": task_id,
        "epoch": int(epoch),
        "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
    }


def three_class_compatibility_regression(device_text: str) -> dict[str, Any]:
    """Guard the historical six-modality, three-class A012 state schema."""
    import scripts.run_c1_broad_hparam_search_v1 as broad

    context = broad.load_context(device_text)
    model = broad.build_model(context, ADAPTER_RANK)
    dropout_audit = broad.apply_dropout_multiplier(model, 1.1)
    schema = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    ]
    schema_sha256 = payload_sha256(schema)
    require(parameter_count(model) == 862_971, "Three-class A012 parameter count changed")
    require(len(schema) == 184 and schema_sha256 == THREE_CLASS_A012_SCHEMA_SHA256, "Three-class A012 state schema changed")
    require(len(model.label_pools) == 3 and len(model._Auxi_classifier) == 3, "Three-class Query/OVR count changed")
    output, _, _, intermediates = infer(model, context["dataset_data"]["Feature"], return_intermediates=True)
    stream_shape = list(intermediates["category_token_streams"].shape)
    require(tuple(output.shape) == (598, 3), "Three-class A012 output shape changed")
    require(stream_shape[1:3] == [3, 6], "Three-class A012 category stream shape changed")
    report = {
        "passed": True,
        "parameter_count": parameter_count(model),
        "state_tensor_count": len(schema),
        "state_key_shape_dtype_schema_sha256": schema_sha256,
        "schema_anchor": "constant derived from the verified real A012 fold-0 checkpoint; checkpoint is not a runtime dependency",
        "query_count": len(model.label_pools),
        "ovr_head_count": len(model._Auxi_classifier),
        "category_token_stream_shape": stream_shape,
        "output_shape": list(output.shape),
        "dropout_module_count": len(dropout_audit),
    }
    del model
    torch.cuda.empty_cache()
    return report


def run_smoke(device_text: str, historical_root: Path) -> None:
    require(device_text == "cuda:0", "Smoke requires cuda:0")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    source_commit = source_gate(historical_root)
    protocols = load_protocols()
    lock = runtime_lock(protocols, source_commit)
    smoke_config = {"runtime_lock": lock, "smoke_epochs": SMOKE_EPOCHS, "tasks": list(TASK_IDS), "arm": "B1"}
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    task_reports: dict[str, Any] = {}
    three_class_regression = three_class_compatibility_regression(device_text)

    for task_index, task_id in enumerate(TASK_IDS):
        context = build_context(protocols[task_id], torch.device(device_text))
        features = context["dataset_data"]["Feature"]
        labels = context["dataset_data"]["Label"]
        train_mask, test_mask, _ = fold_positions(context, 0)
        model, criterion, optimizer, scheduler, object_audit = make_training_objects(
            context, "B1", audit_common=(task_index == 0)
        )
        model.eval()
        with torch.no_grad():
            candidate_logits = model(features)[0]
        zero_output_audit: dict[str, Any] = {}
        if task_index == 0:
            base_model, _ = build_model(context, "B0")
            base_model.eval()
            with torch.no_grad():
                base_logits = base_model(features)[0]
            initial_diff = float((candidate_logits - base_logits).abs().max().cpu())
            require(initial_diff <= 1e-6, "B1 zero-output initialization changed initial logits")
            zero_output_audit = {"b0_b1_initial_logits_max_abs_diff": initial_diff}
            del base_model

        cumulative_gradient = {
            name: 0.0 for name, _ in model.named_parameters() if name.startswith("private_adapters.")
        }
        initial_adapter_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("private_adapters.")
        }
        losses = []
        label_leak_max_diff = 0.0
        for epoch in range(1, SMOKE_EPOCHS + 1):
            loss, gradients, train_outputs = train_update(
                model,
                criterion,
                optimizer,
                features,
                labels,
                train_mask,
                float(context["protocol"]["training"]["grad_clip"]),
            )
            losses.append(loss)
            for name, value in gradients.items():
                cumulative_gradient[name] = max(cumulative_gradient[name], value)
            output, embeddings, auxiliary = train_outputs
            labels_changed = labels.clone()
            labels_changed[test_mask] = 1 - labels_changed[test_mask]
            with torch.no_grad():
                original_loss = criterion(output, labels, train_mask, embeddings, auxiliary)
                changed_loss = criterion(output, labels_changed, train_mask, embeddings, auxiliary)
            label_leak_max_diff = max(label_leak_max_diff, float((original_loss - changed_loss).abs().cpu()))
            require(label_leak_max_diff == 0.0, "Test labels enter the training criterion")
            scheduler.step()
            if isinstance(scheduler, RatioPreservingCustomCosineAnnealingLR):
                scheduler.assert_ratio()

        require(cumulative_gradient and all(math.isfinite(value) and value > 0.0 for value in cumulative_gradient.values()), "Every adapter tensor must receive a finite non-zero gradient by epoch 3")
        adapter_delta = {}
        for name, parameter in model.named_parameters():
            if name in initial_adapter_state:
                value = float((parameter.detach().cpu() - initial_adapter_state[name]).abs().max())
                adapter_delta[name] = value
        require(all(math.isfinite(value) and value > 0.0 for value in adapter_delta.values()), "Every adapter tensor must update by epoch 3")

        output, _, _, intermediates = infer(model, features, return_intermediates=True)
        probabilities = torch.softmax(output, dim=-1)
        simplex_error = float((probabilities.sum(dim=-1) - 1.0).abs().max().cpu())
        require(simplex_error <= 1e-6 and bool(torch.isfinite(probabilities).all()), "Smoke probabilities invalid")
        private = intermediates["private_residuals"]
        shared = intermediates["modal_tokens_post_transformer"]
        ratio = private.norm(dim=-1) / shared.norm(dim=-1).clamp_min(1e-12)
        require(bool(torch.isfinite(ratio).all()), "Private/shared ratio is non-finite")

        checkpoint_path = SMOKE_DIR / f"checkpoint_roundtrip_{task_id}.pt"
        payload = checkpoint_payload(model, optimizer, scheduler, lock, task_id, SMOKE_EPOCHS)
        atomic_torch_save(checkpoint_path, payload)
        loaded = torch.load(checkpoint_path, map_location=context["device"], weights_only=False)
        require(loaded["lock"] == lock and loaded["epoch"] == SMOKE_EPOCHS, "Smoke checkpoint lock changed")
        fresh_model, fresh_criterion, fresh_optimizer, fresh_scheduler, _ = make_training_objects(context, "B1")
        fresh_model.load_state_dict(loaded["model"], strict=True)
        fresh_optimizer.load_state_dict(loaded["optimizer"])
        fresh_scheduler.load_state_dict(loaded["scheduler"])
        fresh_output = infer(fresh_model, features)[0]
        reload_diff = float((fresh_output - output).abs().max().cpu())
        require(reload_diff <= 1e-7, "Smoke checkpoint inference changed")

        task_reports[task_id] = {
            "passed": True,
            "losses": losses,
            "logits_shape": list(output.shape),
            "object_audit": object_audit,
            "zero_output_audit": zero_output_audit,
            "test_label_loss_invariance_max_abs_diff": label_leak_max_diff,
            "adapter_cumulative_max_gradient": cumulative_gradient,
            "adapter_parameter_max_delta": adapter_delta,
            "private_shared_ratio_mean": float(ratio.mean().cpu()),
            "private_shared_ratio_max": float(ratio.max().cpu()),
            "probability_simplex_max_abs_error": simplex_error,
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_reload_logits_max_abs_diff": reload_diff,
            "train_count": int(train_mask.sum().item()),
            "test_count": int(test_mask.sum().item()),
        }
        del model, fresh_model, criterion, fresh_criterion, optimizer, fresh_optimizer
        torch.cuda.empty_cache()

    report_core = {
        "passed": True,
        "runtime_lock": lock,
        "smoke_config_sha256": file_sha256(SMOKE_DIR / "smoke_config.json"),
        "three_class_compatibility_regression": three_class_regression,
        "tasks": task_reports,
    }
    report = {**report_core, "sha256": payload_sha256(report_core)}
    atomic_write_json(SMOKE_DIR / "smoke_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def validate_smoke(lock: dict[str, Any]) -> dict[str, Any]:
    config_path = SMOKE_DIR / "smoke_config.json"
    report_path = SMOKE_DIR / "smoke_report.json"
    require(config_path.is_file() and report_path.is_file(), "Committed-source smoke artifacts are missing")
    config = read_json(config_path)
    report = read_json(report_path)
    require(config == {"runtime_lock": lock, "smoke_epochs": SMOKE_EPOCHS, "tasks": list(TASK_IDS), "arm": "B1"}, "Smoke configuration drifted")
    report_core = {key: value for key, value in report.items() if key != "sha256"}
    require(report.get("sha256") == payload_sha256(report_core), "Smoke report digest changed")
    require(report.get("passed") is True and report.get("runtime_lock") == lock, "Smoke did not pass this source/configuration")
    regression = report.get("three_class_compatibility_regression", {})
    require(
        regression.get("passed") is True
        and regression.get("parameter_count") == 862_971
        and regression.get("state_tensor_count") == 184
        and regression.get("state_key_shape_dtype_schema_sha256") == THREE_CLASS_A012_SCHEMA_SHA256,
        "Three-class compatibility smoke changed",
    )
    require(report.get("smoke_config_sha256") == file_sha256(config_path), "Smoke config hash changed")
    require(set(report.get("tasks", {})) == set(TASK_IDS), "Smoke task coverage changed")
    for task_id in TASK_IDS:
        task = report["tasks"][task_id]
        require(task.get("passed") is True, f"Smoke task failed: {task_id}")
        checkpoint = ROOT / task["checkpoint"]
        require(checkpoint.is_file() and file_sha256(checkpoint) == task["checkpoint_sha256"], f"Smoke checkpoint changed: {task_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(payload.get("lock") == lock and payload.get("task_id") == task_id, f"Smoke checkpoint lock changed: {task_id}")
        require(int(payload.get("epoch", -1)) == SMOKE_EPOCHS, f"Smoke checkpoint epoch changed: {task_id}")
    return report


def fold_lock(lock: dict[str, Any], context: dict[str, Any], arm: str, fold: int) -> dict[str, Any]:
    core = {
        "runtime_lock_sha256": lock["sha256"],
        "source_commit": lock["source_commit"],
        "task_id": context["task_id"],
        "task_protocol_sha256": lock["protocol_sha256"][context["task_id"]],
        "task_fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "arm": arm,
        "fold": int(fold),
        "seed": SEED,
        "epochs": EPOCHS,
    }
    return {**core, "sha256": payload_sha256(core)}


def expected_fold_rows(context: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    rows = context["fold_manifest"]["folds"][fold]["test_rows"]
    require(context["fold_manifest"]["folds"][fold]["fold"] == fold, "Fold manifest order changed")
    return rows


def prediction_rows(
    context: dict[str, Any],
    fold: int,
    logits: np.ndarray,
    probabilities: np.ndarray,
    truth: np.ndarray,
) -> list[dict[str, Any]]:
    expected = expected_fold_rows(context, fold)
    require(logits.shape == probabilities.shape == (len(expected), 2), "Fold output shape changed")
    require(truth.shape == (len(expected),), "Fold truth shape changed")
    predictions = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(expected):
        require(int(truth[index]) == int(anchor["truth"]), "Fold truth no longer aligns to stable subject proxy")
        rows.append(
            {
                "task_id": context["task_id"],
                "arm": "",
                "fold": int(fold),
                "subject_id": anchor["subject_id"],
                "original_csv_index": int(anchor["original_csv_index"]),
                "feature_sha256": anchor["feature_sha256"],
                "truth": int(truth[index]),
                "prediction": int(predictions[index]),
                "logit_0": float(logits[index, 0]),
                "logit_1": float(logits[index, 1]),
                "probability_0": float(probabilities[index, 0]),
                "probability_1": float(probabilities[index, 1]),
                "positive_probability": float(probabilities[index, int(context["protocol"]["positive_index"])]),
            }
        )
    return rows


def rows_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    logits = np.asarray([[float(row["logit_0"]), float(row["logit_1"])] for row in rows], dtype=np.float64)
    probabilities = np.asarray(
        [[float(row["probability_0"]), float(row["probability_1"])] for row in rows], dtype=np.float64
    )
    return truth, logits, probabilities


def metrics_match(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-10) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict) and isinstance(b, dict):
            if not metrics_match(a, b, tolerance):
                return False
        elif isinstance(a, list) and isinstance(b, list):
            if a != b:
                return False
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance):
                return False
        elif a != b:
            return False
    return True


def private_diagnostics(
    context: dict[str, Any],
    model: torch.nn.Module,
    test_mask: torch.Tensor,
    cumulative_gradient: dict[str, float],
    initial_adapter_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    require(hasattr(model, "private_adapters"), "Private diagnostics require B1")
    _, _, _, intermediates = infer(model, context["dataset_data"]["Feature"], return_intermediates=True)
    residuals = intermediates["private_residuals"][test_mask]
    shared = intermediates["modal_tokens_post_transformer"][test_mask]
    residual_norm = residuals.norm(dim=-1)
    shared_norm = shared.norm(dim=-1)
    ratio = residual_norm / shared_norm.clamp_min(1e-8)
    cosine = F.cosine_similarity(intermediates["Y"][test_mask], intermediates["G"][test_mask], dim=-1)
    modalities = [entry["name"] for entry in context["protocol"]["modalities"]]
    residual_by_modality = dict(zip(modalities, residual_norm.mean(dim=0).detach().cpu().tolist()))
    ratio_mean_by_modality = dict(zip(modalities, ratio.mean(dim=0).detach().cpu().tolist()))
    ratio_max_by_modality = dict(zip(modalities, ratio.amax(dim=0).detach().cpu().tolist()))
    parameter_delta: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("private_adapters."):
            require(name in initial_adapter_state, f"Initial adapter tensor missing: {name}")
            parameter_delta[name] = float((parameter.detach().cpu() - initial_adapter_state[name]).abs().max())
    private_trained = bool(
        cumulative_gradient
        and set(cumulative_gradient) == set(parameter_delta)
        and all(math.isfinite(value) and value > 0.0 for value in cumulative_gradient.values())
        and all(math.isfinite(value) for value in parameter_delta.values())
        and max(residual_by_modality.values(), default=0.0) >= 1e-6
    )
    return {
        "residual_mean_norm_by_modality": residual_by_modality,
        "private_shared_ratio_mean_by_modality": ratio_mean_by_modality,
        "private_shared_ratio_max_by_modality": ratio_max_by_modality,
        "private_shared_ratio_mean": float(ratio.mean().cpu()),
        "private_shared_ratio_max": float(ratio.max().cpu()),
        "category_global_cosine_mean": float(cosine.mean().cpu()),
        "adapter_cumulative_max_gradient": cumulative_gradient,
        "adapter_parameter_max_delta": parameter_delta,
        "private_trained": private_trained,
        "private_collapse": not private_trained,
    }


def fold_paths(context: dict[str, Any], arm: str, fold: int) -> dict[str, Path]:
    root = FORMAL_DIR / context["task_id"] / arm / f"fold_{fold:02d}"
    return {
        "root": root,
        "resume": root / "resume.pt",
        "checkpoint": root / "checkpoint_best.pt",
        "oof": root / "oof_predictions.csv",
        "epoch_metrics": root / "epoch_metrics.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
    }


def resume_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    lock: dict[str, Any],
    task_id: str,
    arm: str,
    fold: int,
    epoch: int,
    best: dict[str, Any] | None,
    cumulative_gradient: dict[str, float],
    initial_adapter_state: dict[str, torch.Tensor],
    epoch_history: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "lock": lock,
        "task_id": task_id,
        "arm": arm,
        "fold": int(fold),
        "epoch": int(epoch),
        "optimizer_steps": int(epoch),
        "scheduler_steps": int(epoch),
        "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
        "best": copy.deepcopy(best),
        "cumulative_gradient": dict(cumulative_gradient),
        "initial_adapter_state": {name: tensor.detach().cpu().clone() for name, tensor in initial_adapter_state.items()},
        "epoch_history": copy.deepcopy(epoch_history),
        "elapsed_seconds": float(elapsed_seconds),
    }


def load_resume(
    path: Path,
    expected_lock: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("schema") == 1 and payload.get("lock") == expected_lock, f"Resume lock changed: {path}")
    epoch = int(payload.get("epoch", -1))
    require(0 < epoch <= EPOCHS, f"Resume epoch invalid: {path}")
    require(payload.get("optimizer_steps") == epoch and payload.get("scheduler_steps") == epoch, "Resume step counters changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    require(int(scheduler.last_epoch) == epoch, "Resume scheduler epoch changed")
    if isinstance(scheduler, RatioPreservingCustomCosineAnnealingLR):
        scheduler.assert_ratio()
    restore_rng_state(payload["rng"])
    return payload


def train_fold(context: dict[str, Any], arm: str, fold: int, runtime: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context, arm, fold)
    lock = fold_lock(runtime, context, arm, fold)
    if paths["complete"].is_file():
        return load_completed_fold(context, arm, fold, runtime)
    paths["root"].mkdir(parents=True, exist_ok=True)
    known_targets = [path for name, path in paths.items() if name != "root"]
    known_incomplete = {path.name for path in known_targets} | {path.name + ".tmp" for path in known_targets}
    unknown = [path for path in paths["root"].iterdir() if path.name not in known_incomplete]
    require(not unknown, f"Incomplete fold has unrecognized artifacts: {paths['root']}")

    model, criterion, optimizer, scheduler, object_audit = make_training_objects(context, arm)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = fold_positions(context, fold)
    initial_adapter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    cumulative_gradient = {name: 0.0 for name in initial_adapter_state}
    best: dict[str, Any] | None = None
    epoch_history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before = 0.0
    resumed_from_epoch = 0
    if paths["resume"].is_file():
        resumed = load_resume(paths["resume"], lock, model, optimizer, scheduler)
        require(resumed["task_id"] == context["task_id"] and resumed["arm"] == arm and resumed["fold"] == fold, "Resume ownership changed")
        start_epoch = int(resumed["epoch"]) + 1
        resumed_from_epoch = int(resumed["epoch"])
        best = resumed["best"]
        cumulative_gradient = {name: float(value) for name, value in resumed["cumulative_gradient"].items()}
        initial_adapter_state = {name: tensor.detach().cpu().clone() for name, tensor in resumed["initial_adapter_state"].items()}
        epoch_history = copy.deepcopy(resumed["epoch_history"])
        require([int(row["epoch"]) for row in epoch_history] == list(range(1, resumed_from_epoch + 1)), "Resume epoch history is not contiguous")
        elapsed_before = float(resumed.get("elapsed_seconds", 0.0))

    session_started = time.perf_counter()
    grad_clip = float(context["protocol"]["training"]["grad_clip"])
    for epoch in range(start_epoch, EPOCHS + 1):
        loss, gradients, _ = train_update(model, criterion, optimizer, features, labels, train_mask, grad_clip)
        require(math.isfinite(loss), "Formal loss is non-finite")
        for name, value in gradients.items():
            cumulative_gradient[name] = max(cumulative_gradient.get(name, 0.0), float(value))

        # Historical checkpoint selection: probabilities are materialized first;
        # only then is held-out truth read to evaluate the locked ACC>AUC>F1 rule.
        full_logits = infer(model, features)[0]
        test_logits_t = full_logits[test_mask].detach().cpu()
        test_probabilities_t = torch.softmax(test_logits_t, dim=-1)
        require(bool(torch.isfinite(test_probabilities_t).all()), "Formal probability is non-finite")
        test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        test_logits = test_logits_t.numpy().astype(np.float64)
        test_probabilities = test_probabilities_t.numpy().astype(np.float64)
        metrics = binary_metrics(test_truth, test_probabilities, context["protocol"])
        epoch_history.append(
            {
                "epoch": int(epoch),
                "loss": float(loss),
                "correct": int(metrics["correct"]),
                "acc": float(metrics["acc"]),
                "roc_auc": float(metrics["roc_auc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "bacc": float(metrics["bacc"]),
                "pr_auc": float(metrics["pr_auc"]),
            }
        )
        if best is None or selection_key(metrics, epoch) > tuple(best["selection_key"]):
            best = {
                "epoch": int(epoch),
                "selection_key": list(selection_key(metrics, epoch)),
                "metrics": metrics,
                "model": clone_cpu_state(model),
                "test_logits": test_logits,
                "test_probabilities": test_probabilities,
                "test_truth": test_truth,
            }

        scheduler.step()
        if isinstance(scheduler, RatioPreservingCustomCosineAnnealingLR):
            scheduler.assert_ratio()
        elapsed = elapsed_before + (time.perf_counter() - session_started)
        atomic_torch_save(
            paths["resume"],
            resume_payload(
                model,
                optimizer,
                scheduler,
                lock,
                context["task_id"],
                arm,
                fold,
                epoch,
                best,
                cumulative_gradient,
                initial_adapter_state,
                epoch_history,
                elapsed,
            ),
        )

    require(best is not None, "No best epoch was selected")
    require([int(row["epoch"]) for row in epoch_history] == list(range(1, EPOCHS + 1)), "Formal epoch history is not exactly 1..400")
    require(int(scheduler.last_epoch) == EPOCHS, "Formal scheduler did not step exactly 400 times")
    model.load_state_dict(best["model"], strict=True)
    torch.cuda.synchronize(context["device"])
    inference_started = time.perf_counter()
    timed_logits = infer(model, features)[0]
    torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - inference_started
    timed_test_logits = timed_logits[test_mask].detach().cpu().numpy().astype(np.float64)
    require(float(np.max(np.abs(timed_test_logits - best["test_logits"]))) <= 1e-6, "Timed best-state replay changed logits")
    rows = prediction_rows(context, fold, best["test_logits"], best["test_probabilities"], best["test_truth"])
    for row in rows:
        row["arm"] = arm
    row_metrics = binary_metrics(*rows_arrays(rows)[::2], context["protocol"])
    require(metrics_match(row_metrics, best["metrics"]), "Best metrics changed during OOF materialization")
    diagnostics = None
    if arm == "B1":
        diagnostics = private_diagnostics(context, model, test_mask, cumulative_gradient, initial_adapter_state)

    elapsed_total = elapsed_before + (time.perf_counter() - session_started)
    checkpoint = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "arm": arm,
        "fold": int(fold),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "best_model": best["model"],
        "final_epoch": EPOCHS,
        "optimizer_steps": EPOCHS,
        "scheduler_steps": EPOCHS,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng_state(),
        "elapsed_seconds": float(elapsed_total),
    }
    atomic_torch_save(paths["checkpoint"], checkpoint)
    atomic_write_csv(paths["oof"], rows)
    atomic_write_csv(paths["epoch_metrics"], epoch_history)
    summary = {
        "schema": 1,
        "lock": lock,
        "task_id": context["task_id"],
        "arm": arm,
        "fold": int(fold),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "selection_uses_test_fold_truth": True,
        "probability_materialized_before_selection_truth_read": True,
        "no_additional_test_driven_search": True,
        "object_audit": object_audit,
        "diagnostics": diagnostics,
        "optimizer_steps": EPOCHS,
        "scheduler_steps": EPOCHS,
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds": float(elapsed_total),
        "inference_seconds_full_graph": float(inference_seconds),
        "inference_seconds_per_subject_proxy": float(inference_seconds / context["protocol"]["sample_count"]),
        "oof_count": len(rows),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "epoch_metrics_sha256": file_sha256(paths["epoch_metrics"]),
    }
    atomic_write_json(paths["summary"], summary)
    complete_core = {
        "complete": True,
        "lock": lock,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "epoch_metrics_sha256": file_sha256(paths["epoch_metrics"]),
        "summary_sha256": file_sha256(paths["summary"]),
    }
    atomic_write_json(paths["complete"], {**complete_core, "sha256": payload_sha256(complete_core)})
    print(f"[{context['task_id']} {arm} fold {fold}] best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    return load_completed_fold(context, arm, fold, runtime)


def typed_oof_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    for row in rows:
        typed.append(
            {
                "task_id": row["task_id"],
                "arm": row["arm"],
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
    return typed


def load_completed_fold(context: dict[str, Any], arm: str, fold: int, runtime: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = fold_paths(context, arm, fold)
    require(all(paths[name].is_file() for name in ("checkpoint", "oof", "epoch_metrics", "summary", "complete", "resume")), f"Completed fold artifacts missing: {paths['root']}")
    lock = fold_lock(runtime, context, arm, fold)
    marker = read_json(paths["complete"])
    marker_core = {key: value for key, value in marker.items() if key != "sha256"}
    require(marker.get("sha256") == payload_sha256(marker_core), "Complete marker digest changed")
    require(marker.get("complete") is True and marker.get("lock") == lock, "Complete marker lock changed")
    for key, name in (("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("epoch_metrics_sha256", "epoch_metrics"), ("summary_sha256", "summary")):
        require(marker[key] == file_sha256(paths[name]), f"Completed {name} changed")
    summary = read_json(paths["summary"])
    require(summary.get("lock") == lock and summary.get("task_id") == context["task_id"] and summary.get("arm") == arm and summary.get("fold") == fold, "Fold summary ownership changed")
    require(summary.get("optimizer_steps") == EPOCHS and summary.get("scheduler_steps") == EPOCHS, "Fold step counts changed")
    require(math.isfinite(float(summary.get("inference_seconds_full_graph", -1.0))) and float(summary["inference_seconds_full_graph"]) > 0.0, "Fold inference timing changed")
    rows = typed_oof_rows(read_csv_rows(paths["oof"]))
    expected = expected_fold_rows(context, fold)
    require(len(rows) == len(expected) == int(summary["oof_count"]), "Completed fold OOF count changed")
    for row, anchor in zip(rows, expected):
        require(row["task_id"] == context["task_id"] and row["arm"] == arm and row["fold"] == fold, "Completed OOF ownership changed")
        require(row["subject_id"] == anchor["subject_id"] and row["original_csv_index"] == anchor["original_csv_index"], "Completed OOF subject alignment changed")
        require(row["feature_sha256"] == anchor["feature_sha256"] and row["truth"] == anchor["truth"], "Completed OOF feature/truth alignment changed")
        require(row["prediction"] == int(np.argmax([row["probability_0"], row["probability_1"]])), "Completed OOF prediction changed")
    truth, saved_logits, saved_probabilities = rows_arrays(rows)
    recomputed = binary_metrics(truth, saved_probabilities, context["protocol"])
    require(metrics_match(recomputed, summary["best_metrics"]), "Completed OOF metrics changed")
    epoch_rows = read_csv_rows(paths["epoch_metrics"])
    require([int(row["epoch"]) for row in epoch_rows] == list(range(1, EPOCHS + 1)), "Completed epoch history is not exactly 1..400")
    require(all(math.isfinite(float(row["loss"])) for row in epoch_rows), "Completed epoch history contains invalid loss")
    replay_best = max(
        epoch_rows,
        key=lambda row: (float(row["acc"]), float(row["roc_auc"]), float(row["macro_f1"]), -int(row["epoch"])),
    )
    require(int(replay_best["epoch"]) == int(summary["best_epoch"]), "Completed history does not reproduce the locked best epoch")
    for key in ("correct", "acc", "roc_auc", "macro_f1", "bacc", "pr_auc"):
        require(math.isclose(float(replay_best[key]), float(summary["best_metrics"][key]), rel_tol=0.0, abs_tol=1e-10), f"Completed best metric changed: {key}")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint.get("schema") == 1 and checkpoint.get("lock") == lock, "Best checkpoint lock changed")
    require(checkpoint.get("task_id") == context["task_id"] and checkpoint.get("arm") == arm and checkpoint.get("fold") == fold, "Best checkpoint ownership changed")
    require(checkpoint.get("best_epoch") == summary["best_epoch"] and metrics_match(checkpoint["best_metrics"], summary["best_metrics"]), "Best checkpoint selection changed")
    require(checkpoint.get("optimizer_steps") == EPOCHS and checkpoint.get("scheduler_steps") == EPOCHS, "Best checkpoint step counts changed")
    model, _ = build_model(context, arm, SEED)
    model.load_state_dict(checkpoint["best_model"], strict=True)
    _, test_mask, _ = fold_positions(context, fold)
    replay_logits = infer(model, context["dataset_data"]["Feature"])[0][test_mask].detach().cpu().numpy().astype(np.float64)
    replay_probabilities = torch.softmax(torch.from_numpy(replay_logits), dim=-1).numpy()
    require(float(np.max(np.abs(replay_logits - saved_logits))) <= 1e-6, "Best checkpoint logits do not reproduce OOF")
    require(float(np.max(np.abs(replay_probabilities - saved_probabilities))) <= 1e-6, "Best checkpoint probabilities do not reproduce OOF")
    require(np.array_equal(replay_probabilities.argmax(axis=1), truth * 0 + np.asarray([row["prediction"] for row in rows])), "Best checkpoint predictions do not reproduce OOF")
    if arm == "B1":
        require(isinstance(summary.get("diagnostics", {}).get("private_trained"), bool), "Completed B1 private audit is missing")
    del model
    torch.cuda.empty_cache()
    return summary, rows


def aggregate_arm(
    context: dict[str, Any],
    arm: str,
    fold_summaries: list[dict[str, Any]],
    fold_rows: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(fold_summaries) == len(fold_rows) == len(FOLDS), "Formal fold count changed")
    rows = sorted([row for group in fold_rows for row in group], key=lambda row: int(row["original_csv_index"]))
    expected = sorted(
        [row for fold in context["fold_manifest"]["folds"] for row in fold["test_rows"]],
        key=lambda row: int(row["original_csv_index"]),
    )
    require(len(rows) == len(expected) == int(context["protocol"]["sample_count"]), "Formal OOF count changed")
    require(len({row["subject_id"] for row in rows}) == len(rows), "Formal OOF subject proxy is not unique")
    for row, anchor in zip(rows, expected):
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"], "Formal OOF alignment changed")
    truth, _, probabilities = rows_arrays(rows)
    metrics = binary_metrics(truth, probabilities, context["protocol"])
    fold_metrics = [
        {
            "fold": int(summary["fold"]),
            "best_epoch": int(summary["best_epoch"]),
            "correct": int(summary["best_metrics"]["correct"]),
            "acc": float(summary["best_metrics"]["acc"]),
            "roc_auc": float(summary["best_metrics"]["roc_auc"]),
            "macro_f1": float(summary["best_metrics"]["macro_f1"]),
            "bacc": float(summary["best_metrics"]["bacc"]),
            "pr_auc": float(summary["best_metrics"]["pr_auc"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "inference_seconds_full_graph": float(summary["inference_seconds_full_graph"]),
            "resumed_from_epoch": int(summary["resumed_from_epoch"]),
        }
        for summary in fold_summaries
    ]
    require([row["fold"] for row in fold_metrics] == list(FOLDS), "Formal fold summary order changed")
    accuracies = [row["acc"] for row in fold_metrics]
    output_path = RESULT_DIR / f"{context['task_id']}_{arm}_oof_predictions.csv"
    atomic_write_csv(output_path, rows)
    return (
        {
            "task_id": context["task_id"],
            "arm": arm,
            "metrics": metrics,
            "fold_acc_mean": float(statistics.mean(accuracies)),
            "fold_acc_sample_std": float(statistics.stdev(accuracies)),
            "fold_metrics": fold_metrics,
            "parameter_count": EXPECTED_PARAMETERS[context["task_id"]][arm],
            "training_time_seconds": float(sum(row["elapsed_seconds"] for row in fold_metrics)),
            "inference_time_full_graph_sum_seconds": float(sum(float(summary["inference_seconds_full_graph"]) for summary in fold_summaries)),
            "inference_time_full_graph_mean_seconds": float(statistics.mean(float(summary["inference_seconds_full_graph"]) for summary in fold_summaries)),
            "inference_time_per_subject_proxy_mean_seconds": float(statistics.mean(float(summary["inference_seconds_per_subject_proxy"]) for summary in fold_summaries)),
            "oof_path": output_path.relative_to(ROOT).as_posix(),
            "oof_sha256": file_sha256(output_path),
        },
        rows,
    )


def paired_comparison(
    context: dict[str, Any],
    b0_result: dict[str, Any],
    b1_result: dict[str, Any],
    b0_rows: list[dict[str, Any]],
    b1_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    left = {row["subject_id"]: row for row in b0_rows}
    right = {row["subject_id"]: row for row in b1_rows}
    require(set(left) == set(right), "Paired OOF subject sets changed")
    repairs = damages = changed = 0
    transition_counts: dict[str, int] = {}
    for subject_id in sorted(left):
        old, new = left[subject_id], right[subject_id]
        require(old["truth"] == new["truth"] and old["feature_sha256"] == new["feature_sha256"], "Paired OOF truth/feature changed")
        old_correct = old["prediction"] == old["truth"]
        new_correct = new["prediction"] == new["truth"]
        repairs += int(not old_correct and new_correct)
        damages += int(old_correct and not new_correct)
        changed += int(old["prediction"] != new["prediction"])
        if old["prediction"] != new["prediction"]:
            old_name = context["protocol"]["class_names"][old["prediction"]]
            new_name = context["protocol"]["class_names"][new["prediction"]]
            transition = f"{old_name}->{new_name}"
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
    metric_deltas = {
        key: float(b1_result["metrics"][key] - b0_result["metrics"][key])
        for key in ("acc", "bacc", "macro_f1", "roc_auc", "pr_auc", "weighted_f1", "sen", "spe")
    }
    truth_counts = np.bincount(np.asarray([int(row["truth"]) for row in b0_rows]), minlength=2)
    majority_index = int(np.argmax(truth_counts))
    b0_majority_predictions = int(sum(row["prediction"] == majority_index for row in b0_rows))
    b1_majority_predictions = int(sum(row["prediction"] == majority_index for row in b1_rows))
    return {
        "task_id": context["task_id"],
        "repairs": repairs,
        "damages": damages,
        "changed": changed,
        "net_correct": repairs - damages,
        "correct_delta": int(b1_result["metrics"]["correct"] - b0_result["metrics"]["correct"]),
        "metric_deltas": metric_deltas,
        "positive_tp_delta": int(b1_result["metrics"]["tp"] - b0_result["metrics"]["tp"]),
        "positive_fn_delta": int(b1_result["metrics"]["fn"] - b0_result["metrics"]["fn"]),
        "prediction_transitions": transition_counts,
        "majority_class": context["protocol"]["class_names"][majority_index],
        "true_majority_count": int(truth_counts[majority_index]),
        "b0_majority_prediction_count": b0_majority_predictions,
        "b1_majority_prediction_count": b1_majority_predictions,
        "b1_minus_b0_majority_prediction_count": b1_majority_predictions - b0_majority_predictions,
        "b1_majority_prediction_bias_vs_truth": b1_majority_predictions - int(truth_counts[majority_index]),
    }


def aggregate_private(context: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [summary["diagnostics"] for summary in summaries]
    require(len(diagnostics) == len(FOLDS) and all(item is not None for item in diagnostics), "B1 diagnostics missing")
    modalities = [entry["name"] for entry in context["protocol"]["modalities"]]
    residual = {
        modality: float(statistics.mean(item["residual_mean_norm_by_modality"][modality] for item in diagnostics))
        for modality in modalities
    }
    ratio_mean = {
        modality: float(statistics.mean(item["private_shared_ratio_mean_by_modality"][modality] for item in diagnostics))
        for modality in modalities
    }
    ratio_max = {
        modality: float(max(item["private_shared_ratio_max_by_modality"][modality] for item in diagnostics))
        for modality in modalities
    }
    gradient_names = set(diagnostics[0]["adapter_cumulative_max_gradient"])
    require(all(set(item["adapter_cumulative_max_gradient"]) == gradient_names for item in diagnostics), "Adapter gradient schema changed across folds")
    gradient_max = {
        name: float(max(item["adapter_cumulative_max_gradient"][name] for item in diagnostics))
        for name in sorted(gradient_names)
    }
    delta_max = {
        name: float(max(item["adapter_parameter_max_delta"][name] for item in diagnostics))
        for name in sorted(gradient_names)
    }
    return {
        "task_id": context["task_id"],
        "residual_mean_norm_by_modality": residual,
        "private_shared_ratio_mean_by_modality": ratio_mean,
        "private_shared_ratio_max_by_modality": ratio_max,
        "private_shared_ratio_mean": float(statistics.mean(item["private_shared_ratio_mean"] for item in diagnostics)),
        "private_shared_ratio_max": float(max(item["private_shared_ratio_max"] for item in diagnostics)),
        "category_global_cosine_mean": float(statistics.mean(item["category_global_cosine_mean"] for item in diagnostics)),
        "adapter_max_gradient_by_tensor": gradient_max,
        "adapter_max_parameter_delta_by_tensor": delta_max,
        "private_trained_all_folds": bool(all(item["private_trained"] for item in diagnostics)),
        "private_collapse_any_fold": bool(any(item["private_collapse"] for item in diagnostics)),
        "adapter_rank": ADAPTER_RANK,
        "parameter_count_b0": EXPECTED_PARAMETERS[context["task_id"]]["B0"],
        "parameter_count_b1": EXPECTED_PARAMETERS[context["task_id"]]["B1"],
        "parameter_delta": EXPECTED_PARAMETERS[context["task_id"]]["delta"],
    }


def final_decision(
    task_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    mechanisms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    deltas = {task: comparisons[task]["metric_deltas"] for task in TASK_IDS}
    improved = [task for task in TASK_IDS if deltas[task]["bacc"] > 0.0 or deltas[task]["macro_f1"] > 0.0]
    marked_decline = [task for task in TASK_IDS if deltas[task]["bacc"] < -0.005 or deltas[task]["macro_f1"] < -0.005]
    tad0 = task_results["tadpole_smci_pmci"]["B0"]["metrics"]
    tad1 = task_results["tadpole_smci_pmci"]["B1"]["metrics"]
    tad_added_fn = int(tad1["fn"] - tad0["fn"])
    tad_sen_delta = float(tad1["sen"] - tad0["sen"])
    tad_positive_count = int(tad1["predicted_counts"]["PMCI"])
    tad_collapse = bool(tad_positive_count <= 4 or tad1["sen"] <= 0.10)
    tad_sensitivity_marked_decline = bool(tad_added_fn >= 2 or tad_sen_delta < -0.05)
    imbalance: dict[str, bool] = {}
    for task in ("abide_ads_cn", "abide5_ads_cn"):
        metrics = task_results[task]["B1"]["metrics"]
        n = int(metrics["n"])
        counts = list(metrics["predicted_counts"].values())
        imbalance[task] = bool(
            min(metrics["sen"], metrics["spe"]) < 0.50
            or abs(metrics["sen"] - metrics["spe"]) > 0.35
            or any(count < 0.10 * n for count in counts)
        )
    private_not_trained = [task for task in TASK_IDS if not mechanisms[task]["private_trained_all_folds"]]
    protocol_consistent = True
    any_collapse = tad_collapse or any(imbalance.values())
    stop_reasons: list[str] = []
    if len(marked_decline) >= 2:
        stop_reasons.append("at least two tasks have BACC or Macro-F1 delta below -0.005")
    if tad_sensitivity_marked_decline:
        stop_reasons.append("TADPOLE pMCI sensitivity materially declined")
    if any_collapse:
        stop_reasons.append("a task met the preregistered collapse/imbalance condition")
    if private_not_trained:
        stop_reasons.append("private adapters did not train on every fold")
    if not protocol_consistent:
        stop_reasons.append("protocol inconsistency")

    all_safe = all(deltas[task]["bacc"] >= -0.005 and deltas[task]["macro_f1"] >= -0.005 for task in TASK_IDS)
    go = bool(
        all_safe
        and len(improved) >= 2
        and tad1["tp"] >= tad0["tp"]
        and not tad_collapse
        and not any(imbalance.values())
        and not private_not_trained
    )
    near = bool(
        len(improved) >= 2
        and all_safe
        and tad_added_fn <= 1
        and not any_collapse
        and not private_not_trained
    )
    if stop_reasons:
        decision = "CROSS_DATASET_STOP"
    elif go:
        decision = "CROSS_DATASET_GO"
    elif near:
        decision = "CROSS_DATASET_NEAR"
    else:
        decision = "CROSS_DATASET_MIXED"
    return {
        "decision": decision,
        "priority": "STOP > GO > NEAR > MIXED",
        "improved_tasks": improved,
        "marked_decline_tasks": marked_decline,
        "all_task_bacc_and_macro_f1_deltas_at_least_minus_0_005": all_safe,
        "tadpole_pmci_tp_b0": int(tad0["tp"]),
        "tadpole_pmci_tp_b1": int(tad1["tp"]),
        "tadpole_added_fn": tad_added_fn,
        "tadpole_sensitivity_delta": tad_sen_delta,
        "tadpole_predicted_pmci_b1": tad_positive_count,
        "tadpole_collapse": tad_collapse,
        "abide_imbalance": imbalance,
        "private_not_trained_tasks": private_not_trained,
        "protocol_consistent": protocol_consistent,
        "stop_reasons": stop_reasons,
        "next_recommendation": "SP-LRIF-A012: sum-preserving low-rank branch interaction fusion" if decision == "CROSS_DATASET_GO" else None,
    }


def json_with_digest(path: Path, core: dict[str, Any]) -> dict[str, Any]:
    payload = {**core, "sha256": payload_sha256(core)}
    atomic_write_json(path, payload)
    return payload


def render_report(
    runtime: dict[str, Any],
    task_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    mechanisms: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    total_seconds: float,
) -> str:
    protocols = load_protocols()
    display = {
        "tadpole_smci_pmci": "TADPOLE SMCI_PMCI",
        "abide_ads_cn": "ABIDE ADS_CN",
        "abide5_ads_cn": "ABIDE-5 ADS_CN",
    }
    lines = [
        "# A012/C1 Cross-Dataset Structural Generalization v1",
        "",
        f"- Source commit: `{runtime['source_commit']}`",
        "- Result commit: reported in the final Git handoff after this report and its result files are committed (self-referential SHA is intentionally not embedded)",
        f"- Branch: `{BRANCH}`",
        "- Device: `cuda:0`",
        f"- Decision: **{decision['decision']}**",
        f"- Formal wall time: {total_seconds:.1f} seconds",
        "- B0 is fully rerun Original; B1 is the rank-8 A012 structural branch with private-adapter LR=2x.",
        "- All results are single-model seed-0, ten-fold, 400-epoch OOF evaluations; no ensemble is used.",
        "",
        "## Fixed protocol disclosure",
        "",
        "The paired arms preserve the last formal historical protocol: offline preprocessed inputs, global full-dataset class weights, and held-out-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch. Probabilities are materialized before the held-out truth is read for that fixed selection. No test-driven hyperparameter, feature, threshold, calibration, or model search is performed.",
        "",
        "The ABIDE task name `ADS_CN` retains the repository's historical `ADS` spelling. Here ADS is the ASD/autism-spectrum class: raw label 1 maps to class index 0 and is positive; raw label 2 maps to CN at class index 1. No external subject identifier is claimed. ABIDE and ABIDE-5 share 64 feature columns and 841 feature-identical rows, but are trained and reported separately.",
        "",
        "The historical three-seed NPZ artifacts are provenance only. They are not reused because they do not lock a training source commit or durable external IDs and do not match this one-seed protocol.",
        "",
        "## Dataset and formal protocol",
        "",
        "| Task | N / class counts (index order) | Modalities | LR | WD | Dropout | Modal transformer | DIFFormer | Loss | Scheduler |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for task in TASK_IDS:
        protocol = protocols[task]
        training = protocol["training"]
        counts = ", ".join(f"{name}={protocol['class_counts'][name]}" for name in protocol["class_names"])
        difformer = f"L{training['graph_layers']}/H{training['graph_heads']}, alpha={training['graph_alpha']}, beta={training['graph_beta']}, {training['graph_kernel']}, graph={training['graph_use_graph']}"
        modal = f"L{training['modal_layers']}/H{training['modal_heads']}, hidden={training['hidden_size']}, noise={training['input_noise_std']}, drop_path={training['drop_path']}"
        loss = f"global weighted CE(ls={training['label_smoothing']}) + 2 weighted OVR CE sum + orth={training['loss_rate']}"
        scheduler = f"CustomCosineAnnealingLR(T_max={training['scheduler_t_max']}, eta_min={training['scheduler_eta_min']}); B1 adapter LR=2x ratio-preserved"
        lines.append(f"| {display[task]} | {protocol['sample_count']} / {counts} | {len(protocol['modalities'])} | {training['lr']} | {training['weight_decay']} | {training['drop_rate']} | {modal} | {difformer} | {loss} | {scheduler} |")
    lines.extend([
        "",
        "All tasks use their locked offline processed inputs, global historical class weights, full-batch transductive folds 0..9, seed 0, 400 epochs, Adam, grad clip 1, EMA off, graph disabled, tau 0, and one fresh model/criterion/optimizer/scheduler per fold. ABIDE/ABIDE-5 config fields that declared auxiliary weight 0.2/mean were ineffective in the historical main; the actual two OVR losses are summed at coefficient 1 and are preserved here.",
        "",
        "## Formal results",
        "",
        "| Task | Arm | Correct/N | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | Params | Fold ACC mean +/- SD | Train sec | Mean full-graph inference sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task in TASK_IDS:
        for arm in ARMS:
            result = task_results[task][arm]
            m = result["metrics"]
            lines.append(
                f"| {display[task]} | {arm} | {m['correct']}/{m['n']} | {m['acc']:.7f} | {m['roc_auc']:.7f} | {m['pr_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {m['weighted_f1']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {result['parameter_count']} | {result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f} | {result['training_time_seconds']:.1f} | {result['inference_time_full_graph_mean_seconds']:.6f} |"
            )
    lines.extend(["", "### Confusion matrices and prediction counts", ""])
    for task in TASK_IDS:
        for arm in ARMS:
            m = task_results[task][arm]["metrics"]
            lines.append(f"- {display[task]} {arm}: class order={m['confusion_matrix_class_order']}; confusion={m['confusion_matrix']}; predicted={m['predicted_counts']}; TP/FN/TN/FP={m['tp']}/{m['fn']}/{m['tn']}/{m['fp']}; SEN is positive-class sensitivity and SPE is negative-class specificity.")
    lines.extend(["", "## Paired B1 minus B0 changes", "", "| Task | Correct delta | ACC delta | AUC delta | F1 delta | BACC delta | Weighted-F1 delta | SEN delta | SPE delta | PR-AUC delta | Repairs | Damages | Changed | Majority-pred delta |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for task in TASK_IDS:
        comparison = comparisons[task]
        delta = comparison["metric_deltas"]
        lines.append(
            f"| {display[task]} | {comparison['correct_delta']:+d} | {delta['acc']:+.7f} | {delta['roc_auc']:+.7f} | {delta['macro_f1']:+.7f} | {delta['bacc']:+.7f} | {delta['weighted_f1']:+.7f} | {delta['sen']:+.7f} | {delta['spe']:+.7f} | {delta['pr_auc']:+.7f} | {comparison['repairs']} | {comparison['damages']} | {comparison['changed']} | {comparison['b1_minus_b0_majority_prediction_count']:+d} |"
        )

    tad0 = task_results["tadpole_smci_pmci"]["B0"]["metrics"]
    tad1 = task_results["tadpole_smci_pmci"]["B1"]["metrics"]
    lines.extend(
        [
            "",
            "## TADPOLE minority-class priority view",
            "",
            "| Arm | TP | FN | TN | FP | BACC | Macro-F1 | pMCI SEN | PR-AUC | ROC-AUC | ACC | Predicted pMCI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| B0 | {tad0['tp']} | {tad0['fn']} | {tad0['tn']} | {tad0['fp']} | {tad0['bacc']:.7f} | {tad0['macro_f1']:.7f} | {tad0['sen']:.7f} | {tad0['pr_auc']:.7f} | {tad0['roc_auc']:.7f} | {tad0['acc']:.7f} | {tad0['predicted_counts']['PMCI']} |",
            f"| B1 | {tad1['tp']} | {tad1['fn']} | {tad1['tn']} | {tad1['fp']} | {tad1['bacc']:.7f} | {tad1['macro_f1']:.7f} | {tad1['sen']:.7f} | {tad1['pr_auc']:.7f} | {tad1['roc_auc']:.7f} | {tad1['acc']:.7f} | {tad1['predicted_counts']['PMCI']} |",
            "",
            "## Fold selections",
            "",
        ]
    )
    for task in TASK_IDS:
        lines.extend([f"### {display[task]}", "", "| Fold | B0 best epoch | B0 Correct | B0 ACC | B1 best epoch | B1 Correct | B1 ACC |", "|---:|---:|---:|---:|---:|---:|---:|"])
        b0_folds = task_results[task]["B0"]["fold_metrics"]
        b1_folds = task_results[task]["B1"]["fold_metrics"]
        for left, right in zip(b0_folds, b1_folds):
            lines.append(f"| {left['fold']} | {left['best_epoch']} | {left['correct']} | {left['acc']:.7f} | {right['best_epoch']} | {right['correct']} | {right['acc']:.7f} |")
        lines.append("")

    lines.extend(["## B1 mechanism audit", "", "| Task | Params B0 -> B1 | Private/shared mean | Private/shared max | Category-Global cosine | Private trained all folds |", "|---|---:|---:|---:|---:|---:|"])
    for task in TASK_IDS:
        mechanism = mechanisms[task]
        lines.append(
            f"| {display[task]} | {mechanism['parameter_count_b0']} -> {mechanism['parameter_count_b1']} (+{mechanism['parameter_delta']}) | {mechanism['private_shared_ratio_mean']:.7f} | {mechanism['private_shared_ratio_max']:.7f} | {mechanism['category_global_cosine_mean']:.7f} | {mechanism['private_trained_all_folds']} |"
        )
    for task in TASK_IDS:
        mechanism = mechanisms[task]
        lines.extend(["", f"### {display[task]} per-modality private residual", "", "| Modality | Residual norm mean | Private/shared ratio mean | Private/shared ratio max |", "|---|---:|---:|---:|"])
        for modality, residual in mechanism["residual_mean_norm_by_modality"].items():
            lines.append(f"| {modality} | {residual:.7f} | {mechanism['private_shared_ratio_mean_by_modality'][modality]:.7f} | {mechanism['private_shared_ratio_max_by_modality'][modality]:.7f} |")
        lines.append("")
        lines.append(f"- Per-adapter-tensor maximum gradients: `{json.dumps(mechanism['adapter_max_gradient_by_tensor'], sort_keys=True)}`")
        lines.append(f"- Private collapse on any fold: `{mechanism['private_collapse_any_fold']}`")
    lines.extend(
        [
            "",
            "## Decision audit",
            "",
            f"- Improved tasks (strict BACC>0 or Macro-F1>0): {', '.join(decision['improved_tasks']) or 'none'}",
            f"- Marked-decline tasks (BACC or Macro-F1 delta < -0.005): {', '.join(decision['marked_decline_tasks']) or 'none'}",
            f"- TADPOLE pMCI TP: {decision['tadpole_pmci_tp_b0']} -> {decision['tadpole_pmci_tp_b1']}; added FN={decision['tadpole_added_fn']}; predicted pMCI={decision['tadpole_predicted_pmci_b1']}",
            f"- ABIDE imbalance flags: {json.dumps(decision['abide_imbalance'], sort_keys=True)}",
            f"- Stop reasons: {('; '.join(decision['stop_reasons'])) or 'none'}",
            f"- Final decision: **{decision['decision']}**",
        ]
    )
    if decision["next_recommendation"]:
        lines.append(f"- Recorded next recommendation (not run): `{decision['next_recommendation']}`")
    task_answers = []
    for task in TASK_IDS:
        comp = comparisons[task]
        delta = comp["metric_deltas"]
        task_answers.append(
            f"{display[task]}: Correct {comp['correct_delta']:+d}, BACC {delta['bacc']:+.7f}, Macro-F1 {delta['macro_f1']:+.7f}, ROC-AUC {delta['roc_auc']:+.7f}"
        )
    hard_gain = any(comparisons[task]["correct_delta"] > 0 for task in TASK_IDS)
    bias_text = "; ".join(
        f"{display[task]} majority-pred delta={comparisons[task]['b1_minus_b0_majority_prediction_count']:+d}, bias-vs-truth={comparisons[task]['b1_majority_prediction_bias_vs_truth']:+d}"
        for task in TASK_IDS
    )
    active_modalities = "; ".join(
        f"{display[task]}=" + ",".join(modality for modality, value in mechanisms[task]["residual_mean_norm_by_modality"].items() if value >= 1e-6)
        for task in TASK_IDS
    )
    parameter_text = "; ".join(
        f"{display[task]} +{mechanisms[task]['parameter_delta']} ({100.0 * mechanisms[task]['parameter_delta'] / mechanisms[task]['parameter_count_b0']:.3f}%)"
        for task in TASK_IDS
    )
    mechanism_stable = all(mechanisms[task]["private_trained_all_folds"] and not mechanisms[task]["private_collapse_any_fold"] for task in TASK_IDS)
    continue_fusion = decision["decision"] == "CROSS_DATASET_GO"
    if continue_fusion:
        diagnosis = "The preregistered cross-task gate passed; no instability diagnosis is needed."
    elif mechanism_stable:
        diagnosis = "Private residuals trained and activated on every fold, so a failure to pass GO is more consistent with dataset/task-specific utility than with a disabled private mechanism."
    else:
        diagnosis = "At least one private branch failed its training/activation gate, so mechanism instability is the primary issue."
    lines.extend(
        [
            "",
            "## Answers to the ten required questions",
            "",
            f"1. TADPOLE SMCI_PMCI versus Original: {task_answers[0]}.",
            f"2. ABIDE versus Original: {task_answers[1]}.",
            f"3. ABIDE-5 versus Original: {task_answers[2]}.",
            f"4. Hard-classification gain rather than AUC-only gain: `{hard_gain}`. Repairs/damages are " + "; ".join(f"{display[task]} {comparisons[task]['repairs']}/{comparisons[task]['damages']}" for task in TASK_IDS) + ".",
            f"5. Majority-class bias audit: {bias_text}. Preregistered ABIDE imbalance flags={json.dumps(decision['abide_imbalance'], sort_keys=True)}, TAD collapse={decision['tadpole_collapse']}.",
            f"6. Modalities with activated private residual norm >=1e-6: {active_modalities}. Every adapter tensor's maximum gradient is listed above and in mechanism_diagnostics.json.",
            f"7. Parameter increases: {parameter_text}.",
            f"8. All three tasks satisfy CROSS_DATASET_GO: `{decision['decision'] == 'CROSS_DATASET_GO'}`; final decision is `{decision['decision']}`.",
            f"9. Continue replacing direct Category + Global addition: `{continue_fusion}`. Only a GO records SP-LRIF-A012, and this run does not implement it.",
            f"10. If fusion work is not justified: {diagnosis}",
            "",
            "## Reproduction",
            "",
            "```text",
            "python -u -B scripts/run_cross_dataset_a012_structure_v1.py inspect",
            "python -u -B scripts/run_cross_dataset_a012_structure_v1.py smoke --device cuda:0",
            "python -u -B scripts/run_cross_dataset_a012_structure_v1.py formal --device cuda:0",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_formal(device_text: str, historical_root: Path) -> None:
    require(device_text == "cuda:0", "Formal requires cuda:0")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    started = time.perf_counter()
    source_commit = source_gate(historical_root)
    protocols = load_protocols()
    runtime = runtime_lock(protocols, source_commit)
    smoke = validate_smoke(runtime)
    formal_config_core = {
        "runtime_lock": runtime,
        "smoke_report_sha256": file_sha256(SMOKE_DIR / "smoke_report.json"),
        "smoke_payload_sha256": smoke["sha256"],
        "order": "task -> fold -> B0 then B1",
        "reproduction_command": "python -u -B scripts/run_cross_dataset_a012_structure_v1.py formal --device cuda:0",
    }
    formal_config = {**formal_config_core, "sha256": payload_sha256(formal_config_core)}
    formal_config_path = RESULT_DIR / "formal_config.json"
    if formal_config_path.is_file():
        require(read_json(formal_config_path) == formal_config, "Formal config changed on resume")
    else:
        atomic_write_json(formal_config_path, formal_config)

    task_results: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    mechanisms: dict[str, dict[str, Any]] = {}
    fold_metric_files: dict[str, str] = {}
    for task_id in TASK_IDS:
        context = build_context(protocols[task_id], torch.device(device_text))
        summaries: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        rows_by_arm: dict[str, list[list[dict[str, Any]]]] = {arm: [] for arm in ARMS}
        for fold in FOLDS:
            for arm in ARMS:
                summary, rows = train_fold(context, arm, fold, runtime)
                summaries[arm].append(summary)
                rows_by_arm[arm].append(rows)
        task_results[task_id] = {}
        aggregate_rows: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            result, rows = aggregate_arm(context, arm, summaries[arm], rows_by_arm[arm])
            task_results[task_id][arm] = result
            aggregate_rows[arm] = rows
        comparisons[task_id] = paired_comparison(
            context,
            task_results[task_id]["B0"],
            task_results[task_id]["B1"],
            aggregate_rows["B0"],
            aggregate_rows["B1"],
        )
        mechanisms[task_id] = aggregate_private(context, summaries["B1"])
        fold_core = {
            "task_id": task_id,
            "B0": task_results[task_id]["B0"]["fold_metrics"],
            "B1": task_results[task_id]["B1"]["fold_metrics"],
        }
        fold_path = RESULT_DIR / f"{task_id}_fold_metrics.json"
        json_with_digest(fold_path, fold_core)
        fold_metric_files[task_id] = fold_path.relative_to(ROOT).as_posix()
        del context
        torch.cuda.empty_cache()

    decision = final_decision(task_results, comparisons, mechanisms)
    comparisons_payload = json_with_digest(RESULT_DIR / "comparisons.json", {"tasks": comparisons})
    mechanisms_payload = json_with_digest(RESULT_DIR / "mechanism_diagnostics.json", {"tasks": mechanisms})
    elapsed = time.perf_counter() - started
    reproduction_commands = (
        "python -u -B scripts/run_cross_dataset_a012_structure_v1.py inspect\n"
        "python -u -B scripts/run_cross_dataset_a012_structure_v1.py smoke --device cuda:0\n"
        "python -u -B scripts/run_cross_dataset_a012_structure_v1.py formal --device cuda:0\n"
    )
    atomic_write_text(RESULT_DIR / "reproduction_commands.txt", reproduction_commands)
    report = render_report(runtime, task_results, comparisons, mechanisms, decision, elapsed)
    atomic_write_text(RESULT_DIR / "REPORT.md", report)
    summary_core = {
        "experiment": EXPERIMENT_ID,
        "runtime_lock": runtime,
        "formal_config_sha256": file_sha256(formal_config_path),
        "task_results": task_results,
        "comparisons": comparisons,
        "mechanism_diagnostics": mechanisms,
        "decision": decision,
        "failure_count": 0,
        "formal_wall_time_seconds": float(elapsed),
        "fold_metric_files": fold_metric_files,
        "comparisons_payload_sha256": comparisons_payload["sha256"],
        "mechanisms_payload_sha256": mechanisms_payload["sha256"],
        "report_sha256": file_sha256(RESULT_DIR / "REPORT.md"),
        "reproduction_commands_sha256": file_sha256(RESULT_DIR / "reproduction_commands.txt"),
    }
    final_summary = json_with_digest(RESULT_DIR / "summary.json", summary_core)
    print(json.dumps({"decision": decision, "tasks": {task: {arm: task_results[task][arm]["metrics"] for arm in ARMS} for task in TASK_IDS}, "summary_sha256": final_summary["sha256"]}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Freeze input/protocol/fold provenance without training")
    inspect_parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT), help="Root containing locked legacy RESULT evidence")
    smoke_parser = subparsers.add_parser("smoke", help="Run the three-epoch CUDA mechanism smoke")
    smoke_parser.add_argument("--device", default="cuda:0")
    smoke_parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    formal_parser = subparsers.add_parser("formal", help="Run or strictly resume all paired formal folds")
    formal_parser.add_argument("--device", default="cuda:0")
    formal_parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_root = Path(args.historical_root).resolve()
    if args.mode == "inspect":
        run_inspect(historical_root)
    elif args.mode == "smoke":
        run_smoke(args.device, historical_root)
    elif args.mode == "formal":
        run_formal(args.device, historical_root)
    else:
        raise InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
