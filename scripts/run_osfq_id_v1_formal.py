from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_osfq_id_v1 as smoke
import run_query_free_multibranch as query


SCHEMA_VERSION = "ovr-aligned-shared-query-pool-v1-formal"
CHECKPOINT_SCHEMA = "osfq-id-v1-formal-checkpoint-v1"
MODEL_NAME = "OVR-Aligned Shared-Query Pool v1"
CONFIG_REL = smoke.CONFIG_REL
FORMAL_ROOT_REL = Path("experiments/ovr_aligned_shared_query_v1/formal")
REPORT_REL = Path("reports/osfq_id_v1_fold0_formal_20260805.txt")
REFERENCE_BASELINE_MANIFEST = (
    PRIMARY_ROOT / "experiments/query_baseline_gpu_10fold_seed0/protocol_manifest.json"
)
REFERENCE_FOLD_MANIFEST = (
    PRIMARY_ROOT
    / "experiments/boundary_selective_fixed_mapping_10fold_seed0_v2/formal/fold_manifest.json"
)
REFERENCE_BASELINE_FOLD0_SUMMARY = (
    PRIMARY_ROOT / "experiments/query_baseline_gpu_10fold_seed0/fold_00/summary.json"
)
EXPECTED_BASELINE_MANIFEST_FILE_HASH = (
    "78114ce1b9cdff40d2c00c9546cb92b79d446d6212b19c0567c989c43a17327d"
)
EXPECTED_BASELINE_MANIFEST_CANONICAL_HASH = (
    "930288891ead11b143c3485424c5f141cbcd69a3ec7e4a5d72bd7d8076398926"
)
EXPECTED_FOLD_MANIFEST_FILE_HASH = (
    "0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106"
)
EXPECTED_FOLD_MANIFEST_CANONICAL_HASH = (
    "be020ede2a03d59dbd5e8bbf715cb4d097f8398f9824c122521c3240e00b6515"
)
EXPECTED_FOLD_ASSIGNMENT_HASH = (
    "8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9"
)
EXPECTED_FEATURE_HASH = (
    "2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042"
)
EXPECTED_MODAL_DICT_HASH = (
    "5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273"
)
EXPECTED_BASELINE_FOLD0_SUMMARY_HASH = (
    "97adf77295f9baa53f02a1260c050020e09437df71f7ac12c92c113627a932cd"
)
FOLD = 0
SEED = 0
EPOCHS = 400
EMA_FIRST_UPDATE_EPOCH = 20
ANCHOR_FIRST_ACTIVE_EPOCH = 21
EMA_DECAY = smoke.EMA_DECAY
ARMS = smoke.ARMS
ARM_ORDER = ("s0", "s1", "s2")
CLASS_ORDER = smoke.CLASS_ORDER
SOURCE_PATHS = tuple(smoke.SOURCE_PATHS) + (Path("scripts/run_osfq_id_v1_formal.py"),)
REQUIRED_ARM_ARTIFACTS = (
    "summary.json",
    "protocol_manifest.json",
    "config.ini",
    "initialization_audit.json",
    "epoch_metrics.csv",
    "best_predictions.csv",
    "final_predictions.csv",
    "best_confusion_matrix.csv",
    "final_confusion_matrix.csv",
    "auxiliary_metrics.json",
    "query_pool_diagnostics.json",
    "anchor_diagnostics.json",
    "warmup_boundary_audit.json",
    "checkpoint_best.pt",
    "checkpoint_epoch400.pt",
    "checkpoint_audit.json",
    "structure_audit_report.txt",
    "artifact_manifest.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def recursive_state_hash(value) -> str:
    digest = hashlib.sha256()

    def update(item) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda current: repr(current)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file(), f"Missing formal source: {relative.as_posix()}")
        result[relative.as_posix()] = query.file_hash(path)
    return result


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


def load_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verified_canonical_json(path: Path, expected_file_hash: str, expected_canonical: str) -> dict:
    observed_file_hash = query.file_hash(path)
    require(observed_file_hash == expected_file_hash, f"File hash changed: {path}")
    payload = load_json(path)
    recorded = payload.get("canonical_sha256")
    body = deepcopy(payload)
    body.pop("canonical_sha256", None)
    computed = canonical_hash(body)
    require(recorded == expected_canonical, f"Recorded canonical hash changed: {path}")
    require(computed == expected_canonical, f"Canonical content hash changed: {path}")
    return payload


def validate_historical_protocol(manifest: dict, config) -> dict:
    protocol = manifest["protocol"]
    gates = {
        "dataset": protocol["dataset"] == config.DATA_SET == "TADPOLE",
        "task": protocol["task"] == config.Task == "AD_CN_SMCI",
        "fold0_present": FOLD in protocol["folds"],
        "seed": int(protocol["fresh_seed_per_fold"]) == SEED,
        "epochs": int(protocol["epochs_per_fold"]) == EPOCHS,
        "single_model": protocol["single_model"] is True,
        "no_ensemble": protocol["ensemble"] is False,
        "no_search": protocol["hyperparameter_search"] is False,
        "cuda": protocol["device"] == "cuda:0",
        "transductive": protocol["transductive_full_batch"] is True,
        "semantic_branch": protocol["semantic_branch"] == "both",
        "semantic_fusion": protocol["semantic_fusion"] == "add",
        "category_fusion": protocol["category_branch_fusion"] == "concat",
        "adj_none": protocol["adj_mode"] == "none",
        "graph_disabled": protocol["graph_use_graph"] is False,
        "optimizer": protocol["optimizer"] == "Adam",
        "lr": float(protocol["learning_rate"]) == float(config.lr),
        "weight_decay": float(protocol["weight_decay"]) == float(config.weight_decay),
        "scheduler": protocol["scheduler"] == "CustomCosineAnnealingLR",
        "scheduler_t_max": int(protocol["scheduler_t_max"]) == int(config.T_max) == EPOCHS,
        "scheduler_eta_min": float(protocol["scheduler_eta_min"]) == float(config.Lr_Min),
        "logit_adjust_tau": float(protocol["logit_adjust_tau"]) == float(config.logit_adjust_tau),
        "best_epoch_rule": protocol["best_epoch_rule"] == ["ACC", "Macro-AUC", "Macro-F1"],
        "ovr_loss": protocol["loss"] == "main weighted CE + three original one-vs-rest auxiliary CE",
        "label_smoothing": float(protocol["label_smoothing"]) == 0.05,
        "no_orthogonality": protocol["orthogonality_loss"] is False,
    }
    require(all(gates.values()), f"Historical protocol mismatch: {gates}")
    return {"gates": gates, "passed": True}


def generated_fold_identity(dataset_dict: dict, dataset_data: dict, reference: dict) -> dict:
    sample_count = int(dataset_data["Feature"].size(0))
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
    labels = dataset_data["Label"].detach().cpu().numpy().astype(np.int64)
    require(sample_count == int(reference["sample_count"]) == 598, "Sample count changed")
    require(raw_array_hash(source_indices) == reference["sample_order_sha256"], "Sample order changed")
    require(raw_array_hash(labels) == reference["label_order_sha256"], "Label order changed")
    assignment = np.full(sample_count, -1, dtype=np.int64)
    fold_checks = []
    for fold, reference_fold in enumerate(reference["folds"]):
        train_mask_t, test_mask_t = dataset_data["Mask"][fold]
        train_mask = train_mask_t.detach().cpu().numpy().astype(bool)
        test_mask = test_mask_t.detach().cpu().numpy().astype(bool)
        train_positions = np.flatnonzero(train_mask).astype(np.int64)
        test_positions = np.flatnonzero(test_mask).astype(np.int64)
        assignment[test_positions] = fold
        observed = {
            "fold": fold,
            "train_size": int(train_positions.size),
            "test_size": int(test_positions.size),
            "train_positions": train_positions.tolist(),
            "test_positions": test_positions.tolist(),
            "train_indices": source_indices[train_positions].tolist(),
            "test_indices": source_indices[test_positions].tolist(),
            "split_hash": query.split_hash(source_indices, train_mask_t, test_mask_t),
            "train_mask_sha256": raw_array_hash(train_mask.astype(np.uint8)),
            "test_mask_sha256": raw_array_hash(test_mask.astype(np.uint8)),
        }
        keys = tuple(observed)
        gates = {key: observed[key] == reference_fold[key] for key in keys}
        gates["disjoint"] = not bool(np.any(train_mask & test_mask))
        gates["complementary"] = bool(np.array_equal(test_mask, ~train_mask))
        require(all(gates.values()), f"Fold {fold} identity mismatch")
        fold_checks.append({"fold": fold, "gates": gates, "passed": True})
    assignment_hash = raw_array_hash(assignment.astype(np.uint8))
    require(assignment_hash == reference["test_fold_assignment_sha256"], "Fold assignment changed")
    require(assignment.tolist() == reference["test_fold_assignment_by_shuffled_position"], "Fold assignment payload changed")
    return {
        "sample_count": sample_count,
        "sample_order_sha256": raw_array_hash(source_indices),
        "label_order_sha256": raw_array_hash(labels),
        "test_fold_assignment_sha256": assignment_hash,
        "fold_checks": fold_checks,
        "fold0_split_hash": fold_checks[0]["gates"]["split_hash"] and reference["folds"][0]["split_hash"],
        "passed": True,
    }


def build_lineage(config, dataset_dict, dataset_data, feature_path, dict_path) -> dict:
    baseline_manifest = verified_canonical_json(
        REFERENCE_BASELINE_MANIFEST,
        EXPECTED_BASELINE_MANIFEST_FILE_HASH,
        EXPECTED_BASELINE_MANIFEST_CANONICAL_HASH,
    )
    fold_manifest = verified_canonical_json(
        REFERENCE_FOLD_MANIFEST,
        EXPECTED_FOLD_MANIFEST_FILE_HASH,
        EXPECTED_FOLD_MANIFEST_CANONICAL_HASH,
    )
    feature_path = Path(feature_path).resolve()
    dict_path = Path(dict_path).resolve()
    feature_hash = query.file_hash(feature_path)
    modal_hash = query.file_hash(dict_path)
    require(feature_hash == EXPECTED_FEATURE_HASH, "Feature CSV hash changed")
    require(modal_hash == EXPECTED_MODAL_DICT_HASH, "Modal dictionary hash changed")
    require(feature_hash == baseline_manifest["data"]["feature_csv_sha256"], "Feature hash differs from baseline manifest")
    require(modal_hash == baseline_manifest["data"]["modal_dict_sha256"], "Modal hash differs from baseline manifest")
    require(
        baseline_manifest["fold_fairness"]["reference_manifest_file_sha256"]
        == EXPECTED_FOLD_MANIFEST_FILE_HASH,
        "Baseline fold-manifest file hash reference changed",
    )
    require(
        baseline_manifest["fold_fairness"]["canonical_sha256"]
        == EXPECTED_FOLD_MANIFEST_CANONICAL_HASH,
        "Baseline fold canonical reference changed",
    )
    require(
        baseline_manifest["fold_fairness"]["assignment_sha256"]
        == EXPECTED_FOLD_ASSIGNMENT_HASH
        == fold_manifest["test_fold_assignment_sha256"],
        "Fold assignment hash changed",
    )
    historical_protocol = validate_historical_protocol(baseline_manifest, config)
    fold_identity = generated_fold_identity(dataset_dict, dataset_data, fold_manifest)
    require(fold_identity["fold0_split_hash"] == smoke.EXPECTED_FOLD0_SPLIT_HASH, "fold0 split changed")
    return {
        "baseline_manifest": {
            "absolute_path": str(REFERENCE_BASELINE_MANIFEST),
            "file_sha256": query.file_hash(REFERENCE_BASELINE_MANIFEST),
            "canonical_sha256": baseline_manifest["canonical_sha256"],
        },
        "fold_manifest": {
            "absolute_path": str(REFERENCE_FOLD_MANIFEST),
            "file_sha256": query.file_hash(REFERENCE_FOLD_MANIFEST),
            "canonical_sha256": fold_manifest["canonical_sha256"],
            "assignment_sha256": fold_manifest["test_fold_assignment_sha256"],
        },
        "feature_csv": {
            "absolute_path": str(feature_path),
            "file_sha256": feature_hash,
        },
        "modal_dict": {
            "absolute_path": str(dict_path),
            "file_sha256": modal_hash,
        },
        "historical_protocol": historical_protocol,
        "generated_fold_identity": fold_identity,
        "passed": True,
    }


def current_reference_hashes() -> dict[str, str]:
    feature_path = ROOT / "DATASET/TADPOLE/AD_CN_SMCI_ADNI_processed_standard_data.csv"
    modal_path = ROOT / "DATASET/TADPOLE/AD_CN_SMCI_ADNI_modal_feat_dict.npy"
    return {
        "baseline_manifest": query.file_hash(REFERENCE_BASELINE_MANIFEST),
        "fold_manifest": query.file_hash(REFERENCE_FOLD_MANIFEST),
        "feature_csv": query.file_hash(feature_path),
        "modal_dict": query.file_hash(modal_path),
        "baseline_fold0_summary": query.file_hash(REFERENCE_BASELINE_FOLD0_SUMMARY),
    }


def expected_reference_hashes() -> dict[str, str]:
    return {
        "baseline_manifest": EXPECTED_BASELINE_MANIFEST_FILE_HASH,
        "fold_manifest": EXPECTED_FOLD_MANIFEST_FILE_HASH,
        "feature_csv": EXPECTED_FEATURE_HASH,
        "modal_dict": EXPECTED_MODAL_DICT_HASH,
        "baseline_fold0_summary": EXPECTED_BASELINE_FOLD0_SUMMARY_HASH,
    }


def assert_references_unchanged(stage: str) -> None:
    observed = current_reference_hashes()
    require(observed == expected_reference_hashes(), f"Historical reference/data hash changed at {stage}: {observed}")


def load_preflight() -> tuple[dict, str]:
    path = ROOT / FORMAL_ROOT_REL / "protocol_manifest_final.json"
    payload = load_json(path)
    recorded = payload.get("canonical_sha256")
    body = deepcopy(payload)
    body.pop("canonical_sha256", None)
    require(canonical_hash(body) == recorded, "Formal preflight canonical hash changed")
    require(payload["source_hashes"] == source_hashes(), "Formal source hash changed after prepare")
    assert_references_unchanged("formal preflight load")
    return payload, query.file_hash(path)


def environment_snapshot(device) -> dict:
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


def load_data_context():
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    config = smoke.Config_(str(ROOT), str(ROOT / CONFIG_REL), 0)
    config.Device = device
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
    return config, dataset_dict, dataset_data, device, feature_path, dict_path


def prepare_formal() -> dict:
    final_formal_root = ROOT / FORMAL_ROOT_REL
    formal_root = final_formal_root.parent / ".formal_prepare_in_progress"
    require(
        not final_formal_root.exists(),
        f"Refusing to overwrite formal directory: {final_formal_root}",
    )
    require(
        not formal_root.exists(),
        f"Retained formal prepare directory exists: {formal_root}",
    )
    formal_root.mkdir(parents=True)
    assert_references_unchanged("prepare start")
    config, dataset_dict, dataset_data, device, feature_path, dict_path = load_data_context()
    lineage = build_lineage(config, dataset_dict, dataset_data, feature_path, dict_path)
    locked_sources = source_hashes()

    SET_Random(SEED)
    baseline = smoke.build_model(config, dataset_dict, device, shared=False, anchor_mode="none")
    arm_audits = {}
    candidate_hashes = set()
    for arm in ARM_ORDER:
        SET_Random(SEED)
        candidate = smoke.build_model(
            config,
            dataset_dict,
            device,
            shared=True,
            anchor_mode=ARMS[arm]["anchor_mode"],
        )
        init_audit = smoke.initialization_audit(baseline, candidate)
        structure = smoke.structure_audit(candidate, arm)
        device_audit = smoke.device_audit(
            candidate,
            criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05),
            dataset_dict,
            dataset_data,
            device,
        )
        candidate_hashes.add(init_audit["candidate_full_state_hash"])
        arm_audits[arm] = {
            "definition": ARMS[arm],
            "initialization_audit": init_audit,
            "structure_audit": structure,
            "device_audit": device_audit,
        }
        del candidate
    require(candidate_hashes == {smoke.EXPECTED_CANDIDATE_INITIAL_HASH}, "Arm initialization differs")
    del baseline
    torch.cuda.empty_cache()

    baseline_summary = load_json(REFERENCE_BASELINE_FOLD0_SUMMARY)
    require(query.file_hash(REFERENCE_BASELINE_FOLD0_SUMMARY) == EXPECTED_BASELINE_FOLD0_SUMMARY_HASH, "Baseline fold0 summary changed")
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "git": git_info(),
        "formal_execution_authorized_by_user": True,
        "authorization_text": "已经改进完了是吧，开始正式实验",
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": FOLD,
            "seed": SEED,
            "epochs": EPOCHS,
            "single_model": True,
            "ensemble": False,
            "hyperparameter_search": False,
            "transductive_full_batch": True,
            "device": str(device),
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
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "loss": "main weighted CE + AD-vs-rest CE + CN-vs-rest CE + SMCI-vs-rest CE",
            "label_smoothing": 0.05,
            "orthogonality_loss": False,
            "ema_decay": EMA_DECAY,
            "ema_first_update_epoch": EMA_FIRST_UPDATE_EPOCH,
            "anchor_first_active_epoch": ANCHOR_FIRST_ACTIVE_EPOCH,
            "expected_epoch400_ema_updates": EPOCHS - EMA_FIRST_UPDATE_EPOCH + 1,
            "arm_order": list(ARM_ORDER),
            "split_hash": smoke.EXPECTED_FOLD0_SPLIT_HASH,
        },
        "lineage": lineage,
        "environment": environment_snapshot(device),
        "source_hashes": locked_sources,
        "arms": arm_audits,
        "historical_query_baseline_fold0": {
            "summary_path": str(REFERENCE_BASELINE_FOLD0_SUMMARY),
            "summary_sha256": EXPECTED_BASELINE_FOLD0_SUMMARY_HASH,
            "best_epoch": baseline_summary["result"]["best_epoch"],
            "best_metrics": baseline_summary["result"]["best_main_metrics"],
        },
    }
    preflight["canonical_sha256"] = canonical_hash(preflight)
    smoke.write_json(formal_root / "protocol_manifest_final.json", preflight)
    smoke.write_json(
        formal_root / "run_authorization_receipt.json",
        {
            "authorized": True,
            "received_utc": utc_now(),
            "authorization_text": preflight["authorization_text"],
            "scope": "S0 then S1 then S2; fold0 seed0 400 epochs each; no ensemble or other model",
            "protocol_manifest_sha256": query.file_hash(formal_root / "protocol_manifest_final.json"),
        },
    )
    smoke.write_json(formal_root / "lineage_audit.json", lineage)
    shutil.copyfile(REFERENCE_BASELINE_MANIFEST, formal_root / "reference_baseline_protocol_manifest.json")
    shutil.copyfile(REFERENCE_FOLD_MANIFEST, formal_root / "reference_fold_manifest.json")
    shutil.copyfile(REFERENCE_BASELINE_FOLD0_SUMMARY, formal_root / "reference_query_baseline_fold0_summary.json")
    formal_root.rename(final_formal_root)
    print(
        f"[prepare] PASS manifest_sha={query.file_hash(final_formal_root / 'protocol_manifest_final.json')} "
        f"candidate_init={smoke.EXPECTED_CANDIDATE_INITIAL_HASH}",
        flush=True,
    )
    return preflight


def expected_lifecycle(arm: str, epoch: int) -> dict:
    return {
        "anchor_active": arm != "s0" and epoch >= ANCHOR_FIRST_ACTIVE_EPOCH,
        "anchor_initialized": epoch >= EMA_FIRST_UPDATE_EPOCH,
        "anchor_updates": max(0, epoch - EMA_FIRST_UPDATE_EPOCH + 1),
    }


def make_checkpoint_payload(
    kind: str,
    arm: str,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    reference_logits: torch.Tensor,
    metrics: dict,
    selection_tuple,
    anchor_snapshot: dict,
    preflight_sha: str,
    split_hash: str,
    prediction_sha: str,
) -> dict:
    lifecycle = expected_lifecycle(arm, epoch)
    require(lifecycle["anchor_active"] == bool(anchor_snapshot["active"]), "Checkpoint active state mismatch")
    require(lifecycle["anchor_initialized"] == bool(anchor_snapshot["initialized"]), "Checkpoint initialized state mismatch")
    require(lifecycle["anchor_updates"] == int(anchor_snapshot["updates"]), "Checkpoint update count mismatch")
    mapping = [0, 1, 2] if arm != "s1" else [1, 2, 0]
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_kind": kind,
        "model_name": MODEL_NAME,
        "phase": "formal",
        "arm": {
            "id": arm,
            "anchor_mode": ARMS[arm]["anchor_mode"],
            "source_mapping": mapping,
        },
        "epoch": epoch,
        "protocol": {
            "fold": FOLD,
            "seed": SEED,
            "epochs": EPOCHS,
            "ema_decay": EMA_DECAY,
            "ema_first_update_epoch": EMA_FIRST_UPDATE_EPOCH,
            "anchor_first_active_epoch": ANCHOR_FIRST_ACTIVE_EPOCH,
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "split_hash": split_hash,
        },
        "construction": {
            "query_pool_variant": "shared",
            "category_branch_variant": "original",
            "category_branch_fusion": "concat",
            "semantic_fusion": "add",
            "adj_mode": "none",
            "graph_use_graph": False,
        },
        "lifecycle": {
            **lifecycle,
            "active_during_train": lifecycle["anchor_active"],
            "active_during_eval": lifecycle["anchor_active"],
        },
        "integrity": {
            "protocol_manifest_sha256": preflight_sha,
            "model_state_sha256": query.tensor_hash(model_state),
            "parameter_count": smoke.EXPECTED_CANDIDATE_PARAMETERS,
            "state_tensor_count": smoke.EXPECTED_CANDIDATE_STATE_TENSORS,
            "prediction_file_sha256": prediction_sha,
        },
        "selection": {
            "selection_tuple": list(selection_tuple),
            "metrics": metrics,
        },
        "reference_eval": {
            "full_logits": reference_logits.detach().cpu().clone(),
            "full_logits_sha256": query.tensor_hash({"logits": reference_logits}),
        },
        "anchor_snapshot": anchor_snapshot,
        "model_state_dict": model_state,
        "resume_supported": False,
    }


def restore_and_validate_formal_checkpoint(
    path: Path,
    config,
    dataset_dict: dict,
    dataset_data: dict,
    device,
    expected_preflight_sha: str,
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    require(payload["schema_version"] == CHECKPOINT_SCHEMA, "Checkpoint schema changed")
    require(payload["phase"] == "formal", "Checkpoint is not formal")
    require(payload["resume_supported"] is False, "Formal checkpoint must not support resume")
    arm = payload["arm"]["id"]
    require(arm in ARMS, "Unknown checkpoint arm")
    require(payload["arm"]["anchor_mode"] == ARMS[arm]["anchor_mode"], "Checkpoint arm/mode mismatch")
    expected_mapping = [1, 2, 0] if arm == "s1" else [0, 1, 2]
    require(payload["arm"]["source_mapping"] == expected_mapping, "Checkpoint source mapping changed")
    require(payload["integrity"]["protocol_manifest_sha256"] == expected_preflight_sha, "Checkpoint preflight hash changed")
    epoch = int(payload["epoch"])
    lifecycle = expected_lifecycle(arm, epoch)
    require(
        all(payload["lifecycle"][key] == value for key, value in lifecycle.items()),
        "Checkpoint lifecycle metadata violates epoch formula",
    )
    SET_Random(SEED)
    restored = smoke.build_model(
        config,
        dataset_dict,
        device,
        shared=True,
        anchor_mode=payload["arm"]["anchor_mode"],
    )
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored_state = restored.state_dict()
    require(len(restored_state) == smoke.EXPECTED_CANDIDATE_STATE_TENSORS, "Restored state tensor count changed")
    require(sum(p.numel() for p in restored.parameters()) == smoke.EXPECTED_CANDIDATE_PARAMETERS, "Restored parameter count changed")
    require(query.tensor_hash(restored_state) == payload["integrity"]["model_state_sha256"], "Restored state hash changed")
    require(bool(restored.osfq_anchor_initialized.item()) == lifecycle["anchor_initialized"], "Restored initialized buffer mismatch")
    require(int(restored.osfq_anchor_updates.item()) == lifecycle["anchor_updates"], "Restored update buffer mismatch")
    restored.set_osfq_anchor_active(payload["lifecycle"]["anchor_active"])
    require(bool(restored.osfq_anchor_active) == lifecycle["anchor_active"], "Self-described active restore failed")
    restored.eval()
    with torch.no_grad():
        logits, representations, auxiliary_outputs = restored(dataset_data["Feature"])
        smoke.output_audit(
            logits,
            representations,
            auxiliary_outputs,
            int(dataset_data["Feature"].size(0)),
            device,
        )
        metrics = smoke.clean_metrics(
            query.metric_bundle(
                logits,
                dataset_data["Label"],
                dataset_data["Mask"][FOLD][1],
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
        )
    expected_logits = payload["reference_eval"]["full_logits"].detach().cpu()
    max_abs_diff = float((logits.detach().cpu() - expected_logits).abs().max().item())
    metric_deltas = {
        name: abs(float(metrics[name]) - float(payload["selection"]["metrics"][name]))
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    confusion_equal = metrics["confusion_matrix"] == payload["selection"]["metrics"]["confusion_matrix"]
    result = {
        "path": path.name,
        "checkpoint_sha256": query.file_hash(path),
        "arm": arm,
        "epoch": epoch,
        "metadata_driven_restore": True,
        "restored_anchor_active": bool(restored.osfq_anchor_active),
        "restored_anchor_initialized": bool(restored.osfq_anchor_initialized.item()),
        "restored_anchor_updates": int(restored.osfq_anchor_updates.item()),
        "state_hash_equal": query.tensor_hash(restored_state) == payload["integrity"]["model_state_sha256"],
        "max_abs_logit_diff": max_abs_diff,
        "metric_abs_deltas": metric_deltas,
        "confusion_matrix_equal": confusion_equal,
        "passed": max_abs_diff == 0.0 and max(metric_deltas.values()) <= 1e-10 and confusion_equal,
    }
    require(result["passed"], f"Formal checkpoint restore failed: {result}")
    del restored
    torch.cuda.empty_cache()
    return result


def assert_run_order(arm: str, formal_root: Path) -> None:
    index = ARM_ORDER.index(arm)
    for previous in ARM_ORDER[:index]:
        summary_path = formal_root / ARMS[previous]["directory"] / "summary.json"
        require(summary_path.is_file(), f"Previous arm {previous} is incomplete")
        require(load_json(summary_path).get("formal_passed") is True, f"Previous arm {previous} did not pass")
    for later in ARM_ORDER[index + 1 :]:
        require(
            not (formal_root / ARMS[later]["directory"]).exists(),
            f"Later arm directory already exists before {arm}: {later}",
        )


def preactivation_audit(
    arm: str,
    model,
    logits: torch.Tensor,
    optimizer,
    scheduler,
    formal_root: Path,
) -> dict:
    observed = {
        "epoch": 20,
        "arm": arm,
        "anchor_active": bool(model.osfq_anchor_active),
        "anchor_initialized": bool(model.osfq_anchor_initialized.item()),
        "anchor_updates": int(model.osfq_anchor_updates.item()),
        "model_state_hash": query.tensor_hash(model.state_dict()),
        "logits_hash": query.tensor_hash({"logits": logits}),
        "ema_hash": query.tensor_hash({"ema": model.osfq_anchor_ema}),
        "optimizer_state_hash": recursive_state_hash(optimizer.state_dict()),
        "scheduler_state_hash": recursive_state_hash(scheduler.state_dict()),
    }
    gates = {
        "inactive": observed["anchor_active"] is False,
        "initialized": observed["anchor_initialized"] is True,
        "one_update": observed["anchor_updates"] == 1,
    }
    if arm != "s0":
        reference_path = formal_root / ARMS["s0"]["directory"] / "warmup_boundary_audit.json"
        reference = load_json(reference_path)["observed"]
        for key in (
            "model_state_hash",
            "logits_hash",
            "ema_hash",
            "optimizer_state_hash",
            "scheduler_state_hash",
        ):
            gates[f"matches_s0_{key}"] = observed[key] == reference[key]
    require(all(gates.values()), f"Pre-activation fairness failed for {arm}: {gates}")
    return {"observed": observed, "gates": gates, "passed": True}


def run_formal_arm(arm: str) -> dict:
    formal_root = ROOT / FORMAL_ROOT_REL
    preflight, preflight_sha = load_preflight()
    assert_run_order(arm, formal_root)
    assert_references_unchanged(f"before {arm}")
    final_dir = formal_root / ARMS[arm]["directory"]
    staging_dir = formal_root / f".{ARMS[arm]['directory']}_in_progress"
    require(not final_dir.exists(), f"Refusing to overwrite formal arm: {final_dir}")
    require(not staging_dir.exists(), f"Retained in-progress directory exists: {staging_dir}")
    staging_dir.mkdir()

    config, dataset_dict, dataset_data, device, feature_path, dict_path = load_data_context()
    lineage = build_lineage(config, dataset_dict, dataset_data, feature_path, dict_path)
    require(lineage["passed"], "Lineage audit failed")
    SET_Random(SEED)
    baseline = smoke.build_model(config, dataset_dict, device, shared=False, anchor_mode="none")
    SET_Random(SEED)
    model = smoke.build_model(
        config,
        dataset_dict,
        device,
        shared=True,
        anchor_mode=ARMS[arm]["anchor_mode"],
    )
    init_audit = smoke.initialization_audit(baseline, model)
    structure = smoke.structure_audit(model, arm)
    require(
        init_audit["candidate_full_state_hash"]
        == preflight["arms"][arm]["initialization_audit"]["candidate_full_state_hash"],
        "Fresh formal initialization differs from preflight",
    )
    del baseline
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    device_audit = smoke.device_audit(model, criterion, dataset_dict, dataset_data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.Lr_Min)
    require(len(optimizer.state) == 0, "Formal optimizer is not fresh")
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    smoke.write_json(staging_dir / "initialization_audit.json", init_audit)
    smoke.write_json(
        staging_dir / "protocol_manifest.json",
        {
            "protocol_manifest_final_sha256": preflight_sha,
            "arm": arm,
            "definition": ARMS[arm],
            "protocol": preflight["protocol"],
            "source_hashes": preflight["source_hashes"],
            "lineage": lineage,
        },
    )

    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    train_mask, test_mask = dataset_data["Mask"][FOLD]
    epoch_rows = []
    query_diagnostics = []
    anchor_diagnostics = []
    auxiliary_diagnostics = []
    best = None
    best_state = None
    best_logits = None
    best_rows = None
    best_post_activation = None
    final_state = None
    final_logits = None
    final_rows = None
    final_metrics = None
    warmup_audit = None
    alpha_gradients = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        if epoch in {1, 20, 21, EPOCHS} or epoch % 25 == 0:
            require(source_hashes() == preflight["source_hashes"], f"Source hash changed before {arm} epoch {epoch}")
            assert_references_unchanged(f"before {arm} epoch {epoch}")
        lifecycle = expected_lifecycle(arm, epoch)
        model.set_osfq_anchor_active(lifecycle["anchor_active"])
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, representations, auxiliary_outputs = model(features)
        for representation in representations:
            representation.retain_grad()
        smoke.output_audit(logits, representations, auxiliary_outputs, int(features.size(0)), device)
        loss = criterion(logits, labels, train_mask, representations, auxiliary_outputs)
        components_t = smoke.loss_components(criterion, logits, labels, train_mask, auxiliary_outputs)
        formula_delta = float((loss - components_t["total"]).detach().abs().cpu().item())
        require(formula_delta <= 1e-7, f"{arm} epoch {epoch}: loss formula changed")
        require(loss.is_cuda and bool(torch.isfinite(loss)), f"{arm} epoch {epoch}: invalid loss")
        loss.backward()
        pool_gradient = query.module_gradient_norm(model.label_pools[0])
        query_gradient = float(model.label_pools[0].query.grad.detach().norm().cpu().item())
        alpha_gradient = (
            float(model.osfq_alpha.grad.detach().cpu().item())
            if model.osfq_alpha.grad is not None
            else 0.0
        )
        head_gradients = {
            CLASS_ORDER[idx]: query.module_gradient_norm(head)
            for idx, head in enumerate(model._Auxi_classifier)
        }
        output_gradients = {
            CLASS_ORDER[idx]: float(representation.grad.detach().norm().cpu().item())
            for idx, representation in enumerate(representations)
        }
        require(math.isfinite(pool_gradient) and pool_gradient > 0.0, "Shared pool gradient invalid")
        require(math.isfinite(query_gradient) and query_gradient > 0.0, "Shared query gradient invalid")
        require(math.isfinite(alpha_gradient), "alpha gradient invalid")
        require(all(math.isfinite(value) and value > 0.0 for value in head_gradients.values()), "Auxiliary gradient invalid")
        require(all(math.isfinite(value) and value > 0.0 for value in output_gradients.values()), "Readout gradient invalid")
        require(smoke.all_parameter_gradients_finite(model), "Parameter gradient NaN/Inf")
        if epoch < ANCHOR_FIRST_ACTIVE_EPOCH or arm == "s0":
            require(abs(alpha_gradient) <= 1e-12, f"{arm} epoch {epoch}: alpha gradient leaked before treatment")
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        ema_updated = epoch >= EMA_FIRST_UPDATE_EPOCH
        if ema_updated:
            model.update_osfq_anchor_ema()
        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])

        require(bool(model.osfq_anchor_active) == lifecycle["anchor_active"], "Anchor active lifecycle changed")
        require(bool(model.osfq_anchor_initialized.item()) == lifecycle["anchor_initialized"], "EMA initialized lifecycle changed")
        require(int(model.osfq_anchor_updates.item()) == lifecycle["anchor_updates"], "EMA update lifecycle changed")
        model.eval()
        with torch.no_grad():
            eval_logits, eval_representations, eval_auxiliary = model(features)
            smoke.output_audit(eval_logits, eval_representations, eval_auxiliary, int(features.size(0)), device)
            test_loss = criterion(eval_logits, labels, test_mask, eval_representations, eval_auxiliary)
            test_components_t = smoke.loss_components(criterion, eval_logits, labels, test_mask, eval_auxiliary)
            train_metrics = smoke.clean_metrics(
                query.metric_bundle(
                    eval_logits,
                    labels,
                    train_mask,
                    dataset_dict["Label_Weight"],
                    float(config.logit_adjust_tau),
                )
            )
            test_metrics = smoke.clean_metrics(
                query.metric_bundle(
                    eval_logits,
                    labels,
                    test_mask,
                    dataset_dict["Label_Weight"],
                    float(config.logit_adjust_tau),
                )
            )
            require(
                all(math.isfinite(test_metrics[name]) for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")),
                "Formal metric NaN/Inf",
            )
            similarity = query.branch_similarity(eval_representations, list(CLASS_ORDER))
            query_stats = smoke.pairwise_vectors(model.last_osfq_effective_queries[:, 0])
            direction_stats = smoke.pairwise_vectors(model.last_osfq_centered_directions)
            snapshot = model.osfq_anchor_snapshot()
            train_aux = smoke.auxiliary_metrics(eval_auxiliary, labels, train_mask)
            test_aux = smoke.auxiliary_metrics(eval_auxiliary, labels, test_mask)
        train_components = smoke.scalar_components(components_t)
        test_components = smoke.scalar_components(test_components_t)
        contribution = snapshot["anchor_contribution_norms"]
        row = {
            "epoch": epoch,
            "lr_used": lr_used,
            "lr_next": lr_next,
            "anchor_active": bool(model.osfq_anchor_active),
            "ema_updated_this_epoch": ema_updated,
            "ema_initialized": bool(model.osfq_anchor_initialized.item()),
            "ema_update_count": int(model.osfq_anchor_updates.item()),
            "train_loss": float(loss.detach().cpu().item()),
            "test_loss": float(test_loss.detach().cpu().item()),
            **{f"train_loss_{name}": value for name, value in train_components.items()},
            **{f"test_loss_{name}": value for name, value in test_components.items()},
            **{f"train_{name}": train_metrics[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            **{f"test_{name}": test_metrics[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            "shared_pool_grad_norm": pool_gradient,
            "shared_query_grad_norm": query_gradient,
            "alpha_grad": alpha_gradient,
            "alpha": float(model.osfq_alpha.detach().cpu().item()),
            "AD_aux_head_grad_norm": head_gradients["AD"],
            "CN_aux_head_grad_norm": head_gradients["CN"],
            "SMCI_aux_head_grad_norm": head_gradients["SMCI"],
            "AD_readout_grad_norm": output_gradients["AD"],
            "CN_readout_grad_norm": output_gradients["CN"],
            "SMCI_readout_grad_norm": output_gradients["SMCI"],
            "representation_similarity_mean": similarity["off_diagonal_mean"],
            "query_AD_CN_l2": query_stats["pairs"]["AD_vs_CN"]["l2"],
            "query_AD_SMCI_l2": query_stats["pairs"]["AD_vs_SMCI"]["l2"],
            "query_CN_SMCI_l2": query_stats["pairs"]["CN_vs_SMCI"]["l2"],
            "AD_anchor_contribution_norm": contribution[0],
            "CN_anchor_contribution_norm": contribution[1],
            "SMCI_anchor_contribution_norm": contribution[2],
            "AD_ema_norm": snapshot["ema_direction_norms"][0],
            "CN_ema_norm": snapshot["ema_direction_norms"][1],
            "SMCI_ema_norm": snapshot["ema_direction_norms"][2],
            "centered_zero_sum_error": snapshot["centered_direction_zero_sum_error"],
            "centered_global_rms": snapshot["centered_direction_global_rms"],
        }
        epoch_rows.append(row)
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
                "ema_updated_this_epoch": ema_updated,
                "snapshot": snapshot,
                "centered_direction_stats": direction_stats,
                "raw_ema_direction_stats": smoke.pairwise_vectors(model.osfq_anchor_ema),
            }
        )
        auxiliary_diagnostics.append({"epoch": epoch, "train": train_aux, "test": test_aux})
        alpha_gradients.append(alpha_gradient)

        if epoch == 20:
            warmup_audit = preactivation_audit(
                arm,
                model,
                eval_logits,
                optimizer,
                scheduler,
                formal_root,
            )

        score = (test_metrics["acc"], test_metrics["macro_auc"], test_metrics["macro_f1"])
        current_rows = smoke.prediction_rows(eval_logits, labels, test_mask, dataset_dict, config)
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
                "train_loss_components": deepcopy(train_components),
                "test_loss_components": deepcopy(test_components),
                "auxiliary_metrics": deepcopy(test_aux),
                "anchor_snapshot": deepcopy(snapshot),
            }
            best_state = smoke.clone_cpu_state(model)
            best_logits = eval_logits.detach().cpu().clone()
            best_rows = deepcopy(current_rows)
        if epoch >= ANCHOR_FIRST_ACTIVE_EPOCH and (
            best_post_activation is None or score > best_post_activation["selection_tuple"]
        ):
            best_post_activation = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
                "anchor_snapshot": deepcopy(snapshot),
            }
        if epoch == EPOCHS:
            final_state = smoke.clone_cpu_state(model)
            final_logits = eval_logits.detach().cpu().clone()
            final_rows = deepcopy(current_rows)
            final_metrics = deepcopy(test_metrics)

        if epoch in {1, 20, 21, EPOCHS} or epoch % 25 == 0:
            print(
                f"[{arm}] epoch={epoch:03d}/{EPOCHS} active={model.osfq_anchor_active} "
                f"updates={int(model.osfq_anchor_updates.item()):03d} "
                f"alpha={float(model.osfq_alpha.detach().cpu()):+.6f} "
                f"loss={float(loss.detach().cpu()):.4f} ACC={test_metrics['acc']:.4f} "
                f"Macro-AUC={test_metrics['macro_auc']:.4f}",
                flush=True,
            )

    require(best is not None and best_state is not None and best_logits is not None and best_rows is not None, "Best result missing")
    require(final_state is not None and final_logits is not None and final_rows is not None, "Final result missing")
    require(best_post_activation is not None, "Post-activation diagnostic best missing")
    require(warmup_audit is not None and warmup_audit["passed"], "Warmup fairness audit missing")
    require(int(model.osfq_anchor_updates.item()) == 381, "Final EMA update count must be 381")
    if arm == "s0":
        require(all(abs(value) <= 1e-12 for value in alpha_gradients), "S0 alpha gradient must stay zero")
        require(float(model.osfq_alpha.detach().cpu().item()) == 0.0, "S0 alpha must stay zero")
    else:
        require(any(abs(value) > 1e-12 for value in alpha_gradients[20:]), f"{arm} never received active alpha gradient")

    smoke.write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    smoke.write_csv(staging_dir / "best_predictions.csv", best_rows)
    smoke.write_csv(staging_dir / "final_predictions.csv", final_rows)
    smoke.write_csv(staging_dir / "best_confusion_matrix.csv", smoke.confusion_rows(best["metrics"]["confusion_matrix"]))
    smoke.write_csv(staging_dir / "final_confusion_matrix.csv", smoke.confusion_rows(final_metrics["confusion_matrix"]))
    smoke.write_json(staging_dir / "warmup_boundary_audit.json", warmup_audit)
    smoke.write_json(
        staging_dir / "query_pool_diagnostics.json",
        {
            "definition": "one shared Per_Label_Pool called three times; OVR heads independent",
            "best_epoch": best["epoch"],
            "best_post_activation_epoch": best_post_activation["epoch"],
            "epochs": query_diagnostics,
        },
    )
    smoke.write_json(
        staging_dir / "anchor_diagnostics.json",
        {
            "definition": "stop-gradient normalized OVR direction EMA; centered and global-RMS normalized",
            "ema_first_update_epoch": EMA_FIRST_UPDATE_EPOCH,
            "anchor_first_active_epoch": ANCHOR_FIRST_ACTIVE_EPOCH,
            "ema_decay": EMA_DECAY,
            "expected_final_updates": 381,
            "epochs": anchor_diagnostics,
        },
    )
    smoke.write_json(
        staging_dir / "auxiliary_metrics.json",
        {
            "definition": "unchanged AD/CN/sMCI one-vs-rest binary heads",
            "best_epoch": best["epoch"],
            "best_test": best["auxiliary_metrics"],
            "epochs": auxiliary_diagnostics,
        },
    )
    best_prediction_audit = smoke.validate_prediction_metrics(best_rows, best["metrics"])
    final_prediction_audit = smoke.validate_prediction_metrics(final_rows, final_metrics)
    require(best_prediction_audit["passed"] and final_prediction_audit["passed"], "Prediction audit failed")
    best_prediction_sha = query.file_hash(staging_dir / "best_predictions.csv")
    final_prediction_sha = query.file_hash(staging_dir / "final_predictions.csv")
    best_payload = make_checkpoint_payload(
        "best",
        arm,
        best["epoch"],
        best_state,
        best_logits,
        best["metrics"],
        best["selection_tuple"],
        best["anchor_snapshot"],
        preflight_sha,
        smoke.EXPECTED_FOLD0_SPLIT_HASH,
        best_prediction_sha,
    )
    final_snapshot = anchor_diagnostics[-1]["snapshot"]
    final_score = (
        final_metrics["acc"],
        final_metrics["macro_auc"],
        final_metrics["macro_f1"],
    )
    final_payload = make_checkpoint_payload(
        "epoch400",
        arm,
        EPOCHS,
        final_state,
        final_logits,
        final_metrics,
        final_score,
        final_snapshot,
        preflight_sha,
        smoke.EXPECTED_FOLD0_SPLIT_HASH,
        final_prediction_sha,
    )
    best_checkpoint = staging_dir / "checkpoint_best.pt"
    final_checkpoint = staging_dir / "checkpoint_epoch400.pt"
    torch.save(best_payload, best_checkpoint)
    torch.save(final_payload, final_checkpoint)
    checkpoint_audit = {
        "unique_restore_function": "restore_and_validate_formal_checkpoint",
        "best": restore_and_validate_formal_checkpoint(
            best_checkpoint,
            config,
            dataset_dict,
            dataset_data,
            device,
            preflight_sha,
        ),
        "epoch400": restore_and_validate_formal_checkpoint(
            final_checkpoint,
            config,
            dataset_dict,
            dataset_data,
            device,
            preflight_sha,
        ),
    }
    smoke.write_json(staging_dir / "checkpoint_audit.json", checkpoint_audit)
    require(source_hashes() == preflight["source_hashes"], f"Source hash changed after {arm}")
    assert_references_unchanged(f"after {arm}")

    report_lines = [
        f"{MODEL_NAME} — {arm.upper()} Fold0 Formal 400 Epoch",
        "",
        "formal_passed=True",
        f"arm={arm}: {ARMS[arm]['label']}",
        f"device={device}; gpu={torch.cuda.get_device_name(device)}",
        f"parameters={structure['parameter_count']}; state_tensors={structure['state_tensor_count']}",
        f"initial_state_hash={init_audit['candidate_full_state_hash']}",
        f"common_tensor_count={init_audit['common_tensor_count']}",
        f"max_abs_diff={init_audit['max_abs_diff']}",
        "loss=L_main + L_AD-rest + L_CN-rest + L_sMCI-rest",
        f"EMA first update epoch={EMA_FIRST_UPDATE_EPOCH}",
        f"anchor first active epoch={ANCHOR_FIRST_ACTIVE_EPOCH}",
        f"epoch400 EMA updates={int(model.osfq_anchor_updates.item())}",
        f"best_epoch={best['epoch']}",
        f"best_metrics={best['metrics']}",
        f"best_anchor_active={best['anchor_snapshot']['active']}",
        f"best_post_activation_epoch={best_post_activation['epoch']}",
        f"best_post_activation_metrics={best_post_activation['metrics']}",
        f"epoch400_metrics={final_metrics}",
        f"final_alpha={final_snapshot['alpha']}",
        f"final_anchor_contribution_norms={final_snapshot['anchor_contribution_norms']}",
        "checkpoint restore=self-described metadata; logits max_abs_diff=0",
        "No resume, ensemble, multi-seed, or hyperparameter search.",
    ]
    (staging_dir / "structure_audit_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "arm": arm,
        "arm_label": ARMS[arm]["label"],
        "formal_passed": True,
        "git": git_info(),
        "protocol_manifest_final_sha256": preflight_sha,
        "protocol": preflight["protocol"],
        "lineage_passed": lineage["passed"],
        "initialization_audit": init_audit,
        "structure_audit": structure,
        "device_audit": device_audit,
        "warmup_boundary_audit": warmup_audit,
        "result": {
            "best_epoch": best["epoch"],
            "best_selection_tuple": list(best["selection_tuple"]),
            "best_main_metrics": best["metrics"],
            "best_anchor_snapshot": best["anchor_snapshot"],
            "best_is_post_activation": best["epoch"] >= ANCHOR_FIRST_ACTIVE_EPOCH,
            "best_post_activation_epoch": best_post_activation["epoch"],
            "best_post_activation_metrics": best_post_activation["metrics"],
            "epoch400_main_metrics": final_metrics,
            "epoch400_anchor_snapshot": final_snapshot,
        },
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": {
            "best": best_prediction_audit,
            "epoch400": final_prediction_audit,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "completed_utc": utc_now(),
        },
        "artifact_integrity": {
            "required_count": len(REQUIRED_ARM_ARTIFACTS),
            "present_count": len(REQUIRED_ARM_ARTIFACTS),
            "missing": [],
            "all_present": True,
        },
    }
    smoke.write_json(staging_dir / "summary.json", summary)
    present_without_manifest = [
        name for name in REQUIRED_ARM_ARTIFACTS if name != "artifact_manifest.json"
    ]
    missing = [name for name in present_without_manifest if not (staging_dir / name).is_file()]
    require(not missing, f"Missing formal artifacts before manifest: {missing}")
    artifact_manifest = {
        "required_count": len(REQUIRED_ARM_ARTIFACTS),
        "artifacts": {
            name: {
                "size_bytes": (staging_dir / name).stat().st_size,
                "sha256": query.file_hash(staging_dir / name),
            }
            for name in present_without_manifest
        },
    }
    smoke.write_json(staging_dir / "artifact_manifest.json", artifact_manifest)
    missing = [name for name in REQUIRED_ARM_ARTIFACTS if not (staging_dir / name).is_file()]
    require(not missing, f"Missing formal artifacts: {missing}")
    staging_dir.rename(final_dir)
    print(
        f"[{arm}] FORMAL PASS best_epoch={best['epoch']} ACC={best['metrics']['acc']:.4f} "
        f"Macro-F1={best['metrics']['macro_f1']:.4f} BACC={best['metrics']['bacc']:.4f} "
        f"Macro-AUC={best['metrics']['macro_auc']:.4f} artifacts={len(REQUIRED_ARM_ARTIFACTS)}/{len(REQUIRED_ARM_ARTIFACTS)}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary


def render_final_report(preflight: dict, summaries: dict[str, dict]) -> str:
    baseline = preflight["historical_query_baseline_fold0"]
    baseline_metrics = baseline["best_metrics"]
    lines = [
        "OVR-Aligned Shared-Query Pool v1 — Fold0 Formal 400 Epoch Report",
        "日期：2026-08-05",
        "",
        "状态：S0、S1、S2 均完成 fold0 / seed0 / CUDA / 400 epoch 正式实验。",
        "",
        "一、固定实验条件",
        "dataset=TADPOLE; task=AD_CN_SMCI; fold=0; seed=0; epochs=400",
        "transductive full batch=True; single model=True; ensemble=False; hyperparameter search=False",
        "loss=main weighted CE + AD-rest + CN-rest + sMCI-rest; orthogonality=False",
        "fusion=concat category representation + unchanged Global add; DIFFormer unchanged; graph disabled",
        "optimizer=Adam; scheduler=CustomCosineAnnealingLR(T_max=400)",
        "best epoch=ACC > Macro-AUC > Macro-F1（历史探索性协议）",
        "EMA: epoch20首次更新，epoch21首次激活，epoch400 updates=381。",
        "",
        "二、lineage 与公平性",
        f"baseline manifest SHA={preflight['lineage']['baseline_manifest']['file_sha256']}",
        f"feature CSV SHA={preflight['lineage']['feature_csv']['file_sha256']}",
        f"modal dict SHA={preflight['lineage']['modal_dict']['file_sha256']}",
        f"fold manifest SHA={preflight['lineage']['fold_manifest']['file_sha256']}",
        f"fold0 split SHA={preflight['protocol']['split_hash']}",
        f"candidate initial hash={smoke.EXPECTED_CANDIDATE_INITIAL_HASH}",
        "三个 arm 参数量、初始化、loss、数据、优化器与调度器完全相同。",
        "epoch20 model/logits/EMA/optimizer/scheduler hash 跨 arm 完全一致。",
        "",
        "三、正式结果",
        f"Original Query baseline: best_epoch={baseline['best_epoch']}; ACC={baseline_metrics['acc']}; Macro-F1={baseline_metrics['macro_f1']}; BACC={baseline_metrics['bacc']}; Macro-AUC={baseline_metrics['macro_auc']}; Weighted-F1={baseline_metrics['weighted_f1']}",
    ]
    for arm in ARM_ORDER:
        summary = summaries[arm]
        result = summary["result"]
        metrics = result["best_main_metrics"]
        anchor = result["best_anchor_snapshot"]
        lines.append(
            f"{arm.upper()} {summary['arm_label']}: best_epoch={result['best_epoch']}; "
            f"ACC={metrics['acc']}; Macro-F1={metrics['macro_f1']}; BACC={metrics['bacc']}; "
            f"Macro-AUC={metrics['macro_auc']}; Weighted-F1={metrics['weighted_f1']}; "
            f"best_anchor_active={anchor['active']}; alpha={anchor['alpha']}"
        )
    lines.extend(["", "四、对照解释"])
    s0 = summaries["s0"]["result"]
    s1 = summaries["s1"]["result"]
    s2 = summaries["s2"]["result"]
    for label, left_name, left, right_name, right in (
        ("Original Query vs S0（共享池影响）", "Original", {"best_main_metrics": baseline_metrics}, "S0", s0),
        ("S0 vs S1（任意错配anchor影响）", "S0", s0, "S1", s1),
        ("S1 vs S2（正确类别对齐贡献）", "S1", s1, "S2", s2),
        ("S0 vs S2（完整OSFQ净变化）", "S0", s0, "S2", s2),
    ):
        lm = left["best_main_metrics"]
        rm = right["best_main_metrics"]
        lines.append(
            f"{label}: ΔACC({right_name}-{left_name})={rm['acc']-lm['acc']:+.6f}; "
            f"ΔMacro-F1={rm['macro_f1']-lm['macro_f1']:+.6f}; "
            f"ΔBACC={rm['bacc']-lm['bacc']:+.6f}; ΔMacro-AUC={rm['macro_auc']-lm['macro_auc']:+.6f}"
        )
    s2_support = (
        s2["best_epoch"] >= ANCHOR_FIRST_ACTIVE_EPOCH
        and s2["best_main_metrics"]["acc"] > s1["best_main_metrics"]["acc"]
        and abs(float(s2["best_anchor_snapshot"]["alpha"])) > 1e-8
    )
    lines.extend(
        [
            "",
            "五、机制判断边界",
            f"correct alignment support under primary ACC criterion={s2_support}",
            "只有 S2 优于参数匹配 S1，且最佳点在激活后并有非零 anchor contribution，才支持正确 OVR 对齐贡献。",
            "若 Original Query 优于 S0/S2，说明共享整个 Query Pool 的容量损失仍然重要。",
            "本结果使用 test-fold best epoch，按既定历史协议仅作为探索性 fold0 证据，不是无偏泛化估计。",
            "",
            "六、完整性",
            "三个 arm 均18/18产物；两个自描述checkpoint均由唯一恢复函数复算，logits max_abs_diff=0。",
            "没有运行10-fold、多seed、ensemble或其他模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_suite() -> dict:
    formal_root = ROOT / FORMAL_ROOT_REL
    preflight, preflight_sha = load_preflight()
    summaries = {}
    rows = []
    for arm in ARM_ORDER:
        arm_dir = formal_root / ARMS[arm]["directory"]
        summary = load_json(arm_dir / "summary.json")
        require(summary.get("formal_passed") is True, f"{arm} is not formally passed")
        missing = [name for name in REQUIRED_ARM_ARTIFACTS if not (arm_dir / name).is_file()]
        require(not missing, f"{arm} missing artifacts: {missing}")
        summaries[arm] = summary
        metrics = summary["result"]["best_main_metrics"]
        rows.append(
            {
                "Model": arm.upper(),
                "Definition": summary["arm_label"],
                "Best epoch": summary["result"]["best_epoch"],
                "ACC": metrics["acc"],
                "Macro-F1": metrics["macro_f1"],
                "BACC": metrics["bacc"],
                "Macro-AUC": metrics["macro_auc"],
                "Weighted-F1": metrics["weighted_f1"],
                "Params": summary["structure_audit"]["parameter_count"],
            }
        )
    warmup_hashes = {
        key: {
            summaries[arm]["warmup_boundary_audit"]["observed"][key]
            for arm in ARM_ORDER
        }
        for key in (
            "model_state_hash",
            "logits_hash",
            "ema_hash",
            "optimizer_state_hash",
            "scheduler_state_hash",
        )
    }
    require(all(len(values) == 1 for values in warmup_hashes.values()), "Cross-arm preactivation hashes differ")
    smoke.write_csv(formal_root / "arm_metrics.csv", rows)
    report = render_final_report(preflight, summaries)
    report_path = ROOT / REPORT_REL
    report_path.write_text(report, encoding="utf-8")
    suite = {
        "schema_version": SCHEMA_VERSION,
        "formal_suite_passed": True,
        "protocol_manifest_final_sha256": preflight_sha,
        "completed_arms": list(ARM_ORDER),
        "arm_summaries": {
            arm: {
                "path": str((formal_root / ARMS[arm]["directory"] / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                "best_epoch": summaries[arm]["result"]["best_epoch"],
                "best_metrics": summaries[arm]["result"]["best_main_metrics"],
                "best_anchor_active": summaries[arm]["result"]["best_anchor_snapshot"]["active"],
                "artifacts_complete": summaries[arm]["artifact_integrity"]["all_present"],
            }
            for arm in ARM_ORDER
        },
        "preactivation_fairness": {
            "all_hashes_equal": True,
            "unique_hashes": {key: next(iter(values)) for key, values in warmup_hashes.items()},
        },
        "historical_baseline": preflight["historical_query_baseline_fold0"],
        "report": str(REPORT_REL).replace("\\", "/"),
        "report_sha256": query.file_hash(report_path),
        "completed_utc": utc_now(),
    }
    smoke.write_json(formal_root / "formal_suite_summary.json", suite)
    root_artifacts = (
        "protocol_manifest_final.json",
        "run_authorization_receipt.json",
        "lineage_audit.json",
        "reference_baseline_protocol_manifest.json",
        "reference_fold_manifest.json",
        "reference_query_baseline_fold0_summary.json",
        "arm_metrics.csv",
        "formal_suite_summary.json",
    )
    artifact_manifest = {
        "root_artifacts": {
            name: {
                "size_bytes": (formal_root / name).stat().st_size,
                "sha256": query.file_hash(formal_root / name),
            }
            for name in root_artifacts
        },
        "arm_artifact_manifests": {
            arm: load_json(formal_root / ARMS[arm]["directory"] / "artifact_manifest.json")
            for arm in ARM_ORDER
        },
    }
    smoke.write_json(formal_root / "artifact_manifest.json", artifact_manifest)
    print(f"[suite] FORMAL PASS report={REPORT_REL.as_posix()}", flush=True)
    return suite


def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} formal runner")
    parser.add_argument("--phase", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--arm", choices=ARM_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        require(args.arm is None, "--arm is not used in prepare")
        prepare_formal()
    elif args.phase == "run":
        require(args.arm is not None, "--phase run requires --arm")
        run_formal_arm(args.arm)
    else:
        require(args.arm is None, "--arm is not used in report")
        finalize_suite()


if __name__ == "__main__":
    main()
