from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_osfq_id_v1 as common
import run_query_pool_component_sharing_v1 as seps


EXPERIMENT_NAME = "Class-Private Evidence Reader and Graph v1"
OUTPUT_ROOT = Path("experiments/class_private_evidence_graph_v1")
REPORT_PATH = Path("reports/class_private_evidence_graph_v1_10fold_final.md")
BASE_COMMIT = "765720c1c0a3b2263502e9bf662fe33de7e45428"
SEED = 0
EPOCHS = 400
FOLDS = tuple(range(10))
CLASS_ORDER = ("AD", "CN", "SMCI")
METRIC_NAMES = (
    "acc",
    "macro_f1",
    "bacc",
    "probability_macro_auc",
    "weighted_f1",
)
ARMS = {
    "R": {"low_rank_reader": True, "class_graph": False},
    "G": {"low_rank_reader": False, "class_graph": True},
    "RG": {"low_rank_reader": True, "class_graph": True},
}
EXPECTED_PARAMETERS = {"B": 778_251, "R": 785_187, "G": 787_761, "RG": 794_697}
ORIGINAL_PARAMETERS = 853_131
EXPECTED_FEATURE_SHA256 = "2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042"
EXPECTED_MODAL_SHA256 = "5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273"
EXPECTED_FOLD_MANIFEST_SHA256 = "0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106"
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9"
EXPECTED_SEPS_OOF_SHA256 = "1286231fd513d811e4828319ca1210b6c76635c029107f497356e7e577a94e68"
EXPECTED_ORIGINAL_OOF_SHA256 = "86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565"
FEATURE_REL = Path("DATASET/TADPOLE/AD_CN_SMCI_ADNI_processed_standard_data.csv")
MODAL_REL = Path("DATASET/TADPOLE/AD_CN_SMCI_ADNI_modal_feat_dict.npy")
FOLD_MANIFEST_REL = Path("experiments/ovr_aligned_shared_query_v1/formal/reference_fold_manifest.json")
SEPS_OOF_REL = Path("experiments/query_pool_component_sharing_v1/tenfold_seed0/oof_predictions.csv")
SOURCE_FILES = (
    Path("Model/network.py"),
    Path("Loss/loss_fn.py"),
    Path("Loss/__init__.py"),
    Path("Utils/utils.py"),
    Path("scripts/run_query_pool_component_sharing_v1.py"),
    Path("scripts/run_class_private_evidence_graph_v1.py"),
)
REFERENCE_METRICS = {
    "Original Query": {
        "acc": 0.9297659,
        "macro_f1": 0.9140778,
        "bacc": 0.9140778,
        "probability_macro_auc": 0.9560491,
        "weighted_f1": 0.9297659,
        "parameters": 853_131,
    },
    "SEPS-Q": {
        "acc": 0.9247492,
        "macro_f1": 0.9138657,
        "bacc": 0.9082064,
        "probability_macro_auc": 0.9532899,
        "weighted_f1": 0.9246314,
        "parameters": 778_251,
    },
    "TabPFN E4": {
        "acc": 0.916388,
        "macro_f1": 0.905896,
        "bacc": 0.901319,
        "probability_macro_auc": 0.976419,
        "weighted_f1": 0.916307,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def source_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256(ROOT / path) for path in SOURCE_FILES}


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_model(config, dataset_dict, device, arm: str):
    flags = {"low_rank_reader": False, "class_graph": False}
    if arm != "B":
        flags.update(ARMS[arm])
    model = seps.build_model(
        config,
        dataset_dict,
        device,
        seps.VARIANT,
        **flags,
    )
    common.require(
        parameter_count(model) == EXPECTED_PARAMETERS[arm],
        f"{arm} parameter count changed: {parameter_count(model)}",
    )
    common.require(parameter_count(model) < ORIGINAL_PARAMETERS, f"{arm} is not below Original Query")
    return model


def common_state_audit(base_state: dict, model, arm: str) -> dict:
    candidate = model.state_dict()
    missing = sorted(set(base_state) - set(candidate))
    common.require(not missing, f"{arm}: missing SEPS-Q tensors: {missing}")
    new_names = sorted(set(candidate) - set(base_state))
    allowed_prefixes = []
    if ARMS.get(arm, {}).get("low_rank_reader"):
        allowed_prefixes.append("class_private_low_rank_reader.")
    if ARMS.get(arm, {}).get("class_graph"):
        allowed_prefixes.append("class_conditioned_sparse_evidence_graph.")
    unexpected_new = [
        name for name in new_names
        if not any(name.startswith(prefix) for prefix in allowed_prefixes)
    ]
    common.require(not unexpected_new, f"{arm}: unexpected new state tensors: {unexpected_new}")
    differences = {
        name: float((candidate[name].detach().cpu() - value).abs().max())
        for name, value in base_state.items()
    }
    max_abs_diff = max(differences.values(), default=0.0)
    mismatch = [name for name, value in differences.items() if value != 0.0]
    result = {
        "arm": arm,
        "common_tensor_count": len(base_state),
        "new_tensor_count": len(new_names),
        "new_tensor_names": new_names,
        "max_abs_diff": max_abs_diff,
        "mismatch_names": mismatch,
        "passed": max_abs_diff == 0.0 and not mismatch,
    }
    common.require(result["passed"], f"{arm}: legacy initialization changed")
    return result


def validate_locks() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    common.require(
        branch == "experiment/class-private-evidence-graph-v1",
        f"unexpected branch: {branch}",
    )
    common.require(
        git("merge-base", BASE_COMMIT, head) == BASE_COMMIT,
        "HEAD is not descended from the locked SEPS-Q base",
    )
    common.require(
        git("rev-parse", "refs/remotes/origin/experiment/query-pool-component-sharing-v1")
        == BASE_COMMIT,
        "local target base reference changed",
    )
    feature_sha = sha256(ROOT / FEATURE_REL)
    modal_sha = sha256(ROOT / MODAL_REL)
    fold_manifest_sha = sha256(ROOT / FOLD_MANIFEST_REL)
    seps_oof_sha = sha256(ROOT / SEPS_OOF_REL)
    common.require(feature_sha == EXPECTED_FEATURE_SHA256, "feature CSV hash changed")
    common.require(modal_sha == EXPECTED_MODAL_SHA256, "modal dictionary hash changed")
    common.require(fold_manifest_sha == EXPECTED_FOLD_MANIFEST_SHA256, "fold manifest hash changed")
    common.require(seps_oof_sha == EXPECTED_SEPS_OOF_SHA256, "SEPS-Q OOF hash changed")
    fold_manifest = json.loads((ROOT / FOLD_MANIFEST_REL).read_text(encoding="utf-8"))
    assignment_sha = fold_manifest["test_fold_assignment_sha256"]
    common.require(assignment_sha == EXPECTED_FOLD_ASSIGNMENT_SHA256, "fold assignment changed")
    return {
        "base_branch": "origin/experiment/query-pool-component-sharing-v1",
        "base_commit_sha": BASE_COMMIT,
        "branch": branch,
        "head_commit_sha": head,
        "worktree": str(ROOT),
        "feature_csv": FEATURE_REL.as_posix(),
        "feature_csv_sha256": feature_sha,
        "modal_dictionary": MODAL_REL.as_posix(),
        "modal_dictionary_sha256": modal_sha,
        "fold_manifest": FOLD_MANIFEST_REL.as_posix(),
        "fold_manifest_file_sha256": fold_manifest_sha,
        "fold_split_assignment_sha256": assignment_sha,
        "seps_q_oof_sha256": seps_oof_sha,
        "fold_split_hashes": [item["split_hash"] for item in fold_manifest["folds"]],
    }


def load_context():
    common.require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    device = torch.device("cuda:0")
    common.require(
        seps.normalized_text_hash(ROOT / seps.CONFIG_REL) == seps.EXPECTED_CONFIG_LF_HASH,
        "historical config changed",
    )
    config = Config_(str(ROOT), str(ROOT / seps.CONFIG_REL), 0)
    config.Device = device
    common.require(int(config.T_max) == EPOCHS, "scheduler T_max changed")
    SET_Random(SEED)
    feature_path, dict_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    common.require(Path(feature_path).resolve() == (ROOT / FEATURE_REL).resolve(), "feature path changed")
    common.require(Path(dict_path).resolve() == (ROOT / MODAL_REL).resolve(), "modal path changed")
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    common.require(int(dataset_data["Feature"].shape[0]) == 598, "subject count changed")
    common.require(int(dataset_data["Feature"].shape[1]) == 360, "feature count changed")
    common.require(int(dataset_dict["Class_Num"]) == 3, "class count changed")
    fold_manifest = json.loads((ROOT / FOLD_MANIFEST_REL).read_text(encoding="utf-8"))
    computed_splits = []
    for fold in FOLDS:
        train_mask, test_mask = dataset_data["Mask"][fold]
        split = common.query.split_hash(dataset_dict["Index"], train_mask, test_mask)
        expected = fold_manifest["folds"][fold]["split_hash"]
        common.require(split == expected, f"fold{fold} split changed")
        computed_splits.append(split)
    return config, dataset_dict, dataset_data, device, computed_splits


def environment_manifest() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def metric_bundle(logits, labels, mask, label_weight, tau) -> dict:
    raw = logits[mask]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    adjusted, probabilities, predictions = common.score_tensors(raw, label_weight, tau)
    prediction = predictions.detach().cpu().numpy().astype(np.int64)
    probability = probabilities.detach().cpu().numpy().astype(np.float64)
    adjusted_np = adjusted.detach().cpu().numpy().astype(np.float64)
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(sklearn.metrics.f1_score(truth, prediction, average="macro")),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "probability_macro_auc": float(
            sklearn.metrics.roc_auc_score(onehot, probability, multi_class="ovr", average="macro")
        ),
        "selection_macro_auc_adjusted": float(
            sklearn.metrics.roc_auc_score(onehot, adjusted_np, multi_class="ovr", average="macro")
        ),
        "weighted_f1": float(sklearn.metrics.f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
    }


def prediction_rows(fold, logits, labels, mask, dataset_dict, config) -> list[dict]:
    raw = logits[mask]
    adjusted, probabilities, predictions = common.score_tensors(
        raw,
        dataset_dict["Label_Weight"],
        float(config.logit_adjust_tau),
    )
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    return common.make_prediction_rows(
        fold,
        source_indices,
        truth,
        raw.detach().cpu().numpy(),
        adjusted.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
        predictions.detach().cpu().numpy().astype(np.int64),
    )


def metrics_from_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    probability = np.asarray([
        [
            float(row["probability_AD"]),
            float(row["probability_CN"]),
            float(row["probability_SMCI"]),
        ]
        for row in rows
    ], dtype=np.float64)
    adjusted = np.asarray([
        [
            float(row["adjusted_score_AD"]),
            float(row["adjusted_score_CN"]),
            float(row["adjusted_score_SMCI"]),
        ]
        for row in rows
    ], dtype=np.float64)
    onehot = np.eye(3, dtype=np.int64)[truth]
    common.require(np.isfinite(probability).all(), "probabilities contain NaN/Inf")
    common.require(np.allclose(probability.sum(1), 1.0, atol=1e-6, rtol=0.0), "probability rows do not sum to one")
    common.require(np.array_equal(prediction, probability.argmax(1)), "prediction/probability argmax mismatch")
    return {
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(sklearn.metrics.f1_score(truth, prediction, average="macro")),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "probability_macro_auc": float(
            sklearn.metrics.roc_auc_score(onehot, probability, multi_class="ovr", average="macro")
        ),
        "selection_macro_auc_adjusted": float(
            sklearn.metrics.roc_auc_score(onehot, adjusted, multi_class="ovr", average="macro")
        ),
        "weighted_f1": float(sklearn.metrics.f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
    }


def validate_rows(rows: list[dict], expected: dict) -> dict:
    recomputed = metrics_from_rows(rows)
    names = (*METRIC_NAMES, "selection_macro_auc_adjusted")
    delta = {name: abs(recomputed[name] - expected[name]) for name in names}
    passed = max(delta.values(), default=0.0) <= 1e-10 and (
        recomputed["confusion_matrix"] == expected["confusion_matrix"]
    )
    common.require(passed, f"prediction readback mismatch: {delta}")
    return {
        "row_count": len(rows),
        "metric_abs_delta": delta,
        "confusion_matrix_equal": recomputed["confusion_matrix"] == expected["confusion_matrix"],
        "passed": passed,
    }


def new_parameter_summary(model, arm: str) -> dict:
    prefixes = []
    if ARMS.get(arm, {}).get("low_rank_reader"):
        prefixes.append("class_private_low_rank_reader.")
    if ARMS.get(arm, {}).get("class_graph"):
        prefixes.append("class_conditioned_sparse_evidence_graph.")
    parameters = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    return {
        "arm": arm,
        "total_parameter_count": parameter_count(model),
        "added_parameter_count": sum(parameters.values()),
        "added_named_parameters": parameters,
        "below_original_query": parameter_count(model) < ORIGINAL_PARAMETERS,
    }


def gradient_module_summary(module) -> dict:
    rows = []
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        rows.append({
            "name": name,
            "has_gradient": grad is not None,
            "finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
            "norm": float(grad.detach().norm().cpu()) if grad is not None else None,
        })
    return {
        "parameter_tensors": len(rows),
        "all_have_gradient": all(row["has_gradient"] for row in rows),
        "all_finite": all(row["finite"] for row in rows),
        "nonzero_gradient_tensors": sum(
            bool(row["norm"] is not None and row["norm"] > 0.0) for row in rows
        ),
        "maximum_gradient_norm": max(
            (row["norm"] for row in rows if row["norm"] is not None),
            default=0.0,
        ),
    }


def graph_snapshot(model) -> dict:
    module = model.class_conditioned_sparse_evidence_graph
    common.require(module.last_graph_statistics is not None, "graph diagnostics missing")
    masks = module.last_edge_masks
    pairwise = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        intersection = int((masks[left] & masks[right]).sum().item() // 2)
        union = int((masks[left] | masks[right]).sum().item() // 2)
        pairwise[f"{CLASS_ORDER[left]}_vs_{CLASS_ORDER[right]}"] = {
            "intersection_edges": intersection,
            "union_edges": union,
            "edge_jaccard": float(intersection / union) if union else 1.0,
        }
    pre = module.last_pre_graph
    post = module.last_post_graph
    representation_rows = []
    obvious_oversmoothing = False
    for class_index, class_name in enumerate(CLASS_ORDER):
        pre_class = pre[:, class_index]
        post_class = post[:, class_index]
        pre_norm = F.normalize(pre_class.float(), dim=-1, eps=1e-12)
        post_norm = F.normalize(post_class.float(), dim=-1, eps=1e-12)
        node_count = int(pre_class.size(0))
        offdiag = ~torch.eye(node_count, dtype=torch.bool, device=pre_class.device)
        pre_pair = float((pre_norm @ pre_norm.T)[offdiag].mean().cpu())
        post_pair = float((post_norm @ post_norm.T)[offdiag].mean().cpu())
        delta = post_pair - pre_pair
        oversmoothed = bool(delta > 0.10 and post_pair > 0.90)
        obvious_oversmoothing = obvious_oversmoothing or oversmoothed
        representation_rows.append({
            "class": class_name,
            "mean_pre_to_post_cosine": float(
                F.cosine_similarity(pre_class.float(), post_class.float(), dim=-1).mean().cpu()
            ),
            "mean_offdiag_cosine_pre": pre_pair,
            "mean_offdiag_cosine_post": post_pair,
            "offdiag_cosine_delta": delta,
            "obvious_oversmoothing": oversmoothed,
        })
    identical = {
        f"{CLASS_ORDER[left]}_vs_{CLASS_ORDER[right]}": bool(torch.equal(masks[left], masks[right]))
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    return {
        "gamma": {
            CLASS_ORDER[index]: float(value)
            for index, value in enumerate(module.gamma.detach().cpu().tolist())
        },
        "per_class_graph": {
            CLASS_ORDER[index]: stats
            for index, stats in enumerate(module.last_graph_statistics)
        },
        "edge_jaccard": pairwise,
        "representation": representation_rows,
        "graphs_identical": identical,
        "all_graphs_identical": all(identical.values()),
        "obvious_oversmoothing": obvious_oversmoothing,
    }


def reader_snapshot(model) -> dict:
    module = model.class_private_low_rank_reader
    common.require(
        module.last_residuals is not None
        and module.last_shared_attention is not None
        and module.last_modal_weights is not None,
        "reader diagnostics missing",
    )
    residual = module.last_residuals.float()
    shared = module.last_shared_attention.float()
    weights = module.last_modal_weights.float()
    residual_norm = residual.norm(dim=-1)
    shared_norm = shared.norm(dim=-1).clamp_min(1e-12)
    class_rows = {}
    for class_index, class_name in enumerate(CLASS_ORDER):
        class_rows[class_name] = {
            "mean_residual_norm": float(residual_norm[:, class_index].mean().cpu()),
            "mean_residual_to_shared_norm_ratio": float(
                (residual_norm[:, class_index] / shared_norm[:, class_index]).mean().cpu()
            ),
            "mean_modality_weights": weights[:, class_index].mean(dim=0).cpu().tolist(),
        }
    pairwise = {}
    identical = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        name = f"{CLASS_ORDER[left]}_vs_{CLASS_ORDER[right]}"
        difference = (weights[:, left] - weights[:, right]).abs()
        pairwise[name] = {
            "mean_absolute_difference": float(difference.mean().cpu()),
            "mean_l1_difference": float(difference.sum(dim=-1).mean().cpu()),
            "maximum_absolute_difference": float(difference.max().cpu()),
        }
        identical[name] = bool(float(difference.max().cpu()) == 0.0)
    return {
        "rank": module.rank,
        "class": class_rows,
        "pairwise_weight_difference": pairwise,
        "weights_identical": identical,
        "all_class_readers_identical": all(identical.values()),
    }


def mechanism_snapshot(model, arm: str) -> dict:
    result = {"arm": arm}
    if ARMS[arm]["low_rank_reader"]:
        result["reader"] = reader_snapshot(model)
    if ARMS[arm]["class_graph"]:
        result["graph"] = graph_snapshot(model)
    return result


def initialization_and_structure_validation(config, dataset_dict, dataset_data, device) -> dict:
    features = dataset_data["Feature"]
    SET_Random(SEED)
    baseline = build_model(config, dataset_dict, device, "B")
    baseline_state = common.clone_cpu_state(baseline)
    common.require(
        common.query.tensor_hash(baseline.state_dict()) == seps.EXPECTED_CANDIDATE_HASH,
        "B state hash changed",
    )
    baseline.eval()
    with torch.no_grad():
        baseline_logits = baseline(features)[0]
    arms = {}
    graph_structure = {}
    for arm in ARMS:
        SET_Random(SEED)
        model = build_model(config, dataset_dict, device, arm)
        state_audit = common_state_audit(baseline_state, model, arm)
        incompatible = model.load_state_dict(baseline_state, strict=False)
        common.require(not incompatible.unexpected_keys, f"{arm}: unexpected base keys")
        common.require(
            sorted(incompatible.missing_keys) == state_audit["new_tensor_names"],
            f"{arm}: base load did not isolate exactly the new tensors",
        )
        model.eval()
        if ARMS[arm]["class_graph"]:
            model.class_conditioned_sparse_evidence_graph.set_capture_diagnostics(True)
        with torch.no_grad():
            logits = model(features)[0]
        maximum_difference = float((logits - baseline_logits).abs().max().cpu())
        common.require(maximum_difference <= 1e-6, f"{arm}: initialization equivalence failed")
        zero_gate = {}
        if ARMS[arm]["low_rank_reader"]:
            weights_zero = [
                bool(torch.count_nonzero(layer.weight).item() == 0)
                for layer in model.class_private_low_rank_reader.output_projections
            ]
            zero_gate["reader_B_weights_strictly_zero"] = weights_zero
            common.require(all(weights_zero), f"{arm}: reader B is not zero initialized")
        if ARMS[arm]["class_graph"]:
            gamma_zero = bool(torch.count_nonzero(
                model.class_conditioned_sparse_evidence_graph.gamma
            ).item() == 0)
            zero_gate["gamma_strictly_zero"] = gamma_zero
            common.require(gamma_zero, f"{arm}: gamma is not zero initialized")
            snapshot = graph_snapshot(model)
            graph_structure[arm] = snapshot
            for class_name, stats in snapshot["per_class_graph"].items():
                common.require(stats["symmetry_error"] <= 1e-6, f"{arm}/{class_name}: graph asymmetric")
                common.require(
                    stats["normalized_symmetry_error"] <= 1e-6,
                    f"{arm}/{class_name}: normalized graph asymmetric",
                )
                common.require(stats["nonedge_max_abs"] == 0.0, f"{arm}/{class_name}: nonedge nonzero")
                common.require(
                    stats["minimum_retained_edge_weight"] >= 0.0,
                    f"{arm}/{class_name}: negative retained edge weight",
                )
                common.require(stats["minimum_self_loop"] > 0.0, f"{arm}/{class_name}: self loop missing")
                common.require(stats["all_finite"], f"{arm}/{class_name}: graph non-finite")
        arms[arm] = {
            "parameter_summary": new_parameter_summary(model, arm),
            "legacy_state_audit": state_audit,
            "missing_new_state_keys_after_base_load": sorted(incompatible.missing_keys),
            "zero_gate_audit": zero_gate,
            "raw_logits_max_abs_diff_vs_B": maximum_difference,
            "passed": True,
        }
        del model
        torch.cuda.empty_cache()
    del baseline
    torch.cuda.empty_cache()
    return {
        "baseline_parameter_count": EXPECTED_PARAMETERS["B"],
        "baseline_state_hash": seps.EXPECTED_CANDIDATE_HASH,
        "arms": arms,
        "graph_structure": graph_structure,
        "threshold": 1e-6,
        "passed": True,
    }


def run_gradient_validation(arm, config, dataset_dict, dataset_data, device) -> dict:
    common.require(arm in {"R", "G"}, "gradient validation arm must be R or G")
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    train_mask = dataset_data["Mask"][0][0]
    SET_Random(SEED)
    model = build_model(config, dataset_dict, device, arm)
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    module = (
        model.class_private_low_rank_reader
        if arm == "R" else model.class_conditioned_sparse_evidence_graph
    )
    passes = []
    for pass_index in (1, 2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        common.require(bool(torch.isfinite(loss)), f"{arm}: non-finite gradient validation loss")
        loss.backward()
        summary = gradient_module_summary(module)
        common.require(summary["all_have_gradient"] and summary["all_finite"], f"{arm}: invalid module gradients")
        passes.append({"pass": pass_index, "loss": float(loss.detach().cpu()), **summary})
        optimizer.step()
    if arm == "R":
        common.require(passes[0]["nonzero_gradient_tensors"] >= 3, "R: output projections did not receive gradients")
        common.require(passes[1]["nonzero_gradient_tensors"] > passes[0]["nonzero_gradient_tensors"], "R: upstream reader gradients did not activate")
    else:
        gamma_values = module.gamma.detach().cpu().tolist()
        common.require(all(math.isfinite(value) for value in gamma_values), "G: gamma update non-finite")
        common.require(any(abs(value) > 0.0 for value in gamma_values), "G: gamma did not update")
        graph_transform_grad = float(module.graph_transform.weight.grad.detach().norm().cpu())
        scorer_grad = [
            float(sum(
                parameter.grad.detach().pow(2).sum()
                for parameter in scorer.parameters()
                if parameter.grad is not None
            ).sqrt().cpu())
            for scorer in module.scorers
        ]
        common.require(graph_transform_grad > 0.0, "G: graph transform gradient did not activate")
        common.require(all(value > 0.0 for value in scorer_grad), "G: scorer gradients did not activate")
    result = {
        "arm": arm,
        "passes": passes,
        "gamma_after_update": (
            module.gamma.detach().cpu().tolist() if arm == "G" else None
        ),
        "all_model_gradients_finite": common.all_parameter_gradients_finite(model),
        "passed": True,
    }
    del model, criterion, optimizer
    torch.cuda.empty_cache()
    return result


def run_smoke() -> dict:
    locks = validate_locks()
    config, dataset_dict, dataset_data, device, split_hashes = load_context()
    output = ROOT / OUTPUT_ROOT / "smoke_rg"
    staging = ROOT / OUTPUT_ROOT / ".smoke_rg_in_progress"
    if output.exists():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        common.require(summary.get("passed") is True, "existing smoke did not pass")
        print(f"SMOKE ALREADY PASSED output={output}", flush=True)
        return summary
    common.require(not staging.exists(), f"retained staging directory exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_start = source_hashes()
    initialization = initialization_and_structure_validation(
        config, dataset_dict, dataset_data, device
    )
    gradients = {
        arm: run_gradient_validation(arm, config, dataset_dict, dataset_data, device)
        for arm in ("R", "G")
    }
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    train_mask, test_mask = dataset_data["Mask"][0]
    SET_Random(SEED)
    model = build_model(config, dataset_dict, device, "RG")
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    best = None
    best_state = None
    epoch_rows = []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        common.require(bool(torch.isfinite(loss)), f"RG smoke epoch{epoch}: non-finite loss")
        loss.backward()
        common.require(common.all_parameter_gradients_finite(model), "RG smoke gradients non-finite")
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(features)
            metrics = metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
        score = (
            metrics["acc"],
            metrics["selection_macro_auc_adjusted"],
            metrics["macro_f1"],
        )
        if best is None or score > best["selection_tuple"]:
            best = {"epoch": epoch, "selection_tuple": score, "metrics": deepcopy(metrics)}
            best_state = common.clone_cpu_state(model)
        epoch_rows.append({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach().cpu()),
            **{name: metrics[name] for name in (*METRIC_NAMES, "selection_macro_auc_adjusted")},
        })
    common.require(best is not None and best_state is not None, "RG smoke best missing")
    checkpoint_path = staging / "checkpoint_best.pt"
    torch.save(best_state, checkpoint_path)
    checkpoint_audit = common.checkpoint_roundtrip(checkpoint_path, best_state)
    SET_Random(SEED)
    reloaded = build_model(config, dataset_dict, device, "RG")
    reloaded.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
    )
    reloaded.eval()
    reloaded.class_conditioned_sparse_evidence_graph.set_capture_diagnostics(True)
    with torch.no_grad():
        reload_logits, _, _ = reloaded(features)
        reload_metrics = metric_bundle(
            reload_logits,
            labels,
            test_mask,
            dataset_dict["Label_Weight"],
            float(config.logit_adjust_tau),
        )
        rows = prediction_rows(0, reload_logits, labels, test_mask, dataset_dict, config)
    common.require(reload_metrics == best["metrics"], "RG smoke checkpoint metrics mismatch")
    prediction_audit = validate_rows(rows, reload_metrics)
    graph_after_smoke = graph_snapshot(reloaded)
    common.require(source_hashes() == source_start, "sources changed during smoke")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "phase": "RG CUDA smoke",
        "fold": 0,
        "seed": SEED,
        "epochs": 3,
        "device": str(device),
        "parameter_count": parameter_count(reloaded),
        "lineage": locks,
        "environment": environment_manifest(),
        "fold_split_hashes": split_hashes,
        "initialization_equivalence": initialization,
        "single_step_gradient_validation": gradients,
        "best": {"epoch": best["epoch"], "metrics": reload_metrics},
        "graph_after_smoke": graph_after_smoke,
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": prediction_audit,
        "source_hashes": source_start,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    common.write_json(staging / "summary.json", summary)
    common.write_json(staging / "initialization_equivalence.json", initialization)
    common.write_json(staging / "gradient_validation.json", gradients)
    common.write_json(staging / "graph_structure_validation.json", initialization["graph_structure"])
    common.write_csv(staging / "epoch_metrics.csv", epoch_rows)
    common.write_csv(staging / "best_predictions.csv", rows)
    common.write_csv(
        staging / "best_confusion_matrix.csv",
        common.confusion_rows(reload_metrics["confusion_matrix"]),
    )
    shutil.copyfile(ROOT / seps.CONFIG_REL, staging / "config.ini")
    staging.rename(output)
    lineage_path = ROOT / OUTPUT_ROOT / "lineage_manifest.json"
    common.write_json(lineage_path, {
        **locks,
        "environment": environment_manifest(),
        "protocol": {
            "dataset": "TADPOLE",
            "task": "AD_CN_SMCI",
            "folds": list(FOLDS),
            "seed": SEED,
            "epochs": EPOCHS,
            "transductive_full_batch": True,
            "single_model": True,
            "ensemble": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR(T_max=400)",
            "best_epoch_rule": ["ACC", "historical adjusted-score Macro-AUC", "Macro-F1"],
            "reported_auc": "probability Macro-AUC, multiclass OVR macro",
            "old_graph_use_graph": False,
            "old_adj_mode": "none",
        },
    })
    print(
        f"SMOKE PASS best_epoch={best['epoch']} ACC={reload_metrics['acc']:.4f} "
        f"probability_AUC={reload_metrics['probability_macro_auc']:.4f}",
        flush=True,
    )
    return summary


def completed_fold(
    final_dir: Path,
    arm: str,
    fold: int,
    expected_split_hash: str,
    expected_subject_indices: set[int],
    expected_source_hashes: dict[str, str],
) -> tuple[dict, list[dict]] | None:
    if not final_dir.exists():
        return None
    required = (
        "summary.json",
        "epoch_metrics.csv",
        "best_predictions.csv",
        "best_confusion_matrix.csv",
        "checkpoint_best.pt",
        "config.ini",
    )
    missing = [name for name in required if not (final_dir / name).exists()]
    if fold == 0 and not (final_dir / "mechanism_diagnostics.json").exists():
        missing.append("mechanism_diagnostics.json")
    common.require(not missing, f"{arm}/fold{fold}: incomplete final directory: {missing}")
    summary_path = final_dir / "summary.json"
    predictions_path = final_dir / "best_predictions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_csv_rows(predictions_path)
    common.require(summary.get("formal_fold_passed") is True, f"{arm}/fold{fold}: prior fold failed")
    common.require(summary.get("arm") == arm, f"{arm}/fold{fold}: arm mismatch")
    common.require(int(summary["fold"]) == fold, f"{arm}/fold{fold}: summary mismatch")
    common.require(int(summary["seed"]) == SEED, f"{arm}/fold{fold}: seed mismatch")
    common.require(int(summary["epochs"]) == EPOCHS, f"{arm}/fold{fold}: epoch count mismatch")
    common.require(summary["split_hash"] == expected_split_hash, f"{arm}/fold{fold}: split hash mismatch")
    common.require(summary["source_hashes"] == expected_source_hashes, f"{arm}/fold{fold}: source hash mismatch")
    common.require(
        int(summary["parameter_summary"]["total_parameter_count"]) == EXPECTED_PARAMETERS[arm],
        f"{arm}/fold{fold}: parameter count mismatch",
    )
    epoch_rows = read_csv_rows(final_dir / "epoch_metrics.csv")
    common.require(len(epoch_rows) == EPOCHS, f"{arm}/fold{fold}: epoch metrics incomplete")
    common.require(
        [int(row["epoch"]) for row in epoch_rows] == list(range(1, EPOCHS + 1)),
        f"{arm}/fold{fold}: epoch sequence mismatch",
    )
    subject_indices = [int(row["subject_index"]) for row in rows]
    common.require(
        len(subject_indices) == len(expected_subject_indices)
        and len(set(subject_indices)) == len(subject_indices)
        and set(subject_indices) == expected_subject_indices,
        f"{arm}/fold{fold}: prediction coverage mismatch",
    )
    checkpoint_sha = common.query.file_hash(final_dir / "checkpoint_best.pt")
    common.require(
        checkpoint_sha == summary["checkpoint_audit"]["sha256"],
        f"{arm}/fold{fold}: checkpoint hash mismatch",
    )
    common.require(
        seps.normalized_text_hash(final_dir / "config.ini") == seps.EXPECTED_CONFIG_LF_HASH,
        f"{arm}/fold{fold}: config mismatch",
    )
    validate_rows(rows, summary["best_metrics"])
    return summary, rows


def run_formal_fold(arm, fold, config, dataset_dict, dataset_data, device, arm_root):
    final_dir = arm_root / f"fold_{fold:02d}"
    staging_dir = arm_root / f".fold_{fold:02d}_in_progress"
    common.require(not staging_dir.exists(), f"retained staging directory exists: {staging_dir}")
    train_mask, test_mask = dataset_data["Mask"][fold]
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    split_hash = common.query.split_hash(dataset_dict["Index"], train_mask, test_mask)
    source_start = source_hashes()
    test_subject_indices = set(
        np.asarray(dataset_dict["Index"], dtype=np.int64)[
            test_mask.detach().cpu().numpy().astype(bool)
        ].tolist()
    )
    existing = completed_fold(
        final_dir,
        arm,
        fold,
        split_hash,
        test_subject_indices,
        source_start,
    )
    if existing is not None:
        print(f"[{arm}/fold{fold}] RESUME completed fold", flush=True)
        return existing
    staging_dir.mkdir(parents=True, exist_ok=False)
    SET_Random(SEED)
    baseline = build_model(config, dataset_dict, device, "B")
    baseline_state = common.clone_cpu_state(baseline)
    SET_Random(SEED)
    model = build_model(config, dataset_dict, device, arm)
    legacy_audit = common_state_audit(baseline_state, model, arm)
    del baseline
    torch.cuda.empty_cache()
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    common.require(len(optimizer.state) == 0, f"{arm}/fold{fold}: optimizer is not fresh")
    best = None
    best_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        common.require(bool(torch.isfinite(loss)), f"{arm}/fold{fold}/epoch{epoch}: non-finite loss")
        loss.backward()
        common.require(common.all_parameter_gradients_finite(model), f"{arm}/fold{fold}/epoch{epoch}: non-finite gradients")
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(features)
            common.require(bool(torch.isfinite(eval_logits).all()), f"{arm}/fold{fold}/epoch{epoch}: non-finite logits")
            metrics = metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
        score = (
            metrics["acc"],
            metrics["selection_macro_auc_adjusted"],
            metrics["macro_f1"],
        )
        if best is None or score > best["selection_tuple"]:
            best = {"epoch": epoch, "selection_tuple": score, "metrics": deepcopy(metrics)}
            best_state = common.clone_cpu_state(model)
        epoch_rows.append({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach().cpu()),
            **{name: metrics[name] for name in (*METRIC_NAMES, "selection_macro_auc_adjusted")},
        })
        if epoch in {1, 100, 200, 300, 400}:
            print(
                f"[{arm}/fold{fold}] epoch={epoch:03d}/400 ACC={metrics['acc']:.4f} "
                f"probability_AUC={metrics['probability_macro_auc']:.4f}",
                flush=True,
            )
    common.require(best is not None and best_state is not None, f"{arm}/fold{fold}: best missing")
    checkpoint_path = staging_dir / "checkpoint_best.pt"
    torch.save(best_state, checkpoint_path)
    checkpoint_audit = common.checkpoint_roundtrip(checkpoint_path, best_state)
    SET_Random(SEED)
    reloaded = build_model(config, dataset_dict, device, arm)
    saved_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved_state, strict=True)
    reloaded.eval()
    if ARMS[arm]["class_graph"]:
        reloaded.class_conditioned_sparse_evidence_graph.set_capture_diagnostics(True)
    with torch.no_grad():
        best_logits, _, _ = reloaded(features)
        best_metrics = metric_bundle(
            best_logits,
            labels,
            test_mask,
            dataset_dict["Label_Weight"],
            float(config.logit_adjust_tau),
        )
        rows = prediction_rows(fold, best_logits, labels, test_mask, dataset_dict, config)
    common.require(best_metrics == best["metrics"], f"{arm}/fold{fold}: checkpoint metrics changed")
    prediction_audit = validate_rows(rows, best_metrics)
    mechanism = mechanism_snapshot(reloaded, arm) if fold == 0 else None
    common.require(source_hashes() == source_start, f"{arm}/fold{fold}: sources changed during training")
    summary = {
        "arm": arm,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "split_hash": split_hash,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "best_epoch": best["epoch"],
        "best_epoch_rule": ["ACC", "historical adjusted-score Macro-AUC", "Macro-F1"],
        "best_metrics": best_metrics,
        "parameter_summary": new_parameter_summary(reloaded, arm),
        "legacy_initialization_audit": legacy_audit,
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": prediction_audit,
        "mechanism_diagnostics": mechanism,
        "source_hashes": source_start,
        "elapsed_seconds": time.perf_counter() - started,
        "formal_fold_passed": True,
    }
    common.write_json(staging_dir / "summary.json", summary)
    common.write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    common.write_csv(staging_dir / "best_predictions.csv", rows)
    common.write_csv(
        staging_dir / "best_confusion_matrix.csv",
        common.confusion_rows(best_metrics["confusion_matrix"]),
    )
    if mechanism is not None:
        common.write_json(staging_dir / "mechanism_diagnostics.json", mechanism)
    shutil.copyfile(ROOT / seps.CONFIG_REL, staging_dir / "config.ini")
    staging_dir.rename(final_dir)
    print(
        f"[{arm}/fold{fold}] PASS best_epoch={best['epoch']} ACC={best_metrics['acc']:.4f} "
        f"probability_AUC={best_metrics['probability_macro_auc']:.4f}",
        flush=True,
    )
    del model, reloaded, criterion, optimizer, scheduler, saved_state
    torch.cuda.empty_cache()
    return summary, rows


def aggregate_arm(arm, fold_summaries, oof_rows, arm_root, suite_elapsed) -> dict:
    common.require(len(oof_rows) == 598, f"{arm}: OOF row count mismatch")
    subject_indices = [int(row["subject_index"]) for row in oof_rows]
    common.require(len(set(subject_indices)) == 598, f"{arm}: OOF subject duplication")
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    oof_metrics = metrics_from_rows(oof_rows)
    fold_rows = []
    for summary in fold_summaries:
        fold_rows.append({
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            **{name: summary["best_metrics"][name] for name in (*METRIC_NAMES, "selection_macro_auc_adjusted")},
            "parameters": summary["parameter_summary"]["total_parameter_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
        })
    metric_summary = []
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in fold_rows], dtype=np.float64)
        metric_summary.append({
            "metric": name,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "pooled_oof": float(oof_metrics[name]),
        })
    validate_rows(oof_rows, oof_metrics)
    aggregate = {
        "experiment": EXPERIMENT_NAME,
        "arm": arm,
        "configuration": {
            **ARMS[arm],
            "reader_rank": 8 if ARMS[arm]["low_rank_reader"] else None,
            "graph_top_k": 8 if ARMS[arm]["class_graph"] else None,
            "old_graph_use_graph": False,
            "old_adj_mode": "none",
        },
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "parameter_count": EXPECTED_PARAMETERS[arm],
        "added_parameter_count": EXPECTED_PARAMETERS[arm] - EXPECTED_PARAMETERS["B"],
        "fold_metrics": fold_rows,
        "metric_summary": metric_summary,
        "oof_metrics": oof_metrics,
        "fold0_mechanism_diagnostics": fold_summaries[0]["mechanism_diagnostics"],
        "total_fold_runtime_seconds": float(sum(row["elapsed_seconds"] for row in fold_summaries)),
        "suite_wall_seconds": suite_elapsed,
        "source_hashes": source_hashes(),
        "oof_sha256": None,
        "all_folds_passed": True,
    }
    common.write_csv(arm_root / "fold_metrics.csv", fold_rows)
    common.write_csv(arm_root / "metrics_summary.csv", metric_summary)
    common.write_csv(arm_root / "oof_predictions.csv", oof_rows)
    aggregate["oof_sha256"] = sha256(arm_root / "oof_predictions.csv")
    common.write_json(arm_root / "oof_metrics.json", oof_metrics)
    common.write_csv(
        arm_root / "oof_confusion_matrix.csv",
        common.confusion_rows(oof_metrics["confusion_matrix"]),
    )
    common.write_json(arm_root / "aggregate_summary.json", aggregate)
    return aggregate


def run_formal_arm(arm: str) -> dict:
    common.require(arm in ARMS, f"unknown arm: {arm}")
    smoke_path = ROOT / OUTPUT_ROOT / "smoke_rg/summary.json"
    common.require(smoke_path.exists(), "RG smoke is required before formal runs")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    common.require(smoke.get("passed") is True, "RG smoke did not pass")
    validate_locks()
    config, dataset_dict, dataset_data, device, _ = load_context()
    arm_root = ROOT / OUTPUT_ROOT / arm.lower() / "tenfold_seed0"
    arm_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = arm_root / "aggregate_summary.json"
    suite_started = time.perf_counter()
    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        summary, rows = run_formal_fold(
            arm, fold, config, dataset_dict, dataset_data, device, arm_root
        )
        fold_summaries.append(summary)
        oof_rows.extend(rows)
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        common.require(aggregate.get("all_folds_passed") is True, f"{arm}: existing aggregate failed")
        persisted_rows = read_csv_rows(arm_root / "oof_predictions.csv")
        common.require(
            sorted(oof_rows, key=lambda row: int(row["subject_index"])) == persisted_rows,
            f"{arm}: existing OOF rows mismatch",
        )
        recomputed = metrics_from_rows(persisted_rows)
        for name in (*METRIC_NAMES, "selection_macro_auc_adjusted"):
            common.require(
                abs(recomputed[name] - aggregate["oof_metrics"][name]) <= 1e-10,
                f"{arm}: existing aggregate metric mismatch",
            )
        common.require(
            sha256(arm_root / "oof_predictions.csv") == aggregate["oof_sha256"],
            f"{arm}: existing OOF hash mismatch",
        )
        print(f"{arm} 10-FOLD ALREADY PASSED", flush=True)
        return aggregate
    aggregate = aggregate_arm(
        arm,
        fold_summaries,
        oof_rows,
        arm_root,
        time.perf_counter() - suite_started,
    )
    command = {
        "command": f"{sys.executable} scripts/run_class_private_evidence_graph_v1.py formal --arm {arm}",
        "environment": environment_manifest(),
        "head_commit_sha": git("rev-parse", "HEAD"),
    }
    common.write_json(arm_root / "run_manifest.json", command)
    print(
        f"{arm} 10-FOLD PASS OOF_ACC={aggregate['oof_metrics']['acc']:.4f} "
        f"OOF_probability_AUC={aggregate['oof_metrics']['probability_macro_auc']:.4f}",
        flush=True,
    )
    return aggregate


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * probability))


def paired_comparison(model_name, model_rows, other_name, other_rows) -> tuple[dict, list[dict]]:
    left = {int(row["subject_index"]): row for row in model_rows}
    right = {int(row["subject_index"]): row for row in other_rows}
    common.require(set(left) == set(right) and len(left) == 598, "paired subject alignment failed")
    rows = []
    both_correct = model_only = other_only = both_wrong = 0
    for subject_index in sorted(left):
        model_row = left[subject_index]
        other_row = right[subject_index]
        truth = int(model_row["truth"])
        common.require(truth == int(other_row["truth"]), "paired truth mismatch")
        model_correct = int(model_row["prediction"]) == truth
        other_correct = int(other_row["prediction"]) == truth
        if model_correct and other_correct:
            relation = "both_correct"
            both_correct += 1
        elif model_correct:
            relation = "model_only_correct"
            model_only += 1
        elif other_correct:
            relation = "other_only_correct"
            other_only += 1
        else:
            relation = "both_wrong"
            both_wrong += 1
        rows.append({
            "subject_index": subject_index,
            "truth": truth,
            "model_prediction": int(model_row["prediction"]),
            "other_prediction": int(other_row["prediction"]),
            "model_correct": model_correct,
            "other_correct": other_correct,
            "relation": relation,
        })
    summary = {
        "model": model_name,
        "other": other_name,
        "both_correct": both_correct,
        "model_only_correct": model_only,
        "other_only_correct": other_only,
        "both_wrong": both_wrong,
        "discordant_subjects": model_only + other_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(model_only, other_only),
        "passed": both_correct + model_only + other_only + both_wrong == 598,
    }
    return summary, rows


def locate_original_oof(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        ROOT / "experiments/query_baseline_gpu_10fold_seed0/oof_predictions.csv",
        ROOT.parent.parent / "experiments/query_baseline_gpu_10fold_seed0/oof_predictions.csv",
    ])
    for path in candidates:
        if path.exists():
            resolved = path.resolve()
            common.require(sha256(resolved) == EXPECTED_ORIGINAL_OOF_SHA256, "Original Query OOF hash changed")
            return resolved
    raise RuntimeError("Original Query OOF file not found")


def delta_metrics(left: dict, right: dict) -> dict:
    return {name: float(left[name] - right[name]) for name in METRIC_NAMES}


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def render_report(final: dict) -> str:
    arms = final["arms"]
    best = final["best_arm"]
    best_metrics = arms[best]["oof_metrics"]
    lines = [
        "# Class-Private Evidence Reader and Graph v1 — 10-Fold Final Report",
        "",
        "## Executive result",
        "",
        f"The fixed selection rule chose **{best}**. {final['decision_conclusion']}",
        "",
        "All three arms used the locked TADPOLE 598×360 data, fold0..9, seed 0, 400 epochs, full-batch transductive training, one fresh model/criterion/Adam/scheduler per fold, and the unchanged SEPS-Q loss and historical best-epoch ordering. Original Query, SEPS-Q, and TabPFN were not rerun.",
        "",
        "## Lineage and locked protocol",
        "",
        f"- Base: `origin/experiment/query-pool-component-sharing-v1` at `{BASE_COMMIT}`",
        f"- Branch: `experiment/class-private-evidence-graph-v1`",
        f"- Worktree: `{ROOT}`",
        f"- Feature CSV SHA256: `{EXPECTED_FEATURE_SHA256}`",
        f"- Modal dictionary SHA256: `{EXPECTED_MODAL_SHA256}`",
        f"- Fold manifest SHA256: `{EXPECTED_FOLD_MANIFEST_SHA256}`",
        f"- Fold split assignment SHA256: `{EXPECTED_FOLD_ASSIGNMENT_SHA256}`",
        "- Old graph path remains disabled: `graph_use_graph=False`, `adj_mode=none`, old label-graph controls zero.",
        "- Best epoch ordering: ACC, historical adjusted-score Macro-AUC, then Macro-F1. All reported AUC values below are probability Macro-AUC (multiclass OVR macro).",
        "",
        "## Core results",
        "",
        "| Arm | Parameters | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("R", "G", "RG"):
        aggregate = arms[arm]
        metrics = aggregate["oof_metrics"]
        lines.append(
            f"| {arm} | {aggregate['parameter_count']} | {format_metric(metrics['acc'])} | "
            f"{format_metric(metrics['macro_f1'])} | {format_metric(metrics['bacc'])} | "
            f"{format_metric(metrics['probability_macro_auc'])} | {format_metric(metrics['weighted_f1'])} | "
            f"{aggregate['total_fold_runtime_seconds']:.2f} |"
        )
    for name in ("Original Query", "SEPS-Q", "TabPFN E4"):
        metrics = REFERENCE_METRICS[name]
        lines.append(
            f"| {name} | {metrics.get('parameters', '—')} | {format_metric(metrics['acc'])} | "
            f"{format_metric(metrics['macro_f1'])} | {format_metric(metrics['bacc'])} | "
            f"{format_metric(metrics['probability_macro_auc'])} | {format_metric(metrics['weighted_f1'])} | — |"
        )
    lines.extend(["", "## Fold summaries", ""])
    for arm in ("R", "G", "RG"):
        aggregate = arms[arm]
        lines.extend([
            f"### {arm}",
            "",
            "| Fold | Best epoch | ACC | Probability Macro-AUC |",
            "|---:|---:|---:|---:|",
        ])
        for row in aggregate["fold_metrics"]:
            lines.append(
                f"| {row['fold']} | {row['best_epoch']} | {format_metric(row['acc'])} | "
                f"{format_metric(row['probability_macro_auc'])} |"
            )
        lines.extend(["", "Mean ± sample SD:", ""])
        lines.append("| Metric | Mean ± SD | Pooled OOF |")
        lines.append("|---|---:|---:|")
        for row in aggregate["metric_summary"]:
            lines.append(
                f"| {row['metric']} | {format_metric(row['mean'])} ± {format_metric(row['sample_std'])} | "
                f"{format_metric(row['pooled_oof'])} |"
            )
        lines.extend([
            "",
            "Pooled confusion matrix (rows true, columns predicted; AD/CN/SMCI):",
            "",
            "```text",
            *[str(row) for row in aggregate["oof_metrics"]["confusion_matrix"]],
            "```",
            "",
        ])
    lines.extend([
        "## Paired hard-classification comparisons",
        "",
        "| Comparison | New only correct | Other only correct | Discordant | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, comparison in final["paired_comparisons"].items():
        lines.append(
            f"| {key} | {comparison['model_only_correct']} | {comparison['other_only_correct']} | "
            f"{comparison['discordant_subjects']} | {comparison['exact_mcnemar_two_sided_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Best-arm deltas and target checks",
        "",
        f"Best arm: **{best}**",
        "",
        "| Comparison | ΔACC | ΔMacro-F1 | ΔBACC | ΔProbability Macro-AUC | ΔWeighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for other in ("Original Query", "SEPS-Q"):
        delta = final["best_deltas"][other]
        lines.append(
            f"| {best} − {other} | {delta['acc']:+.6f} | {delta['macro_f1']:+.6f} | "
            f"{delta['bacc']:+.6f} | {delta['probability_macro_auc']:+.6f} | "
            f"{delta['weighted_f1']:+.6f} |"
        )
    lines.extend([
        "",
        f"- Exceeds Original Query ACC 0.9297659: **{final['target_checks']['exceeds_original_acc']}**",
        f"- Macro-F1 at least approximately 0.914: **{final['target_checks']['macro_f1_at_least_0_914']}**",
        f"- BACC at least approximately 0.914: **{final['target_checks']['bacc_at_least_0_914']}**",
        f"- Parameter count below Original Query: **{final['target_checks']['parameters_below_original']}**",
        f"- Learned graph gamma used (G/RG): **{json.dumps(final['target_checks']['graph_gamma_used'])}**",
        f"- Class graphs differ (G/RG): **{json.dumps(final['target_checks']['class_graphs_differ'])}**",
        f"- Class modality weights differ (R/RG): **{json.dumps(final['target_checks']['reader_weights_differ'])}**",
        "",
        "## Minimal fold0 mechanism report",
        "",
    ])
    for arm in ("R", "G", "RG"):
        mechanism = arms[arm]["fold0_mechanism_diagnostics"]
        lines.extend([f"### {arm}", "", "```json", json.dumps(mechanism, ensure_ascii=False, indent=2), "```", ""])
    lines.extend([
        "## Integrity",
        "",
        "- Initialization raw-logit equivalence passed for B/R/G/RG at max absolute difference ≤1e-6.",
        "- Graph symmetry, strict zero nonedges, self-loops, finite values, edge counts/density/degrees, and self-loop-only nodes were validated.",
        "- R gradients, gamma gradients, and post-gamma graph-transform/scorer gradients were finite and active.",
        "- RG fold0 CUDA three-epoch smoke and checkpoint readback passed before formal training.",
        "- Every arm contains ten independently trained fresh models and exactly one OOF prediction for each of 598 subjects.",
        "- Every best checkpoint was read back and its predictions and metrics reproduced before the fold was finalized.",
        "",
        "## Decision",
        "",
        final["decision_conclusion"],
        "",
        "No rank search, k search, multi-seed run, ensemble, TabPFN-guided graph, second dataset, new loss, or alternative GNN head was run.",
        "",
        "## Artifacts",
        "",
        "- Runner: `scripts/run_class_private_evidence_graph_v1.py`",
        "- Smoke and validation: `experiments/class_private_evidence_graph_v1/smoke_rg/`",
        "- R/G/RG ten-fold results: `experiments/class_private_evidence_graph_v1/{r,g,rg}/tenfold_seed0/`",
        "- Final JSON/CSV and paired rows: `experiments/class_private_evidence_graph_v1/final/`",
    ])
    return "\n".join(lines) + "\n"


def summarize(original_oof: str | None) -> dict:
    validate_locks()
    arms = {}
    arm_rows = {}
    for arm in ("R", "G", "RG"):
        root = ROOT / OUTPUT_ROOT / arm.lower() / "tenfold_seed0"
        aggregate_path = root / "aggregate_summary.json"
        common.require(aggregate_path.exists(), f"{arm}: aggregate missing")
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        common.require(aggregate.get("all_folds_passed") is True, f"{arm}: aggregate failed")
        rows = read_csv_rows(root / "oof_predictions.csv")
        recomputed = metrics_from_rows(rows)
        for name in (*METRIC_NAMES, "selection_macro_auc_adjusted"):
            common.require(abs(recomputed[name] - aggregate["oof_metrics"][name]) <= 1e-10, f"{arm}: OOF recompute mismatch")
        arms[arm] = aggregate
        arm_rows[arm] = rows
    selected = max(
        ("R", "G", "RG"),
        key=lambda arm: (
            arms[arm]["oof_metrics"]["acc"],
            arms[arm]["oof_metrics"]["probability_macro_auc"],
            arms[arm]["oof_metrics"]["macro_f1"],
        ),
    )
    seps_rows = read_csv_rows(ROOT / SEPS_OOF_REL)
    original_path = locate_original_oof(original_oof)
    original_rows = read_csv_rows(original_path)
    final_root = ROOT / OUTPUT_ROOT / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    paired = {}
    for arm in ("R", "G", "RG"):
        summary, rows = paired_comparison(arm, arm_rows[arm], "SEPS-Q", seps_rows)
        paired[f"{arm} vs SEPS-Q"] = summary
        common.write_json(final_root / f"paired_{arm.lower()}_vs_seps_q.json", summary)
        common.write_csv(final_root / f"paired_{arm.lower()}_vs_seps_q.csv", rows)
    summary, rows = paired_comparison(selected, arm_rows[selected], "Original Query", original_rows)
    paired[f"{selected} vs Original Query"] = summary
    common.write_json(final_root / "paired_best_vs_original_query.json", summary)
    common.write_csv(final_root / "paired_best_vs_original_query.csv", rows)
    best_metrics = arms[selected]["oof_metrics"]
    best_deltas = {
        name: delta_metrics(best_metrics, REFERENCE_METRICS[name])
        for name in ("Original Query", "SEPS-Q")
    }
    if best_metrics["acc"] < REFERENCE_METRICS["Original Query"]["acc"]:
        conclusion = "全部低于Original：本路线当前不能替代Original。"
    elif selected == "R":
        conclusion = "R最好：低秩类别读取有效，图无稳定贡献。"
    elif selected == "G":
        conclusion = "G最好：类别条件图有效，低秩读取无稳定贡献。"
    else:
        conclusion = "RG最好：两个模块具有互补性。"
    final = {
        "experiment": EXPERIMENT_NAME,
        "base_commit_sha": BASE_COMMIT,
        "branch": git("branch", "--show-current"),
        "head_commit_sha_before_result_commit": git("rev-parse", "HEAD"),
        "arms": arms,
        "selection_rule": [
            "pooled OOF ACC descending",
            "probability Macro-AUC descending",
            "Macro-F1 descending",
        ],
        "best_arm": selected,
        "best_deltas": best_deltas,
        "paired_comparisons": paired,
        "target_checks": {
            "exceeds_original_acc": best_metrics["acc"] > 0.9297659,
            "macro_f1_at_least_0_914": best_metrics["macro_f1"] >= 0.914,
            "bacc_at_least_0_914": best_metrics["bacc"] >= 0.914,
            "parameters_below_original": arms[selected]["parameter_count"] < ORIGINAL_PARAMETERS,
            "graph_gamma_used": {
                arm: any(
                    abs(value) > 0.0
                    for value in arms[arm]["fold0_mechanism_diagnostics"]["graph"]["gamma"].values()
                )
                for arm in ("G", "RG")
            },
            "class_graphs_differ": {
                arm: not arms[arm]["fold0_mechanism_diagnostics"]["graph"]["all_graphs_identical"]
                for arm in ("G", "RG")
            },
            "reader_weights_differ": {
                arm: not arms[arm]["fold0_mechanism_diagnostics"]["reader"]["all_class_readers_identical"]
                for arm in ("R", "RG")
            },
        },
        "decision_conclusion": conclusion,
        "original_query_oof_path": str(original_path),
        "original_query_oof_sha256": sha256(original_path),
        "seps_q_oof_sha256": sha256(ROOT / SEPS_OOF_REL),
        "all_recomputations_passed": True,
    }
    core_rows = []
    for arm in ("R", "G", "RG"):
        metrics = arms[arm]["oof_metrics"]
        core_rows.append({
            "arm": arm,
            "parameters": arms[arm]["parameter_count"],
            **{name: metrics[name] for name in METRIC_NAMES},
            "total_runtime_seconds": arms[arm]["total_fold_runtime_seconds"],
            "selected": arm == selected,
        })
    common.write_csv(final_root / "core_results.csv", core_rows)
    common.write_json(final_root / "final_summary.json", final)
    report_path = ROOT / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(final), encoding="utf-8")
    print(
        f"FINAL SUMMARY PASS best={selected} ACC={best_metrics['acc']:.4f} "
        f"probability_AUC={best_metrics['probability_macro_auc']:.4f}",
        flush=True,
    )
    return final


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")
    formal = subparsers.add_parser("formal")
    formal.add_argument("--arm", choices=tuple(ARMS), required=True)
    final = subparsers.add_parser("summarize")
    final.add_argument("--original-oof")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "smoke":
        run_smoke()
    elif args.command == "formal":
        run_formal_arm(args.arm)
    else:
        summarize(args.original_oof)


if __name__ == "__main__":
    main()
