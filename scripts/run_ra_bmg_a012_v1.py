#!/usr/bin/env python
"""Build and evaluate RA-BMG-A012 v1 without retraining either backbone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from Model.ra_bmg import CLASS_ORDER, RABMGA012  # noqa: E402
import run_c1_broad_hparam_search_v1 as broad  # noqa: E402
import run_cme_dual_branch_v1 as cme  # noqa: E402


EXPERIMENT = "ra_bmg_a012_v1"
BRANCH = "experiment/ra-bmg-a012-v1"
FOLDS = tuple(range(10))
EPSILON = 1e-8
BACC_FLOOR = 0.92663
EXPECTED_PARAMETERS = 862_971
EXPECTED_MAPPED_COEFFICIENTS = 147
CLASS_TOTAL_WEIGHTS = {0: 0.25, 1: 0.50, 2: 0.25}
A012_SPEC = {
    "trial_id": "A012",
    "stage": "A",
    "rank": 8,
    "base_lr": 0.011271416075886307,
    "base_weight_decay": 0.0010917677921787822,
    "lambda_aux": 0.5,
    "adapter_lr_multiplier": 2.0,
    "dropout_multiplier": 1.1,
    "generation_index": 12,
}
C1_ANCHOR = {
    "correct": 560,
    "acc": 0.9364548,
    "macro_f1": 0.9175457,
    "bacc": 0.9163359,
    "macro_auc": 0.9585608,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
A012_ANCHOR = {
    "correct": 562,
    "acc": 0.939799331103679,
    "macro_f1": 0.9294205448780151,
    "bacc": 0.9286299264715336,
    "macro_auc": 0.961524845528226,
    "weighted_f1": 0.9397414920986079,
    "confusion_matrix": [[64, 0, 8], [0, 200, 9], [7, 12, 298]],
}


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def reference_paths() -> dict[str, Path]:
    siblings = ROOT.parent
    c1 = (
        siblings
        / "cme_dual_branch_v1"
        / "experiments"
        / "cme_dual_branch_v1"
        / "c1_shared_private_control_2"
    )
    a012 = (
        siblings
        / "bp_ads_v1"
        / "experiments"
        / "bp_ads_v1"
        / "checkpoint_materialization"
        / "a012"
    )
    bp_root = siblings / "bp_ads_v1"
    return {
        "c1_root": c1,
        "c1_oof": c1 / "oof_predictions.csv",
        "a012_root": a012,
        "a012_oof": a012 / "oof_predictions.csv",
        "a012_config": a012 / "config.json",
        "bp_boundary": bp_root / "experiments" / "bp_ads_v1" / "boundary_feasibility.json",
        "bp_runner": bp_root / "scripts" / "run_bp_ads_v1.py",
    }


def indexed_rows(path: Path, label: str) -> dict[int, dict]:
    require(path.is_file(), f"Missing {label} OOF: {path}")
    rows = read_csv(path)
    indexed = {int(row["subject_index"]): row for row in rows}
    require(len(rows) == len(indexed) == 598, f"{label} OOF subject IDs are not unique 0..597")
    require(sorted(indexed) == list(range(598)), f"{label} OOF subject set changed")
    return indexed


def row_probabilities(rows: dict[int, dict]) -> np.ndarray:
    return np.asarray(
        [
            [float(rows[idx][f"probability_{name}"]) for name in CLASS_ORDER]
            for idx in range(598)
        ],
        dtype=np.float64,
    )


def metric_payload_from_arrays(
    truth: np.ndarray,
    probabilities: np.ndarray,
    simplex_tolerance: float = 1e-7,
) -> dict:
    require(probabilities.shape == (len(truth), 3), "Probability shape changed")
    require(np.isfinite(probabilities).all(), "Probabilities are non-finite")
    require(
        float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
        < float(simplex_tolerance),
        "Probability rows do not sum to one",
    )
    prediction = probabilities.argmax(axis=1)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
    return {
        "correct": int((prediction == truth).sum()),
        "acc": float((prediction == truth).mean()),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(truth, probabilities, multi_class="ovr", average="macro")),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": matrix.tolist(),
    }


def directional_errors(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "AD_to_SMCI": int(((truth == 0) & (prediction == 2)).sum()),
        "SMCI_to_AD": int(((truth == 2) & (prediction == 0)).sum()),
        "CN_to_SMCI": int(((truth == 1) & (prediction == 2)).sum()),
        "SMCI_to_CN": int(((truth == 2) & (prediction == 1)).sum()),
        "AD_to_CN": int(((truth == 0) & (prediction == 1)).sum()),
        "CN_to_AD": int(((truth == 1) & (prediction == 0)).sum()),
    }


def boundary_payload(truth: np.ndarray, prediction: np.ndarray) -> dict:
    directional = directional_errors(truth, prediction)
    return {
        **directional,
        "AD_SMCI": directional["AD_to_SMCI"] + directional["SMCI_to_AD"],
        "CN_SMCI": directional["CN_to_SMCI"] + directional["SMCI_to_CN"],
        "AD_CN": directional["AD_to_CN"] + directional["CN_to_AD"],
    }


def paired_payload(
    truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict:
    cand_pred = candidate.argmax(axis=1)
    base_pred = baseline.argmax(axis=1)
    repairs_mask = (base_pred != truth) & (cand_pred == truth)
    damages_mask = (base_pred == truth) & (cand_pred != truth)
    changed_mask = cand_pred != base_pred

    def sources(mask: np.ndarray, prediction: np.ndarray) -> dict:
        result = {}
        for truth_idx, truth_name in enumerate(CLASS_ORDER):
            for pred_idx, pred_name in enumerate(CLASS_ORDER):
                if truth_idx != pred_idx:
                    result[f"{truth_name}_to_{pred_name}"] = int(
                        (mask & (truth == truth_idx) & (prediction == pred_idx)).sum()
                    )
        return result

    repairs = int(repairs_mask.sum())
    damages = int(damages_mask.sum())
    return {
        "repairs": repairs,
        "damages": damages,
        "net_repairs": repairs - damages,
        "changed": int(changed_mask.sum()),
        "repair_sources_from_baseline_errors": sources(repairs_mask, base_pred),
        "damage_sources_in_candidate_errors": sources(damages_mask, cand_pred),
    }


def validate_reference_oof() -> dict:
    paths = reference_paths()
    c1 = indexed_rows(paths["c1_oof"], "C1")
    a012 = indexed_rows(paths["a012_oof"], "A012")
    for subject in range(598):
        require(
            int(c1[subject]["truth"]) == int(a012[subject]["truth"])
            and int(c1[subject]["fold"]) == int(a012[subject]["fold"]),
            f"C1/A012 subject-label-fold mismatch at {subject}",
        )
    truth = np.asarray([int(c1[idx]["truth"]) for idx in range(598)], dtype=np.int64)
    p_c1, p_a012 = row_probabilities(c1), row_probabilities(a012)
    # Historical CSVs were serialized from float32 and can miss one by about
    # 1.3e-7; the strict <1e-7 gate applies to newly coupled RA probabilities.
    c1_metrics = metric_payload_from_arrays(truth, p_c1, simplex_tolerance=2e-6)
    a012_metrics = metric_payload_from_arrays(truth, p_a012, simplex_tolerance=2e-6)
    require(c1_metrics["correct"] == 560 and c1_metrics["confusion_matrix"] == C1_ANCHOR["confusion_matrix"], "C1 OOF changed")
    require(a012_metrics["correct"] == 562 and a012_metrics["confusion_matrix"] == A012_ANCHOR["confusion_matrix"], "A012 OOF changed")

    # Recompute the already-validated B1 mechanism instead of trusting its name.
    protect_cn = p_c1.argmax(axis=1) == 1
    b1 = np.empty_like(p_c1)
    b1[protect_cn] = p_c1[protect_cn]
    q_cn = p_c1[~protect_cn, 1]
    conditional = p_a012[~protect_cn, 0] / (
        p_a012[~protect_cn, 0] + p_a012[~protect_cn, 2] + 1e-12
    )
    b1[~protect_cn, 1] = q_cn
    b1[~protect_cn, 0] = (1.0 - q_cn) * conditional
    b1[~protect_cn, 2] = (1.0 - q_cn) * (1.0 - conditional)
    b1_metrics = metric_payload_from_arrays(truth, b1, simplex_tolerance=2e-6)
    b1_boundary = boundary_payload(truth, b1.argmax(axis=1))
    require(
        b1_metrics["correct"] == 564
        and b1_boundary["AD_SMCI"] == 17
        and b1_boundary["CN_SMCI"] == 17
        and b1_boundary["AD_CN"] == 0,
        "B1 is not C1-CN protection plus A012 AD/SMCI conditional decision",
    )
    boundary_file = json.loads(paths["bp_boundary"].read_text(encoding="utf-8"))
    require(boundary_file["selected_direction"] == "B1_c1_protect_cn_a012_ads", "Historical B1 selection changed")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": {
            "c1_oof": sha256_file(paths["c1_oof"]),
            "a012_oof": sha256_file(paths["a012_oof"]),
            "bp_boundary": sha256_file(paths["bp_boundary"]),
            "bp_runner": sha256_file(paths["bp_runner"]),
        },
        "c1_metrics_actual_locked_oof": c1_metrics,
        "c1_user_anchor": C1_ANCHOR,
        "a012_metrics": a012_metrics,
        "b1_definition": "C1 argmax-CN rows retain C1; all other rows use C1 q_CN and A012 AD:SMCI conditional probability",
        "b1_metrics": b1_metrics,
        "b1_boundary": b1_boundary,
        "subject_alignment": "subject_index; truth and fold exact",
    }


def checkpoint_inventory() -> list[dict]:
    paths = reference_paths()
    a012_config = json.loads(paths["a012_config"].read_text(encoding="utf-8"))
    require(a012_config["trial"] == A012_SPEC, "A012 checkpoint configuration changed")
    require(a012_config["folds"] == list(FOLDS), "A012 fold list changed")
    inventory = []
    for fold in FOLDS:
        c1_dir = paths["c1_root"] / f"fold_{fold:02d}"
        a012_dir = paths["a012_root"] / f"fold_{fold:02d}"
        c1_checkpoint = c1_dir / "checkpoint_best.pt"
        a012_checkpoint = a012_dir / "checkpoint_best.pt"
        c1_summary_path = c1_dir / "summary.json"
        a012_summary_path = a012_dir / "summary.json"
        for required in (c1_checkpoint, a012_checkpoint, c1_summary_path, a012_summary_path):
            require(required.is_file(), f"Missing required fold artifact: {required}")
        c1_summary = json.loads(c1_summary_path.read_text(encoding="utf-8"))
        a012_summary = json.loads(a012_summary_path.read_text(encoding="utf-8"))
        require(c1_summary["fold"] == fold and c1_summary["arm"] == "c1", f"C1 fold{fold} ownership sidecar changed")
        require(a012_summary["fold"] == fold and a012_summary["config_name"] == "a012", f"A012 fold{fold} ownership sidecar changed")
        a012_payload = torch.load(a012_checkpoint, map_location="cpu", weights_only=True)
        require(a012_payload["fold_lock"]["fold"] == fold and a012_payload["fold_lock"]["config_name"] == "a012", f"A012 fold{fold} embedded lock changed")
        require(tuple(a012_payload["model_state"]["GCN.classifier.weight"].shape) == (3, 48), "A012 final classifier changed")
        c1_state = torch.load(c1_checkpoint, map_location="cpu", weights_only=True)
        require(tuple(c1_state["GCN.classifier.weight"].shape) == (3, 48), "C1 final classifier changed")
        inventory.append(
            {
                "fold": fold,
                "c1_checkpoint": str(c1_checkpoint),
                "c1_checkpoint_sha256": sha256_file(c1_checkpoint),
                "c1_summary": str(c1_summary_path),
                "c1_summary_sha256": sha256_file(c1_summary_path),
                "c1_best_epoch": int(c1_summary["best_epoch"]),
                "a012_checkpoint": str(a012_checkpoint),
                "a012_checkpoint_sha256": sha256_file(a012_checkpoint),
                "a012_summary": str(a012_summary_path),
                "a012_summary_sha256": sha256_file(a012_summary_path),
                "a012_best_epoch": int(a012_payload["best_epoch"]),
                "a012_fold_lock": a012_payload["fold_lock"],
            }
        )
    return inventory


def source_dependencies() -> dict[str, str]:
    relative_paths = (
        ".gitignore",
        "Model/ra_bmg.py",
        "Model/cme_dual_branch.py",
        "Model/network.py",
        "Model/models.py",
        "Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini",
        "scripts/run_ra_bmg_a012_v1.py",
        "scripts/run_c1_broad_hparam_search_v1.py",
        "scripts/run_cme_dual_branch_v1.py",
        "Loss/loss_fn.py",
        "Utils/utils.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relative_paths}


def prepare_config(output_root: Path) -> dict:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong RA-BMG branch")
    reference = validate_reference_oof()
    inventory = checkpoint_inventory()
    payload = {
        "experiment": EXPERIMENT,
        "branch": BRANCH,
        "base_commit": git("rev-parse", "HEAD").stdout.strip(),
        "a012_spec": A012_SPEC,
        "folds": list(FOLDS),
        "class_order": list(CLASS_ORDER),
        "alignment": {
            "fit_rows": "fold train rows only",
            "class_total_weights": {"AD": 0.25, "CN": 0.50, "SMCI": 0.25},
            "algorithm": "float64 weighted Procrustes plus per-dimension variance recovery",
            "epsilon": EPSILON,
            "rotation": "R=U@Vh without determinant correction",
            "cn_margin": "z_CN-logsumexp([z_AD,z_SMCI])",
            "sign_agreement": "sign(mapped_margin)==sign(original_margin)",
            "normalized_rmse": "unweighted RMSE/(population std(original C1 train margin)+1e-8)",
            "full_logit_cosine": "unweighted train-row mean cosine(mapped C1 logits, original C1 logits)",
        },
        "coupling": {
            "q_cn": "softmax(mapped_C1_raw_logits)[CN]",
            "q_ad_given_non_cn": "softmax(A012_raw_logits[AD,SMCI])[AD]",
            "probability_order": list(CLASS_ORDER),
            "evaluation": "direct softmax(log(coupled_probability)); no second logit adjustment",
        },
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "one_backbone_forward": True,
        "expected_a012_parameters": EXPECTED_PARAMETERS,
        "expected_mapped_head_coefficients": EXPECTED_MAPPED_COEFFICIENTS,
        "reference": reference,
        "checkpoint_inventory": inventory,
        "checkpoint_inventory_sha256": payload_sha256({"folds": inventory}),
        "source_dependencies": source_dependencies(),
    }
    payload["config_sha256"] = payload_sha256(payload)
    write_json(output_root / "experiment_config.json", payload)
    return payload


def load_prepared_config(output_root: Path) -> dict:
    path = output_root / "experiment_config.json"
    require(path.is_file(), "Run prepare before smoke/formal")
    config = json.loads(path.read_text(encoding="utf-8"))
    digest_payload = {key: value for key, value in config.items() if key != "config_sha256"}
    require(config["config_sha256"] == payload_sha256(digest_payload), "Prepared config changed")
    require(config["a012_spec"] == A012_SPEC and config["folds"] == list(FOLDS), "Prepared protocol changed")
    require(config["source_dependencies"] == source_dependencies(), "Locked source dependency changed")
    require(config["checkpoint_inventory"] == checkpoint_inventory(), "Checkpoint inventory changed")
    current_reference = validate_reference_oof()
    require(config["reference"]["sha256"] == current_reference["sha256"], "Reference OOF/B1 artifacts changed")
    return config


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "RA-BMG formal protocol requires cuda:0")
    context = broad.load_context(device_text)
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(tuple(context["dataset_data"]["Feature"].shape) == (598, 360), "Dataset shape changed")
    require(int(context["dataset_data"]["Label"].numel()) == 598, "Label count changed")
    require(context["fold_manifest"]["sha256"] == "21474e24414c295d31e2d0c7d1d862a3a07d96ce8dfefe37d0d05f5ec50115c4", "Fold manifest changed")
    return context


def state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def build_backbone(context: dict, state: dict[str, torch.Tensor]):
    model = broad.build_model(context, 8)
    model.load_state_dict(state, strict=True)
    model.eval()
    require(sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS, "A012/C1 parameter count changed")
    require(isinstance(model.GCN.classifier, torch.nn.Linear), "Final classifier is not nn.Linear")
    require(model.GCN.classifier.in_features == 48 and model.GCN.classifier.out_features == 3, "Final classifier dimensions changed")
    return model


def build_a012_backbone(context: dict, state: dict[str, torch.Tensor]):
    model, criterion, optimizer, scheduler, dropout, groups = broad.make_training_objects(
        context, A012_SPEC
    )
    del criterion, optimizer, scheduler, groups
    require(len(dropout) == 10, "A012 Dropout module audit changed")
    final_probabilities = sorted(round(float(item["final_p"]), 12) for item in dropout)
    require(
        final_probabilities == sorted([round(0.335 * 1.1, 12)] * 9 + [round(0.67 * 1.1, 12)]),
        "A012 dropout_multiplier=1.1 was not applied",
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    require(sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS, "A012 parameter count changed")
    model._ra_bmg_dropout_audit = dropout
    return model


def forward_with_hidden(model, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    captured: list[torch.Tensor] = []

    def hook(module, inputs):
        del module
        require(len(inputs) == 1 and isinstance(inputs[0], torch.Tensor), "Classifier hook schema changed")
        captured.append(inputs[0])

    handle = model.GCN.classifier.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            output = model(features)
    finally:
        handle.remove()
    require(isinstance(output, tuple) and len(output) >= 1, "Backbone output schema changed")
    require(len(captured) == 1, "Final classifier hook did not fire exactly once")
    logits, hidden = output[0], captured[0]
    require(tuple(logits.shape) == (598, 3) and tuple(hidden.shape) == (598, 48), "Logit/hidden shape changed")
    require(torch.isfinite(logits).all() and torch.isfinite(hidden).all(), "Backbone output is non-finite")
    return logits.detach(), hidden.detach(), len(captured)


def compare_historical_checkpoint_rows(
    context: dict,
    fold: int,
    logits: torch.Tensor,
    expected_indexed: dict[int, dict],
    label: str,
) -> dict:
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    actual = cme.prediction_rows(
        fold,
        logits,
        labels,
        test_mask,
        context["dataset_dict"],
        context["config"],
    )
    maximum = {"raw_logit": 0.0, "adjusted_score": 0.0, "probability": 0.0}
    for row in actual:
        subject = int(row["subject_index"])
        expected = expected_indexed[subject]
        for key in ("fold", "subject_index", "truth", "prediction"):
            require(int(row[key]) == int(expected[key]), f"{label} fold{fold} {key} OOF readback changed")
        for prefix in maximum:
            difference = max(
                abs(float(row[f"{prefix}_{name}"]) - float(expected[f"{prefix}_{name}"]))
                for name in CLASS_ORDER
            )
            maximum[prefix] = max(maximum[prefix], difference)
    require(max(maximum.values()) <= 1e-6, f"{label} fold{fold} checkpoint does not reproduce locked OOF: {maximum}")
    return maximum


def fit_weighted_alignment(
    hidden_a_train: torch.Tensor,
    hidden_c_train: torch.Tensor,
    labels_train: torch.Tensor,
) -> dict:
    # This API accepts train slices, so test labels cannot enter any statistic.
    a = hidden_a_train.detach().cpu().numpy().astype(np.float64, copy=True)
    c_hidden = hidden_c_train.detach().cpu().numpy().astype(np.float64, copy=True)
    labels = labels_train.detach().cpu().numpy().astype(np.int64, copy=True)
    require(a.shape == c_hidden.shape and a.ndim == 2 and a.shape[1] == 48, "Train hidden dimensions changed")
    weights = np.zeros(len(labels), dtype=np.float64)
    class_counts = {}
    for class_index, total_weight in CLASS_TOTAL_WEIGHTS.items():
        mask = labels == class_index
        count = int(mask.sum())
        require(count > 0, f"Train fold lacks class {class_index}")
        weights[mask] = total_weight / count
        class_counts[CLASS_ORDER[class_index]] = count
    require(abs(float(weights.sum()) - 1.0) <= 1e-14, "Class-balanced weights do not sum to one")
    mu_a = (weights[:, None] * a).sum(axis=0)
    mu_c = (weights[:, None] * c_hidden).sum(axis=0)
    x_a, x_c = a - mu_a, c_hidden - mu_c
    cross = x_a.T @ (weights[:, None] * x_c)
    u, singular_values, vh = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    rotated = x_a @ rotation
    var_c = (weights[:, None] * np.square(x_c)).sum(axis=0)
    var_rotated = (weights[:, None] * np.square(rotated)).sum(axis=0)
    scale = np.sqrt((var_c + EPSILON) / (var_rotated + EPSILON))
    transform = rotation * scale[None, :]
    offset = mu_c - mu_a @ transform
    for name, value in (
        ("singular_values", singular_values),
        ("rotation", rotation),
        ("scale", scale),
        ("transform", transform),
        ("offset", offset),
    ):
        require(np.isfinite(value).all(), f"Alignment {name} is non-finite")
    return {
        "transform": transform,
        "offset": offset,
        "weights": weights,
        "class_counts": class_counts,
        "singular_values": singular_values,
        "scale": scale,
    }


def fold_classifier(
    alignment: dict,
    c1_classifier: torch.nn.Linear,
    hidden_a_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    transform = alignment["transform"]
    offset = alignment["offset"]
    weight_c = c1_classifier.weight.detach().cpu().numpy().astype(np.float64)
    bias_c = c1_classifier.bias.detach().cpu().numpy().astype(np.float64)
    mapped_weight = weight_c @ transform.T
    mapped_bias = bias_c + weight_c @ offset
    a = hidden_a_train.detach().cpu().numpy().astype(np.float64)
    direct = a @ mapped_weight.T + mapped_bias
    explicit = (a @ transform + offset) @ weight_c.T + bias_c
    algebra_error = float(np.max(np.abs(direct - explicit)))
    require(algebra_error <= 1e-10, f"Folded classifier algebra changed: {algebra_error}")
    require(np.isfinite(mapped_weight).all() and np.isfinite(mapped_bias).all(), "Mapped classifier is non-finite")
    return torch.from_numpy(mapped_weight), torch.from_numpy(mapped_bias), algebra_error


def alignment_diagnostics(
    c1_logits_train: torch.Tensor,
    mapped_logits_train: torch.Tensor,
    alignment: dict,
    algebra_error: float,
) -> dict:
    original = c1_logits_train.detach().cpu().numpy().astype(np.float64)
    mapped = mapped_logits_train.detach().cpu().numpy().astype(np.float64)

    def cn_margin(logits: np.ndarray) -> np.ndarray:
        rest = np.logaddexp(logits[:, 0], logits[:, 2])
        return logits[:, 1] - rest

    original_margin, mapped_margin = cn_margin(original), cn_margin(mapped)
    require(float(np.std(original_margin)) > 0 and float(np.std(mapped_margin)) > 0, "CN margin variance collapsed")
    pearson = float(np.corrcoef(original_margin, mapped_margin)[0, 1])
    sign_agreement = float(np.mean(np.sign(original_margin) == np.sign(mapped_margin)))
    rmse = float(np.sqrt(np.mean(np.square(mapped_margin - original_margin))))
    normalized_rmse = rmse / (float(np.std(original_margin, ddof=0)) + EPSILON)
    numerator = np.sum(original * mapped, axis=1)
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(mapped, axis=1)
    cosine = float(np.mean(numerator / np.maximum(denominator, EPSILON)))
    diagnostics = {
        "cn_margin_definition": "z_CN-logsumexp([z_AD,z_SMCI])",
        "cn_margin_pearson": pearson,
        "cn_margin_sign_agreement": sign_agreement,
        "cn_margin_rmse": rmse,
        "cn_margin_normalized_rmse": normalized_rmse,
        "full_c1_logits_mean_cosine": cosine,
        "folded_classifier_algebra_max_abs_diff": algebra_error,
        "svd_singular_min": float(alignment["singular_values"].min()),
        "svd_singular_max": float(alignment["singular_values"].max()),
        "variance_scale_min": float(alignment["scale"].min()),
        "variance_scale_max": float(alignment["scale"].max()),
        "svd_finite": bool(np.isfinite(alignment["singular_values"]).all()),
        "variance_scale_finite": bool(np.isfinite(alignment["scale"]).all()),
        "fit_class_counts": alignment["class_counts"],
        "fit_weight_sum": float(alignment["weights"].sum()),
        "fit_uses_test_labels": False,
    }
    require(all(math.isfinite(value) for key, value in diagnostics.items() if isinstance(value, float)), "Alignment diagnostic is non-finite")
    return diagnostics


def make_runtime_config(prepared: dict, context: dict) -> dict:
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong formal branch")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", prepared["base_commit"], source_commit],
            cwd=ROOT,
        ).returncode
        == 0,
        "Source commit does not descend from prepared base",
    )
    payload = {
        "experiment": EXPERIMENT,
        "branch": BRANCH,
        "source_commit": source_commit,
        "prepared_config_sha256": prepared["config_sha256"],
        "source_dependencies": source_dependencies(),
        "checkpoint_inventory_sha256": prepared["checkpoint_inventory_sha256"],
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "folds": list(FOLDS),
        "device": "cuda:0",
        "a012_spec": A012_SPEC,
        "class_order": list(CLASS_ORDER),
        "epsilon": EPSILON,
        "class_total_weights": {"AD": 0.25, "CN": 0.50, "SMCI": 0.25},
        "mapped_head_coefficients": EXPECTED_MAPPED_COEFFICIENTS,
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "one_backbone_forward": True,
        "graph": False,
    }
    payload["config_sha256"] = payload_sha256(payload)
    return payload


def fold_lock(runtime: dict, prepared: dict, fold: int) -> dict:
    inventory = prepared["checkpoint_inventory"][fold]
    require(inventory["fold"] == fold, "Checkpoint inventory order changed")
    return {
        "experiment": EXPERIMENT,
        "fold": int(fold),
        "source_commit": runtime["source_commit"],
        "runtime_config_sha256": runtime["config_sha256"],
        "prepared_config_sha256": prepared["config_sha256"],
        "fold_manifest_sha256": runtime["fold_manifest_sha256"],
        "c1_checkpoint_sha256": inventory["c1_checkpoint_sha256"],
        "c1_summary_sha256": inventory["c1_summary_sha256"],
        "a012_checkpoint_sha256": inventory["a012_checkpoint_sha256"],
        "a012_summary_sha256": inventory["a012_summary_sha256"],
    }


def load_fold_models(context: dict, prepared: dict, fold: int):
    inventory = prepared["checkpoint_inventory"][fold]
    c1_path = Path(inventory["c1_checkpoint"])
    a012_path = Path(inventory["a012_checkpoint"])
    require(sha256_file(c1_path) == inventory["c1_checkpoint_sha256"], f"C1 fold{fold} checkpoint changed")
    require(sha256_file(a012_path) == inventory["a012_checkpoint_sha256"], f"A012 fold{fold} checkpoint changed")
    c1_sidecar = json.loads(Path(inventory["c1_summary"]).read_text(encoding="utf-8"))
    require(c1_sidecar["fold"] == fold and c1_sidecar["best_epoch"] == inventory["c1_best_epoch"], f"C1 fold{fold} ownership changed")
    c1_state = torch.load(c1_path, map_location="cpu", weights_only=True)
    a012_payload = torch.load(a012_path, map_location="cpu", weights_only=True)
    require(a012_payload["fold_lock"] == inventory["a012_fold_lock"], f"A012 fold{fold} lock changed")
    require(a012_payload["best_epoch"] == inventory["a012_best_epoch"], f"A012 fold{fold} best epoch changed")
    c1_model = build_backbone(context, c1_state)
    a012_model = build_a012_backbone(context, a012_payload["model_state"])
    return c1_model, a012_model


def direct_probability_rows(
    fold: int,
    test_indices: torch.Tensor,
    labels: torch.Tensor,
    logits_out: torch.Tensor,
    probabilities: torch.Tensor,
    a012_raw_logits: torch.Tensor,
    mapped_c1_logits: torch.Tensor,
    dataset_dict: dict,
) -> list[dict]:
    logits_cpu = logits_out.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu()
    a012_cpu = a012_raw_logits.detach().cpu().to(torch.float64)
    mapped_cpu = mapped_c1_logits.detach().cpu().to(torch.float64)
    labels_cpu = labels.detach().cpu()
    rows = []
    stable_subject_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
    for node_position in test_indices.detach().cpu().tolist():
        subject = int(stable_subject_indices[node_position])
        probability = probabilities_cpu[node_position]
        prediction = int(torch.argmax(probability).item())
        row = {
            "fold": int(fold),
            "subject_index": int(subject),
            "truth": int(labels_cpu[node_position].item()),
            "prediction": prediction,
        }
        for index, name in enumerate(CLASS_ORDER):
            row[f"logit_{name}"] = float(logits_cpu[node_position, index].item())
            row[f"probability_{name}"] = float(probability[index].item())
            row[f"a012_raw_logit_{name}"] = float(a012_cpu[node_position, index].item())
            row[f"mapped_c1_raw_logit_{name}"] = float(mapped_cpu[node_position, index].item())
        rows.append(row)
    return rows


def metrics_from_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probability = np.asarray(
        [[float(row[f"probability_{name}"]) for name in CLASS_ORDER] for row in rows],
        dtype=np.float64,
    )
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    require(np.array_equal(prediction, probability.argmax(axis=1)), "Saved prediction is not direct coupled-probability argmax")
    return metric_payload_from_arrays(truth, probability)


def require_rows_match(actual: list[dict], expected: list[dict], context_text: str) -> None:
    actual = sorted(actual, key=lambda row: int(row["subject_index"]))
    expected = sorted(expected, key=lambda row: int(row["subject_index"]))
    require(len(actual) == len(expected), f"{context_text}: row count changed")
    for new, old in zip(actual, expected):
        for key in ("fold", "subject_index", "truth", "prediction"):
            require(int(new[key]) == int(old[key]), f"{context_text}: {key} changed")
        for prefix in ("logit", "probability", "a012_raw_logit", "mapped_c1_raw_logit"):
            difference = max(
                abs(float(new[f"{prefix}_{name}"]) - float(old[f"{prefix}_{name}"]))
                for name in CLASS_ORDER
            )
            require(difference <= 1e-7, f"{context_text}: {prefix} changed by {difference}")


def state_storage(state: dict[str, torch.Tensor]) -> dict:
    tensor_elements = int(sum(value.numel() for value in state.values()))
    tensor_bytes = int(sum(value.numel() * value.element_size() for value in state.values()))
    floating_elements = int(
        sum(value.numel() for value in state.values() if torch.is_floating_point(value))
    )
    return {
        "state_tensor_elements": tensor_elements,
        "state_floating_elements": floating_elements,
        "state_tensor_bytes": tensor_bytes,
    }


def build_fresh_artifact(context: dict, state: dict[str, torch.Tensor]) -> RABMGA012:
    # Construct the exact A012 object (including dropout_multiplier=1.1), then
    # let the enclosing RA state_dict supply all backbone and mapped buffers.
    seed_model, criterion, optimizer, scheduler, dropout, groups = broad.make_training_objects(
        context, A012_SPEC
    )
    del criterion, optimizer, scheduler, groups
    require(len(dropout) == 10, "Fresh A012 dropout audit changed")
    require(
        sorted(round(float(item["final_p"]), 12) for item in dropout)
        == sorted([round(0.335 * 1.1, 12)] * 9 + [round(0.67 * 1.1, 12)]),
        "Fresh A012 dropout_multiplier changed",
    )
    backbone = seed_model
    classifier = backbone.GCN.classifier
    wrapper = RABMGA012(
        backbone,
        torch.zeros(
            (3, classifier.in_features),
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        ),
        torch.zeros(3, device=classifier.bias.device, dtype=classifier.bias.dtype),
    )
    wrapper.load_state_dict(state, strict=True)
    wrapper.eval()
    require(
        wrapper.mapped_c1_weight.device == classifier.weight.device
        and wrapper.mapped_c1_weight.dtype == classifier.weight.dtype
        and wrapper.mapped_c1_bias.device == classifier.bias.device
        and wrapper.mapped_c1_bias.dtype == classifier.bias.dtype,
        "Reloaded mapped buffers do not match A012 classifier device/dtype",
    )
    return wrapper


def audit_artifact_state(wrapper: RABMGA012) -> dict:
    state = wrapper.state_dict()
    keys = set(state)
    require("mapped_c1_weight" in keys and "mapped_c1_bias" in keys, "Mapped C1 buffers missing")
    require(
        all(key.startswith("backbone.") or key in {"mapped_c1_weight", "mapped_c1_bias"} for key in keys),
        "Artifact state contains a second model or unfurled mapping",
    )
    forbidden_mapping_keys = {
        "transform",
        "offset",
        "alignment_transform",
        "alignment_offset",
    }
    require(
        not any("c1_backbone" in key or key in forbidden_mapping_keys for key in keys),
        "Artifact retained forbidden C1/map state",
    )
    parameter_count = int(sum(parameter.numel() for parameter in wrapper.parameters()))
    mapped_coefficients = int(wrapper.mapped_head_coefficients)
    require(parameter_count == EXPECTED_PARAMETERS, "Artifact A012 parameter count changed")
    require(mapped_coefficients == EXPECTED_MAPPED_COEFFICIENTS, "Mapped head coefficient count changed")
    require(len(list(wrapper.children())) == 1 and wrapper.backbone is next(iter(wrapper.children())), "Artifact contains more than one backbone module")
    return {
        "a012_parameter_count": parameter_count,
        "mapped_head_buffer_coefficients": mapped_coefficients,
        "mapped_head_buffer_bytes": int(
            wrapper.mapped_c1_weight.numel() * wrapper.mapped_c1_weight.element_size()
            + wrapper.mapped_c1_bias.numel() * wrapper.mapped_c1_bias.element_size()
        ),
        "inference_parameter_plus_mapped_coefficient_count": parameter_count + mapped_coefficients,
        **state_storage(state),
        "state_key_count": len(state),
        "state_schema": "backbone.* plus mapped_c1_weight/mapped_c1_bias only",
        "contains_c1_backbone": False,
        "contains_unfolded_transform_or_offset": False,
    }


def construct_fold(
    context: dict,
    prepared: dict,
    runtime: dict,
    fold: int,
    artifact_path: Path,
    require_repeat_determinism: bool,
) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    labels = context["dataset_data"]["Label"]
    features = context["dataset_data"]["Feature"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool(torch.any(train_mask & test_mask)), f"Fold{fold} train/test overlap")
    require(bool(torch.all(train_mask | test_mask)), f"Fold{fold} train/test partition incomplete")
    train_indices, test_indices = torch.where(train_mask)[0], torch.where(test_mask)[0]

    c1_model, a012_model = load_fold_models(context, prepared, fold)
    c1_logits, hidden_c, c1_hook_calls = forward_with_hidden(c1_model, features)
    a012_logits, hidden_a, a012_hook_calls = forward_with_hidden(a012_model, features)
    deterministic = {"checked": False}
    if require_repeat_determinism:
        c1_again, hidden_c_again, _ = forward_with_hidden(c1_model, features)
        a012_again, hidden_a_again, _ = forward_with_hidden(a012_model, features)
        deterministic = {
            "checked": True,
            "c1_logit_max_abs_diff": float((c1_again - c1_logits).abs().max().item()),
            "c1_hidden_max_abs_diff": float((hidden_c_again - hidden_c).abs().max().item()),
            "a012_logit_max_abs_diff": float((a012_again - a012_logits).abs().max().item()),
            "a012_hidden_max_abs_diff": float((hidden_a_again - hidden_a).abs().max().item()),
        }
        require(max(value for key, value in deterministic.items() if key.endswith("diff")) == 0.0, "Eval representation extraction is not deterministic")

    references = reference_paths()
    c1_oof = indexed_rows(references["c1_oof"], "C1")
    a012_oof = indexed_rows(references["a012_oof"], "A012")
    checkpoint_readback = {
        "c1": compare_historical_checkpoint_rows(context, fold, c1_logits, c1_oof, "C1"),
        "a012": compare_historical_checkpoint_rows(context, fold, a012_logits, a012_oof, "A012"),
    }
    require(hidden_a.shape == hidden_c.shape == (598, 48), "C1/A012 hidden dimension mismatch")

    # Crucially, only train slices and train labels cross this function boundary.
    alignment = fit_weighted_alignment(
        hidden_a.index_select(0, train_indices),
        hidden_c.index_select(0, train_indices),
        labels.index_select(0, train_indices),
    )
    mapped_weight64, mapped_bias64, algebra_error = fold_classifier(
        alignment, c1_model.GCN.classifier, hidden_a.index_select(0, train_indices)
    )
    mapped_train64 = torch.nn.functional.linear(
        hidden_a.index_select(0, train_indices).detach().cpu().to(torch.float64),
        mapped_weight64,
        mapped_bias64,
    )
    diagnostics = alignment_diagnostics(
        c1_logits.index_select(0, train_indices), mapped_train64, alignment, algebra_error
    )

    # C1 is deliberately destroyed before the artifact wrapper and inference exist.
    del c1_model, hidden_c
    torch.cuda.empty_cache()
    a012_dropout_profile = a012_model._ra_bmg_dropout_audit
    classifier_device = a012_model.GCN.classifier.weight.device
    classifier_dtype = a012_model.GCN.classifier.weight.dtype
    mapped_weight_for_artifact = mapped_weight64.to(
        device=classifier_device, dtype=classifier_dtype
    )
    mapped_bias_for_artifact = mapped_bias64.to(
        device=classifier_device, dtype=classifier_dtype
    )
    wrapper = RABMGA012(
        a012_model, mapped_weight_for_artifact, mapped_bias_for_artifact
    )
    wrapper.eval()
    require(
        wrapper.mapped_c1_weight.device == classifier_device
        and wrapper.mapped_c1_weight.dtype == classifier_dtype
        and wrapper.mapped_c1_bias.device == classifier_device
        and wrapper.mapped_c1_bias.dtype == classifier_dtype,
        "Mapped head device/dtype does not match A012 classifier",
    )
    storage = audit_artifact_state(wrapper)
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.no_grad():
        logits_out = wrapper(features)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    require(wrapper.classifier_hook_calls == 1, "Artifact did not use exactly one A012 backbone/classifier pass")
    probabilities = torch.softmax(logits_out, dim=-1)
    require(wrapper.last_probabilities is not None and wrapper.last_a012_logits is not None and wrapper.last_mapped_c1_logits is not None, "Artifact diagnostics missing")
    probability_row_sum_error = float((probabilities.sum(dim=-1) - 1.0).abs().max().item())
    softmax_roundtrip_error = float((probabilities - wrapper.last_probabilities).abs().max().item())
    require(probability_row_sum_error < 1e-7, f"Probability row-sum error {probability_row_sum_error}")
    require(softmax_roundtrip_error < 1e-7, f"log(probability) roundtrip error {softmax_roundtrip_error}")
    odds_error = float(
        (
            torch.log(probabilities[:, 0] / probabilities[:, 2])
            - (
                wrapper.last_a012_logits[:, 0].to(torch.float64)
                - wrapper.last_a012_logits[:, 2].to(torch.float64)
            )
        )
        .abs()
        .max()
        .item()
    )
    require(odds_error <= 1e-6, f"A012 AD:SMCI conditional odds changed by {odds_error}")
    require(float((wrapper.last_a012_logits - a012_logits).abs().max().item()) == 0.0, "Wrapper changed A012 raw logits")
    expected_mapped32 = torch.nn.functional.linear(
        wrapper._classifier_input,
        wrapper.mapped_c1_weight,
        wrapper.mapped_c1_bias,
    )
    mapped_float32_error = float((expected_mapped32 - wrapper.last_mapped_c1_logits).abs().max().item())
    require(mapped_float32_error == 0.0, "Wrapper mapped-head float32 inference changed")

    rows = direct_probability_rows(
        fold,
        test_indices,
        labels,
        logits_out,
        probabilities,
        wrapper.last_a012_logits,
        wrapper.last_mapped_c1_logits,
        context["dataset_dict"],
    )
    expected_fold = context["fold_manifest"]["folds"][fold]
    expected_subject_truth = dict(
        zip(expected_fold["test_subject_indices"], expected_fold["test_truth"])
    )
    require(
        {int(row["subject_index"]): int(row["truth"]) for row in rows}
        == {int(subject): int(truth) for subject, truth in expected_subject_truth.items()},
        f"Fold{fold} stable subject/truth manifest changed",
    )
    lock = fold_lock(runtime, prepared, fold)
    artifact_state = state_to_cpu(wrapper.state_dict())
    payload = {
        "schema": "ra_bmg_a012_v1_single_backbone_artifact",
        "fold_lock": lock,
        "model_state": artifact_state,
        "class_order": list(CLASS_ORDER),
        "a012_spec": A012_SPEC,
        "storage": storage,
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "one_backbone_forward": True,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(artifact_path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, artifact_path)
    artifact_sha256 = sha256_file(artifact_path)
    artifact_bytes = artifact_path.stat().st_size

    # A strict fresh-object load is the final single-model checkpoint contract.
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    require(loaded["fold_lock"] == lock and loaded["schema"] == payload["schema"], "Artifact metadata changed after save")
    fresh = build_fresh_artifact(context, loaded["model_state"])
    fresh_storage = audit_artifact_state(fresh)
    require(fresh_storage == storage, "Artifact storage schema changed after reload")
    with torch.no_grad():
        reloaded_logits = fresh(features)
    reload_max_abs_diff = float((reloaded_logits - logits_out).abs().max().item())
    require(reload_max_abs_diff == 0.0 and fresh.classifier_hook_calls == 1, "Reloaded artifact output changed")

    summary = {
        "fold": int(fold),
        "fold_lock": lock,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "mapping_fit_subjects": int(train_mask.sum().item()),
        "mapping_fit_uses_test_labels": False,
        "hidden_dimension": int(hidden_a.shape[1]),
        "classifier_hook": "backbone.GCN.classifier forward_pre_hook",
        "a012_dropout_profile": a012_dropout_profile,
        "source_hook_calls": {"c1": c1_hook_calls, "a012": a012_hook_calls},
        "checkpoint_oof_readback_max_abs_diff": checkpoint_readback,
        "deterministic_extraction": deterministic,
        "alignment_diagnostics_train_only": diagnostics,
        "probability_row_sum_max_abs_error": probability_row_sum_error,
        "softmax_log_probability_roundtrip_max_abs_diff": softmax_roundtrip_error,
        "a012_ad_smci_log_odds_max_abs_error_full_batch": odds_error,
        "mapped_head_float32_linear_max_abs_diff": mapped_float32_error,
        "artifact_reload_max_abs_diff": reload_max_abs_diff,
        "storage": {**storage, "artifact_file_bytes": artifact_bytes},
        "artifact_sha256": artifact_sha256,
        "inference_seconds_one_full_batch": float(inference_seconds),
        "construction_seconds": float(time.perf_counter() - started),
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "one_backbone_forward": True,
        "c1_backbone_removed_before_artifact_inference": True,
    }
    del wrapper, fresh, logits_out, reloaded_logits
    torch.cuda.empty_cache()
    return summary, rows


def source_gate(prepared: dict) -> None:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong source branch")
    head = git("rev-parse", "HEAD").stdout.strip()
    require(head != prepared["base_commit"], "Implementation has not been committed after prepare")
    tracked_source = tuple(prepared["source_dependencies"]) + (
        "experiments/ra_bmg_a012_v1/experiment_config.json",
    )
    for relative in tracked_source:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        require(tracked.returncode == 0, f"Source file is not tracked in HEAD: {relative}")
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT
        )
        require(unchanged.returncode == 0, f"Tracked source differs from HEAD: {relative}")
    allowed_new_or_changed = {
        ".gitignore",
        "Model/ra_bmg.py",
        "scripts/run_ra_bmg_a012_v1.py",
        "experiments/ra_bmg_a012_v1/experiment_config.json",
    }
    for relative in tracked_source:
        if relative in allowed_new_or_changed:
            continue
        historical_unchanged = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                prepared["base_commit"],
                "HEAD",
                "--",
                relative,
            ],
            cwd=ROOT,
        )
        require(
            historical_unchanged.returncode == 0,
            f"Locked historical dependency changed from base: {relative}",
        )
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    require(staged.returncode == 0, "Git index is dirty")
    require(prepared["source_dependencies"] == source_dependencies(), "Source dependency changed after prepare")


def run_smoke(context: dict, output_root: Path, prepared: dict) -> dict:
    source_gate(prepared)
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke directory: {smoke_root}")
    smoke_root.mkdir(parents=True)
    runtime = make_runtime_config(prepared, context)
    write_json(smoke_root / "smoke_config.json", runtime)
    summary, rows = construct_fold(
        context,
        prepared,
        runtime,
        fold=0,
        artifact_path=smoke_root / "artifact.pt",
        require_repeat_determinism=True,
    )
    require(summary["hidden_dimension"] == 48, "Smoke hidden dimension mismatch")
    require(summary["probability_row_sum_max_abs_error"] < 1e-7, "Smoke simplex gate failed")
    require(summary["a012_ad_smci_log_odds_max_abs_error_full_batch"] <= 1e-6, "Smoke conditional-odds gate failed")
    require(summary["artifact_reload_max_abs_diff"] == 0.0, "Smoke artifact reload gate failed")
    require(summary["c1_backbone_removed_before_artifact_inference"], "Smoke retained C1 backbone")
    smoke_metrics = metrics_from_rows(rows)
    summary["smoke_metrics_non_gate"] = smoke_metrics
    report = {
        "passed": True,
        "source_commit": runtime["source_commit"],
        "runtime_config_sha256": runtime["config_sha256"],
        "fold": 0,
        "checkpoint_subject_hidden_mapping_probability_artifact_gates": "PASS",
        "fold_accuracy_is_not_a_gate": True,
        "fold_summary": summary,
        "oof_rows": len(rows),
    }
    write_json(smoke_root / "smoke_report.json", report)
    print(
        f"SMOKE PASS fold0 correct={smoke_metrics['correct']}/{summary['test_size']} "
        f"odds_error={summary['a012_ad_smci_log_odds_max_abs_error_full_batch']:.3g}",
        flush=True,
    )
    return report


def validate_completed_fold(
    context: dict,
    prepared: dict,
    runtime: dict,
    fold: int,
    fold_dir: Path,
):
    if not fold_dir.exists():
        return None
    required = {
        "artifact.pt",
        "summary.json",
        "oof_predictions.csv",
        "config.json",
        "complete.json",
    }
    require({path.name for path in fold_dir.iterdir()} == required, f"Fold{fold} completed artifact set changed")
    lock = fold_lock(runtime, prepared, fold)
    require(json.loads((fold_dir / "config.json").read_text(encoding="utf-8")) == lock, f"Fold{fold} config lock changed")
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary["fold_lock"] == lock, f"Fold{fold} summary lock changed")
    artifact_path = fold_dir / "artifact.pt"
    marker_expected = {
        "complete": True,
        **lock,
        "artifact_sha256": sha256_file(artifact_path),
        "summary_sha256": sha256_file(fold_dir / "summary.json"),
        "oof_sha256": sha256_file(fold_dir / "oof_predictions.csv"),
    }
    require(json.loads((fold_dir / "complete.json").read_text(encoding="utf-8")) == marker_expected, f"Fold{fold} marker/file hashes changed")
    require(sha256_file(artifact_path) == summary["artifact_sha256"], f"Fold{fold} artifact hash changed")
    require(
        summary["fold"] == fold
        and summary["mapping_fit_uses_test_labels"] is False
        and summary["train_size"] + summary["test_size"] == 598
        and summary["hidden_dimension"] == 48
        and summary["a012_ad_smci_log_odds_max_abs_error_full_batch"] <= 1e-6
        and summary["probability_row_sum_max_abs_error"] < 1e-7
        and summary["softmax_log_probability_roundtrip_max_abs_diff"] < 1e-7
        and summary["artifact_reload_max_abs_diff"] == 0.0
        and summary["c1_backbone_removed_before_artifact_inference"] is True,
        f"Fold{fold} mechanism gates changed",
    )
    require(
        summary["source_hook_calls"] == {"c1": 1, "a012": 1}
        and summary["alignment_diagnostics_train_only"]["svd_finite"]
        and summary["alignment_diagnostics_train_only"]["variance_scale_finite"],
        f"Fold{fold} hook/alignment diagnostics changed",
    )
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    require(payload["fold_lock"] == lock and payload["schema"] == "ra_bmg_a012_v1_single_backbone_artifact", f"Fold{fold} artifact lock/schema changed")
    require(
        payload["storage"]
        == {
            key: value
            for key, value in summary["storage"].items()
            if key != "artifact_file_bytes"
        }
        and payload["single_model"] is True
        and payload["ensemble"] is False
        and payload["distillation"] is False
        and payload["one_backbone_forward"] is True,
        f"Fold{fold} artifact mechanism/storage metadata changed",
    )
    wrapper = build_fresh_artifact(context, payload["model_state"])
    require(audit_artifact_state(wrapper) == {key: value for key, value in summary["storage"].items() if key != "artifact_file_bytes"}, f"Fold{fold} storage audit changed")
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    test_indices = torch.where(test_mask)[0]
    with torch.no_grad():
        logits_out = wrapper(features)
    probabilities = torch.softmax(logits_out, dim=-1)
    rows = direct_probability_rows(
        fold,
        test_indices,
        labels,
        logits_out,
        probabilities,
        wrapper.last_a012_logits,
        wrapper.last_mapped_c1_logits,
        context["dataset_dict"],
    )
    saved_rows = read_csv(fold_dir / "oof_predictions.csv")
    require_rows_match(rows, saved_rows, f"Fold{fold} artifact OOF readback")
    expected_fold = context["fold_manifest"]["folds"][fold]
    expected = {
        int(subject): int(truth)
        for subject, truth in zip(expected_fold["test_subject_indices"], expected_fold["test_truth"])
    }
    require({int(row["subject_index"]): int(row["truth"]) for row in saved_rows} == expected, f"Fold{fold} stable manifest changed")
    del wrapper
    torch.cuda.empty_cache()
    print(f"RESUME fold={fold} artifact/OOF strict readback PASS", flush=True)
    return summary, saved_rows


def run_formal_fold(
    context: dict,
    output_root: Path,
    prepared: dict,
    runtime: dict,
    fold: int,
):
    folds_root = output_root / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)
    final_dir = folds_root / f"fold_{fold:02d}"
    completed = validate_completed_fold(context, prepared, runtime, fold, final_dir)
    if completed is not None:
        return completed
    staging = folds_root / f".fold_{fold:02d}_staging"
    require(staging.parent.resolve() == folds_root.resolve() and staging.name == f".fold_{fold:02d}_staging", "Unsafe staging path")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    lock = fold_lock(runtime, prepared, fold)
    write_json(staging / "config.json", lock)
    summary, rows = construct_fold(
        context,
        prepared,
        runtime,
        fold,
        staging / "artifact.pt",
        require_repeat_determinism=False,
    )
    write_json(staging / "summary.json", summary)
    write_csv(staging / "oof_predictions.csv", rows)
    marker = {
        "complete": True,
        **lock,
        "artifact_sha256": sha256_file(staging / "artifact.pt"),
        "summary_sha256": sha256_file(staging / "summary.json"),
        "oof_sha256": sha256_file(staging / "oof_predictions.csv"),
    }
    write_json(staging / "complete.json", marker)
    require(not final_dir.exists(), f"Refusing to overwrite completed fold{fold}")
    os.replace(staging, final_dir)
    validated = validate_completed_fold(context, prepared, runtime, fold, final_dir)
    require(validated is not None, f"Fold{fold} completion readback failed")
    print(f"FORMAL fold={fold} artifact/OOF complete", flush=True)
    return validated


def aggregate_diagnostics(fold_summaries: list[dict]) -> dict:
    keys = (
        "cn_margin_pearson",
        "cn_margin_sign_agreement",
        "cn_margin_normalized_rmse",
        "full_c1_logits_mean_cosine",
        "svd_singular_min",
        "svd_singular_max",
        "variance_scale_min",
        "variance_scale_max",
    )
    result = {
        "definitions": {
            "cn_margin": "z_CN-logsumexp([z_AD,z_SMCI]) on fold-train rows",
            "normalized_rmse": "unweighted RMSE/(population std(original C1 train margin)+1e-8)",
            "full_logit_cosine": "unweighted train-row per-subject cosine mean",
        },
        "folds": [
            {"fold": summary["fold"], **summary["alignment_diagnostics_train_only"]}
            for summary in fold_summaries
        ],
    }
    for key in keys:
        values = [float(summary["alignment_diagnostics_train_only"][key]) for summary in fold_summaries]
        result[key] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    result["all_svd_and_variance_scales_finite"] = all(
        summary["alignment_diagnostics_train_only"]["svd_finite"]
        and summary["alignment_diagnostics_train_only"]["variance_scale_finite"]
        for summary in fold_summaries
    )
    result["any_test_label_used_for_mapping"] = any(
        summary["mapping_fit_uses_test_labels"] for summary in fold_summaries
    )
    return result


def decision(metrics: dict, comparison_a012: dict, boundary: dict) -> str:
    success = (
        metrics["correct"] >= 564
        and boundary["AD_CN"] == 0
        and metrics["bacc"] >= BACC_FLOOR
        and comparison_a012["net_repairs"] >= 2
        and boundary["AD_SMCI"] <= 17
        and boundary["CN_SMCI"] <= 19
    )
    weak = (
        metrics["correct"] == 563
        and comparison_a012["repairs"] > comparison_a012["damages"]
        and boundary["AD_CN"] == 0
        and metrics["bacc"] >= BACC_FLOOR
    )
    if success:
        return "RA_BMG_SUCCESS"
    if weak:
        return "RA_BMG_WEAK_GAIN"
    return "RA_BMG_NO_GAIN"


def aggregate_formal(
    context: dict,
    output_root: Path,
    prepared: dict,
    runtime: dict,
    fold_summaries: list[dict],
    fold_rows: list[list[dict]],
    formal_wall_seconds: float,
) -> dict:
    rows = sorted(sum(fold_rows, []), key=lambda row: int(row["subject_index"]))
    require(len(rows) == 598 and [int(row["subject_index"]) for row in rows] == list(range(598)), "Formal OOF is not 598 unique stable subjects")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "Formal OOF fold coverage changed")
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[f"probability_{name}"]) for name in CLASS_ORDER] for row in rows],
        dtype=np.float64,
    )
    prediction = probabilities.argmax(axis=1)
    require(np.array_equal(prediction, np.asarray([int(row["prediction"]) for row in rows])), "Formal OOF prediction changed")
    metrics = metric_payload_from_arrays(truth, probabilities)
    # Per-fold labels are first scored only after the complete 598-row OOF has
    # been assembled and validated above.
    fold_metric_payloads = [metrics_from_rows(rows_for_fold) for rows_for_fold in fold_rows]
    boundary = boundary_payload(truth, prediction)
    predicted_counts = {
        name: int((prediction == index).sum()) for index, name in enumerate(CLASS_ORDER)
    }

    refs = reference_paths()
    c1_indexed = indexed_rows(refs["c1_oof"], "C1")
    a012_indexed = indexed_rows(refs["a012_oof"], "A012")
    c1_probability = row_probabilities(c1_indexed)
    a012_probability = row_probabilities(a012_indexed)
    require(
        all(
            int(row["truth"]) == int(c1_indexed[int(row["subject_index"])]["truth"])
            and int(row["fold"]) == int(c1_indexed[int(row["subject_index"])]["fold"])
            for row in rows
        ),
        "Formal OOF/reference subject alignment changed",
    )
    comparison_a012 = paired_payload(truth, probabilities, a012_probability)
    comparison_c1 = paired_payload(truth, probabilities, c1_probability)
    c1_prediction = c1_probability.argmax(axis=1)
    cn_prediction_agreement = float(np.mean((prediction == 1) == (c1_prediction == 1)))
    odds_errors = []
    for row in rows:
        lhs = math.log(float(row["probability_AD"]) / float(row["probability_SMCI"]))
        rhs = float(row["a012_raw_logit_AD"]) - float(row["a012_raw_logit_SMCI"])
        odds_errors.append(abs(lhs - rhs))
    odds_error = float(max(odds_errors))
    require(odds_error <= 1e-6, f"OOF A012 AD:SMCI odds changed by {odds_error}")

    fold_acc = [float(item["acc"]) for item in fold_metric_payloads]
    diagnostics = aggregate_diagnostics(fold_summaries)
    require(not diagnostics["any_test_label_used_for_mapping"], "A mapping used test labels")
    final_decision = decision(metrics, comparison_a012, boundary)
    artifacts = [
        {
            "fold": summary["fold"],
            "path": f"folds/fold_{summary['fold']:02d}/artifact.pt",
            "sha256": summary["artifact_sha256"],
            "bytes": summary["storage"]["artifact_file_bytes"],
            "fold_lock": summary["fold_lock"],
        }
        for summary in fold_summaries
    ]
    storage = {
        "a012_parameter_count": EXPECTED_PARAMETERS,
        "mapped_head_buffer_coefficients": EXPECTED_MAPPED_COEFFICIENTS,
        "mapped_head_buffer_bytes": fold_summaries[0]["storage"]["mapped_head_buffer_bytes"],
        "inference_parameter_plus_mapped_coefficient_count": EXPECTED_PARAMETERS
        + EXPECTED_MAPPED_COEFFICIENTS,
        "state_tensor_elements_per_fold": fold_summaries[0]["storage"]["state_tensor_elements"],
        "state_tensor_bytes_per_fold": fold_summaries[0]["storage"]["state_tensor_bytes"],
        "artifact_file_bytes_per_fold": [item["bytes"] for item in artifacts],
        "artifact_file_bytes_total": int(sum(item["bytes"] for item in artifacts)),
    }
    summary = {
        "experiment": EXPERIMENT,
        "decision": final_decision,
        "metrics": metrics,
        "predicted_class_counts": predicted_counts,
        "boundary_errors": boundary,
        "fold_acc_mean": float(np.mean(fold_acc)),
        "fold_acc_sample_std": float(np.std(fold_acc, ddof=1)),
        "folds": [
            {
                "fold": item["fold"],
                "correct": fold_metric_payloads[index]["correct"],
                "acc": fold_metric_payloads[index]["acc"],
                "construction_seconds": item["construction_seconds"],
                "inference_seconds_one_full_batch": item["inference_seconds_one_full_batch"],
            }
            for index, item in enumerate(fold_summaries)
        ],
        "comparison_vs_a012": comparison_a012,
        "comparison_vs_c1": comparison_c1,
        "a012_ad_smci_conditional_log_odds_max_abs_error": odds_error,
        "ra_bmg_cn_prediction_vs_c1_cn_prediction_agreement": cn_prediction_agreement,
        "alignment_diagnostics": diagnostics,
        "storage": storage,
        "mechanism": {
            "single_model": True,
            "ensemble": False,
            "distillation": False,
            "one_backbone_forward": True,
            "backbone": "A012 only",
            "mapped_c1_head": "persistent frozen buffers folded into A012 hidden coordinates",
            "mapping_fit_rows": "fold train only",
            "test_labels_used_for_mapping": False,
            "graph": False,
            "direct_probability_evaluation_without_second_logit_adjustment": True,
        },
        "b1_prerequisite": prepared["reference"],
        "c1_anchor": C1_ANCHOR,
        "a012_anchor": A012_ANCHOR,
        "source_commit": runtime["source_commit"],
        "result_commit": "SELF_REFERENCE_NOT_EMBEDDED; use the Git commit tracking this summary",
        "branch": BRANCH,
        "runtime_config": runtime,
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "cpu": platform.processor() or platform.machine(),
        "formal_wall_seconds_this_invocation": float(formal_wall_seconds),
        "construction_seconds_total": float(
            sum(item["construction_seconds"] for item in fold_summaries)
        ),
        "inference_seconds_total_one_forward_per_fold": float(
            sum(item["inference_seconds_one_full_batch"] for item in fold_summaries)
        ),
        "artifacts": artifacts,
        "retained_model": "RA-BMG-A012" if final_decision == "RA_BMG_SUCCESS" else "A012",
        "next_recommendation": (
            "success-only ablation: unaligned graft vs aligned full graft vs aligned boundary-selective graft"
            if final_decision == "RA_BMG_SUCCESS"
            else "DB-MoLA-A012: dual-boundary low-rank feature experts"
        ),
        "reproduction_command": f'"{sys.executable}" -u -B scripts/run_ra_bmg_a012_v1.py formal --device cuda:0',
    }
    write_csv(output_root / "oof_predictions.csv", rows)
    write_json(output_root / "fold_metrics.json", summary["folds"])
    write_json(output_root / "mapping_diagnostics.json", diagnostics)
    write_json(output_root / "artifacts_manifest.json", artifacts)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "formal_config.json", runtime)
    (output_root / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict) -> str:
    metrics = summary["metrics"]
    boundary = summary["boundary_errors"]
    alignment = summary["alignment_diagnostics"]
    comparison_a012 = summary["comparison_vs_a012"]
    comparison_c1 = summary["comparison_vs_c1"]
    fold_text = ", ".join(
        f"{item['fold']}:{item['correct']}/{item['acc']:.7f}" for item in summary["folds"]
    )
    success = summary["decision"] == "RA_BMG_SUCCESS"
    margin_direction = alignment["cn_margin_sign_agreement"]["mean"]
    lines = [
        "# RA-BMG-A012 v1",
        "",
        f"Decision: `{summary['decision']}`. Retained model: `{summary['retained_model']}`.",
        "",
        "## Formal result",
        "",
        (
            f"Correct={metrics['correct']}/598; ACC={metrics['acc']:.7f}; Macro-F1={metrics['macro_f1']:.7f}; "
            f"BACC={metrics['bacc']:.7f}; Probability Macro-AUC={metrics['macro_auc']:.7f}; "
            f"Weighted-F1={metrics['weighted_f1']:.7f}; confusion={metrics['confusion_matrix']}; "
            f"predicted={summary['predicted_class_counts']}."
        ),
        (
            f"AD->sMCI={boundary['AD_to_SMCI']}; sMCI->AD={boundary['SMCI_to_AD']}; "
            f"CN->sMCI={boundary['CN_to_SMCI']}; sMCI->CN={boundary['SMCI_to_CN']}; "
            f"AD-CN={boundary['AD_CN']}; AD-sMCI={boundary['AD_SMCI']}; CN-sMCI={boundary['CN_SMCI']}."
        ),
        f"Fold ACC mean +/- sample SD={summary['fold_acc_mean']:.7f} +/- {summary['fold_acc_sample_std']:.7f}; folds (Correct/ACC): {fold_text}.",
        f"Versus A012 repairs/damages/changed={comparison_a012['repairs']}/{comparison_a012['damages']}/{comparison_a012['changed']} (net={comparison_a012['net_repairs']}); sources={comparison_a012['repair_sources_from_baseline_errors']}/{comparison_a012['damage_sources_in_candidate_errors']}.",
        f"Versus C1 repairs/damages/changed={comparison_c1['repairs']}/{comparison_c1['damages']}/{comparison_c1['changed']} (net={comparison_c1['net_repairs']}).",
        "",
        "## Fixed mechanism diagnostics",
        "",
        f"Train-only mapped C1 CN-margin Pearson mean={alignment['cn_margin_pearson']['mean']:.7f}; sign agreement mean={margin_direction:.7f}; normalized RMSE mean={alignment['cn_margin_normalized_rmse']['mean']:.7f}; full-logit cosine mean={alignment['full_c1_logits_mean_cosine']['mean']:.7f}.",
        f"All SVD/scales finite={alignment['all_svd_and_variance_scales_finite']}; any test label used for mapping={alignment['any_test_label_used_for_mapping']}.",
        f"A012 AD:sMCI conditional log-odds max error={summary['a012_ad_smci_conditional_log_odds_max_abs_error']:.3g}; RA-BMG/C1 CN-prediction agreement={summary['ra_bmg_cn_prediction_vs_c1_cn_prediction_agreement']:.7f}.",
        f"Storage: A012 parameters={summary['storage']['a012_parameter_count']}; mapped head buffers={summary['storage']['mapped_head_buffer_coefficients']} coefficients/{summary['storage']['mapped_head_buffer_bytes']} bytes; inference coefficients={summary['storage']['inference_parameter_plus_mapped_coefficient_count']}; state tensor bytes/fold={summary['storage']['state_tensor_bytes_per_fold']}; artifact bytes total={summary['storage']['artifact_file_bytes_total']}.",
        f"Construction time={summary['construction_seconds_total']:.3f}s; one-forward inference time sum={summary['inference_seconds_total_one_forward_per_fold']:.6f}s; GPU={summary['gpu']}; CPU={summary['cpu']}.",
        "",
        "## Required answers",
        "",
        f"1. Closed-form alignment was measured without model selection: Pearson={alignment['cn_margin_pearson']['mean']:.7f}, sign agreement={margin_direction:.7f}, NRMSE={alignment['cn_margin_normalized_rmse']['mean']:.7f}.",
        f"2. Mapped C1 CN-margin direction agreement was {margin_direction:.7f} on train rows.",
        f"3. A012 AD:sMCI conditional odds were strictly preserved (max error {summary['a012_ad_smci_conditional_log_odds_max_abs_error']:.3g}).",
        f"4. RA-BMG {'reached' if metrics['correct'] >= 564 else 'did not reach'} 564/598 (actual {metrics['correct']}/598).",
        f"5. Boundary exchange is shown by repairs/damages sources above; CN-sMCI ended at {boundary['CN_SMCI']} and AD-sMCI at {boundary['AD_SMCI']}.",
        f"6. AD-CN errors={boundary['AD_CN']}.",
        "7. Final inference uses one A012 backbone; C1 is absent from every artifact state.",
        "8. No test label was used to fit T, c, variance scales, or the mapped head.",
        f"9. Compared with raw weight soup=502, representation alignment plus boundary-selective coupling produced {metrics['correct']}/598 and avoided that collapse.",
        f"10. Final retention: {summary['retained_model']}.",
        "",
        "## Reproduction and Git",
        "",
        f"Source commit=`{summary['source_commit']}`; result commit={summary['result_commit']}; branch=`{summary['branch']}`.",
        "",
        "```text",
        summary["reproduction_command"],
        "```",
        "",
        f"Next recommendation: `{summary['next_recommendation']}`. It was not implemented or run.",
        "",
    ]
    if success:
        lines.extend(
            [
                "Success-only future ablation (not run): A012; raw soup; unaligned graft; aligned full graft; aligned boundary-selective graft.",
                "",
            ]
        )
    return "\n".join(lines)


def run_formal(context: dict, output_root: Path, prepared: dict) -> dict:
    source_gate(prepared)
    formal_started = time.perf_counter()
    runtime = make_runtime_config(prepared, context)
    smoke_path = output_root / "smoke" / "smoke_report.json"
    smoke_config_path = output_root / "smoke" / "smoke_config.json"
    require(smoke_path.is_file() and smoke_config_path.is_file(), "Formal requires a completed smoke")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    require(
        smoke["passed"]
        and smoke["source_commit"] == runtime["source_commit"]
        and smoke["runtime_config_sha256"] == runtime["config_sha256"],
        "Smoke source/config gate changed",
    )
    require(smoke_config == runtime, "Smoke runtime/source dependency hashes changed")
    smoke_summary = smoke["fold_summary"]
    smoke_artifact_path = output_root / "smoke" / "artifact.pt"
    require(
        smoke_artifact_path.is_file()
        and sha256_file(smoke_artifact_path) == smoke_summary["artifact_sha256"],
        "Smoke artifact hash changed",
    )
    require(
        smoke_summary["fold"] == 0
        and smoke_summary["fold_lock"] == fold_lock(runtime, prepared, 0)
        and smoke_summary["mapping_fit_uses_test_labels"] is False
        and smoke_summary["probability_row_sum_max_abs_error"] < 1e-7
        and smoke_summary["a012_ad_smci_log_odds_max_abs_error_full_batch"] <= 1e-6
        and smoke_summary["artifact_reload_max_abs_diff"] == 0.0,
        "Smoke mechanism report changed",
    )
    smoke_payload = torch.load(
        smoke_artifact_path, map_location="cpu", weights_only=True
    )
    require(
        smoke_payload["fold_lock"] == fold_lock(runtime, prepared, 0)
        and smoke_payload["schema"] == "ra_bmg_a012_v1_single_backbone_artifact",
        "Smoke artifact lock/schema changed",
    )
    smoke_wrapper = build_fresh_artifact(context, smoke_payload["model_state"])
    require(
        audit_artifact_state(smoke_wrapper)
        == {
            key: value
            for key, value in smoke_summary["storage"].items()
            if key != "artifact_file_bytes"
        },
        "Smoke artifact storage readback changed",
    )
    del smoke_wrapper
    torch.cuda.empty_cache()
    runtime_path = output_root / "formal_config.json"
    if runtime_path.exists():
        require(json.loads(runtime_path.read_text(encoding="utf-8")) == runtime, "Formal runtime config drift")
    else:
        write_json(runtime_path, runtime)

    fold_summaries, fold_rows = [], []
    for fold in FOLDS:
        summary, rows = run_formal_fold(context, output_root, prepared, runtime, fold)
        fold_summaries.append(summary)
        fold_rows.append(rows)
    summary = aggregate_formal(
        context,
        output_root,
        prepared,
        runtime,
        fold_summaries,
        fold_rows,
        time.perf_counter() - formal_started,
    )
    print(
        f"FORMAL COMPLETE correct={summary['metrics']['correct']}/598 "
        f"decision={summary['decision']}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RA-BMG-A012 v1 closed-form construction")
    parser.add_argument("mode", choices=("prepare", "smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / EXPERIMENT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    require(
        output_root == (ROOT / "experiments" / EXPERIMENT).resolve(),
        "Output root must be experiments/ra_bmg_a012_v1",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        require(not (output_root / "experiment_config.json").exists(), "Prepared config already exists")
        config = prepare_config(output_root)
        print(
            f"PREPARE PASS checkpoints={len(config['checkpoint_inventory'])} "
            f"B1={config['reference']['b1_metrics']['correct']}/598",
            flush=True,
        )
        return
    prepared = load_prepared_config(output_root)
    context = load_context(args.device)
    if args.mode == "smoke":
        run_smoke(context, output_root, prepared)
    else:
        run_formal(context, output_root, prepared)


if __name__ == "__main__":
    main()
