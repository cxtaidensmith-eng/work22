"""Run Boundary-Protected AD-sMCI Specialist and Model Soup v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Model.bp_ads import BoundaryProtectedADSMCISpecialist
from Utils import CustomCosineAnnealingLR, SET_Random
import run_c1_broad_hparam_search_v1 as broad
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "bp_ads_v1"
OUTPUT_REL = Path("experiments/bp_ads_v1")
RUNNER_REL = Path("scripts/run_bp_ads_v1.py")
MODEL_REL = Path("Model/bp_ads.py")
BASE_C1_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
BROAD_IMPLEMENTATION_COMMIT = "13a25931fd8e5a111a7ae1be056f7b1b213c3055"
BRANCH = "experiment/bp-ads-v1"
FOLDS = tuple(range(10))
EPOCHS = 400
SEED = 0
CLASS_NAMES = ("AD", "CN", "SMCI")
EXPERT_LR = 0.01
EXPERT_WEIGHT_DECAY = 0.001
EXPECTED_BACKBONE_PARAMETERS = 862_971
EXPECTED_EXPERT_PARAMETERS = 3_089
EXPECTED_TOTAL_PARAMETERS = 866_060
A012_SPEC = {
    "trial_id": "A012", "stage": "A", "rank": 8,
    "base_lr": 0.011271416075886307,
    "base_weight_decay": 0.0010917677921787822,
    "lambda_aux": 0.5, "adapter_lr_multiplier": 2.0,
    "dropout_multiplier": 1.1, "generation_index": 12,
}
A007_SPEC = {
    "trial_id": "A007", "stage": "A", "rank": 8,
    "base_lr": 0.009954297170728382,
    "base_weight_decay": 0.001118618078176869,
    "lambda_aux": 0.75, "adapter_lr_multiplier": 1.0,
    "dropout_multiplier": 1.1, "generation_index": 7,
}
C1 = {
    "correct": 560, "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448, "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947, "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
A012 = {
    "correct": 562, "acc": 0.939799331103679,
    "macro_f1": 0.9294205448780151, "bacc": 0.9286299264715336,
    "macro_auc": 0.961524845528226, "weighted_f1": 0.9397414920986079,
    "confusion_matrix": [[64, 0, 8], [0, 200, 9], [7, 12, 298]],
}


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check, text=True, encoding="utf-8", errors="replace", capture_output=True,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def source_paths() -> dict[str, Path]:
    siblings = ROOT.parent
    broad_local = siblings / "c1_broad_hparam_search_v1"
    pc_local = siblings / "pc_bbf_c1_v1"
    c1_ckpt_root = siblings / "cme_dual_branch_v1" / "experiments/cme_dual_branch_v1/c1_shared_private_control_2"
    a012_full = broad_local / "experiments/c1_broad_hparam_search_v1/stage_a/trial_A012/oof_predictions.csv"
    a007_full = broad_local / "experiments/c1_broad_hparam_search_v1/stage_a/trial_A007/oof_predictions.csv"
    return {
        "c1_oof": ROOT / "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv",
        "a012_oof": a012_full if a012_full.is_file() else ROOT / "experiments/c1_broad_hparam_search_v1/best_oof_probabilities.csv",
        "a007_oof": a007_full,
        "pc_bbf_oof": pc_local / "experiments/pc_bbf_c1_v1/formal/oof_predictions.csv",
        "c1_checkpoint_root": c1_ckpt_root,
    }


def row_probability(row: dict) -> np.ndarray:
    value = np.asarray([float(row[f"probability_{name}"]) for name in CLASS_NAMES], dtype=np.float64)
    require(bool(np.isfinite(value).all()) and abs(float(value.sum()) - 1.0) <= 1e-5, "Invalid reference probability")
    return value


def indexed_rows(path: Path, name: str, optional: bool = False) -> dict[int, dict] | None:
    if not path.is_file():
        if optional:
            return None
        raise InvariantError(f"Required {name} OOF missing: {path}")
    rows = read_csv(path)
    indexed = {int(row["subject_index"]): row for row in rows}
    require(len(rows) == len(indexed) == 598, f"{name}: expected 598 unique IDs")
    for row in rows:
        row_probability(row)
    return indexed


def metric_payload(truth: np.ndarray, probability: np.ndarray) -> dict:
    metrics = cme.probability_metrics(truth, probability)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    prediction = probability.argmax(axis=1)
    metrics.update({
        "AD_to_SMCI": int(matrix[0, 2]), "SMCI_to_AD": int(matrix[2, 0]),
        "CN_to_SMCI": int(matrix[1, 2]), "SMCI_to_CN": int(matrix[2, 1]),
        "AD_SMCI_errors": int(matrix[0, 2] + matrix[2, 0]),
        "CN_SMCI_errors": int(matrix[1, 2] + matrix[2, 1]),
        "AD_CN_errors": int(matrix[0, 1] + matrix[1, 0]),
        "predicted_class_counts": {CLASS_NAMES[index]: int((prediction == index).sum()) for index in range(3)},
    })
    return metrics


def paired_payload(truth: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> dict:
    new, old = candidate.argmax(axis=1), reference.argmax(axis=1)
    return {
        "repairs": int(((old != truth) & (new == truth)).sum()),
        "damages": int(((old == truth) & (new != truth)).sum()),
        "changed": int((old != new).sum()),
    }


def prepare_static_artifacts(output_root: Path) -> dict:
    paths = source_paths()
    references = {
        "c1": indexed_rows(paths["c1_oof"], "C1"),
        "a012": indexed_rows(paths["a012_oof"], "A012"),
        "pc_bbf": indexed_rows(paths["pc_bbf_oof"], "PC-BBF", optional=True),
        "a007": indexed_rows(paths["a007_oof"], "A007", optional=True),
    }
    ids = sorted(references["c1"])
    require(ids == list(range(598)), "Reference subject index set changed")
    for subject in ids:
        truth = int(references["c1"][subject]["truth"])
        fold = int(references["c1"][subject]["fold"])
        for name, indexed in references.items():
            if indexed is not None:
                require(int(indexed[subject]["truth"]) == truth and int(indexed[subject]["fold"]) == fold, f"{name}/C1 ID-label-fold mismatch")
    truth = np.asarray([int(references["c1"][subject]["truth"]) for subject in ids], dtype=np.int64)
    probabilities = {
        name: np.stack([row_probability(indexed[subject]) for subject in ids])
        for name, indexed in references.items() if indexed is not None
    }
    require(metric_payload(truth, probabilities["c1"])["correct"] == C1["correct"], "C1 reference changed")
    require(metric_payload(truth, probabilities["a012"])["correct"] == A012["correct"], "A012 reference changed")
    aligned = []
    for offset, subject in enumerate(ids):
        row = {"fold": int(references["c1"][subject]["fold"]), "subject_index": subject, "truth": int(truth[offset])}
        for name in ("c1", "a012", "pc_bbf", "a007"):
            if name in probabilities:
                for class_index, class_name in enumerate(CLASS_NAMES):
                    row[f"{name}_probability_{class_name}"] = float(probabilities[name][offset, class_index])
            else:
                for class_name in CLASS_NAMES:
                    row[f"{name}_probability_{class_name}"] = ""
        aligned.append(row)
    write_csv(output_root / "aligned_oof_reference.csv", aligned)

    eps = 1e-12
    boundary = {}
    for name, specialist_name in (("B1_c1_protect_cn_a012_ads", "a012"), ("B2_c1_protect_cn_pc_ads", "pc_bbf")):
        if specialist_name not in probabilities:
            boundary[name] = {"available": False}
            continue
        p_c1, p_specialist = probabilities["c1"], probabilities[specialist_name]
        protect = p_c1.argmax(axis=1) == 1
        result = np.empty_like(p_c1)
        result[protect] = p_c1[protect]
        q_cn = p_c1[~protect, 1]
        conditional = p_specialist[~protect, 0] / (p_specialist[~protect, 0] + p_specialist[~protect, 2] + eps)
        result[~protect, 1] = q_cn
        result[~protect, 0] = (1.0 - q_cn) * conditional
        result[~protect, 2] = (1.0 - q_cn) * (1.0 - conditional)
        metrics = metric_payload(truth, result)
        vs_c1, vs_a012 = paired_payload(truth, result, probabilities["c1"]), paired_payload(truth, result, probabilities["a012"])
        passed = metrics["correct"] >= 563 and vs_c1["repairs"] > vs_c1["damages"] and metrics["AD_CN_errors"] == 0 and metrics["bacc"] >= A012["bacc"] - 0.005
        boundary[name] = {"available": True, "metrics": metrics, "vs_c1": vs_c1, "vs_a012": vs_a012, "protected_subject_count": int(protect.sum()), "protected_probability_max_abs_diff": float(np.max(np.abs(result[protect] - p_c1[protect]))), "passes_formal_gate": passed}
    p_a012, p_c1 = probabilities["a012"], probabilities["c1"]
    protect = p_a012.argmax(axis=1) == 0
    result = np.empty_like(p_a012)
    result[protect] = p_a012[protect]
    q_ad = p_a012[~protect, 0]
    conditional = p_c1[~protect, 1] / (p_c1[~protect, 1] + p_c1[~protect, 2] + eps)
    result[~protect, 0] = q_ad
    result[~protect, 1] = (1.0 - q_ad) * conditional
    result[~protect, 2] = (1.0 - q_ad) * (1.0 - conditional)
    metrics = metric_payload(truth, result)
    vs_c1, vs_a012 = paired_payload(truth, result, p_c1), paired_payload(truth, result, p_a012)
    boundary["B3_a012_protect_ad_c1_cn_smci"] = {"available": True, "metrics": metrics, "vs_c1": vs_c1, "vs_a012": vs_a012, "protected_subject_count": int(protect.sum()), "protected_probability_max_abs_diff": float(np.max(np.abs(result[protect] - p_a012[protect]))), "passes_formal_gate": metrics["correct"] >= 563 and vs_c1["repairs"] > vs_c1["damages"] and metrics["AD_CN_errors"] == 0 and metrics["bacc"] >= A012["bacc"] - 0.005}
    eligible = [name for name, value in boundary.items() if value.get("passes_formal_gate")]
    eligible.sort(key=lambda name: (boundary[name]["metrics"]["correct"], boundary[name]["metrics"]["macro_auc"], boundary[name]["metrics"]["macro_f1"]), reverse=True)
    boundary_payload = {"class_order": list(CLASS_NAMES), "schemes": boundary, "eligible_ranked": eligible, "selected_direction": eligible[0] if eligible else None, "stage_d_required": bool(eligible)}
    write_json(output_root / "boundary_feasibility.json", boundary_payload)

    soups = {}
    combinations = {"S1_probability_only": ("c1", "a012"), "S2_probability_only": ("a012", "a007"), "S3_probability_only": ("c1", "a012", "a007")}
    for name, members in combinations.items():
        if not all(member in probabilities for member in members):
            soups[name] = {"available": False, "members": list(members)}
            continue
        result = sum(probabilities[member] for member in members) / len(members)
        metrics = metric_payload(truth, result)
        vs_c1, vs_a012 = paired_payload(truth, result, p_c1), paired_payload(truth, result, p_a012)
        soups[name] = {"available": True, "probability_feasibility_only": True, "members": list(members), "metrics": metrics, "vs_c1": vs_c1, "vs_a012": vs_a012, "permits_exact_checkpoint_rerun": metrics["correct"] >= 563}
    soup_payload = {"probability_feasibility": soups, "formal_weight_soups": {}, "checkpoint_rerun_allowed_configs": sorted({member for value in soups.values() if value.get("permits_exact_checkpoint_rerun") for member in value["members"] if member in {"a012", "a007"}})}
    write_json(output_root / "model_soup_results.json", soup_payload)

    c1_checkpoints = [paths["c1_checkpoint_root"] / f"fold_{fold:02d}/checkpoint_best.pt" for fold in FOLDS]
    oof_inventory = {}
    for name, indexed in references.items():
        path = paths[f"{name}_oof"]
        columns = list(next(iter(indexed.values()))) if indexed is not None else []
        oof_inventory[name] = {
            "available": indexed is not None,
            "path": str(path),
            "sha256": file_sha256(path) if indexed is not None else None,
            "probabilities_available": all(f"probability_{class_name}" in columns for class_name in CLASS_NAMES) if indexed is not None else False,
            "raw_logits_available": all(f"raw_logit_{class_name}" in columns for class_name in CLASS_NAMES) if indexed is not None else False,
        }
    checkpoint_inventory = []
    for fold, path in enumerate(c1_checkpoints):
        item = {"fold": fold, "path": str(path), "available": path.is_file(), "sha256": file_sha256(path) if path.is_file() else None, "schema": None}
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            item["schema"] = "payload_model_state" if isinstance(payload, dict) and "model_state" in payload else "direct_state_dict"
        checkpoint_inventory.append(item)
    inventory = {
        "experiment": EXPERIMENT, "alignment": {"subjects": 598, "unique_ids": True, "labels_and_folds_match": True, "alignment_key": "subject_index"},
        "oof": oof_inventory,
        "checkpoints": {"c1": {"available_folds": sum(path.is_file() for path in c1_checkpoints), "folds": checkpoint_inventory}, "a012": {"available_folds": 0, "exact_rerun_required": "a012" in soup_payload["checkpoint_rerun_allowed_configs"]}, "a007": {"available_folds": 0, "exact_rerun_required": "a007" in soup_payload["checkpoint_rerun_allowed_configs"]}, "pc_bbf": {"weight_soup_forbidden": True}},
        "test_label_usage": "never enters features, model weights, soup generation, specialist inputs, or gradient-bearing loss; held-out truth is used only by the locked historical best-epoch scoring rule and final OOF metrics",
    }
    write_json(output_root / "artifact_inventory.json", inventory)
    return {"inventory": inventory, "boundary": boundary_payload, "soup": soup_payload, "truth": truth, "probabilities": probabilities, "ids": ids}


def load_context() -> dict:
    context = broad.load_context("cuda:0")
    require(int(context["config"].Hidden_size) == 96, "BP-ADS expects C1 hidden size 96")
    return context


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    require(path.is_file(), f"Checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    require(isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()), f"Unsupported checkpoint schema: {path}")
    return payload


def c1_checkpoint_path(fold: int) -> Path:
    return source_paths()["c1_checkpoint_root"] / f"fold_{fold:02d}/checkpoint_best.pt"


def build_c1_from_checkpoint(context: dict, fold: int):
    model = broad.build_model(context, 8)
    model.load_state_dict(load_model_state(c1_checkpoint_path(fold)), strict=True)
    model.eval()
    require(parameter_count(model) == EXPECTED_BACKBONE_PARAMETERS, "C1 parameter count changed")
    return model


def build_specialist(context: dict, fold: int):
    SET_Random(SEED)
    model = BoundaryProtectedADSMCISpecialist(
        build_c1_from_checkpoint(context, fold),
        hidden_size=int(context["config"].Hidden_size),
        label_weight=context["dataset_dict"]["Label_Weight"],
        logit_adjust_tau=float(context["config"].logit_adjust_tau),
    ).to(context["device"])
    require(model.expert_parameter_count == EXPECTED_EXPERT_PARAMETERS, "Expert parameter count changed")
    require(parameter_count(model) == EXPECTED_TOTAL_PARAMETERS, "BP-ADS parameter count changed")
    require(all(not parameter.requires_grad for parameter in model.backbone.parameters()), "C1 backbone is not fully frozen")
    return model


def class_balanced_boundary_loss(details: dict, labels: torch.Tensor, train_mask: torch.Tensor, test_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    require(not bool((train_mask & test_mask).any()), "Train/test masks overlap")
    train_indices = torch.where(train_mask)[0]
    train_labels = labels[train_indices]
    pair_local_mask = train_labels != 1
    expert_indices = train_indices[pair_local_mask]
    pair_train_labels = train_labels[pair_local_mask]
    target = (pair_train_labels == 0).to(details["boundary_logit"].dtype)
    positives, negatives = int(target.sum()), int(target.numel() - target.sum())
    require(positives > 0 and negatives > 0, "AD/sMCI training classes missing")
    positive_weight = target.numel() / (2.0 * positives)
    negative_weight = target.numel() / (2.0 * negatives)
    weights = torch.where(target > 0.5, target.new_tensor(positive_weight), target.new_tensor(negative_weight))
    values = F.binary_cross_entropy_with_logits(details["boundary_logit"][expert_indices], target, reduction="none")
    loss = (values * weights).mean()
    return loss, {"train_count": int(target.numel()), "AD_count": positives, "SMCI_count": negatives, "AD_weight": positive_weight, "SMCI_weight": negative_weight, "train_subject_indices_sha256": payload_sha256(expert_indices.detach().cpu().tolist()), "test_subject_count_read_by_loss": 0}


def probability_rows(fold: int, probability: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, dataset_dict: dict) -> list[dict]:
    indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[mask.detach().cpu().numpy().astype(bool)]
    values = probability[mask].detach().cpu().numpy()
    truth = labels[mask].detach().cpu().numpy().astype(int)
    prediction = values.argmax(axis=1)
    return [{"fold": fold, "subject_index": int(subject), "truth": int(truth[offset]), "prediction": int(prediction[offset]), **{f"probability_{name}": float(values[offset, index]) for index, name in enumerate(CLASS_NAMES)}} for offset, subject in enumerate(indices)]


def run_smoke(context: dict, output_root: Path, static: dict) -> dict:
    require(static["boundary"]["selected_direction"] == "B1_c1_protect_cn_a012_ads", "Smoke requires the locked B1 direction")
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    model = build_specialist(context, 0)
    features, labels = context["dataset_data"]["Feature"], context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    cache = model.frozen_cache(features)
    initial_probability, initial_details = model.probability_from_cache(cache)
    initial_diff = float((initial_probability - cache["p0"]).abs().max().cpu())
    require(initial_diff <= 1e-7, f"Initial specialist/C1 probability mismatch: {initial_diff}")
    require(initial_diff == 0.0, "Residual specialist initialization should be bitwise C1-equivalent")
    protected = cache["protected_cn"]
    protected_test = protected & test_mask
    require(float((initial_probability[protected] - cache["p0"][protected]).abs().max().cpu()) == 0.0, "CN hard protection changed at initialization")
    c1_base_rows = probability_rows(0, cache["p0"], labels, test_mask, context["dataset_dict"])
    c1_reference_rows = [context["c1_by_subject"][int(row["subject_index"])] for row in c1_base_rows]
    c1_oof_max_diff = require_probability_rows_match(c1_base_rows, c1_reference_rows, "Smoke C1 checkpoint/OOF anchor", tolerance=1e-6)
    initial_expert = clone_cpu_state(model.expert)
    optimizer = torch.optim.Adam(model.expert.parameters(), lr=EXPERT_LR, weight_decay=EXPERT_WEIGHT_DECAY)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min))
    losses, gradients = [], []
    gradient_by_parameter = {name: [] for name, _ in model.expert.named_parameters()}
    train_audit = None
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        probability, details = model.probability_from_cache(cache)
        loss, train_audit = class_balanced_boundary_loss(details, labels, train_mask, test_mask)
        require(bool(torch.isfinite(loss)), f"Smoke epoch{epoch}: non-finite loss")
        loss.backward()
        grads = [parameter.grad for parameter in model.expert.parameters() if parameter.grad is not None]
        require(grads and all(bool(torch.isfinite(gradient).all()) for gradient in grads), f"Smoke epoch{epoch}: non-finite expert gradient")
        grad_norm = float(torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in grads)).cpu())
        require(grad_norm > 0.0, f"Smoke epoch{epoch}: zero expert gradient")
        for name, parameter in model.expert.named_parameters():
            value = 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().max().cpu())
            require(math.isfinite(value), f"Smoke epoch{epoch}: {name} non-finite gradient")
            gradient_by_parameter[name].append(value)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        gradients.append(grad_norm)
    parameter_change_by_name = {name: float((value.detach().cpu() - initial_expert[name]).abs().max()) for name, value in model.expert.state_dict().items()}
    require(all(max(values) > 0.0 for values in gradient_by_parameter.values()), f"Not all expert tensors received a nonzero gradient by epoch3: {gradient_by_parameter}")
    require(all(math.isfinite(value) and value > 0.0 for value in parameter_change_by_name.values()), f"Not all expert tensors changed by epoch3: {parameter_change_by_name}")
    parameter_change = max(parameter_change_by_name.values())
    require(math.isfinite(parameter_change) and parameter_change > 0.0, "Smoke expert parameters did not change")
    model.eval()
    with torch.no_grad():
        probability, _ = model.probability_from_cache(cache)
    require(bool(torch.isfinite(probability).all()), "Smoke probability contains NaN/Inf")
    checkpoint = smoke_root / "checkpoint_roundtrip.pt"
    config = {"experiment": EXPERIMENT, "fold": 0, "epochs": 3, "direction": static["boundary"]["selected_direction"], "source_commit_at_smoke": git("rev-parse", "HEAD").stdout.strip(), "runner_sha256": file_sha256(ROOT / RUNNER_REL), "model_sha256": file_sha256(ROOT / MODEL_REL)}
    torch.save({"model_state": clone_cpu_state(model), "config": config}, checkpoint)
    reloaded = build_specialist(context, 0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        readback, _ = reloaded.probability_from_cache(reloaded.frozen_cache(features))
    roundtrip_diff = float((probability - readback).abs().max().cpu())
    require(roundtrip_diff <= 1e-7, "Smoke strict checkpoint roundtrip failed")
    report = {"passed": True, "fold": 0, "epochs": 3, "direction": static["boundary"]["selected_direction"], "initial_probability_max_abs_diff": initial_diff, "c1_checkpoint_oof_probability_max_abs_diff": c1_oof_max_diff, "c1_checkpoint_oof_prediction_match": True, "losses": losses, "expert_gradient_norms": gradients, "expert_gradient_max_abs_by_parameter": gradient_by_parameter, "expert_parameter_max_abs_change": parameter_change, "expert_parameter_change_by_name": parameter_change_by_name, "train_only_audit": train_audit, "protected_cn_count": int(protected_test.sum()), "checkpoint_strict_load": True, "checkpoint_roundtrip_max_abs_diff": roundtrip_diff, "backbone_parameters": EXPECTED_BACKBONE_PARAMETERS, "expert_parameters": EXPECTED_EXPERT_PARAMETERS, "total_parameters": EXPECTED_TOTAL_PARAMETERS, "single_model": True, "ensemble": False, "config": config, "run_command": f'"{sys.executable}" -u -B scripts/run_bp_ads_v1.py smoke --device cuda:0'}
    write_json(smoke_root / "smoke_report.json", report)
    write_json(smoke_root / "smoke_config.json", config)
    del model, reloaded, optimizer, scheduler
    torch.cuda.empty_cache()
    return report


def validate_rows(rows: list[dict], expected_fold: int | None = None) -> list[dict]:
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(subjects) == len(set(subjects)), "OOF contains duplicate subject IDs")
    if expected_fold is None:
        require(len(rows) == 598 and set(subjects) == set(range(598)), "OOF does not cover 598 subjects")
    for row in rows:
        if expected_fold is not None:
            require(int(row["fold"]) == expected_fold, "OOF row has wrong fold")
        probability = row_probability(row)
        require(int(row["prediction"]) == int(probability.argmax()), "OOF prediction is not probability argmax")
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def require_probability_rows_match(
    actual: list[dict], expected: list[dict], context: str, tolerance: float = 1e-7
) -> float:
    require(len(actual) == len({int(row["subject_index"]) for row in actual}), f"{context}: actual IDs duplicated")
    require(len(expected) == len({int(row["subject_index"]) for row in expected}), f"{context}: expected IDs duplicated")
    actual = sorted(actual, key=lambda row: int(row["subject_index"]))
    expected = sorted(expected, key=lambda row: int(row["subject_index"]))
    require([int(row["subject_index"]) for row in actual] == [int(row["subject_index"]) for row in expected], f"{context}: subject IDs differ")
    maximum = 0.0
    for actual_row, expected_row in zip(actual, expected):
        for key in ("fold", "subject_index", "truth", "prediction"):
            require(int(actual_row[key]) == int(expected_row[key]), f"{context}: {key} differs for subject {actual_row['subject_index']}")
        maximum = max(
            maximum,
            max(abs(float(actual_row[f"probability_{name}"]) - float(expected_row[f"probability_{name}"])) for name in CLASS_NAMES),
        )
    require(maximum <= tolerance, f"{context}: probability max diff {maximum} > {tolerance}")
    return maximum


def checkpoint_trial_config(context: dict, name: str, spec: dict) -> dict:
    payload = {
        "experiment": EXPERIMENT, "purpose": "exact_checkpoint_materialization",
        "config_name": name, "trial": spec, "folds": list(FOLDS), "seed": SEED,
        "epochs": EPOCHS, "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / RUNNER_REL),
        "broad_runner_sha256": file_sha256(ROOT / "scripts/run_c1_broad_hparam_search_v1.py"),
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "single_model": True, "ensemble": False, "graph": False, "ema": False,
    }
    payload["config_sha256"] = payload_sha256(payload)
    return payload


def materialized_checkpoint_path(output_root: Path, name: str, fold: int) -> Path:
    return output_root / "checkpoint_materialization" / name / f"fold_{fold:02d}/checkpoint_best.pt"


def fold_lock(config: dict, fold: int) -> dict:
    return {"fold": fold, "config_name": config["config_name"], "source_commit": config["source_commit"], "config_sha256": config["config_sha256"], "fold_manifest_sha256": config["fold_manifest_sha256"]}


def load_completed_materialized_fold(context: dict, name: str, spec: dict, config: dict, fold: int, final_dir: Path):
    if not final_dir.exists():
        return None
    required = [final_dir / item for item in ("checkpoint_best.pt", "summary.json", "oof_predictions.csv", "config.json", "complete.json")]
    require(all(path.is_file() for path in required), f"{name} fold{fold}: completed materialization is incomplete")
    lock = fold_lock(config, fold)
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == lock, f"{name} fold{fold}: config/source drift")
    marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
    require(marker == {"complete": True, **lock}, f"{name} fold{fold}: completion marker drift")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary["fold_lock"] == lock and 1 <= int(summary["best_epoch"]) <= EPOCHS, f"{name} fold{fold}: summary drift")
    rows = read_csv(final_dir / "oof_predictions.csv")
    validate_rows(rows, expected_fold=fold)
    require(cme.metrics_from_rows(rows) == summary["best_metrics"], f"{name} fold{fold}: OOF/metrics drift")
    checkpoint_payload = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(isinstance(checkpoint_payload, dict) and "model_state" in checkpoint_payload, f"{name} fold{fold}: checkpoint schema drift")
    require(checkpoint_payload.get("fold_lock") == lock, f"{name} fold{fold}: checkpoint lock drift")
    require(int(checkpoint_payload.get("best_epoch", -1)) == int(summary["best_epoch"]), f"{name} fold{fold}: checkpoint epoch drift")
    require(checkpoint_payload.get("best_metrics") == summary["best_metrics"], f"{name} fold{fold}: checkpoint metrics drift")
    model = broad.build_model(context, 8)
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    del model
    torch.cuda.empty_cache()
    print(f"RESUME_CHECKPOINT {name} fold={fold} best_epoch={summary['best_epoch']}", flush=True)
    return summary, rows


def materialize_config_fold(context: dict, output_root: Path, name: str, spec: dict, config: dict, fold: int):
    root = output_root / "checkpoint_materialization" / name
    final_dir = root / f"fold_{fold:02d}"
    resumed = load_completed_materialized_fold(context, name, spec, config, fold, final_dir)
    if resumed is not None:
        return resumed
    staging = root / f".fold_{fold:02d}_in_progress"
    lock = fold_lock(config, fold)
    if staging.exists():
        require((staging / "config.json").is_file() and json.loads((staging / "config.json").read_text(encoding="utf-8")) == lock, f"{name} fold{fold}: in-progress config drift")
        resolved = staging.resolve()
        require(resolved.parent == root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe materialization restart target")
        shutil.rmtree(resolved)
    staging.mkdir(parents=True)
    write_json(staging / "config.json", lock)
    model, criterion, optimizer, scheduler, dropout, groups = broad.make_training_objects(context, spec)
    features, labels = context["dataset_data"]["Feature"], context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool((train_mask & test_mask).any()), f"{name} fold{fold}: train/test overlap")
    best, best_state = None, None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = broad.loss_components(criterion, raw, labels, train_mask, auxiliary, float(spec["lambda_aux"]))["total"]
        require(bool(torch.isfinite(loss)), f"{name} fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(broad.gradients_finite(model), f"{name} fold{fold} epoch{epoch}: non-finite gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            evaluated, _, _ = model(features)
            metrics, selection = cme.selection_metrics(evaluated, labels, test_mask, context["dataset_dict"]["Label_Weight"], float(context["config"].logit_adjust_tau))
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = clone_cpu_state(model)
    require(best is not None and best_state is not None, f"{name} fold{fold}: no best checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _ = model(features)
    rows = cme.prediction_rows(fold, best_raw, labels, test_mask, context["dataset_dict"], context["config"])
    validate_rows(rows, expected_fold=fold)
    require(cme.metrics_from_rows(rows) == best["metrics"], f"{name} fold{fold}: checkpoint OOF readback mismatch")
    checkpoint_payload = {"model_state": best_state, "fold_lock": lock, "best_epoch": best["epoch"], "best_metrics": best["metrics"]}
    torch.save(checkpoint_payload, staging / "checkpoint_best.pt")
    summary = {"config_name": name, "fold": fold, "best_epoch": best["epoch"], "best_metrics": best["metrics"], "parameter_count": parameter_count(model), "elapsed_seconds": float(time.perf_counter() - started), "dropout_module_count": len(dropout), "parameter_group_audit": {"base_parameter_tensors": groups["base_parameter_tensors"], "adapter_parameter_tensors": groups["adapter_parameter_tensors"]}, "fold_lock": lock}
    write_json(staging / "summary.json", summary)
    write_csv(staging / "oof_predictions.csv", rows)
    write_json(staging / "complete.json", {"complete": True, **lock})
    staging.rename(final_dir)
    print(f"MATERIALIZED {name} fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']}", flush=True)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def materialize_config(context: dict, output_root: Path, name: str, spec: dict) -> dict:
    root = output_root / "checkpoint_materialization" / name
    root.mkdir(parents=True, exist_ok=True)
    config = checkpoint_trial_config(context, name, spec)
    config_path = root / "config.json"
    if config_path.is_file():
        require(json.loads(config_path.read_text(encoding="utf-8")) == config, f"{name}: materialization config/source drift")
    else:
        write_json(config_path, config)
    summaries, rows = [], []
    for fold in FOLDS:
        summary, fold_rows = materialize_config_fold(context, output_root, name, spec, config, fold)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = validate_rows(rows)
    metrics = cme.metrics_from_rows(rows)
    expected = A012 if name == "a012" else {"correct": 560, "acc": 0.9364548494983278, "macro_f1": 0.9169142032505032, "bacc": 0.9127578289955842, "macro_auc": 0.9631938876417131, "weighted_f1": 0.9362151012851758, "confusion_matrix": [[60, 0, 12], [0, 201, 8], [9, 9, 299]]}
    require(metrics["correct"] == expected["correct"] and metrics["confusion_matrix"] == expected["confusion_matrix"], f"{name}: exact rerun prediction anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - expected[key]) <= 5e-7, f"{name}: exact rerun metric changed: {key}")
    reference_path = source_paths()[f"{name}_oof"]
    reference_rows = validate_rows(read_csv(reference_path))
    reference_by_subject = {int(row["subject_index"]): row for row in reference_rows}
    maximum_differences = {"raw_logit": 0.0, "adjusted_score": 0.0, "probability": 0.0}
    for row in rows:
        subject = int(row["subject_index"])
        reference = reference_by_subject[subject]
        require(int(row["fold"]) == int(reference["fold"]) and int(row["truth"]) == int(reference["truth"]), f"{name}: external OOF fold/truth drift for subject {subject}")
        require(int(row["prediction"]) == int(reference["prediction"]), f"{name}: external OOF prediction drift for subject {subject}")
        for prefix in maximum_differences:
            difference = max(abs(float(row[f"{prefix}_{class_name}"]) - float(reference[f"{prefix}_{class_name}"])) for class_name in CLASS_NAMES)
            maximum_differences[prefix] = max(maximum_differences[prefix], difference)
    require(all(value <= 1e-6 for value in maximum_differences.values()), f"{name}: exact rerun differs from broad OOF: {maximum_differences}")
    report = {"config_name": name, "trial": spec, "metrics": metrics, "folds": [{"fold": item["fold"], "best_epoch": item["best_epoch"], "correct": item["best_metrics"]["correct"], "acc": item["best_metrics"]["acc"], "elapsed_seconds": item["elapsed_seconds"]} for item in summaries], "training_seconds": float(sum(item["elapsed_seconds"] for item in summaries)), "checkpoint_count": 10, "external_oof_alignment": {"path": str(reference_path), "sha256": file_sha256(reference_path), "prediction_match": True, "max_abs_diff": maximum_differences, "tolerance": 1e-6}, "config": config}
    write_json(root / "report.json", report)
    write_csv(root / "oof_predictions.csv", rows)
    return report


def average_state_dicts(states: list[dict[str, torch.Tensor]], a012_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    require(len(states) == 3, "S3 requires exactly C1/A012/A007 states")
    keys = list(a012_state)
    require(all(list(state) == keys for state in states), "Weight-soup state_dict keys differ")
    output = {}
    for key in keys:
        tensors = [state[key] for state in states]
        require(all(tuple(value.shape) == tuple(tensors[0].shape) for value in tensors), f"Weight-soup tensor shape differs: {key}")
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            output[key] = torch.stack([value.to(dtype=tensors[0].dtype) for value in tensors], dim=0).mean(dim=0)
        else:
            output[key] = a012_state[key].clone()
    return output


def run_weight_soup(context: dict, output_root: Path, aligned: dict) -> tuple[dict, list[dict]]:
    root = output_root / "weight_soup" / "S3_c1_a012_a007_equal"
    root.mkdir(parents=True, exist_ok=True)
    summaries, pooled = [], []
    features, labels = context["dataset_data"]["Feature"], context["dataset_data"]["Label"]
    for fold in FOLDS:
        final_dir = root / f"fold_{fold:02d}"
        lock = {"fold": fold, "members": ["c1", "a012", "a007"], "weights": [1 / 3, 1 / 3, 1 / 3], "source_commit": git("rev-parse", "HEAD").stdout.strip(), "fold_manifest_sha256": context["fold_manifest"]["sha256"]}
        if final_dir.exists():
            required = [final_dir / item for item in ("checkpoint_soup.pt", "summary.json", "oof_predictions.csv", "config.json", "complete.json")]
            require(all(path.is_file() for path in required), f"S3 fold{fold}: completed soup is incomplete")
            require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == lock, f"S3 fold{fold}: resume lock drift")
            marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
            require(marker == {"complete": True, **lock}, f"S3 fold{fold}: completion marker drift")
            summary, rows = json.loads((final_dir / "summary.json").read_text(encoding="utf-8")), read_csv(final_dir / "oof_predictions.csv")
            validate_rows(rows, expected_fold=fold)
            require(summary.get("lock") == lock and summary.get("metrics") == cme.metrics_from_rows(rows), f"S3 fold{fold}: summary/OOF drift")
            checkpoint_payload = torch.load(final_dir / "checkpoint_soup.pt", map_location="cpu", weights_only=True)
            require(isinstance(checkpoint_payload, dict) and checkpoint_payload.get("lock") == lock and "model_state" in checkpoint_payload, f"S3 fold{fold}: checkpoint lock/schema drift")
            model = broad.build_model(context, 8)
            model.load_state_dict(checkpoint_payload["model_state"], strict=True)
            del model
            torch.cuda.empty_cache()
        else:
            staging = root / f".fold_{fold:02d}_in_progress"
            if staging.exists():
                require((staging / "config.json").is_file() and json.loads((staging / "config.json").read_text(encoding="utf-8")) == lock, f"S3 fold{fold}: in-progress lock drift")
                resolved = staging.resolve(); require(resolved.parent == root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe soup restart target"); shutil.rmtree(resolved)
            staging.mkdir(parents=True); write_json(staging / "config.json", lock)
            c1_state = load_model_state(c1_checkpoint_path(fold))
            a012_state = load_model_state(materialized_checkpoint_path(output_root, "a012", fold))
            a007_state = load_model_state(materialized_checkpoint_path(output_root, "a007", fold))
            soup_state = average_state_dicts([c1_state, a012_state, a007_state], a012_state)
            model = broad.build_model(context, 8)
            model.load_state_dict(soup_state, strict=True); model.eval()
            _, test_mask = context["dataset_data"]["Mask"][fold]
            with torch.no_grad(): raw, _, _ = model(features)
            rows = cme.prediction_rows(fold, raw, labels, test_mask, context["dataset_dict"], context["config"])
            validate_rows(rows, expected_fold=fold)
            metrics = cme.metrics_from_rows(rows)
            summary = {"fold": fold, "metrics": metrics, "parameter_count": parameter_count(model), "single_model": True, "ensemble": False, "lock": lock}
            torch.save({"model_state": soup_state, "lock": lock}, staging / "checkpoint_soup.pt")
            write_json(staging / "summary.json", summary); write_csv(staging / "oof_predictions.csv", rows); write_json(staging / "complete.json", {"complete": True, **lock}); staging.rename(final_dir)
            del model; torch.cuda.empty_cache()
        summaries.append(summary); pooled.extend(rows)
    pooled = validate_rows(pooled)
    probability = np.asarray([[float(row[f"probability_{name}"]) for name in CLASS_NAMES] for row in pooled])
    truth = np.asarray([int(row["truth"]) for row in pooled])
    metrics = metric_payload(truth, probability)
    c1_probability, a012_probability = aligned["probabilities"]["c1"], aligned["probabilities"]["a012"]
    vs_c1, vs_a012 = paired_payload(truth, probability, c1_probability), paired_payload(truth, probability, a012_probability)
    retained = metrics["correct"] >= 563 and metrics["AD_CN_errors"] == 0 and vs_c1["repairs"] > vs_c1["damages"] and metrics["bacc"] >= A012["bacc"] - 0.005
    report = {"name": "S3_c1_a012_a007_equal", "members": ["c1", "a012", "a007"], "weights": [1 / 3, 1 / 3, 1 / 3], "metrics": metrics, "vs_c1": vs_c1, "vs_a012": vs_a012, "retained": retained, "single_model": True, "ensemble": False, "folds": summaries}
    write_json(root / "report.json", report); write_csv(root / "oof_predictions.csv", pooled)
    return report, pooled


def specialist_config(context: dict, static: dict) -> dict:
    payload = {
        "experiment": EXPERIMENT,
        "model": "one_frozen_C1_backbone_plus_one_AD_SMCI_expert",
        "direction": static["boundary"]["selected_direction"],
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "expert_hidden": 8,
        "expert_lr": EXPERT_LR,
        "expert_weight_decay": EXPERT_WEIGHT_DECAY,
        "scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "loss": "class-balanced BCE on train-mask AD/sMCI only",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "graph": False,
        "backbone_parameters": EXPECTED_BACKBONE_PARAMETERS,
        "expert_parameters": EXPECTED_EXPERT_PARAMETERS,
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / RUNNER_REL),
        "model_sha256": file_sha256(ROOT / MODEL_REL),
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "c1_checkpoint_sha256_by_fold": {
            str(fold): file_sha256(c1_checkpoint_path(fold)) for fold in FOLDS
        },
    }
    payload["config_sha256"] = payload_sha256(payload)
    return payload


def specialist_fold_lock(config: dict, fold: int) -> dict:
    return {
        "fold": fold,
        "source_commit": config["source_commit"],
        "config_sha256": config["config_sha256"],
        "fold_manifest_sha256": config["fold_manifest_sha256"],
        "c1_checkpoint_sha256": config["c1_checkpoint_sha256_by_fold"][str(fold)],
    }


def load_completed_specialist_fold(
    context: dict, config: dict, fold: int, final_dir: Path
) -> tuple[dict, list[dict]] | None:
    if not final_dir.exists():
        return None
    required = [
        final_dir / name
        for name in (
            "checkpoint_best.pt",
            "summary.json",
            "oof_predictions.csv",
            "config.json",
            "complete.json",
        )
    ]
    require(all(path.is_file() for path in required), f"BP-ADS fold{fold}: completed fold is incomplete")
    lock = specialist_fold_lock(config, fold)
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == lock, f"BP-ADS fold{fold}: config/source drift")
    marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
    require(marker == {"complete": True, **lock}, f"BP-ADS fold{fold}: completion marker drift")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("fold_lock") == lock and 1 <= int(summary.get("best_epoch", 0)) <= EPOCHS, f"BP-ADS fold{fold}: summary drift")
    rows = validate_rows(read_csv(final_dir / "oof_predictions.csv"), expected_fold=fold)
    require(cme.metrics_from_rows(rows) == summary.get("best_metrics"), f"BP-ADS fold{fold}: summary/OOF metrics drift")
    checkpoint = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict) and checkpoint.get("fold_lock") == lock and "model_state" in checkpoint, f"BP-ADS fold{fold}: checkpoint schema/lock drift")
    require(int(checkpoint.get("best_epoch", -1)) == int(summary["best_epoch"]), f"BP-ADS fold{fold}: checkpoint epoch drift")
    require(checkpoint.get("best_metrics") == summary["best_metrics"], f"BP-ADS fold{fold}: checkpoint metrics drift")
    model = build_specialist(context, fold)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    with torch.no_grad():
        readback, details = model.probability_from_cache(model.frozen_cache(features))
    readback_rows = probability_rows(fold, readback, labels, test_mask, context["dataset_dict"])
    require_probability_rows_match(readback_rows, rows, f"BP-ADS fold{fold}: strict checkpoint OOF readback", tolerance=1e-7)
    protected = details["protected_cn"]
    require(float((readback[protected] - details["p0"][protected]).abs().max().cpu()) == 0.0, f"BP-ADS fold{fold}: resumed checkpoint violated CN protection")
    del model
    torch.cuda.empty_cache()
    print(f"RESUME_SPECIALIST fold={fold} best_epoch={summary['best_epoch']}", flush=True)
    return summary, rows


def run_specialist_fold(
    context: dict, output_root: Path, config: dict, fold: int
) -> tuple[dict, list[dict]]:
    root = output_root / "specialist_folds"
    final_dir = root / f"fold_{fold:02d}"
    resumed = load_completed_specialist_fold(context, config, fold, final_dir)
    if resumed is not None:
        return resumed
    lock = specialist_fold_lock(config, fold)
    staging = root / f".fold_{fold:02d}_in_progress"
    if staging.exists():
        require((staging / "config.json").is_file() and json.loads((staging / "config.json").read_text(encoding="utf-8")) == lock, f"BP-ADS fold{fold}: in-progress config drift")
        resolved = staging.resolve()
        require(resolved.parent == root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe BP-ADS restart target")
        shutil.rmtree(resolved)
    staging.mkdir(parents=True)
    write_json(staging / "config.json", lock)

    SET_Random(SEED)
    model = build_specialist(context, fold)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool((train_mask & test_mask).any()), f"BP-ADS fold{fold}: train/test overlap")
    cache = model.frozen_cache(features)
    initial_probability, initial_details = model.probability_from_cache(cache)
    initial_diff = float((initial_probability - cache["p0"]).abs().max().cpu())
    require(initial_diff == 0.0, f"BP-ADS fold{fold}: zero-delta C1 equivalence failed ({initial_diff})")
    protected = cache["protected_cn"]
    protected_test = protected & test_mask
    require(float((initial_probability[protected] - cache["p0"][protected]).abs().max().cpu()) == 0.0, f"BP-ADS fold{fold}: initial CN protection failed")
    c1_base_rows = probability_rows(fold, cache["p0"], labels, test_mask, context["dataset_dict"])
    c1_reference_rows = [context["c1_by_subject"][int(row["subject_index"])] for row in c1_base_rows]
    c1_oof_max_diff = require_probability_rows_match(c1_base_rows, c1_reference_rows, f"BP-ADS fold{fold}: C1 checkpoint/OOF anchor", tolerance=1e-6)
    initial_expert = clone_cpu_state(model.expert)
    optimizer = torch.optim.Adam(model.expert.parameters(), lr=EXPERT_LR, weight_decay=EXPERT_WEIGHT_DECAY)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min))
    best, best_state = None, None
    maximum_gradient = 0.0
    train_audit = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        probability, details = model.probability_from_cache(cache)
        loss, train_audit = class_balanced_boundary_loss(details, labels, train_mask, test_mask)
        require(bool(torch.isfinite(loss)), f"BP-ADS fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.expert.parameters() if parameter.grad is not None]
        require(gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients), f"BP-ADS fold{fold} epoch{epoch}: non-finite expert gradient")
        epoch_max_gradient = max(float(gradient.detach().abs().max().cpu()) for gradient in gradients)
        require(epoch_max_gradient > 0.0, f"BP-ADS fold{fold} epoch{epoch}: zero expert gradient")
        maximum_gradient = max(maximum_gradient, epoch_max_gradient)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            evaluated, evaluated_details = model.probability_from_cache(cache)
        require(bool(torch.isfinite(evaluated).all()), f"BP-ADS fold{fold} epoch{epoch}: non-finite probability")
        require(float((evaluated[protected] - cache["p0"][protected]).abs().max().cpu()) == 0.0, f"BP-ADS fold{fold} epoch{epoch}: CN protection changed")
        test_probability = evaluated[test_mask].detach().cpu().numpy()
        test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        metrics = cme.probability_metrics(test_truth, test_probability)
        selection = (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = clone_cpu_state(model)
    require(best is not None and best_state is not None, f"BP-ADS fold{fold}: no best checkpoint")
    terminal_parameter_change = {
        name: float((value.detach().cpu() - initial_expert[name]).abs().max())
        for name, value in model.expert.state_dict().items()
    }
    require(all(math.isfinite(value) and value > 0.0 for value in terminal_parameter_change.values()), f"BP-ADS fold{fold}: expert tensor did not change during 400-epoch training")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_probability, best_details = model.probability_from_cache(model.frozen_cache(features))
    require(bool(torch.isfinite(best_probability).all()), f"BP-ADS fold{fold}: best probability non-finite")
    require(float((best_probability[protected] - cache["p0"][protected]).abs().max().cpu()) == 0.0, f"BP-ADS fold{fold}: best checkpoint violated CN protection")
    rows = validate_rows(probability_rows(fold, best_probability, labels, test_mask, context["dataset_dict"]), expected_fold=fold)
    require(cme.metrics_from_rows(rows) == best["metrics"], f"BP-ADS fold{fold}: best checkpoint OOF/metrics drift")
    best_parameter_change = {
        name: float((value.detach().cpu() - initial_expert[name]).abs().max())
        for name, value in model.expert.state_dict().items()
    }
    require(all(math.isfinite(value) for value in best_parameter_change.values()) and max(best_parameter_change.values()) > 0.0, f"BP-ADS fold{fold}: best expert checkpoint did not change")
    checkpoint = {
        "model_state": best_state,
        "fold_lock": lock,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
    }
    torch.save(checkpoint, staging / "checkpoint_best.pt")
    summary = {
        "fold": fold,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "initial_probability_max_abs_diff": initial_diff,
        "protected_cn_count": int(protected_test.sum()),
        "protected_cn_probability_max_abs_diff": 0.0,
        "c1_checkpoint_oof_probability_max_abs_diff": c1_oof_max_diff,
        "expert_max_gradient": maximum_gradient,
        "expert_terminal_parameter_change_by_name": terminal_parameter_change,
        "expert_best_parameter_change_by_name": best_parameter_change,
        "train_only_audit": train_audit,
        "backbone_parameters": EXPECTED_BACKBONE_PARAMETERS,
        "expert_parameters": EXPECTED_EXPERT_PARAMETERS,
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "elapsed_seconds": float(time.perf_counter() - started),
        "single_model": True,
        "ensemble": False,
        "fold_lock": lock,
    }
    write_json(staging / "summary.json", summary)
    write_csv(staging / "oof_predictions.csv", rows)
    write_json(staging / "complete.json", {"complete": True, **lock})
    staging.rename(final_dir)
    print(f"SPECIALIST fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']}", flush=True)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def run_specialist(
    context: dict, output_root: Path, static: dict
) -> tuple[dict, list[dict]]:
    require(static["boundary"]["stage_d_required"], "Stage D is forbidden because no Stage B scheme passed")
    require(static["boundary"]["selected_direction"] == "B1_c1_protect_cn_a012_ads", "BP-ADS v1 implements only the selected B1 direction")
    root = output_root / "specialist_folds"
    root.mkdir(parents=True, exist_ok=True)
    config = specialist_config(context, static)
    config_path = root / "config.json"
    if config_path.is_file():
        require(json.loads(config_path.read_text(encoding="utf-8")) == config, "BP-ADS formal config/source drift")
    else:
        write_json(config_path, config)
    summaries, pooled = [], []
    for fold in FOLDS:
        summary, rows = run_specialist_fold(context, output_root, config, fold)
        summaries.append(summary)
        pooled.extend(rows)
    pooled = validate_rows(pooled)
    probability = np.asarray(
        [[float(row[f"probability_{name}"]) for name in CLASS_NAMES] for row in pooled],
        dtype=np.float64,
    )
    truth = np.asarray([int(row["truth"]) for row in pooled], dtype=np.int64)
    metrics = metric_payload(truth, probability)
    c1_probability = static["probabilities"]["c1"]
    a012_probability = static["probabilities"]["a012"]
    vs_c1 = paired_payload(truth, probability, c1_probability)
    vs_a012 = paired_payload(truth, probability, a012_probability)
    c1_prediction = c1_probability.argmax(axis=1)
    protected = c1_prediction == 1
    require(bool(protected.any()), "C1 protected-CN set is empty")
    protected_prediction_changes = int((probability[protected].argmax(axis=1) != c1_prediction[protected]).sum())
    protected_external_max_diff = float(np.max(np.abs(probability[protected] - c1_probability[protected])))
    require(protected_prediction_changes == 0, "BP-ADS changed a C1-protected CN prediction")
    require(protected_external_max_diff <= 1e-6, f"BP-ADS protected probability differs from C1 OOF: {protected_external_max_diff}")
    fold_acc = [float(summary["best_metrics"]["acc"]) for summary in summaries]
    retained = (
        metrics["correct"] >= 563
        and metrics["AD_CN_errors"] == 0
        and vs_c1["repairs"] > vs_c1["damages"]
        and metrics["bacc"] >= A012["bacc"] - 0.005
    )
    report = {
        "name": "BP_ADS_C1_protected_CN_AD_SMCI_specialist",
        "direction": static["boundary"]["selected_direction"],
        "metrics": metrics,
        "vs_c1": vs_c1,
        "vs_a012": vs_a012,
        "retained": retained,
        "protected_cn": {
            "subject_count": int(protected.sum()),
            "prediction_changes": protected_prediction_changes,
            "internal_probability_max_abs_diff": max(float(summary["protected_cn_probability_max_abs_diff"]) for summary in summaries),
            "external_c1_oof_probability_max_abs_diff": protected_external_max_diff,
            "external_tolerance": 1e-6,
        },
        "fold_acc_mean": float(np.mean(fold_acc)),
        "fold_acc_sample_std": float(np.std(fold_acc, ddof=1)),
        "training_seconds": float(sum(float(summary["elapsed_seconds"]) for summary in summaries)),
        "maximum_expert_gradient": max(float(summary["expert_max_gradient"]) for summary in summaries),
        "backbone_parameters": EXPECTED_BACKBONE_PARAMETERS,
        "expert_parameters": EXPECTED_EXPERT_PARAMETERS,
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "single_model": True,
        "ensemble": False,
        "folds": summaries,
        "config": config,
    }
    write_json(root / "report.json", report)
    write_csv(root / "oof_predictions.csv", pooled)
    write_csv(output_root / "specialist_oof.csv", pooled)
    return report, pooled


def candidate_payload(name: str, kind: str, report: dict) -> dict:
    metrics = report["metrics"]
    vs_c1 = report["vs_c1"]
    strong = (
        metrics["correct"] >= 563
        and metrics["AD_CN_errors"] == 0
        and vs_c1["repairs"] > vs_c1["damages"]
        and metrics["bacc"] >= A012["bacc"] - 0.005
    )
    secondary_improvements = sum(
        float(metrics[key]) > float(A012[key])
        for key in ("macro_auc", "macro_f1", "bacc")
    )
    positive_tie = (
        metrics["correct"] == 562
        and secondary_improvements >= 2
        and metrics["AD_CN_errors"] == 0
        and vs_c1["repairs"] > vs_c1["damages"]
        and metrics["bacc"] >= A012["bacc"] - 0.005
    )
    return {
        "name": name,
        "kind": kind,
        "metrics": metrics,
        "vs_c1": vs_c1,
        "vs_a012": report["vs_a012"],
        "strong_go": strong,
        "positive_tie": positive_tie,
        "secondary_improvements_vs_a012": secondary_improvements,
        "category_performance_risk": metrics["AD_CN_errors"] != 0 or metrics["bacc"] < A012["bacc"] - 0.005,
        "single_model": report["single_model"],
        "ensemble": report["ensemble"],
    }


def candidate_rank(candidate: dict) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (float(metrics["correct"]), float(metrics["macro_auc"]), float(metrics["macro_f1"]))


def complementarity_payload(static: dict) -> dict:
    truth = static["truth"]
    c1_prediction = static["probabilities"]["c1"].argmax(axis=1)
    a012_prediction = static["probabilities"]["a012"].argmax(axis=1)
    c1_correct, a012_correct = c1_prediction == truth, a012_prediction == truth
    return {
        "both_correct": int((c1_correct & a012_correct).sum()),
        "C1_only_correct": int((c1_correct & ~a012_correct).sum()),
        "A012_only_correct": int((~c1_correct & a012_correct).sum()),
        "both_wrong": int((~c1_correct & ~a012_correct).sum()),
        "prediction_changed": int((c1_prediction != a012_prediction).sum()),
        "same_error_set": bool(np.array_equal(~c1_correct, ~a012_correct)),
    }


def metrics_markdown(metrics: dict) -> str:
    counts = metrics.get("predicted_class_counts", {})
    return (
        f"Correct={metrics['correct']}/598, ACC={metrics['acc']:.7f}, "
        f"Macro-F1={metrics['macro_f1']:.7f}, BACC={metrics['bacc']:.7f}, "
        f"Probability Macro-AUC={metrics['macro_auc']:.7f}, Weighted-F1={metrics['weighted_f1']:.7f}; "
        f"confusion={metrics['confusion_matrix']}; "
        f"AD–sMCI={metrics.get('AD_SMCI_errors', 'n/a')}, CN–sMCI={metrics.get('CN_SMCI_errors', 'n/a')}, "
        f"AD–CN={metrics.get('AD_CN_errors', 'n/a')}; predicted={counts or 'n/a'}"
    )


def render_report(summary: dict) -> str:
    boundary = summary["boundary_feasibility"]["schemes"]
    soup_probability = summary["model_soup_results"]["probability_feasibility"]
    soup_formal = summary["formal_candidates"]["weight_soup"]
    specialist = summary["formal_candidates"]["specialist"]
    complementarity = summary["complementarity"]
    selected = summary["selected"]
    lines = [
        "# BP-ADS v1 formal report",
        "",
        "## Protocol",
        "",
        f"Source commit: `{summary['source_commit']}`. Dataset=TADPOLE AD_CN_SMCI; folds=0..9; seed=0; 400 epochs/fold; full-batch transductive; graph disabled.",
        "All formal candidates are independently inferable single models (`single_model=True`, `ensemble=False`). Test labels were excluded from specialist loss and inputs.",
        "",
        "## Stage A — artifact alignment and complementarity",
        "",
        f"All 598 subject IDs were unique and aligned explicitly by `subject_index`; labels and folds matched. C1 and A012 do not fail on exactly the same subjects: both correct={complementarity['both_correct']}, C1-only correct={complementarity['C1_only_correct']}, A012-only correct={complementarity['A012_only_correct']}, both wrong={complementarity['both_wrong']}, changed predictions={complementarity['prediction_changed']}.",
        "",
        "## Stage B — boundary feasibility",
        "",
    ]
    for name in ("B1_c1_protect_cn_a012_ads", "B2_c1_protect_cn_pc_ads", "B3_a012_protect_ad_c1_cn_smci"):
        value = boundary[name]
        if not value.get("available"):
            lines.append(f"- {name}: unavailable.")
        else:
            lines.append(f"- {name}: {metrics_markdown(value['metrics'])}; vs C1 repairs/damages/changed={value['vs_c1']['repairs']}/{value['vs_c1']['damages']}/{value['vs_c1']['changed']}; gate={'PASS' if value['passes_formal_gate'] else 'FAIL'}.")
    lines.extend([
        "",
        f"Selected direction: `{summary['boundary_feasibility']['selected_direction']}`. C1-protected CN + A012 AD/sMCI reached {boundary['B1_c1_protect_cn_a012_ads']['metrics']['correct']}/598; PC-BBF substitution reached {boundary['B2_c1_protect_cn_pc_ads']['metrics']['correct'] if boundary['B2_c1_protect_cn_pc_ads'].get('available') else 'unavailable'}/598.",
        "",
        "## Stage C — model soup",
        "",
    ])
    for name in ("S1_probability_only", "S2_probability_only", "S3_probability_only"):
        value = soup_probability[name]
        if value.get("available"):
            lines.append(f"- {name} ({'+'.join(value['members'])}): probability feasibility only, {metrics_markdown(value['metrics'])}; exact checkpoint rerun={'allowed' if value['permits_exact_checkpoint_rerun'] else 'forbidden'}.")
        else:
            lines.append(f"- {name}: unavailable.")
    lines.extend([
        "",
        f"Formal S3 weight soup: {metrics_markdown(soup_formal['metrics'])}; vs C1 repairs/damages/changed={soup_formal['vs_c1']['repairs']}/{soup_formal['vs_c1']['damages']}/{soup_formal['vs_c1']['changed']}; retained={soup_formal['retained']}. It is one averaged state dict, not a probability ensemble.",
        "",
        "## Stage D — frozen-C1 boundary specialist",
        "",
        f"{metrics_markdown(specialist['metrics'])}; vs C1 repairs/damages/changed={specialist['vs_c1']['repairs']}/{specialist['vs_c1']['damages']}/{specialist['vs_c1']['changed']}; vs A012={specialist['vs_a012']['repairs']}/{specialist['vs_a012']['damages']}/{specialist['vs_a012']['changed']}; retained={specialist['retained']}.",
        f"Fold ACC={specialist['fold_acc_mean']:.7f} ± {specialist['fold_acc_sample_std']:.7f} sample SD; training time={specialist['training_seconds']:.1f}s; maximum expert gradient={specialist['maximum_expert_gradient']:.7g}.",
        f"The model has {specialist['backbone_parameters']} frozen C1 parameters + {specialist['expert_parameters']} expert parameters = {specialist['total_parameters']} total. C1-protected CN prediction changes={specialist['protected_cn']['prediction_changes']}; internal max probability difference={specialist['protected_cn']['internal_probability_max_abs_diff']}; external CSV max difference={specialist['protected_cn']['external_c1_oof_probability_max_abs_diff']:.3g}.",
        "",
        "## Required questions",
        "",
        f"1. C1 and A012 complementarity is not confined to the same subjects: C1 uniquely repairs {complementarity['C1_only_correct']} and A012 uniquely repairs {complementarity['A012_only_correct']}; {complementarity['both_wrong']} are wrong in both.",
        f"2. C1-protected CN + A012 AD/sMCI reached {boundary['B1_c1_protect_cn_a012_ads']['metrics']['correct']}/598 (gate PASS); the PC specialist reached {boundary['B2_c1_protect_cn_pc_ads']['metrics']['correct'] if boundary['B2_c1_protect_cn_pc_ads'].get('available') else 'unavailable'}/598 (gate {'PASS' if boundary['B2_c1_protect_cn_pc_ads'].get('passes_formal_gate') else 'FAIL'}).",
        f"3. The formal weight soup {'outperformed' if soup_formal['metrics']['correct'] > A012['correct'] else 'did not outperform'} A012 in Correct ({soup_formal['metrics']['correct']} vs {A012['correct']}).",
        f"4. The trained specialist reached {specialist['metrics']['correct']}/598 versus the zero-training splice upper-bound {boundary['B1_c1_protect_cn_a012_ads']['metrics']['correct']}/598; it {'reproduced or exceeded' if specialist['metrics']['correct'] >= boundary['B1_c1_protect_cn_a012_ads']['metrics']['correct'] else 'did not fully reproduce'} that gain.",
        f"5. CN predictions were strictly protected: {specialist['protected_cn']['prediction_changes']} protected prediction changes and internal probability max difference {specialist['protected_cn']['internal_probability_max_abs_diff']}.",
        f"6. Relative to C1, formal BP-ADS changed AD–sMCI errors from 21 to {specialist['metrics']['AD_SMCI_errors']} and produced {specialist['metrics']['AD_CN_errors']} AD–CN errors.",
        "7. Yes. Both formal candidates are single models with `ensemble=False`; BP-ADS is one frozen C1 backbone plus one 3,089-parameter expert.",
        f"8. Final selection: `{selected['name']}` ({selected['kind']}); {metrics_markdown(selected['metrics'])}.",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
    ])
    if summary["decision"] == "STOP_STATIC_ROUTE":
        lines.extend(["", "锁定A012为当前最佳静态横断面模型，不再扩大静态网络与超参数搜索；后续提升需要纵向访视、转换时间或新的生物标志物监督。"])
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```text",
        summary["run_commands"]["smoke"],
        summary["run_commands"]["formal"],
        "```",
    ])
    return "\n".join(lines)


def write_final_artifacts(
    context: dict,
    output_root: Path,
    static: dict,
    materialized: dict,
    soup_report: dict,
    specialist_report: dict,
    formal_started: float,
) -> dict:
    soup_candidate = candidate_payload("S3_c1_a012_a007_equal", "weight_soup", soup_report)
    specialist_candidate = candidate_payload("BP_ADS_C1_protected_CN_AD_SMCI_specialist", "bp_ads", specialist_report)
    candidates = [soup_candidate, specialist_candidate]
    strong = sorted((candidate for candidate in candidates if candidate["strong_go"]), key=candidate_rank, reverse=True)
    positive = sorted((candidate for candidate in candidates if candidate["positive_tie"]), key=candidate_rank, reverse=True)
    if strong:
        selected = strong[0]
        decision = "MODEL_SOUP_STRONG_GO" if selected["kind"] == "weight_soup" else "BP_ADS_STRONG_GO"
    elif positive:
        selected = positive[0]
        decision = "POSITIVE_TIE"
    else:
        selected = {
            "name": "A012",
            "kind": "locked_reference",
            "metrics": {**A012, **metric_payload(static["truth"], static["probabilities"]["a012"])},
            "single_model": True,
            "ensemble": False,
            "strong_go": False,
            "positive_tie": False,
        }
        decision = "STOP_STATIC_ROUTE"
    soup_top = {
        "probability_feasibility": static["soup"]["probability_feasibility"],
        "formal_weight_soups": {"S3_c1_a012_a007_equal": soup_report},
        "checkpoint_rerun_allowed_configs": static["soup"]["checkpoint_rerun_allowed_configs"],
    }
    complementarity = complementarity_payload(static)
    fold_results = {
        "checkpoint_materialization": {
            name: report["folds"] for name, report in materialized.items()
        },
        "weight_soup": soup_report["folds"],
        "specialist": specialist_report["folds"],
    }
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    summary = {
        "experiment": EXPERIMENT,
        "decision": decision,
        "selected": selected,
        "formal_candidates": {
            "weight_soup": soup_report,
            "specialist": specialist_report,
            "ranking": [candidate["name"] for candidate in sorted(candidates, key=candidate_rank, reverse=True)],
        },
        "complementarity": complementarity,
        "boundary_feasibility": static["boundary"],
        "model_soup_results": soup_top,
        "references": {"C1": C1, "A012": A012},
        "checkpoint_materialization": materialized,
        "source_commit": source_commit,
        "base_c1_commit": BASE_C1_COMMIT,
        "broad_implementation_commit": BROAD_IMPLEMENTATION_COMMIT,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "formal_session_wall_seconds": float(time.perf_counter() - formal_started),
        "run_commands": {
            "smoke": f'"{sys.executable}" -u -B scripts/run_bp_ads_v1.py smoke --device cuda:0',
            "formal": f'"{sys.executable}" -u -B scripts/run_bp_ads_v1.py formal --device cuda:0',
        },
    }
    write_json(output_root / "model_soup_results.json", soup_top)
    write_json(output_root / "fold_results.json", fold_results)
    write_json(output_root / "formal_summary.json", summary)
    (output_root / "REPORT.md").write_text(render_report(summary) + "\n", encoding="utf-8")
    return summary


def experiment_config() -> dict:
    return {
        "experiment": EXPERIMENT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "class_order": list(CLASS_NAMES),
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "graph": False,
        "stage_b_candidates": [
            "B1_c1_protect_cn_a012_ads",
            "B2_c1_protect_cn_pc_ads",
            "B3_a012_protect_ad_c1_cn_smci",
        ],
        "stage_c_candidates": {
            "S1": {"members": ["c1", "a012"], "weights": [0.5, 0.5]},
            "S2": {"members": ["a012", "a007"], "weights": [0.5, 0.5]},
            "S3": {"members": ["c1", "a012", "a007"], "weights": [1 / 3, 1 / 3, 1 / 3]},
        },
        "specialist": {
            "backbone": "fold-specific frozen C1 best checkpoint",
            "hidden": 8,
            "learning_rate": EXPERT_LR,
            "weight_decay": EXPERT_WEIGHT_DECAY,
            "loss": "class-balanced BCE, train-mask AD/sMCI only",
            "scheduler": "CustomCosineAnnealingLR(T_max=400)",
            "parameters": EXPECTED_EXPERT_PARAMETERS,
        },
        "base_c1_commit": BASE_C1_COMMIT,
        "broad_implementation_commit": BROAD_IMPLEMENTATION_COMMIT,
    }


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode == 0, "Source/static/smoke files must be committed before formal training")
    require(git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode == 0, "Committed source/static/smoke files changed")


def validate_formal_gate(output_root: Path) -> None:
    smoke_report_path = output_root / "smoke/smoke_report.json"
    smoke_config_path = output_root / "smoke/smoke_config.json"
    require(smoke_report_path.is_file() and smoke_config_path.is_file(), "Passing CUDA smoke is required before formal training")
    report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    require(report.get("passed") is True and report.get("epochs") == 3 and report.get("fold") == 0, "Invalid smoke report")
    require(report.get("config") == config, "Smoke report/config drift")
    require(report.get("initial_probability_max_abs_diff", 1.0) <= 1e-7, "Smoke zero-delta equivalence failed")
    require(report.get("checkpoint_strict_load") is True, "Smoke checkpoint strict-load gate failed")
    runner_hash, model_hash = file_sha256(ROOT / RUNNER_REL), file_sha256(ROOT / MODEL_REL)
    require(config.get("runner_sha256") == runner_hash and config.get("model_sha256") == model_hash, "Source differs from smoke-tested code")
    current_head = git("rev-parse", "HEAD").stdout.strip()
    smoke_head = config.get("source_commit_at_smoke")
    require(current_head != smoke_head, "Formal training requires a source commit after smoke")
    require(git("merge-base", "--is-ancestor", str(smoke_head), "HEAD", check=False).returncode == 0, "Smoke commit is not an ancestor of formal source")
    locked_c1_dependencies = [
        ROOT / cme.CONFIG_REL,
        ROOT / "Model/cme_dual_branch.py",
        ROOT / "Model/network.py",
        ROOT / "Loss/loss_fn.py",
        ROOT / "Utils/utils.py",
        ROOT / "scripts/run_cme_dual_branch_v1.py",
    ]
    committed = [
        ROOT / RUNNER_REL,
        ROOT / MODEL_REL,
        ROOT / "scripts/run_c1_broad_hparam_search_v1.py",
        *locked_c1_dependencies,
        output_root / "experiment_config.json",
        output_root / "artifact_inventory.json",
        output_root / "aligned_oof_reference.csv",
        output_root / "boundary_feasibility.json",
        output_root / "model_soup_results.json",
        smoke_report_path,
        smoke_config_path,
    ]
    require_committed_unchanged(committed)
    c1_relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in locked_c1_dependencies]
    require(git("diff", "--quiet", BASE_C1_COMMIT, "HEAD", "--", *c1_relative, check=False).returncode == 0, "Locked C1 dependency differs from base commit")
    require(git("diff", "--quiet", BROAD_IMPLEMENTATION_COMMIT, "HEAD", "--", "scripts/run_c1_broad_hparam_search_v1.py", check=False).returncode == 0, "Broad-search training implementation changed")


def run_formal(context: dict, output_root: Path, static: dict) -> dict:
    validate_formal_gate(output_root)
    require(static["boundary"]["stage_d_required"], "No Stage B boundary scheme reached 563; Stage D must not run")
    s3_probability = static["soup"]["probability_feasibility"].get("S3_probability_only", {})
    require(s3_probability.get("available") and s3_probability.get("permits_exact_checkpoint_rerun"), "S3 probability feasibility did not authorize checkpoint materialization")
    require(set(static["soup"]["checkpoint_rerun_allowed_configs"]) == {"a007", "a012"}, "Only A012 and A007 exact reruns are allowed")
    started = time.perf_counter()
    materialized = {
        "a012": materialize_config(context, output_root, "a012", A012_SPEC),
        "a007": materialize_config(context, output_root, "a007", A007_SPEC),
    }
    soup_report, _ = run_weight_soup(context, output_root, static)
    specialist_report, _ = run_specialist(context, output_root, static)
    return write_final_artifacts(context, output_root, static, materialized, soup_report, specialist_report, started)


def verify_branch() -> None:
    require(git("cat-file", "-e", f"{BASE_C1_COMMIT}^{{commit}}", check=False).returncode == 0, "C1 base commit missing")
    require(git("cat-file", "-e", f"{BROAD_IMPLEMENTATION_COMMIT}^{{commit}}", check=False).returncode == 0, "Broad implementation commit missing")
    require(git("merge-base", "--is-ancestor", BROAD_IMPLEMENTATION_COMMIT, "HEAD", check=False).returncode == 0, "BP-ADS branch is not based on broad-search history")
    require(git("branch", "--show-current").stdout.strip() == BRANCH, f"Wrong branch; expected {BRANCH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_branch()
    require(args.device == "cuda:0", "BP-ADS requires cuda:0")
    output_root = args.output_root.resolve()
    require(output_root == (ROOT / OUTPUT_REL).resolve(), "BP-ADS output path is locked")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_config.json", experiment_config())
    static = prepare_static_artifacts(output_root)
    if args.stage == "prepare":
        print(f"PREPARED subjects={len(static['ids'])} boundary={static['boundary']['selected_direction']} stage_d={static['boundary']['stage_d_required']}", flush=True)
        return
    context = load_context()
    if args.stage == "smoke":
        report = run_smoke(context, output_root, static)
        print(f"SMOKE passed={report['passed']} initial_diff={report['initial_probability_max_abs_diff']} expert_parameters={report['expert_parameters']}", flush=True)
    else:
        summary = run_formal(context, output_root, static)
        print(f"{summary['decision']} selected={summary['selected']['name']} correct={summary['selected']['metrics']['correct']}/598", flush=True)


if __name__ == "__main__":
    main()
