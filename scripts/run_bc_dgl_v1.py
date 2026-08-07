"""Boundary-Conditioned Disentangled Gradient Learning v1 runner.

One smoke suite covers both preregistered gradient-routing arms.  Both arms are
then screened on folds 4/7/8.  Only a screen-qualified winning arm may proceed
to the formal ten-fold result, reusing its three completed screen folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Model.bc_dgl import BCDGLModel
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/bc_dgl_v1")
C1_GIT_REF = "refs/remotes/origin/experiment/cme-dual-branch-v1"
C1_OOF_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
C1_REPORT_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/report.json"
C1_BLOB_SHA256 = "6f4ef505641f75236b01ae128f2836bf2f126ced43158f9c4ee21cc090898018"
SOURCE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
BRANCH = "experiment/bc-dgl-v1"
FOLDS = tuple(range(10))
SCREEN_FOLDS = (4, 7, 8)
FORMAL_NEW_FOLDS = (0, 1, 2, 3, 5, 6, 9)
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 3
LAMBDA_BOUNDARY = 0.25
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
C1_PARAMETERS = 862_971
EXPECTED_TRAINING_PARAMETERS = 865_299
EXPECTED_INFERENCE_PARAMETERS = C1_PARAMETERS
ARM_ORDER = ("s", "m")
ARMS = {
    "s": {"name": "BC-DGL-S", "rho": 0.0},
    "m": {"name": "BC-DGL-M", "rho": 0.25},
}
C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
    "ad_smci_errors": 21,
    "cn_smci_errors": 17,
    "ad_cn_errors": 0,
}
C1_SCREEN = {
    "correct": 166,
    "subjects": 179,
    "acc": 0.9273743016759777,
    "macro_f1": 0.896722857963168,
    "bacc": 0.901364522417154,
    "macro_auc": 0.9432606323024206,
    "weighted_f1": 0.9277668135664757,
    "confusion_matrix": [[17, 0, 4], [0, 61, 2], [5, 2, 88]],
    "ad_smci_errors": 9,
    "cn_smci_errors": 4,
    "boundary_errors": 13,
    "ad_cn_errors": 0,
    "fold_correct": {4: 55, 7: 56, 8: 55},
    "fold_best_epoch": {4: 103, 7: 181, 8: 90},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


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


def git_bytes(ref: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "show",
            f"{ref}:{relative_path}",
        ]
    )


def git_text(ref: str, relative_path: str) -> str:
    return git_bytes(ref, relative_path).decode("utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            *args,
        ],
        text=True,
        encoding="utf-8",
        errors="strict",
    ).strip()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def module_gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def probability_metrics(truth, probability) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    require(probability.shape == (len(truth), 3), "Probability shape mismatch")
    require(np.isfinite(probability).all(), "Probability contains NaN/Inf")
    require(
        np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
        "Probability rows do not sum to one",
    )
    prediction = probability.argmax(axis=1)
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "correct": int((prediction == truth).sum()),
        "acc": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(onehot, probability)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
    }


def metrics_from_rows(rows: list[dict]) -> dict:
    probability = np.asarray(
        [
            [
                float(row["probability_AD"]),
                float(row["probability_CN"]),
                float(row["probability_SMCI"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return probability_metrics([int(row["truth"]) for row in rows], probability)


def score_logits(raw_logits: torch.Tensor, label_weight: torch.Tensor, tau: float):
    adjusted = raw_logits - tau * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    probability = torch.softmax(adjusted, dim=-1)
    return adjusted, probability, probability.argmax(dim=-1)


def selection_metrics(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_weight: torch.Tensor,
    tau: float,
) -> tuple[dict, tuple[float, float, float]]:
    _, probability, _ = score_logits(raw_logits, label_weight, tau)
    metrics = probability_metrics(
        labels[mask].detach().cpu().numpy(),
        probability[mask].detach().cpu().numpy(),
    )
    return metrics, (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])


def boundary_counts(metrics: dict) -> dict:
    matrix = metrics["confusion_matrix"]
    return {
        "ad_smci_errors": int(matrix[0][2] + matrix[2][0]),
        "cn_smci_errors": int(matrix[1][2] + matrix[2][1]),
        "ad_cn_errors": int(matrix[0][1] + matrix[1][0]),
        "boundary_errors": int(
            matrix[0][2] + matrix[2][0] + matrix[1][2] + matrix[2][1]
        ),
    }


def parse_c1_rows() -> tuple[list[dict], dict, dict]:
    blob = git_bytes(C1_GIT_REF, C1_OOF_REL)
    require(hashlib.sha256(blob).hexdigest() == C1_BLOB_SHA256, "C1 OOF blob changed")
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    require(len(rows) == 598, "C1 OOF row count changed")
    require(len({int(row["subject_index"]) for row in rows}) == 598, "C1 OOF subject set changed")
    metrics = metrics_from_rows(rows)
    for key in ("correct", "confusion_matrix"):
        require(metrics[key] == C1[key], f"C1 anchor changed: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    screen_rows = [row for row in rows if int(row["fold"]) in SCREEN_FOLDS]
    require(metrics_from_rows(screen_rows)["confusion_matrix"] == C1_SCREEN["confusion_matrix"], "C1 screen anchor changed")
    report = json.loads(git_text(C1_GIT_REF, C1_REPORT_REL))
    require(int(report["parameter_count"]) == C1_PARAMETERS, "C1 parameter anchor changed")
    return rows, metrics, report


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    require(git_value("rev-parse", "HEAD") == SOURCE_COMMIT, "Source HEAD changed before run")
    require(git_value("branch", "--show-current") == BRANCH, "Experiment branch changed")
    device = torch.device("cuda:0")
    config_root = Path(tempfile.gettempdir()) / "work22_bc_dgl_config"
    config = Config_(str(config_root), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(config.DATA_SET == "TADPOLE" and config.Task == "AD_CN_SMCI", "Protocol changed")
    require(int(config.epochs) == EPOCHS and int(config.T_max) == EPOCHS, "Epoch protocol changed")
    SET_Random(SEED)
    feature_path, dictionary_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    normalized_classes = tuple(str(value).upper() for value in class_names)
    require(normalized_classes == CLASS_NAMES, f"Class order changed: {class_names}")
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dictionary_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    require(tuple(dataset_data["Feature"].shape) == (598, 360), "Dataset shape changed")
    require(len(dataset_data["Mask"]) == 10, "Fold count changed")
    c1_rows, c1_metrics, c1_report = parse_c1_rows()
    return {
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "class_index": {name: normalized_classes.index(name) for name in CLASS_NAMES},
        "c1_rows": c1_rows,
        "c1_metrics": c1_metrics,
        "c1_report": c1_report,
        "c1_by_subject": {int(row["subject_index"]): row for row in c1_rows},
    }


def build_model(context: dict, arm: str, boundary_heads_enabled: bool = True) -> BCDGLModel:
    require(arm in ARMS, f"Unknown arm: {arm}")
    config = context["config"]
    SET_Random(SEED)
    model = BCDGLModel(
        context["dataset_dict"],
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
        query_pool_variant="independent",
        category_branch_fusion="concat",
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
        adapter_rank=8,
        rho=float(ARMS[arm]["rho"]),
        boundary_enabled=boundary_heads_enabled,
    ).to(context["device"])
    expected = EXPECTED_TRAINING_PARAMETERS if boundary_heads_enabled else C1_PARAMETERS
    require(parameter_count(model) == expected, f"Parameter count changed: {parameter_count(model)}")
    require(model.inference_parameter_count() == C1_PARAMETERS, "Inference parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def make_fresh_training_objects(context: dict, arm: str):
    model = build_model(context, arm, boundary_heads_enabled=True)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=float(context["config"].Lr_Min),
    )
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")
    return model, criterion, optimizer, scheduler


def boundary_loss(
    intermediates: dict,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    class_index: dict,
) -> tuple[torch.Tensor, list[dict]]:
    ads_logits = intermediates["boundary_logits_ads"]
    cns_logits = intermediates["boundary_logits_cns"]
    require(tuple(ads_logits.shape) == (labels.numel(), 6, 2), "AD-sMCI head shape changed")
    require(tuple(cns_logits.shape) == (labels.numel(), 6, 2), "CN-sMCI head shape changed")
    ad = train_mask & (labels == class_index["AD"])
    cn = train_mask & (labels == class_index["CN"])
    smci = train_mask & (labels == class_index["SMCI"])
    require(bool(ad.any() and cn.any() and smci.any()), "Boundary train class empty")
    modality_losses = []
    diagnostics = []
    for modality_index, modality in enumerate(MODALITY_NAMES):
        ads = 0.5 * F.cross_entropy(
            ads_logits[ad, modality_index],
            torch.zeros(int(ad.sum()), device=labels.device, dtype=torch.long),
        ) + 0.5 * F.cross_entropy(
            ads_logits[smci, modality_index],
            torch.ones(int(smci.sum()), device=labels.device, dtype=torch.long),
        )
        cns = 0.5 * F.cross_entropy(
            cns_logits[cn, modality_index],
            torch.zeros(int(cn.sum()), device=labels.device, dtype=torch.long),
        ) + 0.5 * F.cross_entropy(
            cns_logits[smci, modality_index],
            torch.ones(int(smci.sum()), device=labels.device, dtype=torch.long),
        )
        modality_losses.append(0.5 * (ads + cns))
        diagnostics.append(
            {
                "modality": modality,
                "loss_ads": float(ads.detach().cpu()),
                "loss_cns": float(cns.detach().cpu()),
            }
        )
    return torch.stack(modality_losses).mean(), diagnostics


def prediction_rows(
    fold: int,
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    dataset_dict: dict,
    config,
) -> list[dict]:
    raw = raw_logits[mask]
    adjusted, probability, prediction = score_logits(
        raw, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
    )
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    raw_values = raw.detach().cpu().numpy()
    adjusted_values = adjusted.detach().cpu().numpy()
    probability_values = probability.detach().cpu().numpy()
    prediction_values = prediction.detach().cpu().numpy().astype(np.int64)
    rows = []
    for row_index, subject_index in enumerate(source_indices):
        rows.append(
            {
                "fold": fold,
                "subject_index": int(subject_index),
                "truth": int(truth[row_index]),
                "prediction": int(prediction_values[row_index]),
                "raw_logit_AD": float(raw_values[row_index, 0]),
                "raw_logit_CN": float(raw_values[row_index, 1]),
                "raw_logit_SMCI": float(raw_values[row_index, 2]),
                "adjusted_score_AD": float(adjusted_values[row_index, 0]),
                "adjusted_score_CN": float(adjusted_values[row_index, 1]),
                "adjusted_score_SMCI": float(adjusted_values[row_index, 2]),
                "probability_AD": float(probability_values[row_index, 0]),
                "probability_CN": float(probability_values[row_index, 1]),
                "probability_SMCI": float(probability_values[row_index, 2]),
            }
        )
    return rows


def auxiliary_prediction_rows(
    fold: int,
    intermediates: dict,
    labels: torch.Tensor,
    test_mask: torch.Tensor,
    dataset_dict: dict,
    class_index: dict,
) -> list[dict]:
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
    logits_by_boundary = {
        "AD_SMCI": intermediates["boundary_logits_ads"],
        "CN_SMCI": intermediates["boundary_logits_cns"],
    }
    negative_class = {
        "AD_SMCI": class_index["AD"],
        "CN_SMCI": class_index["CN"],
    }
    smci = class_index["SMCI"]
    rows = []
    for boundary, logits in logits_by_boundary.items():
        include = test_mask & (
            (labels == negative_class[boundary]) | (labels == smci)
        )
        indices = torch.where(include)[0]
        truth_binary = (labels[indices] == smci).to(torch.int64)
        for modality_index, modality in enumerate(MODALITY_NAMES):
            probability = torch.softmax(logits[indices, modality_index], dim=-1)[:, 1]
            for local_index, node_index in enumerate(indices.detach().cpu().tolist()):
                rows.append(
                    {
                        "fold": fold,
                        "subject_index": int(source_indices[node_index]),
                        "boundary": boundary,
                        "modality": modality,
                        "truth_binary": int(truth_binary[local_index].detach().cpu()),
                        "probability_sMCI": float(probability[local_index].detach().cpu()),
                    }
                )
    return rows


def auxiliary_auc(aux_rows: list[dict]) -> dict:
    output = {}
    for modality in MODALITY_NAMES:
        output[modality] = {}
        for boundary in ("AD_SMCI", "CN_SMCI"):
            selected = [
                row
                for row in aux_rows
                if row["modality"] == modality and row["boundary"] == boundary
            ]
            truth = np.asarray([int(row["truth_binary"]) for row in selected], dtype=np.int64)
            probability = np.asarray(
                [float(row["probability_sMCI"]) for row in selected], dtype=np.float64
            )
            require(len(selected) > 0 and len(np.unique(truth)) == 2, f"Aux AUC class missing: {modality}/{boundary}")
            output[modality][boundary] = float(roc_auc_score(truth, probability))
    return output


def fold_config(context: dict, arm: str, fold: int, scope: str) -> dict:
    return {
        "experiment": "Boundary-Conditioned Disentangled Gradient Learning v1",
        "scope": scope,
        "arm": ARMS[arm]["name"],
        "rho": ARMS[arm]["rho"],
        "lambda_boundary": LAMBDA_BOUNDARY,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "subjects": 598,
        "features": 360,
        "class_order": list(CLASS_NAMES),
        "modality_order": list(MODALITY_NAMES),
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "graph_enabled": False,
        "orthogonality": False,
        "optimizer": "Adam",
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "c1_loss": "historical weighted main CE + three OVR auxiliary losses",
        "boundary_loss": "six-modality balanced AD-sMCI/CN-sMCI auxiliary CE",
        "historical_noise_order": "Feature_Modal -> feature-space directional noise -> Modal Encoder",
        "boundary_clean_path": "a separate clean Modal Encoder pass before noise and modal gate",
        "boundary_clean_feature_input_detached": True,
        "fusion_gradient_bridge": "u_noisy.detach() + rho * (u_noisy - u_noisy.detach())",
        "training_parameters": EXPECTED_TRAINING_PARAMETERS,
        "inference_parameters": EXPECTED_INFERENCE_PARAMETERS,
        "auxiliary_heads_used_at_inference": False,
    }


def paired_comparison(candidate_rows: list[dict], c1_by_subject: dict) -> dict:
    repairs = []
    damages = []
    changed = []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        reference = c1_by_subject[subject]
        require(int(reference["fold"]) == int(row["fold"]), "C1/candidate fold mismatch")
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth, "C1/candidate truth mismatch")
        old_prediction = int(reference["prediction"])
        new_prediction = int(row["prediction"])
        if old_prediction != truth and new_prediction == truth:
            repairs.append(subject)
        if old_prediction == truth and new_prediction != truth:
            damages.append(subject)
        if old_prediction != new_prediction:
            changed.append(subject)
    return {
        "repairs": len(repairs),
        "damages": len(damages),
        "net_repairs": len(repairs) - len(damages),
        "changed_predictions": len(changed),
        "repair_subject_indices": repairs,
        "damage_subject_indices": damages,
        "changed_subject_indices": changed,
    }


def common_state(model: BCDGLModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("boundary_heads_")
    }


def boundary_head_state(model: BCDGLModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("boundary_heads_")
    }


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke: {smoke_root}")
    smoke_root.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]

    strict = build_model(context, "s", True)
    mild = build_model(context, "m", True)
    c1_reference = build_model(context, "s", False)
    strict_common = common_state(strict)
    mild_common = common_state(mild)
    c1_common = common_state(c1_reference)
    require(strict_common.keys() == mild_common.keys() == c1_common.keys(), "Common state keys changed")
    require(
        all(torch.equal(strict_common[key], mild_common[key]) for key in strict_common),
        "Strict/mild C1 initialization differs",
    )
    require(
        all(torch.equal(strict_common[key], c1_common[key]) for key in strict_common),
        "Boundary heads changed C1 initialization",
    )
    strict_boundary = boundary_head_state(strict)
    mild_boundary = boundary_head_state(mild)
    require(strict_boundary.keys() == mild_boundary.keys(), "Strict/mild boundary keys differ")
    require(
        all(torch.equal(strict_boundary[key], mild_boundary[key]) for key in strict_boundary),
        "Strict/mild boundary-head initialization differs",
    )
    strict.eval()
    mild.eval()
    c1_reference.eval()
    with torch.no_grad():
        c1_raw, _, _ = c1_reference(features)
        strict_raw, _, _, strict_inter = strict(
            features, return_intermediates=True, compute_boundary=True
        )
        mild_raw, _, _, mild_inter = mild(
            features, return_intermediates=True, compute_boundary=True
        )
    strict_logit_diff = float((strict_raw - c1_raw).abs().max().cpu())
    mild_logit_diff = float((mild_raw - c1_raw).abs().max().cpu())
    require(max(strict_logit_diff, mild_logit_diff) <= 1e-6, "BC-DGL changed C1 logits")
    strict_forward_error = float(strict_inter["fusion_forward_max_abs_error"].cpu())
    mild_forward_error = float(mild_inter["fusion_forward_max_abs_error"].cpu())
    require(max(strict_forward_error, mild_forward_error) <= 1e-7, "rho bridge changed forward values")

    original_boundary_loss, _ = boundary_loss(
        strict_inter, labels, train_mask, context["class_index"]
    )
    altered_labels = labels.clone()
    altered_labels[test_mask] = (altered_labels[test_mask] + 1) % 3
    altered_boundary_loss, _ = boundary_loss(
        strict_inter, altered_labels, train_mask, context["class_index"]
    )
    require(
        torch.equal(original_boundary_loss, altered_boundary_loss),
        "Test labels entered boundary loss",
    )
    del strict, mild, c1_reference, strict_raw, mild_raw, c1_raw
    torch.cuda.empty_cache()

    isolated = {}
    for arm in ARM_ORDER:
        model = build_model(context, arm, True)
        criterion = criterion_query_pool_no_orth(
            context["dataset_dict"], context["device"], label_smoothing=0.05
        )
        model.train()
        model.zero_grad(set_to_none=True)
        raw, branches, auxiliary, inter = model(
            features, return_intermediates=True, compute_boundary=False
        )
        inter["noisy_modal_tokens"].retain_grad()
        inter["u_fusion"].retain_grad()
        main_loss = criterion(raw, labels, train_mask, branches, auxiliary)
        main_loss.backward()
        require(bool(torch.isfinite(main_loss)) and gradients_finite(model), f"{arm} main gradient non-finite")
        noisy_gradient = (
            0.0
            if inter["noisy_modal_tokens"].grad is None
            else float(inter["noisy_modal_tokens"].grad.detach().norm().cpu())
        )
        fusion_gradient = float(inter["u_fusion"].grad.detach().norm().cpu())
        encoder_gradient = module_gradient_norm(model.modal_token_encoder)
        downstream = {
            "modal_transformer": module_gradient_norm(model.shared_transformer),
            "private_residual": module_gradient_norm(model.private_adapters),
            "query": module_gradient_norm(model.label_pools),
            "global": module_gradient_norm(model.Global_Message),
            "difformer": module_gradient_norm(model.GCN),
            "classifier": module_gradient_norm(model.GCN.classifier),
        }
        require(all(value > 0.0 for value in downstream.values()), f"{arm} downstream main gradient missing")
        if arm == "s":
            require(encoder_gradient == 0.0 and noisy_gradient == 0.0, "rho=0 main gradient entered encoder")
        else:
            require(encoder_gradient > 0.0 and noisy_gradient > 0.0, "rho=0.25 encoder gradient missing")
            require(fusion_gradient > 0.0, "rho=0.25 fusion gradient missing")
            gradient_ratio = noisy_gradient / fusion_gradient
            require(abs(gradient_ratio - 0.25) <= 1e-5, f"rho gradient ratio changed: {gradient_ratio}")
        isolated[f"main_{arm}"] = {
            "loss": float(main_loss.detach().cpu()),
            "modal_encoder_gradient": encoder_gradient,
            "noisy_token_gradient": noisy_gradient,
            "fusion_token_gradient": fusion_gradient,
            "observed_rho_ratio": 0.0 if fusion_gradient == 0.0 else noisy_gradient / fusion_gradient,
            "downstream_gradients": downstream,
        }
        del model, criterion, raw, branches, auxiliary, inter
        torch.cuda.empty_cache()

    boundary_model = build_model(context, "s", True)
    boundary_model.train()
    boundary_model.zero_grad(set_to_none=True)
    _, _, _, boundary_inter = boundary_model(
        features, return_intermediates=True, compute_boundary=True
    )
    isolated_boundary_loss, _ = boundary_loss(
        boundary_inter, labels, train_mask, context["class_index"]
    )
    isolated_boundary_loss.backward()
    require(bool(torch.isfinite(isolated_boundary_loss)) and gradients_finite(boundary_model), "Boundary gradient non-finite")
    encoder_modal_gradients = [
        module_gradient_norm(module)
        for module in boundary_model.modal_token_encoder.modal_proj
    ]
    ads_head_gradients = [
        module_gradient_norm(module) for module in boundary_model.boundary_heads_ads
    ]
    cns_head_gradients = [
        module_gradient_norm(module) for module in boundary_model.boundary_heads_cns
    ]
    require(all(value > 0.0 for value in encoder_modal_gradients), "A modal encoder received no boundary gradient")
    require(all(value > 0.0 for value in ads_head_gradients + cns_head_gradients), "A boundary head received no gradient")
    forbidden_boundary_gradients = {
        "modal_gate": 0.0 if boundary_model.modal_gate_logit.grad is None else float(boundary_model.modal_gate_logit.grad.norm().cpu()),
        "modal_transformer": module_gradient_norm(boundary_model.shared_transformer),
        "private_residual": module_gradient_norm(boundary_model.private_adapters),
        "query": module_gradient_norm(boundary_model.label_pools),
        "global": module_gradient_norm(boundary_model.Global_Message),
        "difformer": module_gradient_norm(boundary_model.GCN),
        "classifier": module_gradient_norm(boundary_model.GCN.classifier),
    }
    require(all(value == 0.0 for value in forbidden_boundary_gradients.values()), "Boundary loss entered downstream C1")
    isolated["boundary"] = {
        "loss": float(isolated_boundary_loss.detach().cpu()),
        "encoder_modal_gradients": dict(zip(MODALITY_NAMES, encoder_modal_gradients)),
        "ads_head_gradients": dict(zip(MODALITY_NAMES, ads_head_gradients)),
        "cns_head_gradients": dict(zip(MODALITY_NAMES, cns_head_gradients)),
        "forbidden_downstream_gradients": forbidden_boundary_gradients,
    }
    del boundary_model, boundary_inter
    torch.cuda.empty_cache()

    smoke_epoch_rows = []
    for arm in ARM_ORDER:
        model, criterion, optimizer, scheduler = make_fresh_training_objects(context, arm)
        for epoch in range(1, SMOKE_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            raw, branches, auxiliary, inter = model(
                features, return_intermediates=True, compute_boundary=True
            )
            c1_loss = criterion(raw, labels, train_mask, branches, auxiliary)
            auxiliary_loss, _ = boundary_loss(
                inter, labels, train_mask, context["class_index"]
            )
            total_loss = c1_loss + LAMBDA_BOUNDARY * auxiliary_loss
            require(bool(torch.isfinite(total_loss)), f"{arm} smoke loss non-finite")
            total_loss.backward()
            require(gradients_finite(model), f"{arm} smoke gradient non-finite")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
            optimizer.step()
            scheduler.step()
            smoke_epoch_rows.append(
                {
                    "arm": ARMS[arm]["name"],
                    "rho": ARMS[arm]["rho"],
                    "epoch": epoch,
                    "c1_loss": float(c1_loss.detach().cpu()),
                    "boundary_loss": float(auxiliary_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                }
            )
        del model, criterion, optimizer, scheduler
        torch.cuda.empty_cache()

    payload = {
        "passed": True,
        "epochs_per_arm": SMOKE_EPOCHS,
        "strict_logit_diff_vs_c1": strict_logit_diff,
        "mild_logit_diff_vs_c1": mild_logit_diff,
        "strict_fusion_forward_error": strict_forward_error,
        "mild_fusion_forward_error": mild_forward_error,
        "test_label_invariance": True,
        "class_indices_from_names": context["class_index"],
        "strict_mild_c1_initialization_identical": True,
        "strict_mild_boundary_head_initialization_identical": True,
        "isolated_gradients": isolated,
        "finite": True,
    }
    write_json(smoke_root / "summary.json", payload)
    write_csv(smoke_root / "epoch_metrics.csv", smoke_epoch_rows)
    print("SMOKE PASS", flush=True)
    return payload


def train_fold(
    context: dict,
    arm: str,
    fold: int,
    scope_root: Path,
    scope: str,
) -> tuple[dict, list[dict], list[dict]]:
    final_dir = scope_root / f"fold_{fold:02d}"
    staging_dir = scope_root / f".fold_{fold:02d}_in_progress"
    require(not final_dir.exists() and not staging_dir.exists(), f"Fold output exists: {scope}/{arm}/{fold}")
    staging_dir.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    model, criterion, optimizer, scheduler = make_fresh_training_objects(context, arm)
    config_payload = fold_config(context, arm, fold, scope)
    best = None
    best_state = None
    epoch_rows = []
    ads_head_ever = np.zeros(6, dtype=bool)
    cns_head_ever = np.zeros(6, dtype=bool)
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features, return_intermediates=True, compute_boundary=True
        )
        c1_loss = criterion(raw, labels, train_mask, branches, auxiliary)
        auxiliary_loss, boundary_details = boundary_loss(
            intermediates, labels, train_mask, context["class_index"]
        )
        total_loss = c1_loss + LAMBDA_BOUNDARY * auxiliary_loss
        require(bool(torch.isfinite(total_loss)), f"{arm} fold{fold} epoch{epoch}: non-finite loss")
        total_loss.backward()
        require(gradients_finite(model), f"{arm} fold{fold} epoch{epoch}: non-finite gradient")
        ads_gradients = np.asarray(
            [module_gradient_norm(module) for module in model.boundary_heads_ads],
            dtype=np.float64,
        )
        cns_gradients = np.asarray(
            [module_gradient_norm(module) for module in model.boundary_heads_cns],
            dtype=np.float64,
        )
        ads_head_ever |= ads_gradients > 0.0
        cns_head_ever |= cns_gradients > 0.0
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _ = model(features, compute_boundary=False)
            test_metrics, score = selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
            }
            best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "c1_loss": float(c1_loss.detach().cpu()),
                "boundary_loss": float(auxiliary_loss.detach().cpu()),
                "weighted_boundary_loss": float((LAMBDA_BOUNDARY * auxiliary_loss).detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
                **{
                    f"loss_AD_SMCI_{item['modality']}": item["loss_ads"]
                    for item in boundary_details
                },
                **{
                    f"loss_CN_SMCI_{item['modality']}": item["loss_cns"]
                    for item in boundary_details
                },
            }
        )

    require(best is not None and best_state is not None, f"{arm} fold{fold}: no best state")
    require(bool(ads_head_ever.all() and cns_head_ever.all()), f"{arm} fold{fold}: a boundary head never received gradient")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(
            features, return_intermediates=True, compute_boundary=True
        )
        best_metrics, _ = selection_metrics(
            best_raw,
            labels,
            test_mask,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
        rows = prediction_rows(
            fold,
            best_raw,
            labels,
            test_mask,
            context["dataset_dict"],
            context["config"],
        )
        aux_rows = auxiliary_prediction_rows(
            fold,
            best_intermediates,
            labels,
            test_mask,
            context["dataset_dict"],
            context["class_index"],
        )
    require(best_metrics == best["metrics"], f"{arm} fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"{arm} fold{fold}: prediction readback mismatch")
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "scope": scope,
        "arm": ARMS[arm]["name"],
        "arm_key": arm,
        "rho": ARMS[arm]["rho"],
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "fresh_model": True,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "boundary_counts": boundary_counts(best_metrics),
        "training_parameters": parameter_count(model),
        "inference_parameters": model.inference_parameter_count(),
        "boundary_head_parameters": model.boundary_head_parameter_count(),
        "boundary_head_gradient_nonzero": {
            modality: {
                "AD_SMCI": bool(ads_head_ever[index]),
                "CN_SMCI": bool(cns_head_ever[index]),
            }
            for index, modality in enumerate(MODALITY_NAMES)
        },
        "auxiliary_auc": auxiliary_auc(aux_rows),
        "elapsed_seconds": elapsed,
        "config": config_payload,
    }
    torch.save(best_state, staging_dir / "checkpoint_best.pt")
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_json(staging_dir / "config.json", config_payload)
    write_json(staging_dir / "summary.json", summary)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    write_csv(staging_dir / "auxiliary_predictions.csv", aux_rows)
    write_csv(
        staging_dir / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(best_metrics["confusion_matrix"])
        ],
    )
    staging_dir.rename(final_dir)
    print(
        f"{ARMS[arm]['name']} fold={fold} best_epoch={best['epoch']} "
        f"correct={best_metrics['correct']} ACC={best_metrics['acc']:.7f} "
        f"Macro-F1={best_metrics['macro_f1']:.7f} BACC={best_metrics['bacc']:.7f} "
        f"Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows, aux_rows


def aggregate_fold_outputs(
    arm: str,
    fold_summaries: list[dict],
    rows: list[dict],
    aux_rows: list[dict],
    expected_subjects: int,
) -> dict:
    rows.sort(key=lambda row: int(row["subject_index"]))
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(rows) == expected_subjects, f"Expected {expected_subjects} prediction rows")
    require(len(set(subjects)) == expected_subjects, "OOF subjects are not unique")
    metrics = metrics_from_rows(rows)
    counts = boundary_counts(metrics)
    fold_acc = np.asarray(
        [summary["best_metrics"]["acc"] for summary in fold_summaries],
        dtype=np.float64,
    )
    gradient_flags = {
        modality: {
            boundary: bool(
                all(
                    summary["boundary_head_gradient_nonzero"][modality][boundary]
                    for summary in fold_summaries
                )
            )
            for boundary in ("AD_SMCI", "CN_SMCI")
        }
        for modality in MODALITY_NAMES
    }
    return {
        "arm": ARMS[arm]["name"],
        "arm_key": arm,
        "rho": ARMS[arm]["rho"],
        "training_parameters": EXPECTED_TRAINING_PARAMETERS,
        "inference_parameters": EXPECTED_INFERENCE_PARAMETERS,
        "metrics": metrics,
        **counts,
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_sd": float(fold_acc.std(ddof=1)),
        "fold_results": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "correct": summary["best_metrics"]["correct"],
                "acc": summary["best_metrics"]["acc"],
                "macro_f1": summary["best_metrics"]["macro_f1"],
                "bacc": summary["best_metrics"]["bacc"],
                "macro_auc": summary["best_metrics"]["macro_auc"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "reused_from_screen": bool(summary.get("reused_from_screen", False)),
            }
            for summary in fold_summaries
        ],
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in fold_summaries)),
        "auxiliary_auc": auxiliary_auc(aux_rows),
        "boundary_head_gradient_nonzero": gradient_flags,
    }


def screen_pass(report: dict) -> bool:
    return bool(
        report["metrics"]["correct"] >= C1_SCREEN["correct"]
        and report["boundary_errors"] <= C1_SCREEN["boundary_errors"]
        and report["ad_cn_errors"] <= C1_SCREEN["ad_cn_errors"]
    )


def screen_rank(report: dict) -> tuple:
    metrics = report["metrics"]
    return (
        metrics["correct"],
        metrics["bacc"],
        metrics["macro_f1"],
        metrics["macro_auc"],
        1 if report["arm_key"] == "s" else 0,
    )


def run_screen(context: dict, output_root: Path) -> tuple[dict, dict[str, dict]]:
    screen_root = output_root / "screen"
    require(not screen_root.exists(), f"Refusing to overwrite screen: {screen_root}")
    screen_root.mkdir(parents=True)
    reports = {}
    for arm in ARM_ORDER:
        arm_root = screen_root / ARMS[arm]["name"].lower().replace("-", "_")
        arm_root.mkdir(parents=True)
        summaries = []
        rows = []
        aux_rows = []
        for fold in SCREEN_FOLDS:
            summary, fold_rows, fold_aux_rows = train_fold(
                context, arm, fold, arm_root, "screen"
            )
            summaries.append(summary)
            rows.extend(fold_rows)
            aux_rows.extend(fold_aux_rows)
        report = aggregate_fold_outputs(arm, summaries, rows, aux_rows, C1_SCREEN["subjects"])
        report["modality_boundary_evidence"] = modality_evidence(
            report["auxiliary_auc"], report["boundary_head_gradient_nonzero"]
        )
        report["passes_screen"] = screen_pass(report)
        report["c1_screen_anchor"] = C1_SCREEN
        reports[arm] = {
            "report": report,
            "fold_summaries": summaries,
            "rows": rows,
            "aux_rows": aux_rows,
            "root": arm_root,
        }
        write_csv(arm_root / "oof_predictions.csv", rows)
        write_csv(arm_root / "auxiliary_predictions.csv", aux_rows)
        write_json(arm_root / "metrics.json", report["metrics"])
        write_json(arm_root / "report.json", report)

    qualified = [arm for arm in ARM_ORDER if reports[arm]["report"]["passes_screen"]]
    selected = max(qualified, key=lambda arm: screen_rank(reports[arm]["report"])) if qualified else None
    ranking = sorted(ARM_ORDER, key=lambda arm: screen_rank(reports[arm]["report"]), reverse=True)
    payload = {
        "decision": "SCREEN_PASS" if selected is not None else "BC_DGL_SCREEN_STOP",
        "selected_arm": None if selected is None else ARMS[selected]["name"],
        "selected_arm_key": selected,
        "ranking": [ARMS[arm]["name"] for arm in ranking],
        "hard_gate": {
            "correct_minimum": C1_SCREEN["correct"],
            "boundary_errors_maximum": C1_SCREEN["boundary_errors"],
            "ad_cn_errors_maximum": C1_SCREEN["ad_cn_errors"],
        },
        "arms": {arm: reports[arm]["report"] for arm in ARM_ORDER},
    }
    write_json(screen_root / "summary.json", payload)
    return payload, reports


def checkpoint_readback(
    context: dict,
    arm: str,
    fold: int,
    fold_dir: Path,
) -> dict:
    model = build_model(context, arm, True)
    state = torch.load(
        fold_dir / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][fold]
    with torch.no_grad():
        raw, _, _ = model(features, compute_boundary=False)
        reloaded_rows = prediction_rows(
            fold,
            raw,
            labels,
            test_mask,
            context["dataset_dict"],
            context["config"],
        )
    saved_rows = read_csv(fold_dir / "best_predictions.csv")
    require(len(saved_rows) == len(reloaded_rows), "Checkpoint readback row count changed")
    max_probability_diff = 0.0
    for saved, reloaded in zip(saved_rows, reloaded_rows):
        require(int(saved["subject_index"]) == int(reloaded["subject_index"]), "Checkpoint subject order changed")
        require(int(saved["prediction"]) == int(reloaded["prediction"]), "Checkpoint prediction changed")
        for class_name in CLASS_NAMES:
            max_probability_diff = max(
                max_probability_diff,
                abs(float(saved[f"probability_{class_name}"]) - float(reloaded[f"probability_{class_name}"])),
            )
    require(max_probability_diff <= 1e-7, "Checkpoint probability readback changed")
    del model
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "fold": fold,
        "rows": len(saved_rows),
        "max_probability_abs_diff": max_probability_diff,
    }


def modality_evidence(aux_auc_report: dict, gradient_flags: dict) -> dict:
    by_modality = {}
    for modality in MODALITY_NAMES:
        ads = float(aux_auc_report[modality]["AD_SMCI"])
        cns = float(aux_auc_report[modality]["CN_SMCI"])
        mean_auc = 0.5 * (ads + cns)
        effective = bool(ads > 0.50 and cns > 0.50 and mean_auc > 0.55)
        by_modality[modality] = {
            "AD_SMCI_AUC": ads,
            "CN_SMCI_AUC": cns,
            "mean_AUC": mean_auc,
            "effective_above_random": effective,
            "boundary_head_gradient_nonzero": bool(
                gradient_flags[modality]["AD_SMCI"]
                and gradient_flags[modality]["CN_SMCI"]
            ),
        }
    non_cog = [modality for modality in MODALITY_NAMES if modality != "COG"]
    non_cog_mean = float(np.mean([by_modality[modality]["mean_AUC"] for modality in non_cog]))
    cog_mean = by_modality["COG"]["mean_AUC"]
    effective_non_cog = [
        modality for modality in non_cog if by_modality[modality]["effective_above_random"]
    ]
    return {
        "by_modality": by_modality,
        "COG_mean_AUC": cog_mean,
        "non_COG_mean_AUC": non_cog_mean,
        "COG_minus_non_COG_AUC_gap": float(cog_mean - non_cog_mean),
        "effective_non_COG_modalities": effective_non_cog,
        "effective_non_COG_count": len(effective_non_cog),
        "COG_only": bool(by_modality["COG"]["effective_above_random"] and len(effective_non_cog) < 2),
        "effectiveness_rule": "both boundary AUCs >0.50 and their mean >0.55",
    }


def formal_decision(report: dict) -> tuple[str, str]:
    metrics = report["metrics"]
    comparison = report["comparison_vs_c1"]
    evidence = report["modality_boundary_evidence"]
    major_metric_declines = sum(
        metrics[key] < C1[key] for key in ("macro_f1", "bacc", "macro_auc")
    )
    stop = (
        metrics["correct"] <= 559
        or report["ad_smci_errors"] >= C1["ad_smci_errors"] + 3
        or report["cn_smci_errors"] >= C1["cn_smci_errors"] + 3
        or evidence["COG_only"]
        or report["ad_cn_errors"] > C1["ad_cn_errors"]
        or (
            comparison["repairs"] <= comparison["damages"]
            and major_metric_declines >= 2
        )
    )
    go = (
        metrics["correct"] >= 562
        and metrics["macro_f1"] >= 0.9175457
        and metrics["bacc"] >= 0.9163359
        and report["ad_smci_errors"] <= 21
        and report["cn_smci_errors"] <= 17
        and report["boundary_errors"] <= 36
        and comparison["repairs"] > comparison["damages"]
        and report["ad_cn_errors"] == 0
    )
    near = (
        metrics["correct"] in {560, 561}
        and report["ad_smci_errors"] < C1["ad_smci_errors"] + 3
        and report["cn_smci_errors"] < C1["cn_smci_errors"] + 3
        and evidence["effective_non_COG_count"] >= 2
        and (
            metrics["macro_f1"] > C1["macro_f1"]
            or metrics["bacc"] > C1["bacc"]
            or metrics["macro_auc"] > C1["macro_auc"]
        )
    )
    if go and not stop:
        return "BC_DGL_GO", "Retain this single model; do not launch another experiment automatically."
    if near and not stop:
        return "BC_DGL_NEAR", "A future single adjustment may lower lambda_boundary from 0.25 to 0.10; it was not run."
    return "BC_DGL_STOP", "Stop BC-DGL; do not automatically implement a graph, prototype, gate, or another network."


def validate_screen_fold_for_reuse(
    context: dict,
    selected_arm: str,
    summary: dict,
    source_dir: Path,
    fold_rows: list[dict],
    fold_aux_rows: list[dict],
) -> None:
    fold = int(summary["fold"])
    required_files = (
        "checkpoint_best.pt",
        "config.json",
        "summary.json",
        "epoch_metrics.csv",
        "best_predictions.csv",
        "auxiliary_predictions.csv",
    )
    require(all((source_dir / name).is_file() for name in required_files), f"Screen fold{fold} artifact missing")
    require(summary["scope"] == "screen", f"Screen fold{fold} scope changed")
    require(summary["arm_key"] == selected_arm, f"Screen fold{fold} arm changed")
    require(float(summary["rho"]) == float(ARMS[selected_arm]["rho"]), f"Screen fold{fold} rho changed")
    require(int(summary["seed"]) == SEED and int(summary["epochs"]) == EPOCHS, f"Screen fold{fold} protocol changed")
    require(bool(summary["fresh_model"]), f"Screen fold{fold} was not fresh")
    config = summary["config"]
    require(config["optimizer"] == "Adam", f"Screen fold{fold} optimizer changed")
    require(config["scheduler"] == "CustomCosineAnnealingLR", f"Screen fold{fold} scheduler changed")
    require(int(config["T_max"]) == EPOCHS, f"Screen fold{fold} T_max changed")
    require(int(config["seed"]) == SEED and int(config["epochs"]) == EPOCHS, f"Screen fold{fold} config changed")
    require(float(config["rho"]) == float(ARMS[selected_arm]["rho"]), f"Screen fold{fold} config rho changed")
    require(len(read_csv(source_dir / "epoch_metrics.csv")) == EPOCHS, f"Screen fold{fold} epoch history incomplete")
    persisted_rows = read_csv(source_dir / "best_predictions.csv")
    persisted_aux = read_csv(source_dir / "auxiliary_predictions.csv")
    require(
        [int(row["subject_index"]) for row in persisted_rows]
        == [int(row["subject_index"]) for row in fold_rows],
        f"Screen fold{fold} prediction rows changed",
    )
    require(len(persisted_aux) == len(fold_aux_rows), f"Screen fold{fold} auxiliary rows changed")
    _, test_mask = context["dataset_data"]["Mask"][fold]
    require(len(fold_rows) == int(test_mask.sum()), f"Screen fold{fold} test size changed")


def run_formal(
    context: dict,
    output_root: Path,
    selected_arm: str,
    screen_entry: dict,
) -> dict:
    formal_root = output_root / "formal"
    require(not formal_root.exists(), f"Refusing to overwrite formal: {formal_root}")
    formal_root.mkdir(parents=True)
    fold_summaries = []
    rows = []
    aux_rows = []

    for summary, fold_rows, fold_aux_rows in zip(
        screen_entry["fold_summaries"],
        [
            [row for row in screen_entry["rows"] if int(row["fold"]) == fold]
            for fold in SCREEN_FOLDS
        ],
        [
            [row for row in screen_entry["aux_rows"] if int(row["fold"]) == fold]
            for fold in SCREEN_FOLDS
        ],
    ):
        fold = int(summary["fold"])
        source_dir = screen_entry["root"] / f"fold_{fold:02d}"
        destination = formal_root / f"fold_{fold:02d}"
        validate_screen_fold_for_reuse(
            context,
            selected_arm,
            summary,
            source_dir,
            fold_rows,
            fold_aux_rows,
        )
        shutil.copytree(source_dir, destination)
        reused_summary = deepcopy(summary)
        reused_summary["scope"] = "formal"
        reused_summary["reused_from_screen"] = True
        reused_summary["reuse_source"] = str(source_dir.relative_to(ROOT))
        write_json(destination / "summary.json", reused_summary)
        fold_summaries.append(reused_summary)
        rows.extend(fold_rows)
        aux_rows.extend(fold_aux_rows)

    for fold in FORMAL_NEW_FOLDS:
        summary, fold_rows, fold_aux_rows = train_fold(
            context, selected_arm, fold, formal_root, "formal"
        )
        summary["reused_from_screen"] = False
        write_json(formal_root / f"fold_{fold:02d}" / "summary.json", summary)
        fold_summaries.append(summary)
        rows.extend(fold_rows)
        aux_rows.extend(fold_aux_rows)

    fold_summaries.sort(key=lambda summary: int(summary["fold"]))
    aggregate = aggregate_fold_outputs(
        selected_arm, fold_summaries, rows, aux_rows, 598
    )
    require(sum(sum(row) for row in aggregate["metrics"]["confusion_matrix"]) == 598, "Confusion total changed")
    comparison = paired_comparison(rows, context["c1_by_subject"])
    require(
        comparison["repairs"] - comparison["damages"]
        == aggregate["metrics"]["correct"] - C1["correct"],
        "Paired comparison net mismatch",
    )
    evidence = modality_evidence(
        aggregate["auxiliary_auc"], aggregate["boundary_head_gradient_nonzero"]
    )
    aggregate["comparison_vs_c1"] = comparison
    aggregate["metric_delta_vs_c1"] = {
        "correct": int(aggregate["metrics"]["correct"] - C1["correct"]),
        **{
            key: float(aggregate["metrics"][key] - C1[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }
    aggregate["modality_boundary_evidence"] = evidence
    aggregate["reused_screen_folds"] = list(SCREEN_FOLDS)
    aggregate["newly_trained_formal_folds"] = list(FORMAL_NEW_FOLDS)
    aggregate["inference_auxiliary_heads_disabled"] = True
    aggregate["checkpoint_readback"] = checkpoint_readback(
        context, selected_arm, 0, formal_root / "fold_00"
    )
    decision, recommendation = formal_decision(aggregate)
    aggregate["decision"] = decision
    aggregate["next_recommendation"] = recommendation

    rows.sort(key=lambda row: int(row["subject_index"]))
    write_csv(formal_root / "oof_predictions.csv", rows)
    write_csv(formal_root / "auxiliary_predictions.csv", aux_rows)
    write_csv(
        formal_root / "fold_metrics.csv",
        [
            {
                "fold": item["fold"],
                "best_epoch": item["best_epoch"],
                "correct": item["best_metrics"]["correct"],
                "acc": item["best_metrics"]["acc"],
                "macro_f1": item["best_metrics"]["macro_f1"],
                "bacc": item["best_metrics"]["bacc"],
                "macro_auc": item["best_metrics"]["macro_auc"],
                "elapsed_seconds": item["elapsed_seconds"],
                "reused_from_screen": bool(item.get("reused_from_screen", False)),
            }
            for item in fold_summaries
        ],
    )
    write_csv(
        formal_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(aggregate["metrics"]["confusion_matrix"])
        ],
    )
    write_json(formal_root / "metrics.json", aggregate["metrics"])
    write_json(formal_root / "comparison_vs_c1.json", comparison)
    write_json(formal_root / "report.json", aggregate)
    persisted_metrics = metrics_from_rows(read_csv(formal_root / "oof_predictions.csv"))
    require(persisted_metrics == aggregate["metrics"], "Formal persisted OOF metrics changed")
    return aggregate


def render_report(payload: dict) -> str:
    screen = payload["screen"]
    lines = [
        "# Boundary-Conditioned Disentangled Gradient Learning v1",
        "",
        f"Decision: **{payload['decision']}**",
        f"Selected arm: **{payload.get('selected_arm') or 'None'}**",
        f"Device: {payload['device']['gpu']}",
        f"Total runtime: {payload['total_runtime_seconds']:.3f} s",
        "",
        "## Screen",
        "",
        "| Arm | rho | Parameters | Correct/179 | ACC | Macro-F1 | BACC | AUC | AD-sMCI | CN-sMCI | Time (s) | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARM_ORDER:
        report = screen["arms"][arm]
        metrics = report["metrics"]
        lines.append(
            f"| {report['arm']} | {report['rho']:.2f} | {report['training_parameters']} | "
            f"{metrics['correct']}/179 | {metrics['acc']:.7f} | {metrics['macro_f1']:.7f} | "
            f"{metrics['bacc']:.7f} | {metrics['macro_auc']:.7f} | "
            f"{report['ad_smci_errors']} | {report['cn_smci_errors']} | "
            f"{report['training_seconds']:.2f} | {report['passes_screen']} |"
        )

    for arm in ARM_ORDER:
        report = screen["arms"][arm]
        lines.extend(
            [
                "",
                f"### {report['arm']} folds",
                "",
                "| Fold | Best epoch | ACC |",
                "|---:|---:|---:|",
                *[
                    f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.7f} |"
                    for row in report["fold_results"]
                ],
                "",
                "| Modality | AD-sMCI AUC | CN-sMCI AUC | Head gradients nonzero |",
                "|---|---:|---:|---|",
                *[
                    f"| {modality} | {report['auxiliary_auc'][modality]['AD_SMCI']:.7f} | "
                    f"{report['auxiliary_auc'][modality]['CN_SMCI']:.7f} | "
                    f"{bool(report['boundary_head_gradient_nonzero'][modality]['AD_SMCI'] and report['boundary_head_gradient_nonzero'][modality]['CN_SMCI'])} |"
                    for modality in MODALITY_NAMES
                ],
                "",
                f"COG-minus-non-COG AUC gap: {report['modality_boundary_evidence']['COG_minus_non_COG_AUC_gap']:.7f}; "
                f"effective non-COG modalities: {report['modality_boundary_evidence']['effective_non_COG_modalities']}",
            ]
        )

    lines.extend(
        [
            "",
            f"Registered screen ranking: {' > '.join(screen['ranking'])}.",
        ]
    )

    formal = payload.get("formal")
    if formal is None:
        lines.extend(
            [
                "",
                "## Formal",
                "",
                "Formal ten-fold training was not run because neither preregistered arm passed the hard screen gate.",
                "",
                f"Checkpoint readback: `{payload['checkpoint_readback']}`",
                "",
                f"Mechanism conclusion: {payload['mechanism_conclusion']}",
                f"Next recommendation: {payload['next_recommendation']}",
            ]
        )
        return "\n".join(lines) + "\n"

    metrics = formal["metrics"]
    comparison = formal["comparison_vs_c1"]
    lines.extend(
        [
            "",
            "## Formal ten-fold result",
            "",
            f"- Selected arm / rho: {formal['arm']} / {formal['rho']}",
            f"- Training / inference parameters: {formal['training_parameters']} / {formal['inference_parameters']}",
            f"- Correct: {metrics['correct']}/598",
            f"- ACC: {metrics['acc']:.10f}",
            f"- Macro-F1: {metrics['macro_f1']:.10f}",
            f"- BACC: {metrics['bacc']:.10f}",
            f"- Probability Macro-AUC: {metrics['macro_auc']:.10f}",
            f"- Weighted-F1: {metrics['weighted_f1']:.10f}",
            f"- Confusion matrix: `{metrics['confusion_matrix']}`",
            f"- Formal ten-fold training time: {formal['training_seconds']:.3f} s",
            f"- Reused screen folds: {formal['reused_screen_folds']}",
            "",
            "| Fold | Best epoch | ACC | Reused |",
            "|---:|---:|---:|---|",
            *[
                f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.7f} | {row['reused_from_screen']} |"
                for row in formal["fold_results"]
            ],
            "",
            "## Compared with C1",
            "",
            f"- Repairs / damages / changed: {comparison['repairs']} / {comparison['damages']} / {comparison['changed_predictions']}",
            f"- AD-sMCI errors: {formal['ad_smci_errors']}",
            f"- CN-sMCI errors: {formal['cn_smci_errors']}",
            f"- AD-CN errors: {formal['ad_cn_errors']}",
            f"- No new AD-CN errors: {formal['ad_cn_errors'] <= C1['ad_cn_errors']}",
            f"- Metric deltas: `{formal['metric_delta_vs_c1']}`",
            "",
            "## Modality boundary evidence",
            "",
            "| Modality | AD-sMCI AUC | CN-sMCI AUC | Gradient nonzero | Effective |",
            "|---|---:|---:|---|---|",
        ]
    )
    for modality in MODALITY_NAMES:
        evidence = formal["modality_boundary_evidence"]["by_modality"][modality]
        lines.append(
            f"| {modality} | {evidence['AD_SMCI_AUC']:.7f} | {evidence['CN_SMCI_AUC']:.7f} | "
            f"{evidence['boundary_head_gradient_nonzero']} | {evidence['effective_above_random']} |"
        )
    mechanism = formal["modality_boundary_evidence"]
    lines.extend(
        [
            "",
            f"- COG minus non-COG AUC gap: {mechanism['COG_minus_non_COG_AUC_gap']:.7f}",
            f"- Effective non-COG modalities: {mechanism['effective_non_COG_modalities']}",
            f"- Strict/mild routing winner: {payload['selected_arm']}",
            "- Auxiliary heads are disabled for classification inference.",
            "",
            f"Checkpoint readback: `{formal['checkpoint_readback']}`",
            "",
            f"Mechanism conclusion: {payload['mechanism_conclusion']}",
            f"Next recommendation: {payload['next_recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run smoke, both preregistered screens, and conditional formal folds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.run_all, "Use --run-all; partial experiment modes are disabled")
    output_root = ROOT / OUTPUT_REL
    require(not output_root.exists(), f"Refusing to overwrite output: {output_root}")
    output_root.mkdir(parents=True)
    configs_root = output_root / "configs"
    configs_root.mkdir()
    total_started = time.perf_counter()
    context = load_context()
    base_config = {
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "run_command": f"{sys.executable} scripts/run_bc_dgl_v1.py --run-all",
        "C1_reference": {
            "git_ref": C1_GIT_REF,
            "oof_path": C1_OOF_REL,
            "report_path": C1_REPORT_REL,
            "rerun": False,
        },
        "screen_folds": list(SCREEN_FOLDS),
        "formal_new_folds": list(FORMAL_NEW_FOLDS),
        "non_COG_effectiveness_rule": "both ADS/CNS AUC >0.50 and mean >0.55",
    }
    write_json(configs_root / "experiment.json", base_config)
    for arm in ARM_ORDER:
        write_json(configs_root / f"{arm}.json", fold_config(context, arm, -1, "screen"))

    smoke = run_smoke(context, output_root)
    screen_summary, screen_reports = run_screen(context, output_root)
    selected_arm = screen_summary["selected_arm_key"]
    formal = None
    if selected_arm is not None:
        write_json(
            configs_root / "selected_formal.json",
            fold_config(context, selected_arm, -1, "formal"),
        )
        formal = run_formal(
            context,
            output_root,
            selected_arm,
            screen_reports[selected_arm],
        )
        decision = formal["decision"]
        recommendation = formal["next_recommendation"]
        checkpoint = formal["checkpoint_readback"]
        evidence = formal["modality_boundary_evidence"]
        mechanism_conclusion = (
            f"{ARMS[selected_arm]['name']} won the preregistered screen; "
            f"{evidence['effective_non_COG_count']} non-COG modalities met the fixed auxiliary-evidence rule, "
            f"with COG-minus-non-COG mean AUC gap {evidence['COG_minus_non_COG_AUC_gap']:.7f}."
        )
    else:
        ranked_key = "s" if screen_summary["ranking"][0] == ARMS["s"]["name"] else "m"
        checkpoint = checkpoint_readback(
            context,
            ranked_key,
            SCREEN_FOLDS[0],
            screen_reports[ranked_key]["root"] / f"fold_{SCREEN_FOLDS[0]:02d}",
        )
        decision = "BC_DGL_SCREEN_STOP"
        recommendation = "Neither preregistered arm passed; do not tune lambda or run formal ten-fold training."
        mechanism_conclusion = (
            f"{ARMS[ranked_key]['name']} ranked ahead of the other preregistered routing arm, "
            "but neither preserved both hard-fold accuracy and boundary errors."
        )
        formal_status_root = output_root / "formal"
        formal_status_root.mkdir()
        write_json(
            formal_status_root / "status.json",
            {
                "status": "NOT_RUN_SCREEN_STOP",
                "reason": "Neither preregistered arm passed the fixed hard-fold screen gate.",
                "selected_arm": None,
                "trained_folds": [],
            },
        )

    screen_training_seconds = float(
        sum(screen_reports[arm]["report"]["training_seconds"] for arm in ARM_ORDER)
    )
    new_formal_seconds = 0.0
    if formal is not None:
        new_formal_seconds = float(
            sum(
                row["elapsed_seconds"]
                for row in formal["fold_results"]
                if not row["reused_from_screen"]
            )
        )
    payload = {
        "experiment": "Boundary-Conditioned Disentangled Gradient Learning v1",
        "decision": decision,
        "selected_arm": None if selected_arm is None else ARMS[selected_arm]["name"],
        "selected_arm_key": selected_arm,
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "device": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sklearn": sklearn.__version__,
        },
        "training_parameters": EXPECTED_TRAINING_PARAMETERS,
        "inference_parameters": EXPECTED_INFERENCE_PARAMETERS,
        "smoke": smoke,
        "screen": screen_summary,
        "formal": formal,
        "checkpoint_readback": checkpoint,
        "screen_training_seconds": screen_training_seconds,
        "new_formal_training_seconds": new_formal_seconds,
        "total_executed_training_seconds": screen_training_seconds + new_formal_seconds,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "mechanism_conclusion": mechanism_conclusion,
        "next_recommendation": recommendation,
        "integrity": {
            "C1_not_rerun": True,
            "screen_folds": list(SCREEN_FOLDS),
            "selected_screen_folds_reused": formal is not None,
            "auxiliary_heads_disabled_at_inference": True,
            "checkpoint_readback_once": True,
        },
    }
    write_json(output_root / "final_report.json", payload)
    (output_root / "final_report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(
        f"FINAL decision={decision} selected={payload['selected_arm']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
