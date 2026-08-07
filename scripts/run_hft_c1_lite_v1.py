"""Staged HFT-C1-Lite v1 experiment runner.

Stages are intentionally separate so implementation, screen, and conditional
formal results can be committed independently:

    --smoke   zero-equivalence and fold-4 three-epoch CUDA smoke
    --screen  fixed folds 4/7/8, including at most one preregistered A/B retry
    --formal  conditional ten-fold run using the screen-locked configuration
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
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
from Model.hft_c1_lite import HFTC1LiteModel
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/hft_c1_lite_v1")
C1_GIT_REF = "refs/remotes/origin/experiment/cme-dual-branch-v1"
C1_OOF_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
C1_REPORT_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/report.json"
C1_BLOB_SHA256 = "6f4ef505641f75236b01ae128f2836bf2f126ced43158f9c4ee21cc090898018"
BASE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
BRANCH_PREFIX = "experiment/hft-c1-lite-v1"
FOLDS = tuple(range(10))
SCREEN_FOLDS = (4, 7, 8)
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 3
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
HFT_MODALITIES = ("MRI", "PET", "CSF", "Risk", "ROI")
C1_PARAMETERS = 862_971
HFT_PARAMETERS = 79_776
TOTAL_PARAMETERS = 942_747
EXPECTED_FEATURE_COUNTS = {"MRI": 138, "PET": 150, "CSF": 3, "Risk": 36, "COG": 24, "ROI": 9}
DEFAULT_CAP = 0.15
DEFAULT_TEMPERATURE = 1.0
ADJUSTED_CAP = 0.25
ADJUSTED_TEMPERATURE = 0.5
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
    "subjects": 179,
    "correct": 166,
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


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="strict",
    ).strip()


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


def require_committed_stage_files(paths: list[Path], stage_name: str) -> None:
    relative_paths = [str(path.resolve().relative_to(ROOT.resolve())) for path in paths]
    for relative_path in relative_paths:
        tracked = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        require(tracked, f"{stage_name} prerequisite is not committed: {relative_path}")
    clean = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *relative_paths,
        ],
        check=False,
    ).returncode == 0
    require(clean, f"{stage_name} prerequisite files differ from committed HEAD")


def validate_git_context() -> dict:
    branch = git_value("branch", "--show-current")
    require(
        branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-r"),
        f"Unexpected experiment branch: {branch}",
    )
    subprocess.check_call(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "cat-file",
            "-e",
            f"{BASE_COMMIT}^{{commit}}",
        ]
    )
    ancestor = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            BASE_COMMIT,
            "HEAD",
        ],
        check=False,
    ).returncode == 0
    require(ancestor, "The exact C1 base commit is not an ancestor of HEAD")
    return {"branch": branch, "head": git_value("rev-parse", "HEAD"), "base_commit": BASE_COMMIT}


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
    require(len(rows) == 598 and len({int(row["subject_index"]) for row in rows}) == 598, "C1 OOF integrity changed")
    metrics = metrics_from_rows(rows)
    for key in ("correct", "confusion_matrix"):
        require(metrics[key] == C1[key], f"C1 anchor changed: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    screen_rows = [row for row in rows if int(row["fold"]) in SCREEN_FOLDS]
    screen_metrics = metrics_from_rows(screen_rows)
    require(screen_metrics["correct"] == C1_SCREEN["correct"], "C1 screen correct anchor changed")
    require(screen_metrics["confusion_matrix"] == C1_SCREEN["confusion_matrix"], "C1 screen confusion anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(screen_metrics[key] - C1_SCREEN[key]) <= 5e-7, f"C1 screen anchor changed: {key}")
    report = json.loads(git_bytes(C1_GIT_REF, C1_REPORT_REL).decode("utf-8"))
    require(int(report["parameter_count"]) == C1_PARAMETERS, "C1 parameter anchor changed")
    return rows, metrics, report


def load_feature_names(feature_path: str | Path) -> list[str]:
    with Path(feature_path).open("r", newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    require(len(header) == 361, "Feature CSV header changed")
    return header[:-1]


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    git_context = validate_git_context()
    device = torch.device("cuda:0")
    config_root = Path(tempfile.gettempdir()) / "work22_hft_c1_lite_config"
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
    modal_alias = {
        "MRI": "MRI",
        "PET": "PET",
        "CSF": "CSF",
        "RISK_FACTOR": "Risk",
        "RISK": "Risk",
        "PHS": "Risk",
        "COGNITIVE_TEST": "COG",
        "COG": "COG",
        "ROI_AVERAGE": "ROI",
        "ROI": "ROI",
    }
    actual_modalities = tuple(
        modal_alias.get(str(value).upper(), str(value))
        for value in dataset_dict["Modal_Name"]
    )
    require(actual_modalities == MODALITY_NAMES, f"Modality order changed: {actual_modalities}")
    feature_names = load_feature_names(feature_path)
    feature_names_by_modality = {
        modality: [feature_names[index] for index in dataset_dict["Modal_Index"][modal_index]]
        for modal_index, modality in enumerate(MODALITY_NAMES)
    }
    actual_feature_counts = {
        modality: len(feature_names_by_modality[modality])
        for modality in MODALITY_NAMES
    }
    require(
        actual_feature_counts == EXPECTED_FEATURE_COUNTS,
        f"Modality feature counts changed: {actual_feature_counts}",
    )
    c1_rows, c1_metrics, c1_report = parse_c1_rows()
    return {
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "class_index": {name: normalized_classes.index(name) for name in CLASS_NAMES},
        "feature_names_by_modality": feature_names_by_modality,
        "c1_rows": c1_rows,
        "c1_metrics": c1_metrics,
        "c1_report": c1_report,
        "c1_by_subject": {int(row["subject_index"]): row for row in c1_rows},
        "git": git_context,
    }


def build_model(
    context: dict,
    cap: float,
    attention_temperature: float,
    hft_enabled: bool = True,
) -> HFTC1LiteModel:
    config = context["config"]
    SET_Random(SEED)
    model = HFTC1LiteModel(
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
        cap=float(cap),
        attention_temperature=float(attention_temperature),
        hft_enabled=hft_enabled,
    ).to(context["device"])
    if hft_enabled:
        require(set(model.hft_branches.keys()) == set(HFT_MODALITIES), "HFT branch set changed")
        require("COG" not in model.hft_branches, "COG unexpectedly has an HFT branch")
        require(model.hft_parameter_count() == HFT_PARAMETERS, "HFT parameter count changed")
        require(parameter_count(model) == TOTAL_PARAMETERS, "HFT total parameter count changed")
        require(model.inference_parameter_count() == TOTAL_PARAMETERS, "HFT inference parameter count changed")
        require(model.c1_parameter_count() == C1_PARAMETERS, "HFT C1 backbone parameter count changed")
    else:
        require(parameter_count(model) == C1_PARAMETERS, "C1 reference parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def make_fresh_training_objects(
    context: dict,
    cap: float,
    attention_temperature: float,
):
    model = build_model(context, cap, attention_temperature, True)
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
    require(len(optimizer.state) == 0 and int(scheduler.T_max) == EPOCHS, "Fresh optimizer/scheduler changed")
    return model, criterion, optimizer, scheduler


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


def fold_config(
    context: dict,
    fold: int,
    scope: str,
    cap: float,
    attention_temperature: float,
) -> dict:
    return {
        "experiment": "Hierarchical Fine-Grained Modality Token Residual v1",
        "scope": scope,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "subjects": 598,
        "features": 360,
        "class_order": list(CLASS_NAMES),
        "modality_order": list(MODALITY_NAMES),
        "HFT_modalities": list(HFT_MODALITIES),
        "COG_branch_added": False,
        "cap": float(cap),
        "eps": 1e-8,
        "attention_temperature": float(attention_temperature),
        "single_head_attention": True,
        "query_detached": True,
        "output_projection_zero_initialized": True,
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
        "loss": "complete historical C1 weighted CE plus three OVR auxiliary losses; no HFT auxiliary loss",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
    }


def mechanism_diagnostics(
    intermediates: dict,
    mask: torch.Tensor,
    context: dict,
) -> dict:
    output = {}
    for modality in HFT_MODALITIES:
        shared = intermediates["hft_shared_by_modality"][modality][mask]
        delta = intermediates["hft_delta_by_modality"][modality][mask]
        ratio = intermediates["hft_ratio_by_modality"][modality][mask]
        saturated = intermediates["hft_cap_saturated_by_modality"][modality][mask]
        attention = intermediates["hft_attention_weights_by_modality"][modality][mask]
        require(attention.ndim == 2, f"{modality} attention shape changed")
        feature_names = context["feature_names_by_modality"][modality]
        require(attention.size(1) == len(feature_names), f"{modality} feature-token count changed")
        probability = attention.clamp_min(1e-12)
        denominator = math.log(max(2, attention.size(1)))
        entropy = -(probability * probability.log()).sum(dim=1) / denominator
        require(bool(torch.isfinite(ratio).all() and torch.isfinite(entropy).all()), f"{modality} diagnostic non-finite")
        require(float(ratio.max().detach().cpu()) <= float(intermediates["hft_cap"]) + 1e-6, f"{modality} cap violated")
        output[modality] = {
            "count": int(mask.sum()),
            "ratio_sum": float(ratio.sum().detach().cpu()),
            "ratio_max": float(ratio.max().detach().cpu()),
            "cap_saturation_count": int(saturated.sum().detach().cpu()),
            "entropy_sum": float(entropy.sum().detach().cpu()),
            "attention_sum": attention.sum(dim=0).detach().cpu().tolist(),
            "shared_norm_mean": float(shared.norm(dim=-1).mean().detach().cpu()),
            "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu()),
        }
    return output


def summarize_mechanism(fold_diagnostics: list[dict], context: dict) -> dict:
    result = {}
    for modality in HFT_MODALITIES:
        count = sum(item[modality]["count"] for item in fold_diagnostics)
        ratio_sum = sum(item[modality]["ratio_sum"] for item in fold_diagnostics)
        saturation = sum(item[modality]["cap_saturation_count"] for item in fold_diagnostics)
        entropy_sum = sum(item[modality]["entropy_sum"] for item in fold_diagnostics)
        attention_sum = np.sum(
            [np.asarray(item[modality]["attention_sum"], dtype=np.float64) for item in fold_diagnostics],
            axis=0,
        )
        mean_attention = attention_sum / max(1, count)
        feature_names = context["feature_names_by_modality"][modality]
        top_indices = np.argsort(-mean_attention)[: min(5, len(feature_names))]
        result[modality] = {
            "mean_residual_shared_ratio": float(ratio_sum / max(1, count)),
            "maximum_ratio": float(max(item[modality]["ratio_max"] for item in fold_diagnostics)),
            "cap_saturation_fraction": float(saturation / max(1, count)),
            "normalized_attention_entropy": float(entropy_sum / max(1, count)),
            "top5_features": [
                {
                    "feature_name": feature_names[int(index)],
                    "local_feature_index": int(index),
                    "mean_attention_weight": float(mean_attention[int(index)]),
                }
                for index in top_indices
            ],
        }
    means = [result[modality]["mean_residual_shared_ratio"] for modality in HFT_MODALITIES]
    saturations = [result[modality]["cap_saturation_fraction"] for modality in HFT_MODALITIES]
    entropies = [result[modality]["normalized_attention_entropy"] for modality in HFT_MODALITIES]
    result["overall"] = {
        "five_modality_mean_ratio": float(np.mean(means)),
        "five_modality_mean_saturation": float(np.mean(saturations)),
        "five_modality_mean_entropy": float(np.mean(entropies)),
        "modalities_ratio_at_least_0_05": int(sum(value >= 0.05 for value in means)),
        "all_branches_near_zero": bool(all(value < 0.01 for value in means)),
    }
    return result


def common_c1_state(model: HFTC1LiteModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("hft_branches.")
    }


def upstream_branch_gradient_norm(branch: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in branch.named_parameters()
        if not name.startswith("output_projection.") and parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu())


def experiment_config(context: dict) -> dict:
    return {
        "experiment": "Hierarchical Fine-Grained Modality Token Residual v1",
        "short_name": "HFT-C1-Lite",
        "base_commit": BASE_COMMIT,
        "c1_git_ref": C1_GIT_REF,
        "c1_oof_path": C1_OOF_REL,
        "c1_oof_blob_sha256": C1_BLOB_SHA256,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(FOLDS),
        "screen_folds": list(SCREEN_FOLDS),
        "formal_folds_are_fresh": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "smoke_epochs": SMOKE_EPOCHS,
        "device": "cuda:0",
        "default_cap": DEFAULT_CAP,
        "default_attention_temperature": DEFAULT_TEMPERATURE,
        "adjustment_A": {"cap": ADJUSTED_CAP, "attention_temperature": DEFAULT_TEMPERATURE},
        "adjustment_B": {"cap": DEFAULT_CAP, "attention_temperature": ADJUSTED_TEMPERATURE},
        "feature_counts": EXPECTED_FEATURE_COUNTS,
        "c1_parameters": C1_PARAMETERS,
        "added_parameters": HFT_PARAMETERS,
        "training_parameters": TOTAL_PARAMETERS,
        "inference_parameters": TOTAL_PARAMETERS,
        "screen_C1_anchor": C1_SCREEN,
        "screen_rules": {
            "SCREEN_GO": {
                "correct_minimum": 167,
                "new_AD_CN_errors": 0,
                "adjacent_boundary_errors_maximum": 13,
                "macro_f1_drop_maximum": 0.005,
                "bacc_drop_maximum": 0.005,
            },
            "SCREEN_NEAR": {
                "correct_exact": 166,
                "new_AD_CN_errors": 0,
                "adjacent_boundary_errors_maximum": 13,
                "requires_bacc_or_auc_improvement": True,
                "modalities_ratio_at_least_0_05_minimum": 2,
            },
            "SCREEN_STOP_vetoes": {
                "correct_maximum": 165,
                "AD_SMCI_errors_minimum": C1_SCREEN["ad_smci_errors"] + 2,
                "CN_SMCI_errors_minimum": C1_SCREEN["cn_smci_errors"] + 2,
                "new_AD_CN_errors": True,
                "clearly_fewer_repairs_margin": 2,
                "all_branch_ratio_below": 0.01,
                "widespread_saturation_fraction": 0.50,
            },
            "single_adjustment_after_SCREEN_NEAR_only": {
                "maximum_adjustments": 1,
                "A_priority": True,
                "A_condition": "five-modality mean ratio <0.05 and mean cap saturation <0.05",
                "A_change_only": "cap 0.15 -> 0.25",
                "B_condition": "mean normalized entropy >0.95, mean ratio >=0.01, correct >=166",
                "B_change_only": "attention temperature 1.0 -> 0.5",
                "formal_requires_adjusted_SCREEN_GO_and_correct_minimum": 167,
            },
        },
        "formal_rules": {
            "HFT_GO": {
                "correct_minimum": 561,
                "macro_f1_minimum": C1["macro_f1"] - 0.003,
                "bacc_minimum": C1["bacc"] - 0.003,
                "AD_CN_errors": 0,
            },
            "HFT_NEUTRAL": {
                "correct_exact": 560,
                "requires_auc_or_bacc_improvement": True,
            },
        },
        "optimizer": "Adam",
        "scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "loss": "complete historical C1 loss only",
    }


def write_environment(output_root: Path, context: dict) -> None:
    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "sklearn": sklearn.__version__,
            "branch": context["git"]["branch"],
            "run_commit": context["git"]["head"],
            "base_commit": BASE_COMMIT,
        },
    )


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke output: {smoke_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True)
    write_json(output_root / "config.json", experiment_config(context))
    write_environment(output_root, context)
    shutil.copyfile(ROOT / CONFIG_REL, output_root / "historical_c1_config.ini")

    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][4]

    c1_model = build_model(context, DEFAULT_CAP, DEFAULT_TEMPERATURE, False)
    hft_model = build_model(context, DEFAULT_CAP, DEFAULT_TEMPERATURE, True)
    c1_state = common_c1_state(c1_model)
    hft_state = common_c1_state(hft_model)
    require(c1_state.keys() == hft_state.keys(), "C1/HFT common state keys differ")
    require(
        all(torch.equal(c1_state[key], hft_state[key]) for key in c1_state),
        "Adding HFT perturbed C1 initialization",
    )
    require(
        all(
            torch.count_nonzero(branch.output_projection.weight) == 0
            and torch.count_nonzero(branch.output_projection.bias) == 0
            for branch in hft_model.hft_branches.values()
        ),
        "An HFT output projection is not zero initialized",
    )
    c1_model.eval()
    hft_model.eval()
    with torch.no_grad():
        c1_raw, _, _, c1_inter = c1_model(features, return_intermediates=True)
        hft_raw, _, _, hft_inter = hft_model(features, return_intermediates=True)
    logit_diff = float((c1_raw - hft_raw).abs().max().cpu())
    global_diff = float((c1_inter["G"] - hft_inter["G"]).abs().max().cpu())
    maximum_initial_delta = max(
        float(value.abs().max().cpu())
        for value in hft_inter["hft_delta_by_modality"].values()
    )
    require(logit_diff <= 1e-6, f"Epoch-0 C1 equivalence failed: {logit_diff}")
    require(global_diff <= 1e-7, f"Global path changed: {global_diff}")
    require(maximum_initial_delta == 0.0, "Epoch-0 HFT residual is not exactly zero")
    mechanism_diagnostics(hft_inter, test_mask, context)
    del c1_model, hft_model, c1_raw, hft_raw, c1_inter, hft_inter
    torch.cuda.empty_cache()

    model, criterion, optimizer, scheduler = make_fresh_training_objects(
        context, DEFAULT_CAP, DEFAULT_TEMPERATURE
    )
    output_projection_ever = {modality: False for modality in HFT_MODALITIES}
    upstream_ever = {modality: False for modality in HFT_MODALITIES}
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features, return_intermediates=True
        )
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Smoke epoch {epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"Smoke epoch {epoch}: non-finite gradient")
        projection_gradients = {}
        upstream_gradients = {}
        for modality, branch in model.hft_branches.items():
            projection_gradients[modality] = module_gradient_norm(branch.output_projection)
            upstream_gradients[modality] = upstream_branch_gradient_norm(branch)
            output_projection_ever[modality] |= projection_gradients[modality] > 0.0
            upstream_ever[modality] |= upstream_gradients[modality] > 0.0
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            require(bool(torch.isfinite(raw).all()), "Smoke produced non-finite logits")
            diagnostics = mechanism_diagnostics(intermediates, test_mask, context)
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{f"output_projection_gradient_{m}": projection_gradients[m] for m in HFT_MODALITIES},
                **{f"upstream_gradient_{m}": upstream_gradients[m] for m in HFT_MODALITIES},
                **{f"maximum_ratio_{m}": diagnostics[m]["ratio_max"] for m in HFT_MODALITIES},
            }
        )
        del intermediates
    require(all(output_projection_ever.values()), "An HFT output projection never received gradient")
    require(any(upstream_ever.values()), "No feature-token branch received upstream gradient")
    require(gradients_finite(model), "Smoke ended with non-finite gradient")

    model.eval()
    with torch.no_grad():
        before_raw, _, _, before_inter = model(features, return_intermediates=True)
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save({"model_state": clone_cpu_state(model)}, checkpoint_path)
    reloaded = build_model(context, DEFAULT_CAP, DEFAULT_TEMPERATURE, True)
    checkpoint = torch.load(checkpoint_path, map_location=context["device"], weights_only=True)
    reloaded.load_state_dict(checkpoint["model_state"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        after_raw, _, _, after_inter = reloaded(features, return_intermediates=True)
    reload_diff = float((before_raw - after_raw).abs().max().cpu())
    require(reload_diff <= 1e-6, f"Checkpoint roundtrip changed logits: {reload_diff}")
    final_diagnostics = mechanism_diagnostics(after_inter, test_mask, context)
    elapsed = time.perf_counter() - started
    payload = {
        "passed": True,
        "fold": 4,
        "epochs": SMOKE_EPOCHS,
        "device": torch.cuda.get_device_name(0),
        "epoch0_max_abs_logit_diff_vs_C1": logit_diff,
        "epoch0_global_max_abs_diff_vs_C1": global_diff,
        "epoch0_max_abs_delta": maximum_initial_delta,
        "parameter_count": TOTAL_PARAMETERS,
        "added_parameters": HFT_PARAMETERS,
        "inference_parameters": model.inference_parameter_count(),
        "COG_has_HFT_branch": False,
        "output_projection_gradient_nonzero": output_projection_ever,
        "feature_token_upstream_gradient_nonzero": upstream_ever,
        "at_least_one_feature_token_branch_gradient": any(upstream_ever.values()),
        "finite": True,
        "cap_respected": True,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_readback_max_abs_logit_diff": reload_diff,
        "final_mechanism": summarize_mechanism([final_diagnostics], context),
        "elapsed_seconds": elapsed,
        "run_commit": context["git"]["head"],
    }
    write_csv(smoke_root / "epoch_metrics.csv", epoch_rows)
    write_json(smoke_root / "summary.json", payload)
    write_json(output_root / "summary.json", {"decision": "SMOKE_PASS", "smoke": payload})
    (output_root / "summary.md").write_text(
        "# HFT-C1-Lite v1\n\nDecision: `SMOKE_PASS`\n\n"
        f"Epoch-0 maximum logit difference: `{logit_diff:.3e}`. "
        f"Checkpoint readback difference: `{reload_diff:.3e}`.\n",
        encoding="utf-8",
    )
    print("SMOKE PASS", flush=True)
    del model, criterion, optimizer, scheduler, reloaded, before_raw, after_raw, before_inter, after_inter
    torch.cuda.empty_cache()
    return payload


def train_fold(
    context: dict,
    fold: int,
    scope_root: Path,
    scope: str,
    cap: float,
    attention_temperature: float,
) -> tuple[dict, list[dict], dict]:
    final_dir = scope_root / f"fold_{fold:02d}"
    staging_dir = scope_root / f".fold_{fold:02d}_in_progress"
    require(
        not final_dir.exists() and not staging_dir.exists(),
        f"Refusing to overwrite {scope} fold {fold}",
    )
    staging_dir.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(not bool((train_mask & test_mask).any()), f"Fold {fold} train/test masks overlap")
    require(int(train_mask.sum() + test_mask.sum()) == 598, f"Fold {fold} masks incomplete")
    model, criterion, optimizer, scheduler = make_fresh_training_objects(
        context, cap, attention_temperature
    )
    config_payload = fold_config(context, fold, scope, cap, attention_temperature)
    config_payload.update(
        {
            "training_parameters": TOTAL_PARAMETERS,
            "inference_parameters": TOTAL_PARAMETERS,
            "added_HFT_parameters": HFT_PARAMETERS,
            "feature_counts": EXPECTED_FEATURE_COUNTS,
            "labels_used_only_by_historical_C1_loss_on_train_mask": True,
        }
    )
    best = None
    best_state = None
    epoch_rows = []
    output_projection_ever = {modality: False for modality in HFT_MODALITIES}
    upstream_ever = {modality: False for modality in HFT_MODALITIES}
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"{scope} fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"{scope} fold{fold} epoch{epoch}: non-finite gradient")
        for modality, branch in model.hft_branches.items():
            output_projection_ever[modality] |= module_gradient_norm(branch.output_projection) > 0.0
            upstream_ever[modality] |= upstream_branch_gradient_norm(branch) > 0.0
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _, eval_intermediates = model(
                features, return_intermediates=True
            )
            test_metrics, selection_tuple = selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or selection_tuple > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": selection_tuple,
                "metrics": deepcopy(test_metrics),
            }
            best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "correct": test_metrics["correct"],
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
                **{
                    f"cap_saturation_fraction_{modality}": float(
                        eval_intermediates["hft_cap_saturated_by_modality"][modality][test_mask]
                        .float()
                        .mean()
                        .cpu()
                    )
                    for modality in HFT_MODALITIES
                },
                **{
                    f"mean_residual_shared_ratio_{modality}": float(
                        eval_intermediates["hft_ratio_by_modality"][modality][test_mask]
                        .mean()
                        .cpu()
                    )
                    for modality in HFT_MODALITIES
                },
            }
        )
        del eval_intermediates

    require(best is not None and best_state is not None, f"{scope} fold{fold}: no best state")
    require(all(output_projection_ever.values()), f"{scope} fold{fold}: output projection gradient missing")
    require(any(upstream_ever.values()), f"{scope} fold{fold}: no upstream HFT gradient")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(
            features, return_intermediates=True
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
        diagnostics = mechanism_diagnostics(
            best_intermediates, test_mask, context
        )
    require(best_metrics == best["metrics"], f"{scope} fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"{scope} fold{fold}: saved predictions changed metrics")
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "scope": scope,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "fresh_model": True,
        "fresh_criterion": True,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "boundary_counts": boundary_counts(best_metrics),
        "training_parameters": parameter_count(model),
        "inference_parameters": model.inference_parameter_count(),
        "added_parameters": model.hft_parameter_count(),
        "output_projection_gradient_nonzero": output_projection_ever,
        "feature_token_upstream_gradient_nonzero": upstream_ever,
        "post_warmup_saturation_epoch_fraction_ge_0_50": {
            modality: float(
                np.mean(
                    [
                        row[f"cap_saturation_fraction_{modality}"] >= 0.50
                        for row in epoch_rows[20:]
                    ]
                )
            )
            for modality in HFT_MODALITIES
        },
        "elapsed_seconds": elapsed,
        "config": config_payload,
    }
    torch.save({"model_state": best_state}, staging_dir / "checkpoint_best.pt")
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_json(staging_dir / "config.json", config_payload)
    write_json(staging_dir / "summary.json", summary)
    write_json(staging_dir / "mechanism.json", diagnostics)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    write_csv(
        staging_dir / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(best_metrics["confusion_matrix"])
        ],
    )
    staging_dir.rename(final_dir)
    print(
        f"fold={fold} best_epoch={best['epoch']} correct={best_metrics['correct']} "
        f"ACC={best_metrics['acc']:.7f} Macro-F1={best_metrics['macro_f1']:.7f} "
        f"BACC={best_metrics['bacc']:.7f} Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler, best_raw, best_intermediates
    torch.cuda.empty_cache()
    return summary, rows, diagnostics


def aggregate_fold_outputs(
    context: dict,
    fold_summaries: list[dict],
    rows: list[dict],
    fold_diagnostics: list[dict],
    expected_folds: tuple[int, ...],
    expected_subjects: int,
) -> dict:
    rows.sort(key=lambda row: int(row["subject_index"]))
    require(
        {int(summary["fold"]) for summary in fold_summaries} == set(expected_folds),
        "Fold summaries do not match the preregistered folds",
    )
    require(len(rows) == expected_subjects, f"Expected {expected_subjects} OOF rows")
    require(len({int(row["subject_index"]) for row in rows}) == expected_subjects, "OOF subjects are not unique")
    require(
        {int(row["fold"]) for row in rows} == set(expected_folds),
        "OOF rows contain unexpected folds",
    )
    metrics = metrics_from_rows(rows)
    counts = boundary_counts(metrics)
    fold_acc = np.asarray(
        [summary["best_metrics"]["acc"] for summary in fold_summaries],
        dtype=np.float64,
    )
    report = {
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
                "fresh_run": True,
            }
            for summary in sorted(fold_summaries, key=lambda item: item["fold"])
        ],
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in fold_summaries)),
        "training_parameters": TOTAL_PARAMETERS,
        "inference_parameters": TOTAL_PARAMETERS,
        "added_parameters": HFT_PARAMETERS,
        "mechanism": summarize_mechanism(fold_diagnostics, context),
        "paired_vs_C1": paired_comparison(rows, context["c1_by_subject"]),
        "post_warmup_saturation_epoch_fraction_ge_0_50": {
            modality: float(
                np.mean(
                    [
                        summary["post_warmup_saturation_epoch_fraction_ge_0_50"][modality]
                        for summary in fold_summaries
                    ]
                )
            )
            for modality in HFT_MODALITIES
        },
    }
    return report


def save_aggregate(scope_root: Path, rows: list[dict], report: dict) -> None:
    write_csv(scope_root / "oof_predictions.csv", rows)
    write_csv(
        scope_root / "per_fold_metrics.csv",
        report["fold_results"],
    )
    write_json(scope_root / "metrics.json", report["metrics"])
    write_json(scope_root / "mechanism.json", report["mechanism"])
    write_json(scope_root / "report.json", report)
    write_csv(
        scope_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(report["metrics"]["confusion_matrix"])
        ],
    )


def checkpoint_readback(
    context: dict,
    fold_dir: Path,
    cap: float,
    attention_temperature: float,
) -> dict:
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(fold_dir / "best_predictions.csv")
    config = json.loads((fold_dir / "config.json").read_text(encoding="utf-8"))
    require(config["cap"] == cap and config["attention_temperature"] == attention_temperature, "Checkpoint config changed")
    require(config["epochs"] == EPOCHS and config["seed"] == SEED, "Checkpoint protocol changed")
    model = build_model(context, cap, attention_temperature, True)
    checkpoint = torch.load(
        fold_dir / "checkpoint_best.pt",
        map_location=context["device"],
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    _, test_mask = context["dataset_data"]["Mask"][int(summary["fold"])]
    with torch.no_grad():
        raw, _, _, intermediates = model(features, return_intermediates=True)
    reloaded_rows = prediction_rows(
        int(summary["fold"]),
        raw,
        labels,
        test_mask,
        context["dataset_dict"],
        context["config"],
    )
    require(len(rows) == len(reloaded_rows), "Checkpoint readback row count mismatch")
    require(
        [int(row["subject_index"]) for row in rows]
        == [int(row["subject_index"]) for row in reloaded_rows],
        "Checkpoint readback subject order mismatch",
    )
    require(
        [int(row["prediction"]) for row in rows]
        == [int(row["prediction"]) for row in reloaded_rows],
        "Checkpoint readback predictions mismatch",
    )
    require(metrics_from_rows(rows) == metrics_from_rows(reloaded_rows), "Checkpoint readback metrics mismatch")
    probability_diff = max(
        abs(float(left[f"probability_{name}"]) - float(right[f"probability_{name}"]))
        for left, right in zip(rows, reloaded_rows)
        for name in CLASS_NAMES
    )
    require(probability_diff <= 1e-6, f"Checkpoint probability readback mismatch: {probability_diff}")
    mechanism_diagnostics(intermediates, test_mask, context)
    del model, raw, intermediates
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "fold": int(summary["fold"]),
        "maximum_probability_difference": probability_diff,
    }


def screen_stop_reasons(report: dict) -> list[str]:
    metrics = report["metrics"]
    paired = report["paired_vs_C1"]
    mechanism = report["mechanism"]
    reasons = []
    if metrics["correct"] <= 165:
        reasons.append("correct_at_most_165")
    if report["ad_cn_errors"] > C1_SCREEN["ad_cn_errors"]:
        reasons.append("new_AD_CN_error")
    if report["ad_smci_errors"] >= C1_SCREEN["ad_smci_errors"] + 2:
        reasons.append("AD_SMCI_errors_increased_by_at_least_2")
    if report["cn_smci_errors"] >= C1_SCREEN["cn_smci_errors"] + 2:
        reasons.append("CN_SMCI_errors_increased_by_at_least_2")
    if paired["damages"] >= paired["repairs"] + 2:
        reasons.append("repairs_clearly_fewer_than_damages")
    if mechanism["overall"]["all_branches_near_zero"]:
        reasons.append("all_five_HFT_branches_near_zero")
    widespread_long_saturation = any(
        value >= 0.50
        for value in report["post_warmup_saturation_epoch_fraction_ge_0_50"].values()
    )
    performance_declined = bool(
        metrics["correct"] < C1_SCREEN["correct"]
        or metrics["macro_f1"] < C1_SCREEN["macro_f1"]
        or metrics["bacc"] < C1_SCREEN["bacc"]
    )
    if widespread_long_saturation and performance_declined:
        reasons.append("long_widespread_cap_saturation_with_performance_decline")
    return reasons


def classify_screen(report: dict) -> tuple[str, list[str]]:
    stop_reasons = screen_stop_reasons(report)
    if stop_reasons:
        return "SCREEN_STOP", stop_reasons
    metrics = report["metrics"]
    mechanism = report["mechanism"]["overall"]
    go_checks = {
        "correct_at_least_167": metrics["correct"] >= 167,
        "no_new_AD_CN_errors": report["ad_cn_errors"] == 0,
        "adjacent_errors_not_above_C1": report["boundary_errors"] <= C1_SCREEN["boundary_errors"],
        "macro_f1_drop_within_0_005": metrics["macro_f1"] >= C1_SCREEN["macro_f1"] - 0.005,
        "bacc_drop_within_0_005": metrics["bacc"] >= C1_SCREEN["bacc"] - 0.005,
    }
    report["SCREEN_GO_checks"] = go_checks
    if all(go_checks.values()):
        return "SCREEN_GO", []
    near_checks = {
        "correct_exactly_166": metrics["correct"] == 166,
        "no_new_AD_CN_errors": report["ad_cn_errors"] == 0,
        "adjacent_errors_not_above_C1": report["boundary_errors"] <= C1_SCREEN["boundary_errors"],
        "bacc_or_auc_improved": (
            metrics["bacc"] > C1_SCREEN["bacc"]
            or metrics["macro_auc"] > C1_SCREEN["macro_auc"]
        ),
        "at_least_two_active_modalities": mechanism["modalities_ratio_at_least_0_05"] >= 2,
    }
    report["SCREEN_NEAR_checks"] = near_checks
    if all(near_checks.values()):
        return "SCREEN_NEAR", []
    return "SCREEN_STOP", ["did_not_meet_preregistered_GO_or_NEAR_rule"]


def choose_near_adjustment(report: dict) -> tuple[str | None, dict | None, str]:
    overall = report["mechanism"]["overall"]
    if (
        overall["five_modality_mean_ratio"] < 0.05
        and overall["five_modality_mean_saturation"] < 0.05
    ):
        return (
            "A",
            {"cap": ADJUSTED_CAP, "attention_temperature": DEFAULT_TEMPERATURE},
            "mean residual/shared ratio <0.05 and mean cap saturation <5%",
        )
    if (
        overall["five_modality_mean_entropy"] > 0.95
        and overall["five_modality_mean_ratio"] >= 0.01
        and report["metrics"]["correct"] >= 166
    ):
        return (
            "B",
            {"cap": DEFAULT_CAP, "attention_temperature": ADJUSTED_TEMPERATURE},
            "normalized attention entropy >0.95 with active residual and correct>=166",
        )
    return None, None, "No preregistered single-variable adjustment condition was met"


def run_screen_variant(
    context: dict,
    variant_root: Path,
    variant_name: str,
    cap: float,
    attention_temperature: float,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    variant_root.mkdir(parents=True)
    summaries = []
    rows = []
    diagnostics = []
    for fold in SCREEN_FOLDS:
        summary, fold_rows, fold_diagnostics = train_fold(
            context,
            fold,
            variant_root,
            f"screen_{variant_name}",
            cap,
            attention_temperature,
        )
        summaries.append(summary)
        rows.extend(fold_rows)
        diagnostics.append(fold_diagnostics)
    report = aggregate_fold_outputs(
        context,
        summaries,
        rows,
        diagnostics,
        SCREEN_FOLDS,
        C1_SCREEN["subjects"],
    )
    report.update(
        {
            "variant": variant_name,
            "cap": cap,
            "attention_temperature": attention_temperature,
            "C1_screen_anchor": C1_SCREEN,
        }
    )
    decision, reasons = classify_screen(report)
    report["decision"] = decision
    report["decision_reasons"] = reasons
    report["checkpoint_readback"] = checkpoint_readback(
        context,
        variant_root / "fold_04",
        cap,
        attention_temperature,
    )
    save_aggregate(variant_root, rows, report)
    return report, summaries, rows, diagnostics


def validate_smoke_for_screen(output_root: Path) -> dict:
    smoke_path = output_root / "smoke" / "summary.json"
    require(smoke_path.is_file(), "Run --smoke and commit the implementation before --screen")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke.get("passed") is True, "Smoke did not pass")
    require(smoke.get("epochs") == SMOKE_EPOCHS and smoke.get("fold") == 4, "Smoke protocol changed")
    require(smoke.get("epoch0_max_abs_logit_diff_vs_C1", 1.0) <= 1e-6, "Smoke C1 equivalence missing")
    require_committed_stage_files(
        [
            ROOT / "Model" / "hft_c1_lite.py",
            ROOT / "scripts" / "run_hft_c1_lite_v1.py",
            smoke_path,
        ],
        "Screen",
    )
    return smoke


def run_screen(context: dict, output_root: Path) -> dict:
    validate_smoke_for_screen(output_root)
    screen_root = output_root / "screen"
    require(not screen_root.exists(), f"Refusing to overwrite screen output: {screen_root}")
    screen_root.mkdir(parents=True)
    started = time.perf_counter()
    default_report, _, _, _ = run_screen_variant(
        context,
        screen_root / "default",
        "default",
        DEFAULT_CAP,
        DEFAULT_TEMPERATURE,
    )
    final_report = default_report
    adjustment = {
        "attempted": False,
        "name": None,
        "reason": None,
        "configuration": None,
    }
    if default_report["decision"] == "SCREEN_NEAR":
        adjustment_name, adjustment_config, adjustment_reason = choose_near_adjustment(default_report)
        adjustment.update(
            {
                "name": adjustment_name,
                "reason": adjustment_reason,
                "configuration": adjustment_config,
            }
        )
        if adjustment_config is not None:
            adjustment["attempted"] = True
            adjusted_report, _, _, _ = run_screen_variant(
                context,
                screen_root / f"adjustment_{adjustment_name.lower()}",
                f"adjustment_{adjustment_name.lower()}",
                float(adjustment_config["cap"]),
                float(adjustment_config["attention_temperature"]),
            )
            final_report = adjusted_report
            adjustment["result_decision"] = adjusted_report["decision"]
            adjustment["achieved_required_correct_minimum"] = adjusted_report["metrics"]["correct"] >= 167

    selected_configuration = None
    if final_report["decision"] == "SCREEN_GO":
        selected_configuration = {
            "variant": final_report["variant"],
            "cap": final_report["cap"],
            "attention_temperature": final_report["attention_temperature"],
        }
    payload = {
        "decision": final_report["decision"],
        "default": default_report,
        "adjustment": adjustment,
        "final_screen": final_report,
        "selected_configuration": selected_configuration,
        "formal_authorized": selected_configuration is not None,
        "runtime_seconds": time.perf_counter() - started,
        "branch": context["git"]["branch"],
        "base_commit": BASE_COMMIT,
        "run_commit": context["git"]["head"],
        "worktree": str(ROOT),
        "device": torch.cuda.get_device_name(0),
        "parameters": {
            "C1": C1_PARAMETERS,
            "added_HFT": HFT_PARAMETERS,
            "training": TOTAL_PARAMETERS,
            "inference": TOTAL_PARAMETERS,
        },
        "C1_source": {
            "git_ref": C1_GIT_REF,
            "path": C1_OOF_REL,
            "sha256": C1_BLOB_SHA256,
            "C1_not_retrained": True,
        },
    }
    write_json(screen_root / "summary.json", payload)
    write_json(output_root / "summary.json", payload)
    (output_root / "summary.md").write_text(render_report(payload), encoding="utf-8")
    print(payload["decision"], flush=True)
    return payload


def validate_screen_for_formal(output_root: Path) -> tuple[dict, dict]:
    screen_path = output_root / "screen" / "summary.json"
    require(screen_path.is_file(), "Run --screen and commit its result before --formal")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    require(screen.get("decision") == "SCREEN_GO", "Formal run is forbidden without SCREEN_GO")
    require(screen.get("formal_authorized") is True, "Screen did not authorize formal run")
    selected = screen.get("selected_configuration")
    require(isinstance(selected, dict), "Selected screen configuration missing")
    allowed = {
        (DEFAULT_CAP, DEFAULT_TEMPERATURE),
        (ADJUSTED_CAP, DEFAULT_TEMPERATURE),
        (DEFAULT_CAP, ADJUSTED_TEMPERATURE),
    }
    pair = (float(selected["cap"]), float(selected["attention_temperature"]))
    require(pair in allowed, f"Unregistered selected configuration: {pair}")
    if pair != (DEFAULT_CAP, DEFAULT_TEMPERATURE):
        require(screen["adjustment"]["attempted"] is True, "Adjusted configuration lacks preregistered screen")
        require(screen["final_screen"]["metrics"]["correct"] >= 167, "Adjustment did not reach 167/179")
    require(screen["final_screen"]["decision"] == "SCREEN_GO", "Final screen report is not GO")
    require_committed_stage_files(
        [
            ROOT / "Model" / "hft_c1_lite.py",
            ROOT / "scripts" / "run_hft_c1_lite_v1.py",
            screen_path,
        ],
        "Formal",
    )
    return screen, selected


def formal_decision(report: dict) -> tuple[str, list[str]]:
    metrics = report["metrics"]
    major_damage = []
    if metrics["macro_f1"] < C1["macro_f1"] - 0.003:
        major_damage.append("Macro-F1 below C1 by more than 0.003")
    if metrics["bacc"] < C1["bacc"] - 0.003:
        major_damage.append("BACC below C1 by more than 0.003")
    if report["ad_cn_errors"] > C1["ad_cn_errors"]:
        major_damage.append("new AD-CN error")
    if metrics["correct"] < 560:
        major_damage.append("correct below 560")
    if major_damage:
        return "HFT_STOP", major_damage
    go_checks = {
        "correct_at_least_561": metrics["correct"] >= 561,
        "macro_f1_safe": metrics["macro_f1"] >= C1["macro_f1"] - 0.003,
        "bacc_safe": metrics["bacc"] >= C1["bacc"] - 0.003,
        "no_AD_CN_errors": report["ad_cn_errors"] == 0,
    }
    report["HFT_GO_checks"] = go_checks
    if all(go_checks.values()):
        return "HFT_GO", []
    if (
        metrics["correct"] == 560
        and (metrics["macro_auc"] > C1["macro_auc"] or metrics["bacc"] > C1["bacc"])
    ):
        return "HFT_NEUTRAL", ["correct tied C1 and AUC or BACC improved"]
    return "HFT_STOP", ["did not meet preregistered HFT_GO or HFT_NEUTRAL rule"]


def run_formal(context: dict, output_root: Path) -> dict:
    screen, selected = validate_screen_for_formal(output_root)
    formal_root = output_root / "formal"
    require(not formal_root.exists(), f"Refusing to overwrite formal output: {formal_root}")
    formal_root.mkdir(parents=True)
    cap = float(selected["cap"])
    attention_temperature = float(selected["attention_temperature"])
    summaries = []
    rows = []
    diagnostics = []
    started = time.perf_counter()

    # Formal is deliberately a fresh ten-fold run; screen folds are not reused.
    for fold in FOLDS:
        summary, fold_rows, fold_diagnostics = train_fold(
            context,
            fold,
            formal_root,
            "formal_fresh_10fold",
            cap,
            attention_temperature,
        )
        summaries.append(summary)
        rows.extend(fold_rows)
        diagnostics.append(fold_diagnostics)
    report = aggregate_fold_outputs(
        context,
        summaries,
        rows,
        diagnostics,
        FOLDS,
        598,
    )
    report.update(
        {
            "cap": cap,
            "attention_temperature": attention_temperature,
            "selected_screen_variant": selected["variant"],
            "formal_folds_fresh": True,
            "screen_fold_checkpoints_reused": False,
            "C1_anchor": C1,
            "metric_delta_vs_C1": {
                key: report["metrics"][key] - C1[key]
                for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
            },
        }
    )
    decision, reasons = formal_decision(report)
    report["decision"] = decision
    report["decision_reasons"] = reasons
    report["checkpoint_readback"] = checkpoint_readback(
        context,
        formal_root / "fold_00",
        cap,
        attention_temperature,
    )
    report["total_runtime_seconds"] = time.perf_counter() - started
    save_aggregate(formal_root, rows, report)
    payload = {
        "decision": decision,
        "branch": context["git"]["branch"],
        "base_commit": BASE_COMMIT,
        "run_commit": context["git"]["head"],
        "worktree": str(ROOT),
        "device": torch.cuda.get_device_name(0),
        "parameters": {
            "C1": C1_PARAMETERS,
            "added_HFT": HFT_PARAMETERS,
            "training": TOTAL_PARAMETERS,
            "inference": TOTAL_PARAMETERS,
        },
        "screen": screen,
        "formal": report,
        "C1_source": {
            "git_ref": C1_GIT_REF,
            "path": C1_OOF_REL,
            "sha256": C1_BLOB_SHA256,
            "C1_not_retrained": True,
        },
    }
    write_json(formal_root / "summary.json", payload)
    write_json(output_root / "summary.json", payload)
    (output_root / "summary.md").write_text(render_report(payload), encoding="utf-8")
    print(decision, flush=True)
    return payload


def metric_line(metrics: dict, denominator: int) -> str:
    return (
        f"Correct `{metrics['correct']}/{denominator}`; ACC `{metrics['acc']:.7f}`; "
        f"Macro-F1 `{metrics['macro_f1']:.7f}`; BACC `{metrics['bacc']:.7f}`; "
        f"Probability Macro-AUC `{metrics['macro_auc']:.7f}`; "
        f"Weighted-F1 `{metrics['weighted_f1']:.7f}`."
    )


def mechanism_markdown(mechanism: dict) -> list[str]:
    lines = [
        "| Modality | Mean ratio | Maximum ratio | Cap saturation | Attention entropy | Top features |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for modality in HFT_MODALITIES:
        item = mechanism[modality]
        features = ", ".join(entry["feature_name"] for entry in item["top5_features"])
        lines.append(
            f"| {modality} | {item['mean_residual_shared_ratio']:.6f} | "
            f"{item['maximum_ratio']:.6f} | {item['cap_saturation_fraction']:.6f} | "
            f"{item['normalized_attention_entropy']:.6f} | {features} |"
        )
    return lines


def fold_markdown(report: dict) -> list[str]:
    lines = ["| Fold | Best epoch | Correct | ACC |", "|---:|---:|---:|---:|"]
    for item in report["fold_results"]:
        lines.append(
            f"| {item['fold']} | {item['best_epoch']} | {item['correct']} | {item['acc']:.7f} |"
        )
    return lines


def render_report(payload: dict) -> str:
    decision = payload["decision"]
    lines = [
        "# HFT-C1-Lite v1",
        "",
        f"Decision: `{decision}`",
        "",
        f"Branch: `{payload.get('branch', 'see environment.json')}`  ",
        f"Base commit: `{BASE_COMMIT}`  ",
        f"Run commit: `{payload.get('run_commit', 'see stage summary')}`",
        f"Device: `{payload.get('device', 'see environment.json')}`",
        f"Worktree: `{payload.get('worktree', ROOT)}`",
        "",
    ]
    if "formal" in payload:
        screen_report = payload["screen"]["final_screen"]
        formal = payload["formal"]
        lines.extend(
            [
                "## Screen",
                "",
                metric_line(screen_report["metrics"], 179),
                "",
                f"Confusion matrix: `{screen_report['metrics']['confusion_matrix']}`. "
                f"Repairs/damages: `{screen_report['paired_vs_C1']['repairs']}/"
                f"{screen_report['paired_vs_C1']['damages']}`. AD-sMCI/CN-sMCI/total adjacent errors: "
                f"`{screen_report['ad_smci_errors']}/{screen_report['cn_smci_errors']}/"
                f"{screen_report['boundary_errors']}`; AD-CN errors: `{screen_report['ad_cn_errors']}`.",
                "",
                *fold_markdown(screen_report),
                "",
                "## Formal",
                "",
                metric_line(formal["metrics"], 598),
                "",
                f"Confusion matrix: `{formal['metrics']['confusion_matrix']}`. "
                f"Fold ACC: `{formal['fold_acc_mean']:.7f} +/- {formal['fold_acc_sample_sd']:.7f}` "
                "(sample SD).",
                "",
                f"AD-sMCI/CN-sMCI/total adjacent errors: `{formal['ad_smci_errors']}/"
                f"{formal['cn_smci_errors']}/{formal['boundary_errors']}`; "
                f"AD-CN errors: `{formal['ad_cn_errors']}`.",
                "",
                *fold_markdown(formal),
                "",
                f"Runtime: `{formal['total_runtime_seconds']:.1f}` seconds. Parameters: "
                f"`{TOTAL_PARAMETERS}` total/inference (`+{HFT_PARAMETERS}` vs C1).",
                "",
                "## Minimal mechanism",
                "",
                *mechanism_markdown(formal["mechanism"]),
                "",
                f"Repairs/damages/changed predictions: `{formal['paired_vs_C1']['repairs']}/"
                f"{formal['paired_vs_C1']['damages']}/{formal['paired_vs_C1']['changed_predictions']}`.",
                "",
                "## Conclusion",
                "",
                (
                    "The experiment supports preserving fine-grained modality evidence beyond early summary-token "
                    "compression." if decision == "HFT_GO" else
                    "The preregistered result does not establish a clear gain from fine-grained modality evidence."
                ),
                "",
                f"Exceeded C1 correct count: `{'yes' if formal['metrics']['correct'] > C1['correct'] else 'no'}`. "
                f"Proceed to the next stage: `{'yes' if decision == 'HFT_GO' else 'no'}`.",
            ]
        )
    else:
        screen_report = payload["final_screen"]
        lines.extend(
            [
                "## Screen",
                "",
                metric_line(screen_report["metrics"], 179),
                "",
                f"Confusion matrix: `{screen_report['metrics']['confusion_matrix']}`. "
                f"Repairs/damages: `{screen_report['paired_vs_C1']['repairs']}/"
                f"{screen_report['paired_vs_C1']['damages']}`. AD-sMCI/CN-sMCI/total adjacent errors: "
                f"`{screen_report['ad_smci_errors']}/{screen_report['cn_smci_errors']}/"
                f"{screen_report['boundary_errors']}`; AD-CN errors: `{screen_report['ad_cn_errors']}`.",
                "",
                *fold_markdown(screen_report),
                "",
                f"Runtime: `{payload['runtime_seconds']:.1f}` seconds. Parameters: `{TOTAL_PARAMETERS}` "
                f"total/inference (`+{HFT_PARAMETERS}` vs C1).",
                "",
                "## Minimal mechanism",
                "",
                *mechanism_markdown(screen_report["mechanism"]),
                "",
                "## Conclusion",
                "",
                (
                    "The hard-fold screen supports the fine-grained-evidence hypothesis."
                    if decision == "SCREEN_GO"
                    else "The hard-fold screen does not establish support for the fine-grained-evidence hypothesis."
                ),
                "",
                f"Exceeded C1 hard-fold correct count: `{'yes' if screen_report['metrics']['correct'] > C1_SCREEN['correct'] else 'no'}`. "
                f"Proceed to formal ten-fold training: `{'yes' if decision == 'SCREEN_GO' else 'no'}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--smoke", action="store_true", help="run only epoch-0 checks and fold-4 3-epoch CUDA smoke")
    stage.add_argument("--screen", action="store_true", help="run only the preregistered hard-fold screen")
    stage.add_argument("--formal", action="store_true", help="run a fresh formal ten-fold experiment after SCREEN_GO")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / OUTPUT_REL,
        help="stage artifact root (default: experiments/hft_c1_lite_v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    require(ROOT.resolve() in output_root.parents, "Output root must remain inside the isolated worktree")
    context = load_context()
    stage = "smoke" if args.smoke else "screen" if args.screen else "formal"
    started = time.perf_counter()
    if args.smoke:
        result = run_smoke(context, output_root)
        decision = "SMOKE_PASS"
    elif args.screen:
        result = run_screen(context, output_root)
        decision = result["decision"]
    else:
        result = run_formal(context, output_root)
        decision = result["decision"]
    write_json(
        output_root / "status.json",
        {
            "stage": stage,
            "decision": decision,
            "completed": True,
            "branch": context["git"]["branch"],
            "base_commit": BASE_COMMIT,
            "run_commit": context["git"]["head"],
            "stage_runtime_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
