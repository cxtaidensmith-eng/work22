"""Run Counterfactual Marginal Evidence Guided Dual-Branch Learning v1."""

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
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
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
from Model.cme_dual_branch import CMEDualBranchModel
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/cme_dual_branch_v1")
ORIGINAL_OOF_REL = Path(
    "results/um_ler_frozen_v2/original_cache/original_oof_predictions.csv"
)
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
ARM_NAMES = {
    "c1": "c1_shared_private_control",
    "c2": "c2_counterfactual_marginal_evidence",
    "c3": "c3_boundary_conditional_cme",
}
EXPECTED_PARAMETERS = {"c1": 862_971, "c2": 866_444, "c3": 869_917}
ORIGINAL_PARAMETERS = 853_131
ORIGINAL = {
    "correct": 556,
    "acc": 0.9297658862876255,
    "macro_f1": 0.9140778251271361,
    "bacc": 0.9140778251271361,
    "macro_auc": 0.9560490697782983,
    "weighted_f1": 0.9297658862876255,
    "confusion_matrix": [[62, 0, 10], [0, 198, 11], [10, 11, 296]],
}
UTILITY_WARMUP_EPOCHS = 40
UTILITY_REFRESH_INTERVAL = 10
UTILITY_WEIGHT_C2 = 0.10
UTILITY_WEIGHT_C3 = 0.05
EPSILON = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        errors="replace",
    ).strip()


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def module_gradient_norm(module: torch.nn.Module) -> float:
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).cpu()) if total is not None else 0.0


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


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
    adjusted, probability, _ = score_logits(raw_logits, label_weight, tau)
    del adjusted
    metrics = probability_metrics(
        labels[mask].detach().cpu().numpy(),
        probability[mask].detach().cpu().numpy(),
    )
    return metrics, (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])


def prediction_rows(
    fold: int,
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    dataset_dict: dict,
    config,
    intermediates: dict | None = None,
    arm: str | None = None,
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
    base = []
    for row_index, subject_index in enumerate(source_indices):
        base.append(
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
    if intermediates is None or arm not in {"c2", "c3"}:
        return base
    if arm == "c2":
        weights = intermediates["router_weights"][mask].detach().cpu().numpy()
        for row_index, row in enumerate(base):
            for modality_index, modality in enumerate(MODALITY_NAMES):
                row[f"router_weight_{modality}"] = float(weights[row_index, modality_index])
    else:
        for boundary in ("ad_smci", "cn_smci"):
            weights = intermediates[f"router_weights_{boundary}"][mask].detach().cpu().numpy()
            for row_index, row in enumerate(base):
                for modality_index, modality in enumerate(MODALITY_NAMES):
                    row[f"router_weight_{boundary}_{modality}"] = float(
                        weights[row_index, modality_index]
                    )
    return base


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    device = torch.device("cuda:0")
    config_root = Path(tempfile.gettempdir()) / "work22_cme_config"
    config = Config_(str(config_root), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(config.DATA_SET == "TADPOLE" and config.Task == "AD_CN_SMCI", "Protocol changed")
    require(int(config.epochs) == EPOCHS and int(config.T_max) == EPOCHS, "Epoch protocol changed")
    SET_Random(SEED)
    feature_path, dictionary_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    require(tuple(class_names) == CLASS_NAMES, f"Class order changed: {class_names}")
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
    require(len(dataset_dict["Modal_Name"]) == 6, "Modality count changed")
    modal_alias = {
        "MRI": "MRI",
        "PET": "PET",
        "CSF": "CSF",
        "RISK_FACTOR": "Risk",
        "RISK": "Risk",
        "COGNITIVE_TEST": "COG",
        "COG": "COG",
        "ROI_AVERAGE": "ROI",
        "ROI": "ROI",
    }
    actual_modalities = tuple(
        modal_alias.get(str(value).upper(), str(value))
        for value in dataset_dict["Modal_Name"]
    )
    require(
        actual_modalities == MODALITY_NAMES,
        f"Modality order changed: {dataset_dict['Modal_Name']}",
    )
    original_path = ROOT / ORIGINAL_OOF_REL
    require(original_path.is_file(), f"Original OOF missing: {original_path}")
    original_rows = read_csv(original_path)
    require(len(original_rows) == 598, "Original OOF row count changed")
    original_metrics = metrics_from_rows(original_rows)
    for key in ("correct", "confusion_matrix"):
        require(original_metrics[key] == ORIGINAL[key], f"Original anchor mismatch: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(original_metrics[key] - ORIGINAL[key]) <= 5e-7, f"Original anchor mismatch: {key}")
    return {
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "original_path": original_path,
        "original_rows": original_rows,
        "original_by_subject": {
            int(row["subject_index"]): row for row in original_rows
        },
    }


def original_model(context: dict):
    SET_Random(SEED)
    return query.build_model(
        context["config"], context["dataset_dict"], "original", context["device"]
    )


def build_model(context: dict, arm: str) -> CMEDualBranchModel:
    config = context["config"]
    SET_Random(SEED)
    model = CMEDualBranchModel(
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
        cme_arm=arm,
        adapter_rank=8,
        router_hidden=16,
        modality_embedding_dim=8,
    ).to(context["device"])
    require(parameter_count(model) == EXPECTED_PARAMETERS[arm], f"{arm} parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def utility_probabilities(context: dict, raw_logits: torch.Tensor) -> torch.Tensor:
    # The historical deployed prediction is the logit-adjusted softmax.  CME
    # utility uses that same diagnostic probability for full and deleted paths.
    return score_logits(
        raw_logits,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )[1]


@torch.no_grad()
def generate_utility_cache(
    context: dict,
    model: CMEDualBranchModel,
    arm: str,
    train_mask: torch.Tensor,
    labels_override: torch.Tensor | None = None,
) -> dict:
    require(arm in {"c2", "c3"}, "Utility cache requested for non-utility arm")
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"] if labels_override is None else labels_override
    was_training = model.training
    model.eval()
    full_raw, _, _, full_intermediates = model(features, return_intermediates=True)
    full_probability = utility_probabilities(context, full_raw)
    deletion_raw = []
    deletion_pre_tokens = []
    for modality_index in range(6):
        deleted_raw, _, _, deleted_intermediates = model(
            features,
            return_intermediates=True,
            counterfactual_modal_index=modality_index,
            counterfactual_sample_mask=train_mask,
        )
        deletion_raw.append(deleted_raw)
        deletion_pre_tokens.append(deleted_intermediates["modal_tokens_pre_transformer"])
    if was_training:
        model.train()

    payload = {
        "arm": arm,
        "train_mask": train_mask.detach().clone(),
        "full_probability": full_probability.detach(),
        "target_requires_grad": False,
        "test_labels_consulted": False,
    }
    if arm == "c2":
        train_index = torch.where(train_mask)[0]
        train_labels = labels[train_mask]
        full_nll = torch.zeros_like(labels, dtype=full_probability.dtype)
        full_nll[train_mask] = -full_probability[
            train_index, train_labels
        ].clamp_min(EPSILON).log()
        deleted_nll = []
        for raw in deletion_raw:
            probability = utility_probabilities(context, raw)
            values = torch.zeros_like(full_nll)
            values[train_mask] = -probability[
                train_index, train_labels
            ].clamp_min(EPSILON).log()
            deleted_nll.append(values)
        deleted_nll = torch.stack(deleted_nll, dim=1)
        signed_delta = deleted_nll - full_nll.unsqueeze(1)
        positive = signed_delta.clamp_min(0.0)
        positive_sum = positive.sum(dim=1, keepdim=True)
        valid = train_mask & (positive_sum.squeeze(1) > 0.0)
        target = torch.zeros_like(positive)
        target[valid] = positive[valid] / positive_sum[valid].clamp_min(EPSILON)
        payload.update(
            {
                "target": target.detach(),
                "valid": valid.detach(),
                "signed_delta": signed_delta.detach(),
                "positive_utility": positive.detach(),
            }
        )
    else:
        boundary_payload = {}
        for boundary, class_pair in {
            "ad_smci": (0, 2),
            "cn_smci": (1, 2),
        }.items():
            pair = torch.as_tensor(class_pair, device=labels.device)
            eligible = torch.zeros_like(train_mask)
            train_labels = labels[train_mask]
            eligible[train_mask] = (
                (train_labels == class_pair[0]) | (train_labels == class_pair[1])
            )
            eligible_index = torch.where(eligible)[0]
            eligible_binary_target = (labels[eligible] == class_pair[1]).long()
            full_pair_probability = torch.softmax(full_raw[:, pair], dim=-1)
            full_nll = torch.zeros_like(labels, dtype=full_pair_probability.dtype)
            full_nll[eligible] = -full_pair_probability[
                eligible_index, eligible_binary_target
            ].clamp_min(EPSILON).log()
            deleted_nll = []
            for raw in deletion_raw:
                probability = torch.softmax(raw[:, pair], dim=-1)
                values = torch.zeros_like(full_nll)
                values[eligible] = -probability[
                    eligible_index, eligible_binary_target
                ].clamp_min(EPSILON).log()
                deleted_nll.append(values)
            deleted_nll = torch.stack(deleted_nll, dim=1)
            signed_delta = deleted_nll - full_nll.unsqueeze(1)
            positive = signed_delta.clamp_min(0.0)
            positive_sum = positive.sum(dim=1, keepdim=True)
            valid = eligible & (positive_sum.squeeze(1) > 0.0)
            target = torch.zeros_like(positive)
            target[valid] = positive[valid] / positive_sum[valid].clamp_min(EPSILON)
            boundary_payload[boundary] = {
                "target": target.detach(),
                "valid": valid.detach(),
                "eligible": eligible.detach(),
                "signed_delta": signed_delta.detach(),
                "positive_utility": positive.detach(),
            }
        payload["boundaries"] = boundary_payload

    require(all(not value.requires_grad for value in _utility_tensors(payload)), "Utility target retained a graph")
    require(all(bool(torch.isfinite(value).all()) for value in _utility_tensors(payload)), "Non-finite utility cache")
    # Retain only compact checks, not the six full deleted forward tensors.
    payload["deletion_check"] = {
        "train_deleted_max_abs": [
            float(value[train_mask, modality_index].abs().max().cpu())
            for modality_index, value in enumerate(deletion_pre_tokens)
        ],
        "test_unchanged_max_abs_diff": [
            float(
                (
                    value[~train_mask]
                    - full_intermediates["modal_tokens_pre_transformer"][~train_mask]
                )
                .abs()
                .max()
                .cpu()
            )
            for value in deletion_pre_tokens
        ],
    }
    require(max(payload["deletion_check"]["train_deleted_max_abs"]) == 0.0, "Train token deletion failed")
    require(max(payload["deletion_check"]["test_unchanged_max_abs_diff"]) == 0.0, "Test token changed during utility deletion")
    return payload


def _utility_tensors(payload) -> list[torch.Tensor]:
    tensors = []
    if isinstance(payload, torch.Tensor):
        return [payload]
    if isinstance(payload, dict):
        for value in payload.values():
            tensors.extend(_utility_tensors(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            tensors.extend(_utility_tensors(value))
    return tensors


def utility_loss(arm: str, intermediates: dict, cache: dict) -> tuple[torch.Tensor, dict]:
    zero = intermediates["raw_logits"].new_zeros(())
    if arm == "c1" or cache is None:
        return zero, {"valid_count": 0}
    if arm == "c2":
        valid = cache["valid"]
        if not bool(valid.any()):
            return zero, {"valid_count": 0}
        weights = intermediates["router_weights"][valid].clamp_min(EPSILON)
        target = cache["target"][valid]
        loss = F.kl_div(weights.log(), target, reduction="batchmean")
        return loss, {"valid_count": int(valid.sum())}
    losses = []
    counts = {}
    for boundary in ("ad_smci", "cn_smci"):
        valid = cache["boundaries"][boundary]["valid"]
        counts[f"valid_count_{boundary}"] = int(valid.sum())
        if bool(valid.any()):
            weights = intermediates[f"router_weights_{boundary}"][valid].clamp_min(EPSILON)
            target = cache["boundaries"][boundary]["target"][valid]
            losses.append(F.kl_div(weights.log(), target, reduction="batchmean"))
        else:
            losses.append(zero)
    return UTILITY_WEIGHT_C3 * losses[0] + UTILITY_WEIGHT_C3 * losses[1], counts


def private_diagnostics(
    intermediates: dict, mask: torch.Tensor, adapter_gradient_mean: float
) -> dict:
    residuals = intermediates["private_residuals"][mask]
    shared = intermediates["modal_tokens_post_transformer"][mask]
    residual_norm = residuals.norm(dim=-1)
    shared_norm = shared.norm(dim=-1)
    ratio = residual_norm / shared_norm.clamp_min(1e-8)
    cosine = F.cosine_similarity(
        intermediates["Y"][mask], intermediates["G"][mask], dim=-1
    )
    residual_means = residual_norm.mean(dim=0).detach().cpu().tolist()
    return {
        "residual_mean_norm_by_modality": dict(zip(MODALITY_NAMES, residual_means)),
        "private_shared_ratio_by_modality": dict(
            zip(MODALITY_NAMES, ratio.mean(dim=0).detach().cpu().tolist())
        ),
        "adapter_gradient_mean": float(adapter_gradient_mean),
        "category_global_cosine_mean": float(cosine.mean().detach().cpu()),
        "private_collapse": bool(max(residual_means) < 1e-6 or adapter_gradient_mean <= 0.0),
    }


def router_diagnostics(
    arm: str,
    intermediates: dict,
    cache: dict,
    test_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    if arm == "c2":
        boundaries = {"all": (intermediates["router_weights"], cache)}
    else:
        boundaries = {
            boundary: (
                intermediates[f"router_weights_{boundary}"],
                cache["boundaries"][boundary],
            )
            for boundary in ("ad_smci", "cn_smci")
        }
    output = {}
    for boundary, (weights, utility) in boundaries.items():
        test_weights = weights[test_mask]
        entropy = -(test_weights.clamp_min(EPSILON) * test_weights.clamp_min(EPSILON).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(6.0)
        top_frequency = torch.bincount(test_weights.argmax(dim=1), minlength=6).float()
        top_frequency = top_frequency / max(1, int(test_mask.sum()))
        valid = utility["valid"]
        observation_mask = utility.get("eligible", utility.get("train_mask", valid))
        if bool(valid.any()):
            target = utility["target"][valid]
            valid_weights = weights[valid].clamp_min(EPSILON)
            kl = F.kl_div(valid_weights.log(), target, reduction="batchmean")
        else:
            kl = weights.new_zeros(())
        if bool(observation_mask.any()):
            utility_mean = utility["positive_utility"][observation_mask].mean(dim=0)
            signed_delta_mean = utility["signed_delta"][observation_mask].mean(dim=0)
        else:
            utility_mean = weights.new_zeros(6)
            signed_delta_mean = weights.new_zeros(6)
        by_class = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_mask = test_mask & (labels == class_index)
            by_class[class_name] = dict(
                zip(MODALITY_NAMES, weights[class_mask].mean(dim=0).detach().cpu().tolist())
            )
        mean_weight = test_weights.mean(dim=0)
        output[boundary] = {
            "router_mean_by_modality": dict(
                zip(MODALITY_NAMES, mean_weight.detach().cpu().tolist())
            ),
            "router_mean_by_true_class": by_class,
            "utility_mean_by_modality": dict(
                zip(MODALITY_NAMES, utility_mean.detach().cpu().tolist())
            ),
            "deletion_loss_change_mean_by_modality": dict(
                zip(MODALITY_NAMES, signed_delta_mean.detach().cpu().tolist())
            ),
            "positive_utility_sample_proportion": float(
                valid.sum() / max(1, int(observation_mask.sum()))
            ),
            "router_entropy_mean": float(entropy.mean().detach().cpu()),
            "router_normalized_entropy_mean": float(normalized_entropy.mean().detach().cpu()),
            "router_max_weight_mean": float(test_weights.max(dim=1).values.mean().detach().cpu()),
            "router_top_modality_frequency": dict(
                zip(MODALITY_NAMES, top_frequency.detach().cpu().tolist())
            ),
            "utility_kl_mean": float(kl.detach().cpu()),
            "router_collapse": bool(
                float(normalized_entropy.mean()) < 0.25
                or float(mean_weight.max()) > 0.80
                or float(top_frequency.max()) > 0.90
            ),
            "valid_utility_count": int(valid.sum()),
        }
    if arm == "c3":
        left = intermediates["router_weights_ad_smci"][test_mask]
        right = intermediates["router_weights_cn_smci"][test_mask]
        output["boundary_router_cosine_mean"] = float(
            F.cosine_similarity(left, right, dim=1).mean().detach().cpu()
        )
        output["boundary_router_absolute_difference_mean"] = dict(
            zip(MODALITY_NAMES, (left - right).abs().mean(dim=0).detach().cpu().tolist())
        )
    return output


def make_fresh_training_objects(context: dict, arm: str):
    model = build_model(context, arm)
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
    require(int(scheduler.T_max) == EPOCHS, "Scheduler T_max changed")
    return model, criterion, optimizer, scheduler


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke directory: {smoke_root}")
    smoke_root.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]

    original = original_model(context)
    c1 = build_model(context, "c1")
    original.eval()
    c1.eval()
    with torch.no_grad():
        original_raw, original_branches, original_auxiliary = original(features)
        c1_raw, c1_branches, c1_auxiliary, c1_intermediates = c1(
            features, return_intermediates=True
        )
    initial_logit_diff = float((original_raw - c1_raw).abs().max().cpu())
    residual_max_abs = float(c1_intermediates["private_residuals"].abs().max().cpu())
    require(initial_logit_diff <= 1e-6, "C1 initial logits differ from Original")
    require(residual_max_abs == 0.0, "C1 private residual is not exactly zero")
    require(
        len(c1_branches) == len(c1_auxiliary) == 3
        and all(tuple(value.shape) == (598, 96) for value in c1_branches)
        and all(tuple(value.shape) == (598, 2) for value in c1_auxiliary),
        "C1 Original Query/OVR interface changed",
    )
    del original, c1, original_raw, c1_raw
    torch.cuda.empty_cache()

    arm_checks = {}
    c2_checkpoint_payload = None
    c2_reference = None
    for arm in ("c1", "c2"):
        model, criterion, optimizer, scheduler = make_fresh_training_objects(context, arm)
        maxima = {"adapter": 0.0, "router": 0.0, "global": 0.0}
        utility_cache = None
        last_utility_loss = 0.0
        for epoch in range(1, 4):
            if arm == "c2" and epoch == 1:
                utility_cache = generate_utility_cache(
                    context, model, arm, train_mask
                )
                altered_labels = labels.clone()
                altered_labels[test_mask] = (altered_labels[test_mask] + 1) % 3
                altered_cache = generate_utility_cache(
                    context, model, arm, train_mask, labels_override=altered_labels
                )
                require(
                    torch.equal(utility_cache["target"], altered_cache["target"])
                    and torch.equal(utility_cache["valid"], altered_cache["valid"]),
                    "Test labels entered the C2 utility target",
                )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            raw, branches, auxiliary, intermediates = model(
                features, return_intermediates=True
            )
            original_loss = criterion(raw, labels, train_mask, branches, auxiliary)
            kl_loss, _ = utility_loss(arm, intermediates, utility_cache)
            total_loss = original_loss + (
                UTILITY_WEIGHT_C2 * kl_loss if arm == "c2" else 0.0
            )
            require(bool(torch.isfinite(total_loss)), f"{arm} smoke non-finite loss")
            total_loss.backward()
            require(gradients_finite(model), f"{arm} smoke non-finite gradient")
            maxima["adapter"] = max(
                maxima["adapter"], module_gradient_norm(model.private_adapters)
            )
            maxima["global"] = max(
                maxima["global"], module_gradient_norm(model.Global_Message)
            )
            if arm == "c2":
                maxima["router"] = max(
                    maxima["router"], module_gradient_norm(model.cme_router)
                )
                weights = intermediates["router_weights"]
                require(
                    float((weights.sum(dim=1) - 1.0).abs().max()) <= 1e-6,
                    "C2 router weights do not sum to one",
                )
                last_utility_loss = float(kl_loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(context["config"].grad_clip)
            )
            optimizer.step()
            scheduler.step()
        require(maxima["adapter"] > 0.0, f"{arm} adapter received no gradient")
        require(maxima["global"] > 0.0, f"{arm} Global branch received no gradient")
        if arm == "c2":
            require(maxima["router"] > 0.0, "C2 router received no gradient by epoch 3")
        model.eval()
        with torch.no_grad():
            raw, _, _, intermediates = model(features, return_intermediates=True)
            probability = utility_probabilities(context, raw)
        require(
            float((probability.sum(dim=1) - 1.0).abs().max()) <= 1e-6,
            f"{arm} output probabilities do not sum to one",
        )
        arm_checks[arm] = {
            "epochs": 3,
            "finite_loss_and_gradients": True,
            "maximum_gradient_norms": maxima,
            "utility_loss": last_utility_loss,
            "probability_sum_max_abs_error": float(
                (probability.sum(dim=1) - 1.0).abs().max().cpu()
            ),
        }
        if arm == "c2":
            c2_checkpoint_payload = clone_cpu_state(model)
            c2_reference = {
                "raw": raw.detach().cpu(),
                "weights": intermediates["router_weights"].detach().cpu(),
            }
        del model, criterion, optimizer, scheduler
        torch.cuda.empty_cache()

    require(c2_checkpoint_payload is not None and c2_reference is not None, "C2 smoke state missing")
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save(c2_checkpoint_payload, checkpoint_path)
    reloaded = build_model(context, "c2")
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        reloaded_raw, _, _, reloaded_intermediates = reloaded(
            features, return_intermediates=True
        )
    checkpoint_logit_diff = float(
        (reloaded_raw.detach().cpu() - c2_reference["raw"]).abs().max()
    )
    checkpoint_router_diff = float(
        (
            reloaded_intermediates["router_weights"].detach().cpu()
            - c2_reference["weights"]
        )
        .abs()
        .max()
    )
    require(checkpoint_logit_diff == 0.0 and checkpoint_router_diff == 0.0, "Checkpoint roundtrip changed C2")
    del reloaded
    torch.cuda.empty_cache()

    c3, c3_criterion, c3_optimizer, _ = make_fresh_training_objects(context, "c3")
    c3_cache = generate_utility_cache(context, c3, "c3", train_mask)
    c3.train()
    c3_optimizer.zero_grad(set_to_none=True)
    c3_raw, c3_branches, c3_auxiliary, c3_intermediates = c3(
        features, return_intermediates=True
    )
    c3_original_loss = c3_criterion(
        c3_raw, labels, train_mask, c3_branches, c3_auxiliary
    )
    c3_weighted_utility, c3_counts = utility_loss("c3", c3_intermediates, c3_cache)
    c3_total = c3_original_loss + c3_weighted_utility
    require(bool(torch.isfinite(c3_total)), "C3 smoke non-finite loss")
    c3_total.backward()
    require(gradients_finite(c3), "C3 smoke non-finite gradient")
    require(c3_counts["valid_count_ad_smci"] > 0, "C3 AD-SMCI utility empty")
    require(c3_counts["valid_count_cn_smci"] > 0, "C3 CN-SMCI utility empty")
    arm_checks["c3"] = {
        "single_forward_backward": True,
        "weighted_utility_loss": float(c3_weighted_utility.detach().cpu()),
        **c3_counts,
    }
    del c3, c3_criterion, c3_optimizer
    torch.cuda.empty_cache()

    payload = {
        "passed": True,
        "fold": 0,
        "epochs": 3,
        "initial_equivalence": {
            "private_residual_max_abs": residual_max_abs,
            "c1_vs_original_logit_max_abs_diff": initial_logit_diff,
        },
        "arms": arm_checks,
        "counterfactual_checks": {
            "train_only_deletion": True,
            "test_label_invariance": True,
            "target_has_no_gradient": True,
        },
        "checkpoint_roundtrip": {
            "logit_max_abs_diff": checkpoint_logit_diff,
            "router_max_abs_diff": checkpoint_router_diff,
        },
    }
    write_json(smoke_root / "smoke_report.json", payload)
    print("SMOKE PASS", flush=True)
    return payload


def fold_config(context: dict, arm: str, fold: int) -> dict:
    return {
        "experiment": ARM_NAMES[arm],
        "arm": arm,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "subjects": 598,
        "features": 360,
        "modalities": list(MODALITY_NAMES),
        "class_order": list(CLASS_NAMES),
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "optimizer": "Adam",
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "loss": "historical weighted main CE + three Original OVR auxiliary losses",
        "label_smoothing": 0.05,
        "orthogonality": False,
        "graph_enabled": False,
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "adapter_rank": 8,
        "private_adapter_count": 6,
        "utility_probability": (
            "historical logit-adjusted three-class softmax"
            if arm == "c2"
            else "boundary-restricted raw-logit softmax" if arm == "c3" else "not applicable"
        ),
        "utility_warmup_epochs": UTILITY_WARMUP_EPOCHS,
        "utility_refresh_interval": UTILITY_REFRESH_INTERVAL,
        "utility_weight": 0.0 if arm == "c1" else (UTILITY_WEIGHT_C2 if arm == "c2" else 0.10),
        "global_path_note": "Historical Original Global_Message(X_gated) is unchanged; private evidence enters Category only.",
    }


def train_fold(
    context: dict, arm: str, fold: int, output_root: Path
) -> tuple[dict, list[dict]]:
    final_dir = output_root / f"fold_{fold:02d}"
    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    require(not final_dir.exists(), f"Refusing to resume/overwrite: {final_dir}")
    require(not staging_dir.exists(), f"Retained staging directory exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    model, criterion, optimizer, scheduler = make_fresh_training_objects(context, arm)
    config_payload = fold_config(context, arm, fold)
    epoch_rows = []
    best = None
    best_state = None
    utility_cache = None
    adapter_gradients = []
    router_gradients = []
    global_gradients = []
    refresh_epochs = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        if arm in {"c2", "c3"} and epoch >= 41 and (epoch - 41) % 10 == 0:
            utility_cache = generate_utility_cache(
                context, model, arm, train_mask
            )
            refresh_epochs.append(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features, return_intermediates=True
        )
        original_loss = criterion(raw, labels, train_mask, branches, auxiliary)
        raw_utility_loss, utility_counts = utility_loss(
            arm, intermediates, utility_cache
        )
        if arm == "c2":
            weighted_utility_loss = UTILITY_WEIGHT_C2 * raw_utility_loss
        elif arm == "c3":
            weighted_utility_loss = raw_utility_loss
        else:
            weighted_utility_loss = raw.new_zeros(())
        total_loss = original_loss + weighted_utility_loss
        require(bool(torch.isfinite(total_loss)), f"{arm} fold{fold} epoch{epoch}: non-finite loss")
        total_loss.backward()
        require(gradients_finite(model), f"{arm} fold{fold} epoch{epoch}: non-finite gradient")
        adapter_gradient = module_gradient_norm(model.private_adapters)
        global_gradient = module_gradient_norm(model.Global_Message)
        if arm == "c2":
            router_gradient = module_gradient_norm(model.cme_router)
        elif arm == "c3":
            router_gradient = module_gradient_norm(model.cme_boundary_routers)
        else:
            router_gradient = 0.0
        adapter_gradients.append(adapter_gradient)
        global_gradients.append(global_gradient)
        router_gradients.append(router_gradient)
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(context["config"].grad_clip)
            )
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _, _ = model(features, return_intermediates=True)
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
                "original_loss": float(original_loss.detach().cpu()),
                "utility_loss_unweighted": float(raw_utility_loss.detach().cpu()),
                "utility_loss_weighted": float(weighted_utility_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "adapter_gradient_norm": adapter_gradient,
                "router_gradient_norm": router_gradient,
                "global_gradient_norm": global_gradient,
                "utility_valid_count": int(utility_counts.get("valid_count", 0)),
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
            }
        )

    require(best is not None and best_state is not None, f"{arm} fold{fold}: no best state")
    require(max(adapter_gradients) > 0.0, f"{arm} fold{fold}: adapter never received gradient")
    require(max(global_gradients) > 0.0, f"{arm} fold{fold}: Global never received gradient")
    if arm in {"c2", "c3"}:
        require(max(router_gradients) > 0.0, f"{arm} fold{fold}: router never received gradient")
        require(refresh_epochs == list(range(41, 392, 10)), "Utility refresh schedule changed")

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
            best_intermediates,
            arm,
        )
    require(best_metrics == best["metrics"], f"{arm} fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"{arm} fold{fold}: prediction readback mismatch")

    diagnostics = {
        "private": private_diagnostics(
            best_intermediates,
            test_mask,
            float(np.mean(adapter_gradients)),
        )
    }
    if arm in {"c2", "c3"}:
        best_utility_cache = generate_utility_cache(
            context, model, arm, train_mask
        )
        diagnostics["router_and_utility"] = router_diagnostics(
            arm, best_intermediates, best_utility_cache, test_mask, labels
        )
        diagnostics["utility_observation_scope"] = (
            "fold-train observations; folds overlap and these are not 598 unique OOF subjects"
        )
        diagnostics["deletion_check"] = best_utility_cache["deletion_check"]

    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "arm": arm,
        "name": ARM_NAMES[arm],
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(model),
        "added_parameters_vs_original": parameter_count(model) - ORIGINAL_PARAMETERS,
        "elapsed_seconds": elapsed,
        "utility_refresh_epochs": refresh_epochs,
        "gradient_summary": {
            "adapter_mean": float(np.mean(adapter_gradients)),
            "adapter_max": float(np.max(adapter_gradients)),
            "router_mean": float(np.mean(router_gradients)),
            "router_max": float(np.max(router_gradients)),
            "global_mean": float(np.mean(global_gradients)),
        },
        "config": config_payload,
        "diagnostics": diagnostics,
    }
    torch.save(best_state, staging_dir / "checkpoint_best.pt")
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_json(staging_dir / "summary.json", summary)
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
        f"{arm.upper()} fold={fold} best_epoch={best['epoch']} "
        f"correct={best_metrics['correct']} ACC={best_metrics['acc']:.7f} "
        f"Macro-F1={best_metrics['macro_f1']:.7f} BACC={best_metrics['bacc']:.7f} "
        f"Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def paired_comparison(candidate_rows: list[dict], original_by_subject: dict) -> dict:
    repairs = []
    damages = []
    changed = []
    both_correct = 0
    for row in candidate_rows:
        subject = int(row["subject_index"])
        original = original_by_subject[subject]
        truth = int(row["truth"])
        require(int(original["truth"]) == truth, "Original/candidate truth mismatch")
        require(int(original["fold"]) == int(row["fold"]), "Original/candidate fold mismatch")
        original_prediction = int(original["prediction"])
        candidate_prediction = int(row["prediction"])
        original_correct = original_prediction == truth
        candidate_correct = candidate_prediction == truth
        if original_correct and candidate_correct:
            both_correct += 1
        if not original_correct and candidate_correct:
            repairs.append(subject)
        if original_correct and not candidate_correct:
            damages.append(subject)
        if original_prediction != candidate_prediction:
            changed.append(subject)
    discordant = len(repairs) + len(damages)
    p_value = (
        float(binomtest(len(repairs), discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "repairs": len(repairs),
        "damages": len(damages),
        "net_repairs": len(repairs) - len(damages),
        "changed_predictions": len(changed),
        "both_correct": both_correct,
        "repair_subject_indices": repairs,
        "damage_subject_indices": damages,
        "changed_subject_indices": changed,
        "exact_mcnemar_p_value": p_value,
    }


def boundary_repair_damage(
    candidate_rows: list[dict], original_by_subject: dict
) -> dict:
    output = {}
    for name, classes in {"AD_SMCI": {0, 2}, "CN_SMCI": {1, 2}}.items():
        repair = 0
        damage = 0
        repair_directions = {}
        damage_directions = {}
        for row in candidate_rows:
            truth = int(row["truth"])
            if truth not in classes:
                continue
            subject = int(row["subject_index"])
            original_prediction = int(original_by_subject[subject]["prediction"])
            candidate_prediction = int(row["prediction"])
            original_correct = original_prediction == truth
            candidate_correct = candidate_prediction == truth
            if (
                not original_correct
                and candidate_correct
                and original_prediction in classes
            ):
                repair += 1
                key = f"{CLASS_NAMES[original_prediction]}_to_{CLASS_NAMES[candidate_prediction]}"
                repair_directions[key] = repair_directions.get(key, 0) + 1
            elif (
                original_correct
                and not candidate_correct
                and candidate_prediction in classes
            ):
                damage += 1
                key = f"{CLASS_NAMES[original_prediction]}_to_{CLASS_NAMES[candidate_prediction]}"
                damage_directions[key] = damage_directions.get(key, 0) + 1
        output[name] = {
            "repairs": repair,
            "damages": damage,
            "repair_directions": repair_directions,
            "damage_directions": damage_directions,
        }
    return output


def summarize_private(fold_summaries: list[dict]) -> dict:
    residual = {
        modality: float(
            np.mean(
                [
                    summary["diagnostics"]["private"]["residual_mean_norm_by_modality"][modality]
                    for summary in fold_summaries
                ]
            )
        )
        for modality in MODALITY_NAMES
    }
    ratio = {
        modality: float(
            np.mean(
                [
                    summary["diagnostics"]["private"]["private_shared_ratio_by_modality"][modality]
                    for summary in fold_summaries
                ]
            )
        )
        for modality in MODALITY_NAMES
    }
    return {
        "residual_mean_norm_by_modality": residual,
        "private_shared_ratio_by_modality": ratio,
        "adapter_gradient_mean": float(
            np.mean(
                [summary["diagnostics"]["private"]["adapter_gradient_mean"] for summary in fold_summaries]
            )
        ),
        "category_global_cosine_mean": float(
            np.mean(
                [summary["diagnostics"]["private"]["category_global_cosine_mean"] for summary in fold_summaries]
            )
        ),
        "private_collapse": bool(
            all(summary["diagnostics"]["private"]["private_collapse"] for summary in fold_summaries)
        ),
    }


def oof_router_summary(arm: str, rows: list[dict]) -> dict:
    if arm == "c2":
        boundaries = {"all": "router_weight_"}
    elif arm == "c3":
        boundaries = {
            "ad_smci": "router_weight_ad_smci_",
            "cn_smci": "router_weight_cn_smci_",
        }
    else:
        return {}
    output = {}
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    for boundary, prefix in boundaries.items():
        weights = np.asarray(
            [[float(row[f"{prefix}{modality}"]) for modality in MODALITY_NAMES] for row in rows],
            dtype=np.float64,
        )
        require(np.allclose(weights.sum(axis=1), 1.0, atol=1e-6), "OOF router sum mismatch")
        entropy = -(np.clip(weights, EPSILON, None) * np.log(np.clip(weights, EPSILON, None))).sum(axis=1)
        top = weights.argmax(axis=1)
        top_frequency = np.bincount(top, minlength=6) / len(top)
        by_class = {
            class_name: dict(zip(MODALITY_NAMES, weights[truth == class_index].mean(axis=0).tolist()))
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        mean_weight = weights.mean(axis=0)
        output[boundary] = {
            "mean_by_modality": dict(zip(MODALITY_NAMES, mean_weight.tolist())),
            "mean_by_true_class": by_class,
            "entropy_mean": float(entropy.mean()),
            "normalized_entropy_mean": float(entropy.mean() / math.log(6.0)),
            "maximum_weight_mean": float(weights.max(axis=1).mean()),
            "top_modality_frequency": dict(zip(MODALITY_NAMES, top_frequency.tolist())),
            "collapse": bool(
                entropy.mean() / math.log(6.0) < 0.25
                or mean_weight.max() > 0.80
                or top_frequency.max() > 0.90
            ),
        }
    if arm == "c3":
        left = np.asarray(
            [[float(row[f"router_weight_ad_smci_{m}"]) for m in MODALITY_NAMES] for row in rows]
        )
        right = np.asarray(
            [[float(row[f"router_weight_cn_smci_{m}"]) for m in MODALITY_NAMES] for row in rows]
        )
        output["boundary_router_cosine_mean"] = float(
            np.mean(np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)))
        )
        output["boundary_router_absolute_difference_mean"] = dict(
            zip(MODALITY_NAMES, np.abs(left - right).mean(axis=0).tolist())
        )
    return output


def repair_damage_router_summary(
    arm: str, rows: list[dict], paired: dict
) -> dict:
    if arm not in {"c2", "c3"}:
        return {}
    row_by_subject = {int(row["subject_index"]): row for row in rows}
    groups = {
        "repairs": paired["repair_subject_indices"],
        "damages": paired["damage_subject_indices"],
    }
    output = {}
    prefixes = (
        {"all": "router_weight_"}
        if arm == "c2"
        else {
            "ad_smci": "router_weight_ad_smci_",
            "cn_smci": "router_weight_cn_smci_",
        }
    )
    for group, subjects in groups.items():
        output[group] = {}
        for boundary, prefix in prefixes.items():
            if subjects:
                values = np.asarray(
                    [
                        [float(row_by_subject[subject][f"{prefix}{modality}"]) for modality in MODALITY_NAMES]
                        for subject in subjects
                    ]
                ).mean(axis=0)
                output[group][boundary] = dict(zip(MODALITY_NAMES, values.tolist()))
            else:
                output[group][boundary] = {modality: None for modality in MODALITY_NAMES}
    return output


def summarize_utility(arm: str, fold_summaries: list[dict]) -> dict:
    if arm == "c1":
        return {}
    boundaries = ("all",) if arm == "c2" else ("ad_smci", "cn_smci")
    output = {}
    for boundary in boundaries:
        diagnostics = [
            summary["diagnostics"]["router_and_utility"][boundary]
            for summary in fold_summaries
        ]
        output[boundary] = {
            "positive_utility_mean_by_modality": {
                modality: float(np.mean([item["utility_mean_by_modality"][modality] for item in diagnostics]))
                for modality in MODALITY_NAMES
            },
            "deletion_loss_change_mean_by_modality": {
                modality: float(
                    np.mean([item["deletion_loss_change_mean_by_modality"][modality] for item in diagnostics])
                )
                for modality in MODALITY_NAMES
            },
            "positive_utility_sample_proportion": float(
                np.mean([item["positive_utility_sample_proportion"] for item in diagnostics])
            ),
            "utility_kl_mean": float(np.mean([item["utility_kl_mean"] for item in diagnostics])),
            "valid_utility_count_sum_across_overlapping_train_folds": int(
                sum(item["valid_utility_count"] for item in diagnostics)
            ),
        }
    return output


def run_arm(context: dict, arm: str, experiment_root: Path) -> dict:
    output_root = experiment_root / ARM_NAMES[arm]
    if output_root.exists():
        suffix = 2
        while (experiment_root / f"{ARM_NAMES[arm]}_{suffix}").exists():
            suffix += 1
        output_root = experiment_root / f"{ARM_NAMES[arm]}_{suffix}"
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    fold_summaries = []
    rows = []
    for fold in FOLDS:
        summary, fold_rows = train_fold(context, arm, fold, output_root)
        fold_summaries.append(summary)
        rows.extend(fold_rows)
    require(len(rows) == 598, f"{arm} OOF row count mismatch")
    require(len({int(row["subject_index"]) for row in rows}) == 598, f"{arm} OOF duplicate subject")
    rows = sorted(rows, key=lambda row: int(row["subject_index"]))
    metrics = metrics_from_rows(rows)
    paired = paired_comparison(rows, context["original_by_subject"])
    fold_metrics_rows = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            "correct": summary["best_metrics"]["correct"],
            "acc": summary["best_metrics"]["acc"],
            "macro_f1": summary["best_metrics"]["macro_f1"],
            "bacc": summary["best_metrics"]["bacc"],
            "macro_auc": summary["best_metrics"]["macro_auc"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for summary in fold_summaries
    ]
    acc_values = np.asarray([row["acc"] for row in fold_metrics_rows], dtype=np.float64)
    diagnostics = {
        "private": summarize_private(fold_summaries),
        "router_oof": oof_router_summary(arm, rows),
        "utility_train_observations": summarize_utility(arm, fold_summaries),
        "repair_damage_router_weights": repair_damage_router_summary(arm, rows, paired),
    }
    if arm == "c3":
        diagnostics["adjacent_diagnostic_boundaries"] = boundary_repair_damage(
            rows, context["original_by_subject"]
        )
    report = {
        "arm": arm,
        "name": ARM_NAMES[arm],
        "output_directory": str(output_root.relative_to(ROOT)).replace("\\", "/"),
        "parameter_count": fold_summaries[0]["parameter_count"],
        "added_parameters_vs_original": fold_summaries[0]["added_parameters_vs_original"],
        "metrics": metrics,
        "fold_acc_mean": float(acc_values.mean()),
        "fold_acc_sample_std": float(acc_values.std(ddof=1)),
        "fold_best_epochs": [int(summary["best_epoch"]) for summary in fold_summaries],
        "fold_metrics": fold_metrics_rows,
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in fold_summaries)),
        "wall_seconds": time.perf_counter() - started,
        "paired_vs_original": paired,
        "diagnostics": diagnostics,
        "config": fold_config(context, arm, -1),
        "fold_summaries": fold_summaries,
    }
    write_json(output_root / "config.json", report["config"])
    write_csv(output_root / "oof_predictions.csv", rows)
    write_json(output_root / "metrics.json", metrics)
    write_csv(output_root / "fold_metrics.csv", fold_metrics_rows)
    write_csv(
        output_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(metrics["confusion_matrix"])
        ],
    )
    write_json(output_root / "report.json", report)
    reloaded_rows = read_csv(output_root / "oof_predictions.csv")
    require(metrics_from_rows(reloaded_rows) == metrics, f"{arm} OOF readback mismatch")
    print(
        f"{arm.upper()} COMPLETE correct={metrics['correct']}/598 ACC={metrics['acc']:.7f} "
        f"Macro-F1={metrics['macro_f1']:.7f} BACC={metrics['bacc']:.7f} "
        f"Probability Macro-AUC={metrics['macro_auc']:.7f}",
        flush=True,
    )
    return report


def c2_method_decision(report: dict) -> str:
    metrics = report["metrics"]
    bacc_ok = metrics["bacc"] >= ORIGINAL["bacc"] - 0.005
    if metrics["correct"] >= 557 and bacc_ok:
        return "CME_GO"
    if (
        metrics["correct"] == 556
        and metrics["macro_f1"] > ORIGINAL["macro_f1"]
        and metrics["macro_auc"] > ORIGINAL["macro_auc"]
        and bacc_ok
    ):
        return "CME_CONDITIONAL"
    return "CME_NO_GAIN"


def c3_trigger(report: dict) -> bool:
    metrics = report["metrics"]
    condition_a = metrics["correct"] >= 556
    condition_b = (
        metrics["correct"] == 555
        and metrics["macro_f1"] > ORIGINAL["macro_f1"]
        and metrics["macro_auc"] > ORIGINAL["macro_auc"]
        and metrics["bacc"] >= ORIGINAL["bacc"] - 0.005
    )
    return bool(condition_a or condition_b)


def metric_delta(report: dict) -> dict:
    return {
        "correct": int(report["metrics"]["correct"] - ORIGINAL["correct"]),
        **{
            metric: float(report["metrics"][metric] - ORIGINAL[metric])
            for metric in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }


def router_summary_rows(reports: dict) -> list[dict]:
    rows = []
    for arm in ("c2", "c3"):
        if arm not in reports:
            continue
        for boundary, payload in reports[arm]["diagnostics"]["router_oof"].items():
            if not isinstance(payload, dict) or "mean_by_modality" not in payload:
                continue
            rows.append(
                {
                    "arm": arm,
                    "boundary": boundary,
                    **{f"weight_{m}": payload["mean_by_modality"][m] for m in MODALITY_NAMES},
                    "normalized_entropy_mean": payload["normalized_entropy_mean"],
                    "maximum_weight_mean": payload["maximum_weight_mean"],
                    "collapse": payload["collapse"],
                }
            )
    return rows or [{"arm": "none", "boundary": "none", "collapse": False}]


def utility_summary_rows(reports: dict) -> list[dict]:
    rows = []
    for arm in ("c2", "c3"):
        if arm not in reports:
            continue
        for boundary, payload in reports[arm]["diagnostics"]["utility_train_observations"].items():
            rows.append(
                {
                    "arm": arm,
                    "boundary": boundary,
                    **{
                        f"positive_utility_{m}": payload["positive_utility_mean_by_modality"][m]
                        for m in MODALITY_NAMES
                    },
                    **{
                        f"signed_loss_change_{m}": payload["deletion_loss_change_mean_by_modality"][m]
                        for m in MODALITY_NAMES
                    },
                    "positive_sample_proportion": payload["positive_utility_sample_proportion"],
                    "utility_kl_mean": payload["utility_kl_mean"],
                }
            )
    return rows or [{"arm": "none", "boundary": "none"}]


def experimental_decision(report: dict) -> str:
    metrics = report["metrics"]
    if metrics["correct"] >= 557 and metrics["bacc"] >= ORIGINAL["bacc"] - 0.005:
        return "CME_GO"
    if (
        metrics["correct"] == 556
        and metrics["macro_f1"] > ORIGINAL["macro_f1"]
        and metrics["macro_auc"] > ORIGINAL["macro_auc"]
        and metrics["bacc"] >= ORIGINAL["bacc"] - 0.005
    ):
        return "CME_CONDITIONAL"
    return "CME_ROUTE_NO_NET_IMPROVEMENT"


def render_report(payload: dict) -> str:
    reports = payload["reports"]
    best_arm = payload["best_arm"]
    best = reports[best_arm]
    lines = [
        "# Counterfactual Marginal Evidence Guided Dual-Branch Learning v1",
        "",
        (
            "This experiment studies the patient-level conditional marginal diagnostic value "
            "of modality-private information and uses counterfactual deletion to supervise "
            "private-evidence selection in a dual-branch model."
        ),
        "",
        "## Decision",
        "",
        f"- Final decision: **{payload['final_decision']}**",
        f"- C2 method decision: **{payload['c2_method_decision']}**",
        f"- C3 status: **{payload['c3_status']}**",
        f"- Best experimental arm by Correct > Probability Macro-AUC > Macro-F1: **{best_arm.upper()}**",
        "",
        "## Pooled OOF metrics",
        "",
        "| Model | Params | Correct | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 | Repairs | Damages | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Original | {ORIGINAL_PARAMETERS} | {ORIGINAL['correct']}/598 | {ORIGINAL['acc']:.7f} | "
            f"{ORIGINAL['macro_f1']:.7f} | {ORIGINAL['bacc']:.7f} | {ORIGINAL['macro_auc']:.7f} | "
            f"{ORIGINAL['weighted_f1']:.7f} | - | - | - |"
        ),
    ]
    for arm in ("c1", "c2", "c3"):
        if arm not in reports:
            continue
        report = reports[arm]
        metrics = report["metrics"]
        paired = report["paired_vs_original"]
        lines.append(
            f"| {arm.upper()} | {report['parameter_count']} | {metrics['correct']}/598 | "
            f"{metrics['acc']:.7f} | {metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | "
            f"{metrics['macro_auc']:.7f} | {metrics['weighted_f1']:.7f} | "
            f"{paired['repairs']} | {paired['damages']} | {paired['exact_mcnemar_p_value']:.7g} |"
        )

    lines.extend(["", "## Confusion matrices and fold results", ""])
    for arm in ("c1", "c2", "c3"):
        if arm not in reports:
            continue
        report = reports[arm]
        lines.extend(
            [
                f"### {arm.upper()}",
                "",
                "```text",
                *[str(row) for row in report["metrics"]["confusion_matrix"]],
                "```",
                "",
                (
                    f"10-fold ACC = {report['fold_acc_mean']:.7f} ± "
                    f"{report['fold_acc_sample_std']:.7f} (sample SD); "
                    f"training time = {report['training_seconds']:.3f} s."
                ),
                f"Artifacts: `{report['output_directory']}`.",
                "",
                "| Fold | Best epoch | Correct | ACC |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} |"
                    for row in report["fold_metrics"]
                ],
                "",
            ]
        )

    c1_private = reports["c1"]["diagnostics"]["private"]
    lines.extend(
        [
            "## C1 private-residual diagnostics",
            "",
            f"- Mean adapter gradient norm: {c1_private['adapter_gradient_mean']:.7g}",
            f"- Mean Category–Global cosine: {c1_private['category_global_cosine_mean']:.7f}",
            f"- Private collapse: {c1_private['private_collapse']}",
            "",
            "| Modality | Residual mean norm | Private/shared norm ratio |",
            "|---|---:|---:|",
            *[
                f"| {modality} | {c1_private['residual_mean_norm_by_modality'][modality]:.7f} | "
                f"{c1_private['private_shared_ratio_by_modality'][modality]:.7f} |"
                for modality in MODALITY_NAMES
            ],
            "",
        ]
    )

    for arm in ("c2", "c3"):
        if arm not in reports:
            continue
        router = reports[arm]["diagnostics"]["router_oof"]
        utility = reports[arm]["diagnostics"]["utility_train_observations"]
        lines.extend([f"## {arm.upper()} router and counterfactual utility", ""])
        boundaries = ("all",) if arm == "c2" else ("ad_smci", "cn_smci")
        for boundary in boundaries:
            lines.extend(
                [
                    f"### {boundary.replace('_', '–')}",
                    "",
                    (
                        f"Router normalized entropy={router[boundary]['normalized_entropy_mean']:.7f}; "
                        f"mean max weight={router[boundary]['maximum_weight_mean']:.7f}; "
                        f"collapse={router[boundary]['collapse']}; "
                        f"positive-utility proportion={utility[boundary]['positive_utility_sample_proportion']:.7f}; "
                        f"mean KL={utility[boundary]['utility_kl_mean']:.7f}."
                    ),
                    "",
                    "| Modality | OOF router weight | Positive utility | Mean deletion loss change |",
                    "|---|---:|---:|---:|",
                    *[
                        f"| {modality} | {router[boundary]['mean_by_modality'][modality]:.7f} | "
                        f"{utility[boundary]['positive_utility_mean_by_modality'][modality]:.7f} | "
                        f"{utility[boundary]['deletion_loss_change_mean_by_modality'][modality]:.7f} |"
                        for modality in MODALITY_NAMES
                    ],
                    "",
                ]
            )
            lines.extend(
                [
                    "Router weights by true class:",
                    "",
                    "| Class | MRI | PET | CSF | Risk | COG | ROI |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                    *[
                        f"| {class_name} | "
                        + " | ".join(
                            f"{router[boundary]['mean_by_true_class'][class_name][modality]:.7f}"
                            for modality in MODALITY_NAMES
                        )
                        + " |"
                        for class_name in CLASS_NAMES
                    ],
                    "",
                ]
            )
            if arm == "c2":
                conditioned = reports[arm]["diagnostics"]["repair_damage_router_weights"]
                lines.extend(
                    [
                        "Router weights for Original-relative repairs and damages:",
                        "",
                        "| Group | MRI | PET | CSF | Risk | COG | ROI |",
                        "|---|---:|---:|---:|---:|---:|---:|",
                        *[
                            f"| {group} | "
                            + " | ".join(
                                f"{conditioned[group]['all'][modality]:.7f}"
                                for modality in MODALITY_NAMES
                            )
                            + " |"
                            for group in ("repairs", "damages")
                        ],
                        "",
                    ]
                )
        if arm == "c3":
            lines.extend(
                [
                    (
                        "The two adjacent diagnostic-boundary routers have mean cosine "
                        f"{router['boundary_router_cosine_mean']:.7f}."
                    ),
                    "",
                ]
            )

    c1_better = reports["c1"]["metrics"]["correct"] > ORIGINAL["correct"]
    c2_better_c1 = (
        reports["c2"]["metrics"]["correct"],
        reports["c2"]["metrics"]["macro_auc"],
        reports["c2"]["metrics"]["macro_f1"],
    ) > (
        reports["c1"]["metrics"]["correct"],
        reports["c1"]["metrics"]["macro_auc"],
        reports["c1"]["metrics"]["macro_f1"],
    )
    c2_router = reports["c2"]["diagnostics"]["router_oof"]["all"]
    utility_values = reports["c2"]["diagnostics"]["utility_train_observations"]["all"][
        "positive_utility_mean_by_modality"
    ]
    utility_rank = sorted(utility_values, key=utility_values.get, reverse=True)
    non_cog_frequency = sum(
        value for modality, value in c2_router["top_modality_frequency"].items() if modality != "COG"
    )
    paired = reports["c2"]["paired_vs_original"]
    if paired["repairs"] or paired["damages"]:
        utility_predictive = (
            "No reliable beneficial separation was established: repair/damage-conditioned "
            f"weights differed, but net repairs={paired['net_repairs']} "
            f"({paired['repairs']} repairs, {paired['damages']} damages)."
        )
    else:
        utility_predictive = "No repair/damage prediction changes occurred, so predictive separation cannot be established."
    if "c3" in reports:
        boundary = reports["c3"]["diagnostics"]["adjacent_diagnostic_boundaries"]
        net_ad = boundary["AD_SMCI"]["repairs"] - boundary["AD_SMCI"]["damages"]
        net_cn = boundary["CN_SMCI"]["repairs"] - boundary["CN_SMCI"]["damages"]
        boundary_answer = f"AD–SMCI net={net_ad}; CN–SMCI net={net_cn}."
        router_difference = (
            reports["c3"]["diagnostics"]["router_oof"]["boundary_router_cosine_mean"] < 0.99
        )
    else:
        boundary_answer = "C3 was not triggered, so no boundary-specific source can be assigned."
        router_difference = False
    support_hypothesis = (
        reports["c2"]["metrics"]["correct"] < reports["c1"]["metrics"]["correct"]
        and not reports["c2"]["diagnostics"]["private"]["private_collapse"]
    )
    lines.extend(
        [
            "## Required research questions",
            "",
            f"1. Did ordinary shared/private C1 improve performance? **{c1_better}**.",
            (
                "2. Did private residuals train and diverge? "
                f"Gradient={c1_private['adapter_gradient_mean']:.7g}; collapse={c1_private['private_collapse']}."
            ),
            f"3. Did C2 outperform C1 under the fixed ranking? **{c2_better_c1}**.",
            f"4. Modalities with positive marginal evidence, high to low: **{', '.join(utility_rank)}**.",
            (
                "5. Was the router only COG-dominant? It was strongly COG-dominant but did not meet the "
                f"predeclared collapse threshold: mean COG weight={c2_router['mean_by_modality']['COG']:.7f}; "
                f"collapse={c2_router['collapse']}; non-COG top-selection frequency={non_cog_frequency:.7f}."
            ),
            f"6. Could utility predict repairs/damages? {utility_predictive}",
            f"7. Did C2 reach 557 correct? **{reports['c2']['metrics']['correct'] >= 557}**.",
            (
                "8. If C3 ran, did adjacent diagnostic-boundary routers differ? "
                f"**{router_difference}**."
            ),
            f"9. Which boundary supplied the net gain? {boundary_answer}",
            (
                "10. Does the result support the hypothesis that private information is not automatically "
                f"effective complementary information? **{support_hypothesis}**: C1 showed useful private "
                "capacity, while non-collapsed counterfactual routing in C2 reduced net accuracy."
            ),
            "",
            "## Protocol and provenance",
            "",
            (
                "The six Original Query modal encoders, shared Modal Transformer, three private Query pools, "
                "three OVR heads, Message_MLP, historical Global Message, additive fusion, graph-disabled "
                "DIFFormer, and classifier were retained. The historical Global module consumes X_gated; "
                "keeping this interface was necessary for exact zero-adapter initialization equivalence. "
                "Consequently, counterfactual utility measures Category-token marginal evidence while the "
                "Global evidence path is fixed."
            ),
            "",
            (
                f"Branch={payload['branch']}; source HEAD={payload['source_commit']}; "
                f"device={payload['environment']['gpu']}; total wall time={payload['total_wall_seconds']:.3f} s."
            ),
            f"Original OOF SHA256={payload['original_oof_sha256']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize(context: dict, reports: dict, experiment_root: Path, started: float) -> dict:
    comparison_root = experiment_root / "comparison"
    require(not comparison_root.exists(), f"Comparison directory exists: {comparison_root}")
    comparison_root.mkdir(parents=True)
    best_arm = max(
        reports,
        key=lambda arm: (
            reports[arm]["metrics"]["correct"],
            reports[arm]["metrics"]["macro_auc"],
            reports[arm]["metrics"]["macro_f1"],
        ),
    )
    final_decision = experimental_decision(reports[best_arm])
    c2_decision = c2_method_decision(reports["c2"])
    comparison_rows = [
        {
            "model": "Original",
            "parameters": ORIGINAL_PARAMETERS,
            **{key: ORIGINAL[key] for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            "repairs": "",
            "damages": "",
            "decision": "reference",
        }
    ]
    for arm in ("c1", "c2", "c3"):
        if arm not in reports:
            continue
        comparison_rows.append(
            {
                "model": arm.upper(),
                "parameters": reports[arm]["parameter_count"],
                **{
                    key: reports[arm]["metrics"][key]
                    for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
                },
                "repairs": reports[arm]["paired_vs_original"]["repairs"],
                "damages": reports[arm]["paired_vs_original"]["damages"],
                "decision": experimental_decision(reports[arm]),
            }
        )
    paired_payload = {
        arm: {
            **reports[arm]["paired_vs_original"],
            "delta_vs_original": metric_delta(reports[arm]),
        }
        for arm in reports
    }
    write_csv(comparison_root / "final_comparison.csv", comparison_rows)
    write_json(comparison_root / "paired_comparisons.json", paired_payload)
    write_csv(comparison_root / "router_summary.csv", router_summary_rows(reports))
    write_csv(
        comparison_root / "modality_utility_summary.csv", utility_summary_rows(reports)
    )
    payload = {
        "branch": git_value("branch", "--show-current"),
        "source_commit": git_value("rev-parse", "HEAD"),
        "original_oof_sha256": file_sha256(context["original_path"]),
        "original": ORIGINAL,
        "reports": reports,
        "best_arm": best_arm,
        "best_metrics": reports[best_arm]["metrics"],
        "c2_method_decision": c2_decision,
        "c3_status": "COMPLETED" if "c3" in reports else "SKIP_C3_C2_NOT_PROMISING",
        "final_decision": final_decision,
        "total_wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "run_command": f'"{sys.executable}" scripts/run_cme_dual_branch_v1.py --formal',
    }
    write_json(experiment_root / "FINAL_REPORT.json", payload)
    (experiment_root / "FINAL_REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    return payload


def run_formal(context: dict, experiment_root: Path) -> dict:
    smoke_report_path = experiment_root / "smoke" / "smoke_report.json"
    require(smoke_report_path.is_file(), "Required shared smoke report is missing")
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    require(smoke_report.get("passed") is True, "Shared smoke did not pass")
    started = time.perf_counter()
    reports = {
        "c1": run_arm(context, "c1", experiment_root),
        "c2": run_arm(context, "c2", experiment_root),
    }
    if c3_trigger(reports["c2"]):
        reports["c3"] = run_arm(context, "c3", experiment_root)
        print("C3 trigger satisfied; conditional C3 completed", flush=True)
    else:
        print("SKIP_C3_C2_NOT_PROMISING", flush=True)
    payload = finalize(context, reports, experiment_root, started)
    print(
        f"FINAL {payload['final_decision']} best={payload['best_arm'].upper()} "
        f"correct={payload['best_metrics']['correct']}/598",
        flush=True,
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--render-report", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    expected_branch = git_value("branch", "--show-current")
    require(expected_branch.startswith("experiment/cme-dual-branch-v1"), f"Unexpected branch: {expected_branch}")
    if args.render_report:
        report_path = output_root / "FINAL_REPORT.json"
        require(report_path.is_file(), "FINAL_REPORT.json is missing")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        (output_root / "FINAL_REPORT.md").write_text(
            render_report(payload), encoding="utf-8"
        )
        print("REPORT RENDERED", flush=True)
        return
    context = load_context()
    if args.smoke:
        output_root.mkdir(parents=True, exist_ok=True)
        run_smoke(context, output_root)
    else:
        require(output_root.is_dir(), "Experiment root/smoke does not exist")
        run_formal(context, output_root)


if __name__ == "__main__":
    main()
