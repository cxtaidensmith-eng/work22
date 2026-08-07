"""Run Gate-Protected Private Residual v1 on the locked TADPOLE protocol."""

from __future__ import annotations

import argparse
import csv
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
from Model.gppr import GateProtectedPrivateResidualModel
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/gppr_v1")
C1_GIT_REF = "origin/experiment/cme-dual-branch-v1"
C1_OOF_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
C1_REPORT_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/report.json"
SOURCE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
BRANCH = "experiment/gate-protected-private-residual-v1"
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
EXPECTED_PARAMETERS = 862_977
C1_PARAMETERS = 862_971
ORIGINAL_PARAMETERS = 853_131
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
NON_COG_INDICES = (0, 1, 2, 3, 5)
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
    "private_shared_ratio": {
        "MRI": 0.07565631145,
        "PET": 0.06809283830,
        "CSF": 0.05355358347,
        "Risk": 0.07379904911,
        "COG": 0.27954564616,
        "ROI": 0.07636957616,
    },
}
EPSILON = 1e-12


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


def git_command(*args: str) -> str:
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


def read_git_text(ref: str, relative_path: str) -> str:
    return git_command("show", f"{ref}:{relative_path}")


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
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).cpu()) if total is not None else 0.0


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


def parse_c1_rows() -> tuple[list[dict], dict]:
    text = read_git_text(C1_GIT_REF, C1_OOF_REL)
    rows = list(csv.DictReader(io.StringIO(text)))
    require(len(rows) == 598, "C1 OOF row count changed")
    require(len({int(row["subject_index"]) for row in rows}) == 598, "C1 OOF subjects changed")
    metrics = metrics_from_rows(rows)
    for key in ("correct", "confusion_matrix"):
        require(metrics[key] == C1[key], f"C1 anchor changed: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    return rows, metrics


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    require(git_command("rev-parse", "HEAD") == SOURCE_COMMIT, "Source HEAD changed before run")
    require(git_command("branch", "--show-current") == BRANCH, "Experiment branch changed")
    device = torch.device("cuda:0")
    config_root = Path(tempfile.gettempdir()) / "work22_gppr_config"
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
    require(actual_modalities == MODALITY_NAMES, f"Modality order changed: {actual_modalities}")
    c1_rows, c1_metrics = parse_c1_rows()
    return {
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "c1_rows": c1_rows,
        "c1_metrics": c1_metrics,
        "c1_by_subject": {int(row["subject_index"]): row for row in c1_rows},
    }


def build_model(context: dict, gate_protected: bool = True):
    config = context["config"]
    SET_Random(SEED)
    model = GateProtectedPrivateResidualModel(
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
        gate_protected=gate_protected,
    ).to(context["device"])
    expected = EXPECTED_PARAMETERS if gate_protected else C1_PARAMETERS
    require(parameter_count(model) == expected, f"Parameter count changed: {parameter_count(model)}")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def make_fresh_training_objects(context: dict):
    model = build_model(context, gate_protected=True)
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


def beta_values(model: GateProtectedPrivateResidualModel) -> np.ndarray:
    return model.private_beta().detach().cpu().numpy().astype(np.float64)


def per_adapter_gradient(model: GateProtectedPrivateResidualModel) -> list[float]:
    return [module_gradient_norm(adapter) for adapter in model.private_adapters]


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke: {smoke_root}")
    smoke_root.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]

    c1_reference = build_model(context, gate_protected=False)
    gppr_reference = build_model(context, gate_protected=True)
    c1_reference.eval()
    gppr_reference.eval()
    with torch.no_grad():
        c1_raw, _, _, c1_intermediates = c1_reference(
            features, return_intermediates=True
        )
        gppr_raw, _, _, gppr_intermediates = gppr_reference(
            features, return_intermediates=True
        )
    logit_diff = float((c1_raw - gppr_raw).abs().max().cpu())
    residual_max = float(gppr_intermediates["private_residuals"].abs().max().cpu())
    require(logit_diff <= 1e-6, "Zero-adapter GPPR logits differ from C1")
    require(residual_max == 0.0, "GPPR private residual is not zero initialized")
    require(np.array_equal(beta_values(gppr_reference), np.ones(6)), "Initial beta is not one")
    del c1_reference, gppr_reference, c1_raw, gppr_raw
    torch.cuda.empty_cache()

    model, criterion, optimizer, scheduler = make_fresh_training_objects(context)
    max_gradient = np.zeros(6, dtype=np.float64)
    epoch_rows = []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Smoke epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"Smoke epoch{epoch}: non-finite gradient")
        gradients = np.asarray(per_adapter_gradient(model), dtype=np.float64)
        max_gradient = np.maximum(max_gradient, gradients)
        beta = beta_values(model)
        require(np.isfinite(beta).all(), "Smoke beta is non-finite")
        require(np.all((beta >= 0.5) & (beta <= 1.5)), "Smoke beta outside bounds")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        post_beta = beta_values(model)
        require(np.all((post_beta >= 0.5) & (post_beta <= 1.5)), "Updated beta outside bounds")
        with torch.no_grad():
            model.eval()
            eval_raw, _, _ = model(features)
            probability = score_logits(
                eval_raw,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )[1]
            probability_error = float((probability.sum(dim=1) - 1.0).abs().max().cpu())
        require(probability_error <= 1e-6, "Smoke probability sum mismatch")
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "beta_min": float(post_beta.min()),
                "beta_max": float(post_beta.max()),
                "probability_sum_max_abs_error": probability_error,
                **{
                    f"adapter_gradient_{modality}": float(gradients[index])
                    for index, modality in enumerate(MODALITY_NAMES)
                },
            }
        )
    for index in NON_COG_INDICES:
        require(max_gradient[index] > 0.0, f"{MODALITY_NAMES[index]} adapter received no gradient")
    payload = {
        "passed": True,
        "fold": 0,
        "epochs": 3,
        "parameter_count": parameter_count(model),
        "initial_c1_gppr_logit_max_abs_diff": logit_diff,
        "initial_private_residual_max_abs": residual_max,
        "beta_bounds_observed": [
            float(min(row["beta_min"] for row in epoch_rows)),
            float(max(row["beta_max"] for row in epoch_rows)),
        ],
        "non_cog_adapter_max_gradient": {
            MODALITY_NAMES[index]: float(max_gradient[index])
            for index in NON_COG_INDICES
        },
        "probabilities_sum_to_one": True,
        "all_finite": True,
    }
    write_json(smoke_root / "smoke_report.json", payload)
    write_csv(smoke_root / "epoch_metrics.csv", epoch_rows)
    print("SMOKE PASS", flush=True)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return payload


def fold_config(context: dict, fold: int) -> dict:
    return {
        "experiment": "Gate-Protected Private Residual v1",
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
        "optimizer": "Adam",
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "loss": "historical weighted main CE + three Original OVR auxiliary losses",
        "label_smoothing": 0.05,
        "orthogonality": False,
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "adapter_rank": 8,
        "private_input": "clean Modal Encoder token before directional noise and original modal gate",
        "beta": "1 + 0.5*tanh(a_m), six independent a_m initialized at zero",
        "shared_global_path": "historical C1 path unchanged",
        "parameter_count": EXPECTED_PARAMETERS,
    }


def private_diagnostics(
    intermediates: dict,
    mask: torch.Tensor,
    adapter_gradient_means: list[float],
) -> dict:
    applied = intermediates["applied_private_residuals"][mask]
    shared = intermediates["shared_modal_tokens_post_transformer"][mask]
    applied_norm = applied.norm(dim=-1)
    shared_norm = shared.norm(dim=-1)
    ratio = applied_norm / shared_norm.clamp_min(1e-8)
    cosine = F.cosine_similarity(
        intermediates["Y"][mask], intermediates["G"][mask], dim=-1
    )
    residual_means = applied_norm.mean(dim=0).detach().cpu().numpy()
    ratio_means = ratio.mean(dim=0).detach().cpu().numpy()
    beta = intermediates["private_beta"].detach().cpu().numpy()
    residual_total = float(residual_means.sum())
    cog_share = float(residual_means[4] / max(residual_total, 1e-12))
    collapse = bool(float(residual_means.max()) < 1e-6 or cog_share >= 0.90)
    return {
        "beta_by_modality": dict(zip(MODALITY_NAMES, beta.tolist())),
        "effective_residual_mean_norm_by_modality": dict(
            zip(MODALITY_NAMES, residual_means.tolist())
        ),
        "private_shared_ratio_by_modality": dict(
            zip(MODALITY_NAMES, ratio_means.tolist())
        ),
        "adapter_gradient_mean_by_modality": dict(
            zip(MODALITY_NAMES, [float(value) for value in adapter_gradient_means])
        ),
        "non_cog_adapter_gradient_mean": float(
            np.mean([adapter_gradient_means[index] for index in NON_COG_INDICES])
        ),
        "category_global_cosine_mean": float(cosine.mean().detach().cpu()),
        "cog_effective_residual_share": cog_share,
        "private_collapse": collapse,
        "collapse_rule": "all residuals near zero or COG >= 90% of total effective residual norm",
    }


def train_fold(context: dict, fold: int, formal_root: Path) -> tuple[dict, list[dict]]:
    final_dir = formal_root / f"fold_{fold:02d}"
    staging_dir = formal_root / f".fold_{fold:02d}_in_progress"
    require(not final_dir.exists() and not staging_dir.exists(), f"Fold output already exists: {fold}")
    staging_dir.mkdir(parents=True)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    model, criterion, optimizer, scheduler = make_fresh_training_objects(context)
    config_payload = fold_config(context, fold)
    epoch_rows = []
    best = None
    best_state = None
    adapter_gradients = [[] for _ in MODALITY_NAMES]
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"fold{fold} epoch{epoch}: non-finite gradient")
        gradients = per_adapter_gradient(model)
        for index, value in enumerate(gradients):
            adapter_gradients[index].append(value)
        beta_before = beta_values(model)
        require(np.all((beta_before >= 0.5) & (beta_before <= 1.5)), "beta outside bounds")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        beta_after = beta_values(model)
        require(np.all((beta_after >= 0.5) & (beta_after <= 1.5)), "updated beta outside bounds")

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
                "loss": float(loss.detach().cpu()),
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
                "beta_min": float(beta_after.min()),
                "beta_max": float(beta_after.max()),
                **{
                    f"adapter_gradient_{modality}": float(gradients[index])
                    for index, modality in enumerate(MODALITY_NAMES)
                },
            }
        )

    require(best is not None and best_state is not None, f"fold{fold}: no best state")
    for index in NON_COG_INDICES:
        require(max(adapter_gradients[index]) > 0.0, f"fold{fold}: {MODALITY_NAMES[index]} no gradient")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(features, return_intermediates=True)
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
    require(best_metrics == best["metrics"], f"fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"fold{fold}: saved prediction mismatch")
    gradient_means = [float(np.mean(values)) for values in adapter_gradients]
    diagnostics = private_diagnostics(best_intermediates, test_mask, gradient_means)
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(model),
        "added_parameters_vs_c1": parameter_count(model) - C1_PARAMETERS,
        "elapsed_seconds": elapsed,
        "config": config_payload,
        "diagnostics": diagnostics,
    }
    torch.save(best_state, staging_dir / "checkpoint_best.pt")
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_json(staging_dir / "config.json", config_payload)
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
        f"fold={fold} best_epoch={best['epoch']} correct={best_metrics['correct']} "
        f"ACC={best_metrics['acc']:.7f} Macro-F1={best_metrics['macro_f1']:.7f} "
        f"BACC={best_metrics['bacc']:.7f} Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def paired_comparison(candidate_rows: list[dict], c1_by_subject: dict) -> dict:
    repairs = []
    damages = []
    changed = []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        reference = c1_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth, "C1/GPPR truth mismatch")
        require(int(reference["fold"]) == int(row["fold"]), "C1/GPPR fold mismatch")
        old_prediction = int(reference["prediction"])
        new_prediction = int(row["prediction"])
        if old_prediction != truth and new_prediction == truth:
            repairs.append(subject)
        if old_prediction == truth and new_prediction != truth:
            damages.append(subject)
        if old_prediction != new_prediction:
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
        "repair_subject_indices": repairs,
        "damage_subject_indices": damages,
        "changed_subject_indices": changed,
        "exact_mcnemar_p_value": p_value,
    }


def summarize_diagnostics(fold_summaries: list[dict]) -> dict:
    def modality_mean(key: str) -> dict:
        return {
            modality: float(
                np.mean(
                    [
                        summary["diagnostics"][key][modality]
                        for summary in fold_summaries
                    ]
                )
            )
            for modality in MODALITY_NAMES
        }

    ratio = modality_mean("private_shared_ratio_by_modality")
    residual = modality_mean("effective_residual_mean_norm_by_modality")
    beta = modality_mean("beta_by_modality")
    gradient = modality_mean("adapter_gradient_mean_by_modality")
    cog_share = float(
        np.mean(
            [summary["diagnostics"]["cog_effective_residual_share"] for summary in fold_summaries]
        )
    )
    collapse = bool(max(residual.values()) < 1e-6 or cog_share >= 0.90)
    return {
        "beta_by_modality": beta,
        "effective_residual_mean_norm_by_modality": residual,
        "private_shared_ratio_by_modality": ratio,
        "ratio_delta_vs_c1": {
            modality: ratio[modality] - C1["private_shared_ratio"][modality]
            for modality in MODALITY_NAMES
        },
        "adapter_gradient_mean_by_modality": gradient,
        "non_cog_adapter_gradient_mean": float(
            np.mean([gradient[MODALITY_NAMES[index]] for index in NON_COG_INDICES])
        ),
        "category_global_cosine_mean": float(
            np.mean(
                [summary["diagnostics"]["category_global_cosine_mean"] for summary in fold_summaries]
            )
        ),
        "cog_effective_residual_share": cog_share,
        "private_collapse": collapse,
        "collapse_rule": "all residuals near zero or mean fold COG share >= 90%",
    }


def decision(metrics: dict, comparison: dict, diagnostics: dict) -> tuple[str, str]:
    matrix = metrics["confusion_matrix"]
    ad_smci = int(matrix[0][2] + matrix[2][0])
    cn_smci = int(matrix[1][2] + matrix[2][1])
    go = (
        metrics["correct"] >= 562
        and metrics["macro_f1"] >= 0.9175457
        and metrics["bacc"] >= 0.9163359
        and cn_smci <= 17
        and ad_smci <= 20
        and comparison["repairs"] > comparison["damages"]
    )
    enhanced_modalities = ("MRI", "PET", "Risk", "ROI")
    enhanced = all(
        diagnostics["private_shared_ratio_by_modality"][modality]
        > C1["private_shared_ratio"][modality]
        for modality in enhanced_modalities
    )
    stop_override = (
        metrics["correct"] <= 559
        or diagnostics["private_collapse"]
        or (ad_smci > C1["ad_smci_errors"] and cn_smci >= C1["cn_smci_errors"])
        or (
            metrics["macro_auc"] > C1["macro_auc"]
            and comparison["changed_predictions"] > 0
            and comparison["repairs"] <= comparison["damages"]
        )
    )
    near = (
        metrics["correct"] in {560, 561}
        and cn_smci <= 17
        and (metrics["macro_auc"] > C1["macro_auc"] or ad_smci < C1["ad_smci_errors"])
        and enhanced
    )
    if go and not stop_override:
        return "GPPR_GO", "Retain GPPR as the stronger single-model candidate; do not launch another experiment automatically."
    if near and not stop_override:
        return "GPPR_NEAR", "One future small adjustment could narrow the beta span from +/-0.50 to +/-0.25; it was not run."
    return "GPPR_STOP", "Next route: dual-boundary sMCI multi-prototype; not implemented."


def render_report(payload: dict) -> str:
    metrics = payload["metrics"]
    comparison = payload["comparison_vs_c1"]
    diagnostics = payload["diagnostics"]
    matrix = metrics["confusion_matrix"]
    lines = [
        "# Gate-Protected Private Residual v1",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        "## Performance",
        "",
        f"- Parameters: {payload['parameter_count']:,} (+{payload['added_parameters_vs_c1']} vs C1)",
        f"- Correct: {metrics['correct']}/598",
        f"- ACC: {metrics['acc']:.10f}",
        f"- Macro-F1: {metrics['macro_f1']:.10f}",
        f"- BACC: {metrics['bacc']:.10f}",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.10f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.10f}",
        f"- Confusion matrix: `{matrix}`",
        f"- Fold ACC mean +/- sample SD: {payload['fold_acc_mean']:.10f} +/- {payload['fold_acc_sample_sd']:.10f}",
        f"- Training time: {payload['training_seconds']:.3f} s",
        "",
        "| Fold | Best epoch | ACC |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.10f} |"
        for row in payload["fold_results"]
    )
    lines.extend(
        [
            "",
            "## Compared with C1",
            "",
            f"- Correct delta: {payload['metric_delta_vs_c1']['correct']:+d}",
            f"- ACC delta: {payload['metric_delta_vs_c1']['acc']:+.10f}",
            f"- Macro-F1 delta: {payload['metric_delta_vs_c1']['macro_f1']:+.10f}",
            f"- BACC delta: {payload['metric_delta_vs_c1']['bacc']:+.10f}",
            f"- Probability Macro-AUC delta: {payload['metric_delta_vs_c1']['macro_auc']:+.10f}",
            f"- Repairs / damages: {comparison['repairs']} / {comparison['damages']}",
            f"- AD-sMCI errors: {payload['ad_smci_errors']}",
            f"- CN-sMCI errors: {payload['cn_smci_errors']}",
            f"- Exact McNemar p: {comparison['exact_mcnemar_p_value']:.10f}",
            "",
            "## Minimal mechanism diagnostics",
            "",
            f"- Beta: `{diagnostics['beta_by_modality']}`",
            f"- Effective private/shared ratios: `{diagnostics['private_shared_ratio_by_modality']}`",
            f"- Ratio delta vs C1: `{diagnostics['ratio_delta_vs_c1']}`",
            f"- Non-COG adapter mean gradient: {diagnostics['non_cog_adapter_gradient_mean']:.10f}",
            f"- Category-Global cosine: {diagnostics['category_global_cosine_mean']:.10f}",
            f"- COG effective residual share: {diagnostics['cog_effective_residual_share']:.10f}",
            f"- Private collapse: {diagnostics['private_collapse']}",
            "",
            "Shared/Global computation is the historical C1 path. Only the private residual reads the clean pre-noise, pre-modal-gate encoder token.",
            "",
            f"Next recommendation: {payload['next_recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_formal(context: dict, output_root: Path, smoke: dict, total_started: float) -> dict:
    formal_root = output_root / "formal"
    require(not formal_root.exists(), f"Refusing to overwrite formal output: {formal_root}")
    formal_root.mkdir(parents=True)
    formal_started = time.perf_counter()
    fold_summaries = []
    all_rows = []
    for fold in FOLDS:
        summary, rows = train_fold(context, fold, formal_root)
        fold_summaries.append(summary)
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: int(row["subject_index"]))
    require(len(all_rows) == 598, "OOF must contain 598 rows")
    subjects = [int(row["subject_index"]) for row in all_rows]
    require(len(set(subjects)) == 598, "OOF subjects are not unique")
    require(set(subjects) == set(range(598)), "OOF subject set changed")
    metrics = metrics_from_rows(all_rows)
    require(sum(sum(row) for row in metrics["confusion_matrix"]) == 598, "Confusion matrix total changed")
    comparison = paired_comparison(all_rows, context["c1_by_subject"])
    require(
        comparison["repairs"] - comparison["damages"] == metrics["correct"] - C1["correct"],
        "Paired comparison net does not equal correct delta",
    )
    diagnostics = summarize_diagnostics(fold_summaries)
    matrix = metrics["confusion_matrix"]
    ad_smci = int(matrix[0][2] + matrix[2][0])
    cn_smci = int(matrix[1][2] + matrix[2][1])
    outcome, recommendation = decision(metrics, comparison, diagnostics)
    fold_acc = np.asarray(
        [summary["best_metrics"]["acc"] for summary in fold_summaries], dtype=np.float64
    )
    metric_delta = {
        "correct": int(metrics["correct"] - C1["correct"]),
        **{
            key: float(metrics[key] - C1[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }
    training_seconds = float(sum(summary["elapsed_seconds"] for summary in fold_summaries))
    payload = {
        "experiment": "Gate-Protected Private Residual v1",
        "decision": outcome,
        "next_recommendation": recommendation,
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "run_command": f"{sys.executable} scripts/run_gppr_v1.py --run-all",
        "device": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sklearn": sklearn.__version__,
        },
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameters_vs_original": EXPECTED_PARAMETERS - ORIGINAL_PARAMETERS,
        "added_parameters_vs_c1": EXPECTED_PARAMETERS - C1_PARAMETERS,
        "metrics": metrics,
        "metric_delta_vs_c1": metric_delta,
        "comparison_vs_c1": comparison,
        "ad_smci_errors": ad_smci,
        "cn_smci_errors": cn_smci,
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_sd": float(fold_acc.std(ddof=1)),
        "fold_results": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "acc": summary["best_metrics"]["acc"],
                "macro_f1": summary["best_metrics"]["macro_f1"],
                "bacc": summary["best_metrics"]["bacc"],
                "macro_auc": summary["best_metrics"]["macro_auc"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for summary in fold_summaries
        ],
        "training_seconds": training_seconds,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "diagnostics": diagnostics,
        "smoke": smoke,
        "integrity": {
            "oof_rows": len(all_rows),
            "unique_subjects": len(set(subjects)),
            "confusion_matrix_sum": int(sum(sum(row) for row in metrics["confusion_matrix"])),
            "metrics_recomputed_from_saved_probabilities": True,
            "c1_not_used_by_training_or_selection": True,
        },
        "c1_anchor": C1,
    }
    write_csv(formal_root / "oof_predictions.csv", all_rows)
    write_csv(
        formal_root / "fold_metrics.csv",
        [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "correct": summary["best_metrics"]["correct"],
                "acc": summary["best_metrics"]["acc"],
                "macro_f1": summary["best_metrics"]["macro_f1"],
                "bacc": summary["best_metrics"]["bacc"],
                "macro_auc": summary["best_metrics"]["macro_auc"],
                "weighted_f1": summary["best_metrics"]["weighted_f1"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for summary in fold_summaries
        ],
    )
    write_csv(
        formal_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(metrics["confusion_matrix"])
        ],
    )
    write_json(formal_root / "metrics.json", metrics)
    write_json(formal_root / "comparison_vs_c1.json", comparison)
    write_json(formal_root / "report.json", payload)

    # Required readback: recompute once from the exact persisted probabilities.
    persisted_rows = read_csv(formal_root / "oof_predictions.csv")
    persisted_metrics = metrics_from_rows(persisted_rows)
    require(persisted_metrics == metrics, "Persisted OOF metric readback mismatch")
    write_json(output_root / "FINAL_REPORT.json", payload)
    (output_root / "FINAL_REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(
        f"FINAL decision={outcome} correct={metrics['correct']}/598 "
        f"ACC={metrics['acc']:.7f} Macro-F1={metrics['macro_f1']:.7f} "
        f"BACC={metrics['bacc']:.7f} Probability Macro-AUC={metrics['macro_auc']:.7f}",
        flush=True,
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run smoke then the single formal experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.run_all, "Use --run-all; partial experiment modes are intentionally unsupported")
    output_root = ROOT / OUTPUT_REL
    require(not output_root.exists(), f"Refusing to overwrite experiment output: {output_root}")
    output_root.mkdir(parents=True)
    total_started = time.perf_counter()
    context = load_context()
    config_payload = {
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "protocol": fold_config(context, -1),
        "c1_reference": {
            "git_ref": C1_GIT_REF,
            "oof_path": C1_OOF_REL,
            "report_path": C1_REPORT_REL,
            "rerun": False,
        },
    }
    write_json(output_root / "config.json", config_payload)
    smoke = run_smoke(context, output_root)
    run_formal(context, output_root, smoke, total_started)


if __name__ == "__main__":
    main()
