"""Run Dual-Boundary sMCI Multi-Prototype Residual v1.

The runner preserves the locked C1 training protocol, performs one three-epoch
CUDA smoke check, then runs one seed-0 formal ten-fold experiment.  Historical
C1 OOF predictions are read directly from the targeted remote Git object and
are never regenerated.
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
from Model.dbmp_r import DBMPRModel
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/dbmp_r_v1")
C1_GIT_REF = "refs/remotes/origin/experiment/cme-dual-branch-v1"
C1_OOF_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
C1_REPORT_REL = "experiments/cme_dual_branch_v1/c1_shared_private_control_2/report.json"
C1_OOF_SHA256 = "6f4ef505641f75236b01ae128f2836bf2f126ced43158f9c4ee21cc090898018"
SOURCE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
BRANCH = "experiment/dbmp-r-v1"
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
FORMAL_WARMUP_EPOCHS = 20
SMOKE_WARMUP_EPOCHS = 1
LAMBDA_PROTOTYPE = 0.05
TAU_ASSIGNMENT = 0.25
TAU_PROTOTYPE = 0.25
GAMMA_INITIAL = 0.05
EXPECTED_PARAMETERS = 862_973
C1_PARAMETERS = 862_971
ORIGINAL_PARAMETERS = 853_131
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
EPSILON = 1e-12
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


def tensor_gradient_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().float().norm().cpu())


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
    intermediates: dict,
) -> list[dict]:
    raw = raw_logits[mask]
    base_raw = intermediates["base_logits"][mask]
    adjusted, probability, prediction = score_logits(
        raw, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
    )
    _, _, base_prediction = score_logits(
        base_raw, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
    )
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    raw_values = raw.detach().cpu().numpy()
    base_values = base_raw.detach().cpu().numpy()
    adjusted_values = adjusted.detach().cpu().numpy()
    probability_values = probability.detach().cpu().numpy()
    prediction_values = prediction.detach().cpu().numpy().astype(np.int64)
    base_prediction_values = base_prediction.detach().cpu().numpy().astype(np.int64)
    delta_cn = intermediates["delta_CN"][mask].detach().cpu().numpy()
    delta_ad = intermediates["delta_AD"][mask].detach().cpu().numpy()
    residual_max = (
        intermediates["logit_residual"][mask].abs().max(dim=1).values.detach().cpu().numpy()
    )
    rows = []
    for row_index, subject_index in enumerate(source_indices):
        rows.append(
            {
                "fold": fold,
                "subject_index": int(subject_index),
                "truth": int(truth[row_index]),
                "prediction": int(prediction_values[row_index]),
                "base_prediction_same_checkpoint": int(base_prediction_values[row_index]),
                "residual_changed_prediction": int(
                    prediction_values[row_index] != base_prediction_values[row_index]
                ),
                "raw_logit_AD": float(raw_values[row_index, 0]),
                "raw_logit_CN": float(raw_values[row_index, 1]),
                "raw_logit_SMCI": float(raw_values[row_index, 2]),
                "base_logit_AD": float(base_values[row_index, 0]),
                "base_logit_CN": float(base_values[row_index, 1]),
                "base_logit_SMCI": float(base_values[row_index, 2]),
                "adjusted_score_AD": float(adjusted_values[row_index, 0]),
                "adjusted_score_CN": float(adjusted_values[row_index, 1]),
                "adjusted_score_SMCI": float(adjusted_values[row_index, 2]),
                "probability_AD": float(probability_values[row_index, 0]),
                "probability_CN": float(probability_values[row_index, 1]),
                "probability_SMCI": float(probability_values[row_index, 2]),
                "delta_CN": float(delta_cn[row_index]),
                "delta_AD": float(delta_ad[row_index]),
                "logit_residual_max_abs": float(residual_max[row_index]),
            }
        )
    return rows


def parse_c1_rows() -> tuple[list[dict], dict, dict]:
    text = read_git_text(C1_GIT_REF, C1_OOF_REL)
    digest = hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()
    if digest != C1_OOF_SHA256:
        # `git show` strips its final newline via git_command().  Accept the
        # exact blob digest after adding it back, but metrics remain mandatory.
        digest_without_newline = hashlib.sha256(text.encode("utf-8")).hexdigest()
        require(
            digest_without_newline == C1_OOF_SHA256,
            f"C1 OOF SHA changed: {digest}",
        )
        digest = digest_without_newline
    rows = list(csv.DictReader(io.StringIO(text)))
    require(len(rows) == 598, "C1 OOF row count changed")
    require(len({int(row["subject_index"]) for row in rows}) == 598, "C1 OOF subjects changed")
    metrics = metrics_from_rows(rows)
    for key in ("correct", "confusion_matrix"):
        require(metrics[key] == C1[key], f"C1 anchor changed: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    report = json.loads(read_git_text(C1_GIT_REF, C1_REPORT_REL))
    require(int(report["parameter_count"]) == C1_PARAMETERS, "C1 parameter anchor changed")
    return rows, metrics, report


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    require(git_command("rev-parse", "HEAD") == SOURCE_COMMIT, "Source HEAD changed before run")
    require(git_command("branch", "--show-current") == BRANCH, "Experiment branch changed")
    device = torch.device("cuda:0")
    config_root = Path(tempfile.gettempdir()) / "work22_dbmp_r_config"
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


def build_model(context: dict, dbmp_enabled: bool = True) -> DBMPRModel:
    config = context["config"]
    SET_Random(SEED)
    model = DBMPRModel(
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
        dbmp_enabled=dbmp_enabled,
        tau_assignment=TAU_ASSIGNMENT,
        tau_prototype=TAU_PROTOTYPE,
        gamma_initial=GAMMA_INITIAL,
    ).to(context["device"])
    expected = EXPECTED_PARAMETERS if dbmp_enabled else C1_PARAMETERS
    require(parameter_count(model) == expected, f"Parameter count changed: {parameter_count(model)}")
    require(not any(layer.use_graph for layer in model.GCN.layers), "DIFFormer graph unexpectedly enabled")
    return model


def make_fresh_training_objects(context: dict):
    model = build_model(context, dbmp_enabled=True)
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


def finite_prototype_terms(intermediates: dict) -> bool:
    keys = (
        "center_AD",
        "center_CN",
        "prototype_sMCI_CN",
        "prototype_sMCI_AD",
        "weight_CN",
        "weight_AD",
        "q_CN",
        "q_AD",
        "gamma",
        "delta_CN",
        "delta_AD",
        "logit_residual",
        "loss_prototype",
    )
    return all(bool(torch.isfinite(intermediates[key]).all()) for key in keys)


def prototype_snapshot(intermediates: dict) -> dict:
    weight_cn = intermediates["weight_CN"]
    weight_ad = intermediates["weight_AD"]
    cosine = F.cosine_similarity(
        intermediates["prototype_sMCI_CN"].view(1, -1),
        intermediates["prototype_sMCI_AD"].view(1, -1),
    ).squeeze(0)
    return {
        "gamma_CN": float(intermediates["gamma"][0].detach().cpu()),
        "gamma_AD": float(intermediates["gamma"][1].detach().cpu()),
        "prototype_cosine": float(cosine.detach().cpu()),
        "sum_w_CN": float(weight_cn.sum().detach().cpu()),
        "sum_w_AD": float(weight_ad.sum().detach().cpu()),
        "mean_w_CN": float(weight_cn.mean().detach().cpu()),
        "mean_w_AD": float(weight_ad.mean().detach().cpu()),
    }


def assert_prototype_source(intermediates: dict, train_mask: torch.Tensor) -> None:
    source = intermediates["train_mask"]
    require(torch.equal(source, train_mask.to(source)), "Prototype source is not exactly fold-train")
    union = (
        intermediates["train_AD_mask"]
        | intermediates["train_CN_mask"]
        | intermediates["train_sMCI_mask"]
    )
    require(torch.equal(union, source), "Prototype class masks do not partition fold-train")


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite smoke: {smoke_root}")
    smoke_root.mkdir(parents=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]

    c1_reference = build_model(context, dbmp_enabled=False)
    dbmp_reference = build_model(context, dbmp_enabled=True)
    c1_reference.eval()
    dbmp_reference.eval()
    with torch.no_grad():
        c1_raw, _, _ = c1_reference(features)
        dbmp_raw, _, _, dbmp_intermediates = dbmp_reference(
            features,
            labels=labels,
            train_mask=train_mask,
            residual_enabled=False,
            return_intermediates=True,
        )
    initial_diff = float((c1_raw - dbmp_raw).abs().max().cpu())
    require(initial_diff <= 1e-6, "Warmup-disabled DBMP logits differ from C1")
    require(float(dbmp_intermediates["logit_residual"].abs().max().cpu()) == 0.0, "Warmup residual not zero")
    assert_prototype_source(dbmp_intermediates, train_mask)

    altered_labels = labels.clone()
    altered_labels[test_mask] = (altered_labels[test_mask] + 1) % 3
    with torch.no_grad():
        altered_raw, _, _, altered_intermediates = dbmp_reference(
            features,
            labels=altered_labels,
            train_mask=train_mask,
            residual_enabled=True,
            return_intermediates=True,
        )
        original_label_raw, _, _, original_label_intermediates = dbmp_reference(
            features,
            labels=labels,
            train_mask=train_mask,
            residual_enabled=True,
            return_intermediates=True,
        )
    invariance_keys = (
        "center_AD",
        "center_CN",
        "prototype_sMCI_CN",
        "prototype_sMCI_AD",
        "weight_CN",
        "weight_AD",
    )
    require(
        all(torch.equal(altered_intermediates[key], original_label_intermediates[key]) for key in invariance_keys),
        "Test labels entered prototype construction",
    )
    require(torch.equal(altered_raw, original_label_raw), "Test labels changed DBMP logits")
    del c1_reference, dbmp_reference, c1_raw, dbmp_raw, altered_raw, original_label_raw
    torch.cuda.empty_cache()

    model, criterion, optimizer, scheduler = make_fresh_training_objects(context)
    epoch_rows = []
    gamma_gradient_max = 0.0
    gamma_gradient_cn_max = 0.0
    gamma_gradient_ad_max = 0.0
    representation_gradient_max = 0.0
    for epoch in range(1, 4):
        active = epoch > SMOKE_WARMUP_EPOCHS
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features,
            labels=labels,
            train_mask=train_mask,
            residual_enabled=active,
            return_intermediates=True,
        )
        intermediates["final_representation"].retain_grad()
        historical_loss = criterion(raw, labels, train_mask, branches, auxiliary)
        prototype_loss_weighted = (
            LAMBDA_PROTOTYPE * intermediates["loss_prototype"]
            if active
            else raw.new_zeros(())
        )
        total_loss = historical_loss + prototype_loss_weighted
        require(bool(torch.isfinite(total_loss)), f"Smoke epoch{epoch}: non-finite loss")
        require(finite_prototype_terms(intermediates), f"Smoke epoch{epoch}: non-finite prototype")
        assert_prototype_source(intermediates, train_mask)
        residual_sum_error = float(intermediates["logit_residual"].sum(dim=1).abs().max().detach().cpu())
        require(residual_sum_error <= 1e-7, "Smoke residual is not strictly zero-sum")
        ad_index = context["class_index"]["AD"]
        cn_index = context["class_index"]["CN"]
        smci_index = context["class_index"]["SMCI"]
        cn_gap_error = float(
            (
                (raw[:, smci_index] - raw[:, cn_index])
                - (
                    intermediates["base_logits"][:, smci_index]
                    - intermediates["base_logits"][:, cn_index]
                )
                - intermediates["delta_CN"]
            ).abs().max().detach().cpu()
        )
        ad_gap_error = float(
            (
                (raw[:, ad_index] - raw[:, smci_index])
                - (
                    intermediates["base_logits"][:, ad_index]
                    - intermediates["base_logits"][:, smci_index]
                )
                - intermediates["delta_AD"]
            ).abs().max().detach().cpu()
        )
        require(cn_gap_error <= 1e-6 and ad_gap_error <= 1e-6, "Smoke boundary-gap identity failed")
        gamma = intermediates["gamma"].detach()
        require(bool(((gamma >= 0.0) & (gamma <= 0.5)).all()), "Smoke gamma outside [0,0.5]")
        if not active:
            require(float((raw - intermediates["base_logits"]).abs().max().detach().cpu()) <= 1e-6, "Smoke epoch1 residual active")
        else:
            require(float(intermediates["logit_residual"].abs().max().detach().cpu()) > 0.0, "Smoke residual did not activate")
        total_loss.backward()
        require(gradients_finite(model), f"Smoke epoch{epoch}: non-finite gradient")
        representation_gradient = tensor_gradient_norm(intermediates["final_representation"].grad)
        gamma_gradient_cn = tensor_gradient_norm(model.a_CN.grad)
        gamma_gradient_ad = tensor_gradient_norm(model.a_AD.grad)
        gamma_gradient = float(
            np.hypot(gamma_gradient_cn, gamma_gradient_ad)
        )
        gamma_gradient_cn_max = max(gamma_gradient_cn_max, gamma_gradient_cn)
        gamma_gradient_ad_max = max(gamma_gradient_ad_max, gamma_gradient_ad)
        representation_gradient_max = max(representation_gradient_max, representation_gradient)
        gamma_gradient_max = max(gamma_gradient_max, gamma_gradient)
        if active:
            require(representation_gradient > 0.0, "Final representation received no gradient")
            require(
                gamma_gradient_cn > 0.0 and gamma_gradient_ad > 0.0,
                "Both gamma scalars must receive gradient after activation",
            )
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            updated_gamma = model.boundary_gamma()
            require(bool(((updated_gamma >= 0.0) & (updated_gamma <= 0.5)).all()), "Updated gamma outside bounds")
            model.eval()
            eval_raw, _, _, _ = model(
                features,
                labels=labels,
                train_mask=train_mask,
                residual_enabled=active,
                return_intermediates=True,
            )
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
                "residual_enabled": active,
                "total_loss": float(total_loss.detach().cpu()),
                "historical_loss": float(historical_loss.detach().cpu()),
                "prototype_loss_weighted": float(prototype_loss_weighted.detach().cpu()),
                "gamma_gradient_norm": gamma_gradient,
                "a_CN_gradient_abs": gamma_gradient_cn,
                "a_AD_gradient_abs": gamma_gradient_ad,
                "final_representation_gradient_norm": representation_gradient,
                "residual_sum_max_abs": residual_sum_error,
                "CN_sMCI_gap_identity_max_abs_error": cn_gap_error,
                "AD_sMCI_gap_identity_max_abs_error": ad_gap_error,
                "probability_sum_max_abs_error": probability_error,
                **prototype_snapshot(intermediates),
            }
        )
    require(
        gamma_gradient_cn_max > 0.0
        and gamma_gradient_ad_max > 0.0
        and representation_gradient_max > 0.0,
        "Smoke gradient checks failed",
    )
    payload = {
        "passed": True,
        "fold": 0,
        "epochs": 3,
        "smoke_warmup_epochs": SMOKE_WARMUP_EPOCHS,
        "formal_warmup_epochs_restored": FORMAL_WARMUP_EPOCHS,
        "parameter_count": parameter_count(model),
        "initial_c1_dbmp_logit_max_abs_diff": initial_diff,
        "prototype_train_only": True,
        "test_label_invariance": True,
        "epoch1_residual_disabled": True,
        "epoch2_residual_activated": True,
        "gamma_gradient_max": gamma_gradient_max,
        "gamma_CN_gradient_max": gamma_gradient_cn_max,
        "gamma_AD_gradient_max": gamma_gradient_ad_max,
        "final_representation_gradient_max": representation_gradient_max,
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
        "experiment": "Dual-Boundary sMCI Multi-Prototype Residual v1",
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "subjects": 598,
        "features": 360,
        "class_order": list(CLASS_NAMES),
        "class_index_by_name": context["class_index"],
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
        "loss": "historical weighted main CE on final logits + three unchanged C1 OVR losses + active 0.05 prototype loss",
        "label_smoothing": 0.05,
        "orthogonality": False,
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "C1_private_adapter_rank": 8,
        "prototype_space": "DIFFormer output / original classifier input h only",
        "tau_assignment": TAU_ASSIGNMENT,
        "tau_prototype": TAU_PROTOTYPE,
        "gamma_initial": GAMMA_INITIAL,
        "gamma_cap": 0.5,
        "lambda_prototype": LAMBDA_PROTOTYPE,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "parameter_count": EXPECTED_PARAMETERS,
        "prototype_statement": "boundary-conditioned sMCI prototypes; no clinical subtype or longitudinal progression claim",
    }


def boundary_identity_errors(
    context: dict, final_logits: torch.Tensor, intermediates: dict
) -> tuple[float, float, float]:
    ad_index = context["class_index"]["AD"]
    cn_index = context["class_index"]["CN"]
    smci_index = context["class_index"]["SMCI"]
    base = intermediates["base_logits"]
    residual_sum = float(
        intermediates["logit_residual"].sum(dim=1).abs().max().detach().cpu()
    )
    cn_gap = float(
        (
            (final_logits[:, smci_index] - final_logits[:, cn_index])
            - (base[:, smci_index] - base[:, cn_index])
            - intermediates["delta_CN"]
        ).abs().max().detach().cpu()
    )
    ad_gap = float(
        (
            (final_logits[:, ad_index] - final_logits[:, smci_index])
            - (base[:, ad_index] - base[:, smci_index])
            - intermediates["delta_AD"]
        ).abs().max().detach().cpu()
    )
    return residual_sum, cn_gap, ad_gap


def best_prototype_diagnostics(
    intermediates: dict,
    test_mask: torch.Tensor,
    postwarmup_cosines: list[float],
    gamma_gradient_cn: list[float],
    gamma_gradient_ad: list[float],
) -> dict:
    snapshot = prototype_snapshot(intermediates)
    delta_cn = intermediates["delta_CN"][test_mask].abs()
    delta_ad = intermediates["delta_AD"][test_mask].abs()
    cosine_fraction = float(
        np.mean(np.asarray(postwarmup_cosines, dtype=np.float64) > 0.98)
    )
    return {
        **snapshot,
        "CN_boundary_residual_mean_abs": float(delta_cn.mean().detach().cpu()),
        "CN_boundary_residual_max_abs": float(delta_cn.max().detach().cpu()),
        "AD_boundary_residual_mean_abs": float(delta_ad.mean().detach().cpu()),
        "AD_boundary_residual_max_abs": float(delta_ad.max().detach().cpu()),
        "postwarmup_prototype_cosine_above_0_98_fraction": cosine_fraction,
        "postwarmup_prototype_cosine_observations": len(postwarmup_cosines),
        "gamma_CN_gradient_mean_active_epochs": float(np.mean(gamma_gradient_cn)),
        "gamma_AD_gradient_mean_active_epochs": float(np.mean(gamma_gradient_ad)),
        "prototypes_differentiated_at_best": bool(snapshot["prototype_cosine"] <= 0.98),
    }


def train_fold(
    context: dict, fold: int, formal_root: Path
) -> tuple[dict, list[dict]]:
    final_dir = formal_root / f"fold_{fold:02d}"
    staging_dir = formal_root / f".fold_{fold:02d}_in_progress"
    require(
        not final_dir.exists() and not staging_dir.exists(),
        f"Fold output already exists: {fold}",
    )
    staging_dir.mkdir(parents=True)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    model, criterion, optimizer, scheduler = make_fresh_training_objects(context)
    config_payload = fold_config(context, fold)
    epoch_rows = []
    best = None
    best_state = None
    postwarmup_cosines = []
    gamma_cn_gradients = []
    gamma_ad_gradients = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        active = epoch > FORMAL_WARMUP_EPOCHS
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, intermediates = model(
            features,
            labels=labels,
            train_mask=train_mask,
            residual_enabled=active,
            return_intermediates=True,
        )
        historical_loss = criterion(raw, labels, train_mask, branches, auxiliary)
        weighted_prototype_loss = (
            LAMBDA_PROTOTYPE * intermediates["loss_prototype"]
            if active
            else raw.new_zeros(())
        )
        total_loss = historical_loss + weighted_prototype_loss
        require(bool(torch.isfinite(total_loss)), f"fold{fold} epoch{epoch}: non-finite loss")
        require(finite_prototype_terms(intermediates), f"fold{fold} epoch{epoch}: non-finite prototype")
        assert_prototype_source(intermediates, train_mask)
        residual_sum, cn_gap_error, ad_gap_error = boundary_identity_errors(
            context, raw, intermediates
        )
        require(
            residual_sum <= 1e-7
            and cn_gap_error <= 1e-6
            and ad_gap_error <= 1e-6,
            f"fold{fold} epoch{epoch}: residual identity failed",
        )
        total_loss.backward()
        require(gradients_finite(model), f"fold{fold} epoch{epoch}: non-finite gradient")
        gamma_cn_gradient = tensor_gradient_norm(model.a_CN.grad)
        gamma_ad_gradient = tensor_gradient_norm(model.a_AD.grad)
        if active:
            gamma_cn_gradients.append(gamma_cn_gradient)
            gamma_ad_gradients.append(gamma_ad_gradient)
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(context["config"].grad_clip)
            )
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _, eval_intermediates = model(
                features,
                labels=labels,
                train_mask=train_mask,
                residual_enabled=active,
                return_intermediates=True,
            )
            test_metrics, score = selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
            snapshot = prototype_snapshot(eval_intermediates)
        if active:
            postwarmup_cosines.append(snapshot["prototype_cosine"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(test_metrics),
                "residual_enabled": active,
            }
            best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "prototype_residual_enabled": active,
                "lambda_effective": LAMBDA_PROTOTYPE if active else 0.0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "historical_loss": float(historical_loss.detach().cpu()),
                "prototype_loss_unweighted": float(
                    intermediates["loss_prototype"].detach().cpu()
                ),
                "prototype_loss_weighted": float(weighted_prototype_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "gamma_CN_gradient": gamma_cn_gradient,
                "gamma_AD_gradient": gamma_ad_gradient,
                "residual_sum_max_abs": residual_sum,
                "CN_sMCI_gap_identity_max_abs_error": cn_gap_error,
                "AD_sMCI_gap_identity_max_abs_error": ad_gap_error,
                "acc": test_metrics["acc"],
                "macro_f1": test_metrics["macro_f1"],
                "bacc": test_metrics["bacc"],
                "macro_auc": test_metrics["macro_auc"],
                "weighted_f1": test_metrics["weighted_f1"],
                **snapshot,
            }
        )

    require(best is not None and best_state is not None, f"fold{fold}: no best state")
    require(
        max(gamma_cn_gradients) > 0.0 and max(gamma_ad_gradients) > 0.0,
        f"fold{fold}: both gamma scalars must receive gradient",
    )
    require(len(postwarmup_cosines) == EPOCHS - FORMAL_WARMUP_EPOCHS, "Cosine history incomplete")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(
            features,
            labels=labels,
            train_mask=train_mask,
            residual_enabled=bool(best["residual_enabled"]),
            return_intermediates=True,
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
        )
    require(best_metrics == best["metrics"], f"fold{fold}: best reload changed metrics")
    require(metrics_from_rows(rows) == best_metrics, f"fold{fold}: prediction readback mismatch")
    diagnostics = best_prototype_diagnostics(
        best_intermediates,
        test_mask,
        postwarmup_cosines,
        gamma_cn_gradients,
        gamma_ad_gradients,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "best_epoch": int(best["epoch"]),
        "best_residual_enabled": bool(best["residual_enabled"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(model),
        "added_parameters_vs_c1": parameter_count(model) - C1_PARAMETERS,
        "elapsed_seconds": elapsed,
        "config": config_payload,
        "prototype_diagnostics": diagnostics,
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
        require(int(reference["truth"]) == truth, "C1/DBMP truth mismatch")
        require(int(reference["fold"]) == int(row["fold"]), "C1/DBMP fold mismatch")
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


def summarize_prototypes(fold_summaries: list[dict], rows: list[dict]) -> dict:
    diagnostics = [summary["prototype_diagnostics"] for summary in fold_summaries]
    mean_keys = (
        "gamma_CN",
        "gamma_AD",
        "prototype_cosine",
        "sum_w_CN",
        "sum_w_AD",
        "mean_w_CN",
        "mean_w_AD",
        "postwarmup_prototype_cosine_above_0_98_fraction",
        "gamma_CN_gradient_mean_active_epochs",
        "gamma_AD_gradient_mean_active_epochs",
    )
    output = {
        key: float(np.mean([item[key] for item in diagnostics]))
        for key in mean_keys
    }
    delta_cn = np.asarray([abs(float(row["delta_CN"])) for row in rows])
    delta_ad = np.asarray([abs(float(row["delta_AD"])) for row in rows])
    output.update(
        {
            "CN_boundary_residual_mean_abs": float(delta_cn.mean()),
            "CN_boundary_residual_max_abs": float(delta_cn.max()),
            "AD_boundary_residual_mean_abs": float(delta_ad.mean()),
            "AD_boundary_residual_max_abs": float(delta_ad.max()),
            "prototype_effectively_differentiated": bool(
                output["prototype_cosine"] <= 0.98
                and output["postwarmup_prototype_cosine_above_0_98_fraction"] < 0.80
            ),
            "long_term_undifferentiated": bool(
                output["postwarmup_prototype_cosine_above_0_98_fraction"] >= 0.80
            ),
            "residual_changed_predictions_same_checkpoint": int(
                sum(int(row["residual_changed_prediction"]) for row in rows)
            ),
            "diagnostic_scope": "fold means at each selected best checkpoint; residual magnitudes pooled over 598 held-out OOF subjects",
        }
    )
    return output


def experimental_decision(
    metrics: dict,
    comparison: dict,
    prototypes: dict,
    ad_smci_errors: int,
    cn_smci_errors: int,
) -> tuple[str, str]:
    go = (
        metrics["correct"] >= 562
        and metrics["macro_f1"] >= 0.9175457
        and metrics["bacc"] >= 0.9163359
        and ad_smci_errors <= 21
        and cn_smci_errors <= 17
        and ad_smci_errors + cn_smci_errors <= 36
        and comparison["repairs"] > comparison["damages"]
        and not prototypes["long_term_undifferentiated"]
    )
    stop = (
        metrics["correct"] <= 559
        or ad_smci_errors >= 24
        or cn_smci_errors >= 20
        or prototypes["long_term_undifferentiated"]
        or comparison["damages"] > comparison["repairs"]
    )
    near = (
        metrics["correct"] in {560, 561}
        and ad_smci_errors < 24
        and cn_smci_errors < 20
        and (
            metrics["macro_f1"] > C1["macro_f1"]
            or metrics["bacc"] > C1["bacc"]
            or metrics["macro_auc"] > C1["macro_auc"]
        )
    )
    if go:
        return "DBMP_GO", "Retain DBMP-R as a stronger single-model candidate; no further experiment was started."
    if stop:
        return "DBMP_STOP", "Boundary-Conditioned Disentangled Gradient Learning; not implemented."
    if near:
        if prototypes["prototype_cosine"] > 0.98:
            recommendation = "A single future adjustment could lower tau_assignment from 0.25 to 0.20; it was not run."
        elif comparison["damages"] >= comparison["repairs"]:
            recommendation = "A single future adjustment could lower the gamma cap from 0.50 to 0.30; it was not run."
        else:
            recommendation = "A single future adjustment could lower lambda_prototype from 0.05 to 0.025; it was not run."
        return "DBMP_NEAR", recommendation
    return "DBMP_STOP", "Boundary-Conditioned Disentangled Gradient Learning; not implemented."


def render_report(payload: dict) -> str:
    metrics = payload["metrics"]
    comparison = payload["comparison_vs_c1"]
    prototypes = payload["prototype_diagnostics"]
    lines = [
        "# Dual-Boundary sMCI Multi-Prototype Residual v1",
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
        f"- Confusion matrix: `{metrics['confusion_matrix']}`",
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
            f"- Repairs / damages / changed: {comparison['repairs']} / {comparison['damages']} / {comparison['changed_predictions']}",
            f"- AD-sMCI errors: {payload['ad_smci_errors']}",
            f"- CN-sMCI errors: {payload['cn_smci_errors']}",
            "",
            "## Prototype diagnostics",
            "",
            f"- gamma_CN / gamma_AD: {prototypes['gamma_CN']:.10f} / {prototypes['gamma_AD']:.10f}",
            f"- Prototype cosine: {prototypes['prototype_cosine']:.10f}",
            f"- Effective masses sum(w_CN) / sum(w_AD): {prototypes['sum_w_CN']:.10f} / {prototypes['sum_w_AD']:.10f}",
            f"- Mean weights w_CN / w_AD: {prototypes['mean_w_CN']:.10f} / {prototypes['mean_w_AD']:.10f}",
            f"- CN boundary residual mean/max abs: {prototypes['CN_boundary_residual_mean_abs']:.10f} / {prototypes['CN_boundary_residual_max_abs']:.10f}",
            f"- AD boundary residual mean/max abs: {prototypes['AD_boundary_residual_mean_abs']:.10f} / {prototypes['AD_boundary_residual_max_abs']:.10f}",
            f"- Residual actually changed predictions: {prototypes['residual_changed_predictions_same_checkpoint']}",
            f"- Post-warmup cosine>0.98 fraction: {prototypes['postwarmup_prototype_cosine_above_0_98_fraction']:.10f}",
            f"- Prototypes effectively differentiated: {prototypes['prototype_effectively_differentiated']}",
            f"- Mechanism conclusion: {payload['mechanism_conclusion']}",
            "",
            "These are boundary-conditioned sMCI prototypes only; they are not claimed to be clinical subtypes or longitudinal disease progression states.",
            "",
            f"Next recommendation: {payload['next_recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_formal(
    context: dict, output_root: Path, smoke: dict, total_started: float
) -> dict:
    formal_root = output_root / "formal"
    require(not formal_root.exists(), f"Refusing to overwrite formal output: {formal_root}")
    formal_root.mkdir(parents=True)
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
        comparison["repairs"] - comparison["damages"]
        == metrics["correct"] - C1["correct"],
        "Paired comparison net does not equal correct delta",
    )
    prototypes = summarize_prototypes(fold_summaries, all_rows)
    matrix = metrics["confusion_matrix"]
    ad_smci = int(matrix[0][2] + matrix[2][0])
    cn_smci = int(matrix[1][2] + matrix[2][1])
    outcome, recommendation = experimental_decision(
        metrics, comparison, prototypes, ad_smci, cn_smci
    )
    if prototypes["long_term_undifferentiated"]:
        mechanism_conclusion = (
            "The two boundary-conditioned sMCI prototypes remained nearly collinear; "
            "the bounded residual changed no same-checkpoint held-out argmax, while the "
            "altered training trajectory was net harmful relative to C1."
        )
    elif prototypes["residual_changed_predictions_same_checkpoint"] == 0:
        mechanism_conclusion = (
            "The prototypes differentiated numerically, but the bounded residual did not "
            "cross any held-out decision boundary."
        )
    elif comparison["repairs"] > comparison["damages"]:
        mechanism_conclusion = (
            "The bounded dual-boundary residual produced a positive net correction relative to C1."
        )
    else:
        mechanism_conclusion = (
            "The bounded residual changed predictions, but its corrections were not net beneficial."
        )
    fold_acc = np.asarray(
        [summary["best_metrics"]["acc"] for summary in fold_summaries],
        dtype=np.float64,
    )
    metric_delta = {
        "correct": int(metrics["correct"] - C1["correct"]),
        **{
            key: float(metrics[key] - C1[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }
    training_seconds = float(
        sum(summary["elapsed_seconds"] for summary in fold_summaries)
    )
    payload = {
        "experiment": "Dual-Boundary sMCI Multi-Prototype Residual v1",
        "decision": outcome,
        "next_recommendation": recommendation,
        "mechanism_conclusion": mechanism_conclusion,
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "run_command": f"{sys.executable} scripts/run_dbmp_r_v1.py --run-all",
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
        "prototype_diagnostics": prototypes,
        "smoke": smoke,
        "integrity": {
            "oof_rows": len(all_rows),
            "unique_subjects": len(set(subjects)),
            "confusion_matrix_sum": int(
                sum(sum(row) for row in metrics["confusion_matrix"])
            ),
            "metrics_recomputed_from_saved_probabilities": True,
            "c1_not_used_by_training_or_selection": True,
            "prototype_sources_exactly_fold_train": True,
            "no_test_labels_in_prototype_construction": True,
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
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run the single smoke and then the single formal DBMP-R experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.run_all, "Use --run-all; partial modes are intentionally unsupported")
    output_root = ROOT / OUTPUT_REL
    require(
        not output_root.exists(),
        f"Refusing to overwrite experiment output: {output_root}",
    )
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
            "oof_sha256": C1_OOF_SHA256,
            "rerun": False,
        },
    }
    write_json(output_root / "config.json", config_payload)
    smoke = run_smoke(context, output_root)
    run_formal(context, output_root, smoke, total_started)


if __name__ == "__main__":
    main()
