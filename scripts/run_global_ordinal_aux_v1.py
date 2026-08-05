from __future__ import annotations

import argparse
import csv
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
import sklearn
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query


MODEL_NAME = "Original Query + Global Ordinal Auxiliary"
SCHEMA_VERSION = "global-ordinal-aux-10fold-seed0-v1"
BRANCH = "experiment/global-ordinal-aux-v1"
BASE_COMMIT = "765720c1c0a3b2263502e9bf662fe33de7e45428"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/global_ordinal_aux_v1/tenfold_seed0")
REPORT_REL = Path("reports/global_ordinal_aux_v1_10fold_final.md")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
LAMBDA_ORDINAL = 0.2
EXPECTED_BASE_PARAMETERS = 853_131
EXPECTED_PARAMETERS = 853_230
EXPECTED_CONFIG_HASH = (
    "5f741494141e478a49c0aa6af13e332a2cb98f15a5f29d90d0c057c7804e0cc2"
)
METRIC_NAMES = ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
SOURCE_PATHS = (
    Path("Model/network.py"),
    Path("Model/models.py"),
    Path("Model/layers.py"),
    Path("Loss/loss_fn.py"),
    Path("Utils/utils.py"),
    Path("Utils/data_load.py"),
    CONFIG_REL,
    Path("scripts/run_query_free_multibranch.py"),
    Path("scripts/run_global_ordinal_aux_v1.py"),
)
ORIGINAL = {
    "correct": 556,
    "acc": 0.9297659,
    "macro_f1": 0.9140778,
    "bacc": 0.9140778,
    "macro_auc": 0.9560491,
    "weighted_f1": 0.9297659,
    "parameter_count": EXPECTED_BASE_PARAMETERS,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalized_text_hash(path: Path) -> str:
    import hashlib

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file(), f"Missing source: {relative.as_posix()}")
        result[relative.as_posix()] = query.file_hash(path)
    return result


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def canonical_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def class_mapping(class_names) -> dict:
    names = [str(name) for name in class_names]
    lookup = {canonical_name(name): index for index, name in enumerate(names)}
    require(len(lookup) == len(names), f"Duplicate normalized class names: {names}")
    require(
        {"cn", "smci", "ad"}.issubset(lookup),
        f"Required named classes missing: {names}",
    )
    label_by_stage = {
        "CN": int(lookup["cn"]),
        "sMCI": int(lookup["smci"]),
        "AD": int(lookup["ad"]),
    }
    rank_by_label = {
        label_by_stage["CN"]: 0,
        label_by_stage["sMCI"]: 1,
        label_by_stage["AD"]: 2,
    }
    require(len(rank_by_label) == 3, "Named class labels are not distinct")
    return {
        "dataset_class_names": names,
        "ordinal_order": ["CN", "sMCI", "AD"],
        "label_by_stage": label_by_stage,
        "rank_by_label": {str(key): value for key, value in rank_by_label.items()},
    }


def ordinal_ranks(labels: torch.Tensor, mapping: dict) -> torch.Tensor:
    ranks = torch.full_like(labels, -1)
    for label_text, rank in mapping["rank_by_label"].items():
        ranks[labels == int(label_text)] = int(rank)
    require(bool((ranks >= 0).all()), "Unmapped label found while constructing ordinal targets")
    return ranks


def fold_pos_weights(ranks: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    selected = ranks[train_mask]
    target_1 = selected >= 1
    target_2 = selected >= 2
    counts = []
    weights = []
    for target in (target_1, target_2):
        positive = int(target.sum().item())
        negative = int(target.numel() - positive)
        require(positive > 0 and negative > 0, "Ordinal fold target lacks a class")
        weight = torch.tensor(
            negative / positive, device=ranks.device, dtype=torch.float32
        )
        weights.append(weight)
        counts.append(
            {
                "positive_count": positive,
                "negative_count": negative,
                "pos_weight": float(weight.detach().cpu()),
            }
        )
    return weights[0], weights[1], {"target_1": counts[0], "target_2": counts[1]}


def ordinal_loss(
    model: torch.nn.Module,
    ranks: torch.Tensor,
    train_mask: torch.Tensor,
    pos_weight_1: torch.Tensor,
    pos_weight_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model.last_global_ordinal_logits
    require(logits is not None and tuple(logits.shape) == (ranks.numel(), 2), "Ordinal logits missing")
    target_1 = (ranks >= 1).to(dtype=logits.dtype)
    target_2 = (ranks >= 2).to(dtype=logits.dtype)
    loss_1 = F.binary_cross_entropy_with_logits(
        logits[train_mask, 0], target_1[train_mask], pos_weight=pos_weight_1
    )
    loss_2 = F.binary_cross_entropy_with_logits(
        logits[train_mask, 1], target_2[train_mask], pos_weight=pos_weight_2
    )
    return 0.5 * (loss_1 + loss_2), loss_1, loss_2


def adjusted_probabilities(
    logits: torch.Tensor, label_weight: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjusted = logits - tau * label_weight.to(logits).clamp_min(1e-8).log().view(1, -1)
    probabilities = torch.softmax(adjusted, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return adjusted, probabilities, predictions


def metric_bundle(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_weight: torch.Tensor,
    tau: float,
) -> dict:
    _, probabilities, predictions = adjusted_probabilities(
        logits[mask], label_weight, tau
    )
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    prediction = predictions.detach().cpu().numpy().astype(np.int64)
    probability = probabilities.detach().cpu().numpy()
    onehot = np.eye(probability.shape[1], dtype=np.int64)[truth]
    return {
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(sklearn.metrics.f1_score(truth, prediction, average="macro")),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(
            sklearn.metrics.roc_auc_score(onehot, probability, average="macro")
        ),
        "weighted_f1": float(
            sklearn.metrics.f1_score(truth, prediction, average="weighted")
        ),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(probability.shape[1])
        ).tolist(),
    }


def selection_macro_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_weight: torch.Tensor,
    tau: float,
) -> float:
    """Historical Original selection AUC, computed from adjusted scores."""
    adjusted, _, _ = adjusted_probabilities(logits[mask], label_weight, tau)
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    onehot = np.eye(adjusted.shape[1], dtype=np.int64)[truth]
    return float(
        sklearn.metrics.roc_auc_score(
            onehot, adjusted.detach().cpu().numpy(), average="macro"
        )
    )


def prediction_rows(
    fold: int,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    dataset_dict: dict,
    config,
    class_names: list[str],
) -> list[dict]:
    raw = logits[mask]
    adjusted, probabilities, predictions = adjusted_probabilities(
        raw, dataset_dict["Label_Weight"], float(config.logit_adjust_tau)
    )
    mask_numpy = mask.detach().cpu().numpy().astype(bool)
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[mask_numpy]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    raw_numpy = raw.detach().cpu().numpy()
    adjusted_numpy = adjusted.detach().cpu().numpy()
    probability_numpy = probabilities.detach().cpu().numpy()
    prediction_numpy = predictions.detach().cpu().numpy().astype(np.int64)
    rows = []
    for row_index, subject_index in enumerate(source_indices):
        row = {
            "fold": int(fold),
            "subject_index": int(subject_index),
            "truth": int(truth[row_index]),
            "truth_name": class_names[int(truth[row_index])],
            "prediction": int(prediction_numpy[row_index]),
            "prediction_name": class_names[int(prediction_numpy[row_index])],
        }
        for label_index, class_name in enumerate(class_names):
            safe_name = canonical_name(class_name).upper()
            row[f"raw_logit_{safe_name}"] = float(raw_numpy[row_index, label_index])
            row[f"adjusted_score_{safe_name}"] = float(adjusted_numpy[row_index, label_index])
            row[f"probability_{safe_name}"] = float(probability_numpy[row_index, label_index])
            row[f"probability_label_{label_index}"] = float(probability_numpy[row_index, label_index])
        rows.append(row)
    return rows


def metrics_from_rows(rows: list[dict], class_count: int) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [float(row[f"probability_label_{label}"]) for label in range(class_count)]
            for row in rows
        ],
        dtype=np.float64,
    )
    require(np.isfinite(probabilities).all(), "OOF probabilities contain NaN/Inf")
    require(
        np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
        "OOF probability rows do not sum to one",
    )
    onehot = np.eye(class_count, dtype=np.int64)[truth]
    return {
        "correct": int((truth == prediction).sum()),
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(sklearn.metrics.f1_score(truth, prediction, average="macro")),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(
            sklearn.metrics.roc_auc_score(onehot, probabilities, average="macro")
        ),
        "weighted_f1": float(
            sklearn.metrics.f1_score(truth, prediction, average="weighted")
        ),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(class_count)
        ).tolist(),
    }


def build_model(config, dataset_dict: dict, device: torch.device, ordinal: bool):
    SET_Random(SEED)
    return query.build_model(
        config,
        dataset_dict,
        "original",
        device,
        global_ordinal_aux=ordinal,
    )


def thresholds(model: torch.nn.Module) -> tuple[float, float]:
    tau1, tau2 = model.global_ordinal_thresholds()
    return float(tau1.detach().cpu()), float(tau2.detach().cpu())


def minimal_cuda_check(
    config,
    dataset_dict: dict,
    dataset_data: dict,
    ranks: torch.Tensor,
    device: torch.device,
) -> dict:
    baseline = build_model(config, dataset_dict, device, ordinal=False)
    require(
        parameter_count(baseline) == EXPECTED_BASE_PARAMETERS,
        f"Original Query parameter count changed: {parameter_count(baseline)}",
    )
    baseline_state = clone_cpu_state(baseline)
    del baseline
    torch.cuda.empty_cache()

    model = build_model(config, dataset_dict, device, ordinal=True)
    require(
        parameter_count(model) == EXPECTED_PARAMETERS,
        f"Ordinal parameter count changed: {parameter_count(model)}",
    )
    candidate_state = model.state_dict()
    common_names = set(baseline_state).intersection(candidate_state)
    require(set(baseline_state) == common_names, "Ordinal model changed legacy state keys")
    require(
        all(torch.equal(baseline_state[name], candidate_state[name].detach().cpu()) for name in common_names),
        "Ordinal head changed Original Query initialization",
    )
    train_mask, _ = dataset_data["Mask"][0]
    pos_weight_1, pos_weight_2, weight_summary = fold_pos_weights(ranks, train_mask)
    criterion = criterion_query_pool_no_orth(
        dataset_dict, device, label_smoothing=0.05
    )
    model.train()
    model.zero_grad(set_to_none=True)
    logits, representations, auxiliary = model(dataset_data["Feature"])
    loss_original = criterion(
        logits, dataset_data["Label"], train_mask, representations, auxiliary
    )
    loss_ordinal, loss_1, loss_2 = ordinal_loss(
        model, ranks, train_mask, pos_weight_1, pos_weight_2
    )
    loss_total = loss_original + LAMBDA_ORDINAL * loss_ordinal
    require(bool(torch.isfinite(loss_total)), "Minimal check produced non-finite total loss")
    require(bool(torch.isfinite(logits).all()), "Minimal check produced non-finite logits")
    require(
        bool(torch.isfinite(model.last_global_ordinal_logits).all()),
        "Minimal check produced non-finite ordinal logits",
    )
    tau1, tau2 = thresholds(model)
    require(tau2 > tau1, "Ordinal thresholds are not strictly ordered")
    loss_total.backward()
    head_gradients = [
        parameter.grad
        for parameter in model.global_ordinal_severity_head.parameters()
    ]
    require(all(gradient is not None for gradient in head_gradients), "Ordinal head gradient missing")
    require(
        all(bool(torch.isfinite(gradient).all()) for gradient in head_gradients),
        "Ordinal head gradient contains NaN/Inf",
    )
    head_gradient_norm = float(
        torch.sqrt(sum(gradient.detach().float().pow(2).sum() for gradient in head_gradients)).cpu()
    )
    require(head_gradient_norm > 0.0, "Ordinal head gradient is zero")
    require(
        all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        "Minimal check found non-finite parameter gradients",
    )
    result = {
        "passed": True,
        "fold": 0,
        "device": str(device),
        "base_parameter_count": EXPECTED_BASE_PARAMETERS,
        "ordinal_parameter_count": EXPECTED_PARAMETERS,
        "loss_original": float(loss_original.detach().cpu()),
        "loss_ordinal": float(loss_ordinal.detach().cpu()),
        "loss_ord_1": float(loss_1.detach().cpu()),
        "loss_ord_2": float(loss_2.detach().cpu()),
        "loss_total": float(loss_total.detach().cpu()),
        "ordinal_head_gradient_norm": head_gradient_norm,
        "tau1": tau1,
        "tau2": tau2,
        "pos_weights": weight_summary,
        "all_gradients_finite": True,
    }
    del model, criterion, logits, representations, auxiliary, loss_total
    torch.cuda.empty_cache()
    return result


def validate_fold_metrics(rows: list[dict], expected: dict, class_count: int) -> None:
    actual = metrics_from_rows(rows, class_count)
    for name in METRIC_NAMES:
        require(
            abs(float(actual[name]) - float(expected[name])) <= 1e-10,
            f"Fold prediction metric mismatch for {name}",
        )
    require(actual["confusion_matrix"] == expected["confusion_matrix"], "Fold confusion mismatch")


def load_completed_fold(
    fold: int,
    final_dir: Path,
    staging_dir: Path,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    dataset_dict: dict,
    class_count: int,
    locked_sources: dict,
) -> tuple[dict, list[dict]] | None:
    if not final_dir.exists():
        require(not staging_dir.exists(), f"Incomplete fold staging directory exists: {staging_dir}")
        return None
    require(final_dir.is_dir(), f"Fold path is not a directory: {final_dir}")
    require(not staging_dir.exists(), f"Fold has both final and staging directories: {final_dir}")
    required = ("summary.json", "epoch_metrics.csv", "best_predictions.csv", "checkpoint_best.pt")
    require(all((final_dir / name).is_file() for name in required), f"Completed fold {fold} is incomplete")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("formal_fold_passed") is True, f"Fold {fold} is not marked complete")
    require(int(summary.get("fold", -1)) == fold, f"Fold {fold} summary id mismatch")
    require(int(summary.get("parameter_count", -1)) == EXPECTED_PARAMETERS, "Fold parameter count mismatch")
    require(summary.get("source_hashes") == locked_sources, f"Fold {fold} source hash mismatch")
    require(
        summary.get("split_hash") == query.split_hash(dataset_dict["Index"], train_mask, test_mask),
        f"Fold {fold} split hash mismatch",
    )
    epoch_rows = read_csv(final_dir / "epoch_metrics.csv")
    require(len(epoch_rows) == EPOCHS, f"Fold {fold} epoch count mismatch")
    rows = read_csv(final_dir / "best_predictions.csv")
    require(len(rows) == int(test_mask.sum().item()), f"Fold {fold} prediction count mismatch")
    require({int(row["fold"]) for row in rows} == {fold}, f"Fold {fold} prediction fold mismatch")
    validate_fold_metrics(rows, summary["best_metrics"], class_count)
    require(
        query.file_hash(final_dir / "checkpoint_best.pt") == summary["checkpoint_sha256"],
        f"Fold {fold} checkpoint hash mismatch",
    )
    print(f"[fold{fold}] RESUME best_epoch={summary['best_epoch']} ACC={summary['best_metrics']['acc']:.4f}", flush=True)
    return summary, rows


def run_fold(
    fold: int,
    config,
    dataset_dict: dict,
    dataset_data: dict,
    ranks: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    output_root: Path,
    locked_sources: dict,
) -> tuple[dict, list[dict]]:
    final_dir = output_root / f"fold_{fold:02d}"
    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    train_mask, test_mask = dataset_data["Mask"][fold]
    resumed = load_completed_fold(
        fold,
        final_dir,
        staging_dir,
        train_mask,
        test_mask,
        dataset_dict,
        len(class_names),
        locked_sources,
    )
    if resumed is not None:
        return resumed
    staging_dir.mkdir(parents=True)

    model = build_model(config, dataset_dict, device, ordinal=True)
    require(parameter_count(model) == EXPECTED_PARAMETERS, "Fresh fold parameter count changed")
    criterion = criterion_query_pool_no_orth(
        dataset_dict, device, label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=config.Lr_Min
    )
    pos_weight_1, pos_weight_2, weight_summary = fold_pos_weights(ranks, train_mask)
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    best = None
    best_state = None
    epoch_rows = []
    final_tau1 = None
    final_tau2 = None
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, representations, auxiliary = model(features)
        loss_original = criterion(
            logits, labels, train_mask, representations, auxiliary
        )
        loss_ord, loss_ord_1, loss_ord_2 = ordinal_loss(
            model, ranks, train_mask, pos_weight_1, pos_weight_2
        )
        loss_total = loss_original + LAMBDA_ORDINAL * loss_ord
        require(bool(torch.isfinite(loss_total)), f"fold{fold} epoch{epoch}: non-finite loss")
        loss_total.backward()
        require(
            all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            f"fold{fold} epoch{epoch}: non-finite gradient",
        )
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_logits, eval_representations, eval_auxiliary = model(features)
            require(bool(torch.isfinite(eval_logits).all()), f"fold{fold} epoch{epoch}: invalid logits")
            test_loss_original = criterion(
                eval_logits, labels, test_mask, eval_representations, eval_auxiliary
            )
            metrics = metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            selection_auc = selection_macro_auc(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
            tau1, tau2 = thresholds(model)
        require(tau2 > tau1, f"fold{fold} epoch{epoch}: unordered thresholds")
        require(
            all(math.isfinite(float(metrics[name])) for name in METRIC_NAMES),
            f"fold{fold} epoch{epoch}: non-finite metrics",
        )
        require(math.isfinite(selection_auc), f"fold{fold} epoch{epoch}: non-finite selection AUC")
        score = (metrics["acc"], selection_auc, metrics["macro_f1"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(metrics),
                "tau1": tau1,
                "tau2": tau2,
                "selection_macro_auc": selection_auc,
            }
            best_state = clone_cpu_state(model)
        final_tau1, final_tau2 = tau1, tau2
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr_used": lr_used,
                "lr_next": float(optimizer.param_groups[0]["lr"]),
                "train_loss_total": float(loss_total.detach().cpu()),
                "train_loss_original": float(loss_original.detach().cpu()),
                "train_loss_ordinal": float(loss_ord.detach().cpu()),
                "train_loss_ordinal_1": float(loss_ord_1.detach().cpu()),
                "train_loss_ordinal_2": float(loss_ord_2.detach().cpu()),
                "test_loss_original": float(test_loss_original.detach().cpu()),
                "tau1": tau1,
                "tau2": tau2,
                "selection_macro_auc_adjusted_score": selection_auc,
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
        if epoch in {1, 100, 200, 300, 400}:
            print(
                f"[fold{fold}] epoch={epoch:03d}/400 ACC={metrics['acc']:.4f} "
                f"Macro-AUC(select/prob)={selection_auc:.4f}/{metrics['macro_auc']:.4f}",
                flush=True,
            )

    require(best is not None and best_state is not None, f"fold{fold}: no best state")
    require(final_tau1 is not None and final_tau2 is not None, f"fold{fold}: final thresholds missing")
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_logits, _, _ = model(features)
        best_metrics = metric_bundle(
            best_logits,
            labels,
            test_mask,
            dataset_dict["Label_Weight"],
            float(config.logit_adjust_tau),
        )
        rows = prediction_rows(
            fold, best_logits, labels, test_mask, dataset_dict, config, class_names
        )
    validate_fold_metrics(rows, best_metrics, len(class_names))
    for name in METRIC_NAMES:
        require(
            abs(float(best_metrics[name]) - float(best["metrics"][name])) <= 1e-10,
            f"fold{fold}: best reload mismatch for {name}",
        )
    require(best_metrics["confusion_matrix"] == best["metrics"]["confusion_matrix"], "Best confusion mismatch")

    checkpoint_path = staging_dir / "checkpoint_best.pt"
    torch.save(best_state, checkpoint_path)
    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    confusion_rows = [
        {
            "actual": class_names[row_index],
            **{
                f"predicted_{canonical_name(class_names[column_index]).upper()}": int(
                    best_metrics["confusion_matrix"][row_index][column_index]
                )
                for column_index in range(len(class_names))
            },
        }
        for row_index in range(len(class_names))
    ]
    write_csv(staging_dir / "best_confusion_matrix.csv", confusion_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "fold": fold,
        "formal_fold_passed": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "split_hash": query.split_hash(dataset_dict["Index"], train_mask, test_mask),
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "best_tau1": float(best["tau1"]),
        "best_tau2": float(best["tau2"]),
        "best_selection_macro_auc_adjusted_score": float(best["selection_macro_auc"]),
        "final_tau1": float(final_tau1),
        "final_tau2": float(final_tau2),
        "ordinal_pos_weights": weight_summary,
        "lambda_ordinal": LAMBDA_ORDINAL,
        "parameter_count": EXPECTED_PARAMETERS,
        "elapsed_seconds": elapsed,
        "source_hashes": locked_sources,
        "checkpoint_sha256": query.file_hash(checkpoint_path),
    }
    write_json(staging_dir / "summary.json", summary)
    staging_dir.rename(final_dir)
    print(
        f"[fold{fold}] PASS best_epoch={best['epoch']} ACC={best_metrics['acc']:.4f} "
        f"Macro-F1={best_metrics['macro_f1']:.4f} Macro-AUC(prob)={best_metrics['macro_auc']:.4f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def decision_for(metrics: dict) -> str:
    correct = int(metrics["correct"])
    if correct >= 557:
        return "SUCCESS"
    if correct == 556 and (
        float(metrics["macro_f1"]) > ORIGINAL["macro_f1"]
        or float(metrics["bacc"]) > ORIGINAL["bacc"]
    ):
        return "TIE"
    return "STOP"


def render_report(aggregate: dict, class_names: list[str]) -> str:
    metrics = aggregate["oof_metrics"]
    delta = aggregate["delta_vs_original"]
    folds = aggregate["fold_metrics"]
    confusion = metrics["confusion_matrix"]
    lines = [
        "# Original Query + Global Ordinal Auxiliary — 10-Fold Final Report",
        "",
        f"**Decision: {aggregate['decision']}**",
        "",
        "## Protocol",
        "",
        "TADPOLE / AD_CN_SMCI; folds 0..9; seed=0; 400 epochs; full-batch transductive; single model; no ensemble. The Original Query classification logits remain the inference output. A Global-only ordinal auxiliary loss uses CN < sMCI < AD with lambda=0.2.",
        "",
        "## Pooled OOF result",
        "",
        f"- Parameters: {aggregate['parameter_count']}",
        f"- Correct: {metrics['correct']}/598",
        f"- ACC: {metrics['acc']:.10f} ({delta['acc']:+.10f} vs Original)",
        f"- Macro-F1: {metrics['macro_f1']:.10f} ({delta['macro_f1']:+.10f})",
        f"- BACC: {metrics['bacc']:.10f} ({delta['bacc']:+.10f})",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.10f} ({delta['macro_auc']:+.10f})",
        f"- Weighted-F1: {metrics['weighted_f1']:.10f} ({delta['weighted_f1']:+.10f})",
        f"- Fold ACC: {aggregate['fold_acc_mean']:.10f} ± {aggregate['fold_acc_sample_std']:.10f} (sample SD)",
        f"- Final threshold mean: tau1={aggregate['final_tau1_mean']:.10f}, tau2={aggregate['final_tau2_mean']:.10f}",
        "",
        f"Confusion-matrix order: {class_names}",
        "",
        "```text",
        *[str(row) for row in confusion],
        "```",
        "",
        "## Per-fold best result",
        "",
        "| Fold | Best epoch | ACC |",
        "|---:|---:|---:|",
        *[
            f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.10f} |"
            for row in folds
        ],
        "",
        "## Runtime and provenance",
        "",
        f"- Training time: {aggregate['training_seconds']:.3f} s",
        f"- Total run time: {aggregate['total_seconds']:.3f} s",
        f"- Branch: `{BRANCH}`",
        f"- Source/base commit: `{aggregate['run_source_commit']}`",
        f"- Command: `{aggregate['command']}`",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    parser.add_argument("--report", type=Path, default=ROOT / REPORT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_started = time.perf_counter()
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    device = torch.device("cuda:0")
    require(git_value("branch", "--show-current") == BRANCH, "Wrong experiment branch")
    require(normalized_text_hash(ROOT / CONFIG_REL) == EXPECTED_CONFIG_HASH, "Historical config changed")
    config = Config_(str(ROOT), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(int(config.T_max) == EPOCHS, "Scheduler T_max must remain 400")
    SET_Random(SEED)
    feature_path, dict_path, _, loaded_class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    class_names = [str(name) for name in loaded_class_names]
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        loaded_class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    mapping = class_mapping(class_names)
    ranks = ordinal_ranks(dataset_data["Label"], mapping)
    require(int(dataset_data["Feature"].shape[0]) == 598, "Dataset subject count changed")
    locked_sources = source_hashes()
    minimal_check = minimal_cuda_check(
        config, dataset_dict, dataset_data, ranks, device
    )
    print(
        f"MINIMAL CUDA PASS parameters={EXPECTED_PARAMETERS} tau1={minimal_check['tau1']:.4f} "
        f"tau2={minimal_check['tau2']:.4f}",
        flush=True,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "model": MODEL_NAME,
        "dataset": config.DATA_SET,
        "task": config.Task,
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "scheduler": "CustomCosineAnnealingLR",
        "scheduler_T_max": EPOCHS,
        "lambda_ordinal": LAMBDA_ORDINAL,
        "ordinal_order": mapping["ordinal_order"],
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "best_rule": ["ACC", "historical adjusted-score Macro-AUC", "Macro-F1"],
        "graph_enabled": False,
    }
    write_json(output_root / "experiment_config.json", config_payload)

    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        summary, rows = run_fold(
            fold,
            config,
            dataset_dict,
            dataset_data,
            ranks,
            class_names,
            device,
            output_root,
            locked_sources,
        )
        fold_summaries.append(summary)
        oof_rows.extend(rows)

    require(source_hashes() == locked_sources, "Source changed during formal run")
    require(len(oof_rows) == 598, f"OOF row count mismatch: {len(oof_rows)}")
    require(
        len({int(row["subject_index"]) for row in oof_rows}) == 598,
        "OOF subject indices are not unique",
    )
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    oof_metrics = metrics_from_rows(oof_rows, len(class_names))
    require(oof_metrics["correct"] == round(oof_metrics["acc"] * 598), "Correct count mismatch")
    fold_metric_rows = [
        {
            "fold": int(summary["fold"]),
            "best_epoch": int(summary["best_epoch"]),
            **{name: float(summary["best_metrics"][name]) for name in METRIC_NAMES},
            "final_tau1": float(summary["final_tau1"]),
            "final_tau2": float(summary["final_tau2"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
        }
        for summary in fold_summaries
    ]
    fold_acc = np.asarray([row["acc"] for row in fold_metric_rows], dtype=np.float64)
    total_seconds = time.perf_counter() - suite_started
    training_seconds = float(sum(row["elapsed_seconds"] for row in fold_metric_rows))
    decision = decision_for(oof_metrics)
    delta = {
        "correct": int(oof_metrics["correct"] - ORIGINAL["correct"]),
        **{
            name: float(oof_metrics[name] - ORIGINAL[name])
            for name in METRIC_NAMES
        },
        "parameters": EXPECTED_PARAMETERS - EXPECTED_BASE_PARAMETERS,
    }
    command = " ".join([str(Path(sys.executable)), str(Path(__file__).relative_to(ROOT))])
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "decision": decision,
        "parameter_count": EXPECTED_PARAMETERS,
        "oof_metrics": oof_metrics,
        "delta_vs_original": delta,
        "original_reference": ORIGINAL,
        "fold_metrics": fold_metric_rows,
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_std": float(fold_acc.std(ddof=1)),
        "final_tau1_mean": float(np.mean([row["final_tau1"] for row in fold_metric_rows])),
        "final_tau2_mean": float(np.mean([row["final_tau2"] for row in fold_metric_rows])),
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "class_mapping": mapping,
        "minimal_cuda_check": minimal_check,
        "source_hashes": locked_sources,
        "run_source_commit": git_value("rev-parse", "HEAD"),
        "command": command,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
        },
        "all_folds_passed": True,
    }
    write_csv(output_root / "fold_metrics.csv", fold_metric_rows)
    write_csv(output_root / "oof_predictions.csv", oof_rows)
    write_json(output_root / "pooled_metrics.json", oof_metrics)
    write_csv(
        output_root / "confusion_matrix.csv",
        [
            {
                "actual": class_names[row_index],
                **{
                    f"predicted_{canonical_name(class_names[column_index]).upper()}": int(
                        oof_metrics["confusion_matrix"][row_index][column_index]
                    )
                    for column_index in range(len(class_names))
                },
            }
            for row_index in range(len(class_names))
        ],
    )
    write_json(output_root / "aggregate_summary.json", aggregate)
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "branch": BRANCH,
            "base_commit": BASE_COMMIT,
            "run_source_commit": aggregate["run_source_commit"],
            "command": command,
            "config": config_payload,
            "class_mapping": mapping,
            "minimal_cuda_check": minimal_check,
            "environment": aggregate["environment"],
            "source_hashes": locked_sources,
        },
    )
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        render_report(aggregate, class_names), encoding="utf-8"
    )

    reloaded_rows = read_csv(output_root / "oof_predictions.csv")
    reloaded_metrics = metrics_from_rows(reloaded_rows, len(class_names))
    require(reloaded_metrics == oof_metrics, "OOF readback recomputation mismatch")
    reloaded_aggregate = json.loads(
        (output_root / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    require(reloaded_aggregate["oof_metrics"] == oof_metrics, "Aggregate readback mismatch")
    print(
        f"{decision} correct={oof_metrics['correct']}/598 ACC={oof_metrics['acc']:.7f} "
        f"Macro-F1={oof_metrics['macro_f1']:.7f} BACC={oof_metrics['bacc']:.7f} "
        f"Macro-AUC(prob)={oof_metrics['macro_auc']:.7f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
