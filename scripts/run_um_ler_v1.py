from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
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
from Model.um_ler import UncertaintyGuidedLocalEvidenceRefinement
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query


MODEL_NAME = "Uncertainty-Guided Multimodal Local Evidence Refinement v1"
BASE_COMMIT = "765720c1c0a3b2263502e9bf662fe33de7e45428"
BRANCH_PREFIX = "experiment/um-ler-v1"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
RESULTS_ROOT = Path("results")
SMOKE_REL = RESULTS_ROOT / "um_ler_smoke"
FINAL_JSON_REL = RESULTS_ROOT / "um_ler_v1_final_report.json"
FINAL_MD_REL = RESULTS_ROOT / "um_ler_v1_final_report.md"
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 2
CLASS_NAMES = ("AD", "CN", "SMCI")
EXPECTED_SUBJECTS = 598
EXPECTED_ORIGINAL_PARAMETERS = 853_131
EXPECTED_LOCAL_PARAMETERS = 3_393
EXPECTED_TOTAL_PARAMETERS = 856_524
REFERENCE_COMMIT = "af2957a68cbcc926a84d34de020b83b42417ddda"
REFERENCE_PATH = (
    "experiments/class_private_evidence_graph_v1/final/"
    "paired_best_vs_original_query.csv"
)
ORIGINAL = {
    "correct": 556,
    "acc": 0.9297659,
    "macro_f1": 0.9140778,
    "bacc": 0.9140778,
    "macro_auc": 0.9560491,
    "weighted_f1": 0.9297659,
    "confusion_matrix": [[62, 0, 10], [0, 198, 11], [10, 11, 296]],
    "parameter_count": EXPECTED_ORIGINAL_PARAMETERS,
}
METRIC_NAMES = ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")


@dataclass(frozen=True)
class UMLERConfig:
    top_k: int = 8
    temperature: float = 0.2
    margin_threshold: float = 0.30
    neighbor_confidence_threshold: float = 0.70
    gate_cap: float = 0.30
    nca_weight: float = 0.20
    refine_weight: float = 1.00
    projection_size: int = 32


DEFAULT = UMLERConfig()


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


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def probability_metrics(
    truth: np.ndarray, probabilities: np.ndarray
) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(probabilities.ndim == 2 and probabilities.shape[1] == 3, "Invalid probability shape")
    require(len(truth) == len(probabilities), "Truth/probability length mismatch")
    require(np.isfinite(probabilities).all(), "Probabilities contain NaN/Inf")
    require(
        np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
        "Probability rows do not sum to one",
    )
    prediction = probabilities.argmax(axis=1)
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "correct": int((truth == prediction).sum()),
        "acc": float(sklearn.metrics.accuracy_score(truth, prediction)),
        "macro_f1": float(
            sklearn.metrics.f1_score(truth, prediction, average="macro")
        ),
        "bacc": float(sklearn.metrics.balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(
            sklearn.metrics.roc_auc_score(
                onehot, probabilities, average="macro", multi_class="ovr"
            )
        ),
        "weighted_f1": float(
            sklearn.metrics.f1_score(truth, prediction, average="weighted")
        ),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
    }


def metrics_from_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [
                float(row["final_probability_AD"]),
                float(row["final_probability_CN"]),
                float(row["final_probability_SMCI"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return probability_metrics(truth, probabilities)


def load_reference_predictions() -> dict[int, dict]:
    text = subprocess.check_output(
        ["git", "show", f"{REFERENCE_COMMIT}:{REFERENCE_PATH}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
    )
    rows = list(csv.DictReader(io.StringIO(text)))
    require(len(rows) == EXPECTED_SUBJECTS, "Historical Original reference row count changed")
    result = {
        int(row["subject_index"]): {
            "truth": int(row["truth"]),
            "prediction": int(row["other_prediction"]),
            "correct": str(row["other_correct"]).lower() == "true",
        }
        for row in rows
    }
    require(len(result) == EXPECTED_SUBJECTS, "Historical Original subjects are not unique")
    require(sum(value["correct"] for value in result.values()) == ORIGINAL["correct"], "Historical Original correct count changed")
    return result


def load_context():
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    branch = git_value("branch", "--show-current")
    require(
        branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-rerun"),
        f"Unexpected branch: {branch}",
    )
    device = torch.device("cuda:0")
    config = Config_(str(ROOT), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(int(config.T_max) == EPOCHS, "Historical T_max changed")
    SET_Random(SEED)
    feature_path, dict_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    require(tuple(class_names) == CLASS_NAMES, f"Unexpected class order: {class_names}")
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    require(int(dataset_data["Feature"].shape[0]) == EXPECTED_SUBJECTS, "Subject count changed")
    require(int(dataset_data["Feature"].shape[1]) == 360, "Feature count changed")
    require(len(dataset_dict["Modal_Name"]) == 6, "Modal count changed")
    return config, dataset_dict, dataset_data, device, branch


def build_objects(config, dataset_dict: dict, device: torch.device):
    SET_Random(SEED)
    model = query.build_model(config, dataset_dict, "original", device)
    require(parameter_count(model) == EXPECTED_ORIGINAL_PARAMETERS, "Original parameter count changed")

    # Refiner initialization is deterministic but does not consume the RNG
    # stream used by Original dropout/noise during training.
    cuda_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[cuda_index]):
        torch.manual_seed(104_729)
        torch.cuda.manual_seed_all(104_729)
        refiner = UncertaintyGuidedLocalEvidenceRefinement(
            hidden_size=int(config.Hidden_size),
            projection_size=DEFAULT.projection_size,
        ).to(device)
    require(parameter_count(refiner) == EXPECTED_LOCAL_PARAMETERS, "Local parameter count changed")
    require(
        parameter_count(model) + parameter_count(refiner) == EXPECTED_TOTAL_PARAMETERS,
        "Total parameter count changed",
    )
    criterion = criterion_query_pool_no_orth(
        dataset_dict, device, label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()), "name": "original"},
            {"params": list(refiner.parameters()), "name": "um_ler"},
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=config.Lr_Min
    )
    return model, refiner, criterion, optimizer, scheduler


def original_probability(
    raw_logits: torch.Tensor, label_weight: torch.Tensor, logit_adjust_tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = raw_logits - float(logit_adjust_tau) * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    return adjusted, torch.softmax(adjusted.detach(), dim=-1)


def local_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_labels = labels[train_mask]
    train_counts = torch.bincount(train_labels, minlength=3).to(
        dtype=outputs["p_final"].dtype
    )
    train_size = train_counts.sum()
    # Match the historical (N-count_c)/N weighting formula, but derive the
    # new refinement loss weights strictly from the current training fold.
    refine_class_weight = (train_size - train_counts) / train_size
    nca = F.nll_loss(
        outputs["q"][train_mask].clamp_min(1e-12).log(),
        train_labels,
    )
    refine = F.nll_loss(
        outputs["p_final"][train_mask].clamp_min(1e-12).log(),
        train_labels,
        weight=refine_class_weight,
    )
    return nca, refine


def forward_all(
    model,
    refiner,
    criterion,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    dataset_dict: dict,
    historical_config,
    run_config: UMLERConfig,
) -> dict:
    raw_logits, representations, auxiliary = model(features)
    require(model.last_modal_tokens is not None, "Modal token cache missing")
    adjusted_logits, p_original = original_probability(
        raw_logits,
        dataset_dict["Label_Weight"],
        float(historical_config.logit_adjust_tau),
    )
    local = refiner(
        model.last_modal_tokens,
        p_original,
        labels,
        train_mask,
        top_k=run_config.top_k,
        temperature=run_config.temperature,
        margin_threshold=run_config.margin_threshold,
        neighbor_confidence_threshold=run_config.neighbor_confidence_threshold,
        gate_cap=run_config.gate_cap,
    )
    original_loss = criterion(
        raw_logits, labels, train_mask, representations, auxiliary
    )
    nca_loss, refine_loss = local_losses(
        local, labels, train_mask
    )
    total_loss = (
        original_loss
        + float(run_config.nca_weight) * nca_loss
        + float(run_config.refine_weight) * refine_loss
    )
    return {
        "raw_logits": raw_logits,
        "adjusted_logits": adjusted_logits,
        "representations": representations,
        "auxiliary": auxiliary,
        "original_loss": original_loss,
        "nca_loss": nca_loss,
        "refine_loss": refine_loss,
        "total_loss": total_loss,
        **local,
    }


def clip_gradients(model, refiner, grad_clip: float) -> None:
    if grad_clip > 0:
        # Separate clipping is essential: local gradients cannot rescale the
        # Original gradient norm and therefore cannot alter its trajectory.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), grad_clip)


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def probability_gate_checks(outputs: dict, run_config: UMLERConfig) -> dict:
    sums = {
        name: float((outputs[name].sum(dim=-1) - 1.0).abs().max().detach().cpu())
        for name in ("p", "q", "p_final")
    }
    modal_sum_error = float(
        (outputs["modal_reliability"].sum(dim=-1) - 1.0).abs().max().detach().cpu()
    )
    retrieval_norm_error = float(
        (outputs["retrieval"].norm(dim=-1) - 1.0).abs().max().detach().cpu()
    )
    gate_min = float(outputs["gate"].min().detach().cpu())
    gate_max = float(outputs["gate"].max().detach().cpu())
    passed = (
        max(sums.values()) <= 1e-6
        and modal_sum_error <= 1e-6
        and retrieval_norm_error <= 1e-5
        and gate_min >= -1e-8
        and gate_max <= run_config.gate_cap + 1e-6
    )
    require(passed, "Probability/retrieval/gate invariant failed")
    return {
        "probability_sum_max_abs_error": sums,
        "modal_weight_sum_max_abs_error": modal_sum_error,
        "retrieval_norm_max_abs_error": retrieval_norm_error,
        "gate_min": gate_min,
        "gate_max": gate_max,
        "passed": passed,
    }


def neighbor_checks(
    outputs: dict, labels: torch.Tensor, train_mask: torch.Tensor
) -> dict:
    train_indices = outputs["train_indices"]
    neighbor_indices = outputs["neighbor_indices"]
    test_mask = ~train_mask
    all_memory_from_train = bool(train_mask[neighbor_indices].all())
    train_neighbors = neighbor_indices.index_select(0, train_indices)
    no_train_self = bool(
        (train_neighbors != train_indices.view(-1, 1)).all()
    )
    test_neighbors = neighbor_indices[test_mask]
    test_neighbors_from_train = bool(train_mask[test_neighbors].all())
    train_library_exact = bool(
        torch.equal(
            outputs["train_labels"],
            labels.index_select(0, train_indices),
        )
        and outputs["train_labels"].numel() == int(train_mask.sum().item())
    )
    passed = (
        all_memory_from_train
        and no_train_self
        and test_neighbors_from_train
        and train_library_exact
    )
    require(passed, "Train-only/leave-one-out neighbor invariant failed")
    return {
        "all_memory_from_train": all_memory_from_train,
        "no_train_self_neighbor": no_train_self,
        "test_neighbors_from_train": test_neighbors_from_train,
        "train_label_library_size": int(outputs["train_labels"].numel()),
        "test_label_library_entries": 0,
        "passed": passed,
    }


def smoke_gradient_isolation(
    outputs: dict,
    model: torch.nn.Module,
    refiner: torch.nn.Module,
    run_config: UMLERConfig,
) -> dict:
    model_parameters = list(model.parameters())
    local_parameters = list(refiner.parameters())
    local_loss = (
        float(run_config.nca_weight) * outputs["nca_loss"]
        + float(run_config.refine_weight) * outputs["refine_loss"]
    )
    local_on_original = torch.autograd.grad(
        local_loss,
        model_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    original_on_local = torch.autograd.grad(
        outputs["original_loss"],
        local_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    local_gradients = torch.autograd.grad(
        local_loss,
        local_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    local_gradient_by_name = {
        name: gradient
        for (name, _), gradient in zip(refiner.named_parameters(), local_gradients)
    }
    required_nonzero = (
        "modal_norm.weight",
        "modal_projector.weight",
        "modal_reliability.weight",
    )
    require(all(gradient is None for gradient in local_on_original), "Refinement loss reaches Original")
    require(all(gradient is None for gradient in original_on_local), "Original loss reaches refiner")
    require(
        all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in local_gradients
        ),
        "Local gradient missing or non-finite",
    )
    norms = {
        name: float(gradient.detach().float().norm().cpu())
        for name, gradient in local_gradient_by_name.items()
    }
    require(
        all(norms[name] > 0.0 for name in required_nonzero),
        f"Required local gradient is zero: {norms}",
    )
    return {
        "refinement_gradients_on_original": 0,
        "original_gradients_on_refiner": 0,
        "local_gradient_norms": norms,
        "reliability_bias_zero_is_expected": True,
        "passed": True,
    }


def run_smoke() -> dict:
    config, dataset_dict, dataset_data, device, branch = load_context()
    output_dir = ROOT / SMOKE_REL
    require(not output_dir.exists(), f"Smoke output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model, refiner, criterion, optimizer, scheduler = build_objects(
        config, dataset_dict, device
    )
    train_mask, _ = dataset_data["Mask"][0]
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    epoch_rows = []
    isolation = None
    neighbors = None
    probability_checks = None
    started = time.perf_counter()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        model.train()
        refiner.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_all(
            model,
            refiner,
            criterion,
            features,
            labels,
            train_mask,
            dataset_dict,
            config,
            DEFAULT,
        )
        require(
            all(
                bool(torch.isfinite(outputs[name]))
                for name in ("original_loss", "nca_loss", "refine_loss", "total_loss")
            ),
            "Smoke loss contains NaN/Inf",
        )
        if epoch == 1:
            isolation = smoke_gradient_isolation(outputs, model, refiner, DEFAULT)
            neighbors = neighbor_checks(outputs, labels, train_mask)
            probability_checks = probability_gate_checks(outputs, DEFAULT)
        outputs["total_loss"].backward()
        require(gradients_finite(model), "Original gradient contains NaN/Inf")
        require(gradients_finite(refiner), "Local gradient contains NaN/Inf")
        clip_gradients(model, refiner, float(config.grad_clip))
        optimizer.step()
        scheduler.step()
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss_original": float(outputs["original_loss"].detach().cpu()),
                "loss_nca": float(outputs["nca_loss"].detach().cpu()),
                "loss_refine": float(outputs["refine_loss"].detach().cpu()),
                "loss_total": float(outputs["total_loss"].detach().cpu()),
            }
        )
    summary = {
        "model": MODEL_NAME,
        "branch": branch,
        "base_commit": BASE_COMMIT,
        "fold": 0,
        "seed": SEED,
        "epochs": SMOKE_EPOCHS,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "original_parameters": EXPECTED_ORIGINAL_PARAMETERS,
        "local_parameters": EXPECTED_LOCAL_PARAMETERS,
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "config": asdict(DEFAULT),
        "gradient_isolation": isolation,
        "neighbor_checks": neighbors,
        "probability_checks": probability_checks,
        "epoch_metrics": epoch_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", asdict(DEFAULT))
    print(
        f"smoke fold=0 epochs=2 passed=True total_parameters={EXPECTED_TOTAL_PARAMETERS}",
        flush=True,
    )
    return summary


def tensor_metrics(probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict:
    return probability_metrics(
        labels[mask].detach().cpu().numpy(),
        probabilities[mask].detach().cpu().numpy(),
    )


def prediction_rows(
    fold: int,
    outputs: dict,
    labels: torch.Tensor,
    test_mask: torch.Tensor,
    dataset_dict: dict,
) -> list[dict]:
    indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        test_mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
    original_probability = outputs["p"][test_mask].detach().cpu().numpy()
    local_probability = outputs["q"][test_mask].detach().cpu().numpy()
    final_probability = outputs["p_final"][test_mask].detach().cpu().numpy()
    original_prediction = original_probability.argmax(axis=1)
    final_prediction = final_probability.argmax(axis=1)
    gate = outputs["gate"][test_mask].detach().cpu().numpy()
    margin = outputs["margin"][test_mask].detach().cpu().numpy()
    uncertainty = outputs["uncertainty"][test_mask].detach().cpu().numpy()
    reliability = outputs["local_reliability"][test_mask].detach().cpu().numpy()
    local_max = outputs["local_max"][test_mask].detach().cpu().numpy()
    neighbor_similarity = outputs["neighbor_similarities"][test_mask].detach().cpu().numpy()
    rows = []
    for row_index, subject_index in enumerate(indices):
        row = {
            "fold": fold,
            "subject_index": int(subject_index),
            "truth": int(truth[row_index]),
            "model_original_prediction": int(original_prediction[row_index]),
            "prediction": int(final_prediction[row_index]),
            "gate": float(gate[row_index]),
            "margin": float(margin[row_index]),
            "uncertainty": float(uncertainty[row_index]),
            "local_reliability": float(reliability[row_index]),
            "local_max_probability": float(local_max[row_index]),
            "top_neighbor_similarity": float(neighbor_similarity[row_index, 0]),
            "mean_topk_similarity": float(neighbor_similarity[row_index].mean()),
        }
        for label_index, class_name in enumerate(CLASS_NAMES):
            row[f"original_probability_{class_name}"] = float(
                original_probability[row_index, label_index]
            )
            row[f"local_probability_{class_name}"] = float(
                local_probability[row_index, label_index]
            )
            row[f"final_probability_{class_name}"] = float(
                final_probability[row_index, label_index]
            )
        rows.append(row)
    return rows


def load_completed_fold(
    version_name: str,
    fold: int,
    fold_dir: Path,
    run_config: UMLERConfig,
) -> tuple[dict, list[dict]] | None:
    if not fold_dir.exists():
        return None
    required = (
        fold_dir / "summary.json",
        fold_dir / "epoch_metrics.csv",
        fold_dir / "best_predictions.csv",
        fold_dir / "checkpoint_best.pt",
    )
    require(all(path.is_file() for path in required), f"Incomplete completed fold: {fold_dir}")
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("passed") is True, f"Fold is not marked passed: {fold_dir}")
    require(summary.get("version") == version_name, "Completed fold version mismatch")
    require(int(summary.get("fold", -1)) == fold, "Completed fold id mismatch")
    require(summary.get("run_config") == asdict(run_config), "Completed fold config mismatch")
    require(len(read_csv(fold_dir / "epoch_metrics.csv")) == EPOCHS, "Completed fold epoch count mismatch")
    rows = read_csv(fold_dir / "best_predictions.csv")
    metrics = metrics_from_rows(rows)
    for name in METRIC_NAMES:
        require(
            abs(float(metrics[name]) - float(summary["best_metrics"][name])) <= 1e-10,
            f"Completed fold metric mismatch: {name}",
        )
    print(
        f"version={version_name} fold={fold} best_epoch={summary['best_epoch']} "
        f"correct={metrics['correct']}/{summary['test_size']} "
        f"ACC={metrics['acc']:.7f} Macro-F1={metrics['macro_f1']:.7f} "
        f"BACC={metrics['bacc']:.7f} Probability-Macro-AUC={metrics['macro_auc']:.7f}",
        flush=True,
    )
    return summary, rows


def run_fold(
    version_name: str,
    fold: int,
    run_config: UMLERConfig,
    output_root: Path,
    historical_config,
    dataset_dict: dict,
    dataset_data: dict,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    fold_dir = output_root / f"fold_{fold:02d}"
    completed = load_completed_fold(version_name, fold, fold_dir, run_config)
    if completed is not None:
        return completed
    train_mask, test_mask = dataset_data["Mask"][fold]
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    model, refiner, criterion, optimizer, scheduler = build_objects(
        historical_config, dataset_dict, device
    )
    best = None
    best_model_state = None
    best_refiner_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        refiner.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_all(
            model,
            refiner,
            criterion,
            features,
            labels,
            train_mask,
            dataset_dict,
            historical_config,
            run_config,
        )
        require(bool(torch.isfinite(outputs["total_loss"])), f"{version_name} fold{fold} epoch{epoch}: non-finite loss")
        outputs["total_loss"].backward()
        require(gradients_finite(model), f"{version_name} fold{fold} epoch{epoch}: invalid Original gradient")
        require(gradients_finite(refiner), f"{version_name} fold{fold} epoch{epoch}: invalid local gradient")
        clip_gradients(model, refiner, float(historical_config.grad_clip))
        optimizer.step()
        scheduler.step()

        model.eval()
        refiner.eval()
        with torch.no_grad():
            evaluated = forward_all(
                model,
                refiner,
                criterion,
                features,
                labels,
                train_mask,
                dataset_dict,
                historical_config,
                run_config,
            )
            probability_gate_checks(evaluated, run_config)
            metrics = tensor_metrics(evaluated["p_final"], labels, test_mask)
            train_agreement_weighted = float(
                evaluated["weighted_train_neighbor_agreement"].mean().cpu()
            )
            train_agreement_unweighted = float(
                evaluated["unweighted_train_neighbor_agreement"].mean().cpu()
            )
            test_gate_active_ratio = float(
                (evaluated["gate"][test_mask] > 1e-12).float().mean().cpu()
            )
        score = (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(metrics),
                "train_neighbor_agreement_weighted": train_agreement_weighted,
                "train_neighbor_agreement_unweighted": train_agreement_unweighted,
            }
            best_model_state = clone_cpu_state(model)
            best_refiner_state = clone_cpu_state(refiner)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_original": float(outputs["original_loss"].detach().cpu()),
                "loss_nca": float(outputs["nca_loss"].detach().cpu()),
                "loss_refine": float(outputs["refine_loss"].detach().cpu()),
                "loss_total": float(outputs["total_loss"].detach().cpu()),
                "acc": metrics["acc"],
                "macro_f1": metrics["macro_f1"],
                "bacc": metrics["bacc"],
                "probability_macro_auc": metrics["macro_auc"],
                "weighted_f1": metrics["weighted_f1"],
                "test_gate_active_ratio": test_gate_active_ratio,
                "train_neighbor_agreement_weighted": train_agreement_weighted,
                "train_neighbor_agreement_unweighted": train_agreement_unweighted,
            }
        )
    require(best is not None and best_model_state is not None and best_refiner_state is not None, "Best state missing")
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_model_state, strict=True)
    refiner.load_state_dict(best_refiner_state, strict=True)
    model.eval()
    refiner.eval()
    with torch.no_grad():
        best_outputs = forward_all(
            model,
            refiner,
            criterion,
            features,
            labels,
            train_mask,
            dataset_dict,
            historical_config,
            run_config,
        )
        neighbor_checks(best_outputs, labels, train_mask)
        probability_gate_checks(best_outputs, run_config)
        best_metrics = tensor_metrics(best_outputs["p_final"], labels, test_mask)
        rows = prediction_rows(
            fold, best_outputs, labels, test_mask, dataset_dict
        )
    for name in METRIC_NAMES:
        require(abs(float(best_metrics[name]) - float(best["metrics"][name])) <= 1e-10, f"Best reload mismatch: {name}")
    fold_dir.mkdir(parents=True)
    write_csv(fold_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(fold_dir / "best_predictions.csv", rows)
    checkpoint = {
        "fold": fold,
        "best_epoch": best["epoch"],
        "run_config": asdict(run_config),
        "original_model": best_model_state,
        "local_refiner": best_refiner_state,
    }
    torch.save(checkpoint, fold_dir / "checkpoint_best.pt")
    summary = {
        "version": version_name,
        "fold": fold,
        "best_epoch": best["epoch"],
        "best_metrics": best_metrics,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "split_hash": query.split_hash(dataset_dict["Index"], train_mask, test_mask),
        "run_config": asdict(run_config),
        "parameter_count": EXPECTED_TOTAL_PARAMETERS,
        "original_parameter_count": EXPECTED_ORIGINAL_PARAMETERS,
        "local_parameter_count": EXPECTED_LOCAL_PARAMETERS,
        "first_epoch_nca_loss": float(epoch_rows[0]["loss_nca"]),
        "final_epoch_nca_loss": float(epoch_rows[-1]["loss_nca"]),
        "best_train_neighbor_agreement_weighted": best["train_neighbor_agreement_weighted"],
        "best_train_neighbor_agreement_unweighted": best["train_neighbor_agreement_unweighted"],
        "elapsed_seconds": elapsed,
        "passed": True,
    }
    write_json(fold_dir / "summary.json", summary)
    print(
        f"version={version_name} fold={fold} best_epoch={best['epoch']} "
        f"correct={best_metrics['correct']}/{int(test_mask.sum())} "
        f"ACC={best_metrics['acc']:.7f} Macro-F1={best_metrics['macro_f1']:.7f} "
        f"BACC={best_metrics['bacc']:.7f} Probability-Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, refiner, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def row_diagnostics(rows: list[dict], reference: dict[int, dict]) -> dict:
    gates = np.asarray([float(row["gate"]) for row in rows], dtype=np.float64)
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    model_original = np.asarray(
        [int(row["model_original_prediction"]) for row in rows], dtype=np.int64
    )
    final_prediction = np.asarray(
        [int(row["prediction"]) for row in rows], dtype=np.int64
    )
    subject_indices = [int(row["subject_index"]) for row in rows]
    reference_prediction = np.asarray(
        [reference[index]["prediction"] for index in subject_indices], dtype=np.int64
    )
    reference_truth = np.asarray(
        [reference[index]["truth"] for index in subject_indices], dtype=np.int64
    )
    require(np.array_equal(truth, reference_truth), "Historical Original truth alignment failed")
    active = gates > 1e-12
    changed = model_original != final_prediction
    model_correct = model_original == truth
    final_correct = final_prediction == truth
    reference_correct = reference_prediction == truth
    repairs = changed & ~model_correct & final_correct
    damages = changed & model_correct & ~final_correct
    harmful_to_smci = (
        changed
        & ~final_correct
        & np.isin(truth, [0, 1])
        & (final_prediction == 2)
    )
    active_gates = gates[active]
    return {
        "model_original_correct_at_selected_epochs": int(model_correct.sum()),
        "final_correct": int(final_correct.sum()),
        "historical_original_correct": int(reference_correct.sum()),
        "historical_original_exclusive_correct": int((reference_correct & ~final_correct).sum()),
        "um_ler_exclusive_correct": int((~reference_correct & final_correct).sum()),
        "changed_prediction_count": int(changed.sum()),
        "original_wrong_to_new_correct": int(repairs.sum()),
        "original_correct_to_new_wrong": int(damages.sum()),
        "net_local_corrections": int(repairs.sum() - damages.sum()),
        "harmful_CN_or_AD_to_SMCI": int(harmful_to_smci.sum()),
        "gate_active_count": int(active.sum()),
        "gate_active_ratio": float(active.mean()),
        "active_gate_mean": float(active_gates.mean()) if len(active_gates) else 0.0,
        "active_gate_max": float(active_gates.max()) if len(active_gates) else 0.0,
        "overall_gate_mean": float(gates.mean()),
    }


def render_version_report(summary: dict) -> str:
    metrics = summary["oof_metrics"]
    diagnostics = summary["diagnostics"]
    return "\n".join(
        [
            f"# {MODEL_NAME} - {summary['version']}",
            "",
            f"- Correct: {metrics['correct']}/{EXPECTED_SUBJECTS}",
            f"- ACC: {metrics['acc']:.10f}",
            f"- Macro-F1: {metrics['macro_f1']:.10f}",
            f"- BACC: {metrics['bacc']:.10f}",
            f"- Probability Macro-AUC: {metrics['macro_auc']:.10f}",
            f"- Weighted-F1: {metrics['weighted_f1']:.10f}",
            f"- Confusion matrix: {metrics['confusion_matrix']}",
            f"- Changed predictions: {diagnostics['changed_prediction_count']}",
            f"- Repairs / damages: {diagnostics['original_wrong_to_new_correct']} / {diagnostics['original_correct_to_new_wrong']}",
            f"- Gate active: {diagnostics['gate_active_count']}/{EXPECTED_SUBJECTS} ({diagnostics['gate_active_ratio']:.6f})",
            f"- Active mean/max g: {diagnostics['active_gate_mean']:.6f} / {diagnostics['active_gate_max']:.6f}",
            f"- Runtime seconds: {summary['elapsed_seconds']:.3f}",
            "",
        ]
    )


def run_version(
    version_name: str,
    run_config: UMLERConfig,
    historical_config,
    dataset_dict: dict,
    dataset_data: dict,
    device: torch.device,
    reference: dict[int, dict],
) -> dict:
    output_root = ROOT / RESULTS_ROOT / version_name
    if output_root.exists():
        config_path = output_root / "config.json"
        require(config_path.is_file(), f"Existing version lacks config: {output_root}")
        require(
            json.loads(config_path.read_text(encoding="utf-8")) == asdict(run_config),
            f"Existing version config mismatch: {output_root}",
        )
    else:
        output_root.mkdir(parents=True)
        write_json(output_root / "config.json", asdict(run_config))
    started = time.perf_counter()
    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        fold_summary, rows = run_fold(
            version_name,
            fold,
            run_config,
            output_root,
            historical_config,
            dataset_dict,
            dataset_data,
            device,
        )
        fold_summaries.append(fold_summary)
        oof_rows.extend(rows)
    require(len(oof_rows) == EXPECTED_SUBJECTS, "OOF row count mismatch")
    require(
        len({int(row["subject_index"]) for row in oof_rows}) == EXPECTED_SUBJECTS,
        "OOF subjects are not unique",
    )
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    for row in oof_rows:
        ref = reference[int(row["subject_index"])]
        row["historical_original_prediction"] = ref["prediction"]
        row["historical_original_correct"] = ref["correct"]
    metrics = metrics_from_rows(oof_rows)
    diagnostics = row_diagnostics(oof_rows, reference)
    fold_rows = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            **{name: summary["best_metrics"][name] for name in METRIC_NAMES},
            "correct": summary["best_metrics"]["correct"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for summary in fold_summaries
    ]
    nca_start = float(np.mean([summary["first_epoch_nca_loss"] for summary in fold_summaries]))
    nca_end = float(np.mean([summary["final_epoch_nca_loss"] for summary in fold_summaries]))
    agreement_weighted = float(
        np.mean(
            [
                summary["best_train_neighbor_agreement_weighted"]
                for summary in fold_summaries
            ]
        )
    )
    agreement_unweighted = float(
        np.mean(
            [
                summary["best_train_neighbor_agreement_unweighted"]
                for summary in fold_summaries
            ]
        )
    )
    delta = {
        "correct": metrics["correct"] - ORIGINAL["correct"],
        **{
            name: metrics[name] - ORIGINAL[name]
            for name in METRIC_NAMES
        },
    }
    summary = {
        "model": MODEL_NAME,
        "version": version_name,
        "config": asdict(run_config),
        "parameter_count": EXPECTED_TOTAL_PARAMETERS,
        "additional_parameters": EXPECTED_LOCAL_PARAMETERS,
        "oof_metrics": metrics,
        "delta_vs_original": delta,
        "original_reference": ORIGINAL,
        "diagnostics": diagnostics,
        "fold_metrics": fold_rows,
        "fold_acc_mean": float(np.mean([row["acc"] for row in fold_rows])),
        "fold_acc_sample_std": float(np.std([row["acc"] for row in fold_rows], ddof=1)),
        "mean_nca_loss_epoch1": nca_start,
        "mean_nca_loss_epoch400": nca_end,
        "nca_relative_decline": float((nca_start - nca_end) / max(nca_start, 1e-12)),
        "train_neighbor_agreement_weighted": agreement_weighted,
        "train_neighbor_agreement_unweighted": agreement_unweighted,
        "elapsed_seconds": time.perf_counter() - started,
        "source_commit": git_value("rev-parse", "HEAD"),
    }
    write_csv(output_root / "oof_predictions.csv", oof_rows)
    write_csv(output_root / "fold_metrics.csv", fold_rows)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "report.json", summary)
    write_csv(
        output_root / "confusion_matrix.csv",
        [
            {
                "actual": CLASS_NAMES[row_index],
                **{
                    f"predicted_{CLASS_NAMES[column_index]}": int(
                        metrics["confusion_matrix"][row_index][column_index]
                    )
                    for column_index in range(3)
                },
            }
            for row_index in range(3)
        ],
    )
    (output_root / "report.md").write_text(
        render_version_report(summary), encoding="utf-8"
    )
    reloaded_rows = read_csv(output_root / "oof_predictions.csv")
    require(metrics_from_rows(reloaded_rows) == metrics, "OOF readback recomputation failed")
    return summary


def single_change(config: UMLERConfig) -> dict:
    default = asdict(DEFAULT)
    current = asdict(config)
    return {
        key: current[key]
        for key in current
        if current[key] != default[key]
    }


def config_delta(left: UMLERConfig, right: UMLERConfig) -> dict:
    left_values = asdict(left)
    right_values = asdict(right)
    return {
        key: {"from": left_values[key], "to": right_values[key]}
        for key in left_values
        if left_values[key] != right_values[key]
    }


def candidate_from_current(
    current: UMLERConfig, field: str, value
) -> UMLERConfig:
    candidate = replace(current, **{field: value})
    require(
        len(config_delta(current, candidate)) == 1,
        "Adjustment is not single-variable relative to the current version",
    )
    return candidate


def choose_adjustment(history: list[dict], tried: set[tuple[str, float]]) -> tuple[UMLERConfig, str] | None:
    current = history[-1]
    current_config = UMLERConfig(**current["config"])
    metrics = current["oof_metrics"]
    diagnostics = current["diagnostics"]
    if metrics["correct"] >= 557:
        return None
    if len(history) >= 4:
        return None
    if len(history) >= 3:
        first_no_gain = history[-2]["oof_metrics"]["correct"] <= history[-3]["oof_metrics"]["correct"]
        second_no_gain = history[-1]["oof_metrics"]["correct"] <= history[-2]["oof_metrics"]["correct"]
        if first_no_gain and second_no_gain:
            return None

    over_aggressive = (
        diagnostics["original_correct_to_new_wrong"]
        >= diagnostics["original_wrong_to_new_correct"] + 2
        or diagnostics["gate_active_ratio"] > 0.20
        or metrics["acc"] < ORIGINAL["acc"] - 0.005
    )
    under_active = (
        diagnostics["gate_active_ratio"] < 0.02
        or diagnostics["changed_prediction_count"] < 3
        or diagnostics["original_wrong_to_new_correct"] == 0
    )
    if over_aggressive:
        for field, value, reason in (
            ("gate_cap", 0.20, "over-aggressive: lower gate cap"),
            (
                "neighbor_confidence_threshold",
                0.75,
                "over-aggressive after cap: tighten neighbor confidence",
            ),
        ):
            key = (field, float(value))
            if key not in tried:
                return candidate_from_current(current_config, field, value), reason
        return None
    if under_active:
        for field, value, reason in (
            (
                "neighbor_confidence_threshold",
                0.65,
                "under-active: lower neighbor confidence",
            ),
            ("gate_cap", 0.40, "under-active after confidence: raise gate cap"),
        ):
            key = (field, float(value))
            if key not in tried:
                return candidate_from_current(current_config, field, value), reason
        return None

    direction_unstable = (
        0.02 <= diagnostics["gate_active_ratio"] <= 0.20
        and diagnostics["changed_prediction_count"] >= 3
        and (
            abs(
                diagnostics["original_wrong_to_new_correct"]
                - diagnostics["original_correct_to_new_wrong"]
            )
            <= 1
            or diagnostics["harmful_CN_or_AD_to_SMCI"] > 0
        )
    )
    if direction_unstable:
        if not any(field == "top_k" for field, _ in tried):
            if current["train_neighbor_agreement_unweighted"] < 0.65:
                field, value, reason = "top_k", 5, "mixed neighborhoods: increase locality"
            else:
                field, value, reason = "top_k", 12, "unstable neighborhoods: increase stability"
            key = (field, float(value))
            if key not in tried:
                return candidate_from_current(current_config, field, value), reason

    retrieval_underfit = (
        0.02 <= diagnostics["gate_active_ratio"] <= 0.20
        and current["train_neighbor_agreement_unweighted"] < 0.65
        and current["nca_relative_decline"] < 0.10
        and len(history) >= 2
    )
    if retrieval_underfit:
        key = ("nca_weight", 0.40)
        if key not in tried:
            return candidate_from_current(
                current_config, "nca_weight", 0.40
            ), "retrieval underfit: raise NCA weight"
    return None


def decision(metrics: dict) -> str:
    correct = int(metrics["correct"])
    guard = not (
        metrics["macro_f1"] < ORIGINAL["macro_f1"] - 0.005
        and metrics["macro_auc"] < ORIGINAL["macro_auc"] - 0.005
    )
    if correct <= 554:
        return "STOP_ROUTE"
    if correct <= 556:
        return "NO_ACCURACY_GAIN"
    if not guard:
        return "TARGET_WITH_METRIC_GUARD_FAILURE"
    if correct == 557:
        return "KEEP_MINIMUM_TARGET"
    if correct <= 559:
        return "SUCCESS"
    return "MAIN_MODEL_CANDIDATE"


def render_final_report(payload: dict) -> str:
    selected = payload["selected_result"]
    metrics = selected["oof_metrics"]
    diagnostics = selected["diagnostics"]
    lines = [
        f"# {MODEL_NAME} - Final Report",
        "",
        f"**Decision: {payload['decision']}**",
        "",
        f"- Branch: `{payload['branch']}`",
        f"- Source commit: `{payload['source_commit']}`",
        f"- Device: {payload['environment']['gpu']} ({payload['environment']['device']})",
        f"- Total runtime: {payload['total_elapsed_seconds']:.3f} seconds",
        f"- Selected version: `{selected['version']}`",
        f"- Parameters: {selected['parameter_count']} (+{selected['additional_parameters']} vs Original)",
        f"- Correct: {metrics['correct']}/{EXPECTED_SUBJECTS}",
        f"- ACC: {metrics['acc']:.10f} ({selected['delta_vs_original']['acc']:+.10f})",
        f"- Macro-F1: {metrics['macro_f1']:.10f} ({selected['delta_vs_original']['macro_f1']:+.10f})",
        f"- BACC: {metrics['bacc']:.10f} ({selected['delta_vs_original']['bacc']:+.10f})",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.10f} ({selected['delta_vs_original']['macro_auc']:+.10f})",
        f"- Weighted-F1: {metrics['weighted_f1']:.10f} ({selected['delta_vs_original']['weighted_f1']:+.10f})",
        f"- OOF confusion matrix: {metrics['confusion_matrix']}",
        f"- Historical Original exclusive correct: {diagnostics['historical_original_exclusive_correct']}",
        f"- UM-LER exclusive correct: {diagnostics['um_ler_exclusive_correct']}",
        f"- Actual changed predictions: {diagnostics['changed_prediction_count']}",
        f"- Original wrong -> new correct: {diagnostics['original_wrong_to_new_correct']}",
        f"- Original correct -> new wrong: {diagnostics['original_correct_to_new_wrong']}",
        f"- Gate activation: {diagnostics['gate_active_count']}/{EXPECTED_SUBJECTS} ({diagnostics['gate_active_ratio']:.6f})",
        f"- Active mean/max g: {diagnostics['active_gate_mean']:.6f}/{diagnostics['active_gate_max']:.6f}",
        "",
        "## Per-fold best epoch and ACC",
        "",
        "| Fold | Best epoch | ACC |",
        "|---:|---:|---:|",
        *[
            f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.10f} |"
            for row in selected["fold_metrics"]
        ],
        "",
        "## Attempted versions",
        "",
        "| Version | Correct | ACC | Changes vs default |",
        "|---|---:|---:|---|",
        *[
            f"| {result['version']} | {result['oof_metrics']['correct']} | {result['oof_metrics']['acc']:.10f} | {single_change(UMLERConfig(**result['config'])) or 'default'} |"
            for result in payload["attempted_results"]
        ],
        "",
    ]
    return "\n".join(lines)


def run_formal_all() -> dict:
    historical_config, dataset_dict, dataset_data, device, branch = load_context()
    reference = load_reference_predictions()
    require(not (ROOT / FINAL_JSON_REL).exists(), "Final report already exists")
    suite_started = time.perf_counter()
    history = []
    tuning_log = []
    tried: set[tuple[str, float]] = set()
    run_config = DEFAULT
    for version_index in range(4):
        version_name = "um_ler_v1" if version_index == 0 else f"um_ler_v1_{version_index}"
        result = run_version(
            version_name,
            run_config,
            historical_config,
            dataset_dict,
            dataset_data,
            device,
            reference,
        )
        history.append(result)
        if result["oof_metrics"]["correct"] >= 557:
            break
        adjustment = choose_adjustment(history, tried)
        if adjustment is None:
            break
        next_config, reason = adjustment
        adjacent_change = config_delta(run_config, next_config)
        field, values = next(iter(adjacent_change.items()))
        tried.add((field, float(values["to"])))
        tuning_log.append(
            {
                "after_version": version_name,
                "reason": reason,
                "next_single_change": adjacent_change,
            }
        )
        run_config = next_config

    selected = max(
        history,
        key=lambda result: (
            result["oof_metrics"]["correct"],
            result["oof_metrics"]["acc"],
            result["oof_metrics"]["macro_auc"],
            result["oof_metrics"]["macro_f1"],
        ),
    )
    payload = {
        "model": MODEL_NAME,
        "decision": decision(selected["oof_metrics"]),
        "selected_result": selected,
        "attempted_results": history,
        "tuning_log": tuning_log,
        "branch": branch,
        "source_commit": git_value("rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "total_elapsed_seconds": time.perf_counter() - suite_started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
        },
        "protocol": {
            "dataset": historical_config.DATA_SET,
            "task": historical_config.Task,
            "folds": list(FOLDS),
            "seed": SEED,
            "epochs": EPOCHS,
            "transductive_full_batch": True,
            "single_model": True,
            "ensemble": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR(T_max=400)",
            "best_epoch_rule": ["ACC", "probability Macro-AUC", "Macro-F1"],
            "original_loss_unchanged": True,
            "orthogonality": False,
        },
    }
    write_json(ROOT / FINAL_JSON_REL, payload)
    (ROOT / FINAL_MD_REL).write_text(
        render_final_report(payload), encoding="utf-8"
    )
    reloaded = json.loads((ROOT / FINAL_JSON_REL).read_text(encoding="utf-8"))
    require(
        reloaded["selected_result"]["oof_metrics"] == selected["oof_metrics"],
        "Final JSON readback failed",
    )
    print(
        f"final decision={payload['decision']} selected={selected['version']} "
        f"correct={selected['oof_metrics']['correct']}/{EXPECTED_SUBJECTS} "
        f"ACC={selected['oof_metrics']['acc']:.7f}",
        flush=True,
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal-all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_formal_all()


if __name__ == "__main__":
    main()
