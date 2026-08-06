from __future__ import annotations

import argparse
import csv
import json
import math
import platform
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

from Model.component_localization import (
    LinearProbeHead,
    PEEHead,
    RawFeatureMLP,
    ResidualMLPHead,
    SparseResidualGCNHead,
    build_mutual_knn_graph,
    pee_pairwise_diagnostics,
)
from Model.models import DIFFormer_GraphHead
from Loss import criterion_query_pool_no_orth
from Utils import Config_, CustomCosineAnnealingLR, SET_Random, load_dataset, load_path
import run_query_free_multibranch as query
import run_um_ler_v1 as um_ler


MODEL_NAME = "Multimodal / Graph / Classifier Component Localization v1"
BRANCH_PREFIX = "experiment/component-localization-v1"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
RESULTS_REL = Path("results/component_localization_v1")
ANCHOR_REL = Path("experiments/query_baseline_gpu_10fold_seed0")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 2
CLASS_NAMES = ("AD", "CN", "SMCI")
EXPECTED_SUBJECTS = 598
EXPECTED_FEATURES = 360
EXPECTED_HIDDEN = 96
EXPECTED_ORIGINAL_PARAMETERS = 853_131
EXPECTED_ORIGINAL = {
    "correct": 556,
    "acc": 0.9297658862876255,
    "macro_f1": 0.9140778251271361,
    "bacc": 0.9140778251271361,
    "macro_auc": 0.9560490698,
    "weighted_f1": 0.9297658862876255,
    "confusion_matrix": [[62, 0, 10], [0, 198, 11], [10, 11, 296]],
}
ANCHOR_TOLERANCE = 5e-6
METRIC_NAMES = ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
HEAD_NAMES = ("linear_probe", "residual_mlp", "frozen_difformer", "pee_head")
MODALITY_ALIASES = {
    "MRI": "MRI",
    "PET": "PET",
    "CSF": "CSF",
    "RISK_FACTOR": "Risk",
    "COGNITIVE_TEST": "COG",
    "ROI_AVERAGE": "ROI",
}
_VALIDATED_CHECKPOINT_PATHS: set[str] = set()


class IdentityGraphHead(torch.nn.Module):
    """Expose the pre-head embedding without retaining the replaced classifier."""

    def forward(self, features: torch.Tensor, adjacency=None) -> torch.Tensor:
        return features


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


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def repository_root() -> Path:
    common = Path(git_value("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (ROOT / common).resolve()
    return common.parent


def anchor_root() -> Path:
    root = repository_root() / ANCHOR_REL
    if root.is_dir():
        return root
    fallback = ROOT.parents[1] / ANCHOR_REL
    require(fallback.is_dir(), f"Historical Original anchor missing: {root}")
    return fallback


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def score_logits(
    raw_logits: torch.Tensor,
    label_weight: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = raw_logits - float(tau) * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    return adjusted, torch.softmax(adjusted, dim=-1)


def metrics_from_probability(truth, probability) -> dict:
    return um_ler.probability_metrics(
        np.asarray(truth, dtype=np.int64),
        np.asarray(probability, dtype=np.float64),
    )


def metrics_from_rows(rows: list[dict], probability_prefix: str = "probability") -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probability = np.asarray(
        [
            [float(row[f"{probability_prefix}_{name}"]) for name in CLASS_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    return metrics_from_probability(truth, probability)


def selection_tuple(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["acc"]),
        float(metrics["macro_auc"]),
        float(metrics["macro_f1"]),
    )


def confusion_rows(metrics: dict) -> list[dict]:
    matrix = metrics["confusion_matrix"]
    return [
        {
            "actual": CLASS_NAMES[row],
            **{
                f"predicted_{CLASS_NAMES[column]}": int(matrix[row][column])
                for column in range(3)
            },
        }
        for row in range(3)
    ]


def environment_snapshot(device: torch.device) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }


def load_context() -> dict:
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    branch = git_value("branch", "--show-current")
    require(
        branch == BRANCH_PREFIX or branch.startswith(BRANCH_PREFIX + "-rerun"),
        f"Unexpected branch: {branch}",
    )
    device = torch.device("cuda:0")
    config = Config_(str(ROOT), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(int(config.T_max) == EPOCHS, "Historical scheduler T_max changed")
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
    require(
        tuple(dataset_data["Feature"].shape) == (EXPECTED_SUBJECTS, EXPECTED_FEATURES),
        "Dataset shape changed",
    )
    require(len(dataset_data["Mask"]) == 10, "Fold count changed")
    require(len(dataset_dict["Modal_Name"]) == 6, "Modality count changed")
    source = anchor_root()
    for name in ("oof_predictions.csv", "fold_manifest.json", "protocol_manifest.json"):
        require((source / name).is_file(), f"Historical anchor file missing: {name}")
    anchor_rows = read_csv(source / "oof_predictions.csv")
    require(len(anchor_rows) == EXPECTED_SUBJECTS, "Anchor OOF row count changed")
    require(
        len({int(row["subject_index"]) for row in anchor_rows}) == EXPECTED_SUBJECTS,
        "Anchor subject indices are not unique",
    )
    return {
        "branch": branch,
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "anchor_root": source,
        "anchor_rows": anchor_rows,
        "anchor_order": {
            int(row["subject_index"]): position
            for position, row in enumerate(anchor_rows)
        },
    }


def verify_anchor_source(context: dict) -> dict:
    rows = context["anchor_rows"]
    metrics = metrics_from_rows(rows)
    require(metrics["correct"] == EXPECTED_ORIGINAL["correct"], "STOP_ORIGINAL_ANCHOR_MISMATCH")
    require(metrics["confusion_matrix"] == EXPECTED_ORIGINAL["confusion_matrix"], "STOP_ORIGINAL_ANCHOR_MISMATCH")
    for name in METRIC_NAMES:
        require(
            abs(float(metrics[name]) - float(EXPECTED_ORIGINAL[name])) <= 5e-7,
            f"STOP_ORIGINAL_ANCHOR_MISMATCH: {name}",
        )
    payload = {
        "status": "Original anchor reproduced",
        "metrics": metrics,
        "source": str(context["anchor_root"]),
        "oof_order": "historical Frozen UM-LER v2 order",
    }
    write_json(ROOT / RESULTS_REL / "original_anchor.json", payload)
    return payload


def ordered_rows(rows: list[dict], context: dict) -> list[dict]:
    order = context["anchor_order"]
    require(len(rows) == EXPECTED_SUBJECTS, "OOF row count mismatch")
    require(
        len({int(row["subject_index"]) for row in rows}) == EXPECTED_SUBJECTS,
        "OOF subject indices are not unique",
    )
    return sorted(rows, key=lambda row: order[int(row["subject_index"])])


def build_original(context: dict, fold: int):
    SET_Random(SEED)
    model = query.build_model(
        context["config"], context["dataset_dict"], "original", context["device"]
    )
    require(parameter_count(model) == EXPECTED_ORIGINAL_PARAMETERS, "Original parameter count changed")
    fold_root = context["anchor_root"] / f"fold_{fold:02d}"
    checkpoint = fold_root / "checkpoint_best.pt"
    summary_path = fold_root / "summary.json"
    require(checkpoint.is_file() and summary_path.is_file(), f"Historical fold {fold} incomplete")
    state = torch_load(checkpoint, "cpu")
    require(isinstance(state, dict) and len(state) == 160, "Historical checkpoint format changed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return model, int(summary["result"]["best_epoch"]), checkpoint


def cache_path(fold: int) -> Path:
    return ROOT / RESULTS_REL / "cache" / f"fold_{fold:02d}.pt"


def validate_cache(cache: dict, fold: int, context: dict) -> None:
    required = {
        "schema_version", "fold", "best_epoch", "checkpoint_path", "checkpoint_sha256",
        "H_fused", "raw_logits", "adjusted_logits",
        "original_probability", "labels", "train_mask", "test_mask", "subject_indices",
    }
    require(required <= set(cache), f"Cache keys missing for fold {fold}")
    require(
        cache["schema_version"] == "component-localization-cache-v1",
        "Cache schema changed",
    )
    require(int(cache["fold"]) == fold, "Cache fold mismatch")
    require(tuple(cache["H_fused"].shape) == (EXPECTED_SUBJECTS, EXPECTED_HIDDEN), "H_fused cache shape changed")
    require(tuple(cache["raw_logits"].shape) == (EXPECTED_SUBJECTS, 3), "Raw-logit cache shape changed")
    require(tuple(cache["adjusted_logits"].shape) == (EXPECTED_SUBJECTS, 3), "Adjusted-logit cache shape changed")
    require(tuple(cache["original_probability"].shape) == (EXPECTED_SUBJECTS, 3), "Probability cache shape changed")
    require(tuple(cache["labels"].shape) == (EXPECTED_SUBJECTS,), "Label cache shape changed")
    for name in ("H_fused", "raw_logits", "adjusted_logits", "original_probability"):
        require(bool(torch.isfinite(cache[name]).all()), f"Cached {name} contains NaN/Inf")
    require(
        float((cache["original_probability"].sum(dim=-1) - 1.0).abs().max()) <= 1e-6,
        "Cached Original probabilities do not sum to one",
    )
    checkpoint = Path(cache["checkpoint_path"])
    require(checkpoint.is_file(), "Cached checkpoint provenance path is missing")
    checkpoint_key = str(checkpoint.resolve())
    if checkpoint_key not in _VALIDATED_CHECKPOINT_PATHS:
        require(
            query.file_hash(checkpoint) == cache["checkpoint_sha256"],
            "Cached checkpoint provenance hash changed",
        )
        _VALIDATED_CHECKPOINT_PATHS.add(checkpoint_key)
    expected_indices = torch.as_tensor(context["dataset_dict"]["Index"], dtype=torch.long)
    require(torch.equal(cache["subject_indices"].cpu().long(), expected_indices), "Cached sample indices changed")
    require(
        torch.equal(
            cache["labels"].cpu().long(),
            context["dataset_data"]["Label"].cpu().long(),
        ),
        "Cached labels changed",
    )
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    require(torch.equal(cache["train_mask"].cpu().bool(), train_mask.cpu().bool()), "Cached train mask changed")
    require(torch.equal(cache["test_mask"].cpu().bool(), test_mask.cpu().bool()), "Cached test mask changed")
    require(bool(torch.isfinite(cache["H_fused"]).all()), "Cached H_fused contains NaN/Inf")


def validate_cache_fold_against_anchor(cache: dict, fold: int, context: dict) -> float:
    historical = {
        int(row["subject_index"]): row
        for row in context["anchor_rows"]
        if int(row["fold"]) == fold
    }
    positions = torch.where(cache["test_mask"])[0].tolist()
    require(len(historical) == len(positions), f"Anchor fold {fold} size changed")
    max_delta = 0.0
    for position in positions:
        subject_index = int(cache["subject_indices"][position])
        require(subject_index in historical, f"Anchor fold {fold} subject changed")
        row = historical[subject_index]
        require(int(cache["labels"][position]) == int(row["truth"]), "Anchor truth changed")
        for class_index, class_name in enumerate(CLASS_NAMES):
            max_delta = max(
                max_delta,
                abs(
                    float(cache["original_probability"][position, class_index])
                    - float(row[f"probability_{class_name}"])
                ),
            )
    require(
        max_delta <= ANCHOR_TOLERANCE,
        f"STOP_ORIGINAL_ANCHOR_MISMATCH: fold {fold} max delta {max_delta}",
    )
    return max_delta


def create_fold_cache(context: dict, fold: int, model=None) -> dict:
    own_model = model is None
    best_epoch = None
    checkpoint = None
    if model is None:
        model, best_epoch, checkpoint = build_original(context, fold)
    else:
        summary = json.loads(
            (context["anchor_root"] / f"fold_{fold:02d}" / "summary.json").read_text(encoding="utf-8")
        )
        best_epoch = int(summary["result"]["best_epoch"])
        checkpoint = context["anchor_root"] / f"fold_{fold:02d}" / "checkpoint_best.pt"
    with torch.no_grad():
        raw_logits, _, _, intermediates = model(
            context["dataset_data"]["Feature"], return_intermediates=True
        )
        require(torch.equal(raw_logits, intermediates["raw_logits"]), "Intermediate raw logits changed")
        identity = torch.eye(
            EXPECTED_SUBJECTS,
            device=context["device"],
            dtype=intermediates["H_fused"].dtype,
        )
        replayed_raw = model.GCN(intermediates["H_fused"], identity)
        require(
            torch.equal(raw_logits, replayed_raw),
            "Cached H_fused does not exactly replay the Original classifier",
        )
        adjusted, probability = score_logits(
            raw_logits,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
    checkpoint_sha256 = query.file_hash(checkpoint)
    _VALIDATED_CHECKPOINT_PATHS.add(str(checkpoint.resolve()))
    cache = {
        "schema_version": "component-localization-cache-v1",
        "fold": fold,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "H_fused": intermediates["H_fused"].detach().cpu(),
        "raw_logits": raw_logits.detach().cpu(),
        "adjusted_logits": adjusted.detach().cpu(),
        "original_probability": probability.detach().cpu(),
        "labels": context["dataset_data"]["Label"].detach().cpu().long(),
        "train_mask": context["dataset_data"]["Mask"][fold][0].detach().cpu().bool(),
        "test_mask": context["dataset_data"]["Mask"][fold][1].detach().cpu().bool(),
        "subject_indices": torch.as_tensor(context["dataset_dict"]["Index"], dtype=torch.long),
    }
    validate_cache(cache, fold, context)
    cache["anchor_probability_max_abs_diff"] = validate_cache_fold_against_anchor(
        cache, fold, context
    )
    path = cache_path(fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    if own_model:
        del model
        torch.cuda.empty_cache()
    return cache


def load_or_create_cache(context: dict, fold: int, model=None) -> dict:
    path = cache_path(fold)
    if path.is_file():
        cache = torch_load(path, "cpu")
        validate_cache(cache, fold, context)
        validate_cache_fold_against_anchor(cache, fold, context)
        return cache
    return create_fold_cache(context, fold, model=model)


def cache_to_device(cache: dict, device: torch.device) -> dict:
    result = dict(cache)
    for name in (
        "H_fused", "raw_logits", "adjusted_logits", "original_probability",
        "labels", "train_mask", "test_mask", "subject_indices",
    ):
        result[name] = cache[name].to(device)
    return result


def anchor_rows_from_caches(context: dict) -> list[dict]:
    rows = []
    for fold in FOLDS:
        cache = load_or_create_cache(context, fold)
        positions = torch.where(cache["test_mask"])[0].tolist()
        for position in positions:
            rows.append(
                probability_row(
                    fold,
                    int(cache["subject_indices"][position]),
                    int(cache["labels"][position]),
                    cache["raw_logits"][position].numpy(),
                    cache["adjusted_logits"][position].numpy(),
                    cache["original_probability"][position].numpy(),
                )
            )
    rows = ordered_rows(rows, context)
    metrics = metrics_from_rows(rows)
    require(metrics["correct"] == 556, "STOP_ORIGINAL_ANCHOR_MISMATCH")
    require(metrics["confusion_matrix"] == EXPECTED_ORIGINAL["confusion_matrix"], "STOP_ORIGINAL_ANCHOR_MISMATCH")
    expected = context["anchor_rows"]
    max_delta = 0.0
    for actual, historical in zip(rows, expected):
        require(int(actual["subject_index"]) == int(historical["subject_index"]), "Anchor OOF order changed")
        for name in CLASS_NAMES:
            max_delta = max(
                max_delta,
                abs(float(actual[f"probability_{name}"]) - float(historical[f"probability_{name}"])),
            )
    require(max_delta <= ANCHOR_TOLERANCE, f"STOP_ORIGINAL_ANCHOR_MISMATCH: max delta {max_delta}")
    return rows


def probability_row(
    fold: int,
    subject_index: int,
    truth: int,
    raw,
    adjusted,
    probability,
    **extra,
) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    row = {
        "fold": fold,
        "subject_index": subject_index,
        "truth": truth,
        "prediction": int(probability.argmax()),
        **{f"raw_logit_{name}": float(raw[index]) for index, name in enumerate(CLASS_NAMES)},
        **{f"adjusted_score_{name}": float(adjusted[index]) for index, name in enumerate(CLASS_NAMES)},
        **{f"probability_{name}": float(probability[index]) for index, name in enumerate(CLASS_NAMES)},
    }
    row.update(extra)
    return row


def modality_map(context: dict) -> dict[str, list[int]]:
    result = {}
    for name, indices in zip(
        context["dataset_dict"]["Modal_Name"], context["dataset_dict"]["Modal_Index"]
    ):
        require(name in MODALITY_ALIASES, f"Unexpected modality name: {name}")
        result[MODALITY_ALIASES[name]] = list(indices)
    require(set(result) == {"MRI", "PET", "CSF", "Risk", "COG", "ROI"}, "Modality map changed")
    return result


def modality_settings() -> list[str]:
    modalities = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
    return ["all_modalities", *[f"without_{name}" for name in modalities], *[f"{name}_only" for name in modalities]]


def ablated_features(features: torch.Tensor, setting: str, modalities: dict[str, list[int]]) -> torch.Tensor:
    if setting == "all_modalities":
        return features
    result = features.clone()
    if setting.startswith("without_"):
        name = setting.removeprefix("without_")
        result[:, torch.as_tensor(modalities[name], device=result.device)] = 0.0
        return result
    require(setting.endswith("_only"), f"Unknown modality setting: {setting}")
    name = setting.removesuffix("_only")
    result.zero_()
    indices = torch.as_tensor(modalities[name], device=result.device)
    result[:, indices] = features[:, indices]
    return result


def fold_metrics(rows: list[dict]) -> list[dict]:
    output = []
    for fold in FOLDS:
        subset = [row for row in rows if int(row["fold"]) == fold]
        metrics = metrics_from_rows(subset)
        output.append({"fold": fold, "correct": metrics["correct"], **{name: metrics[name] for name in METRIC_NAMES}})
    return output


def save_probability_experiment(
    output_root: Path,
    config: dict,
    rows: list[dict],
    context: dict,
    *,
    parameter_count_value: int,
    elapsed_seconds: float,
    extra_report: dict | None = None,
) -> dict:
    rows = ordered_rows(rows, context)
    metrics = metrics_from_rows(rows)
    per_fold = fold_metrics(rows)
    report = {
        "config": config,
        "parameter_count": int(parameter_count_value),
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": metrics,
        "fold_metrics": per_fold,
        **(extra_report or {}),
    }
    write_json(output_root / "config.json", config)
    write_csv(output_root / "oof_predictions.csv", rows)
    write_json(output_root / "metrics.json", metrics)
    write_csv(output_root / "fold_metrics.csv", per_fold)
    write_csv(output_root / "confusion_matrix.csv", confusion_rows(metrics))
    write_json(output_root / "report.json", report)
    reloaded = read_csv(output_root / "oof_predictions.csv")
    require(metrics_from_rows(reloaded) == metrics, f"OOF readback failed: {output_root}")
    return report


def run_stage_a(context: dict) -> dict:
    summary_path = ROOT / RESULTS_REL / "modality_inference" / "summary.json"
    if summary_path.is_file() and all(cache_path(fold).is_file() for fold in FOLDS):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Modality inference completed", flush=True)
        return payload
    stage_started = time.perf_counter()
    modalities = modality_map(context)
    settings = modality_settings()
    fold_rows_all = []
    fold_root = ROOT / RESULTS_REL / "modality_inference" / "folds"
    for fold in FOLDS:
        fold_path = fold_root / f"fold_{fold:02d}.csv"
        cache_exists = cache_path(fold).is_file()
        if fold_path.is_file() and cache_exists:
            rows = read_csv(fold_path)
            require(len(rows) == len(settings) * int(context["dataset_data"]["Mask"][fold][1].sum()), "Stage A fold row count changed")
            fold_rows_all.extend(rows)
            continue
        model, _, _ = build_original(context, fold)
        cache_started = time.perf_counter()
        cache = load_or_create_cache(context, fold, model=model)
        cache_elapsed = time.perf_counter() - cache_started if not cache_exists else 0.0
        test_positions = torch.where(context["dataset_data"]["Mask"][fold][1])[0].tolist()
        rows = []
        for setting in settings:
            setting_started = time.perf_counter()
            if setting == "all_modalities":
                raw = cache["raw_logits"].to(context["device"])
                adjusted = cache["adjusted_logits"].to(context["device"])
                probability = cache["original_probability"].to(context["device"])
            else:
                inputs = ablated_features(context["dataset_data"]["Feature"], setting, modalities)
                with torch.no_grad():
                    raw, _, _ = model(inputs)
                    adjusted, probability = score_logits(
                        raw,
                        context["dataset_dict"]["Label_Weight"],
                        float(context["config"].logit_adjust_tau),
                    )
            setting_elapsed = (
                cache_elapsed if setting == "all_modalities" else time.perf_counter() - setting_started
            )
            for position in test_positions:
                rows.append(
                    probability_row(
                        fold,
                        int(context["dataset_dict"]["Index"][position]),
                        int(context["dataset_data"]["Label"][position]),
                        raw[position].detach().cpu().numpy(),
                        adjusted[position].detach().cpu().numpy(),
                        probability[position].detach().cpu().numpy(),
                        setting=setting,
                        setting_elapsed_seconds=setting_elapsed,
                    )
                )
        write_csv(fold_path, rows)
        fold_rows_all.extend(rows)
        del model
        torch.cuda.empty_cache()
    reports = {}
    all_metrics = None
    table_rows = []
    for setting in settings:
        rows = [row for row in fold_rows_all if row["setting"] == setting]
        setting_runtime = sum(
            float(next(row["setting_elapsed_seconds"] for row in rows if int(row["fold"]) == fold))
            for fold in FOLDS
        )
        output = ROOT / RESULTS_REL / "modality_inference" / setting
        report = save_probability_experiment(
            output,
            {"stage": "A", "setting": setting, "zero_is_standardized_mean": True},
            rows,
            context,
            parameter_count_value=EXPECTED_ORIGINAL_PARAMETERS,
            elapsed_seconds=setting_runtime,
        )
        if setting == "all_modalities":
            all_metrics = report["metrics"]
            require(all_metrics["correct"] == 556, "STOP_ORIGINAL_ANCHOR_MISMATCH")
            require(all_metrics["confusion_matrix"] == EXPECTED_ORIGINAL["confusion_matrix"], "STOP_ORIGINAL_ANCHOR_MISMATCH")
        reports[setting] = report
    require(all_metrics is not None, "Stage A all-modal result missing")
    for setting in settings:
        metrics = reports[setting]["metrics"]
        reports[setting]["delta_vs_all"] = {
            "correct": metrics["correct"] - all_metrics["correct"],
            **{name: metrics[name] - all_metrics[name] for name in METRIC_NAMES},
        }
        write_json(
            ROOT / RESULTS_REL / "modality_inference" / setting / "report.json",
            reports[setting],
        )
        table_rows.append(
            {
                "setting": setting,
                "correct": metrics["correct"],
                "delta_correct": metrics["correct"] - all_metrics["correct"],
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
    payload = {
        "stage": "A",
        "settings": reports,
        "table": table_rows,
        "modality_indices": modalities,
        "modality_contribution": {
            modality: {
                "leave_one_out_correct_drop": (
                    all_metrics["correct"]
                    - reports[f"without_{modality}"]["metrics"]["correct"]
                ),
                "leave_one_out_probability_auc_drop": (
                    all_metrics["macro_auc"]
                    - reports[f"without_{modality}"]["metrics"]["macro_auc"]
                ),
                "only_one_correct_gap_to_all": (
                    all_metrics["correct"]
                    - reports[f"{modality}_only"]["metrics"]["correct"]
                ),
                "only_one_probability_auc_gap_to_all": (
                    all_metrics["macro_auc"]
                    - reports[f"{modality}_only"]["metrics"]["macro_auc"]
                ),
            }
            for modality in ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
        },
        "elapsed_seconds": time.perf_counter() - stage_started,
    }
    cache_manifest = {
        "schema_version": "component-localization-cache-manifest-v1",
        "folds": list(FOLDS),
        "sample_count": EXPECTED_SUBJECTS,
        "H_fused_shape": [EXPECTED_SUBJECTS, EXPECTED_HIDDEN],
        "contents": [
            "H_fused", "raw_logits", "adjusted_logits", "original_probability",
            "labels", "train_mask", "test_mask", "subject_indices", "fold",
        ],
        "files": [str(cache_path(fold).relative_to(ROOT)).replace("\\", "/") for fold in FOLDS],
        "original_encoder_forwarded_once_per_fold_for_full_input": True,
    }
    write_json(ROOT / RESULTS_REL / "cache" / "manifest.json", cache_manifest)
    payload["cache_manifest"] = cache_manifest
    # Only publish the resumable stage summary after the checkpoint-derived
    # cache has reproduced the complete 556/598 historical anchor.
    anchor_rows_from_caches(context)
    write_csv(summary_path.with_suffix(".csv"), table_rows)
    write_json(summary_path, payload)
    print("Modality inference completed", flush=True)
    return payload


def build_head(name: str, device: torch.device):
    if name == "linear_probe":
        head = LinearProbeHead()
    elif name == "residual_mlp":
        head = ResidualMLPHead()
    elif name == "frozen_difformer":
        head = DIFFormer_GraphHead(
            Dim_emb=96,
            hidden=48,
            out_channels=3,
            P=0.20,
            num_layers=2,
            num_heads=1,
            graph_weight=0.10,
            alpha=0.10,
            kernel="simple",
            use_graph=False,
        )
    elif name == "pee_head":
        head = PEEHead()
    elif name == "sparse_residual_gcn":
        head = SparseResidualGCNHead()
    else:
        raise ValueError(f"Unknown head: {name}")
    return head.to(device)


def head_raw_logits(head, name: str, features: torch.Tensor, adjacency=None) -> torch.Tensor:
    if name == "frozen_difformer":
        identity = torch.eye(features.shape[0], device=features.device, dtype=features.dtype)
        return head(features, identity)
    if name == "sparse_residual_gcn":
        require(adjacency is not None, "Sparse GCN adjacency missing")
        return head(features, adjacency)
    return head(features)


def head_probability(
    raw_logits: torch.Tensor,
    name: str,
    label_weight: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "pee_head":
        adjusted = raw_logits - float(tau) * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, 1, -1)
        member_probability = torch.softmax(adjusted, dim=-1)
        return adjusted, member_probability.mean(dim=0)
    return score_logits(raw_logits, label_weight, tau)


def head_loss(
    raw_logits: torch.Tensor,
    name: str,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    criterion,
) -> torch.Tensor:
    if name == "pee_head":
        return torch.stack(
            [criterion(member[train_mask], labels[train_mask]) for member in raw_logits]
        ).mean()
    return criterion(raw_logits[train_mask], labels[train_mask])


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def head_prediction_rows(
    fold: int,
    best_epoch: int,
    name: str,
    raw_logits: torch.Tensor,
    adjusted_logits: torch.Tensor,
    probability: torch.Tensor,
    cache: dict,
) -> list[dict]:
    positions = torch.where(cache["test_mask"])[0].tolist()
    rows = []
    for position in positions:
        if name == "pee_head":
            ensemble_raw = raw_logits[:, position].mean(dim=0)
            ensemble_adjusted = adjusted_logits[:, position].mean(dim=0)
        else:
            ensemble_raw = raw_logits[position]
            ensemble_adjusted = adjusted_logits[position]
        extra = {"head": name, "best_epoch": best_epoch}
        if name == "pee_head":
            member_probability = torch.softmax(adjusted_logits[:, position], dim=-1)
            for member in range(member_probability.shape[0]):
                extra[f"member_{member}_prediction"] = int(member_probability[member].argmax())
                for class_index, class_name in enumerate(CLASS_NAMES):
                    extra[f"member_{member}_probability_{class_name}"] = float(member_probability[member, class_index].cpu())
        rows.append(
            probability_row(
                fold,
                int(cache["subject_indices"][position]),
                int(cache["labels"][position]),
                ensemble_raw.detach().cpu().numpy(),
                ensemble_adjusted.detach().cpu().numpy(),
                probability[position].detach().cpu().numpy(),
                **extra,
            )
        )
    return rows


def completed_fold(fold_dir: Path, name: str, config_payload: dict) -> tuple[dict, list[dict]] | None:
    summary_path = fold_dir / "summary.json"
    prediction_path = fold_dir / "best_predictions.csv"
    epoch_path = fold_dir / "epoch_metrics.csv"
    checkpoint_path = fold_dir / "checkpoint_best.pt"
    if not (
        summary_path.is_file()
        and prediction_path.is_file()
        and epoch_path.is_file()
        and checkpoint_path.is_file()
    ):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("passed") is True, f"Incomplete fold: {fold_dir}")
    require(summary.get("name") == name, "Completed fold name mismatch")
    require(summary.get("config") == config_payload, "Completed fold config mismatch")
    require(len(read_csv(epoch_path)) == EPOCHS, "Completed fold epoch count changed")
    rows = read_csv(prediction_path)
    require(len(rows) == int(summary["test_size"]), "Completed fold prediction count changed")
    require(
        len({int(row["subject_index"]) for row in rows}) == len(rows),
        "Completed fold subject indices are not unique",
    )
    require(
        {int(row["fold"]) for row in rows} == {int(summary["fold"])},
        "Completed fold prediction fold id changed",
    )
    metrics = metrics_from_rows(rows)
    require(metrics == summary["best_metrics"], "Completed fold metrics changed")
    return summary, rows


def train_head_fold(
    context: dict,
    name: str,
    fold: int,
    output_root: Path,
    *,
    raw_features: torch.Tensor | None = None,
    epochs: int = EPOCHS,
) -> tuple[dict, list[dict]]:
    config_payload = {
        "name": name,
        "fold": fold,
        "seed": SEED,
        "epochs": epochs,
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "label_smoothing": 0.05,
        "logit_adjust_tau": float(context["config"].logit_adjust_tau),
    }
    fold_dir = output_root / f"fold_{fold:02d}"
    if epochs == EPOCHS:
        resumed = completed_fold(fold_dir, name, config_payload)
        if resumed is not None:
            return resumed
    cache_cpu = load_or_create_cache(context, fold)
    cache = cache_to_device(cache_cpu, context["device"])
    features = cache["H_fused"] if raw_features is None else raw_features
    features = features.detach()
    labels = cache["labels"]
    train_mask = cache["train_mask"]
    test_mask = cache["test_mask"]
    SET_Random(SEED)
    head = build_head(name, context["device"]) if name != "raw_mlp" else RawFeatureMLP(int(features.shape[1]), dropout=float(context["config"].Drop_rate)).to(context["device"])
    effective_name = name if name != "raw_mlp" else "raw_mlp"
    criterion = torch.nn.CrossEntropyLoss(
        weight=context["dataset_dict"]["Label_Weight"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min)
    )
    adjacency = None
    if name == "sparse_residual_gcn":
        adjacency = build_mutual_knn_graph(features, top_k=8)
        require(torch.allclose(adjacency, adjacency.T, atol=1e-6, rtol=0.0), "Graph is not symmetric")
    best = None
    best_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        raw = head_raw_logits(head, name if name != "raw_mlp" else "linear_raw", features, adjacency)
        loss = head_loss(raw, name, labels, train_mask, criterion)
        require(bool(torch.isfinite(loss)), f"{name} fold {fold}: non-finite loss")
        loss.backward()
        require(gradients_finite(head), f"{name} fold {fold}: non-finite gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        if name == "sparse_residual_gcn":
            with torch.no_grad():
                head.gamma_parameter.clamp_(0.0, 0.5)
        scheduler.step()
        head.eval()
        with torch.no_grad():
            evaluated_raw = head_raw_logits(head, name if name != "raw_mlp" else "linear_raw", features, adjacency)
            evaluated_adjusted, evaluated_probability = head_probability(
                evaluated_raw,
                name,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
            metrics = metrics_from_probability(
                labels[test_mask].cpu().numpy(), evaluated_probability[test_mask].cpu().numpy()
            )
        score = selection_tuple(metrics)
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "metrics": metrics}
            best_state = clone_cpu_state(head)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                **{name_: metrics[name_] for name_ in METRIC_NAMES},
            }
        )
    require(best is not None and best_state is not None, "Best head state missing")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    with torch.no_grad():
        best_raw = head_raw_logits(head, name if name != "raw_mlp" else "linear_raw", features, adjacency)
        best_adjusted, best_probability = head_probability(
            best_raw,
            name,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
        best_metrics = metrics_from_probability(
            labels[test_mask].cpu().numpy(), best_probability[test_mask].cpu().numpy()
        )
        rows = head_prediction_rows(
            fold,
            int(best["epoch"]),
            name,
            best_raw,
            best_adjusted,
            best_probability,
            cache,
        )
    require(best_metrics == best["metrics"], "Best-head reload changed metrics")
    extra = {}
    if name == "pee_head":
        member_probability = torch.softmax(best_adjusted.detach(), dim=-1)
        extra["member_diagnostics"] = pee_pairwise_diagnostics(
            member_probability[:, test_mask].cpu()
        )
    if name == "sparse_residual_gcn":
        extra["gamma"] = float(head.gamma.detach().cpu())
        extra["gamma_parameter"] = float(head.gamma_parameter.detach().cpu())
    summary = {
        "passed": True,
        "name": name,
        "fold": fold,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(head),
        "elapsed_seconds": time.perf_counter() - started,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "config": config_payload,
        **extra,
    }
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_dir / "summary.json", summary)
    write_csv(fold_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(fold_dir / "best_predictions.csv", rows)
    torch.save(
        {"summary": summary, "state_dict": best_state}, fold_dir / "checkpoint_best.pt"
    )
    del head, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def aggregate_pee(rows: list[dict]) -> dict:
    member_metrics = {}
    member_probability = []
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    for member in range(4):
        probability = np.asarray(
            [
                [float(row[f"member_{member}_probability_{name}"]) for name in CLASS_NAMES]
                for row in rows
            ],
            dtype=np.float64,
        )
        member_probability.append(probability)
        member_metrics[f"member_{member}"] = metrics_from_probability(truth, probability)
    diagnostics = pee_pairwise_diagnostics(
        torch.as_tensor(np.stack(member_probability), dtype=torch.float32)
    )
    return {"member_metrics": member_metrics, "member_diagnostics": diagnostics}


def run_head_experiment(
    context: dict,
    name: str,
    output_root: Path,
    *,
    raw_feature_indices: list[int] | None = None,
) -> dict:
    report_path = output_root / "report.json"
    if report_path.is_file() and (output_root / "oof_predictions.csv").is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = ordered_rows(read_csv(output_root / "oof_predictions.csv"), context)
        require(metrics_from_rows(rows) == report["metrics"], "Resumed head OOF metrics changed")
        require(
            all((output_root / f"fold_{fold:02d}" / "checkpoint_best.pt").is_file() for fold in FOLDS),
            "Resumed head checkpoints are incomplete",
        )
        return report
    started = time.perf_counter()
    summaries = []
    rows = []
    for fold in FOLDS:
        raw_features = None
        training_name = name
        if raw_feature_indices is not None:
            raw_features = context["dataset_data"]["Feature"][:, raw_feature_indices].detach()
            training_name = "raw_mlp"
        summary, fold_rows = train_head_fold(
            context,
            training_name,
            fold,
            output_root,
            raw_features=raw_features,
        )
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = ordered_rows(rows, context)
    extra = {
        "fold_best_epochs": [summary["best_epoch"] for summary in summaries],
        "fold_summaries": summaries,
    }
    if name == "pee_head":
        extra.update(aggregate_pee(rows))
    if name == "sparse_residual_gcn":
        extra["gamma_fold_values"] = [summary["gamma"] for summary in summaries]
        extra["gamma_mean"] = float(np.mean(extra["gamma_fold_values"]))
    return save_probability_experiment(
        output_root,
        {
            "stage": "B" if name in HEAD_NAMES else "C_or_D",
            "name": name,
            "folds": list(FOLDS),
            "seed": SEED,
            "epochs": EPOCHS,
            "train_mask_only": True,
            "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        },
        rows,
        context,
        parameter_count_value=int(summaries[0]["parameter_count"]),
        elapsed_seconds=time.perf_counter() - started,
        extra_report=extra,
    )


def stage_b_decision(results: dict) -> str:
    counts = {name: int(result["metrics"]["correct"]) for name, result in results.items()}
    pee = counts["pee_head"]
    if pee >= 557 and pee == max(counts.values()):
        return "PEE_HEAD_BEST"
    if counts["residual_mlp"] - EXPECTED_ORIGINAL["correct"] >= 3:
        return "MLP_BETTER_THAN_DIFFORMER"
    if max(counts.values()) <= EXPECTED_ORIGINAL["correct"] - 3:
        return "FROZEN_HEAD_INCONCLUSIVE"
    if max(EXPECTED_ORIGINAL["correct"], counts["frozen_difformer"]) - counts["residual_mlp"] >= 3:
        return "DIFFORMER_RELATION_USEFUL"
    if max(counts.values()) - min(counts.values()) <= 2:
        return "CLASSIFIER_NOT_BOTTLENECK"
    return "FROZEN_HEAD_INCONCLUSIVE"


def run_stage_b(context: dict) -> dict:
    summary_path = ROOT / RESULTS_REL / "frozen_heads" / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Frozen heads completed", flush=True)
        return payload
    results = {
        name: run_head_experiment(
            context, name, ROOT / RESULTS_REL / "frozen_heads" / name
        )
        for name in HEAD_NAMES
    }
    table = [
        {
            "head": "original_joint_difformer",
            "correct": EXPECTED_ORIGINAL["correct"],
            **{metric: EXPECTED_ORIGINAL[metric] for metric in METRIC_NAMES},
            "parameter_count": EXPECTED_ORIGINAL_PARAMETERS,
            "elapsed_seconds": 0.0,
        },
        *[
        {
            "head": name,
            "correct": result["metrics"]["correct"],
            **{metric: result["metrics"][metric] for metric in METRIC_NAMES},
            "parameter_count": result["parameter_count"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        for name, result in results.items()
        ],
    ]
    payload = {
        "stage": "B",
        "original_reference": EXPECTED_ORIGINAL,
        "results": results,
        "table": table,
        "decision": stage_b_decision(results),
    }
    write_csv(summary_path.with_suffix(".csv"), table)
    write_json(summary_path, payload)
    print("Frozen heads completed", flush=True)
    return payload


def stage_c_decision(b2: int, b3: int, c2: int) -> str:
    if c2 - b2 >= 3 and c2 - b3 >= 3:
        return "OLD_GRAPH_BAD_NEW_SPARSE_GRAPH_USEFUL"
    if b3 - b2 >= 3 and c2 <= b3 + 2:
        return "GLOBAL_ATTENTION_USEFUL_GRAPH_BAD"
    if b2 - b3 >= 3 and b2 - c2 >= 3:
        return "CROSS_SUBJECT_PROPAGATION_HARMFUL"
    if max(b2, b3, c2) - min(b2, b3, c2) <= 2:
        return "RELATION_NOT_USEFUL"
    return "INCONCLUSIVE_TREND"


def run_stage_c(context: dict, stage_b: dict) -> dict:
    summary_path = ROOT / RESULTS_REL / "sparse_graph" / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Sparse graph {payload['status']}", flush=True)
        return payload
    b2 = int(stage_b["results"]["residual_mlp"]["metrics"]["correct"])
    b3 = int(stage_b["results"]["frozen_difformer"]["metrics"]["correct"])
    should_run = max(EXPECTED_ORIGINAL["correct"], b3) >= b2 - 2
    if not should_run:
        payload = {"stage": "C", "status": "skipped", "reason": "Residual MLP clearly dominates relation heads"}
    else:
        c2_result = run_head_experiment(
            context,
            "sparse_residual_gcn",
            ROOT / RESULTS_REL / "sparse_graph" / "sparse_residual_gcn",
        )
        c2 = int(c2_result["metrics"]["correct"])
        payload = {
            "stage": "C",
            "status": "completed",
            "C0_residual_mlp": stage_b["results"]["residual_mlp"],
            "C1_frozen_difformer": stage_b["results"]["frozen_difformer"],
            "C2_sparse_residual_gcn": c2_result,
            "decision": stage_c_decision(b2, b3, c2),
        }
    write_json(summary_path, payload)
    print(f"Sparse graph {payload['status']}", flush=True)
    return payload


def should_run_stage_d(stage_a: dict) -> tuple[bool, dict]:
    settings = stage_a["settings"]
    all_correct = int(settings["all_modalities"]["metrics"]["correct"])
    cog_correct = int(settings["COG_only"]["metrics"]["correct"])
    without_cog = int(settings["without_COG"]["metrics"]["correct"])
    other_drops = {
        name: all_correct - int(settings[f"without_{name}"]["metrics"]["correct"])
        for name in ("MRI", "PET", "CSF", "Risk")
    }
    evidence = {
        "COG_only_close_to_all": abs(cog_correct - all_correct) <= 2,
        "without_COG_significant_drop": all_correct - without_cog >= 3,
        "other_modalities_negligible": all(abs(value) <= 2 for value in other_drops.values()),
        "other_modality_correct_drops": other_drops,
    }
    should_run = (
        evidence["COG_only_close_to_all"]
        or evidence["without_COG_significant_drop"]
        or evidence["other_modalities_negligible"]
    )
    return bool(should_run), evidence


def stage_d_decision(d0: int, d1: int, d2: int, best_head_name: str, best_head_correct: int) -> str:
    if max(d0, d1, d2) - min(d0, d1, d2) <= 2:
        return "EXTRA_MODAL_SIGNAL_LIMITED"
    if d1 - d0 >= 3 and d2 <= d1 + 2:
        return "MULTIMODAL_FUSION_INEFFECTIVE"
    if d2 - d1 >= 3 and d2 - d0 >= 3:
        if best_head_name == "pee_head" and best_head_correct - EXPECTED_ORIGINAL["correct"] >= 3:
            return "CLASSIFIER_VARIANCE_BOTTLENECK"
        return "MULTIMODAL_FUSION_EFFECTIVE"
    return "INCONCLUSIVE_TREND"


def reuse_original_for_d2(context: dict, best_head_name: str) -> dict:
    output_root = ROOT / RESULTS_REL / "multimodal_capacity" / "current_multimodal"
    report = save_probability_experiment(
        output_root,
        {
            "stage": "D2",
            "name": "Current Multimodal Encoder + best Stage B head",
            "status": "reused_original",
            "reason": "No Stage B frozen head exceeded Original",
            "best_frozen_head": best_head_name,
        },
        deepcopy(context["anchor_rows"]),
        context,
        parameter_count_value=EXPECTED_ORIGINAL_PARAMETERS,
        elapsed_seconds=0.0,
        extra_report={
            "status": "reused_original",
            "reason": "No Stage B frozen head exceeded Original",
            "best_head": best_head_name,
            "encoder_retrained": False,
        },
    )
    return report


def d2_original_loss(
    criterion,
    head_logits: torch.Tensor,
    head_name: str,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    representations,
    auxiliary_outputs,
) -> torch.Tensor:
    if head_name != "pee_head":
        return criterion(
            head_logits, labels, train_mask, representations, auxiliary_outputs
        )
    main_loss = torch.stack(
        [
            criterion.main_loss(member_logits[train_mask], labels[train_mask])
            for member_logits in head_logits
        ]
    ).mean()
    one_vs_rest = F.one_hot(labels, num_classes=3).transpose(0, 1)
    auxiliary_loss = main_loss.new_zeros(())
    for label_index, (auxiliary_criterion, auxiliary_logits) in enumerate(
        zip(criterion.aux_losses, auxiliary_outputs)
    ):
        auxiliary_loss = auxiliary_loss + auxiliary_criterion(
            auxiliary_logits[train_mask], one_vs_rest[label_index][train_mask]
        )
    return main_loss + auxiliary_loss


def d2_completed_fold(
    fold_dir: Path, head_name: str, config_payload: dict
) -> tuple[dict, list[dict]] | None:
    name = f"current_multimodal_{head_name}"
    return completed_fold(fold_dir, name, config_payload)


def train_d2_fold(
    context: dict,
    head_name: str,
    fold: int,
    output_root: Path,
) -> tuple[dict, list[dict]]:
    experiment_name = f"current_multimodal_{head_name}"
    config_payload = {
        "name": experiment_name,
        "best_stage_b_head": head_name,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "lr": float(context["config"].lr),
        "weight_decay": float(context["config"].weight_decay),
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "loss": "historical weighted main CE + three Original OVR auxiliary losses",
        "label_smoothing": 0.05,
        "logit_adjust_tau": float(context["config"].logit_adjust_tau),
        "end_to_end": True,
    }
    fold_dir = output_root / f"fold_{fold:02d}"
    resumed = d2_completed_fold(fold_dir, head_name, config_payload)
    if resumed is not None:
        return resumed

    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    SET_Random(SEED)
    model = query.build_model(
        context["config"], context["dataset_dict"], "original", context["device"]
    )
    require(parameter_count(model) == EXPECTED_ORIGINAL_PARAMETERS, "D2 Original parameter count changed")
    replaced_difformer_parameters = parameter_count(model.GCN)
    # Remove the historical classifier completely. The identity keeps the
    # network forward API intact while H_fused remains a live tensor feeding the
    # selected external head.
    model.GCN = IdentityGraphHead().to(context["device"])
    active_backbone_parameters = parameter_count(model)
    require(
        active_backbone_parameters
        == EXPECTED_ORIGINAL_PARAMETERS - replaced_difformer_parameters,
        "D2 active-backbone parameter count changed",
    )
    # One deterministic seed-0 RNG stream covers the fresh backbone followed
    # by the replacement head, matching the stated D2 protocol.
    head = build_head(head_name, context["device"])
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + list(head.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min)
    )
    best = None
    best_model_state = None
    best_head_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        _, representations, auxiliary, intermediates = model(
            features, return_intermediates=True
        )
        raw_logits = head_raw_logits(
            head, head_name, intermediates["H_fused"], adjacency=None
        )
        loss = d2_original_loss(
            criterion,
            raw_logits,
            head_name,
            labels,
            train_mask,
            representations,
            auxiliary,
        )
        require(bool(torch.isfinite(loss)), f"D2 {head_name} fold {fold}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"D2 {head_name} fold {fold}: invalid encoder gradient")
        require(gradients_finite(head), f"D2 {head_name} fold {fold}: invalid head gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(context["config"].grad_clip)
            )
        optimizer.step()
        scheduler.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            _, _, _, evaluated_intermediates = model(
                features, return_intermediates=True
            )
            evaluated_raw = head_raw_logits(
                head, head_name, evaluated_intermediates["H_fused"], adjacency=None
            )
            evaluated_adjusted, evaluated_probability = head_probability(
                evaluated_raw,
                head_name,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
            metrics = metrics_from_probability(
                labels[test_mask].cpu().numpy(),
                evaluated_probability[test_mask].cpu().numpy(),
            )
        score = selection_tuple(metrics)
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "metrics": metrics}
            best_model_state = clone_cpu_state(model)
            best_head_state = clone_cpu_state(head)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                **{metric: metrics[metric] for metric in METRIC_NAMES},
            }
        )
    require(
        best is not None and best_model_state is not None and best_head_state is not None,
        "D2 best state missing",
    )
    model.load_state_dict(best_model_state, strict=True)
    head.load_state_dict(best_head_state, strict=True)
    model.eval()
    head.eval()
    with torch.no_grad():
        _, _, _, best_intermediates = model(features, return_intermediates=True)
        best_raw = head_raw_logits(
            head, head_name, best_intermediates["H_fused"], adjacency=None
        )
        best_adjusted, best_probability = head_probability(
            best_raw,
            head_name,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
        best_metrics = metrics_from_probability(
            labels[test_mask].cpu().numpy(), best_probability[test_mask].cpu().numpy()
        )
    cache_like = {
        "labels": labels,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "subject_indices": torch.as_tensor(
            context["dataset_dict"]["Index"], device=context["device"], dtype=torch.long
        ),
    }
    rows = head_prediction_rows(
        fold,
        int(best["epoch"]),
        head_name,
        best_raw,
        best_adjusted,
        best_probability,
        cache_like,
    )
    require(best_metrics == best["metrics"], "D2 best reload changed metrics")
    summary = {
        "passed": True,
        "name": experiment_name,
        "fold": fold,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(model) + parameter_count(head),
        "active_backbone_parameter_count": active_backbone_parameters,
        "replaced_difformer_parameter_count": replaced_difformer_parameters,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "config": config_payload,
    }
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_dir / "summary.json", summary)
    write_csv(fold_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(fold_dir / "best_predictions.csv", rows)
    torch.save(
        {
            "summary": summary,
            "original_model": best_model_state,
            "replacement_head": best_head_state,
        },
        fold_dir / "checkpoint_best.pt",
    )
    del model, head, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    return summary, rows


def run_end_to_end_d2(context: dict, head_name: str) -> dict:
    output_root = ROOT / RESULTS_REL / "multimodal_capacity" / "current_multimodal"
    report_path = output_root / "report.json"
    if report_path.is_file() and (output_root / "oof_predictions.csv").is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    summaries = []
    rows = []
    for fold in FOLDS:
        summary, fold_rows = train_d2_fold(
            context, head_name, fold, output_root
        )
        summaries.append(summary)
        rows.extend(fold_rows)
    extra = {
        "status": "trained_end_to_end",
        "best_stage_b_head": head_name,
        "fold_best_epochs": [summary["best_epoch"] for summary in summaries],
        "fold_summaries": summaries,
        "trainable_parameter_count": int(summaries[0]["trainable_parameter_count"]),
    }
    rows = ordered_rows(rows, context)
    if head_name == "pee_head":
        extra.update(aggregate_pee(rows))
    return save_probability_experiment(
        output_root,
        {
            "stage": "D2",
            "name": "Current Multimodal Encoder + best Stage B head",
            "best_stage_b_head": head_name,
            "end_to_end": True,
            "original_ovr_auxiliary_preserved": True,
            "folds": list(FOLDS),
            "seed": SEED,
            "epochs": EPOCHS,
        },
        rows,
        context,
        parameter_count_value=int(summaries[0]["parameter_count"]),
        elapsed_seconds=time.perf_counter() - started,
        extra_report=extra,
    )


def run_stage_d(context: dict, stage_a: dict, stage_b: dict) -> dict:
    summary_path = ROOT / RESULTS_REL / "multimodal_capacity" / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Multimodal capacity {payload['status']}", flush=True)
        return payload
    should_run, trigger = should_run_stage_d(stage_a)
    if not should_run:
        payload = {"stage": "D", "status": "skipped", "trigger_evidence": trigger}
        write_json(summary_path, payload)
        print("Multimodal capacity skipped", flush=True)
        return payload
    modalities = modality_map(context)
    d0 = run_head_experiment(
        context,
        "cog_only_mlp",
        ROOT / RESULTS_REL / "multimodal_capacity" / "cog_only_mlp",
        raw_feature_indices=modalities["COG"],
    )
    d1 = run_head_experiment(
        context,
        "raw_all_mlp",
        ROOT / RESULTS_REL / "multimodal_capacity" / "raw_all_mlp",
        raw_feature_indices=list(range(EXPECTED_FEATURES)),
    )
    best_head_name, best_head = max(
        stage_b["results"].items(),
        key=lambda item: (
            item[1]["metrics"]["correct"],
            item[1]["metrics"]["macro_auc"],
            item[1]["metrics"]["macro_f1"],
        ),
    )
    if int(best_head["metrics"]["correct"]) <= EXPECTED_ORIGINAL["correct"]:
        d2 = reuse_original_for_d2(context, best_head_name)
    else:
        d2 = run_end_to_end_d2(context, best_head_name)
    payload = {
        "stage": "D",
        "status": "completed",
        "trigger_evidence": trigger,
        "D0_COG_only_MLP": d0,
        "D1_Raw_All_MLP": d1,
        "D2_Current_Multimodal": d2,
        "decision": stage_d_decision(
            int(d0["metrics"]["correct"]),
            int(d1["metrics"]["correct"]),
            int(d2["metrics"]["correct"]),
            best_head_name,
            int(best_head["metrics"]["correct"]),
        ),
    }
    write_json(summary_path, payload)
    print("Multimodal capacity completed", flush=True)
    return payload


def head_smoke_loss(context: dict, name: str, features: torch.Tensor, labels, train_mask, adjacency=None) -> dict:
    SET_Random(SEED)
    if name == "raw_mlp":
        head = RawFeatureMLP(int(features.shape[1]), dropout=float(context["config"].Drop_rate)).to(context["device"])
    else:
        head = build_head(name, context["device"])
    criterion = torch.nn.CrossEntropyLoss(
        weight=context["dataset_dict"]["Label_Weight"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        head.parameters(), lr=float(context["config"].lr), weight_decay=float(context["config"].weight_decay)
    )
    losses = []
    for _ in range(SMOKE_EPOCHS):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        raw = head_raw_logits(head, name if name != "raw_mlp" else "linear_raw", features.detach(), adjacency)
        loss = head_loss(raw, name, labels, train_mask, criterion)
        changed_labels = labels.clone()
        changed_labels[~train_mask] = (changed_labels[~train_mask] + 1) % 3
        changed_loss = head_loss(raw, name, changed_labels, train_mask, criterion)
        require(torch.equal(loss, changed_loss), f"{name}: test labels entered training loss")
        require(bool(torch.isfinite(loss)), f"{name}: smoke loss non-finite")
        loss.backward()
        require(gradients_finite(head), f"{name}: smoke gradient non-finite")
        optimizer.step()
        if name == "sparse_residual_gcn":
            with torch.no_grad():
                head.gamma_parameter.clamp_(0.0, 0.5)
        losses.append(float(loss.detach().cpu()))
    head.eval()
    with torch.no_grad():
        raw = head_raw_logits(head, name if name != "raw_mlp" else "linear_raw", features.detach(), adjacency)
        _, probability = head_probability(
            raw, name, context["dataset_dict"]["Label_Weight"], float(context["config"].logit_adjust_tau)
        )
    require(bool(torch.isfinite(probability).all()), f"{name}: smoke probability non-finite")
    require(float((probability.sum(dim=-1) - 1.0).abs().max()) <= 1e-6, f"{name}: probabilities do not sum to one")
    return {"losses": losses, "parameter_count": parameter_count(head), "passed": True}


def d2_end_to_end_smoke(context: dict) -> dict:
    labels = context["dataset_data"]["Label"]
    features = context["dataset_data"]["Feature"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    SET_Random(SEED)
    model = query.build_model(
        context["config"], context["dataset_dict"], "original", context["device"]
    )
    replaced_parameters = parameter_count(model.GCN)
    model.GCN = IdentityGraphHead().to(context["device"])
    head = build_head("pee_head", context["device"])
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    parameters = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    losses = []
    for _ in range(SMOKE_EPOCHS):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        _, representations, auxiliary, intermediates = model(
            features, return_intermediates=True
        )
        raw_logits = head(intermediates["H_fused"])
        loss = d2_original_loss(
            criterion,
            raw_logits,
            "pee_head",
            labels,
            train_mask,
            representations,
            auxiliary,
        )
        changed_labels = labels.clone()
        changed_labels[test_mask] = (changed_labels[test_mask] + 1) % 3
        changed_loss = d2_original_loss(
            criterion,
            raw_logits,
            "pee_head",
            changed_labels,
            train_mask,
            representations,
            auxiliary,
        )
        require(torch.equal(loss, changed_loss), "D2 smoke: test labels entered Original/OVR loss")
        require(bool(torch.isfinite(loss)), "D2 smoke loss is non-finite")
        loss.backward()
        require(gradients_finite(model), "D2 smoke encoder gradient is non-finite")
        require(gradients_finite(head), "D2 smoke head gradient is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "passed": True,
        "head": "pee_head",
        "epochs": SMOKE_EPOCHS,
        "losses": losses,
        "old_difformer_removed": True,
        "replaced_difformer_parameters": replaced_parameters,
        "active_backbone_parameters": parameter_count(model),
        "replacement_head_parameters": parameter_count(head),
        "original_ovr_loss_preserved": True,
        "test_labels_in_training_loss": False,
    }


def run_smoke() -> dict:
    summary_path = ROOT / RESULTS_REL / "smoke" / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        require(payload.get("passed") is True, "Existing smoke did not pass")
        print("Original anchor reproduced", flush=True)
        return payload
    context = load_context()
    anchor = verify_anchor_source(context)
    model, _, _ = build_original(context, 0)
    features = context["dataset_data"]["Feature"]
    with torch.no_grad():
        default_raw, default_representations, default_auxiliary = model(features)
        extended_raw, extended_representations, extended_auxiliary, intermediates = model(
            features, return_intermediates=True
        )
    require(torch.equal(default_raw, extended_raw), "Default/extended Original logits changed")
    require(torch.equal(extended_raw, intermediates["raw_logits"]), "Intermediate final logits changed")
    require(tuple(intermediates["H_fused"].shape) == (598, 96), "H_fused shape changed")
    require(len(default_representations) == len(extended_representations) == 3, "Representation output changed")
    require(len(default_auxiliary) == len(extended_auxiliary) == 3, "Auxiliary output changed")
    cache = create_fold_cache(context, 0, model=model)
    cache_device = cache_to_device(cache, context["device"])
    labels = cache_device["labels"]
    train_mask = cache_device["train_mask"]
    adjacency = build_mutual_knn_graph(cache_device["H_fused"], top_k=8)
    require(torch.allclose(adjacency, adjacency.T, atol=1e-6, rtol=0.0), "Smoke graph is not symmetric")
    checks = {}
    for name in HEAD_NAMES:
        checks[name] = head_smoke_loss(
            context, name, cache_device["H_fused"], labels, train_mask
        )
    checks["sparse_residual_gcn"] = head_smoke_loss(
        context,
        "sparse_residual_gcn",
        cache_device["H_fused"],
        labels,
        train_mask,
        adjacency,
    )
    modalities = modality_map(context)
    checks["cog_only_mlp"] = head_smoke_loss(
        context,
        "raw_mlp",
        features[:, modalities["COG"]],
        labels,
        train_mask,
    )
    checks["raw_all_mlp"] = head_smoke_loss(
        context, "raw_mlp", features, labels, train_mask
    )
    checks["D2_current_multimodal_pee_head"] = d2_end_to_end_smoke(context)
    require(all(parameter.grad is None for parameter in model.parameters()), "Frozen Original received head gradients")
    payload = {
        "model": MODEL_NAME,
        "passed": True,
        "fold": 0,
        "epochs_per_head": SMOKE_EPOCHS,
        "anchor": anchor,
        "default_extended_logits_max_abs_diff": float((default_raw - extended_raw).abs().max()),
        "intermediate_shapes": {
            key: list(value.shape) if torch.is_tensor(value) else [list(item.shape) for item in value]
            for key, value in intermediates.items()
        },
        "cached_sample_indices_exact": torch.equal(
            cache["subject_indices"], torch.as_tensor(context["dataset_dict"]["Index"], dtype=torch.long)
        ),
        "graph_symmetric": True,
        "graph_uses_labels": False,
        "test_labels_in_training_loss": False,
        "head_checks": checks,
        "environment": environment_snapshot(context["device"]),
    }
    write_json(summary_path, payload)
    print("Original anchor reproduced", flush=True)
    return payload


def component_conclusions(stage_a: dict, stage_b: dict, stage_c: dict, stage_d: dict) -> tuple[list[dict], str]:
    b_decision = stage_b["decision"]
    if stage_d.get("status") == "completed":
        d_decision = stage_d["decision"]
        if d_decision == "EXTRA_MODAL_SIGNAL_LIMITED":
            multimodal = "cognitive-dominant"
        elif d_decision in {"MULTIMODAL_FUSION_EFFECTIVE", "CLASSIFIER_VARIANCE_BOTTLENECK"}:
            multimodal = "effective"
        elif d_decision == "MULTIMODAL_FUSION_INEFFECTIVE":
            multimodal = "ineffective"
        else:
            multimodal = "inconclusive"
    else:
        d_decision = "not triggered"
        multimodal = "inconclusive"
    if b_decision == "MLP_BETTER_THAN_DIFFORMER":
        difformer, classifier = "harmful", "bottleneck"
    elif b_decision == "DIFFORMER_RELATION_USEFUL":
        difformer, classifier = "useful", "sufficient"
    elif b_decision == "PEE_HEAD_BEST":
        difformer, classifier = "unnecessary", "variance-limited"
    elif b_decision == "CLASSIFIER_NOT_BOTTLENECK":
        difformer, classifier = "unnecessary", "sufficient"
    else:
        difformer, classifier = "inconclusive", "sufficient"
    if d_decision == "CLASSIFIER_VARIANCE_BOTTLENECK":
        difformer, classifier = "unnecessary", "variance-limited"
    c_decision = stage_c.get("decision", "not run")
    graph_mapping = {
        "OLD_GRAPH_BAD_NEW_SPARSE_GRAPH_USEFUL": "useful",
        "GLOBAL_ATTENTION_USEFUL_GRAPH_BAD": "graph construction bad",
        "RELATION_NOT_USEFUL": "unnecessary",
        "CROSS_SUBJECT_PROPAGATION_HARMFUL": "unnecessary",
    }
    graph = graph_mapping.get(c_decision, "inconclusive")
    table = [
        {"component": "Multimodal learning", "conclusion": multimodal, "evidence": d_decision},
        {"component": "DIFFormer head", "conclusion": difformer, "evidence": b_decision},
        {"component": "Explicit graph", "conclusion": graph, "evidence": c_decision},
        {"component": "Classification head", "conclusion": classifier, "evidence": b_decision},
    ]
    if b_decision == "PEE_HEAD_BEST":
        recommendation = "REPLACE_DIFFORMER_WITH_PEE_HEAD"
    elif b_decision == "MLP_BETTER_THAN_DIFFORMER":
        recommendation = "REPLACE_DIFFORMER_WITH_MLP"
    elif c_decision == "OLD_GRAPH_BAD_NEW_SPARSE_GRAPH_USEFUL":
        recommendation = "REBUILD_GRAPH_WITH_SPARSE_RESIDUAL_GCN"
    elif d_decision == "EXTRA_MODAL_SIGNAL_LIMITED":
        recommendation = "EXTRA_MODAL_SIGNAL_LIMITED_KEEP_SIMPLE_MODEL"
    else:
        recommendation = "KEEP_DIFFORMER_AND_IMPROVE_MULTIMODAL"
    return table, recommendation


def bottleneck_priority(conclusions: list[dict]) -> list[dict]:
    priority = {
        "bottleneck": 0,
        "variance-limited": 0,
        "ineffective": 1,
        "cognitive-dominant": 1,
        "harmful": 2,
        "graph construction bad": 2,
        "useful": 3,
        "unnecessary": 4,
        "sufficient": 4,
        "effective": 4,
        "inconclusive": 5,
    }
    ordered = sorted(
        conclusions,
        key=lambda row: (priority.get(row["conclusion"], 5), row["component"]),
    )
    return [
        {
            "rank": rank,
            "component": row["component"],
            "conclusion": row["conclusion"],
            "evidence": row["evidence"],
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def render_final_report(payload: dict) -> str:
    lines = [
        f"# {MODEL_NAME}",
        "",
        "## Component conclusions",
        "",
        "| Component | Conclusion | Evidence |",
        "|---|---|---|",
        *[
            f"| {row['component']} | {row['conclusion']} | {row['evidence']} |"
            for row in payload["component_conclusions"]
        ],
        "",
        f"Recommended direction: **{payload['recommendation']}**",
        "",
        "## Original anchor",
        "",
        f"Correct={payload['original_anchor']['metrics']['correct']}/598; "
        f"ACC={payload['original_anchor']['metrics']['acc']:.10f}; "
        f"Macro-F1={payload['original_anchor']['metrics']['macro_f1']:.10f}; "
        f"BACC={payload['original_anchor']['metrics']['bacc']:.10f}; "
        f"Probability Macro-AUC={payload['original_anchor']['metrics']['macro_auc']:.10f}.",
        "",
        "Confusion matrix:",
        "",
        "```text",
        *[str(row) for row in payload["original_anchor"]["metrics"]["confusion_matrix"]],
        "```",
        "",
        "## Stage A: modality inference",
        "",
        "| Setting | Correct | Delta Correct | ACC | Macro-F1 | BACC | Prob AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['setting']} | {row['correct']} | {row['delta_correct']} | "
            f"{row['acc']:.7f} | {row['macro_f1']:.7f} | {row['bacc']:.7f} | {row['macro_auc']:.7f} |"
            for row in payload["stage_a"]["table"]
        ],
        "",
        "## Stage B: frozen heads",
        "",
        "| Head | Correct | ACC | Macro-F1 | BACC | Prob AUC | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['head']} | {row['correct']} | {row['acc']:.7f} | "
            f"{row['macro_f1']:.7f} | {row['bacc']:.7f} | {row['macro_auc']:.7f} | {row['parameter_count']} |"
            for row in payload["stage_b"]["table"]
        ],
        "",
        f"Stage B decision: **{payload['stage_b']['decision']}**",
        "",
        "PEE ensemble diagnostics:",
        "",
        f"- Ensemble: {payload['stage_b']['results']['pee_head']['metrics']['correct']}/598, "
        f"ACC={payload['stage_b']['results']['pee_head']['metrics']['acc']:.7f}",
        f"- Pairwise prediction disagreement: "
        f"{payload['stage_b']['results']['pee_head']['member_diagnostics']['pairwise_prediction_disagreement']:.7f}",
        f"- Pairwise probability cosine: "
        f"{payload['stage_b']['results']['pee_head']['member_diagnostics']['pairwise_probability_cosine']:.7f}",
        f"- Pairwise symmetric KL: "
        f"{payload['stage_b']['results']['pee_head']['member_diagnostics']['pairwise_symmetric_kl']:.7f}",
        "",
        "## Stage C: graph localization",
        "",
        f"Status={payload['stage_c']['status']}; decision={payload['stage_c'].get('decision', 'not run')}.",
        "",
        "## Stage D: multimodal capacity",
        "",
        f"Status={payload['stage_d']['status']}; decision={payload['stage_d'].get('decision', 'not run')}.",
        "",
        "## Bottleneck priority",
        "",
        *[
            f"{row['rank']}. {row['component']}: {row['conclusion']} ({row['evidence']})"
            for row in payload["bottleneck_priority"]
        ],
        "",
        "## Runtime",
        "",
        f"Branch={payload['branch']}; source commit={payload['source_commit']}; "
        f"device={payload['environment']['gpu']}; total seconds={payload['elapsed_seconds']:.3f}.",
        "",
    ]
    if "member_metrics" in payload["stage_b"]["results"]["pee_head"]:
        lines.extend(["## PEE members", ""])
        for name, metrics in payload["stage_b"]["results"]["pee_head"]["member_metrics"].items():
            lines.append(f"- {name}: {metrics['correct']}/598, ACC={metrics['acc']:.7f}")
        lines.append("")
    if payload["stage_c"].get("status") == "completed":
        lines.extend(
            [
                "## Stage C metrics",
                "",
                "| Arm | Correct | ACC | Macro-F1 | BACC | Prob AUC | Gamma |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("C0_residual_mlp", "C0 Residual MLP"),
            ("C1_frozen_difformer", "C1 DIFFormer no graph"),
            ("C2_sparse_residual_gcn", "C2 Sparse Residual GCN"),
        ):
            result = payload["stage_c"][key]
            metrics = result["metrics"]
            gamma = result.get("gamma_mean", "-")
            gamma_text = f"{gamma:.7f}" if isinstance(gamma, (int, float)) else str(gamma)
            lines.append(
                f"| {label} | {metrics['correct']} | {metrics['acc']:.7f} | "
                f"{metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | "
                f"{metrics['macro_auc']:.7f} | {gamma_text} |"
            )
        lines.append("")
    if payload["stage_d"].get("status") == "completed":
        lines.extend(
            [
                "## Stage D metrics",
                "",
                "| Arm | Correct | ACC | Macro-F1 | BACC | Prob AUC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("D0_COG_only_MLP", "D0 COG-only MLP"),
            ("D1_Raw_All_MLP", "D1 Raw-All MLP"),
            ("D2_Current_Multimodal", "D2 Current Multimodal"),
        ):
            result = payload["stage_d"][key]
            metrics = result["metrics"]
            lines.append(
                f"| {label} | {metrics['correct']} | {metrics['acc']:.7f} | "
                f"{metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | "
                f"{metrics['macro_auc']:.7f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_formal_all() -> dict:
    final_path = ROOT / RESULTS_REL / "FINAL_REPORT.json"
    if final_path.is_file():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        print("Final report saved", flush=True)
        return payload
    suite_started = time.perf_counter()
    context = load_context()
    anchor = verify_anchor_source(context)
    smoke_path = ROOT / RESULTS_REL / "smoke" / "summary.json"
    if not smoke_path.is_file():
        run_smoke()
    else:
        print("Original anchor reproduced", flush=True)
    stage_a = run_stage_a(context)
    stage_b = run_stage_b(context)
    stage_c = run_stage_c(context, stage_b)
    stage_d = run_stage_d(context, stage_a, stage_b)
    table, recommendation = component_conclusions(stage_a, stage_b, stage_c, stage_d)
    priority = bottleneck_priority(table)
    payload = {
        "model": MODEL_NAME,
        "branch": context["branch"],
        "source_commit": git_value("rev-parse", "HEAD"),
        "original_anchor": anchor,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": stage_c,
        "stage_d": stage_d,
        "component_conclusions": table,
        "bottleneck_priority": priority,
        "recommendation": recommendation,
        "environment": environment_snapshot(context["device"]),
        "commands": {
            "smoke": "python scripts/run_component_localization_v1.py --smoke",
            "formal": "python scripts/run_component_localization_v1.py --formal-all",
        },
        "elapsed_seconds": time.perf_counter() - suite_started,
    }
    write_json(final_path, payload)
    (ROOT / RESULTS_REL / "FINAL_REPORT.md").write_text(
        render_final_report(payload), encoding="utf-8"
    )
    reloaded = json.loads(final_path.read_text(encoding="utf-8"))
    require(reloaded["recommendation"] == recommendation, "Final report readback failed")
    print("Final report saved", flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the two-epoch fold-0 all-head CUDA smoke test")
    mode.add_argument("--formal-all", action="store_true", help="Run Stages A-D with resume support and save the final report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_formal_all()


if __name__ == "__main__":
    main()
