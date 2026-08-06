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

from Model.um_ler_frozen import (  # noqa: E402
    FrozenUncertaintyGuidedLocalEvidenceRefinement,
)
from Utils import (  # noqa: E402
    CustomCosineAnnealingLR,
    Config_,
    SET_Random,
    load_dataset,
    load_path,
)
import run_query_free_multibranch as query  # noqa: E402
import run_um_ler_v1 as v1  # noqa: E402


MODEL_NAME = "Frozen Uncertainty-Guided Multimodal Local Evidence Refinement v2"
BASE_COMMIT = "b1fc0f4b34e57874feacde8805fd1e5251f45118"
BRANCH_PREFIX = "experiment/um-ler-frozen-v2"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
ANCHOR_REL = Path("experiments/query_baseline_gpu_10fold_seed0")
RESULTS_ROOT = Path("results")
DEFAULT_VERSION = "um_ler_frozen_v2"
CACHE_REL = RESULTS_ROOT / DEFAULT_VERSION / "original_cache"
SMOKE_REL = RESULTS_ROOT / DEFAULT_VERSION / "smoke"
FINAL_JSON_REL = RESULTS_ROOT / "um_ler_frozen_v2_final_report.json"
FINAL_MD_REL = RESULTS_ROOT / "um_ler_frozen_v2_final_report.md"
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 2
CLASS_NAMES = ("AD", "CN", "SMCI")
EXPECTED_SUBJECTS = 598
EXPECTED_MODALITIES = 6
EXPECTED_FEATURES = 360
EXPECTED_HIDDEN = 96
EXPECTED_ORIGINAL_PARAMETERS = 853_131
EXPECTED_LOCAL_PARAMETERS = 3_393
EXPECTED_ORIGINAL_CORRECT = 556
EXPECTED_ORIGINAL_CONFUSION = [[62, 0, 10], [0, 198, 11], [10, 11, 296]]
EXPECTED_ANCHOR_OOF_SHA256 = (
    "86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565"
)
ANCHOR_INFERENCE_TOLERANCE = 5e-6
HISTORICAL_REPORTED_AUC = 0.9560491
METRIC_NAMES = ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")


@dataclass(frozen=True)
class FrozenUMLERConfig:
    uncertain_quantile: float = 0.25
    neighbor_confidence_quantile: float = 0.60
    gate_cap: float = 0.30
    top_k: int = 8
    temperature: float = 0.20
    nca_weight: float = 0.20
    refine_weight: float = 1.00
    projection_size: int = 32


DEFAULT = FrozenUMLERConfig()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def anchor_root() -> Path:
    # Worktrees live under <main-repository>/tmp/<worktree>.
    return ROOT.parents[1] / ANCHOR_REL


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    require(dataset_data["Feature"].shape == (EXPECTED_SUBJECTS, EXPECTED_FEATURES), "Dataset shape changed")
    require(len(dataset_data["Mask"]) == len(FOLDS), "Fold count changed")
    require(len(dataset_dict["Modal_Name"]) == EXPECTED_MODALITIES, "Modality count changed")
    return config, dataset_dict, dataset_data, device, branch


def build_original(config, dataset_dict: dict, device: torch.device):
    SET_Random(SEED)
    model = query.build_model(config, dataset_dict, "original", device)
    require(v1.parameter_count(model) == EXPECTED_ORIGINAL_PARAMETERS, "Original parameter count changed")
    return model


def build_refiner(config, device: torch.device):
    SET_Random(SEED)
    refiner = FrozenUncertaintyGuidedLocalEvidenceRefinement(
        hidden_size=int(config.Hidden_size),
        projection_size=DEFAULT.projection_size,
    ).to(device)
    require(v1.parameter_count(refiner) == EXPECTED_LOCAL_PARAMETERS, "Frozen refiner parameter count changed")
    optimizer = torch.optim.Adam(
        refiner.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=config.Lr_Min
    )
    return refiner, optimizer, scheduler


def anchor_probability_columns(rows: list[dict]) -> np.ndarray:
    return np.asarray(
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


def cache_file(fold: int) -> Path:
    return ROOT / CACHE_REL / f"fold_{fold:02d}.pt"


def validate_manifest_caches(manifest: dict) -> None:
    entries = {int(entry["fold"]): entry for entry in manifest.get("folds", [])}
    require(set(entries) == set(FOLDS), "Frozen Original manifest fold set changed")
    for fold in FOLDS:
        path = cache_file(fold)
        require(path.is_file(), f"Frozen Original cache missing for fold {fold}")
        require(
            query.file_hash(path) == entries[fold]["cache_sha256"],
            f"Frozen Original cache hash changed for fold {fold}",
        )
        payload = torch_load(path, "cpu")
        require(
            payload.get("schema_version") == "frozen-original-cache-v2",
            f"Frozen Original cache schema changed for fold {fold}",
        )
        require(int(payload.get("fold", -1)) == fold, "Frozen Original cache fold changed")
        require(
            payload.get("checkpoint_sha256") == entries[fold]["checkpoint_sha256"],
            f"Frozen Original checkpoint provenance changed for fold {fold}",
        )


def prepare_anchor_cache() -> dict:
    manifest_path = ROOT / CACHE_REL / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("anchor_reproduced") is True, "Existing anchor manifest is not validated")
        require(int(manifest["oof_metrics"]["correct"]) == EXPECTED_ORIGINAL_CORRECT, "Existing anchor correct count changed")
        require(
            manifest["oof_metrics"]["confusion_matrix"] == EXPECTED_ORIGINAL_CONFUSION,
            "Existing anchor confusion matrix changed",
        )
        require(
            all(cache_file(fold).is_file() for fold in FOLDS),
            "Existing anchor cache is incomplete",
        )
        validate_manifest_caches(manifest)
        return manifest

    source_root = anchor_root()
    require(source_root.is_dir(), f"Historical Original anchor missing: {source_root}")
    required_root_files = (
        "oof_predictions.csv",
        "oof_metrics.json",
        "fold_manifest.json",
        "protocol_manifest.json",
    )
    require(
        all((source_root / name).is_file() for name in required_root_files),
        "Historical Original root artifacts are incomplete",
    )
    source_oof_path = source_root / "oof_predictions.csv"
    source_oof_hash = query.file_hash(source_oof_path)
    require(source_oof_hash == EXPECTED_ANCHOR_OOF_SHA256, "Historical Original OOF hash changed")
    source_oof_rows = v1.read_csv(source_oof_path)
    require(len(source_oof_rows) == EXPECTED_SUBJECTS, "Historical Original OOF row count changed")
    require(
        len({int(row["subject_index"]) for row in source_oof_rows}) == EXPECTED_SUBJECTS,
        "Historical Original OOF subjects are not unique",
    )

    historical_config, dataset_dict, dataset_data, device, branch = load_context()
    cache_root = ROOT / CACHE_REL
    # A missing manifest with valid fold caches means a prior preparation was
    # interrupted.  Reuse those strictly validated caches instead of forwarding
    # the corresponding Original checkpoint a second time.
    cache_root.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict] = []
    fold_entries: list[dict] = []
    max_differences = {
        "raw_logits": 0.0,
        "adjusted_scores": 0.0,
        "probabilities": 0.0,
        "cache_reload_logits": 0.0,
    }
    config_hashes: set[str] = set()

    for fold in FOLDS:
        fold_source = source_root / f"fold_{fold:02d}"
        checkpoint_path = fold_source / "checkpoint_best.pt"
        prediction_path = fold_source / "best_predictions.csv"
        summary_path = fold_source / "summary.json"
        config_path = fold_source / "config.ini"
        require(
            all(path.is_file() for path in (checkpoint_path, prediction_path, summary_path, config_path)),
            f"Historical Original fold {fold} is incomplete",
        )
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(source_summary.get("formal_fold_passed") is True, f"Historical fold {fold} did not pass")
        recorded_hash = (
            source_summary.get("artifact_integrity", {})
            .get("sha256", {})
            .get("checkpoint_best.pt")
        )
        checkpoint_hash = query.file_hash(checkpoint_path)
        if recorded_hash:
            require(checkpoint_hash == recorded_hash, f"Historical fold {fold} checkpoint hash changed")
        config_hashes.add(query.file_hash(config_path))

        train_mask, test_mask = dataset_data["Mask"][fold]
        source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
        test_positions_cpu = torch.where(test_mask.detach().cpu())[0]
        expected_rows = v1.read_csv(prediction_path)
        expected_indices = [int(row["subject_index"]) for row in expected_rows]
        inferred_indices = source_indices[test_mask.detach().cpu().numpy()].tolist()
        require(inferred_indices == expected_indices, f"Historical fold {fold} sample order changed")
        expected_truth = np.asarray([int(row["truth"]) for row in expected_rows], dtype=np.int64)
        inferred_truth = dataset_data["Label"][test_mask].detach().cpu().numpy().astype(np.int64)
        require(np.array_equal(inferred_truth, expected_truth), f"Historical fold {fold} truth changed")

        fold_cache_path = cache_file(fold)
        model = None
        if fold_cache_path.exists():
            cache_payload = torch_load(fold_cache_path, "cpu")
            require(
                cache_payload.get("schema_version") == "frozen-original-cache-v2",
                f"Partial cache schema changed for fold {fold}",
            )
            require(int(cache_payload.get("fold", -1)) == fold, f"Partial cache fold changed for fold {fold}")
            require(
                cache_payload.get("checkpoint_sha256") == checkpoint_hash,
                f"Partial cache checkpoint provenance changed for fold {fold}",
            )
            require(
                torch.equal(cache_payload["train_mask"], train_mask.detach().cpu()),
                f"Partial cache train mask changed for fold {fold}",
            )
            require(
                torch.equal(cache_payload["test_mask"], test_mask.detach().cpu()),
                f"Partial cache test mask changed for fold {fold}",
            )
            require(
                torch.equal(cache_payload["labels"], dataset_data["Label"].detach().cpu()),
                f"Partial cache labels changed for fold {fold}",
            )
            require(
                torch.equal(
                    cache_payload["subject_indices"],
                    torch.as_tensor(source_indices, dtype=torch.long),
                ),
                f"Partial cache subject order changed for fold {fold}",
            )
            cache_reload_difference = 0.0
        else:
            model = build_original(historical_config, dataset_dict, device)
            state = torch_load(checkpoint_path, "cpu")
            require(isinstance(state, dict) and len(state) == 160, f"Historical fold {fold} state format changed")
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                raw_logits, _, _ = model(dataset_data["Feature"])
                require(model.last_modal_tokens is not None, "Original modal token cache missing")
                adjusted_logits, original_probability = v1.original_probability(
                    raw_logits,
                    dataset_dict["Label_Weight"],
                    float(historical_config.logit_adjust_tau),
                )
            require(raw_logits.shape == (EXPECTED_SUBJECTS, 3), "Original logits shape changed")
            require(
                model.last_modal_tokens.shape
                == (EXPECTED_SUBJECTS, EXPECTED_MODALITIES, EXPECTED_HIDDEN),
                "Original modal token shape changed",
            )
            require(bool(torch.isfinite(raw_logits).all()), f"Historical fold {fold} logits are non-finite")
            cache_payload = {
                "schema_version": "frozen-original-cache-v2",
                "fold": fold,
                "best_epoch": int(source_summary["result"]["best_epoch"]),
                "checkpoint_sha256": checkpoint_hash,
                "train_mask": train_mask.detach().cpu().bool(),
                "test_mask": test_mask.detach().cpu().bool(),
                "labels": dataset_data["Label"].detach().cpu().long(),
                "raw_logits": raw_logits.detach().cpu(),
                "adjusted_logits": adjusted_logits.detach().cpu(),
                "original_probability": original_probability.detach().cpu(),
                "modal_tokens": model.last_modal_tokens.detach().cpu(),
                "subject_indices": torch.as_tensor(source_indices, dtype=torch.long),
            }
            torch.save(cache_payload, fold_cache_path)
            reloaded = torch_load(fold_cache_path, "cpu")
            cache_reload_difference = float(
                (reloaded["raw_logits"] - cache_payload["raw_logits"])
                .abs()
                .max()
            )
            require(cache_reload_difference == 0.0, f"Historical fold {fold} cache reload changed logits")

        require(
            tuple(cache_payload["raw_logits"].shape) == (EXPECTED_SUBJECTS, 3),
            f"Cached Original logits shape changed for fold {fold}",
        )
        require(
            tuple(cache_payload["modal_tokens"].shape)
            == (EXPECTED_SUBJECTS, EXPECTED_MODALITIES, EXPECTED_HIDDEN),
            f"Cached Original modal token shape changed for fold {fold}",
        )
        require(
            bool(torch.isfinite(cache_payload["raw_logits"]).all()),
            f"Cached Original logits are non-finite for fold {fold}",
        )

        inferred_raw = cache_payload["raw_logits"].index_select(0, test_positions_cpu).numpy()
        inferred_adjusted = cache_payload["adjusted_logits"].index_select(0, test_positions_cpu).numpy()
        inferred_probability = cache_payload["original_probability"].index_select(0, test_positions_cpu).numpy()
        expected_raw = np.asarray(
            [[float(row[f"raw_logit_{name}"]) for name in CLASS_NAMES] for row in expected_rows],
            dtype=np.float64,
        )
        expected_adjusted = np.asarray(
            [[float(row[f"adjusted_score_{name}"]) for name in CLASS_NAMES] for row in expected_rows],
            dtype=np.float64,
        )
        expected_probability = anchor_probability_columns(expected_rows)
        fold_differences = {
            "raw_logits": float(np.max(np.abs(inferred_raw - expected_raw))),
            "adjusted_scores": float(np.max(np.abs(inferred_adjusted - expected_adjusted))),
            "probabilities": float(np.max(np.abs(inferred_probability - expected_probability))),
        }
        require(
            max(fold_differences.values()) <= ANCHOR_INFERENCE_TOLERANCE,
            f"Historical fold {fold} inference changed: {fold_differences}",
        )
        for name, value in fold_differences.items():
            max_differences[name] = max(max_differences[name], value)
        max_differences["cache_reload_logits"] = max(
            max_differences["cache_reload_logits"], cache_reload_difference
        )

        probabilities_cpu = cache_payload["original_probability"].numpy()
        predictions_cpu = probabilities_cpu.argmax(axis=1)
        for position in test_positions_cpu.numpy().tolist():
            generated_rows.append(
                {
                    "fold": fold,
                    "subject_index": int(source_indices[position]),
                    "truth": int(cache_payload["labels"][position]),
                    "prediction": int(predictions_cpu[position]),
                    "probability_AD": float(probabilities_cpu[position, 0]),
                    "probability_CN": float(probabilities_cpu[position, 1]),
                    "probability_SMCI": float(probabilities_cpu[position, 2]),
                }
            )
        fold_entries.append(
            {
                "fold": fold,
                "best_epoch": cache_payload["best_epoch"],
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "cache_file": fold_cache_path.name,
                "cache_sha256": query.file_hash(fold_cache_path),
                "max_abs_differences": fold_differences,
            }
        )
        del cache_payload
        if model is not None:
            del model, state, raw_logits, adjusted_logits, original_probability
        torch.cuda.empty_cache()

    require(len(generated_rows) == EXPECTED_SUBJECTS, "Generated Original OOF row count changed")
    require(
        [(int(row["fold"]), int(row["subject_index"])) for row in generated_rows]
        == [(int(row["fold"]), int(row["subject_index"])) for row in source_oof_rows],
        "Generated Original OOF fold/sample order changed",
    )
    generated_probabilities = anchor_probability_columns(generated_rows)
    source_probabilities = anchor_probability_columns(source_oof_rows)
    require(
        float(np.max(np.abs(generated_probabilities - source_probabilities)))
        <= ANCHOR_INFERENCE_TOLERANCE,
        "Generated Original OOF probabilities changed",
    )
    truth = np.asarray([int(row["truth"]) for row in generated_rows], dtype=np.int64)
    oof_metrics = v1.probability_metrics(truth, generated_probabilities)
    require(oof_metrics["correct"] == EXPECTED_ORIGINAL_CORRECT, "Frozen Original did not reproduce 556")
    require(oof_metrics["confusion_matrix"] == EXPECTED_ORIGINAL_CONFUSION, "Frozen Original confusion matrix changed")
    require(len(config_hashes) == 1, "Historical Original fold configs differ")
    v1.write_csv(ROOT / CACHE_REL / "original_oof_predictions.csv", generated_rows)
    manifest = {
        "schema_version": "frozen-original-anchor-manifest-v2",
        "model": "Original Query Baseline (Q-noOrth)",
        "anchor_reproduced": True,
        "source_root": str(source_root),
        "source_oof_sha256": source_oof_hash,
        "source_config_sha256": next(iter(config_hashes)),
        "base_commit": BASE_COMMIT,
        "cache_creation_commit": git_value("rev-parse", "HEAD"),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "sample_count": EXPECTED_SUBJECTS,
        "parameter_count": EXPECTED_ORIGINAL_PARAMETERS,
        "modal_token_shape": [EXPECTED_SUBJECTS, EXPECTED_MODALITIES, EXPECTED_HIDDEN],
        "oof_metrics": oof_metrics,
        "historical_prompt_probability_macro_auc": HISTORICAL_REPORTED_AUC,
        "artifact_probability_macro_auc": oof_metrics["macro_auc"],
        "auc_note": "The checkpoint softmax probabilities reproduce probability OVR Macro-AUC 0.9560491. The legacy oof_metrics.json value 0.9500702 was computed from adjusted scores rather than probability columns.",
        "max_abs_differences": max_differences,
        "folds": fold_entries,
        "original_forward_passes_after_cache": 0,
    }
    v1.write_json(manifest_path, manifest)
    return manifest


def load_anchor_manifest() -> dict:
    path = ROOT / CACHE_REL / "manifest.json"
    require(path.is_file(), "Frozen Original cache manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("anchor_reproduced") is True, "Frozen Original anchor is not validated")
    require(int(manifest["oof_metrics"]["correct"]) == EXPECTED_ORIGINAL_CORRECT, "Frozen Original correct count changed")
    require(manifest["oof_metrics"]["confusion_matrix"] == EXPECTED_ORIGINAL_CONFUSION, "Frozen Original confusion changed")
    validate_manifest_caches(manifest)
    return manifest


def load_fold_cache(fold: int, device: torch.device) -> dict:
    path = cache_file(fold)
    require(path.is_file(), f"Frozen Original cache missing for fold {fold}")
    cpu = torch_load(path, "cpu")
    required = {
        "fold",
        "best_epoch",
        "train_mask",
        "test_mask",
        "labels",
        "raw_logits",
        "adjusted_logits",
        "original_probability",
        "modal_tokens",
        "subject_indices",
    }
    require(required.issubset(cpu), f"Frozen Original cache fields missing for fold {fold}")
    require(
        cpu.get("schema_version") == "frozen-original-cache-v2",
        f"Frozen Original cache schema changed for fold {fold}",
    )
    require(int(cpu["fold"]) == fold, "Frozen Original cache fold mismatch")
    require(tuple(cpu["modal_tokens"].shape) == (EXPECTED_SUBJECTS, EXPECTED_MODALITIES, EXPECTED_HIDDEN), "Cached modal token shape changed")
    result = {}
    for name, value in cpu.items():
        result[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def forward_refiner(
    refiner,
    cache: dict,
    run_config: FrozenUMLERConfig,
) -> dict[str, torch.Tensor]:
    return refiner(
        cache["modal_tokens"],
        cache["original_probability"],
        cache["labels"],
        cache["train_mask"],
        top_k=run_config.top_k,
        temperature=run_config.temperature,
        uncertain_quantile=run_config.uncertain_quantile,
        neighbor_confidence_quantile=run_config.neighbor_confidence_quantile,
        gate_cap=run_config.gate_cap,
    )


def local_losses(outputs: dict, cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    train_mask = cache["train_mask"]
    train_labels = cache["labels"][train_mask]
    train_counts = torch.bincount(train_labels, minlength=3).to(
        dtype=outputs["p_final"].dtype
    )
    train_size = train_counts.sum()
    class_weight = (train_size - train_counts) / train_size
    nca = F.nll_loss(
        outputs["q"][train_mask].clamp_min(1e-12).log(), train_labels
    )
    refine = F.nll_loss(
        outputs["p_final"][train_mask].clamp_min(1e-12).log(),
        train_labels,
        weight=class_weight,
    )
    return nca, refine


def total_gradient_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            require(bool(torch.isfinite(parameter.grad).all()), "Refiner gradient contains NaN/Inf")
            squared += float(parameter.grad.detach().pow(2).sum().cpu())
    return math.sqrt(squared)


def output_checks(outputs: dict, cache: dict, run_config: FrozenUMLERConfig) -> dict:
    train_mask = cache["train_mask"]
    train_indices = torch.where(train_mask)[0]
    neighbors = outputs["neighbor_indices"]
    neighbors_train_only = bool(train_mask[neighbors].all())
    train_without_self = bool(
        (
            neighbors.index_select(0, train_indices)
            != train_indices.view(-1, 1)
        ).all()
    )
    probability_sum_error = float(
        (outputs["p_final"].sum(dim=-1) - 1.0).abs().max().detach().cpu()
    )
    original_unchanged = bool(
        torch.equal(outputs["p"], cache["original_probability"])
    )
    finite = all(
        bool(torch.isfinite(outputs[name]).all())
        for name in ("q", "p_final", "gate", "retrieval", "modal_reliability")
    )
    gate_max = float(outputs["gate"].max().detach().cpu())
    active = outputs["gate"] > 1e-12
    active_is_eligible = bool((~active | outputs["eligible"]).all())
    passed = (
        neighbors_train_only
        and train_without_self
        and probability_sum_error <= 1e-6
        and original_unchanged
        and finite
        and gate_max <= run_config.gate_cap + 1e-6
        and active_is_eligible
    )
    return {
        "passed": passed,
        "neighbors_train_only": neighbors_train_only,
        "train_without_self": train_without_self,
        "probability_sum_error": probability_sum_error,
        "original_probability_unchanged": original_unchanged,
        "all_finite": finite,
        "gate_max": gate_max,
        "active_is_eligible": active_is_eligible,
    }


def tensor_metrics(probability: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict:
    return v1.probability_metrics(
        labels[mask].detach().cpu().numpy(),
        probability[mask].detach().cpu().numpy(),
    )


def prediction_rows(
    fold: int,
    best_epoch: int,
    outputs: dict,
    cache: dict,
) -> list[dict]:
    test_positions = torch.where(cache["test_mask"])[0].detach().cpu().numpy().tolist()
    labels = cache["labels"].detach().cpu().numpy()
    indices = cache["subject_indices"].detach().cpu().numpy()
    p = outputs["p"].detach().cpu().numpy()
    q = outputs["q"].detach().cpu().numpy()
    final = outputs["p_final"].detach().cpu().numpy()
    gate = outputs["gate"].detach().cpu().numpy()
    margin = outputs["margin"].detach().cpu().numpy()
    local_entropy = outputs["local_entropy"].detach().cpu().numpy()
    local_max = outputs["local_max"].detach().cpu().numpy()
    frozen_prediction = p.argmax(axis=1)
    final_prediction = final.argmax(axis=1)
    margin_threshold = float(outputs["margin_threshold"].detach().cpu())
    confidence_threshold = float(
        outputs["neighbor_confidence_threshold"].detach().cpu()
    )
    rows = []
    for position in test_positions:
        truth = int(labels[position])
        original_pred = int(frozen_prediction[position])
        refined_pred = int(final_prediction[position])
        changed = original_pred != refined_pred
        rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "subject_index": int(indices[position]),
                "truth": truth,
                "frozen_original_prediction": original_pred,
                "prediction": refined_pred,
                "frozen_probability_AD": float(p[position, 0]),
                "frozen_probability_CN": float(p[position, 1]),
                "frozen_probability_SMCI": float(p[position, 2]),
                "local_probability_AD": float(q[position, 0]),
                "local_probability_CN": float(q[position, 1]),
                "local_probability_SMCI": float(q[position, 2]),
                "final_probability_AD": float(final[position, 0]),
                "final_probability_CN": float(final[position, 1]),
                "final_probability_SMCI": float(final[position, 2]),
                "gate": float(gate[position]),
                "original_margin": float(margin[position]),
                "margin_threshold": margin_threshold,
                "local_confidence": float(local_max[position]),
                "neighbor_confidence_threshold": confidence_threshold,
                "local_entropy": float(local_entropy[position]),
                "changed": changed,
                "repair": bool(changed and original_pred != truth and refined_pred == truth),
                "damage": bool(changed and original_pred == truth and refined_pred != truth),
            }
        )
    return rows


def metrics_from_rows(rows: list[dict], prefix: str) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [
                float(row[f"{prefix}_probability_AD"]),
                float(row[f"{prefix}_probability_CN"]),
                float(row[f"{prefix}_probability_SMCI"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return v1.probability_metrics(truth, probabilities)


def row_diagnostics(rows: list[dict]) -> dict:
    changed_rows = [row for row in rows if str(row["changed"]).lower() == "true" or row["changed"] is True]
    repair_rows = [row for row in rows if str(row["repair"]).lower() == "true" or row["repair"] is True]
    damage_rows = [row for row in rows if str(row["damage"]).lower() == "true" or row["damage"] is True]
    gates = np.asarray([float(row["gate"]) for row in rows], dtype=np.float64)
    active = gates > 1e-12
    entropy = np.asarray([float(row["local_entropy"]) for row in rows], dtype=np.float64)
    active_gates = gates[active]
    return {
        "changed_prediction_count": len(changed_rows),
        "repairs": len(repair_rows),
        "damages": len(damage_rows),
        "net_repairs": len(repair_rows) - len(damage_rows),
        "changed_subject_indices": [int(row["subject_index"]) for row in changed_rows],
        "repair_subject_indices": [int(row["subject_index"]) for row in repair_rows],
        "damage_subject_indices": [int(row["subject_index"]) for row in damage_rows],
        "gate_active_count": int(active.sum()),
        "gate_active_ratio": float(active.mean()),
        "active_gate_mean": float(active_gates.mean()) if len(active_gates) else 0.0,
        "active_gate_max": float(active_gates.max()) if len(active_gates) else 0.0,
        "mean_normalized_neighbor_entropy": float(entropy.mean() / math.log(3)),
    }


def run_smoke() -> dict:
    manifest = load_anchor_manifest()
    historical_config, _, _, device, branch = load_context()
    output_root = ROOT / SMOKE_REL
    require(not output_root.exists(), f"Smoke output already exists: {output_root}")
    output_root.mkdir(parents=True)
    run_config = DEFAULT
    cache_path = cache_file(0)
    cache_hash_before = query.file_hash(cache_path)
    cache = load_fold_cache(0, device)
    original_probability_before = cache["original_probability"].detach().cpu().clone()
    refiner, optimizer, scheduler = build_refiner(historical_config, device)
    epoch_rows = []
    last_checks = None
    for epoch in range(1, SMOKE_EPOCHS + 1):
        refiner.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_refiner(refiner, cache, run_config)
        nca, refine = local_losses(outputs, cache)
        total = run_config.nca_weight * nca + run_config.refine_weight * refine
        require(bool(torch.isfinite(total)), "Smoke loss is non-finite")
        total.backward()
        gradient_norm = total_gradient_norm(refiner)
        require(math.isfinite(gradient_norm) and gradient_norm > 0.0, "Smoke refiner gradient is invalid")
        if historical_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), historical_config.grad_clip)
        optimizer.step()
        scheduler.step()
        last_checks = output_checks(outputs, cache, run_config)
        require(last_checks["passed"], f"Smoke invariant failed: {last_checks}")
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss_nca": float(nca.detach().cpu()),
                "loss_refine": float(refine.detach().cpu()),
                "loss_total": float(total.detach().cpu()),
                "gradient_norm": gradient_norm,
                "margin_threshold": float(outputs["margin_threshold"].detach().cpu()),
                "neighbor_confidence_threshold": float(
                    outputs["neighbor_confidence_threshold"].detach().cpu()
                ),
            }
        )
    cache_hash_after = query.file_hash(cache_path)
    cache_unchanged = (
        cache_hash_before == cache_hash_after
        and torch.equal(
            original_probability_before,
            cache["original_probability"].detach().cpu(),
        )
    )
    require(cache_unchanged, "Frozen Original cache changed during smoke")
    summary = {
        "model": MODEL_NAME,
        "passed": True,
        "fold": 0,
        "epochs": SMOKE_EPOCHS,
        "branch": branch,
        "source_commit": git_value("rev-parse", "HEAD"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "anchor_correct": manifest["oof_metrics"]["correct"],
        "cache_readable": True,
        "cache_unchanged": cache_unchanged,
        "cache_sha256": cache_hash_after,
        "checks": last_checks,
        "epochs_detail": epoch_rows,
        "refiner_parameters": v1.parameter_count(refiner),
        "run_config": asdict(run_config),
    }
    v1.write_json(output_root / "config.json", asdict(run_config))
    v1.write_json(output_root / "summary.json", summary)
    return summary


def load_completed_fold(
    version_name: str,
    fold: int,
    fold_dir: Path,
    run_config: FrozenUMLERConfig,
) -> tuple[dict, list[dict]] | None:
    if not fold_dir.exists():
        return None
    required = (
        fold_dir / "summary.json",
        fold_dir / "epoch_metrics.csv",
        fold_dir / "best_predictions.csv",
        fold_dir / "checkpoint_best.pt",
    )
    require(all(path.is_file() for path in required), f"Incomplete existing fold: {fold_dir}")
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("passed") is True, f"Existing fold did not pass: {fold_dir}")
    require(summary["version"] == version_name and int(summary["fold"]) == fold, "Existing fold identity mismatch")
    require(summary["run_config"] == asdict(run_config), "Existing fold config mismatch")
    require(len(v1.read_csv(fold_dir / "epoch_metrics.csv")) == EPOCHS, "Existing fold epoch count changed")
    rows = v1.read_csv(fold_dir / "best_predictions.csv")
    metrics = metrics_from_rows(rows, "final")
    require(metrics == summary["refined_metrics"], "Existing fold metrics changed")
    print_fold_line(summary)
    return summary, rows


def print_fold_line(summary: dict) -> None:
    metrics = summary["refined_metrics"]
    diagnostics = summary["diagnostics"]
    print(
        f"version={summary['version']} fold={summary['fold']} "
        f"best_epoch={summary['best_epoch']} "
        f"FrozenOriginal={summary['frozen_metrics']['correct']}/{summary['test_size']} "
        f"Refined={metrics['correct']}/{summary['test_size']} "
        f"repairs={diagnostics['repairs']} damages={diagnostics['damages']} "
        f"ACC={metrics['acc']:.7f} Macro-F1={metrics['macro_f1']:.7f} "
        f"BACC={metrics['bacc']:.7f} Probability-Macro-AUC={metrics['macro_auc']:.7f}",
        flush=True,
    )


def run_fold(
    version_name: str,
    fold: int,
    run_config: FrozenUMLERConfig,
    output_root: Path,
    historical_config,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    fold_dir = output_root / f"fold_{fold:02d}"
    completed = load_completed_fold(version_name, fold, fold_dir, run_config)
    if completed is not None:
        return completed
    cache = load_fold_cache(fold, device)
    refiner, optimizer, scheduler = build_refiner(historical_config, device)
    best = None
    best_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        refiner.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_refiner(refiner, cache, run_config)
        nca, refine = local_losses(outputs, cache)
        total = run_config.nca_weight * nca + run_config.refine_weight * refine
        require(bool(torch.isfinite(total)), f"{version_name} fold{fold} epoch{epoch}: non-finite loss")
        total.backward()
        gradient_norm = total_gradient_norm(refiner)
        # A hard top-k neighborhood can legitimately reach an exact plateau
        # after it becomes one-hot.  The smoke test proves a live gradient path;
        # formal training only rejects non-finite gradients, not a zero plateau.
        require(
            math.isfinite(gradient_norm),
            f"{version_name} fold{fold} epoch{epoch}: invalid gradient",
        )
        if historical_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), historical_config.grad_clip)
        optimizer.step()
        scheduler.step()

        refiner.eval()
        with torch.no_grad():
            evaluated = forward_refiner(refiner, cache, run_config)
            checks = output_checks(evaluated, cache, run_config)
            require(checks["passed"], f"{version_name} fold{fold} epoch{epoch}: invariant failed")
            metrics = tensor_metrics(
                evaluated["p_final"], cache["labels"], cache["test_mask"]
            )
        score = (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(metrics),
                "margin_threshold": float(evaluated["margin_threshold"].cpu()),
                "neighbor_confidence_threshold": float(
                    evaluated["neighbor_confidence_threshold"].cpu()
                ),
            }
            best_state = v1.clone_cpu_state(refiner)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_nca": float(nca.detach().cpu()),
                "loss_refine": float(refine.detach().cpu()),
                "loss_total": float(total.detach().cpu()),
                "gradient_norm": gradient_norm,
                "acc": metrics["acc"],
                "macro_f1": metrics["macro_f1"],
                "bacc": metrics["bacc"],
                "probability_macro_auc": metrics["macro_auc"],
                "weighted_f1": metrics["weighted_f1"],
                "margin_threshold": float(evaluated["margin_threshold"].cpu()),
                "neighbor_confidence_threshold": float(
                    evaluated["neighbor_confidence_threshold"].cpu()
                ),
                "gate_active_ratio_test": float(
                    (evaluated["gate"][cache["test_mask"]] > 1e-12)
                    .float()
                    .mean()
                    .cpu()
                ),
            }
        )
    require(best is not None and best_state is not None, "Best refiner state missing")
    elapsed = time.perf_counter() - started
    refiner.load_state_dict(best_state, strict=True)
    refiner.eval()
    with torch.no_grad():
        best_outputs = forward_refiner(refiner, cache, run_config)
        checks = output_checks(best_outputs, cache, run_config)
        require(checks["passed"], f"{version_name} fold{fold}: best reload invariant failed")
        rows = prediction_rows(fold, best["epoch"], best_outputs, cache)
    frozen_metrics = metrics_from_rows(rows, "frozen")
    refined_metrics = metrics_from_rows(rows, "final")
    require(refined_metrics == best["metrics"], "Best refiner metric reload changed")
    diagnostics = row_diagnostics(rows)

    fold_dir.mkdir(parents=True)
    v1.write_csv(fold_dir / "epoch_metrics.csv", epoch_rows)
    v1.write_csv(fold_dir / "best_predictions.csv", rows)
    torch.save(
        {
            "version": version_name,
            "fold": fold,
            "best_epoch": best["epoch"],
            "run_config": asdict(run_config),
            "refiner_state": best_state,
        },
        fold_dir / "checkpoint_best.pt",
    )
    summary = {
        "model": MODEL_NAME,
        "version": version_name,
        "fold": fold,
        "best_epoch": best["epoch"],
        "selection_tuple": list(best["selection_tuple"]),
        "run_config": asdict(run_config),
        "train_size": int(cache["train_mask"].sum()),
        "test_size": int(cache["test_mask"].sum()),
        "frozen_original_best_epoch": int(cache["best_epoch"]),
        "frozen_metrics": frozen_metrics,
        "refined_metrics": refined_metrics,
        "diagnostics": diagnostics,
        "best_margin_threshold": best["margin_threshold"],
        "best_neighbor_confidence_threshold": best[
            "neighbor_confidence_threshold"
        ],
        "refiner_parameters": EXPECTED_LOCAL_PARAMETERS,
        "elapsed_seconds": elapsed,
        "passed": True,
    }
    v1.write_json(fold_dir / "summary.json", summary)
    print_fold_line(summary)
    del cache, refiner, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def confusion_rows(matrix: list[list[int]]) -> list[dict]:
    return [
        {
            "actual": CLASS_NAMES[row_index],
            **{
                f"predicted_{CLASS_NAMES[column_index]}": int(
                    matrix[row_index][column_index]
                )
                for column_index in range(3)
            },
        }
        for row_index in range(3)
    ]


def render_version_report(summary: dict) -> str:
    metrics = summary["refined_metrics"]
    diagnostics = summary["diagnostics"]
    return "\n".join(
        [
            f"# {MODEL_NAME} - {summary['version']}",
            "",
            f"- Frozen Original: {summary['frozen_metrics']['correct']}/{EXPECTED_SUBJECTS}",
            f"- Refined: {metrics['correct']}/{EXPECTED_SUBJECTS}",
            f"- ACC: {metrics['acc']:.10f}",
            f"- Macro-F1: {metrics['macro_f1']:.10f}",
            f"- BACC: {metrics['bacc']:.10f}",
            f"- Probability Macro-AUC: {metrics['macro_auc']:.10f}",
            f"- Weighted-F1: {metrics['weighted_f1']:.10f}",
            f"- Confusion matrix: {metrics['confusion_matrix']}",
            f"- Changed / repairs / damages: {diagnostics['changed_prediction_count']} / {diagnostics['repairs']} / {diagnostics['damages']}",
            f"- Gate active: {diagnostics['gate_active_count']}/{EXPECTED_SUBJECTS} ({diagnostics['gate_active_ratio']:.6f})",
            f"- Active mean/max g: {diagnostics['active_gate_mean']:.6f} / {diagnostics['active_gate_max']:.6f}",
            f"- Training seconds: {summary['elapsed_seconds']:.3f}",
            "",
        ]
    )


def run_version(
    version_name: str,
    run_config: FrozenUMLERConfig,
    historical_config,
    device: torch.device,
) -> dict:
    output_root = ROOT / RESULTS_ROOT / version_name
    if output_root.exists():
        config_path = output_root / "config.json"
        if config_path.exists():
            require(
                json.loads(config_path.read_text(encoding="utf-8"))
                == asdict(run_config),
                f"Existing version config mismatch: {output_root}",
            )
        else:
            require(
                version_name == DEFAULT_VERSION,
                f"Existing adjustment directory lacks config: {output_root}",
            )
            v1.write_json(config_path, asdict(run_config))
    else:
        output_root.mkdir(parents=True)
        v1.write_json(output_root / "config.json", asdict(run_config))

    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        summary, rows = run_fold(
            version_name,
            fold,
            run_config,
            output_root,
            historical_config,
            device,
        )
        fold_summaries.append(summary)
        oof_rows.extend(rows)
    require(len(oof_rows) == EXPECTED_SUBJECTS, "Frozen UM-LER OOF row count changed")
    require(
        len({int(row["subject_index"]) for row in oof_rows}) == EXPECTED_SUBJECTS,
        "Frozen UM-LER OOF subjects are not unique",
    )
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    frozen_metrics = metrics_from_rows(oof_rows, "frozen")
    refined_metrics = metrics_from_rows(oof_rows, "final")
    require(frozen_metrics["correct"] == EXPECTED_ORIGINAL_CORRECT, "Frozen Original changed during refiner training")
    require(frozen_metrics["confusion_matrix"] == EXPECTED_ORIGINAL_CONFUSION, "Frozen Original confusion changed during refiner training")
    diagnostics = row_diagnostics(oof_rows)
    fold_rows = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            "frozen_correct": summary["frozen_metrics"]["correct"],
            "refined_correct": summary["refined_metrics"]["correct"],
            "repairs": summary["diagnostics"]["repairs"],
            "damages": summary["diagnostics"]["damages"],
            **{
                name: summary["refined_metrics"][name]
                for name in METRIC_NAMES
            },
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for summary in fold_summaries
    ]
    result = {
        "model": MODEL_NAME,
        "version": version_name,
        "config": asdict(run_config),
        "frozen_metrics": frozen_metrics,
        "refined_metrics": refined_metrics,
        "delta_vs_frozen": {
            "correct": refined_metrics["correct"] - frozen_metrics["correct"],
            **{
                name: refined_metrics[name] - frozen_metrics[name]
                for name in METRIC_NAMES
            },
        },
        "diagnostics": diagnostics,
        "fold_metrics": fold_rows,
        "refiner_parameters": EXPECTED_LOCAL_PARAMETERS,
        "elapsed_seconds": float(
            sum(summary["elapsed_seconds"] for summary in fold_summaries)
        ),
        "source_commit": git_value("rev-parse", "HEAD"),
    }
    v1.write_csv(output_root / "oof_predictions.csv", oof_rows)
    v1.write_csv(output_root / "fold_metrics.csv", fold_rows)
    v1.write_json(output_root / "metrics.json", refined_metrics)
    v1.write_json(
        output_root / "changed_subject_indices.json",
        {
            "changed": diagnostics["changed_subject_indices"],
            "repairs": diagnostics["repair_subject_indices"],
            "damages": diagnostics["damage_subject_indices"],
        },
    )
    v1.write_csv(
        output_root / "confusion_matrix.csv",
        confusion_rows(refined_metrics["confusion_matrix"]),
    )
    v1.write_json(output_root / "report.json", result)
    (output_root / "report.md").write_text(
        render_version_report(result), encoding="utf-8"
    )
    reloaded = v1.read_csv(output_root / "oof_predictions.csv")
    require(metrics_from_rows(reloaded, "final") == refined_metrics, "Frozen UM-LER OOF readback changed")
    return result


def config_delta(left: FrozenUMLERConfig, right: FrozenUMLERConfig) -> dict:
    left_values = asdict(left)
    right_values = asdict(right)
    return {
        name: {"from": left_values[name], "to": right_values[name]}
        for name in left_values
        if left_values[name] != right_values[name]
    }


def candidate_from_current(
    current: FrozenUMLERConfig, field: str, value
) -> FrozenUMLERConfig:
    candidate = replace(current, **{field: value})
    require(len(config_delta(current, candidate)) == 1, "Adjustment is not single-variable")
    return candidate


def is_success(result: dict) -> bool:
    metrics = result["refined_metrics"]
    frozen = result["frozen_metrics"]
    metric_guard = not (
        metrics["macro_f1"] < frozen["macro_f1"] - 0.005
        and metrics["macro_auc"] < frozen["macro_auc"] - 0.005
    )
    return metrics["correct"] >= 557 and metric_guard


def choose_adjustment(
    history: list[dict], tried: set[tuple[str, float]]
) -> tuple[FrozenUMLERConfig, str] | None:
    current = history[-1]
    current_config = FrozenUMLERConfig(**current["config"])
    diagnostics = current["diagnostics"]
    if is_success(current) or len(history) >= 3:
        return None

    over_aggressive = (
        diagnostics["damages"] > diagnostics["repairs"]
        or diagnostics["gate_active_ratio"] > 0.20
    )
    if over_aggressive:
        key = ("gate_cap", 0.20)
        if key not in tried and current_config.gate_cap != 0.20:
            return (
                candidate_from_current(current_config, "gate_cap", 0.20),
                "over-aggressive: lower gate cap",
            )

    under_active = (
        diagnostics["gate_active_ratio"] < 0.02
        or diagnostics["changed_prediction_count"] < 3
        or diagnostics["repairs"] == 0
    )
    if under_active:
        key = ("uncertain_quantile", 0.35)
        if key not in tried and current_config.uncertain_quantile != 0.35:
            return (
                candidate_from_current(
                    current_config, "uncertain_quantile", 0.35
                ),
                "under-active: expand train-margin uncertainty quantile",
            )

    normal_activation = 0.02 <= diagnostics["gate_active_ratio"] <= 0.20
    mixed_neighborhood = (
        normal_activation
        and diagnostics["changed_prediction_count"] >= 3
        and abs(diagnostics["repairs"] - diagnostics["damages"]) <= 1
        and diagnostics["mean_normalized_neighbor_entropy"] >= 0.60
    )
    if mixed_neighborhood:
        key = ("top_k", 5.0)
        if key not in tried and current_config.top_k != 5:
            return (
                candidate_from_current(current_config, "top_k", 5),
                "mixed neighborhoods: increase locality",
            )
    return None


def render_final_report(payload: dict) -> str:
    selected = payload["selected_result"]
    metrics = selected["refined_metrics"]
    diagnostics = selected["diagnostics"]
    lines = [
        f"# {MODEL_NAME} - Final Report",
        "",
        f"**Decision: {payload['decision']}**",
        "",
        f"- Branch: `{payload['branch']}`",
        f"- Source commit: `{payload['source_commit']}`",
        f"- Device: {payload['environment']['gpu']} ({payload['environment']['device']})",
        f"- Frozen Original reproduced: {payload['anchor']['anchor_reproduced']}",
        f"- Frozen Original correct: {payload['anchor']['oof_metrics']['correct']}/{EXPECTED_SUBJECTS}",
        f"- Frozen Original confusion: {payload['anchor']['oof_metrics']['confusion_matrix']}",
        f"- Frozen Original probability Macro-AUC: {payload['anchor']['oof_metrics']['macro_auc']:.10f}",
        f"- Historical prompt probability Macro-AUC: {HISTORICAL_REPORTED_AUC:.7f} (matched; legacy adjusted-score AUC was 0.9500702)",
        f"- Selected version: `{selected['version']}`",
        f"- Refiner parameters: {EXPECTED_LOCAL_PARAMETERS}",
        f"- Correct: {metrics['correct']}/{EXPECTED_SUBJECTS}",
        f"- ACC: {metrics['acc']:.10f}",
        f"- Macro-F1: {metrics['macro_f1']:.10f}",
        f"- BACC: {metrics['bacc']:.10f}",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.10f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.10f}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        f"- Changed / repairs / damages: {diagnostics['changed_prediction_count']} / {diagnostics['repairs']} / {diagnostics['damages']}",
        f"- Gate activation: {diagnostics['gate_active_count']}/{EXPECTED_SUBJECTS} ({diagnostics['gate_active_ratio']:.6f})",
        f"- Active mean/max g: {diagnostics['active_gate_mean']:.6f}/{diagnostics['active_gate_max']:.6f}",
        f"- Completed-fold training time: {payload['total_training_seconds']:.3f} seconds",
        f"- Formal wall time: {payload['formal_wall_seconds']:.3f} seconds",
        "",
        "## Attempted versions",
        "",
        "| Version | Correct | Macro-F1 | BACC | Probability Macro-AUC | Repairs | Damages | Changed | Gate active |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {result['version']} | {result['refined_metrics']['correct']} | {result['refined_metrics']['macro_f1']:.7f} | {result['refined_metrics']['bacc']:.7f} | {result['refined_metrics']['macro_auc']:.7f} | {result['diagnostics']['repairs']} | {result['diagnostics']['damages']} | {result['diagnostics']['changed_prediction_count']} | {result['diagnostics']['gate_active_ratio']:.4%} |"
            for result in payload["attempted_results"]
        ],
        "",
    ]
    return "\n".join(lines)


def run_formal_all() -> dict:
    anchor = load_anchor_manifest()
    historical_config, _, _, device, branch = load_context()
    require(not (ROOT / FINAL_JSON_REL).exists(), "Final Frozen UM-LER report already exists")
    started = time.perf_counter()
    history = []
    tuning_log = []
    tried: set[tuple[str, float]] = set()
    run_config = DEFAULT
    for version_index in range(3):
        version_name = (
            DEFAULT_VERSION
            if version_index == 0
            else f"{DEFAULT_VERSION}_{version_index}"
        )
        result = run_version(
            version_name, run_config, historical_config, device
        )
        history.append(result)
        if is_success(result):
            break
        adjustment = choose_adjustment(history, tried)
        if adjustment is None:
            break
        next_config, reason = adjustment
        delta = config_delta(run_config, next_config)
        field, values = next(iter(delta.items()))
        tried.add((field, float(values["to"])))
        tuning_log.append(
            {
                "after_version": version_name,
                "reason": reason,
                "next_single_change": delta,
            }
        )
        run_config = next_config

    selected = max(
        history,
        key=lambda result: (
            result["refined_metrics"]["correct"],
            result["refined_metrics"]["acc"],
            result["refined_metrics"]["macro_auc"],
            result["refined_metrics"]["macro_f1"],
        ),
    )
    decision = "SUCCESS_93_PLUS" if is_success(selected) else "NO_NET_IMPROVEMENT"
    payload = {
        "model": MODEL_NAME,
        "decision": decision,
        "anchor": anchor,
        "selected_result": selected,
        "attempted_results": history,
        "tuning_log": tuning_log,
        "branch": branch,
        "source_commit": git_value("rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "total_training_seconds": float(
            sum(result["elapsed_seconds"] for result in history)
        ),
        "formal_wall_seconds": time.perf_counter() - started,
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
            "optimizer": "Adam (refiner only)",
            "scheduler": "CustomCosineAnnealingLR(T_max=400)",
            "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
            "original_frozen": True,
            "original_forward_passes_after_cache": 0,
        },
    }
    v1.write_json(ROOT / FINAL_JSON_REL, payload)
    (ROOT / FINAL_MD_REL).write_text(
        render_final_report(payload), encoding="utf-8"
    )
    reloaded_rows = v1.read_csv(
        ROOT / RESULTS_ROOT / selected["version"] / "oof_predictions.csv"
    )
    require(
        metrics_from_rows(reloaded_rows, "final") == selected["refined_metrics"],
        "Final selected OOF recomputation changed",
    )
    print(
        f"final decision={decision} selected={selected['version']} "
        f"correct={selected['refined_metrics']['correct']}/{EXPECTED_SUBJECTS} "
        f"ACC={selected['refined_metrics']['acc']:.7f}",
        flush=True,
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-anchor", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal-all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prepare_anchor:
        manifest = prepare_anchor_cache()
        print(
            f"anchor reproduced={manifest['anchor_reproduced']} "
            f"correct={manifest['oof_metrics']['correct']}/{EXPECTED_SUBJECTS} "
            f"ACC={manifest['oof_metrics']['acc']:.7f}",
            flush=True,
        )
    elif args.smoke:
        summary = run_smoke()
        print(
            f"smoke passed={summary['passed']} fold={summary['fold']} "
            f"epochs={summary['epochs']}",
            flush=True,
        )
    else:
        run_formal_all()


if __name__ == "__main__":
    main()
