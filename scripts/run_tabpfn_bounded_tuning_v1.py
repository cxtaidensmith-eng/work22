#!/usr/bin/env python
"""Run the bounded E4/E8 TabPFN inference arms with locked protocol audits."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

import run_tabpfn_baseline_v1 as base
import run_tabpfn_full_feature_minimal_v1 as full


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "TabPFN-3 Bounded Inference and Calibration Tuning v1"
EXPECTED_BRANCH = "experiment/tabpfn-bounded-tuning-v1"
BASE_BRANCH = "experiment/tabpfn-full-feature-minimal-v1"
BASE_COMMIT = "77f3943f4c5deb9193123ad8d6f0c933da5ecc0d"
RESULT_ROOT = ROOT / "experiments/tabpfn_bounded_tuning_v1"
E2_FORMAL = ROOT / "experiments/tabpfn_full_feature_minimal_v1/formal"
E2_OOF = E2_FORMAL / "oof_predictions.csv"
EXPECTED_E2_OOF_SHA256 = "03732120ebebde721fd06c6336b751a8726b02fc93562c228bd64c3ab0086336"
E2_METRICS = {
    "acc": 0.9113712374581939,
    "macro_f1": 0.8962751307156939,
    "bacc": 0.8863430223425243,
    "macro_auc": 0.976828823792558,
    "weighted_f1": 0.9111080702928556,
    "correct_count": 545,
}
ARM_ESTIMATORS = {"e4": 4, "e8": 8}
MODAL_ORDER = full.MODAL_ORDER

_REQUESTED_ESTIMATORS = 4
_EFFECTIVE_ESTIMATORS = 4

base.EXPECTED_BRANCH = EXPECTED_BRANCH
base.BASE_COMMIT = BASE_COMMIT


def configure_estimators(requested: int, effective: int | None = None) -> None:
    global _REQUESTED_ESTIMATORS, _EFFECTIVE_ESTIMATORS
    base.require(requested in (1, 4, 8), f"Unsupported estimator request: {requested}")
    _REQUESTED_ESTIMATORS = requested
    _EFFECTIVE_ESTIMATORS = effective if effective is not None else requested
    base.require(_EFFECTIVE_ESTIMATORS in (2, 4, 8), "Unexpected effective estimator count")


def constructor_payload(checkpoint: Path, requested: int) -> dict[str, Any]:
    return {
        "model_path": str(checkpoint),
        "n_estimators": requested,
        "auto_scale_n_estimators": True,
        "random_state": 0,
        "device": "cuda:0",
        "fit_mode": "low_memory",
        "memory_saving_mode": True,
        "inference_precision": "auto",
        "show_progress_bar": False,
    }


def make_classifier(checkpoint: Path) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(**constructor_payload(checkpoint, _REQUESTED_ESTIMATORS))


def model_audit(classifier: Any, checkpoint: Path, instance_uuid: str) -> dict[str, Any]:
    params = classifier.get_params(deep=False)
    strict = constructor_payload(checkpoint, _REQUESTED_ESTIMATORS)
    for name, expected in strict.items():
        actual = params[name]
        if name == "model_path":
            base.require(Path(actual).resolve() == checkpoint.resolve(), "Checkpoint path changed")
        else:
            base.require(actual == expected, f"Effective parameter changed: {name}")
    base.require(classifier.n_estimators == _REQUESTED_ESTIMATORS, "Requested n_estimators changed")
    base.require(classifier.n_estimators_ == _EFFECTIVE_ESTIMATORS, "Effective n_estimators_ changed")
    base.require(classifier.auto_scale_n_estimators is True, "Auto scaling disabled")
    base.require(len(classifier.ensemble_configs_) == _EFFECTIVE_ESTIMATORS, "Member count mismatch")
    base.require(len(classifier.models_) == 1, "Expected exactly one checkpoint model")
    base.require(tuple(str(item) for item in classifier.devices_) == ("cuda:0",), "Device changed")
    base.require(np.array_equal(np.asarray(classifier.classes_, dtype=np.int64), base.CLASS_IDS), "Class order changed")
    base.require(int(classifier.n_features_in_) == 360, "Model did not receive 360 features")

    configs = classifier.ensemble_configs_
    model_indices = [int(config._model_index) for config in configs]
    base.require(model_indices == [0] * _EFFECTIVE_ESTIMATORS, "Members do not share checkpoint model index 0")
    raw_member_indices = classifier.ensemble_preprocessor_.subsample_feature_indices
    base.require(len(raw_member_indices) == _EFFECTIVE_ESTIMATORS, "Feature member count mismatch")
    feature_names = np.asarray(classifier.feature_names_in_, dtype=object).astype(str)
    modalities, modal_sizes = full.feature_modalities(feature_names.tolist())
    coverage_counts = np.zeros(360, dtype=np.int64)
    members: list[dict[str, Any]] = []
    member_sets: list[set[int]] = []
    for member_index, (config, raw_indices) in enumerate(zip(configs, raw_member_indices, strict=True)):
        indices = np.arange(360, dtype=np.int64) if raw_indices is None else np.asarray(raw_indices, dtype=np.int64)
        base.require(indices.ndim == 1, f"member {member_index} feature indices not 1D")
        base.require(np.unique(indices).size == indices.size, f"member {member_index} has duplicate indices")
        base.require(np.all((indices >= 0) & (indices < 360)), f"member {member_index} index out of range")
        member_sets.append(set(indices.tolist()))
        coverage_counts[indices] += 1
        preprocess = config.preprocess_config
        members.append(
            {
                "member": member_index,
                "model_index": int(config._model_index),
                "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
                "preprocess_name": preprocess.name,
                "preprocess_repr": str(preprocess),
                "max_features_per_estimator": int(preprocess.max_features_per_estimator),
                "feature_shift_count": int(config.feature_shift_count),
                "feature_shift_decoder": config.feature_shift_decoder,
                "original_feature_count": int(indices.size),
                "original_feature_indices": indices.tolist(),
                "original_feature_names": feature_names[indices].tolist(),
                "original_feature_indices_sha256": base.raw_array_sha256(indices),
                "modal_feature_counts": {
                    modal: int(sum(modalities[index] == modal for index in indices))
                    for modal in MODAL_ORDER
                },
            }
        )
    feature_union = sorted(set().union(*member_sets))
    feature_intersection = sorted(set.intersection(*member_sets))
    uncovered = sorted(set(range(360)) - set(feature_union))
    modal_union_counts = {
        modal: int(sum(modalities[index] == modal for index in feature_union))
        for modal in MODAL_ORDER
    }
    modal_union_rates = {modal: modal_union_counts[modal] / modal_sizes[modal] for modal in MODAL_ORDER}
    coverage_sha = base.canonical_sha256(
        {
            "member_feature_indices": [item["original_feature_indices"] for item in members],
            "feature_union": feature_union,
            "feature_intersection": feature_intersection,
            "coverage_counts": coverage_counts.tolist(),
            "modal_union_counts": modal_union_counts,
        }
    )
    base.require(feature_union == list(range(360)), "Members do not cover exactly 0..359")
    base.require(not uncovered, "Uncovered features remain")
    base.require(all(value == 1.0 for value in modal_union_rates.values()), "A modality is not fully covered")
    transforms = [
        {
            "name": item.name,
            "categorical_name": item.categorical_name,
            "append_original": item.append_original,
            "max_features_per_estimator": int(item.max_features_per_estimator),
            "global_transformer_name": item.global_transformer_name,
        }
        for item in classifier.inference_config_.PREPROCESS_TRANSFORMS
    ]
    base.require(all(item["max_features_per_estimator"] == 200 for item in transforms), "Checkpoint feature limit changed")
    gates = {
        "requested_n_estimators_exact": classifier.n_estimators == _REQUESTED_ESTIMATORS,
        "effective_n_estimators_exact": classifier.n_estimators_ == _EFFECTIVE_ESTIMATORS,
        "auto_scale_enabled": classifier.auto_scale_n_estimators is True,
        "one_loaded_checkpoint_model": len(classifier.models_) == 1,
        "one_classifier_instance": True,
        "internal_member_count_exact": len(configs) == _EFFECTIVE_ESTIMATORS,
        "members_share_checkpoint_model_index_zero": model_indices == [0] * _EFFECTIVE_ESTIMATORS,
        "no_independently_trained_models": True,
        "cuda_0_only": tuple(str(item) for item in classifier.devices_) == ("cuda:0",),
        "class_order_fixed": np.array_equal(np.asarray(classifier.classes_, dtype=np.int64), base.CLASS_IDS),
        "feature_union_is_0_to_359": feature_union == list(range(360)),
        "uncovered_feature_count_is_zero": len(uncovered) == 0,
        "all_modal_union_rates_are_one": all(value == 1.0 for value in modal_union_rates.values()),
    }
    base.require(all(gates.values()), f"Bounded inference gates failed: {gates}")
    return {
        "instance_uuid": instance_uuid,
        "classifier_class": type(classifier).__name__,
        "model_architecture_class": type(classifier.model_).__name__,
        "constructor_parameters": base.safe_json(params),
        "strict_parameters": strict,
        "requested_n_estimators": int(classifier.n_estimators),
        "effective_n_estimators": int(classifier.n_estimators_),
        "auto_scale_n_estimators": bool(classifier.auto_scale_n_estimators),
        "internal_feature_subspace_member_count": len(configs),
        "loaded_checkpoint_model_count": len(classifier.models_),
        "effective_devices": [str(item) for item in classifier.devices_],
        "effective_classes": np.asarray(classifier.classes_, dtype=np.int64).tolist(),
        "n_features_received": int(classifier.n_features_in_),
        "n_train_samples": int(classifier.n_train_samples_),
        "all_checkpoint_preprocess_transforms": transforms,
        "model_semantics": {
            "external_model_count": 1,
            "checkpoint_count": 1,
            "independently_trained_model_count": 0,
            "seed_ensemble": False,
            "probability_fusion": False,
            "internal_feature_subspace_members": _EFFECTIVE_ESTIMATORS,
            "members_share_same_pretrained_weights": True,
        },
        "feature_coverage": {
            "full_input_feature_count": 360,
            "members": members,
            "member_feature_set_intersection_count": len(feature_intersection),
            "member_feature_set_intersection": feature_intersection,
            "member_feature_set_union_count": len(feature_union),
            "member_feature_set_union": feature_union,
            "uncovered_feature_count": len(uncovered),
            "uncovered_feature_indices": uncovered,
            "duplicate_coverage_feature_count": int((coverage_counts > 1).sum()),
            "coverage_counts_by_original_feature": coverage_counts.tolist(),
            "modal_total_feature_counts": modal_sizes,
            "modal_union_feature_counts": modal_union_counts,
            "modal_union_coverage_rates": modal_union_rates,
            "feature_coverage_sha256": coverage_sha,
        },
        "internal_original_feature_subsampling": {
            "external_feature_selection_performed": False,
            "full_input_feature_count": 360,
            "selected_original_feature_count": len(feature_union),
            "selected_original_feature_indices": feature_union,
            "selected_original_feature_names": feature_names[feature_union].tolist(),
            "selected_original_feature_indices_sha256": base.raw_array_sha256(np.asarray(feature_union, dtype=np.int64)),
            "disclosure": f"{_EFFECTIVE_ESTIMATORS} internal members jointly cover all 360 unchanged input columns.",
        },
        "bounded_inference_gates": gates,
    }


base.make_classifier = make_classifier
base.model_audit = model_audit


def write_feature_coverage_files(fold_dir: Path) -> dict[str, Any]:
    audit = base.read_json(fold_dir / "model_audit.json")
    coverage = audit["feature_coverage"]
    members = coverage["members"]
    feature_names = audit["internal_original_feature_subsampling"]["selected_original_feature_names"]
    modalities, _ = full.feature_modalities(feature_names)
    member_sets: list[set[int]] = []
    member_hashes: dict[str, str] = {}
    for member in members:
        member_id = int(member["member"])
        indices = np.asarray(member["original_feature_indices"], dtype=np.int64)
        member_sets.append(set(indices.tolist()))
        path = fold_dir / f"member_{member_id:02d}_feature_indices.csv"
        pd.DataFrame(
            {
                "member": member_id,
                "feature_index": indices,
                "feature_name": np.asarray(feature_names, dtype=object)[indices],
                "modality": np.asarray(modalities, dtype=object)[indices],
            }
        ).to_csv(path, index=False)
        readback = pd.read_csv(path)
        base.require(np.array_equal(readback["feature_index"].to_numpy(dtype=np.int64), indices), "Member index readback failed")
        member_hashes[f"member_{member_id:02d}"] = base.file_sha256(path)
    counts = np.asarray(coverage["coverage_counts_by_original_feature"], dtype=np.int64)
    payload: dict[str, Any] = {
        "feature_index": np.arange(360, dtype=np.int64),
        "feature_name": feature_names,
        "modality": modalities,
    }
    for member_id, member_set in enumerate(member_sets):
        payload[f"member_{member_id:02d}"] = [int(index in member_set) for index in range(360)]
    payload["coverage_count"] = counts
    path = fold_dir / "feature_coverage.csv"
    pd.DataFrame(payload).to_csv(path, index=False)
    readback = pd.read_csv(path)
    base.require(readback.shape == (360, _EFFECTIVE_ESTIMATORS + 4), "Coverage CSV shape mismatch")
    base.require(np.array_equal(readback["coverage_count"].to_numpy(dtype=np.int64), counts), "Coverage count readback failed")
    return {
        "passed": True,
        "requested_n_estimators": _REQUESTED_ESTIMATORS,
        "effective_n_estimators": _EFFECTIVE_ESTIMATORS,
        "feature_coverage_sha256": coverage["feature_coverage_sha256"],
        "member_file_sha256": member_hashes,
        "feature_coverage_csv_sha256": base.file_sha256(path),
    }


def run_fold(protocol: dict[str, Any], checkpoint: Path, fold: int, fold_dir: Path, log) -> tuple[pd.DataFrame, dict[str, Any], str]:
    predictions, metrics, instance_uuid = base.run_fold(protocol, checkpoint, fold, fold_dir, log)
    validation = write_feature_coverage_files(fold_dir)
    base.write_json(fold_dir / "feature_coverage_validation.json", validation)
    log(
        f"fold={fold} feature_coverage=360/360 internal_members={_EFFECTIVE_ESTIMATORS} "
        f"coverage_sha256={validation['feature_coverage_sha256']}"
    )
    return predictions, metrics, instance_uuid


def paired_comparison(tuned_oof: pd.DataFrame, other_path: Path, other_name: str, expected_sha: str, output_dir: Path) -> dict[str, Any]:
    base.require(other_path.is_file(), f"Comparison OOF missing: {other_path}")
    base.require(base.file_sha256(other_path) == expected_sha, f"{other_name} OOF SHA mismatch")
    tuned = tuned_oof.sort_values("subject_index").reset_index(drop=True)
    other = pd.read_csv(other_path, float_precision="round_trip").sort_values("subject_index").reset_index(drop=True)
    for column in ("subject_index", "truth", "fold"):
        base.require(np.array_equal(tuned[column], other[column]), f"{other_name} alignment failed: {column}")
    truth = tuned["truth"].to_numpy(dtype=np.int64)
    tuned_prediction = tuned["prediction"].to_numpy(dtype=np.int64)
    other_prediction = other["prediction"].to_numpy(dtype=np.int64)
    tuned_correct = tuned_prediction == truth
    other_correct = other_prediction == truth
    tuned_only = tuned_correct & ~other_correct
    other_only = ~tuned_correct & other_correct
    discordant = int(tuned_only.sum() + other_only.sum())
    rows = pd.DataFrame(
        {
            "fold": tuned["fold"].to_numpy(dtype=np.int64),
            "subject_index": tuned["subject_index"].to_numpy(dtype=np.int64),
            "truth": truth,
            "tuned_prediction": tuned_prediction,
            f"{other_name}_prediction": other_prediction,
            "tuned_correct": tuned_correct.astype(np.int64),
            f"{other_name}_correct": other_correct.astype(np.int64),
            "tuned_only_correct": tuned_only.astype(np.int64),
            f"{other_name}_only_correct": other_only.astype(np.int64),
        }
    )
    rows.to_csv(output_dir / f"paired_rows_{other_name}.csv", index=False)
    by_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(base.CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {
            "support": int(mask.sum()),
            "tuned_correct": int((tuned_correct & mask).sum()),
            f"{other_name}_correct": int((other_correct & mask).sum()),
            "tuned_only_correct": int((tuned_only & mask).sum()),
            f"{other_name}_only_correct": int((other_only & mask).sum()),
            "both_wrong": int((~tuned_correct & ~other_correct & mask).sum()),
        }
    result = {
        "other_name": other_name,
        "other_oof_path": str(other_path.resolve()),
        "other_oof_sha256": expected_sha,
        "alignment": {"row_count": 598, "unique_subject_indices": 598, "subject_indices_equal": True, "truths_equal": True, "fold_assignments_equal": True},
        "counts": {
            "both_correct": int((tuned_correct & other_correct).sum()),
            "tuned_only_correct": int(tuned_only.sum()),
            f"{other_name}_only_correct": int(other_only.sum()),
            "both_wrong": int((~tuned_correct & ~other_correct).sum()),
            "discordant": discordant,
        },
        "exact_mcnemar_two_sided_p": float(binomtest(int(tuned_only.sum()), discordant, p=0.5).pvalue) if discordant else 1.0,
        "by_true_class": by_class,
        "other_metrics_recomputed": base.recompute_historical_metrics(other),
        "corrected_subject_indices": rows.loc[tuned_only, "subject_index"].astype(int).tolist(),
        "broken_subject_indices": rows.loc[other_only, "subject_index"].astype(int).tolist(),
    }
    base.write_json(output_dir / f"paired_summary_{other_name}.json", result)
    return result


def aggregate_feature_coverage(formal_dir: Path) -> dict[str, Any]:
    audits = [base.read_json(formal_dir / f"fold_{fold:02d}/model_audit.json") for fold in base.FOLDS]
    feature_names = audits[0]["internal_original_feature_subsampling"]["selected_original_feature_names"]
    modalities, modal_sizes = full.feature_modalities(feature_names)
    table: dict[str, Any] = {"feature_index": np.arange(360), "feature_name": feature_names, "modality": modalities}
    matrix: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, audit in enumerate(audits):
        base.require(all(audit["bounded_inference_gates"].values()), f"fold {fold} bounded gates failed")
        coverage = audit["feature_coverage"]
        counts = np.asarray(coverage["coverage_counts_by_original_feature"], dtype=np.int64)
        base.require(np.all(counts >= 1), f"fold {fold} uncovered feature")
        matrix.append(counts)
        table[f"fold_{fold:02d}_coverage_count"] = counts
        fold_rows.append(
            {
                "fold": fold,
                "requested_n_estimators": audit["requested_n_estimators"],
                "effective_n_estimators": audit["effective_n_estimators"],
                "internal_member_count": audit["internal_feature_subspace_member_count"],
                "member_feature_counts": [item["original_feature_count"] for item in coverage["members"]],
                "all_member_intersection_count": coverage["member_feature_set_intersection_count"],
                "union_count": coverage["member_feature_set_union_count"],
                "uncovered_count": coverage["uncovered_feature_count"],
                "duplicate_coverage_feature_count": coverage["duplicate_coverage_feature_count"],
                "modal_union_feature_counts": coverage["modal_union_feature_counts"],
                "modal_union_coverage_rates": coverage["modal_union_coverage_rates"],
                "feature_coverage_sha256": coverage["feature_coverage_sha256"],
            }
        )
    coverage_matrix = np.asarray(matrix, dtype=np.int64)
    table["minimum_coverage_count_across_folds"] = coverage_matrix.min(axis=0)
    table["maximum_coverage_count_across_folds"] = coverage_matrix.max(axis=0)
    table["total_coverage_count_across_folds"] = coverage_matrix.sum(axis=0)
    pooled_path = formal_dir / "pooled_feature_coverage.csv"
    pd.DataFrame(table).to_csv(pooled_path, index=False)
    result = {
        "passed": True,
        "fold_count": 10,
        "expected_effective_n_estimators": _EFFECTIVE_ESTIMATORS,
        "all_folds_effective_n_estimators_exact": all(row["effective_n_estimators"] == _EFFECTIVE_ESTIMATORS for row in fold_rows),
        "all_folds_union_is_360": all(row["union_count"] == 360 for row in fold_rows),
        "all_folds_uncovered_count_zero": all(row["uncovered_count"] == 0 for row in fold_rows),
        "all_folds_all_modalities_100_percent": all(all(value == 1.0 for value in row["modal_union_coverage_rates"].values()) for row in fold_rows),
        "modal_total_feature_counts": modal_sizes,
        "pooled_feature_coverage_csv_sha256": base.file_sha256(pooled_path),
        "pooled_feature_coverage_sha256": base.canonical_sha256({"fold_coverage_counts": coverage_matrix.tolist(), "modalities": modalities}),
        "folds": fold_rows,
    }
    base.require(all(result[key] for key in ("all_folds_effective_n_estimators_exact", "all_folds_union_is_360", "all_folds_uncovered_count_zero", "all_folds_all_modalities_100_percent")), "Pooled coverage failed")
    base.write_json(formal_dir / "feature_coverage_summary.json", result)
    return result


def e8_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    correct = int(sum(metrics["confusion_matrix"][i][i] for i in range(3)))
    condition_1 = correct >= 546
    condition_2 = correct == 545 and metrics["bacc"] > E2_METRICS["bacc"] and metrics["macro_auc"] >= 0.970
    passed = condition_1 or condition_2
    return {
        "schema_version": 1,
        "decision": "RUN_E8" if passed else "SKIP_E8",
        "passed": passed,
        "immutable_rule": {
            "condition_1": "E4 correct_count >= 546",
            "condition_2": "E4 correct_count == 545 and BACC > locked E2 BACC and Macro-AUC >= 0.970",
            "logic": "condition_1 OR condition_2",
        },
        "locked_e2": E2_METRICS,
        "e4": {**{key: metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")}, "correct_count": correct},
        "condition_1_passed": condition_1,
        "condition_2_passed": condition_2,
        "reason": "E4 passed the predeclared E8 gate." if passed else "E4 did not pass either predeclared E8 gate condition; E8 is prohibited.",
    }


def aggregate_formal(staging_dir: Path, arm: str, predictions: list[pd.DataFrame], fold_metrics: list[dict[str, Any]], uuids: list[str], original_oof: Path) -> dict[str, Any]:
    result = base.aggregate_formal(staging_dir, predictions, fold_metrics, uuids, original_oof)
    oof = pd.read_csv(staging_dir / "oof_predictions.csv", float_precision="round_trip")
    result["comparisons"]["e2_full_feature"] = paired_comparison(oof, E2_OOF, "e2_full_feature", EXPECTED_E2_OOF_SHA256, staging_dir)
    if arm == "e8":
        e4_oof = RESULT_ROOT / "e4/formal/oof_predictions.csv"
        base.require(e4_oof.is_file(), "E4 OOF missing for E8 comparison")
        result["comparisons"]["e4"] = paired_comparison(oof, e4_oof, "e4", base.file_sha256(e4_oof), staging_dir)
    result["feature_coverage_summary"] = aggregate_feature_coverage(staging_dir)
    result["model_semantics"] = {
        "external_model_count": 1,
        "checkpoint_count": 1,
        "independently_trained_model_count": 0,
        "internal_feature_subspace_members_per_fold": _EFFECTIVE_ESTIMATORS,
        "seed_ensemble": False,
        "probability_fusion": False,
    }
    if arm == "e4":
        gate = e8_gate(result["oof_metrics"])
        result["e8_gate_decision"] = gate
        base.write_json(staging_dir / "e8_gate_decision.json", gate)
    base.write_json(staging_dir / "aggregate_summary.json", result)
    return result


def validate_formal_readback(formal_dir: Path) -> dict[str, Any]:
    probability_columns = [f"probability_{name}" for name in base.CLASS_ORDER]
    oof = pd.read_csv(formal_dir / "oof_predictions.csv", float_precision="round_trip")
    stored = base.read_json(formal_dir / "oof_metrics.json")
    recomputed = base.metric_bundle(oof["truth"].to_numpy(dtype=np.int64), oof[probability_columns].to_numpy(dtype=np.float64))
    exact = {name: recomputed[name] == stored[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")}
    base.require(all(exact.values()) and recomputed["confusion_matrix"] == stored["confusion_matrix"], "OOF metric readback mismatch")
    base.require(oof.shape[0] == 598 and oof["subject_index"].nunique() == 598, "OOF integrity failed")
    frames: list[pd.DataFrame] = []
    uuids: list[str] = []
    fold_checks: dict[str, Any] = {}
    for fold in base.FOLDS:
        fold_dir = formal_dir / f"fold_{fold:02d}"
        frame = pd.read_csv(fold_dir / "predictions.csv", float_precision="round_trip")
        frames.append(frame)
        metrics = base.read_json(fold_dir / "metrics.json")
        actual = base.metric_bundle(frame["truth"].to_numpy(dtype=np.int64), frame[probability_columns].to_numpy(dtype=np.float64))
        fold_exact = all(actual[name] == metrics[name] for name in exact) and actual["confusion_matrix"] == metrics["confusion_matrix"]
        base.require(fold_exact, f"fold {fold} metric mismatch")
        audit = base.read_json(fold_dir / "model_audit.json")
        base.require(all(audit["bounded_inference_gates"].values()), f"fold {fold} audit failed")
        uuids.append(audit["instance_uuid"])
        fold_checks[f"fold_{fold:02d}"] = {"passed": True, "rows": len(frame), "metrics_exact": True, "effective_n_estimators": audit["effective_n_estimators"], "feature_union_count": audit["feature_coverage"]["member_feature_set_union_count"], "uncovered_feature_count": audit["feature_coverage"]["uncovered_feature_count"]}
    concatenated = pd.concat(frames, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    base.require(np.array_equal(concatenated.to_numpy(), oof.to_numpy()), "OOF differs from fold concatenation")
    base.require(len(set(uuids)) == 10, "Fresh classifier UUID check failed")
    coverage = base.read_json(formal_dir / "feature_coverage_summary.json")
    base.require(coverage["passed"], "Coverage summary failed")
    result = {
        "validated_at_utc": base.utc_now(),
        "passed": True,
        "oof_rows": 598,
        "oof_unique_subject_indices": 598,
        "oof_metric_exactness": exact,
        "oof_confusion_matrix_exact": True,
        "oof_equals_fold_concatenation": True,
        "fresh_classifier_uuid_count": len(set(uuids)),
        "all_folds_effective_n_estimators_exact": True,
        "all_folds_cover_all_360_features": True,
        "all_folds_uncovered_feature_count_zero": True,
        "folds": fold_checks,
        "artifact_sha256": {path.relative_to(formal_dir).as_posix(): base.file_sha256(path) for path in sorted(formal_dir.rglob("*")) if path.is_file() and path.name != "readback_validation.json"},
    }
    base.write_json(formal_dir / "readback_validation.json", result)
    return result


def preflight_payload(protocol: dict[str, Any], checkpoint: Path, arm: str) -> dict[str, Any]:
    requested = ARM_ESTIMATORS[arm]
    payload = base.preflight_payload(protocol, checkpoint)
    payload["experiment"] = EXPERIMENT_NAME
    payload["git"].update({"base_branch": BASE_BRANCH, "base_commit": BASE_COMMIT, "new_branch": EXPECTED_BRANCH, "remote_branch": f"origin/{EXPECTED_BRANCH}"})
    payload["arm"] = arm
    payload["constructor"] = {**constructor_payload(checkpoint, requested), "all_other_parameters": "tabpfn==8.2.0 defaults"}
    payload["protocol"].update({"requested_n_estimators": requested, "expected_effective_n_estimators": requested, "auto_scale_n_estimators": True, "external_model_count": 1, "checkpoint_count": 1, "independently_trained_model_count": 0, "internal_feature_subspace_members": requested, "seed_ensemble": False, "probability_fusion": False, "full_feature_union_required": list(range(360))})
    e2_validation = base.read_json(E2_FORMAL / "readback_validation.json")
    base.require(e2_validation["passed"] is True, "Locked E2 readback is not passed")
    base.require(base.file_sha256(E2_OOF) == EXPECTED_E2_OOF_SHA256, "Locked E2 OOF hash mismatch")
    payload["locked_e2"] = {"formal_dir": E2_FORMAL.relative_to(ROOT).as_posix(), "oof_sha256": EXPECTED_E2_OOF_SHA256, "metrics": E2_METRICS, "readback_passed": True, "read_only": True}
    payload["bounded_scope"] = {"allowed_estimator_arms": [4, 8], "e8_requires_e4_gate": True, "n_estimators_16_or_32_prohibited": True, "other_checkpoint_prohibited": True, "feature_strategy_search_prohibited": True, "auto_tabpfn": False, "phe": False, "fine_tuning": False, "embedding": False, "fusion": False, "hyperparameter_search": False}
    payload["source_files"] = {
        "baseline_runner": {"path": "scripts/run_tabpfn_baseline_v1.py", "sha256": base.file_sha256(ROOT / "scripts/run_tabpfn_baseline_v1.py")},
        "e2_runner": {"path": "scripts/run_tabpfn_full_feature_minimal_v1.py", "sha256": base.file_sha256(ROOT / "scripts/run_tabpfn_full_feature_minimal_v1.py")},
        "runner": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": base.file_sha256(Path(__file__).resolve())},
        "calibration": {"path": "scripts/calibrate_tabpfn_bounded_tuning_v1.py", "sha256": base.file_sha256(ROOT / "scripts/calibrate_tabpfn_bounded_tuning_v1.py")},
        "recompute": {"path": "scripts/recompute_tabpfn_bounded_tuning_v1.py", "sha256": base.file_sha256(ROOT / "scripts/recompute_tabpfn_bounded_tuning_v1.py")},
    }
    payload["static_preflight_passed"] = True
    return payload


def require_e8_gate() -> dict[str, Any]:
    path = RESULT_ROOT / "e4/formal/e8_gate_decision.json"
    base.require(path.is_file(), "E8 requires completed E4 gate")
    gate = base.read_json(path)
    base.require(gate["passed"] is True and gate["decision"] == "RUN_E8", "E8 gate did not pass")
    return gate


def run_experiment(args: argparse.Namespace) -> None:
    requested = ARM_ESTIMATORS[args.arm]
    configure_estimators(requested, requested)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    protocol = base.load_protocol()
    preflight = preflight_payload(protocol, checkpoint, args.arm)
    if args.arm == "e8":
        preflight["e8_gate"] = require_e8_gate()
    if args.mode == "preflight":
        print(json.dumps(base.safe_json(preflight), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.mode == "formal":
        base.require(args.original_oof is not None, "Formal mode requires --original-oof")
        original_oof = Path(args.original_oof).resolve()
        base.require(original_oof.is_file() and base.file_sha256(original_oof) == base.EXPECTED_ORIGINAL_OOF_SHA256, "Original OOF mismatch")
    else:
        original_oof = None
    arm_root = Path(args.output_root).resolve() / args.arm
    if args.mode == "formal":
        smoke = arm_root / "smoke/smoke_summary.json"
        base.require(smoke.is_file() and base.read_json(smoke)["smoke_passed"] is True, "Formal run requires passed smoke")
    arm_root.mkdir(parents=True, exist_ok=True)
    final_dir = arm_root / args.mode
    staging_dir = arm_root / f".{args.mode}.staging"
    base.require(not final_dir.exists(), f"Refusing to overwrite: {final_dir}")
    base.require(not staging_dir.exists(), f"Retained staging exists: {staging_dir}")
    staging_dir.mkdir()
    log_path = staging_dir / "run.txt"

    def log(message: str) -> None:
        line = f"{base.utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    folds = (0,) if args.mode == "smoke" else base.FOLDS
    started = time.perf_counter()
    preflight["experiment_run"] = {"arm": args.arm, "kind": args.mode, "folds": list(folds), "started_at_utc": base.utc_now(), "formal_reportable": args.mode == "formal"}
    base.write_json(staging_dir / "manifest.json", preflight)
    base.write_json(staging_dir / "environment.json", preflight["environment"])
    base.write_json(staging_dir / "data_preflight.json", protocol["audit"])
    log(f"start arm={args.arm} kind={args.mode} requested={requested} branch={preflight['git']['branch']} head={preflight['git']['head']}")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    uuids: list[str] = []
    try:
        for fold in folds:
            frame, fold_metrics, instance_uuid = run_fold(protocol, checkpoint, fold, staging_dir / f"fold_{fold:02d}", log)
            predictions.append(frame)
            metrics.append(fold_metrics)
            uuids.append(instance_uuid)
        if args.mode == "formal":
            aggregate = aggregate_formal(staging_dir, args.arm, predictions, metrics, uuids, original_oof)
            log(f"formal_oof acc={aggregate['oof_metrics']['acc']:.9f} bacc={aggregate['oof_metrics']['bacc']:.9f} macro_auc={aggregate['oof_metrics']['macro_auc']:.9f}")
        else:
            audit = base.read_json(staging_dir / "fold_00/model_audit.json")
            base.write_json(
                staging_dir / "smoke_summary.json",
                {
                    "smoke_passed": True,
                    "excluded_from_formal_results": True,
                    "arm": args.arm,
                    "fold": 0,
                    "metrics": metrics[0],
                    "fresh_classifier_uuid": uuids[0],
                    "checks": {"cuda_available": True, "checkpoint_loaded": True, "requested_n_estimators": requested, "effective_n_estimators": requested, "internal_feature_subspace_members": requested, "full_feature_union_count": audit["feature_coverage"]["member_feature_set_union_count"], "uncovered_feature_count": audit["feature_coverage"]["uncovered_feature_count"], "all_modalities_100_percent": True, "predict_proba_shape": metrics[0]["probability_validation"]["shape"], "probability_rows_sum_to_one": True, "fixed_class_order": list(base.CLASS_ORDER), "no_nan_or_inf": True, "artifacts_saved_and_read_back": True, "metrics_exactly_recomputed": True},
                },
            )
        preflight["experiment_run"].update({"completed_at_utc": base.utc_now(), "wall_seconds": time.perf_counter() - started, "completed": True, "fresh_classifier_instance_uuids": uuids})
        base.write_json(staging_dir / "manifest.json", preflight)
        log(f"complete arm={args.arm} kind={args.mode} wall_seconds={preflight['experiment_run']['wall_seconds']:.3f}")
        if args.mode == "formal":
            base.require(validate_formal_readback(staging_dir)["passed"], "Formal readback failed")
        staging_dir.rename(final_dir)
        print("FORMAL_READBACK_VALIDATION=True" if args.mode == "formal" else "SMOKE_PASSED=True", flush=True)
    except BaseException as exc:
        base.write_json(staging_dir / "failure.json", {"failed_at_utc": base.utc_now(), "exception_type": type(exc).__name__, "message": str(exc), "credentials_recorded": False})
        log(f"failed arm={args.arm} kind={args.mode} exception={type(exc).__name__} message={exc}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded TabPFN E4/E8 inference arms.")
    parser.add_argument("--arm", choices=("e4", "e8"), required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default=str(RESULT_ROOT))
    parser.add_argument("--original-oof", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
