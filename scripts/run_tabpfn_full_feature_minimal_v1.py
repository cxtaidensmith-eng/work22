from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

import run_tabpfn_baseline_v1 as base


ROOT = base.ROOT
EXPERIMENT_NAME = "TabPFN-3 Full-Feature Minimal-Coverage v1"
EXPECTED_BRANCH = "experiment/tabpfn-full-feature-minimal-v1"
BASE_BRANCH = "experiment/tabpfn-baseline-v1"
BASE_COMMIT = "57ecc323edf9d24d32d8be7c0eb1b612e4cca74a"
DEFAULT_OUTPUT_PARENT = ROOT / "experiments/tabpfn_full_feature_minimal_v1"
SINGLE_ESTIMATOR_OOF_PATH = (
    ROOT / "experiments/tabpfn_baseline_v1/formal/oof_predictions.csv"
)
EXPECTED_SINGLE_ESTIMATOR_OOF_SHA256 = (
    "f6a667371a8d5c91939c78a533f31e7c950eadfe9bac51f2fa8de256af6e4f04"
)

BASELINE_CONSTRUCTOR = {
    "n_estimators": 1,
    "auto_scale_n_estimators": False,
    "random_state": 0,
    "device": "cuda:0",
    "fit_mode": "low_memory",
    "memory_saving_mode": True,
    "inference_precision": "auto",
    "show_progress_bar": False,
}
FULL_FEATURE_CONSTRUCTOR = {
    **BASELINE_CONSTRUCTOR,
    "auto_scale_n_estimators": True,
}

MODAL_NAME_MAP = {
    "UCSFFSX": "MRI",
    "UCBERKELEYAV45": "PET",
    "UPENNBIOMK9": "CSF",
    "RISK_FACTOR": "Risk",
    "COGNITIVE_TEST": "Cognitive",
    "ROI_AVERAGE": "ROI",
}
MODAL_ORDER = ("MRI", "PET", "CSF", "Risk", "Cognitive", "ROI")

base.EXPECTED_BRANCH = EXPECTED_BRANCH
base.BASE_COMMIT = BASE_COMMIT


def protocol_delta_audit() -> dict[str, Any]:
    changed = {
        key: {"baseline": BASELINE_CONSTRUCTOR[key], "full_feature": value}
        for key, value in FULL_FEATURE_CONSTRUCTOR.items()
        if BASELINE_CONSTRUCTOR[key] != value
    }
    base.require(
        changed == {
            "auto_scale_n_estimators": {
                "baseline": False,
                "full_feature": True,
            }
        },
        f"Unexpected constructor delta: {changed}",
    )
    return {
        "passed": True,
        "baseline_runner": "scripts/run_tabpfn_baseline_v1.py",
        "baseline_runner_sha256": base.file_sha256(
            ROOT / "scripts/run_tabpfn_baseline_v1.py"
        ),
        "inherited_without_reimplementation": [
            "load_protocol",
            "checkpoint_audit",
            "environment_audit",
            "metric_bundle",
            "validate_fold_readback",
            "fold_metric_table",
            "fold_metric_summary",
            "recompute_historical_metrics",
        ],
        "constructor_differences": changed,
        "allowed_non_model_differences": [
            "experiment/branch/result/report names",
            "two-member feature coverage audit",
            "comparison with locked single-estimator OOF",
        ],
    }


def feature_modalities(feature_names: list[str]) -> tuple[list[str], dict[str, int]]:
    modal_dict = np.load(base.MODAL_DICT_PATH, allow_pickle=True).item()
    name_to_modal: dict[str, str] = {}
    modal_sizes: dict[str, int] = {}
    for raw_name, columns in modal_dict.items():
        modal = MODAL_NAME_MAP[str(raw_name)]
        modal_sizes[modal] = len(columns)
        for column in columns:
            base.require(str(column) not in name_to_modal, f"Duplicate modal feature: {column}")
            name_to_modal[str(column)] = modal
    base.require(tuple(modal_sizes) == MODAL_ORDER, "Unexpected modal order")
    modalities = [name_to_modal[name] for name in feature_names]
    base.require(len(modalities) == 360, "Expected 360 modal assignments")
    base.require(
        modal_sizes
        == {
            "MRI": 138,
            "PET": 150,
            "CSF": 3,
            "Risk": 36,
            "Cognitive": 24,
            "ROI": 9,
        },
        f"Unexpected modal sizes: {modal_sizes}",
    )
    return modalities, modal_sizes


def make_classifier(checkpoint: Path) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(checkpoint),
        n_estimators=1,
        auto_scale_n_estimators=True,
        random_state=0,
        device="cuda:0",
        fit_mode="low_memory",
        memory_saving_mode=True,
        inference_precision="auto",
        show_progress_bar=False,
    )


def model_audit(classifier: Any, checkpoint: Path, instance_uuid: str) -> dict[str, Any]:
    params = classifier.get_params(deep=False)
    strict = {"model_path": str(checkpoint), **FULL_FEATURE_CONSTRUCTOR}
    for name, expected in strict.items():
        actual = params[name]
        if name == "model_path":
            base.require(
                Path(actual).resolve() == checkpoint.resolve(),
                "Effective checkpoint path changed",
            )
        else:
            base.require(actual == expected, f"Effective parameter changed: {name}")

    base.require(classifier.n_estimators == 1, "Requested n_estimators changed")
    base.require(classifier.auto_scale_n_estimators is True, "Auto scaling is disabled")
    base.require(classifier.n_estimators_ == 2, "Effective n_estimators_ is not 2")
    base.require(len(classifier.ensemble_configs_) == 2, "Expected two internal members")
    base.require(len(classifier.models_) == 1, "Expected one loaded checkpoint model")
    base.require(
        tuple(str(device) for device in classifier.devices_) == ("cuda:0",),
        "Effective device is not cuda:0",
    )
    base.require(
        np.array_equal(np.asarray(classifier.classes_, dtype=np.int64), base.CLASS_IDS),
        "Effective class order changed",
    )
    base.require(int(classifier.n_features_in_) == 360, "Model did not receive 360 features")

    configs = classifier.ensemble_configs_
    model_indices = [int(config._model_index) for config in configs]
    base.require(model_indices == [0, 0], f"Members do not share one model: {model_indices}")
    raw_member_indices = classifier.ensemble_preprocessor_.subsample_feature_indices
    base.require(len(raw_member_indices) == 2, "Feature index list is not length two")
    feature_names = np.asarray(classifier.feature_names_in_, dtype=object).astype(str)
    modalities, modal_sizes = feature_modalities(feature_names.tolist())

    members: list[dict[str, Any]] = []
    member_sets: list[set[int]] = []
    coverage_counts = np.zeros(360, dtype=np.int64)
    for member_index, (config, raw_indices) in enumerate(
        zip(configs, raw_member_indices, strict=True)
    ):
        if raw_indices is None:
            indices = np.arange(360, dtype=np.int64)
        else:
            indices = np.asarray(raw_indices, dtype=np.int64)
        base.require(indices.ndim == 1, f"member {member_index} indices not 1D")
        base.require(np.unique(indices).size == indices.size, f"member {member_index} duplicate indices")
        base.require(np.all((indices >= 0) & (indices < 360)), f"member {member_index} index out of range")
        member_set = set(indices.tolist())
        member_sets.append(member_set)
        coverage_counts[indices] += 1
        modal_counts = {
            modal: int(sum(modalities[index] == modal for index in indices))
            for modal in MODAL_ORDER
        }
        preprocess = config.preprocess_config
        members.append(
            {
                "member": member_index,
                "model_index": int(config._model_index),
                "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
                "preprocess_name": preprocess.name,
                "preprocess_repr": str(preprocess),
                "max_features_per_estimator": int(
                    preprocess.max_features_per_estimator
                ),
                "feature_shift_count": int(config.feature_shift_count),
                "feature_shift_decoder": config.feature_shift_decoder,
                "original_feature_count": int(indices.size),
                "original_feature_indices": indices.tolist(),
                "original_feature_names": feature_names[indices].tolist(),
                "original_feature_indices_sha256": base.raw_array_sha256(indices),
                "modal_feature_counts": modal_counts,
            }
        )

    feature_union = sorted(member_sets[0] | member_sets[1])
    feature_intersection = sorted(member_sets[0] & member_sets[1])
    uncovered = sorted(set(range(360)) - set(feature_union))
    duplicate_coverage = int((coverage_counts > 1).sum())
    union_modal_counts = {
        modal: int(sum(modalities[index] == modal for index in feature_union))
        for modal in MODAL_ORDER
    }
    union_modal_rates = {
        modal: union_modal_counts[modal] / modal_sizes[modal] for modal in MODAL_ORDER
    }
    coverage_payload = {
        "member_feature_indices": [member["original_feature_indices"] for member in members],
        "feature_union": feature_union,
        "feature_intersection": feature_intersection,
        "coverage_counts": coverage_counts.tolist(),
        "modal_union_counts": union_modal_counts,
    }
    coverage_sha = base.canonical_sha256(coverage_payload)
    base.require(feature_union == list(range(360)), "Internal members do not cover 0..359")
    base.require(not uncovered, "Uncovered original features remain")
    base.require(all(rate == 1.0 for rate in union_modal_rates.values()), "A modality is not fully covered")

    transforms = [
        {
            "name": config.name,
            "categorical_name": config.categorical_name,
            "append_original": config.append_original,
            "max_features_per_estimator": int(config.max_features_per_estimator),
            "global_transformer_name": config.global_transformer_name,
        }
        for config in classifier.inference_config_.PREPROCESS_TRANSFORMS
    ]
    base.require(
        all(item["max_features_per_estimator"] == 200 for item in transforms),
        "Checkpoint preprocessing feature limit changed",
    )

    gates = {
        "requested_n_estimators_is_one": classifier.n_estimators == 1,
        "auto_scale_enabled": classifier.auto_scale_n_estimators is True,
        "effective_n_estimators_is_two": classifier.n_estimators_ == 2,
        "two_internal_members": len(classifier.ensemble_configs_) == 2,
        "one_loaded_checkpoint_model": len(classifier.models_) == 1,
        "members_share_model_index_zero": model_indices == [0, 0],
        "cuda_0_only": tuple(str(device) for device in classifier.devices_)
        == ("cuda:0",),
        "class_order_fixed": np.array_equal(
            np.asarray(classifier.classes_, dtype=np.int64), base.CLASS_IDS
        ),
        "feature_union_is_0_to_359": feature_union == list(range(360)),
        "uncovered_feature_count_is_zero": len(uncovered) == 0,
        "all_modal_union_rates_are_one": all(
            rate == 1.0 for rate in union_modal_rates.values()
        ),
    }
    base.require(all(gates.values()), f"Full-feature gates failed: {gates}")

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
        "effective_devices": [str(device) for device in classifier.devices_],
        "effective_classes": np.asarray(classifier.classes_, dtype=np.int64).tolist(),
        "n_features_received": int(classifier.n_features_in_),
        "n_train_samples": int(classifier.n_train_samples_),
        "use_autocast": bool(classifier.use_autocast_),
        "forced_inference_dtype": base.safe_json(classifier.forced_inference_dtype_),
        "all_checkpoint_preprocess_transforms": transforms,
        "model_semantics": {
            "external_model_count": 1,
            "checkpoint_count": 1,
            "independently_trained_model_count": 0,
            "seed_ensemble": False,
            "probability_fusion_with_original_or_seps": False,
            "internal_feature_subspace_members": 2,
            "members_share_same_pretrained_weights": True,
            "required_description": "One TabPFN-3 checkpoint and one TabPFNClassifier with two automatically scaled internal feature-subspace inference members, used solely to cover all 360 input features.",
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
            "duplicate_coverage_feature_count": duplicate_coverage,
            "coverage_counts_by_original_feature": coverage_counts.tolist(),
            "modal_total_feature_counts": modal_sizes,
            "modal_union_feature_counts": union_modal_counts,
            "modal_union_coverage_rates": union_modal_rates,
            "feature_coverage_sha256": coverage_sha,
        },
        "internal_original_feature_subsampling": {
            "external_feature_selection_performed": False,
            "full_input_feature_count": 360,
            "selected_original_feature_count": len(feature_union),
            "selected_original_feature_indices": feature_union,
            "selected_original_feature_names": feature_names[feature_union].tolist(),
            "selected_original_feature_indices_sha256": base.raw_array_sha256(
                np.asarray(feature_union, dtype=np.int64)
            ),
            "disclosure": "Two automatically scaled internal members jointly cover all 360 unchanged input columns.",
        },
        "full_feature_minimal_coverage_gates": gates,
    }


base.make_classifier = make_classifier
base.model_audit = model_audit


def write_feature_coverage_files(fold_dir: Path) -> dict[str, Any]:
    audit = base.read_json(fold_dir / "model_audit.json")
    coverage = audit["feature_coverage"]
    members = coverage["members"]
    modalities, _ = feature_modalities(
        audit["internal_original_feature_subsampling"]["selected_original_feature_names"]
    )
    # The selected union is exactly 0..359, so this list is the original feature order.
    feature_names = audit["internal_original_feature_subsampling"][
        "selected_original_feature_names"
    ]
    member_sets = []
    for member in members:
        indices = np.asarray(member["original_feature_indices"], dtype=np.int64)
        member_sets.append(set(indices.tolist()))
        frame = pd.DataFrame(
            {
                "member": int(member["member"]),
                "feature_index": indices,
                "feature_name": np.asarray(feature_names, dtype=object)[indices],
                "modality": np.asarray(modalities, dtype=object)[indices],
            }
        )
        path = fold_dir / f"member_{int(member['member']):02d}_feature_indices.csv"
        frame.to_csv(path, index=False)
        readback = pd.read_csv(path)
        base.require(
            np.array_equal(
                readback["feature_index"].to_numpy(dtype=np.int64), indices
            ),
            f"Feature-index readback failed: {path}",
        )

    counts = np.asarray(
        coverage["coverage_counts_by_original_feature"], dtype=np.int64
    )
    per_feature = pd.DataFrame(
        {
            "feature_index": np.arange(360, dtype=np.int64),
            "feature_name": feature_names,
            "modality": modalities,
            "member_00": [int(index in member_sets[0]) for index in range(360)],
            "member_01": [int(index in member_sets[1]) for index in range(360)],
            "coverage_count": counts,
        }
    )
    per_feature.to_csv(fold_dir / "feature_coverage.csv", index=False)
    readback = pd.read_csv(fold_dir / "feature_coverage.csv")
    base.require(readback.shape == (360, 6), "Feature coverage CSV shape mismatch")
    base.require(
        np.array_equal(readback["coverage_count"].to_numpy(dtype=np.int64), counts),
        "Feature coverage count readback failed",
    )
    return {
        "passed": True,
        "feature_coverage_sha256": coverage["feature_coverage_sha256"],
        "member_file_sha256": {
            f"member_{member_index:02d}": base.file_sha256(
                fold_dir / f"member_{member_index:02d}_feature_indices.csv"
            )
            for member_index in range(2)
        },
        "feature_coverage_csv_sha256": base.file_sha256(
            fold_dir / "feature_coverage.csv"
        ),
    }


def run_fold(
    protocol: dict[str, Any],
    checkpoint: Path,
    fold: int,
    fold_dir: Path,
    log,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    predictions, metrics, instance_uuid = base.run_fold(
        protocol, checkpoint, fold, fold_dir, log
    )
    validation = write_feature_coverage_files(fold_dir)
    base.write_json(fold_dir / "feature_coverage_validation.json", validation)
    log(
        f"fold={fold} feature_coverage=360/360 internal_members=2 "
        f"coverage_sha256={validation['feature_coverage_sha256']}"
    )
    return predictions, metrics, instance_uuid


def paired_comparison(
    full_oof: pd.DataFrame,
    other_path: Path,
    other_name: str,
    expected_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    base.require(other_path.is_file(), f"Historical OOF missing: {other_path}")
    actual_sha = base.file_sha256(other_path)
    base.require(actual_sha == expected_sha256, f"{other_name} OOF SHA mismatch")
    other = (
        pd.read_csv(other_path, float_precision="round_trip")
        .sort_values("subject_index")
        .reset_index(drop=True)
    )
    full = full_oof.sort_values("subject_index").reset_index(drop=True)
    base.require(other.shape[0] == 598, f"{other_name} row count")
    base.require(other["subject_index"].nunique() == 598, f"{other_name} uniqueness")
    for column in ("subject_index", "truth", "fold"):
        base.require(
            np.array_equal(full[column], other[column]),
            f"{other_name} alignment failed: {column}",
        )
    truth = full["truth"].to_numpy(dtype=np.int64)
    full_prediction = full["prediction"].to_numpy(dtype=np.int64)
    other_prediction = other["prediction"].to_numpy(dtype=np.int64)
    full_correct = full_prediction == truth
    other_correct = other_prediction == truth
    full_only = full_correct & ~other_correct
    other_only = ~full_correct & other_correct
    discordant = int(full_only.sum() + other_only.sum())
    p_value = (
        float(binomtest(int(full_only.sum()), discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )
    rows = pd.DataFrame(
        {
            "fold": full["fold"].to_numpy(dtype=np.int64),
            "subject_index": full["subject_index"].to_numpy(dtype=np.int64),
            "truth": truth,
            "full_feature_prediction": full_prediction,
            f"{other_name}_prediction": other_prediction,
            "full_feature_correct": full_correct.astype(np.int64),
            f"{other_name}_correct": other_correct.astype(np.int64),
            "full_feature_only_correct": full_only.astype(np.int64),
            f"{other_name}_only_correct": other_only.astype(np.int64),
        }
    )
    rows.to_csv(output_dir / f"paired_rows_{other_name}.csv", index=False)
    by_class = {}
    for class_id, class_name in enumerate(base.CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {
            "support": int(mask.sum()),
            "full_feature_correct": int((full_correct & mask).sum()),
            f"{other_name}_correct": int((other_correct & mask).sum()),
            "full_feature_only_correct": int((full_only & mask).sum()),
            f"{other_name}_only_correct": int((other_only & mask).sum()),
            "both_wrong": int((~full_correct & ~other_correct & mask).sum()),
        }
    result = {
        "other_name": other_name,
        "other_oof_path": str(other_path.resolve()),
        "other_oof_sha256": actual_sha,
        "alignment": {
            "alignment_key": "subject_index",
            "row_count": 598,
            "unique_subject_indices": 598,
            "subject_indices_equal": True,
            "truths_equal": True,
            "fold_assignments_equal": True,
        },
        "counts": {
            "both_correct": int((full_correct & other_correct).sum()),
            "full_feature_only_correct": int(full_only.sum()),
            f"{other_name}_only_correct": int(other_only.sum()),
            "both_wrong": int((~full_correct & ~other_correct).sum()),
            "discordant": discordant,
        },
        "exact_mcnemar_two_sided_p": p_value,
        "by_true_class": by_class,
        "other_metrics_recomputed": base.recompute_historical_metrics(other),
        "corrected_subject_indices": rows.loc[
            full_only, "subject_index"
        ].astype(int).tolist(),
        "broken_subject_indices": rows.loc[
            other_only, "subject_index"
        ].astype(int).tolist(),
    }
    base.write_json(output_dir / f"paired_summary_{other_name}.json", result)
    return result


def aggregate_feature_coverage(formal_dir: Path) -> dict[str, Any]:
    audits = [
        base.read_json(formal_dir / f"fold_{fold:02d}/model_audit.json")
        for fold in base.FOLDS
    ]
    feature_names = audits[0]["internal_original_feature_subsampling"][
        "selected_original_feature_names"
    ]
    modalities, modal_sizes = feature_modalities(feature_names)
    table: dict[str, Any] = {
        "feature_index": np.arange(360, dtype=np.int64),
        "feature_name": feature_names,
        "modality": modalities,
    }
    fold_rows = []
    coverage_matrix = []
    for fold, audit in enumerate(audits):
        gates = audit["full_feature_minimal_coverage_gates"]
        base.require(all(gates.values()), f"fold {fold} coverage gate failed")
        coverage = audit["feature_coverage"]
        counts = np.asarray(
            coverage["coverage_counts_by_original_feature"], dtype=np.int64
        )
        base.require(np.all(counts >= 1), f"fold {fold} has uncovered feature")
        coverage_matrix.append(counts)
        table[f"fold_{fold:02d}_coverage_count"] = counts
        fold_rows.append(
            {
                "fold": fold,
                "requested_n_estimators": audit["requested_n_estimators"],
                "effective_n_estimators": audit["effective_n_estimators"],
                "internal_member_count": audit[
                    "internal_feature_subspace_member_count"
                ],
                "member_feature_counts": [
                    member["original_feature_count"]
                    for member in coverage["members"]
                ],
                "intersection_count": coverage[
                    "member_feature_set_intersection_count"
                ],
                "union_count": coverage["member_feature_set_union_count"],
                "uncovered_count": coverage["uncovered_feature_count"],
                "duplicate_coverage_feature_count": coverage[
                    "duplicate_coverage_feature_count"
                ],
                "modal_union_feature_counts": coverage[
                    "modal_union_feature_counts"
                ],
                "modal_union_coverage_rates": coverage[
                    "modal_union_coverage_rates"
                ],
                "feature_coverage_sha256": coverage[
                    "feature_coverage_sha256"
                ],
            }
        )
    matrix = np.asarray(coverage_matrix, dtype=np.int64)
    table["minimum_coverage_count_across_folds"] = matrix.min(axis=0)
    table["maximum_coverage_count_across_folds"] = matrix.max(axis=0)
    table["total_coverage_count_across_folds"] = matrix.sum(axis=0)
    pooled = pd.DataFrame(table)
    pooled.to_csv(formal_dir / "pooled_feature_coverage.csv", index=False)
    pooled_sha = base.file_sha256(formal_dir / "pooled_feature_coverage.csv")
    result = {
        "passed": True,
        "fold_count": 10,
        "all_folds_effective_n_estimators_two": all(
            row["effective_n_estimators"] == 2 for row in fold_rows
        ),
        "all_folds_union_is_360": all(row["union_count"] == 360 for row in fold_rows),
        "all_folds_uncovered_count_zero": all(
            row["uncovered_count"] == 0 for row in fold_rows
        ),
        "all_folds_all_modalities_100_percent": all(
            all(rate == 1.0 for rate in row["modal_union_coverage_rates"].values())
            for row in fold_rows
        ),
        "modal_total_feature_counts": modal_sizes,
        "unique_feature_coverage_sha256_values": sorted(
            {row["feature_coverage_sha256"] for row in fold_rows}
        ),
        "pooled_feature_coverage_csv_sha256": pooled_sha,
        "pooled_feature_coverage_sha256": base.canonical_sha256(
            {
                "fold_coverage_counts": matrix.tolist(),
                "modalities": modalities,
            }
        ),
        "folds": fold_rows,
    }
    base.require(
        all(
            result[key]
            for key in (
                "passed",
                "all_folds_effective_n_estimators_two",
                "all_folds_union_is_360",
                "all_folds_uncovered_count_zero",
                "all_folds_all_modalities_100_percent",
            )
        ),
        "Pooled feature coverage failed",
    )
    base.write_json(formal_dir / "feature_coverage_summary.json", result)
    return result


def aggregate_formal(
    staging_dir: Path,
    predictions: list[pd.DataFrame],
    fold_metrics: list[dict[str, Any]],
    instance_uuids: list[str],
    original_oof: Path,
) -> dict[str, Any]:
    result = base.aggregate_formal(
        staging_dir,
        predictions,
        fold_metrics,
        instance_uuids,
        original_oof,
    )
    oof = pd.read_csv(
        staging_dir / "oof_predictions.csv", float_precision="round_trip"
    )
    single_comparison = paired_comparison(
        oof,
        SINGLE_ESTIMATOR_OOF_PATH,
        "tabpfn_single_estimator_v1",
        EXPECTED_SINGLE_ESTIMATOR_OOF_SHA256,
        staging_dir,
    )
    coverage = aggregate_feature_coverage(staging_dir)
    result["comparisons"]["tabpfn_single_estimator_v1"] = single_comparison
    result["feature_coverage_summary"] = coverage
    result["model_semantics"] = {
        "external_model_count": 1,
        "checkpoint_count": 1,
        "independently_trained_model_count": 0,
        "internal_feature_subspace_members_per_fold": 2,
        "seed_ensemble": False,
        "probability_fusion": False,
    }
    base.write_json(staging_dir / "aggregate_summary.json", result)
    return result


def validate_formal_readback(formal_dir: Path) -> dict[str, Any]:
    probability_columns = [f"probability_{name}" for name in base.CLASS_ORDER]
    oof = pd.read_csv(
        formal_dir / "oof_predictions.csv", float_precision="round_trip"
    )
    stored = base.read_json(formal_dir / "oof_metrics.json")
    recomputed = base.metric_bundle(
        oof["truth"].to_numpy(dtype=np.int64),
        oof[probability_columns].to_numpy(dtype=np.float64),
    )
    metric_exact = {
        name: recomputed[name] == stored[name]
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    base.require(all(metric_exact.values()), f"OOF metric mismatch: {metric_exact}")
    base.require(
        recomputed["confusion_matrix"] == stored["confusion_matrix"],
        "OOF confusion mismatch",
    )
    base.require(oof.shape[0] == 598, "OOF row count")
    base.require(oof["subject_index"].nunique() == 598, "OOF subject uniqueness")
    base.require(oof["shuffled_position"].nunique() == 598, "OOF position uniqueness")
    base.require(
        np.array_equal(
            oof["prediction"].to_numpy(dtype=np.int64),
            base.CLASS_IDS[
                np.argmax(
                    oof[probability_columns].to_numpy(dtype=np.float64), axis=1
                )
            ],
        ),
        "OOF prediction is not probability argmax",
    )

    frames = []
    uuids = []
    fold_checks = {}
    for fold in base.FOLDS:
        fold_dir = formal_dir / f"fold_{fold:02d}"
        frame = pd.read_csv(
            fold_dir / "predictions.csv", float_precision="round_trip"
        )
        frames.append(frame)
        metrics = base.read_json(fold_dir / "metrics.json")
        fold_recomputed = base.metric_bundle(
            frame["truth"].to_numpy(dtype=np.int64),
            frame[probability_columns].to_numpy(dtype=np.float64),
        )
        exact = all(
            fold_recomputed[name] == metrics[name]
            for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        ) and fold_recomputed["confusion_matrix"] == metrics["confusion_matrix"]
        base.require(exact, f"fold {fold} metric mismatch")
        audit = base.read_json(fold_dir / "model_audit.json")
        gates = audit["full_feature_minimal_coverage_gates"]
        base.require(all(gates.values()), f"fold {fold} coverage gates failed")
        coverage_validation = base.read_json(
            fold_dir / "feature_coverage_validation.json"
        )
        base.require(coverage_validation["passed"], f"fold {fold} coverage readback")
        uuids.append(audit["instance_uuid"])
        fold_checks[f"fold_{fold:02d}"] = {
            "passed": True,
            "rows": int(frame.shape[0]),
            "metrics_exact": True,
            "effective_n_estimators": audit["effective_n_estimators"],
            "feature_union_count": audit["feature_coverage"][
                "member_feature_set_union_count"
            ],
            "uncovered_feature_count": audit["feature_coverage"][
                "uncovered_feature_count"
            ],
            "feature_coverage_sha256": audit["feature_coverage"][
                "feature_coverage_sha256"
            ],
        }
    concatenated = (
        pd.concat(frames, ignore_index=True)
        .sort_values("subject_index")
        .reset_index(drop=True)
    )
    base.require(
        np.array_equal(concatenated.to_numpy(), oof.to_numpy()),
        "OOF differs from fold concatenation",
    )
    base.require(len(set(uuids)) == 10, "Fresh classifier UUID check failed")
    coverage = base.read_json(formal_dir / "feature_coverage_summary.json")
    base.require(coverage["passed"], "Pooled feature coverage summary failed")
    result = {
        "validated_at_utc": base.utc_now(),
        "passed": True,
        "oof_rows": 598,
        "oof_unique_subject_indices": 598,
        "oof_unique_shuffled_positions": 598,
        "oof_metric_exactness": metric_exact,
        "oof_confusion_matrix_exact": True,
        "oof_equals_fold_concatenation": True,
        "fresh_classifier_uuid_count": len(set(uuids)),
        "all_folds_effective_n_estimators_two": True,
        "all_folds_cover_all_360_features": True,
        "all_folds_uncovered_feature_count_zero": True,
        "pooled_feature_coverage_sha256": coverage[
            "pooled_feature_coverage_sha256"
        ],
        "folds": fold_checks,
        "artifact_sha256": {
            path.relative_to(formal_dir).as_posix(): base.file_sha256(path)
            for path in sorted(formal_dir.rglob("*"))
            if path.is_file() and path.name != "readback_validation.json"
        },
    }
    base.write_json(formal_dir / "readback_validation.json", result)
    return result


def preflight_payload(protocol: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    payload = base.preflight_payload(protocol, checkpoint)
    delta = protocol_delta_audit()
    payload["experiment"] = EXPERIMENT_NAME
    payload["git"].update(
        {
            "base_branch": BASE_BRANCH,
            "base_commit": BASE_COMMIT,
            "new_branch": EXPECTED_BRANCH,
            "remote_branch": f"origin/{EXPECTED_BRANCH}",
        }
    )
    payload["constructor"] = {
        "model_path": str(checkpoint),
        **FULL_FEATURE_CONSTRUCTOR,
        "all_other_parameters": "tabpfn==8.2.0 defaults",
    }
    payload["protocol"].update(
        {
            "requested_n_estimators": 1,
            "expected_effective_n_estimators": 2,
            "auto_scale_n_estimators": True,
            "external_model_count": 1,
            "checkpoint_count": 1,
            "independently_trained_model_count": 0,
            "internal_feature_subspace_members": 2,
            "seed_ensemble": False,
            "probability_fusion": False,
            "full_feature_union_required": list(range(360)),
        }
    )
    payload["protocol_delta_audit"] = delta
    payload["reference_oof"] = {
        "original_query": {
            "path": "external local hash-pinned reference",
            "sha256": base.EXPECTED_ORIGINAL_OOF_SHA256,
        },
        "seps_q_v1": {
            "path": base.SEPS_OOF_PATH.relative_to(ROOT).as_posix(),
            "sha256": base.EXPECTED_SEPS_OOF_SHA256,
        },
        "tabpfn_single_estimator_v1": {
            "path": SINGLE_ESTIMATOR_OOF_PATH.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_SINGLE_ESTIMATOR_OOF_SHA256,
        },
    }
    payload["historical_headline_metrics"]["tabpfn_single_estimator_v1"] = {
        "acc": 0.7642140468227425,
        "macro_f1": 0.7664825798136402,
        "bacc": 0.7610217930033637,
        "macro_auc_probability_recomputed": 0.9120034762565993,
        "weighted_f1": 0.7642741002057829,
    }
    payload["source_files"] = {
        "inherited_baseline_runner": {
            "path": "scripts/run_tabpfn_baseline_v1.py",
            "sha256": base.file_sha256(ROOT / "scripts/run_tabpfn_baseline_v1.py"),
        },
        "runner": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
        "recompute": {
            "path": "scripts/recompute_tabpfn_full_feature_minimal_v1.py",
            "sha256": base.file_sha256(
                ROOT / "scripts/recompute_tabpfn_full_feature_minimal_v1.py"
            ),
        },
    }
    return payload


def run_experiment(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    protocol = base.load_protocol()
    preflight = preflight_payload(protocol, checkpoint)
    base.require(
        base.file_sha256(SINGLE_ESTIMATOR_OOF_PATH)
        == EXPECTED_SINGLE_ESTIMATOR_OOF_SHA256,
        "Single-estimator OOF SHA mismatch",
    )
    if args.mode == "preflight":
        print(
            json.dumps(
                base.safe_json(preflight),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    run_kind = args.mode
    folds = (0,) if run_kind == "smoke" else base.FOLDS
    output_parent = Path(args.output_parent).resolve()
    if run_kind == "formal":
        base.require(args.original_oof is not None, "Formal mode requires --original-oof")
        original_oof = Path(args.original_oof).resolve()
        base.require(original_oof.is_file(), f"Original OOF missing: {original_oof}")
        base.require(
            base.file_sha256(original_oof) == base.EXPECTED_ORIGINAL_OOF_SHA256,
            "Original OOF SHA mismatch",
        )
        smoke_dir = output_parent / "smoke"
        base.require(smoke_dir.is_dir(), "Formal mode requires completed smoke")
        smoke_summary = base.read_json(smoke_dir / "smoke_summary.json")
        base.require(smoke_summary["smoke_passed"] is True, "Smoke did not pass")
        smoke_audit = base.read_json(smoke_dir / "fold_00/model_audit.json")
        base.require(
            all(smoke_audit["full_feature_minimal_coverage_gates"].values()),
            "Smoke full-feature gates failed",
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    final_dir = output_parent / run_kind
    staging_dir = output_parent / f".{run_kind}.staging"
    base.require(not final_dir.exists(), f"Refusing to overwrite: {final_dir}")
    base.require(not staging_dir.exists(), f"Retained staging exists: {staging_dir}")
    staging_dir.mkdir(parents=False)
    log_path = staging_dir / "run.txt"

    def log(message: str) -> None:
        line = f"{base.utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    started = time.perf_counter()
    preflight["experiment_run"] = {
        "kind": run_kind,
        "folds": list(folds),
        "started_at_utc": base.utc_now(),
        "formal_reportable": run_kind == "formal",
    }
    base.write_json(staging_dir / "manifest.json", preflight)
    base.write_json(staging_dir / "environment.json", preflight["environment"])
    base.write_json(staging_dir / "data_preflight.json", protocol["audit"])
    base.write_json(
        staging_dir / "protocol_delta_audit.json",
        preflight["protocol_delta_audit"],
    )
    log(
        f"start kind={run_kind} branch={preflight['git']['branch']} "
        f"head={preflight['git']['head']}"
    )
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    instance_uuids: list[str] = []
    try:
        for fold in folds:
            frame, fold_metrics, instance_uuid = run_fold(
                protocol,
                checkpoint,
                fold,
                staging_dir / f"fold_{fold:02d}",
                log,
            )
            predictions.append(frame)
            metrics.append(fold_metrics)
            instance_uuids.append(instance_uuid)
        if run_kind == "formal":
            aggregate = aggregate_formal(
                staging_dir,
                predictions,
                metrics,
                instance_uuids,
                original_oof,
            )
            log(
                f"formal_oof acc={aggregate['oof_metrics']['acc']:.9f} "
                f"macro_auc={aggregate['oof_metrics']['macro_auc']:.9f}"
            )
        else:
            audit = base.read_json(staging_dir / "fold_00/model_audit.json")
            smoke_summary = {
                "smoke_passed": True,
                "excluded_from_formal_results": True,
                "fold": 0,
                "metrics": metrics[0],
                "fresh_classifier_uuid": instance_uuids[0],
                "checks": {
                    "cuda_available": True,
                    "checkpoint_loaded": True,
                    "requested_n_estimators": 1,
                    "effective_n_estimators": 2,
                    "internal_feature_subspace_members": 2,
                    "full_feature_union_count": audit["feature_coverage"][
                        "member_feature_set_union_count"
                    ],
                    "uncovered_feature_count": audit["feature_coverage"][
                        "uncovered_feature_count"
                    ],
                    "all_modalities_100_percent": True,
                    "predict_proba_shape": metrics[0]["probability_validation"][
                        "shape"
                    ],
                    "probability_rows_sum_to_one": True,
                    "fixed_class_order": list(base.CLASS_ORDER),
                    "no_nan_or_inf": True,
                    "artifacts_saved_and_read_back": True,
                    "metrics_exactly_recomputed": True,
                },
            }
            base.write_json(staging_dir / "smoke_summary.json", smoke_summary)

        preflight["experiment_run"].update(
            {
                "completed_at_utc": base.utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "completed": True,
                "fresh_classifier_instance_uuids": instance_uuids,
            }
        )
        base.write_json(staging_dir / "manifest.json", preflight)
        log(
            f"complete kind={run_kind} "
            f"wall_seconds={preflight['experiment_run']['wall_seconds']:.3f}"
        )
        if run_kind == "formal":
            validation = validate_formal_readback(staging_dir)
            base.require(validation["passed"], "Formal readback failed")
        else:
            base.require(
                base.read_json(staging_dir / "smoke_summary.json")["smoke_passed"],
                "Smoke summary failed",
            )
        staging_dir.rename(final_dir)
        print(
            "FORMAL_READBACK_VALIDATION=True"
            if run_kind == "formal"
            else "SMOKE_PASSED=True",
            flush=True,
        )
    except BaseException as exc:
        base.write_json(
            staging_dir / "failure.json",
            {
                "failed_at_utc": base.utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "credentials_recorded": False,
            },
        )
        log(f"failed kind={run_kind} exception={type(exc).__name__} message={exc}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN-3 minimal two-member full-feature coverage."
    )
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-parent", default=str(DEFAULT_OUTPUT_PARENT))
    parser.add_argument("--original-oof", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
