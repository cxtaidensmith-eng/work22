#!/usr/bin/env python
"""Select one locked arm and calibrate class logits using outer-train cross-fitting."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold

import run_tabpfn_baseline_v1 as base
import run_tabpfn_bounded_tuning_v1 as run


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments/tabpfn_bounded_tuning_v1"
CALIBRATION_DIR = RESULT_ROOT / "calibration"
PROBABILITY_COLUMNS = [f"probability_{name}" for name in base.CLASS_ORDER]
BIAS_VALUES = np.round(np.arange(-1.0, 1.0001, 0.1), 1)


def candidate_paths() -> dict[str, dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = {
        "e2": {"requested_n_estimators": 1, "effective_n_estimators": 2, "formal_dir": run.E2_FORMAL},
        "e4": {"requested_n_estimators": 4, "effective_n_estimators": 4, "formal_dir": RESULT_ROOT / "e4/formal"},
    }
    gate_path = RESULT_ROOT / "e4/formal/e8_gate_decision.json"
    base.require(gate_path.is_file(), "E4 gate decision is missing")
    gate = base.read_json(gate_path)
    e8_dir = RESULT_ROOT / "e8/formal"
    if gate["passed"]:
        base.require(e8_dir.is_dir(), "E8 gate passed but formal E8 is missing")
        paths["e8"] = {"requested_n_estimators": 8, "effective_n_estimators": 8, "formal_dir": e8_dir}
    else:
        base.require(not e8_dir.exists(), "E8 exists despite failed gate")
    return paths


def select_candidate(output_dir: Path) -> dict[str, Any]:
    candidates = []
    for name, item in candidate_paths().items():
        formal_dir = Path(item["formal_dir"])
        validation = base.read_json(formal_dir / "readback_validation.json")
        base.require(validation["passed"] is True, f"{name} readback did not pass")
        metrics = base.read_json(formal_dir / "oof_metrics.json")
        correct = int(sum(metrics["confusion_matrix"][index][index] for index in range(3)))
        candidates.append(
            {
                "name": name,
                "requested_n_estimators": item["requested_n_estimators"],
                "effective_n_estimators": item["effective_n_estimators"],
                "formal_dir": formal_dir.relative_to(ROOT).as_posix(),
                "oof_path": (formal_dir / "oof_predictions.csv").relative_to(ROOT).as_posix(),
                "oof_sha256": base.file_sha256(formal_dir / "oof_predictions.csv"),
                "correct_count": correct,
                **{key: metrics[key] for key in ("acc", "bacc", "macro_f1", "macro_auc", "weighted_f1")},
            }
        )
    ranked = sorted(candidates, key=lambda item: (-item["acc"], -item["bacc"], -item["macro_f1"], -item["macro_auc"], item["requested_n_estimators"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["sorting_tuple"] = [-item["acc"], -item["bacc"], -item["macro_f1"], -item["macro_auc"], item["requested_n_estimators"]]
    selected = ranked[0]
    result = {
        "schema_version": 1,
        "selection_is_exploratory": True,
        "fixed_selection_order": ["pooled OOF ACC descending", "BACC descending", "Macro-F1 descending", "probability Macro-AUC descending", "requested n_estimators ascending"],
        "candidates": ranked,
        "selected": selected,
        "selected_estimator_count_requested": selected["requested_n_estimators"],
        "selected_estimator_count_effective": selected["effective_n_estimators"],
        "reason": f"{selected['name']} ranked first under the immutable lexicographic candidate rule.",
    }
    base.write_json(output_dir / "candidate_selection.json", result)
    return result


def apply_bias(probability: np.ndarray, b_ad: float, b_cn: float) -> np.ndarray:
    logits = np.log(np.maximum(np.asarray(probability, dtype=np.float64), 1e-12))
    logits = logits + np.asarray([b_ad, b_cn, 0.0], dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    base.require(np.isfinite(calibrated).all(), "Calibrated probabilities contain NaN/Inf")
    base.require(np.allclose(calibrated.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "Calibrated probabilities do not sum to one")
    return calibrated


def grid_search(y_true: np.ndarray, probability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for b_ad in BIAS_VALUES:
        for b_cn in BIAS_VALUES:
            calibrated = apply_bias(probability, float(b_ad), float(b_cn))
            metrics = base.metric_bundle(y_true, calibrated)
            rows.append(
                {
                    "b_AD": float(b_ad),
                    "b_CN": float(b_cn),
                    "b_SMCI": 0.0,
                    "correct_count": int(round(metrics["acc"] * len(y_true))),
                    "acc": metrics["acc"],
                    "bacc": metrics["bacc"],
                    "macro_f1": metrics["macro_f1"],
                    "bias_l2_norm": math.hypot(float(b_ad), float(b_cn)),
                }
            )
    frame = pd.DataFrame(rows)
    ranked = frame.sort_values(["acc", "bacc", "macro_f1", "bias_l2_norm", "b_AD", "b_CN"], ascending=[False, False, False, True, True, True], kind="mergesort").reset_index(drop=True)
    selected = ranked.iloc[0].to_dict()
    selected.update(
        {
            "selection_rule": ["ACC descending", "BACC descending", "Macro-F1 descending", "bias L2 norm ascending", "b_AD ascending", "b_CN ascending"],
            "grid_size": 441,
            "grid_b_AD": [-1.0, 1.0, 0.1],
            "grid_b_CN": [-1.0, 1.0, 0.1],
            "b_SMCI_fixed": 0.0,
            "selection_inputs": "outer-train cross-fitted probabilities and outer-train labels only",
            "outer_test_labels_used_for_selection": False,
        }
    )
    return frame, base.safe_json(selected)


def compact_inner_audit(audit: dict[str, Any], inner_fold: int) -> dict[str, Any]:
    return {
        "inner_fold": inner_fold,
        "instance_uuid": audit["instance_uuid"],
        "requested_n_estimators": audit["requested_n_estimators"],
        "effective_n_estimators": audit["effective_n_estimators"],
        "internal_member_count": audit["internal_feature_subspace_member_count"],
        "loaded_checkpoint_model_count": audit["loaded_checkpoint_model_count"],
        "effective_classes": audit["effective_classes"],
        "feature_union_count": audit["feature_coverage"]["member_feature_set_union_count"],
        "uncovered_feature_count": audit["feature_coverage"]["uncovered_feature_count"],
        "modal_union_coverage_rates": audit["feature_coverage"]["modal_union_coverage_rates"],
        "feature_coverage_sha256": audit["feature_coverage"]["feature_coverage_sha256"],
        "all_gates_passed": all(audit["bounded_inference_gates"].values()),
    }


def calibrate_outer_fold(protocol: dict[str, Any], checkpoint: Path, selected: dict[str, Any], outer_fold: int, fold_dir: Path, log) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_dir.mkdir(parents=True, exist_ok=False)
    fold_entry = protocol["fold_manifest"]["folds"][outer_fold]
    outer_train = np.asarray(fold_entry["train_positions"], dtype=np.int64)
    outer_test = np.asarray(fold_entry["test_positions"], dtype=np.int64)
    X = protocol["X"]
    y = protocol["y"]
    source_order = protocol["source_order"]
    feature_names = protocol["feature_names"]
    run.configure_estimators(selected["requested_n_estimators"], selected["effective_n_estimators"])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    probability = np.full((outer_train.size, 3), np.nan, dtype=np.float64)
    inner_assignment = np.full(outer_train.size, -1, dtype=np.int64)
    split_rows: list[dict[str, Any]] = []
    inner_audits: list[dict[str, Any]] = []
    fit_sum = 0.0
    predict_sum = 0.0
    max_cuda_memory = 0
    started = time.perf_counter()
    for inner_fold, (relative_train, relative_validation) in enumerate(splitter.split(outer_train, y[outer_train])):
        inner_train = outer_train[np.asarray(relative_train, dtype=np.int64)]
        inner_validation = outer_train[np.asarray(relative_validation, dtype=np.int64)]
        base.require(not set(inner_train.tolist()) & set(inner_validation.tolist()), "Inner train/validation overlap")
        base.require(not set(outer_test.tolist()) & (set(inner_train.tolist()) | set(inner_validation.tolist())), "Outer-test leakage into inner split")
        base.set_reproducibility()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        classifier = run.make_classifier(checkpoint)
        instance_uuid = str(uuid.uuid4())
        log(f"outer={outer_fold} inner={inner_fold} fresh_classifier_uuid={instance_uuid} train={len(inner_train)} validation={len(inner_validation)}")
        fit_started = time.perf_counter()
        classifier.fit(pd.DataFrame(X[inner_train], columns=feature_names), y[inner_train])
        torch.cuda.synchronize(0)
        fit_seconds = time.perf_counter() - fit_started
        audit = run.model_audit(classifier, checkpoint, instance_uuid)
        predict_started = time.perf_counter()
        inner_probability = np.asarray(classifier.predict_proba(pd.DataFrame(X[inner_validation], columns=feature_names)), dtype=np.float64)
        torch.cuda.synchronize(0)
        predict_seconds = time.perf_counter() - predict_started
        base.require(inner_probability.shape == (inner_validation.size, 3), "Inner probability shape mismatch")
        probability[np.asarray(relative_validation, dtype=np.int64)] = inner_probability
        inner_assignment[np.asarray(relative_validation, dtype=np.int64)] = inner_fold
        split_hash = base.canonical_sha256({"outer_fold": outer_fold, "inner_fold": inner_fold, "train_positions": inner_train.tolist(), "validation_positions": inner_validation.tolist(), "train_source_indices": source_order[inner_train].tolist(), "validation_source_indices": source_order[inner_validation].tolist()})
        split_rows.append({"inner_fold": inner_fold, "split_sha256": split_hash, "train_size": int(inner_train.size), "validation_size": int(inner_validation.size), "train_positions": inner_train.tolist(), "validation_positions": inner_validation.tolist(), "train_source_indices": source_order[inner_train].tolist(), "validation_source_indices": source_order[inner_validation].tolist(), "outer_test_disjoint": True, "instance_uuid": instance_uuid, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        inner_audits.append(compact_inner_audit(audit, inner_fold))
        fit_sum += fit_seconds
        predict_sum += predict_seconds
        max_cuda_memory = max(max_cuda_memory, int(torch.cuda.max_memory_allocated(0)))
        del classifier, inner_probability
        gc.collect()
        torch.cuda.empty_cache()
    base.require(np.isfinite(probability).all(), "Outer-train cross-fitted probability incomplete")
    base.require(np.all(inner_assignment >= 0), "Outer-train sample lacks inner validation prediction")
    base.require(np.bincount(inner_assignment, minlength=5).sum() == outer_train.size, "Inner validation union size mismatch")
    validation_sets = [set(row["validation_positions"]) for row in split_rows]
    base.require(set().union(*validation_sets) == set(outer_train.tolist()), "Inner validation union is not outer train")
    base.require(all(not (validation_sets[left] & validation_sets[right]) for left in range(5) for right in range(left + 1, 5)), "Inner validation sets are not disjoint")
    crossfit = pd.DataFrame({"outer_fold": outer_fold, "outer_train_position": outer_train, "subject_index": source_order[outer_train], "truth": y[outer_train], "inner_validation_fold": inner_assignment, "probability_AD": probability[:, 0], "probability_CN": probability[:, 1], "probability_SMCI": probability[:, 2], "prediction": np.argmax(probability, axis=1)})
    crossfit_path = fold_dir / "inner_cross_fitted_predictions.csv"
    crossfit.to_csv(crossfit_path, index=False, float_format="%.17g")
    base.require(pd.read_csv(crossfit_path, float_precision="round_trip").shape[0] == outer_train.size, "Cross-fit readback failed")
    split_manifest = {"outer_fold": outer_fold, "outer_train_size": int(outer_train.size), "outer_test_size": int(outer_test.size), "splitter": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 0}, "outer_test_positions": outer_test.tolist(), "inner_validation_union_equals_outer_train": True, "inner_validation_sets_pairwise_disjoint": True, "outer_test_absent_from_all_inner_splits": True, "each_outer_train_sample_has_one_cross_fitted_probability": True, "in_sample_probabilities_used": False, "folds": split_rows, "inner_model_audits": inner_audits, "cross_fitted_predictions_sha256": base.file_sha256(crossfit_path)}
    base.write_json(fold_dir / "inner_split_manifest.json", split_manifest)
    grid, selected_bias = grid_search(y[outer_train], probability)
    grid_path = fold_dir / "bias_grid_441.csv"
    grid.to_csv(grid_path, index=False, float_format="%.17g")
    base.require(pd.read_csv(grid_path).shape[0] == 441, "Bias grid readback row count")
    base.write_json(fold_dir / "selected_bias.json", selected_bias)
    raw_fold_path = ROOT / selected["formal_dir"] / f"fold_{outer_fold:02d}/predictions.csv"
    raw = pd.read_csv(raw_fold_path, float_precision="round_trip").sort_values("subject_index").reset_index(drop=True)
    expected_subject = np.sort(source_order[outer_test])
    base.require(np.array_equal(raw["subject_index"].to_numpy(dtype=np.int64), expected_subject), "Selected raw outer-test alignment failed")
    raw_probability = raw[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    calibrated_probability = apply_bias(raw_probability, float(selected_bias["b_AD"]), float(selected_bias["b_CN"]))
    truth = raw["truth"].to_numpy(dtype=np.int64)
    raw_prediction = raw["prediction"].to_numpy(dtype=np.int64)
    calibrated_prediction = np.argmax(calibrated_probability, axis=1).astype(np.int64)
    raw_correct = raw_prediction == truth
    calibrated_correct = calibrated_prediction == truth
    corrected = ~raw_correct & calibrated_correct
    broken = raw_correct & ~calibrated_correct
    output = pd.DataFrame({"fold": outer_fold, "subject_index": raw["subject_index"].to_numpy(dtype=np.int64), "shuffled_position": raw["shuffled_position"].to_numpy(dtype=np.int64), "truth": truth, "raw_probability_AD": raw_probability[:, 0], "raw_probability_CN": raw_probability[:, 1], "raw_probability_SMCI": raw_probability[:, 2], "raw_prediction": raw_prediction, "calibrated_probability_AD": calibrated_probability[:, 0], "calibrated_probability_CN": calibrated_probability[:, 1], "calibrated_probability_SMCI": calibrated_probability[:, 2], "calibrated_prediction": calibrated_prediction, "raw_correct": raw_correct.astype(np.int64), "calibrated_correct": calibrated_correct.astype(np.int64), "corrected": corrected.astype(np.int64), "broken": broken.astype(np.int64)})
    output_path = fold_dir / "outer_test_predictions.csv"
    output.to_csv(output_path, index=False, float_format="%.17g")
    raw_metrics = base.metric_bundle(truth, raw_probability)
    calibrated_metrics = base.metric_bundle(truth, calibrated_probability)
    metrics = {"fold": outer_fold, "train_size": int(outer_train.size), "test_size": int(outer_test.size), "split_sha256": fold_entry["split_hash"], "selected_bias": {"b_AD": selected_bias["b_AD"], "b_CN": selected_bias["b_CN"], "b_SMCI": 0.0}, "raw": raw_metrics, "calibrated": calibrated_metrics, "corrected_count": int(corrected.sum()), "broken_count": int(broken.sum()), "corrected_subject_indices": output.loc[corrected, "subject_index"].astype(int).tolist(), "broken_subject_indices": output.loc[broken, "subject_index"].astype(int).tolist(), "runtime_seconds": {"inner_fit_sum": fit_sum, "inner_predict_sum": predict_sum, "total": time.perf_counter() - started}, "cuda_peak_memory_allocated_bytes": max_cuda_memory, "outer_test_labels_used_for_bias_selection": False, "bias_selected_independently_for_this_outer_fold": True}
    base.write_json(fold_dir / "metrics.json", metrics)
    readback = pd.read_csv(output_path, float_precision="round_trip")
    actual_calibrated = base.metric_bundle(readback["truth"].to_numpy(dtype=np.int64), readback[[f"calibrated_probability_{name}" for name in base.CLASS_ORDER]].to_numpy(dtype=np.float64))
    base.require(all(actual_calibrated[key] == calibrated_metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")), "Calibrated fold metric readback mismatch")
    validation = {"passed": True, "outer_fold": outer_fold, "inner_validation_union_equals_outer_train": True, "inner_validation_sets_pairwise_disjoint": True, "outer_test_leakage_count": 0, "cross_fitted_probability_rows": int(outer_train.size), "grid_rows": 441, "outer_test_labels_used_for_selection": False, "raw_probability_finite": bool(np.isfinite(raw_probability).all()), "calibrated_probability_finite": bool(np.isfinite(calibrated_probability).all()), "calibrated_probability_rows_sum_to_one": bool(np.allclose(calibrated_probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)), "metrics_exactly_recomputed": True, "outer_test_predictions_sha256": base.file_sha256(output_path)}
    base.write_json(fold_dir / "readback_validation.json", validation)
    log(f"outer={outer_fold} selected_bias=({selected_bias['b_AD']},{selected_bias['b_CN']},0.0) raw_acc={raw_metrics['acc']:.9f} calibrated_acc={calibrated_metrics['acc']:.9f} corrected={int(corrected.sum())} broken={int(broken.sum())}")
    return output, metrics


def paired_calibrated(calibrated_oof: pd.DataFrame, other_path: Path, other_name: str, expected_sha: str, output_dir: Path) -> dict[str, Any]:
    base.require(base.file_sha256(other_path) == expected_sha, f"{other_name} OOF hash mismatch")
    other = pd.read_csv(other_path, float_precision="round_trip").sort_values("subject_index").reset_index(drop=True)
    tuned = calibrated_oof.sort_values("subject_index").reset_index(drop=True)
    for column in ("subject_index", "truth", "fold"):
        base.require(np.array_equal(tuned[column], other[column]), f"{other_name} paired alignment failed")
    truth = tuned["truth"].to_numpy(dtype=np.int64)
    tuned_prediction = tuned["prediction"].to_numpy(dtype=np.int64)
    other_prediction = other["prediction"].to_numpy(dtype=np.int64)
    tuned_correct = tuned_prediction == truth
    other_correct = other_prediction == truth
    tuned_only = tuned_correct & ~other_correct
    other_only = ~tuned_correct & other_correct
    discordant = int(tuned_only.sum() + other_only.sum())
    rows = pd.DataFrame({"fold": tuned["fold"], "subject_index": tuned["subject_index"], "truth": truth, "calibrated_prediction": tuned_prediction, f"{other_name}_prediction": other_prediction, "calibrated_correct": tuned_correct.astype(int), f"{other_name}_correct": other_correct.astype(int), "calibrated_only_correct": tuned_only.astype(int), f"{other_name}_only_correct": other_only.astype(int)})
    rows.to_csv(output_dir / f"paired_rows_{other_name}.csv", index=False)
    by_class = {}
    for class_id, class_name in enumerate(base.CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {"support": int(mask.sum()), "calibrated_correct": int((tuned_correct & mask).sum()), f"{other_name}_correct": int((other_correct & mask).sum()), "calibrated_only_correct": int((tuned_only & mask).sum()), f"{other_name}_only_correct": int((other_only & mask).sum()), "both_wrong": int((~tuned_correct & ~other_correct & mask).sum())}
    result = {"other_name": other_name, "other_oof_path": str(other_path.resolve()), "other_oof_sha256": expected_sha, "alignment": {"row_count": 598, "unique_subject_indices": 598, "subject_indices_equal": True, "truths_equal": True, "fold_assignments_equal": True}, "counts": {"both_correct": int((tuned_correct & other_correct).sum()), "calibrated_only_correct": int(tuned_only.sum()), f"{other_name}_only_correct": int(other_only.sum()), "both_wrong": int((~tuned_correct & ~other_correct).sum()), "discordant": discordant}, "exact_mcnemar_two_sided_p": float(binomtest(int(tuned_only.sum()), discordant, p=0.5).pvalue) if discordant else 1.0, "by_true_class": by_class, "other_metrics_recomputed": base.recompute_historical_metrics(other), "corrected_subject_indices": rows.loc[tuned_only, "subject_index"].astype(int).tolist(), "broken_subject_indices": rows.loc[other_only, "subject_index"].astype(int).tolist()}
    base.write_json(output_dir / f"paired_summary_{other_name}.json", result)
    return result


def qualification(calibrated_metrics: dict[str, Any], selected_metrics: dict[str, Any], fold_table: pd.DataFrame, selected_fold_table: pd.DataFrame, comparisons: dict[str, Any], candidate_selection: dict[str, Any]) -> dict[str, Any]:
    minimum_numeric = calibrated_metrics["acc"] >= 0.9200 and calibrated_metrics["bacc"] >= 0.9000 and calibrated_metrics["macro_auc"] >= 0.9700
    no_anomalous_fold = bool((fold_table["acc"] >= 0.80).all())
    meaningful_exclusive = max(comparisons["original_query"]["counts"]["calibrated_only_correct"], comparisons["seps_q_v1"]["counts"]["calibrated_only_correct"]) > 0
    inference_improved = candidate_selection["selected"]["name"] != "e2"
    raw_recall = selected_metrics["per_class"]
    cal_recall = calibrated_metrics["per_class"]
    recall_deltas = {name: cal_recall[name]["recall"] - raw_recall[name]["recall"] for name in base.CLASS_ORDER}
    class_sacrifice = calibrated_metrics["acc"] > selected_metrics["acc"] and min(recall_deltas.values()) < -0.02
    fold_volatility_worse = float(fold_table["acc"].std(ddof=1)) > float(selected_fold_table["acc"].std(ddof=1)) + 0.01
    paired_clearly_favors_both = all(comparisons[name]["counts"][f"{name}_only_correct"] >= comparisons[name]["counts"]["calibrated_only_correct"] + 5 for name in ("original_query", "seps_q_v1"))
    strong = calibrated_metrics["acc"] >= 0.9247492 and calibrated_metrics["bacc"] >= 0.9082064 and calibrated_metrics["macro_auc"] >= 0.9700
    ideal = calibrated_metrics["acc"] >= 0.9247659 and calibrated_metrics["macro_auc"] >= 0.9560491
    stop_reasons = []
    if calibrated_metrics["acc"] < 0.9200: stop_reasons.append("calibrated ACC < 0.9200")
    if calibrated_metrics["bacc"] < 0.9000: stop_reasons.append("calibrated BACC < 0.9000")
    if calibrated_metrics["macro_auc"] < 0.9700: stop_reasons.append("calibrated Macro-AUC < 0.9700")
    if not inference_improved: stop_reasons.append("E4/E8 did not win the fixed uncalibrated candidate selection")
    if class_sacrifice: stop_reasons.append("ACC gain required >0.02 recall loss in at least one class")
    if fold_volatility_worse: stop_reasons.append("fold ACC sample SD worsened by >0.01")
    if paired_clearly_favors_both: stop_reasons.append("paired correct-only counts favor both Original and SEPS by at least 5")
    minimum = minimum_numeric and no_anomalous_fold and meaningful_exclusive
    stop = bool(stop_reasons) or not minimum
    level = "STOP" if stop else ("IDEAL" if ideal else "STRONG" if strong else "MINIMUM")
    return {"schema_version": 1, "operational_definitions_fixed_before_interpretation": {"no_anomalous_fold": "all ten folds complete validation and fold ACC >= 0.80", "meaningful_nonzero_exclusive": "at least one calibrated-only correct sample versus Original or SEPS", "class_sacrifice": "pooled ACC increases while any class recall decreases by more than 0.02", "fold_volatility_worsened": "calibrated fold ACC sample SD exceeds selected raw SD by more than 0.01", "paired_clearly_favors_original_and_seps": "for both comparisons, other-only correct exceeds calibrated-only correct by at least 5"}, "threshold_checks": {"acc_at_least_0_9200": calibrated_metrics["acc"] >= 0.9200, "bacc_at_least_0_9000": calibrated_metrics["bacc"] >= 0.9000, "macro_auc_at_least_0_9700": calibrated_metrics["macro_auc"] >= 0.9700, "no_anomalous_fold": no_anomalous_fold, "meaningful_nonzero_exclusive": meaningful_exclusive, "inference_expansion_selected_over_e2": inference_improved, "class_sacrifice_detected": class_sacrifice, "fold_volatility_worsened": fold_volatility_worse, "paired_clearly_favors_original_and_seps": paired_clearly_favors_both}, "per_class_recall_delta_calibrated_minus_selected": recall_deltas, "minimum_teacher_candidate_numeric": minimum_numeric, "minimum_teacher_candidate": minimum, "strong_teacher_threshold": strong, "ideal_teacher_threshold": ideal, "decision": "STOP" if stop else "GO", "teacher_qualification_level": level, "stop_reasons": stop_reasons, "formal_conclusion": "在有限内部成员扩展和严格训练折内类别校准后，TabPFN-3仍未达到T-MEL软概率教师标准，因此停止原始TabPFN概率蒸馏方案，不再继续调参。" if stop else "TabPFN经过有限推理扩展和训练折内校准后具备一次T-MEL软概率蒸馏验证的资格。"}


def run_calibration(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base.checkpoint_audit(checkpoint)
    protocol = base.load_protocol()
    base.require(not CALIBRATION_DIR.exists(), f"Refusing to overwrite: {CALIBRATION_DIR}")
    staging = RESULT_ROOT / ".calibration.staging"
    base.require(not staging.exists(), f"Retained staging exists: {staging}")
    staging.mkdir(parents=True)
    log_path = staging / "run.txt"

    def log(message: str) -> None:
        line = f"{base.utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle: handle.write(line + "\n")

    started = time.perf_counter()
    selection = select_candidate(staging)
    selected = selection["selected"]
    manifest = {"experiment": run.EXPERIMENT_NAME, "branch": run.EXPECTED_BRANCH, "base_commit": run.BASE_COMMIT, "started_at_utc": base.utc_now(), "selected_candidate": selected, "calibration": {"form": "softmax(log(max(p,1e-12)) + [b_AD,b_CN,0])", "grid": {"b_AD": [-1.0, 1.0, 0.1], "b_CN": [-1.0, 1.0, 0.1], "b_SMCI": 0.0, "count": 441}, "outer_train_inner_split": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 0}, "outer_test_labels_used_for_selection": False, "bias_reused_across_outer_folds": False}, "checkpoint": base.checkpoint_audit(checkpoint), "environment": base.environment_audit(), "data": protocol["audit"]}
    base.write_json(staging / "manifest.json", manifest)
    outputs = []
    fold_metrics = []
    log(f"start selected={selected['name']} requested={selected['requested_n_estimators']} effective={selected['effective_n_estimators']}")
    for outer_fold in base.FOLDS:
        output, metrics = calibrate_outer_fold(protocol, checkpoint, selected, outer_fold, staging / f"fold_{outer_fold:02d}", log)
        outputs.append(output)
        fold_metrics.append(metrics)
    combined = pd.concat(outputs, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    base.require(combined.shape[0] == 598 and combined["subject_index"].nunique() == 598, "Calibrated OOF integrity failed")
    calibrated_oof = pd.DataFrame({"fold": combined["fold"].astype(int), "subject_index": combined["subject_index"].astype(int), "shuffled_position": combined["shuffled_position"].astype(int), "truth": combined["truth"].astype(int), "probability_AD": combined["calibrated_probability_AD"], "probability_CN": combined["calibrated_probability_CN"], "probability_SMCI": combined["calibrated_probability_SMCI"], "prediction": combined["calibrated_prediction"].astype(int)})
    calibrated_oof.to_csv(staging / "oof_predictions.csv", index=False, float_format="%.17g")
    metrics = base.metric_bundle(calibrated_oof["truth"].to_numpy(dtype=np.int64), calibrated_oof[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64))
    base.write_json(staging / "oof_metrics.json", metrics)
    pd.DataFrame(metrics["confusion_matrix"], index=base.CLASS_ORDER, columns=base.CLASS_ORDER).to_csv(staging / "oof_confusion_matrix.csv", index_label="truth\\prediction")
    fold_table = pd.DataFrame([{"fold": item["fold"], "train_size": item["train_size"], "test_size": item["test_size"], "split_sha256": item["split_sha256"], "acc": item["calibrated"]["acc"], "macro_f1": item["calibrated"]["macro_f1"], "bacc": item["calibrated"]["bacc"], "macro_auc": item["calibrated"]["macro_auc"], "weighted_f1": item["calibrated"]["weighted_f1"], "fit_seconds": item["runtime_seconds"]["inner_fit_sum"], "predict_proba_seconds": item["runtime_seconds"]["inner_predict_sum"], "total_seconds": item["runtime_seconds"]["total"], "cuda_peak_memory_allocated_bytes": item["cuda_peak_memory_allocated_bytes"]} for item in fold_metrics])
    fold_table.to_csv(staging / "fold_metrics.csv", index=False, float_format="%.17g")
    base.fold_metric_summary(fold_table, metrics).to_csv(staging / "metrics_summary.csv", index=False, float_format="%.17g")
    selected_oof_path = ROOT / selected["oof_path"]
    selected_metrics = base.read_json(ROOT / selected["formal_dir"] / "oof_metrics.json")
    selected_fold_table = pd.read_csv(ROOT / selected["formal_dir"] / "fold_metrics.csv", float_precision="round_trip")
    comparisons = {"selected_uncalibrated": paired_calibrated(calibrated_oof, selected_oof_path, "selected_uncalibrated", selected["oof_sha256"], staging), "e2": paired_calibrated(calibrated_oof, run.E2_OOF, "e2", run.EXPECTED_E2_OOF_SHA256, staging), "original_query": paired_calibrated(calibrated_oof, Path(args.original_oof).resolve(), "original_query", base.EXPECTED_ORIGINAL_OOF_SHA256, staging), "seps_q_v1": paired_calibrated(calibrated_oof, base.SEPS_OOF_PATH, "seps_q_v1", base.EXPECTED_SEPS_OOF_SHA256, staging)}
    decision = qualification(metrics, selected_metrics, fold_table, selected_fold_table, comparisons, selection)
    base.write_json(staging / "teacher_qualification.json", decision)
    biases = [{"fold": item["fold"], **item["selected_bias"]} for item in fold_metrics]
    pd.DataFrame(biases).to_csv(staging / "selected_biases.csv", index=False)
    summary = {"selected_candidate": selected, "selected_uncalibrated_metrics": selected_metrics, "calibrated_metrics": metrics, "correct_count": int(sum(metrics["confusion_matrix"][i][i] for i in range(3))), "selected_biases": biases, "comparisons": comparisons, "teacher_qualification": decision, "runtime_seconds": {"wall": time.perf_counter() - started, "inner_fit_sum": float(fold_table["fit_seconds"].sum()), "inner_predict_sum": float(fold_table["predict_proba_seconds"].sum())}, "cuda_peak_memory_allocated_bytes": int(fold_table["cuda_peak_memory_allocated_bytes"].max())}
    base.write_json(staging / "aggregate_summary.json", summary)
    manifest.update({"completed_at_utc": base.utc_now(), "completed": True, "wall_seconds": time.perf_counter() - started, "selected_biases": biases, "teacher_decision": decision["decision"]})
    base.write_json(staging / "manifest.json", manifest)
    log(f"complete calibrated_acc={metrics['acc']:.9f} calibrated_bacc={metrics['bacc']:.9f} calibrated_auc={metrics['macro_auc']:.9f} decision={decision['decision']}")
    staging.rename(CALIBRATION_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked outer-train cross-fitted class-bias calibration.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--original-oof", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_calibration(parse_args())
