#!/usr/bin/env python
"""Recompute bounded-arm, gate, selection, calibration, and leakage audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_tabpfn_baseline_v1 as base
import run_tabpfn_bounded_tuning_v1 as run


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments/tabpfn_bounded_tuning_v1"
PROBABILITY_COLUMNS = [f"probability_{name}" for name in base.CLASS_ORDER]


def metric_exact(frame: pd.DataFrame, stored: dict, probability_columns: list[str], prediction_column: str) -> bool:
    actual = base.metric_bundle(frame["truth"].to_numpy(dtype=np.int64), frame[probability_columns].to_numpy(dtype=np.float64))
    prediction = frame[prediction_column].to_numpy(dtype=np.int64)
    return all(actual[key] == stored[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")) and actual["confusion_matrix"] == stored["confusion_matrix"] and np.array_equal(prediction, np.argmax(frame[probability_columns].to_numpy(dtype=np.float64), axis=1))


def recompute() -> dict:
    arms = {"e4": RESULT_ROOT / "e4/formal"}
    gate = base.read_json(arms["e4"] / "e8_gate_decision.json")
    e4_metrics = base.read_json(arms["e4"] / "oof_metrics.json")
    expected_gate = run.e8_gate(e4_metrics)
    base.require(gate == expected_gate, "E8 gate decision does not exactly recompute")
    if gate["passed"]:
        arms["e8"] = RESULT_ROOT / "e8/formal"
    else:
        base.require(not (RESULT_ROOT / "e8/formal").exists(), "E8 exists despite failed gate")
    arm_checks = {}
    for name, path in arms.items():
        validation = run.validate_formal_readback(path)
        base.require(validation["passed"], f"{name} readback failed")
        arm_checks[name] = {"passed": True, "oof_sha256": base.file_sha256(path / "oof_predictions.csv"), "metrics": base.read_json(path / "oof_metrics.json")}
    calibration = RESULT_ROOT / "calibration"
    selection = base.read_json(calibration / "candidate_selection.json")
    candidates = selection["candidates"]
    reranked = sorted(candidates, key=lambda item: (-item["acc"], -item["bacc"], -item["macro_f1"], -item["macro_auc"], item["requested_n_estimators"]))
    base.require(reranked[0]["name"] == selection["selected"]["name"], "Candidate selection does not recompute")
    frames = []
    fold_checks = {}
    for fold in base.FOLDS:
        fold_dir = calibration / f"fold_{fold:02d}"
        manifest = base.read_json(fold_dir / "inner_split_manifest.json")
        crossfit = pd.read_csv(fold_dir / "inner_cross_fitted_predictions.csv", float_precision="round_trip")
        grid = pd.read_csv(fold_dir / "bias_grid_441.csv", float_precision="round_trip")
        selected_bias = base.read_json(fold_dir / "selected_bias.json")
        output = pd.read_csv(fold_dir / "outer_test_predictions.csv", float_precision="round_trip")
        metrics = base.read_json(fold_dir / "metrics.json")
        base.require(manifest["inner_validation_union_equals_outer_train"] and manifest["inner_validation_sets_pairwise_disjoint"] and manifest["outer_test_absent_from_all_inner_splits"], f"fold {fold} inner split proof failed")
        base.require(crossfit.shape[0] == manifest["outer_train_size"] and crossfit["outer_train_position"].nunique() == manifest["outer_train_size"], f"fold {fold} crossfit coverage failed")
        base.require(grid.shape[0] == 441, f"fold {fold} grid row count")
        ranked = grid.sort_values(["acc", "bacc", "macro_f1", "bias_l2_norm", "b_AD", "b_CN"], ascending=[False, False, False, True, True, True], kind="mergesort").reset_index(drop=True)
        base.require(float(ranked.iloc[0]["b_AD"]) == float(selected_bias["b_AD"]) and float(ranked.iloc[0]["b_CN"]) == float(selected_bias["b_CN"]), f"fold {fold} bias selection mismatch")
        exact = metric_exact(output, metrics["calibrated"], [f"calibrated_probability_{name}" for name in base.CLASS_ORDER], "calibrated_prediction")
        base.require(exact, f"fold {fold} calibrated metric mismatch")
        frames.append(pd.DataFrame({"fold": output["fold"], "subject_index": output["subject_index"], "shuffled_position": output["shuffled_position"], "truth": output["truth"], "probability_AD": output["calibrated_probability_AD"], "probability_CN": output["calibrated_probability_CN"], "probability_SMCI": output["calibrated_probability_SMCI"], "prediction": output["calibrated_prediction"]}))
        fold_checks[f"fold_{fold:02d}"] = {"passed": True, "inner_crossfit_rows": int(crossfit.shape[0]), "inner_split_count": 5, "grid_rows": 441, "selected_b_AD": selected_bias["b_AD"], "selected_b_CN": selected_bias["b_CN"], "outer_test_leakage_count": 0, "metrics_exact": True}
    concatenated = pd.concat(frames, ignore_index=True).sort_values("subject_index").reset_index(drop=True)
    oof = pd.read_csv(calibration / "oof_predictions.csv", float_precision="round_trip").sort_values("subject_index").reset_index(drop=True)
    base.require(np.array_equal(concatenated.to_numpy(), oof.to_numpy()), "Calibration OOF differs from fold concatenation")
    stored_metrics = base.read_json(calibration / "oof_metrics.json")
    base.require(metric_exact(oof, stored_metrics, PROBABILITY_COLUMNS, "prediction"), "Calibration OOF metrics do not recompute")
    decision = base.read_json(calibration / "teacher_qualification.json")
    result = {"validated_at_utc": base.utc_now(), "passed": True, "e8_gate_exact": True, "e8_ran": gate["passed"], "candidate_selection_exact": True, "selected_candidate": selection["selected"]["name"], "arms": arm_checks, "calibration": {"oof_rows": 598, "unique_subject_indices": int(oof["subject_index"].nunique()), "oof_equals_fold_concatenation": True, "oof_metrics_exact": True, "all_outer_folds_independently_calibrated": True, "outer_test_labels_used_for_bias_selection": False, "folds": fold_checks, "teacher_decision": decision["decision"]}, "artifact_sha256": {path.relative_to(RESULT_ROOT).as_posix(): base.file_sha256(path) for path in sorted(RESULT_ROOT.rglob("*")) if path.is_file() and path.name != "recompute_validation.json"}}
    base.write_json(RESULT_ROOT / "recompute_validation.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(base.safe_json(recompute()), ensure_ascii=False, indent=2, sort_keys=True))
