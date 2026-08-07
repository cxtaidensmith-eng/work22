#!/usr/bin/env python
"""Run the fixed zero-training TPCF v1 fusion candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import tpcf_common as common


ROOT = common.ROOT
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
RULE_THRESHOLDS = (0.60, 0.70)
RULE_ALPHAS = (0.20, 0.30, 0.40)


def candidate_row(
    name: str,
    family: str,
    alpha: float,
    threshold: float | None,
    probability: np.ndarray,
    truth: np.ndarray,
    original: np.ndarray,
    active: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = common.metric_bundle(truth, probability)
    comparison = common.comparison_with_original(truth, original, probability)
    detail = {
        "name": name,
        "family": family,
        "alpha": alpha,
        "threshold": threshold,
        "active_samples": int(active.sum()),
        "metrics": metrics,
        "comparison": comparison,
    }
    row = {
        "name": name,
        "family": family,
        "alpha": alpha,
        "threshold": threshold,
        "active_samples": int(active.sum()),
        **{key: metrics[key] for key in ("correct", "acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
        "changed_predictions": comparison["changed_predictions"],
        "repairs": comparison["repairs"],
        "damages": comparison["damages"],
    }
    return row, detail


def run(args: argparse.Namespace) -> None:
    output = common.unique_directory(ROOT / "experiments/tpcf_v1/zero_training")
    output.mkdir(parents=True, exist_ok=False)
    aligned, provenance = common.load_aligned_outer_oof(args.original_oof, args.tabpfn_oof)
    truth, original, tabpfn = common.aligned_probabilities(aligned)
    original_prediction = np.argmax(original, axis=1)
    tabpfn_prediction = np.argmax(tabpfn, axis=1)
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    active_masks: dict[str, np.ndarray] = {}

    for alpha in ALPHAS:
        name = f"arithmetic_alpha_{alpha:.2f}"
        probability = (1.0 - alpha) * original + alpha * tabpfn
        active = np.ones(truth.size, dtype=bool)
        row, detail = candidate_row(name, "arithmetic", alpha, None, probability, truth, original, active)
        rows.append(row)
        details[name] = detail
        probabilities[name] = probability
        active_masks[name] = active

    log_original = np.log(np.clip(original, 1e-12, 1.0))
    log_tabpfn = np.log(np.clip(tabpfn, 1e-12, 1.0))
    for alpha in ALPHAS:
        name = f"log_probability_alpha_{alpha:.2f}"
        probability = common.softmax((1.0 - alpha) * log_original + alpha * log_tabpfn)
        active = np.ones(truth.size, dtype=bool)
        row, detail = candidate_row(name, "log_probability", alpha, None, probability, truth, original, active)
        rows.append(row)
        details[name] = detail
        probabilities[name] = probability
        active_masks[name] = active

    margin_original = common.probability_margin(original)
    margin_tabpfn = common.probability_margin(tabpfn)
    entropy_original = common.probability_entropy(original)
    entropy_tabpfn = common.probability_entropy(tabpfn)
    disagreement = original_prediction != tabpfn_prediction
    for threshold in RULE_THRESHOLDS:
        base_active = (
            disagreement
            & (margin_tabpfn > margin_original)
            & (entropy_tabpfn < entropy_original)
            & (tabpfn.max(axis=1) >= threshold)
        )
        for alpha in RULE_ALPHAS:
            name = f"reliability_threshold_{threshold:.2f}_alpha_{alpha:.2f}"
            probability = original.copy()
            probability[base_active] = (
                (1.0 - alpha) * original[base_active] + alpha * tabpfn[base_active]
            )
            row, detail = candidate_row(
                name, "reliability_rule", alpha, threshold, probability, truth, original, base_active
            )
            rows.append(row)
            details[name] = detail
            probabilities[name] = probability
            active_masks[name] = base_active

    common.require(len(rows) == 20, "Expected exactly 20 fixed zero-training candidates")
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row["correct"]),
            -float(row["macro_auc"]),
            -float(row["macro_f1"]),
            float(row["alpha"]),
            str(row["name"]),
        ),
    )
    best = ranked[0]
    best_name = str(best["name"])
    best_probability = probabilities[best_name]
    best_prediction = np.argmax(best_probability, axis=1)
    best_active = active_masks[best_name]
    original_correct = original_prediction == truth
    best_correct = best_prediction == truth

    result_table = pd.DataFrame(rows)
    result_table.to_csv(output / "all_zero_training_results.csv", index=False, float_format="%.17g")
    best_frame = aligned.copy()
    best_frame["best_candidate"] = best_name
    best_frame["active"] = best_active.astype(np.int64)
    for index, class_name in enumerate(common.CLASS_ORDER):
        best_frame[f"fused_probability_{class_name}"] = best_probability[:, index]
    best_frame["fused_prediction"] = best_prediction
    best_frame["changed_prediction"] = (best_prediction != original_prediction).astype(np.int64)
    best_frame["repair"] = ((~original_correct) & best_correct).astype(np.int64)
    best_frame["damage"] = (original_correct & (~best_correct)).astype(np.int64)
    best_frame.to_csv(output / "best_zero_training_predictions.csv", index=False, float_format="%.17g")

    complement = common.complementarity(truth, original, tabpfn)
    complement["run_gate"] = bool(
        complement["oracle_union_correct"] >= 565
        and complement["tabpfn_only_correct"] >= 10
    )
    common.write_json(output / "complementarity.json", complement)
    best_payload = {
        "best_candidate": details[best_name],
        "ranking_rule": ["Correct/ACC", "Probability Macro-AUC", "Macro-F1", "smaller TabPFN weight"],
        "all_candidates": details,
        "input_provenance": provenance,
        "run_gate": complement["run_gate"],
    }
    common.write_json(output / "best_zero_training_metrics.json", best_payload)
    common.write_json(
        output / "config.json",
        {
            "experiment": "TabPFN–Deep Conservative Fusion v1 / zero training",
            "class_order": list(common.CLASS_ORDER),
            "arithmetic_alphas": list(ALPHAS),
            "log_probability_alphas": list(ALPHAS),
            "rule_thresholds": list(RULE_THRESHOLDS),
            "rule_active_alphas": list(RULE_ALPHAS),
            "rule_fusion": "arithmetic probability blend on activated samples",
            "input_provenance": provenance,
        },
    )

    report = [
        "# TPCF v1 — zero-training fusion",
        "",
        f"Original: {provenance['original_metrics']['correct']}/598; "
        f"TabPFN E4: {provenance['tabpfn_metrics']['correct']}/598.",
        "",
        f"Complementarity: both={complement['both_correct']}, "
        f"Original-only={complement['original_only_correct']}, "
        f"TabPFN-only={complement['tabpfn_only_correct']}, "
        f"both-wrong={complement['both_wrong']}, oracle={complement['oracle_union_correct']}.",
        "",
        f"Best fixed candidate: **{best_name}**, {best['correct']}/598, "
        f"ACC={best['acc']:.10f}, Macro-F1={best['macro_f1']:.10f}, "
        f"BACC={best['bacc']:.10f}, Macro-AUC={best['macro_auc']:.10f}; "
        f"repairs={best['repairs']}, damages={best['damages']}.",
        "",
        f"RUN_GATE={complement['run_gate']}.",
    ]
    (output / "zero_training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"Zero-training complete best={best_name} correct={best['correct']} "
        f"repairs={best['repairs']} damages={best['damages']} RUN_GATE={complement['run_gate']}",
        flush=True,
    )
    if not complement["run_gate"]:
        print("STOP_NO_USABLE_COMPLEMENTARITY", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPCF v1 fixed zero-training fusion")
    parser.add_argument("--original-oof")
    parser.add_argument("--tabpfn-oof")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
