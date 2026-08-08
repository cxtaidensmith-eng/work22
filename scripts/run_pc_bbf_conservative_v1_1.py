"""Run PC-BBF-Conservative v1.1 with the single locked cap change.

This runner deliberately reuses the validated PC-BBF v1 model, historical
loss, optimizer grouping, scheduler, and fold training implementation.  The
only model hyperparameter change is ``residual_norm_cap: 0.10 -> 0.075``.
There is no screen stage and no safe/non-destructive auxiliary loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_pc_bbf_c1_v1 as pc_v1


EXPERIMENT_NAME = "pc_bbf_conservative_v1_1"
OUTPUT_REL = Path("experiments/pc_bbf_conservative_v1_1")
PC_BBF_V1_BASE_COMMIT = "c3a2c6ce13241522cc598f58705417106ce7721c"
CAP = 0.075
SAFE_LOSS = False
FORMAL_FOLDS = tuple(range(10))
EXPECTED_PARAMETERS = 866_924
CLASS_NAMES = pc_v1.CLASS_NAMES

C1_REFERENCE = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}

PC_BBF_V1_REFERENCE = {
    "correct": 561,
    "acc": 0.9381270903010034,
    "macro_f1": 0.9266945217489257,
    "bacc": 0.915213972141583,
    "macro_auc": 0.9701189477695239,
    "weighted_f1": 0.9378308464264465,
    "confusion_matrix": [[61, 0, 11], [0, 197, 12], [4, 10, 303]],
    "parameters": 866_924,
    "gate_mean": 0.609751502805729,
    "residual_shared_ratio_mean": 0.05974636813096179,
    "cap_saturation_fraction": 0.3110368043857075,
    "repairs_vs_c1": 18,
    "damages_vs_c1": 17,
    "ad_smci_errors": 15,
    "cn_smci_errors": 22,
    "predicted_smci": 326,
}

_PC_V1_FOLD_CONFIG = pc_v1.fold_config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def conservative_config(context: dict, fold: int = -1) -> dict:
    """Return the locked protocol with the one allowed change made explicit."""

    payload = _PC_V1_FOLD_CONFIG(context, fold, CAP)
    payload.update(
        {
            "experiment": EXPERIMENT_NAME,
            "pc_bbf_cap": CAP,
            "cap_value": CAP,
            "safe_loss": SAFE_LOSS,
            "orthogonality": False,
            "device": str(context["device"]),
            "pc_bbf_v1_base_commit_sha": PC_BBF_V1_BASE_COMMIT,
            "model_change_vs_pc_bbf_v1": "residual_norm_cap: 0.10 -> 0.075",
            "formal_folds": list(FORMAL_FOLDS),
            "resume_completed_folds": True,
        }
    )
    return payload


def _patched_fold_config(context: dict, fold: int, cap_value: float) -> dict:
    require(abs(float(cap_value) - CAP) < 1e-12, f"Unexpected cap: {cap_value}")
    return conservative_config(context, fold)


# ``run_fold`` resolves this function in the imported module at call time.  By
# replacing only its metadata callback, all validated v1 training code remains
# untouched while each new fold records the correct experiment identity.
pc_v1.fold_config = _patched_fold_config


def standardize_c1_rows(path: Path) -> list[dict]:
    """Convert the tracked C1 probe OOF into the common paired-row schema."""

    require(path.is_file(), f"Tracked C1 OOF is missing: {path}")
    output = []
    for source in pc_v1.read_csv(path):
        row = {
            "fold": source["fold"],
            "subject_index": source["subject_index"],
            "truth": source["truth"],
            "prediction": source["c1_prediction"],
        }
        for name in CLASS_NAMES:
            row[f"raw_logit_{name}"] = source[f"c1_raw_logit_{name}"]
            row[f"adjusted_score_{name}"] = source[f"c1_adjusted_score_{name}"]
            row[f"probability_{name}"] = source[f"c1_probability_{name}"]
        output.append(row)
    output.sort(key=lambda row: int(row["subject_index"]))
    validate_oof_rows(output, "C1")
    validate_reference_metrics(pc_v1.metrics_from_rows(output), C1_REFERENCE, "C1")
    return output


def load_pc_bbf_v1_rows(path: Path) -> list[dict]:
    require(path.is_file(), f"Tracked PC-BBF v1 OOF is missing: {path}")
    rows = pc_v1.read_csv(path)
    rows.sort(key=lambda row: int(row["subject_index"]))
    validate_oof_rows(rows, "PC-BBF v1")
    validate_reference_metrics(pc_v1.metrics_from_rows(rows), PC_BBF_V1_REFERENCE, "PC-BBF v1")
    return rows


def validate_reference_metrics(actual: dict, expected: dict, name: str) -> None:
    require(actual["correct"] == expected["correct"], f"{name} correct-count mismatch")
    require(actual["confusion_matrix"] == expected["confusion_matrix"], f"{name} confusion mismatch")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(float(actual[key]) - float(expected[key])) <= 1e-10, f"{name} {key} mismatch")


def validate_oof_rows(rows: list[dict], name: str) -> None:
    require(len(rows) == 598, f"{name} OOF must contain 598 rows")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(subjects) == len(set(subjects)), f"{name} OOF subjects are not unique")
    require(set(subjects) == set(range(598)), f"{name} OOF subject set changed")
    require({int(row["fold"]) for row in rows} == set(FORMAL_FOLDS), f"{name} fold set changed")


def assert_no_safe_loss(context: dict) -> dict:
    """Prove the candidate contains only the historical PC-BBF v1 objective."""

    model = pc_v1.build_model(context, CAP, pc_bbf=True)
    require(parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    require(abs(float(model.pc_bbf.cap_value) - CAP) < 1e-12, "Model cap is not 0.075")
    forbidden_names = [
        name
        for name, _ in list(model.named_modules()) + list(model.named_parameters())
        if "safe" in name.lower() or "lambda_safe" in name.lower()
    ]
    require(not forbidden_names, f"Safe-loss state unexpectedly present: {forbidden_names}")
    audit = {
        "safe_loss": False,
        "lambda_safe_present": False,
        "forbidden_state_names": forbidden_names,
        "pc_bbf_cap": float(model.pc_bbf.cap_value),
        "parameter_count": parameter_count(model),
    }
    del model
    torch.cuda.empty_cache()
    return audit


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_smoke(context: dict, output_root: Path) -> dict:
    audit = assert_no_safe_loss(context)
    report = pc_v1.run_smoke(context, output_root, CAP)
    mechanism = report["mechanism_after_epoch3"]
    require(report.get("passed") is True, "Smoke did not pass")
    require(abs(float(report["cap_value"]) - CAP) < 1e-12, "Smoke cap mismatch")
    require(report["epoch0_c1_logit_max_abs_diff"] <= 1e-6, "Epoch0 equivalence failed")
    require(report["checkpoint_roundtrip_logit_max_abs_diff"] <= 1e-7, "Checkpoint roundtrip failed")
    require(mechanism["residual_shared_ratio_max"] <= CAP + 1e-6, "Residual norm cap violated")
    require(all(math.isfinite(float(value)) and float(value) > 0.0 for value in report["pc_bbf_gradient_max"].values()), "New-module smoke gradient inactive")
    require((output_root / "smoke/checkpoint_roundtrip.pt").is_file(), "Smoke checkpoint is missing")
    report.update(
        {
            "experiment": EXPERIMENT_NAME,
            "pc_bbf_cap": CAP,
            "safe_loss": False,
            "safe_loss_audit": audit,
            "parameter_count": EXPECTED_PARAMETERS,
            "pc_bbf_v1_base_commit_sha": PC_BBF_V1_BASE_COMMIT,
            "run_command": "python -u -B scripts/run_pc_bbf_conservative_v1_1.py smoke --device cuda:0",
        }
    )
    pc_v1.write_json(output_root / "config.json", conservative_config(context))
    pc_v1.write_json(output_root / "smoke/smoke_report.json", report)
    return report


def directional_errors(rows: list[dict]) -> dict[str, int]:
    output = {}
    for truth_index, truth_name in enumerate(CLASS_NAMES):
        for prediction_index, prediction_name in enumerate(CLASS_NAMES):
            if truth_index != prediction_index:
                output[f"{truth_name}_to_{prediction_name}"] = sum(
                    int(row["truth"]) == truth_index and int(row["prediction"]) == prediction_index
                    for row in rows
                )
    return output


def prediction_counts(rows: list[dict]) -> dict[str, int]:
    return {
        name: sum(int(row["prediction"]) == class_index for row in rows)
        for class_index, name in enumerate(CLASS_NAMES)
    }


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float | int]:
    return {
        "correct": int(candidate["correct"] - baseline["correct"]),
        **{
            key: float(candidate[key] - baseline[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }


def pair_status(candidate: dict, baseline: dict) -> str:
    truth = int(candidate["truth"])
    old = int(baseline["prediction"])
    new = int(candidate["prediction"])
    if old != truth and new == truth:
        return "repair"
    if old == truth and new != truth:
        return "damage"
    if old != new:
        return "changed_without_correctness_change"
    return "unchanged"


def paired_rows(candidate_rows: list[dict], c1_rows: list[dict], v1_rows: list[dict]) -> list[dict]:
    c1_map = {int(row["subject_index"]): row for row in c1_rows}
    v1_map = {int(row["subject_index"]): row for row in v1_rows}
    output = []
    for candidate in candidate_rows:
        subject = int(candidate["subject_index"])
        c1 = c1_map[subject]
        v1 = v1_map[subject]
        require(int(candidate["truth"]) == int(c1["truth"]) == int(v1["truth"]), "Paired truth mismatch")
        require(int(candidate["fold"]) == int(c1["fold"]) == int(v1["fold"]), "Paired fold mismatch")
        output.append(
            {
                "fold": int(candidate["fold"]),
                "subject_index": subject,
                "truth": int(candidate["truth"]),
                "c1_prediction": int(c1["prediction"]),
                "pc_bbf_v1_prediction": int(v1["prediction"]),
                "conservative_prediction": int(candidate["prediction"]),
                "status_vs_c1": pair_status(candidate, c1),
                "status_vs_pc_bbf_v1": pair_status(candidate, v1),
            }
        )
    return output


def comparison(candidate_rows: list[dict], baseline_rows: list[dict]) -> dict:
    baseline_map = {int(row["subject_index"]): row for row in baseline_rows}
    return pc_v1.compare_rows(candidate_rows, baseline_map)


def conservative_decision(metrics: dict, vs_c1: dict, boundary: dict) -> tuple[str, dict]:
    target = {
        "correct_at_least_563": metrics["correct"] >= 563,
        "acc_at_least_0p94147": metrics["acc"] >= 0.94147,
        "macro_f1_not_below_c1": metrics["macro_f1"] >= C1_REFERENCE["macro_f1"],
        "bacc_drop_vs_c1_at_most_0p002": metrics["bacc"] >= C1_REFERENCE["bacc"] - 0.002,
    }
    positive = {
        "correct_equals_562": metrics["correct"] == 562,
        "macro_f1_within_0p003_of_pc_bbf_v1": metrics["macro_f1"] >= PC_BBF_V1_REFERENCE["macro_f1"] - 0.003,
        "bacc_within_0p002_of_c1": metrics["bacc"] >= C1_REFERENCE["bacc"] - 0.002,
    }
    stable = {
        "correct_equals_561": metrics["correct"] == 561,
        "damages_vs_c1_below_pc_bbf_v1_17": vs_c1["damages"] < PC_BBF_V1_REFERENCE["damages_vs_c1"],
        "cn_smci_errors_below_pc_bbf_v1_22": boundary["CN_SMCI"] < PC_BBF_V1_REFERENCE["cn_smci_errors"],
        "macro_f1_within_0p003_of_pc_bbf_v1": metrics["macro_f1"] >= PC_BBF_V1_REFERENCE["macro_f1"] - 0.003,
        "bacc_not_below_pc_bbf_v1": metrics["bacc"] >= PC_BBF_V1_REFERENCE["bacc"],
    }
    no_gain = {
        "correct_below_561": metrics["correct"] < 561,
        "repairs_not_above_damages": vs_c1["repairs"] <= vs_c1["damages"],
        "ad_smci_improvement_mostly_lost": boundary["AD_SMCI"] >= 20,
        "all_key_metrics_below_pc_bbf_v1": all(
            metrics[key] < PC_BBF_V1_REFERENCE[key]
            for key in ("acc", "macro_f1", "bacc", "macro_auc")
        ),
    }
    checks = {"target": target, "positive": positive, "stable": stable, "no_gain": no_gain}
    # The preregistered NO_GAIN clauses are hard OR gates and therefore veto
    # every positive label, including an otherwise numerical TARGET.
    if any(no_gain.values()):
        return "PC_BBF_CONSERVATIVE_NO_GAIN", checks
    if all(target.values()):
        return "PC_BBF_CONSERVATIVE_TARGET", checks
    if all(positive.values()):
        return "PC_BBF_CONSERVATIVE_POSITIVE", checks
    if all(stable.values()):
        return "PC_BBF_CONSERVATIVE_STABLE", checks
    return "PC_BBF_CONSERVATIVE_NO_GAIN", checks


def write_oof_views(output_root: Path, rows: list[dict]) -> None:
    pc_v1.write_csv(output_root / "oof_predictions.csv", rows)
    pc_v1.write_csv(
        output_root / "oof_logits.csv",
        [
            {
                "fold": row["fold"],
                "subject_index": row["subject_index"],
                "truth": row["truth"],
                **{f"raw_logit_{name}": row[f"raw_logit_{name}"] for name in CLASS_NAMES},
                **{f"adjusted_score_{name}": row[f"adjusted_score_{name}"] for name in CLASS_NAMES},
            }
            for row in rows
        ],
    )
    pc_v1.write_csv(
        output_root / "oof_probabilities.csv",
        [
            {
                "fold": row["fold"],
                "subject_index": row["subject_index"],
                "truth": row["truth"],
                **{f"probability_{name}": row[f"probability_{name}"] for name in CLASS_NAMES},
            }
            for row in rows
        ],
    )


def write_markdown(path: Path, report: dict) -> None:
    metrics = report["metrics"]
    vs_c1 = report["comparison_vs_c1"]
    vs_v1 = report["comparison_vs_pc_bbf_v1"]
    directions = report["directional_errors"]["candidate"]
    mechanism = report["mechanism"]
    lines = [
        "# PC-BBF-Conservative v1.1 Formal Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"- PC-BBF v1 base commit: `{report['pc_bbf_v1_base_commit_sha']}`",
        f"- Configuration: `pc_bbf_cap={report['pc_bbf_cap']}`, `safe_loss={report['safe_loss']}`",
        f"- Correct: {metrics['correct']}/598",
        f"- ACC: {metrics['acc']:.7f}",
        f"- Macro-F1: {metrics['macro_f1']:.7f}",
        f"- BACC: {metrics['bacc']:.7f}",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.7f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.7f}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        f"- Predicted AD/CN/sMCI: {report['prediction_counts']['AD']} / {report['prediction_counts']['CN']} / {report['prediction_counts']['SMCI']}",
        f"- Ten-fold ACC mean +/- sample SD: {report['fold_acc_mean']:.7f} +/- {report['fold_acc_sample_std']:.7f}",
        f"- Versus C1 repairs/damages/changed: {vs_c1['repairs']} / {vs_c1['damages']} / {vs_c1['changed_predictions']}",
        f"- Versus PC-BBF v1 repairs/damages/changed: {vs_v1['repairs']} / {vs_v1['damages']} / {vs_v1['changed_predictions']}",
        f"- Metric delta versus C1 (Correct/ACC/F1/BACC/AUC): "
        f"{report['metric_deltas']['vs_c1']['correct']:+d} / {report['metric_deltas']['vs_c1']['acc']:+.7f} / "
        f"{report['metric_deltas']['vs_c1']['macro_f1']:+.7f} / {report['metric_deltas']['vs_c1']['bacc']:+.7f} / "
        f"{report['metric_deltas']['vs_c1']['macro_auc']:+.7f}",
        f"- Metric delta versus PC-BBF v1 (Correct/ACC/F1/BACC/AUC): "
        f"{report['metric_deltas']['vs_pc_bbf_v1']['correct']:+d} / {report['metric_deltas']['vs_pc_bbf_v1']['acc']:+.7f} / "
        f"{report['metric_deltas']['vs_pc_bbf_v1']['macro_f1']:+.7f} / {report['metric_deltas']['vs_pc_bbf_v1']['bacc']:+.7f} / "
        f"{report['metric_deltas']['vs_pc_bbf_v1']['macro_auc']:+.7f}",
        f"- AD-sMCI / CN-sMCI / AD-CN errors: {report['boundary_errors']['candidate']['AD_SMCI']} / {report['boundary_errors']['candidate']['CN_SMCI']} / {report['boundary_errors']['candidate']['AD_CN']}",
        f"- CN->sMCI: {directions['CN_to_SMCI']}; sMCI->AD: {directions['SMCI_to_AD']}",
        f"- Gate mean/std/min/max: {mechanism['gate_mean']:.7f} / {mechanism['gate_std']:.7f} / {mechanism['gate_min']:.7f} / {mechanism['gate_max']:.7f}",
        f"- Residual/shared mean/max: {mechanism['residual_shared_ratio_mean']:.7f} / {mechanism['residual_shared_ratio_max']:.7f}",
        f"- Cap saturation: {mechanism['cap_saturation_fraction']:.7f}",
        f"- New-module maximum gradient: {report['new_module_max_gradient']:.7g}",
        f"- Parameters: {report['parameter_count']}",
        f"- Formal training time: {report['training_seconds']:.3f} s",
        f"- Reproduction: `{report['run_command']}`",
        "",
        "## Required questions",
        "",
        f"1. Predicted sMCI moved from 326 toward 317: **{report['required_answers']['smci_prediction_moved_toward_truth']}** ({report['prediction_counts']['SMCI']}).",
        f"2. CN->sMCI below PC-BBF v1's 12: **{report['required_answers']['cn_to_smci_below_12']}** ({directions['CN_to_SMCI']}).",
        f"3. sMCI->AD remains clearly below C1's 10: **{report['required_answers']['smci_to_ad_clearly_below_10']}** ({directions['SMCI_to_AD']}).",
        f"4. Repairs still exceed damages: **{report['required_answers']['repairs_exceed_damages']}** ({vs_c1['repairs']} vs {vs_c1['damages']}).",
        f"5. Reached 563/598: **{report['required_answers']['reached_563']}**.",
        "",
        "## Folds",
        "",
        *[
            f"- fold{fold['fold']}: best epoch {fold['best_epoch']}, correct {fold['correct']}, ACC {fold['acc']:.7f}"
            for fold in report["folds"]
        ],
        "",
        "No additional cap was tested. If the target was not reached, the next recommendation is Gradient-Isolated Pairwise Boundary Fusion (GI-PBF); it was not implemented or run here.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal(context: dict, output_root: Path) -> dict:
    formal_root = output_root / "formal"
    smoke_path = output_root / "smoke/smoke_report.json"
    require(smoke_path.is_file(), "Passing conservative smoke is required before formal training")
    smoke = pc_v1.read_json(smoke_path)
    require(smoke.get("passed") is True, "Conservative smoke did not pass")
    require(abs(float(smoke.get("pc_bbf_cap", -1.0)) - CAP) < 1e-12, "Smoke was not run with cap 0.075")
    require(smoke.get("safe_loss") is False, "Safe loss was not disabled in smoke")

    summaries, candidate_rows = [], []
    for fold in FORMAL_FOLDS:
        summary, rows = pc_v1.run_fold(context, fold, CAP, formal_root)
        require(summary["parameter_count"] == EXPECTED_PARAMETERS, f"fold{fold}: parameter count changed")
        require(abs(float(summary["cap_value"]) - CAP) < 1e-12, f"fold{fold}: cap changed")
        fold_metadata = summary.get("config", {})
        require(fold_metadata.get("experiment") == EXPERIMENT_NAME, f"fold{fold}: experiment identity changed")
        require(abs(float(fold_metadata.get("pc_bbf_cap", -1.0)) - CAP) < 1e-12, f"fold{fold}: recorded cap changed")
        require(fold_metadata.get("safe_loss") is False, f"fold{fold}: safe loss was not disabled")
        require(summary["mechanism"]["residual_shared_ratio_max"] <= CAP + 1e-6, f"fold{fold}: cap violated")
        summaries.append(summary)
        candidate_rows.extend(rows)

    candidate_rows.sort(key=lambda row: int(row["subject_index"]))
    validate_oof_rows(candidate_rows, "PC-BBF-Conservative")
    c1_rows = standardize_c1_rows(ROOT / "experiments/pc_bbf_c1_v1/branch_probe_oof.csv")
    v1_rows = load_pc_bbf_v1_rows(ROOT / "experiments/pc_bbf_c1_v1/formal/oof_predictions.csv")
    metrics = pc_v1.metrics_from_rows(candidate_rows)
    vs_c1 = comparison(candidate_rows, c1_rows)
    vs_v1 = comparison(candidate_rows, v1_rows)
    boundary = pc_v1.boundary_errors(candidate_rows)
    directions = directional_errors(candidate_rows)
    mechanism = pc_v1.summarize_mechanism(summaries)
    require(mechanism["residual_shared_ratio_max"] <= CAP + 1e-6, "Aggregate cap violation")
    decision, decision_checks = conservative_decision(metrics, vs_c1, boundary)
    gradients = {
        name: float(max(summary["gradient_max"][name] for summary in summaries))
        for name in ("input_projection", "residual_output", "gate_output")
    }
    require(all(math.isfinite(value) and value > 0.0 for value in gradients.values()), "New module was inactive")
    fold_acc = [float(summary["best_metrics"]["acc"]) for summary in summaries]
    predicted = prediction_counts(candidate_rows)
    required_answers = {
        "smci_prediction_moved_toward_truth": abs(predicted["SMCI"] - 317) < abs(326 - 317),
        "cn_to_smci_below_12": directions["CN_to_SMCI"] < 12,
        "smci_to_ad_clearly_below_10": directions["SMCI_to_AD"] <= 7,
        "repairs_exceed_damages": vs_c1["repairs"] > vs_c1["damages"],
        "reached_563": metrics["correct"] >= 563,
    }
    report = {
        "stage": "formal",
        "experiment": EXPERIMENT_NAME,
        "decision": decision,
        "decision_checks": decision_checks,
        "pc_bbf_cap": CAP,
        "safe_loss": False,
        "pc_bbf_v1_base_commit_sha": PC_BBF_V1_BASE_COMMIT,
        "run_head": pc_v1.git_value("rev-parse", "HEAD"),
        "device": str(context["device"]),
        "subject_count": 598,
        "metrics": metrics,
        "prediction_counts": predicted,
        "truth_counts": {
            name: sum(int(row["truth"]) == index for row in candidate_rows)
            for index, name in enumerate(CLASS_NAMES)
        },
        "comparison_vs_c1": vs_c1,
        "comparison_vs_pc_bbf_v1": vs_v1,
        "metric_deltas": {
            "vs_c1": metric_delta(metrics, C1_REFERENCE),
            "vs_pc_bbf_v1": metric_delta(metrics, PC_BBF_V1_REFERENCE),
        },
        "reference_metrics": {"c1": C1_REFERENCE, "pc_bbf_v1": PC_BBF_V1_REFERENCE},
        "boundary_errors": {
            "candidate": boundary,
            "c1": pc_v1.boundary_errors(c1_rows),
            "pc_bbf_v1": pc_v1.boundary_errors(v1_rows),
        },
        "directional_errors": {
            "candidate": directions,
            "c1": directional_errors(c1_rows),
            "pc_bbf_v1": directional_errors(v1_rows),
        },
        "mechanism": mechanism,
        "pc_bbf_gradient_max": gradients,
        "new_module_max_gradient": float(max(gradients.values())),
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameters_vs_c1": EXPECTED_PARAMETERS - 862_971,
        "fold_acc_mean": float(np.mean(fold_acc)),
        "fold_acc_sample_std": float(np.std(fold_acc, ddof=1)),
        "folds": [
            {
                "fold": int(summary["fold"]),
                "best_epoch": int(summary["best_epoch"]),
                "correct": int(summary["best_metrics"]["correct"]),
                "acc": float(summary["best_metrics"]["acc"]),
            }
            for summary in summaries
        ],
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "required_answers": required_answers,
        "run_command": "python -u -B scripts/run_pc_bbf_conservative_v1_1.py formal --device cuda:0",
        "next_recommendation_if_target_not_reached": (
            None if decision == "PC_BBF_CONSERVATIVE_TARGET" else "Gradient-Isolated Pairwise Boundary Fusion (GI-PBF)"
        ),
        "next_experiment_implemented_or_run": False,
        "config": conservative_config(context),
    }

    formal_root.mkdir(parents=True, exist_ok=True)
    write_oof_views(formal_root, candidate_rows)
    pair_table = paired_rows(candidate_rows, c1_rows, v1_rows)
    pc_v1.write_csv(formal_root / "paired_repairs_damages.csv", pair_table)
    pc_v1.write_json(
        formal_root / "repairs_damages.json",
        {"comparison_vs_c1": vs_c1, "comparison_vs_pc_bbf_v1": vs_v1},
    )
    pc_v1.write_json(formal_root / "config.json", report["config"])
    pc_v1.write_json(output_root / "config.json", report["config"])
    pc_v1.write_json(formal_root / "report.json", report)
    write_markdown(formal_root / "report.md", report)
    pc_v1.write_csv(
        formal_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(metrics["confusion_matrix"])
        ],
    )
    pc_v1.write_csv(
        formal_root / "fold_metrics.csv",
        [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                **{key: value for key, value in summary["best_metrics"].items() if key != "confusion_matrix"},
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for summary in summaries
        ],
    )
    print(f"{decision} correct={metrics['correct']}/598 ACC={metrics['acc']:.7f}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(abs(CAP - 0.075) < 1e-12, "The sole allowed cap must be 0.075")
    require(SAFE_LOSS is False, "Safe loss must remain disabled")
    require(pc_v1.git_value("merge-base", "--is-ancestor", PC_BBF_V1_BASE_COMMIT, "HEAD") == "", "Branch does not descend from the formal PC-BBF v1 HEAD")
    require(torch.cuda.is_available(), "CUDA unavailable; CPU execution is forbidden")
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index in (None, 0)), "Historical protocol requires cuda:0")
    print(f"PC-BBF-Conservative effective cap={CAP:.3f} safe_loss={SAFE_LOSS}", flush=True)
    context = pc_v1.load_context(device)
    output_root = args.output_root.resolve()
    if args.stage == "smoke":
        run_smoke(context, output_root)
    else:
        run_formal(context, output_root)


if __name__ == "__main__":
    main()
