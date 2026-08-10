"""Registered ABIDE ADS/CN task-level hyperparameter search."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2
from Model import HeterGraph_Model_Kmeans
from Model.cme_dual_branch import CMEDualBranchModel
from Utils import CustomCosineAnnealingLR, SET_Random
from scripts import run_tad_binary_hparam_search_v1 as engine


EXPERIMENT_ID = "abide_hparam_search_v1"
BRANCH = "experiment/abide-hparam-search-v1"
BASE_COMMIT = "0d7a183a7106c673de2817f36c69d780dcd792d5"
HISTORICAL_SOURCE_COMMIT = "17004d29714471b5e1548e04bac0072e854e7e18"
HISTORICAL_RESULT_COMMIT = "e7d18205ddfefaf898a81c999cce15b348331206"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_PATH = RESULT_DIR / "protocol.json"
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
HISTORICAL_DIR = RESULT_DIR / "historical"
SMOKE_DIR = RESULT_DIR / "smoke"
WORK_DIR = RESULT_DIR / "work"
TRIALS_DIR = RESULT_DIR / "trials"
TRIAL_MANIFEST_PATH = RESULT_DIR / "trial_manifest.json"
ALL_TRIALS_PATH = RESULT_DIR / "all_trials.json"
SEARCH_SUMMARY_PATH = RESULT_DIR / "search_summary.json"

HISTORICAL_PATHS = {
    "B0": "experiments/cross_dataset_a012_structure_v1/abide_ads_cn_B0_oof_predictions.csv",
    "B1": "experiments/cross_dataset_a012_structure_v1/abide_ads_cn_B1_oof_predictions.csv",
    "fold_metrics": "experiments/cross_dataset_a012_structure_v1/abide_ads_cn_fold_metrics.json",
    "summary": "experiments/cross_dataset_a012_structure_v1/summary.json",
}
HISTORICAL_ANCHORS = {
    "B0": HISTORICAL_DIR / "historical_B0_oof.csv",
    "B1": HISTORICAL_DIR / "historical_B1_oof.csv",
    "fold_metrics": HISTORICAL_DIR / "historical_fold_metrics.json",
}
REQUIRED_TRACKED = (
    ".gitignore",
    "scripts/run_abide_hparam_search_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocol.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
    f"experiments/{EXPERIMENT_ID}/historical/historical_B0_oof.csv",
    f"experiments/{EXPERIMENT_ID}/historical/historical_B1_oof.csv",
    f"experiments/{EXPERIMENT_ID}/historical/historical_fold_metrics.json",
)
LOCKED_DEPENDENCIES = (
    "Model/cme_dual_branch.py",
    "Model/network.py",
    "Model/models.py",
    "Loss/loss_fn.py",
    "Utils/data_load.py",
    "Utils/graph_load.py",
    "Utils/utils.py",
    "config_re.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
    "scripts/run_tad_binary_hparam_search_v1.py",
    "experiments/cross_dataset_a012_structure_v1/protocols/abide_ads_cn.json",
)


def require(condition: bool, message: str) -> None:
    engine.require(condition, message)


def protocol() -> dict[str, Any]:
    value = engine.read_json(PROTOCOL_PATH)
    require(value["dataset"] == "ABIDE" and value["task"] == "ADS_CN", "Task protocol changed")
    require(value["class_names"] == ["ADS", "CN"], "Class order changed")
    require(value["positive_class"] == "ADS" and value["positive_index"] == 0, "ASD positive direction changed")
    require(value["sample_count"] == 871 and value["class_counts"] == {"ADS": 403, "CN": 468}, "Dataset counts changed")
    require(len(value["modalities"]) == 4, "ABIDE must contain four real modalities")
    return value


def validate_oof(rows: list[dict[str, Any]], context: dict[str, Any], arm: str | None = None) -> None:
    ordered = sorted(rows, key=lambda row: int(row["original_csv_index"]))
    anchors = engine.expected_subjects(context)
    require(len(ordered) == len(anchors) == 871, "OOF count changed")
    require(len({row["subject_id"] for row in ordered}) == 871, "OOF subject IDs are not unique")
    for row, anchor in zip(ordered, anchors):
        require(row["subject_id"] == anchor["subject_id"], "OOF subject alignment changed")
        require(row["feature_sha256"] == anchor["feature_sha256"], "OOF feature anchor changed")
        require(row["truth"] == anchor["truth"] and row["fold"] == anchor["fold"], "OOF truth/fold changed")
        require(row["prediction"] == int(np.argmax([row["probability_0"], row["probability_1"]])), "OOF prediction is not argmax")
        if arm is not None:
            require(row["arm"] == arm, "OOF arm changed")
    probability = np.asarray([[row["probability_0"], row["probability_1"]] for row in ordered], dtype=float)
    require(np.isfinite(probability).all() and float(np.max(np.abs(probability.sum(1) - 1.0))) <= 2e-6, "OOF probabilities invalid")


def historical_payload(write_anchors: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    context = engine.build_context(torch.device("cpu"))
    blobs = {key: engine.git_blob(HISTORICAL_RESULT_COMMIT, relative) for key, relative in HISTORICAL_PATHS.items()}
    if write_anchors:
        for key, destination in HISTORICAL_ANCHORS.items():
            engine.atomic_write_bytes(destination, blobs[key])
    else:
        for key, destination in HISTORICAL_ANCHORS.items():
            require(destination.is_file() and destination.read_bytes() == blobs[key], f"Historical anchor drifted: {key}")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ("B0", "B1"):
        rows = engine.typed_rows(list(csv.DictReader(io.StringIO(blobs[arm].decode("utf-8")))))
        validate_oof(rows, context, arm)
        rows_by_arm[arm] = rows
    fold_metrics = json.loads(blobs["fold_metrics"].decode("utf-8"))
    summary = json.loads(blobs["summary"].decode("utf-8"))
    require(summary["runtime_lock"]["source_commit"] == HISTORICAL_SOURCE_COMMIT, "Historical source commit changed")
    require(summary["runtime_lock"]["seed"] == 0 and summary["runtime_lock"]["epochs"] == 400, "Historical seed/epochs changed")
    require(summary["runtime_lock"]["adapter_rank"] == 8 and summary["runtime_lock"]["adapter_lr_multiplier"] == 2.0, "Historical B1 config changed")
    expected = {
        "B0": {"correct": 761, "acc": 0.8737083811710677, "roc_auc": 0.8829240100952259, "pr_auc": 0.840199801144099, "macro_f1": 0.8732390266620096, "bacc": 0.8740350151640475, "sen": 0.8784119106699751, "spe": 0.8696581196581197},
        "B1": {"correct": 758, "acc": 0.870264064293915, "roc_auc": 0.8750556722020743, "pr_auc": 0.8289356007399968, "macro_f1": 0.8699898684483403, "bacc": 0.8716914805624483, "sen": 0.890818858560794, "spe": 0.8525641025641025},
    }
    computed: dict[str, dict[str, Any]] = {}
    for arm in ("B0", "B1"):
        computed[arm] = engine.metrics(rows_by_arm[arm])
        for key, target in expected[arm].items():
            require(math.isclose(float(computed[arm][key]), target, rel_tol=0.0, abs_tol=1e-9), f"Historical {arm} metric changed: {key}")
        require(len(fold_metrics[arm]) == 10 and [int(row["fold"]) for row in fold_metrics[arm]] == list(engine.FOLDS), f"Historical {arm} folds changed")

    weights = context["dataset_dict"]["Label_Weight"].detach().cpu().numpy().astype(float)
    expected_weights = np.asarray([(871 - 403) / 871, (871 - 468) / 871], dtype=float)
    require(float(np.max(np.abs(weights - expected_weights))) <= 1e-7, "Global full-dataset class weights changed")
    fold_core = {"task_id": context["task_id"], "fold_manifest": context["fold_manifest"]}
    task_result = summary["task_results"]["abide_ads_cn"]
    inspect_core = {
        "experiment": EXPERIMENT_ID,
        "base_commit": BASE_COMMIT,
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_result_commit": HISTORICAL_RESULT_COMMIT,
        "dataset": "ABIDE", "task": "ADS_CN", "sample_count": 871,
        "class_names": ["ADS", "CN"], "positive_class": "ADS", "positive_index": 0,
        "positive_semantics": "ADS is the repository alias for ASD",
        "modalities": 4, "folds": list(engine.FOLDS), "seed": 0, "epochs": 400,
        "class_weight_scope": "global_full_dataset_historical",
        "class_weights": weights.tolist(),
        "criterion": "criterion_lossv2: main weighted CE + two weighted OVR CE terms summed at coefficient 1.0; label smoothing 0.05; orthogonality rate 0",
        "best_epoch_rule": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "test_fold_best_epoch_behavior": True,
        "historical_reuse": {
            "eligible": True,
            "reason": "The registered seed-0 cross-dataset run has this exact ABIDE task, protocol, folds, subject anchors, per-subject probabilities, and per-fold best epochs.",
            "anchor_sha256": {key: engine.bytes_sha256(blobs[key]) for key in HISTORICAL_ANCHORS},
            "metrics": computed, "fold_metrics": fold_metrics,
            "parameter_count": {"B0": int(task_result["B0"]["parameter_count"]), "B1": int(task_result["B1"]["parameter_count"])},
            "mechanism_B1": summary["mechanism_diagnostics"]["abide_ads_cn"],
        },
        "protocol_sha256": engine.file_sha256(PROTOCOL_PATH),
        "dataset_sha256": protocol()["data_sha256"], "modality_sha256": protocol()["modality_sha256"],
        "historical_runner_sha256": engine.file_sha256(ROOT / "scripts" / "run_cross_dataset_a012_structure_v1.py"),
        "historical_loss_sha256": engine.file_sha256(ROOT / "Loss" / "loss_fn.py"),
    }
    del context
    return ({**inspect_core, "sha256": engine.payload_sha256(inspect_core)}, {**fold_core, "sha256": engine.payload_sha256(fold_core)})


def build_model(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    kwargs = engine.historical.model_kwargs(context)
    initialization: dict[str, Any] = {}
    SET_Random(engine.SEED)
    if spec["arm"] == "B0":
        model = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
    else:
        reference = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = engine.capture_rng()
        SET_Random(engine.SEED)
        model = CMEDualBranchModel(
            **kwargs, cme_arm="c1", adapter_rank=int(spec["rank"]),
            router_hidden=16, modality_embedding_dim=8,
        ).to(context["device"])
        difference = engine.common_state_max_diff(reference, model)
        if audit_common:
            require(difference == 0.0, "B0/B1 common initialization changed")
            initialization["common_state_max_abs_diff"] = difference
        else:
            initialization["common_state_pointwise_audit"] = "smoke_only"
        del reference
        engine.restore_rng(post_common_rng)

    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary Query/OVR count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    if spec["arm"] == "B1":
        require(len(model.private_adapters) == 4, "ABIDE requires one private adapter for each of four real modalities")
        require(not hasattr(model, "private_alpha"), "Trainable private alpha is forbidden")
    initialization.update({
        "parameter_count": engine.parameter_count(model),
        "query_count": len(model.label_pools), "ovr_head_count": len(model._Auxi_classifier),
        "adapter_count": len(model.private_adapters) if spec["arm"] == "B1" else 0,
    })
    return model, initialization


def make_training_objects(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    model, initialization = build_model(context, spec, audit_common=audit_common)
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(
        context["dataset_dict"], context["device"], rate=float(training["loss_rate"]),
        label_smoothing=float(training["label_smoothing"]),
    )
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if spec["arm"] == "B0":
        optimizer = torch.optim.Adam(
            [parameter for _, parameter in named], lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
        )
        scheduler = CustomCosineAnnealingLR(optimizer, T_max=engine.EPOCHS, eta_min=float(training["scheduler_eta_min"]))
        group_audit = {
            "base_parameter_tensors": len(named), "adapter_parameter_tensors": 0,
            "base_lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"]),
        }
    else:
        adapter_named = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
        base_named = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
        base_ids = {id(parameter) for _, parameter in base_named}
        adapter_ids = {id(parameter) for _, parameter in adapter_named}
        all_ids = {id(parameter) for _, parameter in named}
        require(len(adapter_named) == 16, "Four private adapters must expose exactly 16 parameter tensors")
        require(not (base_ids & adapter_ids) and base_ids | adapter_ids == all_ids, "Optimizer groups are not exhaustive/disjoint")
        multiplier = float(spec["adapter_lr_multiplier"])
        optimizer = torch.optim.Adam([
            {"params": [parameter for _, parameter in base_named], "lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"]), "group_name": "base"},
            {"params": [parameter for _, parameter in adapter_named], "lr": float(spec["lr"]) * multiplier, "weight_decay": float(spec["weight_decay"]), "group_name": "private_adapters"},
        ])
        scheduler = engine.historical.RatioPreservingCustomCosineAnnealingLR(
            optimizer, T_max=engine.EPOCHS, eta_min=float(training["scheduler_eta_min"]), multiplier=multiplier,
        )
        group_audit = {
            "base_parameter_tensors": len(base_named), "adapter_parameter_tensors": len(adapter_named),
            "adapter_parameter_names": [name for name, _ in adapter_named],
            "base_lr": float(spec["lr"]), "adapter_lr": float(spec["lr"]) * multiplier,
            "weight_decay": float(spec["weight_decay"]),
        }
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler, {**initialization, **group_audit}


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = engine.source_gate()
    runtime = engine.runtime_lock(source)
    base = engine.build_context(torch.device(device_text))
    spec = engine.trial_spec("SMOKE_D1", "smoke", "B1", 0.005, 0.001, 0.45, rank=4, multiplier=0.5)
    context = engine.trial_context(base, spec)
    smoke_config = engine.smoke_runtime_config(runtime, spec)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    engine.atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)

    b0_spec = engine.trial_spec("SMOKE_B0", "smoke", "B0", 0.005, 0.001, 0.45)
    b0_context = engine.trial_context(base, b0_spec)
    b0_model, _ = build_model(b0_context, b0_spec)
    model, criterion, optimizer, scheduler, audit = make_training_objects(context, spec, audit_common=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    labels_before = labels.detach().clone()
    train_mask, test_mask, _ = engine.historical.fold_positions(context, 0)
    require(not bool(torch.any(train_mask & test_mask)), "Smoke fold masks overlap")
    b0_model.eval(); model.eval()
    with torch.no_grad():
        b0_initial = b0_model(features)[0]
        b1_initial = model(features)[0]
    initial_logits_diff = float((b0_initial - b1_initial).abs().max().cpu())
    require(initial_logits_diff == 0.0, "Zero-output adapters changed initial logits")
    del b0_model

    initial_adapter = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name.startswith("private_adapters.")}
    gradient_max = {name: 0.0 for name in initial_adapter}
    losses: list[float] = []
    formula_errors: list[float] = []
    probability_errors: list[float] = []
    for _epoch in range(1, engine.SMOKE_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output, embeddings, auxiliary = model(features)
        require(tuple(output.shape) == (871, 2) and len(auxiliary) == 2, "Smoke output shape changed")
        loss = criterion(output, labels, train_mask, embeddings, auxiliary)
        manual, components = engine.manual_historical_loss(criterion, output, labels, train_mask, embeddings, auxiliary)
        formula_errors.append(float((loss - manual).abs().detach().cpu()))
        require(bool(torch.isfinite(loss)) and formula_errors[-1] <= 1e-7, "Historical loss formula changed")
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("private_adapters."):
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid adapter gradient: {name}")
                gradient_max[name] = max(gradient_max[name], float(parameter.grad.detach().abs().max().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
        optimizer.step(); scheduler.step(); scheduler.assert_ratio()
        losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(features)[0], dim=-1)
        probability_errors.append(float((probability.sum(dim=1) - 1.0).abs().max().cpu()))
    require(torch.equal(labels, labels_before), "Smoke labels changed")
    require(all(math.isfinite(value) for value in losses), "Smoke loss is non-finite")
    require(all(value > 0.0 and math.isfinite(value) for value in gradient_max.values()), "Every adapter tensor must receive a finite nonzero gradient")
    parameter_delta = {name: float((parameter.detach().cpu() - initial_adapter[name]).abs().max()) for name, parameter in model.named_parameters() if name in initial_adapter}
    require(all(value > 0.0 and math.isfinite(value) for value in parameter_delta.values()), "Every adapter tensor must update")
    require(max(probability_errors) <= 2e-6, "Smoke probability simplex changed")

    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    checkpoint = {
        "runtime_lock": runtime, "smoke_config": smoke_config, "spec": spec,
        "epoch": engine.SMOKE_EPOCHS, "model": engine.clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": engine.capture_rng(),
    }
    engine.atomic_torch_save(checkpoint_path, checkpoint)
    restored_model, restored_criterion, restored_optimizer, restored_scheduler, _ = make_training_objects(context, spec)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_model.load_state_dict(restored["model"], strict=True)
    restored_optimizer.load_state_dict(restored["optimizer"]); restored_scheduler.load_state_dict(restored["scheduler"])
    restored_scheduler.assert_ratio()
    require(engine.checkpoint_state_finite(restored), "Smoke checkpoint contains non-finite tensors")
    model.eval(); restored_model.eval()
    with torch.no_grad():
        left = model(features)[0]; right = restored_model(features)[0]
    reload_diff = float((left - right).abs().max().cpu())
    require(reload_diff == 0.0, "Smoke checkpoint strict reload changed logits")
    test_probability = torch.softmax(left[test_mask], dim=-1).detach().cpu().numpy().astype(np.float64)
    test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
    test_metrics = engine.historical.binary_metrics(test_truth, test_probability, protocol())
    test_prediction = test_probability.argmax(1)
    require(test_metrics["tp"] == int(np.sum((test_truth == 0) & (test_prediction == 0))), "ASD TP direction changed")
    require(test_metrics["tn"] == int(np.sum((test_truth == 1) & (test_prediction == 1))), "CN TN direction changed")
    three_class = engine.historical.three_class_compatibility_regression(device_text)

    report_core = {
        "runtime_lock": runtime, "smoke_config_sha256": engine.file_sha256(SMOKE_DIR / "smoke_config.json"),
        "spec": spec, "fold": 0, "epochs": engine.SMOKE_EPOCHS, "losses": losses,
        "historical_loss_components_last_epoch": components,
        "historical_loss_formula_max_abs_error": max(formula_errors),
        "probability_sum_max_abs_error": max(probability_errors),
        "initial_b0_b1_logits_max_abs_diff": initial_logits_diff,
        "adapter_gradient_max_by_tensor": gradient_max, "adapter_parameter_delta_by_tensor": parameter_delta,
        "object_audit": audit, "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        "checkpoint_reload_logits_max_abs_diff": reload_diff,
        "positive_direction_audit": {"positive_class": "ADS/ASD", "positive_index": 0, "negative_class": "CN", "test_metrics": test_metrics},
        "test_labels_unchanged": True, "three_class_compatibility": three_class,
        "device": torch.cuda.get_device_name(torch.device(device_text)), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    engine.atomic_write_json(SMOKE_DIR / "smoke_report.json", {**report_core, "sha256": engine.payload_sha256(report_core)})
    print(json.dumps({"smoke": "PASS", "losses": losses, "params": audit["parameter_count"]}, indent=2))
    del restored_criterion, restored_model, model, base, context
    torch.cuda.empty_cache()


def safety(result: dict[str, Any]) -> dict[str, Any]:
    value = result["metrics"]
    collapse = min(int(value["predicted_counts"]["ADS"]), int(value["predicted_counts"]["CN"])) == 0
    reasons: list[str] = []
    if result["spec"]["arm"] == "B0" and float(value["bacc"]) < 0.869:
        reasons.append("BACC < 0.869")
    if float(value["sen"]) < 0.84:
        reasons.append("ASD sensitivity < 0.84")
    if float(value["spe"]) < 0.84:
        reasons.append("CN specificity < 0.84")
    if abs(float(value["sen"]) - float(value["spe"])) > 0.08:
        reasons.append("abs(SEN-SPE) > 0.08")
    if collapse:
        reasons.append("single-class prediction collapse")
    return {"safe": not reasons, "collapse": collapse, "reasons": reasons}


def is_historical_exact(spec: dict[str, Any]) -> bool:
    exact = math.isclose(spec["lr"], 0.005) and math.isclose(spec["weight_decay"], 0.001) and math.isclose(spec["dropout"], 0.45)
    return exact if spec["arm"] == "B0" else exact and spec["rank"] == 8 and math.isclose(spec["adapter_lr_multiplier"], 2.0)


def structural_safety(tuned_b0: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    pair = engine.paired(tuned_b0, candidate)
    value = candidate["metrics"]
    reasons = list(candidate["safety"]["reasons"])
    if float(value["bacc"]) < float(tuned_b0["metrics"]["bacc"]) - 0.003:
        reasons.append("BACC is more than 0.003 below TUNED_B0")
    if pair["repairs"] <= pair["damages"]:
        reasons.append("repairs <= damages")
    expansion_only = bool(
        float(value["sen"]) > float(tuned_b0["metrics"]["sen"])
        and float(value["spe"]) < float(tuned_b0["metrics"]["spe"])
        and int(value["predicted_counts"]["ADS"]) > int(tuned_b0["metrics"]["predicted_counts"]["ADS"])
    )
    if expansion_only:
        reasons.append("SEN gain is explained only by expansion of the ADS prediction region at the expense of SPE")
    return {
        "safe": not reasons, "reasons": reasons, "collapse": candidate["safety"]["collapse"],
        "repairs": pair["repairs"], "damages": pair["damages"], "changed": pair["changed"],
        "bacc_delta_vs_tuned_b0": pair["metric_deltas"]["bacc"], "prediction_region_expansion_only": expansion_only,
    }


def select_structural_stage(stage: str, candidates: list[dict[str, Any]], tuned_b0: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    audits = {candidate["trial_id"]: structural_safety(tuned_b0, candidate) for candidate in candidates}
    safe = [candidate for candidate in candidates if audits[candidate["trial_id"]]["safe"]]
    overall = max(candidates, key=engine.rank_key)
    selected = max(safe, key=engine.rank_key) if safe else None
    selection = {
        "stage": stage, "candidate_trial_ids": [item["trial_id"] for item in candidates],
        "safe_trial_ids": [item["trial_id"] for item in safe],
        "best_overall_trial_id": overall["trial_id"],
        "selected_trial_id": selected["trial_id"] if selected else None,
        "dynamic_safety": audits,
        "selection_rule": "B1 structural safety > Correct > pooled ROC-AUC > fold mean ROC-AUC > Macro-F1 > BACC > smaller params > earlier trial",
    }
    registry["stages"][stage] = selection
    engine.registry_save(registry)
    return selection


def run_search(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Search requires cuda:0")
    source = engine.source_gate()
    runtime = engine.runtime_lock(source)
    engine.validate_smoke(runtime)
    base_context = engine.build_context(torch.device(device_text))
    registry = engine.registry_load(runtime)
    results: dict[str, dict[str, Any]] = {}

    stage_a_specs = [engine.trial_spec(f"A{index}", "A", "B0", lr, 0.001, 0.45, ordinal=index) for index, lr in enumerate((0.00375, 0.005, 0.00625), 1)]
    for spec in stage_a_specs:
        results[spec["trial_id"]] = engine.ensure_trial(spec, base_context, runtime, registry)
    stage_a = engine.select_stage("A", [results[spec["trial_id"]] for spec in stage_a_specs], registry)
    require(stage_a["selected_trial_id"] is not None, "Stage A has no class-safe candidate")
    best_a = results[stage_a["selected_trial_id"]]

    stage_b_specs = [engine.trial_spec(f"B{index}", "B", "B0", best_a["spec"]["lr"], wd, 0.45, ordinal=3 + index) for index, wd in enumerate((0.0005, 0.001, 0.002), 1)]
    for spec in stage_b_specs:
        results[spec["trial_id"]] = engine.ensure_trial(spec, base_context, runtime, registry)
    stage_b = engine.select_stage("B", [results[spec["trial_id"]] for spec in stage_b_specs], registry)
    require(stage_b["selected_trial_id"] is not None, "Stage B has no class-safe candidate")
    best_b = results[stage_b["selected_trial_id"]]

    stage_c_specs = [engine.trial_spec(f"C{index}", "C", "B0", best_b["spec"]["lr"], best_b["spec"]["weight_decay"], dropout, ordinal=6 + index) for index, dropout in enumerate((0.35, 0.45, 0.55), 1)]
    for spec in stage_c_specs:
        results[spec["trial_id"]] = engine.ensure_trial(spec, base_context, runtime, registry)
    stage_c = engine.select_stage("C", [results[spec["trial_id"]] for spec in stage_c_specs], registry)
    require(stage_c["selected_trial_id"] is not None, "Stage C has no class-safe candidate")
    tuned_b0 = results[stage_c["selected_trial_id"]]

    stage_d_specs: list[dict[str, Any]] = []
    ordinal = 10
    for rank in (4, 8):
        for multiplier in (0.5, 1.0, 2.0):
            stage_d_specs.append(engine.trial_spec(
                f"D{ordinal - 9}", "D", "B1", tuned_b0["spec"]["lr"], tuned_b0["spec"]["weight_decay"],
                tuned_b0["spec"]["dropout"], rank=rank, multiplier=multiplier, ordinal=ordinal,
            ))
            ordinal += 1
    for spec in stage_d_specs:
        results[spec["trial_id"]] = engine.ensure_trial(spec, base_context, runtime, registry)
    stage_d = select_structural_stage("D", [results[spec["trial_id"]] for spec in stage_d_specs], tuned_b0, registry)
    tuned_b1 = results[stage_d["selected_trial_id"]] if stage_d["selected_trial_id"] else results[stage_d["best_overall_trial_id"]]

    pair = engine.paired(tuned_b0, tuned_b1)
    value = tuned_b1["metrics"]
    local_triggered = bool(
        788 <= int(value["correct"]) <= 791 and pair["repairs"] > pair["damages"]
        and float(value["bacc"]) >= 0.90 and float(value["sen"]) >= 0.88
        and float(value["spe"]) >= 0.88 and not tuned_b1["safety"]["collapse"]
    )
    local_reason = "Stage D best entered the registered 788-791 safe local-LR window" if local_triggered else "Stage D best is outside the registered 788-791 safe local-LR window"
    if int(value["correct"]) >= 792 and tuned_b1["fold_acc_mean"] > 0.9082 and tuned_b1["fold_roc_auc_mean"] > 0.9084:
        local_triggered = False; local_reason = "Stage D already reached Correct and fold-mean ACC/AUC targets"
    elif int(value["correct"]) <= 787:
        local_triggered = False; local_reason = "Stage D Correct <= 787"
    if local_triggered:
        local_specs = [
            engine.trial_spec("L1", "LOCAL", "B1", tuned_b1["spec"]["lr"] * 0.9, tuned_b1["spec"]["weight_decay"], tuned_b1["spec"]["dropout"], rank=tuned_b1["spec"]["rank"], multiplier=tuned_b1["spec"]["adapter_lr_multiplier"], ordinal=16),
            engine.trial_spec("L2", "LOCAL", "B1", tuned_b1["spec"]["lr"] * 1.1, tuned_b1["spec"]["weight_decay"], tuned_b1["spec"]["dropout"], rank=tuned_b1["spec"]["rank"], multiplier=tuned_b1["spec"]["adapter_lr_multiplier"], ordinal=17),
        ]
        for spec in local_specs:
            results[spec["trial_id"]] = engine.ensure_trial(spec, base_context, runtime, registry)
        local = select_structural_stage("LOCAL", [tuned_b1, *[results[spec["trial_id"]] for spec in local_specs]], tuned_b0, registry)
        if local["selected_trial_id"] is not None:
            tuned_b1 = results[local["selected_trial_id"]]

    registry["local_refinement"] = {"triggered": local_triggered, "reason": local_reason}
    registry["tuned_b0_trial_id"] = tuned_b0["trial_id"]
    registry["tuned_b1_trial_id"] = tuned_b1["trial_id"]
    engine.registry_save(registry)
    ordered_results = [engine.load_trial_result(entry["spec"], base_context) for entry in sorted(registry["trials"].values(), key=lambda entry: int(entry["spec"]["ordinal"]))]
    all_core = {"runtime_lock": runtime, "trial_count": len(ordered_results), "trials": ordered_results}
    engine.atomic_write_json(ALL_TRIALS_PATH, {**all_core, "sha256": engine.payload_sha256(all_core)})
    search_core = {
        "runtime_lock": runtime, "trial_manifest_sha256": engine.file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": engine.file_sha256(ALL_TRIALS_PATH), "stage_selections": registry["stages"],
        "local_refinement": registry["local_refinement"], "tuned_b0_trial_id": tuned_b0["trial_id"],
        "tuned_b1_trial_id": tuned_b1["trial_id"], "failure_count": 0,
    }
    engine.atomic_write_json(SEARCH_SUMMARY_PATH, {**search_core, "sha256": engine.payload_sha256(search_core)})
    print(json.dumps({"search": "COMPLETE", "tuned_b0": tuned_b0["trial_id"], "tuned_b1": tuned_b1["trial_id"], "local_triggered": local_triggered}, indent=2))
    del base_context
    torch.cuda.empty_cache()


def historical_reference(arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(arm in ("B0", "B1"), "Unknown historical arm")
    spec = engine.trial_spec(
        f"HISTORICAL_{arm}", "HISTORICAL", arm, 0.005, 0.001, 0.45,
        rank=8 if arm == "B1" else 0, multiplier=2.0 if arm == "B1" else 0.0,
        ordinal=-2 if arm == "B0" else -1,
    )
    rows = [{**row, "trial_id": spec["trial_id"]} for row in engine.typed_rows(engine.read_csv(HISTORICAL_ANCHORS[arm]))]
    fold_payload = engine.read_json(HISTORICAL_ANCHORS["fold_metrics"])[arm]
    summaries = [{
        "fold": int(row["fold"]), "best_epoch": int(row["best_epoch"]),
        "best_metrics": {key: (int(row[key]) if key == "correct" else float(row[key])) for key in ("correct", "acc", "roc_auc", "pr_auc", "macro_f1", "bacc")},
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "inference_seconds_full_graph": float(row["inference_seconds_full_graph"]),
        "resumed_from_epoch": int(row.get("resumed_from_epoch", 0)),
    } for row in fold_payload]
    inspect = engine.read_json(INSPECT_PATH)
    result = engine.make_trial_result(
        spec, rows, summaries, int(inspect["historical_reuse"]["parameter_count"][arm]),
        inspect["historical_reuse"]["mechanism_B1"] if arm == "B1" else None,
        f"historical:{HISTORICAL_RESULT_COMMIT}:{arm}",
    )
    return result, rows


def target_gate(result: dict[str, Any]) -> dict[str, Any]:
    value = result["metrics"]
    checks = {
        "correct_at_least_792": int(value["correct"]) >= 792,
        "fold_acc_mean_above_0_9082": float(result["fold_acc_mean"]) > 0.9082,
        "fold_roc_auc_mean_above_0_9084": float(result["fold_roc_auc_mean"]) > 0.9084,
        "bacc_at_least_0_90": float(value["bacc"]) >= 0.90,
        "macro_f1_at_least_0_90": float(value["macro_f1"]) >= 0.90,
        "asd_sen_at_least_0_89": float(value["sen"]) >= 0.89,
        "cn_spe_at_least_0_89": float(value["spe"]) >= 0.89,
        "no_class_collapse": not result["safety"]["collapse"],
    }
    return {"reached": all(checks.values()), "checks": checks}


def final_decision(tuned_b0: dict[str, Any], tuned_b1: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    b0_gate = target_gate(tuned_b0)
    b1_gate = target_gate(tuned_b1)
    pair = comparisons["tuned_b1_vs_tuned_b0"]
    delta = pair["metric_deltas"]
    value = tuned_b1["metrics"]
    structural = structural_safety(tuned_b0, tuned_b1)
    b1_outperforms = engine.rank_key(tuned_b1) > engine.rank_key(tuned_b0)
    positive = bool(
        int(value["correct"]) > int(tuned_b0["metrics"]["correct"])
        and pair["repairs"] > pair["damages"] and delta["bacc"] >= 0.0 and delta["roc_auc"] >= 0.0
        and float(value["sen"]) >= 0.84 and float(value["spe"]) >= 0.84
        and abs(float(value["sen"]) - float(value["spe"])) <= 0.08
        and not structural["prediction_region_expansion_only"] and not tuned_b1["safety"]["collapse"]
    )
    private_no_gain_reasons: list[str] = []
    if int(value["correct"]) <= int(tuned_b0["metrics"]["correct"]): private_no_gain_reasons.append("TUNED_B1 Correct <= TUNED_B0")
    if pair["repairs"] <= pair["damages"]: private_no_gain_reasons.append("repairs <= damages")
    if delta["bacc"] < -0.005: private_no_gain_reasons.append("BACC declined materially")
    if delta["roc_auc"] < -0.005: private_no_gain_reasons.append("ROC-AUC declined materially")
    if structural["prediction_region_expansion_only"]: private_no_gain_reasons.append("SEN gain mainly sacrifices SPE")
    if tuned_b1["safety"]["collapse"]: private_no_gain_reasons.append("class prediction collapse")
    local_triggered = bool(engine.read_json(TRIAL_MANIFEST_PATH)["local_refinement"]["triggered"])
    near = bool(788 <= int(value["correct"]) <= 791 and structural["safe"] and local_triggered)

    historical_b0 = comparisons["tuned_b0_vs_historical_b0"]
    historical_b1 = comparisons["tuned_b1_vs_historical_b1"]
    neither_improved_historical = historical_b0["correct_delta"] <= 0 and historical_b1["correct_delta"] <= 0
    if b1_gate["reached"] and b1_outperforms:
        decision = "ABIDE_PRIVATE_TARGET_GO"
    elif b0_gate["reached"] and not b1_outperforms:
        decision = "ABIDE_BASE_TARGET_ONLY"
    elif positive:
        decision = "ABIDE_TUNE_POSITIVE"
    elif near:
        decision = "ABIDE_TUNE_NEAR"
    elif private_no_gain_reasons:
        decision = "ABIDE_PRIVATE_NO_GAIN"
    else:
        decision = "ABIDE_TUNE_NO_GAIN"
    return {
        "decision": decision,
        "target_status": "ABIDE_TARGET_REACHED" if (b0_gate["reached"] or b1_gate["reached"]) else "ABIDE_TARGET_NOT_REACHED",
        "tuned_b0_target": b0_gate, "tuned_b1_target": b1_gate,
        "tuned_b1_outperforms_tuned_b0": b1_outperforms,
        "tuned_b1_structural_safety": structural, "positive_gain_gate": positive,
        "private_no_gain_reasons": private_no_gain_reasons,
        "neither_tuned_model_improved_historical_correct": neither_improved_historical,
        "local_refinement_triggered": local_triggered,
        "next_action": "continue with a separate ABIDE-5 tuning experiment; do not modify model structure in this round",
    }


def render_report(runtime: dict[str, Any], registry: dict[str, Any], results: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]], comparisons: dict[str, Any], decision: dict[str, Any], total_seconds: float) -> str:
    tuned_b0 = results[registry["tuned_b0_trial_id"]]
    tuned_b1 = results[registry["tuned_b1_trial_id"]]
    displayed = [("Historical B0", references["B0"]), ("Historical B1", references["B1"]), ("TUNED_B0", tuned_b0), ("TUNED_B1", tuned_b1)]
    lines = [
        "# ABIDE Binary Task Hyperparameter Search v1", "",
        f"- Source commit: `{runtime['source_commit']}`",
        "- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)",
        f"- Branch: `{BRANCH}`", "- Device: `cuda:0`",
        f"- Decision: **{decision['decision']}**", f"- Target status: **{decision['target_status']}**",
        f"- Formal aggregation time: {total_seconds:.3f} seconds", "",
        "## Locked protocol", "",
        "All trials use ABIDE ADS_CN (871 subjects: 403 ADS/ASD positive at class index 0 and 468 CN negative at class index 1), four real modalities, seed 0, ten full-batch transductive folds, 400 epochs, historical global full-dataset class weights, criterion_lossv2 with main weighted CE plus two unnormalized weighted OVR CE terms, label smoothing 0.05, orthogonality 0, Adam, grad clip 1, CustomCosineAnnealingLR(T_max=400, eta_min=0.0001), test-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch, graph off, EMA off, and no ensemble.",
        "The calibration/nested-CV experiment is not reused because it changed loss, class-weight scope, and epoch selection.", "",
        "## Selected configurations", "",
        f"- TUNED_B0: `{tuned_b0['trial_id']}`, lr={tuned_b0['spec']['lr']}, WD={tuned_b0['spec']['weight_decay']}, dropout={tuned_b0['spec']['dropout']}.",
        f"- TUNED_B1: `{tuned_b1['trial_id']}`, lr={tuned_b1['spec']['lr']}, WD={tuned_b1['spec']['weight_decay']}, dropout={tuned_b1['spec']['dropout']}, rank={tuned_b1['spec']['rank']}, adapter LR multiplier={tuned_b1['spec']['adapter_lr_multiplier']}.",
        f"- Registered local LR refinement: `{registry['local_refinement']['triggered']}` ({registry['local_refinement']['reason']}).", "",
        "## Primary results", "",
        "| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Fold PR mean +/- SD | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Params | Train sec | Infer sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in displayed:
        m = result["metrics"]
        lines.append(f"| {label} | {m['correct']}/{m['n']} | {m['acc']:.7f} | {result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f} | {m['roc_auc']:.7f} | {result['fold_roc_auc_mean']:.7f} +/- {result['fold_roc_auc_sample_std']:.7f} | {m['pr_auc']:.7f} | {result['fold_pr_auc_mean']:.7f} +/- {result['fold_pr_auc_sample_std']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {m['weighted_f1']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {result['parameter_count']} | {result['training_time_seconds']:.1f} | {result['inference_time_seconds']:.4f} |")
    lines += ["", "### Confusion and prediction counts", ""]
    for label, result in displayed:
        m = result["metrics"]
        lines.append(f"- {label}: confusion={m['confusion_matrix']} in [ADS,CN] order; TP/FN/TN/FP={m['tp']}/{m['fn']}/{m['tn']}/{m['fp']}; predicted ADS/CN={m['predicted_counts']['ADS']}/{m['predicted_counts']['CN']}.")
    lines += ["", "## Registered trials", "", "| Trial | Stage | Arm | LR | WD | Dropout | Rank | Adapter mult | Static safe | Correct | ROC-AUC | Fold AUC mean | Macro-F1 | BACC |", "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for result in sorted(results.values(), key=lambda item: int(item["spec"]["ordinal"])):
        s, m = result["spec"], result["metrics"]
        lines.append(f"| {s['trial_id']} | {s['stage']} | {s['arm']} | {s['lr']:.8g} | {s['weight_decay']:.8g} | {s['dropout']:.4g} | {s['rank']} | {s['adapter_lr_multiplier']:.3g} | {result['safety']['safe']} | {m['correct']} | {m['roc_auc']:.7f} | {result['fold_roc_auc_mean']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} |")
    lines += ["", "## Per-fold selected epochs", ""]
    for label, result in (("TUNED_B0", tuned_b0), ("TUNED_B1", tuned_b1)):
        lines += [f"### {label}", "", "| Fold | Best epoch | Correct | ACC | ROC-AUC |", "|---:|---:|---:|---:|---:|"]
        for row in result["fold_metrics"]:
            lines.append(f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} | {row['roc_auc']:.7f} |")
        lines.append("")
    lines += ["## Paired comparisons", "", "| Comparison | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | Exact McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        d = comparison["metric_deltas"]
        lines.append(f"| {name} | {comparison['correct_delta']:+d} | {comparison['repairs']} | {comparison['damages']} | {comparison['changed']} | {d['acc']:+.7f} | {d['roc_auc']:+.7f} | {d['pr_auc']:+.7f} | {d['macro_f1']:+.7f} | {d['bacc']:+.7f} | {comparison['exact_mcnemar_p']:.8f} |")
    mechanism = tuned_b1.get("mechanism")
    lines += ["", "## TUNED_B1 mechanism diagnostics", ""]
    if mechanism:
        lines += [
            f"- Private/shared ratio mean/max: {mechanism['private_shared_ratio_mean']:.7f} / {mechanism['private_shared_ratio_max']:.7f}.",
            f"- Per-modality ratio mean: `{json.dumps(mechanism['private_shared_ratio_mean_by_modality'], sort_keys=True)}`.",
            f"- Per-modality ratio max: `{json.dumps(mechanism['private_shared_ratio_max_by_modality'], sort_keys=True)}`.",
            f"- Adapter maximum gradients: `{json.dumps(mechanism['adapter_max_gradient_by_tensor'], sort_keys=True)}`.",
            f"- Private collapse: `{mechanism['private_collapse_any_fold']}`; all folds trained: `{mechanism['private_trained_all_folds']}`.",
            f"- Category-Global cosine mean: {mechanism['category_global_cosine_mean']:.7f}.",
            f"- Parameter increase: {mechanism['parameter_delta']} ({mechanism['parameter_count_b0']} -> {mechanism['parameter_count_b1']}).",
        ]
    lines += ["", "## DGFMC comparison", "",
        f"- DGFMC fold mean ACC 90.82%: TUNED_B1={100*tuned_b1['fold_acc_mean']:.4f}% -> `{tuned_b1['fold_acc_mean'] > 0.9082}`.",
        f"- DGFMC fold mean ROC-AUC 90.84%: TUNED_B1={100*tuned_b1['fold_roc_auc_mean']:.4f}% -> `{tuned_b1['fold_roc_auc_mean'] > 0.9084}`.",
        "- The comparison uses ten-fold mean +/- sample SD, not pooled OOF ACC/AUC.", "",
        "## Answers to the fourteen required questions", "",
        f"1. Best base LR/WD/dropout: {tuned_b0['spec']['lr']} / {tuned_b0['spec']['weight_decay']} / {tuned_b0['spec']['dropout']}.",
        f"2. TUNED_B0: {tuned_b0['metrics']['correct']}/871, ACC={tuned_b0['metrics']['acc']:.7f}, AUC={tuned_b0['metrics']['roc_auc']:.7f}.",
        f"3. Best private rank/multiplier: {tuned_b1['spec']['rank']} / {tuned_b1['spec']['adapter_lr_multiplier']}.",
        f"4. TUNED_B1: {tuned_b1['metrics']['correct']}/871, ACC={tuned_b1['metrics']['acc']:.7f}, AUC={tuned_b1['metrics']['roc_auc']:.7f}.",
        f"5. TUNED_B1 exceeds TUNED_B0: `{decision['tuned_b1_outperforms_tuned_b0']}`; repairs/damages={comparisons['tuned_b1_vs_tuned_b0']['repairs']}/{comparisons['tuned_b1_vs_tuned_b0']['damages']}.",
        f"6. Historical private negative gain reversed: `{comparisons['tuned_b1_vs_tuned_b0']['correct_delta'] > 0 and comparisons['tuned_b1_vs_tuned_b0']['repairs'] > comparisons['tuned_b1_vs_tuned_b0']['damages']}`.",
        f"7. Reached 792/871: `{max(tuned_b0['metrics']['correct'], tuned_b1['metrics']['correct']) >= 792}`.",
        f"8. Fold mean ACC exceeds 90.82%: `{tuned_b1['fold_acc_mean'] > 0.9082}` ({tuned_b1['fold_acc_mean']:.7f}).",
        f"9. Fold mean AUC exceeds 90.84%: `{tuned_b1['fold_roc_auc_mean'] > 0.9084}` ({tuned_b1['fold_roc_auc_mean']:.7f}).",
        f"10. SEN and SPE both improve from TUNED_B0: `{comparisons['tuned_b1_vs_tuned_b0']['metric_deltas']['sen'] > 0 and comparisons['tuned_b1_vs_tuned_b0']['metric_deltas']['spe'] > 0}`.",
        f"11. Main source of Correct gain: `{'private adapter' if comparisons['tuned_b1_vs_tuned_b0']['correct_delta'] > 0 else 'base tuning or neither'}`.",
        "12. Continue a separate ABIDE-5 tuning run: `True`.",
        "13. Evidence is sufficient to modify the unified architecture now: `False`; finish ABIDE-5 first.",
        f"14. Final Decision: `{decision['decision']}`.", "",
        "## Reproduction", "",
        f"Run from a clean local branch named `{BRANCH}` at source commit `{runtime['source_commit']}`:", "", "```text",
        "python -u -B scripts/run_abide_hparam_search_v1.py inspect",
        "python -u -B scripts/run_abide_hparam_search_v1.py smoke --device cuda:0",
        "python -u -B scripts/run_abide_hparam_search_v1.py search --device cuda:0",
        "python -u -B scripts/run_abide_hparam_search_v1.py formal --device cuda:0", "```", "",
    ]
    return "\n".join(lines)


def run_formal(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    started = time.perf_counter()
    source = engine.source_gate()
    runtime = engine.runtime_lock(source)
    engine.validate_smoke(runtime)
    base_context = engine.build_context(torch.device(device_text))
    registry, results = engine.validate_search(runtime, base_context)
    tuned_b0 = results[registry["tuned_b0_trial_id"]]
    tuned_b1 = results[registry["tuned_b1_trial_id"]]
    hist_b0, hist_b0_rows = historical_reference("B0")
    hist_b1, hist_b1_rows = historical_reference("B1")
    tuned_b0_rows = engine.parse_trial_rows(engine.trial_dir(tuned_b0["spec"]) / "oof_predictions.csv")
    tuned_b1_rows = engine.parse_trial_rows(engine.trial_dir(tuned_b1["spec"]) / "oof_predictions.csv")
    comparisons = {
        "tuned_b0_vs_historical_b0": engine.paired_rows(hist_b0, hist_b0_rows, tuned_b0, tuned_b0_rows),
        "tuned_b1_vs_historical_b1": engine.paired_rows(hist_b1, hist_b1_rows, tuned_b1, tuned_b1_rows),
        "tuned_b1_vs_tuned_b0": engine.paired_rows(tuned_b0, tuned_b0_rows, tuned_b1, tuned_b1_rows),
    }
    decision = final_decision(tuned_b0, tuned_b1, comparisons)
    engine.atomic_write_csv(RESULT_DIR / "tuned_b0_oof_predictions.csv", tuned_b0_rows)
    engine.atomic_write_csv(RESULT_DIR / "tuned_b1_oof_predictions.csv", tuned_b1_rows)
    formal_core = {
        "runtime_lock": runtime, "smoke_report_sha256": engine.file_sha256(SMOKE_DIR / "smoke_report.json"),
        "search_summary_sha256": engine.file_sha256(SEARCH_SUMMARY_PATH),
        "trial_manifest_sha256": engine.file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": engine.file_sha256(ALL_TRIALS_PATH),
        "mode": "strictly validate and aggregate complete registered trials; no complete trial retrained",
    }
    engine.atomic_write_json(RESULT_DIR / "formal_config.json", {**formal_core, "sha256": engine.payload_sha256(formal_core)})
    engine.atomic_write_json(RESULT_DIR / "comparisons.json", comparisons)
    engine.atomic_write_json(RESULT_DIR / "mechanism_diagnostics.json", {"tuned_b1_trial_id": tuned_b1["trial_id"], "mechanism": tuned_b1.get("mechanism")})
    elapsed = time.perf_counter() - started
    engine.atomic_write_text(RESULT_DIR / "REPORT.md", render_report(runtime, registry, results, {"B0": hist_b0, "B1": hist_b1}, comparisons, decision, elapsed))
    commands = (
        f"# Reproduce from source commit {runtime['source_commit']} on local branch {BRANCH}\n"
        "python -u -B scripts/run_abide_hparam_search_v1.py inspect\n"
        "python -u -B scripts/run_abide_hparam_search_v1.py smoke --device cuda:0\n"
        "python -u -B scripts/run_abide_hparam_search_v1.py search --device cuda:0\n"
        "python -u -B scripts/run_abide_hparam_search_v1.py formal --device cuda:0\n"
    )
    engine.atomic_write_text(RESULT_DIR / "reproduction_commands.txt", commands)
    summary_core = {
        "experiment": EXPERIMENT_ID, "runtime_lock": runtime,
        "historical_b0": hist_b0, "historical_b1": hist_b1,
        "tuned_b0": tuned_b0, "tuned_b1": tuned_b1,
        "comparisons": comparisons, "decision": decision,
        "registered_trial_count": len(results), "failure_count": 0,
        "formal_aggregation_seconds": float(elapsed),
        "trial_manifest_sha256": engine.file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": engine.file_sha256(ALL_TRIALS_PATH),
        "report_sha256": engine.file_sha256(RESULT_DIR / "REPORT.md"),
        "reproduction_commands_sha256": engine.file_sha256(RESULT_DIR / "reproduction_commands.txt"),
        "tuned_b0_oof_sha256": engine.file_sha256(RESULT_DIR / "tuned_b0_oof_predictions.csv"),
        "tuned_b1_oof_sha256": engine.file_sha256(RESULT_DIR / "tuned_b1_oof_predictions.csv"),
    }
    engine.atomic_write_json(RESULT_DIR / "summary.json", {**summary_core, "sha256": engine.payload_sha256(summary_core)})
    print(json.dumps({"formal": "COMPLETE", "decision": decision["decision"], "tuned_b0": tuned_b0["metrics"], "tuned_b1": tuned_b1["metrics"]}, indent=2))


def install_profile() -> None:
    values = {
        "ROOT": ROOT, "EXPERIMENT_ID": EXPERIMENT_ID, "BRANCH": BRANCH, "BASE_COMMIT": BASE_COMMIT,
        "HISTORICAL_RESULT_COMMIT": HISTORICAL_RESULT_COMMIT, "RESULT_DIR": RESULT_DIR,
        "PROTOCOL_PATH": PROTOCOL_PATH, "CONFIG_PATH": CONFIG_PATH, "INSPECT_PATH": INSPECT_PATH,
        "FOLD_MANIFEST_PATH": FOLD_MANIFEST_PATH, "HISTORICAL_DIR": HISTORICAL_DIR,
        "SMOKE_DIR": SMOKE_DIR, "WORK_DIR": WORK_DIR, "TRIALS_DIR": TRIALS_DIR,
        "TRIAL_MANIFEST_PATH": TRIAL_MANIFEST_PATH, "ALL_TRIALS_PATH": ALL_TRIALS_PATH,
        "SEARCH_SUMMARY_PATH": SEARCH_SUMMARY_PATH, "HISTORICAL_PATHS": HISTORICAL_PATHS,
        "HISTORICAL_ANCHORS": HISTORICAL_ANCHORS, "REQUIRED_TRACKED": REQUIRED_TRACKED,
        "LOCKED_DEPENDENCIES": LOCKED_DEPENDENCIES,
    }
    for name, value in values.items():
        setattr(engine, name, value)
    overrides = {
        "protocol": protocol, "validate_oof": validate_oof, "historical_payload": historical_payload,
        "build_model": build_model, "make_training_objects": make_training_objects,
        "run_smoke": run_smoke, "safety": safety, "is_historical_exact": is_historical_exact,
        "run_search": run_search, "historical_reference": historical_reference,
        "target_gate": target_gate, "final_decision": final_decision,
        "render_report": render_report, "run_formal": run_formal,
    }
    for name, value in overrides.items():
        setattr(engine, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect")
    for mode in ("smoke", "search", "formal", "resume"):
        child = subparsers.add_parser(mode)
        child.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    install_profile()
    args = parse_args()
    if args.mode == "inspect": engine.run_inspect()
    elif args.mode == "smoke": run_smoke(args.device)
    elif args.mode in ("search", "resume"): run_search(args.device)
    elif args.mode == "formal": run_formal(args.device)
    else: raise engine.InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
