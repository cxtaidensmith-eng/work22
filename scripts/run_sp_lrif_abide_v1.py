"""Registered fixed-config SP-LRIF-ABIDE v1 experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2  # noqa: E402
from Model.sp_lrif import SPLRIFDualBranchModel  # noqa: E402
from Utils import SET_Random  # noqa: E402
from scripts import run_abide_hparam_search_v1 as parent  # noqa: E402
from scripts import run_tad_binary_hparam_search_v1 as engine  # noqa: E402


EXPERIMENT_ID = "sp_lrif_abide_v1"
BRANCH = "experiment/sp-lrif-abide-v1"
BASE_COMMIT = "02e9caefa3ca89a3e5f13e892049c6b9e0c5ca9b"
PARENT_SOURCE_COMMIT = "c9c7a289145a87fbab355e85828fa5f62f9819b0"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
WORK_DIR = RESULT_DIR / "work"
TRIALS_DIR = WORK_DIR / "compact"
PARENT_DIR = ROOT / "experiments" / "abide_hparam_search_v1"
PARENT_PROTOCOL_PATH = PARENT_DIR / "protocol.json"
PARENT_FOLD_PATH = PARENT_DIR / "fold_manifest.json"
PARENT_SUMMARY_PATH = PARENT_DIR / "summary.json"
B0_OOF_PATH = PARENT_DIR / "tuned_b0_oof_predictions.csv"
B1_OOF_PATH = PARENT_DIR / "tuned_b1_oof_predictions.csv"
FOLDS = tuple(range(10))
EPOCHS = 400
SEED = 0
SMOKE_EPOCHS = 3
INTERACTION_RANK = 4
BASE_LR = 0.00625
WEIGHT_DECAY = 0.002
DROPOUT = 0.45
PRIVATE_RANK = 4
PRIVATE_MULTIPLIER = 2.0
SP_MULTIPLIER = 1.0

REQUIRED_TRACKED = (
    ".gitignore",
    "Model/sp_lrif.py",
    "scripts/run_sp_lrif_abide_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
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
    "scripts/run_abide_hparam_search_v1.py",
    "scripts/run_tad_binary_hparam_search_v1.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
    "experiments/abide_hparam_search_v1/protocol.json",
    "experiments/abide_hparam_search_v1/fold_manifest.json",
    "experiments/abide_hparam_search_v1/summary.json",
    "experiments/abide_hparam_search_v1/tuned_b0_oof_predictions.csv",
    "experiments/abide_hparam_search_v1/tuned_b1_oof_predictions.csv",
)

ORIGINAL_TRAIN_UPDATE = engine.historical.train_update
ORIGINAL_PRIVATE_DIAGNOSTICS = engine.historical.private_diagnostics


def require(condition: bool, message: str) -> None:
    engine.require(condition, message)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def protocol() -> dict[str, Any]:
    value = engine.read_json(PARENT_PROTOCOL_PATH)
    require(value["dataset"] == "ABIDE" and value["task"] == "ADS_CN", "ABIDE task changed")
    require(value["class_names"] == ["ADS", "CN"] and value["positive_index"] == 0, "Class semantics changed")
    require(value["sample_count"] == 871 and len(value["modalities"]) == 4, "Dataset shape changed")
    training = value["training"]
    require(training["epochs"] == 400 and training["seed"] == 0, "Epoch/seed changed")
    require(training["graph_use_graph"] is False and training["ema"] is False, "Graph/EMA changed")
    return value


def fixed_spec(trial_id: str = "SP_LRIF") -> dict[str, Any]:
    base = engine.trial_spec(
        trial_id, "FIXED", "B1", BASE_LR, WEIGHT_DECAY, DROPOUT,
        rank=PRIVATE_RANK, multiplier=PRIVATE_MULTIPLIER, ordinal=1,
    )
    base.update({"interaction_rank": INTERACTION_RANK, "sp_lrif_lr_multiplier": SP_MULTIPLIER})
    core = {key: value for key, value in base.items() if key != "config_sha256"}
    base["config_sha256"] = engine.payload_sha256(core)
    return base


def d3_spec(trial_id: str = "D3_REFERENCE") -> dict[str, Any]:
    return engine.trial_spec(
        trial_id, "REFERENCE", "B1", BASE_LR, WEIGHT_DECAY, DROPOUT,
        rank=PRIVATE_RANK, multiplier=PRIVATE_MULTIPLIER, ordinal=0,
    )


def trial_context(base_context: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    value = dict(base_context)
    value["protocol"] = copy.deepcopy(base_context["protocol"])
    training = value["protocol"]["training"]
    training["lr"] = float(spec["lr"])
    training["weight_decay"] = float(spec["weight_decay"])
    training["drop_rate"] = float(spec["dropout"])
    return value


def reference_payload() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary = engine.read_json(PARENT_SUMMARY_PATH)
    require(summary["runtime_lock"]["source_commit"] == PARENT_SOURCE_COMMIT, "Parent source anchor changed")
    results = {"B0": summary["tuned_b0"], "B1": summary["tuned_b1"]}
    rows = {
        "B0": engine.parse_trial_rows(B0_OOF_PATH),
        "B1": engine.parse_trial_rows(B1_OOF_PATH),
    }
    context = engine.build_context(torch.device("cpu"))
    expected = {
        "B0": {"correct": 777, "acc": 0.8920780711825488, "roc_auc": 0.900951199338296, "macro_f1": 0.8913146212990102, "bacc": 0.890784394816653, "sen": 0.8734491315136477, "spe": 0.9081196581196581},
        "B1": {"correct": 775, "acc": 0.8897818599311137, "roc_auc": 0.88286568683591, "macro_f1": 0.8889136881038167, "bacc": 0.8881306865177833, "sen": 0.8660049627791563, "spe": 0.9102564102564102},
    }
    for arm in ("B0", "B1"):
        parent.validate_oof(rows[arm], context, arm)
        computed = engine.metrics(rows[arm])
        for key, target in expected[arm].items():
            require(math.isclose(float(computed[key]), target, rel_tol=0.0, abs_tol=1e-9), f"{arm} anchor changed: {key}")
        require(results[arm]["metrics"] == computed, f"{arm} summary and OOF differ")
    del context
    return results, rows


def _build_original_d3(context: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    return parent.build_model(context, d3_spec())


def build_model(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    if spec["arm"] == "B0":
        model, audit = _build_original_d3(context)
        audit["reference_role"] = "D3 parameter reference"
        return model, audit
    kwargs = engine.historical.model_kwargs(context)
    SET_Random(SEED)
    reference = parent.HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
    historical_training_rng = engine.capture_rng()
    SET_Random(SEED)
    model = SPLRIFDualBranchModel(
        **kwargs, cme_arm="c1", adapter_rank=PRIVATE_RANK,
        router_hidden=16, modality_embedding_dim=8,
    ).to(context["device"])
    b0_common_diff = engine.common_state_max_diff(reference, model)
    require(b0_common_diff == 0.0, "D3 common initialization changed")
    del reference
    engine.restore_rng(historical_training_rng)
    attach_rng = engine.capture_rng()
    common_before = engine.clone_cpu_state(model)
    model.attach_sp_lrif(INTERACTION_RANK)
    engine.restore_rng(attach_rng)
    common_after = model.state_dict()
    common_diff = max(
        float((tensor - common_after[name].detach().cpu()).abs().max())
        for name, tensor in common_before.items()
    )
    require(common_diff == 0.0, "Attaching SP-LRIF changed a D3 tensor")
    sp_named = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith("sp_lrif.")]
    require(len(sp_named) == 4 and sum(parameter.numel() for _, parameter in sp_named) == 1024, "SP-LRIF schema changed")
    require(bool(torch.count_nonzero(model.sp_lrif.proj_out.weight) == 0), "proj_out is not zero initialized")
    model._sp_lrif_initial_state = {name: parameter.detach().cpu().clone() for name, parameter in sp_named}
    audit = {
        "parameter_count": engine.parameter_count(model),
        "d3_parameter_count": engine.parameter_count(model) - 1024,
        "sp_lrif_parameter_count": 1024,
        "common_state_max_abs_diff": common_diff if audit_common else "smoke_locked_zero",
        "b0_common_state_max_abs_diff": b0_common_diff if audit_common else "smoke_locked_zero",
        "query_count": len(model.label_pools), "ovr_head_count": len(model._Auxi_classifier),
        "adapter_count": len(model.private_adapters), "interaction_rank": INTERACTION_RANK,
        "fusion_dimension": model.sp_lrif.dimension,
    }
    require(audit["parameter_count"] == 336_377 and audit["d3_parameter_count"] == 335_353, "Parameter count changed")
    require(audit["query_count"] == 2 and audit["ovr_head_count"] == 2 and audit["adapter_count"] == 4, "D3 structure changed")
    return model, audit


class ThreeGroupRatioScheduler(engine.historical.RatioPreservingCustomCosineAnnealingLR):
    """Historical LR curve with fixed [base, private*2, SP*1] ratios."""

    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int, eta_min: float, last_epoch: int = -1):
        require(len(optimizer.param_groups) == 3, "SP-LRIF requires three optimizer groups")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.hold_epoch = 20
        self.multiplier = PRIVATE_MULTIPLIER
        self.initial_lrs_locked = [float(group["lr"]) for group in optimizer.param_groups]
        require(self.initial_lrs_locked == [BASE_LR, BASE_LR * 2.0, BASE_LR], "Initial LR groups changed")
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch)
        self.assert_ratio()

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.hold_epoch:
            base = self.initial_lrs_locked[0]
        else:
            current = self.last_epoch - self.hold_epoch
            base = self.eta_min + (self.initial_lrs_locked[0] - self.eta_min) * (
                1.0 + math.cos(math.pi * current / (self.T_max - self.hold_epoch))
            ) / 2.0
        return [base, base * PRIVATE_MULTIPLIER, base]

    def assert_ratio(self) -> None:
        base, private, sp = [float(group["lr"]) for group in self.optimizer.param_groups]
        tolerance = max(1e-14, abs(base) * 1e-12)
        require(abs(private - 2.0 * base) <= tolerance and abs(sp - base) <= tolerance, "LR ratio changed")

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        self.assert_ratio()


def make_training_objects(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    model, initialization = build_model(context, spec, audit_common=audit_common)
    require(spec["arm"] == "B1", "Only the fixed SP-LRIF candidate may train")
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(
        context["dataset_dict"], context["device"], rate=float(training["loss_rate"]),
        label_smoothing=float(training["label_smoothing"]),
    )
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    private = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
    sp = [(name, parameter) for name, parameter in named if name.startswith("sp_lrif.")]
    base = [(name, parameter) for name, parameter in named if not name.startswith(("private_adapters.", "sp_lrif."))]
    groups = [{id(parameter) for _, parameter in group} for group in (base, private, sp)]
    all_ids = {id(parameter) for _, parameter in named}
    require(len(private) == 16 and len(sp) == 4, "Optimizer tensor counts changed")
    require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "Optimizer groups overlap")
    require(groups[0] | groups[1] | groups[2] == all_ids, "Optimizer misses a trainable parameter")
    optimizer = torch.optim.Adam([
        {"params": [p for _, p in base], "lr": BASE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "base"},
        {"params": [p for _, p in private], "lr": BASE_LR * PRIVATE_MULTIPLIER, "weight_decay": WEIGHT_DECAY, "group_name": "private_adapters"},
        {"params": [p for _, p in sp], "lr": BASE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "sp_lrif"},
    ])
    scheduler = ThreeGroupRatioScheduler(
        optimizer, T_max=EPOCHS, eta_min=float(training["scheduler_eta_min"])
    )
    audit = {
        **initialization,
        "base_parameter_tensors": len(base), "adapter_parameter_tensors": len(private),
        "sp_lrif_parameter_tensors": len(sp),
        "base_parameter_names": [name for name, _ in base],
        "adapter_parameter_names": [name for name, _ in private],
        "sp_lrif_parameter_names": [name for name, _ in sp],
        "optimizer_coverage": 1.0, "optimizer_groups_disjoint": True,
        "group_lrs": [BASE_LR, BASE_LR * 2.0, BASE_LR], "weight_decay": WEIGHT_DECAY,
    }
    return model, criterion, optimizer, scheduler, audit


def train_update(model, criterion, optimizer, features, labels, train_mask, grad_clip):
    loss, gradients, output = ORIGINAL_TRAIN_UPDATE(
        model, criterion, optimizer, features, labels, train_mask, grad_clip
    )
    for name, parameter in model.named_parameters():
        if name.startswith("sp_lrif."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid SP gradient: {name}")
            gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    return loss, gradients, output


def private_diagnostics(context, model, test_mask, cumulative_gradient, initial_adapter):
    private_gradient = {name: value for name, value in cumulative_gradient.items() if name.startswith("private_adapters.")}
    base = ORIGINAL_PRIVATE_DIAGNOSTICS(context, model, test_mask, private_gradient, initial_adapter)
    with torch.no_grad():
        _, _, _, inter = engine.historical.infer(model, context["dataset_data"]["Feature"], return_intermediates=True)
        category = inter["Y"][test_mask]
        global_message = inter["G"][test_mask]
        agreement = inter["sp_lrif_agreement"][test_mask]
        disagreement = inter["sp_lrif_disagreement"][test_mask]
        delta = inter["sp_lrif_delta"][test_mask]
        direct = inter["sp_lrif_sum"][test_mask]
        cosine = torch.nn.functional.cosine_similarity(category, global_message, dim=-1)
        ratio = delta.norm(dim=-1) / (direct.norm(dim=-1) + 1e-12)
        enabled_probability = torch.softmax(inter["raw_logits"][test_mask], dim=-1)
        model.sp_lrif.enabled = False
        try:
            disabled_logits = engine.historical.infer(model, context["dataset_data"]["Feature"])[0][test_mask]
        finally:
            model.sp_lrif.enabled = True
        disabled_probability = torch.softmax(disabled_logits, dim=-1)
        probability_difference = (enabled_probability - disabled_probability).abs()
    sp_gradient = {name: float(value) for name, value in cumulative_gradient.items() if name.startswith("sp_lrif.")}
    initial_sp = model._sp_lrif_initial_state
    sp_delta = {
        name: float((parameter.detach().cpu() - initial_sp[name]).abs().max())
        for name, parameter in model.named_parameters() if name.startswith("sp_lrif.")
    }
    require(set(sp_gradient) == set(initial_sp) == set(sp_delta), "SP diagnostic schema changed")
    all_changed = all(math.isfinite(value) and value > 0.0 for value in sp_delta.values())
    delta_ratio_max = float(ratio.max().cpu())
    diagnostics = {
        **base,
        "category_global_cosine_mean": float(cosine.mean().cpu()),
        "delta_direct_sum_ratio_mean": float(ratio.mean().cpu()),
        "delta_direct_sum_ratio_max": delta_ratio_max,
        "agreement_mean_norm": float(agreement.norm(dim=-1).mean().cpu()),
        "disagreement_mean_norm": float(disagreement.norm(dim=-1).mean().cpu()),
        "sp_lrif_cumulative_max_gradient": sp_gradient,
        "sp_lrif_parameter_delta": sp_delta,
        "sp_lrif_all_parameters_changed": all_changed,
        "delta_collapse": bool(delta_ratio_max <= 1e-6),
        "delta_disabled_probability_max_abs_diff": float(probability_difference.max().cpu()),
        "delta_disabled_probability_mean_abs_diff": float(probability_difference.mean().cpu()),
        "delta_disabled_argmax_changed": int((enabled_probability.argmax(1) != disabled_probability.argmax(1)).sum().cpu()),
        "diagnostic_subject_count": int(test_mask.sum().cpu()),
    }
    require(all_changed, "Not every SP-LRIF parameter changed")
    require(not diagnostics["delta_collapse"], "SP-LRIF delta collapsed")
    return diagnostics


def aggregate_mechanism(spec, fold_summaries, d3_parameter_count):
    diagnostics = [summary["diagnostics"] for summary in fold_summaries]
    require(len(diagnostics) == 10 and all(item is not None for item in diagnostics), "Mechanism diagnostics missing")
    private_names = set(diagnostics[0]["adapter_cumulative_max_gradient"])
    sp_names = set(diagnostics[0]["sp_lrif_cumulative_max_gradient"])
    total_subjects = sum(int(item["diagnostic_subject_count"]) for item in diagnostics)
    mechanism = {
        "interaction_rank": INTERACTION_RANK,
        "category_global_cosine_mean": float(statistics.mean(item["category_global_cosine_mean"] for item in diagnostics)),
        "delta_direct_sum_ratio_mean": float(sum(item["delta_direct_sum_ratio_mean"] * item["diagnostic_subject_count"] for item in diagnostics) / total_subjects),
        "delta_direct_sum_ratio_max": float(max(item["delta_direct_sum_ratio_max"] for item in diagnostics)),
        "agreement_mean_norm": float(sum(item["agreement_mean_norm"] * item["diagnostic_subject_count"] for item in diagnostics) / total_subjects),
        "disagreement_mean_norm": float(sum(item["disagreement_mean_norm"] * item["diagnostic_subject_count"] for item in diagnostics) / total_subjects),
        "sp_lrif_max_gradient_by_tensor": {name: float(max(item["sp_lrif_cumulative_max_gradient"][name] for item in diagnostics)) for name in sorted(sp_names)},
        "private_max_gradient_by_tensor": {name: float(max(item["adapter_cumulative_max_gradient"][name] for item in diagnostics)) for name in sorted(private_names)},
        "sp_lrif_all_parameters_changed_all_folds": bool(all(item["sp_lrif_all_parameters_changed"] for item in diagnostics)),
        "delta_collapse_any_fold": bool(any(item["delta_collapse"] for item in diagnostics)),
        "delta_disabled_probability_max_abs_diff": float(max(item["delta_disabled_probability_max_abs_diff"] for item in diagnostics)),
        "delta_disabled_probability_mean_abs_diff": float(sum(item["delta_disabled_probability_mean_abs_diff"] * item["diagnostic_subject_count"] for item in diagnostics) / total_subjects),
        "delta_disabled_argmax_changed": int(sum(item["delta_disabled_argmax_changed"] for item in diagnostics)),
        "private_trained_all_folds": bool(all(item["private_trained"] for item in diagnostics)),
        "private_collapse_any_fold": bool(any(item["private_collapse"] for item in diagnostics)),
        "parameter_count_d3": int(d3_parameter_count),
        "parameter_count_sp_lrif": int(fold_summaries[0]["object_audit"]["parameter_count"]),
        "parameter_delta": int(fold_summaries[0]["object_audit"]["parameter_count"] - d3_parameter_count),
    }
    require(mechanism["parameter_count_d3"] == 335_353 and mechanism["parameter_delta"] == 1024, "Mechanism parameter counts changed")
    return mechanism


def prediction_rows(context, spec, fold, logits, probabilities, truth):
    rows = engine.prediction_rows.__wrapped__(context, spec, fold, logits, probabilities, truth) if hasattr(engine.prediction_rows, "__wrapped__") else None
    if rows is None:
        anchors = engine.expected_fold_rows(context, fold)
        rows = []
        for anchor, label, logit, probability in zip(anchors, truth, logits, probabilities):
            require(int(anchor["truth"]) == int(label), "Fold truth anchor changed")
            rows.append({
                "task_id": context["task_id"], "trial_id": spec["trial_id"], "arm": spec["arm"],
                "fold": int(fold), "subject_id": anchor["subject_id"],
                "original_csv_index": int(anchor["original_csv_index"]), "feature_sha256": anchor["feature_sha256"],
                "truth": int(label), "prediction": int(np.argmax(probability)),
                "logit_0": float(logit[0]), "logit_1": float(logit[1]),
                "probability_0": float(probability[0]), "probability_1": float(probability[1]),
                "positive_probability": float(probability[0]),
            })
    for row in rows:
        row["positive_probability"] = float(row["probability_0"])
    return rows


def validate_oof(rows, context, arm=None):
    parent.validate_oof(rows, context, arm)
    for row in rows:
        require(math.isclose(float(row["positive_probability"]), float(row["probability_0"]), rel_tol=0.0, abs_tol=1e-12), "Positive probability direction changed")


def safety(result):
    counts = result["metrics"]["predicted_counts"]
    collapse = min(int(counts["ADS"]), int(counts["CN"])) < 88
    return {"safe": not collapse, "collapse": collapse, "reasons": ["class prediction collapse"] if collapse else []}


def inspect_payload() -> dict[str, Any]:
    references, rows = reference_payload()
    context = engine.build_context(torch.device("cpu"))
    spec = fixed_spec("INSPECT_SP")
    candidate, audit = build_model(trial_context(context, spec), spec, audit_common=True)
    d3, _ = _build_original_d3(trial_context(context, d3_spec("INSPECT_D3")))
    features = context["dataset_data"]["Feature"]
    candidate.eval(); d3.eval()
    with torch.no_grad():
        d3_logits, _, _, d3_inter = d3(features, return_intermediates=True)
        sp_logits, _, _, sp_inter = candidate(features, return_intermediates=True)
    initial_diff = float((d3_logits - sp_logits).abs().max())
    direct_diff = float((sp_inter["H_fused"] - (sp_inter["Y"] + sp_inter["G"])).abs().max())
    require(initial_diff <= 1e-7 and direct_diff == 0.0, "Initial sum preservation failed")
    core = {
        "experiment": EXPERIMENT_ID, "branch": BRANCH, "parent_commit": BASE_COMMIT,
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "dataset": "ABIDE", "task": "ADS_CN", "sample_count": 871,
        "class_names": ["ADS", "CN"], "positive_class": "ADS/ASD", "positive_index": 0,
        "modalities": 4, "folds": list(FOLDS), "seed": SEED, "epochs": EPOCHS,
        "fixed_spec": fixed_spec(), "object_audit": audit,
        "fusion_interface": {"category_shape": list(sp_inter["Y"].shape), "global_shape": list(sp_inter["G"].shape), "fusion_dimension": int(sp_inter["Y"].shape[-1]), "location": "final Category and Global tensors immediately before DIFFormer/classifier"},
        "initial_logits_max_abs_diff_vs_d3": initial_diff,
        "initial_fused_sum_max_abs_diff": direct_diff,
        "reference_B0": references["B0"], "reference_B1": references["B1"],
        "reference_oof_sha256": {"B0": engine.file_sha256(B0_OOF_PATH), "B1": engine.file_sha256(B1_OOF_PATH)},
        "reference_subject_id_sets_equal": {row["subject_id"] for row in rows["B0"]} == {row["subject_id"] for row in rows["B1"]},
        "parent_protocol_sha256": engine.file_sha256(PARENT_PROTOCOL_PATH),
        "parent_fold_manifest_sha256": engine.file_sha256(PARENT_FOLD_PATH),
        "experiment_config_sha256": engine.file_sha256(CONFIG_PATH),
        "model_sha256": engine.file_sha256(ROOT / "Model" / "sp_lrif.py"),
        "runner_sha256": engine.file_sha256(Path(__file__)),
    }
    del candidate, d3, context
    return {**core, "sha256": engine.payload_sha256(core)}


def run_inspect() -> None:
    payload = inspect_payload()
    engine.atomic_write_json(INSPECT_PATH, payload)
    print(json.dumps({"inspect": "PASS", "B0": 777, "B1": 775, "fusion_dimension": 64, "parent": BASE_COMMIT}, indent=2))


def validate_inspect() -> None:
    require(INSPECT_PATH.is_file() and engine.read_json(INSPECT_PATH) == inspect_payload(), "Inspect manifest drifted")


def source_hashes() -> dict[str, str]:
    return {relative: engine.file_sha256(ROOT / relative) for relative in (*REQUIRED_TRACKED, *LOCKED_DEPENDENCIES)}


def source_gate() -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree drifted")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index drifted")
    head = git("rev-parse", "HEAD").stdout.strip()
    require(head != BASE_COMMIT and git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Source commit ancestry changed")
    changed = {line.strip().replace("\\", "/") for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines() if line.strip()}
    require(changed == set(REQUIRED_TRACKED), f"Source commit scope changed: {sorted(changed)}")
    for relative in REQUIRED_TRACKED:
        require(git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0, f"Untracked source: {relative}")
    for relative in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", relative, check=False).returncode == 0, f"Parent dependency changed: {relative}")
    validate_inspect()
    return head


def runtime_lock(source_commit: str) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID, "source_commit": source_commit,
        "source_hashes": source_hashes(), "experiment_config_sha256": engine.file_sha256(CONFIG_PATH),
        "inspect_sha256": engine.file_sha256(INSPECT_PATH), "parent_commit": BASE_COMMIT,
        "parent_source_commit": PARENT_SOURCE_COMMIT, "fold_manifest_sha256": engine.file_sha256(PARENT_FOLD_PATH),
        "folds": list(FOLDS), "seed": SEED, "epochs": EPOCHS, "device": "cuda:0",
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = source_gate()
    runtime = runtime_lock(source)
    base = engine.build_context(torch.device(device_text))
    spec = fixed_spec("SMOKE_SP_LRIF")
    context = trial_context(base, spec)
    d3, _ = _build_original_d3(context)
    model, criterion, optimizer, scheduler, audit = make_training_objects(context, spec, audit_common=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    labels_before = labels.detach().clone()
    train_mask, test_mask, _ = engine.historical.fold_positions(context, 0)
    d3.eval(); model.eval()
    with torch.no_grad():
        d3_logits = d3(features)[0]
        sp_logits, _, _, inter = model(features, return_intermediates=True)
    initial_logits_diff = float((d3_logits - sp_logits).abs().max().cpu())
    initial_delta_max = float(inter["sp_lrif_delta"].abs().max().cpu())
    require(initial_logits_diff <= 1e-7 and initial_delta_max == 0.0, "Initial equivalence failed")
    common_diff = engine.common_state_max_diff(d3, model)
    require(common_diff == 0.0, "D3/SP common state changed")
    initial_sp = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name.startswith("sp_lrif.")}
    gradients = {name: 0.0 for name in initial_sp}
    gradient_by_epoch: list[dict[str, float]] = []
    losses: list[float] = []
    simplex: list[float] = []
    for epoch in range(1, SMOKE_EPOCHS + 1):
        loss, current, _ = train_update(model, criterion, optimizer, features, labels, train_mask, 1.0)
        sp_current = {name: value for name, value in current.items() if name.startswith("sp_lrif.")}
        gradient_by_epoch.append(sp_current)
        for name, value in sp_current.items(): gradients[name] = max(gradients[name], value)
        scheduler.step(); scheduler.assert_ratio()
        losses.append(float(loss))
        with torch.no_grad(): probability = torch.softmax(engine.historical.infer(model, features)[0], dim=-1)
        simplex.append(float((probability.sum(1) - 1.0).abs().max().cpu()))
    require(gradient_by_epoch[0]["sp_lrif.proj_out.weight"] > 0.0, "proj_out step-1 gradient is zero")
    require(all(math.isfinite(value) and value > 0.0 for value in gradients.values()), "SP projections did not all activate by epoch 3")
    parameter_delta = {name: float((parameter.detach().cpu() - initial_sp[name]).abs().max()) for name, parameter in model.named_parameters() if name in initial_sp}
    require(all(math.isfinite(value) and value > 0.0 for value in parameter_delta.values()), "SP parameters did not all change")
    require(torch.equal(labels, labels_before) and max(simplex) <= 2e-6 and all(math.isfinite(value) for value in losses), "Smoke numerical gate failed")
    smoke_config_core = {"runtime_lock": runtime, "spec": spec, "fold": 0, "epochs": SMOKE_EPOCHS}
    smoke_config = {**smoke_config_core, "sha256": engine.payload_sha256(smoke_config_core)}
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    engine.atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    checkpoint = {"runtime_lock": runtime, "spec": spec, "epoch": 3, "model": engine.clone_cpu_state(model), "optimizer": copy.deepcopy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": engine.capture_rng()}
    engine.atomic_torch_save(checkpoint_path, checkpoint)
    restored, _, restored_optimizer, restored_scheduler, _ = make_training_objects(context, spec)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"]); restored_scheduler.load_state_dict(payload["scheduler"]); restored_scheduler.assert_ratio()
    model.eval(); restored.eval()
    with torch.no_grad(): reload_diff = float((model(features)[0] - restored(features)[0]).abs().max().cpu())
    require(reload_diff == 0.0 and engine.checkpoint_state_finite(payload), "Smoke strict checkpoint reload failed")
    report_core = {
        "runtime_lock": runtime, "smoke_config_sha256": engine.file_sha256(SMOKE_DIR / "smoke_config.json"),
        "spec": spec, "fold": 0, "epochs": 3, "losses": losses,
        "output_shape": list(sp_logits.shape), "initial_delta_max_abs": initial_delta_max,
        "initial_logits_max_abs_diff_vs_d3": initial_logits_diff, "common_state_max_abs_diff": common_diff,
        "probability_sum_max_abs_error": max(simplex), "sp_gradient_by_epoch": gradient_by_epoch,
        "sp_max_gradient_by_tensor": gradients, "sp_parameter_delta_by_tensor": parameter_delta,
        "proj_out_step1_gradient_nonzero": True, "all_projections_active_by_epoch3": True,
        "optimizer_audit": audit, "checkpoint_sha256": engine.file_sha256(checkpoint_path),
        "checkpoint_reload_logits_max_abs_diff": reload_diff, "labels_unchanged": True,
        "device": torch.cuda.get_device_name(torch.device(device_text)), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    engine.atomic_write_json(SMOKE_DIR / "smoke_report.json", {**report_core, "sha256": engine.payload_sha256(report_core)})
    print(json.dumps({"smoke": "PASS", "losses": losses, "params": audit["parameter_count"]}, indent=2))
    del d3, model, restored, base, context
    torch.cuda.empty_cache()


def validate_smoke(runtime: dict[str, Any]) -> dict[str, Any]:
    config_path, report_path, checkpoint_path = SMOKE_DIR / "smoke_config.json", SMOKE_DIR / "smoke_report.json", SMOKE_DIR / "checkpoint_roundtrip.pt"
    require(config_path.is_file() and report_path.is_file() and checkpoint_path.is_file(), "Smoke artifacts missing")
    config, report = engine.read_json(config_path), engine.read_json(report_path)
    require(config["runtime_lock"] == runtime and report["runtime_lock"] == runtime, "Smoke runtime changed")
    core = {key: value for key, value in report.items() if key != "sha256"}
    require(report["sha256"] == engine.payload_sha256(core), "Smoke digest changed")
    require(report["smoke_config_sha256"] == engine.file_sha256(config_path) and report["checkpoint_sha256"] == engine.file_sha256(checkpoint_path), "Smoke artifact hash changed")
    require(report["initial_logits_max_abs_diff_vs_d3"] <= 1e-7 and report["initial_delta_max_abs"] == 0.0, "Smoke equivalence gate failed")
    require(report["proj_out_step1_gradient_nonzero"] and report["all_projections_active_by_epoch3"] and report["checkpoint_reload_logits_max_abs_diff"] == 0.0, "Smoke activation/reload gate failed")
    return report


def reference_result(arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results, rows = reference_payload()
    return results[arm], rows[arm]


def decision(result: dict[str, Any], vs_b0: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    m = result["metrics"]
    b0, _ = reference_result("B0")
    no_collapse = min(int(m["predicted_counts"]["ADS"]), int(m["predicted_counts"]["CN"])) >= 88
    delta_changes = mechanism["delta_disabled_argmax_changed"] > 0
    delta_active = not mechanism["delta_collapse_any_fold"] and delta_changes
    sen_safe = float(m["sen"]) >= float(b0["metrics"]["sen"]) - 0.01
    spe_safe = float(m["spe"]) >= float(b0["metrics"]["spe"]) - 0.01
    target_checks = {
        "correct_at_least_792": int(m["correct"]) >= 792,
        "fold_acc_mean_above_0_9082": float(result["fold_acc_mean"]) > 0.9082,
        "fold_auc_mean_above_0_9084": float(result["fold_roc_auc_mean"]) > 0.9084,
        "no_class_collapse": no_collapse,
        "bacc_not_below_b0": float(m["bacc"]) >= float(b0["metrics"]["bacc"]),
        "macro_f1_not_below_b0": float(m["macro_f1"]) >= float(b0["metrics"]["macro_f1"]),
    }
    strong_checks = {
        "correct_at_least_781": int(m["correct"]) >= 781,
        "net_repairs_at_least_4": int(vs_b0["repairs"] - vs_b0["damages"]) >= 4,
        "no_class_collapse": no_collapse,
        "macro_f1_safe": float(m["macro_f1"]) >= 0.8903146,
        "bacc_safe": float(m["bacc"]) >= 0.8897844,
        "pooled_auc_safe": float(m["roc_auc"]) >= 0.8979512,
        "asd_sen_drop_at_most_0_01": sen_safe,
        "cn_spe_drop_at_most_0_01": spe_safe,
        "delta_active_and_changes_predictions": delta_active,
    }
    near_checks = {
        "correct_778_to_780": 778 <= int(m["correct"]) <= 780,
        "repairs_exceed_damages": int(vs_b0["repairs"]) > int(vs_b0["damages"]),
        "macro_f1_safe": strong_checks["macro_f1_safe"], "bacc_safe": strong_checks["bacc_safe"],
        "pooled_auc_safe": strong_checks["pooled_auc_safe"], "no_class_collapse": no_collapse,
        "delta_changes_predictions": delta_changes,
    }
    if all(target_checks.values()): token = "SP_LRIF_ABIDE_TARGET_REACHED"
    elif all(strong_checks.values()): token = "SP_LRIF_ABIDE_STRONG_GO"
    elif all(near_checks.values()): token = "SP_LRIF_ABIDE_NEAR"
    else: token = "SP_LRIF_ABIDE_NO_GAIN"
    return {"decision": token, "target_checks": target_checks, "strong_checks": strong_checks, "near_checks": near_checks, "delta_active": delta_active, "no_class_collapse": no_collapse}


def render_report(runtime, result, b0, b1, comparisons, outcome, wall_seconds):
    m = result["metrics"]; mech = result["mechanism"]
    lines = [
        "# SP-LRIF-ABIDE v1", "",
        f"- Source commit: `{runtime['source_commit']}`", f"- Parent result commit: `{BASE_COMMIT}`",
        "- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)",
        f"- Branch: `{BRANCH}`", "- Device: `cuda:0`", f"- Decision: **{outcome['decision']}**", "",
        "## Locked protocol", "",
        "ABIDE ADS_CN; 871 subjects; ADS/ASD positive at index 0 and CN negative at index 1; four real modalities; seed 0; ten full-batch transductive folds; 400 epochs; lr 0.00625; weight decay 0.002; dropout 0.45; private rank 4; private LR multiplier 2; historical global class weights and criterion_lossv2; Adam; grad clip 1; unchanged CustomCosine schedule; ACC > ROC-AUC > Macro-F1 > earliest checkpoint; graph/EMA/ensemble off.", "",
        "Only the final Category + Global add was extended to `C + G + delta` with the preregistered biasless rank-4 SP-LRIF formula. No search or second fusion was run.", "",
        "## Results", "",
        "| Model | Correct/N | ACC | pooled AUC | fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Params | Train sec | Infer sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("TUNED_B0", b0), ("D3", b1), ("SP-LRIF", result)):
        x = item["metrics"]
        lines.append(f"| {label} | {x['correct']}/{x['n']} | {x['acc']:.7f} | {x['roc_auc']:.7f} | {item['fold_roc_auc_mean']:.7f} +/- {item['fold_roc_auc_sample_std']:.7f} | {x['pr_auc']:.7f} | {x['macro_f1']:.7f} | {x['bacc']:.7f} | {x['weighted_f1']:.7f} | {x['sen']:.7f} | {x['spe']:.7f} | {item['parameter_count']} | {item['training_time_seconds']:.1f} | {item['inference_time_seconds']:.4f} |")
    lines += ["", f"SP-LRIF confusion [ADS,CN]: `{m['confusion_matrix']}`; predicted ADS/CN={m['predicted_counts']['ADS']}/{m['predicted_counts']['CN']}.", f"Fold ACC mean +/- sample SD: {result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f}.", "", "## Per-fold best checkpoints", "", "| Fold | Best epoch | Correct | ACC | AUC |", "|---:|---:|---:|---:|---:|"]
    for row in result["fold_metrics"]:
        lines.append(f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} | {row['roc_auc']:.7f} |")
    lines += ["", "## Subject-aligned comparisons", "", "| Reference | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | SEN delta | SPE delta | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        d = comparison["metric_deltas"]
        lines.append(f"| {name} | {comparison['correct_delta']:+d} | {comparison['repairs']} | {comparison['damages']} | {comparison['changed']} | {d['acc']:+.7f} | {d['roc_auc']:+.7f} | {d['pr_auc']:+.7f} | {d['macro_f1']:+.7f} | {d['bacc']:+.7f} | {d['sen']:+.7f} | {d['spe']:+.7f} | {comparison['exact_mcnemar_p']:.8f} |")
    lines += ["", "## Minimal mechanism diagnostics", "",
        f"- Initial logits max difference vs D3: {engine.read_json(INSPECT_PATH)['initial_logits_max_abs_diff_vs_d3']:.3g}.",
        f"- Category-Global cosine mean: {mech['category_global_cosine_mean']:.7f}.",
        f"- delta/(C+G) norm ratio mean/max: {mech['delta_direct_sum_ratio_mean']:.7f} / {mech['delta_direct_sum_ratio_max']:.7f}.",
        f"- Agreement/disagreement mean norm: {mech['agreement_mean_norm']:.7f} / {mech['disagreement_mean_norm']:.7f}.",
        f"- SP maximum gradients: `{json.dumps(mech['sp_lrif_max_gradient_by_tensor'], sort_keys=True)}`.",
        f"- Every SP parameter changed in every fold: `{mech['sp_lrif_all_parameters_changed_all_folds']}`; delta collapse: `{mech['delta_collapse_any_fold']}`.",
        f"- Same-checkpoint delta-disabled probability max/mean difference: {mech['delta_disabled_probability_max_abs_diff']:.7f} / {mech['delta_disabled_probability_mean_abs_diff']:.7f}; argmax changed={mech['delta_disabled_argmax_changed']}.",
        f"- Added parameters: {mech['parameter_delta']} ({mech['parameter_count_d3']} -> {mech['parameter_count_sp_lrif']}); formal aggregation wall={wall_seconds:.3f}s.",
        "", "## Answers to the twelve required questions", "",
        f"1. Exceeded D3 775/871: `{m['correct'] > 775}` ({m['correct']}/871).",
        f"2. Exceeded TUNED_B0 777/871: `{m['correct'] > 777}`.",
        f"3. Reached the 781 structural GO line: `{m['correct'] >= 781}`.",
        f"4. Reached the 792 external target: `{m['correct'] >= 792}`.",
        f"5. Gain comes from hard classification, not only AUC: `{comparisons['vs_TUNED_B0']['correct_delta'] > 0}`.",
        f"6. ASD SEN and CN SPE are both safe: `{outcome['strong_checks']['asd_sen_drop_at_most_0_01'] and outcome['strong_checks']['cn_spe_drop_at_most_0_01']}`.",
        f"7. Repairs exceed damages vs B0: `{comparisons['vs_TUNED_B0']['repairs'] > comparisons['vs_TUNED_B0']['damages']}` ({comparisons['vs_TUNED_B0']['repairs']}/{comparisons['vs_TUNED_B0']['damages']}).",
        f"8. Agreement and disagreement both activated: `{mech['agreement_mean_norm'] > 0 and mech['disagreement_mean_norm'] > 0}`.",
        f"9. Delta changed predictions: `{mech['delta_disabled_argmax_changed'] > 0}` ({mech['delta_disabled_argmax_changed']}).",
        f"10. Added parameters/inference overhead: {mech['parameter_delta']} parameters; SP-LRIF total infer={result['inference_time_seconds']:.4f}s versus D3 recorded {b1['inference_time_seconds']:.4f}s.",
        f"11. Final Decision: `{outcome['decision']}`.",
        f"12. Worth extending to three other tasks: `{outcome['decision'] in ('SP_LRIF_ABIDE_TARGET_REACHED','SP_LRIF_ABIDE_STRONG_GO')}`; this run does not execute them.",
        "", "## Reproduction", "", f"Run from a clean local branch named `{BRANCH}` at source commit `{runtime['source_commit']}`:", "", "```text",
        "python -u -B scripts/run_sp_lrif_abide_v1.py inspect",
        "python -u -B scripts/run_sp_lrif_abide_v1.py smoke --device cuda:0",
        "python -u -B scripts/run_sp_lrif_abide_v1.py formal --device cuda:0", "```", "",
    ]
    return "\n".join(lines)


def run_formal(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    started = time.perf_counter()
    source = source_gate(); runtime = runtime_lock(source); validate_smoke(runtime)
    base_context = engine.build_context(torch.device(device_text))
    spec = fixed_spec()
    result = engine.train_trial(spec, base_context, runtime)
    rows = engine.parse_trial_rows(TRIALS_DIR / spec["trial_id"] / "oof_predictions.csv")
    validate_oof(rows, trial_context(base_context, spec), "B1")
    b0, b0_rows = reference_result("B0"); b1, b1_rows = reference_result("B1")
    comparisons = {
        "vs_TUNED_B0": engine.paired_rows(b0, b0_rows, result, rows),
        "vs_D3": engine.paired_rows(b1, b1_rows, result, rows),
    }
    outcome = decision(result, comparisons["vs_TUNED_B0"], result["mechanism"])
    engine.atomic_write_csv(RESULT_DIR / "oof_predictions.csv", sorted(rows, key=lambda row: int(row["original_csv_index"])))
    engine.atomic_write_json(RESULT_DIR / "fold_metrics.json", {"spec": spec, "fold_metrics": result["fold_metrics"]})
    engine.atomic_write_json(RESULT_DIR / "mechanism_diagnostics.json", result["mechanism"])
    engine.atomic_write_json(RESULT_DIR / "comparisons.json", comparisons)
    formal_core = {"runtime_lock": runtime, "spec": spec, "smoke_report_sha256": engine.file_sha256(SMOKE_DIR / "smoke_report.json"), "oof_sha256": engine.file_sha256(RESULT_DIR / "oof_predictions.csv")}
    engine.atomic_write_json(RESULT_DIR / "formal_config.json", {**formal_core, "sha256": engine.payload_sha256(formal_core)})
    elapsed = time.perf_counter() - started
    engine.atomic_write_text(RESULT_DIR / "REPORT.md", render_report(runtime, result, b0, b1, comparisons, outcome, elapsed))
    commands = f"# Reproduce on branch {BRANCH} at source commit {source}\npython -u -B scripts/run_sp_lrif_abide_v1.py inspect\npython -u -B scripts/run_sp_lrif_abide_v1.py smoke --device cuda:0\npython -u -B scripts/run_sp_lrif_abide_v1.py formal --device cuda:0\n"
    engine.atomic_write_text(RESULT_DIR / "reproduction_commands.txt", commands)
    summary_core = {
        "experiment": EXPERIMENT_ID, "runtime_lock": runtime, "spec": spec,
        "reference_B0": b0, "reference_B1_D3": b1, "sp_lrif": result,
        "comparisons": comparisons, "decision": outcome, "failure_count": 0,
        "formal_wall_seconds": float(elapsed), "oof_sha256": engine.file_sha256(RESULT_DIR / "oof_predictions.csv"),
        "fold_metrics_sha256": engine.file_sha256(RESULT_DIR / "fold_metrics.json"),
        "mechanism_sha256": engine.file_sha256(RESULT_DIR / "mechanism_diagnostics.json"),
        "comparisons_sha256": engine.file_sha256(RESULT_DIR / "comparisons.json"),
        "formal_config_sha256": engine.file_sha256(RESULT_DIR / "formal_config.json"),
        "report_sha256": engine.file_sha256(RESULT_DIR / "REPORT.md"),
        "reproduction_commands_sha256": engine.file_sha256(RESULT_DIR / "reproduction_commands.txt"),
    }
    engine.atomic_write_json(RESULT_DIR / "summary.json", {**summary_core, "sha256": engine.payload_sha256(summary_core)})
    print(json.dumps({"formal": "COMPLETE", "correct": result["metrics"]["correct"], "decision": outcome["decision"]}, indent=2))


def install_profile() -> None:
    parent.install_profile()
    values = {
        "ROOT": ROOT, "EXPERIMENT_ID": EXPERIMENT_ID, "BRANCH": BRANCH, "BASE_COMMIT": BASE_COMMIT,
        "RESULT_DIR": RESULT_DIR, "CONFIG_PATH": CONFIG_PATH, "INSPECT_PATH": INSPECT_PATH,
        "FOLD_MANIFEST_PATH": PARENT_FOLD_PATH, "SMOKE_DIR": SMOKE_DIR, "WORK_DIR": WORK_DIR,
        "TRIALS_DIR": TRIALS_DIR, "REQUIRED_TRACKED": REQUIRED_TRACKED, "LOCKED_DEPENDENCIES": LOCKED_DEPENDENCIES,
    }
    for name, value in values.items(): setattr(engine, name, value)
    overrides = {
        "protocol": protocol, "build_model": build_model, "make_training_objects": make_training_objects,
        "trial_context": trial_context, "prediction_rows": prediction_rows, "validate_oof": validate_oof,
        "aggregate_mechanism": aggregate_mechanism, "safety": safety, "source_gate": source_gate,
        "runtime_lock": runtime_lock, "validate_smoke": validate_smoke,
    }
    for name, value in overrides.items(): setattr(engine, name, value)
    engine.historical.train_update = train_update
    engine.historical.private_diagnostics = private_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("inspect")
    for mode in ("smoke", "formal"):
        child = sub.add_parser(mode); child.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    install_profile()
    args = parse_args()
    if args.mode == "inspect": run_inspect()
    elif args.mode == "smoke": run_smoke(args.device)
    elif args.mode == "formal": run_formal(args.device)
    else: raise engine.InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
