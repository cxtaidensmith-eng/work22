#!/usr/bin/env python3
"""TADPOLE SMCI/PMCI D5 orthogonality-loss removal experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect as pyinspect
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
from Loss.loss_fn import orthogonality_lossv2  # noqa: E402
from scripts import run_tad_binary_hparam_search_v1 as d5  # noqa: E402


EXPERIMENT_ID = "tad_binary_no_orth_v1"
BRANCH = "experiment/tad-binary-no-orth-v1"
BASE_COMMIT = "fcf57882ce242d07d4944755003559bfe97c534e"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_PATH = RESULT_DIR / "protocol.json"
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
SMOKE_DIR = RESULT_DIR / "smoke"
FORMAL_DIR = RESULT_DIR / "formal"
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
SMOKE_EPOCHS = 3
OLD_ORTH_RATE = 0.0001
NEW_ORTH_RATE = 0.0
EXPECTED_PARAMETERS = 617197

REFERENCE_DIR = ROOT / "experiments" / "tad_binary_hparam_search_v1"
REFERENCE_OOF = REFERENCE_DIR / "trials" / "D5" / "oof_predictions.csv"
REFERENCE_SUMMARY = REFERENCE_DIR / "trials" / "D5" / "summary.json"
REFERENCE_CONFIG = REFERENCE_DIR / "trials" / "D5" / "config.json"
REFERENCE_FOLDS = REFERENCE_DIR / "fold_manifest.json"
REFERENCE_SHA256 = {
    "oof": "65302affb7968b393221cb1fbd357b49beee6c5aab5b6aebcbde541c09cd01e1",
    "summary": "85078d7d0afdec9b322db304c8e736d899657f38c02aff2b91fdc9c8d8593adf",
    "config": "51a900e61ffb5820602e31a88d9e459781b63e3286bfec867767d93d6070953f",
    "folds": "93b5ffda8f9e9d0e4ad0987f84d5165d72d98e317234b9a02995e01c06a9fdaa",
}
REFERENCE_OOF_CANONICAL_LF_SHA256 = "143c493df779a424728e42444da8ad1d0560b0aeae2334d78a30dc4d070bfc54"

IMPLEMENTATION_PATHS = (
    ".gitignore",
    "scripts/run_tad_binary_no_orth_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocol.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
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
    "scripts/run_tad_binary_hparam_search_v1.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
)


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    d5.atomic_write_json(path, value)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d5.atomic_write_csv(path, rows)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check, capture_output=True, text=True,
    )


def current_head() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def git_blob_sha256(revision: str, relative_path: str) -> str:
    completed = subprocess.run(
        [
            "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
            "cat-file", "blob", f"{revision}:{relative_path}",
        ],
        check=True, capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def protocol() -> dict[str, Any]:
    value = d5.read_json(PROTOCOL_PATH)
    require(value["dataset"] == "TADPOLE", "Dataset changed")
    require(value["task"] == "SMCI_PMCI", "Task changed")
    require(value["sample_count"] == 535, "Sample count changed")
    require(value["class_counts"] == {"SMCI": 490, "PMCI": 45}, "Class counts changed")
    require(value["positive_class"] == "PMCI" and value["positive_index"] == 1, "Positive class changed")
    require(len(value["modalities"]) == 5, "Modality count changed")
    training = value["training"]
    require(training["lr"] == 0.0125, "D5 learning rate changed")
    require(training["weight_decay"] == 0.00025, "D5 weight decay changed")
    require(training["drop_rate"] == 0.67, "D5 dropout changed")
    require(training["loss_rate"] == 0.0, "No-orth loss rate changed")
    require(training["historical_loss_rate"] == OLD_ORTH_RATE, "Historical loss rate changed")
    require(training["epochs"] == EPOCHS and training["seed"] == SEED, "Epoch/seed changed")
    locked_training = {
        "label_smoothing": 0.05,
        "input_noise_std": 0.05,
        "drop_path": 0.05,
        "grad_clip": 1.0,
        "scheduler": "CustomCosineAnnealingLR",
        "scheduler_t_max": 400,
        "scheduler_eta_min": 0.0001,
        "class_weight_scope": "global_historical",
        "graph_use_graph": False,
        "ema": False,
    }
    for key, expected in locked_training.items():
        require(training[key] == expected, f"D5 protocol changed: {key}")
    require(value["private_rank"] == 8, "Private rank changed")
    require(value["adapter_lr_multiplier"] == 1.0, "Adapter LR multiplier changed")
    require(value["prior_free_modality_prior"] is False, "Prior-Free switch enabled")
    return value


def formal_spec() -> dict[str, Any]:
    return d5.trial_spec(
        "D5_NO_ORTH", "NO_ORTH", "B1", 0.0125, 0.00025, 0.67,
        rank=8, multiplier=1.0, ordinal=0,
    )


def build_context(device: torch.device) -> dict[str, Any]:
    return d5.historical.build_context(protocol(), device)


def total_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class NoOrthCriterion:
    """Exact D5 supervised losses, with no call to orthogonality_lossv2."""

    def __init__(self, dataset_dict: dict[str, Any], device: torch.device):
        legacy = criterion_lossv2(
            dataset_dict, device, rate=OLD_ORTH_RATE, label_smoothing=0.05,
        )
        self.CE_loss = legacy.CE_loss
        self.aux_loss_dict = legacy.aux_loss_dict
        self.rate = NEW_ORTH_RATE
        self.historical_rate = OLD_ORTH_RATE
        self.training_raw_orth_evaluations = 0

    def compute(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        _embeddings: Any,
        auxiliary: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main = self.CE_loss(output[mask], labels[mask])
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=2,
        ).transpose(0, 1).reshape(2, -1)
        ovr_0 = self.aux_loss_dict["aux_loss_0"](
            auxiliary[0][mask], one_hot[0][mask],
        )
        ovr_1 = self.aux_loss_dict["aux_loss_1"](
            auxiliary[1][mask], one_hot[1][mask],
        )
        auxiliary_sum = ovr_0 + ovr_1
        total = main + auxiliary_sum
        zero = total.new_zeros(())
        return total, {
            "main_ce": main,
            "ovr_0": ovr_0,
            "ovr_1": ovr_1,
            "auxiliary_sum": auxiliary_sum,
            "orthogonality_contribution": zero,
            "total": total,
        }

    def __call__(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        embeddings: Any,
        auxiliary: Any,
    ) -> torch.Tensor:
        return self.compute(output, labels, mask, embeddings, auxiliary)[0]


def make_training_objects(
    context: dict[str, Any],
) -> tuple[torch.nn.Module, NoOrthCriterion, torch.optim.Optimizer, Any, dict[str, Any]]:
    model, _legacy_criterion, optimizer, scheduler, audit = d5.make_training_objects(
        context, formal_spec(),
    )
    criterion = NoOrthCriterion(context["dataset_dict"], context["device"])
    require(total_parameter_count(model) == EXPECTED_PARAMETERS, "Total parameter count changed")
    require(trainable_parameter_count(model) == EXPECTED_PARAMETERS, "Trainable parameter count changed")
    require(not hasattr(model, "prior_free_modality_prior"), "Prior-Free implementation leaked into D5 base")
    require(math.isclose(float(criterion.rate), 0.0, abs_tol=0.0), "Orthogonality rate is not exactly zero")
    return model, criterion, optimizer, scheduler, audit


def reference_rows() -> list[dict[str, Any]]:
    return d5.parse_trial_rows(REFERENCE_OOF)


def reference_audit(context: dict[str, Any]) -> dict[str, Any]:
    for name, path in {
        "oof": REFERENCE_OOF,
        "summary": REFERENCE_SUMMARY,
        "config": REFERENCE_CONFIG,
        "folds": REFERENCE_FOLDS,
    }.items():
        require(path.is_file(), f"D5 reference missing: {path}")
        require(file_sha256(path) == REFERENCE_SHA256[name], f"D5 reference hash changed: {name}")
    canonical_oof_sha256 = git_blob_sha256(
        "HEAD",
        REFERENCE_OOF.relative_to(ROOT).as_posix(),
    )
    require(
        canonical_oof_sha256 == REFERENCE_OOF_CANONICAL_LF_SHA256,
        "D5 canonical OOF Git-blob hash changed",
    )
    config = d5.read_json(REFERENCE_CONFIG)
    expected_config = {
        "lr": 0.0125, "weight_decay": 0.00025, "dropout": 0.67,
        "rank": 8, "adapter_lr_multiplier": 1.0,
        "epochs": 400, "seed_per_fold": 0,
    }
    for key, expected in expected_config.items():
        require(config[key] == expected, f"D5 config changed: {key}")
    require(config["folds"] == list(FOLDS), "D5 fold list changed")
    rows = reference_rows()
    d5.validate_oof(rows, context, "B1")
    measured = d5.metrics(rows)
    summary = d5.read_json(REFERENCE_SUMMARY)
    require(d5.metrics_close(measured, summary["metrics"]), "D5 OOF/summary metrics disagree")
    require(measured["correct"] == 519, "D5 Correct changed")
    require(measured["confusion_matrix"] == [[483, 7], [9, 36]], "D5 confusion changed")
    registered = d5.read_json(REFERENCE_FOLDS)
    require(registered["fold_manifest"] == context["fold_manifest"], "D5 fold manifest changed")
    return {
        "oof_path": REFERENCE_OOF.relative_to(ROOT).as_posix(),
        "oof_sha256": REFERENCE_SHA256["oof"],
        "oof_working_tree_crlf_sha256": REFERENCE_SHA256["oof"],
        "oof_canonical_lf_git_blob_sha256": canonical_oof_sha256,
        "summary_sha256": REFERENCE_SHA256["summary"],
        "config_sha256": REFERENCE_SHA256["config"],
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "metrics": measured,
        "subject_count": len(rows),
        "unique_subject_count": len({row["subject_id"] for row in rows}),
        "positive_class": "PMCI",
        "config": expected_config,
    }


def state_max_diff(left: torch.nn.Module, right: torch.nn.Module) -> float:
    left_state, right_state = left.state_dict(), right.state_dict()
    require(left_state.keys() == right_state.keys(), "Model state keys changed")
    values = []
    for name in left_state:
        require(left_state[name].shape == right_state[name].shape, f"State shape changed: {name}")
        values.append(float((left_state[name].cpu() - right_state[name].cpu()).abs().max()))
    return max(values, default=0.0)


def optimizer_signature(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {
            "names": [names[id(parameter)] for parameter in group["params"]],
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": list(group["betas"]),
            "eps": float(group["eps"]),
        }
        for group in optimizer.param_groups
    ]


def initialization_fairness(context: dict[str, Any]) -> dict[str, Any]:
    historical_context = copy.deepcopy(context)
    historical_context["protocol"]["training"]["loss_rate"] = OLD_ORTH_RATE
    historical_model, historical_criterion, historical_optimizer, historical_scheduler, _ = (
        d5.make_training_objects(historical_context, formal_spec())
    )
    new_model, new_criterion, new_optimizer, new_scheduler, _ = make_training_objects(context)
    state_diff = state_max_diff(historical_model, new_model)
    require(state_diff == 0.0, "Seeded initialization changed")
    historical_model.eval()
    new_model.eval()
    with torch.no_grad():
        old_logits = historical_model(context["dataset_data"]["Feature"])[0]
        new_logits = new_model(context["dataset_data"]["Feature"])[0]
    logits_diff = float((old_logits - new_logits).abs().max().cpu())
    require(logits_diff <= 1e-7, "Step-0 logits changed")
    old_optimizer = optimizer_signature(historical_model, historical_optimizer)
    new_optimizer_signature = optimizer_signature(new_model, new_optimizer)
    require(old_optimizer == new_optimizer_signature, "Optimizer groups changed")
    require(
        historical_scheduler.state_dict() == new_scheduler.state_dict(),
        "Scheduler state changed",
    )
    gate = torch.sigmoid(new_model.modal_gate_logit.detach()).cpu().tolist()
    noise = (new_model._modal_noise_std.detach() * new_model.noise_scale).cpu().tolist()
    audit = {
        "all_parameter_keys_and_shapes_equal": True,
        "all_initial_parameter_max_abs_diff": state_diff,
        "step0_logits_max_abs_diff": logits_diff,
        "optimizer_groups_equal": True,
        "scheduler_state_equal": True,
        "total_parameter_count_equal": total_parameter_count(historical_model) == total_parameter_count(new_model),
        "trainable_parameter_count_equal": trainable_parameter_count(historical_model) == trainable_parameter_count(new_model),
        "historical_gate_equal": True,
        "historical_noise_equal": True,
        "effective_gate": gate,
        "effective_noise_std": noise,
        "prior_free_modality_prior": False,
        "historical_criterion_rate": float(historical_criterion.rate),
        "new_criterion_rate": float(new_criterion.rate),
        "rng_construction_rule": "both models constructed after identical SET_Random(seed=0) reset in the historical D5 builder",
    }
    del historical_model, new_model
    return audit


def inspect_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    context = build_context(torch.device("cpu"))
    reference = reference_audit(context)
    model, _criterion, _optimizer, _scheduler, object_audit = make_training_objects(context)
    model.eval()
    with torch.no_grad():
        logits, embeddings, auxiliary = model(context["dataset_data"]["Feature"])
    require(tuple(logits.shape) == (535, 2), "Inspect logits shape changed")
    require(len(embeddings) == 2 and len(auxiliary) == 2, "Binary representation schema changed")
    embedding_shapes = [list(value.shape) for value in embeddings]
    require(all(shape[0] == 535 for shape in embedding_shapes), "Representations are not full-dataset")
    normalized = [
        value / torch.clamp(torch.norm(value, dim=1, keepdim=True), min=1e-8)
        for value in embeddings
    ]
    similarity = normalized[0] @ normalized[1].T
    require(tuple(similarity.shape) == (535, 535), "Old orthogonality is not N by N")
    manual_raw = torch.norm(similarity, p="fro") ** 2
    raw = orthogonality_lossv2(embeddings)
    require(float((raw - manual_raw).abs()) <= 1e-4, "Old orthogonality implementation differs from audited formula")
    train_mask, test_mask, _ = d5.historical.fold_positions(context, 0)
    source_lines, source_start = pyinspect.getsourcelines(orthogonality_lossv2)
    criterion_lines, criterion_start = pyinspect.getsourcelines(criterion_lossv2.__call__)
    cross_protocol = ROOT / "experiments" / "cross_dataset_a012_structure_v1" / "protocols"
    abide = d5.read_json(cross_protocol / "abide_ads_cn.json")
    abide5 = d5.read_json(cross_protocol / "abide5_ads_cn.json")
    c1_source = (ROOT / "scripts" / "run_c1_broad_hparam_search_v1.py").read_text(encoding="utf-8")
    modality_names = [entry["name"] for entry in protocol()["modalities"]]
    core = {
        "experiment": EXPERIMENT_ID,
        "dataset": "TADPOLE",
        "task": "SMCI_PMCI",
        "reference": reference,
        "loss_code_audit": {
            "representation_shapes": embedding_shapes,
            "similarity_matrix_shape": list(similarity.shape),
            "produces_n_by_n": True,
            "contains_cross_subject_terms": True,
            "contains_test_rows": True,
            "train_mask_applied_to_orthogonality": False,
            "fold0_train_rows": int(train_mask.sum()),
            "fold0_test_rows": int(test_mask.sum()),
            "original_orthogonality_rate": OLD_ORTH_RATE,
            "original_total_formula": "main_ce + (ovr_0 + ovr_1) + 0.0001 * raw_orthogonality",
            "new_total_formula": "main_ce + (ovr_0 + ovr_1)",
            "raw_orthogonality_fold0_eval": float(raw),
            "orthogonality_function": {
                "path": "Loss/loss_fn.py",
                "start_line": source_start,
                "line_count": len(source_lines),
            },
            "criterion_call": {
                "path": "Loss/loss_fn.py",
                "start_line": criterion_start,
                "line_count": len(criterion_lines),
            },
        },
        "historical_no_orth_context": {
            "ABIDE_loss_rate": abide["training"]["loss_rate"],
            "ABIDE5_loss_rate": abide5["training"]["loss_rate"],
            "A012_three_class_declares_orthogonality_false": '"orthogonality": False' in c1_source,
        },
        "initialization_fairness_fold0": initialization_fairness(context),
        "model_audit": {
            **object_audit,
            "total_parameter_count": total_parameter_count(model),
            "trainable_parameter_count": trainable_parameter_count(model),
            "modality_names": modality_names,
            "effective_gate": torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist(),
            "noise_factor": model._modal_noise_std.detach().cpu().tolist(),
            "effective_noise_std": (model._modal_noise_std.detach() * model.noise_scale).cpu().tolist(),
            "prior_free_modality_prior": False,
        },
        "inspection_scope": "task, D5 reference, fold manifest, loss implementation, initialization only",
    }
    fold_core = {
        "task_id": context["task_id"],
        "source": REFERENCE_FOLDS.relative_to(ROOT).as_posix(),
        "fold_manifest": context["fold_manifest"],
    }
    return (
        {**core, "sha256": payload_sha256(core)},
        {**fold_core, "sha256": payload_sha256(fold_core)},
    )


def run_inspect() -> None:
    inspect_value, folds = inspect_payload()
    atomic_json(CONFIG_PATH, experiment_config())
    atomic_json(INSPECT_PATH, inspect_value)
    atomic_json(FOLD_MANIFEST_PATH, folds)
    print(json.dumps({
        "inspect": "PASS",
        "n_by_n": True,
        "cross_subject_terms": True,
        "contains_test_rows": True,
        "reference_correct": inspect_value["reference"]["metrics"]["correct"],
    }, indent=2))


def validate_inspect() -> dict[str, Any]:
    require(INSPECT_PATH.is_file() and FOLD_MANIFEST_PATH.is_file(), "Run inspect first")
    value = d5.read_json(INSPECT_PATH)
    core = {key: item for key, item in value.items() if key != "sha256"}
    require(value["sha256"] == payload_sha256(core), "Inspect manifest digest changed")
    folds = d5.read_json(FOLD_MANIFEST_PATH)
    fold_core = {key: item for key, item in folds.items() if key != "sha256"}
    require(folds["sha256"] == payload_sha256(fold_core), "Fold manifest digest changed")
    context = build_context(torch.device("cpu"))
    require(folds["fold_manifest"] == context["fold_manifest"], "Fold manifest changed")
    reference_audit(context)
    return value


def experiment_config() -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "arm": "D5_NO_ORTH",
        "dataset": "TADPOLE",
        "task": "SMCI_PMCI",
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs": EPOCHS,
        "smoke_epochs": SMOKE_EPOCHS,
        "only_formal_change": "orthogonality_rate=0.0; raw orthogonality is never evaluated by the training criterion",
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
        "reference": "D5",
        "search": False,
        "single_model": True,
        "ensemble": False,
        "ema": False,
        "decision_tokens": ["NO_ORTH_POSITIVE", "NO_ORTH_NEUTRAL", "NO_ORTH_NO_GAIN"],
    }
    return {**core, "sha256": payload_sha256(core)}


def source_hashes() -> dict[str, str]:
    return {
        path: file_sha256(ROOT / path)
        for path in (*IMPLEMENTATION_PATHS, *LOCKED_DEPENDENCIES)
    }


def source_gate() -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree changed")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index changed")
    head = current_head()
    require(head != BASE_COMMIT, "Implementation must be committed before smoke/formal")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Base is not an ancestor")
    changed = {
        line.strip().replace("\\", "/")
        for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines()
        if line.strip()
    }
    require(changed == set(IMPLEMENTATION_PATHS), f"Source commit scope changed: {sorted(changed)}")
    for path in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", path, check=False).returncode == 0, f"Locked dependency changed: {path}")
    validate_inspect()
    require(d5.read_json(CONFIG_PATH) == experiment_config(), "Experiment config changed")
    return head


def runtime_lock(source_commit: str, device: str) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "experiment_config_sha256": file_sha256(CONFIG_PATH),
        "inspect_sha256": file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": file_sha256(FOLD_MANIFEST_PATH),
        "reference_oof_sha256": REFERENCE_SHA256["oof"],
        "spec": formal_spec(),
        "device": device,
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
    }
    return {**core, "sha256": payload_sha256(core)}


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    squared = sum(
        float(torch.sum(value.detach().double() ** 2).cpu())
        for value in gradients if value is not None
    )
    return math.sqrt(squared)


def smoke_loss_diagnostic(context: dict[str, Any]) -> dict[str, Any]:
    model, criterion, _optimizer, _scheduler, _audit = make_training_objects(context)
    model.train()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = d5.historical.fold_positions(context, 0)
    output, embeddings, auxiliary = model(features)
    supervised, components = criterion.compute(output, labels, train_mask, embeddings, auxiliary)
    raw = orthogonality_lossv2(embeddings)
    weighted = OLD_ORTH_RATE * raw
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    supervised_grad = torch.autograd.grad(supervised, parameters, retain_graph=True, allow_unused=True)
    orth_grad = torch.autograd.grad(weighted, parameters, retain_graph=False, allow_unused=True)
    supervised_norm = gradient_norm(supervised_grad)
    orth_norm = gradient_norm(orth_grad)
    require(supervised_norm > 0.0 and math.isfinite(supervised_norm), "Invalid supervised gradient norm")
    require(math.isfinite(orth_norm), "Invalid legacy orth gradient norm")
    return {
        "fold": 0,
        "representation_shapes": [list(value.shape) for value in embeddings],
        "similarity_matrix_shape": [535, 535],
        "contains_test_rows": bool(test_mask.sum() > 0),
        "raw_orthogonality": float(raw.detach().cpu()),
        "weighted_orthogonality": float(weighted.detach().cpu()),
        "supervised_loss": float(supervised.detach().cpu()),
        "weighted_orth_over_supervised_loss": float((weighted / supervised).detach().cpu()),
        "supervised_gradient_norm": supervised_norm,
        "weighted_orth_gradient_norm": orth_norm,
        "weighted_orth_over_supervised_gradient_norm": orth_norm / supervised_norm,
        "formal_orthogonality_contribution": float(components["orthogonality_contribution"]),
        "diagnostic_entered_optimizer_step": False,
    }


def gradient_topology(context: dict[str, Any], legacy: bool) -> dict[str, Any]:
    if legacy:
        legacy_context = copy.deepcopy(context)
        legacy_context["protocol"]["training"]["loss_rate"] = OLD_ORTH_RATE
        model, criterion, _optimizer, _scheduler, _audit = d5.make_training_objects(
            legacy_context, formal_spec(),
        )
    else:
        model, criterion, _optimizer, _scheduler, _audit = make_training_objects(context)
    model.train()
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _test_mask, _ = d5.historical.fold_positions(context, 0)
    output, embeddings, auxiliary = model(features)
    loss = criterion(output, labels, train_mask, embeddings, auxiliary)
    loss.backward()
    none_names: list[str] = []
    nonfinite_names: list[str] = []
    finite_nonzero_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            none_names.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite_names.append(name)
        elif bool(torch.any(parameter.grad != 0)):
            finite_nonzero_names.append(name)
    return {
        "none": none_names,
        "nonfinite": nonfinite_names,
        "finite_nonzero_count": len(finite_nonzero_names),
        "trainable_tensor_count": sum(1 for parameter in model.parameters() if parameter.requires_grad),
    }


def train_step(
    model: torch.nn.Module,
    criterion: NoOrthCriterion,
    optimizer: torch.optim.Optimizer,
    features: Any,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    grad_clip: float,
    expected_none_names: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, embeddings, auxiliary = model(features)
    loss, components = criterion.compute(output, labels, train_mask, embeddings, auxiliary)
    require(bool(torch.isfinite(loss)), "Training loss is non-finite")
    require(float(components["orthogonality_contribution"].detach().cpu()) == 0.0, "Orthogonality contribution is nonzero")
    loss.backward()
    gradients: dict[str, float] = {}
    actual_none: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            actual_none.add(name)
            continue
        require(bool(torch.isfinite(parameter.grad).all()), f"Non-finite gradient: {name}")
        gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    require(actual_none == expected_none_names, f"Gradient topology changed: {sorted(actual_none ^ expected_none_names)}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    numeric = {
        name: float(value.detach().cpu())
        for name, value in components.items()
    }
    for name in ("total", "main_ce", "ovr_0", "ovr_1", "auxiliary_sum"):
        require(math.isfinite(numeric[name]), f"Non-finite supervised loss component: {name}")
    require(
        numeric["total"] == numeric["main_ce"] + numeric["auxiliary_sum"]
        or math.isclose(
            numeric["total"], numeric["main_ce"] + numeric["auxiliary_sum"],
            rel_tol=0.0, abs_tol=2e-6,
        ),
        "No-orth loss association changed",
    )
    require(criterion.training_raw_orth_evaluations == 0, "Raw orthogonality entered training")
    return numeric, gradients


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = source_gate()
    device = torch.device(device_text)
    context = build_context(device)
    runtime = runtime_lock(source, device_text)
    fairness = initialization_fairness(context)
    diagnostic = smoke_loss_diagnostic(context)
    old_topology = gradient_topology(context, legacy=True)
    new_topology = gradient_topology(context, legacy=False)
    require(old_topology["none"] == new_topology["none"], "Expected gradient topology changed")
    require(not new_topology["nonfinite"], "No-orth smoke has non-finite gradients")

    model, criterion, optimizer, scheduler, audit = make_training_objects(context)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = d5.historical.fold_positions(context, 0)
    require(not bool(torch.any(train_mask & test_mask)), "Fold masks overlap")
    model.eval()
    with torch.no_grad():
        initial_logits = model(features)[0]
    require(tuple(initial_logits.shape) == (535, 2), "Smoke logits shape changed")
    initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    max_adapter_gradient = {name: 0.0 for name in initial_adapter}
    losses: list[float] = []
    component_history: list[dict[str, float]] = []
    probability_errors: list[float] = []
    for _epoch in range(1, SMOKE_EPOCHS + 1):
        components, gradients = train_step(
            model, criterion, optimizer, features, labels, train_mask,
            float(context["protocol"]["training"]["grad_clip"]),
            set(new_topology["none"]),
        )
        for name in max_adapter_gradient:
            max_adapter_gradient[name] = max(
                max_adapter_gradient[name], gradients.get(name, 0.0),
            )
        scheduler.step()
        scheduler.assert_ratio()
        losses.append(components["total"])
        component_history.append(components)
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(features)[0], dim=-1)
        probability_errors.append(
            float((probability.sum(dim=1) - 1.0).abs().max().cpu()),
        )
    require(all(math.isfinite(value) for value in losses), "Smoke loss is non-finite")
    require(all(value > 0.0 for value in max_adapter_gradient.values()), "Private Adapter did not receive finite nonzero gradients")
    adapter_delta = {
        name: float((parameter.detach().cpu() - initial_adapter[name]).abs().max())
        for name, parameter in model.named_parameters() if name in initial_adapter
    }
    require(all(value > 0.0 and math.isfinite(value) for value in adapter_delta.values()), "Private Adapter did not update")
    require(max(probability_errors) <= 2e-6, "Probability simplex changed")
    require(max(row["orthogonality_contribution"] for row in component_history) == 0.0, "Smoke orth contribution is nonzero")

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    smoke_config_core = {
        "runtime_lock": runtime,
        "fold": 0,
        "epochs": SMOKE_EPOCHS,
        "spec": formal_spec(),
        "orthogonality_rate": 0.0,
    }
    smoke_config = {**smoke_config_core, "sha256": payload_sha256(smoke_config_core)}
    atomic_json(SMOKE_DIR / "smoke_config.json", smoke_config)
    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    checkpoint = {
        "schema": 1,
        "runtime_lock": runtime,
        "smoke_config": smoke_config,
        "epoch": SMOKE_EPOCHS,
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
        "model": clone_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": d5.capture_rng(),
    }
    d5.atomic_torch_save(checkpoint_path, checkpoint)
    restored_model, restored_criterion, restored_optimizer, restored_scheduler, _ = (
        make_training_objects(context)
    )
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(restored["runtime_lock"] == runtime, "Smoke checkpoint runtime changed")
    require(restored["orthogonality_rate"] == 0.0, "Smoke checkpoint orth rate changed")
    restored_model.load_state_dict(restored["model"], strict=True)
    restored_optimizer.load_state_dict(restored["optimizer"])
    restored_scheduler.load_state_dict(restored["scheduler"])
    restored_scheduler.assert_ratio()
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        left = model(features)[0]
        right = restored_model(features)[0]
    reload_diff = float((left - right).abs().max().cpu())
    require(reload_diff == 0.0, "Strict checkpoint reload changed logits")
    report_core = {
        "runtime_lock": runtime,
        "smoke_config_sha256": file_sha256(SMOKE_DIR / "smoke_config.json"),
        "fold": 0,
        "epochs": SMOKE_EPOCHS,
        "forward_backward": "PASS",
        "logits_shape": [535, 2],
        "losses": losses,
        "component_history": component_history,
        "probability_sum_max_abs_error": max(probability_errors),
        "orthogonality_rate": 0.0,
        "formal_orthogonality_contribution_max_abs": 0.0,
        "legacy_diagnostic": diagnostic,
        "diagnostic_entered_optimizer_step": False,
        "historical_gradient_topology": old_topology,
        "no_orth_gradient_topology": new_topology,
        "adapter_gradient_max_by_tensor": max_adapter_gradient,
        "adapter_parameter_delta_by_tensor": adapter_delta,
        "initialization_fairness": fairness,
        "object_audit": {
            **audit,
            "total_parameter_count": total_parameter_count(model),
            "trainable_parameter_count": trainable_parameter_count(model),
        },
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_reload_logits_max_abs_diff": reload_diff,
        "effective_gate": torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist(),
        "effective_noise_std": (
            model._modal_noise_std.detach() * model.noise_scale
        ).cpu().tolist(),
        "prior_free_modality_prior": False,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    report = {**report_core, "sha256": payload_sha256(report_core)}
    atomic_json(SMOKE_DIR / "smoke_report.json", report)
    print(json.dumps({
        "smoke": "PASS",
        "losses": losses,
        "raw_orthogonality": diagnostic["raw_orthogonality"],
        "weighted_orth_over_supervised_loss": diagnostic["weighted_orth_over_supervised_loss"],
        "orthogonality_contribution": 0.0,
    }, indent=2))
    del restored_criterion, restored_model, model, context
    torch.cuda.empty_cache()


def validate_smoke(runtime: dict[str, Any]) -> dict[str, Any]:
    config_path = SMOKE_DIR / "smoke_config.json"
    report_path = SMOKE_DIR / "smoke_report.json"
    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    require(config_path.is_file() and report_path.is_file() and checkpoint_path.is_file(), "Smoke artifacts missing")
    config = d5.read_json(config_path)
    report = d5.read_json(report_path)
    config_core = {key: value for key, value in config.items() if key != "sha256"}
    report_core = {key: value for key, value in report.items() if key != "sha256"}
    require(config["sha256"] == payload_sha256(config_core), "Smoke config digest changed")
    require(report["sha256"] == payload_sha256(report_core), "Smoke report digest changed")
    require(config["runtime_lock"] == runtime and report["runtime_lock"] == runtime, "Smoke runtime changed")
    require(report["checkpoint_sha256"] == file_sha256(checkpoint_path), "Smoke checkpoint changed")
    require(report["orthogonality_rate"] == 0.0, "Smoke orth rate changed")
    require(report["formal_orthogonality_contribution_max_abs"] == 0.0, "Smoke orth contribution changed")
    require(report["checkpoint_reload_logits_max_abs_diff"] == 0.0, "Smoke reload gate failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["runtime_lock"] == runtime and checkpoint["orthogonality_rate"] == 0.0, "Smoke checkpoint lock changed")
    return report


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {
        "runtime_sha256": runtime["sha256"],
        "spec": formal_spec(),
        "fold": int(fold),
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
    }
    return {**core, "sha256": payload_sha256(core)}


def fold_paths(fold: int) -> dict[str, Path]:
    root = FORMAL_DIR / f"fold_{fold:02d}"
    return {
        "root": root,
        "resume": root / "resume.pt",
        "checkpoint": root / "checkpoint_best.pt",
        "oof": root / "oof_predictions.csv",
        "history": root / "epoch_metrics.csv",
        "summary": root / "summary.json",
        "complete": root / "complete.json",
    }


def selection_key(value: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(value["acc"]),
        float(value["roc_auc"]),
        float(value["macro_f1"]),
        -int(epoch),
    )


def prediction_rows(
    context: dict[str, Any],
    fold: int,
    logits: np.ndarray,
    probabilities: np.ndarray,
    truth: np.ndarray,
) -> list[dict[str, Any]]:
    return d5.prediction_rows(
        context, formal_spec(), fold, logits, probabilities, truth,
    )


def resume_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    lock: dict[str, Any],
    fold: int,
    epoch: int,
    best: dict[str, Any] | None,
    history: list[dict[str, Any]],
    elapsed: float,
    adapter_gradients: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "lock": lock,
        "fold": int(fold),
        "epoch": int(epoch),
        "optimizer_steps": int(epoch),
        "scheduler_steps": int(epoch),
        "orthogonality_rate": 0.0,
        "model": clone_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": d5.capture_rng(),
        "best": copy.deepcopy(best),
        "history": copy.deepcopy(history),
        "elapsed_seconds": float(elapsed),
        "adapter_cumulative_max_gradient": dict(adapter_gradients),
    }


def load_resume(
    path: Path,
    lock: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(value.get("schema") == 1 and value.get("lock") == lock, f"Resume lock changed: {path}")
    epoch = int(value.get("epoch", -1))
    require(0 < epoch <= EPOCHS, "Resume epoch invalid")
    require(value["optimizer_steps"] == epoch and value["scheduler_steps"] == epoch, "Resume step count changed")
    require(value["orthogonality_rate"] == 0.0, "Resume orth rate changed")
    adapter_names = {
        name for name, _parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    }
    require(
        set(value["adapter_cumulative_max_gradient"]) == adapter_names,
        "Resume adapter audit keys changed",
    )
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    require(int(scheduler.last_epoch) == epoch, "Resume scheduler epoch changed")
    scheduler.assert_ratio()
    d5.restore_rng(value["rng"])
    return value


def train_fold(
    base_context: dict[str, Any],
    fold: int,
    runtime: dict[str, Any],
    smoke: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = d5.trial_context(base_context, formal_spec())
    paths = fold_paths(fold)
    lock = fold_lock(runtime, context, fold)
    if paths["complete"].is_file():
        return load_completed_fold(base_context, fold, runtime)
    paths["root"].mkdir(parents=True, exist_ok=True)
    allowed = {path.name for name, path in paths.items() if name != "root"}
    allowed |= {name + ".tmp" for name in allowed}
    unknown = [path for path in paths["root"].iterdir() if path.name not in allowed]
    require(not unknown, f"Unknown partial fold artifacts: {unknown}")

    model, criterion, optimizer, scheduler, object_audit = make_training_objects(context)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = d5.historical.fold_positions(context, fold)
    expected_none = set(smoke["no_orth_gradient_topology"]["none"])
    adapter_names = [
        name for name, _parameter in model.named_parameters()
        if name.startswith("private_adapters.")
    ]
    adapter_gradients = {name: 0.0 for name in adapter_names}
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resumed_from_epoch = 0
    elapsed_before = 0.0
    if paths["resume"].is_file():
        resumed = load_resume(paths["resume"], lock, model, optimizer, scheduler)
        resumed_from_epoch = int(resumed["epoch"])
        start_epoch = resumed_from_epoch + 1
        best = resumed["best"]
        history = copy.deepcopy(resumed["history"])
        adapter_gradients = {
            name: float(value)
            for name, value in resumed["adapter_cumulative_max_gradient"].items()
        }
        require([row["epoch"] for row in history] == list(range(1, start_epoch)), "Resume history changed")
        elapsed_before = float(resumed["elapsed_seconds"])

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        components, gradients = train_step(
            model, criterion, optimizer, features, labels, train_mask,
            float(context["protocol"]["training"]["grad_clip"]), expected_none,
        )
        for name in adapter_gradients:
            adapter_gradients[name] = max(
                adapter_gradients[name], gradients.get(name, 0.0),
            )
        model.eval()
        with torch.no_grad():
            full_logits = d5.historical.infer(model, features)[0]
            test_logits_tensor = full_logits[test_mask].detach().cpu()
            test_probability_tensor = torch.softmax(test_logits_tensor, dim=-1)
        test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        test_logits = test_logits_tensor.numpy().astype(np.float64)
        test_probabilities = test_probability_tensor.numpy().astype(np.float64)
        epoch_metrics = d5.historical.binary_metrics(
            test_truth, test_probabilities, context["protocol"],
        )
        row = {
            "epoch": int(epoch),
            "loss_total": components["total"],
            "loss_main_ce": components["main_ce"],
            "loss_ovr_0": components["ovr_0"],
            "loss_ovr_1": components["ovr_1"],
            "loss_auxiliary_sum": components["auxiliary_sum"],
            "orthogonality_contribution": components["orthogonality_contribution"],
            "correct": int(epoch_metrics["correct"]),
            "acc": float(epoch_metrics["acc"]),
            "roc_auc": float(epoch_metrics["roc_auc"]),
            "pr_auc": float(epoch_metrics["pr_auc"]),
            "macro_f1": float(epoch_metrics["macro_f1"]),
            "bacc": float(epoch_metrics["bacc"]),
        }
        history.append(row)
        key = selection_key(epoch_metrics, epoch)
        if best is None or key > tuple(best["selection_key"]):
            best = {
                "epoch": int(epoch),
                "selection_key": list(key),
                "metrics": epoch_metrics,
                "components": {
                    key_name: components[key_name]
                    for key_name in (
                        "total", "main_ce", "ovr_0", "ovr_1",
                        "auxiliary_sum", "orthogonality_contribution",
                    )
                },
                "model": clone_state(model),
                "test_logits": test_logits,
                "test_probabilities": test_probabilities,
                "test_truth": test_truth,
            }
        scheduler.step()
        scheduler.assert_ratio()
        if epoch % 20 == 0 or epoch == EPOCHS:
            elapsed = elapsed_before + time.perf_counter() - started
            d5.atomic_torch_save(
                paths["resume"],
                resume_payload(
                    model, optimizer, scheduler, lock, fold, epoch, best,
                    history, elapsed, adapter_gradients,
                ),
            )

    require(best is not None and len(history) == EPOCHS, "Fold history incomplete")
    require([row["epoch"] for row in history] == list(range(1, EPOCHS + 1)), "Epoch history is not 1..400")
    require(int(scheduler.last_epoch) == EPOCHS, "Scheduler did not step 400 times")
    require(max(abs(row["orthogonality_contribution"]) for row in history) == 0.0, "Formal orthogonality contribution is nonzero")
    require(criterion.training_raw_orth_evaluations == 0, "Formal training evaluated raw orthogonality")
    require(all(value > 0.0 and math.isfinite(value) for value in adapter_gradients.values()), "Private Adapter did not train")

    model.load_state_dict(best["model"], strict=True)
    torch.cuda.synchronize(context["device"])
    inference_started = time.perf_counter()
    with torch.no_grad():
        replay_full = d5.historical.infer(model, features)[0]
    torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - inference_started
    replay_test = replay_full[test_mask].detach().cpu().numpy().astype(np.float64)
    require(float(np.max(np.abs(replay_test - best["test_logits"]))) <= 1e-6, "Best-state replay changed logits")
    rows = prediction_rows(
        context, fold, best["test_logits"], best["test_probabilities"],
        best["test_truth"],
    )
    recomputed = d5.metrics(rows)
    require(d5.metrics_close(recomputed, best["metrics"]), "Fold metrics changed")
    elapsed_total = elapsed_before + time.perf_counter() - started
    checkpoint = {
        "schema": 1,
        "lock": lock,
        "spec": formal_spec(),
        "fold": int(fold),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "best_model": best["model"],
        "final_epoch": EPOCHS,
        "optimizer_steps": EPOCHS,
        "scheduler_steps": EPOCHS,
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": d5.capture_rng(),
        "elapsed_seconds": float(elapsed_total),
    }
    d5.atomic_torch_save(paths["checkpoint"], checkpoint)
    atomic_csv(paths["oof"], rows)
    atomic_csv(paths["history"], history)
    best_row = history[int(best["epoch"]) - 1]
    summary = {
        "schema": 1,
        "lock": lock,
        "spec": formal_spec(),
        "fold": int(fold),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "selection_uses_test_fold_truth": True,
        "object_audit": {
            **object_audit,
            "total_parameter_count": total_parameter_count(model),
            "trainable_parameter_count": trainable_parameter_count(model),
        },
        "loss_components": {
            "first": history[0],
            "best": best_row,
            "last": history[-1],
        },
        "formal_orthogonality_contribution_max_abs": 0.0,
        "formal_raw_orthogonality_evaluations": 0,
        "adapter_cumulative_max_gradient": adapter_gradients,
        "optimizer_steps": EPOCHS,
        "scheduler_steps": EPOCHS,
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds": float(elapsed_total),
        "inference_seconds_full_graph": float(inference_seconds),
        "oof_count": len(rows),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "resume_sha256": file_sha256(paths["resume"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "history_sha256": file_sha256(paths["history"]),
        "effective_gate": torch.sigmoid(model.modal_gate_logit.detach()).cpu().tolist(),
        "effective_noise_std": (
            model._modal_noise_std.detach() * model.noise_scale
        ).cpu().tolist(),
        "prior_free_modality_prior": False,
    }
    atomic_json(paths["summary"], summary)
    complete_core = {
        "complete": True,
        "lock": lock,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "resume_sha256": file_sha256(paths["resume"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "history_sha256": file_sha256(paths["history"]),
        "summary_sha256": file_sha256(paths["summary"]),
    }
    complete = {**complete_core, "sha256": payload_sha256(complete_core)}
    atomic_json(paths["complete"], complete)
    print(f"[D5_NO_ORTH fold {fold}] best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    return load_completed_fold(base_context, fold, runtime)


def load_completed_fold(
    base_context: dict[str, Any],
    fold: int,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = d5.trial_context(base_context, formal_spec())
    paths = fold_paths(fold)
    require(all(paths[name].is_file() for name in ("resume", "checkpoint", "oof", "history", "summary", "complete")), f"Completed fold artifacts missing: {fold}")
    lock = fold_lock(runtime, context, fold)
    complete = d5.read_json(paths["complete"])
    complete_core = {key: value for key, value in complete.items() if key != "sha256"}
    require(complete["sha256"] == payload_sha256(complete_core), "Completion digest changed")
    require(complete["complete"] is True and complete["lock"] == lock, "Completion lock changed")
    for key, path_name in (
        ("checkpoint_sha256", "checkpoint"),
        ("resume_sha256", "resume"),
        ("oof_sha256", "oof"),
        ("history_sha256", "history"),
        ("summary_sha256", "summary"),
    ):
        require(complete[key] == file_sha256(paths[path_name]), f"Completed {path_name} changed")
    summary = d5.read_json(paths["summary"])
    require(summary["lock"] == lock and summary["fold"] == fold, "Fold summary ownership changed")
    require(summary["formal_orthogonality_contribution_max_abs"] == 0.0, "Completed fold orth contribution changed")
    require(summary["formal_raw_orthogonality_evaluations"] == 0, "Completed fold raw orth audit changed")
    history = d5.read_csv(paths["history"])
    require([int(row["epoch"]) for row in history] == list(range(1, EPOCHS + 1)), "Completed history changed")
    selected = max(
        history,
        key=lambda row: (
            float(row["acc"]), float(row["roc_auc"]),
            float(row["macro_f1"]), -int(row["epoch"]),
        ),
    )
    require(int(selected["epoch"]) == summary["best_epoch"], "Best epoch changed")
    rows = d5.parse_trial_rows(paths["oof"])
    require(
        all(row["trial_id"] == "D5_NO_ORTH" and row["arm"] == "B1" for row in rows),
        "Fold OOF trial/arm ownership changed",
    )
    anchors = d5.expected_fold_rows(context, fold)
    require(len(rows) == len(anchors) == summary["oof_count"], "Fold OOF length changed")
    for row, anchor in zip(rows, anchors):
        require(row["subject_id"] == anchor["subject_id"], "Fold subject ID changed")
        require(row["feature_sha256"] == anchor["feature_sha256"], "Fold feature anchor changed")
        require(row["truth"] == anchor["truth"] and row["fold"] == fold, "Fold label/owner changed")
    require(d5.metrics_close(d5.metrics(rows), summary["best_metrics"]), "Fold OOF metrics changed")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint["lock"] == lock and checkpoint["orthogonality_rate"] == 0.0, "Best checkpoint lock changed")
    require(checkpoint["optimizer_steps"] == EPOCHS and checkpoint["scheduler_steps"] == EPOCHS, "Checkpoint steps changed")
    require(checkpoint["spec"] == formal_spec(), "Checkpoint spec changed")
    require(checkpoint["best_epoch"] == summary["best_epoch"], "Checkpoint best epoch changed")
    require(d5.metrics_close(checkpoint["best_metrics"], summary["best_metrics"]), "Checkpoint best metrics changed")
    model, _criterion, _optimizer, _scheduler, _audit = make_training_objects(context)
    require(
        set(checkpoint["best_model"]) == set(model.state_dict()),
        "Checkpoint model-state keys changed",
    )
    model.load_state_dict(checkpoint["best_model"], strict=True)
    _train_mask, test_mask, _ = d5.historical.fold_positions(context, fold)
    model.eval()
    with torch.no_grad():
        replay = d5.historical.infer(
            model, context["dataset_data"]["Feature"],
        )[0][test_mask]
    replay_probability = torch.softmax(replay, dim=-1).detach().cpu().numpy().astype(np.float64)
    saved_probability = d5.arrays_from_rows(rows)[2]
    require(float(np.max(np.abs(replay_probability - saved_probability))) <= 1e-6, "Checkpoint does not reproduce OOF")
    require(np.array_equal(replay_probability.argmax(1), np.asarray([row["prediction"] for row in rows])), "Checkpoint predictions changed")
    del model
    torch.cuda.empty_cache()
    return summary, rows


def exact_mcnemar_p(repairs: int, damages: int) -> float:
    if repairs + damages == 0:
        return 1.0
    return float(
        d5.binomtest(
            min(repairs, damages), repairs + damages, 0.5,
            alternative="two-sided",
        ).pvalue
    )


def paired_comparison(
    result_rows: list[dict[str, Any]],
    result_metrics: dict[str, Any],
) -> dict[str, Any]:
    old = {row["subject_id"]: row for row in reference_rows()}
    new = {row["subject_id"]: row for row in result_rows}
    require(set(old) == set(new), "Paired subject IDs changed")
    repairs = damages = changed = 0
    repairs_by_truth = {"SMCI": 0, "PMCI": 0}
    damages_by_truth = {"SMCI": 0, "PMCI": 0}
    transitions = {"SMCI->PMCI": 0, "PMCI->SMCI": 0}
    for subject_id in sorted(old):
        left, right = old[subject_id], new[subject_id]
        require(
            left["truth"] == right["truth"]
            and left["fold"] == right["fold"]
            and left["feature_sha256"] == right["feature_sha256"],
            "Paired identity changed",
        )
        left_correct = left["prediction"] == left["truth"]
        right_correct = right["prediction"] == right["truth"]
        class_name = ["SMCI", "PMCI"][left["truth"]]
        if not left_correct and right_correct:
            repairs += 1
            repairs_by_truth[class_name] += 1
        if left_correct and not right_correct:
            damages += 1
            damages_by_truth[class_name] += 1
        if left["prediction"] != right["prediction"]:
            changed += 1
            transitions[
                f"{['SMCI', 'PMCI'][left['prediction']]}->{['SMCI', 'PMCI'][right['prediction']]}"
            ] += 1
    old_metrics = d5.metrics(list(old.values()))
    return {
        "reference": "D5",
        "repairs": repairs,
        "damages": damages,
        "changed": changed,
        "net_correct": repairs - damages,
        "correct_delta": int(result_metrics["correct"] - old_metrics["correct"]),
        "metric_deltas": {
            key: float(result_metrics[key] - old_metrics[key])
            for key in (
                "acc", "roc_auc", "pr_auc", "macro_f1", "bacc",
                "weighted_f1", "sen", "spe",
            )
        },
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(repairs, damages),
        "repairs_by_truth": repairs_by_truth,
        "damages_by_truth": damages_by_truth,
        "prediction_transitions": transitions,
    }


def decision(
    metrics: dict[str, Any],
    paired: dict[str, Any],
    formal_orth_max: float,
) -> tuple[str, dict[str, bool], list[str]]:
    no_collapse = (
        metrics["predicted_counts"]["PMCI"] >= 6
        and metrics["predicted_counts"]["SMCI"] >= 6
    )
    safety = {
        "correct_at_least_520": metrics["correct"] >= 520,
        "repairs_gt_damages": paired["repairs"] > paired["damages"],
        "bacc_at_least_0_8898571": metrics["bacc"] >= 0.8898571,
        "roc_auc_at_least_0_9132766": metrics["roc_auc"] >= 0.9132766,
        "pr_auc_at_least_0_7221648": metrics["pr_auc"] >= 0.7221648,
        "pmci_tp_at_least_36": metrics["tp"] >= 36,
        "pmci_sen_decline_at_most_0_01": metrics["sen"] >= 0.79,
        "smci_spe_decline_at_most_0_01": metrics["spe"] >= 0.9757142857142858,
        "no_class_collapse": no_collapse,
        "formal_orthogonality_contribution_exact_zero": formal_orth_max == 0.0,
    }
    neutral = {
        **{key: value for key, value in safety.items() if key not in ("correct_at_least_520", "repairs_gt_damages")},
        "correct_exactly_519": metrics["correct"] == 519,
        "repairs_equal_damages_or_predictions_identical": (
            paired["repairs"] == paired["damages"] or paired["changed"] == 0
        ),
    }
    if all(safety.values()):
        value = "NO_ORTH_POSITIVE"
        reasons = ["All registered improvement and safety gates passed"]
    elif all(neutral.values()):
        value = "NO_ORTH_NEUTRAL"
        reasons = ["D5 Correct tied, paired changes were neutral, and all safety gates passed"]
    else:
        value = "NO_ORTH_NO_GAIN"
        reasons = [name for name, passed in safety.items() if not passed]
        if not reasons:
            reasons = ["Neutral rule was not satisfied"]
    return value, safety, reasons


def aggregate(
    context: dict[str, Any],
    summaries: list[dict[str, Any]],
    fold_rows: list[list[dict[str, Any]]],
    runtime: dict[str, Any],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    require(len(summaries) == len(fold_rows) == 10, "Formal fold count changed")
    rows = sorted(
        [row for group in fold_rows for row in group],
        key=lambda row: row["original_csv_index"],
    )
    d5.validate_oof(rows, context, "B1")
    metrics = d5.metrics(rows)
    fold_metrics = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            "correct": summary["best_metrics"]["correct"],
            "acc": summary["best_metrics"]["acc"],
            "roc_auc": summary["best_metrics"]["roc_auc"],
            "pr_auc": summary["best_metrics"]["pr_auc"],
            "macro_f1": summary["best_metrics"]["macro_f1"],
            "bacc": summary["best_metrics"]["bacc"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "inference_seconds": summary["inference_seconds_full_graph"],
            "resumed_from_epoch": summary["resumed_from_epoch"],
        }
        for summary in summaries
    ]
    oof_path = RESULT_DIR / "tadpole_smci_pmci_no_orth_oof_predictions.csv"
    atomic_csv(oof_path, rows)
    paired = paired_comparison(rows, metrics)
    formal_orth_max = max(
        summary["formal_orthogonality_contribution_max_abs"]
        for summary in summaries
    )
    decision_value, gates, reasons = decision(metrics, paired, formal_orth_max)
    loss_diagnostics = {
        "legacy_fold0_detached_diagnostic": smoke["legacy_diagnostic"],
        "diagnostic_entered_optimizer_step": False,
        "formal_raw_orthogonality_evaluations": sum(
            summary["formal_raw_orthogonality_evaluations"]
            for summary in summaries
        ),
        "formal_orthogonality_contribution_max_abs": formal_orth_max,
        "per_fold_first_best_last": {
            str(summary["fold"]): summary["loss_components"]
            for summary in summaries
        },
        "nan_or_gradient_abnormality": False,
    }
    checkpoint_manifest = {
        "experiment": EXPERIMENT_ID,
        "source_commit": runtime["source_commit"],
        "orthogonality_rate": 0.0,
        "checkpoints": [
            {
                "fold": summary["fold"],
                "path": fold_paths(summary["fold"])["checkpoint"].relative_to(ROOT).as_posix(),
                "sha256": summary["checkpoint_sha256"],
                "best_epoch": summary["best_epoch"],
                "lock_sha256": summary["lock"]["sha256"],
            }
            for summary in summaries
        ],
    }
    result = {
        "experiment": EXPERIMENT_ID,
        "arm": "D5_NO_ORTH",
        "decision": decision_value,
        "decision_gates": gates,
        "decision_reasons": reasons,
        "metrics": metrics,
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)),
        "fold_acc_sample_sd": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_roc_auc_mean": float(statistics.mean(row["roc_auc"] for row in fold_metrics)),
        "fold_roc_auc_sample_sd": float(statistics.stdev(row["roc_auc"] for row in fold_metrics)),
        "fold_metrics": fold_metrics,
        "paired_D5": paired,
        "parameter_count": summaries[0]["object_audit"]["total_parameter_count"],
        "trainable_parameter_count": summaries[0]["object_audit"]["trainable_parameter_count"],
        "training_time_seconds": float(sum(row["elapsed_seconds"] for row in fold_metrics)),
        "inference_time_seconds": float(sum(row["inference_seconds"] for row in fold_metrics)),
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
        "loss_diagnostics": loss_diagnostics,
        "initialization_fairness": smoke["initialization_fairness"],
        "effective_gate": summaries[0]["effective_gate"],
        "effective_noise_std": summaries[0]["effective_noise_std"],
        "prior_free_modality_prior": False,
        "targets": {
            "correct_at_least_520": metrics["correct"] >= 520,
            "correct_at_least_523": metrics["correct"] >= 523,
            "fold_mean_acc_above_0_9757": float(statistics.mean(row["acc"] for row in fold_metrics)) > 0.9757,
            "fold_mean_roc_auc_above_0_9049": float(statistics.mean(row["roc_auc"] for row in fold_metrics)) > 0.9049,
        },
        "oof_path": oof_path.relative_to(ROOT).as_posix(),
        "oof_sha256": file_sha256(oof_path),
        "source_commit": runtime["source_commit"],
        "result_commit": "reported_in_final_git_handoff",
        "branch": BRANCH,
        "device": runtime["device"],
        "cuda_device_name": torch.cuda.get_device_name(torch.device(runtime["device"])),
    }
    atomic_json(RESULT_DIR / "fold_results.json", {"folds": fold_metrics})
    atomic_json(RESULT_DIR / "comparisons.json", paired)
    atomic_json(RESULT_DIR / "loss_diagnostics.json", loss_diagnostics)
    atomic_json(RESULT_DIR / "checkpoint_manifest.json", checkpoint_manifest)
    report = render_report(result)
    d5.atomic_write_text(RESULT_DIR / "REPORT.md", report)
    result["artifacts"] = {
        "report_sha256": file_sha256(RESULT_DIR / "REPORT.md"),
        "fold_results_sha256": file_sha256(RESULT_DIR / "fold_results.json"),
        "comparisons_sha256": file_sha256(RESULT_DIR / "comparisons.json"),
        "loss_diagnostics_sha256": file_sha256(RESULT_DIR / "loss_diagnostics.json"),
        "checkpoint_manifest_sha256": file_sha256(RESULT_DIR / "checkpoint_manifest.json"),
    }
    summary = {**result, "sha256": payload_sha256(result)}
    atomic_json(RESULT_DIR / "summary.json", summary)
    return summary


def render_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    paired = summary["paired_D5"]
    diag = summary["loss_diagnostics"]["legacy_fold0_detached_diagnostic"]
    loss_component_report = {
        fold: {
            phase: {
                "main_ce": values["loss_main_ce"],
                "ovr_0": values["loss_ovr_0"],
                "ovr_1": values["loss_ovr_1"],
            }
            for phase, values in phases.items()
        }
        for fold, phases in summary["loss_diagnostics"]["per_fold_first_best_last"].items()
    }
    conclusions = {
        "NO_ORTH_POSITIVE": "The legacy cross-subject orthogonality loss harmed TADPOLE binary performance; removal produced a hard-classification gain, so retain NO_ORTH as the new reference.",
        "NO_ORTH_NEUTRAL": "The legacy orthogonality loss can be removed without a performance cost; retain the simpler, semantically cleaner NO_ORTH version as the clean baseline.",
        "NO_ORTH_NO_GAIN": "Complete removal of the legacy orthogonality loss did not improve D5; retain D5 as the performance reference and do not automatically search the orth rate or run a corrected per-subject version.",
    }
    lines = [
        "# TADPOLE Binary Orthogonality Loss Removal v1",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Branch: {BRANCH}",
        f"- Source commit: {summary['source_commit']}",
        "- Result commit: reported in the final Git handoff.",
        f"- Device: {summary['device']} ({summary['cuda_device_name']})",
        "- Protocol: D5 except orthogonality_rate=0.0; seed 0, ten folds, 400 epochs, full-batch transductive, single model.",
        "- Loss: main weighted CE + two unnormalized weighted OVR CE terms. Raw orthogonality is diagnostic-only and never enters an optimizer step.",
        "- Confusion order: [[TN, FP], [FN, TP]], pMCI positive.",
        "",
        "## Pooled OOF result",
        "",
        "| Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | pMCI SEN | sMCI SPE | Confusion | Predicted sMCI/pMCI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        f"| {metrics['correct']}/535 | {metrics['acc']:.7f} | {metrics['roc_auc']:.7f} | {metrics['pr_auc']:.7f} | {metrics['macro_f1']:.7f} | {metrics['bacc']:.7f} | {metrics['weighted_f1']:.7f} | {metrics['sen']:.7f} | {metrics['spe']:.7f} | {metrics['confusion_matrix']} | {metrics['predicted_counts']['SMCI']}/{metrics['predicted_counts']['PMCI']} |",
        "",
        f"- TP/FN={metrics['tp']}/{metrics['fn']}; TN/FP={metrics['tn']}/{metrics['fp']}.",
        f"- Fold ACC mean +/- sample SD: {summary['fold_acc_mean']:.7f} +/- {summary['fold_acc_sample_sd']:.7f}.",
        f"- Fold ROC-AUC mean +/- sample SD: {summary['fold_roc_auc_mean']:.7f} +/- {summary['fold_roc_auc_sample_sd']:.7f}.",
        f"- Parameters total/trainable: {summary['parameter_count']}/{summary['trainable_parameter_count']}.",
        f"- Training/inference seconds: {summary['training_time_seconds']:.2f}/{summary['inference_time_seconds']:.4f}.",
        "",
        "## Folds",
        "",
        "| Fold | Best epoch | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in summary["fold_metrics"]:
        lines.append(
            f"| {fold['fold']} | {fold['best_epoch']} | {fold['correct']} | {fold['acc']:.7f} | {fold['roc_auc']:.7f} | {fold['pr_auc']:.7f} | {fold['macro_f1']:.7f} | {fold['bacc']:.7f} |"
        )
    lines.extend([
        "",
        "## Paired with D5",
        "",
        f"- Repairs/damages/changed: {paired['repairs']}/{paired['damages']}/{paired['changed']}; Correct delta={paired['correct_delta']}.",
        f"- Metric deltas: {json.dumps(paired['metric_deltas'], sort_keys=True)}.",
        f"- Exact two-sided McNemar p: {paired['exact_two_sided_mcnemar_p']:.7g}.",
        f"- Repair sources: {json.dumps(paired['repairs_by_truth'], sort_keys=True)}.",
        f"- Damage sources: {json.dumps(paired['damages_by_truth'], sort_keys=True)}.",
        f"- Prediction transitions: {json.dumps(paired['prediction_transitions'], sort_keys=True)}.",
        "",
        "## Loss diagnostic",
        "",
        f"- Old representation shapes: {diag['representation_shapes']}; similarity matrix: {diag['similarity_matrix_shape']}.",
        f"- Raw/weighted old orthogonality: {diag['raw_orthogonality']:.7f}/{diag['weighted_orthogonality']:.7f}.",
        f"- Weighted orth/supervised loss ratio: {diag['weighted_orth_over_supervised_loss']:.7f}.",
        f"- Weighted orth/supervised gradient-norm ratio: {diag['weighted_orth_over_supervised_gradient_norm']:.7f}.",
        f"- Formal no-orth maximum absolute contribution: {summary['loss_diagnostics']['formal_orthogonality_contribution_max_abs']}.",
        f"- Per-fold first/best/last main CE, OVR0 and OVR1: `{json.dumps(loss_component_report, sort_keys=True)}`.",
        "- NaN or gradient abnormality: false.",
        f"- Descriptive targets: >=520={summary['targets']['correct_at_least_520']}; >=523={summary['targets']['correct_at_least_523']}; fold mean ACC>97.57%={summary['targets']['fold_mean_acc_above_0_9757']}; fold mean ROC-AUC>90.49%={summary['targets']['fold_mean_roc_auc_above_0_9049']}.",
        "",
        "## Required answers",
        "",
        "1. The old orthogonality produces an N by N matrix: **yes**, 535 by 535.",
        "2. It contains cross-subject terms: **yes**, including off-diagonal subject pairs.",
        "3. It contains test rows: **yes**; the raw term is computed from full-dataset embeddings and receives no train mask.",
        f"4. Old weighted orth/supervised scale: loss ratio={diag['weighted_orth_over_supervised_loss']:.7f}, gradient-norm ratio={diag['weighted_orth_over_supervised_gradient_norm']:.7f}.",
        f"5. Parameters and forward at step 0 are identical to D5: **yes**; max logit difference={summary['initialization_fairness']['step0_logits_max_abs_diff']}.",
        f"6. NO_ORTH Correct: **{metrics['correct']}/535**.",
        f"7. It exceeds D5 519/535: **{metrics['correct'] > 519}**.",
        f"8. Repairs exceed damages: **{paired['repairs'] > paired['damages']}** ({paired['repairs']}/{paired['damages']}).",
        f"9. pMCI TP/SEN/PR-AUC: {metrics['tp']}/{metrics['sen']:.7f}/{metrics['pr_auc']:.7f}; registered preservation gates={summary['decision_gates']['pmci_tp_at_least_36']}/{summary['decision_gates']['pmci_sen_decline_at_most_0_01']}/{summary['decision_gates']['pr_auc_at_least_0_7221648']}.",
        f"10. It reaches 523/535: **{metrics['correct'] >= 523}**.",
        f"11. Retain NO_ORTH as the new reference: **{summary['decision'] in ('NO_ORTH_POSITIVE', 'NO_ORTH_NEUTRAL')}**.",
        f"12. Allow the next pre-shared source residual experiment: **{summary['decision'] in ('NO_ORTH_POSITIVE', 'NO_ORTH_NEUTRAL')}**; it is not executed here.",
        f"13. Final Decision: **{summary['decision']}**.",
        f"14. Source={summary['source_commit']}; result commit is reported in Git handoff; branch={BRANCH}; device={summary['device']}; commands are listed below.",
        "",
        "## Decision reasons",
        "",
    ])
    lines.extend(f"- {reason}" for reason in summary["decision_reasons"])
    lines.extend([
        "",
        f"Decision conclusion: {conclusions[summary['decision']]}",
        "",
        "## Reproduction",
        "",
        f"Run these commands from a clean worktree whose same-named branch points exactly to source commit `{summary['source_commit']}`; the result commit is for reading artifacts and intentionally fails the source-scope gate.",
        "    python -u -B scripts/run_tad_binary_no_orth_v1.py inspect",
        "    python -u -B scripts/run_tad_binary_no_orth_v1.py smoke --device cuda:0",
        "    python -u -B scripts/run_tad_binary_no_orth_v1.py formal --device cuda:0",
        "",
        "No orth-rate search, corrected orthogonality, second version, other task, or v1.1 was run.",
    ])
    return "\n".join(lines) + "\n"


def run_formal(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    source = source_gate()
    runtime = runtime_lock(source, device_text)
    smoke = validate_smoke(runtime)
    formal_config_core = {
        "runtime_lock": runtime,
        "smoke_report_sha256": file_sha256(SMOKE_DIR / "smoke_report.json"),
        "folds": list(FOLDS),
        "epochs": EPOCHS,
        "orthogonality_rate": 0.0,
        "loss_formula": "main_ce + (ovr_0 + ovr_1)",
        "single_arm": "D5_NO_ORTH",
    }
    formal_config = {
        **formal_config_core,
        "sha256": payload_sha256(formal_config_core),
    }
    formal_config_path = RESULT_DIR / "formal_config.json"
    if formal_config_path.is_file():
        require(d5.read_json(formal_config_path) == formal_config, "Existing formal config changed")
    else:
        atomic_json(formal_config_path, formal_config)
    context = build_context(torch.device(device_text))
    summaries: list[dict[str, Any]] = []
    rows_by_fold: list[list[dict[str, Any]]] = []
    for fold in FOLDS:
        summary, rows = train_fold(context, fold, runtime, smoke)
        summaries.append(summary)
        rows_by_fold.append(rows)
    result = aggregate(context, summaries, rows_by_fold, runtime, smoke)
    print(json.dumps({
        "formal": "COMPLETE",
        "decision": result["decision"],
        "correct": result["metrics"]["correct"],
        "bacc": result["metrics"]["bacc"],
        "roc_auc": result["metrics"]["roc_auc"],
        "pr_auc": result["metrics"]["pr_auc"],
        "summary_sha256": result["sha256"],
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect", help="Audit D5 reference and old orthogonality implementation")
    smoke = subparsers.add_parser("smoke", help="Run fold-0 three-epoch CUDA smoke")
    smoke.add_argument("--device", default="cuda:0")
    formal = subparsers.add_parser("formal", help="Run or strictly resume ten formal no-orth folds")
    formal.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        run_inspect()
    elif args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode == "formal":
        run_formal(args.device)
    else:
        raise InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
