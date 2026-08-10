"""Registered TADPOLE SMCI/PMCI hyperparameter search with historical protocol."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2  # noqa: E402
from Loss.loss_fn import orthogonality_lossv2  # noqa: E402
from Model import HeterGraph_Model_Kmeans  # noqa: E402
from Model.cme_dual_branch import CMEDualBranchModel  # noqa: E402
from Utils import CustomCosineAnnealingLR, SET_Random  # noqa: E402
from scripts import run_cross_dataset_a012_structure_v1 as historical  # noqa: E402


EXPERIMENT_ID = "tad_binary_hparam_search_v1"
BRANCH = "experiment/tad-binary-hparam-search-v1"
BASE_COMMIT = "17004d29714471b5e1548e04bac0072e854e7e18"
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
FOLDS = tuple(range(10))
EPOCHS = 400
SEED = 0
SMOKE_EPOCHS = 3

HISTORICAL_PATHS = {
    "B0": "experiments/cross_dataset_a012_structure_v1/tadpole_smci_pmci_B0_oof_predictions.csv",
    "B1": "experiments/cross_dataset_a012_structure_v1/tadpole_smci_pmci_B1_oof_predictions.csv",
    "fold_metrics": "experiments/cross_dataset_a012_structure_v1/tadpole_smci_pmci_fold_metrics.json",
    "summary": "experiments/cross_dataset_a012_structure_v1/summary.json",
}
HISTORICAL_ANCHORS = {
    "B0": HISTORICAL_DIR / "historical_B0_oof.csv",
    "B1": HISTORICAL_DIR / "historical_B1_oof.csv",
    "fold_metrics": HISTORICAL_DIR / "historical_fold_metrics.json",
}
REQUIRED_TRACKED = (
    ".gitignore",
    "scripts/run_tad_binary_hparam_search_v1.py",
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
    "experiments/cross_dataset_a012_structure_v1/protocols/tadpole_smci_pmci.json",
)


class InvariantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"Refusing empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def git_blob(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout


def current_commit() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def capture_rng() -> dict[str, Any]:
    return historical.capture_rng_state()


def restore_rng(value: dict[str, Any]) -> None:
    historical.restore_rng_state(value)


def protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    require(value["dataset"] == "TADPOLE" and value["task"] == "SMCI_PMCI", "Task protocol changed")
    require(value["positive_class"] == "PMCI" and value["positive_index"] == 1, "Positive class changed")
    require(value["sample_count"] == 535 and value["class_counts"] == {"SMCI": 490, "PMCI": 45}, "Dataset counts changed")
    require(len(value["modalities"]) == 5, "Modality count changed")
    return value


def build_context(device: torch.device) -> dict[str, Any]:
    return historical.build_context(protocol(), device)


def typed_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append({
            "task_id": row["task_id"],
            "arm": row["arm"],
            "fold": int(row["fold"]),
            "subject_id": row["subject_id"],
            "original_csv_index": int(row["original_csv_index"]),
            "feature_sha256": row["feature_sha256"],
            "truth": int(row["truth"]),
            "prediction": int(row["prediction"]),
            "logit_0": float(row["logit_0"]),
            "logit_1": float(row["logit_1"]),
            "probability_0": float(row["probability_0"]),
            "probability_1": float(row["probability_1"]),
            "positive_probability": float(row["positive_probability"]),
        })
    return output


def arrays_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    logits = np.asarray([[float(row["logit_0"]), float(row["logit_1"])] for row in rows], dtype=np.float64)
    probabilities = np.asarray([[float(row["probability_0"]), float(row["probability_1"])] for row in rows], dtype=np.float64)
    return truth, logits, probabilities


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth, _, probabilities = arrays_from_rows(rows)
    return historical.binary_metrics(truth, probabilities, protocol())


def metrics_close(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-9) -> bool:
    for key in ("n", "correct", "tp", "fn", "tn", "fp", "confusion_matrix", "predicted_counts"):
        if left[key] != right[key]:
            return False
    for key in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe"):
        if not math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=tolerance):
            return False
    return True


def expected_subjects(context: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [{**row, "fold": int(fold["fold"])} for fold in context["fold_manifest"]["folds"] for row in fold["test_rows"]],
        key=lambda row: int(row["original_csv_index"]),
    )


def validate_oof(rows: list[dict[str, Any]], context: dict[str, Any], arm: str | None = None) -> None:
    ordered = sorted(rows, key=lambda row: int(row["original_csv_index"]))
    anchors = expected_subjects(context)
    require(len(ordered) == len(anchors) == 535, "OOF count changed")
    require(len({row["subject_id"] for row in ordered}) == 535, "OOF subject IDs are not unique")
    for row, anchor in zip(ordered, anchors):
        require(row["subject_id"] == anchor["subject_id"], "OOF subject alignment changed")
        require(row["feature_sha256"] == anchor["feature_sha256"], "OOF feature anchor changed")
        require(row["truth"] == anchor["truth"], "OOF truth changed")
        require(row["fold"] == anchor["fold"], "OOF fold changed")
        require(row["prediction"] == int(np.argmax([row["probability_0"], row["probability_1"]])), "OOF prediction is not argmax")
        if arm is not None:
            require(row["arm"] == arm, "OOF arm changed")
    probability = np.asarray([[row["probability_0"], row["probability_1"]] for row in ordered], dtype=float)
    require(np.isfinite(probability).all() and float(np.max(np.abs(probability.sum(1) - 1.0))) <= 2e-6, "OOF probabilities invalid")


def historical_payload(write_anchors: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    context = build_context(torch.device("cpu"))
    blobs = {key: git_blob(HISTORICAL_RESULT_COMMIT, relative) for key, relative in HISTORICAL_PATHS.items()}
    if write_anchors:
        for key, destination in HISTORICAL_ANCHORS.items():
            atomic_write_bytes(destination, blobs[key])
    else:
        for key, destination in HISTORICAL_ANCHORS.items():
            require(destination.is_file() and destination.read_bytes() == blobs[key], f"Historical anchor drifted: {key}")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ("B0", "B1"):
        rows = typed_rows(list(csv.DictReader(io.StringIO(blobs[arm].decode("utf-8")))))
        validate_oof(rows, context, arm)
        rows_by_arm[arm] = rows
    fold_metrics = json.loads(blobs["fold_metrics"].decode("utf-8"))
    summary = json.loads(blobs["summary"].decode("utf-8"))
    require(summary["runtime_lock"]["source_commit"] == BASE_COMMIT, "Historical source commit changed")
    require(summary["runtime_lock"]["seed"] == 0 and summary["runtime_lock"]["epochs"] == 400, "Historical seed/epochs changed")
    require(summary["runtime_lock"]["adapter_rank"] == 8 and summary["runtime_lock"]["adapter_lr_multiplier"] == 2.0, "Historical B1 config changed")
    expected_metrics = {
        "B0": {"correct": 513, "acc": 0.9588785046728971, "roc_auc": 0.922358276643991, "pr_auc": 0.6378392083225689, "macro_f1": 0.860914161467196, "bacc": 0.8463718820861679, "sen": 0.7111111111111111},
        "B1": {"correct": 515, "acc": 0.9626168224299065, "roc_auc": 0.8794557823129252, "pr_auc": 0.6729808573914714, "macro_f1": 0.8786848072562359, "bacc": 0.8786848072562359, "sen": 0.7777777777777778},
    }
    computed: dict[str, dict[str, Any]] = {}
    for arm in ("B0", "B1"):
        computed[arm] = metrics(rows_by_arm[arm])
        for key, expected in expected_metrics[arm].items():
            require(math.isclose(float(computed[arm][key]), float(expected), rel_tol=0.0, abs_tol=1e-9), f"Historical {arm} metric changed: {key}")
        require(len(fold_metrics[arm]) == 10 and [int(row["fold"]) for row in fold_metrics[arm]] == list(FOLDS), f"Historical {arm} folds changed")

    weights = context["dataset_dict"]["Label_Weight"].detach().cpu().numpy().astype(float)
    expected_weights = np.asarray([(535 - 490) / 535, (535 - 45) / 535], dtype=float)
    require(float(np.max(np.abs(weights - expected_weights))) <= 1e-7, "Global full-dataset class weights changed")
    fold_core = {"task_id": context["task_id"], "fold_manifest": context["fold_manifest"]}
    inspect_core = {
        "experiment": EXPERIMENT_ID,
        "base_commit": BASE_COMMIT,
        "historical_result_commit": HISTORICAL_RESULT_COMMIT,
        "dataset": "TADPOLE",
        "task": "SMCI_PMCI",
        "sample_count": 535,
        "class_names": ["SMCI", "PMCI"],
        "positive_class": "PMCI",
        "positive_index": 1,
        "modalities": 5,
        "folds": list(FOLDS),
        "seed": 0,
        "epochs": 400,
        "class_weight_scope": "global_full_dataset_historical",
        "class_weights": weights.tolist(),
        "criterion": "criterion_lossv2: main weighted CE + two weighted OVR CE sum + 0.0001 orthogonality",
        "best_epoch_rule": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "test_fold_best_epoch_behavior": True,
        "historical_reuse": {
            "eligible": True,
            "reason": "The registered seed-0 cross-dataset run used this exact source, task, folds, global class weights, loss, and best-epoch protocol and provides unique per-subject OOF probabilities plus fold epochs.",
            "anchor_sha256": {key: bytes_sha256(blobs[key]) for key in HISTORICAL_ANCHORS},
            "metrics": computed,
            "fold_metrics": fold_metrics,
            "parameter_count": {
                "B0": int(summary["task_results"]["tadpole_smci_pmci"]["B0"]["parameter_count"]),
                "B1": int(summary["task_results"]["tadpole_smci_pmci"]["B1"]["parameter_count"]),
            },
            "mechanism_B1": summary["mechanism_diagnostics"]["tadpole_smci_pmci"],
        },
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "dataset_sha256": protocol()["data_sha256"],
        "modality_sha256": protocol()["modality_sha256"],
        "historical_runner_sha256": file_sha256(ROOT / "scripts" / "run_cross_dataset_a012_structure_v1.py"),
        "historical_loss_sha256": file_sha256(ROOT / "Loss" / "loss_fn.py"),
    }
    del context
    return ({**inspect_core, "sha256": payload_sha256(inspect_core)}, {**fold_core, "sha256": payload_sha256(fold_core)})


def run_inspect() -> None:
    inspect, folds = historical_payload(write_anchors=True)
    atomic_write_json(INSPECT_PATH, inspect)
    atomic_write_json(FOLD_MANIFEST_PATH, folds)
    print(json.dumps({"inspect": "PASS", "historical_reuse": True, "fold_sha256": folds["sha256"]}, indent=2))


def validate_inspect() -> None:
    require(INSPECT_PATH.is_file() and FOLD_MANIFEST_PATH.is_file(), "Run inspect first")
    expected_inspect, expected_folds = historical_payload(write_anchors=False)
    require(read_json(INSPECT_PATH) == expected_inspect, "Inspect manifest drifted")
    require(read_json(FOLD_MANIFEST_PATH) == expected_folds, "Fold manifest drifted")


def source_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in (*REQUIRED_TRACKED, *LOCKED_DEPENDENCIES)}


def source_gate() -> str:
    require(git("branch", "--show-current").stdout.strip() == BRANCH, "Wrong branch")
    require(git("diff", "--quiet", check=False).returncode == 0, "Tracked worktree drifted")
    require(git("diff", "--cached", "--quiet", check=False).returncode == 0, "Git index drifted")
    head = current_commit()
    require(head != BASE_COMMIT, "Implementation must be committed before execution")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, head, check=False).returncode == 0, "Base commit is not an ancestor")
    changed = {line.strip().replace("\\", "/") for line in git("diff", "--name-only", BASE_COMMIT, head).stdout.splitlines() if line.strip()}
    require(changed == set(REQUIRED_TRACKED), f"Source commit scope changed: {sorted(changed)}")
    for relative in REQUIRED_TRACKED:
        require(git("ls-files", "--error-unmatch", "--", relative, check=False).returncode == 0, f"Untracked source: {relative}")
    for relative in LOCKED_DEPENDENCIES:
        require(git("diff", "--quiet", BASE_COMMIT, "HEAD", "--", relative, check=False).returncode == 0, f"Historical dependency changed: {relative}")
    validate_inspect()
    return head


def runtime_lock(source_commit: str) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_ID,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "experiment_config_sha256": file_sha256(CONFIG_PATH),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "inspect_sha256": file_sha256(INSPECT_PATH),
        "fold_manifest_sha256": file_sha256(FOLD_MANIFEST_PATH),
        "historical_result_commit": HISTORICAL_RESULT_COMMIT,
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs": EPOCHS,
        "device": "cuda:0",
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
    }
    return {**core, "sha256": payload_sha256(core)}


def trial_spec(trial_id: str, stage: str, arm: str, lr: float, wd: float, dropout: float, rank: int = 0, multiplier: float = 0.0, ordinal: int = 0) -> dict[str, Any]:
    core = {
        "trial_id": trial_id,
        "stage": stage,
        "arm": arm,
        "lr": float(lr),
        "weight_decay": float(wd),
        "dropout": float(dropout),
        "rank": int(rank),
        "adapter_lr_multiplier": float(multiplier),
        "ordinal": int(ordinal),
        "epochs": EPOCHS,
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
    }
    require(arm in ("B0", "B1"), "Unknown arm")
    if arm == "B0":
        require(rank == 0 and multiplier == 0.0, "B0 cannot contain adapters")
    else:
        require(rank in (4, 8) and multiplier in (0.5, 1.0, 2.0), "Unregistered adapter config")
    return {**core, "config_sha256": payload_sha256(core)}


def config_key(spec: dict[str, Any]) -> str:
    return payload_sha256({key: spec[key] for key in ("arm", "lr", "weight_decay", "dropout", "rank", "adapter_lr_multiplier", "epochs", "folds", "seed_per_fold")})


def trial_context(base_context: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    value = dict(base_context)
    value["protocol"] = copy.deepcopy(base_context["protocol"])
    training = value["protocol"]["training"]
    training["lr"] = float(spec["lr"])
    training["weight_decay"] = float(spec["weight_decay"])
    training["drop_rate"] = float(spec["dropout"])
    return value


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def common_state_max_diff(left: torch.nn.Module, right: torch.nn.Module) -> float:
    lstate = left.state_dict()
    rstate = right.state_dict()
    require(set(lstate).issubset(set(rstate)), "B1 lost a common state tensor")
    values: list[float] = []
    for name, tensor in lstate.items():
        require(tensor.shape == rstate[name].shape and tensor.dtype == rstate[name].dtype, f"Common tensor schema changed: {name}")
        values.append(float((tensor.detach().cpu() - rstate[name].detach().cpu()).abs().max()))
    return max(values, default=0.0)


def build_model(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    kwargs = historical.model_kwargs(context)
    initialization: dict[str, Any] = {}
    SET_Random(SEED)
    if spec["arm"] == "B0":
        model = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
    else:
        reference = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
        post_common_rng = capture_rng()
        SET_Random(SEED)
        model = CMEDualBranchModel(
            **kwargs,
            cme_arm="c1",
            adapter_rank=int(spec["rank"]),
            router_hidden=16,
            modality_embedding_dim=8,
        ).to(context["device"])
        difference = common_state_max_diff(reference, model)
        if audit_common:
            require(difference == 0.0, "B0/B1 common initialization changed")
            initialization["common_state_max_abs_diff"] = difference
        else:
            initialization["common_state_pointwise_audit"] = "smoke_only"
        del reference
        restore_rng(post_common_rng)

    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary Query/OVR count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    if spec["arm"] == "B1":
        require(len(model.private_adapters) == 5, "Private adapter count changed")
        require(not hasattr(model, "private_alpha"), "Trainable private alpha is forbidden")
    initialization.update({
        "parameter_count": parameter_count(model),
        "query_count": len(model.label_pools),
        "ovr_head_count": len(model._Auxi_classifier),
        "adapter_count": len(model.private_adapters) if spec["arm"] == "B1" else 0,
    })
    return model, initialization


def make_training_objects(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    model, initialization = build_model(context, spec, audit_common=audit_common)
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(
        context["dataset_dict"], context["device"],
        rate=float(training["loss_rate"]),
        label_smoothing=float(training["label_smoothing"]),
    )
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if spec["arm"] == "B0":
        optimizer = torch.optim.Adam(
            [parameter for _, parameter in named], lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
        )
        scheduler = CustomCosineAnnealingLR(
            optimizer, T_max=EPOCHS,
            eta_min=float(training["scheduler_eta_min"]),
        )
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
        require(len(adapter_named) == 20, "Five private adapters must expose exactly 20 tensors")
        require(not (base_ids & adapter_ids) and base_ids | adapter_ids == all_ids, "Optimizer groups are not exhaustive/disjoint")
        multiplier = float(spec["adapter_lr_multiplier"])
        optimizer = torch.optim.Adam([
            {"params": [parameter for _, parameter in base_named], "lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"]), "group_name": "base"},
            {"params": [parameter for _, parameter in adapter_named], "lr": float(spec["lr"]) * multiplier, "weight_decay": float(spec["weight_decay"]), "group_name": "private_adapters"},
        ])
        scheduler = historical.RatioPreservingCustomCosineAnnealingLR(
            optimizer, T_max=EPOCHS,
            eta_min=float(training["scheduler_eta_min"]), multiplier=multiplier,
        )
        group_audit = {
            "base_parameter_tensors": len(base_named),
            "adapter_parameter_tensors": len(adapter_named),
            "adapter_parameter_names": [name for name, _ in adapter_named],
            "base_lr": float(spec["lr"]),
            "adapter_lr": float(spec["lr"]) * multiplier,
            "weight_decay": float(spec["weight_decay"]),
        }
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler, {**initialization, **group_audit}


def manual_historical_loss(criterion: Any, output: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, embeddings: Any, auxiliary: Any) -> tuple[torch.Tensor, dict[str, float]]:
    main = criterion.CE_loss(output[mask], labels[mask])
    one_hot = torch.nn.functional.one_hot(labels, num_classes=2).transpose(0, 1).reshape(2, -1)
    aux_terms = [criterion.aux_loss_dict[f"aux_loss_{index}"](auxiliary[index][mask], one_hot[index][mask]) for index in range(2)]
    orth = orthogonality_lossv2(embeddings)
    # Match criterion_lossv2's exact floating-point association: it first
    # accumulates the two OVR terms, then adds that sum to the main CE.
    auxiliary_sum = aux_terms[0] + aux_terms[1]
    total = main + auxiliary_sum + criterion.rate * orth
    return total, {
        "main": float(main.detach().cpu()),
        "ovr_0": float(aux_terms[0].detach().cpu()),
        "ovr_1": float(aux_terms[1].detach().cpu()),
        "orthogonality": float(orth.detach().cpu()),
        "orthogonality_rate": float(criterion.rate),
    }


def checkpoint_state_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return (not value.is_floating_point()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(checkpoint_state_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(checkpoint_state_finite(item) for item in value)
    return True


def smoke_runtime_config(runtime: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    core = {"runtime_lock": runtime, "spec": spec, "fold": 0, "epochs": SMOKE_EPOCHS}
    return {**core, "sha256": payload_sha256(core)}


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = source_gate()
    runtime = runtime_lock(source)
    base = build_context(torch.device(device_text))
    spec = trial_spec("SMOKE_D1", "smoke", "B1", 0.01, 0.0005, 0.67, rank=4, multiplier=0.5)
    context = trial_context(base, spec)
    smoke_config = smoke_runtime_config(runtime, spec)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SMOKE_DIR / "smoke_config.json", smoke_config)

    b0_spec = trial_spec("SMOKE_B0", "smoke", "B0", 0.01, 0.0005, 0.67)
    b0_context = trial_context(base, b0_spec)
    b0_model, _ = build_model(b0_context, b0_spec)
    model, criterion, optimizer, scheduler, audit = make_training_objects(context, spec, audit_common=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = historical.fold_positions(context, 0)
    require(not bool(torch.any(train_mask & test_mask)), "Smoke fold masks overlap")
    b0_model.eval()
    model.eval()
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
    for _epoch in range(1, SMOKE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, embeddings, auxiliary = model(features)
        require(tuple(output.shape) == (535, 2) and len(auxiliary) == 2, "Smoke output shape changed")
        loss = criterion(output, labels, train_mask, embeddings, auxiliary)
        manual, components = manual_historical_loss(criterion, output, labels, train_mask, embeddings, auxiliary)
        formula_errors.append(float((loss - manual).abs().detach().cpu()))
        require(bool(torch.isfinite(loss)) and formula_errors[-1] <= 1e-7, "Historical loss formula changed")
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("private_adapters."):
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid smoke adapter gradient: {name}")
                gradient_max[name] = max(gradient_max[name], float(parameter.grad.detach().abs().max().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["protocol"]["training"]["grad_clip"]))
        optimizer.step()
        scheduler.step()
        scheduler.assert_ratio()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            probability = torch.softmax(model(features)[0], dim=-1)
        probability_errors.append(float((probability.sum(dim=1) - 1.0).abs().max().cpu()))
    require(all(math.isfinite(value) for value in losses), "Smoke loss is non-finite")
    require(all(value > 0.0 and math.isfinite(value) for value in gradient_max.values()), "Every adapter tensor must receive a finite nonzero gradient within three epochs")
    parameter_delta = {name: float((parameter.detach().cpu() - initial_adapter[name]).abs().max()) for name, parameter in model.named_parameters() if name in initial_adapter}
    require(all(value > 0.0 and math.isfinite(value) for value in parameter_delta.values()), "Every adapter tensor must update within three epochs")
    require(max(probability_errors) <= 2e-6, "Smoke probability simplex changed")

    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    checkpoint = {
        "runtime_lock": runtime, "smoke_config": smoke_config, "spec": spec,
        "epoch": SMOKE_EPOCHS, "model": clone_cpu_state(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": capture_rng(),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    restored_model, restored_criterion, restored_optimizer, restored_scheduler, _ = make_training_objects(context, spec)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(restored["runtime_lock"] == runtime and restored["smoke_config"] == smoke_config, "Smoke checkpoint lock changed")
    restored_model.load_state_dict(restored["model"], strict=True)
    restored_optimizer.load_state_dict(restored["optimizer"])
    restored_scheduler.load_state_dict(restored["scheduler"])
    restored_scheduler.assert_ratio()
    require(checkpoint_state_finite(restored), "Smoke checkpoint contains non-finite tensors")
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        left = model(features)[0]
        right = restored_model(features)[0]
    reload_diff = float((left - right).abs().max().cpu())
    require(reload_diff == 0.0, "Smoke checkpoint strict reload changed logits")
    three_class = historical.three_class_compatibility_regression(device_text)
    require(three_class["parameter_count"] == 862971 and three_class["query_count"] == 3 and three_class["ovr_head_count"] == 3, "Three-class A012 compatibility changed")

    report_core = {
        "runtime_lock": runtime,
        "smoke_config_sha256": file_sha256(SMOKE_DIR / "smoke_config.json"),
        "spec": spec,
        "fold": 0,
        "epochs": SMOKE_EPOCHS,
        "losses": losses,
        "historical_loss_components_last_epoch": components,
        "historical_loss_formula_max_abs_error": max(formula_errors),
        "probability_sum_max_abs_error": max(probability_errors),
        "initial_b0_b1_logits_max_abs_diff": initial_logits_diff,
        "adapter_gradient_max_by_tensor": gradient_max,
        "adapter_parameter_delta_by_tensor": parameter_delta,
        "object_audit": audit,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_reload_logits_max_abs_diff": reload_diff,
        "test_labels_unchanged": True,
        "three_class_compatibility": three_class,
        "device": torch.cuda.get_device_name(torch.device(device_text)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    report = {**report_core, "sha256": payload_sha256(report_core)}
    atomic_write_json(SMOKE_DIR / "smoke_report.json", report)
    print(json.dumps({"smoke": "PASS", "losses": losses, "params": audit["parameter_count"]}, indent=2))
    del restored_criterion, restored_model, model, base, context
    torch.cuda.empty_cache()


def validate_smoke(runtime: dict[str, Any]) -> dict[str, Any]:
    config_path = SMOKE_DIR / "smoke_config.json"
    report_path = SMOKE_DIR / "smoke_report.json"
    checkpoint_path = SMOKE_DIR / "checkpoint_roundtrip.pt"
    require(config_path.is_file() and report_path.is_file() and checkpoint_path.is_file(), "Smoke artifacts missing")
    config = read_json(config_path)
    report = read_json(report_path)
    require(config["runtime_lock"] == runtime and report["runtime_lock"] == runtime, "Smoke runtime lock changed")
    report_core = {key: value for key, value in report.items() if key != "sha256"}
    require(report["sha256"] == payload_sha256(report_core), "Smoke report digest changed")
    require(report["smoke_config_sha256"] == file_sha256(config_path), "Smoke config hash changed")
    require(report["checkpoint_sha256"] == file_sha256(checkpoint_path), "Smoke checkpoint hash changed")
    require(report["historical_loss_formula_max_abs_error"] <= 1e-7 and report["checkpoint_reload_logits_max_abs_diff"] == 0.0, "Smoke gates failed")
    require(all(value > 0.0 for value in report["adapter_gradient_max_by_tensor"].values()), "Smoke adapter gradient gate failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["runtime_lock"] == runtime and checkpoint["epoch"] == SMOKE_EPOCHS, "Smoke checkpoint changed")
    return report


def selection_key(value: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (float(value["acc"]), float(value["roc_auc"]), float(value["macro_f1"]), -int(epoch))


def fold_lock(runtime: dict[str, Any], context: dict[str, Any], spec: dict[str, Any], fold: int) -> dict[str, Any]:
    core = {
        "runtime_sha256": runtime["sha256"],
        "trial_spec": spec,
        "config_key": config_key(spec),
        "fold": int(fold),
        "fold_manifest_sha256": context["fold_manifest"]["sha256"],
    }
    return {**core, "sha256": payload_sha256(core)}


def fold_paths(spec: dict[str, Any], fold: int) -> dict[str, Path]:
    local = WORK_DIR / spec["trial_id"] / f"fold_{fold:02d}"
    compact = TRIALS_DIR / spec["trial_id"]
    return {
        "root": local,
        "resume": local / "resume.pt",
        "checkpoint": local / "checkpoint_best.pt",
        "oof": local / "oof_predictions.csv",
        "history": local / "epoch_metrics.csv",
        "summary": local / "summary.json",
        "complete": local / "complete.json",
        "compact_summary": compact / f"fold_{fold:02d}.json",
        "compact_complete": compact / f"fold_{fold:02d}.complete.json",
    }


def expected_fold_rows(context: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    return context["fold_manifest"]["folds"][fold]["test_rows"]


def prediction_rows(context: dict[str, Any], spec: dict[str, Any], fold: int, logits: np.ndarray, probabilities: np.ndarray, truth: np.ndarray) -> list[dict[str, Any]]:
    anchors = expected_fold_rows(context, fold)
    require(len(anchors) == len(truth) == len(logits) == len(probabilities), "Fold OOF length changed")
    rows: list[dict[str, Any]] = []
    for anchor, label, logit, probability in zip(anchors, truth, logits, probabilities):
        require(int(anchor["truth"]) == int(label), "Fold truth anchor changed")
        rows.append({
            "task_id": context["task_id"],
            "trial_id": spec["trial_id"],
            "arm": spec["arm"],
            "fold": int(fold),
            "subject_id": anchor["subject_id"],
            "original_csv_index": int(anchor["original_csv_index"]),
            "feature_sha256": anchor["feature_sha256"],
            "truth": int(label),
            "prediction": int(np.argmax(probability)),
            "logit_0": float(logit[0]),
            "logit_1": float(logit[1]),
            "probability_0": float(probability[0]),
            "probability_1": float(probability[1]),
            "positive_probability": float(probability[1]),
        })
    return rows


def parse_trial_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(path):
        output.append({
            "task_id": row["task_id"], "trial_id": row["trial_id"], "arm": row["arm"],
            "fold": int(row["fold"]), "subject_id": row["subject_id"],
            "original_csv_index": int(row["original_csv_index"]), "feature_sha256": row["feature_sha256"],
            "truth": int(row["truth"]), "prediction": int(row["prediction"]),
            "logit_0": float(row["logit_0"]), "logit_1": float(row["logit_1"]),
            "probability_0": float(row["probability_0"]), "probability_1": float(row["probability_1"]),
            "positive_probability": float(row["positive_probability"]),
        })
    return output


def resume_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, lock: dict[str, Any], spec: dict[str, Any], fold: int, epoch: int, best: dict[str, Any] | None, gradients: dict[str, float], initial_adapter: dict[str, torch.Tensor], history: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    return {
        "schema": 1, "lock": lock, "spec": spec, "fold": int(fold), "epoch": int(epoch),
        "optimizer_steps": int(epoch), "scheduler_steps": int(epoch),
        "model": clone_cpu_state(model), "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": capture_rng(),
        "best": copy.deepcopy(best), "cumulative_gradient": dict(gradients),
        "initial_adapter_state": {name: tensor.detach().cpu().clone() for name, tensor in initial_adapter.items()},
        "history": copy.deepcopy(history), "elapsed_seconds": float(elapsed),
    }


def load_resume(path: Path, lock: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(value.get("schema") == 1 and value.get("lock") == lock, f"Resume lock changed: {path}")
    epoch = int(value.get("epoch", -1))
    require(0 < epoch <= EPOCHS and value.get("optimizer_steps") == epoch and value.get("scheduler_steps") == epoch, "Resume step count changed")
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    require(int(scheduler.last_epoch) == epoch, "Resume scheduler epoch changed")
    if isinstance(scheduler, historical.RatioPreservingCustomCosineAnnealingLR):
        scheduler.assert_ratio()
    restore_rng(value["rng"])
    return value


def train_fold(base_context: dict[str, Any], spec: dict[str, Any], fold: int, runtime: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = trial_context(base_context, spec)
    paths = fold_paths(spec, fold)
    lock = fold_lock(runtime, context, spec, fold)
    if paths["complete"].is_file() and paths["compact_complete"].is_file():
        return load_completed_fold(base_context, spec, fold, runtime)
    paths["root"].mkdir(parents=True, exist_ok=True)
    allowed = {path.name for name, path in paths.items() if name not in ("root", "compact_summary", "compact_complete")}
    allowed |= {name + ".tmp" for name in allowed}
    unknown = [path for path in paths["root"].iterdir() if path.name not in allowed]
    require(not unknown, f"Unrecognized partial fold artifacts: {unknown}")

    model, criterion, optimizer, scheduler, object_audit = make_training_objects(context, spec)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, _ = historical.fold_positions(context, fold)
    initial_adapter = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name.startswith("private_adapters.")}
    cumulative_gradient = {name: 0.0 for name in initial_adapter}
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resumed_from_epoch = 0
    elapsed_before = 0.0
    if paths["resume"].is_file():
        resumed = load_resume(paths["resume"], lock, model, optimizer, scheduler)
        require(resumed["spec"] == spec and int(resumed["fold"]) == fold, "Resume ownership changed")
        resumed_from_epoch = int(resumed["epoch"])
        start_epoch = resumed_from_epoch + 1
        best = resumed["best"]
        cumulative_gradient = {name: float(value) for name, value in resumed["cumulative_gradient"].items()}
        initial_adapter = {name: tensor.detach().cpu().clone() for name, tensor in resumed["initial_adapter_state"].items()}
        history = copy.deepcopy(resumed["history"])
        require([int(row["epoch"]) for row in history] == list(range(1, start_epoch)), "Resume history changed")
        elapsed_before = float(resumed["elapsed_seconds"])

    session_started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        loss, gradients, _ = historical.train_update(
            model, criterion, optimizer, features, labels, train_mask,
            float(context["protocol"]["training"]["grad_clip"]),
        )
        require(math.isfinite(loss), "Formal loss is non-finite")
        for name, value in gradients.items():
            cumulative_gradient[name] = max(cumulative_gradient.get(name, 0.0), float(value))
        with torch.no_grad():
            full_logits = historical.infer(model, features)[0]
            test_logits_t = full_logits[test_mask].detach().cpu()
            test_probability_t = torch.softmax(test_logits_t, dim=-1)
        test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        test_logits = test_logits_t.numpy().astype(np.float64)
        test_probability = test_probability_t.numpy().astype(np.float64)
        epoch_metrics = historical.binary_metrics(test_truth, test_probability, context["protocol"])
        row = {
            "epoch": int(epoch), "loss": float(loss), "correct": int(epoch_metrics["correct"]),
            "acc": float(epoch_metrics["acc"]), "roc_auc": float(epoch_metrics["roc_auc"]),
            "macro_f1": float(epoch_metrics["macro_f1"]), "bacc": float(epoch_metrics["bacc"]),
            "pr_auc": float(epoch_metrics["pr_auc"]),
        }
        history.append(row)
        key = selection_key(epoch_metrics, epoch)
        if best is None or key > tuple(best["selection_key"]):
            best = {
                "epoch": int(epoch), "selection_key": list(key), "metrics": epoch_metrics,
                "model": clone_cpu_state(model), "test_logits": test_logits,
                "test_probabilities": test_probability, "test_truth": test_truth,
            }
        scheduler.step()
        if isinstance(scheduler, historical.RatioPreservingCustomCosineAnnealingLR):
            scheduler.assert_ratio()
        if epoch % 20 == 0 or epoch == EPOCHS:
            elapsed = elapsed_before + time.perf_counter() - session_started
            atomic_torch_save(paths["resume"], resume_payload(
                model, optimizer, scheduler, lock, spec, fold, epoch, best,
                cumulative_gradient, initial_adapter, history, elapsed,
            ))

    require(best is not None and len(history) == EPOCHS, "Best/history incomplete")
    require([int(row["epoch"]) for row in history] == list(range(1, EPOCHS + 1)), "Epoch history is not 1..400")
    require(int(scheduler.last_epoch) == EPOCHS, "Scheduler did not step 400 times")
    model.load_state_dict(best["model"], strict=True)
    torch.cuda.synchronize(context["device"])
    inference_started = time.perf_counter()
    with torch.no_grad():
        replay_logits_t = historical.infer(model, features)[0]
    torch.cuda.synchronize(context["device"])
    inference_seconds = time.perf_counter() - inference_started
    replay_logits = replay_logits_t[test_mask].detach().cpu().numpy().astype(np.float64)
    require(float(np.max(np.abs(replay_logits - best["test_logits"]))) <= 1e-6, "Best-state replay changed logits")
    rows = prediction_rows(context, spec, fold, best["test_logits"], best["test_probabilities"], best["test_truth"])
    recomputed = metrics(rows)
    require(metrics_close(recomputed, best["metrics"]), "Fold OOF metrics changed")
    diagnostics = None
    if spec["arm"] == "B1":
        diagnostics = historical.private_diagnostics(context, model, test_mask, cumulative_gradient, initial_adapter)
        require(diagnostics["private_trained"] and not diagnostics["private_collapse"], "Private residual did not train")

    elapsed_total = elapsed_before + time.perf_counter() - session_started
    checkpoint = {
        "schema": 1, "lock": lock, "spec": spec, "fold": int(fold),
        "best_epoch": int(best["epoch"]), "best_metrics": best["metrics"],
        "best_model": best["model"], "final_epoch": EPOCHS,
        "optimizer_steps": EPOCHS, "scheduler_steps": EPOCHS,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": capture_rng(),
        "elapsed_seconds": float(elapsed_total),
    }
    atomic_torch_save(paths["checkpoint"], checkpoint)
    atomic_write_csv(paths["oof"], rows)
    atomic_write_csv(paths["history"], history)
    summary = {
        "schema": 1, "lock": lock, "spec": spec, "fold": int(fold),
        "best_epoch": int(best["epoch"]), "best_metrics": best["metrics"],
        "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "selection_uses_test_fold_truth": True,
        "object_audit": object_audit, "diagnostics": diagnostics,
        "optimizer_steps": EPOCHS, "scheduler_steps": EPOCHS,
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_seconds": float(elapsed_total),
        "inference_seconds_full_graph": float(inference_seconds),
        "oof_count": len(rows),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "history_sha256": file_sha256(paths["history"]),
    }
    atomic_write_json(paths["summary"], summary)
    complete_core = {
        "complete": True, "lock": lock,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "oof_sha256": file_sha256(paths["oof"]),
        "history_sha256": file_sha256(paths["history"]),
        "summary_sha256": file_sha256(paths["summary"]),
    }
    complete = {**complete_core, "sha256": payload_sha256(complete_core)}
    atomic_write_json(paths["complete"], complete)
    atomic_write_json(paths["compact_summary"], summary)
    atomic_write_json(paths["compact_complete"], complete)
    print(f"[{spec['trial_id']} fold {fold}] best={best['epoch']} correct={best['metrics']['correct']}/{len(rows)}")
    return load_completed_fold(base_context, spec, fold, runtime)


def load_completed_fold(base_context: dict[str, Any], spec: dict[str, Any], fold: int, runtime: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = trial_context(base_context, spec)
    paths = fold_paths(spec, fold)
    required_names = ("resume", "checkpoint", "oof", "history", "summary", "complete", "compact_summary", "compact_complete")
    require(all(paths[name].is_file() for name in required_names), f"Completed fold artifacts missing: {spec['trial_id']}/{fold}")
    lock = fold_lock(runtime, context, spec, fold)
    summary = read_json(paths["summary"])
    require(summary == read_json(paths["compact_summary"]), "Compact fold summary changed")
    complete = read_json(paths["complete"])
    require(complete == read_json(paths["compact_complete"]), "Compact completion marker changed")
    core = {key: value for key, value in complete.items() if key != "sha256"}
    require(complete["sha256"] == payload_sha256(core) and complete["complete"] is True and complete["lock"] == lock, "Fold completion lock changed")
    for key, name in (("checkpoint_sha256", "checkpoint"), ("oof_sha256", "oof"), ("history_sha256", "history"), ("summary_sha256", "summary")):
        require(complete[key] == file_sha256(paths[name]), f"Completed fold {name} changed")
    require(summary["lock"] == lock and summary["spec"] == spec and int(summary["fold"]) == fold, "Fold summary ownership changed")
    rows = parse_trial_rows(paths["oof"])
    require(len(rows) == int(summary["oof_count"]), "Fold OOF count changed")
    for row, anchor in zip(rows, expected_fold_rows(context, fold)):
        require(row["trial_id"] == spec["trial_id"] and row["fold"] == fold, "Fold OOF ownership changed")
        require(row["subject_id"] == anchor["subject_id"] and row["truth"] == anchor["truth"], "Fold OOF alignment changed")
    require(metrics_close(metrics(rows), summary["best_metrics"]), "Fold OOF metrics changed")
    history = read_csv(paths["history"])
    require([int(row["epoch"]) for row in history] == list(range(1, EPOCHS + 1)), "Fold epoch history changed")
    selected = max(history, key=lambda row: (float(row["acc"]), float(row["roc_auc"]), float(row["macro_f1"]), -int(row["epoch"])))
    require(int(selected["epoch"]) == int(summary["best_epoch"]), "Fold best epoch changed")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    require(checkpoint["lock"] == lock and checkpoint["spec"] == spec and checkpoint["best_epoch"] == summary["best_epoch"], "Best checkpoint lock changed")
    require(checkpoint["optimizer_steps"] == EPOCHS and checkpoint["scheduler_steps"] == EPOCHS, "Best checkpoint steps changed")
    model, _ = build_model(context, spec)
    model.load_state_dict(checkpoint["best_model"], strict=True)
    _, test_mask, _ = historical.fold_positions(context, fold)
    with torch.no_grad():
        replay = historical.infer(model, context["dataset_data"]["Feature"])[0][test_mask]
    replay_probability = torch.softmax(replay, dim=-1).detach().cpu().numpy().astype(np.float64)
    saved_probability = arrays_from_rows(rows)[2]
    require(float(np.max(np.abs(replay_probability - saved_probability))) <= 1e-6, "Checkpoint does not reproduce fold OOF")
    require(np.array_equal(replay_probability.argmax(1), np.asarray([row["prediction"] for row in rows])), "Checkpoint predictions changed")
    del model
    torch.cuda.empty_cache()
    return summary, rows


def aggregate_mechanism(spec: dict[str, Any], fold_summaries: list[dict[str, Any]], b0_parameter_count: int) -> dict[str, Any] | None:
    if spec["arm"] != "B1":
        return None
    diagnostics = [summary["diagnostics"] for summary in fold_summaries]
    require(len(diagnostics) == 10 and all(item is not None for item in diagnostics), "Private diagnostics missing")
    modalities = [entry["name"] for entry in protocol()["modalities"]]
    gradient_names = set(diagnostics[0]["adapter_cumulative_max_gradient"])
    require(all(set(item["adapter_cumulative_max_gradient"]) == gradient_names for item in diagnostics), "Adapter gradient schema changed")
    return {
        "adapter_rank": int(spec["rank"]),
        "adapter_lr_multiplier": float(spec["adapter_lr_multiplier"]),
        "private_shared_ratio_mean_by_modality": {name: float(statistics.mean(item["private_shared_ratio_mean_by_modality"][name] for item in diagnostics)) for name in modalities},
        "private_shared_ratio_max_by_modality": {name: float(max(item["private_shared_ratio_max_by_modality"][name] for item in diagnostics)) for name in modalities},
        "private_shared_ratio_mean": float(statistics.mean(item["private_shared_ratio_mean"] for item in diagnostics)),
        "private_shared_ratio_max": float(max(item["private_shared_ratio_max"] for item in diagnostics)),
        "category_global_cosine_mean": float(statistics.mean(item["category_global_cosine_mean"] for item in diagnostics)),
        "adapter_max_gradient_by_tensor": {name: float(max(item["adapter_cumulative_max_gradient"][name] for item in diagnostics)) for name in sorted(gradient_names)},
        "private_trained_all_folds": bool(all(item["private_trained"] for item in diagnostics)),
        "private_collapse_any_fold": bool(any(item["private_collapse"] for item in diagnostics)),
        "parameter_count_b0": int(b0_parameter_count),
        "parameter_count_b1": int(fold_summaries[0]["object_audit"]["parameter_count"]),
        "parameter_delta": int(fold_summaries[0]["object_audit"]["parameter_count"] - b0_parameter_count),
    }


def safety(result: dict[str, Any]) -> dict[str, Any]:
    value = result["metrics"]
    predicted_pmci = int(value["predicted_counts"]["PMCI"])
    collapse = predicted_pmci <= int(read_json(CONFIG_PATH)["collapse_predicted_pmci_max"])
    reasons: list[str] = []
    if result["spec"]["arm"] == "B0":
        if int(value["tp"]) < 32:
            reasons.append("pMCI TP < 32")
        if float(value["sen"]) + 5e-5 < 0.7111:
            reasons.append("pMCI sensitivity < 0.7111")
        if float(value["bacc"]) < 0.841:
            reasons.append("BACC < 0.841")
    else:
        if int(value["tp"]) < 35:
            reasons.append("pMCI TP < 35")
        # The preregistered 0.7778 is the four-decimal rendering of 35/45.
        if float(value["sen"]) + 5e-5 < 0.7778:
            reasons.append("pMCI sensitivity < 0.7778")
        if float(value["bacc"]) < 0.878:
            reasons.append("BACC < 0.878")
        if float(value["pr_auc"]) < 0.67:
            reasons.append("PR-AUC < 0.67")
    if collapse:
        reasons.append("majority-class prediction collapse")
    return {"safe": not reasons, "collapse": collapse, "reasons": reasons}


def fold_metric_record(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary["best_metrics"]
    return {
        "fold": int(summary["fold"]), "best_epoch": int(summary["best_epoch"]),
        "correct": int(best["correct"]), "acc": float(best["acc"]),
        "roc_auc": float(best["roc_auc"]), "pr_auc": float(best["pr_auc"]),
        "macro_f1": float(best["macro_f1"]), "bacc": float(best["bacc"]),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "inference_seconds_full_graph": float(summary.get("inference_seconds_full_graph", 0.0)),
        "resumed_from_epoch": int(summary.get("resumed_from_epoch", 0)),
    }


def make_trial_result(spec: dict[str, Any], rows: list[dict[str, Any]], fold_summaries: list[dict[str, Any]], parameter_count_value: int, mechanism: dict[str, Any] | None, reused_from: str | None) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["original_csv_index"]))
    value = metrics(ordered)
    fold_metrics = [fold_metric_record(summary) for summary in fold_summaries]
    require([row["fold"] for row in fold_metrics] == list(FOLDS), "Trial fold order changed")
    result = {
        "trial_id": spec["trial_id"], "spec": spec, "config_key": config_key(spec),
        "metrics": value,
        "fold_acc_mean": float(statistics.mean(row["acc"] for row in fold_metrics)),
        "fold_acc_sample_std": float(statistics.stdev(row["acc"] for row in fold_metrics)),
        "fold_roc_auc_mean": float(statistics.mean(row["roc_auc"] for row in fold_metrics)),
        "fold_roc_auc_sample_std": float(statistics.stdev(row["roc_auc"] for row in fold_metrics)),
        "fold_pr_auc_mean": float(statistics.mean(row["pr_auc"] for row in fold_metrics)),
        "fold_pr_auc_sample_std": float(statistics.stdev(row["pr_auc"] for row in fold_metrics)),
        "fold_metrics": fold_metrics,
        "parameter_count": int(parameter_count_value),
        "training_time_seconds": float(sum(row["elapsed_seconds"] for row in fold_metrics)),
        "inference_time_seconds": float(sum(row["inference_seconds_full_graph"] for row in fold_metrics)),
        "mechanism": mechanism,
        "reused": reused_from is not None,
        "reused_from": reused_from,
    }
    result["safety"] = safety(result)
    return result


def trial_dir(spec: dict[str, Any]) -> Path:
    return TRIALS_DIR / spec["trial_id"]


def write_trial_artifacts(spec: dict[str, Any], rows: list[dict[str, Any]], fold_summaries: list[dict[str, Any]], result: dict[str, Any]) -> None:
    root = trial_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "config.json", spec)
    atomic_write_csv(root / "oof_predictions.csv", sorted(rows, key=lambda row: int(row["original_csv_index"])))
    for summary in fold_summaries:
        path = root / f"fold_{int(summary['fold']):02d}.json"
        if not path.is_file() or read_json(path) != summary:
            atomic_write_json(path, summary)
        marker_path = root / f"fold_{int(summary['fold']):02d}.complete.json"
        if not marker_path.is_file():
            marker_core = {"complete": True, "spec_sha256": spec["config_sha256"], "fold": int(summary["fold"]), "summary_sha256": file_sha256(path)}
            atomic_write_json(marker_path, {**marker_core, "sha256": payload_sha256(marker_core)})
    atomic_write_json(root / "summary.json", result)
    complete_core = {
        "complete": True, "spec": spec,
        "config_sha256": file_sha256(root / "config.json"),
        "oof_sha256": file_sha256(root / "oof_predictions.csv"),
        "summary_sha256": file_sha256(root / "summary.json"),
        "fold_summary_sha256": {f"fold_{fold:02d}": file_sha256(root / f"fold_{fold:02d}.json") for fold in FOLDS},
    }
    atomic_write_json(root / "COMPLETE.json", {**complete_core, "sha256": payload_sha256(complete_core)})


def load_trial_result(spec: dict[str, Any], base_context: dict[str, Any]) -> dict[str, Any]:
    root = trial_dir(spec)
    required = [root / "config.json", root / "oof_predictions.csv", root / "summary.json", root / "COMPLETE.json"]
    required += [root / f"fold_{fold:02d}.json" for fold in FOLDS]
    required += [root / f"fold_{fold:02d}.complete.json" for fold in FOLDS]
    require(all(path.is_file() for path in required), f"Trial artifacts incomplete: {spec['trial_id']}")
    require(read_json(root / "config.json") == spec, "Trial config changed")
    complete = read_json(root / "COMPLETE.json")
    core = {key: value for key, value in complete.items() if key != "sha256"}
    require(complete["sha256"] == payload_sha256(core) and complete["complete"] is True and complete["spec"] == spec, "Trial marker changed")
    require(complete["config_sha256"] == file_sha256(root / "config.json") and complete["oof_sha256"] == file_sha256(root / "oof_predictions.csv") and complete["summary_sha256"] == file_sha256(root / "summary.json"), "Trial artifact hash changed")
    for fold in FOLDS:
        fold_path = root / f"fold_{fold:02d}.json"
        require(complete["fold_summary_sha256"][f"fold_{fold:02d}"] == file_sha256(fold_path), "Trial fold summary changed")
        marker = read_json(root / f"fold_{fold:02d}.complete.json")
        marker_core = {key: value for key, value in marker.items() if key != "sha256"}
        require(marker["sha256"] == payload_sha256(marker_core) and marker["summary_sha256"] == file_sha256(fold_path), "Trial fold marker changed")
    rows = parse_trial_rows(root / "oof_predictions.csv")
    context = trial_context(base_context, spec)
    require(all(row["trial_id"] == spec["trial_id"] and row["arm"] == spec["arm"] for row in rows), "Trial OOF ownership changed")
    validate_oof(rows, context, spec["arm"])
    result = read_json(root / "summary.json")
    require(result["spec"] == spec and result["trial_id"] == spec["trial_id"], "Trial result ownership changed")
    require(metrics_close(metrics(rows), result["metrics"]), "Trial metrics changed")
    require(result["safety"] == safety(result), "Trial safety changed")
    return result


def historical_trial(spec: dict[str, Any], base_context: dict[str, Any]) -> dict[str, Any]:
    arm = spec["arm"]
    rows = typed_rows(read_csv(HISTORICAL_ANCHORS[arm]))
    trial_rows = [{**row, "trial_id": spec["trial_id"]} for row in rows]
    fold_payload = read_json(HISTORICAL_ANCHORS["fold_metrics"])[arm]
    inspect = read_json(INSPECT_PATH)
    fold_summaries: list[dict[str, Any]] = []
    for row in fold_payload:
        best = {
            "correct": int(row["correct"]), "acc": float(row["acc"]),
            "roc_auc": float(row["roc_auc"]), "pr_auc": float(row["pr_auc"]),
            "macro_f1": float(row["macro_f1"]), "bacc": float(row["bacc"]),
        }
        fold_summaries.append({
            "schema": "historical_reuse", "spec": spec, "fold": int(row["fold"]),
            "best_epoch": int(row["best_epoch"]), "best_metrics": best,
            "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "inference_seconds_full_graph": float(row["inference_seconds_full_graph"]),
            "resumed_from_epoch": int(row.get("resumed_from_epoch", 0)),
            "historical_result_commit": HISTORICAL_RESULT_COMMIT,
        })
    parameter_value = int(inspect["historical_reuse"]["parameter_count"][arm])
    mechanism = inspect["historical_reuse"]["mechanism_B1"] if arm == "B1" else None
    result = make_trial_result(spec, trial_rows, fold_summaries, parameter_value, mechanism, f"historical:{HISTORICAL_RESULT_COMMIT}:{arm}")
    write_trial_artifacts(spec, trial_rows, fold_summaries, result)
    return load_trial_result(spec, base_context)


def alias_trial(spec: dict[str, Any], source_spec: dict[str, Any], base_context: dict[str, Any]) -> dict[str, Any]:
    source_root = trial_dir(source_spec)
    source_rows = parse_trial_rows(source_root / "oof_predictions.csv")
    rows = [{**row, "trial_id": spec["trial_id"]} for row in source_rows]
    fold_summaries: list[dict[str, Any]] = []
    for fold in FOLDS:
        source_fold = read_json(source_root / f"fold_{fold:02d}.json")
        fold_summaries.append({
            "schema": "duplicate_config_reuse", "spec": spec, "fold": fold,
            "best_epoch": int(source_fold["best_epoch"]),
            "best_metrics": source_fold["best_metrics"],
            "selection": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
            "elapsed_seconds": float(source_fold.get("elapsed_seconds", 0.0)),
            "inference_seconds_full_graph": float(source_fold.get("inference_seconds_full_graph", 0.0)),
            "resumed_from_epoch": int(source_fold.get("resumed_from_epoch", 0)),
            "alias_of": source_spec["trial_id"],
        })
    source_result = read_json(source_root / "summary.json")
    result = make_trial_result(
        spec, rows, fold_summaries, int(source_result["parameter_count"]),
        copy.deepcopy(source_result.get("mechanism")), f"trial:{source_spec['trial_id']}",
    )
    write_trial_artifacts(spec, rows, fold_summaries, result)
    return load_trial_result(spec, base_context)


def train_trial(spec: dict[str, Any], base_context: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    rows_by_fold: list[list[dict[str, Any]]] = []
    for fold in FOLDS:
        summary, rows = train_fold(base_context, spec, fold, runtime)
        summaries.append(summary)
        rows_by_fold.append(rows)
    rows = [row for group in rows_by_fold for row in group]
    context = trial_context(base_context, spec)
    validate_oof(rows, context, spec["arm"])
    parameter_value = int(summaries[0]["object_audit"]["parameter_count"])
    if spec["arm"] == "B0":
        b0_parameter_count = parameter_value
    else:
        b0_spec = trial_spec("PARAM_REFERENCE", "audit", "B0", spec["lr"], spec["weight_decay"], spec["dropout"])
        reference, _ = build_model(trial_context(base_context, b0_spec), b0_spec)
        b0_parameter_count = parameter_count(reference)
        del reference
    mechanism = aggregate_mechanism(spec, summaries, b0_parameter_count)
    result = make_trial_result(spec, rows, summaries, parameter_value, mechanism, None)
    write_trial_artifacts(spec, rows, summaries, result)
    return load_trial_result(spec, base_context)


def is_historical_exact(spec: dict[str, Any]) -> bool:
    base_exact = math.isclose(spec["lr"], 0.01) and math.isclose(spec["weight_decay"], 0.0005) and math.isclose(spec["dropout"], 0.67)
    if spec["arm"] == "B0":
        return base_exact
    return base_exact and spec["rank"] == 8 and math.isclose(spec["adapter_lr_multiplier"], 2.0)


def registry_save(registry: dict[str, Any]) -> None:
    core = {key: value for key, value in registry.items() if key != "sha256"}
    atomic_write_json(TRIAL_MANIFEST_PATH, {**core, "sha256": payload_sha256(core)})


def registry_load(runtime: dict[str, Any]) -> dict[str, Any]:
    if TRIAL_MANIFEST_PATH.is_file():
        value = read_json(TRIAL_MANIFEST_PATH)
        core = {key: item for key, item in value.items() if key != "sha256"}
        require(value["sha256"] == payload_sha256(core) and value["runtime_lock"] == runtime, "Trial manifest lock changed")
        return value
    value = {
        "experiment": EXPERIMENT_ID, "runtime_lock": runtime,
        "strategy": "fixed staged coordinate search A -> B -> C -> D; optional registered local LR pair only",
        "trials": {}, "stages": {}, "local_refinement": {"triggered": False, "reason": "not_evaluated"},
    }
    registry_save(value)
    return read_json(TRIAL_MANIFEST_PATH)


def registry_result_spec(registry: dict[str, Any], trial_id: str) -> dict[str, Any]:
    return registry["trials"][trial_id]["spec"]


def ensure_trial(spec: dict[str, Any], base_context: dict[str, Any], runtime: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if (trial_dir(spec) / "COMPLETE.json").is_file():
        result = load_trial_result(spec, base_context)
    else:
        duplicate: dict[str, Any] | None = None
        for entry in registry["trials"].values():
            if entry.get("status") == "complete" and entry.get("config_key") == config_key(spec):
                duplicate = entry["spec"]
                break
        if duplicate is not None:
            result = alias_trial(spec, duplicate, base_context)
        elif is_historical_exact(spec):
            result = historical_trial(spec, base_context)
        else:
            result = train_trial(spec, base_context, runtime)
    registry["trials"][spec["trial_id"]] = {
        "status": "complete", "spec": spec, "config_key": config_key(spec),
        "summary_path": (trial_dir(spec) / "summary.json").relative_to(ROOT).as_posix(),
        "oof_path": (trial_dir(spec) / "oof_predictions.csv").relative_to(ROOT).as_posix(),
        "reused": bool(result["reused"]), "reused_from": result["reused_from"],
        "metrics": result["metrics"], "safety": result["safety"],
    }
    registry_save(registry)
    return result


def rank_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(result["metrics"]["correct"]), float(result["metrics"]["roc_auc"]),
        float(result["fold_roc_auc_mean"]), float(result["metrics"]["macro_f1"]),
        float(result["metrics"]["bacc"]), -int(result["parameter_count"]),
        -int(result["spec"]["ordinal"]),
    )


def select_stage(stage: str, results: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    require(bool(results), f"No results in stage {stage}")
    safe = [result for result in results if result["safety"]["safe"]]
    overall = max(results, key=rank_key)
    selected = max(safe, key=rank_key) if safe else None
    selection = {
        "stage": stage,
        "candidate_trial_ids": [result["trial_id"] for result in results],
        "safe_trial_ids": [result["trial_id"] for result in safe],
        "best_overall_trial_id": overall["trial_id"],
        "selected_trial_id": selected["trial_id"] if selected else None,
        "selection_rule": "safety > Correct > pooled ROC-AUC > fold mean ROC-AUC > Macro-F1 > BACC > smaller params > earlier trial",
    }
    registry["stages"][stage] = selection
    registry_save(registry)
    return selection


def paired(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["subject_id"]: row for row in parse_trial_rows(trial_dir(left["spec"]) / "oof_predictions.csv")}
    right_rows = {row["subject_id"]: row for row in parse_trial_rows(trial_dir(right["spec"]) / "oof_predictions.csv")}
    require(set(left_rows) == set(right_rows), "Paired subject IDs changed")
    repairs = damages = changed = 0
    for subject_id in sorted(left_rows):
        lrow, rrow = left_rows[subject_id], right_rows[subject_id]
        require(lrow["truth"] == rrow["truth"], "Paired truth changed")
        lcorrect = lrow["prediction"] == lrow["truth"]
        rcorrect = rrow["prediction"] == rrow["truth"]
        repairs += int((not lcorrect) and rcorrect)
        damages += int(lcorrect and (not rcorrect))
        changed += int(lrow["prediction"] != rrow["prediction"])
    pvalue = float(binomtest(min(repairs, damages), repairs + damages, 0.5, alternative="two-sided").pvalue) if repairs + damages else 1.0
    return {
        "left_trial_id": left["trial_id"], "right_trial_id": right["trial_id"],
        "repairs": repairs, "damages": damages, "changed": changed,
        "correct_delta": int(right["metrics"]["correct"] - left["metrics"]["correct"]),
        "metric_deltas": {key: float(right["metrics"][key] - left["metrics"][key]) for key in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe")},
        "exact_mcnemar_p": pvalue,
    }


def run_search(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Search requires cuda:0")
    source = source_gate()
    runtime = runtime_lock(source)
    validate_smoke(runtime)
    base_context = build_context(torch.device(device_text))
    registry = registry_load(runtime)
    results: dict[str, dict[str, Any]] = {}

    stage_a_specs = [trial_spec(f"A{index}", "A", "B0", lr, 0.0005, 0.67, ordinal=index) for index, lr in enumerate((0.0075, 0.01, 0.0125), 1)]
    for spec in stage_a_specs:
        results[spec["trial_id"]] = ensure_trial(spec, base_context, runtime, registry)
    stage_a = select_stage("A", [results[spec["trial_id"]] for spec in stage_a_specs], registry)
    require(stage_a["selected_trial_id"] is not None, "Stage A has no class-safe candidate")
    best_a = results[stage_a["selected_trial_id"]]

    stage_b_specs = [trial_spec(f"B{index}", "B", "B0", best_a["spec"]["lr"], wd, 0.67, ordinal=3 + index) for index, wd in enumerate((0.00025, 0.0005, 0.001), 1)]
    for spec in stage_b_specs:
        results[spec["trial_id"]] = ensure_trial(spec, base_context, runtime, registry)
    stage_b = select_stage("B", [results[spec["trial_id"]] for spec in stage_b_specs], registry)
    require(stage_b["selected_trial_id"] is not None, "Stage B has no class-safe candidate")
    best_b = results[stage_b["selected_trial_id"]]

    stage_c_specs = [trial_spec(f"C{index}", "C", "B0", best_b["spec"]["lr"], best_b["spec"]["weight_decay"], dropout, ordinal=6 + index) for index, dropout in enumerate((0.55, 0.67, 0.75), 1)]
    for spec in stage_c_specs:
        results[spec["trial_id"]] = ensure_trial(spec, base_context, runtime, registry)
    stage_c = select_stage("C", [results[spec["trial_id"]] for spec in stage_c_specs], registry)
    require(stage_c["selected_trial_id"] is not None, "Stage C has no class-safe candidate")
    tuned_b0 = results[stage_c["selected_trial_id"]]

    stage_d_specs: list[dict[str, Any]] = []
    ordinal = 10
    for rank in (4, 8):
        for multiplier in (0.5, 1.0, 2.0):
            trial_id = f"D{ordinal - 9}"
            stage_d_specs.append(trial_spec(trial_id, "D", "B1", tuned_b0["spec"]["lr"], tuned_b0["spec"]["weight_decay"], tuned_b0["spec"]["dropout"], rank=rank, multiplier=multiplier, ordinal=ordinal))
            ordinal += 1
    for spec in stage_d_specs:
        results[spec["trial_id"]] = ensure_trial(spec, base_context, runtime, registry)
    stage_d = select_stage("D", [results[spec["trial_id"]] for spec in stage_d_specs], registry)
    tuned_b1 = results[stage_d["selected_trial_id"]] if stage_d["selected_trial_id"] else results[stage_d["best_overall_trial_id"]]

    local_triggered = False
    local_reason = "Stage D best is outside the registered 519-522 safe-improvement window"
    if stage_d["selected_trial_id"] is not None:
        d_pair = paired(tuned_b0, tuned_b1)
        m = tuned_b1["metrics"]
        local_triggered = bool(
            519 <= int(m["correct"]) <= 522 and d_pair["repairs"] > d_pair["damages"]
            and int(m["tp"]) >= 35 and float(m["bacc"]) >= 0.878 and float(m["pr_auc"]) >= 0.67
        )
        if int(m["correct"]) >= 523 and tuned_b1["fold_acc_mean"] > 0.9757 and tuned_b1["fold_roc_auc_mean"] > 0.9049:
            local_triggered = False
            local_reason = "Stage D already reached the registered target"
        elif int(m["correct"]) <= 518:
            local_triggered = False
            local_reason = "Stage D Correct <= 518"
        elif local_triggered:
            local_reason = "Stage D best entered the registered 519-522 safe-improvement window"
    if local_triggered:
        local_specs = [
            trial_spec("L1", "LOCAL", "B1", tuned_b1["spec"]["lr"] * 0.9, tuned_b1["spec"]["weight_decay"], tuned_b1["spec"]["dropout"], rank=tuned_b1["spec"]["rank"], multiplier=tuned_b1["spec"]["adapter_lr_multiplier"], ordinal=16),
            trial_spec("L2", "LOCAL", "B1", tuned_b1["spec"]["lr"] * 1.1, tuned_b1["spec"]["weight_decay"], tuned_b1["spec"]["dropout"], rank=tuned_b1["spec"]["rank"], multiplier=tuned_b1["spec"]["adapter_lr_multiplier"], ordinal=17),
        ]
        for spec in local_specs:
            results[spec["trial_id"]] = ensure_trial(spec, base_context, runtime, registry)
        local_selection = select_stage("LOCAL", [tuned_b1, *[results[spec["trial_id"]] for spec in local_specs]], registry)
        if local_selection["selected_trial_id"] is not None:
            tuned_b1 = results[local_selection["selected_trial_id"]]
    registry["local_refinement"] = {"triggered": local_triggered, "reason": local_reason}
    registry["tuned_b0_trial_id"] = tuned_b0["trial_id"]
    registry["tuned_b1_trial_id"] = tuned_b1["trial_id"]
    registry_save(registry)

    ordered_results = [load_trial_result(entry["spec"], base_context) for entry in sorted(registry["trials"].values(), key=lambda entry: int(entry["spec"]["ordinal"]))]
    all_trials_core = {"runtime_lock": runtime, "trial_count": len(ordered_results), "trials": ordered_results}
    atomic_write_json(ALL_TRIALS_PATH, {**all_trials_core, "sha256": payload_sha256(all_trials_core)})
    search_core = {
        "runtime_lock": runtime, "trial_manifest_sha256": file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": file_sha256(ALL_TRIALS_PATH),
        "stage_selections": registry["stages"], "local_refinement": registry["local_refinement"],
        "tuned_b0_trial_id": tuned_b0["trial_id"], "tuned_b1_trial_id": tuned_b1["trial_id"],
        "failure_count": 0,
    }
    atomic_write_json(SEARCH_SUMMARY_PATH, {**search_core, "sha256": payload_sha256(search_core)})
    print(json.dumps({"search": "COMPLETE", "tuned_b0": tuned_b0["trial_id"], "tuned_b1": tuned_b1["trial_id"], "local_triggered": local_triggered}, indent=2))
    del base_context
    torch.cuda.empty_cache()


def historical_reference(arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(arm in ("B0", "B1"), "Unknown historical arm")
    spec = trial_spec(f"HISTORICAL_{arm}", "HISTORICAL", arm, 0.01, 0.0005, 0.67, rank=8 if arm == "B1" else 0, multiplier=2.0 if arm == "B1" else 0.0, ordinal=-2 if arm == "B0" else -1)
    rows = [{**row, "trial_id": spec["trial_id"]} for row in typed_rows(read_csv(HISTORICAL_ANCHORS[arm]))]
    fold_payload = read_json(HISTORICAL_ANCHORS["fold_metrics"])[arm]
    summaries = [{
        "fold": int(row["fold"]), "best_epoch": int(row["best_epoch"]),
        "best_metrics": {
            "correct": int(row["correct"]), "acc": float(row["acc"]),
            "roc_auc": float(row["roc_auc"]), "pr_auc": float(row["pr_auc"]),
            "macro_f1": float(row["macro_f1"]), "bacc": float(row["bacc"]),
        },
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "inference_seconds_full_graph": float(row["inference_seconds_full_graph"]),
        "resumed_from_epoch": int(row.get("resumed_from_epoch", 0)),
    } for row in fold_payload]
    inspect = read_json(INSPECT_PATH)
    result = make_trial_result(
        spec, rows, summaries,
        int(inspect["historical_reuse"]["parameter_count"][arm]),
        inspect["historical_reuse"]["mechanism_B1"] if arm == "B1" else None,
        f"historical:{HISTORICAL_RESULT_COMMIT}:{arm}",
    )
    return result, rows


def paired_rows(left: dict[str, Any], left_rows: list[dict[str, Any]], right: dict[str, Any], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {row["subject_id"]: row for row in left_rows}
    right_map = {row["subject_id"]: row for row in right_rows}
    require(set(left_map) == set(right_map), "Paired subject IDs changed")
    repairs = damages = changed = 0
    for subject_id in sorted(left_map):
        lrow, rrow = left_map[subject_id], right_map[subject_id]
        require(lrow["truth"] == rrow["truth"], "Paired truth changed")
        lcorrect = lrow["prediction"] == lrow["truth"]
        rcorrect = rrow["prediction"] == rrow["truth"]
        repairs += int((not lcorrect) and rcorrect)
        damages += int(lcorrect and (not rcorrect))
        changed += int(lrow["prediction"] != rrow["prediction"])
    pvalue = float(binomtest(min(repairs, damages), repairs + damages, 0.5, alternative="two-sided").pvalue) if repairs + damages else 1.0
    return {
        "left_trial_id": left["trial_id"], "right_trial_id": right["trial_id"],
        "repairs": repairs, "damages": damages, "changed": changed,
        "correct_delta": int(right["metrics"]["correct"] - left["metrics"]["correct"]),
        "metric_deltas": {key: float(right["metrics"][key] - left["metrics"][key]) for key in ("acc", "roc_auc", "pr_auc", "macro_f1", "bacc", "weighted_f1", "sen", "spe")},
        "exact_mcnemar_p": pvalue,
    }


def target_gate(result: dict[str, Any]) -> dict[str, Any]:
    value = result["metrics"]
    checks = {
        "correct_at_least_523": int(value["correct"]) >= 523,
        "fold_acc_mean_above_0_9757": float(result["fold_acc_mean"]) > 0.9757,
        "fold_roc_auc_mean_above_0_9049": float(result["fold_roc_auc_mean"]) > 0.9049,
        "pmci_tp_at_least_35": int(value["tp"]) >= 35,
        "bacc_at_least_0_878": float(value["bacc"]) >= 0.878,
        "pr_auc_at_least_0_67": float(value["pr_auc"]) >= 0.67,
        "no_class_collapse": not result["safety"]["collapse"],
    }
    return {"reached": all(checks.values()), "checks": checks}


def final_decision(tuned_b0: dict[str, Any], tuned_b1: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    b0_gate = target_gate(tuned_b0)
    b1_gate = target_gate(tuned_b1)
    b1_better = rank_key(tuned_b1) > rank_key(tuned_b0)
    best = max([result for result in (tuned_b0, tuned_b1) if result["safety"]["safe"]], key=rank_key, default=max((tuned_b0, tuned_b1), key=rank_key))
    reference_comparison = comparisons["tuned_b1_vs_historical_b1"] if best["spec"]["arm"] == "B1" else comparisons["tuned_b0_vs_historical_b0"]
    no_gain_reasons: list[str] = []
    if int(best["metrics"]["correct"]) <= 518:
        no_gain_reasons.append("best Correct <= 518")
    if reference_comparison["repairs"] <= reference_comparison["damages"]:
        no_gain_reasons.append("repairs <= damages")
    delta = reference_comparison["metric_deltas"]
    if delta["sen"] < -0.005:
        no_gain_reasons.append("pMCI sensitivity declined")
    if delta["bacc"] < -0.005:
        no_gain_reasons.append("BACC declined materially")
    if delta["pr_auc"] < -0.005:
        no_gain_reasons.append("PR-AUC declined materially")
    if best["safety"]["collapse"]:
        no_gain_reasons.append("majority-class prediction collapse")

    if b1_gate["reached"] and b1_better:
        decision = "PRIVATE_RESIDUAL_TARGET_GO"
    elif b0_gate["reached"] and not b1_better:
        decision = "BASE_TARGET_ONLY"
    elif 519 <= int(best["metrics"]["correct"]) <= 522 and best["safety"]["safe"] and reference_comparison["repairs"] > reference_comparison["damages"] and not no_gain_reasons:
        decision = "TAD_BINARY_NEAR"
    else:
        decision = "TAD_BINARY_TUNE_NO_GAIN"
    return {
        "decision": decision,
        "target_status": "TAD_BINARY_TARGET_REACHED" if (b0_gate["reached"] or b1_gate["reached"]) else "TAD_BINARY_TARGET_NOT_REACHED",
        "tuned_b0_target": b0_gate,
        "tuned_b1_target": b1_gate,
        "tuned_b1_outperforms_tuned_b0": b1_better,
        "best_safe_trial_id": best["trial_id"],
        "no_gain_reasons": no_gain_reasons,
        "next_action": "stop hyperparameter expansion; modify model structure" if decision == "TAD_BINARY_TUNE_NO_GAIN" else "stop registered search",
    }


def validate_search(runtime: dict[str, Any], base_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(TRIAL_MANIFEST_PATH.is_file() and ALL_TRIALS_PATH.is_file() and SEARCH_SUMMARY_PATH.is_file(), "Search artifacts missing")
    registry = registry_load(runtime)
    require(registry.get("tuned_b0_trial_id") and registry.get("tuned_b1_trial_id"), "Search did not select tuned models")
    all_trials = read_json(ALL_TRIALS_PATH)
    all_core = {key: value for key, value in all_trials.items() if key != "sha256"}
    require(all_trials["sha256"] == payload_sha256(all_core) and all_trials["runtime_lock"] == runtime, "all_trials.json changed")
    search = read_json(SEARCH_SUMMARY_PATH)
    search_core = {key: value for key, value in search.items() if key != "sha256"}
    require(search["sha256"] == payload_sha256(search_core) and search["runtime_lock"] == runtime, "Search summary changed")
    require(search["trial_manifest_sha256"] == file_sha256(TRIAL_MANIFEST_PATH) and search["all_trials_sha256"] == file_sha256(ALL_TRIALS_PATH), "Search hashes changed")
    results: dict[str, dict[str, Any]] = {}
    for trial_id, entry in registry["trials"].items():
        spec = entry["spec"]
        result = load_trial_result(spec, base_context)
        results[trial_id] = result
        if not result["reused"]:
            for fold in FOLDS:
                load_completed_fold(base_context, spec, fold, runtime)
    require(len(results) == int(all_trials["trial_count"]), "Search trial count changed")
    return registry, results


def render_report(runtime: dict[str, Any], registry: dict[str, Any], results: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]], comparisons: dict[str, Any], decision: dict[str, Any], total_seconds: float) -> str:
    tuned_b0 = results[registry["tuned_b0_trial_id"]]
    tuned_b1 = results[registry["tuned_b1_trial_id"]]
    rows = [references["B0"], references["B1"], tuned_b0, tuned_b1]
    lines = [
        "# TADPOLE Binary Task Hyperparameter Search v1", "",
        f"- Source commit: `{runtime['source_commit']}`",
        "- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)",
        f"- Branch: `{BRANCH}`", "- Device: `cuda:0`",
        f"- Decision: **{decision['decision']}**",
        f"- Target status: **{decision['target_status']}**",
        f"- Formal aggregation time: {total_seconds:.3f} seconds", "",
        "## Locked historical protocol", "",
        "All trials use TADPOLE SMCI_PMCI (535 subjects: 490 sMCI, 45 pMCI; pMCI positive), five real modalities, seed 0, ten full-batch transductive folds, 400 epochs, global full-dataset historical class weights, criterion_lossv2 with two unnormalized OVR CE terms, label smoothing 0.05, orthogonality rate 0.0001, Adam, grad clip 1, CustomCosineAnnealingLR(T_max=400), test-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch, graph off, EMA off, and no ensemble.",
        "The previous Binary Task-Adaptive Private Residual Calibration experiment is not reused because it changed loss, class-weight scope, and validation protocol.", "",
        "## Selected configurations", "",
        f"- TUNED_B0: trial `{tuned_b0['trial_id']}`, lr={tuned_b0['spec']['lr']}, weight_decay={tuned_b0['spec']['weight_decay']}, dropout={tuned_b0['spec']['dropout']}.",
        f"- TUNED_B1: trial `{tuned_b1['trial_id']}`, lr={tuned_b1['spec']['lr']}, weight_decay={tuned_b1['spec']['weight_decay']}, dropout={tuned_b1['spec']['dropout']}, rank={tuned_b1['spec']['rank']}, adapter LR multiplier={tuned_b1['spec']['adapter_lr_multiplier']}.",
        f"- Local LR refinement triggered: `{registry['local_refinement']['triggered']}` ({registry['local_refinement']['reason']}).", "",
        "## Primary results", "",
        "| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Fold PR mean +/- SD | Macro-F1 | BACC | Weighted-F1 | pMCI SEN | sMCI SPE | Params | Train sec | Infer sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = ["Historical B0", "Historical B1", "TUNED_B0", "TUNED_B1"]
    for label, result in zip(labels, rows):
        m = result["metrics"]
        lines.append(f"| {label} | {m['correct']}/{m['n']} | {m['acc']:.7f} | {result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f} | {m['roc_auc']:.7f} | {result['fold_roc_auc_mean']:.7f} +/- {result['fold_roc_auc_sample_std']:.7f} | {m['pr_auc']:.7f} | {result['fold_pr_auc_mean']:.7f} +/- {result['fold_pr_auc_sample_std']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {m['weighted_f1']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {result['parameter_count']} | {result['training_time_seconds']:.1f} | {result['inference_time_seconds']:.4f} |")
    lines += ["", "### Confusion and prediction counts", ""]
    for label, result in zip(labels, rows):
        m = result["metrics"]
        lines.append(f"- {label}: confusion={m['confusion_matrix']} (sMCI,pMCI); TP/FN/TN/FP={m['tp']}/{m['fn']}/{m['tn']}/{m['fp']}; predicted sMCI/pMCI={m['predicted_counts']['SMCI']}/{m['predicted_counts']['PMCI']}.")
    lines += ["", "## Registered trial table", "", "| Trial | Stage | Arm | LR | WD | Dropout | Rank | Adapter mult | Safe | Correct | ROC-AUC | Fold AUC mean | PR-AUC | Macro-F1 | BACC |", "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for result in sorted(results.values(), key=lambda item: int(item["spec"]["ordinal"])):
        s, m = result["spec"], result["metrics"]
        lines.append(f"| {s['trial_id']} | {s['stage']} | {s['arm']} | {s['lr']:.8g} | {s['weight_decay']:.8g} | {s['dropout']:.4g} | {s['rank']} | {s['adapter_lr_multiplier']:.3g} | {result['safety']['safe']} | {m['correct']} | {m['roc_auc']:.7f} | {result['fold_roc_auc_mean']:.7f} | {m['pr_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} |")
    lines += ["", "## Pairwise comparisons", "", "| Comparison | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | Exact McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        delta = comparison["metric_deltas"]
        lines.append(f"| {name} | {comparison['correct_delta']:+d} | {comparison['repairs']} | {comparison['damages']} | {comparison['changed']} | {delta['acc']:+.7f} | {delta['roc_auc']:+.7f} | {delta['pr_auc']:+.7f} | {delta['macro_f1']:+.7f} | {delta['bacc']:+.7f} | {comparison['exact_mcnemar_p']:.8f} |")
    mechanism = tuned_b1.get("mechanism")
    lines += ["", "## TUNED_B1 mechanism diagnostics", ""]
    if mechanism:
        lines += [
            f"- Private/shared ratio mean/max: {mechanism['private_shared_ratio_mean']:.7f} / {mechanism['private_shared_ratio_max']:.7f}.",
            f"- Category-Global cosine mean: {mechanism['category_global_cosine_mean']:.7f}.",
            f"- Private collapse: `{mechanism['private_collapse_any_fold']}`; all folds trained: `{mechanism['private_trained_all_folds']}`.",
            f"- Parameter increase: {mechanism['parameter_delta']} ({mechanism['parameter_count_b0']} -> {mechanism['parameter_count_b1']}).",
            f"- Per-modality ratio mean: `{json.dumps(mechanism['private_shared_ratio_mean_by_modality'], sort_keys=True)}`.",
            f"- Per-modality ratio max: `{json.dumps(mechanism['private_shared_ratio_max_by_modality'], sort_keys=True)}`.",
            f"- Adapter maximum gradients: `{json.dumps(mechanism['adapter_max_gradient_by_tensor'], sort_keys=True)}`.",
        ]
    lines += ["", "## DGFMC comparison", "",
        f"- DGFMC fold mean ACC target 97.57%: TUNED_B1={100*tuned_b1['fold_acc_mean']:.4f}% -> `{tuned_b1['fold_acc_mean'] > 0.9757}`.",
        f"- DGFMC fold mean ROC-AUC target 90.49%: TUNED_B1={100*tuned_b1['fold_roc_auc_mean']:.4f}% -> `{tuned_b1['fold_roc_auc_mean'] > 0.9049}`.",
        "- The comparison uses ten-fold mean values, not pooled OOF AUC.", "",
        "## Answers to the twelve required questions", "",
        f"1. Best base parameters: lr={tuned_b0['spec']['lr']}, WD={tuned_b0['spec']['weight_decay']}, dropout={tuned_b0['spec']['dropout']}.",
        f"2. TUNED_B0: {tuned_b0['metrics']['correct']}/535, ACC={tuned_b0['metrics']['acc']:.7f}, pooled ROC-AUC={tuned_b0['metrics']['roc_auc']:.7f}.",
        f"3. Best private configuration: rank={tuned_b1['spec']['rank']}, adapter LR multiplier={tuned_b1['spec']['adapter_lr_multiplier']}.",
        f"4. TUNED_B1: {tuned_b1['metrics']['correct']}/535, ACC={tuned_b1['metrics']['acc']:.7f}, pooled ROC-AUC={tuned_b1['metrics']['roc_auc']:.7f}.",
        f"5. Reached 523/535: `{tuned_b1['metrics']['correct'] >= 523 or tuned_b0['metrics']['correct'] >= 523}`.",
        f"6. TUNED_B1 fold mean ACC exceeds 97.57%: `{tuned_b1['fold_acc_mean'] > 0.9757}` ({tuned_b1['fold_acc_mean']:.7f}).",
        f"7. TUNED_B1 fold mean ROC-AUC exceeds 90.49%: `{tuned_b1['fold_roc_auc_mean'] > 0.9049}` ({tuned_b1['fold_roc_auc_mean']:.7f}).",
        f"8. TUNED_B1 minority metrics: TP={tuned_b1['metrics']['tp']}, SEN={tuned_b1['metrics']['sen']:.7f}, BACC={tuned_b1['metrics']['bacc']:.7f}, PR-AUC={tuned_b1['metrics']['pr_auc']:.7f}; registered safety={tuned_b1['safety']['safe']}.",
        f"9. Main gain source: `{'private adapter' if comparisons['tuned_b1_vs_tuned_b0']['correct_delta'] > 0 else 'base tuning or neither'}`.",
        f"10. TUNED_B1 truly exceeds TUNED_B0: `{decision['tuned_b1_outperforms_tuned_b0']}`; paired repairs/damages={comparisons['tuned_b1_vs_tuned_b0']['repairs']}/{comparisons['tuned_b1_vs_tuned_b0']['damages']}.",
        f"11. Model-structure change still needed: `{decision['decision'] in ('TAD_BINARY_NEAR', 'TAD_BINARY_TUNE_NO_GAIN')}`.",
        f"12. Final Decision: `{decision['decision']}`; target status `{decision['target_status']}`.", "",
        "## Reproduction", "",
        f"Run from a clean local branch named `{BRANCH}` at source commit `{runtime['source_commit']}`:", "", "```text",
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py inspect",
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py smoke --device cuda:0",
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py search --device cuda:0",
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py formal --device cuda:0", "```", "",
    ]
    return "\n".join(lines)


def run_formal(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Formal requires cuda:0")
    started = time.perf_counter()
    source = source_gate()
    runtime = runtime_lock(source)
    validate_smoke(runtime)
    base_context = build_context(torch.device(device_text))
    registry, results = validate_search(runtime, base_context)
    tuned_b0 = results[registry["tuned_b0_trial_id"]]
    tuned_b1 = results[registry["tuned_b1_trial_id"]]
    hist_b0, hist_b0_rows = historical_reference("B0")
    hist_b1, hist_b1_rows = historical_reference("B1")
    tuned_b0_rows = parse_trial_rows(trial_dir(tuned_b0["spec"]) / "oof_predictions.csv")
    tuned_b1_rows = parse_trial_rows(trial_dir(tuned_b1["spec"]) / "oof_predictions.csv")
    comparisons = {
        "tuned_b0_vs_historical_b0": paired_rows(hist_b0, hist_b0_rows, tuned_b0, tuned_b0_rows),
        "tuned_b1_vs_historical_b1": paired_rows(hist_b1, hist_b1_rows, tuned_b1, tuned_b1_rows),
        "tuned_b1_vs_tuned_b0": paired_rows(tuned_b0, tuned_b0_rows, tuned_b1, tuned_b1_rows),
    }
    decision = final_decision(tuned_b0, tuned_b1, comparisons)
    atomic_write_csv(RESULT_DIR / "tuned_b0_oof_predictions.csv", tuned_b0_rows)
    atomic_write_csv(RESULT_DIR / "tuned_b1_oof_predictions.csv", tuned_b1_rows)
    formal_core = {
        "runtime_lock": runtime,
        "smoke_report_sha256": file_sha256(SMOKE_DIR / "smoke_report.json"),
        "search_summary_sha256": file_sha256(SEARCH_SUMMARY_PATH),
        "trial_manifest_sha256": file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": file_sha256(ALL_TRIALS_PATH),
        "mode": "aggregate only; no completed trial retrained",
    }
    atomic_write_json(RESULT_DIR / "formal_config.json", {**formal_core, "sha256": payload_sha256(formal_core)})
    atomic_write_json(RESULT_DIR / "comparisons.json", comparisons)
    atomic_write_json(RESULT_DIR / "mechanism_diagnostics.json", {"tuned_b1_trial_id": tuned_b1["trial_id"], "mechanism": tuned_b1.get("mechanism")})
    elapsed = time.perf_counter() - started
    report = render_report(runtime, registry, results, {"B0": hist_b0, "B1": hist_b1}, comparisons, decision, elapsed)
    atomic_write_text(RESULT_DIR / "REPORT.md", report)
    commands = (
        f"# Reproduce from source commit {runtime['source_commit']} on local branch {BRANCH}\n"
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py inspect\n"
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py smoke --device cuda:0\n"
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py search --device cuda:0\n"
        "python -u -B scripts/run_tad_binary_hparam_search_v1.py formal --device cuda:0\n"
    )
    atomic_write_text(RESULT_DIR / "reproduction_commands.txt", commands)
    summary_core = {
        "experiment": EXPERIMENT_ID, "runtime_lock": runtime,
        "historical_b0": hist_b0, "historical_b1": hist_b1,
        "tuned_b0": tuned_b0, "tuned_b1": tuned_b1,
        "comparisons": comparisons, "decision": decision,
        "registered_trial_count": len(results), "failure_count": 0,
        "formal_aggregation_seconds": float(elapsed),
        "trial_manifest_sha256": file_sha256(TRIAL_MANIFEST_PATH),
        "all_trials_sha256": file_sha256(ALL_TRIALS_PATH),
        "report_sha256": file_sha256(RESULT_DIR / "REPORT.md"),
        "reproduction_commands_sha256": file_sha256(RESULT_DIR / "reproduction_commands.txt"),
        "tuned_b0_oof_sha256": file_sha256(RESULT_DIR / "tuned_b0_oof_predictions.csv"),
        "tuned_b1_oof_sha256": file_sha256(RESULT_DIR / "tuned_b1_oof_predictions.csv"),
    }
    summary = {**summary_core, "sha256": payload_sha256(summary_core)}
    atomic_write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps({"formal": "COMPLETE", "decision": decision["decision"], "tuned_b0": tuned_b0["metrics"], "tuned_b1": tuned_b1["metrics"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inspect", help="Inspect and freeze historical protocol/reuse anchors")
    smoke = subparsers.add_parser("smoke", help="Run fold-0 three-epoch CUDA smoke")
    smoke.add_argument("--device", default="cuda:0")
    search = subparsers.add_parser("search", help="Run or strictly resume the registered staged search")
    search.add_argument("--device", default="cuda:0")
    formal = subparsers.add_parser("formal", help="Strictly validate and aggregate completed search")
    formal.add_argument("--device", default="cuda:0")
    resume = subparsers.add_parser("resume", help="Resume the same locked registered search")
    resume.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        run_inspect()
    elif args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode in ("search", "resume"):
        run_search(args.device)
    elif args.mode == "formal":
        run_formal(args.device)
    else:
        raise InvariantError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
