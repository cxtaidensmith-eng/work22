"""Stage-A branch-complementarity probe for PC-BBF v1.

This script is deliberately read-only with respect to the historical C1 run. It
loads the exact C1 implementation and ten local checkpoints from the sibling
``cme_dual_branch_v1`` worktree, verifies that checkpoint replay reproduces the
formal C1 OOF predictions, and fits fixed fold-local linear probes to the two
pre-addition representations Y (Category) and G (Global).

It does not train C1 and contains no PC-BBF implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils import Config_, SET_Random, load_dataset, load_path


BASE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
EXPECTED_SOURCE_HEAD = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
SOURCE_ARTIFACT_REL = Path(
    "experiments/cme_dual_branch_v1/c1_shared_private_control_2"
)
OUTPUT_REL = Path("experiments/pc_bbf_c1_v1")
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
EXPECTED_C1_PARAMETERS = 862_971
EXPECTED_C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457,
    "bacc": 0.9163359,
    "macro_auc": 0.9585608,
    "weighted_f1": 0.9363660,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def metrics(truth: np.ndarray, probability: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    require(probability.shape == (truth.size, 3), "Probability shape mismatch")
    require(np.isfinite(probability).all(), "Probability contains NaN/Inf")
    require(
        np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
        "Probability rows do not sum to one",
    )
    prediction = probability.argmax(axis=1)
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "correct": int(np.sum(prediction == truth)),
        "acc": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(onehot, probability)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=np.arange(3)
        ).tolist(),
    }


def discover_source(source_root: Path) -> dict:
    source_root = source_root.resolve()
    artifact_root = source_root / SOURCE_ARTIFACT_REL
    model_path = source_root / "Model/cme_dual_branch.py"
    network_path = source_root / "Model/network.py"
    target_network_path = ROOT / "Model/network.py"
    oof_path = artifact_root / "oof_predictions.csv"
    checkpoints = [
        artifact_root / f"fold_{fold:02d}/checkpoint_best.pt" for fold in range(10)
    ]
    for path in (model_path, network_path, target_network_path, oof_path, *checkpoints):
        require(path.is_file(), f"Required C1 source artifact missing: {path}")

    source_head = git_value(source_root, "rev-parse", "HEAD")
    require(
        source_head == EXPECTED_SOURCE_HEAD,
        f"Unexpected C1 source HEAD: {source_head}",
    )
    require(
        git_value(source_root, "diff", "--name-only", "HEAD", "--", "Model/cme_dual_branch.py")
        == "",
        "C1 model source has uncommitted changes",
    )
    require(
        sha256(network_path) == sha256(target_network_path),
        "Target base Model/network.py differs from the C1 source dependency",
    )
    return {
        "source_root": source_root,
        "artifact_root": artifact_root,
        "model_path": model_path,
        "oof_path": oof_path,
        "checkpoints": checkpoints,
        "source_head": source_head,
        "model_sha256": sha256(model_path),
        "network_sha256": sha256(network_path),
        "oof_sha256": sha256(oof_path),
        "checkpoint_sha256": {
            f"fold_{fold:02d}": sha256(path)
            for fold, path in enumerate(checkpoints)
        },
    }


def import_c1_model(model_path: Path):
    module_name = "Model.cme_dual_branch_pc_bbf_probe"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    require(spec is not None and spec.loader is not None, "Cannot load C1 model spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CMEDualBranchModel


def load_context(device: torch.device) -> dict:
    config_root = Path(tempfile.gettempdir()) / "work22_pc_bbf_probe_config"
    config = Config_(str(config_root), str(ROOT / CONFIG_REL), 0)
    config.Device = device
    require(config.DATA_SET == "TADPOLE", "Dataset protocol changed")
    require(config.Task == "AD_CN_SMCI", "Task protocol changed")
    require(int(config.epochs) == 400 and int(config.T_max) == 400, "Epoch protocol changed")
    SET_Random(0)
    feature_path, dictionary_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
    require(tuple(class_names) == CLASS_NAMES, f"Class order changed: {class_names}")
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dictionary_path,
        device,
        class_names,
        config.Shuffle,
        0,
        train_size=config.train_size,
    )
    require(tuple(dataset_data["Feature"].shape) == (598, 360), "Data shape changed")
    require(len(dataset_data["Mask"]) == 10, "Fold count changed")
    require(len(dataset_dict["Modal_Name"]) == 6, "Modality count changed")
    modal_alias = {
        "MRI": "MRI",
        "PET": "PET",
        "CSF": "CSF",
        "RISK_FACTOR": "Risk",
        "RISK": "Risk",
        "COGNITIVE_TEST": "COG",
        "COG": "COG",
        "ROI_AVERAGE": "ROI",
        "ROI": "ROI",
    }
    actual_modalities = tuple(
        modal_alias.get(str(value).upper(), str(value))
        for value in dataset_dict["Modal_Name"]
    )
    require(actual_modalities == MODALITY_NAMES, f"Modality order changed: {actual_modalities}")
    return {
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
        "device": device,
    }


def build_c1_model(context: dict, model_class):
    config = context["config"]
    SET_Random(0)
    model = model_class(
        context["dataset_dict"],
        Herter_Graph=None,
        Hidden_size=config.Hidden_size,
        Drop_rate=config.Drop_rate,
        K=config.ChebGCN_K,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        input_noise_std=config.input_noise_std,
        drop_path=config.drop_path,
        graph_head=config.Graph_head,
        graph_layers=config.graph_layers,
        graph_heads=config.graph_heads,
        graph_beta=config.graph_beta,
        graph_k_order=config.graph_k_order,
        graph_alpha=config.graph_alpha,
        graph_kernel=config.graph_kernel,
        graph_use_graph=False,
        graph_dropout=config.graph_dropout,
        graph_hidden=config.graph_hidden,
        global_word_emb=config.global_word_emb,
        semantic_branch="both",
        semantic_fusion="add",
        category_branch_variant="original",
        query_pool_variant="independent",
        category_branch_fusion="concat",
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
        cme_arm="c1",
        adapter_rank=8,
        router_hidden=16,
        modality_embedding_dim=8,
    ).to(context["device"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(
        parameter_count == EXPECTED_C1_PARAMETERS,
        f"C1 parameter count changed: {parameter_count}",
    )
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph is enabled")
    return model


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), f"Unexpected checkpoint payload: {path}")
    require(payload and all(torch.is_tensor(value) for value in payload.values()), f"Checkpoint is not a raw state dict: {path}")
    return payload


def adjusted_probability(
    raw_logits: torch.Tensor, label_weight: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = raw_logits - tau * label_weight.to(raw_logits).clamp_min(1e-8).log().view(1, -1)
    return adjusted, torch.softmax(adjusted, dim=-1)


def fit_probe(
    train_representation: np.ndarray,
    train_truth: np.ndarray,
    test_representation: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=0,
                    class_weight=None,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(train_representation, train_truth)
    classifier = probe.named_steps["classifier"]
    require(
        np.array_equal(classifier.classes_, np.arange(3)),
        f"Probe class order changed: {classifier.classes_}",
    )
    probability = probe.predict_proba(test_representation)
    warning_messages = [str(item.message) for item in caught]
    return probability, warning_messages


def validate_formal_oof(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    require(len(rows) == 598, f"C1 OOF row count changed: {len(rows)}")
    by_subject = {int(row["subject_index"]): row for row in rows}
    require(len(by_subject) == 598, "C1 OOF subject indices are not unique")
    truth = np.asarray([int(by_subject[index]["truth"]) for index in sorted(by_subject)])
    probability = np.asarray(
        [
            [
                float(by_subject[index]["probability_AD"]),
                float(by_subject[index]["probability_CN"]),
                float(by_subject[index]["probability_SMCI"]),
            ]
            for index in sorted(by_subject)
        ]
    )
    observed = metrics(truth, probability)
    require(observed["correct"] == EXPECTED_C1["correct"], "C1 correct anchor changed")
    require(observed["confusion_matrix"] == EXPECTED_C1["confusion_matrix"], "C1 confusion anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(observed[key] - EXPECTED_C1[key]) <= 5e-7, f"C1 {key} anchor changed: {observed[key]}")
    return by_subject


def run_probe(source: dict, context: dict, output_dir: Path) -> dict:
    started = time.perf_counter()
    formal_rows = read_csv(source["oof_path"])
    formal_by_subject = validate_formal_oof(formal_rows)
    model_class = import_c1_model(source["model_path"])
    dataset_dict = context["dataset_dict"]
    dataset_data = context["dataset_data"]
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)
    label_weight = dataset_dict["Label_Weight"]
    tau = float(context["config"].logit_adjust_tau)

    oof_rows: list[dict] = []
    fold_reports: list[dict] = []
    representation_rows: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    max_fusion_diff = 0.0
    max_raw_logit_diff = 0.0
    max_probability_diff = 0.0
    convergence_warnings: list[dict] = []

    for fold in range(10):
        model = build_c1_model(context, model_class)
        state = load_state(source["checkpoints"][fold])
        incompatible = model.load_state_dict(state, strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, "Strict checkpoint load failed")
        model.eval()
        with torch.inference_mode():
            raw_logits, _, _, intermediates = model(features, return_intermediates=True)
            y_representation = intermediates["Y"]
            g_representation = intermediates["G"]
            fused_representation = intermediates["H_fused"]
            fold_fusion_diff = float(
                (fused_representation - (y_representation + g_representation))
                .abs()
                .max()
                .detach()
                .cpu()
            )
            adjusted, c1_probability = adjusted_probability(raw_logits, label_weight, tau)
        require(fold_fusion_diff <= 1e-6, f"fold{fold}: H_fused != Y + G")
        max_fusion_diff = max(max_fusion_diff, fold_fusion_diff)

        train_mask, test_mask = dataset_data["Mask"][fold]
        train_numpy = train_mask.detach().cpu().numpy().astype(bool)
        test_numpy = test_mask.detach().cpu().numpy().astype(bool)
        train_truth = labels[train_mask].detach().cpu().numpy().astype(np.int64)
        test_truth = labels[test_mask].detach().cpu().numpy().astype(np.int64)
        test_subjects = source_indices[test_numpy]
        y_numpy = y_representation.detach().cpu().numpy()
        g_numpy = g_representation.detach().cpu().numpy()

        category_probability, category_warnings = fit_probe(
            y_numpy[train_numpy], train_truth, y_numpy[test_numpy]
        )
        global_probability, global_warnings = fit_probe(
            g_numpy[train_numpy], train_truth, g_numpy[test_numpy]
        )
        convergence_warnings.extend(
            {"fold": fold, "branch": branch, "message": message}
            for branch, messages in (("category", category_warnings), ("global", global_warnings))
            for message in messages
        )

        test_raw = raw_logits[test_mask].detach().cpu().numpy()
        test_adjusted = adjusted[test_mask].detach().cpu().numpy()
        test_c1_probability = c1_probability[test_mask].detach().cpu().numpy()
        category_prediction = category_probability.argmax(axis=1)
        global_prediction = global_probability.argmax(axis=1)
        fold_raw_diff = 0.0
        fold_probability_diff = 0.0
        for row_index, subject_index in enumerate(test_subjects):
            subject = int(subject_index)
            formal = formal_by_subject[subject]
            require(int(formal["fold"]) == fold, f"fold{fold}: formal fold mismatch for subject {subject}")
            require(int(formal["truth"]) == int(test_truth[row_index]), f"fold{fold}: truth mismatch for subject {subject}")
            formal_raw = np.asarray(
                [float(formal[f"raw_logit_{name}"]) for name in CLASS_NAMES]
            )
            formal_probability = np.asarray(
                [float(formal[f"probability_{name}"]) for name in CLASS_NAMES]
            )
            raw_diff = float(np.max(np.abs(test_raw[row_index] - formal_raw)))
            probability_diff = float(
                np.max(np.abs(test_c1_probability[row_index] - formal_probability))
            )
            fold_raw_diff = max(fold_raw_diff, raw_diff)
            fold_probability_diff = max(fold_probability_diff, probability_diff)
            require(
                int(test_c1_probability[row_index].argmax()) == int(formal["prediction"]),
                f"fold{fold}: replay prediction mismatch for subject {subject}",
            )
            representation_rows.append(
                (
                    subject,
                    fold,
                    int(test_truth[row_index]),
                    y_numpy[test_numpy][row_index].copy(),
                    g_numpy[test_numpy][row_index].copy(),
                )
            )
            row = {
                "fold": fold,
                "subject_index": subject,
                "truth": int(test_truth[row_index]),
                "c1_prediction": int(test_c1_probability[row_index].argmax()),
                "category_prediction": int(category_prediction[row_index]),
                "global_prediction": int(global_prediction[row_index]),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"c1_raw_logit_{class_name}"] = float(test_raw[row_index, class_index])
                row[f"c1_adjusted_score_{class_name}"] = float(test_adjusted[row_index, class_index])
                row[f"c1_probability_{class_name}"] = float(test_c1_probability[row_index, class_index])
                row[f"category_probability_{class_name}"] = float(category_probability[row_index, class_index])
                row[f"global_probability_{class_name}"] = float(global_probability[row_index, class_index])
            oof_rows.append(row)

        require(fold_raw_diff <= 1e-5, f"fold{fold}: checkpoint raw-logit replay mismatch {fold_raw_diff}")
        require(fold_probability_diff <= 1e-6, f"fold{fold}: checkpoint probability replay mismatch {fold_probability_diff}")
        max_raw_logit_diff = max(max_raw_logit_diff, fold_raw_diff)
        max_probability_diff = max(max_probability_diff, fold_probability_diff)
        fold_reports.append(
            {
                "fold": fold,
                "train_size": int(train_numpy.sum()),
                "test_size": int(test_numpy.sum()),
                "checkpoint": str(source["checkpoints"][fold]),
                "checkpoint_sha256": source["checkpoint_sha256"][f"fold_{fold:02d}"],
                "strict_load": True,
                "h_fused_equals_y_plus_g_max_abs_diff": fold_fusion_diff,
                "replay_raw_logit_max_abs_diff": fold_raw_diff,
                "replay_probability_max_abs_diff": fold_probability_diff,
                "category_probe": metrics(test_truth, category_probability),
                "global_probe": metrics(test_truth, global_probability),
            }
        )
        del model, state, raw_logits, adjusted, c1_probability, intermediates
        if context["device"].type == "cuda":
            torch.cuda.empty_cache()

    require(len(oof_rows) == 598, "Probe OOF does not contain 598 rows")
    oof_rows.sort(key=lambda row: row["subject_index"])
    require(
        [row["subject_index"] for row in oof_rows] == sorted(formal_by_subject),
        "Probe and formal C1 subject sets differ",
    )
    truth = np.asarray([row["truth"] for row in oof_rows], dtype=np.int64)
    c1_probability = np.asarray(
        [[row[f"c1_probability_{name}"] for name in CLASS_NAMES] for row in oof_rows]
    )
    category_probability = np.asarray(
        [[row[f"category_probability_{name}"] for name in CLASS_NAMES] for row in oof_rows]
    )
    global_probability = np.asarray(
        [[row[f"global_probability_{name}"] for name in CLASS_NAMES] for row in oof_rows]
    )
    c1_prediction = c1_probability.argmax(axis=1)
    category_prediction = category_probability.argmax(axis=1)
    global_prediction = global_probability.argmax(axis=1)
    c1_correct = c1_prediction == truth
    category_correct = category_prediction == truth
    global_correct = global_prediction == truth

    category_only = category_correct & ~global_correct
    global_only = global_correct & ~category_correct
    c1_error = ~c1_correct
    fixed_by_category = c1_error & category_correct
    fixed_by_global = c1_error & global_correct
    fixed_by_either = fixed_by_category | fixed_by_global
    oracle_correct = c1_correct | category_correct | global_correct
    disagreement = category_prediction != global_prediction
    subject_array = np.asarray([row["subject_index"] for row in oof_rows])

    representation_rows.sort(key=lambda item: item[0])
    require([item[0] for item in representation_rows] == subject_array.tolist(), "Representation order mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_path = output_dir / "branch_representations_oof.npz"
    np.savez_compressed(
        representation_path,
        subject_index=np.asarray([item[0] for item in representation_rows], dtype=np.int64),
        fold=np.asarray([item[1] for item in representation_rows], dtype=np.int64),
        truth=np.asarray([item[2] for item in representation_rows], dtype=np.int64),
        Y=np.stack([item[3] for item in representation_rows]),
        G=np.stack([item[4] for item in representation_rows]),
    )
    oof_path = output_dir / "branch_probe_oof.csv"
    write_csv(oof_path, oof_rows)

    complementarity = {
        "category_only_correct_global_wrong": int(category_only.sum()),
        "global_only_correct_category_wrong": int(global_only.sum()),
        "category_only_subjects": subject_array[category_only].tolist(),
        "global_only_subjects": subject_array[global_only].tolist(),
        "c1_errors_fixed_by_category": int(fixed_by_category.sum()),
        "c1_errors_fixed_by_global": int(fixed_by_global.sum()),
        "c1_errors_fixed_by_either": int(fixed_by_either.sum()),
        "c1_error_subjects_fixed_by_either": subject_array[fixed_by_either].tolist(),
        "oracle_c1_category_global_correct": int(oracle_correct.sum()),
        "category_global_disagreement_count": int(disagreement.sum()),
        "category_global_disagreement_fraction": float(disagreement.mean()),
    }
    criteria = {
        "oracle_at_least_565": complementarity["oracle_c1_category_global_correct"] >= 565,
        "category_has_at_least_3_exclusive_correct": int(category_only.sum()) >= 3,
        "global_has_at_least_3_exclusive_correct": int(global_only.sum()) >= 3,
        "at_least_5_c1_errors_fixed_by_either": int(fixed_by_either.sum()) >= 5,
    }
    decision = "BRANCH_COMPLEMENTARITY_GO" if all(criteria.values()) else "BRANCH_COMPLEMENTARITY_STOP"
    report = {
        "decision": decision,
        "stage": "A_category_global_fusibility_check",
        "pc_bbf_implemented": False,
        "c1_retrained": False,
        "base_commit": BASE_COMMIT,
        "source": {
            "worktree": str(source["source_root"]),
            "head": source["source_head"],
            "model_path": str(source["model_path"]),
            "model_sha256": source["model_sha256"],
            "network_sha256": source["network_sha256"],
            "formal_oof_path": str(source["oof_path"]),
            "formal_oof_sha256": source["oof_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"],
        },
        "protocol": {
            "dataset": "TADPOLE",
            "task": "AD_CN_SMCI",
            "seed": 0,
            "folds": list(range(10)),
            "class_order": list(CLASS_NAMES),
            "probe": {
                "scaler": "StandardScaler fitted on fold train representation only",
                "classifier": "LogisticRegression",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 0,
                "class_weight": None,
            },
        },
        "representation_contract": {
            "Y": "Category representation (Message_MLP output) before historical add",
            "G": "Global representation (Global_Message output) before historical add",
            "historical_fusion": "H_fused = Y + G",
            "max_abs_diff_h_fused_vs_y_plus_g": max_fusion_diff,
            "dimension_Y": int(representation_rows[0][3].shape[0]),
            "dimension_G": int(representation_rows[0][4].shape[0]),
        },
        "checkpoint_replay": {
            "strict_load_all_folds": True,
            "prediction_match_all_598": True,
            "max_abs_raw_logit_diff": max_raw_logit_diff,
            "max_abs_probability_diff": max_probability_diff,
        },
        "metrics": {
            "c1_formal_replayed": metrics(truth, c1_probability),
            "category_probe": metrics(truth, category_probability),
            "global_probe": metrics(truth, global_probability),
        },
        "complementarity": complementarity,
        "decision_criteria": criteria,
        "folds": fold_reports,
        "convergence_warnings": convergence_warnings,
        "artifacts": {
            "probe_oof": str(oof_path.relative_to(ROOT)),
            "representations_oof": str(representation_path.relative_to(ROOT)),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "branch_complementarity_report.json", report)
    write_markdown(output_dir / "branch_complementarity_report.md", report)
    return report


def format_metric_line(name: str, payload: dict) -> str:
    return (
        f"- {name}: Correct={payload['correct']}/598, ACC={payload['acc']:.7f}, "
        f"Macro-F1={payload['macro_f1']:.7f}, BACC={payload['bacc']:.7f}, "
        f"Probability Macro-AUC={payload['macro_auc']:.7f}, "
        f"Weighted-F1={payload['weighted_f1']:.7f}, "
        f"Confusion={payload['confusion_matrix']}"
    )


def write_markdown(path: Path, report: dict) -> None:
    metric_payload = report["metrics"]
    complementarity = report["complementarity"]
    criteria = report["decision_criteria"]
    lines = [
        "# PC-BBF v1 Stage-A Category/Global Complementarity",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "C1 was not retrained and PC-BBF was not implemented in this stage. Each formal C1 "
        "checkpoint was loaded strictly, and Y/G are the exact two inputs to the historical Y + G add.",
        "",
        "## Checkpoint replay and representation contract",
        "",
        f"- C1 source HEAD: `{report['source']['head']}`",
        f"- Strict fold loads: `{report['checkpoint_replay']['strict_load_all_folds']}`",
        f"- Predictions matched formal C1 OOF: `{report['checkpoint_replay']['prediction_match_all_598']}`",
        f"- Maximum raw-logit difference: `{report['checkpoint_replay']['max_abs_raw_logit_diff']:.3e}`",
        f"- Maximum probability difference: `{report['checkpoint_replay']['max_abs_probability_diff']:.3e}`",
        f"- Maximum `H_fused - (Y + G)` difference: `{report['representation_contract']['max_abs_diff_h_fused_vs_y_plus_g']:.3e}`",
        "",
        "## Pooled OOF metrics",
        "",
        format_metric_line("C1", metric_payload["c1_formal_replayed"]),
        format_metric_line("Category-only probe", metric_payload["category_probe"]),
        format_metric_line("Global-only probe", metric_payload["global_probe"]),
        "",
        "## Complementarity",
        "",
        f"- Category-only correct / Global wrong: {complementarity['category_only_correct_global_wrong']}",
        f"- Global-only correct / Category wrong: {complementarity['global_only_correct_category_wrong']}",
        f"- C1 errors fixed by Category: {complementarity['c1_errors_fixed_by_category']}",
        f"- C1 errors fixed by Global: {complementarity['c1_errors_fixed_by_global']}",
        f"- C1 errors fixed by either branch: {complementarity['c1_errors_fixed_by_either']}",
        f"- Oracle(C1, Category, Global): {complementarity['oracle_c1_category_global_correct']}/598",
        f"- Category/Global prediction disagreement: {complementarity['category_global_disagreement_count']}/598 "
        f"({complementarity['category_global_disagreement_fraction']:.3%})",
        "",
        "## Decision criteria",
        "",
        *[f"- {name}: `{value}`" for name, value in criteria.items()],
        "",
        "If the decision is `BRANCH_COMPLEMENTARITY_STOP`, PC-BBF must not be implemented or trained; "
        "the prescribed next recommendation is MR-HGR-C1 multi-relation graph.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT.parent / "cme_dual_branch_v1",
        help="Read-only sibling worktree containing exact C1 code/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_REL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source paths and provenance without loading data/checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(git_value(ROOT, "rev-parse", "HEAD") == BASE_COMMIT, "Target worktree is not at the locked base commit")
    source = discover_source(args.source_root)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_READY",
                    "source_head": source["source_head"],
                    "model_path": str(source["model_path"]),
                    "formal_oof_path": str(source["oof_path"]),
                    "checkpoint_count": len(source["checkpoints"]),
                },
                indent=2,
            )
        )
        return
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    context = load_context(device)
    report = run_probe(source, context, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "category_correct": report["metrics"]["category_probe"]["correct"],
                "global_correct": report["metrics"]["global_probe"]["correct"],
                "oracle_correct": report["complementarity"]["oracle_c1_category_global_correct"],
                "c1_errors_fixed_by_either": report["complementarity"]["c1_errors_fixed_by_either"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
