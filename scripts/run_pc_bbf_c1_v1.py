"""Run Patient-Conditioned Bounded Branch Fusion v1 on the locked C1 protocol.

Stages are explicit: ``smoke`` (fold0, three epochs), ``screen`` (folds 4/7/8),
and ``formal`` (fresh folds 0..9).  ``all`` follows the prescribed decision
gates and never starts formal training unless the difficult-fold screen passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Model.pc_bbf import (
    C1SharedPrivateControlModel,
    PC_BBF_EXPECTED_PARAMETERS,
    PCBBFModel,
)
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_pc_bbf_branch_probe as branch_probe


BASE_COMMIT = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_REL = Path("experiments/pc_bbf_c1_v1")
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
SCREEN_FOLDS = (4, 7, 8)
FORMAL_FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
EPSILON = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def module_gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def gradients_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def probability_metrics(truth, probability) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    require(probability.shape == (len(truth), 3), "Probability shape mismatch")
    require(np.isfinite(probability).all(), "Probability contains NaN/Inf")
    require(np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0), "Probability sum mismatch")
    prediction = probability.argmax(axis=1)
    onehot = np.eye(3, dtype=np.int64)[truth]
    return {
        "correct": int(np.sum(prediction == truth)),
        "acc": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(onehot, probability)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(3)).tolist(),
    }


def score_logits(raw: torch.Tensor, label_weight: torch.Tensor, tau: float):
    adjusted = raw - tau * label_weight.to(raw).clamp_min(1e-8).log().view(1, -1)
    probability = torch.softmax(adjusted, dim=-1)
    return adjusted, probability, probability.argmax(dim=-1)


def selection_metrics(raw, labels, mask, label_weight, tau):
    _, probability, _ = score_logits(raw, label_weight, tau)
    payload = probability_metrics(
        labels[mask].detach().cpu().numpy(), probability[mask].detach().cpu().numpy()
    )
    return payload, (payload["acc"], payload["macro_auc"], payload["macro_f1"])


def load_context(device: torch.device) -> dict:
    config = Config_(
        str(Path(tempfile.gettempdir()) / "work22_pc_bbf_config"),
        str(ROOT / CONFIG_REL),
        0,
    )
    config.Device = device
    require(config.DATA_SET == "TADPOLE" and config.Task == "AD_CN_SMCI", "Protocol changed")
    require(int(config.epochs) == EPOCHS and int(config.T_max) == EPOCHS, "Epoch protocol changed")
    SET_Random(SEED)
    feature_path, dictionary_path, _, class_names = load_path(str(ROOT), config.DATA_SET, config.Task)
    require(tuple(class_names) == CLASS_NAMES, f"Class order changed: {class_names}")
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dictionary_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    require(tuple(dataset_data["Feature"].shape) == (598, 360), "Dataset shape changed")
    require(len(dataset_data["Mask"]) == 10, "Fold count changed")
    return {
        "device": device,
        "config": config,
        "dataset_dict": dataset_dict,
        "dataset_data": dataset_data,
    }


def model_kwargs(context: dict) -> dict:
    config = context["config"]
    return {
        "Herter_Graph": None,
        "Hidden_size": config.Hidden_size,
        "Drop_rate": config.Drop_rate,
        "K": config.ChebGCN_K,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "input_noise_std": config.input_noise_std,
        "drop_path": config.drop_path,
        "graph_head": config.Graph_head,
        "graph_layers": config.graph_layers,
        "graph_heads": config.graph_heads,
        "graph_beta": config.graph_beta,
        "graph_k_order": config.graph_k_order,
        "graph_alpha": config.graph_alpha,
        "graph_kernel": config.graph_kernel,
        "graph_use_graph": False,
        "graph_dropout": config.graph_dropout,
        "graph_hidden": config.graph_hidden,
        "global_word_emb": config.global_word_emb,
        "semantic_branch": "both",
        "semantic_fusion": "add",
        "category_branch_variant": "original",
        "query_pool_variant": "independent",
        "category_branch_fusion": "concat",
        "adj_mode": "none",
        "label_graph_alpha": 0.0,
        "label_graph_topk": 0,
        "label_graph_reg_lambda": 0.0,
        "cme_arm": "c1",
        "adapter_rank": 8,
    }


def build_model(context: dict, cap_value: float, pc_bbf: bool = True):
    SET_Random(SEED)
    klass = PCBBFModel if pc_bbf else C1SharedPrivateControlModel
    kwargs = model_kwargs(context)
    if pc_bbf:
        kwargs.update({"fusion_rank": 8, "cap_value": cap_value})
    model = klass(context["dataset_dict"], **kwargs).to(context["device"])
    expected = PC_BBF_EXPECTED_PARAMETERS if pc_bbf else 862_971
    require(parameter_count(model) == expected, f"Parameter count changed: {parameter_count(model)}")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    return model


def optimizer_groups(model: PCBBFModel, context: dict) -> tuple[list[dict], dict]:
    historical, input_projection, output_and_gate = [], [], []
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if name.startswith("pc_bbf.input_projection."):
            input_projection.append(parameter)
        elif name.startswith("pc_bbf.residual_output.") or name.startswith("pc_bbf.gate_output."):
            output_and_gate.append(parameter)
        elif name.startswith("pc_bbf."):
            raise RuntimeError(f"Unclassified trainable PC-BBF parameter: {name}")
        else:
            historical.append(parameter)
    grouped = historical + input_projection + output_and_gate
    require(len({id(parameter) for parameter in grouped}) == len(grouped), "Duplicate optimizer parameter")
    require({id(parameter) for parameter in grouped} == {id(parameter) for parameter in model.parameters()}, "Optimizer coverage incomplete")
    config = context["config"]
    groups = [
        {
            "params": historical,
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "group_name": "historical_c1",
        },
        {
            "params": input_projection,
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "group_name": "pc_bbf_input_projection",
        },
        {
            "params": output_and_gate,
            "lr": float(config.lr),
            "weight_decay": 0.0,
            "group_name": "pc_bbf_output_and_gate",
        },
    ]
    audit = {
        "coverage": 1.0,
        "total_tensors": len(grouped),
        "total_parameters": int(sum(parameter.numel() for parameter in grouped)),
        "groups": [
            {
                "name": group["group_name"],
                "tensor_count": len(group["params"]),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "parameters": [names[id(parameter)] for parameter in group["params"]],
            }
            for group in groups
        ],
    }
    return groups, audit


def fresh_training_objects(context: dict, cap_value: float):
    model = build_model(context, cap_value, pc_bbf=True)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    groups, audit = optimizer_groups(model, context)
    optimizer = torch.optim.Adam(groups)
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=float(context["config"].Lr_Min)
    )
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")
    require(int(scheduler.T_max) == EPOCHS, "Scheduler T_max changed")
    return model, criterion, optimizer, scheduler, audit


def prediction_rows(context: dict, fold: int, raw: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    labels = context["dataset_data"]["Label"]
    adjusted, probability, prediction = score_logits(
        raw[mask],
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )
    raw_values = raw[mask].detach().cpu().numpy()
    adjusted_values = adjusted.detach().cpu().numpy()
    probability_values = probability.detach().cpu().numpy()
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    predictions = prediction.detach().cpu().numpy().astype(np.int64)
    source_indices = np.asarray(context["dataset_dict"]["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    rows = []
    for index, subject in enumerate(source_indices):
        row = {
            "fold": fold,
            "subject_index": int(subject),
            "truth": int(truth[index]),
            "prediction": int(predictions[index]),
        }
        for class_index, name in enumerate(CLASS_NAMES):
            row[f"raw_logit_{name}"] = float(raw_values[index, class_index])
            row[f"adjusted_score_{name}"] = float(adjusted_values[index, class_index])
            row[f"probability_{name}"] = float(probability_values[index, class_index])
        rows.append(row)
    return rows


def metrics_from_rows(rows: list[dict]) -> dict:
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    probability = np.asarray(
        [[float(row[f"probability_{name}"]) for name in CLASS_NAMES] for row in rows]
    )
    return probability_metrics(truth, probability)


def mechanism_statistics(intermediates: dict, mask: torch.Tensor) -> dict:
    gate = intermediates["pc_bbf_gate"][mask].detach().float()
    cap_scale = intermediates["pc_bbf_cap_scale"][mask].detach().float()
    shared_norm = intermediates["pc_bbf_shared_norm"][mask].detach().float()
    residual_capped = intermediates["pc_bbf_residual_capped"][mask].detach().float()
    applied = intermediates["pc_bbf_applied_residual"][mask].detach().float()
    capped_ratio = residual_capped.norm(dim=-1) / shared_norm.squeeze(-1).clamp_min(1e-8)
    applied_ratio = applied.norm(dim=-1) / shared_norm.squeeze(-1).clamp_min(1e-8)
    return {
        "gate_mean": float(gate.mean().cpu()),
        "gate_std": float(gate.std(unbiased=False).cpu()),
        "gate_min": float(gate.min().cpu()),
        "gate_max": float(gate.max().cpu()),
        "gate_saturation_fraction": float(((gate <= 0.05) | (gate >= 0.95)).float().mean().cpu()),
        "residual_shared_ratio_mean": float(capped_ratio.mean().cpu()),
        "residual_shared_ratio_max": float(capped_ratio.max().cpu()),
        "applied_residual_shared_ratio_mean": float(applied_ratio.mean().cpu()),
        "applied_residual_shared_ratio_max": float(applied_ratio.max().cpu()),
        "cap_saturation_fraction": float((cap_scale < 1.0 - 1e-7).float().mean().cpu()),
        "residual_raw_max_abs": float(intermediates["pc_bbf_residual_raw"][mask].abs().max().detach().cpu()),
    }


def run_smoke(context: dict, output_root: Path, cap_value: float = 0.10) -> dict:
    report_path = output_root / "smoke/smoke_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        require(report.get("passed") is True, "Existing smoke did not pass")
        return report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]

    c1 = build_model(context, cap_value, pc_bbf=False)
    candidate = build_model(context, cap_value, pc_bbf=True)
    c1.eval()
    candidate.eval()
    with torch.no_grad():
        c1_raw, _, _ = c1(features)
        initial_raw, _, _, initial_intermediates = candidate(features, return_intermediates=True)
    epoch0_diff = float((c1_raw - initial_raw).abs().max().cpu())
    require(epoch0_diff <= 1e-6, f"Epoch0 C1 equivalence failed: {epoch0_diff}")
    require(float(initial_intermediates["pc_bbf_residual_raw"].abs().max().cpu()) == 0.0, "Initial residual is nonzero")
    del c1, candidate, c1_raw, initial_raw, initial_intermediates

    model, criterion, optimizer, scheduler, optimizer_audit = fresh_training_objects(context, cap_value)
    before_output = model.pc_bbf.residual_output.weight.detach().clone()
    gradient_max = {"input_projection": 0.0, "residual_output": 0.0, "gate_output": 0.0}
    loss_values = []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Smoke epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"Smoke epoch{epoch}: non-finite gradient")
        gradient_max["input_projection"] = max(gradient_max["input_projection"], module_gradient_norm(model.pc_bbf.input_projection))
        gradient_max["residual_output"] = max(gradient_max["residual_output"], module_gradient_norm(model.pc_bbf.residual_output))
        gradient_max["gate_output"] = max(gradient_max["gate_output"], module_gradient_norm(model.pc_bbf.gate_output))
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        loss_values.append(float(loss.detach().cpu()))
    require(all(value > 0.0 and math.isfinite(value) for value in gradient_max.values()), f"Inactive PC-BBF gradients: {gradient_max}")
    output_delta = float((model.pc_bbf.residual_output.weight.detach() - before_output).abs().max().cpu())
    require(output_delta > 0.0, "Residual output did not update")
    model.eval()
    with torch.no_grad():
        reference_raw, _, _, reference_intermediates = model(features, return_intermediates=True)
    state = clone_cpu_state(model)
    checkpoint_path = report_path.parent / "checkpoint_roundtrip.pt"
    torch.save(state, checkpoint_path)
    reloaded = build_model(context, cap_value, pc_bbf=True)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        reloaded_raw, _, _, _ = reloaded(features, return_intermediates=True)
    roundtrip_diff = float((reference_raw - reloaded_raw).abs().max().cpu())
    require(roundtrip_diff <= 1e-7, f"Checkpoint roundtrip changed logits: {roundtrip_diff}")
    report = {
        "passed": True,
        "fold": 0,
        "epochs": 3,
        "cap_value": cap_value,
        "epoch0_c1_logit_max_abs_diff": epoch0_diff,
        "optimizer": optimizer_audit,
        "loss_values": loss_values,
        "pc_bbf_gradient_max": gradient_max,
        "residual_output_parameter_max_abs_delta": output_delta,
        "checkpoint_roundtrip_logit_max_abs_diff": roundtrip_diff,
        "mechanism_after_epoch3": mechanism_statistics(reference_intermediates, train_mask),
        "no_nan_inf": True,
    }
    write_json(report_path, report)
    print("SMOKE PASS", flush=True)
    return report


def fold_config(context: dict, fold: int, cap_value: float) -> dict:
    return {
        "experiment": "pc_bbf_c1_v1",
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "graph_enabled": False,
        "optimizer": "Adam",
        "lr": float(context["config"].lr),
        "historical_weight_decay": float(context["config"].weight_decay),
        "pc_bbf_output_gate_weight_decay": 0.0,
        "scheduler": "CustomCosineAnnealingLR",
        "T_max": EPOCHS,
        "eta_min": float(context["config"].Lr_Min),
        "loss": "historical weighted main CE + three C1 OVR auxiliary losses",
        "label_smoothing": 0.05,
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "fusion_rank": 8,
        "cap_value": cap_value,
    }


def run_fold(context: dict, fold: int, cap_value: float, output_root: Path) -> tuple[dict, list[dict]]:
    final_dir = output_root / f"fold_{fold:02d}"
    if final_dir.is_dir():
        summary = read_json(final_dir / "summary.json")
        rows = read_csv(final_dir / "best_predictions.csv")
        require(summary["fold"] == fold and abs(summary["cap_value"] - cap_value) < 1e-12, "Completed fold config mismatch")
        require(metrics_from_rows(rows) == summary["best_metrics"], "Completed fold readback mismatch")
        print(f"RESUME fold={fold} best_epoch={summary['best_epoch']} ACC={summary['best_metrics']['acc']:.7f}", flush=True)
        return summary, rows
    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    require(not staging_dir.exists(), f"Retained in-progress directory exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    model, criterion, optimizer, scheduler, optimizer_audit = fresh_training_objects(context, cap_value)
    best = None
    best_state = None
    epoch_rows = []
    gradient_max = {"input_projection": 0.0, "residual_output": 0.0, "gate_output": 0.0}
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(gradients_finite(model), f"fold{fold} epoch{epoch}: non-finite gradient")
        gradient_max["input_projection"] = max(gradient_max["input_projection"], module_gradient_norm(model.pc_bbf.input_projection))
        gradient_max["residual_output"] = max(gradient_max["residual_output"], module_gradient_norm(model.pc_bbf.residual_output))
        gradient_max["gate_output"] = max(gradient_max["gate_output"], module_gradient_norm(model.pc_bbf.gate_output))
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_raw, _, _, _ = model(features, return_intermediates=True)
            eval_metrics, score = selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or score > best["selection_tuple"]:
            best = {"epoch": epoch, "selection_tuple": score, "metrics": deepcopy(eval_metrics)}
            best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "acc": eval_metrics["acc"],
                "macro_f1": eval_metrics["macro_f1"],
                "bacc": eval_metrics["bacc"],
                "macro_auc": eval_metrics["macro_auc"],
                "weighted_f1": eval_metrics["weighted_f1"],
            }
        )
    require(best is not None and best_state is not None, f"fold{fold}: no best checkpoint")
    require(all(value > 0.0 for value in gradient_max.values()), f"fold{fold}: inactive PC-BBF gradient")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(features, return_intermediates=True)
        best_metrics, _ = selection_metrics(
            best_raw,
            labels,
            test_mask,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
    rows = prediction_rows(context, fold, best_raw, test_mask)
    require(best_metrics == best["metrics"] and metrics_from_rows(rows) == best_metrics, f"fold{fold}: best readback mismatch")
    summary = {
        "passed": True,
        "fold": fold,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "cap_value": cap_value,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(model),
        "added_parameters_vs_c1": parameter_count(model) - 862_971,
        "elapsed_seconds": float(time.perf_counter() - started),
        "optimizer": optimizer_audit,
        "gradient_max": gradient_max,
        "mechanism": mechanism_statistics(best_intermediates, test_mask),
        "config": fold_config(context, fold, cap_value),
    }
    torch.save(best_state, staging_dir / "checkpoint_best.pt")
    write_json(staging_dir / "summary.json", summary)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    write_csv(
        staging_dir / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(best_metrics["confusion_matrix"])
        ],
    )
    staging_dir.rename(final_dir)
    print(
        f"fold={fold} best_epoch={best['epoch']} correct={best_metrics['correct']} "
        f"ACC={best_metrics['acc']:.7f} Macro-F1={best_metrics['macro_f1']:.7f} "
        f"BACC={best_metrics['bacc']:.7f} Probability Macro-AUC={best_metrics['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def baseline_map(source: dict) -> dict[int, dict[str, str]]:
    rows = read_csv(source["oof_path"])
    require(len(rows) == 598, "C1 OOF missing rows")
    output = {int(row["subject_index"]): row for row in rows}
    require(len(output) == 598, "C1 OOF subject indices are not unique")
    return output


def compare_rows(candidate_rows: list[dict], c1_by_subject: dict[int, dict[str, str]]) -> dict:
    repairs, damages, changed = [], [], []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        baseline = c1_by_subject[subject]
        truth = int(row["truth"])
        require(int(baseline["truth"]) == truth and int(baseline["fold"]) == int(row["fold"]), "C1 pairing mismatch")
        old_prediction = int(baseline["prediction"])
        new_prediction = int(row["prediction"])
        if old_prediction != truth and new_prediction == truth:
            repairs.append(subject)
        if old_prediction == truth and new_prediction != truth:
            damages.append(subject)
        if old_prediction != new_prediction:
            changed.append(subject)
    return {
        "repairs": len(repairs),
        "damages": len(damages),
        "net_repairs": len(repairs) - len(damages),
        "changed_predictions": len(changed),
        "repair_subject_indices": repairs,
        "damage_subject_indices": damages,
        "changed_subject_indices": changed,
    }


def ad_cn_errors(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if {int(row["truth"]), int(row["prediction"])} == {0, 1}
    )


def boundary_errors(rows: list[dict]) -> dict[str, int]:
    output = {}
    for name, pair in {"AD_SMCI": {0, 2}, "CN_SMCI": {1, 2}, "AD_CN": {0, 1}}.items():
        output[name] = sum(
            1
            for row in rows
            if int(row["truth"]) != int(row["prediction"])
            and {int(row["truth"]), int(row["prediction"])} == pair
        )
    return output


def subset_baseline_rows(c1_by_subject: dict[int, dict[str, str]], folds: tuple[int, ...]) -> list[dict]:
    return [row for row in c1_by_subject.values() if int(row["fold"]) in folds]


def summarize_mechanism(summaries: list[dict]) -> dict:
    weights = np.asarray([summary["test_size"] for summary in summaries], dtype=np.float64)
    weights /= weights.sum()
    mechanisms = [summary["mechanism"] for summary in summaries]

    def weighted(key: str) -> float:
        return float(sum(weight * payload[key] for weight, payload in zip(weights, mechanisms)))

    gate_mean = weighted("gate_mean")
    gate_second_moment = sum(
        weight * (payload["gate_std"] ** 2 + payload["gate_mean"] ** 2)
        for weight, payload in zip(weights, mechanisms)
    )
    return {
        "gate_mean": gate_mean,
        "gate_std": float(math.sqrt(max(0.0, gate_second_moment - gate_mean**2))),
        "gate_min": float(min(payload["gate_min"] for payload in mechanisms)),
        "gate_max": float(max(payload["gate_max"] for payload in mechanisms)),
        "gate_saturation_fraction": weighted("gate_saturation_fraction"),
        "residual_shared_ratio_mean": weighted("residual_shared_ratio_mean"),
        "residual_shared_ratio_max": float(max(payload["residual_shared_ratio_max"] for payload in mechanisms)),
        "applied_residual_shared_ratio_mean": weighted("applied_residual_shared_ratio_mean"),
        "applied_residual_shared_ratio_max": float(max(payload["applied_residual_shared_ratio_max"] for payload in mechanisms)),
        "cap_saturation_fraction": weighted("cap_saturation_fraction"),
        "residual_raw_max_abs": float(max(payload["residual_raw_max_abs"] for payload in mechanisms)),
    }


def screen_decision(metrics_payload: dict, comparison: dict, candidate_rows: list[dict], baseline_rows: list[dict], mechanism: dict) -> tuple[str, list[str]]:
    baseline_metrics = metrics_from_rows(baseline_rows)
    baseline_ad_cn = ad_cn_errors(baseline_rows)
    candidate_ad_cn = ad_cn_errors(candidate_rows)
    active = comparison["changed_predictions"] >= 2 and mechanism["residual_raw_max_abs"] > 1e-8
    candidate_correct_by_fold = {
        fold: sum(
            int(row["prediction"]) == int(row["truth"])
            for row in candidate_rows
            if int(row["fold"]) == fold
        )
        for fold in SCREEN_FOLDS
    }
    baseline_correct_by_fold = {
        fold: sum(
            int(row["prediction"]) == int(row["truth"])
            for row in baseline_rows
            if int(row["fold"]) == fold
        )
        for fold in SCREEN_FOLDS
    }
    excessive_fold_damage = [
        fold
        for fold in SCREEN_FOLDS
        if baseline_correct_by_fold[fold] - candidate_correct_by_fold[fold] > 2
    ]
    if excessive_fold_damage:
        return "SCREEN_STOP", [
            "fold_correct_drop_exceeds_2:" + ",".join(map(str, excessive_fold_damage))
        ]
    go_checks = {
        "correct_at_least_167": metrics_payload["correct"] >= 167,
        "repairs_exceed_damages": comparison["repairs"] > comparison["damages"],
        "no_new_ad_cn_errors": candidate_ad_cn <= baseline_ad_cn,
        "macro_f1_drop_at_most_0.005": metrics_payload["macro_f1"] >= baseline_metrics["macro_f1"] - 0.005,
        "bacc_drop_at_most_0.005": metrics_payload["bacc"] >= baseline_metrics["bacc"] - 0.005,
        "changes_at_least_2": comparison["changed_predictions"] >= 2,
    }
    if all(go_checks.values()):
        return "SCREEN_GO", []
    near_checks = {
        "correct_equals_166": metrics_payload["correct"] == 166,
        "repairs_at_least_damages": comparison["repairs"] >= comparison["damages"],
        "no_new_ad_cn_errors": candidate_ad_cn <= baseline_ad_cn,
        "module_active": active,
        "macro_auc_or_bacc_improved": (
            metrics_payload["macro_auc"] > baseline_metrics["macro_auc"]
            or metrics_payload["bacc"] > baseline_metrics["bacc"]
        ),
    }
    if all(near_checks.values()):
        return "SCREEN_NEAR", []
    failures = [name for name, passed in go_checks.items() if not passed]
    return "SCREEN_STOP", failures


def formal_decision(payload: dict) -> str:
    if payload["correct"] >= 563 and payload["acc"] >= 0.94147 and payload["macro_f1"] >= C1["macro_f1"]:
        return "PC_BBF_TARGET"
    if payload["correct"] >= 561 and payload["macro_f1"] >= C1["macro_f1"] and payload["bacc"] >= C1["bacc"]:
        return "PC_BBF_GO"
    if payload["correct"] == 560 and payload["macro_auc"] >= C1["macro_auc"] + 0.001:
        return "PC_BBF_GO"
    return "PC_BBF_STOP"


def write_report_markdown(path: Path, report: dict) -> None:
    metrics_payload = report["metrics"]
    comparison = report["comparison_vs_c1"]
    lines = [
        f"# PC-BBF C1 v1 {report['stage'].title()} Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"- Correct: {metrics_payload['correct']}/{report['subject_count']}",
        f"- ACC: {metrics_payload['acc']:.7f}",
        f"- Macro-F1: {metrics_payload['macro_f1']:.7f}",
        f"- BACC: {metrics_payload['bacc']:.7f}",
        f"- Probability Macro-AUC: {metrics_payload['macro_auc']:.7f}",
        f"- Weighted-F1: {metrics_payload['weighted_f1']:.7f}",
        f"- Confusion matrix: {metrics_payload['confusion_matrix']}",
        f"- Fold ACC mean ± sample SD: {report['fold_acc_mean']:.7f} ± {report['fold_acc_sample_std']:.7f}",
        f"- Repairs / damages / changed: {comparison['repairs']} / {comparison['damages']} / {comparison['changed_predictions']}",
        f"- Boundary errors (AD–sMCI / CN–sMCI / AD–CN): "
        f"{report['boundary_errors']['candidate']['AD_SMCI']} / "
        f"{report['boundary_errors']['candidate']['CN_SMCI']} / "
        f"{report['boundary_errors']['candidate']['AD_CN']}",
        f"- Parameters: {report['parameter_count']} (+{report['added_parameters_vs_c1']} vs C1)",
        f"- Training time: {report['training_seconds']:.3f} s",
        "",
        "## Folds",
        "",
        *[
            f"- fold{item['fold']}: best epoch {item['best_epoch']}, ACC {item['acc']:.7f}"
            for item in report["folds"]
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_arm(context: dict, folds: tuple[int, ...], cap_value: float, output_root: Path, stage: str, source: dict) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for fold in folds:
        summary, fold_rows = run_fold(context, fold, cap_value, output_root)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows.sort(key=lambda row: int(row["subject_index"]))
    require(len(rows) == sum(int(summary["best_metrics"]["confusion_matrix"][i][j]) for summary in summaries for i in range(3) for j in range(3)), "OOF row count mismatch")
    subject_ids = [int(row["subject_index"]) for row in rows]
    require(len(subject_ids) == len(set(subject_ids)), "OOF subject indices are not unique")
    metrics_payload = metrics_from_rows(rows)
    c1_by_subject = baseline_map(source)
    comparison = compare_rows(rows, c1_by_subject)
    baseline_rows = subset_baseline_rows(c1_by_subject, folds)
    require(
        set(subject_ids) == {int(row["subject_index"]) for row in baseline_rows},
        "Candidate and C1 OOF subject sets differ",
    )
    mechanism = summarize_mechanism(summaries)
    if stage == "screen":
        decision, failures = screen_decision(metrics_payload, comparison, rows, baseline_rows, mechanism)
    else:
        decision, failures = formal_decision(metrics_payload), []
    report = {
        "stage": stage,
        "decision": decision,
        "decision_failures": failures,
        "cap_value": cap_value,
        "fold_ids": list(folds),
        "subject_count": len(rows),
        "metrics": metrics_payload,
        "baseline_metrics_same_subjects": metrics_from_rows(baseline_rows),
        "comparison_vs_c1": comparison,
        "ad_cn_errors": {
            "candidate": ad_cn_errors(rows),
            "c1": ad_cn_errors(baseline_rows),
        },
        "boundary_errors": {
            "candidate": boundary_errors(rows),
            "c1": boundary_errors(baseline_rows),
        },
        "parameter_count": PC_BBF_EXPECTED_PARAMETERS,
        "added_parameters_vs_c1": PC_BBF_EXPECTED_PARAMETERS - 862_971,
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "mechanism": mechanism,
        "pc_bbf_gradient_max": {
            key: float(max(summary["gradient_max"][key] for summary in summaries))
            for key in ("input_projection", "residual_output", "gate_output")
        },
        "folds": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "acc": summary["best_metrics"]["acc"],
                "correct": summary["best_metrics"]["correct"],
            }
            for summary in summaries
        ],
        "fold_acc_mean": float(np.mean([summary["best_metrics"]["acc"] for summary in summaries])),
        "fold_acc_sample_std": float(
            np.std(
                [summary["best_metrics"]["acc"] for summary in summaries],
                ddof=1,
            )
        ),
        "run_command": (
            f"python -u -B scripts/run_pc_bbf_c1_v1.py {stage} "
            f"--device {context['device']}"
        ),
        "run_head": git_value("rev-parse", "HEAD"),
        "config": fold_config(context, -1, cap_value),
    }
    require(
        all(value > 0.0 for value in report["pc_bbf_gradient_max"].values()),
        "Aggregated PC-BBF gradient activity check failed",
    )
    write_csv(output_root / "oof_predictions.csv", rows)
    write_json(output_root / "config.json", report["config"])
    write_json(output_root / "report.json", report)
    write_report_markdown(output_root / "report.md", report)
    write_csv(
        output_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, matrix_row))}
            for index, matrix_row in enumerate(metrics_payload["confusion_matrix"])
        ],
    )
    write_csv(
        output_root / "fold_metrics.csv",
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
    return report


def ensure_stage_a_go(output_root: Path) -> dict:
    path = output_root / "branch_complementarity_report.json"
    require(path.is_file(), "Stage-A complementarity report is missing")
    report = read_json(path)
    require(report.get("decision") == "BRANCH_COMPLEMENTARITY_GO", "Stage A did not pass")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "screen", "formal", "all"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "cme_dual_branch_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(git_value("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD") == "", "Branch does not descend from locked base")
    require(torch.cuda.is_available(), "CUDA unavailable; CPU formal training is forbidden")
    device = torch.device(args.device)
    source = branch_probe.discover_source(args.source_root)
    output_root = args.output_root.resolve()
    ensure_stage_a_go(output_root)
    context = load_context(device)
    if args.stage in {"smoke", "all"}:
        run_smoke(context, output_root, 0.10)
        if args.stage == "smoke":
            return
    else:
        smoke_path = output_root / "smoke/smoke_report.json"
        require(smoke_path.is_file() and read_json(smoke_path).get("passed") is True, "Passing smoke is required")

    screen_report = None
    if args.stage in {"screen", "all"}:
        screen_report = run_arm(
            context,
            SCREEN_FOLDS,
            0.10,
            output_root / "screen_cap_0p10",
            "screen",
            source,
        )
        if screen_report["decision"] == "SCREEN_NEAR":
            adjusted_root = output_root / "screen_cap_0p15"
            screen_report = run_arm(
                context,
                SCREEN_FOLDS,
                0.15,
                adjusted_root,
                "screen",
                source,
            )
            if screen_report["decision"] == "SCREEN_NEAR":
                screen_report["decision"] = "SCREEN_STOP"
                screen_report["decision_failures"] = [
                    "cap_0p15_adjustment_did_not_reach_SCREEN_GO"
                ]
                write_json(adjusted_root / "report.json", screen_report)
                write_report_markdown(adjusted_root / "report.md", screen_report)
        if args.stage == "screen" or screen_report["decision"] != "SCREEN_GO":
            return

    if args.stage == "formal":
        candidates = [
            path
            for path in (output_root / "screen_cap_0p10/report.json", output_root / "screen_cap_0p15/report.json")
            if path.is_file() and read_json(path).get("decision") == "SCREEN_GO"
        ]
        require(bool(candidates), "SCREEN_GO is required before formal training")
        selected_screen = read_json(candidates[-1])
        cap_value = float(selected_screen["cap_value"])
    else:
        require(screen_report is not None and screen_report["decision"] == "SCREEN_GO", "SCREEN_GO is required")
        cap_value = float(screen_report["cap_value"])
    formal = run_arm(
        context,
        FORMAL_FOLDS,
        cap_value,
        output_root / "formal",
        "formal",
        source,
    )
    print(
        f"{formal['decision']} correct={formal['metrics']['correct']}/598 "
        f"ACC={formal['metrics']['acc']:.7f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
