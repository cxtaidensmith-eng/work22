"""Run PC-BBF-Safe v1.1 on the locked PC-BBF v1 protocol.

The model and inference path are unchanged.  Training adds exactly one fixed
non-destructive loss comparing the formal PC-BBF logits with the counterfactual
``H0 = Y + G`` logits under a shared post-fusion random state.  Stages are
explicit: ``smoke``, the directed ``screen`` on folds 2/3/5/6, and ``formal``.
Formal training is rejected unless the stored screen decision is SCREEN_GO.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_pc_bbf_c1_v1 as pc_v1
from Model.pc_bbf_safe import PCBBFSafeModel


BASE_COMMIT = "c3a2c6ce13241522cc598f58705417106ce7721c"
OUTPUT_REL = Path("experiments/pc_bbf_safe_v1_1")
C1_ARTIFACT_REL = Path(
    "experiments/cme_dual_branch_v1/c1_shared_private_control_2"
)
PC_V1_ARTIFACT_REL = Path("experiments/pc_bbf_c1_v1/formal")
CLASS_NAMES = pc_v1.CLASS_NAMES
SCREEN_FOLDS = (2, 3, 5, 6)
FORMAL_FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
CAP_VALUE = 0.10
FUSION_RANK = 8
LAMBDA_SAFE = 0.05
LABEL_SMOOTHING = 0.05
EXPECTED_PARAMETERS = 866_924
EXPECTED_C1_PARAMETERS = 862_971
EXPECTED_SCREEN_COUNTS = {
    "c1": {2: 57, 3: 55, 5: 57, 6: 58},
    "pc_bbf_v1": {2: 58, 3: 57, 5: 53, 6: 56},
}
EXPECTED_C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
}
EXPECTED_PC_V1 = {
    "correct": 561,
    "acc": 0.9381270903010034,
    "macro_f1": 0.9266945,
    "bacc": 0.9152140,
    "macro_auc": 0.9701189,
    "weighted_f1": 0.9378308,
}
RESUME_INTERVAL = 25
EPSILON = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            *args,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    require(isinstance(payload, dict), f"Unexpected checkpoint payload: {path}")
    return payload


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def resolve_references(c1_root: Path, pc_v1_root: Path) -> dict:
    c1_artifact = c1_root.resolve() / C1_ARTIFACT_REL
    pc_artifact = pc_v1_root.resolve() / PC_V1_ARTIFACT_REL
    paths = {
        "c1_oof": c1_artifact / "oof_predictions.csv",
        "pc_v1_oof": pc_artifact / "oof_predictions.csv",
        "pc_v1_report": pc_artifact / "report.json",
        "pc_v1_fold0_checkpoint": pc_artifact / "fold_00/checkpoint_best.pt",
    }
    for name, path in paths.items():
        require(path.is_file(), f"Required read-only reference is missing ({name}): {path}")
    return {**paths, "c1_artifact": c1_artifact, "pc_v1_artifact": pc_artifact}


def normalize_reference_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    require(len(rows) == 598, f"Expected 598 reference OOF rows: {path}")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "fold": int(row["fold"]),
                "subject_index": int(row["subject_index"]),
                "truth": int(row["truth"]),
                "prediction": int(row["prediction"]),
                **{
                    f"probability_{name}": float(row[f"probability_{name}"])
                    for name in CLASS_NAMES
                },
            }
        )
    normalized.sort(key=lambda item: item["subject_index"])
    require(
        [item["subject_index"] for item in normalized] == list(range(598)),
        f"Reference subjects are not the unique 0..597 set: {path}",
    )
    return normalized


def reference_bundle(references: dict) -> dict:
    c1_rows = normalize_reference_rows(references["c1_oof"])
    pc_rows = normalize_reference_rows(references["pc_v1_oof"])
    for c1_row, pc_row in zip(c1_rows, pc_rows):
        require(
            c1_row["subject_index"] == pc_row["subject_index"]
            and c1_row["truth"] == pc_row["truth"]
            and c1_row["fold"] == pc_row["fold"],
            "C1 and PC-BBF v1 OOF pairing mismatch",
        )
    c1_metrics = pc_v1.metrics_from_rows(c1_rows)
    pc_metrics = pc_v1.metrics_from_rows(pc_rows)
    require(c1_metrics["correct"] == EXPECTED_C1["correct"], "Wrong C1 OOF source")
    require(pc_metrics["correct"] == EXPECTED_PC_V1["correct"], "Wrong PC-BBF v1 OOF source")
    report = read_json(references["pc_v1_report"])
    require(report["metrics"]["correct"] == EXPECTED_PC_V1["correct"], "PC-BBF v1 report mismatch")
    return {
        "c1_rows": c1_rows,
        "pc_v1_rows": pc_rows,
        "c1_metrics": c1_metrics,
        "pc_v1_metrics": pc_metrics,
        "pc_v1_report": report,
    }


def build_safe_model(context: dict) -> PCBBFSafeModel:
    pc_v1.SET_Random(SEED)
    kwargs = pc_v1.model_kwargs(context)
    kwargs.update({"fusion_rank": FUSION_RANK, "cap_value": CAP_VALUE})
    model = PCBBFSafeModel(context["dataset_dict"], **kwargs).to(context["device"])
    require(pc_v1.parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    require(
        not any(layer.use_graph for layer in model.GCN.layers),
        "Graph unexpectedly enabled",
    )
    return model


def fresh_training_objects(context: dict):
    model = build_safe_model(context)
    criterion = pc_v1.criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=LABEL_SMOOTHING
    )
    groups, optimizer_audit = pc_v1.optimizer_groups(model, context)
    optimizer = torch.optim.Adam(groups)
    scheduler = pc_v1.CustomCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=float(context["config"].Lr_Min),
    )
    require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")
    require(int(scheduler.T_max) == EPOCHS, "Scheduler T_max changed")
    return model, criterion, optimizer, scheduler, optimizer_audit


def forward_base_safe(model, features: torch.Tensor):
    require(
        hasattr(model, "forward_base_safe"),
        "Model.pc_bbf_safe.PCBBFSafeModel must provide forward_base_safe",
    )
    result = model.forward_base_safe(features, return_intermediates=True)
    require(
        isinstance(result, tuple) and len(result) == 5,
        "forward_base_safe must return (safe, base, branches, auxiliary, intermediates)",
    )
    safe_logits, base_logits, branches, auxiliary, intermediates = result
    require(safe_logits.shape == base_logits.shape, "Base/safe logit shape mismatch")
    return safe_logits, base_logits, branches, auxiliary, intermediates


def per_sample_weighted_ce(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, context: dict
) -> torch.Tensor:
    return F.cross_entropy(
        logits[mask],
        labels[mask],
        weight=context["dataset_dict"]["Label_Weight"].to(logits),
        label_smoothing=LABEL_SMOOTHING,
        reduction="none",
    )


def safe_training_loss(model, criterion, context: dict, train_mask: torch.Tensor) -> dict:
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    safe_logits, base_logits, branches, auxiliary, intermediates = forward_base_safe(
        model, features
    )
    original_loss = criterion(safe_logits, labels, train_mask, branches, auxiliary)
    base_loss_i = per_sample_weighted_ce(base_logits, labels, train_mask, context)
    safe_loss_i = per_sample_weighted_ce(safe_logits, labels, train_mask, context)
    excess_i = torch.relu(safe_loss_i - base_loss_i.detach())
    non_destructive_loss = excess_i.mean()
    total_loss = original_loss + LAMBDA_SAFE * non_destructive_loss
    require(bool(torch.isfinite(total_loss)), "Non-finite total loss")
    require(bool(torch.isfinite(non_destructive_loss)), "Non-finite non-destructive loss")
    return {
        "total": total_loss,
        "original": original_loss,
        "non_destructive": non_destructive_loss,
        "non_destructive_i": excess_i,
        "safe_logits": safe_logits,
        "base_logits": base_logits,
        "intermediates": intermediates,
    }


def safety_statistics(values: list[dict]) -> dict:
    require(bool(values), "No non-destructive loss observations")
    return {
        "epoch_mean": float(np.mean([item["mean"] for item in values])),
        "epoch_min": float(np.min([item["mean"] for item in values])),
        "epoch_max": float(np.max([item["mean"] for item in values])),
        "nonzero_sample_fraction_mean": float(
            np.mean([item["nonzero_fraction"] for item in values])
        ),
        "nonzero_sample_fraction_max": float(
            np.max([item["nonzero_fraction"] for item in values])
        ),
        "lambda_safe": LAMBDA_SAFE,
        "weighted_contribution_mean": float(
            LAMBDA_SAFE * np.mean([item["mean"] for item in values])
        ),
    }


def run_smoke(context: dict, output_root: Path, references: dict) -> dict:
    report_path = output_root / "smoke/smoke_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        require(report.get("passed") is True, "Existing smoke did not pass")
        return report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # The historical PC-BBF checkpoint and result are read, never modified.
    loaded_v1 = build_safe_model(context)
    loaded_v1.load_state_dict(
        load_state(references["pc_v1_fold0_checkpoint"]), strict=True
    )
    old_report = read_json(references["pc_v1_report"])
    require(old_report["metrics"]["correct"] == 561, "PC-BBF v1 result is not the locked run")
    del loaded_v1

    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    require(not bool((train_mask & test_mask).any()), "Fold0 train/test masks overlap")
    require(
        int(train_mask.sum().item() + test_mask.sum().item()) == len(labels),
        "Fold0 masks do not partition the dataset",
    )

    initial = build_safe_model(context)
    initial.train()
    initial_safe, initial_base, _, _, _ = forward_base_safe(initial, features)
    initial_diff = float((initial_safe - initial_base).abs().max().detach().cpu())
    initial_base_i = per_sample_weighted_ce(initial_base, labels, train_mask, context)
    initial_safe_i = per_sample_weighted_ce(initial_safe, labels, train_mask, context)
    initial_non_destructive = float(
        torch.relu(initial_safe_i - initial_base_i.detach()).mean().detach().cpu()
    )
    require(initial_diff <= 1e-6, f"Residual-zero base/safe mismatch: {initial_diff}")
    require(initial_non_destructive <= 1e-8, "Residual-zero safety loss is nonzero")
    del initial, initial_safe, initial_base, initial_base_i, initial_safe_i

    model, criterion, optimizer, scheduler, optimizer_audit = fresh_training_objects(
        context
    )
    require(pc_v1.parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    before_output = model.pc_bbf.residual_output.weight.detach().clone()
    gradient_max = {"input_projection": 0.0, "residual_output": 0.0, "gate_output": 0.0}
    epoch_observations = []
    lambda_identity_max_diff = 0.0
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = safe_training_loss(model, criterion, context, train_mask)
        expected_total = losses["original"] + LAMBDA_SAFE * losses["non_destructive"]
        lambda_identity_max_diff = max(
            lambda_identity_max_diff,
            float((losses["total"] - expected_total).abs().detach().cpu()),
        )
        losses["total"].backward()
        require(pc_v1.gradients_finite(model), f"Smoke epoch{epoch}: non-finite gradient")
        for name in gradient_max:
            gradient_max[name] = max(
                gradient_max[name],
                pc_v1.module_gradient_norm(getattr(model.pc_bbf, name)),
            )
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(context["config"].grad_clip)
            )
        optimizer.step()
        scheduler.step()
        excess = losses["non_destructive_i"].detach()
        epoch_observations.append(
            {
                "epoch": epoch,
                "total": float(losses["total"].detach().cpu()),
                "original": float(losses["original"].detach().cpu()),
                "mean": float(losses["non_destructive"].detach().cpu()),
                "nonzero_fraction": float((excess > 0.0).float().mean().cpu()),
            }
        )
    require(
        all(math.isfinite(value) and value > 0.0 for value in gradient_max.values()),
        f"PC-BBF module gradients are inactive: {gradient_max}",
    )
    require(lambda_identity_max_diff <= 1e-8, "lambda_safe was not applied exactly")
    residual_delta = float(
        (model.pc_bbf.residual_output.weight.detach() - before_output).abs().max().cpu()
    )
    require(residual_delta > 0.0, "Residual output did not update")

    # Formal inference calls only the unchanged safe path.
    model.eval()
    with torch.no_grad():
        safe_reference, _, _, reference_intermediates = model(
            features, return_intermediates=True
        )
    state = pc_v1.clone_cpu_state(model)
    checkpoint_path = report_path.parent / "checkpoint_roundtrip.pt"
    atomic_torch_save(state, checkpoint_path)
    reloaded = build_safe_model(context)
    reloaded.load_state_dict(load_state(checkpoint_path), strict=True)
    reloaded.eval()
    with torch.no_grad():
        reloaded_safe, _, _, _ = reloaded(features, return_intermediates=True)
    roundtrip_diff = float((safe_reference - reloaded_safe).abs().max().cpu())
    require(roundtrip_diff <= 1e-7, f"Checkpoint roundtrip mismatch: {roundtrip_diff}")

    report = {
        "passed": True,
        "fold": 0,
        "epochs": 3,
        "device": str(context["device"]),
        "parameter_count": pc_v1.parameter_count(model),
        "cap_value": CAP_VALUE,
        "fusion_rank": FUSION_RANK,
        "lambda_safe": LAMBDA_SAFE,
        "pc_bbf_v1_checkpoint_loaded_strictly": True,
        "pc_bbf_v1_report_readable": True,
        "residual_zero_base_safe_max_abs_diff": initial_diff,
        "residual_zero_non_destructive_loss": initial_non_destructive,
        "optimizer": optimizer_audit,
        "epochs_detail": epoch_observations,
        "safety_loss": safety_statistics(epoch_observations),
        "pc_bbf_gradient_max": gradient_max,
        "lambda_total_identity_max_abs_diff": lambda_identity_max_diff,
        "residual_output_parameter_max_abs_delta": residual_delta,
        "formal_inference_safe_path_only": True,
        "safe_loss_train_mask_only": True,
        "test_labels_used_by_safe_loss": False,
        "checkpoint_roundtrip_logit_max_abs_diff": roundtrip_diff,
        "mechanism_after_epoch3": pc_v1.mechanism_statistics(
            reference_intermediates, test_mask
        ),
        "no_nan_inf": True,
    }
    write_json(output_root / "config.json", fold_config(context, -1))
    write_json(report_path, report)
    (report_path.parent / "smoke_report.md").write_text(
        "\n".join(
            [
                "# PC-BBF-Safe v1.1 Smoke",
                "",
                "Result: **PASS**",
                "",
                f"- Residual-zero base/safe max difference: {initial_diff:.3e}",
                f"- Residual-zero non-destructive loss: {initial_non_destructive:.3e}",
                f"- Parameters: {report['parameter_count']}",
                f"- Lambda: {LAMBDA_SAFE}",
                f"- Gradient maxima: {gradient_max}",
                f"- Checkpoint roundtrip difference: {roundtrip_diff:.3e}",
                "- Safe loss mask: train only",
                "- Formal inference: unchanged safe path only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("SMOKE PASS", flush=True)
    return report


def fold_config(context: dict, fold: int) -> dict:
    return {
        "experiment": "pc_bbf_safe_v1_1",
        "base_commit": BASE_COMMIT,
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
        "original_loss": "historical weighted main CE + three C1 OVR losses",
        "safe_loss": "mean(relu(weighted_CE_i(safe)-stopgrad(weighted_CE_i(base))))",
        "lambda_safe": LAMBDA_SAFE,
        "label_smoothing": LABEL_SMOOTHING,
        "safe_loss_mask": "train_only",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "fusion_rank": FUSION_RANK,
        "cap_value": CAP_VALUE,
        "inference": "unchanged PC-BBF v1 safe path only",
        "parameter_count": EXPECTED_PARAMETERS,
    }


def save_resume(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best: dict | None,
    best_state: dict | None,
    epoch_rows: list[dict],
    safety_rows: list[dict],
    gradient_max: dict,
    elapsed_seconds: float,
) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,
            "model": pc_v1.clone_cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best,
            "best_state": best_state,
            "epoch_rows": epoch_rows,
            "safety_rows": safety_rows,
            "gradient_max": gradient_max,
            "elapsed_seconds": elapsed_seconds,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def run_fold(
    context: dict, fold: int, output_root: Path
) -> tuple[dict, list[dict]]:
    final_dir = output_root / f"fold_{fold:02d}"
    if final_dir.is_dir():
        summary = read_json(final_dir / "summary.json")
        rows = read_csv(final_dir / "best_predictions.csv")
        require(summary["fold"] == fold, "Completed fold id mismatch")
        require(summary["config"]["lambda_safe"] == LAMBDA_SAFE, "Completed fold lambda mismatch")
        require(pc_v1.metrics_from_rows(rows) == summary["best_metrics"], "Completed fold readback mismatch")
        print(
            f"RESUME fold={fold} best_epoch={summary['best_epoch']} "
            f"ACC={summary['best_metrics']['acc']:.7f}",
            flush=True,
        )
        return summary, rows

    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    staging_dir.mkdir(parents=True, exist_ok=True)
    resume_path = staging_dir / "resume_state.pt"
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    model, criterion, optimizer, scheduler, optimizer_audit = fresh_training_objects(
        context
    )
    require(pc_v1.parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    best = None
    best_state = None
    epoch_rows: list[dict] = []
    safety_rows: list[dict] = []
    gradient_max = {"input_projection": 0.0, "residual_output": 0.0, "gate_output": 0.0}
    start_epoch = 1
    elapsed_before = 0.0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best = resume["best"]
        best_state = resume["best_state"]
        epoch_rows = resume["epoch_rows"]
        safety_rows = resume["safety_rows"]
        gradient_max = resume["gradient_max"]
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        restore_rng_state(resume["rng_state"])
        start_epoch = int(resume["epoch"]) + 1
        require(start_epoch <= EPOCHS + 1, "Resume epoch is invalid")
        print(f"RESUME fold={fold} from_epoch={start_epoch}", flush=True)
    elif any(staging_dir.iterdir()):
        raise RuntimeError(f"In-progress fold lacks a valid resume checkpoint: {staging_dir}")

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = safe_training_loss(model, criterion, context, train_mask)
        losses["total"].backward()
        require(pc_v1.gradients_finite(model), f"fold{fold} epoch{epoch}: non-finite gradient")
        for name in gradient_max:
            gradient_max[name] = max(
                gradient_max[name],
                pc_v1.module_gradient_norm(getattr(model.pc_bbf, name)),
            )
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(context["config"].grad_clip)
            )
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_raw, _, _, _ = model(features, return_intermediates=True)
            eval_metrics, score = pc_v1.selection_metrics(
                eval_raw,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or score > tuple(best["selection_tuple"]):
            best = {
                "epoch": epoch,
                "selection_tuple": list(score),
                "metrics": deepcopy(eval_metrics),
            }
            best_state = pc_v1.clone_cpu_state(model)

        excess = losses["non_destructive_i"].detach()
        safety_observation = {
            "epoch": epoch,
            "mean": float(losses["non_destructive"].detach().cpu()),
            "nonzero_fraction": float((excess > 0.0).float().mean().cpu()),
        }
        safety_rows.append(safety_observation)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_total": float(losses["total"].detach().cpu()),
                "loss_pc_bbf_v1": float(losses["original"].detach().cpu()),
                "loss_non_destructive": safety_observation["mean"],
                "loss_non_destructive_nonzero_fraction": safety_observation[
                    "nonzero_fraction"
                ],
                "lambda_safe": LAMBDA_SAFE,
                "acc": eval_metrics["acc"],
                "macro_f1": eval_metrics["macro_f1"],
                "bacc": eval_metrics["bacc"],
                "macro_auc": eval_metrics["macro_auc"],
                "weighted_f1": eval_metrics["weighted_f1"],
            }
        )
        if epoch % RESUME_INTERVAL == 0 and epoch < EPOCHS:
            save_resume(
                resume_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best,
                best_state,
                epoch_rows,
                safety_rows,
                gradient_max,
                elapsed_before + (time.perf_counter() - started),
            )

    require(best is not None and best_state is not None, f"fold{fold}: no best state")
    require(
        all(math.isfinite(value) and value > 0.0 for value in gradient_max.values()),
        f"fold{fold}: PC-BBF module gradient inactive",
    )
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(features, return_intermediates=True)
        best_metrics, _ = pc_v1.selection_metrics(
            best_raw,
            labels,
            test_mask,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )
    rows = pc_v1.prediction_rows(context, fold, best_raw, test_mask)
    require(best_metrics == best["metrics"], f"fold{fold}: best metrics changed")
    require(pc_v1.metrics_from_rows(rows) == best_metrics, f"fold{fold}: OOF readback mismatch")
    summary = {
        "passed": True,
        "fold": fold,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best_metrics,
        "parameter_count": pc_v1.parameter_count(model),
        "added_parameters_vs_c1": pc_v1.parameter_count(model) - EXPECTED_C1_PARAMETERS,
        "elapsed_seconds": float(elapsed_before + time.perf_counter() - started),
        "optimizer": optimizer_audit,
        "gradient_max": gradient_max,
        "mechanism": pc_v1.mechanism_statistics(best_intermediates, test_mask),
        "safety_loss": safety_statistics(safety_rows),
        "config": fold_config(context, fold),
    }
    atomic_torch_save(best_state, staging_dir / "checkpoint_best.pt")
    write_json(staging_dir / "summary.json", summary)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "best_predictions.csv", rows)
    write_csv(
        staging_dir / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, row))}
            for index, row in enumerate(best_metrics["confusion_matrix"])
        ],
    )
    if resume_path.is_file():
        resume_path.unlink()
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


def rows_by_subject(rows: list[dict]) -> dict[int, dict]:
    output = {int(row["subject_index"]): row for row in rows}
    require(len(output) == len(rows), "Duplicate subject ids")
    return output


def subset_rows(rows: list[dict], folds: tuple[int, ...]) -> list[dict]:
    return [row for row in rows if int(row["fold"]) in folds]


def compare_rows(candidate_rows: list[dict], baseline_rows: list[dict]) -> dict:
    baseline = rows_by_subject(baseline_rows)
    repairs, damages, changed = [], [], []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        require(subject in baseline, f"Missing paired baseline subject {subject}")
        reference = baseline[subject]
        truth = int(row["truth"])
        require(
            int(reference["truth"]) == truth
            and int(reference["fold"]) == int(row["fold"]),
            "Paired baseline metadata mismatch",
        )
        old = int(reference["prediction"])
        new = int(row["prediction"])
        if old != truth and new == truth:
            repairs.append(subject)
        if old == truth and new != truth:
            damages.append(subject)
        if old != new:
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


def boundary_errors(rows: list[dict]) -> dict[str, int]:
    pairs = {"AD_SMCI": {0, 2}, "CN_SMCI": {1, 2}, "AD_CN": {0, 1}}
    return {
        name: sum(
            int(row["truth"]) != int(row["prediction"])
            and {int(row["truth"]), int(row["prediction"])} == pair
            for row in rows
        )
        for name, pair in pairs.items()
    }


def aggregate_safety(summaries: list[dict]) -> dict:
    weights = np.asarray(
        [summary["train_size"] * EPOCHS for summary in summaries], dtype=np.float64
    )
    weights /= weights.sum()
    return {
        "epoch_mean": float(
            sum(weight * summary["safety_loss"]["epoch_mean"] for weight, summary in zip(weights, summaries))
        ),
        "epoch_min": float(min(summary["safety_loss"]["epoch_min"] for summary in summaries)),
        "epoch_max": float(max(summary["safety_loss"]["epoch_max"] for summary in summaries)),
        "nonzero_sample_fraction_mean": float(
            sum(
                weight * summary["safety_loss"]["nonzero_sample_fraction_mean"]
                for weight, summary in zip(weights, summaries)
            )
        ),
        "nonzero_sample_fraction_max": float(
            max(summary["safety_loss"]["nonzero_sample_fraction_max"] for summary in summaries)
        ),
        "lambda_safe": LAMBDA_SAFE,
        "weighted_contribution_mean": float(
            LAMBDA_SAFE
            * sum(weight * summary["safety_loss"]["epoch_mean"] for weight, summary in zip(weights, summaries))
        ),
    }


def screen_decision(report: dict) -> tuple[str, list[str], dict]:
    metrics = report["metrics"]
    pc_metrics = report["reference_metrics_same_subjects"]["pc_bbf_v1"]
    c1_metrics = report["reference_metrics_same_subjects"]["c1"]
    comparison = report["comparisons"]["pc_bbf_v1"]
    boundary = report["boundary_errors"]
    fold_correct = {item["fold"]: item["correct"] for item in report["folds"]}
    gain_pair = fold_correct[2] + fold_correct[3]
    recovery_pair = fold_correct[5] + fold_correct[6]
    safety_active = (
        report["safety_loss"]["epoch_mean"] > EPSILON
        and report["safety_loss"]["nonzero_sample_fraction_mean"] > 0.0
    )
    checks = {
        "four_fold_correct_at_least_227": metrics["correct"] >= 227,
        "fold5_fold6_correct_at_least_111": recovery_pair >= 111,
        "fold2_fold3_correct_at_least_114": gain_pair >= 114,
        "repairs_exceed_damages_vs_pc_bbf_v1": comparison["repairs"] > comparison["damages"],
        "cn_smci_errors_reduced_vs_pc_bbf_v1": (
            boundary["candidate"]["CN_SMCI"] < boundary["pc_bbf_v1"]["CN_SMCI"]
        ),
        "ad_smci_errors_increase_at_most_1_vs_pc_bbf_v1": (
            boundary["candidate"]["AD_SMCI"] <= boundary["pc_bbf_v1"]["AD_SMCI"] + 1
        ),
        "no_ad_cn_errors": boundary["candidate"]["AD_CN"] == 0,
        "macro_f1_no_drop_over_0p005": (
            metrics["macro_f1"] >= max(pc_metrics["macro_f1"], c1_metrics["macro_f1"]) - 0.005
        ),
        "bacc_no_drop_over_0p005": (
            metrics["bacc"] >= max(pc_metrics["bacc"], c1_metrics["bacc"]) - 0.005
        ),
        "safe_loss_active": safety_active,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return ("SCREEN_GO" if not failures else "SCREEN_STOP"), failures, checks


def formal_decision(report: dict) -> tuple[str, dict]:
    metrics = report["metrics"]
    c1_metrics = report["reference_metrics_same_subjects"]["c1"]
    pc_metrics = report["reference_metrics_same_subjects"]["pc_bbf_v1"]
    c1_comparison = report["comparisons"]["c1"]
    boundary = report["boundary_errors"]
    target = (
        metrics["correct"] >= 563
        and metrics["macro_f1"] >= c1_metrics["macro_f1"]
        and metrics["bacc"] >= c1_metrics["bacc"]
    )
    positive = (
        metrics["correct"] >= 562
        and metrics["macro_f1"] >= pc_metrics["macro_f1"]
        and metrics["bacc"] >= c1_metrics["bacc"] - 0.002
    )
    stable_metrics = (
        metrics["macro_f1"] >= pc_metrics["macro_f1"] - 0.002
        and metrics["bacc"] >= pc_metrics["bacc"]
        and metrics["macro_auc"] >= pc_metrics["macro_auc"] - 0.002
    )
    stable = (
        metrics["correct"] == 561
        and c1_comparison["damages"] < 17
        and stable_metrics
    )
    no_gain_reasons = {
        "correct_below_561": metrics["correct"] < 561,
        "repairs_not_above_damages_vs_c1": c1_comparison["repairs"] <= c1_comparison["damages"],
        "cn_smci_not_improved_vs_pc_bbf_v1": (
            boundary["candidate"]["CN_SMCI"] >= boundary["pc_bbf_v1"]["CN_SMCI"]
        ),
    }
    no_gain = any(no_gain_reasons.values())
    if no_gain:
        decision = "PC_BBF_SAFE_NO_GAIN"
    elif target:
        decision = "PC_BBF_SAFE_TARGET_REACHED"
    elif positive:
        decision = "PC_BBF_SAFE_POSITIVE"
    elif stable:
        decision = "PC_BBF_SAFE_STABLE"
    else:
        decision = "PC_BBF_SAFE_NO_GAIN"
    evidence = {
        "target_reached": decision == "PC_BBF_SAFE_TARGET_REACHED",
        "positive_evidence": decision in {
            "PC_BBF_SAFE_TARGET_REACHED",
            "PC_BBF_SAFE_POSITIVE",
        },
        "stability_improved": decision in {
            "PC_BBF_SAFE_TARGET_REACHED",
            "PC_BBF_SAFE_STABLE",
        },
        "stable_metrics_check": stable_metrics,
        "no_gain_triggered": no_gain,
        "no_gain_reasons": no_gain_reasons,
    }
    return decision, evidence


def write_report_markdown(path: Path, report: dict) -> None:
    metrics = report["metrics"]
    c1 = report["reference_metrics_same_subjects"]["c1"]
    pc = report["reference_metrics_same_subjects"]["pc_bbf_v1"]
    c1_cmp = report["comparisons"]["c1"]
    pc_cmp = report["comparisons"]["pc_bbf_v1"]
    mechanism = report["mechanism"]
    safety = report["safety_loss"]
    lines = [
        f"# PC-BBF-Safe v1.1 {report['stage'].title()} Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"- Correct: {metrics['correct']}/{report['subject_count']}",
        f"- ACC: {metrics['acc']:.7f}",
        f"- Macro-F1: {metrics['macro_f1']:.7f}",
        f"- BACC: {metrics['bacc']:.7f}",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.7f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.7f}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        f"- Fold ACC mean ± sample SD: {report['fold_acc_mean']:.7f} ± {report['fold_acc_sample_std']:.7f}",
        f"- Versus C1 repairs/damages/changed: {c1_cmp['repairs']}/{c1_cmp['damages']}/{c1_cmp['changed_predictions']}",
        f"- Versus PC-BBF v1 repairs/damages/changed: {pc_cmp['repairs']}/{pc_cmp['damages']}/{pc_cmp['changed_predictions']}",
        f"- Versus C1 ΔACC/F1/BACC/AUC: {metrics['acc']-c1['acc']:+.7f} / {metrics['macro_f1']-c1['macro_f1']:+.7f} / {metrics['bacc']-c1['bacc']:+.7f} / {metrics['macro_auc']-c1['macro_auc']:+.7f}",
        f"- Versus PC-BBF v1 ΔACC/F1/BACC/AUC: {metrics['acc']-pc['acc']:+.7f} / {metrics['macro_f1']-pc['macro_f1']:+.7f} / {metrics['bacc']-pc['bacc']:+.7f} / {metrics['macro_auc']-pc['macro_auc']:+.7f}",
        f"- Boundary errors AD–sMCI/CN–sMCI/AD–CN: {report['boundary_errors']['candidate']['AD_SMCI']}/{report['boundary_errors']['candidate']['CN_SMCI']}/{report['boundary_errors']['candidate']['AD_CN']}",
        f"- Gate mean/std/min/max: {mechanism['gate_mean']:.7f}/{mechanism['gate_std']:.7f}/{mechanism['gate_min']:.7f}/{mechanism['gate_max']:.7f}",
        f"- Residual/shared mean/max; cap saturation: {mechanism['residual_shared_ratio_mean']:.7f}/{mechanism['residual_shared_ratio_max']:.7f}; {mechanism['cap_saturation_fraction']:.7f}",
        f"- Non-destructive loss mean/nonzero fraction: {safety['epoch_mean']:.7g}/{safety['nonzero_sample_fraction_mean']:.7f}",
        f"- Formal safe inference changed predictions vs PC-BBF v1: {pc_cmp['changed_predictions'] > 0} ({pc_cmp['changed_predictions']})",
        f"- Parameters: {report['parameter_count']}",
        f"- Training time: {report['training_seconds']:.3f} s",
        f"- Reproduction: `{report['run_command']}`",
        "",
        "## Folds",
        "",
        *[
            f"- fold{item['fold']}: best epoch {item['best_epoch']}, correct {item['correct']}, ACC {item['acc']:.7f}"
            for item in report["folds"]
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_arm(
    context: dict,
    folds: tuple[int, ...],
    output_root: Path,
    stage: str,
    references: dict,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    rows: list[dict] = []
    for fold in folds:
        summary, fold_rows = run_fold(context, fold, output_root)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows.sort(key=lambda row: int(row["subject_index"]))
    subject_ids = [int(row["subject_index"]) for row in rows]
    require(len(subject_ids) == len(set(subject_ids)), "Candidate OOF subjects are not unique")

    bundle = reference_bundle(references)
    c1_rows = subset_rows(bundle["c1_rows"], folds)
    pc_rows = subset_rows(bundle["pc_v1_rows"], folds)
    require(
        set(subject_ids) == {int(row["subject_index"]) for row in c1_rows}
        == {int(row["subject_index"]) for row in pc_rows},
        "Candidate/reference OOF subject sets differ",
    )
    if stage == "formal":
        require(len(rows) == 598, "Formal OOF must contain exactly 598 subjects")
    else:
        require(len(rows) == 240, "Directed screen must contain exactly 240 subjects")
    metrics = pc_v1.metrics_from_rows(rows)
    fold_payload = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            "correct": summary["best_metrics"]["correct"],
            "acc": summary["best_metrics"]["acc"],
        }
        for summary in summaries
    ]
    mechanism = pc_v1.summarize_mechanism(summaries)
    report = {
        "stage": stage,
        "decision": "PENDING",
        "decision_failures": [],
        "decision_checks": {},
        "fold_ids": list(folds),
        "subject_count": len(rows),
        "metrics": metrics,
        "reference_metrics_same_subjects": {
            "c1": pc_v1.metrics_from_rows(c1_rows),
            "pc_bbf_v1": pc_v1.metrics_from_rows(pc_rows),
        },
        "comparisons": {
            "c1": compare_rows(rows, c1_rows),
            "pc_bbf_v1": compare_rows(rows, pc_rows),
        },
        "boundary_errors": {
            "candidate": boundary_errors(rows),
            "c1": boundary_errors(c1_rows),
            "pc_bbf_v1": boundary_errors(pc_rows),
        },
        "formal_inference_safe_path_only": True,
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameters_vs_c1": EXPECTED_PARAMETERS - EXPECTED_C1_PARAMETERS,
        "training_seconds": float(
            sum(summary["elapsed_seconds"] for summary in summaries)
        ),
        "mechanism": mechanism,
        "safety_loss": aggregate_safety(summaries),
        "pc_bbf_gradient_max": {
            key: float(max(summary["gradient_max"][key] for summary in summaries))
            for key in ("input_projection", "residual_output", "gate_output")
        },
        "folds": fold_payload,
        "fold_acc_mean": float(np.mean([item["acc"] for item in fold_payload])),
        "fold_acc_sample_std": float(np.std([item["acc"] for item in fold_payload], ddof=1)),
        "run_command": f"python -u -B scripts/run_pc_bbf_safe_v1_1.py {stage} --device {context['device']}",
        "base_commit": BASE_COMMIT,
        "run_head": git_value("rev-parse", "HEAD"),
        "config": fold_config(context, -1),
    }
    if stage == "screen":
        observed_counts = {
            "c1": {
                fold: sum(
                    int(row["prediction"]) == int(row["truth"])
                    for row in c1_rows
                    if int(row["fold"]) == fold
                )
                for fold in SCREEN_FOLDS
            },
            "pc_bbf_v1": {
                fold: sum(
                    int(row["prediction"]) == int(row["truth"])
                    for row in pc_rows
                    if int(row["fold"]) == fold
                )
                for fold in SCREEN_FOLDS
            },
        }
        require(observed_counts == EXPECTED_SCREEN_COUNTS, f"Screen baseline fold counts changed: {observed_counts}")
        report["screen_reference_correct_by_fold"] = observed_counts
        decision, failures, checks = screen_decision(report)
        report["decision"] = decision
        report["decision_failures"] = failures
        report["decision_checks"] = checks
    else:
        decision, evidence = formal_decision(report)
        report["decision"] = decision
        report["decision_checks"] = evidence

    require(
        all(value > 0.0 and math.isfinite(value) for value in report["pc_bbf_gradient_max"].values()),
        "Aggregated PC-BBF gradient activity failed",
    )
    write_csv(output_root / "oof_predictions.csv", rows)
    write_json(output_root / "config.json", report["config"])
    write_json(output_root / "report.json", report)
    write_report_markdown(output_root / "report.md", report)
    write_csv(
        output_root / "confusion_matrix.csv",
        [
            {"truth_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, row))}
            for index, row in enumerate(metrics["confusion_matrix"])
        ],
    )
    write_csv(
        output_root / "fold_metrics.csv",
        [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                **{
                    key: value
                    for key, value in summary["best_metrics"].items()
                    if key != "confusion_matrix"
                },
                "elapsed_seconds": summary["elapsed_seconds"],
                "loss_non_destructive_mean": summary["safety_loss"]["epoch_mean"],
                "loss_non_destructive_nonzero_fraction": summary["safety_loss"][
                    "nonzero_sample_fraction_mean"
                ],
            }
            for summary in summaries
        ],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "screen", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    parser.add_argument(
        "--c1-source-root",
        type=Path,
        default=ROOT.parent / "cme_dual_branch_v1",
        help="Read-only worktree containing the existing C1 OOF",
    )
    parser.add_argument(
        "--pc-bbf-v1-source-root",
        type=Path,
        default=ROOT.parent / "pc_bbf_c1_v1",
        help="Read-only worktree containing PC-BBF v1 OOF/checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(
        git_value("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD") == "",
        "Current branch does not descend from the locked PC-BBF v1 HEAD",
    )
    require(torch.cuda.is_available(), "CUDA is required")
    references = resolve_references(args.c1_source_root, args.pc_bbf_v1_source_root)
    reference_bundle(references)
    context = pc_v1.load_context(torch.device(args.device))
    output_root = args.output_root.resolve()

    if args.stage == "smoke":
        run_smoke(context, output_root, references)
        return

    smoke_path = output_root / "smoke/smoke_report.json"
    require(
        smoke_path.is_file() and read_json(smoke_path).get("passed") is True,
        "Passing smoke is required before training",
    )
    if args.stage == "screen":
        screen = run_arm(
            context, SCREEN_FOLDS, output_root / "screen", "screen", references
        )
        print(
            f"{screen['decision']} correct={screen['metrics']['correct']}/240 "
            f"ACC={screen['metrics']['acc']:.7f}",
            flush=True,
        )
        return

    screen_path = output_root / "screen/report.json"
    require(screen_path.is_file(), "Screen report is missing")
    screen = read_json(screen_path)
    require(screen.get("decision") == "SCREEN_GO", "SCREEN_GO is required for formal training")
    formal = run_arm(
        context, FORMAL_FOLDS, output_root / "formal", "formal", references
    )
    print(
        f"{formal['decision']} correct={formal['metrics']['correct']}/598 "
        f"ACC={formal['metrics']['acc']:.7f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
