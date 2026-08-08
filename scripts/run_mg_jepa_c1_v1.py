"""Run MG-JEPA-C1 v1 under the locked C1 ten-fold protocol.

``smoke`` performs the prescribed fold-0 20-epoch pretraining plus three
supervised steps. ``formal`` runs fresh folds 0..9 with 200 pretraining epochs
and conditionally the only allowed full v1.1 rerun at 400 epochs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, ModelEMA, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "mg_jepa_c1_v1"
OUTPUT_REL = Path("experiments/mg_jepa_c1_v1")
BASE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
C1_OOF_REL = Path("experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv")
CLASS_NAMES = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
FOLDS = tuple(range(10))
SEED = 0
SUPERVISED_EPOCHS = 400
SMOKE_PRETRAIN_EPOCHS = 20
DEFAULT_PRETRAIN_EPOCHS = 200
V1_1_PRETRAIN_EPOCHS = 400
C1_PARAMETERS = 862_971
PRETRAIN_BACKBONE_PARAMETERS = 224_400
PRETRAIN_TRAINABLE_PARAMETERS = 300_240
PRETRAIN_WRAPPER_PARAMETERS = 525_024
PRETRAIN_LR = 1e-4
PRETRAIN_WEIGHT_DECAY = 1e-4
TEACHER_DECAY = 0.99
VISIBLE_FEATURE_MASK_RATE = 0.15
REG_TOKEN_COUNT = 4

C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
PC_BBF = {
    "correct": 561,
    "acc": 0.9381270903010034,
    "macro_f1": 0.9266945217489257,
    "bacc": 0.915213972141583,
    "macro_auc": 0.9701189477695239,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_source_hashes() -> dict[str, str]:
    return {
        "model": file_sha256(ROOT / "Model/mg_jepa_c1.py"),
        "runner": file_sha256(ROOT / "scripts/run_mg_jepa_c1_v1.py"),
        "historical_config": file_sha256(ROOT / cme.CONFIG_REL),
    }


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def clone_ema_state(ema: ModelEMA) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in ema.shadow.items()}


def grad_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).cpu()) if total is not None else 0.0


def grads_finite(parameters) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def capture_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    return cpu, cuda


def restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def metric_delta(candidate: dict, baseline: dict) -> dict:
    result = {"correct": int(candidate["correct"] - baseline["correct"])}
    for key in ("acc", "macro_f1", "bacc", "macro_auc"):
        result[key] = float(candidate[key] - baseline[key])
    return result


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "MG-JEPA-C1 requires cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "Historical device changed")
    config = context["config"]
    require(int(config.epochs) == SUPERVISED_EPOCHS, "Supervised epoch protocol changed")
    require(tuple(context["dataset_dict"]["Class_Names"]) == CLASS_NAMES, "Class order changed")
    require(parameter_count(cme.build_model(context, "c1")) == C1_PARAMETERS, "C1 parameter count changed")
    c1_path = ROOT / C1_OOF_REL
    require(c1_path.is_file(), f"C1 OOF missing: {c1_path}")
    c1_rows = read_csv(c1_path)
    require(len(c1_rows) == 598, "C1 OOF row count changed")
    c1_metrics = cme.metrics_from_rows(c1_rows)
    require(c1_metrics["correct"] == C1["correct"], "C1 correct anchor changed")
    require(c1_metrics["confusion_matrix"] == C1["confusion_matrix"], "C1 confusion anchor changed")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(c1_metrics[key] - C1[key]) <= 5e-7, f"C1 anchor changed: {key}")
    context.update(
        {
            "c1_rows": c1_rows,
            "c1_by_subject": {int(row["subject_index"]): row for row in c1_rows},
        }
    )
    require(len(context["c1_by_subject"]) == 598, "C1 subject duplication")
    return context


def boundary_errors(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        "AD_SMCI": int(matrix[0, 2] + matrix[2, 0]),
        "CN_SMCI": int(matrix[1, 2] + matrix[2, 1]),
        "AD_CN": int(matrix[0, 1] + matrix[1, 0]),
    }


def paired_comparison(candidate_rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"C1 missing subject {subject}")
        reference = reference_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth and int(reference["fold"]) == int(row["fold"]), "C1 OOF alignment changed")
        old, new = int(reference["prediction"]), int(row["prediction"])
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


def validate_oof(rows: list[dict]) -> list[dict]:
    require(len(rows) == 598, "OOF row count changed")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == 598 and set(subjects) == set(range(598)), "OOF subjects invalid")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "OOF fold set changed")
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def mg_api():
    from Model.mg_jepa_c1 import (
        MGJEPAPretrainer,
        load_pretrained_backbone_into_c1,
        make_mask_generator,
        masked_modality_for_epoch,
    )

    return (
        MGJEPAPretrainer,
        load_pretrained_backbone_into_c1,
        make_mask_generator,
        masked_modality_for_epoch,
    )


def config_payload(context: dict, pretrain_epochs: int, version: str, folds) -> dict:
    historical = context["config"]
    return {
        "experiment": EXPERIMENT,
        "version": version,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "base_commit": BASE_COMMIT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(folds),
        "seed_per_fold": SEED,
        "pretrain_random_seed": SEED,
        "pretrain_epochs": int(pretrain_epochs),
        "pretrain_optimizer": "AdamW",
        "pretrain_lr": PRETRAIN_LR,
        "pretrain_weight_decay": PRETRAIN_WEIGHT_DECAY,
        "pretrain_scheduler": f"CosineAnnealingLR(T_max={pretrain_epochs})",
        "teacher_ema_decay": TEACHER_DECAY,
        "visible_feature_mask_rate": VISIBLE_FEATURE_MASK_RATE,
        "masked_modality_schedule": "zero-based pretrain_step % 6",
        "reg_token_count": REG_TOKEN_COUNT,
        "pretrain_train_fold_only": True,
        "supervised_epochs": SUPERVISED_EPOCHS,
        "full_batch_transductive_supervised": True,
        "single_model": True,
        "ensemble": False,
        "multi_seed": False,
        "orthogonality": False,
        "graph_enabled": False,
        "loss": "historical C1 weighted main CE plus three OVR auxiliary losses",
        "supervised_optimizer": "historical C1 Adam",
        "supervised_scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "supervised_lr": float(historical.lr),
        "supervised_weight_decay": float(historical.weight_decay),
        "supervised_lr_min": float(historical.Lr_Min),
        "supervised_grad_clip": float(historical.grad_clip),
        "supervised_label_smoothing": 0.05,
        "hidden_size": int(historical.Hidden_size),
        "drop_rate": float(historical.Drop_rate),
        "num_layers": int(historical.num_layers),
        "num_heads": int(historical.num_heads),
        "input_noise_std": float(historical.input_noise_std),
        "drop_path": float(historical.drop_path),
        "logit_adjust_tau": float(historical.logit_adjust_tau),
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "base_c1_runner_use_ema": False,
        "task_locked_supervised_ema_first_update_epoch": 20,
        "task_locked_supervised_ema_first_selection_epoch": 21,
        "task_locked_supervised_ema_decay": float(context["config"].ema_decay),
        "pretrain_backbone_parameter_count": PRETRAIN_BACKBONE_PARAMETERS,
        "pretrain_trainable_parameter_count": PRETRAIN_TRAINABLE_PARAMETERS,
        "pretrain_wrapper_parameter_count": PRETRAIN_WRAPPER_PARAMETERS,
        "inference_parameter_count": C1_PARAMETERS,
        "collapse_rule": "shared global mean batch std <=1e-4 OR private global mean batch std <=1e-4",
        "collapse_batch_std_threshold": 1e-4,
        "v1_1_last20_relative_decline_threshold": 0.02,
        "historical_config_path": str(cme.CONFIG_REL).replace("\\", "/"),
        "locked_source_sha256": locked_source_hashes(),
        "environment": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
        },
    }


def _named_gradient_groups(pretrainer) -> dict[str, list[torch.nn.Parameter]]:
    groups = {"encoder": [], "transformer": [], "private": [], "shared_predictor": [], "private_predictor": []}
    for name, parameter in pretrainer.named_parameters():
        lower = name.lower()
        if "teacher" in lower:
            continue
        if "shared_predictor" in lower:
            groups["shared_predictor"].append(parameter)
        elif "private_predictor" in lower:
            groups["private_predictor"].append(parameter)
        elif "modal_token_encoder" in lower or "feature_modal" in lower:
            groups["encoder"].append(parameter)
        elif "shared_transformer" in lower or "modal_transformer" in lower:
            groups["transformer"].append(parameter)
        elif "private_adapter" in lower:
            groups["private"].append(parameter)
    return groups


def latent_diagnostics(output: dict) -> dict:
    shared = output["student_shared"].detach().float()
    private = output["student_private"].detach().float()
    require(shared.ndim == 3 and private.ndim == 3 and shared.shape[1] == private.shape[1] == 6, "MG-JEPA latent shape changed")
    shared_std_by_modality = shared.std(dim=0, unbiased=False).mean(dim=-1)
    private_std_by_modality = private.std(dim=0, unbiased=False).mean(dim=-1)
    cosine_payload = output.get("cosines", {})
    if isinstance(cosine_payload, dict):
        shared_cosine = cosine_payload.get("shared")
        private_cosine = cosine_payload.get("private")
    else:
        shared_cosine = private_cosine = None
    if shared_cosine is None:
        teacher_shared = output["teacher_shared"].detach().float()
        shared_cosine = torch.nn.functional.cosine_similarity(shared, teacher_shared, dim=-1).mean(dim=0)
    if private_cosine is None:
        teacher_private = output["teacher_private"].detach().float()
        private_cosine = torch.nn.functional.cosine_similarity(private, teacher_private, dim=-1).mean(dim=0)
    shared_cosine = torch.as_tensor(shared_cosine).detach().float().reshape(-1)
    private_cosine = torch.as_tensor(private_cosine).detach().float().reshape(-1)
    require(shared_cosine.numel() == private_cosine.numel() == 6, "MG-JEPA cosine diagnostic shape changed")
    threshold = 1e-4
    shared_std_mean = float(shared_std_by_modality.mean().cpu())
    private_std_mean = float(private_std_by_modality.mean().cpu())
    collapse = bool(
        not torch.isfinite(shared_std_by_modality).all()
        or not torch.isfinite(private_std_by_modality).all()
        or shared_std_mean <= threshold
        or private_std_mean <= threshold
    )
    return {
        "teacher_student_shared_cosine_by_modality": {name: float(shared_cosine[index].cpu()) for index, name in enumerate(MODALITY_NAMES)},
        "teacher_student_private_cosine_by_modality": {name: float(private_cosine[index].cpu()) for index, name in enumerate(MODALITY_NAMES)},
        "shared_latent_batch_std_by_modality": {name: float(shared_std_by_modality[index].cpu()) for index, name in enumerate(MODALITY_NAMES)},
        "private_latent_batch_std_by_modality": {name: float(private_std_by_modality[index].cpu()) for index, name in enumerate(MODALITY_NAMES)},
        "shared_latent_batch_std_mean": shared_std_mean,
        "private_latent_batch_std_mean": private_std_mean,
        "collapse": collapse,
        "collapse_threshold": threshold,
    }


def teacher_state(pretrainer) -> dict[str, torch.Tensor]:
    teacher = getattr(pretrainer, "teacher", None)
    require(isinstance(teacher, torch.nn.Module), "MG-JEPA teacher module missing")
    return clone_cpu_state(teacher)


def run_pretraining(context: dict, fold: int, pretrain_epochs: int):
    (
        MGJEPAPretrainer,
        load_pretrained_backbone_into_c1,
        make_mask_generator,
        masked_modality_for_epoch,
    ) = mg_api()
    SET_Random(SEED)
    model = cme.build_model(context, "c1")
    require(parameter_count(model) == C1_PARAMETERS, "Initial C1 parameter count changed")
    initial_full_state = clone_cpu_state(model)
    formal_rng = capture_rng()

    # Predictor/mask/dropout randomness is an independent stream.  The exact
    # formal RNG captured above is restored after the temporary modules die.
    SET_Random(SEED)
    pretrainer = MGJEPAPretrainer.from_c1(
        model,
        teacher_decay=TEACHER_DECAY,
        visible_feature_mask_rate=VISIBLE_FEATURE_MASK_RATE,
        reg_token_count=REG_TOKEN_COUNT,
    ).to(context["device"])
    require(parameter_count(pretrainer) == PRETRAIN_WRAPPER_PARAMETERS, "MG-JEPA wrapper parameter count changed")
    train_mask, _ = context["dataset_data"]["Mask"][fold]
    train_idx = train_mask.nonzero(as_tuple=False).flatten()
    require(int(train_idx.numel()) == int(train_mask.sum()), "Train row indices changed")
    x_train = context["dataset_data"]["Feature"][train_idx]
    mask_generator = make_mask_generator(context["device"], SEED)
    parameters = list(pretrainer.pretrain_parameters())
    parameter_ids = {id(parameter) for parameter in parameters}
    require(len(parameter_ids) == len(parameters) and parameters, "Pretrain parameter coverage invalid")
    require(sum(parameter.numel() for parameter in parameters) == PRETRAIN_TRAINABLE_PARAMETERS, "MG-JEPA trainable parameter count changed")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=PRETRAIN_LR,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pretrain_epochs
    )
    teacher_initial = teacher_state(pretrainer)
    teacher_module = getattr(pretrainer, "teacher")
    require(all(not parameter.requires_grad for parameter in teacher_module.parameters()), "Teacher requires gradient")
    gradient_groups = _named_gradient_groups(pretrainer)
    require(all(gradient_groups.values()), "MG-JEPA gradient group discovery failed")
    gradient_max = {name: 0.0 for name in gradient_groups}
    epoch_rows = []
    masked_counts = [0] * 6
    started = time.perf_counter()
    final_output = None
    for step_index in range(pretrain_epochs):
        pretrainer.train()
        teacher_module.eval()
        optimizer.zero_grad(set_to_none=True)
        masked_modality = int(masked_modality_for_epoch(step_index))
        require(masked_modality == step_index % 6, "Masked modality schedule changed")
        masked_counts[masked_modality] += 1
        output = pretrainer(
            x_train,
            masked_modality=masked_modality,
            mask_generator=mask_generator,
        )
        loss = output["loss"]
        require(bool(torch.isfinite(loss)), f"fold{fold} pretrain step{step_index}: non-finite loss")
        loss.backward()
        require(grads_finite(parameters), f"fold{fold}: non-finite pretrain gradient")
        require(all(parameter.grad is None for parameter in teacher_module.parameters()), "Teacher received gradient")
        for name, group in gradient_groups.items():
            gradient_max[name] = max(gradient_max[name], grad_norm(group))
        optimizer.step()
        pretrainer.update_teacher()
        scheduler.step()
        epoch_rows.append(
            {
                "epoch": step_index + 1,
                "masked_modality_index": masked_modality,
                "masked_modality_name": MODALITY_NAMES[masked_modality],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "loss_shared": float(output["loss_shared"].detach().cpu()),
                "loss_private": float(output["loss_private"].detach().cpu()),
            }
        )
        final_output = output
    require(final_output is not None, "Pretraining produced no output")
    require(all(value > 0.0 and math.isfinite(value) for value in gradient_max.values()), "A pretraining group never received finite nonzero gradient")
    teacher_final = teacher_state(pretrainer)
    teacher_delta = max(
        float((teacher_final[name] - value).abs().max())
        for name, value in teacher_initial.items()
        if value.dtype.is_floating_point
    )
    require(teacher_delta > 0.0, "EMA teacher did not update")
    # Report collapse/cosine on one deterministic complete-view pass after the
    # final Teacher EMA update, not on whichever modality happened to be masked
    # in the last optimization step.
    diagnostics = latent_diagnostics(
        pretrainer.complete_view_diagnostics(x_train)
    )
    losses = np.asarray([row["loss"] for row in epoch_rows], dtype=np.float64)
    tail_first = float(losses[-20:-10].mean()) if pretrain_epochs >= 20 else float("nan")
    tail_last = float(losses[-10:].mean()) if pretrain_epochs >= 10 else float("nan")
    tail_relative_decline = (
        float((tail_first - tail_last) / max(abs(tail_first), 1e-12))
        if pretrain_epochs >= 20
        else float("nan")
    )
    pretrained_backbone = {
        name: value.detach().cpu().clone()
        for name, value in pretrainer.export_student_c1_state().items()
    }
    require(pretrained_backbone, "Exported pretrained backbone is empty")
    model.load_state_dict(initial_full_state, strict=True)
    load_pretrained_backbone_into_c1(model, pretrained_backbone)
    require(parameter_count(model) == C1_PARAMETERS, "Temporary pretraining modules leaked into C1")
    current_state = model.state_dict()
    for name, initial_value in initial_full_state.items():
        if name not in pretrained_backbone:
            require(torch.equal(current_state[name].detach().cpu(), initial_value), f"Non-pretrained C1 state changed: {name}")
    restore_rng(formal_rng)
    summary = {
        "fold": fold,
        "epochs": pretrain_epochs,
        "train_subject_count": int(train_idx.numel()),
        "uses_test_subjects": False,
        "uses_labels": False,
        "pretrain_backbone_parameter_count": PRETRAIN_BACKBONE_PARAMETERS,
        "pretrain_trainable_parameter_count": PRETRAIN_TRAINABLE_PARAMETERS,
        "pretrain_wrapper_parameter_count": PRETRAIN_WRAPPER_PARAMETERS,
        "masked_modality_counts": {name: masked_counts[index] for index, name in enumerate(MODALITY_NAMES)},
        "final_loss": epoch_rows[-1]["loss"],
        "final_loss_shared": epoch_rows[-1]["loss_shared"],
        "final_loss_private": epoch_rows[-1]["loss_private"],
        "first5_loss_mean": float(losses[:5].mean()),
        "last5_loss_mean": float(losses[-5:].mean()),
        "last20_first10_loss_mean": tail_first,
        "last20_last10_loss_mean": tail_last,
        "last20_relative_decline": tail_relative_decline,
        "gradient_max": gradient_max,
        "teacher_parameter_max_abs_delta": teacher_delta,
        "diagnostics": diagnostics,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    del pretrainer, optimizer, scheduler, teacher_module
    torch.cuda.empty_cache()
    return model, pretrained_backbone, epoch_rows, summary


def make_supervised_objects(context: dict, model):
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer,
        T_max=SUPERVISED_EPOCHS,
        eta_min=float(context["config"].Lr_Min),
    )
    ema = ModelEMA(model, decay=float(context["config"].ema_decay))
    return criterion, optimizer, scheduler, ema


def supervised_smoke(context: dict, model) -> dict:
    criterion, optimizer, scheduler, ema = make_supervised_objects(context, model)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]
    losses = []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), "Supervised smoke loss is non-finite")
        loss.backward()
        require(grads_finite(model.parameters()), "Supervised smoke gradient is non-finite")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        reference, _, _ = model(features)
    return {
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema": ema,
        "losses": losses,
        "reference_logits": reference,
    }


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_started = time.perf_counter()
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    model, backbone_state, pretrain_rows, pretrain_summary = run_pretraining(
        context, 0, SMOKE_PRETRAIN_EPOCHS
    )
    require(pretrain_summary["last5_loss_mean"] < pretrain_summary["first5_loss_mean"], "Smoke pretraining loss did not decline")
    require(not pretrain_summary["diagnostics"]["collapse"], "Smoke latent collapse detected")
    require(parameter_count(model) == C1_PARAMETERS, "Smoke formal model parameter count changed")
    supervised = supervised_smoke(context, model)
    checkpoint = {
        "model_state": clone_cpu_state(model),
        "pretrained_backbone_state": backbone_state,
        "supervised_optimizer_state": deepcopy(supervised["optimizer"].state_dict()),
        "supervised_scheduler_state": deepcopy(supervised["scheduler"].state_dict()),
        "supervised_ema_shadow": clone_ema_state(supervised["ema"]),
        "config": config_payload(context, SMOKE_PRETRAIN_EPOCHS, "smoke", [0]),
    }
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded = cme.build_model(context, "c1")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    criterion2, optimizer2, scheduler2, ema2 = make_supervised_objects(context, reloaded)
    optimizer2.load_state_dict(payload["supervised_optimizer_state"])
    scheduler2.load_state_dict(payload["supervised_scheduler_state"])
    require(set(payload["supervised_ema_shadow"]) == set(ema2.shadow), "Smoke EMA checkpoint coverage changed")
    for name, value in payload["supervised_ema_shadow"].items():
        ema2.shadow[name].copy_(value.to(ema2.shadow[name]))
    reloaded.eval()
    with torch.no_grad():
        loaded, _, _ = reloaded(context["dataset_data"]["Feature"])
    require(tuple(loaded.shape) == (598, 3), "Smoke formal output shape changed")
    roundtrip = float((supervised["reference_logits"] - loaded).abs().max().cpu())
    require(roundtrip <= 1e-7, "Smoke checkpoint roundtrip failed")
    report = {
        "passed": True,
        "pretrain": pretrain_summary,
        "supervised_smoke_epochs": 3,
        "supervised_losses": supervised["losses"],
        "inference_parameter_count": parameter_count(reloaded),
        "formal_output_shape": list(loaded.shape),
        "checkpoint_contains": [
            "model_state",
            "pretrained_backbone_state",
            "supervised_optimizer_state",
            "supervised_scheduler_state",
            "supervised_ema_shadow",
        ],
        "checkpoint_roundtrip_logit_max_abs_diff": roundtrip,
        "source_content_sha256": locked_source_hashes(),
        "device": str(context["device"]),
        "wall_seconds": float(time.perf_counter() - smoke_started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_mg_jepa_c1_v1.py smoke --device cuda:0',
    }
    write_json(output_root / "config.json", config_payload(context, SMOKE_PRETRAIN_EPOCHS, "smoke", [0]))
    write_csv(smoke_root / "pretrain_loss.csv", pretrain_rows)
    write_json(smoke_root / "smoke_report.json", report)
    del model, reloaded, criterion2, optimizer2, scheduler2, ema2
    torch.cuda.empty_cache()
    return report


def fold_config(context: dict, pretrain_epochs: int, version: str, fold: int) -> dict:
    return config_payload(context, pretrain_epochs, version, [fold])


def load_completed_fold(
    context: dict,
    pretrain_epochs: int,
    version: str,
    fold: int,
    final_dir: Path,
):
    if not final_dir.exists():
        return None
    required = [
        final_dir / "config.json",
        final_dir / "pretrain_loss.csv",
        final_dir / "supervised_epoch_metrics.csv",
        final_dir / "fold_metrics.json",
        final_dir / "fold_oof_predictions.csv",
        final_dir / "pretrained_backbone.pt",
        final_dir / "checkpoint_best.pt",
        final_dir / "COMPLETE.json",
    ]
    require(all(path.is_file() for path in required), f"Incomplete completed fold{fold}")
    expected = fold_config(context, pretrain_epochs, version, fold)
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == expected, f"fold{fold}: config/source changed")
    marker = json.loads((final_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    require(marker == {"complete": True, "fold": fold, "source_commit": expected["source_commit"], "version": version}, f"fold{fold}: completion marker changed")
    summary = json.loads((final_dir / "fold_metrics.json").read_text(encoding="utf-8"))
    rows = read_csv(final_dir / "fold_oof_predictions.csv")
    pretrain_rows = read_csv(final_dir / "pretrain_loss.csv")
    epochs = read_csv(final_dir / "supervised_epoch_metrics.csv")
    require(summary["config"] == expected and summary["fold"] == fold, f"fold{fold}: summary mismatch")
    require(len(pretrain_rows) == pretrain_epochs and [int(row["epoch"]) for row in pretrain_rows] == list(range(1, pretrain_epochs + 1)), f"fold{fold}: pretraining history changed")
    require(all(int(row["masked_modality_index"]) == (int(row["epoch"]) - 1) % 6 for row in pretrain_rows), f"fold{fold}: modality-mask cycle changed")
    require(len(epochs) == SUPERVISED_EPOCHS and [int(row["epoch"]) for row in epochs] == list(range(1, SUPERVISED_EPOCHS + 1)), f"fold{fold}: epoch history changed")
    require(len(rows) == summary["test_size"] and cme.metrics_from_rows(rows) == summary["best_metrics"], f"fold{fold}: OOF changed")
    selection = [
        (float(row["acc"]), float(row["macro_auc"]), float(row["macro_f1"]))
        for row in epochs
    ]
    best_index = max(range(len(selection)), key=lambda index: selection[index])
    require(best_index + 1 == summary["best_epoch"], f"fold{fold}: best epoch selection changed")
    require(selection[best_index] == (summary["best_metrics"]["acc"], summary["best_metrics"]["macro_auc"], summary["best_metrics"]["macro_f1"]), f"fold{fold}: selected metric tuple changed")
    observed_mask_counts = {
        name: sum(int(row["masked_modality_index"]) == index for row in pretrain_rows)
        for index, name in enumerate(MODALITY_NAMES)
    }
    require(observed_mask_counts == summary["pretrain"]["masked_modality_counts"], f"fold{fold}: mask-count summary changed")
    fold_reference = {
        int(row["subject_index"]): row
        for row in context["c1_rows"]
        if int(row["fold"]) == fold
    }
    require(len({int(row["subject_index"]) for row in rows}) == len(rows), f"fold{fold}: duplicate OOF subject")
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in fold_reference and int(row["fold"]) == fold and int(row["truth"]) == int(fold_reference[subject]["truth"]), f"fold{fold}: OOF alignment changed")
        probability = [float(row[f"probability_{name}"]) for name in CLASS_NAMES]
        require(abs(sum(probability) - 1.0) <= 1e-6 and int(np.argmax(probability)) == int(row["prediction"]), f"fold{fold}: OOF probability/prediction mismatch")
    backbone = torch.load(final_dir / "pretrained_backbone.pt", map_location="cpu", weights_only=True)
    require(isinstance(backbone, dict) and len(backbone) == 75, f"fold{fold}: pretrained backbone state changed")
    _, load_pretrained_backbone_into_c1, _, _ = mg_api()
    check_model = cme.build_model(context, "c1")
    load_pretrained_backbone_into_c1(check_model, backbone)
    require(parameter_count(check_model) == C1_PARAMETERS, f"fold{fold}: pretrained overlay changed inference model")
    checkpoint = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == expected and checkpoint["best_epoch"] == summary["best_epoch"], f"fold{fold}: checkpoint metadata changed")
    require(bool(checkpoint["best_used_supervised_ema"]) == bool(summary["best_used_supervised_ema"]), f"fold{fold}: best EMA selection flag changed")
    check_model.load_state_dict(checkpoint["inference_model_state"], strict=True)
    require(set(checkpoint["supervised_ema_shadow_at_best"]) == set(checkpoint["inference_model_state"]), f"fold{fold}: EMA checkpoint coverage changed")
    if checkpoint["best_used_supervised_ema"]:
        require(all(torch.equal(checkpoint["supervised_ema_shadow_at_best"][name], value) for name, value in checkpoint["inference_model_state"].items()), f"fold{fold}: selected EMA weights/checkpoint differ")
    del check_model, checkpoint, backbone
    torch.cuda.empty_cache()
    print(f"RESUME {version} fold={fold} correct={summary['best_metrics']['correct']}", flush=True)
    return summary, rows


def train_fold(
    context: dict,
    pretrain_epochs: int,
    version: str,
    fold: int,
    version_root: Path,
):
    final_dir = version_root / f"fold_{fold:02d}"
    resumed = load_completed_fold(
        context, pretrain_epochs, version, fold, final_dir
    )
    if resumed is not None:
        return resumed
    staging = version_root / f".fold_{fold:02d}_in_progress"
    expected_config = fold_config(context, pretrain_epochs, version, fold)
    allowed_partial_names = {
        "config.json",
        "pretrained_backbone.pt",
        "checkpoint_best.pt",
        "pretrain_loss.csv",
        "supervised_epoch_metrics.csv",
        "fold_metrics.json",
        "fold_oof_predictions.csv",
        "COMPLETE.json",
        "RESTARTED_INCOMPLETE.json",
    }
    restarted_incomplete = False
    if staging.exists():
        partial_config = staging / "config.json"
        require(partial_config.is_file(), f"In-progress fold lacks source/config lock: {staging}")
        require(json.loads(partial_config.read_text(encoding="utf-8")) == expected_config, f"In-progress fold source/config changed: {staging}")
        require({path.name for path in staging.iterdir()} <= allowed_partial_names, f"Unexpected file in in-progress fold: {staging}")
        resolved_staging = staging.resolve()
        require(resolved_staging.parent == version_root.resolve() and resolved_staging.name == f".fold_{fold:02d}_in_progress", "Unsafe in-progress restart target")
        shutil.rmtree(resolved_staging)
        restarted_incomplete = True
        print(f"RESTART_INCOMPLETE {version} fold={fold} with matching source/config", flush=True)
    staging.mkdir(parents=True)
    write_json(staging / "config.json", expected_config)
    if restarted_incomplete:
        write_json(
            staging / "RESTARTED_INCOMPLETE.json",
            {
                "reason": "matching-source/config incomplete fold was safely restarted fresh",
                "fold": fold,
                "version": version,
            },
        )
    overall_started = time.perf_counter()
    model, backbone_state, pretrain_rows, pretrain_summary = run_pretraining(
        context, fold, pretrain_epochs
    )
    require(parameter_count(model) == C1_PARAMETERS, "Fine-tuning model parameter count changed")
    criterion, optimizer, scheduler, ema = make_supervised_objects(context, model)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    best = None
    best_state = None
    best_ema_shadow = None
    epoch_rows = []
    supervised_started = time.perf_counter()
    ema_update_count = 0
    for epoch in range(1, SUPERVISED_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"{version} fold{fold} epoch{epoch}: non-finite supervised loss")
        loss.backward()
        require(grads_finite(model.parameters()), f"{version} fold{fold}: non-finite supervised gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        if epoch >= 20:
            ema.update(model)
            ema_update_count += 1
        use_ema_for_selection = epoch >= 21
        if use_ema_for_selection:
            ema.swap_in(model)
        model.eval()
        with torch.no_grad():
            evaluated, _, _ = model(features)
            metrics, selection = cme.selection_metrics(
                evaluated,
                labels,
                test_mask,
                context["dataset_dict"]["Label_Weight"],
                float(context["config"].logit_adjust_tau),
            )
        if best is None or selection > best["selection"]:
            best = {
                "epoch": epoch,
                "selection": selection,
                "metrics": deepcopy(metrics),
                "used_supervised_ema": use_ema_for_selection,
            }
            # When EMA is active the swapped weights are exactly the weights
            # used for selection and therefore the only valid inference state.
            best_state = clone_cpu_state(model)
            best_ema_shadow = clone_ema_state(ema)
        if use_ema_for_selection:
            ema.swap_out(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "ema_updated": epoch >= 20,
                "ema_used_for_selection": use_ema_for_selection,
                **{key: metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            }
        )
    supervised_seconds = float(time.perf_counter() - supervised_started)
    require(best is not None and best_state is not None and best_ema_shadow is not None, "Best supervised state missing")
    require(ema_update_count == 381, "Supervised EMA update schedule changed")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _ = model(features)
    rows = cme.prediction_rows(
        fold,
        best_raw,
        labels,
        test_mask,
        context["dataset_dict"],
        context["config"],
    )
    require(cme.metrics_from_rows(rows) == best["metrics"], "Best inference state does not match selection")
    comparison = paired_comparison(
        rows,
        {
            int(row["subject_index"]): row
            for row in context["c1_rows"]
            if int(row["fold"]) == fold
        },
    )
    summary = {
        "passed": True,
        "version": version,
        "fold": fold,
        "seed": SEED,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "pretrain": pretrain_summary,
        "best_epoch": int(best["epoch"]),
        "best_used_supervised_ema": bool(best["used_supervised_ema"]),
        "best_metrics": best["metrics"],
        "comparison_vs_c1": comparison,
        "boundary_errors": boundary_errors(best["metrics"]),
        "inference_parameter_count": C1_PARAMETERS,
        "supervised_ema_first_update_epoch": 20,
        "supervised_ema_first_selection_epoch": 21,
        "supervised_ema_update_count": ema_update_count,
        "pretrain_seconds": pretrain_summary["elapsed_seconds"],
        "supervised_seconds": supervised_seconds,
        "wall_seconds": float(time.perf_counter() - overall_started),
        "config": fold_config(context, pretrain_epochs, version, fold),
        "restarted_matching_incomplete_fold": restarted_incomplete,
    }
    config = summary["config"]
    torch.save(backbone_state, staging / "pretrained_backbone.pt")
    torch.save(
        {
            "best_epoch": best["epoch"],
            "best_used_supervised_ema": best["used_supervised_ema"],
            "inference_model_state": best_state,
            "supervised_ema_shadow_at_best": best_ema_shadow,
            "config": config,
        },
        staging / "checkpoint_best.pt",
    )
    write_json(staging / "config.json", config)
    write_csv(staging / "pretrain_loss.csv", pretrain_rows)
    write_csv(staging / "supervised_epoch_metrics.csv", epoch_rows)
    write_json(staging / "fold_metrics.json", summary)
    write_csv(staging / "fold_oof_predictions.csv", rows)
    write_json(
        staging / "COMPLETE.json",
        {
            "complete": True,
            "fold": fold,
            "source_commit": config["source_commit"],
            "version": version,
        },
    )
    staging.rename(final_dir)
    print(
        f"{version} fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']} "
        f"ACC={best['metrics']['acc']:.7f} F1={best['metrics']['macro_f1']:.7f} "
        f"BACC={best['metrics']['bacc']:.7f} AUC={best['metrics']['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler, ema
    torch.cuda.empty_cache()
    return summary, rows


def aggregate_pretrain(summaries: list[dict]) -> dict:
    payloads = [summary["pretrain"] for summary in summaries]
    weights = np.asarray([summary["train_size"] for summary in summaries], dtype=np.float64)
    weights /= weights.sum()
    shared_cosine, private_cosine, shared_std, private_std = {}, {}, {}, {}
    for modality in MODALITY_NAMES:
        shared_cosine[modality] = float(sum(weight * payload["diagnostics"]["teacher_student_shared_cosine_by_modality"][modality] for weight, payload in zip(weights, payloads)))
        private_cosine[modality] = float(sum(weight * payload["diagnostics"]["teacher_student_private_cosine_by_modality"][modality] for weight, payload in zip(weights, payloads)))
        shared_std[modality] = float(sum(weight * payload["diagnostics"]["shared_latent_batch_std_by_modality"][modality] for weight, payload in zip(weights, payloads)))
        private_std[modality] = float(sum(weight * payload["diagnostics"]["private_latent_batch_std_by_modality"][modality] for weight, payload in zip(weights, payloads)))
    tail_first = float(sum(weight * payload["last20_first10_loss_mean"] for weight, payload in zip(weights, payloads)))
    tail_last = float(sum(weight * payload["last20_last10_loss_mean"] for weight, payload in zip(weights, payloads)))
    return {
        "final_loss_mean": float(sum(weight * payload["final_loss"] for weight, payload in zip(weights, payloads))),
        "final_loss_shared_mean": float(sum(weight * payload["final_loss_shared"] for weight, payload in zip(weights, payloads))),
        "final_loss_private_mean": float(sum(weight * payload["final_loss_private"] for weight, payload in zip(weights, payloads))),
        "teacher_student_shared_cosine_by_modality": shared_cosine,
        "teacher_student_private_cosine_by_modality": private_cosine,
        "shared_latent_batch_std_by_modality": shared_std,
        "private_latent_batch_std_by_modality": private_std,
        "collapse": any(payload["diagnostics"]["collapse"] for payload in payloads),
        "collapse_fold_count": sum(payload["diagnostics"]["collapse"] for payload in payloads),
        "last20_first10_loss_mean": tail_first,
        "last20_last10_loss_mean": tail_last,
        "last20_relative_decline": float((tail_first - tail_last) / max(abs(tail_first), 1e-12)),
    }


def decision(metrics: dict, comparison: dict, boundaries: dict, mechanism: dict):
    no_gain = {
        "correct_at_most_560": metrics["correct"] <= 560,
        "repairs_not_above_damages": comparison["repairs"] <= comparison["damages"],
        "ad_cn_error_present": boundaries["AD_CN"] > 0,
        "latent_collapse": mechanism["collapse"],
        "bacc_drop_over_0p005": metrics["bacc"] < C1["bacc"] - 0.005,
    }
    strong = {
        "correct_at_least_563": metrics["correct"] >= 563,
        "no_ad_cn_error": boundaries["AD_CN"] == 0,
        "bacc_drop_at_most_0p003": metrics["bacc"] >= C1["bacc"] - 0.003,
    }
    positive = {
        "correct_equals_562": metrics["correct"] == 562,
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "no_ad_cn_error": boundaries["AD_CN"] == 0,
        "adjacent_errors_at_most_c1_38": boundaries["AD_SMCI"] + boundaries["CN_SMCI"] <= 38,
        "bacc_drop_at_most_0p003": metrics["bacc"] >= C1["bacc"] - 0.003,
    }
    mechanism_only = {
        "correct_equals_561": metrics["correct"] == 561,
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "macro_f1_or_auc_above_c1": metrics["macro_f1"] > C1["macro_f1"] or metrics["macro_auc"] > C1["macro_auc"],
    }
    checks = {"no_gain": no_gain, "strong": strong, "positive": positive, "mechanism_only": mechanism_only}
    if any(no_gain.values()):
        return "MG_JEPA_NO_GAIN", checks
    if all(strong.values()):
        return "MG_JEPA_STRONG_GO", checks
    if all(positive.values()):
        return "MG_JEPA_POSITIVE_GO", checks
    if all(mechanism_only.values()):
        return "MG_JEPA_MECHANISM_ONLY", checks
    return "MG_JEPA_NO_GAIN", checks


def v1_1_trigger(report: dict):
    metrics = report["metrics"]
    comparison = report["comparison_vs_c1"]
    boundaries = report["boundary_errors"]
    mechanism = report["mechanism"]
    checks = {
        "correct_is_561_or_562": metrics["correct"] in (561, 562),
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "no_ad_cn_error": boundaries["AD_CN"] == 0,
        "bacc_drop_at_most_0p003": metrics["bacc"] >= C1["bacc"] - 0.003,
        "no_collapse": not mechanism["collapse"],
        "last20_relative_decline_at_least_2_percent": mechanism["last20_relative_decline"] >= 0.02,
    }
    return all(checks.values()), checks


def run_version(
    context: dict,
    pretrain_epochs: int,
    version: str,
    version_root: Path,
):
    report_path = version_root / "report.json"
    expected_config = config_payload(context, pretrain_epochs, version, FOLDS)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["config"] == expected_config, "Completed version source/config changed")
        rows = validate_oof(read_csv(version_root / "oof_predictions.csv"))
        require(cme.metrics_from_rows(rows) == report["metrics"], "Completed version OOF changed")
        completed_rows = []
        for fold in FOLDS:
            resumed = load_completed_fold(
                context,
                pretrain_epochs,
                version,
                fold,
                version_root / f"fold_{fold:02d}",
            )
            require(resumed is not None, f"Completed version missing fold{fold}")
            completed_rows.extend(resumed[1])
        require(
            validate_oof(completed_rows) == rows,
            "Completed version fold OOF differs from pooled OOF",
        )
        return report, rows
    version_root.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for fold in FOLDS:
        summary, fold_rows = train_fold(
            context, pretrain_epochs, version, fold, version_root
        )
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = validate_oof(rows)
    metrics = cme.metrics_from_rows(rows)
    comparison = paired_comparison(rows, context["c1_by_subject"])
    boundaries = boundary_errors(metrics)
    mechanism = aggregate_pretrain(summaries)
    result_decision, decision_checks = decision(
        metrics, comparison, boundaries, mechanism
    )
    acc_values = np.asarray(
        [summary["best_metrics"]["acc"] for summary in summaries], dtype=np.float64
    )
    predicted_counts = {
        CLASS_NAMES[class_index]: sum(
            int(row["prediction"]) == class_index for row in rows
        )
        for class_index in range(3)
    }
    report = {
        "version": version,
        "pretrain_epochs": pretrain_epochs,
        "decision": result_decision,
        "decision_checks": decision_checks,
        "metrics": metrics,
        "ten_fold_acc_mean": float(acc_values.mean()),
        "ten_fold_acc_sample_std": float(acc_values.std(ddof=1)),
        "comparison_vs_c1": comparison,
        "boundary_errors": boundaries,
        "predicted_class_counts": predicted_counts,
        "mechanism": mechanism,
        "metric_deltas": {
            "vs_c1": metric_delta(metrics, C1),
            "vs_pc_bbf_v1": metric_delta(metrics, PC_BBF),
        },
        "inference_parameter_count": C1_PARAMETERS,
        "pretrain_seconds": float(
            sum(summary["pretrain_seconds"] for summary in summaries)
        ),
        "supervised_seconds": float(
            sum(summary["supervised_seconds"] for summary in summaries)
        ),
        "fold_wall_seconds": float(
            sum(summary["wall_seconds"] for summary in summaries)
        ),
        "folds": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "best_used_supervised_ema": summary[
                    "best_used_supervised_ema"
                ],
                "correct": summary["best_metrics"]["correct"],
                "acc": summary["best_metrics"]["acc"],
            }
            for summary in summaries
        ],
        "config": expected_config,
    }
    write_json(version_root / "report.json", report)
    write_json(version_root / "mechanism_summary.json", mechanism)
    write_csv(version_root / "oof_predictions.csv", rows)
    write_csv(
        version_root / "fold_metrics.csv",
        [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "best_used_supervised_ema": summary[
                    "best_used_supervised_ema"
                ],
                **{
                    key: value
                    for key, value in summary["best_metrics"].items()
                    if key != "confusion_matrix"
                },
                "pretrain_seconds": summary["pretrain_seconds"],
                "supervised_seconds": summary["supervised_seconds"],
            }
            for summary in summaries
        ],
    )
    return report, rows


def render_report(summary: dict) -> str:
    report = summary["selected_report"]
    metrics = report["metrics"]
    comparison = report["comparison_vs_c1"]
    boundaries = report["boundary_errors"]
    mechanism = report["mechanism"]
    lines = [
        "# MG-JEPA-C1 v1",
        "",
        f"Decision: **{summary['decision']}**",
        f"Selected version / pretraining epochs: **{summary['selected_version']} / {report['pretrain_epochs']}**",
        f"Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: {metrics['correct']}/598 / {metrics['acc']:.7f} / {metrics['macro_f1']:.7f} / {metrics['bacc']:.7f} / {metrics['macro_auc']:.7f} / {metrics['weighted_f1']:.7f}",
        f"Confusion matrix: {metrics['confusion_matrix']}",
        f"Ten-fold ACC mean +/- sample SD: {report['ten_fold_acc_mean']:.7f} +/- {report['ten_fold_acc_sample_std']:.7f}",
        "Delta vs C1 (Correct/ACC/F1/BACC/AUC): "
        f"{report['metric_deltas']['vs_c1']['correct']:+d} / {report['metric_deltas']['vs_c1']['acc']:+.7f} / {report['metric_deltas']['vs_c1']['macro_f1']:+.7f} / {report['metric_deltas']['vs_c1']['bacc']:+.7f} / {report['metric_deltas']['vs_c1']['macro_auc']:+.7f}",
        "Delta vs PC-BBF (Correct/ACC/F1/BACC/AUC): "
        f"{report['metric_deltas']['vs_pc_bbf_v1']['correct']:+d} / {report['metric_deltas']['vs_pc_bbf_v1']['acc']:+.7f} / {report['metric_deltas']['vs_pc_bbf_v1']['macro_f1']:+.7f} / {report['metric_deltas']['vs_pc_bbf_v1']['bacc']:+.7f} / {report['metric_deltas']['vs_pc_bbf_v1']['macro_auc']:+.7f}",
        f"Repairs/damages/changed vs C1: {comparison['repairs']} / {comparison['damages']} / {comparison['changed_predictions']}",
        f"AD-sMCI / CN-sMCI / AD-CN errors: {boundaries['AD_SMCI']} / {boundaries['CN_SMCI']} / {boundaries['AD_CN']}",
        f"Predicted AD/CN/sMCI: {report['predicted_class_counts']['AD']} / {report['predicted_class_counts']['CN']} / {report['predicted_class_counts']['SMCI']}",
        f"Inference parameters: {report['inference_parameter_count']}; pretrain/supervised seconds: {report['pretrain_seconds']:.3f} / {report['supervised_seconds']:.3f}",
        f"Source/device/total wall: {summary['source_commit']} / {summary['device']} / {summary['total_wall_seconds']:.3f}s",
        f"Final shared/private loss: {mechanism['final_loss_shared_mean']:.7f} / {mechanism['final_loss_private_mean']:.7f}; collapse={mechanism['collapse']}",
        f"Pretrained-C1 vs C1 OOF prediction changes: {comparison['changed_predictions']}",
        f"v1.1 run: {summary['v1_1_ran']} ({summary['v1_1_trigger_checks']})",
        "",
    ]
    lines.extend(["## Complete-view latent mechanism", ""])
    for modality in MODALITY_NAMES:
        lines.append(
            f"- {modality}: shared/private cosine "
            f"{mechanism['teacher_student_shared_cosine_by_modality'][modality]:.6f}/"
            f"{mechanism['teacher_student_private_cosine_by_modality'][modality]:.6f}; "
            f"shared/private batch std "
            f"{mechanism['shared_latent_batch_std_by_modality'][modality]:.6f}/"
            f"{mechanism['private_latent_batch_std_by_modality'][modality]:.6f}"
        )
    lines.extend(["", "## Per-fold best", ""])
    for fold in report["folds"]:
        lines.append(
            f"- fold {fold['fold']}: best_epoch={fold['best_epoch']}, ACC={fold['acc']:.7f}"
        )
    lines.extend(["", summary["next_conclusion"]])
    return "\n".join(lines) + "\n"


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(
        git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode
        == 0,
        "Source/smoke files are not committed",
    )
    require(
        git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode
        == 0,
        "Locked source/smoke files changed",
    )


def run_formal(context: dict, output_root: Path) -> dict:
    smoke_path = output_root / "smoke/smoke_report.json"
    smoke_config = output_root / "config.json"
    require(smoke_path.is_file() and smoke_config.is_file(), "Committed passing smoke is required")
    smoke_report = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke_configuration = json.loads(smoke_config.read_text(encoding="utf-8"))
    require(smoke_report.get("passed") is True, "Smoke did not pass")
    current_hashes = locked_source_hashes()
    require(smoke_report.get("source_content_sha256") == current_hashes, "Committed source differs from smoke-tested source")
    require(smoke_configuration.get("locked_source_sha256") == current_hashes, "Committed smoke config/source hash differs")
    require_committed_unchanged(
        [
            ROOT / "Model/mg_jepa_c1.py",
            ROOT / "scripts/run_mg_jepa_c1_v1.py",
            ROOT / cme.CONFIG_REL,
            smoke_path,
            smoke_config,
        ]
    )
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    formal_started = time.perf_counter()
    v1_report, v1_rows = run_version(
        context,
        DEFAULT_PRETRAIN_EPOCHS,
        "v1",
        output_root / "formal_v1_pretrain_200",
    )
    trigger, trigger_checks = v1_1_trigger(v1_report)
    v1_1_report = None
    selected_report, selected_rows = v1_report, v1_rows
    if trigger:
        v1_1_report, v1_1_rows = run_version(
            context,
            V1_1_PRETRAIN_EPOCHS,
            "v1_1",
            output_root / "formal_v1_1_pretrain_400",
        )
        # v1.1 is the sole pre-registered continuation.  If its all-AND gate
        # opens, it is the final registered version; no OOF post-hoc selection.
        selected_report, selected_rows = v1_1_report, v1_1_rows
    summary = {
        "decision": selected_report["decision"],
        "selected_version": selected_report["version"],
        "selected_report": selected_report,
        "v1_report": v1_report,
        "v1_1_ran": trigger,
        "v1_1_trigger_checks": trigger_checks,
        "v1_1_report": v1_1_report,
        "v1_1_selection_rule": "if triggered, v1.1 is the final pre-registered version; v1 remains reported separately",
        "source_commit": source_commit,
        "device": str(context["device"]),
        "device_name": torch.cuda.get_device_name(0),
        "pretrain_seconds_all_versions": float(
            v1_report["pretrain_seconds"]
            + (v1_1_report["pretrain_seconds"] if v1_1_report else 0.0)
        ),
        "supervised_seconds_all_versions": float(
            v1_report["supervised_seconds"]
            + (v1_1_report["supervised_seconds"] if v1_1_report else 0.0)
        ),
        "total_wall_seconds": float(time.perf_counter() - formal_started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_mg_jepa_c1_v1.py formal --device cuda:0',
        "next_conclusion": (
            "Target reached; retain MG-JEPA for paper ablation and stop tuning."
            if selected_report["decision"] == "MG_JEPA_STRONG_GO"
            else "Stop after the registered result; do not tune another MG-JEPA setting."
        ),
    }
    write_json(output_root / "config.json", selected_report["config"])
    write_json(output_root / "formal_summary.json", summary)
    write_json(output_root / "mechanism_summary.json", selected_report["mechanism"])
    write_csv(output_root / "oof_predictions.csv", selected_rows)
    write_csv(
        output_root / "fold_metrics.csv",
        [
            {
                "fold": fold["fold"],
                "best_epoch": fold["best_epoch"],
                "best_used_supervised_ema": fold[
                    "best_used_supervised_ema"
                ],
                "correct": fold["correct"],
                "acc": fold["acc"],
            }
            for fold in selected_report["folds"]
        ],
    )
    (output_root / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def verify_branch() -> None:
    require(git("cat-file", "-e", f"{BASE_COMMIT}^{{commit}}", check=False).returncode == 0, "Base commit missing")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD", check=False).returncode == 0, "Branch is not based on C1")
    require(git("branch", "--show-current").stdout.strip() == "experiment/mg-jepa-c1-v1", "Wrong branch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_branch()
    context = load_context(args.device)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        result = run_smoke(context, output_root)
        print(f"SMOKE passed={result['passed']}", flush=True)
    else:
        result = run_formal(context, output_root)
        print(f"{result['decision']} selected={result['selected_version']}", flush=True)


if __name__ == "__main__":
    main()
