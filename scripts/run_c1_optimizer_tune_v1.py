"""Run the locked three-arm C1 optimizer tune experiment.

Only Adam learning rate and weight decay differ across T1/T2/T3.  The real
tracked C1 baseline has EMA disabled; this runner preserves that fact so the
experiment remains a two-coordinate optimizer intervention and records the
conflicting 20/21 EMA statement as not applied.
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
from Utils import CustomCosineAnnealingLR, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "c1_optimizer_tune_v1"
OUTPUT_REL = Path("experiments/c1_optimizer_tune_v1")
BASE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
C1_OOF_REL = Path("experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv")
CLASS_NAMES = ("AD", "CN", "SMCI")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
EXPECTED_PARAMETERS = 862_971
ARMS = {
    "T1": {"lr": 0.0075, "weight_decay": 0.0005},
    "T2": {"lr": 0.0100, "weight_decay": 0.00025},
    "T3": {"lr": 0.0075, "weight_decay": 0.00025},
}
BASE_OPTIMIZER = {"lr": 0.0100, "weight_decay": 0.0005}
C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
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


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def grads_finite(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "C1 optimizer tune requires cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "Historical device changed")
    config = context["config"]
    require(tuple(context["dataset_dict"]["Class_Names"]) == CLASS_NAMES, "Class order changed")
    require(int(config.epochs) == EPOCHS and int(config.T_max) == EPOCHS, "Epoch protocol changed")
    require(abs(float(config.lr) - BASE_OPTIMIZER["lr"]) < 1e-12, "Actual base lr changed")
    require(abs(float(config.weight_decay) - BASE_OPTIMIZER["weight_decay"]) < 1e-12, "Actual base weight decay changed")
    require(abs(float(config.Lr_Min) - 0.0001) < 1e-12, "Historical eta_min changed")
    require(abs(float(config.logit_adjust_tau) - 0.75) < 1e-12, "Historical logit-adjust tau changed")
    require(abs(float(config.grad_clip) - 1.0) < 1e-12, "Historical gradient clip changed")
    require(config.use_ema is False, "Tracked C1 baseline EMA fact changed")
    c1_path = ROOT / C1_OOF_REL
    require(c1_path.is_file(), f"C1 OOF missing: {c1_path}")
    c1_rows = read_csv(c1_path)
    require(len(c1_rows) == 598, "C1 OOF row count changed")
    c1_metrics = cme.metrics_from_rows(c1_rows)
    require(c1_metrics["correct"] == C1["correct"] and c1_metrics["confusion_matrix"] == C1["confusion_matrix"], "C1 anchor changed")
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


def build_model(context: dict):
    SET_Random(SEED)
    model = cme.build_model(context, "c1")
    require(parameter_count(model) == EXPECTED_PARAMETERS, "C1 parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Historical graph unexpectedly enabled")
    return model


def make_training_objects(context: dict, arm: str):
    require(arm in ARMS, f"Unknown optimizer arm: {arm}")
    model = build_model(context)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ARMS[arm]["lr"],
        weight_decay=ARMS[arm]["weight_decay"],
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=float(context["config"].Lr_Min),
    )
    require(len(optimizer.param_groups) == 1, f"{arm}: optimizer param-group count changed")
    group = optimizer.param_groups[0]
    require(abs(float(group["lr"]) - ARMS[arm]["lr"]) < 1e-12 and abs(float(group["weight_decay"]) - ARMS[arm]["weight_decay"]) < 1e-12, f"{arm}: effective lr/wd changed")
    require(int(scheduler.T_max) == EPOCHS and abs(float(scheduler.eta_min) - 0.0001) < 1e-12, f"{arm}: scheduler changed")
    require(optimizer.defaults["betas"] == (0.9, 0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler


def canonical_protocol(context: dict, arm: str, folds) -> dict:
    require(arm in ARMS, f"Unknown arm: {arm}")
    return {
        "experiment": EXPERIMENT,
        "arm": arm,
        "base_commit": BASE_COMMIT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(folds),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "parameter_count": EXPECTED_PARAMETERS,
        "optimizer": "Adam",
        "learning_rate": ARMS[arm]["lr"],
        "weight_decay": ARMS[arm]["weight_decay"],
        "adam_betas": [0.9, 0.999],
        "adam_eps": 1e-8,
        "scheduler": "CustomCosineAnnealingLR",
        "scheduler_t_max": EPOCHS,
        "scheduler_eta_min": float(context["config"].Lr_Min),
        "label_smoothing": 0.05,
        "logit_adjust_tau": float(context["config"].logit_adjust_tau),
        "gradient_clip": float(context["config"].grad_clip),
        "loss": "historical C1 weighted main CE plus three OVR auxiliary losses",
        "orthogonality": False,
        "graph_enabled": False,
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "multi_seed": False,
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "tracked_base_use_ema": False,
        "effective_use_ema": False,
        "ema_20_21_conflict_disposition": "not applied; tracked C1=560 runner/config has EMA disabled, preserving lr/wd-only attribution",
    }


def config_payload(context: dict, arm: str, folds) -> dict:
    protocol = canonical_protocol(context, arm, folds)
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **protocol,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "canonical_config_sha256": hashlib.sha256(encoded).hexdigest(),
        "historical_config_path": str(cme.CONFIG_REL).replace("\\", "/"),
        "historical_config_sha256": file_sha256(ROOT / cme.CONFIG_REL),
        "runner_sha256": file_sha256(ROOT / "scripts/run_c1_optimizer_tune_v1.py"),
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
    }


def metric_delta(candidate: dict, baseline: dict) -> dict:
    return {
        "correct": int(candidate["correct"] - baseline["correct"]),
        **{
            key: float(candidate[key] - baseline[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
        },
    }


def paired_comparison(rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"C1 missing subject {subject}")
        reference = reference_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["fold"]) == int(row["fold"]) and int(reference["truth"]) == truth, "C1 OOF alignment changed")
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


def boundary_errors(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        "AD_SMCI": int(matrix[0, 2] + matrix[2, 0]),
        "CN_SMCI": int(matrix[1, 2] + matrix[2, 1]),
        "AD_CN": int(matrix[0, 1] + matrix[1, 0]),
    }


def validate_oof(rows: list[dict]) -> list[dict]:
    require(len(rows) == 598, "OOF row count changed")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == 598 and set(subjects) == set(range(598)), "OOF subject coverage changed")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "OOF fold set changed")
    validate_prediction_rows(rows)
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def validate_prediction_rows(rows: list[dict], expected_fold: int | None = None) -> None:
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(subjects) == len(set(subjects)), "Prediction rows contain duplicate subjects")
    for row in rows:
        if expected_fold is not None:
            require(int(row["fold"]) == expected_fold, "Prediction row has wrong fold")
        truth = int(row["truth"])
        prediction = int(row["prediction"])
        require(0 <= truth < 3 and 0 <= prediction < 3, "Prediction row has invalid class index")
        probabilities = np.asarray(
            [float(row[f"probability_{name}"]) for name in CLASS_NAMES],
            dtype=np.float64,
        )
        require(bool(np.isfinite(probabilities).all()), "Prediction row has non-finite probability")
        require(abs(float(probabilities.sum()) - 1.0) <= 1e-5, "Prediction probabilities do not sum to one")
        require(int(probabilities.argmax()) == prediction, "Prediction is not probability argmax")


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    started = time.perf_counter()
    model, criterion, optimizer, scheduler = make_training_objects(context, "T1")
    require(len(optimizer.param_groups) == 1, "T1 optimizer param-group count changed")
    group = optimizer.param_groups[0]
    require(abs(float(group["lr"]) - 0.0075) < 1e-12 and abs(float(group["weight_decay"]) - 0.0005) < 1e-12, "T1 smoke optimizer values changed")
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]
    losses, gradient_norms = [], []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"T1 smoke epoch{epoch}: non-finite loss")
        loss.backward()
        require(grads_finite(model), f"T1 smoke epoch{epoch}: non-finite gradient")
        squared = sum(parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.grad is not None)
        gradient_norm = float(torch.sqrt(squared).cpu())
        require(gradient_norm > 0 and math.isfinite(gradient_norm), f"T1 smoke epoch{epoch}: no finite nonzero gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(gradient_norm)
    model.eval()
    with torch.no_grad():
        reference, _, _ = model(features)
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save({"model_state": clone_cpu_state(model), "config": config_payload(context, "T1", [0])}, checkpoint_path)
    reloaded = build_model(context)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        loaded, _, _ = reloaded(features)
    roundtrip = float((reference - loaded).abs().max().cpu())
    require(roundtrip <= 1e-7, "T1 smoke checkpoint roundtrip failed")
    report = {
        "passed": True,
        "arm": "T1",
        "fold": 0,
        "epochs": 3,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "parameter_count": parameter_count(model),
        "optimizer_param_groups": [{"lr": 0.0075, "weight_decay": 0.0005}],
        "effective_use_ema": False,
        "checkpoint_roundtrip_logit_max_abs_diff": roundtrip,
        "run_head_at_smoke": git("rev-parse", "HEAD").stdout.strip(),
        "runner_sha256": file_sha256(ROOT / "scripts/run_c1_optimizer_tune_v1.py"),
        "wall_seconds": float(time.perf_counter() - started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_c1_optimizer_tune_v1.py smoke --device cuda:0',
    }
    write_json(output_root / "config.json", config_payload(context, "T1", [0]))
    write_json(smoke_root / "smoke_report.json", report)
    return report


def load_completed_fold(context: dict, arm: str, fold: int, final_dir: Path):
    if not final_dir.exists():
        return None
    required = [final_dir / name for name in ("config.json", "summary.json", "epoch_metrics.csv", "oof_predictions.csv", "checkpoint_best.pt", "complete.json")]
    require(all(path.is_file() for path in required), f"{arm} fold{fold}: incomplete completed artifact")
    expected = config_payload(context, arm, [fold])
    require(json.loads((final_dir / "config.json").read_text(encoding="utf-8")) == expected, f"{arm} fold{fold}: config/source changed")
    marker = json.loads((final_dir / "complete.json").read_text(encoding="utf-8"))
    require(marker == {"complete": True, "arm": arm, "fold": fold, "source_commit": expected["source_commit"]}, f"{arm} fold{fold}: marker changed")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    epochs = read_csv(final_dir / "epoch_metrics.csv")
    rows = read_csv(final_dir / "oof_predictions.csv")
    require(summary["config"] == expected and len(epochs) == EPOCHS, f"{arm} fold{fold}: summary/history changed")
    require([int(row["epoch"]) for row in epochs] == list(range(1, EPOCHS + 1)), f"{arm} fold{fold}: epoch sequence changed")
    tuples = [(float(row["acc"]), float(row["macro_auc"]), float(row["macro_f1"])) for row in epochs]
    best_index = max(range(EPOCHS), key=lambda index: tuples[index])
    require(best_index + 1 == summary["best_epoch"] and tuples[best_index] == (summary["best_metrics"]["acc"], summary["best_metrics"]["macro_auc"], summary["best_metrics"]["macro_f1"]), f"{arm} fold{fold}: best selection changed")
    require(len(rows) == summary["test_size"] and cme.metrics_from_rows(rows) == summary["best_metrics"], f"{arm} fold{fold}: OOF changed")
    validate_prediction_rows(rows, expected_fold=fold)
    reference = {int(row["subject_index"]): row for row in context["c1_rows"] if int(row["fold"]) == fold}
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in reference and int(row["fold"]) == fold and int(row["truth"]) == int(reference[subject]["truth"]), f"{arm} fold{fold}: OOF alignment changed")
    checkpoint = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == expected and checkpoint["best_epoch"] == summary["best_epoch"], f"{arm} fold{fold}: checkpoint metadata changed")
    check_model = build_model(context)
    check_model.load_state_dict(checkpoint["model_state"], strict=True)
    del check_model, checkpoint
    torch.cuda.empty_cache()
    print(f"RESUME {arm} fold={fold} correct={summary['best_metrics']['correct']}", flush=True)
    return summary, rows


def train_fold(context: dict, arm: str, fold: int, arm_root: Path):
    final_dir = arm_root / f"fold_{fold:02d}"
    resumed = load_completed_fold(context, arm, fold, final_dir)
    if resumed is not None:
        return resumed
    staging = arm_root / f".fold_{fold:02d}_in_progress"
    expected = config_payload(context, arm, [fold])
    if staging.exists():
        require((staging / "config.json").is_file(), f"{arm} fold{fold}: in-progress config missing")
        require(json.loads((staging / "config.json").read_text(encoding="utf-8")) == expected, f"{arm} fold{fold}: in-progress config/source changed")
        resolved = staging.resolve()
        require(resolved.parent == arm_root.resolve() and resolved.name == f".fold_{fold:02d}_in_progress", "Unsafe in-progress restart target")
        shutil.rmtree(resolved)
        print(f"RESTART_INCOMPLETE {arm} fold={fold}", flush=True)
    staging.mkdir(parents=True)
    write_json(staging / "config.json", expected)
    model, criterion, optimizer, scheduler = make_training_objects(context, arm)
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    best = None
    best_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"{arm} fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(grads_finite(model), f"{arm} fold{fold}: non-finite gradient")
        if float(context["config"].grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(context["config"].grad_clip))
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            evaluated, _, _ = model(features)
            metrics, selection = cme.selection_metrics(evaluated, labels, test_mask, context["dataset_dict"]["Label_Weight"], float(context["config"].logit_adjust_tau))
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = clone_cpu_state(model)
        epoch_rows.append({"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "loss": float(loss.detach().cpu()), **{key: metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")}})
    require(best is not None and best_state is not None, f"{arm} fold{fold}: best state missing")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _ = model(features)
    rows = cme.prediction_rows(fold, best_raw, labels, test_mask, context["dataset_dict"], context["config"])
    require(cme.metrics_from_rows(rows) == best["metrics"], f"{arm} fold{fold}: best readback changed")
    summary = {
        "arm": arm,
        "fold": fold,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "parameter_count": parameter_count(model),
        "optimizer": {"lr": ARMS[arm]["lr"], "weight_decay": ARMS[arm]["weight_decay"]},
        "effective_use_ema": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "config": expected,
    }
    torch.save({"model_state": best_state, "best_epoch": best["epoch"], "config": expected}, staging / "checkpoint_best.pt")
    write_json(staging / "summary.json", summary)
    write_csv(staging / "epoch_metrics.csv", epoch_rows)
    write_csv(staging / "oof_predictions.csv", rows)
    write_json(staging / "complete.json", {"complete": True, "arm": arm, "fold": fold, "source_commit": expected["source_commit"]})
    staging.rename(final_dir)
    print(f"{arm} fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']} ACC={best['metrics']['acc']:.7f} F1={best['metrics']['macro_f1']:.7f} BACC={best['metrics']['bacc']:.7f} AUC={best['metrics']['macro_auc']:.7f}", flush=True)
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def run_arm(context: dict, arm: str, arm_root: Path):
    report_path = arm_root / "report.json"
    complete_path = arm_root / "complete.json"
    expected = config_payload(context, arm, FOLDS)
    if report_path.is_file() and complete_path.is_file():
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        require(marker == {"complete": True, "arm": arm, "source_commit": expected["source_commit"]}, f"{arm}: completion marker changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["config"] == expected, f"{arm}: completed config/source changed")
        pooled = validate_oof(read_csv(arm_root / "oof_predictions.csv"))
        require(cme.metrics_from_rows(pooled) == report["metrics"], f"{arm}: pooled OOF changed")
        rebuilt = []
        for fold in FOLDS:
            resumed = load_completed_fold(context, arm, fold, arm_root / f"fold_{fold:02d}")
            require(resumed is not None, f"{arm}: missing completed fold{fold}")
            rebuilt.extend(resumed[1])
        require(validate_oof(rebuilt) == pooled, f"{arm}: fold/pooled OOF mismatch")
        return report, pooled
    if report_path.is_file() or complete_path.is_file():
        if complete_path.is_file():
            marker = json.loads(complete_path.read_text(encoding="utf-8"))
            require(marker == {"complete": True, "arm": arm, "source_commit": expected["source_commit"]}, f"{arm}: stale completion marker changed")
        print(f"REBUILD_ARM_AGGREGATE {arm}", flush=True)
    arm_root.mkdir(parents=True, exist_ok=True)
    arm_config_path = arm_root / "config.json"
    if arm_config_path.is_file():
        require(json.loads(arm_config_path.read_text(encoding="utf-8")) == expected, f"{arm}: arm config/source changed")
    else:
        write_json(arm_config_path, expected)
    summaries, rows = [], []
    for fold in FOLDS:
        summary, fold_rows = train_fold(context, arm, fold, arm_root)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = validate_oof(rows)
    metrics = cme.metrics_from_rows(rows)
    comparison = paired_comparison(rows, context["c1_by_subject"])
    require(
        comparison["repairs"] - comparison["damages"]
        == metrics["correct"] - C1["correct"],
        f"{arm}: paired net repairs does not match correct delta",
    )
    boundaries = boundary_errors(metrics)
    acc = np.asarray([summary["best_metrics"]["acc"] for summary in summaries], dtype=np.float64)
    predicted_counts = {
        CLASS_NAMES[index]: sum(int(row["prediction"]) == index for row in rows)
        for index in range(3)
    }
    report = {
        "arm": arm,
        "metrics": metrics,
        "ten_fold_acc_mean": float(acc.mean()),
        "ten_fold_acc_sample_std": float(acc.std(ddof=1)),
        "metric_delta_vs_c1": metric_delta(metrics, C1),
        "comparison_vs_c1": comparison,
        "boundary_errors": boundaries,
        "predicted_class_counts": predicted_counts,
        "bacc_risk": metrics["bacc"] < C1["bacc"] - 0.005,
        "parameter_count": EXPECTED_PARAMETERS,
        "effective_use_ema": False,
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "folds": [
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "correct": summary["best_metrics"]["correct"],
                "acc": summary["best_metrics"]["acc"],
            }
            for summary in summaries
        ],
        "config": expected,
    }
    write_csv(arm_root / "oof_predictions.csv", rows)
    write_csv(
        arm_root / "fold_metrics.csv",
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
            }
            for summary in summaries
        ],
    )
    matrix = metrics["confusion_matrix"]
    write_csv(
        arm_root / "confusion_matrix.csv",
        [
            {"truth": CLASS_NAMES[index], **{f"pred_{name}": matrix[index][column] for column, name in enumerate(CLASS_NAMES)}}
            for index in range(3)
        ],
    )
    write_json(arm_root / "report.json", report)
    write_json(
        arm_root / "complete.json",
        {"complete": True, "arm": arm, "source_commit": expected["source_commit"]},
    )
    return report, rows


def final_decision(selected: dict):
    metrics = selected["metrics"]
    secondary_above = sum(
        metrics[key] > C1[key] for key in ("macro_f1", "bacc", "macro_auc")
    )
    checks = {
        "strong_correct_at_least_563": metrics["correct"] >= 563,
        "positive_correct_equals_562": metrics["correct"] == 562,
        "weak_correct_equals_561": metrics["correct"] == 561,
        "weak_secondary_metrics_above_c1_at_least_two": secondary_above >= 2,
        "bacc_drop_over_0p005_risk": selected["bacc_risk"],
    }
    if metrics["correct"] >= 563:
        return "C1_TUNE_STRONG_GO", checks
    if metrics["correct"] == 562:
        return "C1_TUNE_POSITIVE", checks
    if metrics["correct"] == 561 and secondary_above >= 2:
        return "C1_TUNE_WEAK_GO", checks
    return "C1_TUNE_NO_GAIN", checks


def render_report(summary: dict) -> str:
    selected = summary["selected_report"]
    metrics = selected["metrics"]
    lines = [
        "# C1 Optimizer Tune v1",
        "",
        f"Decision: **{summary['decision']}**",
        f"Selected arm: **{summary['selected_arm']}** (Correct > AUC > Macro-F1)",
        f"Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: {metrics['correct']}/598 / {metrics['acc']:.7f} / {metrics['macro_f1']:.7f} / {metrics['bacc']:.7f} / {metrics['macro_auc']:.7f} / {metrics['weighted_f1']:.7f}",
        f"Confusion matrix: {metrics['confusion_matrix']}",
        f"BACC risk (drop >0.005 vs C1): {selected['bacc_risk']}",
        f"Parameters: {selected['parameter_count']}; total training/wall seconds: {summary['total_training_seconds']:.3f} / {summary['total_wall_seconds']:.3f}",
        f"Source/device: {summary['source_commit']} / {summary['device_name']}",
        f"Actual C1 base optimizer: lr={BASE_OPTIMIZER['lr']}, weight_decay={BASE_OPTIMIZER['weight_decay']}",
        "EMA protocol: disabled. The requested 20/21 schedule conflicts with the tracked C1=560 runner/config; it was not applied to preserve lr/wd-only attribution.",
        "",
        "## Arms",
        "",
    ]
    for arm in ARMS:
        report = summary["arm_reports"][arm]
        arm_metrics = report["metrics"]
        delta = report["metric_delta_vs_c1"]
        paired = report["comparison_vs_c1"]
        boundary = report["boundary_errors"]
        predicted = report["predicted_class_counts"]
        lines.extend(
            [
                f"### {arm}",
                "",
                f"- lr / weight decay: {ARMS[arm]['lr']} / {ARMS[arm]['weight_decay']}",
                f"- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: {arm_metrics['correct']}/598 / {arm_metrics['acc']:.7f} / {arm_metrics['macro_f1']:.7f} / {arm_metrics['bacc']:.7f} / {arm_metrics['macro_auc']:.7f} / {arm_metrics['weighted_f1']:.7f}",
                f"- Confusion matrix: {arm_metrics['confusion_matrix']}",
                f"- Ten-fold ACC mean +/- sample SD: {report['ten_fold_acc_mean']:.7f} +/- {report['ten_fold_acc_sample_std']:.7f}",
                f"- Delta vs C1 (Correct/ACC/F1/BACC/AUC/Weighted-F1): {delta['correct']:+d} / {delta['acc']:+.7f} / {delta['macro_f1']:+.7f} / {delta['bacc']:+.7f} / {delta['macro_auc']:+.7f} / {delta['weighted_f1']:+.7f}",
                f"- Repairs/damages/changed: {paired['repairs']} / {paired['damages']} / {paired['changed_predictions']}",
                f"- AD-sMCI / CN-sMCI / AD-CN errors: {boundary['AD_SMCI']} / {boundary['CN_SMCI']} / {boundary['AD_CN']}",
                f"- Predicted AD/CN/sMCI: {predicted['AD']} / {predicted['CN']} / {predicted['SMCI']}",
                f"- BACC risk: {report['bacc_risk']}; training seconds: {report['training_seconds']:.3f}",
                "- Fold best epoch/ACC: " + ", ".join(
                    f"{fold['fold']}:{fold['best_epoch']}/{fold['acc']:.7f}"
                    for fold in report["folds"]
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode == 0, "Source/smoke files are not committed")
    require(git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode == 0, "Locked source/smoke files changed")


def run_formal(context: dict, output_root: Path):
    smoke_report_path = output_root / "smoke/smoke_report.json"
    smoke_config_path = output_root / "config.json"
    require(smoke_report_path.is_file() and smoke_config_path.is_file(), "Passing committed T1 smoke is required")
    smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    require(smoke.get("passed") is True and smoke.get("arm") == "T1" and smoke.get("fold") == 0, "Invalid smoke report")
    current_runner_hash = file_sha256(ROOT / "scripts/run_c1_optimizer_tune_v1.py")
    require(smoke.get("runner_sha256") == current_runner_hash, "Committed runner differs from smoke-tested runner")
    require(smoke_config.get("runner_sha256") == current_runner_hash, "Committed smoke config differs from smoke-tested runner")
    require_committed_unchanged(
        [
            ROOT / "scripts/run_c1_optimizer_tune_v1.py",
            ROOT / cme.CONFIG_REL,
            smoke_report_path,
            smoke_config_path,
        ]
    )
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    started = time.perf_counter()
    arm_reports, arm_rows = {}, {}
    for arm in ("T1", "T2", "T3"):
        report, rows = run_arm(context, arm, output_root / arm)
        arm_reports[arm] = report
        arm_rows[arm] = rows
    selected_arm = max(
        ARMS,
        key=lambda arm: (
            arm_reports[arm]["metrics"]["correct"],
            arm_reports[arm]["metrics"]["macro_auc"],
            arm_reports[arm]["metrics"]["macro_f1"],
        ),
    )
    selected = arm_reports[selected_arm]
    decision, checks = final_decision(selected)
    summary = {
        "decision": decision,
        "decision_checks": checks,
        "selected_arm": selected_arm,
        "selected_report": selected,
        "arm_reports": arm_reports,
        "arm_ranking": sorted(
            ARMS,
            key=lambda arm: (
                arm_reports[arm]["metrics"]["correct"],
                arm_reports[arm]["metrics"]["macro_auc"],
                arm_reports[arm]["metrics"]["macro_f1"],
            ),
            reverse=True,
        ),
        "source_commit": source_commit,
        "device": str(context["device"]),
        "device_name": torch.cuda.get_device_name(0),
        "parameter_count": EXPECTED_PARAMETERS,
        "effective_use_ema": False,
        "ema_conflict_disposition": "20/21 schedule not applied because tracked C1=560 has use_ema=false; preserves lr/wd-only attribution",
        "total_training_seconds": float(sum(report["training_seconds"] for report in arm_reports.values())),
        "total_wall_seconds": float(time.perf_counter() - started),
        "run_command": f'"{sys.executable}" -u -B scripts/run_c1_optimizer_tune_v1.py formal --device cuda:0',
    }
    suite_config = {
        "experiment": EXPERIMENT,
        "source_commit": source_commit,
        "base_commit": BASE_COMMIT,
        "c1_reference": C1,
        "actual_base_optimizer": BASE_OPTIMIZER,
        "arms_in_execution_order": ["T1", "T2", "T3"],
        "arm_multipliers": {
            "T1": {"lr": 0.75, "weight_decay": 1.0},
            "T2": {"lr": 1.0, "weight_decay": 0.5},
            "T3": {"lr": 0.75, "weight_decay": 0.5},
        },
        "arm_configs": {arm: arm_reports[arm]["config"] for arm in ARMS},
        "selection_rule": ["Correct", "Probability Macro-AUC", "Macro-F1"],
        "effective_use_ema": False,
        "ema_conflict_disposition": summary["ema_conflict_disposition"],
    }
    # Preserve the committed smoke config so interrupted/completed formal runs
    # can always re-enter the strict smoke/source gate.
    write_json(output_root / "formal_config.json", suite_config)
    write_json(output_root / "FINAL_REPORT.json", summary)
    (output_root / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    write_json(output_root / "selected_config.json", selected["config"])
    write_csv(output_root / "selected_oof_predictions.csv", arm_rows[selected_arm])
    return summary


def verify_branch() -> None:
    require(git("cat-file", "-e", f"{BASE_COMMIT}^{{commit}}", check=False).returncode == 0, "Base commit missing")
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD", check=False).returncode == 0, "Branch is not based on C1")
    require(git("branch", "--show-current").stdout.strip() == "experiment/c1-optimizer-tune-v1", "Wrong branch")


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
        print(f"{result['decision']} selected={result['selected_arm']}", flush=True)


if __name__ == "__main__":
    main()
