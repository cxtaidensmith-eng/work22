from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Loss import criterion_query_pool_no_orth
from Utils import CustomCosineAnnealingLR, SET_Random
import run_class_private_evidence_graph_v1 as prior
import run_osfq_id_v1 as common
import run_query_pool_component_sharing_v1 as seps


EXPERIMENT = "Norm-Capped Reader v1"
ARM = "RCAP"
BRANCH = "experiment/norm-capped-reader-v1"
SOURCE_HEAD = "af2957a68cbcc926a84d34de020b83b42417ddda"
OUTPUT_ROOT = Path("experiments/norm_capped_reader_v1")
REPORT_PATH = Path("reports/norm_capped_reader_v1_10fold_final.md")
SEED = 0
EPOCHS = 400
FOLDS = tuple(range(10))
READER_RANK = 8
READER_NORM_CAP = 0.5
PARAMETERS = 785_187
CLASS_ORDER = ("AD", "CN", "SMCI")
METRICS = ("acc", "macro_f1", "bacc", "probability_macro_auc", "weighted_f1")
SOURCE_FILES = (
    Path("Model/network.py"),
    Path("scripts/run_query_pool_component_sharing_v1.py"),
    Path("scripts/run_norm_capped_reader_v1.py"),
)
R_OOF = Path("experiments/class_private_evidence_graph_v1/r/tenfold_seed0/oof_predictions.csv")
SEPS_REFERENCE = {
    "correct": 553,
    "acc": 0.9247492,
    "macro_f1": 0.9138657,
    "bacc": 0.9082064,
    "probability_macro_auc": 0.9532899,
    "weighted_f1": 0.9246314,
}
R_REFERENCE = {
    "correct": 554,
    "acc": 0.9264214,
    "macro_f1": 0.9173926,
    "bacc": 0.9158708,
    "probability_macro_auc": 0.9554113,
    "weighted_f1": 0.9264254,
}
ORIGINAL_REFERENCE = {
    "correct": 556,
    "acc": 0.9297659,
    "macro_f1": 0.9140778,
    "bacc": 0.9140778,
    "probability_macro_auc": 0.9560491,
    "weighted_f1": 0.9297659,
}


def require(condition: bool, message: str) -> None:
    common.require(condition, message)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256(ROOT / path) for path in SOURCE_FILES}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_scope() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch == BRANCH, f"unexpected branch: {branch}")
    require(head == SOURCE_HEAD, f"unexpected source HEAD: {head}")
    require((ROOT / R_OOF).is_file(), "locked R OOF is missing")
    return {
        "branch": branch,
        "execution_source_commit_sha": head,
        "worktree": str(ROOT),
        "source_hashes": source_hashes(),
    }


def build_model(config, dataset_dict, device):
    model = seps.build_model(
        config,
        dataset_dict,
        device,
        seps.VARIANT,
        low_rank_reader=True,
        class_graph=False,
        reader_norm_cap=READER_NORM_CAP,
    )
    require(parameter_count(model) == PARAMETERS, "RCAP parameter count changed")
    module = model.class_private_low_rank_reader
    require(module.rank == READER_RANK, "reader rank changed")
    require(module.norm_cap == READER_NORM_CAP, "reader norm cap changed")
    require(not model.class_graph_enabled, "graph module must remain disabled")
    return model


def cap_snapshot(model) -> dict:
    module = model.class_private_low_rank_reader
    require(
        module.last_uncapped_residuals is not None
        and module.last_residuals is not None
        and module.last_shared_attention is not None
        and module.last_cap_scales is not None,
        "reader cap diagnostics missing",
    )
    uncapped = module.last_uncapped_residuals.float()
    capped = module.last_residuals.float()
    shared = module.last_shared_attention.float()
    scales = module.last_cap_scales.float()
    shared_norm = shared.norm(dim=-1).clamp_min(1e-12)
    uncapped_ratio = uncapped.norm(dim=-1) / shared_norm
    capped_ratio = capped.norm(dim=-1) / shared_norm
    result = {
        "reader_norm_cap": READER_NORM_CAP,
        "classes": {},
        "maximum_capped_ratio": float(capped_ratio.max().cpu()),
    }
    for index, class_name in enumerate(CLASS_ORDER):
        result["classes"][class_name] = {
            "mean_uncapped_residual_to_shared_ratio": float(
                uncapped_ratio[:, index].mean().cpu()
            ),
            "mean_capped_residual_to_shared_ratio": float(
                capped_ratio[:, index].mean().cpu()
            ),
            "maximum_capped_residual_to_shared_ratio": float(
                capped_ratio[:, index].max().cpu()
            ),
            "capped_subject_fraction": float(
                (scales[:, index, 0] < 1.0).float().mean().cpu()
            ),
        }
    require(result["maximum_capped_ratio"] <= 0.50001, "reader cap bound failed")
    return result


def run_validation() -> dict:
    scope = validate_scope()
    config, dataset_dict, dataset_data, device, split_hashes = prior.load_context()
    output = ROOT / OUTPUT_ROOT / "validation"
    if output.exists():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        require(summary.get("passed") is True, "existing validation did not pass")
        require(summary.get("source_hashes") == source_hashes(), "validation source changed")
        return summary
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    train_mask = dataset_data["Mask"][0][0]

    SET_Random(SEED)
    baseline = prior.build_model(config, dataset_dict, device, "B")
    baseline_state = common.clone_cpu_state(baseline)
    baseline.eval()
    with torch.no_grad():
        baseline_logits = baseline(features)[0]

    SET_Random(SEED)
    model = build_model(config, dataset_dict, device)
    candidate = model.state_dict()
    common_names = sorted(baseline_state)
    require(all(name in candidate for name in common_names), "RCAP lost SEPS-Q state")
    common_max_diff = max(
        float((candidate[name].detach().cpu() - baseline_state[name]).abs().max())
        for name in common_names
    )
    require(common_max_diff == 0.0, "RCAP changed common initialization")
    incompatible = model.load_state_dict(baseline_state, strict=False)
    require(not incompatible.unexpected_keys, "unexpected base state keys")
    model.eval()
    with torch.no_grad():
        initial_logits = model(features)[0]
    initialization_diff = float((initial_logits - baseline_logits).abs().max().cpu())
    require(initialization_diff <= 1e-6, "RCAP initialization equivalence failed")

    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    model.train()
    model.zero_grad(set_to_none=True)
    logits, branches, auxiliary = model(features)
    loss = criterion(logits, labels, train_mask, branches, auxiliary)
    require(bool(torch.isfinite(loss)), "fold0 validation loss is non-finite")
    require(bool(torch.isfinite(logits).all()), "fold0 validation logits are non-finite")
    loss.backward()
    reader_gradients = [
        parameter.grad
        for parameter in model.class_private_low_rank_reader.parameters()
    ]
    require(all(gradient is not None for gradient in reader_gradients), "reader gradient missing")
    require(
        all(bool(torch.isfinite(gradient).all()) for gradient in reader_gradients),
        "reader gradient is non-finite",
    )
    require(common.all_parameter_gradients_finite(model), "model gradient is non-finite")
    cap = cap_snapshot(model)
    summary = {
        "experiment": EXPERIMENT,
        "arm": ARM,
        "scope": scope,
        "fold": 0,
        "seed": SEED,
        "device": str(device),
        "parameter_count": parameter_count(model),
        "initial_raw_logits_max_abs_diff_vs_seps_q": initialization_diff,
        "initialization_threshold": 1e-6,
        "common_initialization_max_abs_diff": common_max_diff,
        "loss": float(loss.detach().cpu()),
        "reader_gradient_tensors": len(reader_gradients),
        "all_reader_gradients_finite": True,
        "all_model_gradients_finite": True,
        "cap_diagnostics": cap,
        "fold_split_hashes": split_hashes,
        "source_hashes": source_hashes(),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    common.write_json(output / "summary.json", summary)
    print(
        f"RCAP VALIDATION PASS init_diff={initialization_diff:.3g} "
        f"max_ratio={cap['maximum_capped_ratio']:.6f}",
        flush=True,
    )
    return summary


def expected_test_subjects(dataset_dict, test_mask) -> set[int]:
    return set(
        np.asarray(dataset_dict["Index"], dtype=np.int64)[
            test_mask.detach().cpu().numpy().astype(bool)
        ].tolist()
    )


def completed_fold(
    final_dir: Path,
    fold: int,
    split_hash: str,
    subjects: set[int],
    expected_sources: dict[str, str],
):
    if not final_dir.exists():
        return None
    required = (
        "summary.json",
        "epoch_metrics.csv",
        "best_predictions.csv",
        "best_confusion_matrix.csv",
        "checkpoint_best.pt",
        "config.ini",
    )
    missing = [name for name in required if not (final_dir / name).is_file()]
    if fold == 0 and not (final_dir / "cap_mechanism.json").is_file():
        missing.append("cap_mechanism.json")
    require(not missing, f"fold{fold}: incomplete final directory: {missing}")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_rows(final_dir / "best_predictions.csv")
    require(summary.get("formal_fold_passed") is True, f"fold{fold}: prior run failed")
    require(summary["arm"] == ARM and int(summary["fold"]) == fold, "fold identity mismatch")
    require(summary["split_hash"] == split_hash, f"fold{fold}: split hash mismatch")
    require(summary["source_hashes"] == expected_sources, f"fold{fold}: source changed")
    epoch_rows = read_rows(final_dir / "epoch_metrics.csv")
    require(len(epoch_rows) == EPOCHS, f"fold{fold}: epoch count mismatch")
    require(
        [int(row["epoch"]) for row in epoch_rows] == list(range(1, EPOCHS + 1)),
        f"fold{fold}: epoch sequence mismatch",
    )
    row_subjects = [int(row["subject_index"]) for row in rows]
    require(
        len(row_subjects) == len(set(row_subjects)) == len(subjects)
        and set(row_subjects) == subjects,
        f"fold{fold}: prediction coverage mismatch",
    )
    require(
        sha256(final_dir / "checkpoint_best.pt") == summary["checkpoint_sha256"],
        f"fold{fold}: checkpoint hash mismatch",
    )
    require(
        seps.normalized_text_hash(final_dir / "config.ini") == seps.EXPECTED_CONFIG_LF_HASH,
        f"fold{fold}: config changed",
    )
    prior.validate_rows(rows, summary["best_metrics"])
    return summary, rows


def run_fold(fold, config, dataset_dict, dataset_data, device, formal_root):
    final_dir = formal_root / f"fold_{fold:02d}"
    staging = formal_root / f".fold_{fold:02d}_in_progress"
    require(not staging.exists(), f"retained staging directory exists: {staging}")
    train_mask, test_mask = dataset_data["Mask"][fold]
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    split_hash = common.query.split_hash(dataset_dict["Index"], train_mask, test_mask)
    sources = source_hashes()
    existing = completed_fold(
        final_dir,
        fold,
        split_hash,
        expected_test_subjects(dataset_dict, test_mask),
        sources,
    )
    if existing is not None:
        print(f"[RCAP/fold{fold}] RESUME completed fold", flush=True)
        return existing

    staging.mkdir(parents=True, exist_ok=False)
    fold_started = time.perf_counter()
    SET_Random(SEED)
    model = build_model(config, dataset_dict, device)
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.Lr_Min
    )
    require(len(optimizer.state) == 0, f"fold{fold}: optimizer is not fresh")
    best = None
    best_state = None
    epoch_rows = []
    training_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"fold{fold}/epoch{epoch}: non-finite loss")
        loss.backward()
        require(common.all_parameter_gradients_finite(model), f"fold{fold}/epoch{epoch}: non-finite gradient")
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_logits = model(features)[0]
            require(bool(torch.isfinite(eval_logits).all()), f"fold{fold}/epoch{epoch}: non-finite logits")
            metrics = prior.metric_bundle(
                eval_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
        selection = (
            metrics["acc"],
            metrics["selection_macro_auc_adjusted"],
            metrics["macro_f1"],
        )
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = common.clone_cpu_state(model)
        epoch_rows.append({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach().cpu()),
            **{name: metrics[name] for name in (*METRICS, "selection_macro_auc_adjusted")},
        })
        if epoch in {1, 100, 200, 300, 400}:
            print(
                f"[RCAP/fold{fold}] epoch={epoch:03d}/400 ACC={metrics['acc']:.4f} "
                f"probability_AUC={metrics['probability_macro_auc']:.4f}",
                flush=True,
            )
    training_elapsed = time.perf_counter() - training_started
    require(best is not None and best_state is not None, f"fold{fold}: best state missing")

    checkpoint = staging / "checkpoint_best.pt"
    torch.save(best_state, checkpoint)
    checkpoint_audit = common.checkpoint_roundtrip(checkpoint, best_state)
    SET_Random(SEED)
    reloaded = build_model(config, dataset_dict, device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    reloaded.eval()
    with torch.no_grad():
        best_logits = reloaded(features)[0]
        best_metrics = prior.metric_bundle(
            best_logits,
            labels,
            test_mask,
            dataset_dict["Label_Weight"],
            float(config.logit_adjust_tau),
        )
        rows = prior.prediction_rows(
            fold, best_logits, labels, test_mask, dataset_dict, config
        )
    require(best_metrics == best["metrics"], f"fold{fold}: checkpoint metrics changed")
    prediction_audit = prior.validate_rows(rows, best_metrics)
    mechanism = cap_snapshot(reloaded) if fold == 0 else None
    require(source_hashes() == sources, f"fold{fold}: source changed during training")
    summary = {
        "experiment": EXPERIMENT,
        "arm": ARM,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "split_hash": split_hash,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "best_epoch": best["epoch"],
        "best_epoch_rule": ["ACC", "historical adjusted-score Macro-AUC", "Macro-F1"],
        "best_metrics": best_metrics,
        "parameter_count": parameter_count(reloaded),
        "checkpoint_sha256": checkpoint_audit["sha256"],
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": prediction_audit,
        "cap_mechanism": mechanism,
        "training_elapsed_seconds": training_elapsed,
        "fold_elapsed_seconds": time.perf_counter() - fold_started,
        "source_hashes": sources,
        "formal_fold_passed": True,
    }
    common.write_json(staging / "summary.json", summary)
    common.write_csv(staging / "epoch_metrics.csv", epoch_rows)
    common.write_csv(staging / "best_predictions.csv", rows)
    common.write_csv(
        staging / "best_confusion_matrix.csv",
        common.confusion_rows(best_metrics["confusion_matrix"]),
    )
    if mechanism is not None:
        common.write_json(staging / "cap_mechanism.json", mechanism)
    shutil.copyfile(ROOT / seps.CONFIG_REL, staging / "config.ini")
    staging.rename(final_dir)
    print(
        f"[RCAP/fold{fold}] PASS best_epoch={best['epoch']} "
        f"ACC={best_metrics['acc']:.4f}",
        flush=True,
    )
    del model, reloaded, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def aggregate(fold_summaries, oof_rows, formal_root, suite_wall_seconds):
    require(len(oof_rows) == 598, "OOF row count mismatch")
    subject_ids = [int(row["subject_index"]) for row in oof_rows]
    require(len(set(subject_ids)) == 598, "OOF subject duplication")
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    oof_metrics = prior.metrics_from_rows(oof_rows)
    prior.validate_rows(oof_rows, oof_metrics)
    fold_rows = [
        {
            "fold": summary["fold"],
            "best_epoch": summary["best_epoch"],
            **{name: summary["best_metrics"][name] for name in (*METRICS, "selection_macro_auc_adjusted")},
            "training_elapsed_seconds": summary["training_elapsed_seconds"],
            "fold_elapsed_seconds": summary["fold_elapsed_seconds"],
        }
        for summary in fold_summaries
    ]
    metric_summary = []
    for name in METRICS:
        values = np.asarray([row[name] for row in fold_rows], dtype=np.float64)
        metric_summary.append({
            "metric": name,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "pooled_oof": float(oof_metrics[name]),
        })
    result = {
        "experiment": EXPERIMENT,
        "arm": ARM,
        "configuration": {
            "low_rank_reader": True,
            "class_graph": False,
            "reader_rank": READER_RANK,
            "reader_norm_cap": READER_NORM_CAP,
            "old_graph_use_graph": False,
            "old_adj_mode": "none",
        },
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs_per_fold": EPOCHS,
        "parameter_count": PARAMETERS,
        "fold_metrics": fold_rows,
        "metric_summary": metric_summary,
        "oof_metrics": oof_metrics,
        "fold0_cap_mechanism": fold_summaries[0]["cap_mechanism"],
        "total_training_seconds": float(sum(row["training_elapsed_seconds"] for row in fold_rows)),
        "total_fold_seconds": float(sum(row["fold_elapsed_seconds"] for row in fold_rows)),
        "suite_wall_seconds": suite_wall_seconds,
        "source_hashes": source_hashes(),
        "all_folds_passed": True,
    }
    common.write_csv(formal_root / "fold_metrics.csv", fold_rows)
    common.write_csv(formal_root / "metrics_summary.csv", metric_summary)
    common.write_csv(formal_root / "oof_predictions.csv", oof_rows)
    result["oof_sha256"] = sha256(formal_root / "oof_predictions.csv")
    common.write_json(formal_root / "oof_metrics.json", oof_metrics)
    common.write_csv(
        formal_root / "oof_confusion_matrix.csv",
        common.confusion_rows(oof_metrics["confusion_matrix"]),
    )
    common.write_json(formal_root / "aggregate_summary.json", result)
    return result


def run_formal() -> dict:
    validate_scope()
    validation_path = ROOT / OUTPUT_ROOT / "validation/summary.json"
    require(validation_path.is_file(), "minimal validation must pass first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    require(validation.get("passed") is True, "minimal validation failed")
    require(validation.get("source_hashes") == source_hashes(), "validation source changed")
    config, dataset_dict, dataset_data, device, _ = prior.load_context()
    formal_root = ROOT / OUTPUT_ROOT / "tenfold_seed0"
    formal_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = formal_root / "aggregate_summary.json"
    if aggregate_path.exists():
        result = json.loads(aggregate_path.read_text(encoding="utf-8"))
        require(result.get("all_folds_passed") is True, "existing aggregate failed")
        require(result.get("source_hashes") == source_hashes(), "aggregate source changed")
        return result
    suite_started = time.perf_counter()
    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        summary, rows = run_fold(
            fold, config, dataset_dict, dataset_data, device, formal_root
        )
        fold_summaries.append(summary)
        oof_rows.extend(rows)
    result = aggregate(
        fold_summaries,
        oof_rows,
        formal_root,
        time.perf_counter() - suite_started,
    )
    common.write_json(formal_root / "run_manifest.json", {
        "command": f"{sys.executable} scripts/run_norm_capped_reader_v1.py formal",
        "branch": BRANCH,
        "execution_source_commit_sha": SOURCE_HEAD,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "source_hashes": source_hashes(),
    })
    print(
        f"RCAP 10-FOLD PASS OOF_ACC={result['oof_metrics']['acc']:.4f} "
        f"OOF_AUC={result['oof_metrics']['probability_macro_auc']:.4f}",
        flush=True,
    )
    return result


def delta(left: dict, right: dict) -> dict:
    return {name: float(left[name] - right[name]) for name in METRICS}


def render_report(final: dict) -> str:
    aggregate_result = final["aggregate"]
    metrics = aggregate_result["oof_metrics"]
    lines = [
        "# Norm-Capped Reader v1 - 10-Fold Final Report",
        "",
        f"Decision: **{final['decision']}**",
        "",
        "## Locked configuration",
        "",
        "- Arm: RCAP (`low_rank_reader=True`, `class_graph=False`)",
        "- Reader rank: 8; per-subject, per-class residual/shared norm cap: 0.5",
        "- TADPOLE AD_CN_SMCI; folds 0..9; seed 0; 400 epochs; full-batch transductive; single model",
        "- Original loss, Adam, CustomCosineAnnealingLR(T_max=400), and historical best-epoch ordering retained",
        f"- Execution source commit: `{SOURCE_HEAD}`",
        "",
        "## Pooled OOF result",
        "",
        f"- Parameters: {PARAMETERS}",
        f"- Correct: {final['correct']} / 598",
        f"- ACC: {metrics['acc']:.7f}",
        f"- Macro-F1: {metrics['macro_f1']:.7f}",
        f"- BACC: {metrics['bacc']:.7f}",
        f"- Probability Macro-AUC: {metrics['probability_macro_auc']:.7f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.7f}",
        f"- Training time: {aggregate_result['total_training_seconds']:.2f} s",
        f"- Total active runtime: {final['total_active_seconds']:.2f} s",
        "- Confusion matrix (rows truth, columns prediction; AD/CN/SMCI):",
        "",
        "```text",
        *[str(row) for row in metrics["confusion_matrix"]],
        "```",
        "",
        "## Fold results",
        "",
        "| Fold | Best epoch | ACC | Probability Macro-AUC |",
        "|---:|---:|---:|---:|",
    ]
    for row in aggregate_result["fold_metrics"]:
        lines.append(
            f"| {row['fold']} | {row['best_epoch']} | {row['acc']:.6f} | "
            f"{row['probability_macro_auc']:.6f} |"
        )
    lines.extend(["", "Mean +/- sample SD:", "", "| Metric | Mean +/- SD | Pooled |", "|---|---:|---:|"])
    for row in aggregate_result["metric_summary"]:
        lines.append(
            f"| {row['metric']} | {row['mean']:.6f} +/- {row['sample_std']:.6f} | "
            f"{row['pooled_oof']:.6f} |"
        )
    lines.extend([
        "",
        "## Comparisons",
        "",
        "| Reference | Delta ACC | Delta Macro-F1 | Delta BACC | Delta probability AUC |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("Original Query", "R", "SEPS-Q"):
        item = final["deltas"][name]
        lines.append(
            f"| {name} | {item['acc']:+.6f} | {item['macro_f1']:+.6f} | "
            f"{item['bacc']:+.6f} | {item['probability_macro_auc']:+.6f} |"
        )
    lines.extend([
        "",
        "Exact McNemar:",
        "",
        "| Comparison | RCAP only correct | Other only correct | Discordant | p-value |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, comparison in final["paired_comparisons"].items():
        lines.append(
            f"| {name} | {comparison['model_only_correct']} | "
            f"{comparison['other_only_correct']} | {comparison['discordant_subjects']} | "
            f"{comparison['exact_mcnemar_two_sided_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Fold0 cap mechanism",
        "",
        "```json",
        json.dumps(aggregate_result["fold0_cap_mechanism"], ensure_ascii=False, indent=2),
        "```",
        "",
        "No other cap value, rank/k search, R/G/RG rerun, multi-seed, ensemble, graph, TabPFN, or additional ablation was run.",
        "",
    ])
    return "\n".join(lines)


def summarize() -> dict:
    summarize_started = time.perf_counter()
    scope = validate_scope()
    formal_root = ROOT / OUTPUT_ROOT / "tenfold_seed0"
    aggregate_path = formal_root / "aggregate_summary.json"
    require(aggregate_path.is_file(), "formal aggregate is missing")
    aggregate_result = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = read_rows(formal_root / "oof_predictions.csv")
    recomputed = prior.metrics_from_rows(rows)
    for name in (*METRICS, "selection_macro_auc_adjusted"):
        require(
            abs(recomputed[name] - aggregate_result["oof_metrics"][name]) <= 1e-10,
            f"OOF metric mismatch: {name}",
        )
    correct = sum(int(row["prediction"]) == int(row["truth"]) for row in rows)
    require(correct == round(recomputed["acc"] * 598), "correct count mismatch")

    r_rows = read_rows(ROOT / R_OOF)
    original_path = prior.locate_original_oof(None)
    original_rows = read_rows(original_path)
    paired = {}
    paired_rows = {}
    for reference_name, reference_rows in (
        ("RCAP vs Original Query", original_rows),
        ("RCAP vs R", r_rows),
    ):
        summary, detail = prior.paired_comparison(
            ARM, rows, reference_name.removeprefix("RCAP vs "), reference_rows
        )
        paired[reference_name] = summary
        paired_rows[reference_name] = detail

    if correct >= 557:
        decision = "SUCCESS: RCAP exceeds Original Query accuracy."
        stop_route = False
    elif correct == 556 and (
        recomputed["macro_f1"] > ORIGINAL_REFERENCE["macro_f1"]
        and recomputed["bacc"] > ORIGINAL_REFERENCE["bacc"]
    ):
        decision = "Accuracy ties Original and balance improves; RCAP is a parameter-efficient candidate."
        stop_route = False
    else:
        decision = "RCAP did not reach the target; stop the SEPS/Reader/Graph improvement route and retain Original Query."
        stop_route = True

    validation = json.loads(
        (ROOT / OUTPUT_ROOT / "validation/summary.json").read_text(encoding="utf-8")
    )
    final_root = ROOT / OUTPUT_ROOT / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    deltas = {
        "Original Query": delta(recomputed, ORIGINAL_REFERENCE),
        "R": delta(recomputed, R_REFERENCE),
        "SEPS-Q": delta(recomputed, SEPS_REFERENCE),
    }
    total_active = (
        float(validation["elapsed_seconds"])
        + float(aggregate_result["total_fold_seconds"])
        + (time.perf_counter() - summarize_started)
    )
    final = {
        "experiment": EXPERIMENT,
        "scope": scope,
        "configuration": aggregate_result["configuration"],
        "correct": correct,
        "aggregate": aggregate_result,
        "deltas": deltas,
        "paired_comparisons": paired,
        "decision": decision,
        "terminate_research_route": stop_route,
        "total_active_seconds": total_active,
        "original_oof_path": str(original_path),
        "original_oof_sha256": sha256(original_path),
        "r_oof_sha256": sha256(ROOT / R_OOF),
        "all_recomputations_passed": True,
    }
    common.write_json(final_root / "final_summary.json", final)
    common.write_json(final_root / "configuration.json", {
        **aggregate_result["configuration"],
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(FOLDS),
        "seed": SEED,
        "epochs": EPOCHS,
        "transductive_full_batch": True,
        "single_model": True,
        "ensemble": False,
        "loss": "historical unchanged",
        "optimizer": "Adam",
        "scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "best_epoch_rule": ["ACC", "historical adjusted-score Macro-AUC", "Macro-F1"],
        "execution_source_commit_sha": SOURCE_HEAD,
        "run_command": f"{sys.executable} scripts/run_norm_capped_reader_v1.py formal",
    })
    for name, comparison in paired.items():
        slug = name.lower().replace(" ", "_")
        common.write_json(final_root / f"{slug}.json", comparison)
        common.write_csv(final_root / f"{slug}.csv", paired_rows[name])
    report = ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(final), encoding="utf-8")
    print(
        f"RCAP SUMMARY PASS correct={correct} ACC={recomputed['acc']:.4f} "
        f"stop_route={stop_route}",
        flush=True,
    )
    return final


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("command", choices=("validate", "formal", "summarize"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "validate":
        run_validation()
    elif args.command == "formal":
        run_formal()
    else:
        summarize()


if __name__ == "__main__":
    main()
