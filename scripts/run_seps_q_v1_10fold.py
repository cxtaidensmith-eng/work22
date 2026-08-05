from __future__ import annotations

import json
import shutil
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
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_osfq_id_v1 as common
import run_query_pool_component_sharing_v1 as seps


OUTPUT_REL = Path("experiments/query_pool_component_sharing_v1/tenfold_seed0")
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
METRIC_NAMES = ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")


def prediction_rows(fold, logits, labels, mask, dataset_dict, config):
    raw = logits[mask]
    adjusted, probabilities, predictions = common.score_tensors(
        raw,
        dataset_dict["Label_Weight"],
        float(config.logit_adjust_tau),
    )
    source_indices = np.asarray(dataset_dict["Index"], dtype=np.int64)[
        mask.detach().cpu().numpy().astype(bool)
    ]
    truth = labels[mask].detach().cpu().numpy().astype(np.int64)
    return common.make_prediction_rows(
        fold,
        source_indices,
        truth,
        raw.detach().cpu().numpy(),
        adjusted.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
        predictions.detach().cpu().numpy().astype(np.int64),
    )


def run_fold(fold, config, dataset_dict, dataset_data, device, output_root):
    final_dir = output_root / f"fold_{fold:02d}"
    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    common.require(not final_dir.exists(), f"Refusing to overwrite {final_dir}")
    common.require(not staging_dir.exists(), f"Retained staging directory exists: {staging_dir}")
    staging_dir.mkdir(parents=True)

    train_mask, test_mask = dataset_data["Mask"][fold]
    split_hash = common.query.split_hash(dataset_dict["Index"], train_mask, test_mask)
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]

    SET_Random(SEED)
    model = seps.build_model(config, dataset_dict, device, seps.VARIANT)
    common.require(seps.parameter_count(model) == seps.EXPECTED_CANDIDATE_PARAMS, "Parameter count changed")
    common.require(
        common.query.tensor_hash(model.state_dict()) == seps.EXPECTED_CANDIDATE_HASH,
        "Fresh initialization changed",
    )
    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.Lr_Min)

    best = None
    best_state = None
    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        common.require(bool(torch.isfinite(loss)), f"fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_logits, eval_branches, eval_auxiliary = model(features)
            common.require(bool(torch.isfinite(eval_logits).all()), f"fold{fold} epoch{epoch}: invalid logits")
            test_loss = criterion(eval_logits, labels, test_mask, eval_branches, eval_auxiliary)
            metrics = common.clean_metrics(
                common.query.metric_bundle(
                    eval_logits,
                    labels,
                    test_mask,
                    dataset_dict["Label_Weight"],
                    float(config.logit_adjust_tau),
                )
            )
        score = (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(metrics),
            }
            best_state = common.clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(loss.detach().cpu()),
                "test_loss": float(test_loss.detach().cpu()),
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
        if epoch in {1, 100, 200, 300, 400}:
            print(
                f"[fold{fold}] epoch={epoch:03d}/400 ACC={metrics['acc']:.4f} "
                f"Macro-AUC={metrics['macro_auc']:.4f}",
                flush=True,
            )

    common.require(best is not None and best_state is not None, f"fold{fold}: best result missing")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_logits, best_branches, best_auxiliary = model(features)
        best_metrics = common.clean_metrics(
            common.query.metric_bundle(
                best_logits,
                labels,
                test_mask,
                dataset_dict["Label_Weight"],
                float(config.logit_adjust_tau),
            )
        )
        rows = prediction_rows(fold, best_logits, labels, test_mask, dataset_dict, config)
    common.require(best_metrics == best["metrics"], f"fold{fold}: best reload mismatch")
    prediction_audit = common.validate_prediction_metrics(rows, best_metrics)

    torch.save(best_state, staging_dir / "checkpoint_best.pt")
    shutil.copyfile(ROOT / seps.CONFIG_REL, staging_dir / "config.ini")
    common.write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    common.write_csv(staging_dir / "best_predictions.csv", rows)
    common.write_csv(
        staging_dir / "best_confusion_matrix.csv",
        common.confusion_rows(best_metrics["confusion_matrix"]),
    )
    summary = {
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "split_hash": split_hash,
        "train_size": int(train_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "best_epoch": best["epoch"],
        "best_metrics": best_metrics,
        "parameter_count": seps.parameter_count(model),
        "prediction_audit": prediction_audit,
        "formal_fold_passed": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.write_json(staging_dir / "summary.json", summary)
    staging_dir.rename(final_dir)
    print(
        f"[fold{fold}] PASS best_epoch={best['epoch']} ACC={best_metrics['acc']:.4f} "
        f"Macro-F1={best_metrics['macro_f1']:.4f} Macro-AUC={best_metrics['macro_auc']:.4f}",
        flush=True,
    )
    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary, rows


def main():
    common.require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    device = torch.device("cuda:0")
    common.require(
        seps.normalized_text_hash(ROOT / seps.CONFIG_REL) == seps.EXPECTED_CONFIG_LF_HASH,
        "Historical config changed",
    )
    output_root = ROOT / OUTPUT_REL
    common.require(not output_root.exists(), f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    config = Config_(str(ROOT), str(ROOT / seps.CONFIG_REL), 0)
    config.Device = device
    common.require(int(config.T_max) == 400, "T_max changed")
    SET_Random(SEED)
    feature_path, dict_path, _, class_names = load_path(str(ROOT), config.DATA_SET, config.Task)
    dataset_dict, dataset_data = load_dataset(
        feature_path,
        dict_path,
        device,
        class_names,
        config.Shuffle,
        SEED,
        train_size=config.train_size,
    )
    locked_sources = {
        "Model/network.py": common.query.file_hash(ROOT / "Model/network.py"),
        "Loss/loss_fn.py": common.query.file_hash(ROOT / "Loss/loss_fn.py"),
        "single_runner": common.query.file_hash(ROOT / "scripts/run_query_pool_component_sharing_v1.py"),
        "tenfold_runner": common.query.file_hash(Path(__file__)),
    }

    suite_started = time.perf_counter()
    fold_summaries = []
    oof_rows = []
    for fold in FOLDS:
        summary, rows = run_fold(
            fold, config, dataset_dict, dataset_data, device, output_root
        )
        fold_summaries.append(summary)
        oof_rows.extend(rows)

    common.require(len(oof_rows) == len(dataset_dict["Index"]), "OOF row count mismatch")
    common.require(len({int(row["subject_index"]) for row in oof_rows}) == len(oof_rows), "OOF subjects not unique")
    oof_rows.sort(key=lambda row: int(row["subject_index"]))
    oof_metrics = common.metrics_from_prediction_rows(oof_rows)

    fold_metric_rows = []
    for summary in fold_summaries:
        fold_metric_rows.append(
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                **{name: summary["best_metrics"][name] for name in METRIC_NAMES},
                "params": summary["parameter_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    metric_summary_rows = []
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in fold_metric_rows], dtype=np.float64)
        metric_summary_rows.append(
            {
                "metric": name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "oof": float(oof_metrics[name]),
            }
        )

    current_sources = {
        "Model/network.py": common.query.file_hash(ROOT / "Model/network.py"),
        "Loss/loss_fn.py": common.query.file_hash(ROOT / "Loss/loss_fn.py"),
        "single_runner": common.query.file_hash(ROOT / "scripts/run_query_pool_component_sharing_v1.py"),
        "tenfold_runner": common.query.file_hash(Path(__file__)),
    }
    common.require(current_sources == locked_sources, "Source changed during 10-fold run")

    common.write_csv(output_root / "fold_metrics.csv", fold_metric_rows)
    common.write_csv(output_root / "metrics_summary.csv", metric_summary_rows)
    common.write_csv(output_root / "oof_predictions.csv", oof_rows)
    common.write_json(output_root / "oof_metrics.json", oof_metrics)
    common.write_csv(
        output_root / "aggregate_confusion_matrix.csv",
        common.confusion_rows(oof_metrics["confusion_matrix"]),
    )
    aggregate = {
        "model": seps.MODEL_NAME,
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "parameter_count": seps.EXPECTED_CANDIDATE_PARAMS,
        "fold_metrics": fold_metric_rows,
        "metric_summary": metric_summary_rows,
        "oof_metrics": oof_metrics,
        "source_hashes": locked_sources,
        "all_folds_passed": True,
        "elapsed_seconds": time.perf_counter() - suite_started,
    }
    common.write_json(output_root / "aggregate_summary.json", aggregate)

    report = [
        f"{seps.MODEL_NAME} - 10-Fold Single-Seed Report",
        "",
        "dataset=TADPOLE; task=AD_CN_SMCI; folds=0..9; seed=0; epochs=400",
        f"parameter_count={seps.EXPECTED_CANDIDATE_PARAMS}",
        "",
        "Pooled OOF metrics:",
        *[f"{name}={oof_metrics[name]}" for name in METRIC_NAMES],
        f"confusion_matrix={oof_metrics['confusion_matrix']}",
        "",
        "Fold mean ± sample std:",
        *[
            f"{row['metric']}={row['mean']} ± {row['std']}"
            for row in metric_summary_rows
        ],
        "",
        f"elapsed_seconds={aggregate['elapsed_seconds']}",
        "all_folds_passed=True",
    ]
    (output_root / "paper_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"10-FOLD PASS OOF_ACC={oof_metrics['acc']:.4f} "
        f"OOF_Macro-F1={oof_metrics['macro_f1']:.4f} "
        f"OOF_BACC={oof_metrics['bacc']:.4f} OOF_AUC={oof_metrics['macro_auc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
