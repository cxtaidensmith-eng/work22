from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from Model import HeterGraph_Model_Kmeans
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path
import run_osfq_id_v1 as common


MODEL_NAME = "Shared-Evidence Private-Semantic Query Pool v1 (SEPS-Q v1)"
VARIANT = "component_shared"
CONFIG_REL = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
OUTPUT_ROOT = Path("experiments/query_pool_component_sharing_v1")
FOLD = 0
SEED = 0
CLASS_ORDER = ("AD", "CN", "SMCI")
EXPECTED_SPLIT_HASH = "1adda8298733c24259acfa957378b356f52db9545ae9c40ef8889ee97bf3ff04"
EXPECTED_CONFIG_LF_HASH = "5f741494141e478a49c0aa6af13e332a2cb98f15a5f29d90d0c057c7804e0cc2"
EXPECTED_BASELINE_PARAMS = 853_131
EXPECTED_BASELINE_STATE_TENSORS = 160
EXPECTED_BASELINE_HASH = "7d5450724928805a255dccf80c4772e2672cbd80c7a14e2fbd8eb73b68671bdc"
EXPECTED_CANDIDATE_PARAMS = 778_251
EXPECTED_CANDIDATE_NAMED_PARAMS = 143
EXPECTED_CANDIDATE_STATE_TENSORS = 148
EXPECTED_CANDIDATE_HASH = "7f9581985309e2a497ad2b9f0308592ab1fa16d1fd46566c6ab313c1170a066a"
EXPECTED_MAPPED_HASH = EXPECTED_CANDIDATE_HASH
BASELINE = {
    "best_epoch": 111,
    "acc": 0.9166666666666666,
    "macro_f1": 0.8977763678660987,
    "bacc": 0.9153225806451614,
    "macro_auc": 0.9683234899447357,
    "weighted_f1": 0.9167761865917399,
    "confusion_matrix": [[7, 0, 1], [0, 21, 0], [2, 2, 27]],
}
SOURCE_FILES = (
    Path("Model/network.py"),
    Path("Loss/loss_fn.py"),
    Path("Loss/__init__.py"),
    Path("Utils/utils.py"),
    Path("scripts/run_query_pool_component_sharing_v1.py"),
)


def normalized_text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.as_posix(): common.query.file_hash(ROOT / path) for path in SOURCE_FILES}


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def named_parameter_count(model) -> int:
    return sum(1 for _ in model.named_parameters())


def build_model(config, dataset_dict: dict, device: torch.device, variant: str):
    return HeterGraph_Model_Kmeans(
        dataset_dict,
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
        query_pool_variant=variant,
        osfq_anchor_mode="none",
        category_branch_fusion="concat",
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
    ).to(device)


def candidate_source_key(candidate_key: str) -> str:
    prefix = "component_shared_pool."
    if not candidate_key.startswith(prefix):
        return candidate_key
    suffix = candidate_key[len(prefix):]
    if suffix.startswith("queries."):
        query_index = suffix.split(".")[1]
        return f"label_pools.{query_index}.query"
    if suffix.startswith("norm_kv.") or suffix.startswith("attn."):
        return f"label_pools.0.{suffix}"
    if suffix.startswith("norm_out."):
        parts = suffix.split(".")
        branch_index = parts[1]
        remainder = ".".join(parts[2:])
        return f"label_pools.{branch_index}.norm_out.{remainder}"
    if suffix.startswith("ffn."):
        parts = suffix.split(".")
        branch_index = parts[1]
        remainder = ".".join(parts[2:])
        return f"label_pools.{branch_index}.ffn.{remainder}"
    raise KeyError(f"No baseline mapping for candidate tensor: {candidate_key}")


def initialization_audit(baseline, candidate) -> dict:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    mapped_baseline = {}
    mapped_candidate = {}
    mismatch_names = []
    max_abs_diff = 0.0
    for candidate_key, candidate_value in candidate_state.items():
        source_key = candidate_source_key(candidate_key)
        common.require(source_key in baseline_state, f"Missing baseline tensor: {source_key}")
        source_value = baseline_state[source_key]
        common.require(source_value.shape == candidate_value.shape, f"Shape mismatch: {candidate_key}")
        difference = float((source_value - candidate_value).abs().max().cpu())
        max_abs_diff = max(max_abs_diff, difference)
        if difference != 0.0:
            mismatch_names.append(candidate_key)
        mapped_baseline[candidate_key] = source_value
        mapped_candidate[candidate_key] = candidate_value

    common_hash_baseline = common.query.tensor_hash(mapped_baseline)
    common_hash_candidate = common.query.tensor_hash(mapped_candidate)
    result = {
        "state_copy_used": False,
        "baseline_parameter_count": parameter_count(baseline),
        "candidate_parameter_count": parameter_count(candidate),
        "candidate_named_parameter_tensors": named_parameter_count(candidate),
        "baseline_state_tensors": len(baseline_state),
        "candidate_state_tensors": len(candidate_state),
        "baseline_full_state_hash": common.query.tensor_hash(baseline_state),
        "candidate_full_state_hash": common.query.tensor_hash(candidate_state),
        "mapped_tensor_count": len(mapped_candidate),
        "mapped_baseline_hash": common_hash_baseline,
        "mapped_candidate_hash": common_hash_candidate,
        "max_abs_diff": max_abs_diff,
        "mismatch_names": mismatch_names,
    }
    result["passed"] = (
        result["baseline_parameter_count"] == EXPECTED_BASELINE_PARAMS
        and result["candidate_parameter_count"] == EXPECTED_CANDIDATE_PARAMS
        and result["candidate_named_parameter_tensors"] == EXPECTED_CANDIDATE_NAMED_PARAMS
        and result["baseline_state_tensors"] == EXPECTED_BASELINE_STATE_TENSORS
        and result["candidate_state_tensors"] == EXPECTED_CANDIDATE_STATE_TENSORS
        and result["baseline_full_state_hash"] == EXPECTED_BASELINE_HASH
        and result["candidate_full_state_hash"] == EXPECTED_CANDIDATE_HASH
        and common_hash_baseline == EXPECTED_MAPPED_HASH
        and common_hash_candidate == EXPECTED_MAPPED_HASH
        and max_abs_diff == 0.0
        and not mismatch_names
    )
    common.require(result["passed"], f"Initialization audit failed: {result}")
    return result


def load_context():
    common.require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    config_hash = normalized_text_hash(ROOT / CONFIG_REL)
    common.require(config_hash == EXPECTED_CONFIG_LF_HASH, "Historical config content changed")
    config = Config_(str(ROOT), str(ROOT / CONFIG_REL), FOLD)
    config.Device = device
    common.require(int(config.T_max) == 400, "Scheduler T_max must remain 400")
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
    train_mask, test_mask = dataset_data["Mask"][FOLD]
    split_hash = common.query.split_hash(dataset_dict["Index"], train_mask, test_mask)
    common.require(split_hash == EXPECTED_SPLIT_HASH, "fold0 split changed")
    return config, dataset_dict, dataset_data, train_mask, test_mask, device, split_hash


def mean_norms(stacked: torch.Tensor) -> list[float]:
    return stacked.detach().float().norm(dim=-1).mean(dim=0).cpu().tolist()


def representation_diagnostics(model, branches, auxiliary, labels, mask) -> dict:
    attention = model.component_shared_pool.last_attention_outputs
    outputs = model.component_shared_pool.last_outputs
    common.require(attention is not None and outputs is not None, "Component diagnostics missing")
    query_vectors = torch.cat(
        [parameter.detach()[:, 0, :] for parameter in model.component_shared_pool.queries],
        dim=0,
    )
    return {
        "query_tokens": common.pairwise_vectors(query_vectors, CLASS_ORDER),
        "attention_stage": {
            "shape": list(attention.shape),
            "mean_feature_norms": mean_norms(attention),
            "similarity": common.query.branch_similarity(
                [attention[:, index] for index in range(3)], list(CLASS_ORDER)
            ),
        },
        "post_ffn_stage": {
            "shape": list(outputs.shape),
            "mean_feature_norms": mean_norms(outputs),
            "similarity": common.query.branch_similarity(branches, list(CLASS_ORDER)),
        },
        "auxiliary_metrics": common.auxiliary_metrics(auxiliary, labels, mask),
    }


def gradient_diagnostics(model) -> dict:
    queries = []
    for parameter in model.component_shared_pool.queries:
        queries.append(float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else 0.0)
    result = {
        "private_query_grad_norms": queries,
        "shared_norm_kv_grad_norm": common.query.module_gradient_norm(model.component_shared_pool.norm_kv),
        "shared_attention_grad_norm": common.query.module_gradient_norm(model.component_shared_pool.attn),
        "private_norm_out_grad_norms": [
            common.query.module_gradient_norm(module) for module in model.component_shared_pool.norm_out
        ],
        "private_ffn_grad_norms": [
            common.query.module_gradient_norm(module) for module in model.component_shared_pool.ffn
        ],
        "auxiliary_head_grad_norms": [
            common.query.module_gradient_norm(module) for module in model._Auxi_classifier
        ],
    }
    all_values = [
        *result["private_query_grad_norms"],
        result["shared_norm_kv_grad_norm"],
        result["shared_attention_grad_norm"],
        *result["private_norm_out_grad_norms"],
        *result["private_ffn_grad_norms"],
        *result["auxiliary_head_grad_norms"],
    ]
    result["all_finite_nonzero"] = all(math.isfinite(value) and value > 0.0 for value in all_values)
    common.require(result["all_finite_nonzero"], f"Gradient gate failed: {result}")
    return result


def run_phase(phase: str) -> dict:
    epochs = 3 if phase == "smoke" else 400
    final_dir = ROOT / OUTPUT_ROOT / phase
    staging_dir = ROOT / OUTPUT_ROOT / f".{phase}_in_progress"
    common.require(not final_dir.exists(), f"Refusing to overwrite: {final_dir}")
    common.require(not staging_dir.exists(), f"Retained staging directory exists: {staging_dir}")
    if phase == "formal":
        smoke_summary_path = ROOT / OUTPUT_ROOT / "smoke/summary.json"
        common.require(smoke_summary_path.exists(), "Smoke result is required before formal")
        smoke_summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
        common.require(bool(smoke_summary.get("passed")), "Smoke did not pass")
    staging_dir.mkdir(parents=True)

    started = time.perf_counter()
    locked_sources = source_hashes()
    config, dataset_dict, dataset_data, train_mask, test_mask, device, split_hash = load_context()
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]

    init_audit = None
    if phase == "smoke":
        SET_Random(SEED)
        baseline = build_model(config, dataset_dict, device, "independent")
        SET_Random(SEED)
        model = build_model(config, dataset_dict, device, VARIANT)
        init_audit = initialization_audit(baseline, model)
        del baseline
        torch.cuda.empty_cache()
    else:
        SET_Random(SEED)
        model = build_model(config, dataset_dict, device, VARIANT)
        current_hash = common.query.tensor_hash(model.state_dict())
        common.require(current_hash == EXPECTED_CANDIDATE_HASH, "Formal fresh initialization changed")

    criterion = criterion_query_pool_no_orth(dataset_dict, device, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.Lr_Min)
    common.require(len(optimizer.state) == 0, "Fresh optimizer unexpectedly has state")
    device_audit = common.device_audit(model, criterion, dataset_dict, dataset_data, device)

    epoch_rows = []
    milestone_diagnostics = []
    diagnostic_epochs = {1, epochs} if phase == "smoke" else {1, 100, 200, 300, 400}
    best = None
    best_state = None
    first_output_audit = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, branches, auxiliary = model(features)
        if epoch == 1:
            first_output_audit = common.output_audit(
                logits, branches, auxiliary, int(features.size(0)), device
            )
        loss = criterion(logits, labels, train_mask, branches, auxiliary)
        common.require(loss.is_cuda and bool(torch.isfinite(loss)), f"Invalid train loss at epoch {epoch}")
        if epoch == 1:
            components = common.loss_components(criterion, logits, labels, train_mask, auxiliary)
            formula_delta = float((loss - components["total"]).detach().abs().cpu())
            common.require(formula_delta <= 1e-7, "Loss formula differs from Query baseline")
        loss.backward()
        epoch_gradient_diagnostics = None
        if epoch in diagnostic_epochs:
            epoch_gradient_diagnostics = gradient_diagnostics(model)
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_logits, eval_branches, eval_auxiliary = model(features)
            common.require(bool(torch.isfinite(eval_logits).all()), f"Invalid logits at epoch {epoch}")
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

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach().cpu()),
            "test_loss": float(test_loss.detach().cpu()),
            **{name: metrics[name] for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
        }
        epoch_rows.append(row)
        score = (metrics["acc"], metrics["macro_auc"], metrics["macro_f1"])
        if best is None or score > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": score,
                "metrics": deepcopy(metrics),
            }
            best_state = common.clone_cpu_state(model)

        if epoch in diagnostic_epochs:
            milestone_diagnostics.append(
                {
                    "epoch": epoch,
                    "gradient": epoch_gradient_diagnostics,
                    "representation": representation_diagnostics(
                        model, eval_branches, eval_auxiliary, labels, test_mask
                    ),
                }
            )
        if epoch in diagnostic_epochs or epoch % 25 == 0:
            print(
                f"[{phase}] epoch={epoch:03d}/{epochs} loss={float(loss.detach().cpu()):.4f} "
                f"ACC={metrics['acc']:.4f} Macro-AUC={metrics['macro_auc']:.4f}",
                flush=True,
            )

    common.require(best is not None and best_state is not None, "Best checkpoint missing")
    checkpoint_path = staging_dir / "checkpoint_best.pt"
    torch.save(best_state, checkpoint_path)
    checkpoint_audit = common.checkpoint_roundtrip(checkpoint_path, best_state)

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
        best_predictions = common.prediction_rows(
            best_logits, labels, test_mask, dataset_dict, config
        )
        best_representation = representation_diagnostics(
            model, best_branches, best_auxiliary, labels, test_mask
        )
    common.require(best_metrics == best["metrics"], "Best checkpoint metrics changed after reload")
    prediction_audit = common.validate_prediction_metrics(best_predictions, best_metrics)

    end_sources = source_hashes()
    common.require(end_sources == locked_sources, "Source files changed during training")
    deltas = {
        name: best_metrics[name] - BASELINE[name]
        for name in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
    }
    elapsed = time.perf_counter() - started
    summary = {
        "model": MODEL_NAME,
        "variant": VARIANT,
        "phase": phase,
        "passed": True,
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": FOLD,
            "seed": SEED,
            "epochs": epochs,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "single_model": True,
            "ensemble": False,
            "semantic_fusion": "add",
            "category_fusion": "concat",
            "adj_mode": "none",
            "graph_use_graph": False,
            "optimizer": "Adam",
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": 400,
            "best_epoch_rule": ["ACC", "Macro-AUC", "Macro-F1"],
            "loss": "main weighted CE + AD-rest + CN-rest + SMCI-rest",
            "orthogonality_loss": False,
            "split_hash": split_hash,
            "train_size": int(train_mask.sum().item()),
            "test_size": int(test_mask.sum().item()),
        },
        "structure": {
            "private_queries": 3,
            "shared_norm_kv": 1,
            "shared_multihead_attention": 1,
            "private_norm_out": 3,
            "private_ffn": 3,
            "parameter_count": parameter_count(model),
            "named_parameter_tensors": named_parameter_count(model),
            "state_tensors": len(model.state_dict()),
        },
        "initialization_audit": init_audit,
        "device_audit": device_audit,
        "first_forward_audit": first_output_audit,
        "best": {"epoch": best["epoch"], "metrics": best_metrics},
        "historical_original_query_baseline": BASELINE,
        "delta_candidate_minus_baseline": deltas,
        "best_representation_diagnostics": best_representation,
        "milestone_diagnostics": milestone_diagnostics,
        "checkpoint_audit": checkpoint_audit,
        "prediction_audit": prediction_audit,
        "source_hashes": locked_sources,
        "elapsed_seconds": elapsed,
    }

    shutil.copyfile(ROOT / CONFIG_REL, staging_dir / "config.ini")
    common.write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    common.write_csv(staging_dir / "best_predictions.csv", best_predictions)
    common.write_csv(
        staging_dir / "best_confusion_matrix.csv",
        common.confusion_rows(best_metrics["confusion_matrix"]),
    )
    common.write_json(staging_dir / "best_diagnostics.json", {
        "best_epoch": best["epoch"],
        "representation": best_representation,
        "milestones": milestone_diagnostics,
    })
    common.write_json(staging_dir / "summary.json", summary)

    report_lines = [
        f"{MODEL_NAME} - {phase} report",
        "",
        f"fold={FOLD}; seed={SEED}; epochs={epochs}; device={device}",
        "Structure: private Query x3 + shared norm_kv/MHA + private norm_out/FFN x3",
        f"Parameters: {parameter_count(model)} (Original Query: {EXPECTED_BASELINE_PARAMS}; delta={parameter_count(model) - EXPECTED_BASELINE_PARAMS})",
        "Loss: main weighted CE + AD-rest + CN-rest + SMCI-rest; orthogonality=False",
        f"Best epoch: {best['epoch']}",
        f"ACC: {best_metrics['acc']}",
        f"Macro-F1: {best_metrics['macro_f1']}",
        f"BACC: {best_metrics['bacc']}",
        f"Macro-AUC: {best_metrics['macro_auc']}",
        f"Weighted-F1: {best_metrics['weighted_f1']}",
        f"Confusion matrix: {best_metrics['confusion_matrix']}",
        "",
        "Delta candidate - Original Query baseline:",
        *[f"{name}: {value:+.6f}" for name, value in deltas.items()],
        "",
        f"Checkpoint roundtrip: {checkpoint_audit['passed']}",
        f"Prediction reproduction: {prediction_audit['passed']}",
        f"Elapsed seconds: {elapsed:.3f}",
    ]
    (staging_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    staging_dir.rename(final_dir)
    print(
        f"PASS phase={phase} best_epoch={best['epoch']} ACC={best_metrics['acc']:.4f} "
        f"Macro-F1={best_metrics['macro_f1']:.4f} BACC={best_metrics['bacc']:.4f} "
        f"Macro-AUC={best_metrics['macro_auc']:.4f} output={final_dir}",
        flush=True,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase(args.phase)


if __name__ == "__main__":
    main()
