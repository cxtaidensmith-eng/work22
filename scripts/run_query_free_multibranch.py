from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_query_free_multibranch, criterion_query_pool_no_orth
from Model import HeterGraph_Model_Kmeans
from Utils import CustomCosineAnnealingLR, Config_, SET_Random, load_dataset, load_path


FOLD = 0
SEED = 0
VARIANT = "query_free_multibranch"
BASELINE_VARIANT = "baseline_q_noorth"
SEMANTIC_FUSION = "add"
BRANCH_PREFIXES = (
    "label_pools.",
    "_Auxi_classifier.",
    "latent_branches.",
    "latent_aux_classifiers.",
)
COMMON_COMPONENTS = {
    "feature_modal_gain_bias": ("Feature_Modal.",),
    "modal_token_encoder": ("modal_token_encoder.",),
    "modal_gate": ("modal_gate_logit",),
    "shared_transformer": ("shared_transformer.",),
    "global_branch": ("Global_Message.",),
    "adjacency_module": ("Adj_Learning.",),
    "message_mlp": ("Message_MLP.",),
    "difformer_and_classifier": ("GCN.",),
}


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_hash(indices, train_mask: torch.Tensor, test_mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    digest.update(train_mask.detach().cpu().numpy().astype(np.uint8).tobytes())
    digest.update(test_mask.detach().cpu().numpy().astype(np.uint8).tobytes())
    return digest.hexdigest()


def git_info() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()

    try:
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": "unknown", "commit": "unknown", "dirty": None}


def build_model(config: Config_, dataset_dict: dict, variant: str, device: torch.device):
    if variant not in {"original", VARIANT}:
        raise ValueError(f"Unsupported variant: {variant}")
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
        semantic_fusion=SEMANTIC_FUSION,
        category_branch_variant=variant,
        adj_mode="none",
        label_graph_alpha=0.0,
        label_graph_topk=0,
        label_graph_reg_lambda=0.0,
    ).to(device)


def is_branch_state(key: str) -> bool:
    return key.startswith(BRANCH_PREFIXES)


@torch.no_grad()
def copy_and_audit_common_initialization(baseline, variant) -> dict:
    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    common_keys = sorted(
        key
        for key in set(baseline_state) & set(variant_state)
        if not is_branch_state(key) and baseline_state[key].shape == variant_state[key].shape
    )
    for key in common_keys:
        variant_state[key].copy_(baseline_state[key])

    baseline_common = {key: baseline_state[key] for key in common_keys}
    variant_common = {key: variant_state[key] for key in common_keys}
    differences = {
        key: float((baseline_common[key] - variant_common[key]).abs().max())
        for key in common_keys
    }

    component_hashes = {}
    for component, prefixes in COMMON_COMPONENTS.items():
        keys = [
            key for key in common_keys
            if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
        ]
        component_hashes[component] = {
            "tensor_count": len(keys),
            "parameter_and_buffer_count": int(sum(baseline_state[key].numel() for key in keys)),
            "baseline_hash": tensor_hash({key: baseline_state[key] for key in keys}),
            "query_free_hash": tensor_hash({key: variant_state[key] for key in keys}),
        }

    return {
        "common_tensor_count": len(common_keys),
        "common_parameter_and_buffer_count": int(
            sum(baseline_state[key].numel() for key in common_keys)
        ),
        "baseline_common_hash": tensor_hash(baseline_common),
        "query_free_common_hash": tensor_hash(variant_common),
        "max_abs_diff": max(differences.values()) if differences else 0.0,
        "mismatched_tensors": [key for key, value in differences.items() if value != 0.0],
        "component_hashes": component_hashes,
    }


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def branch_structure_audit(baseline, variant) -> dict:
    branch_details = {}
    pool_parameter_ids = []
    auxiliary_parameter_ids = []
    for name, branch in variant.latent_branches.items():
        pool_parameter_ids.extend(id(parameter) for parameter in branch.parameters())
        auxiliary_parameter_ids.extend(
            id(parameter) for parameter in variant.latent_aux_classifiers[name].parameters()
        )
        branch_details[name] = {
            "pool_parameter_count": count_parameters(branch),
            "auxiliary_head_parameter_count": count_parameters(
                variant.latent_aux_classifiers[name]
            ),
            "auxiliary_output_classes": variant.latent_aux_classifiers[name].out_features,
            "contains_multihead_attention": any(
                isinstance(module, torch.nn.MultiheadAttention) for module in branch.modules()
            ),
            "parameter_names": [name for name, _ in branch.named_parameters()],
        }
    return {
        "baseline_parameter_count": count_parameters(baseline),
        "query_free_parameter_count": count_parameters(variant),
        "parameter_delta": count_parameters(variant) - count_parameters(baseline),
        "baseline_query_pool_count": len(baseline.label_pools),
        "query_free_has_label_pools": hasattr(variant, "label_pools"),
        "latent_branch_count": len(variant.latent_branches),
        "latent_branch_parameters_independent": (
            len(pool_parameter_ids) == len(set(pool_parameter_ids))
        ),
        "auxiliary_head_parameters_independent": (
            len(auxiliary_parameter_ids) == len(set(auxiliary_parameter_ids))
        ),
        "semantic_fusion": variant.semantic_fusion,
        "semantic_branch": variant.semantic_branch,
        "adj_mode": variant.adj_mode,
        "difformer_class": type(variant.GCN).__name__,
        "difformer_graph_use_flags": [
            bool(layer.use_graph) for layer in getattr(variant.GCN, "layers", [])
        ],
        "branches": branch_details,
    }


def baseline_structure_audit(baseline) -> dict:
    pool_parameter_ids = [
        id(parameter)
        for pool in baseline.label_pools
        for parameter in pool.parameters()
    ]
    auxiliary_parameter_ids = [
        id(parameter)
        for head in baseline._Auxi_classifier
        for parameter in head.parameters()
    ]
    pools = {}
    for branch_idx, (pool, auxiliary_head) in enumerate(
        zip(baseline.label_pools, baseline._Auxi_classifier), start=1
    ):
        pools[f"query_pool_{branch_idx}"] = {
            "pool_parameter_count": count_parameters(pool),
            "auxiliary_head_parameter_count": count_parameters(auxiliary_head),
            "auxiliary_output_classes": auxiliary_head.out_features,
            "contains_multihead_attention": any(
                isinstance(module, torch.nn.MultiheadAttention) for module in pool.modules()
            ),
            "contains_learnable_query": hasattr(pool, "query") and pool.query.requires_grad,
        }
    return {
        "variant": BASELINE_VARIANT,
        "parameter_count": count_parameters(baseline),
        "query_pool_count": len(baseline.label_pools),
        "query_pool_parameters_independent": (
            len(pool_parameter_ids) == len(set(pool_parameter_ids))
        ),
        "auxiliary_head_parameters_independent": (
            len(auxiliary_parameter_ids) == len(set(auxiliary_parameter_ids))
        ),
        "semantic_fusion": baseline.semantic_fusion,
        "semantic_branch": baseline.semantic_branch,
        "adj_mode": baseline.adj_mode,
        "difformer_class": type(baseline.GCN).__name__,
        "difformer_graph_use_flags": [
            bool(layer.use_graph) for layer in getattr(baseline.GCN, "layers", [])
        ],
        "orthogonality_loss": False,
        "auxiliary_supervision": "original three independent one-vs-rest binary heads",
        "query_pools": pools,
    }


def clean_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key not in {"predictions", "truth"}}


def metric_bundle(logits, labels, mask, label_weight, tau):
    adjusted = logits - tau * label_weight.to(logits).clamp_min(1e-8).log().view(1, -1)
    values = adjusted[mask]
    true = labels[mask].detach().cpu().numpy()
    pred = values.argmax(dim=-1).detach().cpu().numpy()
    onehot = F.one_hot(labels[mask].detach().cpu(), num_classes=values.shape[-1]).numpy()
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=np.arange(values.shape[-1]),
        zero_division=0,
    )
    try:
        macro_auc = float(roc_auc_score(onehot, values.detach().cpu().numpy()))
    except ValueError:
        macro_auc = float("nan")
    return {
        "acc": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "bacc": float(balanced_accuracy_score(true, pred)),
        "macro_auc": macro_auc,
        "weighted_f1": float(f1_score(true, pred, average="weighted")),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "class_f1": class_f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(
            true, pred, labels=np.arange(values.shape[-1])
        ).tolist(),
        "predictions": pred.tolist(),
        "truth": true.tolist(),
    }


def branch_similarity(branch_outputs: list[torch.Tensor], branch_names: list[str]) -> dict:
    stacked = torch.stack([F.normalize(output, dim=-1) for output in branch_outputs], dim=0)
    matrix = torch.einsum("knd,lnd->kln", stacked, stacked).mean(dim=-1)
    pairs = {}
    pair_values = []
    for left in range(len(branch_outputs)):
        for right in range(left + 1, len(branch_outputs)):
            per_subject = (stacked[left] * stacked[right]).sum(dim=-1)
            key = f"{branch_names[left]}_vs_{branch_names[right]}"
            pairs[key] = {
                "mean": float(per_subject.mean()),
                "std": float(per_subject.std(unbiased=False)),
            }
            pair_values.append(per_subject)
    off_diagonal = torch.cat(pair_values)
    return {
        "matrix": matrix.detach().cpu().tolist(),
        "pairs": pairs,
        "off_diagonal_mean": float(off_diagonal.mean()),
        "off_diagonal_std": float(off_diagonal.std(unbiased=False)),
    }


def module_gradient_norm(module: torch.nn.Module) -> float:
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    return float(torch.sqrt(squared)) if squared is not None else 0.0


def train_smoke(
    model,
    criterion,
    run_name: str,
    branch_modules: dict[str, torch.nn.Module],
    config,
    dataset_dict,
    dataset_data,
    epochs: int,
    output_dir: Path,
) -> dict:
    branch_names = list(branch_modules)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CustomCosineAnnealingLR(
        optimizer,
        T_max=config.T_max,
        eta_min=config.Lr_Min,
    )
    features = dataset_data["Feature"]
    labels = dataset_data["Label"]
    train_mask, test_mask = dataset_data["Mask"][FOLD]
    tau = float(config.logit_adjust_tau)
    rows = []
    best = None
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, branch_outputs, auxiliary_outputs = model(features)
        loss = criterion(output, labels, train_mask, branch_outputs, auxiliary_outputs)
        loss.backward()
        gradient_norms = {
            name: module_gradient_norm(branch)
            for name, branch in branch_modules.items()
        }
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_output, eval_branches, eval_auxiliary = model(features)
            test_loss = criterion(
                eval_output, labels, test_mask, eval_branches, eval_auxiliary
            )
            train_metrics = metric_bundle(
                eval_output, labels, train_mask, dataset_dict["Label_Weight"], tau
            )
            test_metrics = metric_bundle(
                eval_output, labels, test_mask, dataset_dict["Label_Weight"], tau
            )
            similarity = branch_similarity(eval_branches, branch_names)

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach()),
            "test_loss": float(test_loss),
            **{
                f"train_{key}": value
                for key, value in clean_metrics(train_metrics).items()
                if isinstance(value, (int, float))
            },
            **{
                f"test_{key}": value
                for key, value in clean_metrics(test_metrics).items()
                if isinstance(value, (int, float))
            },
            **{f"{name}_grad_norm": value for name, value in gradient_norms.items()},
            "branch_similarity_off_diagonal_mean": similarity["off_diagonal_mean"],
            "branch_similarity_off_diagonal_std": similarity["off_diagonal_std"],
        }
        for pair, values in similarity["pairs"].items():
            row[f"similarity_{pair}"] = values["mean"]
        rows.append(row)

        score = (
            test_metrics["acc"],
            test_metrics["macro_auc"],
            test_metrics["macro_f1"],
        )
        if best is None or score > best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "metrics": deepcopy(test_metrics),
                "similarity": deepcopy(similarity),
                "gradient_norms": deepcopy(gradient_norms),
            }
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"[{run_name}] epoch={epoch:03d}/{epochs} "
                f"loss={float(loss.detach()):.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f} "
                f"branch_sim={similarity['off_diagonal_mean']:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_path = log_dir / f"{run_name}_epochs.csv"
    with curve_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_state, checkpoint_dir / f"{run_name}_best.pt")
    torch.save(
        model.state_dict(),
        checkpoint_dir / f"{run_name}_epoch{epochs}.pt",
    )

    return {
        "variant": run_name,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_metrics": clean_metrics(best["metrics"]),
        "best_branch_similarity": best["similarity"],
        "best_branch_gradient_norms": best["gradient_norms"],
        "final_metrics": clean_metrics(test_metrics),
        "final_branch_similarity": similarity,
        "final_branch_gradient_norms": gradient_norms,
        "elapsed_seconds": elapsed,
        "curve_csv": str(curve_path.relative_to(ROOT)).replace("\\", "/"),
    }


def render_txt_report(payload: dict) -> str:
    protocol = payload["protocol"]
    structure = payload["structure_audit"]
    initialization = payload["initialization_audit"]
    smoke = payload["smoke_test"]
    best = smoke["best_metrics"]
    final = smoke["final_metrics"]
    lines = [
        "Query-free Multi-Branch v1 结构审计与 Smoke Test 报告",
        "",
        "状态：仅完成 3 epoch smoke test；未运行 400 epoch 正式实验。",
        "",
        "一、实验协议锁定",
        f"dataset={protocol['dataset']}",
        f"task={protocol['task']}",
        f"fold={protocol['fold']}",
        f"seed={protocol['seed']}",
        f"epochs={protocol['epochs']}",
        f"transductive_full_batch={protocol['transductive_full_batch']}",
        f"semantic_fusion={protocol['semantic_fusion']}",
        f"adj_mode={protocol['adj_mode']}",
        f"DIFFormer graph use={protocol['graph_use_graph']}",
        "ensemble=False",
        "orthogonality_loss=False",
        "auxiliary supervision=3 independent full three-class heads",
        "",
        "二、模型结构",
        "三个独立 latent branches：Linear(96,96) -> mean over 6 modalities -> "
        "FFN(96,192,96) -> LayerNorm(value + FFN(value))。",
        f"latent branch count={structure['latent_branch_count']}",
        f"latent branch parameters independent={structure['latent_branch_parameters_independent']}",
        f"auxiliary head parameters independent={structure['auxiliary_head_parameters_independent']}",
        f"query-free contains legacy label pools={structure['query_free_has_label_pools']}",
        f"DIFFormer class={structure['difformer_class']}",
        f"DIFFormer use_graph flags={structure['difformer_graph_use_flags']}",
        "",
        "三、参数量",
        f"baseline={structure['baseline_parameter_count']}",
        f"query-free={structure['query_free_parameter_count']}",
        f"delta={structure['parameter_delta']}",
    ]
    for name, details in structure["branches"].items():
        lines.append(
            f"{name}: pool={details['pool_parameter_count']}, "
            f"aux_head={details['auxiliary_head_parameter_count']}, "
            f"aux_classes={details['auxiliary_output_classes']}, "
            f"contains_attention={details['contains_multihead_attention']}"
        )
    lines.extend(
        [
            "",
            "四、公共模块初始化公平性",
            f"common tensor count={initialization['common_tensor_count']}",
            f"common parameter/buffer count={initialization['common_parameter_and_buffer_count']}",
            f"baseline hash={initialization['baseline_common_hash']}",
            f"query-free hash={initialization['query_free_common_hash']}",
            f"max_abs_diff={initialization['max_abs_diff']}",
            f"mismatched tensors={initialization['mismatched_tensors']}",
            "",
            "五、3 epoch smoke test",
            f"best epoch={smoke['best_epoch']}",
            f"best ACC={best['acc']}",
            f"best Macro-F1={best['macro_f1']}",
            f"best BACC={best['bacc']}",
            f"best Macro-AUC={best['macro_auc']}",
            f"best confusion matrix={best['confusion_matrix']}",
            f"final ACC={final['acc']}",
            f"final Macro-F1={final['macro_f1']}",
            f"final BACC={final['bacc']}",
            f"final Macro-AUC={final['macro_auc']}",
            f"best branch gradient norms={smoke['best_branch_gradient_norms']}",
            f"best branch similarity matrix={smoke['best_branch_similarity']['matrix']}",
            f"elapsed seconds={smoke['elapsed_seconds']}",
            "",
            "六、结论",
            "结构 smoke test 用于确认前向、反向、辅助损失、诊断和保存链路可运行。",
            "3 epoch 指标不用于判断多分支假设，也不能与 400 epoch baseline 作性能结论。",
            "等待人工确认后才能运行 400 epoch 正式实验。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paired_txt_report(payload: dict) -> str:
    protocol = payload["protocol"]
    baseline_structure = payload["baseline_structure_audit"]
    query_free_structure = payload["structure_audit"]
    initialization = payload["initialization_audit"]
    baseline = payload["paired_smoke"][BASELINE_VARIANT]
    query_free = payload["paired_smoke"][VARIANT]
    baseline_best = baseline["best_metrics"]
    query_free_best = query_free["best_metrics"]
    is_formal = protocol.get("run_kind") == "formal"
    run_label = "400-Epoch Formal Experiment" if is_formal else "3-Epoch Smoke Audit"
    lines = [
        f"Paired {run_label}: Original Query Pool No-Orth vs Query-free Multi-Branch v1",
        "",
        (
            "Status: both paired variants completed the confirmed 400-epoch formal run."
            if is_formal
            else "Status: 3-epoch smoke only. The 400-epoch formal experiment was not run."
        ),
        "",
        "1. Locked protocol",
        f"dataset={protocol['dataset']}",
        f"task={protocol['task']}",
        f"fold={protocol['fold']}",
        f"seed={protocol['seed']}",
        f"epochs={protocol['epochs']}",
        f"split_hash={protocol['split_hash']}",
        f"semantic_fusion={protocol['semantic_fusion']}",
        f"adj_mode={protocol['adj_mode']}",
        f"graph_use_graph={protocol['graph_use_graph']}",
        "ensemble=False",
        "orthogonality_loss=False for both variants",
        "optimizer=Adam (same lr and weight decay)",
        f"scheduler=CustomCosineAnnealingLR, T_max={protocol['scheduler_t_max']}",
        "",
        "2. Baseline-Q-noOrth structure",
        f"parameter_count={baseline_structure['parameter_count']}",
        f"query_pool_count={baseline_structure['query_pool_count']}",
        f"query_pool_parameters_independent={baseline_structure['query_pool_parameters_independent']}",
        f"auxiliary_head_parameters_independent={baseline_structure['auxiliary_head_parameters_independent']}",
        f"auxiliary_supervision={baseline_structure['auxiliary_supervision']}",
        f"DIFFormer={baseline_structure['difformer_class']}",
        f"DIFFormer use_graph={baseline_structure['difformer_graph_use_flags']}",
        "",
        "3. Query-free structure",
        f"parameter_count={query_free_structure['query_free_parameter_count']}",
        f"latent_branch_count={query_free_structure['latent_branch_count']}",
        f"latent_branch_parameters_independent={query_free_structure['latent_branch_parameters_independent']}",
        f"auxiliary_head_parameters_independent={query_free_structure['auxiliary_head_parameters_independent']}",
        "",
        "4. Shared-module initialization audit",
        f"common_tensor_count={initialization['common_tensor_count']}",
        f"common_parameter_and_buffer_count={initialization['common_parameter_and_buffer_count']}",
        f"baseline_common_hash={initialization['baseline_common_hash']}",
        f"query_free_common_hash={initialization['query_free_common_hash']}",
        f"max_abs_diff={initialization['max_abs_diff']}",
        f"mismatched_tensors={initialization['mismatched_tensors']}",
        "",
        f"5. Baseline-Q-noOrth {protocol['epochs']}-epoch result",
        f"best_epoch={baseline['best_epoch']}",
        f"best_ACC={baseline_best['acc']}",
        f"best_Macro-F1={baseline_best['macro_f1']}",
        f"best_BACC={baseline_best['bacc']}",
        f"best_Macro-AUC={baseline_best['macro_auc']}",
        f"best_confusion_matrix={baseline_best['confusion_matrix']}",
        f"best_query_pool_gradient_norms={baseline['best_branch_gradient_norms']}",
        "",
        f"6. Query-free Multi-Branch v1 {protocol['epochs']}-epoch result",
        f"best_epoch={query_free['best_epoch']}",
        f"best_ACC={query_free_best['acc']}",
        f"best_Macro-F1={query_free_best['macro_f1']}",
        f"best_BACC={query_free_best['bacc']}",
        f"best_Macro-AUC={query_free_best['macro_auc']}",
        f"best_confusion_matrix={query_free_best['confusion_matrix']}",
        f"best_latent_branch_gradient_norms={query_free['best_branch_gradient_norms']}",
        "",
        "7. Interpretation",
        "Both paired paths completed forward, loss, backward, diagnostics, and artifact saving.",
        (
            "These are the confirmed single-seed formal results; interpret them within the fixed fold-0 protocol."
            if is_formal
            else "Three-epoch metrics are not a method comparison and must not be used for a paper conclusion."
        ),
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("query_free", "paired"),
        default="query_free",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Explicitly unlock only the confirmed paired 400-epoch formal run.",
    )
    parser.add_argument(
        "--output",
        default="experiments/query_free_multibranch/smoke",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.formal:
        if args.mode != "paired" or args.epochs != 400:
            raise ValueError("Formal mode requires --mode paired --epochs 400 exactly")
    elif args.epochs > 3:
        raise ValueError(
            "This v1 runner is smoke-locked to at most 3 epochs. "
            "Formal 400-epoch execution requires a separately confirmed change."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = ROOT / args.config
    config = Config_(str(ROOT), str(config_path), 0 if device.type == "cuda" else None)
    config.Device = device
    SET_Random(SEED)
    feature_path, dict_path, _, class_names = load_path(
        str(ROOT), config.DATA_SET, config.Task
    )
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

    SET_Random(SEED)
    baseline = build_model(config, dataset_dict, "original", device)
    SET_Random(SEED)
    query_free = build_model(config, dataset_dict, VARIANT, device)
    initialization_audit = copy_and_audit_common_initialization(baseline, query_free)
    structure_audit = branch_structure_audit(baseline, query_free)
    baseline_audit = baseline_structure_audit(baseline)

    if initialization_audit["max_abs_diff"] != 0.0:
        raise RuntimeError("Common-module initialization audit failed")
    if initialization_audit["baseline_common_hash"] != initialization_audit["query_free_common_hash"]:
        raise RuntimeError("Common-module hashes differ after explicit state copy")
    if structure_audit["query_free_has_label_pools"]:
        raise RuntimeError("Query-free model unexpectedly contains legacy label pools")
    if structure_audit["latent_branch_count"] != 3:
        raise RuntimeError("Expected exactly three independent latent branches")
    if not structure_audit["latent_branch_parameters_independent"]:
        raise RuntimeError("Latent branch parameters are unexpectedly shared")
    if not structure_audit["auxiliary_head_parameters_independent"]:
        raise RuntimeError("Latent auxiliary head parameters are unexpectedly shared")
    if any(
        details["contains_multihead_attention"]
        for details in structure_audit["branches"].values()
    ):
        raise RuntimeError("A latent branch unexpectedly contains attention")
    if query_free.semantic_fusion != "add" or query_free.adj_mode != "none":
        raise RuntimeError("Protocol lock for fusion/adjacency was violated")
    if any(structure_audit["difformer_graph_use_flags"]):
        raise RuntimeError("DIFFormer graph use must remain disabled")
    if baseline_audit["semantic_fusion"] != "add" or baseline_audit["adj_mode"] != "none":
        raise RuntimeError("Baseline protocol lock for fusion/adjacency was violated")
    if any(baseline_audit["difformer_graph_use_flags"]):
        raise RuntimeError("Baseline DIFFormer graph use must remain disabled")
    if baseline_audit["query_pool_count"] != 3:
        raise RuntimeError("Expected exactly three original query pools")
    if not baseline_audit["query_pool_parameters_independent"]:
        raise RuntimeError("Original query-pool parameters are unexpectedly shared")
    if not baseline_audit["auxiliary_head_parameters_independent"]:
        raise RuntimeError("Original auxiliary-head parameters are unexpectedly shared")

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "git": git_info(),
        "protocol": {
            "dataset": config.DATA_SET,
            "task": config.Task,
            "fold": FOLD,
            "seed": SEED,
            "epochs": args.epochs,
            "run_kind": "formal" if args.formal else "smoke",
            "device": str(device),
            "transductive_full_batch": True,
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "split_hash": split_hash(dataset_dict["Index"], train_mask, test_mask),
            "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_hash": file_hash(config_path),
            "semantic_fusion": SEMANTIC_FUSION,
            "semantic_branch": "both",
            "adj_mode": "none",
            "graph_use_graph": False,
            "ensemble": False,
            "orthogonality_loss": False,
            "auxiliary_supervision": (
                "paired: original one-vs-rest heads for baseline; "
                "three independent full three-class heads for query-free"
                if args.mode == "paired"
                else "three independent full three-class heads"
            ),
            "optimizer": "Adam",
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "scheduler": "CustomCosineAnnealingLR",
            "scheduler_t_max": config.T_max,
            "scheduler_eta_min": config.Lr_Min,
        },
        "structure_audit": structure_audit,
        "baseline_structure_audit": baseline_audit,
        "initialization_audit": initialization_audit,
    }
    if args.mode == "paired":
        baseline_criterion = criterion_query_pool_no_orth(
            dataset_dict,
            config.Device,
            label_smoothing=0.05,
        )
        query_free_criterion = criterion_query_free_multibranch(
            dataset_dict,
            config.Device,
            label_smoothing=0.05,
            aux_weight=1.0,
        )
        SET_Random(SEED)
        baseline_result = train_smoke(
            baseline,
            baseline_criterion,
            BASELINE_VARIANT,
            {
                f"query_pool_{idx + 1}": pool
                for idx, pool in enumerate(baseline.label_pools)
            },
            config,
            dataset_dict,
            dataset_data,
            args.epochs,
            output_dir / BASELINE_VARIANT,
        )
        SET_Random(SEED)
        query_free_result = train_smoke(
            query_free,
            query_free_criterion,
            VARIANT,
            dict(query_free.latent_branches.items()),
            config,
            dataset_dict,
            dataset_data,
            args.epochs,
            output_dir / VARIANT,
        )
        payload["paired_smoke"] = {
            BASELINE_VARIANT: baseline_result,
            VARIANT: query_free_result,
        }
        report_text = render_paired_txt_report(payload)
        printable_results = payload["paired_smoke"]
    else:
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()
        query_free_criterion = criterion_query_free_multibranch(
            dataset_dict,
            config.Device,
            label_smoothing=0.05,
            aux_weight=1.0,
        )
        SET_Random(SEED)
        payload["smoke_test"] = train_smoke(
            query_free,
            query_free_criterion,
            VARIANT,
            dict(query_free.latent_branches.items()),
            config,
            dataset_dict,
            dataset_data,
            args.epochs,
            output_dir,
        )
        report_text = render_txt_report(payload)
        printable_results = payload["smoke_test"]
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "structure_audit_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "report": str(output_dir / "structure_audit_report.txt"),
                "mode": args.mode,
                "smoke_test": printable_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
