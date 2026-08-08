"""Run MR-LHGR-C1 v1 under the locked formal-C1 protocol.

Stages are explicit: ``smoke`` performs only the prescribed three CUDA steps;
``formal`` runs fresh folds 0..9 at cap 0.10 and conditionally performs the one
allowed full ten-fold cap-0.15 rerun.  There is deliberately no screen stage.
"""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Loss import criterion_query_pool_no_orth
from Model.mr_lhgr import MRLHGRC1Model
from Utils import CustomCosineAnnealingLR, ModelEMA, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "mr_lhgr_c1_v1"
OUTPUT_REL = Path("experiments/mr_lhgr_c1_v1")
BASE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
HISTORICAL_CONFIG = Path("Config/T_ADNI3_tune_light_gm64_dif_lr006_wd0007.ini")
C1_OOF = Path("experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv")
CLASS_COLUMNS = ("AD", "CN", "SMCI")
MODALITY_NAMES = ("MRI", "PET", "CSF", "Risk", "COG", "ROI")
MODALITY_DIMS = (138, 150, 3, 36, 24, 9)
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
GRAPH_K = 8
RANK = 8
C1_PARAMETERS = 862_971
GRAPH_PARAMETERS = 10_752
EXPECTED_PARAMETERS = 873_723

C1 = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
ORIGINAL = {
    "correct": 556,
    "acc": 0.9297658862876255,
    "macro_f1": 0.9140778251271361,
    "bacc": 0.9140778251271361,
    "macro_auc": 0.9560490697782983,
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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
    return {
        "correct": int(candidate["correct"] - baseline["correct"]),
        **{
            key: float(candidate[key] - baseline[key])
            for key in ("acc", "macro_f1", "bacc", "macro_auc")
        },
    }


@torch.no_grad()
def build_modality_relation_graphs(
    features: torch.Tensor, modal_indices, *, k: int
) -> tuple[torch.Tensor, ...]:
    """Build the six fixed, label-free, symmetrically normalized graphs once."""
    require(features.ndim == 2 and features.size(0) > 1, "Invalid graph input")
    require(len(modal_indices) == 6, "Expected six modality index sets")
    subject_count = int(features.size(0))
    k_eff = min(int(k), subject_count - 1)
    require(k_eff > 0, "Invalid effective graph k")
    graphs = []
    for indices in modal_indices:
        # Build the immutable topology in float32 on CPU so nearest-neighbour
        # tie handling is independent of the CUDA cdist kernel.  The model's
        # normal `.to(cuda:0)` moves each registered sparse graph once.
        x_modal = features[:, indices].detach().to(device="cpu", dtype=torch.float32)
        embedding = F.normalize(x_modal, p=2, dim=1, eps=1e-8)
        squared_distance = torch.cdist(embedding, embedding, p=2).square().clamp_min_(0.0)
        squared_distance.fill_diagonal_(float("inf"))
        nearest_distance, nearest_index = torch.topk(
            squared_distance, k=k_eff, dim=1, largest=False, sorted=True
        )
        sigma = nearest_distance[:, -1].sqrt().clamp_min_(1e-6)
        sigma_neighbor = sigma[nearest_index]
        directed_weight = torch.exp(
            -nearest_distance / (sigma[:, None] * sigma_neighbor + 1e-8)
        )
        directed = torch.zeros_like(squared_distance)
        directed.scatter_(1, nearest_index, directed_weight)
        weight = torch.maximum(directed, directed.T)
        weight.fill_diagonal_(0.0)
        degree = weight.sum(dim=1).clamp_min_(1e-8)
        inverse_sqrt_degree = degree.rsqrt()
        normalized = inverse_sqrt_degree[:, None] * weight * inverse_sqrt_degree[None, :]
        require(bool(torch.isfinite(normalized).all()), "Non-finite normalized graph")
        graphs.append(normalized.to_sparse_coo().coalesce().detach())
    return tuple(graphs)


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "MR-LHGR-C1 requires cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "Historical device changed")
    require(context["config"].use_ema is False, "Historical C1 EMA protocol changed")
    require(abs(float(context["config"].ema_decay) - 0.99) < 1e-12, "EMA decay changed")
    modal_indices = context["dataset_dict"]["Modal_Index"]
    require(tuple(len(value) for value in modal_indices) == MODALITY_DIMS, "Modal dimensions changed")
    require(tuple(context["dataset_dict"]["Class_Names"]) == CLASS_COLUMNS, "Class order changed")
    c1_path = ROOT / C1_OOF
    require(c1_path.is_file(), f"C1 OOF missing: {c1_path}")
    c1_rows = read_csv(c1_path)
    require(len(c1_rows) == 598, "C1 OOF row count changed")
    c1_metrics = cme.metrics_from_rows(c1_rows)
    for key in ("correct", "confusion_matrix"):
        require(c1_metrics[key] == C1[key], f"C1 anchor mismatch: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(c1_metrics[key] - C1[key]) <= 5e-7, f"C1 anchor mismatch: {key}")

    # This is the sole graph construction call in one runner process.  It uses
    # only the historical preprocessed model input and never receives labels.
    features = context["dataset_data"]["Feature"]
    graphs = build_modality_relation_graphs(features, modal_indices, k=GRAPH_K)
    require(len(graphs) == 6, "Expected six relation graphs")
    graph_audit = validate_graphs(graphs, features.size(0))
    context.update(
        {
            "relation_graphs": graphs,
            "graph_audit": graph_audit,
            "c1_rows": c1_rows,
            "c1_by_subject": {int(row["subject_index"]): row for row in c1_rows},
        }
    )
    require(len(context["c1_by_subject"]) == 598, "C1 subject duplication")
    return context


def validate_graphs(graphs, subject_count: int) -> dict:
    edge_sets = []
    per_modality = {}
    for name, graph in zip(MODALITY_NAMES, graphs):
        require(graph.layout == torch.sparse_coo and graph.is_coalesced(), f"{name}: graph is not coalesced COO")
        require(tuple(graph.shape) == (subject_count, subject_count), f"{name}: graph shape changed")
        require(not graph.requires_grad and bool(torch.isfinite(graph.values()).all()), f"{name}: graph invalid")
        dense = graph.to_dense()
        require(torch.allclose(dense, dense.T, atol=1e-6, rtol=0.0), f"{name}: graph not symmetric")
        require(float(dense.diag().abs().max()) == 0.0, f"{name}: unexpected self-loop")
        nonzero_degree = (dense > 0).sum(dim=1)
        require(int(nonzero_degree.min()) > 0, f"{name}: isolated subject")
        indices = graph.indices().detach().cpu().numpy().T
        undirected = {(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in indices if i != j}
        edge_sets.append(undirected)
        per_modality[name] = {
            "undirected_edge_count": len(undirected),
            "nonzero_degree_mean": float(nonzero_degree.float().mean().cpu()),
            "nonzero_degree_min": int(nonzero_degree.min().cpu()),
            "normalized_weighted_degree_mean": float(dense.sum(dim=1).mean().cpu()),
        }
    jaccards = []
    for left in range(6):
        for right in range(left + 1, 6):
            union = edge_sets[left] | edge_sets[right]
            jaccards.append(len(edge_sets[left] & edge_sets[right]) / max(1, len(union)))
    return {
        "built_once": True,
        "uses_labels": False,
        "k": GRAPH_K,
        "per_modality": per_modality,
        "edge_jaccard": {
            "mean": float(np.mean(jaccards)),
            "min": float(np.min(jaccards)),
            "max": float(np.max(jaccards)),
            "pair_count": len(jaccards),
        },
    }


def build_model(context: dict, cap: float) -> MRLHGRC1Model:
    config = context["config"]
    SET_Random(SEED)
    model = MRLHGRC1Model(
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
        relation_graphs=context["relation_graphs"],
        graph_rank=RANK,
        graph_residual_cap=cap,
    ).to(context["device"])
    require(len(model.mr_lhgr.relation_graphs) == 6, "Relation graphs were not installed")
    require(parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Historical graph unexpectedly enabled")
    return model


def graph_gradient_groups(model: MRLHGRC1Model) -> dict:
    return {
        "low": [[module.weight] for module in model.mr_lhgr.low_projections],
        "high": [[module.weight] for module in model.mr_lhgr.high_projections],
        "output": [model.mr_lhgr.output_projection.weight],
    }


def make_training_objects(context: dict, cap: float):
    model = build_model(context, cap)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    base_parameters = list(model.c1_parameters())
    graph_parameters = list(model.graph_parameters())
    base_ids = {id(value) for value in base_parameters}
    graph_ids = {id(value) for value in graph_parameters}
    all_ids = {id(value) for value in model.parameters()}
    require(not base_ids & graph_ids and base_ids | graph_ids == all_ids, "Optimizer coverage/overlap failed")
    require(sum(value.numel() for value in base_parameters) == C1_PARAMETERS, "Base parameter count changed")
    require(sum(value.numel() for value in graph_parameters) == GRAPH_PARAMETERS, "Graph parameter count changed")
    config = context["config"]
    base_optimizer = torch.optim.Adam(
        base_parameters, lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    graph_optimizer = torch.optim.Adam(graph_parameters, lr=float(config.lr), weight_decay=0.0)
    base_scheduler = CustomCosineAnnealingLR(
        base_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    graph_scheduler = CustomCosineAnnealingLR(
        graph_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    # Historical C1 evaluation has EMA disabled.  We nevertheless maintain a
    # non-invasive shadow solely so smoke/checkpoint coverage of new parameters
    # is explicit; it is never swapped into the model or used for selection.
    ema = ModelEMA(model, decay=float(config.ema_decay))
    graph_names = {name for name, _ in model.named_parameters() if name.startswith("mr_lhgr.")}
    require(graph_names and graph_names <= set(ema.shadow), "Graph parameters absent from EMA shadow")
    audit = {
        "coverage": 1.0,
        "base_parameter_count": C1_PARAMETERS,
        "graph_parameter_count": GRAPH_PARAMETERS,
        "total_parameter_count": EXPECTED_PARAMETERS,
        "base_lr": float(config.lr),
        "base_weight_decay": float(config.weight_decay),
        "graph_lr": float(config.lr),
        "graph_weight_decay": 0.0,
        "historical_ema_enabled": False,
        "ema_shadow_tracked_for_checkpoint": True,
        "ema_decay": float(config.ema_decay),
        "graph_parameter_names": sorted(graph_names),
    }
    return (
        model,
        criterion,
        base_optimizer,
        graph_optimizer,
        base_scheduler,
        graph_scheduler,
        ema,
        base_parameters,
        graph_parameters,
        audit,
    )


def config_payload(context: dict, cap: float, version: str, folds) -> dict:
    return {
        "experiment": EXPERIMENT,
        "version": version,
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "base_commit": BASE_COMMIT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(folds),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "multi_seed": False,
        "orthogonality": False,
        "graph_k": GRAPH_K,
        "relation_rank": RANK,
        "graph_residual_cap": cap,
        "fixed_unlabeled_modality_graphs": True,
        "old_graph_enabled": False,
        "loss": "historical C1 weighted main CE plus three OVR auxiliary losses; no new loss",
        "optimizer": "separate Adam for C1 and graph parameters",
        "scheduler": "separate synchronized CustomCosineAnnealingLR(T_max=400)",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "historical_ema_enabled": False,
        "ema_shadow_checkpoint_audit_only": True,
        "parameter_count": EXPECTED_PARAMETERS,
        "graph_parameter_count": GRAPH_PARAMETERS,
        "v1_1_relation_dependence_ratio_threshold": 0.99,
        "v1_1_identity_residual_max_abs_threshold": 1e-7,
        "v1_1_relation_logit_max_abs_diff_threshold": 1e-8,
        "v1_1_relation_probability_mean_abs_diff_threshold": 1e-10,
        "v1_1_low_high_increment_nonzero_threshold": 1e-8,
        "v1_1_edge_jaccard_max_threshold": 0.99,
        "v1_1_cap_saturation_threshold": 0.70,
        "obvious_bacc_decline_threshold": 0.003,
        "graph_audit": context["graph_audit"],
    }


def prediction_rows(fold: int, raw_logits, labels, mask, context) -> list[dict]:
    return cme.prediction_rows(
        fold,
        raw_logits,
        labels,
        mask,
        context["dataset_dict"],
        context["config"],
    )


def paired_comparison(candidate_rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"C1 missing subject {subject}")
        reference = reference_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth and int(reference["fold"]) == int(row["fold"]), "C1 alignment mismatch")
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


def metric_and_tuple(raw, labels, mask, context):
    return cme.selection_metrics(
        raw,
        labels,
        mask,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )


def _modal_slice(value, modality: int, subjects: int) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        result = value[modality]
    else:
        require(isinstance(value, torch.Tensor), "Expected tensor/list modality intermediate")
        if value.ndim >= 3 and value.shape[0] == subjects and value.shape[1] == 6:
            result = value[:, modality]
        elif value.ndim >= 3 and value.shape[0] == 6 and value.shape[1] == subjects:
            result = value[modality]
        else:
            raise RuntimeError(f"Cannot locate modality dimension in {tuple(value.shape)}")
    require(result.shape[0] == subjects, "Subject dimension changed in relation intermediate")
    return result


def mechanism_from_intermediates(intermediates: dict, mask: torch.Tensor, gradient_max: dict) -> dict:
    subjects = int(mask.numel())
    relation_tokens = intermediates["relation_tokens"]
    normalized_tokens = F.layer_norm(
        relation_tokens,
        (relation_tokens.size(-1),),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    per_modality = {}
    for modality, name in enumerate(MODALITY_NAMES):
        low_response = _modal_slice(intermediates["relation_low_response"], modality, subjects)
        high_response = _modal_slice(intermediates["relation_high_response"], modality, subjects)
        normalized = _modal_slice(normalized_tokens, modality, subjects)
        low_increment = (low_response - normalized)[mask]
        high_increment = high_response[mask]
        delta_low = _modal_slice(intermediates["relation_delta_low"], modality, subjects)[mask]
        delta_high = _modal_slice(intermediates["relation_delta_high"], modality, subjects)[mask]
        per_modality[name] = {
            "low_relation_increment_rms": float(low_increment.float().square().mean().sqrt().cpu()),
            "high_relation_increment_rms": float(high_increment.float().square().mean().sqrt().cpu()),
            "delta_low_norm_mean": float(delta_low.float().norm(dim=-1).mean().cpu()),
            "delta_high_norm_mean": float(delta_high.float().norm(dim=-1).mean().cpu()),
            "wl_gradient_max": float(gradient_max["low"][modality]),
            "wh_gradient_max": float(gradient_max["high"][modality]),
        }
    ratio = intermediates["graph_residual_h0_ratio"][mask].detach().float().reshape(-1)
    cap_scale = intermediates["graph_cap_scale"][mask].detach().float().reshape(-1)
    saturated = (cap_scale < 1.0 - 1e-7).float()
    require(bool(torch.isfinite(ratio).all()), "Non-finite graph/H0 ratio")
    return {
        "per_modality": per_modality,
        "r_graph_h0_ratio_mean": float(ratio.mean().cpu()),
        "r_graph_h0_ratio_max": float(ratio.max().cpu()),
        "cap_saturation_fraction": float(saturated.mean().cpu()),
        "w_out_gradient_max": float(gradient_max["output"]),
    }


def edge_dependence(
    context: dict,
    model,
    features,
    test_mask,
    real_logits: torch.Tensor,
    real: dict,
) -> dict:
    model.eval()
    with torch.no_grad():
        # The normal-relation result is the already-computed best-checkpoint
        # inference.  Identity therefore costs exactly one additional forward.
        identity_logits, _, _, identity = model(
            features, return_intermediates=True, identity_graph=True
        )
        real_r = real["R_graph"][test_mask].float()
        identity_r = identity["R_graph"][test_mask].float()
        relation_tokens = real["relation_tokens"]
        normalized_tokens = F.layer_norm(
            relation_tokens,
            (relation_tokens.size(-1),),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        real_low_increment = (
            real["relation_low_response"] - normalized_tokens
        )[test_mask].float()
        real_high_increment = real["relation_high_response"][test_mask].float()
        numerator_sq = float((real_r - identity_r).square().sum().cpu())
        denominator_sq = float(real_r.square().sum().cpu())
        ratio = math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + 1e-8)
        real_probability = cme.score_logits(
            real_logits,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )[1][test_mask]
        identity_probability = cme.score_logits(
            identity_logits,
            context["dataset_dict"]["Label_Weight"],
            float(context["config"].logit_adjust_tau),
        )[1][test_mask]
        logits_diff = (real_logits[test_mask] - identity_logits[test_mask]).abs()
        probability_diff = (real_probability - identity_probability).abs()
    return {
        "numerator_squared": numerator_sq,
        "denominator_squared": denominator_sq,
        "edge_dependence_ratio": float(ratio),
        "identity_r_graph_max_abs": float(identity_r.abs().max().cpu()),
        "real_low_relation_increment_max_abs": float(
            real_low_increment.abs().max().cpu()
        ),
        "real_high_relation_increment_max_abs": float(
            real_high_increment.abs().max().cpu()
        ),
        "logits_max_abs_diff": float(logits_diff.max().cpu()),
        "probability_abs_sum": float(probability_diff.sum().cpu()),
        "probability_element_count": int(probability_diff.numel()),
        "probability_mean_abs_diff": float(probability_diff.mean().cpu()),
        "argmax_changed_count": int((real_probability.argmax(dim=-1) != identity_probability.argmax(dim=-1)).sum().cpu()),
    }


def initial_and_first_step_check(context: dict, cap: float) -> dict:
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]
    c1_model = cme.build_model(context, "c1")
    c1_rng = capture_rng()
    graph_model = build_model(context, cap)
    graph_rng = capture_rng()
    require(torch.equal(c1_rng[0], graph_rng[0]), "CPU RNG changed by graph module construction")
    if c1_rng[1] is not None:
        require(all(torch.equal(a, b) for a, b in zip(c1_rng[1], graph_rng[1])), "CUDA RNG changed")
    graph_named = dict(graph_model.named_parameters())
    parameter_diff = max(
        float((parameter.detach() - graph_named[name].detach()).abs().max().cpu())
        for name, parameter in c1_model.named_parameters()
    )
    require(parameter_diff == 0.0, "Initial C1 parameters changed")
    c1_model.eval()
    graph_model.eval()
    with torch.no_grad():
        c1_logits, _, _ = c1_model(features)
        graph_logits, _, _, inter = graph_model(features, return_intermediates=True)
    initial_logit_diff = float((c1_logits - graph_logits).abs().max().cpu())
    initial_r_max = float(inter["R_graph"].abs().max().cpu())
    require(initial_logit_diff <= 1e-7 and initial_r_max == 0.0, "Initial equivalence failed")

    c1_model = cme.build_model(context, "c1")
    c1_criterion = criterion_query_pool_no_orth(context["dataset_dict"], context["device"], label_smoothing=0.05)
    c1_optimizer = torch.optim.Adam(
        c1_model.parameters(), lr=float(context["config"].lr), weight_decay=float(context["config"].weight_decay)
    )
    bundle = make_training_objects(context, cap)
    graph_model, graph_criterion, base_optimizer, graph_optimizer = bundle[:4]
    base_parameters, graph_parameters = bundle[7], bundle[8]
    shared_rng = capture_rng()
    restore_rng(shared_rng)
    c1_model.train()
    c1_optimizer.zero_grad(set_to_none=True)
    c1_logits, c1_branches, c1_auxiliary = c1_model(features)
    c1_loss = c1_criterion(c1_logits, labels, train_mask, c1_branches, c1_auxiliary)
    c1_loss.backward()
    c1_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in c1_model.named_parameters()
    }
    restore_rng(shared_rng)
    graph_model.train()
    base_optimizer.zero_grad(set_to_none=True)
    graph_optimizer.zero_grad(set_to_none=True)
    graph_logits, graph_branches, graph_auxiliary, _ = graph_model(
        features, return_intermediates=True
    )
    graph_loss = graph_criterion(graph_logits, labels, train_mask, graph_branches, graph_auxiliary)
    graph_loss.backward()
    graph_named = dict(graph_model.named_parameters())
    gradient_differences = []
    for name, gradient in c1_grads.items():
        candidate = graph_named[name].grad
        require((gradient is None) == (candidate is None), f"Base gradient coverage changed: {name}")
        if gradient is not None:
            gradient_differences.append(
                float((gradient - candidate).abs().max().cpu())
            )
    gradient_diff = max(gradient_differences, default=0.0)
    require(gradient_diff <= 1e-7, "First-step base gradients changed")
    output_gradient = grad_norm([graph_model.mr_lhgr.output_projection.weight])
    require(output_gradient > 0.0 and math.isfinite(output_gradient), "W_out first-step gradient missing")
    torch.nn.utils.clip_grad_norm_(c1_model.parameters(), float(context["config"].grad_clip))
    torch.nn.utils.clip_grad_norm_(base_parameters, float(context["config"].grad_clip))
    torch.nn.utils.clip_grad_norm_(graph_parameters, float(context["config"].grad_clip))
    c1_optimizer.step()
    base_optimizer.step()
    graph_optimizer.step()
    graph_named = dict(graph_model.named_parameters())
    update_diff = max(
        float((parameter.detach() - graph_named[name].detach()).abs().max().cpu())
        for name, parameter in c1_model.named_parameters()
    )
    require(update_diff <= 1e-7, "First-step base update changed")
    return {
        "passed": True,
        "initial_c1_parameter_max_abs_diff": parameter_diff,
        "initial_logit_max_abs_diff": initial_logit_diff,
        "initial_r_graph_max_abs": initial_r_max,
        "first_step_base_gradient_max_abs_diff": gradient_diff,
        "first_step_base_update_max_abs_diff": update_diff,
        "first_step_w_out_gradient_norm": output_gradient,
        "cpu_cuda_rng_preserved": True,
    }


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    equivalence = initial_and_first_step_check(context, 0.10)
    bundle = make_training_objects(context, 0.10)
    (
        model,
        criterion,
        base_optimizer,
        graph_optimizer,
        base_scheduler,
        graph_scheduler,
        ema,
        base_parameters,
        graph_parameters,
        optimizer_audit,
    ) = bundle
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    before_low = [module.weight.detach().clone() for module in model.mr_lhgr.low_projections]
    before_high = [module.weight.detach().clone() for module in model.mr_lhgr.high_projections]
    before_output = model.mr_lhgr.output_projection.weight.detach().clone()
    gradient_max = {"low": [0.0] * 6, "high": [0.0] * 6, "output": 0.0}
    losses = []
    for epoch in range(1, 4):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        graph_optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), "Smoke non-finite loss")
        loss.backward()
        require(grads_finite(base_parameters) and grads_finite(graph_parameters), "Smoke non-finite gradient")
        for modality in range(6):
            gradient_max["low"][modality] = max(
                gradient_max["low"][modality], grad_norm([model.mr_lhgr.low_projections[modality].weight])
            )
            gradient_max["high"][modality] = max(
                gradient_max["high"][modality], grad_norm([model.mr_lhgr.high_projections[modality].weight])
            )
        gradient_max["output"] = max(
            gradient_max["output"],
            grad_norm([model.mr_lhgr.output_projection.weight]),
        )
        torch.nn.utils.clip_grad_norm_(base_parameters, float(context["config"].grad_clip))
        torch.nn.utils.clip_grad_norm_(graph_parameters, float(context["config"].grad_clip))
        base_optimizer.step()
        graph_optimizer.step()
        ema.update(model)
        base_scheduler.step()
        graph_scheduler.step()
        losses.append(float(loss.detach().cpu()))
    low_delta = [float((module.weight.detach() - before).abs().max().cpu()) for module, before in zip(model.mr_lhgr.low_projections, before_low)]
    high_delta = [float((module.weight.detach() - before).abs().max().cpu()) for module, before in zip(model.mr_lhgr.high_projections, before_high)]
    output_delta = float(
        (model.mr_lhgr.output_projection.weight.detach() - before_output)
        .abs()
        .max()
        .cpu()
    )
    require(gradient_max["output"] > 0.0 and output_delta > 0.0, "W_out did not train")
    require(max(gradient_max["low"]) > 0.0 and max(low_delta) > 0.0, "No WL trained by epoch3")
    require(max(gradient_max["high"]) > 0.0 and max(high_delta) > 0.0, "No WH trained by epoch3")
    model.eval()
    with torch.no_grad():
        reference, _, _, inter = model(features, return_intermediates=True)
    mechanism = mechanism_from_intermediates(inter, test_mask, gradient_max)
    require(
        mechanism["r_graph_h0_ratio_max"] <= 0.10 + 1e-6,
        "Smoke graph residual cap exceeded",
    )
    checkpoint = {
        "model_state": clone_cpu_state(model),
        "ema_shadow": clone_ema_state(ema),
        "config": config_payload(context, 0.10, "smoke", [0]),
    }
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded = build_model(context, 0.10)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    require(set(payload["ema_shadow"]) == set(reloaded.state_dict()), "EMA/checkpoint state coverage changed")
    reloaded.eval()
    with torch.no_grad():
        loaded, _, _, _ = reloaded(features, return_intermediates=True)
    roundtrip = float((reference - loaded).abs().max().cpu())
    require(roundtrip <= 1e-7, "Smoke checkpoint roundtrip failed")
    report = {
        "passed": True,
        "epochs": 3,
        "parameter_count": EXPECTED_PARAMETERS,
        "graph_parameter_count": GRAPH_PARAMETERS,
        "equivalence": equivalence,
        "graph_audit": context["graph_audit"],
        "optimizer_and_ema": optimizer_audit,
        "losses": losses,
        "graph_gradient_max": gradient_max,
        "low_parameter_max_abs_delta": low_delta,
        "high_parameter_max_abs_delta": high_delta,
        "w_out_parameter_max_abs_delta": output_delta,
        "mechanism": mechanism,
        "checkpoint_roundtrip_logit_max_abs_diff": roundtrip,
        "run_command": (
            f'"{sys.executable}" -u -B scripts/run_mr_lhgr_c1_v1.py '
            "smoke --device cuda:0"
        ),
    }
    write_json(output_root / "config.json", config_payload(context, 0.10, "smoke", [0]))
    write_json(output_root / "graph_construction_summary.json", context["graph_audit"])
    write_json(smoke_root / "smoke_report.json", report)
    return report


def fold_config(context: dict, cap: float, version: str, fold: int) -> dict:
    return config_payload(context, cap, version, [fold])


def load_completed_fold(context: dict, cap: float, version: str, fold: int, final_dir: Path):
    if not final_dir.exists():
        return None
    required = [final_dir / name for name in ("summary.json", "epoch_metrics.csv", "best_predictions.csv", "checkpoint_best.pt")]
    require(all(path.is_file() for path in required), f"Incomplete completed fold{fold}")
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(final_dir / "best_predictions.csv")
    epochs = read_csv(final_dir / "epoch_metrics.csv")
    require(summary["fold"] == fold and summary["config"] == fold_config(context, cap, version, fold), "Completed fold config mismatch")
    require(len(epochs) == EPOCHS and [int(row["epoch"]) for row in epochs] == list(range(1, EPOCHS + 1)), "Completed epoch history changed")
    require(len(rows) == summary["test_size"] and cme.metrics_from_rows(rows) == summary["best_metrics"], "Completed predictions changed")
    payload = torch.load(final_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    require(payload["best_epoch"] == summary["best_epoch"] and payload["config"] == summary["config"], "Checkpoint metadata mismatch")
    check_model = build_model(context, cap)
    check_model.load_state_dict(payload["model_state"], strict=True)
    require(set(payload["ema_shadow"]) == set(check_model.state_dict()), "Checkpoint EMA coverage mismatch")
    del check_model, payload
    torch.cuda.empty_cache()
    print(f"RESUME {version} fold={fold} correct={summary['best_metrics']['correct']}", flush=True)
    return summary, rows


def train_fold(context: dict, cap: float, version: str, fold: int, output_root: Path):
    final_dir = output_root / f"fold_{fold:02d}"
    resumed = load_completed_fold(context, cap, version, fold, final_dir)
    if resumed is not None:
        return resumed
    staging = output_root / f".fold_{fold:02d}_in_progress"
    require(not staging.exists(), f"Retained staging directory exists: {staging}")
    staging.mkdir(parents=True)
    bundle = make_training_objects(context, cap)
    (
        model,
        criterion,
        base_optimizer,
        graph_optimizer,
        base_scheduler,
        graph_scheduler,
        ema,
        base_parameters,
        graph_parameters,
        optimizer_audit,
    ) = bundle
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    best = None
    best_state = None
    best_ema = None
    epoch_rows = []
    gradient_max = {"low": [0.0] * 6, "high": [0.0] * 6, "output": 0.0}
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        graph_optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(features, return_intermediates=True)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"{version} fold{fold} epoch{epoch}: non-finite loss")
        loss.backward()
        require(grads_finite(base_parameters) and grads_finite(graph_parameters), f"{version} fold{fold}: non-finite gradient")
        for modality in range(6):
            gradient_max["low"][modality] = max(gradient_max["low"][modality], grad_norm([model.mr_lhgr.low_projections[modality].weight]))
            gradient_max["high"][modality] = max(gradient_max["high"][modality], grad_norm([model.mr_lhgr.high_projections[modality].weight]))
        gradient_max["output"] = max(
            gradient_max["output"],
            grad_norm([model.mr_lhgr.output_projection.weight]),
        )
        torch.nn.utils.clip_grad_norm_(base_parameters, float(context["config"].grad_clip))
        torch.nn.utils.clip_grad_norm_(graph_parameters, float(context["config"].grad_clip))
        base_optimizer.step()
        graph_optimizer.step()
        ema.update(model)
        base_scheduler.step()
        graph_scheduler.step()
        model.eval()
        with torch.no_grad():
            evaluated, _, _, _ = model(features, return_intermediates=True)
            metrics, selection = metric_and_tuple(evaluated, labels, test_mask, context)
        if best is None or selection > best["selection"]:
            best = {"epoch": epoch, "selection": selection, "metrics": deepcopy(metrics)}
            best_state = clone_cpu_state(model)
            best_ema = clone_ema_state(ema)
        epoch_rows.append(
            {
                "epoch": epoch,
                "base_lr": float(base_optimizer.param_groups[0]["lr"]),
                "graph_lr": float(graph_optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                **{key: metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            }
        )
    require(best is not None and best_state is not None and best_ema is not None, "Best checkpoint missing")
    require(gradient_max["output"] > 0 and max(gradient_max["low"]) > 0 and max(gradient_max["high"]) > 0, "Graph branch inactive")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        best_raw, _, _, best_intermediates = model(features, return_intermediates=True)
    rows = prediction_rows(fold, best_raw, labels, test_mask, context)
    require(cme.metrics_from_rows(rows) == best["metrics"], "Best prediction readback changed")
    mechanism = mechanism_from_intermediates(best_intermediates, test_mask, gradient_max)
    dependence = edge_dependence(
        context,
        model,
        features,
        test_mask,
        best_raw,
        best_intermediates,
    )
    fold_reference = {int(row["subject_index"]): row for row in context["c1_rows"] if int(row["fold"]) == fold}
    comparison = paired_comparison(rows, fold_reference)
    summary = {
        "passed": True,
        "version": version,
        "cap": cap,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "parameter_count": EXPECTED_PARAMETERS,
        "graph_parameter_count": GRAPH_PARAMETERS,
        "optimizer_and_ema": optimizer_audit,
        "graph_gradient_max": gradient_max,
        "mechanism": mechanism,
        "edge_dependence": dependence,
        "comparison_vs_c1": comparison,
        "boundary_errors": boundary_errors(best["metrics"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "config": fold_config(context, cap, version, fold),
    }
    torch.save(
        {
            "best_epoch": best["epoch"],
            "model_state": best_state,
            "ema_shadow": best_ema,
            "config": summary["config"],
        },
        staging / "checkpoint_best.pt",
    )
    write_json(staging / "summary.json", summary)
    write_csv(staging / "epoch_metrics.csv", epoch_rows)
    write_csv(staging / "best_predictions.csv", rows)
    staging.rename(final_dir)
    print(
        f"{version} fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']} "
        f"ACC={best['metrics']['acc']:.7f} F1={best['metrics']['macro_f1']:.7f} "
        f"BACC={best['metrics']['bacc']:.7f} AUC={best['metrics']['macro_auc']:.7f}",
        flush=True,
    )
    del model, criterion, base_optimizer, graph_optimizer, base_scheduler, graph_scheduler, ema
    torch.cuda.empty_cache()
    return summary, rows


def validate_oof(rows: list[dict]) -> list[dict]:
    require(len(rows) == 598, "Formal OOF row count changed")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == 598 and set(subjects) == set(range(598)), "Formal OOF subjects invalid")
    require({int(row["fold"]) for row in rows} == set(FOLDS), "Formal OOF fold set changed")
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def aggregate_mechanism(context: dict, summaries: list[dict]) -> dict:
    weights = np.asarray([summary["test_size"] for summary in summaries], dtype=np.float64)
    weights /= weights.sum()
    per_modality = {}
    for name in MODALITY_NAMES:
        per_modality[name] = {
            key: float(
                math.sqrt(
                    sum(
                        weight
                        * summary["mechanism"]["per_modality"][name][key] ** 2
                        for weight, summary in zip(weights, summaries)
                    )
                )
            )
            for key in ("low_relation_increment_rms", "high_relation_increment_rms")
        }
        per_modality[name].update(
            {
                key: float(
                    sum(
                        weight * summary["mechanism"]["per_modality"][name][key]
                        for weight, summary in zip(weights, summaries)
                    )
                )
                for key in ("delta_low_norm_mean", "delta_high_norm_mean")
            }
        )
        per_modality[name]["wl_gradient_max"] = float(max(summary["mechanism"]["per_modality"][name]["wl_gradient_max"] for summary in summaries))
        per_modality[name]["wh_gradient_max"] = float(max(summary["mechanism"]["per_modality"][name]["wh_gradient_max"] for summary in summaries))
        per_modality[name].update(context["graph_audit"]["per_modality"][name])
    numerator_sq = sum(summary["edge_dependence"]["numerator_squared"] for summary in summaries)
    denominator_sq = sum(summary["edge_dependence"]["denominator_squared"] for summary in summaries)
    probability_abs_sum = sum(summary["edge_dependence"]["probability_abs_sum"] for summary in summaries)
    probability_count = sum(summary["edge_dependence"]["probability_element_count"] for summary in summaries)
    edge_dependence_ratio = math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + 1e-8)
    dependence = {
        "edge_dependence_ratio": float(edge_dependence_ratio),
        "identity_r_graph_max_abs": float(max(summary["edge_dependence"]["identity_r_graph_max_abs"] for summary in summaries)),
        "real_low_relation_increment_max_abs": float(
            max(
                summary["edge_dependence"]["real_low_relation_increment_max_abs"]
                for summary in summaries
            )
        ),
        "real_high_relation_increment_max_abs": float(
            max(
                summary["edge_dependence"]["real_high_relation_increment_max_abs"]
                for summary in summaries
            )
        ),
        "logits_max_abs_diff": float(max(summary["edge_dependence"]["logits_max_abs_diff"] for summary in summaries)),
        "probability_mean_abs_diff": float(probability_abs_sum / max(1, probability_count)),
        "argmax_changed_count": int(sum(summary["edge_dependence"]["argmax_changed_count"] for summary in summaries)),
    }
    edge_jaccard = context["graph_audit"]["edge_jaccard"]
    dependence["valid"] = bool(
        dependence["identity_r_graph_max_abs"] <= 1e-7
        and dependence["edge_dependence_ratio"] >= 0.99
        and dependence["logits_max_abs_diff"] > 1e-8
        and dependence["probability_mean_abs_diff"] > 1e-10
        and dependence["real_low_relation_increment_max_abs"] > 1e-8
        and dependence["real_high_relation_increment_max_abs"] > 1e-8
        and edge_jaccard["max"] < 0.99
    )
    return {
        "per_modality": per_modality,
        "r_graph_h0_ratio_mean": float(sum(weight * summary["mechanism"]["r_graph_h0_ratio_mean"] for weight, summary in zip(weights, summaries))),
        "r_graph_h0_ratio_max": float(max(summary["mechanism"]["r_graph_h0_ratio_max"] for summary in summaries)),
        "cap_saturation_fraction": float(sum(weight * summary["mechanism"]["cap_saturation_fraction"] for weight, summary in zip(weights, summaries))),
        "w_out_gradient_max": float(max(summary["mechanism"]["w_out_gradient_max"] for summary in summaries)),
        "edge_jaccard": edge_jaccard,
        "edge_dependence": dependence,
    }


def decision(metrics: dict, comparison: dict, boundaries: dict) -> tuple[str, dict]:
    secondary_improvements = sum(
        metrics[key] > PC_BBF[key] for key in ("macro_f1", "bacc", "macro_auc")
    )
    no_gain = {
        "correct_at_most_560": metrics["correct"] <= 560,
        "repairs_not_above_damages": comparison["repairs"] <= comparison["damages"],
        "ad_cn_error_present": boundaries["AD_CN"] > 0,
        "obvious_bacc_decline": metrics["bacc"] < C1["bacc"] - 0.003,
    }
    target = {
        "correct_at_least_563": metrics["correct"] >= 563,
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "no_ad_cn_error": boundaries["AD_CN"] == 0,
        "macro_f1_drop_at_most_0p003": metrics["macro_f1"] >= C1["macro_f1"] - 0.003,
        "bacc_drop_at_most_0p003": metrics["bacc"] >= C1["bacc"] - 0.003,
    }
    positive = {
        "correct_equals_562": metrics["correct"] == 562,
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "macro_f1_stable": metrics["macro_f1"] >= C1["macro_f1"] - 0.003,
        "bacc_stable": metrics["bacc"] >= C1["bacc"] - 0.003,
    }
    secondary = {
        "correct_equals_561": metrics["correct"] == 561,
        "at_least_two_metrics_above_pc_bbf": secondary_improvements >= 2,
    }
    checks = {"no_gain": no_gain, "target": target, "positive": positive, "secondary": secondary}
    if any(no_gain.values()):
        return "MR_LHGR_NO_GAIN", checks
    if all(target.values()):
        return "MR_LHGR_TARGET_REACHED", checks
    if all(positive.values()):
        return "MR_LHGR_POSITIVE_BELOW_TARGET", checks
    if all(secondary.values()):
        return "MR_LHGR_SECONDARY", checks
    return "MR_LHGR_NO_GAIN", checks


def run_version(context: dict, cap: float, version: str, root: Path) -> tuple[dict, list[dict]]:
    report_path = root / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["config"] == config_payload(context, cap, version, FOLDS), "Completed version config changed")
        rows = validate_oof(read_csv(root / "oof_predictions.csv"))
        require(cme.metrics_from_rows(rows) == report["metrics"], "Completed version metrics changed")
        return report, rows
    root.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for fold in FOLDS:
        summary, fold_rows = train_fold(context, cap, version, fold, root)
        summaries.append(summary)
        rows.extend(fold_rows)
    rows = validate_oof(rows)
    metrics = cme.metrics_from_rows(rows)
    comparison = paired_comparison(rows, context["c1_by_subject"])
    boundaries = boundary_errors(metrics)
    mechanism = aggregate_mechanism(context, summaries)
    version_decision, checks = decision(metrics, comparison, boundaries)
    fold_stats = {}
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        values = np.asarray([summary["best_metrics"][key] for summary in summaries])
        fold_stats[key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    report = {
        "version": version,
        "cap": cap,
        "decision": version_decision,
        "decision_checks": checks,
        "metrics": metrics,
        "fold_metric_mean_sample_std": fold_stats,
        "comparison_vs_c1": comparison,
        "boundary_errors": boundaries,
        "mechanism": mechanism,
        "metric_deltas": {
            "vs_original": metric_delta(metrics, ORIGINAL),
            "vs_c1": metric_delta(metrics, C1),
            "vs_pc_bbf_v1": metric_delta(metrics, PC_BBF),
        },
        "parameter_count": EXPECTED_PARAMETERS,
        "graph_parameter_count": GRAPH_PARAMETERS,
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
        "config": config_payload(context, cap, version, FOLDS),
    }
    write_json(root / "report.json", report)
    write_json(root / "graph_mechanism_summary.json", mechanism)
    write_csv(root / "oof_predictions.csv", rows)
    write_csv(
        root / "per_fold_metrics.csv",
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
    return report, rows


def v1_1_trigger(report: dict) -> tuple[bool, dict]:
    boundaries = report["boundary_errors"]
    comparison = report["comparison_vs_c1"]
    mechanism = report["mechanism"]
    checks = {
        "correct_is_561_or_562": report["metrics"]["correct"] in (561, 562),
        "repairs_above_damages": comparison["repairs"] > comparison["damages"],
        "adjacent_boundary_total_below_c1_38": boundaries["AD_SMCI"] + boundaries["CN_SMCI"] < 38,
        "no_ad_cn_error": boundaries["AD_CN"] == 0,
        "cap_saturation_at_least_0p70": mechanism["cap_saturation_fraction"] >= 0.70,
        "graph_relation_dependence_valid": mechanism["edge_dependence"]["valid"],
    }
    return all(checks.values()), checks


def render_report(summary: dict) -> str:
    selected = summary["selected_report"]
    metrics = selected["metrics"]
    mechanism = selected["mechanism"]
    comparison = selected["comparison_vs_c1"]
    boundary = selected["boundary_errors"]
    lines = [
        "# MR-LHGR-C1 v1",
        "",
        f"Decision: **{summary['decision']}**",
        f"Selected version/cap: **{summary['selected_version']} / {summary['selected_cap']:.2f}**",
        f"Parameters: {selected['parameter_count']} ({selected['graph_parameter_count']} graph); total training time {summary['total_training_seconds']:.3f}s",
        f"Pooled Correct/ACC/F1/BACC/AUC/Weighted-F1: {metrics['correct']}/598 / {metrics['acc']:.7f} / {metrics['macro_f1']:.7f} / {metrics['bacc']:.7f} / {metrics['macro_auc']:.7f} / {metrics['weighted_f1']:.7f}",
        f"Confusion matrix: {metrics['confusion_matrix']}",
        f"Ten-fold ACC mean +/- sample SD: {selected['fold_metric_mean_sample_std']['acc']['mean']:.7f} +/- {selected['fold_metric_mean_sample_std']['acc']['sample_std']:.7f}",
        f"Delta vs Original/C1/PC-BBF ACC: {selected['metric_deltas']['vs_original']['acc']:+.7f} / {selected['metric_deltas']['vs_c1']['acc']:+.7f} / {selected['metric_deltas']['vs_pc_bbf_v1']['acc']:+.7f}",
        f"Repairs/damages/changed vs C1: {comparison['repairs']} / {comparison['damages']} / {comparison['changed_predictions']}",
        f"AD-sMCI / CN-sMCI / AD-CN errors: {boundary['AD_SMCI']} / {boundary['CN_SMCI']} / {boundary['AD_CN']}",
        f"Cap saturation: {mechanism['cap_saturation_fraction']:.7f}; R_graph/H0 mean/max: {mechanism['r_graph_h0_ratio_mean']:.7f}/{mechanism['r_graph_h0_ratio_max']:.7f}",
        f"Edge dependence ratio/logit max/prob mean/argmax changed: {mechanism['edge_dependence']['edge_dependence_ratio']:.7f} / {mechanism['edge_dependence']['logits_max_abs_diff']:.7g} / {mechanism['edge_dependence']['probability_mean_abs_diff']:.7g} / {mechanism['edge_dependence']['argmax_changed_count']}",
        f"Edge Jaccard mean/min/max: {mechanism['edge_jaccard']['mean']:.7f} / {mechanism['edge_jaccard']['min']:.7f} / {mechanism['edge_jaccard']['max']:.7f}",
        f"v1.1 run: {summary['v1_1_ran']} ({summary['v1_1_trigger_checks']})",
        "",
        "## Per-modality low/high contribution",
        "",
    ]
    for name, payload in mechanism["per_modality"].items():
        lines.append(
            f"- {name}: degree {payload['nonzero_degree_mean']:.3f}; low/high RMS "
            f"{payload['low_relation_increment_rms']:.6f}/{payload['high_relation_increment_rms']:.6f}; "
            f"delta norms {payload['delta_low_norm_mean']:.6f}/{payload['delta_high_norm_mean']:.6f}"
        )
    lines.extend(["", summary["next_conclusion"]])
    return "\n".join(lines) + "\n"


def require_committed_unchanged(paths: list[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    require(git("ls-files", "--error-unmatch", "--", *relative, check=False).returncode == 0, "Implementation/smoke is not committed")
    require(git("diff", "--quiet", "HEAD", "--", *relative, check=False).returncode == 0, "Locked source/smoke changed")


def run_formal(context: dict, output_root: Path) -> dict:
    smoke_path = output_root / "smoke/smoke_report.json"
    smoke_config = output_root / "config.json"
    require(smoke_path.is_file() and smoke_config.is_file(), "Passing committed smoke is required")
    require(json.loads(smoke_path.read_text(encoding="utf-8")).get("passed") is True, "Smoke did not pass")
    require_committed_unchanged(
        [ROOT / "Model/mr_lhgr.py", ROOT / "scripts/run_mr_lhgr_c1_v1.py", ROOT / HISTORICAL_CONFIG, smoke_path, smoke_config]
    )
    source_commit = git("rev-parse", "HEAD").stdout.strip()
    v1_report, v1_rows = run_version(context, 0.10, "v1", output_root / "formal_v1_cap_0p10")
    trigger, trigger_checks = v1_1_trigger(v1_report)
    versions = [(v1_report, v1_rows)]
    if trigger:
        v1_1_report, v1_1_rows = run_version(context, 0.15, "v1_1", output_root / "formal_v1_1_cap_0p15")
        versions.append((v1_1_report, v1_1_rows))
    selected_report, selected_rows = max(
        versions,
        key=lambda item: (
            item[0]["metrics"]["correct"],
            item[0]["metrics"]["macro_auc"],
            item[0]["metrics"]["macro_f1"],
        ),
    )
    total_training = float(sum(report["training_seconds"] for report, _ in versions))
    summary = {
        "decision": selected_report["decision"],
        "selected_version": selected_report["version"],
        "selected_cap": selected_report["cap"],
        "selected_report": selected_report,
        "v1_report": v1_report,
        "v1_1_ran": trigger,
        "v1_1_trigger_checks": trigger_checks,
        "v1_1_report": versions[1][0] if trigger else None,
        "total_training_seconds": total_training,
        "source_commit": source_commit,
        "run_command": (
            f'"{sys.executable}" -u -B scripts/run_mr_lhgr_c1_v1.py '
            "formal --device cuda:0"
        ),
        "next_conclusion": (
            "Target reached; stop without further MR-LHGR adjustment."
            if selected_report["decision"] == "MR_LHGR_TARGET_REACHED"
            else "Target not reached; stop this experiment without another cap or model."
        ),
    }
    write_json(output_root / "config.json", selected_report["config"])
    write_json(output_root / "formal_summary.json", summary)
    write_json(output_root / "graph_mechanism_summary.json", selected_report["mechanism"])
    write_csv(output_root / "formal_oof_predictions.csv", selected_rows)
    write_csv(
        output_root / "per_fold_metrics.csv",
        [
            {
                "fold": fold["fold"],
                "best_epoch": fold["best_epoch"],
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
    require(git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD", check=False).returncode == 0, "Branch is not based on formal C1")
    require(git("branch", "--show-current").stdout.strip() == "experiment/mr-lhgr-c1-v1", "Wrong branch")


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
    print(f"MR-LHGR-C1 fixed graphs ready; cap={'0.10' if args.stage == 'smoke' else '0.10 (v1)'}", flush=True)
    if args.stage == "smoke":
        result = run_smoke(context, output_root)
        print(f"SMOKE passed={result['passed']}", flush=True)
    else:
        result = run_formal(context, output_root)
        print(f"{result['decision']} selected={result['selected_version']}", flush=True)


if __name__ == "__main__":
    main()
