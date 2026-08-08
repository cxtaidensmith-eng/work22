"""Run Gradient-Isolated Pairwise Boundary Fusion v1.

The historical C1 path is trained exactly as before.  A separate, detached
pairwise correction head is optimized with its own Adam/scheduler, so neither
loss can update the other parameter set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from Model.gi_pbf import (
    C1_EXPECTED_PARAMETERS,
    GI_PBF_ADDED_PARAMETERS_H96_R8,
    GI_PBF_EXPECTED_PARAMETERS_H96_R8,
    GIPBFModel,
)
from Utils import CustomCosineAnnealingLR, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "gi_pbf_v1"
OUTPUT_REL = Path("experiments/gi_pbf_v1")
REQUESTED_BASE = "7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840"
ACTUAL_BASE = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
C1_OOF_REL = Path(
    "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
)
PC_BBF_REF = "refs/remotes/origin/experiment/pc-bbf-c1-v1"
PC_BBF_OOF_REL = "experiments/pc_bbf_c1_v1/formal/oof_predictions.csv"
CLASS_NAMES = ("AD", "CN", "SMCI")
SEED = 0
EPOCHS = 400
SCREEN_FOLDS = (4, 5, 6, 7)
FORMAL_FOLDS = tuple(range(10))
PAIR_RANK = 8
CAP_COEFFICIENT = 0.10
EPSILON = 1e-12

C1_REFERENCE = {
    "correct": 560,
    "acc": 0.9364548494983278,
    "macro_f1": 0.9175457174222448,
    "bacc": 0.9163359339143832,
    "macro_auc": 0.9585607750856947,
    "weighted_f1": 0.936365986325677,
    "confusion_matrix": [[61, 0, 11], [0, 201, 8], [10, 9, 298]],
}
SCREEN_C1 = {
    "correct": 226,
    "macro_f1": 0.9222470238095237,
    "bacc": 0.9222470238095237,
    "macro_auc": 0.9610275549894506,
    "confusion_matrix": [[24, 0, 4], [0, 81, 3], [4, 3, 121]],
    "AD_SMCI": 8,
    "CN_SMCI": 6,
    "AD_CN": 0,
    "fold_correct": {4: 55, 5: 57, 6: 58, 7: 56},
    "fold_best_epoch": {4: 103, 5: 180, 6: 332, 7: 181},
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


def reproduction_command(stage: str) -> str:
    return f'"{Path(sys.executable).resolve()}" -u -B scripts/run_gi_pbf_v1.py {stage} --device cuda:0'


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


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


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
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
        if parameter.grad is not None
    )


def capture_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    return cpu, cuda


def restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "GI-PBF v1 requires device=cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "C1 context device changed")
    c1_path = ROOT / C1_OOF_REL
    require(c1_path.is_file(), f"C1 OOF missing: {c1_path}")
    c1_rows = read_csv(c1_path)
    require(len(c1_rows) == 598, "C1 OOF row count changed")
    c1_metrics = cme.metrics_from_rows(c1_rows)
    for key in ("correct", "confusion_matrix"):
        require(c1_metrics[key] == C1_REFERENCE[key], f"C1 anchor changed: {key}")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(abs(c1_metrics[key] - C1_REFERENCE[key]) <= 5e-7, f"C1 anchor changed: {key}")
    by_subject = {int(row["subject_index"]): row for row in c1_rows}
    require(len(by_subject) == 598, "C1 duplicate subject")
    _, _, _, actual_class_names = cme.load_path(
        str(ROOT), context["config"].DATA_SET, context["config"].Task
    )
    canonical = {str(name).upper(): index for index, name in enumerate(actual_class_names)}
    require(set(canonical) == {"AD", "CN", "SMCI"}, f"Unexpected actual class mapping: {canonical}")
    class_mapping = {"AD": canonical["AD"], "CN": canonical["CN"], "sMCI": canonical["SMCI"]}
    context.update(
        {
            "c1_rows": c1_rows,
            "c1_by_subject": by_subject,
            "actual_class_names": tuple(actual_class_names),
            "class_mapping": class_mapping,
            "historical_config_path": ROOT / cme.CONFIG_REL,
            "historical_config_sha256": hashlib.sha256((ROOT / cme.CONFIG_REL).read_bytes()).hexdigest(),
        }
    )
    return context


def build_gi_model(context: dict) -> GIPBFModel:
    config = context["config"]
    SET_Random(SEED)
    model = GIPBFModel(
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
        class_names=context["actual_class_names"],
        pair_rank=PAIR_RANK,
        cap_coefficient=CAP_COEFFICIENT,
        adapter_rank=8,
        router_hidden=16,
        modality_embedding_dim=8,
    ).to(context["device"])
    require(parameter_count(model) == GI_PBF_EXPECTED_PARAMETERS_H96_R8, "GI-PBF parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    print(f"GI-PBF class mapping={model.class_indices}", flush=True)
    return model


def make_training_objects(context: dict):
    model = build_gi_model(context)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    base_parameters = list(model.c1_parameters())
    correction_parameters = list(model.correction_parameters())
    base_ids = {id(value) for value in base_parameters}
    correction_ids = {id(value) for value in correction_parameters}
    all_ids = {id(value) for value in model.parameters()}
    require(not (base_ids & correction_ids), "Optimizer parameter overlap")
    require(base_ids | correction_ids == all_ids, "Optimizer coverage mismatch")
    require(sum(value.numel() for value in base_parameters) == C1_EXPECTED_PARAMETERS, "C1 parameter count changed")
    require(
        sum(value.numel() for value in correction_parameters) == GI_PBF_ADDED_PARAMETERS_H96_R8,
        "Correction parameter count changed",
    )
    config = context["config"]
    base_optimizer = torch.optim.Adam(
        base_parameters, lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    correction_optimizer = torch.optim.Adam(
        correction_parameters, lr=float(config.lr), weight_decay=0.0
    )
    base_scheduler = CustomCosineAnnealingLR(
        base_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    correction_scheduler = CustomCosineAnnealingLR(
        correction_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    return (
        model,
        criterion,
        base_optimizer,
        correction_optimizer,
        base_scheduler,
        correction_scheduler,
        base_parameters,
        correction_parameters,
    )


def pair_loss(model: GIPBFModel, intermediates: dict, labels: torch.Tensor, train_mask: torch.Tensor):
    losses = []
    counts = {}
    for name, first_class, pair_key in (
        ("ad_smci", "AD", "pair_logits_ad_smci"),
        ("cn_smci", "CN", "pair_logits_cn_smci"),
    ):
        first_index = model.class_indices[first_class]
        smci_index = model.class_indices["sMCI"]
        eligible = train_mask & ((labels == first_index) | (labels == smci_index))
        pair_labels = labels[eligible]
        targets = (pair_labels == smci_index).long()
        n_first = int((pair_labels == first_index).sum())
        n_smci = int((pair_labels == smci_index).sum())
        n_pair = n_first + n_smci
        require(n_first > 0 and n_smci > 0, f"Empty pair class: {name}")
        weights = torch.tensor(
            [n_pair / (2.0 * n_first), n_pair / (2.0 * n_smci)],
            device=labels.device,
            dtype=intermediates[pair_key].dtype,
        )
        losses.append(F.cross_entropy(intermediates[pair_key][eligible], targets, weight=weights))
        counts[name] = {"first": n_first, "smci": n_smci, "total": n_pair, "weights": weights.detach().cpu().tolist()}
    return 0.5 * (losses[0] + losses[1]), {"loss_ad_smci": losses[0], "loss_cn_smci": losses[1], "counts": counts}


def metric_and_tuple(raw_logits, labels, mask, context):
    return cme.selection_metrics(
        raw_logits,
        labels,
        mask,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )


def _grad_is_zero(parameters) -> bool:
    return all(
        parameter.grad is None or bool((parameter.grad.detach() == 0).all())
        for parameter in parameters
    )


def _common_parameter_diff(c1_model, gi_model) -> float:
    gi_named = dict(gi_model.named_parameters())
    differences = []
    for name, parameter in c1_model.named_parameters():
        require(name in gi_named, f"GI-PBF missing C1 parameter {name}")
        differences.append(float((parameter.detach() - gi_named[name].detach()).abs().max().cpu()))
    return max(differences, default=0.0)


def gradient_isolation_check(context: dict) -> dict:
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _ = context["dataset_data"]["Mask"][0]

    # Static initialization and post-construction RNG equality.
    c1_model = cme.build_model(context, "c1")
    c1_rng = capture_rng()
    gi_model = build_gi_model(context)
    gi_rng = capture_rng()
    require(torch.equal(c1_rng[0], gi_rng[0]), "CPU RNG changed by GI construction")
    if c1_rng[1] is not None:
        require(all(torch.equal(left, right) for left, right in zip(c1_rng[1], gi_rng[1])), "CUDA RNG changed by GI construction")
    initial_parameter_diff = _common_parameter_diff(c1_model, gi_model)
    require(initial_parameter_diff == 0.0, "C1 initialization changed")

    c1_model.eval()
    gi_model.eval()
    with torch.no_grad():
        c1_raw, _, _ = c1_model(features)
        base_raw, corrected_raw, _, _, inter = gi_model(features, return_intermediates=True)
    epoch0_base_diff = float((c1_raw - base_raw).abs().max().cpu())
    epoch0_corrected_diff = float((corrected_raw - base_raw).abs().max().cpu())
    require(epoch0_base_diff <= 1e-6 and epoch0_corrected_diff <= 1e-6, "Epoch-0 equality failed")
    require(float(inter["uncertainty_ad_smci"].min()) >= -1e-8 and float(inter["uncertainty_ad_smci"].max()) <= 1.0 + 1e-6, "AD uncertainty range failed")
    require(float(inter["uncertainty_cn_smci"].min()) >= -1e-8 and float(inter["uncertainty_cn_smci"].max()) <= 1.0 + 1e-6, "CN uncertainty range failed")

    # Direction 1: correction loss must not touch C1.
    gi_model.train()
    gi_model.zero_grad(set_to_none=True)
    base_raw, corrected_raw, branches, auxiliary, inter = gi_model(features, return_intermediates=True)
    correction_loss, correction_detail = pair_loss(gi_model, inter, labels, train_mask)
    correction_loss.backward()
    c1_parameters = list(gi_model.c1_parameters())
    correction_parameters = list(gi_model.correction_parameters())
    correction_to_c1_isolated = _grad_is_zero(c1_parameters)
    output_row_gradients = gi_model.gi_pbf.output.weight.grad.detach().norm(dim=1).cpu().tolist()
    require(correction_to_c1_isolated, "Correction loss entered C1 parameters")
    require(all(math.isfinite(value) and value > 0.0 for value in output_row_gradients), "Correction output row inactive")

    # Direction 2: historical base loss must not touch the correction head.
    gi_model.zero_grad(set_to_none=True)
    base_raw, _, branches, auxiliary, _ = gi_model(features, return_intermediates=True)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    base_loss = criterion(base_raw, labels, train_mask, branches, auxiliary)
    base_loss.backward()
    base_to_correction_isolated = _grad_is_zero(correction_parameters)
    require(base_to_correction_isolated, "Base loss entered correction parameters")

    # One historical base step in lockstep with standalone C1.
    del c1_model, gi_model, criterion
    torch.cuda.empty_cache()
    c1_model = cme.build_model(context, "c1")
    c1_criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    c1_optimizer = torch.optim.Adam(
        c1_model.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    gi_bundle = make_training_objects(context)
    gi_model, gi_criterion, gi_base_optimizer = gi_bundle[:3]
    gi_base_parameters = gi_bundle[6]
    shared_rng = capture_rng()
    c1_model.train()
    gi_model.train()
    restore_rng(shared_rng)
    c1_optimizer.zero_grad(set_to_none=True)
    c1_raw, c1_branches, c1_auxiliary = c1_model(features)
    c1_loss = c1_criterion(c1_raw, labels, train_mask, c1_branches, c1_auxiliary)
    c1_loss.backward()
    torch.nn.utils.clip_grad_norm_(c1_model.parameters(), float(context["config"].grad_clip))
    c1_optimizer.step()
    restore_rng(shared_rng)
    gi_base_optimizer.zero_grad(set_to_none=True)
    gi_base_raw, _, gi_branches, gi_auxiliary, _ = gi_model(features, return_intermediates=True)
    gi_base_loss = gi_criterion(gi_base_raw, labels, train_mask, gi_branches, gi_auxiliary)
    gi_base_loss.backward()
    torch.nn.utils.clip_grad_norm_(gi_base_parameters, float(context["config"].grad_clip))
    gi_base_optimizer.step()
    loss_diff = abs(float(c1_loss.detach().cpu()) - float(gi_base_loss.detach().cpu()))
    one_step_parameter_diff = _common_parameter_diff(c1_model, gi_model)
    require(loss_diff <= 1e-8, "Base one-step loss changed")
    require(one_step_parameter_diff <= 1e-7, "Base one-step parameters changed")

    payload = {
        "passed": True,
        "class_mapping": gi_model.class_indices,
        "c1_parameter_count": C1_EXPECTED_PARAMETERS,
        "correction_parameter_count": GI_PBF_ADDED_PARAMETERS_H96_R8,
        "total_parameter_count": GI_PBF_EXPECTED_PARAMETERS_H96_R8,
        "initial_c1_parameter_max_abs_diff": initial_parameter_diff,
        "epoch0_c1_vs_base_logit_max_abs_diff": epoch0_base_diff,
        "epoch0_corrected_vs_base_logit_max_abs_diff": epoch0_corrected_diff,
        "cpu_rng_equal_after_construction": True,
        "cuda_rng_equal_after_construction": True,
        "correction_loss_to_c1_isolated": correction_to_c1_isolated,
        "base_loss_to_correction_isolated": base_to_correction_isolated,
        "correction_output_row_gradient_norms": output_row_gradients,
        "pair_counts": correction_detail["counts"],
        "base_one_step_loss_abs_diff": loss_diff,
        "base_one_step_common_parameter_max_abs_diff": one_step_parameter_diff,
    }
    del c1_model, gi_model
    torch.cuda.empty_cache()
    return payload


def config_payload(context: dict, stage: str, folds) -> dict:
    return {
        "experiment": EXPERIMENT,
        "stage": stage,
        "requested_base_commit": REQUESTED_BASE,
        "actual_c1_formal_base_commit": ACTUAL_BASE,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(folds),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "graph_enabled": False,
        "orthogonality": False,
        "base_loss": "historical weighted main CE plus three C1 OVR losses",
        "correction_loss": "0.5 * (balanced AD-sMCI CE + balanced CN-sMCI CE)",
        "base_optimizer": {"name": "Adam", "lr": float(context["config"].lr), "weight_decay": float(context["config"].weight_decay)},
        "correction_optimizer": {"name": "Adam", "lr": float(context["config"].lr), "weight_decay": 0.0},
        "base_scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "correction_scheduler": "CustomCosineAnnealingLR(T_max=400)",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "historical_config_path": str(context["historical_config_path"].relative_to(ROOT)).replace("\\", "/"),
        "historical_config_sha256": context["historical_config_sha256"],
        "pair_rank": PAIR_RANK,
        "cap_coefficient": CAP_COEFFICIENT,
        "class_mapping": context["class_mapping"],
        "uncertainty_probability": "softmax(raw base logits.detach())",
        "final_probability": "historical logit-adjusted softmax",
        "screen_nonconstant_std_threshold": 1e-8,
        "screen_tanh_saturation_threshold": 0.99,
        "screen_max_saturation_fraction": 0.95,
        "adjacent_boundary_degradation_definition": "any count above the corresponding C1 count",
        "parameter_count": GI_PBF_EXPECTED_PARAMETERS_H96_R8,
    }


def run_smoke(context: dict, output_root: Path) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    isolation = gradient_isolation_check(context)
    write_json(output_root / "gradient_isolation_check.json", isolation)
    write_json(output_root / "config.json", config_payload(context, "smoke", [0]))

    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask = context["dataset_data"]["Mask"][0]
    bundle = make_training_objects(context)
    (
        model,
        criterion,
        base_optimizer,
        correction_optimizer,
        base_scheduler,
        correction_scheduler,
        base_parameters,
        correction_parameters,
    ) = bundle
    model.eval()
    with torch.no_grad():
        initial_base, initial_corrected, _, _, initial_inter = model(features, return_intermediates=True)
    initial_diff = float((initial_corrected - initial_base).abs().max().cpu())
    require(initial_diff <= 1e-6, "Smoke epoch-0 equality failed")
    maxima = {"input_projection": 0.0, "output_ad_smci": 0.0, "output_cn_smci": 0.0}
    losses = []
    last_inter = None
    for epoch in range(1, 4):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        correction_optimizer.zero_grad(set_to_none=True)
        base_raw, corrected_raw, branches, auxiliary, inter = model(features, return_intermediates=True)
        base_loss = criterion(base_raw, labels, train_mask, branches, auxiliary)
        correction_loss, _ = pair_loss(model, inter, labels, train_mask)
        base_loss.backward()
        correction_loss.backward()
        require(grads_finite(base_parameters) and grads_finite(correction_parameters), "Smoke non-finite gradient")
        input_grad = grad_norm(model.gi_pbf.input_projection.parameters())
        row_grads = model.gi_pbf.output.weight.grad.detach().norm(dim=1).cpu().tolist()
        maxima["input_projection"] = max(maxima["input_projection"], input_grad)
        maxima["output_ad_smci"] = max(maxima["output_ad_smci"], row_grads[0])
        maxima["output_cn_smci"] = max(maxima["output_cn_smci"], row_grads[1])
        torch.nn.utils.clip_grad_norm_(base_parameters, float(context["config"].grad_clip))
        torch.nn.utils.clip_grad_norm_(correction_parameters, float(context["config"].grad_clip))
        base_optimizer.step()
        correction_optimizer.step()
        base_scheduler.step()
        correction_scheduler.step()
        total = float((base_loss.detach() + correction_loss.detach()).cpu())
        require(math.isfinite(total), "Smoke non-finite loss")
        losses.append({"epoch": epoch, "base_loss": float(base_loss.detach().cpu()), "correction_loss": float(correction_loss.detach().cpu())})
        last_inter = inter
    require(all(math.isfinite(value) and value > 0.0 for value in maxima.values()), "Smoke correction head inactive")
    require(last_inter is not None, "Smoke missing intermediates")
    for key in ("delta_ad_smci", "delta_cn_smci", "uncertainty_ad_smci", "uncertainty_cn_smci"):
        require(bool(torch.isfinite(last_inter[key]).all()), f"Smoke non-finite {key}")
    for key in ("uncertainty_ad_smci", "uncertainty_cn_smci"):
        require(float(last_inter[key].min()) >= -1e-8 and float(last_inter[key].max()) <= 1.0 + 1e-6, f"Smoke range failed: {key}")

    checkpoint = smoke_root / "checkpoint_roundtrip.pt"
    torch.save(model.state_dict(), checkpoint)
    model.eval()
    with torch.no_grad():
        reference_base, reference_corrected, _, _, _ = model(features, return_intermediates=True)
    reloaded = build_gi_model(context)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=context["device"], weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        loaded_base, loaded_corrected, _, _, _ = reloaded(features, return_intermediates=True)
        inference_corrected, _, _ = reloaded(features)
    base_roundtrip = float((reference_base - loaded_base).abs().max().cpu())
    corrected_roundtrip = float((reference_corrected - loaded_corrected).abs().max().cpu())
    inference_diff = float((inference_corrected - loaded_corrected).abs().max().cpu())
    require(max(base_roundtrip, corrected_roundtrip, inference_diff) <= 1e-6, "Smoke checkpoint/inference mismatch")
    smoke_base_rows = prediction_rows(0, loaded_base, labels, test_mask, context)
    smoke_corrected_rows = prediction_rows(0, loaded_corrected, labels, test_mask, context)
    write_csv(smoke_root / "base_logits_and_predictions.csv", smoke_base_rows)
    write_csv(smoke_root / "corrected_logits_and_predictions.csv", smoke_corrected_rows)
    report = {
        "passed": True,
        "epochs": 3,
        "parameter_count": parameter_count(model),
        "class_mapping": model.class_indices,
        "epoch0_corrected_base_max_abs_diff": initial_diff,
        "losses": losses,
        "correction_gradient_max": maxima,
        "delta_ad_smci_abs_max": float(last_inter["delta_ad_smci"].detach().abs().max().cpu()),
        "delta_cn_smci_abs_max": float(last_inter["delta_cn_smci"].detach().abs().max().cpu()),
        "uncertainty_ad_smci_min_max": [float(last_inter["uncertainty_ad_smci"].min()), float(last_inter["uncertainty_ad_smci"].max())],
        "uncertainty_cn_smci_min_max": [float(last_inter["uncertainty_cn_smci"].min()), float(last_inter["uncertainty_cn_smci"].max())],
        "checkpoint_base_logit_max_abs_diff": base_roundtrip,
        "checkpoint_corrected_logit_max_abs_diff": corrected_roundtrip,
        "ordinary_inference_is_corrected_max_abs_diff": inference_diff,
        "base_logits_saved_for_comparison": True,
        "base_logits_file": "smoke/base_logits_and_predictions.csv",
        "corrected_logits_file": "smoke/corrected_logits_and_predictions.csv",
        "gradient_isolation_check": isolation,
        "run_command": reproduction_command("smoke"),
    }
    write_json(smoke_root / "smoke_report.json", report)
    return report


def prediction_rows(fold: int, raw_logits, labels, mask, context, intermediates=None) -> list[dict]:
    rows = cme.prediction_rows(
        fold,
        raw_logits,
        labels,
        mask,
        context["dataset_dict"],
        context["config"],
    )
    if intermediates is None:
        return rows
    indices = torch.where(mask)[0]
    values = {
        key: intermediates[key][indices].detach().cpu().numpy()
        for key in (
            "raw_delta",
            "delta_ad_smci",
            "delta_cn_smci",
            "uncertainty_ad_smci",
            "uncertainty_cn_smci",
        )
    }
    for row_index, row in enumerate(rows):
        row["raw_delta_ad_smci"] = float(values["raw_delta"][row_index, 0])
        row["raw_delta_cn_smci"] = float(values["raw_delta"][row_index, 1])
        row["delta_ad_smci"] = float(values["delta_ad_smci"][row_index])
        row["delta_cn_smci"] = float(values["delta_cn_smci"][row_index])
        row["uncertainty_ad_smci"] = float(values["uncertainty_ad_smci"][row_index])
        row["uncertainty_cn_smci"] = float(values["uncertainty_cn_smci"][row_index])
    return rows


def diagnostics_from_intermediates(intermediates: dict, mask: torch.Tensor) -> dict:
    output = {}
    for name in ("ad_smci", "cn_smci"):
        raw = intermediates["raw_delta"][mask, 0 if name == "ad_smci" else 1].detach().cpu().numpy()
        delta = intermediates[f"delta_{name}"][mask].detach().cpu().numpy()
        uncertainty = intermediates[f"uncertainty_{name}"][mask].detach().cpu().numpy()
        output[name] = {
            "raw_delta_mean": float(raw.mean()),
            "raw_delta_abs_mean": float(np.abs(raw).mean()),
            "raw_delta_abs_max": float(np.abs(raw).max()),
            "raw_delta_std": float(raw.std(ddof=0)),
            "delta_mean": float(delta.mean()),
            "delta_abs_mean": float(np.abs(delta).mean()),
            "delta_abs_max": float(np.abs(delta).max()),
            "delta_std": float(delta.std(ddof=0)),
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_min": float(uncertainty.min()),
            "uncertainty_max": float(uncertainty.max()),
            "tanh_saturation_fraction": float((np.abs(np.tanh(raw)) >= 0.99).mean()),
        }
    return output


def class_names_in_index_order(mapping: dict) -> list[str]:
    names = [None] * len(mapping)
    for name, index in mapping.items():
        require(0 <= int(index) < len(names), f"Invalid class index: {name}={index}")
        require(names[int(index)] is None, f"Duplicate class index: {index}")
        names[int(index)] = str(name)
    require(all(name is not None for name in names), "Incomplete class mapping")
    return names


def _validate_prediction_rows_for_fold(rows: list[dict], fold: int, context: dict) -> None:
    _, test_mask = context["dataset_data"]["Mask"][fold]
    mask_numpy = test_mask.detach().cpu().numpy().astype(bool)
    source_indices = np.asarray(context["dataset_dict"]["Index"], dtype=np.int64)[mask_numpy]
    truth_values = (
        context["dataset_data"]["Label"][test_mask].detach().cpu().numpy().astype(np.int64)
    )
    expected_truth = {
        int(subject): int(truth) for subject, truth in zip(source_indices, truth_values)
    }
    expected_subjects = set(expected_truth)
    observed_subjects = {int(row["subject_index"]) for row in rows}
    require(len(rows) == len(expected_subjects), f"fold{fold}: prediction row count mismatch")
    require(len(observed_subjects) == len(rows), f"fold{fold}: duplicate prediction subject")
    require(observed_subjects == expected_subjects, f"fold{fold}: prediction subject set mismatch")
    for row in rows:
        subject = int(row["subject_index"])
        require(int(row["fold"]) == fold, f"fold{fold}: prediction fold mismatch")
        require(int(row["truth"]) == expected_truth[subject], f"fold{fold}: prediction truth mismatch")
        reference = context["c1_by_subject"][subject]
        require(
            int(reference["fold"]) == fold and int(reference["truth"]) == int(row["truth"]),
            f"fold{fold}: C1 reference alignment mismatch",
        )


def _load_completed_fold(
    final_dir: Path, fold: int, context: dict
) -> tuple[dict, list[dict], list[dict]] | None:
    if not final_dir.exists():
        return None
    summary_path = final_dir / "summary.json"
    base_path = final_dir / "base_best_predictions.csv"
    corrected_path = final_dir / "corrected_best_predictions.csv"
    epoch_path = final_dir / "epoch_metrics.csv"
    checkpoint = final_dir / "checkpoint_best.pt"
    require(all(path.is_file() for path in (summary_path, base_path, corrected_path, epoch_path, checkpoint)), f"Incomplete completed fold {fold}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_rows = read_csv(base_path)
    corrected_rows = read_csv(corrected_path)
    require(summary["fold"] == fold and summary["epochs"] == EPOCHS, "Completed fold metadata mismatch")
    require(summary.get("seed") == SEED, "Completed fold seed changed")
    require(summary["parameter_count"] == GI_PBF_EXPECTED_PARAMETERS_H96_R8, "Completed fold parameters changed")
    require(summary.get("config") == config_payload(context, "fold", [fold]), "Completed fold config changed")
    epoch_rows = read_csv(epoch_path)
    require(len(epoch_rows) == EPOCHS, "Completed fold epoch history incomplete")
    require([int(row["epoch"]) for row in epoch_rows] == list(range(1, EPOCHS + 1)), "Completed fold epoch sequence changed")
    require(len(base_rows) == len(corrected_rows) == summary["test_size"], "Completed fold prediction size mismatch")
    _validate_prediction_rows_for_fold(base_rows, fold, context)
    _validate_prediction_rows_for_fold(corrected_rows, fold, context)
    require(cme.metrics_from_rows(base_rows) == summary["base_best"]["metrics"], "Completed base metrics changed")
    require(cme.metrics_from_rows(corrected_rows) == summary["corrected_best"]["metrics"], "Completed corrected metrics changed")
    fold_reference = {
        int(row["subject_index"]): row for row in context["c1_rows"] if int(row["fold"]) == fold
    }
    require(summary.get("comparison_vs_c1") == paired_comparison(corrected_rows, fold_reference), "Completed fold paired comparison changed")
    require(
        summary.get("boundary_errors")
        == boundary_errors(summary["corrected_best"]["metrics"], context["class_mapping"]),
        "Completed fold boundary metrics changed",
    )
    checkpoint_payload = torch.load(checkpoint, map_location=context["device"], weights_only=True)
    require(
        int(checkpoint_payload["base_best_epoch"]) == int(summary["base_best"]["epoch"])
        and int(checkpoint_payload["corrected_best_epoch"]) == int(summary["corrected_best"]["epoch"]),
        "Completed fold checkpoint epoch changed",
    )
    require(checkpoint_payload.get("config") == summary["config"], "Completed fold checkpoint config changed")
    check_model = build_gi_model(context)
    check_model.load_state_dict(checkpoint_payload["base_best_state"], strict=True)
    check_model.load_state_dict(checkpoint_payload["corrected_best_state"], strict=True)
    del check_model, checkpoint_payload
    torch.cuda.empty_cache()
    print(f"RESUME fold={fold} base={summary['base_best']['correct']} corrected={summary['corrected_best']['correct']}", flush=True)
    return summary, base_rows, corrected_rows


def train_fold(context: dict, fold: int, output_root: Path) -> tuple[dict, list[dict], list[dict]]:
    final_dir = output_root / f"fold_{fold:02d}"
    loaded = _load_completed_fold(final_dir, fold, context)
    if loaded is not None:
        return loaded
    staging_dir = output_root / f".fold_{fold:02d}_in_progress"
    require(not staging_dir.exists(), f"Retained staging directory exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    bundle = make_training_objects(context)
    (
        model,
        criterion,
        base_optimizer,
        correction_optimizer,
        base_scheduler,
        correction_scheduler,
        base_parameters,
        correction_parameters,
    ) = bundle
    epoch_rows = []
    best_base = None
    best_corrected = None
    base_best_state = None
    corrected_best_state = None
    max_gradients = {"input_projection": 0.0, "output_ad_smci": 0.0, "output_cn_smci": 0.0}
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        correction_optimizer.zero_grad(set_to_none=True)
        base_raw, corrected_raw, branches, auxiliary, intermediates = model(features, return_intermediates=True)
        base_loss = criterion(base_raw, labels, train_mask, branches, auxiliary)
        correction_loss, correction_detail = pair_loss(model, intermediates, labels, train_mask)
        require(bool(torch.isfinite(base_loss)) and bool(torch.isfinite(correction_loss)), f"fold{fold} epoch{epoch}: non-finite loss")
        base_loss.backward()
        correction_loss.backward()
        require(grads_finite(base_parameters) and grads_finite(correction_parameters), f"fold{fold} epoch{epoch}: non-finite gradient")
        input_gradient = grad_norm(model.gi_pbf.input_projection.parameters())
        row_gradient = model.gi_pbf.output.weight.grad.detach().norm(dim=1).cpu().tolist()
        max_gradients["input_projection"] = max(max_gradients["input_projection"], input_gradient)
        max_gradients["output_ad_smci"] = max(max_gradients["output_ad_smci"], row_gradient[0])
        max_gradients["output_cn_smci"] = max(max_gradients["output_cn_smci"], row_gradient[1])
        torch.nn.utils.clip_grad_norm_(base_parameters, float(context["config"].grad_clip))
        torch.nn.utils.clip_grad_norm_(correction_parameters, float(context["config"].grad_clip))
        base_optimizer.step()
        correction_optimizer.step()
        base_scheduler.step()
        correction_scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_base, eval_corrected, _, _, eval_intermediates = model(features, return_intermediates=True)
            base_metrics, base_tuple = metric_and_tuple(eval_base, labels, test_mask, context)
            corrected_metrics, corrected_tuple = metric_and_tuple(eval_corrected, labels, test_mask, context)
        if best_base is None or base_tuple > best_base["selection_tuple"]:
            best_base = {"epoch": epoch, "selection_tuple": base_tuple, "metrics": deepcopy(base_metrics)}
            base_best_state = clone_cpu_state(model)
        if best_corrected is None or corrected_tuple > best_corrected["selection_tuple"]:
            best_corrected = {"epoch": epoch, "selection_tuple": corrected_tuple, "metrics": deepcopy(corrected_metrics)}
            corrected_best_state = clone_cpu_state(model)
        epoch_rows.append(
            {
                "epoch": epoch,
                "base_lr": float(base_optimizer.param_groups[0]["lr"]),
                "correction_lr": float(correction_optimizer.param_groups[0]["lr"]),
                "base_loss": float(base_loss.detach().cpu()),
                "correction_loss": float(correction_loss.detach().cpu()),
                "loss_ad_smci": float(correction_detail["loss_ad_smci"].detach().cpu()),
                "loss_cn_smci": float(correction_detail["loss_cn_smci"].detach().cpu()),
                "input_projection_gradient": input_gradient,
                "output_ad_smci_gradient": row_gradient[0],
                "output_cn_smci_gradient": row_gradient[1],
                **{f"base_{key}": base_metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
                **{f"corrected_{key}": corrected_metrics[key] for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")},
            }
        )
    require(best_base is not None and best_corrected is not None, f"fold{fold}: best state missing")
    require(all(value > 0.0 and math.isfinite(value) for value in max_gradients.values()), f"fold{fold}: correction head inactive")

    model.load_state_dict(base_best_state, strict=True)
    model.eval()
    with torch.no_grad():
        base_best_raw, _, _, _, _ = model(features, return_intermediates=True)
    base_rows = prediction_rows(fold, base_best_raw, labels, test_mask, context)
    require(cme.metrics_from_rows(base_rows) == best_base["metrics"], f"fold{fold}: base best readback mismatch")

    model.load_state_dict(corrected_best_state, strict=True)
    model.eval()
    with torch.no_grad():
        _, corrected_best_raw, _, _, corrected_intermediates = model(features, return_intermediates=True)
    corrected_rows = prediction_rows(fold, corrected_best_raw, labels, test_mask, context, corrected_intermediates)
    require(cme.metrics_from_rows(corrected_rows) == best_corrected["metrics"], f"fold{fold}: corrected best readback mismatch")
    diagnostics = diagnostics_from_intermediates(corrected_intermediates, test_mask)
    fold_reference_rows = [row for row in context["c1_rows"] if int(row["fold"]) == fold]
    fold_reference = {int(row["subject_index"]): row for row in fold_reference_rows}
    comparison = paired_comparison(corrected_rows, fold_reference)
    corrected_boundary = boundary_errors(best_corrected["metrics"], context["class_mapping"])
    c1_fold_metrics = cme.metrics_from_rows(fold_reference_rows)
    c1_boundary = boundary_errors(c1_fold_metrics, context["class_mapping"])
    elapsed = time.perf_counter() - started
    summary = {
        "passed": True,
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "parameter_count": parameter_count(model),
        "base_best": {"epoch": best_base["epoch"], "correct": best_base["metrics"]["correct"], "metrics": best_base["metrics"]},
        "corrected_best": {"epoch": best_corrected["epoch"], "correct": best_corrected["metrics"]["correct"], "metrics": best_corrected["metrics"]},
        "correction_gradient_max": max_gradients,
        "diagnostics": diagnostics,
        "comparison_vs_c1": comparison,
        "boundary_errors": corrected_boundary,
        "c1_boundary_errors": c1_boundary,
        "elapsed_seconds": elapsed,
        "config": config_payload(context, "fold", [fold]),
    }
    torch.save(
        {
            "base_best_epoch": best_base["epoch"],
            "corrected_best_epoch": best_corrected["epoch"],
            "base_best_state": base_best_state,
            "corrected_best_state": corrected_best_state,
            "config": summary["config"],
        },
        staging_dir / "checkpoint_best.pt",
    )
    write_json(staging_dir / "summary.json", summary)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(staging_dir / "base_best_predictions.csv", base_rows)
    write_csv(staging_dir / "corrected_best_predictions.csv", corrected_rows)
    class_names = class_names_in_index_order(context["class_mapping"])
    write_csv(staging_dir / "base_confusion_matrix.csv", _confusion_rows(best_base["metrics"]["confusion_matrix"], class_names))
    write_csv(staging_dir / "corrected_confusion_matrix.csv", _confusion_rows(best_corrected["metrics"]["confusion_matrix"], class_names))
    staging_dir.rename(final_dir)
    print(
        f"fold={fold} base={best_base['metrics']['correct']}@{best_base['epoch']} "
        f"corrected={best_corrected['metrics']['correct']}@{best_corrected['epoch']}",
        flush=True,
    )
    del model, criterion, base_optimizer, correction_optimizer, base_scheduler, correction_scheduler
    torch.cuda.empty_cache()
    return summary, base_rows, corrected_rows


def _confusion_rows(matrix, class_names) -> list[dict]:
    return [
        {"truth_class": class_names[index], **dict(zip(class_names, row))}
        for index, row in enumerate(matrix)
    ]


def paired_comparison(candidate_rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in candidate_rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"Reference missing subject {subject}")
        reference = reference_by_subject[subject]
        truth = int(row["truth"])
        require(int(reference["truth"]) == truth and int(reference["fold"]) == int(row["fold"]), "Reference alignment mismatch")
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


def boundary_errors(metrics: dict, mapping: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    ad = int(mapping["AD"])
    cn = int(mapping["CN"])
    smci = int(mapping["sMCI"])
    return {
        "AD_SMCI": int(matrix[ad, smci] + matrix[smci, ad]),
        "CN_SMCI": int(matrix[cn, smci] + matrix[smci, cn]),
        "AD_CN": int(matrix[ad, cn] + matrix[cn, ad]),
    }


def validate_unique(rows: list[dict], expected_count: int, expected_folds) -> list[dict]:
    require(len(rows) == expected_count, f"OOF row count mismatch: {len(rows)}")
    subjects = [int(row["subject_index"]) for row in rows]
    require(len(set(subjects)) == expected_count, "OOF duplicate subject")
    require(set(int(row["fold"]) for row in rows) == set(expected_folds), "OOF fold set mismatch")
    return sorted(rows, key=lambda row: int(row["subject_index"]))


def write_oof_views(root: Path, prefix: str, rows: list[dict]) -> None:
    write_csv(root / f"{prefix}_oof_predictions.csv", rows)
    write_csv(
        root / f"{prefix}_oof_logits.csv",
        [
            {
                "fold": row["fold"],
                "subject_index": row["subject_index"],
                "truth": row["truth"],
                **{f"raw_logit_{name}": row[f"raw_logit_{name}"] for name in CLASS_NAMES},
                **{f"adjusted_score_{name}": row[f"adjusted_score_{name}"] for name in CLASS_NAMES},
            }
            for row in rows
        ],
    )
    write_csv(
        root / f"{prefix}_oof_probabilities.csv",
        [
            {
                "fold": row["fold"],
                "subject_index": row["subject_index"],
                "truth": row["truth"],
                **{f"probability_{name}": row[f"probability_{name}"] for name in CLASS_NAMES},
            }
            for row in rows
        ],
    )


def aggregate_diagnostics(rows: list[dict], summaries: list[dict]) -> dict:
    output = {}
    for name in ("ad_smci", "cn_smci"):
        raw = np.asarray([float(row[f"raw_delta_{name}"]) for row in rows])
        delta = np.asarray([float(row[f"delta_{name}"]) for row in rows])
        uncertainty = np.asarray([float(row[f"uncertainty_{name}"]) for row in rows])
        output[name] = {
            "raw_delta_mean": float(raw.mean()),
            "raw_delta_abs_mean": float(np.abs(raw).mean()),
            "raw_delta_abs_max": float(np.abs(raw).max()),
            "raw_delta_std": float(raw.std(ddof=0)),
            "delta_mean": float(delta.mean()),
            "delta_abs_mean": float(np.abs(delta).mean()),
            "delta_abs_max": float(np.abs(delta).max()),
            "delta_std": float(delta.std(ddof=0)),
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_min": float(uncertainty.min()),
            "uncertainty_max": float(uncertainty.max()),
            "tanh_saturation_fraction": float((np.abs(np.tanh(raw)) >= 0.99).mean()),
            "gradient_max": float(max(summary["correction_gradient_max"][f"output_{name}"] for summary in summaries)),
        }
    output["input_projection_gradient_max"] = float(max(summary["correction_gradient_max"]["input_projection"] for summary in summaries))
    return output


def reference_subset(context: dict, folds) -> tuple[list[dict], dict]:
    rows = [row for row in context["c1_rows"] if int(row["fold"]) in set(folds)]
    return rows, {int(row["subject_index"]): row for row in rows}


def base_reproduction(base_rows: list[dict], summaries: list[dict], reference_rows: list[dict]) -> dict:
    base_by_subject = {int(row["subject_index"]): row for row in base_rows}
    max_probability_diff = 0.0
    prediction_equal = True
    for reference in reference_rows:
        subject = int(reference["subject_index"])
        require(subject in base_by_subject, "Base reproduction subject missing")
        base = base_by_subject[subject]
        require(int(base["truth"]) == int(reference["truth"]) and int(base["fold"]) == int(reference["fold"]), "Base/reference mismatch")
        prediction_equal &= int(base["prediction"]) == int(reference["prediction"])
        for name in CLASS_NAMES:
            max_probability_diff = max(max_probability_diff, abs(float(base[f"probability_{name}"]) - float(reference[f"probability_{name}"])))
    expected_epochs = {fold: SCREEN_C1["fold_best_epoch"][fold] for fold in SCREEN_FOLDS} if set(int(row["fold"]) for row in reference_rows) == set(SCREEN_FOLDS) else None
    epoch_equal = True if expected_epochs is None else all(int(summary["base_best"]["epoch"]) == expected_epochs[int(summary["fold"])] for summary in summaries)
    metrics_equal = cme.metrics_from_rows(base_rows) == cme.metrics_from_rows(reference_rows)
    return {
        "passed": bool(prediction_equal and epoch_equal and metrics_equal and max_probability_diff <= 1e-6),
        "prediction_equal": bool(prediction_equal),
        "best_epoch_equal": bool(epoch_equal),
        "metrics_equal": bool(metrics_equal),
        "probability_max_abs_diff": max_probability_diff,
    }


def fold_metrics_rows(summaries: list[dict]) -> list[dict]:
    return [
        {
            "fold": summary["fold"],
            "base_best_epoch": summary["base_best"]["epoch"],
            "base_correct": summary["base_best"]["correct"],
            "base_acc": summary["base_best"]["metrics"]["acc"],
            "corrected_best_epoch": summary["corrected_best"]["epoch"],
            "corrected_correct": summary["corrected_best"]["correct"],
            "corrected_acc": summary["corrected_best"]["metrics"]["acc"],
            "corrected_macro_f1": summary["corrected_best"]["metrics"]["macro_f1"],
            "corrected_bacc": summary["corrected_best"]["metrics"]["bacc"],
            "corrected_macro_auc": summary["corrected_best"]["metrics"]["macro_auc"],
            "corrected_weighted_f1": summary["corrected_best"]["metrics"]["weighted_f1"],
            "repairs_vs_c1": summary["comparison_vs_c1"]["repairs"],
            "damages_vs_c1": summary["comparison_vs_c1"]["damages"],
            "changed_vs_c1": summary["comparison_vs_c1"]["changed_predictions"],
            "ad_smci_errors": summary["boundary_errors"]["AD_SMCI"],
            "cn_smci_errors": summary["boundary_errors"]["CN_SMCI"],
            "ad_cn_errors": summary["boundary_errors"]["AD_CN"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for summary in summaries
    ]


def run_folds(context: dict, folds, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    root.mkdir(parents=True, exist_ok=True)
    summaries, base_rows, corrected_rows = [], [], []
    for fold in folds:
        summary, base, corrected = train_fold(context, fold, root)
        summaries.append(summary)
        base_rows.extend(base)
        corrected_rows.extend(corrected)
    expected = sum(int(context["dataset_data"]["Mask"][fold][1].sum()) for fold in folds)
    base_rows = validate_unique(base_rows, expected, folds)
    corrected_rows = validate_unique(corrected_rows, expected, folds)
    require([int(row["subject_index"]) for row in base_rows] == [int(row["subject_index"]) for row in corrected_rows], "Base/corrected OOF subjects differ")
    return summaries, base_rows, corrected_rows


def screen_decision(metrics, paired, boundary, diagnostics, reproduction) -> tuple[str, dict]:
    checks = {
        "base_trajectory_reproduced": reproduction["passed"],
        "correct_at_least_227": metrics["correct"] >= 227,
        "repairs_above_damages": paired["repairs"] > paired["damages"],
        "no_ad_cn_errors": boundary["AD_CN"] == 0,
        "ad_smci_not_above_c1_8": boundary["AD_SMCI"] <= SCREEN_C1["AD_SMCI"],
        "cn_smci_not_above_c1_6": boundary["CN_SMCI"] <= SCREEN_C1["CN_SMCI"],
        "macro_f1_within_0p005_of_c1": metrics["macro_f1"] >= SCREEN_C1["macro_f1"] - 0.005,
        "bacc_within_0p005_of_c1": metrics["bacc"] >= SCREEN_C1["bacc"] - 0.005,
        "changed_at_least_2": paired["changed_predictions"] >= 2,
        "both_heads_trained_nonconstant": all(
            diagnostics[name]["gradient_max"] > 1e-8
            and diagnostics[name]["raw_delta_std"] > 1e-8
            and diagnostics[name]["delta_std"] > 1e-8
            and diagnostics[name]["tanh_saturation_fraction"] < 0.95
            for name in ("ad_smci", "cn_smci")
        ),
    }
    if not checks["base_trajectory_reproduced"]:
        return "GI_PBF_IMPLEMENTATION_INVALID", checks
    return ("SCREEN_GO" if all(checks.values()) else "SCREEN_STOP"), checks


def render_report(report: dict) -> str:
    metrics = report["corrected_metrics"]
    lines = [
        f"# GI-PBF v1 {report['stage'].title()} Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"- Actual C1 base commit: `{ACTUAL_BASE}`",
        f"- Corrected: {metrics['correct']}/{report['subject_count']}",
        f"- ACC / Macro-F1 / BACC / AUC / Weighted-F1: {metrics['acc']:.7f} / {metrics['macro_f1']:.7f} / {metrics['bacc']:.7f} / {metrics['macro_auc']:.7f} / {metrics['weighted_f1']:.7f}",
        f"- Confusion: {metrics['confusion_matrix']}",
        f"- Base trajectory reproduced: {report['base_reproduction']['passed']}",
        f"- Repairs / damages / changed vs C1: {report['comparison_vs_c1']['repairs']} / {report['comparison_vs_c1']['damages']} / {report['comparison_vs_c1']['changed_predictions']}",
        f"- AD-sMCI / CN-sMCI / AD-CN errors: {report['boundary_errors']['AD_SMCI']} / {report['boundary_errors']['CN_SMCI']} / {report['boundary_errors']['AD_CN']}",
        f"- Parameters: {report['parameter_count']}",
        f"- Training seconds: {report['training_seconds']:.3f}",
        f"- Reproduction: `{report['run_command']}`",
    ]
    if "comparison_vs_pc_bbf_v1" in report:
        paired_pc = report["comparison_vs_pc_bbf_v1"]
        lines.append(
            f"- Repairs / damages / changed vs PC-BBF v1: {paired_pc['repairs']} / {paired_pc['damages']} / {paired_pc['changed_predictions']}",
        )
    if "fold_metric_mean_sample_std" in report:
        lines.extend(["", "## Ten-fold metric mean +/- sample SD", ""])
        for key, values in report["fold_metric_mean_sample_std"].items():
            lines.append(f"- {key}: {values['mean']:.10f} +/- {values['sample_std']:.10f}")
    lines.extend(["", "## Fold best epochs", ""])
    for fold in report["folds"]:
        lines.append(
            f"- fold{fold['fold']}: base {fold['base_correct']}@{fold['base_best_epoch']}; corrected {fold['corrected_correct']}@{fold['corrected_best_epoch']}"
        )
    lines.extend(["", "## Correction diagnostics", "", json.dumps(report["correction_diagnostics"], ensure_ascii=False, indent=2)])
    if report.get("next_recommendation"):
        lines.extend(["", f"Next recommendation: {report['next_recommendation']} (not implemented or run)."])
    return "\n".join(lines) + "\n"


def write_aggregate_outputs(root: Path, summaries, base_rows, corrected_rows, report) -> None:
    write_json(root / "config.json", report["config"])
    write_json(root / "report.json", report)
    (root / "report.md").write_text(render_report(report), encoding="utf-8")
    write_oof_views(root, "base", base_rows)
    write_oof_views(root, "corrected", corrected_rows)
    write_csv(root / "fold_metrics.csv", fold_metrics_rows(summaries))
    class_names = class_names_in_index_order(report["config"]["class_mapping"])
    write_csv(root / "base_confusion_matrix.csv", _confusion_rows(report["base_metrics"]["confusion_matrix"], class_names))
    write_csv(root / "corrected_confusion_matrix.csv", _confusion_rows(report["corrected_metrics"]["confusion_matrix"], class_names))
    write_json(root / "repairs_damages.json", report["comparison_vs_c1"])


def require_committed_unchanged(paths: list[Path], label: str) -> None:
    relative_paths = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    tracked = git("ls-files", "--error-unmatch", "--", *relative_paths, check=False)
    require(tracked.returncode == 0, f"{label} is not committed")
    changed = git("diff", "--quiet", "HEAD", "--", *relative_paths, check=False)
    require(changed.returncode == 0, f"{label} differs from HEAD")


def prepare_resumable_stage(root: Path, report_name: str = "report.json") -> None:
    require(not (root / report_name).exists(), f"Refusing to overwrite completed stage: {root}")
    root.mkdir(parents=True, exist_ok=True)


def run_screen(context: dict, output_root: Path) -> dict:
    screen_root = output_root / "screen"
    smoke_path = output_root / "smoke" / "smoke_report.json"
    smoke_config_path = output_root / "config.json"
    require(smoke_path.is_file(), "Smoke report missing")
    require(smoke_config_path.is_file(), "Smoke config missing")
    smoke_report = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke_report.get("passed") is True, "Smoke did not pass")
    require(
        json.loads(smoke_config_path.read_text(encoding="utf-8"))
        == config_payload(context, "smoke", [0]),
        "Smoke configuration changed",
    )
    require_committed_unchanged(
        [
            ROOT / "Model" / "gi_pbf.py",
            ROOT / "scripts" / "run_gi_pbf_v1.py",
            context["historical_config_path"],
            smoke_path,
            smoke_config_path,
        ],
        "GI-PBF implementation and smoke",
    )
    prepare_resumable_stage(screen_root)
    started = time.perf_counter()
    summaries, base_rows, corrected_rows = run_folds(context, SCREEN_FOLDS, screen_root)
    reference_rows, reference_by_subject = reference_subset(context, SCREEN_FOLDS)
    base_metrics = cme.metrics_from_rows(base_rows)
    corrected_metrics = cme.metrics_from_rows(corrected_rows)
    reference_metrics = cme.metrics_from_rows(reference_rows)
    require(reference_metrics["correct"] == SCREEN_C1["correct"], "Screen C1 anchor changed")
    reproduction = base_reproduction(base_rows, summaries, reference_rows)
    paired = paired_comparison(corrected_rows, reference_by_subject)
    boundary = boundary_errors(corrected_metrics, context["class_mapping"])
    diagnostics = aggregate_diagnostics(corrected_rows, summaries)
    decision, checks = screen_decision(corrected_metrics, paired, boundary, diagnostics, reproduction)
    fold_rows = fold_metrics_rows(summaries)
    report = {
        "stage": "screen",
        "decision": decision,
        "decision_checks": checks,
        "subject_count": len(corrected_rows),
        "folds_used": list(SCREEN_FOLDS),
        "base_metrics": base_metrics,
        "corrected_metrics": corrected_metrics,
        "c1_reference_metrics": reference_metrics,
        "base_reproduction": reproduction,
        "comparison_vs_c1": paired,
        "boundary_errors": boundary,
        "c1_boundary_errors": {key: SCREEN_C1[key] for key in ("AD_SMCI", "CN_SMCI", "AD_CN")},
        "correction_diagnostics": diagnostics,
        "parameter_count": GI_PBF_EXPECTED_PARAMETERS_H96_R8,
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "wall_seconds": time.perf_counter() - started,
        "folds": fold_rows,
        "config": config_payload(context, "screen", SCREEN_FOLDS),
        "run_head": git("rev-parse", "HEAD").stdout.strip(),
        "run_command": reproduction_command("screen"),
        "formal_authorized": decision == "SCREEN_GO",
        "next_recommendation": None if decision == "SCREEN_GO" else "MR-LHGR: modality-specific multi-relation low/high-frequency graph residual",
        "next_experiment_implemented_or_run": False,
    }
    write_aggregate_outputs(screen_root, summaries, base_rows, corrected_rows, report)
    print(f"{decision} correct={corrected_metrics['correct']}/240", flush=True)
    return report


def load_pc_bbf_rows() -> list[dict]:
    spec = f"{PC_BBF_REF}:{PC_BBF_OOF_REL}"
    result = git("show", spec)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    require(len(rows) == 598 and len({int(row['subject_index']) for row in rows}) == 598, "PC-BBF v1 OOF invalid")
    metrics = cme.metrics_from_rows(rows)
    require(metrics["correct"] == 561, "PC-BBF v1 anchor changed")
    return rows


def formal_decision(metrics, paired, boundary) -> tuple[str, dict]:
    no_gain = {
        "correct_at_most_560": metrics["correct"] <= 560,
        "repairs_not_above_damages": paired["repairs"] <= paired["damages"],
        "ad_smci_above_c1": boundary["AD_SMCI"] > 21,
        "cn_smci_above_c1": boundary["CN_SMCI"] > 17,
    }
    target = {
        "correct_at_least_563": metrics["correct"] >= 563,
        "macro_f1_at_least_c1": metrics["macro_f1"] >= C1_REFERENCE["macro_f1"],
        "bacc_within_0p002_of_c1": metrics["bacc"] >= C1_REFERENCE["bacc"] - 0.002,
        "no_ad_cn_errors": boundary["AD_CN"] == 0,
    }
    positive = {
        "correct_equals_562": metrics["correct"] == 562,
        "repairs_above_damages": paired["repairs"] > paired["damages"],
        "macro_f1_at_least_c1": metrics["macro_f1"] >= C1_REFERENCE["macro_f1"],
        "bacc_at_least_c1": metrics["bacc"] >= C1_REFERENCE["bacc"],
    }
    stable = {
        "correct_equals_561": metrics["correct"] == 561,
        "ad_smci_not_above_c1": boundary["AD_SMCI"] <= 21,
        "cn_smci_not_above_c1": boundary["CN_SMCI"] <= 17,
        "macro_f1_improved": metrics["macro_f1"] > C1_REFERENCE["macro_f1"],
        "macro_auc_improved": metrics["macro_auc"] > C1_REFERENCE["macro_auc"],
    }
    checks = {"no_gain": no_gain, "target": target, "positive": positive, "stable": stable}
    if any(no_gain.values()):
        return "GI_PBF_NO_GAIN", checks
    if all(target.values()):
        return "GI_PBF_TARGET_REACHED", checks
    if all(positive.values()):
        return "GI_PBF_POSITIVE", checks
    if all(stable.values()):
        return "GI_PBF_STABLE", checks
    return "GI_PBF_NO_GAIN", checks


def run_formal(context: dict, output_root: Path) -> dict:
    smoke_path = output_root / "smoke" / "smoke_report.json"
    smoke_config_path = output_root / "config.json"
    require(smoke_path.is_file(), "Smoke report missing")
    require(smoke_config_path.is_file(), "Smoke config missing")
    smoke_report = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke_report.get("passed") is True, "Smoke did not pass")
    require(
        json.loads(smoke_config_path.read_text(encoding="utf-8"))
        == config_payload(context, "smoke", [0]),
        "Smoke configuration changed",
    )
    screen_report_path = output_root / "screen" / "report.json"
    require(screen_report_path.is_file(), "Screen report missing")
    screen_report = json.loads(screen_report_path.read_text(encoding="utf-8"))
    require(screen_report["decision"] == "SCREEN_GO", "Formal forbidden without SCREEN_GO")
    require(
        screen_report.get("config") == config_payload(context, "screen", SCREEN_FOLDS),
        "Screen configuration changed",
    )
    screen_head = str(screen_report.get("run_head", ""))
    require(bool(screen_head), "Screen implementation commit missing")
    require(git("cat-file", "-e", f"{screen_head}^{{commit}}", check=False).returncode == 0, "Screen implementation commit unavailable")
    source_diff = git(
        "diff",
        "--quiet",
        screen_head,
        "HEAD",
        "--",
        "Model/gi_pbf.py",
        "scripts/run_gi_pbf_v1.py",
        check=False,
    )
    require(source_diff.returncode == 0, "GI-PBF source changed after screen")
    require_committed_unchanged(
        [
            ROOT / "Model" / "gi_pbf.py",
            ROOT / "scripts" / "run_gi_pbf_v1.py",
            context["historical_config_path"],
            smoke_path,
            smoke_config_path,
        ],
        "GI-PBF locked implementation and smoke",
    )
    formal_root = output_root / "formal"
    prepare_resumable_stage(formal_root)
    started = time.perf_counter()
    summaries, base_rows, corrected_rows = run_folds(context, FORMAL_FOLDS, formal_root)
    reference_rows = context["c1_rows"]
    reference_by_subject = context["c1_by_subject"]
    base_metrics = cme.metrics_from_rows(base_rows)
    corrected_metrics = cme.metrics_from_rows(corrected_rows)
    reproduction = base_reproduction(base_rows, summaries, reference_rows)
    reproduction["correct_560"] = base_metrics["correct"] == 560
    reproduction["formal_metrics_match"] = base_metrics == cme.metrics_from_rows(reference_rows)
    reproduction["passed"] = bool(reproduction["passed"] and reproduction["correct_560"] and reproduction["formal_metrics_match"])
    paired_c1 = paired_comparison(corrected_rows, reference_by_subject)
    pc_rows = load_pc_bbf_rows()
    paired_pc = paired_comparison(corrected_rows, {int(row["subject_index"]): row for row in pc_rows})
    boundary = boundary_errors(corrected_metrics, context["class_mapping"])
    diagnostics = aggregate_diagnostics(corrected_rows, summaries)
    if not reproduction["passed"]:
        decision, checks = "GI_PBF_IMPLEMENTATION_INVALID", {"base_reproduction": reproduction}
    else:
        decision, checks = formal_decision(corrected_metrics, paired_c1, boundary)
    fold_metric_mean_sample_std = {}
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        values = np.asarray(
            [summary["corrected_best"]["metrics"][key] for summary in summaries],
            dtype=np.float64,
        )
        fold_metric_mean_sample_std[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
        }
    report = {
        "stage": "formal",
        "decision": decision,
        "decision_checks": checks,
        "subject_count": 598,
        "base_metrics": base_metrics,
        "corrected_metrics": corrected_metrics,
        "base_reproduction": reproduction,
        "comparison_vs_c1": paired_c1,
        "comparison_vs_pc_bbf_v1": paired_pc,
        "boundary_errors": boundary,
        "correction_diagnostics": diagnostics,
        "fold_metric_mean_sample_std": fold_metric_mean_sample_std,
        "fold_acc_mean": fold_metric_mean_sample_std["acc"]["mean"],
        "fold_acc_sample_std": fold_metric_mean_sample_std["acc"]["sample_std"],
        "parameter_count": GI_PBF_EXPECTED_PARAMETERS_H96_R8,
        "training_seconds": float(sum(summary["elapsed_seconds"] for summary in summaries)),
        "wall_seconds": time.perf_counter() - started,
        "folds": fold_metrics_rows(summaries),
        "config": config_payload(context, "formal", FORMAL_FOLDS),
        "run_head": git("rev-parse", "HEAD").stdout.strip(),
        "run_command": reproduction_command("formal"),
        "next_recommendation": None if decision == "GI_PBF_TARGET_REACHED" else "MR-LHGR: modality-specific multi-relation low/high-frequency graph residual",
        "next_experiment_implemented_or_run": False,
    }
    write_aggregate_outputs(formal_root, summaries, base_rows, corrected_rows, report)
    write_json(formal_root / "comparison_vs_pc_bbf_v1.json", paired_pc)
    print(f"{decision} correct={corrected_metrics['correct']}/598", flush=True)
    return report


def verify_base_and_branch() -> None:
    require(git("cat-file", "-e", f"{ACTUAL_BASE}^{{commit}}").returncode == 0, "Actual base missing")
    ancestor = git("merge-base", "--is-ancestor", ACTUAL_BASE, "HEAD", check=False)
    require(ancestor.returncode == 0, "Current HEAD is not based on formal C1")
    branch = git("branch", "--show-current").stdout.strip()
    require(branch == "experiment/gi-pbf-v1", f"Wrong branch: {branch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "screen", "formal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_base_and_branch()
    context = load_context(args.device)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"GI-PBF actual C1 base={ACTUAL_BASE}", flush=True)
    if args.stage == "smoke":
        report = run_smoke(context, output_root)
    elif args.stage == "screen":
        report = run_screen(context, output_root)
    else:
        report = run_formal(context, output_root)
    print(f"{args.stage.upper()} {report.get('decision', 'PASS')}", flush=True)


if __name__ == "__main__":
    main()
