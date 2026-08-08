"""Run CBTR-C1 v1 and the single preregistered cap-0.15 follow-up.

``smoke`` performs only the prescribed fold-0 CUDA checks. ``formal`` runs
fresh folds 0..9 at cap 0.10 and runs v1.1 only when every locked gate passes.
There is no screen or hyperparameter-search entry point.
"""

from __future__ import annotations

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Loss import criterion_query_pool_no_orth
from Model.cbtr_c1 import CBTRC1Model
from Utils import CustomCosineAnnealingLR, ModelEMA, SET_Random
import run_cme_dual_branch_v1 as cme


EXPERIMENT = "cbtr_c1_v1"
OUTPUT_REL = Path("experiments/cbtr_c1_v1")
BASE_COMMIT = "90326eef6ab1a8a4111e71a33280f8f7113c1ea7"
C1_OOF_REL = Path(
    "experiments/cme_dual_branch_v1/c1_shared_private_control_2/oof_predictions.csv"
)
FOLDS = tuple(range(10))
SEED = 0
EPOCHS = 400
CLASS_NAMES = ("AD", "CN", "SMCI")
KEY_DIM = VALUE_DIM = 16
PER_CLASS_K = 4
TEMPERATURE = 0.2
C1_PARAMETERS = 862_971
CBTR_PARAMETERS = 3_376
EXPECTED_PARAMETERS = 866_347

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
    "weighted_f1": 0.9378308,
}
ORIGINAL_BACC = 0.9140778


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


def source_commit() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def require_committed_source() -> tuple[str, str]:
    tracked = ("Model/cbtr_c1.py", "scripts/run_cbtr_c1_v1.py")
    for path in tracked:
        require(
            git("ls-files", "--error-unmatch", path, check=False).returncode == 0,
            f"Source must be committed before smoke: {path}",
        )
    last_touches = [
        git("log", "-1", "--format=%H", "--", path).stdout.strip()
        for path in tracked
    ]
    require(all(last_touches), "CBTR implementation commit is missing")
    require(
        len(set(last_touches)) == 1,
        "Model and runner must belong to one implementation source commit",
    )
    implementation = last_touches[0]
    require(implementation != BASE_COMMIT, "CBTR implementation source commit is missing")
    require(
        git("diff", "--quiet", implementation, "--", *tracked, check=False).returncode == 0,
        "CBTR source differs from the implementation commit",
    )
    run_head = source_commit()
    require(
        git("merge-base", "--is-ancestor", implementation, run_head, check=False).returncode == 0,
        "Current HEAD does not descend from the CBTR implementation commit",
    )
    historical_config = cme.CONFIG_REL.as_posix()
    require(
        git("ls-files", "--error-unmatch", historical_config, check=False).returncode == 0,
        "Historical C1 config is not tracked",
    )
    require(
        git(
            "diff",
            "--quiet",
            BASE_COMMIT,
            "--",
            historical_config,
            check=False,
        ).returncode
        == 0,
        "Historical C1 config differs from the locked base commit",
    )
    return implementation, run_head


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def clone_cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def clone_ema_state(ema: ModelEMA) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in ema.shadow.items()
    }


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
    cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None
    )
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
            for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1")
            if key in candidate and key in baseline
        },
    }


def expected_oof_alignment(context: dict) -> dict[int, tuple[int, int]]:
    source_indices = np.asarray(context["dataset_dict"]["Index"], dtype=np.int64)
    labels = context["dataset_data"]["Label"].detach().cpu().numpy().astype(np.int64)
    expected = {}
    for fold, (_, test_mask) in enumerate(context["dataset_data"]["Mask"]):
        internal = test_mask.detach().cpu().numpy().astype(bool)
        for subject, truth in zip(source_indices[internal], labels[internal]):
            require(int(subject) not in expected, "Fold test masks overlap")
            expected[int(subject)] = (fold, int(truth))
    require(len(expected) == 598, "Fold test masks do not cover 598 unique subjects")
    return expected


def validate_reference_rows(rows: list[dict], expected: dict, label: str) -> dict:
    require(len(rows) == 598, f"{label} OOF row count changed")
    by_subject = {}
    for row in rows:
        subject = int(row["subject_index"])
        require(subject not in by_subject and subject in expected, f"{label} subject alignment failed")
        fold, truth = expected[subject]
        require(int(row["fold"]) == fold, f"{label} fold mismatch for subject {subject}")
        require(int(row["truth"]) == truth, f"{label} truth mismatch for subject {subject}")
        by_subject[subject] = row
    return by_subject


def read_git_csv(ref: str, path: str) -> list[dict] | None:
    result = git("show", f"{ref}:{path}", check=False)
    if result.returncode != 0:
        return None
    return list(csv.DictReader(io.StringIO(result.stdout)))


def load_context(device_text: str) -> dict:
    require(device_text == "cuda:0", "CBTR-C1 requires --device cuda:0")
    context = cme.load_context()
    require(str(context["device"]) == "cuda:0", "Historical device changed")
    require(tuple(context["dataset_dict"]["Class_Names"]) == CLASS_NAMES, "Class order changed")
    require(tuple(context["dataset_data"]["Feature"].shape) == (598, 360), "Dataset changed")
    require(len(context["dataset_data"]["Mask"]) == 10, "Fold count changed")
    require(int(context["config"].epochs) == EPOCHS, "Epoch protocol changed")
    require(int(context["config"].T_max) == EPOCHS, "Scheduler protocol changed")
    require(context["config"].use_ema is False, "Historical C1 EMA setting changed")
    expected = expected_oof_alignment(context)
    c1_path = ROOT / C1_OOF_REL
    c1_rows = read_csv(c1_path) if c1_path.is_file() else None
    c1_by_subject = None
    if c1_rows is not None:
        c1_by_subject = validate_reference_rows(c1_rows, expected, "C1")
        c1_metrics = cme.metrics_from_rows(c1_rows)
        for key in ("correct", "confusion_matrix"):
            require(c1_metrics[key] == C1[key], f"C1 anchor mismatch: {key}")
        for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
            require(abs(c1_metrics[key] - C1[key]) <= 5e-7, f"C1 anchor mismatch: {key}")

    pc_rows = read_git_csv(
        "origin/experiment/pc-bbf-c1-v1",
        "experiments/pc_bbf_c1_v1/formal/oof_predictions.csv",
    )
    pc_by_subject = None
    if pc_rows is not None:
        pc_by_subject = validate_reference_rows(pc_rows, expected, "PC-BBF")
        pc_metrics = cme.metrics_from_rows(pc_rows)
        require(pc_metrics["correct"] == PC_BBF["correct"], "PC-BBF anchor mismatch: correct")
        for key in ("acc", "macro_f1", "bacc", "macro_auc"):
            require(abs(pc_metrics[key] - PC_BBF[key]) <= 5e-7, f"PC-BBF anchor mismatch: {key}")
    context.update(
        {
            "c1_path": c1_path,
            "c1_rows": c1_rows or [],
            "c1_by_subject": c1_by_subject,
            "c1_alignment_available": c1_by_subject is not None,
            "pc_bbf_rows": pc_rows or [],
            "pc_bbf_by_subject": pc_by_subject,
            "pc_bbf_alignment_available": pc_by_subject is not None,
            "expected_oof_alignment": expected,
        }
    )
    return context


def build_model(context: dict, cap: float) -> CBTRC1Model:
    config = context["config"]
    SET_Random(SEED)
    model = CBTRC1Model(
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
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        retrieval_k=PER_CLASS_K,
        retrieval_temperature=TEMPERATURE,
        retrieval_residual_cap=float(cap),
    ).to(context["device"])
    require(parameter_count(model) == EXPECTED_PARAMETERS, "Parameter count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph was enabled")
    return model


def make_training_objects(context: dict, cap: float):
    model = build_model(context, cap)
    criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    base_parameters = list(model.c1_parameters())
    retrieval_parameters = list(model.retrieval_parameters())
    base_ids = {id(parameter) for parameter in base_parameters}
    retrieval_ids = {id(parameter) for parameter in retrieval_parameters}
    all_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    require(not base_ids & retrieval_ids, "Optimizer parameter overlap")
    require(base_ids | retrieval_ids == all_ids, "Optimizer parameter coverage failed")
    require(sum(p.numel() for p in base_parameters) == C1_PARAMETERS, "Base parameters changed")
    require(sum(p.numel() for p in retrieval_parameters) == CBTR_PARAMETERS, "CBTR parameters changed")
    config = context["config"]
    base_optimizer = torch.optim.Adam(
        base_parameters,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    retrieval_optimizer = torch.optim.Adam(
        retrieval_parameters,
        lr=float(config.lr),
        betas=base_optimizer.defaults["betas"],
        eps=base_optimizer.defaults["eps"],
        weight_decay=0.0,
    )
    base_scheduler = CustomCosineAnnealingLR(
        base_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    retrieval_scheduler = CustomCosineAnnealingLR(
        retrieval_optimizer, T_max=EPOCHS, eta_min=float(config.Lr_Min)
    )
    # C1 evaluation historically disables EMA.  A shadow is maintained only
    # to prove that CBTR is covered by checkpoint/recovery; it is never swapped
    # into the model and never participates in selection or OOF inference.
    ema = ModelEMA(model, decay=float(config.ema_decay))
    cbtr_names = {
        name for name, _ in model.named_parameters() if name.startswith("cbtr.")
    }
    require(cbtr_names and cbtr_names <= set(ema.shadow), "CBTR absent from EMA shadow")
    audit = {
        "base_parameter_count": C1_PARAMETERS,
        "retrieval_parameter_count": CBTR_PARAMETERS,
        "total_parameter_count": EXPECTED_PARAMETERS,
        "optimizer_disjoint": True,
        "optimizer_union_complete": True,
        "base_lr": float(config.lr),
        "base_weight_decay": float(config.weight_decay),
        "retrieval_lr": float(config.lr),
        "retrieval_weight_decay": 0.0,
        "historical_ema_enabled": False,
        "ema_shadow_checkpoint_audit_only": True,
        "ema_decay": float(config.ema_decay),
        "cbtr_parameter_names": sorted(cbtr_names),
    }
    return (
        model,
        criterion,
        base_optimizer,
        retrieval_optimizer,
        base_scheduler,
        retrieval_scheduler,
        ema,
        base_parameters,
        retrieval_parameters,
        audit,
    )


def train_memory(context: dict, fold: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask, test_mask = context["dataset_data"]["Mask"][fold]
    labels = context["dataset_data"]["Label"]
    train_idx = train_mask.nonzero(as_tuple=False).flatten()
    train_y = labels[train_idx]
    require(train_idx.numel() + test_mask.sum().item() == labels.numel(), "Fold partition changed")
    require(not bool(test_mask[train_idx].any()), "Test subject entered retrieval memory")
    for class_index in range(3):
        require(int((train_y == class_index).sum()) > PER_CLASS_K, f"Fold {fold} class {class_index} too small")
    return train_mask, test_mask, train_idx, train_y


def config_payload(context: dict, cap: float, version: str, source: str) -> dict:
    config = context["config"]
    return {
        "experiment": EXPERIMENT,
        "version": version,
        "source_commit": source,
        "base_commit": BASE_COMMIT,
        "dataset": "TADPOLE",
        "task": "AD_CN_SMCI",
        "folds": list(FOLDS),
        "seed_per_fold": SEED,
        "epochs_per_fold": EPOCHS,
        "device": "cuda:0",
        "full_batch_transductive": True,
        "single_model": True,
        "ensemble": False,
        "hyperparameter_search": False,
        "orthogonality": False,
        "graph_use_graph": False,
        "adj_mode": "none",
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "per_class_k": PER_CLASS_K,
        "temperature": TEMPERATURE,
        "retrieval_residual_cap": float(cap),
        "retrieval_memory": "fold train_idx and aligned train_y only",
        "loss": "historical C1 weighted main CE plus three OVR losses",
        "optimizers": "historical C1 Adam plus disjoint CBTR Adam(weight_decay=0)",
        "schedulers": "two synchronized CustomCosineAnnealingLR(T_max=400)",
        "gradient_clip": "historical clip on base parameters only; no CBTR clipping",
        "best_epoch_rule": ["ACC", "Probability Macro-AUC", "Macro-F1"],
        "historical_ema_enabled": False,
        "ema_shadow_checkpoint_audit_only": True,
        "historical_config_file": cme.CONFIG_REL.as_posix(),
        "historical_protocol": {
            "hidden_size": int(config.Hidden_size),
            "drop_rate": float(config.Drop_rate),
            "chebgcn_k": int(config.ChebGCN_K),
            "num_layers": int(config.num_layers),
            "num_heads": int(config.num_heads),
            "input_noise_std": float(config.input_noise_std),
            "drop_path": float(config.drop_path),
            "graph_head": str(config.Graph_head),
            "graph_layers": int(config.graph_layers),
            "graph_heads": int(config.graph_heads),
            "graph_beta": float(config.graph_beta),
            "graph_k_order": int(config.graph_k_order),
            "graph_alpha": float(config.graph_alpha),
            "graph_kernel": str(config.graph_kernel),
            "graph_dropout": float(config.graph_dropout),
            "graph_hidden": int(config.graph_hidden),
            "global_word_emb": int(config.global_word_emb),
            "learning_rate": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "eta_min": float(config.Lr_Min),
            "grad_clip": float(config.grad_clip),
            "label_smoothing": 0.05,
            "logit_adjust_tau": float(config.logit_adjust_tau),
            "ema_decay": float(config.ema_decay),
            "use_ema": bool(config.use_ema),
        },
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameter_count": CBTR_PARAMETERS,
    }


def metric_and_tuple(raw, labels, mask, context):
    return cme.selection_metrics(
        raw,
        labels,
        mask,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )


def prediction_rows(fold: int, raw, labels, mask, context) -> list[dict]:
    return cme.prediction_rows(
        fold,
        raw,
        labels,
        mask,
        context["dataset_dict"],
        context["config"],
    )


def paired_comparison(rows: list[dict], reference_by_subject: dict) -> dict:
    repairs, damages, changed = [], [], []
    for row in rows:
        subject = int(row["subject_index"])
        require(subject in reference_by_subject, f"C1 missing subject {subject}")
        reference = reference_by_subject[subject]
        require(int(reference["fold"]) == int(row["fold"]), "C1 fold alignment failed")
        require(int(reference["truth"]) == int(row["truth"]), "C1 truth alignment failed")
        truth = int(row["truth"])
        old, new = int(reference["prediction"]), int(row["prediction"])
        if old != truth and new == truth:
            repairs.append(subject)
        if old == truth and new != truth:
            damages.append(subject)
        if old != new:
            changed.append(subject)
    discordant = len(repairs) + len(damages)
    return {
        "available": True,
        "repairs": len(repairs),
        "damages": len(damages),
        "net_repairs": len(repairs) - len(damages),
        "changed": len(changed),
        "repair_subject_indices": repairs,
        "damage_subject_indices": damages,
        "changed_subject_indices": changed,
        "mcnemar_exact_p": (
            float(cme.binomtest(len(repairs), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
    }


def boundary_errors(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        "AD_SMCI": int(matrix[0, 2] + matrix[2, 0]),
        "CN_SMCI": int(matrix[1, 2] + matrix[2, 1]),
        "AD_CN": int(matrix[0, 1] + matrix[1, 0]),
        "adjacent_total": int(matrix[0, 2] + matrix[2, 0] + matrix[1, 2] + matrix[2, 1]),
    }


def initial_and_first_step_check(context: dict, cap: float) -> dict:
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, _, train_idx, train_y = train_memory(context, 0)

    c1_model = cme.build_model(context, "c1")
    c1_rng = capture_rng()
    cbtr_model = build_model(context, cap)
    cbtr_rng = capture_rng()
    c1_order = [name for name, _ in c1_model.named_parameters()]
    cbtr_base_order = [
        name for name, _ in cbtr_model.named_parameters() if not name.startswith("cbtr.")
    ]
    require(c1_order == cbtr_base_order, "Base parameter names/order changed")
    require(torch.equal(c1_rng[0], cbtr_rng[0]), "CBTR construction changed CPU RNG")
    if c1_rng[1] is not None:
        require(
            all(torch.equal(a, b) for a, b in zip(c1_rng[1], cbtr_rng[1])),
            "CBTR construction changed CUDA RNG",
        )
    cbtr_named = dict(cbtr_model.named_parameters())
    parameter_diff = max(
        float((parameter.detach() - cbtr_named[name].detach()).abs().max().cpu())
        for name, parameter in c1_model.named_parameters()
    )
    require(parameter_diff == 0.0, "Initial C1 parameters changed")
    c1_model.eval()
    cbtr_model.eval()
    with torch.no_grad():
        c1_logits, _, _ = c1_model(features)
        cbtr_logits, _, _, inter = cbtr_model(
            features, train_idx, train_y, return_intermediates=True
        )
    initial_logit_diff = float((c1_logits - cbtr_logits).abs().max().cpu())
    initial_residual = float(inter["R_retrieval"].abs().max().cpu())
    require(initial_logit_diff <= 1e-7, "Initial logits differ from C1")
    require(initial_residual == 0.0, "Initial retrieval residual is not zero")

    # Rebuild both arms fresh and run their first update from the same RNG.
    c1_model = cme.build_model(context, "c1")
    c1_criterion = criterion_query_pool_no_orth(
        context["dataset_dict"], context["device"], label_smoothing=0.05
    )
    c1_optimizer = torch.optim.Adam(
        c1_model.parameters(),
        lr=float(context["config"].lr),
        weight_decay=float(context["config"].weight_decay),
    )
    bundle = make_training_objects(context, cap)
    cbtr_model, cbtr_criterion, base_optimizer, retrieval_optimizer = bundle[:4]
    base_parameters, retrieval_parameters = bundle[7], bundle[8]
    shared_rng = capture_rng()

    restore_rng(shared_rng)
    c1_model.train()
    c1_optimizer.zero_grad(set_to_none=True)
    raw, branches, auxiliary = c1_model(features)
    loss = c1_criterion(raw, labels, train_mask, branches, auxiliary)
    loss.backward()
    c1_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in c1_model.named_parameters()
    }

    restore_rng(shared_rng)
    cbtr_model.train()
    base_optimizer.zero_grad(set_to_none=True)
    retrieval_optimizer.zero_grad(set_to_none=True)
    raw, branches, auxiliary, _ = cbtr_model(
        features, train_idx, train_y, return_intermediates=True
    )
    loss = cbtr_criterion(raw, labels, train_mask, branches, auxiliary)
    loss.backward()
    require(grads_finite(base_parameters + retrieval_parameters), "First-step gradient is non-finite")
    cbtr_named = dict(cbtr_model.named_parameters())
    gradient_differences = []
    for name, reference_gradient in c1_gradients.items():
        candidate_gradient = cbtr_named[name].grad
        require(
            (reference_gradient is None) == (candidate_gradient is None),
            f"First-step base gradient coverage changed: {name}",
        )
        if reference_gradient is not None:
            gradient_differences.append(
                float((reference_gradient - candidate_gradient).abs().max().cpu())
            )
    gradient_diff = max(gradient_differences, default=0.0)
    require(gradient_diff <= 1e-7, "First-step base gradients changed")
    output_gradient = grad_norm([cbtr_model.cbtr.output_projection.weight])
    require(output_gradient > 0.0 and math.isfinite(output_gradient), "W_out gradient missing")

    # The historical clip remains on C1 parameters only. CBTR is not clipped.
    torch.nn.utils.clip_grad_norm_(
        c1_model.parameters(), float(context["config"].grad_clip)
    )
    torch.nn.utils.clip_grad_norm_(
        base_parameters, float(context["config"].grad_clip)
    )
    c1_optimizer.step()
    base_optimizer.step()
    retrieval_optimizer.step()
    cbtr_named = dict(cbtr_model.named_parameters())
    update_diff = max(
        float((parameter.detach() - cbtr_named[name].detach()).abs().max().cpu())
        for name, parameter in c1_model.named_parameters()
    )
    require(update_diff <= 1e-7, "First-step base update changed")
    return {
        "passed": True,
        "initial_c1_parameter_max_abs_diff": parameter_diff,
        "initial_logit_max_abs_diff": initial_logit_diff,
        "initial_residual_max_abs": initial_residual,
        "first_step_base_gradient_max_abs_diff": gradient_diff,
        "first_step_base_parameter_max_abs_diff": update_diff,
        "first_step_w_out_gradient_norm": output_gradient,
        "cpu_cuda_rng_preserved": True,
        "base_only_historical_gradient_clip": True,
    }


def retrieval_audit(
    intermediates: dict,
    train_idx: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict:
    counts = intermediates["candidate_class_counts"]
    candidates = intermediates["candidate_indices"]
    require(
        bool((counts == torch.tensor([4, 4, 4], device=counts.device)).all()),
        "Candidate counts are not 4/4/4",
    )
    require(bool(intermediates["candidate_is_train"].all()), "Non-train memory candidate")
    require(int(intermediates["train_self_hit_count"].item()) == 0, "Train self-hit")
    require(int(intermediates["test_candidate_count"].item()) == 0, "Test memory candidate")
    memory = set(train_idx.detach().cpu().tolist())
    require(bool(train_mask[candidates].all()), "Candidate failed independent train-mask check")
    require(not bool(test_mask[candidates].any()), "Test patient entered candidate tensor")
    require(
        all(int(value) in memory for value in candidates.detach().cpu().reshape(-1).tolist()),
        "Candidate is outside train_idx",
    )
    return {
        "candidate_counts_per_query": [4, 4, 4],
        "candidate_count_total_per_query": 12,
        "all_candidates_in_train_mask": True,
        "train_self_hit_count": 0,
        "test_candidate_count": 0,
        "memory_size": int(train_idx.numel()),
    }


def run_smoke(context: dict, output_root: Path, source: str) -> dict:
    smoke_root = output_root / "smoke"
    require(not smoke_root.exists(), f"Refusing to overwrite {smoke_root}")
    smoke_root.mkdir(parents=True)
    equivalence = initial_and_first_step_check(context, 0.10)
    bundle = make_training_objects(context, 0.10)
    (
        model,
        criterion,
        base_optimizer,
        retrieval_optimizer,
        base_scheduler,
        retrieval_scheduler,
        ema,
        base_parameters,
        retrieval_parameters,
        optimizer_audit,
    ) = bundle
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, train_idx, train_y = train_memory(context, 0)
    model.eval()
    with torch.no_grad():
        _, _, _, initial_intermediates = model(
            features, train_idx, train_y, return_intermediates=True
        )
    memory_audit = retrieval_audit(
        initial_intermediates, train_idx, train_mask, test_mask
    )

    tracked = {
        "w_out": model.cbtr.output_projection.weight,
        "w_key": model.cbtr.key_projection.weight,
        "w_diff": model.cbtr.diff_projection.weight,
        "label_embedding": model.cbtr.label_embedding.weight,
    }
    before = {name: value.detach().clone() for name, value in tracked.items()}
    gradient_max = {name: 0.0 for name in tracked}
    losses = []
    for epoch in range(1, 4):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        retrieval_optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary, _ = model(
            features, train_idx, train_y, return_intermediates=True
        )
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Smoke epoch {epoch}: non-finite loss")
        loss.backward()
        require(
            grads_finite(base_parameters + retrieval_parameters),
            f"Smoke epoch {epoch}: non-finite gradient",
        )
        for name, parameter in tracked.items():
            gradient_max[name] = max(gradient_max[name], grad_norm([parameter]))
        torch.nn.utils.clip_grad_norm_(
            base_parameters, float(context["config"].grad_clip)
        )
        base_optimizer.step()
        retrieval_optimizer.step()
        base_scheduler.step()
        retrieval_scheduler.step()
        ema.update(model)
        losses.append(float(loss.detach().cpu()))

    parameter_change = {
        name: float((parameter.detach() - before[name]).abs().max().cpu())
        for name, parameter in tracked.items()
    }
    for name in tracked:
        require(
            gradient_max[name] > 0.0 and math.isfinite(gradient_max[name]),
            f"{name} did not receive a finite non-zero gradient by epoch3",
        )
        require(parameter_change[name] > 0.0, f"{name} did not change by epoch3")
    model.eval()
    with torch.no_grad():
        reference_logits, _, _, inter = model(
            features, train_idx, train_y, return_intermediates=True
        )
    ratio = inter["retrieval_residual_h0_ratio"].detach().float()
    residual_max = float(inter["R_retrieval"].abs().max().cpu())
    require(residual_max > 0.0, "Retrieval residual is still zero after smoke")
    require(bool(torch.isfinite(ratio).all()), "Smoke residual ratio is non-finite")
    require(float(ratio.max().cpu()) <= 0.10 + 1e-6, "Smoke cap exceeded")

    checkpoint = {
        "model_state": clone_cpu_state(model),
        "ema_shadow": clone_ema_state(ema),
        "base_optimizer": deepcopy(base_optimizer.state_dict()),
        "retrieval_optimizer": deepcopy(retrieval_optimizer.state_dict()),
        "base_scheduler": deepcopy(base_scheduler.state_dict()),
        "retrieval_scheduler": deepcopy(retrieval_scheduler.state_dict()),
        "epoch": 3,
        "source_commit": source,
        "config": config_payload(context, 0.10, "smoke", source),
    }
    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded_bundle = make_training_objects(context, 0.10)
    (
        reloaded,
        _,
        reloaded_base_optimizer,
        reloaded_retrieval_optimizer,
        reloaded_base_scheduler,
        reloaded_retrieval_scheduler,
        reloaded_ema,
    ) = reloaded_bundle[:7]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    reloaded_base_optimizer.load_state_dict(payload["base_optimizer"])
    reloaded_retrieval_optimizer.load_state_dict(payload["retrieval_optimizer"])
    reloaded_base_scheduler.load_state_dict(payload["base_scheduler"])
    reloaded_retrieval_scheduler.load_state_dict(payload["retrieval_scheduler"])
    require(set(payload["ema_shadow"]) == set(reloaded_ema.shadow), "EMA keys changed")
    reloaded_ema.shadow = {
        name: value.to(context["device"]) for name, value in payload["ema_shadow"].items()
    }
    reloaded.eval()
    with torch.no_grad():
        roundtrip_logits, _, _ = reloaded(features, train_idx, train_y)
    roundtrip_diff = float((reference_logits - roundtrip_logits).abs().max().cpu())
    require(roundtrip_diff == 0.0, "Checkpoint strict roundtrip changed logits")
    summary = {
        "passed": True,
        "source_commit": source,
        "run_head_at_start": context["run_head"],
        "device": "cuda:0",
        "epochs": 3,
        "losses": losses,
        "equivalence": equivalence,
        "train_only_memory_audit": memory_audit,
        "optimizer_audit": optimizer_audit,
        "gradient_max": gradient_max,
        "parameter_change_max_abs": parameter_change,
        "retrieval_residual_max_abs": residual_max,
        "retrieval_residual_h0_ratio_max": float(ratio.max().cpu()),
        "checkpoint_strict_roundtrip": True,
        "checkpoint_logit_max_abs_diff": roundtrip_diff,
    }
    write_json(smoke_root / "config.json", config_payload(context, 0.10, "smoke", source))
    write_json(smoke_root / "summary.json", summary)
    print("CBTR smoke PASS", flush=True)
    return summary


def _optimizer_state_cpu(state: dict) -> dict:
    cloned = deepcopy(state)
    for payload in cloned.get("state", {}).values():
        for key, value in list(payload.items()):
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().cpu().clone()
    return cloned


def train_fold(
    context: dict,
    fold: int,
    cap: float,
    version_root: Path,
    source: str,
) -> tuple[dict, list[dict]]:
    final_dir = version_root / f"fold_{fold:02d}"
    staging_dir = version_root / f".fold_{fold:02d}_in_progress"
    require(not final_dir.exists(), f"Refusing to overwrite completed {final_dir}")
    require(not staging_dir.exists(), f"Stale in-progress fold requires explicit audit: {staging_dir}")
    staging_dir.mkdir(parents=True)
    write_json(
        staging_dir / "fold_run_config.json",
        {"fold": fold, "cap": float(cap), "source_commit": source},
    )
    fold_started = time.perf_counter()
    bundle = make_training_objects(context, cap)
    (
        model,
        criterion,
        base_optimizer,
        retrieval_optimizer,
        base_scheduler,
        retrieval_scheduler,
        ema,
        base_parameters,
        retrieval_parameters,
        optimizer_audit,
    ) = bundle
    features = context["dataset_data"]["Feature"]
    labels = context["dataset_data"]["Label"]
    train_mask, test_mask, train_idx, train_y = train_memory(context, fold)
    best = None
    best_payload = None
    retrieval_gradient_max = {
        "w_out": 0.0,
        "w_key": 0.0,
        "w_diff": 0.0,
        "label_embedding": 0.0,
    }
    tracked = {
        "w_out": model.cbtr.output_projection.weight,
        "w_key": model.cbtr.key_projection.weight,
        "w_diff": model.cbtr.diff_projection.weight,
        "label_embedding": model.cbtr.label_embedding.weight,
    }
    epoch_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        base_optimizer.zero_grad(set_to_none=True)
        retrieval_optimizer.zero_grad(set_to_none=True)
        raw, branches, auxiliary = model(features, train_idx, train_y)
        loss = criterion(raw, labels, train_mask, branches, auxiliary)
        require(bool(torch.isfinite(loss)), f"Fold {fold} epoch {epoch}: non-finite loss")
        loss.backward()
        require(
            grads_finite(base_parameters + retrieval_parameters),
            f"Fold {fold} epoch {epoch}: non-finite gradient",
        )
        for name, parameter in tracked.items():
            retrieval_gradient_max[name] = max(
                retrieval_gradient_max[name], grad_norm([parameter])
            )
        # Preserve the historical C1 clip exactly; CBTR has no added clipping.
        torch.nn.utils.clip_grad_norm_(
            base_parameters, float(context["config"].grad_clip)
        )
        base_optimizer.step()
        retrieval_optimizer.step()
        base_scheduler.step()
        retrieval_scheduler.step()
        ema.update(model)

        model.eval()
        with torch.no_grad():
            evaluation_raw, _, _ = model(features, train_idx, train_y)
            metrics, selection_tuple = metric_and_tuple(
                evaluation_raw, labels, test_mask, context
            )
        epoch_history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "correct": int(metrics["correct"]),
                "acc": float(metrics["acc"]),
                "macro_auc": float(metrics["macro_auc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "bacc": float(metrics["bacc"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "base_lr": float(base_optimizer.param_groups[0]["lr"]),
                "retrieval_lr": float(retrieval_optimizer.param_groups[0]["lr"]),
            }
        )
        if best is None or selection_tuple > best["selection_tuple"]:
            best = {
                "epoch": epoch,
                "selection_tuple": selection_tuple,
                "metrics": metrics,
                "loss": float(loss.detach().cpu()),
            }
            best_payload = {
                "model_state": clone_cpu_state(model),
                "ema_shadow": clone_ema_state(ema),
                "base_optimizer": _optimizer_state_cpu(base_optimizer.state_dict()),
                "retrieval_optimizer": _optimizer_state_cpu(retrieval_optimizer.state_dict()),
                "base_scheduler": deepcopy(base_scheduler.state_dict()),
                "retrieval_scheduler": deepcopy(retrieval_scheduler.state_dict()),
                "epoch": epoch,
                "source_commit": source,
                "cap": float(cap),
            }
    require(best is not None and best_payload is not None, f"Fold {fold}: no best epoch")
    for name, value in retrieval_gradient_max.items():
        require(value > 0.0 and math.isfinite(value), f"Fold {fold}: {name} never activated")

    model.load_state_dict(best_payload["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        on_raw, _, _, on_intermediates = model(
            features, train_idx, train_y, return_intermediates=True
        )
        off_raw, _, _, _ = model(
            features,
            train_idx,
            train_y,
            return_intermediates=True,
            retrieval_enabled=False,
        )
    reloaded_metrics, _ = metric_and_tuple(on_raw, labels, test_mask, context)
    require(reloaded_metrics == best["metrics"], f"Fold {fold}: best reload changed metrics")
    retrieval_audit(on_intermediates, train_idx, train_mask, test_mask)
    rows = prediction_rows(fold, on_raw, labels, test_mask, context)
    _, off_probability, off_prediction = cme.score_logits(
        off_raw,
        context["dataset_dict"]["Label_Weight"],
        float(context["config"].logit_adjust_tau),
    )
    internal = test_mask.nonzero(as_tuple=False).flatten()
    class_mass = on_intermediates["attention_class_mass"][internal].detach().cpu().numpy()
    entropy = on_intermediates["attention_entropy"][internal].detach().cpu().numpy()
    effective = on_intermediates["effective_neighbor_count"][internal].detach().cpu().numpy()
    ratio = on_intermediates["retrieval_residual_h0_ratio"][internal].detach().cpu().numpy()
    saturated = on_intermediates["retrieval_cap_saturated"][internal].detach().cpu().numpy()
    off_raw_np = off_raw[internal].detach().cpu().numpy()
    off_probability_np = off_probability[internal].detach().cpu().numpy()
    off_prediction_np = off_prediction[internal].detach().cpu().numpy()
    for index, row in enumerate(rows):
        row.update(
            {
                "attention_mass_AD": float(class_mass[index, 0]),
                "attention_mass_CN": float(class_mass[index, 1]),
                "attention_mass_SMCI": float(class_mass[index, 2]),
                "attention_entropy": float(entropy[index]),
                "effective_neighbor_count": float(effective[index]),
                "retrieval_residual_h0_ratio": float(ratio[index]),
                "cap_saturated": int(bool(saturated[index])),
                "off_raw_logit_AD": float(off_raw_np[index, 0]),
                "off_raw_logit_CN": float(off_raw_np[index, 1]),
                "off_raw_logit_SMCI": float(off_raw_np[index, 2]),
                "off_probability_AD": float(off_probability_np[index, 0]),
                "off_probability_CN": float(off_probability_np[index, 1]),
                "off_probability_SMCI": float(off_probability_np[index, 2]),
                "off_prediction": int(off_prediction_np[index]),
            }
        )
    require(cme.metrics_from_rows(rows) == best["metrics"], f"Fold {fold}: row readback changed metrics")
    elapsed = time.perf_counter() - fold_started
    summary = {
        "fold": fold,
        "source_commit": source,
        "cap": float(cap),
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "selection_tuple": list(best["selection_tuple"]),
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameter_count": CBTR_PARAMETERS,
        "retrieval_gradient_max": retrieval_gradient_max,
        "optimizer_audit": optimizer_audit,
        "train_memory_size": int(train_idx.numel()),
        "test_size": int(test_mask.sum().item()),
        "elapsed_seconds": float(elapsed),
    }
    torch.save(best_payload, staging_dir / "checkpoint_best.pt")
    write_json(staging_dir / "fold_metrics.json", summary)
    write_csv(staging_dir / "epoch_metrics.csv", epoch_history)
    write_csv(staging_dir / "fold_oof_predictions.csv", rows)
    write_json(
        staging_dir / "completion_marker.json",
        {
            "complete": True,
            "fold": fold,
            "source_commit": source,
            "cap": float(cap),
            "row_count": len(rows),
        },
    )
    staging_dir.rename(final_dir)
    print(
        f"fold={fold} best_epoch={best['epoch']} correct={best['metrics']['correct']} "
        f"ACC={best['metrics']['acc']:.7f}",
        flush=True,
    )
    del model, criterion, base_optimizer, retrieval_optimizer, ema
    torch.cuda.empty_cache()
    return summary, rows


def paired_or_unavailable(rows: list[dict], reference_by_subject) -> dict:
    if reference_by_subject is None:
        return {
            "available": False,
            "repairs": "unavailable",
            "damages": "unavailable",
            "net_repairs": "unavailable",
            "changed": "unavailable",
        }
    return paired_comparison(rows, reference_by_subject)


def mechanism_summary(rows: list[dict], paired_c1: dict) -> dict:
    class_mass = np.asarray(
        [
            [
                float(row["attention_mass_AD"]),
                float(row["attention_mass_CN"]),
                float(row["attention_mass_SMCI"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    truth = np.asarray([int(row["truth"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    off_prediction = np.asarray([int(row["off_prediction"]) for row in rows], dtype=np.int64)
    entropy = np.asarray([float(row["attention_entropy"]) for row in rows])
    effective = np.asarray([float(row["effective_neighbor_count"]) for row in rows])
    ratio = np.asarray([float(row["retrieval_residual_h0_ratio"]) for row in rows])
    saturated = np.asarray([int(row["cap_saturated"]) for row in rows])
    raw_on = np.asarray(
        [
            [float(row[f"raw_logit_{name}"]) for name in CLASS_NAMES]
            for row in rows
        ]
    )
    raw_off = np.asarray(
        [
            [float(row[f"off_raw_logit_{name}"]) for name in CLASS_NAMES]
            for row in rows
        ]
    )
    probability_on = np.asarray(
        [
            [float(row[f"probability_{name}"]) for name in CLASS_NAMES]
            for row in rows
        ]
    )
    probability_off = np.asarray(
        [
            [float(row[f"off_probability_{name}"]) for name in CLASS_NAMES]
            for row in rows
        ]
    )
    direct_repairs_mask = (off_prediction != truth) & (prediction == truth)
    direct_damages_mask = (off_prediction == truth) & (prediction != truth)
    direct_changed_mask = off_prediction != prediction
    true_class_mass = class_mass[np.arange(len(rows)), truth]
    subject_to_index = {
        int(row["subject_index"]): index for index, row in enumerate(rows)
    }

    def subset_mass(subjects) -> dict:
        indices = [subject_to_index[int(subject)] for subject in subjects]
        if not indices:
            return {"count": 0, "true_class_attention_mass_mean": None}
        return {
            "count": len(indices),
            "true_class_attention_mass_mean": float(true_class_mass[indices].mean()),
        }

    repairs_mass = (
        subset_mass(paired_c1["repair_subject_indices"])
        if paired_c1.get("available")
        else {"count": "unavailable", "true_class_attention_mass_mean": None}
    )
    damages_mass = (
        subset_mass(paired_c1["damage_subject_indices"])
        if paired_c1.get("available")
        else {"count": "unavailable", "true_class_attention_mass_mean": None}
    )
    return {
        "candidate_attention_mass_mean": {
            name: float(class_mass[:, index].mean())
            for index, name in enumerate(CLASS_NAMES)
        },
        "attention_entropy_mean": float(entropy.mean()),
        "effective_neighbor_count_mean": float(effective.mean()),
        "retrieval_class_mass_argmax_accuracy": float(
            (class_mass.argmax(axis=1) == truth).mean()
        ),
        "c1_repairs_true_class_attention": repairs_mass,
        "c1_damages_true_class_attention": damages_mass,
        "retrieval_residual_h0_ratio_mean": float(ratio.mean()),
        "retrieval_residual_h0_ratio_max": float(ratio.max()),
        "cap_saturation_fraction": float(saturated.mean()),
        "same_checkpoint_r_on_vs_off": {
            "logit_max_abs_difference": float(np.abs(raw_on - raw_off).max()),
            "probability_max_abs_difference": float(
                np.abs(probability_on - probability_off).max()
            ),
            "argmax_changed": int(direct_changed_mask.sum()),
            "direct_repairs": int(direct_repairs_mask.sum()),
            "direct_damages": int(direct_damages_mask.sum()),
            "direct_changed_subject_indices": [
                int(rows[index]["subject_index"])
                for index in np.flatnonzero(direct_changed_mask)
            ],
        },
        "test_labels_used_offline_after_complete_oof_only": True,
    }


def predicted_counts(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        name: int(matrix[:, index].sum()) for index, name in enumerate(CLASS_NAMES)
    }


def validate_completed_fold(
    context: dict,
    fold: int,
    cap: float,
    final_dir: Path,
    source: str,
) -> tuple[dict, list[dict]]:
    marker_path = final_dir / "completion_marker.json"
    summary_path = final_dir / "fold_metrics.json"
    rows_path = final_dir / "fold_oof_predictions.csv"
    epoch_path = final_dir / "epoch_metrics.csv"
    checkpoint_path = final_dir / "checkpoint_best.pt"
    require(
        marker_path.is_file()
        and summary_path.is_file()
        and rows_path.is_file()
        and epoch_path.is_file()
        and checkpoint_path.is_file(),
        f"Incomplete fold {fold}",
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("complete") is True, f"Fold {fold} marker is not complete")
    require(int(marker.get("fold")) == fold, f"Fold {fold} marker fold mismatch")
    require(marker.get("source_commit") == source, f"Fold {fold} source mismatch")
    require(abs(float(marker.get("cap")) - cap) <= 1e-12, f"Fold {fold} cap mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_csv(rows_path)
    epoch_rows = read_csv(epoch_path)
    _, test_mask, _, _ = train_memory(context, fold)
    require(len(rows) == int(test_mask.sum()), f"Fold {fold} OOF count mismatch")
    require(int(marker.get("row_count")) == len(rows), f"Fold {fold} marker row count mismatch")
    require(int(summary.get("fold")) == fold, f"Fold {fold} summary fold mismatch")
    require(summary.get("source_commit") == source, f"Fold {fold} summary source mismatch")
    require(abs(float(summary.get("cap")) - cap) <= 1e-12, f"Fold {fold} summary cap mismatch")
    require(int(summary.get("parameter_count")) == EXPECTED_PARAMETERS, f"Fold {fold} parameter mismatch")
    require(int(summary.get("added_parameter_count")) == CBTR_PARAMETERS, f"Fold {fold} added parameter mismatch")
    require(1 <= int(summary.get("best_epoch")) <= EPOCHS, f"Fold {fold} best epoch invalid")
    require(len(epoch_rows) == EPOCHS, f"Fold {fold} epoch history incomplete")
    require(
        [int(row["epoch"]) for row in epoch_rows] == list(range(1, EPOCHS + 1)),
        f"Fold {fold} epoch sequence changed",
    )
    independently_best = max(
        epoch_rows,
        key=lambda row: (
            float(row["acc"]),
            float(row["macro_auc"]),
            float(row["macro_f1"]),
        ),
    )
    require(
        int(independently_best["epoch"]) == int(summary["best_epoch"]),
        f"Fold {fold} independent best-epoch selection mismatch",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required_checkpoint_fields = {
        "model_state",
        "ema_shadow",
        "base_optimizer",
        "retrieval_optimizer",
        "base_scheduler",
        "retrieval_scheduler",
        "epoch",
        "source_commit",
        "cap",
    }
    require(required_checkpoint_fields <= set(checkpoint), f"Fold {fold} checkpoint fields missing")
    require(int(checkpoint["epoch"]) == int(summary["best_epoch"]), f"Fold {fold} checkpoint epoch mismatch")
    require(checkpoint["source_commit"] == source, f"Fold {fold} checkpoint source mismatch")
    require(abs(float(checkpoint["cap"]) - cap) <= 1e-12, f"Fold {fold} checkpoint cap mismatch")
    require(
        any(name.startswith("cbtr.") for name in checkpoint["model_state"]),
        f"Fold {fold} checkpoint omits CBTR model state",
    )
    require(
        any(name.startswith("cbtr.") for name in checkpoint["ema_shadow"]),
        f"Fold {fold} checkpoint omits CBTR EMA state",
    )
    expected = context["expected_oof_alignment"]
    subjects = []
    for row in rows:
        subject = int(row["subject_index"])
        subjects.append(subject)
        require(expected[subject] == (fold, int(row["truth"])), f"Fold {fold} OOF alignment mismatch")
        probability = np.asarray(
            [float(row[f"probability_{name}"]) for name in CLASS_NAMES]
        )
        require(int(row["prediction"]) == int(probability.argmax()), f"Fold {fold} prediction/probability mismatch")
    require(len(subjects) == len(set(subjects)), f"Fold {fold} subjects are duplicated")
    require(cme.metrics_from_rows(rows) == summary["best_metrics"], f"Fold {fold} metrics readback mismatch")
    return summary, rows


def run_version(
    context: dict,
    output_root: Path,
    version: str,
    cap: float,
    source: str,
) -> dict:
    version_root = output_root / version
    payload = config_payload(context, cap, version, source)
    config_path = version_root / "config.json"
    if version_root.exists():
        require(config_path.is_file(), f"Existing {version_root} has no config")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        require(existing == payload, f"Existing {version_root} config/source differs")
    else:
        version_root.mkdir(parents=True)
        write_json(config_path, payload)
    wall_started = time.perf_counter()
    fold_summaries = []
    rows = []
    resumed_folds = []
    for fold in FOLDS:
        final_dir = version_root / f"fold_{fold:02d}"
        staging_dir = version_root / f".fold_{fold:02d}_in_progress"
        if final_dir.exists():
            summary, fold_rows = validate_completed_fold(
                context, fold, cap, final_dir, source
            )
            resumed_folds.append(fold)
        else:
            if staging_dir.exists():
                run_config_path = staging_dir / "fold_run_config.json"
                require(run_config_path.is_file(), f"Cannot validate interrupted fold {fold}")
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                require(
                    run_config
                    == {"fold": fold, "cap": float(cap), "source_commit": source},
                    f"Interrupted fold {fold} source/config mismatch",
                )
                shutil.rmtree(staging_dir)
            summary, fold_rows = train_fold(
                context, fold, cap, version_root, source
            )
        fold_summaries.append(summary)
        rows.extend(fold_rows)
    require(len(rows) == 598, "OOF row count is not 598")
    by_subject = {int(row["subject_index"]): row for row in rows}
    require(len(by_subject) == 598, "OOF subjects are not unique")
    require(set(by_subject) == set(context["expected_oof_alignment"]), "OOF subject coverage changed")
    for subject, row in by_subject.items():
        require(
            context["expected_oof_alignment"][subject]
            == (int(row["fold"]), int(row["truth"])),
            f"OOF fold/truth mismatch for subject {subject}",
        )
    rows.sort(key=lambda row: int(row["subject_index"]))
    metrics = cme.metrics_from_rows(rows)
    paired_c1 = paired_or_unavailable(rows, context["c1_by_subject"])
    paired_pc = paired_or_unavailable(rows, context["pc_bbf_by_subject"])
    boundaries = boundary_errors(metrics)
    mechanism = mechanism_summary(rows, paired_c1)
    fold_rows = [
        {
            "fold": int(summary["fold"]),
            "best_epoch": int(summary["best_epoch"]),
            "correct": int(summary["best_metrics"]["correct"]),
            "acc": float(summary["best_metrics"]["acc"]),
            "macro_f1": float(summary["best_metrics"]["macro_f1"]),
            "bacc": float(summary["best_metrics"]["bacc"]),
            "macro_auc": float(summary["best_metrics"]["macro_auc"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
        }
        for summary in fold_summaries
    ]
    fold_acc = np.asarray([row["acc"] for row in fold_rows], dtype=np.float64)
    summary = {
        "experiment": EXPERIMENT,
        "version": version,
        "cap": float(cap),
        "source_commit": source,
        "metrics": metrics,
        "fold_acc_mean": float(fold_acc.mean()),
        "fold_acc_sample_std": float(fold_acc.std(ddof=1)),
        "fold_metrics": fold_rows,
        "parameter_count": EXPECTED_PARAMETERS,
        "added_parameter_count": CBTR_PARAMETERS,
        "training_seconds": float(sum(row["elapsed_seconds"] for row in fold_rows)),
        "wall_seconds": float(time.perf_counter() - wall_started),
        "predicted_counts": predicted_counts(metrics),
        "boundary_errors": boundaries,
        "delta_vs_c1": metric_delta(metrics, C1),
        "delta_vs_pc_bbf": metric_delta(metrics, PC_BBF),
        "paired_vs_c1": paired_c1,
        "paired_vs_pc_bbf": paired_pc,
        "c1_oof_alignment_available": context["c1_alignment_available"],
        "pc_bbf_oof_alignment_available": context["pc_bbf_alignment_available"],
        "resumed_completed_folds": resumed_folds,
        "mechanism": mechanism,
    }
    write_csv(version_root / "oof_predictions.csv", rows)
    write_csv(version_root / "fold_metrics.csv", fold_rows)
    write_json(version_root / "mechanism_summary.json", mechanism)
    write_json(version_root / "formal_summary.json", summary)
    write_version_report(version_root / "REPORT.md", summary)
    require(cme.metrics_from_rows(read_csv(version_root / "oof_predictions.csv")) == metrics, "OOF readback mismatch")
    return summary


def write_version_report(path: Path, summary: dict) -> None:
    metrics = summary["metrics"]
    paired = summary["paired_vs_c1"]
    paired_pc = summary["paired_vs_pc_bbf"]
    lines = [
        f"# CBTR-C1 {summary['version']}",
        "",
        f"Source commit: `{summary['source_commit']}`; cap={summary['cap']:.2f}.",
        "",
        "## Formal ten-fold result",
        "",
        f"- Correct: {metrics['correct']}/598",
        f"- ACC: {metrics['acc']:.7f}",
        f"- Macro-F1: {metrics['macro_f1']:.7f}",
        f"- BACC: {metrics['bacc']:.7f}",
        f"- Probability Macro-AUC: {metrics['macro_auc']:.7f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.7f}",
        f"- Confusion: `{metrics['confusion_matrix']}`",
        f"- Fold ACC: {summary['fold_acc_mean']:.7f} +/- {summary['fold_acc_sample_std']:.7f} sample SD",
        f"- Repairs/damages/changed vs C1: {paired['repairs']}/{paired['damages']}/{paired['changed']}",
        f"- Repairs/damages/changed vs PC-BBF: {paired_pc['repairs']}/{paired_pc['damages']}/{paired_pc['changed']}",
        f"- AD-sMCI / CN-sMCI / AD-CN: {summary['boundary_errors']['AD_SMCI']} / "
        f"{summary['boundary_errors']['CN_SMCI']} / {summary['boundary_errors']['AD_CN']}",
        f"- Predicted AD/CN/sMCI: {summary['predicted_counts']['AD']}/"
        f"{summary['predicted_counts']['CN']}/{summary['predicted_counts']['SMCI']}",
        f"- Delta vs C1: `{summary['delta_vs_c1']}`",
        f"- Delta vs PC-BBF: `{summary['delta_vs_pc_bbf']}`",
        f"- Parameters: {summary['parameter_count']} (+{summary['added_parameter_count']})",
        f"- Training / wall seconds: {summary['training_seconds']:.3f} / {summary['wall_seconds']:.3f}",
        "",
        "## Retrieval mechanism",
        "",
        f"- Class attention mass: `{summary['mechanism']['candidate_attention_mass_mean']}`",
        f"- Entropy / effective neighbors: {summary['mechanism']['attention_entropy_mean']:.6f} / "
        f"{summary['mechanism']['effective_neighbor_count_mean']:.6f}",
        f"- Residual/H0 mean / max: {summary['mechanism']['retrieval_residual_h0_ratio_mean']:.6f} / "
        f"{summary['mechanism']['retrieval_residual_h0_ratio_max']:.6f}",
        f"- Cap saturation: {summary['mechanism']['cap_saturation_fraction']:.6f}",
        f"- Same-checkpoint R-on/off: `{summary['mechanism']['same_checkpoint_r_on_vs_off']}`",
        "",
        "Test-label mechanism summaries were computed only after all 598 OOF rows were complete.",
        "",
        "## Per-fold selected epochs",
        "",
        "| Fold | Best epoch | Correct | ACC |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} |"
        for row in summary["fold_metrics"]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def v1_1_gate(v1: dict) -> dict:
    metrics = v1["metrics"]
    paired = v1["paired_vs_c1"]
    boundary = v1["boundary_errors"]
    mechanism = v1["mechanism"]
    direct = mechanism["same_checkpoint_r_on_vs_off"]
    checks = {
        "correct_is_561_or_562": int(metrics["correct"]) in {561, 562},
        "c1_oof_aligned": bool(v1["c1_oof_alignment_available"]),
        "repairs_gt_damages": bool(
            paired.get("available")
            and int(paired["repairs"]) > int(paired["damages"])
        ),
        "ad_cn_zero": int(boundary["AD_CN"]) == 0,
        "adjacent_errors_below_38": int(boundary["adjacent_total"]) < 38,
        "bacc_within_c1_minus_0p003": float(metrics["bacc"]) >= C1["bacc"] - 0.003,
        "cap_saturation_at_least_0p70": float(mechanism["cap_saturation_fraction"]) >= 0.70,
        "r_on_off_argmax_changed_at_least_2": int(direct["argmax_changed"]) >= 2,
        "r_on_off_direct_repairs_ge_damages": int(direct["direct_repairs"])
        >= int(direct["direct_damages"]),
    }
    return {"trigger": all(checks.values()), "checks": checks}


def select_version(v1: dict, v1_1: dict | None) -> tuple[str, dict, dict]:
    if v1_1 is None:
        return "formal_v1", v1, {"reason": "v1.1_not_run"}
    tie_bacc_ok = float(v1_1["metrics"]["bacc"]) >= float(v1["metrics"]["bacc"]) - 0.003
    selected = False
    reason = "v1_retained"
    if int(v1_1["metrics"]["correct"]) > int(v1["metrics"]["correct"]):
        selected, reason = True, "v1.1_higher_correct"
    elif tie_bacc_ok and int(v1_1["metrics"]["correct"]) == int(v1["metrics"]["correct"]):
        old_tuple = (float(v1["metrics"]["macro_auc"]), float(v1["metrics"]["macro_f1"]))
        new_tuple = (float(v1_1["metrics"]["macro_auc"]), float(v1_1["metrics"]["macro_f1"]))
        if new_tuple > old_tuple:
            selected, reason = True, "correct_tie_auc_then_macro_f1"
    if (
        int(v1_1["metrics"]["correct"]) == int(v1["metrics"]["correct"])
        and not tie_bacc_ok
    ):
        reason = "v1.1_bacc_more_than_0.003_below_v1"
    return (
        ("formal_v1_1", v1_1, {"reason": reason})
        if selected
        else ("formal_v1", v1, {"reason": reason})
    )


def final_decision(summary: dict) -> str:
    metrics = summary["metrics"]
    boundary = summary["boundary_errors"]
    paired = summary["paired_vs_c1"]
    repairs_not_greater = bool(
        paired.get("available")
        and int(paired["repairs"]) <= int(paired["damages"])
    )
    # Hard NO_GAIN conditions take precedence over nominal score buckets.
    if (
        int(metrics["correct"]) <= 560
        or repairs_not_greater
        or int(boundary["AD_CN"]) > 0
        or float(metrics["bacc"]) < C1["bacc"] - 0.005
    ):
        return "NO_GAIN"
    if (
        int(metrics["correct"]) >= 563
        and int(boundary["AD_CN"]) == 0
        and float(metrics["bacc"]) >= C1["bacc"] - 0.003
    ):
        return "STRONG_GO"
    if (
        int(metrics["correct"]) == 562
        and int(boundary["AD_CN"]) == 0
        and float(metrics["bacc"]) >= ORIGINAL_BACC
        and int(boundary["adjacent_total"]) <= 37
    ):
        return "POSITIVE_GO"
    stronger = sum(
        float(metrics[key]) > float(PC_BBF[key])
        for key in ("macro_f1", "bacc", "macro_auc")
    )
    if (
        int(metrics["correct"]) == 561
        and stronger >= 2
        and paired.get("available")
        and int(paired["repairs"]) > int(paired["damages"])
    ):
        return "MECHANISM_ONLY"
    return "NO_GAIN"


def write_final_report(path: Path, payload: dict) -> None:
    summary = payload["selected_summary"]
    metrics = summary["metrics"]
    paired = summary["paired_vs_c1"]
    paired_pc = summary["paired_vs_pc_bbf"]
    lines = [
        "# CBTR-C1 v1 final report",
        "",
        f"Decision: **{payload['decision']}**",
        f"v1.1 run: **{payload['v1_1_ran']}**; selected: **{payload['selected_version']}**.",
        "",
        f"Correct={metrics['correct']}/598; ACC={metrics['acc']:.7f}; Macro-F1={metrics['macro_f1']:.7f}; "
        f"BACC={metrics['bacc']:.7f}; Probability Macro-AUC={metrics['macro_auc']:.7f}; "
        f"Weighted-F1={metrics['weighted_f1']:.7f}.",
        "",
        f"Confusion: `{metrics['confusion_matrix']}`.",
        f"Repairs/damages/changed vs C1: {paired['repairs']}/{paired['damages']}/{paired['changed']}.",
        f"Repairs/damages/changed vs PC-BBF: {paired_pc['repairs']}/{paired_pc['damages']}/{paired_pc['changed']}.",
        f"AD-sMCI / CN-sMCI / AD-CN: {summary['boundary_errors']['AD_SMCI']} / "
        f"{summary['boundary_errors']['CN_SMCI']} / {summary['boundary_errors']['AD_CN']}.",
        f"Predicted AD/CN/sMCI: {summary['predicted_counts']['AD']}/"
        f"{summary['predicted_counts']['CN']}/{summary['predicted_counts']['SMCI']}.",
        f"Delta vs C1: `{summary['delta_vs_c1']}`.",
        f"Delta vs PC-BBF: `{summary['delta_vs_pc_bbf']}`.",
        f"Parameters={summary['parameter_count']} (+{summary['added_parameter_count']}); "
        f"training={summary['training_seconds']:.3f}s, wall={summary['wall_seconds']:.3f}s.",
        f"Total formal wall={payload['formal_total_wall_seconds']:.3f}s.",
        "",
        "The retrieval memory was fold-train-only; test-label mechanism statistics were produced offline after complete OOF assembly.",
        "",
        "## Selected per-fold epochs",
        "",
        "| Fold | Best epoch | Correct | ACC |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['fold']} | {row['best_epoch']} | {row['correct']} | {row['acc']:.7f} |"
        for row in summary["fold_metrics"]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal(context: dict, output_root: Path, source: str) -> dict:
    smoke_path = output_root / "smoke" / "summary.json"
    smoke_config_path = output_root / "smoke" / "config.json"
    require(smoke_path.is_file(), "Formal requires the completed three-epoch smoke")
    require(smoke_config_path.is_file(), "Formal requires the locked smoke config")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke.get("passed") is True, "Smoke did not pass")
    require(smoke.get("source_commit") == source, "Smoke/source commit mismatch")
    smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    require(
        smoke_config == config_payload(context, 0.10, "smoke", source),
        "Historical protocol or smoke configuration changed before formal training",
    )
    formal_started = time.perf_counter()
    v1 = run_version(context, output_root, "formal_v1", 0.10, source)
    gate = v1_1_gate(v1)
    v1_1 = (
        run_version(context, output_root, "formal_v1_1", 0.15, source)
        if gate["trigger"]
        else None
    )
    selected_version, selected, selection = select_version(v1, v1_1)
    decision = final_decision(selected)
    payload = {
        "experiment": EXPERIMENT,
        "source_commit": source,
        "run_head_at_start": context["run_head"],
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "v1_1_ran": v1_1 is not None,
        "v1_1_gate": gate,
        "selected_version": selected_version,
        "selection": selection,
        "selected_summary": selected,
        "v1_summary": v1,
        "v1_1_summary": v1_1,
        "formal_total_wall_seconds": float(time.perf_counter() - formal_started),
        "reproduction_commands": {
            "smoke": f'"{sys.executable}" -u -B scripts/run_cbtr_c1_v1.py smoke --device cuda:0',
            "formal": f'"{sys.executable}" -u -B scripts/run_cbtr_c1_v1.py formal --device cuda:0',
        },
    }
    write_json(output_root / "config.json", config_payload(context, selected["cap"], selected_version, source))
    write_json(output_root / "formal_summary.json", payload)
    write_json(output_root / "mechanism_summary.json", selected["mechanism"])
    write_csv(output_root / "fold_metrics.csv", selected["fold_metrics"])
    selected_rows = read_csv(output_root / selected_version / "oof_predictions.csv")
    write_csv(output_root / "oof_predictions.csv", selected_rows)
    write_final_report(output_root / "REPORT.md", payload)
    print(
        f"FINAL decision={decision} selected={selected_version} "
        f"correct={selected['metrics']['correct']}/598",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("smoke", "formal"):
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, run_head = require_committed_source()
    context = load_context(args.device)
    context["run_head"] = run_head
    output_root = ROOT / OUTPUT_REL
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        run_smoke(context, output_root, source)
    else:
        run_formal(context, output_root, source)


if __name__ == "__main__":
    main()
