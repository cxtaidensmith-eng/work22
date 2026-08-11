"""Fixed SP-LRIF cross-task validation on three registered tasks."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2, criterion_query_pool_no_orth  # noqa: E402
from Model import HeterGraph_Model_Kmeans  # noqa: E402
from Model.cme_dual_branch import CMEDualBranchModel  # noqa: E402
from Model.sp_lrif import SPLRIFDualBranchModel  # noqa: E402
from Utils import SET_Random  # noqa: E402
from scripts import run_tad_binary_hparam_search_v1 as binary  # noqa: E402
from scripts import run_c1_broad_hparam_search_v1 as broad  # noqa: E402

EXPERIMENT = "sp_lrif_cross_task_v1"
BRANCH = "experiment/sp-lrif-cross-task-v1"
BASE_COMMIT = "6c1400078a5efc4a6e3f69c962c78f6b6f3be2dd"
RESULT = ROOT / "experiments" / EXPERIMENT
CONFIG = RESULT / "experiment_config.json"
INSPECT = RESULT / "inspect_manifest.json"
SMOKE = RESULT / "smoke"
FOLDS = tuple(range(10))
EPOCHS = 400
SEED = 0
INTERACTION_RANK = 4
RUNNER_REL = "scripts/run_sp_lrif_cross_task_v1.py"
MODEL_REL = "Model/sp_lrif.py"

REFERENCE_COMMITS = {
    "tad_binary": "fcf57882ce242d07d4944755003559bfe97c534e",
    "abide5": "a23134cf23e3c95cb237931d92c327c001545a06",
    "tad_triclass": "c37c336eff5b5b7d9b62e98f84587dccf02b61e2",
    "abide": "83aa87adf86a822eeca3841d62e818ec70596fdb",
}

BINARY_PROFILES: dict[str, dict[str, Any]] = {
    "tad_binary": {
        "protocol_source": ("worktree", "experiments/tad_binary_hparam_search_v1/protocol.json"),
        "reference_commit": REFERENCE_COMMITS["tad_binary"],
        "reference_oof": "experiments/tad_binary_hparam_search_v1/tuned_b1_oof_predictions.csv",
        "reference_summary": "experiments/tad_binary_hparam_search_v1/search_summary.json",
        "trial_id": "D5", "lr": 0.0125, "weight_decay": 0.00025, "dropout": 0.67,
        "rank": 8, "private_multiplier": 1.0, "positive_index": 1,
        "sample_count": 535, "modalities": 5, "expected_correct": 519,
    },
    "abide5": {
        "protocol_source": (REFERENCE_COMMITS["abide5"], "experiments/abide5_hparam_search_v1/protocol.json"),
        "reference_commit": REFERENCE_COMMITS["abide5"],
        "reference_oof": "experiments/abide5_hparam_search_v1/tuned_b1_oof_predictions.csv",
        "reference_summary": "experiments/abide5_hparam_search_v1/search_summary.json",
        "trial_id": "D3", "lr": 0.005, "weight_decay": 0.0005, "dropout": 0.45,
        "rank": 4, "private_multiplier": 2.0, "positive_index": 0,
        "sample_count": 864, "modalities": 5, "expected_correct": 769,
    },
}

TRI_PROFILE = {
    "trial_id": "A012", "rank": 8, "base_lr": 0.011271416075886307,
    "base_weight_decay": 0.0010917677921787822, "lambda_aux": 0.5,
    "adapter_lr_multiplier": 2.0, "dropout_multiplier": 1.1,
    "expected_correct": 562,
}

REQUIRED_SOURCE = (
    ".gitignore", RUNNER_REL, f"experiments/{EXPERIMENT}/experiment_config.json",
    f"experiments/{EXPERIMENT}/inspect_manifest.json",
)
LOCKED = (
    MODEL_REL, "Model/cme_dual_branch.py", "Model/network.py", "Model/models.py",
    "Loss/loss_fn.py", "Utils/utils.py", "scripts/run_tad_binary_hparam_search_v1.py",
    "scripts/run_c1_broad_hparam_search_v1.py", "scripts/run_sp_lrif_abide_v1.py",
)

class InvariantError(RuntimeError):
    pass

def require(value: bool, message: str) -> None:
    if not value:
        raise InvariantError(message)

def git(*args: str, check: bool = True, binary_output: bool = False):
    return subprocess.run(["git", *args], cwd=ROOT, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=not binary_output)

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def git_blob(commit: str, path: str) -> bytes:
    return git("show", f"{commit}:{path}", binary_output=True).stdout

def blob_json(commit: str, path: str) -> Any:
    return json.loads(git_blob(commit, path).decode("utf-8"))

def blob_csv(commit: str, path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(git_blob(commit, path).decode("utf-8"))))

def write_json(path: Path, value: Any) -> None:
    binary.atomic_write_json(path, value)

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    binary.atomic_write_csv(path, rows)

def read_json(path: Path) -> Any:
    return binary.read_json(path)

def config() -> dict[str, Any]:
    return read_json(CONFIG)

def task_protocol(task: str) -> dict[str, Any]:
    profile = BINARY_PROFILES[task]
    commit, path = profile["protocol_source"]
    value = read_json(ROOT / path) if commit == "worktree" else blob_json(commit, path)
    require(value["sample_count"] == profile["sample_count"] and len(value["modalities"]) == profile["modalities"], f"{task}: protocol changed")
    require(value["positive_index"] == profile["positive_index"], f"{task}: positive class changed")
    return value

def typed_binary_rows(raw: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for r in raw:
        rows.append({
            "task_id": r["task_id"], "trial_id": r.get("trial_id", "REFERENCE"), "arm": r["arm"],
            "fold": int(r["fold"]), "subject_id": r["subject_id"],
            "original_csv_index": int(r["original_csv_index"]), "feature_sha256": r["feature_sha256"],
            "truth": int(r["truth"]), "prediction": int(r["prediction"]),
            "logit_0": float(r["logit_0"]), "logit_1": float(r["logit_1"]),
            "probability_0": float(r["probability_0"]), "probability_1": float(r["probability_1"]),
            "positive_probability": float(r["positive_probability"]),
        })
    return rows

def reference_binary(task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = BINARY_PROFILES[task]
    rows = typed_binary_rows(blob_csv(profile["reference_commit"], profile["reference_oof"]))
    truth = np.asarray([r["truth"] for r in rows], dtype=np.int64)
    probs = np.asarray([[r["probability_0"], r["probability_1"]] for r in rows], dtype=float)
    metrics = binary.historical.binary_metrics(truth, probs, task_protocol(task))
    require(metrics["correct"] == profile["expected_correct"] and len(rows) == profile["sample_count"], f"{task}: reference anchor changed")
    require(len({r["subject_id"] for r in rows}) == len(rows), f"{task}: duplicate reference IDs")
    return metrics, rows

def reference_tri() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = ROOT / "experiments" / "c1_broad_hparam_search_v1"
    predictions = broad.read_csv(root / "best_oof_predictions.csv")
    probabilities = {int(r["subject_index"]):r for r in broad.read_csv(root / "best_oof_probabilities.csv")}
    rows=[]
    for row in predictions:
        probability=probabilities[int(row["subject_index"])]
        require(int(row["fold"])==int(probability["fold"]) and int(row["truth"])==int(probability["truth"]),"A012 reference OOF split changed")
        rows.append({**row,"probability_AD":float(probability["probability_AD"]),"probability_CN":float(probability["probability_CN"]),"probability_SMCI":float(probability["probability_SMCI"])})
    metrics = broad.cme.metrics_from_rows(rows)
    require(metrics["correct"] == 562 and len(rows) == 598, "A012 reference changed")
    return metrics, rows

def paired(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], id_key: str) -> dict[str, Any]:
    old = {str(r[id_key]): r for r in reference}; new = {str(r[id_key]): r for r in candidate}
    require(set(old) == set(new), "Paired subject sets changed")
    repairs = damages = changed = 0
    for key in old:
        a, b = old[key], new[key]
        require(int(a["truth"]) == int(b["truth"]) and int(a["fold"]) == int(b["fold"]), "Paired truth/fold changed")
        ca = int(a["prediction"]) == int(a["truth"]); cb = int(b["prediction"]) == int(b["truth"])
        repairs += int(not ca and cb); damages += int(ca and not cb); changed += int(a["prediction"]) != int(b["prediction"])
    p = float(binomtest(min(repairs, damages), repairs + damages, .5).pvalue) if repairs + damages else 1.0
    return {"repairs": repairs, "damages": damages, "changed": changed, "net": repairs-damages, "exact_mcnemar_p": p}

ORIGINAL_TRAIN_UPDATE = binary.historical.train_update
ORIGINAL_PRIVATE_DIAGNOSTICS = binary.historical.private_diagnostics
ACTIVE_BINARY_TASK = "tad_binary"

def binary_spec(task: str, trial_id: str = "SP_LRIF") -> dict[str, Any]:
    p = BINARY_PROFILES[task]
    return binary.trial_spec(trial_id, "FIXED", "B1", p["lr"], p["weight_decay"], p["dropout"],
                             rank=p["rank"], multiplier=p["private_multiplier"], ordinal=0)

def binary_trial_context(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    value = dict(base); value["protocol"] = copy.deepcopy(base["protocol"])
    tr = value["protocol"]["training"]
    tr["lr"] = float(spec["lr"]); tr["weight_decay"] = float(spec["weight_decay"]); tr["drop_rate"] = float(spec["dropout"])
    return value

def build_private_reference(context: dict[str, Any], task: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    p = BINARY_PROFILES[task]
    kwargs = binary.historical.model_kwargs(context)
    SET_Random(SEED)
    model = CMEDualBranchModel(**kwargs, cme_arm="c1", adapter_rank=p["rank"], router_hidden=16, modality_embedding_dim=8).to(context["device"])
    audit = {"parameter_count": binary.parameter_count(model), "adapter_count": len(model.private_adapters)}
    require(audit["adapter_count"] == p["modalities"], f"{task}: adapter count changed")
    return model, audit

def build_binary_model(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    task = ACTIVE_BINARY_TASK; p = BINARY_PROFILES[task]
    if spec["arm"] == "B0":
        model, audit = build_private_reference(context, task); audit["reference_role"] = p["trial_id"]
        return model, audit
    kwargs = binary.historical.model_kwargs(context)
    SET_Random(SEED)
    reference = CMEDualBranchModel(**kwargs, cme_arm="c1", adapter_rank=p["rank"], router_hidden=16, modality_embedding_dim=8).to(context["device"])
    post_reference_rng = binary.capture_rng()
    SET_Random(SEED)
    model = SPLRIFDualBranchModel(**kwargs, cme_arm="c1", adapter_rank=p["rank"], router_hidden=16, modality_embedding_dim=8).to(context["device"])
    pre_attach_diff = binary.common_state_max_diff(reference, model)
    require(pre_attach_diff == 0.0, f"{task}: reference/SP common initialization changed")
    del reference
    binary.restore_rng(post_reference_rng)
    attach_rng = binary.capture_rng(); common_before = binary.clone_cpu_state(model)
    model.attach_sp_lrif(INTERACTION_RANK)
    binary.restore_rng(attach_rng)
    common_diff = max(float((tensor - model.state_dict()[name].detach().cpu()).abs().max()) for name, tensor in common_before.items())
    require(common_diff == 0.0, f"{task}: attaching SP changed existing state")
    sp = [(n, q) for n, q in model.named_parameters() if n.startswith("sp_lrif.")]
    expected_sp = 16 * int(model.sp_lrif.dimension)
    require(len(sp) == 4 and sum(q.numel() for _, q in sp) == expected_sp, f"{task}: SP schema changed")
    require(int(torch.count_nonzero(model.sp_lrif.proj_out.weight)) == 0, f"{task}: proj_out not zero")
    model._sp_lrif_initial_state = {n: q.detach().cpu().clone() for n, q in sp}
    audit = {
        "parameter_count": binary.parameter_count(model), "reference_parameter_count": binary.parameter_count(model)-expected_sp,
        "sp_lrif_parameter_count": expected_sp, "fusion_dimension": int(model.sp_lrif.dimension),
        "interaction_rank": 4, "query_count": len(model.label_pools), "ovr_head_count": len(model._Auxi_classifier),
        "adapter_count": len(model.private_adapters), "common_state_max_abs_diff": common_diff if audit_common else "smoke_locked_zero",
        "reference_common_state_max_abs_diff": pre_attach_diff if audit_common else "smoke_locked_zero",
    }
    require(audit["query_count"] == 2 and audit["ovr_head_count"] == 2 and audit["adapter_count"] == p["modalities"], f"{task}: structure changed")
    return model, audit

class ThreeGroupScheduler(binary.historical.RatioPreservingCustomCosineAnnealingLR):
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, private_multiplier: float, eta_min: float, last_epoch: int = -1):
        require(len(optimizer.param_groups) == 3, "Three optimizer groups required")
        self.T_max = EPOCHS; self.eta_min = float(eta_min); self.hold_epoch = 20
        self.multiplier = float(private_multiplier); self.base_initial = float(base_lr)
        self.initial_lrs_locked = [self.base_initial, self.base_initial*self.multiplier, self.base_initial]
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch)
        self.assert_ratio()
    def get_lr(self) -> list[float]:
        if self.last_epoch < self.hold_epoch:
            base = self.base_initial
        else:
            current = self.last_epoch-self.hold_epoch
            base = self.eta_min+(self.base_initial-self.eta_min)*(1+math.cos(math.pi*current/(EPOCHS-self.hold_epoch)))/2
        return [base, base*self.multiplier, base]
    def assert_ratio(self) -> None:
        base, private, sp = [float(g["lr"]) for g in self.optimizer.param_groups]
        tol = max(1e-14, abs(base)*1e-12)
        require(abs(private-base*self.multiplier) <= tol and abs(sp-base) <= tol, "SP/private LR ratio changed")
    def step(self, epoch: int | None = None) -> None:
        super().step(epoch); self.assert_ratio()

def make_binary_training(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    task = ACTIVE_BINARY_TASK; p = BINARY_PROFILES[task]
    model, initialization = build_binary_model(context, spec, audit_common)
    require(spec["arm"] == "B1", "Only fixed SP candidate may train")
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(context["dataset_dict"], context["device"], rate=float(training["loss_rate"]), label_smoothing=float(training["label_smoothing"]))
    named = [(n,q) for n,q in model.named_parameters() if q.requires_grad]
    private = [(n,q) for n,q in named if n.startswith("private_adapters.")]
    sp = [(n,q) for n,q in named if n.startswith("sp_lrif.")]
    base = [(n,q) for n,q in named if not n.startswith(("private_adapters.", "sp_lrif."))]
    sets = [{id(q) for _,q in group} for group in (base,private,sp)]
    require(len(private) == 4*p["modalities"] and len(sp) == 4, f"{task}: optimizer tensor schema changed")
    require(not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]) and set.union(*sets) == {id(q) for _,q in named}, "Optimizer coverage changed")
    optimizer = torch.optim.Adam([
        {"params":[q for _,q in base],"lr":p["lr"],"weight_decay":p["weight_decay"],"group_name":"base"},
        {"params":[q for _,q in private],"lr":p["lr"]*p["private_multiplier"],"weight_decay":p["weight_decay"],"group_name":"private_adapters"},
        {"params":[q for _,q in sp],"lr":p["lr"],"weight_decay":p["weight_decay"],"group_name":"sp_lrif"},
    ])
    scheduler = ThreeGroupScheduler(optimizer,p["lr"],p["private_multiplier"],float(training["scheduler_eta_min"]))
    audit = {**initialization, "base_parameter_tensors":len(base),"adapter_parameter_tensors":len(private),"sp_lrif_parameter_tensors":len(sp),
             "adapter_parameter_names":[n for n,_ in private],"sp_lrif_parameter_names":[n for n,_ in sp],"optimizer_coverage":1.0,
             "optimizer_groups_disjoint":True,"group_lrs":[p["lr"],p["lr"]*p["private_multiplier"],p["lr"]],"weight_decay":p["weight_decay"]}
    return model, criterion, optimizer, scheduler, audit

def sp_train_update(model, criterion, optimizer, features, labels, train_mask, grad_clip):
    loss, gradients, output = ORIGINAL_TRAIN_UPDATE(model, criterion, optimizer, features, labels, train_mask, grad_clip)
    for name, parameter in model.named_parameters():
        if name.startswith("sp_lrif."):
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"Invalid SP gradient: {name}")
            gradients[name] = float(parameter.grad.detach().abs().max().cpu())
    return loss, gradients, output

def sp_diagnostics(context, model, test_mask, cumulative_gradient, initial_adapter):
    private_gradient = {n:v for n,v in cumulative_gradient.items() if n.startswith("private_adapters.")}
    base = ORIGINAL_PRIVATE_DIAGNOSTICS(context,model,test_mask,private_gradient,initial_adapter)
    with torch.no_grad():
        _,_,_,inter = binary.historical.infer(model,context["dataset_data"]["Feature"],return_intermediates=True)
        category, global_message = inter["Y"][test_mask], inter["G"][test_mask]
        agreement, disagreement = inter["sp_lrif_agreement"][test_mask], inter["sp_lrif_disagreement"][test_mask]
        delta, direct = inter["sp_lrif_delta"][test_mask], inter["sp_lrif_sum"][test_mask]
        cosine = torch.nn.functional.cosine_similarity(category,global_message,dim=-1)
        ratio = delta.norm(dim=-1)/(direct.norm(dim=-1)+1e-12)
        enabled = torch.softmax(inter["raw_logits"][test_mask],dim=-1)
        model.sp_lrif.enabled=False
        try: disabled=torch.softmax(binary.historical.infer(model,context["dataset_data"]["Feature"])[0][test_mask],dim=-1)
        finally: model.sp_lrif.enabled=True
        difference=(enabled-disabled).abs()
    gradients={n:float(v) for n,v in cumulative_gradient.items() if n.startswith("sp_lrif.")}
    deltas={n:float((q.detach().cpu()-model._sp_lrif_initial_state[n]).abs().max()) for n,q in model.named_parameters() if n.startswith("sp_lrif.")}
    require(set(gradients)==set(deltas)==set(model._sp_lrif_initial_state),"SP diagnostics schema changed")
    changed=all(math.isfinite(v) and v>0 for v in deltas.values()); collapse=float(ratio.max().cpu())<=1e-6
    require(changed and not collapse,"SP inactive/collapsed")
    return {**base,"category_global_cosine_mean":float(cosine.mean().cpu()),"delta_direct_sum_ratio_mean":float(ratio.mean().cpu()),
            "delta_direct_sum_ratio_max":float(ratio.max().cpu()),"agreement_mean_norm":float(agreement.norm(dim=-1).mean().cpu()),
            "disagreement_mean_norm":float(disagreement.norm(dim=-1).mean().cpu()),"sp_lrif_cumulative_max_gradient":gradients,
            "sp_lrif_parameter_delta":deltas,"sp_lrif_all_parameters_changed":changed,"delta_collapse":collapse,
            "delta_disabled_probability_max_abs_diff":float(difference.max().cpu()),"delta_disabled_probability_mean_abs_diff":float(difference.mean().cpu()),
            "delta_disabled_argmax_changed":int((enabled.argmax(1)!=disabled.argmax(1)).sum().cpu()),"diagnostic_subject_count":int(test_mask.sum().cpu())}

def binary_prediction_rows(context, spec, fold, logits, probabilities, truth):
    anchors=binary.expected_fold_rows(context,fold); rows=[]; positive=BINARY_PROFILES[ACTIVE_BINARY_TASK]["positive_index"]
    require(len(anchors)==len(truth)==len(logits)==len(probabilities),"Binary OOF length changed")
    for a,y,z,p in zip(anchors,truth,logits,probabilities):
        require(int(a["truth"])==int(y),"Binary fold truth changed")
        rows.append({"task_id":context["task_id"],"trial_id":spec["trial_id"],"arm":spec["arm"],"fold":int(fold),
                     "subject_id":a["subject_id"],"original_csv_index":int(a["original_csv_index"]),"feature_sha256":a["feature_sha256"],
                     "truth":int(y),"prediction":int(np.argmax(p)),"logit_0":float(z[0]),"logit_1":float(z[1]),
                     "probability_0":float(p[0]),"probability_1":float(p[1]),"positive_probability":float(p[positive])})
    return rows

def validate_binary_oof(rows,context,arm=None):
    anchors=sorted(binary.expected_subjects(context),key=lambda r:int(r["original_csv_index"]))
    ordered=sorted(rows,key=lambda r:int(r["original_csv_index"])); n=BINARY_PROFILES[ACTIVE_BINARY_TASK]["sample_count"]
    require(len(ordered)==len(anchors)==n and len({r["subject_id"] for r in ordered})==n,"Binary OOF coverage changed")
    positive=BINARY_PROFILES[ACTIVE_BINARY_TASK]["positive_index"]
    for r,a in zip(ordered,anchors):
        require(r["subject_id"]==a["subject_id"] and r["feature_sha256"]==a["feature_sha256"] and int(r["truth"])==int(a["truth"]) and int(r["fold"])==int(a["fold"]),"Binary OOF anchor changed")
        prob=np.asarray([float(r["probability_0"]),float(r["probability_1"])]); require(np.isfinite(prob).all() and abs(float(prob.sum())-1)<=2e-6,"Invalid binary probability")
        require(int(prob.argmax())==int(r["prediction"]) and abs(float(r["positive_probability"])-float(prob[positive]))<=1e-12,"Binary probability direction changed")
        if arm is not None: require(r["arm"]==arm,"Binary arm changed")

def aggregate_sp(spec,fold_summaries,reference_parameters):
    diagnostics=[s["diagnostics"] for s in fold_summaries]; require(len(diagnostics)==10,"Missing mechanism diagnostics")
    sp_names=set(diagnostics[0]["sp_lrif_cumulative_max_gradient"]); private_names=set(diagnostics[0]["adapter_cumulative_max_gradient"])
    total=sum(int(d["diagnostic_subject_count"]) for d in diagnostics)
    value={
        "interaction_rank":4,"category_global_cosine_mean":float(statistics.mean(d["category_global_cosine_mean"] for d in diagnostics)),
        "delta_direct_sum_ratio_mean":float(sum(d["delta_direct_sum_ratio_mean"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
        "delta_direct_sum_ratio_max":float(max(d["delta_direct_sum_ratio_max"] for d in diagnostics)),
        "agreement_mean_norm":float(sum(d["agreement_mean_norm"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
        "disagreement_mean_norm":float(sum(d["disagreement_mean_norm"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
        "sp_lrif_max_gradient_by_tensor":{n:float(max(d["sp_lrif_cumulative_max_gradient"][n] for d in diagnostics)) for n in sorted(sp_names)},
        "private_max_gradient_by_tensor":{n:float(max(d["adapter_cumulative_max_gradient"][n] for d in diagnostics)) for n in sorted(private_names)},
        "sp_lrif_all_parameters_changed_all_folds":bool(all(d["sp_lrif_all_parameters_changed"] for d in diagnostics)),
        "delta_collapse_any_fold":bool(any(d["delta_collapse"] for d in diagnostics)),
        "delta_disabled_probability_max_abs_diff":float(max(d["delta_disabled_probability_max_abs_diff"] for d in diagnostics)),
        "delta_disabled_probability_mean_abs_diff":float(sum(d["delta_disabled_probability_mean_abs_diff"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
        "delta_disabled_argmax_changed":int(sum(d["delta_disabled_argmax_changed"] for d in diagnostics)),
        "private_trained_all_folds":bool(all(d["private_trained"] for d in diagnostics)),"private_collapse_any_fold":bool(any(d["private_collapse"] for d in diagnostics)),
        "reference_parameter_count":int(reference_parameters),"sp_lrif_parameter_count":int(fold_summaries[0]["object_audit"]["parameter_count"]),
        "parameter_delta":int(fold_summaries[0]["object_audit"]["parameter_count"]-reference_parameters),
    }
    require(value["sp_lrif_all_parameters_changed_all_folds"] and not value["delta_collapse_any_fold"],"SP mechanism failed")
    return value

def binary_safety(result):
    counts=result["metrics"]["predicted_counts"]
    collapse=min(int(v) for v in counts.values())<max(2,int(.02*BINARY_PROFILES[ACTIVE_BINARY_TASK]["sample_count"]))
    return {"safe":not collapse,"collapse":collapse,"reasons":["class prediction collapse"] if collapse else []}

def install_binary_profile(task: str) -> dict[str, Any]:
    global ACTIVE_BINARY_TASK
    ACTIVE_BINARY_TASK=task; p=BINARY_PROFILES[task]
    task_root=RESULT/task
    protocol_value=task_protocol(task)
    def current_protocol(): return copy.deepcopy(protocol_value)
    values={"ROOT":ROOT,"EXPERIMENT_ID":EXPERIMENT,"BRANCH":BRANCH,"BASE_COMMIT":BASE_COMMIT,
            "RESULT_DIR":task_root,"PROTOCOL_PATH":CONFIG,"CONFIG_PATH":CONFIG,"INSPECT_PATH":INSPECT,
            "FOLD_MANIFEST_PATH":INSPECT,"SMOKE_DIR":SMOKE/task,"WORK_DIR":task_root/"work","TRIALS_DIR":task_root/"trials"}
    for n,v in values.items(): setattr(binary,n,v)
    for n,v in {"protocol":current_protocol,"build_model":build_binary_model,"make_training_objects":make_binary_training,
                "trial_context":binary_trial_context,"prediction_rows":binary_prediction_rows,"validate_oof":validate_binary_oof,
                "aggregate_mechanism":aggregate_sp,"safety":binary_safety}.items(): setattr(binary,n,v)
    binary.historical.train_update=sp_train_update; binary.historical.private_diagnostics=sp_diagnostics
    return binary.build_context(torch.device("cpu"))

def binary_runtime(task: str, source: str) -> dict[str, Any]:
    p=BINARY_PROFILES[task]
    core={"experiment":EXPERIMENT,"task":task,"source_commit":source,"runner_sha256":file_sha(ROOT/RUNNER_REL),
          "model_sha256":file_sha(ROOT/MODEL_REL),"config_sha256":file_sha(CONFIG),"reference_commit":p["reference_commit"],
          "folds":list(FOLDS),"seed":0,"epochs":400,"device":"cuda:0","selection":"ACC > ROC-AUC > Macro-F1 > earliest epoch"}
    return {**core,"sha256":digest(core)}

def binary_initial_probe(task: str,device:torch.device) -> dict[str,Any]:
    install_binary_profile(task); base=binary.build_context(device); spec=binary_spec(task,"PROBE")
    context=binary_trial_context(base,spec); ref,_=build_private_reference(context,task); model,_,_,_,audit=make_binary_training(context,spec,True)
    features=context["dataset_data"]["Feature"]; ref.eval(); model.eval()
    with torch.no_grad():
        a=ref(features)[0]; b,_,_,inter=model(features,return_intermediates=True)
    logits=float((a-b).abs().max().cpu()); delta=float(inter["sp_lrif_delta"].abs().max().cpu())
    require(logits<=1e-7 and delta==0.0,"Binary initial equivalence failed")
    del ref,model,base,context; torch.cuda.empty_cache()
    return {"initial_logits_max_abs_diff":logits,"initial_delta_max_abs":delta,"object_audit":audit}

def tri_kwargs(context: dict[str, Any]) -> dict[str, Any]:
    c=context["config"]
    return dict(DATASET_Dict=context["dataset_dict"],Herter_Graph=None,Hidden_size=c.Hidden_size,Drop_rate=c.Drop_rate,
                K=c.ChebGCN_K,num_layers=c.num_layers,num_heads=c.num_heads,input_noise_std=c.input_noise_std,
                drop_path=c.drop_path,graph_head=c.Graph_head,graph_layers=c.graph_layers,graph_heads=c.graph_heads,
                graph_beta=c.graph_beta,graph_k_order=c.graph_k_order,graph_alpha=c.graph_alpha,graph_kernel=c.graph_kernel,
                graph_use_graph=False,graph_dropout=c.graph_dropout,graph_hidden=c.graph_hidden,global_word_emb=c.global_word_emb,
                semantic_branch="both",semantic_fusion="add",category_branch_variant="original",query_pool_variant="independent",
                category_branch_fusion="concat",adj_mode="none",label_graph_alpha=0.0,label_graph_topk=0,label_graph_reg_lambda=0.0,
                cme_arm="c1",adapter_rank=8,router_hidden=16,modality_embedding_dim=8)

def build_tri_reference(context: dict[str, Any]):
    model=broad.build_model(context,8); dropout=broad.apply_dropout_multiplier(model,1.1)
    return model,dropout

def build_tri_candidate(context: dict[str, Any],audit_common:bool=False):
    reference,ref_dropout=build_tri_reference(context); post_rng=binary.capture_rng()
    SET_Random(SEED); model=SPLRIFDualBranchModel(**tri_kwargs(context)).to(context["device"]); dropout=broad.apply_dropout_multiplier(model,1.1)
    common=binary.common_state_max_diff(reference,model); require(common==0.0 and dropout==ref_dropout,"A012/SP initialization changed")
    del reference; binary.restore_rng(post_rng); attach_rng=binary.capture_rng(); before=binary.clone_cpu_state(model)
    model.attach_sp_lrif(4); binary.restore_rng(attach_rng)
    attach=max(float((v-model.state_dict()[n].detach().cpu()).abs().max()) for n,v in before.items())
    sp=[(n,q) for n,q in model.named_parameters() if n.startswith("sp_lrif.")]
    require(attach==0.0 and len(sp)==4 and sum(q.numel() for _,q in sp)==1536 and int(torch.count_nonzero(model.sp_lrif.proj_out.weight))==0,"Tri SP schema changed")
    model._sp_lrif_initial_state={n:q.detach().cpu().clone() for n,q in sp}
    audit={"parameter_count":broad.parameter_count(model),"reference_parameter_count":862971,"sp_lrif_parameter_count":1536,
           "fusion_dimension":96,"interaction_rank":4,"common_state_max_abs_diff":common if audit_common else "smoke_locked_zero",
           "attach_state_max_abs_diff":attach if audit_common else "smoke_locked_zero","dropout_modules":dropout,
           "query_count":len(model.label_pools),"ovr_head_count":len(model._Auxi_classifier),"adapter_count":len(model.private_adapters)}
    require(audit["parameter_count"]==864507 and audit["query_count"]==audit["ovr_head_count"]==3 and audit["adapter_count"]==6,"Tri object profile changed")
    return model,audit

def make_tri_training(context:dict[str,Any],audit_common:bool=False):
    model,audit=build_tri_candidate(context,audit_common); criterion=criterion_query_pool_no_orth(context["dataset_dict"],context["device"],label_smoothing=.05)
    named=[(n,q) for n,q in model.named_parameters() if q.requires_grad]
    private=[(n,q) for n,q in named if n.startswith("private_adapters.")]; sp=[(n,q) for n,q in named if n.startswith("sp_lrif.")]
    base=[(n,q) for n,q in named if not n.startswith(("private_adapters.","sp_lrif."))]
    ids=[{id(q) for _,q in group} for group in (base,private,sp)]
    require(len(private)==24 and len(sp)==4 and not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]) and set.union(*ids)=={id(q) for _,q in named},"Tri optimizer partition changed")
    p=TRI_PROFILE; optimizer=torch.optim.Adam([
        {"params":[q for _,q in base],"lr":p["base_lr"],"weight_decay":p["base_weight_decay"],"group_name":"base"},
        {"params":[q for _,q in private],"lr":p["base_lr"]*2,"weight_decay":p["base_weight_decay"],"group_name":"private_adapters"},
        {"params":[q for _,q in sp],"lr":p["base_lr"],"weight_decay":p["base_weight_decay"],"group_name":"sp_lrif"},])
    scheduler=ThreeGroupScheduler(optimizer,p["base_lr"],2.0,float(context["config"].Lr_Min))
    audit.update({"base_parameter_tensors":len(base),"adapter_parameter_tensors":24,"sp_lrif_parameter_tensors":4,
                  "adapter_parameter_names":[n for n,_ in private],"sp_lrif_parameter_names":[n for n,_ in sp],"optimizer_coverage":1.0,"optimizer_groups_disjoint":True,
                  "group_lrs":[p["base_lr"],p["base_lr"]*2,p["base_lr"]]})
    return model,criterion,optimizer,scheduler,audit

def tri_lock(runtime:dict[str,Any],context:dict[str,Any],fold:int)->dict[str,Any]:
    core={"runtime":runtime,"task":"tad_triclass","fold":fold,"spec":TRI_PROFILE,"fold_manifest_sha256":context["fold_manifest"]["sha256"]}
    return {**core,"sha256":digest(core)}

def tri_paths(fold:int)->dict[str,Path]:
    root=RESULT/"tad_triclass"/"work"/f"fold_{fold:02d}"
    return {"root":root,"resume":root/"resume.pt","checkpoint":root/"checkpoint_best.pt","oof":root/"oof_predictions.csv",
            "history":root/"epoch_metrics.csv","summary":root/"summary.json","complete":root/"complete.json"}

def tri_typed_rows(path:Path)->list[dict[str,Any]]:
    rows=[]
    for r in binary.read_csv(path):
        rows.append({"fold":int(r["fold"]),"subject_index":int(r["subject_index"]),"truth":int(r["truth"]),"prediction":int(r["prediction"]),
                     "raw_logit_AD":float(r["raw_logit_AD"]),"raw_logit_CN":float(r["raw_logit_CN"]),"raw_logit_SMCI":float(r["raw_logit_SMCI"]),
                     "adjusted_logit_AD":float(r["adjusted_logit_AD"]),"adjusted_logit_CN":float(r["adjusted_logit_CN"]),"adjusted_logit_SMCI":float(r["adjusted_logit_SMCI"]),
                     "probability_AD":float(r["probability_AD"]),"probability_CN":float(r["probability_CN"]),"probability_SMCI":float(r["probability_SMCI"])})
    return rows

def tri_validate_rows(rows:list[dict[str,Any]],fold:int|None=None)->list[dict[str,Any]]:
    broad.validate_prediction_rows(rows,fold)
    if fold is None: return broad.validate_oof(rows)
    return sorted(rows,key=lambda r:int(r["subject_index"]))

def tri_mechanism(context,model,test_mask,cumulative,initial_private):
    with torch.no_grad():
        _,_,_,inter=model(context["dataset_data"]["Feature"],return_intermediates=True)
        c,g=inter["Y"][test_mask],inter["G"][test_mask]; agreement=inter["sp_lrif_agreement"][test_mask]; disagreement=inter["sp_lrif_disagreement"][test_mask]
        delta,direct=inter["sp_lrif_delta"][test_mask],inter["sp_lrif_sum"][test_mask]
        cosine=torch.nn.functional.cosine_similarity(c,g,dim=-1); ratio=delta.norm(dim=-1)/(direct.norm(dim=-1)+1e-12)
        enabled=torch.softmax(broad.cme.score_logits(inter["raw_logits"][test_mask],context["dataset_dict"]["Label_Weight"],float(context["config"].logit_adjust_tau)),dim=-1)
        model.sp_lrif.enabled=False
        try:
            disabled_raw=model(context["dataset_data"]["Feature"])[0][test_mask]
            disabled=torch.softmax(broad.cme.score_logits(disabled_raw,context["dataset_dict"]["Label_Weight"],float(context["config"].logit_adjust_tau)),dim=-1)
        finally:model.sp_lrif.enabled=True
        diff=(enabled-disabled).abs()
    spgrad={n:float(v) for n,v in cumulative.items() if n.startswith("sp_lrif.")}; privategrad={n:float(v) for n,v in cumulative.items() if n.startswith("private_adapters.")}
    spdelta={n:float((q.detach().cpu()-model._sp_lrif_initial_state[n]).abs().max()) for n,q in model.named_parameters() if n.startswith("sp_lrif.")}
    privatedelta={n:float((q.detach().cpu()-initial_private[n]).abs().max()) for n,q in model.named_parameters() if n.startswith("private_adapters.")}
    collapse=float(ratio.max().cpu())<=1e-6
    require(all(v>0 and math.isfinite(v) for v in spgrad.values()) and all(v>0 and math.isfinite(v) for v in spdelta.values()) and all(v>0 and math.isfinite(v) for v in privategrad.values()) and all(v>0 and math.isfinite(v) for v in privatedelta.values()) and not collapse,"Tri SP/private inactive")
    return {"category_global_cosine_mean":float(cosine.mean().cpu()),"delta_direct_sum_ratio_mean":float(ratio.mean().cpu()),"delta_direct_sum_ratio_max":float(ratio.max().cpu()),
            "agreement_mean_norm":float(agreement.norm(dim=-1).mean().cpu()),"disagreement_mean_norm":float(disagreement.norm(dim=-1).mean().cpu()),
            "sp_lrif_cumulative_max_gradient":spgrad,"sp_lrif_parameter_delta":spdelta,"sp_lrif_all_parameters_changed":True,"delta_collapse":collapse,
            "delta_disabled_probability_max_abs_diff":float(diff.max().cpu()),"delta_disabled_probability_mean_abs_diff":float(diff.mean().cpu()),
            "delta_disabled_argmax_changed":int((enabled.argmax(1)!=disabled.argmax(1)).sum().cpu()),"private_cumulative_max_gradient":privategrad,
            "private_parameter_delta":privatedelta,"private_trained":True,"diagnostic_subject_count":int(test_mask.sum().cpu())}

def load_completed_tri(context:dict[str,Any],runtime:dict[str,Any],fold:int):
    p=tri_paths(fold); required=[p[k] for k in ("checkpoint","oof","history","summary","complete")]
    if not p["complete"].is_file(): return None
    require(all(x.is_file() for x in required),f"Tri fold{fold}: completed artifacts missing")
    lock=tri_lock(runtime,context,fold); marker=read_json(p["complete"]); summary=read_json(p["summary"])
    require(marker["lock"]==lock and marker["complete"] is True and summary["lock"]==lock,"Tri completed lock changed")
    for key in ("checkpoint","oof","history","summary"): require(marker["sha256"][key]==file_sha(p[key]),f"Tri fold{fold}: artifact hash changed")
    rows=tri_typed_rows(p["oof"]); tri_validate_rows(rows,fold)
    expected=broad.fold_expected_subjects(context,fold); require({int(r["subject_index"]):int(r["truth"]) for r in rows}==expected,"Tri fold ownership changed")
    metrics=broad.cme.metrics_from_rows(rows); require(metrics==summary["best_metrics"],"Tri completed metrics changed")
    history=binary.read_csv(p["history"]); require(len(history)==400 and [int(r["epoch"]) for r in history]==list(range(1,401)),"Tri history changed")
    selected=max(history,key=lambda r:(float(r["acc"]),float(r["macro_auc"]),float(r["macro_f1"]),-int(r["epoch"])))
    require(int(selected["epoch"])==int(summary["best_epoch"]),"Tri best epoch changed")
    payload=torch.load(p["checkpoint"],map_location="cpu",weights_only=False); require(payload["lock"]==lock and int(payload["best_epoch"])==int(summary["best_epoch"]),"Tri checkpoint lock changed")
    model,_,_,_,_=make_tri_training(context); model.load_state_dict(payload["model"],strict=True); model.eval()
    labels=context["dataset_data"]["Label"]; test_mask=context["dataset_data"]["Mask"][fold][1]
    with torch.no_grad(): replay=model(context["dataset_data"]["Feature"])[0]
    replay_rows=broad.cme.prediction_rows(fold,replay,labels,test_mask,context["dataset_dict"],context["config"])
    saved={int(r["subject_index"]):r for r in rows}
    for r in replay_rows:
        a=saved[int(r["subject_index"])]
        require(int(a["prediction"])==int(r["prediction"]) and max(abs(float(a[f"probability_{n}"])-float(r[f"probability_{n}"])) for n in broad.CLASS_NAMES)<=1e-6,"Tri checkpoint replay changed")
    del model; torch.cuda.empty_cache(); return summary,rows

def train_tri_fold(context:dict[str,Any],runtime:dict[str,Any],fold:int):
    completed=load_completed_tri(context,runtime,fold)
    if completed is not None:
        print(f"RESUME tad_triclass fold={fold} complete",flush=True); return completed
    p=tri_paths(fold); p["root"].mkdir(parents=True,exist_ok=True); lock=tri_lock(runtime,context,fold)
    allowed={x.name for k,x in p.items() if k!="root"}|{x.name+".tmp" for k,x in p.items() if k!="root"}
    require(not [x for x in p["root"].iterdir() if x.name not in allowed],f"Tri fold{fold}: unknown partial artifact")
    model,criterion,optimizer,scheduler,audit=make_tri_training(context)
    features=context["dataset_data"]["Feature"]; labels=context["dataset_data"]["Label"]; train_mask,test_mask=context["dataset_data"]["Mask"][fold]
    initial_private={n:q.detach().cpu().clone() for n,q in model.named_parameters() if n.startswith("private_adapters.")}
    cumulative={n:0.0 for n,q in model.named_parameters() if n.startswith(("private_adapters.","sp_lrif."))}
    history=[]; best=None; start=1; elapsed_before=0.0; resumed=0
    if p["resume"].is_file():
        value=torch.load(p["resume"],map_location="cpu",weights_only=False); require(value["lock"]==lock,"Tri resume lock changed")
        resumed=int(value["epoch"]); require(0<resumed<=400 and value["optimizer_steps"]==value["scheduler_steps"]==resumed,"Tri resume steps changed")
        model.load_state_dict(value["model"],strict=True); optimizer.load_state_dict(value["optimizer"]); scheduler.load_state_dict(value["scheduler"]); scheduler.assert_ratio()
        require(int(scheduler.last_epoch)==resumed,"Tri resume scheduler changed"); binary.restore_rng(value["rng"])
        history=copy.deepcopy(value["history"]); best=copy.deepcopy(value["best"]); cumulative={n:float(v) for n,v in value["cumulative_gradient"].items()}
        initial_private={n:v.detach().cpu().clone() for n,v in value["initial_private"].items()}; elapsed_before=float(value["elapsed_seconds"]); start=resumed+1
        require([int(r["epoch"]) for r in history]==list(range(1,start)),"Tri resume history changed")
    session=time.perf_counter(); last_lrs=None
    for epoch in range(start,401):
        model.train(); optimizer.zero_grad(set_to_none=True); raw,branches,aux=model(features)
        components=broad.loss_components(criterion,raw,labels,train_mask,aux,.5); loss=components["total"]
        require(bool(torch.isfinite(loss)),f"Tri fold{fold} epoch{epoch}: nonfinite loss"); loss.backward()
        require(broad.gradients_finite(model),f"Tri fold{fold} epoch{epoch}: nonfinite gradient")
        for n,q in model.named_parameters():
            if n in cumulative:
                require(q.grad is not None,f"Tri missing gradient {n}"); cumulative[n]=max(cumulative[n],float(q.grad.detach().abs().max().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(context["config"].grad_clip)); optimizer.step(); scheduler.step(); scheduler.assert_ratio()
        last_lrs=[float(g["lr"]) for g in optimizer.param_groups]
        model.eval()
        with torch.no_grad(): evaluated=model(features)[0]; metrics,selection=broad.cme.selection_metrics(evaluated,labels,test_mask,context["dataset_dict"]["Label_Weight"],float(context["config"].logit_adjust_tau))
        row={"epoch":epoch,"loss":float(loss.detach().cpu()),"correct":int(metrics["correct"]),"acc":float(metrics["acc"]),"macro_auc":float(metrics["macro_auc"]),"macro_f1":float(metrics["macro_f1"]),"bacc":float(metrics["bacc"])}; history.append(row)
        if best is None or tuple(selection)>tuple(best["selection"]):
            best={"epoch":epoch,"selection":list(selection),"metrics":copy.deepcopy(metrics),"model":binary.clone_cpu_state(model)}
        if epoch%20==0 or epoch==400:
            elapsed=elapsed_before+time.perf_counter()-session
            payload={"schema":1,"lock":lock,"epoch":epoch,"optimizer_steps":epoch,"scheduler_steps":epoch,"model":binary.clone_cpu_state(model),
                     "optimizer":copy.deepcopy(optimizer.state_dict()),"scheduler":copy.deepcopy(scheduler.state_dict()),"rng":binary.capture_rng(),"best":copy.deepcopy(best),
                     "cumulative_gradient":cumulative,"initial_private":initial_private,"history":history,"elapsed_seconds":elapsed}
            binary.atomic_torch_save(p["resume"],payload)
    require(best is not None and len(history)==400,"Tri training incomplete"); elapsed=elapsed_before+time.perf_counter()-session
    model.load_state_dict(best["model"],strict=True); model.eval(); torch.cuda.synchronize(); infer_start=time.perf_counter()
    with torch.no_grad(): best_raw=model(features)[0]
    torch.cuda.synchronize(); inference=time.perf_counter()-infer_start
    rows=broad.cme.prediction_rows(fold,best_raw,labels,test_mask,context["dataset_dict"],context["config"]); tri_validate_rows(rows,fold)
    metrics=broad.cme.metrics_from_rows(rows); require(metrics==best["metrics"],"Tri best OOF changed")
    diagnostics=tri_mechanism(context,model,test_mask,cumulative,initial_private)
    checkpoint={"schema":1,"lock":lock,"best_epoch":best["epoch"],"best_metrics":metrics,"model":best["model"],"parameter_count":audit["parameter_count"]}
    binary.atomic_torch_save(p["checkpoint"],checkpoint); write_csv(p["oof"],rows); write_csv(p["history"],history)
    summary={"fold":fold,"lock":lock,"best_epoch":best["epoch"],"best_metrics":metrics,"object_audit":audit,"diagnostics":diagnostics,
             "elapsed_seconds":float(elapsed),"inference_seconds_full_graph":float(inference),"resumed_from_epoch":resumed,"last_group_lrs":last_lrs}
    write_json(p["summary"],summary)
    marker={"complete":True,"lock":lock,"sha256":{k:file_sha(p[k]) for k in ("checkpoint","oof","history","summary")}}
    write_json(p["complete"],marker)
    del model,criterion,optimizer,scheduler; torch.cuda.empty_cache()
    print(f"tad_triclass fold={fold} best={best['epoch']} correct={metrics['correct']}",flush=True)
    return load_completed_tri(context,runtime,fold)

def inspect_payload()->dict[str,Any]:
    binary_refs={}; fold_hashes={}
    for task in BINARY_PROFILES:
        metrics,rows=reference_binary(task); install_binary_profile(task); context=binary.build_context(torch.device("cpu"))
        validate_binary_oof(rows,context,"B1")
        binary_refs[task]={"metrics":metrics,"oof_sha256":hashlib.sha256(git_blob(BINARY_PROFILES[task]["reference_commit"],BINARY_PROFILES[task]["reference_oof"])).hexdigest(),
                           "protocol":task_protocol(task),"reference_commit":BINARY_PROFILES[task]["reference_commit"]}
        fold_hashes[task]=context["fold_manifest"]["sha256"]
        del context
    tri_metrics,tri_rows=reference_tri(); broad.validate_oof(tri_rows)
    tri_context=broad.load_context("cuda:0"); fold_hashes["tad_triclass"]=tri_context["fold_manifest"]["sha256"]
    model_blob=git_blob(BASE_COMMIT,MODEL_REL); model_hash=hashlib.sha256(model_blob).hexdigest()
    require(git("diff","--quiet",BASE_COMMIT,"HEAD","--",MODEL_REL,check=False).returncode==0 and git("diff","--quiet","--",MODEL_REL,check=False).returncode==0,"Frozen SP-LRIF module differs from ABIDE source")
    abide_summary=blob_json(REFERENCE_COMMITS["abide"],"experiments/sp_lrif_abide_v1/summary.json")
    require(int(abide_summary["sp_lrif"]["metrics"]["correct"])==780,"ABIDE SP anchor changed")
    core={"experiment":EXPERIMENT,"branch":BRANCH,"base_commit":BASE_COMMIT,"sp_lrif_source_commit":BASE_COMMIT,
          "sp_lrif_model_sha256":model_hash,"interaction_rank":4,"binary_references":binary_refs,
          "tad_triclass_reference":{"metrics":tri_metrics,"oof_sha256":file_sha(ROOT/"experiments/c1_broad_hparam_search_v1/best_oof_predictions.csv"),"reference_commit":REFERENCE_COMMITS["tad_triclass"],"spec":TRI_PROFILE},
          "abide_existing":{"reference_commit":REFERENCE_COMMITS["abide"],"pre_sp_correct":777,"sp_lrif_correct":780,"net":3,"hard_classification":"ACC_POSITIVE","ranking":"RANKING_MIXED"},
          "fold_manifest_sha256":fold_hashes,"folds":list(FOLDS),"seed_per_fold":0,"epochs":400,
          "config_sha256":file_sha(CONFIG),"runner_sha256":file_sha(ROOT/RUNNER_REL)}
    del tri_context; torch.cuda.empty_cache(); return {**core,"sha256":digest(core)}

def run_inspect()->None:
    value=inspect_payload(); write_json(INSPECT,value)
    print(json.dumps({"inspect":"PASS","tasks":["tad_binary","abide5","tad_triclass"],"SP_model_sha256":value["sp_lrif_model_sha256"]},indent=2))

def validate_inspect()->None:
    require(INSPECT.is_file() and read_json(INSPECT)==inspect_payload(),"Inspect manifest drifted")

def source_gate()->str:
    require(git("branch","--show-current").stdout.strip()==BRANCH,"Wrong branch")
    require(git("diff","--quiet",check=False).returncode==0 and git("diff","--cached","--quiet",check=False).returncode==0,"Tracked/index source drifted")
    head=git("rev-parse","HEAD").stdout.strip(); require(head!=BASE_COMMIT and git("merge-base","--is-ancestor",BASE_COMMIT,head,check=False).returncode==0,"Source ancestry changed")
    changed={x.strip().replace("\\","/") for x in git("diff","--name-only",BASE_COMMIT,head).stdout.splitlines() if x.strip()}
    require(changed==set(REQUIRED_SOURCE),f"Source scope changed: {sorted(changed)}")
    for rel in REQUIRED_SOURCE: require(git("ls-files","--error-unmatch","--",rel,check=False).returncode==0,f"Untracked source: {rel}")
    for rel in LOCKED: require(git("diff","--quiet",BASE_COMMIT,"HEAD","--",rel,check=False).returncode==0,f"Frozen dependency changed: {rel}")
    validate_inspect(); return head

def runtime_lock(source:str)->dict[str,Any]:
    core={"experiment":EXPERIMENT,"source_commit":source,"runner_sha256":file_sha(ROOT/RUNNER_REL),"model_sha256":file_sha(ROOT/MODEL_REL),
          "config_sha256":file_sha(CONFIG),"inspect_sha256":file_sha(INSPECT),"reference_commits":REFERENCE_COMMITS,
          "folds":list(FOLDS),"seed":0,"epochs":400,"device":"cuda:0"}
    return {**core,"sha256":digest(core)}

def smoke_binary(task:str,device:torch.device,runtime:dict[str,Any])->dict[str,Any]:
    install_binary_profile(task); base=binary.build_context(device); spec=binary_spec(task,"SMOKE_SP_LRIF"); context=binary_trial_context(base,spec)
    reference,_=build_private_reference(context,task); model,criterion,optimizer,scheduler,audit=make_binary_training(context,spec,True)
    features=context["dataset_data"]["Feature"]; labels=context["dataset_data"]["Label"]; labels_before=labels.detach().clone(); train_mask=context["dataset_data"]["Mask"][0][0]
    reference.eval(); model.eval()
    with torch.no_grad(): old=reference(features)[0]; new,_,_,inter=model(features,return_intermediates=True)
    initial=float((old-new).abs().max().cpu()); initial_delta=float(inter["sp_lrif_delta"].abs().max().cpu()); require(initial<=1e-7 and initial_delta==0,"Binary smoke equivalence failed")
    initial_sp={n:q.detach().cpu().clone() for n,q in model.named_parameters() if n.startswith("sp_lrif.")}; gradients={n:0.0 for n in initial_sp}; by_epoch=[]; losses=[]; simplex=[]
    for epoch in range(1,4):
        loss,current,_=sp_train_update(model,criterion,optimizer,features,labels,train_mask,float(context["protocol"]["training"]["grad_clip"])); scheduler.step(); scheduler.assert_ratio()
        now={n:float(v) for n,v in current.items() if n.startswith("sp_lrif.")}; by_epoch.append(now)
        for n,v in now.items():gradients[n]=max(gradients[n],v)
        losses.append(float(loss)); model.eval()
        with torch.no_grad():prob=torch.softmax(model(features)[0],dim=-1)
        simplex.append(float((prob.sum(1)-1).abs().max().cpu()))
    deltas={n:float((q.detach().cpu()-initial_sp[n]).abs().max()) for n,q in model.named_parameters() if n in initial_sp}
    require(by_epoch[0]["sp_lrif.proj_out.weight"]>0 and all(v>0 and math.isfinite(v) for v in gradients.values()) and all(v>0 and math.isfinite(v) for v in deltas.values()),"Binary SP did not activate by epoch3")
    require(torch.equal(labels,labels_before) and max(simplex)<=2e-6,"Binary smoke numeric failure")
    path=SMOKE/task/"checkpoint_roundtrip.pt"; checkpoint={"runtime":runtime,"task":task,"epoch":3,"model":binary.clone_cpu_state(model),"optimizer":copy.deepcopy(optimizer.state_dict()),"scheduler":copy.deepcopy(scheduler.state_dict()),"rng":binary.capture_rng()}; binary.atomic_torch_save(path,checkpoint)
    restored,_,ropt,rsched,_=make_binary_training(context,spec); payload=torch.load(path,map_location="cpu",weights_only=False); restored.load_state_dict(payload["model"],strict=True); ropt.load_state_dict(payload["optimizer"]); rsched.load_state_dict(payload["scheduler"]); rsched.assert_ratio(); model.eval();restored.eval()
    with torch.no_grad(): reload=float((model(features)[0]-restored(features)[0]).abs().max().cpu())
    require(reload==0,"Binary smoke checkpoint reload failed")
    value={"task":task,"runtime":runtime,"fold":0,"epochs":3,"initial_logits_max_abs_diff":initial,"initial_delta_max_abs":initial_delta,"losses":losses,
           "probability_sum_max_abs_error":max(simplex),"sp_gradient_by_epoch":by_epoch,"sp_max_gradient_by_tensor":gradients,"sp_parameter_delta_by_tensor":deltas,
           "proj_out_step1_gradient_nonzero":True,"all_projections_active_by_epoch3":True,"optimizer_audit":audit,"checkpoint_sha256":file_sha(path),"checkpoint_reload_logits_max_abs_diff":reload}
    del reference,model,restored,base,context;torch.cuda.empty_cache();return value

def smoke_tri(device:torch.device,runtime:dict[str,Any])->dict[str,Any]:
    context=broad.load_context("cuda:0"); reference,_=build_tri_reference(context); model,criterion,optimizer,scheduler,audit=make_tri_training(context,True)
    features=context["dataset_data"]["Feature"];labels=context["dataset_data"]["Label"];train_mask=context["dataset_data"]["Mask"][0][0];reference.eval();model.eval()
    with torch.no_grad():old=reference(features)[0];new,_,_,inter=model(features,return_intermediates=True)
    initial=float((old-new).abs().max().cpu());delta=float(inter["sp_lrif_delta"].abs().max().cpu());require(initial<=1e-7 and delta==0,"Tri smoke equivalence failed")
    initial_sp={n:q.detach().cpu().clone() for n,q in model.named_parameters() if n.startswith("sp_lrif.")}; gradients={n:0.0 for n in initial_sp};by_epoch=[];losses=[];simplex=[]
    for epoch in range(1,4):
        model.train();optimizer.zero_grad(set_to_none=True);raw,_,aux=model(features);loss=broad.loss_components(criterion,raw,labels,train_mask,aux,.5)["total"];loss.backward()
        now={n:float(q.grad.detach().abs().max().cpu()) for n,q in model.named_parameters() if n.startswith("sp_lrif.")};by_epoch.append(now)
        for n,v in now.items():gradients[n]=max(gradients[n],v)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();losses.append(float(loss.detach().cpu()));model.eval()
        with torch.no_grad():prob=torch.softmax(model(features)[0],dim=-1)
        simplex.append(float((prob.sum(1)-1).abs().max().cpu()))
    changes={n:float((q.detach().cpu()-initial_sp[n]).abs().max()) for n,q in model.named_parameters() if n in initial_sp}
    require(by_epoch[0]["sp_lrif.proj_out.weight"]>0 and all(v>0 and math.isfinite(v) for v in gradients.values()) and all(v>0 and math.isfinite(v) for v in changes.values()),"Tri SP did not activate")
    path=SMOKE/"tad_triclass"/"checkpoint_roundtrip.pt";payload={"runtime":runtime,"task":"tad_triclass","epoch":3,"model":binary.clone_cpu_state(model),"optimizer":copy.deepcopy(optimizer.state_dict()),"scheduler":copy.deepcopy(scheduler.state_dict())};binary.atomic_torch_save(path,payload)
    restored,_,ropt,rsched,_=make_tri_training(context);loaded=torch.load(path,map_location="cpu",weights_only=False);restored.load_state_dict(loaded["model"],strict=True);ropt.load_state_dict(loaded["optimizer"]);rsched.load_state_dict(loaded["scheduler"]);rsched.assert_ratio();model.eval();restored.eval()
    with torch.no_grad():reload=float((model(features)[0]-restored(features)[0]).abs().max().cpu())
    require(reload==0 and max(simplex)<=2e-6,"Tri smoke reload/numeric failed")
    value={"task":"tad_triclass","runtime":runtime,"fold":0,"epochs":3,"initial_logits_max_abs_diff":initial,"initial_delta_max_abs":delta,"losses":losses,
           "probability_sum_max_abs_error":max(simplex),"sp_gradient_by_epoch":by_epoch,"sp_max_gradient_by_tensor":gradients,"sp_parameter_delta_by_tensor":changes,
           "proj_out_step1_gradient_nonzero":True,"all_projections_active_by_epoch3":True,"optimizer_audit":audit,"checkpoint_sha256":file_sha(path),"checkpoint_reload_logits_max_abs_diff":reload}
    del reference,model,restored,context;torch.cuda.empty_cache();return value

def run_smoke(device_text:str)->None:
    require(device_text=="cuda:0" and torch.cuda.is_available(),"Smoke requires cuda:0")
    source=source_gate(); runtime=runtime_lock(source); SMOKE.mkdir(parents=True,exist_ok=True)
    reports={}
    for task in ("tad_binary","abide5"):
        reports[task]=smoke_binary(task,torch.device(device_text),binary_runtime(task,source))
    reports["tad_triclass"]=smoke_tri(torch.device(device_text),runtime)
    config_core={"runtime":runtime,"tasks":list(reports),"fold":0,"epochs":3}; smoke_config={**config_core,"sha256":digest(config_core)}
    write_json(SMOKE/"smoke_config.json",smoke_config)
    report_core={"runtime":runtime,"smoke_config_sha256":file_sha(SMOKE/"smoke_config.json"),"tasks":reports,"passed":True,
                 "device":torch.cuda.get_device_name(0),"torch_version":torch.__version__,"cuda_version":torch.version.cuda}
    write_json(SMOKE/"smoke_report.json",{**report_core,"sha256":digest(report_core)})
    print(json.dumps({"smoke":"PASS","tasks":{k:v["losses"] for k,v in reports.items()}},indent=2))

def validate_smoke(runtime:dict[str,Any])->dict[str,Any]:
    cp=SMOKE/"smoke_config.json";rp=SMOKE/"smoke_report.json";require(cp.is_file() and rp.is_file(),"Smoke artifacts missing")
    cfg=read_json(cp);report=read_json(rp);core={k:v for k,v in report.items() if k!="sha256"}
    require(cfg["runtime"]==runtime and report["runtime"]==runtime and report["sha256"]==digest(core) and report["smoke_config_sha256"]==file_sha(cp),"Smoke lock changed")
    for task,value in report["tasks"].items():
        checkpoint=SMOKE/task/"checkpoint_roundtrip.pt";require(checkpoint.is_file() and value["checkpoint_sha256"]==file_sha(checkpoint),f"{task}: smoke checkpoint changed")
        require(value["initial_logits_max_abs_diff"]<=1e-7 and value["initial_delta_max_abs"]==0 and value["all_projections_active_by_epoch3"] and value["checkpoint_reload_logits_max_abs_diff"]==0,f"{task}: smoke gate changed")
    return report

def aggregate_tri(context:dict[str,Any],runtime:dict[str,Any])->tuple[dict[str,Any],list[dict[str,Any]]]:
    summaries=[];rows=[]
    for fold in FOLDS:
        summary,group=train_tri_fold(context,runtime,fold);summaries.append(summary);rows.extend(group)
    rows=tri_validate_rows(rows);metrics=broad.cme.metrics_from_rows(rows);fold_acc=[float(s["best_metrics"]["acc"]) for s in summaries]
    diagnostics=[s["diagnostics"] for s in summaries];total=sum(d["diagnostic_subject_count"] for d in diagnostics);sp_names=set(diagnostics[0]["sp_lrif_cumulative_max_gradient"])
    mechanism={"interaction_rank":4,"category_global_cosine_mean":float(statistics.mean(d["category_global_cosine_mean"] for d in diagnostics)),
               "delta_direct_sum_ratio_mean":float(sum(d["delta_direct_sum_ratio_mean"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
               "delta_direct_sum_ratio_max":float(max(d["delta_direct_sum_ratio_max"] for d in diagnostics)),
               "agreement_mean_norm":float(sum(d["agreement_mean_norm"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
               "disagreement_mean_norm":float(sum(d["disagreement_mean_norm"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
               "sp_lrif_max_gradient_by_tensor":{n:float(max(d["sp_lrif_cumulative_max_gradient"][n] for d in diagnostics)) for n in sorted(sp_names)},
               "sp_lrif_all_parameters_changed_all_folds":bool(all(d["sp_lrif_all_parameters_changed"] for d in diagnostics)),
               "delta_collapse_any_fold":bool(any(d["delta_collapse"] for d in diagnostics)),
               "delta_disabled_probability_max_abs_diff":float(max(d["delta_disabled_probability_max_abs_diff"] for d in diagnostics)),
               "delta_disabled_probability_mean_abs_diff":float(sum(d["delta_disabled_probability_mean_abs_diff"]*d["diagnostic_subject_count"] for d in diagnostics)/total),
               "delta_disabled_argmax_changed":int(sum(d["delta_disabled_argmax_changed"] for d in diagnostics)),
               "private_trained_all_folds":bool(all(d["private_trained"] for d in diagnostics)),"reference_parameter_count":862971,"parameter_count":864507,"parameter_delta":1536}
    matrix=np.asarray(metrics["confusion_matrix"],dtype=int);boundaries={"AD_SMCI":int(matrix[0,2]+matrix[2,0]),"CN_SMCI":int(matrix[1,2]+matrix[2,1]),"AD_CN":int(matrix[0,1]+matrix[1,0])}
    predicted={name:sum(int(r["prediction"])==i for r in rows) for i,name in enumerate(broad.CLASS_NAMES)}
    result={"task":"tad_triclass","metrics":metrics,"fold_acc_mean":float(statistics.mean(fold_acc)),"fold_acc_sample_std":float(statistics.stdev(fold_acc)),
            "fold_metrics":[{"fold":s["fold"],"best_epoch":s["best_epoch"],"correct":s["best_metrics"]["correct"],"acc":s["best_metrics"]["acc"]} for s in summaries],
            "parameter_count":864507,"training_time_seconds":float(sum(s["elapsed_seconds"] for s in summaries)),"inference_time_seconds":float(sum(s["inference_seconds_full_graph"] for s in summaries)),
            "predicted_counts":predicted,"boundary_errors":boundaries,"mechanism":mechanism}
    return result,rows

def classification_labels(task:str,result:dict[str,Any],reference:dict[str,Any],comparison:dict[str,Any])->dict[str,Any]:
    m=result["metrics"]; delta=int(m["correct"])-int(reference["correct"])
    hard="ACC_POSITIVE" if delta>0 and comparison["repairs"]>comparison["damages"] else ("ACC_TIED" if delta==0 else "ACC_NEGATIVE")
    ranking_keys=["roc_auc","pr_auc","macro_f1","bacc"] if task!="tad_triclass" else ["macro_auc","macro_f1","bacc"]
    declines={k:float(m[k])-float(reference[k]) for k in ranking_keys}; worsened=any(v<-.005 for v in declines.values())
    ranking="RANKING_MIXED" if delta>0 and worsened else ("RANKING_NEGATIVE" if delta<=0 and worsened else "RANKING_SAFE")
    if task=="tad_binary":
        target="TARGET_REACHED" if m["correct"]>=523 else ("ACC_POSITIVE" if m["correct"]>=520 else "BELOW_TARGET")
        safe=float(m["bacc"])>=float(reference["bacc"])-.003 and float(m["sen"])>=float(reference["sen"])-.02
    elif task=="abide5":
        target="TARGET_REACHED" if m["correct"]>=787 else ("ACC_POSITIVE" if m["correct"]>=770 else "BELOW_TARGET")
        safe=float(m["bacc"])>=float(reference["bacc"])-.003 and float(m["sen"])>=float(reference["sen"])-.01 and float(m["spe"])>=float(reference["spe"])-.01
    else:
        target="STRONG_GAIN" if m["correct"]>=564 else ("ACC_POSITIVE" if m["correct"]>=563 else "BELOW_TARGET")
        safe=result["boundary_errors"]["AD_CN"]==0 and float(m["macro_f1"])>=float(reference["macro_f1"])-.003 and float(m["bacc"])>=float(reference["bacc"])-.003
    return {"hard_classification":hard,"ranking_balance":ranking,"target":target,"target_safety_met":bool(safe),"correct_delta":delta,"metric_deltas":declines}

def cross_decision(tasks:dict[str,Any],abide:dict[str,Any])->dict[str,Any]:
    names=("tad_binary","abide5","tad_triclass");deltas=[tasks[n]["labels"]["correct_delta"] for n in names]
    collapse=any(tasks[n]["collapse"] for n in names);bacc_bad=any(tasks[n]["labels"]["metric_deltas"]["bacc"]<-.01 for n in names)
    inactive=any(tasks[n]["result"]["mechanism"].get("delta_collapse_any_fold",False) or not tasks[n]["result"]["mechanism"].get("sp_lrif_all_parameters_changed_all_folds",False) for n in names)
    adcn=tasks["tad_triclass"]["result"]["boundary_errors"]["AD_CN"]>0; positives=sum(x>0 for x in deltas);new_sum=sum(deltas);four_sum=new_sum+3
    supported=positives>=2 and min(deltas)>=-1 and new_sum>=3 and four_sum>=6 and not(collapse or bacc_bad or inactive or adcn)
    near=positives==1 and all(x>=-1 for x in deltas) and four_sum>0 and not(collapse or bacc_bad or inactive or adcn)
    stop=sum(x<0 for x in deltas)>=2 or bacc_bad or collapse or adcn or inactive
    if supported:token="SP_LRIF_CROSS_TASK_SUPPORTED"
    elif stop:token="SP_LRIF_CROSS_TASK_STOP"
    elif near:token="SP_LRIF_CROSS_TASK_NEAR"
    else:token="SP_LRIF_TASK_DEPENDENT"
    return {"decision":token,"new_task_correct_deltas":dict(zip(names,deltas)),"new_task_net_sum":new_sum,"four_task_net_sum":four_sum,"new_task_positive_count":positives,
            "checks":{"collapse":collapse,"bacc_drop_over_0_01":bacc_bad,"inactive":inactive,"tad_triclass_ad_cn_error":adcn},"abide_existing":abide}

def run_binary_formal(task:str,source:str)->dict[str,Any]:
    install_binary_profile(task);base=binary.build_context(torch.device("cuda:0"));spec=binary_spec(task);runtime=binary_runtime(task,source)
    result=binary.train_trial(spec,base,runtime);path=RESULT/task/"trials"/spec["trial_id"]/"oof_predictions.csv";rows=binary.parse_trial_rows(path)
    validate_binary_oof(rows,binary_trial_context(base,spec),"B1");reference,refrows=reference_binary(task);comparison=paired(refrows,rows,"subject_id")
    require(comparison["net"]==int(result["metrics"]["correct"])-int(reference["correct"]),f"{task}: paired delta changed")
    labels=classification_labels(task,result,reference,comparison);collapse=bool(result["safety"]["collapse"])
    write_csv(RESULT/task/"oof_predictions.csv",sorted(rows,key=lambda r:int(r["original_csv_index"])))
    write_json(RESULT/task/"fold_metrics.json",{"task":task,"fold_metrics":result["fold_metrics"]})
    write_json(RESULT/task/"mechanism_diagnostics.json",result["mechanism"])
    value={"task":task,"result":result,"reference":reference,"comparison":comparison,"labels":labels,"collapse":collapse,
           "oof_sha256":file_sha(RESULT/task/"oof_predictions.csv")}
    write_json(RESULT/task/"result.json",value);del base;torch.cuda.empty_cache();return value

def run_tri_formal(source:str,runtime:dict[str,Any])->dict[str,Any]:
    context=broad.load_context("cuda:0");result,rows=aggregate_tri(context,runtime);reference,refrows=reference_tri();comparison=paired(refrows,rows,"subject_index")
    require(comparison["net"]==int(result["metrics"]["correct"])-562,"Tri paired delta changed")
    labels=classification_labels("tad_triclass",result,reference,comparison);collapse=min(result["predicted_counts"].values())<10
    write_csv(RESULT/"tad_triclass"/"oof_predictions.csv",rows);write_json(RESULT/"tad_triclass"/"fold_metrics.json",{"task":"tad_triclass","fold_metrics":result["fold_metrics"]})
    write_json(RESULT/"tad_triclass"/"mechanism_diagnostics.json",result["mechanism"])
    value={"task":"tad_triclass","result":result,"reference":reference,"comparison":comparison,"labels":labels,"collapse":collapse,
           "oof_sha256":file_sha(RESULT/"tad_triclass"/"oof_predictions.csv")}
    write_json(RESULT/"tad_triclass"/"result.json",value);del context;torch.cuda.empty_cache();return value

def fmt(x:Any)->str:
    return f"{float(x):.7f}" if isinstance(x,(float,np.floating)) else str(x)

def render_report(runtime:dict[str,Any],tasks:dict[str,Any],abide:dict[str,Any],decision:dict[str,Any],wall:float)->str:
    lines=["# SP-LRIF Cross-Task Validation v1","","Fixed structure: the exact ABIDE SP-LRIF formula, interaction rank 4, zero-initialized bias-free output projection. No search or second version was run.","",
           f"Source commit: `{runtime['source_commit']}`  ",f"Decision: `{decision['decision']}`  ",f"Formal wall time: {wall:.3f} s","","## Results",""]
    for task in ("tad_binary","abide5","tad_triclass"):
        item=tasks[task];m=item["result"]["metrics"];r=item["reference"];c=item["comparison"];lab=item["labels"]
        lines += [f"### {task}","",f"- Correct: {m['correct']}/{m.get('n',535 if task=='tad_binary' else 864 if task=='abide5' else 598)} (reference {r['correct']}, delta {lab['correct_delta']:+d})",
                  f"- Labels: `{lab['hard_classification']}` / `{lab['ranking_balance']}`; task target `{lab['target']}`, target safety={lab['target_safety_met']}",
                  f"- ACC={fmt(m['acc'])}; Macro-F1={fmt(m['macro_f1'])}; BACC={fmt(m['bacc'])}; Weighted-F1={fmt(m['weighted_f1'])}"]
        if task!="tad_triclass":
            lines += [f"- pooled ROC-AUC={fmt(m['roc_auc'])}; PR-AUC={fmt(m['pr_auc'])}; SEN={fmt(m['sen'])}; SPE={fmt(m['spe'])}",
                      f"- Confusion={m['confusion_matrix']}; predicted={m['predicted_counts']}",
                      f"- Fold ROC-AUC={item['result']['fold_roc_auc_mean']:.7f} +/- {item['result']['fold_roc_auc_sample_std']:.7f}; fold ACC={item['result']['fold_acc_mean']:.7f} +/- {item['result']['fold_acc_sample_std']:.7f}"]
        else:
            lines += [f"- Probability Macro-AUC={fmt(m['macro_auc'])}; confusion={m['confusion_matrix']}; predicted={item['result']['predicted_counts']}",
                      f"- Boundary errors={item['result']['boundary_errors']}; fold ACC={item['result']['fold_acc_mean']:.7f} +/- {item['result']['fold_acc_sample_std']:.7f}"]
        lines += [f"- repairs/damages/changed={c['repairs']}/{c['damages']}/{c['changed']}; exact McNemar p={c['exact_mcnemar_p']:.7g}",
                  f"- Parameter count={item['result']['parameter_count']}; train={item['result']['training_time_seconds']:.3f}s; inference={item['result']['inference_time_seconds']:.6f}s",
                  f"- Mechanism: cosine={item['result']['mechanism']['category_global_cosine_mean']:.7f}; delta ratio mean/max={item['result']['mechanism']['delta_direct_sum_ratio_mean']:.7f}/{item['result']['mechanism']['delta_direct_sum_ratio_max']:.7f}; agreement/disagreement norm={item['result']['mechanism']['agreement_mean_norm']:.7f}/{item['result']['mechanism']['disagreement_mean_norm']:.7f}; disabled-delta argmax changes={item['result']['mechanism']['delta_disabled_argmax_changed']}",""]
    lines += ["### Existing ABIDE result","",f"- Pre-SP best: 777/871; SP-LRIF: 780/871; net +3; `ACC_POSITIVE` / `RANKING_MIXED`.","",
              "## Cross-task decision","",f"New-task deltas: {decision['new_task_correct_deltas']}; new-task sum={decision['new_task_net_sum']:+d}; four-task sum={decision['four_task_net_sum']:+d}.","",
              "## Required answers","",
              f"1. TADPOLE binary Correct improved: {'yes' if tasks['tad_binary']['labels']['correct_delta']>0 else 'no'}.",
              f"2. ABIDE-5 Correct improved: {'yes' if tasks['abide5']['labels']['correct_delta']>0 else 'no'}.",
              f"3. TADPOLE tri-class reached 563/564: {'yes' if tasks['tad_triclass']['result']['metrics']['correct']>=563 else 'no'}.",
              f"4. New tasks with hard gains: {decision['new_task_positive_count']}.",
              f"5. Four-task total Correct net: {decision['four_task_net_sum']:+d}.",
              "6. ACC gains with AUC/PR-AUC decline are identified by each task's RANKING_MIXED label above.",
              "7. Agreement and disagreement activation is reported per task; all four projection tensors must have nonzero gradients and parameter changes in every fold.",
              "8. Direct inference-time influence is the disabled-delta argmax count; any remaining gain is trajectory-mediated and is not described as a direct delta repair.",
              f"9. Cross-task effectiveness: {decision['decision']=='SP_LRIF_CROSS_TASK_SUPPORTED'}.",
              f"10. SP_LRIF_CROSS_TASK_SUPPORTED met: {decision['decision']=='SP_LRIF_CROSS_TASK_SUPPORTED'}.",
              f"11. Final-paper module recommendation follows `{decision['decision']}` and does not authorize formula changes.",
              "12. Retain the task reference model wherever the SP candidate is negative; retain SP only where the hard/ranking trade-off is accepted by the preregistered decision.","",
              "## Reproduction","",f"Training reproduction must use a clean local branch named `{BRANCH}` pointing at source commit `{runtime['source_commit']}` (not the later result commit):","",
              "```text",f'"{sys.executable}" -u -B scripts/run_sp_lrif_cross_task_v1.py inspect',f'"{sys.executable}" -u -B scripts/run_sp_lrif_cross_task_v1.py smoke --device cuda:0',f'"{sys.executable}" -u -B scripts/run_sp_lrif_cross_task_v1.py formal --task all --device cuda:0',"```",""]
    return "\n".join(lines)

def run_formal(task_choice:str,device_text:str)->None:
    require(device_text=="cuda:0" and torch.cuda.is_available(),"Formal requires cuda:0")
    source=source_gate();runtime=runtime_lock(source);validate_smoke(runtime);started=time.perf_counter();tasks={}
    selected=("tad_binary","abide5","tad_triclass") if task_choice=="all" else (task_choice,)
    for task in selected:
        tasks[task]=run_binary_formal(task,source) if task in BINARY_PROFILES else run_tri_formal(source,runtime)
    if task_choice!="all":
        print(json.dumps({"formal":task_choice,"status":"COMPLETE"},indent=2));return
    abide_summary=blob_json(REFERENCE_COMMITS["abide"],"experiments/sp_lrif_abide_v1/summary.json")
    abide={"reference_commit":REFERENCE_COMMITS["abide"],"pre_sp_correct":777,"sp_lrif_correct":780,"net":3,"hard_classification":"ACC_POSITIVE","ranking":"RANKING_MIXED",
           "metrics":abide_summary["sp_lrif"]["metrics"]}
    decision=cross_decision(tasks,abide);wall=time.perf_counter()-started
    write_json(RESULT/"comparisons.json",{k:{"comparison":v["comparison"],"labels":v["labels"]} for k,v in tasks.items()})
    write_json(RESULT/"mechanism_diagnostics.json",{k:v["result"]["mechanism"] for k,v in tasks.items()})
    formal_core={"runtime":runtime,"tasks":["tad_binary","abide5","tad_triclass"],"smoke_report_sha256":file_sha(SMOKE/"smoke_report.json")};write_json(RESULT/"formal_config.json",{**formal_core,"sha256":digest(formal_core)})
    report=render_report(runtime,tasks,abide,decision,wall);binary.atomic_write_text(RESULT/"REPORT.md",report)
    summary_core={"experiment":EXPERIMENT,"runtime":runtime,"tasks":tasks,"abide_existing":abide,"decision":decision,"formal_wall_seconds":wall,
                  "artifact_sha256":{"REPORT.md":file_sha(RESULT/"REPORT.md"),"comparisons.json":file_sha(RESULT/"comparisons.json"),"mechanism_diagnostics.json":file_sha(RESULT/"mechanism_diagnostics.json"),"formal_config.json":file_sha(RESULT/"formal_config.json")}}
    write_json(RESULT/"summary.json",{**summary_core,"sha256":digest(summary_core)})
    print(json.dumps({"formal":"COMPLETE","decision":decision["decision"],"deltas":decision["new_task_correct_deltas"]},indent=2))

def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True);sub.add_parser("inspect")
    smoke=sub.add_parser("smoke");smoke.add_argument("--device",default="cuda:0")
    formal=sub.add_parser("formal");formal.add_argument("--task",choices=["all","tad_binary","abide5","tad_triclass"],default="all");formal.add_argument("--device",default="cuda:0")
    return parser.parse_args()

def main()->None:
    args=parse_args()
    if args.mode=="inspect":run_inspect()
    elif args.mode=="smoke":run_smoke(args.device)
    else:run_formal(args.task,args.device)

if __name__=="__main__":main()
