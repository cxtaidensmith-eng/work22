"""Registered ABIDE-5 ADS/CN task-level hyperparameter search."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Loss import criterion_lossv2
from Model import HeterGraph_Model_Kmeans
from Model.cme_dual_branch import CMEDualBranchModel
from Utils import CustomCosineAnnealingLR, SET_Random
from scripts import run_abide_hparam_search_v1 as parent

engine = parent.engine
_BASE_PREDICTION_ROWS = engine.prediction_rows
EXPERIMENT_ID = "abide5_hparam_search_v1"
BRANCH = "experiment/abide5-hparam-search-v1"
BASE_COMMIT = "c9c7a289145a87fbab355e85828fa5f62f9819b0"
HISTORICAL_SOURCE_COMMIT = "17004d29714471b5e1548e04bac0072e854e7e18"
HISTORICAL_RESULT_COMMIT = "e7d18205ddfefaf898a81c999cce15b348331206"
RESULT_DIR = ROOT / "experiments" / EXPERIMENT_ID
PROTOCOL_PATH = RESULT_DIR / "protocol.json"
CONFIG_PATH = RESULT_DIR / "experiment_config.json"
INSPECT_PATH = RESULT_DIR / "inspect_manifest.json"
FOLD_MANIFEST_PATH = RESULT_DIR / "fold_manifest.json"
REGISTERED_SEARCH_PATH = RESULT_DIR / "registered_search.json"
HISTORICAL_DIR = RESULT_DIR / "historical"
SMOKE_DIR = RESULT_DIR / "smoke"
WORK_DIR = RESULT_DIR / "work"
TRIALS_DIR = RESULT_DIR / "trials"
TRIAL_MANIFEST_PATH = RESULT_DIR / "trial_manifest.json"
ALL_TRIALS_PATH = RESULT_DIR / "all_trials.json"
SEARCH_SUMMARY_PATH = RESULT_DIR / "search_summary.json"
HISTORICAL_PATHS = {
    "B0": "experiments/cross_dataset_a012_structure_v1/abide5_ads_cn_B0_oof_predictions.csv",
    "B1": "experiments/cross_dataset_a012_structure_v1/abide5_ads_cn_B1_oof_predictions.csv",
    "fold_metrics": "experiments/cross_dataset_a012_structure_v1/abide5_ads_cn_fold_metrics.json",
    "summary": "experiments/cross_dataset_a012_structure_v1/summary.json",
}
HISTORICAL_ANCHORS = {
    "B0": HISTORICAL_DIR / "historical_B0_oof.csv",
    "B1": HISTORICAL_DIR / "historical_B1_oof.csv",
    "fold_metrics": HISTORICAL_DIR / "historical_fold_metrics.json",
}
REQUIRED_TRACKED = (
    ".gitignore", "scripts/run_abide5_hparam_search_v1.py",
    f"experiments/{EXPERIMENT_ID}/experiment_config.json",
    f"experiments/{EXPERIMENT_ID}/protocol.json",
    f"experiments/{EXPERIMENT_ID}/inspect_manifest.json",
    f"experiments/{EXPERIMENT_ID}/fold_manifest.json",
    f"experiments/{EXPERIMENT_ID}/registered_search.json",
    f"experiments/{EXPERIMENT_ID}/historical/historical_B0_oof.csv",
    f"experiments/{EXPERIMENT_ID}/historical/historical_B1_oof.csv",
    f"experiments/{EXPERIMENT_ID}/historical/historical_fold_metrics.json",
)
LOCKED_DEPENDENCIES = (
    "Model/cme_dual_branch.py", "Model/network.py", "Model/models.py",
    "Loss/loss_fn.py", "Utils/data_load.py", "Utils/graph_load.py", "Utils/utils.py", "config_re.py",
    "scripts/run_cross_dataset_a012_structure_v1.py",
    "scripts/run_tad_binary_hparam_search_v1.py", "scripts/run_abide_hparam_search_v1.py",
    "experiments/cross_dataset_a012_structure_v1/protocols/abide5_ads_cn.json",
)


def require(condition: bool, message: str) -> None:
    engine.require(condition, message)


def protocol() -> dict[str, Any]:
    value = engine.read_json(PROTOCOL_PATH)
    require(value["dataset"] == "ABIDE-5" and value["task"] == "ADS_CN", "ABIDE-5 task changed")
    require(value["class_names"] == ["ADS", "CN"] and value["positive_index"] == 0, "ASD positive direction changed")
    require(value["sample_count"] == 864 and value["class_counts"] == {"ADS": 397, "CN": 467}, "ABIDE-5 counts changed")
    require(len(value["modalities"]) == 5 and sum(item["feature_count"] for item in value["modalities"]) == 470, "Five-modality manifest changed")
    training = value["training"]
    require(training["lr"] == 0.005 and training["weight_decay"] == 0.0005 and training["drop_rate"] == 0.45, "Historical center changed")
    return value


def validate_oof(rows: list[dict[str, Any]], context: dict[str, Any], arm: str | None = None) -> None:
    ordered = sorted(rows, key=lambda row: int(row["original_csv_index"]))
    anchors = engine.expected_subjects(context)
    require(len(ordered) == len(anchors) == 864 and len({row["subject_id"] for row in ordered}) == 864, "OOF coverage changed")
    for row, anchor in zip(ordered, anchors):
        require(row["subject_id"] == anchor["subject_id"] and row["feature_sha256"] == anchor["feature_sha256"], "OOF subject anchor changed")
        require(row["truth"] == anchor["truth"] and row["fold"] == anchor["fold"], "OOF truth/fold changed")
        require(row["prediction"] == int(np.argmax([row["probability_0"], row["probability_1"]])), "OOF argmax changed")
        require(abs(float(row["positive_probability"]) - float(row["probability_0"])) <= 2e-6, "OOF ADS-positive probability changed")
        if arm is not None: require(row["arm"] == arm, "OOF arm changed")
    probability = np.asarray([[row["probability_0"], row["probability_1"]] for row in ordered], dtype=float)
    require(np.isfinite(probability).all() and float(np.max(np.abs(probability.sum(1) - 1.0))) <= 2e-6, "OOF probabilities invalid")


def prediction_rows(context: dict[str, Any], spec: dict[str, Any], fold: int, logits: np.ndarray, probabilities: np.ndarray, truth: np.ndarray) -> list[dict[str, Any]]:
    """Preserve the shared engine schema while honoring ABIDE-5's class-0 positive label."""
    rows = _BASE_PREDICTION_ROWS(context, spec, fold, logits, probabilities, truth)
    positive_index = int(context["protocol"]["positive_index"])
    require(positive_index == 0, "ABIDE-5 positive index changed")
    for row, probability in zip(rows, probabilities):
        row["positive_probability"] = float(probability[positive_index])
    return rows


def registered_search() -> dict[str, Any]:
    training = protocol()["training"]
    lr, wd, dropout = float(training["lr"]), float(training["weight_decay"]), float(training["drop_rate"])
    core = {
        "historical_center": {"lr": lr, "weight_decay": wd, "dropout": dropout},
        "stage_a_lr": [0.75 * lr, lr, 1.25 * lr, 1.5 * lr],
        "stage_a_fixed": {"weight_decay": wd, "dropout": dropout},
        "stage_b_weight_decay": [0.5 * wd, wd, 2.0 * wd, 3.0 * wd],
        "stage_b_fixed_dropout": dropout,
        "stage_c_dropout": [dropout - 0.10, dropout, dropout + 0.10],
        "stage_d_rank": [4, 8], "stage_d_adapter_lr_multiplier": [0.5, 1.0, 2.0],
        "local_lr_factors": [0.9, 1.1], "local_correct_window": [783, 786],
        "selection": "safe > pooled Correct > pooled ROC-AUC > Macro-F1 > BACC > fewer params > earlier trial",
    }
    return {**core, "sha256": engine.payload_sha256(core)}


def historical_payload(write_anchors: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    context = engine.build_context(torch.device("cpu"))
    blobs = {key: engine.git_blob(HISTORICAL_RESULT_COMMIT, relative) for key, relative in HISTORICAL_PATHS.items()}
    if write_anchors:
        for key, destination in HISTORICAL_ANCHORS.items(): engine.atomic_write_bytes(destination, blobs[key])
        engine.atomic_write_json(REGISTERED_SEARCH_PATH, registered_search())
    else:
        for key, destination in HISTORICAL_ANCHORS.items(): require(destination.is_file() and destination.read_bytes() == blobs[key], f"Historical anchor drifted: {key}")
        require(REGISTERED_SEARCH_PATH.is_file() and engine.read_json(REGISTERED_SEARCH_PATH) == registered_search(), "Registered search drifted")
    rows_by_arm = {}
    for arm in ("B0", "B1"):
        rows = engine.typed_rows(list(csv.DictReader(io.StringIO(blobs[arm].decode("utf-8")))))
        validate_oof(rows, context, arm); rows_by_arm[arm] = rows
    fold_metrics = json.loads(blobs["fold_metrics"].decode("utf-8"))
    summary = json.loads(blobs["summary"].decode("utf-8"))
    require(summary["runtime_lock"]["source_commit"] == HISTORICAL_SOURCE_COMMIT, "Historical source changed")
    expected = {
        "B0": {"correct": 768, "acc": 0.8888888888888888, "roc_auc": 0.9066931321096663, "macro_f1": 0.8879280525769631, "bacc": 0.8872108263798618},
        "B1": {"correct": 766, "acc": 0.8865740740740741, "roc_auc": 0.8960889756686929, "macro_f1": 0.8860244233378562, "bacc": 0.8867685370471254},
    }
    computed = {}
    for arm in ("B0", "B1"):
        computed[arm] = engine.metrics(rows_by_arm[arm])
        for key, target in expected[arm].items(): require(math.isclose(float(computed[arm][key]), target, rel_tol=0, abs_tol=1e-9), f"Historical {arm} metric changed: {key}")
        require(len(fold_metrics[arm]) == 10, f"Historical {arm} fold metrics changed")
    weights = context["dataset_dict"]["Label_Weight"].detach().cpu().numpy().astype(float)
    require(float(np.max(np.abs(weights - np.asarray([(864-397)/864, (864-467)/864])))) <= 1e-7, "Historical global weights changed")
    task_result = summary["task_results"]["abide5_ads_cn"]
    fold_core = {"task_id": context["task_id"], "fold_manifest": context["fold_manifest"]}
    inspect_core = {
        "experiment": EXPERIMENT_ID, "base_commit": BASE_COMMIT,
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT, "historical_result_commit": HISTORICAL_RESULT_COMMIT,
        "dataset": "ABIDE-5", "task": "ADS_CN", "sample_count": 864,
        "class_names": ["ADS", "CN"], "positive_class": "ADS/ASD", "positive_index": 0,
        "modalities": 5, "feature_count": 470, "folds": list(engine.FOLDS), "seed": 0, "epochs": 400,
        "class_weight_scope": "global_full_dataset_historical", "class_weights": weights.tolist(),
        "criterion": "criterion_lossv2: main weighted CE + two unnormalized weighted OVR CE; label smoothing 0.05; orthogonality 0",
        "best_epoch_rule": "ACC > ROC-AUC > Macro-F1 > earliest epoch",
        "historical_reuse": {
            "eligible": True, "anchor_sha256": {key: engine.bytes_sha256(blobs[key]) for key in HISTORICAL_ANCHORS},
            "metrics": computed, "fold_metrics": fold_metrics,
            "parameter_count": {"B0": int(task_result["B0"]["parameter_count"]), "B1": int(task_result["B1"]["parameter_count"])},
            "mechanism_B1": summary["mechanism_diagnostics"]["abide5_ads_cn"],
        },
        "registered_search_sha256": registered_search()["sha256"],
        "protocol_sha256": engine.file_sha256(PROTOCOL_PATH),
        "dataset_sha256": protocol()["data_sha256"], "modality_sha256": protocol()["modality_sha256"],
    }
    del context
    return ({**inspect_core, "sha256": engine.payload_sha256(inspect_core)}, {**fold_core, "sha256": engine.payload_sha256(fold_core)})


def build_model(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    kwargs = engine.historical.model_kwargs(context); initialization = {}
    SET_Random(engine.SEED)
    if spec["arm"] == "B0":
        model = HeterGraph_Model_Kmeans(**kwargs).to(context["device"])
    else:
        reference = HeterGraph_Model_Kmeans(**kwargs).to(context["device"]); post_common_rng = engine.capture_rng()
        SET_Random(engine.SEED)
        model = CMEDualBranchModel(**kwargs, cme_arm="c1", adapter_rank=int(spec["rank"]), router_hidden=16, modality_embedding_dim=8).to(context["device"])
        difference = engine.common_state_max_diff(reference, model)
        if audit_common: require(difference == 0.0, "B0/B1 common initialization changed"); initialization["common_state_max_abs_diff"] = difference
        else: initialization["common_state_pointwise_audit"] = "smoke_only"
        del reference; engine.restore_rng(post_common_rng)
    require(len(model.label_pools) == 2 and len(model._Auxi_classifier) == 2, "Binary Query/OVR count changed")
    require(not any(layer.use_graph for layer in model.GCN.layers), "Graph unexpectedly enabled")
    if spec["arm"] == "B1":
        require(len(model.private_adapters) == 5, "ABIDE-5 requires five independent adapters")
        require(not hasattr(model, "private_alpha"), "Trainable alpha is forbidden")
    initialization.update({"parameter_count": engine.parameter_count(model), "query_count": len(model.label_pools), "ovr_head_count": len(model._Auxi_classifier), "adapter_count": len(model.private_adapters) if spec["arm"] == "B1" else 0})
    return model, initialization


def make_training_objects(context: dict[str, Any], spec: dict[str, Any], audit_common: bool = False):
    model, initialization = build_model(context, spec, audit_common)
    training = context["protocol"]["training"]
    criterion = criterion_lossv2(context["dataset_dict"], context["device"], rate=float(training["loss_rate"]), label_smoothing=float(training["label_smoothing"]))
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if spec["arm"] == "B0":
        optimizer = torch.optim.Adam([parameter for _, parameter in named], lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
        scheduler = CustomCosineAnnealingLR(optimizer, T_max=engine.EPOCHS, eta_min=float(training["scheduler_eta_min"]))
        audit = {"base_parameter_tensors": len(named), "adapter_parameter_tensors": 0, "base_lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"])}
    else:
        adapter = [(name, parameter) for name, parameter in named if name.startswith("private_adapters.")]
        base = [(name, parameter) for name, parameter in named if not name.startswith("private_adapters.")]
        require(len(adapter) == 20, "Five adapters must expose 20 parameter tensors")
        base_ids, adapter_ids, all_ids = {id(p) for _,p in base}, {id(p) for _,p in adapter}, {id(p) for _,p in named}
        require(not (base_ids & adapter_ids) and base_ids | adapter_ids == all_ids, "Optimizer partition changed")
        multiplier = float(spec["adapter_lr_multiplier"])
        optimizer = torch.optim.Adam([
            {"params": [p for _,p in base], "lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"]), "group_name": "base"},
            {"params": [p for _,p in adapter], "lr": float(spec["lr"])*multiplier, "weight_decay": float(spec["weight_decay"]), "group_name": "private_adapters"},
        ])
        scheduler = engine.historical.RatioPreservingCustomCosineAnnealingLR(optimizer, T_max=engine.EPOCHS, eta_min=float(training["scheduler_eta_min"]), multiplier=multiplier)
        audit = {"base_parameter_tensors": len(base), "adapter_parameter_tensors": len(adapter), "adapter_parameter_names": [name for name,_ in adapter], "base_lr": float(spec["lr"]), "adapter_lr": float(spec["lr"])*multiplier, "weight_decay": float(spec["weight_decay"])}
    require(optimizer.defaults["betas"] == (0.9,0.999) and optimizer.defaults["eps"] == 1e-8, "Adam defaults changed")
    return model, criterion, optimizer, scheduler, {**initialization, **audit}


def run_smoke(device_text: str) -> None:
    require(device_text == "cuda:0" and torch.cuda.is_available(), "Smoke requires cuda:0")
    source = engine.source_gate(); runtime = engine.runtime_lock(source); base = engine.build_context(torch.device(device_text))
    spec = engine.trial_spec("SMOKE_D1", "smoke", "B1", 0.005, 0.0005, 0.45, rank=4, multiplier=0.5)
    context = engine.trial_context(base, spec); smoke_config = engine.smoke_runtime_config(runtime, spec)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True); engine.atomic_write_json(SMOKE_DIR/"smoke_config.json", smoke_config)
    b0_spec = engine.trial_spec("SMOKE_B0", "smoke", "B0", 0.005, 0.0005, 0.45)
    b0_model,_ = build_model(engine.trial_context(base,b0_spec), b0_spec)
    model,criterion,optimizer,scheduler,audit = make_training_objects(context,spec,audit_common=True)
    features,labels = context["dataset_data"]["Feature"],context["dataset_data"]["Label"]
    labels_before=labels.detach().clone(); train_mask,test_mask,_=engine.historical.fold_positions(context,0)
    b0_model.eval(); model.eval()
    with torch.no_grad(): initial_diff=float((b0_model(features)[0]-model(features)[0]).abs().max().cpu())
    require(initial_diff==0.0,"Zero-output adapters changed initial logits"); del b0_model
    initial={name:p.detach().cpu().clone() for name,p in model.named_parameters() if name.startswith("private_adapters.")}
    gradients={name:0.0 for name in initial}; losses=[]; formula_errors=[]; simplex=[]
    for _epoch in range(1,4):
        model.train(); optimizer.zero_grad(set_to_none=True); output,embeddings,auxiliary=model(features)
        require(tuple(output.shape)==(864,2) and len(auxiliary)==2,"Smoke output changed")
        loss=criterion(output,labels,train_mask,embeddings,auxiliary); manual,components=engine.manual_historical_loss(criterion,output,labels,train_mask,embeddings,auxiliary)
        formula_errors.append(float((loss-manual).abs().detach().cpu())); require(bool(torch.isfinite(loss)) and formula_errors[-1]<=1e-7,"Historical loss changed")
        loss.backward()
        for name,p in model.named_parameters():
            if name.startswith("private_adapters."):
                require(p.grad is not None and bool(torch.isfinite(p.grad).all()),f"Invalid adapter gradient: {name}")
                gradients[name]=max(gradients[name],float(p.grad.detach().abs().max().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(context["protocol"]["training"]["grad_clip"])); optimizer.step(); scheduler.step(); scheduler.assert_ratio(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): probability=torch.softmax(model(features)[0],dim=-1)
        simplex.append(float((probability.sum(1)-1).abs().max().cpu()))
    require(torch.equal(labels,labels_before),"Labels changed")
    delta={name:float((p.detach().cpu()-initial[name]).abs().max()) for name,p in model.named_parameters() if name in initial}
    require(all(v>0 and math.isfinite(v) for v in gradients.values()) and all(v>0 and math.isfinite(v) for v in delta.values()),"Adapter smoke gate failed")
    checkpoint_path=SMOKE_DIR/"checkpoint_roundtrip.pt"
    payload={"runtime_lock":runtime,"smoke_config":smoke_config,"spec":spec,"epoch":3,"model":engine.clone_cpu_state(model),"optimizer":copy.deepcopy(optimizer.state_dict()),"scheduler":copy.deepcopy(scheduler.state_dict()),"rng":engine.capture_rng()}
    engine.atomic_torch_save(checkpoint_path,payload)
    restored,_,restored_optimizer,restored_scheduler,_=make_training_objects(context,spec); loaded=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    restored.load_state_dict(loaded["model"],strict=True); restored_optimizer.load_state_dict(loaded["optimizer"]); restored_scheduler.load_state_dict(loaded["scheduler"]); restored_scheduler.assert_ratio()
    model.eval(); restored.eval()
    with torch.no_grad(): reload_diff=float((model(features)[0]-restored(features)[0]).abs().max().cpu()); test_probability=torch.softmax(model(features)[0][test_mask],-1).detach().cpu().numpy()
    require(reload_diff==0.0,"Checkpoint reload changed logits")
    test_truth=labels[test_mask].detach().cpu().numpy(); test_metrics=engine.historical.binary_metrics(test_truth,test_probability,protocol()); test_prediction=test_probability.argmax(1)
    require(test_metrics["tp"]==int(np.sum((test_truth==0)&(test_prediction==0))) and test_metrics["tn"]==int(np.sum((test_truth==1)&(test_prediction==1))),"SEN/SPE direction changed")
    report_core={"runtime_lock":runtime,"smoke_config_sha256":engine.file_sha256(SMOKE_DIR/"smoke_config.json"),"spec":spec,"fold":0,"epochs":3,"losses":losses,"historical_loss_components_last_epoch":components,"historical_loss_formula_max_abs_error":max(formula_errors),"probability_sum_max_abs_error":max(simplex),"initial_b0_b1_logits_max_abs_diff":initial_diff,"adapter_gradient_max_by_tensor":gradients,"adapter_parameter_delta_by_tensor":delta,"object_audit":audit,"checkpoint_sha256":engine.file_sha256(checkpoint_path),"checkpoint_reload_logits_max_abs_diff":reload_diff,"positive_direction_audit":{"positive_class":"ADS/ASD","positive_index":0,"test_metrics":test_metrics},"test_labels_unchanged":True,"device":torch.cuda.get_device_name(torch.device(device_text)),"torch_version":torch.__version__,"cuda_version":torch.version.cuda}
    engine.atomic_write_json(SMOKE_DIR/"smoke_report.json",{**report_core,"sha256":engine.payload_sha256(report_core)}); print(json.dumps({"smoke":"PASS","losses":losses,"params":audit["parameter_count"]},indent=2))
    del model,restored,base,context; torch.cuda.empty_cache()


def safety(result: dict[str, Any]) -> dict[str, Any]:
    m=result["metrics"]; collapse=min(int(m["predicted_counts"]["ADS"]),int(m["predicted_counts"]["CN"]))==0; reasons=[]
    if result["spec"]["arm"]=="B0":
        historical_bacc=float(engine.read_json(INSPECT_PATH)["historical_reuse"]["metrics"]["B0"]["bacc"])
        if float(m["bacc"])<historical_bacc-0.005: reasons.append("BACC is more than 0.005 below Historical B0")
    if collapse: reasons.append("single-class prediction collapse")
    return {"safe":not reasons,"collapse":collapse,"reasons":reasons}


def rank_key(result: dict[str, Any]) -> tuple[Any,...]:
    return (int(result["metrics"]["correct"]),float(result["metrics"]["roc_auc"]),float(result["metrics"]["macro_f1"]),float(result["metrics"]["bacc"]),-int(result["parameter_count"]),-int(result["spec"]["ordinal"]))


def is_historical_exact(spec: dict[str, Any]) -> bool:
    exact=math.isclose(spec["lr"],0.005) and math.isclose(spec["weight_decay"],0.0005) and math.isclose(spec["dropout"],0.45)
    return exact if spec["arm"]=="B0" else exact and spec["rank"]==8 and math.isclose(spec["adapter_lr_multiplier"],2.0)


def structural_safety(tuned_b0: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    pair=engine.paired(tuned_b0,candidate); reasons=list(candidate["safety"]["reasons"])
    if float(candidate["metrics"]["bacc"])<float(tuned_b0["metrics"]["bacc"])-0.005: reasons.append("BACC is more than 0.005 below TUNED_B0")
    mechanism=candidate.get("mechanism")
    if mechanism and (not mechanism["private_trained_all_folds"] or mechanism["private_collapse_any_fold"]): reasons.append("private residual did not train or collapsed")
    return {"safe":not reasons,"reasons":reasons,"collapse":candidate["safety"]["collapse"],"repairs":pair["repairs"],"damages":pair["damages"],"changed":pair["changed"],"bacc_delta_vs_tuned_b0":pair["metric_deltas"]["bacc"]}


def select_structural(candidates:list[dict[str,Any]],tuned_b0:dict[str,Any],registry:dict[str,Any],stage:str="D") -> dict[str,Any]:
    audits={item["trial_id"]:structural_safety(tuned_b0,item) for item in candidates}; safe=[item for item in candidates if audits[item["trial_id"]]["safe"]]
    overall=max(candidates,key=rank_key); selected=max(safe,key=rank_key) if safe else None
    value={"stage":stage,"candidate_trial_ids":[x["trial_id"] for x in candidates],"safe_trial_ids":[x["trial_id"] for x in safe],"best_overall_trial_id":overall["trial_id"],"selected_trial_id":selected["trial_id"] if selected else None,"dynamic_safety":audits,"selection_rule":"B1 safety > Correct > pooled ROC-AUC > Macro-F1 > BACC > fewer params > earlier trial"}
    registry["stages"][stage]=value; engine.registry_save(registry); return value


def select_b0_all(candidates:list[dict[str,Any]],registry:dict[str,Any]) -> dict[str,Any]:
    safe=[item for item in candidates if item["safety"]["safe"]]; require(bool(safe),"No safe B0 candidate")
    selected=max(safe,key=rank_key); value={"stage":"B0_ALL","candidate_trial_ids":[x["trial_id"] for x in candidates],"safe_trial_ids":[x["trial_id"] for x in safe],"selected_trial_id":selected["trial_id"],"selection_rule":"all completed safe B0 trials > Correct > pooled ROC-AUC > Macro-F1 > BACC > fewer params > earlier trial"}
    registry["stages"]["B0_ALL"]=value; engine.registry_save(registry); return value


def run_search(device_text:str) -> None:
    require(device_text=="cuda:0" and torch.cuda.is_available(),"Search requires cuda:0")
    source=engine.source_gate(); runtime=engine.runtime_lock(source); engine.validate_smoke(runtime); base_context=engine.build_context(torch.device(device_text)); registry=engine.registry_load(runtime); results={}; registered=engine.read_json(REGISTERED_SEARCH_PATH)
    stage_a_specs=[engine.trial_spec(f"A{i}","A","B0",lr,0.0005,0.45,ordinal=i) for i,lr in enumerate(registered["stage_a_lr"],1)]
    for spec in stage_a_specs: results[spec["trial_id"]]=engine.ensure_trial(spec,base_context,runtime,registry)
    stage_a=engine.select_stage("A",[results[s["trial_id"]] for s in stage_a_specs],registry); require(stage_a["selected_trial_id"],"Stage A has no safe candidate"); best_a=results[stage_a["selected_trial_id"]]
    stage_b_specs=[engine.trial_spec(f"B{i}","B","B0",best_a["spec"]["lr"],wd,0.45,ordinal=4+i) for i,wd in enumerate(registered["stage_b_weight_decay"],1)]
    for spec in stage_b_specs: results[spec["trial_id"]]=engine.ensure_trial(spec,base_context,runtime,registry)
    stage_b=engine.select_stage("B",[results[s["trial_id"]] for s in stage_b_specs],registry); require(stage_b["selected_trial_id"],"Stage B has no safe candidate"); best_b=results[stage_b["selected_trial_id"]]
    stage_c_specs=[engine.trial_spec(f"C{i}","C","B0",best_b["spec"]["lr"],best_b["spec"]["weight_decay"],drop,ordinal=8+i) for i,drop in enumerate(registered["stage_c_dropout"],1)]
    for spec in stage_c_specs: results[spec["trial_id"]]=engine.ensure_trial(spec,base_context,runtime,registry)
    engine.select_stage("C",[results[s["trial_id"]] for s in stage_c_specs],registry)
    b0_candidates=[results[s["trial_id"]] for s in [*stage_a_specs,*stage_b_specs,*stage_c_specs]]
    tuned_b0=results[select_b0_all(b0_candidates,registry)["selected_trial_id"]]
    stage_d_specs=[]; ordinal=12
    for rank in (4,8):
        for multiplier in (0.5,1.0,2.0):
            stage_d_specs.append(engine.trial_spec(f"D{ordinal-11}","D","B1",tuned_b0["spec"]["lr"],tuned_b0["spec"]["weight_decay"],tuned_b0["spec"]["dropout"],rank=rank,multiplier=multiplier,ordinal=ordinal)); ordinal+=1
    for spec in stage_d_specs: results[spec["trial_id"]]=engine.ensure_trial(spec,base_context,runtime,registry)
    stage_d=select_structural([results[s["trial_id"]] for s in stage_d_specs],tuned_b0,registry)
    tuned_b1=results[stage_d["selected_trial_id"]] if stage_d["selected_trial_id"] else results[stage_d["best_overall_trial_id"]]
    safe_final=[tuned_b0]+([tuned_b1] if structural_safety(tuned_b0,tuned_b1)["safe"] else []); current=max(safe_final,key=rank_key)
    if current["spec"]["arm"]=="B1": reference=tuned_b0; pair=engine.paired(reference,current)
    else:
        reference,_=historical_reference("B0"); pair=engine.paired(reference,current)
    local_triggered=bool(783<=int(current["metrics"]["correct"])<=786 and pair["repairs"]>pair["damages"] and not current["safety"]["collapse"] and (current["spec"]["arm"]=="B0" or structural_safety(tuned_b0,current)["safe"]))
    local_reason="Current best safe model entered registered 783-786 LR window" if local_triggered else "Current best safe model is outside registered local-refinement gates"
    if int(current["metrics"]["correct"])<=782: local_reason="Current best Correct <= 782"
    elif int(current["metrics"]["correct"])>=787: local_reason="Current best Correct >= 787"
    if local_triggered:
        local_specs=[engine.trial_spec("L1","LOCAL",current["spec"]["arm"],current["spec"]["lr"]*0.9,current["spec"]["weight_decay"],current["spec"]["dropout"],rank=current["spec"]["rank"],multiplier=current["spec"]["adapter_lr_multiplier"],ordinal=18),engine.trial_spec("L2","LOCAL",current["spec"]["arm"],current["spec"]["lr"]*1.1,current["spec"]["weight_decay"],current["spec"]["dropout"],rank=current["spec"]["rank"],multiplier=current["spec"]["adapter_lr_multiplier"],ordinal=19)]
        for spec in local_specs: results[spec["trial_id"]]=engine.ensure_trial(spec,base_context,runtime,registry)
        if current["spec"]["arm"]=="B0":
            candidates=[tuned_b0,*[results[s["trial_id"]] for s in local_specs]]; tuned_b0=max([x for x in candidates if x["safety"]["safe"]],key=rank_key); registry["stages"]["LOCAL"]={"candidate_trial_ids":[x["trial_id"] for x in candidates],"selected_trial_id":tuned_b0["trial_id"],"selected_arm":"B0"}; engine.registry_save(registry)
        else:
            local=select_structural([tuned_b1,*[results[s["trial_id"]] for s in local_specs]],tuned_b0,registry,"LOCAL"); tuned_b1=results[local["selected_trial_id"]] if local["selected_trial_id"] else tuned_b1
    registry["local_refinement"]={"triggered":local_triggered,"reason":local_reason}; registry["tuned_b0_trial_id"]=tuned_b0["trial_id"]; registry["tuned_b1_trial_id"]=tuned_b1["trial_id"]; engine.registry_save(registry)
    ordered=[engine.load_trial_result(entry["spec"],base_context) for entry in sorted(registry["trials"].values(),key=lambda x:int(x["spec"]["ordinal"]))]; core={"runtime_lock":runtime,"trial_count":len(ordered),"trials":ordered}; engine.atomic_write_json(ALL_TRIALS_PATH,{**core,"sha256":engine.payload_sha256(core)})
    search_core={"runtime_lock":runtime,"registered_search_sha256":engine.file_sha256(REGISTERED_SEARCH_PATH),"trial_manifest_sha256":engine.file_sha256(TRIAL_MANIFEST_PATH),"all_trials_sha256":engine.file_sha256(ALL_TRIALS_PATH),"stage_selections":registry["stages"],"local_refinement":registry["local_refinement"],"tuned_b0_trial_id":tuned_b0["trial_id"],"tuned_b1_trial_id":tuned_b1["trial_id"],"failure_count":0}; engine.atomic_write_json(SEARCH_SUMMARY_PATH,{**search_core,"sha256":engine.payload_sha256(search_core)}); print(json.dumps({"search":"COMPLETE","tuned_b0":tuned_b0["trial_id"],"tuned_b1":tuned_b1["trial_id"],"local":local_triggered},indent=2)); del base_context; torch.cuda.empty_cache()


def historical_reference(arm:str) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    spec=engine.trial_spec(f"HISTORICAL_{arm}","HISTORICAL",arm,0.005,0.0005,0.45,rank=8 if arm=="B1" else 0,multiplier=2.0 if arm=="B1" else 0.0,ordinal=-2 if arm=="B0" else -1)
    rows=[{**row,"trial_id":spec["trial_id"]} for row in engine.typed_rows(engine.read_csv(HISTORICAL_ANCHORS[arm]))]; folds=engine.read_json(HISTORICAL_ANCHORS["fold_metrics"])[arm]
    summaries=[{"fold":int(row["fold"]),"best_epoch":int(row["best_epoch"]),"best_metrics":{key:(int(row[key]) if key=="correct" else float(row[key])) for key in ("correct","acc","roc_auc","pr_auc","macro_f1","bacc")},"elapsed_seconds":float(row["elapsed_seconds"]),"inference_seconds_full_graph":float(row["inference_seconds_full_graph"]),"resumed_from_epoch":int(row.get("resumed_from_epoch",0))} for row in folds]
    inspect=engine.read_json(INSPECT_PATH); result=engine.make_trial_result(spec,rows,summaries,int(inspect["historical_reuse"]["parameter_count"][arm]),inspect["historical_reuse"]["mechanism_B1"] if arm=="B1" else None,f"historical:{HISTORICAL_RESULT_COMMIT}:{arm}"); return result,rows


def target_gate(result:dict[str,Any]) -> dict[str,Any]:
    m=result["metrics"]; imbalance=min(float(m["sen"]),float(m["spe"]))<0.84 or abs(float(m["sen"])-float(m["spe"]))>0.08
    checks={"correct_at_least_787":int(m["correct"])>=787,"fold_acc_mean_above_0_9105":float(result["fold_acc_mean"])>0.9105,"fold_roc_auc_mean_above_0_9099":float(result["fold_roc_auc_mean"])>0.9099,"macro_f1_at_least_0_90":float(m["macro_f1"])>=0.90,"bacc_at_least_0_90":float(m["bacc"])>=0.90,"no_sen_spe_imbalance":not imbalance,"no_collapse":not result["safety"]["collapse"]}
    return {"reached":all(checks.values()),"checks":checks}


def final_decision(tuned_b0:dict[str,Any],tuned_b1:dict[str,Any],comparisons:dict[str,Any]) -> dict[str,Any]:
    b0_gate,b1_gate=target_gate(tuned_b0),target_gate(tuned_b1); pair=comparisons["tuned_b1_vs_tuned_b0"]; structural=structural_safety(tuned_b0,tuned_b1)
    private_eligible=bool(structural["safe"] and int(tuned_b1["metrics"]["correct"])>int(tuned_b0["metrics"]["correct"]) and pair["repairs"]>pair["damages"])
    retained=tuned_b1 if private_eligible and rank_key(tuned_b1)>rank_key(tuned_b0) else tuned_b0; retained_gate=target_gate(retained)
    if retained_gate["reached"] and retained["spec"]["arm"]=="B1": decision="ABIDE5_PRIVATE_TARGET_GO"
    elif retained_gate["reached"]: decision="ABIDE5_BASE_TARGET_ONLY"
    elif retained["spec"]["arm"]=="B1": decision="ABIDE5_PRIVATE_POSITIVE"
    elif int(retained["metrics"]["correct"])>768: decision="ABIDE5_BASE_TUNE_POSITIVE"
    else: decision="ABIDE5_TUNE_NO_GAIN"
    if retained["spec"]["arm"]=="B0" and (int(tuned_b1["metrics"]["correct"])<=int(tuned_b0["metrics"]["correct"]) or pair["repairs"]<=pair["damages"]): private_status="ABIDE5_PRIVATE_NO_GAIN"
    else: private_status="ABIDE5_PRIVATE_SUPPORTED"
    return {"decision":decision,"private_status":private_status,"target_status":"ABIDE5_TARGET_REACHED" if retained_gate["reached"] else "ABIDE5_TARGET_NOT_REACHED","tuned_b0_target":b0_gate,"tuned_b1_target":b1_gate,"private_eligible_as_final":private_eligible,"tuned_b1_structural_safety":structural,"retained_trial_id":retained["trial_id"],"retained_arm":retained["spec"]["arm"],"next_action":"stop registered ABIDE-5 search"}


def render_report(runtime:dict[str,Any],registry:dict[str,Any],results:dict[str,dict[str,Any]],references:dict[str,dict[str,Any]],comparisons:dict[str,Any],decision:dict[str,Any],seconds:float) -> str:
    b0,b1=results[registry["tuned_b0_trial_id"]],results[registry["tuned_b1_trial_id"]]; displayed=[("Historical B0",references["B0"]),("Historical B1",references["B1"]),("TUNED_B0",b0),("TUNED_B1",b1)]
    lines=["# ABIDE-5 Binary Task Hyperparameter Search v1","",f"- Source commit: `{runtime['source_commit']}`","- Result commit: reported in final Git handoff",f"- Branch: `{BRANCH}`","- Device: `cuda:0`",f"- Decision: **{decision['decision']}**",f"- Private status: **{decision['private_status']}**",f"- Retained model: `{decision['retained_trial_id']}` ({decision['retained_arm']})",f"- Formal aggregation: {seconds:.3f} seconds","","## Locked protocol","","ABIDE-5 ADS_CN only: 864 subjects (397 ADS/ASD positive index 0, 467 CN negative index 1), five real modalities, seed 0, ten full-batch transductive folds, 400 epochs, historical global class weights, criterion_lossv2 main weighted CE plus two unnormalized weighted OVR CE terms, label smoothing 0.05, orthogonality 0, Adam, grad clip 1, historical cosine scheduler, graph/EMA/ensemble off, and test-fold best selection by ACC > ROC-AUC > Macro-F1 > earliest.","", "## Selected configurations","",f"- TUNED_B0: `{b0['trial_id']}`, lr={b0['spec']['lr']}, WD={b0['spec']['weight_decay']}, dropout={b0['spec']['dropout']}.",f"- TUNED_B1: `{b1['trial_id']}`, lr={b1['spec']['lr']}, WD={b1['spec']['weight_decay']}, dropout={b1['spec']['dropout']}, rank={b1['spec']['rank']}, multiplier={b1['spec']['adapter_lr_multiplier']}.",f"- LOCAL triggered: `{registry['local_refinement']['triggered']}` ({registry['local_refinement']['reason']}).","","## Results","","| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Confusion [ADS,CN] | Pred ADS/CN | Params | Train sec | Infer sec |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for label,result in displayed:
        m=result["metrics"]; lines.append(f"| {label} | {m['correct']}/{m['n']} | {m['acc']:.7f} | {result['fold_acc_mean']:.7f} +/- {result['fold_acc_sample_std']:.7f} | {m['roc_auc']:.7f} | {result['fold_roc_auc_mean']:.7f} +/- {result['fold_roc_auc_sample_std']:.7f} | {m['pr_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} | {m['weighted_f1']:.7f} | {m['sen']:.7f} | {m['spe']:.7f} | {m['confusion_matrix']} | {m['predicted_counts']['ADS']}/{m['predicted_counts']['CN']} | {result['parameter_count']} | {result['training_time_seconds']:.1f} | {result['inference_time_seconds']:.4f} |")
    lines += ["","## Registered trials","","| Trial | Stage | Arm | LR | WD | Dropout | Rank | Mult | Safe | Correct | AUC | F1 | BACC |","|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for result in sorted(results.values(),key=lambda x:int(x["spec"]["ordinal"])):
        s,m=result["spec"],result["metrics"]; lines.append(f"| {s['trial_id']} | {s['stage']} | {s['arm']} | {s['lr']:.8g} | {s['weight_decay']:.8g} | {s['dropout']:.4g} | {s['rank']} | {s['adapter_lr_multiplier']:.3g} | {result['safety']['safe']} | {m['correct']} | {m['roc_auc']:.7f} | {m['macro_f1']:.7f} | {m['bacc']:.7f} |")
    lines += ["","## Pairwise comparisons","",f"- TUNED_B0 vs Historical B0: repairs/damages/changed={comparisons['tuned_b0_vs_historical_b0']['repairs']}/{comparisons['tuned_b0_vs_historical_b0']['damages']}/{comparisons['tuned_b0_vs_historical_b0']['changed']}, exact McNemar p={comparisons['tuned_b0_vs_historical_b0']['exact_mcnemar_p']:.8f}.",f"- TUNED_B1 vs Historical B1: repairs/damages/changed={comparisons['tuned_b1_vs_historical_b1']['repairs']}/{comparisons['tuned_b1_vs_historical_b1']['damages']}/{comparisons['tuned_b1_vs_historical_b1']['changed']}, exact McNemar p={comparisons['tuned_b1_vs_historical_b1']['exact_mcnemar_p']:.8f}.",f"- TUNED_B1 vs TUNED_B0: repairs/damages/changed={comparisons['tuned_b1_vs_tuned_b0']['repairs']}/{comparisons['tuned_b1_vs_tuned_b0']['damages']}/{comparisons['tuned_b1_vs_tuned_b0']['changed']}, exact McNemar p={comparisons['tuned_b1_vs_tuned_b0']['exact_mcnemar_p']:.8f}.","","## DGFMC target","",f"- Correct >=787: `{max(b0['metrics']['correct'],b1['metrics']['correct'])>=787}`.",f"- Retained fold mean ACC >91.05%: `{(b1 if decision['retained_arm']=='B1' else b0)['fold_acc_mean']>0.9105}`.",f"- Retained fold mean ROC-AUC >90.99%: `{(b1 if decision['retained_arm']=='B1' else b0)['fold_roc_auc_mean']>0.9099}`.","","## Private mechanism","",f"- TUNED_B1 mechanism: `{json.dumps(b1.get('mechanism'),sort_keys=True)}`.","","## Conclusion","",f"- B0 improved over history: `{comparisons['tuned_b0_vs_historical_b0']['correct_delta']>0}`.",f"- Private residual adds hard-classification gain over TUNED_B0: `{comparisons['tuned_b1_vs_tuned_b0']['correct_delta']>0 and comparisons['tuned_b1_vs_tuned_b0']['repairs']>comparisons['tuned_b1_vs_tuned_b0']['damages']}`.",f"- Final Decision: `{decision['decision']}`; private status `{decision['private_status']}`.","","## Reproduction","",f"Run a clean local branch named `{BRANCH}` at source commit `{runtime['source_commit']}`:","","```text","python -u -B scripts/run_abide5_hparam_search_v1.py inspect","python -u -B scripts/run_abide5_hparam_search_v1.py smoke --device cuda:0","python -u -B scripts/run_abide5_hparam_search_v1.py search --device cuda:0","python -u -B scripts/run_abide5_hparam_search_v1.py formal --device cuda:0","```",""]
    return "\n".join(lines)


def run_formal(device_text:str) -> None:
    require(device_text=="cuda:0" and torch.cuda.is_available(),"Formal requires cuda:0"); started=time.perf_counter(); source=engine.source_gate(); runtime=engine.runtime_lock(source); engine.validate_smoke(runtime); context=engine.build_context(torch.device(device_text)); registry,results=engine.validate_search(runtime,context)
    b0,b1=results[registry["tuned_b0_trial_id"]],results[registry["tuned_b1_trial_id"]]; hist0,hist0rows=historical_reference("B0"); hist1,hist1rows=historical_reference("B1"); b0rows=engine.parse_trial_rows(engine.trial_dir(b0["spec"])/"oof_predictions.csv"); b1rows=engine.parse_trial_rows(engine.trial_dir(b1["spec"])/"oof_predictions.csv")
    comparisons={"tuned_b0_vs_historical_b0":engine.paired_rows(hist0,hist0rows,b0,b0rows),"tuned_b1_vs_historical_b1":engine.paired_rows(hist1,hist1rows,b1,b1rows),"tuned_b1_vs_tuned_b0":engine.paired_rows(b0,b0rows,b1,b1rows)}; decision=final_decision(b0,b1,comparisons)
    retained=b1 if decision["retained_arm"]=="B1" else b0; retained_rows=b1rows if decision["retained_arm"]=="B1" else b0rows
    engine.atomic_write_csv(RESULT_DIR/"tuned_b0_oof_predictions.csv",b0rows); engine.atomic_write_csv(RESULT_DIR/"tuned_b1_oof_predictions.csv",b1rows); engine.atomic_write_csv(RESULT_DIR/"retained_oof_predictions.csv",retained_rows)
    formal_core={"runtime_lock":runtime,"registered_search_sha256":engine.file_sha256(REGISTERED_SEARCH_PATH),"smoke_report_sha256":engine.file_sha256(SMOKE_DIR/"smoke_report.json"),"search_summary_sha256":engine.file_sha256(SEARCH_SUMMARY_PATH),"trial_manifest_sha256":engine.file_sha256(TRIAL_MANIFEST_PATH),"all_trials_sha256":engine.file_sha256(ALL_TRIALS_PATH),"mode":"strict validation and aggregate only"}; engine.atomic_write_json(RESULT_DIR/"formal_config.json",{**formal_core,"sha256":engine.payload_sha256(formal_core)}); engine.atomic_write_json(RESULT_DIR/"comparisons.json",comparisons); engine.atomic_write_json(RESULT_DIR/"mechanism_diagnostics.json",{"tuned_b1_trial_id":b1["trial_id"],"mechanism":b1.get("mechanism")})
    elapsed=time.perf_counter()-started; engine.atomic_write_text(RESULT_DIR/"REPORT.md",render_report(runtime,registry,results,{"B0":hist0,"B1":hist1},comparisons,decision,elapsed)); commands=f"# Reproduce from source commit {runtime['source_commit']} on local branch {BRANCH}\npython -u -B scripts/run_abide5_hparam_search_v1.py inspect\npython -u -B scripts/run_abide5_hparam_search_v1.py smoke --device cuda:0\npython -u -B scripts/run_abide5_hparam_search_v1.py search --device cuda:0\npython -u -B scripts/run_abide5_hparam_search_v1.py formal --device cuda:0\n"; engine.atomic_write_text(RESULT_DIR/"reproduction_commands.txt",commands)
    core={"experiment":EXPERIMENT_ID,"runtime_lock":runtime,"historical_b0":hist0,"historical_b1":hist1,"tuned_b0":b0,"tuned_b1":b1,"retained_model":retained,"comparisons":comparisons,"decision":decision,"registered_trial_count":len(results),"failure_count":0,"formal_aggregation_seconds":float(elapsed),"registered_search_sha256":engine.file_sha256(REGISTERED_SEARCH_PATH),"trial_manifest_sha256":engine.file_sha256(TRIAL_MANIFEST_PATH),"all_trials_sha256":engine.file_sha256(ALL_TRIALS_PATH),"report_sha256":engine.file_sha256(RESULT_DIR/"REPORT.md"),"reproduction_commands_sha256":engine.file_sha256(RESULT_DIR/"reproduction_commands.txt"),"retained_oof_sha256":engine.file_sha256(RESULT_DIR/"retained_oof_predictions.csv")}; engine.atomic_write_json(RESULT_DIR/"summary.json",{**core,"sha256":engine.payload_sha256(core)}); print(json.dumps({"formal":"COMPLETE","decision":decision["decision"],"retained":retained["trial_id"],"metrics":retained["metrics"]},indent=2))


def install_profile() -> None:
    parent.install_profile()
    values={"ROOT":ROOT,"EXPERIMENT_ID":EXPERIMENT_ID,"BRANCH":BRANCH,"BASE_COMMIT":BASE_COMMIT,"HISTORICAL_RESULT_COMMIT":HISTORICAL_RESULT_COMMIT,"RESULT_DIR":RESULT_DIR,"PROTOCOL_PATH":PROTOCOL_PATH,"CONFIG_PATH":CONFIG_PATH,"INSPECT_PATH":INSPECT_PATH,"FOLD_MANIFEST_PATH":FOLD_MANIFEST_PATH,"HISTORICAL_DIR":HISTORICAL_DIR,"SMOKE_DIR":SMOKE_DIR,"WORK_DIR":WORK_DIR,"TRIALS_DIR":TRIALS_DIR,"TRIAL_MANIFEST_PATH":TRIAL_MANIFEST_PATH,"ALL_TRIALS_PATH":ALL_TRIALS_PATH,"SEARCH_SUMMARY_PATH":SEARCH_SUMMARY_PATH,"HISTORICAL_PATHS":HISTORICAL_PATHS,"HISTORICAL_ANCHORS":HISTORICAL_ANCHORS,"REQUIRED_TRACKED":REQUIRED_TRACKED,"LOCKED_DEPENDENCIES":LOCKED_DEPENDENCIES}
    for name,value in values.items(): setattr(engine,name,value)
    overrides={"protocol":protocol,"validate_oof":validate_oof,"prediction_rows":prediction_rows,"historical_payload":historical_payload,"build_model":build_model,"make_training_objects":make_training_objects,"run_smoke":run_smoke,"safety":safety,"rank_key":rank_key,"is_historical_exact":is_historical_exact,"run_search":run_search,"historical_reference":historical_reference,"target_gate":target_gate,"final_decision":final_decision,"render_report":render_report,"run_formal":run_formal}
    for name,value in overrides.items(): setattr(engine,name,value)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest="mode",required=True); subs.add_parser("inspect")
    for mode in ("smoke","search","formal","resume"):
        child=subs.add_parser(mode); child.add_argument("--device",default="cuda:0")
    return parser.parse_args()


def main() -> None:
    install_profile(); args=parse_args()
    if args.mode=="inspect": engine.run_inspect()
    elif args.mode=="smoke": run_smoke(args.device)
    elif args.mode in ("search","resume"): run_search(args.device)
    elif args.mode=="formal": run_formal(args.device)
    else: raise engine.InvariantError(args.mode)


if __name__=="__main__": main()
