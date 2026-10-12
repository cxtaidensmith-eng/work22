"""End-to-end fixed versus trainable neighborhood pilot on original fold 1."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT,setup,dump,sha,timestamp,scores
from reference_selection_experiment import read,model_hash
from Utils import SET_Random,CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer,optimizer_manifest
from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel
from Model.learned_topology import LearnedTopologyModel

SEEDS=(0,1,2)
MODES=('fixed','learned')
K,TEMPERATURE,EPOCHS=10,.1,1000
OUT=ROOT/'Result/learned_topology_pilot'


def verified_setup():
    path=ROOT/'Result/reference_selection/PROTOCOL.json'
    original=read(path)
    for source,digest in original['code_sha256'].items():
        assert sha(ROOT/source)==digest,source
    cfg,dd,data,graphs,paths=setup(k=K)
    assert {Path(p).name:sha(p) for p in paths}==original['data_sha256']
    assert all(getattr(cfg,k)==v for k,v in original['original_config'].items())
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()==original['candidate_hash']
    return cfg,dd,data,graphs,original


def make_model(dd,cfg,data,graphs,seed,mode):
    SET_Random(seed)
    return LearnedTopologyModel(dd,cfg,graphs,data['Feature'],seed,mode,K,TEMPERATURE).to(cfg.Device)


def predictions(path,logits,data,dd,model):
    test=data['Mask'][0][1]
    np.savez_compressed(path,logits=logits[test].detach().cpu().numpy(),
        labels=data['Label'][test].cpu().numpy(),source_row=np.asarray(dd['Index'])[test.cpu().numpy()],
        neighbors=model.propagation.last_neighbors.cpu().numpy(),
        support=model.propagation.last_support.cpu().numpy())


def initialization_check(dd,cfg,data,graphs,seed):
    results=[];reference=None
    for mode in MODES:
        model=make_model(dd,cfg,data,graphs,seed,mode).eval()
        with torch.no_grad():logits=model(data['Feature'])[0]
        assert torch.equal(model.propagation.last_support,model.propagation.fixed_support)
        if reference is None:reference=logits.detach().clone()
        else:torch.testing.assert_close(logits,reference,atol=0,rtol=0)
        results.append(dict(mode=mode,model_sha256=model_hash(model),
            parameters=sum(p.numel() for p in model.parameters())))
        del model,logits
    assert results[0]['model_sha256']==results[1]['model_sha256']
    SET_Random(seed)
    original=RelationSheafModel(dd,cfg,graphs,'content',training_seed=seed).to(cfg.Device).eval()
    with torch.no_grad():old=original(data['Feature'])[0]
    error=float((reference-old).abs().max())
    torch.testing.assert_close(reference,old,atol=2e-5,rtol=2e-5)
    return dict(seed=seed,arms=results,original_c_max_abs_logit_error=error,
        identical_initial_neighbors=True,identical_initial_parameters=True)


def smoke(cfg,dd,data,graphs):
    evidence=initialization_check(dd,cfg,data,graphs,0)
    fit,test=data['Mask'][0]
    train_y=torch.where(fit,data['Label'],torch.zeros_like(data['Label']))
    rows=[]
    for mode in MODES:
        model=make_model(dd,cfg,data,graphs,0,mode)
        optimizer=relation_optimizer(model,cfg,0.)
        criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();started=time.perf_counter()
        for epoch in range(10):
            model.train();optimizer.zero_grad(set_to_none=True)
            logits,embedding,aux=model(data['Feature'])
            loss=criterion(logits,train_y,fit,embedding,aux)
            loss.backward()
            assert torch.isfinite(model.propagation.relation_blend.grad).all()
            assert float(model.propagation.relation_blend.grad.abs().sum())>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step();model.eval()
            with torch.no_grad():model(data['Feature'])
        torch.cuda.synchronize()
        rows.append(dict(mode=mode,seconds_per_epoch=(time.perf_counter()-started)/10,
            peak_mb=torch.cuda.max_memory_allocated()/2**20,topology=model.propagation.topology_diagnostics()))
        del model,optimizer,criterion,logits,embedding,aux,loss
        torch.cuda.empty_cache()
    dump(ROOT/'Result/learned_topology_smoke.json',dict(initialization=evidence,runs=rows,
        note='Engineering-only check; test accuracy not inspected or used for tuning'))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def summarize():
    records=[]
    for seed in SEEDS:
        for mode in MODES:
            path=OUT/f'seed_{seed}'/mode/'complete.json'
            if path.exists():records.append(read(path))
    result=dict(completed=len(records),planned=6,records=records,updated_at=timestamp())
    dump(OUT/'summary.json',result)
    return result


def train_arm(cfg,dd,data,graphs,seed,mode):
    folder=OUT/f'seed_{seed}'/mode
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    model=make_model(dd,cfg,data,graphs,seed,mode)
    initial_hash=model_hash(model)
    optimizer=relation_optimizer(model,cfg,0.)
    dump(folder/'optimizer_groups.json',optimizer_manifest(model,optimizer))
    scheduler=CustomCosineAnnealingLR(optimizer,T_max=cfg.T_max,eta_min=cfg.Lr_Min)
    criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    x,y=data['Feature'],data['Label']
    fit,test=data['Mask'][0]
    train_y=torch.where(fit,y,torch.zeros_like(y))
    diagnostics=[];history=[];previous=None
    best_correct=-1;best=None
    fields=['epoch','train_loss','test_acc','test_nll','lr','blend_grad_norm','query_grad_norm','seconds']
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for epoch in range(1,EPOCHS+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            logits,embedding,auxiliary=model(x)
            loss=criterion(logits,train_y,fit,embedding,auxiliary)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            blend_grad=float(model.propagation.relation_blend.grad.norm())
            query_grad=float(model.propagation.edge_query.weight.grad.norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step();model.eval()
            with torch.no_grad():
                logits=model(x)[0]
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite predictions')
                correct=int((logits[test].argmax(-1)==y[test]).sum())
                nll=float(F.cross_entropy(logits[test],y[test]))
            row=dict(epoch=epoch,train_loss=float(loss.detach()),test_acc=correct/int(test.sum()),test_nll=nll,
                lr=optimizer.param_groups[0]['lr'],blend_grad_norm=blend_grad,query_grad_norm=query_grad,
                seconds=time.perf_counter()-started)
            history.append(row);writer.writerow(row)
            if correct>best_correct:
                best_correct=correct
                best=dict(epoch=epoch,**scores(logits[test],y[test]))
                torch.save(dict(state_dict=model.state_dict(),seed=seed,mode=mode,epoch=epoch),folder/'best.pt')
                predictions(folder/'predictions.npz',logits,data,dd,model)
                dump(folder/'best_topology.json',model.propagation.topology_diagnostics())
            if epoch in (1,20) or epoch%100==0:
                topology=model.propagation.topology_diagnostics(previous)
                assert topology['symmetric'] and topology['no_self_edges']
                if mode=='fixed':assert torch.equal(model.propagation.last_support,model.propagation.fixed_support)
                previous=model.propagation.last_support.clone()
                diagnostics.append(dict(epoch=epoch,**topology));dump(folder/'topology_trace.json',diagnostics)
                stream.flush()
                status=dict(state='training',seed=seed,mode=mode,epoch=epoch,best_acc=best['acc'],acc=row['test_acc'],
                    new_choice_fraction=float(np.mean([r['new_choice_fraction'] for r in topology['modalities']])),
                    updated_at=timestamp())
                dump(OUT/'status.json',status);print(json.dumps(status),flush=True)
            scheduler.step()
    torch.save(dict(state_dict=model.state_dict(),seed=seed,mode=mode,epoch=EPOCHS),folder/'last.pt')
    predictions(folder/'last_predictions.npz',logits,data,dd,model)
    record=dict(seed=seed,mode=mode,fold=1,best=best,last=dict(epoch=EPOCHS,**scores(logits[test],y[test])),
        mean_last100_acc=float(np.mean([r['test_acc'] for r in history[-100:]])),
        parameters=sum(p.numel() for p in model.parameters()),initial_model_sha256=initial_hash,
        best_checkpoint_sha256=sha(folder/'best.pt'),last_checkpoint_sha256=sha(folder/'last.pt'),
        seconds=time.perf_counter()-started,peak_mb=torch.cuda.max_memory_allocated()/2**20,
        last_topology=diagnostics[-1],completed_at=timestamp())
    dump(folder/'complete.json',record)
    summarize()
    print(json.dumps(dict(event='arm_complete',seed=seed,mode=mode,best=best['acc'],last=record['last']['acc'])),flush=True)
    del model,optimizer,criterion,scheduler,logits,embedding,auxiliary,loss
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,graphs,original=verified_setup()
    if args.smoke:
        smoke(cfg,dd,data,graphs);return
    if OUT.exists() and not args.resume:raise FileExistsError(OUT)
    plan=dict(experiment='End-to-end fixed-budget learned topology pilot',folds=[1],seeds=list(SEEDS),modes=list(MODES),
        epochs=EPOCHS,initialization='From scratch; no freezing, pretraining, distillation, or warm-up',
        primary_selection='First maximum TEST ACC, original protocol; exploratory',
        secondary='Fixed epoch 1000 and last100 mean',
        learned_scope='All off-diagonal cohort pairs; 10 outgoing choices per relation modality, reverse-edge union',
        symmetric_degree_caveat='Outgoing budget matched; post-union degrees can differ and must be reported',
        temperature=TEMPERATURE,
        relation_score='raw modal cosine + tanh(beta_m) * cosine(existing edge_query of modal token)',
        weight='Original C formula with raw modal cosine replaced by the same relation score in BOTH arms',
        extra_parameters='Six beta_m in both arms, initialized to zero; existing encoder/query reused',
        selector='Hard stable top-k forward; implicit-gradient fixed-mass logistic soft-union straight-through backward',
        surrogate_caveat='Biased surrogate gradient, inspired by differentiable selection; not original DGG reproduction',
        operator='Original content-only cooperative operator, two passes, D+I normalization; matched dense kernels',
        same_initialization='Both arms equal common weights, supports and predictions; initial predictions checked against C',
        training='Original Adam, graph decay 0, patient decay .0007, original scheduler/loss/noise/dropout, all parameters trainable',
        labels='Original training-mask supervised losses; all-cohort unsupervised orthogonality and original class weights retained',
        baseline='Matched fixed graph uses same six extra scoring parameters; historical exact C reported separately',
        expansion='Finish three seeds; no automatic ten-fold expansion',original_protocol_sha256=sha(ROOT/'Result/reference_selection/PROTOCOL.json'),
        code_sha256={p:sha(ROOT/p) for p in ('Model/learned_topology.py','learned_topology_pilot.py','tests/test_learned_topology.py')})
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():assert read(OUT/'PLAN.json')==plan,'Frozen plan changed'
    else:dump(OUT/'PLAN.json',plan)
    try:
        for seed in SEEDS:
            evidence=initialization_check(dd,cfg,data,graphs,seed)
            dump(OUT/f'initialization_seed_{seed}.json',evidence)
            for mode in MODES:train_arm(cfg,dd,data,graphs,seed,mode)
        summary=summarize();assert summary['completed']==6
        dump(OUT/'COMPLETE.json',summary)
        dump(OUT/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),updated_at=timestamp()));raise


if __name__=='__main__':main()
