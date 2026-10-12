"""Bounded mechanism diagnosis, not a tuned performance benchmark."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from relation_experiment import ROOT,setup,dump,sha,timestamp
from alternating_cooperative_experiment import read,verify_manifest
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Loss.loss_fn import orthogonality_loss
from Utils import SET_Random,CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer

OUT=ROOT/'Result/abide_c_diagnosis_20260927'
REF=ROOT/'Result/c_abide_seed0_20260924'
CHECKS={1,2,3,5,10,20,25,50,100,150,200}


def activation(v):
    v=v.detach()
    return dict(norm=float(v.norm()),sample_std=float(v.std(0).mean()),
        positive_fraction=float((v>0).float().mean()),zero_fraction=float((v==0).float().mean()),
        dead_channels=int((v.abs().max(0).values==0).sum()) if v.ndim==2 else None)


def evaluate(model,data,fold):
    trace={};handles=[]
    for name,module in [('tokens',model.modal_token_encoder),('graph',model.propagation),
        ('semantic',model.Message_MLP),('own',model.Global_Message),
        ('classifier_pre',model.classifier[0]),('classifier_active',model.classifier[1])]:
        def hook(mod,args,out,key=name):trace[key]=activation(out)
        handles.append(module.register_forward_hook(hook))
    model.eval()
    try:
        with torch.no_grad():z,emb,aux=model(data['Feature'])
    finally:
        for h in handles:h.remove()
    fit,test=data['Mask'][fold-1];y=data['Label']
    for label,mask in [('train',fit),('test',test)]:
        pred=z[mask].argmax(-1)
        trace[label]=dict(acc=float((pred==y[mask]).double().mean()),
            pred_counts=torch.bincount(pred,minlength=2).tolist(),
            logit_std=float(z[mask].std(0).mean()),margin_std=float((z[mask,1]-z[mask,0]).std()))
    trace['gates']=model.modal_gate_logit.sigmoid().detach().cpu().tolist()
    trace['aux_train_acc']=[float((a[fit].argmax(-1)==(y[fit]==i).long()).double().mean()) for i,a in enumerate(aux)]
    return trace


def parts(model,criterion,z,emb,aux,y,fit):
    return dict(ce=criterion.CE_loss(z[fit],y[fit]),
        aux=sum(criterion.aux_loss_dict[f'aux_loss_{i}'](a[fit],(y[fit]==i).long()) for i,a in enumerate(aux)),
        ortho=criterion.rate*orthogonality_loss(emb))


def run_arm(cfg,dd,data,graphs,mode,fold=1):
    SET_Random(0);model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    if mode=='leaky_head':model.classifier[1]=nn.LeakyReLU(.01)
    if mode=='no_graph':model.propagation.forward=lambda tokens:model.propagation.output_norm(tokens)
    optimizer=relation_optimizer(model,cfg,0.)
    if mode=='low_lr':
        for group in optimizer.param_groups:group['lr']=.001
    scheduler=CustomCosineAnnealingLR(optimizer,T_max=cfg.T_max,eta_min=cfg.Lr_Min)
    criterion=MSCQGNLoss(dd,cfg.Device,rate=0. if mode=='no_ortho' else cfg.Loss_rate,
        aux_loss_weight=cfg.aux_loss_weight)
    x,y=data['Feature'],data['Label'];fit,test=data['Mask'][fold-1]
    records=[];curve=[];start=time.perf_counter()
    for epoch in range(1,201):
        model.train();optimizer.zero_grad(set_to_none=True)
        z,emb,aux=model(x);components=parts(model,criterion,z,emb,aux,y,fit)
        loss=components['ce']+components['aux']+components['ortho']
        assert torch.isfinite(loss)
        details={}
        if epoch in CHECKS:
            params=list(model.parameters())
            for name,value in components.items():
                grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                details[name]=dict(value=float(value.detach()),grad_norm=float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None))))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
        optimizer.step();model.eval()
        with torch.no_grad():zz=model(x)[0]
        curve.append(dict(epoch=epoch,train_loss=float(loss.detach()),test_acc=float((zz[test].argmax(-1)==y[test]).double().mean())))
        if epoch in CHECKS:
            records.append(dict(epoch=epoch,loss_parts=details,total_gradient_norm=float(norm),**evaluate(model,data,fold)))
            dump(OUT/f'{mode}_fold{fold}_trace.json',records)
            dump(OUT/'status.json',dict(state='running',mode=mode,fold=fold,epoch=epoch,updated_at=timestamp()))
            print(json.dumps(dict(mode=mode,fold=fold,epoch=epoch,train=records[-1]['train'],test=records[-1]['test'],head=records[-1]['classifier_active'])),flush=True)
        scheduler.step()
    # Eval hooks and gradient probes must not perturb the original RNG sequence.
    if mode=='baseline':
        import csv
        with (REF/f'content/fold_{fold:02d}/epochs.csv').open() as stream:old=list(csv.DictReader(stream))[:200]
        differences=[abs(a['train_loss']-float(b['train_loss'])) for a,b in zip(curve,old)]
        replay=dict(max_train_loss_difference=max(differences),test_acc_matches=sum(abs(a['test_acc']-float(b['test_acc']))<1e-10 for a,b in zip(curve,old)))
    else:replay=None
    result=dict(mode=mode,fold=fold,epochs=200,final=records[-1],trace=records,curve=curve,
        seconds=time.perf_counter()-start,baseline_reproduction=replay)
    dump(OUT/f'{mode}_fold{fold}.json',result)
    del model,optimizer,criterion,z,zz,emb,aux,loss,components
    torch.cuda.empty_cache()
    return result


def main():
    if OUT.exists():raise FileExistsError(OUT)
    cfg,dd,data,graphs,paths=setup(0,'Config/c_abide_transfer.ini',10)
    plan=read(REF/'PLAN.json');verify_manifest(plan['code_sha256'])
    assert {Path(p).name:sha(p) for p in paths}==plan['protocol']['data_sha256']
    indices=[i for group in dd['Modal_Index'] for i in group]
    assert sorted(indices)==list(range(data['Feature'].shape[1]))
    OUT.mkdir()
    dump(OUT/'PLAN.json',dict(created_at=timestamp(),folds=[1],epochs=200,
        modes=['baseline','low_lr','no_ortho','no_graph','leaky_head'],
        definitions=dict(low_lr='lr .006->.001 only',no_ortho='orthogonality coefficient .0001->0 only',
            no_graph='replace graph residual with same output LayerNorm only',leaky_head='classifier ReLU->LeakyReLU(.01) only'),
        scope='Bounded diagnostic; selected known failing fold1; no claim of heldout confirmation; no checkpoint selection or main-model changes.',
        source_sha256=sha(__file__),reference=plan['protocol']))
    SET_Random(0);model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    replay=[]
    for fold in range(1,11):
        cp=torch.load(REF/f'content/fold_{fold:02d}/best.pt',map_location=cfg.Device,weights_only=False)
        model.load_state_dict(cp['state_dict']);replay.append(dict(fold=fold,epoch=cp['best']['epoch'],**evaluate(model,data,fold)))
    dump(OUT/'best_checkpoint_diagnosis.json',replay)
    del model
    results=[]
    for mode in ['baseline','low_lr','no_ortho','no_graph','leaky_head']:
        results.append(run_arm(cfg,dd,data,graphs,mode))
    dump(OUT/'summary.json',[{k:r[k] for k in ('mode','fold','epochs','seconds','final','baseline_reproduction')} for r in results])
    dump(OUT/'status.json',dict(state='complete',updated_at=timestamp(),runs=len(results)))


if __name__=='__main__':main()
