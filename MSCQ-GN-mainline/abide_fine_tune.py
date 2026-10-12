"""Frozen-network ABIDE local search: 8 coarse + 4 adaptive refinements, seed0."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import diagnose_abide_capacity as runner
from abide_noise_adaptation import make_model as noise_model
from abide_late_readout_experiment import inputs
from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import read, verify_manifest
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from Loss import MSCQGNLoss

OUT=ROOT/'Result/abide_fine_tune_20260927'
PREV=ROOT/'Result/abide_noise_adaptation_20260927'
REF=PREV/'full/N1'
SCREEN=(1,9)
CHECK=(3,7)
BASE=dict(lr=.001,weight_decay=.002,aux_loss_weight=.25,Drop_rate=.45)
COARSE={
    'L06':dict(lr=.0006),'L08':dict(lr=.0008),'L12':dict(lr=.0012),
    'W03':dict(weight_decay=.003),'W04':dict(weight_decay=.004),
    'A015':dict(aux_loss_weight=.15),'A04':dict(aux_loss_weight=.4),
    'D035':dict(Drop_rate=.35),
}


def configure(cfg,spec):
    conf=copy.copy(cfg)
    for k,v in spec.items():setattr(conf,k,v)
    return conf


def make_model(dd,cfg,graphs,mode,training_seed=0):
    return noise_model(dd,cfg,graphs,mode,training_seed=training_seed,arm='N1')


def prepare(cfg,dd,data,graphs,protocol):
    prev=read(PREV/'PLAN.json');verify_manifest(prev['code_sha256'])
    assert read(PREV/'COMPLETE.json')['all_audits_passed']
    assert sha(PREV/'data_splits_graphs.npz')==prev['split_graph_archive_sha256']
    with np.load(PREV/'data_splits_graphs.npz') as p:
        np.testing.assert_array_equal(p['source_row'],dd['Index'])
        np.testing.assert_array_equal(p['labels'],data['Label'].cpu().numpy())
        np.testing.assert_array_equal(p['test_masks'],torch.stack([m for _,m in data['Mask']]).cpu().numpy())
        for i,g in enumerate(graphs):np.testing.assert_array_equal(p[f'edge_{i}'],g['edge'].numpy())
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);return plan
    sources=dict(prev['code_sha256']);sources[Path(__file__).name]=sha(__file__)
    references={}
    for f in range(1,11):
        for name in ('best.pt','last.pt','best_predictions.npz','last_predictions.npz','epochs.csv','complete.json','noise_and_prediction_audit.json'):
            p=REF/f'fold_{f:02d}'/name;references[str(p.relative_to(ROOT))]=sha(p)
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    shutil.copy2(PREV/'data_splits_graphs.npz',OUT/'data_splits_graphs.npz')
    plan=dict(created_at=timestamp(),baseline=BASE,coarse=COARSE,seed=0,epochs=1000,screen_folds=SCREEN,
        check_folds=CHECK,maximum_new_runs=34,protocol=protocol,
        fixed='C content network, 726551 parameters, N1 noise (.15,.15,.15,.03), graph decay .0005, orth1e-6, scheduler1000/minlr.0001, gradient clip1, original data/graphs/splits/label smoothing.',
        refine='Rank N1 and all coarse configurations by two-fold best ACC, tie by late100 mean then earlier insertion. Around leader create four configs: lr*0.85, lr*1.15, weight_decay*0.75, weight_decay*1.25, rounded12 decimals; reuse exact duplicates. Other leader settings retained.',
        selection='At most top2 distinct candidates: two-fold mean gain >=0.005 vs N1; no fold loses more than one correct prediction; last predicts both classes. Evaluate folds3,7. Four-fold gate: mean gain >=0.005, at least3 folds not worse, all last checkpoints nonconstant. Highest ranked eligible fills remaining6 folds. Ties: mean best ACC, late100 mean, insertion order. Retain N1 if completed tenfold candidate does not improve mean ACC. No per-fold config selection.',
        scope='Same original first maximum TEST ACC selection. Adaptive exploratory tuning with reused test folds; no independent validation or global optimum claim. No new seeds, architecture, thresholds, data preprocessing, graph support, optimizer algorithm or new candidates outside this plan.',
        code_sha256=sources,reference_sha256=references,split_graph_archive_sha256=sha(OUT/'data_splits_graphs.npz'))
    for name in sources:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json',plan);return plan


def preflight(cfg,dd,data,graphs,specs):
    if (OUT/'preflight.json').exists():return
    with (REF/'fold_01/epochs.csv').open() as s:expected=list(csv.DictReader(s))[:3]
    params=None;rng=None;eval_ref=None;checks={}
    for name,spec in specs.items():
        conf=configure(cfg,spec);SET_Random(0);model=make_model(dd,conf,graphs,'content').to(conf.Device)
        if params is None:
            params={n:p.detach().clone() for n,p in model.named_parameters()};rng=torch.get_rng_state()
        else:
            assert torch.equal(rng,torch.get_rng_state())
            for n,p in model.named_parameters():torch.testing.assert_close(p,params[n],atol=0,rtol=0)
        assert sum(p.numel() for p in model.parameters())==726551
        torch.testing.assert_close((model._modal_noise_std*model.noise_scale).cpu(),torch.tensor([.15,.15,.15,.03]))
        model.eval()
        with torch.no_grad():z=model(data['Feature'])[0]
        if eval_ref is None:eval_ref=z.clone()
        else:
            torch.testing.assert_close(z,eval_ref,atol=2e-7,rtol=1e-6)
            torch.testing.assert_close(z.argmax(1),eval_ref.argmax(1),atol=0,rtol=0)
        opt=relation_optimizer(model,conf,.0005)
        assert [g['weight_decay'] for g in opt.param_groups]==[spec['weight_decay'],.0005]
        assert all(g['lr']==spec['lr'] for g in opt.param_groups)
        criterion=MSCQGNLoss(dd,conf.Device,rate=conf.Loss_rate,aux_loss_weight=conf.aux_loss_weight)
        sched=CustomCosineAnnealingLR(opt,T_max=conf.T_max,eta_min=conf.Lr_Min);losses=[]
        for i in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            z,emb,aux=model(data['Feature']);loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
            assert torch.isfinite(loss)
            if name=='N1':assert abs(float(loss.detach())-float(expected[i]['training_loss']))<2e-5
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),conf.grad_clip,error_if_nonfinite=True);opt.step();sched.step()
            losses.append(float(loss.detach()))
        checks[name]=dict(spec=spec,parameters=726551,first3_losses=losses,passed=True)
        del model,opt,criterion,z,emb,aux,loss
    dump(OUT/'preflight.json',checks)
    del params,eval_ref;torch.cuda.empty_cache()


def audit(folder,name,fold,cfg,dd,data,graphs):
    model=make_model(dd,cfg,graphs,'content').to(cfg.Device).eval();noise=model._modal_noise_std.clone()
    test=data['Mask'][fold-1][1];details=[]
    for kind in ('best','last'):
        cp=torch.load(folder/f'{kind}.pt',map_location=cfg.Device,weights_only=False)
        assert (cp['arm'],cp['fold'],cp['seed'])==(name,fold,0)
        model.load_state_dict(cp['state_dict']);torch.testing.assert_close(model._modal_noise_std,noise,atol=0,rtol=0)
        with torch.no_grad():z=model(data['Feature'])[0][test].cpu().numpy()
        with np.load(folder/f'{kind}_predictions.npz') as p:
            np.testing.assert_allclose(z,p['logits'],atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(z.argmax(1),p['logits'].argmax(1))
            np.testing.assert_array_equal(data['Label'][test].cpu().numpy(),p['labels'])
            if kind=='best':np.testing.assert_array_equal(np.array(dd['Index'])[test.cpu().numpy()],p['row_index'])
        details.append(dict(kind=kind,sha256=sha(folder/f'{kind}.pt'),pred_counts=np.bincount(z.argmax(1),minlength=2).tolist()))
    dump(folder/'fine_audit.json',dict(passed=True,checks=details))
    return details[-1]['pred_counts']


def records_for(name,folds):
    root=REF if name=='N1' else OUT/'runs'/name
    return [read(root/f'fold_{f:02d}/complete.json') for f in folds]


def rank(names,folds):
    def score(name):
        rr=records_for(name,folds)
        return (np.mean([r['best']['acc'] for r in rr]),np.mean([r['late100_mean_acc'] for r in rr]))
    return sorted(names,key=score,reverse=True)


def gate(name,folds,checking=False):
    rr=records_for(name,folds);base=records_for('N1',folds)
    gains=[a['best']['acc']-b['best']['acc'] for a,b in zip(rr,base)]
    nonconstant=all(min(read(OUT/'runs'/name/f'fold_{f:02d}/fine_audit.json')['checks'][-1]['pred_counts'])>0 for f in folds)
    if checking:passing=sum(d>=-1e-12 for d in gains)>=3
    else:
        passing=True
        for f,d in zip(folds,gains):
            with np.load(REF/f'fold_{f:02d}/best_predictions.npz') as p:n=len(p['labels'])
            passing=passing and d>=-1/n-1e-12
    return dict(passed=bool(np.mean(gains)>=.005-1e-12 and passing and nonconstant),gains=gains,
        mean_gain=float(np.mean(gains)),nonconstant=nonconstant)


def report(specs,selection=None):
    lines=['# ABIDE C精细调参','',
        '固定seed0、原十折、每折1000轮及首次最高测试ACC选轮。结构与图不变；N1为完整十折基线。筛选成绩不能视为十折成绩。','',
        '| 配置 | 学习率 | 权重衰减 | 辅助权重 | Dropout | 第1折最佳 | 第9折最佳 | 两折均值 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,spec in specs.items():
        root=REF if name=='N1' else OUT/'runs'/name
        if not all((root/f'fold_{f:02d}/complete.json').exists() for f in SCREEN):continue
        rr=records_for(name,SCREEN);a=[r['best']['acc']*100 for r in rr]
        lines.append(f"| {name} | {spec['lr']:.7f} | {spec['weight_decay']:.6f} | {spec['aux_loss_weight']:.2f} | {spec['Drop_rate']:.2f} | {a[0]:.2f}% | {a[1]:.2f}% | {np.mean(a):.2f}% |")
    if selection is not None:lines+=['',f"四折后晋级完整十折：{selection['selected'] or '无'}。"]
    lines+=['','这是自适应、重复使用测试折的探索性调参。所有配置与结果均保留；未按折挑选不同配置，也未改动阈值。']
    (OUT/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_one(name,fold,spec,cfg,dd,data,graphs,stage):
    dump(OUT/'status.json',dict(state=stage,arm=name,fold=fold,spec=spec,updated_at=timestamp()))
    conf=configure(cfg,spec);folder=OUT/'runs'/name/f'fold_{fold:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'effective_settings.json').exists():assert read(folder/'effective_settings.json')==spec
    else:dump(folder/'effective_settings.json',spec)
    r=runner.run(name,fold,conf,dd,data,graphs)
    counts=audit(folder,name,fold,conf,dd,data,graphs)
    assert len(counts)==2;torch.cuda.empty_cache()
    return r


def analyze(specs,selection,dd):
    rows=[]
    for name in specs:
        root=REF if name=='N1' else OUT/'runs'/name
        for f in range(1,11):
            folder=root/f'fold_{f:02d}'
            if not (folder/'complete.json').exists():continue
            r=read(folder/'complete.json')
            with (folder/'epochs.csv').open() as s:curve=list(csv.DictReader(s))
            assert [int(x['epoch']) for x in curve]==list(range(1,1001))
            assert np.argmax([float(x['test_acc']) for x in curve])+1==r['best']['epoch']
            with np.load(folder/'best_predictions.npz') as p:z=p['logits'];y=p['labels'];rid=p['row_index']
            with np.load(REF/f'fold_{f:02d}/best_predictions.npz') as p:
                np.testing.assert_array_equal(rid,p['row_index']);np.testing.assert_array_equal(y,p['labels']);old=p['logits'].argmax(1)
            prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);pred=z.argmax(1)
            assert float((pred==y).mean())==r['best']['acc']
            rows.append(dict(arm=name,fold=f,epoch=r['best']['epoch'],acc=r['best']['acc'],baseline_acc=float((old==y).mean()),
                last_acc=r['last']['test_acc'],late100_acc=r['late100_mean_acc'],macro_f1=float(f1_score(y,pred,average='macro')),
                balanced_acc=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob[:,1])),
                corrected=int(((old!=y)&(pred==y)).sum()),harmed=int(((old==y)&(pred!=y)).sum())))
    full={}
    for name in specs:
        rr=[r for r in rows if r['arm']==name]
        if len(rr)!=10:continue
        full[name]={k:float(np.mean([r[k] for r in rr])) for k in ['acc','last_acc','late100_acc','macro_f1','balanced_acc','auc']}
        full[name].update(std_acc=float(np.std([r['acc'] for r in rr])),corrected=sum(r['corrected'] for r in rr),harmed=sum(r['harmed'] for r in rr),
            wins=sum(r['acc']>r['baseline_acc'] for r in rr),other6_delta=float(np.mean([r['acc']-r['baseline_acc'] for r in rr if r['fold'] not in SCREEN+CHECK])))
    chosen='N1'
    if selection['selected'] and full[selection['selected']]['acc']>full['N1']['acc']:chosen=selection['selected']
    result=dict(rows=rows,full=full,retained=chosen,selection=selection);dump(OUT/'analysis.json',result)
    text=['','## 完整十折对照','', '| 配置 | 最佳ACC | 宏F1 | 平衡ACC | 概率AUC | 第1000轮ACC |','|---|---:|---:|---:|---:|---:|']
    for name,m in full.items():text.append(f"| {name} | {100*m['acc']:.4f}% | {100*m['macro_f1']:.4f}% | {100*m['balanced_acc']:.4f}% | {100*m['auc']:.4f}% | {100*m['last_acc']:.4f}% |")
    text+=['',f'本轮结束后保留配置：{chosen}。未达到晋级条件的候选不外推十折成绩。']
    with (OUT/'结果.md').open('a',encoding='utf-8') as s:s.write('\n'.join(text)+'\n')
    return result


def main():
    cfg,dd,data,graphs,protocol=inputs();plan=prepare(cfg,dd,data,graphs,protocol)
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    if (OUT/'COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as s:s.write(str(os.getpid()))
    specs={'N1':dict(BASE),**{n:{**BASE,**v} for n,v in COARSE.items()}}
    runner.OUT=OUT/'runs';runner.RelationSheafModel=make_model
    try:
        preflight(cfg,dd,data,graphs,specs)
        for name in COARSE:
            for fold in SCREEN:run_one(name,fold,specs[name],cfg,dd,data,graphs,'coarse')
            report(specs)
        leader=rank(list(specs),SCREEN)[0];refine={};aliases={}
        for name,key,factor in [('R_L85','lr',.85),('R_L115','lr',1.15),('R_W75','weight_decay',.75),('R_W125','weight_decay',1.25)]:
            spec=dict(specs[leader]);spec[key]=round(spec[key]*factor,12)
            same=next((n for n,v in specs.items() if v==spec),None)
            if same:aliases[name]=same
            else:refine[name]=spec;specs[name]=spec
        dump(OUT/'refinement.json',dict(leader=leader,candidates=refine,aliases=aliases));dump(OUT/'all_settings.json',specs)
        for name in refine:
            for fold in SCREEN:run_one(name,fold,specs[name],cfg,dd,data,graphs,'refine')
            report(specs)
        gates={n:gate(n,SCREEN) for n in specs if n!='N1'}
        finalists=rank([n for n,g in gates.items() if g['passed']],SCREEN)[:2]
        dump(OUT/'screen_selection.json',dict(gates=gates,finalists=finalists))
        for name in finalists:
            for fold in CHECK:run_one(name,fold,specs[name],cfg,dd,data,graphs,'check')
        checkgates={n:gate(n,SCREEN+CHECK,True) for n in finalists}
        eligible=rank([n for n,g in checkgates.items() if g['passed']],SCREEN+CHECK)
        selected=eligible[0] if eligible else None
        selection=dict(selected=selected,finalists=finalists,check_gates=checkgates);dump(OUT/'selection.json',selection)
        report(specs,selection)
        if selected:
            for fold in range(1,11):
                if fold not in SCREEN+CHECK:run_one(selected,fold,specs[selected],cfg,dd,data,graphs,'full')
        result=analyze(specs,selection,dd)
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz')==plan['split_graph_archive_sha256']
        count=len(list((OUT/'runs').glob('*/fold_*/complete.json')));assert count<=34
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),new_runs=count,retained=result['retained'],selected=selected,all_audits_passed=True))
        dump(OUT/'status.json',dict(state='complete',retained=result['retained'],updated_at=timestamp()))
        print(json.dumps(dict(retained=result['retained'],full=result['full'])),flush=True)
    except BaseException as e:
        dump(OUT/'status.json',dict(state='failed',error=repr(e),updated_at=timestamp()));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
