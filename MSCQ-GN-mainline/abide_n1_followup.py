"""Bounded N4/N5/N6 screen against N1; at most one tenfold extension."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy
import csv
import shutil
from pathlib import Path
from functools import partial
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import diagnose_abide_capacity as runner
from abide_noise_adaptation import make_model as original_factory
from abide_late_readout_experiment import inputs
from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import read, verify_manifest
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from Loss import MSCQGNLoss

OUT=ROOT/'Result/abide_n1_followup_20260927'
PREV=ROOT/'Result/abide_noise_adaptation_20260927'
REF=PREV/'full/N1'
FOLDS=(1,9)
ARMS=('N4','N5','N6')
SETTINGS={'N1':(.03,.25),'N4':(.01,.25),'N5':(.06,.25),'N6':(.03,.10)}


def config_for(cfg,arm):
    result=copy.copy(cfg)
    result.aux_loss_weight=SETTINGS[arm][1]
    return result


def make_model(dd,cfg,graphs,mode,training_seed=0,arm='N1'):
    assert cfg.aux_loss_weight==SETTINGS[arm][1]
    model=original_factory(dd,cfg,graphs,mode,training_seed=training_seed,arm='N1')
    with torch.no_grad():
        model._modal_noise_std[list(dd['Modal_Name']).index('Correlation')]=SETTINGS[arm][0]/model.noise_scale
    return model


def prepare(cfg,dd,data,graphs,protocol):
    prev=read(PREV/'PLAN.json')
    verify_manifest(prev['code_sha256']);verify_manifest(prev['reference_sha256'])
    assert read(PREV/'COMPLETE.json')['all_audits_passed']
    assert sha(PREV/'data_splits_graphs.npz')==prev['split_graph_archive_sha256']
    with np.load(PREV/'data_splits_graphs.npz') as a:
        np.testing.assert_array_equal(a['source_row'],dd['Index'])
        np.testing.assert_array_equal(a['labels'],data['Label'].cpu().numpy())
        np.testing.assert_array_equal(a['test_masks'],torch.stack([m for _,m in data['Mask']]).cpu().numpy())
        for i,g in enumerate(graphs):np.testing.assert_array_equal(a[f'edge_{i}'],g['edge'].numpy())
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);return plan
    sources=dict(prev['code_sha256']);sources[Path(__file__).name]=sha(__file__)
    refs={}
    for f in range(1,11):
        for name in ('complete.json','epochs.csv','best.pt','last.pt','best_predictions.npz','last_predictions.npz','noise_and_prediction_audit.json'):
            p=REF/f'fold_{f:02d}'/name;refs[str(p.relative_to(ROOT))]=sha(p)
    with (REF/'fold_01/epochs.csv').open() as s:expected=list(csv.DictReader(s))[:3]
    checks={};common=None;init_rng=None;groups=None;eval_ref=None
    for arm in ('N1',)+ARMS:
        conf=config_for(cfg,arm);SET_Random(0)
        model=make_model(dd,conf,graphs,'content',arm=arm).to(conf.Device)
        opt=relation_optimizer(model,conf,.0005)
        current_groups=optimizer_manifest(model,opt)
        if common is None:
            common={n:p.detach().clone() for n,p in model.named_parameters()}
            init_rng=torch.get_rng_state();groups=current_groups
        else:
            assert torch.equal(init_rng,torch.get_rng_state()) and groups==current_groups
            for n,p in model.named_parameters():torch.testing.assert_close(p,common[n],atol=0,rtol=0)
        model.eval()
        with torch.no_grad():z=model(data['Feature'])[0]
        if eval_ref is None:eval_ref=z.detach().clone()
        else:
            # GPU scatter reductions may differ at roundoff despite identical weights.
            torch.testing.assert_close(z,eval_ref,atol=2e-7,rtol=1e-6)
            torch.testing.assert_close(z.argmax(1),eval_ref.argmax(1),atol=0,rtol=0)
        assert sum(p.numel() for p in model.parameters())==726551
        noise=(model._modal_noise_std*model.noise_scale).detach().cpu().tolist()
        torch.testing.assert_close(torch.tensor(noise),torch.tensor([.15,.15,.15,SETTINGS[arm][0]]))
        criterion=MSCQGNLoss(dd,conf.Device,rate=conf.Loss_rate,aux_loss_weight=conf.aux_loss_weight)
        sched=CustomCosineAnnealingLR(opt,T_max=conf.T_max,eta_min=conf.Lr_Min)
        losses=[]
        for i in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            z,emb,aux=model(data['Feature']);loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
            assert torch.isfinite(loss)
            if arm=='N1':assert abs(float(loss.detach())-float(expected[i]['training_loss']))<2e-5
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),conf.grad_clip,error_if_nonfinite=True);opt.step();sched.step()
            losses.append(float(loss.detach()))
        checks[arm]=dict(noise_sd=dict(zip(dd['Modal_Name'],noise)),aux_weight=conf.aux_loss_weight,
            parameters=726551,first3_losses=losses,common_initialization_and_eval_verified=True,
            train_mask_smoke_passed=True,baseline_first3_reproduced=arm=='N1')
        del model,opt,criterion,z,emb,aux,loss
    del common,eval_ref;torch.cuda.empty_cache()
    OUT.mkdir(exist_ok=False);(OUT/'pilot').mkdir();(OUT/'full').mkdir()
    shutil.copy2(PREV/'data_splits_graphs.npz',OUT/'data_splits_graphs.npz')
    plan=dict(created_at=timestamp(),baseline='N1',settings=SETTINGS,arms=ARMS,screen_folds=FOLDS,
        seed=0,epochs=1000,protocol=protocol,checks=checks,
        selection='Both pilot folds strictly exceed N1, equal-fold mean gain >=0.01, and both last checkpoints predict both classes. Select highest mean; ties N4,N5,N6 order. At most one tenfold follow-up, reusing pilot folds1,9. No extra candidates or combined changes.',
        scope='Original C, same data, graph, splits, initialization, optimizer, scheduler and first maximum TEST ACC selection. N4/N5 change only Correlation noise; N6 only auxiliary weight relative to N1. Exploratory test-selected tuning, not independent validation.',
        code_sha256=sources,reference_sha256=refs,split_graph_archive_sha256=sha(OUT/'data_splits_graphs.npz'))
    for name in sources:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json',plan);return plan


def audit(folder,arm,fold,cfg,dd,data,graphs):
    model=make_model(dd,cfg,graphs,'content',arm=arm).to(cfg.Device).eval()
    noise=model._modal_noise_std.detach().clone();test=data['Mask'][fold-1][1];checks=[]
    for kind in ('best','last'):
        cp=torch.load(folder/f'{kind}.pt',map_location=cfg.Device,weights_only=False)
        assert (cp['arm'],cp['fold'],cp['seed'])==(arm,fold,0)
        model.load_state_dict(cp['state_dict']);torch.testing.assert_close(model._modal_noise_std,noise,atol=0,rtol=0)
        with torch.no_grad():z=model(data['Feature'])[0][test].cpu().numpy()
        with np.load(folder/f'{kind}_predictions.npz') as a:
            np.testing.assert_allclose(z,a['logits'],atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(z.argmax(1),a['logits'].argmax(1))
            np.testing.assert_array_equal(a['labels'],data['Label'][test].cpu().numpy())
            if kind=='best':np.testing.assert_array_equal(a['row_index'],np.array(dd['Index'])[test.cpu().numpy()])
        checks.append(dict(checkpoint=kind,sha256=sha(folder/f'{kind}.pt'),pred_counts=np.bincount(z.argmax(1),minlength=2).tolist(),passed=True))
    dump(folder/'adaptation_audit.json',dict(passed=True,checks=checks,settings=SETTINGS[arm]))
    return checks[-1]['pred_counts']


def report(pilot,selection=None,full=None):
    refs={f:read(REF/f'fold_{f:02d}/complete.json') for f in range(1,11)}
    lines=['# ABIDE C：N1后续调参','',
        '固定seed0、原第1和9折、每折1000轮；沿用首次最高测试ACC选轮。N1为本轮基线。','',
        'N4：Correlation噪声0.03→0.01；N5：0.03→0.06；N6：噪声保留0.03，仅辅助损失权重0.25→0.10。其余均不变。','',
        '| 配置 | 第1折最佳ACC | 第9折最佳ACC | 两折均值 | 末轮均值 |','|---|---:|---:|---:|---:|']
    for arm,rr in [('N1',[refs[f] for f in FOLDS])]+[(a,[r for r in pilot if r['arm']==a]) for a in ARMS]:
        if len(rr)==2:
            aa=[r['best']['acc']*100 for r in rr]
            lines.append(f"| {arm} | {aa[0]:.2f}% | {aa[1]:.2f}% | {np.mean(aa):.2f}% | {100*np.mean([r['last']['test_acc'] for r in rr]):.2f}% |")
    if selection is not None:lines+=['',f"按预定规则进入完整十折：{selection['selected'] or '无'}。"]
    if full:
        lines+=['',f"完整十折最佳ACC均值：{100*np.mean([r['best']['acc'] for r in full]):.4f}%；第1000轮均值：{100*np.mean([r['last']['test_acc'] for r in full]):.4f}%。筛选折直接复用。"]
    lines+=['','筛选要求：两折均严格超过N1、平均提高至少1个百分点，且两折末轮均预测两类。最多补齐一个候选；所有候选保留，不按折混选。沿用测试集选轮且重复调参，结果属于探索性证据。']
    (OUT/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze(pilot,selected,full,dd,data):
    rows=[];all_y=[];all_pred=[]
    stage='full' if full else 'pilot';records=full or pilot
    for r in records:
        arm=r['arm'];f=r['fold'];folder=OUT/stage/arm/f'fold_{f:02d}'
        with (folder/'epochs.csv').open() as s:curve=list(csv.DictReader(s))
        assert [int(a['epoch']) for a in curve]==list(range(1,1001))
        assert np.argmax([float(a['test_acc']) for a in curve])+1==r['best']['epoch']
        with np.load(folder/'best_predictions.npz') as a:z=a['logits'];y=a['labels'];rid=a['row_index']
        with np.load(REF/f'fold_{f:02d}/best_predictions.npz') as a:
            np.testing.assert_array_equal(rid,a['row_index']);np.testing.assert_array_equal(y,a['labels']);old=a['logits'].argmax(1)
        pred=z.argmax(1);prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        assert float((pred==y).mean())==r['best']['acc']
        rows.append(dict(arm=arm,fold=f,epoch=r['best']['epoch'],acc=float((pred==y).mean()),baseline_acc=float((old==y).mean()),
            last_acc=r['last']['test_acc'],macro_f1=float(f1_score(y,pred,average='macro')),
            balanced_acc=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob[:,1])),
            corrected=int(((old!=y)&(pred==y)).sum()),harmed=int(((old==y)&(pred!=y)).sum())))
        all_y.extend(y);all_pred.extend(pred)
    result=dict(stage=stage,rows=rows,selected=selected)
    if full:
        assert len(all_y)==len(dd['Index']) and sorted(r['fold'] for r in rows)==list(range(1,11))
        result['full_summary']={k:float(np.mean([r[k] for r in rows])) for k in ['acc','baseline_acc','last_acc','macro_f1','balanced_acc','auc']}
        result['full_summary'].update(corrected=sum(r['corrected'] for r in rows),harmed=sum(r['harmed'] for r in rows),
            wins=sum(r['acc']>r['baseline_acc'] for r in rows),other8_delta=float(np.mean([r['acc']-r['baseline_acc'] for r in rows if r['fold'] not in FOLDS])),
            confusion_matrix=confusion_matrix(all_y,all_pred).tolist())
    dump(OUT/'analysis.json',result)
    return result


def main():
    cfg,dd,data,graphs,protocol=inputs();plan=prepare(cfg,dd,data,graphs,protocol)
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    if (OUT/'COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as s:s.write(str(os.getpid()))
    try:
        runner.OUT=OUT/'pilot';pilot=[]
        for arm in ARMS:
            conf=config_for(cfg,arm);runner.RelationSheafModel=partial(make_model,arm=arm)
            for f in FOLDS:
                dump(OUT/'status.json',dict(state='pilot',arm=arm,fold=f,updated_at=timestamp()))
                folder=runner.OUT/arm/f'fold_{f:02d}';folder.mkdir(parents=True,exist_ok=True)
                dump(folder/'effective_settings.json',dict(arm=arm,Correlation_noise_sd=SETTINGS[arm][0],aux_loss_weight=conf.aux_loss_weight))
                r=runner.run(arm,f,conf,dd,data,graphs)
                r['last_pred_counts']=audit(folder,arm,f,conf,dd,data,graphs)
                pilot.append(r);dump(OUT/'pilot_results.json',pilot);report(pilot);torch.cuda.empty_cache()
        refs={f:read(REF/f'fold_{f:02d}/complete.json')['best']['acc'] for f in FOLDS}
        eligible=[];scores={}
        for arm in ARMS:
            rr=[r for r in pilot if r['arm']==arm];deltas=[r['best']['acc']-refs[r['fold']] for r in rr]
            scores[arm]=dict(mean_acc=float(np.mean([r['best']['acc'] for r in rr])),deltas=deltas)
            if all(d>0 for d in deltas) and np.mean(deltas)>=.01 and all(min(r['last_pred_counts'])>0 for r in rr):eligible.append(arm)
        selected=max(eligible,key=lambda a:scores[a]['mean_acc']) if eligible else None
        selection=dict(selected=selected,eligible=eligible,scores=scores);dump(OUT/'selection.json',selection);report(pilot,selection)
        full=[]
        if selected:
            conf=config_for(cfg,selected);runner.OUT=OUT/'full';runner.RelationSheafModel=partial(make_model,arm=selected)
            for f in FOLDS:
                src=OUT/'pilot'/selected/f'fold_{f:02d}';dst=OUT/'full'/selected/f'fold_{f:02d}'
                if not dst.exists():shutil.copytree(src,dst)
                else:
                    for p in src.iterdir():
                        if p.is_file():assert sha(p)==sha(dst/p.name)
            for f in range(1,11):
                dump(OUT/'status.json',dict(state='full',arm=selected,fold=f,updated_at=timestamp()))
                folder=runner.OUT/selected/f'fold_{f:02d}';folder.mkdir(parents=True,exist_ok=True)
                dump(folder/'effective_settings.json',dict(arm=selected,Correlation_noise_sd=SETTINGS[selected][0],aux_loss_weight=conf.aux_loss_weight))
                r=runner.run(selected,f,conf,dd,data,graphs);r['last_pred_counts']=audit(folder,selected,f,conf,dd,data,graphs)
                full.append(r);dump(OUT/'full_results.json',full);torch.cuda.empty_cache()
        report(pilot,selection,full);analysis=analyze(pilot,selected,full,dd,data)
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz')==plan['split_graph_archive_sha256']
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),pilot_runs=6,additional_runs=8 if selected else 0,
            selected=selected,full_tenfold_completed=bool(full),all_audits_passed=True))
        dump(OUT/'status.json',dict(state='complete',selected=selected,updated_at=timestamp()))
        print(str(selection),flush=True)
    except BaseException as e:
        dump(OUT/'status.json',dict(state='failed',error=repr(e),updated_at=timestamp()));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
