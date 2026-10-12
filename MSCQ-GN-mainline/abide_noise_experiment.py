"""Noise-only 2x2 ABIDE screen; one predeclared conditional tenfold extension."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv
import json
import shutil
from pathlib import Path
from functools import partial
import numpy as np
import torch
import diagnose_abide_capacity as runner
from relation_experiment import ROOT,setup,dump,sha,timestamp
from alternating_cooperative_experiment import read,verify_manifest
from abide_noise_adaptation import ARMS,make_model
from abide_late_readout_experiment import inputs
from Model.relation_sheaf import RelationSheafModel
from Utils import SET_Random
from Utils.relation_optimizer import relation_optimizer,optimizer_manifest
from Loss import MSCQGNLoss

OUT=ROOT/'Result/abide_noise_adaptation_20260927'
REF=ROOT/'Result/c_abide_repair_v2_20260927/full/W'
FOLDS=[1,9]


def prepare(cfg,dd,data,graphs,protocol):
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);return plan
    base=read(REF/'PROTOCOL.json');verify_manifest(base['code_sha256'])
    sources=dict(base['code_sha256'])
    for name in [Path(__file__).name,'abide_noise_adaptation.py','diagnose_abide_capacity.py',
                 'abide_late_readout_experiment.py','Model/late_patient_readout.py','tests/test_abide_noise_adaptation.py']:
        sources[name]=sha(ROOT/name)
    refs={}
    for fold in range(1,11):
        for name in ['best.pt','complete.json','epochs.csv','predictions.npz','last_predictions.npz']:
            path=REF/f'content/fold_{fold:02d}'/name;refs[str(path.relative_to(ROOT))]=sha(path)
    records={};common=None;rng=None;opt_groups=None
    for arm in ('W',)+ARMS:
        SET_Random(0);model=make_model(dd,cfg,graphs,'content',training_seed=0,arm=arm).to(cfg.Device)
        current_rng=torch.get_rng_state()
        opt=relation_optimizer(model,cfg,.0005)
        groups=optimizer_manifest(model,opt)
        if common is None:
            common={n:p.detach().clone() for n,p in model.named_parameters()};rng=current_rng;opt_groups=groups
        else:
            assert torch.equal(rng,current_rng) and groups==opt_groups
            for n,p in model.named_parameters():torch.testing.assert_close(p,common[n],atol=0,rtol=0)
        assert sum(p.numel() for p in model.parameters())==726551
        std=(model._modal_noise_std*model.noise_scale).detach().cpu().tolist()
        criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
        for _ in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            z,emb,aux=model(data['Feature']);loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
            assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True);opt.step()
        records[arm]=dict(effective_noise_std=dict(zip(dd['Modal_Name'],std)),parameters=726551,
                          three_step_train_mask_smoke_passed=True,test_metrics_inspected=False)
        del model,opt,criterion,z,emb,aux,loss
    del common;torch.cuda.empty_cache()
    OUT.mkdir(exist_ok=False);(OUT/'pilot').mkdir();(OUT/'full').mkdir()
    protocol=dict(protocol);protocol['output_scope']='noise-only C experiment, not late-readout experiment'
    plan=dict(created_at=timestamp(),protocol=protocol,arms=ARMS,screen_folds=FOLDS,seed=0,epochs=1000,
        definitions=dict(W='PHENO/ANAT/FUNC/Correlation noise SD .15 each',
            N1='Correlation SD .03 only; other groups .15',N2='PHENO SD0 only; other groups .15',
            N3='PHENO SD0 and Correlation SD .03; ANAT/FUNC .15'),
        checks=records,network='Original C/W unchanged. Override persistent _modal_noise_std buffer only; full randn_like draw retained for zero-noise columns.',
        optimizer='Same W Adam settings: patient weight decay .002, early graph .0005; lr .001, cosine1000, floor .0001; auxiliary .25; orthogonality1e-6.',
        selection='Eligible only if both pilot folds have strictly higher best TEST ACC than W, two-fold mean gain >=0.01 (1 percentage point), and both last checkpoints predict both classes. Select highest two-fold mean; ties follow N1,N2,N3. At most one full tenfold extension. Reuse identical completed pilot folds1,9 and train remaining8 folds with same configuration/seed/1000 epochs. If none eligible, stop. No automatic new noise levels or auxiliary-loss experiments.',
        scope='Original folds and first strict maximum TEST ACC selection retained. Exploratory tuning, not independent validation or guarantee of90%. Never select different arms per fold.',
        code_sha256=sources,reference_sha256=refs)
    for name in sources:
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    np.savez_compressed(OUT/'data_splits_graphs.npz',source_row=np.array(dd['Index']),labels=data['Label'].cpu().numpy(),
        test_masks=torch.stack([m for _,m in data['Mask']]).cpu().numpy(),
        **{f'edge_{i}':g['edge'].numpy() for i,g in enumerate(graphs)})
    plan['split_graph_archive_sha256']=sha(OUT/'data_splits_graphs.npz')
    dump(OUT/'PLAN.json',plan);return plan


def audit_extra(folder,arm,fold,cfg,dd,data,graphs):
    """The reused runner already checks logits; additionally verify noise and rows."""
    model=make_model(dd,cfg,graphs,'content',training_seed=0,arm=arm).to(cfg.Device).eval()
    expected=model._modal_noise_std.detach().clone();fit,test=data['Mask'][fold-1]
    checks=[]
    for kind in ('best','last'):
        cp=torch.load(folder/f'{kind}.pt',map_location=cfg.Device,weights_only=False)
        assert (cp['fold'],cp['arm'],cp['seed'])==(fold,arm,0)
        model.load_state_dict(cp['state_dict']);torch.testing.assert_close(model._modal_noise_std,expected,atol=0,rtol=0)
        with torch.no_grad():z=model(data['Feature'])[0][test]
        with np.load(folder/f'{kind}_predictions.npz') as saved:
            np.testing.assert_allclose(z.cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(z.argmax(-1).cpu().numpy(),saved['logits'].argmax(-1))
            np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
            if kind=='best':np.testing.assert_array_equal(saved['row_index'],np.array(dd['Index'])[test.cpu().numpy()])
            counts=np.bincount(saved['logits'].argmax(-1),minlength=2).tolist()
        checks.append(dict(checkpoint=kind,passed=True,pred_counts=counts,sha256=sha(folder/f'{kind}.pt')))
    dump(folder/'noise_and_prediction_audit.json',dict(checks=checks,passed=True,
        effective_noise_std=(model._modal_noise_std*model.noise_scale).cpu().tolist()))
    return checks[-1]['pred_counts']


def summary(pilot,selection=None,full=None):
    base=[read(REF/f'content/fold_{f:02d}/complete.json') for f in FOLDS]
    lines=['# ABIDE C输入噪声适配','',
        '固定seed0、原第1和9折、每折1000轮；只改变指定模态的输入噪声，保留C/W网络与训练设置。沿用最高测试ACC选轮，属于探索性调参。','',
        '| 版本 | 第1折最佳ACC | 第9折最佳ACC | 两折均值 |','|---|---:|---:|---:|',
        f"| W | {base[0]['best']['acc']*100:.2f}% | {base[1]['best']['acc']*100:.2f}% | {np.mean([r['best']['acc'] for r in base])*100:.2f}% |"]
    for arm in ARMS:
        rs=[r for r in pilot if r['arm']==arm]
        if len(rs)==2:
            aa=[r['best']['acc']*100 for r in rs]
            lines.append(f'| {arm} | {aa[0]:.2f}% | {aa[1]:.2f}% | {np.mean(aa):.2f}% |')
    lines+=['','N1：仅Correlation噪声0.15→0.03；N2：仅PHENO噪声0.15→0；N3：合并两项。ANAT/FUNC仍为0.15。所有版本参数量726551、相同初始化、相同随机数使用顺序。',
        '', '仅两折都严格提高、平均提高至少1个百分点且末轮均预测两类者可进入十折；最多推进两折均值最高的一个候选。没有候选满足则结束。']
    if selection is not None:
        lines.append(f"\n按预定规则进入完整十折的候选：{selection['selected'] or '无'}。")
    if full:
        lines.append(f"\n完整十折：{selection['selected']}，最佳ACC均值{100*np.mean([r['best']['acc'] for r in full]):.4f}%，固定第1000轮均值{100*np.mean([r['last']['test_acc'] for r in full]):.4f}%。第1、9折直接复用同配置试验，不重新抽选。")
    (OUT/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cfg,dd,data,graphs,protocol=inputs();plan=prepare(cfg,dd,data,graphs,protocol)
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    if (OUT/'COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    runner.OUT=OUT/'pilot'
    try:
        runner.replay_reference();pilot=[]
        for arm in ARMS:
            runner.RelationSheafModel=partial(make_model,arm=arm)
            for fold in FOLDS:
                dump(OUT/'status.json',dict(state='pilot',arm=arm,fold=fold,updated_at=timestamp()))
                r=runner.run(arm,fold,cfg,dd,data,graphs)
                r['last_pred_counts']=audit_extra(runner.OUT/arm/f'fold_{fold:02d}',arm,fold,cfg,dd,data,graphs)
                pilot.append(r);dump(OUT/'pilot_results.json',pilot);summary(pilot);torch.cuda.empty_cache()
        refs={f:read(REF/f'content/fold_{f:02d}/complete.json')['best']['acc'] for f in FOLDS}
        scores={};eligible=[]
        for arm in ARMS:
            rr=[r for r in pilot if r['arm']==arm]
            delta=[r['best']['acc']-refs[r['fold']] for r in rr]
            scores[arm]=dict(mean_best_acc=float(np.mean([r['best']['acc'] for r in rr])),deltas=delta)
            if all(d>0 for d in delta) and np.mean(delta)>=.01 and all(min(r['last_pred_counts'])>0 for r in rr):eligible.append(arm)
        selected=max(eligible,key=lambda arm:scores[arm]['mean_best_acc']) if eligible else None
        selection=dict(selected=selected,eligible=eligible,scores=scores)
        dump(OUT/'selection.json',selection);summary(pilot,selection)
        full=[]
        if selected:
            runner.OUT=OUT/'full';runner.RelationSheafModel=partial(make_model,arm=selected)
            for fold in FOLDS:
                src=OUT/'pilot'/selected/f'fold_{fold:02d}';dst=OUT/'full'/selected/f'fold_{fold:02d}'
                if not dst.exists():shutil.copytree(src,dst)
                else:
                    for file in src.iterdir():
                        if file.is_file():assert sha(file)==sha(dst/file.name)
            for fold in range(1,11):
                dump(OUT/'status.json',dict(state='full',arm=selected,fold=fold,updated_at=timestamp()))
                r=runner.run(selected,fold,cfg,dd,data,graphs)
                r['last_pred_counts']=audit_extra(runner.OUT/selected/f'fold_{fold:02d}',selected,fold,cfg,dd,data,graphs)
                full.append(r);dump(OUT/'full_results.json',full);torch.cuda.empty_cache()
            summary(pilot,selection,full)
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz')==plan['split_graph_archive_sha256']
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),pilot_runs=6,selected=selected,
            additional_runs=8 if selected else 0,full_tenfold_completed=bool(selected),all_audits_passed=True))
        dump(OUT/'status.json',dict(state='complete',selected=selected,updated_at=timestamp()))
    except BaseException as exc:
        dump(OUT/'status.json',dict(state='failed',error=repr(exc),updated_at=timestamp()));raise
    finally:
        runner.RelationSheafModel=RelationSheafModel;lock.unlink(missing_ok=True)


if __name__=='__main__':main()
