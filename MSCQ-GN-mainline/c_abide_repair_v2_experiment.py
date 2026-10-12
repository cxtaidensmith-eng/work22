"""Bounded ABIDE regularization screen and one conditional tenfold follow-up.

Exploratory selection retains the user's historical test-ACC protocol, so these
results are not independent confirmation. No open-ended search or goal at90%.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch
from relation_experiment import ROOT,setup,train,build_protocol,dump,sha,timestamp,scores
from alternating_cooperative_experiment import read,verify_manifest
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Utils import SET_Random
from Utils.relation_optimizer import relation_optimizer
import c_abide_repair_experiment as audit_helper

OUT=ROOT/'Result/c_abide_repair_v2_20260927'
REF=ROOT/'Result/c_abide_repair_v1_seed0_20260927'
MODES=('G','W','A')
PILOT_FOLDS=[1,4,6]


def args_for(mode,full=False):
    return SimpleNamespace(config=f'Config/c_abide_repair_v2_{mode}.ini',
        output=str(OUT/('full' if full else 'pilot')/mode),cuda=0,k=10,modes=['content'],
        folds=list(range(1,11)) if full else PILOT_FOLDS,epochs=None if full else 300,
        training_seed=0,diagnostics=False,graph_weight_decay=.0005,resume=True)


def check_protocol(args):
    cfg,dd,data,graphs,paths=setup(0,args.config,10)
    base=read(REF/'PROTOCOL.json')
    actual=build_protocol(args,cfg,dd,data,graphs,paths)
    for key in ('data_sha256','candidate_hash','samples','class_names','split_seed','training_seed'):
        assert base[key]==actual[key],key
    with np.load(REF/'splits_and_graphs.npz') as saved:
        np.testing.assert_array_equal(saved['source_row'],dd['Index'])
        np.testing.assert_array_equal(saved['labels'],data['Label'].cpu().numpy())
        np.testing.assert_array_equal(saved['test_masks'],torch.stack([t for _,t in data['Mask']]).cpu().numpy())
    assert cfg.epochs==1000 and cfg.Hidden_size==96 and cfg.num_layers==5
    return cfg,dd,data,graphs,actual


def prepare():
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_manifest']);return plan
    reference=read(REF/'PLAN.json');verify_manifest(reference['code_sha256'])
    sources=dict(reference['code_sha256']);sources[os.path.basename(__file__)]=sha(__file__)
    sources['c_abide_repair_experiment.py']=sha(ROOT/'c_abide_repair_experiment.py')
    protocols={};checks=[]
    for mode in MODES:
        args=args_for(mode);cfg,dd,data,graphs,protocol=check_protocol(args);protocols[mode]=protocol
        sources[args.config]=sha(ROOT/args.config)
        SET_Random(0);model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
        assert sum(p.numel() for p in model.parameters())==726551
        optimizer=relation_optimizer(model,cfg,.0005)
        criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
        for _ in range(3):
            model.train();optimizer.zero_grad(set_to_none=True)
            z,emb,aux=model(data['Feature']);loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
            assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True);optimizer.step()
        checks.append(dict(mode=mode,passed=True,test_metrics_inspected=False))
        del model,optimizer,criterion,z,emb,aux,loss,cfg,dd,data,graphs
        torch.cuda.empty_cache()
    manifest={}
    for fold in PILOT_FOLDS:
        for file in ('complete.json','epochs.csv','predictions.npz','last_predictions.npz'):
            path=REF/f'content/fold_{fold:02d}'/file;manifest[str(path.relative_to(ROOT))]=sha(path)
    plan=dict(created_at=timestamp(),screen_modes=list(MODES),screen_folds=PILOT_FOLDS,screen_epochs=300,
        parameters=726551,protocols=protocols,checks=checks,
        definitions=dict(G='v1 + graph weight decay0.0005',W='G + patient-path weight decay0.002',
                         A='graph decay0.0005, patient decay0.001, dropout0.60, input noise0.10, drop_path0.10'),
        selection='Highest equal-fold mean best TEST ACC on fixed folds1,4,6; ties resolved G,W,A order. Follow up only if improvement over v1 in the SAME first300 epochs is at least0.005 (0.5 percentage point) and all three last checkpoints predict both classes. Only one full tenfold follow-up, fixed seed0, 1000 epochs. No automatic retuning.',
        interpretation='Screen uses previously inspected test folds; full tenfold is exploratory, not an independent confirmation. Does not change split or historical checkpoint rule. Preserve all candidate results; do not cherry-pick folds or combine candidates.',
        code_sha256=sources,reference_manifest=manifest)
    OUT.mkdir(parents=True,exist_ok=False)
    for name in sources:
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json',plan);return plan


def audit_run(mode,full=False):
    args=args_for(mode,full);cfg,dd,data,graphs,_=check_protocol(args);out=ROOT/args.output
    SET_Random(0);model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device).eval()
    records=[]
    for fold in args.folds:
        folder=out/f'content/fold_{fold:02d}';record=read(folder/'complete.json')
        with (folder/'epochs.csv').open() as stream:curve=list(csv.DictReader(stream))
        count=1000 if full else 300
        assert [int(r['epoch']) for r in curve]==list(range(1,count+1))
        assert int(np.argmax([float(r['test_acc']) for r in curve]))+1==record['best']['epoch']
        cp=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
        assert (cp['fold'],cp['mode'],cp['seed'])==(fold,'content',0)
        model.load_state_dict(cp['state_dict']);test=data['Mask'][fold-1][1]
        with torch.no_grad():z=model(data['Feature'])[0][test]
        with np.load(folder/'predictions.npz') as saved:
            np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
            np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
            np.testing.assert_allclose(saved['logits'],z.cpu().numpy(),atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(saved['logits'].argmax(1),z.argmax(1).cpu().numpy())
        actual=scores(z,data['Label'][test]);assert actual['acc']==record['best']['acc']
        with np.load(folder/'last_predictions.npz') as last:
            pred=last['logits'].argmax(1);counts=np.bincount(pred,minlength=2).tolist()
            assert np.isfinite(last['logits']).all()
            assert float((pred==last['labels']).mean())==record['last']['acc']
        records.append(dict(fold=fold,best=record['best'],last=record['last'],last_pred_counts=counts))
    result=dict(passed=True,verified_best_checkpoints=len(records),records=records,
        mean_best_acc=float(np.mean([r['best']['acc'] for r in records])),
        mean_last_acc=float(np.mean([r['last']['acc'] for r in records])),
        all_last_predict_both=all(min(r['last_pred_counts'])>0 for r in records))
    dump(out/'audit.json',result)
    del model;torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');a=parser.parse_args()
    plan=prepare()
    if a.prepare:print(json.dumps(dict(state='prepared',checks=plan['checks'])));return
    if (OUT/'COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        results={}
        for mode in MODES:
            verify_manifest(plan['code_sha256'])
            dump(OUT/'status.json',dict(state='screening',mode=mode,pid=os.getpid(),updated_at=timestamp()))
            train(args_for(mode));results[mode]=audit_run(mode)
            dump(OUT/'screen_results.json',results)
        baseline=[]
        for fold in PILOT_FOLDS:
            with (REF/f'content/fold_{fold:02d}/epochs.csv').open() as stream:curve=list(csv.DictReader(stream))[:300]
            baseline.append(max(float(r['test_acc']) for r in curve))
        reference=float(np.mean(baseline))
        eligible=[m for m in MODES if results[m]['all_last_predict_both'] and results[m]['mean_best_acc']>=reference+.005]
        selected=max(eligible,key=lambda m:results[m]['mean_best_acc']) if eligible else None
        selection=dict(reference_first300_mean_best_acc=reference,selected=selected,candidates=results)
        dump(OUT/'selection.json',selection)
        lines=['# ABIDE C第二轮训练适配','', '固定seed0，三折300轮筛选；沿用最高测试ACC选轮，结果属于探索性调参。','',
               '| 版本 | 三折最佳ACC | 三折第300轮ACC | 三折均预测两类 |','|---|---:|---:|---|',
               f'| v1前300轮 | {100*reference:.4f}% | — | — |']
        for m,r in results.items():lines.append(f"| {m} | {100*r['mean_best_acc']:.4f}% | {100*r['mean_last_acc']:.4f}% | {r['all_last_predict_both']} |")
        if selected:
            lines.append(f'\n按预定规则进入完整十折：{selected}。')
            (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            dump(OUT/'status.json',dict(state='full_training',mode=selected,pid=os.getpid(),updated_at=timestamp()))
            train(args_for(selected,True));audit_run(selected,True)
            cfg,dd,data,graphs,_=check_protocol(args_for(selected,True))
            # Reuse binary metrics/report audit with this run's frozen provenance.
            audit_helper.OUT=OUT/'full'/selected;audit_helper.DATASET='ABIDE'
            run_plan=dict(code_sha256=plan['code_sha256'],samples=dd['Sample_Num'],
                class_counts=torch.bincount(data['Label'],minlength=2).cpu().tolist())
            audit_helper.audit_and_report(cfg,dd,data,graphs,run_plan)
            summary=read(OUT/'full'/selected/'summary.json')['variants']['content']
            lines.append(f"\n完整十折最佳ACC：{100*summary['mean_acc']:.4f}%；第1000轮：{100*summary['mean_final_epoch_acc']:.4f}%。")
            lines.append('完整十折不保证优于v1；请保留所有结果并报告与v1的比较。')
        else:lines.append('\n没有候选满足预定改进和非退化条件；本轮停止，不自动追加试验。')
        (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_manifest'])
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),selection=selection))
        dump(OUT/'status.json',dict(state='complete',selected=selected,updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),updated_at=timestamp()));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
