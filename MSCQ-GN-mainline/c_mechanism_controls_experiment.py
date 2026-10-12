"""Frozen C vs D1/D2: seed0, original ten folds, 1000 epochs; no tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import json
import subprocess
import sys
import time
import numpy as np
import torch
from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (base_setup, read, state_hash, rng_state,
    assert_rng_equal, metrics, verify_manifest, verify_reference, save_predictions)
from analyze_joint_evidence_routing import _record, _aggregate, _pair
from Model.relation_sheaf import RelationSheafModel
from Model.c_mechanism_controls import MechanismControlModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

OUT=ROOT/'Result/c_mechanism_controls_seed0_20260924'
REFERENCE=ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODES=('D1','D2')
FOLDS=tuple(range(1,11))
SOURCES=('Model/c_mechanism_controls.py','c_mechanism_controls_experiment.py',
         'tests/test_c_mechanism_controls.py','alternating_cooperative_experiment.py',
         'analyze_joint_evidence_routing.py')


def make_model(dd,cfg,graphs,mode):
    return MechanismControlModel(dd,cfg,graphs,mode,training_seed=0).to(cfg.Device)


def summarize(plan):
    folders={'C':REFERENCE,**{m:OUT/'seed_0'/m for m in MODES}}
    records={m:{f:_record(base/f'fold_{f:02d}',f,1000) for f in FOLDS
                  if (base/f'fold_{f:02d}/complete.json').exists()} for m,base in folders.items()}
    variants={m:_aggregate(v,list(FOLDS)) for m,v in records.items()}
    pairs=[_pair('C',m,records['C'],records[m],plan['class_names']) for m in MODES]
    complete=sum(len(records[m]) for m in MODES)
    result=dict(updated_at=timestamp(),completed_new_runs=complete,planned_new_runs=20,
        variants=variants,matched_comparisons=pairs,
        caveat='Seed0 only; historical first maximum TEST ACC; D2 is L^2 scalar diffusion, not vanilla GCN.')
    dump(OUT/'summary.json',result)
    lines=['# C机制对照：D1、D2','',f'完成 {complete}/20 次新训练；复用 C 原10折。',
        'seed0，原十折，每折1000轮，首次最高测试ACC选轮；部分折不代表完整结论。','',
        '| 模型 | 已完成折 | 最佳ACC | 宏F1 | 末轮ACC | 参数量 |','|---|---:|---:|---:|---:|---:|']
    for m,v in variants.items():
        if v['completed_folds']==10:
            lines.append(f"| {m} | 10/10 | {100*v['best']['acc']:.4f}% | {100*v['best']['macro_f1']:.4f}% | {100*v['last']['acc']:.4f}% | {plan['parameters'][m]} |")
        else:lines.append(f"| {m} | {v['completed_folds']}/10 | 待完成 | — | — | {plan['parameters'][m]} |")
    for p in pairs:
        if p['matched_folds']:
            b=p['stages']['best']
            lines.append(f"\n{p['candidate']} − C：配对{len(p['matched_folds'])}折，ACC {100*b['mean_acc_delta']:+.4f}个百分点；纠正{b['corrected']}，新增错误{b['harmed']}。")
    lines += ['','D1仅保留r=n，按唯一有效关系归一化；删除不用的60个关系边权参数。',
        'D2删除内容映射网络，使用恒等发送/接收映射的标量L²扩散；保留原边权、两次图传播和残差。',
        '这些实验验证机制的效用，不单独证明新颖性；D2不覆盖所有普通GNN。']
    (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def prepare():
    cfg,dd,data,graphs,hashes,original=base_setup()
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json')
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_manifest'])
        assert hashes==plan['data_sha256']
        assert sha(OUT/'data_splits_and_graphs.npz')==plan['splits_artifact_sha256']
        return cfg,dd,data,graphs,plan
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
                         '-p','test_c_mechanism_controls.py','-v'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    manifest,audit=verify_reference(cfg,dd,data,graphs)
    SET_Random(0)
    original_model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    original_state={k:v.clone() for k,v in original_model.state_dict().items()}
    original_rng=rng_state();initial={};checks=[]
    for mode in ('C',*MODES):
        SET_Random(0);model=make_model(dd,cfg,graphs,mode)
        assert_rng_equal(original_rng,rng_state())
        for name,value in model.state_dict().items():
            expected=original_state[name]
            if mode=='D1' and name in ('propagation.edge_similarity','propagation.edge_bias'):
                expected=expected.diagonal()
            torch.testing.assert_close(value,expected,atol=0,rtol=0)
        initial[mode]=dict(state_sha256=state_hash(model),parameters=sum(p.numel() for p in model.parameters()))
        if mode=='C':
            for training in (True,False):
                SET_Random(27);original_model.train(training)
                with torch.no_grad():a=original_model(data['Feature'])
                after=rng_state()
                SET_Random(27);model.train(training)
                with torch.no_grad():b=model(data['Feature'])
                # CUDA sparse accumulation can differ at roundoff scale across calls.
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6);assert_rng_equal(after,rng_state())
        else:
            optimizer=relation_optimizer(model,cfg,0.)
            loss_fn=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for step in range(8):
                model.train();optimizer.zero_grad(set_to_none=True)
                z,emb,aux=model(data['Feature'])
                loss=loss_fn(z,data['Label'],data['Mask'][0][0],emb,aux)
                assert torch.isfinite(loss);loss.backward()
                for name,p in model.named_parameters():
                    if p.grad is not None:assert torch.isfinite(p.grad).all(),name
                    if name.startswith('propagation.'):assert p.grad is not None,name
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
                optimizer.step();model.eval()
                with torch.no_grad():assert torch.isfinite(model(data['Feature'])[0]).all()
            torch.cuda.synchronize()
            checks.append(dict(mode=mode,seconds_per_epoch=(time.perf_counter()-start)/8,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,test_metrics_inspected=False))
            del optimizer,loss_fn,z,emb,aux,loss
        del model
    del original_model,original_state
    torch.cuda.empty_cache()
    sources=tuple(dict.fromkeys((*original['code_sha256'],*SOURCES)))
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    plan=dict(experiment='C mechanism controls D1 matched-only and D2 scalar diffusion',
        training_seed=0,split_seed=0,modes=list(MODES),folds=list(FOLDS),epochs=1000,planned_runs=20,
        reused_C_runs=10,parameters={m:v['parameters'] for m,v in initial.items()},initialization=initial,
        class_names=dd['Class_Names'],modalities=dd['Modal_Name'],samples=dd['Sample_Num'],
        original_config=original['original_config'],data_sha256=hashes,candidate_hash=original['candidate_hash'],
        splits_artifact_sha256=sha(OUT/'data_splits_and_graphs.npz'),
        checkpoint_selection=original['checkpoint_selection'],transductive=True,
        optimizer='Original Adam/cosine schedule; graph-only decay=0, remaining decay=0.0007',
        architecture={'D1':'r=n only, mean over one active relation, six compact groups; remove unused offdiagonal edge parameters',
                      'D2':'All r,n; remove learned content send/receive maps; scalar L^2 with D+I normalization, two passes, original residual sign'},
        scope='D1/D2 seed0 only; no D3, new dataset, tuning or additional seeds in this job.',
        limitation='D1 has fewer neighbors; D2 has fewer parameters and is not a vanilla GCN baseline. Historical test-selected exploratory estimates.',
        common_initialization_and_rng_verified=True,smoke=checks,reference_audit=audit,
        code_sha256={name:sha(ROOT/name) for name in sources},reference_manifest=manifest,
        environment=dict(executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target=OUT/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    dump(OUT/'PLAN.json',plan);summarize(plan)
    dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp(),planned_runs=20))
    return cfg,dd,data,graphs,plan


def train_arm(cfg,dd,data,graphs,plan,fold,mode):
    folder=OUT/f'seed_0/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists():return
    verify_manifest(plan['code_sha256']);folder.mkdir(parents=True,exist_ok=True)
    SET_Random(0);model=make_model(dd,cfg,graphs,mode)
    assert state_hash(model)==plan['initialization'][mode]['state_sha256']
    criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    optimizer=relation_optimizer(model,cfg,0.)
    dump(folder/'optimizer_groups.json',optimizer_manifest(model,optimizer))
    scheduler=CustomCosineAnnealingLR(optimizer,T_max=cfg.T_max,eta_min=cfg.Lr_Min)
    x,y=data['Feature'],data['Label'];fit,test=data['Mask'][fold-1]
    best_correct=-1;history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','test_loss','test_acc','lr','seconds']);writer.writeheader()
        for epoch in range(1,1001):
            model.train();optimizer.zero_grad(set_to_none=True)
            z,emb,aux=model(x);loss=criterion(z,y,fit,emb,aux)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step();model.eval()
            with torch.no_grad():
                z,emb,aux=model(x);test_loss=criterion(z,y,test,emb,aux)
                if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite logits')
            correct=int((z[test].argmax(-1)==y[test]).sum())
            row=dict(epoch=epoch,train_loss=float(loss.detach()),test_loss=float(test_loss),
                test_acc=correct/int(test.sum()),lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-started)
            writer.writerow(row);history.append(row['test_acc'])
            for stage in (['best'] if correct>best_correct else [])+(['last'] if epoch==1000 else []):
                score=dict(epoch=epoch,**metrics(z[test],y[test]))
                if stage=='best':best_correct=correct;best=score
                else:last=score
                temp=folder/f'{stage}.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,metrics=score,mode=mode,fold=fold,seed=0,
                    protocol_sha256=sha(OUT/'PLAN.json')),temp);temp.replace(folder/f'{stage}.pt')
                save_predictions(folder/('predictions.npz' if stage=='best' else 'last_predictions.npz'),z,y,test,dd['Index'])
                dump(folder/f'{stage}.json',score)
            scheduler.step()
            if epoch==1 or epoch%25==0:
                stream.flush();status=dict(state='running',updated_at=timestamp(),pid=os.getpid(),mode=mode,
                    fold=fold,seed=0,epochs=1000,planned_runs=20,best_acc=best['acc'],**row)
                dump(OUT/'status.json',status);print(json.dumps(status),flush=True)
    record=dict(mode=mode,fold=fold,seed=0,parameters=plan['parameters'][mode],best=best,last=last,
        initial_model_sha256=plan['initialization'][mode]['state_sha256'],mean_last100_acc=float(np.mean(history[-100:])),
        best_checkpoint_sha256=sha(folder/'best.pt'),last_checkpoint_sha256=sha(folder/'last.pt'),
        completed_at=timestamp(),seconds=time.perf_counter()-started,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json',record)
    del model,optimizer,criterion,z,emb,aux,loss
    torch.cuda.empty_cache();summarize(plan)


def audit_completed(cfg,dd,data,graphs,plan):
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_manifest']);rows=[]
    for mode in MODES:
        model=make_model(dd,cfg,graphs,mode).eval()
        for fold in FOLDS:
            folder=OUT/f'seed_0/{mode}/fold_{fold:02d}';record=read(folder/'complete.json')
            with (folder/'epochs.csv').open() as stream:epochs=list(csv.DictReader(stream))
            with (REFERENCE/f'fold_{fold:02d}/epochs.csv').open() as stream:old=list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs]==list(range(1,1001))
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1==record['best']['epoch']
            np.testing.assert_array_equal([float(r['lr']) for r in epochs],[float(r['lr']) for r in old])
            test=data['Mask'][fold-1][1]
            for stage,filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                assert sha(folder/f'{stage}.pt')==record[f'{stage}_checkpoint_sha256']
                cp=torch.load(folder/f'{stage}.pt',map_location=cfg.Device,weights_only=False)
                assert cp['protocol_sha256']==sha(OUT/'PLAN.json')
                assert (cp['mode'],cp['fold'],cp['seed'])==(mode,fold,0)
                model.load_state_dict(cp['state_dict'])
                with torch.no_grad():logits=model(data['Feature'])[0][test]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'],logits.cpu().numpy(),atol=2e-5,rtol=2e-5)
                    np.testing.assert_array_equal(saved['logits'].argmax(-1),logits.argmax(-1).cpu().numpy())
                actual=metrics(logits,data['Label'][test])
                for key in ('acc','auc','f1','macro_f1'):assert abs(actual[key]-record[stage][key])<1e-12
                rows.append(dict(mode=mode,fold=fold,stage=stage,passed=True))
        del model
    result=dict(passed=True,verified_runs=20,verified_checkpoints=len(rows),records=rows)
    dump(OUT/'audit.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    cfg,dd,data,graphs,plan=prepare()
    if args.prepare:print(json.dumps(dict(state='prepared',parameters=plan['parameters'],smoke=plan['smoke'])),flush=True);return
    if (OUT/'COMPLETE.json').exists():print('Already complete.');return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES:train_arm(cfg,dd,data,graphs,plan,fold,mode)
        audit=audit_completed(cfg,dd,data,graphs,plan);summary=summarize(plan)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary))
        dump(OUT/'status.json',dict(state='complete',completed_runs=20,updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
