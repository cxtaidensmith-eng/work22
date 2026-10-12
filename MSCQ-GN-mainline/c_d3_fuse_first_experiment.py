"""Frozen C vs D3 fuse-first: seed0, original ten folds, 1000 epochs; no tuning."""
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
from Model.c_fuse_first import FuseFirstModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

OUT=ROOT/'Result/c_d3_fuse_first_seed0_20260925'
REFERENCE=ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODES=('D3',)
FOLDS=tuple(range(1,11))
SOURCES=('Model/c_fuse_first.py','c_d3_fuse_first_experiment.py',
         'tests/test_c_fuse_first.py','tests/test_relation_sheaf.py',
         'c_mechanism_controls_experiment.py','alternating_cooperative_experiment.py',
         'analyze_joint_evidence_routing.py')


def make_model(dd,cfg,graphs,mode):
    assert mode == 'D3'
    return FuseFirstModel(dd,cfg,graphs,training_seed=0).to(cfg.Device)


def summarize(plan):
    folders={'C':REFERENCE,'D3':OUT/'seed_0/D3'}
    records={m:{f:_record(base/f'fold_{f:02d}',f,1000) for f in FOLDS
                  if (base/f'fold_{f:02d}/complete.json').exists()} for m,base in folders.items()}
    variants={m:_aggregate(v,list(FOLDS)) for m,v in records.items()}
    pair=_pair('C','D3',records['C'],records['D3'],plan['class_names'])
    complete=len(records['D3'])
    result=dict(updated_at=timestamp(),completed_new_runs=complete,planned_new_runs=10,
        variants=variants,matched_comparisons=[pair],
        caveat='Seed0 historical TEST-selected results. Placement control includes moving graph after nonlinear readout and auxiliary losses; not proof of modal identity alone.')
    dump(OUT/'summary.json',result)
    lines=['# D3：先融合再传播','',f'新训练完成 {complete}/10；复用 C 原10折。',
        '原十折，seed0，每折1000轮，首次最高测试ACC选轮。','',
        '| 模型 | 完成折数 | 最佳ACC | 宏F1 | 末轮ACC | 参数量 |','|---|---:|---:|---:|---:|---:|']
    for m,v in variants.items():
        if v['completed_folds']==10:
            lines.append(f"| {m} | 10/10 | {100*v['best']['acc']:.4f}% | {100*v['best']['macro_f1']:.4f}% | {100*v['last']['acc']:.4f}% | {plan['parameters'][m]} |")
        else:lines.append(f"| {m} | {v['completed_folds']}/10 | 待完成 | — | — | {plan['parameters'][m]} |")
    if pair['matched_folds']:
        b=pair['stages']['best']
        lines.append(f"\nD3−C：配对{len(pair['matched_folds'])}折，ACC {100*b['mean_acc_delta']:+.4f}个百分点；纠正{b['corrected']}，新增错误{b['harmed']}。")
    lines += ['', 'D3保留全部895096个参数：先Transformer及类别读出/Message_MLP得到患者向量，再广播到六个原传播通道，传播后取平均。',
        '六通道是参数匹配的滤波器组，收到相同患者向量，不再携带六种不同模态内容；每通道仍使用全部六张候选图。',
        '自身分支与相加分类保持原样；辅助损失位于图传播之前，主分类损失仍端到端训练全部路径。',
        '此对照检验完整的早期模态状态传播与后期融合患者传播，不能单独分离读出位置、辅助监督位置和归一化位置的影响。']
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
                         '-p','test_c_fuse_first.py','-v'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    manifest,audit=verify_reference(cfg,dd,data,graphs)
    SET_Random(0)
    c=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    initial_rng=rng_state()
    SET_Random(0);model=make_model(dd,cfg,graphs,'D3')
    assert_rng_equal(initial_rng,rng_state())
    assert state_hash(c)==state_hash(model)
    for name,value in c.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
    for name in ('edge','similarity','edge_group'):
        torch.testing.assert_close(getattr(c.propagation,name),getattr(model.propagation,name),atol=0,rtol=0)
    initial={m:dict(state_sha256=state_hash(model),parameters=sum(p.numel() for p in model.parameters())) for m in ('C','D3')}
    assert initial['D3']['parameters']==895096
    # Reordering deterministic graph operations preserves the stochastic call sequence.
    for training in (False,True):
        SET_Random(47);c.train(training)
        with torch.no_grad():c(data['Feature'])
        after=rng_state()
        SET_Random(47);model.train(training)
        with torch.no_grad():model(data['Feature'])
        assert_rng_equal(after,rng_state())
    old_groups=read(REFERENCE/'fold_01/optimizer_groups.json')
    optimizer=relation_optimizer(model,cfg,0.)
    assert optimizer_manifest(model,optimizer)==old_groups
    loss_fn=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();gradient_max={}
    for step in range(8):
        model.train();optimizer.zero_grad(set_to_none=True)
        z,emb,aux=model(data['Feature'])
        loss=loss_fn(z,data['Label'],data['Mask'][0][0],emb,aux)
        assert torch.isfinite(loss);loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
            if name.startswith('propagation.'):
                assert p.grad is not None,name
                gradient_max[name]=max(gradient_max.get(name,0.),float(p.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
        optimizer.step();model.eval()
        with torch.no_grad():assert torch.isfinite(model(data['Feature'])[0]).all()
    assert all(v>0 for v in gradient_max.values()),gradient_max
    torch.cuda.synchronize()
    checks=dict(seconds_per_epoch=(time.perf_counter()-start)/8,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        graph_gradient_maxima=gradient_max,test_metrics_inspected=False,
        common_initialization=True,optimizer_groups_identical=True,stochastic_streams_identical=True)
    del c,model,optimizer,loss_fn,z,emb,aux,loss
    torch.cuda.empty_cache()
    sources=tuple(dict.fromkeys((*original['code_sha256'],*SOURCES)))
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    plan=dict(experiment='D3 patient fusion before graph propagation',
        training_seed=0,split_seed=0,modes=list(MODES),folds=list(FOLDS),epochs=1000,planned_runs=10,
        reused_C_runs=10,parameters={m:v['parameters'] for m,v in initial.items()},initialization=initial,
        class_names=dd['Class_Names'],modalities=dd['Modal_Name'],samples=dd['Sample_Num'],
        original_config=original['original_config'],data_sha256=hashes,candidate_hash=original['candidate_hash'],
        splits_artifact_sha256=sha(OUT/'data_splits_and_graphs.npz'),checkpoint_selection=original['checkpoint_selection'],transductive=True,
        optimizer='Original Adam/cosine; identical parameter groups: graph-only decay=0, remainder=0.0007',
        architecture='Original modality encoder -> five Transformer blocks -> class queries/Message_MLP -> fused N x 96 -> broadcast to six filter channels -> original cooperative graph propagation -> mean channels -> add own branch -> original classifier',
        matched='All parameters, initialization, graph edges, two propagation passes, training objective and random streams retained. No new modules or tuning.',
        limitation='Graph now follows nonlinear readout; auxiliary/orthogonality losses are pre-graph. LayerNorm placement and gradient paths change. Channels are filter banks receiving identical fused content, not different modalities. This tests early versus late graph placement, not modal identity in isolation.',
        scope='D3 seed0 ten folds only; no additional seeds, datasets or search.',smoke=checks,reference_audit=audit,
        code_sha256={name:sha(ROOT/name) for name in sources},reference_manifest=manifest,
        environment=dict(executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target=OUT/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    dump(OUT/'PLAN.json',plan);summarize(plan)
    dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp(),planned_runs=10))
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
                    fold=fold,seed=0,epochs=1000,planned_runs=10,best_acc=best['acc'],**row)
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
    result=dict(passed=True,verified_runs=10,verified_checkpoints=len(rows),records=rows)
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
        dump(OUT/'status.json',dict(state='complete',completed_runs=10,updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
