"""Bounded 2x2 capacity diagnosis; retain C, data, folds and 1000-epoch rule.

Three new capacity settings x two historical-cache folds. Existing W is the
fourth cell. This is exploratory diagnosis, not a new tenfold result.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from relation_experiment import ROOT, setup, build_protocol, dump, sha, timestamp, scores
from alternating_cooperative_experiment import read, verify_manifest
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Loss.loss_fn import orthogonality_loss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

OUT = ROOT/'Result/abide_capacity_diagnosis_20260927'
REF = ROOT/'Result/c_abide_repair_v2_20260927/full/W'
FOLDS = [1, 9]
ARMS = {'depth3': (96, 3), 'width64': (64, 5), 'small64_3': (64, 3)}


def inputs(arm):
    args = SimpleNamespace(config=str((OUT/'configs'/f'{arm}.ini').relative_to(ROOT)),
        output=str(OUT/arm), cuda=0, k=10, modes=['content'], folds=FOLDS,
        epochs=None, training_seed=0, diagnostics=False, graph_weight_decay=.0005)
    cfg, dd, data, graphs, paths = setup(0, args.config, 10)
    protocol = build_protocol(args, cfg, dd, data, graphs, paths)
    base = read(REF/'PROTOCOL.json')
    for key in ['candidate_hash', 'data_sha256', 'samples', 'class_names',
                'epochs', 'split_seed', 'training_seed', 'checkpoint_selection',
                'graph_weight_decay_override']:
        assert protocol[key] == base[key], key
    for key, value in base['original_config'].items():
        if key not in ['Hidden_size', 'num_layers']:
            assert protocol['original_config'][key] == value, key
    with np.load(REF/'splits_and_graphs.npz') as saved:
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(saved['test_masks'], torch.stack([m for _,m in data['Mask']]).cpu().numpy())
    return cfg, dd, data, graphs, protocol


def prepare():
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json'); verify_manifest(plan['code_sha256']); return plan
    base = read(REF/'PROTOCOL.json'); verify_manifest(base['code_sha256'])
    OUT.mkdir(exist_ok=False)
    (OUT/'configs').mkdir()
    config = (ROOT/'Config/c_abide_repair_v2_W.ini').read_text(encoding='utf-8')
    sources = dict(base['code_sha256']); sources[Path(__file__).name] = sha(__file__)
    protocols = {}
    for arm, (width, depth) in ARMS.items():
        path = OUT/'configs'/f'{arm}.ini'
        path.write_text(config.replace('Hidden_size = 96', f'Hidden_size = {width}').replace('num_layers = 5', f'num_layers = {depth}'), encoding='utf-8')
        sources[str(path.relative_to(ROOT))] = sha(path)
        *_, protocols[arm] = inputs(arm)
    ref_hashes = {}
    for fold in FOLDS:
        for name in ['best.pt', 'complete.json', 'epochs.csv', 'predictions.npz']:
            path = REF/f'content/fold_{fold:02d}'/name
            ref_hashes[str(path.relative_to(ROOT))] = sha(path)
    plan = dict(created_at=timestamp(), arms=ARMS, folds=FOLDS, epochs=1000,
        training_seed=0, graph_weight_decay=.0005, protocols=protocols,
        reference=str(REF), reference_sha256=ref_hashes, code_sha256=sources,
        scope='Capacity factorial only, unchanged C propagation. Fixed historical-cache folds1,9. No automatic full-tenfold follow-up or further tuning. Same test-selected rule; not heldout evidence. Width changes all Hidden_size-dependent layers; global_word_emb remains64. Different tensor shapes imply different random draws despite common seed.')
    for name in sources:
        target=OUT/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json', plan)
    return plan


def assess(model, data, fold, criterion):
    model.eval(); fit,test=data['Mask'][fold-1]; y=data['Label']
    with torch.no_grad():
        z,emb,aux = model(data['Feature'])
        details = dict(train_acc=float((z[fit].argmax(-1)==y[fit]).double().mean()),
            test_acc=float((z[test].argmax(-1)==y[test]).double().mean()),
            train_ce=float(criterion.CE_loss(z[fit],y[fit])),
            test_ce=float(criterion.CE_loss(z[test],y[test])),
            ortho=float(criterion.rate*orthogonality_loss(emb)),
            test_margin_std=float((z[test,1]-z[test,0]).std()),
            graph_step=float(model.propagation.step_logit.sigmoid()))
    return z,details


def replay_reference():
    cfg,dd,data,graphs,_=setup(0,'Config/c_abide_repair_v2_W.ini',10)
    SET_Random(0); model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    opt=relation_optimizer(model,cfg,.0005)
    sched=CustomCosineAnnealingLR(opt,T_max=cfg.T_max,eta_min=cfg.Lr_Min)
    criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    with (REF/'content/fold_01/epochs.csv').open() as s: expected=list(csv.DictReader(s))[:3]
    replay=[]
    for i in range(3):
        model.train(); opt.zero_grad(set_to_none=True)
        z,emb,aux=model(data['Feature']); loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True); opt.step()
        _,details=assess(model,data,1,criterion)
        err=abs(float(loss.detach())-float(expected[i]['train_loss']))
        assert err < 2e-5 and details['test_acc']==float(expected[i]['test_acc'])
        replay.append(dict(epoch=i+1,loss_absolute_difference=err,test_acc=details['test_acc']))
        sched.step()
    records=[]
    for fold in FOLDS:
        folder=REF/f'content/fold_{fold:02d}'
        cp=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
        model.load_state_dict(cp['state_dict']); z,details=assess(model,data,fold,criterion)
        with np.load(folder/'predictions.npz') as saved:
            np.testing.assert_allclose(z[data['Mask'][fold-1][1]].cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
        records.append(dict(fold=fold,best_epoch=cp['best']['epoch'],**details))
    dump(OUT/'reference_replay.json',dict(first3_training_replay=replay,best_records=records))


def run(arm,fold,cfg,dd,data,graphs):
    folder=OUT/arm/f'fold_{fold:02d}'
    if (folder/'complete.json').exists(): return read(folder/'complete.json')
    folder.mkdir(parents=True,exist_ok=True)
    SET_Random(0); model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    opt=relation_optimizer(model,cfg,.0005)
    dump(folder/'optimizer_groups.json',optimizer_manifest(model,opt))
    sched=CustomCosineAnnealingLR(opt,T_max=cfg.T_max,eta_min=cfg.Lr_Min)
    criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    fit,test=data['Mask'][fold-1]; y=data['Label']; best_count=-1; best=None; rows=[]
    if all((folder/name).exists() for name in ['epochs.csv','last.pt','best.pt','best_predictions.npz','last_predictions.npz']):
        with (folder/'epochs.csv').open() as stream:
            rows=[{k:(int(v) if k=='epoch' else float(v)) for k,v in r.items()} for r in csv.DictReader(stream)]
        assert [r['epoch'] for r in rows]==list(range(1,1001))
        cp=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
        return finalize(arm,fold,folder,model,data,criterion,rows,cp['best'],None)
    start=time.perf_counter()
    for epoch in range(1,1001):
        model.train(); opt.zero_grad(set_to_none=True)
        z,emb,aux=model(data['Feature']); loss=criterion(z,y,fit,emb,aux)
        assert torch.isfinite(loss)
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True); opt.step()
        z,detail=assess(model,data,fold,criterion)
        row=dict(epoch=epoch,training_loss=float(loss.detach()),gradient_norm=float(norm),lr=opt.param_groups[0]['lr'],**detail); rows.append(row)
        correct=int((z[test].argmax(-1)==y[test]).sum())
        if correct>best_count:
            best_count=correct; best=dict(epoch=epoch,**detail,**scores(z[test],y[test]))
            torch.save(dict(state_dict=model.state_dict(),fold=fold,arm=arm,seed=0,best=best),folder/'best.pt')
            np.savez_compressed(folder/'best_predictions.npz',logits=z[test].detach().cpu().numpy(),labels=y[test].cpu().numpy(),row_index=np.asarray(dd['Index'])[test.cpu().numpy()])
        sched.step()
        if epoch%100==0 or epoch==1:
            state=dict(state='running',arm=arm,fold=fold,epoch=epoch,best=best['acc'],train_acc=detail['train_acc'],test_acc=detail['test_acc'],updated_at=timestamp())
            dump(OUT/'status.json',state); print(json.dumps(state),flush=True)
    torch.save(dict(state_dict=model.state_dict(),fold=fold,arm=arm,seed=0,epoch=1000),folder/'last.pt')
    np.savez_compressed(folder/'last_predictions.npz',logits=z[test].detach().cpu().numpy(),labels=y[test].cpu().numpy())
    with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return finalize(arm,fold,folder,model,data,criterion,rows,best,time.perf_counter()-start)


def finalize(arm,fold,folder,model,data,criterion,rows,best,seconds):
    test=data['Mask'][fold-1][1]
    assert int(np.argmax([r['test_acc'] for r in rows]))+1==best['epoch']
    last_cp=torch.load(folder/'last.pt',map_location=data['Feature'].device,weights_only=False)
    assert (last_cp['fold'],last_cp['arm'],last_cp['seed'],last_cp['epoch'])==(fold,arm,0,1000)
    model.load_state_dict(last_cp['state_dict']); last_z,_=assess(model,data,fold,criterion)
    with np.load(folder/'last_predictions.npz') as saved:
        np.testing.assert_allclose(last_z[test].detach().cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(last_z[test].argmax(1).cpu().numpy(),saved['logits'].argmax(1))
    checkpoint=torch.load(folder/'best.pt',map_location=data['Feature'].device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); audit_z,audit_detail=assess(model,data,fold,criterion)
    with np.load(folder/'best_predictions.npz') as saved:
        np.testing.assert_allclose(audit_z[test].detach().cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(audit_z[test].argmax(1).cpu().numpy(),saved['logits'].argmax(1))
        np.testing.assert_array_equal(data['Label'][test].cpu().numpy(),saved['labels'])
    # Compare integer prediction counts, not two reductions differing by 1 ULP.
    correct=int((audit_z[test].argmax(1)==data['Label'][test]).sum())
    assert correct/int(test.sum())==best['acc']
    result=dict(arm=arm,fold=fold,parameters=sum(p.numel() for p in model.parameters()),best=best,last=rows[-1],
        late100_mean_acc=float(np.mean([r['test_acc'] for r in rows[-100:]])),seconds=seconds,audit_passed=True,
        recovered_completed_training=seconds is None)
    dump(folder/'complete.json',result); return result


def report(records):
    refs=read(OUT/'reference_replay.json')['best_records']
    lines=['# ABIDE C容量诊断','', '固定seed0、原十折中的第1和9折、每折1000轮；沿用首次最高测试ACC选轮。仅诊断，非十折新成绩。', '',
        '| 设置 | 第1折最佳ACC | 第9折最佳ACC | 两折均值 |', '|---|---:|---:|---:|']
    base=[r['test_acc'] for r in refs]
    lines.append(f'| W：96维5层 | {base[0]*100:.2f}% | {base[1]*100:.2f}% | {np.mean(base)*100:.2f}% |')
    for arm in ARMS:
        rr=[r for r in records if r['arm']==arm]
        if len(rr)==2:
            aa=[r['best']['acc'] for r in rr]
            lines.append(f'| {arm} | {aa[0]*100:.2f}% | {aa[1]*100:.2f}% | {np.mean(aa)*100:.2f}% |')
    lines += ['', '仅改变Hidden_size/num_layers，其他配置、数据、图、划分和训练代码保持一致。宽度影响所有依赖Hidden_size的层，自身分支word_emb固定64。相同seed但不同尺寸的参数不代表逐元素相同初始化，因此不能据单种子两折把变化归因于纯粹的参数量。',
        '', '没有重训原框架，没有替换C传播，没有自动启动完整十折。训练曲线新增评估模式下训练ACC、训练/测试CE、正交项与图步长，最佳及末轮权重均保存。']
    (OUT/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    plan=prepare(); verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
    if (OUT/'COMPLETE.json').exists(): return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        replay_reference(); records=[]
        for arm in ARMS:
            cfg,dd,data,graphs,protocol=inputs(arm)
            (OUT/arm).mkdir(parents=True,exist_ok=True)
            dump(OUT/arm/'PROTOCOL.json',protocol)
            for fold in FOLDS:
                records.append(run(arm,fold,cfg,dd,data,graphs)); dump(OUT/'results.json',records); report(records)
                torch.cuda.empty_cache()
        verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),runs=len(records),all_audits_passed=all(r['audit_passed'] for r in records)))
        dump(OUT/'status.json',dict(state='complete',runs=len(records),updated_at=timestamp()))
    except BaseException as exc:
        dump(OUT/'status.json',dict(state='failed',error=repr(exc),updated_at=timestamp())); raise
    finally: lock.unlink(missing_ok=True)


if __name__=='__main__': main()
