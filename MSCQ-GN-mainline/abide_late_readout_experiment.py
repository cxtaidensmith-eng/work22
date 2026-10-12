"""Four bounded runs: late self vs late population, folds1/9, seed0, 1000 epochs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import diagnose_abide_capacity as runner
from relation_experiment import ROOT,setup,build_protocol,dump,sha,timestamp
from alternating_cooperative_experiment import read,verify_manifest
from Model.late_patient_readout import LateReadoutModel
from Model.relation_sheaf import RelationSheafModel
from Utils import SET_Random
from Utils.relation_optimizer import relation_optimizer,optimizer_manifest
from Loss import MSCQGNLoss

OUT=ROOT/'Result/abide_late_readout_20260927'
REF=ROOT/'Result/c_abide_repair_v2_20260927/full/W'
ARMS=('self','population')
FOLDS=[1,9]


def inputs():
    args=SimpleNamespace(config='Config/c_abide_repair_v2_W.ini',output=str(OUT),cuda=0,k=10,
        modes=['content'],folds=FOLDS,epochs=None,training_seed=0,diagnostics=False,graph_weight_decay=.0005)
    cfg,dd,data,graphs,paths=setup(0,args.config,10)
    protocol=build_protocol(args,cfg,dd,data,graphs,paths)
    ref=read(REF/'PROTOCOL.json')
    for key in ('original_config','candidate_hash','data_sha256','samples','class_names',
                'epochs','training_seed','split_seed','checkpoint_selection','graph_weight_decay_override'):
        assert protocol[key]==ref[key],key
    with np.load(REF/'splits_and_graphs.npz') as saved:
        np.testing.assert_array_equal(saved['source_row'],dd['Index'])
        np.testing.assert_array_equal(saved['labels'],data['Label'].cpu().numpy())
        np.testing.assert_array_equal(saved['test_masks'],torch.stack([t for _,t in data['Mask']]).cpu().numpy())
    return cfg,dd,data,graphs,protocol


def prepare(cfg,dd,data,graphs,protocol):
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);return plan
    ref=read(REF/'PROTOCOL.json');verify_manifest(ref['code_sha256'])
    sources=dict(ref['code_sha256'])
    for name in [Path(__file__).name,'Model/late_patient_readout.py','diagnose_abide_capacity.py','tests/test_late_patient_readout.py']:
        sources[name]=sha(ROOT/name)
    reference_hashes={}
    for fold in FOLDS:
        for name in ('best.pt','complete.json','epochs.csv','predictions.npz'):
            path=REF/f'content/fold_{fold:02d}'/name;reference_hashes[str(path.relative_to(ROOT))]=sha(path)
    models=[];rng=[];opts=[];counts={}
    for arm in ARMS:
        SET_Random(0);model=LateReadoutModel(dd,cfg,graphs,arm,training_seed=0).to(cfg.Device)
        models.append(model);rng.append(torch.get_rng_state());counts[arm]=sum(p.numel() for p in model.parameters())
        opts.append(optimizer_manifest(model,relation_optimizer(model,cfg,.0005)))
    assert counts['self']==counts['population'] and opts[0]==opts[1]
    assert torch.equal(rng[0],rng[1])
    for (na,a),(nb,b) in zip(models[0].named_parameters(),models[1].named_parameters()):
        assert na==nb;torch.testing.assert_close(a,b,atol=0,rtol=0)
    late=models[1].classifier[0]
    edge=late.edge.cpu().numpy();weight=late.weight.cpu().numpy()
    OUT.mkdir(exist_ok=False)
    np.savez_compressed(OUT/'late_graph_and_splits.npz',edge=edge,weight=weight,
        labels=data['Label'].cpu().numpy(),source_row=np.array(dd['Index']),
        test_masks=torch.stack([m for _,m in data['Mask']]).cpu().numpy())
    plan=dict(created_at=timestamp(),arms=ARMS,folds=FOLDS,epochs=1000,seed=0,
        reference=str(REF),parameters=counts,protocol=protocol,
        late_block='z + 0.1 W_out GELU(W_in P LayerNorm(z)); H96->H96->H96; no added dropout',
        self_operator='P=I',population_operator='P=0.5 I + 0.5 D^-1 A, A=binary union of existing4 modality supports, no self edges',
        late_graph_edges=edge.shape[1],late_graph_sha256=sha(OUT/'late_graph_and_splits.npz'),
        optimizer='Preserve W: early graph decay0.0005; all other parameters including both matched late blocks decay0.002. Adam lr0.001, original cosine scheduler. No label-informed graph.',
        initialization='All C common weights preserved; added block uses forked RNG seed2903. Both controls have identical active parameters and RNG stream.',
        interpretation='Fixed original folds1,9, first highest TEST ACC over1000 epochs. Exploratory diagnosis, not tenfold result or heldout confirmation. If population beats self consistently it supports this implementation of late group integration; improvement of both over C may be added readout/optimization. No automatic topology/fusion tuning or tenfold follow-up.',
        tests='3 targeted tests passed: dense forward/gradient, dependency, exact common initialization/RNG and zero-residual equivalence, active gradients.',
        code_sha256=sources,reference_sha256=reference_hashes)
    for name in sources:
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json',plan)
    del models;torch.cuda.empty_cache()
    return plan


def summarize(records):
    base=[read(REF/f'content/fold_{f:02d}/complete.json') for f in FOLDS]
    names={'self':'后置自身读出','population':'后置患者间传播'}
    lines=['# ABIDE C后置患者传播诊断','',
        '固定seed0，原十折中的第1、9折，每折1000轮；其余沿用W。按首次最高测试ACC选轮，属于探索性两折诊断。','',
        '| 版本 | 第1折最佳ACC | 第9折最佳ACC | 两折均值 |','|---|---:|---:|---:|',
        f"| 原W/C | {base[0]['best']['acc']*100:.2f}% | {base[1]['best']['acc']*100:.2f}% | {np.mean([r['best']['acc'] for r in base])*100:.2f}% |"]
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        if len(rows)==2:
            aa=[r['best']['acc']*100 for r in rows]
            lines.append(f'| {names[arm]} | {aa[0]:.2f}% | {aa[1]:.2f}% | {np.mean(aa):.2f}% |')
    lines+=['','两种后置块的参数量、初始化、随机状态和正则化完全相同。仅P从单位矩阵变为0.5I+0.5邻居平均；邻接是现有模态候选边的并集。C前部传播、双分支相加和原分类器保持原实现。',
        '','每次训练保存最佳/末轮权重与预测并重放；新增训练ACC及CE曲线。原模型与两种新模型之间有额外参数差异，机制结论主要应比较两个等容量新版本。']
    (OUT/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cfg,dd,data,graphs,protocol=inputs();plan=prepare(cfg,dd,data,graphs,protocol)
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    if (OUT/'COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    runner.OUT=OUT
    try:
        runner.replay_reference()
        records=[]
        for arm in ARMS:
            def factory(dd,cfg,graphs,mode,training_seed=0,late_mode=arm):
                assert mode=='content'
                return LateReadoutModel(dd,cfg,graphs,late_mode,training_seed=training_seed)
            runner.RelationSheafModel=factory
            for fold in FOLDS:
                records.append(runner.run(arm,fold,cfg,dd,data,graphs))
                dump(OUT/'results.json',records);summarize(records);torch.cuda.empty_cache()
        verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'late_graph_and_splits.npz')==plan['late_graph_sha256']
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),runs=4,checkpoints_replayed=8,all_audits_passed=all(r['audit_passed'] for r in records)))
        dump(OUT/'status.json',dict(state='complete',runs=4,updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),updated_at=timestamp()));raise
    finally:
        runner.RelationSheafModel=RelationSheafModel;lock.unlink(missing_ok=True)


if __name__=='__main__':main()
