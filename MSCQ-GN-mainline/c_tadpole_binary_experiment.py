"""Direct transfer of frozen C to available TADPOLE SMCI/PMCI, seed0."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from relation_experiment import ROOT, setup, train, scores, dump, sha, timestamp, build_protocol
from alternating_cooperative_experiment import read, verify_manifest
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Utils import SET_Random
from Utils.relation_optimizer import relation_optimizer

OUT=ROOT/'Result/c_tadpole_binary_seed0_20260924'
CONFIG='Config/c_tadpole_binary_transfer.ini'


def arguments():
    return SimpleNamespace(config=CONFIG,output=str(OUT),cuda=0,k=10,modes=['content'],
        folds=list(range(1,11)),epochs=None,training_seed=0,diagnostics=False,
        graph_weight_decay=0.,resume=True)


def prepare():
    args=arguments()
    cfg,dd,data,graphs,paths=setup(0,CONFIG,10)
    assert cfg.Device.type=='cuda' and dd['Class_Names']==['SMCI','PMCI']
    old=read(ROOT/'Result/relation_mechanism_nodecay/seed_0/PROTOCOL.json')
    verify_manifest(old['code_sha256'])
    for k,v in old['original_config'].items():
        if k!='Task':assert getattr(cfg,k)==v,k
    assert cfg.epochs==1000 and len(data['Mask'])==10
    protocol=build_protocol(args,cfg,dd,data,graphs,paths)
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256'])
        assert plan['protocol']==protocol
        return cfg,dd,data,graphs,plan
    masks=torch.stack([t for _,t in data['Mask']])
    assert torch.all(masks.sum(0)==1)
    assert torch.isfinite(data['Feature']).all()
    assert sorted(data['Label'].unique().tolist())==[0,1]
    for fit,test in data['Mask']:
        assert not (fit&test).any() and (fit|test).all()
        assert data['Label'][fit].unique().numel()==data['Label'][test].unique().numel()==2
    SET_Random(0)
    model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=relation_optimizer(model,cfg,0.)
    criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    for _ in range(6):
        model.train();optimizer.zero_grad(set_to_none=True)
        z,emb,aux=model(data['Feature'])
        assert z.shape==(dd['Sample_Num'],2) and len(emb)==len(aux)==2
        loss=criterion(z,data['Label'],data['Mask'][0][0],emb,aux)
        assert torch.isfinite(loss);loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
            if name.startswith('propagation.'):assert p.grad is not None,name
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
        optimizer.step();model.eval()
        with torch.no_grad():assert torch.isfinite(model(data['Feature'])[0]).all()
    del model,optimizer,criterion,z,emb,aux,loss
    torch.cuda.empty_cache()
    counts=torch.bincount(data['Label'],minlength=2).cpu().tolist()
    sources={**protocol['code_sha256'],__file__.split(os.sep)[-1]:sha(__file__)}
    plan=dict(experiment='Frozen C transfer to TADPOLE SMCI/PMCI',protocol=protocol,
        parameters=parameters,samples=dd['Sample_Num'],class_names=dd['Class_Names'],class_counts=counts,
        modalities=dict(zip(dd['Modal_Name'],map(len,dd['Modal_Index']))),
        training='seed0, original stratified tenfold protocol, 1000 epochs; original test-ACC selection; graph decay0',
        adaptation='Actual file has five modalities and no PET; five candidate graphs, 25 relation/content groups, two class queries and output classes. No missing PET imputation.',
        supplementary_metrics='PMCI-positive probability AUC/AP, balanced accuracy, PMCI recall, SMCI recall; majority-class baseline; no extra threshold tuning or alternative checkpoint selection.',
        caveat='Same TADPOLE source, distinct classification task; not an independent external cohort. Original processed features and full-cohort class weights retained. No patient ID is present to establish independence from the prior task.',
        smoke=dict(passed=True,steps=6,test_metrics_inspected=False),code_sha256=sources,
        environment=dict(executable=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
    OUT.mkdir(parents=True,exist_ok=False)
    for name in sources:
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    dump(OUT/'PLAN.json',plan)
    dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp(),planned_runs=10))
    return cfg,dd,data,graphs,plan


def audit_and_report(cfg,dd,data,graphs,plan):
    verify_manifest(plan['code_sha256']);rows=[]
    model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device).eval()
    for fold in range(1,11):
        folder=OUT/f'content/fold_{fold:02d}';record=read(folder/'complete.json')
        with (folder/'epochs.csv').open() as stream:curve=list(csv.DictReader(stream))
        assert [int(r['epoch']) for r in curve]==list(range(1,1001))
        assert int(np.argmax([float(r['test_acc']) for r in curve]))+1==record['best']['epoch']
        cp=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
        assert (cp['mode'],cp['fold'],cp['seed'])==('content',fold,0)
        model.load_state_dict(cp['state_dict'])
        test=data['Mask'][fold-1][1]
        with torch.no_grad():replay=model(data['Feature'])[0][test]
        for stage,file in (('best','predictions.npz'),('last','last_predictions.npz')):
            with np.load(folder/file) as saved:
                z=torch.as_tensor(saved['logits'],device=cfg.Device)
                y=torch.as_tensor(saved['labels'],device=cfg.Device)
                np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
                torch.testing.assert_close(y,data['Label'][test],atol=0,rtol=0)
                if stage=='best':
                    torch.testing.assert_close(z,replay,atol=2e-5,rtol=2e-5)
                    assert torch.equal(z.argmax(-1),replay.argmax(-1))
                actual=scores(z,y)
                for key in ('acc','auc','f1','macro_f1'):assert abs(actual[key]-record[stage][key])<1e-12
                truth=y.cpu().numpy();prob=z.softmax(-1)[:,1].cpu().numpy();pred=z.argmax(-1).cpu().numpy()
                rows.append(dict(fold=fold,stage=stage,epoch=record[stage]['epoch'],**actual,
                    pmci_auc=float(roc_auc_score(truth,prob)),pmci_ap=float(average_precision_score(truth,prob)),
                    balanced_acc=float(balanced_accuracy_score(truth,pred))))
    fields=('acc','macro_f1','balanced_acc','pmci_auc','pmci_ap')
    aggregate={stage:{k:float(np.mean([r[k] for r in rows if r['stage']==stage])) for k in fields} for stage in ('best','last')}
    result=dict(passed=True,verified_best_checkpoints=10,verified_saved_prediction_files=20,records=rows,mean_metrics=aggregate,
        majority_accuracy=plan['class_counts'][0]/plan['samples'],caveat='Last epoch metrics verified from predictions; legacy runner saves only best checkpoint.')
    dump(OUT/'audit.json',result)
    lines=['# C：TADPOLE SMCI/PMCI 二分类','',f"样本 {plan['samples']}：SMCI {plan['class_counts'][0]}，PMCI {plan['class_counts'][1]}。",
        '实际输入为五个模态，无PET。seed0、十折、每折1000轮；沿用原测试ACC选轮。',
        '同一TADPOLE来源的另一任务，不作为独立外部队列验证。','',
        '| 检查点 | ACC | 宏F1 | 平衡准确率 | PMCI概率AUC | PMCI平均精确率AP |','|---|---:|---:|---:|---:|---:|']
    for stage,v in aggregate.items():lines.append('| '+stage+' | '+' | '.join(f'{100*v[k]:.4f}%' for k in fields)+' |')
    lines += ['',f"始终预测SMCI的全体样本ACC为 {100*result['majority_accuracy']:.4f}%；应结合PMCI召回率判断效果。",
              '10个最佳检查点重放通过，20个最佳/末轮预测文件复核通过；逐折类别召回率见audit.json。',
              '概率AUC/AP为补充指标，不与原始双logit宏AUC混用；未调阈值或改变检查点选择。']
    (OUT/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(OUT/'VERIFIED_COMPLETE.json',dict(completed_at=timestamp(),audit=result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');a=parser.parse_args()
    cfg,dd,data,graphs,plan=prepare()
    if a.prepare:print(json.dumps({k:plan[k] for k in ('samples','class_counts','modalities','parameters','smoke')},ensure_ascii=False),flush=True);return
    if (OUT/'VERIFIED_COMPLETE.json').exists():return
    lock=OUT/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        train(arguments())
        dump(OUT/'status.json',dict(state='auditing',updated_at=timestamp()))
        audit_and_report(cfg,dd,data,graphs,plan)
        dump(OUT/'status.json',dict(state='complete',verified=True,completed_runs=10,updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),updated_at=timestamp()));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
