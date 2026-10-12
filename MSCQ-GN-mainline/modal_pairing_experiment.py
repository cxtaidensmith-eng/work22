"""Fixed within-fold PET/risk pairing controls, seed 0, no model changes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import modal_evidence_experiment as historical
import cognitive_pair_experiment as pair_round
from relation_experiment import ROOT,dump,sha,timestamp
from unimodal_round_experiment import read,point,verify_manifest
from Utils import SET_Random
from Utils.modal_pairing import within_split_derangement,permute_modal_block


OUT=ROOT/'Result/modal_pairing_seed0_20260923'
REFERENCE=pair_round.OUT
ARMS={'COG_PET_SHUFFLED':('COG_PET','PET'),
      'COG_RISK_SHUFFLED':('COG_RISK','RISK_FACTOR')}
REFERENCE_MODES=('COG','COG_PET','COG_RISK')
MAPPING_SEED_BASE=71100
NEW_SOURCES=('Utils/modal_pairing.py','modal_pairing_experiment.py','tests/test_modal_pairing.py')


def make_model(dd,cfg,mode,seed):
    return pair_round.make_model(dd,cfg,ARMS[mode][0],seed)


def transformed_data(dd,data,mode,mapping):
    modality=ARMS[mode][1]
    columns=dd['Modal_Index'][dd['Modal_Name'].index(modality)]
    changed=dict(data)
    changed['Feature']=permute_modal_block(data['Feature'],columns,mapping)
    assert changed['Label'] is data['Label'] and changed['Mask'] is data['Mask']
    return changed


def make_mappings(data):
    return np.stack([within_split_derangement(fit.cpu().numpy(),test.cpu().numpy(),MAPPING_SEED_BASE+fold)
                     for fold,(fit,test) in enumerate(data['Mask'],1)])


def verify_references(cfg,dd,data,hashes):
    plan=read(REFERENCE/'PLAN.json')
    historical.verify_sources(plan)
    assert plan['data_sha256']==hashes and plan['seeds']==[0]
    assert plan['epochs']==cfg.epochs==1000
    assert plan['environment']['torch']==torch.__version__
    manifest={str((REFERENCE/'PLAN.json').relative_to(ROOT)):sha(REFERENCE/'PLAN.json')}
    verified=[]
    for mode in REFERENCE_MODES:
        model=pair_round.make_model(dd,cfg,mode,0).eval()
        initial_hash=historical.state_hash(model)
        for fold in range(1,11):
            folder=REFERENCE/f'seed_0/{mode}/fold_{fold:02d}'
            record=read(folder/'complete.json')
            assert (record['seed'],record['fold'],record['mode'])==(0,fold,mode)
            assert initial_hash==record['initial_model_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            assert len(rows)==1000 and [int(r['epoch']) for r in rows]==list(range(1,1001))
            assert int(np.argmax([float(r['test_acc']) for r in rows]))+1==record['best']['epoch']
            _,test=data['Mask'][fold-1]
            for stage in ('best','last'):
                path=folder/f'{stage}.pt'
                assert sha(path)==record[f'{stage}_checkpoint_sha256']
                checkpoint=torch.load(path,map_location=cfg.Device,weights_only=False)
                assert checkpoint['protocol_sha256']==sha(REFERENCE/'PLAN.json')
                model.load_state_dict(checkpoint['state_dict'],strict=True)
                with torch.no_grad():
                    logits=model(data['Feature'])['logits'][test]
                saved=point(folder,stage)
                np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
                np.testing.assert_allclose(saved['logits'],logits.cpu().numpy(),atol=2e-5,rtol=2e-5)
                metrics=historical.extra_scores(logits,data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','nll','brier'):
                    assert abs(metrics[key]-record[stage][key])<2e-5
            for file in ('complete.json','best.json','epochs.csv','best.pt','last.pt','predictions.npz','last_predictions.npz'):
                path=folder/file
                manifest[str(path.relative_to(ROOT))]=sha(path)
            verified.append(dict(mode=mode,seed=0,fold=fold))
    return manifest,dict(passed=True,seeds=[0],verified_runs=30,verified_checkpoints=60,records=verified)


def smoke(cfg,dd,data,mappings):
    checks=[]
    fit,_=data['Mask'][0]
    for mode,(real,_) in ARMS.items():
        SET_Random(0)
        model=make_model(dd,cfg,mode,0)
        original=pair_round.make_model(dd,cfg,real,0)
        assert historical.state_hash(model)==historical.state_hash(original)
        record=read(REFERENCE/f'seed_0/{real}/fold_01/complete.json')
        assert historical.state_hash(model)==record['initial_model_sha256']
        assert model.parameter_counts()['total']==record['parameters']
        changed=transformed_data(dd,data,mode,mappings[0])
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        start=time.perf_counter()
        for _ in range(20):
            model.train();optimizer.zero_grad(set_to_none=True)
            output=model(changed['Feature'])
            loss=F.cross_entropy(output['logits'][fit],data['Label'][fit],weight=dd['Label_Weight'],label_smoothing=.05)
            loss.backward()
            for name,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        checks.append(dict(mode=mode,matched_initial_state_and_parameter_count=True,
                           parameters=model.parameter_counts()['total'],seconds_per_epoch=(time.perf_counter()-start)/20))
    return dict(passed=True,inspected_test_metrics=False,runs=checks)


def summarize(out,plan):
    records,folders,variants={},{},{}
    for mode in list(ARMS)+list(REFERENCE_MODES):
        records[mode],folders[mode]={},{}
        for fold in plan['folds']:
            base=out if mode in ARMS else REFERENCE
            folder=base/f'seed_0/{mode}/fold_{fold:02d}'
            if (folder/'complete.json').exists():
                records[mode][fold]=read(folder/'complete.json')
                folders[mode][fold]=folder
        rows=list(records[mode].values())
        v=dict(completed_runs=len(rows),planned_runs=10,seed=0,reused=mode not in ARMS)
        if rows:
            for stage in ('best','last'):
                v[stage]={key:np.mean([r[stage][key] for r in rows],axis=0).tolist()
                          for key in ('acc','auc','f1','macro_f1','nll','brier','per_class_recall')}
            v['parameters']=rows[0]['parameters']
            v['mean_last100_acc']=float(np.mean([r['mean_last100_acc'] for r in rows]))
        variants[mode]=v
    pairs=[(real,shuffled) for shuffled,(real,_) in ARMS.items()]
    pairs+=[(mode,'COG') for mode in list(ARMS)+['COG_PET','COG_RISK']]
    comparisons=[]
    for left,right in pairs:
        folds=sorted(records[left].keys()&records[right].keys())
        if not folds:
            continue
        result=dict(left=left,right=right,matched_runs=len(folds),stages={})
        for stage in ('best','last'):
            by_class=np.zeros((len(plan['class_names']),2),dtype=int)
            corrected=harmed=0
            deltas=[]
            for fold in folds:
                a,b=point(folders[left][fold],stage),point(folders[right][fold],stage)
                np.testing.assert_array_equal(a['row_index'],b['row_index'])
                np.testing.assert_array_equal(a['labels'],b['labels'])
                ca,cb=a['logits'].argmax(-1)==a['labels'],b['logits'].argmax(-1)==b['labels']
                gain,harm=ca&~cb,~ca&cb
                corrected+=int(gain.sum());harmed+=int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c]+=[int((gain&(a['labels']==c)).sum()),int((harm&(a['labels']==c)).sum())]
                deltas.append(float(ca.mean()-cb.mean()))
            result['stages'][stage]=dict(mean_acc_delta=float(np.mean(deltas)),corrected=corrected,harmed=harmed,
                net_corrected=corrected-harmed,per_class_corrected_harmed=by_class.tolist(),
                fold_acc_deltas=dict(zip(folds,deltas)),fold_wins=sum(d>1e-12 for d in deltas),
                fold_ties=sum(abs(d)<=1e-12 for d in deltas),fold_losses=sum(d< -1e-12 for d in deltas))
        comparisons.append(result)
    completed=sum(variants[mode]['completed_runs'] for mode in ARMS)
    summary=dict(updated_at=timestamp(),seeds=[0],completed_new_runs=completed,planned_new_runs=20,
                 reused_runs=30,variants=variants,comparisons=comparisons)
    dump(out/'summary.json',summary)
    lines=['# 真实模态配对与置乱配对：PET／风险因素','',f'新增训练完成 {completed}/20；复用种子 0 的认知、真实 PET、真实风险因素共 30 次训练。',
           '种子 0，原十折、1000 轮、首次最高测试 ACC；普通拼接，最终分类 CE。',
           '置乱整组模态特征，不使用标签；训练／测试内部独立交换，每折固定映射。',
           '真实与置乱组模型结构、参数量、初始化、训练设置相同。','',
           '| 模型 | 完成 | 择优 ACC | AD 召回 | 末轮 ACC |','|---|---:|---:|---:|---:|']
    for mode,v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {mode} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['best']['per_class_recall'][0]*100:.2f}% | {v['last']['acc']*100:.2f}% |")
        else:
            lines.append(f'| {mode} | 0/10 | — | — | — |')
    lines+=['','不同完成折数不能直接排名；同折配对纠错见 summary.json。',
            '单次置乱、单训练种子属于信息消融筛查，不是置换显著性检验。',
            '真实优于置乱只说明对应关系影响当前模型；仍须结合认知单模态判断净增益，不能直接称为纯互补信息。',
            '本轮没有删除其他模态，也不会自动启动新结构或其他种子。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def prepare():
    historical.SEEDS=(0,)
    cfg,dd,data,original,hashes,baseline=historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    mappings=make_mappings(data)
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json')
        historical.verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256']==hashes and plan['baseline_manifest']==baseline
        assert plan['seeds']==[0] and plan['planned_runs']==20
        assert plan['environment']['torch']==torch.__version__
        assert sha(OUT/'pairing_maps.npz')==plan['pairing_maps_sha256']
        with np.load(OUT/'pairing_maps.npz') as saved:
            np.testing.assert_array_equal(saved['donor_rows'],mappings)
            np.testing.assert_array_equal(saved['source_row'],np.asarray(dd['Index']))
        return cfg,dd,data,plan,mappings
    manifest,audit=verify_references(cfg,dd,data,hashes)
    checks=smoke(cfg,dd,data,mappings)
    ref_plan=read(REFERENCE/'PLAN.json')
    sources=tuple(dict.fromkeys(tuple(ref_plan['code_sha256'])+NEW_SOURCES))
    OUT.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(OUT/'pairing_maps.npz',donor_rows=mappings,source_row=np.asarray(dd['Index']),
                        donor_source_row=np.asarray(dd['Index'])[mappings],labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([t.cpu().numpy() for _,t in data['Mask']]))
    plan=dict(experiment='True versus fixed shuffled PET/risk patient pairing',seeds=[0],folds=list(range(1,11)),
        modes=list(ARMS),arms=ARMS,epochs=1000,planned_runs=20,reused_runs=30,
        class_names=dd['Class_Names'],samples=dd['Sample_Num'],original_config=original['original_config'],
        objective='Original final fused-logit CE only, original weights/smoothing, training mask',
        mapping=dict(seed_base=MAPPING_SEED_BASE,recipe='Per fold seed 71100+fold; independent NumPy SeedSequence role 0=train, 1=test',
                     block='All columns of one modality together; train/test separately; no original-row matches',
                     shared_between_arms=True,labels_used=False,fixed_for_all_epochs=True,training_rng_unchanged=True),
        pairing_maps_sha256=sha(OUT/'pairing_maps.npz'),
        model='Exactly the true COG_PET / COG_RISK SubsetFusionModel, same initialization and parameter count',
        training='Unchanged historical.train_arm; only selected modality input rows are permuted',
        priors=ref_plan['priors'],randomness=ref_plan['randomness'],checkpoint_selection=ref_plan['checkpoint_selection'],
        data_sha256=hashes,baseline_manifest=baseline,reference_manifest=manifest,
        code_sha256={source:sha(ROOT/source) for source in sources},smoke=checks,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        interpretation='One pairing realization per fold, one training seed; no permutation p-value. True>shuffled is not by itself proof of net gain over cognitive-only or pure synergy.',
        scope='No new architecture, no conditional gate, no extra seeds or automatic tuning')
    dump(OUT/'PLAN.json',plan);dump(OUT/'reference_audit.json',audit)
    for source in sources:
        dest=OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/source).read_bytes())
    summarize(OUT,plan);dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp()))
    return cfg,dd,data,plan,mappings


def audit_completed(cfg,dd,data,plan,mappings):
    assert sha(OUT/'pairing_maps.npz')==plan['pairing_maps_sha256']
    verified=[]
    for fold in plan['folds']:
        for mode,(real,_) in ARMS.items():
            changed=transformed_data(dd,data,mode,mappings[fold-1])
            local=dict(plan,folds=[fold],modes=[mode],planned_runs=1)
            replay=historical.audit_completed(cfg,dd,changed,local)
            record=read(OUT/f'seed_0/{mode}/fold_{fold:02d}/complete.json')
            reference=read(REFERENCE/f'seed_0/{real}/fold_{fold:02d}/complete.json')
            assert record['initial_model_sha256']==reference['initial_model_sha256']
            assert record['parameters']==reference['parameters']
            verified.extend(replay['records'])
    audit=dict(passed=True,verified_runs=len(verified),planned_runs=20,verified_checkpoints=2*len(verified),
               matched_real_model_initialization=True,matched_parameter_count=True,
               within_split_derangement_verified=True,records=verified,updated_at=timestamp())
    assert len(verified)==20
    dump(OUT/'audit.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,plan,mappings=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared',planned_runs=20,smoke=plan['smoke']),ensure_ascii=False),flush=True)
        return
    historical.OUT=OUT;historical.make_model=make_model;historical.summarize=summarize
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            for mode in plan['modes']:
                changed=transformed_data(dd,data,mode,mappings[fold-1])
                historical.train_arm(cfg,dd,changed,plan,0,fold,mode)
        dump(OUT/'status.json',dict(state='auditing',pid=os.getpid(),updated_at=timestamp()))
        audit=audit_completed(cfg,dd,data,plan,mappings)
        verify_manifest(plan['reference_manifest'])
        summary=summarize(OUT,plan)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),pid=os.getpid(),updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
