"""Seed-0 retrained leave-one-modality-out controls in full ordinary fusion."""
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
from Model.modal_evidence import ModalEvidenceModel
from Model.subset_fusion import SubsetFusionModel
from relation_experiment import ROOT,dump,sha,timestamp
from unimodal_round_experiment import read,point,verify_manifest


OUT=ROOT/'Result/full_modal_ablation_seed0_20260923'
REFERENCE=ROOT/'Result/modal_evidence_fusion_20260922'
EXCLUDED={'NO_MRI':'MRI','NO_PET':'PET','NO_CSF':'CSF','NO_RISK':'RISK_FACTOR',
          'NO_COG':'COGNITIVE_TEST','NO_ROI':'ROI_AVERAGE'}
NEW_SOURCES=('Model/subset_fusion.py','unimodal_round_experiment.py',
             'Model/single_modal_evidence.py','full_modal_ablation_experiment.py',
             'tests/test_full_modal_ablation.py')


def make_model(dd,cfg,mode,seed):
    if mode not in EXCLUDED:
        raise ValueError(mode)
    return SubsetFusionModel(dd,cfg,[m for m in dd['Modal_Name'] if m!=EXCLUDED[mode]],seed).to(cfg.Device)


def load_full_state(model,state):
    """Translate full-fusion checkpoint parameters without changing weights."""
    selected=model.selected.tolist()
    def load_part(module,prefix):
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    with torch.no_grad():
        model.gain.copy_(state['Feature_Modal.modal_gain.weight'][selected])
        model.bias.copy_(state['Feature_Modal.modal_bias.weight'][selected])
        model.modality_id.copy_(state['modal_token_encoder.modal_id_embedding.weight'][selected])
        model.gate_logit.copy_(state['modal_gate_logit'][selected])
        model.noise_std.copy_(state['_modal_noise_std'][selected])
        for local,original in enumerate(selected):
            load_part(model.encoders[local],f'modal_token_encoder.modal_proj.{original}.')
        load_part(model.norm,'modal_token_encoder.input_norm.')
        weight=state['fusion_head.0.weight']
        columns=torch.cat([torch.arange(i*model.hidden,(i+1)*model.hidden,device=weight.device) for i in selected])
        model.head[0].weight.copy_(weight.index_select(1,columns))
        model.head[0].bias.copy_(state['fusion_head.0.bias'])
        model.head[2].weight.copy_(state['fusion_head.2.weight'])
        model.head[2].bias.copy_(state['fusion_head.2.bias'])


def verify_reference(cfg,dd,data,hashes):
    plan=read(REFERENCE/'PLAN.json')
    historical.verify_sources(plan)
    assert plan['data_sha256']==hashes and plan['epochs']==cfg.epochs==1000
    assert plan['environment']['torch']==torch.__version__
    manifest={str((REFERENCE/'PLAN.json').relative_to(ROOT)):sha(REFERENCE/'PLAN.json')}
    full=ModalEvidenceModel(dd,cfg,'fusion',0,rank=8).to(cfg.Device).eval()
    initial_hash=historical.state_hash(full)
    converted=SubsetFusionModel(dd,cfg,dd['Modal_Name'],0).to(cfg.Device).eval()
    verified=[]
    for fold in range(1,11):
        folder=REFERENCE/f'seed_0/fusion/fold_{fold:02d}'
        record=read(folder/'complete.json')
        assert (record['seed'],record['fold'],record['mode'])==(0,fold,'fusion')
        assert initial_hash==record['initial_model_sha256']
        with (folder/'epochs.csv').open(encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==1000 and [int(r['epoch']) for r in rows]==list(range(1,1001))
        assert int(np.argmax([float(r['test_acc']) for r in rows]))+1==record['best']['epoch']
        _,test=data['Mask'][fold-1]
        errors={}
        for stage in ('best','last'):
            path=folder/f'{stage}.pt'
            assert sha(path)==record[f'{stage}_checkpoint_sha256']
            checkpoint=torch.load(path,map_location=cfg.Device,weights_only=False)
            assert checkpoint['protocol_sha256']==sha(REFERENCE/'PLAN.json')
            full.load_state_dict(checkpoint['state_dict'],strict=True)
            load_full_state(converted,checkpoint['state_dict'])
            with torch.no_grad():
                original_output=full(data['Feature'])['logits'][test]
                converted_output=converted(data['Feature'])['logits'][test]
            saved=point(folder,stage)
            np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
            np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
            np.testing.assert_allclose(saved['logits'],original_output.cpu().numpy(),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(converted_output,original_output,atol=2e-5,rtol=2e-5)
            metrics=historical.extra_scores(original_output,data['Label'][test])
            for key in ('acc','auc','f1','macro_f1','nll','brier'):
                assert abs(metrics[key]-record[stage][key])<2e-5
            errors[stage]=float((converted_output-original_output).abs().max())
        for filename in ('complete.json','best.json','epochs.csv','best.pt','last.pt','predictions.npz','last_predictions.npz'):
            path=folder/filename
            manifest[str(path.relative_to(ROOT))]=sha(path)
        verified.append(dict(seed=0,fold=fold,converted_max_abs_error=errors))
    return manifest,dict(passed=True,seeds=[0],verified_runs=10,verified_checkpoints=20,
                         converted_checkpoint_replays=20,records=verified)


def smoke(cfg,dd,data):
    runs=[]
    fit,_=data['Mask'][0]
    for mode in EXCLUDED:
        model=make_model(dd,cfg,mode,0)
        assert len(model.modalities)==len(dd['Modal_Name'])-1
        assert EXCLUDED[mode] not in model.modalities
        removed=dd['Modal_Index'][dd['Modal_Name'].index(EXCLUDED[mode])]
        changed=data['Feature'].clone();changed[:,removed]=1234
        for training in (False,True):
            model.train(training)
            torch.manual_seed(701)
            with torch.no_grad():
                original=model(data['Feature'])['logits']
            torch.manual_seed(701)
            with torch.no_grad():
                perturbed=model(changed)['logits']
            torch.testing.assert_close(original,perturbed,atol=0,rtol=0)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        start=time.perf_counter()
        for _ in range(20):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(data['Feature'])['logits'][fit],data['Label'][fit],
                                 weight=dd['Label_Weight'],label_smoothing=.05)
            loss.backward()
            for name,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        runs.append(dict(mode=mode,excluded=EXCLUDED[mode],input_isolation_passed=True,
                         counts=model.parameter_counts(),seconds_per_epoch=(time.perf_counter()-start)/20))
    return dict(passed=True,inspected_test_metrics=False,runs=runs)


def summarize(out,plan):
    records,folders,variants={},{},{}
    for mode in ['FULL']+list(EXCLUDED):
        records[mode],folders[mode]={},{}
        for fold in plan['folds']:
            folder=(REFERENCE/f'seed_0/fusion/fold_{fold:02d}' if mode=='FULL'
                    else out/f'seed_0/{mode}/fold_{fold:02d}')
            if (folder/'complete.json').exists():
                records[mode][fold]=read(folder/'complete.json')
                folders[mode][fold]=folder
        rows=list(records[mode].values())
        v=dict(completed_runs=len(rows),planned_runs=10,seed=0,reused=mode=='FULL')
        if rows:
            for stage in ('best','last'):
                v[stage]={key:np.mean([r[stage][key] for r in rows],axis=0).tolist()
                          for key in ('acc','auc','f1','macro_f1','nll','brier','per_class_recall')}
            v['parameters']=rows[0]['parameters']
            v['mean_last100_acc']=float(np.mean([r['mean_last100_acc'] for r in rows]))
        variants[mode]=v
    comparisons=[]
    for removed in EXCLUDED:
        folds=sorted(records['FULL'].keys()&records[removed].keys())
        if not folds:
            continue
        result=dict(left='FULL',right=removed,excluded_modality=EXCLUDED[removed],matched_runs=len(folds),stages={})
        for stage in ('best','last'):
            by_class=np.zeros((len(plan['class_names']),2),dtype=int)
            corrected=harmed=0
            deltas=[]
            for fold in folds:
                a,b=point(folders['FULL'][fold],stage),point(folders[removed][fold],stage)
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
    completed=sum(variants[mode]['completed_runs'] for mode in EXCLUDED)
    summary=dict(updated_at=timestamp(),seeds=[0],completed_new_runs=completed,planned_new_runs=60,
                 reused_runs=10,variants=variants,comparisons=comparisons,
                 comparison_sign='Positive comparison delta = FULL better than retrained leave-one-out model')
    dump(out/'summary.json',summary)
    lines=['# 六模态普通融合：逐一移除并重新训练','',f'新增训练完成 {completed}/60；复用种子 0 的全六模态普通拼接 10 折。',
           '种子 0，原十折、1000 轮、首次最高测试 ACC；仅最终分类 CE，不含患者图。',
           '每组移除一类模态的编码器及分类头输入列；分类头隐藏宽度固定为 43。参数量随移除模态变化。','',
           '| 模型 | 完成 | 择优 ACC | 末轮 ACC | 参数量 |','|---|---:|---:|---:|---:|']
    for mode,v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {mode} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['last']['acc']*100:.2f}% | {v['parameters']} |")
        else:
            lines.append(f'| {mode} | 0/10 | — | — | — |')
    lines+=['','不同完成折数不能直接排名；summary.json 的配对差值为 FULL 减移除组，正值表示保留该模态更好。',
            '这是当前融合框架内重新训练的消融，包含信息、参数量和优化变化，不是纯信息贡献或等参数对照。',
            '当前普通融合不等同于历史含患者图的 C 模型；不能据此外推完整 C 中的模态价值。',
            '本轮不自动追加组合搜索、模型结构或其他种子。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def prepare():
    historical.SEEDS=(0,)
    cfg,dd,data,original,hashes,baseline=historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json')
        historical.verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256']==hashes and plan['baseline_manifest']==baseline
        assert plan['seeds']==[0] and plan['planned_runs']==60
        assert plan['environment']['torch']==torch.__version__
        return cfg,dd,data,plan
    manifest,audit=verify_reference(cfg,dd,data,hashes)
    checks=smoke(cfg,dd,data)
    assert all(r['counts']['fusion_width']==43 for r in checks['runs'])
    ref_plan=read(REFERENCE/'PLAN.json')
    sources=tuple(dict.fromkeys(historical.SOURCES+NEW_SOURCES))
    plan=dict(experiment='All-six-modality fusion with one-modality-at-a-time retrained removal',
        seeds=[0],folds=list(range(1,11)),modes=list(EXCLUDED),excluded=EXCLUDED,
        epochs=1000,planned_runs=60,reused_runs=10,class_names=dd['Class_Names'],samples=dd['Sample_Num'],
        modalities=dd['Modal_Name'],modal_dimensions=[len(i) for i in dd['Modal_Index']],
        original_config=original['original_config'],objective='Original train-mask CE with class weights and smoothing .05',
        head='Linear-GELU-Linear, width 43 in all arms; selected input columns from same historical initialization',
        input_scope='Five remaining modalities only; omitted values cannot affect output; no graph',
        training='Unchanged historical.train_arm; Adam, original scheduler, weight decay and clip',
        priors=ref_plan['priors'],checkpoint_selection=ref_plan['checkpoint_selection'],
        randomness='Original full-shape input noise and token dropout; training seed 0 reset each arm/fold',
        initialization='Common encoder/head weights match; all-full checkpoint translation replay verified within tolerance',
        schedule='fold then omitted modality',
        data_sha256=hashes,baseline_manifest=baseline,reference_manifest=manifest,
        code_sha256={source:sha(ROOT/source) for source in sources},smoke=checks,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        interpretation='Model-dependent retrained ablation, not pure conditional information. Active parameter counts differ. Different floating-point implementations need not have bitwise-identical long training trajectories.',
        scope='Single training seed, no automatic subset search or conditional gating')
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'PLAN.json',plan);dump(OUT/'reference_audit.json',audit)
    for source in sources:
        dest=OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/source).read_bytes())
    np.savez_compressed(OUT/'data_and_splits.npz',source_row=np.asarray(dd['Index']),labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([test.cpu().numpy() for _,test in data['Mask']]))
    summarize(OUT,plan);dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp()))
    return cfg,dd,data,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,plan=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared',planned_runs=60,smoke=plan['smoke']),ensure_ascii=False),flush=True)
        return
    historical.OUT=OUT;historical.make_model=make_model;historical.summarize=summarize
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            for mode in plan['modes']:
                historical.train_arm(cfg,dd,data,plan,0,fold,mode)
        dump(OUT/'status.json',dict(state='auditing',pid=os.getpid(),updated_at=timestamp()))
        audit=historical.audit_completed(cfg,dd,data,plan)
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
