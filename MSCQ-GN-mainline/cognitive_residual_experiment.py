"""Seed-0 direct cognitive path plus joint supplement, final CE only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import modal_evidence_experiment as historical
import cognitive_pair_experiment as pair_round
import cognitive_pet_aux_experiment as previous
from Model.cognitive_residual import CognitiveResidualModel
from relation_experiment import ROOT, dump, sha, timestamp
from unimodal_round_experiment import read, point, verify_manifest
from Utils import SET_Random


OUT = ROOT / 'Result/cognitive_residual_seed0_20260923'
REFERENCE = previous.REFERENCE
MODE = 'COG_PET_RESIDUAL'
NEW_SOURCES = ('Model/cognitive_residual.py', 'cognitive_residual_experiment.py',
               'tests/test_cognitive_residual.py')


def make_model(dd, cfg, mode, seed):
    if mode != MODE:
        raise ValueError(mode)
    return CognitiveResidualModel(dd, cfg, 'PET', seed).to(cfg.Device)


def smoke(cfg, dd, data):
    cognitive = pair_round.make_model(dd, cfg, 'COG', 0)
    joint = pair_round.make_model(dd, cfg, 'COG_PET', 0)
    model = make_model(dd, cfg, MODE, 0)
    for name, state in joint.state_dict().items():
        if name not in ('head.2.weight', 'head.2.bias'):
            torch.testing.assert_close(state, model.state_dict()[name], atol=0, rtol=0)
    for name, state in cognitive.head.state_dict().items():
        torch.testing.assert_close(state, model.cognitive_head.state_dict()[name], atol=0, rtol=0)
    for training in (False, True):
        cognitive.train(training); model.train(training)
        SET_Random(1702)
        with torch.no_grad():
            expected = cognitive(data['Feature'])['logits']
        after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        SET_Random(1702)
        with torch.no_grad():
            output = model(data['Feature'])
        torch.testing.assert_close(expected, output['logits'], atol=2e-6, rtol=2e-6)
        assert not torch.count_nonzero(output['pair_logits'])
        assert torch.equal(after[0], torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(after[1],torch.cuda.get_rng_state_all()))
    SET_Random(0)
    model = make_model(dd, cfg, MODE, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fit, _ = data['Mask'][0]
    start = time.perf_counter()
    for step in range(20):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(data['Feature'])
        loss = F.cross_entropy(output['logits'][fit], data['Label'][fit],
                               weight=dd['Label_Weight'], label_smoothing=.05)
        loss.backward()
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if step > 0:
            assert model.head[0].weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    return dict(passed=True, inspected_test_metrics=False, initial_prediction_matches_cognitive=True,
                initial_joint_supplement_zero=True, matched_common_parameters_and_rng=True,
                counts=model.parameter_counts(), seconds_per_epoch=(time.perf_counter()-start)/20)


def summarize(out, plan):
    records, folders, variants = {}, {}, {}
    for mode in (MODE, 'COG_PET', 'COG'):
        records[mode], folders[mode] = {}, {}
        for fold in plan['folds']:
            base = out if mode == MODE else REFERENCE
            folder = base/f'seed_0/{mode}/fold_{fold:02d}'
            if (folder/'complete.json').exists():
                records[mode][fold]=read(folder/'complete.json')
                folders[mode][fold]=folder
        rows=list(records[mode].values())
        v=dict(completed_runs=len(rows),planned_runs=10,seed=0,reused=mode!=MODE)
        if rows:
            for stage in ('best','last'):
                v[stage]={key:np.mean([r[stage][key] for r in rows],axis=0).tolist()
                          for key in ('acc','auc','f1','macro_f1','nll','brier','per_class_recall')}
            v['parameters']=rows[0]['parameters']
            v['mean_last100_acc']=float(np.mean([r['mean_last100_acc'] for r in rows]))
        variants[mode]=v
    comparisons=[]
    for reference in ('COG_PET','COG'):
        folds=sorted(records[MODE].keys() & records[reference].keys())
        if not folds:
            continue
        result=dict(left=MODE,right=reference,matched_runs=len(folds),stages={})
        for stage in ('best','last'):
            by_class=np.zeros((len(plan['class_names']),2),dtype=int)
            corrected=harmed=0
            deltas=[]
            for fold in folds:
                a,b=point(folders[MODE][fold],stage),point(folders[reference][fold],stage)
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
    completed=variants[MODE]['completed_runs']
    summary=dict(updated_at=timestamp(),seeds=[0],completed_new_runs=completed,planned_new_runs=10,
                 reused_runs=20,variants=variants,comparisons=comparisons)
    dump(out/'summary.json',summary)
    lines=['# 认知直接路径＋认知/PET 补充路径','',f'新增训练完成 {completed}/10；历史参照 20 次。',
           '种子 0，原十折、1000 轮、首次最高测试 ACC；从头端到端训练，仅最终分类 CE。',
           '最终分数 = 认知直接分类分数 + 双模态补充分数；补充分支输出层零初始化。',
           '不使用训练好的认知检查点，不冻结认知路径；直接路径的存在不保证其性能不下降。','',
           '| 模型 | 完成 | 择优 ACC | AD 召回 | 末轮 ACC |','|---|---:|---:|---:|---:|']
    for mode,v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {mode} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['best']['per_class_recall'][0]*100:.2f}% | {v['last']['acc']*100:.2f}% |")
        else:
            lines.append(f'| {mode} | 0/10 | — | — | — |')
    lines+=['','不同完成折数不能直接排名；同折配对结果见 summary.json。',
            '补充分支允许依赖认知和 PET，不能称为纯交互或纯 PET 增量；有效参数量与普通拼接不同。',
            '固定检查点的去补充分支诊断仅反映模型依赖，不等同于重新训练的消融。',
            '本轮结束后不自动追加权重搜索、条件门控或其他种子。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def component_diagnostic(plan):
    result=dict(description='Fixed selected checkpoints, full prediction versus direct component; no retraining',
                class_names=plan['class_names'],stages={})
    for stage,filename in (('best','predictions.npz'),('last','last_predictions.npz')):
        direct_metrics=[];full_metrics=[];corrected=harmed=0
        by_class=np.zeros((len(plan['class_names']),2),dtype=int)
        fold_rows=[]
        for fold in plan['folds']:
            with np.load(OUT/f'seed_0/{MODE}/fold_{fold:02d}'/filename) as saved:
                full=saved['logits'];direct=saved['unimodal_logits'][:,0];supplement=saved['pair_logits'][:,0]
                y=saved['labels']
                np.testing.assert_allclose(full,direct+supplement,rtol=2e-5,atol=2e-5)
                fm=historical.extra_scores(torch.tensor(full),torch.tensor(y))
                dm=historical.extra_scores(torch.tensor(direct),torch.tensor(y))
                full_metrics.append(fm);direct_metrics.append(dm)
                cf,cd=full.argmax(-1)==y,direct.argmax(-1)==y
                gain,harm=cf&~cd,~cf&cd
                corrected+=int(gain.sum());harmed+=int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c]+=[int((gain&(y==c)).sum()),int((harm&(y==c)).sum())]
                fold_rows.append(dict(fold=fold,full_acc=fm['acc'],direct_only_acc=dm['acc'],
                                      corrected=int(gain.sum()),harmed=int(harm.sum())))
        result['stages'][stage]=dict(
            full={key:np.mean([v[key] for v in full_metrics],axis=0).tolist()
                  for key in ('acc','macro_f1','per_class_recall')},
            direct_only={key:np.mean([v[key] for v in direct_metrics],axis=0).tolist()
                         for key in ('acc','macro_f1','per_class_recall')},
            corrected=corrected,harmed=harmed,net_corrected=corrected-harmed,
            per_class_corrected_harmed=by_class.tolist(),folds=fold_rows)
    dump(OUT/'component_diagnostic.json',result)
    return result


def prepare():
    historical.SEEDS=(0,)
    cfg,dd,data,original,hashes,baseline=historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json')
        historical.verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256']==hashes and plan['baseline_manifest']==baseline
        assert plan['seeds']==[0] and plan['planned_runs']==10
        assert plan['environment']['torch']==torch.__version__
        return cfg,dd,data,plan
    manifest,audit=previous.verify_references(cfg,dd,data,hashes)
    checks=smoke(cfg,dd,data)
    reference_plan=read(REFERENCE/'PLAN.json')
    prior_source_plan=read(previous.OUT/'PLAN.json')
    sources=tuple(dict.fromkeys(tuple(prior_source_plan['code_sha256'])+NEW_SOURCES))
    plan=dict(experiment='Direct cognitive predictor plus zero-initialized joint supplement',
        seeds=[0],folds=list(range(1,11)),modes=[MODE],epochs=1000,planned_runs=10,reused_runs=20,
        class_names=dd['Class_Names'],samples=dd['Sample_Num'],original_config=original['original_config'],
        objective='Original class-weighted CE on final summed logits only, train mask, smoothing .05',
        class_weights=dd['Label_Weight'].detach().cpu().tolist(),
        architecture='Original COG_PET encoders; 43-wide COG-only head plus 43-wide joint head; sum class logits',
        initialization='COG head identical to cognitive-only control; joint head output layer zero; other common parameters unchanged',
        training='Unchanged historical.train_arm including Adam, LR schedule, weight decay and global grad clip',
        parameter_counts=checks['counts'],priors=reference_plan['priors'],randomness=reference_plan['randomness'],
        checkpoint_selection=reference_plan['checkpoint_selection'],
        training_scope='From scratch, both paths trainable, no pretrained checkpoint or auxiliary loss; no conditional selector',
        component_names=dict(unimodal_logits=['cognitive_direct'],pair_logits=['joint_supplement']),
        data_sha256=hashes,baseline_manifest=baseline,reference_manifest=manifest,
        code_sha256={source:sha(ROOT/source) for source in sources},smoke=checks,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        caveat='Structural direct path is not a guarantee of preserved accuracy. Unequal capacity; supplement not identified as pure interaction. Exploratory test-selected single seed.',
        followup='No automatic extra arms, seeds, tuning or patient-conditioned gates')
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'PLAN.json',plan);dump(OUT/'reference_audit.json',audit)
    for source in sources:
        dest=OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/source).read_bytes())
    np.savez_compressed(OUT/'data_and_splits.npz',source_row=np.asarray(dd['Index']),
                        labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([test.cpu().numpy() for _,test in data['Mask']]))
    summarize(OUT,plan);dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp()))
    return cfg,dd,data,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,plan=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared',planned_runs=10,smoke=plan['smoke']),ensure_ascii=False),flush=True)
        return
    historical.OUT=OUT;historical.make_model=make_model;historical.summarize=summarize
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            historical.train_arm(cfg,dd,data,plan,0,fold,MODE)
        audit=historical.audit_completed(cfg,dd,data,plan)
        diagnostic=component_diagnostic(plan)
        verify_manifest(plan['reference_manifest'])
        summary=summarize(OUT,plan)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary,
                                    component_diagnostic=diagnostic))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),pid=os.getpid(),updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
