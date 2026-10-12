"""From-training no-graph control plus passive G2 training diagnostics."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
from relation_experiment import ROOT, dump, sha, timestamp, train

LEGACY = ROOT/'Result/relation_sheaf_tadpole_original'
FOLLOWUP = ROOT/'Result/relation_sheaf_g2_followup'


def g2_folder(seed, fold):
    base = LEGACY if seed == 0 else FOLLOWUP/f'seed_{seed}'
    return base/'static'/f'fold_{fold:02d}'


def summarize(out, seeds, folds, smoke):
    comparisons, diagnostics = [], []
    for seed in seeds:
        diagnostic = out/f'diagnostic_seed_{seed}'/'static/fold_01/diagnostics_summary.json'
        if diagnostic.exists():
            data = json.loads(diagnostic.read_text(encoding='utf-8'))
            first, last = data['first'], data['last']
            diagnostics.append(dict(seed=seed,fold=1,observations=data['observations'],
                first_update_ratio=first['eval_update_ratio'], final_update_ratio=last['eval_update_ratio'],
                first_value_norm=first['value_weight_before'],final_value_norm=last['value_weight_after'],
                first_output_norm=first['output_weight_before'],final_output_norm=last['output_weight_after'],
                first_value_decay_loss_ratio=first['value_decay_to_loss_grad_ratio'],
                final_value_decay_loss_ratio=last['value_decay_to_loss_grad_ratio'],
                first_output_decay_loss_ratio=first['output_decay_to_loss_grad_ratio'],
                final_output_decay_loss_ratio=last['output_decay_to_loss_grad_ratio']))
        own, ref, deltas, matched = [], [], [], []
        corrected = harmed = 0
        for fold in folds:
            folder=out/f'self_seed_{seed}'/'self_only'/f'fold_{fold:02d}'
            if not (folder/'complete.json').exists():
                continue
            result=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            own.append(result['best']['acc'])
            if smoke:
                continue
            base=g2_folder(seed,fold)
            original=json.loads((base/'complete.json').read_text(encoding='utf-8'))
            if result['seed']!=seed or original['seed']!=seed:
                raise RuntimeError('Seed mismatch in paired comparison')
            with np.load(base/'predictions.npz') as a, np.load(folder/'predictions.npz') as b:
                assert np.array_equal(a['row_index'],b['row_index']) and np.array_equal(a['labels'],b['labels'])
                ca,cb=a['logits'].argmax(-1)==a['labels'],b['logits'].argmax(-1)==b['labels']
                corrected+=int((~ca&cb).sum());harmed+=int((ca&~cb).sum())
            ref.append(original['best']['acc']);matched.append(fold)
            deltas.append(result['best']['acc']-original['best']['acc'])
        item=dict(seed=seed,completed_folds=len(own),planned_folds=len(folds),matched_folds=matched)
        if own:
            item['self_only_mean_acc']=float(np.mean(own))
        if ref:
            item.update(g2_mean_acc=float(np.mean(ref)),self_minus_g2=float(np.mean(deltas)),
                        corrected=corrected,harmed=harmed)
        comparisons.append(item)
    full=[item for item in comparisons if len(item['matched_folds'])==len(folds)]
    report=dict(updated_at=timestamp(),smoke=smoke,no_graph_comparisons=comparisons,
                completed_diagnostic_runs=diagnostics,
                comparison='No-graph minus original G2; paired seeds/folds, no ensemble.',
                caveat='Historical test-ACC selection. Diagnostic retraining on fold 1 is separate from ten-fold performance comparisons.')
    if full:
        report['complete_paired_seeds']=[item['seed'] for item in full]
        report['mean_self_minus_g2']=float(np.mean([item['self_minus_g2'] for item in full]))
    dump(out/'summary.json',report)
    lines=['# 无图对照与 G2 训练过程诊断','',
           '保持原十折、每次 1000 轮与首次最高测试 ACC 选轮；训练种子为 0、1、2。',
           '无图模型从训练开始就移除患者间传播，保留同样的自身编码、归一化与读出。',
           'G2 的诊断重训固定在第 1 折，不混入已完成的 G2 十折成绩。','']
    if smoke:
        lines.extend(['**本目录是短试运行，轮数/折数缩短，不能作为实验成绩。**',''])
    for item in comparisons:
        text=f"- 无图对照 seed {item['seed']}：完成 {item['completed_folds']}/{item['planned_folds']} 折"
        if 'self_only_mean_acc' in item:
            text+=f"，ACC {100*item['self_only_mean_acc']:.2f}%"
        if 'self_minus_g2' in item:
            text+=f"；相同折 G2 {100*item['g2_mean_acc']:.2f}%，差值 {100*item['self_minus_g2']:+.2f} 个百分点"
        lines.append(text+'。')
    lines.extend(['','已完成的 G2 第 1 折训练诊断：',''])
    for item in diagnostics:
        lines.append(f"- seed {item['seed']}：图更新/输入范数比从第一轮 {100*item['first_update_ratio']:.6g}% "
                     f"到末轮 {100*item['final_update_ratio']:.6g}%；value 权重范数 "
                     f"{item['first_value_norm']:.6g} → {item['final_value_norm']:.6g}。")
    lines.extend(['','完整变化见各诊断折的 diagnostics.csv：前 20 轮逐轮记录，此后每 10 轮记录。',
                  '梯度范数比较只能定位线索，不能单独证明权重衰减是原因。等所有计划训练完成后再判断图分支贡献。'])
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main(args):
    out=(ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume')
    seeds,folds=([0],[1]) if args.smoke else ([0,1,2],list(range(1,11)))
    if not args.smoke and not (FOLLOWUP/'COMPLETE.json').exists():
        raise RuntimeError('Complete the previous G2 experiment before this audit')
    jobs=[dict(kind='diagnostic',seed=seed,modes=['static'],folds=[1],diagnostics=True) for seed in seeds]
    jobs += [dict(kind='self',seed=seed,modes=['self_only'],folds=folds,diagnostics=False) for seed in seeds]
    sources=['relation_audit.py','relation_experiment.py','Model/relation_sheaf.py','Utils/graph_diagnostics.py']
    plan=dict(experiment='From-training no-graph ablation and passive G2 lifecycle diagnostics',
              smoke=args.smoke,epochs=3 if args.smoke else 1000,seeds=seeds,jobs=jobs,
              new_fold_runs=sum(len(job['folds']) for job in jobs),
              original_protocol_sha256=sha(LEGACY/'PROTOCOL.json'),
              parameter_policy='Graph-only parameters removed in self_only; no inactive capacity padding.',
              diagnostic_policy='Observe original G2; do not change optimizer/loss/forward. Fold 1 fixed before running.',
              code_sha256={name:sha(ROOT/name) for name in sources})
    if (out/'PLAN.json').exists() and json.loads((out/'PLAN.json').read_text(encoding='utf-8'))!=plan:
        raise RuntimeError('Plan/code changed; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'PLAN.json',plan)

    def progress():
        for name,digest in plan['code_sha256'].items():
            if sha(ROOT/name)!=digest:
                raise RuntimeError(f'Source changed during execution: {name}')
        return summarize(out,seeds,folds,args.smoke)

    progress()
    try:
        for job in jobs:
            progress()
            folder=out/f"{job['kind']}_seed_{job['seed']}"
            dump(out/'status.json',dict(state='running',updated_at=timestamp(),job=job,
                                       detail_file=str(folder/'status.json')))
            child=SimpleNamespace(config='Config/main_tadpole_3class.ini',output=str(folder),cuda=args.cuda,
                                  k=10,modes=job['modes'],folds=job['folds'],epochs=3 if args.smoke else None,
                                  training_seed=job['seed'],diagnostics=job['diagnostics'],resume=args.resume,
                                  reference_protocol=None if args.smoke else str(LEGACY/'PROTOCOL.json'))
            train(child,progress_callback=progress)
        dump(out/'COMPLETE.json',progress())
        dump(out/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Result/relation_graph_audit')
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())
