"""Pilot: remove graph-only weight decay, using three seeds on fixed fold 1."""
import argparse
import csv
import json
from types import SimpleNamespace
import numpy as np
import torch
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import ROOT, dump, sha, setup, timestamp, train

BASELINE = ROOT/'Result/relation_graph_audit'
LEGACY = ROOT/'Result/relation_sheaf_tadpole_original'


def summarize(out, seeds):
    reports = []
    for seed in seeds:
        folder = out/f'seed_{seed}/static/fold_01'
        if not (folder/'complete.json').exists():
            continue
        result = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        diagnostic = json.loads((folder/'diagnostics_summary.json').read_text(encoding='utf-8'))
        rows = list(csv.DictReader((folder/'diagnostics.csv').open(encoding='utf-8')))
        original = BASELINE/f'diagnostic_seed_{seed}/static/fold_01'
        reference = json.loads((original/'complete.json').read_text(encoding='utf-8'))
        trace = json.loads((original/'diagnostics_summary.json').read_text(encoding='utf-8'))
        item = dict(seed=seed, fold=1, epochs=result['last']['epoch'],
                    no_graph_decay_best=result['best'], original_diagnostic_best=reference['best'],
                    best_acc_difference=result['best']['acc']-reference['best']['acc'],
                    no_graph_decay_final=result['last'], original_diagnostic_final=reference['last'],
                    initial_update_ratio=diagnostic['first']['eval_update_ratio'],
                    final_update_ratio=diagnostic['last']['eval_update_ratio'],
                    original_final_update_ratio=trace['last']['eval_update_ratio'],
                    last100_observed_update_ratio_median=float(np.median([
                        float(r['eval_update_ratio']) for r in rows if int(r['epoch']) > result['last']['epoch']-100])),
                    final_value_norm=diagnostic['last']['value_weight_after'],
                    final_output_norm=diagnostic['last']['output_weight_after'],
                    final_graph_decay_norms={g:diagnostic['last'][g+'_decay_grad']
                        for g in ('value','output','maps','edges','step','graph_input_norm')})
        reports.append(item)
    summary = dict(updated_at=timestamp(),completed_seeds=len(reports),planned_seeds=len(seeds),
                   smoke=json.loads((out/'PLAN.json').read_text(encoding='utf-8'))['smoke'],
                   reports=reports,scope='Fixed fold 1 pilot, not ten-fold performance',
                   baseline='Earlier passive G2 diagnostic retraining, same fold and seeds; original ten-fold results unchanged',
                   caveat='Historical first highest TEST ACC checkpoint selection; 60 test patients shared across seeds')
    dump(out/'summary.json', summary)
    lines = ['# 图分支取消权重衰减：固定第 1 折诊断', '',
             '只将图传播专用参数的 weight_decay 从 0.0007 改为 0；输出 LayerNorm、自身编码与分类器保持 0.0007。',
             '其余结构、损失、划分、1000 轮、Adam、学习率和原测试 ACC 选轮保持不变。',
             '只覆盖预先固定的第 1 折，不作为十折成绩。旧诊断重训为参照，不替换原 G2 十折结果。', '']
    for item in reports:
        lines.append(f"- seed {item['seed']}：完成 {item['epochs']} 轮，最佳 ACC "
                     f"{100*item['no_graph_decay_best']['acc']:.2f}%（原诊断 "
                     f"{100*item['original_diagnostic_best']['acc']:.2f}%）；末轮图更新/输入范数比 "
                     f"{100*item['final_update_ratio']:.6g}%（原 {100*item['original_final_update_ratio']:.6g}%）。")
    lines += ['', '图消息维持非零不是机制成功，仍须检查预测收益及固定检查点的图/关系依赖。']
    if summary['smoke']:
        lines.insert(2,'**本目录只有 3 轮试运行，不是正式实验成绩。**')
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def checkpoint_dependence(out, seeds, cuda):
    cfg, dd, data, graphs, _ = setup(cuda)
    records = []
    for seed in seeds:
        folder = out/f'seed_{seed}/static/fold_01'
        checkpoint = torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=True)
        model = RelationSheafModel(dd,cfg,graphs,'static',training_seed=seed).to(cfg.Device).eval()
        model.load_state_dict(checkpoint['state_dict'])
        test = data['Mask'][0][1]
        y = data['Label'][test]
        with torch.no_grad():
            original = model(data['Feature'])[0][test]
            with np.load(folder/'predictions.npz') as saved:
                expected = torch.as_tensor(saved['logits'],device=cfg.Device)
                replay_error = float((original-expected).abs().max())
                if not torch.equal(original.argmax(-1),expected.argmax(-1)):
                    raise RuntimeError('Checkpoint replay changed predicted classes')
            prop = model.propagation
            original_maps = prop.map_values
            def collapsed_maps(z):
                values = original_maps(z)
                return values.mean(0,keepdim=True).expand_as(values)
            prop.map_values = collapsed_maps
            try:
                collapsed = model(data['Feature'])[0][test]
            finally:
                del prop.map_values
            step = prop.step_logit.clone()
            prop.step_logit.fill_(-100)
            try:
                no_graph = model(data['Feature'])[0][test]
            finally:
                prop.step_logit.copy_(step)
            correct = original.argmax(-1)==y
            record = dict(seed=seed,fold=1,checkpoint_epoch=checkpoint['best']['epoch'],
                          n=int(test.sum()),original_acc=float(correct.float().mean()),replay_max_error=replay_error)
            for name, logits in [('collapse_relation_maps',collapsed),('disable_graph',no_graph)]:
                changed = logits.argmax(-1)!=original.argmax(-1)
                modified_correct = logits.argmax(-1)==y
                record[name] = dict(changed=int(changed.sum()),corrected=int((~correct&modified_correct).sum()),
                                    harmed=int((correct&~modified_correct).sum()),
                                    acc=float(modified_correct.float().mean()),
                                    mean_abs_probability_change=float((logits.softmax(-1)-original.softmax(-1)).abs().mean()))
            records.append(record)
        del model
    result = dict(records=records,scope='Post-hoc dependence of selected pilot checkpoints only; not causal validation of mechanism')
    dump(out/'checkpoint_dependence.json',result)
    return result


def main(args):
    out = (ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume')
    if not (BASELINE/'COMPLETE.json').exists():
        raise RuntimeError('Complete graph audit before the decay pilot')
    seeds = [0] if args.smoke else [0,1,2]
    sources = ['relation_decay.py','relation_experiment.py','Utils/relation_optimizer.py',
               'Utils/graph_diagnostics.py','Model/relation_sheaf.py']
    plan = dict(experiment='Graph-only weight decay zero; fixed-fold pilot',smoke=args.smoke,
                seeds=seeds,folds=[1],epochs=3 if args.smoke else 1000,
                graph_weight_decay=0.0,patient_weight_decay=0.0007,
                graph_parameter_scope='propagation.* except propagation.output_norm.*',
                baseline_protocol_sha256={str(seed):sha(BASELINE/f'diagnostic_seed_{seed}/PROTOCOL.json') for seed in seeds},
                code_sha256={name:sha(ROOT/name) for name in sources},
                decision='Inspect all three seeds. Nonzero updates alone do not establish benefit; no automatic ten-fold launch.')
    if (out/'PLAN.json').exists() and json.loads((out/'PLAN.json').read_text(encoding='utf-8'))!=plan:
        raise RuntimeError('Plan/source changed; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'PLAN.json',plan)
    def progress():
        for name,digest in plan['code_sha256'].items():
            if sha(ROOT/name)!=digest:
                raise RuntimeError(f'Source changed during execution: {name}')
        return summarize(out,seeds)
    try:
        for seed in seeds:
            progress()
            folder=out/f'seed_{seed}'
            dump(out/'status.json',dict(state='running',updated_at=timestamp(),seed=seed,detail_file=str(folder/'status.json')))
            child=SimpleNamespace(config='Config/main_tadpole_3class.ini',output=str(folder),cuda=args.cuda,
                                  k=10,modes=['static'],folds=[1],epochs=3 if args.smoke else None,
                                  training_seed=seed,diagnostics=True,graph_weight_decay=0.0,resume=args.resume,
                                  reference_protocol=None if args.smoke else str(LEGACY/'PROTOCOL.json'))
            train(child,progress_callback=progress)
        checkpoint_dependence(out,seeds,args.cuda)
        dump(out/'COMPLETE.json',progress())
        dump(out/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Result/relation_graph_decay_pilot')
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())
