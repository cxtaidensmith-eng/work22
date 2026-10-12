"""Ten-fold G2 vs equal-capacity content control with graph-only decay zero."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import ROOT, dump, sha, setup, timestamp, train

MODES = ('content', 'static')
NAMES = {'content':'C：仅按内容模态变换', 'static':'G2：按关系来源与内容模态变换'}
LEGACY = ROOT/'Result/relation_sheaf_tadpole_original'
PILOT = ROOT/'Result/relation_graph_decay_pilot'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def folder_for(out, seed, mode, fold):
    return out/f'seed_{seed}'/mode/f'fold_{fold:02d}'


def aggregate(out, seeds, folds, smoke):
    records, variants, paired = {}, {}, []
    for mode in MODES:
        per_seed, count = {}, 0
        for seed in seeds:
            current = {}
            for fold in folds:
                path = folder_for(out,seed,mode,fold)/'complete.json'
                if not path.exists():
                    continue
                record = read(path)
                if (record['seed'],record['fold'],record['mode'])!=(seed,fold,mode):
                    raise RuntimeError(f'Result identity mismatch: {path}')
                current[fold] = record
            records[mode,seed] = current
            count += len(current)
            if len(current)==len(folds):
                per_seed[str(seed)] = {metric:float(np.mean([r['best'][metric] for r in current.values()]))
                                      for metric in ('acc','macro_f1','auc')}
                per_seed[str(seed)]['final_acc'] = float(np.mean([r['last']['acc'] for r in current.values()]))
        item = dict(completed_fold_runs=count,planned_fold_runs=len(seeds)*len(folds),seed_means=per_seed)
        if per_seed:
            item['mean_acc_complete_seeds'] = float(np.mean([r['acc'] for r in per_seed.values()]))
            if len(per_seed)>1:
                item['std_seed_mean_acc'] = float(np.std([r['acc'] for r in per_seed.values()],ddof=1))
        variants[mode] = item
    for seed in seeds:
        common = sorted(set(records['content',seed]) & set(records['static',seed]))
        if not common:
            continue
        details = []
        for fold in common:
            with np.load(folder_for(out,seed,'content',fold)/'predictions.npz') as c, \
                 np.load(folder_for(out,seed,'static',fold)/'predictions.npz') as g:
                if not (np.array_equal(c['labels'],g['labels']) and np.array_equal(c['row_index'],g['row_index'])):
                    raise RuntimeError('Paired predictions use different patients')
                cc,cg = c['logits'].argmax(-1)==c['labels'],g['logits'].argmax(-1)==g['labels']
                for mode,correct in [('content',cc),('static',cg)]:
                    if abs(float(correct.mean())-records[mode,seed][fold]['best']['acc'])>1e-12:
                        raise RuntimeError('Saved prediction ACC does not match record')
                details.append(dict(fold=fold,content_acc=float(cc.mean()),g2_acc=float(cg.mean()),
                                    delta=float(cg.mean()-cc.mean()),n=len(cc),
                                    corrected=int((~cc&cg).sum()),harmed=int((cc&~cg).sum())))
        deltas = [r['delta'] for r in details]
        # Fold 1 informed the optimization pilot. Report folds 2-10 separately too.
        remaining = [r for r in details if r['fold']!=1]
        paired.append(dict(seed=seed,matched_folds=common,folds=details,
                           mean_delta=float(np.mean(deltas)),
                           content_mean=float(np.mean([r['content_acc'] for r in details])),
                           g2_mean=float(np.mean([r['g2_acc'] for r in details])),
                           wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas),
                           corrected=sum(r['corrected'] for r in details),harmed=sum(r['harmed'] for r in details),
                           mean_delta_excluding_pilot_fold=float(np.mean([r['delta'] for r in remaining])) if remaining else None))
    full = [r for r in paired if len(r['matched_folds'])==len(folds)]
    result = dict(updated_at=timestamp(),smoke=smoke,variants=variants,paired=paired,
                  complete_paired_seeds=[r['seed'] for r in full],
                  mean_delta_complete_paired_seeds=float(np.mean([r['mean_delta'] for r in full])) if full else None,
                  aggregation='Mean of fold ACC within each seed, then mean seeds; no ensemble or best-seed selection',
                  caveat='Historical test-ACC checkpoint selection; repeated seeds share patients; fold 1 informed the decay pilot')
    dump(out/'summary.json',result)
    lines = ['# 关系来源是否有额外价值：实验进度','',
             'G2 与仅内容对照均取消图专用参数权重衰减；自身编码、输出归一化和读出仍为 0.0007。',
             '两个版本均为 895,096 个有效参数；候选图、标量边权机制、传播层数和分类后端一致。',
             '仅内容对照保留边权中的关系证据，移除的是内容变换对关系来源的条件依赖。',
             '原划分、1000 轮、原损失/Adam/学习率、首次最高测试 ACC 选轮不变。','']
    if smoke:
        lines += ['**当前为短试运行，不能作为正式成绩。**','']
    for mode,info in variants.items():
        lines.append(f"- {NAMES[mode]}：完成 {info['completed_fold_runs']}/{info['planned_fold_runs']} 个折次。")
    lines += ['','相同种子、相同已完成折上的配对比较；正值表示 G2 更高：','']
    for row in paired:
        lines.append(f"- seed {row['seed']}，{len(row['matched_folds'])}/{len(folds)} 折："
                     f"C {100*row['content_mean']:.2f}%，G2 {100*row['g2_mean']:.2f}%，"
                     f"差值 {100*row['mean_delta']:+.2f} 个百分点；"
                     f"{row['wins']} 胜/{row['ties']} 平/{row['losses']} 负。")
    lines += ['','完整结论等待三个种子全部完成；同时单列未参与优化诊断的第 2–10 折。',
              '每折只保留一个最佳检查点、小型预测文件和诊断日志；训练结束后自动进行检查点依赖分析。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def dependencies(out,seeds,folds,cuda):
    cfg,dd,data,graphs,_ = setup(cuda)
    results = []
    for seed in seeds:
        for fold in folds:
            for mode in MODES:
                folder = folder_for(out,seed,mode,fold)
                checkpoint = torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=True)
                model = RelationSheafModel(dd,cfg,graphs,mode,training_seed=seed).to(cfg.Device).eval()
                model.load_state_dict(checkpoint['state_dict'])
                test = data['Mask'][fold-1][1]
                y = data['Label'][test]
                with torch.no_grad():
                    logits = model(data['Feature'])[0][test]
                    with np.load(folder/'predictions.npz') as saved:
                        expected = torch.as_tensor(saved['logits'],device=cfg.Device)
                    if not torch.equal(logits.argmax(-1),expected.argmax(-1)):
                        raise RuntimeError('Checkpoint replay changed classes')
                    prop=model.propagation
                    step=prop.step_logit.clone()
                    prop.step_logit.fill_(-100)
                    try:
                        no_graph=model(data['Feature'])[0][test]
                    finally:
                        prop.step_logit.copy_(step)
                    variants={'disable_graph':no_graph}
                    if mode=='static':
                        original_maps=prop.map_values
                        def collapsed(z):
                            values=original_maps(z)
                            return values.mean(0,keepdim=True).expand_as(values)
                        prop.map_values=collapsed
                        try:
                            variants['collapse_relation_maps']=model(data['Feature'])[0][test]
                        finally:
                            del prop.map_values
                    correct=logits.argmax(-1)==y
                    record=dict(seed=seed,fold=fold,mode=mode,n=int(test.sum()),
                                original_acc=float(correct.double().mean()),
                                replay_max_error=float((logits-expected).abs().max()))
                    for name,modified in variants.items():
                        after=modified.argmax(-1)==y
                        record[name]=dict(acc=float(after.double().mean()),
                            changed=int((modified.argmax(-1)!=logits.argmax(-1)).sum()),
                            corrected=int((~correct&after).sum()),harmed=int((correct&~after).sum()),
                            mean_abs_probability_change=float((modified.softmax(-1)-logits.softmax(-1)).abs().mean()))
                    results.append(record)
                del model
                dump(out/'checkpoint_dependence.json',dict(records=results,completed=len(results),
                     planned=len(seeds)*len(folds)*len(MODES),caveat='Fixed-checkpoint dependence only; not a retrained performance baseline'))
    return results


def verify_and_report(out,seeds,folds,smoke):
    summary=aggregate(out,seeds,folds,smoke)
    checks=[]
    for seed in seeds:
        parent=out/f'seed_{seed}'
        baseline=read(PILOT/f'seed_{seed}/PROTOCOL.json')
        protocol=read(parent/'PROTOCOL.json')
        for key in ('original_config','candidate_hash','data_sha256','split_seed','training_seed','checkpoint_selection'):
            if baseline[key]!=protocol[key]:
                raise RuntimeError(f'Protocol mismatch: {key}')
        with np.load(parent/'splits_and_graphs.npz') as a,np.load(PILOT/f'seed_{seed}/splits_and_graphs.npz') as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                raise RuntimeError('Cohort/splits/graphs changed')
        for mode in MODES:
            for fold in folds:
                folder=folder_for(out,seed,mode,fold)
                records=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8')))
                expected_epochs=3 if smoke else 1000
                if [int(r['epoch']) for r in records]!=list(range(1,expected_epochs+1)):
                    raise RuntimeError('Unexpected epoch sequence')
                acc=[float(r['test_acc']) for r in records]
                result=read(folder/'complete.json')
                if result['best']['epoch']!=acc.index(max(acc))+1:
                    raise RuntimeError('Checkpoint is not first maximum test ACC')
                groups=read(folder/'optimizer_groups.json')
                if [g['weight_decay'] for g in groups]!=[.0007,0.] or [g['parameters'] for g in groups]!=[874091,21005]:
                    raise RuntimeError('Unexpected optimizer parameter groups')
                diagnostic=read(folder/'diagnostics_summary.json')
                checks.append(dict(seed=seed,fold=fold,mode=mode,
                    final_update_ratio=diagnostic['last']['eval_update_ratio'],
                    last100_acc=float(np.mean(acc[-100:]))))
    dependence=read(out/'checkpoint_dependence.json')
    report=dict(summary=summary,checks=checks,checkpoint_dependence=dependence,
                verified_epochs_selection_groups_splits=True,
                limits=['Historical test-selected estimates','Same patients repeated across seeds',
                        'Fold 1 informed optimization choice','Post-hoc dependence does not establish mechanism superiority'])
    dump(out/'analysis.json',report)
    if smoke:
        return report
    c=summary['variants']['content']['mean_acc_complete_seeds']
    g=summary['variants']['static']['mean_acc_complete_seeds']
    differences=[r['mean_delta'] for r in summary['paired']]
    if all(d>0 for d in differences):
        conclusion='三个种子的平均 ACC 均支持 G2，但仍需结合逐折幅度、类别表现与依赖分析评价，不能直接宣称统计显著或独立验证。'
    elif all(d<=0 for d in differences):
        conclusion='三个种子均未显示 G2 的额外平均收益；当前证据不支持关系来源条件优于仅内容变换。'
    else:
        conclusion='不同种子的结果方向不一致，尚未获得关系来源条件带来稳定额外收益的证据。'
    lines=['# 关系来源条件与仅内容对照：完整实验结果','',conclusion,'',
           'TADPOLE AD/CN/SMCI；三个种子各跑原十折，每次 1000 轮，首次最高测试 ACC 选轮。两种模型均完整重训。',
           '两者均取消图专用参数衰减、自身参数保持 0.0007，总参数量均为 895,096。','',
           f'三个种子的十折均值：C **{100*c:.4f}%**，G2 **{100*g:.4f}%**，差值 **{100*(g-c):+.4f} 个百分点**。','']
    for r in summary['paired']:
        lines.append(f"- seed {r['seed']}：C {100*r['content_mean']:.4f}%，G2 {100*r['g2_mean']:.4f}%，"
                     f"差值 {100*r['mean_delta']:+.4f} 个百分点；{r['wins']} 胜、{r['ties']} 平、{r['losses']} 负；"
                     f"纠正 {r['corrected']} 人，新增错误 {r['harmed']} 人。")
        lines.append(f"  单列第 2–10 折，差值 {100*r['mean_delta_excluding_pilot_fold']:+.4f} 个百分点。")
    lines+=['','两组都保留边权中的关系信息；核心比较只检验内容变换额外依赖关系来源是否有用。','',
            '以下为最佳检查点的事后干预，不能视为独立重训的消融成绩：','']
    for mode in MODES:
        selected=[r for r in dependence['records'] if r['mode']==mode]
        delta=float(np.mean([r['disable_graph']['acc']-r['original_acc'] for r in selected]))
        changed=sum(r['disable_graph']['changed'] for r in selected)
        lines.append(f'- {NAMES[mode]}：关闭图消息后的折均 ACC 变化 {100*delta:+.4f} 个百分点；改变 {changed} 次测试预测。')
        if mode=='static':
            delta=float(np.mean([r['collapse_relation_maps']['acc']-r['original_acc'] for r in selected]))
            lines.append(f'  合并关系来源变换后的折均 ACC 变化 {100*delta:+.4f} 个百分点。')
    lines+=['','所有 60 次训练的轮数、首次最高测试 ACC 选轮、参数组、数据划分和候选图已核对。',
            '三个种子复用同一批 598 位患者，预测变化次数不能当成不同患者数。第 1 折曾用于优化诊断，因此补充列出第 2–10 折；后者仍沿用历史测试选轮，不能当作独立确认性评估。',
            '完整逐折指标及诊断见 analysis.json，运行来源和固定计划见 PLAN.json。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main(args):
    out=(ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume')
    seeds,folds=([0,1],[1]) if args.smoke else ([0,1,2],list(range(1,11)))
    if not (PILOT/'COMPLETE.json').exists():
        raise RuntimeError('Complete the decay pilot first')
    sources=sorted(set(read(LEGACY/'PROTOCOL.json')['code_sha256']) | {
        'relation_mechanism.py','relation_experiment.py','Model/relation_sheaf.py',
        'Utils/relation_optimizer.py','Utils/graph_diagnostics.py'})
    plan=dict(experiment='G2 vs equal-capacity content control, graph-only decay zero',
              smoke=args.smoke,seeds=seeds,folds=folds,modes=list(MODES),epochs=3 if args.smoke else 1000,
              new_fold_runs=len(seeds)*len(folds)*len(MODES),reused_fold_runs=0,
              graph_weight_decay=0.,patient_weight_decay=.0007,
              primary='Mean paired G2-minus-content ACC across all ten folds and three seeds',
              secondary='Per-seed/fold results; folds 2-10 separately; macro F1/AUC; final/late performance; checkpoint dependence',
              decision='No interim model/config changes; report all planned seeds, including negative results',
              code_sha256={name:sha(ROOT/name) for name in sources},
              original_protocol_sha256=sha(LEGACY/'PROTOCOL.json'),
              pilot_protocol_sha256={str(s):sha(PILOT/f'seed_{s}/PROTOCOL.json') for s in seeds})
    if (out/'PLAN.json').exists() and read(out/'PLAN.json')!=plan:
        raise RuntimeError('Plan/code changed; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'PLAN.json',plan)
    def progress():
        for source,digest in plan['code_sha256'].items():
            if sha(ROOT/source)!=digest:
                raise RuntimeError(f'Source changed during run: {source}')
        return aggregate(out,seeds,folds,args.smoke)
    try:
        for seed in seeds:
            progress()
            folder=out/f'seed_{seed}'
            dump(out/'status.json',dict(state='running',phase='training',seed=seed,updated_at=timestamp(),
                                       detail_file=str(folder/'status.json')))
            child=SimpleNamespace(config='Config/main_tadpole_3class.ini',output=str(folder),cuda=args.cuda,k=10,
                                  modes=list(MODES),folds=folds,epochs=3 if args.smoke else None,
                                  training_seed=seed,diagnostics=True,graph_weight_decay=0.,resume=args.resume,
                                  reference_protocol=None if args.smoke else str(LEGACY/'PROTOCOL.json'))
            train(child,progress_callback=progress)
        progress()
        dump(out/'status.json',dict(state='running',phase='checkpoint_dependence',updated_at=timestamp()))
        dependencies(out,seeds,folds,args.cuda)
        verify_and_report(out,seeds,folds,args.smoke)
        dump(out/'COMPLETE.json',progress())
        dump(out/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Result/relation_mechanism_nodecay')
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())
