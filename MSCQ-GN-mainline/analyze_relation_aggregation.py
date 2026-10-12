"""Paired reports under the frozen C selection protocol."""
import csv
from pathlib import Path
import numpy as np
from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C':'原 C：六路平均', 'R1':'R1：患者共享权重', 'R2':'R2：患者独立权重'}
PARAMETERS = {'C':895096, 'R1':898236, 'R2':898236}


def summarize(out, plan):
    out=Path(out)
    assert plan['training_seed']==0 and plan['modes']==['R1','R2']
    folders={'C':REFERENCE, **{m:out/'seed_0'/m for m in plan['modes']}}
    records={m:{f:_record(base/f'fold_{f:02d}',f,1000) for f in plan['folds']
                if (base/f'fold_{f:02d}/complete.json').exists()} for m,base in folders.items()}
    for m,values in records.items():
        for r in values.values():
            assert r['parameters']==PARAMETERS[m]
            assert r['record']['mode']==('content' if m=='C' else m)
    variants={m:_aggregate(v,plan['folds']) for m,v in records.items()}
    pairs=[_pair(a,b,records[a],records[b],plan['class_names'])
           for a,b in (('C','R1'),('C','R2'),('R1','R2'))]
    completed=sum(len(records[m]) for m in plan['modes'])
    result=dict(completed_new_runs=completed,planned_new_runs=20,all_new_complete=completed==20,
                training_seed=0,variants=variants,matched_comparisons=pairs,
                aggregation='Equal fold means; paired comparisons use matching completed folds only.',
                caveats=['Original first-best TEST ACC, seed0 only.',
                         'R1 and R2 have identical parameter counts; both add 3140 parameters to C.',
                         'R1 softmax(mean scores) differs from posthoc mean(R2 softmax scores).',
                         'Fixed-checkpoint interventions are dependency diagnostics, not retrained ablations.'])
    _dump(out/'summary.json',result)
    lines=['# C 关系来源汇总实验：seed 0 原十折','',f'新增训练完成 {completed}/20；复用原 C 的10折。',
        '每折1000轮，保留原首次最高测试ACC选轮；只改变六路传播更新的汇总。','',
        '| 模型 | 完成折数 | 最佳ACC | 宏F1 | 第1000轮ACC | 末100轮ACC | 参数量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,v in variants.items():
        if v['completed_folds']!=10:
            lines.append(f"| {NAMES[m]} | {v['completed_folds']}/10 | 待完成 | — | — | — | {PARAMETERS[m]} |")
        else:
            lines.append(f"| {NAMES[m]} | 10/10 | {100*v['best']['acc']:.4f}% | {100*v['best']['macro_f1']:.4f}% | "
                         f"{100*v['last']['acc']:.4f}% | {100*v['mean_last100_acc']:.4f}% | {PARAMETERS[m]} |")
    lines += ['', '以下配对仅覆盖共同完成折，不将部分折当作完整十折结论：','']
    for p in pairs:
        if p['matched_folds']:
            b=p['stages']['best']
            lines.append(f"- {NAMES[p['candidate']]} − {NAMES[p['reference']]}，{len(p['matched_folds'])}折："
                         f"ACC {100*b['mean_acc_delta']:+.4f}个百分点；纠正{b['corrected']}，新增错误{b['harmed']}，净{b['net']:+d}；"
                         f"{b['fold_wins']}胜/{b['fold_ties']}平/{b['fold_losses']}负。")
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def write_report(out, plan, audit):
    out=Path(out); result=summarize(out,plan)
    assert result['all_new_complete'] and audit['passed'] and audit['verified_checkpoints']==40
    r1,r2,individual=[p['stages']['best']['mean_acc_delta'] for p in result['matched_comparisons']]
    if r2>1e-12 and individual>1e-12:
        conclusion='R2最佳ACC同时高于C和R1，支持后续检验患者条件关系汇总；单种子不构成稳定优势或创新性证明。'
    elif r1>1e-12 and individual<=1e-12:
        conclusion='共享关系权重高于C，患者独立权重没有超过共享权重；本轮不支持强调个体化选择。'
    elif r2<=1e-12 and individual>1e-12:
        conclusion='R2高于R1但没有超过C，暂不支持替换原模型。'
    else:
        conclusion='本轮未建立新汇总方式优于C的最佳ACC优势，暂不扩展该版本。'
    lines=['# C 关系来源汇总：结果分析','',conclusion,'',
           (out/'进度与结果.md').read_text(encoding='utf-8'),
           '20次新增训练及40个最佳/末轮检查点已复核；原C的10个最佳检查点在准备阶段重放。',
           'R1、R2参数量相同，共用参数、评分网络初始化以及CPU/CUDA随机流对齐。新增3140参数按原图专用参数策略不衰减。',
           '两组都保留六张固定候选图、原边权、发送/接收变换、传播算子、双分支和相加分类，只替换六路更新平均的位置。',
           'R1先对全队列评分取平均再softmax；R2直接按患者softmax。不使用标签计算权重。','',
           '| 模型/检查点 | 正常ACC | 替换为患者平均权重 | 替换为等权 | 患者间权重标准差均值 |',
           '|---|---:|---:|---:|---:|']
    for mode in ('R1','R2'):
        for stage in ('best','last'):
            rows=[r for r in audit['records'] if r['mode']==mode and r['stage']==stage]
            means={k:np.mean([r['interventions'][k]['metrics']['acc'] for r in rows])
                   for k in ('normal','patient_mean','uniform')}
            variability=np.mean([r['diagnostics']['patient_weight_std'] for r in rows])
            lines.append(f"| {mode}/{stage} | {100*means['normal']:.4f}% | {100*means['patient_mean']:.4f}% | {100*means['uniform']:.4f}% | {variability:.6f} |")
    lines += ['', '患者平均干预对已训练模型的softmax后权重取全队列平均，不等于从头训练的R1。',
              '干预固定参数及检查点，不重新选轮。非均匀权重或预测依赖不能单独证明有益关系、因果机制或分类收益。',
              '每个检查点的完整权重以weights_best.npz/weights_last.npz保存，轴顺序为患者、关系来源、内容模态。',
              '保留原测试集选轮规则，仅seed0；未增加独立测试，也不自动追加训练、种子或调参。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'paired_folds.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['comparison','stage','fold','reference_acc','candidate_acc','acc_delta','corrected','harmed','net','patients']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for p in result['matched_comparisons']:
            for stage,v in p['stages'].items():
                for r in v['folds']: writer.writerow(dict(comparison=p['comparison'],stage=stage,**r))
    return result
