"""Checked, same-seed/fold reports for the branch-fusion experiment."""
import csv
from pathlib import Path
import numpy as np
from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C':'原 C：相加', 'concat':'拼接融合', 'wide':'相加＋容量对照'}
PARAMETERS = {'C':895096, 'concat':899704, 'wide':899696}


def summarize(out, plan):
    out=Path(out)
    assert plan['training_seed']==0 and plan['modes']==['concat','wide']
    folders={'C':REFERENCE, **{m:out/'seed_0'/m for m in plan['modes']}}
    records={m:{f:_record(base/f'fold_{f:02d}',f,1000) for f in plan['folds']
                if (base/f'fold_{f:02d}/complete.json').exists()} for m,base in folders.items()}
    for m, values in records.items():
        for f,r in values.items():
            assert r['parameters']==PARAMETERS[m]
            assert r['record']['mode']==('content' if m=='C' else m)
    variants={m:_aggregate(v,plan['folds']) for m,v in records.items()}
    pairs=[_pair(a,b,records[a],records[b],plan['class_names'])
           for a,b in (('C','concat'),('C','wide'),('wide','concat'))]
    completed=sum(len(records[m]) for m in plan['modes'])
    result=dict(completed_new_runs=completed,planned_new_runs=20,all_new_complete=completed==20,
        training_seed=0,variants=variants,matched_comparisons=pairs,
        aggregation='Equal mean of ten folds; comparisons only use matching completed folds.',
        caveats=['Single training seed, original first-best TEST ACC protocol.',
                 'Concat and wide differ by 8 active parameters; width and optimization also differ.',
                 'Branch zeroing is a fixed-checkpoint dependency diagnostic, not retrained ablation.'])
    _dump(out/'summary.json',result)
    lines=['# C 双分支融合：seed 0 原十折','',f'新增训练完成 {completed}/20；复用原 C 的10折。',
        '每折1000轮，原数据、候选图、传播、分支、辅助损失、优化器设置和首次最高测试ACC选轮均保留。','',
        '| 模型 | 完成折数 | 最佳ACC | 宏F1 | 第1000轮ACC | 末100轮ACC | 参数量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,v in variants.items():
        if v['completed_folds']!=10:
            lines.append(f"| {NAMES[m]} | {v['completed_folds']}/10 | 待完成 | — | — | — | {PARAMETERS[m]} |")
        else:
            lines.append(f"| {NAMES[m]} | 10/10 | {100*v['best']['acc']:.4f}% | {100*v['best']['macro_f1']:.4f}% | "
                f"{100*v['last']['acc']:.4f}% | {100*v['mean_last100_acc']:.4f}% | {PARAMETERS[m]} |")
    lines += ['', '以下仅比较共同完成折；部分折不能当作完整十折结论：','']
    for p in pairs:
        if p['matched_folds']:
            b=p['stages']['best']
            lines.append(f"- {NAMES[p['candidate']]} − {NAMES[p['reference']]}，{len(p['matched_folds'])}折："
                f"ACC {100*b['mean_acc_delta']:+.4f}个百分点；纠正{b['corrected']}，新增错误{b['harmed']}，净{b['net']:+d}；"
                f"{b['fold_wins']}胜/{b['fold_ties']}平/{b['fold_losses']}负。")
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def write_report(out, plan, audit):
    out=Path(out);result=summarize(out,plan)
    assert result['all_new_complete'] and audit['passed'] and audit['verified_checkpoints']==40
    a=result['matched_comparisons'][0]['stages']['best']['mean_acc_delta']
    b=result['matched_comparisons'][2]['stages']['best']['mean_acc_delta']
    if a>1e-12 and b>1e-12:
        conclusion='本轮拼接的最佳ACC高于原C和近等容量相加对照，可进入后续重复验证；单种子尚不能确立稳定机制优势。'
    elif a>1e-12:
        conclusion='拼接高于原C，但没有同时超过近等容量相加对照，尚不足以把提升归因于分支来源保留。'
    else:
        conclusion='本轮拼接没有超过原C的最佳ACC，尚未支持直接用该拼接版本替代相加。'
    lines=['# C 双分支融合：结果分析','',conclusion,'',
        (out/'进度与结果.md').read_text(encoding='utf-8'),
        '新增20次训练与40个最佳/末轮检查点已复核；原C的10个最佳检查点在准备阶段重放。',
        '原C没有last.pt，因此不报告原C末轮分支干预。',
        '拼接头192→48→3，相加容量对照96→94→3；总参数分别899704和899696，差8个。',
        '拼接的两个初始投影均复制随机初始化C的W；容量对照前48单元继承C，新单元输出权重从零开始、随后参与训练。',
        '所有模型从头训练，未使用已训练C权重热启动；共用模块的初始化与随机数流对齐。',
        '加宽头新增46单元使用独立同概率dropout随机流，避免改变原48单元及主干的随机流。','',
        '## 固定检查点分支依赖','',
        '| 模型/检查点 | 完整ACC | 关闭自身分支 | 关闭上下文增强分支 |',
        '|---|---:|---:|---:|']
    for mode in ('C','concat','wide'):
        values=audit['reference_interventions'] if mode=='C' else [r for r in audit['records'] if r['mode']==mode]
        for stage in ('best','last'):
            selected=[r['interventions'] for r in values if r['stage']==stage]
            if not selected: continue
            means={k:float(np.mean([v[k]['metrics']['acc'] for v in selected]))
                   for k in ('both','semantic_only','own_only')}
            lines.append(f"| {NAMES[mode]}/{stage} | {100*means['both']:.4f}% | {100*means['semantic_only']:.4f}% | {100*means['own_only']:.4f}% |")
    lines += ['', '干预只将最终融合前的一条向量置零，保留全部已训练权重、另一分支和分类器偏置；没有重新训练或重新选轮。',
        'semantic分支同时包含自身与邻居信息；own分支是患者自身多模态编码器。依赖、向量范数和投影差异均不等于因果或独有互补信息。',
        '本轮仅seed0、原十折和原测试选轮，未新增独立测试；不自动追加种子、调参或新模型。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'paired_folds.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['comparison','stage','fold','reference_acc','candidate_acc','acc_delta','corrected','harmed','net','patients']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for p in result['matched_comparisons']:
            for stage,v in p['stages'].items():
                for r in v['folds']:writer.writerow(dict(comparison=p['comparison'],stage=stage,**r))
    return result
