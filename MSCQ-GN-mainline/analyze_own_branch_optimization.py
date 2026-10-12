"""Paired reports for the 2x2 own-gate / token-FFN experiment."""
import csv
from pathlib import Path
import numpy as np
from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C', 'NO_GATE': '取消自身门控', 'FFN': '增加逐模态残差 MLP',
         'NO_GATE_FFN': '取消自身门控＋残差 MLP'}
PARAMETERS = {'C': 895096, 'NO_GATE': 895096, 'FFN': 911672, 'NO_GATE_FFN': 911672}
PAIRS = (('C', 'NO_GATE'), ('C', 'FFN'), ('C', 'NO_GATE_FFN'),
         ('NO_GATE', 'NO_GATE_FFN'), ('FFN', 'NO_GATE_FFN'))


def summarize(out, plan):
    out = Path(out)
    assert plan['modes'] == ['NO_GATE', 'FFN', 'NO_GATE_FFN']
    folders = {'C': REFERENCE, **{m: out/'seed_0'/m for m in plan['modes']}}
    records = {m: {f: _record(base/f'fold_{f:02d}', f, 1000) for f in plan['folds']
                   if (base/f'fold_{f:02d}'/'complete.json').exists()} for m, base in folders.items()}
    for mode, values in records.items():
        for r in values.values():
            assert r['parameters'] == PARAMETERS[mode]
            assert r['record']['mode'] == ('content' if mode == 'C' else mode)
    variants = {m: _aggregate(v, plan['folds']) for m, v in records.items()}
    pairs = [_pair(a, b, records[a], records[b], plan['class_names']) for a, b in PAIRS]
    common = sorted(set.intersection(*(set(records[m]) for m in NAMES)))
    interaction = {}
    for stage in ('best', 'last'):
        rows = [dict(fold=f, acc_interaction=
                     records['NO_GATE_FFN'][f]['record'][stage]['acc']
                     - records['NO_GATE'][f]['record'][stage]['acc']
                     - records['FFN'][f]['record'][stage]['acc']
                     + records['C'][f]['record'][stage]['acc']) for f in common]
        interaction[stage] = dict(folds=rows,
            mean_acc_interaction=float(np.mean([r['acc_interaction'] for r in rows])) if rows else None)
    completed = sum(len(records[m]) for m in plan['modes'])
    result = dict(completed_new_runs=completed, planned_new_runs=30, all_new_complete=completed == 30,
        training_seed=0, variants=variants, matched_comparisons=pairs, interaction=interaction,
        aggregation='Equal-fold means; prediction error counts use matched patient rows.',
        caveats=['Original first-maximum TEST ACC selection; one training seed.',
                 'Own gate removal changes initial activations and optimization; no prior absorption into projections.',
                 'Population gate, input gain/bias and differential input noise remain enabled.',
                 'FFN adds capacity as well as a new residual processing step; no separate capacity control this round.',
                 'Own-gate parameters remain active in the population path, so NO_GATE removes no parameters.'])
    _dump(out/'summary.json', result)
    lines = ['# C 自身分支优化：进度与结果', '',
        f'新增训练完成 {completed}/30；原 C 十折经检查点重放后复用。',
        'TAD AD/CN/SMCI，seed0，原十折，每折1000轮，首次最高测试ACC选点。', '',
        '未完成十折的版本不显示整体成绩；配对比较仅使用共同完成折。', '',
        '| 版本 | 完成折数 | 最佳ACC | 宏F1 | 第1000轮ACC | 末100轮ACC | 参数量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m, v in variants.items():
        if v['completed_folds'] != 10:
            lines.append(f"| {NAMES[m]} | {v['completed_folds']}/10 | 待完成 | — | — | — | {PARAMETERS[m]} |")
        else:
            lines.append(f"| {NAMES[m]} | 10/10 | {100*v['best']['acc']:.4f}% | "
                f"{100*v['best']['macro_f1']:.4f}% | {100*v['last']['acc']:.4f}% | "
                f"{100*v['mean_last100_acc']:.4f}% | {PARAMETERS[m]} |")
    lines += ['', '共同完成折上的配对结果（不能把部分折推广到十折）：', '']
    for p in pairs:
        if p['matched_folds']:
            b = p['stages']['best']
            lines.append(f"- {NAMES[p['candidate']]} − {NAMES[p['reference']]}，{len(p['matched_folds'])}折："
                f"ACC {100*b['mean_acc_delta']:+.4f}个百分点；纠正{b['corrected']}、新增错误{b['harmed']}，"
                f"净{b['net']:+d}；{b['fold_wins']}胜/{b['fold_ties']}平/{b['fold_losses']}负。")
    lines += ['', '只有自身分支不再乘门控；群体分支继续使用原门控。残差MLP在第二层注意力后、'
        '六模态展开前执行，64→128→64，共享于模态和患者，最后一层零初始化。',
        '不增加损失、不额外调参、不追加其他种子。FFN有效也不能单独排除新增容量的作用。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (out/'paired_folds.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['comparison','stage','fold','reference_acc','candidate_acc','acc_delta',
                  'corrected','harmed','net','patients']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for p in pairs:
            for stage, values in p['stages'].items():
                for r in values['folds']: writer.writerow(dict(comparison=p['comparison'], stage=stage, **r))
    return result


def write_report(out, plan, audit):
    out = Path(out)
    result = summarize(out, plan)
    assert result['all_new_complete'] and audit['passed'] and audit['verified_checkpoints'] == 60
    ranked = sorted(NAMES, key=lambda m: result['variants'][m]['best']['acc'], reverse=True)
    top = ranked[0]
    if result['variants'][top]['best']['acc'] > result['variants']['C']['best']['acc']+1e-12:
        conclusion = f'本轮主指标最高为“{NAMES[top]}”，可作为后续复验候选；单种子结果不代表稳定收益。'
    else:
        conclusion = '本轮新增版本没有超过原 C 的最佳 ACC，当前证据不足以替换原自身分支。'
    lines = ['# C 自身分支优化：结果分析', '', conclusion, '',
        (out/'进度与结果.md').read_text(encoding='utf-8'), '',
        '30次新增训练、60个最佳/末轮检查点均复核；原 C 的10个最佳检查点在准备阶段重放。',
        '各类别召回率、混淆矩阵、配对病例及两因素交互描述见 summary.json / paired_folds.csv。',
        '所有版本从头训练；原 C 权重仅用于核查，不作为新训练起点。',
        '取消自身门控可能改变共享门控在联合训练中的梯度；群体分支结构保留不意味着训练后参数相同。',
        '两因素交互是描述性结果，不能据此宣称因果机制或方法创新。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
