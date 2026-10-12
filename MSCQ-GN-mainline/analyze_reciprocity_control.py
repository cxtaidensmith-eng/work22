"""Validated, paired reporting for the preregistered reciprocity experiment."""
from pathlib import Path
from analyze_joint_evidence_routing import _aggregate, _dump, _pair, _record

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C', 'REAL': '真实互选标记', 'SHUFFLED': '随机标记对照'}


def summarize(out, plan):
    out = Path(out)
    folds = list(plan['folds'])
    assert folds == list(range(1, 11)) and plan['modes'] == ['REAL', 'SHUFFLED']
    assert plan['epochs'] == 1000 and plan['training_seed'] == 0
    directories = {'C': REFERENCE, **{m: out/'seed_0'/m for m in plan['modes']}}
    records = {m: {f: _record(base/f'fold_{f:02d}', f, 1000) for f in folds
                   if (base/f'fold_{f:02d}/complete.json').exists()} for m, base in directories.items()}
    for m, rows in records.items():
        for row in rows.values():
            assert row['record']['mode'] == ('content' if m == 'C' else m)
            assert row['parameters'] == plan['parameters'][m]
    variants = {m: _aggregate(rows, folds) for m, rows in records.items()}
    pairs = [_pair(a, b, records[a], records[b], plan['class_names'])
             for a, b in (('C', 'REAL'), ('C', 'SHUFFLED'), ('SHUFFLED', 'REAL'))]
    count = sum(len(records[m]) for m in plan['modes'])
    complete = count == 20
    caveats = [
        '沿用首次最高测试 ACC 选轮；单种子探索，不是独立验证的泛化估计。',
        '随机标记按每种关系的无向边打乱，保持总数和反向对称，不保持逐患者互选度或相似度分布。',
        '一个固定随机标记实例；不能替代多次随机化或多种子验证。',
        '只研究互选/单向状态的内容条件边权调节，没有实现非对称方向传播。',
        '无关系共享系数消融，因此不能单独确立内容条件化的必要性。',
        '原 C 复用经过代码、数据、划分、图和最佳权重重放核验的十折结果。',
    ]
    conclusion = '十折尚未全部完成，不根据部分折作方法优劣结论。'
    if complete:
        real_c = pairs[0]['stages']['best']['mean_acc_delta']
        real_random = pairs[2]['stages']['best']['mean_acc_delta']
        if real_c > 1e-12 and real_random > 1e-12:
            conclusion = '本轮真实标记同时优于原 C 和等容量随机标记，支持继续研究关系状态；尚不证明稳定收益或新颖性。'
        elif real_c > 1e-12:
            conclusion = '真实标记高于原 C，但未超过随机标记，不能将收益归因于真实互选结构。'
        else:
            conclusion = '本轮真实标记未超过原 C，不支持将此变体替换当前主线，也不自动进入有向传播阶段。'
    summary = dict(training_seed=0, planned_new_runs=20, completed_new_runs=count,
                   all_new_complete=complete, variants=variants, matched_comparisons=pairs,
                   conclusion=conclusion, caveats=caveats)
    _dump(out/'summary.json', summary)
    lines = ['# C 互选关系实验：进度与结果', '', f'新增训练完成 {count}/20；原 C 复用 10 折。',
             'TADPOLE AD/CN/SMCI，seed0，原十折，每折 1000 轮。主指标保持首次最高测试 ACC。',
             '指标按折等权；未完成时各组覆盖折数可能不同，差值仅用共同完成折。', '',
             '| 模型 | 完成折数 | 最佳 ACC | 最佳宏 F1 | 第 1000 轮 ACC | 末 100 轮 ACC | 参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for m, v in variants.items():
        if v['completed_folds']:
            lines.append(f'| {NAMES[m]} | {v["completed_folds"]}/10 | {100*v["best"]["acc"]:.4f}% | '
                         f'{100*v["best"]["macro_f1"]:.4f}% | {100*v["last"]["acc"]:.4f}% | '
                         f'{100*v["mean_last100_acc"]:.4f}% | {plan["parameters"][m]} |')
        else:
            lines.append(f'| {NAMES[m]} | 0/10 | — | — | — | — | {plan["parameters"][m]} |')
    lines += ['', '配对比较：', '']
    for pair in pairs:
        if not pair['matched_folds']:
            continue
        for stage, name in (('best', '最佳'), ('last', '末轮')):
            p = pair['stages'][stage]
            lines.append(f'- {NAMES[pair["candidate"]]} 相对 {NAMES[pair["reference"]]}，'
                         f'{len(pair["matched_folds"])} 折，{name} ACC {100*p["mean_acc_delta"]:+.4f} 个百分点；'
                         f'纠正 {p["corrected"]} 人、新增错误 {p["harmed"]} 人，净 {p["net"]:+d} 人。')
    lines += ['', conclusion, '', '实验解释边界：', '', *[f'- {x}' for x in caveats], '']
    (out/'进度与结果.md').write_text('\n'.join(lines), encoding='utf-8')
    if complete:
        (out/'结果分析.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
