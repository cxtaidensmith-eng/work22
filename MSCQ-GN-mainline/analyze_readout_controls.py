"""CPU-only reports for C readout training controls; does not run training."""
from pathlib import Path
import numpy as np
from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C', 'QK_NODECAY': '仅 query/QK 零衰减', 'UNIFORM': '从头等权读取'}
PARAMETERS = {'C': 895096, 'QK_NODECAY': 895096, 'UNIFORM': 838936}


def collect(out, plan):
    assert plan['training_seed'] == 0 and plan['modes'] == ['QK_NODECAY', 'UNIFORM']
    assert plan['folds'] == list(range(1, 11)) and plan['epochs'] == 1000
    folders = {'C': REFERENCE, **{m: Path(out)/'seed_0'/m for m in plan['modes']}}
    records = {m: {f: _record(base/f'fold_{f:02d}', f, 1000) for f in plan['folds']
                   if (base/f'fold_{f:02d}/complete.json').exists()} for m, base in folders.items()}
    for mode, rows in records.items():
        for row in rows.values():
            assert row['parameters'] == PARAMETERS[mode]
            assert row['record']['mode'] == ('content' if mode == 'C' else mode)
    return folders, records


def summarize(out, plan):
    out = Path(out)
    _, records = collect(out, plan)
    variants = {m: _aggregate(v, plan['folds']) for m, v in records.items()}
    pairs = [_pair(a, b, records[a], records[b], plan['class_names'])
             for a, b in (('C', 'QK_NODECAY'), ('C', 'UNIFORM'), ('UNIFORM', 'QK_NODECAY'))]
    completed = sum(len(records[m]) for m in plan['modes'])
    result = dict(completed_new_runs=completed, planned_new_runs=20, all_new_complete=completed == 20,
                  training_seed=0, variants=variants, matched_comparisons=pairs,
                  aggregation='Equal weight across original folds; paired comparisons use common folds only.',
                  caveats=['Original first maximum TEST ACC; exploratory, single training seed.',
                           'UNIFORM removes query/Q/K parameters but retains value/output/FFN and attention dropout.',
                           'Restoring nonuniform attention alone is not evidence of classification improvement.'])
    _dump(out/'summary.json', result)
    lines = ['# C 类别读取训练对照', '', f'新增训练完成 {completed}/20；复用原 C 的10折。',
             'TAD三分类，seed0，原十折，每折1000轮，首次最高测试ACC选轮。', '',
             '| 模型 | 完成折数 | 最佳ACC | 宏F1 | 第1000轮ACC | 末100轮ACC | 活跃参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, value in variants.items():
        if value['completed_folds'] != 10:
            lines.append(f"| {NAMES[mode]} | {value['completed_folds']}/10 | 待完成 | — | — | — | {PARAMETERS[mode]} |")
        else:
            lines.append(f"| {NAMES[mode]} | 10/10 | {100*value['best']['acc']:.4f}% | "
                         f"{100*value['best']['macro_f1']:.4f}% | {100*value['last']['acc']:.4f}% | "
                         f"{100*value['mean_last100_acc']:.4f}% | {PARAMETERS[mode]} |")
    lines += ['', '共同完成折的配对比较（部分折不能作为完整十折结论）：', '']
    for pair in pairs:
        if pair['matched_folds']:
            value = pair['stages']['best']
            lines.append(f"- {NAMES[pair['candidate']]} − {NAMES[pair['reference']]}，"
                         f"{len(pair['matched_folds'])}折：{100*value['mean_acc_delta']:+.4f}个百分点；"
                         f"纠正{value['corrected']}，新增错误{value['harmed']}。")
    lines += ['', '本轮只检验读取训练机制；早期Key来源、动态图和传播映射均不是本轮新增实验。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def export_oof(out, plan):
    folders, records = collect(out, plan)
    reference = None
    for mode, base in folders.items():
        assert len(records[mode]) == 10
        for stage, filename in [('best', 'predictions.npz'), ('last', 'last_predictions.npz')]:
            parts = []
            for fold in plan['folds']:
                with np.load(base/f'fold_{fold:02d}'/filename, allow_pickle=False) as data:
                    parts.append({k: data[k].copy() for k in ('row_index', 'labels', 'logits')})
            rows = np.concatenate([p['row_index'] for p in parts])
            labels = np.concatenate([p['labels'] for p in parts])
            logits = np.concatenate([p['logits'] for p in parts])
            folds = np.concatenate([np.full(len(p['labels']), f) for f, p in zip(plan['folds'], parts)])
            order = np.argsort(rows, kind='stable')
            rows, labels, logits, folds = rows[order], labels[order], logits[order], folds[order]
            assert len(rows) == len(np.unique(rows)) == plan['samples']
            if reference is None:
                reference = rows, labels
            np.testing.assert_array_equal(rows, reference[0])
            np.testing.assert_array_equal(labels, reference[1])
            np.savez_compressed(Path(out)/f'oof_{mode}_{stage}.npz', row_index=rows, labels=labels,
                                logits=logits, fold=folds)


def write_report(out, plan, audit):
    out = Path(out)
    summary = summarize(out, plan)
    assert summary['all_new_complete'] and audit['passed']
    assert audit['verified_runs'] == 20 and audit['verified_checkpoints'] == 40
    variants = summary['variants']
    c, q, u = [variants[m]['best']['acc'] for m in ('C', 'QK_NODECAY', 'UNIFORM')]
    if q > max(c, u) + 1e-12:
        conclusion = '本轮零衰减读取的最佳ACC高于原C及等权读取，支持进一步检验选择端训练；单种子不能确立稳定优势。'
    elif u >= q - 1e-12 and u >= c - 1e-12:
        conclusion = '本轮等权读取的最佳ACC不低于另外两组，未建立学习非均匀模态选择的必要性。'
    else:
        conclusion = '本轮未同时建立零衰减读取优于原C及等权读取的最佳ACC优势，应保留原C并结合配对错误与末轮分析。'
    lines = ['# C 类别读取训练：结果分析', '', conclusion, '',
             (out/'进度与结果.md').read_text(encoding='utf-8'),
             '20次新增训练及40个最佳/末轮检查点已复核；原C十个最佳权重在准备阶段重放。',
             'QK_NODECAY只取消query、Q/K权重与偏置的耦合L2；V、输出投影及FFN沿用原衰减。',
             'UNIFORM从匹配的V/输出/FFN初始参数开始训练，保留注意力dropout；活跃参数减少56160。',
             '注意力诊断和强制等权干预见audit.json。干预用于测量检查点依赖，不等于从头训练消融。',
             '原C没有末轮权重；其末轮指标来自既有保存预测。',
             '本轮没有自动启动早期Key来源实验；需结合这里的分类结果与读取诊断确定后续方案。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    export_oof(out, plan)
    return summary
