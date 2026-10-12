"""CPU reports for the fixed seed-0 grouped-content experiment."""
import csv
from pathlib import Path
import numpy as np
from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C：单组映射', 'G2': '两组隐通道映射', 'W38': '单组映射＋容量对照'}
PARAMETERS = {'C': 895096, 'G1': 895096, 'G2': 895228, 'W38': 895222}


def collect(out, plan):
    assert plan['training_seed'] == 0 and plan['modes'] == ['G2', 'W38']
    assert plan['folds'] == list(range(1, 11)) and plan['epochs'] == 1000
    folders = {'C': REFERENCE, **{m: Path(out)/'seed_0'/m for m in plan['modes']}}
    records = {m: {f: _record(base/f'fold_{f:02d}', f, 1000) for f in plan['folds']
                   if (base/f'fold_{f:02d}/complete.json').exists()} for m, base in folders.items()}
    for mode, values in records.items():
        for value in values.values():
            assert value['parameters'] == PARAMETERS[mode]
            assert value['record']['mode'] == ('content' if mode == 'C' else mode)
    return folders, records


def summarize(out, plan):
    out = Path(out)
    _, records = collect(out, plan)
    variants = {m: _aggregate(v, plan['folds']) for m, v in records.items()}
    pairs = [_pair(a, b, records[a], records[b], plan['class_names'])
             for a, b in (('C', 'G2'), ('C', 'W38'), ('W38', 'G2'))]
    completed = sum(len(records[m]) for m in plan['modes'])
    result = dict(completed_new_runs=completed, planned_new_runs=20,
        all_new_complete=completed == 20, training_seed=0, variants=variants,
        matched_comparisons=pairs,
        aggregation='Equal mean of original ten folds; pairs use matching completed folds only.',
        caveats=['Single seed; original first maximum TEST ACC checkpoint selection.',
                 'G2 and W38 differ by 6 active parameters; capacity is approximately matched.',
                 'Collapse is a fixed-checkpoint diagnostic, not a retrained or causal ablation.'])
    _dump(out/'summary.json', result)
    lines = ['# C 内容映射分组：seed 0 原十折', '',
        f'新增训练完成 {completed}/20；复用原 C 的10折。',
        '每折1000轮，原数据、图支持、损失、优化器、随机种子和首次最高测试ACC选轮保持一致。', '',
        '| 模型 | 完成折数 | 最佳ACC | 宏F1 | 第1000轮ACC | 末100轮ACC | 参数量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode, value in variants.items():
        if value['completed_folds'] != 10:
            lines.append(f"| {NAMES[mode]} | {value['completed_folds']}/10 | 待完成 | — | — | — | {PARAMETERS[mode]} |")
        else:
            lines.append(f"| {NAMES[mode]} | 10/10 | {100*value['best']['acc']:.4f}% | "
                f"{100*value['best']['macro_f1']:.4f}% | {100*value['last']['acc']:.4f}% | "
                f"{100*value['mean_last100_acc']:.4f}% | {PARAMETERS[mode]} |")
    lines += ['', '以下仅比较共同完成折；部分折不能当作完整十折结论：', '']
    for pair in pairs:
        if pair['matched_folds']:
            item = pair['stages']['best']
            lines.append(f"- {NAMES[pair['candidate']]} − {NAMES[pair['reference']]}，"
                f"{len(pair['matched_folds'])}折：ACC {100*item['mean_acc_delta']:+.4f}个百分点；"
                f"纠正{item['corrected']}、新增错误{item['harmed']}、净{item['net']:+d}；"
                f"{item['fold_wins']}胜/{item['fold_ties']}平/{item['fold_losses']}负。")
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def export_oof(out, plan):
    folders, records = collect(out, plan)
    reference_rows = reference_labels = None
    manifest = {}
    for mode, base in folders.items():
        assert len(records[mode]) == 10
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            parts = []
            for fold in plan['folds']:
                with np.load(base/f'fold_{fold:02d}'/filename, allow_pickle=False) as saved:
                    parts.append({key: saved[key].copy() for key in ('row_index', 'labels', 'logits')})
            rows = np.concatenate([p['row_index'] for p in parts])
            labels = np.concatenate([p['labels'] for p in parts])
            logits = np.concatenate([p['logits'] for p in parts])
            folds = np.concatenate([np.full(len(p['labels']), f) for f, p in zip(plan['folds'], parts)])
            order = np.argsort(rows, kind='stable')
            rows, labels, logits, folds = rows[order], labels[order], logits[order], folds[order]
            assert len(rows) == len(np.unique(rows)) == plan['samples']
            if reference_rows is None:
                reference_rows, reference_labels = rows, labels
            np.testing.assert_array_equal(rows, reference_rows)
            np.testing.assert_array_equal(labels, reference_labels)
            target = Path(out)/f'oof_{mode}_{stage}.npz'
            np.savez_compressed(target, row_index=rows, labels=labels, logits=logits, fold=folds)
            manifest[f'{mode}_{stage}'] = dict(file=target.name, patients=len(rows),
                pooled_acc=float((logits.argmax(-1) == labels).mean()),
                interpretation='OOF rows; each checkpoint selected by original TEST ACC protocol.')
    _dump(Path(out)/'oof_manifest.json', manifest)


def write_report(out, plan, audit):
    out = Path(out)
    result = summarize(out, plan)
    assert result['all_new_complete'] and audit['passed']
    assert audit['verified_runs'] == 20 and audit['verified_checkpoints'] == 40
    a = result['matched_comparisons'][0]['stages']['best']['mean_acc_delta']
    b = result['matched_comparisons'][2]['stages']['best']['mean_acc_delta']
    if a > 1e-12 and b > 1e-12:
        conclusion = '本轮G2的最佳ACC高于原C与近等容量W38，可继续重复验证；单种子尚不能确立稳定机制优势。'
    elif a > 1e-12:
        conclusion = 'G2高于原C，但没有同时超过近等容量W38，尚不足以将提升归因于通道组映射。'
    else:
        conclusion = '本轮G2没有超过原C的最佳ACC，尚未支持直接用两组版本替代C。'
    lines = ['# C 内容映射分组：结果分析', '', conclusion, '',
        (out/'进度与结果.md').read_text(encoding='utf-8'),
        '新增20次完整训练及40个最佳/末轮检查点已重放复核；原C的10个最佳检查点在准备阶段重放。',
        '原C没有last.pt，其末轮指标来自原始保存预测，未声称完成原C末轮权重重放。',
        '本轮G2专指两组隐通道映射，与历史同名关系条件G2不是同一个实验。',
        'G2按原[2,H/2]布局的第二轴分成两组，每组学习独立发送/接收角度与尺度；隐通道组不代表脑区。',
        '映射仍由内容模态ID产生，患者间共享；并未新增患者条件、动态图支持或模态分组。',
        'G2新增132参数；W38将单组映射条件器宽度32改为38，新增126参数；二者差6参数，属于近似容量对照。',
        '两组映射初始复制C；W38前32单元复制C，新增6单元输出权重初始为零并在训练中激活。',
        '所有新训练均从随机初始化开始，共用模块与随机流对齐，没有使用已训练C权重热启动。', '',
        '## 固定权重下的组映射折叠', '',
        '| 模型/检查点 | 正常ACC | 折叠ACC | 改变预测数 | 纠正/新增错误 | 平均组参数标准差 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ('G2', 'W38'):
        for stage in ('best', 'last'):
            rows = [r for r in audit['records'] if r['mode'] == mode and r['stage'] == stage]
            assert len(rows) == 10
            normal = np.mean([r['interventions']['normal']['metrics']['acc'] for r in rows])
            collapse = np.mean([r['interventions']['collapse']['metrics']['acc'] for r in rows])
            counts = {k: sum(r['interventions']['collapse'][k] for r in rows)
                      for k in ('changed', 'corrected', 'harmed')}
            spread = np.mean([r['diagnostics']['mean_group_parameter_std'] for r in rows])
            lines.append(f"| {NAMES[mode]}/{stage} | {100*normal:.4f}% | {100*collapse:.4f}% | "
                f"{counts['changed']} | {counts['corrected']}/{counts['harmed']} | {spread:.6f} |")
    lines += ['', '折叠先对原始角度/尺度logits跨组取均值再广播，再构成发送/接收映射；不是对变换矩阵直接平均。',
        '仅在既定检查点推断，不重新训练、不重新选轮；可描述对组差异的依赖，不能解释成因果效应或独有互补信息。',
        'W38只有一组，折叠应完全等价，作为诊断路径的实现检查。',
        '本轮遵循原测试选轮规则，不是独立测试集评估；全部候选预先固定，不基于预检测试分数筛选。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (out/'paired_folds.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['comparison', 'stage', 'fold', 'reference_acc', 'candidate_acc', 'acc_delta',
                  'corrected', 'harmed', 'net', 'patients']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in result['matched_comparisons']:
            for stage, value in pair['stages'].items():
                for row in value['folds']:
                    writer.writerow(dict(comparison=pair['comparison'], stage=stage, **row))
    export_oof(out, plan)
    return result
