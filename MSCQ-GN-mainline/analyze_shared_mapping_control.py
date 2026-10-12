"""Read-only metrics for the shared learnable mapping ablation of C.

Only the specified experiment output directory receives reports. The runner
owns source/protocol checks and checkpoint replay; no training is started here.
"""
from datetime import datetime, timezone
from pathlib import Path

from analyze_joint_evidence_routing import (
    _aggregate, _dump, _pair, _parameter_count, _record,
)


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C：模态条件映射', 'SHARED_MAP': '共享可学习映射'}
PARAMETERS = {'C': 895096, 'SHARED_MAP': 895016}


def _collect(out, plan):
    folds = list(plan.get('folds', range(1, 11)))
    if (plan.get('training_seed', 0) != 0
            or list(plan.get('modes', ['SHARED_MAP'])) != ['SHARED_MAP']
            or folds != list(range(1, 11)) or int(plan.get('epochs', 1000)) != 1000):
        raise ValueError('Expected SHARED_MAP, seed0, original 10 folds, 1000 epochs')
    classes = list(plan.get('class_names', ('AD', 'CN', 'SMCI')))
    if len(classes) != 3 or len(set(classes)) != 3:
        raise ValueError('Expected three unique class names')
    for mode, expected in PARAMETERS.items():
        value = plan.get('parameters', {}).get(mode, expected)
        if _parameter_count(value) != expected:
            raise ValueError(f'{mode}: planned parameter count differs from control design')
    folders = {'C': REFERENCE, 'SHARED_MAP': Path(out) / 'seed_0' / 'SHARED_MAP'}
    records = {
        mode: {fold: _record(base / f'fold_{fold:02d}', fold, 1000)
               for fold in folds if (base / f'fold_{fold:02d}' / 'complete.json').exists()}
        for mode, base in folders.items()
    }
    for mode, rows in records.items():
        for fold, row in rows.items():
            expected_mode = 'content' if mode == 'C' else mode
            if row['record']['mode'] != expected_mode or row['parameters'] != PARAMETERS[mode]:
                raise ValueError(f'{mode} fold {fold}: incorrect mode or active parameter count')
            for stage in ('best', 'last'):
                if row['predictions'][stage]['classes'] != len(classes):
                    raise ValueError(f'{mode} fold {fold}: incorrect prediction class count')
    return folds, classes, records


def _percent(value):
    return '—' if value is None else f'{100 * value:.4f}%'


def _conclusion(complete, pair):
    if not complete or len(pair['matched_folds']) != 10:
        return ('十折尚未全部完成，当前仅记录共同完成折的配对结果；'
                '不据部分折判断模态条件映射是否必要。')
    delta = pair['stages']['best']['mean_acc_delta']
    if delta < -1e-12:
        return ('共享映射的十折最佳 ACC 低于原 C；在本轮固定设置下，'
                '这一结果支持保留按内容模态区分映射。'
                '它不能单独证明此机制具有稳定优势或临床解释性。')
    if delta > 1e-12:
        return ('共享映射的十折最佳 ACC 高于原 C；本轮没有支持模态专属映射的必要性，'
                '共享映射可作为待复核的简化候选。不能据单种子结果确立稳定优势。')
    return ('共享映射与原 C 的十折最佳 ACC 相同；本轮未显示区分内容模态映射的净分类收益。'
            '仍需结合纠错／新增错误及末轮行为判断机制差异。')


def summarize(out, plan):
    """Validate completed records, write equal-fold reports, and return summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    folds, classes, records = _collect(out, plan)
    variants = {mode: _aggregate(rows, folds) for mode, rows in records.items()}
    pair = _pair('C', 'SHARED_MAP', records['C'], records['SHARED_MAP'], classes)
    completed = len(records['SHARED_MAP'])
    complete = completed == len(folds)
    parameter_delta = PARAMETERS['SHARED_MAP'] - PARAMETERS['C']
    conclusion = _conclusion(complete, pair)
    result = dict(
        updated_at=datetime.now(timezone.utc).isoformat(), training_seed=0,
        planned_folds=folds, planned_new_runs=len(folds), completed_new_runs=completed,
        all_new_complete=complete, all_reference_complete=len(records['C']) == len(folds),
        variants=variants, matched_comparisons=[pair], class_names=classes,
        reference_directory=str(REFERENCE),
        aggregation='Equal-fold metric means; paired comparisons use common folds only.',
        selection='First maximum TEST ACC; last epoch and last-100 means are secondary diagnostics.',
        mapping_design=dict(
            original='Six learnable 16-dimensional content IDs and the same 16->32->4 conditioner',
            shared='One learnable 16-dimensional shared ID and the same 16->32->4 conditioner',
            initialization='Shared ID is the mean of the six native C initial content IDs',
            roles='Sending and receiving maps remain distinct; both are shared across modalities',
            parameter_counts=PARAMETERS, parameter_delta=parameter_delta,
            parameter_delta_percent=100 * parameter_delta / PARAMETERS['C']),
        conclusion=conclusion,
        caveats=[
            'Single training seed under the original test-selected exploratory protocol.',
            'Sharing removes 80 active parameters; this is near-matched, not exactly equal capacity.',
            'Initialization matches common parameters, not the original mode-specific map outputs.',
            'Existing relation-by-content paths and learned edge weights are retained.',
            'Checkpoint/source/protocol audits are performed separately by the runner.',
            'Any observed difference does not establish novelty, causal or clinical interpretation.',
        ],
    )
    _dump(out / 'summary.json', result)

    lines = ['# C 共享可学习映射消融：进度与结果', '',
             f'新增训练完成 {completed}/10 次；原 C 已读取 {len(records["C"])}/10 折。',
             'TAD 三分类，seed0，原十折，每折 1000 轮；保留首次最高测试 ACC 的选轮规则。', '',
             '各指标按折等权平均。未完成时各组可能覆盖不同折数；'
             '下表显示进度与已完成折描述值，不进行完整十折排名。'
             '配对差值只使用相同患者、相同划分的共同已完成折。', '',
             '| 模型 | 完成折数 | 最佳 ACC | 最佳宏 F1 | 最佳 AUC | 最佳轮数均值 | 活跃参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, value in variants.items():
        stage = value.get('best', {})
        epoch = '—' if not stage else f'{stage["mean_epoch"]:.1f}'
        lines.append(f'| {NAMES[mode]} | {value["completed_folds"]}/10 | '
                     f'{_percent(stage.get("acc"))} | {_percent(stage.get("macro_f1"))} | '
                     f'{_percent(stage.get("auc"))} | {epoch} | {PARAMETERS[mode]} |')
    lines += ['', '| 模型 | 第 1000 轮 ACC | 末轮宏 F1 | 末轮 AUC | 末 100 轮平均 ACC |',
              '|---|---:|---:|---:|---:|']
    for mode, value in variants.items():
        stage = value.get('last', {})
        lines.append(f'| {NAMES[mode]} | {_percent(stage.get("acc"))} | '
                     f'{_percent(stage.get("macro_f1"))} | {_percent(stage.get("auc"))} | '
                     f'{_percent(value.get("mean_last100_acc"))} |')

    lines += ['', '各类召回率同样按已完成折等权平均：', '',
              '| 模型 | 选点 | ' + ' | '.join(classes) + ' |',
              '|---|---|' + '---:|' * len(classes)]
    for mode, value in variants.items():
        for stage, label in (('best', '最佳'), ('last', '末轮')):
            recalls = value.get(stage, {}).get('per_class_recall', [None] * len(classes))
            if len(recalls) != len(classes):
                raise ValueError(f'{mode}/{stage}: class recall length is inconsistent')
            lines.append(f'| {NAMES[mode]} | {label} | '
                         + ' | '.join(_percent(item) for item in recalls) + ' |')

    lines += ['', '配对比较（共享映射减原 C）：', '']
    if not pair['matched_folds']:
        lines.append('尚无共同完成折。')
    else:
        for stage, label in (('best', '最佳'), ('last', '末轮')):
            item = pair['stages'][stage]
            lines.append(f'- {len(pair["matched_folds"])} 折，{label}：'
                         f'ACC {100 * item["mean_acc_delta"]:+.4f} 个百分点；'
                         f'纠正 {item["corrected"]} 人，新增错误 {item["harmed"]} 人，'
                         f'净 {item["net"]:+d} 人；'
                         f'{item["fold_wins"]} 胜/{item["fold_ties"]} 平/{item["fold_losses"]} 负。')
        lines.append(f'- 共同完成折末 100 轮平均 ACC 差值：'
                     f'{100 * pair["mean_last100_acc_delta"]:+.4f} 个百分点。')
        lines += ['', '纠正／新增错误按匹配患者计数，用于描述决策差异，不能替代折均值。', '',
                  '| 选点 | 类别 | 纠正 | 新增错误 | 净变化 | 折均召回差值 |',
                  '|---|---|---:|---:|---:|---:|']
        for stage, label in (('best', '最佳'), ('last', '末轮')):
            for name, item in pair['stages'][stage]['per_class'].items():
                delta = item['mean_recall_delta']
                text = '—' if delta is None else f'{100 * delta:+.4f} 个百分点'
                lines.append(f'| {label} | {name} | {item["corrected"]} | '
                             f'{item["harmed"]} | {item["net"]:+d} | {text} |')

    lines += ['', conclusion, '',
              '设计边界：将六个 16 维内容模态 ID 替换为一个可学习的共享 ID，'
              '初值取原 C 六个 ID 的均值；保留相同的 16→32→4 映射生成器。'
              '发送映射与接收映射仍然分别学习，只取消它们在内容模态之间的区别。'
              '全部关系来源×内容模态路径及内容相关边权保留。', '',
              f'参数量由 {PARAMETERS["C"]} 降为 {PARAMETERS["SHARED_MAP"]}，'
              f'减少 80 个（{100 * 80 / PARAMETERS["C"]:.4f}%）；'
              '这是接近容量的对照，并非严格等参数对照。'
              '公共参数初始化保持匹配，但共享后的初始映射与原 C 各模态映射并不完全相同。', '',
              '本轮沿用原测试选点协议，属于单种子探索结果。末轮与末 100 轮用于辅助诊断，'
              '不会事后替换主指标。训练记录完整性检查在本报告生成时执行；'
              '权重重放、代码和协议复核以运行器另行生成的审计记录为准。', '']
    temporary = out / '进度与结果.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(out / '进度与结果.md')
    return result
