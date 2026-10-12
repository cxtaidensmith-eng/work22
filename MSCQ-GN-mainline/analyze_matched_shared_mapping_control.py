"""Read-only matched 2x2 control analysis for the retained TAD3 seed0 protocol.

The only new arm is D1_SHARED_MAP. Historical files/checkpoints are never
modified or replayed. summarize(out, plan) writes reports only beneath out.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_joint_evidence_routing import (
    SCALARS, STAGES, _aggregate, _dump, _pair, _parameter_count, _record,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'Result/matched_shared_mapping_seed0_20261011'
REFERENCES = {
    'C': ROOT / 'Result/relation_mechanism_nodecay/seed_0/content',
    'D1': ROOT / 'Result/c_mechanism_controls_seed0_20260924/seed_0/D1',
    'SHARED_MAP': ROOT / 'Result/shared_mapping_seed0_20261003/seed_0/SHARED_MAP',
}
MODES = ('C', 'D1', 'SHARED_MAP', 'D1_SHARED_MAP')
PARAMETERS = {'C': 895096, 'D1': 895036, 'SHARED_MAP': 895016, 'D1_SHARED_MAP': 894956}
NAMES = {
    'C': 'C：交叉路径＋内容专属映射',
    'D1': 'D1：对角路径＋内容专属映射',
    'SHARED_MAP': 'SHARED_MAP：交叉路径＋共享映射',
    'D1_SHARED_MAP': 'D1_SHARED_MAP：对角路径＋共享映射',
}
FACTORS = {
    'C': {'relation_content_paths': 'crossed', 'mapping': 'content_conditioned'},
    'D1': {'relation_content_paths': 'diagonal', 'mapping': 'content_conditioned'},
    'SHARED_MAP': {'relation_content_paths': 'crossed', 'mapping': 'shared'},
    'D1_SHARED_MAP': {'relation_content_paths': 'diagonal', 'mapping': 'shared'},
}
# Every difference is candidate minus reference; each retains the other factor.
CONTRASTS = (
    ('mapping_with_crossed_paths', 'SHARED_MAP', 'C', '交叉路径中：内容专属映射减共享映射'),
    ('mapping_with_diagonal_paths', 'D1_SHARED_MAP', 'D1', '对角路径中：内容专属映射减共享映射'),
    ('crossed_paths_with_conditioned_maps', 'D1', 'C', '内容专属映射中：交叉路径减对角路径'),
    ('crossed_paths_with_shared_maps', 'D1_SHARED_MAP', 'SHARED_MAP', '共享映射中：交叉路径减对角路径'),
)
ALL_STAGES = ('best', 'last', 'last100')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def _collect(out, plan):
    folds = [int(fold) for fold in plan.get('folds', range(1, 11))]
    if (folds != list(range(1, 11))
            or int(plan.get('training_seed', 0)) != 0
            or int(plan.get('split_seed', 0)) != 0
            or int(plan.get('epochs', 1000)) != 1000
            or list(plan.get('modes', ['D1_SHARED_MAP'])) != ['D1_SHARED_MAP']):
        raise ValueError('Expected only D1_SHARED_MAP, seed0, original ten folds and 1000 epochs')
    classes = list(plan.get('class_names', ('AD', 'CN', 'SMCI')))
    if len(classes) != 3 or len(set(classes)) != 3:
        raise ValueError('Expected three unique class names')
    for mode, expected in PARAMETERS.items():
        actual = plan.get('parameters', {}).get(mode, expected)
        if _parameter_count(actual) != expected:
            raise ValueError(f'{mode}: planned parameter count differs from the matched design')
    folders = {**REFERENCES, 'D1_SHARED_MAP': Path(out) / 'seed_0/D1_SHARED_MAP'}
    records = {mode: {} for mode in MODES}
    sources = {}
    for mode, base in folders.items():
        seen = set()
        for fold in folds:
            folder = base / f'fold_{fold:02d}'
            if not (folder / 'complete.json').exists():
                continue
            value = _record(folder, fold, 1000)
            record = value['record']
            expected_mode = 'content' if mode == 'C' else mode
            if (record.get('mode') != expected_mode or record.get('seed') != 0
                    or int(record.get('fold', -1)) != fold
                    or value['parameters'] != PARAMETERS[mode]):
                raise ValueError(f'{mode} fold {fold}: incorrect mode/fold/seed/parameter count')
            for stage in STAGES:
                prediction = value['predictions'][stage]
                if prediction['classes'] != len(classes):
                    raise ValueError(f'{mode} fold {fold}: class count differs')
                rows = prediction['rows']
                if (not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any()):
                    raise ValueError(f'{mode} fold {fold}: patient row indices must be nonnegative integers')
                # Check the saved confusion/recall values against matching predictions.
                labels, predicted = prediction['labels'], prediction['predicted']
                confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
                np.add.at(confusion, (labels, predicted), 1)
                saved = record[stage]
                if 'confusion_matrix' in saved:
                    np.testing.assert_array_equal(confusion, saved['confusion_matrix'])
                if 'per_class_recall' in saved:
                    np.testing.assert_allclose(
                        np.diag(confusion) / np.maximum(confusion.sum(1), 1),
                        saved['per_class_recall'], atol=1e-8, rtol=0)
                for metric in SCALARS:
                    if metric in saved and not np.isfinite(float(saved[metric])):
                        raise ValueError(f'{mode} fold {fold}: nonfinite {stage} {metric}')
            heldout = set(value['predictions']['best']['rows'].tolist())
            if seen & heldout:
                raise ValueError(f'{mode}: heldout patients recur across different folds')
            seen.update(heldout)
            value['heldout_patients'] = len(heldout)
            records[mode][fold] = value
            for filename in ('complete.json', 'epochs.csv', 'predictions.npz', 'last_predictions.npz'):
                path = folder / filename
                sources[str(path.resolve())] = _sha(path)
        if len(records[mode]) == len(folds):
            expected_patients = int(plan.get('samples', 598))
            if len(seen) != expected_patients:
                raise ValueError(f'{mode}: full ten-fold heldout total differs from {expected_patients}')
    # All available arms for a fold must identify exactly the same heldout people.
    for fold in folds:
        available = [mode for mode in MODES if fold in records[mode]]
        if not available:
            continue
        reference = records[available[0]][fold]['predictions']['best']
        for mode in available[1:]:
            candidate = records[mode][fold]['predictions']['best']
            np.testing.assert_array_equal(reference['rows'], candidate['rows'])
            np.testing.assert_array_equal(reference['labels'], candidate['labels'])
    return folds, classes, folders, records, sources


def _describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'mean': None, 'sd': None, 'min': None, 'max': None}
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite descriptive values')
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), max=float(values.max()))


def _pair_details(identifier, reference, candidate, label, records, classes):
    pair = _pair(reference, candidate, records[reference], records[candidate], classes)
    pair.update(id=identifier, label_zh=label,
                factor_change='content-conditioned minus shared' if identifier.startswith('mapping')
                              else 'crossed minus diagonal',
                single_factor_configuration_contrast=True)
    common = pair['matched_folds']
    for stage in STAGES:
        if not common:
            continue
        item = pair['stages'][stage]
        item['metric_deltas'] = {}
        for metric in SCALARS:
            if all(metric in records[arm][fold]['record'][stage]
                   for arm in (reference, candidate) for fold in common):
                deltas = [float(records[candidate][fold]['record'][stage][metric])
                          - float(records[reference][fold]['record'][stage][metric]) for fold in common]
                item['metric_deltas'][metric] = {
                    **_describe(deltas),
                    'folds': [dict(fold=fold, delta=delta) for fold, delta in zip(common, deltas)],
                }
        item['mean_acc_delta_percentage_points'] = 100 * item['mean_acc_delta']
    if common:
        details = [dict(fold=fold,
                        reference_acc=records[reference][fold]['mean_last100_acc'],
                        candidate_acc=records[candidate][fold]['mean_last100_acc'],
                        acc_delta=records[candidate][fold]['mean_last100_acc']
                                  - records[reference][fold]['mean_last100_acc']) for fold in common]
        values = [row['acc_delta'] for row in details]
        description = _describe(values)
        pair['stages']['last100'] = {
            'mean_acc_delta': description['mean'],
            'mean_acc_delta_percentage_points': 100 * description['mean'],
            'sd_acc_delta': description['sd'],
            'fold_wins': sum(value > 1e-12 for value in values),
            'fold_ties': sum(abs(value) <= 1e-12 for value in values),
            'fold_losses': sum(value < -1e-12 for value in values),
            'folds': details,
            'prediction_counts_available': False,
            'note': 'Mean TEST ACC over epochs 901-1000; no averaged-prediction interpretation.',
        }
    return pair


def _interaction(records):
    common = sorted(set.intersection(*(set(records[mode]) for mode in MODES)))
    result = {
        'formula': 'C - SHARED_MAP - D1 + D1_SHARED_MAP',
        'equivalent': '(C - SHARED_MAP) - (D1 - D1_SHARED_MAP)',
        'metric': 'ACC proportion',
        'matched_folds': common,
        'aggregation': 'Compute within each matched fold, then average folds equally.',
        'interpretation': 'Descriptive difference of configuration contrasts, not a causal or significant interaction.',
        'stages': {},
    }
    for stage in ALL_STAGES:
        details = []
        for fold in common:
            values = {
                mode: records[mode][fold]['mean_last100_acc'] if stage == 'last100'
                else float(records[mode][fold]['record'][stage]['acc']) for mode in MODES
            }
            delta = values['C'] - values['SHARED_MAP'] - values['D1'] + values['D1_SHARED_MAP']
            details.append(dict(fold=fold, **values, interaction=delta,
                                interaction_percentage_points=100 * delta))
        if details:
            description = _describe([row['interaction'] for row in details])
            result['stages'][stage] = {
                **description, 'mean_acc_interaction': description['mean'],
                'mean_acc_interaction_percentage_points': 100 * description['mean'],
                'fold_positive': sum(row['interaction'] > 1e-12 for row in details),
                'fold_zero': sum(abs(row['interaction']) <= 1e-12 for row in details),
                'fold_negative': sum(row['interaction'] < -1e-12 for row in details),
                'folds': details,
            }
    return result


def _write_group_csv(out, mode, records):
    fields = ['training_seed', 'fold', 'mode', 'parameters', 'heldout_patients']
    fields += [f'{stage}_{key}' for stage in STAGES for key in ('epoch', *SCALARS)]
    fields += ['last100_acc', 'last100_first_epoch', 'last100_final_epoch']
    target = out / f'metrics_{mode}.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold, item in sorted(records.items()):
            row = dict(training_seed=0, fold=fold, mode=mode, parameters=item['parameters'],
                       heldout_patients=item['heldout_patients'], last100_acc=item['mean_last100_acc'],
                       last100_first_epoch=901, last100_final_epoch=1000)
            for stage in STAGES:
                for metric in ('epoch', *SCALARS):
                    row[f'{stage}_{metric}'] = item['record'][stage].get(metric, '')
            writer.writerow(row)
    temporary.replace(target)


def _pct(value):
    return '未完成' if value is None else f'{100 * value:.4f}%'


def _write_report(out, summary):
    complete = summary['all_four_complete']
    variants = summary['variants']
    lines = ['匹配共享映射对照：实验结果说明', '',
             f"新增 D1_SHARED_MAP 完成 {summary['completed_new_runs']}/10 折；"
             f"四组全部完成：{'是' if complete else '否'}。",
             'seed0，原十折划分，每折1000轮；主指标为首次达到最高 TEST ACC 的检查点。',
             '各指标按折等权平均。末轮和末100轮为辅助诊断，不能替换主指标。', '',
             '四组结构：']
    for mode in MODES:
        value = variants[mode]
        lines.append(f"- {NAMES[mode]}；{value['completed_folds']}/10 折；活跃参数 {PARAMETERS[mode]}。")
    lines += ['', '各组描述值（完成折数不齐时不作四组总体排名）：']
    for mode in MODES:
        value = variants[mode]
        lines.append(f"- {mode}：best ACC {_pct(value.get('best', {}).get('acc'))}；"
                     f"last ACC {_pct(value.get('last', {}).get('acc'))}；"
                     f"last100 ACC {_pct(value.get('mean_last100_acc'))}；"
                     f"best 宏F1 {_pct(value.get('best', {}).get('macro_f1'))}；"
                     f"best AUC {_pct(value.get('best', {}).get('auc'))}。")
    lines += ['', '四个单因素配置对比（差值均为前述候选组减基准组，单位为 ACC 百分点）：',
              '各对比仅使用共同完成且患者行号、标签完全一致的折。'
              '纠正／新增错误是匹配患者的描述计数，与折均值采用不同权重。']
    for pair in summary['matched_comparisons']:
        lines += ['', pair['label_zh'] + f"；共同完成 {len(pair['matched_folds'])}/10 折。"]
        for stage, label in (('best', '最佳选点'), ('last', '第1000轮'), ('last100', '末100轮平均')):
            if stage not in pair['stages']:
                lines.append(f"- {label}：无共同完成折。")
                continue
            value = pair['stages'][stage]
            text = (f"- {label}：{100 * value['mean_acc_delta']:+.4f} 个百分点；"
                    f"{value['fold_wins']} 升／{value['fold_ties']} 平／{value['fold_losses']} 降。")
            if stage != 'last100':
                text += (f" 纠正 {value['corrected']} 人，新增错误 {value['harmed']} 人，"
                         f"净 {value['net']:+d} 人。")
            lines.append(text)
    lines += ['', '描述性交互：C−SHARED_MAP−D1＋D1_SHARED_MAP。',
              '等价于“交叉路径中专属映射的变化”减“对角路径中专属映射的变化”；'
              '先在同一折计算，再将共同折等权平均。']
    interaction = summary['descriptive_interaction']
    for stage, label in (('best', '最佳选点'), ('last', '第1000轮'), ('last100', '末100轮平均')):
        value = interaction['stages'].get(stage)
        lines.append(f"- {label}：无四组共同完成折。" if value is None else
                     f"- {label}：{100 * value['mean']:+.4f} 个百分点；"
                     f"共同 {len(interaction['matched_folds'])} 折，"
                     f"{value['fold_positive']} 正／{value['fold_zero']} 零／{value['fold_negative']} 负。")
    lines += ['', summary['conclusion'], '',
              '解释边界：D1_SHARED_MAP 仅将原 D1 compact 六路的六个16维内容ID，'
              '替换为一个以这六个初始ID均值初始化的可学习共享ID；发送和接收映射仍分别学习，'
              '映射生成器及原 D1 机制保留。共享变体均减少80个活跃参数；'
              '交叉与对角结构相差60个活跃参数，四组并非严格等容量或等初始函数。',
              '这是单训练种子、历史 TEST 选点下的匹配探索比较。配对十折不作为独立样本，'
              '不计算显著性、置信区间或因果机制效应。描述性交互不自动证明协同创新或临床价值。',
              '本分析核对保存预测、完整训练曲线、折/种子/模式/参数量以及患者配对；'
              '没有重放或修改模型检查点，也不要求历史 C 提供 last.pt。'
              '代码、权重重放与训练协议来源审计由实验运行器另行完成。',
              '四个 metrics_*.csv 保存各组逐折 best/last/last100 指标；'
              'summary.json 保存四个对比、逐折交互、类别纠错和输入文件哈希。', '']
    _atomic_text(out / '实验结果说明.txt', '\n'.join(lines))


def summarize(out, plan):
    """Validate available folds, write four CSVs and descriptive summaries.

    May be called during training. Partial results never yield a final ranking.
    No checkpoint load, model inference, training, or historical-file mutation.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    folds, classes, folders, records, sources = _collect(out, plan)
    variants = {mode: _aggregate(records[mode], folds) for mode in MODES}
    for mode, value in variants.items():
        value['factor_levels'] = FACTORS[mode]
        value['expected_parameters'] = PARAMETERS[mode]
        if records[mode]:
            value['last100'] = {
                'acc': value['mean_last100_acc'],
                'description': _describe([row['mean_last100_acc'] for row in records[mode].values()]),
                'epochs': [901, 1000], 'auxiliary_only': True,
            }
            for row in value['folds']:
                row['heldout_patients'] = records[mode][row['fold']]['heldout_patients']
    comparisons = [_pair_details(identifier, reference, candidate, label, records, classes)
                   for identifier, reference, candidate, label in CONTRASTS]
    completed = len(records['D1_SHARED_MAP'])
    all_complete = all(len(records[mode]) == 10 for mode in MODES)
    conclusion = (
        '四组十折已全部完成。以下差值完整报告四个预定单因素配置对比和逐折描述性交互，'
        '不以辅助指标替换原 best ACC，不作显著性或因果结论。'
        if all_complete else
        '四组十折尚未全部完成，当前仅报告进度与共同折的描述结果；不进行四组总体排名，'
        '不判断专属映射、交叉路径或二者交互是否带来完整十折收益。'
    )
    summary = dict(
        updated_at=datetime.now(timezone.utc).isoformat(), training_seed=0, split_seed=0,
        planned_folds=folds, planned_new_runs=10, completed_new_runs=completed,
        all_new_complete=completed == 10,
        all_reference_complete=all(len(records[mode]) == 10 for mode in REFERENCES),
        all_four_complete=all_complete, variants=variants, matched_comparisons=comparisons,
        descriptive_interaction=_interaction(records), class_names=classes,
        directories={mode: str(path.resolve()) for mode, path in folders.items()},
        parameter_counts=PARAMETERS,
        factorial_design=FACTORS,
        selection='First maximum TEST ACC over 1000 epochs; best is the retained primary endpoint.',
        auxiliary_endpoints=['Epoch 1000 TEST metrics', 'Mean TEST ACC over epochs 901-1000'],
        aggregation='Equal-fold metric means; all contrasts use matching completed folds.',
        units={'acc': 'proportion', 'acc_percentage_points': '100 times an ACC difference'},
        sample_sd_note='Descriptive delta SD uses ddof=1; legacy variant std_acc uses ddof=0.',
        source_sha256=sources,
        analysis_source_sha256={str(Path(__file__).resolve()): _sha(__file__),
                                str(ROOT / 'analyze_joint_evidence_routing.py'):
                                _sha(ROOT / 'analyze_joint_evidence_routing.py')},
        checkpoint_replay_performed=False, training_started_by_analysis=False,
        conclusion=conclusion,
        caveats=[
            'Single training seed, transductive historical TEST-selected exploratory protocol.',
            'The four single-factor configuration contrasts are near-matched, not exact-capacity causal effects.',
            'Sharing changes conditioner input, initial map values, and 80 active parameters.',
            'Crossed versus compact diagonal relation-content paths differ by 60 active parameters.',
            'Fold results are paired descriptive values, not independent replicates for significance testing.',
            'Last and last100 diagnostics cannot replace the pre-retained primary best ACC.',
            'Checkpoint/source/protocol audits remain the responsibility of the experiment runner.',
        ],
    )
    for mode in MODES:
        _write_group_csv(out, mode, records[mode])
    _dump(out / 'summary.json', summary)
    _write_report(out, summary)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = json.loads((args.output / 'PLAN.json').read_text(encoding='utf-8-sig'))
    result = summarize(args.output, plan)
    print(json.dumps({'completed_new_runs': result['completed_new_runs'],
                      'all_four_complete': result['all_four_complete']}, ensure_ascii=False))
