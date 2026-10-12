"""Read-only analysis of the NO_CONTENT_SIMILARITY seed0 matched control.

summarize(out, plan) can update progress after any completed fold. Historical
C checkpoints are not replayed or modified; C last_predictions.npz is enough.
"""
import csv
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_joint_evidence_routing import SCALARS, STAGES, _aggregate, _dump, _pair, _parameter_count, _record


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0/content'
DEFAULT_OUTPUT = ROOT / 'Result/no_content_similarity_seed0_20261011'
CONTROL = 'NO_CONTENT_SIMILARITY'
MODES = ('C', CONTROL)
PARAMETERS = {'C': 895096, CONTROL: 893560}
NAMES = {'C': '原 C：保留当前内容相似度项', CONTROL: '对照：去除当前内容相似度项'}
PLOT_METRICS = ('acc', 'macro_f1', 'auc')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def _describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, sd=None, min=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite descriptive values')
    return dict(mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), max=float(values.max()))


def _check_class_metrics(value, classes, folder):
    """Recompute confusion, recall and F1 from saved-logit argmax predictions."""
    for stage in STAGES:
        prediction = value['predictions'][stage]
        if prediction['classes'] != len(classes):
            raise ValueError(f'{folder}: inconsistent class count')
        rows = prediction['rows']
        if not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any():
            raise ValueError(f'{folder}: invalid patient indices')
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        np.add.at(confusion, (prediction['labels'], prediction['predicted']), 1)
        score = value['record'][stage]
        for metric in PLOT_METRICS:
            if metric not in score or not np.isfinite(float(score[metric])):
                raise ValueError(f'{folder}: missing/nonfinite {stage} {metric}')
        support = confusion.sum(1)
        recall = np.diag(confusion) / np.maximum(support, 1)
        denominator = support + confusion.sum(0)
        f1_per_class = 2 * np.diag(confusion) / np.maximum(denominator, 1)
        weighted_f1 = float(np.sum(f1_per_class * support) / support.sum())
        np.testing.assert_array_equal(confusion, score['confusion_matrix'])
        np.testing.assert_allclose(recall, score['per_class_recall'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(f1_per_class.mean(), score['macro_f1'], atol=1e-8, rtol=0)
        if 'f1' in score:
            np.testing.assert_allclose(weighted_f1, score['f1'], atol=1e-8, rtol=0)
        filename = 'predictions.npz' if stage == 'best' else 'last_predictions.npz'
        with np.load(folder / filename, allow_pickle=False) as saved:
            logits = np.asarray(saved['logits'])
        # Retain the historical implementation: macro OVR AUC on raw logits.
        # Applying softmax here would silently change the original metric.
        auc = roc_auc_score(np.eye(len(classes))[prediction['labels']], logits,
                            average='macro', multi_class='ovr')
        np.testing.assert_allclose(auc, score['auc'], atol=1e-8, rtol=0)
        for metric in SCALARS:
            if metric in score and not np.isfinite(float(score[metric])):
                raise ValueError(f'{folder}: nonfinite {stage} {metric}')


def _collect(out, plan):
    folds = [int(fold) for fold in plan.get('folds', range(1, 11))]
    if (folds != list(range(1, 11))
            or int(plan.get('training_seed', 0)) != 0
            or int(plan.get('split_seed', 0)) != 0
            or int(plan.get('epochs', 1000)) != 1000
            or list(plan.get('modes', [CONTROL])) != [CONTROL]):
        raise ValueError('Expected NO_CONTENT_SIMILARITY, seed0, original ten folds and 1000 epochs')
    classes = list(plan.get('class_names', ('AD', 'CN', 'SMCI')))
    if len(classes) != 3 or len(set(classes)) != 3:
        raise ValueError('Expected three unique class names in saved prediction order')
    for mode, expected in PARAMETERS.items():
        if _parameter_count(plan.get('parameters', {}).get(mode, expected)) != expected:
            raise ValueError(f'{mode}: incorrect planned parameter count')
    folders = {'C': REFERENCE, CONTROL: Path(out) / 'seed_0' / CONTROL}
    records, sources = {mode: {} for mode in MODES}, {}
    for mode, base in folders.items():
        seen = set()
        for fold in folds:
            folder = base / f'fold_{fold:02d}'
            if not (folder / 'complete.json').exists():
                continue
            value = _record(folder, fold, 1000)
            record = value['record']
            expected_mode = 'content' if mode == 'C' else CONTROL
            if (record.get('mode') != expected_mode or record.get('seed') != 0
                    or int(record.get('fold', -1)) != fold or value['parameters'] != PARAMETERS[mode]):
                raise ValueError(f'{mode} fold {fold}: incorrect mode/seed/fold/parameters')
            _check_class_metrics(value, classes, folder)
            heldout = set(value['predictions']['best']['rows'].tolist())
            if seen & heldout:
                raise ValueError(f'{mode}: heldout patients recur across folds')
            seen.update(heldout)
            value['heldout_patients'] = len(heldout)
            records[mode][fold] = value
            for filename in ('complete.json', 'epochs.csv', 'predictions.npz', 'last_predictions.npz'):
                path = folder / filename
                sources[str(path.resolve())] = _sha(path)
        if len(records[mode]) == 10 and len(seen) != int(plan.get('samples', 598)):
            raise ValueError(f'{mode}: full heldout patient total differs from planned sample count')
    for fold in sorted(set(records['C']) & set(records[CONTROL])):
        for stage in STAGES:
            left, right = records['C'][fold]['predictions'][stage], records[CONTROL][fold]['predictions'][stage]
            np.testing.assert_array_equal(left['rows'], right['rows'])
            np.testing.assert_array_equal(left['labels'], right['labels'])
    return folds, classes, folders, records, sources


def _comparison(records, classes):
    result = _pair(CONTROL, 'C', records[CONTROL], records['C'], classes)
    result['id'] = 'retain_current_content_similarity'
    result['label_zh'] = '完整 C 减去除当前内容相似度项的对照'
    common = result['matched_folds']
    for stage in STAGES:
        if not common:
            continue
        item = result['stages'][stage]
        item['metric_deltas'] = {}
        for metric in SCALARS:
            if not all(metric in records[mode][fold]['record'][stage]
                       for mode in MODES for fold in common):
                continue
            details = [
                dict(fold=fold, reference=float(records[CONTROL][fold]['record'][stage][metric]),
                     candidate=float(records['C'][fold]['record'][stage][metric]),
                     delta=float(records['C'][fold]['record'][stage][metric])
                           - float(records[CONTROL][fold]['record'][stage][metric]))
                for fold in common
            ]
            values = [row['delta'] for row in details]
            item['metric_deltas'][metric] = {
                **_describe(values), 'folds': details,
                'mean_percentage_points': float(np.mean(values) * 100),
                'fold_positive': sum(value > 1e-12 for value in values),
                'fold_zero': sum(abs(value) <= 1e-12 for value in values),
                'fold_negative': sum(value < -1e-12 for value in values),
                'difference_scale': 'raw metric; percentage points are 100 times the difference',
            }
        item['mean_acc_delta_percentage_points'] = 100 * item['mean_acc_delta']
    if common:
        details = [
            dict(fold=fold, reference_acc=records[CONTROL][fold]['mean_last100_acc'],
                 candidate_acc=records['C'][fold]['mean_last100_acc'],
                 acc_delta=records['C'][fold]['mean_last100_acc'] - records[CONTROL][fold]['mean_last100_acc'])
            for fold in common
        ]
        values = [row['acc_delta'] for row in details]
        description = _describe(values)
        result['stages']['last100'] = {
            'mean_acc_delta': description['mean'],
            'mean_acc_delta_percentage_points': 100 * description['mean'],
            'sd_acc_delta': description['sd'],
            'fold_wins': sum(value > 1e-12 for value in values),
            'fold_ties': sum(abs(value) <= 1e-12 for value in values),
            'fold_losses': sum(value < -1e-12 for value in values),
            'folds': details, 'prediction_counts_available': False,
            'note': 'Mean TEST ACC over epochs 901-1000; not an averaged-prediction endpoint.',
        }
    return result


def _csv(out, mode, records):
    fields = ['training_seed', 'fold', 'mode', 'parameters', 'heldout_patients']
    fields += [f'{stage}_{metric}' for stage in STAGES
               for metric in ('epoch', *SCALARS, 'per_class_recall', 'confusion_matrix')]
    fields += ['last100_acc', 'last100_first_epoch', 'last100_final_epoch']
    target = out / f'metrics_{mode}.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold, value in sorted(records.items()):
            row = dict(training_seed=0, fold=fold, mode=mode, parameters=value['parameters'],
                       heldout_patients=value['heldout_patients'], last100_acc=value['mean_last100_acc'],
                       last100_first_epoch=901, last100_final_epoch=1000)
            for stage in STAGES:
                for metric in ('epoch', *SCALARS, 'per_class_recall', 'confusion_matrix'):
                    item = value['record'][stage].get(metric, '')
                    row[f'{stage}_{metric}'] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
            writer.writerow(row)
    temporary.replace(target)


def _percent(value):
    return '未完成' if value is None else f'{100 * value:.4f}%'


def _report(out, result):
    lines = ['去除当前内容相似度项：实验结果说明', '',
             f"新增对照完成 {result['completed_new_runs']}/10 折；原 C 已读取 "
             f"{result['variants']['C']['completed_folds']}/10 折。",
             'seed0，原十折，每折1000轮；主指标沿用首次最高 TEST ACC 的选轮规则。',
             '指标按折等权平均；未全部完成时只报告进度和共同折，不作完整十折收益结论。', '']
    for mode, value in result['variants'].items():
        lines += [f"{NAMES[mode]}：{value['completed_folds']}/10 折，参数 {PARAMETERS[mode]}。"]
        for stage, label in (('best', '最佳选点'), ('last', '第1000轮')):
            score = value.get(stage, {})
            lines.append(f"- {label}：ACC {_percent(score.get('acc'))}，"
                         f"宏F1 {_percent(score.get('macro_f1'))}，AUC {_percent(score.get('auc'))}。")
            if 'per_class_recall' in score:
                lines.append('  折均召回：' + '；'.join(
                    f'{name} {_percent(recall)}' for name, recall in zip(result['class_names'], score['per_class_recall'])) + '。')
                lines.append('  混淆矩阵（已完成折计数之和；行=真实类，列=预测类）：'
                             + json.dumps(score['confusion_matrix_sum'], ensure_ascii=False))
        lines += [f"- 末100轮平均 ACC：{_percent(value.get('mean_last100_acc'))}。", '']
    pair = result['matched_comparisons'][0]
    lines += ['同折配对：差值均为“完整 C 减去除项的对照”，正值表示完整 C 的指标较高，负值表示较低。',
              f"共同完成 {len(pair['matched_folds'])}/10 折；患者行号与标签完全匹配。", '']
    for stage, label in (('best', '最佳选点'), ('last', '第1000轮'), ('last100', '末100轮平均')):
        item = pair['stages'].get(stage)
        if item is None:
            lines.append(f'- {label}：无共同完成折。')
            continue
        lines.append(f"- {label} ACC：{100 * item['mean_acc_delta']:+.4f} 个百分点；"
                     f"{item['fold_wins']} 升／{item['fold_ties']} 平／{item['fold_losses']} 降。")
        if stage != 'last100':
            for metric, name in (('macro_f1', '宏F1'), ('auc', 'AUC')):
                lines.append(f"  {name}：{100 * item['metric_deltas'][metric]['mean']:+.4f} 个百分点。")
            lines.append(f"  纠正 {item['corrected']} 人，新增错误 {item['harmed']} 人，净 {item['net']:+d} 人。")
            for name, values in item['per_class'].items():
                lines.append(f"  {name}：纠正 {values['corrected']}，新增错误 {values['harmed']}，"
                             f"净 {values['net']:+d}；折均召回差 "
                             f"{100 * values['mean_recall_delta']:+.4f} 个百分点。")
    lines += ['', result['conclusion'], '',
              '机制边界：只去除边权中的当前传播内容相似度项，不是去除所有内容依赖。'
              '关系相似度、按关系×内容索引的参数、六张图与36条路径、内容ID及原发送／接收映射保留。'
              '改变边权后按原算子重算度和归一化。tokens仍参与消息内容和整体预测。',
              f'原 C {PARAMETERS["C"]} 参数，对照 {PARAMETERS[CONTROL]} 参数，减少1536个。'
              '这不是严格等参数量或等函数的初始化对照。',
              '本分析校验1000轮完整曲线、首次最高TEST ACC、保存logits的argmax准确率、'
              '混淆矩阵/召回/F1/AUC、模式/种子/参数量、患者配对和输入sha。历史C不要求last.pt。'
              'AUC保留原记录对raw logits计算宏平均OVR的口径，不改用softmax。',
              '末轮和末100轮仅作辅助观察，不事后替换主指标。'
              '这是单种子、历史TEST选点下的描述性匹配比较；不计算p值、CI或独立样本推断。',
              '权重重放、训练来源与代码协议审计由实验运行器另行完成。'
              '纠正/新增错误计数采用患者权重，不能替代按折等权的指标差值。',
              '逐折数据见两份metrics_*.csv；完整配对/各类变化、汇总混淆矩阵及文件哈希见summary.json。', '']
    _text(out / '实验结果说明.txt', '\n'.join(lines))


def summarize(out, plan):
    """Validate completed folds and write two CSVs, summary.json and Chinese text."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    folds, classes, folders, records, sources = _collect(out, plan)
    variants = {mode: _aggregate(values, folds) for mode, values in records.items()}
    for mode, value in variants.items():
        value['expected_parameters'] = PARAMETERS[mode]
        if records[mode]:
            value['last100'] = {'acc': value['mean_last100_acc'], 'epochs': [901, 1000],
                               'description': _describe([row['mean_last100_acc'] for row in records[mode].values()]),
                               'auxiliary_only': True}
            for fold in value['folds']:
                fold['heldout_patients'] = records[mode][fold['fold']]['heldout_patients']
    pair = _comparison(records, classes)
    complete = all(len(records[mode]) == 10 for mode in MODES)
    if not complete:
        conclusion = '两组十折尚未全部完成，不判断去除当前内容相似度项的完整十折收益。'
    else:
        delta = pair['stages']['best']['mean_acc_delta']
        direction = '低于' if delta < -1e-12 else '高于' if delta > 1e-12 else '等于'
        conclusion = (f'完整十折中，完整 C 的主指标最佳 ACC {direction}去除项的对照；'
                      f'折均差 {100 * delta:+.4f} 个百分点。'
                      '这只描述本轮移除该项后的匹配结果，不能建立显著性、稳定优势或因果机制效应。')
    result = dict(
        updated_at=datetime.now(timezone.utc).isoformat(), training_seed=0, split_seed=0,
        planned_folds=folds, planned_new_runs=10, completed_new_runs=len(records[CONTROL]),
        all_new_complete=len(records[CONTROL]) == 10,
        all_reference_complete=len(records['C']) == 10, all_complete=complete, all_two_complete=complete,
        variants=variants, matched_comparisons=[pair], class_names=classes,
        directories={mode: str(path.resolve()) for mode, path in folders.items()},
        reference_directory=str(REFERENCE.resolve()), parameter_counts=PARAMETERS,
        selection='First maximum TEST ACC over 1000 epochs; retained primary endpoint.',
        auxiliary_endpoints=['Epoch 1000 TEST metrics', 'Mean TEST ACC over epochs 901-1000'],
        aggregation='Equal-fold metric means; paired comparisons use matching completed folds only.',
        auc_definition='Historical macro OVR AUC calculated on raw logits, not softmax probabilities.',
        units='ACC/F1/AUC are proportions; displayed percentage-point differences multiply deltas by 100.',
        source_sha256=sources,
        analysis_source_sha256={str(Path(__file__).resolve()): _sha(__file__),
                                str(ROOT / 'analyze_joint_evidence_routing.py'): _sha(ROOT / 'analyze_joint_evidence_routing.py')},
        training_started_by_analysis=False, checkpoint_replay_performed=False,
        sd_note='Descriptive delta SD uses ddof=1; legacy variant std_acc uses ddof=0.',
        conclusion=conclusion,
        caveats=[
            'Removing the current content-similarity term does not remove all content dependence.',
            'The same six graphs, 36 relation-content paths and original S/T mappings are retained.',
            'The control removes 1536 active parameters and recomputes weighted degree/normalization.',
            'Single training seed under the retained historical TEST-selected exploratory protocol.',
            'Paired folds are descriptive, not independent replicates for inference.',
            'Last and last100 do not replace the original best TEST ACC endpoint.',
            'Checkpoint/source/protocol audits are separate runner responsibilities.',
        ],
    )
    for mode in MODES:
        _csv(out, mode, records[mode])
    _dump(out / 'summary.json', result)
    _report(out, result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = json.loads((args.output / 'PLAN.json').read_text(encoding='utf-8-sig'))
    result = summarize(args.output, plan)
    print(json.dumps({'completed_new_runs': result['completed_new_runs'],
                      'all_two_complete': result['all_two_complete']}, ensure_ascii=False))
