"""Read-only metric checks and reports for the joint evidence routing round.

Only the designated output directory receives summary/report files. Historical
records and predictions are never changed; no training or checkpoint replay is
started here. The runner owns the stronger checkpoint and provenance audit.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0/content'
NAMES = {'C': '原 C：内容模态传播', 'own': '自身模态路由', 'joint': '患者与模态联合路由'}
STAGES = ('best', 'last')
SCALARS = ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _parameter_count(value):
    if isinstance(value, dict):
        for key in ('total', 'trainable', 'training'):
            if key in value:
                return int(value[key])
        raise ValueError('Parameter dictionary must identify its total count')
    return int(value)


def _prediction(folder, stage, record):
    filename = 'predictions.npz' if stage == 'best' else 'last_predictions.npz'
    with np.load(folder / filename, allow_pickle=False) as saved:
        logits = np.asarray(saved['logits'])
        labels = np.asarray(saved['labels'])
        rows = np.asarray(saved['row_index'])
    if (logits.ndim != 2 or labels.ndim != 1 or rows.ndim != 1
            or len(logits) != len(labels) or len(rows) != len(labels) or not len(labels)):
        raise ValueError(f'{folder}/{filename}: inconsistent prediction shapes')
    if not np.isfinite(logits).all() or not np.isfinite(labels).all():
        raise ValueError(f'{folder}/{filename}: nonfinite predictions/labels')
    if (not np.array_equal(labels, labels.astype(np.int64))
            or labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError(f'{folder}/{filename}: invalid class labels')
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f'{folder}/{filename}: duplicate patient rows')
    predicted = logits.argmax(-1)
    correct = predicted == labels
    accuracy = float(correct.mean())
    if not np.isclose(accuracy, float(record[stage]['acc']), atol=1e-8, rtol=0):
        raise ValueError(f'{folder}/{filename}: saved predictions disagree with recorded ACC')
    return dict(labels=labels.astype(np.int64), rows=rows, predicted=predicted,
                correct=correct, classes=logits.shape[1], acc=accuracy)


def _record(folder, fold, epochs):
    record = _read(folder / 'complete.json')
    if int(record['fold']) != fold or int(record.get('seed', 0)) != 0:
        raise ValueError(f'{folder}: incorrect fold or training seed')
    predictions = {stage: _prediction(folder, stage, record) for stage in STAGES}
    np.testing.assert_array_equal(predictions['best']['rows'], predictions['last']['rows'])
    np.testing.assert_array_equal(predictions['best']['labels'], predictions['last']['labels'])
    with (folder / 'epochs.csv').open(encoding='utf-8-sig', newline='') as stream:
        curve = list(csv.DictReader(stream))
    if len(curve) != epochs or [int(row['epoch']) for row in curve] != list(range(1, epochs + 1)):
        raise ValueError(f'{folder}: incomplete or out-of-order epoch log')
    accuracy = np.asarray([float(row['test_acc']) for row in curve])
    if not np.isfinite(accuracy).all():
        raise ValueError(f'{folder}: nonfinite epoch accuracy')
    if int(record['best']['epoch']) != int(accuracy.argmax()) + 1:
        raise ValueError(f'{folder}: best checkpoint is not the first maximum TEST ACC')
    if int(record['last']['epoch']) != epochs:
        raise ValueError(f'{folder}: last checkpoint is not the planned final epoch')
    for stage, value in (('best', accuracy.max()), ('last', accuracy[-1])):
        if not np.isclose(value, record[stage]['acc'], atol=1e-8, rtol=0):
            raise ValueError(f'{folder}: epoch curve disagrees with {stage} ACC')
    mean_last100 = float(accuracy[-100:].mean())
    if 'mean_last100_acc' in record and not np.isclose(
            mean_last100, record['mean_last100_acc'], atol=1e-8, rtol=0):
        raise ValueError(f'{folder}: recorded last-100 average disagrees with epoch log')
    return dict(record=record, predictions=predictions, mean_last100_acc=mean_last100,
                parameters=_parameter_count(record['parameters']), folder=str(folder))


def _aggregate(records, folds):
    result = dict(completed_folds=len(records), planned_folds=len(folds),
                  completed_fold_ids=sorted(records))
    if not records:
        return result
    values = list(records.values())
    parameters = sorted({value['parameters'] for value in values})
    result['parameters'] = parameters[0] if len(parameters) == 1 else parameters
    result['seconds_total'] = float(sum(value['record'].get('seconds', 0) for value in values))
    result['mean_last100_acc'] = float(np.mean([value['mean_last100_acc'] for value in values]))
    for stage in STAGES:
        scores = [value['record'][stage] for value in values]
        means = {}
        for key in SCALARS:
            if all(key in score for score in scores):
                vector = np.asarray([score[key] for score in scores], dtype=float)
                if not np.isfinite(vector).all():
                    raise ValueError(f'Nonfinite {stage} {key}')
                means[key] = float(vector.mean())
        means['std_acc'] = float(np.std([score['acc'] for score in scores]))
        for key in ('per_class_recall', 'confusion_matrix'):
            if all(key in score for score in scores):
                array = np.asarray([score[key] for score in scores], dtype=float)
                means[key] = array.mean(axis=0).tolist()
                if key == 'confusion_matrix':
                    means['confusion_matrix_sum'] = array.sum(axis=0).astype(int).tolist()
        means['mean_epoch'] = float(np.mean([score['epoch'] for score in scores]))
        result[stage] = means
    result['mean_acc'] = result['best']['acc']
    result['mean_final_epoch_acc'] = result['last']['acc']
    result['folds'] = [dict(fold=fold, best=records[fold]['record']['best'],
                          last=records[fold]['record']['last'],
                          mean_last100_acc=records[fold]['mean_last100_acc'])
                       for fold in sorted(records)]
    return result


def _pair(reference, candidate, left, right, class_names):
    common = sorted(set(left) & set(right))
    result = dict(reference=reference, candidate=candidate,
                  comparison=f'{candidate} vs {reference}', matched_folds=common,
                  delta_definition='candidate minus reference', stages={})
    if not common:
        return result
    for stage in STAGES:
        corrected = harmed = patients = wins = ties = losses = 0
        deltas, details = [], []
        class_values = {name: dict(corrected=0, harmed=0, net=0, patients=0,
                                  recall_deltas=[]) for name in class_names}
        for fold in common:
            a, b = left[fold]['predictions'][stage], right[fold]['predictions'][stage]
            np.testing.assert_array_equal(a['rows'], b['rows'])
            np.testing.assert_array_equal(a['labels'], b['labels'])
            if a['classes'] != len(class_names) or b['classes'] != len(class_names):
                raise ValueError('Class names and saved prediction columns disagree')
            good, bad = ~a['correct'] & b['correct'], a['correct'] & ~b['correct']
            c, h = int(good.sum()), int(bad.sum())
            corrected += c
            harmed += h
            patients += len(a['labels'])
            wins += c > h
            ties += c == h
            losses += c < h
            delta = b['acc'] - a['acc']
            deltas.append(delta)
            details.append(dict(fold=fold, reference_acc=a['acc'], candidate_acc=b['acc'],
                                acc_delta=delta, corrected=c, harmed=h, net=c-h,
                                patients=len(a['labels'])))
            for index, name in enumerate(class_names):
                mask = a['labels'] == index
                item = class_values[name]
                item['corrected'] += int((good & mask).sum())
                item['harmed'] += int((bad & mask).sum())
                item['patients'] += int(mask.sum())
                if mask.any():
                    item['recall_deltas'].append(float(b['correct'][mask].mean()
                                                       - a['correct'][mask].mean()))
        for item in class_values.values():
            item['net'] = item['corrected'] - item['harmed']
            ds = item.pop('recall_deltas')
            item['folds_with_class'] = len(ds)
            item['mean_recall_delta'] = float(np.mean(ds)) if ds else None
        result['stages'][stage] = dict(mean_acc_delta=float(np.mean(deltas)),
            corrected=corrected, harmed=harmed, net=corrected-harmed, patients=patients,
            fold_wins=int(wins), fold_ties=int(ties), fold_losses=int(losses),
            per_class=class_values, folds=details)
    result['mean_last100_acc_delta'] = float(np.mean([
        right[fold]['mean_last100_acc'] - left[fold]['mean_last100_acc'] for fold in common]))
    return result


def _table(summary):
    lines = ['| 模型 | 完成折数 | 最佳 ACC | 宏 F1 | 末轮 ACC | 末100轮平均 ACC | 参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, value in summary['variants'].items():
        completed = f"{value['completed_folds']}/{value['planned_folds']}"
        if not value['completed_folds']:
            lines.append(f'| {NAMES[mode]} | {completed} | — | — | — | — | — |')
            continue
        macro = value['best'].get('macro_f1')
        macro_text = '—' if macro is None else f'{100*macro:.2f}%'
        lines.append(f"| {NAMES[mode]} | {completed} | {100*value['best']['acc']:.2f}% | "
                     f"{macro_text} | {100*value['last']['acc']:.2f}% | "
                     f"{100*value['mean_last100_acc']:.2f}% | {value['parameters']} |")
    return lines


def summarize(out, plan):
    """Check completed records and update equal-fold summaries; return a dict."""
    out = Path(out)
    folds = [int(fold) for fold in plan.get('folds', range(1, 11))]
    modes = list(plan.get('modes', ('own', 'joint')))
    if modes != ['own', 'joint'] or len(folds) != len(set(folds)):
        raise ValueError('Expected own/joint and unique planned folds')
    epochs = int(plan.get('epochs', 1000))
    classes = list(plan.get('class_names', ('AD', 'CN', 'SMCI')))
    directories = {'C': REFERENCE, **{mode: out / 'seed_0' / mode for mode in modes}}
    records = {mode: {fold: _record(base / f'fold_{fold:02d}', fold, epochs)
                      for fold in folds if (base / f'fold_{fold:02d}' / 'complete.json').exists()}
               for mode, base in directories.items()}
    variants = {mode: _aggregate(items, folds) for mode, items in records.items()}
    comparisons = [_pair(a, b, records[a], records[b], classes)
                   for a, b in (('C', 'own'), ('C', 'joint'), ('own', 'joint'))]
    summary = dict(updated_at=datetime.now(timezone.utc).isoformat(), training_seed=0,
        planned_folds=folds, planned_new_runs=len(folds)*len(modes),
        completed_new_runs=sum(len(records[mode]) for mode in modes),
        all_new_complete=all(len(records[mode]) == len(folds) for mode in modes),
        variants=variants, matched_comparisons=comparisons,
        class_names=classes, reference_directory=str(REFERENCE),
        aggregation='Equal-fold metric means; pooled corrected/harmed counts are descriptive only',
        selection='First maximum TEST ACC under the retained original protocol',
        caveats=['Single training seed, original test-selected exploratory comparison',
                 'Best/last comparisons always use common completed folds with matching patient rows',
                 'C versus routing changes the propagation operator and capacity; not a pure single-factor effect',
                 'Routing weights are not causal explanations or evidence of novelty'])
    _dump(out / 'summary.json', summary)
    lines = ['# 联合证据路由实验：进度与结果', '',
             f"新增训练完成 {summary['completed_new_runs']}/{summary['planned_new_runs']} 次。"
             f'训练种子 0，每折 {epochs} 轮，原划分和首次最高测试 ACC 选点规则保留。', '',
             '各指标按折等权平均。未完成时表中各组可能覆盖不同折数，不据此进行总体排名；'
             '以下配对差值仅使用共同已完成折。', '', *_table(summary), '',
             '配对差值均为候选减参照；纠正和新增错误是匹配患者的计数，不用于替代折均值。', '']
    for comparison in comparisons:
        if not comparison['matched_folds']:
            continue
        for stage, label in (('best', '最佳'), ('last', '末轮')):
            item = comparison['stages'][stage]
            lines.append(f"- {comparison['comparison']}，{len(comparison['matched_folds'])} 折，{label}："
                         f"ACC {100*item['mean_acc_delta']:+.3f} 个百分点；纠正 {item['corrected']}，"
                         f"新增错误 {item['harmed']}，净 {item['net']:+d} 人；"
                         f"{item['fold_wins']} 胜/{item['fold_ties']} 平/{item['fold_losses']} 负。")
    lines += ['', '主指标仍是原协议的最佳 ACC；末轮与末100轮仅用于训练行为诊断。'
              'C 与新路由同时存在算子和参数容量差异。路由权重不等于因果贡献。', '']
    (out / '进度与结果.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def write_report(out, plan, audit):
    """Write the final Chinese report after the caller's audit; return summary."""
    out = Path(out)
    summary = summarize(out, plan)
    if not summary['all_new_complete'] or summary['variants']['C']['completed_folds'] != len(summary['planned_folds']):
        raise ValueError('A final report requires all planned routing and reference folds')
    if audit is None or (isinstance(audit, dict) and (
            audit.get('passed') is False or audit.get('ok') is False
            or audit.get('state') in ('failed', 'error'))):
        raise ValueError('A final report requires the completed successful runner audit')
    own_count = summary['variants']['own']['parameters']
    joint_count = summary['variants']['joint']['parameters']
    if own_count != joint_count:
        raise ValueError('Own and joint routing must have the same parameter count')
    pairs = {(item['reference'], item['candidate']): item for item in summary['matched_comparisons']}
    joint_c = pairs['C', 'joint']['stages']['best']
    joint_own = pairs['own', 'joint']['stages']['best']
    own_c = pairs['C', 'own']['stages']['best']
    eps = 1e-12
    if joint_c['mean_acc_delta'] > eps and joint_own['mean_acc_delta'] > eps:
        conclusion = ('联合路由在本轮最佳 ACC 上同时高于原 C 和自身路由，形成了值得后续复核的候选结果。'
                      '这仍是单种子、原测试选点下的探索结果，不能据此宣称稳定优势或方法创新已经成立。')
    elif joint_own['mean_acc_delta'] > eps and joint_c['mean_acc_delta'] <= eps:
        conclusion = ('加入其他患者来源提高了新路由架构相对自身路由的表现，但尚未超过原 C。'
                      '这支持继续审视跨患者来源在该架构中的作用，尚不足以取代当前性能参照。')
    elif own_c['mean_acc_delta'] > eps and joint_own['mean_acc_delta'] <= eps:
        conclusion = ('自身路由高于原 C，而加入其他患者来源没有进一步提高最佳 ACC。'
                      '当前结果更支持保留自身交互候选，不能把收益归因于跨患者传播。')
    else:
        conclusion = ('本轮没有建立联合路由同时优于原 C 和自身路由的最佳 ACC 优势。'
                      '该版本不能作为已验证有效的主方法，不以路由可视化或末轮改善替代主指标结论。')
    lines = ['# 联合证据路由实验：结果分析', '', conclusion, '',
             f"本轮新增 {summary['completed_new_runs']} 次训练，训练种子 0，"
             f"每折 {int(plan.get('epochs', 1000))} 轮，沿用原十折和首次最高测试 ACC 选点规则。"
             '本报告核对了保存预测、逐折记录、完整训练曲线和患者配对；模型检查点与来源复核由实验运行器完成，'
             '详见 audit.json。', '', *_table(summary), '',
             '## 原协议下的配对结果', '',
             'ACC 差值采用逐折等权平均；净正确人数汇总共同测试患者。两种口径权重不同，'
             '不能把非常小的折均值变化解释为更多患者被正确分类。', '']
    for comparison in summary['matched_comparisons']:
        lines += [f"### {NAMES[comparison['candidate']]} 相对 {NAMES[comparison['reference']]}", '']
        for stage, label in (('best', '最佳检查点'), ('last', '第1000轮')):
            item = comparison['stages'][stage]
            lines.append(f"- {label}：ACC {100*item['mean_acc_delta']:+.4f} 个百分点；"
                         f"纠正 {item['corrected']} 人、新增错误 {item['harmed']} 人、净 {item['net']:+d} 人；"
                         f"逐折 {item['fold_wins']} 胜/{item['fold_ties']} 平/{item['fold_losses']} 负。")
            parts = []
            for name, value in item['per_class'].items():
                recall = value['mean_recall_delta']
                recall_text = '无可比较折' if recall is None else f'{100*recall:+.3f} 个百分点'
                parts.append(f"{name} 净 {value['net']:+d} 人（纠正 {value['corrected']}、"
                             f"新增错误 {value['harmed']}；折均召回 {recall_text}）")
            lines += ['', '  类别变化：' + '；'.join(parts) + '。', '']
        lines += [f"末100轮平均 ACC 差值：{100*comparison['mean_last100_acc_delta']:+.4f} 个百分点。", '']
    lines += ['## 可以和不能据此推断的内容', '',
              '- 自身路由与联合路由参数量相同，比较的是同一路由架构是否开放跨患者来源。'
              '开放来源也会改变归一化竞争与训练轨迹，不能把结果直接称为邻居信息的固有价值。',
              '- 原 C 是在模态 token 上进行早期内容传播、随后进行模态交互和类别查询读出的模型。'
              '新模型与 C 的传播算子和容量不同，C 的差值不是一个独立因素的因果效应。',
              '- 末轮、末100轮、宏 F1 和各类别召回用于补充判断；原主指标仍为首次最高测试 ACC。'
              '不能因末轮改善就把最佳分类效果持平或下降宣布为成功。',
              '- 路由权重只反映模型内部的分配，不能当作因果解释、患者真实生物关系或独有互补信息的证明。',
              '- 本轮仅使用训练种子 0；没有新增独立测试集，也没有建立统计显著性或新颖性保证。'
              '不自动追加实验或改变实验协议。', '',
              '逐折指标、阶段配对纠错、类别变化及参数量见 summary.json；'
              '完整检查点核验见 audit.json，固定实验定义见 PLAN.json。', '']
    (out / '结果分析.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
