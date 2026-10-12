"""CPU-only report for the completed, independently audited ABIDE-5 search.

Writes only the consolidated Markdown/JSON after checking frozen sources,
references, configuration hashes, audited artifacts, and all complete OOF runs.
Does not import torch, model code, experiment drivers, or GPU libraries.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT/'Result/abide5_fine_tune_20260928'
REF = ROOT/'Result/c_abide5_seed0_20260924'
OUTPUT_STEM = ROOT/'Result/ABIDE-5精细调参汇总_20260928'
FOLDS = list(range(1, 11))
SCREEN = [1, 3, 6, 8]
REMAINING = [2, 4, 5, 7, 9, 10]
METRICS = ('acc', 'macro_f1', 'balanced_acc', 'ads_recall', 'cn_recall', 'ads_auc', 'ads_ap')
PARAMETERS = ('lr', 'weight_decay', 'Drop_rate', 'input_noise_std',
              'aux_loss_weight', 'Loss_rate', 'graph_weight_decay')
INPUT_HASHES = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    actual = digest(path)
    if expected is not None:
        require(actual == expected, f'Frozen/audited input changed: {key}')
    if key in INPUT_HASHES:
        require(INPUT_HASHES[key] == actual, f'Input changed between reads: {key}')
    INPUT_HASHES[key] = actual


def read(path):
    pin(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(actual, expected, context):
    require(np.isfinite(actual) and np.isfinite(expected)
            and abs(float(actual)-float(expected)) <= 1e-12,
            f'{context}: {actual!r} != {expected!r}')


def equivalent(actual, expected, context):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{context}: dictionary keys')
        for key in expected:
            equivalent(actual[key], expected[key], f'{context}.{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{context}: list length')
        for index, value in enumerate(expected):
            equivalent(actual[index], value, f'{context}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        number(actual, expected, context)
    else:
        require(actual == expected, f'{context}: value differs')


def verify_manifest(manifest):
    for relative, expected in manifest.items():
        pin(ROOT/relative, expected)


def calculate(logits, labels):
    """Integer confusion metrics; ADS-positive softmax AUC/AP, with ties."""
    z, y = np.asarray(logits), np.asarray(labels)
    require(z.shape == (len(y), 2) and np.isfinite(z).all(), 'Invalid binary logits')
    require(np.array_equal(np.unique(y), [0, 1]), 'Both ADS and CN must be present')
    predicted = z.argmax(1)
    matrix = np.bincount(2*y.astype(np.int64)+predicted, minlength=4).reshape(2, 2)
    tp = matrix.diagonal().astype(float)
    recall = tp/matrix.sum(1)
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros(2), where=denominator != 0)
    exponential = np.exp(z-z.max(1, keepdims=True))
    probability = exponential[:, 0]/exponential.sum(1)
    positive, negative = probability[y == 0], probability[y == 1]
    auc = float(((positive[:, None] > negative[None, :]).sum()
        + .5*(positive[:, None] == negative[None, :]).sum())/(len(positive)*len(negative)))
    order = np.argsort(-probability, kind='stable')
    score, truth = probability[order], (y[order] == 0).astype(int)
    ends = np.flatnonzero(np.r_[score[1:] != score[:-1], True])
    cumulative_tp = np.cumsum(truth)[ends]
    gained_recall = (cumulative_tp-np.r_[0, cumulative_tp[:-1]])/len(positive)
    ap = float(np.sum(gained_recall*cumulative_tp/(ends+1)))
    return dict(acc=float(tp.sum()/len(y)), macro_f1=float(f1.mean()),
        balanced_acc=float(recall.mean()), ads_recall=float(recall[0]), cn_recall=float(recall[1]),
        ads_auc=auc, ads_ap=ap, confusion_matrix=matrix.tolist())


def check_metrics(actual, expected, context):
    for key in METRICS:
        number(actual[key], expected[key], f'{context}: {key}')
    if 'confusion_matrix' in expected:
        require(actual['confusion_matrix'] == expected['confusion_matrix'], f'{context}: confusion matrix')


def load_oof(path):
    pin(path)
    with np.load(path) as saved:
        require(set(saved.files) == {'labels', 'logits', 'row_index', 'fold'}, f'OOF schema: {path}')
        return {key: saved[key].copy() for key in saved.files}


def folder_for(name, fold):
    return (REF/'content' if name == 'BASE' else OUT/'runs'/name)/f'fold_{fold:02d}'


def load_and_verify():
    require((OUT/'COMPLETE.json').is_file(), 'Training is incomplete')
    require(not (OUT/'RUNNING.lock').exists(), 'Training lock remains')
    require((OUT/'independent_audit.json').is_file(), 'Independent audit has not completed')
    artifacts = {key: read(OUT/f'{key}.json') for key in (
        'PLAN', 'COMPLETE', 'independent_audit', 'analysis', 'screen_vs_remaining',
        'best_configuration', 'all_settings', 'selection', 'refinement', 'summary', 'config_sha256')}
    plan, completion, audit = (artifacts[key] for key in ('PLAN', 'COMPLETE', 'independent_audit'))
    analysis, best = artifacts['analysis'], artifacts['best_configuration']
    require(completion['all_audits_passed'] is True and audit['passed'] is True, 'An audit failed')
    retained = analysis['retained']
    require(completion['retained'] == audit['retained'] == best['name'] == retained, 'Winner identity differs')
    equivalent(analysis['full'], audit['full'], 'Audited full metrics')
    equivalent(analysis['pairs'], audit['pairs'], 'Audited paired metrics')
    equivalent(artifacts['screen_vs_remaining'], analysis['screen_vs_remaining'], 'Screen/remaining artifact')
    equivalent(artifacts['screen_vs_remaining'], audit['screen_vs_remaining'], 'Audited screen/remaining metrics')
    require(plan['samples'] == audit['samples'] == 864, 'This is not the 864-row ABIDE-5 task')
    require(plan['counts'] == audit['class_counts'] == [397, 467], 'ABIDE-5 class counts changed')
    require(plan['parameters'] == audit['parameters'] == best['parameters'] == 768348, 'C parameter count changed')
    require(plan['seed'] == best['seed'] == 0 and plan['epochs'] == best['epochs'] == 1000, 'Protocol changed')
    require(plan['screen_folds'] == SCREEN and plan['remaining_folds'] == REMAINING, 'Screen changed')
    require(plan['baseline_protocol']['original_config']['DATA_SET'] == 'ABIDE-5', 'Wrong dataset')
    require(plan['baseline_protocol']['class_names'] == ['ADS', 'CN'], 'Wrong class index order')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    pin(ROOT/'audit_abide5_fine_tune.py', audit['auditor_sha256'])
    for filename, expected in audit['artifact_sha256'].items():
        pin(OUT/filename, expected)
    pin(OUT/'data_splits_graphs.npz', plan['split_archive_sha256'])
    settings = artifacts['all_settings']
    require(set(artifacts['config_sha256']) == set(settings), 'Config hash coverage changed')
    for name, expected in artifacts['config_sha256'].items():
        pin(OUT/'configs'/f'{name}.ini', expected)
        require(set(settings[name]) == set(PARAMETERS), f'{name}: unexpected tuned parameter')
    require(best['overrides'] == settings[retained] == analysis['settings'], 'Exported parameter values differ')
    require(Path(best['config']).resolve() == OUT/'configs'/f'{retained}.ini', 'Exported INI path differs')
    require(best['driver'] == 'abide5_fine_tune.py' and best['k'] == 10, 'Wrong reproduction entry')
    number(best['graph_weight_decay'], settings[retained]['graph_weight_decay'], 'Exported graph decay')
    np.testing.assert_allclose(best['effective_noise_sd'], [3*settings[retained]['input_noise_std']]*5,
                               atol=1e-12, rtol=0)
    candidates = [name for name in settings if name != 'BASE']
    finalists = artifacts['selection']['finalists']
    expected_runs = 4*len(candidates)+6*len(finalists)
    require(completion['new_runs'] == audit['new_runs'] == analysis['new_runs'] == expected_runs,
            'Training count differs')
    require(completion['verified_new_checkpoints'] == analysis['verified_new_checkpoints']
            == audit['replayed_new_checkpoints'] == 2*expected_runs, 'New replay count differs')
    require(audit['replayed_baseline_best_checkpoints'] == audit['baseline_last_prediction_files_verified'] == 10,
            'Historical baseline audit count differs')
    require(audit['prediction_files_verified'] == 2*(expected_runs+10), 'Prediction file count differs')
    require(set(audit['first_three_training_replays']) == set(settings), 'First-three replay coverage differs')
    require(all(record['passed'] is True for record in audit['first_three_training_replays'].values()),
            'First-three replay failed')
    return artifacts


def assemble(artifacts):
    analysis, settings, audit = (artifacts[key] for key in ('analysis', 'all_settings', 'independent_audit'))
    retained = analysis['retained']
    names = [name for name in settings if name == 'BASE' or name in artifacts['selection']['finalists']]
    require(list(analysis['full']) == names, 'Full configuration order/coverage differs')
    with np.load(OUT/'data_splits_graphs.npz') as saved:
        source_rows, source_labels, masks = (saved[key].copy() for key in ('source_row', 'labels', 'test_masks'))
    require(len(source_rows) == len(np.unique(source_rows)) == 864, 'Source row identity is not unique')
    np.testing.assert_array_equal(np.bincount(source_labels, minlength=2), [397, 467])
    np.testing.assert_array_equal(masks.sum(0), np.ones(864, dtype=int))
    oofs = {name: load_oof(OUT/'oof'/f'{name}.npz') for name in names}
    baseline = oofs['BASE']
    rows_by_key = {(row['name'], row['fold']): row for row in analysis['records']}
    require(len(rows_by_key) == len(analysis['records']) == 10*len(names), 'Fold record coverage differs')
    full_configs = []
    for name in names:
        oof = oofs[name]
        require(len(oof['row_index']) == len(np.unique(oof['row_index'])) == 864, f'{name}: OOF row count')
        np.testing.assert_array_equal(np.sort(oof['row_index']), np.sort(source_rows))
        np.testing.assert_array_equal(oof['row_index'], baseline['row_index'])
        np.testing.assert_array_equal(oof['labels'], baseline['labels'])
        np.testing.assert_array_equal(oof['fold'], baseline['fold'])
        fold_records, best_metrics, last_metrics = [], [], []
        for fold in FOLDS:
            selected, mask = oof['fold'] == fold, masks[fold-1]
            np.testing.assert_array_equal(oof['row_index'][selected], source_rows[mask])
            np.testing.assert_array_equal(oof['labels'][selected], source_labels[mask])
            folder = folder_for(name, fold)
            record = read(folder/'complete.json')
            require(record['parameters'] == 768348 and record['fold'] == fold, f'{name}/{fold}: model or fold changed')
            pin(folder/'epochs.csv')
            with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
                curve = list(csv.DictReader(stream))
            require([int(row['epoch']) for row in curve] == list(range(1, 1001)), f'{name}/{fold}: incomplete epochs')
            acc = np.asarray([float(row['test_acc']) for row in curve])
            require(np.isfinite(acc).all(), f'{name}/{fold}: nonfinite curve')
            first_best = int(acc.argmax())+1
            require(first_best == record['best']['epoch'], f'{name}/{fold}: checkpoint rule changed')
            detailed = rows_by_key[name, fold]
            require(detailed['best_epoch'] == first_best, f'{name}/{fold}: reported best epoch')
            measured = calculate(oof['logits'][selected], oof['labels'][selected])
            check_metrics(measured, detailed['best'], f'{name}/{fold}: best')
            number(measured['acc'], acc[first_best-1], f'{name}/{fold}: curve maximum')
            number(measured['acc'], record['best']['acc'], f'{name}/{fold}: recorded maximum')
            best_file = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
            pin(folder/best_file)
            with np.load(folder/best_file) as saved:
                for key in ('labels', 'logits', 'row_index'):
                    np.testing.assert_array_equal(oof[key][selected], saved[key])
            pin(folder/'last_predictions.npz')
            with np.load(folder/'last_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['labels'], source_labels[mask])
                if 'row_index' in saved:
                    np.testing.assert_array_equal(saved['row_index'], source_rows[mask])
                last = calculate(saved['logits'], saved['labels'])
            check_metrics(last, detailed['last'], f'{name}/{fold}: last')
            number(last['acc'], acc[-1], f'{name}/{fold}: last curve')
            number(float(acc[-100:].mean()), detailed['late100_acc'], f'{name}/{fold}: last100')
            best_metrics.append(measured)
            last_metrics.append(last)
            fold_records.append(dict(fold=fold, samples=int(selected.sum()), best_epoch=first_best,
                best=measured, last=last, late100_acc=float(acc[-100:].mean())))
        full = analysis['full'][name]
        require(full['folds'] == FOLDS, f'{name}: incomplete configuration called tenfold')
        for stage, measured in (('best', best_metrics), ('last', last_metrics)):
            for key in METRICS:
                number(float(np.mean([row[key] for row in measured])), full[stage][key], f'{name}: {stage} mean {key}')
        number(float(np.std([row['acc'] for row in best_metrics])), full['acc_std'], f'{name}: fold ACC std')
        number(float(np.mean([row['late100_acc'] for row in fold_records])), full['late100_acc'],
               f'{name}: mean late100 ACC')
        pooled = calculate(oof['logits'], oof['labels'])
        check_metrics(pooled, analysis['pairs'][name]['pooled'], f'{name}: pooled')
        correct = oof['logits'].argmax(1) == oof['labels']
        old_correct = baseline['logits'].argmax(1) == baseline['labels']
        corrected, harmed = correct & ~old_correct, ~correct & old_correct
        number(int(corrected.sum()), analysis['pairs'][name]['corrected'], f'{name}: corrected')
        number(int(harmed.sum()), analysis['pairs'][name]['harmed'], f'{name}: harmed')
        deltas = [int(correct[oof['fold'] == fold].sum())-int(old_correct[oof['fold'] == fold].sum())
                  for fold in FOLDS]
        for key, count in (('wins', sum(delta > 0 for delta in deltas)),
                           ('ties', sum(delta == 0 for delta in deltas)),
                           ('losses', sum(delta < 0 for delta in deltas))):
            number(count, analysis['pairs'][name][key], f'{name}: fold {key}')
        full_configs.append(dict(name=name, settings=settings[name], best=full['best'], last=full['last'],
            late100_acc=full['late100_acc'], acc_std=full['acc_std'], pooled=pooled,
            pairs=analysis['pairs'][name], folds=fold_records, screen_vs_remaining=analysis['screen_vs_remaining'][name]))
    def rank_key(name):
        full = analysis['full'][name]
        return tuple(round(value, 12) for value in (full['best']['acc'], full['best']['macro_f1'],
                     full['best']['balanced_acc'], full['late100_acc']))
    full_order = sorted(names, key=rank_key, reverse=True)
    require(full_order == analysis['full_order'] and retained == full_order[0], 'Best complete configuration ranking differs')
    final_export = load_oof(OUT/'retained_oof_predictions.npz')
    for key in final_export:
        np.testing.assert_array_equal(final_export[key], oofs[retained][key])
    final = oofs[retained]
    base_pred, final_pred = baseline['logits'].argmax(1), final['logits'].argmax(1)
    old_correct, correct = base_pred == baseline['labels'], final_pred == final['labels']
    corrected, harmed = correct & ~old_correct, ~correct & old_correct
    by_name = {row['name']: row for row in full_configs}
    comparison = dict(base='BASE', retained=retained,
        best_difference_pp={key: 100*(analysis['full'][retained]['best'][key]-analysis['full']['BASE']['best'][key]) for key in METRICS},
        corrected=int(corrected.sum()), harmed=int(harmed.sum()), net_corrected=int(corrected.sum()-harmed.sum()),
        per_class={}, fold_changes=[], changed_rows=[])
    for label, title in enumerate(('ADS', 'CN')):
        mask = final['labels'] == label
        comparison['per_class'][title] = dict(samples=int(mask.sum()), original_errors=int((~old_correct & mask).sum()),
            final_errors=int((~correct & mask).sum()), corrected=int((corrected & mask).sum()), harmed=int((harmed & mask).sum()))
    for old, new in zip(by_name['BASE']['folds'], by_name[retained]['folds']):
        selected = final['fold'] == old['fold']
        comparison['fold_changes'].append(dict(fold=old['fold'], samples=old['samples'],
            original_best_epoch=old['best_epoch'], final_best_epoch=new['best_epoch'], original_acc=old['best']['acc'],
            final_acc=new['best']['acc'], delta_pp=100*(new['best']['acc']-old['best']['acc']),
            final_last_acc=new['last']['acc'], corrected=int((corrected & selected).sum()), harmed=int((harmed & selected).sum())))
    for index in np.flatnonzero(corrected | harmed):
        comparison['changed_rows'].append(dict(row_index=int(final['row_index'][index]), fold=int(final['fold'][index]),
            label=int(final['labels'][index]), original_prediction=int(base_pred[index]), final_prediction=int(final_pred[index]),
            change='corrected' if corrected[index] else 'harmed'))
    differences = {key: dict(original=settings['BASE'][key], retained=settings[retained][key]) for key in PARAMETERS
                   if settings['BASE'][key] != settings[retained][key]}
    screening = []
    for name, value in artifacts['summary']['candidates'].items():
        require(name in settings, f'Unknown summary candidate: {name}')
        expected_folds = FOLDS if name in names else SCREEN
        require(value['folds'] == expected_folds, f'{name}: unexpected completed folds')
        if name in names:
            value = analysis['screen_vs_remaining'][name]['screen']
        screening.append(dict(name=name, screen_folds=SCREEN, screen_best=value['best'],
            completed_folds=expected_folds, full_tenfold=name in names, settings=settings[name]))
    require(len(screening) == len(settings), 'Missing screening candidate')
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__),
        task='ABIDE-5 ADS/CN', class_index={'ADS': 0, 'CN': 1}, samples=864, class_counts={'ADS': 397, 'CN': 467},
        features=470, modalities={'PHENO': 48, 'ANAT': 6, 'FUNC': 10, 'MRI': 150, 'fMRI': 256},
        model='C: RelationSheafModel(mode=content)', parameters=768348, seed=0, folds=10, epochs_per_fold=1000,
        retained=retained, full_order=full_order, full_configurations=full_configs, comparison=comparison,
        parameter_changes=differences, original_settings=settings['BASE'], retained_settings=settings[retained],
        candidate_count=len(settings)-1, coarse_count=len(artifacts['PLAN']['coarse']),
        refinement_count=len(artifacts['refinement']['candidates']), refinements=artifacts['refinement'],
        finalists=artifacts['selection']['finalists'], screening=screening,
        new_training_runs=audit['new_runs'], independent_new_checkpoint_replays=audit['replayed_new_checkpoints'],
        independent_baseline_checkpoint_replays=10, independent_baseline_last_prediction_files=10,
        independent_prediction_files=audit['prediction_files_verified'],
        first_three_replayed_configurations=len(audit['first_three_training_replays']),
        reproduction=artifacts['best_configuration'],
        limitations=['Best results select the first highest TEST-ACC epoch among 1000; they are not last-epoch scores.',
            'Hyperparameter search repeatedly reuses the same test folds; verification of execution does not provide independent generalization evidence.',
            'Only seed0 was tested. This is best within the completed search, not a global optimum or multi-seed confirmation.',
            'Fourfold screening scores are not complete-tenfold scores; the remaining six folds retain test-based checkpoint selection.',
            'ABIDE-5 overlaps ABIDE and is not an established independent external cohort; their metrics must not be interchanged.',
            'Mean tenfold metrics and pooled 864-row metrics have different denominators and need not be identical.',
            'Unique CSV row indices do not establish independent clinical patient IDs.',
            'Historical BASE has no last.pt; its last predictions were verified without final-weight replay.'],
        input_sha256=dict(INPUT_HASHES))


def link(path, label):
    return f'[{label}](<{Path(path).as_posix()}>)'


def markdown(report):
    percent = lambda value: f'{100*value:.4f}%'
    title = {'acc': 'ACC', 'macro_f1': '宏F1', 'balanced_acc': '平衡ACC', 'ads_recall': 'ADS召回',
             'cn_recall': 'CN召回', 'ads_auc': 'ADS概率AUC', 'ads_ap': 'ADS平均精确率AP'}
    by_name = {row['name']: row for row in report['full_configurations']}
    old, final, compare = by_name['BASE'], by_name[report['retained']], report['comparison']
    lines = ['# ABIDE-5：C 框架精细调参汇总', '',
        f"最终保留 **{report['retained']}**。完整十折最佳ACC由 **{percent(old['best']['acc'])}** 变为 **{percent(final['best']['acc'])}**，变化 **{compare['best_difference_pp']['acc']:+.4f} 个百分点**。864行逐一对齐，纠正原错误 **{compare['corrected']}例**、新增错误 **{compare['harmed']}例**，净变化 **{compare['net_corrected']:+d}例**。", '',
        '本报告对应ABIDE-5 ADS/CN：864例（ADS 397、CN 467），470个特征，PHENO48／ANAT6／FUNC10／MRI150／fMRI256。C网络仍为768348个参数；seed0、原十折、每折1000轮。这里的ADS沿用项目标签名，类别索引为ADS=0、CN=1。', '',
        '**“最佳”表示每折1000轮中首次达到最高测试ACC的检查点，不表示第1000轮。** 下列十折指标均为各折等权均值。', '',
        '## 原配置与最终配置', '', '| 十折均值 | BASE最佳 | 最终最佳 | 变化（百分点） | BASE第1000轮 | 最终第1000轮 |',
        '|---|---:|---:|---:|---:|---:|']
    for key in METRICS:
        lines.append(f"| {title[key]} | {percent(old['best'][key])} | {percent(final['best'][key])} | {compare['best_difference_pp'][key]:+.4f} | {percent(old['last'][key])} | {percent(final['last'][key])} |")
    lines += ['', f"最后100轮平均ACC：BASE为{percent(old['late100_acc'])}，最终配置为{percent(final['late100_acc'])}。这与‘最佳选轮’和‘第1000轮’是三个不同口径。", '',
        '## 全部完成十折的配置', '', '| 配置 | 最佳ACC | 宏F1 | 平衡ACC | ADS召回 | CN召回 | ADS概率AUC | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in report['full_order']:
        row = by_name[name]
        values = [row['best'][key] for key in ('acc', 'macro_f1', 'balanced_acc', 'ads_recall', 'cn_recall', 'ads_auc')]+[row['last']['acc']]
        lines.append('| '+name+' | '+' | '.join(percent(value) for value in values)+' |')
    lines += ['', '最终以同一配置覆盖全部十折，未按折拼接不同配置。排序依次使用十折最佳ACC、宏F1、平衡ACC、最后100轮平均ACC（各四舍五入12位）；全项并列时保留候选原始插入顺序。', '',
        '## 实际参数变化', '', '| 参数 | BASE | 最终配置 | 是否改变 |', '|---|---:|---:|---|']
    for key in PARAMETERS:
        a, b = report['original_settings'][key], report['retained_settings'][key]
        lines.append(f"| `{key}` | {a:g} | {b:g} | {'是' if a != b else '否'} |")
    lines += ['', f"本轮保留配置改变{len(report['parameter_changes'])}项训练参数。输入噪声配置`input_noise_std`是缩放参数：五个模态的实际训练噪声标准差均为它的3倍，BASE为{3*report['original_settings']['input_noise_std']:g}，最终为{3*report['retained_settings']['input_noise_std']:g}。", '',
        'C网络、隐藏维度96、Transformer层数5、头数4、自身分支维度64、五个模态、原始已处理数据、固定cosine top10双向并集图、类别权重、标签平滑0.05均保持不变。优化器仍为Adam、梯度裁剪1、前20轮恒定学习率后余弦调度至最低学习率0.0001。', '',
        '患者分支衰减与图分支衰减分别记录；图分支范围为`propagation.*`但不含`output_norm`。旧配置读取器不会自动使用INI内的`graph_weight_decay`，复现必须通过本次驱动的`configure`和`retained_optimizer`显式应用该值，不能仅把INI交给旧入口。', '',
        link(report['reproduction']['config'], '最终INI')+'；'+link(OUT/'best_configuration.json', '完整复现配置')+'；'+link(ROOT/'abide5_fine_tune.py', '本次实验驱动')+'。', '',
        '## 各折改善与最佳轮次', '', '| 折 | 样本数 | BASE最佳轮次 | 最终最佳轮次 | BASE最佳ACC | 最终最佳ACC | 变化（百分点） | 最终第1000轮ACC | 纠正／新增错误 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in compare['fold_changes']:
        lines.append(f"| {row['fold']} | {row['samples']} | {row['original_best_epoch']} | {row['final_best_epoch']} | {percent(row['original_acc'])} | {percent(row['final_acc'])} | {row['delta_pp']:+.4f} | {percent(row['final_last_acc'])} | {row['corrected']}／{row['harmed']} |")
    p = final['pairs']
    lines += ['', f"相对BASE，逐折ACC为{p['wins']}折提高、{p['ties']}折持平、{p['losses']}折下降。", '',
        '## 864行合并预测与错误转移', '',
        '混淆矩阵行是真实类别、列是预测类别。合并指标直接以864行计算，不与前面的十折等权均值混用。', '',
        '| 配置 | ADS→ADS | ADS→CN | CN→ADS | CN→CN | 合并ACC | 合并ADS召回 | 合并CN召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in (('原BASE', old), ('最终'+report['retained'], final)):
        m, cm = row['pooled'], row['pooled']['confusion_matrix']
        lines.append(f"| {name} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} | {percent(m['acc'])} | {percent(m['ads_recall'])} | {percent(m['cn_recall'])} |")
    lines += ['', '| 真实类别 | 样本数 | BASE错误 | 最终错误 | 纠正 | 新增错误 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in compare['per_class'].items():
        lines.append(f"| {name} | {row['samples']} | {row['original_errors']} | {row['final_errors']} | {row['corrected']} | {row['harmed']} |")
    lines += ['', '## 搜索与复核完成情况', '',
        f"完成{report['coarse_count']}组单因素粗搜和{report['refinement_count']}组去重后的局部精调，共{report['candidate_count']}组非BASE候选；每组先在原第1、3、6、8折完整训练1000轮。前两名及第二名主ACC并列者补齐剩余六折，共{len(report['finalists'])}组晋级。", '',
        f"新增 **{report['new_training_runs']}次1000轮训练**，**{report['independent_new_checkpoint_replays']}个新best／last检查点**完成独立重放。另重放原BASE的10个best检查点，并核查其10个末轮预测文件；BASE历史上没有last.pt。合计核查{report['independent_prediction_files']}份预测文件，并重放{report['first_three_replayed_configurations']}套配置的首3步，核对初始化、随机状态、实际噪声与优化器分组。", '',
        '训练数公式为“非BASE候选数×4＋晋级候选数×6”。预检首3步和检查点推理重放不计为新增完整训练。', '',
        '| 候选 | 四折筛选ACC | 四折宏F1 | 完成折数 | 是否完整十折 |', '|---|---:|---:|---:|---|']
    for row in report['screening']:
        lines.append(f"| {row['name']} | {percent(row['screen_best']['acc'])} | {percent(row['screen_best']['macro_f1'])} | {len(row['completed_folds'])} | {'是' if row['full_tenfold'] else '否'} |")
    lines += ['', '四折筛选成绩不能写成完整十折成绩；未晋级候选没有完整十折结论。', '',
        '| 完整配置 | 筛选四折ACC | 相对BASE变化（百分点） | 剩余六折ACC | 相对BASE变化（百分点） |',
        '|---|---:|---:|---:|---:|']
    for row in report['full_configurations']:
        s = row['screen_vs_remaining']
        lines.append(f"| {row['name']} | {percent(s['screen']['best']['acc'])} | {100*s['screen_delta_acc']:+.4f} | {percent(s['remaining']['best']['acc'])} | {100*s['remaining_delta_acc']:+.4f} |")
    lines += ['', '## 结论边界与可复核记录', '',
        '最终配置是当前搜索范围内、完成完整十折后按既定规则保留的最佳配置，不代表已达到全局最优或性能上限。当前仅seed0，尚无多种子确认。', '',
        '沿用原协议：测试折参与最佳轮次选择，调参也重复使用测试折；剩余六折仍采用测试ACC选轮，不能称为独立最终验证。独立审计核实执行、参数、检查点和指标的一致性，不等同于独立泛化确认。', '',
        'ABIDE-5与ABIDE是高度重叠的样本／特征版本，不能混用二者成绩，也不能把本结果称为独立外部队列验证。数据行唯一性不能代替临床患者ID独立性核查。', '',
        '- '+link(OUT/'PLAN.json', '冻结实验计划')+'；'+link(OUT/'analysis.json', '全部完整结果')+'。',
        '- '+link(OUT/'independent_audit.json', '独立审计记录')+'；'+link(OUT/'screen_vs_remaining.json', '筛选四折与剩余六折')+'。',
        '- '+link(OUTPUT_STEM.with_suffix('.json'), '机器可读汇总、逐行纠错、参数及输入哈希')+'。']
    return '\n'.join(lines)+'\n'


def main():
    artifacts = load_and_verify()
    report = assemble(artifacts)
    text = markdown(report)
    require(not (OUT/'RUNNING.lock').exists(), 'Training restarted while assembling the report')
    for relative, expected in INPUT_HASHES.items():
        require(digest(ROOT/relative) == expected, f'Input changed during summary: {relative}')
    for suffix, content in (('.md', text), ('.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))):
        target = OUTPUT_STEM.with_suffix(suffix)
        temporary = target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(target)
    base = next(row for row in report['full_configurations'] if row['name'] == 'BASE')
    final = next(row for row in report['full_configurations'] if row['name'] == report['retained'])
    print(json.dumps(dict(passed=True, retained=report['retained'], baseline_acc=base['best']['acc'],
        final_acc=final['best']['acc'], new_training_runs=report['new_training_runs'],
        markdown=str(OUTPUT_STEM.with_suffix('.md')), json=str(OUTPUT_STEM.with_suffix('.json'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
