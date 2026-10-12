"""CPU-only, independently calculated summary after all three TAD searches pass audit.

Reads immutable experiment artifacts and writes only the consolidated Markdown
and JSON reports. Does not import torch, a model, a runner, or any GPU package.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT/'Result'
ORIGINAL = RESULT/'c_tadpole_binary_seed0_20260924'
ROUND_PATHS = (
    RESULT/'tad_binary_fine_tune_20260928',
    RESULT/'tad_binary_joint_tune_20260928',
    RESULT/'tad_binary_loss_tune_20260928',
)
OUTPUT_STEM = RESULT/'TAD二分类调参汇总_20260928'
FOLDS = list(range(1, 11))
METRIC_KEYS = ('acc', 'macro_f1', 'pmci_recall', 'smci_recall', 'pmci_auc')
PARAMETER_KEYS = ('lr', 'weight_decay', 'Drop_rate', 'input_noise_std',
                  'aux_loss_weight', 'Loss_rate', 'label_smoothing')
INPUT_HASHES = {}
FOLD_CACHE = {}
VERIFIED_FROZEN_HASHES = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path)
    relative, current = str(path.relative_to(ROOT)), digest(path)
    if relative in INPUT_HASHES:
        require(INPUT_HASHES[relative] == current, f'Input changed between reads:{relative}')
    INPUT_HASHES[relative] = current


def read(path):
    pin(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_frozen(manifest):
    for relative, expected in manifest.items():
        if relative not in VERIFIED_FROZEN_HASHES:
            VERIFIED_FROZEN_HASHES[relative] = digest(ROOT/relative)
        require(VERIFIED_FROZEN_HASHES[relative] == expected, f'Frozen source/reference changed:{relative}')
        INPUT_HASHES[relative] = expected


def same_number(left, right, context):
    require(np.isfinite(left) and np.isfinite(right) and abs(float(left)-float(right)) <= 1e-12,
            f'{context}: {left!r} != {right!r}')


def normalized_settings(spec):
    result = {**spec, 'label_smoothing': spec.get('label_smoothing', .05)}
    require(set(result) == set(PARAMETER_KEYS), 'Unexpected parameter keys')
    return {key: float(result[key]) for key in PARAMETER_KEYS}


def load_rounds():
    rounds = []
    # Check every completion/lock before reading detailed results. A completed
    # training run without a completed independent audit is not sufficient.
    for path in ROUND_PATHS:
        require((path/'COMPLETE.json').is_file(), f'Search is incomplete:{path}')
        require(not (path/'RUNNING.lock').exists(), f'Search is still locked:{path}')
        require((path/'independent_audit.json').is_file(), f'Independent audit missing:{path}')
    for number, path in enumerate(ROUND_PATHS, 1):
        row = {key: read(path/f'{key}.json') for key in
               ('PLAN', 'all_settings', 'analysis', 'summary', 'selection', 'COMPLETE',
                'independent_audit', 'best_configuration')}
        require(row['COMPLETE']['all_audits_passed'] is True and row['independent_audit']['passed'] is True,
                f'Failed completion or independent audit:{path}')
        retained = row['analysis']['retained']
        require(row['COMPLETE']['retained'] == row['independent_audit']['retained']
                == row['best_configuration']['name'] == retained, f'Winner identity differs:{path}')
        require(row['analysis']['full'] == row['independent_audit']['full'], f'Audited metrics changed:{path}')
        require(row['analysis']['new_runs'] == row['independent_audit']['new_runs']
                == row['COMPLETE']['new_runs'], f'Training count differs:{path}')
        verify_frozen(row['PLAN']['code_sha256'])
        verify_frozen(row['PLAN']['reference_sha256'])
        for filename, expected in row['independent_audit']['artifact_sha256'].items():
            require(digest(path/filename) == expected, f'Audited artifact changed:{path/filename}')
        row.update(number=number, path=path)
        rounds.append(row)
    require(rounds[2]['PLAN']['anchor']['name'] == rounds[1]['analysis']['retained'],
            'Third-round anchor is not the second-round winner')
    require(rounds[2]['PLAN']['anchor']['tenfold_metrics']
            == rounds[1]['analysis']['full'][rounds[1]['analysis']['retained']], 'Anchor metrics disagree')
    return rounds


def resolve(round_record, name, fold):
    number, plan = round_record['number'], round_record['PLAN']
    if number == 3:
        if name == 'ANCHOR':
            return ROOT/plan['anchor']['folders'][str(fold)], plan['anchor']['best_filename']
        return round_record['path']/'runs'/name/f'fold_{fold:02d}', 'best_predictions.npz'
    if name == 'BASE':
        return ORIGINAL/'content'/f'fold_{fold:02d}', 'predictions.npz'
    if number == 1 or fold in plan['reused_folds'].get(name, []):
        return ROUND_PATHS[0]/'runs'/name/f'fold_{fold:02d}', 'best_predictions.npz'
    return round_record['path']/'runs'/name/f'fold_{fold:02d}', 'best_predictions.npz'


def metrics(logits, labels):
    """Confusion metrics and rank-equivalent pairwise probability AUC, no sklearn."""
    z, y = np.asarray(logits), np.asarray(labels)
    require(z.shape == (len(y), 2) and np.isfinite(z).all(), 'Invalid binary logits')
    require(np.array_equal(np.unique(y), [0, 1]), 'Both classes must be present')
    predicted = z.argmax(1)
    matrix = np.bincount(2*y.astype(np.int64)+predicted, minlength=4).reshape(2, 2)
    tp = matrix.diagonal().astype(float)
    recall = tp/matrix.sum(1)
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros(2), where=denominator != 0)
    exponential = np.exp(z-z.max(1, keepdims=True))
    probability = exponential[:, 1]/exponential.sum(1)
    positive, negative = probability[y == 1], probability[y == 0]
    auc = float(((positive[:, None] > negative[None]).sum()
                 + .5*(positive[:, None] == negative[None]).sum())/(len(positive)*len(negative)))
    return dict(acc=float(tp.sum()/len(y)), macro_f1=float(f1.mean()),
                pmci_recall=float(recall[1]), smci_recall=float(recall[0]), pmci_auc=auc,
                balanced_acc=float(recall.mean()), confusion_matrix=matrix.tolist())


def load_fold(round_record, name, fold):
    folder, filename = resolve(round_record, name, fold)
    cache_key = str(folder/filename)
    if cache_key in FOLD_CACHE:
        return FOLD_CACHE[cache_key]
    record = read(folder/'complete.json')
    require(record['fold'] == fold and record['parameters'] == 716396, f'Fold/model mismatch:{folder}')
    pin(folder/'epochs.csv')
    with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
        curve = list(csv.DictReader(stream))
    require([int(row['epoch']) for row in curve] == list(range(1, 1001)), f'Incomplete curve:{folder}')
    curve_acc = np.asarray([float(row['test_acc']) for row in curve])
    epoch = int(curve_acc.argmax())+1
    require(epoch == record['best']['epoch'], f'Checkpoint is not first max test ACC:{folder}')
    pin(folder/filename)
    with np.load(folder/filename) as saved:
        logits, labels, ids = (saved[key].copy() for key in ('logits', 'labels', 'row_index'))
    require(len(ids) == len(np.unique(ids)) == len(labels), f'Repeated/missing fold rows:{folder}')
    measured = metrics(logits, labels)
    same_number(measured['acc'], record['best']['acc'], f'{folder}: recorded ACC')
    same_number(measured['acc'], curve_acc[epoch-1], f'{folder}: curve ACC')
    FOLD_CACHE[cache_key] = dict(folder=str(folder), fold=fold, best_epoch=epoch,
        metrics=measured, logits=logits, labels=labels, row_index=ids,
        last_acc=float(curve_acc[-1]), late100_acc=float(curve_acc[-100:].mean()))
    return FOLD_CACHE[cache_key]


def mean_metrics(rows):
    return {key: float(np.mean([row['metrics'][key] for row in rows])) for key in METRIC_KEYS}


def source_link(path, label):
    return f'[{label}](<{Path(path).as_posix()}>)'


def assemble(rounds):
    unique_configs, fingerprint_index, stages, screening = [], {}, [], []
    round_titles = ('单参数粗筛与局部精调', '四折联合参数搜索', '固定第二轮赢家的单因素损失搜索')
    for rr in rounds:
        number, audit, plan = rr['number'], rr['independent_audit'], rr['PLAN']
        candidate_names = [name for name in rr['all_settings'] if name != 'BASE'] if number == 1 else plan['candidate_names']
        reused_checks = (audit['replayed_baseline_best_checkpoints'] if number == 1 else
                         audit['replayed_reused_checkpoints'] if number == 2 else audit['replayed_anchor_checkpoints'])
        stages.append(dict(round=number, title=round_titles[number-1], path=str(rr['path']),
            candidate_count=len(candidate_names), screen_folds=plan['screen_folds'],
            new_runs=audit['new_runs'], independent_new_checkpoint_replays=audit['replayed_new_checkpoints'],
            independent_reused_checkpoint_replays=reused_checks,
            independent_prediction_files=audit['prediction_files_verified'], retained=rr['analysis']['retained'],
            full_configuration_count=len(rr['analysis']['full']), selected_finalists=rr['selection']['finalists']))
        for name, full in rr['analysis']['full'].items():
            require(full['folds'] == FOLDS, f'Incomplete configuration mislabeled full:{number}/{name}')
            rows = [load_fold(rr, name, fold) for fold in FOLDS]
            calculated = mean_metrics(rows)
            for key, value in calculated.items():
                same_number(value, full['best'][key], f'{number}/{name}: mean {key}')
            same_number(np.mean([row['last_acc'] for row in rows]), full['last']['acc'], f'{number}/{name}: last ACC')
            settings = normalized_settings(rr['all_settings'][name])
            fingerprint = tuple(settings.items())
            alias = dict(round=number, name=name, source=str(rr['path']/'analysis.json'))
            if fingerprint in fingerprint_index:
                previous = unique_configs[fingerprint_index[fingerprint]]
                for key in METRIC_KEYS:
                    same_number(previous['fold_mean'][key], calculated[key], 'Reused complete configuration metrics')
                previous['aliases'].append(alias)
            else:
                fingerprint_index[fingerprint] = len(unique_configs)
                unique_configs.append(dict(name=name, first_completed_round=number, aliases=[alias],
                    settings=settings, fold_mean=calculated, last_epoch_acc=full['last']['acc'],
                    late100_acc=full['late100_acc'], fold_acc_std=full['acc_std'],
                    selection_score=[full['best']['acc'], full['best']['macro_f1'],
                                     full['best']['pmci_recall'], full['late100_acc']]))
        for name in candidate_names:
            folds = list(plan['screen_folds'])
            calculated = mean_metrics([load_fold(rr, name, fold) for fold in folds])
            completed = rr['summary']['candidates'][name]['folds']
            screening.append(dict(round=number, name=name, screen_folds=folds, screen_metrics=calculated,
                completed_folds=completed, completed_tenfold=completed == FOLDS,
                settings=normalized_settings(rr['all_settings'][name])))

    original_rows = [load_fold(rounds[0], 'BASE', fold) for fold in FOLDS]
    final_name = rounds[-1]['analysis']['retained']
    final_rows = [load_fold(rounds[-1], final_name, fold) for fold in FOLDS]
    def oof(rows):
        ids = np.concatenate([row['row_index'] for row in rows])
        labels = np.concatenate([row['labels'] for row in rows])
        logits = np.concatenate([row['logits'] for row in rows])
        require(len(ids) == len(np.unique(ids)) == 535, 'OOF must contain each source row exactly once')
        order = np.argsort(ids, kind='stable')
        return ids[order], labels[order], logits[order]
    base_ids, base_labels, base_logits = oof(original_rows)
    final_ids, final_labels, final_logits = oof(final_rows)
    np.testing.assert_array_equal(base_ids, final_ids)
    np.testing.assert_array_equal(base_labels, final_labels)
    np.testing.assert_array_equal(np.bincount(base_labels, minlength=2), [490, 45])
    for rr in rounds:
        pin(rr['path']/'data_splits_graphs.npz')
        with np.load(rr['path']/'data_splits_graphs.npz') as archive:
            order = np.argsort(archive['source_row'], kind='stable')
            np.testing.assert_array_equal(base_ids, archive['source_row'][order])
            np.testing.assert_array_equal(base_labels, archive['labels'][order])
    pin(rounds[-1]['path']/'retained_oof_predictions.npz')
    with np.load(rounds[-1]['path']/'retained_oof_predictions.npz') as exported:
        order = np.argsort(exported['row_index'], kind='stable')
        np.testing.assert_array_equal(final_ids, exported['row_index'][order])
        np.testing.assert_array_equal(final_labels, exported['labels'][order])
        np.testing.assert_array_equal(final_logits, exported['logits'][order])
    base_prediction, final_prediction = base_logits.argmax(1), final_logits.argmax(1)
    base_correct, final_correct = base_prediction == base_labels, final_prediction == final_labels
    corrected, harmed = final_correct & ~base_correct, base_correct & ~final_correct
    base_mean, final_mean = mean_metrics(original_rows), mean_metrics(final_rows)
    per_class = {}
    for code, label in enumerate(('SMCI', 'PMCI')):
        mask = base_labels == code
        per_class[label] = dict(samples=int(mask.sum()), corrected=int((corrected & mask).sum()),
            harmed=int((harmed & mask).sum()), original_errors=int((~base_correct & mask).sum()),
            final_errors=int((~final_correct & mask).sum()))
    changes = [dict(row_index=int(base_ids[index]), label=int(base_labels[index]),
                    original_prediction=int(base_prediction[index]), final_prediction=int(final_prediction[index]),
                    change='corrected' if corrected[index] else 'harmed')
               for index in np.flatnonzero(corrected | harmed)]
    fold_comparison = []
    for old, new in zip(original_rows, final_rows):
        np.testing.assert_array_equal(old['row_index'], new['row_index'])
        fold_comparison.append(dict(fold=old['fold'], samples=len(old['labels']),
            original_best_epoch=old['best_epoch'], final_best_epoch=new['best_epoch'],
            original_acc=old['metrics']['acc'], final_acc=new['metrics']['acc'],
            difference_pp=100*(new['metrics']['acc']-old['metrics']['acc'])))
    original_settings = normalized_settings(rounds[0]['all_settings']['BASE'])
    final_settings = normalized_settings(rounds[-1]['all_settings'][final_name])
    final_fingerprint = tuple(final_settings.items())
    final_display_name = unique_configs[fingerprint_index[final_fingerprint]]['name']
    best_complete_score = max(tuple(row['selection_score']) for row in unique_configs)
    final_full = rounds[-1]['analysis']['full'][final_name]
    final_score = (final_full['best']['acc'], final_full['best']['macro_f1'],
                   final_full['best']['pmci_recall'], final_full['late100_acc'])
    require(final_score == best_complete_score, 'Final selection is not best among all complete configurations')
    comparison = dict(original_name='BASE', final_name=final_name, final_display_name=final_display_name,
        final_anchor_source=rounds[-1]['PLAN']['anchor']['name'],
        original_fold_mean=base_mean, final_fold_mean=final_mean,
        difference_pp={key: 100*(final_mean[key]-base_mean[key]) for key in METRIC_KEYS},
        original_pooled=metrics(base_logits, base_labels), final_pooled=metrics(final_logits, final_labels),
        corrected=int(corrected.sum()), harmed=int(harmed.sum()), net_corrected=int(corrected.sum()-harmed.sum()),
        changed_rows=changes, per_class=per_class, fold_comparison=fold_comparison,
        original_settings=original_settings, final_settings=final_settings)
    export = rounds[-1]['best_configuration']
    require(export['reproduction_entrypoint'] == 'tad_binary_loss_tune.py'
            and export['config_alone_reproduces_loss'] is False, 'Loss reproduction limitation missing')
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), summary_script_sha256=digest(__file__),
        task='TADPOLE SMCI/PMCI', samples=535, class_counts={'SMCI': 490, 'PMCI': 45},
        model='C: RelationSheafModel(mode=content)', parameters=716396, seed=0, folds=10, epochs_per_fold=1000,
        rounds=stages, total_new_training_runs=sum(row['new_runs'] for row in stages),
        total_independent_new_checkpoint_replays=sum(row['independent_new_checkpoint_replays'] for row in stages),
        current_followup_new_training_runs=sum(row['new_runs'] for row in stages[1:]),
        current_followup_new_checkpoint_replays=sum(row['independent_new_checkpoint_replays'] for row in stages[1:]),
        complete_configurations=unique_configs, screening_candidates=screening, comparison=comparison,
        best_complete_selection_score=list(best_complete_score),
        tied_best_complete_configurations=[row['name'] for row in unique_configs
                                          if tuple(row['selection_score']) == best_complete_score],
        reproduction={key: export[key] for key in ('config', 'overrides', 'reproduction_entrypoint',
                      'config_alone_reproduces_loss', 'reproduction_note')},
        limitations=['All folds use the first highest TEST-ACC epoch among 1000 epochs, not the final epoch.',
            'Searches adaptively reuse test folds; the independent audit verifies artifacts and execution, not independent generalization.',
            'Only seed0 was used; this is not multi-seed confirmation or a global optimum.',
            'Full tenfold means are unweighted fold means; pooled metrics use all535 row-aligned predictions and need not be identical.',
            'Source-row uniqueness does not establish unique independent clinical patient IDs.',
            'Checkpoint replay counts include repeated reuse across rounds; they are not additional independent training runs.'],
        input_sha256=dict(INPUT_HASHES))


def markdown(report):
    c = report['comparison']
    percent = lambda value: f'{100*value:.4f}%'
    lines = ['# TAD 二分类 C 框架调参汇总', '',
        f"最终保留 **{c['final_display_name']}**（第三轮导出名称：`{c['final_name']}`）。原配置十折最佳ACC为 **{percent(c['original_fold_mean']['acc'])}**，最终为 **{percent(c['final_fold_mean']['acc'])}**，变化 **{c['difference_pp']['acc']:+.4f}个百分点**。",
        f"535行逐一对齐后，纠正原错误 **{c['corrected']}例**，新增错误 **{c['harmed']}例**，净纠正 **{c['net_corrected']}例**。", '',
        'TADPOLE SMCI/PMCI，535行（SMCI 490、PMCI 45），C网络716396个参数；固定seed0、原十折和每折1000轮。所有“最佳”结果均按每折首次最高测试ACC选轮，不是第1000轮成绩。', '',
        '## 三轮完成情况', '',
        '| 轮次 | 搜索范围 | 探索候选数 | 新训练次数 | 新检查点独立重放 | 复用检查点重放 | 核查预测文件 | 本轮保留 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in report['rounds']:
        lines.append(f"| {row['round']} | {row['title']} | {row['candidate_count']} | {row['new_runs']} | {row['independent_new_checkpoint_replays']} | {row['independent_reused_checkpoint_replays']} | {row['independent_prediction_files']} | {row['retained']} |")
    lines += ['', f"三轮合计新增 **{report['total_new_training_runs']}次完整训练**，对应 **{report['total_independent_new_checkpoint_replays']}个新增最佳／末轮检查点**完成独立重放。原BASE十折是历史引用，不计入新增训练数。复用重放次数可能跨轮重复，不能相加当成独立实验。原BASE没有末轮权重，仅复核其末轮预测文件。", '',
        f"本次继续搜索（第二、三轮）新增 **{report['current_followup_new_training_runs']}次训练**，对应 **{report['current_followup_new_checkpoint_replays']}个新增检查点**独立重放；第一轮40次训练属于此前已完成工作。", '',
        '## 全部完整十折配置', '',
        '相同参数配置按实际七项训练参数去重；保留来源别名。两折／四折筛选分数另列。', '',
        '| 配置 | 首次补齐轮次 | 最佳ACC | 宏F1 | PMCI召回 | SMCI召回 | PMCI概率AUC | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['complete_configurations']:
        m = row['fold_mean']
        lines.append(f"| {row['name']} | {row['first_completed_round']} | {percent(m['acc'])} | {percent(m['macro_f1'])} | {percent(m['pmci_recall'])} | {percent(m['smci_recall'])} | {percent(m['pmci_auc'])} | {percent(row['last_epoch_acc'])} |")
    lines += ['', '复用别名：', '']
    for row in report['complete_configurations']:
        labels = '、'.join(f"第{alias['round']}轮 `{alias['name']}`" for alias in row['aliases'])
        lines.append(f"- `{row['name']}`：{labels}。")
    lines += ['', '## 原配置与最终配置', '',
        '| 十折均值指标 | 原BASE | 最终配置 | 变化（百分点） |',
        '|---|---:|---:|---:|']
    metric_names = dict(acc='ACC', macro_f1='宏F1', pmci_recall='PMCI召回', smci_recall='SMCI召回', pmci_auc='PMCI概率AUC')
    for key, label in metric_names.items():
        lines.append(f"| {label} | {percent(c['original_fold_mean'][key])} | {percent(c['final_fold_mean'][key])} | {c['difference_pp'][key]:+.4f} |")
    lines += ['', '下表混淆矩阵直接来自535行对齐后的折外预测；行是真实标签，列是预测标签。十折均值与合并535行指标的分母不同，不能混用。', '',
        '| 配置 | SMCI→SMCI | SMCI→PMCI | PMCI→SMCI | PMCI→PMCI | 合并ACC |',
        '|---|---:|---:|---:|---:|---:|']
    for label, value in (('原BASE', c['original_pooled']), ('最终配置', c['final_pooled'])):
        cm = value['confusion_matrix']
        lines.append(f"| {label} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} | {percent(value['acc'])} |")
    lines += ['', '| 真实类别 | 样本数 | 原错误数 | 最终错误数 | 纠正 | 新增错误 |', '|---|---:|---:|---:|---:|---:|']
    for label, value in c['per_class'].items():
        lines.append(f"| {label} | {value['samples']} | {value['original_errors']} | {value['final_errors']} | {value['corrected']} | {value['harmed']} |")
    lines += ['', '## 十折最佳轮次及ACC', '',
        '| 折 | 样本数 | 原最佳轮次 | 最终最佳轮次 | 原ACC | 最终ACC | 变化（百分点） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in c['fold_comparison']:
        lines.append(f"| {row['fold']} | {row['samples']} | {row['original_best_epoch']} | {row['final_best_epoch']} | {percent(row['original_acc'])} | {percent(row['final_acc'])} | {row['difference_pp']:+.4f} |")
    lines += ['', '## 参数与复现', '', '| 参数 | 原BASE | 最终配置 |', '|---|---:|---:|']
    for key in PARAMETER_KEYS:
        lines.append(f"| `{key}` | {c['original_settings'][key]:g} | {c['final_settings'][key]:g} |")
    lines += ['', '数据、C结构、候选图、类别权重、Adam优化器、图参数零weight decay、梯度裁剪1、前20轮保持学习率及后续余弦调度均沿用既定设置。', '',
        '复现本轮标签平滑0或0.02的候选，须使用 `tad_binary_loss_tune.py` 的显式损失工厂，或向 `MSCQGNLoss` 显式传入对应 `label_smoothing`，同时作用于主分类和两个辅助分类CE。旧 `Config_` 和旧训练入口不读取该新增字段。最终保留配置的平滑系数仍是旧默认值0.05，因此该字段对最终配置没有引入新的损失行为；完整复现仍需同时保留本文记录的图参数零weight decay等训练设置。', '',
        source_link(report['reproduction']['config'], '最终配置INI')+'；'+source_link(ROUND_PATHS[2]/'best_configuration.json', '完整参数与复现说明')+'。', '',
        '## 筛选候选：不能当成完整十折结果', '',
        '下面每个分数只对应指定筛选折。已补齐十折的配置也保留其筛选值以披露选择过程；未补齐候选不能据此认定无效。', '',
        '| 轮次 | 候选 | 筛选折 | 筛选ACC | 筛选宏F1 | 本轮完成折数 | 本轮是否完整十折 |',
        '|---|---|---|---:|---:|---:|---|']
    for row in report['screening_candidates']:
        folds = '、'.join(str(fold) for fold in row['screen_folds'])
        m = row['screen_metrics']
        lines.append(f"| {row['round']} | {row['name']} | {folds} | {percent(m['acc'])} | {percent(m['macro_f1'])} | {len(row['completed_folds'])} | {'是' if row['completed_tenfold'] else '否'} |")
    lines += ['', '## 结论边界与记录', '',
        '最终配置是在这三轮已测试并完成十折的配置范围内按既定排序保留的最好结果，不代表全局最优或性能上限。当前只有seed0，未进行多种子确认。', '',
        '保持了原先按测试ACC选轮和自适应测试折调参的规则。独立审计确认的是代码执行、检查点、指标与数据行的一致性，不是独立泛化验证。唯一数据行也不能代替临床患者ID独立性确认。', '']
    for stage in report['rounds']:
        path = Path(stage['path'])
        lines.append(f"- 第{stage['round']}轮："+source_link(path/'analysis.json', '完整结果')+'；'+source_link(path/'independent_audit.json', '独立审计')+'。')
    lines.append('- '+source_link(OUTPUT_STEM.with_suffix('.json'), '机器可读汇总、逐行纠错与输入哈希')+'。')
    return '\n'.join(lines)+'\n'


def main():
    rounds = load_rounds()
    report = assemble(rounds)
    # Capture any input edits during assembly before emitting a report.
    for relative, expected in INPUT_HASHES.items():
        require(digest(ROOT/relative) == expected, f'Input changed while assembling:{relative}')
    text = markdown(report)
    for suffix, value in (('.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)), ('.md', text)):
        target = OUTPUT_STEM.with_suffix(suffix)
        temporary = target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(value, encoding='utf-8')
        temporary.replace(target)
    c = report['comparison']
    print(json.dumps(dict(passed=True, final_configuration=c['final_display_name'],
        original_acc=c['original_fold_mean']['acc'], final_acc=c['final_fold_mean']['acc'],
        corrected=c['corrected'], harmed=c['harmed'], new_training_runs=report['total_new_training_runs'],
        output_markdown=str(OUTPUT_STEM.with_suffix('.md')), output_json=str(OUTPUT_STEM.with_suffix('.json'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
