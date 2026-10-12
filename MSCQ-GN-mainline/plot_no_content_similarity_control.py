"""Chinese PNG figures for the completed no-current-content-similarity control.

Only saved summary/CSV files are plotted. Final generation requires both ten-
fold arms, the COMPLETE marker, matching records, and all source hashes.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'Result/no_content_similarity_seed0_20261011'
DEFAULT_OUTPUT = ROOT / 'Visualization/no_content_similarity_seed0_20261011'
CONTROL = 'NO_CONTENT_SIMILARITY'
MODES = ('C', CONTROL)
PARAMETERS = {'C': 895096, CONTROL: 893560}
NAMES = {'C': '原 C', CONTROL: '去除内容相似度项'}
COLORS = {'C': '#D55E00', CONTROL: '#0072B2'}
METRICS = ('acc', 'macro_f1', 'auc')
METRIC_NAMES = {'acc': 'ACC', 'macro_f1': '宏 F1', 'auc': '宏平均 OVR AUC'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _font():
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if Path(filename).is_file():
            font_manager.fontManager.addfont(filename)
            name = font_manager.FontProperties(fname=filename).get_name()
            plt.rcParams['font.family'] = name
            return {'name': name, 'file': filename, 'sha256': _sha(filename)}
    for name in ('Microsoft YaHei', 'SimHei'):
        try:
            filename = font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams['font.family'] = name
        return {'name': name, 'file': filename, 'sha256': _sha(filename)}
    raise RuntimeError('Chinese output requires an actual Microsoft YaHei or SimHei font')


def _load(folder):
    complete_path = folder / 'COMPLETE.json'
    complete = _read(complete_path)
    if complete.get('audit', {}).get('passed') is not True:
        raise ValueError('Refusing final figures: completed successful experiment audit is required')
    summary_path = folder / 'summary.json'
    summary = _read(summary_path)
    if (summary.get('all_complete') is not True or summary.get('all_new_complete') is not True
            or summary.get('all_reference_complete') is not True
            or summary.get('completed_new_runs') != 10
            or summary.get('training_seed') != 0
            or summary.get('planned_folds') != list(range(1, 11))):
        raise ValueError('Refusing final figures: two completed seed0 ten-fold arms are required')
    data, fold_sizes = {}, {}
    source_files = {str(summary_path.resolve()): _sha(summary_path),
                    str(complete_path.resolve()): _sha(complete_path)}
    for mode in MODES:
        path = folder / f'metrics_{mode}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        variant = summary['variants'][mode]
        if (len(rows) != 10 or [int(row['fold']) for row in rows] != list(range(1, 11))
                or any(row['mode'] != mode or int(row['training_seed']) != 0
                       or int(row['parameters']) != PARAMETERS[mode] for row in rows)
                or variant['completed_folds'] != 10
                or variant['completed_fold_ids'] != list(range(1, 11))
                or variant['parameters'] != PARAMETERS[mode]):
            raise ValueError(f'{mode}: incorrect CSV or summary completion/mode/seed/parameters')
        fold_sizes[mode] = [int(row['heldout_patients']) for row in rows]
        if sum(fold_sizes[mode]) != 598:
            raise ValueError(f'{mode}: expected 598 unique heldout patients')
        data[mode] = {'best': {}, 'last': {}, 'last100': {}}
        for stage in ('best', 'last'):
            for metric in METRICS:
                values = np.asarray([float(row[f'{stage}_{metric}']) for row in rows])
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError(f'{mode}/{stage}/{metric}: invalid score')
                np.testing.assert_allclose(values.mean(), variant[stage][metric], atol=1e-12, rtol=0)
                np.testing.assert_allclose(
                    values, [fold[stage][metric] for fold in variant['folds']], atol=1e-12, rtol=0)
                data[mode][stage][metric] = values
            for index, row in enumerate(rows):
                for field in ('per_class_recall', 'confusion_matrix'):
                    np.testing.assert_allclose(json.loads(row[f'{stage}_{field}']),
                                               variant['folds'][index][stage][field], atol=1e-12, rtol=0)
        values = np.asarray([float(row['last100_acc']) for row in rows])
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f'{mode}: invalid last100 ACC')
        np.testing.assert_allclose(values.mean(), variant['mean_last100_acc'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(values, [fold['mean_last100_acc'] for fold in variant['folds']],
                                   atol=1e-12, rtol=0)
        data[mode]['last100']['acc'] = values
        if any(int(row['last_epoch']) != 1000 or not 1 <= int(row['best_epoch']) <= 1000
               or int(row['last100_first_epoch']) != 901 or int(row['last100_final_epoch']) != 1000
               for row in rows):
            raise ValueError(f'{mode}: incorrect epoch selection')
        source_files[str(path.resolve())] = _sha(path)
    if fold_sizes['C'] != fold_sizes[CONTROL]:
        raise ValueError('Arms have inconsistent heldout fold sizes')
    if len(summary['matched_comparisons']) != 1:
        raise ValueError('Expected one prespecified paired comparison')
    pair = summary['matched_comparisons'][0]
    if (pair['reference'] != CONTROL or pair['candidate'] != 'C'
            or pair['id'] != 'retain_current_content_similarity'
            or pair['matched_folds'] != list(range(1, 11))):
        raise ValueError('Comparison must be complete C minus NO_CONTENT_SIMILARITY')
    for stage in ('best', 'last'):
        for metric in METRICS:
            delta = data['C'][stage][metric] - data[CONTROL][stage][metric]
            saved = pair['stages'][stage]['metric_deltas'][metric]
            np.testing.assert_allclose(delta.mean(), saved['mean'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(delta, [row['delta'] for row in saved['folds']], atol=1e-12, rtol=0)
    delta = data['C']['last100']['acc'] - data[CONTROL]['last100']['acc']
    np.testing.assert_allclose(delta.mean(), pair['stages']['last100']['mean_acc_delta'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(delta, [row['acc_delta'] for row in pair['stages']['last100']['folds']],
                               atol=1e-12, rtol=0)
    upstream = summary.get('source_sha256', {})
    if len(upstream) != 80:
        raise ValueError('Expected 80 complete/curve/best-prediction/last-prediction source hashes')
    for path, digest in upstream.items():
        if _sha(path) != digest:
            raise ValueError(f'Changed input after analysis: {path}')
    for path, digest in summary.get('analysis_source_sha256', {}).items():
        if _sha(path) != digest:
            raise ValueError(f'Changed frozen analysis source: {path}')
    if (folder / 'PLAN.json').exists():
        source_files[str((folder / 'PLAN.json').resolve())] = _sha(folder / 'PLAN.json')
    return summary, data, source_files


def _range(values):
    values = np.asarray(values, dtype=float) * 100
    margin = max(1., .10 * float(values.max() - values.min()))
    lower = max(0., 5 * np.floor((values.min() - margin) / 5))
    upper = min(100., 5 * np.ceil((values.max() + margin) / 5))
    if upper <= lower:
        lower, upper = max(0., lower - 5), min(100., upper + 5)
    return float(lower), float(upper)


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e5e9', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10.5)


def _points(ax, data, stage, metric, bounds):
    # Leave a gap at each group centre so the mean cannot obscure a fold point.
    jitter = np.r_[np.linspace(-.24, -.08, 5), np.linspace(.08, .24, 5)]
    labels = []
    for x, mode in enumerate(MODES):
        values = 100 * data[mode][stage][metric]
        ax.scatter(x + jitter, values, color=COLORS[mode], s=36, alpha=.86,
                   edgecolors='white', linewidth=.45, zorder=3, clip_on=False)
        mean = float(values.mean())
        ax.scatter(x, mean, marker='D', color='#20252b', s=80, edgecolors='white',
                   linewidth=.7, zorder=4, clip_on=False)
        labels.append(f'{NAMES[mode]}\n均值 {mean:.3f}%')
    ax.set_xticks(range(2), labels, fontsize=11.5)
    ax.set_xlim(-.48, 1.48)
    ax.set_ylim(bounds)
    ax.set_yticks(np.arange(bounds[0], bounds[1] + .1, 5))
    _style(ax)


def _legend():
    return [Line2D([], [], marker='o', linestyle='none', color='#687583', markersize=7,
                   label='小点：全部 10 折'),
            Line2D([], [], marker='D', linestyle='none', color='#20252b', markersize=8,
                   label='黑色菱形：10 折等权均值')]


def _save(fig, output, name):
    path = output / name
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    return path


def _primary(data, output):
    bounds = _range([value for mode in MODES for metric in METRICS for value in data[mode]['best'][metric]])
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.8), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.265, top=.71, wspace=.25)
    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        _points(ax, data, 'best', metric, bounds)
        ax.set_title(f'({chr(97+index)}) {METRIC_NAMES[metric]}', fontsize=15, loc='left', pad=14)
    axes[0].set_ylabel('指标数值（%）', fontsize=13)
    fig.suptitle('当前内容相似度项：原 C 与去除项对照', fontsize=22, fontweight='bold', y=.976)
    fig.text(.5, .880, f'主选点：1000 轮中首次最高 TEST ACC　｜　三面板共用纵轴：{bounds[0]:g}–{bounds[1]:g}%',
             ha='center', fontsize=12)
    fig.legend(handles=_legend(), loc='upper center', bbox_to_anchor=(.5, .846),
               ncol=2, frameon=False, fontsize=11.5)
    fig.text(.065, .127, '原 C 保留当前内容相似度项；对照仅去除该项，其他内容依赖仍保留。\n'
             '三项指标取同一最佳 ACC 检查点；AUC 沿用原 logits 的宏平均 OVR 口径。',
             fontsize=10.7, color='#49515a', linespacing=1.5)
    fig.text(.065, .034, 'seed0，598 名患者，原十折；折点用于描述，不视为独立样本。原 C 895096 参数，对照 893560 参数。',
             fontsize=10, color='#606971')
    path = _save(fig, output, 'fig01_best_metrics_two_groups_zh.png')
    return path, {'stage': 'best', 'axis_range_percent': bounds,
                  'panels': {metric: {'fold_points_per_arm': 10, 'arms': 2, 'fold_points_total': 20,
                             'equal_fold_means_percent': {mode: float(data[mode]['best'][metric].mean() * 100)
                                                         for mode in MODES}}
                             for metric in METRICS}}


def _paired(data, output):
    deltas = {metric: 100 * (data['C']['best'][metric] - data[CONTROL]['best'][metric]) for metric in METRICS}
    maximum = max(float(np.abs(values).max()) for values in deltas.values())
    limit = max(1., .5 * np.ceil((maximum + .30) / .5))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.285, top=.71, wspace=.25)
    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        values = deltas[metric]
        mean = float(values.mean())
        if abs(mean) < 1e-12:
            mean = 0.
        ax.axhline(0, color='#75808b', linestyle='--', linewidth=1.2)
        ax.plot(range(1, 11), values, linestyle='none', marker='o', markersize=6.5,
                color='#0072B2', zorder=3)
        ax.axhline(values.mean(), color='#D55E00', linestyle=':', linewidth=1.8)
        ax.set_xlim(.55, 10.45)
        ax.set_ylim(-limit, limit)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel(f'原十折编号\n平均差 {mean:+.4f} 个百分点', fontsize=11.5)
        ax.set_title(f'({chr(97+index)}) {METRIC_NAMES[metric]}', fontsize=15, loc='left', pad=14)
        _style(ax)
    axes[0].set_ylabel('完整 C − 对照（百分点）', fontsize=13)
    fig.suptitle('完整 C 减对照：逐折配对指标变化', fontsize=22, fontweight='bold', y=.976)
    fig.text(.5, .880, f'最佳 ACC 选点下的三项指标　｜　共用纵轴：−{limit:g} 至 +{limit:g} 个百分点',
             ha='center', fontsize=12)
    handles = [Line2D([], [], marker='o', linestyle='none', color='#0072B2', label='同一折的差值'),
               Line2D([], [], linestyle=':', color='#D55E00', label='10 折平均差'),
               Line2D([], [], linestyle='--', color='#75808b', label='无变化')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .846),
               ncol=3, frameon=False, fontsize=11.5)
    fig.text(.065, .127, '差值始终为“完整 C − 去除当前内容相似度项的对照”；正值表示完整 C 较高，负值表示对照较高。\n'
             '每点匹配同一折和同一批留出患者；三个面板反复涉及这 598 名患者。',
             fontsize=10.7, color='#49515a', linespacing=1.5)
    fig.text(.065, .034, '单种子、原 TEST 选点下的描述比较；不计算置信区间、显著性或独立样本推断。AUC 保留原 logits 口径。',
             fontsize=10, color='#606971')
    path = _save(fig, output, 'fig02_best_paired_metric_differences_zh.png')
    return path, {'stage': 'best', 'difference_definition': 'C minus NO_CONTENT_SIMILARITY',
                  'axis_range_percentage_points': [-limit, limit],
                  'panels': {metric: {'matched_fold_points': 10, 'delta_percentage_points': values.tolist(),
                             'equal_fold_mean_percentage_points': float(values.mean())}
                             for metric, values in deltas.items()}}


def _auxiliary(data, output):
    bounds = _range([value for mode in MODES for stage in ('last', 'last100')
                     for value in data[mode][stage]['acc']])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 7.8), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, bottom=.265, top=.70, wspace=.22)
    for index, (ax, stage, title) in enumerate(zip(axes, ('last', 'last100'), ('第 1000 轮', '第 901–1000 轮平均'))):
        _points(ax, data, stage, 'acc', bounds)
        ax.set_title(f'({chr(97+index)}) {title}', fontsize=15, loc='left', pad=14)
    axes[0].set_ylabel('ACC（%）', fontsize=13)
    fig.suptitle('辅助观察：训练末期的 ACC', fontsize=22, fontweight='bold', y=.976)
    fig.text(.5, .880, f'辅助结果不替换主指标　｜　两个面板共用纵轴：{bounds[0]:g}–{bounds[1]:g}%',
             ha='center', fontsize=12)
    fig.legend(handles=_legend(), loc='upper center', bbox_to_anchor=(.5, .846),
               ncol=2, frameon=False, fontsize=11.5)
    fig.text(.09, .127, '末 100 轮先在折内平均，再将十折等权平均；每组全部 10 折均显示。\n'
             '组间配对方向仍为完整 C 减对照；此处显示各组分数，不替换最佳 ACC 主指标。',
             fontsize=10.7, color='#49515a', linespacing=1.5)
    fig.text(.09, .034, 'seed0，598 名患者；单种子描述比较，无置信区间或显著性标记。去除该项不等于去除所有内容依赖。',
             fontsize=9.8, color='#606971')
    path = _save(fig, output, 'fig03_auxiliary_late_acc_zh.png')
    return path, {'auxiliary_only': True, 'axis_range_percent': bounds,
                  'panels': {stage: {'fold_points_per_arm': 10, 'arms': 2, 'fold_points_total': 20,
                             'equal_fold_means_percent': {mode: float(data[mode][stage]['acc'].mean() * 100)
                                                         for mode in MODES},
                             'mean_C_minus_control_percentage_points':
                             float((data['C'][stage]['acc'] - data[CONTROL][stage]['acc']).mean() * 100)}
                             for stage in ('last', 'last100')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, data, sources = _load(args.input)
    font = _font()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'axes.labelpad': 10, 'savefig.dpi': 300})
    args.output.mkdir(parents=True, exist_ok=True)
    products = []
    for operation in (_primary, _paired, _auxiliary):
        path, details = operation(data, args.output)
        products.append({'file': str(path.resolve()), 'sha256': _sha(path),
                         'bytes': path.stat().st_size, 'details': details})
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training_seed': 0, 'folds': list(range(1, 11)), 'heldout_unique_patients_per_arm': 598,
        'patient_sets_recur_across_arms_metrics_and_panels': True,
        'difference_definition': 'C minus NO_CONTENT_SIMILARITY',
        'positive_difference': 'The complete C score is higher; descriptive only.',
        'source_files_sha256': sources,
        'validated_upstream_source_files_sha256': summary['source_sha256'],
        'validated_analysis_source_sha256': summary['analysis_source_sha256'],
        'plot_source': {'file': str(Path(__file__).resolve()), 'sha256': _sha(__file__)},
        'font': font, 'dpi': 300, 'format': 'Ordinary Chinese PNG',
        'group_order': list(MODES), 'parameter_counts': PARAMETERS,
        'primary_endpoint': summary['selection'], 'aggregation': summary['aggregation'],
        'auc_definition': summary['auc_definition'],
        'point_offsets': 'Deterministic horizontal separation for all ten folds; not uncertainty.',
        'confidence_intervals_or_significance_tests': False,
        'training_or_checkpoint_inference_performed': False,
        'products': products,
    }
    target = args.output / 'figures_metadata.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'figures': [item['file'] for item in products],
                      'metadata': str(target.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
