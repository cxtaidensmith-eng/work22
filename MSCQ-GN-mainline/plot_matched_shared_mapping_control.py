"""Plot the completed matched shared-mapping control using saved metrics only.

No training, inference, checkpoint replay, or rewriting of analysis outputs.
All four arms must have ten folds before any final PNG is generated.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'Result/matched_shared_mapping_seed0_20261011'
DEFAULT_OUTPUT = ROOT / 'Visualization/matched_shared_mapping_seed0_20261011'
MODES = ('C', 'D1', 'SHARED_MAP', 'D1_SHARED_MAP')
ORDER = ('D1_SHARED_MAP', 'D1', 'SHARED_MAP', 'C')
PARAMETERS = {'C': 895096, 'D1': 895036, 'SHARED_MAP': 895016, 'D1_SHARED_MAP': 894956}
NAMES = {
    'D1_SHARED_MAP': '自身模态关系\n共享变换',
    'D1': '自身模态关系\n专属变换',
    'SHARED_MAP': '全部模态关系\n共享变换',
    'C': '全部模态关系\n专属变换',
}
COLORS = {'D1_SHARED_MAP': '#56B4E9', 'D1': '#E69F00',
          'SHARED_MAP': '#0072B2', 'C': '#D55E00'}
CONTRASTS = (
    ('mapping_with_crossed_paths', 'SHARED_MAP', 'C', '专属变换 − 共享变换\n全部模态关系固定'),
    ('mapping_with_diagonal_paths', 'D1_SHARED_MAP', 'D1', '专属变换 − 共享变换\n自身模态关系固定'),
    ('crossed_paths_with_conditioned_maps', 'D1', 'C', '全部关系 − 自身模态关系\n专属变换固定'),
    ('crossed_paths_with_shared_maps', 'D1_SHARED_MAP', 'SHARED_MAP', '全部关系 − 自身模态关系\n共享变换固定'),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _font():
    candidates = ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf')
    for candidate in candidates:
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            properties = font_manager.FontProperties(fname=candidate)
            plt.rcParams['font.family'] = properties.get_name()
            return {'name': properties.get_name(), 'path': candidate, 'sha256': _sha(candidate)}
    for name in ('Microsoft YaHei', 'SimHei'):
        try:
            filename = font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams['font.family'] = name
        return {'name': name, 'path': filename, 'sha256': _sha(filename)}
    raise RuntimeError('A real Microsoft YaHei or SimHei Chinese font is required')


def _load(folder):
    complete_path = folder / 'COMPLETE.json'
    complete = _read(complete_path)
    audit = complete.get('audit', {})
    if (audit.get('passed') is not True or audit.get('verified_runs') != 10
            or audit.get('verified_checkpoints') != 20):
        raise ValueError('Refusing final figures: completed ten-run/twenty-checkpoint audit is required')
    summary_path = folder / 'summary.json'
    summary = _read(summary_path)
    if (summary.get('all_four_complete') is not True
            or summary.get('all_new_complete') is not True
            or summary.get('all_reference_complete') is not True
            or summary.get('training_seed') != 0
            or summary.get('planned_folds') != list(range(1, 11))):
        raise ValueError('Refusing final figures: four complete seed0 ten-fold arms are required')
    rows, data = {}, {}
    source_files = {str(summary_path.resolve()): _sha(summary_path),
                    str(complete_path.resolve()): _sha(complete_path)}
    for mode in MODES:
        csv_path = folder / f'metrics_{mode}.csv'
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            items = list(csv.DictReader(stream))
        if (len(items) != 10 or [int(row['fold']) for row in items] != list(range(1, 11))
                or any(row['mode'] != mode or int(row['training_seed']) != 0
                       or int(row['parameters']) != PARAMETERS[mode] for row in items)):
            raise ValueError(f'{mode}: incomplete or inconsistent CSV fold/mode/seed/parameters')
        variant = summary['variants'][mode]
        if (variant['completed_folds'] != 10
                or variant['completed_fold_ids'] != list(range(1, 11))
                or variant['parameters'] != PARAMETERS[mode]):
            raise ValueError(f'{mode}: inconsistent summary progress or parameter count')
        data[mode] = {}
        for stage, column in (('best', 'best_acc'), ('last', 'last_acc'), ('last100', 'last100_acc')):
            values = np.asarray([float(row[column]) for row in items])
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError(f'{mode}/{stage}: invalid accuracy')
            data[mode][stage] = values
            expected = (variant['mean_last100_acc'] if stage == 'last100' else variant[stage]['acc'])
            np.testing.assert_allclose(values.mean(), expected, atol=1e-12, rtol=0)
            for index, fold in enumerate(variant['folds']):
                expected_fold = (fold['mean_last100_acc'] if stage == 'last100' else fold[stage]['acc'])
                np.testing.assert_allclose(values[index], expected_fold, atol=1e-12, rtol=0)
        if any(int(row['last_epoch']) != 1000 or not 1 <= int(row['best_epoch']) <= 1000
               or int(row['last100_first_epoch']) != 901 or int(row['last100_final_epoch']) != 1000
               for row in items):
            raise ValueError(f'{mode}: inconsistent epoch selection')
        if sum(int(row['heldout_patients']) for row in items) != 598:
            raise ValueError(f'{mode}: expected 598 unique heldout patients over ten folds')
        rows[mode] = items
        source_files[str(csv_path.resolve())] = _sha(csv_path)
    if any([row['heldout_patients'] for row in rows[mode]]
           != [row['heldout_patients'] for row in rows['C']] for mode in MODES):
        raise ValueError('Arms have inconsistent heldout fold sizes')
    comparisons = {item['id']: item for item in summary['matched_comparisons']}
    if set(comparisons) != {item[0] for item in CONTRASTS}:
        raise ValueError('Expected exactly the four planned single-factor contrasts')
    for identifier, reference, candidate, _ in CONTRASTS:
        pair = comparisons[identifier]
        if (pair['reference'] != reference or pair['candidate'] != candidate
                or pair['matched_folds'] != list(range(1, 11))):
            raise ValueError(f'{identifier}: incorrect contrast definition or matched folds')
        for stage in ('best', 'last', 'last100'):
            deltas = data[candidate][stage] - data[reference][stage]
            saved = pair['stages'][stage]
            np.testing.assert_allclose(deltas.mean(), saved['mean_acc_delta'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(deltas, [row['acc_delta'] for row in saved['folds']],
                                       atol=1e-12, rtol=0)
    interaction = summary['descriptive_interaction']
    if (interaction['matched_folds'] != list(range(1, 11))
            or interaction['formula'] != 'C - SHARED_MAP - D1 + D1_SHARED_MAP'):
        raise ValueError('Incorrect descriptive interaction definition')
    for stage in ('best', 'last', 'last100'):
        values = data['C'][stage] - data['SHARED_MAP'][stage] - data['D1'][stage] + data['D1_SHARED_MAP'][stage]
        np.testing.assert_allclose(values, [row['interaction'] for row in interaction['stages'][stage]['folds']],
                                   atol=1e-12, rtol=0)
    upstream = summary.get('source_sha256', {})
    if len(upstream) != 160:
        raise ValueError('Expected all 160 historical/new record, curve and prediction hashes')
    for path, digest in upstream.items():
        if _sha(path) != digest:
            raise ValueError(f'Changed upstream input: {path}')
    for path, digest in summary.get('analysis_source_sha256', {}).items():
        if _sha(path) != digest:
            raise ValueError(f'Analysis source changed after summarization: {path}')
    if (folder / 'PLAN.json').exists():
        source_files[str((folder / 'PLAN.json').resolve())] = _sha(folder / 'PLAN.json')
    return summary, data, source_files


def _range(values):
    """A visible 5-percentage-point grid with margin, including every real fold."""
    values = np.asarray(values, dtype=float) * 100
    margin = max(1., .10 * float(values.max() - values.min()))
    lower = max(0., 5 * np.floor((values.min() - margin) / 5))
    upper = min(100., 5 * np.ceil((values.max() + margin) / 5))
    if upper <= lower:
        lower, upper = max(0., lower - 5), min(100., upper + 5)
    return float(lower), float(upper)


def _axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e5e9', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)


def _points(ax, data, stage, bounds):
    jitter = np.linspace(-.19, .19, 10)
    for x, mode in enumerate(ORDER):
        values = 100 * data[mode][stage]
        ax.scatter(x + jitter, values, color=COLORS[mode], s=40, alpha=.84,
                   edgecolors='white', linewidth=.4, zorder=3, clip_on=False)
        mean = float(values.mean())
        ax.scatter(x, mean, s=92, marker='D', color='#20252b', edgecolor='white',
                   linewidth=.7, zorder=4, clip_on=False)
        ax.text(x, bounds[1] - .055 * (bounds[1] - bounds[0]), f'{mean:.3f}%',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='#20252b', zorder=5)
    ax.set_xticks(range(4), [NAMES[mode] for mode in ORDER], fontsize=12)
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(bounds)
    ax.set_yticks(np.arange(bounds[0], bounds[1] + .1, 5))
    ax.set_ylabel('ACC（%）', fontsize=13)
    _axes(ax)


def _legend():
    return [Line2D([], [], marker='o', linestyle='none', markersize=7,
                   color='#6e7781', label='小点：每折结果（10 折全部显示）'),
            Line2D([], [], marker='D', linestyle='none', markersize=8,
                   color='#20252b', label='黑色菱形：10 折等权均值')]


def _save(fig, output, filename):
    path = output / filename
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    return path


def _main(data, output):
    bounds = _range([value for mode in MODES for value in data[mode]['best']])
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    fig.subplots_adjust(left=.12, right=.96, bottom=.25, top=.755)
    _points(ax, data, 'best', bounds)
    fig.suptitle('关系来源与传播变换：四组匹配对照', y=.971, fontsize=21, fontweight='bold')
    fig.text(.5, .877, f'主指标：原规则选取的最佳测试 ACC　｜　纵轴范围：{bounds[0]:g}–{bounds[1]:g}%',
             ha='center', fontsize=12)
    fig.legend(handles=_legend(), loc='upper center', bbox_to_anchor=(.5, .848),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.12, .113,
             '自身模态关系：内容仅使用本模态的关系；全部模态关系：内容可使用六种模态关系。\n'
             '专属／共享变换：发送和接收变换是否区分内容模态；发送与接收仍分别学习。',
             ha='left', va='center', fontsize=10.5, linespacing=1.55, color='#49515a')
    fig.text(.12, .037, 'seed0，598 名患者，原十折与 1000 轮首次最高 TEST ACC 选点；折点为描述结果，不表示独立重复或显著性。',
             ha='left', fontsize=9.4, color='#606971')
    path = _save(fig, output, 'fig01_best_acc_four_groups_zh.png')
    return path, {'metric': 'best TEST ACC percentage', 'axis_range_percent': bounds,
                  'fold_points_per_arm': 10, 'arms': 4, 'fold_points_total': 40,
                  'equal_fold_means_percent': {mode: float(data[mode]['best'].mean() * 100) for mode in MODES}}


def _paired(data, output):
    differences = {identifier: 100 * (data[candidate]['best'] - data[reference]['best'])
                   for identifier, reference, candidate, _ in CONTRASTS}
    maximum = max(float(np.abs(values).max()) for values in differences.values())
    limit = max(1., .5 * np.ceil((maximum + .30) / .5))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.965, bottom=.24, top=.78, wspace=.27, hspace=.80)
    for index, (ax, (identifier, _, _, label)) in enumerate(zip(axes.ravel(), CONTRASTS)):
        values = differences[identifier]
        ax.axhline(0, color='#75808b', linestyle='--', linewidth=1.2)
        ax.plot(range(1, 11), values, marker='o', linestyle='none', markersize=7,
                color='#0072B2', zorder=3)
        ax.axhline(values.mean(), color='#D55E00', linestyle=':', linewidth=1.8)
        ax.set_title(f'({chr(97 + index)}) {label}', fontsize=13.5, loc='left', pad=12)
        ax.text(.97, .95, f'均值 {values.mean():+.4f} 个百分点', transform=ax.transAxes,
                ha='right', va='top', fontsize=10.7, color='#A54212',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
        ax.set_ylim(-limit, limit)
        ax.set_xlim(.55, 10.45)
        ax.set_xticks(range(1, 11))
        if index >= 2:
            ax.set_xlabel('原十折编号', fontsize=12)
        ax.set_ylabel('ACC 差值（百分点）', fontsize=12)
        ax.tick_params(labelbottom=True)
        _axes(ax)
    fig.suptitle('两个因素分别改变时：逐折配对差值', y=.976, fontsize=21, fontweight='bold')
    fig.text(.5, .906, f'主指标：最佳测试 ACC　｜　四面板共用纵轴：−{limit:g} 至 +{limit:g} 个百分点',
             ha='center', fontsize=12)
    handles = [Line2D([], [], marker='o', linestyle='none', color='#0072B2', label='同一折的配对差值'),
               Line2D([], [], linestyle=':', color='#D55E00', label='10 折等权平均差值'),
               Line2D([], [], linestyle='--', color='#75808b', label='无变化')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .873),
               ncol=3, frameon=False, fontsize=11)
    fig.text(.105, .100,
             '正值表示增加标题所列因素后，原选轮规则下的 ACC 更高；负值表示更低。\n'
             '每点比较同一折、同一批留出患者；各图反复涉及这 598 名患者，不能把折点当独立样本。',
             ha='left', fontsize=10.2, color='#49515a', linespacing=1.5)
    fig.text(.105, .035, '单种子探索比较；四组并非严格等参数量。完整描述性交互见 summary.json；无置信区间或显著性标记。',
             ha='left', fontsize=9.5, color='#606971')
    path = _save(fig, output, 'fig02_best_acc_paired_differences_zh.png')
    return path, {'metric': 'candidate minus reference best ACC, percentage points',
                  'shared_axis_range_percentage_points': [-limit, limit],
                  'panels': {identifier: {'reference': reference, 'candidate': candidate,
                             'matched_fold_points': 10, 'delta_percentage_points': differences[identifier].tolist(),
                             'equal_fold_mean_percentage_points': float(differences[identifier].mean())}
                             for identifier, reference, candidate, _ in CONTRASTS}}


def _auxiliary(data, output):
    bounds = _range([value for mode in MODES for stage in ('last', 'last100') for value in data[mode][stage]])
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.0), sharey=True)
    fig.subplots_adjust(left=.075, right=.97, bottom=.25, top=.715, wspace=.18)
    for ax, stage, title in zip(axes, ('last', 'last100'), ('(a) 第 1000 轮', '(b) 第 901–1000 轮平均')):
        _points(ax, data, stage, bounds)
        ax.set_title(title, fontsize=15, loc='left', pad=13)
        ax.tick_params(axis='x', labelsize=11)
    fig.suptitle('辅助观察：训练末期的 ACC', y=.972, fontsize=21, fontweight='bold')
    fig.text(.5, .883, f'独立于主指标　｜　两个面板共用纵轴范围：{bounds[0]:g}–{bounds[1]:g}%',
             ha='center', fontsize=12)
    fig.legend(handles=_legend(), loc='upper center', bbox_to_anchor=(.5, .85),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.075, .112, '小点均为一折结果；末 100 轮先在折内平均，再将十折等权平均。\n'
             '这些结果用于观察训练末期行为，不能事后替换最佳测试 ACC 主指标。',
             ha='left', fontsize=11, color='#49515a', linespacing=1.6)
    fig.text(.075, .036, 'seed0，598 名患者；共享／专属变换与关系来源定义同主图。无独立样本推断、置信区间或显著性标记。',
             ha='left', fontsize=10, color='#606971')
    path = _save(fig, output, 'fig03_auxiliary_late_acc_zh.png')
    return path, {'auxiliary_only': True, 'axis_range_percent': bounds,
                  'panels': {stage: {'fold_points_per_arm': 10, 'arms': 4,
                             'equal_fold_means_percent': {mode: float(data[mode][stage].mean() * 100)
                                                         for mode in MODES}}
                             for stage in ('last', 'last100')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, data, source_files = _load(args.input)
    font = _font()
    plt.rcParams.update({'axes.unicode_minus': False, 'savefig.dpi': 300,
                         'font.size': 11, 'axes.labelpad': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    args.output.mkdir(parents=True, exist_ok=True)
    products = []
    for operation in (_main, _paired, _auxiliary):
        path, details = operation(data, args.output)
        products.append({'file': str(path.resolve()), 'sha256': _sha(path),
                         'bytes': path.stat().st_size, 'details': details})
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'analysis': 'Completed matched 2x2 shared-mapping configuration comparison',
        'training_seed': 0, 'folds': list(range(1, 11)), 'heldout_unique_patients_per_arm': 598,
        'patient_sets_recur_across_arms_and_contrasts': True,
        'source_files_sha256': source_files,
        'validated_upstream_source_files_sha256': summary['source_sha256'],
        'validated_analysis_source_sha256': summary.get('analysis_source_sha256', {}),
        'plot_source': {'file': str(Path(__file__).resolve()), 'sha256': _sha(__file__)},
        'font': font, 'dpi': 300, 'format': 'Ordinary Chinese PNG',
        'group_order': list(ORDER), 'parameter_counts': PARAMETERS,
        'primary_endpoint': summary['selection'], 'aggregation': summary['aggregation'],
        'fold_jitter': 'Deterministic horizontal offsets solely to avoid overlap; not uncertainty.',
        'confidence_intervals_or_significance_tests': False,
        'training_or_checkpoint_inference_performed': False,
        'claims': 'Descriptive paired comparison only, not causal or independent-sample inference.',
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
