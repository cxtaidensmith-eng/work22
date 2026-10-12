"""Plot real, frozen-checkpoint edge/content diagnostics for Framework C.

This script reads the saved fold NPZ and receiver-balanced CSV metrics. It does
not train a model, run checkpoint inference, or construct illustrative data.
The two languages use exactly the same observations, axes and color limits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BLUE = '#0072B2'
ORANGE = '#D55E00'
INK = '#263238'
GRID = '#E5E9ED'
CHINESE_LABELS = {
    'MRI': 'MRI', 'PET': 'PET', 'CSF': '脑脊液', 'Risk': '风险因素',
    'Cognition': '认知', 'ROI': '脑区',
    'RISK_FACTOR': '风险因素', 'COGNITIVE_TEST': '认知', 'ROI_AVERAGE': '脑区均值',
}
ENGLISH_LABELS = {'RISK_FACTOR': 'Risk factors', 'COGNITIVE_TEST': 'Cognition', 'ROI_AVERAGE': 'ROI average'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receiver_mean(values: np.ndarray, receivers: np.ndarray) -> float:
    """Equal weight per receiver, irrespective of candidate neighborhood size."""
    _, inv = np.unique(receivers, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=np.asarray(values, dtype=np.float64))
    return float((sums / counts).mean())


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty metrics table: {path}')
    return rows


def check_close(actual: float, expected: str | float, label: str) -> None:
    expected = float(expected)
    if not np.isfinite(expected) or not np.isclose(actual, expected, atol=2e-6, rtol=2e-5):
        raise ValueError(f'Metric mismatch {label}: recomputed={actual}, CSV={expected}')


def load_data(source: Path) -> dict:
    """Validate contract, then check plotted CSV measures against saved edges."""
    content_path = source / 'fold_content_metrics.csv'
    pair_path = source / 'fold_pair_metrics.csv'
    content_rows, pair_rows = read_csv(content_path), read_csv(pair_path)
    content = {}
    pairs = {}
    for row in content_rows:
        key = (int(row['fold']), int(row['relation_index']), int(row['content_index']))
        if key in content:
            raise ValueError(f'Duplicate content metric: {key}')
        content[key] = row
    for row in pair_rows:
        key = (int(row['fold']), int(row['relation_index']),
               int(row['content_a_index']), int(row['content_b_index']))
        if key in pairs:
            raise ValueError(f'Duplicate pair metric: {key}')
        pairs[key] = row

    folds = {}
    sources = [content_path, pair_path, Path(__file__).resolve()]
    names = None
    for fold in range(1, 11):
        path = source / f'fold_{fold:02d}.npz'
        sources.append(path)
        with np.load(path, allow_pickle=False) as saved:
            fold_names = [str(x) for x in saved['modality_names'].tolist()]
            if names is None:
                names = fold_names
                if len(names) != 6 or len(set(names)) != 6:
                    raise ValueError(f'Expected six distinct modalities, received {names}')
            elif fold_names != names:
                raise ValueError(f'Modality order differs in {path.name}')
            records = []
            for relation in range(6):
                record = {}
                for name in ('weight', 'no_affinity_weight', 'constant_affinity_weight',
                             'affinity', 'relation_term', 'bias_term', 'receiver', 'sender'):
                    record[name] = np.asarray(saved[f'r{relation}_{name}']).copy()
                receivers, senders = record['receiver'], record['sender']
                if receivers.ndim != 1 or senders.shape != receivers.shape or len(receivers) == 0:
                    raise ValueError(f'Invalid edge arrays: fold={fold}, relation={relation}')
                if not np.issubdtype(receivers.dtype, np.integer) or not np.issubdtype(senders.dtype, np.integer):
                    raise ValueError('Receiver and sender arrays must contain integer row indices')
                for name in ('weight', 'no_affinity_weight', 'constant_affinity_weight',
                             'affinity', 'relation_term', 'bias_term'):
                    values = record[name]
                    if values.shape != (6, len(receivers)) or not np.isfinite(values).all():
                        raise ValueError(f'Invalid {name}: fold={fold}, relation={relation}, shape={values.shape}')
                    if name.endswith('weight') and (values.min() < -1e-7 or values.max() > 1 + 1e-7):
                        raise ValueError(f'Weight outside [0, 1]: fold={fold}, relation={relation}')
                for modality in range(6):
                    key = (fold, relation, modality)
                    row = content[key]
                    if row['relation'] != names[relation] or row['content'] != names[modality]:
                        raise ValueError(f'Modality label mismatch: {key}')
                    full = record['weight'][modality]
                    removed = record['no_affinity_weight'][modality]
                    constant = record['constant_affinity_weight'][modality]
                    check_close(receiver_mean(np.abs(full-removed), receivers),
                                row['mean_abs_drop_affinity'], f'{key}:drop_affinity')
                    check_close(receiver_mean(np.abs(full-constant), receivers),
                                row['mean_abs_replace_constant_affinity'], f'{key}:constant_affinity')
                    check_close(receiver_mean(full, receivers), row['mean_weight'], f'{key}:mean_weight')
                records.append(record)
            folds[fold] = records
    if len(content) != 10 * 6 * 6:
        raise ValueError(f'Expected 360 fold/content metrics, received {len(content)}')

    normalized_names = [x.casefold().strip() for x in names]
    if 'mri' not in normalized_names or 'pet' not in normalized_names:
        raise ValueError(f'MRI and PET labels are required; received {names}')
    mri, pet = normalized_names.index('mri'), normalized_names.index('pet')

    def pair_row(fold: int, relation: int, a: int, b: int) -> dict:
        return pairs.get((fold, relation, a, b), pairs.get((fold, relation, b, a)))

    # Every quantity used in the heatmap or component panel is checked here.
    for fold, relations in folds.items():
        for relation, record in enumerate(relations):
            receivers = record['receiver']
            for a in range(6):
                for b in range(a + 1, 6):
                    row = pair_row(fold, relation, a, b)
                    if row is None:
                        raise ValueError(f'Missing unordered pair metric {(fold, relation, a, b)}')
                    for name, field in (
                        ('weight', 'mean_abs_weight_difference'),
                        ('no_affinity_weight', 'mean_abs_no_affinity_weight_difference'),
                        ('constant_affinity_weight', 'mean_abs_constant_affinity_weight_difference'),
                        ('affinity', 'mean_abs_affinity_difference'),
                        ('relation_term', 'mean_abs_relation_term_difference'),
                        ('bias_term', 'mean_abs_bias_difference')):
                        check_close(receiver_mean(np.abs(record[name][a]-record[name][b]), receivers),
                                    row[field], f'{fold}/{relation}/{a}/{b}:{field}')
                    logits = record['affinity'].astype(np.float64) + record['relation_term'] + record['bias_term']
                    check_close(receiver_mean(np.abs(logits[a]-logits[b]), receivers),
                                row['mean_abs_logit_difference'], f'{fold}/{relation}/{a}/{b}:logit')

    matrix_full = np.zeros((6, 6), dtype=float)
    matrix_no_affinity = np.zeros_like(matrix_full)
    for relation in range(6):
        for modality in range(6):
            if modality != mri:
                rows = [pair_row(fold, relation, mri, modality) for fold in range(1, 11)]
                matrix_full[relation, modality] = np.mean([float(x['mean_abs_weight_difference']) for x in rows])
                matrix_no_affinity[relation, modality] = np.mean([float(x['mean_abs_no_affinity_weight_difference']) for x in rows])

    interventions = {}
    for field in ('mean_abs_drop_affinity', 'mean_abs_replace_constant_affinity'):
        interventions[field] = np.array([[float(content[fold, mri, modality][field])
            for modality in range(6)] for fold in range(1, 11)])
    component_fields = ['mean_abs_affinity_difference', 'mean_abs_relation_term_difference',
                        'mean_abs_bias_difference', 'mean_abs_logit_difference']
    components = np.array([[float(pair_row(fold, mri, mri, pet)[field])
        for field in component_fields] for fold in range(1, 11)])
    mri_records = [folds[fold][mri] for fold in range(1, 11)]
    pooled_full = np.concatenate([record['weight'][[mri, pet]] for record in mri_records], axis=1)
    pooled_removed = np.concatenate([record['no_affinity_weight'][[mri, pet]] for record in mri_records], axis=1)
    sample_counts = {
        'folds': 10,
        'mri_relation_edge_records': int(pooled_full.shape[1]),
        'mri_relation_receiver_rows': int(len(np.unique(np.concatenate([x['receiver'] for x in mri_records])))),
        'mri_relation_by_fold': [dict(fold=fold, edge_records=len(record['receiver']),
            receiver_rows=len(np.unique(record['receiver']))) for fold, record in enumerate(mri_records, start=1)],
        'all_relations_by_fold': [dict(fold=fold, relation=names[relation],
            edge_records=len(record['receiver']), receiver_rows=len(np.unique(record['receiver'])))
            for fold, records in folds.items() for relation, record in enumerate(records)],
    }
    patient_distributions = {}
    patient_indices = []
    for fold, record in enumerate(mri_records, start=1):
        receiver_rows, inv = np.unique(record['receiver'], return_inverse=True)
        patient_indices.append(receiver_rows)
    if len(np.unique(np.concatenate(patient_indices))) != sum(len(x) for x in patient_indices):
        raise ValueError('Held-out receiving patients appear in more than one fold')
    for setting in ('weight', 'no_affinity_weight'):
        patient_distributions[setting] = {}
        for modality in range(6):
            if modality == mri:
                continue
            values, sampling_weights, fold_samples = [], [], []
            for fold, record in enumerate(mri_records, start=1):
                receiver_rows, inv = np.unique(record['receiver'], return_inverse=True)
                counts = np.bincount(inv)
                differences = np.abs(record[setting][modality]-record[setting][mri]).astype(np.float64)
                patient_means = np.bincount(inv, weights=differences)/counts
                values.append(patient_means)
                sampling_weights.append(np.full(len(patient_means), 1.0/(10*len(patient_means))))
                fold_samples.append(dict(fold=fold, receiving_patients=len(patient_means),
                    per_patient_weight=float(1.0/(10*len(patient_means))),
                    patient_mean_absolute_differences=patient_means.tolist()))
            values, sampling_weights = np.concatenate(values), np.concatenate(sampling_weights)
            if not np.isclose(sampling_weights.sum(), 1.0, atol=1e-12):
                raise ValueError('Patient ECDF weights do not sum to one')
            patient_distributions[setting][modality] = dict(values=values,
                weights=sampling_weights, fold_samples=fold_samples)
            reference_value = (matrix_full if setting == 'weight' else matrix_no_affinity)[mri, modality]
            check_close(float(np.dot(values, sampling_weights)), reference_value,
                        f'patient_ECDF/{setting}/{modality}:equal_fold_mean')
    protocol = {}
    for filename in ('PLAN.json', 'COMPLETE.json', 'summary.json'):
        path = source/filename
        if path.exists():
            sources.append(path)
            if filename == 'PLAN.json':
                protocol = json.loads(path.read_text(encoding='utf-8-sig'))
    return dict(names=names, mri=mri, pet=pet, pooled_full=pooled_full,
        pooled_removed=pooled_removed, interventions=interventions, components=components,
        matrix_full=matrix_full, matrix_no_affinity=matrix_no_affinity,
        sources=sources, sample_counts=sample_counts,
        content_rows=len(content_rows), pair_rows=len(pair_rows), protocol=protocol,
        patient_distributions=patient_distributions)


def configure_fonts(font_manager, chinese: bool) -> dict:
    if not chinese:
        return dict(name='DejaVu Sans', path=font_manager.findfont('DejaVu Sans'))
    installed = {entry.name: entry.fname for entry in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Microsoft YaHei UI', 'Noto Sans CJK SC'):
        if name in installed:
            return dict(name=name, path=installed[name])
    # Explicit Windows paths also cover stale or incomplete Matplotlib caches.
    for filename in ('msyh.ttc', 'simhei.ttf', 'msyhbd.ttc'):
        path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return dict(name=font_manager.FontProperties(fname=str(path)).get_name(), path=str(path))
    raise RuntimeError('A real Chinese font is required for the Chinese figures; none was found')


def style_axes(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#AEB7BF')
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)


def save_figure(fig, output: Path, stem: str, chinese: bool) -> list[Path]:
    paths = []
    for extension in (('png',) if chinese else ('png', 'pdf')):
        path = output / f'{stem}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.16)
        paths.append(path)
    return paths


def draw_main(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = [CHINESE_LABELS.get(x, x) for x in data['names']] if chinese else [ENGLISH_LABELS.get(x, x) for x in data['names']]
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 11.7))
    fig.subplots_adjust(left=.085, right=.96, top=.87, bottom=.23, wspace=.38, hspace=.50)
    title = '固定检查点的描述性分析：关系与内容的边权' if chinese else 'Frozen-checkpoint descriptive analysis: relation and content weights'
    fig.suptitle(title, x=.06, y=.975, ha='left', fontsize=16, fontweight='bold')
    seed = data['protocol'].get('training_seed', 'not supplied')
    subtitle = (f'MRI 来源关系；模型参数固定；全部 10 折；训练种子 {seed}' if chinese else
                f'MRI-defined relations | fixed model parameters | all 10 folds | training seed {seed}')
    fig.text(.06, .932, subtitle, fontsize=11, color='#53616E')

    hexes = []
    for index, (ax, values) in enumerate(zip(axes[0], (data['pooled_full'], data['pooled_removed']))):
        panel_title = (('a  原始边权', 'b  去除当前内容相似度项后的边权') if chinese else
                       ('a  Original edge weights', 'b  Current-content affinity removed'))[index]
        mesh = ax.hexbin(values[0], values[1], gridsize=42, extent=(0, 1, 0, 1),
                         mincnt=1, cmap='viridis', linewidths=0, rasterized=True)
        hexes.append(mesh)
        ax.plot([0, 1], [0, 1], '--', linewidth=1, color='#7A8590', zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel('MRI 内容的边权' if chinese else 'Edge weight for MRI content')
        ax.set_ylabel('PET 内容的边权' if chinese else 'Edge weight for PET content')
        ax.set_title(panel_title, loc='left', fontsize=11.4, fontweight='bold', pad=12)
        ax.text(.04, .94, f"n = {data['sample_counts']['mri_relation_edge_records']:,}",
                transform=ax.transAxes, va='top', fontsize=9, color=INK,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=3))
        style_axes(ax)
    hex_count_max = max(float(mesh.get_array().max()) for mesh in hexes)
    shared_norm = colors.LogNorm(vmin=1, vmax=max(hex_count_max, 1.000001))
    for ax, mesh in zip(axes[0], hexes):
        mesh.set_norm(shared_norm)
        bar = fig.colorbar(mesh, ax=ax, fraction=.045, pad=.035)
        bar.set_label('候选边记录数（对数色轴）' if chinese else 'Candidate-edge records (log scale)', fontsize=9)
        bar.ax.tick_params(labelsize=8)

    ax = axes[1, 0]
    x = np.arange(6)
    jitter = np.linspace(-.105, .105, 10)
    intervention_metadata = {}
    series = (
        ('mean_abs_drop_affinity', BLUE, -.15,
         '去除相似度项' if chinese else 'Remove affinity'),
        ('mean_abs_replace_constant_affinity', ORANGE, .15,
         '以组均值替代相似度项' if chinese else 'Replace affinity by group mean'),
    )
    for field, color, offset, label in series:
        values = data['interventions'][field]
        for fold in range(10):
            ax.scatter(x+offset+jitter[fold], values[fold], s=23, color=color,
                       alpha=.65, edgecolors='white', linewidths=.35, zorder=3)
        means = values.mean(axis=0)
        ax.scatter(x+offset, means, s=95, marker='_', color=color, linewidths=2.8,
                   zorder=5, label=label)
        intervention_metadata[field] = dict(fold_values=values.tolist(), mean=means.tolist())
    ax.set_xticks(x, names, rotation=25, ha='right', rotation_mode='anchor')
    ax.set_xlim(-.55, 5.55)
    ax.set_ylim(0, max(float(v.max()) for v in data['interventions'].values())*1.30)
    ax.set_xlabel('传播内容模态' if chinese else 'Content modality')
    ax.set_ylabel('平均绝对边权变化' if chinese else 'Mean absolute edge-weight change')
    ax.set_title('c  当前内容相似度项的作用幅度' if chinese else
                 'c  Weight changes under affinity edits', loc='left', fontsize=11.4, fontweight='bold', pad=12)
    ax.legend(frameon=False, fontsize=8.8, loc='upper left')
    ax.grid(axis='y', color=GRID, linewidth=.7)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    style_axes(ax)

    ax = axes[1, 1]
    component_names = (['内容项', '关系项', '偏置项', '总差异'] if chinese else
                       ['Content\naffinity', 'Relation\nterm', 'Bias\nterm', 'Total\nlogit'])
    component_colors = [BLUE, '#009E73', ORANGE, '#555D66']
    for index, color in enumerate(component_colors):
        ax.scatter(index+jitter, data['components'][:, index], color=color, s=26,
                   edgecolors='white', linewidths=.4, alpha=.70, zorder=3)
        ax.scatter(index, data['components'][:, index].mean(), marker='_', s=190,
                   linewidths=3, color=color, zorder=4)
    ax.set_xticks(range(4), component_names)
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('MRI 与 PET 内容的 logit 差分量' if chinese else 'Components of the MRI–PET logit difference')
    ax.set_ylabel('平均绝对 logit 差' if chinese else 'Mean absolute logit difference')
    ax.set_title('d  边权差异的分量描述' if chinese else
                 'd  Components of the weight-logit difference', loc='left', fontsize=11.4, fontweight='bold', pad=12)
    ax.grid(axis='y', color=GRID, linewidth=.7)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    style_axes(ax)

    if chinese:
        notes = [
            'a–b：同一批候选边，坐标固定为 0–1；边记录并非独立受试者样本。',
            'c–d：点为 10 折结果，横线为折均值；先在每位接收患者内平均，再在折内平均。',
            'd：各分量取绝对值后不可相加；总差异先对有符号分量求和，再取绝对值。',
            '历史测试准确率选轮的检查点；本图不检验分类效用、统计显著性或因果机制。',
        ]
    else:
        notes = [
            'a–b: identical candidate-edge records; both weight axes fixed at 0–1. Edge records are not independent patient samples.',
            'c–d: dots = 10 folds; bars = fold means. Average within each receiver first, then across receivers within each fold.',
            'd: absolute components are not additive; the total is the absolute value after summing the signed components.',
            'Historical test-accuracy-selected checkpoints; no claim of predictive utility, statistical significance or causality.',
        ]
    for index, note in enumerate(notes):
        fig.text(.06, .103-index*.023, note, fontsize=9.2, color='#53616E')
    stem = 'fig01_edge_content_weights_zh' if chinese else 'fig01_edge_content_weights'
    files = save_figure(fig, output, stem, chinese)
    plt.close(fig)
    meta = dict(hexbin_gridsize=42, hexbin_count_vmin=1, hexbin_count_vmax=hex_count_max,
        hexbin_weight_axis_limits=[0, 1], relation_source=data['names'][data['mri']],
        panels_a_b=data['sample_counts'], panel_c=intervention_metadata,
        panel_d=dict(fold_values=data['components'].tolist(),
                     mean=data['components'].mean(axis=0).tolist(),
                     order=['content_affinity', 'relation_term', 'bias_term', 'total_logit'],
                     absolute_components_are_additive=False))
    return files, meta


def draw_supplement(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = [CHINESE_LABELS.get(x, x) for x in data['names']] if chinese else [ENGLISH_LABELS.get(x, x) for x in data['names']]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.8))
    fig.subplots_adjust(left=.09, right=.91, bottom=.30, top=.82, wspace=.35)
    title = ('固定检查点的描述性分析：相对于 MRI 内容的边权差异' if chinese else
             'Frozen-checkpoint descriptive analysis: weight differences from MRI content')
    fig.suptitle(title, x=.055, y=.975, ha='left', fontsize=15.5, fontweight='bold')
    subtitle = ('相同候选边上的内容差异；10 折均值；共同色轴' if chinese else
                'Content differences on identical candidate edges | 10-fold means | shared color scale')
    fig.text(.055, .913, subtitle, fontsize=11, color='#53616E')
    matrices = [data['matrix_full'], data['matrix_no_affinity']]
    actual_max = float(max(x.max() for x in matrices))
    vmax = actual_max if actual_max > 0 else 1.0
    norm = colors.Normalize(vmin=0, vmax=vmax)
    titles = (['a  原始边权', 'b  去除当前内容相似度项'] if chinese else
              ['a  Original weights', 'b  Without current-content affinity'])
    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap='cividis', norm=norm, aspect='equal')
        ax.set_xticks(range(6), names, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(6), names)
        ax.set_xlabel('传播内容模态' if chinese else 'Content modality')
        ax.set_ylabel('关系来源模态' if chinese else 'Relation-source modality')
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.set_xticks(np.arange(-.5, 6), minor=True)
        ax.set_yticks(np.arange(-.5, 6), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.4)
        ax.tick_params(which='both', length=0)
        for relation in range(6):
            for content in range(6):
                value = matrix[relation, content]
                label = '0' if value == 0 else f'{value:.3g}'
                ax.text(content, relation, label, ha='center', va='center', fontsize=9.7,
                        color='white' if value/vmax < .42 else '#17212A')
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar_ax = fig.add_axes([.932, .35, .016, .42])
    bar = fig.colorbar(im, cax=bar_ax)
    bar.set_label('平均绝对边权差（原始 0–1 单位）' if chinese else
                  'Mean absolute weight difference (original 0–1 units)', fontsize=10)
    bar.ax.tick_params(labelsize=9)
    if chinese:
        notes = [
            '每格比较同一批候选边上，该列内容与 MRI 内容的权重；MRI 列为参照零值。',
            '先在接收患者内平均，再在折内平均，最后对 10 折等权平均。',
            '两图共用 0 至实际最大值的色轴；展示计算差异，不表示分类贡献或因果重要性。',
        ]
    else:
        notes = [
            'Each cell compares that content with MRI content on identical candidate edges; the MRI reference column is zero.',
            'Average within receivers, across receivers within folds, then equally across the 10 folds.',
            'Both panels share a color axis from zero to the observed maximum; differences are not predictive or causal importance.',
        ]
    for index, note in enumerate(notes):
        fig.text(.055, .117-index*.029, note, fontsize=9.5, color='#53616E')
    stem = 'fig02_relation_content_weight_differences_zh' if chinese else 'fig02_relation_content_weight_differences'
    files = save_figure(fig, output, stem, chinese)
    plt.close(fig)
    return files, dict(reference_content=data['names'][data['mri']], color_limits=[0, vmax],
        actual_maximum=actual_max, zero_matrix_color_fallback=bool(actual_max == 0),
        matrix_full=data['matrix_full'].tolist(), matrix_no_affinity=data['matrix_no_affinity'].tolist(),
        fold_count_per_cell=10, averaging='receiver-balanced within fold, then equal fold mean',
        source_sample_counts=data['sample_counts']['all_relations_by_fold'])


def draw_patient_distributions(plt, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = [CHINESE_LABELS.get(x, x) for x in data['names']] if chinese else [ENGLISH_LABELS.get(x, x) for x in data['names']]
    modalities = [modality for modality in range(6) if modality != data['mri']]
    settings = ('weight', 'no_affinity_weight')
    actual_max = max(float(data['patient_distributions'][setting][modality]['values'].max())
                     for setting in settings for modality in modalities)
    xmax = min(1.0, max(.05, float(np.ceil(actual_max/.05)*.05)))
    curve_colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8))
    fig.subplots_adjust(left=.08, right=.965, top=.79, bottom=.26, wspace=.29)
    title = ('固定检查点的描述性分析：患者边权差异分布' if chinese else
             'Frozen-checkpoint descriptive analysis: patient weight-difference distributions')
    fig.suptitle(title, x=.055, y=.975, ha='left', fontsize=15.0, fontweight='bold')
    subtitle = ('MRI 来源关系；各内容相对于 MRI；每折等权的经验累积分布' if chinese else
                'MRI-defined relations | each content relative to MRI | equal-fold weighted empirical CDF')
    fig.text(.055, .909, subtitle, fontsize=10.8, color='#53616E')
    panel_titles = (['a  原始边权', 'b  去除当前内容相似度项'] if chinese else
                    ['a  Original weights', 'b  Current-content affinity removed'])
    meta = dict(relation_source=data['names'][data['mri']], reference_content=data['names'][data['mri']],
        receiving_patients=data['sample_counts']['mri_relation_receiver_rows'], fold_count=10,
        x_limits=[0, xmax], observed_maximum=actual_max, rounding_step=.05,
        definition='ECDF of per-receiver mean absolute edge-weight differences',
        weighting='Each fold receives weight 1/10; each receiver within a fold receives weight 1/(10*n_fold)',
        interpretation='Descriptive weighted empirical distributions; no significance or predictive utility claim.',
        curves={})
    for ax, setting, panel_title in zip(axes, settings, panel_titles):
        meta['curves'][setting] = {}
        for modality, color in zip(modalities, curve_colors):
            observation = data['patient_distributions'][setting][modality]
            order = np.argsort(observation['values'], kind='stable')
            ordered_values = observation['values'][order]
            ordered_weights = observation['weights'][order]
            cumulative = np.cumsum(ordered_weights)
            cumulative[-1] = 1.0  # floating-point closure, not a data modification
            ax.step(np.r_[0, ordered_values, xmax], np.r_[0, cumulative, 1.0],
                    where='post', linewidth=1.9, color=color, label=names[modality])
            meta['curves'][setting][data['names'][modality]] = dict(
                patient_count=len(ordered_values), fold_samples=observation['fold_samples'],
                sorted_patient_mean_differences=ordered_values.tolist(),
                sorted_patient_weights=ordered_weights.tolist(),
                weighted_cumulative_probabilities=cumulative.tolist(),
                equal_fold_mean=float(np.dot(observation['values'], observation['weights'])))
        ax.set_title(panel_title, loc='left', fontsize=11.5, fontweight='bold', pad=13)
        ax.set(xlim=(0, xmax), ylim=(0, 1.025))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel('每位患者的平均绝对边权差' if chinese else 'Per-patient mean absolute edge-weight difference')
        ax.set_ylabel('折等权累计比例' if chinese else 'Equal-fold weighted cumulative proportion')
        ax.grid(color=GRID, linewidth=.65)
        ax.legend(frameon=False, loc='lower right', fontsize=8.7,
                  title=f"n = {data['sample_counts']['mri_relation_receiver_rows']} " + ('位患者' if chinese else 'receiving patients'),
                  title_fontsize=9.0)
        style_axes(ax)
    if chinese:
        notes = [
            '先对每位接收患者的候选边求平均绝对差；每折占 1/10，折内患者等权。',
            '两图使用相同的横轴范围；范围上限为共同观测最大值向上取整至 0.05。',
            '描述权重分布差异；不进行边级显著性检验，不表示分类效用或因果重要性。',
        ]
    else:
        notes = [
            'Average absolute differences across each receiver’s candidate edges; each fold has weight 1/10, with equal receivers within fold.',
            'Both panels share the same x range, rounded up from the joint observed maximum to the next 0.05.',
            'Descriptive distributions only; no edge-level significance test, predictive-utility claim or causal interpretation.',
        ]
    for index, note in enumerate(notes):
        fig.text(.055, .12-index*.034, note, fontsize=9.1, color='#53616E')
    stem = 'fig03_patient_weight_difference_distributions_zh' if chinese else 'fig03_patient_weight_difference_distributions'
    files = save_figure(fig, output, stem, chinese)
    plt.close(fig)
    return files, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'Result/c_edge_content_weights_20261009')
    parser.add_argument('--output', type=Path, default=ROOT/'Visualization/c_edge_content_weights_20261009')
    parser.add_argument('--distribution-only', action='store_true', help='Preserve existing figures and add only patient ECDF outputs')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    data = load_data(source)
    output.mkdir(parents=True, exist_ok=True)
    # Keep the plotting cache with these outputs rather than touching other runs.
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager

    files = []
    metadata = dict(
        analysis='Frozen-checkpoint descriptive analysis',
        interpretation='Computational differences only; not predictive utility, statistical significance or causal importance.',
        input_directory=str(source), output_directory=str(output),
        modality_order=data['names'],
        averaging='Within each receiver first; equally across receivers within fold; then equally across folds.',
        edge_record_caution='Pooled candidate-edge records are not independent patient observations.',
        csv_rows=dict(content=data['content_rows'], pair=data['pair_rows']),
        source_protocol=data['protocol'],
        source_sha256={str(path): digest(path) for path in data['sources']},
        fonts={},
    )
    meta_path = output/'figures_metadata.json'
    if args.distribution_only and meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
        for key in ('main_figure', 'supplementary_figure'):
            if key in previous:
                metadata[key] = previous[key]
        files.extend(Path(record['path']) for record in previous.get('outputs', [])
                     if not Path(record['path']).name.startswith('fig03_'))
    for chinese in (False, True):
        font = configure_fonts(font_manager, chinese)
        metadata['fonts']['zh' if chinese else 'en'] = font
        plt.rcParams.update({'font.family': font['name'], 'font.size': 10.5,
            'axes.labelsize': 10.5, 'axes.titlesize': 12, 'axes.unicode_minus': False,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
        if not args.distribution_only:
            main_files, main_meta = draw_main(plt, colors, data, output, chinese)
            supplement_files, supplement_meta = draw_supplement(plt, colors, data, output, chinese)
            files.extend(main_files+supplement_files)
            if not chinese:
                metadata['main_figure'] = main_meta
                metadata['supplementary_figure'] = supplement_meta
        distribution_files, distribution_meta = draw_patient_distributions(plt, data, output, chinese)
        files.extend(distribution_files)
        if not chinese:
            metadata['patient_distribution_figure'] = distribution_meta
    metadata['outputs'] = [dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size) for path in files]
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=[str(path) for path in files], metadata=str(meta_path),
        candidate_edge_records=data['sample_counts']['mri_relation_edge_records'],
        receiver_rows=data['sample_counts']['mri_relation_receiver_rows'],
        max_weight_difference=metadata.get('supplementary_figure', {}).get('actual_maximum'),
        patient_difference_xmax=metadata['patient_distribution_figure']['x_limits'][1]), ensure_ascii=False))


if __name__ == '__main__':
    main()
