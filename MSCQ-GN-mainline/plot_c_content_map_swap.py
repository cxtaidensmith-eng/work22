"""Visualize real Framework C frozen-checkpoint complete S/T map swaps.

All thirty off-diagonal swaps and six identity controls are displayed. This
script only reads completed saved outputs: no training or model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INK, GRID = '#263238', '#E5E9ED'
BLUE, ORANGE = '#0072B2', '#D55E00'
ENGLISH_LABELS = {'RISK_FACTOR': 'Risk factors', 'COGNITIVE_TEST': 'Cognition', 'ROI_AVERAGE': 'ROI average'}
CHINESE_LABELS = {'MRI': 'MRI', 'PET': 'PET', 'CSF': '脑脊液',
                  'RISK_FACTOR': '风险因素', 'COGNITIVE_TEST': '认知', 'ROI_AVERAGE': '脑区均值'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def display_names(names: list[str], chinese: bool) -> list[str]:
    lookup = CHINESE_LABELS if chinese else ENGLISH_LABELS
    return [lookup.get(name, name) for name in names]


def compact_scientific(value: float) -> str:
    if value == 0:
        return '0'
    return f'{value:+.2e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def classification_values(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=-1)
    logsum = maximum + np.log(np.exp(values-maximum[..., None]).sum(axis=-1))
    labels = np.broadcast_to(labels, values.shape[:-1])
    true = np.take_along_axis(values, labels[..., None], axis=-1)[..., 0]
    return logsum-true, values.argmax(axis=-1)


def load_data(source: Path) -> dict:
    complete_path = source/'COMPLETE.json'
    if not complete_path.exists():
        raise RuntimeError('Wait for COMPLETE.json: partial experiment outputs must not be plotted')
    complete = read_json(complete_path)
    if complete.get('state') != 'complete' or complete.get('folds') != 10 or complete.get('sample_count') != 598:
        raise ValueError('Expected completed ten-fold, 598-patient analysis')
    if complete.get('combinations') != 36 or complete.get('no_training') is not True:
        raise ValueError('Expected all 36 map assignments and a frozen-checkpoint analysis')
    summary_path, aggregate_path, plan_path = source/'summary.json', source/'aggregate.npz', source/'PLAN.json'
    summary, plan = read_json(summary_path), read_json(plan_path)
    with np.load(aggregate_path, allow_pickle=False) as saved:
        aggregate = {name: saved[name].copy() for name in saved.files}
    names = [str(x) for x in aggregate['modality_names'].tolist()]
    if len(names) != 6 or len(set(names)) != 6 or summary['modalities'] != names:
        raise ValueError('Expected a consistent order of six content modalities')
    np.testing.assert_array_equal(aggregate['folds'], np.arange(1, 11))
    sources = [Path(__file__).resolve(), complete_path, summary_path, aggregate_path, plan_path]
    fold_means, relative_means, fold_accuracies = [], [], []
    counts = {name: np.zeros((6, 6), dtype=np.int64) for name in ('changed', 'corrected', 'harmed')}
    union_counts = {name: 0 for name in counts}
    heldout_rows, fold_samples = [], []
    diagonal = np.arange(6)
    for fold in range(1, 11):
        path, audit_path = source/f'fold_{fold:02d}.npz', source/f'fold_{fold:02d}_audit.json'
        sources.extend((path, audit_path))
        with np.load(path, allow_pickle=False) as saved:
            a = {name: saved[name].copy() for name in saved.files}
        if [str(x) for x in a['modality_names'].tolist()] != names or int(a['fold']) != fold:
            raise ValueError(f'Fold or modality order mismatch in {path.name}')
        labels, rows = a['labels'].astype(np.int64), a['heldout_row_ordinals']
        n = len(labels)
        if rows.shape != (n,) or len(np.unique(rows)) != n:
            raise ValueError(f'Invalid held-out row ordinals in {path.name}')
        if a['baseline_logits'].shape != (n, 3) or a['swapped_logits'].shape != (6, 6, n, 3):
            raise ValueError(f'Unexpected logits dimensions in {path.name}')
        if not np.isfinite(a['swapped_logits']).all() or not np.isfinite(a['baseline_logits']).all():
            raise ValueError('Logits must be finite')
        base_nll, base_prediction = classification_values(a['baseline_logits'], labels)
        new_nll, new_prediction = classification_values(a['swapped_logits'], labels)
        delta = new_nll-base_nll[None, None, :]
        np.testing.assert_allclose(a['baseline_nll'], base_nll, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['swapped_nll'], new_nll, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['delta_nll'], delta, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a['swapped_logits'][diagonal, diagonal],
                                      np.broadcast_to(a['baseline_logits'], (6, n, 3)))
        base_correct = base_prediction == labels
        new_correct = new_prediction == labels[None, None, :]
        recomputed = dict(changed=new_prediction != base_prediction[None, None, :],
                          corrected=(~base_correct[None, None, :]) & new_correct,
                          harmed=base_correct[None, None, :] & (~new_correct))
        for name, array in recomputed.items():
            np.testing.assert_array_equal(a[name], array)
            counts[name] += array.sum(axis=-1)
            union_counts[name] += int(array.any(axis=(0, 1)).sum())
        if np.any(a['corrected'] & a['harmed']) or np.any(a['corrected'] & ~a['changed']) or np.any(a['harmed'] & ~a['changed']):
            raise ValueError('Classification transition masks are inconsistent')
        for name in ('delta_nll', 'delta_margin', 'changed', 'corrected', 'harmed', 'relative_l2', 'absolute_l2'):
            if not np.all(a[name][diagonal, diagonal] == 0):
                raise ValueError(f'Identity control is not exactly zero: {fold}/{name}')
        if a['baseline_propagation'].shape[:2] != (n, 6) or a['target_propagation'].shape[:3] != (6, 6, n):
            raise ValueError('Unexpected propagation-output dimensions')
        recomputed_relative = np.zeros((6, 6, n), dtype=np.float64)
        for target in range(6):
            old = a['baseline_propagation'][:, target].astype(np.float64)
            new = a['target_propagation'][target].astype(np.float64)
            absolute = np.linalg.norm(new-old[None, :, :], axis=-1)
            relative = absolute/np.maximum(np.linalg.norm(old, axis=-1), 1e-12)[None, :]
            np.testing.assert_allclose(a['absolute_l2'][target], absolute, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(a['relative_l2'][target], relative, atol=1e-12, rtol=1e-12)
            recomputed_relative[target] = relative
        fold_means.append(delta.mean(axis=-1))
        relative_means.append(recomputed_relative.mean(axis=-1))
        fold_accuracies.append((recomputed['corrected'].sum(axis=-1)-recomputed['harmed'].sum(axis=-1))/n)
        heldout_rows.append(rows)
        audit = read_json(audit_path)
        for field in ('baseline_argmax_exact', 'prior_weights_exact', 'all_swap_edges_and_weights_exact',
                      'all_non_target_propagation_exact', 'all_diagonal_interventions_exact',
                      'full_scale_normalization_recomputed', 'unchanged_parameters_and_buffers',
                      'all_hooks_restored', 'all_eval'):
            if audit.get(field) is not True:
                raise ValueError(f'Failed intervention audit: fold={fold}, field={field}')
        if audit['output_sha256'] != digest(path):
            raise ValueError(f'Fold data hash does not match audit: {path.name}')
        fold_samples.append(dict(fold=fold, heldout_patients=n))
    np.testing.assert_array_equal(np.sort(np.concatenate(heldout_rows)), np.arange(598))
    fold_means, relative_means, fold_accuracies = map(np.asarray, (fold_means, relative_means, fold_accuracies))
    for actual, field in ((fold_means, 'fold_mean_delta_nll'), (fold_means.mean(0), 'mean_delta_nll'),
                          (fold_means.std(0, ddof=1), 'sd_delta_nll'),
                          (relative_means, 'fold_mean_relative_l2'), (relative_means.mean(0), 'mean_relative_l2'),
                          (fold_accuracies, 'fold_delta_acc'), (fold_accuracies.mean(0), 'mean_delta_acc'),
                          ((counts['corrected']-counts['harmed'])/598, 'pooled_delta_acc')):
        np.testing.assert_allclose(aggregate[field], actual, atol=1e-12, rtol=1e-12)
    for name in counts:
        np.testing.assert_array_equal(aggregate[name+'_counts'], counts[name])
        np.testing.assert_array_equal(summary[name+'_counts'], counts[name])
        if summary[f'unique_patients_{name}_under_any_swap'] != union_counts[name]:
            raise ValueError(f'Unique-patient union count mismatch: {name}')
    np.testing.assert_allclose(summary['mean_delta_nll_matrix'], fold_means.mean(0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(summary['mean_relative_l2_matrix'], relative_means.mean(0), atol=1e-12, rtol=0)
    for path in sources[2:]:
        entry = complete.get('files', {}).get(path.name)
        if entry is None or entry['sha256'] != digest(path) or entry['bytes'] != path.stat().st_size:
            raise ValueError(f'Completed-result manifest mismatch: {path.name}')
    return dict(names=names, mean_nll=fold_means.mean(0), fold_nll=fold_means,
        relative_l2=relative_means.mean(0), counts=counts, union_counts=union_counts,
        fold_samples=fold_samples, patients=598, sources=sources, protocol=plan,
        audit=dict(complete_result_manifest_verified=True, fold_count=10, patients=598,
                   all_nll_recomputed=True, all_classification_counts_recomputed=True,
                   all_target_relative_l2_recomputed=True, all_identity_controls_exactly_zero=True,
                   aggregate_and_unique_union_counts_verified=True,
                   recorded_intervention_invariants_passed=True))


def configure_fonts(font_manager, chinese: bool) -> dict:
    if not chinese:
        return dict(name='DejaVu Sans', path=font_manager.findfont('DejaVu Sans'))
    installed = {entry.name: entry.fname for entry in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Microsoft YaHei UI', 'Noto Sans CJK SC'):
        if name in installed:
            return dict(name=name, path=installed[name])
    for filename in ('msyh.ttc', 'simhei.ttf'):
        path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return dict(name=font_manager.FontProperties(fname=str(path)).get_name(), path=str(path))
    raise RuntimeError('No installed Chinese font found')


def save_figure(fig, output: Path, stem: str, chinese: bool) -> list[Path]:
    paths = []
    for extension in (('png',) if chinese else ('png', 'pdf')):
        path = output/f'{stem}{"_zh" if chinese else ""}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.16)
        paths.append(path)
    return paths


def annotate_matrix(ax, matrix: np.ndarray, names: list[str], chinese: bool,
                    fmt, norm, cmap, fontsize: float = 10):
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')
    ax.set_xticks(range(6), names, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(6), names)
    ax.set_xlabel('提供 S/T 映射的供体内容' if chinese else 'Donor content supplying S/T maps')
    ax.set_ylabel('被替换映射的当前内容' if chinese else 'Target content receiving donor maps')
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.25)
    ax.tick_params(which='both', length=0)
    for target in range(6):
        for donor in range(6):
            value = matrix[target, donor]
            color_position = float(norm(value))
            if norm.vmin < 0:
                text_color = 'white' if abs(color_position-.5) > .30 else INK
            else:
                text_color = 'white' if color_position < .42 else INK
            ax.text(donor, target, fmt(value), ha='center', va='center', fontsize=fontsize,
                    color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_nll_heatmap(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    matrix = data['mean_nll']
    actual_max = float(np.abs(matrix).max())
    limit = actual_max if actual_max > 0 else 1e-12
    fig, ax = plt.subplots(figsize=(9.6, 9.3))
    fig.subplots_adjust(left=.12, right=.91, top=.82, bottom=.27)
    fig.suptitle('互换不同模态的传播变换：预测损失变化' if chinese else
                 'Frozen-checkpoint S/T map-swap dependency diagnostic',
                 x=.055, y=.975, ha='left', fontsize=15.5, fontweight='bold')
    fig.text(.055, .921, '互换后 − 原模型；10 折等权均值' if chinese else
             'Mean held-out ΔNLL: swapped minus original | equal mean of all 10 folds',
             fontsize=11, color='#53616E')
    im = annotate_matrix(ax, matrix, names, chinese, compact_scientific,
                         colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), 'RdBu_r', fontsize=9.4)
    if chinese:
        ax.set_ylabel('当前传播的内容')
        ax.set_xlabel('借用哪种模态的发送／接收变换')
    ax.set_title('正值：损失升高；负值：损失降低' if chinese else
                 'Positive: higher loss | negative: lower loss', loc='left', fontsize=11.5, pad=15)
    bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.035)
    bar.set_label('平均 ΔNLL（原始损失单位）' if chinese else 'Mean ΔNLL (original loss units)', fontsize=10)
    bar.ax.tick_params(labelsize=9)
    notes = ([
        '固定 10 折历史检查点，仅互换目标内容的完整发送 / 接收映射；归一化随供体尺度重新计算。',
        '候选图、边权和其余参数保持固定；对角线为自身替换，全部指标严格为零。',
        '完整映射同时影响自身项与邻居项；互换依赖不等于模态重要性或重训泛化收益。',
        '历史测试准确率选轮；本图为描述性诊断，保留所有正负效应，不作显著性结论。',
    ] if chinese else [
        'All 10 historical checkpoints are frozen. Swap the complete send/receive pair; normalization uses the donor scales.',
        'Candidate edges, weights and all other parameters stay fixed. Diagonals are identity swaps and exactly zero.',
        'Complete maps affect self and neighbor terms. Swap dependence is not modality importance or a from-training benefit.',
        'Historical test-accuracy-selected checkpoints; descriptive diagnostics with all signed effects and no significance claim.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .145-i*.026, note, fontsize=9.0, color='#53616E')
    files = save_figure(fig, output, 'fig01_content_map_swap_nll', chinese)
    plt.close(fig)
    return files, dict(mean_delta_nll=matrix.tolist(), signed_color_limits=[-limit, limit],
        actual_max_absolute_mean=actual_max, sign='Swapped minus original; positive = higher held-out NLL',
        rows='Target/current content', columns='Donor content', folds_per_cell=10,
        heldout_patients_per_cell=598, fold_samples=data['fold_samples'],
        diagonal='All six identity swaps are exactly zero')


def draw_fold_points(plt, ticker, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    values = data['fold_nll']
    actual_max = float(np.abs(values).max())
    limit = actual_max*1.10 if actual_max > 0 else .1
    fig, axes = plt.subplots(6, 6, figsize=(13.7, 11.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.975, top=.84, bottom=.18, hspace=.31, wspace=.18)
    fig.suptitle('映射互换的逐折响应：全部 30 个互换及 6 个自身对照' if chinese else
                 'Fold-wise map-swap responses: all 30 swaps and six identity controls',
                 x=.055, y=.975, ha='left', fontsize=15.4, fontweight='bold')
    fig.text(.055, .925,
        '列：供体内容；行：当前内容；所有面板采用相同的有符号 ΔNLL 纵轴' if chinese else
        'Columns: donor content | rows: target content | identical signed ΔNLL axes in all panels',
        fontsize=11, color='#53616E')
    for target in range(6):
        for donor in range(6):
            ax = axes[target, donor]
            ax.set_xlim(.5, 10.5)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([1, 5, 10])
            ax.set_yticks([-limit/1.10, 0, limit/1.10])
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:.2g}'))
            ax.tick_params(axis='both', labelsize=7.7, length=2)
            ax.axhline(0, color='#A3ADB6', linewidth=.65, linestyle='--', zorder=1)
            if target == donor:
                ax.set_facecolor('#F1F4F6')
                ax.text(.5, .50, ('自身对照\nΔNLL = 0' if chinese else 'Identity\nΔNLL = 0'),
                        transform=ax.transAxes, ha='center', va='center', fontsize=8.8, color='#546370',
                        bbox=dict(facecolor='#F1F4F6', edgecolor='none', pad=2))
            else:
                ax.scatter(np.arange(1, 11), values[:, target, donor], s=18, color=BLUE,
                           alpha=.88, linewidths=.25, edgecolors='white', zorder=3)
                ax.axhline(values[:, target, donor].mean(), color=ORANGE, linewidth=1.1, alpha=.92, zorder=2)
            if target == 0:
                ax.set_title(names[donor], fontsize=10, pad=10, fontweight='bold')
            if donor == 0:
                ax.set_ylabel(names[target], fontsize=9.3, labelpad=8)
            if target == 5:
                ax.set_xlabel('折' if chinese else 'Fold', fontsize=8.3, labelpad=3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#C3CBD2')
    fig.text(.027, .53, '留出患者的 ΔNLL（互换 − 原模型）' if chinese else
             'Held-out ΔNLL (swapped minus original)', rotation=90, ha='center', va='center', fontsize=10.5)
    notes = ([
        '每个蓝点为一折的留出患者平均变化，橙线为 10 折等权均值；灰虚线为零。',
        '折点描述波动，不是独立重复实验或置信区间；未按结果选择或省略互换。',
        '正值为损失升高，负值为损失降低；固定模型互换可引入分布偏移，不证明重训收益。',
    ] if chinese else [
        'Blue dots: held-out patient mean within each fold. Orange line: equal mean of 10 folds. Gray dashed line: zero.',
        'Fold points describe variability, not independent replicates or confidence intervals; no swaps are selected or omitted by result.',
        'Positive = higher loss; negative = lower loss. Frozen-model swaps can induce distribution shift and do not establish a training benefit.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .090-i*.022, note, fontsize=9.0, color='#53616E')
    files = save_figure(fig, output, 'fig02_content_map_swap_fold_responses', chinese)
    plt.close(fig)
    return files, dict(fold_mean_delta_nll=values.tolist(), y_limits=[-limit, limit],
        actual_max_absolute_fold_mean=actual_max, fold_count_per_cell=10,
        x_values=list(range(1, 11)), all_off_diagonal_cells_shown=True,
        confidence_intervals=False, identity_controls_shown_as_exact_zero=True)


def draw_functional_responses(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.7))
    fig.subplots_adjust(left=.105, right=.96, top=.85, bottom=.24, wspace=.34, hspace=.46)
    fig.suptitle('映射互换后的预测与传播输出响应' if chinese else
                 'Prediction and propagation-output responses after map swaps',
                 x=.055, y=.975, ha='left', fontsize=15.5, fontweight='bold')
    fig.text(.055, .924,
        '每格均为同一批 598 位留出患者；变化与收益分别记录' if chinese else
        'Each cell uses the same 598 held-out patients | computational changes and prediction transitions are recorded separately',
        fontsize=10.6, color='#53616E')
    count_max = int(max(matrix.max() for matrix in data['counts'].values()))
    count_limit = max(1, count_max)
    relative_max = float(data['relative_l2'].max())
    relative_limit = relative_max if relative_max > 0 else 1.0
    titles = (['a  预测类别改变', 'b  错误被纠正', 'c  正确变为错误', 'd  目标内容的传播输出相对变化'] if chinese else
              ['a  Predicted class changed', 'b  Wrong prediction corrected',
               'c  Correct prediction became wrong', 'd  Target propagation: relative change'])
    for ax, name, title in zip(axes.flat, ('changed', 'corrected', 'harmed', 'relative_l2'), titles):
        count_panel = name != 'relative_l2'
        matrix = data['counts'][name] if count_panel else data['relative_l2']
        limit = count_limit if count_panel else relative_limit
        formatter = (lambda value: str(int(value))) if count_panel else (lambda value: '0' if value == 0 else f'{value:.3g}')
        im = annotate_matrix(ax, matrix, names, chinese, formatter,
                             colors.Normalize(vmin=0, vmax=limit), 'cividis', fontsize=8.6)
        ax.set_title(title, loc='left', fontsize=10.9, fontweight='bold', pad=14)
        ax.tick_params(labelsize=9.5)
        ax.xaxis.label.set_size(9.5)
        ax.yaxis.label.set_size(9.5)
        bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
        bar.set_label(('患者数' if chinese else 'Patients') if count_panel else
                      ('相对 L2（比值）' if chinese else 'Relative L2 (ratio)'), fontsize=9)
        bar.ax.tick_params(labelsize=8)
        if count_panel and count_limit <= 12:
            bar.set_ticks(np.arange(0, count_limit+1))
    changed, corrected, harmed = (data['union_counts'][key] for key in ('changed', 'corrected', 'harmed'))
    notes = ([
        f'跨全部互换的唯一患者：预测改变 {changed}，被纠正 {corrected}，受损 {harmed}。',
        '纠正与受损是预测改变的互不重叠子集；每格按唯一留出患者计数，同一患者可跨格重复。',
        '各格人数不可累加；预测改变还可能包括错类间切换，不一定等于纠正人数与受损人数之和。',
        '相对 L2 = ||互换输出 − 原输出|| / max(||原输出||, 1e−12)；先折内患者均值，再折均值。',
        '传播变化不等于分类收益；完整映射互换同时改变自身项与邻居项。',
    ] if chinese else [
        f'Union across swaps: {changed} unique patients changed class, {corrected} corrected, {harmed} harmed.',
        'Corrected and harmed are disjoint subsets of changed. Each cell counts distinct held-out patients, who can recur across cells.',
        'Cell counts must not be summed. Class changes can include wrong-to-other-wrong transitions, beyond corrected plus harmed.',
        'Relative L2 = ||swapped output - original output|| / max(||original output||, 1e-12); patient means, then equal fold mean.',
        'Propagation change is not predictive benefit. Complete map swaps affect both self and neighbor terms.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .127-i*.022, note, fontsize=9.0, color='#53616E')
    files = save_figure(fig, output, 'fig03_content_map_swap_functional_responses', chinese)
    plt.close(fig)
    return files, dict(count_matrices={name: matrix.tolist() for name, matrix in data['counts'].items()},
        shared_count_color_limits=[0, count_limit], actual_maximum_count=count_max,
        mean_relative_l2=data['relative_l2'].tolist(), relative_l2_color_limits=[0, relative_limit],
        relative_l2_actual_maximum=relative_max, relative_l2_epsilon=1e-12,
        relative_l2_definition='norm(swapped target output - baseline target output) / max(norm(baseline target output), 1e-12)',
        relative_l2_aggregation='Within-fold patient mean, then equal mean of ten folds',
        patients_per_cell=598, unique_union_counts=data['union_counts'],
        cell_counts_are_additive=False, changed_equals_corrected_plus_harmed=False,
        corrected_and_harmed_union_sets_disjoint=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'Result/c_content_map_swap_20261010')
    parser.add_argument('--output', type=Path, default=ROOT/'Visualization/c_content_map_swap_20261010')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    data = load_data(source)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager, ticker

    metadata = dict(analysis='Frozen-checkpoint complete S/T content-map swap dependency diagnostic',
        input_directory=str(source), output_directory=str(output), modality_order=data['names'],
        source_sha256={str(path): digest(path) for path in data['sources']}, fonts={},
        source_protocol=data['protocol'], data_audit=data['audit'],
        interpretation='Trained-model assignment dependence, not from-training generalization gain, modality importance or isolated neighbor-message effect.')
    files = []
    for chinese in (False, True):
        font = configure_fonts(font_manager, chinese)
        metadata['fonts']['zh' if chinese else 'en'] = font
        plt.rcParams.update({'font.family': font['name'], 'font.size': 10.5,
            'axes.labelsize': 10.5, 'axes.titlesize': 12, 'axes.unicode_minus': False,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
        main_files, main_meta = draw_nll_heatmap(plt, colors, data, output, chinese)
        fold_files, fold_meta = draw_fold_points(plt, ticker, data, output, chinese)
        response_files, response_meta = draw_functional_responses(plt, colors, data, output, chinese)
        files.extend(main_files+fold_files+response_files)
        if not chinese:
            metadata['nll_heatmap'] = main_meta
            metadata['fold_response_figure'] = fold_meta
            metadata['functional_response_figure'] = response_meta
    metadata['outputs'] = [dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size) for path in files]
    meta_path = output/'figures_metadata.json'
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=[str(path) for path in files], metadata=str(meta_path),
        signed_mean_nll_range=[float(data['mean_nll'].min()), float(data['mean_nll'].max())],
        unique_union_counts=data['union_counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
