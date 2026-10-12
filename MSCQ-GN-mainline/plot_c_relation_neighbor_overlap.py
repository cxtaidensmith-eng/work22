"""Publication figures for the actual static Framework C candidate graphs.

Read saved graph analysis only. No model inference, training, random baseline,
diagnostic labels or synthetic observations are used by this plotting script.
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
MRI_BLUE, PET_ORANGE, COMMON_GREEN = '#0072B2', '#D55E00', '#009E73'
ENGLISH_LABELS = {'RISK_FACTOR': 'Risk factors', 'COGNITIVE_TEST': 'Cognition', 'ROI_AVERAGE': 'ROI average'}
CHINESE_LABELS = {'MRI': 'MRI', 'PET': 'PET', 'CSF': '脑脊液',
                  'RISK_FACTOR': '风险因素', 'COGNITIVE_TEST': '认知', 'ROI_AVERAGE': '脑区均值'}
CURVE_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#8F7300']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_names(names: list[str], chinese: bool) -> list[str]:
    lookup = CHINESE_LABELS if chinese else ENGLISH_LABELS
    return [lookup.get(name, name) for name in names]


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


def style_axes(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#AEB7BF')
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)


def save_figure(fig, output: Path, stem: str, chinese: bool) -> list[Path]:
    paths = []
    for extension in (('png',) if chinese else ('png', 'pdf')):
        path = output/f'{stem}{"_zh" if chinese else ""}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.16)
        paths.append(path)
    return paths


def draw_overlap(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    matrix = data['mean_jaccard']
    values = data['patient_jaccard'][:, data['mri'], data['pet']]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6 if chinese else 7.4),
                             gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.09, right=.965, top=.79, bottom=.24 if chinese else .30, wspace=.42)
    title = ('静态候选图的描述性分析：关系来源的邻居重合' if chinese else
             'Static candidate-graph analysis: neighbor overlap across relation sources')
    fig.suptitle(title, x=.055, y=.975, ha='left', fontsize=15.0, fontweight='bold')
    fig.text(.055, .912,
        (f'全部 {data["patients"]} 位患者；各模态 cosine top-10 候选关系与反向边取并集' if chinese else
         f'All {data["patients"]} patients | per-modality cosine top-10 support, unioned with reverse edges'),
        fontsize=10.7, color='#53616E')
    ax = axes[0]
    diagonal = np.eye(6, dtype=bool)
    masked_matrix = np.ma.masked_where(diagonal, matrix)
    cmap = plt.get_cmap('cividis').copy()
    cmap.set_bad('#E8EDF1')
    heatmap_max = .12
    if float(matrix[~diagonal].max()) > heatmap_max:
        raise ValueError('Cross-modality Jaccard exceeds declared heatmap color range')
    im = ax.imshow(masked_matrix, cmap=cmap, norm=colors.Normalize(vmin=0, vmax=heatmap_max))
    ax.set_xticks(range(6), names, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(6), names)
    ax.set_xlabel('关系来源模态' if chinese else 'Relation-source modality')
    ax.set_ylabel('关系来源模态' if chinese else 'Relation-source modality')
    ax.set_title('a  患者平均邻居 Jaccard' if chinese else 'a  Mean patient-level neighbor Jaccard',
                 loc='left', fontsize=11.5, fontweight='bold', pad=14)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.3)
    ax.tick_params(which='both', length=0)
    for r in range(6):
        for c in range(6):
            ax.text(c, r, '—' if r == c else f'{matrix[r, c]:.3f}', ha='center', va='center', fontsize=9.5,
                    color=INK if r == c or matrix[r, c] >= heatmap_max*.42 else 'white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
    bar.set_label('跨模态均值（0–0.12）' if chinese else 'Cross-modality mean (0–0.12)', fontsize=9)
    bar.ax.tick_params(labelsize=8)

    ax = axes[1]
    histogram_max = .32
    if values.max() > histogram_max:
        raise ValueError('Patient Jaccard exceeds declared histogram range')
    bins = np.linspace(0, histogram_max, 17)
    counts, edges, patches = ax.hist(values, bins=bins, color=MRI_BLUE,
                                    alpha=.83, edgecolor='white', linewidth=.8)
    mean, median = float(values.mean()), float(np.median(values))
    ax.axvline(mean, color=PET_ORANGE, linewidth=1.8,
               label=('均值' if chinese else 'Mean')+f' = {mean:.3f}')
    ax.axvline(median, color='#444D56', linewidth=1.7, linestyle='--',
               label=('中位数' if chinese else 'Median')+f' = {median:.3f}')
    ax.set_title('b  MRI / PET 邻居重合的患者分布' if chinese else 'b  Patient distribution: MRI / PET overlap',
                 loc='left', fontsize=11.5, fontweight='bold', pad=14)
    ax.set_xlim(0, histogram_max)
    ax.set_ylim(0, max(float(counts.max())*1.23, 1))
    ax.set_xlabel('每位患者的邻居 Jaccard' if chinese else 'Patient-level neighbor Jaccard')
    ax.set_ylabel('患者数' if chinese else 'Number of patients')
    ax.legend(frameon=False, loc='upper right', fontsize=9.5,
              title=f'n = {data["patients"]}', title_fontsize=9.5)
    ax.grid(axis='y', color=GRID, linewidth=.65)
    style_axes(ax)
    notes = ([
        'Jaccard 理论范围为 0–1；热图显示 0–0.12，自比较对角线为 1，灰色屏蔽。',
        '直方图范围为 0–0.32，覆盖全部患者；Jaccard 为邻居交集 / 并集，每位患者等权。',
        '使用最终对称候选支持，无自身连接；degree 可以超过 10，不使用传播权重或诊断标签。',
        '描述关系差异；不推断互补关系的分类收益，也不进行边级显著性检验。',
    ] if chinese else [
        'Jaccard has theoretical range 0–1; heatmap shows 0–0.12. Self-comparison diagonals equal 1 and are masked in gray.',
        'Histogram range 0–0.32 includes all patients; Jaccard = neighbor intersection / union, with equal patient weights.',
        'Final symmetric support has no self-edges and can have degree greater than 10; no propagation weights or diagnosis labels.',
        'Descriptive graph differences only; no inference of predictive complementarity or edge-level significance.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .12-i*.028, note, fontsize=9.0, color='#53616E')
    files = save_figure(fig, output, 'fig01_relation_neighbor_overlap', chinese)
    plt.close(fig)
    return files, dict(mean_patient_jaccard=matrix.tolist(), patients=data['patients'],
        theoretical_jaccard_range=[0, 1], heatmap_color_limits=[0, heatmap_max],
        heatmap_diagonal='Masked in gray; original self-comparison value is 1',
        histogram_axis_limits=[0, histogram_max], histogram_bins=edges.tolist(), histogram_counts=counts.astype(int).tolist(),
        mri_pet_mean=mean, mri_pet_median=median,
        mri_pet_quantiles={str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        definition='Patient-level neighbor intersection/union, then equal mean of all patients')


def draw_patient_graphs(plt, patches, lines, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    member = data['representative_membership']
    mri_members, pet_members = member[data['mri']], member[data['pet']]
    columns = np.flatnonzero(mri_members | pet_members)
    labels = [data['anonymous_neighbors'][col] for col in columns]
    categories = ['shared' if mri_members[col] and pet_members[col] else
                  'MRI_only' if mri_members[col] else 'PET_only' for col in columns]
    category_colors = {'shared': COMMON_GREEN, 'MRI_only': MRI_BLUE, 'PET_only': PET_ORANGE}
    angles = np.linspace(np.pi/2, np.pi/2+2*np.pi, len(columns), endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.045, right=.98, top=.78, bottom=.22, wspace=.10)
    title = ('同一代表患者的候选邻居：MRI 与 PET 关系' if chinese else
             'Candidate neighbors of one representative patient: MRI and PET relations')
    fig.suptitle(title, x=.055, y=.975, ha='left', fontsize=15.2, fontweight='bold')
    fig.text(.055, .912,
        ('同一匿名患者 P；两个面板使用相同邻居位置、匿名编号与集合归属颜色' if chinese else
         'The same anonymized patient P | identical neighbor positions, anonymous IDs and membership colors'),
        fontsize=10.7, color='#53616E')
    for ax, modality, panel_letter in zip(axes, (data['mri'], data['pet']), ('a', 'b')):
        selected = member[modality, columns]
        for index, (position, category, active) in enumerate(zip(positions, categories, selected)):
            color = category_colors[category]
            if active:
                ax.plot([0, position[0]], [0, position[1]], color=color, alpha=.58, linewidth=.95, zorder=1)
            ax.scatter(*position, s=85, color=color, alpha=1.0 if active else .17,
                       edgecolors='white', linewidth=.85, zorder=3)
            label_position = position*1.12
            ax.text(*label_position, labels[index], ha='center', va='center',
                    fontsize=7.9 if len(columns) <= 35 else 7.1,
                    color=INK if active else '#A3ADB6')
        ax.scatter(0, 0, s=370, color='#37474F', edgecolors='white', linewidth=1.3, zorder=4)
        ax.text(0, 0, 'P', color='white', ha='center', va='center', fontweight='bold', fontsize=12, zorder=5)
        ax.set(xlim=(-1.30, 1.30), ylim=(-1.30, 1.30), aspect='equal')
        ax.axis('off')
        degree = int(member[modality].sum())
        ax.set_title(f'{panel_letter}  {names[modality]} '+
                     (f'关系：{degree} 位邻居' if chinese else f'relations: {degree} neighbors'),
                     fontsize=12, fontweight='bold', pad=10)
    legend_labels = (['MRI / PET 共同邻居', '仅 MRI 邻居', '仅 PET 邻居'] if chinese else
                     ['Shared MRI / PET neighbor', 'MRI-only neighbor', 'PET-only neighbor'])
    handles = [lines.Line2D([], [], color=color, marker='o', linestyle='', markersize=8,
                           label=label) for color, label in zip(
        (COMMON_GREEN, MRI_BLUE, PET_ORANGE), legend_labels)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .13), frameon=False,
               ncol=3, fontsize=9.6, columnspacing=2.5)
    notes = ([
        '淡色节点：属于另一面板的邻居，本面板没有候选连接；节点颜色只表示邻居集合归属。',
        '仅绘制目标患者与邻居的连接，邻居之间的边省略；不显示诊断标签或传播权重。',
        '代表患者按 MRI/PET Jaccard 最接近全体中位数选择；并集 degree 与行序用于打破并列。',
    ] if chinese else [
        'Faint nodes belong to the other panel’s neighbor set, with no edge in this panel; colors encode set membership only.',
        'Only target-to-neighbor connections are drawn; neighbor-to-neighbor edges are omitted. No diagnosis labels or edge weights.',
        'Patient selection: closest to median MRI/PET Jaccard, then closest union degree to median, then analysis row order.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .091-i*.027, note, fontsize=9.1, color='#53616E')
    files = save_figure(fig, output, 'fig02_same_patient_neighbor_graphs', chinese)
    plt.close(fig)
    common = int((mri_members & pet_members).sum())
    union = len(columns)
    return files, dict(anonymous_patient='P', anonymous_neighbor_labels=labels,
        identical_layout_positions=positions.tolist(), membership_categories=categories,
        mri_degree=int(mri_members.sum()), pet_degree=int(pet_members.sum()),
        shared_neighbors=common, mri_only_neighbors=int((mri_members & ~pet_members).sum()),
        pet_only_neighbors=int((pet_members & ~mri_members).sum()), union_neighbors=union,
        jaccard=float(common/union), category_colors=category_colors,
        inactive_nodes='Faint, no candidate edge in that panel',
        representative_selection=data['representative_selection'])


def draw_membership_degree(plt, colors, data: dict, output: Path, chinese: bool) -> tuple[list[Path], dict]:
    names = display_names(data['names'], chinese)
    membership, degree = data['representative_membership'], data['degree']
    fig, axes = plt.subplots(2, 1, figsize=(14.0, 9.2), gridspec_kw={'height_ratios': [1.0, 1.25]})
    fig.subplots_adjust(left=.09, right=.965, top=.83, bottom=.20, hspace=.66)
    title = ('邻居集合与实际 degree：代表患者及全体患者' if chinese else
             'Neighbor membership and actual degree: representative and full-cohort analysis')
    fig.suptitle(title, x=.055, y=.975, ha='left', fontsize=15.4, fontweight='bold')
    fig.text(.055, .922,
        ('静态候选图；不使用传播权重或诊断类别；匿名编号与病例图保持一致' if chinese else
         'Static candidate support | no propagation weights or diagnosis classes | IDs consistent with the patient graphs'),
        fontsize=10.7, color='#53616E')
    ax = axes[0]
    cmap = colors.ListedColormap(['#F4F6F8', MRI_BLUE])
    ax.imshow(membership, cmap=cmap, norm=colors.Normalize(vmin=0, vmax=1), aspect='auto')
    ax.set_xticks(range(membership.shape[1]), data['anonymous_neighbors'], rotation=90,
                  fontsize=6.5 if membership.shape[1] > 65 else 7.2)
    ax.set_yticks(range(6), names)
    ax.set_xlabel('患者 P 的六模态邻居并集（匿名编号）' if chinese else
                  'Union of patient P’s neighbors across all six modalities (anonymous IDs)', labelpad=7)
    ax.set_ylabel('关系来源' if chinese else 'Relation source')
    ax.set_title('a  同一患者的六模态邻居成员矩阵' if chinese else
                 'a  Six-modality neighbor membership for the same patient', loc='left', fontsize=12, fontweight='bold', pad=14)
    ax.set_xticks(np.arange(-.5, membership.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which='minor', color='white', linewidth=.6)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.matplotlib.patches.Patch(facecolor=MRI_BLUE, label='属于邻居集合' if chinese else 'Neighbor present'),
               plt.matplotlib.patches.Patch(facecolor='#F4F6F8', edgecolor='#CDD5DD',
                                             label='不属于邻居集合' if chinese else 'Neighbor absent')]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2, loc='upper right',
              bbox_to_anchor=(1.0, 1.22))

    ax = axes[1]
    degree_max = int(degree.max())
    xmax = max(15, int(np.ceil(degree_max/5)*5))
    for modality, color in enumerate(CURVE_COLORS):
        values = np.sort(degree[:, modality])
        probabilities = np.arange(1, len(values)+1)/len(values)
        ax.step(np.r_[0, values, xmax], np.r_[0, probabilities, 1.0], where='post',
                linewidth=1.8, color=color, label=names[modality])
    ax.axvline(10, color='#8C969F', linestyle='--', linewidth=1.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.025)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel('最终对称候选图的邻居数（degree）' if chinese else 'Neighbor count in final symmetric candidate support (degree)')
    ax.set_ylabel('患者累计比例' if chinese else 'Cumulative patient proportion')
    ax.set_title('b  全体患者的实际 degree 分布' if chinese else
                 'b  Actual degree distributions across all patients', loc='left', fontsize=12, fontweight='bold', pad=14)
    ax.legend(frameon=False, loc='lower right', ncol=3, fontsize=9,
              title=f'n = {data["patients"]} '+('每个模态' if chinese else 'per modality'), title_fontsize=9)
    ax.grid(color=GRID, linewidth=.65)
    style_axes(ax)
    notes = ([
        'a：蓝色表示候选邻居成员，浅色表示非成员；显示六模态邻居并集中的全部匿名邻居。',
        'b：每位患者等权；虚线为初始 top-10 选择数，反向边并集可增加最终 degree。',
        '这些统计描述候选关系差异，不能证明跨来源传播带来分类收益。',
    ] if chinese else [
        'a: blue = candidate neighbor present; pale = absent. Every neighbor in the six-modality union is included.',
        'b: equal weight per patient; dashed line marks initial top-10 selection. Union with reverse edges may increase final degree.',
        'These statistics describe candidate relations; they do not establish a classification benefit from cross-source propagation.',
    ])
    for i, note in enumerate(notes):
        fig.text(.055, .113-i*.025, note, fontsize=9.4, color='#53616E')
    files = save_figure(fig, output, 'fig03_neighbor_membership_and_degree', chinese)
    plt.close(fig)
    return files, dict(anonymous_patient='P', anonymous_neighbors=data['anonymous_neighbors'],
        membership=membership.astype(int).tolist(), degree_x_limits=[0, xmax],
        degree_summary={data['names'][modality]: dict(n=data['patients'],
            minimum=int(degree[:, modality].min()), maximum=int(degree[:, modality].max()),
            mean=float(degree[:, modality].mean()), median=float(np.median(degree[:, modality])),
            unique_degree_values=np.unique(degree[:, modality]).astype(int).tolist(),
            counts=np.unique(degree[:, modality], return_counts=True)[1].astype(int).tolist())
            for modality in range(6)})


def load_data(source: Path) -> dict:
    npz_path, summary_path = source/'analysis.npz', source/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    with np.load(npz_path, allow_pickle=False) as saved:
        names = [str(x) for x in saved['modality_names'].tolist()]
        adjacency = saved['adjacency'].copy()
        selections = saved['directed_topk_support'].copy()
        degree = saved['degree'].copy()
        jaccard = saved['jaccard'].copy()
        intersection, union = saved['intersection'].copy(), saved['union'].copy()
        mean = saved['mean_jaccard'].copy()
        example = int(saved['example_patient'].item())
        example_union = saved['example_union'].copy()
        membership = saved['example_membership'].copy()
        mri, pet = int(saved['mri_index'].item()), int(saved['pet_index'].item())
    patients = len(adjacency[0])
    if len(names) != 6 or len(set(names)) != 6 or adjacency.shape != (6, patients, patients) or patients != 598:
        raise ValueError('Expected six distinct static graphs and all 598 patients')
    if adjacency.dtype != bool or selections.dtype != bool or membership.dtype != bool:
        raise ValueError('Graph support and membership arrays must be boolean')
    if np.diagonal(adjacency, axis1=1, axis2=2).any() or np.diagonal(selections, axis1=1, axis2=2).any():
        raise ValueError('Candidate support must not contain self-edges')
    np.testing.assert_array_equal(adjacency, adjacency.transpose(0, 2, 1))
    np.testing.assert_array_equal(adjacency, selections | selections.transpose(0, 2, 1))
    np.testing.assert_array_equal(selections.sum(axis=2), np.full((6, patients), 10))
    np.testing.assert_array_equal(degree, adjacency.sum(axis=2))
    if degree.min() < 10:
        raise ValueError('Final union support cannot have degree below directed top-10 selection')
    for a in range(6):
        for b in range(6):
            actual_intersection = (adjacency[a] & adjacency[b]).sum(axis=1)
            actual_union = (adjacency[a] | adjacency[b]).sum(axis=1)
            np.testing.assert_array_equal(intersection[a, b], actual_intersection)
            np.testing.assert_array_equal(union[a, b], actual_union)
            np.testing.assert_allclose(jaccard[a, b], actual_intersection/actual_union, atol=1e-12, rtol=0)
    np.testing.assert_allclose(mean, jaccard.mean(axis=2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(mean, summary['mean_jaccard_matrix'], atol=1e-12, rtol=0)
    if summary['sample_count'] != patients or summary['modalities'] != names:
        raise ValueError('Summary sample count or modality order does not match the NPZ')
    if names[mri] != 'MRI' or names[pet] != 'PET':
        raise ValueError('MRI/PET indices do not match modality labels')
    median = float(np.median(jaccard[mri, pet]))
    union_median = float(np.median(union[mri, pet]))
    declared_order = np.lexsort((np.arange(patients), np.abs(union[mri, pet]-union_median),
                                np.abs(jaccard[mri, pet]-median)))
    if example != int(declared_order[0]) or summary['example']['analysis_row_ordinal'] != example:
        raise ValueError('Representative patient does not match the declared median-based selection rule')
    actual_example_union = np.flatnonzero(adjacency[:, example].any(axis=0))
    np.testing.assert_array_equal(example_union, actual_example_union)
    np.testing.assert_array_equal(membership, adjacency[:, example, example_union])
    for pair in summary['pairs']:
        a, b = names.index(pair['modality_a']), names.index(pair['modality_b'])
        stats = pair['jaccard']
        for field, value in (('mean', jaccard[a, b].mean()), ('median', np.median(jaccard[a, b])),
                             ('minimum', jaccard[a, b].min()), ('maximum', jaccard[a, b].max())):
            np.testing.assert_allclose(value, stats[field], atol=1e-12, rtol=0)
    sources = [npz_path, summary_path, Path(__file__).resolve()]
    for name in ('PLAN.json', 'COMPLETE.json'):
        path = source/name
        if path.exists():
            sources.append(path)
    return dict(names=names, patients=patients, mri=mri, pet=pet,
        patient_jaccard=jaccard.transpose(2, 0, 1), mean_jaccard=mean,
        degree=degree.T, representative_membership=membership,
        anonymous_neighbors=[f'N{index+1:02d}' for index in range(len(example_union))],
        representative_selection=summary['example']['selection_criterion'],
        sources=sources, audit=dict(patient_count=patients, modality_count=6,
            symmetric_support=True, self_edges=False, directed_topk=10,
            symmetry_union_recomputed=True, degree_recomputed=True,
            all_patient_jaccard_recomputed=True, summary_statistics_checked=True,
            representative_selection_recomputed=True, representative_membership_recomputed=True,
            diagnostic_labels_used=False, propagation_weights_used=False, chance_reference_plotted=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'Result/c_relation_neighbor_overlap_20261010')
    parser.add_argument('--output', type=Path, default=ROOT/'Visualization/c_relation_neighbor_overlap_20261010')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    data = load_data(source)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager, patches, lines

    metadata = dict(analysis='Static candidate-graph descriptive analysis',
        input_directory=str(source), output_directory=str(output), modality_order=data['names'],
        source_sha256={str(path): digest(path) for path in data['sources']}, fonts={},
        audit=data['audit'],
        interpretation='Descriptive topology only, without diagnostic labels, training or classification-effect inference.')
    files = []
    for chinese in (False, True):
        font = configure_fonts(font_manager, chinese)
        metadata['fonts']['zh' if chinese else 'en'] = font
        plt.rcParams.update({'font.family': font['name'], 'font.size': 10.5,
            'axes.labelsize': 10.5, 'axes.titlesize': 12, 'axes.unicode_minus': False,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
        overlap_files, overlap_meta = draw_overlap(plt, colors, data, output, chinese)
        graph_files, graph_meta = draw_patient_graphs(plt, patches, lines, data, output, chinese)
        degree_files, degree_meta = draw_membership_degree(plt, colors, data, output, chinese)
        files.extend(overlap_files+graph_files+degree_files)
        if not chinese:
            metadata['overlap_figure'] = overlap_meta
            metadata['representative_graph_figure'] = graph_meta
            metadata['membership_degree_figure'] = degree_meta
    metadata['outputs'] = [dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size) for path in files]
    meta_path = output/'figures_metadata.json'
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=[str(path) for path in files], metadata=str(meta_path),
                         patients=data['patients']), ensure_ascii=False))


if __name__ == '__main__':
    main()
