"""Publication views of fixed-checkpoint relation/content path interventions."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib_shared_vis'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

SOURCE = ROOT/'Result/c_shared_mapping_visualization_20261003/relation_paths'
OUT = ROOT/'Visualization/c_shared_mapping_20261003'
LABELS = ['MRI', 'PET', 'CSF', 'Risk', 'Cognition', 'ROI']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (SOURCE/'COMPLETE.json').exists(), 'Fixed-checkpoint audit must finish first'
    with (SOURCE/'aggregate.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 36
    union_file = SOURCE/'posthoc_union_and_numerical_caution.json'
    union = json.loads(union_file.read_text(encoding='utf-8'))['subsets']['all_36']
    nll, errors, delta_acc, signs = [np.full((6, 6), np.nan) for _ in range(4)]
    seen = set()
    for row in records:
        r, m = int(row['relation_index']), int(row['content_index'])
        assert (r, m) not in seen
        seen.add((r, m))
        nll[r, m] = float(row['mean_delta_nll'])
        errors[r, m] = int(row['total_harmed'])-int(row['total_corrected'])
        delta_acc[r, m] = float(row['mean_delta_acc_pp'])
        signs[r, m] = int(row['positive_nll_folds'])
    assert all(np.isfinite(x).all() for x in (nll, errors, delta_acc, signs))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
    limits = []
    for ax, matrix, title, label in zip(axes, (nll, errors),
            ('a  Loss change after masking one path', 'b  Net additional classification errors'),
            ('Mean NLL(masked) - NLL(original)', 'New errors - corrected errors (patients)')):
        limit = float(np.abs(matrix).max())
        limit = max(limit, 1e-8 if matrix is nll else 1)
        limits.append(limit)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='equal')
        ax.set_xticks(range(6), LABELS, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(6), LABELS)
        ax.set_xlabel('Content being propagated')
        ax.set_ylabel('Modality defining the relation graph')
        ax.set_title(title, loc='left', pad=14, weight='bold')
        ax.set_xticks(np.arange(-.5, 6), minor=True)
        ax.set_yticks(np.arange(-.5, 6), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.3)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.tick_params(which='major', length=0)
        for r in range(6):
            for m in range(6):
                value = matrix[r, m]
                text = (f'{value:+.2g}' if abs(value) > 1e-12 else '0') if matrix is nll else f'{int(value):+d}' if value else '0'
                ax.text(m, r, text, ha='center', va='center', fontsize=9,
                    color='white' if abs(value)/limit > .58 else '#263342')
        for i in range(6):
            ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor='#334155', linewidth=1.1, zorder=5))
        bar = fig.colorbar(im, ax=ax, shrink=.80, pad=.035)
        bar.set_label(label, fontsize=9)
        bar.ax.tick_params(labelsize=8)
        if matrix is errors and limit <= 4:
            bar.set_ticks(np.arange(-int(limit), int(limit)+1))
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle('C: relation-source × content dependence', x=.055, ha='left', fontsize=16, weight='bold', y=.98)
    fig.text(.055, .918, 'TAD three-class · seed 0 · all 10 selected checkpoints · 36 separate masks per checkpoint', fontsize=10, color='#526174')
    fig.text(.055, .055, 'Positive / red: removing this path hurts. Outlined cells: same-modality paths. Each cell is a separate intervention.', fontsize=9, color='#526174')
    fig.text(.055, .025, 'Weights, maps, full two-pass self contribution and 1/6 source averaging are preserved. Sensitivities are not additive or causal.', fontsize=9, color='#526174')
    fig.text(.055, -.005, f'Across all masks, only {union["unique_changed_patients"]} distinct patients change predicted class; repeated cells do not represent distinct patients.', fontsize=9, color='#526174')
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.21, wspace=.42)
    stem = OUT/'fig05_relation_content_dependence'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.'+suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    (OUT/'fig05_relation_path_values.csv').write_bytes((SOURCE/'aggregate.csv').read_bytes())
    meta = dict(source_sha256={str(path.relative_to(ROOT)): digest(path) for path in
        (SOURCE/'aggregate.csv', SOURCE/'COMPLETE.json', union_file, Path(__file__))},
        matrix_definition='rows: relation source; columns: propagated content',
        figures=['fig05_relation_content_dependence.'+x for x in ('png','pdf','svg')],
        nll_color_limit=limits[0], error_color_limit=limits[1],
        mean_delta_nll=nll.tolist(), net_added_errors=errors.astype(int).tolist(),
        mean_delta_acc_pp=delta_acc.tolist(), positive_nll_folds=signs.astype(int).tolist(),
        interpretation='Positive means harmful removal; fixed-checkpoint dependence, not causal or additive importance.')
    (OUT/'fig05_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=str(stem), nll_range=[float(nll.min()),float(nll.max())],
        added_error_range=[int(errors.min()),int(errors.max())]), ensure_ascii=False))


if __name__ == '__main__':
    main()
