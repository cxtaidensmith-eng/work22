"""CPU-only, checkpoint-backed visualization of learned C mapping parameters.

This script extracts the saved embedding/MLP directly. It does not construct a
patient model, run a classifier, change checkpoints, or start any training.
"""
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib_shared_vis'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.nn import functional as F

from Model.relation_sheaf import conformal_maps


OUT = ROOT / 'Visualization/c_shared_mapping_20261003'
EXPERIMENT = ROOT / 'Result/shared_mapping_seed0_20261003'
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0/content'
COLORS = {'C': '#226B9C', 'SHARED_MAP': '#DC8736'}
MODALITIES = ['MRI', 'PET', 'CSF', 'RISK_FACTOR', 'COGNITIVE_TEST', 'ROI_AVERAGE']
LABELS = ['MRI', 'PET', 'CSF', 'Risk\nfactors', 'Cognition', 'ROI\naverage', 'Shared']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrap_degrees(radians):
    return torch.atan2(torch.sin(radians), torch.cos(radians)) * (180.0 / math.pi)


def extract_checkpoint(path, mode, fold, stage, complete, manifest):
    digest = sha(path)
    if mode == 'C':
        if manifest[relative(path)] != digest:
            raise ValueError(f'Frozen reference hash mismatch: {path}')
    elif complete[f'{stage}_checkpoint_sha256'] != digest:
        raise ValueError(f'Completed checkpoint hash mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if int(checkpoint['fold']) != fold or int(checkpoint['seed']) != 0:
        raise ValueError(f'Fold or seed mismatch: {path}')
    if checkpoint['mode'] != ('content' if mode == 'C' else mode):
        raise ValueError(f'Mode mismatch: {path}')
    epoch = checkpoint['best']['epoch'] if mode == 'C' else checkpoint['epoch']
    if int(epoch) != int(complete[stage]['epoch']):
        raise ValueError(f'Checkpoint and completed record disagree on epoch: {path}')
    state = checkpoint['state_dict']
    embedding = state['propagation.content_id.weight']
    expected_rows = 6 if mode == 'C' else 1
    if list(embedding.shape) != [expected_rows, 16] or embedding.device.type != 'cpu':
        raise ValueError(f'Unexpected content embedding: {path}')
    with torch.no_grad():
        hidden = F.gelu(F.linear(embedding, state['propagation.conditioner.0.weight'],
                                state['propagation.conditioner.0.bias']))
        parameters = F.linear(hidden, state['propagation.conditioner.2.weight'],
                              state['propagation.conditioner.2.bias'])
        send, receive, send_scale, receive_scale = conformal_maps(parameters)
        angles = math.pi * torch.tanh(parameters[:, :2])
        scales = 2.0 * torch.sigmoid(parameters[:, 2:])
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rotations = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2, 2)
        manual = scales[:, :, None, None] * rotations
        torch.testing.assert_close(send, manual[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(receive, manual[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(send_scale, scales[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(receive_scale, scales[:, 1], rtol=0, atol=0)
        relative_angle = wrap_degrees(angles[:, 0] - angles[:, 1])
        normalized_relative = torch.bmm(receive.transpose(-1, -2), send)
        normalized_relative /= (send_scale * receive_scale)[:, None, None]
        from_matrices = torch.atan2(normalized_relative[:, 1, 0],
                                    normalized_relative[:, 0, 0])
        torch.testing.assert_close(wrap_degrees(from_matrices - relative_angle * math.pi / 180),
                                   torch.zeros_like(relative_angle), atol=2e-5, rtol=0)
    if not all(torch.isfinite(value).all() for value in (parameters, send, receive, relative_angle)):
        raise ValueError(f'Nonfinite mapping parameters: {path}')
    rows = []
    for index in range(expected_rows):
        row = dict(model=mode, stage=stage, fold=fold, epoch=int(epoch),
                   modality=MODALITIES[index] if mode == 'C' else 'SHARED',
                   modality_index=index if mode == 'C' else -1,
                   send_scale=float(send_scale[index]), receive_scale=float(receive_scale[index]),
                   relative_rotation_degrees=float(relative_angle[index]),
                   relative_scale_product=float(send_scale[index] * receive_scale[index]),
                   checkpoint=relative(path), checkpoint_sha256=digest)
        for index_parameter, name in enumerate(('send_angle_logit', 'receive_angle_logit',
                                                'send_scale_logit', 'receive_scale_logit')):
            row[name] = float(parameters[index, index_parameter])
        for name, matrices in (('send', send), ('receive', receive)):
            for i in range(2):
                for j in range(2):
                    row[f'{name}_{i}{j}'] = float(matrices[index, i, j])
        rows.append(row)
    return rows, dict(path=relative(path), sha256=digest, model=mode, stage=stage,
                      fold=fold, epoch=int(epoch), extracted_mappings=expected_rows)


def circular_summary(degrees):
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    real, imaginary = float(np.cos(radians).mean()), float(np.sin(radians).mean())
    resultant = float(np.hypot(real, imaginary))
    # Predeclared concentration rule: diffuse circular samples have no mean mark.
    valid = resultant >= 0.2
    mean = float(np.rad2deg(np.arctan2(imaginary, real))) if valid else None
    return dict(circular_mean_degrees=mean, resultant_length=resultant,
                mean_displayed=valid, count=len(radians))


def statistics(rows):
    result = []
    for mode, stage in (('C', 'best'), ('SHARED_MAP', 'best'), ('SHARED_MAP', 'last')):
        modalities = MODALITIES if mode == 'C' else ['SHARED']
        for modality in modalities:
            values = [r for r in rows if (r['model'], r['stage'], r['modality']) == (mode, stage, modality)]
            if len(values) != 10 or sorted(r['fold'] for r in values) != list(range(1, 11)):
                raise ValueError(f'Expected 10 distinct folds for {mode}/{stage}/{modality}')
            item = dict(model=mode, stage=stage, modality=modality, n_folds=10)
            for key in ('send_scale', 'receive_scale'):
                data = np.asarray([r[key] for r in values])
                item.update({f'{key}_mean': float(data.mean()), f'{key}_median': float(np.median(data)),
                             f'{key}_minimum': float(data.min()), f'{key}_maximum': float(data.max())})
            circle = circular_summary([r['relative_rotation_degrees'] for r in values])
            item.update(circle)
            item['relative_rotation_minimum_degrees'] = min(r['relative_rotation_degrees'] for r in values)
            item['relative_rotation_maximum_degrees'] = max(r['relative_rotation_degrees'] for r in values)
            result.append(item)
    return result


def plot(rows, summaries):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2))
    fig.subplots_adjust(left=0.061, right=0.99, bottom=0.26, top=0.80, wspace=0.29)
    coordinates = list(range(6)) + [6.65]
    jitter = np.linspace(-0.16, 0.16, 10)
    specifications = [
        ('send_scale', 'a  Sending scale', 'Scale factor', (0, 2), 1),
        ('receive_scale', 'b  Receiving scale', 'Scale factor', (0, 2), 1),
        ('relative_rotation_degrees', 'c  Relative rotation', r'$\theta_S-\theta_T$ (degrees)', (-180, 180), 0),
    ]
    for ax, (key, title, ylabel, limits, baseline) in zip(axes, specifications):
        ax.axhline(baseline, color='#888888', linestyle=(0, (4, 3)), linewidth=0.85, zorder=0)
        ax.axvline(5.80, color='#D9DEE3', linewidth=0.85, zorder=0)
        # Thin lines identify within-fold C profiles; they do not imply trend or causality.
        for fold in range(1, 11):
            values = [next(row[key] for row in rows if row['model'] == 'C' and row['stage'] == 'best'
                           and row['fold'] == fold and row['modality'] == modality) for modality in MODALITIES]
            # Wrapped angles across a boundary are not joined through the center of the circle.
            if key == 'relative_rotation_degrees':
                for j in range(5):
                    if abs(values[j+1] - values[j]) < 180:
                        ax.plot([coordinates[j] + jitter[fold-1], coordinates[j+1] + jitter[fold-1]],
                                values[j:j+2], color=COLORS['C'], alpha=0.15, linewidth=0.7, zorder=1)
            else:
                ax.plot(np.arange(6) + jitter[fold-1], values, color=COLORS['C'],
                        alpha=0.15, linewidth=0.7, zorder=1)
        for index, modality in enumerate(MODALITIES + ['SHARED']):
            mode = 'C' if index < 6 else 'SHARED_MAP'
            chosen = sorted([row for row in rows if (row['model'], row['stage'], row['modality'])
                             == (mode, 'best', modality)], key=lambda row: row['fold'])
            data = [row[key] for row in chosen]
            ax.scatter(coordinates[index] + jitter, data, s=22, color=COLORS[mode],
                       alpha=0.66, edgecolors='white', linewidths=0.3, zorder=3)
            summary = next(item for item in summaries if (item['model'], item['stage'], item['modality'])
                           == (mode, 'best', modality))
            center = (summary['circular_mean_degrees'] if key == 'relative_rotation_degrees'
                      else summary[f'{key}_median'])
            if center is not None:
                ax.plot([coordinates[index] - 0.26, coordinates[index] + 0.26], [center, center],
                        color=COLORS[mode], linewidth=2.6, solid_capstyle='round', zorder=4)
        ax.set(xlim=(-0.6, 7.15), ylim=limits, ylabel=ylabel, xticks=coordinates,
               xticklabels=LABELS)
        ax.set_title(title, loc='left', fontweight='bold', pad=10)
        ax.tick_params(axis='x', labelsize=8, length=0, pad=9)
        ax.grid(axis='y', color='#E7EAED', linewidth=0.6, zorder=0)
        if key == 'relative_rotation_degrees':
            ax.set_yticks([-180, -90, 0, 90, 180])
        else:
            ax.set_yticks([0, 0.5, 1, 1.5, 2])
    fig.suptitle('Learned send/receive mapping parameters', x=0.061, y=0.967,
                 ha='left', fontsize=16, fontweight='bold')
    fig.text(0.061, 0.892, 'TAD three-class task  |  seed 0  |  10 folds  |  first maximum test-ACC checkpoints',
             color='#535D67', fontsize=10)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=COLORS['C'], markersize=6, label='Original C'),
                        Line2D([], [], marker='o', linestyle='none', color=COLORS['SHARED_MAP'], markersize=6, label='Shared mapping')],
               loc='lower left', bbox_to_anchor=(0.055, 0.115), ncol=2, frameon=False, fontsize=9)
    fig.text(0.061, 0.087, 'Dots: all 10 folds. Bars: scale medians; circular means for relative rotation. '
             'Faint lines link C modalities within the same fold.', fontsize=8.3, color='#535D67')
    fig.text(0.061, 0.041, 'These are learned operator parameters, not modality importance or clinical effects. '
             'Only relative rotation is shown; common rotations are not identifiable.', fontsize=8.3, color='#535D67')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'fig04_mapping_parameters.{extension}', dpi=300)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = read(EXPERIMENT / 'PLAN.json')
    if plan['modalities'] != MODALITIES or plan['training_seed'] != 0 or plan['folds'] != list(range(1, 11)):
        raise ValueError('Unexpected frozen experiment modality order or protocol')
    audit = read(EXPERIMENT / 'audit.json')
    if not audit['passed'] or audit['verified_checkpoints'] != 20:
        raise ValueError('Expected completed successful shared-mapping checkpoint audit')
    sources, rows = [], []
    for mode, stage in (('C', 'best'), ('SHARED_MAP', 'best'), ('SHARED_MAP', 'last')):
        base = REFERENCE if mode == 'C' else EXPERIMENT / 'seed_0' / mode
        for fold in range(1, 11):
            folder = base / f'fold_{fold:02d}'
            extracted, source = extract_checkpoint(folder / f'{stage}.pt', mode, fold, stage,
                                                   read(folder / 'complete.json'), plan['reference_manifest'])
            rows.extend(extracted)
            sources.append(source)
    if len(rows) != 80 or len(sources) != 30:
        raise ValueError('Expected 60 C best, 10 shared best, and 10 shared last mappings')
    summaries = statistics(rows)
    write_csv(OUT / 'fig04_mapping_parameters_all_checkpoints.csv', rows)
    write_csv(OUT / 'fig04_mapping_parameter_summary.csv', summaries)
    plot(rows, summaries)
    files = sorted(OUT.glob('fig04*'))
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(), task='TAD three-class, seed0, original ten folds',
        figure='fig04_mapping_parameters', figure_stages=['C best', 'SHARED_MAP best'],
        csv_supplement='SHARED_MAP last only; C last checkpoint weights do not exist in the reference archive',
        checkpoints_loaded=30, extracted_parameter_sets=80, source_checkpoints=sources,
        script=dict(path=relative(Path(__file__)), sha256=sha(__file__)),
        mapping_formula=dict(angles='pi * tanh(parameters[..., :2])',
                             scales='2 * sigmoid(parameters[..., 2:])',
                             map='scale * [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]',
                             relative_angle='wrap_to_[-180,180](theta_S - theta_T), in degrees',
                             verification='send/receive matrices exactly match Model.relation_sheaf.conformal_maps; '
                                          'relative angle independently checked from normalized T^T S, atol 2e-5 degrees'),
        reading=dict(scale_bars='median across 10 folds', relative_angle_bars='circular mean across 10 folds',
                     circular_mean_rule='Display only if mean resultant length >= 0.2; all fold values always shown',
                     axes='Full scale domain 0 to 2 and full relative-angle domain -180 to 180 degrees',
                     thin_lines='C modalities within each fold; wrapped angle boundary jumps are not connected',
                     shared_column='One learned mapping used by all six content modalities'),
        statistics=summaries,
        interpretation=[
            'C mapping parameters depend on content modality, not on individual patient, relation source, or input state.',
            'Relative rotation removes an arbitrary common rotation shared by S and T; raw angles are not an identifiable importance score.',
            'Scale changes also affect the degree normalization and two composed graph passes. A scale alone is not a linear contribution or reliability weight.',
            'Observed mapping differences are descriptive. They do not identify a causal reason for classification changes or establish clinical modality importance.',
            'Different folds have distinct training data and learned coordinates. Circular summaries describe fitted relative rotations, not a shared anatomical coordinate system.',
            'Best checkpoints use the retained first maximum TEST ACC selection; this figure is an exploratory single-seed mechanism description.',
            'No training, GPU computation, prediction intervention, or checkpoint modification was performed.',
        ],
        output_sha256={path.name: sha(path) for path in files},
    )
    (OUT / 'maps_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(checkpoints=30, parameter_sets=80, files=[path.name for path in files],
                          output=str(OUT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
