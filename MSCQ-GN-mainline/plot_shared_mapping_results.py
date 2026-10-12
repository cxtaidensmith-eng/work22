"""Plot saved C/shared-mapping results; no training or checkpoint inference.

Outputs only fig01/02/03 artifacts, aggregate CSVs and figures_metrics.json in
Visualization/c_shared_mapping_20261003. Raw patient identifiers are not exported.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT/'Result/shared_mapping_seed0_20261003'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
OUT = ROOT/'Visualization/c_shared_mapping_20261003'
C_BLUE, SHARED_ORANGE = '#226B9C', '#DC8736'
CLASSES = ('AD', 'CN', 'SMCI')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    summary, audit, complete = [read(RESULT/name) for name in ('summary.json', 'audit.json', 'COMPLETE.json')]
    assert summary['all_new_complete'] and summary['completed_new_runs'] == 10
    assert audit['passed'] and audit['verified_runs'] == 10 and audit['verified_checkpoints'] == 20
    assert len(audit['records']) == 20 and all(r['passed'] for r in audit['records'])
    assert {(r['fold'], r['stage']) for r in audit['records']} == {(f, s) for f in range(1, 11) for s in ('best', 'last')}
    assert complete['audit'] == audit and complete['summary'] == summary
    assert not (RESULT/'RUNNING.lock').exists()
    sources = {RESULT/name for name in ('PLAN.json', 'summary.json', 'audit.json', 'COMPLETE.json')}
    curves, predictions, folds = {}, {}, []
    for mode, base in (('C', REFERENCE), ('SHARED_MAP', RESULT/'seed_0/SHARED_MAP')):
        curves[mode], predictions[mode] = [], {}
        for fold in range(1, 11):
            folder = base/f'fold_{fold:02d}'
            sources.update(folder/name for name in ('complete.json', 'epochs.csv', 'predictions.npz', 'last_predictions.npz'))
            record = read(folder/'complete.json')
            with (folder/'epochs.csv').open(encoding='utf-8-sig') as stream:
                history = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in history] == list(range(1, 1001))
            curve = np.asarray([float(r['test_acc']) for r in history])
            assert np.isfinite(curve).all() and ((curve >= 0) & (curve <= 1)).all()
            assert record['best']['epoch'] == int(curve.argmax())+1 and record['last']['epoch'] == 1000
            curves[mode].append(curve)
            predictions[mode][fold] = {}
            for stage, name in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
                with np.load(folder/name, allow_pickle=False) as saved:
                    labels, rows, logits = saved['labels'], saved['row_index'], saved['logits']
                assert len(rows) == len(np.unique(rows)) and logits.shape == (len(labels), 3)
                assert np.isfinite(logits).all()
                predicted = logits.argmax(-1)
                cm = np.bincount(3*labels.astype(int)+predicted, minlength=9).reshape(3, 3)
                denominator = cm.sum(0)+cm.sum(1)
                macro_f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator > 0).mean()
                accuracy = float((predicted == labels).mean())
                assert abs(accuracy-record[stage]['acc']) < 1e-12
                assert abs(macro_f1-record[stage]['macro_f1']) < 1e-12
                np.testing.assert_array_equal(cm, record[stage]['confusion_matrix'])
                assert abs(accuracy-(curve.max() if stage == 'best' else curve[-1])) < 1e-12
                predictions[mode][fold][stage] = dict(labels=labels, rows=rows, predicted=predicted)
                if mode == 'SHARED_MAP':
                    reference = predictions['C'][fold][stage]
                    np.testing.assert_array_equal(rows, reference['rows'])
                    np.testing.assert_array_equal(labels, reference['labels'])
            if mode == 'SHARED_MAP':
                c = curves['C'][fold-1]
                folds.append(dict(fold=fold, patients=len(labels),
                    C_best_acc=c.max(), shared_best_acc=curve.max(), best_difference_pp=100*(curve.max()-c.max()),
                    C_best_epoch=int(c.argmax())+1, shared_best_epoch=int(curve.argmax())+1,
                    C_epoch1000_acc=c[-1], shared_epoch1000_acc=curve[-1],
                    C_last100_acc=c[-100:].mean(), shared_last100_acc=curve[-100:].mean()))
        curves[mode] = np.stack(curves[mode])
        all_rows = np.concatenate([predictions[mode][f]['best']['rows'] for f in range(1, 11)])
        assert len(all_rows) == len(np.unique(all_rows)) == 598
        assert abs(curves[mode].max(1).mean()-summary['variants'][mode]['best']['acc']) < 1e-12
        assert abs(curves[mode][:, -1].mean()-summary['variants'][mode]['last']['acc']) < 1e-12
        assert abs(curves[mode][:, -100:].mean()-summary['variants'][mode]['mean_last100_acc']) < 1e-12
    stages = []
    for name, values in (('best', {m: a.max(1) for m, a in curves.items()}),
                         ('epoch1000', {m: a[:, -1] for m, a in curves.items()}),
                         ('last100', {m: a[:, -100:].mean(1) for m, a in curves.items()})):
        stages.append(dict(stage=name, C_mean_acc=float(values['C'].mean()),
            shared_mean_acc=float(values['SHARED_MAP'].mean()),
            shared_minus_C_pp=float(100*(values['SHARED_MAP']-values['C']).mean()),
            C_between_fold_sd=float(values['C'].std(ddof=0)),
            shared_between_fold_sd=float(values['SHARED_MAP'].std(ddof=0))))
    errors = []
    for stage in ('best', 'last'):
        counts = {k: np.zeros(3, dtype=int) for k in ('patients', 'corrected', 'harmed', 'C_correct', 'shared_correct')}
        for fold in range(1, 11):
            c, shared = predictions['C'][fold][stage], predictions['SHARED_MAP'][fold][stage]
            labels = c['labels']
            a, b = c['predicted'] == labels, shared['predicted'] == labels
            for key, mask in (('patients', np.ones(len(labels), dtype=bool)), ('corrected', ~a & b),
                              ('harmed', a & ~b), ('C_correct', a), ('shared_correct', b)):
                counts[key] += np.bincount(labels[mask].astype(int), minlength=3)
        for i, label in enumerate(CLASSES):
            errors.append(dict(stage=stage, class_name=label,
                **{k: int(v[i]) for k, v in counts.items()}, net=int(counts['corrected'][i]-counts['harmed'][i])))
        assert sum(r['net'] for r in errors if r['stage'] == stage) == sum(counts['shared_correct']-counts['C_correct'])
    return summary, curves, folds, stages, errors, sorted(sources)


def trailing(values, window=25):
    cumulative = np.cumsum(values, axis=1)
    totals = cumulative.copy()
    totals[:, window:] -= cumulative[:, :-window]
    return totals/np.minimum(np.arange(1, values.shape[1]+1), window)[None, :]


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#B5BDC4')
    ax.tick_params(colors='#35434E', length=3)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E7EBEF', linewidth=.65)


def save_figure(fig, stem, files):
    for extension in ('png', 'pdf', 'svg'):
        path = OUT/f'{stem}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.13)
        files.append(path)


def figure_one(plt, folds, stages, files):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(left=.06, right=.97, top=.77, bottom=.19, wspace=.33)
    fig.suptitle('Content-conditioned versus shared mapping', x=.06, y=.98, ha='left', fontsize=17, weight='bold')
    ax = axes[0]
    x = np.arange(1, 11)
    c = 100*np.asarray([r['C_best_acc'] for r in folds])
    shared = 100*np.asarray([r['shared_best_acc'] for r in folds])
    for f, a, b in zip(x, c, shared):
        ax.plot([f-.13, f+.13], [a, b], color='#A8B0B7', linewidth=1.5, zorder=1)
    ax.scatter(x-.13, c, s=45, color=C_BLUE, label='Modality-specific C', zorder=3)
    ax.scatter(x+.13, shared, s=42, marker='s', color=SHARED_ORANGE, label='Shared maps', zorder=3)
    ax.set(xticks=x, xlabel='Test fold', ylabel='Best test accuracy (%)', xlim=(.5, 10.5))
    ax.set_ylim(np.floor(min(c.min(), shared.min())/2)*2-1, 100)
    ax.set_title('A  Paired best checkpoints', loc='left', pad=12, weight='bold')
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=2, loc='upper left',
               bbox_to_anchor=(.055, .925), fontsize=10)
    style_axes(ax)
    ax = axes[1]
    differences = np.asarray([r['shared_minus_C_pp'] for r in stages])
    labels = ['Best checkpoint', 'Epoch 1000', 'Last 100 epochs']
    ax.barh(np.arange(3), differences, height=.48, color=SHARED_ORANGE, zorder=3)
    for i, (value, row) in enumerate(zip(differences, stages)):
        ax.text(value+(.07 if value >= 0 else -.07), i, f'{value:+.2f}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=11, weight='bold')
        ax.text(1.73, i+.30, f"{100*row['C_mean_acc']:.2f}% → {100*row['shared_mean_acc']:.2f}%",
                fontsize=8.5, color='#596874', ha='right', va='center')
    ax.axvline(0, color='#64727D', linewidth=1)
    ax.set(yticks=np.arange(3), yticklabels=labels, xlim=(-1.12, 1.8),
           xlabel='Shared − C accuracy (percentage points)')
    ax.invert_yaxis()
    ax.set_ylim(2.52, -.52)
    ax.set_title('B  Three prespecified summaries', loc='left', pad=12, weight='bold')
    style_axes(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color='#E7EBEF', linewidth=.65)
    fig.text(.06, .065, 'Same seed 0 and 10 folds. Best = first maximum TEST ACC within each fold. Arrows show C → Shared.',
             fontsize=9, color='#596874')
    save_figure(fig, 'fig01_fold_and_stage_comparison', files)
    plt.close(fig)


def figure_two(plt, curves, files):
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.0), sharex=True, gridspec_kw={'height_ratios': [1.55, 1]})
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.16, hspace=.28)
    fig.suptitle('Training trajectories across all 1,000 epochs', x=.07, y=.98, ha='left', fontsize=17, weight='bold')
    epochs = np.arange(1, 1001)
    smooth = {m: trailing(a)*100 for m, a in curves.items()}
    for mode, label, color in (('C', 'Modality-specific C', C_BLUE), ('SHARED_MAP', 'Shared maps', SHARED_ORANGE)):
        raw = curves[mode].mean(0)*100
        mean, sd = smooth[mode].mean(0), smooth[mode].std(0, ddof=0)
        axes[0].plot(epochs, raw, color=color, linewidth=.55, alpha=.25)
        axes[0].fill_between(epochs, mean-sd, mean+sd, color=color, alpha=.10, linewidth=0)
        axes[0].plot(epochs, mean, color=color, linewidth=1.8, label=label)
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_title('A  Ten-fold mean and descriptive fold variation', loc='left', weight='bold', pad=10)
    axes[0].legend(loc='lower left', bbox_to_anchor=(-.01, 1.16), frameon=False, ncol=2, fontsize=10)
    difference = (curves['SHARED_MAP']-curves['C'])*100
    delta = (smooth['SHARED_MAP']-smooth['C']).mean(0)
    axes[1].plot(epochs, difference.mean(0), color='#7E8B96', alpha=.28, linewidth=.55)
    axes[1].plot(epochs, delta, color='#35434E', linewidth=1.35)
    axes[1].fill_between(epochs, 0, delta, where=delta >= 0, interpolate=True, color=SHARED_ORANGE, alpha=.25)
    axes[1].fill_between(epochs, 0, delta, where=delta < 0, interpolate=True, color=C_BLUE, alpha=.23)
    axes[1].axhline(0, color='#596874', linewidth=.8)
    axes[1].set(xlabel='Epoch', ylabel='Shared − C (pp)', xlim=(1, 1000))
    axes[1].set_title('B  Paired accuracy difference: positive favors shared mapping', loc='left', weight='bold', pad=10)
    for ax in axes:
        style_axes(ax)
    fig.text(.07, .08, 'Solid lines: trailing 25-epoch means (shorter window for epochs 1–24). Faint lines: raw ten-fold means.', fontsize=9, color='#596874')
    fig.text(.07, .052, 'Top shading: ±1 between-fold SD of smoothed fold curves (ddof=0); descriptive variation, not a confidence interval.', fontsize=9, color='#596874')
    fig.text(.07, .024, 'Mean per-fold best ACC is not the maximum of a mean learning curve. Same patients and split; one training seed.', fontsize=9, color='#596874')
    save_figure(fig, 'fig02_training_trajectories', files)
    plt.close(fig)
    rows = []
    for t, epoch in enumerate(epochs):
        row = {'epoch': int(epoch)}
        for mode, prefix in (('C', 'C'), ('SHARED_MAP', 'shared')):
            row.update({f'{prefix}_fold_{f+1:02d}_test_acc': float(curves[mode][f, t]) for f in range(10)})
            row.update({f'{prefix}_raw_fold_mean_acc': float(curves[mode][:, t].mean()),
                        f'{prefix}_raw_between_fold_sd': float(curves[mode][:, t].std(ddof=0)),
                        f'{prefix}_trailing25_fold_mean_acc': float(smooth[mode][:, t].mean()/100),
                        f'{prefix}_trailing25_between_fold_sd': float(smooth[mode][:, t].std(ddof=0)/100)})
        row['shared_minus_C_raw_mean_pp'] = float(difference[:, t].mean())
        row['shared_minus_C_trailing25_mean_pp'] = float(delta[t])
        rows.append(row)
    path = OUT/'fig02_epoch_curves_raw_and_smoothed.csv'
    write_csv(path, rows)
    files.append(path)


def figure_three(plt, errors, files):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    fig.subplots_adjust(left=.06, right=.98, top=.76, bottom=.19, wspace=.20)
    fig.suptitle('Patient-level corrections and new errors by class', x=.06, y=.98, ha='left', fontsize=17, weight='bold')
    for ax, stage, title in zip(axes, ('best', 'last'), ('A  Best checkpoints', 'B  Epoch 1000')):
        selected = [r for r in errors if r['stage'] == stage]
        x = np.arange(3)
        good = np.asarray([r['corrected'] for r in selected])
        bad = np.asarray([r['harmed'] for r in selected])
        a = ax.bar(x-.17, good, width=.30, color=SHARED_ORANGE, label='Corrected by shared mapping', zorder=3)
        b = ax.bar(x+.17, bad, width=.30, facecolor='white', edgecolor=SHARED_ORANGE,
                   hatch='////', linewidth=1.2, label='New errors from shared mapping', zorder=3)
        ax.bar_label(a, fontsize=11, padding=3)
        ax.bar_label(b, fontsize=11, padding=3)
        for i, row in enumerate(selected):
            ax.text(i, 9.65, f"Net {row['net']:+d}", ha='center', fontsize=11, weight='bold', color='#35434E')
        ax.set(xticks=x, xticklabels=[f'{r["class_name"]}\n(n={r["patients"]})' for r in selected], ylim=(0, 11), yticks=range(0, 10, 2))
        ax.set_title(title, loc='left', weight='bold', pad=10)
        ax.text(.02, .94, f"Total: {good.sum()} corrected, {bad.sum()} new errors; net {good.sum()-bad.sum():+d}",
                transform=ax.transAxes, fontsize=9, color='#596874')
        style_axes(ax)
    axes[0].set_ylabel('Number of patients')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc='upper left', bbox_to_anchor=(.055, .91), fontsize=10)
    fig.text(.06, .07, 'All changes compare shared mapping with C on the same held-out rows. Net = corrected − new errors.', fontsize=9, color='#596874')
    fig.text(.06, .033, 'Each stage covers 598 patients once. Best and epoch-1000 panels contain the same patients, not independent cohorts.', fontsize=9, color='#596874')
    save_figure(fig, 'fig03_classwise_error_changes', files)
    plt.close(fig)


def main():
    cache = ROOT/'tmp/matplotlib_shared_vis'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'shared-map-20261003'})
    summary, curves, folds, stages, errors, sources = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, rows in (('fig01_fold_metrics.csv', folds), ('fig01_stage_summary.csv', stages),
                       ('fig03_classwise_error_changes.csv', errors)):
        path = OUT/name
        write_csv(path, rows)
        files.append(path)
    figure_one(plt, folds, stages, files)
    figure_two(plt, curves, files)
    figure_three(plt, errors, files)
    manifest = dict(passed=True, experiment='shared_mapping_seed0_20261003',
        source_mode='Saved predictions and epoch logs only; no training or model inference',
        training_seed=0, folds=list(range(1, 11)), patients=598, stages=stages,
        classwise_error_changes=errors, dpi=300, formats=['PNG', 'PDF', 'SVG'],
        smoothing=dict(method='Trailing arithmetic mean', window=25, minimum_periods=1,
                       raw_all_fold_curves_in='fig02_epoch_curves_raw_and_smoothed.csv'),
        variation='Between-fold SD with ddof=0; descriptive only, not confidence intervals',
        caveats=['Original first maximum TEST ACC checkpoint selection; one seed.',
                 'Same patient folds reused across epochs and variants; no independent final validation.',
                 'Best-checkpoint averages differ from maxima of mean learning curves.',
                 'Class error counts are pooled; primary ACC comparisons use equal-fold means.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources},
        plotter_sha256=sha(Path(__file__)),
        output_files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in files},
        assertions=['Ten new fits and twenty checkpoints passed saved audit; COMPLETE equals audit/summary.',
                    'Every fold has 1000 epochs and first-maximum best selection.',
                    'ACC/macro-F1/confusion matrices rechecked from saved predictions.',
                    'Stage means match the frozen summary; matched prediction rows and labels agree.',
                    '598 unique held-out rows per model; no patient identifiers exported.'])
    (OUT/'figures_metrics.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, output=str(OUT), exported_files=len(files)+1,
                         source_files=len(sources), stages=stages), indent=2))


if __name__ == '__main__':
    main()
