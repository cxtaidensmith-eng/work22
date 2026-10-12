"""Independent inference-only audit of the second TAD binary C search.

Does not import either tuning runner. Reuses the independent first-round
auditor's confusion-derived metrics, configuration checks and frozen-checkpoint
replay. Run after COMPLETE.json exists and RUNNING.lock is absent.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_tad_binary_fine_tune import (
    ROOT, CONFIG, KEYS, audit_fold, calculate, check_config_file, digest,
    mean_result, ordered, read, require, same, verify_manifest,
)
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import setup


DEFAULT_OUT = ROOT/'Result/tad_binary_joint_tune_20260928'
DEFAULT_PREVIOUS = ROOT/'Result/tad_binary_fine_tune_20260928'
DEFAULT_REFERENCE = ROOT/'Result/c_tadpole_binary_seed0_20260924'
BASE = dict(lr=.006, weight_decay=.0007, Drop_rate=.64,
            input_noise_std=.1, aux_loss_weight=1., Loss_rate=.0001)
CHANGES = {
    'O00001': dict(Loss_rate=.00001),
    'W002': dict(weight_decay=.002),
    'J_DN': dict(Drop_rate=.50, input_noise_std=.15),
    'J_MID': dict(Drop_rate=.57, input_noise_std=.125),
    'J_OW': dict(Loss_rate=.00001, weight_decay=.002),
    'J_LDN': dict(lr=.003, Drop_rate=.50, input_noise_std=.15),
    'J_DW': dict(Drop_rate=.70, weight_decay=.002),
}
REFERENCES = ['BASE', 'D050', 'N015']
SCREEN = [1, 3, 6, 8]
REMAINING = [2, 4, 5, 7, 9, 10]
ALL_FOLDS = list(range(1, 11))
REUSED = {name: ALL_FOLDS for name in REFERENCES} | {'O00001': [1, 6], 'W002': [1, 6]}


def resolve_folder(name, fold, out, previous, reference):
    # Resolve from the prespecified reuse set, not arbitrary files that appear
    # later in another output directory.
    if name == 'BASE':
        return reference/'content'/f'fold_{fold:02d}'
    if fold in REUSED.get(name, []):
        return previous/'runs'/name/f'fold_{fold:02d}'
    return out/'runs'/name/f'fold_{fold:02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--previous', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    out, previous, reference = args.output.resolve(), args.previous.resolve(), args.reference.resolve()
    require((out/'COMPLETE.json').is_file(), 'Training is incomplete; audit will not use the GPU')
    require(not (out/'RUNNING.lock').exists(), 'Training lock is present')
    require(not (previous/'RUNNING.lock').exists(), 'Previous search is still running')
    completion = read(out/'COMPLETE.json')
    require(completion['all_audits_passed'] is True, 'Training completion has failed audits')
    require(read(previous/'independent_audit.json')['passed'] is True, 'Previous audit did not pass')
    require(read(previous/'COMPLETE.json')['all_audits_passed'] is True, 'Previous search did not pass')
    plan, specs, analysis = (read(out/name) for name in ('PLAN.json', 'all_settings.json', 'analysis.json'))
    selection = read(out/'selection.json')
    for key, expected in dict(base=BASE, changes=CHANGES, reference_names=REFERENCES,
            candidate_names=list(CHANGES), screen_folds=SCREEN, seed=0, epochs=1000,
            maximum_new_runs=66, initial_screen_new_runs=24, reused_folds=REUSED).items():
        same(plan[key], expected, f'Prespecified plan:{key}')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    required_references = [previous/name for name in
                           ('PLAN.json', 'all_settings.json', 'COMPLETE.json', 'independent_audit.json')]
    for name, folds in REUSED.items():
        for fold in folds:
            folder = resolve_folder(name, fold, out, previous, reference)
            filenames = ['best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json',
                         'predictions.npz' if name == 'BASE' else 'best_predictions.npz', 'last_predictions.npz']
            if name != 'BASE':
                filenames += ['last.pt', 'effective_settings.json']
            required_references.extend(folder/filename for filename in filenames)
    for path in required_references:
        require(str(path.relative_to(ROOT)) in plan['reference_sha256'], f'Reused artifact not hash-pinned:{path}')
    for filename, expected in plan['code_sha256'].items():
        require(digest(out/'source_snapshot'/filename) == expected, f'Snapshot mismatch:{filename}')
    require(digest(out/'data_splits_graphs.npz') == plan['split_archive_sha256'], 'Split archive hash changed')
    base_plan = read(reference/'PLAN.json')
    same(plan['baseline_protocol'], base_plan['protocol'], 'Baseline protocol')
    previous_specs = read(previous/'all_settings.json')
    expected_specs = {name: previous_specs[name] for name in REFERENCES} | {
        name: {**BASE, **change} for name, change in CHANGES.items()}
    same(expected_specs['BASE'], BASE, 'Historical BASE settings')
    same(expected_specs['D050'], {**BASE, 'Drop_rate': .50}, 'Historical D050 settings')
    same(expected_specs['N015'], {**BASE, 'input_noise_std': .15}, 'Historical N015 settings')
    same(specs, expected_specs, 'All settings')
    same(list(specs), list(expected_specs), 'Stable insertion order')
    for name in ('O00001', 'W002'):
        same(specs[name], previous_specs[name], f'Reused settings:{name}')

    cfg, dd, data, graphs, paths = setup(args.cuda, CONFIG, 10)
    require(dd['Sample_Num'] == 535 and dd['Class_Names'] == ['SMCI', 'PMCI'], 'Wrong task')
    require(len(graphs) == 5, 'Expected the original five modality graphs')
    same(torch.bincount(data['Label']).cpu().tolist(), [490, 45], 'Class counts')
    for path in paths:
        require(digest(path) == base_plan['protocol']['data_sha256'][Path(path).name], f'Data changed:{path}')
    for key, value in base_plan['protocol']['original_config'].items():
        same(getattr(cfg, key), value, f'Unmodified base config:{key}')
    source_rows = np.asarray(dd['Index'])
    require(len(source_rows) == len(set(source_rows.tolist())) == 535, 'Source row IDs are not unique')
    masks = torch.stack([mask for _, mask in data['Mask']]).cpu().numpy()
    np.testing.assert_array_equal(masks.sum(0), np.ones(535, dtype=int))
    for train, test in data['Mask']:
        require(not torch.any(train & test).item(), 'Train/test masks overlap')
        require(torch.all(train | test).item(), 'Train/test masks do not cover the cohort')
    for archive_path in (reference/'splits_and_graphs.npz', previous/'data_splits_graphs.npz', out/'data_splits_graphs.npz'):
        with np.load(archive_path) as archive:
            np.testing.assert_array_equal(archive['source_row'], source_rows)
            np.testing.assert_array_equal(archive['labels'], data['Label'].cpu().numpy())
            np.testing.assert_array_equal(archive['test_masks'], masks)
            for index, graph in enumerate(graphs):
                np.testing.assert_array_equal(archive[f'edge_{index}'], graph['edge'].numpy())
                np.testing.assert_array_equal(archive[f'neighbors_{index}'], graph['neighbors'].numpy())
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    same(graph_hash, base_plan['protocol']['candidate_hash'], 'Original graph candidates')

    rows, predictions, details = {}, {}, {}
    new_runs = reused_runs = new_replays = reused_replays = 0
    discovered_new = {(p.parent.parent.name, int(p.parent.name.split('_')[-1]))
                      for p in (out/'runs').glob('*/fold_*/complete.json')}
    require(all(name in CHANGES for name, _ in discovered_new), 'Unexpected new-training arm')
    for name, spec in specs.items():
        require(set(spec) == set(KEYS), f'{name}: unexpected hyperparameter')
        require(all(math.isfinite(float(v)) and v > 0 for v in spec.values()), f'{name}: invalid setting')
        check_config_file(out/'configs'/f'{name}.ini', ROOT/CONFIG, spec)
        conf = copy.copy(cfg)
        for key, value in spec.items():
            setattr(conf, key, value)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        require(sum(p.numel() for p in model.parameters()) == 716396, f'{name}: network size changed')
        rows[name], predictions[name], details[name] = {}, {}, {}
        for fold in ALL_FOLDS:
            folder = resolve_folder(name, fold, out, previous, reference)
            reused = fold in REUSED.get(name, [])
            if reused:
                require((folder/'complete.json').is_file(), f'Missing reused fold:{name}/{fold}')
                require((name, fold) not in discovered_new, f'Reused fold unnecessarily retrained:{name}/{fold}')
            elif not (folder/'complete.json').is_file():
                require(not folder.exists(), f'Incomplete training folder:{folder}')
                continue
            if name != 'BASE':
                old = previous/'runs'/name/f'fold_{fold:02d}'
                require(reused or not (old/'complete.json').exists(), f'Unexpected unpinned old fold:{old}')
            result, prediction, detail = audit_fold(name, fold, folder, model, spec, data, source_rows)
            rows[name][fold], predictions[name][fold], details[name][fold] = result, prediction, detail
            replay_count = sum(int(value['checkpoint_replayed']) for value in detail.values())
            if reused:
                reused_runs += 1
                reused_replays += replay_count
            else:
                new_runs += 1
                new_replays += replay_count
            print(json.dumps(dict(audit=name, fold=fold, reused=reused, passed=True)), flush=True)
        del model
        if cfg.Device.type == 'cuda':
            torch.cuda.empty_cache()
    same(reused_runs, 34, 'Reused fold count')
    same(reused_replays, 58, 'Reused checkpoint replay count')
    same(new_replays, 2*new_runs, 'New checkpoint replay count')
    same(new_runs, len(discovered_new), 'Recognized new fold count')
    require(new_runs <= plan['maximum_new_runs'], 'New runs exceed the prespecified bound')
    for name in specs:
        require(set(SCREEN).issubset(rows[name]), f'{name}: missing screening fold')
    candidate_order = ordered(list(CHANGES), rows, SCREEN)
    reference_order = ordered(list(specs), rows, SCREEN)
    cutoff = mean_result(rows[candidate_order[1]], SCREEN)['best']['acc']
    finalists = [name for name in candidate_order
                  if mean_result(rows[name], SCREEN)['best']['acc'] >= cutoff-1e-12]
    same(selection['candidate_order'], candidate_order, 'Four-fold candidate ranking')
    same(selection['reference_order'], reference_order, 'Four-fold all-settings ranking')
    same(selection['cutoff_acc'], cutoff, 'Four-fold promotion cutoff')
    same(selection['finalists'], finalists, 'All primary-ACC ties promoted')
    require(2 <= len(finalists) <= 7, 'Finalist count outside prespecified bounds')
    same(new_runs, 24+6*len(finalists), 'Completed run budget')
    for name in specs:
        expected_folds = ALL_FOLDS if name in REFERENCES or name in finalists else SCREEN
        same(sorted(rows[name]), expected_folds, f'{name}: exact completed fold set')
    full_names = REFERENCES+finalists
    retained = ordered(full_names, rows, ALL_FOLDS)[0]
    full = {name: mean_result(rows[name], ALL_FOLDS) for name in full_names}
    same(analysis['retained'], retained, 'Retained unified configuration')
    same(completion['retained'], retained, 'Completion retained configuration')
    same(analysis['settings'], specs[retained], 'Retained settings')
    same(analysis['full'], full, 'Full tenfold means')
    same(analysis['records'], [rows[name][fold] for name in full_names for fold in ALL_FOLDS], 'Fold records')
    same(analysis['new_runs'], new_runs, 'Analysis run count')
    same(completion['new_runs'], new_runs, 'Completion run count')
    same(analysis['verified_new_checkpoints'], new_replays, 'Reported checkpoint count')
    summary = read(out/'summary.json')['candidates']
    same(set(summary), set(specs), 'Summary candidate coverage')
    for name in specs:
        same(summary[name], mean_result(rows[name], sorted(rows[name])), f'{name}: summary metrics')
    stage_comparison = {name: dict(screen=mean_result(rows[name], SCREEN),
                        remaining_six=mean_result(rows[name], REMAINING)) for name in full_names}
    same(read(out/'screen_vs_remaining.json'), stage_comparison, 'Screen versus remaining six folds')
    pairs = {}
    for name in full_names:
        for stage in ('best', 'last'):
            ids = np.concatenate([predictions[name][f][stage]['row_index'] for f in ALL_FOLDS])
            require(len(ids) == len(np.unique(ids)) == 535, f'{name}/{stage}: duplicate/missing source rows')
            np.testing.assert_array_equal(np.sort(ids), np.sort(source_rows))
        selected = [predictions[name][f]['best'] for f in ALL_FOLDS]
        logits = np.concatenate([p['logits'] for p in selected])
        labels = np.concatenate([p['labels'] for p in selected])
        ids = np.concatenate([p['row_index'] for p in selected])
        baseline = np.concatenate([predictions['BASE'][f]['best']['logits'] for f in ALL_FOLDS])
        correct, old = logits.argmax(1) == labels, baseline.argmax(1) == labels
        pooled, _ = calculate(logits, labels)
        pairs[name] = dict(corrected=int(np.sum(correct & ~old)), harmed=int(np.sum(~correct & old)), pooled=pooled)
        if name == retained:
            with np.load(out/'retained_oof_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['labels'], labels)
                np.testing.assert_array_equal(saved['row_index'], ids)
                np.testing.assert_array_equal(saved['logits'], logits)
    same(analysis['pairs'], pairs, 'Paired corrections and pooled metrics')
    exported = read(out/'best_configuration.json')
    for key, value in dict(name=retained, overrides=specs[retained], graph_weight_decay=0.,
            k=10, seed=0, epochs=1000, parameters=716396,
            model='RelationSheafModel(mode=content)').items():
        same(exported[key], value, f'Exported configuration:{key}')
    require(Path(exported['config']).resolve() == (out/'configs'/f'{retained}.ini').resolve(), 'Exported config path')
    status = read(out/'status.json')
    for key, value in dict(state='complete', retained=retained, new_runs=new_runs, finalists=finalists).items():
        same(status[key], value, f'Final status:{key}')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    artifacts = ('PLAN.json', 'all_settings.json', 'analysis.json', 'selection.json', 'COMPLETE.json',
                 'data_splits_graphs.npz', 'retained_oof_predictions.npz', 'best_configuration.json',
                 'screen_vs_remaining.json', 'summary.json')
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), shared_independent_auditor_sha256=digest(ROOT/'audit_tad_binary_fine_tune.py'),
        device=str(cfg.Device), independent_of_training_metric_and_rank_functions=True,
        new_runs=new_runs, reused_runs=reused_runs, replayed_new_checkpoints=new_replays,
        replayed_reused_checkpoints=reused_replays, baseline_last_prediction_files_verified=10,
        prediction_files_verified=2*(new_runs+reused_runs), finalists=finalists, retained=retained,
        full=full, pairs=pairs, screen_vs_remaining=stage_comparison, checkpoint_details=details,
        data_rows=535, class_counts=[490, 45], parameters=716396, graph_weight_decay=0.,
        graph_candidate_hash=graph_hash,
        selection_verified='First maximum test ACC; four-fold ranking with all primary-ACC ties at second place promoted; one configuration across all ten folds.',
        artifact_sha256={name: digest(out/name) for name in artifacts},
        limitations=['Original BASE has no last.pt; its ten last prediction files and labels were checked without weight replay.',
            'Nonbaseline last prediction files omit row_index; checkpoint replay verifies the fixed test-mask order.',
            'Unique source row IDs do not establish independent clinical patient IDs.',
            'Historical test-selected epochs and adaptive reuse of test folds are retained; this is not independent validation or evidence of a global optimum.'])
    target = out/'independent_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(passed=True, retained=retained, new_runs=new_runs,
        replayed_checkpoints=new_replays+reused_replays, mean_acc=full[retained]['best']['acc'],
        pmci_recall=full[retained]['best']['pmci_recall'], output=str(target))), flush=True)


if __name__ == '__main__':
    main()
