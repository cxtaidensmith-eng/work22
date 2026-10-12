"""Independent audit of the six-fold ABIDE-5 N005 completion extension.

Imports only the independent main auditor's helpers, never the tuning drivers
or their metric/ranking implementations. CUDA replay is gated by completion,
lock removal, and the passed original independent audit.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import audit_abide5_fine_tune as independent
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import setup


ROOT = Path(__file__).resolve().parent
MAIN = ROOT/'Result/abide5_fine_tune_20260928'
REFERENCE = ROOT/'Result/c_abide5_seed0_20260924'
DEFAULT_OUT = ROOT/'Result/abide5_tie_completion_20260928'
SCREEN = [1, 3, 6, 8]
MISSING = [2, 4, 5, 7, 9, 10]
FOLDS = list(range(1, 11))
NAMES = ['BASE', 'N005', 'N015', 'A050']
SPECS = {
    'BASE': dict(independent.BASE),
    'N005': {**independent.BASE, 'input_noise_std': .05},
    'N015': {**independent.BASE, 'input_noise_std': .15},
    'A050': {**independent.BASE, 'aux_loss_weight': .5},
}
read, same, require = independent.read, independent.same, independent.require
digest, verify_manifest = independent.digest, independent.verify_manifest


def folder_for(out, name, fold):
    if name == 'BASE':
        return REFERENCE/'content'/f'fold_{fold:02d}'
    if name == 'N005' and fold in MISSING:
        return out/'runs'/name/f'fold_{fold:02d}'
    require(name in NAMES and fold in FOLDS, 'Unexpected candidate/fold')
    return MAIN/'runs'/name/f'fold_{fold:02d}'


def check_frozen_inputs(out, plan, original_audit):
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    for filename, expected in plan['code_sha256'].items():
        same(digest(out/'source_snapshot'/filename), expected, f'Extension source snapshot: {filename}')
    require('abide5_tie_completion.py' in plan['code_sha256'], 'Extension driver not frozen')
    for filename, expected in original_audit['artifact_sha256'].items():
        same(digest(MAIN/filename), expected, f'Original audited artifact: {filename}')
    same(digest(ROOT/'audit_abide5_fine_tune.py'), original_audit['auditor_sha256'], 'Original independent auditor')
    numeric_policy = original_audit['first_three_loss_numeric_policy']
    same(digest(numeric_policy['evidence_path']), numeric_policy['evidence_sha256'], 'Original float32 diagnostic evidence')
    same(digest(numeric_policy['failed_attempt_manifest']), numeric_policy['failed_attempt_manifest_sha256'],
         'Original preserved failure manifest')
    same(numeric_policy['atol'], independent.FIRST_THREE_LOSS_ATOL, 'Shared first-three loss atol')
    same(numeric_policy['rtol'], independent.FIRST_THREE_LOSS_RTOL, 'Shared first-three loss rtol')
    same(digest(out/'data_splits_graphs.npz'), plan['split_archive_sha256'], 'Extension archive')
    for filename in ('PLAN.json', 'COMPLETE.json', 'independent_audit.json', 'analysis.json',
                     'all_settings.json', 'selection.json', 'preflight.json', 'config_sha256.json',
                     'screen_vs_remaining.json', 'best_configuration.json', 'data_splits_graphs.npz'):
        require(str((MAIN/filename).relative_to(ROOT)) in plan['reference_sha256'],
                f'Unpinned original artifact: {filename}')
    required_reused = []
    for name in NAMES:
        for fold in SCREEN if name == 'N005' else FOLDS:
            folder = folder_for(out, name, fold)
            filenames = ['best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json',
                         'predictions.npz' if name == 'BASE' else 'best_predictions.npz', 'last_predictions.npz']
            if name != 'BASE':
                filenames += ['last.pt', 'effective_settings.json', 'fine_audit.json']
            for filename in filenames:
                require(str((folder/filename).relative_to(ROOT)) in plan['reference_sha256'],
                        f'Unpinned reused fold artifact: {folder/filename}')
            required_reused.append(dict(name=name, fold=fold, folder=str(folder.relative_to(ROOT))))
    same(plan['reused_folds'], required_reused, 'Exact reused fold manifest')
    return required_reused


def verify_screen_eligibility(out, plan, main_specs, masks, labels, source_rows):
    counts = {}
    for name in main_specs:
        correct = samples = 0
        for fold in SCREEN:
            folder = (REFERENCE/'content' if name == 'BASE' else MAIN/'runs'/name)/f'fold_{fold:02d}'
            filename = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
            path = folder/filename
            require(str(path.relative_to(ROOT)) in plan['reference_sha256'],
                    f'Eligibility prediction not frozen: {path}')
            require(str((folder/'complete.json').relative_to(ROOT)) in plan['reference_sha256'],
                    f'Eligibility completion record not frozen: {folder}')
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['labels'], labels[masks[fold-1]])
                np.testing.assert_array_equal(saved['row_index'], source_rows[masks[fold-1]])
                require(np.isfinite(saved['logits']).all(), 'Nonfinite eligibility logits')
                hit = int((saved['logits'].argmax(1) == saved['labels']).sum())
                same(hit/len(saved['labels']), read(folder/'complete.json')['best']['acc'],
                     f'{name}/{fold}: eligibility accuracy')
                correct += hit
                samples += len(saved['labels'])
        same(samples, 346, f'{name}: eligibility sample count')
        counts[name] = correct
    selection = read(MAIN/'selection.json')
    same(selection['finalists'], ['N015', 'A050'], 'Original finalists unchanged')
    same(plan['original_selection'], selection, 'Original selection preserved')
    eligible = [name for name in main_specs if name != 'BASE' and name not in selection['finalists']
                and counts[name] >= max(counts.values())]
    same(max(counts.values()), 315, 'Pooled fourfold maximum correct')
    same(eligible, ['N005'], 'Sole omitted pooled-correct tie')
    same(plan['eligible'], eligible, 'Extension eligible candidates')
    same(plan['screen_correct_counts'], counts, 'All original screening correct counts')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    require((out/'COMPLETE.json').is_file(), 'Extension training is incomplete; no CUDA audit')
    require(not (out/'RUNNING.lock').exists(), 'Extension training lock remains')
    require((MAIN/'COMPLETE.json').is_file() and not (MAIN/'RUNNING.lock').exists(), 'Original search is not complete')
    original_audit = read(MAIN/'independent_audit.json')
    require(original_audit['passed'] is True, 'Original independent audit has not passed')
    same(original_audit['new_runs'], 72, 'Original new training count')
    completion, plan, specs, analysis = (read(out/filename) for filename in (
        'COMPLETE.json', 'PLAN.json', 'all_settings.json', 'analysis.json'))
    require(completion['all_audits_passed'] is True, 'Extension completion indicates failed audits')
    expected_counts = dict(new_runs=6, verified_new_checkpoints=12, reused_n005_runs=4,
        completed_n005_tenfold_runs=10, original_search_new_runs=72, total_search_plus_extension_new_runs=78)
    for key, value in expected_counts.items():
        same(completion[key], value, f'Completion count: {key}')
        same(analysis[key], value, f'Analysis count: {key}')
    for key, value in dict(specs=SPECS, seed=0, epochs=1000, samples=864, parameters=768348,
            counts=[397, 467], screen_folds=SCREEN, reused_n005_folds=SCREEN, new_folds=MISSING,
            new_runs=6, reused_n005_runs=4, original_search_new_runs=72,
            total_search_plus_extension_new_runs=78, compare_order=NAMES).items():
        same(plan[key], value, f'Prespecified extension: {key}')
    same(specs, SPECS, 'Exact extension configurations')
    same(list(specs), NAMES, 'Original candidate insertion order')
    main_plan, main_specs = read(MAIN/'PLAN.json'), read(MAIN/'all_settings.json')
    for name in NAMES:
        same(specs[name], main_specs[name], f'{name}: reused parameters')
    same(plan['baseline_protocol'], main_plan['baseline_protocol'], 'Baseline protocol retained')
    same(plan['original_plan_sha256'], digest(MAIN/'PLAN.json'), 'Original frozen plan')
    require((ROOT/plan['source_experiment']).resolve() == MAIN, 'Original search path')
    require((ROOT/plan['base_reference']).resolve() == REFERENCE, 'Historical baseline path')
    reused = check_frozen_inputs(out, plan, original_audit)
    same(len(reused), 34, 'Reused fold count')
    hashes = read(out/'config_sha256.json')
    same(set(hashes), set(NAMES), 'Extension config hash coverage')
    for name in NAMES:
        same(digest(out/'configs'/f'{name}.ini'), hashes[name], f'{name}: config hash')
        independent.check_config(out/'configs'/f'{name}.ini', specs[name])
    with np.load(MAIN/'data_splits_graphs.npz') as archive:
        source_rows, labels, masks = (archive[key].copy() for key in ('source_row', 'labels', 'test_masks'))
    counts = verify_screen_eligibility(out, plan, main_specs, masks, labels, source_rows)

    # All training/lock/provenance/eligibility checks precede CUDA initialization.
    cfg, dd, data, graphs, paths = setup(args.cuda, independent.CONFIG, 10)
    require(cfg.Device.type == 'cuda', 'CUDA checkpoint replay required')
    same(dd['Sample_Num'], 864, 'ABIDE-5 row count')
    same(dd['Class_Names'], ['ADS', 'CN'], 'Class order')
    same(dd['Modal_Name'], ['PHENO', 'ANAT', 'FUNC', 'MRI', 'fMRI'], 'Modality names')
    same([len(v) for v in dd['Modal_Index']], [48, 6, 10, 150, 256], 'Modality dimensions')
    for path in paths:
        same(digest(path), plan['baseline_protocol']['data_sha256'][Path(path).name], f'Dataset: {path}')
    for key, value in plan['baseline_protocol']['original_config'].items():
        same(getattr(cfg, key), value, f'Baseline config: {key}')
    np.testing.assert_array_equal(source_rows, dd['Index'])
    np.testing.assert_array_equal(labels, data['Label'].cpu().numpy())
    np.testing.assert_array_equal(masks, torch.stack([mask for _, mask in data['Mask']]).cpu().numpy())
    np.testing.assert_array_equal(masks.sum(0), np.ones(864, dtype=int))
    same(len(np.unique(source_rows)), 864, 'Unique source rows')
    same(np.bincount(labels).tolist(), [397, 467], 'Class counts')
    for path in (MAIN/'data_splits_graphs.npz', out/'data_splits_graphs.npz', REFERENCE/'splits_and_graphs.npz'):
        independent.check_archive(path, dd, data, graphs)
    graph_hash = hashlib.sha256(b''.join(graph['edge'].numpy().tobytes() for graph in graphs)).hexdigest()
    same(graph_hash, plan['baseline_protocol']['candidate_hash'], 'Candidate graph hash')
    preflight = read(out/'preflight.json')
    same(preflight['passed'], True, 'Extension preflight')
    same(set(preflight['checks']), set(NAMES), 'Preflight setting coverage')
    same(preflight['verified_reference_checkpoints'], 10, 'Baseline preflight checkpoint count')
    same([item['fold'] for item in preflight['baseline_checkpoints']], FOLDS, 'Baseline preflight folds')
    for item in preflight['baseline_checkpoints']:
        same(item['passed'], True, 'Baseline preflight checkpoint passed')
        same(item['best_epoch'], read(folder_for(out, 'BASE', item['fold'])/'complete.json')['best']['epoch'],
             'Baseline preflight best epoch')
        same(item['checkpoint_sha256'], digest(folder_for(out, 'BASE', item['fold'])/'best.pt'),
             'Baseline preflight checkpoint hash')
    main_preflight = read(MAIN/'preflight.json')['checks']
    for name in NAMES:
        for key in ('spec', 'parameters', 'init_parameter_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256',
                    'optimizer_groups', 'effective_noise_sd'):
            same(preflight['checks'][name][key], main_preflight[name][key], f'{name}: original preflight {key}')
    rows, predictions, details, replay3 = {}, {}, {}, {}
    baseline_initial = baseline_logits = None
    new_replays = reused_n005_replays = other_replays = 0
    for name in NAMES:
        conf = independent.configure(cfg, specs[name])
        replay3[name], initial, initial_logits = independent.first_three(name, conf, dd, data, graphs,
            specs[name], folder_for(out, name, 1), preflight['checks'][name])
        if name == 'BASE':
            baseline_initial, baseline_logits = initial, initial_logits
        else:
            same(set(initial), set(baseline_initial), f'{name}: unchanged parameter names')
            for key in initial:
                torch.testing.assert_close(initial[key], baseline_initial[key], atol=0, rtol=0)
            torch.testing.assert_close(initial_logits, baseline_logits, atol=2e-7, rtol=1e-6)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        same(sum(param.numel() for param in model.parameters()), 768348, f'{name}: unchanged network size')
        rows[name], predictions[name], details[name] = {}, {}, {}
        for fold in FOLDS:
            folder = folder_for(out, name, fold)
            row, pred, detail = independent.audit_fold(name, fold, folder, model, specs[name], data, source_rows)
            rows[name][fold], predictions[name][fold], details[name][fold] = row, pred, detail
            count = sum(int(value['checkpoint_replayed']) for value in detail.values())
            if name == 'N005' and fold in MISSING:
                new_replays += count
            elif name == 'N005':
                reused_n005_replays += count
            else:
                other_replays += count
            print(json.dumps(dict(audit=name, fold=fold, new_training=name == 'N005' and fold in MISSING,
                                  passed=True)), flush=True)
        del model, initial, initial_logits
        torch.cuda.empty_cache()
    same((new_replays, reused_n005_replays, other_replays), (12, 8, 50), 'Actual checkpoint replay counts')
    discovered = {(path.parent.parent.name, int(path.parent.name.split('_')[-1]))
                  for path in (out/'runs').glob('*/fold_*/complete.json')}
    same(discovered, {('N005', fold) for fold in MISSING}, 'Exactly six new N005 training runs')
    same({path.name for path in (out/'runs').iterdir() if path.is_dir()}, {'N005'}, 'No extra run candidates')
    same({int(path.name.split('_')[-1]) for path in (out/'runs/N005').iterdir() if path.is_dir()},
         set(MISSING), 'No extra or incomplete N005 fold folders')
    reuse_preflight = read(out/'reuse_preflight.json')
    for key, value in dict(passed=True, verified_reused_n005_checkpoints=8,
                          initialization_rng_optimizer_noise_match_original=True).items():
        same(reuse_preflight[key], value, f'Reuse preflight: {key}')
    expected_checkpoints = [dict(fold=fold, stage=stage, passed=True,
        checkpoint_sha256=digest(folder_for(out, 'N005', fold)/f'{stage}.pt')) for fold in SCREEN for stage in ('best', 'last')]
    same(reuse_preflight['reused_n005_checkpoints'], expected_checkpoints, 'All reused N005 checkpoint hashes')
    trace = reuse_preflight['n005_first_three_vs_original_training']
    reuse_loss_comparisons = []
    same(len(trace), 3, 'Reuse first-three trace length')
    curve = independent.curve_for(folder_for(out, 'N005', 1))
    for index, item in enumerate(trace):
        current, old = preflight['checks']['N005']['first3'][index], main_preflight['N005']['first3'][index]
        same(item['epoch'], index+1, 'Reuse trace epoch')
        same(item['loss'], current['loss'], 'Reuse preflight observed loss')
        same(item['training_loss'], float(curve[index]['training_loss']), 'Reuse original training loss')
        same(item['test_acc'], current['test_acc'], 'Reuse preflight test ACC')
        same(item['loss_absolute_difference'], abs(item['loss']-item['training_loss']), 'Reuse training loss error')
        same(item['old_preflight_loss_absolute_difference'], abs(item['loss']-old['loss']), 'Reuse old preflight loss error')
        reuse_loss_comparisons.append(dict(epoch=index+1,
            original_training=independent.compare_first_three_loss(item['loss'], item['training_loss'],
                'Reuse recorded loss consistency'),
            original_preflight=independent.compare_first_three_loss(item['loss'], old['loss'],
                'Reuse versus original preflight loss')))
    full = {name: independent.aggregate(rows[name], FOLDS) for name in NAMES}
    full_order = independent.ordered(NAMES, rows, FOLDS)
    retained = full_order[0]
    stages = {name: dict(screen=independent.aggregate(rows[name], SCREEN),
        remaining=independent.aggregate(rows[name], MISSING),
        screen_delta_acc=independent.aggregate(rows[name], SCREEN)['best']['acc']-independent.aggregate(rows['BASE'], SCREEN)['best']['acc'],
        remaining_delta_acc=independent.aggregate(rows[name], MISSING)['best']['acc']-independent.aggregate(rows['BASE'], MISSING)['best']['acc'])
        for name in NAMES}
    pairs = {}
    for name in NAMES:
        for stage in ('best', 'last'):
            ids = np.concatenate([predictions[name][fold][stage]['row_index'] for fold in FOLDS])
            same(len(ids), 864, f'{name}/{stage}: OOF size')
            same(len(np.unique(ids)), 864, f'{name}/{stage}: OOF unique rows')
            np.testing.assert_array_equal(np.sort(ids), np.sort(source_rows))
        oof = {key: np.concatenate([predictions[name][fold]['best'][key] for fold in FOLDS])
               for key in ('labels', 'logits', 'row_index', 'fold')}
        base = np.concatenate([predictions['BASE'][fold]['best']['logits'] for fold in FOLDS])
        correct, old = oof['logits'].argmax(1) == oof['labels'], base.argmax(1) == oof['labels']
        pooled, _ = independent.calculate(oof['logits'], oof['labels'])
        deltas = [int(np.trace(rows[name][fold]['best']['confusion_matrix']))
                  -int(np.trace(rows['BASE'][fold]['best']['confusion_matrix'])) for fold in FOLDS]
        pairs[name] = dict(corrected=int(np.sum(correct & ~old)), harmed=int(np.sum(~correct & old)),
            wins=sum(value > 0 for value in deltas), ties=sum(value == 0 for value in deltas),
            losses=sum(value < 0 for value in deltas), pooled=pooled)
        independent.check_oof(out/'oof'/f'{name}.npz', oof)
        if name == retained:
            independent.check_oof(out/'retained_oof_predictions.npz', oof)
    expected_analysis = dict(retained=retained, full_order=full_order, full=full, pairs=pairs,
        records=[rows[name][fold] for name in NAMES for fold in FOLDS], settings=specs[retained],
        screen_vs_remaining=stages, **expected_counts)
    same(analysis, expected_analysis, 'Independent extension analysis')
    same(completion['retained'], retained, 'Completion retained configuration')
    same(read(out/'screen_vs_remaining.json'), stages, 'Extension screen/remaining results')
    for name in ('BASE', 'N015', 'A050'):
        same(full[name], original_audit['full'][name], f'{name}: original audited complete metrics')
        same(pairs[name], original_audit['pairs'][name], f'{name}: original audited paired metrics')
    summary = read(out/'summary.json')['candidates']
    same(summary, full, 'Extension full summary')
    exported = read(out/'best_configuration.json')
    for key, value in dict(name=retained, driver='abide5_tie_completion.py', base_config=independent.CONFIG,
            overrides=specs[retained], model='RelationSheafModel(mode=content)', graph_weight_decay=specs[retained]['graph_weight_decay'],
            k=10, seed=0, epochs=1000, parameters=768348).items():
        same(exported[key], value, f'Extension exported config: {key}')
    require(Path(exported['config']).resolve() == out/'configs'/f'{retained}.ini', 'Extension exported INI path')
    same(exported['effective_noise_sd'], [3*specs[retained]['input_noise_std']]*5, 'Exported effective noise')
    status = read(out/'status.json')
    for key, value in dict(state='complete', retained=retained, new_runs=6).items():
        same(status[key], value, f'Extension status: {key}')
    check_frozen_inputs(out, plan, original_audit)
    for name in NAMES:
        same(digest(out/'configs'/f'{name}.ini'), hashes[name], f'{name}: final config hash')
    artifacts = ('PLAN.json', 'COMPLETE.json', 'all_settings.json', 'analysis.json', 'preflight.json',
        'reuse_preflight.json', 'config_sha256.json', 'data_splits_graphs.npz', 'screen_vs_remaining.json',
        'best_configuration.json', 'retained_oof_predictions.npz', 'summary.json')
    artifact_hashes = {filename: digest(out/filename) for filename in artifacts}
    artifact_hashes.update({f'oof/{name}.npz': digest(out/'oof'/f'{name}.npz') for name in NAMES})
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), shared_independent_auditor_sha256=digest(ROOT/'audit_abide5_fine_tune.py'),
        original_independent_audit_sha256=digest(MAIN/'independent_audit.json'),
        independent_of_tuning_metrics_and_ranking=True, device=str(cfg.Device), **expected_counts,
        replayed_new_checkpoints=new_replays, replayed_reused_n005_checkpoints=reused_n005_replays,
        replayed_other_reference_checkpoints=other_replays, total_checkpoint_replays=70,
        baseline_last_prediction_files_verified=10, prediction_files_verified=80,
        first_three_training_replays=replay3,
        n005_reuse_preflight_loss_comparisons=reuse_loss_comparisons,
        first_three_loss_numeric_policy=original_audit['first_three_loss_numeric_policy'],
        retained=retained, full_order=full_order, full=full, pairs=pairs, screen_vs_remaining=stages,
        screen_correct_counts=counts, checkpoint_details=details, samples=864, class_counts=[397, 467],
        parameters=768348, graph_candidate_hash=graph_hash, artifact_sha256=artifact_hashes,
        limitations=['The six N005 runs are new; its four screening folds and the other complete configurations are reused.',
            'The original 72-run plan and its selection are unchanged. This separately disclosed extension completes a pooled-correct tie.',
            'Historical BASE has no last.pt; its final saved predictions are verified without final-weight replay.',
            'Test-selected epochs and reused test folds remain exploratory, not independent generalization or a global optimum.',
            'Unique CSV row indices are not clinical patient IDs. ABIDE-5 overlaps ABIDE and is not an independent external cohort.'])
    target = out/'independent_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(passed=True, retained=retained, new_runs=6, total_search_runs=78,
        replayed_checkpoints=70, mean_acc=full[retained]['best']['acc'], output=str(target))), flush=True)


if __name__ == '__main__':
    main()
