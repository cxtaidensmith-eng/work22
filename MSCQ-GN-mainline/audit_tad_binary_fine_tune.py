"""Independent, inference-only audit of the completed TAD binary C search.

Does not import the tuning runner or its metric/ranking implementations. The
shared data loader and model are used only to reconstruct the frozen experiment.
Run only after COMPLETE.json exists and RUNNING.lock has been removed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import configparser
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from Model.relation_sheaf import RelationSheafModel
from relation_experiment import setup


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT/'Result/tad_binary_fine_tune_20260928'
DEFAULT_REF = ROOT/'Result/c_tadpole_binary_seed0_20260924'
CONFIG = 'Config/c_tadpole_binary_transfer.ini'
KEYS = ('lr', 'weight_decay', 'Drop_rate', 'input_noise_std',
        'aux_loss_weight', 'Loss_rate')
MEAN_KEYS = ('acc', 'macro_f1', 'balanced_acc', 'pmci_auc', 'pmci_ap',
             'pmci_recall', 'smci_recall')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, context, atol=1e-12):
    """Compare independently calculated JSON-shaped results recursively."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{context}: dictionary keys differ')
        for key in expected:
            same(actual[key], expected[key], f'{context}.{key}', atol)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{context}: lengths differ')
        for index, value in enumerate(expected):
            same(actual[index], value, f'{context}[{index}]', atol)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
                f'{context}: nonfinite value')
        require(abs(float(actual)-float(expected)) <= atol,
                f'{context}: {actual!r} != {expected!r}')
    else:
        require(actual == expected, f'{context}: {actual!r} != {expected!r}')


def verify_manifest(manifest):
    for filename, expected in manifest.items():
        require(digest(ROOT/filename) == expected, f'Hash mismatch: {filename}')


def calculate(logits, labels):
    """Confusion-derived classification metrics plus PMCI probability AUC/AP."""
    z = np.asarray(logits)
    y = np.asarray(labels)
    require(z.ndim == 2 and z.shape == (len(y), 2), 'Wrong logit shape')
    require(np.isfinite(z).all(), 'Nonfinite logits')
    require(np.array_equal(np.unique(y), [0, 1]), 'Both classes required')
    pred = z.argmax(axis=1)
    matrix = np.bincount(2*y.astype(np.int64)+pred, minlength=4).reshape(2, 2)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    recall = tp/support
    denominator = matrix.sum(axis=0)+support
    f1 = np.divide(2*tp, denominator, out=np.zeros(2), where=denominator != 0)
    exponent = np.exp(z-z.max(axis=1, keepdims=True))
    probability = exponent[:, 1]/exponent.sum(axis=1)
    metrics = dict(acc=float(tp.sum()/matrix.sum()), macro_f1=float(f1.mean()),
        balanced_acc=float(recall.mean()), pmci_auc=float(roc_auc_score(y, probability)),
        pmci_ap=float(average_precision_score(y, probability)),
        smci_recall=float(recall[0]), pmci_recall=float(recall[1]),
        confusion_matrix=matrix.tolist())
    legacy = dict(acc=metrics['acc'], macro_f1=metrics['macro_f1'],
        f1=float(np.sum(f1*support)/support.sum()),
        auc=float((roc_auc_score(1-y, z[:, 0])+roc_auc_score(y, z[:, 1]))/2),
        per_class_recall=recall.tolist(), confusion_matrix=matrix.tolist())
    return metrics, legacy


def mean_result(rows, folds):
    chosen = [rows[f] for f in folds]
    value = {stage: {key: float(np.mean([r[stage][key] for r in chosen]))
                     for key in MEAN_KEYS} for stage in ('best', 'last')}
    value.update(folds=list(folds),
        late100_acc=float(np.mean([r['late100_acc'] for r in chosen])),
        acc_std=float(np.std([r['best']['acc'] for r in chosen])))
    return value


def ordered(names, rows, folds):
    def key(name):
        summary = mean_result(rows[name], folds)
        return (summary['best']['acc'], summary['best']['macro_f1'],
                summary['best']['pmci_recall'], summary['late100_acc'])
    return sorted(names, key=key, reverse=True)


def check_config_file(path, base_path, spec):
    parser = configparser.ConfigParser()
    parser.read(path, encoding='utf-8')
    baseline = configparser.ConfigParser()
    baseline.read(base_path, encoding='utf-8')
    allowed = {('Modal' if k in ('Drop_rate', 'input_noise_std') else 'Optim',
                k.lower()): v for k, v in spec.items()}
    require(parser.sections() == baseline.sections(), f'{path}: config sections changed')
    for section in baseline.sections():
        expected_keys = set(baseline[section]) | {k for s, k in allowed if s == section}
        require(set(parser[section]) == expected_keys, f'{path}: unexpected config options')
        for key in expected_keys:
            if (section, key) in allowed:
                same(parser.getfloat(section, key), allowed[section, key], f'{path}:{key}')
            else:
                same(parser[section][key], baseline[section][key], f'{path}:{key}')


def check_optimizer(folder, model, spec):
    groups = read(folder/'optimizer_groups.json')
    require(len(groups) == 2, f'{folder}: optimizer group count')
    named = list(model.named_parameters())
    graph = lambda n: n.startswith('propagation.') and not n.startswith('propagation.output_norm.')
    for index, (expected_decay, select) in enumerate(((spec['weight_decay'], False), (0., True))):
        selected = [(n, p) for n, p in named if graph(n) == select]
        group = groups[index]
        same(group['group'], index, f'{folder}: optimizer index')
        same(group['weight_decay'], expected_decay, f'{folder}: weight decay')
        same(group['lr'], spec['lr'], f'{folder}: initial learning rate')
        same(group['parameters'], sum(p.numel() for _, p in selected), f'{folder}: group size')
        same(group['names'], [n for n, _ in selected], f'{folder}: group membership')


def audit_fold(name, fold, folder, model, spec, data, source_rows):
    baseline = name == 'BASE'
    record = read(folder/'complete.json')
    require(record['parameters'] == 716396 and record['fold'] == fold,
            f'{folder}: parameter count or fold mismatch')
    if baseline:
        require(record['mode'] == 'content' and record['seed'] == 0, f'{folder}: baseline identity')
    else:
        require(record['arm'] == name and record['audit_passed'] is True, f'{folder}: run identity')
        same(read(folder/'effective_settings.json'), spec, f'{folder}: effective settings')
    check_optimizer(folder, model, spec)
    with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
        curve = list(csv.DictReader(stream))
    require([int(row['epoch']) for row in curve] == list(range(1, 1001)),
            f'{folder}: expected 1000 consecutive epochs')
    numeric = np.asarray([[float(v) for v in row.values()] for row in curve])
    require(np.isfinite(numeric).all(), f'{folder}: nonfinite learning curve')
    curve_acc = np.asarray([float(row['test_acc']) for row in curve])
    best_epoch = int(curve_acc.argmax())+1
    require(record['best']['epoch'] == best_epoch, f'{folder}: first-maximum selection changed')
    for row in curve:
        tick = int(row['epoch'])-1
        expected_lr = spec['lr'] if tick < 20 else .0001+(spec['lr']-.0001)*(1+math.cos(math.pi*(tick-20)/980))/2
        same(float(row['lr']), expected_lr, f'{folder}: epoch {tick+1} learning rate')
    mask = data['Mask'][fold-1][1]
    expected_y = data['Label'][mask].cpu().numpy()
    expected_id = source_rows[mask.cpu().numpy()]
    require(len(expected_y) in (53, 54), f'{folder}: unexpected fold size')
    np.testing.assert_allclose(curve_acc*len(expected_y), np.rint(curve_acc*len(expected_y)), atol=1e-10, rtol=0)
    result = dict(name=name, fold=fold, best_epoch=best_epoch, parameters=716396,
                  late100_acc=float(curve_acc[-100:].mean()))
    details = {}
    predictions = {}
    native_noise = model._modal_noise_std.detach().clone()
    for stage in ('best', 'last'):
        filename = ('predictions.npz' if baseline else 'best_predictions.npz') if stage == 'best' else 'last_predictions.npz'
        with np.load(folder/filename) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        np.testing.assert_array_equal(saved['labels'], expected_y)
        has_rows = 'row_index' in saved
        if has_rows:
            np.testing.assert_array_equal(saved['row_index'], expected_id)
        else:
            require(not baseline and stage == 'last', f'{folder}: missing explicit row identity')
        actual, legacy = calculate(saved['logits'], saved['labels'])
        same(actual['acc'], float(curve_acc[best_epoch-1 if stage == 'best' else -1]), f'{folder}: {stage} curve ACC')
        if stage == 'best' or baseline:
            for key, value in legacy.items():
                same(value, record[stage][key], f'{folder}: {stage}.{key}')
        else:
            same(record['last']['test_acc'], actual['acc'], f'{folder}: recorded final ACC')
            require(record['last']['epoch'] == 1000, f'{folder}: final epoch')
        replayed = not (baseline and stage == 'last')
        max_error = None
        if replayed:
            checkpoint = torch.load(folder/f'{stage}.pt', map_location=data['Feature'].device, weights_only=False)
            require(checkpoint['fold'] == fold and checkpoint['seed'] == 0, f'{folder}: {stage} checkpoint identity')
            require(checkpoint.get('mode') == 'content' if baseline else checkpoint.get('arm') == name,
                    f'{folder}: {stage} checkpoint model identity')
            if stage == 'best':
                require(checkpoint['best']['epoch'] == best_epoch, f'{folder}: best checkpoint epoch')
                same(checkpoint['best'], record['best'], f'{folder}: best checkpoint metadata')
            else:
                require(checkpoint['epoch'] == 1000, f'{folder}: last checkpoint epoch')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            torch.testing.assert_close(model._modal_noise_std, native_noise, atol=0, rtol=0)
            model.eval()
            with torch.inference_mode():
                replay = model(data['Feature'])[0][mask].cpu().numpy()
            require(np.isfinite(replay).all(), f'{folder}: replay nonfinite')
            np.testing.assert_allclose(replay, saved['logits'], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(replay.argmax(1), saved['logits'].argmax(1))
            max_error = float(np.max(np.abs(replay-saved['logits'])))
            del checkpoint, replay
        result[stage] = actual
        predictions[stage] = dict(logits=saved['logits'], labels=expected_y, row_index=expected_id)
        details[stage] = dict(checkpoint_replayed=replayed, maximum_logit_absolute_error=max_error,
            explicit_row_index=has_rows,
            row_identity='Saved index verified' if has_rows else 'Fixed test-mask order verified by checkpoint replay',
            limitation='Original baseline has no last.pt; only saved last predictions and labels verified' if not replayed else None)
    return result, predictions, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REF)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    out, ref = args.output.resolve(), args.reference.resolve()
    require((out/'COMPLETE.json').is_file(), 'Training has not completed; audit will not use the GPU')
    require(not (out/'RUNNING.lock').exists(), 'Training lock is present; wait for training exit')
    completion = read(out/'COMPLETE.json')
    require(completion['all_audits_passed'] is True, 'Training completion indicates failed audits')
    plan, specs, analysis = (read(out/name) for name in ('PLAN.json', 'all_settings.json', 'analysis.json'))
    selection, refinement = read(out/'selection.json'), read(out/'refinement.json')
    require(plan['seed'] == 0 and plan['epochs'] == 1000, 'Protocol changed')
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
    require(digest(out/'data_splits_graphs.npz') == plan['split_archive_sha256'], 'Split archive hash changed')
    base_plan = read(ref/'PLAN.json')
    same(plan['baseline_protocol'], base_plan['protocol'], 'Baseline protocol')
    cfg, dd, data, graphs, paths = setup(args.cuda, CONFIG, 10)
    for path in paths:
        require(digest(path) == base_plan['protocol']['data_sha256'][Path(path).name], f'Data changed: {path}')
    require(dd['Sample_Num'] == 535 and dd['Class_Names'] == ['SMCI', 'PMCI'], 'Wrong dataset/task')
    same(torch.bincount(data['Label']).cpu().tolist(), [490, 45], 'Class counts')
    source_rows = np.asarray(dd['Index'])
    masks = torch.stack([mask for _, mask in data['Mask']]).cpu().numpy()
    np.testing.assert_array_equal(masks.sum(0), np.ones(535, dtype=int))
    require(len(set(source_rows.tolist())) == 535, 'Source rows are not unique')
    for archive_path in (ref/'splits_and_graphs.npz', out/'data_splits_graphs.npz'):
        with np.load(archive_path) as archive:
            np.testing.assert_array_equal(archive['source_row'], source_rows)
            np.testing.assert_array_equal(archive['labels'], data['Label'].cpu().numpy())
            np.testing.assert_array_equal(archive['test_masks'], masks)
            for index, graph in enumerate(graphs):
                np.testing.assert_array_equal(archive[f'edge_{index}'], graph['edge'].numpy())
                np.testing.assert_array_equal(archive[f'neighbors_{index}'], graph['neighbors'].numpy())
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    require(graph_hash == base_plan['protocol']['candidate_hash'], 'Candidate graph changed')
    for key, value in base_plan['protocol']['original_config'].items():
        same(getattr(cfg, key), value, f'Baseline config:{key}')
    same(specs['BASE'], plan['base'], 'BASE settings')
    for name, change in plan['coarse'].items():
        same(specs[name], {**plan['base'], **change}, f'Coarse settings:{name}')
    rows, prediction_cache, details = {}, {}, {}
    new_runs = 0
    for name, spec in specs.items():
        require(set(spec) == set(KEYS), f'{name}: unexpected hyperparameter change')
        require(all(math.isfinite(float(value)) and value > 0 for value in spec.values()), f'{name}: invalid setting')
        check_config_file(out/'configs'/f'{name}.ini', ROOT/CONFIG, spec)
        conf = copy.copy(cfg)
        for key, value in spec.items():
            setattr(conf, key, value)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        require(sum(p.numel() for p in model.parameters()) == 716396, f'{name}: network size changed')
        parent = ref/'content' if name == 'BASE' else out/'runs'/name
        folders = sorted(parent.glob('fold_*/complete.json'))
        require(bool(folders), f'{name}: no completed folds')
        rows[name], prediction_cache[name], details[name] = {}, {}, {}
        for complete in folders:
            fold = int(complete.parent.name.split('_')[-1])
            require(1 <= fold <= 10, f'{name}: invalid fold')
            result, predictions, detail = audit_fold(name, fold, complete.parent, model, spec, data, source_rows)
            rows[name][fold], prediction_cache[name][fold], details[name][fold] = result, predictions, detail
            new_runs += int(name != 'BASE')
            print(json.dumps(dict(audit=name, fold=fold, passed=True), ensure_ascii=False), flush=True)
        del model
        if cfg.Device.type == 'cuda':
            torch.cuda.empty_cache()
    discovered = list((out/'runs').glob('*/fold_*/complete.json'))
    require(new_runs == len(discovered) <= plan['maximum_new_runs'], 'Unrecognized/excess completed runs')
    same(new_runs, completion['new_runs'], 'Completion run count')
    same(new_runs, analysis['new_runs'], 'Analysis run count')
    same(2*new_runs, analysis['verified_new_checkpoints'], 'Checkpoint count')
    screen = plan['screen_folds']
    coarse_names = ['BASE']+list(plan['coarse'])
    leader = ordered(coarse_names, rows, screen)[0]
    require(refinement['leader'] == leader, 'Refinement leader does not match ranking')
    expected_specs = {n: specs[n] for n in coarse_names}
    expected_aliases = {}
    for name, key, scale in [('R_L075', 'lr', .75), ('R_L125', 'lr', 1.25),
                             ('R_A050', 'aux_loss_weight', .5), ('R_O010', 'Loss_rate', .1)]:
        candidate = {**specs[leader], key: round(specs[leader][key]*scale, 12)}
        duplicate = next((n for n, setting in expected_specs.items() if setting == candidate), None)
        if duplicate is not None:
            expected_aliases[name] = duplicate
        else:
            expected_specs[name] = candidate
    same(specs, expected_specs, 'Search settings and refinement')
    same(list(specs), list(expected_specs), 'Stable candidate insertion order')
    same(refinement['aliases'], expected_aliases, 'Deduplicated refinements')
    screen_order = ordered(list(specs), rows, screen)
    same(selection['screen_order'], screen_order, 'Screen ranking')
    finalists = [name for name in screen_order if name != 'BASE'][:2]
    same(selection['finalists'], finalists, 'Selected finalists')
    for name in specs:
        expected_folds = list(range(1, 11)) if name == 'BASE' or name in finalists else sorted(screen)
        same(sorted(rows[name]), expected_folds, f'{name}: completed fold set')
    names = ['BASE']+finalists
    retained = ordered(names, rows, list(range(1, 11)))[0]
    same(analysis['retained'], retained, 'Retained configuration')
    same(completion['retained'], retained, 'Completion retained configuration')
    same(analysis['settings'], specs[retained], 'Retained settings')
    full = {name: mean_result(rows[name], list(range(1, 11))) for name in names}
    same(full, analysis['full'], 'Complete tenfold metrics')
    records = [rows[name][fold] for name in names for fold in range(1, 11)]
    same(records, analysis['records'], 'Reported fold records')
    summary = read(out/'summary.json')['candidates']
    for name in specs:
        same(mean_result(rows[name], sorted(rows[name])), summary[name], f'{name}: progress summary')
    pairs = {}
    for name in names:
        for stage in ('best', 'last'):
            ids = np.concatenate([prediction_cache[name][f][stage]['row_index'] for f in range(1, 11)])
            require(len(ids) == len(np.unique(ids)) == 535, f'{name}/{stage}: not exactly one prediction per source row')
            np.testing.assert_array_equal(np.sort(ids), np.sort(source_rows))
        selected = [prediction_cache[name][f]['best'] for f in range(1, 11)]
        logits = np.concatenate([p['logits'] for p in selected])
        labels = np.concatenate([p['labels'] for p in selected])
        ids = np.concatenate([p['row_index'] for p in selected])
        baseline_logits = np.concatenate([prediction_cache['BASE'][f]['best']['logits'] for f in range(1, 11)])
        correct, old = logits.argmax(1) == labels, baseline_logits.argmax(1) == labels
        pooled, _ = calculate(logits, labels)
        pairs[name] = dict(corrected=int(np.sum(correct & ~old)), harmed=int(np.sum(~correct & old)), pooled=pooled)
        if name == retained:
            with np.load(out/'retained_oof_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['labels'], labels)
                np.testing.assert_array_equal(saved['row_index'], ids)
                np.testing.assert_array_equal(saved['logits'], logits)
    same(pairs, analysis['pairs'], 'Paired corrections and pooled metrics')
    exported = read(out/'best_configuration.json')
    for key, value in dict(name=retained, overrides=specs[retained], graph_weight_decay=0.,
                           k=10, seed=0, epochs=1000, parameters=716396,
                           model='RelationSheafModel(mode=content)').items():
        same(exported[key], value, f'Exported configuration:{key}')
    require(Path(exported['config']).resolve() == (out/'configs'/f'{retained}.ini').resolve(), 'Exported config path')
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), device=str(cfg.Device), independent_of_training_metric_and_rank_functions=True,
        new_runs=new_runs, replayed_new_checkpoints=2*new_runs, replayed_baseline_best_checkpoints=10,
        baseline_last_prediction_files_verified=10, prediction_files_verified=2*new_runs+20,
        retained=retained, full=full, pairs=pairs, checkpoint_details=details,
        data_rows=535, class_counts=[490, 45], parameters=716396, graph_weight_decay=0.,
        selection_verified='First maximum test ACC; one configuration for all folds; adaptive screen/refinement/final ranking reproduced',
        artifact_sha256={name: digest(out/name) for name in ('PLAN.json', 'all_settings.json', 'analysis.json',
            'selection.json', 'refinement.json', 'COMPLETE.json', 'data_splits_graphs.npz', 'retained_oof_predictions.npz')},
        limitations=['Baseline last checkpoint was not saved; its last prediction labels and metrics were verified without weight replay.',
            'New last prediction files omit row_index; fixed mask order is established by checkpoint replay and label checks.',
            'Source row identity does not establish independent clinical patient IDs.',
            'Retained historical test-selected and adaptive test-fold tuning protocol; not independent validation or a global optimum.'])
    target = out/'independent_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(passed=True, retained=retained, new_runs=new_runs,
        replayed_new_checkpoints=2*new_runs, mean_acc=full[retained]['best']['acc'],
        pmci_recall=full[retained]['best']['pmci_recall'], output=str(target)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
