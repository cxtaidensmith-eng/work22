"""Independent audit of the frozen C binary loss-parameter search.

No training/search-driver functions are imported. Saved predictions and frozen
best/final weights are replayed, and each new candidate's first three training
steps are independently reconstructed to verify the effective loss parameters.
Never invoke while this search or the preceding joint search is running.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_tad_binary_fine_tune import (
    ROOT, CONFIG, KEYS, audit_fold, calculate, check_config_file, digest,
    mean_result, ordered, read, require, same, verify_manifest,
)
from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel
from Utils import SET_Random
from relation_experiment import setup


DEFAULT_OUT = ROOT/'Result/tad_binary_loss_tune_20260928'
DEFAULT_JOINT = ROOT/'Result/tad_binary_joint_tune_20260928'
DEFAULT_PREVIOUS = ROOT/'Result/tad_binary_fine_tune_20260928'
DEFAULT_REFERENCE = ROOT/'Result/c_tadpole_binary_seed0_20260924'
SCREEN = [1, 3, 6, 8]
REMAINING = [2, 4, 5, 7, 9, 10]
ALL_FOLDS = list(range(1, 11))
LOSS_KEYS = (*KEYS, 'label_smoothing')
CHANGES = {
    'LS000': dict(label_smoothing=0.),
    'LS002': dict(label_smoothing=.02),
    'A050': dict(aux_loss_weight=.50),
    'A075': dict(aux_loss_weight=.75),
}
REPRODUCTION = dict(
    reproduction_entrypoint='tad_binary_loss_tune.py',
    config_alone_reproduces_loss=False,
    reproduction_note='Legacy Config_ and training entry points do not read label_smoothing from the exported INI. Reproduce with this driver and its explicit loss_factory, or explicitly pass the selected label_smoothing to MSCQGNLoss for the main and both auxiliary CE losses; the INI alone is insufficient.',
)


def explicit_loss(dd, device, spec):
    criterion = MSCQGNLoss(dd, device, rate=spec['Loss_rate'],
        aux_loss_weight=spec['aux_loss_weight'], label_smoothing=spec['label_smoothing'])
    same(criterion.CE_loss.label_smoothing, spec['label_smoothing'], 'Main label smoothing')
    same(criterion.aux_loss_weight, spec['aux_loss_weight'], 'Auxiliary loss weight')
    same(criterion.rate, spec['Loss_rate'], 'Orthogonality loss weight')
    same(criterion.aux_supervision, 'ovr', 'Auxiliary supervision')
    require(len(criterion.aux_loss_dict) == 2, 'Expected two auxiliary losses')
    expected_weights = [45/535, 490/535]
    np.testing.assert_allclose(criterion.CE_loss.weight.cpu().numpy(), expected_weights, atol=1e-7, rtol=0)
    for index in range(2):
        loss = criterion.aux_loss_dict[f'aux_loss_{index}']
        same(loss.label_smoothing, spec['label_smoothing'], f'Auxiliary {index} label smoothing')
        weights = [1-expected_weights[index], expected_weights[index]]
        np.testing.assert_allclose(loss.weight.cpu().numpy(), weights, atol=1e-7, rtol=0)
    return criterion


def manual_weighted_ce(logits, labels, weights, smoothing):
    # PyTorch weighted integer-target mean CE divides both components by the
    # sum of the hard-target class weights, including when targets are smoothed.
    logp = F.log_softmax(logits, dim=-1)
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)*weights[labels]
    smooth = -(logp*weights[None]).sum(dim=1)
    denominator = weights[labels].sum()
    return ((1-smoothing)*nll.sum()+smoothing* smooth.sum()/logits.shape[1])/denominator


def verify_total_loss(criterion, z, y, fit, emb, aux, spec):
    actual = criterion(z, y, fit, emb, aux)
    main = manual_weighted_ce(z[fit], y[fit], criterion.CE_loss.weight, spec['label_smoothing'])
    auxiliary = 0
    for index, head in enumerate(aux):
        target = (y == index).long()
        weight = criterion.aux_loss_dict[f'aux_loss_{index}'].weight
        auxiliary = auxiliary+manual_weighted_ce(head[fit], target[fit], weight, spec['label_smoothing'])
    orthogonality = 0
    for i in range(len(emb)):
        for j in range(i+1, len(emb)):
            left = emb[i]/emb[i].norm(dim=1, keepdim=True).clamp_min(1e-8)
            right = emb[j]/emb[j].norm(dim=1, keepdim=True).clamp_min(1e-8)
            orthogonality = orthogonality+torch.norm(left@right.T, p='fro')**2
    expected = main+spec['aux_loss_weight']*auxiliary+spec['Loss_rate']*orthogonality
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    return actual


def replay_initial_steps(name, folder, spec, cfg, dd, data, graphs):
    """Three bounded verification updates, never used to select a configuration."""
    conf = copy.copy(cfg)
    for key, value in spec.items():
        setattr(conf, key, value)
    SET_Random(0)
    model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device)
    graph, patient = [], []
    for parameter_name, parameter in model.named_parameters():
        is_graph = parameter_name.startswith('propagation.') and not parameter_name.startswith('propagation.output_norm.')
        (graph if is_graph else patient).append(parameter)
    optimizer = torch.optim.Adam([
        dict(params=patient, weight_decay=spec['weight_decay']),
        dict(params=graph, weight_decay=0.),
    ], lr=spec['lr'], weight_decay=spec['weight_decay'])
    criterion = explicit_loss(dd, cfg.Device, spec)
    with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
        expected = list(csv.DictReader(stream))[:3]
    fit, test = data['Mask'][0]
    y = data['Label']
    reports = []
    for index, row in enumerate(expected):
        same(int(row['epoch']), index+1, f'{name}: replay epoch')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z, emb, aux = model(data['Feature'])
        loss = verify_total_loss(criterion, z, y, fit, emb, aux, spec)
        observed = float(loss.detach())
        expected_loss = float(row.get('training_loss', row.get('train_loss')))
        require(abs(observed-expected_loss) < 2e-5, f'{name}: first-step loss mismatch at {index+1}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip, error_if_nonfinite=True)
        if 'gradient_norm' in row:
            np.testing.assert_allclose(float(norm), float(row['gradient_norm']), atol=2e-5, rtol=2e-5)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(data['Feature'])[0]
            acc = float((logits[test].argmax(1) == y[test]).double().mean())
            train_ce = float(criterion.CE_loss(logits[fit], y[fit]))
            test_ce = float(criterion.CE_loss(logits[test], y[test]))
        same(acc, float(row['test_acc']), f'{name}: first-step test ACC {index+1}')
        for key, actual in (('train_ce', train_ce), ('test_ce', test_ce)):
            if key in row:
                np.testing.assert_allclose(actual, float(row[key]), atol=2e-5, rtol=2e-5)
        reports.append(dict(epoch=index+1, loss_absolute_error=abs(observed-expected_loss), test_acc=acc))
    del model, optimizer, criterion, logits, loss, z, emb, aux
    if cfg.Device.type == 'cuda':
        torch.cuda.empty_cache()
    return reports


def check_checkpoint_loss(name, fold, folder, model, spec, dd, data):
    """Cross-check recorded smoothed main losses at both persisted stages."""
    with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
        curve = list(csv.DictReader(stream))
    fit, test = data['Mask'][fold-1]
    y = data['Label']
    weight = torch.tensor([45/535, 490/535], dtype=torch.float64, device=y.device)
    results = {}
    for stage in ('best', 'last'):
        path = folder/f'{stage}.pt'
        if not path.exists():
            require(stage == 'last' and name == 'BASE', f'Missing checkpoint:{path}')
            continue
        checkpoint = torch.load(path, map_location=y.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        epoch = checkpoint['best']['epoch'] if stage == 'best' else 1000
        with torch.inference_mode():
            logits = model(data['Feature'])[0].double()
            ces = {label: float(manual_weighted_ce(logits[mask], y[mask], weight,
                    spec['label_smoothing'])) for label, mask in (('train_ce', fit), ('test_ce', test))}
        row = curve[epoch-1]
        for key, value in ces.items():
            # Original BASE did not record these fields; new/reused search runs do.
            if key in row:
                np.testing.assert_allclose(value, float(row[key]), atol=2e-5, rtol=2e-5)
        results[stage] = dict(epoch=epoch, checked_ce_fields=[key for key in ces if key in row], **ces)
        del checkpoint, logits
    return results


def verify_effective_loss(path, spec, dd, device):
    value = read(path)
    criterion = explicit_loss(dd, device, spec)
    expected = dict(label_smoothing=spec['label_smoothing'],
        auxiliary_label_smoothing={f'aux_loss_{i}': spec['label_smoothing'] for i in range(2)},
        class_weights=criterion.CE_loss.weight.cpu().tolist(),
        auxiliary_class_weights={f'aux_loss_{i}': criterion.aux_loss_dict[f'aux_loss_{i}'].weight.cpu().tolist()
                                 for i in range(2)},
        aux_loss_weight=spec['aux_loss_weight'], orthogonality_rate=spec['Loss_rate'],
        aux_supervision='ovr', reduction='mean',
        formula='main weighted CE + aux_loss_weight * sum(two weighted OVR CEs) + orthogonality_rate * original full-cohort orthogonality')
    same(value, expected, f'Effective loss:{path}')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--joint', type=Path, default=DEFAULT_JOINT)
    parser.add_argument('--previous', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    out, joint, previous, reference = (getattr(args, key).resolve()
                                      for key in ('output', 'joint', 'previous', 'reference'))
    for path in (out, joint, previous):
        require((path/'COMPLETE.json').is_file(), f'Training is incomplete:{path}; no GPU use')
        require(not (path/'RUNNING.lock').exists(), f'Training lock present:{path}')
        require(read(path/'COMPLETE.json')['all_audits_passed'] is True, f'Failed completion:{path}')
    joint_audit = read(joint/'independent_audit.json')
    require(joint_audit['passed'] is True, 'Joint search independent audit did not pass')
    require(read(previous/'independent_audit.json')['passed'] is True, 'Initial search audit did not pass')
    completion = read(out/'COMPLETE.json')
    plan, specs, analysis = (read(out/name) for name in ('PLAN.json', 'all_settings.json', 'analysis.json'))
    selection = read(out/'selection.json')
    prior_plan, prior_analysis, prior_settings, prior_export = (read(joint/name) for name in
        ('PLAN.json', 'analysis.json', 'all_settings.json', 'best_configuration.json'))
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    verify_manifest(prior_plan['code_sha256'])
    verify_manifest(prior_plan['reference_sha256'])
    for filename, expected in joint_audit['artifact_sha256'].items():
        same(digest(joint/filename), expected, f'Joint audited artifact:{filename}')
    for filename, expected in prior_plan['reference_sha256'].items():
        same(plan['reference_sha256'].get(filename), expected, f'Inherited reference pin:{filename}')
    for filename, expected in prior_plan['code_sha256'].items():
        same(plan['code_sha256'].get(filename), expected, f'Inherited code pin:{filename}')
    require('tad_binary_loss_tune.py' in plan['code_sha256'], 'Loss search runner not hash-pinned')
    for filename, expected in plan['code_sha256'].items():
        same(digest(out/'source_snapshot'/filename), expected, f'Source snapshot:{filename}')
    same(digest(out/'data_splits_graphs.npz'), plan['split_archive_sha256'], 'Split archive hash')
    same(plan['baseline_protocol'], prior_plan['baseline_protocol'], 'Baseline protocol')
    for key, expected in dict(requested_changes=CHANGES, screen_folds=SCREEN, seed=0, epochs=1000).items():
        same(plan[key], expected, f'Prespecified plan:{key}')
    for key, expected in REPRODUCTION.items():
        same(plan[key], expected, f'Prespecified reproduction requirement:{key}')

    anchor = plan['anchor']
    original_name = prior_analysis['retained']
    same(original_name, joint_audit['retained'], 'Joint audited winner')
    same(original_name, prior_export['name'], 'Joint exported winner')
    same(original_name, read(joint/'COMPLETE.json')['retained'], 'Joint completed winner')
    same(prior_analysis['settings'], prior_settings[original_name], 'Joint winner settings')
    same(prior_export['overrides'], prior_settings[original_name], 'Joint exported settings')
    same(set(prior_settings[original_name]), set(KEYS), 'Anchor training keys')
    same(prior_analysis['full'], joint_audit['full'], 'Joint audited full metrics')
    joint_order = sorted(prior_analysis['full'], key=lambda name: (
        prior_analysis['full'][name]['best']['acc'], prior_analysis['full'][name]['best']['macro_f1'],
        prior_analysis['full'][name]['best']['pmci_recall'], prior_analysis['full'][name]['late100_acc']), reverse=True)
    same(original_name, joint_order[0], 'Single complete-tenfold joint winner')
    same(prior_analysis['full'][original_name]['folds'], ALL_FOLDS, 'Anchor is complete tenfold')
    folders = {}
    for fold in ALL_FOLDS:
        if original_name == 'BASE':
            folder = reference/'content'/f'fold_{fold:02d}'
        elif fold in prior_plan['reused_folds'].get(original_name, []):
            folder = previous/'runs'/original_name/f'fold_{fold:02d}'
        else:
            folder = joint/'runs'/original_name/f'fold_{fold:02d}'
        folders[str(fold)] = str(folder.relative_to(ROOT))
    expected_anchor = dict(name=original_name,
        settings={**prior_settings[original_name], 'label_smoothing': .05}, folders=folders,
        best_filename='predictions.npz' if original_name == 'BASE' else 'best_predictions.npz',
        has_last_checkpoint=original_name != 'BASE', reference_config=prior_export['config'],
        tenfold_metrics=prior_analysis['full'][original_name], round_two_plan_sha256=digest(joint/'PLAN.json'),
        round_two_independent_audit_sha256=digest(joint/'independent_audit.json'))
    same(anchor, expected_anchor, 'Anchor identity, settings and paths')
    expected_specs, aliases = {'ANCHOR': dict(expected_anchor['settings'])}, {}
    for name, change in CHANGES.items():
        spec = {**expected_anchor['settings'], **change}
        duplicate = next((key for key, existing in expected_specs.items() if existing == spec), None)
        if duplicate is None:
            expected_specs[name] = spec
        else:
            aliases[name] = duplicate
    candidates = [name for name in expected_specs if name != 'ANCHOR']
    same(specs, expected_specs, 'Single-factor loss configurations')
    same(list(specs), list(expected_specs), 'Stable insertion order')
    same(plan['aliases'], aliases, 'Deduplicated candidate aliases')
    same(plan['candidate_names'], candidates, 'Candidate names')
    same(plan['initial_screen_new_runs'], 4*len(candidates), 'Screen training budget')
    same(plan['maximum_new_runs'], 10*len(candidates), 'Maximum training budget')

    def resolve(name, fold):
        return ROOT/folders[str(fold)] if name == 'ANCHOR' else out/'runs'/name/f'fold_{fold:02d}'

    required_references = [joint/name for name in ('PLAN.json', 'all_settings.json', 'analysis.json',
        'COMPLETE.json', 'independent_audit.json', 'best_configuration.json', 'selection.json',
        'screen_vs_remaining.json', 'summary.json', 'data_splits_graphs.npz')]
    required_references.append(Path(anchor['reference_config']))
    for fold in ALL_FOLDS:
        filenames = ['best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json',
                     anchor['best_filename'], 'last_predictions.npz']
        if anchor['has_last_checkpoint']:
            filenames += ['last.pt', 'effective_settings.json']
        required_references.extend(resolve('ANCHOR', fold)/filename for filename in filenames)
    for path in required_references:
        require(str(path.relative_to(ROOT)) in plan['reference_sha256'], f'Reference not hash-pinned:{path}')

    cfg, dd, data, graphs, paths = setup(args.cuda, CONFIG, 10)
    require(dd['Sample_Num'] == 535 and dd['Class_Names'] == ['SMCI', 'PMCI'], 'Wrong task')
    same(torch.bincount(data['Label']).cpu().tolist(), [490, 45], 'Class counts')
    require(len(graphs) == 5, 'Expected original five modality graphs')
    protocol = plan['baseline_protocol']
    for path in paths:
        same(digest(path), protocol['data_sha256'][Path(path).name], f'Data hash:{path}')
    for key, value in protocol['original_config'].items():
        same(getattr(cfg, key), value, f'Original base configuration:{key}')
    source_rows = np.asarray(dd['Index'])
    require(len(source_rows) == len(np.unique(source_rows)) == 535, 'Source row identity mismatch')
    masks = torch.stack([mask for _, mask in data['Mask']]).cpu().numpy()
    np.testing.assert_array_equal(masks.sum(0), np.ones(535, dtype=int))
    for fit, test in data['Mask']:
        require(not torch.any(fit & test).item() and torch.all(fit | test).item(), 'Fold partition mismatch')
    for archive_path in (reference/'splits_and_graphs.npz', joint/'data_splits_graphs.npz', out/'data_splits_graphs.npz'):
        with np.load(archive_path) as archive:
            np.testing.assert_array_equal(archive['source_row'], source_rows)
            np.testing.assert_array_equal(archive['labels'], data['Label'].cpu().numpy())
            np.testing.assert_array_equal(archive['test_masks'], masks)
            for index, graph in enumerate(graphs):
                np.testing.assert_array_equal(archive[f'edge_{index}'], graph['edge'].numpy())
                np.testing.assert_array_equal(archive[f'neighbors_{index}'], graph['neighbors'].numpy())
    graph_hash = hashlib.sha256(b''.join(graph['edge'].numpy().tobytes() for graph in graphs)).hexdigest()
    same(graph_hash, protocol['candidate_hash'], 'Graph candidate hash')

    preflight = read(out/'preflight.json')
    require(preflight['passed'] and preflight['initialization_identical']
            and preflight['anchor_smoothing_05_first3_reproduced'], 'Loss preflight failed')
    same(set(preflight['checks']), set(specs), 'Preflight candidate coverage')
    anchor_replay = read(out/'anchor_replay.json')
    require(anchor_replay['passed'], 'Anchor preflight replay failed')
    same(anchor_replay['source_name'], original_name, 'Anchor preflight source')
    same(set(anchor_replay['folds']), {str(fold) for fold in ALL_FOLDS}, 'Anchor replay folds')
    rows, predictions, details, loss_details, initial_replay = {}, {}, {}, {}, {}
    new_runs = new_replays = anchor_replays = 0
    discovered_paths = list((out/'runs').glob('*/fold_*/complete.json'))
    discovered = {(path.parent.parent.name, int(path.parent.name.split('_')[-1])) for path in discovered_paths}
    same(len(discovered), len(discovered_paths), 'Unique candidate/fold artifact directories')
    require(all(name in candidates for name, _ in discovered), 'Unrecognized new-training arm')
    for name, spec in specs.items():
        same(set(spec), set(LOSS_KEYS), f'{name}: configuration keys')
        require(all(math.isfinite(float(spec[key])) and spec[key] > 0 for key in KEYS), f'{name}: invalid positive setting')
        require(math.isfinite(spec['label_smoothing']) and 0 <= spec['label_smoothing'] < 1, f'{name}: invalid smoothing')
        check_config_file(out/'configs'/f'{name}.ini', ROOT/CONFIG, spec)
        conf = copy.copy(cfg)
        for key, value in spec.items():
            setattr(conf, key, value)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        require(sum(p.numel() for p in model.parameters()) == 716396, f'{name}: changed architecture')
        rows[name], predictions[name], details[name], loss_details[name] = {}, {}, {}, {}
        for fold in ALL_FOLDS:
            folder = resolve(name, fold)
            if name == 'ANCHOR':
                require((folder/'complete.json').exists(), f'Missing anchor fold:{fold}')
            elif not (folder/'complete.json').exists():
                require(not folder.exists(), f'Incomplete candidate fold:{folder}')
                continue
            audited_name = original_name if name == 'ANCHOR' else name
            audited_spec = {key: spec[key] for key in KEYS} if name == 'ANCHOR' else spec
            result, prediction, detail = audit_fold(audited_name, fold, folder, model, audited_spec, data, source_rows)
            result['name'] = name
            rows[name][fold], predictions[name][fold], details[name][fold] = result, prediction, detail
            loss_details[name][fold] = check_checkpoint_loss(audited_name, fold, folder, model, spec, dd, data)
            count = sum(int(value['checkpoint_replayed']) for value in detail.values())
            if name == 'ANCHOR':
                anchor_replays += count
                for stage in ('best', 'last'):
                    same(anchor_replay['folds'][str(fold)][stage]['checkpoint_replayed'],
                         detail[stage]['checkpoint_replayed'], f'Anchor replay availability:{fold}/{stage}')
            else:
                new_runs += 1
                new_replays += count
                effective = verify_effective_loss(folder/'effective_loss.json', spec, dd, cfg.Device)
                same(preflight['checks'][name]['effective_loss'], effective, f'{name}: preflight effective loss')
            print(json.dumps(dict(audit=name, fold=fold, passed=True)), flush=True)
        del model
        if cfg.Device.type == 'cuda':
            torch.cuda.empty_cache()
        initial_replay[name] = replay_initial_steps(name, resolve(name, 1), spec, cfg, dd, data, graphs)
        recorded_first3 = preflight['checks'][name]['first3_losses']
        with (resolve(name, 1)/'epochs.csv').open(encoding='utf-8', newline='') as stream:
            first3 = list(csv.DictReader(stream))[:3]
        for epoch, (value, row) in enumerate(zip(recorded_first3, first3), 1):
            expected = float(row.get('training_loss', row.get('train_loss')))
            require(abs(value-expected) < 2e-5, f'{name}: preflight loss versus actual training at {epoch}')
    same(anchor_replays, 20 if anchor['has_last_checkpoint'] else 10, 'Reused anchor checkpoint count')
    same(new_replays, 2*new_runs, 'New checkpoint replay count')
    same(new_runs, len(discovered), 'Recognized new fold count')
    require(new_runs <= plan['maximum_new_runs'], 'Excess new training runs')
    for name in specs:
        require(set(SCREEN).issubset(rows[name]), f'{name}: missing screening fold')
    candidate_order = ordered(candidates, rows, SCREEN)
    cutoff = mean_result(rows[candidate_order[1]], SCREEN)['best']['acc']
    finalists = [name for name in candidate_order if mean_result(rows[name], SCREEN)['best']['acc'] >= cutoff-1e-12]
    for key, expected in dict(candidate_order=candidate_order, reference_order=ordered(list(specs), rows, SCREEN),
                              cutoff_acc=cutoff, finalists=finalists).items():
        same(selection[key], expected, f'Fourfold selection:{key}')
    require(2 <= len(finalists) <= len(candidates), 'Invalid finalist count')
    same(new_runs, 4*len(candidates)+6*len(finalists), 'Prespecified completed run budget')
    for name in specs:
        same(sorted(rows[name]), ALL_FOLDS if name == 'ANCHOR' or name in finalists else SCREEN,
             f'{name}: completed fold set')
    names = ['ANCHOR']+finalists
    retained = ordered(names, rows, ALL_FOLDS)[0]
    full = {name: mean_result(rows[name], ALL_FOLDS) for name in names}
    same(full['ANCHOR'], anchor['tenfold_metrics'], 'Replayed complete anchor metrics')
    same(analysis['retained'], retained, 'Retained single configuration')
    same(completion['retained'], retained, 'Completion configuration')
    same(analysis['anchor_source'], original_name, 'Analysis anchor source')
    same(analysis['settings'], specs[retained], 'Retained settings')
    same(analysis['full'], full, 'Full tenfold metrics')
    same(analysis['records'], [rows[name][fold] for name in names for fold in ALL_FOLDS], 'Fold records')
    same(analysis['new_runs'], new_runs, 'Analysis new runs')
    same(completion['new_runs'], new_runs, 'Completion new runs')
    same(analysis['verified_new_checkpoints'], new_replays, 'Analysis checkpoint count')
    summary = read(out/'summary.json')['candidates']
    same(set(summary), set(specs), 'Summary candidate coverage')
    for name in specs:
        same(summary[name], mean_result(rows[name], sorted(rows[name])), f'{name}: summary metrics')
    stage_comparison = {name: dict(screen=mean_result(rows[name], SCREEN),
        remaining_six=mean_result(rows[name], REMAINING)) for name in names}
    same(read(out/'screen_vs_remaining.json'), stage_comparison, 'Screen versus remaining six folds')
    pairs = {}
    for name in names:
        for stage in ('best', 'last'):
            ids = np.concatenate([predictions[name][fold][stage]['row_index'] for fold in ALL_FOLDS])
            require(len(ids) == len(np.unique(ids)) == 535, f'{name}/{stage}: missing/repeated OOF rows')
            np.testing.assert_array_equal(np.sort(ids), np.sort(source_rows))
        chosen = [predictions[name][fold]['best'] for fold in ALL_FOLDS]
        logits = np.concatenate([value['logits'] for value in chosen])
        labels = np.concatenate([value['labels'] for value in chosen])
        ids = np.concatenate([value['row_index'] for value in chosen])
        old_logits = np.concatenate([predictions['ANCHOR'][fold]['best']['logits'] for fold in ALL_FOLDS])
        correct, old = logits.argmax(1) == labels, old_logits.argmax(1) == labels
        pooled, _ = calculate(logits, labels)
        pairs[name] = dict(corrected=int(np.sum(correct & ~old)), harmed=int(np.sum(~correct & old)), pooled=pooled)
        if name == retained:
            with np.load(out/'retained_oof_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['labels'], labels)
                np.testing.assert_array_equal(saved['row_index'], ids)
                np.testing.assert_array_equal(saved['logits'], logits)
    same(analysis['pairs'], pairs, 'Paired corrections versus anchor and pooled metrics')
    exported = read(out/'best_configuration.json')
    for key, expected in dict(name=retained, anchor_source=original_name, overrides=specs[retained],
        graph_weight_decay=0., k=10, seed=0, epochs=1000, parameters=716396,
        model='RelationSheafModel(mode=content)').items():
        same(exported[key], expected, f'Exported configuration:{key}')
    for key, expected in REPRODUCTION.items():
        same(exported[key], expected, f'Exported reproduction requirement:{key}')
    require(Path(exported['config']).resolve() == (out/'configs'/f'{retained}.ini').resolve(), 'Exported configuration path')
    status = read(out/'status.json')
    for key, expected in dict(state='complete', retained=retained, new_runs=new_runs, finalists=finalists).items():
        same(status[key], expected, f'Final status:{key}')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    artifacts = ('PLAN.json', 'all_settings.json', 'analysis.json', 'selection.json', 'COMPLETE.json',
        'data_splits_graphs.npz', 'retained_oof_predictions.npz', 'best_configuration.json',
        'screen_vs_remaining.json', 'summary.json', 'preflight.json', 'anchor_replay.json')
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), shared_independent_auditor_sha256=digest(ROOT/'audit_tad_binary_fine_tune.py'),
        device=str(cfg.Device), independent_of_training_metric_and_rank_functions=True,
        new_runs=new_runs, reused_runs=10, replayed_new_checkpoints=new_replays,
        replayed_anchor_checkpoints=anchor_replays, prediction_files_verified=2*(new_runs+10),
        first_three_steps_independently_replayed=initial_replay, checkpoint_main_loss_replays=loss_details,
        effective_loss_manifests_verified=new_runs, anchor_source=original_name,
        reproduction_requirements=REPRODUCTION,
        finalists=finalists, retained=retained, full=full, pairs=pairs,
        screen_vs_remaining=stage_comparison, checkpoint_details=details,
        data_rows=535, class_counts=[490, 45], parameters=716396, graph_weight_decay=0.,
        graph_candidate_hash=graph_hash,
        selection_verified='First maximum test ACC; all fourfold ACC ties at second-place cutoff promoted; one configuration across all ten folds.',
        artifact_sha256={name: digest(out/name) for name in artifacts},
        limitations=(['Original BASE anchor has no last.pt; its ten last prediction files and labels were checked without weight replay.']
                     if original_name == 'BASE' else [])+[
            'Nonbaseline last prediction files omit row_index; checkpoint replay verifies the fixed test-mask order.',
            'Unique source row IDs do not establish independent clinical patient IDs.',
            'First-three-step replays verify loss-parameter execution; they are verification updates and are not new selected experiment runs.',
            'Historical test-selected epochs and adaptive reuse of test folds are retained; this is not independent validation or evidence of a global optimum.'])
    target = out/'independent_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(passed=True, retained=retained, new_runs=new_runs,
        replayed_checkpoints=new_replays+anchor_replays, mean_acc=full[retained]['best']['acc'],
        pmci_recall=full[retained]['best']['pmci_recall'], output=str(target))), flush=True)


if __name__ == '__main__':
    main()
