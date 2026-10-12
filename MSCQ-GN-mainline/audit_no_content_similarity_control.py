"""Independent CPU final audit of C versus its retrained affinity ablation.

Never imports the training runner or starts training. Before COMPLETE exists,
the script reports not_ready and writes nothing. Historical results and frozen
sources are read-only; a final audit is written only to the new result folder.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import copy
import csv
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Result/no_content_similarity_seed0_20261011'
MODE = 'NO_CONTENT_SIMILARITY'
PARAMETERS = {'C': 895096, MODE: 893560}
BASES = {'C': ROOT / 'Result/relation_mechanism_nodecay/seed_0/content',
         MODE: OUT / 'seed_0' / MODE}
FOLDS = tuple(range(1, 11))
STAGES = ('best', 'last')
SCALARS = ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier')
ARRAY_METRICS = ('per_class_recall', 'confusion_matrix')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def near(actual, expected, label, tolerance=1e-12):
    if not np.allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=False):
        raise AssertionError(label)


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def calculate(logits, labels):
    prediction = logits.argmax(1)
    precise = logits.astype(np.float64)
    maximum = precise.max(1, keepdims=True)
    exp = np.exp(precise - maximum)
    probability = exp / exp.sum(1, keepdims=True)
    target = np.eye(3)[labels]
    return dict(
        acc=float((prediction == labels).mean()),
        auc=float(roc_auc_score(target, logits, average='macro', multi_class='ovr')),
        f1=float(f1_score(labels, prediction, average='weighted', zero_division=0)),
        macro_f1=float(f1_score(labels, prediction, average='macro', zero_division=0)),
        nll=float(np.mean(np.log(exp.sum(1)) + maximum[:, 0]
                          - precise[np.arange(len(labels)), labels])),
        brier=float(np.mean(((probability - target) ** 2).sum(1))),
        per_class_recall=recall_score(labels, prediction, labels=[0, 1, 2],
                                    average=None, zero_division=0).tolist(),
        confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    )


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_provenance(plan, summary):
    for name, digest in plan['code_sha256'].items():
        require(sha(ROOT / name) == digest, 'current source ' + name)
        require(sha(OUT / 'source_snapshot' / name) == digest, 'snapshot ' + name)
    for name, digest in plan['reference_manifest'].items():
        require(sha(ROOT / name) == digest, 'historical reference ' + name)
    for name, digest in read(OUT / 'artifacts_manifest.json').items():
        require(sha(OUT / name) == digest, 'immutable artifact ' + name)
    for name, digest in plan['data_sha256'].items():
        require(sha(ROOT / 'DATASET/TADPOLE' / name) == digest, 'data ' + name)
    for key in ('source_sha256', 'analysis_source_sha256'):
        for name, digest in summary[key].items():
            require(sha(Path(name)) == digest, key + ' ' + name)
    original = read(BASES['C'].parent / 'PROTOCOL.json')
    for key in ('original_config', 'data_sha256', 'candidate_hash',
                'checkpoint_selection', 'training_seed', 'split_seed', 'epochs'):
        require(plan[key] == original[key], 'original protocol ' + key)
    require(original['graph_weight_decay_override'] == 0, 'original graph decay')
    require(plan['common_initialization_and_rng_verified'], 'construction comparison')


def verify_graphs(plan):
    with np.load(OUT / 'data_splits_and_graphs.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    require(sha(OUT / 'data_splits_and_graphs.npz') == plan['splits_artifact_sha256'],
            'new graph artifact hash')
    require(sha(BASES['C'].parent / 'splits_and_graphs.npz') == plan['splits_artifact_sha256'],
            'original graph artifact hash')
    masks = arrays['test_masks']
    require(masks.shape == (10, 598) and np.all(masks.sum(0) == 1), 'heldout coverage')
    edge_arrays = []
    for modality in range(6):
        neighbors = arrays[f'neighbors_{modality}']
        require(neighbors.shape == (598, 10), 'original top10 neighbors')
        adjacency = np.zeros((598, 598), dtype=bool)
        adjacency[np.arange(598)[:, None], neighbors] = True
        adjacency |= adjacency.T
        require(not np.diag(adjacency).any(), 'no self edges')
        edge = np.array(np.nonzero(adjacency))
        np.testing.assert_array_equal(arrays[f'edge_{modality}'], edge)
        edge_arrays.append(edge)
    digest = hashlib.sha256(b''.join(edge.tobytes() for edge in edge_arrays)).hexdigest()
    require(digest == plan['candidate_hash'], 'candidate graph hash')
    return arrays


def verify_predictions_and_tables(plan, summary, arrays):
    expected_groups = copy.deepcopy(read(BASES['C'] / 'fold_01/optimizer_groups.json'))
    expected_groups[1]['parameters'] -= 1536
    expected_groups[1]['names'].remove('propagation.edge_query.weight')
    require(expected_groups == plan['optimizer_groups'][MODE], 'planned optimizer groups')
    require(expected_groups[0]['parameters'] == 874091
            and expected_groups[1]['parameters'] == 19469, 'active optimizer counts')
    records = {mode: {} for mode in BASES}
    predictions = {mode: {} for mode in BASES}
    computed = {mode: {} for mode in BASES}
    late = {mode: {} for mode in BASES}
    for mode, base in BASES.items():
        for fold in FOLDS:
            folder = base / f'fold_{fold:02d}'
            record = read(folder / 'complete.json')
            records[mode][fold] = record
            require((record['mode'], record['seed'], record['fold'], record['parameters'])
                    == ('content' if mode == 'C' else MODE, 0, fold, PARAMETERS[mode]),
                    'fold identity')
            curve = rows(folder / 'epochs.csv')
            require([int(row['epoch']) for row in curve] == list(range(1, 1001)), '1000 epochs')
            history = np.array([[float(row[key]) for key in ('train_loss', 'test_loss', 'test_acc', 'lr')]
                                for row in curve])
            require(np.isfinite(history).all(), 'finite training history')
            reference_lr = [float(row['lr']) for row in rows(BASES['C'] / f'fold_{fold:02d}/epochs.csv')]
            near(history[:, 3], reference_lr, 'exact learning rate', 0)
            require(int(np.argmax(history[:, 2])) + 1 == record['best']['epoch'], 'first max selection')
            require(record['last']['epoch'] == 1000, 'last epoch')
            late[mode][fold] = float(history[-100:, 2].mean())
            if 'mean_last100_acc' in record:
                near(late[mode][fold], record['mean_last100_acc'], 'last100 record')
            if mode == MODE:
                require(read(folder / 'optimizer_groups.json') == expected_groups, 'fold optimizer')
                require(record['initial_model_sha256'] == plan['initialization'][MODE]['state_sha256'],
                        'fold initial state')
            else:
                require(read(folder / 'optimizer_groups.json')
                        == read(base / 'fold_01/optimizer_groups.json'), 'historical optimizer')
            predictions[mode][fold] = {}
            computed[mode][fold] = {}
            for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
                with np.load(folder / filename, allow_pickle=False) as saved:
                    logits = saved['logits'].copy()
                    labels = saved['labels'].copy()
                    patients = saved['row_index'].copy()
                mask = arrays['test_masks'][fold - 1]
                np.testing.assert_array_equal(patients, arrays['source_row'][mask])
                np.testing.assert_array_equal(labels, arrays['labels'][mask])
                require(logits.shape == (len(labels), 3) and np.isfinite(logits).all(), 'finite logits')
                metrics = calculate(logits, labels)
                computed[mode][fold][stage] = metrics
                predictions[mode][fold][stage] = dict(logits=logits, labels=labels, patients=patients,
                                                       correct=logits.argmax(1) == labels)
                for key, value in metrics.items():
                    if key in record[stage]:
                        near(value, record[stage][key], f'{mode}/{fold}/{stage}/{key}',
                             2e-5 if key in ('nll', 'brier') else 1e-12)
                near(history[record[stage]['epoch'] - 1, 2], metrics['acc'], 'selected curve accuracy')
                if mode == MODE:
                    require(sha(folder / f'{stage}.pt') == record[f'{stage}_checkpoint_sha256'],
                            'new checkpoint hash')
        table = rows(OUT / f'metrics_{mode}.csv')
        require([int(row['fold']) for row in table] == list(FOLDS), 'CSV complete folds')
        for row in table:
            fold = int(row['fold'])
            require((row['mode'], int(row['training_seed']), int(row['parameters']))
                    == (mode, 0, PARAMETERS[mode]), 'CSV identity')
            require(int(row['heldout_patients']) == int(arrays['test_masks'][fold - 1].sum()), 'CSV patients')
            require(int(row['last100_first_epoch']) == 901 and int(row['last100_final_epoch']) == 1000,
                    'CSV late interval')
            near(float(row['last100_acc']), late[mode][fold], 'CSV late mean')
            for stage in STAGES:
                require(int(row[f'{stage}_epoch']) == records[mode][fold][stage]['epoch'], 'CSV epoch')
                for key in (*SCALARS, *ARRAY_METRICS):
                    value = records[mode][fold][stage].get(key)
                    entry = row[f'{stage}_{key}']
                    if value is None:
                        require(entry == '', 'CSV absent historical metric')
                    else:
                        parsed = json.loads(entry) if key in ARRAY_METRICS else float(entry)
                        near(parsed, value, 'CSV recorded ' + key)
        variant = summary['variants'][mode]
        require(variant['completed_folds'] == 10 and variant['parameters'] == PARAMETERS[mode],
                'variant complete')
        near(variant['mean_last100_acc'], np.mean(list(late[mode].values())), 'variant late mean')
        near(variant['last100']['acc'], variant['mean_last100_acc'], 'variant late alias')
        for stage in STAGES:
            for key in SCALARS:
                if key in variant[stage]:
                    near(variant[stage][key], np.mean([records[mode][fold][stage][key] for fold in FOLDS]),
                         'variant mean ' + key)
            near(variant[stage]['std_acc'], np.std([computed[mode][fold][stage]['acc'] for fold in FOLDS]),
                 'variant population SD')
            for key in ARRAY_METRICS:
                near(variant[stage][key], np.mean([computed[mode][fold][stage][key] for fold in FOLDS], axis=0),
                     'variant array mean')
            near(variant[stage]['confusion_matrix_sum'],
                 np.sum([computed[mode][fold][stage]['confusion_matrix'] for fold in FOLDS], axis=0),
                 'variant confusion sum')
    return records, predictions, computed, late, expected_groups


def verify_pair(plan, summary, records, predictions, computed, late):
    require(len(summary['matched_comparisons']) == 1, 'one planned contrast')
    pair = summary['matched_comparisons'][0]
    require((pair['id'], pair['reference'], pair['candidate'])
            == ('retain_current_content_similarity', MODE, 'C'), 'prespecified contrast direction')
    require(pair['matched_folds'] == list(FOLDS), 'paired complete folds')
    for stage in ('best', 'last', 'last100'):
        item = pair['stages'][stage]
        deltas = [late['C'][fold] - late[MODE][fold] if stage == 'last100'
                  else computed['C'][fold][stage]['acc'] - computed[MODE][fold][stage]['acc'] for fold in FOLDS]
        near(item['mean_acc_delta'], np.mean(deltas), 'paired mean delta')
        near(item['mean_acc_delta_percentage_points'], 100 * np.mean(deltas), 'paired pp')
        require((item['fold_wins'], item['fold_ties'], item['fold_losses'])
                == (sum(delta > 1e-12 for delta in deltas), sum(abs(delta) <= 1e-12 for delta in deltas),
                    sum(delta < -1e-12 for delta in deltas)), 'paired signs')
        require(len(item['folds']) == 10, 'paired fold row count')
        for fold, row in zip(FOLDS, item['folds']):
            require(row['fold'] == fold, 'paired row fold')
            near(row['acc_delta'], deltas[fold - 1], 'paired row delta')
        if stage == 'last100':
            near(item['sd_acc_delta'], np.std(deltas, ddof=1), 'late paired SD')
            continue
        corrected = harmed = 0
        per_class = {name: dict(corrected=0, harmed=0, patients=0, recalls=[])
                     for name in plan['class_names']}
        for fold in FOLDS:
            reference, candidate = predictions[MODE][fold][stage], predictions['C'][fold][stage]
            np.testing.assert_array_equal(reference['patients'], candidate['patients'])
            np.testing.assert_array_equal(reference['labels'], candidate['labels'])
            good = ~reference['correct'] & candidate['correct']
            bad = reference['correct'] & ~candidate['correct']
            corrected += int(good.sum())
            harmed += int(bad.sum())
            for class_id, name in enumerate(plan['class_names']):
                mask = reference['labels'] == class_id
                counts = per_class[name]
                counts['corrected'] += int((good & mask).sum())
                counts['harmed'] += int((bad & mask).sum())
                counts['patients'] += int(mask.sum())
                counts['recalls'].append(float(candidate['correct'][mask].mean()
                                               - reference['correct'][mask].mean()))
        require((item['corrected'], item['harmed'], item['net'], item['patients'])
                == (corrected, harmed, corrected - harmed, 598), 'paired patient counts')
        for name, counts in per_class.items():
            saved = item['per_class'][name]
            require((saved['corrected'], saved['harmed'], saved['net'], saved['patients'])
                    == (counts['corrected'], counts['harmed'], counts['corrected'] - counts['harmed'],
                        counts['patients']), 'paired class counts')
            near(saved['mean_recall_delta'], np.mean(counts['recalls']), 'paired recall delta')
        for key, value in item['metric_deltas'].items():
            values = [records['C'][fold][stage][key] - records[MODE][fold][stage][key] for fold in FOLDS]
            for name, expected in (('mean', np.mean(values)), ('sd', np.std(values, ddof=1)),
                                   ('min', min(values)), ('max', max(values))):
                near(value[name], expected, 'paired metric ' + name)
            near(value['mean_percentage_points'], 100 * np.mean(values), 'paired metric pp')
            require((value['fold_positive'], value['fold_zero'], value['fold_negative'])
                    == (sum(x > 1e-12 for x in values), sum(abs(x) <= 1e-12 for x in values),
                        sum(x < -1e-12 for x in values)), 'metric signs')
            for fold, row in zip(FOLDS, value['folds']):
                require(row['fold'] == fold, 'metric delta row fold')
                near(row['reference'], records[MODE][fold][stage][key], 'metric reference')
                near(row['candidate'], records['C'][fold][stage][key], 'metric candidate')
                near(row['delta'], values[fold - 1], 'metric delta row')


def cpu_checkpoints(plan, arrays, records, predictions, expected_groups):
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA before CPU setup')
    from relation_experiment import setup
    from Model.no_content_similarity_control import NoContentSimilarityControlModel
    from Model.relation_sheaf import RelationSheafModel, conformal_maps
    from Utils import SET_Random
    from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

    cfg, dd, data, graphs, _ = setup(cuda=-1)
    require(str(cfg.Device) == 'cpu' and data['Feature'].device.type == 'cpu', 'CPU setup')
    np.testing.assert_array_equal(np.asarray(dd['Index']), arrays['source_row'])
    np.testing.assert_array_equal(data['Label'].numpy(), arrays['labels'])
    for fold, (_, test) in enumerate(data['Mask']):
        np.testing.assert_array_equal(test.numpy(), arrays['test_masks'][fold])
    for modality, graph in enumerate(graphs):
        np.testing.assert_array_equal(graph['edge'].numpy(), arrays[f'edge_{modality}'])
    SET_Random(0)
    native = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).eval()
    require(state_sha(native) == plan['reference_audit']['initial_model_sha256'], 'native initial hash')
    cpu_stream = torch.get_rng_state().clone()
    SET_Random(0)
    model = NoContentSimilarityControlModel(dd, cfg, graphs, MODE, training_seed=0).eval()
    require(torch.equal(torch.get_rng_state(), cpu_stream), 'constructor CPU RNG')
    require(state_sha(model) == plan['initialization'][MODE]['state_sha256'], 'CPU initial hash')
    require(sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[MODE], 'CPU capacity')
    require(optimizer_manifest(model, relation_optimizer(model, cfg, 0.)) == expected_groups, 'CPU optimizer')
    removed = 'propagation.edge_query.weight'
    require(set(model.state_dict()) == set(native.state_dict()) - {removed}, 'only query removed')
    for name, value in model.state_dict().items():
        require(torch.equal(value, native.state_dict()[name]), 'common tensor ' + name)
    expected_edge, expected_similarity, expected_group = [], [], []
    for relation, graph in enumerate(graphs):
        for content in range(6):
            group = relation * 6 + content
            expected_edge.append(graph['edge'] + group * 598)
            expected_similarity.append(graph['similarity'])
            expected_group.append(torch.full((graph['edge'].shape[1],), group, dtype=torch.long))
    for name, expected in (('edge', torch.cat(expected_edge, 1)),
                           ('similarity', torch.cat(expected_similarity)),
                           ('edge_group', torch.cat(expected_group))):
        require(torch.equal(getattr(model.propagation, name), expected), '36 path buffer ' + name)
    del native
    checkpoints, replays = [], []
    for fold in FOLDS:
        historical = torch.load(BASES['C'] / f'fold_{fold:02d}/best.pt', map_location='cpu', weights_only=False)
        require((historical['mode'], historical['fold'], historical['seed'], historical['protocol'])
                == ('content', fold, 0, 'original test-selected'), 'historical checkpoint schema')
        require(historical['best'] == records['C'][fold]['best'], 'historical best metadata')
        for stage in STAGES:
            folder = BASES[MODE] / f'fold_{fold:02d}'
            checkpoint = torch.load(folder / f'{stage}.pt', map_location='cpu', weights_only=False)
            require((checkpoint['mode'], checkpoint['fold'], checkpoint['seed'], checkpoint['epoch'])
                    == (MODE, fold, 0, records[MODE][fold][stage]['epoch']), 'new checkpoint identity')
            require(checkpoint['protocol_sha256'] == sha(OUT / 'PLAN.json'), 'checkpoint protocol SHA')
            require(checkpoint['metrics'] == records[MODE][fold][stage] == read(folder / f'{stage}.json'),
                    'checkpoint metrics metadata')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            propagation = model.propagation
            require(not hasattr(propagation, 'edge_query') and removed not in checkpoint['state_dict'],
                    'query absent in module and checkpoint')
            require(tuple(propagation.edge_similarity.shape) == tuple(propagation.edge_bias.shape) == (6, 6),
                    'relation-content coefficients')
            require(tuple(propagation.content_id.weight.shape) == (6, 16), 'content specific IDs')
            require(propagation.mode == 'content' and int(propagation.edge_group.max()) + 1 == 36,
                    'content mode and 36 groups')
            with torch.no_grad():
                raw = propagation.conditioner(propagation.content_id(torch.arange(6)))
                send, receive, send_scale, receive_scale = conformal_maps(raw)
                weight = torch.sigmoid(propagation.similarity
                                       * propagation.edge_similarity.flatten()[propagation.edge_group]
                                       + propagation.edge_bias.flatten()[propagation.edge_group])
            require(torch.isfinite(raw).all().item() and torch.isfinite(weight).all().item(), 'finite maps/weights')
            require(torch.all((send_scale > 0) & (send_scale < 2)
                              & (receive_scale > 0) & (receive_scale < 2)).item(), 'map scale range')
            checkpoints.append(dict(fold=fold, stage=stage, query_absent=True, active_paths=36,
                                    coefficient_shape=[6, 6], content_id_shape=[6, 16],
                                    send_scale=send_scale.tolist(), receive_scale=receive_scale.tolist(),
                                    edge_weight_range=[float(weight.min()), float(weight.max())]))
            if fold == 1:
                with torch.no_grad():
                    logits = model(data['Feature'])[0][data['Mask'][0][1]].numpy()
                expected = predictions[MODE][fold][stage]['logits']
                np.testing.assert_allclose(logits, expected, atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(logits.argmax(1), expected.argmax(1))
                replays.append(dict(fold=1, stage=stage, device='cpu',
                                    max_absolute_logit_error=float(np.max(np.abs(logits - expected))),
                                    all_predicted_classes_identical=True))
    require(not torch.cuda.is_initialized(), 'CUDA stayed uninitialized')
    return dict(passed=True, new_checkpoint_metadata_and_structure_checked=20,
                historical_best_checkpoint_metadata_checked=10, checkpoints=checkpoints,
                independent_cpu_forward_replays=replays, cuda_initialized=False,
                remaining_new_forward_replays='All20 new checkpoints replayed in runner GPU audit.')


def main():
    if not (OUT / 'COMPLETE.json').exists() or not all(
            (BASES[MODE] / f'fold_{fold:02d}/complete.json').exists() for fold in FOLDS):
        print(json.dumps(dict(state='not_ready', final_audit_written=False)))
        return 2
    if (OUT / 'independent_audit.json').exists():
        raise FileExistsError('Final successful independent audit already exists')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), passed=False,
                  cpu_only=True, no_new_training=True, old_results_and_frozen_sources_read_only=True,
                  checks={}, errors=[])
    try:
        plan = read(OUT / 'PLAN.json')
        summary = read(OUT / 'summary.json')
        complete = read(OUT / 'COMPLETE.json')
        require(summary['all_two_complete'] and summary['all_new_complete']
                and summary['all_reference_complete'] and summary['completed_new_runs'] == 10, 'completion')
        require(complete['summary'] == summary, 'COMPLETE embedded summary')
        require(complete['audit']['passed'] and complete['audit']['verified_checkpoints'] == 20,
                'runner replay audit')
        require(plan['training_seed'] == plan['split_seed'] == 0 and plan['epochs'] == 1000
                and plan['folds'] == list(FOLDS) and plan['modes'] == [MODE], 'planned scope')
        require(plan['parameters'] == PARAMETERS, 'planned parameters')
        verify_provenance(plan, summary)
        report['checks']['frozen_provenance'] = dict(passed=True, source_count=len(plan['code_sha256']),
            reference_hash_count=len(plan['reference_manifest']), analysis_input_hash_count=len(summary['source_sha256']))
        arrays = verify_graphs(plan)
        report['checks']['original_graphs_and_folds'] = dict(passed=True, original_graphs=6,
            topk=10, sample_count=598, each_patient_heldout_once=True)
        records, predictions, computed, late, expected_groups = verify_predictions_and_tables(plan, summary, arrays)
        report['checks']['records_predictions_csv'] = dict(passed=True, folds=20, prediction_files=40,
            new_checkpoint_hashes=20, csv_files=2, unique_patients_per_arm=598)
        verify_pair(plan, summary, records, predictions, computed, late)
        report['checks']['paired_contrast'] = dict(passed=True, contrast='C minus NO_CONTENT_SIMILARITY',
            stages=['best', 'last', 'last100'], patients_paired=True, equal_fold_aggregation=True)
        report['checks']['cpu_checkpoints_and_replay'] = cpu_checkpoints(plan, arrays, records, predictions, expected_groups)
        verify_provenance(plan, summary)
        report['limitations'] = [
            'Historical seed0/test-selection protocol retained; exploratory descriptive estimates.',
            'Removing the dedicated query reduces1536 active parameters; not strict capacity matching.',
            'Relation/content indexed a,b and content-specific S/T remain; not all content dependence removed.',
            'Original C last.pt is absent: its last-stage evidence is saved predictions, not checkpoint replay.',
            'Independent CPU forward replay is fold1 best/last; runner audit replays all20 new checkpoints on GPU.',
        ]
        report['delivery_hashes'] = {path.name: sha(path) for path in
            [OUT / 'PLAN.json', OUT / 'COMPLETE.json', OUT / 'summary.json',
             *[OUT / f'metrics_{mode}.csv' for mode in BASES]]}
        report['passed'] = True
    except Exception as error:
        report['errors'].append(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['checker_sha256'] = sha(Path(__file__))
    target = OUT / ('independent_audit.json' if report['passed'] else 'independent_audit_failed.json')
    if target.exists():
        target = OUT / (target.stem + '_' + datetime.now().strftime('%H%M%S') + '.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=report['passed'], path=str(target), checks=list(report['checks']),
                         errors=report['errors']), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
