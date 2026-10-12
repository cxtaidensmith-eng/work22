"""Frozen C checkpoints: relation/content edge-weight decomposition.

No training, model selection, topology changes, or classification interventions.
All summaries first average receiving edges within each held-out patient, then
average patients within a fold. Between-fold variability is descriptive only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import Model.relation_sheaf as sheaf_module
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import ROOT, setup, dump, sha, timestamp


REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0'
DEFAULT_OUT = ROOT / 'Result/c_edge_content_weights_20261009'
ATOL, RTOL = 2e-5, 2e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty results')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def receiver_means(values, receivers):
    """One equally weighted summary per receiving patient, not per edge."""
    unique, inverse = np.unique(receivers, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    return unique, sums / counts


def balanced_mean(values, receivers):
    return float(receiver_means(values, receivers)[1].mean())


def finite_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise FloatingPointError('Non-finite extracted values')
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=0)),
                minimum=float(a.min()), p05=float(np.quantile(a, .05)),
                median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max()))


def verify_inputs(cfg, dd, data, graphs, paths):
    original = read(REFERENCE / 'PROTOCOL.json')
    assert cfg.DATA_SET == 'TADPOLE' and cfg.Task == 'AD_CN_SMCI'
    assert original['training_seed'] == original['split_seed'] == cfg.seed == 0
    assert dd['Sample_Num'] == 598 and len(dd['Modal_Name']) == 6
    for key, value in original['original_config'].items():
        assert getattr(cfg, key) == value, key
    for name, digest in original['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert {Path(p).name: sha(p) for p in paths} == original['data_sha256']
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    assert graph_hash == original['candidate_hash']
    masks = np.stack([test.cpu().numpy() for _, test in data['Mask']])
    assert masks.shape == (10, 598) and (masks.sum(0) == 1).all()
    with np.load(REFERENCE / 'splits_and_graphs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_masks'], masks)
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        for r, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{r}'], graph['edge'].numpy())
    return original, graph_hash


@torch.no_grad()
def extract_fold(fold, cfg, dd, data, graphs, out, modalities):
    folder = REFERENCE / 'content' / f'fold_{fold:02d}'
    checkpoint_path = folder / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=cfg.Device, weights_only=False)
    assert checkpoint['mode'] == 'content' and checkpoint['seed'] == 0 and checkpoint['fold'] == fold
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 895096
    state_before = tensor_hash(model)
    captured = {}

    def token_hook(module, args):
        assert 'tokens' not in captured
        captured['tokens'] = args[0].detach().clone()

    original_operator = sheaf_module.cooperative_operator

    def operator_wrapper(x, edge, weight, *args):
        assert 'weight' not in captured
        captured['weight'] = weight.detach().clone()
        return original_operator(x, edge, weight, *args)

    hook = model.propagation.register_forward_pre_hook(token_hook)
    try:
        sheaf_module.cooperative_operator = operator_wrapper
        logits = model(data['Feature'])[0]
    finally:
        hook.remove()
        sheaf_module.cooperative_operator = original_operator
    assert tensor_hash(model) == state_before
    assert not model.training and all(not module.training for module in model.modules())
    test = data['Mask'][fold - 1][1]
    test_np = test.cpu().numpy()
    with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test_np])
        np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
        actual = logits[test].cpu().numpy()
        np.testing.assert_allclose(actual, saved['logits'], atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(actual.argmax(1), saved['logits'].argmax(1))
        replay_error = float(np.abs(actual - saved['logits']).max())
    p = model.propagation
    z = p.input_norm(captured['tokens'])
    count, m, _ = z.shape
    q = F.normalize(p.edge_query(z), dim=-1).permute(1, 0, 2)
    q = q[None].expand(m, -1, -1, -1).reshape(m * m * count, -1)
    i, j = p.edge
    affinity = (q[i] * q[j]).sum(-1)
    relation_term = p.similarity * p.edge_similarity.flatten()[p.edge_group]
    bias_term = p.edge_bias.flatten()[p.edge_group]
    logit = affinity + relation_term + bias_term
    full_weight = torch.sigmoid(logit)
    torch.testing.assert_close(full_weight, captured['weight'], atol=0, rtol=0)
    no_affinity_weight = torch.sigmoid(relation_term + bias_term)
    for a in (affinity, relation_term, bias_term, full_weight, no_affinity_weight):
        assert torch.isfinite(a).all()
    assert (full_weight > 0).all() and (full_weight < 1).all()
    arrays = dict(modality_names=np.asarray(modalities, dtype='U32'),
                  fold=np.asarray(fold), test_patient_count=np.asarray(int(test.sum())),
                  selection_epoch=np.asarray(checkpoint['best']['epoch']))
    content_rows, pair_rows, patient_rows = [], [], []
    symmetry_error = 0.
    group_index = p.edge_group.cpu().numpy()
    all_receiver = (i % count).cpu().numpy()
    all_sender = (j % count).cpu().numpy()
    fields = dict(weight=full_weight, no_affinity_weight=no_affinity_weight,
                  affinity=affinity, relation_term=relation_term, bias_term=bias_term,
                  logit=logit)
    source_arrays = {key: value.cpu().numpy() for key, value in fields.items()}
    for r, relation in enumerate(modalities):
        group_masks = [group_index == r * m + c for c in range(m)]
        base_recv, base_send = all_receiver[group_masks[0]], all_sender[group_masks[0]]
        np.testing.assert_array_equal(np.stack([all_receiver[a] for a in group_masks]),
                                      np.broadcast_to(base_recv, (m, len(base_recv))))
        np.testing.assert_array_equal(np.stack([all_sender[a] for a in group_masks]),
                                      np.broadcast_to(base_send, (m, len(base_send))))
        all_key = base_recv * count + base_send
        order = np.argsort(all_key)
        rev_position = np.searchsorted(all_key[order], base_send * count + base_recv)
        reverse = order[rev_position]
        np.testing.assert_array_equal(all_key[reverse], base_send * count + base_recv)
        full_group = np.stack([source_arrays['weight'][a] for a in group_masks])
        symmetry_error = max(symmetry_error, float(np.abs(full_group - full_group[:, reverse]).max()))
        selected = test_np[base_recv]
        receivers, senders = base_recv[selected], base_send[selected]
        arrays[f'r{r}_receiver'], arrays[f'r{r}_sender'] = receivers, senders
        arrays[f'r{r}_similarity'] = graphs[r]['similarity'].numpy()[selected]
        unique, counts = np.unique(receivers, return_counts=True)
        assert len(unique) == int(test.sum()) and counts.min() >= 10
        for key in fields:
            arrays[f'r{r}_{key}'] = np.stack([source_arrays[key][a][selected] for a in group_masks])
        group_affinity_mean = arrays[f'r{r}_affinity'].mean(axis=1, dtype=np.float64).astype(np.float32)
        constant_logit = (arrays[f'r{r}_relation_term'] + arrays[f'r{r}_bias_term']
                          + group_affinity_mean[:, None])
        # Replacing affinity by its empirical group mean retains its offset,
        # while removing edge-specific affinity variability. No logits are rerun.
        arrays[f'r{r}_constant_affinity_weight'] = torch.sigmoid(torch.from_numpy(constant_logit)).numpy()
        arrays[f'r{r}_constant_affinity_value'] = group_affinity_mean
        for c, content in enumerate(modalities):
            w = arrays[f'r{r}_weight'][c]
            w0 = arrays[f'r{r}_no_affinity_weight'][c]
            wc = arrays[f'r{r}_constant_affinity_weight'][c]
            row = dict(fold=fold, relation=relation, content=content,
                       relation_index=r, content_index=c, test_patients=len(unique),
                       receiver_edge_count=len(receivers),
                       relation_coefficient=float(p.edge_similarity[r, c]),
                       type_bias=float(p.edge_bias[r, c]),
                       mean_weight=balanced_mean(w, receivers),
                       mean_no_affinity_weight=balanced_mean(w0, receivers),
                       mean_abs_drop_affinity=balanced_mean(np.abs(w - w0), receivers),
                       mean_abs_replace_constant_affinity=balanced_mean(np.abs(w - wc), receivers))
            for key in ('weight', 'affinity', 'relation_term'):
                summary = finite_summary(arrays[f'r{r}_{key}'][c])
                row.update({f'edge_{key}_{key2}': value for key2, value in summary.items()})
            content_rows.append(row)
        for a, b in itertools.combinations(range(m), 2):
            components = {
                'weight_difference': arrays[f'r{r}_weight'][a] - arrays[f'r{r}_weight'][b],
                'no_affinity_weight_difference': arrays[f'r{r}_no_affinity_weight'][a] - arrays[f'r{r}_no_affinity_weight'][b],
                'constant_affinity_weight_difference': arrays[f'r{r}_constant_affinity_weight'][a] - arrays[f'r{r}_constant_affinity_weight'][b],
                'affinity_difference': arrays[f'r{r}_affinity'][a] - arrays[f'r{r}_affinity'][b],
                'relation_term_difference': arrays[f'r{r}_relation_term'][a] - arrays[f'r{r}_relation_term'][b],
                'bias_difference': arrays[f'r{r}_bias_term'][a] - arrays[f'r{r}_bias_term'][b],
                'logit_difference': arrays[f'r{r}_logit'][a] - arrays[f'r{r}_logit'][b],
            }
            np.testing.assert_allclose(components['logit_difference'],
                components['affinity_difference'] + components['relation_term_difference']
                + components['bias_difference'], atol=1e-6, rtol=1e-5)
            row = dict(fold=fold, relation=relation, content_a=modalities[a], content_b=modalities[b],
                       relation_index=r, content_a_index=a, content_b_index=b,
                       test_patients=len(unique), receiver_edge_count=len(receivers))
            patient_values = {}
            for key, value in components.items():
                ids, means = receiver_means(np.abs(value), receivers)
                np.testing.assert_array_equal(ids, unique)
                row[f'mean_abs_{key}'] = float(means.mean())
                row[f'mean_signed_{key}'] = balanced_mean(value, receivers)
                patient_values[key] = means
            wa, wb = arrays[f'r{r}_weight'][a].astype(np.float64), arrays[f'r{r}_weight'][b].astype(np.float64)
            row['edge_pearson_correlation'] = (float(np.corrcoef(wa, wb)[0, 1])
                if wa.std() > 1e-12 and wb.std() > 1e-12 else None)
            pair_rows.append(row)
            for patient_ordinal in range(len(unique)):
                patient_row = dict(fold=fold, heldout_ordinal=patient_ordinal,
                    relation=relation, content_a=modalities[a], content_b=modalities[b])
                patient_row.update({f'mean_abs_{key}': float(v[patient_ordinal])
                                    for key, v in patient_values.items()})
                patient_rows.append(patient_row)
    assert symmetry_error <= 1e-7
    np.savez_compressed(out / f'fold_{fold:02d}.npz', **arrays)
    audit = dict(fold=fold, seed=0, selection_epoch=checkpoint['best']['epoch'],
        test_patients=int(test.sum()), parameter_count=895096,
        checkpoint_sha256=sha(checkpoint_path), original_predictions_sha256=sha(folder / 'predictions.npz'),
        original_acc=checkpoint['best']['acc'], prediction_argmax_match=True,
        replay_max_abs_logit_error=replay_error, extracted_weight_max_abs_error=0.,
        reverse_edge_max_abs_weight_error=symmetry_error, unchanged_model_state=True,
        output_sha256=sha(out / f'fold_{fold:02d}.npz'))
    del model
    return content_rows, pair_rows, patient_rows, audit


def aggregate_rows(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for identity, members in groups.items():
        assert len(members) == 10
        item = dict(zip(keys, identity))
        item['fold_count'] = len(members)
        for field in members[0]:
            if field.startswith('mean_'):
                values = np.asarray([row[field] for row in members], dtype=np.float64)
                item[field] = float(values.mean())
                item[field + '_fold_sd'] = float(values.std(ddof=1))
                item[field + '_fold_min'] = float(values.min())
                item[field + '_fold_max'] = float(values.max())
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cuda', type=int, default=-1)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty analysis directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cfg, dd, data, graphs, paths = setup(cuda=args.cuda, k=10)
    original, graph_hash = verify_inputs(cfg, dd, data, graphs, paths)
    names = list(dd['Modal_Name'])
    assert names[:2] == ['MRI', 'PET']
    plan = dict(created_at=timestamp(), analysis='C fixed-checkpoint content-dependent edge weights',
        task='TADPOLE AD/CN/SMCI', training_seed=0, folds=list(range(1, 11)),
        checkpoint_selection='Historical first maximum TEST ACC; no new selection',
        device=str(cfg.Device), no_training=True, no_classification_interventions=True,
        main_relation='MRI', main_content_pair=['MRI', 'PET'], all_relation_and_content_pairs_exported=True,
        modalities=names, sample_count=598, candidate_hash=graph_hash,
        weight_formula='sigmoid(content_affinity + a[relation,content]*raw_similarity + b[relation,content])',
        no_affinity_formula='sigmoid(a*raw_similarity+b); frozen parameters; edge-weight diagnostic only',
        constant_affinity_formula='sigmoid(a*raw_similarity+b+mean_affinity[fold,relation,content])',
        constant_affinity_scope='Mean over the selected held-out receiving edges; diagnostic, no prediction rerun',
        sampling='All candidate incoming edges whose receiver belongs to the held-out fold',
        statistical_unit='Receiving patient: edge means within patient, then equal patient mean within fold',
        aggregation='Equal mean of ten folds; sample fold SD is descriptive, not a confidence interval',
        dependence='Patients share edges; both orientations occur when both receivers are held-out; no edge-wise p values',
        paired_logit_difference='Signed difference exactly decomposes; absolute component magnitudes are not additive',
        scope='Within historical exploratory checkpoints; weight differences do not prove classification utility or clinical causality',
        original_protocol_sha256=sha(REFERENCE / 'PROTOCOL.json'),
        source_sha256={str(Path(p).relative_to(ROOT)).replace('\\', '/'): sha(p)
                       for p in [Path(__file__), ROOT / 'Model/relation_sheaf.py',
                                 ROOT / 'relation_experiment.py', ROOT / 'Config/main_tadpole_3class.ini']},
        data_sha256=original['data_sha256'])
    dump(out / 'PLAN.json', plan)
    started = time.perf_counter()
    content_rows, pair_rows, patient_rows, audits = [], [], [], []
    for fold in range(1, 11):
        c, p, patients, audit = extract_fold(fold, cfg, dd, data, graphs, out, names)
        content_rows.extend(c)
        pair_rows.extend(p)
        patient_rows.extend(patients)
        audits.append(audit)
        dump(out / 'status.json', dict(state='running', completed_folds=fold,
             planned_folds=10, no_training=True, updated_at=timestamp()))
        print(json.dumps(dict(fold=fold, replay_max_error=audit['replay_max_abs_logit_error'],
                              extracted_weight_error=0, test_patients=audit['test_patients'])), flush=True)
    assert sum(row['test_patients'] for row in audits) == 598
    assert len(content_rows) == 360 and len(pair_rows) == 900 and len(patient_rows) == 598 * 6 * 15
    write_csv(out / 'fold_content_metrics.csv', content_rows)
    write_csv(out / 'fold_pair_metrics.csv', pair_rows)
    write_csv(out / 'patient_pair_metrics.csv', patient_rows)
    write_csv(out / 'replay_audit.csv', audits)
    content_aggregate = aggregate_rows(content_rows, ['relation', 'content', 'relation_index', 'content_index'])
    pair_aggregate = aggregate_rows(pair_rows, ['relation', 'content_a', 'content_b',
                                               'relation_index', 'content_a_index', 'content_b_index'])
    write_csv(out / 'aggregate_content_metrics.csv', content_aggregate)
    write_csv(out / 'aggregate_pair_metrics.csv', pair_aggregate)
    main_pair = next(row for row in pair_aggregate if row['relation'] == 'MRI'
                     and row['content_a'] == 'MRI' and row['content_b'] == 'PET')
    main_content = [row for row in content_aggregate if row['relation'] == 'MRI']
    summary = dict(completed_at=timestamp(), folds=10, sample_count=598, modalities=names,
        no_training=True, main_pair=main_pair, main_relation_contents=main_content,
        maximum_replay_logit_error=max(row['replay_max_abs_logit_error'] for row in audits),
        all_argmax_predictions_match=True, all_extracted_weights_exactly_match_forward=True,
        content_rows=len(content_rows), pair_rows=len(pair_rows), patient_pair_rows=len(patient_rows),
        elapsed_seconds=time.perf_counter() - started,
        interpretation=plan['scope'])
    dump(out / 'summary.json', summary)
    manifest = {file.name: dict(sha256=sha(file), bytes=file.stat().st_size)
                for file in sorted(out.iterdir()) if file.is_file() and file.name != 'status.json'}
    dump(out / 'COMPLETE.json', dict(passed=True, no_training=True, folds=10,
         all_predictions_replayed=True, all_forward_weights_exact=True, files=manifest))
    dump(out / 'status.json', dict(state='complete', completed_folds=10, planned_folds=10,
                                 no_training=True, updated_at=timestamp()))
    print(json.dumps(dict(state='complete', main_pair=main_pair), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
