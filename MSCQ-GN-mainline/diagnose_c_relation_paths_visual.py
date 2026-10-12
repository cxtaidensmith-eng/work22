"""Fixed-checkpoint, off-patient dependence of C's 36 relation/content paths.

No fitting, optimizer, checkpoint selection, topology change or new seed. Each
intervention retains the exact diagonal blocks of the COMPOSED two-pass
operator, including graph backtracking to the same patient. Only one path's
off-patient term is removed; all degrees, weights, S/T maps and the original
1/M relation average remain fixed. Entries are nonadditive dependencies of
historically TEST-selected checkpoints, not causal or clinical contributions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp
from Model.relation_sheaf import RelationSheafModel, cooperative_operator, conformal_maps
from Model.reference_selection import composed_self_part
from diagnose_c_disagreements import FrozenFlow, overridden_logits


OUT = ROOT/'Result/c_shared_mapping_visualization_20261003/relation_paths'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0'
FOLDS = tuple(range(1, 11))
ATOL, RTOL = 2e-5, 2e-5
NLL_SIGN_EPSILON = 1e-8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_source_closure():
    """Freeze local Python import dependencies, including package entry points."""
    pending, found = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        parts = path.parent.relative_to(ROOT).parts
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name.split('.') for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = list(parts[:len(parts)-node.level+1]) if node.level else []
                base = prefix + (node.module.split('.') if node.module else [])
                modules = [base] + [base+item.name.split('.') for item in node.names if item.name != '*']
            for module in modules:
                if not module:
                    continue
                candidate = ROOT.joinpath(*module)
                choices = [candidate.with_suffix('.py'), candidate/'__init__.py']
                choices += [ROOT.joinpath(*module[:i])/'__init__.py' for i in range(1, len(module))]
                pending.extend(p.resolve() for p in choices if p.is_file() and p.resolve() not in found)
    return sorted(str(path.relative_to(ROOT)).replace('\\', '/') for path in found)


def hash_tensors(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state(model):
    return dict(parameters_and_persistent_buffers=hash_tensors(model.state_dict().items()),
                all_buffers=hash_tensors(model.named_buffers()),
                training_flags={name: module.training for name, module in model.named_modules()},
                hooks={name: [len(module._forward_pre_hooks), len(module._forward_hooks)]
                       for name, module in model.named_modules()})


def rng_state():
    return [torch.get_rng_state().clone()] + [value.clone() for value in torch.cuda.get_rng_state_all()]


def assert_rng_equal(left, right):
    assert len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def assert_close(left, right):
    torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
    return float((left-right).abs().max())


@torch.no_grad()
def dense_self_regression():
    """Independent 8x8 dense operator checks the complete diagonal in float64."""
    count, channels = 4, 3
    adjacency = torch.tensor([[0., .7, .4, 0.], [.7, 0., .2, .6],
                              [.4, .2, 0., .5], [0., .6, .5, 0.]], dtype=torch.float64)
    edge = adjacency.nonzero().T.contiguous()
    weights = adjacency[edge[0], edge[1]]
    parameter = torch.tensor([.21, -.36, .42, -.27], dtype=torch.float64)
    send, receive, cs, ct = conformal_maps(parameter[None].expand(count, -1))
    generator = torch.Generator().manual_seed(81031)
    values = torch.randn(count, 2, channels, generator=generator, dtype=torch.float64)
    degree = adjacency.sum(1)
    matrices = []
    for scale in (cs, ct):
        diagonal = degree*scale.square()
        inverse = (diagonal+1).rsqrt()
        matrix = torch.zeros(count*2, count*2, dtype=torch.float64)
        for i in range(count):
            matrix[2*i:2*i+2, 2*i:2*i+2] = torch.eye(2, dtype=torch.float64)*diagonal[i]/(diagonal[i]+1)
            for j in range(count):
                if i != j:
                    matrix[2*i:2*i+2, 2*j:2*j+2] = (
                        -adjacency[i, j]*inverse[i]*inverse[j]*(receive[i].T@send[j]))
        matrices.append(matrix)
    composed = matrices[1]@matrices[0]
    expected_full = (composed@values.reshape(count*2, channels)).reshape_as(values)
    expected_own = torch.stack([composed[2*i:2*i+2, 2*i:2*i+2]@values[i] for i in range(count)])
    actual_full = cooperative_operator(values, edge, weights, send, receive, cs, ct)
    constant = composed_self_part(values, edge, weights, send, receive, cs, ct, constant_maps=True)
    general = composed_self_part(values, edge, weights, send, receive, cs, ct, constant_maps=False)
    for actual, expected in ((actual_full, expected_full), (constant, expected_own),
                             (general, expected_own), (actual_full-constant, expected_full-expected_own)):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    ds, dt = degree*cs.square(), degree*ct.square()
    naive_own = (ds/(ds+1)*dt/(dt+1))[:, None, None]*values
    backtracking = float((expected_own-naive_own).norm())
    assert backtracking > 1e-6, 'The regression must actually exercise the omitted return paths'
    return dict(passed=True, dtype='float64', patients=count, dense_dimension=count*2,
        full_max_error=float((actual_full-expected_full).abs().max()),
        constant_self_max_error=float((constant-expected_own).abs().max()),
        general_self_max_error=float((general-expected_own).abs().max()),
        nonzero_backtracking_norm=backtracking,
        interpretation='Self is the complete block diagonal of L_receive @ L_send, including two-hop return paths.')


def validate_input(cfg, dd, data, graphs, paths):
    original = read(REFERENCE/'PROTOCOL.json')
    assert cfg.DATA_SET == 'TADPOLE' and cfg.Task == 'AD_CN_SMCI'
    assert original['training_seed'] == original['split_seed'] == cfg.seed == 0
    assert dd['Sample_Num'] == 598 and len(dd['Modal_Name']) == 6 and dd['Class_Num'] == 3
    for name, value in original['original_config'].items():
        assert getattr(cfg, name) == value, name
    for name, digest in original['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert {Path(path).name: sha(path) for path in paths} == original['data_sha256']
    graph_hash = hashlib.sha256(b''.join(graph['edge'].numpy().tobytes() for graph in graphs)).hexdigest()
    assert graph_hash == original['candidate_hash']
    masks = np.stack([test.cpu().numpy() for _, test in data['Mask']])
    assert len(masks) == 10 and (masks.sum(0) == 1).all()
    with np.load(REFERENCE/'splits_and_graphs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_masks'], masks)
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        for index, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{index}'], graph['edge'].numpy())
            np.testing.assert_array_equal(saved[f'neighbors_{index}'], graph['neighbors'].numpy())
    files = [REFERENCE/'PROTOCOL.json', REFERENCE/'splits_and_graphs.npz',
             ROOT/'reference_fixed10/data_and_splits.npz', ROOT/'Config/main_tadpole_3class.ini']
    files.extend(Path(path) for path in paths)
    for fold in FOLDS:
        folder = REFERENCE/f'content/fold_{fold:02d}'
        files.extend(folder/name for name in ('best.pt', 'best.json', 'complete.json', 'predictions.npz'))
    inputs = {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): sha(path) for path in files}
    return original, inputs


def verify_manifest(plan):
    for collection in ('code_sha256', 'input_sha256'):
        for name, digest in plan[collection].items():
            assert sha(ROOT/name) == digest, (collection, name)


def write_csv(path, rows):
    assert rows
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def numpy_nll(logits, labels):
    # Scoring is consistently float64/CPU for baseline and all interventions.
    return float(F.cross_entropy(torch.as_tensor(logits, dtype=torch.float64),
                                 torch.as_tensor(labels, dtype=torch.long)))


@torch.no_grad()
def run_fold(cfg, dd, data, graphs, fold):
    source = REFERENCE/f'content/fold_{fold:02d}'
    checkpoint = torch.load(source/'best.pt', map_location=cfg.Device, weights_only=False)
    record = read(source/'complete.json')
    assert (record['mode'], record['fold'], record['seed']) == ('content', fold, 0)
    assert checkpoint['best']['epoch'] == record['best']['epoch']
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 895096
    assert all(not module.training for module in model.modules())
    original_state, original_rng = model_state(model), rng_state()
    x, y = data['Feature'], data['Label']
    test = data['Mask'][fold-1][1]
    flow = FrozenFlow(model, x)
    with np.load(source/'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
        np.testing.assert_array_equal(saved['labels'], y[test].cpu().numpy())
        saved_logits = torch.as_tensor(saved['logits'], device=cfg.Device)
        saved_error = assert_close(flow.logits[test], saved_logits)
        assert torch.equal(flow.logits[test].argmax(-1), saved_logits.argmax(-1))
    baseline = flow.logits[test].cpu().numpy()
    labels = y[test].cpu().numpy()
    base_prediction = baseline.argmax(-1)
    base_correct = base_prediction == labels
    assert abs(float(base_correct.mean())-record['best']['acc']) < 1e-12
    base_nll = numpy_nll(baseline, labels)
    full = cooperative_operator(flow.values, flow.p.edge, flow.weights, *flow.maps)
    own = composed_self_part(flow.values, flow.p.edge, flow.weights, *flow.maps, constant_maps=True)
    off_patient = (full-own).reshape(flow.m, flow.m, flow.count, flow.h)
    full_error = assert_close(flow.collapse(full), flow.delta)
    own_error = assert_close(flow.collapse(own), flow.own)
    foreign_error = assert_close(off_patient.mean(0).permute(1, 0, 2), flow.neighbor)
    # Removing all 36 off-patient terms collectively must recover complete self.
    collective = flow.delta.clone()
    for relation in range(flow.m):
        for content in range(flow.m):
            collective[:, content] -= off_patient[relation, content]/flow.m
    collective_error = assert_close(collective, flow.own)
    reconstructed = overridden_logits(model, x, flow.restore(flow.delta))[test]
    reconstruction_error = assert_close(reconstructed, flow.logits[test])
    assert torch.equal(reconstructed.argmax(-1), flow.logits[test].argmax(-1))
    collective_tokens_error = assert_close(flow.restore(collective), flow.restore(flow.own))
    rows, predictions = [], []
    for relation, relation_name in enumerate(dd['Modal_Name']):
        for content, content_name in enumerate(dd['Modal_Name']):
            changed_delta = flow.delta.clone()
            changed_delta[:, content] -= off_patient[relation, content]/flow.m
            logits = overridden_logits(model, x, flow.restore(changed_delta))[test].cpu().numpy()
            assert np.isfinite(logits).all()
            predicted = logits.argmax(-1)
            correct = predicted == labels
            corrected = int((~base_correct & correct).sum())
            harmed = int((base_correct & ~correct).sum())
            delta_acc = 100*(float(correct.mean())-float(base_correct.mean()))
            assert abs(delta_acc-100*(corrected-harmed)/len(labels)) < 1e-10
            ablated_nll = numpy_nll(logits, labels)
            rows.append(dict(fold=fold, selected_epoch=record['best']['epoch'],
                relation_index=relation, content_index=content,
                relation_name=relation_name, content_name=content_name,
                patients=len(labels), baseline_nll=base_nll, ablated_nll=ablated_nll,
                delta_nll=ablated_nll-base_nll, baseline_acc=float(base_correct.mean()),
                ablated_acc=float(correct.mean()), delta_acc_pp=delta_acc,
                changed=int((predicted != base_prediction).sum()), corrected=corrected, harmed=harmed))
            predictions.append(logits)
    assert model_state(model) == original_state, 'Diagnostic mutated model, buffers, flags or hooks'
    assert_rng_equal(original_rng, rng_state())
    folder = OUT/f'fold_{fold:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/'diagnostic_predictions.npz', baseline_logits=baseline,
        labels=labels, intervention_logits=np.stack(predictions),
        relation_index=np.asarray([row['relation_index'] for row in rows]),
        content_index=np.asarray([row['content_index'] for row in rows]))
    write_csv(folder/'metrics.csv', rows)
    audit = dict(fold=fold, seed=0, selected_epoch=record['best']['epoch'],
        patients=len(labels), saved_checkpoint_sha256=sha(source/'best.pt'),
        saved_prediction_max_error=saved_error, saved_prediction_argmax_exact=True,
        unmodified_reconstruction_max_error=reconstruction_error, reconstruction_argmax_exact=True,
        full_path_sum_max_error=full_error, self_path_sum_max_error=own_error,
        off_patient_sum_max_error=foreign_error, collective_self_max_error=collective_error,
        collective_self_tokens_max_error=collective_tokens_error,
        original_model_state=original_state, unchanged_model_state=True, unchanged_rng=True,
        unchanged_topology_weights_maps=True, normalization_divisor=flow.m,
        numeric_files_sha256={name: sha(folder/name) for name in ('metrics.csv', 'diagnostic_predictions.npz')})
    dump(folder/'complete.json', audit)
    del model, flow, full, own, off_patient, checkpoint
    torch.cuda.empty_cache()
    return rows, audit


def aggregate(all_rows, modalities):
    result = []
    for relation, relation_name in enumerate(modalities):
        for content, content_name in enumerate(modalities):
            rows = [row for row in all_rows if row['relation_index'] == relation and row['content_index'] == content]
            assert sorted(row['fold'] for row in rows) == list(FOLDS)
            nll = np.asarray([row['delta_nll'] for row in rows])
            acc = np.asarray([row['delta_acc_pp'] for row in rows])
            result.append(dict(relation_index=relation, content_index=content,
                relation_name=relation_name, content_name=content_name,
                mean_delta_nll=float(nll.mean()), sd_delta_nll=float(nll.std(ddof=1)),
                mean_delta_acc_pp=float(acc.mean()), sd_delta_acc_pp=float(acc.std(ddof=1)),
                total_changed=sum(row['changed'] for row in rows),
                total_corrected=sum(row['corrected'] for row in rows),
                total_harmed=sum(row['harmed'] for row in rows),
                positive_nll_folds=int((nll > NLL_SIGN_EPSILON).sum()),
                negative_nll_folds=int((nll < -NLL_SIGN_EPSILON).sum()),
                near_zero_nll_folds=int((np.abs(nll) <= NLL_SIGN_EPSILON).sum()),
                positive_acc_folds=int((acc > 1e-10).sum()),
                negative_acc_folds=int((acc < -1e-10).sum()),
                zero_acc_folds=int((np.abs(acc) <= 1e-10).sum())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {OUT}; use --resume for verified continuation')
    cfg, dd, data, graphs, paths = setup()
    original, inputs = validate_input(cfg, dd, data, graphs, paths)
    sources = local_source_closure()
    proposed = dict(experiment='Per-path off-patient dependence in original C best checkpoints',
        seed=0, folds=list(FOLDS), modalities=dd['Modal_Name'], samples=dd['Sample_Num'],
        relation_content_paths=36, planned_interventions=360, epochs_trained=0,
        checkpoint_selection=original['checkpoint_selection'],
        candidate_hash=original['candidate_hash'], input_sha256=inputs,
        code_sha256={name: sha(ROOT/name) for name in sources},
        operator='Freeze full L_receive @ L_send. Retain its entire block diagonal, including two-hop return paths; remove one (relation,content) off-patient block contribution before the original mean/6, GELU, output projection, residual and LayerNorm.',
        scoring='Original held-out fold rows, equal-fold averages. Delta=intervened minus unmodified. CPU float64 unweighted classification NLL for both; not the training loss.',
        positive_nll='Removing this path worsens checkpoint NLL; does not imply causal/clinical benefit.',
        positive_acc='Removing this path improves checkpoint ACC.',
        sd='Sample standard deviation across ten folds (ddof=1), not a standard error or confidence interval.',
        sign_epsilon_nll=NLL_SIGN_EPSILON,
        limits='Fixed TEST-selected checkpoints, one training seed, transductive cohort. Entries are nonadditive fixed-weight dependence, not retrained ablations, causal effects or clinical explanations.',
        privacy='Numerical files contain fold-local prediction order only; no original source-row or patient identifiers exported.',
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(cfg.Device)))
    if OUT.exists():
        plan = read(OUT/'PLAN.json')
        assert plan == proposed, 'Inputs or sources changed since this diagnostic began'
        verify_manifest(plan)
        if (OUT/'COMPLETE.json').exists():
            complete = read(OUT/'COMPLETE.json')
            for name, digest in complete['output_sha256'].items():
                assert sha(OUT/name) == digest, name
            print(json.dumps(dict(state='already_complete', directory=str(OUT))), flush=True)
            return
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        plan = proposed
        dump(OUT/'PLAN.json', plan)
        for name in sources:
            path = OUT/'source_snapshot'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    try:
        regression = dense_self_regression()
        dump(OUT/'self_diagonal_regression.json', regression)
        rows, audits = [], []
        for fold in FOLDS:
            folder = OUT/f'fold_{fold:02d}'
            if (folder/'complete.json').exists():
                audit = read(folder/'complete.json')
                for name, digest in audit['numeric_files_sha256'].items():
                    assert sha(folder/name) == digest, name
                with (folder/'metrics.csv').open(encoding='utf-8-sig') as stream:
                    fold_rows = list(csv.DictReader(stream))
                integer = ('fold','selected_epoch','relation_index','content_index','patients','changed','corrected','harmed')
                text_fields = ('relation_name','content_name')
                for row in fold_rows:
                    for key in row:
                        row[key] = row[key] if key in text_fields else int(row[key]) if key in integer else float(row[key])
            else:
                fold_rows, audit = run_fold(cfg, dd, data, graphs, fold)
            assert len(fold_rows) == 36 and all(row['fold'] == fold for row in fold_rows)
            rows.extend(fold_rows); audits.append(audit)
            status = dict(state='running', completed_folds=fold, planned_folds=10,
                completed_interventions=len(rows), seconds=time.perf_counter()-started, updated_at=timestamp())
            dump(OUT/'status.json', status)
            print(json.dumps(status), flush=True)
        verify_manifest(plan)
        assert len(rows) == 360 and sum(row['patients'] for row in rows if row['relation_index']==row['content_index']==0) == 598
        summary = aggregate(rows, dd['Modal_Name'])
        write_csv(OUT/'fold_metrics.csv', rows)
        write_csv(OUT/'aggregate.csv', summary)
        matrix_keys = ('mean_delta_nll','sd_delta_nll','mean_delta_acc_pp','sd_delta_acc_pp',
                       'total_changed','total_corrected','total_harmed','positive_nll_folds')
        matrices = {key: [[next(row[key] for row in summary if row['relation_index']==r and row['content_index']==c)
                          for c in range(6)] for r in range(6)] for key in matrix_keys}
        dump(OUT/'aggregate.json', dict(modalities=dd['Modal_Name'],
            axes=dict(rows='relation_source', columns='content_modality'), matrices=matrices,
            fold_aggregation='Equal-fold mean; SD ddof=1', nll_sign_epsilon=NLL_SIGN_EPSILON,
            delta_definition='intervened minus original; positive NLL=worse, negative ACC=worse',
            limits=plan['limits'], rows=summary))
        dump(OUT/'baseline_audit.json', dict(passed=True, checkpoints=10,
            verified_interventions=360, self_diagonal_regression=regression, records=audits))
        outputs = [path for path in OUT.rglob('*') if path.is_file() and
                   'source_snapshot' not in path.parts and path.name not in ('RUNNING.lock','status.json','COMPLETE.json')]
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), elapsed_seconds=time.perf_counter()-started,
            passed=True, checkpoints=10, interventions=360, no_training=True,
            output_sha256={str(path.relative_to(OUT)).replace('\\','/'): sha(path) for path in outputs}))
        dump(OUT/'status.json', dict(state='complete', completed_folds=10, completed_interventions=360, updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
