"""Train C without current-content affinity under the retained seed0 protocol."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (base_setup, read, state_hash, rng_state,
    assert_rng_equal, metrics, verify_manifest, verify_reference, save_predictions)
from Model.no_content_similarity_control import NoContentSimilarityControlModel
from Model.relation_sheaf import RelationSheafModel, conformal_maps
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_no_content_similarity_control import summarize

OUT = ROOT/'Result/no_content_similarity_seed0_20261011'
C_REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODE = 'NO_CONTENT_SIMILARITY'
FOLDS = tuple(range(1, 11))
PARAMETERS = {'C': 895096, MODE: 893560}
SOURCES = ('Model/no_content_similarity_control.py', 'no_content_similarity_seed0_experiment.py',
    'analyze_no_content_similarity_control.py', 'tests/test_no_content_similarity_control.py',
    'alternating_cooperative_experiment.py', 'analyze_alternating_cooperative.py',
    'analyze_joint_evidence_routing.py', 'Model/layers.py', 'Model/__init__.py',
    'Loss/__init__.py', 'Utils/__init__.py')


def make_model(dd, cfg, graphs):
    return NoContentSimilarityControlModel(dd, cfg, graphs, MODE, training_seed=0).to(cfg.Device)


def verify_artifacts(plan):
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_manifest'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


def map_diagnostics(model):
    p = model.propagation
    assert not hasattr(p, 'edge_query')
    assert p.mode == 'content' and p.edge_similarity.shape == p.edge_bias.shape == (6, 6)
    with torch.no_grad():
        raw = p.conditioner(p.content_id(torch.arange(p.modalities, device=p.content_id.weight.device)))
        send, receive, cs, ct = conformal_maps(raw)
        weight = torch.sigmoid(p.similarity * p.edge_similarity.flatten()[p.edge_group]
                               + p.edge_bias.flatten()[p.edge_group])
    assert torch.isfinite(weight).all() and torch.isfinite(raw).all()
    assert int(p.edge_group.max()) + 1 == p.modalities**2
    return dict(current_content_similarity_removed=True, edge_query_absent=True,
        coefficient_shape=list(p.edge_similarity.shape), active_groups=p.modalities**2,
        raw_map_parameters=raw.cpu().tolist(), send_scale=cs.cpu().tolist(),
        receive_scale=ct.cpu().tolist(), weight_min=float(weight.min()), weight_max=float(weight.max()),
        content_indexed_coefficients_retained=True, content_specific_maps_retained=True)


def assert_groups(model, cfg):
    groups = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
    expected = copy.deepcopy(read(C_REFERENCE/'fold_01/optimizer_groups.json'))
    expected[1]['parameters'] -= 1536
    expected[1]['names'].remove('propagation.edge_query.weight')
    assert groups == expected
    assert groups[0]['parameters'] == 874091 and groups[1]['parameters'] == 19469
    return groups


def reference_checks(cfg, dd, data, graphs, original):
    manifest, audit = verify_reference(cfg, dd, data, graphs)
    SET_Random(0)
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    original_state_hash = state_hash(model)
    expected_groups = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
    records = []
    for fold in FOLDS:
        folder = C_REFERENCE/f'fold_{fold:02d}'
        record = read(folder/'complete.json')
        assert read(folder/'optimizer_groups.json') == expected_groups
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert (checkpoint['mode'], checkpoint['fold'], checkpoint['seed']) == ('content', fold, 0)
        # Original C uses its historical checkpoint schema, not newer runners'.
        assert checkpoint['protocol'] == 'original test-selected'
        assert checkpoint['best'] == record['best'] == read(folder/'best.json')
        with (folder/'epochs.csv').open() as stream: curve = list(csv.DictReader(stream))
        assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in curve]).all()
        assert record['last']['epoch'] == 1000
        test = data['Mask'][fold-1][1]
        for stage, filename in (('best','predictions.npz'),('last','last_predictions.npz')):
            with np.load(folder/filename, allow_pickle=False) as saved:
                logits = torch.from_numpy(saved['logits']).to(cfg.Device)
            actual = metrics(logits, data['Label'][test])
            for key in ('acc','auc','f1','macro_f1','per_class_recall','confusion_matrix'):
                np.testing.assert_allclose(actual[key], record[stage][key], atol=1e-12, rtol=0)
        records.append(dict(fold=fold, protocol_and_optimizer_verified=True))
    del model
    torch.cuda.empty_cache()
    audit.update(initial_model_sha256=original_state_hash, additional_provenance_checks=records)
    return manifest, audit


def smoke(cfg, dd, data, graphs):
    SET_Random(0)
    original = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    native_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    native_rng = rng_state()
    SET_Random(0)
    model = make_model(dd, cfg, graphs)
    assert_rng_equal(native_rng, rng_state())
    removed = 'propagation.edge_query.weight'
    assert list(dict(model.named_parameters())) == [n for n,_ in original.named_parameters() if n != removed]
    assert set(model.state_dict()) == set(native_state) - {removed}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, native_state[name], atol=0, rtol=0)
    for name in ('edge', 'similarity', 'edge_group'):
        torch.testing.assert_close(getattr(model.propagation, name), getattr(original.propagation, name), atol=0, rtol=0)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[MODE]
    initial = dict(state_sha256=state_hash(model), parameters=PARAMETERS[MODE])
    groups = assert_groups(model, cfg)
    maps = map_diagnostics(model)
    with torch.no_grad():
        original.propagation.edge_query.weight.zero_()
    for training in (False, True):
        SET_Random(27); original.train(training)
        with torch.no_grad(): left = original(data['Feature'])
        stream = rng_state()
        SET_Random(27); model.train(training)
        with torch.no_grad(): right = model(data['Feature'])
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
        assert_rng_equal(stream, rng_state())
    optimizer = relation_optimizer(model, cfg, 0.)
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    before = model.propagation.content_id.weight.detach().clone()
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for _ in range(8):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, embeddings, auxiliary = model(data['Feature'])
        loss = criterion(logits, data['Label'], data['Mask'][0][0], embeddings, auxiliary)
        assert torch.isfinite(loss); loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None: assert torch.isfinite(parameter.grad).all(), name
            if name.startswith('propagation.'): assert parameter.grad is not None, name
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step(); model.eval()
        with torch.no_grad(): assert torch.isfinite(model(data['Feature'])[0]).all()
    assert not torch.equal(before, model.propagation.content_id.weight)
    torch.cuda.synchronize()
    check = dict(steps=8, seconds_per_epoch=(time.perf_counter()-started)/8,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        test_metrics_inspected=False, initial_maps=maps, trained_maps=map_diagnostics(model))
    del model, original, native_state, optimizer, criterion, left, right, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache()
    return initial, groups, check


def prepare():
    cfg, dd, data, graphs, hashes, original = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json'); verify_artifacts(plan)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == original['candidate_hash']
        assert plan['modes'] == [MODE] and plan['parameters'] == PARAMETERS
        return cfg, dd, data, graphs, plan
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_no_content_similarity_control.py', '-v'], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8')
    if test.returncode or 'Ran 0 tests' in test.stderr: raise RuntimeError(test.stdout+test.stderr)
    manifest, audit = reference_checks(cfg, dd, data, graphs, original)
    initial, groups, check = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys((*original['code_sha256'], *SOURCES)))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr, encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((C_REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    plan = dict(experiment='C without current propagation-content affinity; retrained matched control',
        created_at=timestamp(), training_seed=0, split_seed=0, modes=[MODE], folds=list(FOLDS), epochs=1000,
        planned_runs=10, reused_runs=10, parameters=PARAMETERS, initialization={MODE: initial},
        optimizer_groups={MODE: groups}, class_names=dd['Class_Names'], modalities=dd['Modal_Name'],
        samples=dd['Sample_Num'], original_config=original['original_config'], data_sha256=hashes,
        candidate_hash=original['candidate_hash'], checkpoint_selection=original['checkpoint_selection'],
        splits_artifact_sha256=sha(OUT/'data_splits_and_graphs.npz'), transductive=True,
        optimizer='Original Adam/cosine; graph-only decay 0, other decay 0.0007',
        architecture='Original C: six graphs and 36 relation-content paths, content-specific S/T, two native cooperative passes, D+I normalization, relation averaging, residual and readout unchanged. Remove only current-content cosine affinity and its dedicated 96-to-16 projection.',
        edge_weight_formula='sigmoid(a[relation,content]*raw_similarity+b[relation,content])',
        retained_content_dependence='a/b remain indexed by relation and content; S/T remain content-specific. This is not removal of all content conditioning.',
        shared_initialization='Construct original C first, then delete edge_query; all common parameters and constructor RNG identical.',
        limitation='Removes 1536 active parameters; shared initialization is equal but full-model initial functions differ. Single seed, transductive historical TEST-selected exploratory comparison.',
        scope='Only NO_CONTENT_SIMILARITY seed0 ten-fold; no tuning or protocol change.',
        common_initialization_and_rng_verified=True, smoke=check, reference_audit=audit,
        code_sha256={name: sha(ROOT/name) for name in sources}, reference_manifest=manifest,
        environment=dict(executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target = OUT/'source_snapshot'/source; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'artifacts_manifest.json', {name: sha(OUT/name) for name in
        ('PLAN.json', 'implementation_checks.log', 'data_splits_and_graphs.npz')})
    verify_artifacts(plan); summarize(OUT, plan)
    dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=10))
    return cfg, dd, data, graphs, plan


def train_fold(cfg, dd, data, graphs, plan, fold):
    folder = OUT/f'seed_0/{MODE}/fold_{fold:02d}'
    if (folder/'complete.json').exists(): return
    if folder.exists(): raise FileExistsError(f'Partial fold must be preserved before a clean retry: {folder}')
    verify_artifacts(plan); folder.mkdir(parents=True, exist_ok=False)
    SET_Random(0); model = make_model(dd, cfg, graphs)
    assert state_hash(model) == plan['initialization'][MODE]['state_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    assert optimizer_manifest(model, optimizer) == plan['optimizer_groups'][MODE]
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']; fit, test = data['Mask'][fold-1]
    best_correct = -1; history = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x); loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step(); model.eval()
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
                if not torch.isfinite(logits).all() or not torch.isfinite(test_loss):
                    raise FloatingPointError('Nonfinite evaluation')
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
            for stage in (['best'] if correct > best_correct else []) + (['last'] if epoch == 1000 else []):
                score = dict(epoch=epoch, **metrics(logits[test], y[test]))
                if stage == 'best': best_correct = correct; best = score
                else: last = score
                temp = folder/f'{stage}.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, mode=MODE,
                    fold=fold, seed=0, protocol_sha256=sha(OUT/'PLAN.json')), temp)
                temp.replace(folder/f'{stage}.pt')
                save_predictions(folder/('predictions.npz' if stage == 'best' else 'last_predictions.npz'),
                                 logits, y, test, dd['Index'])
                dump(folder/f'{stage}.json', score)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=MODE,
                    fold=fold, seed=0, epochs=1000, planned_runs=10, completed_runs=fold-1, best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                if epoch == 1 or epoch % 100 == 0: print(json.dumps(status), flush=True)
    record = dict(mode=MODE, fold=fold, seed=0, parameters=PARAMETERS[MODE], best=best, last=last,
        initial_model_sha256=plan['initialization'][MODE]['state_sha256'], mean_last100_acc=float(np.mean(history[-100:])),
        best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
        completed_at=timestamp(), seconds=time.perf_counter()-started,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20, final_maps=map_diagnostics(model))
    dump(folder/'complete.json', record)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache(); summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_artifacts(plan); rows = []
    model = make_model(dd, cfg, graphs).eval()
    for fold in FOLDS:
        folder = OUT/f'seed_0/{MODE}/fold_{fold:02d}'; record = read(folder/'complete.json')
        assert (record['mode'], record['fold'], record['seed']) == (MODE, fold, 0)
        assert record['parameters'] == PARAMETERS[MODE]
        assert record['initial_model_sha256'] == plan['initialization'][MODE]['state_sha256']
        assert read(folder/'optimizer_groups.json') == plan['optimizer_groups'][MODE]
        with (folder/'epochs.csv').open() as stream: curve = list(csv.DictReader(stream))
        with (C_REFERENCE/f'fold_{fold:02d}/epochs.csv').open() as stream: old = list(csv.DictReader(stream))
        assert [int(r['epoch']) for r in curve] == list(range(1, 1001))
        assert np.isfinite([[float(r[k]) for k in ('train_loss', 'test_loss', 'test_acc', 'lr')] for r in curve]).all()
        assert int(np.argmax([float(r['test_acc']) for r in curve]))+1 == record['best']['epoch']
        assert record['last']['epoch'] == 1000
        np.testing.assert_array_equal([float(r['lr']) for r in curve], [float(r['lr']) for r in old])
        test = data['Mask'][fold-1][1]
        for stage, prediction in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
            cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
            assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
            assert (cp['mode'], cp['fold'], cp['seed']) == (MODE, fold, 0)
            assert cp['metrics'] == record[stage] == read(folder/f'{stage}.json')
            assert cp['epoch'] == record[stage]['epoch']
            assert abs(float(curve[cp['epoch']-1]['test_acc'])-record[stage]['acc']) < 1e-12
            model.load_state_dict(cp['state_dict'], strict=True)
            with torch.no_grad(): logits = model(data['Feature'])[0][test]
            with np.load(folder/prediction, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                np.testing.assert_allclose(saved['logits'], logits.cpu().numpy(), atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(saved['logits'].argmax(-1), logits.argmax(-1).cpu().numpy())
            actual = metrics(logits, data['Label'][test])
            for key in ('acc', 'auc', 'f1', 'macro_f1', 'per_class_recall', 'confusion_matrix'):
                np.testing.assert_allclose(actual[key], record[stage][key], atol=1e-12, rtol=0)
            for key in ('nll', 'brier'):
                np.testing.assert_allclose(actual[key], record[stage][key], atol=2e-5, rtol=2e-5)
            rows.append(dict(fold=fold, stage=stage, passed=True, maps=map_diagnostics(model)))
    result = dict(passed=True, verified_runs=10, verified_checkpoints=len(rows), records=rows)
    dump(OUT/'audit.json', result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', parameters=PARAMETERS, smoke=plan['smoke'])), flush=True); return
    if (OUT/'COMPLETE.json').exists(): print('Already complete.'); return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        for fold in FOLDS: train_fold(cfg, dd, data, graphs, plan, fold)
        audit = audit_completed(cfg, dd, data, graphs, plan); summary = summarize(OUT, plan)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT/'status.json', dict(state='complete', completed_runs=10, updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error))); raise
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
