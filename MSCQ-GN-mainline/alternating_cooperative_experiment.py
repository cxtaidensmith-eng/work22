"""Four fixed cooperative feedback arms under the retained C seed-0 protocol.

No tuning or automatic follow-up arms. Existing code and results are read-only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp, scores
from Model.relation_sheaf import RelationSheafModel
from Model.alternating_cooperative import AlternatingCooperativeModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_alternating_cooperative import summarize, write_report


OUT = ROOT / 'Result/alternating_cooperative_seed0_20260923'
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay'
MODES = ('N0', 'N1', 'A0', 'A1')
DIAGNOSTIC_EPOCHS = (1, 100, 500, 1000)
FOLDS = tuple(range(1, 11))
NEW_SOURCES = ('Model/alternating_cooperative.py', 'alternating_cooperative_experiment.py',
               'analyze_alternating_cooperative.py', 'tests/test_alternating_cooperative.py',
               'analyze_joint_evidence_routing.py', 'tests/test_modal_evidence.py', 'Model/modal_evidence.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def state_hash(model, names=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if names is None or name in names:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all())


def assert_rng_equal(left, right):
    assert torch.equal(left[0], right[0])
    assert len(left[1]) == len(right[1])
    assert all(torch.equal(a, b) for a, b in zip(left[1], right[1]))


def metrics(logits, labels):
    result = scores(logits, labels)
    target = F.one_hot(labels, logits.shape[-1]).float()
    result.update(nll=float(F.cross_entropy(logits, labels)),
                  brier=float((logits.softmax(-1) - target).square().sum(-1).mean()))
    return result


def make_model(dd, cfg, graphs, mode):
    return AlternatingCooperativeModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def verify_manifest(manifest):
    for name, digest in manifest.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen source or reference changed: {name}')


def verify_sources(plan):
    verify_manifest(plan['code_sha256'])


def base_setup():
    cfg, dd, data, graphs, paths = setup()
    old = read(REFERENCE / 'seed_0/PROTOCOL.json')
    verify_manifest(old['code_sha256'])
    for key, value in old['original_config'].items():
        assert getattr(cfg, key) == value, key
    hashes = {Path(p).name: sha(p) for p in paths}
    assert hashes == old['data_sha256']
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    assert graph_hash == old['candidate_hash']
    assert cfg.epochs == old['epochs'] == 1000 and len(data['Mask']) == 10
    assert old['training_seed'] == old['split_seed'] == 0
    assert old['graph_weight_decay_override'] == 0.
    if not torch.cuda.is_available() or torch.__version__ != old['torch']:
        raise RuntimeError('Use the retained tabgsl-gpu runtime')
    return cfg, dd, data, graphs, hashes, old


def verify_reference(cfg, dd, data, graphs):
    manifest = {}
    for filename in ('PLAN.json', 'seed_0/PROTOCOL.json', 'seed_0/splits_and_graphs.npz'):
        path = REFERENCE / filename
        manifest[str(path.relative_to(ROOT))] = sha(path)
    with np.load(REFERENCE/'seed_0/splits_and_graphs.npz') as saved:
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(saved['test_masks'], np.stack([t.cpu().numpy() for _, t in data['Mask']]))
        for i, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{i}'], graph['edge'].numpy())
            np.testing.assert_array_equal(saved[f'neighbors_{i}'], graph['neighbors'].numpy())
    SET_Random(0)
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    rows = []
    for fold in FOLDS:
        folder = REFERENCE / f'seed_0/content/fold_{fold:02d}'
        record = read(folder / 'complete.json')
        assert (record['mode'], record['fold'], record['seed']) == ('content', fold, 0)
        assert record['parameters'] == sum(p.numel() for p in model.parameters()) == 895096
        with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
            epochs = list(csv.DictReader(stream))
        assert [int(r['epoch']) for r in epochs] == list(range(1, 1001))
        assert int(np.argmax([float(r['test_acc']) for r in epochs])) + 1 == record['best']['epoch']
        checkpoint = torch.load(folder / 'best.pt', map_location=cfg.Device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        with torch.no_grad():
            replay = model(data['Feature'])[0]
        test = data['Mask'][fold - 1][1]
        error = None
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            with np.load(folder / filename) as saved:
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                z = torch.from_numpy(saved['logits']).to(cfg.Device)
                actual = scores(z, data['Label'][test])
                for key in ('acc', 'auc', 'f1', 'macro_f1'):
                    assert abs(actual[key] - record[stage][key]) < 1e-12
                if stage == 'best':
                    np.testing.assert_allclose(saved['logits'], replay[test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                    error = float(np.max(np.abs(saved['logits'] - replay[test].cpu().numpy())))
        for filename in ('complete.json', 'best.json', 'epochs.csv', 'best.pt',
                         'predictions.npz', 'last_predictions.npz', 'optimizer_groups.json'):
            path = folder / filename
            manifest[str(path.relative_to(ROOT))] = sha(path)
        rows.append(dict(fold=fold, best_replay_max_abs_error=error, last_saved_prediction_metrics_verified=True))
    return manifest, dict(passed=True, runs=10, replayed_checkpoints=10,
                         caveat='C has no last.pt; last metrics checked against saved predictions only', records=rows)


def smoke(cfg, dd, data, graphs):
    models, streams = {}, {}
    for mode in ('original', 'C', *MODES):
        SET_Random(0)
        models[mode] = (RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
                        if mode == 'original' else make_model(dd, cfg, graphs, mode))
        streams[mode] = rng_state()
    hashes = {mode: state_hash(model) for mode, model in models.items()}
    assert len(set(hashes.values())) == 1
    assert all(sum(p.numel() for p in model.parameters()) == 895096 for model in models.values())
    outputs, forward_rng = {}, {}
    for mode, model in models.items():
        assert_rng_equal(streams['original'], streams[mode])
        SET_Random(3701)
        model.train()
        with torch.no_grad():
            outputs[mode] = model(data['Feature'])
            assert torch.isfinite(outputs[mode][0]).all()
        forward_rng[mode] = rng_state()
        assert_rng_equal(forward_rng['original'], forward_rng[mode])
    torch.testing.assert_close(outputs['original'], outputs['C'], atol=2e-6, rtol=2e-6)
    single_call_max_error = float((outputs['original'][0]-outputs['C'][0]).abs().max())
    initial_hash = hashes['original']
    del models, outputs, model
    torch.cuda.empty_cache()
    checks = []
    fit = data['Mask'][0][0]
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode)
        criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
        optimizer = relation_optimizer(model, cfg, 0.)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step in range(12):
            model.capture_diagnostics = False
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(data['Feature'])
            loss = criterion(output[0], data['Label'], fit, output[1], output[2])
            assert torch.isfinite(loss)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (mode, name)
                if name.startswith('propagation.'):
                    assert p.grad is not None, (mode, name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            model.capture_diagnostics = step == 11
            with torch.no_grad():
                assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        trace = model.last_diagnostics
        assert trace['reused_same_weight_tensor'] == mode.endswith('0')
        assert (trace['mean_absolute_edge_weight_change'] == 0) == mode.endswith('0')
        checks.append(dict(mode=mode, parameters=sum(p.numel() for p in model.parameters()),
                           seconds_per_epoch=(time.perf_counter()-started)/12,
                           peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                           optimizer_groups=optimizer_manifest(model, optimizer), diagnostics=trace))
        del model, optimizer, output, loss
        torch.cuda.empty_cache()
    assert all(c['optimizer_groups'] == checks[0]['optimizer_groups'] for c in checks)
    original_groups = read(REFERENCE/'seed_0/content/fold_01/optimizer_groups.json')
    assert checks[0]['optimizer_groups'] == original_groups
    return dict(passed=True, test_metrics_inspected=False, full_state_hashes=hashes,
                matched_parameters_initialization_and_rng=True, matched_input_noise_dropout=True,
                single_call_train_forward_max_abs_error=single_call_max_error,
                initial_model_sha256=initial_hash, runs=checks)


def prepare():
    cfg, dd, data, graphs, hashes, old = base_setup()
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        verify_sources(plan); verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == old['candidate_hash']
        assert plan['modes'] == list(MODES) and plan['seeds'] == [0] and plan['planned_runs'] == 40
        return cfg, dd, data, graphs, plan
    test = subprocess.run([sys.executable, 'tests/test_alternating_cooperative.py'], cwd=ROOT,
                          capture_output=True, text=True, encoding='utf-8', check=True)
    manifest, reference_audit = verify_reference(cfg, dd, data, graphs)
    checks = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys(tuple(old['code_sha256']) + NEW_SOURCES))
    plan = dict(experiment='Alternating modal interaction and shared cooperative propagation: fixed 2x2',
        seeds=[0], split_seed=0, folds=list(FOLDS), modes=list(MODES), epochs=1000,
        planned_runs=40, reused_runs=10, primary_candidate='A1', class_names=dd['Class_Names'],
        modalities=dd['Modal_Name'], samples=dd['Sample_Num'], original_config=old['original_config'],
        checkpoint_selection=old['checkpoint_selection'], original_protocol_sha256=sha(REFERENCE/'seed_0/PROTOCOL.json'),
        primary='A1 vs C: equal-fold first-best TEST ACC and paired net corrected patients',
        secondary='Macro F1, class recalls, raw-logit AUC, epoch 1000 and last-100 mean; no alternative selection',
        comparisons=['A1-C','A1-N1','A1-A0','N1-N0','A0-N0','N0-C','N1-C','A0-C'],
        interaction='(A1-A0)-(N1-N0), computed within fold before averaging',
        decision='Complete all 40 runs. No interim tuning, extra arms, losses or seeds. A1 predeclared; net 1-2 patients is weak evidence.',
        architecture='C content propagation shared twice; all five original F blocks once; unchanged encoder/readout/global branch',
        schedules=dict(C='P F1 F2 F3 F4 F5', N0='P P F1 F2 F3 F4 F5; reuse W0',
                       N1='P P F1 F2 F3 F4 F5; refresh W1', A0='P F1 F2 P F3 F4 F5; reuse W0',
                       A1='P F1 F2 P F3 F4 F5; refresh W1'),
        reuse='Within forward only, never detach; second values always use current tokens',
        maps='Retained C content-ID maps; shared parameters, no new patient-state conditioner',
        candidate_hash=old['candidate_hash'], candidate_k=10,
        support='Original six symmetric cosine top-10 reverse unions; fixed supports and raw cosines; refreshed continuous weights only',
        operator='Retained C residual and output LN once per call; two diffusion passes within each P, using identical W/S/T',
        parameters=895096, computational_caveat='New arms have four diffusion passes; C has two. Refresh arms compute G twice.',
        objective='Unchanged MSCQGNLoss: weighted smoothed final CE + OVR auxiliary CE + original all-cohort orthogonality',
        class_weights=dd['Label_Weight'].cpu().tolist(),
        optimizer='Unchanged relation_optimizer(..., 0): propagation except output_norm decay=0; remaining decay=.0007',
        training='From scratch, seed reset each arm/fold; original Adam, scheduler, clip and constructor RNG',
        run_order='For fold 1..10: N0, N1, A0, A1',
        diagnostics=dict(epochs=list(DIAGNOSTIC_EPOCHS), checkpoint_stages=['best','last'],
            values=['Each P update/input norm, before output LN','Mean absolute W1-W0'],
            interpretation='Noise-free eval only; descriptive, not causal or a checkpoint selection criterion'),
        transductive=True, preprocessing='Original processed CSV, row order, masks, cohort weights, modal gate/noise priors',
        reference_manifest=manifest, data_sha256=hashes, code_sha256={p:sha(ROOT/p) for p in sources},
        unit_tests=dict(passed=True, output=test.stdout+test.stderr),
        smoke=checks, environment=dict(executable=sys.executable, python=sys.version, torch=torch.__version__,
                                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        interpretation='Single-seed original test-selected exploration, not an independent test or proof of novelty')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT/'PLAN.json', plan); dump(OUT/'reference_audit.json', reference_audit)
    for source in sources:
        destination = OUT/'source_snapshot'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/source).read_bytes())
    arrays = dict(source_row=np.asarray(dd['Index']), labels=data['Label'].cpu().numpy(),
                  test_masks=np.stack([t.cpu().numpy() for _, t in data['Mask']]))
    for i, graph in enumerate(graphs):
        for key in ('edge','similarity','neighbors','degree'):
            arrays[f'{key}_{i}'] = graph[key].numpy()
    np.savez_compressed(OUT/'data_splits_and_graphs.npz', **arrays)
    dump(OUT/'artifacts_manifest.json', {'data_splits_and_graphs.npz': sha(OUT/'data_splits_and_graphs.npz')})
    summarize(OUT, plan)
    dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=40))
    return cfg, dd, data, graphs, plan


def save_checkpoint(folder, stage, model, epoch, score, fold, mode, plan):
    temp = folder/f'{stage}.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, seed=0, fold=fold,
                    mode=mode, protocol_sha256=sha(OUT/'PLAN.json')), temp)
    temp.replace(folder/f'{stage}.pt')


def save_predictions(path, logits, labels, test, source_rows):
    np.savez_compressed(path, logits=logits[test].detach().cpu().numpy(), labels=labels[test].cpu().numpy(),
                        row_index=np.asarray(source_rows)[test.cpu().numpy()])


def train_arm(cfg, dd, data, graphs, plan, fold, mode):
    folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists():
        return
    verify_sources(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(0)
    model = make_model(dd, cfg, graphs, mode)
    initial_hash = state_hash(model)
    assert initial_hash == plan['smoke']['initial_model_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold-1]
    best_correct, best, last, history = -1, None, None, []
    diagnostics = {}
    parameters = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','train_loss','test_loss','test_acc','lr','seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.capture_diagnostics = False
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} fold {fold} epoch {epoch}: nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            model.capture_diagnostics = epoch in DIAGNOSTIC_EPOCHS
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
            if model.capture_diagnostics:
                diagnostics[str(epoch)] = model.last_diagnostics
                dump(folder/'diagnostics.json', diagnostics)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                       test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'],
                       seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'best', model, epoch, best, fold, mode, plan)
                save_predictions(folder/'predictions.npz', logits, y, test, dd['Index'])
                dump(folder/'best.json', best)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                              fold=fold, seed=0, epochs=1000, planned_runs=40,
                              parameters=parameters, best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                print(json.dumps(status), flush=True)
            if epoch == 1000:
                last = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'last', model, epoch, last, fold, mode, plan)
                save_predictions(folder/'last_predictions.npz', logits, y, test, dd['Index'])
    record = dict(mode=mode, fold=fold, seed=0, parameters=parameters, best=best, last=last,
                  initial_model_sha256=initial_hash, mean_last100_acc=float(np.mean(history[-100:])),
                  best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
                  completed_at=timestamp(), seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache()
    summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_sources(plan); verify_manifest(plan['reference_manifest'])
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest
    rows = []
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode).eval()
        for fold in FOLDS:
            folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
            record = read(folder/'complete.json')
            assert (record['mode'], record['seed'], record['fold']) == (mode, 0, fold)
            assert record['initial_model_sha256'] == plan['smoke']['initial_model_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                epochs = list(csv.DictReader(stream))
            with (REFERENCE/f'seed_0/content/fold_{fold:02d}/epochs.csv').open(encoding='utf-8') as stream:
                old_epochs = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1,1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1 == record['best']['epoch']
            np.testing.assert_array_equal([float(r['lr']) for r in epochs], [float(r['lr']) for r in old_epochs])
            test = data['Mask'][fold-1][1]
            for stage, filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (cp['mode'],cp['seed'],cp['fold']) == (mode,0,fold)
                assert cp['epoch'] == record[stage]['epoch']
                assert cp['metrics'] == record[stage]
                model.load_state_dict(cp['state_dict'], strict=True)
                model.capture_diagnostics = True
                with torch.no_grad():
                    logits = model(data['Feature'])[0]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'],logits[test].cpu().numpy(),atol=2e-5,rtol=2e-5)
                    error = float(np.max(np.abs(saved['logits']-logits[test].cpu().numpy())))
                actual = metrics(logits[test],data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','nll','brier'):
                    assert abs(actual[key]-record[stage][key]) < 2e-5
                np.testing.assert_array_equal(actual['confusion_matrix'], record[stage]['confusion_matrix'])
                np.testing.assert_allclose(actual['per_class_recall'], record[stage]['per_class_recall'], atol=1e-12)
                diagnostic = model.last_diagnostics
                assert diagnostic['reused_same_weight_tensor'] == mode.endswith('0')
                if mode.endswith('0'):
                    assert diagnostic['mean_absolute_edge_weight_change'] == 0.
                rows.append(dict(mode=mode,fold=fold,stage=stage,max_abs_replay_error=error,diagnostics=diagnostic))
            diagnostics = read(folder/'diagnostics.json')
            assert sorted(map(int, diagnostics)) == list(DIAGNOSTIC_EPOCHS)
            assert sum(p.numel() for p in model.parameters()) == record['parameters'] == 895096
            for value in diagnostics.values():
                assert value['reused_same_weight_tensor'] == mode.endswith('0')
                assert np.isfinite(value['mean_absolute_edge_weight_change'])
                for call in ('first','second'):
                    assert np.isfinite(value[call]['graph_update_over_input_norm'])
            optimizer = relation_optimizer(model, cfg, 0.)
            assert optimizer_manifest(model, optimizer) == read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result = dict(passed=True, verified_runs=40, verified_checkpoints=80, records=rows,
                  same_original_lr_schedule=True, same_original_source_and_reference_hashes=True)
    dump(OUT/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=40, smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    if (OUT/'COMPLETE.json').exists():
        verify_sources(plan); verify_manifest(plan['reference_manifest'])
        print('Already complete; existing results retained.', flush=True)
        return
    lock = OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES:
                train_arm(cfg,dd,data,graphs,plan,fold,mode)
        dump(OUT/'status.json',dict(state='auditing',pid=os.getpid(),updated_at=timestamp()))
        audit = audit_completed(cfg,dd,data,graphs,plan)
        summary = summarize(OUT,plan)
        report = write_report(OUT,plan,audit)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary,report=report))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp(),completed_runs=40))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',pid=os.getpid(),updated_at=timestamp(),error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
