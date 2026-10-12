"""C versus a shared learnable mapping generator; retained TAD3 seed0 protocol."""
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
from Model.relation_sheaf import RelationSheafModel, conformal_maps
from Model.shared_mapping_control import SharedMappingControlModel
from analyze_shared_mapping_control import summarize
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

OUT = ROOT/'Result/shared_mapping_seed0_20261003'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODES = ('SHARED_MAP',)
FOLDS = tuple(range(1, 11))
PARAMETERS = {'C': 895096, 'SHARED_MAP': 895016}
SOURCES = ('Model/shared_mapping_control.py', 'shared_mapping_seed0_experiment.py',
    'Model/layers.py', 'Model/__init__.py', 'Loss/__init__.py', 'Utils/__init__.py',
    'analyze_shared_mapping_control.py', 'tests/test_shared_mapping_control.py',
    'tests/test_modal_evidence.py', 'Model/modal_evidence.py',
    'alternating_cooperative_experiment.py', 'Model/alternating_cooperative.py',
    'analyze_alternating_cooperative.py', 'analyze_joint_evidence_routing.py')


def make_model(dd, cfg, graphs, mode='SHARED_MAP'):
    return SharedMappingControlModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def verify_artifacts(plan):
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_manifest'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


def assert_groups(model, cfg, mode):
    groups = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
    expected = copy.deepcopy(read(REFERENCE/'fold_01/optimizer_groups.json'))
    if mode == 'SHARED_MAP':
        expected[1]['parameters'] -= 80
    assert groups == expected
    return groups


def map_diagnostics(model, dd, cfg):
    p = model.propagation
    z = torch.zeros(dd['Sample_Num'], len(dd['Modal_Name']), cfg.Hidden_size, device=cfg.Device)
    with torch.no_grad():
        values = p.map_values(z)
        first = values[0, 0, 0]
        torch.testing.assert_close(values, first.expand_as(values), atol=0, rtol=0)
        send, receive, cs, ct = conformal_maps(first[None])
    return dict(shared_across_all_relations_modalities_and_patients=True,
        embedding_shape=list(p.content_id.weight.shape), generator_output=first.cpu().tolist(),
        send=send[0].cpu().tolist(), receive=receive[0].cpu().tolist(),
        send_scale=float(cs[0]), receive_scale=float(ct[0]))


def smoke(cfg, dd, data, graphs):
    SET_Random(0)
    native = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    native_state = {k: v.clone() for k, v in native.state_dict().items()}
    native_rng = rng_state()
    initial, checks, groups = {}, [], {}
    for mode in ('C', *MODES):
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode)
        assert_rng_equal(native_rng, rng_state())
        assert set(model.state_dict()) == set(native_state)
        for name, value in model.state_dict().items():
            expected = native_state[name]
            if mode == 'SHARED_MAP' and name == 'propagation.content_id.weight':
                # The native constructor creates these embeddings on CPU.
                expected = expected.cpu().mean(0, keepdim=True).to(value.device)
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[mode]
        initial[mode] = dict(state_sha256=state_hash(model), parameters=PARAMETERS[mode])
        groups[mode] = assert_groups(model, cfg, mode)
        if mode == 'C':
            for training in (False, True):
                SET_Random(27); native.train(training)
                with torch.no_grad(): left = native(data['Feature'])
                after = rng_state()
                SET_Random(27); model.train(training)
                with torch.no_grad(): right = model(data['Feature'])
                torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
                assert_rng_equal(after, rng_state())
            del left, right
        else:
            initial_maps = map_diagnostics(model, dd, cfg)
            optimizer = relation_optimizer(model, cfg, 0.)
            criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
            started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            before = model.propagation.content_id.weight.detach().clone()
            for _ in range(8):
                model.train(); optimizer.zero_grad(set_to_none=True)
                z, emb, aux = model(data['Feature'])
                loss = criterion(z, data['Label'], data['Mask'][0][0], emb, aux)
                assert torch.isfinite(loss); loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all(), name
                    if name.startswith('propagation.'): assert p.grad is not None, name
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step(); model.eval()
                with torch.no_grad(): assert torch.isfinite(model(data['Feature'])[0]).all()
            assert not torch.equal(before, model.propagation.content_id.weight)
            torch.cuda.synchronize()
            checks.append(dict(mode=mode, steps=8, seconds_per_epoch=(time.perf_counter()-started)/8,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                test_metrics_inspected=False, initial_maps=initial_maps,
                trained_maps=map_diagnostics(model, dd, cfg)))
            del optimizer, criterion, z, emb, aux, loss
        del model
    del native, native_state
    torch.cuda.empty_cache()
    return initial, groups, checks


def prepare():
    cfg, dd, data, graphs, hashes, original = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json'); verify_artifacts(plan)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == original['candidate_hash']
        assert plan['modes'] == list(MODES) and plan['folds'] == list(FOLDS)
        assert plan['parameters'] == PARAMETERS and plan['training_seed'] == 0
        return cfg, dd, data, graphs, plan
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_shared_mapping_control.py', '-v'], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8')
    if test.returncode: raise RuntimeError(test.stdout + test.stderr)
    manifest, audit = verify_reference(cfg, dd, data, graphs)
    initial, groups, checks = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys((*original['code_sha256'], *SOURCES)))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr, encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    plan = dict(experiment='C versus shared learnable send/receive mapping generator',
        created_at=timestamp(), training_seed=0, split_seed=0, modes=list(MODES), folds=list(FOLDS),
        epochs=1000, planned_runs=10, reused_C_runs=10, parameters=PARAMETERS,
        initialization=initial, optimizer_groups=groups, class_names=dd['Class_Names'],
        modalities=dd['Modal_Name'], samples=dd['Sample_Num'], original_config=original['original_config'],
        data_sha256=hashes, candidate_hash=original['candidate_hash'],
        splits_artifact_sha256=sha(OUT/'data_splits_and_graphs.npz'),
        checkpoint_selection=original['checkpoint_selection'], transductive=True,
        optimizer='Original Adam/cosine; graph-only weight decay 0, remaining weight decay 0.0007',
        architecture='Replace six modality embeddings by one learned shared embedding; retain the same 16->32->4 conditioner, separate S/T, all 36 relation-content paths, edge learning, two operator passes, residuals and classifier.',
        shared_initialization='Arithmetic mean of the original six initialized modality embeddings; same native constructor and RNG; no initialization chosen using outcomes.',
        limitation='80 fewer active parameters; generator inputs and initial maps change. Not exact equal-capacity or equal-function initialization. Seed0 and historical first-max TEST ACC, exploratory estimates.',
        scope='Only SHARED_MAP ten-fold training; no tuning, additional seeds or datasets.',
        common_initialization_and_rng_verified=True, smoke=checks, reference_audit=audit,
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


def train_arm(cfg, dd, data, graphs, plan, fold, mode):
    folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists(): return
    verify_artifacts(plan); folder.mkdir(parents=True, exist_ok=True)
    SET_Random(0); model = make_model(dd, cfg, graphs, mode)
    assert state_hash(model) == plan['initialization'][mode]['state_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    assert optimizer_manifest(model, optimizer) == plan['optimizer_groups'][mode]
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']; fit, test = data['Mask'][fold-1]
    best_correct = -1; history = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','train_loss','test_loss','test_acc','lr','seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.train(); optimizer.zero_grad(set_to_none=True)
            z, emb, aux = model(x); loss = criterion(z, y, fit, emb, aux)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step(); model.eval()
            with torch.no_grad():
                z, emb, aux = model(x); test_loss = criterion(z, y, test, emb, aux)
                if not torch.isfinite(z).all() or not torch.isfinite(test_loss):
                    raise FloatingPointError('Nonfinite evaluation')
            correct = int((z[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'],
                seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
            for stage in (['best'] if correct > best_correct else []) + (['last'] if epoch == 1000 else []):
                score = dict(epoch=epoch, **metrics(z[test], y[test]))
                if stage == 'best': best_correct = correct; best = score
                else: last = score
                temp = folder/f'{stage}.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, mode=mode,
                    fold=fold, seed=0, protocol_sha256=sha(OUT/'PLAN.json')), temp)
                temp.replace(folder/f'{stage}.pt')
                save_predictions(folder/('predictions.npz' if stage == 'best' else 'last_predictions.npz'),
                    z, y, test, dd['Index'])
                dump(folder/f'{stage}.json', score)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                    fold=fold, seed=0, epochs=1000, planned_runs=10, completed_runs=fold-1,
                    best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                if epoch == 1 or epoch % 100 == 0: print(json.dumps(status), flush=True)
    record = dict(mode=mode, fold=fold, seed=0, parameters=PARAMETERS[mode], best=best, last=last,
        initial_model_sha256=plan['initialization'][mode]['state_sha256'],
        mean_last100_acc=float(np.mean(history[-100:])), best_checkpoint_sha256=sha(folder/'best.pt'),
        last_checkpoint_sha256=sha(folder/'last.pt'), completed_at=timestamp(),
        seconds=time.perf_counter()-started, peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        final_maps=map_diagnostics(model, dd, cfg))
    dump(folder/'complete.json', record)
    del model, optimizer, criterion, z, emb, aux, loss
    torch.cuda.empty_cache(); summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_artifacts(plan); rows = []
    for mode in MODES:
        model = make_model(dd, cfg, graphs, mode).eval()
        for fold in FOLDS:
            folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'; record = read(folder/'complete.json')
            assert (record['mode'], record['fold'], record['seed']) == (mode, fold, 0)
            assert record['parameters'] == PARAMETERS[mode]
            assert record['initial_model_sha256'] == plan['initialization'][mode]['state_sha256']
            assert read(folder/'optimizer_groups.json') == plan['optimizer_groups'][mode]
            with (folder/'epochs.csv').open() as stream: epochs = list(csv.DictReader(stream))
            with (REFERENCE/f'fold_{fold:02d}/epochs.csv').open() as stream: old = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1, 1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1 == record['best']['epoch']
            assert record['last']['epoch'] == 1000
            np.testing.assert_array_equal([float(r['lr']) for r in epochs], [float(r['lr']) for r in old])
            test = data['Mask'][fold-1][1]
            for stage, filename in (('best','predictions.npz'), ('last','last_predictions.npz')):
                assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (cp['mode'], cp['fold'], cp['seed']) == (mode, fold, 0)
                assert cp['metrics'] == record[stage] == read(folder/f'{stage}.json')
                assert cp['epoch'] == record[stage]['epoch']
                assert abs(float(epochs[cp['epoch']-1]['test_acc'])-record[stage]['acc']) < 1e-12
                model.load_state_dict(cp['state_dict'], strict=True)
                with torch.no_grad(): logits = model(data['Feature'])[0][test]
                assert torch.isfinite(logits).all()
                with np.load(folder/filename, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'], logits.cpu().numpy(), atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(saved['logits'].argmax(-1), logits.argmax(-1).cpu().numpy())
                actual = metrics(logits, data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','per_class_recall','confusion_matrix'):
                    np.testing.assert_allclose(actual[key], record[stage][key], atol=1e-12, rtol=0)
                for key in ('nll','brier'):
                    np.testing.assert_allclose(actual[key], record[stage][key], atol=2e-5, rtol=2e-5)
                rows.append(dict(mode=mode, fold=fold, stage=stage, passed=True,
                    maps=map_diagnostics(model, dd, cfg)))
        del model
    result = dict(passed=True, verified_runs=10, verified_checkpoints=len(rows), records=rows)
    dump(OUT/'audit.json', result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', parameters=plan['parameters'], smoke=plan['smoke'])), flush=True)
        return
    if (OUT/'COMPLETE.json').exists(): print('Already complete.'); return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES: train_arm(cfg, dd, data, graphs, plan, fold, mode)
        audit = audit_completed(cfg, dd, data, graphs, plan); summary = summarize(OUT, plan)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT/'status.json', dict(state='complete', completed_runs=10, updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error))); raise
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
