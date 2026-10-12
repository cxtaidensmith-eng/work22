"""C readout controls: query/QK decay isolation and retrained uniform pooling.

Original TAD3 seed0 ten folds, 1000 epochs, first maximum TEST ACC. This runner
never starts early-key or any subsequent experiments automatically.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (base_setup, read, state_hash, rng_state,
    assert_rng_equal, metrics, verify_manifest, verify_reference, save_predictions)
from Model.relation_sheaf import RelationSheafModel
from Model.readout_controls import (ReadoutControlModel, readout_optimizer,
    readout_optimizer_manifest, readout_diagnostics, uniform_intervention)
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_readout_controls import summarize, write_report, PARAMETERS

OUT = ROOT/'Result/readout_controls_seed0_20260929'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay'
MODES = ('QK_NODECAY', 'UNIFORM')
FOLDS = tuple(range(1, 11))
DIAGNOSTIC_EPOCHS = (1, 100, 500, 1000)


def make_model(dd, cfg, graphs, mode):
    return ReadoutControlModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if value is None or isinstance(value, str):
        return True
    return bool(np.isfinite(value))


def verify_artifacts(plan):
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_manifest'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


def smoke(cfg, dd, data, graphs):
    models, streams, initial = {}, {}, {}
    for mode in ('original', 'C', *MODES):
        SET_Random(0)
        models[mode] = (RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
                        if mode == 'original' else make_model(dd, cfg, graphs, mode))
        streams[mode] = rng_state()
        assert_rng_equal(streams['original'], streams[mode])
        expected = PARAMETERS['C' if mode == 'original' else mode]
        assert sum(p.numel() for p in models[mode].parameters()) == expected
        for name, value in models['original'].state_dict().items():
            if mode == 'UNIFORM' and name.startswith('label_pools.'):
                continue
            torch.testing.assert_close(value, models[mode].state_dict()[name], atol=0, rtol=0)
        initial[mode] = dict(state_sha256=state_hash(models[mode]), parameters=expected)
    assert len({initial[m]['state_sha256'] for m in ('original', 'C', 'QK_NODECAY')}) == 1
    errors = {}
    for training in (False, True):
        expected = after = None
        for mode in ('original', 'C', 'QK_NODECAY'):
            SET_Random(3701)
            models[mode].train(training)
            with torch.no_grad():
                output = models[mode](data['Feature'])
            if expected is None:
                expected, after = output, rng_state()
            # Same native kernels/weights/RNG; CUDA sparse aggregation replay
            # is checked at the existing C audit tolerance, not bitwise output.
            torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
            assert_rng_equal(after, rng_state())
            errors[f'{mode}_{training}'] = float((output[0]-expected[0]).abs().max())
    # Full-model uniform control: original MHA with exactly zero Q/K weights.
    reference = models['original']
    with torch.no_grad():
        for pool in reference.label_pools:
            h = pool.attn.embed_dim
            pool.attn.in_proj_weight[:2*h].zero_()
            pool.attn.in_proj_bias[:2*h].zero_()
    for training in (False, True):
        SET_Random(3711)
        reference.train(training)
        with torch.no_grad():
            expected = reference(data['Feature'])
        after = rng_state()
        SET_Random(3711)
        models['UNIFORM'].train(training)
        with torch.no_grad():
            output = models['UNIFORM'](data['Feature'])
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
        assert_rng_equal(after, rng_state())
        errors[f'uniform_zero_qk_{training}'] = float((output[0]-expected[0]).abs().max())
    assert optimizer_manifest(models['C'], readout_optimizer(models['C'], cfg)) == read(
        REFERENCE/'seed_0/content/fold_01/optimizer_groups.json')
    del models, reference, expected, output, pool
    torch.cuda.empty_cache()
    checks = []
    fit = data['Mask'][0][0]
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode)
        assert state_hash(model) == initial[mode]['state_sha256']
        criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
        optimizer = readout_optimizer(model, cfg)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(8):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(data['Feature'])
            loss = criterion(logits, data['Label'], fit, embeddings, auxiliary)
            assert torch.isfinite(loss)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        digest, state = state_hash(model), rng_state()
        diagnostic = readout_diagnostics(model, data['Feature'])
        assert finite_tree(diagnostic)
        assert state_hash(model) == digest
        assert_rng_equal(state, rng_state())
        checks.append(dict(mode=mode, passed=True, test_metrics_inspected=False,
                           seconds_per_epoch=(time.perf_counter()-started)/8,
                           peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                           diagnostic=diagnostic,
                           optimizer_groups=readout_optimizer_manifest(model, optimizer)))
        del model, criterion, optimizer, logits, embeddings, auxiliary, loss
        torch.cuda.empty_cache()
    return dict(passed=True, test_metrics_inspected=False, initialization=initial,
                matched_common_initialization_and_rng=True, forward_max_abs_errors=errors,
                uniform_attention_dropout_rng_matched=True, smoke=checks)


def prepare():
    cfg, dd, data, graphs, hashes, old = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        verify_artifacts(plan)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == old['candidate_hash']
        assert plan['modes'] == list(MODES) and plan['training_seed'] == 0
        assert plan['folds'] == list(FOLDS) and plan['epochs'] == 1000 and plan['planned_runs'] == 20
        return cfg, dd, data, graphs, plan
    log = ROOT/'readout_controls_checks.log'
    text = log.read_text(encoding='utf-8-sig')
    assert 'OK' in text and 'FAILED' not in text
    reference_manifest, reference_audit = verify_reference(cfg, dd, data, graphs)
    checks = smoke(cfg, dd, data, graphs)
    sources = set(old['code_sha256'])
    sources.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py'))
    for directory in ('Model', 'Loss', 'Utils'):
        sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py'))
    sources.update(('tests/test_readout_controls.py', 'tests/test_modal_evidence.py',
                    'Result/类别读取对照实验方案_20260929.md'))
    plan = dict(experiment='C class-readout training controls before early-key experiments',
                training_seed=0, split_seed=0, modes=list(MODES), folds=list(FOLDS), epochs=1000,
                planned_runs=20, reused_C_runs=10, parameters=PARAMETERS,
                samples=dd['Sample_Num'], class_names=dd['Class_Names'], modalities=dd['Modal_Name'],
                original_config=old['original_config'], class_weights=dd['Label_Weight'].cpu().tolist(),
                data_sha256=hashes, candidate_hash=old['candidate_hash'], candidate_k=10,
                transductive=True, checkpoint_selection=old['checkpoint_selection'],
                architecture=dict(C='Original content-mode C, unchanged full forward path.',
                    QK_NODECAY='Same network and active parameters as C. Query and Q/K weights/biases have zero L2; packed V slice keeps original coupled L2 after clipping.',
                    UNIFORM='Delete inactive query/Q/K, keep copied initial V/output/norm/FFN; SDPA with zero Q/K enforces uniform logits and retains original attention dropout.'),
                optimizer='Original Adam and cosine schedule. Graph-only decay0; patient decay.0007 except QK_NODECAY selection parameters. No AdamW or decoupled decay.',
                objective='Unchanged class-weighted label-smoothed final CE, class OVR auxiliary CE weight1, all-cohort orthogonality rate.0001.',
                initialization='Fresh seed0 initialization every fold, identical retained parameters and constructor RNG. Uniform is intentionally not the same initial function as learned C.',
                run_order='Fold1..10, QK_NODECAY then UNIFORM. All folds planned before observing results.',
                primary='Best TEST ACC equal-fold mean, QK_NODECAY vs C and vs UNIFORM; UNIFORM vs C.',
                secondary='MacroF1, class recalls, epoch1000, last100, corrected/harmed, attention entropy/deviation, Q/K/V norms, fixed-checkpoint uniform intervention.',
                diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),
                decision_boundary='Nonuniform attention alone is not success. Early-key experiment requires reviewing these results and is not launched by this driver.',
                scope='No new splits/seeds/hyperparameter tuning/early-key/graph/fusion changes. Original test-selected exploratory protocol retained.',
                code_sha256={name: sha(ROOT/name) for name in sorted(sources)},
                reference_manifest=reference_manifest, reference_audit=reference_audit,
                initialization_checks=checks,
                environment=dict(executable=sys.executable, python=sys.version, torch=torch.__version__,
                                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    OUT.mkdir(parents=True, exist_ok=False)
    for name in sorted(sources):
        path = OUT/'source_snapshot'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE/'seed_0/splits_and_graphs.npz').read_bytes())
    (OUT/'implementation_checks.log').write_bytes(log.read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'artifacts_manifest.json', {name: sha(OUT/name) for name in
         ('PLAN.json', 'data_splits_and_graphs.npz', 'implementation_checks.log')})
    summarize(OUT, plan)
    dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=20))
    return cfg, dd, data, graphs, plan


def save_checkpoint(folder, stage, model, epoch, score, fold, mode):
    temporary = folder/f'{stage}.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, seed=0, fold=fold,
                    mode=mode, protocol_sha256=sha(OUT/'PLAN.json')), temporary)
    temporary.replace(folder/f'{stage}.pt')


def train_arm(cfg, dd, data, graphs, plan, fold, mode):
    folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists():
        return
    verify_artifacts(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(0)
    model = make_model(dd, cfg, graphs, mode)
    initial = state_hash(model)
    assert initial == plan['initialization_checks']['initialization'][mode]['state_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = readout_optimizer(model, cfg)
    dump(folder/'optimizer_groups.json', readout_optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold-1]
    best_correct, best, last, history, diagnostics = -1, None, None, [], {}
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == PARAMETERS[mode]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} fold{fold} epoch{epoch}: nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                assert torch.isfinite(logits).all()
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
                assert torch.isfinite(test_loss)
            if epoch in DIAGNOSTIC_EPOCHS:
                before = rng_state()
                diagnostics[str(epoch)] = readout_diagnostics(model, x)
                assert finite_tree(diagnostics[str(epoch)])
                assert_rng_equal(before, rng_state())
                dump(folder/'diagnostics.json', diagnostics)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                       test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'],
                       seconds=time.perf_counter()-started)
            writer.writerow(row)
            history.append(row['test_acc'])
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'best', model, epoch, best, fold, mode)
                save_predictions(folder/'predictions.npz', logits, y, test, dd['Index'])
                dump(folder/'best.json', best)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                              fold=fold, seed=0, epochs=1000, planned_runs=20, parameters=parameters,
                              completed_runs=sum((OUT/f'seed_0/{m}/fold_{f:02d}/complete.json').exists()
                                                 for m in MODES for f in FOLDS), best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                if epoch == 1 or epoch % 100 == 0:
                    print(json.dumps(status), flush=True)
            if epoch == 1000:
                last = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'last', model, epoch, last, fold, mode)
                save_predictions(folder/'last_predictions.npz', logits, y, test, dd['Index'])
    record = dict(mode=mode, fold=fold, seed=0, parameters=parameters, best=best, last=last,
                  initial_model_sha256=initial, mean_last100_acc=float(np.mean(history[-100:])),
                  best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
                  completed_at=timestamp(), seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache()
    summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_artifacts(plan)
    rows = []
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode).eval()
        for fold in FOLDS:
            folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
            record = read(folder/'complete.json')
            assert (record['mode'], record['seed'], record['fold']) == (mode, 0, fold)
            assert record['parameters'] == PARAMETERS[mode]
            assert record['initial_model_sha256'] == plan['initialization_checks']['initialization'][mode]['state_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                epochs = list(csv.DictReader(stream))
            with (REFERENCE/f'seed_0/content/fold_{fold:02d}/epochs.csv').open(encoding='utf-8') as stream:
                original = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1, 1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss', 'test_loss', 'test_acc', 'lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1 == record['best']['epoch']
            assert record['last']['epoch'] == 1000
            assert abs(float(np.mean([float(r['test_acc']) for r in epochs[-100:]]))-record['mean_last100_acc']) < 1e-12
            np.testing.assert_array_equal([float(r['lr']) for r in epochs], [float(r['lr']) for r in original])
            test = data['Mask'][fold-1][1]
            for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
                assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (cp['mode'], cp['seed'], cp['fold']) == (mode, 0, fold)
                assert cp['epoch'] == record[stage]['epoch'] and cp['metrics'] == record[stage]
                model.load_state_dict(cp['state_dict'], strict=True)
                with torch.no_grad():
                    logits = model(data['Feature'])[0]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'], logits[test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(saved['logits'].argmax(-1), logits[test].argmax(-1).cpu().numpy())
                    error = float(np.max(np.abs(saved['logits']-logits[test].cpu().numpy())))
                actual = metrics(logits[test], data['Label'][test])
                for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                    assert abs(actual[key]-record[stage][key]) < 2e-5
                np.testing.assert_array_equal(actual['confusion_matrix'], record[stage]['confusion_matrix'])
                digest, before = state_hash(model), rng_state()
                diagnostic = readout_diagnostics(model, data['Feature'])
                normal, uniform = uniform_intervention(model, data['Feature'])
                normal, uniform = normal[test], uniform[test]
                torch.testing.assert_close(normal, logits[test], atol=2e-5, rtol=2e-5)
                target = data['Label'][test]
                correct, changed_correct = normal.argmax(-1) == target, uniform.argmax(-1) == target
                intervention = dict(metrics=metrics(uniform, target),
                                    changed=int((normal.argmax(-1) != uniform.argmax(-1)).sum()),
                                    corrected=int((~correct & changed_correct).sum()),
                                    harmed=int((correct & ~changed_correct).sum()),
                                    max_abs_logit_change=float((normal-uniform).abs().max()))
                if mode == 'UNIFORM':
                    torch.testing.assert_close(normal, uniform, atol=0, rtol=0)
                assert state_hash(model) == digest
                assert_rng_equal(before, rng_state())
                assert finite_tree(diagnostic)
                rows.append(dict(mode=mode, seed=0, fold=fold, stage=stage,
                                 max_abs_replay_error=error, diagnostics=diagnostic,
                                 uniform_intervention=intervention))
            diagnostics = read(folder/'diagnostics.json')
            assert sorted(map(int, diagnostics)) == list(DIAGNOSTIC_EPOCHS) and finite_tree(diagnostics)
            assert readout_optimizer_manifest(model, readout_optimizer(model, cfg)) == read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result = dict(passed=True, verified_runs=20, verified_checkpoints=40, records=rows,
                  reference_best_checkpoints_replayed_at_preparation=10,
                  same_original_lr_schedule=True, source_reference_and_prediction_checks=True,
                  reference_last_caveat='Original C has no last.pt; only saved last predictions available.')
    dump(OUT/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Reference replay and implementation smoke only')
    parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS))
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not set(args.folds).issubset(FOLDS):
        parser.error('Use unique original folds from 1 through 10')
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=20, parameters=plan['parameters'],
                              initialization_checks=plan['initialization_checks'])), flush=True)
        return
    if (OUT/'COMPLETE.json').exists():
        print('Already complete; retained results unchanged.', flush=True)
        return
    lock = OUT/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in args.folds:
            for mode in MODES:
                train_arm(cfg, dd, data, graphs, plan, fold, mode)
        summary = summarize(OUT, plan)
        if not summary['all_new_complete']:
            dump(OUT/'status.json', dict(state='partial_complete', updated_at=timestamp(),
                                       completed_runs=summary['completed_new_runs'], planned_runs=20))
            return
        dump(OUT/'status.json', dict(state='auditing', pid=os.getpid(), updated_at=timestamp()))
        audit = audit_completed(cfg, dd, data, graphs, plan)
        summary = write_report(OUT, plan, audit)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT/'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp(), completed_runs=20))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', pid=os.getpid(), updated_at=timestamp(), error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
