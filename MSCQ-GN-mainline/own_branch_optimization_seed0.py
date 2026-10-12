"""Run the fixed TAD3 own-gate/FFN factorial: 30 new full GPU fits."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import json
import subprocess
import sys
import time
import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (
    base_setup, read, state_hash, rng_state, assert_rng_equal, metrics,
    verify_manifest, verify_reference, save_predictions)
from Model.relation_sheaf import RelationSheafModel
from Model.own_branch_optimization import OwnBranchOptimizationModel
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from Loss import MSCQGNLoss
from analyze_own_branch_optimization import summarize, write_report, PARAMETERS

OUT = ROOT/'Result/own_branch_optimization_seed0_20261004'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODES = ('NO_GATE', 'FFN', 'NO_GATE_FFN')
FOLDS = tuple(range(1, 11))
DIAGNOSTIC_EPOCHS = (1, 100, 500, 1000)
SOURCES = ('Model/own_branch_optimization.py', 'own_branch_optimization_seed0.py',
    'analyze_own_branch_optimization.py', 'tests/test_own_branch_optimization.py',
    'tests/test_modal_evidence.py', 'Model/modal_evidence.py',
    'alternating_cooperative_experiment.py', 'Model/alternating_cooperative.py',
    'analyze_alternating_cooperative.py', 'analyze_joint_evidence_routing.py',
    'Model/layers.py', 'Model/__init__.py', 'Loss/__init__.py', 'Utils/__init__.py')


def make_model(dd, cfg, graphs, mode):
    return OwnBranchOptimizationModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def verify_artifacts(plan):
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_manifest'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


def smoke(cfg, dd, data, graphs):
    SET_Random(0)
    native = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    native_rng = rng_state()
    common = set(dict(native.named_parameters()))
    base_groups = read(REFERENCE/'fold_01/optimizer_groups.json')
    initial, groups, checks = {}, {}, []
    for mode in ('C', *MODES):
        SET_Random(0); model = make_model(dd, cfg, graphs, mode)
        assert_rng_equal(native_rng, rng_state())
        for name, value in native.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[mode]
        initial[mode] = dict(state_sha256=state_hash(model), common_sha256=state_hash(model, common))
        groups[mode] = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
        assert groups[mode][1] == base_groups[1]
        assert groups[mode][0]['weight_decay'] == cfg.weight_decay == .0007
        assert [n for n in groups[mode][0]['names'] if '.token_ffn.' not in n] == base_groups[0]['names']
        assert groups[mode][0]['parameters']-base_groups[0]['parameters'] == PARAMETERS[mode]-PARAMETERS['C']
        errors = {}
        for training in (False, True):
            SET_Random(720); native.train(training)
            with torch.no_grad(): left = native(data['Feature'])
            state = rng_state()
            SET_Random(720); model.train(training)
            with torch.no_grad(): right = model(data['Feature'])
            assert_rng_equal(state, rng_state())
            torch.testing.assert_close(left[1:], right[1:], atol=2e-5, rtol=2e-5)
            if mode in ('C', 'FFN'):
                torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
            errors[str(training)] = float((left[0]-right[0]).abs().max())
        if mode in MODES:
            optimizer = relation_optimizer(model, cfg, 0.)
            criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
            before_ffn = model.Global_Message.token_ffn[0].weight.detach().clone() if model.use_own_ffn else None
            started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for _ in range(8):
                model.train(); optimizer.zero_grad(set_to_none=True)
                z, emb, aux = model(data['Feature'])
                loss = criterion(z, data['Label'], data['Mask'][0][0], emb, aux)
                assert torch.isfinite(loss); loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all(), name
                    if name.startswith('propagation.') or '.token_ffn.' in name or name == 'modal_gate_logit':
                        assert p.grad is not None, name
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step(); model.eval()
                with torch.no_grad(): assert torch.isfinite(model(data['Feature'])[0]).all()
            torch.cuda.synchronize()
            if model.use_own_ffn:
                assert float(model.Global_Message.token_ffn[2].weight.abs().max()) > 0
                assert float((model.Global_Message.token_ffn[0].weight-before_ffn).abs().max()) > 0
            model.capture_diagnostics = True
            with torch.no_grad(): model(data['Feature'])
            checks.append(dict(mode=mode, steps=8, test_metrics_inspected=False,
                initial_full_output_errors_vs_C=errors,
                seconds_per_epoch=(time.perf_counter()-started)/8,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                trained_diagnostics=model.last_diagnostics))
            del optimizer, criterion, z, emb, aux, loss
        del model, left, right
    assert len({v['common_sha256'] for v in initial.values()}) == 1
    assert initial['NO_GATE']['state_sha256'] == initial['C']['state_sha256']
    assert initial['FFN']['state_sha256'] == initial['NO_GATE_FFN']['state_sha256']
    del native
    torch.cuda.empty_cache()
    return initial, groups, checks


def prepare():
    cfg, dd, data, graphs, hashes, old = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json'); verify_artifacts(plan)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == old['candidate_hash']
        assert plan['modes'] == list(MODES) and plan['parameters'] == PARAMETERS
        return cfg, dd, data, graphs, plan
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_own_branch_optimization.py', '-v'], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8')
    if test.returncode: raise RuntimeError(test.stdout+test.stderr)
    manifest, audit = verify_reference(cfg, dd, data, graphs)
    initial, groups, checks = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys((*old['code_sha256'], *SOURCES)))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr, encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    plan = dict(experiment='Own-branch gate removal and shared per-token residual FFN, 2x2 design',
        created_at=timestamp(), training_seed=0, split_seed=0, modes=list(MODES), folds=list(FOLDS),
        epochs=1000, planned_runs=30, reused_C_runs=10, parameters=PARAMETERS,
        initialization=initial, optimizer_groups=groups, class_names=dd['Class_Names'],
        modalities=dd['Modal_Name'], samples=dd['Sample_Num'], original_config=old['original_config'],
        data_sha256=hashes, candidate_hash=old['candidate_hash'],
        checkpoint_selection=old['checkpoint_selection'], transductive=True,
        architecture=dict(own_gate='Removed ONLY for NO_GATE/NO_GATE_FFN; population path retains it.',
            ffn='Shared Linear64->128, GELU, Linear128->64, residual after own second attention/LN and before flatten.',
            unchanged='Original input gain/bias/noise, population propagation, own attention, readout, sum, classifier and losses.'),
        initialization_rule='Fresh seed0 C weights for all common parameters. No gate absorbed into own projections. FFN second Linear weight/bias zero. C=FFN and NO_GATE=NO_GATE_FFN initial functions; gate-off intentionally differs from C.',
        optimizer='Original Adam/cosine; graph-only weight decay 0, remaining and FFN parameters 0.0007.',
        run_order='Fold-major: NO_GATE, FFN, NO_GATE_FFN, 1000 epochs each; no interim adaptation.',
        scope='Only TAD3 seed0 30 fits; no auto extra seeds, hyperparameter search, capacity controls or base-model promotion.',
        comparison='Gate effect with and without FFN; FFN effect with and without gate; combined vs C. Best primary, last/last100 secondary.',
        common_initialization_and_rng_verified=True, smoke=checks, reference_audit=audit,
        code_sha256={name: sha(ROOT/name) for name in sources}, reference_manifest=manifest,
        environment=dict(executable=sys.executable, python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target = OUT/'source_snapshot'/source; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'artifacts_manifest.json', {name: sha(OUT/name) for name in
        ('PLAN.json', 'implementation_checks.log', 'data_splits_and_graphs.npz')})
    verify_artifacts(plan); summarize(OUT, plan)
    dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=30))
    return cfg, dd, data, graphs, plan


def save_checkpoint(folder, stage, model, epoch, score, fold, mode):
    temporary = folder/f'{stage}.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, seed=0,
        fold=fold, mode=mode, protocol_sha256=sha(OUT/'PLAN.json')), temporary)
    temporary.replace(folder/f'{stage}.pt')


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
    best_correct, best, history, diagnostics = -1, None, [], {}
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','train_loss','test_loss','test_acc','lr','seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.capture_diagnostics = False
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss): raise FloatingPointError(f'{mode}/{fold}/{epoch}: nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step(); model.eval()
            model.capture_diagnostics = epoch in DIAGNOSTIC_EPOCHS
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
            if model.capture_diagnostics:
                diagnostics[str(epoch)] = model.last_diagnostics
                dump(folder/'diagnostics.json', diagnostics)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
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
                    fold=fold, seed=0, epochs=1000, planned_runs=30, best_acc=best['acc'], **row)
                dump(OUT/'status.json', status); print(json.dumps(status), flush=True)
            if epoch == 1000:
                last = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'last', model, epoch, last, fold, mode)
                save_predictions(folder/'last_predictions.npz', logits, y, test, dd['Index'])
    record = dict(mode=mode, fold=fold, seed=0, parameters=PARAMETERS[mode], best=best, last=last,
        initial_model_sha256=plan['initialization'][mode]['state_sha256'],
        mean_last100_acc=float(np.mean(history[-100:])), best_checkpoint_sha256=sha(folder/'best.pt'),
        last_checkpoint_sha256=sha(folder/'last.pt'), completed_at=timestamp(),
        seconds=time.perf_counter()-started, peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache(); summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_artifacts(plan)
    rows = []
    for mode in MODES:
        SET_Random(0); model = make_model(dd, cfg, graphs, mode).eval()
        for fold in FOLDS:
            folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
            record = read(folder/'complete.json')
            assert (record['mode'], record['seed'], record['fold']) == (mode, 0, fold)
            assert record['parameters'] == PARAMETERS[mode]
            assert record['initial_model_sha256'] == plan['initialization'][mode]['state_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream: epochs = list(csv.DictReader(stream))
            with (REFERENCE/f'fold_{fold:02d}'/'epochs.csv').open(encoding='utf-8') as stream: original = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1, 1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1 == record['best']['epoch']
            np.testing.assert_array_equal([float(r['lr']) for r in epochs], [float(r['lr']) for r in original])
            assert abs(float(np.mean([float(r['test_acc']) for r in epochs[-100:]]))-record['mean_last100_acc']) < 1e-12
            test = data['Mask'][fold-1][1]
            for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
                assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (cp['mode'], cp['seed'], cp['fold']) == (mode, 0, fold)
                assert cp['epoch'] == record[stage]['epoch'] and cp['metrics'] == record[stage]
                model.load_state_dict(cp['state_dict'], strict=True); model.capture_diagnostics = True
                with torch.no_grad(): logits = model(data['Feature'])[0]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'], logits[test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(saved['logits'].argmax(-1), logits[test].argmax(-1).cpu().numpy())
                    error = float(np.max(np.abs(saved['logits']-logits[test].cpu().numpy())))
                actual = metrics(logits[test], data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','nll','brier'):
                    assert abs(actual[key]-record[stage][key]) < 2e-5, key
                np.testing.assert_array_equal(actual['confusion_matrix'], record[stage]['confusion_matrix'])
                np.testing.assert_allclose(actual['per_class_recall'], record[stage]['per_class_recall'], atol=1e-12)
                rows.append(dict(mode=mode, seed=0, fold=fold, stage=stage, max_abs_replay_error=error,
                                 diagnostics=model.last_diagnostics))
            assert sorted(map(int, read(folder/'diagnostics.json'))) == list(DIAGNOSTIC_EPOCHS)
            assert optimizer_manifest(model, relation_optimizer(model, cfg, 0.)) == read(folder/'optimizer_groups.json')
        del model; torch.cuda.empty_cache()
    result = dict(passed=True, verified_runs=30, verified_checkpoints=60, records=rows,
        reference_best_checkpoints_replayed_at_preparation=10, same_original_lr_schedule=True,
        same_original_source_and_reference_hashes=True)
    dump(OUT/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=30, smoke=plan['smoke'])), flush=True)
        return
    if (OUT/'COMPLETE.json').exists():
        print('Already complete; retained results unchanged.', flush=True); return
    lock = OUT/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES: train_arm(cfg, dd, data, graphs, plan, fold, mode)
        dump(OUT/'status.json', dict(state='auditing', pid=os.getpid(), updated_at=timestamp()))
        audit = audit_completed(cfg, dd, data, graphs, plan)
        summary = write_report(OUT, plan, audit)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit_passed=audit['passed'],
            audit_sha256=sha(OUT/'audit.json'), summary=summary))
        dump(OUT/'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp(), completed_runs=30))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', pid=os.getpid(), updated_at=timestamp(), error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
