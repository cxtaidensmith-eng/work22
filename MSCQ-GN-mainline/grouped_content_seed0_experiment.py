"""Fixed G2 hidden-channel grouping and near-capacity W38 control; original C seed0 protocol."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import json
import sys
import time
import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (base_setup, read, state_hash, rng_state,
    assert_rng_equal, metrics, verify_manifest, verify_reference, save_predictions)
from Model.relation_sheaf import RelationSheafModel
from Model.grouped_content_propagation import GroupedContentModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_grouped_content import summarize, write_report, PARAMETERS

OUT = ROOT/'Result/grouped_content_seed0_20260929'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay'
MODES = ('G2', 'W38')
FOLDS = tuple(range(1, 11))
DIAGNOSTIC_EPOCHS = (1, 100, 500, 1000)
NEW_SOURCES = ('Model/grouped_content_propagation.py', 'grouped_content_seed0_experiment.py',
    'analyze_grouped_content.py', 'tests/test_grouped_content_propagation.py',
    'alternating_cooperative_experiment.py', 'Model/alternating_cooperative.py',
    'analyze_alternating_cooperative.py', 'analyze_joint_evidence_routing.py',
    'tests/test_modal_evidence.py', 'Model/modal_evidence.py', 'Utils/__init__.py', 'Loss/__init__.py',
    'Result/分组内容传播实验方案_20260929.md')


def make_model(dd, cfg, graphs, mode):
    return GroupedContentModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def verify_sources(plan):
    verify_manifest(plan['code_sha256'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return bool(np.isfinite(value))


def verify_artifacts(plan):
    verify_sources(plan)
    verify_manifest(plan['reference_manifest'])
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


@torch.no_grad()
def interventions(model, data, test):
    model.eval()
    model.capture_diagnostics = True
    model.propagation.intervention = 'normal'
    original_hash, before = state_hash(model), rng_state()
    normal = model(data['Feature'])[0][test]
    normal_parameters = model.propagation.group_map_parameters().clone()
    prediction = normal.argmax(-1)
    target = data['Label'][test]
    correct = prediction == target
    diagnostic = dict(model.last_diagnostics)
    result = {'normal': dict(metrics=metrics(normal, target), changed=0, corrected=0, harmed=0)}
    try:
        model.propagation.intervention = 'collapse'
        if model.variant == 'W38':
            torch.testing.assert_close(normal_parameters, model.propagation.group_map_parameters(), atol=0, rtol=0)
        logits = model(data['Feature'])[0][test]
        pred = logits.argmax(-1)
        good = pred == target
        result['collapse'] = dict(metrics=metrics(logits, target), changed=int((pred != prediction).sum()),
            corrected=int((~correct & good).sum()), harmed=int((correct & ~good).sum()),
            max_abs_logit_change=float((logits-normal).abs().max()))
        if model.variant == 'W38':
            torch.testing.assert_close(normal, logits, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(prediction, pred, atol=0, rtol=0)
        assert finite_tree(diagnostic)
        assert state_hash(model) == original_hash
        assert_rng_equal(before, rng_state())
    finally:
        model.propagation.intervention = 'normal'
    return result, diagnostic


def smoke(cfg, dd, data, graphs):
    models, states, initial = {}, {}, {}
    for mode in ('original', 'C', 'G1', *MODES):
        SET_Random(0)
        models[mode] = (RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
                        if mode == 'original' else make_model(dd, cfg, graphs, mode))
        model = models[mode]
        states[mode] = rng_state()
        assert_rng_equal(states['original'], states[mode])
        count = sum(p.numel() for p in model.parameters())
        assert count == PARAMETERS['C' if mode == 'original' else mode]
        for name, value in models['original'].state_dict().items():
            if mode in ('original', 'C', 'G1') or not name.startswith('propagation.conditioner.'):
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        if mode in MODES:
            old, new = models['original'].propagation.conditioner, model.propagation.conditioner
            torch.testing.assert_close(old[0].weight, new[0].weight[:32], atol=0, rtol=0)
            torch.testing.assert_close(old[0].bias, new[0].bias[:32], atol=0, rtol=0)
            for group in range(2 if mode == 'G2' else 1):
                torch.testing.assert_close(old[2].weight, new[2].weight[4*group:4*(group+1), :32], atol=0, rtol=0)
                torch.testing.assert_close(old[2].bias, new[2].bias[4*group:4*(group+1)], atol=0, rtol=0)
            if mode == 'W38':
                assert torch.count_nonzero(new[2].weight[:, 32:]) == 0
        initial[mode] = dict(state_sha256=state_hash(model), parameters=count)
    assert len({initial[m]['state_sha256'] for m in ('original', 'C', 'G1')}) == 1
    errors = {}
    for train in (False, True):
        ref = ref_rng = None
        for mode, model in models.items():
            SET_Random(3701)
            model.train(train)
            with torch.no_grad():
                output = model(data['Feature'])
            if mode == 'original':
                ref, ref_rng = output, rng_state()
            torch.testing.assert_close(ref, output, atol=2e-5, rtol=2e-5)
            assert_rng_equal(ref_rng, rng_state())
            errors[f'{mode}_{train}'] = float((ref[0]-output[0]).abs().max())
    del models, model, ref, output, old, new
    torch.cuda.empty_cache()
    rows = []
    original_groups = read(REFERENCE/'seed_0/content/fold_01/optimizer_groups.json')
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, graphs, mode)
        assert state_hash(model) == initial[mode]['state_sha256']
        optimizer = relation_optimizer(model, cfg, 0.)
        groups = optimizer_manifest(model, optimizer)
        assert groups[0] == original_groups[0]
        assert groups[1]['names'] == original_groups[1]['names']
        assert groups[1]['weight_decay'] == 0 and groups[1]['lr'] == cfg.lr
        assert groups[1]['parameters']-original_groups[1]['parameters'] == PARAMETERS[mode]-PARAMETERS['C']
        criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
        fit = data['Mask'][0][0]
        incoming_before = model.propagation.conditioner[0].weight.detach().clone()
        extra_input_gradients, extra_output_gradients = [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step in range(8):
            model.train()
            model.capture_diagnostics = False
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(data['Feature'])
            loss = criterion(logits, data['Label'], fit, embeddings, auxiliary)
            assert torch.isfinite(loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), (mode, name)
                if name.startswith('propagation.'):
                    assert parameter.grad is not None, (mode, name)
            if mode == 'W38':
                extra_input_gradients.append(float(model.propagation.conditioner[0].weight.grad[32:].abs().max()))
                extra_output_gradients.append(float(model.propagation.conditioner[2].weight.grad[:, 32:].abs().max()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            model.capture_diagnostics = step == 7
            with torch.no_grad():
                assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        assert finite_tree(model.last_diagnostics)
        if mode == 'W38':
            assert extra_output_gradients[0] > 0 and max(extra_input_gradients[1:]) > 0
            assert (model.propagation.conditioner[0].weight[32:]-incoming_before[32:]).abs().max() > 0
            assert model.propagation.conditioner[2].weight[:, 32:].abs().max() > 0
        else:
            assert model.last_diagnostics['mean_group_parameter_std'] > 0
        rows.append(dict(mode=mode, passed=True, test_metrics_inspected=False,
            seconds_per_epoch=(time.perf_counter()-started)/8,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            diagnostics=model.last_diagnostics, optimizer_groups=groups,
            extra_input_gradient_max=extra_input_gradients, extra_output_gradient_max=extra_output_gradients))
        del model, optimizer, criterion, logits, embeddings, auxiliary, loss, incoming_before
        torch.cuda.empty_cache()
    return dict(passed=True, test_metrics_inspected=False, initialization=initial,
        initial_output_max_errors=errors, common_initialization_and_rng_matched=True, smoke=rows)


def prepare():
    cfg, dd, data, graphs, hashes, original = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        verify_artifacts(plan)
        assert plan['training_seed'] == 0 and plan['modes'] == list(MODES)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == original['candidate_hash']
        assert plan['folds'] == list(FOLDS) and plan['epochs'] == 1000 and plan['planned_runs'] == 20
        return cfg, dd, data, graphs, plan
    check_log = ROOT/'grouped_content_checks.log'
    check_text = check_log.read_text(encoding='utf-8-sig')
    assert 'OK' in check_text and 'FAILED' not in check_text
    manifest, reference_audit = verify_reference(cfg, dd, data, graphs)
    checks = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys((*original['code_sha256'], *NEW_SOURCES)))
    plan = dict(experiment='Two hidden-channel groups for C content-conditioned propagation',
        training_seed=0, split_seed=0, modes=list(MODES), folds=list(FOLDS), epochs=1000,
        planned_runs=20, reused_C_runs=10, class_names=dd['Class_Names'],
        modalities=dd['Modal_Name'], samples=dd['Sample_Num'], parameters=PARAMETERS,
        original_config=original['original_config'], class_weights=dd['Label_Weight'].cpu().tolist(),
        data_sha256=hashes, candidate_hash=original['candidate_hash'], candidate_k=10, transductive=True,
        checkpoint_selection=original['checkpoint_selection'],
        architecture=dict(C='One content-modality-conditioned 2x2 send/receive map for all hidden channels',
            G2='Split last axis of original [2,H/2] layout into two equal channel groups; separate maps',
            W38='Single group; conditioner hidden width 32 to 38, 6 fewer parameters than G2',
            G1='Implementation equivalence wrapper only; not a trained arm'),
        initialization='Fresh seed0 C initialization every fold. G2 copies the initial C map into both groups. W38 copies first32 units and zero-initializes output weights of extra6 units. No trained-checkpoint warm start.',
        randomness='Original construction and training CPU/CUDA RNG streams preserved; extra constructor draws isolated with fork_rng and private seed29101. Original noise/dropout unchanged.',
        optimizer='Original Adam lr=.006: propagation except output_norm decay=0; all remaining parameters decay=.0007; original cosine scheduler/clip1.',
        objective='Unchanged MSCQGNLoss: class-weighted label-smoothed final CE + OVR auxiliary CE weight1 + all-cohort orthogonality rate.0001.',
        run_order='For each original fold 1..10: G2 then W38; 20 new full fits. --folds only schedules a subset, never changes planned ten folds.',
        interventions='Fixed best/last: average raw map angle/scale logits across groups then broadcast before conformal map. No retraining or reselection; W38 single-group identity control.',
        primary='Equal-fold best TEST ACC: G2-C and G2-W38, both required. W38 is approximately capacity matched, not identical function class.',
        secondary='Macro F1, original raw-logit OVR AUC, class recalls, epoch1000, last100 mean, paired corrected/harmed, and map-collapse dependency.',
        scope='Fixed candidates, no test-metric selection in smoke, no new support graph/loss/seeds/hyperparameter tuning/automatic follow-up.',
        code_sha256={name: sha(ROOT/name) for name in sources}, reference_manifest=manifest,
        initialization_checks=checks, reference_audit=reference_audit,
        environment=dict(executable=sys.executable, python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    OUT.mkdir(parents=True, exist_ok=False)
    for source in sources:
        destination = OUT/'source_snapshot'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/source).read_bytes())
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE/'seed_0/splits_and_graphs.npz').read_bytes())
    (OUT/'implementation_checks.log').write_bytes(check_log.read_bytes())
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
    initial_hash = state_hash(model)
    assert initial_hash == plan['initialization_checks']['initialization'][mode]['state_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
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
            model.capture_diagnostics = False
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
            model.capture_diagnostics = epoch in DIAGNOSTIC_EPOCHS
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                assert torch.isfinite(logits).all()
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
                assert torch.isfinite(test_loss)
            if model.capture_diagnostics:
                assert finite_tree(model.last_diagnostics)
                diagnostics[str(epoch)] = model.last_diagnostics
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
                model.capture_diagnostics = True
                model.propagation.intervention = 'normal'
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
                result, diagnostic = interventions(model, data, test)
                rows.append(dict(mode=mode, seed=0, fold=fold, stage=stage,
                    max_abs_replay_error=error, interventions=result, diagnostics=diagnostic))
            diagnostic = read(folder/'diagnostics.json')
            assert sorted(map(int, diagnostic)) == list(DIAGNOSTIC_EPOCHS) and finite_tree(diagnostic)
            assert sum(p.numel() for p in model.parameters()) == PARAMETERS[mode]
            assert optimizer_manifest(model, relation_optimizer(model, cfg, 0.)) == read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result = dict(passed=True, verified_runs=20, verified_checkpoints=40, records=rows,
        reference_best_checkpoints_replayed_at_preparation=10,
        same_original_lr_schedule=True, same_original_source_and_reference_hashes=True,
        reference_last_caveat='Original C has no last checkpoint; only saved last predictions available.')
    dump(OUT/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Reference replay and 8-step implementation smoke only')
    parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS), help='Original folds to schedule; full plan remains ten folds')
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not set(args.folds).issubset(FOLDS):
        parser.error('--folds must contain unique integers from 1 through 10')
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=20, parameters=plan['parameters'],
            checks=plan['initialization_checks'])), flush=True)
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
                completed_runs=summary['completed_new_runs'], planned_runs=20, requested_folds=args.folds))
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
