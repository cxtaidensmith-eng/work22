"""Predeclared additive / pairwise / capacity-matched fusion experiment.

All arms are trained from scratch. The retained first-best TEST ACC selection
is explicitly exploratory. No patient graph, auxiliary head supervision, or
interaction selection is introduced in this first-round experiment.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp, scores
from Utils import SET_Random, CustomCosineAnnealingLR
from Model.modal_evidence import ModalEvidenceModel
from analyze_modal_evidence import summarize


OUT = ROOT / 'Result/modal_evidence_fusion_20260922'
MODES = ('additive', 'pairwise', 'fusion')
SEEDS = (0, 1, 2)
FOLDS = tuple(range(1, 11))
RANK = 8
BASELINE = ROOT / 'Result/relation_mechanism_nodecay'
SOURCES = ('Model/modal_evidence.py', 'modal_evidence_experiment.py',
           'analyze_modal_evidence.py', 'tests/test_modal_evidence.py',
           'Model/network.py', 'Model/models.py', 'Model/relation_sheaf.py',
           'relation_experiment.py', 'Utils/utils.py', 'Utils/data_load.py',
           'Utils/graph_load.py', 'Config/main_tadpole_3class.ini')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def state_hash(model, names=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if names is not None and name not in names:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(dd, cfg, mode, seed):
    return ModalEvidenceModel(dd, cfg, mode, training_seed=seed, rank=RANK).to(cfg.Device)


def verified_setup():
    cfg, dd, data, graphs, paths = setup()
    original = read(BASELINE / 'seed_0/PROTOCOL.json')
    for source, digest in original['code_sha256'].items():
        if sha(ROOT / source) != digest:
            raise RuntimeError(f'Original source changed: {source}')
    for key, value in original['original_config'].items():
        if getattr(cfg, key) != value:
            raise RuntimeError(f'Original configuration changed: {key}')
    data_hashes = {Path(path).name: sha(path) for path in paths}
    if data_hashes != original['data_sha256']:
        raise RuntimeError('Original data hashes differ')
    if cfg.epochs != 1000 or len(data['Mask']) != 10:
        raise RuntimeError('Requires retained 1000-epoch ten-fold configuration')
    baseline_manifest = {}
    for seed in SEEDS:
        source_protocol = BASELINE / f'seed_{seed}/PROTOCOL.json'
        reference = read(source_protocol)
        if reference['data_sha256'] != data_hashes:
            raise RuntimeError(f'C baseline data differs for seed {seed}')
        baseline_manifest[str(source_protocol.relative_to(ROOT))] = sha(source_protocol)
        for fold in FOLDS:
            folder = BASELINE / f'seed_{seed}/content/fold_{fold:02d}'
            test = data['Mask'][fold - 1][1].cpu().numpy()
            with np.load(folder / 'predictions.npz') as prediction:
                np.testing.assert_array_equal(prediction['labels'], data['Label'].cpu().numpy()[test])
                np.testing.assert_array_equal(prediction['row_index'], np.asarray(dd['Index'])[test])
            for name in ('predictions.npz', 'last_predictions.npz', 'epochs.csv', 'best.json', 'complete.json'):
                path = folder / name
                baseline_manifest[str(path.relative_to(ROOT))] = sha(path)
    return cfg, dd, data, original, data_hashes, baseline_manifest


def initialization_check(dd, cfg, x, seed):
    models = {}
    for mode in MODES:
        SET_Random(seed)
        cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        models[mode] = make_model(dd, cfg, mode, seed)
        assert torch.equal(cpu, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    common = set.intersection(*(set(model.state_dict()) for model in models.values()))
    hashes = {mode: state_hash(model, common) for mode, model in models.items()}
    assert len(set(hashes.values())) == 1
    common_ab = set(models['additive'].state_dict())
    assert state_hash(models['additive'], common_ab) == state_hash(models['pairwise'], common_ab)
    outputs = {}
    for mode, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[mode] = model(x)['logits'].detach()
    torch.testing.assert_close(outputs['additive'], outputs['pairwise'], rtol=0, atol=0)
    training_tokens = []
    training_logits = []
    end_rng = []
    for mode, model in models.items():
        SET_Random(7700 + seed)
        model.train()
        with torch.no_grad():
            result = model(x)
        training_tokens.append(result['tokens'])
        training_logits.append(result['logits'])
        end_rng.append((torch.get_rng_state(), torch.cuda.get_rng_state_all()))
    for tokens in training_tokens[1:]:
        torch.testing.assert_close(tokens, training_tokens[0], rtol=0, atol=0)
    torch.testing.assert_close(training_logits[0], training_logits[1], rtol=0, atol=0)
    for cpu, cuda in end_rng[1:]:
        assert torch.equal(cpu, end_rng[0][0])
        assert all(torch.equal(a, b) for a, b in zip(cuda, end_rng[0][1]))
    counts = {mode: model.parameter_counts() for mode, model in models.items()}
    assert abs(counts['pairwise']['total'] - counts['fusion']['total']) / counts['pairwise']['total'] < .01
    evidence = dict(seed=seed, common_state_sha256=hashes,
                    additive_pairwise_common_state_identical=True,
                    initial_eval_and_training_logits_identical=True,
                    matching_input_noise_and_dropout=True,
                    constructor_and_forward_rng_preserved=True, counts=counts)
    del models, outputs, training_tokens, training_logits, result
    torch.cuda.empty_cache()
    return evidence


def build_plan(cfg, dd, original, data_hashes, baseline_manifest, initialization):
    return dict(experiment='Independent modal evidence plus low-rank pairwise supplements',
                seeds=list(SEEDS), folds=list(FOLDS), modes=list(MODES), epochs=cfg.epochs,
                planned_runs=len(SEEDS) * len(FOLDS) * len(MODES), rank=RANK,
                dataset=cfg.DATA_SET, task=cfg.Task, samples=dd['Sample_Num'],
                class_names=dd['Class_Names'], modalities=dd['Modal_Name'],
                modal_dimensions=[len(v) for v in dd['Modal_Index']],
                original_config=original['original_config'],
                model_initialization_seed='3100 + training seed; same initialization across folds',
                training_rng='SET_Random(training_seed); mode construction restores CPU/all CUDA RNG',
                main_loss='Original class-weighted CE with label_smoothing=0.05, training mask only',
                class_weights=dd['Label_Weight'].detach().cpu().tolist(),
                supervision='Final summed/fused prediction only; no unimodal or class-query auxiliary loss',
                baseline_difference='C has graph, two contextual branches, OVR heads and orthogonality loss; C is an external performance reference, not a single-component ablation',
                preprocessing='Original processed CSV and exact source-row order unchanged',
                split_seed=cfg.seed, checkpoint_selection='First maximum TEST ACC, retained exploratory protocol',
                secondary_results='Fixed epoch 1000 and mean final 100 epoch ACC',
                optimizer='Adam with original lr/weight_decay; original grad clip and CustomCosineAnnealingLR stepped after evaluation',
                dropout='One cfg.Drop_rate token dropout shared by A/B/F; no graph/Transformer/drop_path or additional F dropout',
                priors='Original modality gate logits and per-modality input noise reused exactly',
                models={
                    'additive': 'Per-modality Linear(H,C), sum class logits',
                    'pairwise': 'Additive plus every unordered modality pair: Wout((Wleft*z_m)*(Wright*z_n))/sqrt(rank), no biases in pair branch; Wout initially zero',
                    'fusion': 'Concatenate the same tokens then Linear-GELU-Linear; hidden width matches all B-head parameters within rounding'},
                interpretation='B learns pairwise supplements, not proven pure interactions; unimodal summands are jointly trained and are not independently trained single-modality classifiers',
                followup='No sparse selector in this run; no automatic hyperparameter changes based on test results',
                initialization=initialization,
                original_protocol_sha256=sha(BASELINE / 'seed_0/PROTOCOL.json'),
                data_sha256=data_hashes, baseline_manifest=baseline_manifest,
                code_sha256={source: sha(ROOT / source) for source in SOURCES},
                environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                                 cuda=torch.version.cuda, device=torch.cuda.get_device_name(cfg.Device),
                                 threads=torch.get_num_threads()),
                caveat='Test-selected checkpoints are exploratory; repeated seeds do not create independent patients. No claim of proven patient-level selection or nonredundant interactions.')


def verify_sources(plan):
    for source, digest in plan['code_sha256'].items():
        if sha(ROOT / source) != digest:
            raise RuntimeError(f'Frozen experimental source changed: {source}')


def extra_scores(logits, labels):
    result = scores(logits, labels)
    probabilities = logits.softmax(-1)
    targets = F.one_hot(labels, logits.shape[-1]).float()
    result.update(nll=float(F.cross_entropy(logits, labels)),
                  brier=float((probabilities - targets).square().sum(-1).mean()))
    return result


def save_prediction(path, output, y, test, source_rows):
    values = dict(logits=output['logits'][test].detach().cpu().numpy(),
                  labels=y[test].detach().cpu().numpy(), row_index=np.asarray(source_rows)[test.cpu().numpy()])
    for name in ('unimodal_logits', 'pair_logits'):
        item = output[name]
        values[name] = (item[test].detach().cpu().numpy() if item is not None
                        else np.empty((int(test.sum()), 0, output['logits'].shape[-1]), dtype=np.float32))
    np.savez_compressed(path, **values)


def smoke(cfg, dd, data):
    initialization = initialization_check(dd, cfg, data['Feature'], 0)
    rows = []
    fit, _ = data['Mask'][0]
    for mode in MODES:
        SET_Random(0)
        model = make_model(dd, cfg, mode, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for epoch in range(20):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(data['Feature'])
            loss = F.cross_entropy(output['logits'][fit], data['Label'][fit],
                                   weight=dd['Label_Weight'], label_smoothing=.05)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f'Missing/nonfinite gradient: {mode}/{name}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                assert torch.isfinite(model(data['Feature'])['logits']).all()
        torch.cuda.synchronize()
        rows.append(dict(mode=mode, seconds_per_epoch=(time.perf_counter() - started) / 20,
                         peak_mb=torch.cuda.max_memory_allocated() / 2**20, counts=model.parameter_counts()))
        del model, optimizer, output, loss
        torch.cuda.empty_cache()
    evidence = dict(passed=True, initialization=initialization, runs=rows,
                    no_test_metrics_inspected=True, updated_at=timestamp())
    dump(ROOT / 'Result/modal_evidence_smoke_20260922.json', evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


def train_arm(cfg, dd, data, plan, seed, fold, mode):
    folder = OUT / f'seed_{seed}/{mode}/fold_{fold:02d}'
    if (folder / 'complete.json').exists():
        return
    verify_sources(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(seed)
    model = make_model(dd, cfg, mode, seed)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold - 1]
    train_y = torch.where(fit, y, torch.zeros_like(y))
    best_correct, best = -1, None
    history = []
    fields = ('epoch', 'train_loss', 'test_acc', 'test_nll', 'lr', 'grad_norm', 'seconds')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = F.cross_entropy(output['logits'][fit], train_y[fit],
                                   weight=dd['Label_Weight'], label_smoothing=.05)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss: {seed}/{fold}/{mode}/{epoch}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                                     error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                output = model(x)
                logits = output['logits']
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                correct = int((logits[test].argmax(-1) == y[test]).sum())
                nll = float(F.cross_entropy(logits[test], y[test]))
            row = dict(epoch=epoch, train_loss=float(loss.detach()),
                       test_acc=correct / int(test.sum()), test_nll=nll,
                       lr=optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm),
                       seconds=time.perf_counter() - started)
            writer.writerow(row)
            history.append(row)
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **extra_scores(logits[test], y[test]))
                checkpoint = dict(state_dict=model.state_dict(), seed=seed, fold=fold, mode=mode,
                                  epoch=epoch, rank=RANK, protocol_sha256=sha(OUT / 'PLAN.json'))
                torch.save(checkpoint, folder / 'best.pt')
                save_prediction(folder / 'predictions.npz', output, y, test, dd['Index'])
                dump(folder / 'best.json', best)
            if epoch == 1 or epoch % 100 == 0:
                stream.flush()
                status = dict(state='training', seed=seed, fold=fold, mode=mode, epoch=epoch,
                              best_acc=best['acc'], current_acc=row['test_acc'],
                              planned_runs=plan['planned_runs'], pid=os.getpid(), updated_at=timestamp())
                dump(OUT / 'status.json', status)
                print(json.dumps(status), flush=True)
            scheduler.step()
    last = dict(epoch=cfg.epochs, **extra_scores(logits[test], y[test]))
    torch.save(dict(state_dict=model.state_dict(), seed=seed, fold=fold, mode=mode, rank=RANK,
                    epoch=cfg.epochs, protocol_sha256=sha(OUT / 'PLAN.json')), folder / 'last.pt')
    save_prediction(folder / 'last_predictions.npz', output, y, test, dd['Index'])
    record = dict(seed=seed, mode=mode, fold=fold, best=best, last=last,
                  mean_last100_acc=float(np.mean([r['test_acc'] for r in history[-100:]])),
                  parameters=model.parameter_counts()['total'], counts=model.parameter_counts(),
                  initial_model_sha256=initial_hash,
                  best_checkpoint_sha256=sha(folder / 'best.pt'),
                  last_checkpoint_sha256=sha(folder / 'last.pt'),
                  seconds=time.perf_counter() - started,
                  peak_mb=torch.cuda.max_memory_allocated() / 2**20, completed_at=timestamp())
    dump(folder / 'complete.json', record)
    summarize(OUT, plan)
    print(json.dumps(dict(event='arm_complete', seed=seed, fold=fold, mode=mode,
                          best_acc=best['acc'], last_acc=last['acc'])), flush=True)
    del model, optimizer, scheduler, output, logits, loss
    torch.cuda.empty_cache()


def audit_completed(cfg, dd, data, plan, require_all=True):
    verify_sources(plan)
    for source, digest in plan['baseline_manifest'].items():
        if sha(ROOT / source) != digest:
            raise RuntimeError(f'Frozen C reference changed: {source}')
    verified = []
    for seed in plan['seeds']:
        for fold in plan['folds']:
            _, test = data['Mask'][fold - 1]
            baseline_folder = BASELINE / f'seed_{seed}/content/fold_{fold:02d}'
            with (baseline_folder / 'epochs.csv').open(encoding='utf-8') as stream:
                baseline_history = list(csv.DictReader(stream))
            assert len(baseline_history) == plan['epochs']
            for mode in plan['modes']:
                folder = OUT / f'seed_{seed}/{mode}/fold_{fold:02d}'
                if not (folder / 'complete.json').exists():
                    if require_all:
                        raise RuntimeError(f'Missing completion: {folder}')
                    continue
                record = read(folder / 'complete.json')
                with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
                    history = list(csv.DictReader(stream))
                assert len(history) == plan['epochs']
                assert [int(row['epoch']) for row in history] == list(range(1, plan['epochs'] + 1))
                np.testing.assert_allclose([float(row['lr']) for row in history],
                                           [float(row['lr']) for row in baseline_history],
                                           rtol=0, atol=1e-14)
                assert abs(float(np.mean([float(row['test_acc']) for row in history[-100:]]))
                           - record['mean_last100_acc']) < 1e-12
                best_index = int(np.argmax([float(row['test_acc']) for row in history]))
                assert int(history[best_index]['epoch']) == record['best']['epoch']
                model = make_model(dd, cfg, mode, seed).eval()
                assert state_hash(model) == record['initial_model_sha256']
                errors = {}
                for label, checkpoint_name, prediction_name in (
                    ('best', 'best.pt', 'predictions.npz'), ('last', 'last.pt', 'last_predictions.npz')):
                    assert sha(folder / checkpoint_name) == record[f'{label}_checkpoint_sha256']
                    checkpoint = torch.load(folder / checkpoint_name, map_location=cfg.Device, weights_only=False)
                    assert checkpoint['protocol_sha256'] == sha(OUT / 'PLAN.json')
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                    with torch.no_grad():
                        result = model(data['Feature'])
                    with np.load(folder / prediction_name) as saved:
                        np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                        actual = result['logits'][test].cpu().numpy()
                        np.testing.assert_allclose(actual, saved['logits'], rtol=2e-5, atol=2e-5)
                        errors[label] = float(np.max(np.abs(actual - saved['logits'])))
                        for name in ('unimodal_logits', 'pair_logits'):
                            if result[name] is not None:
                                np.testing.assert_allclose(result[name][test].cpu().numpy(), saved[name],
                                                           rtol=2e-5, atol=2e-5)
                    metrics = extra_scores(result['logits'][test], data['Label'][test])
                    for name in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                        assert abs(metrics[name] - record[label][name]) < 2e-5, name
                verified.append(dict(seed=seed, fold=fold, mode=mode, max_abs_replay_error=errors))
                del model, result, checkpoint
                torch.cuda.empty_cache()
    audit = dict(passed=True, verified_runs=len(verified), planned_runs=plan['planned_runs'],
                 verified_checkpoints=2 * len(verified), records=verified, updated_at=timestamp())
    dump(OUT / 'audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, original, data_hashes, baseline_manifest = verified_setup()
    if args.smoke:
        smoke(cfg, dd, data)
        return
    if args.audit:
        audit = audit_completed(cfg, dd, data, read(OUT / 'PLAN.json'), require_all=False)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        return
    if OUT.exists() and not args.resume:
        raise FileExistsError(f'{OUT}; use --resume to continue the frozen plan')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        if (OUT / 'PLAN.json').exists():
            plan = read(OUT / 'PLAN.json')
            verify_sources(plan)
            assert plan['data_sha256'] == data_hashes
            assert plan['baseline_manifest'] == baseline_manifest
        else:
            smoke_result = read(ROOT / 'Result/modal_evidence_smoke_20260922.json')
            assert smoke_result['passed']
            initialization = [initialization_check(dd, cfg, data['Feature'], seed) for seed in SEEDS]
            plan = build_plan(cfg, dd, original, data_hashes, baseline_manifest, initialization)
            dump(OUT / 'PLAN.json', plan)
            for source in SOURCES:
                destination = OUT / 'source_snapshot' / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / source).read_bytes())
            np.savez_compressed(OUT / 'data_and_splits.npz', source_row=np.asarray(dd['Index']),
                                labels=data['Label'].cpu().numpy(),
                                test_masks=np.stack([test.cpu().numpy() for _, test in data['Mask']]))
        dump(OUT / 'status.json', dict(state='starting', pid=os.getpid(), updated_at=timestamp()))
        summarize(OUT, plan)
        for seed in SEEDS:
            for fold in FOLDS:
                for mode in MODES:
                    train_arm(cfg, dd, data, plan, seed, fold, mode)
        audit = audit_completed(cfg, dd, data, plan)
        summary = summarize(OUT, plan)
        dump(OUT / 'COMPLETE.json', dict(summary=summary, audit=audit, completed_at=timestamp()))
        dump(OUT / 'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp()))
    except BaseException as error:
        dump(OUT / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
