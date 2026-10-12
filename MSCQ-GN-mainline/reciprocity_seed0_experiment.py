"""Two fixed reciprocity arms plus audited historical C; no tuning/extra stages."""
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
    verify_manifest, verify_reference, save_predictions,
)
from Model.relation_sheaf import RelationSheafModel
from Model.reciprocity_control import ReciprocityControlModel, SHUFFLE_SEED
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from Loss import MSCQGNLoss
from analyze_reciprocity_control import summarize

OUT = ROOT/'Result/reciprocity_seed0_20261003'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay/seed_0/content'
MODES = ('REAL', 'SHUFFLED')
FOLDS = tuple(range(1, 11))
PARAMETERS = {'C': 895096, 'REAL': 895132, 'SHUFFLED': 895132}
SOURCES = ('Model/reciprocity_control.py', 'reciprocity_seed0_experiment.py',
    'analyze_reciprocity_control.py', 'tests/test_reciprocity_control.py',
    'tests/test_shared_mapping_control.py', 'Model/shared_mapping_control.py',
    'tests/test_modal_evidence.py', 'Model/modal_evidence.py',
    'alternating_cooperative_experiment.py', 'Model/alternating_cooperative.py',
    'analyze_alternating_cooperative.py', 'analyze_joint_evidence_routing.py',
    'Model/layers.py', 'Model/__init__.py', 'Loss/__init__.py', 'Utils/__init__.py')


def make_model(dd, cfg, graphs, mode):
    return ReciprocityControlModel(dd, cfg, graphs, mode, training_seed=0).to(cfg.Device)


def verify_artifacts(plan):
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_manifest'])
    for name, digest in plan['code_sha256'].items():
        assert sha(OUT/'source_snapshot'/name) == digest, name
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest, name


def diagnostics(model):
    p = model.propagation
    return dict(coefficients=p.reciprocity_bias.detach().cpu().tolist(),
        bounded_logit_contrasts=p.reciprocity_bias.detach().tanh().cpu().tolist(),
        max_absolute_contrast=float(p.reciprocity_bias.detach().tanh().abs().max()),
        metadata_statistics=model.reciprocity_statistics)


def smoke(cfg, dd, data, graphs):
    SET_Random(0)
    native = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    native_rng = rng_state()
    common = set(dict(native.named_parameters()))
    base_groups = read(REFERENCE/'fold_01/optimizer_groups.json')
    initial, groups, checks, codes = {}, {}, [], {}
    for mode in ('C', *MODES):
        SET_Random(0); model = make_model(dd, cfg, graphs, mode)
        assert_rng_equal(native_rng, rng_state())
        for name, value in native.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
        for name in ('edge', 'edge_group', 'similarity'):
            torch.testing.assert_close(getattr(model.propagation, name),
                                       getattr(native.propagation, name), atol=0, rtol=0)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[mode]
        initial[mode] = dict(state_sha256=state_hash(model), common_sha256=state_hash(model, common))
        groups[mode] = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
        assert groups[mode][0] == base_groups[0]
        observed = groups[mode][1]
        assert observed['parameters'] == base_groups[1]['parameters'] + (0 if mode == 'C' else 36)
        assert [n for n in observed['names'] if n != 'propagation.reciprocity_bias'] == base_groups[1]['names']
        assert observed['weight_decay'] == 0. and observed['lr'] == cfg.lr
        for training in (False, True):
            SET_Random(720); native.train(training)
            with torch.no_grad(): left = native(data['Feature'])
            state = rng_state()
            SET_Random(720); model.train(training)
            with torch.no_grad(): right = model(data['Feature'])
            assert_rng_equal(state, rng_state())
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
        if mode in MODES:
            codes[mode] = model.propagation.reciprocity_code.detach().cpu()
            assert all(s['mutual_edges'] > 0 and s['unilateral_edges'] > 0
                       for s in model.reciprocity_statistics)
            optimizer = relation_optimizer(model, cfg, 0.)
            criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
            started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
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
            torch.cuda.synchronize()
            assert bool((model.propagation.reciprocity_bias != 0).all())
            checks.append(dict(mode=mode, steps=8, test_metrics_inspected=False,
                seconds_per_epoch=(time.perf_counter()-started)/8,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                trained_diagnostics=diagnostics(model)))
            del optimizer, criterion, z, emb, aux, loss
        del model, left, right
    assert len({v['common_sha256'] for v in initial.values()}) == 1
    assert not torch.equal(codes['REAL'], codes['SHUFFLED'])
    assert groups['REAL'] == groups['SHUFFLED']
    del native
    torch.cuda.empty_cache()
    return initial, groups, checks, codes


def prepare():
    cfg, dd, data, graphs, hashes, old = base_setup()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json'); verify_artifacts(plan)
        assert plan['data_sha256'] == hashes and plan['candidate_hash'] == old['candidate_hash']
        assert plan['modes'] == list(MODES) and plan['parameters'] == PARAMETERS
        return cfg, dd, data, graphs, plan
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_reciprocity_control.py', '-v'], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8')
    if test.returncode: raise RuntimeError(test.stdout+test.stderr)
    manifest, audit = verify_reference(cfg, dd, data, graphs)
    initial, groups, checks, codes = smoke(cfg, dd, data, graphs)
    sources = tuple(dict.fromkeys((*old['code_sha256'], *SOURCES)))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'implementation_checks.log').write_text(test.stdout+test.stderr, encoding='utf-8')
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent/'splits_and_graphs.npz').read_bytes())
    np.savez_compressed(OUT/'reciprocity_codes.npz', **{m: c.numpy() for m,c in codes.items()})
    plan = dict(experiment='Content-conditioned symmetric reciprocal-status edge-logit adjustment',
        created_at=timestamp(), training_seed=0, split_seed=0, modes=list(MODES), folds=list(FOLDS),
        epochs=1000, planned_runs=20, reused_C_runs=10, parameters=PARAMETERS,
        initialization=initial, optimizer_groups=groups, class_names=dd['Class_Names'],
        modalities=dd['Modal_Name'], samples=dd['Sample_Num'], original_config=old['original_config'],
        data_sha256=hashes, candidate_hash=old['candidate_hash'],
        checkpoint_selection=old['checkpoint_selection'], transductive=True,
        edge_rule='sigmoid(original_C_logit + tanh(beta[r,m]) * (mutual_edge_label - relation_mean))',
        initialization_rule='All 36 new beta coefficients are zero; same common weights, RNG and initial function as native C.',
        random_control=dict(seed=SHUFFLE_SEED, relation_seed='base + relation_index',
            design='One permutation of reciprocal labels over unordered edges per relation, fixed across folds.',
            preserves='Original edges, similarities, relation-wise label counts, symmetry and active parameter count.',
            does_not_preserve='Per-node mutual degree or label/similarity association.'),
        optimizer='Original Adam/cosine; graph-only weight decay 0, other parameters 0.0007.',
        scope='Only REAL and SHUFFLED seed0 TAD3 ten-fold runs; no new seeds, tuning or directional operator.',
        decision='Interpret true metadata only against both native C and equal-capacity SHUFFLED; no automatic next-stage training.',
        common_initialization_and_rng_verified=True, smoke=checks, reference_audit=audit,
        code_sha256={name: sha(ROOT/name) for name in sources}, reference_manifest=manifest,
        environment=dict(executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target = OUT/'source_snapshot'/source; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'artifacts_manifest.json', {name: sha(OUT/name) for name in
        ('PLAN.json', 'implementation_checks.log', 'data_splits_and_graphs.npz', 'reciprocity_codes.npz')})
    verify_artifacts(plan); summarize(OUT, plan)
    dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=20))
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
                temporary = folder/f'{stage}.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, mode=mode,
                    fold=fold, seed=0, protocol_sha256=sha(OUT/'PLAN.json')), temporary)
                temporary.replace(folder/f'{stage}.pt')
                save_predictions(folder/('predictions.npz' if stage == 'best' else 'last_predictions.npz'),
                                 z, y, test, dd['Index'])
                dump(folder/f'{stage}.json', score)
                dump(folder/f'{stage}_reciprocity.json', diagnostics(model))
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                done = len(list((OUT/'seed_0').glob('*/fold_*/complete.json')))
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                    fold=fold, seed=0, epochs=1000, planned_runs=20, completed_runs=done,
                    best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                if epoch == 1 or epoch % 100 == 0: print(json.dumps(status), flush=True)
    dump(folder/'complete.json', dict(mode=mode, fold=fold, seed=0, parameters=PARAMETERS[mode],
        best=best, last=last, initial_model_sha256=plan['initialization'][mode]['state_sha256'],
        mean_last100_acc=float(np.mean(history[-100:])), best_checkpoint_sha256=sha(folder/'best.pt'),
        last_checkpoint_sha256=sha(folder/'last.pt'), completed_at=timestamp(),
        seconds=time.perf_counter()-started, peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        final_reciprocity=diagnostics(model)))
    del model, optimizer, criterion, z, emb, aux, loss
    torch.cuda.empty_cache(); summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_artifacts(plan); rows = []
    for mode in MODES:
        model = make_model(dd, cfg, graphs, mode).eval()
        expected_code = model.propagation.reciprocity_code.clone()
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
                checkpoint = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert checkpoint['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (checkpoint['mode'], checkpoint['fold'], checkpoint['seed']) == (mode, fold, 0)
                assert checkpoint['metrics'] == record[stage] == read(folder/f'{stage}.json')
                assert checkpoint['epoch'] == record[stage]['epoch']
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                torch.testing.assert_close(model.propagation.reciprocity_code, expected_code, atol=0, rtol=0)
                with torch.no_grad(): logits = model(data['Feature'])[0][test]
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
                rows.append(dict(mode=mode, fold=fold, stage=stage, passed=True))
        del model, expected_code
    result = dict(passed=True, verified_runs=20, verified_checkpoints=len(rows), records=rows)
    dump(OUT/'audit.json', result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', parameters=PARAMETERS, smoke=plan['smoke'])), flush=True)
        return
    if (OUT/'COMPLETE.json').exists(): print('Already complete.'); return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES: train_arm(cfg, dd, data, graphs, plan, fold, mode)
        audit = audit_completed(cfg, dd, data, graphs, plan)
        summary = summarize(OUT, plan)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT/'status.json', dict(state='complete', completed_runs=20, updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error))); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
