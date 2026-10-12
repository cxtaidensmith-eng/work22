"""One fixed direct-cognitive CE term on the unchanged residual model."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import modal_evidence_experiment as historical
import cognitive_pet_aux_experiment as auxiliary_round
import cognitive_residual_experiment as residual_round
from Model.cognitive_residual import CognitiveResidualModel
from Model.cognitive_direct_supervision import classification_objective
from relation_experiment import ROOT, dump, sha, timestamp
from unimodal_round_experiment import read, point, verify_manifest
from Utils import SET_Random, CustomCosineAnnealingLR


OUT = ROOT / 'Result/cognitive_direct_supervision_seed0_20260923'
MODE = 'COG_PET_RESIDUAL_DIRECT_CE'
DIRECT_WEIGHT = .05
REFERENCES = {residual_round.MODE: residual_round.OUT,
              'COG_PET': auxiliary_round.REFERENCE, 'COG': auxiliary_round.REFERENCE}
NEW_SOURCES = ('Model/cognitive_direct_supervision.py',
               'cognitive_direct_supervision_experiment.py',
               'tests/test_cognitive_direct_supervision.py')


def make_model(dd, cfg, mode, seed):
    if mode != MODE:
        raise ValueError(mode)
    return CognitiveResidualModel(dd, cfg, 'PET', seed).to(cfg.Device)


def verify_references(cfg, dd, data, hashes):
    manifest, audit = auxiliary_round.verify_references(cfg, dd, data, hashes)
    plan = read(residual_round.OUT / 'PLAN.json')
    historical.verify_sources(plan)
    verify_manifest(plan['reference_manifest'])
    assert plan['seeds'] == [0] and plan['data_sha256'] == hashes
    assert plan['epochs'] == cfg.epochs == 1000
    assert plan['environment']['torch'] == torch.__version__
    manifest[str((residual_round.OUT / 'PLAN.json').relative_to(ROOT))] = sha(residual_round.OUT / 'PLAN.json')
    model = make_model(dd, cfg, MODE, 0).eval()
    initial_hash = historical.state_hash(model)
    verified = []
    for fold in range(1, 11):
        folder = residual_round.OUT / f'seed_0/{residual_round.MODE}/fold_{fold:02d}'
        record = read(folder / 'complete.json')
        assert (record['seed'], record['fold'], record['mode']) == (0, fold, residual_round.MODE)
        assert record['initial_model_sha256'] == initial_hash
        assert record['parameters'] == model.parameter_counts()['total'] == 40900
        with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1000 and [int(r['epoch']) for r in rows] == list(range(1, 1001))
        assert int(np.argmax([float(r['test_acc']) for r in rows])) + 1 == record['best']['epoch']
        _, test = data['Mask'][fold - 1]
        errors = {}
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            path = folder / f'{stage}.pt'
            assert sha(path) == record[f'{stage}_checkpoint_sha256']
            checkpoint = torch.load(path, map_location=cfg.Device, weights_only=False)
            assert checkpoint['protocol_sha256'] == sha(residual_round.OUT / 'PLAN.json')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with torch.no_grad():
                output = model(data['Feature'])
            with np.load(folder / filename) as saved:
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                for key in ('logits', 'unimodal_logits', 'pair_logits'):
                    np.testing.assert_allclose(saved[key], output[key][test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                errors[stage] = float(np.max(np.abs(saved['logits'] - output['logits'][test].cpu().numpy())))
            metrics = historical.extra_scores(output['logits'][test], data['Label'][test])
            for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                assert abs(metrics[key] - record[stage][key]) < 2e-5
        for filename in ('complete.json', 'best.json', 'epochs.csv', 'best.pt', 'last.pt',
                         'predictions.npz', 'last_predictions.npz'):
            path = folder / filename
            manifest[str(path.relative_to(ROOT))] = sha(path)
        verified.append(dict(mode=residual_round.MODE, seed=0, fold=fold, max_abs_replay_error=errors))
    return manifest, dict(passed=True, seeds=[0], verified_runs=30, verified_checkpoints=60,
                          cognitive_and_fusion=audit, residual_records=verified,
                          residual_initial_state_and_parameter_count_match=True)


def smoke(cfg, dd, data):
    original = residual_round.make_model(dd, cfg, residual_round.MODE, 0)
    model = make_model(dd, cfg, MODE, 0)
    assert historical.state_hash(original) == historical.state_hash(model)
    for training in (False, True):
        original.train(training); model.train(training)
        SET_Random(2701)
        with torch.no_grad():
            expected = original(data['Feature'])
        after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        SET_Random(2701)
        with torch.no_grad():
            actual = model(data['Feature'])
        for key in ('logits', 'unimodal_logits', 'pair_logits'):
            torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
        assert torch.equal(after[0], torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(after[1], torch.cuda.get_rng_state_all()))
    SET_Random(0)
    model = make_model(dd, cfg, MODE, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fit, _ = data['Mask'][0]
    start = time.perf_counter()
    for step in range(20):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(data['Feature'])
        losses = classification_objective(output, data['Label'], fit, dd['Label_Weight'], DIRECT_WEIGHT)
        losses['total'].backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if step > 0:
            assert model.head[0].weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    return dict(passed=True, inspected_test_metrics=False, same_initial_state_and_forward=True,
                matched_rng=True, counts=model.parameter_counts(),
                seconds_per_epoch=(time.perf_counter() - start) / 20)


def summarize(out, plan):
    records, folders, variants = {}, {}, {}
    for mode in (MODE, *REFERENCES):
        records[mode], folders[mode] = {}, {}
        base = out if mode == MODE else REFERENCES[mode]
        for fold in plan['folds']:
            folder = base / f'seed_0/{mode}/fold_{fold:02d}'
            if (folder / 'complete.json').exists():
                records[mode][fold] = read(folder / 'complete.json')
                folders[mode][fold] = folder
        rows = list(records[mode].values())
        v = dict(completed_runs=len(rows), planned_runs=10, seed=0, reused=mode != MODE)
        if rows:
            for stage in ('best', 'last'):
                v[stage] = {key: np.mean([r[stage][key] for r in rows], axis=0).tolist()
                            for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier', 'per_class_recall')}
            v['parameters'] = rows[0]['parameters']
            v['mean_last100_acc'] = float(np.mean([r['mean_last100_acc'] for r in rows]))
        variants[mode] = v
    comparisons = []
    for reference in REFERENCES:
        folds = sorted(records[MODE].keys() & records[reference].keys())
        if not folds:
            continue
        result = dict(left=MODE, right=reference, matched_runs=len(folds), stages={})
        for stage in ('best', 'last'):
            by_class = np.zeros((len(plan['class_names']), 2), dtype=int)
            corrected = harmed = 0
            deltas = []
            for fold in folds:
                a, b = point(folders[MODE][fold], stage), point(folders[reference][fold], stage)
                np.testing.assert_array_equal(a['row_index'], b['row_index'])
                np.testing.assert_array_equal(a['labels'], b['labels'])
                ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                gain, harm = ca & ~cb, ~ca & cb
                corrected += int(gain.sum()); harmed += int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c] += [int((gain & (a['labels'] == c)).sum()), int((harm & (a['labels'] == c)).sum())]
                deltas.append(float(ca.mean() - cb.mean()))
            result['stages'][stage] = dict(mean_acc_delta=float(np.mean(deltas)), corrected=corrected,
                harmed=harmed, net_corrected=corrected-harmed, per_class_corrected_harmed=by_class.tolist(),
                fold_acc_deltas=dict(zip(folds, deltas)), fold_wins=sum(d > 1e-12 for d in deltas),
                fold_ties=sum(abs(d) <= 1e-12 for d in deltas), fold_losses=sum(d < -1e-12 for d in deltas))
        comparisons.append(result)
    completed = variants[MODE]['completed_runs']
    summary = dict(updated_at=timestamp(), seeds=[0], completed_new_runs=completed, planned_new_runs=10,
                   reused_runs=30, variants=variants, comparisons=comparisons)
    dump(out / 'summary.json', summary)
    labels = {MODE: '残差＋认知直接监督', residual_round.MODE: '原残差模型',
              'COG_PET': '普通认知＋PET 融合', 'COG': '认知单模态'}
    lines = ['# 认知直接分支监督：固定权重 0.05', '', f'新增训练完成 {completed}/10；复用历史参照 30 次。',
             '种子 0，原十折、1000 轮、首次最高测试 ACC；原残差模型结构及所有初始化不变。',
             '总损失 = 最终预测 CE + 0.05 × 认知直接分支 CE；两项均使用训练折、原类别权重和平滑。', '',
             '| 模型 | 完成 | 择优 ACC | 宏 F1 | AD 召回 | 末轮 ACC |', '|---|---:|---:|---:|---:|---:|']
    for mode, v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {labels[mode]} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['best']['macro_f1']*100:.2f}% | {v['best']['per_class_recall'][0]*100:.2f}% | {v['last']['acc']*100:.2f}% |")
        else:
            lines.append(f'| {labels[mode]} | 0/10 | — | — | — | — |')
    lines += ['', '不同完成折数不能直接排名；同折配对结果见 summary.json。',
              '认知分支仅在最终模型选中的同一检查点及第 1000 轮诊断，不单独选轮。',
              '认知分支改善与最终分类改善分开判断；仅末轮改善不替代原主指标。',
              '本轮从头端到端训练；联合补充分支同时读取认知及 PET，不等于纯 PET 增量。',
              '不自动追加权重搜索、门控、其他模态或种子。']
    (out / '进度与结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def train_arm(cfg, dd, data, plan, fold):
    """Historical loop with only direct CE added to the training objective."""
    folder = OUT / f'seed_0/{MODE}/fold_{fold:02d}'
    if (folder / 'complete.json').exists():
        return
    historical.verify_sources(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(0)
    model = make_model(dd, cfg, MODE, 0)
    initial_hash = historical.state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold - 1]
    train_y = torch.where(fit, y, torch.zeros_like(y))
    best_correct, best, history = -1, None, []
    fields = ('epoch', 'train_loss', 'main_loss', 'direct_loss', 'test_acc', 'test_nll', 'lr', 'grad_norm', 'seconds')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(x)
            losses = classification_objective(output, train_y, fit, dd['Label_Weight'], plan['direct_weight'])
            loss = losses['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: fold {fold}, epoch {epoch}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step(); model.eval()
            with torch.no_grad():
                output = model(x); logits = output['logits']
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                correct = int((logits[test].argmax(-1) == y[test]).sum())
                nll = float(F.cross_entropy(logits[test], y[test]))
            row = dict(epoch=epoch, train_loss=float(loss.detach()), main_loss=float(losses['main'].detach()),
                direct_loss=float(losses['direct'].detach()), test_acc=correct / int(test.sum()), test_nll=nll,
                lr=optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm), seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row)
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **historical.extra_scores(logits[test], y[test]))
                torch.save(dict(state_dict=model.state_dict(), seed=0, fold=fold, mode=MODE, epoch=epoch,
                                direct_weight=plan['direct_weight'], protocol_sha256=sha(OUT / 'PLAN.json')),
                           folder / 'best.pt')
                historical.save_prediction(folder / 'predictions.npz', output, y, test, dd['Index'])
                dump(folder / 'best.json', best)
            if epoch == 1 or epoch % 100 == 0:
                stream.flush()
                status = dict(state='training', seed=0, fold=fold, mode=MODE, epoch=epoch,
                              best_acc=best['acc'], current_acc=row['test_acc'], planned_runs=10,
                              pid=os.getpid(), updated_at=timestamp())
                dump(OUT / 'status.json', status); print(json.dumps(status), flush=True)
            scheduler.step()
    last = dict(epoch=cfg.epochs, **historical.extra_scores(logits[test], y[test]))
    torch.save(dict(state_dict=model.state_dict(), seed=0, fold=fold, mode=MODE, epoch=cfg.epochs,
                    direct_weight=plan['direct_weight'], protocol_sha256=sha(OUT / 'PLAN.json')), folder / 'last.pt')
    historical.save_prediction(folder / 'last_predictions.npz', output, y, test, dd['Index'])
    dump(folder / 'complete.json', dict(seed=0, mode=MODE, fold=fold, best=best, last=last,
        mean_last100_acc=float(np.mean([r['test_acc'] for r in history[-100:]])),
        parameters=model.parameter_counts()['total'], counts=model.parameter_counts(),
        initial_model_sha256=initial_hash, best_checkpoint_sha256=sha(folder / 'best.pt'),
        last_checkpoint_sha256=sha(folder / 'last.pt'), seconds=time.perf_counter()-started,
        peak_mb=torch.cuda.max_memory_allocated()/2**20, completed_at=timestamp()))
    summarize(OUT, plan)
    print(json.dumps(dict(event='arm_complete', fold=fold, mode=MODE,
                          best_acc=best['acc'], last_acc=last['acc'])), flush=True)
    del model, optimizer, scheduler, output, logits, loss, losses
    torch.cuda.empty_cache()


def component_diagnostics(plan):
    old_out, old_mode = residual_round.OUT, residual_round.MODE
    try:
        diagnostics = {}
        for name, base in ((MODE, OUT), (old_mode, old_out)):
            residual_round.OUT, residual_round.MODE = base, name
            # Run the frozen diagnostic without writing into the frozen reference directory.
            original_dump = residual_round.dump
            residual_round.dump = lambda path, value: None
            try:
                diagnostics[name] = residual_round.component_diagnostic(plan)
            finally:
                residual_round.dump = original_dump
        dump(OUT / 'component_diagnostic.json', diagnostics)
        return diagnostics
    finally:
        residual_round.OUT, residual_round.MODE = old_out, old_mode


def prepare():
    historical.SEEDS = (0,)
    cfg, dd, data, original, hashes, baseline = historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        historical.verify_sources(plan); verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['baseline_manifest'] == baseline
        assert plan['direct_weight'] == DIRECT_WEIGHT and plan['seeds'] == [0] and plan['planned_runs'] == 10
        assert plan['environment']['torch'] == torch.__version__
        return cfg, dd, data, plan
    manifest, audit = verify_references(cfg, dd, data, hashes)
    checks = smoke(cfg, dd, data)
    reference_plan = read(residual_round.OUT / 'PLAN.json')
    sources = tuple(dict.fromkeys(tuple(reference_plan['code_sha256']) + NEW_SOURCES))
    plan = dict(experiment='Unchanged cognitive residual model with direct cognitive CE supervision',
        seeds=[0], folds=list(range(1, 11)), modes=[MODE], epochs=1000, planned_runs=10, reused_runs=30,
        class_names=dd['Class_Names'], samples=dd['Sample_Num'], original_config=original['original_config'],
        direct_weight=DIRECT_WEIGHT, objective='CE(final logits) + 0.05 * CE(cognitive direct logits)',
        loss_scope='Both terms use only training-mask labels, original class weights and smoothing .05',
        class_weights=dd['Label_Weight'].detach().cpu().tolist(),
        architecture=reference_plan['architecture'], initialization=reference_plan['initialization'],
        parameter_counts=checks['counts'], priors=reference_plan['priors'], randomness=reference_plan['randomness'],
        training='Original Adam, LR schedule, weight decay and global clipping; same forward and initialization; new direct CE only',
        checkpoint_selection=reference_plan['checkpoint_selection'],
        component_diagnostic='Evaluate direct branch at final-model selected checkpoint and epoch 1000 only; no branch-specific selection',
        interpretation='Direct-branch recovery and improvement in final classification are separate hypotheses; supplement reads both modalities',
        criterion='Primary ACC versus original residual and ordinary COG_PET, macro F1/AD recall and paired correction; late-only gain does not replace primary outcome',
        data_sha256=hashes, baseline_manifest=baseline, reference_manifest=manifest,
        code_sha256={source: sha(ROOT/source) for source in sources}, smoke=checks,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        scope='From scratch end-to-end; one fixed weight, one seed; no extra heads, freezing, gate changes or automatic tuning')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT / 'PLAN.json', plan); dump(OUT / 'reference_audit.json', audit)
    for source in sources:
        dest = OUT / 'source_snapshot' / source
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/source).read_bytes())
    np.savez_compressed(OUT / 'data_and_splits.npz', source_row=np.asarray(dd['Index']),
                        labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([test.cpu().numpy() for _, test in data['Mask']]))
    summarize(OUT, plan); dump(OUT / 'status.json', dict(state='prepared', updated_at=timestamp()))
    return cfg, dd, data, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=10, direct_weight=DIRECT_WEIGHT,
                              smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    historical.OUT = OUT; historical.make_model = make_model
    lock = OUT / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            train_arm(cfg, dd, data, plan, fold)
        dump(OUT / 'status.json', dict(state='auditing', pid=os.getpid(), updated_at=timestamp()))
        audit = historical.audit_completed(cfg, dd, data, plan)
        checked_rows = 0
        for fold in plan['folds']:
            folder = OUT / f'seed_0/{MODE}/fold_{fold:02d}'
            old_folder = residual_round.OUT / f'seed_0/{residual_round.MODE}/fold_{fold:02d}'
            assert read(folder/'complete.json')['initial_model_sha256'] == read(old_folder/'complete.json')['initial_model_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    assert abs(float(row['train_loss'])-float(row['main_loss'])-DIRECT_WEIGHT*float(row['direct_loss'])) < 2e-6
                    checked_rows += 1
        diagnostics = component_diagnostics(plan)
        verify_manifest(plan['reference_manifest'])
        summary = summarize(OUT, plan)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit,
            loss_accounting=dict(passed=True, checked_epochs=checked_rows), summary=summary,
            component_diagnostic=diagnostics))
        dump(OUT/'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
