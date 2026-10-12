"""One seed-0 auxiliary-supervision arm versus frozen cognitive/PET controls.

Original splits, optimizer, scheduler and first-max-test-ACC selection retained.
Only the training objective gains two unimodal CE heads with total weight 0.1.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch

import modal_evidence_experiment as historical
import cognitive_pair_experiment as pair_round
from Model.aux_subset_fusion import AuxiliarySubsetFusionModel, classification_objective
from relation_experiment import ROOT, dump, sha, timestamp
from unimodal_round_experiment import read, point, verify_manifest
from Utils import SET_Random, CustomCosineAnnealingLR


OUT = ROOT / 'Result/cognitive_pet_aux_seed0_20260923'
REFERENCE = ROOT / 'Result/cognitive_pair_screen_seed0_20260923'
MODE = 'COG_PET_AUX'
AUXILIARY_WEIGHT = .1
NEW_SOURCES = ('Model/aux_subset_fusion.py', 'cognitive_pet_aux_experiment.py',
               'tests/test_aux_subset_fusion.py')


def make_model(dd, cfg, mode, seed):
    if mode != MODE:
        raise ValueError(mode)
    return AuxiliarySubsetFusionModel(dd, cfg, ('COGNITIVE_TEST', 'PET'), seed).to(cfg.Device)


def verify_references(cfg, dd, data, data_hashes):
    plan = read(REFERENCE / 'PLAN.json')
    historical.verify_sources(plan)
    assert plan['seeds'] == [0] and plan['data_sha256'] == data_hashes
    assert plan['epochs'] == cfg.epochs == 1000
    assert plan['environment']['torch'] == torch.__version__
    manifest = {str((REFERENCE / 'PLAN.json').relative_to(ROOT)): sha(REFERENCE / 'PLAN.json')}
    verified = []
    for mode in ('COG', 'COG_PET'):
        model = pair_round.make_model(dd, cfg, mode, 0).eval()
        for fold in range(1, 11):
            folder = REFERENCE / f'seed_0/{mode}/fold_{fold:02d}'
            record = read(folder / 'complete.json')
            assert (record['seed'], record['fold'], record['mode']) == (0, fold, mode)
            with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 1000
            assert [int(r['epoch']) for r in rows] == list(range(1, 1001))
            assert int(np.argmax([float(r['test_acc']) for r in rows])) + 1 == record['best']['epoch']
            _, test = data['Mask'][fold - 1]
            for stage in ('best', 'last'):
                path = folder / f'{stage}.pt'
                assert sha(path) == record[f'{stage}_checkpoint_sha256']
                checkpoint = torch.load(path, map_location=cfg.Device, weights_only=False)
                assert checkpoint['protocol_sha256'] == sha(REFERENCE / 'PLAN.json')
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                with torch.no_grad():
                    logits = model(data['Feature'])['logits'][test]
                saved = point(folder, stage)
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                np.testing.assert_allclose(saved['logits'], logits.cpu().numpy(), rtol=2e-5, atol=2e-5)
                metrics = historical.extra_scores(logits, data['Label'][test])
                for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                    assert abs(metrics[key] - record[stage][key]) < 2e-5
            for file in ('complete.json', 'best.json', 'epochs.csv', 'best.pt', 'last.pt',
                         'predictions.npz', 'last_predictions.npz'):
                path = folder / file
                manifest[str(path.relative_to(ROOT))] = sha(path)
            verified.append(dict(mode=mode, seed=0, fold=fold))
    return manifest, dict(passed=True, seeds=[0], verified_runs=20,
                         verified_checkpoints=40, records=verified)


def smoke(cfg, dd, data):
    plain = pair_round.make_model(dd, cfg, 'COG_PET', 0)
    model = make_model(dd, cfg, MODE, 0)
    for name, state in plain.state_dict().items():
        torch.testing.assert_close(state, model.state_dict()[name], atol=0, rtol=0)
    for training in (False, True):
        plain.train(training); model.train(training)
        SET_Random(807)
        with torch.no_grad():
            a = plain(data['Feature'])
        after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        SET_Random(807)
        with torch.no_grad():
            b = model(data['Feature'])
        torch.testing.assert_close(a['logits'], b['logits'], atol=0, rtol=0)
        assert torch.equal(after[0], torch.get_rng_state())
        assert all(torch.equal(x, y) for x, y in zip(after[1], torch.cuda.get_rng_state_all()))
    SET_Random(0)
    model = make_model(dd, cfg, MODE, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fit, _ = data['Mask'][0]
    start = time.perf_counter()
    for _ in range(20):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = classification_objective(model(data['Feature']), data['Label'], fit,
                                        dd['Label_Weight'], AUXILIARY_WEIGHT)
        loss['total'].backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    return dict(passed=True, inspected_test_metrics=False, matched_initial_backbone=True,
                matched_initial_predictions_and_rng=True, counts=model.parameter_counts(),
                seconds_per_epoch=(time.perf_counter()-start)/20)


def summarize(out, plan):
    records, folders, variants = {}, {}, {}
    for mode in (MODE, 'COG_PET', 'COG'):
        records[mode], folders[mode] = {}, {}
        for fold in plan['folds']:
            base = out if mode == MODE else REFERENCE
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
    for reference in ('COG_PET', 'COG'):
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
                deltas.append(float(ca.mean()-cb.mean()))
            result['stages'][stage] = dict(mean_acc_delta=float(np.mean(deltas)),
                corrected=corrected, harmed=harmed, net_corrected=corrected-harmed,
                per_class_corrected_harmed=by_class.tolist(), fold_acc_deltas=dict(zip(folds, deltas)),
                fold_wins=sum(d > 1e-12 for d in deltas), fold_ties=sum(abs(d) <= 1e-12 for d in deltas),
                fold_losses=sum(d < -1e-12 for d in deltas))
        comparisons.append(result)
    complete = variants[MODE]['completed_runs']
    summary = dict(updated_at=timestamp(), seeds=[0], completed_new_runs=complete,
                   planned_new_runs=10, reused_runs=20, variants=variants, comparisons=comparisons)
    dump(out / 'summary.json', summary)
    lines = ['# 认知＋PET：辅助分类监督对照', '', f'新增训练完成 {complete}/10；历史参照 20 次。',
             '种子 0，原十折、1000 轮、首次最高测试 ACC；最终预测仍由原融合头输出。',
             '总损失 = 融合 CE + 0.1 × 两个单模态 CE 的均值（每个辅助项权重 0.05）。', '',
             '| 模型 | 完成 | 择优 ACC | AD 召回 | 末轮 ACC |', '|---|---:|---:|---:|---:|']
    for mode, v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {mode} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['best']['per_class_recall'][0]*100:.2f}% | {v['last']['acc']*100:.2f}% |")
        else:
            lines.append(f'| {mode} | 0/10 | — | — | — |')
    lines += ['', '不同完成折数不能直接排名；同折纠错和类别统计见 summary.json。',
              '主指标不变；仅末轮改善不能称为主分类效果提高。单种子结果不代表跨种子稳定。',
              '本轮不追加其他权重、模态组合、种子或交互结构。']
    (out / '进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def train_arm(cfg, dd, data, plan, fold):
    """Historical loop, with an explicit auxiliary CE term and loss logging."""
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
    fit, test = data['Mask'][fold-1]
    train_y = torch.where(fit, y, torch.zeros_like(y))
    best_correct, best, history = -1, None, []
    names = list(model.modalities)
    fields = ('epoch', 'train_loss', 'main_loss', 'auxiliary_mean_loss',
              'pet_aux_loss', 'cog_aux_loss', 'test_acc', 'test_nll', 'lr', 'grad_norm', 'seconds')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, cfg.epochs+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(x)
            losses = classification_objective(output, train_y, fit, dd['Label_Weight'], plan['auxiliary_weight'])
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
                nll = float(torch.nn.functional.cross_entropy(logits[test], y[test]))
            row = dict(epoch=epoch, train_loss=float(loss.detach()), main_loss=float(losses['main'].detach()),
                auxiliary_mean_loss=float(losses['auxiliary_mean'].detach()),
                pet_aux_loss=float(losses['auxiliary'][names.index('PET')].detach()),
                cog_aux_loss=float(losses['auxiliary'][names.index('COGNITIVE_TEST')].detach()),
                test_acc=correct/int(test.sum()), test_nll=nll, lr=optimizer.param_groups[0]['lr'],
                grad_norm=float(grad_norm), seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row)
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **historical.extra_scores(logits[test], y[test]))
                torch.save(dict(state_dict=model.state_dict(), seed=0, fold=fold, mode=MODE,
                                epoch=epoch, auxiliary_weight=plan['auxiliary_weight'],
                                protocol_sha256=sha(OUT / 'PLAN.json')), folder / 'best.pt')
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
                    auxiliary_weight=plan['auxiliary_weight'], protocol_sha256=sha(OUT / 'PLAN.json')),
               folder / 'last.pt')
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


def prepare():
    historical.SEEDS = (0,)
    cfg, dd, data, original, hashes, baseline = historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        historical.verify_sources(plan); verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['baseline_manifest'] == baseline
        assert plan['auxiliary_weight'] == AUXILIARY_WEIGHT and plan['seeds'] == [0]
        assert plan['environment']['torch'] == torch.__version__
        return cfg, dd, data, plan
    manifest, audit = verify_references(cfg, dd, data, hashes)
    checks = smoke(cfg, dd, data)
    reference_plan = read(REFERENCE / 'PLAN.json')
    sources = tuple(dict.fromkeys(tuple(reference_plan['code_sha256']) + NEW_SOURCES))
    plan = dict(experiment='Cognitive + PET: unimodal auxiliary CE with unchanged fusion prediction',
        seeds=[0], folds=list(range(1,11)), modes=[MODE], epochs=1000, planned_runs=10, reused_runs=20,
        class_names=dd['Class_Names'], samples=dd['Sample_Num'], original_config=original['original_config'],
        auxiliary_weight=AUXILIARY_WEIGHT, objective='fusion_CE + 0.1 * mean(PET_CE, COG_CE)',
        loss_scope='All three CE terms use train-mask labels, original class weights and smoothing .05',
        class_weights=dd['Label_Weight'].detach().cpu().tolist(),
        head='Unchanged 43-wide Linear-GELU-Linear fusion; new Linear(96,3) per-modal training-only heads',
        inference='Final class logits from original fusion head only; no score summation or ensembling',
        initialization='Backbone and fusion head exactly match COG_PET; new heads seed 4100 + training seed, RNG restored',
        training='Original Adam/lr/weight_decay, clip=1 and cosine scheduler; clip all parameters including auxiliary heads',
        priors=reference_plan['priors'], randomness=reference_plan['randomness'],
        checkpoint_selection=reference_plan['checkpoint_selection'],
        data_sha256=hashes, baseline_manifest=baseline, reference_manifest=manifest,
        code_sha256={source:sha(ROOT/source) for source in sources}, smoke=checks,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        scope='One fixed weight, one seed, one new arm; no automatic tuning or follow-up experiments',
        caveat='Test-selected exploratory single-seed comparison; late-epoch gain alone is not improvement in the primary metric')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT/'PLAN.json', plan); dump(OUT/'reference_audit.json', audit)
    for source in sources:
        dest=OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/source).read_bytes())
    np.savez_compressed(OUT/'data_and_splits.npz', source_row=np.asarray(dd['Index']),
                        labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([t.cpu().numpy() for _,t in data['Mask']]))
    summarize(OUT, plan); dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp()))
    return cfg, dd, data, plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args=parser.parse_args()
    cfg,dd,data,plan=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=10, auxiliary_weight=AUXILIARY_WEIGHT,
                              smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    historical.OUT=OUT; historical.make_model=make_model
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            train_arm(cfg,dd,data,plan,fold)
        audit=historical.audit_completed(cfg,dd,data,plan)
        for fold in plan['folds']:
            with (OUT/f'seed_0/{MODE}/fold_{fold:02d}/epochs.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            for r in rows:
                assert abs(float(r['train_loss'])-float(r['main_loss'])-.1*float(r['auxiliary_mean_loss']))<2e-6
                assert abs(float(r['auxiliary_mean_loss'])-.5*(float(r['pet_aux_loss'])+float(r['cog_aux_loss'])))<2e-6
        verify_manifest(plan['reference_manifest'])
        summary=summarize(OUT,plan)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,loss_accounting_passed=True,summary=summary))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',error=repr(error),pid=os.getpid(),updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
