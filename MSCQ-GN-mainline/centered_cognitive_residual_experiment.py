"""Fixed seed-0 null-token-centered supplement with unchanged direct CE."""
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
import cognitive_direct_supervision_experiment as previous
from Model.centered_cognitive_residual import CenteredCognitiveResidualModel
from Model.cognitive_residual import CognitiveResidualModel
from Model.cognitive_direct_supervision import classification_objective
from relation_experiment import ROOT, dump, sha, timestamp
from unimodal_round_experiment import read, point, verify_manifest
from Utils import SET_Random


OUT = ROOT / 'Result/centered_cognitive_residual_seed0_20260923'
MODE = 'COG_PET_CENTERED_DIRECT_CE'
CONTROL_MODE, CONTROL_OUT = previous.MODE, previous.OUT
REFERENCES = {CONTROL_MODE: CONTROL_OUT, **previous.REFERENCES}
DIRECT_WEIGHT = .05
NEW_SOURCES = ('Model/centered_cognitive_residual.py',
               'centered_cognitive_residual_experiment.py',
               'tests/test_centered_cognitive_residual.py')


def make_model(dd, cfg, mode, seed):
    if mode != MODE:
        raise ValueError(mode)
    return CenteredCognitiveResidualModel(dd, cfg, 'PET', seed).to(cfg.Device)


def verify_references(cfg, dd, data, hashes):
    manifest, old_audit = previous.verify_references(cfg, dd, data, hashes)
    plan = read(CONTROL_OUT / 'PLAN.json')
    historical.verify_sources(plan); verify_manifest(plan['reference_manifest'])
    assert plan['seeds'] == [0] and plan['data_sha256'] == hashes
    assert plan['direct_weight'] == DIRECT_WEIGHT and plan['epochs'] == cfg.epochs == 1000
    assert plan['environment']['torch'] == torch.__version__
    manifest[str((CONTROL_OUT/'PLAN.json').relative_to(ROOT))] = sha(CONTROL_OUT/'PLAN.json')
    model = CognitiveResidualModel(dd, cfg, 'PET', 0).to(cfg.Device).eval()
    candidate = make_model(dd, cfg, MODE, 0).eval()
    initial_hash = historical.state_hash(model)
    assert historical.state_hash(candidate) == initial_hash
    verified = []
    for fold in range(1, 11):
        folder = CONTROL_OUT / f'seed_0/{CONTROL_MODE}/fold_{fold:02d}'
        record = read(folder/'complete.json')
        assert (record['seed'], record['fold'], record['mode']) == (0, fold, CONTROL_MODE)
        assert record['initial_model_sha256'] == initial_hash
        assert record['parameters'] == candidate.parameter_counts()['total'] == 40900
        with (folder/'epochs.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1000 and [int(r['epoch']) for r in rows] == list(range(1, 1001))
        assert int(np.argmax([float(r['test_acc']) for r in rows]))+1 == record['best']['epoch']
        _, test = data['Mask'][fold-1]
        errors = {}
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            path = folder/f'{stage}.pt'
            assert sha(path) == record[f'{stage}_checkpoint_sha256']
            checkpoint = torch.load(path, map_location=cfg.Device, weights_only=False)
            assert checkpoint['protocol_sha256'] == sha(CONTROL_OUT/'PLAN.json')
            assert checkpoint['direct_weight'] == DIRECT_WEIGHT
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with torch.no_grad():
                output = model(data['Feature'])
            with np.load(folder/filename) as saved:
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                for key in ('logits', 'unimodal_logits', 'pair_logits'):
                    np.testing.assert_allclose(saved[key], output[key][test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                errors[stage] = float(np.max(np.abs(saved['logits']-output['logits'][test].cpu().numpy())))
            metrics = historical.extra_scores(output['logits'][test], data['Label'][test])
            for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                assert abs(metrics[key]-record[stage][key]) < 2e-5
        for filename in ('complete.json', 'best.json', 'epochs.csv', 'best.pt', 'last.pt',
                         'predictions.npz', 'last_predictions.npz'):
            path = folder/filename
            manifest[str(path.relative_to(ROOT))] = sha(path)
        verified.append(dict(mode=CONTROL_MODE, seed=0, fold=fold, max_abs_replay_error=errors))
    return manifest, dict(passed=True, seeds=[0], verified_runs=40, verified_checkpoints=80,
                          earlier_references=old_audit, direct_supervision_records=verified,
                          identical_initial_state_and_nominal_parameter_count=True)


def smoke(cfg, dd, data):
    original = CognitiveResidualModel(dd, cfg, 'PET', 0).to(cfg.Device)
    model = make_model(dd, cfg, MODE, 0)
    assert historical.state_hash(original) == historical.state_hash(model)
    for training in (False, True):
        original.train(training); model.train(training)
        SET_Random(3701)
        with torch.no_grad():
            expected = original(data['Feature'])
        after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        SET_Random(3701)
        with torch.no_grad():
            actual = model(data['Feature'])
        for key in ('logits', 'tokens', 'unimodal_logits', 'pair_logits'):
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
        loss = classification_objective(output, data['Label'], fit, dd['Label_Weight'], DIRECT_WEIGHT)['total']
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert not torch.count_nonzero(model.head[-1].bias.grad)
        if step > 0:
            assert model.head[0].weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(data['Feature'])
        torch.testing.assert_close(output['pair_logits'][:, 0],
                                   output['raw_joint_logits']-output['reference_joint_logits'], atol=0, rtol=0)
        null = output['tokens'].clone(); null[:, model.companion_index] = 0
        assert not torch.count_nonzero(model.supplement_from_tokens(null))
        assert not torch.count_nonzero(model.head[-1].bias)
    torch.cuda.synchronize()
    return dict(passed=True, inspected_test_metrics=False, same_initial_state_and_predictions=True,
                matched_tokens_and_rng=True, null_companion_supplement_exactly_zero=True,
                canceled_output_bias_stays_zero=True, counts=model.parameter_counts(),
                seconds_per_step=(time.perf_counter()-start)/20)


def summarize(out, plan):
    records, folders, variants = {}, {}, {}
    for mode in (MODE, *REFERENCES):
        records[mode], folders[mode] = {}, {}
        base = out if mode == MODE else REFERENCES[mode]
        for fold in plan['folds']:
            folder = base/f'seed_0/{mode}/fold_{fold:02d}'
            if (folder/'complete.json').exists():
                records[mode][fold] = read(folder/'complete.json'); folders[mode][fold] = folder
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
            corrected = harmed = 0
            by_class = np.zeros((len(plan['class_names']), 2), dtype=int)
            deltas = []
            for fold in folds:
                a, b = point(folders[MODE][fold], stage), point(folders[reference][fold], stage)
                np.testing.assert_array_equal(a['row_index'], b['row_index'])
                np.testing.assert_array_equal(a['labels'], b['labels'])
                ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                gain, harm = ca & ~cb, ~ca & cb
                corrected += int(gain.sum()); harmed += int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c] += [int((gain & (a['labels']==c)).sum()), int((harm & (a['labels']==c)).sum())]
                deltas.append(float(ca.mean()-cb.mean()))
            result['stages'][stage] = dict(mean_acc_delta=float(np.mean(deltas)), corrected=corrected,
                harmed=harmed, net_corrected=corrected-harmed, per_class_corrected_harmed=by_class.tolist(),
                fold_acc_deltas=dict(zip(folds, deltas)), fold_wins=sum(d>1e-12 for d in deltas),
                fold_ties=sum(abs(d)<=1e-12 for d in deltas), fold_losses=sum(d< -1e-12 for d in deltas))
        comparisons.append(result)
    complete = variants[MODE]['completed_runs']
    summary = dict(updated_at=timestamp(), seeds=[0], completed_new_runs=complete, planned_new_runs=10,
                   reused_runs=40, variants=variants, comparisons=comparisons)
    dump(out/'summary.json', summary)
    labels = {MODE:'零 PET 表示参照补充', CONTROL_MODE:'上一轮认知直接监督',
              'COG_PET_RESIDUAL':'原残差模型', 'COG_PET':'普通认知＋PET 融合', 'COG':'认知单模态'}
    lines = ['# 以 PET 零表示为参照的补充分支', '', f'新增训练完成 {complete}/10；复用 40 次历史训练。',
             '种子 0，原十折、1000 轮、首次最高测试 ACC；认知分支监督权重仍为 0.05。',
             '补充分数 = g(认知表示, PET 表示) − g(认知表示, 0)，两项都反向传播且复用同一次噪声和 dropout。', '',
             '| 模型 | 完成 | 最佳 ACC | 宏 F1 | AD 召回 | 末轮 ACC |', '|---|---:|---:|---:|---:|---:|']
    for mode, v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {labels[mode]} | {v['completed_runs']}/10 | {v['best']['acc']*100:.2f}% | {v['best']['macro_f1']*100:.2f}% | {v['best']['per_class_recall'][0]*100:.2f}% | {v['last']['acc']*100:.2f}% |")
        else:
            lines.append(f'| {labels[mode]} | 0/10 | — | — | — | — |')
    lines += ['', '不同完成折数不能直接排名；配对纠错见 summary.json。',
              '零参照定义在编码后的模态表示，不是原始 PET 特征；不等同于真实患者或缺失模态。',
              '只保证零 PET 表示时补充为零，不证明纯跨模态交互或 PET 的因果贡献。',
              '名义参数 40900 不变，联合头末层的 3 个偏置在差分中抵消；运算量增加。',
              '本轮不自动追加超参数、其他模态或种子。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def component_diagnostics(cfg, dd, data, plan):
    result = dict(class_names=plan['class_names'], description='Fixed final-model checkpoints; no branch-specific selection', variants={})
    for mode, base in ((MODE, OUT), (CONTROL_MODE, CONTROL_OUT)):
        centered = mode == MODE
        model = (make_model(dd, cfg, MODE, 0) if centered else CognitiveResidualModel(dd, cfg, 'PET', 0).to(cfg.Device)).eval()
        stages = {}
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            full_metrics, direct_metrics, null_metrics = [], [], []
            corrected = harmed = 0
            by_class = np.zeros((len(plan['class_names']), 2), dtype=int)
            pred_counts = np.zeros(len(plan['class_names']), dtype=int)
            rows = []
            for fold in plan['folds']:
                folder = base/f'seed_0/{mode}/fold_{fold:02d}'
                checkpoint = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                _, test = data['Mask'][fold-1]
                with torch.no_grad():
                    output = model(data['Feature'])
                    null = output['tokens'].clone(); null[:, model.modalities.index('PET')] = 0
                    direct = output['unimodal_logits'][:, 0]
                    if centered:
                        null_supplement = model.supplement_from_tokens(null)
                        assert not torch.count_nonzero(null_supplement)
                        assert not torch.count_nonzero(model.head[-1].bias)
                        torch.testing.assert_close(output['pair_logits'][:, 0],
                            output['raw_joint_logits']-output['reference_joint_logits'], atol=0, rtol=0)
                    else:
                        null_supplement = model.head(null.flatten(start_dim=1))
                    null_final = direct + null_supplement
                with np.load(folder/filename) as saved:
                    full, dc, y = saved['logits'], saved['unimodal_logits'][:, 0], saved['labels']
                    np.testing.assert_allclose(full, output['logits'][test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(full, dc+saved['pair_logits'][:, 0], atol=2e-5, rtol=2e-5)
                fm = historical.extra_scores(torch.tensor(full), torch.tensor(y))
                dm = historical.extra_scores(torch.tensor(dc), torch.tensor(y))
                nm = historical.extra_scores(null_final[test], data['Label'][test])
                full_metrics.append(fm); direct_metrics.append(dm); null_metrics.append(nm)
                cf, cd = full.argmax(-1)==y, dc.argmax(-1)==y
                gain, harm = cf & ~cd, ~cf & cd
                corrected += int(gain.sum()); harmed += int(harm.sum())
                pred_counts += np.bincount(dc.argmax(-1), minlength=len(pred_counts))
                for c in range(len(by_class)):
                    by_class[c] += [int((gain & (y==c)).sum()), int((harm & (y==c)).sum())]
                rows.append(dict(fold=fold, epoch=checkpoint['epoch'], full_acc=fm['acc'],
                    direct_acc=dm['acc'], null_pet_token_final_acc=nm['acc'], corrected=int(gain.sum()), harmed=int(harm.sum())))
            aggregate = lambda values: {key:np.mean([v[key] for v in values], axis=0).tolist()
                                       for key in ('acc', 'macro_f1', 'per_class_recall')}
            stages[stage] = dict(full=aggregate(full_metrics), direct_only=aggregate(direct_metrics),
                null_pet_token_final=aggregate(null_metrics), corrected=corrected, harmed=harmed,
                net_corrected=corrected-harmed, per_class_corrected_harmed=by_class.tolist(),
                direct_prediction_counts=pred_counts.tolist(), folds=rows)
        result['variants'][mode] = stages
    dump(OUT/'component_diagnostic.json', result)
    return result


def prepare():
    historical.SEEDS = (0,)
    cfg, dd, data, original, hashes, baseline = historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        historical.verify_sources(plan); verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['baseline_manifest'] == baseline
        assert plan['seeds'] == [0] and plan['planned_runs'] == 10 and plan['direct_weight'] == DIRECT_WEIGHT
        assert plan['environment']['torch'] == torch.__version__
        return cfg, dd, data, plan
    manifest, audit = verify_references(cfg, dd, data, hashes)
    checks = smoke(cfg, dd, data)
    old_plan = read(CONTROL_OUT/'PLAN.json')
    sources = tuple(dict.fromkeys(tuple(old_plan['code_sha256']) + NEW_SOURCES))
    plan = dict(experiment='Null PET-token-centered joint supplement with unchanged direct supervision',
        seeds=[0], folds=list(range(1, 11)), modes=[MODE], epochs=1000, planned_runs=10, reused_runs=40,
        class_names=dd['Class_Names'], samples=dd['Sample_Num'], original_config=original['original_config'],
        direct_weight=DIRECT_WEIGHT, objective=old_plan['objective'], loss_scope=old_plan['loss_scope'],
        class_weights=old_plan['class_weights'],
        architecture='Same residual parameters; final = cognitive_direct + g(c,p)-g(c,0)',
        null_reference='Zero encoded companion token after normalization/gate/dropout; same cognitive token and shared deterministic head, both calls differentiated',
        initialization='Exactly same state_dict as existing direct-supervised residual; zero last joint layer gives same initial outputs',
        parameter_counts=checks['counts'], canceled_parameter_count=3,
        parameter_caveat='40900 nominal parameters retained; three final joint biases cancel, effective dependency changes; additional head computation',
        priors=old_plan['priors'], randomness=old_plan['randomness'],
        training='Unmodified previous.train_arm with current factory/output/summary set only in this process; same optimizer, clipping, scheduler, objective .05',
        checkpoint_selection=old_plan['checkpoint_selection'],
        component_diagnostic='Direct and zero-PET-token full predictions at jointly selected best and epoch 1000; no additional selection',
        interpretation='Boundary condition r(c,0)=0, not pure interaction or causal PET contribution; nonzero constant PET embeddings may still carry cognitive-only dependence',
        criterion='Primary ACC and paired net correction versus direct-supervised residual and ordinary fusion; branch recovery or late-only improvement alone is insufficient',
        data_sha256=hashes, baseline_manifest=baseline, reference_manifest=manifest,
        code_sha256={source:sha(ROOT/source) for source in sources}, smoke=checks,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        scope='One new arm, seed 0, fixed weight; no tuning, freezing, extra modality combinations or further seeds')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT/'PLAN.json', plan); dump(OUT/'reference_audit.json', audit)
    for source in sources:
        dest = OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/source).read_bytes())
    np.savez_compressed(OUT/'data_and_splits.npz', source_row=np.asarray(dd['Index']), labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([test.cpu().numpy() for _, test in data['Mask']]))
    summarize(OUT, plan); dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp()))
    return cfg, dd, data, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=10, smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    # Reuse the exact prior training loop; frozen source files and old results are untouched.
    previous.OUT = OUT; previous.MODE = MODE; previous.make_model = make_model; previous.summarize = summarize
    historical.OUT = OUT; historical.make_model = make_model
    lock = OUT/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for fold in plan['folds']:
            previous.train_arm(cfg, dd, data, plan, fold)
        dump(OUT/'status.json', dict(state='auditing', pid=os.getpid(), updated_at=timestamp()))
        audit = historical.audit_completed(cfg, dd, data, plan)
        checked = 0
        for fold in plan['folds']:
            folder = OUT/f'seed_0/{MODE}/fold_{fold:02d}'
            control_folder = CONTROL_OUT/f'seed_0/{CONTROL_MODE}/fold_{fold:02d}'
            assert read(folder/'complete.json')['initial_model_sha256'] == read(control_folder/'complete.json')['initial_model_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    assert abs(float(row['train_loss'])-float(row['main_loss'])-DIRECT_WEIGHT*float(row['direct_loss'])) < 2e-6
                    checked += 1
        diagnostics = component_diagnostics(cfg, dd, data, plan)
        verify_manifest(plan['reference_manifest'])
        summary = summarize(OUT, plan)
        dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), audit=audit,
             loss_accounting=dict(passed=True, checked_epochs=checked), component_diagnostic=diagnostics, summary=summary))
        dump(OUT/'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp()))
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
