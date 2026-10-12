"""Complete the retained-protocol information-gain round with unimodal fits.

Reuse audited additive/fusion results without changing them. Inject an independent
unimodal model into the unchanged historical training and replay-audit functions.
No graph or follow-up interaction experiment is scheduled by this runner.
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
from torch.nn import functional as F

import modal_evidence_experiment as historical
from Model.single_modal_evidence import SingleModalEvidenceModel
from relation_experiment import ROOT, dump, sha, timestamp
from Utils import SET_Random

OUT = ROOT / 'Result/unimodal_information_gain_20260923'
REFERENCE = ROOT / 'Result/modal_evidence_fusion_20260922'
NEW_SOURCES = ('Model/single_modal_evidence.py', 'unimodal_round_experiment.py',
               'tests/test_single_modal_evidence.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_model(dd, cfg, mode, seed):
    return SingleModalEvidenceModel(dd, cfg, mode, seed).to(cfg.Device)


def verify_reference(cfg, dd, data, data_hashes):
    plan = read(REFERENCE / 'PLAN.json')
    historical.verify_sources(plan)
    assert plan['data_sha256'] == data_hashes
    assert plan['seeds'] == [0, 1, 2] and plan['folds'] == list(range(1, 11))
    assert plan['epochs'] == cfg.epochs == 1000
    manifest = {str((REFERENCE / 'PLAN.json').relative_to(ROOT)): sha(REFERENCE / 'PLAN.json')}
    verified = []
    for seed in plan['seeds']:
        for mode in ('additive', 'fusion'):
            model = historical.make_model(dd, cfg, mode, seed).eval()
            for fold in plan['folds']:
                folder = REFERENCE / f'seed_{seed}/{mode}/fold_{fold:02d}'
                record = read(folder / 'complete.json')
                assert (record['seed'], record['fold'], record['mode']) == (seed, fold, mode)
                with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 1000
                assert [int(r['epoch']) for r in rows] == list(range(1, 1001))
                accuracies = [float(r['test_acc']) for r in rows]
                assert int(np.argmax(accuracies)) + 1 == record['best']['epoch']
                _, mask = data['Mask'][fold - 1]
                for stage, ckpt_name, pred_name in (
                        ('best', 'best.pt', 'predictions.npz'),
                        ('last', 'last.pt', 'last_predictions.npz')):
                    assert sha(folder / ckpt_name) == record[f'{stage}_checkpoint_sha256']
                    checkpoint = torch.load(folder / ckpt_name, map_location=cfg.Device, weights_only=False)
                    assert checkpoint['protocol_sha256'] == sha(REFERENCE / 'PLAN.json')
                    model.load_state_dict(checkpoint['state_dict'])
                    with torch.no_grad():
                        output = model(data['Feature'])['logits'][mask]
                    with np.load(folder / pred_name) as saved:
                        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[mask.cpu().numpy()])
                        np.testing.assert_array_equal(saved['labels'], data['Label'][mask].cpu().numpy())
                        np.testing.assert_allclose(saved['logits'], output.cpu().numpy(), atol=2e-5, rtol=2e-5)
                    metrics = historical.extra_scores(output, data['Label'][mask])
                    for metric in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                        assert abs(metrics[metric] - record[stage][metric]) < 2e-5
                for name in ('complete.json', 'best.json', 'epochs.csv', 'best.pt', 'last.pt',
                             'predictions.npz', 'last_predictions.npz'):
                    file = folder / name
                    manifest[str(file.relative_to(ROOT))] = sha(file)
                verified.append(dict(seed=seed, mode=mode, fold=fold))
            del model
    return manifest, dict(passed=True, verified_runs=len(verified),
                         verified_checkpoints=len(verified) * 2, records=verified)


def verify_manifest(manifest):
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Frozen reference changed: {name}')


def smoke(cfg, dd, data):
    result = []
    fit, _ = data['Mask'][0]
    for mode in dd['Modal_Name']:
        SET_Random(0)
        model = make_model(dd, cfg, mode, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model(data['Feature'])['logits']
            loss = F.cross_entropy(z[fit], data['Label'][fit], weight=dd['Label_Weight'], label_smoothing=.05)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                assert torch.isfinite(model.eval()(data['Feature'])['logits']).all()
        torch.cuda.synchronize()
        result.append(dict(mode=mode, seconds_per_epoch=(time.perf_counter()-start)/20,
                           counts=model.parameter_counts()))
    return dict(passed=True, runs=result, inspected_test_metrics=False)


def point(folder, stage):
    name = 'predictions.npz' if stage == 'best' else 'last_predictions.npz'
    with np.load(folder / name) as p:
        return {key: p[key].copy() for key in ('logits', 'labels', 'row_index')}


def summarize(out, plan):
    modes = list(plan['modes']) + ['additive', 'fusion']
    records, folders, variants = {}, {}, {}
    for mode in modes:
        records[mode], folders[mode] = {}, {}
        base = REFERENCE if mode in ('additive', 'fusion') else out
        for seed in plan['seeds']:
            for fold in plan['folds']:
                folder = base / f'seed_{seed}/{mode}/fold_{fold:02d}'
                if (folder / 'complete.json').exists():
                    records[mode][seed, fold] = read(folder / 'complete.json')
                    folders[mode][seed, fold] = folder
        per_seed = {}
        for seed in plan['seeds']:
            rows = [v for (s, _), v in records[mode].items() if s == seed]
            if not rows:
                continue
            per_seed[str(seed)] = dict(folds=len(rows))
            for stage in ('best', 'last'):
                per_seed[str(seed)][stage] = {
                    metric: np.mean([r[stage][metric] for r in rows], axis=0).tolist()
                    for metric in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier', 'per_class_recall')}
            per_seed[str(seed)]['mean_last100_acc'] = float(np.mean([r['mean_last100_acc'] for r in rows]))
        item = dict(completed_runs=len(records[mode]), planned_runs=30,
                    reused=mode in ('additive', 'fusion'), per_seed=per_seed)
        if per_seed:
            for stage in ('best', 'last'):
                item[stage] = {metric: np.mean([v[stage][metric] for v in per_seed.values()], axis=0).tolist()
                               for metric in next(iter(per_seed.values()))[stage]}
            item['mean_last100_acc'] = float(np.mean([v['mean_last100_acc'] for v in per_seed.values()]))
        variants[mode] = item
    comparisons = []
    pairs = [(joint, single) for joint in ('additive', 'fusion') for single in plan['modes']]
    pairs.append(('fusion', 'additive'))
    for left, right in pairs:
        keys = sorted(records[left].keys() & records[right].keys())
        if not keys:
            continue
        result = dict(left=left, right=right, matched_runs=len(keys), stages={})
        for stage in ('best', 'last'):
            corrected = harmed = 0
            by_class = np.zeros((len(plan['class_names']), 2), dtype=int)
            deltas = []
            for key in keys:
                a, b = point(folders[left][key], stage), point(folders[right][key], stage)
                np.testing.assert_array_equal(a['row_index'], b['row_index'])
                np.testing.assert_array_equal(a['labels'], b['labels'])
                ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                gain, harm = ca & ~cb, ~ca & cb
                corrected += int(gain.sum())
                harmed += int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c] += [int((gain & (a['labels'] == c)).sum()), int((harm & (a['labels'] == c)).sum())]
                deltas.append(dict(seed=key[0], fold=key[1], acc_delta=float(ca.mean()-cb.mean())))
            seed_deltas = {str(s): float(np.mean([d['acc_delta'] for d in deltas if d['seed'] == s]))
                           for s in sorted({d['seed'] for d in deltas})}
            result['stages'][stage] = dict(mean_acc_delta=float(np.mean(list(seed_deltas.values()))),
                                           per_seed_delta=seed_deltas, corrected=corrected, harmed=harmed,
                                           per_class_corrected_harmed=by_class.tolist())
        comparisons.append(result)
    completed = sum(variants[m]['completed_runs'] for m in plan['modes'])
    summary = dict(updated_at=timestamp(), completed_new_runs=completed, planned_new_runs=180,
                   reused_runs=60, variants=variants, comparisons=comparisons,
                   caveat='First maximum TEST ACC. Counts across seeds are repeated patient predictions, not distinct patients. Partial unmatched means cannot establish rankings.')
    dump(out / 'summary.json', summary)
    lines = ['# 原协议下的单模态与融合比较', '',
             f'新增训练完成 {completed}/180；复用已核验的相加与拼接融合结果 60 次。',
             '固定原 10 折，训练种子 0/1/2，每次 1000 轮；取首次最高测试 ACC。',
             '每组独立从头训练；保留原门控先验、差异化噪声和优化设置。', '',
             '| 模型 | 完成 | 测试择优 ACC | 固定末轮 ACC |', '|---|---:|---:|---:|']
    for mode, v in variants.items():
        if v['completed_runs']:
            lines.append(f"| {mode} | {v['completed_runs']}/30 | {100*v['best']['acc']:.2f}% | {100*v['last']['acc']:.2f}% |")
        else:
            lines.append(f'| {mode} | 0/30 | — | — |')
    lines += ['', '不同完成度的均值不能直接排序。逐种子、同折配对纠错及类别统计见 summary.json。',
              '沿用测试集选模，结果用于探索性比较；不同种子不增加独立患者数量。',
              '历史 C 模型不作为本轮单因素对照。本轮结束后不自动启动新的交互结构实验。']
    (out / '进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def prepare():
    cfg, dd, data, original, hashes, baseline = historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Use the historical tabgsl-gpu Python environment')
    reference_plan = read(REFERENCE / 'PLAN.json')
    if torch.__version__ != reference_plan['environment']['torch']:
        raise RuntimeError('PyTorch differs from the reused experiment')
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        historical.verify_sources(plan)
        verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['baseline_manifest'] == baseline
        return cfg, dd, data, plan
    manifest, audit = verify_reference(cfg, dd, data, hashes)
    checks = smoke(cfg, dd, data)
    sources = tuple(dict.fromkeys(historical.SOURCES + NEW_SOURCES))
    plan = dict(experiment='Independent unimodal fits versus audited additive and ordinary fusion',
                seeds=[0, 1, 2], folds=list(range(1, 11)), modes=dd['Modal_Name'], epochs=1000,
                planned_runs=180, reused_runs=60, class_names=dd['Class_Names'], samples=dd['Sample_Num'],
                modal_dimensions=[len(x) for x in dd['Modal_Index']], original_config=original['original_config'],
                checkpoint_selection=reference_plan['checkpoint_selection'], class_weights=reference_plan['class_weights'],
                data_sha256=hashes, baseline_manifest=baseline, reference_manifest=manifest,
                code_sha256={s: sha(ROOT / s) for s in sources}, smoke=checks,
                model='Pruned single modality from identically initialized additive encoder/head; trained independently',
                randomness='Original full-shape noise/dropout draws; unused feature values have no prediction path',
                training='Unchanged historical.train_arm: Adam, original scheduler, clip, weighted CE with smoothing .05',
                retained_priors=reference_plan['priors'],
                loss_scope='Final classifier CE only, like reused A/F; no OVR, orthogonality, graph, auxiliary loss',
                schedule='fold, seed, modality; same initialization/training seed in each fold',
                environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
                caveat='Historical test-selected protocol retained by user request; exploratory inference only')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT / 'reference_audit.json', audit)
    dump(OUT / 'PLAN.json', plan)
    for source in sources:
        dest = OUT / 'source_snapshot' / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / source).read_bytes())
    np.savez_compressed(OUT / 'data_and_splits.npz', source_row=np.asarray(dd['Index']),
                        labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([t.cpu().numpy() for _, t in data['Mask']]))
    summarize(OUT, plan)
    dump(OUT / 'status.json', dict(state='prepared', updated_at=timestamp()))
    return cfg, dd, data, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=180, reused_runs=60, smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    # Inject only the new model/output/report into the unchanged training loop.
    historical.OUT = OUT
    historical.make_model = make_model
    historical.summarize = summarize
    lock = OUT / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        dump(OUT / 'status.json', dict(state='starting', pid=os.getpid(), updated_at=timestamp()))
        for fold in plan['folds']:
            for seed in plan['seeds']:
                for mode in plan['modes']:
                    historical.train_arm(cfg, dd, data, plan, seed, fold, mode)
        audit = historical.audit_completed(cfg, dd, data, plan)
        verify_manifest(plan['reference_manifest'])
        summary = summarize(OUT, plan)
        dump(OUT / 'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT / 'status.json', dict(state='complete', pid=os.getpid(), updated_at=timestamp()))
    except BaseException as error:
        dump(OUT / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
