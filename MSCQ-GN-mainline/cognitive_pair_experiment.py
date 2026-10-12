"""Seed-0 cognitive-plus-one-modality screening, retaining the original protocol.

Six arms, ten original folds, 1000 epochs; immutable plan and replay audit.
No auxiliary supervision, interaction selector, or automatic follow-up training.
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
from Model.subset_fusion import SubsetFusionModel
from Model.single_modal_evidence import SingleModalEvidenceModel
from relation_experiment import ROOT, dump, sha, timestamp
from unimodal_round_experiment import read, point, verify_manifest
from Utils import SET_Random


OUT = ROOT / 'Result/cognitive_pair_screen_seed0_20260923'
ARMS = {
    'COG': ('COGNITIVE_TEST',),
    'COG_MRI': ('COGNITIVE_TEST', 'MRI'),
    'COG_PET': ('COGNITIVE_TEST', 'PET'),
    'COG_CSF': ('COGNITIVE_TEST', 'CSF'),
    'COG_RISK': ('COGNITIVE_TEST', 'RISK_FACTOR'),
    'COG_ROI': ('COGNITIVE_TEST', 'ROI_AVERAGE'),
}
REFERENCES = {
    'COG_linear_reference': ('unimodal_information_gain_20260923', 'COGNITIVE_TEST'),
    'full_additive_reference': ('modal_evidence_fusion_20260922', 'additive'),
    'full_fusion_reference': ('modal_evidence_fusion_20260922', 'fusion'),
}
NEW_SOURCES = ('Model/subset_fusion.py', 'cognitive_pair_experiment.py',
               'tests/test_subset_fusion.py', 'Model/single_modal_evidence.py',
               'unimodal_round_experiment.py')


def make_model(dd, cfg, mode, seed):
    return SubsetFusionModel(dd, cfg, ARMS[mode], seed).to(cfg.Device)


def reference_folder(name, fold):
    directory, mode = REFERENCES[name]
    return ROOT / 'Result' / directory / f'seed_0/{mode}/fold_{fold:02d}'


def verify_references(cfg, dd, data, hashes):
    manifest, verified = {}, []
    for name, (directory, mode) in REFERENCES.items():
        base = ROOT / 'Result' / directory
        plan = read(base / 'PLAN.json')
        historical.verify_sources(plan)
        assert plan['data_sha256'] == hashes
        assert plan['epochs'] == cfg.epochs == 1000
        assert plan['environment']['torch'] == torch.__version__
        manifest[str((base / 'PLAN.json').relative_to(ROOT))] = sha(base / 'PLAN.json')
        if mode == 'COGNITIVE_TEST':
            model = SingleModalEvidenceModel(dd, cfg, mode, 0).to(cfg.Device).eval()
        else:
            model = historical.make_model(dd, cfg, mode, 0).eval()
        for fold in range(1, 11):
            folder = reference_folder(name, fold)
            record = read(folder / 'complete.json')
            assert (record['seed'], record['fold'], record['mode']) == (0, fold, mode)
            with (folder / 'epochs.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 1000
            assert [int(r['epoch']) for r in rows] == list(range(1, 1001))
            assert int(np.argmax([float(r['test_acc']) for r in rows])) + 1 == record['best']['epoch']
            _, test = data['Mask'][fold - 1]
            for stage, checkpoint_name in (('best', 'best.pt'), ('last', 'last.pt')):
                assert sha(folder / checkpoint_name) == record[f'{stage}_checkpoint_sha256']
                checkpoint = torch.load(folder / checkpoint_name, map_location=cfg.Device, weights_only=False)
                assert checkpoint['protocol_sha256'] == sha(base / 'PLAN.json')
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                with torch.no_grad():
                    logits = model(data['Feature'])['logits'][test]
                prediction = point(folder, stage)
                np.testing.assert_array_equal(prediction['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(prediction['labels'], data['Label'][test].cpu().numpy())
                np.testing.assert_allclose(prediction['logits'], logits.cpu().numpy(), atol=2e-5, rtol=2e-5)
                metrics = historical.extra_scores(logits, data['Label'][test])
                for key in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier'):
                    assert abs(metrics[key] - record[stage][key]) < 2e-5
            for file in ('complete.json', 'best.json', 'epochs.csv', 'best.pt', 'last.pt',
                         'predictions.npz', 'last_predictions.npz'):
                path = folder / file
                manifest[str(path.relative_to(ROOT))] = sha(path)
            verified.append(dict(reference=name, seed=0, fold=fold))
        del model
    return manifest, dict(passed=True, seeds=[0], verified_runs=len(verified),
                         verified_checkpoints=2 * len(verified), records=verified)


def smoke(cfg, dd, data):
    runs = []
    fit, _ = data['Mask'][0]
    for arm in ARMS:
        SET_Random(0)
        model = make_model(dd, cfg, arm, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(data['Feature'])['logits'][fit], data['Label'][fit],
                                   weight=dd['Label_Weight'], label_smoothing=.05)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                assert torch.isfinite(model.eval()(data['Feature'])['logits']).all()
        torch.cuda.synchronize()
        runs.append(dict(mode=arm, seconds_per_epoch=(time.perf_counter()-start)/20,
                         counts=model.parameter_counts()))
        del model, optimizer, loss
    return dict(passed=True, inspected_test_metrics=False, runs=runs)


def summarize(out, plan):
    records, folders, variants = {}, {}, {}
    for mode in list(ARMS) + list(REFERENCES):
        records[mode], folders[mode] = {}, {}
        for fold in plan['folds']:
            folder = (reference_folder(mode, fold) if mode in REFERENCES
                      else out / f'seed_0/{mode}/fold_{fold:02d}')
            if (folder / 'complete.json').exists():
                records[mode][fold] = read(folder / 'complete.json')
                folders[mode][fold] = folder
        rows = list(records[mode].values())
        item = dict(completed_runs=len(rows), planned_runs=10, seed=0, reused=mode in REFERENCES)
        if rows:
            for stage in ('best', 'last'):
                item[stage] = {metric: np.mean([r[stage][metric] for r in rows], axis=0).tolist()
                               for metric in ('acc', 'auc', 'f1', 'macro_f1', 'nll', 'brier', 'per_class_recall')}
            item['mean_last100_acc'] = float(np.mean([r['mean_last100_acc'] for r in rows]))
            item['parameters'] = rows[0]['parameters']
        variants[mode] = item
    pairs = [(mode, 'COG') for mode in ARMS if mode != 'COG']
    pairs += [('COG', 'COG_linear_reference'), ('full_fusion_reference', 'COG')]
    comparisons = []
    for left, right in pairs:
        folds = sorted(records[left].keys() & records[right].keys())
        if not folds:
            continue
        comparison = dict(left=left, right=right, matched_runs=len(folds), stages={})
        for stage in ('best', 'last'):
            by_class = np.zeros((len(plan['class_names']), 2), dtype=int)
            deltas, corrected, harmed = [], 0, 0
            for fold in folds:
                a, b = point(folders[left][fold], stage), point(folders[right][fold], stage)
                np.testing.assert_array_equal(a['row_index'], b['row_index'])
                np.testing.assert_array_equal(a['labels'], b['labels'])
                ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                gain, harm = ca & ~cb, ~ca & cb
                corrected += int(gain.sum()); harmed += int(harm.sum())
                for c in range(len(by_class)):
                    by_class[c] += [int((gain & (a['labels'] == c)).sum()), int((harm & (a['labels'] == c)).sum())]
                deltas.append(float(ca.mean() - cb.mean()))
            comparison['stages'][stage] = dict(mean_acc_delta=float(np.mean(deltas)),
                corrected=corrected, harmed=harmed, net_corrected=corrected-harmed,
                per_class_corrected_harmed=by_class.tolist(), fold_acc_deltas=dict(zip(folds, deltas)),
                fold_wins=sum(d > 1e-12 for d in deltas), fold_ties=sum(abs(d) <= 1e-12 for d in deltas),
                fold_losses=sum(d < -1e-12 for d in deltas))
        comparisons.append(comparison)
    completed = sum(variants[m]['completed_runs'] for m in ARMS)
    summary = dict(updated_at=timestamp(), seeds=[0], completed_new_runs=completed, planned_new_runs=60,
                   reused_runs=30, variants=variants, comparisons=comparisons,
                   caveat='Single-seed exploratory screening under first-best TEST ACC; no cross-seed stability claim. Unequal completed folds cannot be ranked.')
    dump(out / 'summary.json', summary)
    lines = ['# 认知加单个模态：种子 0 筛查', '', f'新增训练完成 {completed}/60；参照仅使用种子 0 的 30 次历史训练。',
             '原十折、1000 轮、首次最高测试 ACC；普通拼接，只监督最终分类输出。',
             '分类头深度和隐藏宽度统一，输入维度及有效参数量不同；无辅助监督或条件门控。', '',
             '| 模型 | 完成 | 择优 ACC | 末轮 ACC | 参数量 |', '|---|---:|---:|---:|---:|']
    for mode, item in variants.items():
        if item['completed_runs']:
            lines.append(f"| {mode} | {item['completed_runs']}/10 | {100*item['best']['acc']:.2f}% | {100*item['last']['acc']:.2f}% | {item['parameters']} |")
        else:
            lines.append(f'| {mode} | 0/10 | — | — | — |')
    lines += ['', '不同完成折数的均值不能直接排名；summary.json 提供同折配对纠错和类别统计。',
              '本轮用于筛查，单种子不代表稳定性；选中的组合仍需后续验证。结束后不会自动扩展种子或启动交互模型。']
    (out / '进度与结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def prepare():
    # Restrict baseline verification and all result comparisons to seed 0.
    historical.SEEDS = (0,)
    cfg, dd, data, original, hashes, baseline = historical.verified_setup()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires historical tabgsl-gpu runtime')
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        historical.verify_sources(plan); verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256'] == hashes and plan['baseline_manifest'] == baseline
        assert plan['seeds'] == [0] and plan['planned_runs'] == 60
        assert plan['environment']['torch'] == torch.__version__
        return cfg, dd, data, plan
    manifest, audit = verify_references(cfg, dd, data, hashes)
    checks = smoke(cfg, dd, data)
    widths = {r['counts']['fusion_width'] for r in checks['runs']}
    assert len(widths) == 1
    sources = tuple(dict.fromkeys(historical.SOURCES + NEW_SOURCES))
    plan = dict(experiment='Cognitive plus one modality: fixed-head-width seed-0 screening',
        seeds=[0], folds=list(range(1, 11)), modes=list(ARMS), arms=ARMS,
        epochs=1000, planned_runs=60, reused_runs=30, samples=dd['Sample_Num'],
        class_names=dd['Class_Names'], original_config=original['original_config'],
        data_sha256=hashes, baseline_manifest=baseline, reference_manifest=manifest,
        code_sha256={source: sha(ROOT / source) for source in sources}, smoke=checks,
        checkpoint_selection='First maximum TEST ACC, unchanged exploratory protocol',
        class_weights=dd['Label_Weight'].detach().cpu().tolist(),
        head=dict(structure='Linear-GELU-Linear', width=next(iter(widths)),
                  input='H or 2H, canonical modality order',
                  initialization='Slice selected columns from historical full-fusion head; shared biases/output layer identical',
                  capacity='Same depth and hidden width, unequal active parameter counts explicitly reported'),
        encoder='Only selected encoders/feature gains/biases/IDs/gates; original shared LayerNorm',
        randomness='Full-shape noise and dropout draws retained; excluded feature values cannot affect predictions',
        training='Unchanged historical.train_arm: Adam, scheduler, clip, weighted CE with smoothing .05',
        loss='Final CE only; no auxiliary heads, graph, interaction selector, or new regularizer',
        priors='Exact historical modality gate logits and differentiated input noise',
        schedule='fold then arm, training seed 0 reset for every run',
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        caveat='Screening only; no multi-seed stability or causal-information claim. No automatic follow-up runs.')
    OUT.mkdir(parents=True, exist_ok=False)
    dump(OUT / 'PLAN.json', plan); dump(OUT / 'reference_audit.json', audit)
    for source in sources:
        destination = OUT / 'source_snapshot' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    np.savez_compressed(OUT / 'data_and_splits.npz', source_row=np.asarray(dd['Index']),
                        labels=data['Label'].cpu().numpy(),
                        test_masks=np.stack([test.cpu().numpy() for _, test in data['Mask']]))
    summarize(OUT, plan)
    dump(OUT / 'status.json', dict(state='prepared', updated_at=timestamp()))
    return cfg, dd, data, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, plan = prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared', planned_runs=60, reused_runs=30, smoke=plan['smoke']), ensure_ascii=False), flush=True)
        return
    historical.OUT = OUT
    historical.make_model = make_model
    historical.summarize = summarize
    lock = OUT / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        dump(OUT / 'status.json', dict(state='starting', pid=os.getpid(), updated_at=timestamp()))
        for fold in plan['folds']:
            for mode in plan['modes']:
                historical.train_arm(cfg, dd, data, plan, 0, fold, mode)
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
