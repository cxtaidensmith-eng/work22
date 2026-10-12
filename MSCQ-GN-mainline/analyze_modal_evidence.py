"""Audit saved predictions and summarize the modal-evidence model comparison.

This module never trains models or changes historical baseline artifacts.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from relation_experiment import ROOT, dump, timestamp


NAMES = {'additive': 'A：单模态证据相加', 'pairwise': 'B：加入模态对补充',
         'fusion': 'F：普通融合', 'content': 'C：历史患者图模型'}
PAIRS = [('pairwise', 'additive'), ('pairwise', 'fusion'),
         ('additive', 'content'), ('pairwise', 'content'), ('fusion', 'content')]
BASE = ROOT / 'Result/relation_mechanism_nodecay'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prediction(folder, stage, metadata, historical, fold):
    path = folder / ('predictions.npz' if stage == 'best' else 'last_predictions.npz')
    with np.load(path, allow_pickle=False) as saved:
        z, y, row = (saved[k].copy() for k in ('logits', 'labels', 'row_index'))
    if z.ndim != 2 or z.shape[0] != len(y) or len(y) != len(row) or not np.isfinite(z).all():
        raise ValueError(f'Invalid prediction arrays: {path}')
    mask = historical['test_masks'][fold - 1]
    if not np.array_equal(row, historical['source_row'][mask]):
        raise ValueError(f'Historical patient order mismatch: {path}')
    if not np.array_equal(y, historical['labels'][mask]):
        raise ValueError(f'Historical labels mismatch: {path}')
    if len(np.unique(row)) != len(row) or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError(f'Invalid patient IDs or labels: {path}')
    pred = z.argmax(-1)
    correct = pred == y
    recall = [float((pred[y == c] == c).mean()) if (y == c).any() else 0.
              for c in range(z.shape[1])]
    if not np.isclose(float(correct.mean()), metadata['acc'], rtol=0, atol=1e-8):
        raise ValueError(f'Saved ACC mismatch: {path}')
    if 'per_class_recall' in metadata and not np.allclose(
            recall, metadata['per_class_recall'], rtol=0, atol=1e-8):
        raise ValueError(f'Saved recall mismatch: {path}')
    return dict(labels=y, row_index=row, correct=correct, acc=float(correct.mean()),
                per_class_recall=recall, path=str(path), sha256=_sha(path))


def _hierarchical(rows, key):
    """Equal folds within seed, then equal completed seeds; never patient pooling."""
    seeds = sorted({r['seed'] for r in rows})
    return np.mean([np.mean([r[key] for r in rows if r['seed'] == s], axis=0)
                    for s in seeds], axis=0).tolist() if seeds else None


def _baseline(plan, historical):
    records, points, entries, protocols = {}, {}, [], {}
    source_cache = {}
    for seed in plan['seeds']:
        parent = BASE / f'seed_{seed}'
        protocol_path = parent / 'PROTOCOL.json'
        protocol_error = None
        try:
            p = _read(protocol_path)
            protocols[str(seed)] = dict(path=str(protocol_path), sha256=_sha(protocol_path),
                                       epochs=p['epochs'], training_seed=p['training_seed'])
            if p['training_seed'] != seed or p['split_seed'] != 0:
                raise ValueError('Baseline seed mismatch')
            for rel, expected in p['code_sha256'].items():
                source_cache.setdefault(rel, _sha(ROOT / rel))
                if source_cache[rel] != expected:
                    raise ValueError(f'Baseline source changed: {rel}')
            for name, expected in p['data_sha256'].items():
                if _sha(ROOT / 'DATASET/TADPOLE' / name) != expected:
                    raise ValueError(f'Baseline data changed: {name}')
        except (OSError, ValueError, KeyError) as e:
            protocol_error = str(e)
        for fold in plan['folds']:
            folder = parent / 'content' / f'fold_{fold:02d}'
            entry = dict(seed=seed, fold=fold, folder=str(folder), best_ok=False, last_ok=False)
            if protocol_error:
                entry['error'] = protocol_error
                entries.append(entry)
                continue
            try:
                record = _read(folder / 'complete.json')
                best = _read(folder / 'best.json')
                if (record['seed'], record['fold'], record['mode']) != (seed, fold, 'content'):
                    raise ValueError('Baseline complete.json identity mismatch')
                if best != record['best']:
                    raise ValueError('Baseline best.json/complete.json mismatch')
                with (folder / 'epochs.csv').open(encoding='utf-8', newline='') as f:
                    epochs = list(csv.DictReader(f))
                acc = [float(r['test_acc']) for r in epochs]
                if len(epochs) != p['epochs'] or [int(r['epoch']) for r in epochs] != list(range(1, p['epochs'] + 1)):
                    raise ValueError('Incomplete baseline epoch history')
                if best['epoch'] != acc.index(max(acc)) + 1 or not np.isclose(best['acc'], max(acc)):
                    raise ValueError('Baseline first maximum test-ACC selection mismatch')
                point = _prediction(folder, 'best', best, historical, fold)
                points[(seed, fold, 'best')] = point
                entry.update(best_ok=True, best_prediction_sha256=point['sha256'],
                             best_epoch=best['epoch'], epochs=p['epochs'])
                record['mean_last100_acc'] = float(np.mean(acc[-100:]))
                records[(seed, fold)] = record
                try:
                    if record['last']['epoch'] != p['epochs'] or not np.isclose(record['last']['acc'], acc[-1]):
                        raise ValueError('Baseline final epoch mismatch')
                    point = _prediction(folder, 'last', record['last'], historical, fold)
                    points[(seed, fold, 'last')] = point
                    entry.update(last_ok=True, last_prediction_sha256=point['sha256'])
                except (OSError, ValueError, KeyError) as e:
                    entry['last_error'] = str(e)
            except (OSError, ValueError, KeyError) as e:
                entry['error'] = str(e)
            entries.append(entry)
    return records, points, dict(
        scope='All planned historical C seed/fold artifacts, prediction replay and first-max selection',
        protocols=protocols, folds=entries,
        best_ok=bool(entries) and all(e['best_ok'] for e in entries),
        last_ok=bool(entries) and all(e['last_ok'] for e in entries),
        epoch_budget_matches=bool(protocols) and all(p['epochs'] == plan['epochs'] for p in protocols.values()),
        note='Artifact audit only; no checkpoint inference rerun; C uses graph/OVR/orthogonality losses unlike A/B/F')


def _variant(records, seeds, folds):
    by_seed = {}
    for seed in seeds:
        rows = [r for (s, _), r in sorted(records.items()) if s == seed]
        item = dict(completed_folds=[r['fold'] for r in rows], planned_folds=folds)
        if rows:
            item.update(parameters=sorted({r['parameters'] for r in rows}),
                        mean_last100_acc=float(np.mean([r['mean_last100_acc'] for r in rows])),
                        best_epochs=[r['best']['epoch'] for r in rows],
                        counts=[r.get('counts') for r in rows])
            for stage in ('best', 'last'):
                item[stage] = {key: np.mean([r[stage][key] for r in rows], axis=0).tolist()
                               for key in ('acc', 'auc', 'f1', 'macro_f1', 'per_class_recall')
                               if all(key in r[stage] for r in rows)}
        by_seed[str(seed)] = item
    active = [v for v in by_seed.values() if v['completed_folds']]
    result = dict(completed_runs=len(records), planned_runs=len(seeds) * len(folds), per_seed=by_seed)
    if active:
        result['mean_last100_acc'] = float(np.mean([v['mean_last100_acc'] for v in active]))
        for stage in ('best', 'last'):
            result[stage] = {k: np.mean([v[stage][k] for v in active], axis=0).tolist()
                             for k in active[0][stage] if all(k in v[stage] for v in active)}
    return result


def summarize(out: Path, plan: dict) -> dict:
    out = Path(out)
    seeds, folds, modes = list(plan['seeds']), list(plan['folds']), list(plan['modes'])
    if not seeds or not folds or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError('Plan requires unique, nonempty seeds and folds')
    with np.load(ROOT / 'reference_fixed10/data_and_splits.npz', allow_pickle=False) as a:
        historical = {k: a[k].copy() for k in ('source_row', 'labels', 'test_masks')}
    records, points = {m: {} for m in modes}, {}
    for mode in modes:
        for seed in seeds:
            for fold in folds:
                folder = out / f'seed_{seed}' / mode / f'fold_{fold:02d}'
                if not (folder / 'complete.json').exists():
                    continue
                r = _read(folder / 'complete.json')
                if (r['seed'], r['fold'], r['mode']) != (seed, fold, mode):
                    raise ValueError(f'Run identity mismatch: {folder}')
                records[mode][(seed, fold)] = r
                for stage in ('best', 'last'):
                    points[(mode, seed, fold, stage)] = _prediction(folder, stage, r[stage], historical, fold)
    baseline_records, baseline_points, baseline_audit = _baseline(plan, historical)
    records['content'] = baseline_records
    for (seed, fold, stage), p in baseline_points.items():
        points[('content', seed, fold, stage)] = p
    variants = {mode: _variant(records[mode], seeds, folds) for mode in modes}
    comparisons = []
    for left, right in PAIRS:
        if left not in modes or (right != 'content' and right not in modes):
            continue
        common = sorted(set(records[left]) & set(records[right]))
        comparison = dict(comparison=f'{left} vs {right}', left=left, right=right,
                          common_seed_folds=[list(k) for k in common])
        for stage in ('best', 'last'):
            if right == 'content' and not (baseline_audit[f'{stage}_ok'] and baseline_audit['epoch_budget_matches']):
                comparison[stage] = dict(status='not_reported', reason='Historical baseline audit incomplete or different epoch budget')
                continue
            rows = []
            for seed, fold in common:
                a, b = points[(left, seed, fold, stage)], points[(right, seed, fold, stage)]
                if not np.array_equal(a['labels'], b['labels']) or not np.array_equal(a['row_index'], b['row_index']):
                    raise ValueError('Unaligned paired predictions')
                rows.append(dict(seed=seed, fold=fold, left_acc=a['acc'], right_acc=b['acc'],
                                 delta_acc=a['acc'] - b['acc'],
                                 corrected=int((a['correct'] & ~b['correct']).sum()),
                                 harmed=int((~a['correct'] & b['correct']).sum()),
                                 left_per_class_recall=a['per_class_recall'],
                                 right_per_class_recall=b['per_class_recall'],
                                 unique_patient_rows=a['row_index'].tolist()))
            comparison[stage] = dict(status='complete' if len(rows) == len(seeds) * len(folds) else 'partial',
                                     runs=len(rows), folds=rows,
                                     mean_delta_acc=_hierarchical(rows, 'delta_acc'),
                                     left_mean_acc=_hierarchical(rows, 'left_acc'),
                                     right_mean_acc=_hierarchical(rows, 'right_acc'),
                                     left_per_class_recall=_hierarchical(rows, 'left_per_class_recall'),
                                     right_per_class_recall=_hierarchical(rows, 'right_per_class_recall'),
                                     corrected_evaluations=sum(r['corrected'] for r in rows),
                                     harmed_evaluations=sum(r['harmed'] for r in rows),
                                     distinct_patients=len({i for r in rows for i in r['unique_patient_rows']}),
                                     averaging='Equal completed folds within each seed, then equal completed seeds')
        comparisons.append(comparison)
    result = dict(updated_at=timestamp(), seeds=seeds, folds=folds, modes=modes, epochs=plan['epochs'],
                  variants=variants, comparisons=comparisons, baseline_audit=baseline_audit,
                  class_names=['AD', 'CN', 'SMCI'],
                  caveat='First maximum TEST ACC; exploratory only. C is a historical different-model/loss reference. Repeated seeds are not independent patients.')
    dump(out / 'summary.json', result)
    lines = ['# 模态证据实验：进度与结果', '',
             f"TADPOLE AD/CN/SMCI；训练种子 {seeds}，计划折 {folds}，每次 {plan['epochs']} 轮。",
             '沿用首次最高测试 ACC 选轮；这些是探索性结果，不是独立验证集选轮的测试估计。',
             'A/B/F 均仅使用带原类别权重、标签平滑 0.05 的最终分类损失。C 保留图传播、类别辅助任务与正交约束；与 C 的比较是模型方案比较，不是仅改变一个因素的消融。',
             '统计先对每个种子的共同完成折取均值，再对种子取均值。未完成的折不计入；不同版本完成范围不同时，仅配对比较可直接对应。', '']
    for mode, v in variants.items():
        line = f"- {NAMES[mode]}：完成 {v['completed_runs']}/{v['planned_runs']} 个种子/折"
        if v['completed_runs']:
            line += f"；选轮 ACC {100*v['best']['acc']:.2f}%，末轮 {100*v['last']['acc']:.2f}%，末 100 轮 {100*v['mean_last100_acc']:.2f}%"
        lines.append(line + '。')
        for seed, s in v['per_seed'].items():
            if s['completed_folds']:
                lines.append(f"  seed {seed} 已完成折：{s['completed_folds']}；选中轮次：{s['best_epochs']}。")
    lines.extend(['', '配对结果（纠正/新增错误均为重复评估次数，不能当作不同患者人数）：', ''])
    for pair in comparisons:
        for stage, label in [('best', '选轮'), ('last', '末轮')]:
            v = pair[stage]
            name = f"{NAMES[pair['left']]} 相对 {NAMES[pair['right']]}"
            if v['status'] == 'not_reported':
                lines.append(f'- {name}，{label}：暂不报告，C 尚未完成该范围核查或训练轮数不同。')
            elif v['runs']:
                recalls = ' / '.join(f'{100*a:.2f}% 对 {100*b:.2f}%' for a, b in
                                     zip(v['left_per_class_recall'], v['right_per_class_recall']))
                lines.append(f"- {name}，{label}：共同完成 {v['runs']} 个种子/折，ACC 差 {100*v['mean_delta_acc']:+.2f} 个百分点；纠正 {v['corrected_evaluations']} 次，新增错误 {v['harmed_evaluations']} 次。AD/CN/SMCI 折均召回率依次为 {recalls}。")
    lines.extend(['', f"C 来源：{BASE}；已核查计划范围的源码/数据散列、患者顺序、保存预测指标及首次最高测试 ACC。",
                  f"C 选轮审计通过：{baseline_audit['best_ok']}；末轮审计通过：{baseline_audit['last_ok']}；训练轮数一致：{baseline_audit['epoch_budget_matches']}。",
                  '本次汇总核查保存的预测和日志，没有重新运行历史 C 检查点推理。完整逐折指标、类别召回率、配对范围及来源散列见 summary.json。', ''])
    (out / '进度与结果.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
