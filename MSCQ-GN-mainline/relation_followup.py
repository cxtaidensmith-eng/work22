"""G2 mechanism control and independent training-seed repeats.

Preserves the original ten folds and 1000-epoch test-ACC selection policy.
Seed 0 G1/G2 are referenced in place; no historical outputs are overwritten.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from relation_experiment import ROOT, dump, sha, timestamp, train

NAMES = {'shared': 'G1 统一变换', 'content': 'C 仅内容条件（与 G2 等参数量）',
         'static': 'G2 关系与内容条件'}
MODES = tuple(NAMES)
LEGACY = ROOT/'Result/relation_sheaf_tadpole_original'


def result_root(out, seed, mode, smoke):
    if seed == 0 and mode in ('shared', 'static') and not smoke:
        return LEGACY
    return out/f'seed_{seed}'


def aggregate(out, seeds, folds, smoke):
    records, variants, comparisons = {}, {}, []
    for mode in MODES:
        complete_seeds = []
        seed_means = {}
        count = 0
        for seed in seeds:
            base = result_root(out, seed, mode, smoke)
            rows = {}
            for fold in folds:
                path = base/mode/f'fold_{fold:02d}'/'complete.json'
                if path.exists():
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if record['seed'] != seed or record['fold'] != fold or record['mode'] != mode:
                        raise RuntimeError(f'Mislabeled result: {path}')
                    rows[fold] = record
            records[mode, seed] = rows
            count += len(rows)
            if len(rows) == len(folds):
                complete_seeds.append(seed)
                seed_means[str(seed)] = float(np.mean([r['best']['acc'] for r in rows.values()]))
        variants[mode] = dict(completed_fold_runs=count, planned_fold_runs=len(seeds)*len(folds),
                              complete_seeds=complete_seeds, seed_mean_acc=seed_means)
        if seed_means:
            values = list(seed_means.values())
            variants[mode]['mean_acc_complete_seeds'] = float(np.mean(values))
            if len(values) > 1:
                variants[mode]['std_across_seed_means'] = float(np.std(values, ddof=1))
    for left, right in [('shared', 'static'), ('content', 'static'), ('shared', 'content')]:
        by_seed = []
        for seed in seeds:
            common = sorted(set(records[left, seed]) & set(records[right, seed]))
            if not common:
                continue
            corrected, harmed, deltas = 0, 0, []
            for fold in common:
                a_path = result_root(out, seed, left, smoke)/left/f'fold_{fold:02d}'/'predictions.npz'
                b_path = result_root(out, seed, right, smoke)/right/f'fold_{fold:02d}'/'predictions.npz'
                with np.load(a_path) as a, np.load(b_path) as b:
                    if not (np.array_equal(a['labels'], b['labels']) and np.array_equal(a['row_index'], b['row_index'])):
                        raise RuntimeError('Paired predictions do not refer to the same patients')
                    ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                    corrected += int((~ca & cb).sum())
                    harmed += int((ca & ~cb).sum())
                    deltas.append(float(cb.mean() - ca.mean()))
            by_seed.append(dict(seed=seed, matched_folds=common, mean_acc_delta=float(np.mean(deltas)),
                                wins=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas),
                                losses=sum(d < 0 for d in deltas), corrected=corrected, harmed=harmed))
        complete_pairs = [r for r in by_seed if len(r['matched_folds']) == len(folds)]
        item = dict(comparison=f'{right} vs {left}', by_seed=by_seed,
                    complete_paired_seeds=[r['seed'] for r in complete_pairs])
        if complete_pairs:
            item['mean_delta_complete_paired_seeds'] = float(np.mean([r['mean_acc_delta'] for r in complete_pairs]))
        comparisons.append(item)
    result = dict(updated_at=timestamp(), smoke=smoke, variants=variants, matched_comparisons=comparisons,
                  aggregation='Average fold ACC within each seed; then average seeds. No prediction ensemble.',
                  caveat='Historical test-selected protocol. Only compare matched folds and training seeds.')
    dump(out/'summary.json', result)
    lines = ['# G2 后续实验进度', '',
             '问题一：变换需要关系来源，还是只区分内容模态就足够？',
             '问题二：G2 相对 G1 的收益能否在不同训练随机种子下重复？', '',
             f"训练种子：{seeds}。原数据顺序与十折划分固定；逐 seed 独立评估，不做预测集成。",
             '按原最高测试 ACC 选轮。只比较相同种子、相同已完成折。', '']
    if smoke:
        lines.extend(['**这是短试运行，不是正式实验结果。**', ''])
    else:
        lines.extend(['每次训练 1000 轮。种子 0 的 G1/G2 引用第一轮结果，其余为新训练。', ''])
    for mode, info in variants.items():
        lines.append(f"- {NAMES[mode]}：完成 {info['completed_fold_runs']}/{info['planned_fold_runs']} 个折次。")
        if info['seed_mean_acc']:
            means = '；'.join(f"seed {seed}：{100*value:.2f}%" for seed, value in info['seed_mean_acc'].items())
            lines.append(f'  已完成整组折的 ACC：{means}。')
    lines.extend(['', '配对比较（正值表示前者更好）：', ''])
    for pair in comparisons:
        for item in pair['by_seed']:
            lines.append(f"- {pair['comparison']}，seed {item['seed']}，{len(item['matched_folds'])} 折："
                         f"{100*item['mean_acc_delta']:+.2f} 个百分点；纠正 {item['corrected']} 人，新增错误 {item['harmed']} 人。")
    lines.extend(['', '完整判断等待所有计划训练完成，不选取最好种子作为总成绩。',
                  '同一患者在不同种子下重复评估，各种子的纠正人数不能相加当成独立患者数。'])
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(args):
    out = (ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume')
    seeds, folds = ([0, 1], [1]) if args.smoke else ([0, 1, 2], list(range(1, 11)))
    epochs = 3 if args.smoke else None
    legacy_protocol = LEGACY/'PROTOCOL.json'
    if not args.smoke:
        if not (LEGACY/'COMPLETE.json').exists():
            raise RuntimeError('First-round experiment must finish before reusing its results')
        verification = ROOT/'Result/relation_sheaf_g2_checks/extension_verification.json'
        if not verification.exists():
            raise RuntimeError('Run the legacy compatibility verification before the formal follow-up')
        checked = json.loads(verification.read_text(encoding='utf-8'))
        if not checked.get('passed') or checked['model_sha256'] != sha(ROOT/'Model/relation_sheaf.py'):
            raise RuntimeError('The model has changed since legacy compatibility verification')
    jobs = [dict(training_seed=seed, modes=['content'] if seed == 0 and not args.smoke else list(MODES))
            for seed in seeds]
    plan = dict(experiment='G2 relation-source necessity and training-seed repeats',
                smoke=args.smoke, seeds=seeds, folds=folds, epochs=epochs or 1000,
                new_jobs=jobs, reused=[] if args.smoke else [dict(training_seed=0, modes=['shared', 'static'],
                                                               source=str(LEGACY))],
                new_fold_runs=sum(len(job['modes'])*len(folds) for job in jobs),
                control='content_id[M,16] -> MLP[16,32,4]; same 895096 active parameters as static',
                original_protocol_sha256=None if args.smoke else sha(legacy_protocol),
                code_sha256={name: sha(ROOT/name) for name in ['relation_followup.py', 'relation_experiment.py',
                                                               'Model/relation_sheaf.py']})
    if (out/'PLAN.json').exists() and json.loads((out/'PLAN.json').read_text(encoding='utf-8')) != plan:
        raise RuntimeError('Plan/code changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'PLAN.json', plan)
    aggregate(out, seeds, folds, args.smoke)
    try:
        for job in jobs:
            seed = job['training_seed']
            dump(out/'status.json', dict(state='running', training_seed=seed, modes=job['modes'],
                                         updated_at=timestamp(), detail_file=str(out/f'seed_{seed}'/'status.json')))
            child = SimpleNamespace(config='Config/main_tadpole_3class.ini', output=str(out/f'seed_{seed}'),
                                    cuda=args.cuda, k=10, modes=job['modes'], folds=folds,
                                    epochs=epochs, training_seed=seed, resume=args.resume,
                                    reference_protocol=None if args.smoke else str(legacy_protocol))
            train(child, progress_callback=lambda: aggregate(out, seeds, folds, args.smoke))
        result = aggregate(out, seeds, folds, args.smoke)
        dump(out/'COMPLETE.json', result)
        dump(out/'status.json', dict(state='complete', updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='Result/relation_sheaf_g2_followup')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two seeds, one fold, three epochs; no reuse of historical results')
    main(parser.parse_args())
