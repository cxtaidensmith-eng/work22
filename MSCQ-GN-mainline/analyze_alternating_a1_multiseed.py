"""Seed-matched C/A1 reports. No training and no writes to historical results."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_joint_evidence_routing import _prediction, _aggregate, _pair, _dump

ROOT = Path(__file__).resolve().parent
OLD_A1 = ROOT/'Result/alternating_cooperative_seed0_20260923/seed_0/A1'
C_ROOT = ROOT/'Result/relation_mechanism_nodecay'
FOLDS = tuple(range(1,11))
SEEDS = (0,1,2)


def read_record(folder, seed, fold, mode):
    record = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    assert (record['seed'],record['fold'],record['mode']) == (seed,fold,mode)
    assert record['parameters'] == 895096
    predictions = {s:_prediction(folder,s,record) for s in ('best','last')}
    np.testing.assert_array_equal(predictions['best']['rows'],predictions['last']['rows'])
    np.testing.assert_array_equal(predictions['best']['labels'],predictions['last']['labels'])
    with (folder/'epochs.csv').open(encoding='utf-8') as stream:
        curve = list(csv.DictReader(stream))
    assert [int(r['epoch']) for r in curve] == list(range(1,1001))
    acc = np.array([float(r['test_acc']) for r in curve])
    assert np.isfinite(acc).all()
    assert record['best']['epoch'] == int(acc.argmax())+1
    assert record['best']['acc'] == acc.max()
    assert record['last']['epoch'] == 1000 and record['last']['acc'] == acc[-1]
    last100 = float(acc[-100:].mean())
    if 'mean_last100_acc' in record:
        assert abs(record['mean_last100_acc']-last100) < 1e-12
    return dict(record=record,predictions=predictions,mean_last100_acc=last100,
                parameters=record['parameters'],folder=str(folder))


def seed_aggregate(values):
    """Only full three-seed results may enter the overall comparison."""
    assert len(values) == 3 and all(v['completed_folds'] == 10 for v in values)
    result = dict(mean_last100_acc=float(np.mean([v['mean_last100_acc'] for v in values])),
                  std_definition='Population std (ddof=0) of the three ten-fold means')
    for stage in ('best','last'):
        result[stage] = {}
        for key in ('acc','auc','f1','macro_f1'):
            vector = [v[stage][key] for v in values]
            result[stage][key] = dict(mean=float(np.mean(vector)),std=float(np.std(vector)))
        result[stage]['per_class_recall'] = np.mean([v[stage]['per_class_recall'] for v in values],axis=0).tolist()
    return result


def summarize(out, plan):
    out = Path(out)
    assert plan['training_seeds'] == [1,2] and plan['comparison_seeds'] == [0,1,2]
    per_seed, completed, fold_rows = {}, 0, []
    for seed in SEEDS:
        folders = dict(C=C_ROOT/f'seed_{seed}/content', A1=OLD_A1 if seed==0 else out/f'seed_{seed}/A1')
        records = {m:{f:read_record(base/f'fold_{f:02d}',seed,f,'content' if m=='C' else 'A1')
                      for f in FOLDS if (base/f'fold_{f:02d}/complete.json').exists()}
                   for m,base in folders.items()}
        if seed != 0:
            completed += len(records['A1'])
        variants = {m:_aggregate(v,FOLDS) for m,v in records.items()}
        comparison = _pair('C','A1',records['C'],records['A1'],plan['class_names'])
        per_seed[str(seed)] = dict(variants=variants,paired=comparison)
        for fold in comparison['matched_folds']:
            for stage in ('best','last'):
                c,a = records['C'][fold],records['A1'][fold]
                left,right = c['predictions'][stage]['correct'],a['predictions'][stage]['correct']
                fold_rows.append(dict(seed=seed,fold=fold,stage=stage,
                    c_acc=c['record'][stage]['acc'],a1_acc=a['record'][stage]['acc'],
                    a1_minus_c=a['record'][stage]['acc']-c['record'][stage]['acc'],
                    corrected=int((~left&right).sum()),harmed=int((left&~right).sum()),
                    net=int(right.sum())-int(left.sum())))
    all_complete = completed == 20
    result = dict(completed_new_runs=completed,planned_new_runs=20,reused_a1_seed0_runs=10,
                  all_new_complete=all_complete,per_seed=per_seed,
                  aggregation='Mean ten folds per seed; then equal mean over seeds 0/1/2. No best-seed selection or ensembling.',
                  patient_count_caveat='Same 598 rows reused across seeds; pooled changes across seeds are row-seed cases, not independent patients.')
    if all_complete:
        result['overall'] = {m:seed_aggregate([per_seed[str(s)]['variants'][m] for s in SEEDS]) for m in ('C','A1')}
        result['paired_overall'] = {}
        for stage in ('best','last'):
            pairs = [per_seed[str(s)]['paired']['stages'][stage] for s in SEEDS]
            deltas = [p['mean_acc_delta'] for p in pairs]
            result['paired_overall'][stage] = dict(mean_acc_delta=float(np.mean(deltas)),
                std_seed_delta=float(np.std(deltas)),seed_deltas=deltas,
                seed_wins=sum(v>1e-12 for v in deltas),seed_ties=sum(abs(v)<=1e-12 for v in deltas),
                seed_losses=sum(v< -1e-12 for v in deltas),
                corrected_row_seed_cases=sum(v['corrected'] for v in pairs),
                harmed_row_seed_cases=sum(v['harmed'] for v in pairs),
                net_row_seed_cases=sum(v['net'] for v in pairs),
                per_class={name:dict(corrected=sum(v['per_class'][name]['corrected'] for v in pairs),
                                    harmed=sum(v['per_class'][name]['harmed'] for v in pairs),
                                    net=sum(v['per_class'][name]['net'] for v in pairs))
                           for name in plan['class_names']})
    _dump(out/'summary.json',result)
    lines = ['# A1 seed 1、2：进度与三种子比较','',
             f'新增训练完成 {completed}/20 次；复用 A1 seed 0 的10折及 C 三个种子的30折。',
             '数据划分固定为原 split seed 0；每次1000轮。仅改变训练种子，模型结构和所有超参数保持冻结。','',
             '| 训练种子 | A1完成折数 | C最佳ACC | A1最佳ACC | 配对差值 |',
             '|---|---:|---:|---:|---:|']
    for seed,item in per_seed.items():
        c,a = item['variants']['C'],item['variants']['A1']
        if a['completed_folds'] == 10:
            lines.append(f"| {seed} | 10/10 | {100*c['best']['acc']:.4f}% | {100*a['best']['acc']:.4f}% | "
                         f"{100*item['paired']['stages']['best']['mean_acc_delta']:+.4f}个百分点 |")
        else:
            lines.append(f"| {seed} | {a['completed_folds']}/10 | {100*c['best']['acc']:.4f}% | 待完成 | 待完成 |")
    lines += ['', '仅完整十折写入上表；部分折数据保存在summary.json，配对计算只使用共同完成折。', '']
    for seed,item in per_seed.items():
        pair = item['paired']
        if not pair['matched_folds']:
            continue
        for stage,label in (('best','最佳'),('last','第1000轮')):
            v=pair['stages'][stage]
            lines.append(f"- seed {seed}，{len(pair['matched_folds'])}折，{label}：A1−C {100*v['mean_acc_delta']:+.4f}个百分点，"
                         f"纠正{v['corrected']}，新增错误{v['harmed']}，净{v['net']:+d}；"
                         f"逐折{v['fold_wins']}胜/{v['fold_ties']}平/{v['fold_losses']}负。")
    if all_complete:
        lines += ['', '| 模型 | 三种子最佳ACC均值±标准差 | 宏F1均值 | 第1000轮ACC均值 | 末100轮ACC均值 |',
                  '|---|---:|---:|---:|---:|']
        for mode,v in result['overall'].items():
            lines.append(f"| {mode} | {100*v['best']['acc']['mean']:.4f}% ± {100*v['best']['acc']['std']:.4f}个百分点 | "
                         f"{100*v['best']['macro_f1']['mean']:.4f}% | {100*v['last']['acc']['mean']:.4f}% | "
                         f"{100*v['mean_last100_acc']:.4f}% |")
        lines += ['', '标准差统计对象为三个种子的十折均值，ddof=0；各种子使用同一组598条样本。',
                  '最佳ACC沿用测试集选轮规则；增加训练种子不构成新增独立测试，也不单独证明反馈机制。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'paired_folds.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','fold','stage','c_acc','a1_acc','a1_minus_c','corrected','harmed','net'])
        writer.writeheader();writer.writerows(fold_rows)
    return result


def write_report(out,plan,audit):
    out=Path(out)
    result=summarize(out,plan)
    assert result['all_new_complete'] and audit['passed'] and audit['verified_checkpoints']==40
    primary=result['paired_overall']['best']
    if primary['mean_acc_delta']>1e-12 and primary['seed_losses']==0:
        conclusion='A1在三个已有/新增种子的平均最佳ACC上高于C，且没有种子低于C；这是原协议下的重复种子证据，仍非独立测试验证。'
    elif primary['mean_acc_delta']>1e-12:
        conclusion='A1的三种子平均最佳ACC高于C，但优势随种子变化，应结合逐种子幅度和配对纠错判断。'
    elif abs(primary['mean_acc_delta'])<=1e-12:
        conclusion='A1与C的三种子平均最佳ACC持平，尚未形成额外分类收益。'
    else:
        conclusion='A1的三种子平均最佳ACC低于C，本次补充实验没有建立完整A1优于C的分类优势。'
    lines=['# A1三种子结果分析','',conclusion,'',
           '新增20次训练完成，40个新增best/last检查点已回放。A1 seed0和C历史结果保持原样。','',
           (out/'进度与结果.md').read_text(encoding='utf-8'),'','类别变化（A1−C，最佳检查点）：','']
    for seed,item in result['per_seed'].items():
        per_class=item['paired']['stages']['best']['per_class']
        lines.append(f"- seed {seed}："+'；'.join(f"{name}净{v['net']:+d}条样本，折均召回{100*v['mean_recall_delta']:+.3f}个百分点" for name,v in per_class.items())+'。')
    lines += ['', '本轮仅补充A1训练种子1、2，没有补跑N0/N1/A0；因此不能将多种子A1对C的差值独立归因于交替或边权重估。',
              '跨种子的纠错计数包含同一患者的重复预测，不当作1794名独立患者。未进行种子挑选或预测集成。',
              '主指标继续采用原测试ACC选轮规则；末轮、末100轮、宏F1和类别召回不用于改变选点。','']
    (out/'结果分析.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
