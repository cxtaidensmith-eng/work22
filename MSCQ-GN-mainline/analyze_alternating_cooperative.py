"""Fixed comparisons for the four-arm cooperative feedback experiment."""
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_joint_evidence_routing import _record, _aggregate, _pair, _dump, REFERENCE

MODES = ('N0', 'N1', 'A0', 'A1')
PAIRS = (('C', 'A1'), ('N1', 'A1'), ('A0', 'A1'), ('N0', 'N1'),
         ('N0', 'A0'), ('C', 'N0'), ('C', 'N1'), ('C', 'A0'))
NAMES = dict(C='C：原单次传播', N0='N0：连续传播／复用边权', N1='N1：连续传播／重算边权',
             A0='A0：交替传播／复用边权', A1='A1：交替传播／重算边权（主候选）')


def interaction(records):
    common = sorted(set.intersection(*(set(records[m]) for m in MODES)))
    result = dict(definition='(A1-A0)-(N1-N0)', matched_folds=common, stages={})
    for stage in ('best', 'last'):
        values = [dict(fold=f, delta=(records['A1'][f]['record'][stage]['acc']
                    - records['A0'][f]['record'][stage]['acc'])
                    - (records['N1'][f]['record'][stage]['acc']
                       - records['N0'][f]['record'][stage]['acc'])) for f in common]
        result['stages'][stage] = dict(folds=values,
            mean_acc_delta=float(np.mean([v['delta'] for v in values])) if values else None)
    return result


def table(summary):
    lines = ['| 模型 | 完成折数 | 最佳 ACC | 宏 F1 | 末轮 ACC | 末100轮 ACC | 参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, v in summary['variants'].items():
        if not v['completed_folds']:
            lines.append(f'| {NAMES[mode]} | 0/10 | — | — | — | — | — |')
        else:
            lines.append(f"| {NAMES[mode]} | {v['completed_folds']}/10 | {100*v['best']['acc']:.3f}% | "
                f"{100*v['best']['macro_f1']:.3f}% | {100*v['last']['acc']:.3f}% | "
                f"{100*v['mean_last100_acc']:.3f}% | {v['parameters']} |")
    return lines


def summarize(out, plan):
    out = Path(out)
    assert plan['modes'] == list(MODES) and plan['primary_candidate'] == 'A1'
    folds = plan['folds']
    directories = dict(C=REFERENCE, **{m: out/'seed_0'/m for m in MODES})
    records = {m: {f: _record(base/f'fold_{f:02d}', f, plan['epochs']) for f in folds
                   if (base/f'fold_{f:02d}/complete.json').exists()} for m, base in directories.items()}
    comparisons = [_pair(a, b, records[a], records[b], plan['class_names']) for a, b in PAIRS]
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), training_seed=0,
        primary_candidate='A1', planned_new_runs=40,
        completed_new_runs=sum(len(records[m]) for m in MODES),
        all_new_complete=all(len(records[m]) == 10 for m in MODES),
        variants={m: _aggregate(r, folds) for m, r in records.items()},
        matched_comparisons=comparisons, interaction=interaction(records),
        aggregation='Equal-fold means; patient corrected/harmed counts pooled descriptively',
        selection=plan['checkpoint_selection'],
        caveats=['A1 was predeclared; no best-of-four model selection',
                 'Single seed and historical test-selected checkpoints; exploratory evidence',
                 'C has one P; all four new arms have two shared P calls and five F calls',
                 'Fixed candidate supports; state-dependent edge weights, not topology reselection'])
    _dump(out/'summary.json', result)
    lines = ['# 交替反馈实验：进度与结果', '',
        f"已完成 {result['completed_new_runs']}/40 次训练；种子0，原十折，每折1000轮。主候选固定为A1。", '',
        '各指标按折等权平均；未完成时，各组覆盖折数可能不同，不据此排名。配对比较仅采用共同完成折。', '',
        *table(result), '', '以下差值均为候选减参照：', '']
    fold_rows, patient_rows = [], []
    for pair in comparisons:
        if not pair['matched_folds']:
            continue
        a, b = pair['reference'], pair['candidate']
        for stage, label in (('best', '最佳'), ('last', '末轮')):
            v = pair['stages'][stage]
            lines.append(f"- {b}−{a}，{label}，{len(pair['matched_folds'])}折：ACC {100*v['mean_acc_delta']:+.4f}个百分点；"
                         f"纠正{v['corrected']}，新增错误{v['harmed']}，净{v['net']:+d}人；"
                         f"{v['fold_wins']}胜/{v['fold_ties']}平/{v['fold_losses']}负。")
            for f in pair['matched_folds']:
                left, right = records[a][f], records[b][f]
                delta = right['record'][stage]['acc']-left['record'][stage]['acc']
                fold_rows.append(dict(reference=a, candidate=b, stage=stage, fold=f, acc_delta=delta))
                p, q = left['predictions'][stage], right['predictions'][stage]
                for i in np.flatnonzero(p['predicted'] != q['predicted']):
                    patient_rows.append(dict(reference=a, candidate=b, stage=stage, fold=f,
                        row_index=int(p['rows'][i]), label=plan['class_names'][int(p['labels'][i])],
                        reference_prediction=plan['class_names'][int(p['predicted'][i])],
                        candidate_prediction=plan['class_names'][int(q['predicted'][i])],
                        corrected=int(not p['correct'][i] and q['correct'][i]),
                        harmed=int(p['correct'][i] and not q['correct'][i])))
    for stage, label in (('best', '最佳'), ('last', '末轮')):
        delta = result['interaction']['stages'][stage]['mean_acc_delta']
        if delta is not None:
            lines += ['', f"{label}交互差值 (A1−A0)−(N1−N0)：{100*delta:+.4f}个百分点。"]
    lines += ['', '主指标保持原协议的最佳ACC。宏F1、类别召回、末轮及末100轮只作补充。'
              '边权和图更新幅度属于机制描述，不能单独证明分类收益或因果贡献。', '']
    (out/'进度与结果.md').write_text('\n'.join(lines), encoding='utf-8')
    for filename, rows, fields in (
        ('fold_differences.csv', fold_rows, ['reference','candidate','stage','fold','acc_delta']),
        ('patient_prediction_changes.csv', patient_rows, ['reference','candidate','stage','fold','row_index',
            'label','reference_prediction','candidate_prediction','corrected','harmed'])):
        with (out/filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    return result


def write_report(out, plan, audit):
    out = Path(out)
    result = summarize(out, plan)
    assert result['all_new_complete'] and audit['passed'] and audit['verified_checkpoints'] == 80
    assert all(v['parameters'] == 895096 for v in result['variants'].values())
    pairs = {(p['reference'], p['candidate']): p for p in result['matched_comparisons']}
    primary = pairs['C','A1']['stages']['best']
    if primary['mean_acc_delta'] <= 1e-12 or primary['net'] <= 0:
        conclusion = 'A1未建立同时具有最佳ACC提升和患者净纠错收益的分类优势，本轮不能宣布主方法成功。'
    elif primary['net'] <= 2:
        conclusion = 'A1的分类收益仅为净多判对1–2人，应记为弱线索，不能称为性能突破。'
    else:
        conclusion = 'A1在本轮原协议下形成候选分类收益；仍需结合逐折及机制对照判断，不据此声称稳定优势或统计显著性。'
    progress = (out/'进度与结果.md').read_text(encoding='utf-8')
    lines = ['# 交替反馈实验：完整分析', '', conclusion, '',
        '40次训练完成，80个新增best/last检查点均经回放核验。旧C回放10个best检查点，'
        '末轮仅核查已有保存预测；没有旧C的last模型回放。', '', progress, '',
        '类别变化（折均召回与汇总净人数分别报告）：', '']
    for pair in result['matched_comparisons']:
        for stage, label in (('best','最佳'), ('last','末轮')):
            parts = []
            for name, v in pair['stages'][stage]['per_class'].items():
                parts.append(f"{name}净{v['net']:+d}人，召回{100*v['mean_recall_delta']:+.3f}个百分点")
            lines.append(f"- {pair['comparison']}，{label}："+'；'.join(parts)+'。')
    smci = plan['class_names'].index('SMCI')
    lines += ['', 'SMCI错误去向（汇总患者数）：', '']
    for m, v in result['variants'].items():
        for stage, label in (('best','最佳'), ('last','末轮')):
            matrix = v[stage]['confusion_matrix_sum']
            destinations = [f"{name}：{matrix[smci][i]}" for i, name in enumerate(plan['class_names']) if i != smci]
            lines.append(f"- {m}，{label}："+'；'.join(destinations)+'。')
    lines += ['', '机制判断必须分别看A1−N1和A1−A0。交互差值描述本设置中重算的收益差异，'
              '不等于一般性因果证明。全部组保留五层模态交互和相同有效参数；C与新组的传播调用次数不同。', '',
              '本轮无中途调参、自动追加种子或新损失。若未取得有效收益，停止该版本；'
              '不会从四组中事后挑出最高者替代预定A1的主结论。', '',
              '逐折差值见fold_differences.csv，患者预测变化见patient_prediction_changes.csv，'
              '详细指标见summary.json，诊断及复核见audit.json，源码与协议见PLAN.json和source_snapshot。', '']
    (out/'结果分析.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
