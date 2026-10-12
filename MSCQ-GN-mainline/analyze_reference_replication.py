"""Audit and export the completed selection-space replication, without refitting."""
import csv
import json
from pathlib import Path

import numpy as np

from relation_experiment import ROOT, dump, sha, timestamp
from reference_selection_experiment import read


def audit_existing_c_errors(out):
    """Supplementary descriptive comparison, not a cause-of-error diagnosis."""
    base = ROOT/'Result/reference_selection'
    protocol = read(base/'PROTOCOL.json')
    records, provenance, comparisons = [], {}, []
    for seed in (0, 1, 2):
        self_base = ROOT/'Result/relation_graph_audit'/f'self_seed_{seed}'
        self_protocol = read(self_base/'PROTOCOL.json')
        for field in ('original_config', 'data_sha256', 'checkpoint_selection', 'folds', 'epochs', 'split_seed'):
            assert self_protocol[field] == protocol[field], field
        for name in ('Model/relation_sheaf.py', 'Model/network.py', 'Model/models.py', 'Loss/loss_fn.py'):
            assert self_protocol['code_sha256'][name] == protocol['code_sha256'][name], name
        provenance[str(self_base/'PROTOCOL.json')] = sha(self_base/'PROTOCOL.json')
        for fold in range(1, 11):
            cp = base/f'seed_{seed}'/'c_real'/f'fold_{fold:02d}'/'predictions.npz'
            sp = self_base/'self_only'/f'fold_{fold:02d}'/'predictions.npz'
            with np.load(cp) as c, np.load(sp) as s:
                np.testing.assert_array_equal(c['labels'], s['labels'])
                np.testing.assert_array_equal(c['row_index'], s['row_index'])
                pc, ps, labels = c['logits'].argmax(-1), s['logits'].argmax(-1), c['labels']
                cc, sc = pc == labels, ps == labels
                comparisons.append(dict(seed=seed, fold=fold, c_acc=float(cc.mean()), self_acc=float(sc.mean())))
                records.extend(dict(seed=seed, fold=fold, source_row=int(row), true_class=int(y),
                    c_prediction=int(p), self_prediction=int(q), c_correct=bool(a), self_correct=bool(b))
                    for row,y,p,q,a,b in zip(c['row_index'],labels,pc,ps,cc,sc))
            provenance[str(cp)] = sha(cp)
            provenance[str(sp)] = sha(sp)
    def count(rows):
        return dict(evaluations=len(rows), c_correct=sum(r['c_correct'] for r in rows),
            self_correct=sum(r['self_correct'] for r in rows),
            both_wrong=sum(not r['c_correct'] and not r['self_correct'] for r in rows),
            c_wrong_self_right=sum(not r['c_correct'] and r['self_correct'] for r in rows),
            c_right_self_wrong=sum(r['c_correct'] and not r['self_correct'] for r in rows))
    by_patient = {}
    for r in records:
        by_patient.setdefault(r['source_row'], []).append(r)
    assert len(by_patient) == 598 and all(len(rows) == 3 for rows in by_patient.values())
    result = dict(scope='Supplementary existing saved outer-test predictions; both models independently selected by historical TEST ACC',
        limitation='Differences include retraining and checkpoint selection; self-right/C-wrong does not prove neighbor contamination',
        total=count(records), by_seed={str(seed): count([r for r in records if r['seed']==seed]) for seed in (0,1,2)},
        by_class={name: count([r for r in records if r['true_class']==i]) for i,name in enumerate(protocol['class_names'])},
        c_errors_in_all_three_seeds=sum(all(not r['c_correct'] for r in rows) for rows in by_patient.values()),
        both_models_wrong_in_all_three_seeds=sum(all(not r['c_correct'] and not r['self_correct'] for r in rows) for rows in by_patient.values()),
        c_fold_mean_acc=float(np.mean([r['c_acc'] for r in comparisons])),
        self_fold_mean_acc=float(np.mean([r['self_acc'] for r in comparisons])),
        provenance_sha256=provenance)
    dump(out/'已有C与无图错误核对.json', result)
    with (out/'已有C与无图逐患者核对.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return result


def main():
    out = ROOT/'Result/reference_selection_replication'
    plan = read(out/'PLAN.json')
    result = read(out/'analysis.json')
    assert read(out/'status.json')['state'] == 'complete'
    assert result['completed_cells'] == result['planned_cells'] == 4
    assert sha(ROOT/'reference_selection_replication.py') == plan['runner_sha256']
    assert sha(ROOT/'reference_selection_space.py') == plan['analysis_helper_sha256']
    protocol_path = ROOT/'Result/reference_selection/PROTOCOL.json'
    assert sha(protocol_path) == plan['protocol_sha256']
    protocol = read(protocol_path)
    for path, digest in protocol['code_sha256'].items():
        assert sha(ROOT/path) == digest, f'Historical source changed: {path}'
    rows, patients = [], []
    for cell in result['cells']:
        fold, seed = cell['outer_fold'], cell['training_seed']
        pooled = cell['one_shared_mask_across_both_teachers']
        n = pooled['patients']
        row = dict(fold=fold, seed=seed, held_patients=n,
            all_open_correct=pooled['all_open']['correct'],
            shared_acc_correct=pooled['global_acc_oracle']['correct'],
            individual_acc_ceiling=pooled['individual_acc_oracle']['correct'],
            extra_individual_correct=pooled['individual_acc_oracle']['additional_correct_over_global_acc'],
            extra_individual_acc_pp=100*(pooled['individual_acc_oracle']['acc']-pooled['global_acc_oracle']['acc']),
            per_teacher_extra_individual_correct=cell['per_teacher_shared']['extra_correct_over_global_acc_oracle'],
            unchanged_predictions=pooled['prediction_changes']['invariant_count'],
            all_open_errors=pooled['prediction_changes']['all_open_errors'],
            errors_uncorrectable_by_any_mask=pooled['prediction_changes']['uncorrectable_by_any_mask'],
            all_open_nll=pooled['all_open']['nll'],
            shared_best_nll=pooled['global_nll_oracle']['nll'],
            individual_best_nll=pooled['individual_nll_oracle']['nll'],
            individual_nll_gain_over_shared=pooled['global_nll_oracle']['nll']-pooled['individual_nll_oracle']['nll'],
            shared_acc_mask_id=pooled['global_acc_oracle']['mask_id'],
            shared_nll_mask_id=pooled['global_nll_oracle']['mask_id'])
        rows.append(row)
        for teacher in cell['teachers']:
            folder = out/f'fold_{fold:02d}'/f'seed_{seed}'/f"inner_{teacher['inner']}"
            assert sha(folder/'last.pt') == teacher['checkpoint_sha256']
            assert sha(folder/'predictions.npz') == teacher['predictions_sha256']
            with (folder/'epochs.csv').open(newline='', encoding='utf-8') as stream:
                epochs = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1, 1001))
            assert not any('test' in key or 'held' in key for key in epochs[0])
            with np.load(folder/'predictions.npz') as saved:
                pred = saved['logits'].argmax(-1)
                labels = saved['labels']
                any_correct = (pred == labels[None]).any(0)
                for j, source_row in enumerate(saved['source_row']):
                    patients.append(dict(fold=fold, seed=seed, inner=teacher['inner'], source_row=int(source_row),
                        true_class=int(labels[j]), all_open_prediction=int(pred[-1, j]),
                        shared_acc_prediction=int(pred[row['shared_acc_mask_id'], j]),
                        any_mask_correct=int(any_correct[j]),
                        distinct_predicted_classes=len(np.unique(pred[:, j]))))
    for filename, records in [('逐组结果.csv', rows), ('逐患者诊断.csv', patients)]:
        with (out/filename).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    total = sum(r['held_patients'] for r in rows)
    unchanged = sum(r['unchanged_predictions'] for r in rows)
    errors = sum(r['all_open_errors'] for r in rows)
    uncorrectable = sum(r['errors_uncorrectable_by_any_mask'] for r in rows)
    counts = dict(evaluation_records=total, distinct_source_rows=len({p['source_row'] for p in patients}),
        invariant_records=unchanged, invariant_fraction=unchanged/total,
        all_open_error_records=errors, uncorrectable_error_records=uncorrectable,
        count_warning='Repeated evaluations of overlapping patients; no independence or significance claim')
    dump(out/'report_audit.json', dict(passed=True, updated_at=timestamp(),
        report_source_sha256=sha(Path(__file__)), all_8_training_histories_exactly_1000_epochs=True,
        no_test_or_holdout_monitoring_columns=True, historical_source_hashes_unchanged=True, counts=counts))
    lines = ['# 参照选择空间复核', '',
        '**结论：新增四组均存在少量额外个体选择空间，但每组仅多判对 1、1、1、2 位内部留出患者，'
        '相当于 0.19、0.19、0.19、0.37 个百分点。** 因此应修正“完全没有额外空间”的表述；'
        '本轮也没有证明实际选择器能取得这些带答案的理想收益。', '',
        '处理建议是保留 C、已有普通门控及收益门控对照，暂停扩大当前“聚合后模态二值开关”的复杂度。'
        '下一步先对 C 与无图模型出现分歧的病例做固定模型的传播干预，判断差异是否确由传播造成；'
        '本轮已经完成两模型错误重合统计，但尚未实施这一新的针对性干预。', '',
        '本轮预先固定第 2、3 外层折及训练种子 0、1，每组使用两个互补内部划分，完成 8 个始终开启参照的教师。'
        '同一折在两个种子间保持内部划分不变。全部固定训练 1000 轮，使用末轮检查点；不根据留出集或外层测试集选轮。', '',
        '每个教师遍历六种内容模态的 64 个参照开关组合。保持模型参数、自身信息、候选患者图和边权不变，'
        '只控制已经聚合的跨患者内容。开关不等于选择具体邻居；关闭后也不是独立训练的自身模型。', '',
        '**最佳统一组合和逐人理想上限均使用内部留出标签，是带答案的诊断，不是新模型的预测成绩。**'
        '这些教师只用半个外层训练折，不能将其 ACC 与主模型十折 94.15% 直接比较。', '',
        '## 逐组 ACC 诊断', '',
        '以下“统一最佳”要求同一折、同一种子的两个内部教师共用一个组合；“逐人上限”表示只要某个组合能判对该患者，就算正确。', '']
    for r in rows:
        lines.append(f"- 第 {r['fold']} 折、种子 {r['seed']}：全开 {r['all_open_correct']}/{r['held_patients']}；"
            f"统一最佳 {r['shared_acc_correct']}/{r['held_patients']}；逐人上限 {r['individual_acc_ceiling']}/{r['held_patients']}，"
            f"比统一最佳多 {r['extra_individual_correct']} 位（{r['extra_individual_acc_pp']:.3f} 个百分点）。"
            f"若允许每位教师分别使用其最佳统一组合，额外空间为 {r['per_teacher_extra_individual_correct']} 位。")
    lines += ['', '## 类别变化与损失', '',
        f'四组共 {total} 次重复患者评估，涉及 {counts["distinct_source_rows"]} 个不同患者行，不能视为 {total} 位独立患者。'
        f'其中 {unchanged} 次（{100*unchanged/total:.2f}%）在全部 64 种组合下预测类别不变。'
        f'全开时的 {errors} 次错误中，{uncorrectable} 次无论怎样组合仍然错误。', '',
        '主要选择标准 NLL 与上述 ACC 理想上限分别计算。逐人最低 NLL 相对于同组统一最低 NLL 的改进如下：', '']
    for r in rows:
        lines.append(f"- 第 {r['fold']} 折、种子 {r['seed']}：{r['shared_best_nll']:.6f} → "
            f"{r['individual_best_nll']:.6f}，降低 {r['individual_nll_gain_over_shared']:.6f}。")
    lines += ['', '## 解释边界', '',
        '本轮只检验固定全开教师上的、聚合后内容模态二值开关。关闭参照相对于教师训练条件发生变化。'
        '结果不构成连续门控、逐邻居选择、其他传播位置或重新训练模型的普遍上限。', '',
        '同一批患者在不同种子和外层训练子集中重复出现，结果仅作复核和描述，不把它们当独立样本作显著性证明。'
        '两位教师分别拟合自身的内部训练分区，全部组合选择和诊断只使用其内部留出标签，外层测试标签未用于该诊断。', '',
        '原 C、原实验协议、此前全部结果未修改。12 项现有门控/选择空间检查通过；'
        '原始数据与代码散列、候选图、内部划分、保存检查点重载以及逐患者组合同时应用均已检查。', '',
        '详细记录见 `PLAN.json`、`analysis.json`、`audit.json`、`report_audit.json`、`逐组结果.csv` 和 `逐患者诊断.csv`。']
    existing = audit_existing_c_errors(out)
    e = existing['total']
    lines += ['', '## 补充：已有 C 与从头训练无图模型的错误重合', '',
        '这部分是训练等待期间对已有保存结果的补充核对，不属于新教师复核的预先指定指标。'
        '两种模型的划分、数据、原配置和模型共同部分一致，均采用原先首次最高测试 ACC 选轮；'
        '分别选出的检查点可能来自不同轮次，因此只能作探索性描述。', '',
        f"在三种子十折的 {e['evaluations']} 次重复评估中，C 错误 {e['evaluations']-e['c_correct']} 次："
        f"其中 {e['both_wrong']} 次无图模型也错，{e['c_wrong_self_right']} 次无图模型正确。"
        f"另一方面，C 有 {e['c_right_self_wrong']} 次预测正确而无图模型错误。", '',
        f"共有 {existing['c_errors_in_all_three_seeds']} 位患者在 C 的三个种子下均预测错误，"
        f"其中 {existing['both_models_wrong_in_all_three_seeds']} 位在两种模型、三个种子下都预测错误。", '',
        '这些结果可以用于区分“两个模型都难以处理的病例”和“模型之间出现分歧的病例”。'
        '但无图对而 C 错不等于已经证明邻居带来干扰，因为差异还包括重新训练和选轮。'
        '要定位到传播机制，需要进一步在固定模型上作控制干预，不能直接把这批病例当成有害邻居的监督标签。', '',
        '详细对应关系保存在 `已有C与无图错误核对.json` 与 `已有C与无图逐患者核对.csv`。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(cells=rows, counts=counts,
        existing_c_error_comparison={k:v for k,v in existing.items() if k!='provenance_sha256'}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
